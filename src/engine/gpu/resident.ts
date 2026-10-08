import type { ForceResult, SimState } from '../types';
import { massOf } from '../atoms';

/*
 * GPU-resident velocity Verlet (docs/design/notebook.md, "Backends", v2).
 *
 * The force-only backend reads forces back every step, so each step pays a
 * GPU round trip. Here whole steps stay on the GPU: per step,
 *   kickDrift  v += dt/2 F/m; x += dt v; wrap into the box, update images
 *   clear/count/scan/scatter   cell list rebuilt on the GPU (atomic counts,
 *              one-workgroup prefix sum, scatter into cell-sorted slots)
 *   force      the cell-list LJ kernel over sorted slots (as in
 *              webgpuForces.ts), energy and virial halved per pair
 *   kick2      v += dt/2 F/m (+ enforce2d)
 * and positions, velocities, forces, images, energy and virial are read back
 * only when the host needs them (thermo, frames, dumps, end of run).
 * Supports fix nve (+ enforce2d); thermostatted runs use the per-step path.
 *
 * Fits WebGPU compatibility mode: at most 4 storage buffers per entry point
 * and at most 128 invocations per workgroup (the scan uses one workgroup of
 * 64). fp32 throughout; positions are relative to the box origin.
 */

const BUF_MAP_READ = 0x0001;
const BUF_COPY_SRC = 0x0004;
const BUF_COPY_DST = 0x0008;
const BUF_UNIFORM = 0x0040;
const BUF_STORAGE = 0x0080;
const MAP_READ = 0x0001;
const WG = 64;

const SHADER = /* wgsl */ `
struct P {
  L : vec3<f32>,
  n : u32,
  nc : vec3<u32>,
  stride : u32,
  two : u32,
  ncell : u32,
  enforce2d : u32,
  dt : f32,
};

@group(0) @binding(0) var<uniform> p : P;
@group(0) @binding(1) var<storage, read_write> atomX : array<vec4<f32>>;     // x - lo, w = type
@group(0) @binding(2) var<storage, read_write> atomV : array<vec4<f32>>;     // v, w = dt/2 * ftm2v / m
@group(0) @binding(3) var<storage, read_write> atomF : array<vec4<f32>>;     // f
@group(0) @binding(4) var<storage, read_write> image : array<vec4<i32>>;
@group(0) @binding(5) var<storage, read_write> counts : array<atomic<u32>>;  // atoms per cell
@group(0) @binding(6) var<storage, read_write> cellData : array<u32>;        // start[ncell+1] | cellOf[n] | rank[n]
@group(0) @binding(7) var<storage, read_write> sortedPos : array<vec4<f32>>; // per slot: x, w = type
@group(0) @binding(8) var<storage, read> coef : array<vec4<f32>>;            // [cutsq f12 f6 _], [e12 e6 eshift _]
@group(0) @binding(9) var<storage, read_write> slotOut : array<vec4<f32>>;   // per slot: [f, e], [w, _, _, _]

fn cellOf3(x : vec3<f32>) -> vec3<u32> {
  let q = vec3<f32>(p.nc) * x / p.L;
  var c = vec3<u32>(max(floor(q), vec3<f32>(0.0)));
  c = min(c, p.nc - vec3<u32>(1u));
  if (p.two == 1u) { c.z = 0u; }
  return c;
}
fn cellIndex(c : vec3<u32>) -> u32 { return (c.z * p.nc.y + c.y) * p.nc.x + c.x; }

@compute @workgroup_size(${WG})
fn kickDrift(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let v = atomV[i];
  var vel = v.xyz + v.w * atomF[i].xyz;
  if (p.enforce2d == 1u) { vel.z = 0.0; }
  let xi = atomX[i];
  var x = xi.xyz + p.dt * vel;
  var im = image[i];
  for (var d = 0u; d < 3u; d = d + 1u) {
    if (d == 2u && p.two == 1u) { continue; }
    if (x[d] >= p.L[d]) { x[d] = x[d] - p.L[d]; im[d] = im[d] + 1; }
    else if (x[d] < 0.0) { x[d] = x[d] + p.L[d]; im[d] = im[d] - 1; }
    // fp32 rounding can land exactly on L after the subtraction
    if (x[d] >= p.L[d]) { x[d] = 0.0; }
  }
  atomV[i] = vec4<f32>(vel, v.w);
  atomX[i] = vec4<f32>(x, xi.w);
  image[i] = im;
}

@compute @workgroup_size(${WG})
fn clearCounts(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x < p.ncell) { atomicStore(&counts[gid.x], 0u); }
}

@compute @workgroup_size(${WG})
fn countCells(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let c = cellIndex(cellOf3(atomX[i].xyz));
  let r = atomicAdd(&counts[c], 1u);
  cellData[p.ncell + 1u + i] = c;
  cellData[p.ncell + 1u + p.n + i] = r;
}

var<workgroup> partial : array<u32, ${WG}>;

@compute @workgroup_size(${WG})
fn scan(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  let per = (p.ncell + ${WG - 1}u) / ${WG}u;
  let s0 = min(t * per, p.ncell);
  let s1 = min(s0 + per, p.ncell);
  var sum = 0u;
  for (var c = s0; c < s1; c = c + 1u) { sum = sum + atomicLoad(&counts[c]); }
  partial[t] = sum;
  workgroupBarrier();
  if (t == 0u) {
    var acc = 0u;
    for (var k = 0u; k < ${WG}u; k = k + 1u) { let v = partial[k]; partial[k] = acc; acc = acc + v; }
    cellData[p.ncell] = acc;
  }
  workgroupBarrier();
  var acc = partial[t];
  for (var c = s0; c < s1; c = c + 1u) { cellData[c] = acc; acc = acc + atomicLoad(&counts[c]); }
}

@compute @workgroup_size(${WG})
fn scatter(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let c = cellData[p.ncell + 1u + i];
  let slot = cellData[c] + cellData[p.ncell + 1u + p.n + i];
  sortedPos[slot] = atomX[i];
}

fn wrapCell(c : i32, n : u32) -> u32 {
  let m = i32(n);
  return u32(((c % m) + m) % m);
}

@compute @workgroup_size(${WG})
fn force(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let pi = sortedPos[i];
  let ti = u32(pi.w);
  let c3 = cellOf3(pi.xyz);
  let cx = i32(c3.x); let cy = i32(c3.y); let cz = i32(c3.z);
  var zr = 1;
  if (p.two == 1u) { zr = 0; }
  var f = vec3<f32>(0.0, 0.0, 0.0);
  var e = 0.0;
  var w = 0.0;
  for (var dz = -zr; dz <= zr; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let c2 = (wrapCell(cz + dz, p.nc.z) * p.nc.y + wrapCell(cy + dy, p.nc.y)) * p.nc.x + wrapCell(cx + dx, p.nc.x);
        let jEnd = cellData[c2 + 1u];
        for (var j = cellData[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = sortedPos[j];
          var d = pi.xyz - pj.xyz;
          d = d - p.L * round(d / p.L);
          if (p.two == 1u) { d.z = 0.0; }
          let r2 = dot(d, d);
          let k = ti * p.stride + u32(pj.w);
          let a = coef[2u * k];
          if (r2 >= a.x) { continue; }
          let b = coef[2u * k + 1u];
          let u = 1.0 / r2;
          let u3 = u * u * u;
          let fpair = u * u3 * (a.y * u3 - a.z);
          f = f + fpair * d;
          e = e + 0.5 * (u3 * (b.x * u3 - b.y) - b.z);
          w = w + 0.5 * fpair * r2;
        }
      }
    }
  }
  slotOut[2u * i] = vec4<f32>(f, e);
  slotOut[2u * i + 1u] = vec4<f32>(w, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(${WG})
fn kick2(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let c = cellData[p.ncell + 1u + i];
  let slot = cellData[c] + cellData[p.ncell + 1u + p.n + i];
  var fi = slotOut[2u * slot].xyz;
  if (p.enforce2d == 1u) { fi.z = 0.0; }
  atomF[i] = vec4<f32>(fi, 0.0);
  let v = atomV[i];
  var vel = v.xyz + v.w * fi;
  if (p.enforce2d == 1u) { vel.z = 0.0; }
  atomV[i] = vec4<f32>(vel, v.w);
}
`;

const ENTRY = ['kickDrift', 'clearCounts', 'countCells', 'scan', 'scatter', 'force', 'kick2'] as const;
type Entry = typeof ENTRY[number];
/** Bindings each entry point uses (its 'auto' layout holds exactly these). */
const USES: Record<Entry, number[]> = {
  kickDrift: [0, 1, 2, 3, 4],
  clearCounts: [0, 5],
  countCells: [0, 1, 5, 6],
  scan: [0, 5, 6],
  scatter: [0, 1, 6, 7],
  force: [0, 6, 7, 8, 9],
  kick2: [0, 2, 3, 6, 9],
};

export interface AdvanceOptions {
  nc: [number, number, number];
  stride: number;
  /** Coefficients as packed by webgpuForces.ts (8 floats per type pair). */
  coef: Float32Array;
  enforce2d: boolean;
}

export class ResidentStepper {
  private params: GPUBuffer;
  private bufs: Record<number, GPUBuffer> = {};
  private staging: GPUBuffer | null = null;
  private atomCap = 0;
  private cellCap = 0;
  private coefCap = 0;
  private groups: Record<Entry, GPUBindGroup> | null = null;

  private constructor(private device: GPUDevice, private pipelines: Record<Entry, GPUComputePipeline>) {
    this.params = device.createBuffer({ size: 64, usage: BUF_UNIFORM | BUF_COPY_DST });
  }

  /** Compiles the kernels; rejects if this device cannot run them. */
  static async create(device: GPUDevice): Promise<ResidentStepper> {
    device.pushErrorScope('validation');
    const module = device.createShaderModule({ code: SHADER });
    const pipelines = Object.fromEntries(ENTRY.map((e) => [e,
      device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: e } })])) as Record<Entry, GPUComputePipeline>;
    const err = await device.popErrorScope();
    if (err) throw new Error(`WebGPU resident kernels: ${err.message}`);
    return new ResidentStepper(device, pipelines);
  }

  private ensure(n: number, ncell: number, coefBytes: number): void {
    const needAtoms = this.atomCap < n;
    const needCells = this.cellCap < ncell + 1 + 2 * n || needAtoms;
    const needCoef = this.coefCap < coefBytes;
    if (!needAtoms && !needCells && !needCoef && this.groups) return;
    const mk = (size: number, usage: number) => this.device.createBuffer({ size: Math.max(16, size), usage });
    const S = BUF_STORAGE | BUF_COPY_DST | BUF_COPY_SRC;
    if (needAtoms) {
      this.atomCap = Math.max(64, Math.ceil(n * 1.25));
      for (const b of [1, 2, 3, 4, 7, 9]) this.bufs[b]?.destroy();
      this.staging?.destroy();
      this.bufs[1] = mk(16 * this.atomCap, S);
      this.bufs[2] = mk(16 * this.atomCap, S);
      this.bufs[3] = mk(16 * this.atomCap, S);
      this.bufs[4] = mk(16 * this.atomCap, S);
      this.bufs[7] = mk(16 * this.atomCap, S);
      this.bufs[9] = mk(32 * this.atomCap, S);
      this.staging = mk(96 * this.atomCap, BUF_MAP_READ | BUF_COPY_DST);
    }
    if (needCells) {
      this.cellCap = Math.max(256, Math.ceil((ncell + 1 + 2 * this.atomCap) * 1.25));
      this.bufs[5]?.destroy();
      this.bufs[6]?.destroy();
      this.bufs[5] = mk(4 * this.cellCap, S);
      this.bufs[6] = mk(4 * this.cellCap, S);
    }
    if (needCoef) {
      this.coefCap = coefBytes;
      this.bufs[8]?.destroy();
      this.bufs[8] = mk(coefBytes, S);
    }
    this.groups = Object.fromEntries(ENTRY.map((e) => [e, this.device.createBindGroup({
      layout: this.pipelines[e].getBindGroupLayout(0),
      entries: USES[e].map((b) => ({ binding: b, resource: { buffer: b === 0 ? this.params : this.bufs[b] } })),
    })])) as Record<Entry, GPUBindGroup>;
  }

  /** Advances `s` by `nsteps` velocity-Verlet steps on the GPU; returns the final forces' pe and virial. */
  async advance(s: SimState, nsteps: number, o: AdvanceOptions): Promise<ForceResult> {
    const n = s.n;
    const two = s.dimension === 2;
    const ncell = o.nc[0] * o.nc[1] * o.nc[2];
    this.ensure(n, ncell, o.coef.byteLength);
    const lo = s.box.lo;
    const L = [0, 1, 2].map((d) => s.box.hi[d] - lo[d]);
    // upload the host state (it may have changed between runs: velocity, ...)
    const X = new Float32Array(4 * n), V = new Float32Array(4 * n), F = new Float32Array(4 * n);
    const I = new Int32Array(4 * n);
    const kf = 0.5 * s.dt * s.units.ftm2v;
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        X[4 * i + d] = s.x[3 * i + d] - lo[d];
        V[4 * i + d] = s.v[3 * i + d];
        F[4 * i + d] = s.f[3 * i + d];
        I[4 * i + d] = s.image[3 * i + d];
      }
      X[4 * i + 3] = s.type[i];
      V[4 * i + 3] = kf / massOf(s, i);
    }
    const q = this.device.queue;
    // an invalid command buffer is skipped silently, which would read back zeros
    this.device.pushErrorScope('out-of-memory');
    this.device.pushErrorScope('validation');
    const pbuf = new ArrayBuffer(64);
    const pf = new Float32Array(pbuf);
    const pu = new Uint32Array(pbuf);
    pf[0] = L[0]; pf[1] = L[1]; pf[2] = L[2]; pu[3] = n;
    pu[4] = o.nc[0]; pu[5] = o.nc[1]; pu[6] = o.nc[2]; pu[7] = o.stride;
    pu[8] = two ? 1 : 0; pu[9] = ncell; pu[10] = o.enforce2d ? 1 : 0; pf[11] = s.dt;
    q.writeBuffer(this.params, 0, pbuf);
    q.writeBuffer(this.bufs[1], 0, X);
    q.writeBuffer(this.bufs[2], 0, V);
    q.writeBuffer(this.bufs[3], 0, F);
    q.writeBuffer(this.bufs[4], 0, I);
    q.writeBuffer(this.bufs[8], 0, o.coef);

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const g = this.groups!;
    const run = (e: Entry, groups: number) => { pass.setPipeline(this.pipelines[e]); pass.setBindGroup(0, g[e]); pass.dispatchWorkgroups(groups); };
    const atomGroups = Math.ceil(n / WG);
    const cellGroups = Math.ceil(ncell / WG);
    for (let k = 0; k < nsteps; k++) {
      run('kickDrift', atomGroups);
      run('clearCounts', cellGroups);
      run('countCells', atomGroups);
      run('scan', 1);
      run('scatter', atomGroups);
      run('force', atomGroups);
      run('kick2', atomGroups);
    }
    pass.end();
    // one staging buffer: X | V | F | image | slotOut
    const st = this.staging!;
    enc.copyBufferToBuffer(this.bufs[1], 0, st, 0, 16 * n);
    enc.copyBufferToBuffer(this.bufs[2], 0, st, 16 * n, 16 * n);
    enc.copyBufferToBuffer(this.bufs[3], 0, st, 32 * n, 16 * n);
    enc.copyBufferToBuffer(this.bufs[4], 0, st, 48 * n, 16 * n);
    enc.copyBufferToBuffer(this.bufs[9], 0, st, 64 * n, 32 * n);
    q.submit([enc.finish()]);
    const errs = [await this.device.popErrorScope(), await this.device.popErrorScope()];
    const err = errs.find((e) => e);
    if (err) throw new Error(`WebGPU: ${err.message}`);
    await st.mapAsync(MAP_READ, 0, 96 * n);
    const raw = st.getMappedRange(0, 96 * n);
    const Xo = new Float32Array(raw, 0, 4 * n);
    const Vo = new Float32Array(raw, 16 * n, 4 * n);
    const Fo = new Float32Array(raw, 32 * n, 4 * n);
    const Io = new Int32Array(raw, 48 * n, 4 * n);
    const So = new Float32Array(raw, 64 * n, 8 * n);
    let pe = 0;
    let virial = 0;
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        s.x[3 * i + d] = Xo[4 * i + d] + lo[d];
        s.v[3 * i + d] = Vo[4 * i + d];
        s.f[3 * i + d] = Fo[4 * i + d];
        s.image[3 * i + d] = Io[4 * i + d];
      }
      pe += So[8 * i + 3];
      virial += So[8 * i + 4];
    }
    st.unmap();
    return { pe, virial };
  }

  dispose(): void {
    for (const b of Object.values(this.bufs)) b.destroy();
    this.staging?.destroy();
    this.params.destroy();
    this.bufs = {};
  }
}
