import type { ForceBackend, ForceResult, PairTable, SimState } from '../types';
import { pairArrays, type PairArrays } from '../pairs';
import { CpuForceBackend } from '../cpu/forces';
import { ResidentStepper } from './resident';

/*
 * WebGPU Lennard-Jones forces (docs/design/notebook.md, "Backends", v1).
 *
 * Per call the CPU builds a cell list (cells of edge >= the largest cutoff,
 * at least 3 per periodic dimension, as in cpu/forces.ts), sorts the atoms by
 * cell and uploads fp32 positions relative to the box origin. One compute
 * invocation per atom scans its own and the 26 (8 in 2D) neighbouring cells,
 * applies the minimum image, and accumulates its OWN force plus half of each
 * pair's energy and virial — every pair is evaluated from both sides, so no
 * atomics are needed. The CPU reads the per-atom results back, scatters the
 * forces into state.f and sums energy and virial in fp64.
 *
 * Boxes shorter than 3 cutoffs in some dimension have too few cells for the
 * 27-cell scan to be exact; those (small) systems use the fp64 CPU backend.
 *
 * Kernel physics, with u = 1/r^2 and u3 = u^3 (pairs.ts):
 *   F.r/r^2 = u u3 (f12 u3 - f6),  E = u3 (e12 u3 - e6) - eshift,  W = F.r.
 * Written from the textbook LJ force law; no LAMMPS GPU-package code.
 */

// WebGPU flag values (the TS DOM lib declares the types but not these
// namespaces as values in every configuration).
const BUF_MAP_READ = 0x0001;
const BUF_COPY_SRC = 0x0004;
const BUF_COPY_DST = 0x0008;
const BUF_UNIFORM = 0x0040;
const BUF_STORAGE = 0x0080;
const MAP_READ = 0x0001;

const WORKGROUP = 64;

const KERNEL = /* wgsl */ `
struct Params {
  L : vec3<f32>,
  n : u32,
  nc : vec3<u32>,
  stride : u32,
  two : u32,
  cellOfBase : u32,   // offset of the per-slot cell indices inside ints
  pad1 : u32,
  pad2 : u32,
};

// Four storage buffers: the limit of WebGPU compatibility mode, which is how
// some GPUs (e.g. NVIDIA through ANGLE on Linux) are exposed.
@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;      // xyz, w = type
@group(0) @binding(2) var<storage, read> ints : array<u32>;           // cellStart[ncell+1], then cellOf[n]
@group(0) @binding(3) var<storage, read> coef : array<vec4<f32>>;     // [cutsq f12 f6 _], [e12 e6 eshift _]
@group(0) @binding(4) var<storage, read_write> outv : array<f32>;     // fx fy fz e w per slot

fn wrapCell(c : i32, n : u32) -> u32 {
  let m = i32(n);
  return u32(((c % m) + m) % m);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  let L = P.L;
  let nc = P.nc;
  let c = ints[P.cellOfBase + i];
  let cx = i32(c % nc.x);
  let cy = i32((c / nc.x) % nc.y);
  let cz = i32(c / (nc.x * nc.y));
  var zr = 1;
  if (P.two == 1u) { zr = 0; }
  var f = vec3<f32>(0.0, 0.0, 0.0);
  var e = 0.0;
  var w = 0.0;
  for (var dz = -zr; dz <= zr; dz = dz + 1) {
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let c2 = (wrapCell(cz + dz, nc.z) * nc.y + wrapCell(cy + dy, nc.y)) * nc.x + wrapCell(cx + dx, nc.x);
        let jEnd = ints[c2 + 1u];
        for (var j = ints[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = pos[j];
          var d = pi.xyz - pj.xyz;
          d = d - L * round(d / L);
          if (P.two == 1u) { d.z = 0.0; }
          let r2 = dot(d, d);
          let k = ti * P.stride + u32(pj.w);
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
  let o = 5u * i;
  outv[o] = f.x;
  outv[o + 1u] = f.y;
  outv[o + 2u] = f.z;
  outv[o + 3u] = e;
  outv[o + 4u] = w;
}
`;

type AdapterRequest = { powerPreference?: 'high-performance'; featureLevel?: 'compatibility' };
type Gpu = { requestAdapter(opts?: AdapterRequest): Promise<GPUAdapter | null> };

const navigatorGpu = (): Gpu | null => {
  const nav = (globalThis as { navigator?: { gpu?: Gpu } }).navigator;
  return nav?.gpu ?? null;
};

const isFallback = (a: GPUAdapter): boolean =>
  !!((a.info as GPUAdapterInfo & { isFallbackAdapter?: boolean })?.isFallbackAdapter
    ?? (a as GPUAdapter & { isFallbackAdapter?: boolean }).isFallbackAdapter);

export interface WebGpuOptions {
  /**
   * Accept a software fallback adapter (SwiftShader). It runs WebGPU on the
   * CPU and is slower than the CPU engine, so the notebook declines it; the
   * real-browser tests use it to check the kernel on machines without a GPU.
   */
  allowFallback?: boolean;
}

/** What WebGPU this browser offers, without creating a device. */
export const webgpuAdapterKind = async (): Promise<'hardware' | 'software' | 'none'> => {
  const gpu = navigatorGpu();
  if (!gpu) return 'none';
  let software = false;
  for (const o of [{ powerPreference: 'high-performance' }, { powerPreference: 'high-performance', featureLevel: 'compatibility' }] as AdapterRequest[]) {
    let a: GPUAdapter | null = null;
    try { a = await gpu.requestAdapter(o); } catch { a = null; }
    if (!a) continue;
    if (!isFallback(a)) return 'hardware';
    software = true;
  }
  return software ? 'software' : 'none';
};

/**
 * A WebGPU backend on the best adapter: a hardware core-WebGPU adapter, else
 * a hardware compatibility-mode adapter (how e.g. NVIDIA GPUs are exposed
 * through ANGLE on Linux), else — only with allowFallback — a software one.
 * Null when there is none of those.
 */
export const createWebGpuBackend = async (opts: WebGpuOptions = {}): Promise<WebGpuForceBackend | null> => {
  const gpu = navigatorGpu();
  if (!gpu) return null;
  const ask = async (o: AdapterRequest): Promise<GPUAdapter | null> => {
    try { return await gpu.requestAdapter(o); } catch { return null; }
  };
  const tried: GPUAdapter[] = [];
  let chosen: GPUAdapter | null = null;
  for (const o of [{ powerPreference: 'high-performance' }, { powerPreference: 'high-performance', featureLevel: 'compatibility' }] as AdapterRequest[]) {
    const a = await ask(o);
    if (!a) continue;
    if (!isFallback(a)) { chosen = a; break; }
    tried.push(a);
  }
  if (!chosen && opts.allowFallback) chosen = tried[0] ?? null;
  if (!chosen) return null;
  const device = await chosen.requestDevice();
  const info = chosen.info;
  const name = [info?.vendor, info?.architecture].filter(Boolean).join(' ') || info?.description || 'adapter';
  const tags = [
    isFallback(chosen) ? 'software' : '',
    (chosen as GPUAdapter & { featureLevel?: string }).featureLevel === 'compatibility' ? 'compat' : '',
  ].filter(Boolean);
  return new WebGpuForceBackend(device, `WebGPU · ${name}${tags.length ? ` (${tags.join(', ')})` : ''}`, isFallback(chosen));
};

interface Buffers {
  pos: GPUBuffer;
  out: GPUBuffer;
  staging: GPUBuffer;
  atomCap: number;
  ints: GPUBuffer;
  intsCap: number;
  coef: GPUBuffer;
  coefCap: number;
  bind: GPUBindGroup | null;
}

export class WebGpuForceBackend implements ForceBackend {
  readonly kind = 'webgpu' as const;
  private pipeline: GPUComputePipeline;
  private params: GPUBuffer;
  private bufs: Buffers | null = null;
  private cpu = new CpuForceBackend();
  private cache: { key: string; arrays: PairArrays; coef: Float32Array } | null = null;
  // CPU-side scratch reused across calls
  private sorted = new Int32Array(0);
  private cellOfAtom = new Int32Array(0);
  private posData = new Float32Array(0);
  private intsData = new Uint32Array(0);
  private disposed = false;
  private resident: Promise<ResidentStepper> | null = null;

  constructor(private device: GPUDevice, readonly label: string, readonly software = false) {
    const module = device.createShaderModule({ code: KERNEL });
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    this.params = device.createBuffer({ size: 48, usage: BUF_UNIFORM | BUF_COPY_DST });
  }

  private coefFor(table: PairTable): { arrays: PairArrays; coef: Float32Array } {
    const key = JSON.stringify([table.pairs, table.shift]);
    if (!this.cache || this.cache.key !== key) {
      const arrays = pairArrays(table);
      const npairs = arrays.stride * arrays.stride;
      const coef = new Float32Array(8 * npairs);
      for (let k = 0; k < npairs; k++) {
        coef.set([arrays.cutsq[k], arrays.f12[k], arrays.f6[k], 0, arrays.e12[k], arrays.e6[k], arrays.eshift[k], 0], 8 * k);
      }
      this.cache = { key, arrays, coef };
    }
    return this.cache;
  }

  private buffer(size: number, usage: number): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, size), usage });
  }

  /** Grows GPU buffers to fit n atoms, ncell cells and the coefficient table. */
  private ensure(n: number, ncell: number, coefBytes: number): Buffers {
    let b = this.bufs;
    const intsNeed = ncell + 1 + n;
    const needAtoms = !b || b.atomCap < n;
    const needInts = !b || b.intsCap < intsNeed;
    const needCoef = !b || b.coefCap < coefBytes;
    if (!needAtoms && !needInts && !needCoef) return b!;
    const atomCap = needAtoms ? Math.max(64, Math.ceil(n * 1.25)) : b!.atomCap;
    const intsCap = needInts ? Math.max(128, Math.ceil(intsNeed * 1.25)) : b!.intsCap;
    const coefCap = needCoef ? coefBytes : b!.coefCap;
    if (b && needAtoms) { b.pos.destroy(); b.out.destroy(); b.staging.destroy(); }
    if (b && needInts) b.ints.destroy();
    if (b && needCoef) b.coef.destroy();
    b = {
      pos: needAtoms ? this.buffer(16 * atomCap, BUF_STORAGE | BUF_COPY_DST) : b!.pos,
      out: needAtoms ? this.buffer(20 * atomCap, BUF_STORAGE | BUF_COPY_SRC) : b!.out,
      staging: needAtoms ? this.buffer(20 * atomCap, BUF_MAP_READ | BUF_COPY_DST) : b!.staging,
      atomCap,
      ints: needInts ? this.buffer(4 * intsCap, BUF_STORAGE | BUF_COPY_DST) : b!.ints,
      intsCap,
      coef: needCoef ? this.buffer(coefBytes, BUF_STORAGE | BUF_COPY_DST) : b!.coef,
      coefCap,
      bind: null,
    };
    b.bind = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: b.pos } },
        { binding: 2, resource: { buffer: b.ints } },
        { binding: 3, resource: { buffer: b.coef } },
        { binding: 4, resource: { buffer: b.out } },
      ],
    });
    this.bufs = b;
    return b;
  }

  async compute(state: SimState, table: PairTable): Promise<ForceResult> {
    if (this.disposed) throw new Error('the WebGPU backend was disposed');
    const { arrays, coef } = this.coefFor(table);
    const { n, x, type } = state;
    if (n === 0 || arrays.maxCutoff <= 0) {
      state.f.fill(0);
      return { pe: 0, virial: 0 };
    }
    const two = state.dimension === 2;
    const L = [0, 1, 2].map((d) => state.box.hi[d] - state.box.lo[d]);
    const nc = [0, 1, 2].map((d) => (d === 2 && two ? 1 : Math.floor(L[d] / arrays.maxCutoff)));
    if (nc.some((c, d) => !(d === 2 && two) && c < 3)) return this.cpu.compute(state, table);
    const ncell = nc[0] * nc[1] * nc[2];

    // --- counting sort by cell (same assignment as cpu/cells.ts) ---
    if (this.sorted.length < n) {
      this.sorted = new Int32Array(n);
      this.cellOfAtom = new Int32Array(n);
      this.posData = new Float32Array(4 * n);
    }
    if (this.intsData.length < ncell + 1 + n) this.intsData = new Uint32Array(ncell + 1 + n);
    const { sorted, cellOfAtom, posData } = this;
    // ints = [cellStart (ncell + 1) | cellOf per slot (n)]
    const ints = this.intsData;
    const base = ncell + 1;
    ints.fill(0, 0, base);
    const lo = state.box.lo;
    const s = [nc[0] / L[0], nc[1] / L[1], nc[2] / L[2]];
    for (let i = 0; i < n; i++) {
      let c = 0;
      for (let d = 2; d >= 0; d--) {
        let k = d === 2 && two ? 0 : Math.floor((x[3 * i + d] - lo[d]) * s[d]);
        if (k >= nc[d]) k = nc[d] - 1; else if (k < 0) k = 0;
        c = c * nc[d] + k;
      }
      cellOfAtom[i] = c;
      ints[c + 1]++;
    }
    for (let c = 0; c < ncell; c++) ints[c + 1] += ints[c];
    const fill = ints.slice(0, ncell);
    for (let i = 0; i < n; i++) {
      const c = cellOfAtom[i];
      const slot = fill[c]++;
      sorted[slot] = i;
      ints[base + slot] = c;
      posData[4 * slot] = x[3 * i] - lo[0];
      posData[4 * slot + 1] = x[3 * i + 1] - lo[1];
      posData[4 * slot + 2] = two ? 0 : x[3 * i + 2] - lo[2];
      posData[4 * slot + 3] = type[i];
    }

    // --- upload, dispatch, read back ---
    const b = this.ensure(n, ncell, coef.byteLength);
    const params = new ArrayBuffer(48);
    const pf = new Float32Array(params);
    const pu = new Uint32Array(params);
    pf[0] = L[0]; pf[1] = L[1]; pf[2] = L[2]; pu[3] = n;
    pu[4] = nc[0]; pu[5] = nc[1]; pu[6] = nc[2]; pu[7] = arrays.stride;
    pu[8] = two ? 1 : 0; pu[9] = base;
    const q = this.device.queue;
    q.writeBuffer(this.params, 0, params);
    q.writeBuffer(b.pos, 0, posData.buffer, posData.byteOffset, 16 * n);
    q.writeBuffer(b.ints, 0, ints.buffer, ints.byteOffset, 4 * (base + n));
    q.writeBuffer(b.coef, 0, coef.buffer, coef.byteOffset, coef.byteLength);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, b.bind!);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP));
    pass.end();
    enc.copyBufferToBuffer(b.out, 0, b.staging, 0, 20 * n);
    q.submit([enc.finish()]);
    await b.staging.mapAsync(MAP_READ, 0, 20 * n);
    const outv = new Float32Array(b.staging.getMappedRange(0, 20 * n));
    const f = state.f;
    let pe = 0;
    let virial = 0;
    for (let slot = 0; slot < n; slot++) {
      const i = sorted[slot];
      const o = 5 * slot;
      f[3 * i] = outv[o];
      f[3 * i + 1] = outv[o + 1];
      f[3 * i + 2] = outv[o + 2];
      pe += outv[o + 3];
      virial += outv[o + 4];
    }
    b.staging.unmap();
    return { pe, virial };
  }

  /** True when `advance` can run this system entirely on the GPU. */
  canAdvance(state: SimState, table: PairTable): boolean {
    const { arrays } = this.coefFor(table);
    if (state.n === 0 || !(arrays.maxCutoff > 0)) return false;
    const two = state.dimension === 2;
    return [0, 1, 2].every((d) => (d === 2 && two)
      || (state.box.periodic[d] && Math.floor((state.box.hi[d] - state.box.lo[d]) / arrays.maxCutoff) >= 3));
  }

  /**
   * Runs `nsteps` velocity-Verlet steps (fix nve, optionally enforce2d)
   * without leaving the GPU, then writes x, v, f and images back into
   * `state`; returns the final step's energy and virial.
   */
  async advance(state: SimState, table: PairTable, nsteps: number, opts: { enforce2d: boolean }): Promise<ForceResult> {
    if (this.disposed) throw new Error('the WebGPU backend was disposed');
    const { arrays, coef } = this.coefFor(table);
    const two = state.dimension === 2;
    const nc = [0, 1, 2].map((d) => (d === 2 && two ? 1 : Math.floor((state.box.hi[d] - state.box.lo[d]) / arrays.maxCutoff))) as [number, number, number];
    this.resident ??= ResidentStepper.create(this.device);
    return (await this.resident).advance(state, nsteps, { nc, stride: arrays.stride, coef, enforce2d: opts.enforce2d });
  }

  dispose(): void {
    if (this.disposed) return;
    this.resident?.then((r) => r.dispose(), () => undefined);
    this.disposed = true;
    const b = this.bufs;
    if (b) for (const buf of [b.pos, b.out, b.staging, b.ints, b.coef]) buf.destroy();
    this.params.destroy();
    this.bufs = null;
    this.cpu.dispose();
    this.device.destroy();
  }
}
