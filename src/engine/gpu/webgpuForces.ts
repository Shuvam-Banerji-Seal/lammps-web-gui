import type { ForceBackend, ForceResult, PairTable, SimState } from '../types';
import { pairArrays, type PairArrays } from '../pairs';
import { CpuForceBackend } from '../cpu/forces';
import { ResidentStepper } from './resident';
import { TABLE_LAW, tableInnerError, type GpuPairSpec } from './pairSpec';
import { buildAtomMap, type SpecialList } from '../atoms';
import { WGSL_STYLE_LAWS } from './styleLaws';
import { EAM_WORKGROUP, KERNEL_EAM } from './eamKernels';
import { KERNEL_SW, SW_WORKGROUP } from './swKernels';
import type { GpuSwSpec } from './swSpec';
import { KERNEL_TERSOFF, TERSOFF_WORKGROUP } from './tersoffKernels';
import type { GpuTersoffSpec } from './tersoffSpec';
import type { GpuEamSpec } from './eamSpec';

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
 * The pair-style kernel (KERNEL_STYLES) scans every cell once along such an
 * edge instead, which the minimum image keeps exact down to 2 cutoffs.
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

/*
 * The kernel of the other pair styles (gpu/pairSpec.ts): the same cell scan, a style switch per pair, coefficients
 * [cutsqA cutsqC a b] [c d eshift cscale] [pad] per type pair, the charges as f32 bits after the cell indices in ints,
 * and per slot fx fy fz evdwl ecoul w. The laws are the documented ones the CPU styles compute (pairSpec.ts
 * pairSpecEval is the same arithmetic in fp64).
 */
const KERNEL_STYLES = /* wgsl */ `
struct Params {
  L : vec3<f32>,
  n : u32,
  nc : vec3<u32>,
  stride : u32,
  two : u32,
  cellOfBase : u32,
  style : u32,      // the short-range law (pairSpec.ts SPEC_LAW)
  qBase : u32,
  qqrd2e : f32,
  specBase : u32,   // special bonds: each slot's first entry in ints at specBase + slot (0: none)
  gewald : f32,     // G-ewald of the long-range Coulomb styles (0: the bare Coulomb term)
  pad2 : u32,
  wlj : vec4<f32>,  // special_bonds lj weights by order (1-2, 1-3, 1-4 at 1..3)
  wc : vec4<f32>,   // and coul weights
};

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ints : array<u32>;
@group(0) @binding(3) var<storage, read> coef : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outv : array<f32>;

fn wrapCell(c : i32, n : u32) -> u32 {
  let m = i32(n);
  return u32(((c % m) + m) % m);
}

${WGSL_STYLE_LAWS}

// pair_style table (pairSpec.ts tableLaw): the tables after the type pairs' coefficients, read as flat floats; A =
// [cutsq 0 rsq1 1/delta], B = [e g e2 cscale] offsets, C = [g2 last h^2/6 mode (0 linear, 1 spline, 2 lookup)]
fn tabAt(k : u32) -> f32 {
  return coef[k >> 2u][k & 3u];
}

fn tableLaw(r2 : f32, A : vec4<f32>, B : vec4<f32>, C : vec4<f32>) -> vec2<f32> {
  let p = (r2 - A.z) * A.w;
  let last = i32(C.y);
  let eo = u32(B.x);
  let go = u32(B.y);
  if (C.w == 2.0) {
    let m = u32(clamp(i32(floor(p)), 0, last));
    return vec2<f32>(tabAt(go + m), tabAt(eo + m));
  }
  var m = i32(floor(p));
  var t = p - floor(p);
  if (m >= last) { m = last - 1; t = 1.0; } else if (m < 0) { m = 0; t = 0.0; }
  let um = u32(m);
  if (C.w == 1.0) {
    let a = 1.0 - t;
    let b = t;
    let ca = a * a * a - a;
    let cb = b * b * b - b;
    let e2o = u32(B.z);
    let g2o = u32(C.x);
    let e = a * tabAt(eo + um) + b * tabAt(eo + um + 1u) + (ca * tabAt(e2o + um) + cb * tabAt(e2o + um + 1u)) * C.z;
    let g = a * tabAt(go + um) + b * tabAt(go + um + 1u) + (ca * tabAt(g2o + um) + cb * tabAt(g2o + um + 1u)) * C.z;
    return vec2<f32>(g, e);
  }
  let e0 = tabAt(eo + um);
  let g0 = tabAt(go + um);
  return vec2<f32>(g0 + t * (tabAt(go + um + 1u) - g0), e0 + t * (tabAt(eo + um + 1u) - e0));
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  var qi = 0.0;
  if (P.qBase > 0u) { qi = bitcast<f32>(ints[P.qBase + i]); }
  let L = P.L;
  let nc = P.nc;
  let c = ints[P.cellOfBase + i];
  let cx = i32(c % nc.x);
  let cy = i32((c / nc.x) % nc.y);
  let cz = i32(c / (nc.x * nc.y));
  // the 27 cells around the atom's own; along an edge of fewer than 3 cells (an edge of 2 to 3 cutoffs, or z in 2D)
  // each cell once, where the minimum image finds the one copy of a partner within the cutoff
  let lo = vec3<i32>(select(-1, 0, nc.x < 3u), select(-1, 0, nc.y < 3u), select(-1, 0, nc.z < 3u));
  let hi = vec3<i32>(select(1, i32(nc.x) - 1, nc.x < 3u), select(1, i32(nc.y) - 1, nc.y < 3u), select(1, i32(nc.z) - 1, nc.z < 3u));
  var f = vec3<f32>(0.0, 0.0, 0.0);
  var ev = 0.0;
  var ec = 0.0;
  var w = 0.0;
  var bad = 0.0;
  for (var dz = lo.z; dz <= hi.z; dz = dz + 1) {
    for (var dy = lo.y; dy <= hi.y; dy = dy + 1) {
      for (var dx = lo.x; dx <= hi.x; dx = dx + 1) {
        let c2 = (wrapCell(cz + dz, nc.z) * nc.y + wrapCell(cy + dy, nc.y)) * nc.x + wrapCell(cx + dx, nc.x);
        let jEnd = ints[c2 + 1u];
        for (var j = ints[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = pos[j];
          var d = pi.xyz - pj.xyz;
          d = d - L * round(d / L);
          if (P.two == 1u) { d.z = 0.0; }
          let r2 = dot(d, d);
          let k = 3u * (ti * P.stride + u32(pj.w));
          let A = coef[k];
          if (r2 >= max(A.x, A.y)) { continue; }
          let B = coef[k + 1u];
          // a special pair (1-2, 1-3, 1-4): its weights, and no pair at all when both are 0 (as the neighbor list
          // without kspace; with it the pair stays, for the Coulomb correction below)
          var flj = 1.0;
          var fc = 1.0;
          if (P.specBase > 0u) {
            let t1 = ints[P.specBase + i + 1u];
            for (var t = ints[P.specBase + i]; t < t1; t = t + 1u) {
              let e = ints[t];
              if ((e >> 2u) == j) {
                flj = P.wlj[e & 3u];
                fc = P.wc[e & 3u];
                break;
              }
            }
            if (flj == 0.0 && fc == 0.0 && P.gewald == 0.0) { continue; }
          }
          var fp = 0.0;
          if (r2 < A.x) {
            var sr = vec2<f32>(0.0, 0.0);
            if (P.style == 20u) {
              // inside a table's inner cutoff: flagged (the CPU names the pair and stops, as the CPU style does)
              if (r2 < A.z) { bad = 1.0; } else { sr = tableLaw(r2, A, B, coef[k + 2u]); }
            } else {
              sr = shortRange(P.style, r2, A, B, coef[k + 2u]);
            }
            fp = flj * sr.x;
            ev = ev + 0.5 * flj * sr.y;
          }
          if (r2 < A.y && P.qBase > 0u) {
            let qj = bitcast<f32>(ints[P.qBase + j]);
            if (qi != 0.0 && qj != 0.0) {
              let r = sqrt(r2);
              let pre = P.qqrd2e * B.w * qi * qj / r;
              if (P.gewald > 0.0) {
                // the Ewald real-space term, erfc by the polynomial fit (erfc.ts erfcPoly), less (1 - w) of the bare
                // term for a special pair (pairSpec.ts pairSpecEval)
                let gr = P.gewald * r;
                let ex = exp(-gr * gr);
                let t = 1.0 / (1.0 + 0.3275911 * gr);
                let erfc = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * ex;
                fp = fp + pre * (erfc + 1.12837917 * gr * ex - (1.0 - fc)) / r2;
                ec = ec + 0.5 * pre * (erfc - (1.0 - fc));
              } else {
                fp = fp + fc * pre / r2;
                ec = ec + 0.5 * fc * pre;
              }
            }
          }
          f = f + fp * d;
          w = w + 0.5 * fp * r2;
        }
      }
    }
  }
  let o = 6u * i;
  outv[o] = f.x;
  outv[o + 1u] = f.y;
  outv[o + 2u] = f.z;
  outv[o + 3u] = ev;
  outv[o + 4u] = select(ec, -1.0, bad > 0.0);
  outv[o + 5u] = w;
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

/** What the resource monitor shows about the WebGPU adapter the engine would use (no device is created). */
export interface WebGpuAdapterInfo {
  kind: 'hardware' | 'software' | 'none';
  /** e.g. "nvidia ampere", from GPUAdapterInfo vendor and architecture (may be empty: browsers hide them). */
  name: string;
  /** Exposed only through WebGPU compatibility mode (e.g. NVIDIA through ANGLE on Linux). */
  compatibility: boolean;
  /** Largest storage buffer one binding may hold, in MB (bounds the atoms the GPU path can take). */
  maxStorageBufferMB: number;
}

export const webgpuAdapterInfo = async (): Promise<WebGpuAdapterInfo> => {
  const none: WebGpuAdapterInfo = { kind: 'none', name: '', compatibility: false, maxStorageBufferMB: 0 };
  const gpu = navigatorGpu();
  if (!gpu) return none;
  let software: WebGpuAdapterInfo | null = null;
  for (const o of [{ powerPreference: 'high-performance' }, { powerPreference: 'high-performance', featureLevel: 'compatibility' }] as AdapterRequest[]) {
    let a: GPUAdapter | null = null;
    try { a = await gpu.requestAdapter(o); } catch { a = null; }
    if (!a) continue;
    const info = a.info;
    const r: WebGpuAdapterInfo = {
      kind: isFallback(a) ? 'software' : 'hardware',
      name: [info?.vendor, info?.architecture].filter(Boolean).join(' ') || info?.description || '',
      compatibility: o.featureLevel === 'compatibility',
      maxStorageBufferMB: Math.round((a.limits?.maxStorageBufferBindingSize ?? 0) / 2 ** 20),
    };
    if (r.kind === 'hardware') return r;
    software ??= r;
  }
  return software ?? none;
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

/** The f32 coefficients for the kernels' array<vec4<f32>> (a multiple of four floats; pair table's data ends anywhere). */
const padToVec4 = (c: Float64Array): Float32Array => {
  const out = new Float32Array(Math.ceil(c.length / 4) * 4);
  out.set(c);
  return out;
};

/**
 * The compute pipelines of one WGSL module, created inside a validation error scope: a kernel that does not compile, or
 * an entry point that does not validate, stops the run with the compiler's message. Without the scope an invalid
 * pipeline dispatches nothing, and the run would integrate with zero forces.
 */
const checkedPipelines = async <E extends string>(dev: GPUDevice, code: string, entries: readonly E[]): Promise<Record<E, GPUComputePipeline>> => {
  dev.pushErrorScope('validation');
  const module = dev.createShaderModule({ code });
  const out = {} as Record<E, GPUComputePipeline>;
  for (const e of entries) out[e] = dev.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: e } });
  const err = await dev.popErrorScope();
  if (err) {
    const info = await module.getCompilationInfo();
    const first = info.messages.find((m) => m.type === 'error');
    throw new Error(`a GPU kernel did not compile: ${first ? `line ${first.lineNum}: ${first.message}` : err.message}`);
  }
  return out;
};

/**
 * Runs one force evaluation inside a validation error scope, so that an invalid bind group or dispatch (which WebGPU
 * skips, leaving the output buffer as it was) stops the run instead of giving zero forces.
 */
const validated = async <T>(dev: GPUDevice, run: () => Promise<T>): Promise<T> => {
  dev.pushErrorScope('validation');
  let out: T;
  try {
    out = await run();
  } finally {
    const err = await dev.popErrorScope();
    if (err) throw new Error(`a GPU force evaluation failed validation: ${err.message}`);
  }
  return out!;
};

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
    const n = state.n;
    if (n === 0 || arrays.maxCutoff <= 0) {
      state.f.fill(0);
      return { pe: 0, virial: 0 };
    }
    const cells = this.sortByCell(state, arrays.maxCutoff, 0);
    if (!cells) return this.cpu.compute(state, table);
    const { L, nc, ncell, base, two } = cells;
    const { sorted, posData } = this;
    const ints = this.intsData;

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

  /**
   * The counting sort by cell (the same assignment as cpu/cells.ts) of compute() and computeSpec(): sorted[slot] the
   * atom of each slot, posData its fp32 position from the box origin and type, intsData the cell starts (ncell + 1) then
   * the cell of each slot (n), with `extra` more words after them. Null when a periodic dimension has fewer than minCells
   * cells: 3 for the 27-cell scan of compute(); the scan of KERNEL_STYLES takes every cell once along a shorter edge.
   */
  private sortByCell(state: SimState, maxCutoff: number, extra: number, minCells = 3):
    { L: number[]; nc: number[]; ncell: number; base: number; two: boolean } | null {
    const { n, x, type } = state;
    const two = state.dimension === 2;
    const L = [0, 1, 2].map((d) => state.box.hi[d] - state.box.lo[d]);
    const nc = [0, 1, 2].map((d) => (d === 2 && two ? 1 : Math.floor(L[d] / maxCutoff)));
    if (nc.some((c, d) => !(d === 2 && two) && c < minCells)) return null;
    const ncell = nc[0] * nc[1] * nc[2];
    if (this.sorted.length < n) {
      this.sorted = new Int32Array(n);
      this.cellOfAtom = new Int32Array(n);
      this.posData = new Float32Array(4 * n);
    }
    if (this.intsData.length < ncell + 1 + n + extra) this.intsData = new Uint32Array(ncell + 1 + n + extra);
    const { sorted, cellOfAtom, posData } = this;
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
    return { L, nc, ncell, base, two };
  }

  private stylesPipeline: GPUComputePipeline | null = null;
  private styleParams: GPUBuffer | null = null;
  private sbufs: { pos: GPUBuffer; out: GPUBuffer; staging: GPUBuffer; atomCap: number; ints: GPUBuffer; intsCap: number; coef: GPUBuffer; coefCap: number; bind: GPUBindGroup } | null = null;
  private specCache: { key: string; coef: Float32Array } | null = null;

  /**
   * Forces, van der Waals and Coulomb energies and the scalar virial of the pair styles of gpu/pairSpec.ts (the kernel
   * KERNEL_STYLES), fp32 on the GPU like compute(); the energies and the virial are summed in fp64 here.
   */
  async computeSpec(state: SimState, spec: GpuPairSpec, special: SpecialList | null = null): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    return validated(this.device, () => this.computeSpecUnchecked(state, spec, special));
  }

  private async computeSpecUnchecked(state: SimState, spec: GpuPairSpec, special: SpecialList | null): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    if (this.disposed) throw new Error('the WebGPU backend was disposed');
    const n = state.n;
    if (n === 0 || !(spec.maxCutoff > 0)) {
      state.f.fill(0);
      return { pe: 0, virial: 0, evdwl: 0, ecoul: 0 };
    }
    // special bonds: each slot's entries (partner slot << 2 | order) after the charges, the slots' starts before them
    const sp = spec.special && special ? special : null;
    const spEntries = sp ? sp.offset[n] : 0;
    const extra = (spec.charges ? n : 0) + (sp ? n + 1 + spEntries : 0);
    // the minimum image is the only copy of a partner within the cutoff while each edge is at least 2 cutoffs
    const flat = state.dimension === 2;
    if ([0, 1, 2].some((d) => !(d === 2 && flat) && state.box.hi[d] - state.box.lo[d] < 2 * spec.maxCutoff)) {
      throw new Error('the box has fewer than 2 cutoffs in a periodic dimension');
    }
    const cells = this.sortByCell(state, spec.maxCutoff, extra, 1);
    if (!cells) throw new Error('the box has no cell in a periodic dimension');
    const { L, nc, base, two } = cells;
    const { sorted, posData } = this;
    const ints = this.intsData;
    if (spec.charges) {
      // the charges as f32 bits after the cells of the slots
      const qf = new Float32Array(ints.buffer, ints.byteOffset + 4 * (base + n), n);
      for (let slot = 0; slot < n; slot++) qf[slot] = state.q[sorted[slot]];
    }
    let specBase = 0;
    let intsLen = base + n + (spec.charges ? n : 0);
    if (sp) {
      specBase = intsLen;
      const entryBase = specBase + n + 1;
      const slotOf = new Int32Array(n);
      for (let slot = 0; slot < n; slot++) slotOf[sorted[slot]] = slot;
      const idx = buildAtomMap(state);
      const wl = spec.special!.lj, wcl = spec.special!.coul;
      let t = entryBase;
      for (let slot = 0; slot < n; slot++) {
        ints[specBase + slot] = t;
        const a = sorted[slot];
        const e1: number = sp.offset[a + 1];
        for (let e: number = sp.offset[a]; e < e1; e++) {
          const o: number = sp.order[e];
          if (wl[o] === 1 && wcl[o] === 1) continue;
          const pid: number = sp.partner[e];
          const pb: number = pid < idx.length ? idx[pid] : -1;
          if (pb < 0 || pb >= n) continue;
          ints[t++] = (slotOf[pb] << 2) | o;
        }
      }
      ints[specBase + n] = t;
      intsLen = t;
    }
    if (!this.specCache || this.specCache.key !== spec.key) this.specCache = { key: spec.key, coef: padToVec4(spec.coef) };
    const coef = this.specCache.coef;
    const dev = this.device;
    if (!this.stylesPipeline) {
      this.stylesPipeline = (await checkedPipelines(dev, KERNEL_STYLES, ['main'] as const)).main;
      this.styleParams = dev.createBuffer({ size: 96, usage: BUF_UNIFORM | BUF_COPY_DST });
    }
    let b = this.sbufs;
    if (!b || b.atomCap < n || b.intsCap < intsLen || b.coefCap < coef.byteLength) {
      if (b) for (const buf of [b.pos, b.out, b.staging, b.ints, b.coef]) buf.destroy();
      const atomCap = Math.max(64, Math.ceil(n * 1.25)), intsCap = Math.max(128, Math.ceil(intsLen * 1.25));
      const pos = this.buffer(16 * atomCap, BUF_STORAGE | BUF_COPY_DST);
      const out = this.buffer(24 * atomCap, BUF_STORAGE | BUF_COPY_SRC);
      const staging = this.buffer(24 * atomCap, BUF_MAP_READ | BUF_COPY_DST);
      const intsBuf = this.buffer(4 * intsCap, BUF_STORAGE | BUF_COPY_DST);
      const coefBuf = this.buffer(coef.byteLength, BUF_STORAGE | BUF_COPY_DST);
      const bind = dev.createBindGroup({
        layout: this.stylesPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.styleParams! } },
          { binding: 1, resource: { buffer: pos } },
          { binding: 2, resource: { buffer: intsBuf } },
          { binding: 3, resource: { buffer: coefBuf } },
          { binding: 4, resource: { buffer: out } },
        ],
      });
      b = this.sbufs = { pos, out, staging, atomCap, ints: intsBuf, intsCap, coef: coefBuf, coefCap: coef.byteLength, bind };
    }
    const params = new ArrayBuffer(96);
    const pf = new Float32Array(params);
    const pu = new Uint32Array(params);
    pf[0] = L[0]; pf[1] = L[1]; pf[2] = L[2]; pu[3] = n;
    pu[4] = nc[0]; pu[5] = nc[1]; pu[6] = nc[2]; pu[7] = spec.stride;
    pu[8] = two ? 1 : 0; pu[9] = base; pu[10] = spec.law; pu[11] = spec.charges ? base + n : 0;
    pf[12] = spec.qqrd2e; pu[13] = specBase; pf[14] = spec.gewald;
    if (spec.special) for (let o = 0; o < 4; o++) { pf[16 + o] = spec.special.lj[o]; pf[20 + o] = spec.special.coul[o]; }
    const q = dev.queue;
    q.writeBuffer(this.styleParams!, 0, params);
    q.writeBuffer(b.pos, 0, posData.buffer, posData.byteOffset, 16 * n);
    q.writeBuffer(b.ints, 0, ints.buffer, ints.byteOffset, 4 * intsLen);
    q.writeBuffer(b.coef, 0, coef.buffer, coef.byteOffset, coef.byteLength);
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.stylesPipeline);
    pass.setBindGroup(0, b.bind);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP));
    pass.end();
    enc.copyBufferToBuffer(b.out, 0, b.staging, 0, 24 * n);
    q.submit([enc.finish()]);
    await b.staging.mapAsync(MAP_READ, 0, 24 * n);
    const outv = new Float32Array(b.staging.getMappedRange(0, 24 * n));
    const f = state.f;
    let evdwl = 0, ecoul = 0, virial = 0;
    // pair_style table: an atom with a pair inside a table's inner cutoff (ecoul slot -1; the style has no Coulomb)
    const table = spec.law === TABLE_LAW;
    let bad = -1;
    for (let slot = 0; slot < n; slot++) {
      const i = sorted[slot];
      const o = 6 * slot;
      f[3 * i] = outv[o];
      f[3 * i + 1] = outv[o + 1];
      f[3 * i + 2] = outv[o + 2];
      evdwl += outv[o + 3];
      if (table && outv[o + 4] < 0 && bad < 0) bad = i;
      else ecoul += outv[o + 4];
      virial += outv[o + 5];
    }
    b.staging.unmap();
    if (bad >= 0) throw new Error(tableInnerError(spec, state, bad));
    return { pe: evdwl + ecoul, virial, evdwl, ecoul };
  }

  /** True when `advanceSpec` can run this system entirely on the GPU: 3 cells per periodic edge. */
  private eamPipes: { rho: GPUComputePipeline; force: GPUComputePipeline; params: GPUBuffer } | null = null;
  private ebufs: {
    pos: GPUBuffer; out: GPUBuffer; staging: GPUBuffer; ints: GPUBuffer; coef: GPUBuffer;
    atomCap: number; intsCap: number; coefCap: number; bindRho: GPUBindGroup; bindForce: GPUBindGroup; coefKey: string;
  } | null = null;
  private eamRho = new Float64Array(0);
  private eamFp = new Float64Array(0);

  /**
   * One force evaluation of pair eam, eam/alloy or eam/fs (gpu/eamSpec.ts, gpu/eamKernels.ts): the densities on the GPU,
   * read back; embed(rho, fp) (the style's own fp64 embedding step, atoms in state order) returns the embedding energy
   * and fills F'; F' goes up as f32 bits after the cell data; then the forces, pair energies and virial on the GPU.
   * Two round trips per call; the sums are fp64 here.
   */
  async computeEAM(state: SimState, spec: GpuEamSpec, embed: (rho: Float64Array, fp: Float64Array) => number): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    return validated(this.device, () => this.computeEAMUnchecked(state, spec, embed));
  }

  private async computeEAMUnchecked(state: SimState, spec: GpuEamSpec, embed: (rho: Float64Array, fp: Float64Array) => number): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    if (this.disposed) throw new Error('the WebGPU backend was disposed');
    const n = state.n;
    if (n === 0) return { pe: 0, virial: 0, evdwl: 0, ecoul: 0 };
    // the minimum image is the only copy of a partner within the cutoff while each edge is at least 2 cutoffs
    const flat = state.dimension === 2;
    if ([0, 1, 2].some((d) => !(d === 2 && flat) && state.box.hi[d] - state.box.lo[d] < 2 * spec.maxCutoff)) {
      throw new Error('the box has fewer than 2 cutoffs in a periodic dimension');
    }
    const cells = this.sortByCell(state, spec.maxCutoff, n, 1);
    if (!cells) throw new Error('the box has no cell in a periodic dimension');
    const { L, nc, base, two } = cells;
    const { sorted, posData } = this;
    const ints = this.intsData;
    const fpBase = base + n, intsLen = fpBase + n;
    const dev = this.device;
    if (!this.eamPipes) {
      const pl = await checkedPipelines(dev, KERNEL_EAM, ['rho', 'force'] as const);
      this.eamPipes = {
        rho: pl.rho,
        force: pl.force,
        params: dev.createBuffer({ size: 64, usage: BUF_UNIFORM | BUF_COPY_DST }),
      };
    }
    const pipes = this.eamPipes;
    let b = this.ebufs;
    if (!b || b.atomCap < n || b.intsCap < intsLen || b.coefCap < spec.coef.byteLength) {
      if (b) for (const buf of [b.pos, b.out, b.staging, b.ints, b.coef]) buf.destroy();
      const atomCap = Math.max(64, Math.ceil(n * 1.25)), intsCap = Math.max(128, Math.ceil(intsLen * 1.25));
      const coefCap = Math.max(spec.coef.byteLength, b?.coefCap ?? 0);
      const pos = this.buffer(16 * atomCap, BUF_STORAGE | BUF_COPY_DST);
      const out = this.buffer(24 * atomCap, BUF_STORAGE | BUF_COPY_SRC);
      const staging = this.buffer(24 * atomCap, BUF_MAP_READ | BUF_COPY_DST);
      const intsBuf = this.buffer(4 * intsCap, BUF_STORAGE | BUF_COPY_DST);
      const coefBuf = this.buffer(coefCap, BUF_STORAGE | BUF_COPY_DST);
      const entries = [
        { binding: 0, resource: { buffer: pipes.params } },
        { binding: 1, resource: { buffer: pos } },
        { binding: 2, resource: { buffer: intsBuf } },
        { binding: 3, resource: { buffer: coefBuf } },
        { binding: 4, resource: { buffer: out } },
      ];
      b = this.ebufs = {
        pos, out, staging, ints: intsBuf, coef: coefBuf, atomCap, intsCap, coefCap, coefKey: '',
        bindRho: dev.createBindGroup({ layout: pipes.rho.getBindGroupLayout(0), entries }),
        bindForce: dev.createBindGroup({ layout: pipes.force.getBindGroupLayout(0), entries }),
      };
    }
    const params = new ArrayBuffer(64);
    const pf = new Float32Array(params), pu = new Uint32Array(params);
    pf[0] = L[0]; pf[1] = L[1]; pf[2] = L[2]; pu[3] = n;
    pu[4] = nc[0]; pu[5] = nc[1]; pu[6] = nc[2]; pu[7] = spec.stride;
    pu[8] = two ? 1 : 0; pu[9] = base; pu[10] = fpBase; pu[11] = spec.nr;
    pf[12] = spec.rdr;
    const q = dev.queue;
    q.writeBuffer(pipes.params, 0, params);
    q.writeBuffer(b.pos, 0, posData.buffer, posData.byteOffset, 16 * n);
    q.writeBuffer(b.ints, 0, ints.buffer, ints.byteOffset, 4 * fpBase);
    if (b.coefKey !== spec.key) {
      q.writeBuffer(b.coef, 0, spec.coef.buffer, spec.coef.byteOffset, spec.coef.byteLength);
      b.coefKey = spec.key;
    }
    const groups = Math.ceil(n / EAM_WORKGROUP);
    // the densities
    let enc = dev.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(pipes.rho);
    pass.setBindGroup(0, b.bindRho);
    pass.dispatchWorkgroups(groups);
    pass.end();
    enc.copyBufferToBuffer(b.out, 0, b.staging, 0, 4 * n);
    q.submit([enc.finish()]);
    await b.staging.mapAsync(MAP_READ, 0, 4 * n);
    if (this.eamRho.length < n) { this.eamRho = new Float64Array(n); this.eamFp = new Float64Array(n); }
    const rho = this.eamRho.subarray(0, n), fp = this.eamFp.subarray(0, n);
    {
      const r = new Float32Array(b.staging.getMappedRange(0, 4 * n));
      for (let slot = 0; slot < n; slot++) rho[sorted[slot]] = r[slot];
    }
    b.staging.unmap();
    // the embedding step (fp64, the style's own), F' up in slot order
    const eEmbed = embed(rho, fp);
    const fpf = new Float32Array(ints.buffer, ints.byteOffset + 4 * fpBase, n);
    for (let slot = 0; slot < n; slot++) fpf[slot] = fp[sorted[slot]];
    q.writeBuffer(b.ints, 4 * fpBase, ints.buffer, ints.byteOffset + 4 * fpBase, 4 * n);
    // the forces
    enc = dev.createCommandEncoder();
    pass = enc.beginComputePass();
    pass.setPipeline(pipes.force);
    pass.setBindGroup(0, b.bindForce);
    pass.dispatchWorkgroups(groups);
    pass.end();
    enc.copyBufferToBuffer(b.out, 0, b.staging, 0, 24 * n);
    q.submit([enc.finish()]);
    await b.staging.mapAsync(MAP_READ, 0, 24 * n);
    const outv = new Float32Array(b.staging.getMappedRange(0, 24 * n));
    const f = state.f;
    let epair = 0, virial = 0;
    for (let slot = 0; slot < n; slot++) {
      const i = sorted[slot];
      const o = 6 * slot;
      f[3 * i] = outv[o];
      f[3 * i + 1] = outv[o + 1];
      f[3 * i + 2] = outv[o + 2];
      epair += outv[o + 3];
      virial += outv[o + 5];
    }
    b.staging.unmap();
    const evdwl = epair + eEmbed;
    return { pe: evdwl, virial, evdwl, ecoul: 0 };
  }

  private swPipes: { nbr: GPUComputePipeline; force: GPUComputePipeline; params: GPUBuffer } | null = null;
  private swbufs: {
    pos: GPUBuffer; out: GPUBuffer; staging: GPUBuffer; ints: GPUBuffer; coef: GPUBuffer;
    atomCap: number; intsCap: number; coefCap: number; bindNbr: GPUBindGroup; bindForce: GPUBindGroup; coefKey: string;
  } | null = null;
  /** The neighbor-list length of computeSW (grown when an atom has more neighbors). */
  private swMaxn = 32;

  /**
   * One force evaluation of pair sw or sw/mod (gpu/swSpec.ts, gpu/swKernels.ts): the neighbor lists and the gathered
   * forces in one submission, one readback; the sums are fp64 here. When an atom has more neighbors than the lists
   * hold, the lists grow and the evaluation runs again.
   */
  async computeSW(state: SimState, spec: GpuSwSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    if (this.disposed) throw new Error('the WebGPU backend was disposed');
    const n = state.n;
    if (n === 0) return { pe: 0, virial: 0, evdwl: 0, ecoul: 0 };
    const flat = state.dimension === 2;
    if ([0, 1, 2].some((d) => !(d === 2 && flat) && state.box.hi[d] - state.box.lo[d] < 2 * spec.maxCutoff)) {
      throw new Error('the box has fewer than 2 cutoffs in a periodic dimension');
    }
    for (;;) {
      const maxn = this.swMaxn;
      const r = await validated(this.device, () => this.swOnce(state, spec, maxn));
      if (r) return r;
      // a list was cut short: twice as long, and again
      if (maxn >= 4096) throw new Error('more than 4096 neighbors within the sw cutoff of an atom');
      this.swMaxn = 2 * maxn;
    }
  }

  private async swOnce(state: SimState, spec: GpuSwSpec, maxn: number): Promise<(ForceResult & { evdwl: number; ecoul: number }) | null> {
    const n = state.n;
    const cells = this.sortByCell(state, spec.maxCutoff, 0, 1);
    if (!cells) throw new Error('the box has no cell in a periodic dimension');
    const { L, nc, base, two } = cells;
    const { sorted, posData } = this;
    const ints = this.intsData;
    const countBase = base + n, listBase = countBase + n, intsLen = listBase + n * maxn;
    const dev = this.device;
    if (!this.swPipes) {
      const pl = await checkedPipelines(dev, KERNEL_SW, ['nbr', 'force'] as const);
      this.swPipes = { nbr: pl.nbr, force: pl.force, params: dev.createBuffer({ size: 80, usage: BUF_UNIFORM | BUF_COPY_DST }) };
    }
    const pipes = this.swPipes;
    const coefF32 = Float32Array.from(spec.coef);
    let b = this.swbufs;
    if (!b || b.atomCap < n || b.intsCap < intsLen || b.coefCap < coefF32.byteLength) {
      if (b) for (const buf of [b.pos, b.out, b.staging, b.ints, b.coef]) buf.destroy();
      const atomCap = Math.max(64, Math.ceil(n * 1.25)), intsCap = Math.max(128, Math.ceil(intsLen * 1.25));
      const coefCap = Math.max(coefF32.byteLength, b?.coefCap ?? 0);
      const pos = this.buffer(16 * atomCap, BUF_STORAGE | BUF_COPY_DST);
      const out = this.buffer(24 * atomCap, BUF_STORAGE | BUF_COPY_SRC);
      const staging = this.buffer(24 * atomCap, BUF_MAP_READ | BUF_COPY_DST);
      const intsBuf = this.buffer(4 * intsCap, BUF_STORAGE | BUF_COPY_DST);
      const coefBuf = this.buffer(coefCap, BUF_STORAGE | BUF_COPY_DST);
      const entries = [
        { binding: 0, resource: { buffer: pipes.params } },
        { binding: 1, resource: { buffer: pos } },
        { binding: 2, resource: { buffer: intsBuf } },
        { binding: 3, resource: { buffer: coefBuf } },
        { binding: 4, resource: { buffer: out } },
      ];
      b = this.swbufs = {
        pos, out, staging, ints: intsBuf, coef: coefBuf, atomCap, intsCap, coefCap, coefKey: '',
        // the nbr pass does not use outv, so its 'auto' layout has no binding 4 (a bind group with it is invalid)
        bindNbr: dev.createBindGroup({ layout: pipes.nbr.getBindGroupLayout(0), entries: entries.slice(0, 4) }),
        bindForce: dev.createBindGroup({ layout: pipes.force.getBindGroupLayout(0), entries }),
      };
    }
    const params = new ArrayBuffer(80);
    const pf = new Float32Array(params), pu = new Uint32Array(params);
    pf[0] = L[0]; pf[1] = L[1]; pf[2] = L[2]; pu[3] = n;
    pu[4] = nc[0]; pu[5] = nc[1]; pu[6] = nc[2]; pu[7] = spec.stride;
    pu[8] = two ? 1 : 0; pu[9] = base; pu[10] = maxn; pu[11] = countBase;
    pu[12] = listBase; pu[13] = spec.tripBase; pu[14] = spec.threebody ? 1 : 0; pu[15] = spec.mod ? 1 : 0;
    pf[16] = spec.mod?.delta1 ?? 0; pf[17] = spec.mod?.delta2 ?? 0;
    const q = dev.queue;
    q.writeBuffer(pipes.params, 0, params);
    q.writeBuffer(b.pos, 0, posData.buffer, posData.byteOffset, 16 * n);
    q.writeBuffer(b.ints, 0, ints.buffer, ints.byteOffset, 4 * countBase);
    if (b.coefKey !== spec.key) {
      q.writeBuffer(b.coef, 0, coefF32.buffer, coefF32.byteOffset, coefF32.byteLength);
      b.coefKey = spec.key;
    }
    const groups = Math.ceil(n / SW_WORKGROUP);
    const enc = dev.createCommandEncoder();
    for (const [pipe, bind] of [[pipes.nbr, b.bindNbr], [pipes.force, b.bindForce]] as const) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(groups);
      pass.end();
    }
    enc.copyBufferToBuffer(b.out, 0, b.staging, 0, 24 * n);
    q.submit([enc.finish()]);
    await b.staging.mapAsync(MAP_READ, 0, 24 * n);
    const outv = new Float32Array(b.staging.getMappedRange(0, 24 * n));
    let short = false;
    for (let slot = 0; slot < n && !short; slot++) if (outv[6 * slot + 4] < 0) short = true;
    if (short) { b.staging.unmap(); return null; }
    const f = state.f;
    let e = 0, virial = 0;
    for (let slot = 0; slot < n; slot++) {
      const i = sorted[slot];
      const o = 6 * slot;
      f[3 * i] = outv[o];
      f[3 * i + 1] = outv[o + 1];
      f[3 * i + 2] = outv[o + 2];
      e += outv[o + 3];
      virial += outv[o + 5];
    }
    b.staging.unmap();
    return { pe: e, virial, evdwl: e, ecoul: 0 };
  }

  private terPipes: { nbr: GPUComputePipeline; bond: GPUComputePipeline; force: GPUComputePipeline; params: GPUBuffer } | null = null;
  private terbufs: {
    pos: GPUBuffer; out: GPUBuffer; staging: GPUBuffer; ints: GPUBuffer; coef: GPUBuffer; bond: GPUBuffer;
    atomCap: number; intsCap: number; coefCap: number; bondCap: number; binds: [GPUBindGroup, GPUBindGroup, GPUBindGroup]; coefKey: string;
  } | null = null;
  /** The neighbor-list length of computeTersoff (grown when an atom has more neighbors). */
  private terMaxn = 16;

  /**
   * One force evaluation of pair tersoff (gpu/tersoffSpec.ts, gpu/tersoffKernels.ts): the neighbor lists, the bond
   * orders and the gathered forces in one submission, one readback; the sums are fp64 here. When an atom has more
   * neighbors than the lists hold, the lists grow and the evaluation runs again.
   */
  async computeTersoff(state: SimState, spec: GpuTersoffSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    if (this.disposed) throw new Error('the WebGPU backend was disposed');
    const n = state.n;
    if (n === 0) return { pe: 0, virial: 0, evdwl: 0, ecoul: 0 };
    const flat = state.dimension === 2;
    if ([0, 1, 2].some((d) => !(d === 2 && flat) && state.box.hi[d] - state.box.lo[d] < 2 * spec.maxCutoff)) {
      throw new Error('the box has fewer than 2 cutoffs in a periodic dimension');
    }
    for (;;) {
      const maxn = this.terMaxn;
      const r = await validated(this.device, () => this.tersoffOnce(state, spec, maxn));
      if (r) return r;
      // a list was cut short: twice as long, and again
      if (maxn >= 4096) throw new Error('more than 4096 neighbors within the tersoff cutoff of an atom');
      this.terMaxn = 2 * maxn;
    }
  }

  private async tersoffOnce(state: SimState, spec: GpuTersoffSpec, maxn: number): Promise<(ForceResult & { evdwl: number; ecoul: number }) | null> {
    const n = state.n;
    const cells = this.sortByCell(state, spec.maxCutoff, 0, 1);
    if (!cells) throw new Error('the box has no cell in a periodic dimension');
    const { L, nc, base, two } = cells;
    const { sorted, posData } = this;
    const ints = this.intsData;
    const countBase = base + n, listBase = countBase + n, intsLen = listBase + n * maxn;
    const dev = this.device;
    if (!this.terPipes) {
      const pl = await checkedPipelines(dev, KERNEL_TERSOFF, ['nbr', 'bond', 'force'] as const);
      this.terPipes = { ...pl, params: dev.createBuffer({ size: 64, usage: BUF_UNIFORM | BUF_COPY_DST }) };
    }
    const pipes = this.terPipes;
    const coefF32 = Float32Array.from(spec.coef);
    let b = this.terbufs;
    if (!b || b.atomCap < n || b.intsCap < intsLen || b.coefCap < coefF32.byteLength || b.bondCap < n * maxn) {
      if (b) for (const buf of [b.pos, b.out, b.staging, b.ints, b.coef, b.bond]) buf.destroy();
      const atomCap = Math.max(64, Math.ceil(n * 1.25)), intsCap = Math.max(128, Math.ceil(intsLen * 1.25));
      const coefCap = Math.max(coefF32.byteLength, b?.coefCap ?? 0), bondCap = Math.max(64, Math.ceil(n * maxn * 1.25));
      const pos = this.buffer(16 * atomCap, BUF_STORAGE | BUF_COPY_DST);
      const out = this.buffer(24 * atomCap, BUF_STORAGE | BUF_COPY_SRC);
      const staging = this.buffer(24 * atomCap, BUF_MAP_READ | BUF_COPY_DST);
      const intsBuf = this.buffer(4 * intsCap, BUF_STORAGE | BUF_COPY_DST);
      const coefBuf = this.buffer(coefCap, BUF_STORAGE | BUF_COPY_DST);
      const bond = this.buffer(8 * bondCap, BUF_STORAGE);
      const entries = [
        { binding: 0, resource: { buffer: pipes.params } },
        { binding: 1, resource: { buffer: pos } },
        { binding: 2, resource: { buffer: intsBuf } },
        { binding: 3, resource: { buffer: coefBuf } },
        { binding: 4, resource: { buffer: bond } },
        { binding: 5, resource: { buffer: out } },
      ];
      // each entry point's 'auto' layout has only the bindings it uses (a bind group with more is invalid)
      const bindFor = (pipe: GPUComputePipeline, k: number) => dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: entries.slice(0, k) });
      b = this.terbufs = {
        pos, out, staging, ints: intsBuf, coef: coefBuf, bond, atomCap, intsCap, coefCap, bondCap, coefKey: '',
        binds: [bindFor(pipes.nbr, 4), bindFor(pipes.bond, 5), bindFor(pipes.force, 6)],
      };
    }
    const params = new ArrayBuffer(64);
    const pf = new Float32Array(params), pu = new Uint32Array(params);
    pf[0] = L[0]; pf[1] = L[1]; pf[2] = L[2]; pu[3] = n;
    pu[4] = nc[0]; pu[5] = nc[1]; pu[6] = nc[2]; pu[7] = spec.stride;
    pu[8] = two ? 1 : 0; pu[9] = base; pu[10] = maxn; pu[11] = countBase;
    pu[12] = listBase; pu[13] = spec.tripBase; pf[14] = spec.shift;
    const q = dev.queue;
    q.writeBuffer(pipes.params, 0, params);
    q.writeBuffer(b.pos, 0, posData.buffer, posData.byteOffset, 16 * n);
    q.writeBuffer(b.ints, 0, ints.buffer, ints.byteOffset, 4 * countBase);
    if (b.coefKey !== spec.key) {
      q.writeBuffer(b.coef, 0, coefF32.buffer, coefF32.byteOffset, coefF32.byteLength);
      b.coefKey = spec.key;
    }
    const groups = Math.ceil(n / TERSOFF_WORKGROUP);
    const enc = dev.createCommandEncoder();
    for (const [pipe, bind] of [[pipes.nbr, b.binds[0]], [pipes.bond, b.binds[1]], [pipes.force, b.binds[2]]] as const) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(groups);
      pass.end();
    }
    enc.copyBufferToBuffer(b.out, 0, b.staging, 0, 24 * n);
    q.submit([enc.finish()]);
    await b.staging.mapAsync(MAP_READ, 0, 24 * n);
    const outv = new Float32Array(b.staging.getMappedRange(0, 24 * n));
    let short = false;
    for (let slot = 0; slot < n && !short; slot++) if (outv[6 * slot + 4] < 0) short = true;
    if (short) { b.staging.unmap(); return null; }
    const f = state.f;
    let e = 0, virial = 0;
    for (let slot = 0; slot < n; slot++) {
      const i = sorted[slot];
      const o = 6 * slot;
      f[3 * i] = outv[o];
      f[3 * i + 1] = outv[o + 1];
      f[3 * i + 2] = outv[o + 2];
      e += outv[o + 3];
      virial += outv[o + 5];
    }
    b.staging.unmap();
    return { pe: e, virial, evdwl: e, ecoul: 0 };
  }

  canAdvanceSpec(state: SimState, spec: GpuPairSpec): boolean {
    // the whole-step kernel has the analytic laws only (styleLaws.ts), not pair_style table's tables
    if (state.n === 0 || !(spec.maxCutoff > 0) || spec.law === TABLE_LAW) return false;
    const two = state.dimension === 2;
    return [0, 1, 2].every((d) => (d === 2 && two)
      || (state.box.periodic[d] && Math.floor((state.box.hi[d] - state.box.lo[d]) / spec.maxCutoff) >= 3));
  }

  /** `advance` for a style of gpu/pairSpec.ts (the charges of a charged one go up with the coefficients). */
  async advanceSpec(state: SimState, spec: GpuPairSpec, nsteps: number, opts: { enforce2d: boolean }): Promise<ForceResult> {
    if (this.disposed) throw new Error('the WebGPU backend was disposed');
    if (!this.specCache || this.specCache.key !== spec.key) this.specCache = { key: spec.key, coef: padToVec4(spec.coef) };
    const two = state.dimension === 2;
    const nc = [0, 1, 2].map((d) => (d === 2 && two ? 1 : Math.floor((state.box.hi[d] - state.box.lo[d]) / spec.maxCutoff))) as [number, number, number];
    this.resident ??= ResidentStepper.create(this.device);
    const charges = spec.charges ? { q: state.q, qqrd2e: spec.qqrd2e } : null;
    return (await this.resident).advance(state, nsteps, { nc, stride: spec.stride, coef: this.specCache.coef, style: spec.law, charges, enforce2d: opts.enforce2d });
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
    return (await this.resident).advance(state, nsteps, { nc, stride: arrays.stride, coef, style: 0, enforce2d: opts.enforce2d });
  }

  dispose(): void {
    if (this.disposed) return;
    this.resident?.then((r) => r.dispose(), () => undefined);
    this.disposed = true;
    const b = this.bufs;
    if (b) for (const buf of [b.pos, b.out, b.staging, b.ints, b.coef]) buf.destroy();
    const sb = this.sbufs;
    if (sb) for (const buf of [sb.pos, sb.out, sb.staging, sb.ints, sb.coef]) buf.destroy();
    const eb = this.ebufs;
    if (eb) for (const buf of [eb.pos, eb.out, eb.staging, eb.ints, eb.coef]) buf.destroy();
    const wb = this.swbufs;
    if (wb) for (const buf of [wb.pos, wb.out, wb.staging, wb.ints, wb.coef]) buf.destroy();
    this.swPipes?.params.destroy();
    const tb = this.terbufs;
    if (tb) for (const buf of [tb.pos, tb.out, tb.staging, tb.ints, tb.coef, tb.bond]) buf.destroy();
    this.terPipes?.params.destroy();
    this.eamPipes?.params.destroy();
    this.styleParams?.destroy();
    this.params.destroy();
    this.bufs = null;
    this.cpu.dispose();
    this.device.destroy();
  }
}
