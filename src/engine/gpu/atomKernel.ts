import type { ForceResult, SimState } from '../types';
import { WGSL_BOX_IMAGE } from './cellSort';

/*
 * A pair style's own one-pass GPU kernel, for a style whose forces need per-atom inputs beside the positions (pair
 * eff/cut: charge, electron spin and radius). The style's lazy module builds it (a gplHook under the style's name, see
 * run/accel.ts), webgpuForces.ts computeAtomKernel runs it:
 *   force: each atom's partners from the cells around its own (cellSort.ts, every cell once along an edge of fewer than
 *          3 cells), its own force and per-atom outputs, ATOM_KERNEL_OUT values per atom in outv (by cell slot; the
 *          backend hands them to finish() in atom order).
 * The kernel reads WGSL_ATOM_KERNEL_HEAD's bindings: P (the box, the cell grid and eight style scalars), pos (x y z
 * type, from the box origin), ints (the cell starts, then each slot's cell at P.cellOfBase), coef (the style's
 * parameters), ext (four per-atom values, extra()) and outv. Every binding must be used by the entry point (an
 * 'auto' layout leaves an unused one out, and the bind group is then invalid).
 */
export const ATOM_KERNEL_WORKGROUP = 64;
export const ATOM_KERNEL_OUT = 16;

export interface GpuAtomKernel {
  /** The pair style (the gplHook word). */
  readonly name: string;
  /** WGSL source with the entry point force, starting with WGSL_ATOM_KERNEL_HEAD. */
  readonly code: string;
  /** Identity of coef (uploaded again when it changes). */
  readonly key: string;
  readonly coef: Float64Array;
  /** The type-pair table stride of coef (ntypes + 1), P.stride. */
  readonly stride: number;
  /** The largest pair cutoff: the cell size and the minimum-image check. */
  readonly maxCutoff: number;
  /** Eight style scalars of this evaluation (P.u0, P.u1). */
  scalars(state: SimState): number[];
  /** Four values per atom, atom order (ext). */
  extra(state: SimState, out: Float32Array): void;
  /** The forces, per-atom outputs, energies and virial from ATOM_KERNEL_OUT values per atom (atom order). */
  finish(state: SimState, out: ArrayLike<number>): ForceResult & { evdwl: number; ecoul: number };
  /** The kernel's arithmetic in fp64 (ATOM_KERNEL_OUT values per atom, atom order): the tests' stand-in for the GPU. */
  fp64(state: SimState): Float64Array;
}

/** Pair styles that build a GpuAtomKernel (a gplHook named by the style with a gpuKernel(pair, sys) helper). */
export const GPU_ATOM_KERNEL_STYLES: readonly string[] = ['eff/cut'];

/** The uniform of an atom kernel: 96 bytes (webgpuForces.ts computeAtomKernel writes it). */
export const WGSL_ATOM_KERNEL_HEAD = /* wgsl */ `
struct Params {
  L : vec3<f32>,
  n : u32,
  nc : vec3<u32>,
  stride : u32,
  two : u32,
  cellOfBase : u32,
  pad0 : u32,
  pad1 : u32,
  tilt : vec4<f32>,  // xy xz yz: the minimum image of a triclinic box (cellSort.ts WGSL_BOX_IMAGE)
  u0 : vec4<f32>,
  u1 : vec4<f32>,
};

${WGSL_BOX_IMAGE}

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ints : array<u32>;
@group(0) @binding(3) var<storage, read> coef : array<f32>;
@group(0) @binding(4) var<storage, read> ext : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> outv : array<f32>;

fn wrapCell(c : i32, n : u32) -> u32 {
  let m = i32(n);
  return u32(((c % m) + m) % m);
}

// b - a by the minimum image (the identity along a non-periodic edge, cellSort.ts)
fn minImage(a : vec3<f32>, b : vec3<f32>) -> vec3<f32> {
  var d = b - a;
  d = boxImage(d);
  if (P.two == 1u) { d.z = 0.0; }
  return d;
}
`;
