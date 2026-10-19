/*
 * The contract of the ACE descriptor of pair_style mliap (docs.lammps.org/pair_mliap.html: "*ace* indicates the
 * atomic cluster expansion (ACE) descriptors. A single additional argument specifies the filename containing
 * parameters, settings, and definitions of the ace descriptors ... in the ctilde file format, e.g. in the potential
 * file format with `*.ace` or `*.yace` extensions from pair_style pace"). The implementation is ported from the
 * ML-PACE code and lives in src/engine-gpl (registered as the lazily loaded HOOKS entry 'ace'); the core reads it
 * with gplHook('ace').
 */
import type { PairCompute } from '../force/types';
import type { NeighList } from '../neighbor';
import type { Compute } from '../compute/compute';
import type { System } from '../system';

export interface MliapAceDescriptor {
  /** The element names of the file, in species order. */
  readonly elements: readonly string[];
  /** The number of descriptors (the largest of the species). */
  readonly ndescriptors: number;
  /** The squared neighbour cutoff of the element pair (ei, ej). */
  cutsq(ei: number, ej: number): number;
  /**
   * The descriptors of atom i and their gradients for its neighbours jlist[0..jnum) (indices into x and type, which
   * hold owned and ghost atoms): B[k], and dB[(k * jnum + jj) * 3 + d] = dB_k/dr_j along d.
   */
  evaluate(i: number, x: Float64Array, type: Int32Array, jnum: number, jlist: Int32Array): { B: Float64Array; dB: Float64Array };
  /** The full list's entries with each owned atom's row in the order native LAMMPS builds it (sums follow it). */
  nativeNeighbors(pc: PairCompute, full: NeighList): Int32Array;
}

export interface MliapAceHook {
  load(text: string, filename: string): MliapAceDescriptor;
  /** compute ID group mliap ... descriptor ace file ... (the arguments after the style name). */
  compute(sys: System, id: string, group: string, args: string[]): Compute;
}
