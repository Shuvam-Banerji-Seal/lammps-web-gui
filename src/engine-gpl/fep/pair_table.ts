/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/fix_adapt_fep.cpp and src/FEP/compute_fep.cpp (pair lookup through pair->extract(), and the
 * type-pair ranges of utils::bounds_typelabel) at commit b332f28, by Agilio Padua (ENS de Lyon & CNRS) and the
 * LAMMPS developers. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Shared helpers of fix adapt/fep and compute fep: the pair parameter a "pair pstyle param I J" argument names
 * (native's pair_match and extract(), with fix adapt's accessors), the type-pair list of an I J range, and the
 * energy bookkeeping of compute fep.
 */
import { Pair, StyleError, typeBounds, type AdaptAccess } from '../../engine/force/types';
import { PAIR_EXTRACT } from '../../engine/fix/adaptPairsTable';
import { ACCEL_ALIASES } from '../../engine/accelAliases';
import { pairAdaptAccess } from '../../engine/fix/adaptPairs';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import type { System } from '../../engine/system';

/** The pair parameter a "pair pstyle pparam I J" argument of fix adapt/fep or compute fep names. */
export interface PairTable {
  /** The (sub-)style whose extract() gives the parameter. */
  style: Pair;
  /** The defined pair style; its reinit() re-derives the coefficients (native force->pair->reinit()). */
  owner: Pair;
  /** Native extract()'s per-type-pair array of the parameter (fix/adaptPairs.ts). */
  access: Extract<AdaptAccess, { dim: 2 }>;
}

/** The FEP command a lookup is for: its messages differ. */
export type FepCommand = 'fix adapt/fep' | 'compute fep';

/**
 * The parameter, as native LAMMPS does (src/FEP/fix_adapt_fep.cpp and compute_fep.cpp init at b332f28):
 * Force::pair_match(pstyle, 1), the defined style of that exact name or else the only hybrid sub-style of that name
 * (no style:N form), its extract() name (fix/adaptPairsTable.ts) and dim (fix adapt/fep takes dim 2 only; compute fep
 * does not check it and would write a global value as an array, which this engine refuses), and with a hybrid style
 * the sub-style of every type pair of the range.
 */
export const resolvePairTable = (sys: System, cmd: FepCommand, pstyle: string, pparam: string, pairs: Array<[number, number]>): PairTable => {
  const owner = sys.ff.pair;
  const fix = cmd === 'fix adapt/fep';
  if (!owner && !fix) throw new StyleError('compute fep pair requires pair interactions');
  // an accelerator spelling (lj/cut/omp) names its base style, as the engine runs it
  const word = ACCEL_ALIASES.get(pstyle) ?? pstyle;
  let style: Pair | null = null;
  if (owner && owner.name === word) style = owner;
  else if (owner instanceof PairHybrid) {
    const subs = owner.subs.filter((x) => x.name === word);
    if (subs.length === 1) style = subs[0].style;
  }
  if (!style || !owner) throw new StyleError(fix ? 'Fix adapt/fep pair style does not exist' : `Compute fep pair style ${pstyle} not found`);
  const entry = (PAIR_EXTRACT[style.name] ?? []).find((e) => e[0] === pparam);
  if (!entry) throw new StyleError(fix ? 'Fix adapt/fep pair style param not supported' : `Compute fep pair style ${pstyle} param ${pparam} not supported`);
  if (entry[1] !== 2) {
    throw new StyleError(fix
      ? `Pair style parameter ${pparam} is not compatible with fix adapt/fep`
      : `compute fep: pair style ${pstyle} param ${pparam} is a global value (dim ${entry[1]}), which native compute fep writes as a per-type-pair array (undefined behaviour); not supported by the browser engine`);
  }
  if (owner instanceof PairHybrid) {
    for (const [i, j] of pairs) {
      if (!owner.subsOf(i, j).some((k) => owner.subs[k].name === word)) {
        throw new StyleError(`${fix ? 'Fix adapt/fep' : 'compute fep'} type pair range is not valid for pair hybrid sub-style`);
      }
    }
  }
  const access = pairAdaptAccess(style, style.name, pstyle, pparam, fix ? 'Fix adapt/fep' : 'Compute fep');
  if (access.dim !== 2) throw new StyleError(`${cmd}: pair style ${pstyle} param ${pparam} is not a per-type-pair array`);
  return { style, owner, access };
};

/**
 * The type pairs (I <= J) of an I J range. "*" and "n*m" ranges as in typeBounds. A range with one
 * type on each side is swapped when I > J (fix adapt/fep does this; compute fep does not).
 */
export const typePairs = (iw: string, jw: string, ntypes: number, swap: boolean): Array<[number, number]> => {
  let [ilo, ihi] = typeBounds(iw, ntypes);
  let [jlo, jhi] = typeBounds(jw, ntypes);
  if (swap && ilo === ihi && jlo === jhi && ilo > jlo) [ilo, ihi, jlo, jhi] = [jlo, jhi, ilo, ihi];
  const out: Array<[number, number]> = [];
  for (let i = ilo; i <= ihi; i++) for (let j = Math.max(jlo, i); j <= jhi; j++) out.push([i, j]);
  return out;
};

/** Writes one value to the [i][j] entry of the parameter (native writes j >= i; init_one mirrors where it does). */
export const setPairValue = (t: PairTable, i: number, j: number, value: number): void => {
  t.access.set(i, j, value);
};

/** pair_modify tail yes on the style or a hybrid sub-style (fix adapt and compute fep check it). */
export const hasTail = (p: Pair): boolean => {
  if (p.tail) return true;
  return p instanceof PairHybrid && p.subs.some((x) => x.style.tail);
};

/**
 * After a value changed: native force->pair->reinit() (src/pair.cpp Pair::reinit at b332f28: "Fix adapt interface to
 * pair style X is not supported" without reinitflag, else init_one for every type pair, the run's cutoffs kept), the
 * tail summed again.
 */
export const reinitPair = (sys: System, owner: Pair): void => {
  if (!owner.reinitFlag) throw new StyleError(`Fix adapt interface to pair style ${owner.name} is not supported`);
  owner.reinit();
  if (hasTail(owner)) sys.ff.reinitTail(sys.state);
  owner.version++;
  sys.ff.pairStale = true;
};

/**
 * Re-derives the kspace charge sums after the charges changed (native kspace->qsum_qsq()). The engine's kspace
 * styles keep qsum and qsqsum from init(), and their constant energy term (the self energy) reads those, so a
 * changed charge would otherwise leave the k-space energy at its old value.
 */
export const refreshChargeSums = (sys: System): void => {
  sys.nb.refreshCharges(sys.state);
  const k = sys.ff.kspace as { qsum?: number; qsqsum?: number } | null;
  if (!k || !('qsqsum' in k)) return;
  const s = sys.state;
  let qs = 0, q2 = 0;
  for (let i = 0; i < s.n; i++) { qs += s.q[i]; q2 += s.q[i] * s.q[i]; }
  k.qsum = qs;
  k.qsqsum = q2;
};
