/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/pair.cpp (Pair::ev_tally, Pair::ev_tally3, Pair::v_tally2_newton, Pair::v_tally3, Pair::v_tally4) at
 * commit b332f28, as used by the ReaxFF tally calls of src/REAXFF/reaxff_*.cpp. Only the per-atom parts are ported
 * here (pc.eatom and pc.vatom); the global energies and virial stay with the force terms. Authors credited in the
 * C++ file headers: Hasan Metin Aktulga (Purdue University, now Lawrence Berkeley National Laboratory) and the LAMMPS
 * developers; Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * Every helper works on the owned+ghost index space of the ReaxFF evaluation (the engine folds ghost entries into
 * their owners afterwards). The virial is 6 per atom in the order xx yy zz xy xz yz.
 */

/** pc.eatom (per-atom energy) and pc.vatom (6 per atom), or null when not requested. */
export interface PerAtomTally {
  eatom: Float64Array | null;
  vatom: Float64Array | null;
}

/** A 3-vector (a relative position or a force). */
type V3 = ArrayLike<number>;

const THIRD = 1.0 / 3.0;

/** The six components (xx yy zz xy xz yz) of the sum of del_a (x) f_a over the given pairs, as the C++ tallies compute them. */
const outer = (terms: Array<[V3, V3]>): number[] => {
  const v = [0, 0, 0, 0, 0, 0];
  for (const [d, f] of terms) {
    v[0] += d[0] * f[0];
    v[1] += d[1] * f[1];
    v[2] += d[2] * f[2];
    v[3] += d[0] * f[1];
    v[4] += d[0] * f[2];
    v[5] += d[1] * f[2];
  }
  return v;
};

/** Adds weight * v to the virial of each listed atom. */
const addVirial = (va: Float64Array, atoms: number[], v: number[], weight: number): void => {
  for (const a of atoms) {
    for (let c = 0; c < 6; c++) va[6 * a + c] += weight * v[c];
  }
};

/**
 * Pair::ev_tally with newton on, energy part only (the bond terms of reaxff_bonds.cpp): half of the energy to i and
 * half to j, so i == j gives the whole energy to that atom.
 */
export const tallyEnergy = (pa: PerAtomTally, i: number, j: number, e: number): void => {
  if (pa.eatom) {
    const half = 0.5 * e;
    pa.eatom[i] += half;
    pa.eatom[j] += half;
  }
};

/**
 * Pair::ev_tally with newton on, per-atom part: half of the energy (evdwl + ecoul) and half of the virial
 * del (x) del * fpair (del = (dx, dy, dz)) to each of i and j.
 */
export const tallyPair = (
  pa: PerAtomTally, i: number, j: number, evdwl: number, ecoul: number, fpair: number, dx: number, dy: number, dz: number,
): void => {
  if (pa.eatom) {
    const e = 0.5 * (evdwl + ecoul);
    pa.eatom[i] += e;
    pa.eatom[j] += e;
  }
  if (pa.vatom) {
    const v = [dx * dx * fpair, dy * dy * fpair, dz * dz * fpair, dx * dy * fpair, dx * dz * fpair, dy * dz * fpair];
    addVirial(pa.vatom, [i, j], v, 0.5);
  }
};

/** Pair::ev_tally3 (reaxff_hydrogen_bonds.cpp): a third of the energy and of drji (x) fj + drki (x) fk to each of i, j, k. */
export const tallyEnergy3 = (
  pa: PerAtomTally, i: number, j: number, k: number, evdwl: number, ecoul: number, fj: V3, fk: V3, drji: V3, drki: V3,
): void => {
  if (pa.eatom) {
    const e = THIRD * (evdwl + ecoul);
    pa.eatom[i] += e;
    pa.eatom[j] += e;
    pa.eatom[k] += e;
  }
  if (pa.vatom) addVirial(pa.vatom, [i, j, k], outer([[drji, fj], [drki, fk]]), THIRD);
};

/** Pair::v_tally2_newton: the whole virial deli (x) fi to atom i. */
export const tallyVirial2 = (pa: PerAtomTally, i: number, fi: V3, deli: V3): void => {
  if (pa.vatom) addVirial(pa.vatom, [i], outer([[deli, fi]]), 1);
};

/** Pair::v_tally3: a third of drik (x) fi + drjk (x) fj to each of i, j, k. */
export const tallyVirial3 = (pa: PerAtomTally, i: number, j: number, k: number, fi: V3, fj: V3, drik: V3, drjk: V3): void => {
  if (pa.vatom) addVirial(pa.vatom, [i, j, k], outer([[drik, fi], [drjk, fj]]), THIRD);
};

/** Pair::v_tally4: a quarter of drim (x) fi + drjm (x) fj + drkm (x) fk to each of i, j, k, m. */
export const tallyVirial4 = (
  pa: PerAtomTally, i: number, j: number, k: number, m: number, fi: V3, fj: V3, fk: V3, drim: V3, drjm: V3, drkm: V3,
): void => {
  if (pa.vatom) addVirial(pa.vatom, [i, j, k, m], outer([[drim, fi], [drjm, fj], [drkm, fk]]), 0.25);
};

/** The relative position x_a - x_b of two atoms of the flat position array x (3 per atom). */
export const relPos = (x: ArrayLike<number>, a: number, b: number): number[] => [
  x[3 * a] - x[3 * b], x[3 * a + 1] - x[3 * b + 1], x[3 * a + 2] - x[3 * b + 2],
];
