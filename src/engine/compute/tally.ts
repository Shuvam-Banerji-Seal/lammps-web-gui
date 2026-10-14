import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * Pairwise group computes (docs.lammps.org/compute_group_group.html and
 * docs.lammps.org/compute_tally.html): compute group/group, pe/tally,
 * force/tally and stress/tally. All four visit the pairs of the half neighbour
 * list that have one atom in group1 and the other in group2, and use the pair
 * style's single() for the energy and force of each pair. Bonded, kspace and
 * many-body terms are not part of them.
 *
 * Quotes used:
 *   compute_group_group.html: "Define a computation that calculates the total energy and force
 *   interaction between two groups of atoms: the compute group and the specified group2."
 *   "The two groups can be the same."
 *   "If the *pair* keyword is set to *yes*, which is the default, then the the interaction energy
 *   will include a pair component which is defined as the pairwise energy between all pairs of
 *   atoms where one atom in the pair is in the first group and the other is in the second group."
 *   "For the *inter* setting, the two atoms must be in different molecules."
 *   "For the *intra* setting, the two atoms must be in the same molecule."
 *   "This compute calculates a global scalar (the energy) and a global vector of length 3 (force),
 *   which can be accessed by indices 1--3."
 *   "The option defaults are pair = yes, kspace = no, boundary = yes, molecule = off."
 *   "If the *kspace* keyword is set to *yes*, which is not the default, and if a
 *   :doc:`kspace_style <kspace_style>` is defined, then the interaction energy will include a Kspace
 *   component which is the long-range Coulombic energy between all the atoms in the first group and
 *   all the atoms in the second group."
 *   compute_tally.html: "Define a computation that calculates properties between two groups of atoms by
 *   accumulating them from pairwise non-bonded computations."
 *   "Compute *pe/tally* calculates a global scalar (the energy) and a per atom scalar (the contributions
 *   of the single atom to the global scalar)."
 *   "Compute *force/tally* calculates a global scalar (the force magnitude) and a per atom 3-element
 *   vector (force contribution from each atom)."
 *   "Compute *stress/tally* calculates a global scalar (average of the diagonal elements of the stress
 *   tensor) and a per atom vector (the six elements of stress tensor contributions from the individual
 *   atom)."
 *   "The pairwise contributions are computing via a callback that the compute registers with the
 *   non-bonded pairwise force computation."
 *
 * Measured with native LAMMPS (black box, 4 atoms, lj/cut/coul/cut, units lj, which print thermo
 * values normalised by the atom count; the dumps are not normalised):
 *   - a pair (i, j) counts once, whatever the two groups are: group1 = group2 = one atom gives 0, and
 *     compute group/group of a group with itself counts each unordered pair once;
 *   - the scalar of pe/tally, group/group and force/tally sums the counted pairs once: pe/tally with both
 *     groups "all" equals the total pair energy;
 *   - pe/tally per-atom values: each atom of a counted pair gets half of the pair energy, in two columns
 *     (evdwl, ecoul); summed over all atoms that is the scalar;
 *   - force/tally per-atom values: atom i gets the force on it from j and atom j the opposite force; the
 *     scalar is the magnitude of the sum over the atoms of group1, which is the group/group vector;
 *   - stress/tally per-atom values: each atom of a counted pair gets -1/2 of the pair virial (the outer
 *     product d (x) F, d = x_i - x_j, F the force on i); the scalar is the pair virial trace divided by 3
 *     and is not divided by the atom count in thermo output (unlike the energy and force scalars).
 *   - group/group with lj/cut/coul/cut and special bonds: the special factors enter through the neighbour
 *     list, as for the pair energy.
 * The per-atom outputs are listed by compute ID in the engine's per-atom convention (length n, indexed by
 * the local atom index); ghosts add to their owner.
 *
 * Not implemented (refused by name, or not registered): kspace yes for group/group (the long-range
 * group-group term is not in this engine), molecule keyword for the tally computes, pe/mol/tally,
 * heat/flux/tally and heat/flux/virial/tally.
 */

export type MolMode = 'off' | 'inter' | 'intra';

const NEIGH_MASK = (1 << 30) - 1;

/**
 * Calls cb once per pair counted by a tally compute: a pair of the half list with one atom in
 * bit1 and the other in bit2 (in either order), filtered by the molecule mode. cb gets the
 * local indices, the pair energy split into evdwl and ecoul, the force-over-r of the pair
 * (force on i = fforce * d) and d = x_i - x_j (minimum image through the neighbour list).
 */
export const walkTallyPairs = (
  sys: System,
  owner: string,
  bit1: number,
  bit2: number,
  mol: MolMode,
  cb: (i: number, j: number, evdwl: number, ecoul: number, fforce: number, dx: number, dy: number, dz: number) => void,
): void => {
  const pair = sys.ff.pair;
  if (!pair) throw new StyleError(`compute ${owner} needs a pair_style (none is defined)`);
  if (!pair.single || pair.singleNeedsSeparation()) throw new StyleError(`compute ${owner} needs a pair_style that supports single() (pairwise evaluation); pair_style ${pair.name} does not`);
  sys.forces();
  const s = sys.state;
  const nb = sys.nb;
  const list = nb.half;
  if (!list) throw new StyleError(`compute ${owner}: no half neighbour list is available`);
  // special-bond weights of the last force evaluation (index = special bits of the neighbour)
  const ff = sys.ff as unknown as { specialLJ: Float64Array; specialCoul: Float64Array };
  const sLJ = ff.specialLJ, sC = ff.specialCoul;
  const nt = pair.ntypes + 1;
  const nbr = list.neighbors;
  const xa = nb.xall;
  for (let i = 0; i < list.inum; i++) {
    const mi = s.mask[i];
    const k0: number = list.firstneigh[i], k1: number = k0 + list.numneigh[i];
    for (let k: number = k0; k < k1; k++) {
      const jj: number = nbr[k];
      const j: number = jj & NEIGH_MASK;
      const sb = jj >>> 30;
      const oj = nb.owner[j];
      const mj = s.mask[oj];
      const counted = ((mi & bit1) !== 0 && (mj & bit2) !== 0) || ((mj & bit1) !== 0 && (mi & bit2) !== 0);
      if (!counted) continue;
      if (mol !== 'off') {
        const same = s.molecule[i] === s.molecule[oj];
        if (mol === 'inter' && same) continue;
        if (mol === 'intra' && !same) continue;
      }
      const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const ti = nb.typeall[i], tj = nb.typeall[j];
      if (rsq >= pair.cutsq[ti * nt + tj]) continue;
      const fc = sC[sb], fl = sLJ[sb];
      const qi = nb.qall[i], qj = nb.qall[j];
      const oi = nb.owner[i];
      const tot = pair.single(oi, oj, ti, tj, rsq, fc, fl, qi, qj);
      // Coulomb part alone (the LJ factor 0 switches the LJ term off); the rest is the LJ part
      const ecoul = pair.single(oi, oj, ti, tj, rsq, fc, 0, qi, qj).eng;
      cb(i, j, tot.eng - ecoul, ecoul, tot.fforce, dx, dy, dz);
    }
  }
};

/** Per-atom tally values are indexed by the local atom index; a ghost neighbour adds to its owner. */
const atomIndex = (sys: System, j: number): number => sys.nb.owner[j];

/** compute group/group group-ID group2-ID [pair yes|no] [kspace yes|no] [boundary yes|no] [molecule off|inter|intra]. */
export class ComputeGroupGroup extends Compute {
  readonly style = 'group/group';
  private readonly group2Bit: number;
  private readonly pairOn: boolean;
  private readonly molMode: MolMode;
  private cacheEpoch = -1;
  private energy = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError(`compute ${id} group/group needs the group2-ID`);
    this.group2Bit = sys.groups.bit(args[0]);
    let pairOn = true, mol: MolMode = 'off';
    for (let a = 1; a < args.length; a += 2) {
      const key = args[a], val = args[a + 1];
      if (val === undefined) throw new StyleError(`compute ${id} group/group: keyword ${key} needs a value`);
      if (key === 'pair' || key === 'kspace' || key === 'boundary') {
        if (val !== 'yes' && val !== 'no') throw new StyleError(`compute ${id} group/group: ${key} must be yes or no`);
        if (key === 'pair') pairOn = val === 'yes';
        if (key === 'kspace' && val === 'yes') {
          throw new StyleError(`compute ${id} group/group: kspace yes is not supported by the browser engine (the long-range group-group term is not implemented)`);
        }
      } else if (key === 'molecule') {
        if (val !== 'off' && val !== 'inter' && val !== 'intra') throw new StyleError(`compute ${id} group/group: molecule must be off, inter or intra`);
        mol = val;
      } else {
        throw new StyleError(`compute ${id} group/group: unknown keyword '${key}'`);
      }
    }
    this.pairOn = pairOn;
    this.molMode = mol;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.vector = new Float64Array(3);
  }

  /** One pass over the counted pairs gives the energy (scalar) and the force on group1 (vector). */
  private evaluate(): void {
    if (this.cacheEpoch === this.sys.epoch) return;
    const f = new Float64Array(3);
    let e = 0;
    if (this.pairOn) {
      const s = this.sys.state;
      walkTallyPairs(this.sys, this.id, this.groupBit, this.group2Bit, this.molMode, (i, j, evdwl, ecoul, fforce, dx, dy, dz) => {
        e += evdwl + ecoul;
        // the force on the group1 atoms of the pair (both when both atoms are in group1)
        const fx = fforce * dx, fy = fforce * dy, fz = fforce * dz;
        if (s.mask[i] & this.groupBit) { f[0] += fx; f[1] += fy; f[2] += fz; }
        if (s.mask[atomIndex(this.sys, j)] & this.groupBit) { f[0] -= fx; f[1] -= fy; f[2] -= fz; }
      });
    }
    this.energy = e;
    this.vector.set(f);
    this.cacheEpoch = this.sys.epoch;
  }

  protected computeScalar(): number {
    this.evaluate();
    return this.energy;
  }

  protected computeVector(): void {
    this.evaluate();
  }
}

/**
 * compute pe/tally, force/tally and stress/tally group1-ID style group2-ID. The per-atom values are
 * kept in arrayAtom (pe/tally: 2 columns evdwl, ecoul; force/tally: 3; stress/tally: 6, xx yy zz xy xz yz).
 */
export class ComputePairTally extends Compute {
  readonly style: string;
  private readonly group2Bit: number;
  private readonly cols: number;
  private cacheEpoch = -1;
  private total = 0;

  constructor(sys: System, id: string, group: string, args: string[], style: 'pe/tally' | 'force/tally' | 'stress/tally') {
    super(sys, id, group, args);
    this.style = style;
    if (args.length !== 1) throw new StyleError(`compute ${id} ${style} takes one group2-ID (no keywords)`);
    this.group2Bit = sys.groups.bit(args[0]);
    this.cols = style === 'pe/tally' ? 2 : style === 'force/tally' ? 3 : 6;
    this.scalarFlag = true;
    this.peratomFlag = true;
    this.sizePeratomCols = this.cols;
    this.extscalar = style === 'stress/tally' ? 0 : 1;
    this.extvector = 0;
  }

  private evaluate(): void {
    if (this.cacheEpoch === this.sys.epoch) return;
    const s = this.sys.state;
    const n = s.n;
    const arr = new Float64Array(this.cols * n);
    let total = 0;
    const style = this.style;
    walkTallyPairs(this.sys, this.id, this.groupBit, this.group2Bit, 'off', (i, j, evdwl, ecoul, fforce, dx, dy, dz) => {
      const oj = atomIndex(this.sys, j);
      if (style === 'pe/tally') {
        total += evdwl + ecoul;
        const h = 0.5 * (evdwl), hc = 0.5 * ecoul;
        arr[2 * i] += h; arr[2 * i + 1] += hc;
        arr[2 * oj] += h; arr[2 * oj + 1] += hc;
      } else if (style === 'force/tally') {
        // force on i from j, and the opposite force on j
        const fx = fforce * dx, fy = fforce * dy, fz = fforce * dz;
        arr[3 * i] += fx; arr[3 * i + 1] += fy; arr[3 * i + 2] += fz;
        arr[3 * oj] -= fx; arr[3 * oj + 1] -= fy; arr[3 * oj + 2] -= fz;
      } else {
        // pair virial d (x) F with F = fforce d: trace fforce |d|^2, and -1/2 of the outer product per atom
        total += fforce * (dx * dx + dy * dy + dz * dz);
        const xx = -0.5 * fforce * dx * dx, yy = -0.5 * fforce * dy * dy, zz = -0.5 * fforce * dz * dz;
        const xy = -0.5 * fforce * dx * dy, xz = -0.5 * fforce * dx * dz, yz = -0.5 * fforce * dy * dz;
        for (const a of [i, oj]) {
          arr[6 * a] += xx; arr[6 * a + 1] += yy; arr[6 * a + 2] += zz;
          arr[6 * a + 3] += xy; arr[6 * a + 4] += xz; arr[6 * a + 5] += yz;
        }
      }
    });
    if (style === 'force/tally') {
      // scalar: magnitude of the force summed over the atoms of group1
      let fx = 0, fy = 0, fz = 0;
      for (let a = 0; a < n; a++) {
        if (!(s.mask[a] & this.groupBit)) continue;
        fx += arr[3 * a]; fy += arr[3 * a + 1]; fz += arr[3 * a + 2];
      }
      total = Math.sqrt(fx * fx + fy * fy + fz * fz);
    } else if (style === 'stress/tally') {
      // scalar: average of the diagonal of the stress tensor, the pair virial trace over 3
      total = total / 3;
      // per-atom values are in pressure*volume units (as compute stress/atom); the scalar is not converted
      // (measured with native LAMMPS, black box: in units real the per-atom values are nktv2p = 68568.415
      // times the energy-unit sums, while compute stress/tally's scalar equals the energy-unit trace / 3)
      const nktv2p = this.sys.state.units.nktv2p;
      if (nktv2p !== 1) for (let k = 0; k < arr.length; k++) arr[k] *= nktv2p;
    }
    this.arrayAtom = arr;
    this.total = total;
    this.cacheEpoch = this.sys.epoch;
  }

  protected computeScalar(): number {
    this.evaluate();
    return this.total;
  }

  protected computePeratom(): void {
    this.evaluate();
  }
}
