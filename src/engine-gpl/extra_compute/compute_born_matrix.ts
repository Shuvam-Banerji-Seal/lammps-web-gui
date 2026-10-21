/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_born_matrix.cpp and compute_born_matrix.h (ComputeBornMatrix) at commit b332f28.
 * Contributing authors: Germain Clavier (TUe), Aidan Thompson (Sandia). Copyright Sandia Corporation (GNU General Public
 * License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * compute ID group-ID born/matrix [keyword ...] (docs.lammps.org/compute_born_matrix.html): the 21 independent second
 * derivatives of the potential energy with respect to strain, C11 C22 C33 C44 C55 C66 C12 C13 C14 C15 C16 C23 C24 C25
 * C26 C34 C35 C36 C45 C46 C56 in Voigt order xx yy zz yz xz xy, in energy units (extvector 0).
 *
 * Analytic (keywords pair, bond, angle, dihedral, improper; none = all): each term's born_matrix (born_styles.ts) gives
 * du and du2, combined with the strain derivatives of its distance or cosine. A style without born_matrix stops at the
 * compute command, a style missing then leaves its term out, and a kspace style stops it. improper is accepted but never
 * summed (native compute_vector has no improper term) and stops for every improper style (none has born_matrix).
 *
 * Two departures from native for the dihedral term: native takes a dihedral's type in the style's born_matrix from
 * neighbor->dihedrallist[i] while i is the dihedral's index on its second atom, and its coordinates from the atoms'
 * stored positions without the minimum image; here every dihedral uses its own type and minimum-image bonds. With one
 * dihedral type and no dihedral split by a periodic boundary the two agree. Measured with native LAMMPS (opls chains,
 * tests/oracle/w418born_bonded.data): its analytic bond and angle terms equal the numdiff values to 1e-10, while its
 * analytic dihedral term differs from them by about 13% of its size; the port keeps native's formula.
 *
 * numdiff delta virial-ID: the finite differences of the virial of compute virial-ID under strains of +-delta
 * (extra_compute/born_matrix.ts). Native skips the word after numdiff's three.
 */
import { Compute } from '../../engine/compute/compute';
import { termRows } from '../../engine/compute/local';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import {
  angleBorn, angleBornEnabled, bondBorn, bondBornEnabled, dihedralBorn, dihedralBornEnabled, pairBorn, pairBornEnabled,
  type Out, type PairArgs,
} from './born_styles';
import { bornMatrixNumdiff } from './born_matrix';

const SMALL = 1e-16;

/** The 3d indices of the six Voigt stress components. */
const SIGMA_ALBE: readonly (readonly [number, number])[] = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]];

/** The Voigt index pair of each of the 21 outputs. */
const C_ALBE: readonly (readonly [number, number])[] = [
  [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [1, 2], [1, 3], [1, 4], [1, 5],
  [2, 3], [2, 4], [2, 5],
  [3, 4], [3, 5],
  [4, 5],
];

/** The 3d indices of each of the 21 outputs. */
const ALBEMUNU: readonly (readonly [number, number, number, number])[] = [
  [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [0, 2, 0, 2], [0, 1, 0, 1],
  [0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 1, 2], [0, 0, 0, 2], [0, 0, 0, 1],
  [1, 1, 2, 2], [1, 1, 1, 2], [1, 1, 0, 2], [1, 1, 0, 1],
  [2, 2, 1, 2], [2, 2, 0, 2], [2, 2, 0, 1],
  [1, 2, 0, 2], [1, 2, 0, 1],
  [0, 1, 0, 2],
];

export class ComputeBornMatrix extends Compute {
  readonly style = 'born/matrix';
  private pairflag = false;
  private bondflag = false;
  private angleflag = false;
  private dihedflag = false;
  private impflag = false;
  private readonly numflag: boolean;
  private readonly numdelta: number;
  private readonly idVirial: string;
  private readonly values = new Float64Array(21);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const illegal = 'Illegal compute born/matrix command';
    let numflag = false, numdelta = 0, idVirial = '';
    if (args.length === 0) {
      this.pairflag = this.bondflag = this.angleflag = this.dihedflag = this.impflag = true;
    } else {
      let iarg = 0;
      while (iarg < args.length) {
        const w = args[iarg];
        if (w === 'numdiff') {
          if (iarg + 3 > args.length) throw new StyleError(illegal);
          numflag = true;
          numdelta = parseNum(args[iarg + 1], 'compute born/matrix delta');
          if (numdelta <= 0.0) throw new StyleError(illegal);
          idVirial = args[iarg + 2];
          const c = sys.computes.find((x) => x.id === idVirial);
          if (!c) throw new StyleError(`Could not find compute born/matrix pressure ID ${idVirial}`);
          if (!c.pressFlag) throw new StyleError(`Compute born/matrix pressure ID ${idVirial} does not compute pressure`);
          iarg += 3;
        } else if (w === 'pair') this.pairflag = true;
        else if (w === 'bond') this.bondflag = true;
        else if (w === 'angle') this.angleflag = true;
        else if (w === 'dihedral') this.dihedflag = true;
        else if (w === 'improper') this.impflag = true;
        else throw new StyleError(illegal);
        ++iarg;
      }
    }
    this.numflag = numflag;
    this.numdelta = numdelta;
    this.idVirial = idVirial;

    const ff = sys.ff;
    const mixed = 'Illegal compute born/matrix command: cannot mix numflag and other flags';
    if (this.pairflag) {
      if (numflag) throw new StyleError(mixed);
      if (ff.pair) {
        if (!pairBornEnabled(ff.pair) || (ff.pair.name === 'hybrid/molecular' && sys.run.ranOnce)) {
          // PairHybridMolecular::init_style clears born_matrix_enable, so it stops once a run has initialized it
          throw new StyleError(`Pair style ${ff.pair.name} does not support compute born/matrix`);
        }
      } else this.pairflag = false;
    }
    const bonded = [
      ['bondflag', 'Bond', ff.bond, bondBornEnabled],
      ['angleflag', 'Angle', ff.angle, angleBornEnabled],
      ['dihedflag', 'Dihedral', ff.dihedral, dihedralBornEnabled],
      ['impflag', 'Improper', ff.improper, () => false],
    ] as const;
    for (const [flag, label, style, enabled] of bonded) {
      if (!this[flag]) continue;
      if (numflag) throw new StyleError(mixed);
      if (style) {
        if (!enabled(style)) throw new StyleError(`${label} style ${style.name} does not support compute born/matrix`);
      } else this[flag] = false;
    }
    if (ff.kspace && !numflag) throw new StyleError('KSpace contribution not supported by compute born/matrix');

    this.vectorFlag = true;
    this.sizeVector = 21;
    this.extvector = 0;
    this.vector = new Float64Array(21);
  }

  init(): void {
    if (this.numflag && !this.sys.computes.some((x) => x.id === this.idVirial)) {
      throw new StyleError(`Could not find compute born/matrix pressure ID ${this.idVirial}`);
    }
  }

  protected computeVector(): void {
    const v = this.values;
    if (!this.numflag) {
      v.fill(0);
      this.sys.forces();
      if (this.pairflag) this.computePairs();
      if (this.bondflag) this.computeBonds();
      if (this.angleflag) this.computeAngles();
      if (this.dihedflag) this.computeDihedrals();
    } else {
      v.set(bornMatrixNumdiff(this.sys, this.id, this.numdelta, this.idVirial));
    }
    this.vector.set(v);
  }

  /** compute_pairs: every pair of the group within the pair cutoff, (du2 - du / r) r_a r_b r_c r_d / r^2. */
  private computePairs(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const pair = sys.ff.pair;
    const list = nb.half;
    if (!pair || !list) return;
    const ff = sys.ff as unknown as { specialLJ: Float64Array; specialCoul: Float64Array; dielectric: number };
    const qqrd2e = s.units.qqr2e * (pair.coulConstScale ?? 1) / ff.dielectric;
    const nt = pair.ntypes + 1;
    const x = nb.xall, nbrs = list.neighbors;
    const rij = [0, 0, 0];
    const a: PairArgs = { ti: 0, tj: 0, rsq: 0, factorCoul: 1, factorLJ: 1, qi: 0, qj: 0, qqrd2e };
    const out: Out = [0, 0];
    const v = this.values;
    for (let i = 0; i < list.inum; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nbrs[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        if (!(s.mask[nb.owner[j]] & this.groupBit)) continue;
        rij[0] = x[3 * j] - x[3 * i];
        rij[1] = x[3 * j + 1] - x[3 * i + 1];
        rij[2] = x[3 * j + 2] - x[3 * i + 2];
        const rsq = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
        const r2inv = 1.0 / rsq;
        const rinv = Math.sqrt(r2inv);
        const ti = nb.typeall[i], tj = nb.typeall[j];
        if (rsq >= pair.cutsq[ti * nt + tj]) continue;
        a.ti = ti; a.tj = tj; a.rsq = rsq;
        a.factorLJ = ff.specialLJ[sb]; a.factorCoul = ff.specialCoul[sb];
        a.qi = nb.qall[i]; a.qj = nb.qall[j];
        pairBorn(pair, a, out);
        // the half list visits each pair once (native halves the term over its full list)
        const pref = out[1] - out[0] * rinv;
        for (let m = 0; m < 21; m++) {
          const [p, q, r, t] = ALBEMUNU[m];
          v[m] += pref * rij[p] * rij[q] * rij[r] * rij[t] * r2inv;
        }
      }
    }
  }

  /** Minimum-image r_b - r_a of two owned atoms. */
  private vec(a: number, b: number): number[] {
    const x = this.sys.state.x;
    const d = [x[3 * b] - x[3 * a], x[3 * b + 1] - x[3 * a + 1], x[3 * b + 2] - x[3 * a + 2]];
    this.sys.geom.minimumImage(d);
    return d;
  }

  /** compute_bonds: (du2 - du / r) r_a r_b r_c r_d / r^2 for every bond of the group. */
  private computeBonds(): void {
    const sys = this.sys;
    const bond = sys.ff.bond;
    if (!bond) return;
    const list = sys.state.topo.bonds;
    const boltz = sys.state.units.boltz;
    const v = this.values;
    for (const { k, idx } of termRows(sys, 'bond', this.groupBit)) {
      const btype = list.type[k];
      if (btype <= 0) continue;
      const rij = this.vec(idx[0], idx[1]);
      const rsq = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
      const r2inv = 1.0 / rsq;
      const rinv = Math.sqrt(r2inv);
      const [du, du2] = bondBorn(bond, btype, rsq, boltz);
      const pref = du2 - du * rinv;
      for (let m = 0; m < 21; m++) {
        const [p, q, r, t] = ALBEMUNU[m];
        v[m] += pref * rij[p] * rij[q] * rij[r] * rij[t] * r2inv;
      }
    }
  }

  /** compute_angles: du d2cos + du2 dcos_e dcos_f with the strain derivatives of cos(theta). */
  private computeAngles(): void {
    const sys = this.sys;
    const angle = sys.ff.angle;
    if (!angle) return;
    const list = sys.state.topo.angles;
    const v = this.values;
    const dcos = new Float64Array(6);
    for (const { k, idx } of termRows(sys, 'angle', this.groupBit)) {
      const atype = list.type[k];
      if (atype <= 0) continue;
      const del1 = this.vec(idx[1], idx[0]);
      const del2 = this.vec(idx[1], idx[2]);
      const rsq1 = del1[0] * del1[0] + del1[1] * del1[1] + del1[2] * del1[2];
      const rsq1inv = 1.0 / rsq1;
      const r1 = Math.sqrt(rsq1);
      const rsq2 = del2[0] * del2[0] + del2[1] * del2[1] + del2[2] * del2[2];
      const rsq2inv = 1.0 / rsq2;
      const r2 = Math.sqrt(rsq2);
      // the style's born_matrix takes cos(theta) of the unperturbed bonds
      let cStyle = (del1[0] * del2[0] + del1[1] * del2[1] + del1[2] * del2[2]) / (r1 * r2);
      if (cStyle > 1.0) cStyle = 1.0;
      if (cStyle < -1.0) cStyle = -1.0;
      let r1r2 = del1[0] * del2[0] + del1[1] * del2[1] + del1[2] * del2[2];
      let r1r2inv: number;
      if (r1r2 === 0.0) {
        // a zero cosine has an undefined logarithm: tilt the first bond slightly toward the second
        for (let a = 0; a < 3; a++) del1[a] += SMALL * del2[a];
        r1r2 = del1[0] * del2[0] + del1[1] * del2[1] + del1[2] * del2[2];
      }
      r1r2inv = 1 / r1r2;
      let cost = r1r2 / (r1 * r2);
      if (cost > 1.0) cost = 1.0;
      if (cost < -1.0) cost = -1.0;
      const cinv = 1.0 / cost;
      const [duang, du2ang] = angleBorn(angle, atype, cStyle);
      for (let m = 0; m < 6; m++) {
        const [a, b] = SIGMA_ALBE[m];
        dcos[m] = cost * ((del1[a] * del2[b] + del1[b] * del2[a]) * r1r2inv - del1[a] * del1[b] * rsq1inv - del2[a] * del2[b] * rsq2inv);
      }
      for (let m = 0; m < 21; m++) {
        const [a, b, c, d] = ALBEMUNU[m];
        const [e, f] = C_ALBE[m];
        const d2lncos = 2 * (del1[a] * del1[b] * del1[c] * del1[d] * rsq1inv * rsq1inv +
          del2[a] * del2[b] * del2[c] * del2[d] * rsq2inv * rsq2inv) -
          (del1[a] * del2[b] + del1[b] * del2[a]) * (del1[c] * del2[d] + del1[d] * del2[c]) * r1r2inv * r1r2inv;
        const d2cos = cost * d2lncos + dcos[e] * dcos[f] * cinv;
        v[m] += duang * d2cos + du2ang * dcos[e] * dcos[f];
      }
    }
  }

  /** compute_dihedrals: dudih d2cos + du2dih dcos_e dcos_f with the strain derivatives of cos(phi). */
  private computeDihedrals(): void {
    const sys = this.sys;
    const dihedral = sys.ff.dihedral;
    if (!dihedral) return;
    const list = sys.state.topo.dihedrals;
    const v = this.values;
    const dab = new Float64Array(6), daa = new Float64Array(6), dbb = new Float64Array(6), dcos = new Float64Array(6);
    for (const { k, idx } of termRows(sys, 'dihedral', this.groupBit)) {
      const [i1, i2, i3, i4] = idx;
      // the style's own bond vectors (vb1 = x1 - x2, vb2 = x3 - x2, vb3 = x4 - x3)
      const [dudih, du2dih] = dihedralBorn(dihedral, list.type[k], this.vec(i2, i1), this.vec(i2, i3), this.vec(i3, i4));
      const b1 = this.vec(i1, i2), b2 = this.vec(i2, i3), b3 = this.vec(i3, i4);
      const cross = (): [number, number, number, number, number, number] => [
        b1[1] * b2[2] - b1[2] * b2[1], b1[2] * b2[0] - b1[0] * b2[2], b1[0] * b2[1] - b1[1] * b2[0],
        b2[1] * b3[2] - b2[2] * b3[1], b2[2] * b3[0] - b2[0] * b3[2], b2[0] * b3[1] - b2[1] * b3[0],
      ];
      const dot = (p: number[], q: number[]): number => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
      let b1sq = dot(b1, b1), b2sq = dot(b2, b2), b3sq = dot(b3, b3);
      let b1b2 = dot(b1, b2), b1b3 = dot(b1, b3), b2b3 = dot(b2, b3);
      let [ax, ay, az, bx, by, bz] = cross();
      const inverses = () => {
        const rasq = ax * ax + ay * ay + az * az;
        const rbsq = bx * bx + by * by + bz * bz;
        const dotab = ax * bx + ay * by + az * bz;
        const ra2inv = rasq > 0 ? 1.0 / rasq : 0.0;
        const rb2inv = rbsq > 0 ? 1.0 / rbsq : 0.0;
        const dotabinv = dotab !== 0 ? 1.0 / dotab : 0.0;
        const rabinv = Math.sqrt(ra2inv * rb2inv);
        return { ra2inv, rb2inv, dotabinv, co: dotab * rabinv };
      };
      let inv = inverses();
      if (inv.co === 0.0) {
        // a zero cosine has an undefined logarithm: tilt the first bond slightly toward the third
        for (let a = 0; a < 3; a++) b1[a] += SMALL * b3[a];
        b1sq = dot(b1, b1); b3sq = dot(b3, b3);
        b1b2 = dot(b1, b2); b1b3 = dot(b1, b3); b2b3 = dot(b2, b3);
        [ax, ay, az, bx, by, bz] = cross();
        inv = inverses();
      }
      const { ra2inv, rb2inv, dotabinv } = inv;
      let co = inv.co;
      if (co > 1.0) co = 1.0;
      if (co < -1.0) co = -1.0;
      for (let m = 0; m < 6; m++) {
        const [al, be] = SIGMA_ALBE[m];
        daa[m] = 2 * (b2sq * b1[al] * b1[be] + b1sq * b2[al] * b2[be] - b1b2 * (b1[al] * b2[be] + b2[al] * b1[be]));
        dbb[m] = 2 * (b2sq * b3[al] * b3[be] + b3sq * b2[al] * b2[be] - b2b3 * (b3[al] * b2[be] + b2[al] * b3[be]));
        dab[m] = b1b2 * (b2[al] * b3[be] + b3[al] * b2[be]) + b2b3 * (b1[al] * b2[be] + b2[al] * b1[be]) -
          b1b3 * (b2[al] * b2[be] + b2[al] * b2[be]) - b2sq * (b1[al] * b3[be] + b3[al] * b1[be]);
        dcos[m] = 0.5 * co * (2 * dotabinv * dab[m] - ra2inv * daa[m] - rb2inv * dbb[m]);
      }
      for (let m = 0; m < 21; m++) {
        const [al, be, mu, nu] = ALBEMUNU[m];
        const [e, f] = C_ALBE[m];
        const d2aa = 4 * (b1[al] * b1[be] * b2[mu] * b2[nu] + b1[mu] * b1[nu] * b2[al] * b2[be]) -
          2 * (b1[al] * b2[be] + b1[be] * b2[al]) * (b1[mu] * b2[nu] + b1[nu] * b2[mu]);
        const d2bb = 4 * (b3[al] * b3[be] * b2[mu] * b2[nu] + b3[mu] * b3[nu] * b2[al] * b2[be]) -
          2 * (b3[al] * b2[be] + b3[be] * b2[al]) * (b3[mu] * b2[nu] + b3[nu] * b2[mu]);
        const d2ab = (b1[al] * b2[be] + b2[al] * b1[be]) * (b3[mu] * b2[nu] + b2[mu] * b3[nu]) +
          (b3[al] * b2[be] + b2[al] * b3[be]) * (b1[mu] * b2[nu] + b2[mu] * b1[nu]) -
          (b1[al] * b3[be] + b3[al] * b1[be]) * (b2[mu] * b2[nu] + b2[mu] * b2[nu]) -
          (b2[al] * b2[be] + b2[al] * b2[be]) * (b1[mu] * b3[nu] + b3[mu] * b1[nu]);
        let d2cos = -2 * dotabinv * dotabinv * dab[e] * dab[f] + 2 * dotabinv * d2ab +
          ra2inv * ra2inv * daa[e] * daa[f] - ra2inv * d2aa +
          rb2inv * rb2inv * dbb[e] * dbb[f] - rb2inv * d2bb +
          2 * dcos[e] * dcos[f] / co / co;
        d2cos *= 0.5 * co;
        v[m] += dudih * d2cos;
        v[m] += du2dih * dcos[e] * dcos[f];
      }
    }
  }
}
