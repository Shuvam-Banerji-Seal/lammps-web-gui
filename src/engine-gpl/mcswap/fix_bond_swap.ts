/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MC/fix_bond_swap.cpp and
 * src/MC/fix_bond_swap.h (Paul Crozier and Robert Auhl's bond-swap algorithm, the Auhl et al. 2003 melt equilibration:
 * "Equilibration of Long Chain Polymer Melts in Computer Simulations", J. Chem. Phys. 119, 12718). LAMMPS is Copyright
 * (2003) Sandia Corporation and distributed under the GNU General Public License, version 2. This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). The bond and angle energies are the single() formulas of src/MOLECULE/bond_fene.cpp and
 * src/MOLECULE/angle_harmonic.cpp, read at b332f28; the pair energy is the pair style's single().
 *
 * fix bond/swap (docs.lammps.org/fix_bond_swap.html): every nevery steps, a random fraction of the group atoms I is
 * tried; for each, the group atoms J in the neighbour list of I (in a random order), in the same molecule, with
 * bonds I-Inext and J-Jnext (Inext and Jnext in the group, in the same molecule, distinct), are tested as a foursome:
 * the bonds become I-Jnext and J-Inext. All four new and old bonds must be shorter than the cutoff. The energy change of
 * the non-bonded pairs, the bonds and (with an angle style) the angles of the foursome decides the swap: accepted if it
 * is negative, else with the Boltzmann factor at the current temperature of the group. Only the first valid foursome
 * of each event is evaluated (native's goto done).
 *
 * Native structure kept here, for one processor with newton bond on (the default; the bond and angle storage is then
 * one entry per bond on its first atom, and per angle on its central atom): the candidate pairs are the half list of the
 * native order (nativePairs), without the 1-2 neighbours of I (native's special_bonds 0 1 1 excludes them from the list);
 * the permutations draw from RanMars(seed) as native does (the shuffle of the group atoms, then one shuffle per tested
 * atom, and one uniform per Metropolis test with delta >= 0). A swap that is accepted sets the next reneighbouring to
 * the current step and refreshes the topology.
 *
 * Not reproduced: newton bond off (native stores each bond on both atoms; refused here), bond/history fixes (no history
 * is reset), and the bond and angle styles other than harmonic and fene, and the pair styles without a single() (native
 * requires single(); refused by name). The temperature is the group's compute temp, computed here as native computes it.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isMolecularStyle, nativeOrder } from '../../engine/atoms';
import { NEIGHMASK } from '../../engine/neighbor';
import { RanMars } from '../random_mars';
import { exp as libmExp } from '../libm/exp';
import { log as libmLog } from '../libm/log';
import { nativePairs } from '../neighbor/nativeOrder';
import { oneTwoOrder, refreshTopology } from '../mc/bond_topo';
import { acos as libmAcos } from '../libm/acos';

/** utils::numeric (utils.cpp). */
const numeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return v;
};
/** utils::inumeric (utils.cpp). */
const inumeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isInteger(v)) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return v;
};

/** MY_CUBEROOT2 (math_const.h): 2^(1/3). */
const MY_CUBEROOT2 = 1.25992104989487316476;

/** The pair style's single(): energy of one pair at squared distance rsq (full interaction, factors 1). */
type PairSingle = (i: number, j: number, itype: number, jtype: number, rsq: number, fc: number, flj: number, qi: number, qj: number) => { eng: number; fforce: number };

export class FixBondSwap extends Fix {
  readonly style = 'bond/swap';
  private readonly fraction: number;
  private readonly cutsq: number;
  private readonly random: RanMars;
  private naccept = 0;
  private foursome = 0;
  private angleflag = false;
  private nextReneighbor = -1;
  /** the bond and angle energy functions of the styles this port evaluates */
  private bondKind: 'harmonic' | 'fene' = 'harmonic';
  private angleKind: 'harmonic' = 'harmonic';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4) throw new StyleError('Illegal fix bond/swap command');
    this.nevery = inumeric(args[0]);
    if (this.nevery <= 0) throw new StyleError('Illegal fix bond/swap command');
    this.vectorFlag = true;
    this.sizeVector = 2;
    this.extvector = 0;
    this.globalFreq = 1;
    this.fraction = numeric(args[1]);
    const cutoff = numeric(args[2]);
    this.cutsq = cutoff * cutoff;
    const seed = inumeric(args[3]);
    if (!isMolecularStyle(sys.state.atomStyle)) throw new StyleError('Cannot use fix bond/swap with non-molecular systems');
    this.random = new RanMars(seed);
  }

  /** FixBondSwap::init: the pair, bond, angle and special checks of native, and the styles this port evaluates. */
  init(): void {
    const sys = this.sys;
    const ff = sys.ff;
    if (!ff.pair || !ff.bond) throw new StyleError('Fix bond/swap requires pair and bond styles');
    if (typeof (ff.pair as unknown as { single?: unknown }).single !== 'function') {
      throw new StyleError('Pair style does not support fix bond/swap');
    }
    if (ff.dihedral || ff.improper) throw new StyleError('Fix bond/swap cannot be used with dihedral or improper styles');
    const lj = ff.special.lj;
    if (lj[0] !== 0.0 || lj[1] !== 1.0 || lj[2] !== 1.0) throw new StyleError('Fix bond/swap requires special_bonds = 0,1,1');
    if (!sys.newtonBond) {
      throw new StyleError('fix bond/swap: newton bond off (bonds stored by both atoms) is not supported by the browser engine');
    }
    if (ff.bond.name === 'harmonic') this.bondKind = 'harmonic';
    else if (ff.bond.name === 'fene') this.bondKind = 'fene';
    else throw new StyleError(`fix bond/swap: bond style ${ff.bond.name} is not supported by the browser engine (harmonic and fene are)`);
    this.angleflag = Boolean(ff.angle);
    if (ff.angle && ff.angle.name !== 'harmonic') {
      throw new StyleError(`fix bond/swap: angle style ${ff.angle.name} is not supported by the browser engine (harmonic is)`);
    }
    this.naccept = 0;
    this.foursome = 0;
  }

  reneighborDue(step: number): boolean {
    return step === this.nextReneighbor;
  }

  /** compute_vector: 0 = accepted swaps, 1 = foursomes evaluated. */
  computeVector(i: number): number {
    return i === 0 ? this.naccept : this.foursome;
  }

  /** The compute temp of the group all, as native computes it (dof = dimension N - dimension - fix dof). */
  private temperature(): number {
    const sys = this.sys;
    const s = sys.state;
    const allBit = sys.groups.bit('all');
    const dof = sys.dimension * s.n - sys.dimension - sys.dofRemoved(allBit);
    const order = nativeOrder(s);
    let t = 0;
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      const m = s.rmass ? s.rmass[i] : s.massByType[s.type[i]];
      t += m * (s.v[3 * i] * s.v[3 * i] + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2]);
    }
    const tfactor = dof > 0 ? s.units.mvv2e / (dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  /** dist_rsq: the squared minimum-image distance of owned atoms i and j (FixBondSwap::dist_rsq). */
  private distRsq(i: number, j: number): number {
    const s = this.sys.state;
    const d = [s.x[3 * i] - s.x[3 * j], s.x[3 * i + 1] - s.x[3 * j + 1], s.x[3 * i + 2] - s.x[3 * j + 2]];
    this.sys.geom.minimumImage(d);
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  }

  /** pair_eng: the pair style's single() with full interactions. */
  private pairEng(i: number, j: number): number {
    const s = this.sys.state;
    const pair = this.sys.ff.pair as unknown as { single: PairSingle };
    const rsq = this.distRsq(i, j);
    return pair.single(i, j, s.type[i], s.type[j], rsq, 1.0, 1.0, s.q[i], s.q[j]).eng;
  }

  /** bond_eng: the bond style's single() at the squared distance of i and j. */
  private bondEng(btype: number, i: number, j: number): number {
    const sys = this.sys;
    const rsq = this.distRsq(i, j);
    const params = (sys.ff.bond as unknown as { params: { p(name: string): Float64Array } }).params;
    if (this.bondKind === 'harmonic') {
      const K = params.p('K')[btype], r0 = params.p('r0')[btype];
      const r = Math.sqrt(rsq);
      const dr = r - r0;
      const rk = K * dr;
      return rk * dr;
    }
    // BondFENE::single
    const R0 = params.p('R0')[btype], k = params.p('K')[btype];
    const eps = params.p('epsilon')[btype], sig = params.p('sigma')[btype];
    const r0sq = R0 * R0;
    let rlogarg = 1.0 - rsq / r0sq;
    if (rlogarg < 0.1) {
      sys.warn(`FENE bond too long: ${sys.state.step} ${Math.sqrt(rsq)}`);
      if (rlogarg <= -3.0) throw new StyleError('Bad FENE bond');
      rlogarg = 0.1;
    }
    let eng = -0.5 * k * r0sq * libmLog(rlogarg);
    if (rsq < MY_CUBEROOT2 * sig * sig) {
      const sr2 = (sig * sig) / rsq;
      const sr6 = sr2 * sr2 * sr2;
      eng += 4.0 * eps * sr6 * (sr6 - 1.0) + eps;
    }
    return eng;
  }

  /** angle_eng: the angle style's single() for the angle i-j-k (zero when an end is missing, -1). */
  private angleEng(atype: number, i: number, j: number, k: number): number {
    if (i === -1 || k === -1) return 0.0;
    const sys = this.sys;
    const s = sys.state;
    const params = (sys.ff.angle as unknown as { params: { p(name: string): Float64Array } }).params;
    const K = params.p('K')[atype], theta0 = params.p('theta0')[atype];
    const d1 = [s.x[3 * i] - s.x[3 * j], s.x[3 * i + 1] - s.x[3 * j + 1], s.x[3 * i + 2] - s.x[3 * j + 2]];
    sys.geom.minimumImage(d1);
    const r1 = Math.sqrt(d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]);
    const d2 = [s.x[3 * k] - s.x[3 * j], s.x[3 * k + 1] - s.x[3 * j + 1], s.x[3 * k + 2] - s.x[3 * j + 2]];
    sys.geom.minimumImage(d2);
    const r2 = Math.sqrt(d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]);
    let c = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];
    c /= r1 * r2;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    const dtheta = libmAcos(c) - theta0;
    const tk = K * dtheta;
    return tk * dtheta;
  }

  /** neighbor_permutation: a one-pass Fisher-Yates shuffle of 0 .. n-1 (native's draws, one per position but the last). */
  private neighborPermutation(n: number): Int32Array {
    const permute = new Int32Array(n);
    for (let i = 0; i < n; i++) permute[i] = i;
    for (let i = 0; i < n - 1; i++) {
      const j = i + Math.trunc(this.random.uniform() * (n - i));
      const tmp = permute[i];
      permute[i] = permute[j];
      permute[j] = tmp;
    }
    return permute;
  }

  /** post_integrate: the bond swap attempts of this event (native FixBondSwap::post_integrate). */
  postIntegrate(): void {
    const sys = this.sys;
    const s = sys.state;
    if (s.step % this.nevery !== 0) return;
    const tCurrent = this.temperature();
    const gb = this.groupBit;
    const boltz = s.units.boltz;
    const nlocal = s.n;
    const nb = sys.nb;
    const B = s.topo.bonds;
    const A = s.topo.angles;

    // tag -> owned index
    const index = new Map<number, number>();
    for (let i = 0; i < nlocal; i++) index.set(s.id[i], i);
    const idx = (tag: number): number => index.get(tag) ?? -1;

    // the stored bonds of each atom (owner = first atom of the bond) and angles (owner = central atom), in storage order
    const ownedBonds: number[][] = Array.from({ length: nlocal }, () => []);
    for (let e = 0; e < B.n; e++) {
      if (B.type[e] === 0) continue;
      const o = idx(B.atoms[2 * e]);
      if (o >= 0) ownedBonds[o].push(e);
    }
    const ownedAngles: number[][] = Array.from({ length: nlocal }, () => []);
    if (this.angleflag) {
      for (let e = 0; e < A.n; e++) {
        if (A.type[e] === 0) continue;
        const o = idx(A.atoms[3 * e + 1]);
        if (o >= 0) ownedAngles[o].push(e);
      }
    }

    // the half neighbour list of this event in native order; 1-2 neighbours of each atom are not in it (special 0 1 1)
    const oneTwo = oneTwoOrder(s);
    const list = nb.occasionalList(s, nb.cutneighmax, false, s.ntypes);
    const pairs = nativePairs(s, nb, sys.geom, list, true);
    const perAtom: number[][] = Array.from({ length: nlocal }, () => []);
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const jx = pairs[p + 1] & NEIGHMASK;
      const partnerTag = s.id[jx < nlocal ? jx : nb.owner[jx]];
      const ones = oneTwo.get(s.id[i]);
      if (ones && ones.includes(partnerTag)) continue;
      perAtom[i].push(pairs[p + 1]);
    }

    // randomize the group atoms (native local order), then the candidate count
    const alist: number[] = [];
    const order = nativeOrder(s);
    for (let k = 0; k < nlocal; k++) {
      const i = order[k];
      if (s.mask[i] & gb) alist.push(i);
    }
    const neligible = alist.length;
    for (let i = 0; i < neligible - 1; i++) {
      const j = i + Math.trunc(this.random.uniform() * (neligible - i));
      const tmp = alist[i];
      alist[i] = alist[j];
      alist[j] = tmp;
    }
    const ntest = Math.trunc(this.fraction * neligible);

    let accept = false;
    // the first valid foursome of the event (native's goto done)
    let found: null | {
      i: number; j: number; inext: number; jnext: number; iprev: number; ilast: number; jprev: number; jlast: number;
      ibondtype: number; jbondtype: number; iangletype: number; jangletype: number; inextangletype: number; jnextangletype: number;
    } = null;
    let delta = 0;
    search:
    for (let itest = 0; itest < ntest; itest++) {
      const i = alist[itest];
      const jl = perAtom[i];
      const jnum = jl.length;
      const permute = this.neighborPermutation(jnum);
      for (let jj = 0; jj < jnum; jj++) {
        const jx = jl[permute[jj]] & NEIGHMASK;
        if (jx >= nlocal) continue;
        const j = jx;
        if (!(s.mask[j] & gb)) continue;
        if (s.molecule[i] !== s.molecule[j]) continue;
        const itag = s.id[i], jtag = s.id[j];
        for (const eBi of ownedBonds[i]) {
          const inext = idx(B.atoms[2 * eBi + 1]);
          if (inext < 0 || inext >= nlocal) continue;
          if (!(s.mask[inext] & gb)) continue;
          const ibondtype = B.type[eBi];
          for (const eBj of ownedBonds[j]) {
            const jnext = idx(B.atoms[2 * eBj + 1]);
            if (jnext < 0 || jnext >= nlocal) continue;
            if (!(s.mask[jnext] & gb)) continue;
            const jbondtype = B.type[eBj];
            if (s.molecule[inext] !== s.molecule[jnext]) continue;
            if (inext === jnext || inext === j) continue;
            if (this.distRsq(i, inext) >= this.cutsq) continue;
            if (this.distRsq(j, jnext) >= this.cutsq) continue;
            if (this.distRsq(i, jnext) >= this.cutsq) continue;
            if (this.distRsq(j, inext) >= this.cutsq) continue;

            const inexttag = s.id[inext], jnexttag = s.id[jnext];
            // angles of the four atoms (newton bond on: stored by the central atom)
            let iprev = -1, ilast = -1, jprev = -1, jlast = -1;
            let iangletype = 0, inextangletype = 0, jangletype = 0, jnextangletype = 0;
            if (this.angleflag) {
              for (const e of ownedAngles[i]) {
                const i1 = A.atoms[3 * e], i2 = A.atoms[3 * e + 1], i3 = A.atoms[3 * e + 2];
                if (i2 === itag && i3 === inexttag) iprev = idx(i1);
                else if (i1 === inexttag && i2 === itag) iprev = idx(i3);
                if (iprev >= 0) { iangletype = A.type[e]; break; }
              }
              for (const e of ownedAngles[inext]) {
                const i1 = A.atoms[3 * e], i2 = A.atoms[3 * e + 1], i3 = A.atoms[3 * e + 2];
                if (i1 === itag && i2 === inexttag) ilast = idx(i3);
                else if (i2 === inexttag && i3 === itag) ilast = idx(i1);
                if (ilast >= 0) { inextangletype = A.type[e]; break; }
              }
              for (const e of ownedAngles[j]) {
                const j1 = A.atoms[3 * e], j2 = A.atoms[3 * e + 1], j3 = A.atoms[3 * e + 2];
                if (j2 === jtag && j3 === jnexttag) jprev = idx(j1);
                else if (j1 === jnexttag && j2 === jtag) jprev = idx(j3);
                if (jprev >= 0) { jangletype = A.type[e]; break; }
              }
              for (const e of ownedAngles[jnext]) {
                const j1 = A.atoms[3 * e], j2 = A.atoms[3 * e + 1], j3 = A.atoms[3 * e + 2];
                if (j1 === jtag && j2 === jnexttag) jlast = idx(j3);
                else if (j2 === jnexttag && j3 === jtag) jlast = idx(j1);
                if (jlast >= 0) { jnextangletype = A.type[e]; break; }
              }
            }

            // energy change of the foursome: non-bonded, bonds, then angles
            delta = this.pairEng(i, inext) + this.pairEng(j, jnext) - this.pairEng(i, jnext) - this.pairEng(inext, j);
            delta += this.bondEng(ibondtype, i, jnext) + this.bondEng(jbondtype, j, inext) -
              this.bondEng(ibondtype, i, inext) - this.bondEng(jbondtype, j, jnext);
            if (this.angleflag) {
              delta += (((((((this.angleEng(iangletype, iprev, i, jnext) + this.angleEng(jnextangletype, i, jnext, jlast)) +
                this.angleEng(jangletype, jprev, j, inext)) + this.angleEng(inextangletype, j, inext, ilast)) -
                this.angleEng(iangletype, iprev, i, inext)) - this.angleEng(inextangletype, i, inext, ilast)) -
                this.angleEng(jangletype, jprev, j, jnext)) - this.angleEng(jnextangletype, j, jnext, jlast));
            }
            this.foursome++;
            found = { i, j, inext, jnext, iprev, ilast, jprev, jlast, ibondtype, jbondtype, iangletype, jangletype, inextangletype, jnextangletype };
            break search;
          }
        }
      }
    }
    if (found) {
      if (delta < 0.0) accept = true;
      else {
        const factor = libmExp(-delta / boltz / tCurrent);
        if (this.random.uniform() < factor) accept = true;
      }
    }

    if (accept) this.nextReneighbor = s.step;
    if (!accept || !found) return;
    this.naccept++;
    this.applySwap(found, ownedBonds, ownedAngles, oneTwo);
  }

  /** The bond, special and angle changes of an accepted foursome (native post_integrate after the decision). */
  private applySwap(
    f: { i: number; j: number; inext: number; jnext: number; iprev: number; ilast: number; jprev: number; jlast: number },
    ownedBonds: number[][], ownedAngles: number[][], oneTwo: Map<number, number[]>,
  ): void {
    const s = this.sys.state;
    const B = s.topo.bonds;
    const A = s.topo.angles;
    const { i, j, inext, jnext } = f;
    const tag = (k: number) => s.id[k];
    const itag = tag(i), inexttag = tag(inext), jtag = tag(j), jnexttag = tag(jnext);
    // bonds: i-inext -> i-jnext (owned by i), j-jnext -> j-inext (owned by j), inext-i -> inext-j, jnext-j -> jnext-i
    for (const e of ownedBonds[i]) if (B.atoms[2 * e + 1] === inexttag) B.atoms[2 * e + 1] = jnexttag;
    for (const e of ownedBonds[j]) if (B.atoms[2 * e + 1] === jnexttag) B.atoms[2 * e + 1] = inexttag;
    for (const e of ownedBonds[inext]) if (B.atoms[2 * e + 1] === itag) B.atoms[2 * e + 1] = jtag;
    for (const e of ownedBonds[jnext]) if (B.atoms[2 * e + 1] === jtag) B.atoms[2 * e + 1] = itag;

    // 1-2 neighbour lists (special 0 1 1): inext -> jnext on i, jnext -> inext on j, i -> j on inext, j -> i on jnext
    const swapIn = (list: number[] | undefined, from: number, to: number) => {
      if (!list) return;
      for (let m = 0; m < list.length; m++) if (list[m] === from) list[m] = to;
    };
    swapIn(oneTwo.get(itag), inexttag, jnexttag);
    swapIn(oneTwo.get(jtag), jnexttag, inexttag);
    swapIn(oneTwo.get(inexttag), itag, jtag);
    swapIn(oneTwo.get(jnexttag), jtag, itag);

    if (!this.angleflag) {
      refreshTopology(this.sys);
      return;
    }
    const iprevtag = f.iprev >= 0 ? tag(f.iprev) : 0;
    const ilasttag = f.ilast >= 0 ? tag(f.ilast) : 0;
    const jprevtag = f.jprev >= 0 ? tag(f.jprev) : 0;
    const jlasttag = f.jlast >= 0 ? tag(f.jlast) : 0;
    // angles stored by each atom (its central atom): literal native loops over the owned lists
    for (const e of ownedAngles[i]) {
      const a1 = A.atoms[3 * e], a2 = A.atoms[3 * e + 1], a3 = A.atoms[3 * e + 2];
      if (a1 === iprevtag && a2 === itag && a3 === inexttag) A.atoms[3 * e + 2] = jnexttag;
      else if (a1 === inexttag && a2 === itag && a3 === iprevtag) A.atoms[3 * e] = jnexttag;
      else if (a1 === itag && a2 === inexttag && a3 === ilasttag) { A.atoms[3 * e + 1] = jnexttag; A.atoms[3 * e + 2] = jlasttag; }
      else if (a1 === ilasttag && a2 === inexttag && a3 === itag) { A.atoms[3 * e] = jlasttag; A.atoms[3 * e + 1] = jnexttag; }
    }
    for (const e of ownedAngles[j]) {
      const a1 = A.atoms[3 * e], a2 = A.atoms[3 * e + 1], a3 = A.atoms[3 * e + 2];
      if (a1 === jprevtag && a2 === jtag && a3 === jnexttag) A.atoms[3 * e + 2] = inexttag;
      else if (a1 === jnexttag && a2 === jtag && a3 === jprevtag) A.atoms[3 * e] = inexttag;
      else if (a1 === jtag && a2 === jnexttag && a3 === jlasttag) { A.atoms[3 * e + 1] = inexttag; A.atoms[3 * e + 2] = ilasttag; }
      else if (a1 === jlasttag && a2 === jnexttag && a3 === jtag) { A.atoms[3 * e] = ilasttag; A.atoms[3 * e + 1] = inexttag; }
    }
    for (const e of ownedAngles[inext]) {
      const a1 = A.atoms[3 * e], a2 = A.atoms[3 * e + 1], a3 = A.atoms[3 * e + 2];
      if (a1 === iprevtag && a2 === itag && a3 === inexttag) { A.atoms[3 * e] = jprevtag; A.atoms[3 * e + 1] = jtag; }
      else if (a1 === inexttag && a2 === itag && a3 === iprevtag) { A.atoms[3 * e + 1] = jtag; A.atoms[3 * e + 2] = jprevtag; }
      else if (a1 === itag && a2 === inexttag && a3 === ilasttag) A.atoms[3 * e] = jtag;
      else if (a1 === ilasttag && a2 === inexttag && a3 === itag) A.atoms[3 * e + 2] = jtag;
    }
    for (const e of ownedAngles[jnext]) {
      const a1 = A.atoms[3 * e], a2 = A.atoms[3 * e + 1], a3 = A.atoms[3 * e + 2];
      if (a1 === jprevtag && a2 === jtag && a3 === jnexttag) { A.atoms[3 * e] = iprevtag; A.atoms[3 * e + 1] = itag; }
      else if (a1 === jnexttag && a2 === jtag && a3 === jprevtag) { A.atoms[3 * e + 1] = itag; A.atoms[3 * e + 2] = iprevtag; }
      else if (a1 === jtag && a2 === jnexttag && a3 === jlasttag) A.atoms[3 * e] = itag;
      else if (a1 === jlasttag && a2 === jnexttag && a3 === jtag) A.atoms[3 * e + 2] = itag;
    }
    refreshTopology(this.sys);
  }
}
