/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/pair_body_rounded_polyhedron.cpp
 * and pair_body_rounded_polyhedron.h (PairBodyRoundedPolyhedron: pair_style body/rounded/polyhedron, the settings,
 * the coefficients, the neighbour loop over the body atoms and the per-atom body geometry). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE). Contributing author (from the source): Trung Dac Nguyen (ndactrung@gmail.com).
 *
 * docs.lammps.org/pair_body_rounded_polyhedron.html: "pair_style body/rounded/polyhedron c_n c_t mu A_ua cutoff
 * keyword" and "pair_coeff * * k_n k_na". The two-body contact law is in rounded_polyhedron_law.ts.
 *
 * Supported: the elastic and cohesive normal law, the damping c_n and c_t, the friction mu, and the keyword history
 * (the tangential spring of the pair, its shear kept per pair of atom IDs, see contact_history.ts). Refused by name:
 * pair_modify shift and tail. The work of the non-conservative forces (pvector) is not kept.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { parseNum } from '../../engine/force/util';
import { BODY_RECORD, POLY_OFF } from '../../engine/bodies';
import { quatToMatrix } from '../../engine/compute/asphere';
import { PolyhedronPairLaw, makePBody, polyAtomRadius, type PBody, type V3 } from './rounded_polyhedron_law';
import { tallyBodyPair } from './contact_history';

export class PairBodyRoundedPolyhedron extends Pair {
  readonly name = 'body/rounded/polyhedron';
  virialFdotr = true;
  cutGlobal = 0;
  cn = 0.1;
  ct = 0.2;
  mu = 0;
  Aua = 1;
  bodyRadius = 0;
  p!: PairParams;
  private readonly law = new PolyhedronPairLaw();
  /** keyword history: the tangential shear of each pair of bodies, keyed by the pair of atom IDs (lower ID first). */
  private history = false;
  private shear = new Map<string, V3>();
  /** native's pair orientation (see compute()): the pair list it was built from, and its keys */
  private readonly nativeOrder = new NativePairOrder();
  private natPairsRef: Int32Array | null = null;
  private natKeys = new Set<string>();

  settings(args: string[]): void {
    if (args.length < 5) throw new StyleError('Illegal pair_style body/rounded/polyhedron command');
    this.cn = parseNum(args[0], 'c_n');
    this.ct = parseNum(args[1], 'c_t');
    this.mu = parseNum(args[2], 'mu');
    this.Aua = parseNum(args[3], 'A_ua');
    this.cutGlobal = parseNum(args[4], 'cutoff');
    // native settings(): a negative A_ua is reset to 1
    if (this.Aua < 0) this.Aua = 1;
    // native settings(): the keyword history keeps the tangential displacement of each pair of bodies
    this.history = false;
    for (let iarg = 5; iarg < args.length; iarg++) {
      if (args[iarg] !== 'history') throw new StyleError(`Unknown pair_style body/rounded/polyhedron keyword ${args[iarg]}`);
      this.history = true;
    }
    this.shear.clear();
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['kn', 'kna', 'kt']);
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    // native coeff(): the tangential stiffness k_t (5th argument) defaults to 2/7 of k_n, as in gran/hooke/history
    const kn = parseNum(args[2], 'k_n');
    const kt = args.length === 5 ? parseNum(args[4], 'k_t') : (2.0 / 7.0) * kn;
    if (kt < 0) throw new StyleError('Tangential stiffness of pair style body/rounded/polyhedron must not be negative');
    this.p.setRange(args[0], args[1], [kn, parseNum(args[3], 'k_na'), kt]);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) {
      if (!this.p.isSet(i, i) || !this.p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      this.p.setMixed(i, j, 'kn', Math.sqrt(this.p.get('kn', i, i) * this.p.get('kn', j, j)));
      this.p.setMixed(i, j, 'kna', Math.sqrt(this.p.get('kna', i, i) * this.p.get('kna', j, j)));
      this.p.setMixed(i, j, 'kt', Math.sqrt(this.p.get('kt', i, i) * this.p.get('kt', j, j)));
    }
    // neighbour cutoff: centres within the two largest radii plus the surface cutoff (native init_one: maxrad + maxrad + cut_inner)
    return 2 * this.bodyRadius + this.cutGlobal;
  }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) return;
    const rec = s.custom.get(BODY_RECORD);
    if (!rec) throw new StyleError('Pair body/rounded/polyhedron requires atom style body rounded/polyhedron');
    const B = rec.cols;
    let rmax = 0;
    for (let o = 0; o < s.n; o++) {
      const nsub = rec.data[B * o];
      if (nsub < 1) continue;
      // the atom radius of the record (native atom->radius): the rounded radius for one sub-particle, else enclosing plus rounded
      rmax = Math.max(rmax, nsub === 1 ? rec.data[B * o + 7] : rec.data[B * o + 6] + rec.data[B * o + 7]);
    }
    this.bodyRadius = rmax;
  }

  init(ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError('pair_modify shift and tail are not supported by pair_style body/rounded/polyhedron');
    super.init(ctx);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const rec = s.custom.get(BODY_RECORD);
    if (!rec || !s.quat || !s.torque || !s.angmom || !s.v) throw new StyleError('Pair body/rounded/polyhedron requires atom style body rounded/polyhedron');
    if (!s.atomStyle.startsWith('body rounded/polyhedron')) throw new StyleError(`pair_style body/rounded/polyhedron requires atom_style body rounded/polyhedron (current: ${s.atomStyle})`);
    const B = rec.cols, bd = rec.data;
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const tq = new Float64Array(3 * pc.nall);
    const law = this.law;
    law.f = f;
    law.tq = tq;
    law.energy = 0;
    const cache = new Map<number, PBody | null>();
    const hist = this.history;
    const touched = new Set<string>();
    const dt = s.dt;
    const update = pc.historyUpdate === true;
    // the body of atom idx (owned or ghost), built from the body record and the state of its owner
    const getBody = (idx: number): PBody | null => {
      const hit = cache.get(idx);
      if (hit !== undefined) return hit;
      const o = owner[idx];
      let res: PBody | null = null;
      const nsub = bd[B * o];
      if (nsub >= 1) {
        const q = s.quat!;
        const R = quatToMatrix(q[4 * o], q[4 * o + 1], q[4 * o + 2], q[4 * o + 3]);
        const L = [s.angmom![3 * o], s.angmom![3 * o + 1], s.angmom![3 * o + 2]];
        const I = [bd[B * o + 1], bd[B * o + 2], bd[B * o + 3]];
        // angmom_to_omega (native): omega from the angular momentum and the principal moments
        const omega: V3 = [0, 0, 0];
        for (let k = 0; k < 3; k++) {
          const e: V3 = [R[k], R[3 + k], R[6 + k]];
          const wb = I[k] === 0 ? 0 : (L[0] * e[0] + L[1] * e[1] + L[2] * e[2]) / I[k];
          omega[0] += wb * e[0];
          omega[1] += wb * e[1];
          omega[2] += wb * e[2];
        }
        const vcm: V3 = [s.v[3 * o], s.v[3 * o + 1], s.v[3 * o + 2]];
        const verts: V3[] = [];
        for (let k = 0; k < nsub; k++) verts.push([bd[B * o + POLY_OFF + 3 * k], bd[B * o + POLY_OFF + 3 * k + 1], bd[B * o + POLY_OFF + 3 * k + 2]]);
        const nedge = bd[B * o + 4], nface = bd[B * o + 5];
        const eo = B * o + POLY_OFF + 3 * nsub;
        const edges: [number, number][] = [];
        for (let e = 0; e < nedge; e++) edges.push([bd[eo + 2 * e], bd[eo + 2 * e + 1]]);
        const fo = eo + 2 * nedge;
        const faces: number[][] = [];
        for (let fc = 0; fc < nface; fc++) {
          const fv: number[] = [];
          for (let m = 0; m < 4; m++) if (bd[fo + 4 * fc + m] >= 0) fv.push(bd[fo + 4 * fc + m]);
          faces.push(fv);
        }
        const c: V3 = [x[3 * idx], x[3 * idx + 1], x[3 * idx + 2]];
        res = makePBody(idx, c, vcm, omega, R, verts, edges, faces, bd[B * o + 6], bd[B * o + 7]);
      }
      cache.set(idx, res);
      return res;
    };
    // native's pairs (npair_bin.cpp, newton on) as keys of (owner, owner, image shift): a pair is taken with the body
    // whose list holds the other one first, as native stores it; a pair native does not store keeps the engine's order
    const gi = nb.gimage;
    const pairKey = (a: number, b: number): string =>
      `${owner[a]}:${owner[b]}:${gi[3 * b] - gi[3 * a]},${gi[3 * b + 1] - gi[3 * a + 1]},${gi[3 * b + 2] - gi[3 * a + 2]}`;
    const nativeFirst = (a: number, b: number): boolean => this.natKeys.has(pairKey(a, b)) || !this.natKeys.has(pairKey(b, a));
    const natPairs = this.nativeOrder.pairs(s, pc.nb, pc.geom, list);
    if (natPairs !== this.natPairsRef) {
      this.natPairsRef = natPairs;
      this.natKeys = new Set<string>();
      for (let q = 0; q < natPairs.length; q += 2) this.natKeys.add(pairKey(natPairs[q], natPairs[q + 1] & NEIGHMASK));
    }
    for (let i = 0; i < list.inum; i++) {
      const ti = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = ti * nt + type[j];
        if (rsq >= cutsq[t]) continue;
        if (jj >>> SBBITS !== 0 && (sL[jj >>> SBBITS] !== 1 || sC[jj >>> SBBITS] !== 1)) {
          throw new StyleError('pair_style body/rounded/polyhedron does not support special_bonds factors other than 1');
        }
        // the pair in native's orientation: native's contact area (rescale_cohesive_forces, src/BODY/pair_body_rounded_polyhedron.cpp
        // at b332f28) is taken at the contact points of its first body (contacts[m].xi), so the order of the two bodies matters
        const ii = nativeFirst(i, j) ? i : j;
        const jn = ii === i ? j : i;
        const bi = getBody(ii), bj = getBody(jn);
        if (!bi || !bj) continue;
        // native compute(): the pair is in the interaction range when r <= radius_i + radius_j + cut_inner
        const r = Math.sqrt(rsq);
        if (!(r <= polyAtomRadius(bi) + polyAtomRadius(bj) + this.cutGlobal)) continue;
        // the shear is stored as seen from the atom with the lower ID (the sign flips for the other one)
        const ida = s.id[owner[ii]], idb = s.id[owner[jn]];
        const lowerIsI = ida < idb;
        const key = lowerIsI ? `${ida}:${idb}` : `${idb}:${ida}`;
        let shear: V3 = [0, 0, 0];
        if (hist) {
          const v = this.shear.get(key);
          if (v) shear = lowerIsI ? v : [-v[0], -v[1], -v[2]];
        }
        const tii = type[ii], tjj = type[jn];
        law.P = {
          kn: this.p.get('kn', tii, tjj), kna: this.p.get('kna', tii, tjj),
          cn: this.cn, ct: this.ct, mu: this.mu, Aua: this.Aua, cut: this.cutGlobal,
          history: hist, kt: this.p.get('kt', tii, tjj), dt, update, shear,
        };
        // the pair's energy and its force on ii (facc of the native ev_tally_xyz), from the law's accumulators
        const tallyAtoms = pc.eatom !== null || pc.vatom !== null;
        const e0 = law.energy, f0x = f[3 * ii], f0y = f[3 * ii + 1], f0z = f[3 * ii + 2];
        law.interact(bi, bj);
        if (hist) {
          const sh = law.applyHistory();
          if (sh) {
            this.shear.set(key, lowerIsI ? sh : [-sh[0], -sh[1], -sh[2]]);
            touched.add(key);
          }
        }
        if (tallyAtoms) {
          const facc: [number, number, number] = [f[3 * ii] - f0x, f[3 * ii + 1] - f0y, f[3 * ii + 2] - f0z];
          tallyBodyPair(pc, ii, jn, law.energy - e0, facc, x[3 * ii] - x[3 * jn], x[3 * ii + 1] - x[3 * jn + 1], x[3 * ii + 2] - x[3 * jn + 2]);
        }
      }
    }
    // a pair without a tangential spring this step loses its shear (native: touch = 0 zeroes the displacement)
    if (hist) for (const key of this.shear.keys()) if (!touched.has(key)) this.shear.delete(key);
    nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += law.energy;
  }
}
