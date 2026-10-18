/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/pair_body_rounded_polygon.cpp and
 * pair_body_rounded_polygon.h (PairBodyRoundedPolygon: pair_style body/rounded/polygon, the neighbour loop, the
 * settings and the coefficients). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author (from the
 * source): Trung Dac Nguyen (ndactrung@gmail.com).
 *
 * docs.lammps.org/pair_body_rounded_polygon.html: "pair_style body/rounded/polygon c_n c_t mu delta_ua cutoff keyword"
 * and "pair_coeff * * k_n k_na". The contact law, the damping and the friction are in rounded_polygon_law.ts; the
 * contact history (keyword history) is in contact_history.ts.
 *
 * Supported: the elastic and cohesive normal law, the damping c_n and c_t, the friction mu, and the keyword history
 * (the tangential spring of the pair, its shear kept per pair of atom IDs). Refused by name: pair_modify shift and tail.
 * Without the history the friction of the native source acts at one contact (the one with the largest overlap).
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { BODY_RECORD, bodyRecordOffset } from '../../engine/bodies';
import { tallyBodyPair, type V3 } from './contact_history';
import { quatToMatrix } from '../../engine/compute/asphere';
import {
  atomRadius, bodyGeom, roundedPolygonPair, type BodyGeom, type RPParams,
} from './rounded_polygon_law';

export class PairBodyRoundedPolygon extends Pair {
  readonly name = 'body/rounded/polygon';
  virialFdotr = true;
  cutGlobal = 0;
  dua = 0;
  cn = 0;
  ct = 0;
  mu = 0;
  bodyRadius = 0;
  p!: PairParams;
  /** keyword history: the tangential shear of each pair of bodies, keyed by the pair of atom IDs (lower ID first). */
  private history = false;
  private shear = new Map<string, V3>();

  settings(args: string[]): void {
    if (args.length < 5) throw new StyleError('Illegal pair_style body/rounded/polygon command');
    this.cn = parseNum(args[0], 'c_n');
    this.ct = parseNum(args[1], 'c_t');
    this.mu = parseNum(args[2], 'mu');
    this.dua = parseNum(args[3], 'delta_ua');
    this.cutGlobal = parseNum(args[4], 'cutoff');
    // native settings(): a negative delta_ua is reset to 1
    if (this.dua < 0) this.dua = 1;
    // native settings(): the keyword history keeps the tangential displacement of each pair of bodies
    this.history = false;
    for (let a = 5; a < args.length; a++) {
      if (args[a] !== 'history') throw new StyleError(`Unknown pair_style body/rounded/polygon keyword ${args[a]}`);
      this.history = true;
    }
    this.shear.clear();
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style body/rounded/polygon cutoff must be > 0');
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
    if (kt < 0) throw new StyleError('Tangential stiffness of pair style body/rounded/polygon must not be negative');
    this.p.setRange(args[0], args[1], [kn, parseNum(args[3], 'k_na'), kt]);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) {
      if (!this.p.isSet(i, i) || !this.p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      this.p.setMixed(i, j, 'kn', Math.sqrt(this.p.get('kn', i, i) * this.p.get('kn', j, j)));
      this.p.setMixed(i, j, 'kna', Math.sqrt(this.p.get('kna', i, i) * this.p.get('kna', j, j)));
      this.p.setMixed(i, j, 'kt', Math.sqrt(this.p.get('kt', i, i) * this.p.get('kt', j, j)));
    }
    // neighbour cutoff: two centres closer than twice the largest body radius plus the surface cutoff
    return 2 * this.bodyRadius + this.cutGlobal;
  }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) return;
    const rec = s.custom.get(BODY_RECORD);
    if (!rec) throw new StyleError('Pair body/rounded/polygon requires atom style body rounded/polygon');
    const B = rec.cols, off = bodyRecordOffset('rounded/polygon');
    let rmax = 0;
    for (let o = 0; o < s.n; o++) {
      const nv = rec.data[B * o];
      let far = 0;
      for (let k = 0; k < nv; k++) {
        const dx = rec.data[B * o + off + 3 * k], dy = rec.data[B * o + off + 3 * k + 1];
        far = Math.max(far, Math.hypot(dx, dy));
      }
      rmax = Math.max(rmax, far + 0.5 * rec.data[B * o + 4]);
    }
    this.bodyRadius = rmax;
  }

  init(ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError('pair_modify shift and tail are not supported by pair_style body/rounded/polygon');
    super.init(ctx);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const rec = s.custom.get(BODY_RECORD);
    if (!rec || !s.quat || !s.torque) throw new StyleError('Pair body/rounded/polygon requires atom style body rounded/polygon');
    if (!s.atomStyle.startsWith('body rounded/polygon')) throw new StyleError(`pair_style body/rounded/polygon requires atom_style body rounded/polygon (current: ${s.atomStyle})`);
    const B = rec.cols, bd = rec.data;
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const tq = new Float64Array(3 * pc.nall);
    const off = bodyRecordOffset('rounded/polygon');
    // the space-frame geometry of every owned body (centre and velocities are set per atom in the pair loop)
    const geo: (BodyGeom | null)[] = new Array(nlocal).fill(null);
    for (let o = 0; o < nlocal; o++) {
      const nv = bd[B * o];
      if (nv < 1) continue;
      const q = s.quat!;
      const R = quatToMatrix(q[4 * o], q[4 * o + 1], q[4 * o + 2], q[4 * o + 3]);
      const body: V3[] = [];
      for (let k = 0; k < nv; k++) body.push([bd[B * o + off + 3 * k], bd[B * o + off + 3 * k + 1], bd[B * o + off + 3 * k + 2]]);
      const omega = this.spaceOmega(s, o, bd, B);
      geo[o] = bodyGeom([0, 0, 0], [s.v[3 * o], s.v[3 * o + 1], s.v[3 * o + 2]], omega, body, R, bd[B * o + 4]);
    }
    const params: RPParams = { kn: 0, kna: 0, cn: this.cn, ct: this.ct, mu: this.mu, dua: this.dua, cut: this.cutGlobal };
    const hist = this.history;
    const touched = new Set<string>();
    const dt = s.dt;
    const update = pc.historyUpdate === true;
    let evdwl = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const gi = geo[oi];
      const ti = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const gj = geo[oj];
        const t = ti * nt + type[j];
        const xi: V3 = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
        const xj: V3 = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
        const rsq = (xi[0] - xj[0]) ** 2 + (xi[1] - xj[1]) ** 2 + (xi[2] - xj[2]) ** 2;
        if (rsq >= cutsq[t]) continue;
        if (!gi || !gj) continue; // a point particle does not interact through the body style
        if (jj >>> SBBITS !== 0 && (sL[jj >>> SBBITS] !== 1 || sC[jj >>> SBBITS] !== 1)) {
          throw new StyleError('pair_style body/rounded/polygon does not support special_bonds factors other than 1');
        }
        // native pair loop: the pair interacts when the centres are within the two radii and the cutoff
        const r = Math.sqrt(rsq);
        if (!(r <= atomRadius(gi) + atomRadius(gj) + this.cutGlobal)) continue;
        const kn = this.p.get('kn', ti, type[j]), kna = this.p.get('kna', ti, type[j]);
        const bi: BodyGeom = { ...gi, c: xi };
        const bj: BodyGeom = { ...gj, c: xj };
        // the shear is stored as seen from the atom with the lower ID (the sign flips for the other one)
        const ida = s.id[oi], idb = s.id[oj];
        const lowerIsI = ida < idb;
        const key = lowerIsI ? `${ida}:${idb}` : `${idb}:${ida}`;
        let shear: V3 = [0, 0, 0];
        if (hist) {
          const v = this.shear.get(key);
          if (v) shear = lowerIsI ? v : [-v[0], -v[1], -v[2]];
        }
        const res = roundedPolygonPair([bi, bj], { ...params, kn, kna, kt: this.p.get('kt', ti, type[j]), history: hist, shear, dt, update });
        if (hist && res.shear) {
          const sh = res.shear;
          this.shear.set(key, lowerIsI ? sh : [-sh[0], -sh[1], -sh[2]]);
          touched.add(key);
        }
        f[3 * i] += res.f[0][0]; f[3 * i + 1] += res.f[0][1]; f[3 * i + 2] += res.f[0][2];
        f[3 * j] += res.f[1][0]; f[3 * j + 1] += res.f[1][1]; f[3 * j + 2] += res.f[1][2];
        for (let d = 0; d < 3; d++) {
          tq[3 * i + d] += res.t[0][d];
          tq[3 * j + d] += res.t[1][d];
        }
        evdwl += res.energy;
        // ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0, facc, delx, dely, delz): facc the force of the pair on i
        if (pc.eatom || pc.vatom) tallyBodyPair(pc, i, j, res.energy, res.f[0], xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]);
      }
    }
    // a pair without a tangential spring this step loses its shear (native: touch = 0 zeroes the displacement)
    if (hist) for (const key of this.shear.keys()) if (!touched.has(key)) this.shear.delete(key);
    nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }

  /** Space-frame angular velocity of owned body o from its angular momentum (native angmom_to_omega). */
  private spaceOmega(s: PairCompute['s'], o: number, bd: Float64Array, B: number): V3 {
    if (!s.angmom) return [0, 0, 0];
    const q = s.quat!;
    const R = quatToMatrix(q[4 * o], q[4 * o + 1], q[4 * o + 2], q[4 * o + 3]);
    const m: V3 = [s.angmom[3 * o], s.angmom[3 * o + 1], s.angmom[3 * o + 2]];
    const I = [bd[B * o + 1], bd[B * o + 2], bd[B * o + 3]];
    const w: V3 = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      // the principal axis k is column k of R
      const e: V3 = [R[k], R[3 + k], R[6 + k]];
      const wb = I[k] === 0 ? 0 : (m[0] * e[0] + m[1] * e[1] + m[2] * e[2]) / I[k];
      w[0] += wb * e[0];
      w[1] += wb * e[1];
      w[2] += wb * e[2];
    }
    return w;
  }
}
