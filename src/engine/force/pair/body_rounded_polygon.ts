import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';
import { BODY_RECORD, bodyRecordOffset } from '../../bodies';
import { quatToMatrix } from '../../compute/asphere';

/*
 * pair_style body/rounded/polygon (wave 50; registered in registry/body48.ts). Implemented: the elastic normal law,
 * the cohesive normal law with the delta_ua scaling, the contact torques, the normal damping c_n at every contact
 * and the tangential damping c_t at every overlapping contact, with the velocities and spin of the bodies. Refused by
 * name: friction (mu != 0) and the history keyword. Friction is implemented in roundedPairForce (checked against
 * native in one-contact configurations) but not registered: the two-contact collision diverges from native at step 69.
 *
 * docs.lammps.org/pair_body_rounded_polygon.html: "pair_style body/rounded/polygon c_n c_t mu delta_ua cutoff
 * keyword" and "The forces between vertex-vertex, vertex-edge, and edge-edge overlaps are given by:" (the
 * piecewise F_n with the elastic term -k_n delta_n and the cohesive term -j_a k_na (r_c - delta_n) for
 * delta_n <= 0, and -k_na (r_c - delta_n) for 0 < delta_n <= r_c, with delta_n = d - (R_i + R_j)). "Here, positive
 * values of F_n are repulsive." "the cohesive force at the contacts is scaled by the factor" j_a >= 1, which grows
 * with the length of the contact region (see delta_ua). "The elastic force k_n delta_n is not scaled."
 * "Each vertex of one particle interacts with the point of the other particle nearest to it, which lies inside an
 * edge or at a vertex." "Following Fraige, the contact forces are applied at up to two contacts, the two contacts
 * farthest apart, whose distance is the length of the contact region."
 *
 * Measured with native LAMMPS (black box), with two rounded squares (edge 4, vertices at +-2, diameter 0.5):
 *   - one contact (vertex-vertex) or each of two vertex-edge contacts: the elastic force is k_n * ov per
 *     contact (ov = -delta_n), e.g. 100 * 0.1 per contact; the pair energy is k_n ov^2 / 2 per contact
 *     (1.0 for two contacts at ov 0.1, k_n 100; the thermo pe of lj units is per atom, half of it);
 *   - the cohesive force per contact is k_na (r_c - delta_n) (3.6 = 2 * 2 * 0.9 for two contacts at delta_n 0.1),
 *     the energy -k_na (r_c - delta_n)^2 / 2 per contact (-1.62 for two contacts);
 *   - the energy does not depend on delta_ua; the cohesive force of a pair with two overlapping contacts is scaled
 *     by j_a = max(1, L / delta_ua), with L = (1/2) * |(v1 - v2) . t| where v1, v2 are the vertices of the two
 *     overlapping contacts and t is the unit tangent perpendicular to the contact normal of the first one.
 *     Measured: j_a * 1 at 0.01 for a face contact (L 0.25 at y0 3.5, the cohesion total 110 = 4.4 * 25),
 *     and for a tilted pair (rotation 0.05, body2 at (4.3, 3.0)): cohesion -251.03 at delta_ua 0.01 and -8.368 at
 *     0.3, with L = 0.5581 (j_a 55.81 and 1.860);
 *   - a single overlapping contact, or contacts with delta_n > 0, are not scaled (j_a = 1), measured at
 *     (4.4, 3.0) rotation 0.1: -2.5853 at delta_ua 0.01 and 0.3;
 *   - the normal force is applied at the contact, and its torque is r x F with r the tangential offset of the
 *     contact from the centre of mass: the tilted pair above gives tqz 21.6156580982125 on body 2, matching the
 *     engine's torque to 1e-12.
 *
 * Contact geometry: the surface separation of a vertex and an edge (or vertex) is delta_n = d - diameter, with d
 * the distance of the two centres (vertex centre to the nearest point of the other polygon's boundary); both
 * rounded surfaces have radius diameter/2.
 */

export interface RoundedBody {
  /** Centre of mass (x, y). */
  c: [number, number];
  /** Velocity (x, y) of the centre of mass (vel) and angular velocity about z (0 when not given). */
  vel?: [number, number];
  w?: number;
  /** World-frame vertex positions (x, y) in vertex order (a closed polygon). */
  v: [number, number][];
}

export interface RoundedContact {
  /** 'VE' vertex-edge, 'VV' vertex-vertex. */
  kind: 'VE' | 'VV';
  /** 0: the vertex belongs to body A; 1: to body B. */
  owner: 0 | 1;
  /** Vertex of the owner. */
  pv: [number, number];
  /** Point of the other body's boundary nearest to pv. */
  pq: [number, number];
  /** Unit normal from the other body toward the owner. */
  n: [number, number];
  /** Surface separation delta_n = d - diameter. */
  g: number;
  /** Edge of the other body containing pq (its two vertices). */
  eY: [[number, number], [number, number]];
  /** Vertex index on the owner. */
  iv: number;
}

const near = (p: [number, number], a: [number, number], b: [number, number]): { s: number; q: [number, number] } => {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  const s = Math.min(1, Math.max(0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
  return { s, q: [a[0] + s * dx, a[1] + s * dy] };
};

/** Nearest point of the boundary of polygon v to p: distance, edge index k (edge k -> k+1), parameter s, point. */
const nearestBoundary = (p: [number, number], v: [number, number][]): { d: number; k: number; s: number; q: [number, number] } => {
  let best = { d: Infinity, k: 0, s: 0, q: [0, 0] as [number, number] };
  const n = v.length;
  for (let k = 0; k < n; k++) {
    const { s, q } = near(p, v[k], v[(k + 1) % n]);
    const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
    if (d < best.d - 1e-14) best = { d, k, s, q };
  }
  return best;
};

/**
 * Contacts between two rounded polygons (src/engine/force/pair/body_rounded_polygon.ts header). Each vertex of one
 * body interacts with the nearest point of the other body's boundary; a vertex that is the nearest vertex of the
 * other body and vice versa is one vertex-vertex contact, counted once (from body A).
 */
export const roundedContacts = (A: RoundedBody, B: RoundedBody, diameter: number, rc: number): RoundedContact[] => {
  const out: RoundedContact[] = [];
  for (const [X, Y, owner] of [[A, B, 0], [B, A, 1]] as const) {
    for (let iv = 0; iv < X.v.length; iv++) {
      const p = X.v[iv];
      const nb = nearestBoundary(p, Y.v);
      if (nb.d === 0) continue;
      const atVertex = nb.s <= 1e-12 || nb.s >= 1 - 1e-12;
      let kind: 'VE' | 'VV' = 'VE';
      if (atVertex) {
        const yk = nb.s <= 1e-12 ? nb.k : (nb.k + 1) % Y.v.length;
        const back = nearestBoundary(Y.v[yk], X.v);
        if (Math.hypot(back.q[0] - p[0], back.q[1] - p[1]) > 1e-9) continue; // not mutual
        if (owner === 1) continue; // counted once, from body A
        kind = 'VV';
      }
      const g = nb.d - diameter;
      if (!(g < rc)) continue;
      const eY: [[number, number], [number, number]] = [Y.v[nb.k], Y.v[(nb.k + 1) % Y.v.length]];
      out.push({ kind, owner, pv: p, pq: nb.q, n: [(p[0] - nb.q[0]) / nb.d, (p[1] - nb.q[1]) / nb.d], g, eY, iv });
    }
  }
  return out;
};

/** Per-body forces of one pair: force on A and B (2d), torques (z) on A and B, and the pair energy. */
export interface RoundedPairResult {
  fA: [number, number];
  fB: [number, number];
  tzA: number;
  tzB: number;
  energy: number;
  /** Number of contacts that carried a force (at most two). */
  nContacts: number;
  /** j_a applied to the cohesion of the overlapping contacts (1 when not scaled). */
  ja: number;
}

/** Damping and friction coefficients of pair_style body/rounded/polygon (c_n, c_t, mu). */
export interface RoundedDamping {
  cn: number;
  ct: number;
  mu: number;
}

/**
 * Normal forces and energy of two rounded polygons with the laws measured above. kn, kna: the k_n and k_na
 * coefficients; dua: delta_ua. Only the normal force is applied (no damping, no friction).
 */
export const roundedPairForce = (A: RoundedBody, B: RoundedBody, diameter: number, rc: number, kn: number, kna: number, dua: number, damp: RoundedDamping = { cn: 0, ct: 0, mu: 0 }): RoundedPairResult => {
  let cs = roundedContacts(A, B, diameter, rc);
  if (cs.length > 2) {
    let best = { d: -1, i: 0, j: 1 };
    for (let i = 0; i < cs.length; i++) {
      for (let j = i + 1; j < cs.length; j++) {
        const d = Math.hypot(cs[i].pv[0] - cs[j].pv[0], cs[i].pv[1] - cs[j].pv[1]);
        if (d > best.d) best = { d, i, j };
      }
    }
    cs = [cs[best.i], cs[best.j]];
  }
  const neg = cs.filter((c) => c.g < 0);
  let ja = 1;
  if (neg.length === 2) {
    const [c1, c2] = neg;
    const t: [number, number] = [-c1.n[1], c1.n[0]];
    const L = 0.5 * Math.abs((c1.pv[0] - c2.pv[0]) * t[0] + (c1.pv[1] - c2.pv[1]) * t[1]);
    ja = Math.max(1, L / dua);
  }
  const fA: [number, number] = [0, 0], fB: [number, number] = [0, 0];
  let tzA = 0, tzB = 0, energy = 0;
  // the contact with the largest overlap (Fraige: the friction acts there, see the header)
  let tangential: RoundedContact | null = null;
  for (const c of cs) if (c.g < 0 && (tangential === null || c.g < tangential.g)) tangential = c;
  for (const c of cs) {
    // contact point: midway between the vertex centre and the nearest boundary point (measured, see the header)
    const mid: [number, number] = [(c.pv[0] + c.pq[0]) / 2, (c.pv[1] + c.pq[1]) / 2];
    const elastic = c.g < 0 ? kn * -c.g : 0;
    const cohesive = -kna * (rc - c.g) * (c.g < 0 ? ja : 1);
    // force on A along the normal n (from B to A): repulsion positive
    const sA = c.owner === 0 ? 1 : -1;
    const Fn = elastic + cohesive;
    const on: [number, number] = [sA * Fn * c.n[0], sA * Fn * c.n[1]];
    if (c.g < 0 && (damp.cn !== 0 || damp.ct !== 0 || damp.mu !== 0)) {
      // relative velocity of the two contact points, v_A - v_B, with v_c = v + w x (mid - c)
      const vc = (b: RoundedBody): [number, number] => {
        const v = b.vel ?? [0, 0], w = b.w ?? 0;
        return [v[0] - w * (mid[1] - b.c[1]), v[1] + w * (mid[0] - b.c[0])];
      };
      const vA = vc(A), vB = vc(B);
      const vr: [number, number] = [vA[0] - vB[0], vA[1] - vB[1]];
      const t: [number, number] = [-c.n[1], c.n[0]];
      const vn = vr[0] * c.n[0] + vr[1] * c.n[1];
      const vt = vr[0] * t[0] + vr[1] * t[1];
      // -c_n v_n along n; -min(mu k_n |delta_n|, c_t |v_t|) v_t/|v_t| - c_t v_t along t (the friction term
      // needs c_t > 0; its limit uses the elastic force k_n |delta_n|, not the cohesion)
      // the normal damping acts at each contact; the single tangential force of the pair (friction and the
      // tangential damping) acts only at the contact with the largest overlap
      const dn = -damp.cn * vn;
      on[0] += dn * c.n[0];
      on[1] += dn * c.n[1];
      if (c.g < 0) {
        const fr = c === tangential ? Math.min(damp.mu * kn * -c.g, damp.ct * Math.abs(vt)) : 0;
        const ft = (vt !== 0 ? -fr * Math.sign(vt) : 0) - damp.ct * vt;
        // the damping and friction terms are even under a flip of n and t, so they need no owner sign
        on[0] += ft * t[0];
        on[1] += ft * t[1];
      }
    }
    const pA = mid, pB = mid;
    fA[0] += on[0]; fA[1] += on[1];
    fB[0] -= on[0]; fB[1] -= on[1];
    tzA += (pA[0] - A.c[0]) * on[1] - (pA[1] - A.c[1]) * on[0];
    tzB += (pB[0] - B.c[0]) * -on[1] - (pB[1] - B.c[1]) * -on[0];
    // energy: k_n ov^2 / 2 and -k_na (rc - g)^2 / 2 per contact (not scaled by j_a, no damping energy)
    if (c.g < 0) energy += (kn * c.g * c.g) / 2;
    energy += (-kna * (rc - c.g) * (rc - c.g)) / 2;
  }
  return { fA, fB, tzA, tzB, energy, nContacts: cs.length, ja };
};

/**
 * pair_style body/rounded/polygon c_n c_t mu delta_ua cutoff, pair_coeff k_n k_na. Refused by name (StyleError):
 * mu != 0 (friction, see the settings note) and the history keyword.
 */
export class PairBodyRoundedPolygon extends Pair {
  readonly name = 'body/rounded/polygon';
  virialFdotr = true;
  cutGlobal = 0;
  dua = 0;
  damp: RoundedDamping = { cn: 0, ct: 0, mu: 0 };
  bodyRadius = 0;
  p!: PairParams;

  settings(args: string[]): void {
    if (args.length < 5) throw new StyleError('Illegal pair_style body/rounded/polygon command');
    this.damp = { cn: parseNum(args[0], 'c_n'), ct: parseNum(args[1], 'c_t'), mu: parseNum(args[2], 'mu') };
    // friction (mu > 0) is not matched to native yet: the two-contact collision of w50bc_damped diverges at step 69 (see the notes)
    if (this.damp.mu !== 0) throw new StyleError('pair_style body/rounded/polygon: friction (mu != 0) is not implemented yet by the browser engine; c_n and c_t are');
    if (args.length > 5) throw new StyleError('pair_style body/rounded/polygon: the history keyword is not implemented yet by the browser engine');
    this.dua = parseNum(args[3], 'delta_ua');
    this.cutGlobal = parseNum(args[4], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style body/rounded/polygon cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['kn', 'kna']);
  }

  coeff(args: string[]): void {
    if (args.length !== 4) throw new StyleError('Incorrect args for pair coefficients');
    this.p.setRange(args[0], args[1], [parseNum(args[2], 'k_n'), parseNum(args[3], 'k_na')]);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) {
      if (!this.p.isSet(i, i) || !this.p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      this.p.setMixed(i, j, 'kn', Math.sqrt(this.p.get('kn', i, i) * this.p.get('kn', j, j)));
      this.p.setMixed(i, j, 'kna', Math.sqrt(this.p.get('kna', i, i) * this.p.get('kna', j, j)));
    }
    // neighbour cutoff: two centres closer than twice the largest body radius plus the surface cutoff
    return 2 * this.bodyRadius + this.cutGlobal;
  }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) return;
    const rec = s.custom.get(BODY_RECORD);
    if (!rec) throw new StyleError('Pair body/rounded/polygon requires atom style body rounded/polygon');
    const off = bodyRecordOffset('rounded/polygon');
    let rmax = 0;
    for (let o = 0; o < s.n; o++) {
      const nv = rec.data[rec.cols * o];
      let far = 0;
      for (let k = 0; k < nv; k++) {
        const dx = rec.data[rec.cols * o + off + 3 * k], dy = rec.data[rec.cols * o + off + 3 * k + 1];
        far = Math.max(far, Math.hypot(dx, dy));
      }
      rmax = Math.max(rmax, far + 0.5 * rec.data[rec.cols * o + 4]);
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
    // world-frame vertex offsets (dx, dy) of every owned body, from the body-frame record (x, y components)
    const offs: [number, number][][] = new Array(nlocal);
    const diam: number[] = new Array(nlocal);
    for (let o = 0; o < nlocal; o++) {
      const nv = bd[B * o];
      diam[o] = bd[B * o + 4];
      const R = quatToMatrix(s.quat[4 * o], s.quat[4 * o + 1], s.quat[4 * o + 2], s.quat[4 * o + 3]);
      const list2: [number, number][] = [];
      for (let k = 0; k < nv; k++) {
        const d0 = bd[B * o + off + 3 * k], d1 = bd[B * o + off + 3 * k + 1], d2 = bd[B * o + off + 3 * k + 2];
        list2.push([R[0] * d0 + R[1] * d1 + R[2] * d2, R[3] * d0 + R[4] * d1 + R[5] * d2]);
      }
      offs[o] = list2;
    }
    // velocity of each owned body's centre and its angular velocity about z (2d bodies): w = R I^-1 R^T L
    const vel: [number, number][] = new Array(nlocal);
    const spin: number[] = new Array(nlocal);
    for (let o = 0; o < nlocal; o++) {
      vel[o] = [s.v[3 * o], s.v[3 * o + 1]];
      spin[o] = 0;
      if (s.angmom) {
        const q = s.quat!;
        const R = quatToMatrix(q[4 * o], q[4 * o + 1], q[4 * o + 2], q[4 * o + 3]);
        const L = [s.angmom[3 * o], s.angmom[3 * o + 1], s.angmom[3 * o + 2]];
        const Lb = [0, 1, 2].map((k) => R[k] * L[0] + R[3 + k] * L[1] + R[6 + k] * L[2]);
        const I = [bd[B * o + 1], bd[B * o + 2], bd[B * o + 3]];
        const wb = [Lb[0] / I[0], Lb[1] / I[1], Lb[2] / I[2]];
        spin[o] = R[6] * wb[0] + R[7] * wb[1] + R[8] * wb[2];
      }
    }
    let evdwl = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const xi = x[3 * i], yi = x[3 * i + 1];
      const ti = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const xj = x[3 * j], yj = x[3 * j + 1];
        const t = ti * nt + type[j];
        const rsq = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (x[3 * i + 2] - x[3 * j + 2]) ** 2;
        if (rsq >= cutsq[t]) continue;
        if (jj >>> SBBITS !== 0 && (sL[jj >>> SBBITS] !== 1 || sC[jj >>> SBBITS] !== 1)) {
          throw new StyleError('pair_style body/rounded/polygon does not support special_bonds factors other than 1');
        }
        if (offs[oi].length === 0 || offs[oj].length === 0) continue;
        const kn = this.p.get('kn', ti, type[j]), kna = this.p.get('kna', ti, type[j]);
        const rc = this.cutGlobal;
        const diameter = diam[oi];
        const A = { c: [xi, yi] as [number, number], v: offs[oi].map(([dx, dy]) => [xi + dx, yi + dy] as [number, number]), vel: vel[oi], w: spin[oi] };
        const Bb = { c: [xj, yj] as [number, number], v: offs[oj].map(([dx, dy]) => [xj + dx, yj + dy] as [number, number]), vel: vel[oj], w: spin[oj] };
        const r = roundedPairForce(A, Bb, diameter, rc, kn, kna, this.dua, this.damp);
        f[3 * i] += r.fA[0]; f[3 * i + 1] += r.fA[1];
        f[3 * j] += r.fB[0]; f[3 * j + 1] += r.fB[1];
        tq[3 * i + 2] += r.tzA;
        tq[3 * j + 2] += r.tzB;
        evdwl += r.energy;
      }
    }
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
