/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/GRANSURF/pair_surf_granular.cpp (prewalk_connections3d, walk_connections3d,
 * calculate_3d_forces, calculate_3d_edge_force, dist_away_from_edge, dist_away_from_2_tris) at commit b332f28.
 * Contributing authors of pair_surf_granular.cpp: Joel Clemmer (SNL), Dan Bolintineanu (SNL).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * The functions work on the contact list of one sphere (contact_surfs of the native pair) and the connectivity of
 * the triangles (connect3d, keyed by atom ID). Contacts are looked up by atom ID (contacts_map in the native code).
 */
import type { Connect3d } from './connect3d';
import {
  BIG, CONCAVE, CONVEX, EPSILON, EXTERNAL, FLAT, INTERNAL, OPPOSITE_SIDE, UNCONNECTED, flipSide, type ContactSurf,
} from './surf_contact';

type V3 = [number, number, number];

const dot = (a: V3, b: ArrayLike<number>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
const normv = (a: V3): V3 => {
  const l = len(a);
  return [a[0] / l, a[1] / l, a[2] / l];
};
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const scaleAdd = (s: number, a: V3, b: V3): V3 => [s * a[0] + b[0], s * a[1] + b[1], s * a[2] + b[2]];
const copy = (a: ArrayLike<number>): V3 => [a[0], a[1], a[2]];

/** The position of the lowest contact index among the given contact positions (std::set order of the native walk). */
const minByIndex = (keys: Iterable<number>, cs: ContactSurf[]): number => {
  let best = -1;
  for (const k of keys) if (best === -1 || cs[k].index < cs[best].index) best = k;
  return best;
};

/** The contact list of one sphere: tags to positions. */
export type PositionOf = Map<number, number>;

/** Shared state of the contact list of one sphere (the native contact_surfs with contacts_map). */
export interface Contacts3d {
  cs: ContactSurf[];
  posByTag: PositionOf;
  conn: Map<number, Connect3d>;
}

/** prewalk_connections3d: the consistent side of each contact's triangle normal (the sign of surfNorm). */
export function prewalk3d(ctx: Contacts3d): void {
  const { cs, posByTag, conn } = ctx;
  if (!cs.length) return;
  const walked = new Set<number>();
  const toWalk = new Map<number, number>();
  toWalk.set(0, cs[0].nside);
  while (toWalk.size) {
    const pos = minByIndex(toWalk.keys(), cs);
    const nsidej = toWalk.get(pos)!;
    toWalk.delete(pos);
    walked.add(pos);
    const c = cs[pos];
    if (nsidej === OPPOSITE_SIDE) for (let a = 0; a < 3; a++) c.surfNorm[a] = -c.surfNorm[a];
    const cn = conn.get(c.tag)!;
    const nbrs = [...cn.e[0], ...cn.e[1], ...cn.e[2], ...cn.c[0], ...cn.c[1], ...cn.c[2]];
    for (const nb of nbrs) {
      const kpos = posByTag.get(nb.tag);
      if (kpos === undefined) continue;
      if (!walked.has(kpos) && !toWalk.has(kpos)) {
        const nsidek = nsidej === OPPOSITE_SIDE ? flipSide(nb.nside) : nb.nside;
        cs[kpos].nside = nsidek;
        toWalk.set(kpos, nsidek);
      }
    }
    if (toWalk.size === 0) {
      for (let nc = 0; nc < cs.length; nc++) if (!walked.has(nc)) toWalk.set(nc, cs[nc].nside);
    }
  }
}

/** walk_connections3d: the flat-connected contacts of the first unprocessed contact, in composite order. */
export function walk3d(ctx: Contacts3d, processed: Set<number>, start: number): number[] {
  const { cs, posByTag, conn } = ctx;
  const composite: number[] = [];
  const toWalk = new Set<number>([start]);
  const toAdd = new Set<number>();
  while (toWalk.size) {
    const n = minByIndex(toWalk, cs);
    toWalk.delete(n);
    processed.add(n);
    composite.push(n);
    const c = cs[n];
    const jflag = c.flag;
    const cn = conn.get(c.tag)!;
    // edges: 0 for edge 12 (flags -1, -4, -5), 1 for edge 23 (-2, -5, -6), 2 for edge 31 (-3, -4, -6)
    const edgeAtJoint = [jflag === -1 || jflag === -4 || jflag === -5, jflag === -2 || jflag === -5 || jflag === -6,
      jflag === -3 || jflag === -4 || jflag === -6];
    for (let which = 0; which < 3; which++) {
      for (const nb of cn.e[which]) {
        const m = posByTag.get(nb.tag);
        if (m === undefined) continue;
        let aflag = nb.aflag;
        if (c.nside === OPPOSITE_SIDE) aflag = aflag === CONVEX ? CONCAVE : CONVEX;
        if (nb.fflag === FLAT) {
          if (c.type === cs[m].type && !processed.has(m)) toAdd.add(m);
        } else if (aflag === CONVEX) {
          if (c.overlap > cs[m].overlap - EPSILON) cs[m].convexIndex = c.index;
        }
        if (edgeAtJoint[which]) {
          c.cindex.push(nb.tag);
          c.cwhich.push(which);
          c.caflag.push(aflag);
        }
      }
    }
    // corners 1, 2, 3 (flags -4, -5, -6)
    const cornerAtJoint = [jflag === -4, jflag === -5, jflag === -6];
    for (let which = 0; which < 3; which++) {
      for (const nb of cn.c[which]) {
        const m = posByTag.get(nb.tag);
        if (m === undefined) continue;
        if (nb.fflag === FLAT) {
          if (c.type === cs[m].type && !processed.has(m)) toAdd.add(m);
        }
        if (cornerAtJoint[which]) {
          c.cindex.push(nb.tag);
          c.cwhich.push(-1);
          c.caflag.push(-1);
        }
      }
    }
    for (const k of [...toAdd].sort((a, b) => cs[a].index - cs[b].index)) {
      if (c.convexIndex !== -1) cs[k].convexIndex = c.index;
      toWalk.add(k);
    }
    toAdd.clear();
  }
  return composite;
}

/** calculate_3d_edge_force: the force direction of a contact on a convex or concave triangle edge. */
function edgeForce(aflag: number, jnorm: V3, knorm: V3, drperp: V3): V3 {
  if (aflag === CONCAVE) return copy(jnorm);
  const dotjk = dot(jnorm, knorm);
  const dotjr = dot(jnorm, drperp);
  const dotkr = dot(knorm, drperp);
  let dotmin = Math.min(dotjk, dotjr);
  dotmin = Math.min(dotmin, dotkr);
  if (dotmin === dotjk) return copy(drperp);
  if (dotmin === dotjr) return copy(knorm);
  return copy(jnorm);
}

/** dist_away_from_edge: planar distance of dr from the edge pt1-pt2 (pt3 is the third corner); -1 when anti-aligned. */
function distAwayFromEdge(pt1: V3, pt2: V3, pt3: V3, jnorm: V3, dr: V3): number {
  const edge = sub(pt1, pt2);
  const edgeNorm = normv(cross(jnorm, edge));
  const p3line = sub(pt3, pt1);
  let d = dot(edgeNorm, p3line);
  let signFlip = 1.0;
  if (d > 0.0) signFlip = -1.0;
  d = signFlip * dot(dr, edgeNorm);
  if (d > -EPSILON) return d;
  return -1;
}

/** dist_away_from_2_tris: distance of dr outside the planes of two triangle normals. */
function distAway2Tris(jnorm: V3, knorm: V3, dr: V3): number {
  const dotjk = dot(jnorm, knorm);
  const dotjr = dot(jnorm, dr);
  const dotkr = dot(knorm, dr);
  let dotmin = Math.min(dotjk, dotjr);
  dotmin = Math.min(dotmin, dotkr);
  if (dotmin === dotjk) return dot(cross(jnorm, knorm), dr);
  const residual = scaleAdd(-dotjr, jnorm, dr);
  return len(residual);
}

/** Corner-point indices of the edge tests of calculate_3d_forces for corner flags -4, -5, -6. */
const CORNER_WHICH: Record<number, [number, number]> = { [-4]: [0, 2], [-5]: [0, 1], [-6]: [2, 1] };

/**
 * calculate_3d_forces: the overlap and the force direction of each composite contact of a triangle surface.
 * Returns the largest overlap; sets the weights and drForce of the contacts (the pair then forms the force).
 */
export function calculate3d(ctx: Contacts3d, composite: number[]): number {
  const { cs, posByTag, conn } = ctx;
  let maxOverlap = -BIG;
  let maxOverlapExt = -BIG;
  let ucFlag = false;
  for (const n of composite) {
    const c = cs[n];
    if (c.convexIndex !== -1) c.overlap = 0.0;
    maxOverlap = Math.max(maxOverlap, c.overlap);
    if (c.external === EXTERNAL) maxOverlapExt = Math.max(maxOverlapExt, c.overlap);
    if (c.external === UNCONNECTED) ucFlag = true;
  }
  if (maxOverlap < EPSILON) return maxOverlap;
  let wExt = 0.0;
  if (maxOverlapExt !== -BIG) {
    wExt = maxOverlapExt / maxOverlap;
    wExt *= wExt;
  }
  const wInt = 1.0 - wExt;
  // the two most aligned neighbours across the contact's edges (ck1, ck2)
  for (const n of composite) {
    const c = cs[n];
    const jnorm = copy(c.surfNorm);
    let which1 = -1, which2 = -1;
    if (CORNER_WHICH[c.flag]) [which1, which2] = CORNER_WHICH[c.flag];
    let maxDot1 = -2.0, maxDot2 = -2.0;
    c.ck1 = c.ck2 = -1;
    for (let i = 0; i < c.cindex.length; i++) {
      const k = c.cindex[i];
      const m = posByTag.get(k)!;
      const d = dot(jnorm, cs[m].surfNorm);
      if (c.flag > -4) {
        if (d > maxDot1) {
          c.ck1 = k;
          c.caflag1 = c.caflag[i];
          maxDot1 = d;
        }
      } else {
        if (c.cwhich[i] === which1 && d > maxDot1) {
          c.ck1 = k;
          c.caflag1 = c.caflag[i];
          maxDot1 = d;
        }
        if (c.cwhich[i] === which2 && d > maxDot2) {
          c.ck2 = k;
          c.caflag2 = c.caflag[i];
          maxDot2 = d;
        }
      }
    }
  }
  // the connection weight of unconnected edges: how far the contact lies out of the plane of its neighbours
  let wConnect = 1.0;
  if (ucFlag) {
    let maxDistUc = 0.0;
    let minDistC = BIG;
    for (const n of composite) {
      const c = cs[n];
      const flag = c.flag;
      if (flag === 1) {
        minDistC = 0.0;
        break;
      }
      let which1 = -1, which2 = -1;
      if (CORNER_WHICH[flag]) [which1, which2] = CORNER_WHICH[flag];
      else which1 = -(1 + flag);
      const rmag = c.rmag;
      const dr = copy(c.dr);
      const jnorm = copy(c.surfNorm);
      const corners = c.corners;
      if (c.external === UNCONNECTED) {
        const jc = conn.get(c.tag)!;
        if (flag < -3 && jc.externalEdge[which1] === UNCONNECTED && jc.externalEdge[which2] === UNCONNECTED) {
          maxDistUc = Math.max(maxDistUc, rmag);
        } else {
          if (jc.externalEdge[0] === UNCONNECTED && (flag === -1 || flag === -4 || flag === -5)) {
            const d = distAwayFromEdge(corners[0], corners[1], corners[2], jnorm, dr);
            if (d !== -1) maxDistUc = Math.max(maxDistUc, d * rmag);
          }
          if (jc.externalEdge[1] === UNCONNECTED && (flag === -2 || flag === -5 || flag === -6)) {
            const d = distAwayFromEdge(corners[1], corners[2], corners[0], jnorm, dr);
            if (d !== -1) maxDistUc = Math.max(maxDistUc, d * rmag);
          }
          if (jc.externalEdge[2] === UNCONNECTED && (flag === -3 || flag === -4 || flag === -6)) {
            const d = distAwayFromEdge(corners[0], corners[2], corners[1], jnorm, dr);
            if (d !== -1) maxDistUc = Math.max(maxDistUc, d * rmag);
          }
        }
      }
      if (which1 !== -1 && c.ck1 !== -1) {
        const knorm = copy(cs[posByTag.get(c.ck1)!].surfNorm);
        minDistC = Math.min(minDistC, distAway2Tris(jnorm, knorm, dr) * rmag);
      }
      if (which2 !== -1 && c.ck2 !== -1) {
        const knorm = copy(cs[posByTag.get(c.ck2)!].surfNorm);
        minDistC = Math.min(minDistC, distAway2Tris(jnorm, knorm, dr) * rmag);
      }
    }
    if (maxDistUc > 0.0) wConnect = Math.max(0.0, Math.min(1.0, 1.0 - Math.min(minDistC, maxDistUc) / maxOverlap));
  }
  // the weights and the force directions of each contact
  for (const n of composite) {
    const c = cs[n];
    const jnorm = copy(c.surfNorm);
    const dr = copy(c.dr);
    const overlap = c.overlap;
    const rmag = c.rmag;
    const flag = c.flag;
    const external = c.external;
    if (external === EXTERNAL) c.weight = wExt;
    else if (external === INTERNAL) c.weight = wInt;
    if (external !== UNCONNECTED) {
      let wInPlane = 1.0;
      if (wConnect < 1.0) {
        const d = rmag * dot(jnorm, dr);
        if (d < rmag) wInPlane = Math.max(0.0, Math.min(1.0, d / (rmag * (1.0 - wConnect))));
      }
      c.weight *= wConnect * wInPlane;
    }
    if (overlap < EPSILON) continue;
    if (flag === 1) continue;
    if (flag < 0 && flag > -4) {
      if (c.ck1 !== -1) {
        const knorm = copy(cs[posByTag.get(c.ck1)!].surfNorm);
        c.drForce = edgeForce(c.caflag1, jnorm, knorm, dr);
      }
    }
    if (flag < -3) {
      let pt: number, pt1: number, pt2: number, which1: number, which2: number;
      if (flag === -4) {
        pt = 0; pt1 = 1; pt2 = 2; which1 = 0; which2 = 2;
      } else if (flag === -5) {
        pt = 1; pt1 = 0; pt2 = 2; which1 = 0; which2 = 1;
      } else {
        pt = 2; pt1 = 0; pt2 = 1; which1 = 2; which2 = 1;
      }
      const jc = conn.get(c.tag)!;
      const external1 = jc.externalEdge[which1];
      const external2 = jc.externalEdge[which2];
      const corners = c.corners;
      const line1 = normv(sub(corners[pt], corners[pt1]));
      const line2 = normv(sub(corners[pt], corners[pt2]));
      let d = dot(dr, jnorm);
      const drInPlane0 = scaleAdd(-d, jnorm, dr);
      const distIp = len(drInPlane0);
      if (distIp < EPSILON) {
        c.drForce = copy(jnorm);
        return maxOverlap;
      }
      const drInPlane = drInPlane0.map((v) => v / distIp) as V3;
      const dot1a = dot(line1, dr);
      const dot2a = dot(line2, dr);
      const dot1ip = Math.max(0.0, dot(line1, drInPlane));
      const dot2ip = Math.max(0.0, dot(line2, drInPlane));
      if (dot1ip < EPSILON && dot2ip < EPSILON) {
        c.drForce = copy(dr);
        return maxOverlap;
      }
      let fntot: V3 = [0, 0, 0];
      let w1ip = 1.0, w2ip = 1.0;
      d = dot(dr, line1);
      const dr1 = normv(scaleAdd(-d, line1, dr));
      let fn1 = copy(dr1);
      d = dot(dr, line2);
      const dr2 = normv(scaleAdd(-d, line2, dr));
      let fn2 = copy(dr2);
      let dot1xp = -1, dot2xp = -1;
      if (c.ck1 !== -1) {
        const m = posByTag.get(c.ck1)!;
        const knorm = copy(cs[m].surfNorm);
        fn1 = edgeForce(c.caflag1, jnorm, knorm, dr1);
        if (c.caflag1 !== CONCAVE && cs[m].convexIndex !== -1) {
          const normave = normv([jnorm[0] + knorm[0], jnorm[1] + knorm[1], jnorm[2] + knorm[2]]);
          dot1xp = dot(dr1, normave);
        }
      }
      if (ucFlag && external1 !== UNCONNECTED && wConnect < 1.0) {
        const dd = rmag * dot(jnorm, dr);
        if (dd < rmag) w1ip = Math.max(0.0, Math.min(1.0, dd / (rmag * (1.0 - wConnect))));
      }
      if (c.ck2 !== -1) {
        const m = posByTag.get(c.ck2)!;
        const knorm = copy(cs[m].surfNorm);
        fn2 = edgeForce(c.caflag2, jnorm, knorm, dr2);
        if (c.caflag2 !== CONCAVE && cs[m].convexIndex !== -1) {
          const normave = normv([jnorm[0] + knorm[0], jnorm[1] + knorm[1], jnorm[2] + knorm[2]]);
          dot2xp = dot(dr2, normave);
        }
      }
      if (ucFlag && external2 !== UNCONNECTED && wConnect < 1.0) {
        const dd = rmag * dot(jnorm, dr);
        if (dd < rmag) w2ip = Math.max(0.0, Math.min(1.0, dd / (rmag * (1.0 - wConnect))));
      }
      let w1 = 1 - dot1a;
      let w2 = 1 - dot2a;
      w1 *= dot2ip;
      w2 *= dot1ip;
      if (dot1xp !== -1) w2 *= 1 - dot1xp;
      if (dot2xp !== -1) w1 *= 1 - dot2xp;
      if (dot(fn1, line2) < 0.0 && (external1 !== UNCONNECTED || wConnect === 1.0)) fn1 = copy(jnorm);
      if (dot(fn2, line1) < 0.0 && (external2 !== UNCONNECTED || wConnect === 1.0)) fn2 = copy(jnorm);
      if (external === UNCONNECTED) {
        if (external1 !== UNCONNECTED) {
          const wtmp = wConnect * w1ip;
          fn1 = [0, 1, 2].map((a) => wtmp * fn1[a] + (1.0 - wtmp) * dr1[a]) as V3;
        }
        if (external2 !== UNCONNECTED) {
          const wtmp = wConnect * w2ip;
          fn2 = [0, 1, 2].map((a) => wtmp * fn2[a] + (1.0 - wtmp) * dr2[a]) as V3;
        }
      }
      fntot = scaleAdd(w1, fn1, fntot);
      fntot = scaleAdd(w2, fn2, fntot);
      if (w1 < EPSILON && w2 < EPSILON) {
        if (dot1xp !== -1 && dot2xp !== -1) fntot = copy(dr);
        else if (dot1xp !== -1) fntot = copy(fn1);
        else if (dot2xp !== -1) fntot = copy(fn2);
        else fntot = [fn1[0] + fn2[0], fn1[1] + fn2[1], fn1[2] + fn2[2]];
      }
      c.drForce = normv(fntot);
    }
  }
  return maxOverlap;
}
