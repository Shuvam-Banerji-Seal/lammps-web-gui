/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/GRANSURF/pair_surf_granular.cpp (PairSurfGranular::compute: the per-sphere
 * sort, prewalk_connections2d, walk_connections2d, calculate_2d_forces and the composite geometry),
 * src/GRANSURF/fix_surface_global.cpp (FixSurfaceGlobal::post_force: the same sequence for the global surfaces),
 * src/GRANSURF/fix_surface.cpp (contact_compare, contact_presort, contact_sort).
 * Contributing authors of pair_surf_granular.cpp: Joel Clemmer (SNL), Dan Bolintineanu (SNL).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * The walks and the force directions of the contacts of ONE sphere. They do not depend on the contact history or on
 * the forces, so the walks of all composites can be done before the forces are added (the native loop interleaves
 * them, with the same result). Shared by pair surf/granular (fix surface/local surfaces) and fix surface/global.
 */
import type { Connect2d } from './connect2d';
import type { Connect3d } from './connect3d';
import { calculate3d, prewalk3d, walk3d, type Contacts3d } from './forces3d';
import {
  BIG, CONCAVE, CONVEX, EPSILON, FLAT, OPPOSITE_SIDE, flipSide, contactLess, type ContactSurf,
} from './surf_contact';

type V3 = [number, number, number];

/** One composite contact of a sphere that gives a force: the contact position, its composite and its overlap. */
export interface SolvedContact {
  n: number;
  composite: number[];
  maxOverlap: number;
}

/**
 * Sorts the contacts of one sphere (in place) and returns the composite contacts that give a force, in the native
 * processing order. is3: triangles (conn3) or lines (conn2). Contacts with an overlap below EPSILON give no force.
 */
export function solveSphereContacts(
  cs: ContactSurf[], is3: boolean, conn2: Map<number, Connect2d> | null, conn3: Map<number, Connect3d> | null,
): SolvedContact[] {
  cs.sort((a, b) => (contactLess(a, b, false) ? -1 : contactLess(b, a, false) ? 1 : 0));
  let posByTag = indexByTag(cs);
  if (is3) prewalk3d({ cs, posByTag, conn: conn3! });
  else prewalk2d(cs, posByTag, conn2!);
  cs.sort((a, b) => (contactLess(a, b, true) ? -1 : contactLess(b, a, true) ? 1 : 0));
  posByTag = indexByTag(cs);
  const processed = new Set<number>();
  const out: SolvedContact[] = [];
  for (let n = 0; n < cs.length; n++) {
    if (processed.has(n)) continue;
    let composite: number[];
    let maxOverlap: number;
    if (is3) {
      const ctx: Contacts3d = { cs, posByTag, conn: conn3! };
      composite = walk3d(ctx, processed, n);
      maxOverlap = calculate3d(ctx, composite);
    } else {
      composite = walk2d(cs, posByTag, conn2!, processed, n);
      maxOverlap = calc2d(cs, posByTag, composite);
    }
    if (maxOverlap < EPSILON) continue;
    out.push({ n, composite, maxOverlap });
  }
  return out;
}

/**
 * Direction and length of the contact from the sphere centre to the surface (the dr of the native post_force after
 * the composite average): radi minus the maximum overlap along the overlap-weighted normal of the composite.
 */
export function contactDr(cs: ContactSurf[], composite: number[], n: number, maxOverlap: number, radi: number): V3 {
  const dr: V3 = [0, 0, 0];
  if (composite.length > 1) {
    for (const m of composite) {
      const c = cs[m];
      if (c.overlap < EPSILON) continue;
      const w = c.overlap * c.weight;
      for (let a = 0; a < 3; a++) dr[a] += w * c.drForce[a];
    }
    const len = Math.hypot(dr[0], dr[1], dr[2]);
    for (let a = 0; a < 3; a++) dr[a] = (dr[a] / len) * (radi - maxOverlap);
  } else {
    for (let a = 0; a < 3; a++) dr[a] = cs[n].drForce[a] * (radi - maxOverlap);
  }
  return dr;
}

/** Map from the tag (surface ID) to the position in the contact list. */
export function indexByTag(cs: ContactSurf[]): Map<number, number> {
  const m = new Map<number, number>();
  for (let n = 0; n < cs.length; n++) if (!m.has(cs[n].tag)) m.set(cs[n].tag, n);
  return m;
}

/** The position with the smallest contact index (the std::map / std::set order of the native walks). */
function minByIndex(keys: Iterable<number>, cs: ContactSurf[]): number {
  let best = -1;
  for (const k of keys) if (best === -1 || cs[k].index < cs[best].index) best = k;
  return best;
}

/** prewalk_connections2d: the consistent side of each contact's element normal (the sign of surfNorm). */
export function prewalk2d(cs: ContactSurf[], posByTag: Map<number, number>, conn: Map<number, Connect2d>): void {
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
    for (const nbr of [...cn.p[0], ...cn.p[1]]) {
      const kpos = posByTag.get(nbr.tag);
      if (kpos === undefined) continue;
      if (!walked.has(kpos) && !toWalk.has(kpos)) {
        const nsidek = nsidej === OPPOSITE_SIDE ? flipSide(nbr.nside) : nbr.nside;
        cs[kpos].nside = nsidek;
        toWalk.set(kpos, nsidek);
      }
    }
    if (toWalk.size === 0) {
      for (let nc = 0; nc < cs.length; nc++) if (!walked.has(nc)) toWalk.set(nc, cs[nc].nside);
    }
  }
}

/** walk_connections2d: the flat-connected contacts of the first unprocessed contact, in composite order. */
export function walk2d(
  cs: ContactSurf[], posByTag: Map<number, number>, conn: Map<number, Connect2d>, processed: Set<number>, start: number,
): number[] {
  const composite: number[] = [];
  const toWalk = new Set<number>([start]);
  const toAdd = new Set<number>();
  while (toWalk.size) {
    const n = minByIndex(toWalk.values(), cs);
    toWalk.delete(n);
    processed.add(n);
    composite.push(n);
    const c = cs[n];
    const jflag = c.flag;
    const cn = conn.get(c.tag)!;
    for (let side = 0; side < 2; side++) {
      const contactAtJoint = side === 0 ? jflag === -1 : jflag === -2;
      for (const nbr of cn.p[side]) {
        const m = posByTag.get(nbr.tag);
        if (m === undefined) continue;
        let aflag = nbr.aflag;
        if (c.nside === OPPOSITE_SIDE) aflag = aflag === CONVEX ? CONCAVE : CONVEX;
        if (nbr.fflag === FLAT) {
          if (c.type === cs[m].type && !processed.has(m)) toAdd.add(m);
        } else if (aflag === CONVEX) {
          if (c.overlap > cs[m].overlap - EPSILON) cs[m].convexIndex = c.index;
        }
        if (contactAtJoint) {
          c.cindex.push(nbr.tag);
          c.caflag.push(aflag);
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

/** calculate_2d_forces: the overlap and the force direction of each composite contact; returns the largest overlap. */
export function calc2d(cs: ContactSurf[], posByTag: Map<number, number>, composite: number[]): number {
  let maxOverlap = -BIG;
  let maxOverlapExt = -BIG;
  for (const n of composite) {
    const c = cs[n];
    if (c.convexIndex !== -1) c.overlap = 0.0;
    maxOverlap = Math.max(maxOverlap, c.overlap);
    if (c.external) maxOverlapExt = Math.max(maxOverlapExt, c.overlap);
  }
  if (maxOverlap < EPSILON) return maxOverlap;
  let wExt = 0.0;
  if (maxOverlapExt !== -BIG) {
    wExt = maxOverlapExt / maxOverlap;
    wExt *= wExt;
  }
  const wInt = 1.0 - wExt;
  for (const n of composite) {
    const c = cs[n];
    c.weight = c.external ? wExt : wInt;
    if (c.overlap < EPSILON) continue;
    if (c.flag === 1) continue;
    let maxDot = -2;
    let ck = -1;
    let caflag = -1;
    for (let i = 0; i < c.cindex.length; i++) {
      const m = posByTag.get(c.cindex[i])!;
      const dot = c.surfNorm[0] * cs[m].surfNorm[0] + c.surfNorm[1] * cs[m].surfNorm[1] + c.surfNorm[2] * cs[m].surfNorm[2];
      if (dot > maxDot) {
        ck = c.cindex[i];
        caflag = c.caflag[i];
        maxDot = dot;
      }
    }
    if (ck === -1) continue;
    if (caflag === CONCAVE) {
      c.drForce = [c.surfNorm[0], c.surfNorm[1], c.surfNorm[2]];
      continue;
    }
    // the end point of this element where the contact sits, and the direction to the other element's far end
    const pt = c.flag === -1 ? 0 : 1;
    const ptx = c.ends[pt];
    const mk = posByTag.get(ck)!;
    let kline = [cs[mk].ends[0][0] - ptx[0], cs[mk].ends[0][1] - ptx[1], cs[mk].ends[0][2] - ptx[2]];
    if (kline[0] * kline[0] + kline[1] * kline[1] + kline[2] * kline[2] < EPSILON) {
      kline = [cs[mk].ends[1][0] - ptx[0], cs[mk].ends[1][1] - ptx[1], cs[mk].ends[1][2] - ptx[2]];
    }
    const dot = c.dr[0] * kline[0] + c.dr[1] * kline[1] + c.dr[2] * kline[2];
    if (dot > 0) c.drForce = [cs[mk].surfNorm[0], cs[mk].surfNorm[1], cs[mk].surfNorm[2]];
  }
  return maxOverlap;
}
