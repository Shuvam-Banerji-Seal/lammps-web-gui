/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/fix_wall_gran_region.cpp (FixWallGranRegion::post_force, the region and history part of its
 * per-atom loop, update_contacts and init) at commit b332f28; the region objects of ../region (src/region*.cpp).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The native region side of fix wall/gran/region, as the hook the engine's FixWallGranGranular calls (lazy group
 * gpl_wall_gran_region, HOOKS word wall/gran/region). The engine keeps the contact force model (granular.ts
 * granularContact, native's GranularModel); this file gives it native's contacts: the region match, Region::surface's
 * contact list (delx, dely, delz, r, radius) in native's arithmetic, the wall velocity at each contact
 * (Region::velocity_contact) and native's per-atom shear history bookkeeping (ncontact, walls, history_many, c2r).
 *
 * c2r is one array of the fix, as native's: update_contacts sets c2r[m] = inew for the kept history slot m of new contact
 * inew, and the force loop reads history_many[i][c2r[ic]] for contact ic. Both are native's indexing (entries it does not
 * set keep the values of an earlier atom).
 */
import { BlockRegion, CompoundRegion, ConeRegion, PlaneRegion, SphereRegion, type Region } from '../../engine/region';
import { StyleError } from '../../engine/force/types';
import type { NativeRegion, NativeRegionEnv } from '../region/native_region';
import { NativeBlock } from '../region/region_block';
import { NativePlane, NativeSphere } from '../region/region_plane_sphere';
import { NativeCylinder } from '../region/region_cylinder';
import { NativeCone } from '../region/region_cone';
import { NativeCompound } from '../region/region_compound';

/** The engine side the native regions read. */
export interface RegionWallSys {
  regionEnv: { variable(name: string): number; remap?(p: number[]): void };
  region(id: string): Region;
}

/** The native region of every engine region of a session: shared by the walls (native's regions are global objects). */
const mirrors = new WeakMap<RegionWallSys, Map<Region, NativeRegion | null>>();

const applyOptions = (core: Region, r: NativeRegion): void => {
  r.interior = core.interior ? 1 : 0;
  for (const f of core.openFaces) { r.open_faces[f - 1] = 1; r.openflag = 1; }
  if (core.move) {
    r.moveflag = 1;
    [r.xstr, r.ystr, r.zstr] = core.move;
  }
  if (core.rotate) {
    const a = core.rotate.a ?? core.rotate.r;
    r.setRotate(core.rotate.theta, core.rotate.p[0], core.rotate.p[1], core.rotate.p[2], a[0], a[1], a[2]);
  }
  r.dynamic = r.moveflag || r.rotateflag ? 1 : 0;
};

/** The native region of an engine region, or null for a style this port does not have (the engine path is used). */
const mirrorOf = (sys: RegionWallSys, core: Region): NativeRegion | null => {
  let m = mirrors.get(sys);
  if (!m) mirrors.set(sys, (m = new Map()));
  if (m.has(core)) return m.get(core)!;
  const e = sys.regionEnv;
  const env: NativeRegionEnv = { variable: (n) => e.variable(n), remap: (p) => e.remap?.(p) };
  let r: NativeRegion | null = null;
  if (core instanceof BlockRegion) {
    const b = new NativeBlock(core.id, env, core.b);
    applyOptions(core, b);
    b.setup();
    r = b;
  } else if (core instanceof SphereRegion) {
    const s = new NativeSphere(core.id, env, core.c, core.r);
    applyOptions(core, s);
    s.setup();
    r = s;
  } else if (core instanceof PlaneRegion) {
    const p = new NativePlane(core.id, env, core.pt, core.nrm);
    applyOptions(core, p);
    p.setup();
    r = p;
  } else if (core instanceof ConeRegion) {
    const c = core.style === 'cylinder'
      ? new NativeCylinder(core.id, env, core.axis, core.c1, core.c2, core.radlo, core.lo, core.hi)
      : new NativeCone(core.id, env, core.axis, core.c1, core.c2, core.radlo, core.radhi, core.lo, core.hi);
    applyOptions(core, c);
    c.setup();
    r = c;
  } else if (core instanceof CompoundRegion) {
    const subs: NativeRegion[] = [];
    for (const id of core.members) {
      const s = mirrorOf(sys, sys.region(id));
      if (!s) { m.set(core, null); return null; }
      subs.push(s);
    }
    const c = new NativeCompound(core.id, core.style as 'intersect' | 'union', env, subs);
    c.interior = core.interior ? 1 : 0;
    for (const f of core.openFaces) { c.open_faces[f - 1] = 1; c.openflag = 1; }
    c.setup();
    r = c;
  }
  m.set(core, r);
  return r;
};

/** Region::init of every region at a run's init: "vel_timestep = -1;" (the motion history prev stays). */
const initTree = (r: NativeRegion): void => {
  r.vel_timestep = -1;
  if (r instanceof NativeCompound) for (const s of r.reglist) initTree(s);
};

/** ncontact[i], walls[i][], history_many[i][][] of one atom. */
interface AtomHistory { n: number; walls: Int32Array; slots: Float64Array[] }

export class NativeRegionWall {
  readonly tmax: number;
  private readonly c2r: Int32Array;
  private readonly atoms = new Map<number, AtomHistory>();
  private cur: AtomHistory | null = null;

  constructor(readonly region: NativeRegion, private readonly sizeHistory: number, private readonly useHistory: boolean) {
    this.tmax = region.tmax;
    this.c2r = new Int32Array(this.tmax);
  }

  /** The wall's init at a run (Domain::init of the regions). */
  init(): void {
    initTree(this.region);
  }

  /** post_force before the atom loop: prematch and set_velocity of a dynamic or variable-shape region; true if so. */
  begin(step: number, dt: number): boolean {
    const regiondynamic = this.region.dynamic_check() !== 0;
    if (regiondynamic) {
      this.region.prematch();
      this.region.set_velocity(step, dt);
    }
    return regiondynamic;
  }

  /**
   * One atom of post_force up to its contact loop: the region match, the surface contacts within cutoff and the history
   * upkeep. Returns the contact count, or -1 when native skips the atom (outside the region, or no contact with history).
   */
  atom(aid: number, x: number, y: number, z: number, cutoff: number): number {
    const region = this.region;
    if (!region.match(x, y, z)) return -1;
    const nc = region.surface(x, y, z, cutoff);
    if (nc > this.tmax) throw new StyleError('Too many wallgran/region contacts for one particle');
    if (!this.useHistory) return nc;
    let h = this.atoms.get(aid);
    if (nc === 0) {
      if (h) h.n = 0;
      return -1;
    }
    if (!h) {
      const all = new Float64Array(this.tmax * this.sizeHistory);
      const slots = Array.from({ length: this.tmax }, (_, k) => all.subarray(k * this.sizeHistory, (k + 1) * this.sizeHistory));
      h = { n: 0, walls: new Int32Array(this.tmax), slots };
      this.atoms.set(aid, h);
    }
    this.cur = h;
    if (nc === 1) {
      this.c2r[0] = 0;
      const iwall = region.cIwall[0];
      if (h.n === 0) {
        h.n = 1;
        h.walls[0] = iwall;
        h.slots[0].fill(0.0);
      } else if (h.n > 1 || iwall !== h.walls[0]) this.update_contacts(h, nc);
    } else this.update_contacts(h, nc);
    return nc;
  }

  /** history_many[i][c2r[ic]] of the current atom. */
  history(ic: number): Float64Array {
    return this.cur!.slots[this.c2r[ic]];
  }

  /**
   * FixWallGranRegion::update_contacts: drop the old contacts whose wall is not among the new ones (the last one copied
   * over), then add the new walls with a zeroed history; c2r[m] = inew for a wall kept in slot m.
   */
  private update_contacts(h: AtomHistory, nc: number): void {
    const region = this.region, walls = h.walls, slots = h.slots, c2r = this.c2r;
    let iold = 0;
    while (iold < h.n) {
      let m: number;
      for (m = 0; m < nc; m++) if (region.cIwall[m] === walls[iold]) break;
      if (m >= nc) {
        const ilast = h.n - 1;
        slots[iold].set(slots[ilast]);
        walls[iold] = walls[ilast];
        h.n--;
      } else iold++;
    }
    const nold = h.n;
    for (let inew = 0; inew < nc; inew++) {
      const iwall = region.cIwall[inew];
      let m: number;
      for (m = 0; m < nold; m++) if (walls[m] === iwall) break;
      if (m < nold) c2r[m] = inew;
      else {
        const iadd = h.n;
        c2r[iadd] = inew;
        slots[iadd].fill(0.0);
        walls[iadd] = iwall;
        h.n++;
      }
    }
  }
}

/** The hook of the core fix: a native region wall for the region, or null when a (sub-)region style has no port. */
export const createRegionWall = (sys: RegionWallSys, regionId: string, sizeHistory: number, useHistory: boolean): NativeRegionWall | null => {
  const r = mirrorOf(sys, sys.region(regionId));
  return r ? new NativeRegionWall(r, sizeHistory, useHistory) : null;
};
