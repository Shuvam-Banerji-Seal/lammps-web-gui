/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/region_intersect.cpp, src/region_intersect.h, src/region_union.cpp and src/region_union.h (RegIntersect
 * and RegUnion: the flags and contact counts of the constructors, inside, surface, surface_interior, surface_exterior,
 * shape_update, pretransform, set_velocity, velocity_contact, reset_vel) at commit b332f28. Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * As native does, the sub-regions are the region objects themselves (shared with every other compound and wall that
 * names them): surface_exterior flips their interior flag for the call, set_velocity is guarded per sub-region by its
 * timestep, and velocity_contact reads the contact list of the sub-region the contact came from.
 *
 * Native's surface_interior and surface_exterior take each sub-region's contacts from Region::surface, which already
 * rotates them back to the lab frame, and then, for a rotating sub-region, forward-transform the contacts n - ncontacts
 * to n - 1 once more (a second rotation, applied to the last ncontacts kept contacts whichever sub-region they came
 * from). This file does the same. When fewer than ncontacts were kept so far the native loop starts at a negative index
 * (a write before the contact array); here it starts at 0, which gives native's values for every contact in the list.
 */
import { NativeRegion, type NativeRegionEnv } from './native_region';

export class NativeCompound extends NativeRegion {
  /** contact_indx: the sub-region (ilist) and its contact (ic) of each contact. */
  private ciList = new Int32Array(0);
  private ciIc = new Int32Array(0);
  private readonly xo = new Float64Array(3);
  private readonly xs = new Float64Array(3);

  constructor(id: string, style: 'intersect' | 'union', env: NativeRegionEnv, readonly reglist: NativeRegion[]) {
    super(id, style, env);
    this.nregion = reglist.length;
    // this region is variable shape or dynamic if any of sub-regions are
    for (const r of reglist) {
      if (r.varshape) this.varshape = 1;
      if (r.dynamic) this.dynamic = 1;
      if (r.moveflag) this.moveflag = 1;
      if (r.rotateflag) this.rotateflag = 1;
    }
  }

  /** cmax = the sum of the sub-regions' cmax; tmax = the sum of their tmax inside, the sub-region count outside. */
  setup(): void {
    let cmax = 0;
    for (const r of this.reglist) cmax += r.cmax;
    this.allocContacts(cmax);
    this.ciList = new Int32Array(cmax);
    this.ciIc = new Int32Array(cmax);
    let tmax = 0;
    for (const r of this.reglist) tmax += this.interior ? r.tmax : 1;
    this.tmax = tmax;
  }

  /**
   * The compound region's own transform (Region::match calls it when a sub-region is dynamic): its displacement and
   * angle stay 0 (union and intersect take no move or rotate keyword, and their pretransform only updates the
   * sub-regions), so the point is unchanged.
   */
  inverse_transform(_t: Float64Array): void { /* identity */ }
  forward_transform(_t: Float64Array): void { /* identity */ }

  inside(x: number, y: number, z: number): number {
    const reg = this.reglist;
    let ilist: number;
    if (this.style === 'intersect') {
      for (ilist = 0; ilist < this.nregion; ilist++) if (!reg[ilist].match(x, y, z)) break;
      return ilist === this.nregion ? 1 : 0;
    }
    for (ilist = 0; ilist < this.nregion; ilist++) if (reg[ilist].match(x, y, z)) break;
    return ilist === this.nregion ? 0 : 1;
  }

  /** The compound's surface: no remap or transform of its own (each sub-region's surface does both). */
  surface(x: number, y: number, z: number, cutoff: number): number {
    const xo = this.xo;
    xo[0] = x; xo[1] = y; xo[2] = z;
    if (!this.openflag) return this.interior ? this.surface_interior(xo, cutoff) : this.surface_exterior(xo, cutoff);
    return this.surface_exterior(xo, cutoff) + this.surface_interior(xo, cutoff);
  }

  surface_interior(xorig: Float64Array, cutoff: number): number {
    // intersect: keep a contact whose surface point matches every other sub-region; union: one that no other closed
    // sub-region matches
    return this.collect(xorig, cutoff, true);
  }

  surface_exterior(xorig: Float64Array, cutoff: number): number {
    const reg = this.reglist;
    for (const r of reg) r.interior ^= 1;
    const n = this.collect(xorig, cutoff, false);
    for (const r of reg) r.interior ^= 1;
    return n;
  }

  /**
   * The loop of surface_interior (interior) and surface_exterior (!interior, sub-regions already flipped). The test of
   * the other sub-regions: intersect interior and union exterior keep the contact when all others match (intersect) /
   * none matches (union exterior: "if (match) break"); see the cases below. Wall IDs: interior adds the sum of the
   * earlier sub-regions' cmax to the sub-region's iwall, exterior uses the sub-region index.
   */
  private collect(xorig: Float64Array, cutoff: number, interior: boolean): number {
    const reg = this.reglist, nregion = this.nregion;
    const intersect = this.style === 'intersect';
    const xs = this.xs;
    let n = 0;
    let walloffset = 0;
    for (let ilist = 0; ilist < nregion; ilist++) {
      const region = reg[ilist];
      const ncontacts = region.surface(xorig[0], xorig[1], xorig[2], cutoff);
      for (let m = 0; m < ncontacts; m++) {
        const sx = xorig[0] - region.cDelx[m];
        const sy = xorig[1] - region.cDely[m];
        const sz = xorig[2] - region.cDelz[m];
        let jlist: number;
        for (jlist = 0; jlist < nregion; jlist++) {
          if (jlist === ilist) continue;
          const other = reg[jlist];
          if (intersect === interior) {
            // intersect interior, union exterior
            if (intersect) { if (!other.match(sx, sy, sz)) break; }
            else if (other.match(sx, sy, sz)) break;
          } else if (intersect) {
            // intersect exterior
            if (other.match(sx, sy, sz)) break;
          } else if (other.match(sx, sy, sz) && !other.openflag) break;
        }
        if (jlist === nregion) {
          this.cR[n] = region.cR[m];
          this.cRadius[n] = region.cRadius[m];
          this.cDelx[n] = region.cDelx[m];
          this.cDely[n] = region.cDely[m];
          this.cDelz[n] = region.cDelz[m];
          this.cIwall[n] = interior ? region.cIwall[m] + walloffset : ilist;
          this.cVarflag[n] = region.cVarflag[m];
          this.ciList[n] = ilist;
          this.ciIc[n] = m;
          n++;
        }
      }
      if (region.rotateflag && ncontacts) {
        for (let i = Math.max(0, n - ncontacts); i < n; i++) {
          xs[0] = xorig[0] - this.cDelx[i];
          xs[1] = xorig[1] - this.cDely[i];
          xs[2] = xorig[2] - this.cDelz[i];
          region.forward_transform(xs);
          this.cDelx[i] = xorig[0] - xs[0];
          this.cDely[i] = xorig[1] - xs[1];
          this.cDelz[i] = xorig[2] - xs[2];
        }
      }
      // increment by cmax instead of tmax to ensure possible wall IDs for sub-regions are non overlapping
      if (interior) walloffset += region.cmax;
    }
    return n;
  }

  shape_update(): void {
    for (const r of this.reglist) r.shape_update();
  }

  pretransform(): void {
    for (const r of this.reglist) r.pretransform();
  }

  set_velocity(step: number, dt: number): void {
    for (const r of this.reglist) r.set_velocity(step, dt);
  }

  velocity_contact(vwall: Float64Array, x: ArrayLike<number>, ic: number, dt: number): void {
    this.reglist[this.ciList[ic]].velocity_contact(vwall, x, this.ciIc[ic], dt);
  }

  reset_vel(): void {
    for (const r of this.reglist) r.reset_vel();
  }
}
