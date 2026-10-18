/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/RIGID/compute_rigid_local.cpp
 * and compute_rigid_local.h (class ComputeRigidLocal, compute rigid/local): one row per rigid body of a fix rigid/small,
 * taken from the body owned by the body's owning atom, which is in the compute's group. Contributing author (from the
 * sources): Trung Dac Nguyen (ORNL). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The owning atom (fix_rigid_small.cpp create_bodies: "find atom closest to geometric center") is set when the body is
 * created. The engine's body does not keep it, so it is found here from the atom positions of the first evaluation
 * (the setup configuration, when the body is created in the usual scripts) with the same rule.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { buildAtomMap, nativeOrder } from '../../engine/atoms';
import { rigidBodiesOf, type RigidBodyView } from './rigid_bodies';

const KEYWORDS = [
  'id', 'mol', 'mass', 'x', 'y', 'z', 'xu', 'yu', 'zu', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'ix', 'iy', 'iz',
  'tqx', 'tqy', 'tqz', 'omegax', 'omegay', 'omegaz', 'angmomx', 'angmomy', 'angmomz',
  'quatw', 'quati', 'quatj', 'quatk', 'inertiax', 'inertiay', 'inertiaz',
];

interface Ctx { id: Int32Array | number[]; molecule: Int32Array; L: number[] }
type Getter = (b: RigidBodyView, owner: number, c: Ctx) => number;

/** The per-body value of each keyword (compute_rigid_local.cpp compute_rigid, the switch on rstyle). */
const GETTERS: Record<string, Getter> = {
  id: (_b, o, c) => c.id[o],
  mol: (_b, o, c) => c.molecule[o],
  mass: (b) => b.mass,
  x: (b) => b.xcm[0], y: (b) => b.xcm[1], z: (b) => b.xcm[2],
  // native: body->xcm + ((image & IMGMASK) - IMGMAX) * xprd, the wrapped xcm plus the image shift
  xu: (b, _o, c) => b.xcm[0] + b.image[0] * c.L[0],
  yu: (b, _o, c) => b.xcm[1] + b.image[1] * c.L[1],
  zu: (b, _o, c) => b.xcm[2] + b.image[2] * c.L[2],
  vx: (b) => b.vcm[0], vy: (b) => b.vcm[1], vz: (b) => b.vcm[2],
  fx: (b) => b.fcm[0], fy: (b) => b.fcm[1], fz: (b) => b.fcm[2],
  ix: (b) => b.image[0], iy: (b) => b.image[1], iz: (b) => b.image[2],
  tqx: (b) => b.torque[0], tqy: (b) => b.torque[1], tqz: (b) => b.torque[2],
  omegax: (b) => b.omega[0], omegay: (b) => b.omega[1], omegaz: (b) => b.omega[2],
  angmomx: (b) => b.angmom[0], angmomy: (b) => b.angmom[1], angmomz: (b) => b.angmom[2],
  quatw: (b) => b.quat[0], quati: (b) => b.quat[1], quatj: (b) => b.quat[2], quatk: (b) => b.quat[3],
  inertiax: (b) => b.inertia[0], inertiay: (b) => b.inertia[1], inertiaz: (b) => b.inertia[2],
};

export class ComputeRigidLocal extends Compute {
  readonly style: string = 'rigid/local';
  private readonly rfix: string;
  private readonly keys: string[];
  /** owning atom index of each body, found on the first evaluation (see the header) */
  private owners: number[] | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal compute rigid/local command');
    this.rfix = args[0];
    this.keys = args.slice(1);
    for (const k of this.keys) if (!KEYWORDS.includes(k)) throw new StyleError('Invalid keyword in compute rigid/local command');
    this.localFlag = true;
    this.sizeLocalCols = this.keys.length === 1 ? 0 : this.keys.length;
  }

  /** compute_rigid_local.cpp init: the fix must exist and be a rigid/small fix. */
  override init(): void {
    const fix = this.sys.fixes.find((f) => f.id === this.rfix);
    if (!fix) throw new StyleError(`FixRigidSmall ID ${this.rfix} for compute rigid/local does not exist`);
    if (!fix.style.startsWith('rigid') || !fix.style.includes('small')) {
      throw new StyleError(`Fix ID ${this.rfix} for compute rigid/local does not point to fix rigid/small`);
    }
  }

  /**
   * fix_rigid_small.cpp create_bodies with the rendezvous rule: the owner of a body is the atom nearest the centre of
   * the body's bounding box, ties to the lower atom ID. Positions are unwrapped with the image flags.
   */
  private findOwners(bodies: RigidBodyView[]): number[] {
    const s = this.sys.state;
    const box = s.box;
    const L = [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]];
    const map = buildAtomMap(s);
    return bodies.map((b) => {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      const pos = b.atoms.map((id) => {
        const i = map[id];
        if (i < 0) throw new StyleError(`compute rigid/local: body atom ${id} does not exist`);
        const p = [0, 1, 2].map((d) => s.x[3 * i + d] + s.image[3 * i + d] * L[d]);
        for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]); }
        return { i, p };
      });
      const ctr = [0, 1, 2].map((d) => 0.5 * (lo[d] + hi[d]));
      let best = -1, bestD2 = Infinity;
      for (const { i, p } of pos) {
        const d2 = (p[0] - ctr[0]) * (p[0] - ctr[0]) + (p[1] - ctr[1]) * (p[1] - ctr[1]) + (p[2] - ctr[2]) * (p[2] - ctr[2]);
        if (best < 0 || d2 < bestD2 || (d2 === bestD2 && s.id[i] < s.id[best])) { bestD2 = d2; best = i; }
      }
      return best;
    });
  }

  /** The bodies with their owning atoms: null when the fix holds no bodies. */
  private bodiesAndOwners(): { bodies: RigidBodyView[]; owners: number[] } | null {
    const fix = this.sys.fixes.find((f) => f.id === this.rfix);
    if (!fix) return null;
    const bodies = rigidBodiesOf(this.sys, fix);
    if (!bodies) return null;
    if (this.owners === null || this.owners.length !== bodies.length) this.owners = this.findOwners(bodies);
    return { bodies, owners: this.owners };
  }

  /** compute_rigid_local.cpp compute_local: one row per body whose owning atom is in the group, in native atom order. */
  protected computeLocal(): Float64Array<ArrayBuffer> {
    const s = this.sys.state;
    const found = this.bodiesAndOwners();
    const ncol = Math.max(1, this.keys.length);
    const rows: number[] = [];
    if (found) {
      const ownerBody = new Map<number, number>();
      found.owners.forEach((a, b) => ownerBody.set(a, b));
      const box = s.box;
      const ctx: Ctx = { id: s.id, molecule: s.molecule, L: [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]] };
      for (const i of nativeOrder(s)) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const bi = ownerBody.get(i);
        if (bi === undefined) continue;
        const b = found.bodies[bi];
        for (const k of this.keys) rows.push(GETTERS[k](b, i, ctx));
      }
    }
    this.localRows = rows.length / ncol;
    return Float64Array.from(rows);
  }
}
