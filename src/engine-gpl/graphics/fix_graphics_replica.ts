/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/fix_graphics_replica.cpp
 * (FixGraphicsReplica: keywords display and average; end_of_step() collects the atoms of the fix group sorted by ID,
 * unwraps them and draws one sphere per atom and replica (display) and one averaged sphere per atom (average)).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * This engine runs one replica, so the universe has one world: the display spheres are the atoms themselves, and the
 * average of a single replica has zero variance, so its sphere diameter is 0 (or 1 when the average radius is 0), as
 * the C++ gives for nworlds = 1. The positions are unwrapped with the image flags and remapped into the periodic box
 * (Domain::remap), in ID order.
 */
import type { GraphicsObject, SphereObject, Vec3 } from './objects';

export interface ReplicaOptions {
  /** display radius; null when the display keyword is absent. */
  display: number | null;
  /** average radius; null when the average keyword is absent. */
  average: number | null;
}

export interface ReplicaAtom {
  id: number;
  type: number;
  /** Wrapped position. */
  x: Vec3;
  /** Image flags (box crossings). */
  image: Vec3;
  inGroup: boolean;
}

export interface ReplicaBox {
  lo: Vec3;
  hi: Vec3;
  prd: Vec3;
  periodic: [boolean, boolean, boolean];
}

/** Keyword parser of FixGraphicsReplica (display <radius>, average <radius>). */
export function parseReplicaArgs(args: string[]): ReplicaOptions {
  const opt: ReplicaOptions = { display: null, average: null };
  let i = 0;
  while (i < args.length) {
    const key = args[i];
    if (key === 'display' || key === 'average') {
      if (i + 2 > args.length) throw new Error(`Too few arguments for fix graphics/replica ${key}`);
      const r = Number(args[i + 1]);
      if (!Number.isFinite(r)) throw new Error(`Invalid fix graphics/replica ${key} radius ${args[i + 1]}`);
      if (key === 'display') opt.display = r;
      else opt.average = r;
      i += 2;
    } else {
      throw new Error(`Unknown fix graphics/replica keyword: ${key}`);
    }
  }
  return opt;
}

/** Domain::remap for an orthogonal box: wrap each periodic coordinate into [lo, hi). */
export function remapPosition(x: Vec3, box: ReplicaBox): Vec3 {
  const out: Vec3 = [x[0], x[1], x[2]];
  for (let d = 0; d < 3; d++) {
    if (!box.periodic[d]) continue;
    while (out[d] < box.lo[d]) out[d] += box.prd[d];
    while (out[d] >= box.hi[d]) out[d] -= box.prd[d];
    out[d] = Math.max(out[d], box.lo[d]);
  }
  return out;
}

/** The sphere objects of one snapshot, in the C++ order: all display spheres, then the average spheres. */
export function replicaObjects(
  opt: ReplicaOptions,
  atoms: ReplicaAtom[],
  box: ReplicaBox,
): GraphicsObject[] {
  const group = atoms.filter((a) => a.inGroup).sort((a, b) => a.id - b.id);
  // unwrapped positions (Domain::unmap) before the remap of each output
  const unwrapped = group.map((a): Vec3 => [
    a.x[0] + a.image[0] * box.prd[0],
    a.x[1] + a.image[1] * box.prd[1],
    a.x[2] + a.image[2] * box.prd[2],
  ]);
  const out: GraphicsObject[] = [];
  if (opt.display !== null) {
    for (let i = 0; i < group.length; i++) {
      const p = remapPosition(unwrapped[i], box);
      const s: SphereObject = { kind: 'sphere', type: group[i].type, x: p[0], y: p[1], z: p[2], diameter: opt.display };
      out.push(s);
    }
  }
  if (opt.average !== null) {
    // one replica: the average is the position itself and the variance is zero
    for (let i = 0; i < group.length; i++) {
      const p = remapPosition(unwrapped[i], box);
      const diameter = opt.average === 0.0 ? 1.0 : opt.average * Math.sqrt(0.0);
      const s: SphereObject = { kind: 'sphere', type: group[i].type, x: p[0], y: p[1], z: p[2], diameter };
      out.push(s);
    }
  }
  return out;
}
