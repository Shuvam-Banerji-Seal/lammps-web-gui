/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/fix_graphics_periodic.cpp
 * (FixGraphicsPeriodic: the keywords xlo..zhi, radius, atoms, bonds; end_of_step() builds the periodic image
 * spheres and bonds). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * For each atom of the group, and for each image shift (ix, iy, iz) in the requested faces, not (0, 0, 0), a sphere
 * is emitted (atoms on) and then one bond per bonded partner in the group, to the partner's closest image (bonds on).
 * The order and the counts are those of end_of_step(): per atom, per shift (x outer, z inner), sphere then bonds.
 * A negative sphere diameter (radius auto, the default) is the renderer's default size, as in the C++ (2 * -1).
 */
import type { BondObject, GraphicsObject, SphereObject, Vec3 } from './objects';
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import type { ImageObject } from '../../engine/output/image_objects';
import { parseInteger } from './common';
import { drawGraphicsObject } from './draw';

export interface PeriodicOptions {
  /** Image faces requested by the keywords xlo, xhi, ylo, yhi, zlo, zhi. */
  faces: { xlo: boolean; xhi: boolean; ylo: boolean; yhi: boolean; zlo: boolean; zhi: boolean };
  /** radius value; -1 means auto (the default). */
  radius: number;
  atoms: boolean;
  bonds: boolean;
}

export interface PeriodicAtom {
  id: number;
  type: number;
  x: Vec3;
  inGroup: boolean;
  /** IDs of the bonded partners of this atom (the bond_atom list). */
  bondPartners: number[];
}

export interface PeriodicBox {
  prd: Vec3;
  periodic: [boolean, boolean, boolean];
}

/** The keyword parser of FixGraphicsPeriodic (defaults: atoms on, bonds on if molecular, radius auto). */
export function parsePeriodicArgs(args: string[], molecular: boolean): PeriodicOptions {
  const opt: PeriodicOptions = {
    faces: { xlo: false, xhi: false, ylo: false, yhi: false, zlo: false, zhi: false },
    radius: -1.0,
    atoms: true,
    bonds: molecular,
  };
  const bool = (v: string): boolean => {
    if (/^(on|yes|true|1)$/i.test(v)) return true;
    if (/^(off|no|false|0)$/i.test(v)) return false;
    throw new Error(`Expected boolean parameter instead of '${v}' in fix graphics/periodic`);
  };
  let i = 0;
  while (i < args.length) {
    const key = args[i];
    if (key in opt.faces) {
      (opt.faces as Record<string, boolean>)[key] = true;
      i += 1;
    } else if (key === 'radius') {
      if (i + 2 > args.length) throw new Error('Illegal fix graphics/periodic radius: missing argument');
      if (args[i + 1] === 'auto') opt.radius = -1.0;
      else {
        const r = Number(args[i + 1]);
        if (!Number.isFinite(r) || r <= 0.0) throw new Error('Fix graphics/periodic radius must be > 0');
        opt.radius = r;
      }
      i += 2;
    } else if (key === 'atoms') {
      if (i + 2 > args.length) throw new Error('Illegal fix graphics/periodic atoms: missing argument');
      opt.atoms = bool(args[i + 1]);
      i += 2;
    } else if (key === 'bonds') {
      if (i + 2 > args.length) throw new Error('Illegal fix graphics/periodic bonds: missing argument');
      opt.bonds = bool(args[i + 1]);
      i += 2;
    } else {
      throw new Error(`Unknown fix graphics/periodic keyword: ${key}`);
    }
  }
  if (opt.bonds && !molecular) {
    throw new Error('Cannot display periodic images of bonds with non-molecular system');
  }
  return opt;
}

/** Position of partner m as seen from i: the periodic image of m closest to i (Domain::closest_image). */
export function closestImage(xi: Vec3, xm: Vec3, box: PeriodicBox): Vec3 {
  const out: Vec3 = [xm[0], xm[1], xm[2]];
  for (let d = 0; d < 3; d++) {
    if (!box.periodic[d]) continue;
    const delta = xm[d] - xi[d];
    out[d] = xm[d] - box.prd[d] * Math.round(delta / box.prd[d]);
  }
  return out;
}

/** The image objects of one snapshot, in the order the C++ fills them. */
export function periodicObjects(
  opt: PeriodicOptions,
  atoms: PeriodicAtom[],
  box: PeriodicBox,
): GraphicsObject[] {
  const byId = new Map<number, PeriodicAtom>();
  for (const a of atoms) byId.set(a.id, a);
  const r = opt.radius;
  const lo = (f: boolean) => (f ? -1 : 0);
  const hi = (f: boolean) => (f ? 1 : 0);
  const out: GraphicsObject[] = [];
  for (const a of atoms) {
    if (!a.inGroup) continue;
    for (let ix = lo(opt.faces.xlo); ix <= hi(opt.faces.xhi); ix++) {
      for (let iy = lo(opt.faces.ylo); iy <= hi(opt.faces.yhi); iy++) {
        for (let iz = lo(opt.faces.zlo); iz <= hi(opt.faces.zhi); iz++) {
          if (ix === 0 && iy === 0 && iz === 0) continue;
          const sx = ix * box.prd[0];
          const sy = iy * box.prd[1];
          const sz = iz * box.prd[2];
          if (opt.atoms) {
            const s: SphereObject = {
              kind: 'sphere', type: a.type,
              x: a.x[0] + sx, y: a.x[1] + sy, z: a.x[2] + sz,
              diameter: 2.0 * r,
            };
            out.push(s);
          }
          if (opt.bonds) {
            for (const pid of a.bondPartners) {
              const m = byId.get(pid);
              if (!m || !m.inGroup) continue;
              const xm = closestImage(a.x, m.x, box);
              const b: BondObject = {
                kind: 'bond', types: [a.type, m.type],
                x1: a.x[0] + sx, y1: a.x[1] + sy, z1: a.x[2] + sz,
                x2: xm[0] + sx, y2: xm[1] + sy, z2: xm[2] + sz,
              };
              out.push(b);
            }
          }
        }
      }
    }
  }
  return out;
}

/** The fix: "fix ID group-ID graphics/periodic N keyword values ..." (fix_graphics_periodic.cpp). */
export class FixGraphicsPeriodicFix extends Fix {
  readonly style = 'graphics/periodic';
  private readonly opt: PeriodicOptions;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError(`usage: fix ${id} group-ID graphics/periodic N keyword values ...`);
    this.nevery = parseInteger(args[0], `fix ${id} graphics/periodic N`);
    if (this.nevery <= 0) throw new StyleError(`fix ${id} graphics/periodic: N must be > 0`);
    const molecular = sys.state.topo.bonds.n > 0;
    try {
      this.opt = parsePeriodicArgs(args.slice(1), molecular);
    } catch (e) {
      throw new StyleError(`fix ${id} graphics/periodic: ${(e as Error).message}`);
    }
  }

  /** The image spheres and bonds of the snapshot (periodicObjects), in the C++ order. */
  imageObjects(): ImageObject[] {
    const s = this.sys.state;
    const n = s.n;
    const partners = new Map<number, number[]>();
    // newton_bond on: a bond is stored with its first atom (the data file rule), so it is drawn once
    const bonds = s.topo.bonds;
    for (let b = 0; b < bonds.n; b++) {
      const a1 = bonds.atoms[2 * b];
      const list = partners.get(a1);
      if (list) list.push(bonds.atoms[2 * b + 1]);
      else partners.set(a1, [bonds.atoms[2 * b + 1]]);
    }
    const atoms: PeriodicAtom[] = [];
    for (let i = 0; i < n; i++) {
      atoms.push({
        id: s.id[i],
        type: s.type[i],
        x: [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]],
        inGroup: this.inGroup(i),
        bondPartners: partners.get(s.id[i]) ?? [],
      });
    }
    const box = s.box;
    const objs = periodicObjects(this.opt, atoms, {
      prd: [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]],
      periodic: [box.periodic[0], box.periodic[1], box.periodic[2]],
    });
    return objs.map((o): ImageObject => ({ kind: 'draw', draw: (f, st) => drawGraphicsObject(f, st, o) }));
  }
}

export const makeFixGraphicsPeriodic: FixFactory = (sys, id, group, args) => new FixGraphicsPeriodicFix(sys, id, group, args);
