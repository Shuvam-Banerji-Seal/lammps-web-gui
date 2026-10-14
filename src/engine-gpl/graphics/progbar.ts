/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/fix_graphics_objects.cpp, the
 * "progbar" item of FixGraphicsObjects (the constructor, init() and the progbar branch of end_of_step()). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * A progress bar is 2 + tics cylinders: the background bar (type1, full length), the fill (type2, 0.75 diameter,
 * shifted by 0.2 or 0.15 diameters) whose moving end follows the progress, and one thin tic cylinder per tic
 * (1.1 diameter) at evenly spaced positions. The tic spacing is length / (tics - 1), as in the C++ source.
 */
import type { CylinderObject } from './objects';

export type ProgbarDim = 'x' | 'y' | 'z';

export interface ProgbarSpec {
  type1: number;
  type2: number;
  dim: ProgbarDim;
  /** Centre of the bar. */
  pos: [number, number, number];
  length: number;
  /** The diameter as given in the input (already multiplied by 2 by the fix). */
  diameter: number;
  tics: number;
}

const AXIS = { x: 0, y: 1, z: 2 } as const;

/** The progbar checks of fix_graphics_objects.cpp: length in (0, 2*prd], tics in 0..20, dimension x/y/z. */
export function parseProgbar(
  type1: number,
  type2: number,
  dim: string,
  pos: [number, number, number],
  length: number,
  diameterRadius: number,
  tics: number,
  prd: [number, number, number],
): ProgbarSpec {
  if (dim !== 'x' && dim !== 'y' && dim !== 'z') {
    throw new Error(`Unsupported progress bar dimension string ${dim}`);
  }
  if (!(length > 0.0) || length > 2.0 * prd[AXIS[dim]]) {
    throw new Error(`Illegal progress bar length ${length}`);
  }
  if (tics < 0 || tics > 20) throw new Error(`Unsupported number of progress bar tics ${tics}`);
  return { type1, type2, dim, pos: [...pos], length, diameter: 2.0 * diameterRadius, tics };
}

/** Number of objects one progbar item contributes (numobjs += 2 + tics in the constructor). */
export function progbarObjectCount(spec: ProgbarSpec): number {
  return 2 + spec.tics;
}

/**
 * The progbar objects for one snapshot. progress is clamped to [1e-10, 1] as in end_of_step(); the caller evaluates
 * the progress variable (or passes the constant).
 */
export function progbarObjects(spec: ProgbarSpec, progress: number): CylinderObject[] {
  const a = AXIS[spec.dim];
  const [px, py, pz] = spec.pos;
  const p = Math.max(Math.min(progress, 1.0), 1.0e-10);
  const half = 0.5 * spec.length;
  const d = spec.diameter;
  const out: CylinderObject[] = [];

  // background bar, full length along dim
  const bg: CylinderObject = {
    kind: 'cylinder', type: spec.type1,
    x1: px, y1: py, z1: pz, x2: px, y2: py, z2: pz, diameter: d,
  };
  if (a === 0) { bg.x1 -= half; bg.x2 += half; }
  else if (a === 1) { bg.y1 -= half; bg.y2 += half; }
  else { bg.z1 -= half; bg.z2 += half; }
  out.push(bg);

  // fill: the start end is fixed at the bar start, the moving end follows the progress
  const fill: CylinderObject = {
    kind: 'cylinder', type: spec.type2,
    x1: px, y1: py, z1: pz, x2: px, y2: py, z2: pz, diameter: 0.75 * d,
  };
  if (a === 0) {
    fill.x1 -= half;
    fill.x2 -= half;
    fill.z1 += 0.2 * d;
    fill.z2 += 0.2 * d;
  } else if (a === 1) {
    fill.y1 -= half;
    fill.y2 -= half;
    fill.x1 += 0.15 * d;
    fill.x2 += 0.15 * d;
  } else {
    fill.z1 -= half;
    fill.z2 -= half;
    fill.x1 += 0.15 * d;
    fill.x2 += 0.15 * d;
  }
  // end_of_step() sets the moving coordinate of the fill's first end
  const moved = (p - 0.5) * spec.length;
  if (a === 0) fill.x1 = px + moved;
  else if (a === 1) fill.y1 = py + moved;
  else fill.z1 = pz + moved;
  out.push(fill);

  // tics at lo + k * delta, each 0.1 delta wide
  const delta = spec.length / (spec.tics - 1);
  const lo = spec.pos[a] - half;
  for (let i = 0; i < spec.tics; i++) {
    const tic: CylinderObject = {
      kind: 'cylinder', type: spec.type1,
      x1: px, y1: py, z1: pz, x2: px, y2: py, z2: pz, diameter: 1.1 * d,
    };
    const c1 = lo + delta * i - 0.05 * delta;
    const c2 = lo + delta * i + 0.05 * delta;
    if (a === 0) { tic.x1 = c1; tic.x2 = c2; }
    else if (a === 1) { tic.y1 = c1; tic.y2 = c2; }
    else { tic.z1 = c1; tic.z2 = c2; }
    out.push(tic);
  }
  return out;
}
