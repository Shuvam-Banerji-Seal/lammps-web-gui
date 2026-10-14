/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/GRANSURF/fix_surface_local.cpp and src/GRANSURF/fix_surface_local.h (FixSurfaceLocal:
 * constructor argument handling, assign2d and assign3d for the input mol source, check of the atom style),
 * src/GRANSURF/fix_surface.cpp (extract_from_molecule) and src/molecule.cpp (line_segments, triangles).
 * Contributing authors of the GRANSURF package: Joel Clemmer (SNL), Dan Bolintineanu (SNL).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_surface_local.html: "The fix surface/local command ... stores the connectivity of the line or
 * triangle particles that make up the surface, used by pair styles such as surf/granular." The connectivity itself
 * is built by the pair style (gransurf/connect2d.ts, connect3d.ts) at its first compute after the run is set up.
 *
 * "input mol" (fix_surface_local.html) takes the ID of a molecule template with Lines (2d) or Triangles (3d) and
 * creates one line or triangle particle per entry, as native does in assign2d and assign3d: the particle sits at the
 * midpoint (centroid), has the molecule-ID and type of the entry, and its end points or corners are its bonus data.
 *
 * Browser-engine differences:
 *   - "input stl" (fix_surface.cpp extract_from_stlfile: ASCII STL, degenerate triangles skipped, points keyed by
 *     coordinates and molecule-ID) and "flat" (the flat threshold of the connectivity) are supported as native does;
 *   - native identifies the points of the template by molecule-ID and coordinates, so entries of two molecule-IDs
 *     that share a point are not connected there; the browser engine connects by coordinates, so that case is refused;
 *   - a particle whose centre lies outside the box is refused (native remaps it into a periodic box first).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { appendAtoms, isLineStyle, isTriStyle, LINE_BONUS, LINE_FLAG, maxAtomId, TRI_BONUS, TRI_BONUS_COLS, TRI_FLAG } from '../../engine/atoms';
import { lineFromEnds } from '../../engine/lineTri';
import type { SurfaceEntry } from '../../engine/molecule';
import { TRI_BODY_HOOK } from '../tri/tri_bonus';
import { parseStl } from '../../engine/stl';
import { stlFirstSolid } from './stl_solid';

type V3 = [number, number, number];

/** fix_surface_local.cpp: FLATTHRESH (the default flat threshold, 1 - cos(1 degree)). */
const FLATTHRESH = 0.00015230484360876085;

/** One "input" keyword: a molecule template, or an STL file with its surface type and molecule-ID. */
type Source = { kind: 'mol'; id: string } | { kind: 'stl'; stype: number; smol: number; file: string };

/**
 * The flat threshold of the surface/local fix defined on a system state (fix_surface_local.cpp: flatthresh, used by its
 * connectivity). The pair surf/granular builds its connectivity from the same particles and reads it here.
 */
const FLAT_THRESHOLD = new WeakMap<object, number>();
export const surfaceLocalFlatthresh = (st: object): number => FLAT_THRESHOLD.get(st) ?? FLATTHRESH;

/** The end points (2d) or corners (3d) of one surface entry, from its flat coordinates. */
const pointsOf = (e: SurfaceEntry, dim: number): V3[] => {
  const w = dim === 2 ? 2 : 3;
  const out: V3[] = [];
  for (let c = 0; c < e.p.length; c += w) out.push(dim === 2 ? [e.p[c], e.p[c + 1], 0] : [e.p[c], e.p[c + 1], e.p[c + 2]]);
  return out;
};

/**
 * fix_surface.cpp extract_from_stlfile: the triangles of an ASCII STL file, each with the surface type and molecule-ID
 * of its input keyword. A triangle whose edge cross product is exactly zero is skipped (native nskip), so the
 * particle IDs that follow count only the kept triangles.
 */
const stlEntries = (sys: System, src: { stype: number; smol: number; file: string }, dim: number): SurfaceEntry[] => {
  if (dim === 2) throw new StyleError('Fix surface cannot use an STL file for 2d simulations');
  if (src.stype < 1) throw new StyleError('STL surface type must be >= 1');
  const out: SurfaceEntry[] = [];
  for (const tri of parseStl(stlFirstSolid(sys.readFile(src.file)), src.file)) {
    const [a, b, c] = tri.v;
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    if (cr[0] === 0 && cr[1] === 0 && cr[2] === 0) continue;
    out.push({ mol: src.smol, type: src.stype, p: [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]] });
  }
  return out;
};

export class FixSurfaceLocal extends Fix {
  readonly style: string = 'surface/local';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_surface_local.cpp: "input" keywords come first, each "input mol ID" or "input stl type mol-ID file"
    const sources: Source[] = [];
    let k = 0;
    while (k < args.length && args[k] === 'input') {
      if (args[k + 1] === 'mol' && args[k + 2] !== undefined) {
        sources.push({ kind: 'mol', id: args[k + 2] });
        k += 3;
      } else if (args[k + 1] === 'stl') {
        // fix_surface_local.cpp: "input stl" takes the surface type, the molecule-ID and the file name
        const stype = Number(args[k + 2]), smol = Number(args[k + 3]);
        const file = args[k + 4];
        if (file === undefined || !Number.isInteger(stype) || !Number.isInteger(smol)) {
          throw new StyleError('fix surface/local input stl needs type, molecule-ID and file name');
        }
        sources.push({ kind: 'stl', stype, smol, file });
        k += 5;
      } else {
        throw new StyleError(`fix surface/local keyword 'input ${args[k + 1] ?? ''}' is not supported`);
      }
    }
    // fix_surface_local.cpp: "flat" sets the flat threshold, 1 - cos(flat degrees); the default is one degree
    let flatthresh = FLATTHRESH;
    while (k < args.length) {
      if (args[k] !== 'flat') throw new StyleError(`fix surface/local keyword '${args[k]}' is not supported`);
      const flat = Number(args[k + 1]);
      // fix_surface_local.cpp: "Invalid value {} for fix surface/local flat keyword" when flat is outside 0..90
      if (args[k + 1] === undefined || !(flat >= 0 && flat <= 90)) throw new StyleError(`Invalid value ${args[k + 1] ?? ''} for fix surface/local flat keyword`);
      flatthresh = 1.0 - Math.cos((Math.PI * flat) / 180.0);
      k += 2;
    }
    FLAT_THRESHOLD.set(sys.state, flatthresh);
    if (sources.length) this.createFromTemplates(sys, sources);
  }

  /** assign2d / assign3d: one particle per Lines (2d) or Triangles (3d) entry of the input templates. */
  private createFromTemplates(sys: System, sources: Source[]): void {
    const dim = sys.dimension;
    const s = sys.state;
    if (dim === 2 && !isLineStyle(s.atomStyle)) throw new StyleError('Fix surface/local requires atom style line');
    if (dim === 3 && !isTriStyle(s.atomStyle)) throw new StyleError('Fix surface/local requires atom style tri');
    const entries: SurfaceEntry[] = [];
    for (const src of sources) {
      if (src.kind === 'stl') {
        for (const e of stlEntries(sys, src, dim)) entries.push(e);
        continue;
      }
      // fix_surface.cpp, extract_from_molecule: "Molecule template ID for fix surface does not exist"
      const sets = sys.molecules.get(src.id);
      if (!sets) throw new StyleError(`Molecule template ID for fix surface does not exist: ${src.id}`);
      for (const t of sets) {
        const list = dim === 2 ? t.lines : t.tris;
        // fix_surface.cpp: "Fix surface molecule must have lines" / "Fix surface molecule must have triangles"
        if (!list) throw new StyleError(dim === 2 ? 'Fix surface molecule must have lines' : 'Fix surface molecule must have triangles');
        for (const e of list) entries.push(e);
      }
    }
    if (!entries.length) throw new StyleError('Fix surface/local defines no line/triangle particles');
    // fix_surface.cpp keys the points by (coordinates, molecule-ID): a point shared by two molecule-IDs is refused here
    const owner = new Map<string, number>();
    for (const e of entries) {
      for (const p of pointsOf(e, dim)) {
        const key = p.join(',');
        const m = owner.get(key);
        if (m !== undefined && m !== e.mol) throw new StyleError(`fix surface/local: a point shared by molecule-IDs ${m} and ${e.mol} is not supported by the browser engine`);
        owner.set(key, e.mol);
      }
    }
    const box = s.box;
    if (box.triclinic) throw new StyleError('fix surface/local input mol in a triclinic box is not supported by the browser engine');
    const n = entries.length;
    const n0 = s.n;
    const x = new Float64Array(3 * n), type = new Int32Array(n), mol = new Int32Array(n), id = new Int32Array(n);
    const idmax = maxAtomId(s);
    const lineBonus: { length: number; theta: number }[] = [];
    const triBonus: ReturnType<typeof TRI_BODY_HOOK.triBody>[] = [];
    entries.forEach((e, i) => {
      const pts = pointsOf(e, dim);
      let c: V3;
      if (dim === 2) {
        lineBonus.push(lineFromEnds(pts[0][0], pts[0][1], pts[1][0], pts[1][1]));
        c = [0.5 * (pts[0][0] + pts[1][0]), 0.5 * (pts[0][1] + pts[1][1]), 0];
      } else {
        triBonus.push(TRI_BODY_HOOK.triBody([pts[0], pts[1], pts[2]], 1));
        c = [0, 1, 2].map((d) => (pts[0][d] + pts[1][d] + pts[2][d]) / 3) as V3;
      }
      for (let d = 0; d < 3; d++) {
        if (!(c[d] >= box.lo[d] && c[d] < box.hi[d])) {
          throw new StyleError(`fix surface/local: the centre of surface entry ${i + 1} lies outside the box (not supported by the browser engine)`);
        }
      }
      if (!Number.isInteger(e.type) || e.type < 1 || e.type > s.ntypes) throw new StyleError(`Invalid ${dim === 2 ? 'line' : 'tri'} type ${e.type} in fix surface/local`);
      x.set(c, 3 * i);
      type[i] = e.type;
      mol[i] = e.mol;
      // native: tag = idmaxall + i + 1 (the surface particles follow the atoms that exist)
      id[i] = idmax + i + 1;
    });
    // native assign2d / assign3d: the particle is created with radius 0 and density 1 (rmass), then the bonus data
    // sets the radius and mass (atom_vec_line.cpp data_atom_bonus: radius = length / 2, rmass = density * length;
    // atom_vec_tri.cpp: the mass is the density times the area)
    appendAtoms(s, { x, type, id, molecule: mol, radius: 0, rmass: 1 });
    for (let i = 0; i < n; i++) {
      const a = n0 + i;
      if (dim === 2) {
        const lb = lineBonus[i];
        s.custom.get(LINE_FLAG)!.data[a] = 1;
        const bonus = s.custom.get(LINE_BONUS)!.data;
        bonus[2 * a] = lb.length;
        bonus[2 * a + 1] = lb.theta;
        s.radius![a] = 0.5 * lb.length;
        s.rmass![a] = lb.length;
      } else {
        const tb = triBonus[i];
        s.custom.get(TRI_FLAG)!.data[a] = 1;
        const bonus = s.custom.get(TRI_BONUS)!.data;
        for (let k = 0; k < 9; k++) bonus[TRI_BONUS_COLS * a + k] = tb.corners[k];
        for (let d = 0; d < 3; d++) bonus[TRI_BONUS_COLS * a + 9 + d] = tb.moments[d];
        s.radius![a] = tb.radius;
        s.rmass![a] = tb.mass;
        for (let d = 0; d < 4; d++) s.quat![4 * a + d] = tb.quat[d];
      }
    }
    sys.bump();
  }

  init(): void {
    const s = this.sys.state;
    if (this.sys.dimension === 2 && !isLineStyle(s.atomStyle)) throw new StyleError('Fix surface/local requires atom style line in 2d');
    if (this.sys.dimension === 3 && !isTriStyle(s.atomStyle)) throw new StyleError('Fix surface/local requires atom style tri in 3d');
  }
}
