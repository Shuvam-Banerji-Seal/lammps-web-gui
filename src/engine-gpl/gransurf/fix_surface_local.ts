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
 *   - input stl and the flat keyword are refused with a StyleError naming them;
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

type V3 = [number, number, number];

/** The end points (2d) or corners (3d) of one surface entry, from its flat coordinates. */
const pointsOf = (e: SurfaceEntry, dim: number): V3[] => {
  const w = dim === 2 ? 2 : 3;
  const out: V3[] = [];
  for (let c = 0; c < e.p.length; c += w) out.push(dim === 2 ? [e.p[c], e.p[c + 1], 0] : [e.p[c], e.p[c + 1], e.p[c + 2]]);
  return out;
};

export class FixSurfaceLocal extends Fix {
  readonly style: string = 'surface/local';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_surface_local.cpp: "input" keywords come first, each "input mol ID" or "input stl ..."
    const sources: string[] = [];
    let k = 0;
    while (k < args.length && args[k] === 'input') {
      if (args[k + 1] === 'mol' && args[k + 2] !== undefined) {
        sources.push(args[k + 2]);
        k += 3;
      } else if (args[k + 1] === 'stl') {
        throw new StyleError('fix surface/local input stl (surfaces from STL files) is not supported by the browser engine; use input mol with Lines or Triangles');
      } else {
        throw new StyleError(`fix surface/local keyword 'input ${args[k + 1] ?? ''}' is not supported`);
      }
    }
    // fix_surface_local.cpp: the optional keywords (flat) are not supported in this engine
    if (k < args.length) throw new StyleError(`fix surface/local keyword '${args[k]}' is not supported`);
    if (sources.length) this.createFromTemplates(sys, sources);
  }

  /** assign2d / assign3d: one particle per Lines (2d) or Triangles (3d) entry of the input templates. */
  private createFromTemplates(sys: System, sources: string[]): void {
    const dim = sys.dimension;
    const s = sys.state;
    if (dim === 2 && !isLineStyle(s.atomStyle)) throw new StyleError('Fix surface/local requires atom style line');
    if (dim === 3 && !isTriStyle(s.atomStyle)) throw new StyleError('Fix surface/local requires atom style tri');
    const entries: SurfaceEntry[] = [];
    for (const src of sources) {
      // fix_surface.cpp, extract_from_molecule: "Molecule template ID for fix surface does not exist"
      const sets = sys.molecules.get(src);
      if (!sets) throw new StyleError(`Molecule template ID for fix surface does not exist: ${src}`);
      for (const t of sets) {
        const list = dim === 2 ? t.lines : t.tris;
        // fix_surface.cpp: "Fix surface molecule must have lines" / "Fix surface molecule must have triangles"
        if (!list) throw new StyleError(dim === 2 ? 'Fix surface molecule must have lines' : 'Fix surface molecule must have triangles');
        entries.push(...list);
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
