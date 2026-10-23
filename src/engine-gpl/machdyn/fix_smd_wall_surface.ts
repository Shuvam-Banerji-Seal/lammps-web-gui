/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/fix_smd_wall_surface.cpp and fix_smd_wall_surface.h (FixSMDWallSurface: fix smd/wall_surface,
 * read_triangles, setup) at commit b332f28. MACHDYN author (from the sources): Georg C. Ganzenmueller, Fraunhofer EMI.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix smd/wall_surface file type molID (docs.lammps.org/fix_smd_wall_surface.html): the triangles of an STL file become
 * atoms of the given type, one per triangle, placed at the centroid of the triangle. The contact radius and the kernel
 * radius of each triangle are the largest distance from the centroid to a vertex; the unit normal is kept in x0 and the
 * three vertices in the deformation prop. Only triangles whose centroid lies in the box of this process are kept (the
 * whole box on one process). The molecule ID must be >= 65535. The atoms are made once, at the setup of the first run
 * (native: "Always run for 0 timesteps directly after loading STL file!").
 */
import { StyleError } from '../../engine/force/types';
import { inumeric } from '../../engine/force/util';
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { appendAtoms } from '../../engine/atoms';
import { SMD_CONTACT, SMD_DEFGRAD } from '../../engine/smd';
import { parseStl } from './stl_reader';

/** Molecule IDs at or above this mark the wall triangles (pair_smd_triangulated_surface.cpp: mol >= 65535). */
const WALL_MOLECULE_MIN = 65535;

/** Relative widening of a periodic subdomain at its boundary (native EPSILON in fix_smd_wall_surface.cpp). */
const EPSILON = 1.0e-6;

export class FixSMDWallSurface extends Fix {
  readonly style = 'smd/wall_surface';
  private readonly filename: string;
  private readonly wallType: number;
  private readonly wallMol: number;
  private done = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('Illegal number of arguments for fix smd/wall_surface');
    this.filename = args[0];
    // As native LAMMPS does (src/MACHDYN/fix_smd_wall_surface.cpp FixSMDWallSurface::FixSMDWallSurface at b332f28):
    // utils::inumeric reads the particle type and the molecule id, then the molecule id is range-checked.
    this.wallType = inumeric(args[1]);
    this.wallMol = inumeric(args[2]);
    if (this.wallMol < WALL_MOLECULE_MIN) throw new StyleError('wall molcule id must be >= 65535');
    if (this.wallType < 1) throw new StyleError(`Illegal fix smd/wall_surface particle type '${args[1]}'`);
  }

  /** For minimization: setup as with dynamics (native min_setup). */
  minSetup(): void {
    this.setup();
  }

  /**
   * Native setup (Verlet::setup calls the fixes' setup after the setup force evaluation): the triangles are read once,
   * so the setup forces and energies of that run are those without the wall atoms, and the wall takes part from the
   * next run on (native: "Always run for 0 timesteps directly after loading STL file!").
   */
  override setup(): void {
    if (this.done) return;
    this.done = true;
    const s = this.sys.state;
    if (!s.custom.has(SMD_DEFGRAD) || !s.custom.has(SMD_CONTACT) || !s.x0 || !s.radius) {
      throw new StyleError('fix smd/wall_surface needs atom_style smd');
    }
    const mesh = parseStl(this.sys.readFile(this.filename, (n, r) => `Error reading triangles from file ${n}: Cannot open STL file ${n}: ${r}`));
    this.sys.log(`fix smd/wall_surface reads ${mesh.triangles.length} triangles from ${this.filename}`);
    // the subdomain of this process: the box, widened by EPSILON of its length at the periodic boundaries
    const lo = [0, 0, 0], hi = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const eps = s.box.periodic[d] ? EPSILON * (s.box.hi[d] - s.box.lo[d]) : 0;
      lo[d] = s.box.lo[d] - eps;
      hi[d] = s.box.hi[d] + eps;
    }
    const xs: number[] = [], vs: number[] = [], nrm: number[] = [], rad: number[] = [];
    for (const tri of mesh.triangles) {
      const [v0, v1, v2] = tri.vert;
      const c = [0, 1, 2].map((k) => (v0[k] + v1[k] + v2[k]) / 3.0);
      const dist = (p: number[]) => Math.sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2);
      const r = Math.max(dist(v0), dist(v1), dist(v2));
      if (!(c[0] >= lo[0] && c[0] < hi[0] && c[1] >= lo[1] && c[1] < hi[1] && c[2] >= lo[2] && c[2] < hi[2])) continue;
      xs.push(c[0], c[1], c[2]);
      rad.push(r);
      nrm.push(tri.normal[0], tri.normal[1], tri.normal[2]);
      vs.push(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);
    }
    const n = rad.length;
    if (n === 0) return;
    const custom = new Map<string, Float64Array>([
      [SMD_CONTACT, Float64Array.from(rad)],
      [SMD_DEFGRAD, Float64Array.from(vs)],
    ]);
    appendAtoms(s, {
      x: Float64Array.from(xs),
      type: this.wallType,
      molecule: this.wallMol,
      radius: Float64Array.from(rad),
      x0: Float64Array.from(nrm),
      custom,
    });
    this.sys.atomsChanged();
    // the forces and energies of this setup were evaluated before the wall atoms existed, as in native
    this.sys.forcesCurrent();
  }
}
