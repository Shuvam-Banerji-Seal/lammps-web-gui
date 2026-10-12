/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/fix_nve_line.cpp and src/ASPHERE/fix_nve_line.h at commit b332f28 (fix nve/line: velocity
 * Verlet integration of 2d line segments, the translation and the rotation about z). The sources carry no
 * contributing-author line; copyright Sandia Corporation (LAMMPS development team, GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_nve_line.html: "Perform constant NVE integration to update position, velocity, orientation, and angular velocity for line segment particles in the group each timestep", with the moment of inertia of a segment of mass m and
 * length L being m L^2 / 12 (the INERTIA constant of the source). The orientation theta is kept in (-pi, pi].
 *
 * Browser-engine difference: the segment length and orientation are the core custom properties of each atom
 * (atoms.ts LINE_BONUS) rather than the atom->bonus array.
 */
import { FixNVE } from '../../engine/fix/nve';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isLineAtom, isLineStyle, LINE_BONUS } from '../../engine/atoms';

/** Moment of inertia prefactor of a line segment (fix_nve_line.cpp: INERTIA = 1/12). */
const INERTIA = 1.0 / 12.0;
const TWOPI = 2.0 * Math.PI;

export class FixNVELine extends FixNVE {
  readonly style: string = 'nve/line';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_nve_line.cpp: "if (narg != 3) error->all(FLERR,"Illegal fix nve/line command");"
    if (args.length) throw new StyleError('Illegal fix nve/line command');
  }

  init(): void {
    const s = this.sys.state;
    if (!isLineStyle(s.atomStyle)) throw new StyleError('Fix nve/line requires atom style line');
    if (this.sys.dimension !== 2) throw new StyleError('Fix nve/line can only be used for 2d simulations');
    // fix_nve_line.cpp init: every atom of the group must be a line segment (no point particles)
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !isLineAtom(s, i)) throw new StyleError('Fix nve/line requires line particles');
    }
    super.init();
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const bonus = s.custom.get(LINE_BONUS)!.data;
    const { x, v, f, mask } = s;
    const omega = s.omega!, torque = s.torque!, rmass = s.rmass!;
    const dtf = this.dtf, dtv = this.dtv;
    // d_omega/dt = torque / inertia; the inertia is rmass L^2 / 12 with rmass the mass of the segment
    const dtfrotate = dtf / INERTIA;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const dtfm = dtf / rmass[i];
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      x[3 * i] += dtv * v[3 * i];
      x[3 * i + 1] += dtv * v[3 * i + 1];
      const length = bonus[2 * i];
      let theta = bonus[2 * i + 1];
      const dtirotate = dtfrotate / (length * length * rmass[i]);
      omega[3 * i + 2] += dtirotate * torque[3 * i + 2];
      theta += dtv * omega[3 * i + 2];
      // bound theta to (-pi, pi]
      while (theta <= -Math.PI) theta += TWOPI;
      while (theta > Math.PI) theta -= TWOPI;
      bonus[2 * i + 1] = theta;
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const bonus = s.custom.get(LINE_BONUS)!.data;
    const { v, f, mask } = s;
    const omega = s.omega!, torque = s.torque!, rmass = s.rmass!;
    const dtf = this.dtf;
    const dtfrotate = dtf / INERTIA;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const dtfm = dtf / rmass[i];
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      const length = bonus[2 * i];
      const dtirotate = dtfrotate / (length * length * rmass[i]);
      omega[3 * i + 2] += dtirotate * torque[3 * i + 2];
    }
  }
}
