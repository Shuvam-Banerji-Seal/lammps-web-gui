/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/fix_nve_tri.cpp and src/ASPHERE/fix_nve_tri.h at commit b332f28 (fix nve/tri: velocity Verlet
 * for 3d triangular particles, with the angular momentum kicked by the torque and the orientation advanced by the
 * Richardson-corrected quaternion update of MathExtra::richardson). The sources carry no contributing-author line;
 * copyright Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_nve_tri.html: "Perform constant NVE integration to update position, velocity, orientation, and angular momentum for triangular particles in the group each timestep".
 *
 * Browser-engine difference: the principal moments of each triangle are the core custom property TRI_BONUS (columns
 * 9 to 11). The orientation update is the port of MathExtra::richardson in ../asphere/math_extra.ts (mq_to_omega,
 * then richardson with dtq = dt/2).
 */
import { FixNVE } from '../../engine/fix/nve';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isTriAtom, isTriStyle, TRI_BONUS, TRI_BONUS_COLS } from '../../engine/atoms';
import { richardson } from '../asphere/math_extra';

export class FixNVETri extends FixNVE {
  readonly style: string = 'nve/tri';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_nve_tri.cpp: "if (narg != 3) error->all(FLERR,"Illegal fix nve/tri command");"
    if (args.length) throw new StyleError('Illegal fix nve/tri command');
  }

  init(): void {
    const s = this.sys.state;
    if (!isTriStyle(s.atomStyle)) throw new StyleError('Fix nve/tri requires atom style tri');
    if (this.sys.dimension !== 3) throw new StyleError('Fix nve/tri can only be used for 3d simulations');
    // fix_nve_tri.cpp init: the Fix nve/tri requires tri particles message
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !isTriAtom(s, i)) throw new StyleError('Fix nve/tri requires tri particles');
    }
    super.init();
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const bonus = s.custom.get(TRI_BONUS)!.data;
    const { x, v, f, mask } = s;
    const L = s.angmom!, torque = s.torque!, rmass = s.rmass!, quat = s.quat!;
    const dtf = this.dtf, dtv = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const dtfm = dtf / rmass[i];
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      x[3 * i] += dtv * v[3 * i];
      x[3 * i + 1] += dtv * v[3 * i + 1];
      x[3 * i + 2] += dtv * v[3 * i + 2];
      L[3 * i] += dtf * torque[3 * i];
      L[3 * i + 1] += dtf * torque[3 * i + 1];
      L[3 * i + 2] += dtf * torque[3 * i + 2];
      const I = [bonus[TRI_BONUS_COLS * i + 9], bonus[TRI_BONUS_COLS * i + 10], bonus[TRI_BONUS_COLS * i + 11]] as [number, number, number];
      richardson(quat, 4 * i, L, 3 * i, I, dtv);
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const { v, f, mask } = s;
    const L = s.angmom!, torque = s.torque!, rmass = s.rmass!;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const dtfm = dtf / rmass[i];
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      L[3 * i] += dtf * torque[3 * i];
      L[3 * i + 1] += dtf * torque[3 * i + 1];
      L[3 * i + 2] += dtf * torque[3 * i + 2];
    }
  }
}
