/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/fix_nve_eff.cpp and fix_nve_eff.h
 * (class FixNVEEff, fix nve/eff): velocity-Verlet integration of the nuclei and of the electron radius. The
 * radial velocity of an electron (|espin| = 1) moves with the radial force erforce / (dim/4), and the radius
 * moves with the radial velocity; fixed cores (espin 2) keep their radius. Contributing author (from the
 * source): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { FixNVE } from '../../engine/fix/nve';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RADIUS, ELECTRON_RFORCE, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';

export class FixNVEEff extends FixNVE {
  override readonly style: string = 'nve/eff';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix nve/eff takes no arguments (fix_nve_eff.cpp constructor: the electron style is required)
    if (args.length) throw new StyleError('fix nve/eff takes no arguments');
  }

  override init(): void {
    const s = this.sys.state;
    if (!isElectronStyle(s.atomStyle)) throw new StyleError('Fix nve/eff requires atom style electron');
    this.resetDt();
  }

  /** fix_nve_eff.cpp initial_integrate: v += dtfm F, x += dtv v, and for electrons ervel += dtfm erforce / mefactor, eradius += dtv ervel. */
  override initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf, dtv = this.dtv;
    const mefactor = this.sys.dimension / 4.0;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const eradius = s.custom.get(ELECTRON_RADIUS)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const erforce = s.custom.get(ELECTRON_RFORCE)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(s, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      x[3 * i] += dtv * v[3 * i];
      x[3 * i + 1] += dtv * v[3 * i + 1];
      x[3 * i + 2] += dtv * v[3 * i + 2];
      if (Math.abs(spin[i]) === 1) {
        ervel[i] += dtfm * erforce[i] / mefactor;
        eradius[i] += dtv * ervel[i];
      }
    }
  }

  /** fix_nve_eff.cpp final_integrate: v += dtfm F and, for electrons, ervel += dtfm erforce / mefactor. */
  override finalIntegrate(): void {
    const s = this.sys.state;
    const { v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf;
    const mefactor = this.sys.dimension / 4.0;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const erforce = s.custom.get(ELECTRON_RFORCE)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(s, i);
      v[3 * i] += dtfm * f[3 * i];
      v[3 * i + 1] += dtfm * f[3 * i + 1];
      v[3 * i + 2] += dtfm * f[3 * i + 2];
      if (Math.abs(spin[i]) === 1) ervel[i] += dtfm * erforce[i] / mefactor;
    }
  }
}
