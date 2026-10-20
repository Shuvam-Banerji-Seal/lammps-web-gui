/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/fix_nh_eff.cpp and fix_nh_eff.h
 * (class FixNHEff: the Nose-Hoover fixes of the electron force field), src/EFF/fix_nvt_eff.cpp, fix_npt_eff.cpp and
 * fix_nph_eff.cpp (fix nvt/eff, npt/eff, nph/eff). The electron radial velocity follows the nuclear velocity in every
 * half step (nve_v, nve_x and nh_v_temp of fix_nh_eff.cpp); the temperature is the one of compute temp/eff.
 * Contributing author (from the sources): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * fix nvt/eff, npt/eff and nph/eff (FixNHEffCore) are the core FixNH (src/engine/fix/nh.ts) with the electron terms
 * added to its half kick, drift and thermostat scaling, so they take every fix nh keyword (iso, aniso, tri, x ... xy,
 * couple, mtk, nreset, dilate, ...), as fix_nh_eff.cpp derives from FixNH. FixNHEff, on the partial port FixNHBase
 * (./fix_nh.ts), is the base of fix nvt/sllod/eff.
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RADIUS, ELECTRON_RFORCE, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';
import { ownCompute } from '../../engine/fix/util';
import { FixNH } from '../../engine/fix/nh';
import { FixNHBase } from './fix_nh';

export class FixNHEff extends FixNHBase {
  constructor(sys: System, id: string, group: string, args: string[], tempId: string, style: string, pressId: string | null = null) {
    super(sys, id, group, args, tempId, style, pressId);
    if (!isElectronStyle(sys.state.atomStyle)) throw new StyleError(`Fix ${style} requires atom style electron`);
  }

  /** fix_nh_eff.cpp nve_v: the nuclear half kick, then ervel += dtfm erforce / (dim/4) for electrons. */
  protected override nveV(): void {
    super.nveV();
    const s = this.sys.state;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const erforce = s.custom.get(ELECTRON_RFORCE)!.data;
    const mefactor = this.sys.dimension / 4.0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (Math.abs(spin[i]) === 1) {
        const dtfm = this.dtf / massOf(s, i);
        ervel[i] += dtfm * erforce[i] / mefactor;
      }
    }
  }

  /** fix_nh_eff.cpp nve_x: the nuclear drift, then the electron radial velocity and radius. */
  protected override nveX(): void {
    super.nveX();
    const s = this.sys.state;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const eradius = s.custom.get(ELECTRON_RADIUS)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const erforce = s.custom.get(ELECTRON_RFORCE)!.data;
    const mefactor = this.sys.dimension / 4.0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const dtfm = this.dtf / massOf(s, i);
      if (Math.abs(spin[i]) === 1) {
        ervel[i] += dtfm * erforce[i] / mefactor;
        eradius[i] += this.dtv * ervel[i];
      }
    }
  }

  /** fix_nh_eff.cpp nh_v_temp: the thermostat scales the electron radial velocities too. */
  protected override nhVTemp(): void {
    super.nhVTemp();
    const s = this.sys.state;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (Math.abs(spin[i]) === 1) ervel[i] *= this.factorEta;
    }
  }
}

/** The core FixNH methods that the eFF electron terms extend (private there; replaced on the instance, as nh_sphere.ts does). */
type NHSteps = {
  kick(): void;
  drift(): void;
  scaleVelocities(factor: number | number[]): void;
};

/**
 * fix nvt/eff, npt/eff, nph/eff: the core FixNH with fix_nh_eff.cpp's electron terms, and its own compute temp/eff
 * ("fix-ID_temp", over the fix group for nvt/eff and over all atoms for npt/eff and nph/eff, whose pressure compute
 * "fix-ID_press" takes it), as fix_nvt_eff.cpp, fix_npt_eff.cpp and fix_nph_eff.cpp create.
 */
class FixNHEffCore extends FixNH {
  readonly style: string;

  constructor(sys: System, id: string, group: string, args: string[], variant: 'nvt' | 'npt' | 'nph') {
    const name = `${variant}/eff`;
    super(sys, id, group, args, variant, name);
    this.style = name;
    if (!isElectronStyle(sys.state.atomStyle)) throw new StyleError(`Fix ${name} requires atom style electron`);
    ownCompute(sys, `${id}_temp`, variant === 'nvt' ? group : 'all', 'temp/eff', []);

    const base = FixNH.prototype as unknown as NHSteps;
    const self = this as unknown as NHSteps;
    const baseKick = base.kick, baseDrift = base.drift, baseScale = base.scaleVelocities;
    self.kick = () => {
      baseKick.call(this);
      this.electronKick();
    };
    self.drift = () => {
      baseDrift.call(this);
      this.electronDrift();
    };
    self.scaleVelocities = (factor) => {
      baseScale.call(this, factor);
      this.electronScale(factor);
    };
  }

  /** dtf = dt/2 ftm2v, the half-step factor of fix nh. */
  private dtfOf(): number {
    const s = this.sys.state;
    return 0.5 * s.dt * s.units.ftm2v;
  }

  /** fix_nh_eff.cpp nve_v: ervel += dtfm erforce / (dim/4) for the electrons, after the nuclear half kick. */
  private electronKick(): void {
    const s = this.sys.state;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const erforce = s.custom.get(ELECTRON_RFORCE)!.data;
    const mefactor = this.sys.dimension / 4.0;
    const dtf = this.dtfOf();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (Math.abs(spin[i]) === 1) {
        const dtfm = dtf / massOf(s, i);
        ervel[i] += dtfm * erforce[i] / mefactor;
      }
    }
  }

  /** fix_nh_eff.cpp nve_x: after the nuclear drift, ervel += dtfm erforce / (dim/4) and eradius += dt ervel. */
  private electronDrift(): void {
    const s = this.sys.state;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const eradius = s.custom.get(ELECTRON_RADIUS)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const erforce = s.custom.get(ELECTRON_RFORCE)!.data;
    const mefactor = this.sys.dimension / 4.0;
    const dtf = this.dtfOf(), dtv = s.dt;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const dtfm = dtf / massOf(s, i);
      if (Math.abs(spin[i]) === 1) {
        ervel[i] += dtfm * erforce[i] / mefactor;
        eradius[i] += dtv * ervel[i];
      }
    }
  }

  /** fix_nh_eff.cpp nh_v_temp: the thermostat factor (factor_eta) scales the electron radial velocities too. */
  private electronScale(factor: number | number[]): void {
    const s = this.sys.state;
    const f = typeof factor === 'number' ? factor : factor[0];
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (Math.abs(spin[i]) === 1) ervel[i] *= f;
    }
  }
}

/** fix_nvt_eff.cpp: fix nvt/eff temp Tstart Tstop Tdamp ..., with compute temp/eff over the fix group. */
export class FixNVTEff extends FixNHEffCore {
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'nvt');
  }
}

/** fix_npt_eff.cpp: fix npt/eff temp ... and a barostat keyword, with compute temp/eff and pressure over all atoms. */
export class FixNPTEff extends FixNHEffCore {
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'npt');
  }
}

/** fix_nph_eff.cpp: fix nph/eff with a barostat keyword and no thermostat, with compute temp/eff and pressure over all atoms. */
export class FixNPHEff extends FixNHEffCore {
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'nph');
  }
}
