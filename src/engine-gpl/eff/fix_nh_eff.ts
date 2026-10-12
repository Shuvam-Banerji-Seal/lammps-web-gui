/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/fix_nh_eff.cpp and fix_nh_eff.h
 * (class FixNHEff: the Nose-Hoover fixes of the electron force field), src/EFF/fix_nvt_eff.cpp (class FixNVTEff,
 * fix nvt/eff). The electron radial velocity follows the nuclear velocity in every half step (nve_v, nve_x and
 * nh_v_temp of fix_nh_eff.cpp); the temperature is the one of compute temp/eff. Contributing author (from the
 * sources): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RADIUS, ELECTRON_RFORCE, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';
import { ownCompute, removeCompute } from '../../engine/fix/util';
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

/** fix_nvt_eff.cpp: fix nvt/eff temp Tstart Tstop Tperiod, with its own compute temp/eff (id = fix-ID + temp). */
export class FixNVTEff extends FixNHEff {
  private readonly ownTempId: string;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const tempId = `${id}_temp`;
    super(sys, id, group, args, tempId, 'nvt/eff');
    // fix_nvt_eff.cpp: "Pressure control can not be used with fix nvt/eff"
    if (this.hasPress()) throw new StyleError('Pressure control can not be used with fix nvt/eff');
    this.ownTempId = tempId;
    ownCompute(sys, tempId, group, 'temp/eff', []);
  }

  override destroy(): void {
    removeCompute(this.sys, this.ownTempId);
  }
}

/** fix_npt_eff.cpp: fix npt/eff temp ... iso ...; the temperature compute is over all atoms (pressure is global). */
export class FixNPTEff extends FixNHEff {
  private readonly ownTempId: string;
  private readonly ownPressId: string;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const tempId = `${id}_temp`, pressId = `${id}_press`;
    super(sys, id, group, args, tempId, 'npt/eff', pressId);
    if (!this.hasTemp()) throw new StyleError('Temperature control must be used with fix npt/eff');
    if (!this.hasPress()) throw new StyleError('Pressure control must be used with fix npt/eff');
    this.ownTempId = tempId;
    this.ownPressId = pressId;
    ownCompute(sys, tempId, 'all', 'temp/eff', []);
    ownCompute(sys, pressId, 'all', 'pressure', [tempId]);
  }

  override destroy(): void {
    removeCompute(this.sys, this.ownTempId);
    removeCompute(this.sys, this.ownPressId);
  }
}

/** fix_nph_eff.cpp: fix nph/eff iso ...; no thermostat (the temperature is the one of compute temp/eff over all atoms). */
export class FixNPHEff extends FixNHEff {
  private readonly ownTempId: string;
  private readonly ownPressId: string;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const tempId = `${id}_temp`, pressId = `${id}_press`;
    super(sys, id, group, args, tempId, 'nph/eff', pressId);
    if (this.hasTemp()) throw new StyleError('Temperature control can not be used with fix nph/eff');
    if (!this.hasPress()) throw new StyleError('Pressure control must be used with fix nph/eff');
    this.ownTempId = tempId;
    this.ownPressId = pressId;
    ownCompute(sys, tempId, 'all', 'temp/eff', []);
    ownCompute(sys, pressId, 'all', 'pressure', [tempId]);
  }

  override destroy(): void {
    removeCompute(this.sys, this.ownTempId);
    removeCompute(this.sys, this.ownPressId);
  }
}
