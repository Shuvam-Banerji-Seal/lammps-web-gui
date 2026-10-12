/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ELECTRODE/fix_electrode_thermo.cpp and fix_electrode_thermo.h (the thermo-potentiostat: the potential
 * difference between two electrodes relaxes towards its target with a stochastic Langevin-like charge update) at
 * commit b332f28, and the RanMars generator of src/random_mars.cpp (src/engine-gpl/random_mars.ts). ELECTRODE
 * contributing authors: Ludwig Ahrens-Iwers (TUHH), Shern Tee (UQ), Robert Meissner (TUHH), and the LAMMPS developers.
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { RanMars } from '../random_mars';
import { FixElectrodeConp } from './fix_electrode_conp';

/** fix_electrode_thermo.cpp SMALL: the relaxation time must exceed it. */
const SMALL = 1e-16;

export class FixElectrodeThermo extends FixElectrodeConp {
  readonly style: string = 'electrode/thermo';
  private readonly thermoRandom: RanMars;
  private vacCap = 0;
  private deltaPsi0 = 0;
  private readonly groupQOld = [0, 0];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (this.groupBits.length !== 2) throw new StyleError('Number of electrodes != two in electrode/thermo');
    if ((this.psiVar[0] === null) !== (this.psiVar[1] === null)) throw new StyleError('Potentials in electrode/thermo must have same style');
    if (this.algo !== 'mat_inv') throw new StyleError('Algorithm not allowed in electrode/thermo');
    if (this.thermoTime < SMALL) throw new StyleError('Keyword temp not set or zero in electrode/thermo');
    this.thermoRandom = new RanMars(this.thermoInit);
    if (this.psiVar[0] === null) this.deltaPsi0 = this.psiConst[1] - this.psiConst[0];
  }

  protected allowsTemp(): boolean { return true; }
  protected styleName(): string { return 'electrode/thermo'; }

  /** Vacuum capacitance between the two electrodes from the macro capacitance. */
  protected computeMacroMatrices(): void {
    super.computeMacroMatrices();
    const mc = this.macroCapacitance;
    if (this.symm) this.vacCap = mc[0][0];
    else this.vacCap = (mc[0][0] * mc[1][1] - mc[0][1] * mc[0][1]) / (mc[0][0] + mc[1][1] + 2 * mc[0][1]);
  }

  /** Total electrode charges after the last step (needed by update_psi). */
  protected preUpdate(): void {
    const s = this.sys.state;
    for (let g = 0; g < 2; g++) {
      this.groupQOld[g] = 0;
      for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBits[g]) this.groupQOld[g] += s.q[i];
    }
  }

  /** update_psi(): the potentials that relax the potential difference towards its target, with thermal noise. */
  protected updatePsi(): void {
    const s = this.sys.state;
    const dt = s.dt;
    const me = this.macroElastance;
    // group_q_eff is the charge that corresponds to the potential after the previous step
    const qEff0 = this.groupQOld[0] - this.sbCharges[0];
    const qEff1 = this.symm ? 0 : this.groupQOld[1] - this.sbCharges[1];
    const psiOld0 = me[0][0] * qEff0 + me[0][1] * qEff1;
    const psiOld1 = me[1][0] * qEff0 + me[1][1] * qEff1;
    const deltaPsi = psiOld1 - psiOld0;
    // target potential difference from the input parameters
    if (this.psiVar[0] !== null) {
      this.deltaPsi0 = this.sys.vars.scalar(this.psiVar[1]!, this.sys.formulaEnv) - this.sys.vars.scalar(this.psiVar[0]!, this.sys.formulaEnv);
    }
    let deltaCharge = 0.5 * (this.groupQOld[1] - this.groupQOld[0])
      - this.vacCap * (deltaPsi - this.deltaPsi0) * (1 - Math.exp(-dt / this.thermoTime));
    deltaCharge += Math.sqrt((this.thermoTemp * this.vacCap) * (1 - Math.exp(-2 * dt / this.thermoTime))) * this.thermoRandom.gaussian();
    const r0 = -deltaCharge - this.sbCharges[0];
    const r1 = this.symm ? 0 : deltaCharge - this.sbCharges[1];
    this.groupPsi[0] = me[0][0] * r0 + me[0][1] * r1;
    this.groupPsi[1] = me[1][0] * r0 + me[1][1] * r1;
  }
}
