/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/fix_dpd_energy.cpp and
 * fix_dpd_energy.h (FixDPDenergy: fix dpd/energy), src/DPD-REACT/fix_eos_cv.cpp and fix_eos_cv.h (FixEOScv: fix
 * eos/cv), and src/DPD-REACT/compute_dpd.cpp and compute_dpd.h (ComputeDpd: compute dpd). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE). Contributing author (from the sources above): James Larentzos (U.S. Army Research
 * Laboratory).
 *
 * docs.lammps.org/fix_dpd_energy.html: the fix integrates the internal energies with the rates the dpd/fdt/energy pair
 * computes (half a step at each of initial and final integrate). docs.lammps.org/fix_eos_cv.html: "The internal
 * temperature dpdTheta is computed from the internal energies uCond + uMech and the heat capacity cv" (dpdTheta =
 * (uCond + uMech) / cv); at the start of a run uCond is set to 0 and uMech to cv * dpdTheta. docs.lammps.org/compute_dpd.html:
 * the vector gives the sums of uCond, uMech and uChem, the harmonic mean temperature N / sum(1/dpdTheta) and N.
 */
import { Fix } from '../../engine/fix/fix';
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { DPD_THETA, DPD_UCOND, DPD_UMECH, DPD_UCHEM } from '../../engine/atoms';
import { PairDPDfdtEnergy } from './pair_dpd_fdt';

/** fix dpd/energy (no arguments): needs pair dpd/fdt/energy and atom_style dpd. */
export class FixDPDenergy extends Fix {
  readonly style = 'dpd/energy';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal fix dpd/energy command');
    if (!sys.state.custom.has(DPD_THETA)) throw new StyleError('Must use atom_style dpd with fix dpd/energy');
  }

  private pair(): PairDPDfdtEnergy {
    const p = this.sys.ff.pair;
    if (!(p instanceof PairDPDfdtEnergy)) throw new StyleError('Must use pair_style dpd/fdt/energy with fix dpd/energy');
    return p;
  }

  /** Half-step update of uCond and uMech by the pair's rates (both integrate hooks use the same update). */
  private bump(): void {
    const s = this.sys.state;
    const p = this.pair();
    const uc = s.custom.get(DPD_UCOND)!.data, um = s.custom.get(DPD_UMECH)!.data;
    const h = 0.5 * s.dt;
    const n = Math.min(s.n, p.duCond.length);
    for (let i = 0; i < n; i++) {
      uc[i] += h * p.duCond[i];
      um[i] += h * p.duMech[i];
    }
  }

  initialIntegrate(): void { this.bump(); }
  finalIntegrate(): void { this.bump(); }
}

/** fix eos/cv cv (docs.lammps.org/fix_eos_cv.html): one argument, the heat capacity cv > 0. */
export class FixEOScv extends Fix {
  readonly style = 'eos/cv';
  private readonly cv: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Illegal fix eos/cv command');
    this.cv = Number(args[0]);
    if (!(this.cv > 0)) throw new StyleError('EOS cv must be > 0.0');
    this.nevery = 1;
    if (!sys.state.custom.has(DPD_THETA)) throw new StyleError('FixEOScv requires atom_style with internal temperature and energies (e.g. dpd)');
  }

  /** At the start of a run: uCond = 0 and uMech = cv dpdTheta for atoms of the group. */
  init(): void {
    const s = this.sys.state;
    const th = s.custom.get(DPD_THETA)!.data, uc = s.custom.get(DPD_UCOND)!.data, um = s.custom.get(DPD_UMECH)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (!(th[i] > 0)) throw new StyleError('Internal temperature <= zero');
      uc[i] = 0;
      um[i] = this.cv * th[i];
    }
  }

  private update(): void {
    const s = this.sys.state;
    const th = s.custom.get(DPD_THETA)!.data, uc = s.custom.get(DPD_UCOND)!.data, um = s.custom.get(DPD_UMECH)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      th[i] = (uc[i] + um[i]) / this.cv;
      if (!(th[i] > 0)) throw new StyleError('Internal temperature <= zero');
    }
  }

  postIntegrate(): void { this.update(); }
  endOfStep(): void { this.update(); }
}

/** compute dpd (no arguments): vector of 5 (docs.lammps.org/compute_dpd.html). */
export class ComputeDpd extends Compute {
  readonly style = 'dpd';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute dpd command');
    if (!sys.state.custom.has(DPD_THETA)) throw new StyleError('compute dpd requires atom_style with internal temperature and energies (e.g. dpd)');
    this.vectorFlag = true;
    this.sizeVector = 5;
    this.extvector = 0;
    this.vector = new Float64Array(5);
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const uc = s.custom.get(DPD_UCOND)!.data, um = s.custom.get(DPD_UMECH)!.data, uh = s.custom.get(DPD_UCHEM)!.data;
    const th = s.custom.get(DPD_THETA)!.data;
    let sCond = 0, sMech = 0, sChem = 0, sInv = 0, count = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      sCond += uc[i];
      sMech += um[i];
      sChem += uh[i];
      sInv += 1.0 / th[i];
      count += 1.0;
    }
    this.vector[0] = sCond;
    this.vector[1] = sMech;
    this.vector[2] = sChem;
    this.vector[3] = count / sInv;
    this.vector[4] = count;
  }
}
