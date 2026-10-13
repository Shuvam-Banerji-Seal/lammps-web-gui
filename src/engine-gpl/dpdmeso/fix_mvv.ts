/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-MESO/fix_mvv_dpd.cpp and fix_mvv_dpd.h
 * (FixMvvDPD: fix mvv/dpd), src/DPD-MESO/fix_mvv_edpd.cpp and fix_mvv_edpd.h (FixMvvEDPD: fix mvv/edpd) and
 * src/DPD-MESO/fix_mvv_tdpd.cpp and fix_mvv_tdpd.h (FixMvvTDPD: fix mvv/tdpd). Contributing author (from the sources):
 * Zhen Li (Clemson University). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The modified velocity-Verlet integrator of docs.lammps.org/fix_mvv_dpd.html: "Setting verlet = 0.5 recovers the standard
 * velocity-Verlet algorithm." The half-step velocity vest is stored per atom (the vest property of the DPD-MESO atom styles)
 * and the positions advance with it; fix mvv/edpd also advances the temperature with the heat flux, and fix mvv/tdpd the
 * chemical concentrations with their fluxes.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import { DPD_VEST, EDPD_CV, EDPD_FLUX, EDPD_TEMP, EDPD_VEST_TEMP, TDPD_CC, TDPD_CC_FLUX, isEdpdStyle, isTdpdStyle, massOf, tdpdSpecies } from '../../engine/atoms';
import type { System } from '../../engine/system';

/** Names of the pair styles (and hybrid sub-styles) of the system. */
const pairNames = (sys: System): string[] => {
  const p = sys.ff.pair;
  if (!p) return [];
  return p instanceof PairHybrid ? p.subs.map((x) => x.name) : [p.name];
};

/** Common part: the verlet argument, the step sizes and the half-step velocity. */
abstract class FixMvvMeso extends Fix {
  protected dtv = 0;
  protected dtf = 0;
  protected verlet = 0.5;

  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args);
    this.timeIntegrate = true;
    if (args.length > 1) throw new StyleError(`Illegal fix ${style} command`);
    if (args.length === 1) this.verlet = parseNum(args[0], 'verlet');
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  init(): void { this.resetDt(); }

  /** True for an atom in the fix's group. */
  protected groupAtom(i: number): boolean {
    return (this.sys.state.mask[i] & this.groupBit) !== 0;
  }

  /** Half-step velocity and position update of the three styles: vest = v + dtfm f, x += dt vest, v += 2 verlet dtfm f. */
  protected kick(): void {
    const s = this.sys.state;
    const { x, v, f } = s;
    const vest = s.custom.get(DPD_VEST)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!this.groupAtom(i)) continue;
      const dtfm = this.dtf / massOf(s, i);
      vest[3 * i] = v[3 * i] + dtfm * f[3 * i];
      vest[3 * i + 1] = v[3 * i + 1] + dtfm * f[3 * i + 1];
      vest[3 * i + 2] = v[3 * i + 2] + dtfm * f[3 * i + 2];
      x[3 * i] += this.dtv * vest[3 * i];
      x[3 * i + 1] += this.dtv * vest[3 * i + 1];
      x[3 * i + 2] += this.dtv * vest[3 * i + 2];
      v[3 * i] += 2.0 * this.verlet * dtfm * f[3 * i];
      v[3 * i + 1] += 2.0 * this.verlet * dtfm * f[3 * i + 1];
      v[3 * i + 2] += 2.0 * this.verlet * dtfm * f[3 * i + 2];
    }
  }

  /** Full-step velocity: v = vest + dtfm f. */
  protected finishVelocity(): void {
    const s = this.sys.state;
    const { v, f } = s;
    const vest = s.custom.get(DPD_VEST)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!this.groupAtom(i)) continue;
      const dtfm = this.dtf / massOf(s, i);
      v[3 * i] = vest[3 * i] + dtfm * f[3 * i];
      v[3 * i + 1] = vest[3 * i + 1] + dtfm * f[3 * i + 1];
      v[3 * i + 2] = vest[3 * i + 2] + dtfm * f[3 * i + 2];
    }
  }
}

/** fix mvv/dpd (docs.lammps.org/fix_mvv_dpd.html): pair dpd or mdpd, atom style mdpd, edpd or tdpd (the vest property). */
export class FixMvvDPD extends FixMvvMeso {
  readonly style = 'mvv/dpd';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'mvv/dpd');
  }

  init(): void {
    const s = this.sys.state;
    if (!s.custom.has(DPD_VEST)) throw new StyleError('Fix mvv/dpd requires atom attribute vest e.g. from atom style mdpd');
    // fix_mvv_dpd.cpp init(): "Must use a dpd or mdpd pair style with fix mvv/dpd"
    if (!pairNames(this.sys).some((n) => n.startsWith('mdpd') || n.startsWith('dpd'))) {
      throw new StyleError('Must use a dpd or mdpd pair style with fix mvv/dpd');
    }
    this.resetDt();
  }

  initialIntegrate(): void { this.kick(); }
  finalIntegrate(): void { this.finishVelocity(); }
}

/** fix mvv/edpd (docs.lammps.org/fix_mvv_edpd.html): pair edpd and the temperature of each atom. */
export class FixMvvEDPD extends FixMvvMeso {
  readonly style = 'mvv/edpd';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'mvv/edpd');
  }

  init(): void {
    const s = this.sys.state;
    if (!isEdpdStyle(s.atomStyle)) throw new StyleError('Fix mvv/edpd requires atom style edpd');
    if (!pairNames(this.sys).includes('edpd')) throw new StyleError('Must use pair style edpd with fix mvv/edpd');
    this.resetDt();
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const T = s.custom.get(EDPD_TEMP)!.data, Q = s.custom.get(EDPD_FLUX)!.data, cv = s.custom.get(EDPD_CV)!.data;
    const vt = s.custom.get(EDPD_VEST_TEMP)!.data;
    const { x, v, f } = s;
    const vest = s.custom.get(DPD_VEST)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!this.groupAtom(i)) continue;
      const dtfm = this.dtf / massOf(s, i);
      const dtT = 0.5 * this.dtv / cv[i];
      vest[3 * i] = v[3 * i] + dtfm * f[3 * i];
      vest[3 * i + 1] = v[3 * i + 1] + dtfm * f[3 * i + 1];
      vest[3 * i + 2] = v[3 * i + 2] + dtfm * f[3 * i + 2];
      vt[i] = T[i] + dtT * Q[i];
      x[3 * i] += this.dtv * vest[3 * i];
      x[3 * i + 1] += this.dtv * vest[3 * i + 1];
      x[3 * i + 2] += this.dtv * vest[3 * i + 2];
      v[3 * i] += 2.0 * this.verlet * dtfm * f[3 * i];
      v[3 * i + 1] += 2.0 * this.verlet * dtfm * f[3 * i + 1];
      v[3 * i + 2] += 2.0 * this.verlet * dtfm * f[3 * i + 2];
      T[i] += 2.0 * this.verlet * dtT * Q[i];
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const T = s.custom.get(EDPD_TEMP)!.data, Q = s.custom.get(EDPD_FLUX)!.data, cv = s.custom.get(EDPD_CV)!.data;
    const vt = s.custom.get(EDPD_VEST_TEMP)!.data;
    const { v, f } = s;
    const vest = s.custom.get(DPD_VEST)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!this.groupAtom(i)) continue;
      const dtfm = this.dtf / massOf(s, i);
      const dtT = 0.5 * this.dtv / cv[i];
      v[3 * i] = vest[3 * i] + dtfm * f[3 * i];
      v[3 * i + 1] = vest[3 * i + 1] + dtfm * f[3 * i + 1];
      v[3 * i + 2] = vest[3 * i + 2] + dtfm * f[3 * i + 2];
      T[i] = vt[i] + dtT * Q[i];
    }
  }
}

/** fix mvv/tdpd (docs.lammps.org/fix_mvv_tdpd.html): pair tdpd and the chemical concentrations of each atom. */
export class FixMvvTDPD extends FixMvvMeso {
  readonly style = 'mvv/tdpd';
  private nsp = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'mvv/tdpd');
  }

  init(): void {
    const s = this.sys.state;
    if (!isTdpdStyle(s.atomStyle)) throw new StyleError('Fix mvv/tdpd requires atom style tdpd');
    this.nsp = tdpdSpecies(s.atomStyle);
    if (!pairNames(this.sys).includes('tdpd')) throw new StyleError('Must use pair style tdpd with fix mvv/tdpd');
    this.resetDt();
  }

  /** Concentrations advance by half a step with their fluxes (fix_mvv_tdpd.cpp: cc += 0.5 dtv cc_flux). */
  private advanceCc(): void {
    const s = this.sys.state;
    const cc = s.custom.get(TDPD_CC)!.data, flux = s.custom.get(TDPD_CC_FLUX)!.data;
    const nsp = this.nsp;
    for (let i = 0; i < s.n; i++) {
      if (!this.groupAtom(i)) continue;
      for (let k = 0; k < nsp; k++) cc[nsp * i + k] += 0.5 * this.dtv * flux[nsp * i + k];
    }
  }

  initialIntegrate(): void {
    this.kick();
    this.advanceCc();
  }

  finalIntegrate(): void {
    this.finishVelocity();
    this.advanceCc();
  }
}
