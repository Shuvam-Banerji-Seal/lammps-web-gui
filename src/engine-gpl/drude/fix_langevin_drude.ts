/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/fix_langevin_drude.cpp and
 * src/DRUDE/fix_langevin_drude.h (class FixLangevinDrude, fix langevin/drude). Copyright (2003) Sandia Corporation;
 * GPL-2.0 (src/engine-gpl/LICENSE). DRUDE package authors: Alain Dequidt, Julien Devemy, Agilio Padua; contributors
 * Kateryna Goloviznina and Zheng Gong.
 *
 * Langevin thermostat for core-Drude pairs: one stream (Marsaglia RanMars) for the core-of-pair motion and one for the
 * relative (Drude) motion, each with its own temperature and damping period. Non-polarizable atoms take the core
 * stream. Random numbers are drawn in native atom order (s.order), as LAMMPS walks its local atom list.
 * Syntax: fix ID group langevin/drude Tcore period_core seed_core Tdrude period_drude seed_drude [zero yes|no]
 * (docs.lammps.org/fix_langevin_drude.html). Constant temperatures only (no v_ variables).
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { massOf } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { DRUDE_TYPE, NOPOL_TYPE, drudeModelOf } from './drude_model';

const num = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || w === '' || !Number.isFinite(v)) throw new StyleError(`Illegal fix langevin/drude ${what}: '${w ?? ''}'`);
  return v;
};

export class FixLangevinDrude extends Fix {
  readonly style = 'langevin/drude';
  private readonly tCore: number;
  private readonly periodCore: number;
  private readonly tDrude: number;
  private readonly periodDrude: number;
  private readonly rngCore: RanMars;
  private readonly rngDrude: RanMars;
  private zero = false;
  private nCore = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 6) throw new StyleError('Illegal fix langevin/drude command');
    for (const w of [args[0], args[3]]) if (w.startsWith('v_')) throw new StyleError('fix langevin/drude: variable temperatures are not supported');
    this.tCore = num(args[0], 'Tcore');
    this.periodCore = num(args[1], 'period');
    const seedCore = num(args[2], 'seed');
    this.tDrude = num(args[3], 'Tdrude');
    this.periodDrude = num(args[4], 'period');
    const seedDrude = num(args[5], 'seed');
    if (this.periodCore <= 0 || this.periodDrude <= 0) throw new StyleError('Fix langevin/drude period must be > 0.0');
    if (seedCore <= 0 || seedDrude <= 0) throw new StyleError('Illegal langevin/drude seed');
    for (let k = 6; k < args.length; k += 2) {
      if (args[k] !== 'zero' || args[k + 1] === undefined) throw new StyleError('Illegal fix langevin/drude command');
      const yn = args[k + 1];
      if (yn !== 'yes' && yn !== 'no') throw new StyleError(`Expected boolean parameter instead of '${yn}' in input script`);
      this.zero = yn === 'yes';
    }
    this.rngCore = new RanMars(seedCore);
    this.rngDrude = new RanMars(seedDrude);
  }

  init(): void {
    drudeModelOf(this.sys.state, 'fix langevin/drude');
  }

  /** Setup only counts the cores for zero yes (LAMMPS FixLangevinDrude::setup applies no post_force). */
  override setup(): void {
    const s = this.sys.state;
    const model = drudeModelOf(s, 'fix langevin/drude');
    let n = 0;
    for (let i = 0; i < s.n; i++) if ((s.mask[i] & this.groupBit) && model.kind[s.type[i]] !== DRUDE_TYPE) n++;
    this.nCore = n;
  }

  override postForce(): void {
    const s = this.sys.state;
    const model = drudeModelOf(s, 'fix langevin/drude');
    const partner = model.partnerIndex(s);
    const { v, f, mask, type, order } = s;
    const u = s.units;
    const dim = this.sys.dimension;
    const dt = s.dt;
    const ftm2v = u.ftm2v, mvv2e = u.mvv2e, kb = u.boltz;
    const tc = this.tCore, td = this.tDrude;
    const fcoreloc = [0, 0, 0];
    const fcore = [0, 0, 0], fdrude = [0, 0, 0], vcore = [0, 0, 0], vdrude = [0, 0, 0];
    for (let p = 0; p < s.n; p++) {
      const i = order[p];
      if (!(mask[i] & this.groupBit)) continue;
      if (model.kind[type[i]] === NOPOL_TYPE) {
        const mi = massOf(s, i);
        const gcore = mi / this.periodCore / ftm2v;
        const ccore = Math.sqrt((2.0 * gcore * kb * tc) / dt / ftm2v / mvv2e);
        for (let k = 0; k < dim; k++) {
          fcore[k] = ccore * this.rngCore.gaussian() - gcore * v[3 * i + k];
          if (this.zero) fcoreloc[k] += fcore[k];
          f[3 * i + k] += fcore[k];
        }
      } else {
        if (model.kind[type[i]] === DRUDE_TYPE) continue; // handled with its core
        const j = partner[i];
        if (j < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[i]} is not defined`);
        let mi = massOf(s, i), mj = massOf(s, j);
        const mtot = mi + mj;
        const mu = (mi * mj) / mtot;
        mi /= mtot;
        mj /= mtot;
        const gcore = mtot / this.periodCore / ftm2v;
        const gdrude = mu / this.periodDrude / ftm2v;
        const ccore = Math.sqrt((2.0 * gcore * kb * tc) / dt / ftm2v / mvv2e);
        const cdrude = Math.sqrt((2.0 * gdrude * kb * td) / dt / ftm2v / mvv2e);
        for (let k = 0; k < dim; k++) {
          vcore[k] = mi * v[3 * i + k] + mj * v[3 * j + k];
          vdrude[k] = v[3 * j + k] - v[3 * i + k];
          fcore[k] = ccore * this.rngCore.gaussian() - gcore * vcore[k];
          fdrude[k] = cdrude * this.rngDrude.gaussian() - gdrude * vdrude[k];
          if (this.zero) fcoreloc[k] += fcore[k];
          f[3 * i + k] += mi * fcore[k] - fdrude[k];
          f[3 * j + k] += mj * fcore[k] + fdrude[k];
        }
      }
    }
    if (this.zero) this.removeDrift(fcoreloc, model, partner);
  }

  /** zero yes: removes the mean core random force so the total stays zero (LAMMPS post_force, zero block). */
  private removeDrift(fcoreloc: number[], model: ReturnType<typeof drudeModelOf>, partner: Int32Array): void {
    const s = this.sys.state;
    const dim = this.sys.dimension;
    const sum = [0, 0, 0];
    for (let k = 0; k < dim; k++) sum[k] = fcoreloc[k] / this.nCore;
    const { f, mask, type, order } = s;
    for (let p = 0; p < s.n; p++) {
      const i = order[p];
      if (!(mask[i] & this.groupBit)) continue;
      if (model.kind[type[i]] === NOPOL_TYPE) {
        for (let k = 0; k < dim; k++) f[3 * i + k] -= sum[k];
      } else {
        if (model.kind[type[i]] === DRUDE_TYPE) continue;
        const j = partner[i];
        let mi = massOf(s, i), mj = massOf(s, j);
        const mtot = mi + mj;
        mi /= mtot;
        mj /= mtot;
        for (let k = 0; k < dim; k++) {
          f[3 * i + k] -= mi * sum[k];
          f[3 * j + k] -= mj * sum[k];
        }
      }
    }
  }
}
