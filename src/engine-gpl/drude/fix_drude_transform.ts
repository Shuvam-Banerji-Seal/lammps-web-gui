/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/fix_drude_transform.cpp and
 * src/DRUDE/fix_drude_transform.h (class FixDrudeTransform<inverse>, fix drude/transform/direct and
 * fix drude/transform/inverse). Copyright (2003) Sandia Corporation; GPL-2.0 (src/engine-gpl/LICENSE). DRUDE package
 * authors: Alain Dequidt, Julien Devemy, Agilio Padua; contributors Kateryna Goloviznina and Zheng Gong.
 *
 * The direct transform takes each core-Drude pair to reduced coordinates at the start and end of a timestep
 * (the Drude position relative to its core, the core moved to the pair's centre of mass, and the masses reduced),
 * so that the thermostats integrate the relative motion; the inverse transform takes them back. Written for owned
 * atoms: the Drude image nearest its core is the minimum-image displacement (LAMMPS uses the closest image index).
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { DRUDE_TYPE, CORE_TYPE, drudeModelOf, type DrudeModel } from './drude_model';

export class FixDrudeTransform extends Fix {
  readonly style: string;
  /** Per type: (1 - mcoeff) scales the mass; 2 = not a Drude type (LAMMPS mcoeff). */
  private mcoeff = new Float64Array(0);
  private model: DrudeModel | null = null;

  constructor(sys: System, id: string, group: string, args: string[], private readonly inverse: boolean) {
    super(sys, id, group, args);
    this.style = inverse ? 'drude/transform/inverse' : 'drude/transform/direct';
    // fix_drude_transform.cpp: one message for both styles
    if (args.length) throw new StyleError('Illegal fix drude/transform command');
  }

  init(): void {
    this.model = drudeModelOf(this.sys.state, `fix ${this.style}`);
  }

  /** Setup computes the reduced-mass coefficients (LAMMPS setup()); it applies no post_force. */
  override setup(): void {
    const s = this.sys.state;
    const model = drudeModelOf(s, `fix ${this.style}`);
    this.model = model;
    const nt = s.ntypes;
    const mc = new Float64Array(nt + 1).fill(2.0);
    const partner = model.partnerIndex(s);
    for (let i = 0; i < s.n; i++) {
      if (model.kind[s.type[i]] !== DRUDE_TYPE) continue;
      const j = partner[i];
      if (mc[s.type[i]] < 1.5) {
        if (mc[s.type[j]] > 1.5) throw new StyleError('There must be one Drude type per core type');
        continue;
      }
      const md = s.massByType[s.type[i]], mcore = s.massByType[s.type[j]];
      mc[s.type[i]] = md / (md + mcore);
      mc[s.type[j]] = -md / mcore;
    }
    this.mcoeff = mc;
  }

  override initialIntegrate(): void {
    if (this.inverse) this.reducedToReal(); else this.realToReduced();
  }

  override finalIntegrate(): void {
    if (this.inverse) this.reducedToReal(); else this.realToReduced();
  }

  private realToReduced(): void {
    const s = this.sys.state;
    const model = this.requireModel();
    const partner = model.partnerIndex(s);
    const dim = this.sys.dimension;
    const mc = this.mcoeff;
    for (let t = 1; t <= s.ntypes; t++) if (mc[t] < 1.5) s.massByType[t] *= 1.0 - mc[t];
    // the reduced state changes the values of the computes the thermostats read (LAMMPS evaluates them on demand)
    const { x, v, f, mask, type } = s;
    const d = new Float64Array(3);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit) || model.kind[type[i]] !== CORE_TYPE) continue;
      const j = partner[i];
      const coeff = mc[type[j]];
      for (let k = 0; k < 3; k++) d[k] = x[3 * j + k] - x[3 * i + k];
      this.sys.geom.minimumImage(d);
      for (let k = 0; k < dim; k++) {
        x[3 * j + k] = d[k];
        x[3 * i + k] += coeff * x[3 * j + k];
        v[3 * j + k] -= v[3 * i + k];
        v[3 * i + k] += coeff * v[3 * j + k];
        f[3 * i + k] += f[3 * j + k];
        f[3 * j + k] -= coeff * f[3 * i + k];
      }
    }
    this.sys.refreshComputes();
  }

  private reducedToReal(): void {
    const s = this.sys.state;
    const model = this.requireModel();
    const partner = model.partnerIndex(s);
    const dim = this.sys.dimension;
    const mc = this.mcoeff;
    const { x, v, f, mask, type } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit) || model.kind[type[i]] !== CORE_TYPE) continue;
      const j = partner[i];
      const coeff = mc[type[j]];
      for (let k = 0; k < dim; k++) {
        x[3 * i + k] -= coeff * x[3 * j + k];
        x[3 * j + k] += x[3 * i + k];
        v[3 * i + k] -= coeff * v[3 * j + k];
        v[3 * j + k] += v[3 * i + k];
        f[3 * j + k] += coeff * f[3 * i + k];
        f[3 * i + k] -= f[3 * j + k];
      }
    }
    for (let t = 1; t <= s.ntypes; t++) if (mc[t] < 1.5) s.massByType[t] /= 1.0 - mc[t];
    this.sys.refreshComputes();
  }

  private requireModel(): DrudeModel {
    if (!this.model) this.model = drudeModelOf(this.sys.state, `fix ${this.style}`);
    return this.model;
  }
}
