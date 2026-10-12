/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/compute_temp_drude.cpp and
 * src/DRUDE/compute_temp_drude.h (class ComputeTempDrude, compute temp/drude). Copyright (2003) Sandia Corporation;
 * GPL-2.0 (src/engine-gpl/LICENSE). DRUDE package authors: Alain Dequidt, Julien Devemy, Agilio Padua; contributors
 * Kateryna Goloviznina and Zheng Gong.
 *
 * compute temp/drude: the temperatures of the atoms (core, with its Drude particle as the relative motion) and of
 * the Drude particles, from the centre-of-mass and relative kinetic energies of each core / Drude pair.
 * Vector: temp_core, temp_drude, dof_core, dof_drude, kineng_core, kineng_drude. Scalar: temp_core.
 */
import { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { massOf } from '../../engine/atoms';
import { DRUDE_TYPE, NOPOL_TYPE, drudeModelOf } from './drude_model';

export class ComputeTempDrude extends Compute {
  readonly style = 'temp/drude';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`Incorrect number of arguments for compute temp/drude (${args.join(' ')})`);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extscalar = 0;
    this.extvector = -1;
    this.extlist = [0, 0, 1, 1, 1, 1];
    this.vector = new Float64Array(6);
  }

  init(): void {
    drudeModelOf(this.sys.state, 'compute temp/drude');
  }

  protected computeScalar(): number {
    this.computeVector();
    return this.vector[0];
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const model = drudeModelOf(s, 'compute temp/drude');
    const partner = model.partnerIndex(s);
    const dim = this.sys.dimension;
    const { v, type, mask } = s;
    let nCore = 0, nDrude = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      if (model.kind[type[i]] === DRUDE_TYPE) nDrude++;
      else nCore++;
    }
    const dofCore = dim * nCore - this.sys.dofRemoved(this.groupBit);
    const dofDrude = dim * nDrude;
    let keCore = 0, keDrude = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit) || model.kind[type[i]] === DRUDE_TYPE) continue;
      const mi = massOf(s, i);
      if (model.kind[type[i]] === NOPOL_TYPE) {
        let e = 0;
        for (let k = 0; k < dim; k++) e += v[3 * i + k] * v[3 * i + k];
        keCore += mi * e;
      } else {
        const j = partner[i];
        if (j < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[i]} is not defined`);
        const mj = massOf(s, j);
        const mtotInv = 1.0 / (mi + mj);
        let ecore = 0, edrude = 0;
        for (let k = 0; k < dim; k++) {
          const vc = v[3 * i + k], vd = v[3 * j + k];
          const v1 = mj * vd + mi * vc;
          ecore += v1 * v1;
          const v2 = vd - vc;
          edrude += v2 * v2;
        }
        keCore += mtotInv * ecore;
        keDrude += mtotInv * mi * mj * edrude;
      }
    }
    const mvv2e = s.units.mvv2e, kb = s.units.boltz;
    keCore *= 0.5 * mvv2e;
    keDrude *= 0.5 * mvv2e;
    this.vector[0] = (2.0 * keCore) / (dofCore * kb);
    this.vector[1] = (2.0 * keDrude) / (dofDrude * kb);
    this.vector[2] = dofCore;
    this.vector[3] = dofDrude;
    this.vector[4] = keCore;
    this.vector[5] = keDrude;
  }
}
