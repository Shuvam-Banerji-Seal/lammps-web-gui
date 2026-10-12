/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ELECTRODE/fix_electrode_conq.cpp and fix_electrode_conq.h (the constant-charge fix: the electrode charges
 * are the targets, the potentials follow from the elastance matrix; with the CG algorithms the charge is held per
 * electrode, and the group potentials are averaged for output) at commit b332f28. ELECTRODE contributing authors:
 * Ludwig Ahrens-Iwers (TUHH), Shern Tee (UQ), Robert Meissner (TUHH), and the LAMMPS developers. Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { FixElectrodeConp } from './fix_electrode_conp';

export class FixElectrodeConq extends FixElectrodeConp {
  readonly style: string = 'electrode/conq';
  /** Target charge of each electrode (the potential argument of the conp fix, read as a charge). */
  private readonly groupQ: number[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // copy const-style values across because update_psi will change group_psi
    this.groupQ = [...this.psiConst];
    if (this.symm) {
      if (this.groupBits.length === 1) throw new StyleError('Keyword symm on not allowed in electrode/conq with only one electrode');
      this.sys.warn('Fix electrode/conq with keyword symm ignores the charge setting for the last electrode listed');
      if (this.algo !== 'mat_inv') {
        // needed for CG algos
        let lastQ = 0;
        for (let g = 0; g < this.groupBits.length - 1; g++) lastQ -= this.groupQ[g];
        this.groupQ[this.groupBits.length - 1] = lastQ;
      }
    }
  }

  protected allowsQtotal(): boolean { return false; }
  protected styleName(): string { return 'electrode/conq'; }

  private groupCount(): number[] {
    const count = new Array<number>(this.groupBits.length).fill(0);
    for (let k = 0; k < this.ngroup; k++) count[this.groupOfElec[k]]++;
    return count;
  }

  /** update_psi(): the group charges (variables update them) give the potentials through the elastance. */
  protected updatePsi(): void {
    const G = this.groupBits.length;
    const numsymm = G - (this.symm ? 1 : 0);
    let symmUpdateBack = false;
    for (let g = 0; g < numsymm; g++) {
      const v = this.psiVar[g];
      if (v === null) continue;
      this.groupQ[g] = this.sys.vars.scalar(v, this.sys.formulaEnv);
      symmUpdateBack = true;
    }
    if (this.algo === 'mat_inv') {
      const remainder = new Array<number>(G).fill(0);
      for (let g = 0; g < numsymm; g++) remainder[g] = this.groupQ[g] - this.sbCharges[g];
      for (let g = 0; g < G; g++) {
        let vtmp = 0;
        for (let h = 0; h < G; h++) vtmp += this.macroElastance[g][h] * remainder[h];
        this.groupPsi[g] = vtmp;
      }
    } else {
      if (this.symm && symmUpdateBack) {
        // needed for CG algos
        let lastQ = 0;
        for (let g = 0; g < G - 1; g++) lastQ -= this.groupQ[g];
        this.groupQ[G - 1] = lastQ;
      }
      this.groupPsi.fill(0);
    }
  }

  /** Correct the charge of each electrode to its target by adding a homogeneous charge. */
  protected constraintCorrection(x: Float64Array): Float64Array {
    const G = this.groupBits.length, count = this.groupCount();
    const sums = new Array<number>(G).fill(0);
    for (let i = 0; i < x.length; i++) sums[this.groupOfElec[i]] += x[i];
    for (let g = 0; g < G; g++) {
      sums[g] -= this.groupQ[g];
      sums[g] /= count[g];
    }
    const out = Float64Array.from(x);
    for (let i = 0; i < out.length; i++) out[i] -= sums[this.groupOfElec[i]];
    return out;
  }

  /** Project into the direction that conserves the charge of each electrode (cf. M. Shariff (1995)). */
  protected constraintProjection(x: Float64Array): Float64Array {
    const G = this.groupBits.length, count = this.groupCount();
    const sums = new Array<number>(G).fill(0);
    for (let i = 0; i < x.length; i++) sums[this.groupOfElec[i]] += x[i];
    for (let g = 0; g < G; g++) sums[g] /= count[g];
    const out = Float64Array.from(x);
    for (let i = 0; i < out.length; i++) out[i] -= sums[this.groupOfElec[i]];
    return out;
  }

  /** The group potential as the average over each electrode, for output with the CG algorithms. */
  protected recomputePotential(b: Float64Array, q: Float64Array): void {
    const G = this.groupBits.length, count = this.groupCount();
    const a = this.eleEle(q);
    const psiSums = new Array<number>(G).fill(0);
    for (let i = 0; i < b.length; i++) psiSums[this.groupOfElec[i]] += (a[i] + b[i]) / this.evscale;
    for (let g = 0; g < G; g++) this.groupPsi[g] = psiSums[g] / count[g];
  }
}
