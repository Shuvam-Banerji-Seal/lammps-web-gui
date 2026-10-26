/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/fix_polarize_bem_icc.cpp and fix_polarize_bem_icc.h at commit b332f28 (Trung Nguyen and Monica Olvera de la Cruz, Northwestern University, and the DIELECTRIC package authors). Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix polarize/bem/icc N tol: induced surface charges by the image-charge (ICC) iteration of Tyagi et al. (J. Chem.
 * Phys. 2010, 132, 154112) with the boundary element form of Barros, Sinkovits, Luijten (J. Chem. Phys. 2014, 140,
 * 064903), relaxed by successive over-relaxation (fix_modify omega, default 0.7). Each sweep evaluates the force field
 * once; the sweep is Jacobi-like (all patches use the field of the sweep). Not ported: fix_modify rand is refused.
 */
import { Fix, FIX_MODIFY_KEYWORDS } from '../../engine/fix/fix';
import { inumeric, logical, numeric, parseNum } from '../../engine/force/util';
import { isDielectricStyle } from '../../engine/atoms';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { dielArrays, dielectricKSpaceField, keepingVirial, dielectricPairField, epsilon0e2q, fieldAt } from './common';
import { polarizeModify, setDielectricParams } from './fix_polarize_bem_gmres';

const MY_4PI = 4 * Math.PI;
const MY_2PI = 2 * Math.PI;

export class FixPolarizeBEMICC extends Fix {
  readonly style = 'polarize/bem/icc';
  vectorFlag = true;
  sizeVector = 2;

  private readonly every: number;
  private readonly tol: number;
  private itrMax = 50;
  private omega = 0.7;
  private kspaceOn = true;
  private iterations = 0;
  private rho = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_polarize_bem_icc.cpp constructor: the count, the atom style, then N (utils::inumeric, >= 0) and tol
    if (args.length < 2) throw new StyleError('Illegal fix polarize/bem/icc command');
    if (!isDielectricStyle(sys.state.atomStyle)) throw new StyleError('Fix polarize requires atom style dielectric');
    const every = inumeric(args[0]);
    if (every < 0) throw new StyleError('Illegal fix polarize/bem/icc command');
    this.every = every;
    this.tol = parseNum(args[1], 'tolerance');
  }

  private evaluate(): void {
    this.sys.computeForcesInRun({});
  }

  /** The kspace field (pppm/dielectric or msm/dielectric) when the fix uses kspace, else null. */
  private kspaceField(): Float64Array | null {
    const k = this.sys.ff.kspace;
    return k && this.kspaceOn ? dielectricKSpaceField(k as { name: string; efield?: Float64Array }) : null;
  }

  private pairE(): Float64Array {
    const e = dielectricPairField(this.sys.ff.pair);
    if (!e) throw new StyleError(`Pair style not compatible with fix ${this.id} (${this.style})`);
    return e;
  }

  setup(): void {
    this.pairE();
    const k = this.sys.ff.kspace;
    if (k && this.kspaceOn && !dielectricKSpaceField(k as { name: string; efield?: Float64Array })) throw new StyleError(`Kspace style not compatible with fix ${this.id} (${this.style})`);
    // LAMMPS leaves the energies and forces of the last field evaluation inside the solver (no force
    // evaluation follows the setup in Verlet::setup), so they are not recomputed with the final charges here
    keepingVirial(this.sys.ff.acc, () => this.computeInduced());
  }

  preForce(): void {
    if (this.every === 0) return;
    if (this.sys.state.step % this.every) return;
    this.computeInduced();
  }

  private computeInduced(): void {
    const s = this.sys.state;
    const n = s.n;
    const d = dielArrays(s, `fix ${this.id}`);
    const e0 = epsilon0e2q(s.units.style);
    const qs = d.qs, q = s.q;
    const E = new Float64Array(3);
    // Eq. (64) of Barros et al.: the initial scaled charges from the field of the current charges
    this.evaluate();
    for (let i = 0; i < n; i++) {
      if (!this.inGroup(i)) continue;
      fieldAt(this.pairE(), this.kspaceField(), i, E);
      const ndotE = e0 * (E[0] * d.mu[4 * i] + E[1] * d.mu[4 * i + 1] + E[2] * d.mu[4 * i + 2]) / d.epsilon[i] / MY_2PI;
      const qFree = q[i];
      const qBound = (1.0 / d.em[i] - 1) * qFree - (d.ed[i] / (2 * d.em[i])) * ndotE * d.area[i];
      qs[i] = qFree + qBound;
    }
    let itr = 0;
    let rho = 0;
    for (itr = 0; itr < this.itrMax; itr++) {
      this.evaluate();
      const pe = this.pairE();
      let tol = 0;
      for (let i = 0; i < n; i++) {
        if (!this.inGroup(i)) continue;
        const qFree = q[i];
        const qtmp = qs[i] - qFree;
        fieldAt(pe, this.kspaceField(), i, E);
        // Eq. (69) of Barros et al.: the bound charge of patch i relaxed towards the field value
        const ndotE = e0 * (E[0] * d.mu[4 * i] + E[1] * d.mu[4 * i + 1] + E[2] * d.mu[4 * i + 2]) / MY_4PI / d.epsilon[i];
        let qBound = qs[i] - qFree;
        qBound = (1 - this.omega) * qBound + this.omega * ((1.0 / d.em[i] - 1) * qFree - (d.ed[i] / d.em[i]) * ndotE * d.area[i]);
        qs[i] = qFree + qBound;
        const delta = Math.abs(qtmp - qBound);
        const r = Math.abs(qtmp) > 0 ? delta / Math.abs(qtmp) : 0;
        if (tol < r) tol = r;
      }
      rho = tol;
      if (itr > 0 && rho < this.tol) break;
    }
    this.iterations = itr;
    this.rho = rho;
  }

  modify(key: string, values: string[]): number {
    if (FIX_MODIFY_KEYWORDS.has(key)) return super.modify(key, values);
    return polarizeModify(key, values, {
      itr_max: { n: 1, apply: ([w]) => { this.itrMax = inumeric(w); } },
      omega: { n: 1, apply: ([w]) => { this.omega = numeric(w); } },
      kspace: { n: 1, apply: ([w]) => { this.kspaceOn = logical(w); } },
      dielectrics: { n: 5, apply: (v) => setDielectricParams(this.sys, (i) => this.inGroup(i), v, this.id, this.style, true) },
      rand: { n: 2, apply: ([a, b]) => {
        numeric(a); inumeric(b);
        throw new StyleError(`fix_modify rand is not supported by fix ${this.id} (${this.style}) in the browser engine`);
      } },
    });
  }

  computeVector(i: number): number {
    if (i === 0) return this.iterations;
    if (i === 1) return this.rho;
    return 0;
  }
}
