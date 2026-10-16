/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/PERI/pair_peri_ves.cpp and pair_peri_ves.h (PairPeriVES: pair_style peri/ves) at commit b332f28.
 * Contributing authors: Rezwanur Rahman, J.T. Foster (UTSA). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * pair_style peri/ves: the viscoelastic state-based peridynamic model (docs.lammps.org/pair_peri.html). pair_coeff
 * I J K G cut s00 alpha lambdai taui: the deviatoric force state is split into an elastic part (weight 1 - lambdai)
 * and a viscoelastic part (weight lambdai) whose back extension relaxes over the time constant taui.
 */
import { PERI_GEOM_LEN, G_DR, G_DX, G_DY, G_DZ, G_J, G_OM, G_R, G_R0, G_VS, PeriBondPair, type PeriStep } from './pair_peri_bonds';
import { exp as libmExp } from '../libm/exp';

const PERI_VES_NAMES = ['K', 'G', 'cut', 's00', 'alpha', 'lambdai', 'taubi'] as const;

export class PairPeriVES extends PeriBondPair {
  readonly name = 'peri/ves';
  protected readonly names = PERI_VES_NAMES;

  protected bondForces(st: PeriStep, smin_new: Float64Array, s0_new: Float64Array): void {
    const { nl, f, V, type, dt } = st;
    const g = new Float64Array(PERI_GEOM_LEN);
    for (let i = 0; i < nl; i++) {
      const ti = type[i];
      const G = this.c('G', ti, ti);
      const lambdai = this.c('lambdai', ti, ti);
      const taui = this.c('taubi', ti, ti);
      // exponential relaxation of the back extension over one step (pair_peri_ves.cpp: c1, decay, betai)
      const c1 = taui / dt;
      const decay = libmExp(-1.0 / c1);
      const betai = 1.0 - c1 * (1.0 - decay);
      const thi = this.theta[i];
      let first = true;
      for (let k = this.pStart[i]; k < this.pStart[i + 1]; k++) {
        if (this.pBr[k]) continue;
        this.bondGeom(st, i, k, g);
        const j = g[G_J], dx = g[G_DX], dy = g[G_DY], dz = g[G_DZ];
        const r = g[G_R], dr = g[G_DR], r0 = g[G_R0], om = g[G_OM], vs = g[G_VS];
        const tj = type[j];
        const wi = this.wvol[i], wj = this.wvol[j];
        const vj = V[j] * vs;
        // elastic part (pair_peri_ves.cpp: rk, fbond)
        const rk = 3.0 * this.c('K', ti, ti) * vj * (om * thi / wi + om * this.theta[j] / wj) * r0;
        let fbond = r > 0 ? -(rk / r) : 0;
        // viscoelastic part: deviatoric extension and its back extension at this step
        const devext = dr - (thi * r0) / 3.0;
        const deltaed = devext - this.devA[k];
        const edbNp1 = this.devA[k] * (1 - decay) + this.devB[k] * decay + betai * deltaed;
        const shear = G * vj * (om / wi + om / wj);
        let rkNew = (1 - lambdai) * 15.0 * shear * devext;
        rkNew += lambdai * 15.0 * shear * (devext - edbNp1);
        const fbondVE = r > 0 ? -(rkNew / r) : 0;
        fbond += fbondVE;
        f[3 * i] += dx * fbond; f[3 * i + 1] += dy * fbond; f[3 * i + 2] += dz * fbond;
        // energy of the directed bond (eflag block), tallied as LAMMPS does for the two halves
        const evdwl = (0.5 * 15 * (1 - lambdai) * G / wi * om * devext * devext
          + 0.5 * 15 * lambdai * G / wi * om * (devext - edbNp1) * (devext - edbNp1)) * V[j] * vs;
        this.tallyBond(st, i, evdwl, fbond, g);
        this.devA[k] = devext;
        this.devB[k] = edbNp1;
        this.breakAndTrack(i, k, dr / r0, smin_new, s0_new, first, ti, tj, j);
        first = false;
      }
    }
  }
}
