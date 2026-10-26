/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the parts of src/ML-POD/eapod.cpp and
 * eapod.h (class EAPOD; contributing author Ngoc Cuong Nguyen (MIT); copyright Sandia Corporation) that the fitpod command
 * uses: peratombase_descriptors with the descriptor derivatives (twobodydescderiv, threebodydescderiv, fourbodydescderiv),
 * myneighbors, descriptors (single environment), base_descriptors, peratomenergyforce, energyforce, tallyforce and
 * mknewcoeff. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The radial and angular bases and the descriptor values come from the evaluation port in eapod.ts (PODModel), which
 * pair_style pod uses; the derivatives here follow the native functions term by term. DGEMV calls are written out in the
 * reference BLAS order (native links OpenBLAS, whose kernels may sum in another order: agreement is at rounding level).
 */
import { PODModel, type PODCoeffFile, type PODParams } from './eapod';

/** The neighbour data of one configuration (fitpod podfullneighborlist): positions y, owner alist, CSR jlist. */
export interface PodNeighbors {
  y: Float64Array;
  alist: Int32Array;
  jlist: Int32Array;
  pairnumsum: Int32Array;
}

export class FitPODModel extends PODModel {
  /** onebody descriptor count (nl1 = nd1 = 1: the engine's POD port takes onebody 1 only). */
  readonly nl1 = 1;
  private bdBuf = new Float64Array(0);
  private bddBuf = new Float64Array(0);

  constructor(params: PODParams, coeffFile: PODCoeffFile) {
    super(params, coeffFile);
  }

  /** EAPOD::mknewcoeff: the model takes the fitted coefficients. */
  mknewcoeff(c: Float64Array): void {
    this.coeff.set(c.subarray(0, this.nCoeffAll));
  }

  /** EAPOD::myneighbors: r_ij = x_j - x_i of the neighbours of atom i, their owners and types. */
  private myneighbors(nb: PodNeighbors, x: Float64Array, atomtype: Int32Array, i: number) {
    const start = nb.pairnumsum[i];
    const m = nb.pairnumsum[i + 1] - start;
    const rij = new Float64Array(3 * m);
    const ai = new Int32Array(m), aj = new Int32Array(m), tj = new Int32Array(m);
    const y = nb.y;
    for (let l = 0; l < m; l++) {
      const j = nb.jlist[l + start];
      ai[l] = i;
      aj[l] = nb.alist[j];
      tj[l] = atomtype[nb.alist[j]];
      rij[3 * l] = y[3 * j] - y[3 * i];
      rij[3 * l + 1] = y[3 * j + 1] - y[3 * i + 1];
      rij[3 * l + 2] = y[3 * j + 2] - y[3 * i + 2];
    }
    return { rij, ai, aj, ti: atomtype[i], tj, m };
  }

  /** EAPOD::peratombase_descriptors: bd (Mdesc) and bdd (3*Nj*Mdesc, bdd[d + 3*n + 3*Nj*m]) of one atom. */
  peratombaseDescriptors(rij: Float64Array, tj: Int32Array, nj: number): { bd: Float64Array; bdd: Float64Array } {
    const Mdesc = this.Mdesc;
    if (this.bdBuf.length < Mdesc) this.bdBuf = new Float64Array(Mdesc);
    if (this.bddBuf.length < 3 * nj * Mdesc) this.bddBuf = new Float64Array(3 * nj * Mdesc);
    const bd = this.bdBuf, bdd = this.bddBuf;
    bd.fill(0, 0, Mdesc);
    bdd.fill(0, 0, 3 * nj * Mdesc);
    if (nj === 0) return { bd, bdd };
    const nl2 = this.nl2, nl3 = this.nl3, nl4 = this.nl4;
    const s = this.scratch(nj);
    this.radialbasis(s, rij, nj);
    this.projectRadial(s, nj);
    if (nl2 > 0) {
      // twobodydescderiv
      const nrbf2 = this.nrbf2;
      for (let m = 0; m < nrbf2; m++) {
        for (let n = 0; n < nj; n++) {
          const i2 = n + nj * m;
          const i1 = n + nj * m + nj * nrbf2 * (tj[n] - 1);
          bd[m + nrbf2 * (tj[n] - 1)] += s.rbf[i2];
          bdd[3 * i1] += s.rbfx[i2];
          bdd[1 + 3 * i1] += s.rbfy[i2];
          bdd[2 + 3 * i1] += s.rbfz[i2];
        }
      }
    }
    if (nl3 > 0 && nj > 1) {
      this.angularbasis(s, rij, nj);
      this.radialangularbasis(s, tj, nj);
      this.threebodydesc(bd, nl2);
      this.threebodydescderiv(bdd, 3 * nj * nl2, s.ux, s.uy, s.uz, tj, nj);
      if (nl4 > 0 && nj > 2) this.fourbodydescderiv(bd, nl2 + nl3, bdd, 3 * nj * (nl2 + nl3), s.ux, s.uy, s.uz, tj, nj);
    }
    return { bd, bdd };
  }

  /** EAPOD::threebodydescderiv: dd3 at offset o (Ux etc. in the K3 layout). */
  private threebodydescderiv(dd3: Float64Array, o: number, Ux: Float64Array, Uy: Float64Array, Uz: Float64Array, atomtype: Int32Array, N: number): void {
    const Ne = this.nelements, K3 = this.K3, nrbf3 = this.nrbf3, nabf3 = this.nabf3;
    const sumU = this.sumU, pn3 = this.pn3, pc3 = this.pc3;
    if (Ne === 1) {
      for (let m = 0; m < nrbf3; m++) {
        for (let p = 0; p < nabf3; p++) {
          const n1 = pn3[p], nn = pn3[p + 1] - n1;
          for (let q = 0; q < nn; q++) {
            const t1 = pc3[n1 + q] * sumU[(n1 + q) + K3 * m];
            for (let j = 0; j < N; j++) {
              const f = 2.0 * t1;
              const ii = o + 3 * j + 3 * N * (p + nabf3 * m);
              const jj = j + N * (n1 + q) + N * K3 * m;
              dd3[ii] += f * Ux[jj];
              dd3[1 + ii] += f * Uy[jj];
              dd3[2 + ii] += f * Uz[jj];
            }
          }
        }
      }
      return;
    }
    for (let m = 0; m < nrbf3; m++) {
      for (let p = 0; p < nabf3; p++) {
        const n1 = pn3[p], nn = pn3[p + 1] - n1;
        for (let q = 0; q < nn; q++) {
          for (let i1 = 0; i1 < Ne; i1++) {
            const t1 = pc3[n1 + q] * sumU[i1 + Ne * (n1 + q) + Ne * K3 * m];
            for (let j = 0; j < N; j++) {
              const i2 = atomtype[j] - 1;
              const k = this.elemindex[i2 + Ne * i1];
              const f = i1 === i2 ? 2.0 * t1 : t1;
              const ii = o + 3 * j + 3 * N * (p + nabf3 * m + nabf3 * nrbf3 * k);
              const jj = j + N * (n1 + q) + N * K3 * m;
              dd3[ii] += f * Ux[jj];
              dd3[1 + ii] += f * Uy[jj];
              dd3[2 + ii] += f * Uz[jj];
            }
          }
        }
      }
    }
  }

  /**
   * EAPOD::fourbodydescderiv: d4 at offset od and dd4 at offset o. Native first copies sumU, Ux, Uy, Uz into the K4
   * layout when K4 < K3; reading them in the K3 layout gives the same values.
   */
  private fourbodydescderiv(d4: Float64Array, od: number, dd4: Float64Array, o: number, Ux: Float64Array, Uy: Float64Array, Uz: Float64Array, atomtype: Int32Array, N: number): void {
    const Ne = this.nelements, K = this.K3, nrbf4 = this.nrbf4, nabf4 = this.nabf4;
    const sumU = this.sumU, pa4 = this.pa4, pb4 = this.pb4, pc4 = this.pc4;
    const Me = Ne * (Ne + 1) * (Ne + 2) / 6;
    for (let m = 0; m < nabf4 * nrbf4 * Me; m++) d4[od + m] = 0.0;
    const Q = pa4[nabf4];
    if (Ne === 1) {
      for (let m = 0; m < nrbf4; m++) {
        for (let p = 0; p < nabf4; p++) {
          const n1 = pa4[p], nn = pa4[p + 1] - n1;
          for (let q = 0; q < nn; q++) {
            const c = pc4[n1 + q];
            const j1 = pb4[n1 + q], j2 = pb4[n1 + q + Q], j3 = pb4[n1 + q + 2 * Q];
            const c1 = c * sumU[j1 + K * m];
            const c2 = c * sumU[j2 + K * m];
            const t12 = c1 * sumU[j2 + K * m];
            const c3 = sumU[j3 + K * m];
            const t13 = c1 * c3;
            const t23 = c2 * c3;
            const kk = p + nabf4 * m;
            const ii = o + 3 * N * (p + nabf4 * m);
            d4[od + kk] += t12 * c3;
            for (let j = 0; j < N; j++) {
              let jj = j + N * j3 + N * K * m;
              dd4[3 * j + ii] += t12 * Ux[jj];
              dd4[1 + 3 * j + ii] += t12 * Uy[jj];
              dd4[2 + 3 * j + ii] += t12 * Uz[jj];
              jj = j + N * j2 + N * K * m;
              dd4[3 * j + ii] += t13 * Ux[jj];
              dd4[1 + 3 * j + ii] += t13 * Uy[jj];
              dd4[2 + 3 * j + ii] += t13 * Uz[jj];
              jj = j + N * j1 + N * K * m;
              dd4[3 * j + ii] += t23 * Ux[jj];
              dd4[1 + 3 * j + ii] += t23 * Uy[jj];
              dd4[2 + 3 * j + ii] += t23 * Uz[jj];
            }
          }
        }
      }
      return;
    }
    for (let m = 0; m < nrbf4; m++) {
      for (let p = 0; p < nabf4; p++) {
        const n1 = pa4[p], nn = pa4[p + 1] - n1;
        for (let q = 0; q < nn; q++) {
          const c = pc4[n1 + q];
          const j1 = pb4[n1 + q], j2 = pb4[n1 + q + Q], j3 = pb4[n1 + q + 2 * Q];
          let k = 0;
          for (let i1 = 0; i1 < Ne; i1++) {
            const c1 = c * sumU[i1 + Ne * j1 + Ne * K * m];
            for (let i2 = i1; i2 < Ne; i2++) {
              const c2 = c * sumU[i2 + Ne * j2 + Ne * K * m];
              const t12 = c1 * sumU[i2 + Ne * j2 + Ne * K * m];
              for (let i3 = i2; i3 < Ne; i3++) {
                const c3 = sumU[i3 + Ne * j3 + Ne * K * m];
                const t13 = c1 * c3;
                const t23 = c2 * c3;
                const kk = p + nabf4 * m + nabf4 * nrbf4 * k;
                const ii = o + 3 * N * (p + nabf4 * m + nabf4 * nrbf4 * k);
                d4[od + kk] += t12 * c3;
                for (let j = 0; j < N; j++) {
                  const tj = atomtype[j] - 1;
                  if (tj === i3) {
                    const jj = j + N * j3 + N * K * m;
                    dd4[3 * j + ii] += t12 * Ux[jj];
                    dd4[1 + 3 * j + ii] += t12 * Uy[jj];
                    dd4[2 + 3 * j + ii] += t12 * Uz[jj];
                  }
                  if (tj === i2) {
                    const jj = j + N * j2 + N * K * m;
                    dd4[3 * j + ii] += t13 * Ux[jj];
                    dd4[1 + 3 * j + ii] += t13 * Uy[jj];
                    dd4[2 + 3 * j + ii] += t13 * Uz[jj];
                  }
                  if (tj === i1) {
                    const jj = j + N * j1 + N * K * m;
                    dd4[3 * j + ii] += t23 * Ux[jj];
                    dd4[1 + 3 * j + ii] += t23 * Uy[jj];
                    dd4[2 + 3 * j + ii] += t23 * Uz[jj];
                  }
                }
                k += 1;
              }
            }
          }
        }
      }
    }
  }

  /**
   * EAPOD::descriptors (one environment): the global descriptors gd (nCoeffAll) of a configuration, their derivatives gdd
   * (gdd[d + 3*atom + 3*natom*k]) and the per-atom base descriptors basedesc[i + natom*m].
   */
  descriptors(gd: Float64Array, gdd: Float64Array, basedesc: Float64Array, nb: PodNeighbors, atomtype: Int32Array, natom: number): void {
    const nCoeffAll = this.nCoeffAll, Mdesc = this.Mdesc, ncpe = this.nCoeffPerElement;
    gd.fill(0, 0, nCoeffAll);
    gdd.fill(0, 0, 3 * natom * nCoeffAll);
    basedesc.fill(0, 0, natom * Mdesc);
    for (let i = 0; i < natom; i++) {
      const Nj = nb.pairnumsum[i + 1] - nb.pairnumsum[i];
      // one-body descriptor
      gd[ncpe * (atomtype[i] - 1)] += 1.0;
      if (Nj > 0) {
        const { rij, ai, aj, ti, tj } = this.myneighbors(nb, nb.y, atomtype, i);
        const { bd, bdd } = this.peratombaseDescriptors(rij, tj, Nj);
        for (let m = 0; m < Mdesc; m++) {
          basedesc[i + natom * m] = bd[m];
          const k = ncpe * (ti - 1) + this.nl1 + m;
          gd[k] += bd[m];
          for (let n = 0; n < Nj; n++) {
            const im = 3 * ai[n] + 3 * natom * k;
            const jm = 3 * aj[n] + 3 * natom * k;
            const nm = 3 * n + 3 * Nj * m;
            gdd[im] += bdd[nm];
            gdd[1 + im] += bdd[1 + nm];
            gdd[2 + im] += bdd[2 + nm];
            gdd[jm] -= bdd[nm];
            gdd[1 + jm] -= bdd[1 + nm];
            gdd[2 + jm] -= bdd[2 + nm];
          }
        }
      }
    }
  }

  /** EAPOD::base_descriptors: basedesc[i + natom*m]. */
  baseDescriptorsAll(basedesc: Float64Array, nb: PodNeighbors, atomtype: Int32Array, natom: number): void {
    const Mdesc = this.Mdesc;
    basedesc.fill(0, 0, natom * Mdesc);
    for (let i = 0; i < natom; i++) {
      const Nj = nb.pairnumsum[i + 1] - nb.pairnumsum[i];
      if (Nj > 0) {
        const { rij, tj } = this.myneighbors(nb, nb.y, atomtype, i);
        const { bd } = this.peratombaseDescriptors(rij, tj, Nj);
        for (let m = 0; m < Mdesc; m++) basedesc[i + natom * m] = bd[m];
      }
    }
  }

  /** EAPOD::energyforce (one environment): the energy of a configuration and the forces (3 per atom). */
  energyforce(force: Float64Array, nb: PodNeighbors, atomtype: Int32Array, natom: number): number {
    let etot = 0.0;
    force.fill(0, 0, 3 * natom);
    const coeff = this.coeff, ncpe = this.nCoeffPerElement, Mdesc = this.Mdesc;
    for (let i = 0; i < natom; i++) {
      const Nj = nb.pairnumsum[i + 1] - nb.pairnumsum[i];
      if (Nj === 0) {
        etot += coeff[ncpe * (atomtype[i] - 1)];
        continue;
      }
      const { rij, ai, aj, ti, tj } = this.myneighbors(nb, nb.y, atomtype, i);
      // peratomenergyforce, single environment
      const c1 = ncpe * (ti - 1);
      let e = coeff[c1];
      const { bd, bdd } = this.peratombaseDescriptors(rij, tj, Nj);
      for (let m = 0; m < Mdesc; m++) e += coeff[c1 + 1 + m] * bd[m];
      // DGEMV('N', 3*Nj, Mdesc, 1, bdd, 3*Nj, coeff1 + 1, 1, 0, fij, 1), the reference order: columns outer
      const N = 3 * Nj;
      const fij = new Float64Array(N);
      for (let m = 0; m < Mdesc; m++) {
        const temp = coeff[c1 + 1 + m];
        for (let r = 0; r < N; r++) fij[r] += temp * bdd[r + N * m];
      }
      etot += e;
      // tallyforce
      for (let n = 0; n < Nj; n++) {
        const im = 3 * ai[n], jm = 3 * aj[n], nm = 3 * n;
        force[im] += fij[nm]; force[1 + im] += fij[1 + nm]; force[2 + im] += fij[2 + nm];
        force[jm] -= fij[nm]; force[1 + jm] -= fij[1 + nm]; force[2 + jm] -= fij[2 + nm];
      }
    }
    return etot;
  }
}
