/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/ewald_dipole_spin.cpp and
 * ewald_dipole_spin.h (EwaldDipoleSpin: kspace_style ewald/dipole/spin, the long-range dipolar field of the spins,
 * with the setup of src/KSPACE/ewald_dipole.cpp: the Newton solution for g_ewald, the rms error of equation 46 of Wang
 * et al., JCP 115, 6351 (2001), used for kmax) and the k-vector set of src/KSPACE/ewald.cpp (Ewald::coeffs). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing authors (from ewald_dipole_spin.cpp): Julien Tranchida
 * (SNL) and Stan Moore (SNL).
 *
 * The moment of atom i is mu_i = sp_i * |sp_i| (the spin times its norm); the energies are scaled by
 * mub2mu0 = mub^2 mu_0 / (4 pi) (src/SPIN, mub = 9.274e-4, mu_0 = 784.15 eV/Ang/A^2), so the Ewald sums are those of
 * the point dipoles mu_i with the prefactor C = mub2mu0 in place of qqrd2e. The precession field fm_long of atom i is
 * -(|sp_i| mub2mu0 / hbar) times the field of the dipoles (native: fm_long -= sp[3] * spscale2 * tk). The k-space
 * forces act on the atoms (f += mub2mu0 ek). The Ewald energy, self term, forces and virial are those of
 * EwaldDipole (src/engine/force/kspace/ewald_dipole.ts), with the moments above.
 *
 * accuracy: the relative accuracy of kspace_style times two_charge_force (qqr2e in metal units). The real-space
 * cutoff is the pair style's cut_coul. Native's k-space vectors: the axis vectors up to kmax = max_d kmax_d, the
 * planes and the cells within |n_d| <= kmax_d, all with |K|^2 <= gsqmx = 1.00001 max_d (2 pi kmax_d / L_d)^2.
 * Triclinic and non-periodic boxes and kspace_modify slab, force, mesh, order are refused (native: not supported).
 */
import { KSpace, StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { parseNum } from '../../engine/force/util';
import { modifyNoEffect } from '../../engine/force/kspace/ewald';
import { dipoleKSpaceForceError, dipoleRealForceError } from '../../engine/force/kspace/ewald_dipole';
import { fmLongOf } from './dipole_long_field';
import { HBAR, MUB2MU0 } from './pair_spin_dipole_cut';
import { log as libmLog } from '../libm/log';
import { sinFma as libmSin, cosFma as libmCos } from '../libm/sincos_fma';
import { exp as libmExp } from '../libm/exp';

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** Ewald::setup's rms_dipole with the source's mu2 (equation 46): the kmax criterion along one box length. */
const rmsDipole = (km: number, prd: number, V: number, g: number, natoms: number, mu2: number): number =>
  dipoleKSpaceForceError(km, g, prd, V, natoms, mu2);

export class KSpaceEwaldDipoleSpin extends KSpace {
  readonly name = 'ewald/dipole/spin';
  /** kspace_modify gewald: a user value, or 0 for the automatic choice. */
  private gewaldUser = 0;
  /** kspace_modify kmax/ewald: a user value per dimension, or null for the automatic choice. */
  private kmaxUser: [number, number, number] | null = null;
  private kmaxDim: [number, number, number] = [0, 0, 0];
  private kmaxAll = 0;
  private mu2 = 0;
  private musq = 0;
  private natoms = 0;
  private cutoff = 0;
  private accuracyAbs = 0;
  /** Half-space wave vectors K (3 per vector). */
  private kvec = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('Illegal kspace_style ewald/dipole/spin command');
    this.accuracy = parseNum(args[0], 'accuracy');
    if (!(this.accuracy > 0)) throw new StyleError('KSpace accuracy must be > 0');
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'gewald': {
        this.gewaldUser = parseNum(values[0], 'gewald');
        return 1;
      }
      case 'kmax/ewald': {
        const k = [0, 1, 2].map((d) => parseNum(values[d], 'kmax/ewald'));
        if (k.some((v) => !Number.isInteger(v) || v < 0)) throw new StyleError('kmax/ewald values must be integers >= 0');
        this.kmaxUser = k.every((v) => v === 0) ? null : (k as [number, number, number]);
        return 3;
      }
      case 'force': case 'slab': case 'mesh': case 'order': case 'diff':
        throw new StyleError(`kspace_modify ${key} is not supported by kspace_style ewald/dipole/spin`);
      default:
        return modifyNoEffect(key, values);
    }
  }

  init(s: SimState, geom: Geometry, cutCoul: number, _qqrd2e: number, ctx: StyleContext): void {
    if (!s.sp) throw new StyleError('kspace_style ewald/dipole/spin requires atom_style spin');
    if (s.units.style !== 'metal') throw new StyleError("'metal' units have to be used with spins");
    const p = s.box.periodic;
    if (!p[0] || !p[1] || !p[2]) throw new StyleError('Cannot use nonperiodic boundaries with EwaldDipoleSpin');
    if (s.box.triclinic || geom.xy !== 0 || geom.xz !== 0 || geom.yz !== 0) {
      throw new StyleError('Cannot (yet) use EwaldDipoleSpin with triclinic box');
    }
    const sp = s.sp;
    let musq = 0;
    for (let i = 0; i < s.n; i++) {
      const sx = sp[4 * i] * sp[4 * i + 3], sy = sp[4 * i + 1] * sp[4 * i + 3], sz = sp[4 * i + 2] * sp[4 * i + 3];
      musq += sx * sx + sy * sy + sz * sz;
    }
    this.musq = musq;
    this.mu2 = musq * MUB2MU0;
    if (this.mu2 === 0) throw new StyleError('Using kspace solver EwaldDipoleSpin on system with no spins');
    this.natoms = s.n;
    this.cutoff = cutCoul;
    // two_charge_force of metal units (qqr2e, qelectron = angstrom = 1): the accuracy is in force units
    this.accuracyAbs = this.accuracy * s.units.qqr2e;
    const V = geom.volume(3);
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    const acc = this.accuracyAbs;
    if (this.gewaldUser > 0) {
      this.gEwald = this.gewaldUser;
    } else {
      // ewald/dipole/spin init: the old estimate, then the Newton solver (ewald_dipole.cpp NewtonSolve)
      let g = (acc * Math.sqrt(this.natoms * cutCoul * xprd * yprd * zprd)) / (2 * this.mu2);
      g = g >= 1 ? (1.35 - 0.15 * libmLog(acc)) / cutCoul : Math.sqrt(-libmLog(g)) / cutCoul;
      const gNew = this.newtonSolve(g, cutCoul, this.natoms, xprd * yprd * zprd, this.mu2);
      if (gNew > 0) g = gNew;
      else ctx.log('WARNING: Ewald/disp Newton solver failed, using old method to estimate g_ewald');
      this.gEwald = g;
    }
    const L = [xprd, yprd, zprd];
    const g = this.gEwald;
    // Ewald::setup: kmax per dimension, the smallest m with the rms error of equation 46 at or below the accuracy
    if (this.kmaxUser) {
      this.kmaxDim = [...this.kmaxUser];
    } else {
      this.kmaxDim = [0, 1, 2].map((d) => {
        let m = 1;
        while (rmsDipole(m, L[d], V, g, this.natoms, this.mu2) > acc && m < 1000) m++;
        return m;
      }) as [number, number, number];
    }
    this.kmaxAll = Math.max(this.kmaxDim[0], this.kmaxDim[1], this.kmaxDim[2]);
    this.buildVectors(L);
    ctx.log(`  G vector (1/distance) = ${g.toPrecision(8)}\n  KSpace vectors: actual = ${this.kvec.length / 3}\n` +
      `                  kxmax kymax kzmax  = ${this.kmaxDim.join(' ')}`);
  }

  /** Newton solution of equation 33 for g (ewald_dipole.cpp NewtonSolve with its numerical derivative); -1 on failure. */
  private newtonSolve(x0: number, rc: number, natoms: number, V: number, mu2: number): number {
    const f = (x: number) => dipoleRealForceError(x, rc, V, natoms, mu2) - this.accuracyAbs;
    const deriv = (x: number) => (f(x + 0.000001) - f(x)) / 0.000001;
    let x = x0;
    for (let it = 0; it < 10000; it++) {
      const dfx = deriv(x);
      if (dfx === 0 || dfx !== dfx) return -1;
      let dx = f(x) / dfx;
      while (x - dx <= 0) dx *= 0.5;
      x -= dx;
      if (Math.abs(dx) < 0.00001) return x;
      if (x !== x) return -1;
    }
    return -1;
  }

  /**
   * The k-vectors of Ewald::coeffs: n with |n_d| <= kmax_d for the planes and cells, the axis vectors up to kmax (the
   * largest of the three), in the half space, and |K|^2 <= gsqmx.
   */
  private buildVectors(L: number[]): void {
    const km = this.kmaxDim;
    const kmAll = this.kmaxAll;
    let gsqmx = 0;
    for (let d = 0; d < 3; d++) gsqmx = Math.max(gsqmx, ((2 * Math.PI * km[d]) / L[d]) ** 2);
    gsqmx *= 1.00001;
    const list: number[] = [];
    for (let nx = 0; nx <= kmAll; nx++) {
      for (let ny = -kmAll; ny <= kmAll; ny++) {
        for (let nz = -kmAll; nz <= kmAll; nz++) {
          if (nx === 0 && (ny < 0 || (ny === 0 && nz <= 0))) continue;
          const n = [nx, ny, nz];
          const nonzero = n.filter((v) => v !== 0).length;
          if (nonzero > 1 && n.some((v, d) => Math.abs(v) > km[d])) continue;
          const Kx = (2 * Math.PI * nx) / L[0], Ky = (2 * Math.PI * ny) / L[1], Kz = (2 * Math.PI * nz) / L[2];
          if (Kx * Kx + Ky * Ky + Kz * Kz > gsqmx) continue;
          list.push(Kx, Ky, Kz);
        }
      }
    }
    this.kvec = Float64Array.from(list);
  }

  compute(kc: KSpaceCompute): void {
    const s = kc.s, g = this.gEwald, geom = kc.geom;
    const sp = s.sp!;
    const V = geom.volume(3);
    // the vectors follow the box (the kmax of the setup stays fixed)
    this.buildVectors([geom.lx, geom.ly, geom.lz]);
    const n = s.n, x = s.x, f = kc.f, C = MUB2MU0;
    const mu = new Float64Array(3 * n);
    const spn = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      spn[i] = sp[4 * i + 3];
      for (let a = 0; a < 3; a++) mu[3 * i + a] = sp[4 * i + a] * spn[i];
    }
    const fl = fmLongOf(s, true);
    const eatom = kc.eatom, vatom = kc.vatom;
    const pref = C * 4 * Math.PI / V;
    const tk = new Float64Array(3 * n);
    const cosv = new Float64Array(n), sinv = new Float64Array(n);
    const nk = this.kvec.length / 3;
    const g2 = 4 * g * g;
    const W = [0, 0, 0, 0, 0, 0];
    let esum = 0;
    for (let v = 0; v < nk; v++) {
      const Kx = this.kvec[3 * v], Ky = this.kvec[3 * v + 1], Kz = this.kvec[3 * v + 2];
      const K2 = Kx * Kx + Ky * Ky + Kz * Kz;
      const Gk = libmExp(-K2 / g2) / K2;
      let Mr = 0, Mi = 0;
      const P = [0, 0, 0], Pi = [0, 0, 0];
      for (let i = 0; i < n; i++) {
        const ph = Kx * x[3 * i] + Ky * x[3 * i + 1] + Kz * x[3 * i + 2];
        const c = libmCos(ph), sn = libmSin(ph);
        cosv[i] = c; sinv[i] = sn;
        const mk = mu[3 * i] * Kx + mu[3 * i + 1] * Ky + mu[3 * i + 2] * Kz;
        Mr += mk * c; Mi += mk * sn;
        for (let a = 0; a < 3; a++) { P[a] += mu[3 * i + a] * c; Pi[a] += mu[3 * i + a] * sn; }
      }
      esum += Gk * (Mr * Mr + Mi * Mi);
      // the virial as native's uk*vg[k][j] - vc[k][j], with vc[k][j] = -sum_i partial_peratom_i * sp_ia * eg[k][b]:
      // sp_x with K_y for xy, sp_x with K_z for xz, sp_y with K_z for yz
      const M2 = Mr * Mr + Mi * Mi;
      const b = 2 * (1 / K2 + 1 / (4 * g * g));
      const Kv = [Kx, Ky, Kz];
      const pairs: [number, number, number][] = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 3], [0, 2, 4], [1, 2, 5]];
      for (const [a, c, idx] of pairs) {
        const dl = a === c ? 1 : 0;
        W[idx] += Gk * (M2 * (dl - b * Kv[a] * Kv[c]) + 2 * Kv[c] * (P[a] * Mr + Pi[a] * Mi));
      }
      // per-atom energy and virial (native: eatom[i] += mudotk*ug[k]*partial_peratom;
      // vatom[i][j] += ug[k]*mudotk*vg[k][j]*partial_peratom - vcik[j])
      if (eatom || vatom) {
        for (let i = 0; i < n; i++) {
          const mk = mu[3 * i] * Kx + mu[3 * i + 1] * Ky + mu[3 * i + 2] * Kz;
          const re = Mr * cosv[i] + Mi * sinv[i];
          if (eatom) eatom[i] += pref * Gk * mk * re;
          if (vatom) {
            for (const [a, c, idx] of pairs) {
              const dl = a === c ? 1 : 0;
              vatom[6 * i + idx] += pref * Gk * re * (mk * (dl - b * Kv[a] * Kv[c]) + 2 * Kv[c] * mu[3 * i + a]);
            }
          }
        }
      }
      const cf = C * (8 * Math.PI / V) * Gk;
      const tf = 2 * (4 * Math.PI / V) * Gk;
      for (let i = 0; i < n; i++) {
        const c = cosv[i], sn = sinv[i];
        const mk = mu[3 * i] * Kx + mu[3 * i + 1] * Ky + mu[3 * i + 2] * Kz;
        const im = Mr * sn - Mi * c;
        const re = Mr * c + Mi * sn;
        const t = cf * mk * im;
        f[3 * i] += t * Kx; f[3 * i + 1] += t * Ky; f[3 * i + 2] += t * Kz;
        const w = tf * re;
        tk[3 * i] += w * Kx; tk[3 * i + 1] += w * Ky; tk[3 * i + 2] += w * Kz;
      }
    }
    // precession field of each spin: fm_long = -(sp_norm mub2mu0 / hbar) tk (native: spscale2 = mub2mu0 / hbar)
    const spscale2 = MUB2MU0 / HBAR;
    for (let i = 0; i < n; i++) {
      const id = 3 * s.id[i];
      for (let a = 0; a < 3; a++) fl[id + a] = -spn[i] * spscale2 * tk[3 * i + a];
    }
    const self = -C * (TWO_OVER_SQRTPI * g * g * g / 3) * this.musq;
    if (eatom) {
      for (let i = 0; i < n; i++) {
        eatom[i] -= C * (TWO_OVER_SQRTPI * g * g * g / 3) * (mu[3 * i] ** 2 + mu[3 * i + 1] ** 2 + mu[3 * i + 2] ** 2);
      }
    }
    kc.acc.elong += pref * esum + self;
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += pref * W[c];
  }
}
