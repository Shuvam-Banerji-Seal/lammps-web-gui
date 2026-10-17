/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pppm_dipole.cpp and pppm_dipole.h (PPPMDipole, kspace_style pppm/dipole: init, set_grid_global,
 * set_grid_local, compute_df_kspace_dipole, compute_qopt_dipole, compute_gf_dipole, newton_raphson_f (the charge
 * channel of the rspace error), newton_raphson_f_dipole, find_gewald_dipole, derivf_dipole, compute_gf_denom,
 * make_rho_dipole, poisson_ik_dipole (the charge and dipole channels: the energy, the virial and the charge and dipole
 * fields and field gradients), fieldforce_ik_dipole (the charge force, the dipole force and the torque) and
 * musum_musq), with the stencil and G-ewald helpers of src/KSPACE/pppm.cpp (factorable, adjust_gewald, derivf,
 * set_grid_local, particle_map, compute_rho1d, compute_rho_coeff, compute_qopt (the charge k-space error, shared with
 * src/engine-gpl/pppm/grid.ts)) and MathSpecial::powsinxx of src/math_special.h, at the same commit. Contributing
 * authors of pppm_dipole.cpp: Stan Moore (SNL), Julien Tranchida (SNL). Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The spin style pppm/dipole/spin (src/engine-gpl/spin/pppm_dipole_spin.ts) is a subclass: it takes the moments
 * mu_i = sp_i |sp_i| with the mub2mu0 prefactor, and sends the field to the precession buffer instead of the torque.
 *
 * Charges: a system with point charges (atom attribute q, nonzero charges at the compute) adds the charge-charge and
 * charge-dipole channels of native poisson_ik_dipole and fieldforce_ik_dipole, the charge term of the rspace and
 * k-space error estimates (newton_raphson_f, compute_df_kspace_dipole), and the self and neutralization terms of
 * compute (qsum, qsqsum). A non-neutral system is accepted with a warning, as native (KSpace::qsum_qsq).
 *
 * Not ported (refused by name): per-atom energy and virial (compute pe/atom, stress/atom), the slab correction
 * (kspace_modify slab), kspace_modify diff ad (native: not with dipoles), kspace_modify force, kspace_modify minorder,
 * triclinic boxes and non-periodic boundaries.
 */
import { StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import { KSpaceBase, modifyNoEffect, ANGSTROM, QELECTRON } from '../../engine/force/kspace/ewald';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { parseNum } from '../../engine/force/util';
import { fft3d, isFFTSize } from '../../engine/force/fft';
import { computeQopt } from '../pppm/grid';
import { log as libmLog } from '../libm/log';
import { sin as libmSin } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

/** pppm.cpp OFFSET: keeps int() of a negative grid coordinate at the right value. */
const OFFSET = 16384;
/** pppm.cpp SMALL and LARGE (adjust_gewald). */
const SMALL = 0.00001;
const LARGE = 10000;
/** pppm_dipole.cpp EPS_HOC: the image sums of the influence function. */
const EPS_HOC = 1.0e-7;
/** pppm_dipole.h MAXORDER. */
const MAXORDER = 7;
const MY_PIS = Math.sqrt(Math.PI);
/** math_const.h MY_PI2 (pi/2): the neutralizing-background term of the energy. */
const MY_PI2 = Math.PI / 2;

const square = (x: number): number => x * x;

/** MathSpecial::powsinxx (math_special.h at b332f28): (sin x / x)^n for an integer n >= 0. */
const powsinxx = (x: number, n: number): number => {
  if (x === 0) return 1;
  let ww = libmSin(x) / x;
  let yy = 1;
  for (; n !== 0; n >>= 1, ww *= ww) if (n & 1) yy *= ww;
  return yy;
};

/** The periodic image of a mesh index (native: the ghost cells of the brick are summed onto the owned mesh). */
const wrap = (a: number, n: number): number => ((a % n) + n) % n;

/** The signed mesh index of native setup: per = i - N * (2 i / N) (integer division). */
const perOf = (i: number, n: number): number => i - n * Math.trunc((2 * i) / n);

/** PPPMDipole::gf_denom: (sx sy sz)^2 from the Gamma-expansion coefficients. */
const gfDenom = (b: Float64Array, order: number, x: number, y: number, z: number): number => {
  let sx = 0, sy = 0, sz = 0;
  for (let l = order - 1; l >= 0; l--) {
    sx = b[l] + sx * x;
    sy = b[l] + sy * y;
    sz = b[l] + sz * z;
  }
  const s = sx * sy * sz;
  return s * s;
};

/** PPPMDipole::compute_gf_denom: the coefficients of the Gamma(2n) expansion. */
const gfDenomCoefficients = (order: number): Float64Array => {
  const b = new Float64Array(order);
  for (let l = 1; l < order; l++) b[l] = 0;
  b[0] = 1;
  for (let m = 1; m < order; m++) {
    for (let l = m; l > 0; l--) b[l] = 4.0 * (b[l] * (l - m) * (l - m - 0.5) - b[l - 1] * (l - m - 1) * (l - m - 1));
    // after the loop l is 0 (native: gf_b[0] = 4.0 * (gf_b[0]*(l-m)*(l-m-0.5)))
    b[0] = 4.0 * (b[0] * (0 - m) * (0 - m - 0.5));
  }
  let ifact = 1;
  for (let k = 1; k < 2 * order; k++) ifact *= k;
  const gaminv = 1.0 / ifact;
  for (let l = 0; l < order; l++) b[l] *= gaminv;
  return b;
};

/**
 * Real-space error estimate of PPPMDipole::newton_raphson_f_dipole (and newton_raphson_f): the dipolar term, with the
 * charge-charge term added in quadrature when q2 (qqrd2e sum q^2) is nonzero.
 */
const rspaceError = (g: number, rc: number, natoms: number, vol: number, mu2: number, q2: number): number => {
  const a = rc * g;
  const rg2 = a * a;
  const rg4 = rg2 * rg2;
  const rg6 = rg4 * rg2;
  const Cc = 4.0 * rg4 + 6.0 * rg2 + 3.0;
  const Dc = 8.0 * rg6 + 20.0 * rg4 + 30.0 * rg2 + 15.0;
  const dd = (mu2 / Math.sqrt(vol * libmPow(g, 4) * libmPow(rc, 9) * natoms)) *
    Math.sqrt((13.0 / 6.0) * Cc * Cc + (2.0 / 15.0) * Dc * Dc - (13.0 / 15.0) * Cc * Dc) * libmExp(-rg2);
  if (q2 === 0) return dd;
  const dq = (2.0 * q2 * libmExp(-rg2)) / Math.sqrt(natoms * rc * vol);
  return Math.sqrt(dd * dd + dq * dq);
};

export class KSpacePPPMDipole extends KSpaceBase {
  readonly name: string;
  /** The class name native uses in its messages (PPPMDipole, PPPMDipoleSpin). */
  protected readonly label: string;
  /** kspace_modify order (stencil order of the charge assignment, native default 5). */
  private order = 5;
  /** kspace_modify mesh: user grid sizes, or null for the automatic choice. */
  private meshUser: [number, number, number] | null = null;
  /** Absolute accuracy in force units (accuracy_relative times two_charge_force, native init). */
  private acc = 0;
  private mu2 = 0;
  /** sum |mu_i|^2 at init (native musqsum: updated only when the atom count changes, so kept here). */
  private musqsum = 0;
  /** Charge sums at init (native qsum_qsq in init): qsum, qsqsum and q2 = qsqsum qqrd2e. */
  private qsumCharge = 0;
  private qsqsumCharge = 0;
  private q2 = 0;
  private cutoff = 0;
  private nxyz: [number, number, number] = [0, 0, 0];
  /** set_grid_local: the assignment shifts and the stencil bounds. */
  private shift = 0;
  private shiftone = 0;
  private nlower = 0;
  private nupper = 0;
  /** Gamma-expansion coefficients of the influence function (gf_b). */
  private gfB: Float64Array = new Float64Array(0);
  /** compute_rho_coeff, packed as rhoCoeff[l * order + (k - kmin)]. */
  private rhoCoeff: Float64Array = new Float64Array(0);
  private kmin = 0;
  /**
   * The influence functions (greensfn: dipole-dipole, greensfnQq: charge-charge, greensfnQm: charge-dipole) and the
   * virial factors (vg) of the current box, with the mesh wavevectors.
   */
  private greenKey = '';
  private greensfn: Float64Array = new Float64Array(0);
  private greensfnQq: Float64Array = new Float64Array(0);
  private greensfnQm: Float64Array = new Float64Array(0);
  private vg: Float64Array = new Float64Array(0);
  private fk: Float64Array[] = [new Float64Array(0), new Float64Array(0), new Float64Array(0)];

  constructor(name = 'pppm/dipole', label = 'PPPMDipole') {
    super();
    this.name = name;
    this.label = label;
  }

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError(`Illegal kspace_style ${this.name} command`);
    // PPPM::settings: accuracy_relative = fabs(numeric)
    this.accuracy = Math.abs(parseNum(args[0], 'accuracy'));
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'gewald': {
        this.gewaldUser = parseNum(values[0], 'gewald');
        if (this.gewaldUser < 0) throw new StyleError('kspace_modify gewald must be >= 0');
        return 1;
      }
      case 'mesh': {
        const m = [0, 1, 2].map((d) => parseNum(values[d], 'mesh'));
        if (m.some((v) => !Number.isInteger(v) || v < 0)) throw new StyleError('kspace_modify mesh values must be integers >= 0');
        if (m.every((v) => v === 0)) this.meshUser = null;
        else if (m.some((v) => v === 0)) throw new StyleError('kspace_modify mesh needs three values > 0, or 0 0 0 for the automatic mesh');
        else this.meshUser = m as [number, number, number];
        return 3;
      }
      case 'order': {
        const o = parseNum(values[0], 'order');
        if (!Number.isInteger(o) || o < 2 || o > MAXORDER) throw new StyleError(`${this.label} order cannot be < 2 or > ${MAXORDER}`);
        this.order = o;
        return 1;
      }
      case 'slab': case 'force': case 'diff': case 'minorder':
        throw new StyleError(`kspace_modify ${key} is not supported by kspace_style ${this.name}`);
      default:
        return modifyNoEffect(key, values);
    }
  }

  /** The moments of the atoms, 3 values per atom (native mu; the spin style overrides it with sp |sp|). */
  protected moments(s: SimState): Float64Array {
    const mu = s.mu!, out = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) out[3 * i + d] = mu[4 * i + d];
    return out;
  }

  /** Native: "Kspace style requires atom attribute mu" (the spin style requires sp). */
  protected checkMoments(s: SimState): void {
    if (!s.mu) throw new StyleError('Kspace style requires atom attribute mu');
  }

  /** Native: "Cannot (yet) use 'electron' units with dipoles" (the spin style requires metal units). */
  protected checkUnits(s: SimState): void {
    if (s.units.style === 'electron') throw new StyleError("Cannot (yet) use 'electron' units with dipoles");
  }

  /** The energy and force prefactor: qqrd2e for dipoles (native mu2 = musqsum qqrd2e), mub2mu0 for spins. */
  protected prefactor(qqrd2e: number): number {
    return qqrd2e;
  }

  /** Native musum_musq: "Using kspace solver PPPMDipole on system with no dipoles" (the spin style says spins). */
  protected noMomentError(): string {
    return `Using kspace solver ${this.label} on system with no dipoles`;
  }

  /** Called once per compute before the atoms are visited (the spin style resets its precession buffer here). */
  protected beginField(_kc: KSpaceCompute): void {}

  /**
   * The dipole field of atom i: E = field of the dipole and charge densities at the atom (native ux, uy, uz brick
   * interpolation, with the sign of fieldforce_ik_dipole). The dipole style adds the torque mu x E (native
   * t += qfactor (mu x E)).
   */
  protected applyField(kc: KSpaceCompute, i: number, C: number, E: Float64Array, mu: Float64Array): void {
    const t = kc.torque;
    if (!t) return;
    const mx = mu[3 * i], my = mu[3 * i + 1], mz = mu[3 * i + 2];
    const ex = E[0], ey = E[1], ez = E[2];
    t[3 * i] += C * (my * ez - mz * ey);
    t[3 * i + 1] += C * (mz * ex - mx * ez);
    t[3 * i + 2] += C * (mx * ey - my * ex);
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    this.checkSystem(s, geom);
    // PPPM::init: the absolute accuracy is accuracy_relative times two_charge_force (force qqr2e qelectron^2 / angstrom^2)
    const style = s.units.style;
    const twoChargeForce = s.units.qqr2e * QELECTRON[style] * QELECTRON[style] / (ANGSTROM[style] * ANGSTROM[style]);
    this.cutoff = cutCoul;
    this.acc = this.accuracy * twoChargeForce;
    // musum_musq: musqsum = sum |mu_i|^2, mu2 = musqsum * prefactor
    const mu = this.moments(s);
    let musq = 0;
    for (let i = 0; i < 3 * s.n; i++) musq += mu[i] * mu[i];
    this.musqsum = musq;
    this.mu2 = musq * this.prefactor(qqrd2e);
    // qsum_qsq(0) in init: the charge sums; q2 = qsqsum qqrd2e (native KSpace::qsum_qsq)
    let qs = 0, q2s = 0;
    if (s.q) for (let i = 0; i < s.n; i++) {
      qs += s.q[i];
      q2s += s.q[i] * s.q[i];
    }
    this.qsumCharge = qs;
    this.qsqsumCharge = q2s;
    this.q2 = q2s * qqrd2e;
    if (Math.abs(qs) > SMALL) ctx.log(`WARNING: System is not charge neutral, net charge = ${qs.toPrecision(8)}`);
    this.natoms = s.n;
    if (this.mu2 === 0) throw new StyleError(this.noMomentError());
    this.gfB = gfDenomCoefficients(this.order);
    const lx = geom.lx, ly = geom.ly, lz = geom.lz;
    this.setGridGlobal(lx, ly, lz, ctx);
    this.setGridLocal();
    // pppm.cpp init: if (!gewaldflag) adjust_gewald()
    if (!(this.gewaldUser > 0)) this.adjustGewald(lx, ly, lz);
    this.rhoCoeff = this.computeRhoCoeff();
    this.greenKey = '';
    this.buildGreen(lx, ly, lz);
    ctx.log(`  G vector (1/distance) = ${this.gEwald.toPrecision(8)}\n  grid = ${this.nxyz.join(' ')}\n  stencil order = ${this.order}`);
  }

  /** The system checks of PPPMDipole::init (the spin style overrides the attribute and unit checks). */
  private checkSystem(s: SimState, geom: Geometry): void {
    this.checkMoments(s);
    this.checkUnits(s);
    if (s.dimension === 2) throw new StyleError(`Cannot use ${this.label} with 2d simulation`);
    const p = s.box.periodic;
    if (!p[0] || !p[1] || !p[2]) throw new StyleError(`Cannot use nonperiodic boundaries with ${this.label}`);
    if (s.box.triclinic || geom.xy !== 0 || geom.xz !== 0 || geom.yz !== 0) {
      throw new StyleError(`Cannot yet use triclinic cells with ${this.label}`);
    }
    if (this.order < 2 || this.order > MAXORDER) throw new StyleError(`${this.label} order cannot be < 2 or > ${MAXORDER}`);
  }

  /**
   * Estimated k-space force error of PPPMDipole::compute_df_kspace_dipole for the given G and mesh: the dipolar term
   * (compute_qopt_dipole), with the charge-charge term (the base PPPM compute_qopt) added in quadrature when q2 != 0.
   */
  private dfKspace(g: number, lx: number, ly: number, lz: number): number {
    const qopt = this.qoptDipole(g, lx, ly, lz);
    const dd = Math.sqrt(qopt / this.natoms) * this.mu2 / (3.0 * lx * ly * lz);
    if (this.q2 === 0) return dd;
    const [nx, ny, nz] = this.nxyz;
    const qoptQ = computeQopt(nx, ny, nz, g, lx, ly, lz, this.order);
    const dq = Math.sqrt(qoptQ / this.natoms) * this.q2 / (lx * ly * lz);
    return Math.sqrt(dd * dd + dq * dq);
  }

  /** PPPMDipole::compute_qopt_dipole (ik differentiation, images |n| <= 2). */
  private qoptDipole(g: number, lx: number, ly: number, lz: number): number {
    const [nx, ny, nz] = this.nxyz;
    const unitkx = (2 * Math.PI) / lx, unitky = (2 * Math.PI) / ly, unitkz = (2 * Math.PI) / lz;
    const twoorder = 2 * this.order;
    let qopt = 0;
    for (let m = 0; m < nz; m++) {
      const mper = perOf(m, nz);
      const snz = square(libmSin(0.5 * unitkz * mper * lz / nz));
      for (let l = 0; l < ny; l++) {
        const lper = perOf(l, ny);
        const sny = square(libmSin(0.5 * unitky * lper * ly / ny));
        for (let k = 0; k < nx; k++) {
          const kper = perOf(k, nx);
          const snx = square(libmSin(0.5 * unitkx * kper * lx / nx));
          const sqk = square(unitkx * kper) + square(unitky * lper) + square(unitkz * mper);
          if (sqk === 0) continue;
          const denominator = gfDenom(this.gfB, this.order, snx, sny, snz);
          let sum1 = 0, sum2 = 0;
          for (let ix = -2; ix <= 2; ix++) {
            const qx = unitkx * (kper + nx * ix);
            const sx = libmExp(-0.25 * square(qx / g));
            const wx = powsinxx(0.5 * qx * lx / nx, twoorder);
            for (let iy = -2; iy <= 2; iy++) {
              const qy = unitky * (lper + ny * iy);
              const sy = libmExp(-0.25 * square(qy / g));
              const wy = powsinxx(0.5 * qy * ly / ny, twoorder);
              for (let iz = -2; iz <= 2; iz++) {
                const qz = unitkz * (mper + nz * iz);
                const sz = libmExp(-0.25 * square(qz / g));
                const wz = powsinxx(0.5 * qz * lz / nz, twoorder);
                const dot1 = unitkx * kper * qx + unitky * lper * qy + unitkz * mper * qz;
                const dot2 = qx * qx + qy * qy + qz * qz;
                const u1 = sx * sy * sz;
                const w2 = wx * wy * wz;
                const phi = (u1 * 4 * Math.PI) / dot2;
                const top = dot1 * dot1 * dot1 * w2 * phi;
                sum1 += phi * phi * dot2 * dot2 * dot2;
                sum2 += top * top / sqk / sqk / sqk;
              }
            }
          }
          qopt += sum1 - sum2 / denominator;
        }
      }
    }
    return qopt;
  }

  /** PPPMDipole::find_gewald_dipole: Newton-Raphson on the dipolar real-space error (accuracy target). */
  private findGewald(x0: number, rc: number, natoms: number, vol: number, mu2: number): number {
    const f = (x: number) => rspaceError(x, rc, natoms, vol, mu2, this.q2) - this.acc;
    const deriv = (x: number) => (f(x + 0.000001) - f(x)) / 0.000001;
    let x = x0;
    for (let i = 0; i < 10000; i++) {
      const dfx = deriv(x);
      if (dfx === 0 || dfx !== dfx) return -1;
      let dx = f(x) / dfx;
      while (x - dx <= 0.0) dx *= 0.5;
      x -= dx;
      if (Math.abs(dx) < 0.00001) return x;
      if (x !== x) return -1;
    }
    return -1;
  }

  /** PPPMDipole::set_grid_global: the G-ewald and the mesh of the accuracy target (or the user values). */
  private setGridGlobal(lx: number, ly: number, lz: number, ctx: StyleContext): void {
    const vol = lx * ly * lz;
    if (!(this.gewaldUser > 0)) {
      if (!(this.acc > 0)) throw new StyleError('KSpace accuracy must be > 0');
      let g = (1.35 - 0.15 * libmLog(this.acc)) / this.cutoff;
      const gNew = this.findGewald(g, this.cutoff, this.natoms, vol, this.mu2);
      if (gNew > 0) g = gNew;
      else ctx.log('WARNING: PPPMDipole dipole Newton solver failed, using old method to estimate g_ewald');
      this.gEwald = g;
    } else {
      this.gEwald = this.gewaldUser;
    }
    if (!this.meshUser) {
      let h = 4.0 / this.gEwald;
      let count = 0;
      for (;;) {
        let nx = Math.trunc(lx / h), ny = Math.trunc(ly / h), nz = Math.trunc(lz / h);
        if (nx <= 1) nx = 2;
        if (ny <= 1) ny = 2;
        if (nz <= 1) nz = 2;
        this.nxyz = [nx, ny, nz];
        const df = this.dfKspace(this.gEwald, lx, ly, lz);
        count++;
        if (df <= this.acc) break;
        if (count > 500) throw new StyleError('Could not compute grid size');
        h *= 0.95;
      }
    } else {
      this.nxyz = [this.meshUser[0], this.meshUser[1], this.meshUser[2]];
    }
    const n = this.nxyz;
    for (let d = 0; d < 3; d++) while (!isFFTSize(n[d])) n[d]++;
  }

  /** PPPM::set_grid_local: the charge-assignment shifts and stencil bounds of the order. */
  private setGridLocal(): void {
    const order = this.order;
    this.shift = order % 2 ? OFFSET + 0.5 : OFFSET;
    this.shiftone = order % 2 ? 0.0 : 0.5;
    this.nlower = -Math.trunc((order - 1) / 2);
    this.nupper = Math.trunc(order / 2);
    this.kmin = this.nlower;
  }

  /** PPPM::adjust_gewald with the dipolar newton_raphson_f (the forward-difference derivf of pppm.cpp). */
  private adjustGewald(lx: number, ly: number, lz: number): void {
    const vol = lx * ly * lz;
    const f = (g: number) => rspaceError(g, this.cutoff, this.natoms, vol, this.mu2, this.q2) - this.dfKspace(g, lx, ly, lz);
    const derivf = (g: number) => {
      const h = 0.000001;
      const f1 = f(g);
      const f2 = f(g + h);
      return (f2 - f1) / h;
    };
    let g = this.gEwald;
    for (let i = 0; i < LARGE; i++) {
      const dfx = derivf(g);
      if (dfx === 0.0 || dfx !== dfx) break;
      let dx = f(g) / dfx;
      while (g - dx <= 0.0) dx *= 0.5;
      g -= dx;
      if (Math.abs(f(g)) < SMALL) {
        this.gEwald = g;
        return;
      }
    }
    throw new StyleError('Could not compute g_ewald');
  }

  /** PPPM::compute_rho_coeff: the coefficients of the weight function of the given order. */
  private computeRhoCoeff(): Float64Array {
    const P = this.order;
    // a[l][k + P] for k in [-P, P]
    const a: Float64Array[] = [];
    for (let l = 0; l < P; l++) a.push(new Float64Array(2 * P + 1));
    a[0][P] = 1.0;
    for (let j = 1; j < P; j++) {
      for (let k = -j; k <= j; k += 2) {
        let s = 0.0;
        for (let l = 0; l < j; l++) {
          a[l + 1][k + P] = (a[l][k + 1 + P] - a[l][k - 1 + P]) / (l + 1);
          s += libmPow(0.5, l + 1) * (a[l][k - 1 + P] + libmPow(-1.0, l) * a[l][k + 1 + P]) / (l + 1);
        }
        a[0][k + P] = s;
      }
    }
    const rho = new Float64Array(P * P);
    let m = 0;
    for (let k = -(P - 1); k < P; k += 2) {
      for (let l = 0; l < P; l++) rho[l * P + m] = a[l][k + P];
      m++;
    }
    return rho;
  }

  /** PPPM::compute_rho1d for the three directions: out[d * P + (k - kmin)] for k in [nlower, nupper]. */
  private rho1dAll(dx: number, dy: number, dz: number, out: Float64Array): void {
    const P = this.order, rc = this.rhoCoeff;
    const dd = [dx, dy, dz];
    for (let d = 0; d < 3; d++) {
      for (let k = this.nlower; k <= this.nupper; k++) {
        let r = 0;
        for (let l = P - 1; l >= 0; l--) r = rc[l * P + (k - this.kmin)] + r * dd[d];
        out[d * P + (k - this.kmin)] = r;
      }
    }
  }

  /**
   * PPPMDipole::compute_gf_dipole (the influence functions of the aliasing sums, with the mesh wavevectors of setup):
   * greensfnQq (charge-charge, numerator (k.k_n)^1 / |k|^2), greensfnQm (charge-dipole, (k.k_n)^2 / |k|^4) and
   * greensfn (dipole-dipole, (k.k_n)^3 / |k|^6); and the virial factors vg of PPPMDipole::setup, for the current box.
   */
  private buildGreen(lx: number, ly: number, lz: number): void {
    const [nx, ny, nz] = this.nxyz;
    const N = [nx, ny, nz];
    const L = [lx, ly, lz];
    const unit = [0, 1, 2].map((d) => (2 * Math.PI) / L[d]);
    const g = this.gEwald;
    const P = this.order;
    const twoorder = 2 * P;
    const fk: Float64Array[] = [new Float64Array(nx), new Float64Array(ny), new Float64Array(nz)];
    for (let d = 0; d < 3; d++) for (let i = 0; i < N[d]; i++) fk[d][i] = unit[d] * perOf(i, N[d]);
    this.fk = fk;
    // native: nbx = max(int(g Lx / (pi Nx) (-ln EPS_HOC)^(1/4)), 2)
    const nb = [0, 1, 2].map((d) => {
      const c = Math.trunc((g * L[d] / (Math.PI * N[d])) * libmPow(-libmLog(EPS_HOC), 0.25));
      return Math.max(c, 2);
    });
    const Ng = nx * ny * nz;
    const G = new Float64Array(Ng);
    const GQQ = new Float64Array(Ng);
    const GQM = new Float64Array(Ng);
    const VG = new Float64Array(6 * Ng);
    for (let m = 0; m < nz; m++) {
      const mper = perOf(m, nz);
      for (let l = 0; l < ny; l++) {
        const lper = perOf(l, ny);
        for (let k = 0; k < nx; k++) {
          const kper = perOf(k, nx);
          const snx = square(libmSin(0.5 * unit[0] * kper * lx / nx));
          const sny = square(libmSin(0.5 * unit[1] * lper * ly / ny));
          const snz = square(libmSin(0.5 * unit[2] * mper * lz / nz));
          const idx = (m * ny + l) * nx + k;
          const kx = unit[0] * kper, ky = unit[1] * lper, kz = unit[2] * mper;
          const sqk = square(unit[0] * kper) + square(unit[1] * lper) + square(unit[2] * mper);
          if (sqk === 0) continue;
          const denominator = gfDenom(this.gfB, P, snx, sny, snz);
          let sum1 = 0, sum1Qq = 0, sum1Qm = 0;
          for (let ix = -nb[0]; ix <= nb[0]; ix++) {
            const qx = unit[0] * (kper + nx * ix);
            const sx = libmExp(-0.25 * square(qx / g));
            const wx = powsinxx(0.5 * qx * lx / nx, twoorder);
            for (let iy = -nb[1]; iy <= nb[1]; iy++) {
              const qy = unit[1] * (lper + ny * iy);
              const sy = libmExp(-0.25 * square(qy / g));
              const wy = powsinxx(0.5 * qy * ly / ny, twoorder);
              for (let iz = -nb[2]; iz <= nb[2]; iz++) {
                const qz = unit[2] * (mper + nz * iz);
                const sz = libmExp(-0.25 * square(qz / g));
                const wz = powsinxx(0.5 * qz * lz / nz, twoorder);
                const dot1 = unit[0] * kper * qx + unit[1] * lper * qy + unit[2] * mper * qz;
                const dot2 = qx * qx + qy * qy + qz * qz;
                const u1 = sx * sy * sz;
                const w2 = wx * wy * wz;
                const phi = u1 * 4 * Math.PI / dot2;
                const base = w2 * phi;
                sum1Qq += dot1 * base / sqk;
                sum1Qm += dot1 * dot1 * base / sqk / sqk;
                sum1 += dot1 * dot1 * dot1 * base / sqk / sqk / sqk;
              }
            }
          }
          G[idx] = sum1 / denominator;
          GQQ[idx] = sum1Qq / denominator;
          GQM[idx] = sum1Qm / denominator;
          // PPPMDipole::setup: vg from the wavevector K = (kx, ky, kz)
          const vterm = -2.0 * (1.0 / sqk + 0.25 / (g * g));
          VG[6 * idx] = 1.0 + vterm * kx * kx;
          VG[6 * idx + 1] = 1.0 + vterm * ky * ky;
          VG[6 * idx + 2] = 1.0 + vterm * kz * kz;
          VG[6 * idx + 3] = vterm * kx * ky;
          VG[6 * idx + 4] = vterm * kx * kz;
          VG[6 * idx + 5] = vterm * ky * kz;
        }
      }
    }
    this.greensfn = G;
    this.greensfnQq = GQQ;
    this.greensfnQm = GQM;
    this.vg = VG;
    this.greenKey = [lx, ly, lz, g, nx, ny, nz, P].join(',');
  }

  /** The Green's function of the box of this call (setup of the source runs on every run). */
  private ensureGreen(lx: number, ly: number, lz: number): void {
    const key = [lx, ly, lz, this.gEwald, ...this.nxyz, this.order].join(',');
    if (key !== this.greenKey) this.buildGreen(lx, ly, lz);
  }

  compute(kc: KSpaceCompute): void {
    if (kc.eatom || kc.vatom) {
      throw new StyleError(`compute pe/atom and stress/atom with kspace_style ${this.name} are not supported by the browser engine`);
    }
    const s = kc.s, geom = kc.geom, n = s.n;
    // native compute returns when there are no dipoles; musqsum is the init-time value (as in native)
    if (this.musqsum === 0) return;
    const lx = geom.lx, ly = geom.ly, lz = geom.lz;
    this.ensureGreen(lx, ly, lz);
    const [nx, ny, nz] = this.nxyz;
    const Ng = nx * ny * nz;
    const V = lx * ly * lz;
    const P = this.order;
    const C = this.prefactor(kc.qqrd2e);
    const mu = this.moments(s);
    // the charge channel runs when some atom carries a charge now (a zero charge adds nothing, as in native)
    const q = s.q;
    let hasQ = false;
    if (q) for (let i = 0; i < n; i++) if (q[i] !== 0) { hasQ = true; break; }
    const x = s.x, f = kc.f;
    const lo = s.box.lo;
    const delinv = [nx / lx, ny / ly, nz / lz];
    const delvolinv = delinv[0] * delinv[1] * delinv[2];
    const shift = this.shift, shiftone = this.shiftone;
    const part = new Int32Array(3 * n);
    const r = new Float64Array(3 * P);
    const ds = [0, 0, 0];
    // make_rho_dipole: the three moment densities (and the charge density) on the mesh (real parts; the imaginary
    // parts are filled by the FFT)
    const nDens = hasQ ? 4 : 3;
    const dens: Float64Array[] = [];
    for (let c = 0; c < nDens; c++) dens.push(new Float64Array(2 * Ng));
    const mom = [0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        const u = (x[3 * i + d] - lo[d]) * delinv[d];
        const p = Math.trunc(u + shift) - OFFSET;
        part[3 * i + d] = p;
        ds[d] = p + shiftone - (x[3 * i + d] - lo[d]) * delinv[d];
      }
      this.rho1dAll(ds[0], ds[1], ds[2], r);
      mom[0] = mu[3 * i]; mom[1] = mu[3 * i + 1]; mom[2] = mu[3 * i + 2];
      mom[3] = hasQ ? q![i] : 0;
      const px = part[3 * i], py = part[3 * i + 1], pz = part[3 * i + 2];
      for (let c = 0; c < nDens; c++) {
        const zc = delvolinv * mom[c];
        const dc = dens[c];
        for (let nn = this.nlower; nn <= this.nupper; nn++) {
          const mz = wrap(nn + pz, nz);
          const yc = zc * r[2 * P + (nn - this.kmin)];
          for (let mm = this.nlower; mm <= this.nupper; mm++) {
            const my = wrap(mm + py, ny);
            const xc = yc * r[1 * P + (mm - this.kmin)];
            for (let ll = this.nlower; ll <= this.nupper; ll++) {
              const mx = wrap(ll + px, nx);
              dc[2 * ((mz * ny + my) * nx + mx)] += xc * r[ll - this.kmin];
            }
          }
        }
      }
    }
    // poisson_ik_dipole: forward transforms of the densities, and D = k . P(k) (the dipole structure factor)
    for (let c = 0; c < nDens; c++) fft3d(dens[c], nx, ny, nz, -1);
    const Dre = new Float64Array(Ng), Dim = new Float64Array(Ng);
    const fk = this.fk;
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const idx = (iz * ny + iy) * nx + ix;
          const kx = fk[0][ix], ky = fk[1][iy], kz = fk[2][iz];
          const Mx = 2 * idx, Mxi = Mx + 1;
          Dre[idx] = kx * dens[0][Mx] + ky * dens[1][Mx] + kz * dens[2][Mx];
          Dim[idx] = kx * dens[0][Mxi] + ky * dens[1][Mxi] + kz * dens[2][Mxi];
        }
      }
    }
    const scaleinv = 1.0 / Ng;
    const s2 = scaleinv * scaleinv;
    const G = this.greensfn, GQQ = this.greensfnQq, GQM = this.greensfnQm, VG = this.vg;
    // spectra of the fields (native fills work4 and inverse-transforms it; the brick values are the real parts):
    //   spec[0..2]  dipole E:  k_a (gdd D + i gqm Q)
    //   spec[3..8]  dipole field gradient: k_a k_b (i gdd D - gqm Q)
    //   spec[9..11] charge E:  k_a (i gqq Q + gqm D)           (charge channel only)
    const nSpec = hasQ ? 12 : 9;
    const spec: Float64Array[] = [];
    for (let c = 0; c < nSpec; c++) spec.push(new Float64Array(2 * Ng));
    let esum = 0;
    const W = [0, 0, 0, 0, 0, 0];
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const idx = (iz * ny + iy) * nx + ix;
          const g0 = G[idx];
          const gqq = GQQ[idx], gqm = GQM[idx];
          const kx = fk[0][ix], ky = fk[1][iy], kz = fk[2][iz];
          const Dr = Dre[idx], Di = Dim[idx];
          const Qr = hasQ ? dens[3][2 * idx] : 0, Qi = hasQ ? dens[3][2 * idx + 1] : 0;
          const M0r = dens[0][2 * idx], M0i = dens[0][2 * idx + 1];
          const M1r = dens[1][2 * idx], M1i = dens[1][2 * idx + 1];
          const M2r = dens[2][2 * idx], M2i = dens[2][2 * idx + 1];
          let eng = s2 * g0 * (Dr * Dr + Di * Di);
          if (hasQ) eng += s2 * gqq * (Qr * Qr + Qi * Qi) + 2.0 * s2 * gqm * (Qr * Di - Qi * Dr);
          esum += eng;
          // virial: eng * vg plus the structure-factor strain terms of the moment densities
          for (let jj = 0; jj < 6; jj++) W[jj] += eng * VG[6 * idx + jj];
          const MxD = M0r * Dr + M0i * Di;
          const MyD = M1r * Dr + M1i * Di;
          const MzD = M2r * Dr + M2i * Di;
          W[0] += 2.0 * s2 * g0 * kx * MxD;
          W[1] += 2.0 * s2 * g0 * ky * MyD;
          W[2] += 2.0 * s2 * g0 * kz * MzD;
          W[3] += 2.0 * s2 * g0 * ky * MxD;
          W[4] += 2.0 * s2 * g0 * kz * MxD;
          W[5] += 2.0 * s2 * g0 * kz * MyD;
          if (hasQ) {
            // charge-dipole strain terms: 2 s2 gqm k_b Im(Q* P_a)
            const IxQ = Qr * M0i - Qi * M0r;
            const IyQ = Qr * M1i - Qi * M1r;
            const IzQ = Qr * M2i - Qi * M2r;
            W[0] += 2.0 * s2 * gqm * kx * IxQ;
            W[1] += 2.0 * s2 * gqm * ky * IyQ;
            W[2] += 2.0 * s2 * gqm * kz * IzQ;
            W[3] += 2.0 * s2 * gqm * ky * IxQ;
            W[4] += 2.0 * s2 * gqm * kz * IxQ;
            W[5] += 2.0 * s2 * gqm * kz * IyQ;
          }
          const sgdd = scaleinv * g0, sgqq = scaleinv * gqq, sgqm = scaleinv * gqm;
          const ka = [kx, ky, kz];
          for (let a = 0; a < 3; a++) {
            spec[a][2 * idx] = ka[a] * (sgdd * Dr - sgqm * Qi);
            spec[a][2 * idx + 1] = ka[a] * (sgdd * Di + sgqm * Qr);
          }
          // gradient pairs (slot, k_a, k_b): the real part is -k_a k_b (gdd Di + gqm Qr), the imaginary part
          // k_a k_b (gdd Dr - gqm Qi)
          const pairs: [number, number, number][] = [[3, kx, kx], [4, ky, ky], [5, kz, kz], [6, kx, ky], [7, kx, kz], [8, ky, kz]];
          for (const [slot, ca, cb] of pairs) {
            const kk = ca * cb;
            spec[slot][2 * idx] = -kk * (sgdd * Di + sgqm * Qr);
            spec[slot][2 * idx + 1] = kk * (sgdd * Dr - sgqm * Qi);
          }
          if (hasQ) {
            // charge E: k_a (i gqq Q + gqm D)
            for (let a = 0; a < 3; a++) {
              spec[9 + a][2 * idx] = ka[a] * (-sgqq * Qi + sgqm * Dr);
              spec[9 + a][2 * idx + 1] = ka[a] * (sgqq * Qr + sgqm * Di);
            }
          }
        }
      }
    }
    // backward transforms (unnormalized); the real parts are the brick values of native
    for (let c = 0; c < nSpec; c++) fft3d(spec[c], nx, ny, nz, 1);
    this.beginField(kc);
    const E = new Float64Array(3);
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        const u = (x[3 * i + d] - lo[d]) * delinv[d];
        const p = Math.trunc(u + shift) - OFFSET;
        part[3 * i + d] = p;
        ds[d] = p + shiftone - (x[3 * i + d] - lo[d]) * delinv[d];
      }
      this.rho1dAll(ds[0], ds[1], ds[2], r);
      const px = part[3 * i], py = part[3 * i + 1], pz = part[3 * i + 2];
      let ex = 0, ey = 0, ez = 0;
      let eqx = 0, eqy = 0, eqz = 0;
      const vv = [0, 0, 0, 0, 0, 0];
      for (let nn = this.nlower; nn <= this.nupper; nn++) {
        const mz = wrap(nn + pz, nz);
        const z0 = r[2 * P + (nn - this.kmin)];
        for (let mm = this.nlower; mm <= this.nupper; mm++) {
          const my = wrap(mm + py, ny);
          const y0 = z0 * r[1 * P + (mm - this.kmin)];
          for (let ll = this.nlower; ll <= this.nupper; ll++) {
            const mx = wrap(ll + px, nx);
            const x0 = y0 * r[ll - this.kmin];
            const idx2 = 2 * ((mz * ny + my) * nx + mx);
            ex -= x0 * spec[0][idx2];
            ey -= x0 * spec[1][idx2];
            ez -= x0 * spec[2][idx2];
            for (let c = 0; c < 6; c++) vv[c] -= x0 * spec[3 + c][idx2];
            if (hasQ) {
              eqx -= x0 * spec[9][idx2];
              eqy -= x0 * spec[10][idx2];
              eqz -= x0 * spec[11][idx2];
            }
          }
        }
      }
      // fieldforce_ik_dipole: the force on the charge from the total field, the force from the field gradient on the
      // dipole, and the field itself (torque)
      if (hasQ) {
        const qi = q![i];
        f[3 * i] += C * qi * eqx;
        f[3 * i + 1] += C * qi * eqy;
        f[3 * i + 2] += C * qi * eqz;
      }
      const mux = mu[3 * i], muy = mu[3 * i + 1], muz = mu[3 * i + 2];
      const vxx = vv[0], vyy = vv[1], vzz = vv[2], vxy = vv[3], vxz = vv[4], vyz = vv[5];
      f[3 * i] += C * (vxx * mux + vxy * muy + vxz * muz);
      f[3 * i + 1] += C * (vxy * mux + vyy * muy + vyz * muz);
      f[3 * i + 2] += C * (vxz * mux + vyz * muy + vzz * muz);
      E[0] = ex; E[1] = ey; E[2] = ez;
      this.applyField(kc, i, C, E, mu);
    }
    // energy: 0.5 V sum, minus the self term and the neutralizing-background term, times the prefactor (native
    // compute); virial: 0.5 prefactor V sum
    const g3 = this.gEwald * this.gEwald * this.gEwald;
    const g = this.gEwald;
    let energy = 0.5 * V * esum - this.musqsum * 2.0 * g3 / 3.0 / MY_PIS;
    energy -= g * this.qsqsumCharge / MY_PIS + MY_PI2 * this.qsumCharge * this.qsumCharge / (g * g * V);
    kc.acc.elong += energy * C;
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += 0.5 * C * V * W[c];
  }
}
