/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-SMOOTH/pair_sdpd_taitwater_isothermal.cpp
 * and pair_sdpd_taitwater_isothermal.h (PairSDPDTaitwaterIsothermal: pair_style sdpd/taitwater/isothermal T mu [seed],
 * pair_coeff I J rho0 soundspeed cutoff). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author
 * (from the source): Morteza Jalalvand (IASBS), with the references Espanol and Revenga, Phys Rev E 67, 026705 (2003).
 *
 * Model (docs.lammps.org/pair_sdpd_taitwater_isothermal.html): the Tait equation of state of an isothermal fluid with
 * the Lucy kernel; the conservative force is -m_i m_j (f_i + f_j) dW/dr, the viscous force is the Espanol-Revenga
 * (5/3) mu m_i m_j dW/dr (v_ij - ...) / (rho_i rho_j) and the thermal fluctuation is a random force from independent
 * increments of a Wiener process matrix (the diagonal and the upper triangle, each a standard normal deviate), scaled
 * by sqrt(-4 kB T f_visc / dt) / r. The density rate follows from the same pair.
 *
 * The random deviates follow native LAMMPS: one RanMars stream seeded with 1 + seed (nprocs + me, plus the optional
 * seed argument), six Gaussian draws per pair inside the cutoff, in the order of the pair list of native LAMMPS
 * (src/engine-gpl/neighbor/nativeOrder.ts, as the dpd styles do).
 *
 * Ghost atoms: the source's forces on ghosts are kept (newton on); the rate drho of a ghost goes to its owner, and
 * the density, the Tait pressure and the velocity estimate of a ghost are its owner's (forward communication).
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { PairSPHBase } from '../sph/pair_sph_base';
import { LUCY_DERIV_2D, LUCY_DERIV_3D, sphArrays, sphGhostVest } from '../sph/common';
import { RanMars } from '../random_mars';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { tallyXYZ } from '../dpd/pair_dpd';

/** pair_sdpd_taitwater_isothermal.cpp: "static const double sqrt_2_inv = std::sqrt(0.5);". */
const SQRT_2_INV = Math.sqrt(0.5);

export class PairSDPDTaitwaterIsothermal extends PairSPHBase {
  readonly name: string = 'sdpd/taitwater/isothermal';
  protected readonly attributesText = 'density and its rate';
  /** The constructor checks no atom attributes; init_style does (attributesMessage). */
  protected readonly attributeFlags = [] as const;
  override needsHalf = true;
  /** Global settings: temperature, dynamic viscosity, and the random stream of the pair style. */
  private temperature = 0;
  private viscosity = 0;
  private rng: RanMars | null = null;
  private boltz = 1;
  private readonly order = new NativePairOrder();
  /** Per type: reference density, sound speed and the Tait bulk factor B. */
  private rho0 = new Float64Array(0);
  private soundspeed = new Float64Array(0);
  private bulk = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.rho0 = new Float64Array(this.nt);
    this.soundspeed = new Float64Array(this.nt);
    this.bulk = new Float64Array(this.nt);
  }

  /** settings(): "T mu [seed]"; seed = nprocs + me + the optional argument, with nprocs = me = 0 on one processor. */
  settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 3) {
      throw new StyleError('Illegal number of arguments for pair_style sdpd/taitwater/isothermal');
    }
    this.temperature = parseNum(args[0], 'temperature');
    this.viscosity = parseNum(args[1], 'viscosity');
    if (this.temperature <= 0) throw new StyleError('Temperature must be positive');
    if (this.viscosity <= 0) throw new StyleError('Viscosity must be positive');
    let seed = 1;
    if (args.length === 3) seed += Math.trunc(parseNum(args[2], 'seed'));
    this.rng = new RanMars(seed);
  }

  /** pair_coeff I J rho0 soundspeed cutoff (B = c^2 rho0 / 7). */
  coeff(args: string[]): void {
    const r = this.coeffRange(args, 3, 'sdpd/taitwater/isothermal');
    const rho0One = parseNum(r.values[0], 'rho0');
    const soundspeedOne = parseNum(r.values[1], 'soundspeed');
    const cutOne = parseNum(r.values[2], 'cutoff');
    const bOne = soundspeedOne * soundspeedOne * rho0One / 7.0;
    if (rho0One <= 0) throw new StyleError('Density must be positive');
    if (soundspeedOne <= 0) throw new StyleError('Sound speed must be positive');
    if (cutOne <= 0) throw new StyleError('Cutoff must be positive');
    for (let i = r.ilo; i <= r.ihi; i++) {
      this.rho0[i] = rho0One;
      this.soundspeed[i] = soundspeedOne;
      this.bulk[i] = bOne;
    }
    this.forPairs(r.ilo, r.ihi, r.jlo, r.jhi, (i, j) => { this.h[i * this.nt + j] = cutOne; });
  }

  protected override attributesMessage(): string {
    return 'Pair style dpd/taitwater/isothermal requires atom attributes rho and drho';
  }

  override initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    this.boltz = ctx.s ? ctx.s.units.boltz : 1;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const rng = this.rng;
    if (!rng) throw new StyleError('pair sdpd/taitwater/isothermal used before pair_style');
    const a = sphArrays(s, 'Pair sdpd/taitwater/isothermal');
    const { rho, drho, vest: v } = a;
    // the ghosts' vest of a setup, when it is not their owners' (sph/common.ts sphGhostVest)
    const gv = sphGhostVest(s);
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const mass = s.massByType;
    const dim3 = s.dimension === 3;
    const nt = this.nt, h = this.h, cutsq = this.cutsq;
    const dtinv = 1.0 / s.dt;
    const kB = this.boltz;
    const T = this.temperature, mu = this.viscosity;
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    const w = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const j = pairs[p + 1] & NEIGHMASK;
      const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const it = type[i], jt = type[j];
      const t = it * nt + jt;
      if (!(rsq < cutsq[t])) continue;
      const oj = owner[j];
      const vj = gv && j >= pc.nlocal ? gv : v;
      const imass = mass[it], jmass = mass[jt];
      const hh = h[t];
      const ih = 1.0 / hh;
      const ihsq = ih * ih;
      const r = Math.sqrt(rsq);
      let wfd = hh - r;
      if (dim3) wfd = LUCY_DERIV_3D * wfd * wfd * ihsq * ihsq * ihsq * ih;
      else wfd = LUCY_DERIV_2D * wfd * wfd * ihsq * ihsq * ihsq;
      // Tait pressures of i and j (pair_sdpd_taitwater_isothermal.cpp compute)
      let tmp = rho[i] / this.rho0[it];
      let fi = tmp * tmp * tmp;
      fi = this.bulk[it] * (fi * fi * tmp - 1.0) / (rho[i] * rho[i]);
      tmp = rho[oj] / this.rho0[jt];
      let fj = tmp * tmp * tmp;
      fj = this.bulk[jt] * (fj * fj * tmp - 1.0) / (rho[oj] * rho[oj]);
      const velx = v[3 * i] - vj[3 * oj], vely = v[3 * i + 1] - vj[3 * oj + 1], velz = v[3 * i + 2] - vj[3 * oj + 2];
      const delVdotDelR = delx * velx + dely * vely + delz * velz;
      // Espanol viscosity (Espanol, 2003)
      const fvisc = (5.0 / 3.0) * mu * imass * jmass * wfd / (rho[i] * rho[oj]);
      const fpair = -imass * jmass * (fi + fj) * wfd;
      // independent increments of a Wiener process matrix (the source's draw order)
      w[0] = rng.gaussian();
      w[4] = rng.gaussian();
      w[8] = rng.gaussian();
      w[1] = w[3] = SQRT_2_INV * rng.gaussian();
      w[2] = w[6] = SQRT_2_INV * rng.gaussian();
      w[5] = w[7] = SQRT_2_INV * rng.gaussian();
      const prefactor = Math.sqrt(-4.0 * kB * T * fvisc * dtinv) / r;
      const fr0 = prefactor * (w[0] * delx + w[1] * dely + w[2] * delz);
      const fr1 = prefactor * (w[3] * delx + w[4] * dely + w[5] * delz);
      const fr2 = prefactor * (w[6] * delx + w[7] * dely + w[8] * delz);
      const gx = delx * fpair + (velx + delx * delVdotDelR / rsq) * fvisc + fr0;
      const gy = dely * fpair + (vely + dely * delVdotDelR / rsq) * fvisc + fr1;
      const gz = delz * fpair + (velz + delz * delVdotDelR / rsq) * fvisc + fr2;
      f[3 * i] += gx;
      f[3 * i + 1] += gy;
      f[3 * i + 2] += gz;
      drho[i] += jmass * delVdotDelR * wfd;
      f[3 * j] -= gx;
      f[3 * j + 1] -= gy;
      f[3 * j + 2] -= gz;
      drho[oj] += imass * delVdotDelR * wfd;
      if (pc.vatom) tallyXYZ(pc, i, j, 0, delx * fpair, dely * fpair, delz * fpair, delx, dely, delz);
    }
  }
}
