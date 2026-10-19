/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/pair_smd_ulsph.cpp and pair_smd_ulsph.h (PairULSPH: pair_style smd/ulsph; PreCompute, compute,
 * AssembleStressTensor, ComputeHourglassForce, coeff, settings) and the Tait and linear equations of state and the
 * spiky kernel of src/MACHDYN/smd_material_models.cpp and smd_kernels.h at commit b332f28. MACHDYN author (from the
 * sources): Georg C. Ganzenmueller, Fraunhofer EMI. LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Updated Lagrangian SPH for fluids (docs.lammps.org/pair_smd.html, the ulsph keywords): the pressure comes from an
 * equation of state of the current density, the stress is -p I, the pair force is the SPH stress force plus a linear
 * artificial viscosity (*COMMON q1) and an optional hourglass control, the density is integrated in fix
 * smd/integrate_ulsph through the velocity gradient, and the stable time increment is the sound-speed limit.
 * Supported here: the settings *DENSITY_CONTINUITY *VELOCITY_GRADIENT *NO_GRADIENT_CORRECTION; the coefficients *COMMON,
 * *EOS_TAIT and *EOS_LINEAR. Other keywords (density summation, gradient correction, strength and viscosity models,
 * artificial pressure and stress, the perfect gas) are refused with a StyleError naming them.
 *
 * The pair loops are the native ones: the newton-off neighbor list is walked once, the owned atoms of each pair are
 * updated, and the ghost partner's values are read from its owner (the native forward communication).
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { SMD_DEFGRAD, SMD_DESPH, SMD_VEST } from '../../engine/smd';
import { spikyKernel } from './pair_smd_tlsph';
import { pow as libmPow } from '../libm/pow';

/** BIG: initial value of the stable time increment of each compute (native: dtCFL = 1.0e22). */
const BIG_DT = 1.0e22;

/** EOS of one type (pair_smd.html: *EOS_TAIT exponent, *EOS_LINEAR). */
type Eos = { kind: 'tait'; exponent: number } | { kind: 'linear' };

/** Per type material data (pair_smd_ulsph.cpp coeff: *COMMON rho0 c0 q1 cp hg). */
interface TypeParams {
  rho0: number;
  c0: number;
  q1: number;
  hg: number;
  bulk: number;
  eos: Eos;
}

/**
 * TaitEOS_density (smd_material_models.cpp): B = rho0 c0^2 / exponent, p = B ((rho/rho0)^exponent - 1), and the sound
 * speed from the bulk modulus rho dp/drho = B exponent (rho/rho0)^exponent.
 */
const taitEOS = (exponent: number, c0Ref: number, rhoRef: number, rho: number): { p: number; c: number } => {
  const B = rhoRef * c0Ref * c0Ref / exponent;
  const tmp = libmPow(rho / rhoRef, exponent);
  const p = B * (tmp - 1.0);
  const bulkModulus = B * tmp * exponent;
  return { p, c: Math.sqrt(bulkModulus / rho) };
};

export class PairSMDUlsph extends Pair {
  readonly name = 'smd/ulsph';
  /** The explicit virial tally (ev_tally_xyz with newton off), not the fdotr sum. */
  virialFdotr = false;
  explicitVirial = true;
  private params: (TypeParams | null)[] = [];
  private active = new Uint8Array(0);
  private maxrad = new Float64Array(0);
  /** native's size list (NeighConst::REQ_SIZE), in the enumeration of the newton pair setting (set at init) */
  private order = new NativePairOrder();
  /** Critical time step of the last force evaluation (fix smd/adjust_dt reads it). */
  dtCFL = BIG_DT;
  /** Per owned atom: velocity gradient L (3x3 row-major, 9 per atom), shape of the deformation F, Cauchy stress, sound speed. */
  L = new Float64Array(0);
  F = new Float64Array(0);
  stress = new Float64Array(0);
  c0 = new Float64Array(0);
  /** Neighbors within the kernel radius (native numNeighs) and the effective shear modulus (0 for fluids, native effm). */
  numNeighs = new Int32Array(0);
  effm = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Illegal number of arguments for pair_style smd/ulsph (expected *DENSITY_CONTINUITY *VELOCITY_GRADIENT *NO_GRADIENT_CORRECTION)');
    if (args[0] === '*DENSITY_SUMMATION') throw new StyleError('pair smd/ulsph keyword *DENSITY_SUMMATION is not supported by the browser engine yet (use *DENSITY_CONTINUITY)');
    if (args[0] !== '*DENSITY_CONTINUITY') throw new StyleError(`Illegal settings keyword for first keyword of pair style smd/ulsph: ${args[0]}`);
    if (args[1] === '*NO_VELOCITY_GRADIENT') throw new StyleError('pair smd/ulsph keyword *NO_VELOCITY_GRADIENT is not supported by the browser engine yet (use *VELOCITY_GRADIENT)');
    if (args[1] !== '*VELOCITY_GRADIENT') throw new StyleError(`Illegal settings keyword for second keyword of pair style smd/ulsph: ${args[1]}`);
    if (args[2] === '*GRADIENT_CORRECTION') throw new StyleError('pair smd/ulsph keyword *GRADIENT_CORRECTION is not supported by the browser engine yet (use *NO_GRADIENT_CORRECTION)');
    if (args[2] !== '*NO_GRADIENT_CORRECTION') throw new StyleError(`Illegal settings keyword for third keyword of pair style smd/ulsph: ${args[2]}`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.params = Array.from({ length: nt }, () => null);
    this.active = new Uint8Array(nt * nt);
    this.maxrad = new Float64Array(nt);
  }

  /**
   * pair_coeff I I *COMMON rho0 c0 q1 cp hg *EOS_TAIT exponent | *EOS_LINEAR ... *END. Only same-type coefficients; the
   * cross keyword *CROSS and the other material keywords are refused.
   */
  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 3) throw new StyleError('number of arguments for pair smd/ulsph is too small');
    if (!this.ntypes) throw new StyleError('pair_coeff before the box exists');
    const i = parseInt(args[0], 10), j = parseInt(args[1], 10);
    if (!(i >= 1 && i <= this.ntypes) || i !== j) throw new StyleError('pair smd/ulsph coefficients can only be given for a type with itself (cross type pairs are not supported by the browser engine yet)');
    if (args[2] !== '*COMMON') throw new StyleError('common keyword missing!');
    const common = args.slice(3, 8);
    if (common.length < 5 || args[8] === undefined || !args[8].startsWith('*')) throw new StyleError('expected 5 arguments following *COMMON');
    const [rho0, c0, q1, , hg] = common.map((w, k) => parseNum(w, ['rho0', 'c0', 'q1', 'cp', 'hg'][k]));
    const p: TypeParams = { rho0, c0, q1, hg, bulk: c0 * c0 * rho0, eos: { kind: 'linear' } };
    let eosSet = false;
    let k = 8;
    for (;;) {
      if (k >= args.length) throw new StyleError('no *END terminates the pair smd/ulsph coefficients');
      const kw = args[k];
      if (kw === '*END') break;
      if (!kw.startsWith('*')) throw new StyleError(`unknown pair smd/ulsph keyword ${kw}`);
      let e = k + 1;
      while (e < args.length && !args[e].startsWith('*')) e++;
      if (e >= args.length) throw new StyleError(`no *KEYWORD terminates ${kw}`);
      const vals = args.slice(k + 1, e);
      const need = (n: number) => {
        if (vals.length !== n) throw new StyleError(`expected ${n} arguments following ${kw} but got ${vals.length}`);
        return vals.map((w, q) => parseNum(w, `${kw} value ${q + 1}`));
      };
      switch (kw) {
        case '*EOS_TAIT': p.eos = { kind: 'tait', exponent: need(1)[0] }; eosSet = true; break;
        case '*EOS_LINEAR': need(0); p.eos = { kind: 'linear' }; eosSet = true; break;
        case '*EOS_PERFECT_GAS':
        case '*STRENGTH_LINEAR':
        case '*STRENGTH_LINEAR_PLASTIC':
        case '*VISCOSITY_NEWTON':
        case '*ARTIFICIAL_PRESSURE':
        case '*ARTIFICIAL_STRESS':
          throw new StyleError(`pair smd/ulsph material keyword ${kw} is not supported by the browser engine yet (supported: *EOS_TAIT, *EOS_LINEAR)`);
        default:
          throw new StyleError(`unknown *KEYWORD: ${kw}`);
      }
      k = e;
    }
    if (!eosSet) throw new StyleError(`must specify an EOS for particle type ${i}`);
    this.params[i] = p;
    this.active[i * (this.ntypes + 1) + i] = 1;
    this.version++;
  }

  /** Cutoff: the sum of the largest kernel radii of the two types (native init_one, no frozen particles). */
  initOne(i: number, j: number): number {
    if (i !== j || !this.active[i * (this.ntypes + 1) + i]) throw new StyleError('All pair coeffs are not set');
    return this.maxrad[i] + this.maxrad[j];
  }

  initStyle(ctx: StyleContext): void {
    const nt = this.ntypes + 1;
    this.order = new NativePairOrder(ctx.newtonPair !== false, { allow: (a, b) => !!this.active[a * nt + b] });
    const s = ctx.s;
    if (!s) throw new StyleError('pair smd/ulsph needs a box and atoms');
    if (!s.radius || !s.rmass || !s.vfrac || !s.custom.has(SMD_DEFGRAD)) throw new StyleError('pair smd/ulsph needs atom_style smd');
    const onerad = new Float64Array(this.ntypes + 1);
    for (let i = 0; i < s.n; i++) onerad[s.type[i]] = Math.max(onerad[s.type[i]], s.radius[i]);
    this.maxrad = onerad;
  }

  /** Per-atom arrays of the owned atoms (grown on demand). */
  private ensure(n: number): void {
    if (this.numNeighs.length === n) return;
    this.L = new Float64Array(9 * n);
    this.F = new Float64Array(9 * n);
    this.stress = new Float64Array(9 * n);
    this.c0 = new Float64Array(n);
    this.numNeighs = new Int32Array(n);
    this.effm = new Float64Array(n);
  }

  private material(t: number): TypeParams {
    const p = this.params[t];
    if (!p || !this.active[t * (this.ntypes + 1) + t]) throw new StyleError(`pair smd/ulsph: type ${t} has no coefficients`);
    return p;
  }

  /**
   * One evaluation (native compute): the velocity gradient and the deformation increment (PreCompute), the stress of each
   * owned atom (AssembleStressTensor), then the pair forces, energy rates and neighbor counts.
   */
  compute(pc: PairCompute): void {
    const s = pc.s;
    const n = s.n;
    const nlocal = pc.nlocal;
    this.ensure(n);
    const x = pc.x, f = pc.f, type = pc.type;
    const owner = pc.nb.owner;
    const vest = s.custom.get(SMD_VEST)!.data;
    const desph = s.custom.get(SMD_DESPH)!.data;
    const acc9 = s.custom.get(SMD_DEFGRAD)!.data;
    const vfrac = s.vfrac!, rmass = s.rmass!, radius = s.radius!;
    const dim = s.dimension;
    const dt = s.dt;
    const active = (t: number) => !!this.active[t * (this.ntypes + 1) + t];
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    const L = this.L, F = this.F, S9 = this.stress, c0a = this.c0, nn = this.numNeighs;
    const dv = [0, 0, 0], g = [0, 0, 0], dx = [0, 0, 0];

    // zero the accumulators of the owned atoms (native: shepardWeight, smoothVel, numNeighs; then PreCompute's L and F)
    nn.fill(0, 0, nlocal);
    for (let i = 0; i < nlocal; i++) {
      if (!active(type[i])) continue;
      L.fill(0, 9 * i, 9 * i + 9);
      F.fill(0, 9 * i, 9 * i + 9);
      // native: the accumulated strain is cleared at the first step (update->ntimestep == 0)
      if (s.step === 0) acc9.fill(0, 9 * i, 9 * i + 9);
    }

    // PreCompute: velocity gradient L and deformation rate F of each atom (Ltmp = -dv g^T, Ftmp = dv g^T)
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      if (i >= nlocal) continue;
      const j = pairs[p + 1] & NEIGHMASK;
      const oj = owner[j];
      const ddx = x[3 * j] - x[3 * i], ddy = x[3 * j + 1] - x[3 * i + 1], ddz = x[3 * j + 2] - x[3 * i + 2];
      const rSq = ddx * ddx + ddy * ddy + ddz * ddz;
      const h = radius[i] + radius[oj];
      if (!(rSq < h * h)) continue;
      const r = Math.sqrt(rSq);
      const jvol = vfrac[oj], ivol = vfrac[i];
      dx[0] = ddx; dx[1] = ddy; dx[2] = ddz;
      for (let a = 0; a < 3; a++) dv[a] = vest[3 * oj + a] - vest[3 * i + a];
      const [, wfd] = spikyKernel(h, r, dim);
      for (let a = 0; a < 3; a++) g[a] = (wfd / r) * dx[a];
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          const Lt = -dv[a] * g[b];
          const Ft = dv[a] * g[b];
          L[9 * i + 3 * a + b] += jvol * Lt;
          F[9 * i + 3 * a + b] += jvol * Ft;
          if (j < nlocal) {
            L[9 * j + 3 * a + b] += ivol * Lt;
            F[9 * j + 3 * a + b] += ivol * Ft;
          }
        }
      }
    }
    // the strain increment of each owned atom: D = dt/2 (L + L^T) (native atom_data9 accumulation)
    for (let i = 0; i < nlocal; i++) {
      if (!active(type[i])) continue;
      const d = (a: number, b: number) => dt * 0.5 * (L[9 * i + 3 * a + b] + L[9 * i + 3 * b + a]);
      acc9[9 * i] += d(0, 0);
      acc9[9 * i + 1] += d(1, 1);
      acc9[9 * i + 2] += d(2, 2);
      acc9[9 * i + 3] += d(0, 1);
      acc9[9 * i + 4] += d(0, 2);
      acc9[9 * i + 5] += d(1, 2);
    }

    // AssembleStressTensor: the equation of state gives the pressure and the sound speed; the stress is -p I
    let dtCFL = BIG_DT;
    for (let i = 0; i < nlocal; i++) {
      const it = type[i];
      if (!active(it)) continue;
      const P = this.material(it);
      const rho = rmass[i] / vfrac[i];
      let newPressure = 0.0;
      if (P.eos.kind === 'tait') {
        const t = taitEOS(P.eos.exponent, P.c0, P.rho0, rho);
        newPressure = t.p;
        c0a[i] = t.c;
      } else {
        newPressure = P.bulk * (rho / P.rho0 - 1.0);
        c0a[i] = P.c0;
      }
      const K_eff = c0a[i] * c0a[i] * rho;
      S9.fill(0, 9 * i, 9 * i + 9);
      S9[9 * i] = -newPressure;
      S9[9 * i + 4] = -newPressure;
      S9[9 * i + 8] = -newPressure;
      const G_eff = 0.0;
      this.effm[i] = G_eff;
      const M = K_eff + 4.0 * G_eff / 3.0;
      const pWaveSpeed = Math.sqrt(M / rho);
      const cfl = 2 * radius[i] / pWaveSpeed;
      if (cfl < dtCFL) dtCFL = cfl;
    }
    this.dtCFL = dtCFL;

    // pair forces (native compute): stress, Monaghan-type artificial viscosity and hourglass control
    const q1 = (t: number) => this.material(t).q1;
    const hgOn = (t: number) => this.material(t).hg > 0.0;
    const Sx = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    const fs = [0, 0, 0], fv = [0, 0, 0], fh = [0, 0, 0], sum = [0, 0, 0];
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      if (i >= nlocal) continue;
      const j = pairs[p + 1] & NEIGHMASK;
      const oj = owner[j];
      const itype = type[i], jtype = type[oj];
      const ddx = x[3 * j] - x[3 * i], ddy = x[3 * j + 1] - x[3 * i + 1], ddz = x[3 * j + 2] - x[3 * i + 2];
      const rSq = ddx * ddx + ddy * ddy + ddz * ddz;
      const h = radius[i] + radius[oj];
      if (!(rSq < h * h)) continue;
      dx[0] = ddx; dx[1] = ddy; dx[2] = ddz;
      for (let a = 0; a < 3; a++) dv[a] = vest[3 * oj + a] - vest[3 * i + a];
      const r = Math.sqrt(rSq);
      const jvol = vfrac[oj], ivol = vfrac[i];
      const [, wfd] = spikyKernel(h, r, dim);
      for (let a = 0; a < 3; a++) g[a] = (wfd / r) * dx[a];
      const delVdotDelR0 = (dx[0] * dv[0] + dx[1] * dv[1] + dx[2] * dv[2]) / (r + 0.1 * h);
      for (let q = 0; q < 9; q++) Sx[q] = S9[9 * i + q] + S9[9 * oj + q];
      // f_stress = -ivol jvol S g (DO NOT TOUCH SIGN, native)
      const sc = -ivol * jvol;
      for (let a = 0; a < 3; a++) {
        let acc = 0;
        for (let b = 0; b < 3; b++) acc += sc * Sx[3 * a + b] * g[b];
        fs[a] = acc;
      }
      // artificial viscosity: the approach speed is limited to 1.1 c_ij (LimitDoubleMagnitude)
      const c_ij = 0.5 * (c0a[i] + c0a[oj]);
      const limit = 1.1 * c_ij;
      const delVdotDelR = Math.abs(delVdotDelR0) > limit ? limit * Math.sign(delVdotDelR0) : delVdotDelR0;
      const mu_ij = h * delVdotDelR / (r + 0.1 * h);
      const rho_ij = 0.5 * (rmass[i] / ivol + rmass[oj] / jvol);
      const visc_magnitude = 0.5 * (q1(itype) + q1(jtype)) * c_ij * mu_ij / rho_ij;
      const cv = -rmass[i] * rmass[oj] * visc_magnitude;
      for (let a = 0; a < 3; a++) fv[a] = cv * g[a];
      // hourglass control (native ComputeHourglassForce, only when both types have an amplitude)
      if (hgOn(itype) && hgOn(jtype)) {
        let dvEstSq = 0, dvDot = 0;
        const est = [0, 0, 0];
        for (let a = 0; a < 3; a++) {
          let acc = 0;
          for (let b = 0; b < 3; b++) acc += (F[9 * i + 3 * a + b] + F[9 * oj + 3 * a + b]) * dx[b];
          est[a] = -0.5 * acc;
        }
        for (let a = 0; a < 3; a++) {
          dvEstSq += est[a] * est[a];
          dvDot += est[a] * dv[a];
        }
        const dvEstNorm = Math.sqrt(dvEstSq);
        const dvNorm = Math.sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
        const hurz = dvDot / (dvEstNorm * dvNorm + 1.0e-16);
        if (hurz < 0.0) {
          for (let a = 0; a < 3; a++) fh[a] = cv * g[a];
        } else {
          fh[0] = fh[1] = fh[2] = 0;
        }
      } else {
        fh[0] = fh[1] = fh[2] = 0;
      }
      for (let a = 0; a < 3; a++) sum[a] = fs[a] + fv[a] + fh[a];
      const deltaE = sum[0] * dv[0] + sum[1] * dv[1] + sum[2] * dv[2];
      for (let a = 0; a < 3; a++) f[3 * i + a] += sum[a];
      desph[i] += deltaE;
      nn[i] += 1;
      const w = j < nlocal ? 1.0 : 0.5;
      const vir = pc.acc.virial;
      vir[0] += w * dx[0] * sum[0];
      vir[1] += w * dx[1] * sum[1];
      vir[2] += w * dx[2] * sum[2];
      vir[3] += w * dx[0] * sum[1];
      vir[4] += w * dx[0] * sum[2];
      vir[5] += w * dx[1] * sum[2];
      // per-atom: half of dx (x) sum to i, and to j when it is owned (ev_tally_xyz, newton off)
      if (pc.vatom) {
        const va = pc.vatom;
        const w6 = [0.5 * dx[0] * sum[0], 0.5 * dx[1] * sum[1], 0.5 * dx[2] * sum[2], 0.5 * dx[0] * sum[1], 0.5 * dx[0] * sum[2], 0.5 * dx[1] * sum[2]];
        for (let c = 0; c < 6; c++) va[6 * i + c] += w6[c];
        if (j < nlocal) for (let c = 0; c < 6; c++) va[6 * j + c] += w6[c];
      }
      if (j < nlocal) {
        for (let a = 0; a < 3; a++) f[3 * j + a] -= sum[a];
        desph[j] += deltaE;
        nn[j] += 1;
      }
    }
  }
}
