/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/pair_smd_tlsph.cpp and pair_smd_tlsph.h (PairTlsph: pair_style smd/tlsph; PreCompute,
 * AssembleStress, ComputeForces, ComputePressure, ComputeStressDeviator, effective_longitudinal_modulus), the linear
 * EOS and linear strength of src/MACHDYN/smd_material_models.cpp, the spiky kernel of src/MACHDYN/smd_kernels.h, and the
 * partner (reference configuration) list of src/MACHDYN/fix_smd_tlsph_reference_configuration.cpp at commit b332f28.
 * MACHDYN authors (from the sources): Georg C. Ganzenmueller, Fraunhofer EMI. LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Smooth Mach Dynamics, total Lagrangian SPH (docs.lammps.org/pair_smd.html): the particles of a type with coefficients
 * interact through a reference configuration (the partner list, built once at the start of the run from the
 * positions of that time). Each owned atom sums its forces from its partners: the first Piola-Kirchhoff stress of both
 * atoms, an artificial viscosity and an hourglass control. The stress is linear elastic (*STRENGTH_LINEAR) with a
 * linear EOS (*EOS_LINEAR); other material models, failure models and reference configuration updates are refused.
 *
 * Differences from the native code that are kept on purpose: the partner list is built by an all-pairs search with
 * periodic images in the periodic dimensions of the box (native: the neighbor list), in the order of the atoms.
 * The sums over partners therefore run in another order, which changes the last bits of the forces only.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { SMD_DAMAGE, SMD_DESPH, SMD_EPS, SMD_EPSRATE, SMD_STRESS, SMD_VEST } from '../../engine/smd';
import {
  add3, det3, deviator3, identity3, inverse3, mat3, mul3, norm3, polDec, pseudoInverseSVD, scale3, setM, sub3, trace3,
  transpose3, type Mat3,
} from './smd_math';
import { log as libmLog } from '../libm/log';
import { pow as libmPow } from '../libm/pow';

/** pair_smd_tlsph.cpp: DETF_MIN and DETF_MAX (deletion of a particle outside this range of det F). */
const DETF_MIN = 0.2;
const DETF_MAX = 2.0;

/** smd_kernels.h, spiky_kernel_and_derivative: wf and its derivative for a pair at distance r within the kernel radius h. */
export const spikyKernel = (h: number, r: number, dimension: number): [number, number] => {
  if (r > h) return [0, 0];
  const hr = h - r;
  let wfd: number, wf: number;
  if (dimension === 2) {
    const n = 0.3141592654e0 * h * h * h * h * h;
    wfd = -3.0e0 * hr * hr / n;
    wf = -0.333333333333e0 * hr * wfd;
  } else {
    wfd = -14.0323944878e0 * hr * hr / (h * h * h * h * h * h);
    wf = -0.333333333333e0 * hr * wfd;
  }
  return [wf, wfd];
};

/** SMD_Math::LimitDoubleMagnitude: |x| is limited to the given magnitude, keeping its sign. */
const limitMagnitude = (x: number, limit: number): number => (Math.abs(x) > limit ? limit * Math.sign(x || 1) : x);

/** Per type material parameters (pair_smd_tlsph.cpp, coeff: *COMMON and the derived moduli). */
interface TypeParams {
  rho0: number; E: number; nu: number; q1: number; q2: number; hgAmp: number; cp: number;
  lambda: number; G: number; M: number; c: number; K: number;
  eos: 'linear' | 'polynomial' | 'none';
  /** *EOS_POLYNOMIAL C0 .. C6 (pair_smd.html, the polynomial equation of state). */
  eosC: number[];
  strength: 'linear' | 'linear_plastic' | 'johnson_cook' | 'none';
  /** *STRENGTH_LINEAR_PLASTIC yield stress and hardening parameter (yield = sigma_y + H ep). */
  yieldStress: number; hardening: number;
  /** *JOHNSON_COOK A B a C epdot0 T0 Tmelt M (the temperature factor of the source is not applied, see johnsonCook). */
  jc: { A: number; B: number; a: number; C: number; epdot0: number; T0: number; Tmelt: number; M: number };
  /** *FAILURE_MAX_PLASTIC_STRAIN threshold, or null (no failure): the failure is an integration-point-wise model. */
  failMaxPlastic: number | null;
}

/**
 * polynomialEOS (smd_material_models.cpp): mu = rho / rho0 - 1, the polynomial in mu (the C2 term only for mu > 0), with
 * the sign flipped; returns the pressure pFinal. The energy and C4 to C6 do not enter (as native).
 */
const polynomialEOS = (rho: number, rho0: number, C: number[], _pInitial: number, _dt: number): number => {
  const mu = rho / rho0 - 1.0;
  let pFinal: number;
  if (mu > 0.0) pFinal = C[0] + C[1] * mu + C[2] * mu * mu + C[3] * mu * mu * mu;
  else pFinal = C[0] + C[1] * mu + C[3] * mu * mu * mu;
  return -pFinal;
};

/**
 * JohnsonCookStrength (smd_material_models.cpp, the version compiled in): yield = (A + B ep^a) (1 + C ln(max(epdot/epdot0, 1))),
 * with ep the effective plastic strain and epdot its rate; the temperature factor is commented out in the source, so the
 * yield stress does not depend on the energy. Returns the final deviator, its rate and the plastic strain increment.
 */
const johnsonCook = (P: TypeParams, dt: number, ep: number, epdot: number, sigmaInitialDev: Mat3, dDev: Mat3): { final: Mat3; inc: number } => {
  const { A, B, a, C, epdot0 } = P.jc;
  const epdotRatio = Math.max(epdot / epdot0, 1.0);
  const yieldStress = (A + B * libmPow(ep, a)) * (1.0 + C * libmLog(epdotRatio));
  return linearPlasticReturn(P, dt, yieldStress, sigmaInitialDev, dDev);
};

/**
 * LinearPlasticStrength (smd_material_models.cpp): elastic trial deviator, and where its norm sqrt(3/2)|s| exceeds the yield
 * stress, the deviator is scaled back to it and the plastic strain increment is (J2 - sigma_y) / (3 G).
 */
const linearPlasticReturn = (P: TypeParams, dt: number, yieldStress: number, sigmaInitialDev: Mat3, dDev: Mat3): { final: Mat3; inc: number } => {
  const devRate = scale3(dDev, 2.0 * P.G);
  const trial = add3(sigmaInitialDev, scale3(devRate, dt));
  const J2 = Math.sqrt(3.0 / 2.0) * norm3(trial);
  if (J2 < yieldStress) return { final: trial, inc: 0 };
  return { final: scale3(trial, yieldStress / J2), inc: (J2 - yieldStress) / (3.0 * P.G) };
};

/** PairTlsph::effective_longitudinal_modulus (no damage dependence in this model). */
const effectiveModulus = (P: TypeParams, dt: number, dIso: number, pRate: number, dDev: Mat3, sigmaDevRate: Mat3, dim: number): { K: number; mu: number; M: number } => {
  let K: number;
  if (dt * dIso > 1.0e-6) {
    K = pRate / dIso;
    if (K < 0) K = P.K;
  } else {
    K = P.K;
  }
  let mu: number, shearRateSq: number;
  // (row, col) of the matrices is 3 * row + col
  const d = (a: number, b: number) => dDev[3 * a + b];
  const sr = (a: number, b: number) => sigmaDevRate[3 * a + b];
  if (dim === 3) {
    mu = 0.5 * (sr(0, 1) / (d(0, 1) + 1.0e-16) + sr(0, 2) / (d(0, 2) + 1.0e-16) + sr(1, 2) / (d(1, 2) + 1.0e-16));
    shearRateSq = d(0, 1) * d(0, 1) + d(0, 2) * d(0, 2) + d(1, 2) * d(1, 2);
  } else {
    mu = 0.5 * (sr(0, 1) / (d(0, 1) + 1.0e-16));
    shearRateSq = d(0, 1) * d(0, 1);
  }
  if (dt * dt * shearRateSq < 1.0e-8) mu = P.G;
  if (mu < P.G) mu = P.G;
  if (mu < 0) throw new StyleError(`mu_eff = ${mu} is negative`);
  let M = K + 4.0 * mu / 3.0;
  if (M < P.M) M = P.M;
  return { K, mu, M };
};

/** One partner of an atom in the reference configuration: its tag (0 = removed) and the kernel values at that time. */
interface Partner { tag: number; wf: number; wfd: number; degradation: number; energyPerBond: number }

/** Per-atom persistent state of the SMD pair (per-atom arrays of the native style, owned atoms only). */
interface Work {
  n: number;
  K: Mat3[]; Fincr: Mat3[]; Fdot: Mat3[]; R: Mat3[]; W: Mat3[]; D: Mat3[]; PK1: Mat3[]; CauchyStress: Mat3[];
  FincrInv: Mat3[]; detF: Float64Array; smoothVel: Float64Array; numNeighs: Int32Array; hourglass: Float64Array;
  particleDt: Float64Array;
}

const newWork = (n: number): Work => ({
  n,
  K: Array.from({ length: n }, () => identity3()), Fincr: Array.from({ length: n }, () => identity3()),
  Fdot: Array.from({ length: n }, () => mat3()), R: Array.from({ length: n }, () => identity3()),
  W: Array.from({ length: n }, () => mat3()), D: Array.from({ length: n }, () => mat3()),
  PK1: Array.from({ length: n }, () => mat3()), CauchyStress: Array.from({ length: n }, () => mat3()),
  FincrInv: Array.from({ length: n }, () => identity3()), detF: new Float64Array(n).fill(1),
  smoothVel: new Float64Array(3 * n), numNeighs: new Int32Array(n), hourglass: new Float64Array(n),
  particleDt: new Float64Array(n),
});


export class PairSMDTlsph extends Pair {
  readonly name = 'smd/tlsph';
  /** Global virial from the explicit sums (ev_tally_xyz), not fdotr; a hybrid sub-style takes it from pc.acc.virial. */
  virialFdotr = false;
  explicitVirial = true;
  private setflag = new Uint8Array(0);
  private params: (TypeParams | null)[] = [];
  private maxrad = new Float64Array(0);
  private first = true;
  /** Critical time step of the last force evaluation (smd/adjust_dt reads it; fix extract dtCFL). */
  dtCFL = 0;
  private partners: Partner[][] = [];
  private readonly order = new NativePairOrder();
  /** Per-atom state of the last evaluation (read by the smd computes). */
  work: Work = newWork(0);
  /** Atom ID -> owned index for the current evaluation (set by compute). */
  private tagMap: Int32Array = new Int32Array(0);
  /** Scratch vectors of the partner loops (no allocation per partner). */
  private readonly scratch = { d0: new Float64Array(3), dx: new Float64Array(3), dv: new Float64Array(3), dvint: new Float64Array(3), g: new Float64Array(3), Fs: new Float64Array(3), fs: new Float64Array(3) };
  private log: ((text: string) => void) | null = null;
  /** pair_style smd/tlsph arguments: none are accepted in this port (the update keyword is not supported). */
  settings(args: string[], _ctx: StyleContext): void {
    if (args.length) throw new StyleError(`pair_style smd/tlsph keyword '${args[0]}' is not supported by the browser engine yet (only the default reference configuration)`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.setflag = new Uint8Array((ntypes + 1) * (ntypes + 1));
    this.params = Array.from({ length: ntypes + 1 }, () => null);
    this.maxrad = new Float64Array(ntypes + 1);
  }

  /**
   * pair_coeff I I *COMMON rho0 E nu q1 q2 hg cp *STRENGTH_LINEAR *EOS_LINEAR *END (pair_smd.html and the sources).
   * Other material keywords are refused with a StyleError naming them.
   */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 3) throw new StyleError('number of arguments for pair smd/tlsph is too small');
    if (!this.ntypes) throw new StyleError('pair_coeff before the box exists');
    const i = parseInt(args[0], 10), j = parseInt(args[1], 10);
    if (!(i >= 1 && i <= this.ntypes) || i !== j) throw new StyleError('TLSPH coefficients can only be specified between particles of same type!');
    if (args[2] !== '*COMMON') throw new StyleError('common keyword missing!');
    const common = args.slice(3, 10);
    if (common.length < 7 || args[10] === undefined || !args[10].startsWith('*')) throw new StyleError('expected 7 arguments following *COMMON');
    const [rho0, E, nu, q1, q2, hgAmp, cp] = common.map((w, k) => parseNum(w, ['rho0', 'E', 'nu', 'q1', 'q2', 'hg', 'cp'][k]));
    const lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    const G = E / (2 * (1 + nu));
    const M = lambda + 2 * G;
    const c = Math.sqrt((lambda + 2 * G) / rho0);
    const K = lambda + 2 * G / 3;
    const p: TypeParams = {
      rho0, E, nu, q1, q2, hgAmp, cp, lambda, G, M, c, K, eos: 'none', eosC: [0, 0, 0, 0, 0, 0, 0],
      strength: 'none', yieldStress: 0, hardening: 0, jc: { A: 0, B: 0, a: 0, C: 0, epdot0: 1, T0: 0, Tmelt: 1, M: 1 }, failMaxPlastic: null,
    };
    let k = 10;
    for (;;) {
      if (k >= args.length) throw new StyleError('no *END terminates *COMMON material keywords');
      const kw = args[k];
      if (kw === '*END') break;
      if (!kw.startsWith('*')) throw new StyleError(`unknown pair smd/tlsph keyword ${kw}`);
      // the values of a keyword are the words up to the next keyword (pair_coeff.html: "*KEYWORD" blocks)
      let e = k + 1;
      while (e < args.length && !args[e].startsWith('*')) e++;
      if (e >= args.length) throw new StyleError(`no *KEYWORD terminates ${kw}`);
      const vals = args.slice(k + 1, e);
      const need = (n: number) => {
        if (vals.length !== n) throw new StyleError(`expected ${n} arguments following ${kw} but got ${vals.length}`);
        return vals.map((w, q) => parseNum(w, `${kw} value ${q + 1}`));
      };
      switch (kw) {
        case '*STRENGTH_LINEAR': need(0); p.strength = 'linear'; break;
        case '*EOS_LINEAR': need(0); p.eos = 'linear'; break;
        case '*EOS_POLYNOMIAL': p.eos = 'polynomial'; p.eosC = need(7); break;
        case '*JOHNSON_COOK': {
          const [A, B, a, C, epdot0, T0, Tmelt, M] = need(8);
          p.strength = 'johnson_cook';
          p.jc = { A, B, a, C, epdot0, T0, Tmelt, M };
          break;
        }
        // LAMMPS: "*STRENGTH_LINEAR_PLASTIC yield_stress hardening_parameter" (the source reads 2 arguments)
        case '*STRENGTH_LINEAR_PLASTIC': {
          const [sigmaY, H] = need(2);
          p.strength = 'linear_plastic';
          p.yieldStress = sigmaY;
          p.hardening = H;
          break;
        }
        case '*FAILURE_MAX_PLASTIC_STRAIN': p.failMaxPlastic = need(1)[0]; break;
        default:
          throw new StyleError(`pair smd/tlsph material keyword ${kw} is not supported by the browser engine yet (supported: *STRENGTH_LINEAR, *STRENGTH_LINEAR_PLASTIC, *EOS_LINEAR, *EOS_POLYNOMIAL, *JOHNSON_COOK, *FAILURE_MAX_PLASTIC_STRAIN)`);
      }
      k = e;
    }
    this.params[i] = p;
    this.setflag[i * (this.ntypes + 1) + i] = 1;
    this.log = ctx.log;
    this.version++;
  }

  /** Cutoff of a type pair: the sum of the largest kernel radii of the two types (native init_one). */
  initOne(i: number, j: number): number {
    if (!this.setflag[i * (this.ntypes + 1) + j]) throw new StyleError('All pair coeffs are not set');
    return this.maxrad[i] + this.maxrad[j];
  }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) throw new StyleError('pair smd/tlsph needs a box and atoms');
    if (!s.radius || !s.rmass || !s.vfrac || !s.x0) throw new StyleError('pair smd/tlsph needs atom_style smd');
    const onerad = new Float64Array(this.ntypes + 1);
    for (let i = 0; i < s.n; i++) onerad[s.type[i]] = Math.max(onerad[s.type[i]], s.radius[i]);
    this.maxrad = onerad;
    this.log = ctx.log;
  }

  /** Material parameters of a type (the coefficients), or an error for a type without them. */
  private material(t: number): TypeParams {
    const p = this.params[t];
    if (!p || !this.setflag[t * (this.ntypes + 1) + t]) throw new StyleError(`pair smd/tlsph: type ${t} has no coefficients`);
    return p;
  }

  /** Periodic dimensions used for minimum images and partner images (2d: only x and y, as native communication). */
  private periodicDims(s: SimState): number[] {
    const dims: number[] = [];
    for (let d = 0; d < (s.dimension === 2 ? 2 : 3); d++) if (s.box.periodic[d]) dims.push(d);
    return dims;
  }

  /** Domain::minimum_image for a displacement in the periodic dimensions. */
  private minimumImage(s: SimState, dx: Float64Array, dims: number[]): void {
    for (const d of dims) {
      const L = s.box.hi[d] - s.box.lo[d];
      const half = 0.5 * L;
      if (Math.abs(dx[d]) > half) dx[d] -= (dx[d] < 0 ? -L : L);
    }
  }

  /**
   * Partner list (native: fix SMD_TLSPH_NEIGHBORS, setup): from the pairs of the neighbor list, in the order of the native
   * enumeration (nativeOrder.ts), every pair closer than the sum of the kernel radii is added to the partners of the
   * first atom and, when the second one is owned, to its partners too (newton off). A ghost image adds the owner's tag.
   * Summing the forces in the same partner order as native keeps the rounding of the sums the same.
   */
  private buildPartners(pc: PairCompute): void {
    const s = pc.s;
    const n = s.n;
    const nlocal = pc.nlocal;
    const xall = pc.x;
    const owner = pc.nb.owner;
    const partners: Partner[][] = Array.from({ length: n }, () => []);
    const active = (i: number) => !!this.params[s.type[i]] && !!this.setflag[s.type[i] * (this.ntypes + 1) + s.type[i]];
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const jIdx = pairs[p + 1] & NEIGHMASK;
      if (i >= n || !active(i)) continue;
      const o = owner[jIdx];
      // a periodic image of the atom itself (no native meaning in a reference configuration of a single atom)
      if (o === i && jIdx >= nlocal) continue;
      if (o >= n || !active(o)) continue;
      const dx = xall[3 * i] - xall[3 * jIdx], dy = xall[3 * i + 1] - xall[3 * jIdx + 1];
      const dz = s.dimension === 2 ? 0 : xall[3 * i + 2] - xall[3 * jIdx + 2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const h = s.radius![i] + s.radius![o];
      if (!(r < h)) continue;
      const [wf, wfd] = spikyKernel(h, r, s.dimension);
      // the native partner arrays of kernel values are single precision (float wf_list, wfd_list)
      partners[i].push({ tag: s.id[o], wf: Math.fround(wf), wfd: Math.fround(wfd), degradation: 0, energyPerBond: 0 });
      if (jIdx < nlocal) partners[o].push({ tag: s.id[i], wf: Math.fround(wf), wfd: Math.fround(wfd), degradation: 0, energyPerBond: 0 });
    }
    this.partners = partners;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const n = s.n;
    if (this.work.n !== n) this.work = newWork(n);
    const W = this.work;
    this.tagMap = this.tagIndex(s);
    if (this.first) {
      // native: the first evaluation of the style only zeroes its per-atom state; the partner list is built now
      this.first = false;
      W.Fincr.forEach((m) => m.fill(0));
      W.detF.fill(0);
      W.smoothVel.fill(0);
      W.D.forEach((m) => m.fill(0));
      W.numNeighs.fill(0);
      W.CauchyStress.forEach((m) => m.fill(0));
      W.hourglass.fill(0);
      W.particleDt.fill(0);
      this.buildPartners(pc);
      return;
    }
    this.preCompute(s);
    this.assembleStress(s);
    this.computeForces(s, pc);
  }

  /** PairTlsph::PreCompute: the deformation gradient increment, rotation and strain rate of every owned atom. */
  private preCompute(s: SimState): void {
    const n = s.n, W = this.work;
    const dims = this.periodicDims(s);
    const periodic = dims.length > 0;
    const x = s.x, v = s.v;
    const vest = s.custom.get(SMD_VEST)!.data;
    const x0 = s.x0!;
    const vfrac = s.vfrac!;
    const mol = s.molecule;
    const radius = s.radius!;
    const tagIndex = this.tagMap;
    const dim = s.dimension;
    const d0 = this.scratch.d0, dx = this.scratch.dx, dv = this.scratch.dv, dvint = this.scratch.dvint, g = this.scratch.g;
    for (let i = 0; i < n; i++) {
      const itype = s.type[i];
      if (!this.setflag[itype * (this.ntypes + 1) + itype]) continue;
      const K = W.K[i], Fincr = W.Fincr[i], Fdot = W.Fdot[i];
      K.fill(0);
      Fincr.fill(0);
      Fdot.fill(0);
      W.numNeighs[i] = 0;
      W.smoothVel[3 * i] = W.smoothVel[3 * i + 1] = W.smoothVel[3 * i + 2] = 0;
      W.hourglass[i] = 0;
      if (mol[i] < 0) continue;
      const [wf0] = spikyKernel(2 * radius[i], 0, dim);
      const voli = vfrac[i];
      let shepard = wf0 * voli;
      const xi0 = x[3 * i], xi1 = x[3 * i + 1], xi2 = x[3 * i + 2];
      const x0i0 = x0[3 * i], x0i1 = x0[3 * i + 1], x0i2 = x0[3 * i + 2];
      const vi0 = vest[3 * i], vi1 = vest[3 * i + 1], vi2 = vest[3 * i + 2];
      const vint0 = v[3 * i], vint1 = v[3 * i + 1], vint2 = v[3 * i + 2];
      let sm0 = 0, sm1 = 0, sm2 = 0;
      const plist = this.partners[i] ?? [];
      const keep: Partner[] = [];
      for (const p of plist) {
        if (p.tag === 0) continue;
        const j = tagIndex[p.tag];
        if (j < 0) { p.tag = 0; continue; }
        keep.push(p);
        if (mol[j] < 0) continue;
        if (mol[i] !== mol[j]) continue;
        d0[0] = x0[3 * j] - x0i0; d0[1] = x0[3 * j + 1] - x0i1; d0[2] = x0[3 * j + 2] - x0i2;
        if (periodic) this.minimumImage(s, d0, dims);
        dx[0] = x[3 * j] - xi0; dx[1] = x[3 * j + 1] - xi1; dx[2] = x[3 * j + 2] - xi2;
        dv[0] = vest[3 * j] - vi0; dv[1] = vest[3 * j + 1] - vi1; dv[2] = vest[3 * j + 2] - vi2;
        dvint[0] = v[3 * j] - vint0; dvint[1] = v[3 * j + 1] - vint1; dvint[2] = v[3 * j + 2] - vint2;
        const r0 = Math.sqrt(d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2]);
        const volj = vfrac[j];
        const scale = 1 - p.degradation;
        const wf = p.wf * scale, wfd = p.wfd * scale;
        const kf = wfd / r0;
        g[0] = kf * d0[0]; g[1] = kf * d0[1]; g[2] = kf * d0[2];
        for (let a2 = 0; a2 < 3; a2++) {
          for (let b2 = 0; b2 < 3; b2++) {
            K[3 * a2 + b2] += volj * (-g[a2] * d0[b2]);
            Fdot[3 * a2 + b2] += volj * (-dv[a2] * g[b2]);
            Fincr[3 * a2 + b2] += volj * (-(dx[a2] - d0[a2]) * g[b2]);
          }
        }
        shepard += volj * wf;
        sm0 += volj * wf * dvint[0]; sm1 += volj * wf * dvint[1]; sm2 += volj * wf * dvint[2];
        W.numNeighs[i]++;
      }
      this.partners[i] = keep;
      if (shepard > 0) {
        W.smoothVel[3 * i] = sm0 / shepard;
        W.smoothVel[3 * i + 1] = sm1 / shepard;
        W.smoothVel[3 * i + 2] = sm2 / shepard;
      } else {
        W.smoothVel[3 * i] = W.smoothVel[3 * i + 1] = W.smoothVel[3 * i + 2] = 0;
      }
      pseudoInverseSVD(K);
      setM(Fdot, mul3(Fdot, K));
      setM(Fincr, mul3(Fincr, K));
      for (let c = 0; c < 9; c += 4) Fincr[c] += 1;
      // JAUMANN is false in the native code: the rotation comes from the polar decomposition of the increment
      const R = W.R[i];
      const U = mat3();
      const status = polDec(Fincr, R, U);
      if (!status) {
        mol[i] = -1;
      } else {
        setM(Fincr, mul3(R, U));
      }
      W.detF[i] = det3(Fincr);
      setM(W.FincrInv[i], inverse3(Fincr));
      const L = mul3(Fdot, W.FincrInv[i]);
      const Lt = transpose3(L);
      setM(W.W[i], scale3(sub3(L, Lt), 0.5));
      setM(W.D[i], mul3(mul3(transpose3(R), scale3(add3(L, Lt), 0.5)), R));
      if (W.detF[i] < DETF_MIN || W.detF[i] > DETF_MAX || W.numNeighs[i] === 0) {
        this.log?.(`deleting particle [${s.id[i]}] because det(F)=${W.detF[i]}f is outside stable range ${DETF_MIN} -- ${DETF_MAX}`);
        mol[i] = -1;
      }
      if (mol[i] < 0) {
        W.D[i].fill(0);
        Fdot.fill(0);
        setM(Fincr, identity3());
        W.smoothVel[3 * i] = W.smoothVel[3 * i + 1] = W.smoothVel[3 * i + 2] = 0;
        W.detF[i] = 1;
        setM(K, identity3());
        v[3 * i] = 0; v[3 * i + 1] = 0; v[3 * i + 2] = 0;
      }
    }
  }

  /** Tag -> owned index (-1 when absent); atom IDs are the partner keys. */
  private tagIndex(s: SimState): Int32Array {
    let maxTag = 0;
    for (let i = 0; i < s.n; i++) if (s.id[i] > maxTag) maxTag = s.id[i];
    const idx = new Int32Array(maxTag + 1).fill(-1);
    for (let i = 0; i < s.n; i++) idx[s.id[i]] = i;
    return idx;
  }

  /** PairTlsph::AssembleStress: the Cauchy stress of each atom, its PK1 stress and the wave-speed time step. */
  private assembleStress(s: SimState): void {
    const n = s.n, W = this.work;
    const dt = s.dt;
    const smdStress = s.custom.get(SMD_STRESS)!.data;
    const epsRate = s.custom.get(SMD_EPSRATE)!.data;
    const epsPl = s.custom.get(SMD_EPS)!.data;
    const damage = s.custom.get(SMD_DAMAGE)!.data;
    const vfrac = s.vfrac!, rmass = s.rmass!, radius = s.radius!;
    const mol = s.molecule;
    this.dtCFL = 1.0e22;
    for (let i = 0; i < n; i++) {
      W.particleDt[i] = 0;
      const itype = s.type[i];
      if (!this.setflag[itype * (this.ntypes + 1) + itype]) continue;
      const P = this.material(itype);
      if (mol[i] > 0) {
        const o = 6 * i;
        const sigmaInitial = mat3();
        sigmaInitial[0] = smdStress[o]; sigmaInitial[1] = smdStress[o + 1]; sigmaInitial[2] = smdStress[o + 2];
        sigmaInitial[4] = smdStress[o + 3]; sigmaInitial[5] = smdStress[o + 4]; sigmaInitial[8] = smdStress[o + 5];
        sigmaInitial[3] = sigmaInitial[1]; sigmaInitial[6] = sigmaInitial[2]; sigmaInitial[7] = sigmaInitial[5];
        const pInitial = trace3(sigmaInitial) / 3;
        const sigmaInitialDev = deviator3(sigmaInitial);
        const D = W.D[i];
        const dIso = trace3(D);
        const dDev = deviator3(D);
        const rho = rmass[i] / (W.detF[i] * vfrac[i]);
        // ComputePressure
        let pFinal = 0;
        if (P.eos === 'linear') {
          // LinearEOS: p_rate = lambda * d_iso with the bulk modulus; pFinal = pInitial + dt p_rate
          const pRate = P.K * dIso;
          pFinal = pInitial + dt * pRate;
        } else if (P.eos === 'polynomial') {
          pFinal = polynomialEOS(rho, P.rho0, P.eosC, pInitial, dt);
        }
        // ComputeStressDeviator (STRENGTH_LINEAR: rate = 2 G d_dev; Johnson-Cook: yield stress from the plastic strain)
        let sigmaFinalDev = mat3();
        let plasticInc = 0;
        if (P.strength === 'linear') {
          const rate = scale3(dDev, 2 * P.G);
          sigmaFinalDev = add3(sigmaInitialDev, scale3(rate, dt));
        } else if (P.strength === 'linear_plastic') {
          // native: yieldStress = Lookup[YIELD_STRESS] + Lookup[HARDENING_PARAMETER] * eff_plastic_strain[i]
          const jc = linearPlasticReturn(P, dt, P.yieldStress + P.hardening * epsPl[i], sigmaInitialDev, dDev);
          sigmaFinalDev = jc.final;
          plasticInc = jc.inc;
        } else if (P.strength === 'johnson_cook') {
          const jc = johnsonCook(P, dt, epsPl[i], epsRate[i], sigmaInitialDev, dDev);
          sigmaFinalDev = jc.final;
          plasticInc = jc.inc;
        }
        epsPl[i] += plasticInc;
        // the rate of the effective plastic strain decays and follows the increment (tav: native time scale)
        epsRate[i] -= epsRate[i] * dt / (1000 * radius[i] / P.c);
        epsRate[i] += plasticInc / (1000 * radius[i] / P.c);
        epsRate[i] = Math.max(0, epsRate[i]);
        const eye = identity3();
        const sigmaFinal = add3(scale3(eye, pFinal), sigmaFinalDev);
        // non-Jaumann: T = R sigmaFinal R^T; the stored stress is the unrotated one
        const T = mul3(mul3(W.R[i], sigmaFinal), transpose3(W.R[i]));
        smdStress[o] = sigmaFinal[0];
        smdStress[o + 1] = sigmaFinal[1];
        smdStress[o + 2] = sigmaFinal[2];
        smdStress[o + 3] = sigmaFinal[4];
        smdStress[o + 4] = sigmaFinal[5];
        smdStress[o + 5] = sigmaFinal[8];
        // ComputeDamage (failure model integration-point-wise): the maximum plastic strain criterion
        if (P.failMaxPlastic !== null && epsPl[i] >= P.failMaxPlastic) damage[i] = 1.0;
        setM(W.CauchyStress[i], T);
        const PK1 = mul3(mul3(scale3(T, W.detF[i]), transpose3(W.FincrInv[i])), W.K[i]);
        setM(W.PK1[i], PK1);
        const deltaSigma = sub3(sigmaFinal, sigmaInitial);
        const pRate = trace3(deltaSigma) / (3.0 * dt + 1.0e-16);
        const sigmaDevRate = scale3(deviator3(deltaSigma), 1 / (dt + 1.0e-16));
        const { M: Meff } = effectiveModulus(P, dt, dIso, pRate, dDev, sigmaDevRate, s.dimension);
        const pWave = Math.sqrt(Meff / rho);
        W.particleDt[i] = 2.0 * radius[i] / pWave;
        this.dtCFL = Math.min(this.dtCFL, W.particleDt[i]);
      } else {
        W.PK1[i].fill(0);
        setM(W.K[i], identity3());
        W.CauchyStress[i].fill(0);
        for (let c = 0; c < 6; c++) smdStress[6 * i + c] = 0;
      }
    }
  }

  /** ComputeForces: the forces of the partners of each owned atom, the viscosity and the hourglass control. */
  private computeForces(s: SimState, pc: PairCompute): void {
    const n = s.n, W = this.work;
    const dims = this.periodicDims(s);
    const periodic = dims.length > 0;
    const x = s.x, vest = s.custom.get(SMD_VEST)!.data;
    const x0 = s.x0!, f = pc.f;
    const vfrac = s.vfrac!, rmass = s.rmass!, radius = s.radius!;
    const mol = s.molecule;
    const damage = s.custom.get(SMD_DAMAGE)!.data;
    const epsPl = s.custom.get(SMD_EPS)!.data;
    const desph = s.custom.get(SMD_DESPH)!.data;
    const tagIndex = this.tagMap;
    const v = pc.acc.virial;
    const d0 = this.scratch.d0, dx = this.scratch.dx, dv = this.scratch.dv, Fs = this.scratch.Fs;
    for (let i = 0; i < n; i++) {
      if (mol[i] < 0) continue;
      const itype = s.type[i];
      if (!this.setflag[itype * (this.ntypes + 1) + itype]) continue;
      const P = this.material(itype);
      const voli = vfrac[i];
      const [wf0] = spikyKernel(2 * radius[i], 0, s.dimension);
      let shepard = wf0 * voli;
      const xi0 = x[3 * i], xi1 = x[3 * i + 1], xi2 = x[3 * i + 2];
      const x0i0 = x0[3 * i], x0i1 = x0[3 * i + 1], x0i2 = x0[3 * i + 2];
      const vi0 = vest[3 * i], vi1 = vest[3 * i + 1], vi2 = vest[3 * i + 2];
      const PK1i = W.PK1[i], Fi = W.Fincr[i];
      for (const p of this.partners[i] ?? []) {
        if (p.tag === 0) continue;
        const j = tagIndex[p.tag];
        if (j < 0) { p.tag = 0; continue; }
        if (mol[j] < 0) continue;
        if (mol[i] !== mol[j]) continue;
        if (s.type[j] !== itype) throw new StyleError('particle pair is not of same type!');
        d0[0] = x0[3 * j] - x0i0; d0[1] = x0[3 * j + 1] - x0i1; d0[2] = x0[3 * j + 2] - x0i2;
        if (periodic) this.minimumImage(s, d0, dims);
        const r0Sq = d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2];
        const h = radius[i] + radius[j];
        const r0 = Math.sqrt(r0Sq);
        const volj = vfrac[j];
        dx[0] = x[3 * j] - xi0; dx[1] = x[3 * j + 1] - xi1; dx[2] = x[3 * j + 2] - xi2;
        dv[0] = vest[3 * j] - vi0; dv[1] = vest[3 * j + 1] - vi1; dv[2] = vest[3 * j + 2] - vi2;
        const r = Math.sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
        const scale = 1 - p.degradation;
        const wf = p.wf * scale, wfd = p.wfd * scale;
        const kf = wfd / r0;
        const g0 = kf * d0[0], g1 = kf * d0[1], g2 = kf * d0[2];
        // f_stress = -voli volj (PK1_i + PK1_j) g
        const PKi = PK1i, PKj = W.PK1[j];
        const fs = this.scratch.fs;
        for (let a = 0; a < 3; a++) {
          const row = (PKi[3 * a] + PKj[3 * a]) * g0 + (PKi[3 * a + 1] + PKj[3 * a + 1]) * g1 + (PKi[3 * a + 2] + PKj[3 * a + 2]) * g2;
          fs[a] = -voli * volj * row;
        }
        const dxdv = dx[0] * dv[0] + dx[1] * dv[1] + dx[2] * dv[2];
        const delVdotDelR = limitMagnitude(dxdv / (r + 0.1 * h), 0.01 * P.c);
        const muIJ = h * delVdotDelR / (r + 0.1 * h);
        const viscMagnitude = (-P.q1 * P.c * muIJ + P.q2 * muIJ * muIJ) / P.rho0;
        const visc = rmass[i] * rmass[j] * viscMagnitude * wfd / (r + 1.0e-2 * h);
        // gamma = 0.5 (Fincr_i + Fincr_j) dx0 - dx
        const Fj = W.Fincr[j];
        for (let a = 0; a < 3; a++) {
          Fs[a] = 0.5 * ((Fi[3 * a] + Fj[3 * a]) * d0[0] + (Fi[3 * a + 1] + Fj[3 * a + 1]) * d0[1] + (Fi[3 * a + 2] + Fj[3 * a + 2]) * d0[2]) - dx[a];
        }
        const gg = Fs[0] * Fs[0] + Fs[1] * Fs[1] + Fs[2] * Fs[2];
        let hgErr = Math.sqrt(gg) / r0;
        W.hourglass[i] += volj * wf * hgErr;
        // hgCoef: the hourglass force is hgCoef dx (plastic branch: rmass_i rmass_j hg_mag wfd / (r + 0.01 h))
        let hgCoef: number;
        if (Math.max(epsPl[i], epsPl[j]) > 1.0e-3) {
          const delta = Fs[0] * dx[0] + Fs[1] * dx[1] + Fs[2] * dx[2];
          let hgMag: number;
          if (delVdotDelR * delta < 0.0) {
            hgErr = Math.max(hgErr, 0.05);
            hgMag = -hgErr * P.hgAmp * P.c * muIJ / P.rho0;
          } else {
            hgMag = 0.0;
          }
          hgCoef = rmass[i] * rmass[j] * hgMag * wfd / (r + 1.0e-2 * h);
        } else {
          const gdx = limitMagnitude(Fs[0] * dx[0] + Fs[1] * dx[1] + Fs[2] * dx[2], 0.1 * r);
          const delta = 0.5 * gdx / (r + 0.1 * h);
          let hgMag = P.hgAmp * delta / (r0Sq + 0.01 * h * h);
          hgMag *= -voli * volj * wf * P.E;
          hgCoef = hgMag / (r + 0.01 * h);
        }
        // f_visc = visc dx and f_hg = hgCoef dx (scaled by the damage of both atoms); sumForces = f_stress + f_visc + f_hg
        const dmg = (1 - damage[i]) * (1 - damage[j]);
        const cvis = visc + dmg * hgCoef;
        const sum0 = fs[0] + cvis * dx[0], sum1 = fs[1] + cvis * dx[1], sum2 = fs[2] + cvis * dx[2];
        f[3 * i] += sum0;
        f[3 * i + 1] += sum1;
        f[3 * i + 2] += sum2;
        desph[i] += 0.5 * (sum0 * dv[0] + sum1 * dv[1] + sum2 * dv[2]);
        // ev_tally_xyz with j owned (newton off): the pair virial is dx (x) sum
        v[0] += dx[0] * sum0; v[1] += dx[1] * sum1; v[2] += dx[2] * sum2;
        v[3] += dx[0] * sum1; v[4] += dx[0] * sum2; v[5] += dx[1] * sum2;
        // per-atom: half of dx (x) sum to each of i and the (owned) j (ev_tally_xyz, newton off)
        if (pc.vatom) {
          const va = pc.vatom;
          const w6 = [0.5 * dx[0] * sum0, 0.5 * dx[1] * sum1, 0.5 * dx[2] * sum2, 0.5 * dx[0] * sum1, 0.5 * dx[0] * sum2, 0.5 * dx[1] * sum2];
          for (let c = 0; c < 6; c++) { va[6 * i + c] += w6[c]; va[6 * j + c] += w6[c]; }
        }
        shepard += wf * volj;
        // failure model integration-point-wise (maximum plastic strain): the bond softens once both atoms are damaged
        if (P.failMaxPlastic !== null) {
          const strain1d = (r - r0) / r0;
          if (strain1d > 0.0) {
            if (damage[i] === 1.0 && damage[j] === 1.0) {
              if (p.energyPerBond === 0.0) {
                p.energyPerBond = Math.fround(strain1d);
              } else {
                const strain1dMax = p.energyPerBond;
                const softening = 2.0 * strain1dMax;
                p.degradation = Math.fround(strain1d > strain1dMax ? (strain1d - strain1dMax) / softening : 0.0);
              }
            }
            if (p.degradation >= 1.0) p.tag = 0;
          } else {
            p.degradation = 0.0;
          }
        }
      }
      if (shepard !== 0 && Math.abs(W.hourglass[i]) < 1.0e300) W.hourglass[i] /= shepard;
    }
  }
}
