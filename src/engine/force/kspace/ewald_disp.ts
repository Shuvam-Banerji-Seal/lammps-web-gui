import { KSpace, Pair, StyleError, type KSpaceCompute, type StyleContext } from '../types';
import type { SimState } from '../../types';
import type { Geometry } from '../../domain';
import { parseNum } from '../util';
import { KSpaceEwald, invertUpper } from './ewald';
import { PairLJLongCoulLong } from '../pair/lj_long';
import { erfcExact } from '../erfc';

/*
 * kspace_style ewald/disp — docs.lammps.org/kspace_style.html (source:
 * plans/lammps-docs/kspace_style.rst): "The *ewald/disp* style adds a long-range
 * dispersion sum option for :math:`1/r^6` potentials and is useful for
 * simulation of interfaces (Veld). It also performs standard Coulombic Ewald
 * summations, but in a more efficient manner than the *ewald* style. The
 * :math:`1/r^6` capability means that Lennard-Jones or Buckingham potentials can
 * be used without a cutoff, i.e. they become full long-range potentials."
 * kspace_modify.html: "gewald/disp value = rinv ... rinv = G-ewald parameter for
 * dispersion"; "mix/disp value = pair or geom or none ... With *geom*, geometric
 * mixing is enforced on the dispersion coefficients in the kspace coefficients."
 *
 * Dispersion (in 't Veld, Ismail, Grest 2007; textbook Ewald for r^-6). The pair
 * energy -C6/r^6 is split as -C6 g6(g r)/r^6 (real space, pair style) plus
 * -C6 (1 - g6(g r))/r^6 (here), with g6(x) = exp(-x^2)(1 + x^2 + x^4/2) and
 * 1 - g6(x) = int_0^1 t^5 exp(-x^2 t^2) x^6 dt. Its Fourier transform is
 * pi^(3/2) g^3 f(b)/3 with b = k/(2g) and f(b) = (1 - 2 b^2) exp(-b^2) +
 * 2 sqrt(pi) b^3 erfc(b). With Q(k) = sum_ij C_ij cos(k . r_ij) and the
 * half-space k sum (each vector stands for +k and -k):
 *   E_k    = -(pi^(3/2) g^3 / (3V)) sum_half f(b) Q(k)
 *   E_0    = -(pi^(3/2) g^3 / (6V)) sum_ij C_ij               (k = 0 term)
 *   E_self = (g^6 / 12) sum_i C_ii                             (removes i = j, n = 0)
 * The check against native LAMMPS (black box, lj units, fcc 256 atoms, cutoff 2.5,
 * gewald/disp 0.4): elong equals E_k + E_0 + E_self to 1e-15 (the default
 * pair_modify table/disp 12 changes only the real-space term, see pair_lj_long.ts).
 * Virial: W_ab = sum_half (G Q / V)(delta_ab + k_a k_b f'(b^2)/(2 g^2 f)) with
 * f' = d f / d(b^2) = 3 (sqrt(pi b^2) erfc(b) - exp(-b^2)), plus delta_ab E_0.
 * C_ij are the pair's C6 coefficients (lj4 = 4 eps sigma^6) for kspace
 * mix/disp pair (and none), or sqrt(C_ii C_jj) for mix/disp geom. The sum over
 * types is exact (structure factors per type), so no eigen-splitting is used.
 *
 * Coulomb: delegated to the ewald code (ewald.ts KSpaceEwald) when the pair style
 * has coulLong and a positive Coulomb cutoff.
 *
 * Not implemented (StyleError): the automatic dispersion G-ewald (native LAMMPS
 * chooses it from the accuracy by a rule the engine has not reproduced; measured
 * with native LAMMPS: G = 1.2494455 at accuracy 1e-3 and 1.6995757 at 1e-6 for
 * cutoff 2.5, independent of system size), kspace_modify splittol (native
 * truncates the eigen-split of the dispersion matrix), disp/auto yes, slab with
 * dispersion, and non-periodic boxes.
 * Convergence: the reciprocal dispersion sum is taken to b^2 = 36 (f(b) < 1e-15
 * beyond), not the k-space vector count native's automatic choice gives. Measured
 * with native LAMMPS: the native kmax depends on gewald/disp and accuracy, and
 * the two sums agree when native's own sum is converged (fcc probe).
 */

/** Converged reciprocal cutoff: b^2 = k^2 / (4 g^2) <= BMAX2. */
const BMAX2 = 36;

export class KSpaceEwaldDisp extends KSpace {
  readonly name: string = 'ewald/disp';
  /** Coulomb part (ewald.ts), used when the pair style has a Coulomb long-range term. */
  private readonly coulomb = new KSpaceEwald();
  private pair: Pair | null = null;
  private coulOn = false;
  private dispOn = false;
  private gDispUser = 0;
  private gDisp = 0;
  private mixDisp: 'pair' | 'geom' | 'none' = 'pair';
  private volume = 0;
  private nt = 0;
  private c6 = new Float64Array(0);
  /** Half-space dispersion vectors: k (3 per vector), G-factor and virial coefficient. */
  private kv = new Float64Array(0);
  private gf = new Float64Array(0);
  private vc = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('usage: kspace_style ewald/disp accuracy');
    this.accuracy = parseNum(args[0], 'accuracy');
    if (!(this.accuracy > 0)) throw new StyleError('kspace accuracy must be > 0');
    this.coulomb.settings(args);
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'gewald/disp': this.gDispUser = parseNum(values[0], 'gewald/disp'); return 1;
      case 'mix/disp': {
        const v = values[0];
        if (v !== 'pair' && v !== 'geom' && v !== 'none') throw new StyleError(`kspace_modify mix/disp: invalid value '${v}' (pair, geom or none)`);
        this.mixDisp = v;
        return 1;
      }
      case 'splittol': throw new StyleError('kspace_modify splittol is not implemented for kspace_style ewald/disp (the eigen-split of the dispersion matrix)');
      case 'disp/auto':
        if (values[0] === 'yes') throw new StyleError('kspace_modify disp/auto yes (automatic dispersion parameters) is not implemented');
        return 1;
      case 'mesh/disp': case 'order/disp': case 'force/disp/real': case 'force/disp/kspace':
        throw new StyleError(`kspace_modify ${key} is not implemented for kspace_style ewald/disp`);
      default:
        return this.coulomb.modify(key, values);
    }
  }

  linkPair(pair: Pair | null): void { this.pair = pair; }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    const pair = this.pair;
    this.dispOn = pair instanceof PairLJLongCoulLong && pair.ljLong;
    this.coulOn = !!pair?.coulLong && cutCoul > 0;
    if (!this.coulOn && !this.dispOn) {
      throw new StyleError('kspace_style ewald/disp needs a pair style with a long-range term (lj/long/coul/long with flag_lj long or flag_coul long, or coul/long)');
    }
    if (this.coulOn) {
      this.coulomb.init(s, geom, cutCoul, qqrd2e, ctx);
      this.gEwald = this.coulomb.gEwald;
    } else {
      this.gEwald = 0;
    }
    if (!this.dispOn) return;
    const p = pair as PairLJLongCoulLong;
    if (this.coulomb.slab !== 1) throw new StyleError('kspace_modify slab is not implemented for kspace_style ewald/disp with dispersion');
    const per = s.box.periodic;
    if (!per[0] || !per[1] || !per[2]) throw new StyleError('kspace_style ewald/disp needs a fully periodic box');
    if (!(this.gDispUser > 0)) {
      throw new StyleError('kspace_style ewald/disp: the automatic dispersion G-ewald is not implemented; set kspace_modify gewald/disp (native LAMMPS chooses it from the accuracy, a rule not reproduced by the engine)');
    }
    this.gDisp = this.gDispUser;
    p.gEwaldDisp = this.gDisp;
    // C6_ij from the pair (lj4 = 4 eps sigma^6), or the geometric rule for mix/disp geom
    this.nt = p.ntypes + 1;
    const nt = this.nt;
    this.c6 = Float64Array.from(p.lj4.subarray(0, nt * nt));
    if (this.mixDisp === 'geom') {
      for (let t = 1; t < nt; t++) for (let u = 1; u < nt; u++) {
        this.c6[t * nt + u] = Math.sqrt(this.c6[t * nt + t] * this.c6[u * nt + u]);
      }
    }
    this.volume = geom.volume(3);
    this.buildVectors(geom);
    ctx.log(`  G vector (dispersion, 1/distance) = ${this.gDisp}`);
  }

  /** Half-space vectors with b^2 <= BMAX2 for the current box (rows of H^-1 as in ewald.ts). */
  private buildVectors(geom: Geometry): void {
    const g = this.gDisp;
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, geom.lz);
    const kc = 2 * g * Math.sqrt(BMAX2);
    // |n_d| <= |k| |a_d| / (2 pi), with a_d the box edge vectors (columns of H)
    const alen = [geom.lx, Math.hypot(geom.xy, geom.ly), Math.hypot(geom.xz, geom.yz, geom.lz)];
    const nm = alen.map((a) => Math.floor(kc * a / (2 * Math.PI)) + 1);
    const kv: number[] = [], gf: number[] = [], vc: number[] = [];
    const pre = -(Math.PI ** 1.5) * g * g * g / 3;
    for (let nx = 0; nx <= nm[0]; nx++) {
      for (let ny = -nm[1]; ny <= nm[1]; ny++) {
        for (let nz = -nm[2]; nz <= nm[2]; nz++) {
          if (nx === 0 && (ny < 0 || (ny === 0 && nz <= 0))) continue;
          const kx = 2 * Math.PI * (nx * inv[0][0] + ny * inv[1][0] + nz * inv[2][0]);
          const ky = 2 * Math.PI * (nx * inv[0][1] + ny * inv[1][1] + nz * inv[2][1]);
          const kz = 2 * Math.PI * (nx * inv[0][2] + ny * inv[1][2] + nz * inv[2][2]);
          const u = (kx * kx + ky * ky + kz * kz) / (4 * g * g);
          if (u > BMAX2) continue;
          const { f, fu } = dispKernel(u);
          kv.push(kx, ky, kz);
          gf.push(pre * f);
          vc.push(fu / (2 * g * g * f));
        }
      }
    }
    this.kv = Float64Array.from(kv);
    this.gf = Float64Array.from(gf);
    this.vc = Float64Array.from(vc);
  }

  compute(kc: KSpaceCompute): void {
    if (this.coulOn) this.coulomb.compute(kc);
    if (this.dispOn) this.computeDispersion(kc);
  }

  private computeDispersion(kc: KSpaceCompute): void {
    const s = kc.s;
    const n = s.n;
    const g = this.gDisp;
    this.volume = kc.geom.volume(3);
    this.buildVectors(kc.geom);
    const V = this.volume;
    const nt = this.nt, c6 = this.c6;
    const type = s.type, x = s.x;
    const Nt = new Float64Array(nt);
    for (let i = 0; i < n; i++) Nt[type[i]] += 1;
    const cosi = new Float64Array(n), sini = new Float64Array(n);
    const Cs = new Float64Array(nt), Sn = new Float64Array(nt);
    const nvec = this.gf.length;
    let eSum = 0;
    const w = [0, 0, 0, 0, 0, 0];
    for (let v = 0; v < nvec; v++) {
      const kx = this.kv[3 * v], ky = this.kv[3 * v + 1], kz = this.kv[3 * v + 2];
      const G = this.gf[v];
      Cs.fill(0); Sn.fill(0);
      for (let i = 0; i < n; i++) {
        const ph = kx * x[3 * i] + ky * x[3 * i + 1] + kz * x[3 * i + 2];
        const c = Math.cos(ph), sn = Math.sin(ph);
        cosi[i] = c; sini[i] = sn;
        Cs[type[i]] += c; Sn[type[i]] += sn;
      }
      let Q = 0;
      for (let t = 1; t < nt; t++) for (let u = 1; u < nt; u++) {
        Q += c6[t * nt + u] * (Cs[t] * Cs[u] + Sn[t] * Sn[u]);
      }
      eSum += G * Q;
      const wq = G * Q * this.vc[v];
      w[0] += G * Q + wq * kx * kx; w[1] += G * Q + wq * ky * ky; w[2] += G * Q + wq * kz * kz;
      w[3] += wq * kx * ky; w[4] += wq * kx * kz; w[5] += wq * ky * kz;
      // F_m = (2 G / V) k sum_u C_{t_m u} [sin_m Cs_u - cos_m Sn_u]
      const pf = 2 * G / V;
      for (let m = 0; m < n; m++) {
        const tm = type[m];
        let T = 0;
        for (let u = 1; u < nt; u++) T += c6[tm * nt + u] * (sini[m] * Cs[u] - cosi[m] * Sn[u]);
        const fm = pf * T;
        kc.f[3 * m] += fm * kx; kc.f[3 * m + 1] += fm * ky; kc.f[3 * m + 2] += fm * kz;
      }
    }
    // k = 0 term and self term (kspace/ewald_disp header)
    let Q0 = 0;
    for (let t = 1; t < nt; t++) for (let u = 1; u < nt; u++) Q0 += c6[t * nt + u] * Nt[t] * Nt[u];
    const g3 = g * g * g;
    const E0 = -(Math.PI ** 1.5) * g3 * Q0 / (6 * V);
    let csum = 0;
    for (let t = 1; t < nt; t++) csum += Nt[t] * c6[t * nt + t];
    const Eself = (g ** 6 / 12) * csum;
    kc.acc.elong += eSum / V + E0 + Eself;
    const vfac = 1 / V;
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += w[c] * vfac;
    for (let c = 0; c < 3; c++) kc.acc.vlong[c] += E0;
  }
}

/** f(u) = (1 - 2u) e^-u + 2 sqrt(pi) u^(3/2) erfc(sqrt u) and df/du, u = b^2. */
const dispKernel = (u: number): { f: number; fu: number } => {
  const b = Math.sqrt(u);
  const e = Math.exp(-u);
  const erfc = erfcExact(b);
  const f = (1 - 2 * u) * e + 2 * Math.sqrt(Math.PI) * u * b * erfc;
  const fu = 3 * (Math.sqrt(Math.PI) * b * erfc - e);
  return { f, fu };
};
