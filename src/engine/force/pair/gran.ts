import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../types';
import type { SimState } from '../../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style gran/hooke, gran/hooke/history, gran/hertz/history —
 * docs.lammps.org/pair_gran.html:
 *   pair_style style Kn Kt gamma_n gamma_t xmu dampflag keyword
 * "The *gran* styles use the following formulas for the frictional force
 * between two granular particles ... when the distance r between two
 * particles of radii Ri and Rj is less than their contact distance d = Ri +
 * Rj.  There is no force between the particles when r > d."
 *   F_{hk} = (k_n \delta \mathbf{n}_{ij} - m_{eff} \gamma_n\mathbf{ v}_n) -
 *            (k_t \boldsymbol{\Delta} \mathbf{s}_t + m_{eff} \gamma_t \mathbf{v}_t)
 *   F_{hz} = \sqrt{\delta} \sqrt{\frac{R_i R_j}{R_i + R_j}} F_{hk}
 * "The shear force is a "history" effect that accounts for the tangential
 * displacement between the particles for the duration of the time they are
 * in contact.  This term is included in pair styles *hooke/history* and
 * *hertz/history*, but is not included in pair style *hooke*\ .  The
 * tangential damping force term is included in all three pair styles if
 * *dampflag* is set to 1; it is not included if *dampflag* is set to 0."
 * "If a NULL is used for :math:`K_t`, then a default value is used where
 * :math:`K_t = 2/7 K_n`.  If a NULL is used for :math:`\gamma_t`, then a
 * default value is used where :math:`\gamma_t = 1/2 \gamma_n`."
 * xmu "is the upper limit of the tangential force through the Coulomb
 * criterion Ft = xmu\*Fn".
 *
 * Measured with native LAMMPS (black box, two spheres of different radius
 * and density with set velocities and angular velocities; forces and
 * torques to 15 digits):
 * - with n = (x_i - x_j)/r, the tangential relative velocity is
 *   v_t = v_r - (v_r.n) n - (R_i w_i + R_j w_j) x n, v_r = v_i - v_j;
 * - the tangential force is capped at xmu |F_n| by scaling;
 * - torques: tau_i = -R_i n x F_t and tau_j = -R_j n x F_t (F_t on i);
 * - limit_damping zeroes the whole contact (normal, tangential and torque)
 *   when the normal force is attractive;
 * - without comm_modify vel yes native stops at init (ghost atoms must
 *   store velocities);
 * - with fix freeze on one particle of a contact, m_eff is the other
 *   particle's mass (frozen sphere at rest against a moving one: 200 +
 *   m_2 gamma_n v_n, not the reduced mass); pair_gran.html says the same for
 *   rigid bodies ("its mass is replaced by the mass of the rigid body").
 * Shear history (hooke/history, hertz/history): the tangential displacement
 * accumulates v_t dt in each timestep (not at run setup), is kept in the
 * tangent plane, enters F_t as -k_t s, and is reset when the particles lose
 * contact; at the Coulomb cap it is rescaled so that k_t s + m_eff gamma_t
 * v_t matches the capped force (the Cundall-Strack sliding rule). Measured:
 * when the cap is hit with no stored displacement (contacts at run setup),
 * native drops the tangential force altogether.
 * The pair energy is zero: "energy is not conserved in these dissipative
 * potentials".
 */

type GranKind = 'gran/hooke' | 'gran/hooke/history' | 'gran/hertz/history';

export class PairGran extends Pair {
  readonly name: GranKind;
  virialFdotr = true;
  kn = 0; kt = 0; gn = 0; gt = 0; xmu = 0; dampflag = 1;
  limitDamping = false;
  private readonly history: boolean;
  private readonly hertz: boolean;
  private set = new Uint8Array(0);
  /** Contact shear displacement per atom-ID pair "lo:hi", oriented as seen from the lower ID. */
  private shear = new Map<string, Float64Array>();

  constructor(kind: GranKind) {
    super();
    this.name = kind;
    this.history = kind !== 'gran/hooke';
    this.hertz = kind === 'gran/hertz/history';
  }

  settings(args: string[]): void {
    if (args.length < 6 || args.length > 7) throw new StyleError(`usage: pair_style ${this.name} Kn Kt gamma_n gamma_t xmu dampflag [limit_damping]`);
    this.kn = parseNum(args[0], 'Kn');
    this.kt = args[1] === 'NULL' ? (this.kn * 2) / 7 : parseNum(args[1], 'Kt');
    this.gn = parseNum(args[2], 'gamma_n');
    this.gt = args[3] === 'NULL' ? 0.5 * this.gn : parseNum(args[3], 'gamma_t');
    this.xmu = parseNum(args[4], 'xmu');
    const d = args[5];
    if (d !== '0' && d !== '1') throw new StyleError(`pair_style ${this.name}: dampflag must be 0 or 1, got '${d}'`);
    this.dampflag = Number(d);
    if (args.length === 7) {
      if (args[6] !== 'limit_damping') throw new StyleError(`pair_style ${this.name}: unknown keyword '${args[6]}' (only limit_damping)`);
      this.limitDamping = true;
    }
    if (this.kn < 0 || this.kt < 0 || this.gn < 0 || this.gt < 0 || this.xmu < 0 || this.xmu > 10000) {
      throw new StyleError(`Illegal pair_style ${this.name} command: Kn, Kt, gamma_n, gamma_t must be >= 0 and xmu in 0..1e4`);
    }
    if (this.dampflag === 0) this.gt = 0;
    this.shear.clear();
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.set = new Uint8Array((ntypes + 1) * (ntypes + 1));
  }

  coeff(args: string[]): void {
    // "For granular styles there are no additional coefficients to set for each pair of atom types"
    if (args.length !== 2) throw new StyleError(`usage: pair_coeff I J (pair style ${this.name} takes no coefficients)`);
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const nt = this.ntypes + 1;
    for (let i = ilo; i <= ihi; i++) for (let j = jlo; j <= jhi; j++) this.set[i * nt + j] = this.set[j * nt + i] = 1;
  }

  /** Group bit of fix freeze (StyleContext.freezeGroupBit), 0 when none. */
  private freezeBit = 0;

  initStyle(ctx: StyleContext): void {
    this.state = ctx.s;
    this.freezeBit = ctx.freezeGroupBit ?? 0;
    if (!ctx.ghostVelocity) throw new StyleError('Pair gran/h* requires ghost atoms store velocity (use comm_modify vel yes)');
    const s = ctx.s;
    if (s && (!s.radius || !s.rmass || !s.omega)) throw new StyleError(`pair_style ${this.name} requires atom_style sphere (radius, rmass, omega)`);
  }

  /** Neighbor cutoff for a type pair: the sum of the largest radii of the two types. */
  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.set[i * nt + j]) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    return this.maxRadius(i) + this.maxRadius(j);
  }

  private state: SimState | null = null;

  private maxRadius(t: number): number {
    const s = this.state;
    if (!s || !s.radius) return 0;
    let m = 0;
    for (let k = 0; k < s.n; k++) if (s.type[k] === t && s.radius[k] > m) m = s.radius[k];
    return m;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const list = pc.half!;
    const { x, f } = pc;
    const nb = pc.nb;
    const nall = pc.nall;
    const owner = nb.owner;
    const radius = s.radius!, rmass = s.rmass!, omega = s.omega!, v = s.v, id = s.id;
    const dt = s.dt;
    const update = pc.historyUpdate === true;
    const tq = new Float64Array(3 * nall);
    const seen = this.history ? new Set<string>() : null;
    const { kn, kt, gn, gt, xmu } = this;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const ri = radius[oi], mi = rmass[oi];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const oj = owner[j];
        const rj = radius[oj];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const radsum = ri + rj;
        let key = '', flip = false;
        if (this.history) {
          const a = id[oi], b = id[oj];
          flip = a > b;
          key = flip ? `${b}:${a}` : `${a}:${b}`;
        }
        if (rsq >= radsum * radsum) {
          if (this.history) this.shear.delete(key);
          continue;
        }
        const r = Math.sqrt(rsq), rinv = 1 / r, rsqinv = 1 / rsq;
        const mj = rmass[oj];
        // relative translational velocity and its normal part
        const vrx = v[3 * oi] - v[3 * oj], vry = v[3 * oi + 1] - v[3 * oj + 1], vrz = v[3 * oi + 2] - v[3 * oj + 2];
        const vnnr = vrx * dx + vry * dy + vrz * dz;
        const vnx = dx * vnnr * rsqinv, vny = dy * vnnr * rsqinv, vnz = dz * vnnr * rsqinv;
        // relative rotational velocity (R_i w_i + R_j w_j) x n
        const wx = (ri * omega[3 * oi] + rj * omega[3 * oj]) * rinv;
        const wy = (ri * omega[3 * oi + 1] + rj * omega[3 * oj + 1]) * rinv;
        const wz = (ri * omega[3 * oi + 2] + rj * omega[3 * oj + 2]) * rinv;
        const vtx = vrx - vnx - (wy * dz - wz * dy);
        const vty = vry - vny - (wz * dx - wx * dz);
        const vtz = vrz - vnz - (wx * dy - wy * dx);
        // measured: with fix freeze on one of the two particles, m_eff is the other particle's mass
        let meff = (mi * mj) / (mi + mj);
        if (this.freezeBit) {
          if (s.mask[oi] & this.freezeBit) meff = mj;
          else if (s.mask[oj] & this.freezeBit) meff = mi;
        }
        const poly = this.hertz ? Math.sqrt(((radsum - r) * ri * rj) / radsum) : 1;
        // normal force F_n = ccel * (dx, dy, dz)
        const ccel = (kn * (radsum - r) * rinv - meff * gn * vnnr * rsqinv) * poly;
        if (this.limitDamping && ccel < 0) {
          if (this.history) this.shear.delete(key);
          continue;
        }
        // tangential force
        let sh: Float64Array | null = null;
        let shx = 0, shy = 0, shz = 0;
        if (this.history) {
          seen!.add(key);
          sh = this.shear.get(key) ?? new Float64Array(3);
          const sg = flip ? -1 : 1;
          shx = sg * sh[0]; shy = sg * sh[1]; shz = sg * sh[2];
          if (update) {
            shx += vtx * dt; shy += vty * dt; shz += vtz * dt;
            // keep the displacement in the tangent plane
            const rsht = (shx * dx + shy * dy + shz * dz) * rsqinv;
            shx -= rsht * dx; shy -= rsht * dy; shz -= rsht * dz;
          }
        }
        let fsx = -(kt * shx + meff * gt * vtx) * poly;
        let fsy = -(kt * shy + meff * gt * vty) * poly;
        let fsz = -(kt * shz + meff * gt * vtz) * poly;
        const fs = Math.sqrt(fsx * fsx + fsy * fsy + fsz * fsz);
        const fn = xmu * Math.abs(ccel * r);
        if (fs > fn) {
          let scale = fs > 0 ? fn / fs : 0;
          if (this.history) {
            // measured: with no stored shear (a contact at run setup) native drops the capped
            // tangential force entirely instead of scaling it
            if (shx === 0 && shy === 0 && shz === 0) scale = 0;
            else if (kt > 0) {
              const c = (meff * gt) / kt;
              shx = scale * (shx + c * vtx) - c * vtx;
              shy = scale * (shy + c * vty) - c * vty;
              shz = scale * (shz + c * vtz) - c * vtz;
            }
          }
          fsx *= scale; fsy *= scale; fsz *= scale;
        }
        if (this.history) {
          const sg = flip ? -1 : 1;
          sh![0] = sg * shx; sh![1] = sg * shy; sh![2] = sg * shz;
          this.shear.set(key, sh!);
        }
        const fx = ccel * dx + fsx, fy = ccel * dy + fsy, fz = ccel * dz + fsz;
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        // torques: -R n x F_t on both particles
        const tx = (dy * fsz - dz * fsy) * rinv, ty = (dz * fsx - dx * fsz) * rinv, tz = (dx * fsy - dy * fsx) * rinv;
        tq[3 * i] -= ri * tx; tq[3 * i + 1] -= ri * ty; tq[3 * i + 2] -= ri * tz;
        tq[3 * j] -= rj * tx; tq[3 * j + 1] -= rj * ty; tq[3 * j + 2] -= rj * tz;
      }
    }
    if (this.history) for (const key of this.shear.keys()) if (!seen!.has(key)) this.shear.delete(key);
    nb.reverseSum(tq, 3, s.torque!);
  }

  dataCoeffs(): string[] | null { return null; }
  dataCoeffsIJ(): string[] | null { return null; }
}
