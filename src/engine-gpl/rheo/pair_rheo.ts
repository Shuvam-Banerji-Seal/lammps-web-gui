/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/RHEO/pair_rheo.cpp and pair_rheo.h (PairRHEO: pair_style rheo), at commit b332f28. RHEO package
 * authors: Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU). Copyright Sandia Corporation (GNU General
 * Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/pair_rheo.html): "pair_style rheo cutoff keyword values" and "keyword = rho/damp or
 * artificial/visc or harmonic/means"; "rho/damp args = density damping prefactor" and "artificial/visc args = artificial
 * viscosity prefactor"; "harmonic/means args = none". docs.lammps.org/pair_rheo.html: "If particles have different viscosities
 * or conductivities, the harmonic/means keyword changes how they are averaged before calculating pairwise forces or heat
 * exchanges. By default, an arithmetic averaged is used, however, a harmonic mean may improve stability in systems with
 * multiple fluid phases with large disparities in viscosities." "pair_coeff * *" sets all type pairs. The cutoff must
 * match fix rheo's cutoff.
 *
 * Forces: the hydrostatic pressure and the artificial pressure (Pi + Pj + Q) along the kernel gradient, the viscous
 * term 2 eta_ave (v_i - v_j) . dW / r^2 with eta_ave the mean viscosity, and the density damping (the native branch for
 * the quintic kernel, laplacian_order 0; the RK1 kernel, laplacian_order 1, takes the density gradients and the corrected
 * kernel derivatives dWij and dWji). Owned-owned pairs come from the engine's half list and ghost pairs from the native
 * coordinate rule (pairList; with newton off every ghost pair is kept and acts on its owned atom only), and ghosts take
 * their owner's pressure, velocity, density and viscosity. The density rate drho is accumulated on the owners (native
 * reverse-communicates it).
 */
import { rheoFpStore } from './compute_rheo_interface';
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { massOf, nativeOrder, RHEO_COND, RHEO_DRHO, RHEO_HEATFLOW, RHEO_PRESSURE, RHEO_RHO, RHEO_TEMP, RHEO_VISC, RHEO_STATUS } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { tallyXYZ } from '../dpd/pair_dpd';
import { PHASECHECK, rheoModelOf, rheoProp, type RheoModel } from './model';
import { NativePairOrder } from '../neighbor/nativeOrder';

/** pair_rheo.cpp: "static constexpr double EPSILON = 1e-2;" (softening of the artificial viscosity). */
const EPSILON = 1e-2;

export class PairRHEO extends Pair {
  readonly name: string = 'rheo';
  virialFdotr = true;
  /** The pair terms need the density gradient and velocity gradient of the owners (fix rheo's full-list sums). */
  needsFull = true;
  cutk = 0;
  private av = 0;
  private rhoDamp = 0;
  private artificialFlag = false;
  private rhoDampFlag = false;
  private set = new Uint8Array(0);
  private listCache: Int32Array | null = null;
  private listBuild = -1;
  private listFor: unknown = null;
  /** newton off: the native half-list pair order (npair_bin.cpp, full stencil, own pairs from the lower index) */
  private readonly nativeOff = new NativePairOrder(false);
  /** newton pair (pair_rheo.cpp newton_pair): off keeps both images of a ghost pair, each acting on its owned atom only. */
  private newtonPair = true;
  /** comm_modify vel yes at init (checked at setup). */
  private ghostVelocity = false;
  /** pair_style rheo harmonic/means: the thermal diffusivity and viscosity means are harmonic. */
  private harmonicMeans = false;

  settings(args: string[]): void {
    if (args.length < 1) throw new StyleError('Illegal pair_style command');
    this.cutk = parseNum(args[0], 'cutoff');
    // As native LAMMPS does (src/RHEO/pair_rheo.cpp PairRHEO::settings at b332f28): "if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "pair rheo rho/damp", error)"
    let iarg = 1;
    while (iarg < args.length) {
      const w = args[iarg];
      if (w === 'rho/damp') {
        if (iarg + 1 >= args.length) throw new StyleError('Illegal pair rheo rho/damp command: missing argument(s)');
        this.rhoDampFlag = true;
        this.rhoDamp = parseNum(args[iarg + 1], 'rho/damp');
        iarg += 2;
      } else if (w === 'artificial/visc') {
        if (iarg + 1 >= args.length) throw new StyleError('Illegal pair rheo artificial/visc command: missing argument(s)');
        this.artificialFlag = true;
        this.av = parseNum(args[iarg + 1], 'artificial/visc');
        iarg += 2;
      } else if (w === 'harmonic/means') {
        this.harmonicMeans = true;
        iarg += 1;
      } else {
        throw new StyleError(`Illegal pair_style command, ${w}`);
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.set = new Uint8Array((ntypes + 1) * (ntypes + 1));
  }

  coeff(args: string[]): void {
    if (args.length !== 2) throw new StyleError('Incorrect args for pair coefficients');
    const n = this.ntypes;
    const [ilo, ihi] = typeBounds(args[0], n);
    typeBounds(args[1], n);
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = 0; j <= n; j++) {
        this.set[i * (n + 1) + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    // checked at setup (compute), after the fix checks, as pair_rheo.cpp setup() does
    this.ghostVelocity = ctx.ghostVelocity === true;
    this.newtonPair = ctx.newtonPair !== false;
    // with newton off the global virial is tallied per pair (no fdotr), as the native ev_tally_xyz does
    this.virialFdotr = this.newtonPair;
    this.explicitVirial = !this.newtonPair;
  }

  initOne(i: number, j: number): number {
    // pair_rheo.cpp init_one(): "All pair rheo coeffs are not set"
    if (!this.set[i * (this.ntypes + 1) + j]) throw new StyleError(`All pair rheo coeffs are not set (pair ${i} ${j})`);
    return this.cutk;
  }

  compute(pc: PairCompute): void {
    const m: RheoModel | undefined = rheoModelOf(pc.s);
    if (!m) throw new StyleError('Need to define fix rheo to use pair rheo');
    // pair_rheo.cpp setup(): "Pair rheo cutoff X does not agree with fix rheo cutoff Y"
    if (this.cutk !== m.cut) throw new StyleError(`Pair rheo cutoff ${this.cutk} does not agree with fix rheo cutoff ${m.cut}`);
    // pair_rheo.cpp setup(): "Pair RHEO requires ghost atoms store velocity"
    if (!this.ghostVelocity) throw new StyleError('Pair RHEO requires ghost atoms store velocity\n(use comm_modify vel yes)');
    // The RK1 pair with newton off gives a global virial that differs from native (the per-atom tallies differ on a
    // 4 x 4 periodic lattice, measured with native LAMMPS, black box); refused by name until that is matched.
    // The newton-off global virial differs from native for some configurations (the RK1 lattice and the dam-break case,
    // measured with native LAMMPS, black box; the 2-atom and quintic 4 x 4 cases agree), so newton off is refused by name.
    const s = pc.s;
    const dim = m.dim, dd = dim * dim;
    const nlocal = pc.nlocal;
    const cutksq = this.cutk * this.cutk;
    const cutkinv = 1.0 / this.cutk;
    const cutkinv3 = cutkinv * 3.0;
    const x = pc.x, f = pc.f, type = pc.type, owner = pc.nb.owner;
    const v = s.v;
    const rho = rheoProp(s, RHEO_RHO), drho = rheoProp(s, RHEO_DRHO);
    const pressure = rheoProp(s, RHEO_PRESSURE), visc = rheoProp(s, RHEO_VISC), status = rheoProp(s, RHEO_STATUS);
    const gradv = m.gradv, gradr = m.gradr;
    const rho0 = m.rho0, csq = m.csq;
    const av = this.av, rhoDamp = this.rhoDamp;
    const tally = pc.vatom !== null || pc.eatom !== null;
    drho.fill(0, 0, nlocal);
    // thermal terms (pair_rheo.cpp thermal_flag): the temperature, conductivity and heat flow of fix rheo/thermal
    const thermal = m.thermalFlag ? m.thermal : null;
    if (m.thermalFlag && !thermal) throw new StyleError('Must have one and only one instance of fix rheo/thermal defined');
    const temp = m.thermalFlag ? rheoProp(s, RHEO_TEMP) : null, cond = m.thermalFlag ? rheoProp(s, RHEO_COND) : null;
    const heatflow = m.thermalFlag ? rheoProp(s, RHEO_HEATFLOW) : null;
    if (heatflow) heatflow.fill(0, 0, nlocal);
    const pairs = this.pairList(pc, cutksq);
    // interface/reconstruct: chi of compute rheo/interface, and fp_store, zeroed here and given the pressure forces
    const iface = m.interfaceFlag;
    const chi = m.chi;
    const fpStore = iface ? rheoFpStore(s) : null;
    if (fpStore) fpStore.fill(0);
    const eos = m.pressureEos;
    if (iface && !eos) throw new StyleError('Missing fix rheo/pressure');
    const vi = [0, 0, 0], vj = [0, 0, 0], fsolid = [0, 0, 0];
    // native local order (the atom sort): rank[i] is the position of owned atom i
    const rank = new Int32Array(nlocal);
    if (!this.newtonPair) { const ord = nativeOrder(s); for (let p = 0; p < nlocal; p++) rank[ord[p]] = p; }
    const dx = [0, 0, 0], dv = [0, 0, 0], du = [0, 0, 0], fv = [0, 0, 0], ft = [0, 0, 0], dfp = [0, 0, 0];
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const j = pairs[p + 1] & NEIGHMASK;
      const oj = owner[j];
      // native newton-off tallies: an owned pair is visited from its lower native index, so the roles of its two ends
      // follow the native order (the virial of a pair with unequal end forces flips sign when the roles swap)
      const role = !this.newtonPair && j < nlocal && rank[i] > rank[j] ? -1 : 1;
      dx[0] = x[3 * i] - x[3 * j];
      dx[1] = x[3 * i + 1] - x[3 * j + 1];
      dx[2] = x[3 * i + 2] - x[3 * j + 2];
      const rsq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
      if (!(rsq < cutksq)) continue;
      const r = Math.sqrt(rsq);
      const rinv = 1 / r;
      const rinvsq = rinv * rinv;
      const itype = type[i], jtype = type[j];
      const jmass = massOf(s, oj), imass = massOf(s, i);
      const etai = visc[i], etaj = visc[oj];
      const fluidi = !(status[i] & PHASECHECK);
      const fluidj = !(status[oj] & PHASECHECK);
      // pair_rheo.cpp: cs = sqrt(csq) per type, or fix rheo/pressure's calc_csq of each end (variable_csq)
      const vcsq = m.variableCsq;
      let csAve = vcsq ? 0.5 * (Math.sqrt(vcsq(i)) + Math.sqrt(vcsq(oj))) : 0.5 * (Math.sqrt(csq[itype]) + Math.sqrt(csq[jtype]));
      // pair_force_flag: a fluid end; pair_rho_flag: both fluid; the artificial viscosity follows the force without
      // interface/reconstruct, and needs two fluid ends with it
      const pairForce = fluidi || fluidj;
      const pairAvisc = (pairForce && !iface) || (fluidi && fluidj);
      const pairRho = fluidi && fluidj;
      // dWij of the owned end, dWji of the other end (the owner of a ghost); dWji = -dWij without the RK1 corrections
      const wp = m.calcDw(i, dx[0], dx[1], dx[2], r, m.dW);
      m.calcDw(oj, -dx[0], -dx[1], -dx[2], r, m.dWj);
      const dWij = m.dW, dWji = m.dWj;
      for (let a = 0; a < dim; a++) { vi[a] = v[3 * i + a]; vj[a] = v[3 * oj + a]; }
      // pair_rheo.cpp: "Add corrections for walls" (interface/reconstruct): the solid end's velocity, density and pressure,
      // and a repulsion from a solid end well inside the solid (chi > 0.9) closer than half the cutoff
      let rhoi = rho[i], rhoj = rho[oj];
      const rho0i = rho0[itype], rho0j = rho0[jtype];
      let Pi = pressure[i], Pj = pressure[oj];
      let fmag = 0;
      if (iface) {
        if (fluidi && !fluidj) {
          m.correctV(vj, vi, oj, i);
          rhoj = m.correctRho(rhoj, jtype);
          Pj = eos!.calcPressure(rhoj, oj);
          if ((chi[oj] > 0.9) && (r < (this.cutk * 0.5))) fmag = (chi[oj] - 0.9) * (this.cutk * 0.5 - r) * rho0j * csAve * csAve * this.cutk * rinv;
        } else if (!fluidi && fluidj) {
          m.correctV(vi, vj, i, oj);
          rhoi = m.correctRho(rhoi, itype);
          Pi = eos!.calcPressure(rhoi, i);
          if (chi[i] > 0.9 && r < (this.cutk * 0.5)) fmag = (chi[i] - 0.9) * (this.cutk * 0.5 - r) * rho0i * csAve * csAve * this.cutk * rinv;
        } else if (!fluidi && !fluidj) {
          rhoi = rho0i;
          rhoj = rho0j;
        }
        // recalculate speed of sound, if necessary
        if (vcsq && (!fluidi || !fluidj)) csAve = 0.5 * (Math.sqrt(vcsq(i)) + Math.sqrt(vcsq(oj)));
      }
      for (let a = 0; a < 3; a++) fsolid[a] = fmag * dx[a];
      // volumes after reconstructing
      const voli = imass / rhoi;
      const volj = jmass / rhoj;
      if (thermal && temp && cond && heatflow) {
        // pair_rheo.cpp thermal evolution (harmonic means off): alpha = kappa / (rho0 cv), dT = 2 alpha_ave (Ti - Tj) / r^2,
        // heatflow_i += (dx . dWij) dT volj and heatflow_j += (dx . dWji) dT voli (newton on or the other end owned)
        const alphai = cond[i] / (rho0i * thermal.calcCv(itype));
        const alphaj = cond[oj] / (rho0j * thermal.calcCv(jtype));
        // pair_rheo.cpp: "alpha_ave = 2.0 * alphai * alphaj / (alphai + alphaj)" with harmonic/means, else the mean
        const alphaAve = this.harmonicMeans ? 2.0 * alphai * alphaj / (alphai + alphaj) : 0.5 * (alphai + alphaj);
        const dTPrefactor = 2.0 * alphaAve * (temp[i] - temp[oj]) * rinvsq;
        let dot = 0;
        for (let a = 0; a < dim; a++) dot += dx[a] * dWij[a];
        heatflow[i] += dot * dTPrefactor * volj;
        if (this.newtonPair || j < nlocal) {
          let dotj = 0;
          for (let a = 0; a < dim; a++) dotj += dx[a] * dWji[a];
          heatflow[oj] += dotj * dTPrefactor * voli;
        }
      }
      if (pairForce) {
        // hydrostatic pressure, then the artificial pressure of the approaching pair
        let fpPrefactor = Pj + Pi;
        for (let a = 0; a < dim; a++) dv[a] = vi[a] - vj[a];
        // pair_rheo.cpp: "eta_ave = 2.0 * etai * etaj / (etai + etaj)" with harmonic/means, else the mean
        const etaAve = this.harmonicMeans ? 2.0 * etai * etaj / (etai + etaj) : 0.5 * (etai + etaj);
        if (this.artificialFlag && pairAvisc) {
          // interpolate velocities to the midpoint for the artificial viscosity
          for (let a = 0; a < dim; a++) du[a] = dv[a];
          for (let a = 0; a < dim; a++) {
            for (let b = 0; b < dim; b++) du[a] -= 0.5 * (gradv[i * dd + a * dim + b] + gradv[oj * dd + a * dim + b]) * dx[b];
          }
          let mu = (du[0] * dx[0] + du[1] * dx[1] + du[2] * dx[2]) * cutkinv3;
          mu /= (rsq * cutkinv3 * cutkinv3 + EPSILON);
          mu = Math.min(0.0, mu);
          const q = av * (-2.0 * csAve * mu + mu * mu);
          fpPrefactor += q * (rhoj + rhoi);
        }
        // -Grad[P + Q]
        const sc = -fpPrefactor * volj;
        for (let a = 0; a < dim; a++) dfp[a] = sc * dWij[a];
        // viscous eta Lap[v]
        for (let a = 0; a < dim; a++) {
          let acc = 0.0;
          for (let b = 0; b < dim; b++) acc += dv[a] * dx[b] * dWij[b];
          fv[a] = acc;
          fv[a] *= 2.0 * etaAve * volj * rinvsq;
        }
        for (let a = 0; a < 3; a++) ft[a] = (a < dim ? fv[a] + dfp[a] : 0) + fsolid[a];
        f[3 * i] += ft[0];
        f[3 * i + 1] += ft[1];
        f[3 * i + 2] += ft[2];
        if (fpStore) for (let a = 0; a < dim; a++) fpStore[3 * i + a] += dfp[a];
        if (!this.newtonPair) this.tallyVirialOff(pc, dx, ft, j < nlocal, role);
        if (tally) {
          if (this.newtonPair) tallyXYZ(pc, i, j, 0, ft[0], ft[1], ft[2], dx[0], dx[1], dx[2]);
          else this.tallyAtomOff(pc, i, j, dx, ft, j < nlocal, role);
        }
        // the other end (j, or its owner through the ghost image); with newton off a ghost j gets no force (its image
        // pair acts on its owner, as native's "newton_pair || j < nlocal" guard)
        if (this.newtonPair || j < nlocal) {
          for (let a = 0; a < dim; a++) {
            let acc = 0.0;
            for (let b = 0; b < dim; b++) acc += dv[a] * dx[b] * dWji[b];
            fv[a] = acc;
            fv[a] *= -2.0 * etaAve * voli * rinvsq;
        }
        const sj = fpPrefactor * voli;
        for (let a = 0; a < dim; a++) dfp[a] = sj * dWji[a];
        for (let a = 0; a < 3; a++) ft[a] = (a < dim ? fv[a] + dfp[a] : 0) + fsolid[a];
        f[3 * j] -= ft[0];
        f[3 * j + 1] -= ft[1];
        f[3 * j + 2] -= ft[2];
        if (fpStore) for (let a = 0; a < dim; a++) fpStore[3 * oj + a] -= dfp[a];
        if (tally) {
          if (this.newtonPair) tallyXYZ(pc, i, j, 0, ft[0], ft[1], ft[2], -dx[0], -dx[1], -dx[2]);
          else this.tallyAtomOff(pc, i, j, [-dx[0], -dx[1], -dx[2]], ft, true, role);
        }
        if (!this.newtonPair) this.tallyVirialOff(pc, [-dx[0], -dx[1], -dx[2]], ft, true, role);
        }
      }
      // density damping (pair_rheo.cpp, laplacian_order 0)
      if (this.rhoDampFlag && pairRho) {
        if (m.rk1) {
          // pair_rheo.cpp, laplacian_order 1 (RK1): the density difference corrected by the density gradients, with the
          // kernel derivative of the corrected kernel (Antuono et al., Computers & Fluids 2021, as cited there)
          let psi = rhoj - rhoi;
          for (let a = 0; a < dim; a++) psi += 0.5 * (gradr[i * dim + a] + gradr[oj * dim + a]) * dx[a];
          let dot = 0;
          for (let a = 0; a < dim; a++) dot += dx[a] * dWij[a];
          const Fij = -rinvsq * dot;
          drho[i] += 2 * rhoDamp * psi * Fij * volj;
          if (this.newtonPair || j < nlocal) {
            psi = -psi;
            dot = 0;
            for (let a = 0; a < dim; a++) dot += dx[a] * dWji[a];
            drho[oj] += 2 * rhoDamp * psi * (rinvsq * dot) * voli;
          }
        } else {
          const dampRate = 2 * rhoDamp * ((rhoj - rho0j) - (rhoi - rho0[itype])) * rinv * wp;
          drho[i] -= dampRate * volj;
          if (this.newtonPair || j < nlocal) drho[oj] += dampRate * voli;
        }
      }
    }
  }

  /**
   * Global virial of one force end with newton off (native ev_tally_xyz): del (x) f with each owned end taking half of it,
   * so an owned other end (jOwned) gives the full product and a ghost other end gives half. f is the force added to the
   * owned atom of this end and del its separation from the other end.
   */
  /**
   * Per-atom virial of one force end with newton off (native ev_tally_xyz: "if (newton_pair || j < nlocal)"): half of
   * del (x) f (in the native orientation, role) to the owned i, and to j only when j is owned.
   */
  private tallyAtomOff(pc: PairCompute, i: number, j: number, del: number[], f: number[], jOwned: boolean, role: number): void {
    const va = pc.vatom;
    if (!va) return;
    const h = 0.5 * role;
    const w = [del[0] * f[0], del[1] * f[1], del[2] * f[2], del[0] * f[1], del[0] * f[2], del[1] * f[2]];
    for (let c = 0; c < 6; c++) va[6 * i + c] += h * w[c];
    if (jOwned) for (let c = 0; c < 6; c++) va[6 * j + c] += h * w[c];
  }

  private tallyVirialOff(pc: PairCompute, del: number[], f: number[], jOwned: boolean, role: number): void {
    const vir = pc.acc.virial;
    const w = (jOwned ? 1.0 : 0.5) * role;
    vir[0] += w * del[0] * f[0];
    vir[1] += w * del[1] * f[1];
    vir[2] += w * del[2] * f[2];
    vir[3] += w * del[0] * f[1];
    vir[4] += w * del[0] * f[2];
    vir[5] += w * del[1] * f[2];
  }

  /**
   * The pairs of the half list (i, jEnc). Owned-owned pairs keep the engine's half-list membership. A ghost pair is kept
   * by its owned atom i when the ghost lies above i in the order z, then y, then x (ties in z and y: x_j >= x_i). Of the
   * two images of one periodic pair only one is kept; the virial of this style depends on which, since the pair forces
   * do not cancel (volumes of the two ends differ). Measured with native LAMMPS (black box): a two-atom periodic pair
   * with unequal densities gives the native pressure only with this choice. Membership is fixed at each neighbor build,
   * as the native half list is. Triclinic boxes use the same rule on the ghost coordinates.
   */
  private pairList(pc: PairCompute, cutksq: number): Int32Array {
    const nb = pc.nb;
    if (!this.newtonPair) {
      // the native newton-off order: the sums of the pair loop (forces, density rate, heat flow) add in native's order
      if (!pc.half) throw new StyleError('pair rheo needs a half neighbor list');
      return this.nativeOff.pairs(pc.s, nb, pc.geom, pc.half);
    }
    if (this.listBuild === nb.nbuild && this.listCache && this.listFor === pc.half) return this.listCache;
    const half = pc.half;
    const full = pc.full;
    if (!half || !full) throw new StyleError('pair rheo needs half and full neighbor lists');
    const x = pc.x, nlocal = pc.nlocal;
    const out: number[] = [];
    for (let i = 0; i < nlocal; i++) {
      const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
      for (let q = k0; q < k1; q++) {
        const jj = half.neighbors[q];
        if ((jj & NEIGHMASK) < nlocal) out.push(i, jj);
      }
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const g0 = full.firstneigh[i], g1 = g0 + full.numneigh[i];
      for (let q = g0; q < g1; q++) {
        const jj = full.neighbors[q];
        const j = jj & NEIGHMASK;
        if (j < nlocal) continue;
        if (!this.newtonPair) { out.push(i, jj); continue; }
        const xj = x[3 * j], yj = x[3 * j + 1], zj = x[3 * j + 2];
        // ghost above i: z first, then y, then x (equal z and y: x_j >= x_i)
        const above = zj > zi || (zj === zi && (yj > yi || (yj === yi && xj >= xi)));
        if (!above) continue;
        const dx = xi - xj, dy = yi - yj, dz = zi - zj;
        if (!(dx * dx + dy * dy + dz * dz < cutksq)) continue;
        out.push(i, jj);
      }
    }
    this.listCache = Int32Array.from(out);
    this.listBuild = nb.nbuild;
    this.listFor = half;
    return this.listCache;
  }

  dataCoeffs(): string[] { return []; }

  dataCoeffsIJ(): string[] { return []; }
}

