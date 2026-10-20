/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: the RHEO package's fix_rheo.cpp / fix_rheo.h (FixRHEO, the shared state, time integration), its
 * compute_rheo_kernel.cpp (the quintic kernel and its derivative), compute_rheo_grad.cpp (velocity and density
 * gradients) and compute_rheo_vshift.cpp (the shifting velocity), at commit b332f28. RHEO package authors: Joel Clemmer
 * (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU). Copyright Sandia Corporation (GNU General Public License). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The shared state of the RHEO styles: the kernel, the owned-atom gradients and shift velocities, and the per-type
 * reference density and sound speed. Pair rheo, fix rheo/viscosity and fix rheo/pressure reach it through
 * rheoModelOf(s). The per-atom density, status, pressure, viscosity and density rate are internal atom properties
 * (src/engine/atoms.ts RHEO_*); a ghost atom reads its owner's values (nb.owner), as the other pair styles do.
 *
 * Gradients and shift velocities are summed over the full neighbor list of each owned atom (the native code sums
 * over the half list and reverse-communicates the ghost contributions; both give the same pair terms).
 */
import type { SimState } from '../../engine/types';
import type { Neighbor } from '../../engine/neighbor';
import { NEIGHMASK } from '../../engine/neighbor';
import { massOf, RHEO_ESPH, RHEO_RHO, RHEO_STATUS } from '../../engine/atoms';
import type { Geometry } from '../../engine/domain';
import { NativePairOrder } from '../neighbor/nativeOrder';

/** fix_rheo.h, RHEO_NS: "PHASECHECK = 0x00000003" (solid or future phase bits) and the temporary option bits. */
export const PHASECHECK = 0x3;
export const STATUS_SOLID = 1 << 0;
export const STATUS_NO_SHIFT = 1 << 6;
export const STATUS_NO_INTEGRATION = 1 << 7;
/** fix_rheo.h: surface status bits (set by the surface detection). */
export const STATUS_SURFACE = 1 << 4;
export const STATUS_SPLASH = 1 << 5;
/** fix_rheo.h: "OPTIONSMASK = 0xFFFFFC3F" (clears the temporary options, bits 6 to 9). */
export const OPTIONSMASK = 0xFFFFFC3F;
/** fix_rheo.h: "STATUS_FREEZING = 1 << 8", "STATUS_MELTING = 1 << 9" and "PHASEMASK = 0xFFFFFFFC" (clears bits 0 and 1). */
export const STATUS_FREEZING = 1 << 8;
export const STATUS_MELTING = 1 << 9;
export const PHASEMASK = 0xFFFFFFFC;

/**
 * Internal per-atom property: the number of formed shell bonds of each atom (native fix property/atom i_shell_nbond,
 * named "shell_nbond" by bond_rheo_shell.cpp). Created on first use, grown with the atoms like the other custom props.
 */
export const SHELL_NBOND = '#rheoshellnbond';
export const shellNbond = (s: SimState): Float64Array => {
  let c = s.custom.get(SHELL_NBOND);
  if (!c) {
    c = { int: true, cols: 0, data: new Float64Array(s.n) };
    s.custom.set(SHELL_NBOND, c);
  }
  return c.data;
};

/** The thermal part of fix rheo/thermal as the other RHEO styles use it (cv per type, see rheo/thermal.ts). */
export interface RheoThermal {
  calcCv(itype: number): number;
}

/** Shared RHEO state for one simulation (one fix rheo). */
/** math_const.h THIRD */
const THIRD = 1.0 / 3.0;

export class RheoModel {
  readonly dim: 2 | 3;
  /** fix rheo cutoff and the derived kernel constants (compute_rheo_kernel.cpp init()). */
  readonly cut: number;
  readonly cutsq: number;
  readonly cutinv: number;
  preW: number;
  preWp: number;
  /** fix rheo kernel style (compute_rheo_kernel.cpp kernel_style). */
  kernelStyle: 'quintic' | 'wendland/c4' | 'RK0' | 'RK1' | 'RK2' = 'quintic';
  /**
   * The reproducing-kernel correction order: -1 for quintic and wendland/c4, 0 for RK0, 1 for RK1, 2 for RK2. RK0's
   * correction changes only the kernel values that no caller reads, so it behaves as the quintic kernel; RK1 and RK2
   * correct the kernel gradients (their moment matrices have Mdim = 1 + dim and 1 + dim + dim (dim + 1) / 2 rows).
   */
  correctionOrder = -1;
  /** Equilibrium density and squared sound speed per type (index 0 unused). */
  readonly rho0: Float64Array;
  readonly csq: Float64Array;
  /** fix rheo shift: 1 for the types that shift, 0 for "exclude/type"; null without the shift keyword. */
  shiftType: Uint8Array | null = null;
  /** fix rheo shift: the shifting velocity is on. */
  shiftFlag = false;
  /**
   * fix rheo shift scale/cross/type scale cmin rmin (native shift_cross_type_flag, shift_scale, shift_cmin,
   * shift_rmin): the shift normal to an interface between atom types is scaled by scale for atoms whose cross-type
   * colour ct reaches cmin, and an atom whose same-type kernel-weighted distance sum is below rmin does not shift.
   */
  shiftCrossType = false;
  shiftScale = 0;
  shiftCmin = 0;
  shiftRmin = 0;
  /**
   * fix rheo interface/reconstruct (compute_rheo_interface.cpp): chi per owned atom is the kernel-weighted fraction of
   * neighbors in its own phase (fluid or not); the densities of non-fluid atoms are reconstructed from their fluid
   * neighbors' pressures (compute_rheo_interface.ts), and fluid-solid pairs use the corrected solid velocity and density.
   */
  interfaceFlag = false;
  chi = new Float64Array(0);
  /** fix rheo/pressure's equation of state, set by that fix: the pressure of a density, its inverse, and whether it has one. */
  pressureEos: { calcPressure(rho: number, i: number): number; calcRho(p: number, i: number): number; invertible: boolean } | null = null;

  /**
   * compute_rheo_interface.cpp correct_v: the solid end's velocity becomes (v_solid - v_fluid) times the wall prefactor
   * max(2 cut (chi_solid - 0.5), 0) / max(2 cut (chi_fluid - 0.5), sqrt(3) cut / 12); the indices are owned atoms.
   */
  correctV(vSolid: number[], vFluid: number[], iSolid: number, iFluid: number): void {
    const wallMax = Math.sqrt(3.0) / 12.0 * this.cut;
    const wallNumer = Math.max(2.0 * this.cut * (this.chi[iSolid] - 0.5), 0.0);
    const wallDenom = Math.max(2.0 * this.cut * (this.chi[iFluid] - 0.5), wallMax);
    const wallPrefactor = wallNumer / wallDenom;
    vSolid[0] = (vSolid[0] - vFluid[0]) * wallPrefactor;
    vSolid[1] = (vSolid[1] - vFluid[1]) * wallPrefactor;
    vSolid[2] = (vSolid[2] - vFluid[2]) * wallPrefactor;
  }

  /** compute_rheo_interface.cpp correct_rho: a solid end's density is at least the equilibrium density of its type. */
  correctRho(rhoSolid: number, typeSolid: number): number {
    return Math.max(this.rho0[typeSolid], rhoSolid);
  }
  /** gradv (dim*dim per atom, uxx uxy ... row-major), gradr (dim per atom), vshift (dim per atom). */
  gradv = new Float64Array(0);
  gradr = new Float64Array(0);
  vshift = new Float64Array(0);
  /**
   * Scratch for the kernel derivative of one pair: dW is dWij (the owned end i), dWj the other end (dWji). For the
   * quintic kernel dWji = -dWij; the RK1 corrections make them differ.
   */
  readonly dW = new Float64Array(3);
  readonly dWj = new Float64Array(3);
  /**
   * RK1 reproducing kernel (compute_rheo_kernel.cpp, kstyle RK1): zmin is the coordination threshold; coord is the
   * neighbor count of each owned atom; C holds the correction coefficients (per owned atom, row k of (1 + dim) rows,
   * each of Mdim = 1 + dim entries, row-major); corrOK is 1 when the atom's coefficients are set (enough neighbors and
   * a positive-definite moment matrix). An atom without them uses the quintic kernel.
   */
  /** The kernel gradients are corrected (RK1 or RK2): pair rheo also takes laplacian_order 1 or 2. */
  get rk1(): boolean { return this.correctionOrder >= 1; }
  /** The size of the polynomial basis of the corrections (Mdim). */
  get mdim(): number {
    const d = this.dim;
    return this.correctionOrder === 2 ? 1 + d + (d * (d + 1)) / 2 : 1 + d;
  }
  zmin = 0;
  /**
   * Surface detection by coordination (fix rheo surface/detection coordination zmin splash): the thresholds, the
   * surface normal per owned atom (nsurface, dim per atom) and the color gradient it is built from (gradC).
   */
  surfaceFlag = false;
  zminSurface = 0;
  zminSplash = 0;
  /** surface/detection divergence: the threshold of the position divergence divr (below it an atom is at the surface). */
  surfaceDivr = false;
  divrSurface = 0;
  /** divr_i = -sum_j dW_ij . r_ij V_j of each owned atom (compute_rheo_surface.cpp), set by the surface detection. */
  divr = new Float64Array(0);
  nsurface = new Float64Array(0);
  gradC = new Float64Array(0);
  /**
   * Surface distance per owned atom (compute_rheo_surface.cpp rsurface): 0 for a surface or splash atom, the kernel cutoff
   * for a bulk group atom, and the smallest distance to a surface atom for a fluid neighbor of one. Set by the surface
   * detection (pre_force); fix rheo/oxidation and bond rheo/shell read it.
   */
  rsurface = new Float64Array(0);
  /** fix rheo/oxidation settings (bond cutoff, bond type, surface distance), null without the fix. */
  oxidation: { cut: number; btype: number; rsurf: number } | null = null;
  /**
   * fix rheo/pressure's sound speed of an atom when it depends on the state (variable_csq, the ideal/gas style):
   * pair rheo then uses calc_csq(rho, i) instead of the per-type csq. null when every type has a constant csq.
   */
  variableCsq: ((i: number) => number) | null = null;
  /** fix rheo rho/sum: density by kernel summation; selfMass uses the own mass in the sum (self/mass). */
  rhosumFlag = false;
  rhosumSelfMass = false;
  /**
   * fix rheo thermal keyword: the energy gradient (grade, dim per owned atom) is computed and fix rheo/thermal is the
   * thermal fix the pair and compute styles read (null without a fix rheo/thermal).
   */
  thermalFlag = false;
  thermal: RheoThermal | null = null;
  grade = new Float64Array(0);
  coord = new Int32Array(0);
  corrOK = new Uint8Array(0);
  C = new Float64Array(0);

  constructor(dim: 2 | 3, cut: number, ntypes: number) {
    this.dim = dim;
    this.cut = cut;
    this.cutsq = cut * cut;
    this.cutinv = 1.0 / cut;
    const cutsqinv = this.cutinv * this.cutinv;
    // compute_rheo_kernel.cpp init(): "pre_w = 7.0 / (478.0 * MY_PI) * 9 * cutsqinv;" for dim 2, and
    // "pre_wp = pre_w * 3.0 * cutinv;" (the 3d constants are not needed by the 2d styles of this port)
    if (dim === 3) {
      this.preW = 1.0 / (120.0 * Math.PI) * 27.0 * cutsqinv * this.cutinv;
    } else {
      this.preW = 7.0 / (478.0 * Math.PI) * 9 * cutsqinv;
    }
    this.preWp = this.preW * 3.0 * this.cutinv;
    this.rho0 = new Float64Array(ntypes + 1).fill(1.0);
    this.csq = new Float64Array(ntypes + 1).fill(1.0);
  }

  /**
   * fix rheo's kernel style: the wendland/c4 pre-factors (compute_rheo_kernel.cpp init(): "pre_w = 495.0 / (32.0 * MY_PI *
   * cutsq * cut)" in 3d, "pre_w = 9.0 / (MY_PI * cutsq)" in 2d, "pre_wp = pre_w * cutinv"), and the correction order.
   */
  setKernel(style: 'quintic' | 'wendland/c4' | 'RK0' | 'RK1' | 'RK2'): void {
    this.kernelStyle = style;
    this.correctionOrder = style === 'RK0' ? 0 : style === 'RK1' ? 1 : style === 'RK2' ? 2 : -1;
    if (style === 'wendland/c4') {
      if (this.dim === 3) this.preW = 495.0 / (32.0 * Math.PI * this.cutsq * this.cut);
      else this.preW = 9.0 / (Math.PI * this.cutsq);
      this.preWp = this.preW * this.cutinv;
    }
  }

  /** compute_rheo_kernel.cpp calc_w (its return value) and calc_w_self: the wendland/c4 kernel, else the quintic one. */
  calcW(r: number): number {
    return this.kernelStyle === 'wendland/c4' ? this.wWendland(r) : this.wQuintic(r);
  }

  /** compute_rheo_kernel.cpp calc_w_wendlandc4(r). */
  wWendland(r: number): number {
    const s = r * this.cutinv;
    let w: number;
    if (s > 1.0) w = 0.0;
    else {
      let tmp6 = (1.0 - s) * (1.0 - s);
      tmp6 *= tmp6 * tmp6;
      w = tmp6 * (1.0 + 6.0 * s + 35.0 * THIRD * s * s);
    }
    return w * this.preW;
  }

  /** compute_rheo_kernel.cpp calc_dw_wendlandc4: the scalar derivative (pre-factor included). */
  dwWendland(r: number): number {
    const s = r * this.cutinv;
    let wp: number;
    if (s > 1.0) wp = 0.0;
    else {
      const tmp1 = 1.0 - s;
      let tmp5 = tmp1 * tmp1;
      tmp5 = tmp5 * tmp5 * tmp1;
      const tmp6 = tmp5 * tmp1;
      wp = tmp6 * (6.0 + 70.0 * THIRD * s);
      wp -= 6 * tmp5 * (1.0 + 6.0 * s + 35.0 * THIRD * s * s);
    }
    return wp * this.preWp;
  }

  /** The correction basis H of a pair (calc_dw_rk1, calc_dw_rk2): 1, del/cut, and for RK2 the quadratic terms. */
  basis(delx: number, dely: number, delz: number): Float64Array {
    const H = this.H, ci = this.cutinv, c2 = ci * ci;
    H[0] = 1.0;
    H[1] = delx * ci;
    H[2] = dely * ci;
    if (this.dim === 2) {
      if (this.correctionOrder === 2) {
        H[3] = 0.5 * delx * delx * c2;
        H[4] = 0.5 * dely * dely * c2;
        H[5] = delx * dely * c2;
      }
    } else {
      H[3] = delz * ci;
      if (this.correctionOrder === 2) {
        H[4] = 0.5 * delx * delx * c2;
        H[5] = 0.5 * dely * dely * c2;
        H[6] = 0.5 * delz * delz * c2;
        H[7] = delx * dely * c2;
        H[8] = delx * delz * c2;
        H[9] = dely * delz * c2;
      }
    }
    return H;
  }

  /** compute_rheo_kernel.cpp calc_w_quintic(r): the quintic spline W(r) (pre-factor included). */
  wQuintic(r: number): number {
    const cutinv = this.cutinv;
    const s = r * 3.0 * cutinv;
    let w: number;
    if (s <= 3.0) {
      const tmp3 = 3.0 - s;
      const tmp3sq = tmp3 * tmp3;
      w = tmp3sq * tmp3sq * tmp3;
    } else {
      w = 0.0;
    }
    if (s <= 2.0) {
      const tmp2 = 2.0 - s;
      const tmp2sq = tmp2 * tmp2;
      w -= 6.0 * tmp2sq * tmp2sq * tmp2;
    }
    if (s <= 1.0) {
      const tmp1 = 1.0 - s;
      const tmp1sq = tmp1 * tmp1;
      w += 15.0 * tmp1sq * tmp1sq * tmp1;
    }
    return w * this.preW;
  }

  /** compute_rheo_kernel.cpp calc_dw_scalar_quintic(r): dW/dr (pre-factor included). */
  dwScalar(r: number): number {
    const cutinv = this.cutinv;
    const s = r * 3.0 * cutinv;
    let wp: number;
    if (s <= 3.0) {
      const tmp3 = 3.0 - s;
      const tmp3sq = tmp3 * tmp3;
      wp = -5.0 * tmp3sq * tmp3sq;
    } else {
      wp = 0.0;
    }
    if (s <= 2.0) {
      const tmp2 = 2.0 - s;
      const tmp2sq = tmp2 * tmp2;
      wp += 30.0 * tmp2sq * tmp2sq;
    }
    if (s <= 1.0) {
      const tmp1 = 1.0 - s;
      const tmp1sq = tmp1 * tmp1;
      wp -= 75.0 * tmp1sq * tmp1sq;
    }
    return wp * this.preWp;
  }

  /**
   * compute_rheo_kernel.cpp calc_dw(): the scalar derivative wp (returned; pre-factor included) and the vector derivative
   * of the pair into out: dWij of the owned atom o for del = x_o - x_j. wendland/c4: its derivative; quintic and RK0:
   * the quintic one (calc_dw_quintic). RK1 and RK2 with the corrections of o (corrOK), calc_dw_rk1 / calc_dw_rk2:
   * dW_a = w * sum_b C[o][1+a][b] H[b] over the basis H of the order. The native code leaves dWij unchanged for an
   * uncorrected atom (it reads the value of the previous pair); the port uses the quintic derivative there.
   */
  /**
   * calc_dw of one end: the gradient of the corrected kernel (RK1, RK2) for an atom with corrections, else of the
   * quintic (wendland/c4) kernel. fix_rheo.html: a particle with fewer neighbors "will revert to a non-reproducing
   * quintic kernel until it gains more neighbors". Native calc_dw (compute_rheo_kernel.cpp at b332f28) leaves its member
   * dWij or dWji unset for such an end, so it keeps the previous call's gradient; the engine follows the doc.
   */
  calcDw(o: number, delx: number, dely: number, delz: number, r: number, out: Float64Array): number {
    if (this.kernelStyle === 'wendland/c4') {
      const wp = this.dwWendland(r);
      const wprinv = wp / r;
      out[0] = delx * wprinv; out[1] = dely * wprinv; out[2] = delz * wprinv;
      return wp;
    }
    const wp = this.dwScalar(r);
    if (this.rk1 && this.corrOK[o]) {
      const dim = this.dim, Mdim = this.mdim;
      const H = this.basis(delx, dely, delz);
      const w = this.wQuintic(r);
      const base = o * (1 + dim) * Mdim;
      for (let a = 0; a < dim; a++) {
        let acc = 0;
        for (let b = 0; b < Mdim; b++) acc += this.C[base + (1 + a) * Mdim + b] * H[b];
        out[a] = acc;
        out[a] *= w;
      }
      return wp;
    }
    const wprinv = wp / r;
    out[0] = delx * wprinv;
    out[1] = dely * wprinv;
    out[2] = delz * wprinv;
    return wp;
  }

  /** Scratch of the correction basis H (at most 10 entries, RK2 in 3d). */
  readonly H = new Float64Array(10);
}

/*
 * Keyed by the custom-property map of the state, not the state object: a style that works on a copy of the state (the
 * bonded hybrid hands each sub-style its own topology) keeps the same map, and the map is only replaced by a restart.
 */
const models = new WeakMap<SimState['custom'], RheoModel>();

/** Registers the shared state of the simulation (fix rheo's constructor). */
export const setRheoModel = (s: SimState, m: RheoModel): void => { models.set(s.custom, m); };
/** The shared state of the simulation, or undefined when no fix rheo is defined. */
export const rheoModelOf = (s: SimState): RheoModel | undefined => models.get(s.custom);

/** Per-atom property arrays of the RHEO atom style (throws when the style is not atom_style rheo). */
export const rheoProp = (s: SimState, name: string): Float64Array => {
  const c = s.custom.get(name);
  if (!c) throw new Error(`RHEO property ${name} missing (atom_style rheo required)`);
  return c.data;
};

/**
 * The native newton-off half-list pairs (npair_bin.cpp, newton off, in native order) of the RHEO sums: compute_rheo_grad,
 * compute_rheo_vshift and compute_rheo_surface are all REQ_DEFAULT (half) requests, and each adds the owned end of a pair
 * and, for an owned partner, the other end. One list is shared, since every sum reads the same half list.
 */
const halfPairs = new NativePairOrder(false);
export const rheoHalfPairs = (s: SimState, nb: Neighbor, geom: Geometry): Int32Array => {
  if (!nb.half) throw new Error('rheo sums need a half neighbor list');
  return halfPairs.pairs(s, nb, geom, nb.half);
};

/**
 * The densities of a pair with interface/reconstruct ("Add corrections for walls" of the RHEO computes): a solid end
 * next to a fluid one takes correct_rho, two solid ends their equilibrium densities.
 */
export const interfaceRho = (m: RheoModel, type: Int32Array, fluidi: boolean, fluidj: boolean, rhoi: number, rhoj: number,
  i: number, oj: number): [number, number] => {
  if (fluidi && !fluidj) return [rhoi, m.correctRho(rhoj, type[oj])];
  if (!fluidi && fluidj) return [m.correctRho(rhoi, type[i]), rhoj];
  if (!fluidi && !fluidj) return [m.rho0[type[i]], m.rho0[type[oj]]];
  return [rhoi, rhoj];
};

/**
 * compute_rheo_grad.cpp compute_peratom(): gradv_i,ab = -sum_j (v_i - v_j)_a Vj dW_ij,b and gradr_i,a =
 * -sum_j (rho_i - rho_j) Vj dW_ij,a, with Vj = m_j / rho_j. The native half list (REQ_DEFAULT, newton off) visits each
 * owned pair once: the owned end i takes -(...) Vj dWij, and an owned partner j takes +(...) Vi dWji. Ghost neighbors
 * read the owner's velocity, density and mass; positions come from the neighbor list (xall).
 */
export const computeRheoGrad = (s: SimState, nb: Neighbor, m: RheoModel, geom: Geometry): void => {
  const n = s.n, dim = m.dim, dd = dim * dim;
  if (m.gradv.length < n * dd) m.gradv = new Float64Array(n * dd);
  if (m.gradr.length < n * dim) m.gradr = new Float64Array(n * dim);
  const gradv = m.gradv, gradr = m.gradr;
  gradv.fill(0, 0, n * dd);
  gradr.fill(0, 0, n * dim);
  const energyGrad = m.thermalFlag;
  if (energyGrad) {
    if (m.grade.length < n * dim) m.grade = new Float64Array(n * dim);
    m.grade.fill(0, 0, n * dim);
  }
  const x = nb.xall, owner = nb.owner;
  const v = s.v, rho = rheoProp(s, RHEO_RHO);
  const energy = energyGrad ? rheoProp(s, RHEO_ESPH) : null;
  const pairs = rheoHalfPairs(s, nb, geom);
  const status = rheoProp(s, RHEO_STATUS), type = s.type;
  const vij = [0, 0, 0], vi = [0, 0, 0], vj = [0, 0, 0];
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p];
    const j = pairs[p + 1] & NEIGHMASK;
    const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
    const rsq = delx * delx + dely * dely + delz * delz;
    if (rsq >= m.cutsq) continue;
    const oj = owner[j];
    let rhoi = rho[i], rhoj = rho[oj];
    for (let a = 0; a < 3; a++) { vi[a] = v[3 * i + a]; vj[a] = v[3 * oj + a]; }
    // compute_rheo_grad.cpp: "Add corrections for walls" (interface/reconstruct)
    if (m.interfaceFlag) {
      const fluidi = !(status[i] & PHASECHECK), fluidj = !(status[oj] & PHASECHECK);
      if (fluidi && !fluidj) {
        m.correctV(vj, vi, oj, i);
        rhoj = m.correctRho(rhoj, type[oj]);
      } else if (!fluidi && fluidj) {
        m.correctV(vi, vj, i, oj);
        rhoi = m.correctRho(rhoi, type[i]);
      } else if (!fluidi && !fluidj) {
        rhoi = m.rho0[type[i]];
        rhoj = m.rho0[type[oj]];
      }
    }
    const Voli = massOf(s, i) / rhoi, Volj = massOf(s, oj) / rhoj;
    for (let a = 0; a < dim; a++) vij[a] = vi[a] - vj[a];
    const drho = rhoi - rhoj;
    const de = energy ? energy[i] - energy[oj] : 0;
    const r = Math.sqrt(rsq);
    m.calcDw(i, delx, dely, delz, r, m.dW);
    m.calcDw(oj, -delx, -dely, -delz, r, m.dWj);
    const dWij = m.dW, dWji = m.dWj;
    for (let a = 0; a < dim; a++) {
      for (let b = 0; b < dim; b++) gradv[i * dd + a * dim + b] -= vij[a] * Volj * dWij[b];
      gradr[i * dim + a] -= drho * Volj * dWij[a];
      // compute_rheo_grad.cpp: "grade[i][a] -= de * Volj * dWij[a]" (energy keyword, fix rheo thermal)
      if (energy) m.grade[i * dim + a] -= de * Volj * dWij[a];
    }
    if (j < n) {
      for (let a = 0; a < dim; a++) {
        for (let b = 0; b < dim; b++) gradv[oj * dd + a * dim + b] += vij[a] * Voli * dWji[b];
        gradr[oj * dim + a] += drho * Voli * dWji[a];
        if (energy) m.grade[oj * dim + a] += de * Voli * dWji[a];
      }
    }
  }
};

/**
 * compute_rheo_vshift.cpp compute_peratom() (orthogonal and triclinic boxes alike, no interface, no cross-type
 * correction): the shifting velocity of each owned atom from the native half list. An owned pair adds the i end
 * vmag_i Vj dr dx and, for an owned partner j, the other end -vmag_j Vi dr dx (dr from the i end, as native does).
 */
export const computeRheoVShift = (s: SimState, nb: Neighbor, m: RheoModel, groupBit: number, geom: Geometry): void => {
  const n = s.n, dim = m.dim;
  if (m.vshift.length < n * dim) m.vshift = new Float64Array(n * dim);
  const vshift = m.vshift;
  vshift.fill(0, 0, n * dim);
  const x = nb.xall, owner = nb.owner;
  const v = s.v, rho = rheoProp(s, RHEO_RHO), status = rheoProp(s, RHEO_STATUS), mask = s.mask;
  const cutthird = m.cut / 3.0;
  const pairs = rheoHalfPairs(s, nb, geom);
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p];
    const j = pairs[p + 1] & NEIGHMASK;
    const oj = owner[j];
    const fluidi = !(status[i] & PHASECHECK);
    const fluidj = !(status[oj] & PHASECHECK);
    if (!fluidi && !fluidj) continue;
    if ((status[i] & STATUS_NO_SHIFT) && (status[oj] & STATUS_NO_SHIFT)) continue;
    const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
    const rsq = dx * dx + dy * dy + dz * dz;
    if (rsq >= m.cutsq) continue;
    const r = Math.sqrt(rsq);
    const rinv = 1 / r;
    // compute_rheo_vshift.cpp: "Add corrections for walls" (interface/reconstruct; the corrected solid velocity is not
    // read here, as only a fluid end shifts)
    let rhoi = rho[i], rhoj = rho[oj];
    if (m.interfaceFlag) [rhoi, rhoj] = interfaceRho(m, s.type, fluidi, fluidj, rhoi, rhoj, i, oj);
    const voli = massOf(s, i) / rhoi;
    const volj = massOf(s, oj) / rhoj;
    const wp = m.calcDw(i, dx, dy, dz, r, m.dW);
    const w = m.calcW(r);
    const w0 = m.calcW(cutthird);
    const w4 = w * w * w * w / (w0 * w0 * w0 * w0);
    const dr = -2 * cutthird * (1 + 0.2 * w4) * wp * rinv;
    if ((mask[i] & groupBit) && fluidi) {
      let vmag2 = 0;
      for (let a = 0; a < dim; a++) vmag2 += v[3 * i + a] * v[3 * i + a];
      const vmag = Math.sqrt(vmag2);
      const prefactor = vmag * volj * dr;
      vshift[i * dim] += prefactor * dx;
      vshift[i * dim + 1] += prefactor * dy;
      if (dim === 3) vshift[i * dim + 2] += prefactor * dz;
    }
    if (j < n && (mask[j] & groupBit) && fluidj) {
      let vmag2 = 0;
      for (let a = 0; a < dim; a++) vmag2 += v[3 * j + a] * v[3 * j + a];
      const vmag = Math.sqrt(vmag2);
      const prefactor = vmag * voli * dr;
      vshift[j * dim] -= prefactor * dx;
      vshift[j * dim + 1] -= prefactor * dy;
      if (dim === 3) vshift[j * dim + 2] -= prefactor * dz;
    }
  }
  // compute_rheo_vshift.cpp: "Zero any excluded types" (shift_type 0 for exclude/type)
  if (m.shiftType) {
    for (let i = 0; i < n; i++) {
      if (!m.shiftType[s.type[i]]) for (let a = 0; a < dim; a++) vshift[i * dim + a] = 0.0;
    }
  }
  if (m.shiftCrossType) correctRheoTypeInterface(s, nb, m, pairs);
};

/**
 * compute_rheo_vshift.cpp correct_type_interface() (fix rheo shift scale/cross/type), after Yang, Rakhsha, Hu and
 * Negrut 2022: over the same half pairs (rsq <= cutsq here), the cross-type colour ct_i = sum_j [t_i != t_j] V_j W_ij,
 * its gradient cgradt_i = -sum_j [t_i != t_j] V_j dW_ij (V_i from the per-type mass, as native), and the same-type sum
 * rsame_i = sum_j [t_i == t_j] W_ij r_ij. Then for every owned atom: rsame below rmin removes the shift; ct of at least
 * cmin removes (1 - scale) of the shift's component along the unit colour gradient.
 */
const correctRheoTypeInterface = (s: SimState, nb: Neighbor, m: RheoModel, pairs: Int32Array): void => {
  const n = s.n, dim = m.dim;
  const x = nb.xall, owner = nb.owner, type = s.type;
  const rho = rheoProp(s, RHEO_RHO), status = rheoProp(s, RHEO_STATUS);
  const vshift = m.vshift;
  const ct = new Float64Array(n), rsame = new Float64Array(n), cgradt = new Float64Array(3 * n);
  // colour function
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p];
    const j = pairs[p + 1] & NEIGHMASK;
    const oj = owner[j];
    const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
    const rsq = dx * dx + dy * dy + dz * dz;
    if (rsq > m.cutsq) continue;
    const r = Math.sqrt(rsq);
    let rhoi = rho[i], rhoj = rho[oj];
    if (m.interfaceFlag) [rhoi, rhoj] = interfaceRho(m, type, !(status[i] & PHASECHECK), !(status[oj] & PHASECHECK), rhoi, rhoj, i, oj);
    const voli = massOf(s, i) / rhoi;
    const volj = massOf(s, oj) / rhoj;
    const w = m.calcW(r);
    const ctmp = type[i] !== type[oj] ? 1 : 0;
    ct[i] += ctmp * volj * w;
    if (j < n) ct[j] += ctmp * voli * w;
  }
  // colour gradient and the same-type distance sum
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p];
    const j = pairs[p + 1] & NEIGHMASK;
    const oj = owner[j];
    const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
    const rsq = dx * dx + dy * dy + dz * dz;
    if (rsq > m.cutsq) continue;
    const r = Math.sqrt(rsq);
    let rhoi = rho[i], rhoj = rho[oj];
    if (m.interfaceFlag) [rhoi, rhoj] = interfaceRho(m, type, !(status[i] & PHASECHECK), !(status[oj] & PHASECHECK), rhoi, rhoj, i, oj);
    const voli = massOf(s, i) / rhoi;
    const volj = massOf(s, oj) / rhoj;
    const w = m.calcW(r);
    m.calcDw(i, dx, dy, dz, r, m.dW);
    m.calcDw(oj, -dx, -dy, -dz, r, m.dWj);
    const ctmp = type[i] !== type[oj] ? 1 : 0;
    for (let a = 0; a < dim; a++) {
      cgradt[3 * i + a] -= ctmp * volj * m.dW[a];
      if (j < n) cgradt[3 * j + a] -= ctmp * voli * m.dWj[a];
    }
    if (type[i] === type[oj]) {
      rsame[i] += w * r;
      if (j < n) rsame[j] += w * r;
    }
  }
  // remove the normal shifting component of the interfacial atoms
  const ntmp = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    // if isolated, just don't shift
    if (rsame[i] < m.shiftRmin) {
      for (let a = 0; a < dim; a++) vshift[i * dim + a] = 0.0;
      continue;
    }
    if (ct[i] < m.shiftCmin) continue;
    let minv = 0;
    for (let a = 0; a < dim; a++) minv += cgradt[3 * i + a] * cgradt[3 * i + a];
    if (minv !== 0) minv = 1 / Math.sqrt(minv);
    for (let a = 0; a < dim; a++) ntmp[a] = cgradt[3 * i + a] * minv;
    let dot = 0.0;
    for (let a = 0; a < dim; a++) dot += ntmp[a] * vshift[i * dim + a];
    for (let a = 0; a < dim; a++) vshift[i * dim + a] -= (1.0 - m.shiftScale) * ntmp[a] * dot;
  }
};

/**
 * compute_rheo_vshift.cpp correct_surfaces(): the shifting velocity of a surface atom loses its component along the
 * surface normal (the tangential part is kept) unless it points into the bulk (dot >= 0 is shifted as is); a splash atom
 * does not shift. Non-fluid atoms are skipped. Called in pre_force after the surface detection, when shift is on.
 */
export const correctRheoSurfaces = (s: SimState, m: RheoModel, groupBit: number): void => {
  const n = s.n, dim = m.dim;
  const status = rheoProp(s, RHEO_STATUS), mask = s.mask;
  const vs = m.vshift, ns = m.nsurface;
  for (let i = 0; i < n; i++) {
    if (!(mask[i] & groupBit)) continue;
    if (status[i] & PHASECHECK) continue;
    if (status[i] & STATUS_SURFACE) {
      const nx = ns[i * dim], ny = ns[i * dim + 1];
      const nz = dim === 3 ? ns[i * dim + 2] : 0;
      const vx = vs[i * dim], vy = vs[i * dim + 1];
      const vz = dim === 3 ? vs[i * dim + 2] : 0;
      let dot = nx * vx + ny * vy;
      if (dim === 3) dot += nz * vz;
      // allowing shifting into the bulk
      if (dot < 0.0) continue;
      vs[i * dim] = (1 - nx * nx) * vx - nx * ny * vy;
      vs[i * dim + 1] = (1 - ny * ny) * vy - nx * ny * vx;
      if (dim === 3) {
        vs[i * dim] -= nx * nz * vz;
        vs[i * dim + 1] -= ny * nz * vz;
        vs[i * dim + 2] = (1 - nz * nz) * vz - nz * ny * vy - nx * nz * vx;
      }
    } else if (status[i] & STATUS_SPLASH) {
      for (let a = 0; a < dim; a++) vs[i * dim + a] = 0.0;
    }
  }
};
