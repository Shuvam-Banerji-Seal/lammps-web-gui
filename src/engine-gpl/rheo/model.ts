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
export class RheoModel {
  readonly dim: 2 | 3;
  /** fix rheo cutoff and the derived kernel constants (compute_rheo_kernel.cpp init()). */
  readonly cut: number;
  readonly cutsq: number;
  readonly cutinv: number;
  readonly preW: number;
  readonly preWp: number;
  /** Equilibrium density and squared sound speed per type (index 0 unused). */
  readonly rho0: Float64Array;
  readonly csq: Float64Array;
  /** fix rheo shift: 1 for the types that shift, 0 for "exclude/type"; null without the shift keyword. */
  shiftType: Uint8Array | null = null;
  /** fix rheo shift: the shifting velocity is on. */
  shiftFlag = false;
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
  rk1 = false;
  zmin = 0;
  /**
   * Surface detection by coordination (fix rheo surface/detection coordination zmin splash): the thresholds, the
   * surface normal per owned atom (nsurface, dim per atom) and the color gradient it is built from (gradC).
   */
  surfaceFlag = false;
  zminSurface = 0;
  zminSplash = 0;
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
   * compute_rheo_kernel.cpp calc_dw(): the scalar derivative wp (returned, quintic; pre-factor included) and the vector
   * derivative of the pair into out: dWij of the owned atom o for del = x_o - x_j. With the RK1 correction of o
   * (corrOK), calc_dw_rk1(): dW_a = w * sum_b C[o][1+a][b] H[b], with H = (1, del/cut); otherwise the quintic
   * derivative dW = del wp / r (calc_dw_quintic). The native code leaves dWij unchanged for an uncorrected atom (it
   * reads the value of the previous pair); the port uses the quintic derivative there.
   */
  calcDw(o: number, delx: number, dely: number, delz: number, r: number, out: Float64Array): number {
    const wp = this.dwScalar(r);
    if (this.rk1 && this.corrOK[o]) {
      const dim = this.dim, Mdim = dim + 1;
      const H = this.H;
      H[0] = 1.0;
      H[1] = delx * this.cutinv;
      H[2] = dely * this.cutinv;
      if (dim === 3) H[3] = delz * this.cutinv;
      const w = this.wQuintic(r);
      const base = o * Mdim * Mdim;
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

  /** Scratch of the RK1 correction basis H (1, x/cut, y/cut, (z/cut)). */
  readonly H = new Float64Array(4);
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
 * compute_rheo_grad.cpp compute_peratom(): gradv_i,ab = -sum_j (v_i - v_j)_a Vj dW_ij,b and gradr_i,a =
 * -sum_j (rho_i - rho_j) Vj dW_ij,a, with Vj = m_j / rho_j, summed over the full neighbor list of each owned atom.
 * Ghost neighbors read the owner's velocity and density; positions come from the neighbor list (xall).
 */
export const computeRheoGrad = (s: SimState, nb: Neighbor, m: RheoModel): void => {
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
  const full = nb.full;
  if (!full) throw new Error('rheo grad needs a full neighbor list');
  const x = nb.xall, owner = nb.owner, nbr = full.neighbors;
  const v = s.v, rho = rheoProp(s, RHEO_RHO);
  const energy = energyGrad ? rheoProp(s, RHEO_ESPH) : null;
  const vij = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const rhoi = rho[i];
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= m.cutsq) continue;
      const o = owner[j];
      const rhoj = rho[o];
      const Volj = massOf(s, o) / rhoj;
      for (let a = 0; a < dim; a++) vij[a] = v[3 * i + a] - v[3 * o + a];
      const drho = rhoi - rhoj;
      const de = energy ? energy[i] - energy[o] : 0;
      m.calcDw(i, delx, dely, delz, Math.sqrt(rsq), m.dW);
      const dW = m.dW;
      for (let a = 0; a < dim; a++) {
        for (let b = 0; b < dim; b++) gradv[i * dd + a * dim + b] -= vij[a] * Volj * dW[b];
        gradr[i * dim + a] -= drho * Volj * dW[a];
        // compute_rheo_grad.cpp: "grade[i][a] -= de * Volj * dWij[a]" (energy keyword, fix rheo thermal)
        if (energy) m.grade[i * dim + a] -= de * Volj * dW[a];
      }
    }
  }
};

/**
 * compute_rheo_vshift.cpp compute_peratom() (orthogonal and triclinic boxes alike, no interface, no cross-type
 * correction): the shifting velocity of each owned atom from its full neighbor list. The pair term of the native
 * half list is split between its two atoms, so the full-list sum gives the same vshift.
 */
export const computeRheoVShift = (s: SimState, nb: Neighbor, m: RheoModel, groupBit: number): void => {
  const n = s.n, dim = m.dim;
  if (m.vshift.length < n * dim) m.vshift = new Float64Array(n * dim);
  const vshift = m.vshift;
  vshift.fill(0, 0, n * dim);
  const full = nb.full;
  if (!full) throw new Error('rheo vshift needs a full neighbor list');
  const x = nb.xall, owner = nb.owner, nbr = full.neighbors;
  const v = s.v, rho = rheoProp(s, RHEO_RHO), status = rheoProp(s, RHEO_STATUS), mask = s.mask;
  const cutthird = m.cut / 3.0;
  for (let i = 0; i < n; i++) {
    const fluidi = !(status[i] & PHASECHECK);
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const o = owner[j];
      const fluidj = !(status[o] & PHASECHECK);
      if (!fluidi && !fluidj) continue;
      if ((status[i] & STATUS_NO_SHIFT) && (status[o] & STATUS_NO_SHIFT)) continue;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (rsq >= m.cutsq) continue;
      const r = Math.sqrt(rsq);
      const rinv = 1 / r;
      const volj = massOf(s, o) / rho[o];
      const wp = m.calcDw(i, dx, dy, dz, r, m.dW);
      const w = m.wQuintic(r);
      const w0 = m.wQuintic(cutthird);
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
    }
  }
  // compute_rheo_vshift.cpp: "Zero any excluded types" (shift_type 0 for exclude/type)
  if (m.shiftType) {
    for (let i = 0; i < n; i++) {
      if (!m.shiftType[s.type[i]]) for (let a = 0; a < dim; a++) vshift[i * dim + a] = 0.0;
    }
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
      } else {
        vs[i * dim + 2] = 0.0;
      }
    } else if (status[i] & STATUS_SPLASH) {
      for (let a = 0; a < dim; a++) vs[i * dim + a] = 0.0;
    }
  }
};
