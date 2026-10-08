import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../types';
import type { SimState } from '../../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';
import { mdrCheckTriangles, mdrDampCoeff, mdrElastic } from './granular_mdr';

/*
 * pair_style granular — docs.lammps.org/pair_granular.html (plans/lammps-docs/pair_granular.rst).
 *   pair_style granular cutoff
 *   pair_coeff I J normal-model normal-args ... [damping d] tangential t args
 *              [rolling r args] [twisting w args] [limit_damping] [cutoff c]
 * Normal models (doc, "The first required keyword for the pair_coeff command is
 * the normal contact model"), with the arguments the doc lists for each: hooke
 * (k_n, eta_n0 or e), hertz (k_n, eta_n0 or e), hertz/material (E, eta_n0 or e,
 * nu), dmt (E, eta_n0 or e, nu, gamma) and jkr (E, eta_n0 or e, nu, gamma):
 *   "\mathbf{F}_{ne, Hooke} = k_n \delta_{ij} \mathbf{n}"
 *   "\mathbf{F}_{ne, Hertz} = k_n R_{eff}^{1/2}\delta_{ij}^{3/2} \mathbf{n}"
 *   "\mathbf{F}_{ne, Hertz/material} = \frac{4}{3} E_{eff} R^{1/2}\delta_{ij}^{3/2} \mathbf{n}"
 *   "\mathbf{F}_{ne, dmt} = \left(\frac{4}{3} E R^{1/2}\delta_{ij}^{3/2} - 4\pi\gamma R\right)\mathbf{n}"
 *   "\delta = a^2/R - 2\sqrt{\pi \gamma a/E}" (jkr)
 * Damping (normal): "\mathbf{F}_{n,damp} = -\eta_n \mathbf{v}_{n,rel}"; eta_n = eta_n0 (velocity),
 * "\eta_n = \eta_{n0} m_{eff}" (mass_velocity), "\eta_n = \eta_{n0}\ a m_{eff}" with
 * "a =\sqrt{R\delta}" (viscoelastic), "\eta_n = \alpha (m_{eff}k_{nd})^{1/2}" (tsuji).
 * Tangential models with their listed arguments: linear_nohistory (x_gamma,t, mu_s),
 * linear_history (k_t, x_gamma,t, mu_s), mindlin (k_t or NULL, x_gamma,t, mu_s),
 * mindlin/force, mindlin_rescale, mindlin_rescale/force.
 * "\mathbf{F}_t =  -\min(\mu_t F_{n0}, \|\mathbf{F}_\mathrm{t,damp}\|) \mathbf{t}" (linear_nohistory).
 * "The tangential damping prefactor :math:`\eta_t` is calculated by scaling" the normal damping.
 * Rolling sds takes k_roll, gamma_roll, mu_roll and twisting sds k_twist, gamma_twist, mu_twist (doc lists);
 * twisting marshall uses "k_{twist} = 0.5k_ta^2", "\eta_{twist} = 0.5\eta_ta^2", "\mu_{twist} = \frac{2}{3}a\mu_t".
 * limit_damping: "If the optional *limit_damping* keyword is used, this option will zero out the normal
 * component of the force if there is an effective attractive force. This keyword cannot be used with the JKR or DMT models."
 * Mixing: "geometric averaging for most quantities"; "E_{eff,ij} = \left(\frac{1-\nu_i^2}{E_i} + \frac{1-\nu_j^2}{E_j}\right)^{-1}".
 * Mindlin: "k_t = 8G_{eff}" with "G_{eff} = \left(\frac{2-\nu_i}{G_i} + \frac{2-\nu_j}{G_j}\right)^{-1}" when k_t is NULL.
 * Torques: "\mathbf{\tau}_i = -(R_i - 0.5 \delta) \mathbf{n} \times \mathbf{F}_t" (contact point at the centre of overlap).
 * The energy is zero: "The single() function of these pair styles returns 0.0 for the energy of a pairwise interaction".
 *
 * Measured with native LAMMPS (black box; two-sphere runs dumped every step at %.15g, and the
 * w6granular_* oracle cases):
 * - dmt: Hertz(E_eff) - 4 pi gamma R for delta > 0 and no force for delta <= 0; the friction
 *   cap uses F_n0 = |F_ne + 2 F_pulloff| with F_pulloff = 4 pi gamma R.
 * - jkr: contact is lost at delta(a_c) with a_c = (9 pi gamma R^2 / (4 E_eff))^(1/3) (pull-off
 *   force -3 pi gamma R), not at the minimum of delta(a); a pair at delta < 0 feels a force only
 *   if it has been in contact before; the neighbor cutoff is radsum - delta_c.
 * - rolling: F0 = -k xi - gamma v with torque -R_eff n x F; a capped rolling force with no stored
 *   displacement drops the force; the rolling displacement is rescaled like the tangential one.
 * - twisting: the capped torque takes sgn(Omega).
 * - a frozen partner (fix freeze) contributes no mass to m_eff.
 * The multi-body periodic rolling case (w6granular_rolling) still differs by about 1e-5 at step
 * 150 although two-sphere rolling and twisting runs match to 1e-12.
 * mdr (granular_mdr.ts, measured on two equal spheres): the elastic branch is implemented for particle pairs with
 * equal radii, the damping mdr 1 and mdr 2, and the tangential linear models. The plastic branch (overlap at or
 * beyond the yield displacement), adhesion (surface energy > 0), unequal radii, mindlin tangential models and
 * the marshall twisting model throw a StyleError.
 * Not implemented (StyleError): damping coeff_restitution with mdr, heat, synchronized_verlet.
 */

type NormalKind = 'hooke' | 'hertz' | 'hertz/material' | 'dmt' | 'jkr' | 'mdr';
type DampKind = 'velocity' | 'mass_velocity' | 'viscoelastic' | 'tsuji' | 'coeff_restitution' | 'mdr';
type TangKind = 'linear_nohistory' | 'linear_history' | 'mindlin' | 'mindlin/force' | 'mindlin_rescale' | 'mindlin_rescale/force';
type RollKind = 'none' | 'sds';
type TwistKind = 'none' | 'sds' | 'marshall';

/** One pair_coeff line as written. */
export interface Spec {
  normal: NormalKind;
  /** hooke / hertz: spring constant k_n. */
  kn: number;
  /** hertz/material, dmt, jkr: Young's modulus, Poisson ratio, surface energy. */
  E: number; nu: number; gamma: number;
  /** eta_n0, or the restitution coefficient e for tsuji. */
  eta: number;
  /** mdr: yield stress Y, critical confinement ratio psi_b (bulk response, not used before yield), damping class d_type. */
  Y: number; psib: number; dtype: number;
  damp: DampKind;
  tang: TangKind;
  /** tangential stiffness; ktNull means NULL (from the shear modulus). */
  kt: number; ktNull: boolean;
  xgt: number; mu: number;
  roll: RollKind; kr: number; gr: number; mr: number;
  twist: TwistKind; kw: number; gw: number; mw: number;
  limit: boolean;
  cutoff: number | null;
}

/** Resolved parameters for one type pair (i <= j), after mixing. */
export interface Pm {
  normal: NormalKind;
  kn: number;
  /** Effective modulus E_eff (doc: E_eff = E/(2(1-nu^2)) for identical types). */
  Eeff: number;
  gamma: number;
  eta: number;
  /** mdr fields (absent for the other models; the legacy wall path builds a Pm without them). */
  Y?: number; psib?: number; dtype?: number;
  damp: DampKind;
  tang: TangKind;
  kt: number;
  xgt: number; mu: number;
  roll: RollKind; kr: number; gr: number; mr: number;
  twist: TwistKind; kw: number; gw: number; mw: number;
  limit: boolean;
  cutoff: number;
  /** Compatibility key: pairs of types are mixed only when the model choices agree. */
  sig: string;
}

const NUM_ARGS: Record<NormalKind, number> = { hooke: 2, hertz: 2, 'hertz/material': 3, dmt: 4, jkr: 4, mdr: 6 };
const DAMPS: Record<string, DampKind> = { velocity: 'velocity', mass_velocity: 'mass_velocity', viscoelastic: 'viscoelastic', tsuji: 'tsuji', coeff_restitution: 'coeff_restitution' };
const TANGS: Record<string, TangKind> = {
  linear_nohistory: 'linear_nohistory', linear_history: 'linear_history', mindlin: 'mindlin',
  'mindlin/force': 'mindlin/force', mindlin_rescale: 'mindlin_rescale', 'mindlin_rescale/force': 'mindlin_rescale/force',
};

/** Coefficient alpha of the tsuji damping, from e (doc: alpha = 1.2728 - 4.2783e + 11.087e^2 - 22.348e^3 + 27.467e^4 - 18.022e^5 + 4.8218e^6). */
const tsujiAlpha = (e: number): number =>
  1.2728 - 4.2783 * e + 11.087 * e ** 2 - 22.348 * e ** 3 + 27.467 * e ** 4 - 18.022 * e ** 5 + 4.8218 * e ** 6;

/** Parses the arguments of one pair_coeff (after I J). */
export function parseGranularSpec(args: string[]): Spec {
  const word = args[0];
  if (!(word in NUM_ARGS)) throw new StyleError(`pair_style granular: unknown normal model '${word}' (hooke, hertz, hertz/material, dmt, jkr, mdr)`);
  const normal = word as NormalKind;
  const nn = NUM_ARGS[normal];
  if (args.length < 1 + nn) throw new StyleError(`pair_coeff granular ${normal} needs ${nn} numeric arguments`);
  const num = args.slice(1, 1 + nn).map((w, k) => parseNum(w, `${normal} argument ${k + 1}`));
  const sp: Spec = {
    normal, kn: 0, E: 0, nu: 0, gamma: 0, eta: 0, Y: 0, psib: 0, dtype: 0,
    damp: 'viscoelastic', tang: 'linear_nohistory', kt: 0, ktNull: false, xgt: 0, mu: 0,
    roll: 'none', kr: 0, gr: 0, mr: 0, twist: 'none', kw: 0, gw: 0, mw: 0, limit: false, cutoff: null,
  };
  if (normal === 'hooke' || normal === 'hertz') {
    sp.kn = num[0];
    sp.eta = num[1];
  } else if (normal === 'mdr') {
    // doc (pair_granular.html, normal model 6, in this order): E, nu, Y, Delta gamma, psi_b, eta_n0
    sp.E = num[0];
    sp.nu = num[1];
    sp.Y = num[2];
    sp.gamma = num[3];
    sp.psib = num[4];
    sp.eta = num[5];
    // doc: the surface energy is "\Delta\gamma \ge 0"; a non-zero value is the adhesive (Part I) branch, not implemented here
    if (sp.gamma !== 0) throw new StyleError("pair_style granular: 'mdr' is not implemented for a surface energy above 0 (adhesive mdr) in this engine");
    if (!(sp.E > 0)) throw new StyleError("pair_style granular: normal model 'mdr' needs Young's modulus E > 0");
    if (!(sp.nu >= 0 && sp.nu <= 0.5)) throw new StyleError("pair_style granular: normal model 'mdr' needs Poisson's ratio 0 <= nu <= 0.5");
    if (!(sp.Y >= 0)) throw new StyleError("pair_style granular: normal model 'mdr' needs yield stress Y >= 0");
    if (!(sp.psib >= 0 && sp.psib <= 1)) throw new StyleError("pair_style granular: normal model 'mdr' needs critical confinement ratio 0 <= psi_b <= 1");
    if (!(sp.eta >= 0)) throw new StyleError("pair_style granular: normal model 'mdr' needs damping coefficient eta_n0 >= 0");
  } else {
    sp.E = num[0];
    sp.eta = num[1];
    sp.nu = num[2];
    if (normal === 'dmt' || normal === 'jkr') sp.gamma = num[3];
  }
  let tangSeen = false;
  let dampSeen = false;
  let k = 1 + nn;
  const need = (n: number, what: string) => {
    if (args.length < k + n) throw new StyleError(`pair_coeff granular: '${what}' needs ${n} arguments`);
  };
  const next = () => args[k++];
  while (k < args.length) {
    const kw = next();
    if (kw === 'damping') {
      need(1, 'damping');
      dampSeen = true;
      const d = next();
      if (d === 'mdr') {
        // doc: "The *mdr* damping class contains multiple damping models that can be toggled between by specifying different integer values for the :math:`d_{type}` input parameter."
        need(1, 'damping mdr');
        const dt = parseNum(next(), 'damping mdr d_type');
        if (dt !== 1 && dt !== 2) throw new StyleError(`pair_style granular: damping mdr d_type must be 1 or 2 (got ${dt})`);
        sp.damp = 'mdr';
        sp.dtype = dt;
        continue;
      }
      if (!(d in DAMPS)) throw new StyleError(`pair_style granular: unknown damping '${d}'`);
      sp.damp = DAMPS[d];
    } else if (kw === 'tangential') {
      need(1, 'tangential');
      const t = next();
      if (t === 'synchronized_verlet') throw new StyleError("pair_style granular: 'synchronized_verlet' is not implemented in this engine");
      if (!(t in TANGS)) throw new StyleError(`pair_style granular: unknown tangential model '${t}'`);
      sp.tang = TANGS[t];
      tangSeen = true;
      if (sp.tang === 'linear_nohistory') {
        need(2, 'tangential linear_nohistory');
        sp.xgt = parseNum(next(), 'x_gamma_t');
        sp.mu = parseNum(next(), 'mu_s');
      } else {
        need(3, `tangential ${t}`);
        const ktw = next();
        if (ktw === 'NULL') {
          if (sp.tang === 'linear_history') throw new StyleError('pair_style granular: tangential linear_history needs a numeric k_t (NULL is for the mindlin models)');
          sp.ktNull = true;
        } else sp.kt = parseNum(ktw, 'k_t');
        sp.xgt = parseNum(next(), 'x_gamma_t');
        sp.mu = parseNum(next(), 'mu_s');
      }
    } else if (kw === 'rolling') {
      need(1, 'rolling');
      const r = next();
      if (r === 'none') sp.roll = 'none';
      else if (r === 'sds') {
        need(3, 'rolling sds');
        sp.roll = 'sds';
        sp.kr = parseNum(next(), 'k_roll');
        sp.gr = parseNum(next(), 'gamma_roll');
        sp.mr = parseNum(next(), 'mu_roll');
      } else throw new StyleError(`pair_style granular: unknown rolling model '${r}' (none, sds)`);
    } else if (kw === 'twisting') {
      need(1, 'twisting');
      const w = next();
      if (w === 'none') sp.twist = 'none';
      else if (w === 'marshall') sp.twist = 'marshall';
      else if (w === 'sds') {
        need(3, 'twisting sds');
        sp.twist = 'sds';
        sp.kw = parseNum(next(), 'k_twist');
        sp.gw = parseNum(next(), 'gamma_twist');
        sp.mw = parseNum(next(), 'mu_twist');
      } else throw new StyleError(`pair_style granular: unknown twisting model '${w}' (none, sds, marshall)`);
    } else if (kw === 'limit_damping') {
      sp.limit = true;
    } else if (kw === 'cutoff') {
      need(1, 'cutoff');
      sp.cutoff = parseNum(next(), 'cutoff');
    } else if (kw === 'heat') {
      throw new StyleError("pair_style granular: the 'heat' keyword is not implemented in this engine");
    } else {
      throw new StyleError(`pair_coeff granular: unknown keyword '${kw}'`);
    }
  }
  if (!tangSeen) throw new StyleError('pair_coeff granular: the required keyword tangential is missing');
  // doc: "If you use the *mdr* normal model the only supported damping option is the *mdr* damping class described below."
  if (normal === 'mdr' && (!dampSeen || sp.damp !== 'mdr')) {
    throw new StyleError("pair_style granular: normal model 'mdr' needs damping mdr d_type (the only supported damping option)");
  }
  if (sp.damp === 'mdr' && normal !== 'mdr') throw new StyleError("pair_style granular: damping 'mdr' needs the normal model 'mdr'");
  if (normal === 'mdr' && sp.tang !== 'linear_nohistory' && sp.tang !== 'linear_history') {
    throw new StyleError(`pair_style granular: tangential ${sp.tang} with normal model 'mdr' is not implemented in this engine`);
  }
  if (normal === 'mdr' && sp.twist === 'marshall') {
    throw new StyleError("pair_style granular: twisting marshall with normal model 'mdr' is not implemented in this engine");
  }
  if ((normal === 'dmt' || normal === 'jkr') && (sp.damp === 'tsuji' || sp.damp === 'coeff_restitution')) {
    throw new StyleError(`pair_style granular: damping ${sp.damp} is not compatible with the ${normal} model`);
  }
  if (sp.damp === 'coeff_restitution' && !(sp.eta > 0 && sp.eta <= 1)) {
    throw new StyleError('pair_style granular: damping coeff_restitution needs the restitution coefficient e in (0, 1]');
  }
  if ((normal === 'dmt' || normal === 'jkr') && sp.limit) {
    throw new StyleError(`pair_style granular: limit_damping cannot be used with the ${normal} model`);
  }
  if (sp.damp === 'tsuji' && sp.eta < 0) throw new StyleError('pair_style granular: restitution coefficient e must be >= 0');
  return sp;
}

/** Effective modulus of a material pair from its own E and nu (doc: E_eff = E/(2(1-nu^2)) for like types). */
const effModulus = (E: number, nu: number): number => E / (2 * (1 - nu * nu));

/**
 * JKR contact (doc: "\\delta = a^2/R - 2\\sqrt{\\pi \\gamma a/E}" and
 * "\\mathbf{F}_{ne, jkr} = \\left(\\frac{4Ea^3}{3R} - 2\\pi a^2\\sqrt{\\frac{4\\gamma E}{\\pi a}}\\right)\\mathbf{n}").
 * a is the root on the stable branch (a >= a*, the minimum of delta(a) at which the
 * tensile branch ends); delta < delta_c (the pull-off overlap) is out of contact.
 * LAMMPS inverts the relation for a numerically (doc); here by bisection.
 */
const jklDelta = (a: number, R: number, E: number, gamma: number): number => (a * a) / R - 2 * Math.sqrt((Math.PI * gamma * a) / E);
/** Contact radius at pull-off (the stable branch ends there; the force is then -3 pi gamma R). Measured with native LAMMPS: contact is lost at delta(a_c), not at the minimum of delta(a). */
const jklAStar = (R: number, E: number, gamma: number): number => Math.cbrt((9 * Math.PI * gamma * R * R) / (4 * E));
/** Pull-off overlap delta_c = delta(a*) (negative): the separation at which the JKR force vanishes is radsum - delta_c. */
export const jklPullOff = (R: number, E: number, gamma: number): number => jklDelta(jklAStar(R, E, gamma), R, E, gamma);

export function jklContact(delta: number, R: number, E: number, gamma: number): { a: number; f: number } | null {
  const aStar = jklAStar(R, E, gamma);
  const dc = jklDelta(aStar, R, E, gamma);
  if (delta < dc) return null;
  let lo = aStar, hi = aStar;
  while (jklDelta(hi, R, E, gamma) < delta) hi *= 2;
  for (let k = 0; k < 200; k++) {
    const mid = 0.5 * (lo + hi);
    if (jklDelta(mid, R, E, gamma) < delta) lo = mid; else hi = mid;
  }
  const a = 0.5 * (lo + hi);
  const f = (4 * E * a ** 3) / (3 * R) - 2 * Math.PI * a * a * Math.sqrt((4 * gamma * E) / (Math.PI * a));
  return { a, f };
}

/** Inputs of one contact (pair or wall): unit normal n from j to i, overlap, effective and per-particle radii. */
export interface ContactIn {
  pm: Pm;
  nx: number; ny: number; nz: number;
  r: number; delta: number;
  /** Effective radius of the contact (R_i R_j / (R_i + R_j); the particle radius for a flat wall). */
  Rf: number;
  /** Radii used for the torque arms (R_i - delta/2 on i, R_j - delta/2 on j) and the tangential rotation term. */
  ri: number; rj: number;
  meff: number;
  /** v_i - v_j (for a wall: the particle velocity minus the wall velocity). */
  vrx: number; vry: number; vrz: number;
  oix: number; oiy: number; oiz: number;
  ojx: number; ojy: number; ojz: number;
  dt: number;
  /** True inside a timestep: the history advances. */
  update: boolean;
  /**
   * mdr damping is evaluated unless this is the setup force evaluation at step 0 (measured: the first run's setup has no
   * mdr damping; a setup at a later step, e.g. the second of two runs, has it). Default true.
   */
  mdrDamp?: boolean;
  /** -1 when the stored history is in the frame of the other atom (ID order). */
  sg: number;
  /**
   * Torque arm of particle i is its full radius R_i (the contact point on a wall surface, measured with
   * native LAMMPS for fix wall/gran granular); by default the arm is R_i - delta/2 (contact at the overlap centre).
   */
  surfaceArm?: boolean;
  /**
   * Legacy hertz/history wall forces (fix wall/gran hertz/history): the normal force, the damping and the
   * tangential force (spring and damping, not the displacement update) are scaled by sqrt(delta R_eff).
   * Measured-to-legacy: gran.ts and wall/gran use poly = sqrt(delta R) for the whole contact. Default 1.
   */
  poly?: number;
}

/** Outputs of one contact: force on i (minus on j), torques on i and j. */
export interface ContactOut {
  fx: number; fy: number; fz: number;
  tix: number; tiy: number; tiz: number;
  tjx: number; tjy: number; tjz: number;
  /** The contact keeps a history record (the caller stores sh and marks the key as seen). */
  hist: boolean;
}

export const newContactOut = (): ContactOut => ({ fx: 0, fy: 0, fz: 0, tix: 0, tiy: 0, tiz: 0, tjx: 0, tjy: 0, tjz: 0, hist: false });

/**
 * One granular contact: normal (hooke/hertz/hertz-material/dmt/jkr), damping, tangential (linear, mindlin),
 * rolling sds and twisting. sh holds the history: [0..2] tangential displacement (or elastic force), [3] the
 * previous contact radius, [4..6] rolling displacement, [7] twisting displacement; it is read and written in
 * the frame given by sg. Returns false when there is no contact (the caller drops the history record).
 */
export function granularContact(c: ContactIn, sh: Float64Array, out: ContactOut, hadHistory: boolean): boolean {
  const { pm, nx, ny, nz, r, delta, Rf, ri, rj, meff, dt, update, sg } = c;
  const jkr = pm.normal === 'jkr';
  const mdr = pm.normal === 'mdr';
  // jkr hysteresis (doc): the tensile range delta < 0 applies only once the pair has been in contact
  if (jkr && delta < 0 && !hadHistory) return false;
  const sqrtR = Math.sqrt(Rf);
  const vrx = c.vrx, vry = c.vry, vrz = c.vrz;
  const vn = vrx * nx + vry * ny + vrz * nz; // (v_i - v_j) . n
  const vnx = nx * vn, vny = ny * vn, vnz = nz * vn;
  // (R_i w_i + R_j w_j) x n
  const Ax = ri * c.oix + rj * c.ojx;
  const Ay = ri * c.oiy + rj * c.ojy;
  const Az = ri * c.oiz + rj * c.ojz;
  const wnx = Ay * nz - Az * ny, wny = Az * nx - Ax * nz, wnz = Ax * ny - Ay * nx;
  const vtx = vrx - vnx - wnx, vty = vry - vny - wny, vtz = vrz - vnz - wnz;
  // normal force along n (positive repulsive): elastic part, then damping
  let fne = 0;
  let contactA = 0;
  // pull-off force of the cohesive models (doc: F_pulloff = 4 pi gamma R for dmt, 3 pi gamma R for jkr)
  const fPull = pm.normal === 'dmt' ? 4 * Math.PI * pm.gamma * Rf : jkr ? 3 * Math.PI * pm.gamma * Rf : 0;
  if (mdr) {
    // mdr: equal radii only (the unequal-radius formula was not identified; a 0.7 % mismatch to the candidate forms was measured)
    if (ri !== rj) throw new StyleError("pair_style granular: normal model 'mdr' with particles of unequal radii is not implemented in this engine");
    if (delta <= 0) return false;
    const m = mdrElastic(delta, ri, 2 * pm.Eeff, pm.Y ?? 0);
    fne = m.fne;
    contactA = m.a;
  } else if (jkr) {
    const cc = jklContact(delta, Rf, pm.Eeff, pm.gamma);
    if (!cc) return false;
    fne = cc.f;
    contactA = cc.a;
  } else {
    if (delta <= 0) return false;
    contactA = Math.sqrt(Rf * delta);
    if (pm.normal === 'hooke') fne = pm.kn * delta;
    else if (pm.normal === 'hertz') fne = pm.kn * sqrtR * delta * Math.sqrt(delta);
    else fne = (4 / 3) * pm.Eeff * sqrtR * delta * Math.sqrt(delta);
    // dmt: cohesion -4 pi gamma R on top of the Hertz term (doc: F_ne,dmt = (4/3 E R^1/2 delta^3/2 - 4 pi gamma R) n)
    fne -= pm.normal === 'dmt' ? fPull : 0;
  }
  const contact = contactA;
  let etaN: number;
  switch (pm.damp) {
    case 'velocity': etaN = pm.eta; break;
    case 'mass_velocity': etaN = pm.eta * meff; break;
    case 'viscoelastic': etaN = pm.eta * contact * meff; break;
    // doc: the mdr damping class (d_type 1 or 2); the coefficients are measured in granular_mdr.ts
    case 'mdr': etaN = (c.mdrDamp ?? true) ? mdrDampCoeff(pm.dtype ?? 0, pm.eta, meff, delta, ri, 2 * pm.Eeff) : 0; break;
    case 'coeff_restitution': {
      // doc: eta_n = sqrt(4 m_eff k_nd / (1 + (pi / log e)^2)) for hooke; otherwise
      // eta_n = -2 sqrt(5/6) log(e) / sqrt(pi^2 + log(e)^2) * sqrt(3/2 k_nd m_eff); k_nd = F_elastic / delta.
      // Measured with native LAMMPS (black box): both forms match with no extra factor (ratio 1.0 for hooke and hertz).
      const knd = fne / delta;
      const le = Math.log(pm.eta);
      etaN = pm.normal === 'hooke'
        ? Math.sqrt((4 * meff * knd) / (1 + (Math.PI / le) ** 2))
        : (-2 * Math.sqrt(5 / 6) * le / Math.sqrt(Math.PI * Math.PI + le * le)) * Math.sqrt(1.5 * knd * meff);
      break;
    }
    default: {
      // knd: the elastic normal force per unit overlap; measured with native LAMMPS (black box):
      // eta_n = sqrt(2) * alpha(e) * sqrt(m_eff * knd) for hooke, hertz and hertz/material alike
      const knd = fne / delta;
      etaN = Math.SQRT2 * tsujiAlpha(pm.eta) * Math.sqrt(meff * knd);
    }
  }
  const poly = c.poly ?? 1;
  const fn = poly * (fne - etaN * vn);
  if (pm.limit && fn < 0) return false;
  // F_n0 for the Coulomb cap: |F_ne| for non-cohesive models, |F_ne + 2 F_pulloff| for dmt and jkr (doc)
  const fnAbs = pm.normal === 'dmt' || jkr ? Math.abs(fne + 2 * fPull) : Math.abs(fn);
  const fnx = fn * nx, fny = fn * ny, fnz = fn * nz;
  // tangential force: damping from eta_t = x_gamma_t eta_n; the spring is -k_t xi (linear_history),
  // -k_t a xi with a = sqrt(R delta) (mindlin), or the stored elastic force F_te (mindlin/force)
  const etaT = pm.xgt * etaN;
  // the poly factor scales the forces (not the displacement update, and not the history-cap coefficient)
  const etaTf = etaT * poly;
  const dampx = -etaTf * vtx, dampy = -etaTf * vty, dampz = -etaTf * vtz;
  const kind = pm.tang;
  // jkr keeps a contact record too, so the tensile branch can be recognised (see hysteresis above)
  const hist = kind !== 'linear_nohistory' || pm.roll !== 'none' || pm.twist !== 'none' || jkr;
  const forceVariant = kind === 'mindlin/force' || kind === 'mindlin_rescale/force';
  const rescale = kind === 'mindlin_rescale' || kind === 'mindlin_rescale/force';
  const aC = contactA;
  const keff = kind === 'linear_history' ? pm.kt : kind === 'linear_nohistory' ? 0 : pm.kt * aC;
  let shx = 0, shy = 0, shz = 0;
  if (hist) {
    shx = sg * sh[0]; shy = sg * sh[1]; shz = sg * sh[2];
    if (update) {
      // keep the prior displacement (or elastic force) in the tangent plane, rescaled to its previous magnitude
      const mag0 = Math.hypot(shx, shy, shz);
      const sn = shx * nx + shy * ny + shz * nz;
      let px = shx - sn * nx, py = shy - sn * ny, pz = shz - sn * nz;
      const mag1 = Math.hypot(px, py, pz);
      if (mag1 > 0) {
        const cr = mag0 / mag1;
        px *= cr; py *= cr; pz *= cr;
      }
      shx = px; shy = py; shz = pz;
      // mindlin_rescale: on unloading (a < a_prev) the stored value is scaled by a / a_prev
      const aPrev = sh[3];
      if (rescale && aPrev > 0 && aC < aPrev) {
        const cr = aC / aPrev;
        shx *= cr; shy *= cr; shz *= cr;
      }
      if (forceVariant) {
        shx -= keff * vtx * dt; shy -= keff * vty * dt; shz -= keff * vtz * dt;
      } else {
        shx += vtx * dt; shy += vty * dt; shz += vtz * dt;
      }
    }
  }
  // spring force on the particle: F_s = -k_eff xi (displacement models) or F_te (force models)
  const spring = (q: number) => poly * (forceVariant ? q : -keff * q);
  let ftx = spring(shx) + dampx, fty = spring(shy) + dampy, ftz = spring(shz) + dampz;
  const ft = Math.hypot(ftx, fty, ftz);
  const fcap = pm.mu * fnAbs;
  if (ft > fcap) {
    let scale = ft > 0 ? fcap / ft : 0;
    if (hist) {
      // with no stored displacement (a contact at run setup) the capped force is dropped
      if (shx === 0 && shy === 0 && shz === 0) scale = 0;
      else if (keff > 0) {
        if (forceVariant) {
          // F_te = s (F_te + F_damp) - F_damp
          shx = scale * (shx + dampx) - dampx;
          shy = scale * (shy + dampy) - dampy;
          shz = scale * (shz + dampz) - dampz;
        } else {
          const cr = etaT / keff;
          shx = scale * (shx + cr * vtx) - cr * vtx;
          shy = scale * (shy + cr * vty) - cr * vty;
          shz = scale * (shz + cr * vtz) - cr * vtz;
        }
      }
    }
    ftx *= scale; fty *= scale; ftz *= scale;
  }
  // rolling sds: pseudo-force F_roll = s (-k_r xi_r - gamma_r v_roll), acting as a torque only (measured sign)
  let rlx = 0, rly = 0, rlz = 0;
  if (pm.roll === 'sds') {
    const wdx = c.oix - c.ojx, wdy = c.oiy - c.ojy, wdz = c.oiz - c.ojz;
    // v_roll = -R (w_i - w_j) x n, with R the effective radius
    const vrlx = -Rf * (wdy * nz - wdz * ny), vrly = -Rf * (wdz * nx - wdx * nz), vrlz = -Rf * (wdx * ny - wdy * nx);
    let qx = sh[4], qy = sh[5], qz = sh[6];
    if (update) {
      // prior rolling displacement rotated into the tangent plane, rescaled to its previous magnitude
      const mag0 = Math.hypot(qx, qy, qz);
      const sn = qx * nx + qy * ny + qz * nz;
      let px = qx - sn * nx, py = qy - sn * ny, pz = qz - sn * nz;
      const mag1 = Math.hypot(px, py, pz);
      if (mag1 > 0) {
        const cr = mag0 / mag1;
        px *= cr; py *= cr; pz *= cr;
      }
      qx = px + vrlx * dt; qy = py + vrly * dt; qz = pz + vrlz * dt;
    }
    const k = pm.kr, g = pm.gr;
    let fx0 = -k * qx - g * vrlx, fy0 = -k * qy - g * vrly, fz0 = -k * qz - g * vrlz;
    const f0 = Math.hypot(fx0, fy0, fz0);
    const cap = pm.mr * fnAbs;
    if (f0 > cap) {
      // measured as for the tangential force: with no stored displacement the capped rolling force is dropped
      const sc = f0 > 0 && (qx !== 0 || qy !== 0 || qz !== 0) ? cap / f0 : 0;
      if (k > 0 && sc > 0) {
        // F0 = -k xi - gamma v; capped: xi = s xi - gamma v (1 - s) / k
        const cr = -(g * (1 - sc)) / k;
        qx = sc * qx + cr * vrlx; qy = sc * qy + cr * vrly; qz = sc * qz + cr * vrlz;
      }
      fx0 *= sc; fy0 *= sc; fz0 *= sc;
    }
    sh[4] = qx; sh[5] = qy; sh[6] = qz;
    rlx = fx0; rly = fy0; rlz = fz0;
  }
  // twisting: torque along n, tau = clamp(-k xi - gamma Omega_tw); no force
  let twq = 0;
  if (pm.twist !== 'none') {
    let k: number, g: number, mu: number;
    if (pm.twist === 'sds') { k = pm.kw; g = pm.gw; mu = pm.mw; }
    else {
      // marshall: k_tw = 0.5 k_t a^2, gamma_tw = 0.5 eta_t a^2, mu_tw = 2/3 a mu_t
      k = 0.5 * pm.kt * aC * aC; g = 0.5 * etaT * aC * aC; mu = (2 / 3) * aC * pm.mu;
    }
    const om = (c.oix - c.ojx) * nx + (c.oiy - c.ojy) * ny + (c.oiz - c.ojz) * nz;
    let q = sh[7];
    if (update) q += om * dt;
    const t0 = -k * q - g * om;
    const cap = mu * fnAbs;
    let tau = t0;
    if (Math.abs(t0) > cap) {
      // doc: the capped twisting displacement is xi = (mu F sgn(Omega) - gamma Omega) / k, i.e. tau = -mu F sgn(Omega)
      tau = om > 0 ? -cap : cap;
      if (k > 0) q = -(tau + g * om) / k;
    }
    sh[7] = q;
    twq = tau;
  }
  if (hist) {
    sh[0] = sg * shx; sh[1] = sg * shy; sh[2] = sg * shz;
    sh[3] = aC;
  }
  out.fx = fnx + ftx; out.fy = fny + fty; out.fz = fnz + ftz;
  // torques: -(R - delta/2) n x F_t on each particle (contact point at the centre of the overlap)
  const cx = ny * ftz - nz * fty, cy = nz * ftx - nx * ftz, cz = nx * fty - ny * ftx;
  const ci = c.surfaceArm ? ri : ri - 0.5 * delta, cj = rj - 0.5 * delta;
  out.tix = -ci * cx; out.tiy = -ci * cy; out.tiz = -ci * cz;
  out.tjx = -cj * cx; out.tjy = -cj * cy; out.tjz = -cj * cz;
  // rolling torque on i: R n x F_roll (and its opposite on j); twisting torque tau n on i
  if (pm.roll === 'sds') {
    const rx = -Rf * (ny * rlz - nz * rly), ry = -Rf * (nz * rlx - nx * rlz), rz = -Rf * (nx * rly - ny * rlx);
    out.tix += rx; out.tiy += ry; out.tiz += rz;
    out.tjx -= rx; out.tjy -= ry; out.tjz -= rz;
  }
  if (twq !== 0) {
    out.tix += twq * nx; out.tiy += twq * ny; out.tiz += twq * nz;
    out.tjx -= twq * nx; out.tjy -= twq * ny; out.tjz -= twq * nz;
  }
  out.hist = hist;
  return true;
}

export class PairGranular extends Pair {
  readonly name = 'granular';
  virialFdotr = true;
  /** Global cutoff (pair_style granular cutoff), or null. */
  private globalCut: number | null = null;
  /** Explicit pair_coeff specs per type pair (i*nt+j, i <= j) and the resolved parameters (both orders). */
  private specs: (Spec | null)[] = [];
  private pm: (Pm | null)[] = [];
  private state: SimState | null = null;
  /** Group bit of fix freeze (StyleContext.freezeGroupBit), 0 when none. */
  private freezeBit = 0;
  /** Contact history per atom-ID pair "lo:hi": [sx, sy, sz] in the frame of the lower ID. */
  private shear = new Map<string, Float64Array>();

  settings(args: string[]): void {
    if (args.length > 1) throw new StyleError('usage: pair_style granular [cutoff]');
    this.globalCut = args.length === 1 ? parseNum(args[0], 'cutoff') : null;
    if (this.globalCut !== null && !(this.globalCut > 0)) throw new StyleError('pair_style granular: cutoff must be > 0');
    this.shear.clear();
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.specs = new Array(nt * nt).fill(null);
    this.pm = new Array(nt * nt).fill(null);
  }

  coeff(args: string[]): void {
    if (args.length < 3) throw new StyleError('usage: pair_coeff I J normal-model args tangential ...');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const sp = parseGranularSpec(args.slice(2));
    const nt = this.ntypes + 1;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) this.specs[i * nt + j] = { ...sp };
    }
    // "I J" given as e.g. "2 1" without wildcards still means the pair (1, 2)
    for (let i = ilo; i <= ihi; i++) for (let j = jlo; j <= jhi; j++) if (j < i) this.specs[j * nt + i] = { ...sp };
  }

  initStyle(ctx: StyleContext): void {
    this.state = ctx.s;
    this.freezeBit = ctx.freezeGroupBit ?? 0;
    if (!ctx.ghostVelocity) throw new StyleError('pair_style granular requires ghost atoms store velocity (use comm_modify vel yes)');
    const s = ctx.s;
    if (s && (!s.radius || !s.rmass || !s.omega)) throw new StyleError('pair_style granular requires atom_style sphere (radius, rmass, omega)');
    // pair_granular.html (mdr): "Newton's third law must be set to *off*." (measured with native
    // LAMMPS, black box: mdr with the default newton on stops with an error)
    if (ctx.newtonPair !== false && this.specs.some((sp) => sp?.normal === 'mdr')) {
      throw new StyleError("pair_style granular: normal model 'mdr' requires newton off");
    }
    this.resolve();
  }

  /** Resolves the parameters of every type pair: explicit coefficients, else mixing of like models. */
  private resolve(): void {
    const nt = this.ntypes + 1;
    const self = (t: number) => this.specs[t * nt + t];
    this.pm = new Array(nt * nt).fill(null);
    // doc: "The definition of multiple *mdr* models in the *pair_style* is currently not supported."
    let mdrRef: Spec | null = null;
    for (const sp of this.specs) {
      if (!sp || sp.normal !== 'mdr') continue;
      if (!mdrRef) mdrRef = sp;
      else if (sp.E !== mdrRef.E || sp.nu !== mdrRef.nu || sp.Y !== mdrRef.Y || sp.gamma !== mdrRef.gamma
        || sp.psib !== mdrRef.psib || sp.eta !== mdrRef.eta || sp.dtype !== mdrRef.dtype) {
        throw new StyleError("pair_style granular: multiple 'mdr' models (different E, nu, Y, Delta gamma, psi_b, eta_n0 or d_type) are not supported");
      }
    }
    if (mdrRef && this.specs.some((sp) => sp && sp.normal !== 'mdr')) {
      throw new StyleError("pair_style granular: the 'mdr' normal model cannot be combined with a different normal model in the pair_style");
    }
    for (let i = 1; i < nt; i++) {
      for (let j = i; j < nt; j++) {
        const sp = this.specs[i * nt + j];
        if (sp) {
          // the pair's own material values; the shear modulus for NULL k_t comes from the like-type pairs when set
          const si = self(i) ?? sp, sj = self(j) ?? sp;
          this.pm[i * nt + j] = this.pmOf(sp, sp.E ? effModulus(sp.E, sp.nu) : 0, si, sj);
          continue;
        }
        const a = self(i), b = self(j);
        if (!a || !b) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
        const sa = this.pmOf(a, a.E ? effModulus(a.E, a.nu) : 0, a, a);
        const sb = this.pmOf(b, b.E ? effModulus(b.E, b.nu) : 0, b, b);
        if (sa.sig !== sb.sig) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j}): models of types ${i} and ${j} differ and cannot be mixed`);
        const g = (x: number, y: number) => Math.sqrt(x * y);
        const Eeff = sa.Eeff && sb.Eeff ? (2 * sa.Eeff * sb.Eeff) / (sa.Eeff + sb.Eeff) : 0;
        // NULL k_t: k_t = 8 G_eff, G_eff = ((2-nu_i)/G_i + (2-nu_j)/G_j)^-1, G = E/(2(1+nu))
        const kt = a.ktNull
          ? 8 / ((2 - a.nu) / (a.E / (2 * (1 + a.nu))) + (2 - b.nu) / (b.E / (2 * (1 + b.nu))))
          : g(a.kt, b.kt);
        const mixed: Spec = { ...a, kn: g(a.kn, b.kn), E: a.E, nu: a.nu, gamma: g(a.gamma, b.gamma), eta: g(a.eta, b.eta),
          kt, ktNull: false, xgt: g(a.xgt, b.xgt), mu: g(a.mu, b.mu), kr: g(a.kr, b.kr), gr: g(a.gr, b.gr), mr: g(a.mr, b.mr),
          kw: g(a.kw, b.kw), gw: g(a.gw, b.gw), mw: g(a.mw, b.mw), cutoff: null };
        const p = this.pmOf(mixed, Eeff, a, b);
        this.pm[i * nt + j] = p;
      }
    }
    for (let i = 1; i < nt; i++) for (let j = 1; j < i; j++) this.pm[i * nt + j] = this.pm[j * nt + i];
  }

  /**
   * Resolved parameters of a wall/particle pair (fix wall/gran granular): the particle's own material
   * against a flat wall; E_eff = E / (2 (1 - nu^2)) as in the doc note for fix wall/gran.
   */
  wallParams(sp: Spec): Pm {
    // doc: "The *mdr* model currently only supports *fix wall/gran/region*, not *fix wall/gran*."
    if (sp.normal === 'mdr') throw new StyleError("fix wall/gran with normal model 'mdr' is not implemented in this engine (the doc supports fix wall/gran/region only)");
    return this.pmOf(sp, sp.E ? effModulus(sp.E, sp.nu) : 0, sp, sp);
  }

  /** Resolved parameters of a spec whose effective modulus is eff (kt NULL uses the given like-type specs). */
  private pmOf(sp: Spec, eff: number, si: Spec, sj: Spec): Pm {
    let kt = sp.kt;
    if (sp.ktNull) {
      const gi = si.E / (2 * (1 + si.nu)), gj = sj.E / (2 * (1 + sj.nu));
      if (!(gi > 0) || !(gj > 0)) throw new StyleError('pair_style granular: tangential k_t NULL needs E and nu for the types');
      kt = 8 / ((2 - si.nu) / gi + (2 - sj.nu) / gj);
    }
    const sig = [sp.normal, sp.damp, sp.tang, sp.roll, sp.twist, sp.limit ? 1 : 0].join('|');
    return {
      normal: sp.normal, kn: sp.kn, Eeff: eff, gamma: sp.gamma, eta: sp.eta, Y: sp.Y, psib: sp.psib, dtype: sp.dtype,
      damp: sp.damp, tang: sp.tang,
      kt, xgt: sp.xgt, mu: sp.mu, roll: sp.roll, kr: sp.kr, gr: sp.gr, mr: sp.mr,
      twist: sp.twist, kw: sp.kw, gw: sp.gw, mw: sp.mw, limit: sp.limit,
      cutoff: sp.cutoff ?? this.globalCut ?? -1, sig,
    };
  }

  private maxRadius(t: number): number {
    const s = this.state;
    if (!s || !s.radius) return 0;
    let m = 0;
    for (let k = 0; k < s.n; k++) if (s.type[k] === t && s.radius[k] > m) m = s.radius[k];
    return m;
  }

  /** Neighbor cutoff for a type pair: the global or per-pair cutoff, else the sum of the largest radii. */
  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const p = this.pm[i * nt + j];
    if (!p) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    if (p.cutoff > 0) return p.cutoff;
    const rs = this.maxRadius(i) + this.maxRadius(j);
    // jkr: contacts extend to the pull-off separation radsum - delta_c (delta_c < 0)
    if (p.normal === 'jkr' && rs > 0) {
      const rm = (this.maxRadius(i) * this.maxRadius(j)) / rs;
      return rs - jklPullOff(rm, p.Eeff, p.gamma);
    }
    return rs;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const list = pc.half!;
    const { x, f } = pc;
    const nb = pc.nb;
    const nall = pc.nall;
    const owner = nb.owner;
    const radius = s.radius!, rmass = s.rmass!, omega = s.omega!, v = s.v, id = s.id, type = s.type;
    const dt = s.dt;
    const update = pc.historyUpdate === true;
    const nt = this.ntypes + 1;
    const tq = new Float64Array(3 * nall);
    const seen = new Set<string>();
    // mdr: the active contacts (atom-ID pairs), checked for closed triangles after the loop
    const mdrPairs: Array<[number, number]> = [];
    const out = newContactOut();
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
        const pm = this.pm[type[oi] * nt + type[oj]];
        if (!pm) continue;
        const a = id[oi], b = id[oj];
        const flip = a > b;
        const key = flip ? `${b}:${a}` : `${a}:${b}`;
        const radsum = ri + rj;
        const jkr = pm.normal === 'jkr';
        if (rsq >= radsum * radsum && !jkr) {
          this.shear.delete(key);
          continue;
        }
        const r = Math.sqrt(rsq), rinv = 1 / r;
        const mj = rmass[oj];
        const delta = radsum - r;
        const Rf = (ri * rj) / radsum;
        // m_eff; a contact with a particle of the fix freeze group uses the other particle's mass (measured in gran.ts)
        let meff = (mi * mj) / (mi + mj);
        if (this.freezeBit) {
          if (s.mask[oi] & this.freezeBit) meff = mj;
          else if (s.mask[oj] & this.freezeBit) meff = mi;
        }
        const sh = this.shear.get(key) ?? new Float64Array(8);
        const ok = granularContact({
          pm, nx: dx * rinv, ny: dy * rinv, nz: dz * rinv, r, delta, Rf, ri, rj, meff,
          vrx: v[3 * oi] - v[3 * oj], vry: v[3 * oi + 1] - v[3 * oj + 1], vrz: v[3 * oi + 2] - v[3 * oj + 2],
          oix: omega[3 * oi], oiy: omega[3 * oi + 1], oiz: omega[3 * oi + 2],
          ojx: omega[3 * oj], ojy: omega[3 * oj + 1], ojz: omega[3 * oj + 2],
          dt, update, sg: flip ? -1 : 1, mdrDamp: update || s.step > 0,
        }, sh, out, this.shear.has(key));
        if (!ok) {
          this.shear.delete(key);
          continue;
        }
        if (out.hist) {
          seen.add(key);
          this.shear.set(key, sh);
        }
        if (pm.normal === 'mdr') mdrPairs.push([a, b]);
        f[3 * i] += out.fx; f[3 * i + 1] += out.fy; f[3 * i + 2] += out.fz;
        f[3 * j] -= out.fx; f[3 * j + 1] -= out.fy; f[3 * j + 2] -= out.fz;
        tq[3 * i] += out.tix; tq[3 * i + 1] += out.tiy; tq[3 * i + 2] += out.tiz;
        tq[3 * j] += out.tjx; tq[3 * j + 1] += out.tjy; tq[3 * j + 2] += out.tjz;
      }
    }
    if (this.shear.size) for (const key of this.shear.keys()) if (!seen.has(key)) this.shear.delete(key);
    mdrCheckTriangles(mdrPairs);
    nb.reverseSum(tq, 3, s.torque!);
  }

  dataCoeffs(): string[] | null { return null; }
  dataCoeffsIJ(): string[] | null { return null; }
}
