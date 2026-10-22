import { Pair, StyleError, tallyAtomXyz, typeBounds, type PairCompute, type PairSep, type StyleContext, pairCoeffArgs } from '../types';
import type { SimState } from '../../types';
import { NEIGHMASK } from '../../neighbor';
import { numeric, parseNum } from '../util';
import { gplHook } from '../../gplHooks';
import { HEAT_FLOW, HEAT_TEMPERATURE } from '../../atoms';
import { libm } from '../../libm';

/*
 * pair_style granular heat keyword (heat conduction) — docs.lammps.org/pair_granular.html:
 *   "The optional *heat* keyword enables heat conduction." The options currently supported are none, radius (k_s) and area (h_s).
 *   "If the *heat* keyword is not specified, the model defaults to *none*."
 *   "For *heat* *radius*, the heat Q conducted between two particles is given by" Q = 2 k_s a dT (formula outside quotes),
 *   "For *heat* *area*, the heat Q conducted between two particles is given by" Q = h_s A dT, A = pi a^2.
 *   dT is "the difference in the two particles' temperature" and "a is the radius of the contact and depends on the
 *   normal force model" (a = contactA of granularContact: sqrt(R_eff delta) for hooke, hertz and hertz/material).
 * Sign: heat flows into particle i with Q_i = 2 k_s a (T_j - T_i) (or h_s A (T_j - T_i)); j receives -Q_i.
 * Measured with native LAMMPS (black box): the step-0 setup value on atom 1 (T 1) against atom 2 (T 3) is +0.4472 for
 * radius 0.5 and +0.1571 for area 0.5 at contact delta 0.2, i.e. the sign above and a = sqrt(R_eff delta).
 * Measured with native LAMMPS (black box): mixed coefficients take the geometric mean of the heat coefficient
 * (1-1 area 0.5 and 2-2 area 1.5, no 1-2 line: Q matches h_12 = sqrt(0.5 * 1.5)).
 * Measured with native LAMMPS (black box): heat none on one type pair and heat area on another is accepted when
 * the cross pair is given explicitly; mixing two different heat models is refused.
 * Heatflow accumulates over the pair force evaluations and is zeroed by fix heat/flow (fix_heat_flow.ts): see that file.
 */

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
 * mdr: the normal model (plastic contact, adhesion, unequal radii, the bulk response, the closed-triangle penalty and fix
 * wall/gran/region) is the port in src/engine-gpl/granular/mdr (lazy hook 'mdr'); the tangential models, rolling and
 * twisting of this file act on its normal force. Not implemented (StyleError): mindlin tangential models and the marshall
 * twisting model with mdr, synchronized_verlet with mdr or mindlin, fix wall/gran (the plane walls) with mdr.
 * Not implemented (StyleError): rolling or twisting with synchronized_verlet; synchronized_verlet with mdr or mindlin.
 * synchronized_verlet (docs.lammps.org/pair_granular.html): "The *synchronized_verlet* flag implements an alternate
 * Velocity-Verlet integration scheme", as detailed in Vyas et al. (2025; arXiv 2410.14798, section 5, Improved
 * velocity-Verlet integration). Its contact uses the half-step separation n' = normalize(r n - (dt/2) v_r) (formula ours, from the paper's
 * half-step normal from x(t+dt) and v(t+dt/2)) for the normal force direction, the normal damping velocity v_r.n' and the
 * tangential velocity in the plane n' (the rotation term uses the same n'). The tangential history is kept in the plane n:
 * the stored history is moved to the plane n', advanced by the half-step tangential velocity times dt, and returned to the
 * plane n with its magnitude; the tangential damping velocity is the half-step tangential velocity rotated to n with its
 * magnitude, on history steps only. Measured with native LAMMPS (black box, two spheres, hooke and hertz/material, dt 0.001
 * and 0.01): the step-0 setup force is along n' with the setup velocities (no half kick) and the setup tangential damping
 * uses the unrotated n'-plane velocity; the contact torque arm uses n'; the step-1 force matches these rules to 1e-14.
 * Measured with native LAMMPS (black box): the synchronized and standard forces differ by about 2e-5 relative at step 1
 * of the hooke pair (dt 0.01), so the flag changes the result.
 */

/**
 * The mdr normal model (Zunker and Kamrin 2024; pair_granular.html "mdr") is implemented in src/engine-gpl/granular/mdr,
 * behind the lazy hook 'mdr' (src/engine/gplHooks.ts, lazyStyles.ts group gpl_granular_mdr). The core only forwards the
 * contacts: the normal force and damping prefactor come from MdrApi, the tangential part stays here (granularContact).
 */
/** The coefficients of one mdr pair_coeff: E, nu, Y, Delta gamma, psi_b, eta_n0 and the damping class d_type. */
export interface MdrMaterialCore { E: number; nu: number; Y: number; gamma: number; psib: number; eta: number; dtype: number }

/** One particle pair (both orders of the atom IDs are allowed; the model keys its history by the lower ID). */
export interface MdrPairIn {
  ida: number; idb: number;
  /** Owned atom indices of the pair. */
  oi: number; oj: number;
  /** Apparent radii (the radius property, which the model's radius update changes). */
  ri: number; rj: number;
  /** Overlap radsum - r (gm->delta). */
  delta: number;
  /** Effective radius ri rj / (ri + rj). */
  Rf: number;
  /** Unit vector from j to i. */
  nx: number; ny: number; nz: number;
  meff: number;
  /** Advance the history (not at the run setup). */
  hu: boolean;
}

/** One wall contact of fix wall/gran/region with the mdr model. */
export interface MdrWallIn {
  /** History key of the contact (the wall, the contact and the atom). */
  key: string;
  ai: number;
  /** Apparent radius of the atom and its overlap with the wall surface. */
  R: number;
  delta: number;
  Rf: number;
  nx: number; ny: number; nz: number;
  meff: number;
  hu: boolean;
  /** The wall's own coefficients (its surface energy may differ from the pair's). */
  p: MdrMaterialCore;
}

/** The normal force of one contact (wij-weighted, both sides), the contact radius and the damping prefactor eta_n. */
export interface MdrNormalOut {
  fne: number;
  a: number;
  etaN: number;
  /** Contact area of the mdr model (native svector: Ac_avg / 2, the wall's per-atom data and the pair's pN value). */
  svector?: number;
}

/** Contacts of one wall for one atom: the history key and the distance to the surface (radius - overlap). */
export type WallContacts = (i: number) => Array<{ key: string; dist: number }>;

/** The mdr contact model of one pair style instance (implemented in src/engine-gpl/granular/mdr/model.ts). */
export interface MdrApi {
  /** fix GRANULAR/MDR pre_force, at the start of a force evaluation (hu: a step, not the run setup). */
  beginStep(s: SimState, pc: PairCompute, hu: boolean): void;
  pairNormal(c: MdrPairIn): MdrNormalOut;
  pairDrop(ida: number, idb: number): void;
  wallNormal(c: MdrWallIn): MdrNormalOut;
  wallDrop(key: string): void;
  wallProvider(name: string, contacts: WallContacts): void;
}

/** The lazy hook 'mdr' (src/engine-gpl/registry/granular_mdr.ts). */
export interface GranularMdrHook {
  create(mat: MdrMaterialCore): MdrApi;
}

/** The model and the pair coefficients of the pair style of a system (walls read them at their setup). */
const mdrRegistry = new WeakMap<SimState, { api: MdrApi; mat: MdrMaterialCore }>();
export const mdrPairOf = (s: SimState): { api: MdrApi; mat: MdrMaterialCore } | undefined => mdrRegistry.get(s);

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
  /** synchronized_verlet keyword (pair_granular.html): the half-step tangential update of the contact. */
  sync: boolean;
  cutoff: number | null;
  /** heat conduction model and its coefficient (k_s for radius, h_s for area). */
  heat: HeatKind; hk: number;
}

/** Heat conduction model of pair_coeff (pair_granular.html "heat" keyword). */
export type HeatKind = 'none' | 'radius' | 'area';

/** Heat flowing into particle i from particle j (or from a wall at temperature Tw): Q = 2 k_s a dT or h_s pi a^2 dT. */
export function heatFlowOf(heat: HeatKind, hk: number, a: number, dT: number): number {
  if (heat === 'radius') return 2 * hk * a * dT;
  if (heat === 'area') return hk * Math.PI * a * a * dT;
  return 0;
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
  /** synchronized_verlet: the contact uses the half-step plane for the tangential history and damping. */
  sync?: boolean;
  cutoff: number;
  heat: HeatKind; hk: number;
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
    roll: 'none', kr: 0, gr: 0, mr: 0, twist: 'none', kw: 0, gw: 0, mw: 0, limit: false, sync: false, cutoff: null,
    heat: 'none', hk: 0,
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
    // doc: the surface energy is "\Delta\gamma \ge 0" (the adhesive branch is in src/engine-gpl/granular/mdr)
    if (!(sp.gamma >= 0)) throw new StyleError("pair_style granular: normal model 'mdr' needs a surface energy Delta gamma >= 0");
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
    } else if (kw === 'synchronized_verlet') {
      sp.sync = true;
    } else if (kw === 'cutoff') {
      need(1, 'cutoff');
      sp.cutoff = parseNum(next(), 'cutoff');
    } else if (kw === 'heat') {
      // pair_granular.html: the heat models are none, radius and area (see the header of this file)
      need(1, 'heat');
      const hm = next();
      if (hm === 'none') sp.heat = 'none';
      else if (hm === 'radius' || hm === 'area') {
        need(1, `heat ${hm}`);
        sp.heat = hm;
        sp.hk = parseNum(next(), `heat ${hm} coefficient`);
        // native refuses a negative coefficient (measured: heat area -0.5 stops with Illegal area heat model)
        if (!(sp.hk >= 0)) throw new StyleError(`pair_coeff granular: heat ${hm} coefficient must be >= 0 (got ${sp.hk})`);
      } else throw new StyleError(`pair_coeff granular: unknown heat model '${hm}' (none, radius, area)`);
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
  if (sp.sync && (sp.roll !== 'none' || sp.twist !== 'none')) {
    throw new StyleError('pair_style granular: synchronized_verlet with rolling or twisting is not implemented in this engine');
  }
  if (sp.sync && normal === 'mdr') throw new StyleError("pair_style granular: synchronized_verlet with normal model 'mdr' is not implemented in this engine");
  if (sp.sync && sp.tang !== 'linear_nohistory' && sp.tang !== 'linear_history') {
    throw new StyleError(`pair_style granular: synchronized_verlet with tangential ${sp.tang} is not implemented in this engine`);
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

/**
 * Projects v onto the plane normal to the unit vector m and rescales the projection to the magnitude mag; the zero
 * vector when the projection vanishes.
 */
function planeMap(vx: number, vy: number, vz: number, mx: number, my: number, mz: number, mag: number): [number, number, number] {
  const d = vx * mx + vy * my + vz * mz;
  const px = vx - d * mx, py = vy - d * my, pz = vz - d * mz;
  const pn = Math.hypot(px, py, pz);
  if (!(pn > 0)) return [0, 0, 0];
  const sc = mag / pn;
  return [px * sc, py * sc, pz * sc];
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
  /** The mdr normal model's force and damping prefactor (the normal force replaces the elastic law above). */
  normal?: MdrNormalOut;
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
  /** Contact radius a of the normal model (the heat models use it). */
  a: number;
  /** Native single() values: the normal force Fntot (normal plus damping) on i, the tangential force on i, the rolling force, the twisting torque. */
  fntot: number;
  fsx: number; fsy: number; fsz: number;
  frx: number; fry: number; frz: number;
  mtt: number;
}

export const newContactOut = (): ContactOut => ({ fx: 0, fy: 0, fz: 0, tix: 0, tiy: 0, tiz: 0, tjx: 0, tjy: 0, tjz: 0, hist: false, a: 0, fntot: 0, fsx: 0, fsy: 0, fsz: 0, frx: 0, fry: 0, frz: 0, mtt: 0 });

/** Native EPSILON of the frame-update tests (GRANULAR/gran_sub_mod_tangential.cpp and gran_sub_mod_rolling.cpp: 1e-10). */
const FRAME_EPS = 1e-10;

/**
 * A history vector moved into the tangent plane of the new normal n (native GranSubMod::rotate_rescale_vec: remove the normal
 * part, rescale to the previous magnitude), done only when native's frame-update test holds: |v . n| k > FRAME_EPS F_crit
 * (sn = v . n; F_crit = mu |F_n|, the tangential or rolling Fscrit/Frcrit). Otherwise the vector is returned unchanged.
 */
function frameRotate(vx: number, vy: number, vz: number, nx: number, ny: number, nz: number, sn: number, k: number, fcrit: number): [number, number, number] {
  if (!(Math.abs(sn) * k > FRAME_EPS * fcrit)) return [vx, vy, vz];
  const mag0 = Math.hypot(vx, vy, vz);
  let px = vx - sn * nx, py = vy - sn * ny, pz = vz - sn * nz;
  const mag1 = Math.hypot(px, py, pz);
  if (mag1 > 0) {
    const cr = mag0 / mag1;
    px *= cr; py *= cr; pz *= cr;
  }
  return [px, py, pz];
}

/**
 * One granular contact: normal (hooke/hertz/hertz-material/dmt/jkr), damping, tangential (linear, mindlin),
 * rolling sds and twisting. sh holds the history: [0..2] tangential displacement (or elastic force), [3] the
 * previous contact radius, [4..6] rolling displacement, [7] twisting displacement; it is read and written in
 * the frame given by sg. Returns false when there is no contact (the caller drops the history record).
 */
export function granularContact(c: ContactIn, sh: Float64Array, out: ContactOut, hadHistory: boolean): boolean {
  const { pm, nx, ny, nz, delta, Rf, ri, rj, meff, dt, update, sg } = c;
  const jkr = pm.normal === 'jkr';
  // jkr hysteresis (doc): the tensile range delta < 0 applies only once the pair has been in contact
  if (jkr && delta < 0 && !hadHistory) return false;
  const sqrtR = Math.sqrt(Rf);
  const vrx = c.vrx, vry = c.vry, vrz = c.vrz;
  // synchronized_verlet (see the header): the normal force direction, the normal damping velocity and the tangential
  // velocity use the half-step plane n' built from the half-step separation (also at setup, where the measured setup force
  // is along n' with the setup velocities); the elastic normal magnitude keeps delta at the full step.
  const sync = pm.sync === true;
  let hx = nx, hy = ny, hz = nz;
  if (sync) {
    const ex = c.r * nx - 0.5 * dt * vrx, ey = c.r * ny - 0.5 * dt * vry, ez = c.r * nz - 0.5 * dt * vrz;
    const el = Math.hypot(ex, ey, ez);
    if (el > 0) { hx = ex / el; hy = ey / el; hz = ez / el; }
  }
  const vn = vrx * hx + vry * hy + vrz * hz; // (v_i - v_j) . n (n' when synchronized)
  const vnx = hx * vn, vny = hy * vn, vnz = hz * vn;
  // (R_i w_i + R_j w_j) x n
  const Ax = ri * c.oix + rj * c.ojx;
  const Ay = ri * c.oiy + rj * c.ojy;
  const Az = ri * c.oiz + rj * c.ojz;
  const wnx = Ay * hz - Az * hy, wny = Az * hx - Ax * hz, wnz = Ax * hy - Ay * hx;
  const vtx = vrx - vnx - wnx, vty = vry - vny - wny, vtz = vrz - vnz - wnz;
  // normal force along n (positive repulsive): elastic part, then damping
  let fne = 0;
  let contactA = 0;
  // pull-off force of the cohesive models (doc: F_pulloff = 4 pi gamma R for dmt, 3 pi gamma R for jkr)
  const fPull = pm.normal === 'dmt' ? 4 * Math.PI * pm.gamma * Rf : jkr ? 3 * Math.PI * pm.gamma * Rf : 0;
  if (c.normal) {
    // mdr: the normal model is evaluated by the MDR hook (src/engine-gpl/granular/mdr); it has its own contact test
    fne = c.normal.fne;
    contactA = c.normal.a;
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
  // the mdr damping prefactor comes with the normal force (c.normal.etaN)
  let etaN = c.normal ? c.normal.etaN : 0;
  if (!c.normal) switch (pm.damp) {
    case 'velocity': etaN = pm.eta; break;
    case 'mass_velocity': etaN = pm.eta * meff; break;
    case 'viscoelastic': etaN = pm.eta * contact * meff; break;
    case 'coeff_restitution': {
      // doc: eta_n = sqrt(4 m_eff k_nd / (1 + (pi / log e)^2)) for hooke; otherwise
      // eta_n = -2 sqrt(5/6) log(e) / sqrt(pi^2 + log(e)^2) * sqrt(3/2 k_nd m_eff); k_nd = F_elastic / delta.
      // Measured with native LAMMPS (black box): both forms match with no extra factor (ratio 1.0 for hooke and hertz).
      const knd = fne / delta;
      const le = libm.log(pm.eta);
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
  // synchronized: the normal force acts along the half-step normal n' (measured with native LAMMPS, see the header)
  const fnx = fn * hx, fny = fn * hy, fnz = fn * hz;
  // tangential force: damping from eta_t = x_gamma_t eta_n; the spring is -k_t xi (linear_history),
  // -k_t a xi with a = sqrt(R delta) (mindlin), or the stored elastic force F_te (mindlin/force)
  const etaT = pm.xgt * etaN;
  // the poly factor scales the forces (not the displacement update, and not the history-cap coefficient)
  const etaTf = etaT * poly;
  // synchronized, history steps: the half-step tangential velocity is rotated to the full-step plane n and keeps its magnitude
  // (at setup it is used unrotated, measured with native LAMMPS)
  const [vdx, vdy, vdz] = sync && update ? planeMap(vtx, vty, vtz, nx, ny, nz, Math.hypot(vtx, vty, vtz)) : [vtx, vty, vtz];
  const dampx = -etaTf * vdx, dampy = -etaTf * vdy, dampz = -etaTf * vdz;
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
    if (sync && update) {
      // synchronized history: the prior history is kept in the plane n' with its magnitude, advanced by the half-step
      // tangential velocity, and returned to the plane n with the same magnitude
      const [px, py, pz] = planeMap(shx, shy, shz, hx, hy, hz, Math.hypot(shx, shy, shz));
      const ux = px + vtx * dt, uy = py + vty * dt, uz = pz + vtz * dt;
      [shx, shy, shz] = planeMap(ux, uy, uz, nx, ny, nz, Math.hypot(ux, uy, uz));
    } else if (update) {
      // keep the prior displacement (or elastic force) in the tangent plane, rescaled to its previous magnitude (frameRotate)
      const sn = shx * nx + shy * ny + shz * nz;
      [shx, shy, shz] = frameRotate(shx, shy, shz, nx, ny, nz, sn, keff, fnAbs * pm.mu);
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
          shx = scale * (shx + cr * vdx) - cr * vdx;
          shy = scale * (shy + cr * vdy) - cr * vdy;
          shz = scale * (shz + cr * vdz) - cr * vdz;
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
    const k = pm.kr, g = pm.gr;
    if (update) {
      // prior rolling displacement rotated into the tangent plane (frameRotate), then the step's rolling velocity
      const sn = qx * nx + qy * ny + qz * nz;
      const [px, py, pz] = frameRotate(qx, qy, qz, nx, ny, nz, sn, k, fnAbs * pm.mr);
      qx = px + vrlx * dt; qy = py + vrly * dt; qz = pz + vrlz * dt;
    }
    let fx0 = -k * qx - g * vrlx, fy0 = -k * qy - g * vrly, fz0 = -k * qz - g * vrlz;
    const f0 = Math.hypot(fx0, fy0, fz0);
    const cap = pm.mr * fnAbs;
    if (f0 > cap) {
      // measured as for the tangential force: with no stored displacement the capped rolling force is dropped
      const sc = f0 > 0 && (qx !== 0 || qy !== 0 || qz !== 0) ? cap / f0 : 0;
      // native (GranSubModRollingSDS::calculate_forces, the capped branch with a non-zero displacement): the displacement
      // is rebuilt even when the cap is zero (s = 0 leaves -gamma v / k), and only a zero displacement is left alone
      if (k > 0 && (qx !== 0 || qy !== 0 || qz !== 0)) {
        // F0 = -k xi - gamma v; capped: xi = s xi - gamma v (1 - s) / k
        const cr = -(g * (1 - sc)) / k;
        qx = sc * qx + cr * vrlx; qy = sc * qy + cr * vrly; qz = sc * qz + cr * vrlz;
      }
      fx0 *= sc; fy0 *= sc; fz0 *= sc;
    }
    // native GranSubModRollingSDS::calculate_forces (GRANULAR/gran_sub_mod_rolling.cpp, b332f28) writes the displacement
    // back only "if (history_update)": the capped rebuild above is local to the force of a setup or force-only evaluation
    // (history_update = 0), so the stored displacement is kept for the next full step
    if (update) { sh[4] = qx; sh[5] = qy; sh[6] = qz; }
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
  const cx = hy * ftz - hz * fty, cy = hz * ftx - hx * ftz, cz = hx * fty - hy * ftx;
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
  out.a = contactA;
  out.fntot = fn; out.fsx = ftx; out.fsy = fty; out.fsz = ftz;
  // the rolling force is reported with native's sign (the engine's pseudo-force is the negative of native fr; its torque uses it so)
  out.frx = -rlx; out.fry = -rly; out.frz = -rlz; out.mtt = twq;
  return true;
}

export class PairGranular extends Pair {
  readonly name = 'granular';
  /** As native LAMMPS does (src/GRANULAR/pair_granular.cpp at b332f28: no_virial_fdotr_compute = 1): the virial is
   * tallied contact by contact (ev_tally_xyz with dx = x_i - x_j). */
  virialFdotr = false;
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
  /** Some pair_coeff has a heat model (the temperature and heatflow properties are then needed). */
  private heatOn = false;
  /** The mdr contact model when a pair_coeff uses the mdr normal model (src/engine-gpl/granular/mdr). */
  private mdr: MdrApi | null = null;

  settings(args: string[]): void {
    // As native LAMMPS does (src/GRANULAR/pair_granular.cpp PairGranular::settings at b332f28): one word is the global
    // cutoff (utils::numeric); any other count of words leaves the cutoffs to the particle sizes
    this.globalCut = args.length === 1 ? numeric(args[0]) : null;
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
    if (args.length < 3) throw pairCoeffArgs('usage: pair_coeff I J normal-model args tangential ...');
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
    if (!ctx.ghostVelocity) throw new StyleError('Pair granular requires ghost atoms store velocity\n(use comm_modify vel yes)');
    const s = ctx.s;
    if (s && (!s.radius || !s.rmass || !s.omega)) throw new StyleError('pair_style granular requires atom_style sphere (radius, rmass, omega)');
    // pair_granular.html (mdr): "Newton's third law must be set to *off*." (measured with native
    // LAMMPS, black box: mdr with the default newton on stops with an error)
    if (ctx.newtonPair !== false && this.specs.some((sp) => sp?.normal === 'mdr')) {
      throw new StyleError("pair_style granular: normal model 'mdr' requires newton off");
    }
    // a size list (Pair.sizeList) unless a model acts beyond contact: as native LAMMPS does (src/GRANULAR/pair_granular.cpp
    // at b332f28: "if (!beyond_contact) req_flags |= NeighConst::REQ_SIZE;"), where only the jkr normal model sets
    // beyond_contact (src/GRANULAR/gran_sub_mod_normal.cpp)
    this.sizeList = !this.specs.some((sp) => sp?.normal === 'jkr');
    this.heatOn = this.specs.some((sp) => sp && sp.heat !== 'none');
    if (this.heatOn && (!s || !s.custom.has(HEAT_TEMPERATURE) || !s.custom.has(HEAT_FLOW))) {
      throw new StyleError('pair_style granular: heat conduction (heat radius or area) needs fix property/atom temperature heatflow');
    }
    this.resolve();
    // mdr: the model is created by the lazy hook and registered for the walls of this system
    const mdrSp = this.specs.find((x) => x?.normal === 'mdr');
    this.mdr = null;
    if (mdrSp) {
      if (!s) throw new StyleError("pair_style granular: normal model 'mdr' needs a box (create_box or read_data first)");
      const mat: MdrMaterialCore = { E: mdrSp.E, nu: mdrSp.nu, Y: mdrSp.Y, gamma: mdrSp.gamma, psib: mdrSp.psib, eta: mdrSp.eta, dtype: mdrSp.dtype };
      // the contact histories and the fix state live as long as the system: a run's init keeps the model of an unchanged material
      const prev = mdrRegistry.get(s);
      const same = prev && (Object.keys(mat) as Array<keyof MdrMaterialCore>).every((k) => prev.mat[k] === mat[k]);
      const api = same ? prev.api : gplHook<GranularMdrHook>('mdr').create(mat);
      this.mdr = api;
      if (!same) mdrRegistry.set(s, { api, mat });
    }
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
        // a pair that cannot be mixed stays unset: initOne throws for it when this style is used for the pair, and a
        // hybrid sub-style never asks for the pairs of other sub-styles (pair_style hybrid granular surf/granular)
        if (!a || !b) continue;
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
          kw: g(a.kw, b.kw), gw: g(a.gw, b.gw), mw: g(a.mw, b.mw), cutoff: null, hk: g(a.hk, b.hk) };
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
  wallParams(sp: Spec, region = false): Pm {
    // doc: "The *mdr* model currently only supports *fix wall/gran/region*, not *fix wall/gran*."
    if (sp.normal === 'mdr' && !region) throw new StyleError("fix wall/gran with normal model 'mdr' is not implemented in this engine (the doc supports fix wall/gran/region only)");
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
    const sig = [sp.normal, sp.damp, sp.tang, sp.roll, sp.twist, sp.limit ? 1 : 0, sp.sync ? 1 : 0, sp.heat].join('|');
    return {
      normal: sp.normal, kn: sp.kn, Eeff: eff, gamma: sp.gamma, eta: sp.eta, Y: sp.Y, psib: sp.psib, dtype: sp.dtype,
      damp: sp.damp, tang: sp.tang,
      kt, xgt: sp.xgt, mu: sp.mu, roll: sp.roll, kr: sp.kr, gr: sp.gr, mr: sp.mr,
      twist: sp.twist, kw: sp.kw, gw: sp.gw, mw: sp.mw, limit: sp.limit, sync: sp.sync,
      cutoff: sp.cutoff ?? this.globalCut ?? -1, heat: sp.heat, hk: sp.hk, sig,
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
    // pair_style gran/* and fix wall/gran replace the mass of an atom in a rigid body by the body mass
    // (docs.lammps.org/pair_gran.html, see atoms.ts contactMassArray); pair_granular.html states no such rule
    // and it was not measured for this style, so a rigid body with pair_style granular is refused.
    if (s.rigidBodyMass && s.rigidBodyMass.size > 0) {
      throw new StyleError('pair_style granular with atoms in rigid bodies is not supported by the browser engine (the contact mass rule for rigid bodies is not documented for this style and was not measured)');
    }
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
    // mdr: the fix's pre-force update (per-atom state, penalty, mean surface displacement) before the contacts
    const mdr = this.mdr;
    if (mdr) mdr.beginStep(s, pc, update);
    const out = newContactOut();
    // heat conduction: heatflow accumulates (zeroed by fix heat/flow), Q_i = heat(a, T_j - T_i) and Q_j = -Q_i
    const T = this.heatOn ? s.custom.get(HEAT_TEMPERATURE)!.data : null;
    const HF = this.heatOn ? s.custom.get(HEAT_FLOW)!.data : null;
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
          if (mdr) mdr.pairDrop(a, b);
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
        const normal = mdr && pm.normal === 'mdr'
          ? mdr.pairNormal({ ida: a, idb: b, oi, oj, ri, rj, delta, Rf, nx: dx * rinv, ny: dy * rinv, nz: dz * rinv, meff, hu: update })
          : undefined;
        const ok = granularContact({
          pm, nx: dx * rinv, ny: dy * rinv, nz: dz * rinv, r, delta, Rf, ri, rj, meff,
          vrx: v[3 * oi] - v[3 * oj], vry: v[3 * oi + 1] - v[3 * oj + 1], vrz: v[3 * oi + 2] - v[3 * oj + 2],
          oix: omega[3 * oi], oiy: omega[3 * oi + 1], oiz: omega[3 * oi + 2],
          ojx: omega[3 * oj], ojy: omega[3 * oj + 1], ojz: omega[3 * oj + 2],
          dt, update, sg: flip ? -1 : 1, normal,
        }, sh, out, this.shear.has(key));
        if (!ok) {
          this.shear.delete(key);
          continue;
        }
        if (out.hist) {
          seen.add(key);
          this.shear.set(key, sh);
        }
        if (T && pm.heat !== 'none') {
          const q = heatFlowOf(pm.heat, pm.hk, out.a, T[oj] - T[oi]);
          HF![oi] += q;
          // measured with native LAMMPS (black box): a periodic image partner (j >= the owned atom count) does not get
          // its share back, so the heat of a contact across a periodic boundary is counted once, on the owned side
          if (j < s.n) HF![oj] -= q;
        }
        f[3 * i] += out.fx; f[3 * i + 1] += out.fy; f[3 * i + 2] += out.fz;
        f[3 * j] -= out.fx; f[3 * j + 1] -= out.fy; f[3 * j + 2] -= out.fz;
        // per-atom virial of the contact, as native LAMMPS does (src/GRANULAR/pair_granular.cpp compute at b332f28:
        // ev_tally_xyz with no energy and dx = x_i - x_j)
        const vir = pc.acc.virial;
        vir[0] += dx * out.fx; vir[1] += dy * out.fy; vir[2] += dz * out.fz;
        vir[3] += dx * out.fy; vir[4] += dx * out.fz; vir[5] += dy * out.fz;
        if (pc.vatom) tallyAtomXyz(pc, i, j, 0, out.fx, out.fy, out.fz, dx, dy, dz);
        tq[3 * i] += out.tix; tq[3 * i + 1] += out.tiy; tq[3 * i + 2] += out.tiz;
        tq[3 * j] += out.tjx; tq[3 * j + 1] += out.tjy; tq[3 * j + 2] += out.tjz;
      }
    }
    if (this.shear.size) for (const key of this.shear.keys()) if (!seen.has(key)) this.shear.delete(key);
    nb.reverseSum(tq, 3, s.torque!);
  }

  /**
   * Native single() (GRANULAR/pair_granular.cpp, single_extra = 12 plus the sub-model values, docs.lammps.org/pair_granular.html:
   * "The first 3 are the components of the tangential force between particles I and J, acting on particle I. The fourth is the
   * magnitude of this tangential force. The next 3 (5-7) are the components of the rolling torque acting on particle I. The next
   * entry (8) is the magnitude of the rolling torque. The next entry (9) is the magnitude of the twisting torque acting about the
   * vector connecting the two particle centers. The next 3 (10-12) are the components of the vector connecting the centers of the
   * two particles (x_I - x_J)."). Then, "First, any extra values from the normal sub-model are appended followed by the damping,
   * tangential, rolling, twisting, then heat models." The contact is evaluated with the stored history read only (native sets
   * history_update = 0); fforce is Fntot / r and the energy is 0 ("The single() function of these pair styles returns 0.0 for the
   * energy"). Pairs without contact give zeros.
   */
  single(i: number, j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, del?: PairSep) {
    const extra = this.singleExtra();
    const svector = new Float64Array(extra);
    const none = { eng: 0, fforce: 0, svector };
    const s = this.state;
    if (!s) throw new StyleError('pair granular single(): the pair style has no state (run setup first)');
    // the contact normal is x_i - x_j: a caller that does not pass the separation (stress/prof, tally) is refused, not guessed
    if (!del) throw new StyleError('pair granular single() needs the separation vector x_i - x_j (compute pair/local passes it; this compute does not)');
    if (factorLJ === 0) return none;
    const nt = this.ntypes + 1;
    const pm = this.pm[itype * nt + jtype];
    if (!pm) return none;
    const radius = s.radius!, rmass = s.rmass!, omega = s.omega!, v = s.v, id = s.id;
    const ri = radius[i], rj = radius[j];
    const radsum = ri + rj;
    const jkr = pm.normal === 'jkr';
    if (rsq >= radsum * radsum && !jkr) return none;
    const r = Math.sqrt(rsq), rinv = 1 / r;
    const delta = radsum - r;
    const Rf = (ri * rj) / radsum;
    const mi = rmass[i], mj = rmass[j];
    let meff = (mi * mj) / (mi + mj);
    if (this.freezeBit) {
      if (s.mask[i] & this.freezeBit) meff = mj;
      else if (s.mask[j] & this.freezeBit) meff = mi;
    }
    const a = id[i], b = id[j];
    const flip = a > b;
    const key = flip ? `${b}:${a}` : `${a}:${b}`;
    const stored = this.shear.get(key);
    // a copy: the contact code writes its history arrays even without an update
    const sh = stored ? Float64Array.from(stored) : new Float64Array(8);
    const nx = del.dx * rinv, ny = del.dy * rinv, nz = del.dz * rinv;
    const normal = this.mdr && pm.normal === 'mdr'
      ? this.mdr.pairNormal({ ida: a, idb: b, oi: i, oj: j, ri, rj, delta, Rf, nx, ny, nz, meff, hu: false })
      : undefined;
    const out = newContactOut();
    const ok = granularContact({
      pm, nx, ny, nz, r, delta, Rf, ri, rj, meff,
      vrx: v[3 * i] - v[3 * j], vry: v[3 * i + 1] - v[3 * j + 1], vrz: v[3 * i + 2] - v[3 * j + 2],
      oix: omega[3 * i], oiy: omega[3 * i + 1], oiz: omega[3 * i + 2],
      ojx: omega[3 * j], ojy: omega[3 * j + 1], ojz: omega[3 * j + 2],
      dt: s.dt, update: false, sg: flip ? -1 : 1, normal,
    }, sh, out, this.shear.has(key));
    if (!ok) return none;
    svector[0] = out.fsx; svector[1] = out.fsy; svector[2] = out.fsz;
    svector[3] = Math.sqrt(out.fsx * out.fsx + out.fsy * out.fsy + out.fsz * out.fsz);
    svector[4] = out.frx; svector[5] = out.fry; svector[6] = out.frz;
    svector[7] = Math.sqrt(out.frx * out.frx + out.fry * out.fry + out.frz * out.frz);
    svector[8] = out.mtt;
    svector[9] = del.dx; svector[10] = del.dy; svector[11] = del.dz;
    // normal sub-model (mdr: the contact area), then the heat sub-model (the heat flow into i)
    let k = 12;
    if (normal && pm.normal === 'mdr') svector[k++] = normal.svector ?? 0;
    if (this.heatOn && pm.heat !== 'none') {
      const T = s.custom.get(HEAT_TEMPERATURE)!.data;
      svector[k++] = heatFlowOf(pm.heat, pm.hk, out.a, T[j] - T[i]);
    }
    return { eng: 0, fforce: out.fntot * rinv, svector };
  }

  /**
   * Native single_extra (pair_granular.cpp: 12, raised to 12 + the most extra quantities of one model when a pair_coeff uses the
   * mdr normal model (its contact area) or a heat model): the pN values that compute pair/local can read.
   */
  singleExtra(): number {
    let extra = 0;
    for (const p of this.pm) if (p) extra = Math.max(extra, (p.normal === 'mdr' ? 1 : 0) + (p.heat !== 'none' ? 1 : 0));
    return 12 + extra;
  }

  dataCoeffs(): string[] | null { return null; }
  dataCoeffsIJ(): string[] | null { return null; }
}
