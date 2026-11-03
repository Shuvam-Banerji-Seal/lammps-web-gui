import { Pair, StyleError, tallyAtomXyz, typeBounds, type PairCompute, type PairSep, type StyleContext, pairCoeffArgs } from '../types';
import type { SimState } from '../../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { numeric, parseNum } from '../util';
import { gplHook } from '../../gplHooks';
import type { PairLocalHook, PairOrderSource } from '../../compute/local';
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
  /** History key of the contact: the wall and its face ("fixID|face"), and the atom ID. */
  face: string;
  aid: number;
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
export type WallContacts = (i: number) => Array<{ face: string; aid: number; dist: number }>;

/** The mdr contact model of one pair style instance (implemented in src/engine-gpl/granular/mdr/model.ts). */
export interface MdrApi {
  /** fix GRANULAR/MDR pre_force, at the start of a force evaluation (hu: a step, not the run setup). */
  beginStep(s: SimState, pc: PairCompute, hu: boolean): void;
  pairNormal(c: MdrPairIn): MdrNormalOut;
  pairDrop(ida: number, idb: number): void;
  wallNormal(c: MdrWallIn): MdrNormalOut;
  wallDrop(face: string, aid: number): void;
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
/**
 * linear_history_classic: the tangential model of the classic fstyles (fix wall/gran/region hooke/history and
 * hertz/history), which updates and projects its displacement as native GranSubModTangentialLinearHistoryClassic does.
 */
type TangKind = 'linear_nohistory' | 'linear_history' | 'linear_history_classic' | 'mindlin_classic' | 'mindlin' | 'mindlin/force' | 'mindlin_rescale' | 'mindlin_rescale/force';
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
  /** Effective modulus E_eff (Emix: mix_stiffnessE of the pair, or mix_stiffnessE_wall for a wall). */
  Eeff: number;
  /** k of hertz/material, dmt and jkr: FOURTHIRDS Emix (gran_sub_mod_normal.cpp coeffs_to_local and mix_coeffs). */
  kHM: number;
  /** damp of the tsuji and coeff_restitution damping (gran_sub_mod_damping.cpp init), 0 for the others. */
  dampC: number;
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

/** Parses the arguments of one pair_coeff (after I J). */
export function parseGranularSpec(args: string[]): Spec {
  // As native LAMMPS does (src/GRANULAR/granular_model.cpp GranularModel::add_sub_model and construct_sub_model at
  // b332f28): each sub-model is a name ("Must specify granular sub model name" when there is none, "Illegal model type"
  // when it is unknown) and its coefficients ("Insufficient arguments provided for <name> model"), read by
  // utils::numeric
  const word = args[0];
  if (word === undefined) throw new StyleError('Must specify granular sub model name');
  if (!(word in NUM_ARGS)) throw new StyleError(`Illegal model type ${word}`);
  const normal = word as NormalKind;
  const nn = NUM_ARGS[normal];
  if (args.length < 1 + nn) throw new StyleError(`Insufficient arguments provided for ${normal} model`);
  const num = args.slice(1, 1 + nn).map((w) => numeric(w));
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
  // a sub-model keyword (damping, tangential, rolling, twisting, heat) with no name after it
  const needName = () => {
    if (args.length < k + 1) throw new StyleError('Must specify granular sub model name');
  };
  // As native LAMMPS does (src/GRANULAR/granular_model.cpp GranularModel::add_sub_model at b332f28): a sub-model with fewer
  // coefficients left than it needs is "Insufficient arguments provided for <model> model".
  const needModel = (n: number, model: string) => {
    if (args.length < k + n) throw new StyleError(`Insufficient arguments provided for ${model} model`);
  };
  const next = () => args[k++];
  while (k < args.length) {
    const kw = next();
    if (kw === 'damping') {
      needName();
      dampSeen = true;
      const d = next();
      if (d === 'mdr') {
        // doc: "The *mdr* damping class contains multiple damping models that can be toggled between by specifying different integer values for the :math:`d_{type}` input parameter."
        needModel(1, 'mdr');
        const dt = parseNum(next(), 'damping mdr d_type');
        if (dt !== 1 && dt !== 2) throw new StyleError(`pair_style granular: damping mdr d_type must be 1 or 2 (got ${dt})`);
        sp.damp = 'mdr';
        sp.dtype = dt;
        continue;
      }
      if (!(d in DAMPS)) throw new StyleError(`Illegal model type ${d}`);
      sp.damp = DAMPS[d];
    } else if (kw === 'tangential') {
      needName();
      const t = next();
      if (!(t in TANGS)) throw new StyleError(`Illegal model type ${t}`);
      sp.tang = TANGS[t];
      tangSeen = true;
      if (sp.tang === 'linear_nohistory') {
        needModel(2, t);
        sp.xgt = parseNum(next(), 'x_gamma_t');
        sp.mu = parseNum(next(), 'mu_s');
      } else {
        needModel(3, t);
        const ktw = next();
        if (ktw === 'NULL') {
          if (sp.tang === 'linear_history') throw new StyleError('pair_style granular: tangential linear_history needs a numeric k_t (NULL is for the mindlin models)');
          sp.ktNull = true;
        } else sp.kt = parseNum(ktw, 'k_t');
        sp.xgt = parseNum(next(), 'x_gamma_t');
        sp.mu = parseNum(next(), 'mu_s');
      }
    } else if (kw === 'rolling') {
      needName();
      const r = next();
      if (r === 'none') sp.roll = 'none';
      else if (r === 'sds') {
        needModel(3, 'sds');
        sp.roll = 'sds';
        sp.kr = parseNum(next(), 'k_roll');
        sp.gr = parseNum(next(), 'gamma_roll');
        sp.mr = parseNum(next(), 'mu_roll');
      } else throw new StyleError(`Illegal model type ${r}`);
    } else if (kw === 'twisting') {
      needName();
      const w = next();
      if (w === 'none') sp.twist = 'none';
      else if (w === 'marshall') sp.twist = 'marshall';
      else if (w === 'sds') {
        needModel(3, 'sds');
        sp.twist = 'sds';
        sp.kw = parseNum(next(), 'k_twist');
        sp.gw = parseNum(next(), 'gamma_twist');
        sp.mw = parseNum(next(), 'mu_twist');
      } else throw new StyleError(`Illegal model type ${w}`);
    } else if (kw === 'limit_damping') {
      sp.limit = true;
    } else if (kw === 'synchronized_verlet') {
      sp.sync = true;
    } else if (kw === 'cutoff') {
      if (args.length < k + 1) throw new StyleError('Illegal pair_coeff command, not enough parameters for cutoff keyword');
      sp.cutoff = parseNum(next(), 'cutoff');
    } else if (kw === 'heat') {
      // pair_granular.html: the heat models are none, radius and area (see the header of this file)
      needName();
      const hm = next();
      if (hm === 'none') sp.heat = 'none';
      else if (hm === 'radius' || hm === 'area') {
        needModel(1, hm);
        sp.heat = hm;
        sp.hk = parseNum(next(), `heat ${hm} coefficient`);
        // native refuses a negative coefficient (measured: heat area -0.5 stops with Illegal area heat model)
        // src/GRANULAR/gran_sub_mod_heat.cpp coeffs_to_local at b332f28
        if (sp.hk < 0.0) throw new StyleError(`Illegal ${hm} heat model`);
      } else throw new StyleError(`Illegal model type ${hm}`);
    } else {
      // GRANULAR/pair_granular.cpp PairGranular::coeff (native at b332f28): an unrecognised word stops the pair_coeff line
      throw new StyleError(`Illegal pair_coeff command ${kw}`);
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
  /** The squared separation and radius sum of the touch test (default r^2 and ri + rj), and x_i - x_j (synchronized). */
  rsq?: number; radsum?: number; dx?: number; dy?: number; dz?: number;
  /** The mdr normal model's force and damping prefactor (the normal force replaces the elastic law above). */
  normal?: MdrNormalOut;
  /** -1 when the stored history is in the frame of the other atom (ID order). */
  sg: number;
  /**
   * Torque arm of particle i is its full radius R_i (the contact point on a wall surface, measured with
   * native LAMMPS for fix wall/gran granular); by default the arm is R_i - delta/2 (contact at the overlap centre).
   */
  surfaceArm?: boolean;
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

/** src/math_const.h and gran_sub_mod_*.cpp constants at b332f28. */
const MY_PI = 3.14159265358979323846;
const MY_2PI = 6.28318530717958647692;
const MY_SQRT2 = 1.41421356237309504880;
const FOURTHIRDS = 1.33333333333333333333;
const TWOTHIRDS = 2.0 / 3.0;
const TWOROOTFIVEBYSIX = 1.82574185835055380345;
const ROOTTHREEBYTWO = 1.22474487139158894067;

/** MathExtra::len3 (src/math_extra.h at b332f28): sqrt(v0 v0 + v1 v1 + v2 v2). */
const len3 = (x: number, y: number, z: number): number => Math.sqrt(x * x + y * y + z * z);

/** The MAX(0, x) of src/pointers.h ("((A) > (B) ? (A) : (B))"): x itself for -0 and NaN. */
const max0 = (x: number): number => (0 > x ? 0 : x);

/**
 * GranSubMod::rotate_rescale_vec (src/GRANULAR/gran_sub_mod.cpp at b332f28): v without its component along the unit n,
 * rescaled to its previous length (zero when nothing is left).
 */
function rotateRescale(vx: number, vy: number, vz: number, nx: number, ny: number, nz: number): [number, number, number] {
  const rsht = vx * nx + vy * ny + vz * nz;
  const shrmag = len3(vx, vy, vz);
  const px = vx - rsht * nx, py = vy - rsht * ny, pz = vz - rsht * nz;
  const prjmag = len3(px, py, pz);
  const t = prjmag > 0 ? shrmag / prjmag : 0;
  return [t * px, t * py, t * pz];
}

/** The JKR contact radius and pull-off distance of native's closed form (the lazy GPL hook 'jkr'). */
export interface GranularJkrHook {
  contactRadius(dR: number, Reff: number, Emix: number, cohesion: number): number;
  pulloff(Reff: number, Emix: number, cohesion: number): number;
}
let jkrLoaded: GranularJkrHook | null = null;
const jkrHook = (): GranularJkrHook => (jkrLoaded ??= gplHook<GranularJkrHook>('jkr'));

/** GranSubMod::mix_stiffnessE, mix_stiffnessG, mix_stiffnessE_wall and mix_stiffnessG_wall (gran_sub_mod.cpp at b332f28). */
const mixE = (E1: number, E2: number, n1: number, n2: number): number => 1 / ((1 - n1 * n1) / E1 + (1 - n2 * n2) / E2);
const mixG = (E1: number, E2: number, n1: number, n2: number): number =>
  1 / (2 * (2 - n1) * (1 + n1) / E1 + 2 * (2 - n2) * (1 + n2) / E2);
const mixEwall = (E: number, n: number): number => E / (2 * (1 - n * n));
const mixGwall = (E: number, n: number): number => E / (4.0 * (2 - n) * (1 + n));

/** MathSpecial::powint (src/math_special.h at b332f28): x^n by binary powering. */
const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x, yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

/**
 * The damp of the tsuji damping (GranSubModDampingTsuji::init, "Eq. 53 from Marshall 2009", times MY_SQRT2) and of the
 * coeff_restitution damping (GranSubModDampingCoeffRestitution::init) for the coefficient of restitution cor.
 */
const dampConstant = (damp: DampKind, normal: NormalKind, cor: number): number => {
  if (damp === 'tsuji') {
    let d = 1.2728 - 4.2783 * cor + 11.087 * (cor * cor);
    d += -22.348 * (cor * cor * cor) + 27.467 * powint(cor, 4);
    d += -18.022 * powint(cor, 5) + 4.8218 * powint(cor, 6);
    return d * MY_SQRT2;
  }
  if (damp === 'coeff_restitution') {
    const logcor = libm.log(cor);
    if (normal === 'hooke') return -2 * logcor / Math.sqrt(MY_PI * MY_PI + logcor * logcor);
    let d = -ROOTTHREEBYTWO * TWOROOTFIVEBYSIX * logcor;
    d /= Math.sqrt(MY_PI * MY_PI + logcor * logcor);
    return d;
  }
  return 0;
};

/**
 * One granular contact: normal (hooke/hertz/hertz-material/dmt/jkr), damping, tangential (linear, mindlin),
 * rolling sds and twisting. sh holds the history: [0..2] tangential displacement (or elastic force), read and written in
 * the frame given by sg, [3] the previous contact radius, [4..6] rolling displacement and [7] twisting displacement, as
 * native's pair stores them (PairGranular keeps them in native's frame: see nativeFrame); a pair style's record has two
 * more entries ([8], [9]) for that. Returns false when there is no contact (the caller drops the history record).
 */
export function granularContact(c: ContactIn, sh: Float64Array, out: ContactOut, hadHistory: boolean): boolean {
  // As native LAMMPS does (src/GRANULAR/granular_model.cpp GranularModel::check_contact and calculate_forces, and the
  // sub-models of gran_sub_mod_normal.cpp, gran_sub_mod_damping.cpp, gran_sub_mod_tangential.cpp,
  // gran_sub_mod_rolling.cpp and gran_sub_mod_twisting.cpp at b332f28): every quantity below is computed with native's
  // operations in native's order (vector lengths as len3, rotate_rescale_vec, the caps), so a contact gives native's
  // values bit for bit. The rolling displacement and force are kept with the opposite sign of native's (exact negations).
  const { pm, nx, ny, nz, delta, Rf, ri, rj, meff, dt, update, sg } = c;
  const jkr = pm.normal === 'jkr';
  const rsq = c.rsq ?? c.r * c.r;
  const radsum = c.radsum ?? ri + rj;
  // touch: JKR::touch (a contact in place holds on to the pull-off distance), the default rsq < radsum^2 otherwise
  if (jkr && !c.normal) {
    if (hadHistory) {
      const dist = radsum + jkrHook().pulloff(Rf, pm.Eeff, pm.gamma);
      if (!(rsq < dist * dist)) return false;
    } else if (!(rsq < radsum * radsum)) return false;
  }
  const vrx = c.vrx, vry = c.vry, vrz = c.vrz;
  // synchronized_verlet: nx is the normalised half-step separation x_i - x_j - dt/2 (v_i - v_j) (norm3), and the full-step
  // normal is kept (nx_unrotated)
  const sync = pm.sync === true;
  let hx = nx, hy = ny, hz = nz;
  if (sync && c.dx !== undefined) {
    const t = 0.5 * dt;
    const ex = c.dx - t * vrx, ey = c.dy! - t * vry, ez = c.dz! - t * vrz;
    const scl = 1.0 / Math.sqrt(ex * ex + ey * ey + ez * ez);
    hx = ex * scl; hy = ey * scl; hz = ez * scl;
  }
  const vn = vrx * hx + vry * hy + vrz * hz; // vnnr = dot3(vr, nx)
  const vnx = vn * hx, vny = vn * hy, vnz = vn * hz;
  // wr = radi omega_i + radj omega_j; vtr = vt - wr x nx
  const Ax = ri * c.oix + rj * c.ojx;
  const Ay = ri * c.oiy + rj * c.ojy;
  const Az = ri * c.oiz + rj * c.ojz;
  const wnx = Ay * hz - Az * hy, wny = Az * hx - Ax * hz, wnz = Ax * hy - Ay * hx;
  const vtx = vrx - vnx - wnx, vty = vry - vny - wny, vtz = vrz - vnz - wnz;
  const vrel = len3(vtx, vty, vtz);
  // normal force (positive repulsive): the elastic part, then the damping prefactor
  let fne = 0;
  let contactA = 0;
  // F_pulloff of the cohesive models (DMT: 4 pi cohesion Reff, JKR: 3 pi cohesion Reff)
  const fPull = pm.normal === 'dmt' ? 4.0 * MY_PI * pm.gamma * Rf : jkr ? 3.0 * MY_PI * pm.gamma * Rf : 0;
  const dR = delta * Rf;
  if (c.normal) {
    // mdr: the normal model is evaluated by the MDR hook (src/engine-gpl/granular/mdr); it has its own contact test
    fne = c.normal.fne;
    contactA = c.normal.a;
  } else if (jkr) {
    const a = jkrHook().contactRadius(dR, Rf, pm.Eeff, pm.gamma);
    const a2 = a * a;
    fne = pm.kHM * a * a2 / Rf - MY_2PI * a2 * Math.sqrt(4.0 * pm.gamma * pm.Eeff / (MY_PI * a));
    contactA = a;
  } else {
    if (!(rsq < radsum * radsum)) return false;
    contactA = Math.sqrt(dR);
    if (pm.normal === 'hooke') fne = pm.kn * delta;
    else if (pm.normal === 'hertz') fne = pm.kn * contactA * delta;
    else fne = pm.kHM * contactA * delta;
    if (pm.normal === 'dmt') fne -= fPull;
  }
  // the mdr damping prefactor comes with the normal force (c.normal.etaN)
  let etaN = c.normal ? c.normal.etaN : 0;
  if (!c.normal) switch (pm.damp) {
    case 'velocity': etaN = pm.eta; break;
    case 'mass_velocity': etaN = pm.eta * meff; break;
    case 'viscoelastic': etaN = pm.eta * meff * contactA; break;
    default: {
      // tsuji and coeff_restitution: damp sqrt(MAX(0, meff Fnormal / delta)) (0 at delta <= 0)
      const sqrt1 = delta > 0.0 ? max0(meff * fne / delta) : 0.0;
      etaN = pm.dampC * Math.sqrt(sqrt1);
    }
  }
  let fn = fne - etaN * vn;
  if (pm.limit && fn < 0.0) fn = 0.0;
  // Fncrit: |Fne + 2 F_pulloff| for dmt and jkr, |Fntot| otherwise
  const fnAbs = pm.normal === 'dmt' || jkr ? Math.abs(fne + 2.0 * fPull) : Math.abs(fn);
  const fnx = fn * hx, fny = fn * hy, fnz = fn * hz;
  // tangential: damp = xt damp_prefactor, Fscrit = Fncrit mu
  const etaT = pm.xgt * etaN;
  const etaTf = etaT;
  const kind = pm.tang;
  // jkr keeps a contact record too, so the pull-off branch can be recognised (see touch above)
  const hist = kind !== 'linear_nohistory' || pm.roll !== 'none' || pm.twist !== 'none' || jkr;
  const forceVariant = kind === 'mindlin/force' || kind === 'mindlin_rescale/force';
  const rescale = kind === 'mindlin_rescale' || kind === 'mindlin_rescale/force';
  const aC = contactA;
  // the classic models of fix wall/gran (GranSubModTangentialLinearHistoryClassic; mindlin_classic with
  // contact_radius_flag scales k by the contact radius)
  const classic = kind === 'linear_history_classic' || kind === 'mindlin_classic';
  const keff = kind === 'linear_history' || kind === 'linear_history_classic' ? pm.kt : kind === 'linear_nohistory' ? 0 : pm.kt * aC;
  const fcap = fnAbs * pm.mu;
  let ftx = 0, fty = 0, ftz = 0;
  let shx = 0, shy = 0, shz = 0;
  if (kind === 'linear_nohistory') {
    // Ft = MIN(Fscrit, damp vrel) / vrel, fs = -Ft vtr
    const fsmag = etaTf * vrel;
    const Ft = vrel !== 0.0 ? (pm.mu * fnAbs < fsmag ? pm.mu * fnAbs : fsmag) / vrel : 0.0;
    ftx = -Ft * vtx; fty = -Ft * vty; ftz = -Ft * vtz;
  } else {
    shx = sg * sh[0]; shy = sg * sh[1]; shz = sg * sh[2];
    // the frame-update test: |h . n| k > EPSILON Fscrit (k_scaled for mindlin, none for the force variants)
    const kTest = forceVariant ? 1 : keff;
    let frameUpdate = false;
    let shrmag = 0;
    if (classic) {
      if (update) { shx += dt * vtx; shy += dt * vty; shz += dt * vtz; }
      shrmag = len3(shx, shy, shz);
      if (update) {
        const rsht = shx * nx + shy * ny + shz * nz;
        shx -= rsht * nx; shy -= rsht * ny; shz -= rsht * nz;
      }
    } else {
      // mindlin_rescale: on unloading the shear displacements/force are scaled by contact_radius / history[3]
      if (rescale && aC < sh[3]) {
        const cr = aC / sh[3];
        shx *= cr; shy *= cr; shz *= cr;
      }
      if (update) {
        let rsht = shx * hx + shy * hy + shz * hz;
        frameUpdate = Math.abs(rsht) * kTest > FRAME_EPS * fcap;
        if (frameUpdate) [shx, shy, shz] = rotateRescale(shx, shy, shz, hx, hy, hz);
        if (forceVariant) {
          const kd = -keff * dt;
          shx += kd * vtx; shy += kd * vty; shz += kd * vtz;
        } else {
          shx += dt * vtx; shy += dt * vty; shz += dt * vtz;
        }
        if (sync) {
          // second projection to the full-step normal
          rsht = shx * nx + shy * ny + shz * nz;
          frameUpdate = Math.abs(rsht) * kTest > FRAME_EPS * fcap;
          if (frameUpdate) [shx, shy, shz] = rotateRescale(shx, shy, shz, nx, ny, nz);
        }
      }
    }
    // fs = -k h (the elastic force itself for the force variants), then the damping of vtr (rotated to the full-step
    // normal when synchronized and the frame was updated)
    if (forceVariant) { ftx = shx; fty = shy; ftz = shz; } else {
      // -k h; mindlin_classic: -k contact_radius h ("scale3(-k * contact_radius, history, fs)")
      const mk = kind === 'mindlin_classic' ? -pm.kt * aC : -keff;
      ftx = mk * shx; fty = mk * shy; ftz = mk * shz;
    }
    let vdx = vtx, vdy = vty, vdz = vtz;
    if (frameUpdate && sync) [vdx, vdy, vdz] = rotateRescale(vtx, vty, vtz, nx, ny, nz);
    const md = -etaTf;
    const dampx = md * vdx, dampy = md * vdy, dampz = md * vdz;
    ftx += dampx; fty += dampy; ftz += dampz;
    const magfs = len3(ftx, fty, ftz);
    if (magfs > fcap) {
      if (!classic) shrmag = len3(shx, shy, shz);
      if (shrmag !== 0.0) {
        const s = fcap / magfs;
        ftx = s * ftx; fty = s * fty; ftz = s * ftz;
        // the history from the elastic part of the rescaled force
        shx = ftx - dampx; shy = fty - dampy; shz = ftz - dampz;
        if (!forceVariant) {
          const ik = kind === 'mindlin_classic' ? -1.0 / (pm.kt * aC) : -1.0 / keff;
          shx = ik * shx; shy = ik * shy; shz = ik * shz;
        }
      } else {
        ftx = 0; fty = 0; ftz = 0;
      }
    }
    if (rescale && update) sh[3] = aC;
  }
  // rolling sds (the displacement and pseudo-force with the opposite sign of native's: q = -history, rl = -fr)
  let rlx = 0, rly = 0, rlz = 0;
  if (pm.roll === 'sds') {
    const wdx = c.oix - c.ojx, wdy = c.oiy - c.ojy, wdz = c.oiz - c.ojz;
    // -vrl: native vrl = Reff (relrot x nx)
    const vrlx = -Rf * (wdy * nz - wdz * ny), vrly = -Rf * (wdz * nx - wdx * nz), vrlz = -Rf * (wdx * ny - wdy * nx);
    let qx = sh[4], qy = sh[5], qz = sh[6];
    const k = pm.kr, g = pm.gr;
    const frcrit = pm.mr * fnAbs;
    if (update) {
      const rolldotn = qx * nx + qy * ny + qz * nz;
      if (Math.abs(rolldotn) * k > FRAME_EPS * frcrit) [qx, qy, qz] = rotateRescale(qx, qy, qz, nx, ny, nz);
      qx += dt * vrlx; qy += dt * vrly; qz += dt * vrlz;
    }
    const mk = -k, mg = -g;
    let fx0 = mk * qx + mg * vrlx, fy0 = mk * qy + mg * vrly, fz0 = mk * qz + mg * vrlz;
    const magfr = len3(fx0, fy0, fz0);
    if (magfr > frcrit) {
      if (len3(qx, qy, qz) !== 0.0) {
        const kInv = 1.0 / k, magfrInv = 1.0 / magfr;
        const s1 = -frcrit * kInv * magfrInv, s2 = -g * kInv;
        qx = s1 * fx0 + s2 * vrlx; qy = s1 * fy0 + s2 * vrly; qz = s1 * fz0 + s2 * vrlz;
        const s3 = frcrit * magfrInv;
        fx0 = s3 * fx0; fy0 = s3 * fy0; fz0 = s3 * fz0;
      } else {
        fx0 = 0; fy0 = 0; fz0 = 0;
      }
    }
    if (update) { sh[4] = qx; sh[5] = qy; sh[6] = qz; }
    rlx = fx0; rly = fy0; rlz = fz0;
  }
  // twisting: torque along n, tau = -k h - damp magtwist, capped at Mtcrit = mu Fncrit
  let twq = 0;
  if (pm.twist !== 'none') {
    let k: number, g: number, mu: number;
    if (pm.twist === 'sds') { k = pm.kw; g = pm.gw; mu = pm.mw; }
    else {
      // marshall: k = 0.5 k_tang a^2, damp = 0.5 tdamp a^2, mu = 2/3 mu_tang a
      k = 0.5 * pm.kt * aC * aC; g = 0.5 * etaT * aC * aC; mu = TWOTHIRDS * pm.mu * aC;
    }
    const om = (c.oix - c.ojx) * nx + (c.oiy - c.ojy) * ny + (c.oiz - c.ojz) * nz;
    let q = sh[7];
    if (update) q += om * dt;
    let tau = -k * q - g * om;
    const signtwist = (om > 0 ? 1 : 0) - (om < 0 ? 1 : 0);
    const mtcrit = mu * fnAbs;
    if (Math.abs(tau) > mtcrit) {
      q = (mtcrit * signtwist - g * om) / k;
      tau = -mtcrit * signtwist;
    }
    sh[7] = q;
    twq = tau;
  }
  if (hist && kind !== 'linear_nohistory') {
    sh[0] = sg * shx; sh[1] = sg * shy; sh[2] = sg * shz;
    if (!rescale) sh[3] = aC;
  }
  out.fx = fnx + ftx; out.fy = fny + fty; out.fz = fnz + ftz;
  // torques: -(nx x fs), scaled by radi - delta/2 and radj - delta/2 (a wall: radi)
  const cx = hy * ftz - hz * fty, cy = hz * ftx - hx * ftz, cz = hx * fty - hy * ftx;
  const ci = c.surfaceArm ? ri : ri - 0.5 * delta, cj = rj - 0.5 * delta;
  out.tix = -cx * ci; out.tiy = -cy * ci; out.tiz = -cz * ci;
  out.tjx = -cx * cj; out.tjy = -cy * cj; out.tjz = -cz * cj;
  // rolling torque Reff (nx x fr) on i (and its opposite on j); twisting torque tau nx on i
  if (pm.roll === 'sds') {
    const rx = -Rf * (ny * rlz - nz * rly), ry = -Rf * (nz * rlx - nx * rlz), rz = -Rf * (nx * rly - ny * rlx);
    out.tix += rx; out.tiy += ry; out.tiz += rz;
    out.tjx -= rx; out.tjy -= ry; out.tjz -= rz;
  }
  if (pm.twist !== 'none') {
    const tx = twq * nx, ty = twq * ny, tz = twq * nz;
    out.tix += tx; out.tiy += ty; out.tiz += tz;
    out.tjx -= tx; out.tjy -= ty; out.tjz -= tz;
  }
  out.hist = hist;
  out.a = contactA;
  out.fntot = fn; out.fsx = ftx; out.fsy = fty; out.fsz = ftz;
  // the rolling force is reported with native's sign (the engine's pseudo-force is the negative of native fr)
  out.frx = -rlx; out.fry = -rly; out.frz = -rlz; out.mtt = twq;
  return true;
}
/**
 * A pair style's contact record: the 8 values of granularContact, then native's orientation of the pair when its rolling
 * and twisting values were last put in native's frame (+1: native's row atom has the lower ID, -1: the higher one, 0:
 * not known), the neighbor build it was read at, and the compute call that last kept it (SEEN).
 */
const HIST_LEN = 11;
const SEEN = 10;

/** The contact history key of atom IDs lo < hi: one number while hi < 2^26, else a string. */
const pairKey = (lo: number, hi: number): number | string => (hi < 67108864 ? lo * 67108864 + hi : `${lo}:${hi}`);

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
  /** Contact history per atom-ID pair (lo, hi) (key: pairKey). */
  private shear = new Map<number | string, Float64Array>();
  /** The compute calls so far: a record's SEEN slot holds the last one that kept it. */
  private computeStamp = 0;
  /** compute's contact input (refilled for every contact). */
  private readonly contactIn: ContactIn = {
    pm: null as unknown as Pm, nx: 0, ny: 0, nz: 0, r: 0, delta: 0, Rf: 0, ri: 0, rj: 0, meff: 0,
    vrx: 0, vry: 0, vrz: 0, oix: 0, oiy: 0, oiz: 0, ojx: 0, ojy: 0, ojz: 0, dt: 0, update: false,
    rsq: 0, radsum: 0, dx: 0, dy: 0, dz: 0, normal: undefined, sg: 1,
  };
  /** Some pair_coeff has a heat model (the temperature and heatflow properties are then needed). */
  private heatOn = false;
  /** The mdr contact model when a pair_coeff uses the mdr normal model (src/engine-gpl/granular/mdr). */
  private mdr: MdrApi | null = null;
  /** newton pair (StyleContext.newtonPair): the half list native builds. */
  private newton = true;
  /** Native's pair order (the lazily loaded pair/local helper) and its rows of the current neighbor build. */
  private pairOrder: PairOrderSource | null = null;
  private rows: { build: number; start: Int32Array; part: Int32Array } | null = null;

  /** Some type pair has a rolling or twisting model (their histories are kept in native's frame). */
  private rollOrTwist(): boolean {
    for (const p of this.pm) if (p && (p.roll !== 'none' || p.twist !== 'none')) return true;
    return false;
  }

  /**
   * The rows of native's half list at this neighbor build (pair/local's native pair order): the partners of each owned
   * atom, as engine indices.
   */
  private nativeRows(pc: PairCompute): { build: number; start: Int32Array; part: Int32Array } {
    const nb = pc.nb;
    if (this.rows && this.rows.build === nb.nbuild) return this.rows;
    this.pairOrder ??= gplHook<PairLocalHook>('granular').pairOrder(this.newton);
    const pairs = this.pairOrder.pairs(pc.s, nb, pc.geom, pc.half!);
    const n = nb.nlocal;
    const start = new Int32Array(n + 1);
    for (let p = 0; p < pairs.length; p += 2) start[pairs[p] + 1]++;
    for (let i = 0; i < n; i++) start[i + 1] += start[i];
    const fill = start.slice(0, n);
    const part = new Int32Array(pairs.length >> 1);
    for (let p = 0; p < pairs.length; p += 2) part[fill[pairs[p]]++] = pairs[p + 1] & NEIGHMASK;
    this.rows = { build: nb.nbuild, start, part };
    return this.rows;
  }

  /**
   * Native's orientation of the pair of owned i and j (owned or ghost) at this build: +1 when native's half list has it in
   * the row of the atom with the lower ID, -1 in the row of the other one, 0 when its rows do not have it.
   */
  private nativeSign(i: number, j: number, owner: Int32Array, id: Int32Array): number {
    const { start, part } = this.rows!;
    const oj = owner[j];
    for (let e = start[i]; e < start[i + 1]; e++) if (owner[part[e]] === oj) return id[i] <= id[oj] ? 1 : -1;
    if (oj < start.length - 1) for (let e = start[oj]; e < start[oj + 1]; e++) if (owner[part[e]] === i) return id[oj] <= id[i] ? 1 : -1;
    return 0;
  }

  /**
   * As native LAMMPS does (src/GRANULAR/granular_model.cpp init, src/GRANULAR/pair_granular.cpp transfer_history and
   * src/fix_neigh_history.cpp pre_exchange at b332f28): a pair's history is kept by the atom whose row of the half list
   * holds the pair, and the other atom gets it times transfer_history_factor, which is -1 for every value of a sub-model
   * that keeps the default ("jvalues[n] = -onevalues[n]"; mindlin's contact radius and mdr keep theirs). When a neighbor
   * build gives the pair to the other atom's row, the rolling and twisting displacements are therefore negated, although
   * the rolling and twisting velocities do not change sign with the swap. The tangential displacement is antisymmetric
   * and stays in the frame of the lower ID (sg). Here: a new record takes native's orientation at this build, and a kept
   * one read at an earlier build has those values negated when the orientation changed.
   */
  private nativeFrame(sh: Float64Array, kept: boolean, i: number, j: number, nb: PairCompute['nb'], id: Int32Array): void {
    const build = this.rows!.build;
    if (kept && sh[9] === build) return;
    const o = this.nativeSign(i, j, nb.owner, id);
    if (kept && sh[8] !== 0 && o !== 0 && o !== sh[8]) {
      sh[4] = -sh[4]; sh[5] = -sh[5]; sh[6] = -sh[6]; sh[7] = -sh[7];
    }
    sh[8] = o;
    sh[9] = build;
  }

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
    if (this.newton !== (ctx.newtonPair !== false)) this.pairOrder = null;
    this.newton = ctx.newtonPair !== false;
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
          // As native LAMMPS does (src/GRANULAR/gran_sub_mod_normal.cpp and gran_sub_mod_tangential.cpp coeffs_to_local
          // at b332f28): a pair_coeff's own material values, Emix = mix_stiffnessE(E, E, nu, nu) and, for a NULL k_t,
          // k = 8 mix_stiffnessG(E, E, nu, nu)
          this.pm[i * nt + j] = this.pmOf(sp, sp.E ? mixE(sp.E, sp.E, sp.nu, sp.nu) : 0, () => 8.0 * mixG(sp.E, sp.E, sp.nu, sp.nu));
          continue;
        }
        const a = self(i), b = self(j);
        // a pair that cannot be mixed stays unset: initOne throws for it when this style is used for the pair, and a
        // hybrid sub-style never asks for the pairs of other sub-styles (pair_style hybrid granular surf/granular)
        if (!a || !b) continue;
        const sigOf = (q: Spec) => [q.normal, q.damp, q.tang, q.roll, q.twist, q.limit ? 1 : 0, q.sync ? 1 : 0, q.heat].join('|');
        if (sigOf(a) !== sigOf(b)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j}): models of types ${i} and ${j} differ and cannot be mixed`);
        // As native LAMMPS does (src/GRANULAR/granular_model.cpp mix_coeffs and the sub-models' mix_coeffs at b332f28):
        // E by mix_stiffnessE(E_i, E_j, nu_i, nu_j), every other coefficient (nu, damping, cohesion, stiffnesses,
        // friction) by mix_geom = sqrt(x_i x_j); a NULL k_t on either side stays NULL, and mindlin then takes
        // 8 mix_stiffnessG of the mixed normal model's E and nu (its get_emod is the mixed Emix)
        const g = (x: number, y: number) => Math.sqrt(x * y);
        const Eeff = a.E && b.E ? mixE(a.E, b.E, a.nu, b.nu) : 0;
        const nuMix = g(a.nu, b.nu);
        const ktNull = a.ktNull || b.ktNull;
        const mixed: Spec = { ...a, kn: g(a.kn, b.kn), E: Eeff, nu: nuMix, gamma: g(a.gamma, b.gamma), eta: g(a.eta, b.eta),
          kt: ktNull ? 0 : g(a.kt, b.kt), ktNull, xgt: g(a.xgt, b.xgt), mu: g(a.mu, b.mu), kr: g(a.kr, b.kr), gr: g(a.gr, b.gr),
          mr: g(a.mr, b.mr), kw: g(a.kw, b.kw), gw: g(a.gw, b.gw), mw: g(a.mw, b.mw), cutoff: null, hk: g(a.hk, b.hk) };
        const p = this.pmOf(mixed, Eeff, () => 8.0 * mixG(Eeff, Eeff, nuMix, nuMix));
        this.pm[i * nt + j] = p;
      }
    }
    for (let i = 1; i < nt; i++) for (let j = 1; j < i; j++) this.pm[i * nt + j] = this.pm[j * nt + i];
  }

  /**
   * Resolved parameters of a wall/particle pair (fix wall/gran granular): the particle's own material
   * against a flat wall; as native LAMMPS does (gran_sub_mod_normal.cpp and gran_sub_mod_tangential.cpp coeffs_to_local
   * at b332f28, contact_type WALL): Emix = mix_stiffnessE_wall(E, nu) and a NULL k_t = 8 mix_stiffnessG_wall(E, nu).
   */
  wallParams(sp: Spec, region = false): Pm {
    // doc: "The *mdr* model currently only supports *fix wall/gran/region*, not *fix wall/gran*."
    if (sp.normal === 'mdr' && !region) throw new StyleError("fix wall/gran with normal model 'mdr' is not implemented in this engine (the doc supports fix wall/gran/region only)");
    return this.pmOf(sp, sp.E ? mixEwall(sp.E, sp.nu) : 0, () => 8.0 * mixGwall(sp.E, sp.nu));
  }

  /** Resolved parameters of a spec whose effective modulus is eff (ktNull: the k_t of a NULL tangential stiffness). */
  private pmOf(sp: Spec, eff: number, ktNull: () => number): Pm {
    let kt = sp.kt;
    if (sp.ktNull) {
      kt = sp.E > 0 ? ktNull() : 0;
      if (!(kt > 0)) throw new StyleError('pair_style granular: tangential k_t NULL needs E and nu for the types');
    }
    const material = sp.normal === 'hertz/material' || sp.normal === 'dmt' || sp.normal === 'jkr';
    const sig = [sp.normal, sp.damp, sp.tang, sp.roll, sp.twist, sp.limit ? 1 : 0, sp.sync ? 1 : 0, sp.heat].join('|');
    return {
      normal: sp.normal, kn: sp.kn, Eeff: eff, kHM: material ? FOURTHIRDS * eff : 0, dampC: dampConstant(sp.damp, sp.normal, sp.eta),
      gamma: sp.gamma, eta: sp.eta, Y: sp.Y, psib: sp.psib, dtype: sp.dtype,
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
    // As native LAMMPS does (src/GRANULAR/pair_granular.cpp init_one at b332f28): a model acting beyond contact (jkr)
    // adds the pull-off distance of the largest radii, "cutoff += pulloff"
    if (p.normal === 'jkr' && rs > 0) {
      const ri = this.maxRadius(i), rj = this.maxRadius(j);
      return rs + jkrHook().pulloff(ri * rj / (ri + rj), p.Eeff, p.gamma);
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
    // the records kept past this call: those of the contacts that keep a history now (a pair out of contact, or whose
    // contact ends, has no record; the others are dropped after the loop, where nothing reads them before)
    const stamp = ++this.computeStamp;
    // mdr: the fix's pre-force update (per-atom state, penalty, mean surface displacement) before the contacts
    const mdr = this.mdr;
    if (mdr) mdr.beginStep(s, pc, update);
    const out = newContactOut();
    const cin = this.contactIn;
    // heat conduction: heatflow accumulates (zeroed by fix heat/flow), Q_i = heat(a, T_j - T_i) and Q_j = -Q_i
    const T = this.heatOn ? s.custom.get(HEAT_TEMPERATURE)!.data : null;
    const HF = this.heatOn ? s.custom.get(HEAT_FLOW)!.data : null;
    // rolling and twisting histories follow native's half-list orientation (nativeFrame)
    const frame = this.rollOrTwist() ? this.nativeRows(pc) : null;
    const sLJ = pc.specialLJ;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const ri = radius[oi], mi = rmass[oi];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jraw = list.neighbors[k];
        const j = jraw & NEIGHMASK;
        const oj = owner[j];
        // As native LAMMPS does (src/GRANULAR/pair_granular.cpp compute at b332f28): "factor_lj = special_lj[sbmask(j)];
        // ... if (factor_lj == 0) continue;" ahead of the contact test, so the pair's touch flag and history stay as
        // they were (special_bonds.html: "The LJ factors are applied to the remaining terms that the potential
        // calculates, whether they represent LJ interactions or not.")
        const factorLJ = sLJ[jraw >>> SBBITS];
        if (factorLJ === 0) {
          if (this.shear.size) {
            const a0 = id[oi], b0 = id[oj];
            const key0 = a0 > b0 ? pairKey(b0, a0) : pairKey(a0, b0);
            const rec = this.shear.get(key0);
            if (rec) rec[SEEN] = stamp;
          }
          continue;
        }
        const rj = radius[oj];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const pm = this.pm[type[oi] * nt + type[oj]];
        if (!pm) continue;
        const a = id[oi], b = id[oj];
        const flip = a > b;
        const radsum = ri + rj;
        const jkr = pm.normal === 'jkr';
        if (rsq >= radsum * radsum && !jkr) {
          if (mdr) mdr.pairDrop(a, b);
          continue;
        }
        const key = flip ? pairKey(b, a) : pairKey(a, b);
        const r = Math.sqrt(rsq), rinv = 1 / r;
        const delta = radsum - r;
        // the pair's record from the last call, looked up once (a new record goes into the map below only when kept)
        const stored = this.shear.get(key);
        // jkr without a record: native's touch test rsq < radsum^2 (granularContact), nothing to evaluate otherwise
        if (jkr && stored === undefined && !(rsq < radsum * radsum)) continue;
        const mj = rmass[oj];
        const Rf = (ri * rj) / radsum;
        // m_eff; a contact with a particle of the fix freeze group uses the other particle's mass (measured in gran.ts)
        let meff = (mi * mj) / (mi + mj);
        if (this.freezeBit) {
          if (s.mask[oi] & this.freezeBit) meff = mj;
          else if (s.mask[oj] & this.freezeBit) meff = mi;
        }
        const sh = stored ?? new Float64Array(HIST_LEN);
        if (frame && (pm.roll !== 'none' || pm.twist !== 'none')) this.nativeFrame(sh, stored !== undefined, i, j, nb, id);
        const normal = mdr && pm.normal === 'mdr'
          ? mdr.pairNormal({ ida: a, idb: b, oi, oj, ri, rj, delta, Rf, nx: dx * rinv, ny: dy * rinv, nz: dz * rinv, meff, hu: update })
          : undefined;
        // the contact input, one object refilled for every contact (granularContact only reads it)
        const c = cin;
        c.pm = pm; c.nx = dx * rinv; c.ny = dy * rinv; c.nz = dz * rinv; c.r = r; c.delta = delta; c.Rf = Rf;
        c.ri = ri; c.rj = rj; c.meff = meff; c.rsq = rsq; c.radsum = radsum; c.dx = dx; c.dy = dy; c.dz = dz;
        c.vrx = v[3 * oi] - v[3 * oj]; c.vry = v[3 * oi + 1] - v[3 * oj + 1]; c.vrz = v[3 * oi + 2] - v[3 * oj + 2];
        c.oix = omega[3 * oi]; c.oiy = omega[3 * oi + 1]; c.oiz = omega[3 * oi + 2];
        c.ojx = omega[3 * oj]; c.ojy = omega[3 * oj + 1]; c.ojz = omega[3 * oj + 2];
        c.dt = dt; c.update = update; c.sg = flip ? -1 : 1; c.normal = normal;
        const ok = granularContact(c, sh, out, stored !== undefined);
        if (!ok) continue;
        if (out.hist) {
          sh[SEEN] = stamp;
          if (!stored) this.shear.set(key, sh);
        }
        if (T && pm.heat !== 'none') {
          const q = heatFlowOf(pm.heat, pm.hk, out.a, T[oj] - T[oi]);
          HF![oi] += q;
          // measured with native LAMMPS (black box): a periodic image partner (j >= the owned atom count) does not get
          // its share back, so the heat of a contact across a periodic boundary is counted once, on the owned side
          if (j < s.n) HF![oj] -= q;
        }
        // native compute: "scale3(factor_lj, forces)", "scale3(factor_lj, torquesi)" and "scale3(factor_lj, torquesj)"
        // (the heat flow is not scaled)
        if (factorLJ !== 1) {
          out.fx *= factorLJ; out.fy *= factorLJ; out.fz *= factorLJ;
          out.tix *= factorLJ; out.tiy *= factorLJ; out.tiz *= factorLJ;
          out.tjx *= factorLJ; out.tjy *= factorLJ; out.tjz *= factorLJ;
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
    // the records not kept by this call go (as the contacts that keep no history and the pairs out of the list)
    if (this.shear.size) for (const [key, rec] of this.shear) if (rec[SEEN] !== stamp) this.shear.delete(key);
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
    const key = flip ? pairKey(b, a) : pairKey(a, b);
    const stored = this.shear.get(key);
    // a copy: the contact code writes its history arrays even without an update
    const sh = stored ? Float64Array.from(stored) : new Float64Array(8);
    const nx = del.dx * rinv, ny = del.dy * rinv, nz = del.dz * rinv;
    const normal = this.mdr && pm.normal === 'mdr'
      ? this.mdr.pairNormal({ ida: a, idb: b, oi: i, oj: j, ri, rj, delta, Rf, nx, ny, nz, meff, hu: false })
      : undefined;
    const out = newContactOut();
    const ok = granularContact({
      pm, nx, ny, nz, r, delta, Rf, ri, rj, meff, rsq, radsum, dx: del.dx, dy: del.dy, dz: del.dz,
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
