/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/APIP/atom_vec_apip.h (the
 * ApipLambdaRequired flags) and the setup() of src/APIP/pair_eam_apip.cpp and pair_pace_apip.cpp (which fixes switch on
 * the thermostat and local-averaging bookkeeping), at commit b332f28. Author named in the sources: David Immel (FZ
 * Juelich), with the LAMMPS Sandia copyright notice (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Shared pieces of the adaptive-precision (APIP) pair styles: the lambda_required bits and the per-atom fields of
 * atom_style apip (src/engine/atoms.ts) seen through the owner of each ghost.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import {
  APIP_E_FAST, APIP_E_PRECISE, APIP_F_CONST_LAMBDA, APIP_F_DYN_LAMBDA, APIP_LAMBDA, APIP_LAMBDA_CONST, APIP_LAMBDA_REQUIRED,
  isApipStyle,
} from '../../engine/atoms';

/** ApipLambdaRequired (atom_vec_apip.h). */
export const UNKNOWN = 0;
export const SIMPLE = 1 << 0;
export const NO_SIMPLE = 1 << 1;
export const COMPLEX = 1 << 2;
export const NO_COMPLEX = 1 << 3;

/** The per-atom APIP fields of one force evaluation; lambda and lambdaConst are indexed by owned and ghost atoms. */
export interface ApipFields {
  /** apip_lambda of every atom (ghosts carry their owner's value). */
  lambda: Float64Array;
  /** apip_lambda_const of every atom, or null without the thermostat fields. */
  lambdaConst: Float64Array | null;
  /** apip_lambda_required of the owned atoms. */
  lambdaRequired: Float64Array;
  eFast: Float64Array;
  ePrecise: Float64Array;
  /** apip_f_const_lambda / apip_f_dyn_lambda of the owned atoms (3 per atom), or null in the conservative mode. */
  fConstLambda: Float64Array | null;
  fDynLambda: Float64Array | null;
}

const ghostCopy = (owned: Float64Array, pc: PairCompute): Float64Array => {
  const out = new Float64Array(pc.nall);
  const owner = pc.nb.owner;
  for (let k = 0; k < pc.nall; k++) out[k] = owned[owner[k]];
  return out;
};

/** The APIP fields of the state (atom_style apip); refuses another atom style with the native message of the style. */
export const apipFields = (pc: PairCompute): ApipFields => {
  const c = pc.s.custom;
  const get = (k: string): Float64Array | null => c.get(k)?.data ?? null;
  const lam = get(APIP_LAMBDA), req = get(APIP_LAMBDA_REQUIRED);
  if (!lam || !req) throw new StyleError('the APIP pair styles require atom_style apip');
  const lc = get(APIP_LAMBDA_CONST);
  return {
    lambda: ghostCopy(lam, pc),
    lambdaConst: lc ? ghostCopy(lc, pc) : null,
    lambdaRequired: req,
    eFast: get(APIP_E_FAST)!,
    ePrecise: get(APIP_E_PRECISE)!,
    fConstLambda: get(APIP_F_CONST_LAMBDA),
    fDynLambda: get(APIP_F_DYN_LAMBDA),
  };
};

/** The init_style checks shared by the APIP pairs (messages as native). */
export const checkApipAtomStyle = (ctx: StyleContext, style: string): void => {
  if (ctx.s && !isApipStyle(ctx.s.atomStyle)) throw new StyleError(`Pair style ${style} requires an atom style with lambda`);
  if (ctx.newtonPair === false) throw new StyleError(`Pair style ${style} requires newton pair on`);
};

/**
 * setup(): lambda_la when a fix lambda/la/csp/apip exists and lambda_thermostat when a fix lambda_thermostat/apip exists;
 * the thermostat needs the fields of atom_style apip thermostat (native messages).
 */
export const apipModes = (ctx: StyleContext, style: string): { thermostat: boolean; la: boolean } => {
  const la = !!ctx.fixOf?.('lambda/la/csp/apip');
  const thermostat = !!ctx.fixOf?.('lambda_thermostat/apip');
  if (thermostat && ctx.s && !ctx.s.custom.has(APIP_LAMBDA_CONST)) {
    throw new StyleError(`Pair style ${style} requires an atom style with lambda_const for a local thermostat.`);
  }
  return { thermostat, la };
};

/** Adds the ghost rows (3 per atom) of a scratch array into the owned rows of a per-atom field (reverse communication). */
export const reverseAdd3 = (pc: PairCompute, scratch: Float64Array, owned: Float64Array): void => {
  const owner = pc.nb.owner;
  for (let k = 0; k < pc.nall; k++) {
    const i = owner[k];
    owned[3 * i] += scratch[3 * k];
    owned[3 * i + 1] += scratch[3 * k + 1];
    owned[3 * i + 2] += scratch[3 * k + 2];
  }
};
