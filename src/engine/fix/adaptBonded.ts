import { StyleError, type Bonded, type BondedAdaptAccess } from '../force/types';
import { SimpleBonded } from '../force/bonded_util';
import { BONDED_EXTRACT } from './adaptBondedTable';

/**
 * Bonded styles that keep native's vector as one of their coefficient tables (SimpleBonded): native extract() name ->
 * [the engine's parameter, native's stored value per engine value]. Read from the extract() and coeff() functions of
 * each style (src/*.cpp at b332f28). A style that keeps the value another way gives it through Bonded.adaptParam.
 */
const BONDED_ADAPT: Record<string, Record<string, [string, number]>> = {
  'bond harmonic': { k: ['K', 1], r0: ['r0', 1] },
  'bond morse': { d0: ['D', 1], alpha: ['alpha', 1], r0: ['r0', 1] },
  'bond fene': { k: ['K', 1], r0: ['R0', 1] },
  'bond fene/expand': { k: ['K', 1], r0: ['R0', 1], epsilon: ['epsilon', 1], sigma: ['sigma', 1], shift: ['delta', 1] },
  'bond gromos': { k: ['K', 1], r0: ['r0', 1] },
  'bond nonlinear': { lamda: ['lam', 1], epsilon: ['epsilon', 1], r0: ['r0', 1] },
  'bond class2': { k2: ['K2', 1], k3: ['K3', 1], k4: ['K4', 1], r0: ['r0', 1] },
  'angle harmonic': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle charmm': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle cosine': { k: ['K', 1] },
  'angle cosine/squared': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle cosine/squared/restricted': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle fourier': { k: ['K', 1], C0: ['C0', 1], C1: ['C1', 1], C2: ['C2', 1] },
  'angle fourier/simple': { k: ['K', 1], C: ['c', 1], N: ['n', 1] },
  'angle quartic': { k2: ['K2', 1], k3: ['K3', 1], k4: ['K4', 1], theta0: ['theta0', 1] },
  'angle mm3': { k2: ['K', 1], theta0: ['theta0', 1] },
  'dihedral charmm': { k: ['K', 1] },
  'dihedral opls': { k1: ['K1', 0.5], k2: ['K2', 0.5], k3: ['K3', 0.5], k4: ['K4', 0.5] },
  'dihedral multi/harmonic': { a1: ['A1', 1], a2: ['A2', 1], a3: ['A3', 1], a4: ['A4', 1], a5: ['A5', 1] },
  'dihedral quadratic': { k: ['K', 1], phi0: ['phi0', 1] },
  'dihedral helix': { a: ['A', 1], b: ['B', 1], c: ['C', 1] },
  'dihedral cosine/squared/restricted': { k: ['K', 1], phi0: ['phi0', 1] },
  'improper harmonic': { k: ['K', 1], chi0: ['chi0', 1] },
  'improper cvff': { k: ['K', 1] },
  'improper fourier': { k: ['K', 1], C0: ['C0', 1], C1: ['C1', 1], C2: ['C2', 1] },
  'improper cossq': { k: ['K', 1], chi0: ['chi0', 1] },
  'improper distance': { k2: ['K2', 1], k4: ['K4', 1] },
  'improper distharm': { k: ['K', 1], d0: ['d0', 1] },
  'improper sqdistharm': { k: ['K', 1] },
  'improper inversion/harmonic': { k: ['K', 1 / 3], w0: ['omega0', 1] },
  // MOFFF/angle_cosine_buck6d.cpp (th0 in radians), EXTRA-MOLECULE/angle_cosine_delta.h (cosine/squared's k and
  // theta0), YAFF/angle_cross.cpp, DIPOLE/angle_dipole.cpp (gamma0 in radians), CG-SPICA/angle_spica.cpp
  'angle cosine/buck6d': { k: ['K', 1], th0: ['theta0', 1] },
  'angle cosine/delta': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle cross': { kss: ['Kss', 1], kbs0: ['Kbs0', 1], kbs1: ['Kbs1', 1], r00: ['r00', 1], r01: ['r01', 1], theta0: ['theta0', 1] },
  'angle dipole': { k: ['k', 1], gamma0: ['gamma0', 1] },
  'angle sdk': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle spica': { k: ['K', 1], theta0: ['theta0', 1] },
  // YAFF/bond_mm3.cpp, EXTRA-MOLECULE/bond_quartic_exp.cpp, EXTRA-MOLECULE/bond_fene_nm.cpp (BondFENE's k and r0),
  // MESONT/bond_mesocnt.cpp (BondHarmonic's k and r0)
  'bond mm3': { k2: ['K', 1], r0: ['r0', 1] },
  'bond quartic/exp': { r0: ['r0', 1], k2: ['k2', 1], k3: ['k3', 1], k4: ['k4', 1], A: ['A', 1], B: ['B', 1] },
  'bond fene/nm': { k: ['K', 1], r0: ['R0', 1] },
  'bond mesocnt': { k: ['K', 1], r0: ['r0', 1] },
  // MOLECULE/dihedral_charmmfsw.cpp, CLASS2/dihedral_class2.cpp and dihedral_class2xe.cpp (phi in radians)
  'dihedral charmmfsw': { k: ['K', 1] },
  'dihedral class2': { k1: ['K1', 1], k2: ['K2', 1], k3: ['K3', 1], phi1: ['phi1', 1], phi2: ['phi2', 1], phi3: ['phi3', 1] },
  'dihedral class2xe': { k1: ['K1', 1], k2: ['K2', 1], k3: ['K3', 1], phi1: ['phi1', 1], phi2: ['phi2', 1], phi3: ['phi3', 1] },
};

/** Whether the style's native extract() knows the name (BONDED_EXTRACT). */
export const bondedExtractHas = (kind: string, style: string, name: string): boolean =>
  (BONDED_EXTRACT[`${kind} ${style}`] ?? []).includes(name);

/**
 * Where this engine keeps the parameter native's extract() hands fix adapt (bondedExtractHas has accepted it): the
 * style's own accessor (Bonded.adaptParam) or the coefficient table BONDED_ADAPT names, as native's stored value; an
 * error naming the parameter when the engine keeps no such value.
 */
export const bondedAdaptAccess = (style: Bonded, kind: string, name: string): BondedAdaptAccess => {
  const own = style.adaptParam(name);
  if (own) return own;
  const m = BONDED_ADAPT[`${kind} ${style.name}`]?.[name];
  if (m && style instanceof SimpleBonded) {
    const arr = style.params.p(m[0]), ratio = m[1];
    return { get: (i) => arr[i] * ratio, set: (i, v) => { arr[i] = v / ratio; } };
  }
  throw new StyleError(`Fix adapt ${kind} style ${style.name} param ${name}: the browser engine cannot change this parameter of ${kind} style ${style.name}`);
};
