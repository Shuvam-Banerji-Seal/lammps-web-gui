import type { AtomStyle, CustomProp, SimState } from './types';

/*
 * atom_style smd (core, independent of the GPL ports; the SMD pair, fix and compute styles are in
 * src/engine-gpl/machdyn).
 *
 * docs.lammps.org/atom_style.html, row smd: "*atomic* + molecule, radius, rmass + "smd data"", and the text: "The *smd*
 * style is for Smooth Particle Mach dynamics. Both fluids and solids can be modeled. Particles store the mass and volume
 * of an integration point, a kernel diameter used for calculating the field variables (e.g. stress and deformation) and
 * a contact radius for calculating repulsive forces which prevent individual physical bodies from penetrating each
 * other."
 *
 * The per-atom SMD data are internal per-atom props ('#' cannot start an i_name or d_name): the volume (vfrac), the
 * reference position (x0), the contact radius, the deformation gradient (9 values, row-major), the estimated velocity
 * (vest), the Cauchy stress (6 values: xx yy zz xy xz yz), the internal energy per unit mass (esph) and its rate
 * (desph), the effective plastic strain and its rate, and the damage. vfrac, x0, radius and rmass are fields of
 * SimState like the peri style.
 *
 * Defaults of an atom made by create_atoms (docs.lammps.org/create_atoms.html for the molecule and radius of the
 * style; the rest is the same as native create_atoms, see SMD_DEFAULTS): molecule 1, vfrac 1, rmass 1, radius 0.5,
 * contact radius 0.5, x0 = x, deformation gradient = identity, all other values 0.
 */

export const SMD_CONTACT = '#smdcontact';
export const SMD_DEFGRAD = '#smdF';
export const SMD_VEST = '#smdvest';
export const SMD_STRESS = '#smdstress';
export const SMD_ESPH = '#smdesph';
export const SMD_DESPH = '#smddesph';
export const SMD_EPS = '#smdeps';
export const SMD_EPSRATE = '#smdepsrate';
export const SMD_DAMAGE = '#smddamage';

/** Column count of each vector-valued SMD prop (0 = scalar). */
export const SMD_COLS: Record<string, number> = {
  [SMD_CONTACT]: 0, [SMD_DEFGRAD]: 9, [SMD_VEST]: 3, [SMD_STRESS]: 6, [SMD_ESPH]: 0, [SMD_DESPH]: 0,
  [SMD_EPS]: 0, [SMD_EPSRATE]: 0, [SMD_DAMAGE]: 0,
};

/** Scalar defaults of a created SMD atom (SMD_DEFGRAD is filled with the identity separately). */
export const SMD_DEFAULTS: Record<string, number> = {
  [SMD_CONTACT]: 0.5, [SMD_ESPH]: 0, [SMD_DESPH]: 0, [SMD_EPS]: 0, [SMD_EPSRATE]: 0, [SMD_DAMAGE]: 0,
};

/** Molecule ID given to a created SMD atom (create_atoms: the style's create_atom_post sets molecule 1). */
export const SMD_DEFAULT_MOLECULE = 1;

/** atom_style smd: the atom style has the SMD data. */
export const isSmdStyle = (st: AtomStyle): boolean => st === 'smd';

/** The internal props of an SMD atom style (empty arrays, filled as atoms are added). */
export const smdCustom = (st: AtomStyle): Map<string, CustomProp> => {
  const m = new Map<string, CustomProp>();
  if (!isSmdStyle(st)) return m;
  for (const [name, cols] of Object.entries(SMD_COLS)) m.set(name, { int: false, cols, data: new Float64Array(0) });
  return m;
};

/** Fill the SMD props of atoms [n0, n) that were not given: defaults of a created atom (identity deformation gradient). */
export const fillSmdDefaults = (s: SimState, n0: number, n: number, given: ReadonlySet<string>): void => {
  for (const [name, value] of Object.entries(SMD_DEFAULTS)) {
    if (given.has(name)) continue;
    s.custom.get(name)!.data.fill(value, n0, n);
  }
  if (!given.has(SMD_DEFGRAD)) {
    const F = s.custom.get(SMD_DEFGRAD)!.data;
    for (let i = n0; i < n; i++) {
      F.fill(0, 9 * i, 9 * i + 9);
      F[9 * i] = 1; F[9 * i + 4] = 1; F[9 * i + 8] = 1;
    }
  }
  if (!given.has(SMD_VEST)) s.custom.get(SMD_VEST)!.data.fill(0, 3 * n0, 3 * n);
  if (!given.has(SMD_STRESS)) s.custom.get(SMD_STRESS)!.data.fill(0, 6 * n0, 6 * n);
};
