import { Compute } from './compute';
import { StyleError } from '../force/types';
import { parseBodyStyle } from '../bodies';
import type { System } from '../system';
import { massOf, CUSTOM_ATTR, customAttr, isEllipsoid, isBpmSphereStyle, HEAT_FLOW, HEAT_TEMPERATURE, ELECTRON_SPIN, ELECTRON_RADIUS, ELECTRON_RVEL, ELECTRON_RFORCE, isSuperEllipsoidStyle, isLineStyle, isTriStyle, isLineAtom, isTriAtom, LINE_BONUS, TRI_BONUS, TRI_BONUS_COLS } from '../atoms';
import { lineEnds, triangleCorner } from '../lineTri';
import { SUPER_INERTIA, superBlock } from '../superellipsoid';

/*
 * Per-atom computes: ke/atom, pe/atom, stress/atom and property/atom.
 * Each class cites and quotes the docs.lammps.org page it implements.
 */

const PE_KEYWORDS = ['pair', 'bond', 'angle', 'dihedral', 'improper', 'kspace', 'fix'] as const;
const VIRIAL_TERMS = ['pair', 'bond', 'angle', 'dihedral', 'improper', 'kspace', 'fix'] as const;
const STRESS_KEYWORDS = ['ke', ...VIRIAL_TERMS, 'virial'] as const;

/*
 * compute ID group-ID ke/atom — docs.lammps.org/compute_ke_atom.html:
 *   compute ID group-ID ke/atom
 * "Define a computation that calculates the per-atom translational kinetic
 * energy for each atom in a group." "The kinetic energy is simply
 * :math:`\frac12 m v^2`, where :math:`m` is the mass and :math:`v` is the
 * velocity of each atom."
 * "The value of the kinetic energy will be 0.0 for atoms not in the
 * specified compute group." "This compute calculates a per-atom vector" and
 * "The per-atom vector values will be in energy units." Default: none.
 */
/**
 * atom_style electron attributes of compute property/atom: compute_property_atom.html lists "espin = electron spin",
 * "eradius = electron radius", "ervel = electron radial velocity" and "erforce = electron radial force". Measured with
 * native LAMMPS (black box): spin is accepted as well and gives the electron spin.
 */
const ELECTRON_ATTRS = new Map<string, string>([['spin', ELECTRON_SPIN], ['espin', ELECTRON_SPIN], ['eradius', ELECTRON_RADIUS], ['ervel', ELECTRON_RVEL], ['erforce', ELECTRON_RFORCE]]);

export class ComputeKEAtom extends Compute {
  readonly style = 'ke/atom';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('compute ke/atom takes no arguments');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const v = s.v;
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.vectorAtom[i] = 0.5 * s.units.mvv2e * m * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
    }
  }
}

/*
 * compute ID group-ID pe/atom keyword ... — docs.lammps.org/compute_pe_atom.html:
 *   compute ID group-ID pe/atom keyword ...
 * "zero or more keywords may be appended" and
 *   keyword = pair or bond or angle or dihedral or improper or kspace or fix
 * "Define a computation that computes the per-atom potential energy for
 * each atom in a group." "If no extra keywords are listed, then the
 * potential energy is the sum of pair, bond, angle, dihedral, improper,
 * k-space (long-range), and fix energy (i.e., it is as though all the
 * keywords were listed). If any extra keywords are listed, then only those
 * components are summed to compute the potential energy."
 * "Note that the energy of each atom is due to its interaction with all
 * other atoms in the simulation, not just with other atoms in the group."
 * "For an energy contribution produced by a small set of atoms (e.g., 4
 * atoms in a dihedral or 3 atoms in a Tersoff 3-body interaction), that
 * energy is assigned in equal portions to each atom in the set (e.g., 1/4
 * of the dihedral energy to each of the four atoms)."
 * "This compute calculates a per-atom vector" and "The per-atom vector
 * values will be in energy units." Default: none.
 *
 * Engine limitation: the force field tallies one per-atom energy summed
 * over all its terms (pair, bonded, kspace); there is no per-term split and
 * fixes provide no per-atom energy. A keyword subset that differs from the
 * terms present, or a nonzero fix energy, is a StyleError (never ignored).
 * "The per-atom energy does not include any Lennard-Jones tail corrections
 * to the energy added by the pair_modify tail yes command" (note on the
 * page) — tail corrections are global only, as in the engine.
 */
export class ComputePEAtom extends Compute {
  readonly style = 'pe/atom';
  private readonly terms: Set<string>;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    for (const a of args) {
      if (!(PE_KEYWORDS as readonly string[]).includes(a)) throw new StyleError(`unknown compute pe/atom keyword '${a}'`);
    }
    this.terms = new Set(args.length ? args : PE_KEYWORDS);
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
    this.needsEatom = true;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    this.sys.peratomEnergy();
    const ff = this.sys.ff;
    const out = new Float64Array(s.n);
    for (const t of PE_KEYWORDS) {
      if (t === 'fix' || !this.terms.has(t)) continue;
      const arr = ff.eatomTerm[t];
      if (arr) for (let i = 0; i < s.n; i++) out[i] += arr[i];
    }
    // compute_pe_atom.html: "The fix_modify energy yes command must also be specified if a fix is
    // to contribute per-atom potential energy to this command." and "See the doc pages for individual
    // fixes for details of which ones compute a per-atom potential energy": a fix without one adds
    // nothing (measured with native LAMMPS, black box: fix wall/lj93 and wall/harmonic with
    // fix_modify energy yes leave compute pe/atom fix at 0).
    if (this.terms.has('fix')) {
      for (const f of this.sys.fixes) if (f.thermoEnergy && f.energyAtom) f.energyAtom(out);
    }
    for (let i = 0; i < s.n; i++) if (!(s.mask[i] & this.groupBit)) out[i] = 0;
    this.vectorAtom = out;
  }
}

/*
 * compute ID group-ID stress/atom temp-ID keyword ... —
 * docs.lammps.org/compute_stress_atom.html:
 *   compute ID group-ID style temp-ID keyword ...
 * "temp-ID = ID of compute that calculates temperature, can be NULL if not
 * needed" and
 *   keyword = ke or pair or bond or angle or dihedral or improper or kspace
 *   or fix or virial
 * "The stress tensor for atom I is given by the following formula, where a
 * and b take on values x, y, z to generate the components of the tensor:"
 *   S_ab = - m v_a v_b - W_ab
 * "If no extra keywords are listed, the kinetic contribution and all of the
 * virial contribution terms are included in the per-atom stress tensor. If
 * any extra keywords are listed, only those terms are summed to compute the
 * tensor. The virial keyword means include all terms except the kinetic
 * energy ke."
 * "The temp-ID argument can be used to affect the per-atom velocities used
 * in the kinetic energy contribution to the total stress. If the kinetic
 * energy is not included in the stress, than the temperature compute is not
 * used and can be specified as NULL. If the kinetic energy is included and
 * you wish to use atom velocities as-is, then temp-ID can also be specified
 * as NULL. If desired, the specified temperature compute can be one that
 * subtracts off a bias to leave each atom with only a thermal velocity to
 * use in the formula above, e.g. by subtracting a background streaming
 * velocity."
 * "Compute stress/atom calculates a per-atom array with 6 columns" and
 * "The ordering of the 6 columns for stress/atom is as follows: xx, yy, zz,
 * xy, xz, yz." "The per-atom array values will be in pressure*volume units"
 * ("It is also really a stress*volume formulation"). Default: "By default
 * the compute includes contributions from the keywords:
 * ke pair bond angle dihedral improper kspace fix".
 *
 * Engine limitation: the force field tallies one per-atom virial summed
 * over all its terms (pair, bonded, kspace); there is no per-term split and
 * fixes provide no per-atom virial. A keyword subset that differs from the
 * terms present, or a nonzero fix virial, is a StyleError (never ignored).
 */
export class ComputeStressAtom extends Compute {
  readonly style = 'stress/atom';
  tempId: string | null;
  private readonly terms: Set<string>;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('usage: compute ID group-ID stress/atom temp-ID [keywords]');
    this.tempId = args[0] === 'NULL' ? null : args[0];
    const rest = args.slice(1);
    for (const a of rest) {
      if (!(STRESS_KEYWORDS as readonly string[]).includes(a)) throw new StyleError(`unknown compute stress/atom keyword '${a}'`);
    }
    const expanded = rest.includes('virial') ? [...rest.filter((k) => k !== 'virial'), ...VIRIAL_TERMS] : rest;
    this.terms = new Set(expanded.length ? expanded : ['ke', ...VIRIAL_TERMS]);
    this.peratomFlag = true;
    this.sizePeratomCols = 6;
    this.needsVatom = true;
  }

  init(): void {
    if (this.tempId) {
      const c = this.sys.compute(this.tempId);
      if (!c.tempFlag) throw new StyleError(`compute ${this.id}: compute ${this.tempId} does not compute a temperature`);
    }
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const v = s.v;
    const per = this.sys.peratomEnergy();
    const a = this.sys.forces();
    const vsum = new Float64Array(6 * s.n);
    for (const t of VIRIAL_TERMS) {
      if (t === 'fix' || !this.terms.has(t)) continue;
      const arr = this.sys.ff.vatomTerm[t];
      if (arr) for (let k = 0; k < 6 * s.n; k++) vsum[k] += arr[k];
    }
    if (this.terms.has('fix')) {
      for (const f of this.sys.fixes) {
        if (!f.thermoVirial) continue;
        if (f.virialAtom) { f.virialAtom(vsum); continue; }
        for (let c = 0; c < 6; c++) {
          if (f.virial[c] !== 0) throw new StyleError(`compute ${this.id} (stress/atom): fix ${f.id} (${f.style}) has a virial but no per-atom virial in this engine`);
        }
      }
    }
    this.arrayAtom = new Float64Array(6 * s.n);
    const arr = this.arrayAtom;
    if (this.terms.has('ke')) {
      const t = this.tempId ? this.sys.compute(this.tempId) : null;
      if (t && t.hasBias()) { t.computeBias(); t.removeBiasAll(); }
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const k = s.units.mvv2e * massOf(s, i);
        arr[6 * i] = -k * v[3 * i] * v[3 * i];
        arr[6 * i + 1] = -k * v[3 * i + 1] * v[3 * i + 1];
        arr[6 * i + 2] = -k * v[3 * i + 2] * v[3 * i + 2];
        arr[6 * i + 3] = -k * v[3 * i] * v[3 * i + 1];
        arr[6 * i + 4] = -k * v[3 * i] * v[3 * i + 2];
        arr[6 * i + 5] = -k * v[3 * i + 1] * v[3 * i + 2];
      }
      if (t && t.hasBias()) t.restoreBiasAll();
    }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      for (let c = 0; c < 6; c++) arr[6 * i + c] -= vsum[6 * i + c];
    }
    // compute_stress_atom.html: "The per-atom array values will be in pressure*volume units"; the
    // kinetic and virial sums above are in energy units, so both take the pressure conversion
    // (measured with native LAMMPS, black box: in units real the summed values are nktv2p = 68568.415
    // times the energy-unit sums, for the kinetic and the virial parts alike)
    const nktv2p = s.units.nktv2p;
    if (nktv2p !== 1) for (let k = 0; k < arr.length; k++) arr[k] *= nktv2p;
  }
}

/*
 * compute ID group-ID property/atom input1 input2 ... —
 * docs.lammps.org/compute_property_atom.html:
 *   compute ID group-ID property/atom input1 input2 ...
 * "input = one or more atom attributes" with (extract):
 *   id = atom ID, mol = molecule ID, proc = ID of processor that owns atom,
 *   type = atom type, mass = atom mass,
 *   x,y,z = unscaled atom coordinates, xs,ys,zs = scaled atom coordinates,
 *   xu,yu,zu = unwrapped atom coordinates, ix,iy,iz = box image that the
 *   atom is in, vx,vy,vz = atom velocities, fx,fy,fz = forces on atoms,
 *   q = atom charge
 * "Define a computation that simply stores atom attributes for each atom in
 * the group." "The values are stored in a per-atom vector or array as
 * discussed below. Zeroes are stored for atoms not in the specified group or
 * for quantities that are not defined for a particular particle in the
 * group" (e.g. shapex for a non-ellipsoid). "This compute calculates a per-atom vector or per-atom array
 * depending on the number of input values. If a single input is specified, a
 * per-atom vector is produced. If two or more inputs are specified, a
 * per-atom array is produced where the number of columns = the number of
 * inputs." "The vector or array values will be in whatever units the
 * corresponding attribute is in". Default: none.
 *
 * Attributes without backing data in this engine's atom styles (dipoles,
 * spins, bonds counts, finite-size shapes, package properties, custom
 * fix property/atom vectors, ...) are StyleErrors, never silent zeroes.
 */
const PROPERTY_ATTRS = [
  'id', 'mol', 'proc', 'type', 'mass',
  'x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu', 'ix', 'iy', 'iz',
  'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'q',
] as const;

const SPHERE_ATTRS = new Set(['radius', 'diameter', 'omegax', 'omegay', 'omegaz']);
/** compute_property_atom.html: "*tqx,tqy,tqz* = torque on finite-size particles" (sphere and ellipsoid styles). */
const TORQUE_ATTRS = new Set(['tqx', 'tqy', 'tqz']);
/**
 * compute_property_atom.html: "*angmomx,angmomy,angmomz* = angular momentum of aspherical particle",
 * "*shapex,shapey,shapez* = 3 diameters of aspherical particle", "*quatw,quati,quatj,quatk* = quaternion
 * components for aspherical or body particles". Measured with native LAMMPS (black box): a point particle of
 * atom style ellipsoid reports shape 1 1 1 and quaternion 1 0 0 0 (not the zeroes the page describes).
 */
const ELLIPSOID_ATTRS = new Set(['angmomx', 'angmomy', 'angmomz', 'shapex', 'shapey', 'shapez', 'quatw', 'quati', 'quatj', 'quatk']);
/** compute_property_atom.html: block1, block2 (superellipsoids) and inertiax, inertiay, inertiaz (superellipsoids). */
const SUPER_ATTRS = new Set(['block1', 'block2', 'inertiax', 'inertiay', 'inertiaz']);
/** compute_property_atom.html: "*mux,muy,muz* = orientation of dipole moment of atom" (and mu, its magnitude). */
const DIPOLE_ATTRS = new Set(['mux', 'muy', 'muz', 'mu']);
/**
 * compute_property_atom.html: "*spx, spy, spz* = direction of the atomic magnetic spin", "*sp* = magintude of
 * atomic magnetic spin moment", the fmx, fmy, fmz columns of the page (magnetic force acting on the atomic spin) (the precession
 * vector of the spin pair style). The magnitude is in Bohr magnetons.
 */
const SPIN_ATTRS = new Set(['spx', 'spy', 'spz', 'sp', 'fmx', 'fmy', 'fmz']);
/**
 * compute_property_atom.html: "*temperature* = internal temperature of spherical particle" and "*heatflow* = internal
 * heat flow of spherical particle" (values from fix property/atom temperature heatflow).
 */
const HEAT_ATTRS: Record<string, string> = { temperature: HEAT_TEMPERATURE, heatflow: HEAT_FLOW };
/**
 * compute_property_atom.html: "*end1x, end1y, end1z, end2x, end2y, end2z* = end points of line segment" and
 * "*corner1x, corner1y, corner1z, corner2x, ...* = corner points of triangle" (atom_style line and tri).
 * Native LAMMPS (src/compute_property_atom.cpp at b332f28, pack_end1x and pack_corner1x): the x and y ends are the
 * centre -/+ half the length along theta, and a group atom that is not a line (or triangle) gives 0; the z ends
 * (pack_end1z, pack_end2z) are the atom's own z for every group atom, and the corner z is the z of the rotated
 * corner (pack_corner1z).
 */
const LINE_END = /^end([12])([xyz])$/;
const TRI_CORNER = /^corner([123])([xyz])$/;

/**
 * The reason a per-atom attribute cannot be stored by this atom style, or null when it can. Shared by compute
 * property/atom and fix store/state (both take the same attribute words). Custom i_/d_ names are checked by customAttr.
 */
export const peratomAttrProblem = (sys: System, a: string): string | null => {
  if (HEAT_ATTRS[a] && !sys.state.custom.has(HEAT_ATTRS[a])) return 'needs fix property/atom temperature heatflow';
  // compute_property_atom.html: "espin, eradius, ervel, erforce" (atom_style electron; "spin" is the older name of espin)
  const ek = ELECTRON_ATTRS.get(a);
  if (ek && !sys.state.custom.has(ek)) return 'needs atom_style electron';
  if (LINE_END.test(a) && !isLineStyle(sys.atomStyle)) return 'requires atom style line';
  if (TRI_CORNER.test(a) && !isTriStyle(sys.atomStyle)) return 'requires atom style tri';
  if (SPHERE_ATTRS.has(a) && !sys.state.radius) return 'needs atom_style sphere';
  if (TORQUE_ATTRS.has(a) && !sys.state.torque) return 'needs atom_style sphere or ellipsoid';
  // atom_style bpm/sphere stores a quaternion (atoms.ts isBpmSphereStyle) but no shape
  // atom_style body stores the orientation and the angular momentum of each body (bodies.ts, compute_property_atom.html: "quaternion ... for aspherical or body particles")
  const bodyQuatOrAngmom = parseBodyStyle(sys.atomStyle) !== null && (/^quat[wijk]$/.test(a) || /^angmom[xyz]$/.test(a));
  if (ELLIPSOID_ATTRS.has(a) && !sys.state.shape && !bodyQuatOrAngmom && !(isBpmSphereStyle(sys.atomStyle) && /^quat[wijk]$/.test(a))) return 'needs atom_style ellipsoid';
  if (SUPER_ATTRS.has(a) && !isSuperEllipsoidStyle(sys.atomStyle)) return 'requires atom style ellipsoid superellipsoid';
  if (DIPOLE_ATTRS.has(a) && !sys.state.mu) return 'needs atom_style dipole';
  if (SPIN_ATTRS.has(a) && !sys.state.sp) return 'needs atom_style spin';
  if (CUSTOM_ATTR.test(a)) customAttr(sys.state, a);
  return null;
};

/**
 * Writes the per-atom values of attribute `attr` for the atoms in the group (groupBit) into out at out[stride * i + col];
 * atoms outside the group are not written (callers zero them). The attribute must pass peratomAttrProblem.
 */
export const fillPeratomAttr = (sys: System, attr: string, groupBit: number, out: Float64Array, stride: number, col: number): void => {
  const s = sys.state;
  const g = sys.geom;
  const u = [0, 0, 0];
  const lam = [0, 0, 0];
  // direction component: 'x', 'ix'/'vx'/'fx' end in it; 'xs'/'xu'/'xsu' start with it
  let d = 'xyz'.indexOf(attr[attr.length - 1]);
  if (d < 0) d = 'xyz'.indexOf(attr[0]);
  const custom = customAttr(s, attr);
  const heat = HEAT_ATTRS[attr] ? s.custom.get(HEAT_ATTRS[attr])!.data : null;
  const ek = ELECTRON_ATTRS.get(attr);
  const electron = ek ? s.custom.get(ek)!.data : null;
  const le = LINE_END.exec(attr), tc = TRI_CORNER.exec(attr);
  const lineBonus = le ? s.custom.get(LINE_BONUS)!.data : null;
  const triBonus = tc ? s.custom.get(TRI_BONUS)!.data : null;
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & groupBit)) continue;
    const set = (val: number) => { out[stride * i + col] = val; };
    if (custom) { set(custom(i)); continue; }
    if (heat) { set(heat[i]); continue; }
    if (electron) { set(electron[i]); continue; }
    if (le) {
      // pack_end1z / pack_end2z: the atom's z for every group atom
      if (d === 2) { set(s.x[3 * i + 2]); continue; }
      if (!isLineAtom(s, i)) { set(0); continue; }
      const ends = lineEnds(s.x[3 * i], s.x[3 * i + 1], lineBonus![2 * i], lineBonus![2 * i + 1]);
      // lineEnds centres the segment on (x, y) and returns (end1x, end1y, end2x, end2y)
      set(ends[2 * (Number(le[1]) - 1) + d]);
      continue;
    }
    if (tc) {
      // pack_corner*z: the rotated corner's own z (corner = centre + R c, as native pack_corner1x)
      if (!isTriAtom(s, i)) { set(0); continue; }
      const corners = triBonus!.subarray(TRI_BONUS_COLS * i, TRI_BONUS_COLS * i + 9);
      const q = s.quat!.subarray(4 * i, 4 * i + 4);
      set(triangleCorner([s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]], q, corners, Number(tc[1]) - 1)[d]);
      continue;
    }
    switch (attr) {
      case 'radius': set(s.radius![i]); break;
      case 'diameter': set(2 * s.radius![i]); break;
      case 'omegax': case 'omegay': case 'omegaz': set(s.omega![3 * i + d]); break;
      case 'tqx': case 'tqy': case 'tqz': set(s.torque![3 * i + d]); break;
      case 'angmomx': case 'angmomy': case 'angmomz': set(s.angmom![3 * i + d]); break;
      case 'shapex': case 'shapey': case 'shapez': set(isEllipsoid(s, i) ? 2 * s.shape![3 * i + d] : 1); break;
      case 'block1': case 'block2': set(superBlock(s, i)[attr === 'block1' ? 0 : 1]); break;
      case 'inertiax': case 'inertiay': case 'inertiaz': set(s.custom.get(SUPER_INERTIA)!.data[3 * i + d]); break;
      case 'quatw': case 'quati': case 'quatj': case 'quatk':
        set((isEllipsoid(s, i) || (s.quat && (isBpmSphereStyle(s.atomStyle) || parseBodyStyle(s.atomStyle) !== null))) ? s.quat![4 * i + 'wijk'.indexOf(attr[4])] : attr === 'quatw' ? 1 : 0);
        break;
      case 'mux': case 'muy': case 'muz': set(s.mu![4 * i + 'xyz'.indexOf(attr[2])]); break;
      case 'mu': set(s.mu![4 * i + 3]); break;
      case 'spx': case 'spy': case 'spz': set(s.sp![4 * i + 'xyz'.indexOf(attr[2])]); break;
      case 'sp': set(s.sp![4 * i + 3]); break;
      case 'fmx': case 'fmy': case 'fmz': set(s.fm ? s.fm[3 * i + 'xyz'.indexOf(attr[2])] : 0); break;
      case 'id': set(s.id[i]); break;
      case 'mol': set(s.molecule[i]); break;
      case 'proc': set(0); break;
      case 'type': set(s.type[i]); break;
      case 'mass': set(massOf(s, i)); break;
      case 'x': case 'y': case 'z': set(s.x[3 * i + d]); break;
      case 'xs': case 'ys': case 'zs':
        g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
        set(lam[d]);
        break;
      // fix store/state (fix_store_state.html): "xsu,ysu,zsu = scaled unwrapped atom coordinates"; native
      // pack_xsu is the scaled position plus the image count (src/fix_store_state.cpp at b332f28)
      case 'xsu': case 'ysu': case 'zsu':
        g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
        set(lam[d] + s.image[3 * i + d]);
        break;
      case 'xu': case 'yu': case 'zu':
        g.unwrap(s.x, s.image, i, u);
        set(u[d]);
        break;
      case 'ix': case 'iy': case 'iz': set(s.image[3 * i + d]); break;
      case 'vx': case 'vy': case 'vz': set(s.v[3 * i + d]); break;
      case 'fx': case 'fy': case 'fz': set(s.f[3 * i + d]); break;
      case 'q': set(s.q[i]); break;
    }
  }
};

export class ComputePropertyAtom extends Compute {
  readonly style = 'property/atom';
  private readonly attrs: string[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError('usage: compute ID group-ID property/atom input1 input2 ...');
    for (const a of args) {
      // compute_property_atom.html also lists "i_name, d_name, i2_name[I], d2_name[I]" (fix
      // property/atom) and the finite-size sphere attributes radius, diameter, omega*, tq*
      if (!(PROPERTY_ATTRS as readonly string[]).includes(a) && !SPHERE_ATTRS.has(a) && !TORQUE_ATTRS.has(a) && !ELLIPSOID_ATTRS.has(a) && !SUPER_ATTRS.has(a) && !DIPOLE_ATTRS.has(a) && !SPIN_ATTRS.has(a) && !CUSTOM_ATTR.test(a) && !HEAT_ATTRS[a] && !ELECTRON_ATTRS.has(a) && !LINE_END.test(a) && !TRI_CORNER.test(a)) {
        throw new StyleError(`compute property/atom: attribute '${a}' is not supported by the browser engine`);
      }
      const problem = peratomAttrProblem(sys, a);
      if (problem) throw new StyleError(`compute property/atom ${a} ${problem}`);
    }
    this.attrs = [...args];
    this.peratomFlag = true;
    this.sizePeratomCols = args.length > 1 ? args.length : 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const m = this.attrs.length;
    const out = new Float64Array(m > 1 ? m * s.n : s.n);
    for (let k = 0; k < m; k++) fillPeratomAttr(this.sys, this.attrs[k], this.groupBit, out, m > 1 ? m : 1, m > 1 ? k : 0);
    if (m > 1) this.arrayAtom = out; else this.vectorAtom = out;
  }
}
