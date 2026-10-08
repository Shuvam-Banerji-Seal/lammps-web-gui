import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf, CUSTOM_ATTR, customAttr } from '../atoms';

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
    const per = this.sys.peratomEnergy();
    const a = this.sys.forces();
    const termEnergy: Record<string, number> = {
      pair: a.evdwl + a.ecoul, bond: a.ebond, angle: a.eangle, dihedral: a.edihed, improper: a.eimp, kspace: a.elong,
    };
    for (const t of PE_KEYWORDS) {
      if (t === 'fix' || this.terms.has(t)) continue;
      const e = termEnergy[t];
      if (e !== 0) throw new StyleError(`compute ${this.id} (pe/atom): the '${t}' term is present (energy ${e}) but is not among the requested keywords`);
    }
    if (this.terms.has('fix')) {
      const fixE = this.sys.fixEnergy();
      if (fixE !== 0) throw new StyleError(`compute ${this.id} (pe/atom): per-atom fix energy is not available (fix energy = ${fixE})`);
    }
    this.vectorAtom = new Float64Array(s.n);
    const eatom = per.eatom!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.vectorAtom[i] = eatom[i];
    }
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
    const termVirial: Record<string, Float64Array> = {
      pair: a.virial, bond: a.vbond, angle: a.vangle, dihedral: a.vdihed, improper: a.vimp, kspace: a.vlong,
    };
    for (const t of VIRIAL_TERMS) {
      if (t === 'fix' || this.terms.has(t)) continue;
      const w = termVirial[t];
      for (let c = 0; c < 6; c++) {
        if (w[c] !== 0) throw new StyleError(`compute ${this.id} (stress/atom): the '${t}' term is present (virial component ${c} = ${w[c]}) but is not among the requested keywords`);
      }
    }
    if (this.terms.has('fix')) {
      const w = new Float64Array(6);
      this.sys.fixVirial(w);
      for (let c = 0; c < 6; c++) {
        if (w[c] !== 0) throw new StyleError(`compute ${this.id} (stress/atom): per-atom fix virial is not available (fix virial component ${c} = ${w[c]})`);
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
    const vatom = per.vatom!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      for (let c = 0; c < 6; c++) arr[6 * i + c] -= vatom[6 * i + c];
    }
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

const SPHERE_ATTRS = new Set(['radius', 'diameter', 'omegax', 'omegay', 'omegaz', 'tqx', 'tqy', 'tqz']);

export class ComputePropertyAtom extends Compute {
  readonly style = 'property/atom';
  private readonly attrs: string[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError('usage: compute ID group-ID property/atom input1 input2 ...');
    for (const a of args) {
      // compute_property_atom.html also lists "i_name, d_name, i2_name[I], d2_name[I]" (fix
      // property/atom) and the finite-size sphere attributes radius, diameter, omega*, tq*
      if (!(PROPERTY_ATTRS as readonly string[]).includes(a) && !SPHERE_ATTRS.has(a) && !CUSTOM_ATTR.test(a)) {
        throw new StyleError(`compute property/atom: attribute '${a}' is not supported by the browser engine`);
      }
      if (SPHERE_ATTRS.has(a) && !sys.state.radius) throw new StyleError(`compute property/atom ${a} needs atom_style sphere`);
      if (CUSTOM_ATTR.test(a)) customAttr(sys.state, a);
    }
    this.attrs = [...args];
    this.peratomFlag = true;
    this.sizePeratomCols = args.length > 1 ? args.length : 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const m = this.attrs.length;
    const out = new Float64Array(m > 1 ? m * s.n : s.n);
    const u = [0, 0, 0];
    const lam = [0, 0, 0];
    for (let k = 0; k < m; k++) {
      const attr = this.attrs[k];
      // direction component: 'x', 'ix'/'vx'/'fx' end in it; 'xs'/'xu' start with it
      let d = 'xyz'.indexOf(attr[attr.length - 1]);
      if (d < 0) d = 'xyz'.indexOf(attr[0]);
      const set = (i: number, val: number) => { out[m > 1 ? m * i + k : i] = val; };
      const custom = customAttr(s, attr);
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        if (custom) { set(i, custom(i)); continue; }
        switch (attr) {
          case 'radius': set(i, s.radius![i]); break;
          case 'diameter': set(i, 2 * s.radius![i]); break;
          case 'omegax': case 'omegay': case 'omegaz': set(i, s.omega![3 * i + d]); break;
          case 'tqx': case 'tqy': case 'tqz': set(i, s.torque![3 * i + d]); break;
          case 'id': set(i, s.id[i]); break;
          case 'mol': set(i, s.molecule[i]); break;
          case 'proc': set(i, 0); break;
          case 'type': set(i, s.type[i]); break;
          case 'mass': set(i, massOf(s, i)); break;
          case 'x': case 'y': case 'z': set(i, s.x[3 * i + d]); break;
          case 'xs': case 'ys': case 'zs':
            g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
            set(i, lam[d]);
            break;
          case 'xu': case 'yu': case 'zu':
            g.unwrap(s.x, s.image, i, u);
            set(i, u[d]);
            break;
          case 'ix': case 'iy': case 'iz': set(i, s.image[3 * i + d]); break;
          case 'vx': case 'vy': case 'vz': set(i, s.v[3 * i + d]); break;
          case 'fx': case 'fy': case 'fz': set(i, s.f[3 * i + d]); break;
          case 'q': set(i, s.q[i]); break;
        }
      }
    }
    if (m > 1) this.arrayAtom = out; else this.vectorAtom = out;
  }
}
