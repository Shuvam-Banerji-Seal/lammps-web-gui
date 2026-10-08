import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { TopoList } from '../types';
import { buildAtomMap } from '../atoms';
import { parseInt_ } from '../force/util';

/*
 * Wave-18 simple computes: nbond/atom (per-atom), count/type (global), and
 * erotate/sphere/atom (per-atom). Every class cites and quotes the
 * docs.lammps.org page it implements; behaviour the pages leave out was
 * measured with native LAMMPS (black box, /home/roy/.local/bin/lmp, probes
 * in plans/scratch/misc18) and is marked "Measured" below.
 */

/*
 * compute ID group-ID nbond/atom keyword value — docs.lammps.org/compute_nbond_atom.html
 * (versionadded 4May2022):
 *   compute ID group-ID nbond/atom keyword value
 *   keyword = *bond/type*
 *     *bond/type* value = *btype*
 *       *btype* = bond type included in count
 * "Define a computation that computes the number of bonds each atom is
 * part of.  Bonds which are broken are not counted in the tally."
 * "The number of bonds will be zero for atoms not in the specified
 * compute group. This compute does not depend on Newton bond settings."
 * "If the keyword *bond/type* is specified, only bonds of *btype* are
 * counted." Output info: "This compute calculates a per-atom vector, which
 * can be accessed by any command that uses per-atom values from a compute
 * as input." Restrictions: "This compute is part of the BPM package.  It is
 * only enabled if LAMMPS was built with that package." — the binary used for
 * the oracle cases has it (Measured: `compute nbond/atom` runs there), and
 * the browser engine has no package concept, so the style is always available.
 *
 * Measured with native LAMMPS (black box; 90-atom branched-chain data file,
 * 72 bonds, run 0, atom groups id 1:3 and molecule 1):
 * - the tally is per atom over every bond it is part of, whether or not the
 *   partner is in the group: group id 1:3 gives 1, 3, 2, i.e. atom 2 keeps its
 *   three bonds although two of them reach atoms 4 and 5 outside the group;
 * - each bond is counted once for each of its two atoms, so the sum over a
 *   group of the whole system is twice the number of bonds (144 for 72 bonds);
 * - bonds that are off (type < 1: broken type-0 bonds, and types turned off
 *   by delete_bonds without remove) are NOT counted: after
 *   `delete_bonds frag bond 2` on atoms 1..4 the per-atom tally falls from
 *   144 to 142 (one bond lost for each of its two atoms), while
 *   `compute nbond/atom bond/type 2` falls from 72 to 70;
 * - `bond/type` compares the bond type exactly, and a type that no bond has
 *   (0, 3, -1, ...) yields a zero vector instead of an error.
 *
 * Measured with native LAMMPS (black box): the error texts this mirrors as
 * StyleErrors are Unknown compute nbond/atom command badkw for an unknown
 * keyword and Illegal compute nbond/atom bond/type command: missing
 * argument(s) for the keyword without a value.
 */
export class ComputeNBondAtom extends Compute {
  readonly style = 'nbond/atom';
  /** 0 = count every bond type, else the single type of the bond/type keyword. */
  private readonly btype: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length === 0) {
      this.btype = 0;
    } else if (args[0] !== 'bond/type') {
      throw new StyleError(`Unknown compute nbond/atom command ${args[0]}`);
    } else if (args.length < 2) {
      throw new StyleError('Illegal compute nbond/atom bond/type command: missing argument(s)');
    } else if (args.length > 2) {
      throw new StyleError('Illegal compute nbond/atom bond/type command: too many arguments');
    } else {
      this.btype = parseInt_(args[1], `compute ${id} (nbond/atom) bond/type`);
    }
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const b = s.topo.bonds;
    const map = buildAtomMap(s);
    const out = new Float64Array(s.n);
    for (let e = 0; e < b.n; e++) {
      const t = b.type[e];
      // broken (0) and turned-off (< 0) bonds are not counted in the tally
      if (t < 1) continue;
      if (this.btype !== 0 && t !== this.btype) continue;
      for (let w = 0; w < 2; w++) {
        const i = map[b.atoms[2 * e + w]];
        if (i < 0 || !(s.mask[i] & this.groupBit)) continue;
        out[i]++;
      }
    }
    this.vectorAtom = out;
  }
}

/*
 * compute ID group-ID count/type mode — docs.lammps.org/compute_count_type.html
 * (versionadded 15Jun2023):
 *   compute ID group-ID count/type mode
 *   mode = *atom* or *bond* or *angle* or *dihedral* or *improper*
 * "Define a computation that counts the current number of atoms for each
 * atom type.  Or the number of bonds (angles, dihedrals, impropers) for
 * each bond (angle, dihedral, improper) type."
 * "for this command, bonds (angles, etc) are the topological kind enumerated
 * in a data file, initially read by the read_data command or defined by the
 * molecule command." — the engine's sys.state.topo lists are exactly that.
 * "These commands can turn off topological bonds (angles, etc) by setting
 * their bond (angle, etc) types to negative values.  This command
 * includes the turned-off bonds (angles, etc) in the count for each type"
 * (so a type -1 bond counts in the slot of type 1; Measured: after
 * `delete_bonds frag bond 1` on a group holding two type-1 bonds, the type-1
 * count stays at its unaltered value of 36 for 36 type-1 bonds).
 * "If the *mode* setting is *atom* then the count of atoms for each atom
 * type is tallied.  Only atoms in the specified group are counted."
 * "If the *mode* setting is *bond* then the count of bonds for each bond
 * type is tallied.  Only bonds with both atoms in the specified group are
 * counted." "For *mode* = *bond*, broken bonds with a bond type of zero are
 * also counted. ... Note that the group setting is ignored for broken bonds;
 * all broken bonds in the system are counted."
 * "If the *mode* setting is *angle* then the count of angles for each angle
 * type is tallied.  Only angles with all 3 atoms in the specified group are
 * counted." ("If the *mode* setting is *dihedral* ... Only dihedrals with all
 * 4 atoms in the specified group are counted." and the same for *improper*.)
 * Output info: "This compute calculates a global vector of counts.  If the
 * mode is *atom* or *bond* or *angle* or *dihedral* or *improper*, then the
 * vector length is the number of atom types or bond types or angle types or
 * dihedral types or improper types, respectively." "If the mode is *bond*
 * this compute also calculates a global scalar which is the number of broken
 * bonds with type = 0". "The scalar and vector values calculated by this
 * compute are both "intensive"." — measured: in lj units (thermo_modify norm
 * yes by default) the counts are the full counts, 36 36 18 for the 90-atom
 * oracle data file, and the same for every group.
 *
 * Measured with native LAMMPS (black box; the 90-atom branched-chain data
 * file with 3 atom, 2 bond, 2 angle, 1 dihedral and 1 improper types):
 * - group all: 36 36 18 atoms, 36 36 bonds, 54 18 angles, 36 dihedrals,
 *   18 impropers (one entry per type, in type order);
 * - group molecule 1:9 (45 atoms) gives 18 18 9 / 18 18 / 27 9 / 18 / 9;
 * - group id 1:3 (a fragment cut through bonds and angles) gives 1 2 0 / 2 0 /
 *   1 0 / 0 / 0, i.e. entries with an atom outside the group are dropped.
 * Measured with native LAMMPS (black box): the error texts this mirrors as
 * StyleErrors are Incorrect number of args for compute count/type command
 * (mode missing or with extra arguments), Invalid compute count/type keyword
 * bogus (unknown mode), and Compute count/type bond command with no bonds
 * defined (raised at init, as native does).
 */
const COUNT_MODES = ['atom', 'bond', 'angle', 'dihedral', 'improper'] as const;
type CountMode = (typeof COUNT_MODES)[number];

export class ComputeCountType extends Compute {
  readonly style = 'count/type';
  private readonly mode: CountMode;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Incorrect number of args for compute count/type command');
    if (!(COUNT_MODES as readonly string[]).includes(args[0])) {
      throw new StyleError(`Invalid compute count/type keyword ${args[0]}`);
    }
    this.mode = args[0] as CountMode;
    this.vectorFlag = true;
    this.scalarFlag = this.mode === 'bond';
    this.sizeVector = this.nTypes();
    this.extvector = 0;
    this.extscalar = 0;
    this.vector = new Float64Array(this.sizeVector);
  }

  /** Vector length per mode (the type count of the tallied topology kind). */
  private nTypes(): number {
    const t = this.sys.state.topo;
    switch (this.mode) {
      case 'atom': return this.sys.state.ntypes;
      case 'bond': return t.nbondtypes;
      case 'angle': return t.nangletypes;
      case 'dihedral': return t.ndihedraltypes;
      default: return t.nimpropertypes;
    }
  }

  private list(): TopoList | null {
    const t = this.sys.state.topo;
    return this.mode === 'atom' ? null : t[`${this.mode}s` as 'bonds'];
  }

  init(): void {
    if (this.mode !== 'atom' && this.nTypes() < 1) {
      throw new StyleError(`Compute ${this.id} (count/type) ${this.mode} command with no ${this.mode}s defined`);
    }
    this.sizeVector = this.nTypes();
    if (this.vector.length !== this.sizeVector) this.vector = new Float64Array(this.sizeVector);
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const n = this.nTypes();
    if (this.sizeVector !== n) { this.sizeVector = n; this.vector = new Float64Array(n); }
    else if (this.vector.length !== n) this.vector = new Float64Array(n);
    const out = this.vector;
    out.fill(0);
    if (this.mode === 'atom') {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const t = s.type[i];
        if (t >= 1 && t <= n) out[t - 1]++;
      }
      return;
    }
    const list = this.list()!;
    const width = list.width;
    const map = buildAtomMap(s);
    for (let e = 0; e < list.n; e++) {
      const t = list.type[e];
      // broken (type 0) bonds are counted by the scalar, not in the vector
      const slot = Math.abs(t);
      if (slot < 1 || slot > n) continue;
      let all = true;
      for (let w = 0; w < width; w++) {
        const i = map[list.atoms[width * e + w]];
        if (i < 0 || !(s.mask[i] & this.groupBit)) { all = false; break; }
      }
      if (all) out[slot - 1]++;
    }
  }

  /** The number of broken (type 0) bonds; native ignores the group for them. */
  protected computeScalar(): number {
    const b = this.sys.state.topo.bonds;
    let n = 0;
    for (let e = 0; e < b.n; e++) if (b.type[e] === 0) n++;
    return n;
  }
}

/*
 * compute ID group-ID erotate/sphere/atom — docs.lammps.org/compute_erotate_sphere_atom.html:
 *   compute ID group-ID erotate/sphere/atom
 * "Define a computation that calculates the rotational kinetic energy for
 * each particle in a group." "The rotational energy is computed as
 * :math:`\frac12 I \omega^2`, where :math:`I` is the moment of inertia for a
 * sphere and :math:`\omega` is the particle's angular velocity."
 * "For 2d models, particles are treated as spheres, not disks, meaning their
 * moment of inertia will be the same as in 3d."
 * "The value of the rotational kinetic energy will be 0.0 for atoms not
 * in the specified compute group or for point particles with a radius of 0.0."
 * Output info: "This compute calculates a per-atom vector" and "The per-atom
 * vector values will be in energy units." Default: none.
 *
 * I = 2/5 m r^2 for a solid sphere (the same inertia the global
 * erotate/sphere uses, see compute_erotate_sphere.html and compute/sphere.ts),
 * with m the per-atom mass of atom_style sphere, r its radius and omega its
 * angular velocity; the energy factor mvv2e converts it to energy units.
 *
 * Measured with native LAMMPS (black box, atom_style sphere, run 0):
 * - lj units, three atoms of mass 3, diameter 2 (r = 1) and omega from
 *   0.5 sin(0.3 id), 0.5 cos(0.7 id), 0.2 sin(1.1 id): the per-atom values
 *   0.11990937800635637, 0.067844487303841169, 0.13086780439792664 equal
 *   0.5 * 0.4 * m * r^2 * (wx^2 + wy^2 + wz^2) with mvv2e = 1, and their sum
 *   equals the global compute erotate/sphere;
 * - real units, mass 3 (g/mol), diameter 2 (A): 286.59029162099409 =
 *   0.5 * 0.4 * 3 * 1^2 * 0.199848 * 2390.0566, i.e. the per-atom value is in
 *   energy units (kcal/mol) with the same mvv2e as every other energy;
 * - a particle whose radius was set to 0 (diameter 0) has the value 0 even
 *   with a nonzero omega, and an atom outside the compute group is 0.
 */
const SPHERE_INERTIA = 0.4;

export class ComputeERotateSphereAtom extends Compute {
  readonly style = 'erotate/sphere/atom';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('compute erotate/sphere/atom takes no arguments');
    if (!sys.state.omega || !sys.state.radius || !sys.state.rmass) {
      throw new StyleError('Compute erotate/sphere/atom requires atom attribute omega');
    }
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const w = s.omega!, r = s.radius!, m = s.rmass!;
    const k = 0.5 * SPHERE_INERTIA * s.units.mvv2e;
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      out[i] = k * m[i] * r[i] * r[i] * (w[3 * i] * w[3 * i] + w[3 * i + 1] * w[3 * i + 1] + w[3 * i + 2] * w[3 * i + 2]);
    }
    this.vectorAtom = out;
  }
}