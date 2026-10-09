import { Fix } from './fix';
import { StyleError } from '../force/types';
import { RanPark } from '../rng';
import { appendAtoms, deleteAtoms, maxAtomId, hasChargeStyle, isMolecularStyle } from '../atoms';
import { appendMolecule, num, posInt, paramValue } from './pour';
import { geometricCenter, rotationMatrix, type MoleculeTemplate } from '../molecule';
import { BlockRegion, SphereRegion, EllipsoidRegion, ConeRegion, type Region } from '../region';
import { newAccum, type Accum, type Bonded, type BondedCompute } from '../force/types';
import { PairHybrid } from '../force/pair/hybrid';
import type { System } from '../system';
import type { SimState, TopoList } from '../types';

/*
 * fix ID group-ID widom N M type seed T keyword values ... — docs.lammps.org/fix_widom.html
 * (plans/lammps-docs/fix_widom.rst). "This fix performs Widom insertions of
 * atoms or molecules at the given temperature as discussed in (Frenkel).
 * Specific uses include computation of Henry constants of small molecules in
 * microporous materials or amorphous systems." "Every N timesteps the fix
 * attempts M number of Widom insertions of atoms or molecules." "If the *mol*
 * keyword is used, only molecule insertions are performed. Conversely, if the
 * *mol* keyword is not used, only atom insertions are performed." Supported
 * keywords: mol, region, full_energy, charge, intra_energy. Everything else
 * throws a StyleError naming it.
 *
 * "The excess chemical potential mu_ex is defined as:" and the doc formula
 *   \mu_{ex} = -kT \ln(<\exp(-(U_{N+1}-U_N)/{k_B T})>)
 * The fix computes a global vector of length 3, listed as "average excess
 * chemical potential on each timestep", "average difference in potential energy
 * on each timestep" and "volume of the insertion region"; the page calls the
 * vector values intensive. The reported values are refreshed only on the
 * timesteps the fix is invoked; between events the last values are held.
 *
 * Measured with native LAMMPS (black box, units lj, atom_style atomic, one
 * fixed atom, pair_style soft, seeds 12345 and 999): the fix consumes its own
 * RanPark stream seeded with the seed and NO discarded draws (unlike fix pour
 * and fix deposit, which discard 30). For each insertion the draws are, in
 * order: x, y, z (each lo + u (hi - lo), lo/hi the simulation box or, with the
 * region keyword, the region's bounding box); then a random rotation axis by
 * rejection sampling: three draws a = 2u-1, b = 2u-1, c = 2u-1, repeated until
 * a^2 + b^2 + c^2 < 1; then one draw for the angle theta = 2 pi u. The molecule
 * is placed with its geometric center at the insertion point and rotated about
 * that point by theta about (a, b, c) (same rotationMatrix as create_atoms mol
 * and fix deposit, but drawn from the fix's RanPark, not a RanMars). An atom
 * insertion draws only x, y, z.
 *
 * Measured with native LAMMPS (black box): the first event is at the first
 * step of a run (step 1 when the fix is defined before the run, like fix pour
 * and fix deposit), later events every N steps. Each event's vector value is
 * the average over its own M insertions only (no running total across events):
 * f_ID[2] = (1/M) sum exp(-dU_j / (k_B T)) and f_ID[1] = -k_B T ln(f_ID[2]),
 * with f_ID[1] = 0 when f_ID[2] is exactly 0 (all Boltzmann factors
 * underflowed). f_ID[3] is the insertion volume (box volume, or the region
 * volume). Measured in real units (T = 298, units.boltz = 0.0019872067):
 * f_ID[1] = -boltz T ln(f_ID[2]) to 5 digits, e.g. the in.widom.spce log's
 * step 10 row (f[2] = 1.6274344e-147 gives 200.15).
 *
 * The energy difference U_{N+1}-U_N is evaluated by temporarily appending the
 * probe atom(s) to the system and taking the total potential-energy difference
 * (which covers every pair, bond, angle, dihedral, improper, kspace and tail
 * term). This is exactly the full_energy path. With full_energy off, native
 * excludes the inserted molecule's intramolecular energy, so that constant
 * (the template's bonded energy) is subtracted. intra_energy is subtracted as
 * the doc requires: "an amount of energy that is subtracted from the final
 * energy when a molecule is inserted".
 *
 * Not measured / not matched: native's region insertion stream could not be
 * reproduced (with a region, even one equal to the whole box, the positions
 * are not a linear map of consecutive RanPark draws). The engine draws the
 * point uniformly in the region's bounding box with its own RanPark stream
 * (rejecting points outside a non-rectangular region, up to 1000 tries) and
 * reports the region's exact volume; the atom/molecule no-region cases match
 * native exactly.
 */

type Vec3 = [number, number, number];

/** A minimal TopoList from template-local entries [type, ids...]. */
const topoList = (entries: number[][], width: 2 | 3 | 4): TopoList => {
  const type = new Int32Array(entries.length);
  const atoms = new Int32Array(entries.length * width);
  entries.forEach((e, k) => {
    type[k] = e[0];
    for (let w = 0; w < width; w++) atoms[k * width + w] = e[1 + w];
  });
  return { n: entries.length, width, type, atoms };
};

/** Largest molecule ID in the system (0 when none). */
const maxMoleculeId = (s: SimState): number => {
  let m = 0;
  for (let i = 0; i < s.n; i++) if (s.molecule[i] > m) m = s.molecule[i];
  return m;
};

/** Bounding box and volume of an insertion region (side in). */
const regionBox = (sys: System, r: Region): { lo: Vec3; hi: Vec3; volume: number } => {
  if (r instanceof BlockRegion) {
    const b = r.b.map((p) => paramValue(sys, p));
    const lo: Vec3 = [b[0], b[2], b[4]], hi: Vec3 = [b[1], b[3], b[5]];
    return { lo, hi, volume: (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) };
  }
  if (r instanceof SphereRegion) {
    const c = r.c.map((p) => paramValue(sys, p));
    const rad = paramValue(sys, r.r);
    return { lo: [c[0] - rad, c[1] - rad, c[2] - rad], hi: [c[0] + rad, c[1] + rad, c[2] + rad], volume: (4 / 3) * Math.PI * rad ** 3 };
  }
  if (r instanceof EllipsoidRegion) {
    const c = r.c.map((p) => paramValue(sys, p));
    const a = r.ax.map((p) => paramValue(sys, p));
    return { lo: [c[0] - a[0], c[1] - a[1], c[2] - a[2]], hi: [c[0] + a[0], c[1] + a[1], c[2] + a[2]], volume: (4 / 3) * Math.PI * a[0] * a[1] * a[2] };
  }
  if (r instanceof ConeRegion) {
    const bb = r.bbox();
    if (!bb) throw new StyleError(`fix widom: region '${r.id}' (${r.style}) has no finite bounding box`);
    const r1 = paramValue(sys, r.radlo), r2 = paramValue(sys, r.radhi);
    const h = paramValue(sys, r.hi) - paramValue(sys, r.lo);
    const volume = r.style === 'cylinder' ? Math.PI * r1 * r1 * Math.abs(h) : (Math.PI / 3) * Math.abs(h) * (r1 * r1 + r1 * r2 + r2 * r2);
    return { lo: [bb.lo[0], bb.lo[1], bb.lo[2]], hi: [bb.hi[0], bb.hi[1], bb.hi[2]], volume };
  }
  throw new StyleError(`fix widom: the volume of a '${r.style}' region is not supported by the browser engine`);
};

export class FixWidom extends Fix {
  readonly style = 'widom';
  readonly N: number;
  readonly M: number;
  readonly type: number;
  readonly T: number;
  private readonly rng: RanPark;
  private readonly molTemplateId: string | null;
  private molTemplate: MoleculeTemplate | null = null;
  private readonly regionId: string | null;
  private reg: { lo: Vec3; hi: Vec3; volume: number } | null = null;
  private region: Region | null = null;
  private fullEnergy = false;
  private readonly charge: number | null;
  private readonly intraEnergy: number;
  private intraMolEnergy = 0;
  private nextStep: number;
  private mu = 0;
  private avgBoltz = 0;
  private volume = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (sys.dimension !== 3) throw new StyleError('fix widom: only 3d simulations are supported');
    if (args.length < 5) throw new StyleError('usage: fix ID group-ID widom N M type seed T keyword values ...');
    this.N = posInt(args[0], 'N');
    this.M = posInt(args[1], 'M');
    this.rng = new RanPark(posInt(args[3], 'seed'));
    this.T = num(args[4], 'T');
    if (!(this.T > 0)) throw new StyleError('fix widom: T must be > 0');
    let molTemplateId: string | null = null;
    let regionId: string | null = null;
    let fullEnergy = false;
    let charge: number | null = null;
    let intraEnergy = 0;
    let i = 5;
    while (i < args.length) {
      const key = args[i];
      const v = (k: number) => {
        if (args[i + k] === undefined) throw new StyleError(`fix widom: keyword '${key}' needs ${k} value(s)`);
        return args[i + k];
      };
      if (key === 'mol') { molTemplateId = v(1); i += 2; }
      else if (key === 'region') { regionId = v(1); i += 2; }
      else if (key === 'full_energy') { fullEnergy = true; i += 1; }
      else if (key === 'charge') { charge = num(v(1), 'charge'); i += 2; }
      else if (key === 'intra_energy') { intraEnergy = num(v(1), 'intra_energy'); i += 2; }
      else throw new StyleError(`fix widom: unknown keyword '${key}'`);
    }
    const typeInt = num(args[2], 'type');
    if (!Number.isInteger(typeInt)) throw new StyleError(`fix widom: type must be an integer, got '${args[2]}'`);
    if (molTemplateId) {
      if (typeInt !== 0) throw new StyleError('fix widom: type must be 0 when the mol keyword is used');
    } else if (typeInt < 1) throw new StyleError('fix widom: type must be a positive integer without the mol keyword');
    this.type = typeInt;
    this.molTemplateId = molTemplateId;
    this.regionId = regionId;
    this.fullEnergy = fullEnergy;
    this.charge = charge;
    this.intraEnergy = intraEnergy;
    this.nextStep = sys.hasBox ? sys.state.step + 1 : 1;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 0;
    this.extvector = 0;
  }

  init(): void {
    const s = this.sys.state;
    const pair = this.sys.ff.pair;
    if (!pair) throw new StyleError('fix widom requires a pair style');
    // fix_widom.html: "The *full_energy* option is needed for systems with complicated
    // potential energy calculations, including the following:" (a list naming long-range electrostatics
    // (kspace), many-body pair styles, hybrid pair styles, eam pair styles and tail corrections) and
    // "In these cases, LAMMPS will automatically apply the *full_energy* keyword and issue a warning message."
    const auto = !!this.sys.ff.kspace || pair.coulLong || pair.tail || pair.manybody || pair instanceof PairHybrid
      || pair.name === 'eam' || pair.name.startsWith('eam/');
    if (auto && !this.fullEnergy) {
      this.fullEnergy = true;
      this.sys.warn('fix widom: full_energy is required for this pair style and was enabled automatically');
    }
    if (this.molTemplateId) {
      if (!isMolecularStyle(s.atomStyle)) throw new StyleError(`fix widom mol: atom_style ${s.atomStyle} cannot store molecule IDs and bonds`);
      const sets = this.sys.molecules.get(this.molTemplateId);
      if (!sets || !sets.length) throw new StyleError(`fix widom mol: molecule template '${this.molTemplateId}' does not exist`);
      this.molTemplate = sets[0];
      if (sets.length > 1 || new Set(this.molTemplate.mol ?? []).size > 1) {
        throw new StyleError(`fix widom mol: molecule template '${this.molTemplateId}' defines more than one molecule, which the browser engine does not support`);
      }
      // Measured with native LAMMPS (black box): an atom of the group with molecule ID 0 is an error
      // (All mol IDs should be set for fix widom group atoms).
      for (let i = 0; i < s.n; i++) {
        if ((s.mask[i] & this.groupBit) !== 0 && !(s.molecule[i] > 0)) {
          throw new StyleError('fix widom: All mol IDs should be set for fix widom group atoms');
        }
      }
      for (const ty of this.molTemplate.type) if (ty < 1 || ty > s.ntypes) throw new StyleError(`fix widom mol: molecule template atom type ${ty} is outside 1..${s.ntypes}`);
      if (this.molTemplate.q && !hasChargeStyle(s.atomStyle)) throw new StyleError(`fix widom mol: molecule template '${this.molTemplateId}' has charges, which atom_style ${s.atomStyle} cannot store`);
      if (!this.fullEnergy) this.intraMolEnergy = this.moleculeIntraEnergy(this.molTemplate);
    } else {
      if (this.type > s.ntypes) throw new StyleError(`fix widom: atom type ${this.type} is larger than ntypes ${s.ntypes}`);
      if (this.charge !== null && !hasChargeStyle(s.atomStyle)) throw new StyleError(`fix widom charge: atom_style ${s.atomStyle} cannot store a charge`);
    }
    if (this.regionId) {
      const r = this.sys.region(this.regionId);
      if (!r.interior) throw new StyleError('fix widom: the region must be defined with side in');
      if (r.dynamic) throw new StyleError('fix widom: a dynamic region (move or rotate) is not supported');
      this.region = r;
      this.reg = regionBox(this.sys, r);
    } else {
      this.reg = null;
    }
    this.volume = this.reg ? this.reg.volume : (s.box.hi[0] - s.box.lo[0]) * (s.box.hi[1] - s.box.lo[1]) * (s.box.hi[2] - s.box.lo[2]);
  }

  /** One event at the first step and every N steps after (measured, see the header). */
  postForce(): void {
    if (this.sys.state.step !== this.nextStep) return;
    this.event();
    this.nextStep += this.N;
  }

  private event(): void {
    const sys = this.sys;
    const s = sys.state;
    const kT = s.units.boltz * this.T;
    const eBefore = this.potentialEnergy();
    let sum = 0;
    let ninsert = 0;
    for (let k = 0; k < this.M; k++) {
      const p = this.drawPosition();
      if (!p) continue;
      sum += Math.exp(-this.trial(p, eBefore) / kT);
      ninsert++;
    }
    const avg = ninsert > 0 ? sum / ninsert : 0;
    this.avgBoltz = avg;
    this.mu = avg > 0 ? -kT * Math.log(avg) : 0;
  }

  /** Random point: the box, or the region bounding box with rejection for non-rectangular regions. */
  private drawPosition(): Vec3 | null {
    const s = this.sys.state;
    const lo = this.reg ? this.reg.lo : (s.box.lo as unknown as Vec3);
    const hi = this.reg ? this.reg.hi : (s.box.hi as unknown as Vec3);
    if (!this.region || this.region instanceof BlockRegion) {
      return [lo[0] + this.rng.uniform() * (hi[0] - lo[0]), lo[1] + this.rng.uniform() * (hi[1] - lo[1]), lo[2] + this.rng.uniform() * (hi[2] - lo[2])];
    }
    // "random trial points are generated within the rectangular bounding box until a point is found that
    // lies inside the region. If no valid point is generated after 1000 trials, no insertion is performed."
    for (let t = 0; t < 1000; t++) {
      const p: Vec3 = [lo[0] + this.rng.uniform() * (hi[0] - lo[0]), lo[1] + this.rng.uniform() * (hi[1] - lo[1]), lo[2] + this.rng.uniform() * (hi[2] - lo[2])];
      if (this.region.match(p[0], p[1], p[2])) return p;
    }
    return null;
  }

  /** One insertion: append the probe, take the energy difference, remove it. */
  private trial(p: Vec3, eBefore: number): number {
    const sys = this.sys;
    const s = sys.state;
    const n0 = s.n;
    // the temporary insertion must be invisible to the rest of the step: keep the positions, image
    // flags, forces, torques and storage order and put them back afterwards
    const xSave = s.x.slice();
    const imageSave = s.image.slice();
    const fSave = s.f.slice();
    const orderSave = s.order.slice();
    const torqueSave = s.torque ? s.torque.slice() : null;
    let dU: number;
    if (this.molTemplate) {
      const t = this.molTemplate;
      const rot = this.drawRotation();
      const center = geometricCenter(t);
      const coords = new Float64Array(3 * t.natoms);
      for (let a = 0; a < t.natoms; a++) {
        const rx = t.x[3 * a] - center[0], ry = t.x[3 * a + 1] - center[1], rz = t.x[3 * a + 2] - center[2];
        coords[3 * a] = p[0] + rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz;
        coords[3 * a + 1] = p[1] + rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz;
        coords[3 * a + 2] = p[2] + rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz;
      }
      appendMolecule(sys, t, 0, coords, [0, 0, 0], maxMoleculeId(s) + 1, this.groupBit);
      sys.atomsChanged();
      dU = this.potentialEnergy() - eBefore;
      if (!this.fullEnergy) dU -= this.intraMolEnergy;
    } else {
      const xa = Float64Array.from(p);
      appendAtoms(s, { x: xa, type: this.type, q: this.charge ?? undefined, mask: this.groupBit });
      sys.atomsChanged();
      dU = this.potentialEnergy() - eBefore;
    }
    dU -= this.intraEnergy;
    const del = new Uint8Array(s.n);
    for (let k = n0; k < s.n; k++) del[k] = 1;
    deleteAtoms(s, del);
    sys.atomsChanged();
    s.x.set(xSave); s.image.set(imageSave); s.f.set(fSave); s.order.set(orderSave);
    if (torqueSave && s.torque) s.torque.set(torqueSave);
    return dU;
  }

  /** Random rotation axis (rejection in the unit sphere) and angle 2 pi u (measured, see the header). */
  private drawRotation(): number[][] {
    let ax: number, ay: number, az: number;
    do {
      ax = 2 * this.rng.uniform() - 1;
      ay = 2 * this.rng.uniform() - 1;
      az = 2 * this.rng.uniform() - 1;
    } while (ax * ax + ay * ay + az * az >= 1 || (ax === 0 && ay === 0 && az === 0));
    const theta = 2 * Math.PI * this.rng.uniform();
    return rotationMatrix(theta, ax, ay, az);
  }

  private potentialEnergy(): number {
    const a: Accum = this.sys.forces();
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp;
  }

  /**
   * The inserted molecule's intramolecular energy (bonds, angles, dihedrals,
   * impropers), evaluated once on the rigid template coordinates; it is the
   * same for every insertion because "Inserted molecules can have different
   * orientations, but they will all have the same intramolecular configuration"
   * (fix_widom.rst).
   */
  private moleculeIntraEnergy(t: MoleculeTemplate): number {
    const n = t.natoms;
    const topo = {
      bonds: topoList(t.bonds, 2), angles: topoList(t.angles, 3),
      dihedrals: topoList(t.dihedrals, 4), impropers: topoList(t.impropers, 4),
      nbondtypes: 0, nangletypes: 0, ndihedraltypes: 0, nimpropertypes: 0,
    };
    const fake = { n, x: Float64Array.from(t.x), topo } as unknown as SimState;
    const map = new Int32Array(n + 1).fill(-1);
    for (let i = 0; i < n; i++) map[i + 1] = i;
    const acc = newAccum();
    const f = new Float64Array(3 * n);
    const bc: BondedCompute = { s: fake, geom: this.sys.geom, map, f, acc, virial: acc.vbond, eatom: null, vatom: null, warn: () => {} };
    const run = (style: Bonded | null, virial: Float64Array) => { if (style) { bc.virial = virial; style.compute(bc); } };
    run(this.sys.ff.bond, acc.vbond);
    run(this.sys.ff.angle, acc.vangle);
    run(this.sys.ff.dihedral, acc.vdihed);
    run(this.sys.ff.improper, acc.vimp);
    return acc.ebond + acc.eangle + acc.edihed + acc.eimp;
  }

  computeVector(i: number): number {
    if (i === 0) return this.mu;
    if (i === 1) return this.avgBoltz;
    if (i === 2) return this.volume;
    throw new StyleError(`fix ${this.id} does not compute vector element ${i + 1}`);
  }
}
