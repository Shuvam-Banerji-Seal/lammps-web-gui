import type { SimState, TopoList } from '../types';
import type { Geometry } from '../domain';
import type { Neighbor, SpecialSettings } from '../neighbor';
import { buildAtomMap, buildSpecial, type SpecialList } from '../atoms';
import { clearAccum, newAccum, StyleError, type Accum, type Bonded, type KSpace, type Pair, type StyleContext } from './types';
import { PairHybrid } from './pair/hybrid';

/*
 * The force field: one pair style (possibly hybrid), bond/angle/dihedral/
 * improper styles, an optional kspace style, special_bonds weights and the
 * dielectric constant. compute() runs the documented sequence
 * (docs.lammps.org/Developer_flow.html):
 *   force_clear(); pair->compute(); bond->compute(); angle->compute();
 *   dihedral->compute(); improper->compute(); kspace->compute();
 *   comm->reverse_comm()
 * special_bonds defaults (docs.lammps.org/special_bonds.html): "All 3
 * Lennard-Jones and 3 Coulombic weighting coefficients = 0.0, angle = no,
 * dihedral = no." dielectric: "The value is used in the denominator of the
 * formulas for Coulombic interactions" (docs.lammps.org/dielectric.html).
 */

export interface SpecialBonds {
  lj: [number, number, number];
  coul: [number, number, number];
  angle: boolean;
  dihedral: boolean;
}

export const defaultSpecial = (): SpecialBonds => ({ lj: [0, 0, 0], coul: [0, 0, 0], angle: false, dihedral: false });

export interface ComputeFlags {
  /** Per-atom energy / virial wanted on this step (compute pe/atom, stress/atom). */
  eatom?: boolean;
  vatom?: boolean;
}

export class ForceField {
  pair: Pair | null = null;
  bond: Bonded | null = null;
  angle: Bonded | null = null;
  dihedral: Bonded | null = null;
  improper: Bonded | null = null;
  kspace: KSpace | null = null;
  special: SpecialBonds = defaultSpecial();
  dielectric = 1;

  readonly acc: Accum = newAccum();
  /** Per owned atom, after compute() with the flags set. */
  eatom: Float64Array | null = null;
  vatom: Float64Array | null = null;
  /** Tail corrections: energy = etailV / V, pressure term = ptailV / V^2 (in energy/volume). */
  etailV = 0;
  ptailV = 0;
  private specialList: SpecialList | null = null;
  private map: Int32Array = new Int32Array(0);
  private specialLJ = new Float64Array([1, 0, 0, 0]);
  private specialCoul = new Float64Array([1, 0, 0, 0]);

  /** Everything molecular (bonds/angles/...) has a style or there are none. */
  private checkTopology(s: SimState): void {
    const t = s.topo;
    const need = (n: number, style: Bonded | null, kind: string) => {
      if (n > 0 && !style) throw new StyleError(`${n} ${kind}s are defined but no ${kind}_style is set`);
    };
    need(t.bonds.n, this.bond, 'bond');
    need(t.angles.n, this.angle, 'angle');
    need(t.dihedrals.n, this.dihedral, 'dihedral');
    need(t.impropers.n, this.improper, 'improper');
  }

  /**
   * Setup before a run / force evaluation: checks coefficients, mixes, sets
   * neighbor requirements and the special list.
   */
  /**
   * Bond and angle lists without entries that a constraint fix (fix shake)
   * switched off; used while the live lists still have the counts it saw.
   */
  private topoOverride: { bonds: TopoList; angles: TopoList; bondsN: number; anglesN: number } | null = null;

  setTopologyOverride(o: { bonds: TopoList; angles: TopoList; bondsN: number; anglesN: number } | null): void {
    this.topoOverride = o;
  }

  /** Run-log warnings for bonded styles, set at init from the style context. */
  private warn: (text: string) => void = () => {};

  init(s: SimState, nb: Neighbor, geom: Geometry, ctx: StyleContext): void {
    this.warn = (t) => ctx.log(`WARNING: ${t}`);
    this.checkTopology(s);
    if (this.pair) {
      if (this.pair.ntypes !== s.ntypes) this.pair.allocate(s.ntypes);
      this.pair.init(ctx);
    }
    for (const b of [this.bond, this.angle, this.dihedral, this.improper]) b?.init(ctx);
    this.specialLJ = new Float64Array([1, ...this.special.lj]);
    this.specialCoul = new Float64Array([1, ...this.special.coul]);
    const manybody = this.pair?.manybody ?? false;
    const hasTopo = s.topo.bonds.n > 0;
    this.specialList = hasTopo && !manybody ? buildSpecial(s, { angle: this.special.angle, dihedral: this.special.dihedral }) : null;
    const nt = s.ntypes + 1;
    const cutoff = this.pair ? this.pair.cut : new Float64Array(nt * nt);
    const settings: SpecialSettings = {
      lj: [1, ...this.special.lj] as SpecialSettings['lj'],
      coul: [1, ...this.special.coul] as SpecialSettings['coul'],
      keepExcluded: !!this.kspace || (this.pair?.coulLong ?? false) || (this.pair?.keepExcluded ?? false),
    };
    if (this.pair instanceof PairHybrid) this.pair.checkSpecial(this.special);
    nb.init({
      half: this.pair ? this.pair.needsHalf : false,
      full: this.pair ? this.pair.needsFull : false,
      cutoff, ntypes: s.ntypes, special: this.specialList, specialSettings: settings,
    });
    if (this.pair?.coulLong && !this.kspace) {
      throw new StyleError(`pair style ${this.pair.name} needs a kspace style (kspace_style ewald or pppm)`);
    }
    if (this.kspace) {
      if (!this.pair) throw new StyleError('kspace_style needs a pair style with a long-range Coulomb part');
      const cutCoul = this.pair.extract('cut_coul');
      if (typeof cutCoul !== 'number') throw new StyleError(`kspace_style ${this.kspace.name} is not compatible with pair style ${this.pair.name}`);
      this.kspace.init(s, geom, cutCoul, s.units.qqr2e / this.dielectric, ctx);
      this.pair.gEwald = this.kspace.gEwald;
    }
    this.updateTail(s);
  }

  /** Tail corrections depend on the number of atoms of each type. */
  updateTail(s: SimState): void {
    this.etailV = 0;
    this.ptailV = 0;
    if (!this.pair || !this.pair.tail) return;
    const count = new Float64Array(s.ntypes + 1);
    for (let i = 0; i < s.n; i++) count[s.type[i]]++;
    const t = this.pair.tailSums(count);
    this.etailV = t.etail;
    this.ptailV = t.ptail;
  }

  /** The topology changed or atoms were added/removed: rebuild special lists before the next build. */
  topologyChanged(s: SimState): void {
    const manybody = this.pair?.manybody ?? false;
    this.specialList = s.topo.bonds.n > 0 && !manybody
      ? buildSpecial(s, { angle: this.special.angle, dihedral: this.special.dihedral }) : null;
  }

  /** Forces on every owned atom into s.f; energies and virial into this.acc. */
  compute(s: SimState, nb: Neighbor, geom: Geometry, flags: ComputeFlags = {}): Accum {
    const acc = this.acc;
    clearAccum(acc);
    s.f.fill(0, 0, 3 * s.n);
    nb.clearForces();
    const nall = nb.nall;
    const eatomAll = flags.eatom ? new Float64Array(nall) : null;
    const vatomAll = flags.vatom ? new Float64Array(6 * nall) : null;
    const qqrd2e = s.units.qqr2e / this.dielectric;
    if (this.pair) {
      this.pair.compute({
        s, nb, geom, x: nb.xall, f: nb.fall, type: nb.typeall, q: nb.qall,
        nlocal: nb.nlocal, nall, half: nb.half, full: nb.full,
        specialLJ: this.specialLJ, specialCoul: this.specialCoul, qqrd2e, acc,
        eatom: eatomAll, vatom: vatomAll,
      });
      if (this.pair.virialFdotr) {
        const xa = nb.xall, fa = nb.fall;
        let v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
        for (let k = 0; k < 3 * nall; k += 3) {
          const x = xa[k], y = xa[k + 1], z = xa[k + 2];
          const fx = fa[k], fy = fa[k + 1], fz = fa[k + 2];
          v0 += x * fx; v1 += y * fy; v2 += z * fz; v3 += y * fx; v4 += z * fx; v5 += z * fy;
        }
        const v = acc.virial;
        v[0] += v0; v[1] += v1; v[2] += v2; v[3] += v3; v[4] += v4; v[5] += v5;
      }
    }
    const eatom = flags.eatom ? new Float64Array(s.n) : null;
    const vatom = flags.vatom ? new Float64Array(6 * s.n) : null;
    if (this.bond || this.angle || this.dihedral || this.improper) {
      if (this.map.length === 0 || this.mapStale(s)) this.map = buildAtomMap(s);
      const ov = this.topoOverride;
      const sb = ov && ov.bondsN === s.topo.bonds.n && ov.anglesN === s.topo.angles.n
        ? { ...s, topo: { ...s.topo, bonds: ov.bonds, angles: ov.angles } }
        : s;
      const bc = { s: sb, geom, map: this.map, f: s.f, acc, eatom, vatom, virial: acc.vbond, warn: this.warn };
      this.bond?.compute(bc);
      bc.virial = acc.vangle;
      this.angle?.compute(bc);
      bc.virial = acc.vdihed;
      this.dihedral?.compute(bc);
      bc.virial = acc.vimp;
      this.improper?.compute(bc);
    }
    if (this.kspace) this.kspace.compute({ s, geom, f: s.f, qqrd2e, acc, eatom, vatom });
    nb.reverseComm(s.f);
    if (eatomAll && eatom) nb.reverseSum(eatomAll, 1, eatom);
    if (vatomAll && vatom) nb.reverseSum(vatomAll, 6, vatom);
    if (this.etailV !== 0) acc.evdwl += this.etailV / geom.volume(s.dimension);
    this.eatom = eatom;
    this.vatom = vatom;
    return acc;
  }

  /** Atoms were added or deleted since the map was built. */
  private mapStale(s: SimState): boolean {
    for (let i = 0; i < s.n; i += Math.max(1, Math.floor(s.n / 16))) {
      if (s.id[i] >= this.map.length || this.map[s.id[i]] !== i) return true;
    }
    return s.n > 0 && (this.map.length <= s.id[s.n - 1] || this.map[s.id[s.n - 1]] !== s.n - 1);
  }

  /** Forces the id map to be rebuilt (atoms added, deleted or reordered). */
  atomsChanged(): void {
    this.map = new Int32Array(0);
  }

  /** Pressure contribution of the tail correction, energy/volume. */
  ptail(volume: number): number {
    return this.ptailV / (volume * volume);
  }
}
