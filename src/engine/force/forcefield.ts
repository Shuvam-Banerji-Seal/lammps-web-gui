import type { SimState, TopoList } from '../types';
import type { Geometry } from '../domain';
import type { Neighbor, SpecialSettings } from '../neighbor';
import { buildAtomMap, buildSpecial, filterTopo, type SpecialList } from '../atoms';
import { BOND_REMOVED, clearAccum, newAccum, StyleError, type Accum, type Bonded, type KSpace, type Pair, type PairCompute, type StyleContext } from './types';
import { PairHybrid } from './pair/hybrid';
import { SMD_DESPH } from '../smd';

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
  /** special_bonds one/five yes: the 1-5 neighbour list is kept (atom_style amoeba only). */
  one5?: boolean;
}

/**
 * delete_bonds without remove (delete_bonds.html) switches an interaction off by negating its type: the bonded
 * styles skip such entries, while the special lists still see them (they are topology, not interactions).
 */
const switchedOff = (t: SimState['topo']): boolean => {
  for (const l of [t.bonds, t.angles, t.dihedrals, t.impropers]) for (let e = 0; e < l.n; e++) if (l.type[e] < 0) return true;
  return false;
};

const withoutSwitchedOff = (t: SimState['topo']): SimState['topo'] => {
  const copy = (l: TopoList): TopoList => {
    const out: TopoList = { ...l, type: Int32Array.from(l.type.subarray(0, l.n)), atoms: Int32Array.from(l.atoms.subarray(0, l.n * l.width)) };
    filterTopo(out, (_ids, type) => type >= 0);
    return out;
  };
  return { ...t, bonds: copy(t.bonds), angles: copy(t.angles), dihedrals: copy(t.dihedrals), impropers: copy(t.impropers) };
};

export const defaultSpecial = (): SpecialBonds => ({ lj: [0, 0, 0], coul: [0, 0, 0], angle: false, dihedral: false });

export interface ComputeFlags {
  /** Per-atom energy / virial wanted on this step (compute pe/atom, stress/atom). */
  eatom?: boolean;
  vatom?: boolean;
  /** Per-atom centroid virial of the angle, dihedral and improper terms wanted (compute centroid/stress/atom). */
  cvatom?: boolean;
  /** A force evaluation inside a timestep: history-dependent pair styles advance their history (not at setup or between runs). */
  step?: boolean;
}

/** The force-field terms compute pe/atom and stress/atom can select. */
export type PerAtomTerm = 'pair' | 'bond' | 'angle' | 'dihedral' | 'improper' | 'kspace';

/**
 * Which force-field terms one evaluation covers, for run_style respa (run/respa.ts): each rRESPA
 * level is evaluated with only the terms assigned to it. has('pair') is true when the pair term
 * (or any hybrid sub-style of it) belongs to the level; subPair(k) selects hybrid sub-style k.
 * Without a selector compute() evaluates every term (the verlet path).
 */
export interface LevelSelect {
  has(term: PerAtomTerm): boolean;
  subPair?: (k: number) => boolean;
}

export class ForceField {
  pair: Pair | null = null;
  /**
   * Name of the pair style a restart file named but could not store (restart.ts PAIR_NOT_IN_RESTART);
   * cleared by the next pair_style command. Measured with native LAMMPS (black box): a run after
   * read_restart without a new pair_style stops with the error Must re-specify non-restarted pair
   * style (sw) after read_restart.
   */
  pairNotRestarted: string | null = null;
  /** Shared-memory threads for the pair term (cpu/pairThreads.ts), set by the session when available. */
  pairThreads: { run(pair: Pair, pc: PairCompute): boolean; readonly calls: number } | null = null;
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
  /**
   * The same per term (pair, bond, angle, dihedral, improper, kspace), for compute pe/atom and
   * stress/atom keyword subsets; a term without a style has no entry. eatom/vatom are their sums.
   */
  eatomTerm: Partial<Record<PerAtomTerm, Float64Array>> = {};
  vatomTerm: Partial<Record<PerAtomTerm, Float64Array>> = {};
  /** Per-atom centroid virial (9 per atom) of the angle, dihedral and improper terms, when ComputeFlags.cvatom. */
  cvatomTerm: Partial<Record<PerAtomTerm, Float64Array>> = {};
  /** Tail corrections: energy = etailV / V, pressure term = ptailV / V^2 (in energy/volume). */
  etailV = 0;
  ptailV = 0;
  private specialList: SpecialList | null = null;
  /**
   * The 1-2/1-3/1-4 lists are stale: a delete_bonds without its special keyword removed bonds and kept the old lists
   * (native: "special" rebuilds them only when asked, src/delete_bonds.cpp at b332f28). They are kept until a normal
   * topology change rebuilds them.
   */
  private specialStale = false;
  /**
   * Rewrites the special lists after they are built from the bonds (a fix may set it and must clear it on unfix).
   * Used by fix drude, whose Drude particles take their core's special neighbors (fix_drude.html).
   */
  specialHook: ((list: SpecialList, s: SimState) => SpecialList) | null = null;
  private map: Int32Array = new Int32Array(0);
  private specialLJ = new Float64Array([1, 0, 0, 0]);
  private specialCoul = new Float64Array([1, 0, 0, 0]);

  /**
   * Topology without a style is allowed (those terms add nothing). Measured
   * with native LAMMPS (black box): each init warns "Bonds are defined but no
   * bond style is set" (likewise Angles, Dihedrals, Impropers), followed for
   * bonds, angles and dihedrals by the warning "Likewise 1-2 special neighbor
   * interactions != 1.0" (the 1-3 and 1-4 forms of it are the same text with
   * those numbers) when that special_bonds lj or coul weight is not 1.
   */
  private checkTopology(s: SimState): void {
    const t = s.topo, sp = this.special;
    const need = (n: number, style: Bonded | null, title: string, kind: string, order: number) => {
      if (n === 0 || style) return;
      this.warn(`${title} are defined but no ${kind} style is set`);
      if (order > 0 && (sp.lj[order - 1] !== 1 || sp.coul[order - 1] !== 1)) {
        this.warn(`Likewise 1-${order + 1} special neighbor interactions != 1.0`);
      }
    };
    need(t.bonds.n, this.bond, 'Bonds', 'bond', 1);
    need(t.angles.n, this.angle, 'Angles', 'angle', 2);
    need(t.dihedrals.n, this.dihedral, 'Dihedrals', 'dihedral', 3);
    need(t.impropers.n, this.improper, 'Impropers', 'improper', 0);
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
    if (!this.pair && this.pairNotRestarted) throw new StyleError(`Must re-specify non-restarted pair style (${this.pairNotRestarted}) after read_restart`);
    this.warn = (t) => ctx.log(`WARNING: ${t}`);
    this.checkTopology(s);
    this.pair?.linkKSpace?.(this.kspace);
    // A solver that sets the pair style's Coulomb cutoff (MSM cutoff/adjust) does it before the pair style's init, as
    // native LAMMPS's Force::init does (src/force.cpp at b332f28: "kspace->init(); // kspace must come before pair").
    if (this.kspace?.preparePair && this.pair) {
      if (this.pair.ntypes !== s.ntypes) this.pair.allocate(s.ntypes);
      this.kspace.preparePair(this.pair, s, geom, s.units.qqr2e * this.pair.coulConstScale / this.dielectric, ctx);
    }
    if (this.pair) {
      if (this.pair.ntypes !== s.ntypes) this.pair.allocate(s.ntypes);
      this.pair.linkBonded?.({ bond: (t) => this.bond?.equilibrium(t) ?? Number.NaN, angle: (t) => this.angle?.equilibrium(t) ?? Number.NaN });
      this.pair.init(ctx);
    }
    for (const b of [this.bond, this.angle, this.dihedral, this.improper]) b?.linkForceField?.({ pair: this.pair, special: this.special });
    for (const b of [this.bond, this.angle, this.dihedral, this.improper]) b?.init(ctx);
    this.specialLJ = new Float64Array([1, ...this.special.lj]);
    this.specialCoul = new Float64Array([1, ...this.special.coul]);
    const manybody = this.pair?.manybody ?? false;
    const hasTopo = s.topo.bonds.n > 0;
    // stale lists stay even when no bond is left (native keeps its special arrays until a rebuild)
    if (!this.specialStale) this.specialList = hasTopo && !manybody ? this.buildSpecialList(s) : null;
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
      size: (this.pair?.sizeList ?? false) && s.radius !== null,
    });
    if (this.pair?.coulLong && !this.kspace) {
      throw new StyleError(`pair style ${this.pair.name} needs a kspace style (kspace_style ewald, pppm or msm)`);
    }
    if (this.kspace) {
      if (!this.pair) throw new StyleError('kspace_style needs a pair style with a long-range Coulomb part');
      const cutCoul = this.pair.extract('cut_coul');
      if (typeof cutCoul !== 'number') throw new StyleError(`kspace_style ${this.kspace.name} is not compatible with pair style ${this.pair.name}`);
      // pair_zero.html: "pair style *zero/coul* cannot be used with kspace styles that require a pair style
      // providing dispersion or TIP4P support (e.g. *pppm/disp* or *pppm/tip4p*)"; the refusal text is native's (measured)
      if (this.pair.name === 'zero/coul' && /disp|tip4p/.test(this.kspace.name)) {
        throw new StyleError(`KSpace style ${this.kspace.name} is incompatible with Pair style zero/coul`);
      }
      this.kspace.linkPair?.(this.pair);
      // As native LAMMPS does (src/kspace.cpp KSpace::pair_check at b332f28): the dispersion flags of the pair and the
      // kspace style must agree (pppm/disp needs a pair with dispersion; ewald, pppm and msm refuse one)
      if (this.kspace.dispersion() !== this.pair.dispersionFlag) {
        throw new StyleError(`KSpace style ${this.kspace.name} is incompatible with Pair style ${this.pair.name}`);
      }
      // paramsFrozen: a charge-changing fix (fix charge/regulation) keeps the init-time G-ewald, kmax and mesh
      if (this.kspace.paramsFrozen && this.kspace.gEwald > 0) this.kspace.updateChargeSums(s);
      else this.kspace.init(s, geom, cutCoul, s.units.qqr2e * this.pair.coulConstScale / this.dielectric, ctx);
      this.pair.gEwald = this.kspace.gEwald;
    }
    if (this.pair) this.pair.version++;
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

  /**
   * The topology changed or atoms were added/removed: rebuilds the special list and hands it to the neighbor list, so
   * the next build excludes the pairs of the new topology (the force field and the neighbor list keep one list).
   */
  topologyChanged(s: SimState, nb: Neighbor, keepSpecial = false): void {
    // keepSpecial: the bonds changed without the special keyword, so the old lists stay (see specialStale)
    if (keepSpecial) {
      this.specialStale = true;
      return;
    }
    this.specialStale = false;
    const manybody = this.pair?.manybody ?? false;
    this.specialList = s.topo.bonds.n > 0 && !manybody ? this.buildSpecialList(s) : null;
    nb.setSpecial(this.specialList);
  }

  /**
   * Special lists from the bond topology, passed through specialHook when a fix has set one (fix drude
   * changes them for Drude particles, docs.lammps.org/fix_drude.html).
   */
  private buildSpecialList(s: SimState): SpecialList {
    const bond = this.bond;
    const ignoreBond = bond?.ignoresSpecial ? (t: number) => bond.ignoresSpecial!(t) : undefined;
    // As native LAMMPS does (src/special.cpp Special::build at b332f28): the 1-3 and 1-4 lists are not built when
    // their lj and coul weights are all 1.0 (the 1-4 list alone when only its weights are 1.0), unless one/five needs
    // them. A weight of 1.0 leaves a pair as if it had no special status, and skipping the lists keeps densely bonded
    // systems fast (bpm with special_bonds lj 0 1 1 and ~120 bonds per atom reached nearly every atom at depth 3).
    const { lj, coul } = this.special;
    const one = (k: 1 | 2) => lj[k] === 1 && coul[k] === 1;
    const maxOrder: 1 | 2 | 3 = this.special.one5 ? 3 : one(1) && one(2) ? 1 : one(2) ? 2 : 3;
    const list = buildSpecial(s, { angle: this.special.angle, dihedral: this.special.dihedral, ignoreBond, maxOrder });
    return this.specialHook ? this.specialHook(list, s) : list;
  }

  /** Forces on every owned atom into s.f; energies and virial into this.acc. */
  compute(s: SimState, nb: Neighbor, geom: Geometry, flags: ComputeFlags = {}, sel?: LevelSelect): Accum {
    const acc = this.acc;
    clearAccum(acc);
    s.f.fill(0, 0, 3 * s.n);
    s.torque?.fill(0, 0, 3 * s.n);
    // atom_vec_smd force_clear (native AtomVecSMD::force_clear): the energy rate desph of the SMD styles starts at zero
    // for every evaluation, before the sub-styles of a hybrid add to it (the styles do not clear it themselves)
    s.custom.get(SMD_DESPH)?.data.fill(0);
    nb.clearForces();
    const nall = nb.nall;
    const eatomAll = flags.eatom ? new Float64Array(nall) : null;
    const vatomAll = flags.vatom ? new Float64Array(6 * nall) : null;
    const qqrd2e = s.units.qqr2e * (this.pair?.coulConstScale ?? 1) / this.dielectric;
    const want = (t: PerAtomTerm): boolean => !sel || sel.has(t);
    // pair_modify compute no (pair_modify.html): the pair term is skipped; kspace and bonded terms still run
    if (this.pair && this.pair.computeOn && want('pair')) {
      const pc: PairCompute = {
        s, nb, geom, x: nb.xall, f: nb.fall, type: nb.typeall, q: nb.qall,
        nlocal: nb.nlocal, nall, half: nb.half, full: nb.full,
        specialLJ: this.specialLJ, specialCoul: this.specialCoul, qqrd2e, acc, historyUpdate: !!flags.step,
        eatom: eatomAll, vatom: vatomAll,
      };
      if (sel) {
        // a level evaluation runs the pair term directly (the threaded path has no sub-style selection)
        if (this.pair instanceof PairHybrid) this.pair.subSelect = sel.subPair ?? null;
        try { this.pair.compute(pc); } finally { if (this.pair instanceof PairHybrid) this.pair.subSelect = null; }
      } else if (!this.pairThreads?.run(this.pair, pc)) this.pair.compute(pc);
      if (this.pair.virialFdotr && !pc.fdotrDone) {
        const xa = nb.xall, fa = nb.fall;
        let v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
        for (let k = 0; k < 3 * nall; k += 3) {
          const x = xa[k], y = xa[k + 1], z = xa[k + 2];
          const fx = fa[k], fy = fa[k + 1], fz = fa[k + 2];
          // xy = sum x f_y etc.; matters only for non-central forces with torques (granular contacts)
          v0 += x * fx; v1 += y * fy; v2 += z * fz; v3 += x * fy; v4 += x * fz; v5 += y * fz;
        }
        const v = acc.virial;
        v[0] += v0; v[1] += v1; v[2] += v2; v[3] += v3; v[4] += v4; v[5] += v5;
      }
    }
    // per-atom energy and virial are tallied per term, then summed
    const eTerm: Partial<Record<PerAtomTerm, Float64Array>> = {};
    const vTerm: Partial<Record<PerAtomTerm, Float64Array>> = {};
    const cvTerm: Partial<Record<PerAtomTerm, Float64Array>> = {};
    const termArrays = (t: PerAtomTerm): { eatom: Float64Array | null; vatom: Float64Array | null } => {
      const e = flags.eatom ? (eTerm[t] = new Float64Array(s.n)) : null;
      const v = flags.vatom ? (vTerm[t] = new Float64Array(6 * s.n)) : null;
      return { eatom: e, vatom: v };
    };
    const centroidArray = (t: PerAtomTerm): Float64Array | null =>
      flags.cvatom && (t === 'angle' || t === 'dihedral' || t === 'improper') ? (cvTerm[t] = new Float64Array(9 * s.n)) : null;
    // the pairwise parts of bonded terms that native tallies through the pair style (BondedCompute.pairEatom)
    const pair14E = flags.eatom ? new Float64Array(s.n) : null;
    const pair14V = flags.vatom ? new Float64Array(6 * s.n) : null;
    if (this.bond || this.angle || this.dihedral || this.improper) {
      if (this.map.length === 0 || this.mapStale(s)) this.map = buildAtomMap(s);
      const ov = this.topoOverride;
      const sb0 = ov && ov.bondsN === s.topo.bonds.n && ov.anglesN === s.topo.angles.n
        ? { ...s, topo: { ...s.topo, bonds: ov.bonds, angles: ov.angles } }
        : s;
      const sb = switchedOff(sb0.topo) ? { ...sb0, topo: withoutSwitchedOff(sb0.topo) } : sb0;
      const bc = { s: sb, geom, map: this.map, f: s.f, acc, eatom: null as Float64Array | null, vatom: null as Float64Array | null, cvatom: null as Float64Array | null, pairEatom: pair14E, pairVatom: pair14V, virial: acc.vbond, warn: this.warn, historyUpdate: !!flags.step, nbuild: nb.nbuild, copies: nb.bondedCopies() };
      const run = (style: Bonded | null, t: PerAtomTerm, virial: Float64Array) => {
        if (!style || !want(t)) return;
        Object.assign(bc, termArrays(t), { cvatom: centroidArray(t) });
        bc.virial = virial;
        style.compute(bc);
      };
      run(this.bond, 'bond', acc.vbond);
      // bpm bonds that broke this step (type set to 0): their 1-2 special entries go, so the next neighbor build uses the new lists
      if (this.bond?.takeTopologyChange?.()) {
        // bonds a style deleted (BOND_REMOVED) leave the topology (native removes them from the atoms' bond lists)
        filterTopo(s.topo.bonds, (_ids, type) => type !== BOND_REMOVED);
        this.topologyChanged(s, nb);
      }
      run(this.angle, 'angle', acc.vangle);
      run(this.dihedral, 'dihedral', acc.vdihed);
      run(this.improper, 'improper', acc.vimp);
    }
    if (this.kspace && this.kspace.computeOn && want('kspace')) this.kspace.compute({ s, geom, f: s.f, qqrd2e, acc, torque: s.torque, ...termArrays('kspace') });
    nb.reverseComm(s.f);
    if (this.pair && (eatomAll || vatomAll)) {
      const pt = termArrays('pair');
      if (eatomAll && pt.eatom) nb.reverseSum(eatomAll, 1, pt.eatom);
      if (vatomAll && pt.vatom) nb.reverseSum(vatomAll, 6, pt.vatom);
    }
    if (pair14E && pair14E.some((x) => x !== 0)) {
      const e = eTerm.pair ?? (eTerm.pair = new Float64Array(s.n));
      for (let i = 0; i < s.n; i++) e[i] += pair14E[i];
    }
    if (pair14V && pair14V.some((x) => x !== 0)) {
      const v = vTerm.pair ?? (vTerm.pair = new Float64Array(6 * s.n));
      for (let k = 0; k < 6 * s.n; k++) v[k] += pair14V[k];
    }
    // the tail correction is added once per run_style respa step by the integrator, not per level
    if (this.etailV !== 0 && !sel) acc.evdwl += this.etailV / geom.volume(s.dimension);
    const sum = (terms: Partial<Record<PerAtomTerm, Float64Array>>, len: number): Float64Array => {
      const out = new Float64Array(len);
      for (const arr of Object.values(terms)) for (let k = 0; k < len; k++) out[k] += arr![k];
      return out;
    };
    this.eatomTerm = eTerm;
    this.vatomTerm = vTerm;
    this.cvatomTerm = cvTerm;
    this.eatom = flags.eatom ? sum(eTerm, s.n) : null;
    this.vatom = flags.vatom ? sum(vTerm, 6 * s.n) : null;
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
