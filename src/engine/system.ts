import type { AtomStyle, EngineEvent, SimState, UnitSystem } from './types';
import type { MoleculeTemplate } from './molecule';
import { UNIT_SYSTEMS } from './units';
import { Geometry, shrinkWrap } from './domain';
import { Neighbor } from './neighbor';
import { ForceField, type ComputeFlags } from './force/forcefield';
import { StyleError, type Accum, type StyleContext } from './force/types';
import { Groups } from './group';
import type { Region, RegionEnv } from './region';
import type { Compute } from './compute/compute';
import type { Fix } from './fix/fix';
import { Variables } from './variables';
import { type FormulaEnv, type Mode, type Value } from './formula';
import type { Lattice } from './lattice';
import { Rng } from './rng';
import { Thermo } from './output/thermo';
import type { Dump } from './output/dump';
import { buildAtomMap, deleteAtoms as deleteAtomsImpl, massOf, customAttr } from './atoms';
import { groupFunction } from './groupfn';
import { defaultMinSettings, type MinSettings } from './run/min';

/*
 * The whole simulation: settings, atoms, box, neighbor lists, force field,
 * groups, regions, computes, fixes, variables, thermo and file output. Commands
 * (commands/*.ts) operate on it; the run loop (run/verlet.ts) advances it.
 *
 * Two counters make derived values safe to cache:
 *   stateVersion  bumps when positions, atoms, box or the force field change;
 *                 forces() recomputes when it differs from forcesVersion.
 *   epoch         bumps with stateVersion and whenever output wants fresh
 *                 compute values (thermo re-invokes its computes every time it
 *                 prints, as LAMMPS's thermo output does).
 */

export interface SessionIO {
  emit(ev: EngineEvent): void;
  /** A command wrote (or appended to) a file: dump, write_data, print file/append. */
  writeFile?(name: string, text: string, append: boolean): void;
}

export interface RunInfo {
  inRun: boolean;
  /** First and last step of the current (or last) run. */
  firstStep: number;
  lastStep: number;
  /** run start/stop keywords, else equal to first/last. */
  beginStep: number;
  endStep: number;
  /** Wall clock at the start of the current run (ms). */
  t0: number;
  /** Any run has been performed (for vdisplace etc. between runs). */
  ranOnce: boolean;
}

export class System {
  // ---- settings that precede the box
  units: UnitSystem = UNIT_SYSTEMS.lj;
  dimension: 2 | 3 = 3;
  boundary: SimState['box']['boundary'] = [['p', 'p'], ['p', 'p'], ['p', 'p']];
  atomStyle: AtomStyle = 'atomic';
  lattice: Lattice | null = null;
  /** timestep given before the box exists. */
  pendingDt: number | null = null;

  private _state: SimState | null = null;
  geom!: Geometry;
  readonly nb: Neighbor;
  readonly ff = new ForceField();
  readonly groups = new Groups();
  readonly regions = new Map<string, Region>();
  readonly computes: Compute[] = [];
  readonly fixes: Fix[] = [];
  readonly vars: Variables;
  readonly thermo: Thermo;
  readonly dumps: Dump[] = [];
  /** min_style and min_modify settings. */
  minStyle = 'cg';
  minSettings: MinSettings = defaultMinSettings();
  /** Style names per category for is_available(): command, compute, fix, pair_style, ... */
  registries: Record<string, string[]> = {};
  /** Files the session can read: uploads plus everything it wrote. */
  readonly files = new Map<string, string>();
  /** molecule templates by ID (each a list of sets; create_atoms uses the first). */
  readonly molecules = new Map<string, MoleculeTemplate[]>();
  run: RunInfo = { inRun: false, firstStep: 0, lastStep: 0, beginStep: 0, endStep: 0, t0: 0, ranOnce: false };

  epoch = 0;
  stateVersion = 0;
  private forcesVersion = -1;
  private flagsUsed: ComputeFlags = {};
  private rngEqual: Rng | null = null;
  private rngAtom: Rng | null = null;

  constructor(readonly io: SessionIO) {
    this.nb = new Neighbor(this.units.skin);
    this.vars = new Variables((name) => this.readFile(name));
    this.thermo = new Thermo(this);
  }

  // ---------------------------------------------------------------- state

  get hasBox(): boolean { return this._state !== null; }

  get state(): SimState {
    if (!this._state) throw new StyleError('this needs a simulation box (create_box or read_data first)');
    return this._state;
  }

  /** Installs the state created by create_box / read_data. */
  setState(s: SimState): void {
    this._state = s;
    this.geom = new Geometry(s.box);
    if (this.pendingDt !== null) { s.dt = this.pendingDt; this.pendingDt = null; }
    this.ff.pair?.allocate(s.ntypes);
    // fix property/atom defined before the box adds its per-atom properties to the new state
    for (const f of this.fixes) (f as { attachState?: (st: SimState) => void }).attachState?.(s);
    this.bump();
  }

  /** Drops everything (clear). */
  reset(): void {
    this._state = null;
  }

  /** Positions, atoms, box or force field changed. */
  bump(): void {
    this.stateVersion++;
    this.epoch++;
  }

  /** Compute values must be re-evaluated (velocities changed, thermo output). */
  refreshComputes(): void {
    this.epoch++;
  }

  /** Forces were computed for the current state by the run loop. */
  forcesCurrent(): void {
    this.forcesVersion = this.stateVersion;
  }

  /** comm_modify vel yes|no (comm_modify.html: ghost atoms store velocity info). */
  ghostVelocity = false;

  styleContext(): StyleContext {
    return { s: this._state, readFile: (n) => this.readFile(n), log: (t) => this.log(t), ghostVelocity: this.ghostVelocity };
  }

  /** log file: a copy of the log text goes to this file (log.html). */
  private logFile: string | null = null;

  setLogFile(name: string | null, append: boolean): void {
    this.logFile = name;
    if (name && !append) this.writeFile(name, '', false);
  }

  log(text: string): void {
    this.io.emit({ kind: 'log', text });
    if (this.logFile) this.writeFile(this.logFile, text + '\n', true);
  }

  warn(text: string): void {
    this.log(`WARNING: ${text}`);
  }

  readFile(name: string): string {
    const f = this.files.get(name);
    if (f === undefined) {
      throw new StyleError(`cannot open file ${name}: add it to the notebook's files (upload) or write it earlier in the session`);
    }
    return f;
  }

  writeFile(name: string, text: string, append: boolean): void {
    this.files.set(name, (append ? this.files.get(name) ?? '' : '') + text);
    this.io.writeFile?.(name, text, append);
  }

  // ---------------------------------------------------------------- forces

  /** What per-atom force-field data the defined computes need. */
  computeFlags(): ComputeFlags {
    let eatom = false, vatom = false;
    for (const c of this.computes) { eatom ||= c.needsEatom; vatom ||= c.needsVatom; }
    return { eatom, vatom };
  }

  /**
   * Prepares the force field, remaps atoms into periodic boxes, re-fits
   * shrink-wrapped faces, checks for lost atoms and builds ghosts and
   * neighbor lists (the setup half of a run, Developer_flow.html).
   */
  setupNeighbors(): void {
    const s = this.state;
    this.ff.init(s, this.nb, this.geom, this.styleContext());
    this.pbc();
    this.nb.build(s, this.geom, s.step);
  }

  /** domain->pbc() and reset_box(): wrap periodic coordinates, shrink-wrap, drop lost atoms. */
  /**
   * Changes the box (used by box-changing fixes such as fix deform): new
   * bounds and tilt factors (xy, xz, yz); atoms whose mask has `remapBit` keep
   * their fractional coordinates, the rest keep their Cartesian ones. Other
   * fixes are told through boxChanged(); computes are invalidated.
   */
  setBox(lo: readonly number[], hi: readonly number[], tilt: readonly number[], remapBit: number, from?: Fix): void {
    const s = this.state;
    const g = this.geom;
    const lam = new Float64Array(3 * s.n);
    const tmp = [0, 0, 0];
    if (remapBit) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & remapBit)) continue;
        g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
        lam[3 * i] = tmp[0]; lam[3 * i + 1] = tmp[1]; lam[3 * i + 2] = tmp[2];
      }
    }
    const b = s.box;
    for (let d = 0; d < 3; d++) { b.lo[d] = lo[d]; b.hi[d] = hi[d]; b.tilt[d] = tilt[d]; }
    b.minLo = [...b.lo]; b.minHi = [...b.hi];
    g.update();
    if (remapBit) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & remapBit)) continue;
        g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
        s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
      }
    }
    for (const f of this.fixes) if (f !== from) f.boxChanged?.();
    this.bump();
  }

  pbc(): void {
    const s = this.state;
    const g = this.geom;
    for (let i = 0; i < s.n; i++) g.remap(s.x, s.image, i);
    if (shrinkWrap(s, g)) this.nb.forwardComm(s, g);
    this.checkLost();
  }

  /** Atoms outside a fixed (f) boundary: "it will be deleted on the next timestep that reneighboring occurs" (boundary.html). */
  private checkLost(): void {
    const s = this.state;
    const g = this.geom;
    const fixed = [0, 1, 2].filter((d) => s.box.boundary[d][0] === 'f' || s.box.boundary[d][1] === 'f');
    if (!fixed.length) return;
    const lam = [0, 0, 0];
    const lost = new Uint8Array(s.n);
    let nlost = 0;
    for (let i = 0; i < s.n; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      for (const d of fixed) {
        if (s.dimension === 2 && d === 2) continue;
        if ((s.box.boundary[d][0] === 'f' && lam[d] < 0) || (s.box.boundary[d][1] === 'f' && lam[d] > 1)) {
          lost[i] = 1; nlost++; break;
        }
      }
    }
    if (!nlost) return;
    const mode = this.thermo.lost;
    if (mode === 'error') throw new StyleError(`lost atoms: ${nlost} atom(s) left the box through a fixed (f) boundary (thermo_modify lost ignore/warn allows this)`);
    if (mode === 'warn') this.warn(`lost ${nlost} atom(s) through a fixed boundary`);
    this.deleteAtoms(lost);
  }

  /** Deletes flagged atoms and keeps every dependent structure consistent. */
  deleteAtoms(flags: Uint8Array): number {
    const n = deleteAtomsImpl(this.state, flags);
    if (n) this.atomsChanged();
    return n;
  }

  /** Atoms were added, deleted or reordered. */
  atomsChanged(): void {
    this.ff.atomsChanged();
    this.ff.topologyChanged(this.state);
    this.ff.updateTail(this.state);
    this.nb.lastBuild = -1;
    this.bump();
  }

  /** Forces, energies and virial for the current state (computed on demand between runs). */
  forces(): Accum {
    if (this.forcesVersion !== this.stateVersion) {
      const flags = this.computeFlags();
      this.setupNeighbors();
      this.ff.compute(this.state, this.nb, this.geom, flags);
      this.flagsUsed = flags;
      this.forcesVersion = this.stateVersion;
    }
    return this.ff.acc;
  }

  /** Per-atom energy / virial for the current state. */
  peratomEnergy(): { eatom: Float64Array | null; vatom: Float64Array | null } {
    const want = this.computeFlags();
    if ((want.eatom && !this.flagsUsed.eatom) || (want.vatom && !this.flagsUsed.vatom)) this.forcesVersion = -1;
    this.forces();
    return { eatom: this.ff.eatom, vatom: this.ff.vatom };
  }

  /** Force evaluation inside the run loop (lists already current). */
  computeForcesInRun(flags: ComputeFlags): Accum {
    this.flagsUsed = flags;
    return this.ff.compute(this.state, this.nb, this.geom, flags);
  }

  fixEnergy(): number {
    let e = 0;
    for (const f of this.fixes) if (f.thermoEnergy) e += f.energy();
    return e;
  }

  fixVirial(out: Float64Array): void {
    for (const f of this.fixes) {
      if (!f.thermoVirial) continue;
      for (let c = 0; c < 6; c++) out[c] += f.virial[c];
    }
  }

  dofRemoved(groupBit: number): number {
    let n = 0;
    for (const f of this.fixes) n += f.dofRemoved(groupBit);
    return n;
  }

  /** Cumulative energy exchanged with thermostat/barostat reservoirs (thermo ecouple). */
  ecouple(): number {
    let e = 0;
    for (const f of this.fixes) e += f.ecouple?.() ?? 0;
    return e;
  }

  // ---------------------------------------------------------------- lookups

  compute(id: string): Compute {
    const c = this.computes.find((x) => x.id === id);
    if (!c) throw new StyleError(`compute ID '${id}' does not exist`);
    return c;
  }

  fix(id: string): Fix {
    const f = this.fixes.find((x) => x.id === id);
    if (!f) throw new StyleError(`fix ID '${id}' does not exist`);
    return f;
  }

  region(id: string): Region {
    const r = this.regions.get(id);
    if (!r) throw new StyleError(`region ID '${id}' does not exist`);
    return r;
  }

  get regionEnv(): RegionEnv {
    return {
      variable: (name) => this.vars.scalar(name, this.formulaEnv),
      region: (id) => this.regions.get(id),
    };
  }

  /** Per-atom group mask bit by name. */
  groupBit(name: string): number { return this.groups.bit(name); }

  // ---------------------------------------------------------------- formulas

  readonly formulaEnv: FormulaEnv = {
    thermo: (name) => this.thermo.keyword(name),
    variable: (name, mode) => this.variableValue(name, mode),
    variableElement: (name, i) => {
      const v = this.vars.evalVector(name, this.formulaEnv);
      if (i < 1 || i > v.length) throw new StyleError(`v_${name}[${i}] is out of range (1..${v.length})`);
      return v[i - 1];
    },
    reference: (kind, id, i, j, mode) => this.reference(kind, id, i, j, mode),
    atomValue: (name, id) => this.atomValue(name, id),
    atomVector: (name) => this.atomVector(name),
    raw: (fn, args, mode) => groupFunction(this, fn, args, mode),
    timing: () => ({
      step: this.hasBox ? this.state.step : 0,
      startStep: this.run.beginStep,
      stopStep: this.run.endStep,
      dt: this.hasBox ? this.state.dt : this.units.dt,
      inRun: this.run.inRun,
      firstStep: this.run.firstStep,
    }),
    random: (lo, hi, seed, mode) => this.randomValue(lo, hi, seed, mode, false),
    normal: (mu, sg, seed, mode) => this.randomValue(mu, sg, seed, mode, true),
    natoms: () => (this.hasBox ? this.state.n : 0),
  };

  private variableValue(name: string, mode: Mode): Value {
    const s = this._state;
    return this.vars.value(name, this.formulaEnv, mode, s ? s.id : new Int32Array(0), s ? s.n : 0);
  }

  private randomValue(a: number, b: number, seed: number, mode: Mode, gauss: boolean): Value {
    if (!(seed > 0)) throw new StyleError('random()/normal() seed must be > 0');
    const draw = (r: Rng) => (gauss ? a + b * r.gaussian() : a + (b - a) * r.uniform());
    if (mode !== 'atom') {
      this.rngEqual ??= new Rng(Math.trunc(seed));
      return draw(this.rngEqual);
    }
    this.rngAtom ??= new Rng(Math.trunc(seed) + 1);
    const n = this.state.n;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = draw(this.rngAtom);
    return out;
  }

  private reference(kind: 'c' | 'C' | 'f' | 'F', id: string, i: number | null, j: number | null, mode: Mode): Value {
    const s = this.state;
    if (kind === 'C' || kind === 'F') {
      // per-atom value of the atom with ID i (equal-style access to per-atom data)
      if (i === null) throw new StyleError(`${kind}_${id} needs an atom ID index`);
      const vals = kind === 'C' ? this.compute(id).peratomValues() : this.fixPeratom(id);
      const cols = kind === 'C' ? this.compute(id).sizePeratomCols : this.fix(id).sizePeratomCols;
      const k = this.indexOfId(i);
      if (k < 0) return 0;
      if (cols === 0) return vals[k];
      if (j === null || j < 1 || j > cols) throw new StyleError(`${kind}_${id}[${i}] needs a column index 1..${cols}`);
      return vals[k * cols + j - 1];
    }
    if (kind === 'c') {
      const c = this.compute(id);
      if (mode === 'atom' && c.peratomFlag) {
        const vals = c.peratomValues();
        if (c.sizePeratomCols === 0) {
          if (i !== null) throw new StyleError(`c_${id} is a per-atom vector: no index in an atom-style formula`);
          return vals;
        }
        if (i === null || i < 1 || i > c.sizePeratomCols) throw new StyleError(`c_${id} needs a column index 1..${c.sizePeratomCols}`);
        const out = new Float64Array(s.n);
        for (let k = 0; k < s.n; k++) out[k] = vals[k * c.sizePeratomCols + i - 1];
        return out;
      }
      if (mode === 'vector' && i === null && c.vectorFlag) return c.vectorValues();
      if (mode === 'vector' && i !== null && j === null && c.arrayFlag) {
        const a = c.arrayValues();
        const out = new Float64Array(c.sizeArrayRows);
        for (let r = 0; r < c.sizeArrayRows; r++) out[r] = a[r * c.sizeArrayCols + i - 1];
        return out;
      }
      if (i === null) return c.scalarValue();
      if (j === null) {
        const v = c.vectorValues();
        if (i < 1 || i > v.length) throw new StyleError(`c_${id}[${i}] is out of range (1..${v.length})`);
        return v[i - 1];
      }
      const a = c.arrayValues();
      if (i < 1 || i > c.sizeArrayRows || j < 1 || j > c.sizeArrayCols) throw new StyleError(`c_${id}[${i}][${j}] is out of range`);
      return a[(i - 1) * c.sizeArrayCols + j - 1];
    }
    const f = this.fix(id);
    if (mode === 'atom' && f.peratomFlag) {
      const vals = this.fixPeratom(id);
      if (f.sizePeratomCols === 0) return vals;
      if (i === null) throw new StyleError(`f_${id} needs a column index`);
      const out = new Float64Array(s.n);
      for (let k = 0; k < s.n; k++) out[k] = vals[k * f.sizePeratomCols + i - 1];
      return out;
    }
    if (mode === 'vector' && i === null && f.vectorFlag) {
      const out = new Float64Array(f.sizeVector);
      for (let k = 0; k < f.sizeVector; k++) out[k] = f.computeVector(k);
      return out;
    }
    if (i === null) return f.computeScalar();
    if (j === null) {
      if (i < 1 || i > f.sizeVector) throw new StyleError(`f_${id}[${i}] is out of range (1..${f.sizeVector})`);
      return f.computeVector(i - 1);
    }
    return f.computeArray(i - 1, j - 1);
  }

  private fixPeratom(id: string): Float64Array {
    const f = this.fix(id);
    if (!f.peratomFlag) throw new StyleError(`fix ${id} does not compute per-atom values`);
    return f.sizePeratomCols === 0 ? f.vectorAtom : f.arrayAtom;
  }

  private idMap: Int32Array | null = null;
  private idMapVersion = -1;

  /** Index of the atom with this ID, or -1. */
  indexOfId(id: number): number {
    if (!this.idMap || this.idMapVersion !== this.stateVersion) {
      this.idMap = buildAtomMap(this.state);
      this.idMapVersion = this.stateVersion;
    }
    return id >= 0 && id < this.idMap.length ? this.idMap[id] : -1;
  }

  private atomValue(name: string, id: number): number {
    const k = this.indexOfId(id);
    if (k < 0) return 0;
    return this.atomVectorAt(name, k);
  }

  private atomVectorAt(name: string, i: number): number {
    const s = this.state;
    switch (name) {
      case 'id': return s.id[i];
      case 'mass': return massOf(s, i);
      case 'type': return s.type[i];
      case 'mol': return s.molecule[i];
      case 'x': return s.x[3 * i];
      case 'y': return s.x[3 * i + 1];
      case 'z': return s.x[3 * i + 2];
      case 'vx': return s.v[3 * i];
      case 'vy': return s.v[3 * i + 1];
      case 'vz': return s.v[3 * i + 2];
      case 'fx': return s.f[3 * i];
      case 'fy': return s.f[3 * i + 1];
      case 'fz': return s.f[3 * i + 2];
      case 'q': return s.q[i];
      // variable.html: "atom vector = id, mass, type, mol, radius, q, x, y, z, vx, vy, vz, fx, fy, fz"
      case 'radius':
        if (!s.radius) throw new StyleError('variable uses atom property radius, which needs atom_style sphere');
        return s.radius[i];
    }
    const custom = customAttr(s, name);
    if (custom) return custom(i);
    throw new StyleError(`unknown atom vector ${name}`);
  }

  private atomVector(name: string): Float64Array {
    const s = this.state;
    if (name.startsWith('f')) this.forces();
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) out[i] = this.atomVectorAt(name, i);
    return out;
  }

  /** An equal-style formula's value (used by commands that take v_name arguments). */
  equalVariable(name: string): number {
    return this.vars.scalar(name, this.formulaEnv);
  }

  /** Per-atom values of an atom-style variable. */
  atomVariable(name: string): Float64Array {
    const s = this.state;
    return this.vars.evalAtom(name, this.formulaEnv, s.id, s.n);
  }
}

