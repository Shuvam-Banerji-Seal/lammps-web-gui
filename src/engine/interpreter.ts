import {
  EngineError, THERMO_KEYWORDS,
  type EngineEvent, type FixSpec, type ForceBackend, type ForceResult, type PairTable, type SimState,
  type ThermoKeyword, type UnitSystem,
} from './types';
import { UNIT_SYSTEMS, isUnitStyle } from './units';
import { Rng } from './rng';
import { isLatticeStyle, LATTICE_STYLES, latticePoints, makeLattice, type Lattice } from './lattice';
import { newPairTable, resolvePairs, setPairCoeff, typeRange } from './pairs';
import { CpuForceBackend } from './cpu/forces';
import type { Fix } from './integrate';
import { makeFix } from './fixes';
import { addAtoms, emptyState, run as runSteps } from './md';
import { createVelocities, scaleVelocities, setVelocities, zeroAngularMomentum, zeroMomentum } from './velocity';
import { thermoRow } from './observables';
import { evaluateFormula } from './expr';
import { formatNumber, mapUnquoted, splitCommands, substituteVariables, tokenize } from './script';

/*
 * A LAMMPS-input-subset interpreter for the in-browser notebook
 * (docs/design/notebook.md). Every command it accepts follows the
 * documented semantics on docs.lammps.org (quoted at each handler); every
 * command it does not support is an EngineError naming the command, the line
 * and the supported set — never a silent no-op.
 */

export const SUPPORTED_COMMANDS = [
  'units', 'dimension', 'boundary', 'atom_style', 'lattice', 'region', 'create_box', 'create_atoms',
  'mass', 'velocity', 'pair_style', 'pair_coeff', 'pair_modify', 'neighbor', 'neigh_modify',
  'timestep', 'fix', 'unfix', 'thermo', 'thermo_style', 'thermo_modify', 'run', 'dump', 'undump',
  'write_data', 'print', 'variable', 'reset_timestep', 'clear',
] as const;

const FIX_STYLES = ['nve', 'langevin', 'temp/berendsen', 'temp/rescale', 'nvt', 'enforce2d'];

/** thermo_style one: "step temp epair emol etotal press" (docs.lammps.org/thermo_style.html). */
const THERMO_ONE: ThermoKeyword[] = ['step', 'temp', 'epair', 'emol', 'etotal', 'press'];

export interface SessionIO {
  emit(ev: EngineEvent): void;
  /** A command wrote (or appended to) a file: dump, write_data, print file/append. */
  writeFile?(name: string, text: string, append: boolean): void;
}

interface Region { id: string; lo: number[]; hi: number[] }
interface Variable { style: 'equal' | 'string' | 'index'; value: string }
interface Dump {
  id: string;
  every: number;
  file: string;
  style: 'atom' | 'custom';
  columns: string[];
  lastStep: number;
}

const DUMP_COLUMNS = ['id', 'type', 'mass', 'x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu',
  'ix', 'iy', 'iz', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz'];

/** Thrown when a run is cancelled; the session stays usable. */
export class RunCancelled extends Error {
  constructor() { super('run cancelled'); this.name = 'RunCancelled'; }
}

export class Session {
  private units: UnitSystem = UNIT_SYSTEMS.lj;
  private dimension: 2 | 3 = 3;
  private lattice: Lattice | null = null;
  private regions = new Map<string, Region>();
  private state: SimState | null = null;
  private pairCutoff: number | null = null;
  private table: PairTable | null = null;
  private fixes: { spec: FixSpec; fix: Fix }[] = [];
  private thermoEvery = 0;
  private thermoKeywords: ThermoKeyword[] = THERMO_ONE;
  private norm: boolean | null = null;
  private dumps: Dump[] = [];
  private variables = new Map<string, Variable>();
  private forces: ForceResult | null = null;
  private exprRng: Rng | null = null;
  private cancelled = false;
  private line = 0;
  private command = '';

  /** fp64 CPU forces for thermo keywords evaluated between runs (any backend). */
  private probe = new CpuForceBackend();

  constructor(
    private io: SessionIO,
    private backend: ForceBackend = new CpuForceBackend(),
    /** Also emit a 'frame' every this many steps during a run (0 = only at dumps and at the end). */
    private frameEvery = 0,
  ) {}

  /** The current system (null before create_box). */
  get system(): SimState | null { return this.state; }
  get backendLabel(): string { return this.backend.label; }

  /** Swaps the force backend between runs; the system and settings stay. */
  setBackend(backend: ForceBackend): void {
    this.backend = backend;
    this.invalidate();
  }

  /** Requests that a running `run` stop after its current step. */
  cancel(): void { this.cancelled = true; }

  /**
   * Executes input text. Stops at the first error, which is emitted as an
   * 'error' event and rethrown as an EngineError.
   */
  async execute(text: string, firstLine = 1): Promise<void> {
    this.cancelled = false;
    for (const cmd of splitCommands(text, firstLine)) {
      this.line = cmd.line;
      this.command = '';
      try {
        const substituted = mapUnquoted(cmd.text, (part) => substituteVariables(
          part, (name) => this.variableText(name), (f, fmt) => this.immediate(f, fmt)));
        const words = tokenize(substituted);
        if (words.length === 0) continue;
        this.command = words[0];
        await this.dispatch(words[0], words.slice(1));
      } catch (e) {
        if (e instanceof RunCancelled) {
          this.io.emit({ kind: 'log', text: `Run cancelled at step ${this.state?.step ?? 0}` });
          throw e;
        }
        const err = e instanceof EngineError ? e
          : new EngineError(e instanceof Error ? e.message : String(e), this.line, this.command);
        this.io.emit({ kind: 'error', message: err.message, line: err.line, command: err.command });
        throw err;
      }
    }
  }

  private fail(message: string): never {
    throw new EngineError(message, this.line, this.command);
  }

  private log(text: string): void {
    this.io.emit({ kind: 'log', text });
  }

  // ---------------------------------------------------------------- helpers

  private num(word: string | undefined, what: string): number {
    if (word === undefined) this.fail(`missing ${what}`);
    const v = Number(word);
    if (word.trim() === '' || !Number.isFinite(v)) this.fail(`expected a number for ${what}, got '${word}'`);
    return v;
  }

  private int(word: string | undefined, what: string): number {
    const v = this.num(word, what);
    if (!Number.isInteger(v)) this.fail(`expected an integer for ${what}, got '${word}'`);
    return v;
  }

  private yesNo(word: string | undefined, what: string): boolean {
    if (word === 'yes') return true;
    if (word === 'no') return false;
    return this.fail(`${what} must be yes or no, got '${word ?? ''}'`);
  }

  private needBox(): SimState {
    if (!this.state) this.fail(`${this.command} needs a simulation box: use create_box first`);
    return this.state;
  }

  private needAllGroup(group: string | undefined): void {
    if (group !== 'all') this.fail(`only the group 'all' is supported (groups are not implemented), got '${group ?? ''}'`);
  }

  private noBoxYet(): void {
    // "This command must be used before the simulation box is defined"
    if (this.state) this.fail(`${this.command} must be used before the simulation box is defined (create_box)`);
  }

  private variableText(name: string): string {
    const v = this.variables.get(name);
    if (!v) this.fail(`substitution for undefined variable '${name}'`);
    return v.style === 'equal' ? formatNumber(this.evaluate(v.value), '%.15g') : v.value;
  }

  private immediate(formula: string, format?: string): string {
    return formatNumber(this.evaluate(formula), format ?? '%.20g');
  }

  private evaluate(formula: string, depth = 0): number {
    if (depth > 32) this.fail('variable references are nested too deeply (a variable refers to itself?)');
    return evaluateFormula(formula, {
      thermo: (name) => this.thermoValue(name),
      variable: (name) => {
        const v = this.variables.get(name);
        if (!v) this.fail(`formula references undefined variable v_${name}`);
        if (v.style === 'equal') return this.evaluate(v.value, depth + 1);
        const n = Number(v.value);
        if (!Number.isFinite(n)) this.fail(`variable ${name} = '${v.value}' is not numeric`);
        return n;
      },
      rng: (seed) => {
        if (!this.exprRng) this.exprRng = new Rng(seed);
        return this.exprRng;
      },
    });
  }

  private thermoValue(name: string): number | undefined {
    if (!(THERMO_KEYWORDS as readonly string[]).includes(name)) return undefined;
    const s = this.state;
    if (!s) this.fail(`thermo keyword '${name}' used before the simulation box exists`);
    return thermoRow(s, [name as ThermoKeyword], this.currentForces(), { norm: this.norm ?? undefined })[name as ThermoKeyword];
  }

  /** Forces for the current positions (computed on demand between runs). */
  private currentForces(): ForceResult {
    const s = this.needBox();
    if (!this.forces) {
      this.forces = this.probe.compute(s, this.tableForRun(false));
    }
    return this.forces;
  }

  private tableForRun(strict: boolean): PairTable {
    const s = this.needBox();
    if (!this.table) return newPairTable(s.ntypes, 0);
    const missing = resolvePairs(this.table);
    if (missing.length && strict) this.fail(`all pair coefficients must be set before a run; missing: ${missing.join(', ')}`);
    return this.table;
  }

  private invalidate(): void {
    this.forces = null;
  }

  // ---------------------------------------------------------------- commands

  private async dispatch(cmd: string, a: string[]): Promise<void> {
    switch (cmd) {
      case 'units': return this.cmdUnits(a);
      case 'dimension': return this.cmdDimension(a);
      case 'boundary': return this.cmdBoundary(a);
      case 'atom_style': return this.cmdAtomStyle(a);
      case 'lattice': return this.cmdLattice(a);
      case 'region': return this.cmdRegion(a);
      case 'create_box': return this.cmdCreateBox(a);
      case 'create_atoms': return this.cmdCreateAtoms(a);
      case 'mass': return this.cmdMass(a);
      case 'velocity': return this.cmdVelocity(a);
      case 'pair_style': return this.cmdPairStyle(a);
      case 'pair_coeff': return this.cmdPairCoeff(a);
      case 'pair_modify': return this.cmdPairModify(a);
      case 'neighbor':
      case 'neigh_modify':
        this.log(`${cmd}: accepted; the engine evaluates every pair inside the cutoff on every step, so neighbor-list settings have no effect`);
        return;
      case 'timestep': return this.cmdTimestep(a);
      case 'fix': return this.cmdFix(a);
      case 'unfix': return this.cmdUnfix(a);
      case 'thermo': return this.cmdThermo(a);
      case 'thermo_style': return this.cmdThermoStyle(a);
      case 'thermo_modify': return this.cmdThermoModify(a);
      case 'run': return this.cmdRun(a);
      case 'dump': return this.cmdDump(a);
      case 'undump': return this.cmdUndump(a);
      case 'write_data': return this.cmdWriteData(a);
      case 'print': return this.cmdPrint(a);
      case 'variable': return this.cmdVariable(a);
      case 'reset_timestep': return this.cmdResetTimestep(a);
      case 'clear': return this.cmdClear(a);
      default:
        this.fail(`'${cmd}' is not supported by the in-browser engine (it is not LAMMPS). Supported commands: ${SUPPORTED_COMMANDS.join(', ')}`);
    }
  }

  /** units style — docs.lammps.org/units.html; "This command cannot be used after the simulation box is defined". */
  private cmdUnits(a: string[]): void {
    this.noBoxYet();
    if (a.length !== 1) this.fail('usage: units lj|real|metal');
    if (!isUnitStyle(a[0])) this.fail(`units '${a[0]}' is not supported; supported: lj, real, metal`);
    this.units = UNIT_SYSTEMS[a[0]];
  }

  /** dimension N — docs.lammps.org/dimension.html, default 3. */
  private cmdDimension(a: string[]): void {
    this.noBoxYet();
    if (a[0] !== '2' && a[0] !== '3') this.fail('usage: dimension 2|3');
    this.dimension = a[0] === '2' ? 2 : 3;
  }

  /** boundary x y z — default "boundary p p p"; v1 supports only fully periodic boxes. */
  private cmdBoundary(a: string[]): void {
    this.noBoxYet();
    if (a.length !== 3 || a.some((b) => b !== 'p')) {
      this.fail(`boundary '${a.join(' ')}' is not supported; the engine runs fully periodic boxes only (boundary p p p)`);
    }
  }

  /** atom_style — "The default atom style is atomic"; only atomic in v1. */
  private cmdAtomStyle(a: string[]): void {
    this.noBoxYet();
    if (a.length !== 1 || a[0] !== 'atomic') this.fail(`atom_style '${a.join(' ')}' is not supported; only atomic`);
  }

  /** lattice style scale — docs.lammps.org/lattice.html; keywords are not supported in v1. */
  private cmdLattice(a: string[]): void {
    if (a.length < 2) this.fail('usage: lattice style scale');
    if (a[0] === 'none') { this.num(a[1], 'lattice none scale'); this.lattice = null; return; }
    if (!isLatticeStyle(a[0])) this.fail(`lattice style '${a[0]}' is not supported; supported: none, ${LATTICE_STYLES.join(', ')}`);
    if (a.length > 2) this.fail(`lattice keywords (${a.slice(2).join(' ')}) are not supported; only 'lattice style scale'`);
    try {
      this.lattice = makeLattice(a[0], this.num(a[1], 'lattice scale'), this.units, this.dimension);
    } catch (e) {
      this.fail((e as Error).message);
    }
    const sp = this.lattice!.spacing.map((v) => formatNumber(v, '%.8g')).join(' ');
    this.log(`Lattice spacing in x,y,z = ${sp}`);
  }

  /**
   * region ID block xlo xhi ylo yhi zlo zhi [side in] [units lattice|box].
   * docs.lammps.org/region.html: "INF represents a large number (1.0e20)";
   * EDGE "extends to the current simulation box boundary"; defaults
   * "side = in, units = lattice"; with lattice units "the lattice spacing in
   * dimension x is applied to xlo and xhi"; "Coordinates exactly on the
   * region boundary are considered to be interior to the region."
   */
  private cmdRegion(a: string[]): void {
    const [id, style] = a;
    if (!id || !style) this.fail('usage: region ID block xlo xhi ylo yhi zlo zhi [units lattice|box]');
    if (style !== 'block') this.fail(`region style '${style}' is not supported; only block`);
    if (a.length < 8) this.fail('region block needs xlo xhi ylo yhi zlo zhi');
    let units: 'lattice' | 'box' = 'lattice';
    for (let k = 8; k < a.length; k += 2) {
      const [key, val] = [a[k], a[k + 1]];
      if (key === 'units' && (val === 'lattice' || val === 'box')) units = val;
      else if (key === 'side' && val === 'in') { /* default */ }
      else this.fail(`region keyword '${key} ${val ?? ''}' is not supported; only 'units lattice|box' and 'side in'`);
    }
    if (units === 'lattice' && !this.lattice) {
      this.fail("region uses lattice units (the default) but no lattice is defined: use 'lattice ...' first or 'units box'");
    }
    const scale = units === 'lattice' ? this.lattice!.spacing : [1, 1, 1];
    const lo: number[] = [];
    const hi: number[] = [];
    for (let d = 0; d < 3; d++) {
      for (const [k, out, sign] of [[2 + 2 * d, lo, -1], [3 + 2 * d, hi, 1]] as const) {
        const w = a[k];
        if (w === 'INF') out.push(sign * 1e20);
        else if (w === 'EDGE') {
          if (!this.state) this.fail('EDGE needs an existing simulation box');
          out.push(sign < 0 ? this.state.box.lo[d] : this.state.box.hi[d]);
        } else out.push(this.num(w, 'region bound') * scale[d]);
      }
      if (!(lo[d] < hi[d])) this.fail(`region ${id}: lo must be smaller than hi in dimension ${'xyz'[d]}`);
    }
    this.regions.set(id, { id, lo, hi });
  }

  /**
   * create_box N region-ID — docs.lammps.org/create_box.html: the box is the
   * region's extent; "For two-dimensional simulations, the z-axis bounds
   * must bracket zero."
   */
  private cmdCreateBox(a: string[]): void {
    if (this.state) this.fail('a simulation box already exists (use clear to start over)');
    if (a.length !== 2) this.fail('usage: create_box N region-ID (keywords are not supported)');
    const n = this.int(a[0], 'number of atom types');
    if (n < 1) this.fail('create_box needs at least 1 atom type');
    const reg = this.regions.get(a[1]);
    if (!reg) this.fail(`unknown region '${a[1]}'`);
    if ([...reg.lo, ...reg.hi].some((v) => Math.abs(v) >= 1e20)) this.fail('create_box needs a finite region (no INF bounds)');
    if (this.dimension === 2 && !(reg.lo[2] < 0 && reg.hi[2] > 0)) {
      this.fail('for a 2d simulation the region z bounds must bracket zero (e.g. -0.5 0.5)');
    }
    this.state = emptyState(this.units, this.dimension,
      { lo: [reg.lo[0], reg.lo[1], reg.lo[2]], hi: [reg.hi[0], reg.hi[1], reg.hi[2]], periodic: [true, true, true] }, n);
    if (this.pairCutoff !== null) this.table = newPairTable(n, this.pairCutoff);
    if (this.pendingDt !== null) { this.state.dt = this.pendingDt; this.pendingDt = null; }
    this.invalidate();
    const f = (v: number[]) => v.map((x) => formatNumber(x, '%.8g')).join(' ');
    this.log(`Created orthogonal box = (${f(reg.lo)}) to (${f(reg.hi)})`);
  }

  /**
   * create_atoms type box | region ID | single x y z | random N seed region-ID.
   * docs.lammps.org/create_atoms.html: defaults "overlap not checked,
   * maxtry = 10, and units = lattice"; random: "particles are created one by
   * one using the specified random number seed"; overlap: not "closer than
   * the specified distance from any other particle".
   */
  private cmdCreateAtoms(a: string[]): void {
    const s = this.needBox();
    const type = this.int(a[0], 'atom type');
    if (type < 1 || type > s.ntypes) this.fail(`atom type ${type} is outside 1..${s.ntypes}`);
    const style = a[1];
    let rest: string[];
    let pts: Float64Array;
    if (style === 'box' || style === 'region') {
      if (!this.lattice) this.fail(`create_atoms ${style} needs a lattice: use 'lattice ...' first`);
      let region: Region | undefined;
      if (style === 'region') {
        region = this.regions.get(a[2]);
        if (!region) this.fail(`unknown region '${a[2]}'`);
        rest = a.slice(3);
      } else rest = a.slice(2);
      this.noKeywords(rest);
      pts = latticePoints(this.lattice, s.box.lo, s.box.hi, this.dimension, region);
    } else if (style === 'single') {
      const opts = this.keywords(a.slice(5), ['units']);
      const units = opts.units ?? 'lattice';
      if (units !== 'lattice' && units !== 'box') this.fail("units must be lattice or box");
      if (units === 'lattice' && !this.lattice) this.fail('create_atoms single in lattice units needs a lattice (or add units box)');
      const scale = units === 'lattice' ? this.lattice!.spacing : [1, 1, 1];
      const p = [0, 1, 2].map((d) => this.num(a[2 + d], 'coordinate') * scale[d]);
      if (this.dimension === 2) p[2] = 0;
      for (let d = 0; d < 3; d++) {
        if (p[d] < s.box.lo[d] || p[d] >= s.box.hi[d]) this.fail(`the point (${p.join(', ')}) is outside the box`);
      }
      pts = Float64Array.from(p);
    } else if (style === 'random') {
      const count = this.int(a[2], 'number of atoms');
      const seed = this.int(a[3], 'seed');
      if (seed <= 0) this.fail('seed must be a positive integer');
      const regionId = a[4];
      if (!regionId) this.fail('usage: create_atoms type random N seed region-ID|NULL');
      const region = regionId === 'NULL' ? undefined : this.regions.get(regionId);
      if (regionId !== 'NULL' && !region) this.fail(`unknown region '${regionId}'`);
      const opts = this.keywords(a.slice(5), ['overlap', 'maxtry', 'units']);
      const overlap = opts.overlap !== undefined ? this.num(opts.overlap, 'overlap') : 0;
      const maxtry = opts.maxtry !== undefined ? this.int(opts.maxtry, 'maxtry') : 10;
      pts = this.randomPoints(s, count, seed, region, overlap, maxtry);
      if (pts.length / 3 < count) this.log(`WARNING: only ${pts.length / 3} of ${count} atoms could be placed (overlap ${overlap}, maxtry ${maxtry})`);
    } else {
      this.fail(`create_atoms style '${style ?? ''}' is not supported; supported: box, region, single, random`);
    }
    const added = addAtoms(s, pts, type);
    this.invalidate();
    this.log(`Created ${added} atoms`);
    this.emitFrame();
  }

  private randomPoints(
    s: SimState, count: number, seed: number, region: Region | undefined, overlap: number, maxtry: number,
  ): Float64Array {
    const rng = new Rng(seed);
    const lo = [0, 1, 2].map((d) => Math.max(s.box.lo[d], region ? region.lo[d] : -Infinity));
    const hi = [0, 1, 2].map((d) => Math.min(s.box.hi[d], region ? region.hi[d] : Infinity));
    if (lo.some((v, d) => !(v < hi[d]) && !(d === 2 && this.dimension === 2))) this.fail('the region does not overlap the box');
    const L = [0, 1, 2].map((d) => s.box.hi[d] - s.box.lo[d]);
    const placed: number[] = [];
    const existing = Array.from(s.x);
    const tooClose = (p: number[]) => {
      if (overlap <= 0) return false;
      const all = [existing, placed];
      for (const arr of all) {
        for (let j = 0; j < arr.length; j += 3) {
          let r2 = 0;
          for (let d = 0; d < 3; d++) {
            let dx = p[d] - arr[j + d];
            dx -= L[d] * Math.round(dx / L[d]);
            r2 += dx * dx;
          }
          if (r2 < overlap * overlap) return true;
        }
      }
      return false;
    };
    for (let k = 0; k < count; k++) {
      for (let attempt = 0; attempt < (overlap > 0 ? maxtry : 1); attempt++) {
        const p = [0, 1, 2].map((d) => (d === 2 && this.dimension === 2 ? 0 : lo[d] + rng.uniform() * (hi[d] - lo[d])));
        if (!tooClose(p)) { placed.push(...p); break; }
      }
    }
    return Float64Array.from(placed);
  }

  private noKeywords(rest: string[]): void {
    if (rest.length) this.fail(`unsupported create_atoms keywords: ${rest.join(' ')}`);
  }

  /** key value pairs restricted to `allowed`. */
  private keywords(words: string[], allowed: string[]): Record<string, string> {
    const out: Record<string, string> = {};
    for (let k = 0; k < words.length; k += 2) {
      if (!allowed.includes(words[k])) this.fail(`keyword '${words[k]}' is not supported here; supported: ${allowed.join(', ') || 'none'}`);
      if (words[k + 1] === undefined) this.fail(`keyword '${words[k]}' needs a value`);
      out[words[k]] = words[k + 1];
    }
    return out;
  }

  /** mass I value — I may be a wildcard; "All masses must be defined before a simulation is run." */
  private cmdMass(a: string[]): void {
    const s = this.needBox();
    if (a.length !== 2) this.fail('usage: mass I value');
    const m = this.num(a[1], 'mass');
    if (!(m > 0)) this.fail('mass must be > 0');
    let range: [number, number];
    try { range = typeRange(a[0], s.ntypes); } catch (e) { this.fail((e as Error).message); }
    for (let t = range[0]; t <= range[1]; t++) s.massByType[t] = m;
    this.invalidate();
  }

  /**
   * velocity group create T seed | scale T | set vx vy vz | zero linear|angular.
   * docs.lammps.org/velocity.html defaults: "dist = uniform, sum = no,
   * mom = yes, rot = no, bias = no, loop = all, and units = lattice".
   */
  private cmdVelocity(a: string[]): void {
    const s = this.needBox();
    this.needAllGroup(a[0]);
    this.requireMasses(s);
    const style = a[1];
    if (style === 'create') {
      const t = this.num(a[2], 'temperature');
      const seed = this.int(a[3], 'seed');
      if (seed <= 0) this.fail('seed must be a positive integer');
      const kw = this.keywords(a.slice(4), ['dist', 'mom', 'rot', 'loop', 'sum', 'bias']);
      if (kw.dist && kw.dist !== 'uniform' && kw.dist !== 'gaussian') this.fail('dist must be uniform or gaussian');
      if (kw.loop && !['all', 'local', 'geom'].includes(kw.loop)) this.fail('loop must be all, local or geom');
      if (kw.sum && this.yesNo(kw.sum, 'sum')) this.fail('sum yes is not supported');
      if (kw.bias && this.yesNo(kw.bias, 'bias')) this.fail('bias yes is not supported');
      createVelocities(s, t, seed, {
        dist: (kw.dist as 'uniform' | 'gaussian' | undefined) ?? 'uniform',
        mom: kw.mom ? this.yesNo(kw.mom, 'mom') : true,
        rot: kw.rot ? this.yesNo(kw.rot, 'rot') : false,
      });
    } else if (style === 'scale') {
      if (a.length !== 3) this.fail('usage: velocity all scale T');
      scaleVelocities(s, this.num(a[2], 'temperature'));
    } else if (style === 'set') {
      const kw = this.keywords(a.slice(5), ['units', 'sum']);
      const units = kw.units ?? 'lattice';
      if (units !== 'lattice' && units !== 'box') this.fail('units must be lattice or box');
      if (kw.sum && this.yesNo(kw.sum, 'sum')) this.fail('sum yes is not supported');
      if (units === 'lattice' && !this.lattice) this.fail("velocity set uses lattice units by default but no lattice is defined; add 'units box'");
      const scale = units === 'lattice' ? this.lattice!.spacing : [1, 1, 1];
      const comp = (w: string | undefined, d: number) => (w === 'NULL' ? null : this.num(w, 'velocity component') * scale[d]);
      setVelocities(s, comp(a[2], 0), comp(a[3], 1), comp(a[4], 2));
    } else if (style === 'zero') {
      if (a[2] === 'linear') zeroMomentum(s);
      else if (a[2] === 'angular') zeroAngularMomentum(s);
      else this.fail('usage: velocity all zero linear|angular');
    } else {
      this.fail(`velocity style '${style ?? ''}' is not supported; supported: create, scale, set, zero`);
    }
  }

  private requireMasses(s: SimState): void {
    for (let t = 1; t <= s.ntypes; t++) {
      if (!(s.massByType[t] > 0)) this.fail(`the mass of atom type ${t} is not set (use mass ${t} value)`);
    }
  }

  /** pair_style lj/cut rc — only lj/cut in v1. */
  private cmdPairStyle(a: string[]): void {
    if (a[0] !== 'lj/cut') this.fail(`pair_style '${a[0] ?? ''}' is not supported; only lj/cut`);
    if (a.length !== 2) this.fail('usage: pair_style lj/cut cutoff');
    const rc = this.num(a[1], 'cutoff');
    if (!(rc > 0)) this.fail('cutoff must be > 0');
    this.pairCutoff = rc;
    this.table = this.state ? newPairTable(this.state.ntypes, rc) : null;
    this.invalidate();
  }

  /** pair_coeff I J epsilon sigma [cutoff]. */
  private cmdPairCoeff(a: string[]): void {
    this.needBox();
    if (!this.table || this.pairCutoff === null) this.fail('pair_coeff needs a pair_style first');
    if (a.length !== 4 && a.length !== 5) this.fail('usage: pair_coeff I J epsilon sigma [cutoff]');
    try {
      setPairCoeff(this.table, a[0], a[1], this.num(a[2], 'epsilon'), this.num(a[3], 'sigma'),
        a[4] !== undefined ? this.num(a[4], 'cutoff') : undefined);
    } catch (e) {
      if (e instanceof EngineError) throw e;
      this.fail((e as Error).message);
    }
    this.invalidate();
  }

  /** pair_modify shift yes|no / mix geometric|arithmetic|sixthpower. */
  private cmdPairModify(a: string[]): void {
    if (!this.table) this.fail('pair_modify needs a pair_style and a box first');
    if (a.length === 0 || a.length % 2) this.fail('usage: pair_modify keyword value ...');
    for (let k = 0; k < a.length; k += 2) {
      if (a[k] === 'shift') this.table.shift = this.yesNo(a[k + 1], 'shift');
      else if (a[k] === 'mix' && ['geometric', 'arithmetic', 'sixthpower'].includes(a[k + 1])) {
        this.table.mix = a[k + 1] as PairTable['mix'];
        // re-mix pairs that were not set explicitly
        for (let i = 0; i < this.table.pairs.length; i++) if (!this.table.explicit[i]) this.table.pairs[i] = undefined;
      } else this.fail(`pair_modify '${a[k]} ${a[k + 1]}' is not supported; supported: shift yes|no, mix geometric|arithmetic|sixthpower`);
    }
    this.invalidate();
  }

  /** timestep dt — defaults per units style (docs.lammps.org/units.html). */
  private cmdTimestep(a: string[]): void {
    const dt = this.num(a[0], 'timestep');
    if (!(dt > 0)) this.fail('timestep must be > 0');
    if (this.state) this.state.dt = dt;
    else this.pendingDt = dt;
  }
  private pendingDt: number | null = null;

  /**
   * fix ID all style args.
   *   nve; enforce2d;
   *   langevin Tstart Tstop damp seed (docs.lammps.org/fix_langevin.html);
   *   temp/berendsen Tstart Tstop Tdamp; temp/rescale N Tstart Tstop window fraction;
   *   nvt temp Tstart Tstop Tdamp (docs.lammps.org/fix_nh.html).
   */
  private cmdFix(a: string[]): void {
    const s = this.needBox();
    const [id, group, style] = a;
    if (!id || !style) this.fail('usage: fix ID group-ID style args');
    this.needAllGroup(group);
    const r = a.slice(3);
    const exact = (n: number, usage: string) => { if (r.length !== n) this.fail(`usage: fix ID all ${usage}`); };
    let spec: FixSpec;
    switch (style) {
      case 'nve': exact(0, 'nve'); spec = { style, id, group: 'all' }; break;
      case 'enforce2d':
        exact(0, 'enforce2d');
        if (s.dimension !== 2) this.fail('fix enforce2d needs dimension 2');
        spec = { style, id, group: 'all' };
        break;
      case 'langevin': {
        exact(4, 'langevin Tstart Tstop damp seed');
        const seed = this.int(r[3], 'seed');
        if (seed <= 0) this.fail('seed must be a positive integer');
        spec = { style, id, group: 'all', tStart: this.num(r[0], 'Tstart'), tStop: this.num(r[1], 'Tstop'), damp: this.num(r[2], 'damp'), seed };
        break;
      }
      case 'temp/berendsen':
        exact(3, 'temp/berendsen Tstart Tstop Tdamp');
        spec = { style, id, group: 'all', tStart: this.num(r[0], 'Tstart'), tStop: this.num(r[1], 'Tstop'), damp: this.num(r[2], 'Tdamp') };
        break;
      case 'temp/rescale':
        exact(5, 'temp/rescale N Tstart Tstop window fraction');
        spec = {
          style, id, group: 'all', every: this.int(r[0], 'N'), tStart: this.num(r[1], 'Tstart'), tStop: this.num(r[2], 'Tstop'),
          window: this.num(r[3], 'window'), fraction: this.num(r[4], 'fraction'),
        };
        break;
      case 'nvt':
        if (r[0] !== 'temp' || r.length !== 4) this.fail('usage: fix ID all nvt temp Tstart Tstop Tdamp (other keywords are not supported)');
        spec = { style, id, group: 'all', tStart: this.num(r[1], 'Tstart'), tStop: this.num(r[2], 'Tstop'), damp: this.num(r[3], 'Tdamp') };
        break;
      default:
        this.fail(`fix style '${style}' is not supported; supported: ${FIX_STYLES.join(', ')}`);
    }
    if ('damp' in spec && !(spec.damp > 0)) this.fail('the damping parameter must be > 0');
    let fix: Fix;
    try { fix = makeFix(spec); } catch (e) { this.fail((e as Error).message); }
    const i = this.fixes.findIndex((f) => f.spec.id === id);
    if (i >= 0) this.fixes[i] = { spec, fix }; else this.fixes.push({ spec, fix });
  }

  private cmdUnfix(a: string[]): void {
    const i = this.fixes.findIndex((f) => f.spec.id === a[0]);
    if (i < 0) this.fail(`no fix with ID '${a[0] ?? ''}'`);
    this.fixes.splice(i, 1);
  }

  /** thermo N — "thermo 0" (default) prints only the first and last step. */
  private cmdThermo(a: string[]): void {
    const n = this.int(a[0], 'thermo interval');
    if (n < 0) this.fail('thermo interval must be >= 0');
    this.thermoEvery = n;
  }

  /** thermo_style one | custom keywords... */
  private cmdThermoStyle(a: string[]): void {
    if (a[0] === 'one' && a.length === 1) { this.thermoKeywords = THERMO_ONE; return; }
    if (a[0] !== 'custom') this.fail(`thermo_style '${a[0] ?? ''}' is not supported; use one or custom`);
    if (a.length < 2) this.fail('thermo_style custom needs at least one keyword');
    const bad = a.slice(1).filter((k) => !(THERMO_KEYWORDS as readonly string[]).includes(k));
    if (bad.length) this.fail(`unsupported thermo keyword(s): ${bad.join(', ')}; supported: ${THERMO_KEYWORDS.join(' ')}`);
    this.thermoKeywords = a.slice(1) as ThermoKeyword[];
  }

  /** thermo_modify norm yes|no — "norm = yes for unit style of lj, norm = no for ... real and metal". */
  private cmdThermoModify(a: string[]): void {
    if (a.length !== 2 || a[0] !== 'norm') this.fail("only 'thermo_modify norm yes|no' is supported");
    this.norm = this.yesNo(a[1], 'norm');
  }

  /** reset_timestep N */
  private cmdResetTimestep(a: string[]): void {
    const s = this.needBox();
    const n = this.int(a[0], 'timestep');
    if (n < 0) this.fail('timestep must be >= 0');
    s.step = n;
  }

  /** clear — keeps "input script variables" (docs.lammps.org/clear.html). */
  private cmdClear(a: string[]): void {
    if (a.length) this.fail('clear takes no arguments');
    const vars = this.variables;
    Object.assign(this, new Session(this.io, this.backend, this.frameEvery));
    this.variables = vars;
    this.log('Cleared: all atoms, settings and fixes reset (variables kept)');
  }

  /**
   * variable name equal formula | string text | index text... | delete.
   * docs.lammps.org/variable.html: "When a variable command is encountered
   * in the input script and the variable name has already been specified,
   * the command is ignored" except that "string, ..., equal ... ARE
   * redefined each time the command is encountered."
   */
  private cmdVariable(a: string[]): void {
    const [name, style] = a;
    if (!name || !style) this.fail('usage: variable name style args');
    if (!/^[A-Za-z0-9_]+$/.test(name)) this.fail(`invalid variable name '${name}'`);
    if (style === 'delete') { this.variables.delete(name); return; }
    if (style === 'equal' || style === 'string') {
      if (a.length !== 3) this.fail(`variable ${style} takes exactly one argument; quote a formula or text that contains spaces`);
      this.variables.set(name, { style, value: a[2] });
      return;
    }
    if (style === 'index') {
      if (a.length < 3) this.fail('variable index needs at least one value');
      if (!this.variables.has(name)) this.variables.set(name, { style, value: a[2] });
      return;
    }
    this.fail(`variable style '${style}' is not supported; supported: equal, string, index, delete`);
  }

  /**
   * print text [file f | append f | screen yes|no] — "If the text string
   * contains variables, they will be evaluated and their current values
   * printed" (docs.lammps.org/print.html).
   */
  private cmdPrint(a: string[]): void {
    if (a.length === 0) this.fail('usage: print "text"');
    const text = substituteVariables(a[0], (n) => this.variableText(n), (f, fmt) => this.immediate(f, fmt));
    const kw = this.keywords(a.slice(1), ['file', 'append', 'screen', 'universe']);
    if (kw.screen === undefined || this.yesNo(kw.screen, 'screen')) this.log(text);
    if (kw.file) this.io.writeFile?.(kw.file, text + '\n', false);
    if (kw.append) this.io.writeFile?.(kw.append, text + '\n', true);
  }

  /**
   * dump ID all atom|custom N file [columns] — docs.lammps.org/dump.html:
   * atom writes "id type xs ys zs"; "Dumps are performed on timesteps that
   * are a multiple of N (including timestep 0)"; "If a '*' character
   * appears in the filename, then one file per snapshot is written and the
   * '*' character is replaced with the timestep value."
   */
  private cmdDump(a: string[]): void {
    this.needBox();
    const [id, group, style, every, file, ...cols] = a;
    if (!id || !style || !every || !file) this.fail('usage: dump ID all atom|custom N file [columns]');
    this.needAllGroup(group);
    if (this.dumps.some((d) => d.id === id)) this.fail(`dump ID '${id}' already exists (undump it first)`);
    const n = this.int(every, 'dump interval');
    if (n < 1) this.fail('dump interval must be >= 1');
    let columns: string[];
    if (style === 'atom') {
      if (cols.length) this.fail('dump atom takes no column list (use dump custom)');
      columns = ['id', 'type', 'xs', 'ys', 'zs'];
    } else if (style === 'custom') {
      if (!cols.length) this.fail('dump custom needs a column list, e.g. id type x y z');
      const bad = cols.filter((c) => !DUMP_COLUMNS.includes(c));
      if (bad.length) this.fail(`unsupported dump column(s): ${bad.join(', ')}; supported: ${DUMP_COLUMNS.join(' ')}`);
      columns = cols;
    } else {
      this.fail(`dump style '${style}' is not supported; supported: atom, custom`);
    }
    this.dumps.push({ id, every: n, file, style, columns, lastStep: -1 });
  }

  private cmdUndump(a: string[]): void {
    const i = this.dumps.findIndex((d) => d.id === a[0]);
    if (i < 0) this.fail(`no dump with ID '${a[0] ?? ''}'`);
    this.dumps.splice(i, 1);
  }

  private writeDumps(): void {
    const s = this.state!;
    for (const d of this.dumps) {
      if (d.lastStep === s.step || s.step % d.every !== 0) continue;
      d.lastStep = s.step;
      const multi = d.file.includes('*');
      const name = multi ? d.file.replace('*', String(s.step)) : d.file;
      this.io.writeFile?.(name, dumpSnapshot(s, d.columns), !multi);
    }
  }

  /** write_data file — header, Masses, Pair Coeffs (ii), Atoms # atomic with image flags, Velocities. */
  private cmdWriteData(a: string[]): void {
    const s = this.needBox();
    const [file, ...rest] = a;
    if (!file) this.fail('usage: write_data file [nocoeff]');
    const kw = rest.filter((w) => w !== 'nocoeff');
    if (kw.length) this.fail(`write_data keywords ${kw.join(' ')} are not supported; only nocoeff`);
    this.io.writeFile?.(file, dataFile(s, rest.includes('nocoeff') ? null : this.table), false);
    this.log(`Wrote ${s.n} atoms to ${file}`);
  }

  /**
   * run N [upto] — docs.lammps.org/run.html: "A value of N = 0 is
   * acceptable; only the thermodynamics of the system are computed and
   * printed"; upto: "perform a run starting at the current timestep up to
   * the specified timestep".
   */
  private async cmdRun(a: string[]): Promise<void> {
    const s = this.needBox();
    let n = this.int(a[0], 'number of steps');
    if (a.length > 2 || (a.length === 2 && a[1] !== 'upto')) this.fail("run keywords other than 'upto' are not supported");
    if (a[1] === 'upto') n -= s.step;
    if (n < 0) this.fail('run needs N >= 0 (with upto: a timestep at or after the current one)');
    this.requireMasses(s);
    const table = this.tableForRun(true);
    if (this.pendingDt !== null) { s.dt = this.pendingDt; this.pendingDt = null; }
    const integrators = this.fixes.filter((f) => f.fix.integrates);
    if (integrators.length > 1) {
      this.fail(`fixes ${integrators.map((f) => f.spec.id).join(' and ')} both integrate the same atoms; keep one of nve / nvt`);
    }
    if (integrators.length === 0 && n > 0) this.log('WARNING: no time-integration fix (nve or nvt) is defined, so atoms will not move');
    const keywords = this.thermoKeywords;
    this.io.emit({ kind: 'thermo-header', keywords });
    const t0 = performance.now();
    const startStep = s.step;
    this.forces = await runSteps(s, table, this.backend, this.fixes.map((f) => f.fix), n, {
      thermoEvery: this.thermoEvery,
      keywords,
      norm: this.norm ?? undefined,
      onThermo: (row) => this.io.emit({ kind: 'thermo', row }),
      onStep: () => {
        this.writeDumps();
        const dumped = this.dumps.some((d) => d.lastStep === s.step);
        if (dumped || (this.frameEvery > 0 && s.step % this.frameEvery === 0)) this.emitFrame();
        return !this.cancelled;
      },
      onSetup: () => this.writeDumps(),
      yieldEvery: 25,
    });
    const seconds = (performance.now() - t0) / 1000;
    this.emitFrame();
    this.log(`Loop time of ${seconds.toFixed(3)} s for ${s.step - startStep} steps with ${s.n} atoms (${this.backend.label})`);
    this.io.emit({ kind: 'done', steps: s.step - startStep, seconds, backend: this.backend.label });
    if (this.cancelled) throw new RunCancelled();
  }

  private emitFrame(): void {
    const s = this.state;
    if (!s) return;
    this.io.emit({
      kind: 'frame', step: s.step, x: Float64Array.from(s.x), image: Int32Array.from(s.image),
      type: Int32Array.from(s.type), id: Int32Array.from(s.id),
      box: { lo: [...s.box.lo], hi: [...s.box.hi], periodic: [...s.box.periodic] },
    });
  }
}

const fmtFloat = (v: number) => formatNumber(v, '%.8g');

/** One snapshot in the LAMMPS dump text format. */
export const dumpSnapshot = (s: SimState, columns: string[]): string => {
  const L = [0, 1, 2].map((d) => s.box.hi[d] - s.box.lo[d]);
  const lines = [
    'ITEM: TIMESTEP', String(s.step),
    'ITEM: NUMBER OF ATOMS', String(s.n),
    'ITEM: BOX BOUNDS pp pp pp',
    ...[0, 1, 2].map((d) => `${s.box.lo[d].toExponential(16)} ${s.box.hi[d].toExponential(16)}`),
    `ITEM: ATOMS ${columns.join(' ')}`,
  ];
  const value = (i: number, c: string): string => {
    const d = 'xyz'.indexOf(c[0]);   // x, xs, xu
    switch (c) {
      case 'id': return String(s.id[i]);
      case 'type': return String(s.type[i]);
      case 'mass': return fmtFloat(s.massByType[s.type[i]]);
      case 'x': case 'y': case 'z': return fmtFloat(s.x[3 * i + d]);
      case 'xs': case 'ys': case 'zs': return fmtFloat((s.x[3 * i + d] - s.box.lo[d]) / L[d]);
      case 'xu': case 'yu': case 'zu': return fmtFloat(s.x[3 * i + d] + s.image[3 * i + d] * L[d]);
      case 'ix': case 'iy': case 'iz': return String(s.image[3 * i + 'xyz'.indexOf(c[1])]);
      case 'vx': case 'vy': case 'vz': return fmtFloat(s.v[3 * i + 'xyz'.indexOf(c[1])]);
      case 'fx': case 'fy': case 'fz': return fmtFloat(s.f[3 * i + 'xyz'.indexOf(c[1])]);
      default: return '0';
    }
  };
  for (let i = 0; i < s.n; i++) lines.push(columns.map((c) => value(i, c)).join(' '));
  return lines.join('\n') + '\n';
};

/** A LAMMPS data file (atom_style atomic) for the current state. */
export const dataFile = (s: SimState, table: PairTable | null): string => {
  const out: string[] = [
    `LAMMPS data file written by the LAMMPS Web GUI notebook engine, timestep = ${s.step}, units = ${s.units.style}`,
    '',
    `${s.n} atoms`,
    `${s.ntypes} atom types`,
    '',
    ...[0, 1, 2].map((d) => `${s.box.lo[d].toExponential(16)} ${s.box.hi[d].toExponential(16)} ${'xyz'[d]}lo ${'xyz'[d]}hi`),
    '',
    'Masses',
    '',
  ];
  for (let t = 1; t <= s.ntypes; t++) out.push(`${t} ${fmtFloat(s.massByType[t])}`);
  if (table) {
    const diag = Array.from({ length: s.ntypes }, (_, k) => table.pairs[(k + 1) * (table.ntypes + 1) + k + 1]);
    if (diag.every(Boolean)) {
      out.push('', 'Pair Coeffs # lj/cut', '');
      diag.forEach((p, k) => out.push(`${k + 1} ${fmtFloat(p!.epsilon)} ${fmtFloat(p!.sigma)}`));
    }
  }
  out.push('', 'Atoms # atomic', '');
  for (let i = 0; i < s.n; i++) {
    out.push(`${s.id[i]} ${s.type[i]} ${s.x[3 * i].toPrecision(16)} ${s.x[3 * i + 1].toPrecision(16)} ${s.x[3 * i + 2].toPrecision(16)} ${s.image[3 * i]} ${s.image[3 * i + 1]} ${s.image[3 * i + 2]}`);
  }
  out.push('', 'Velocities', '');
  for (let i = 0; i < s.n; i++) {
    out.push(`${s.id[i]} ${s.v[3 * i].toPrecision(16)} ${s.v[3 * i + 1].toPrecision(16)} ${s.v[3 * i + 2].toPrecision(16)}`);
  }
  return out.join('\n') + '\n';
};
