import type { RespaSettings } from './run/respa';
import type { AtomStyle, EngineEvent, SimState, UnitSystem } from './types';
import type { MoleculeTemplate } from './molecule';
import { UNIT_SYSTEMS } from './units';
import { Geometry, shrinkWrap } from './domain';
import { Neighbor } from './neighbor';
import { ForceField, type ComputeFlags } from './force/forcefield';
import { gunzip } from './gunzip';
import { StyleError, type Accum, type FileOpener, type StyleContext } from './force/types';
import { Groups } from './group';
import type { Region, RegionEnv } from './region';
import type { Compute } from './compute/compute';
import type { Fix } from './fix/fix';
import { Variables } from './variables';
import { ATOM_VECTORS, FormulaError, errorUrl, evaluateScalar, resolveIndex, type FormulaEnv, type Index, type Mode, type Value } from './formula';
import { substituteVariables } from './script';
import { formatNumber } from './cformat';
import type { Lattice } from './lattice';
import { RanMars } from './rng';
import { Thermo } from './output/thermo';
import type { Dump } from './output/dump';
import type { Handler } from './commands/args';
import { buildAtomMap, deleteAtoms as deleteAtomsImpl, massOf, nativeOrder, hasMolecule, hasCharge, isMolecularStyle } from './atoms';
import { groupFunction } from './groupfn';
import { LabelMap, typeWord } from './labelmap';
import { defaultMinSettings, type MinSettings } from './run/min';
import { normalizeFileName, resolveFileName } from './fileName';
import { gplHook } from './gplHooks';

/** A lazily loaded pair-style table of src/engine-gpl (null when not loaded: the style's own checks apply). */
const pairTable = (word: string, style: string): string | null => {
  try {
    return gplHook<(s: string) => string | null>(word)(style);
  } catch {
    return null;
  }
};

/** A pair style's native init_one message for an unset I,J pair (the lazily loaded table; null when not loaded). */
const pairUnsetText = (style: string): string | null => pairTable('pair_unset', style);

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
  /** The step whose energy was last tallied (native update->eflag_global): a thermo output's step; -1 before any. */
  eflagStep: number;
}

/**
 * The generator of random() and normal() in formulas: native's RanMars (Variable::math_function at b332f28 makes one
 * per variable store), whose gaussian() returns the first of a Marsaglia polar pair and keeps the second for the next
 * call (RanMars::gaussian, random_mars.cpp at b332f28).
 */
class FormulaRandom {
  private readonly mars: RanMars;
  private second: number | null = null;
  constructor(seed: number) {
    this.mars = new RanMars(seed);
    // native's RanMars constructor ends with one uniform() (random_mars.cpp at b332f28)
    this.mars.uniform();
  }
  uniform(): number { return this.mars.uniform(); }
  gaussian(): number {
    if (this.second !== null) {
      const g = this.second;
      this.second = null;
      return g;
    }
    const [first, second] = this.mars.polarPair();
    this.second = second;
    return first;
  }
}

export class System {
  /** Frames carry the velocities and masses of the atoms (EngineEvent 'frame', v and mass). */
  frameVelocities = false;
  /** The per-atom masses last sent with a frame; a frame repeats them only when they change (or at a run's start). */
  frameMassSent: Float64Array | null = null;
  // ---- settings that precede the box
  units: UnitSystem = UNIT_SYSTEMS.lj;
  dimension: 2 | 3 = 3;
  /**
   * extra/bond/per/atom (create_box, read_data): the bonds per atom the storage holds. A read_data after the box
   * exists may not bring more bonds on an atom than this (create_box.html and read_data.html describe the keyword;
   * the rule is measured with native LAMMPS, see data.ts).
   */
  extraBondPerAtom = 0;
  /**
   * The amounts a create_box expanded its shrink-wrapped faces by (SMALL = 1e-4 of the box length, native
   * set_initial_box); null once the box has been reset (pbc), or for a box read from a file. See create_atoms.
   */
  initSmall: [number, number, number] | null = null;
  boundary: SimState['box']['boundary'] = [['p', 'p'], ['p', 'p'], ['p', 'p']];
  atomStyle: AtomStyle = 'atomic';
  /**
   * atom_style sphere 1 or bpm/sphere 1 (alone or in hybrid): dynamic particle radii. atom_style.html: "*sphere* arg =
   * 0/1 (optional) for static/dynamic particle radii". As native AtomVecSphere::init does (src/atom_vec_sphere.cpp at
   * b332f28), fix adapt and fix adapt/fep that change diameters need it.
   */
  radvary = false;
  lattice: Lattice | null = null;
  /** timestep given before the box exists. */
  pendingDt: number | null = null;

  private _state: SimState | null = null;
  geom!: Geometry;
  readonly nb: Neighbor;
  readonly ff = new ForceField();
  readonly groups = new Groups();
  /** type labels (labelmap.ts): the labelmap command and the Type Labels sections of read_data */
  labels = new LabelMap();
  readonly regions = new Map<string, Region>();
  readonly computes: Compute[] = [];
  readonly fixes: Fix[] = [];
  /**
   * The systems of all partitions (replica 0 first), the same list for every replica. A single system without
   * partitions. Set by the session; a fix coupled to the same fix on the other replicas (fix pimd) reads them.
   */
  replicaPeers: () => readonly System[] = () => [this];
  /** run_style respa settings (run/respa.ts); null for run_style verlet. */
  respa: RespaSettings | null = null;
  /** run_style verlet/split or verlet/split/rk (two partitions: Rspace and Kspace); null for the other run styles. */
  splitRun: 'verlet/split' | 'verlet/split/rk' | null = null;
  readonly vars: Variables;
  readonly thermo: Thermo;
  readonly dumps: Dump[] = [];
  /** min_style and min_modify settings. */
  minStyle = 'cg';
  minSettings: MinSettings = defaultMinSettings();
  /** Style names per category for is_available(): command, compute, fix, pair_style, ... */
  registries: Record<string, string[]> = {};
  /** The input command handlers (commands/index.ts), set by the session; fix set runs the set command's handler. */
  commands: Readonly<Record<string, Handler>> = {};
  /** Files the session can read: uploads plus everything it wrote. */
  readonly files = new Map<string, string>();
  /** shell cd: the working directory, relative to the notebook root (''); see shellFs.ts. */
  cwd = '';
  /** shell mkdir: directories made by the shell (root-relative names); see shellFs.ts. */
  readonly dirs = new Set<string>();
  /** shell putenv: environment variables; LAMMPS_POTENTIALS is the directory a potential file is looked up in. */
  readonly env = new Map<string, string>();
  /** molecule templates by ID (each a list of sets; create_atoms uses the first). */
  readonly molecules = new Map<string, MoleculeTemplate[]>();
  run: RunInfo = { inRun: false, firstStep: 0, lastStep: 0, beginStep: 0, endStep: 0, t0: 0, ranOnce: false, eflagStep: -1 };

  epoch = 0;
  stateVersion = 0;
  private forcesVersion = -1;
  private flagsUsed: ComputeFlags = {};
  private rngEqual: FormulaRandom | null = null;
  private rngAtom: FormulaRandom | null = null;

  constructor(readonly io: SessionIO) {
    this.nb = new Neighbor(this.units.skin);
    this.vars = new Variables((name) => this.readFile(name), (text) => this.warn(text), (name) => this.env.get(name));
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
    this.initSmall = null;
    this.geom = new Geometry(s.box);
    if (this.pendingDt !== null) { s.dt = this.pendingDt; this.pendingDt = null; }
    // a new state with the same number of types keeps the pair coefficients (measured with native LAMMPS, black
    // box: replicate after pair_coeff runs with the same coefficients); a different count needs new tables
    if (this.ff.pair && this.ff.pair.ntypes !== s.ntypes) this.ff.pair.allocate(s.ntypes);
    // fix property/atom defined before the box adds its per-atom properties to the new state
    for (const f of this.fixes) (f as { attachState?: (st: SimState) => void }).attachState?.(s);
    this.bump();
  }

  /** Drops everything (clear). */
  reset(): void {
    this._state = null;
    this.labels = new LabelMap();
  }

  /** Positions, atoms, box or force field changed. */
  bump(): void {
    this.stateVersion++;
    this.epoch++;
  }

  /**
   * Compute values must be re-evaluated (velocities changed, thermo output). Rule for readers (fixes that read a
   * compute between the integrator steps): call refreshComputes() before reading. The epoch is only bumped by
   * bump() after positions or the box change, and by this call; a value read after finalIntegrate (velocities
   * changed) and before endOfStep is stale unless refreshed. All readers in fix/ do refresh first (thermostats
   * current(), nh currentTemp(), valueOf() in fix/util.ts, box_relax).
   */
  refreshComputes(): void {
    this.epoch++;
  }

  /**
   * A command that initializes the whole system as a run does (native lmp->init(): write_data, write_restart,
   * create_bonds, delete_bonds, delete_atoms overlap, set of topology types, balance, reset_atoms) leaves every compute
   * initialized, so variable formulas may read them afterwards (Compute::init_flags in Modify::init, at b332f28).
   */
  markComputesInitialized(): void {
    for (const c of this.computes) c.initialized = true;
  }

  /** True when the stored forces belong to the current state. */
  get hasForces(): boolean { return this.forcesVersion === this.stateVersion; }

  /** Forces were computed for the current state by the run loop. */
  forcesCurrent(flags?: ComputeFlags): void {
    this.forcesVersion = this.stateVersion;
    // the per-atom terms an integrator assembled itself (run_style respa) are those of these flags
    if (flags) this.flagsUsed = flags;
  }

  /** comm_modify vel yes|no (comm_modify.html: ghost atoms store velocity info). */
  ghostVelocity = false;
  /** newton command, pair setting (newton.html default on); the engine's results do not depend on it. */
  newtonPair = true;
  /** newton command, bond setting (newton.html: "flag2 = on or off for bonded interactions"; default on). */
  newtonBond = true;
  /** comm_style brick|tiled (only fix balance rcb cares in a one-process engine). */
  commStyle: 'brick' | 'tiled' = 'brick';

  styleContext(): StyleContext {
    const freeze = this.fixes.find((f) => f.style === 'freeze');
    return { s: this._state, readFile: (n, o) => this.readFile(n, o), log: (t) => this.log(t), ghostVelocity: this.ghostVelocity, freezeGroupBit: freeze ? freeze.groupBit : 0, newtonPair: this.newtonPair, equalVariable: (n: string) => this.equalVariable(n), atomTypeWord: (w: string) => typeWord(this.labels, 'atom', w), typeWord: (kind, w) => typeWord(this.labels, kind, w), unitStyle: this.units.style, pairUnsetText,
      pairNeedsQ: (style: string) => (this._state && !hasCharge(this._state) ? pairTable('pair_q', style) : null), xlattice: this.lattice ? this.lattice.spacing[0] : 1, addFix: (make) => this.addStyleFix(make), fixOf: (st) => this.fixes.find((f) => f.style === st) ?? null, fixById: (id) => this.fixes.find((f) => f.id === id) ?? null, initialBoxExpansion: this.initSmall, runSteps: () => ({ begin: this.run.beginStep, end: this.run.endStep }) };
  }

  /** A fix a style adds at its own command time (StyleContext.addFix): replaces a fix with the same ID. */
  private addStyleFix(make: (sys: System) => Fix): void {
    const f = make(this);
    const k = this.fixes.findIndex((x) => x.id === f.id);
    if (k >= 0) this.fixes[k] = f; else this.fixes.push(f);
  }

  /** log file: a copy of the log text goes to this file (log.html). */
  private logFile: string | null = null;

  setLogFile(name: string | null, append: boolean): void {
    this.logFile = name;
    if (name && !append) this.writeFile(name, '', false);
  }

  /** A log file is open (thermo tables are formatted only when one is). */
  get logOpen(): boolean {
    return this.logFile !== null;
  }

  log(text: string): void {
    this.io.emit({ kind: 'log', text });
    if (this.logFile) this.writeFile(this.logFile, text + '\n', true);
  }

  /** Thermo table lines go to the log file only: the screen gets the rows as thermo events. */
  logTable(lines: string[]): void {
    if (this.logFile && lines.length) this.writeFile(this.logFile, lines.join('\n') + '\n', true);
  }

  /** thermo_modify warn (thermo.ts): ignore suppresses every warning; past the limit one final line is printed. */
  warnOff = false;
  warnLimit = 100;
  warnCount = 0;

  warn(text: string): void {
    if (this.warnOff) return;
    this.warnCount++;
    if (this.warnCount <= this.warnLimit) { this.log(`WARNING: ${text}`); return; }
    if (this.warnCount === this.warnLimit + 1) {
      this.log(`WARNING: Too many warnings: ${this.warnCount} vs ${this.warnLimit}. All future warnings will be suppressed`);
    }
  }

  /** The root-relative session-file name of a name given in the input, from the working directory (shell cd). */
  resolveFile(name: string): string {
    return this.cwd === '' ? normalizeFileName(name) : resolveFileName(this.cwd, name);
  }

  /**
   * A file of the notebook. A missing file stops with native's message (`opener` is the caller's wording, as "Cannot
   * open restart file"; utils::getsyserror gives "No such file or directory"), then the engine's hint on a second line.
   */
  readFile(name: string, opener: FileOpener = 'Cannot open file'): string {
    let f = this.files.get(this.resolveFile(name));
    // As native LAMMPS does (src/utils.cpp get_potential_file_path and src/platform.cpp list_pathenv at b332f28): a file
    // not found where its name points is looked up, by its base name, in each directory of the colon-separated
    // LAMMPS_POTENTIALS environment variable (set by shell putenv), in order
    const potDirs = this.env.get('LAMMPS_POTENTIALS');
    if (f === undefined && potDirs !== undefined) {
      const base = name.slice(name.lastIndexOf('/') + 1);
      for (const dir of potDirs.split(':')) {
        f = this.files.get(resolveFileName('', `${dir}/${base}`));
        if (f !== undefined) break;
      }
    }
    if (f === undefined) {
      const first = typeof opener === 'string' ? `${opener} ${name}: No such file or directory` : opener(name, 'No such file or directory');
      throw new StyleError(`${first}\nThe notebook has no file ${name}: add it to the notebook's files (upload) or write it earlier in the session`);
    }
    // A gzipped file arrives one character per byte (the notebook reads .gz uploads as bytes); read_data
    // takes a file that "can be ASCII text or a gzipped text file (detected by a .gz suffix)" (docs.lammps.org/read_data.html).
    if (name.endsWith('.gz') && f.charCodeAt(0) === 0x1f && f.charCodeAt(1) === 0x8b) {
      const bytes = new Uint8Array(f.length);
      for (let k = 0; k < f.length; k++) bytes[k] = f.charCodeAt(k) & 0xff;
      return new TextDecoder().decode(gunzip(bytes));
    }
    return f;
  }

  writeFile(name: string, text: string, append: boolean): void {
    this.writeFileAt(this.resolveFile(name), text, append);
  }

  /**
   * Writes a session file under its root-relative name, already resolved from the working directory when it was
   * defined (a dump opens its file at the dump command, as native LAMMPS does, so a later shell cd does not move it).
   */
  writeFileAt(key: string, text: string, append: boolean): void {
    this.files.set(key, (append ? this.files.get(key) ?? '' : '') + text);
    this.io.writeFile?.(key, text, append);
  }

  // ---------------------------------------------------------------- forces

  /** stateVersion at which a run's init last initialized the pair style (initPair), or -1. */
  private pairInitVersion = -1;

  /** The pair part of native's force->init, at the start of a run's init (ForceField.initPair). */
  initPair(): void {
    this.ff.initPair(this.state, this.geom, this.styleContext());
    this.pairInitVersion = this.stateVersion;
  }

  /** What per-atom force-field data the defined computes need. */
  computeFlags(): ComputeFlags {
    let eatom = false, vatom = false, cvatom = false;
    for (const c of this.computes) { eatom ||= c.needsEatom; vatom ||= c.needsVatom; cvatom ||= c.needsCvatom; }
    return cvatom ? { eatom, vatom, cvatom } : { eatom, vatom };
  }

  /**
   * Prepares the force field, remaps atoms into periodic boxes, re-fits
   * shrink-wrapped faces, checks for lost atoms and builds ghosts and
   * neighbor lists (the setup half of a run, Developer_flow.html).
   */
  setupNeighbors(): void {
    const s = this.state;
    // the run's init has initialized the pair style (initPair) and nothing has changed since
    const pairDone = this.pairInitVersion === this.stateVersion && !this.ff.pairStale;
    this.pairInitVersion = -1;
    this.ff.init(s, this.nb, this.geom, this.styleContext(), pairDone);
    this.pbc();
    this.nb.build(s, this.geom, s.step);
    for (const f of this.fixes) f.afterForceInit?.();
    this.checkDynamicWithin();
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

  /**
   * A box flip (fix deform, and fix nh and its variants with flip yes): the tilts become xy, xz, yz (tilt), with the
   * box vectors added given by m (xy), n (xz) and p (yz), each +1, -1 or 0. As native LAMMPS does (src/domain.cpp
   * image_flip and remap, src/fix_deform.cpp pre_exchange at b332f28): the new edge vectors are B' = B + m A and
   * C' = C + p B + n A, so the image counts that describe the same unwrapped positions change as ybox -= p zbox,
   * then xbox -= m ybox + n zbox (with the new ybox); every atom is then mapped into the new box (Domain::remap,
   * velocities untouched) and taken to fractional coordinates and back (native's x2lamda and lamda2x around the
   * migration of atoms). The neighbor lists must be rebuilt on this step (the caller's reneighborDue).
   */
  flipBox(tilt: readonly number[], m: number, n: number, p: number, from?: Fix): void {
    const s = this.state;
    const g = this.geom;
    const b = s.box;
    for (let d = 0; d < 3; d++) b.tilt[d] = tilt[d];
    g.update();
    const img = s.image;
    for (let i = 0; i < s.n; i++) {
      const zbox = img[3 * i + 2];
      const ybox = img[3 * i + 1] - p * zbox;
      img[3 * i + 1] = ybox;
      img[3 * i] -= m * ybox + n * zbox;
    }
    const tmp = [0, 0, 0];
    for (let i = 0; i < s.n; i++) g.remap(s.x, s.image, i);
    for (let i = 0; i < s.n; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      g.fromLamda(tmp[0], tmp[1], tmp[2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
    }
    for (const f of this.fixes) if (f !== from) f.boxChanged?.();
    this.bump();
  }

  /**
   * As native LAMMPS does (src/output.cpp Output::reset_timestep at b332f28): outside a run the timestep cannot be reset
   * while a single-file dump that has already written a snapshot is defined ("Cannot reset timestep with active dump -
   * must undump first").
   */
  checkTimestepReset(): void {
    if (this.run.inRun) return;
    if (this.dumps.some((d) => d.lastStep >= 0 && !d.file.includes('*'))) {
      throw new StyleError('Cannot reset timestep with active dump - must undump first');
    }
  }

  pbc(): void {
    const s = this.state;
    const g = this.geom;
    // the first reset of the box (domain->pbc and reset_box at setup) replaces the create_box expansion
    this.initSmall = null;
    for (let i = 0; i < s.n; i++) g.remap(s.x, s.image, i);
    if (shrinkWrap(s, g)) this.nb.forwardComm(s, g);
    this.checkLost();
  }

  /** Atoms outside a fixed (f) boundary: "it will be deleted on the next timestep that reneighboring occurs" (boundary.html). */
  private checkLost(): void {
    const lost = this.lostFlags();
    if (!lost) return;
    const mode = this.thermo.lost;
    // Measured with native LAMMPS (black box, boundary f f f, 256 lj atoms, run 10): "ERROR: Lost atoms: original 256
    // current 215" after the step whose reneighboring lost 41 atoms; the same step is the first one that fails here.
    if (mode === 'error') throw new StyleError(`Lost atoms: original ${this.state.n} current ${this.state.n - lost.nlost}`);
    if (mode === 'warn') this.warn(`lost ${lost.nlost} atom(s) through a fixed boundary`);
    this.deleteAtoms(lost.flags);
  }

  /**
   * change_box: atoms left outside a fixed boundary follow the thermo_modify lost setting.
   * change_box.html: "Atoms that end up outside of non-periodic boundaries as a result of this
   * command are now handled according to the thermo_modify lost setting". Measured with native
   * LAMMPS (black box, boundary f p p, 20 atoms, change_box all x final 0 5 without remap): the
   * default error stops the command with Lost atoms via change_box: original 20 current 9; warn
   * prints the same text as a WARNING and deletes the atoms; ignore deletes them silently; with
   * remap no atom is lost.
   */
  checkLostBoxChange(): void {
    const lost = this.lostFlags();
    if (!lost) return;
    const s = this.state;
    const mode = this.thermo.lost;
    const text = `Lost atoms via change_box: original ${s.n} current ${s.n - lost.nlost}`;
    if (mode === 'error') throw new StyleError(text);
    if (mode === 'warn') this.warn(text);
    this.deleteAtoms(lost.flags);
  }

  /** Flags the atoms outside a fixed (f) boundary; null when none is lost. */
  private lostFlags(): { flags: Uint8Array; nlost: number } | null {
    const s = this.state;
    const g = this.geom;
    const fixed = [0, 1, 2].filter((d) => s.box.boundary[d][0] === 'f' || s.box.boundary[d][1] === 'f');
    if (!fixed.length) return null;
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
    return nlost ? { flags: lost, nlost } : null;
  }

  /** Deletes flagged atoms and keeps every dependent structure consistent. */
  deleteAtoms(flags: Uint8Array): number {
    const n = deleteAtomsImpl(this.state, flags);
    if (n) this.atomsChanged();
    return n;
  }

  /** Atoms were added, deleted or reordered. */
  atomsChanged(keepSpecial = false): void {
    this.ff.atomsChanged();
    this.ff.topologyChanged(this.state, this.nb, keepSpecial);
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
    if ((want.eatom && !this.flagsUsed.eatom) || (want.vatom && !this.flagsUsed.vatom) || (want.cvatom && !this.flagsUsed.cvatom)) this.forcesVersion = -1;
    this.forces();
    return { eatom: this.ff.eatom, vatom: this.ff.vatom };
  }

  /** Force evaluation inside the run loop (lists already current). */
  computeForcesInRun(flags: ComputeFlags): Accum {
    this.flagsUsed = flags;
    return this.ff.compute(this.state, this.nb, this.geom, flags);
  }

  /**
   * The fix energies frozen at the last energy evaluation of a minimizer whose spins move after that evaluation
   * (min_style spin/lbfgs). As native LAMMPS does (src/SPIN/fix_precession_spin.cpp at b332f28, compute_scalar
   * returns eprec, summed in post_force): a thermo row shows the fix energies of the force evaluation, not of the
   * spins after the update. null outside such a minimizer.
   */
  fixEnergyFrozen: Map<{ energy(): number }, number> | null = null;

  /** Energy of one fix: the value frozen by freezeFixEnergies(), else the live value. */
  fixEnergyOf(f: { energy(): number }): number {
    const frozen = this.fixEnergyFrozen?.get(f);
    return frozen ?? f.energy();
  }

  /** Freezes every fix energy at the current spins (see fixEnergyFrozen). */
  freezeFixEnergies(): void {
    const m = new Map<{ energy(): number }, number>();
    for (const f of this.fixes) m.set(f, f.energy());
    this.fixEnergyFrozen = m;
  }

  fixEnergy(): number {
    let e = 0;
    for (const f of this.fixes) if (f.thermoEnergy) e += this.fixEnergyOf(f);
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

  /** The compute with this ID; a missing one is an error with `missing` (the caller's native message) or a generic one. */
  compute(id: string, missing?: string): Compute {
    const c = this.computes.find((x) => x.id === id);
    if (!c) throw new StyleError(missing ?? `compute ID '${id}' does not exist`);
    return c;
  }

  /** The fix with this ID; a missing one is an error with `missing` (the caller's native message) or a generic one. */
  fix(id: string, missing?: string): Fix {
    const f = this.fixes.find((x) => x.id === id);
    if (!f) throw new StyleError(missing ?? `fix ID '${id}' does not exist`);
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
      remap: (p) => { if (this.hasBox) this.geom.remapPoint(p); },
    };
  }

  /** Per-atom group mask bit by name. */
  groupBit(name: string): number { return this.groups.bit(name); }

  /** Fix records from read_restart, by fix ID, until a fix with the same ID and style is re-specified (fix/restartRecord.ts). */
  pendingFixData = new Map<string, import('./fix/restartRecord').FixRecord>();

  /** atom_modify map was given (atomic systems have no map by default). */
  atomMapSet = false;
  /** atom_modify first group-ID (null = all); neigh_modify include and comm_modify group must name the same group. */
  firstGroup: string | null = null;

  /** restart N: periodic restart files, one entry per mode (single file name, or two toggled names). */
  restartOut: { mode: 'single' | 'toggle'; every: number; everyVar: string | null; files: string[]; flip: number; next: number }[] = [];

  /** atom_modify sort Nfreq binsize (atom_modify.html: "By default, sorting is enabled with a frequency of 1000 and a binsize of 0.0"). */
  sortEvery = 1000;
  sortBinsize = 0;
  private nextSort = 0;

  /**
   * Native LAMMPS's spatial sort of its atom list, applied to SimState.order (the engine keeps its
   * own arrays). atom_modify.html: "Reordering is performed every *Nfreq* timesteps during a
   * dynamics run or iterations during a minimization.  More precisely, reordering occurs at the
   * first reneighboring that occurs after the target timestep." "If *binsize* is set to 0.0, then
   * a binsize equal to half the :doc:`neighbor <neighbor>` cutoff distance (force cutoff plus skin
   * distance) is used" "After the atoms have been binned, they are reordered so that atoms in the
   * same bin are adjacent to each other". "If no neighbor cutoff is defined, sorting will be
   * turned off."
   * Measured with native LAMMPS (black box, unsorted dumps of the atom IDs): every run and
   * minimization setup sorts; during a run the sort happens at the first reneighboring with
   * step >= floor(last/N)*N + N (N = 5 and reneighboring every 3 steps: 6, 12, 15, 21, 27, 30);
   * each dimension has floor(L / binsize) bins, x varies fastest, and atoms of one bin keep
   * their previous relative order.
   */
  sortAtoms(setup: boolean): void {
    if (!(this.sortEvery > 0)) return;
    const s = this.state;
    if (!setup && s.step < this.nextSort) return;
    this.nextSort = Math.floor(s.step / this.sortEvery) * this.sortEvery + this.sortEvery;
    const binsize = this.sortBinsize > 0 ? this.sortBinsize : 0.5 * this.nb.cutneighmax;
    if (!(binsize > 0) || s.n < 2) return;
    const b = s.box;
    // atom.cpp setup_sort_bins: a triclinic box is binned in the bounding box of the tilted cell (domain->bbox of the
    // lamda box), which is Domain::set_global_box's boxlo_bound / boxhi_bound
    const lo = [b.lo[0], b.lo[1], b.lo[2]];
    const hi = [b.hi[0], b.hi[1], b.hi[2]];
    if (b.triclinic) {
      const [xy, xz, yz] = b.tilt;
      lo[0] = Math.min(lo[0], lo[0] + xy);
      lo[0] = Math.min(lo[0], lo[0] + xz);
      lo[1] = Math.min(lo[1], lo[1] + yz);
      hi[0] = Math.max(hi[0], hi[0] + xy);
      hi[0] = Math.max(hi[0], hi[0] + xz);
      hi[1] = Math.max(hi[1], hi[1] + yz);
    }
    const len = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const nb = len.map((l, d) => (d === 2 && s.dimension === 2 ? 1 : Math.max(1, Math.floor(l / binsize))));
    const bin = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) {
      let key = 0;
      for (let d = 2; d >= 0; d--) {
        // bin = (x - lo) times the inverse bin width: the rounding decides the bin of atoms on a bin edge
        // (measured with native write_data: the order of the w83sph_shock2d atoms matches only with this form)
        let c = Math.floor((s.x[3 * i + d] - lo[d]) * (nb[d] / len[d]));
        if (c < 0) c = 0; else if (c >= nb[d]) c = nb[d] - 1;
        key = key * nb[d] + c;
      }
      bin[i] = key;
    }
    const ord = Array.from(nativeOrder(s));
    // Array.prototype.sort is stable: atoms of a bin keep their order
    ord.sort((p, q) => bin[p] - bin[q]);
    s.order = Int32Array.from(ord);
  }

  /** Run-time checks of the dynamic groups (group.html; native error texts, measured black box). */
  checkDynamicGroups(): void {
    for (const [bit, g] of this.groups.dynamic) {
      const name = this.groups.names[Math.log2(bit >>> 0)] ?? '';
      const p = this.groups.find(g.parent);
      if (p < 0) throw new StyleError(`Group dynamic parent group ${g.parent} does not exist`);
      if (this.groups.isDynamic(this.groups.bit(g.parent))) throw new StyleError(`Dynamic group parent group ${g.parent} cannot be dynamic`);
      if (g.region && !this.regions.has(g.region)) throw new StyleError(`Region ${g.region} for dynamic group ${name} does not exist`);
      if (g.variable) {
        const v = this.vars.get(g.variable);
        if (!v) throw new StyleError(`Variable '${g.variable}' for dynamic group ${name} does not exist`);
        if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Variable '${g.variable}' for dynamic group ${name} is of incompatible style`);
      }
      if (g.property && !this.state.custom.has(g.property)) throw new StyleError(`Custom per-atom vector ${g.property} for dynamic group ${name} does not exist`);
      if (g.exclude && this.groups.find(g.exclude) < 0) throw new StyleError(`Group ID ${g.exclude} requested by group dynamic exclude does not exist`);
      if (g.includeMolecule && !hasMolecule(this.state)) throw new StyleError('Dynamic Group include molecule setting requires atom attribute molecule');
    }
  }

  /**
   * A within cutoff is a custom neighbor list cutoff (cutoff + skin), which must fit in the
   * communication cutoff. Measured with native LAMMPS (black box, lj/cut 2.5 with the default skin
   * 0.3): within 2.9 stops with Custom neighbor list cutoff 3.1999999999999997 is too large for
   * communication cutoff 2.8"; within 2.5 runs.
   */
  private checkDynamicWithin(): void {
    for (const g of this.groups.dynamic.values()) {
      if (g.within === null) continue;
      const custom = g.within + this.nb.skin;
      if (custom > this.nb.cutghost) {
        throw new StyleError(`Custom neighbor list cutoff ${custom} is too large for communication cutoff ${this.nb.cutghost}`);
      }
    }
  }

  /**
   * (Re)assigns the atoms of the dynamic groups — at the start of a run (setup) and on steps that
   * are a multiple of each group's every, after the forces and before the fixes (group.html:
   * "The point in the timestep at which atoms are assigned to a dynamic group is after interatomic
   * forces have been computed, but before any fixes which alter forces or otherwise update the
   * system have been invoked.").
   */
  assignDynamicGroups(setup: boolean): void {
    if (!this.groups.dynamic.size) return;
    const s = this.state;
    let changed = false;
    for (const [bit, g] of this.groups.dynamic) {
      if (!setup && s.step % g.every !== 0) continue;
      changed = true;
      const parentBit = this.groups.bit(g.parent);
      const region = g.region ? this.region(g.region) : null;
      const vals = g.variable ? this.atomVariable(g.variable) : null;
      const prop = g.property ? this.state.custom.get(g.property)! : null;
      // the parent atoms that pass region, variable and property are the selection
      const member = new Uint8Array(s.n);
      for (let i = 0; i < s.n; i++) {
        let keep = (s.mask[i] & parentBit) !== 0;
        if (keep && region) keep = region.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
        if (keep && vals) keep = vals[i] !== 0;
        if (keep && prop) keep = prop.data[i] !== 0;
        member[i] = keep ? 1 : 0;
      }
      if (g.within !== null) this.addWithin(member, parentBit, g.within);
      if (g.includeMolecule) this.addMolecules(member);
      if (g.exclude) {
        const exBit = this.groups.bit(g.exclude);
        for (let i = 0; i < s.n; i++) if (s.mask[i] & exBit) member[i] = 0;
      }
      const clear = ~bit;
      for (let i = 0; i < s.n; i++) s.mask[i] = member[i] ? s.mask[i] | bit : s.mask[i] & clear;
    }
    // compute values cached earlier in this step belong to the old membership
    if (changed) this.refreshComputes();
  }

  /**
   * within: adds the atoms that are not in the parent group and lie within the cutoff of a selected
   * atom (periodic images count, the distance test is inclusive). Measured with native LAMMPS
   * (black box): see group.ts. Orthogonal boxes use a cell grid of cell width at least the cutoff;
   * triclinic boxes test all pairs.
   */
  private addWithin(member: Uint8Array, parentBit: number, cutoff: number): void {
    const s = this.state, g = this.geom;
    const cut2 = cutoff * cutoff;
    const srcs: number[] = [];
    for (let i = 0; i < s.n; i++) if (member[i]) srcs.push(i);
    if (!srcs.length) return;
    const d = [0, 0, 0];
    const near = (j: number, i: number): boolean => {
      d[0] = s.x[3 * j] - s.x[3 * i]; d[1] = s.x[3 * j + 1] - s.x[3 * i + 1]; d[2] = s.x[3 * j + 2] - s.x[3 * i + 2];
      g.minimumImage(d);
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= cut2;
    };
    if (s.box.triclinic) {
      for (let j = 0; j < s.n; j++) {
        if (s.mask[j] & parentBit) continue;
        for (const i of srcs) if (near(j, i)) { member[j] = 1; break; }
      }
      return;
    }
    const lo = g.lo, len = [g.lx, g.ly, g.lz];
    const nc = [0, 0, 0], w = [cutoff, cutoff, cutoff];
    for (let k = 0; k < 3; k++) {
      if (g.periodic[k]) { nc[k] = Math.max(1, Math.floor(len[k] / cutoff)); w[k] = len[k] / nc[k]; }
    }
    const cell = (x: number, k: number): number => {
      const c = Math.floor((x - lo[k]) / w[k]);
      return g.periodic[k] ? ((c % nc[k]) + nc[k]) % nc[k] : c;
    };
    const buckets = new Map<string, number[]>();
    for (const i of srcs) {
      const key = `${cell(s.x[3 * i], 0)},${cell(s.x[3 * i + 1], 1)},${cell(s.x[3 * i + 2], 2)}`;
      const b = buckets.get(key);
      if (b) b.push(i); else buckets.set(key, [i]);
    }
    for (let j = 0; j < s.n; j++) {
      if (s.mask[j] & parentBit) continue;
      const cx = cell(s.x[3 * j], 0), cy = cell(s.x[3 * j + 1], 1), cz = cell(s.x[3 * j + 2], 2);
      let found = false;
      for (let ox = -1; ox <= 1 && !found; ox++) {
        for (let oy = -1; oy <= 1 && !found; oy++) {
          for (let oz = -1; oz <= 1 && !found; oz++) {
            let ix = cx + ox, iy = cy + oy, iz = cz + oz;
            if (g.periodic[0]) ix = ((ix % nc[0]) + nc[0]) % nc[0];
            if (g.periodic[1]) iy = ((iy % nc[1]) + nc[1]) % nc[1];
            if (g.periodic[2]) iz = ((iz % nc[2]) + nc[2]) % nc[2];
            const b = buckets.get(`${ix},${iy},${iz}`);
            if (!b) continue;
            for (const i of b) if (near(j, i)) { found = true; break; }
          }
        }
      }
      if (found) member[j] = 1;
    }
  }

  /** include molecule: adds the atoms that share a molecule ID (0 is ignored) with a member. */
  private addMolecules(member: Uint8Array): void {
    const s = this.state;
    const ids = new Set<number>();
    for (let i = 0; i < s.n; i++) if (member[i] && s.molecule[i] !== 0) ids.add(s.molecule[i]);
    for (let i = 0; i < s.n; i++) if (!member[i] && s.molecule[i] !== 0 && ids.has(s.molecule[i])) member[i] = 1;
  }

  // ---------------------------------------------------------------- formulas

  readonly formulaEnv: FormulaEnv = {
    name: (name, mode) => this.formulaName(name, mode),
    variable: (name, index, mode) => this.variableRef(name, index, mode),
    reference: (kind, id, i, j, mode) => this.reference(kind, id, i, j, mode),
    atomValue: (name, index) => this.atomValue(name, index),
    custom: (name, b1, b2, mode) => this.customRef(name, b1, b2, mode),
    indexVariable: (name) => this.vars.bracketValue(name, this.formulaEnv),
    raw: (fn, args, mode, index) => groupFunction(this, fn, args, mode, index),
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
    hasFixStyle: (style) => this.fixes.some((f) => f.style === style),
  };

  /*
   * The references of a formula. As native LAMMPS does (src/variable.cpp Variable::evaluate at b332f28): computes, fixes,
   * atom values and thermo keywords need the box; the compute, fix or variable is looked up before its brackets are
   * read; a reference must give what the formula's style takes (equal: a global scalar, an element of a global vector
   * or array, or C_ID[I] / F_ID[I] of one atom; vector: a global vector or a column of a global array; atom: a per-atom
   * vector or a column of a per-atom array), else "does not compute the requested property"; a compute must have been
   * initialized by a run (or another command that initializes the system); a fix that is due on some steps only can be
   * read on those steps of a run.
   */

  private formulaBox(): void {
    if (!this.hasBox) throw new FormulaError(`Variable evaluation before simulation box is defined${errorUrl(30)}`);
  }

  /** A name that is no constant: an atom vector, else a thermo keyword. */
  private formulaName(name: string, mode: Mode): Value {
    this.formulaBox();
    if (ATOM_VECTORS.includes(name)) {
      if (mode === 'atom') return this.atomVector(name);
      if (mode === 'equal') throw new StyleError('Atom vector in equal-style variable formula');
      throw new StyleError(`atom vector '${name}' can only be used in an atom-style variable (use ${name}[ID] for one atom)`);
    }
    const v = this.thermo.keyword(name);
    if (v === undefined) throw new FormulaError(`Invalid thermo keyword '${name}' in variable formula`);
    return v;
  }

  /** v_name and v_name[I]: the name, then the brackets, then the variable's style (messages that name the referenced variable). */
  private variableRef(name: string, index: Index | null, mode: Mode): Value {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`Invalid variable reference v_${name} in variable formula`);
    const s = this._state;
    const ids = s ? s.id : new Int32Array(0), n = s ? s.n : 0;
    if (index === null) return this.vars.value(name, this.formulaEnv, mode, ids, n);
    const i = resolveIndex(index, this.formulaEnv);
    if (v.style === 'vector') {
      const vec = this.vars.evalVector(name, this.formulaEnv);
      if (i > vec.length) throw new FormulaError(`Variable ${name}: Invalid index into vector-style variable`, true);
      return vec[i - 1];
    }
    if (v.style === 'atom' || v.style === 'atomfile') {
      const vals = this.vars.evalAtom(name, this.formulaEnv, ids, n);
      return this.peratom(i, (k) => vals[k]);
    }
    throw new FormulaError(`Variable ${name}: Mismatched variable in variable formula`, true);
  }

  /** The atom map exists: molecular atom styles have one, atomic ones after atom_modify map (native Atom::map_style). */
  private get hasAtomMap(): boolean { return this.atomMapSet || isMolecularStyle(this.atomStyle); }

  /** One atom's value of per-atom data (native Variable::peratom2global): 0 for an ID no atom has. */
  private peratom(id: number, read: (k: number) => number, noMap = 'Indexed per-atom vector in variable formula without atom map'): number {
    if (!this.hasAtomMap) throw new StyleError(noMap);
    const s = this.state;
    let maxId = 0;
    for (let k = 0; k < s.n; k++) if (s.id[k] > maxId) maxId = s.id[k];
    if (id > maxId) throw new StyleError('Variable atom ID is too large');
    const k = this.indexOfId(id);
    return k < 0 ? 0 : read(k);
  }

  private reference(kind: 'c' | 'C' | 'f' | 'F', id: string, i: Index | null, j: Index | null, mode: Mode): Value {
    this.formulaBox();
    const idx = (ix: Index | null) => (ix === null ? null : resolveIndex(ix, this.formulaEnv));
    if (kind === 'c' || kind === 'C') {
      const c = this.computes.find((x) => x.id === id);
      if (!c) throw new FormulaError(`Invalid compute ID '${id}' in variable formula`);
      const i1 = idx(i);
      return this.computeRef(c, kind === 'c', i1, idx(j), mode);
    }
    const f = this.fixes.find((x) => x.id === id);
    if (!f) throw new FormulaError(`Invalid fix ID '${id}' in variable formula`);
    const i1 = idx(i);
    return this.fixRef(f, kind === 'f', i1, idx(j), mode);
  }

  private computeRef(c: Compute, lower: boolean, i: number | null, j: number | null, mode: Mode): Value {
    const mismatch = () => new FormulaError(`Compute '${c.id}' in variable formula does not compute the requested property. ${errorUrl(37)}`);
    const init = () => {
      if (!c.initialized) throw new FormulaError('Variable formula compute cannot be invoked before initialization by a run');
    };
    const range = (what: string) => new FormulaError(`Variable formula compute ${what} is accessed out-of-range${errorUrl(20)}`);
    const column = (a: Float64Array, rows: number, cols: number, col: number) => {
      const out = new Float64Array(rows);
      for (let r = 0; r < rows; r++) out[r] = a[r * cols + col - 1];
      return out;
    };
    if (mode === 'equal') {
      if (lower && i === null) {
        if (!c.scalarFlag) throw mismatch();
        init();
        return c.scalarValue();
      }
      if (lower && j === null) {
        if (!c.vectorFlag) throw mismatch();
        init();
        // the length is read after the compute ran, for a compute whose vector length varies
        const v = c.vectorValues();
        if (i! > v.length) throw range('vector');
        return v[i! - 1];
      }
      if (lower) {
        if (!c.arrayFlag) throw mismatch();
        if (j! > c.sizeArrayCols) throw range('array');
        init();
        const a = c.arrayValues();
        if (i! > c.sizeArrayRows) throw range('array');
        return a[(i! - 1) * c.sizeArrayCols + j! - 1];
      }
      if (i === null || !c.peratomFlag) throw mismatch();
      if (j === null) {
        if (c.sizePeratomCols) throw mismatch();
        init();
        const vals = c.peratomValues();
        return this.peratom(i, (k) => vals[k]);
      }
      if (!c.sizePeratomCols) throw mismatch();
      if (j > c.sizePeratomCols) throw range('array');
      init();
      const vals = c.peratomValues();
      const w = c.sizePeratomCols;
      return this.peratom(i, (k) => vals[k * w + j - 1]);
    }
    if (mode === 'vector') {
      if (lower && i === null) {
        if (!c.vectorFlag) throw mismatch();
        init();
        const v = c.vectorValues();
        if (v.length === 0) throw new FormulaError('Variable formula compute vector is zero length');
        return v;
      }
      if (lower && j === null) {
        if (!c.arrayFlag) throw mismatch();
        init();
        if (i! > c.sizeArrayCols) throw range('array');
        const a = c.arrayValues();
        if (c.sizeArrayRows === 0) throw new FormulaError('Variable formula compute array has zero rows');
        return column(a, c.sizeArrayRows, c.sizeArrayCols, i!);
      }
      throw mismatch();
    }
    if (lower && i === null) {
      if (!c.peratomFlag || c.sizePeratomCols) throw mismatch();
      init();
      return c.peratomValues();
    }
    if (lower && j === null) {
      if (!c.peratomFlag || !c.sizePeratomCols) throw mismatch();
      if (i! > c.sizePeratomCols) throw range('array');
      init();
      return column(c.peratomValues(), this.state.n, c.sizePeratomCols, i!);
    }
    throw mismatch();
  }

  private fixRef(f: Fix, lower: boolean, i: number | null, j: number | null, mode: Mode): Value {
    const mismatch = () => new FormulaError(`Fix '${f.id}' in variable formula does not compute the requested property. ${errorUrl(37)}`);
    const range = (what: string) => new FormulaError(`Variable formula fix ${what} is accessed out-of-range${errorUrl(20)}`);
    // native's texts differ in one word ("at a compatible time" / "at compatible time")
    const due = (a: string) => {
      if (this.run.inRun && f.globalFreq > 0 && this.state.step % f.globalFreq !== 0) {
        throw new FormulaError(`Fix in variable not computed at ${a}compatible time${errorUrl(7)}`);
      }
    };
    const peratom = () => (f.sizePeratomCols === 0 ? f.vectorAtom : f.arrayAtom);
    if (mode === 'equal') {
      if (lower && i === null) {
        if (!f.scalarFlag) throw mismatch();
        due('a ');
        return f.computeScalar();
      }
      // a fix whose vector length or row count varies is read as 0 past its end, without the range error
      if (lower && j === null) {
        if (!f.vectorFlag) throw mismatch();
        if (i! > f.sizeVector && !f.sizeVectorVariable) throw range('vector');
        due('a ');
        if (f.sizeVectorVariable && i! > f.sizeVector) return 0;
        return f.computeVector(i! - 1);
      }
      if (lower) {
        if (!f.arrayFlag) throw mismatch();
        if (i! > f.sizeArrayRows && !f.sizeArrayRowsVariable) throw range('array');
        if (j! > f.sizeArrayCols) throw range('array');
        due('a ');
        if (f.sizeArrayRowsVariable && i! > f.sizeArrayRows) return 0;
        return f.computeArray(i! - 1, j! - 1);
      }
      if (i === null || !f.peratomFlag) throw mismatch();
      if (j === null) {
        if (f.sizePeratomCols) throw mismatch();
        const vals = peratom();
        return this.peratom(i, (k) => vals[k]);
      }
      if (!f.sizePeratomCols) throw mismatch();
      if (j > f.sizePeratomCols) throw range('array');
      const vals = peratom();
      const w = f.sizePeratomCols;
      return this.peratom(i, (k) => vals[k * w + j - 1]);
    }
    if (mode === 'vector') {
      if (lower && i === null) {
        if (!f.vectorFlag) throw mismatch();
        if (f.sizeVector === 0) throw new FormulaError('Variable formula fix vector is zero length');
        due('');
        const out = new Float64Array(f.sizeVector);
        for (let k = 0; k < f.sizeVector; k++) out[k] = f.computeVector(k);
        return out;
      }
      if (lower && j === null) {
        if (!f.arrayFlag) throw mismatch();
        if (f.sizeArrayRows === 0) throw new FormulaError('Variable formula fix array is zero length');
        if (i! > f.sizeArrayCols) throw range('array');
        due('a ');
        const out = new Float64Array(f.sizeArrayRows);
        for (let r = 0; r < f.sizeArrayRows; r++) out[r] = f.computeArray(r, i! - 1);
        return out;
      }
      throw mismatch();
    }
    if (lower && i === null) {
      if (!f.peratomFlag || f.sizePeratomCols) throw mismatch();
      return f.vectorAtom;
    }
    if (lower && j === null) {
      if (!f.peratomFlag || !f.sizePeratomCols) throw mismatch();
      if (i! > f.sizePeratomCols) throw range('array');
      const w = f.sizePeratomCols;
      const n = this.state.n;
      const out = new Float64Array(n);
      for (let k = 0; k < n; k++) out[k] = f.arrayAtom[k * w + i! - 1];
      return out;
    }
    throw mismatch();
  }

  private randomValue(a: number, b: number, seed: number, mode: Mode, gauss: boolean): Value {
    const draw = (r: FormulaRandom) => (gauss ? a + b * r.gaussian() : a + (b - a) * r.uniform());
    // As native LAMMPS does (Variable::math_function and collapse_tree at b332f28): the seed, read as an int, makes the
    // generator at the first call and must be positive then; the per-atom generator takes the seed plus the process rank
    // (0 here, one process)
    const make = (offset: number) => {
      const s = Math.trunc(seed);
      if (s <= 0) throw mode === 'equal' ? new FormulaError('Invalid math function in variable formula') : new StyleError('Invalid math function in variable formula');
      if (s + offset > 900000000) throw new StyleError(`Invalid seed ${s + offset} for the Marsaglia random # generator`);
      return new FormulaRandom(s + offset);
    };
    if (mode !== 'atom') {
      this.rngEqual ??= make(0);
      return draw(this.rngEqual);
    }
    this.rngAtom ??= make(0);
    const n = this.state.n;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = draw(this.rngAtom);
    return out;
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

  /** NAME[ID]: native's box check, brackets, atom map and then the atom's value (peratom2global). */
  private atomValue(name: string, index: Index): number {
    this.formulaBox();
    const id = resolveIndex(index, this.formulaEnv);
    return this.peratom(id, (k) => {
      if (!ATOM_VECTORS.includes(name)) throw new StyleError(`Invalid atom vector ${name} in variable formula`);
      if (!this.atomHas(name)) throw new StyleError('Variable uses atom property that isn\'t allocated');
      return this.atomVectorAt(name, k);
    });
  }

  /** The atom style stores mol, radius and q (native molecule_flag, radius_flag, q_flag). */
  private atomHas(name: string): boolean {
    const s = this.state;
    if (name === 'mol') return hasMolecule(s);
    if (name === 'radius') return !!s.radius;
    if (name === 'q') return hasCharge(s);
    return true;
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
      case 'radius': return s.radius![i];
    }
    throw new StyleError(`Invalid atom vector ${name} in variable formula`);
  }

  private atomVector(name: string): Float64Array {
    const s = this.state;
    // native atom_vector: "Variable uses atom property 'mol' that isn't allocated" (the same for radius and q)
    if (!this.atomHas(name)) throw new StyleError(`Variable uses atom property '${name}' that isn't allocated`);
    if (name.startsWith('f')) this.forces();
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) out[i] = this.atomVectorAt(name, i);
    return out;
  }

  /**
   * i_name, d_name, i2_name and d2_name. As native LAMMPS does (Variable::evaluate, variable.cpp at b332f28): the property
   * is found by its name, then the brackets are read; a vector takes no bracket (its per-atom values) or one (the atom
   * ID), an array one (a column) or two (atom ID and column). Measured with native LAMMPS (2 Sep 2026): native reads
   * J of i2_name[I][J] as 0-based (J = Ncol runs into the next atom's row); the engine follows the documented 1-based
   * columns (variable.html: "i2_name[I][J]").
   */
  private customRef(word: string, b1: Index | null, b2: Index | null, mode: Mode): Value {
    this.formulaBox();
    const s = this.state;
    const prop = s.custom.get(word.slice(/^[id]2_/.test(word) ? 3 : 2));
    const bad = () => new FormulaError(`Invalid custom atom property reference ${word} in variable formula`);
    if (!prop) throw bad();
    const i1 = b1 === null ? null : resolveIndex(b1, this.formulaEnv);
    const i2 = b2 === null ? null : resolveIndex(b2, this.formulaEnv);
    const w = prop.cols;
    // a per-atom vector of an equal- or vector-style formula is no value there (native's formula has none left)
    const perAtom = (col: number) => {
      if (mode !== 'atom') throw new FormulaError('Invalid syntax in variable formula');
      const out = new Float64Array(s.n);
      for (let k = 0; k < s.n; k++) out[k] = w ? prop.data[k * w + col - 1] : prop.data[k];
      return out;
    };
    const noMap = 'Referenced custom atom property in variable formula without atom map';
    if (i1 === null) {
      if (w) throw bad();
      return perAtom(0);
    }
    if (i2 === null) {
      if (!w) return this.peratom(i1, (k) => prop.data[k], noMap);
      if (i1 > w) throw bad();
      return perAtom(i1);
    }
    if (!w || i2 > w) throw bad();
    return this.peratom(i1, (k) => prop.data[k * w + i2 - 1], noMap);
  }

  /** An equal-style formula's value (used by commands that take v_name arguments). */
  /**
   * Input::substitute on a text (a fix's label or colour-scale text): $x, ${name} and $(formula[:fmt]) with the variables
   * of the run now (Input::substitute, input.cpp at b332f28; the same rule as the command words of the session).
   */
  substituteText(text: string): string {
    return substituteVariables(text, (name) => this.vars.text(name, this.formulaEnv),
      (f, fmt) => formatNumber(evaluateScalar(f, this.formulaEnv), fmt ?? '%.20g'));
  }

  equalVariable(name: string): number {
    return this.vars.scalar(name, this.formulaEnv);
  }

  /** Per-atom values of an atom-style variable. */
  atomVariable(name: string): Float64Array {
    const s = this.state;
    return this.vars.evalAtom(name, this.formulaEnv, s.id, s.n);
  }
}

