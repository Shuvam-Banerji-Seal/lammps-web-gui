import { EngineError, type EngineEvent, type ForceBackend } from './types';
import type { ForceField } from './force/forcefield';
import { System, type SessionIO } from './system';
import { RUN_API } from './commands/runApi';
import { StyleError } from './force/types';
import { removedStyleError, renamedCommand } from './deprecated';
import { mapUnquoted, splitCommands, substituteVariables, tokenize, formatNumber, type RawCommand } from './script';
import { evaluateScalar } from './formula';
import { COMMANDS, UNAVAILABLE_COMMANDS, type Ctx } from './commands';
import { ensureStyles, LAZY_COMMANDS, styleNames } from './styles';
import { evaluateBoolean } from './boolean';
import { CpuForceBackend } from './cpu/forces';
import { RunCancelled, QuitSignal } from './errors';
import { ACCEL_ALIASES } from './accelAliases';
import { normalizeFileName } from './fileName';

/*
 * The notebook's LAMMPS input interpreter (engine v2). It reads input as a
 * stream of commands with a program counter so that the documented control
 * flow works:
 *   label  — docs.lammps.org/label.html: "Label this line of the input
 *            script. Labels are used by the jump command."
 *   jump   — docs.lammps.org/jump.html: "If the word "SELF" is used for the
 *            filename, then the current input script is re-opened and read
 *            again." "the new file is scanned (without executing commands)
 *            until the label is found, and commands are executed from that
 *            point forward."
 *   next   — docs.lammps.org/next.html: when a variable is exhausted "the
 *            next jump command encountered is skipped".
 *   include — docs.lammps.org/include.html: "This command opens a new input
 *            script file and begins reading LAMMPS commands from that file.
 *            When the new file is finished, the original file is returned to."
 *   if     — docs.lammps.org/if.html.
 * Variable substitution happens when a command is read, as in
 * docs.lammps.org/Commands_parse.html. In the notebook a cell is a "file":
 * jump SELF re-reads the cell; other file names come from the notebook's
 * files. Unsupported commands are an EngineError naming the command — never
 * a silent no-op.
 */

export { RunCancelled, QuitSignal };

interface Frame {
  /** Cell or file name, for messages. */
  name: string;
  text: string;
  firstLine: number;
  cmds: RawCommand[];
  pc: number;
}

/** The shared-memory pair threads a backend carries (cpu/pairThreads.ts SharedThreadsBackend), if any. */
const pairThreadsOf = (b: ForceBackend): ForceField['pairThreads'] =>
  (b as { pairThreads?: ForceField['pairThreads'] }).pairThreads ?? null;

/**
 * The partition commands that drive all replicas together and read their words per replica (each replica's own
 * variables, as lammps -partition): neb (neb.html) and the tempering commands (temper.html, temper/npt, temper/grem).
 */
const REPLICA_DRIVEN = new Set(['neb', 'neb/spin', 'temper', 'temper/npt', 'temper/grem', 'prd', 'tad']);

/**
 * Commands that run once for the session rather than on every replica (partitions): control flow, `clear`,
 * `echo`, and `neb`, which drives all replicas together (neb.html: "a NEB calculation performs a damped
 * dynamics minimization across all the replicas"), and `neb/spin` (neb_spin.html: "a GNEB calculation performs a
 * minimization across all the replicas.").
 */
const SESSION_LEVEL = new Set(['label', 'jump', 'next', 'include', 'if', 'quit', 'clear', 'echo', 'neb', 'neb/spin']);
/** Words that make an if a session-level command (its branch is not run per replica). */
const SESSION_FLOW_WORDS = /\b(label|jump|next|include|quit|clear|echo|neb|run|minimize|rerun)\b/;

/**
 * The commands whose words can name a style, with the index of the first such word. fix and compute take an
 * ID and a group before the style; every other word of the commands below is a style name, a sub-style of a
 * hybrid style, or a number (pair_coeff's type ranges). suffix and package name accelerator packages, not
 * styles, and the other commands take no style name, so they are not rewritten.
 * *_write (pair_write, bond_write, angle_write, ...) takes a style name too; the lazily loaded gpl_cmds group implements them.
 */
const STYLE_WORDS_FROM: Record<string, number> = {
  pair_style: 0, pair_coeff: 0, pair_modify: 0, pair_write: 0,
  bond_style: 0, bond_coeff: 0, bond_write: 0,
  angle_style: 0, angle_coeff: 0, angle_write: 0,
  dihedral_style: 0, dihedral_coeff: 0, dihedral_write: 0,
  improper_style: 0, improper_coeff: 0, improper_write: 0,
  kspace_style: 0,
  fix: 2, compute: 2,
};

/** Commands that run native's lmp->init(), which opens the single-file dumps (see dispatch). */
const INIT_COMMANDS = new Set(['run', 'minimize', 'rerun', 'write_data', 'write_restart', 'write_coeff', 'balance']);

export class Session {
  /** The active system: replica 0 between commands, the replica being run while a command runs on all of them. */
  sys: System;
  /** One System per partition (replica); a single entry unless partitions > 1. */
  private replicas: System[] = [];
  private frames: Frame[] = [];
  private skipJump = false;
  /** True while a command runs on the replicas one by one. */
  private onReplica = false;
  /** The words of the replica-driven command being run (REPLICA_DRIVEN), as each replica parsed them. */
  private nebWords: string[][] | null = null;
  /** True while a run of coupled replicas (a fix with `coupled`, fix pimd) runs in lockstep on all of them. */
  private coupledRun = false;
  /**
   * Test harness switch (scripts/oracle, tests/oracleHarness.ts): the thermo rows and the written files of the
   * replicas other than the first are passed on (thermo rows tagged with their partition). Off, they are dropped as
   * before, so the app shows replica 0 only.
   */
  recordPartitionOutput = false;
  private cancelled = false;
  private line = 0;
  /** The text of the command being run (partition re-runs the rest of its line). */
  private text = '';
  private command = '';
  echo: 'none' | 'screen' | 'log' | 'both' = 'none';
  /** Accelerator names already reported in this session (each is logged once). */
  private accelNoted = new Set<string>();

  constructor(
    private io: SessionIO,
    private backend: ForceBackend = new CpuForceBackend(),
    /** Also emit a 'frame' every this many steps during a run (0 = only at dumps and at the end). */
    readonly frameEvery = 0,
    /**
     * Number of partitions (replicas, as with lammps -partition N): every command runs on each replica in turn,
     * variables of style world/universe/uloop take the replica's value, and `neb` runs the replicas together.
     */
    readonly partitions = 1,
  ) {
    if (!Number.isInteger(partitions) || partitions < 1) throw new EngineError(`partitions must be an integer >= 1 (got ${partitions})`, 0, '');
    for (let k = 0; k < partitions; k++) this.replicas.push(this.newSystem(k));
    this.sys = this.replicas[0];
  }

  private newSystem(k = 0): System {
    const sys = new System(k === 0 ? this.io : this.replicaIo(k));
    sys.registries = { ...styleNames(), command: SUPPORTED_COMMANDS.slice() };
    sys.commands = COMMANDS;
    sys.ff.pairThreads = pairThreadsOf(this.backend);
    sys.vars.partition = k;
    sys.vars.nworlds = this.partitions;
    sys.replicaPeers = () => this.replicas;
    return sys;
  }

  /** The I/O of replica k (k > 0): dropped, unless recordPartitionOutput. */
  private replicaIo(k: number): SessionIO {
    return {
      emit: (ev) => { if (this.recordPartitionOutput && ev.kind === 'thermo') this.io.emit({ ...ev, partition: k }); },
      writeFile: (name, text, append) => { if (this.recordPartitionOutput) this.io.writeFile?.(name, text, append); },
    };
  }

  /** The systems of all partitions, replica 0 first (the neb and tempering commands read and drive them). */
  get replicaSystems(): readonly System[] { return this.replicas; }

  /** The replicas a run of coupled systems steps in lockstep (null for an ordinary run). */
  get coupledSystems(): readonly System[] | null { return this.coupledRun ? this.replicas : null; }

  /** Whether frames carry velocities and masses (EngineEvent 'frame', v and mass); off by default. */
  setFrameVelocities(on: boolean): void {
    for (const sys of this.replicas) sys.frameVelocities = on;
  }

  /** Sends an event through the session's output (the neb command's table and log lines). */
  emit(ev: EngineEvent): void { this.io.emit(ev); }

  /** The arguments of the running replica-driven command (neb, temper) as replica r parsed them (after $-substitution with its variables). */
  argsOf(r: number): readonly string[] | undefined { return this.nebWords?.[r]?.slice(1); }

  /** The current system (null before create_box). */
  get system() { return this.sys.hasBox ? this.sys.state : null; }
  get backendLabel(): string { return this.backend.label; }
  get forceBackend(): ForceBackend { return this.backend; }

  /** Swaps the force backend between runs; the system and settings stay. */
  setBackend(backend: ForceBackend): void {
    this.backend = backend;
    for (const sys of this.replicas) {
      sys.ff.pairThreads = pairThreadsOf(backend);
      sys.bump();
    }
  }

  /** Requests that a running `run` stop after its current step. */
  cancel(): void { this.cancelled = true; }
  get isCancelled(): boolean { return this.cancelled; }

  /** Makes a file available to read_data, include, jump, potential files. */
  addFile(name: string, text: string): void {
    name = normalizeFileName(name);
    for (const sys of this.replicas) sys.files.set(name, text);
  }

  /** Runs a change of the session's virtual file system (the shell built-ins) on every replica. */
  forEachReplica(fn: (sys: System) => void): void {
    for (const sys of this.replicas) fn(sys);
  }

  /** Forgets a file added with addFile (or written by the session). */
  removeFile(name: string): void {
    name = normalizeFileName(name);
    for (const sys of this.replicas) sys.files.delete(name);
  }

  /** clear: a fresh system per replica; "input script variables" survive (clear.html). */
  clear(): void {
    const old = this.replicas;
    this.replicas = old.map((prev, k) => {
      const sys = this.newSystem(k);
      for (const [name, v] of prev.vars.vars) sys.vars.vars.set(name, v);
      sys.vars.clear();
      for (const [name, text] of prev.files) sys.files.set(name, text);
      // shell cd, mkdir and putenv state survives clear (clear.html changes only the simulation)
      sys.cwd = prev.cwd;
      for (const d of prev.dirs) sys.dirs.add(d);
      for (const [k, v] of prev.env) sys.env.set(k, v);
      return sys;
    });
    this.sys = this.replicas[0];
  }

  /**
   * Executes input text. Stops at the first error, which is emitted as an
   * 'error' event and rethrown as an EngineError.
   */
  async execute(text: string, firstLine = 1, name = 'cell'): Promise<void> {
    this.cancelled = false;
    this.frames = [];
    // the C library log, sin and cos of the native binary (libm.ts) and native LAMMPS's Richardson orientation update
    // (orientation in fix/nve_asphere.ts): installed before the first command
    await ensureStyles(['libm', 'mathextra']);
    try {
      await this.runText(text, firstLine, name);
    } catch (e) {
      if (e instanceof QuitSignal) { this.io.emit({ kind: 'log', text: 'Total wall time: quit' }); return; }
      throw e;
    }
  }

  private async runText(text: string, firstLine: number, name: string): Promise<void> {
    const frame: Frame = { name, text, firstLine, cmds: splitCommands(text, firstLine), pc: 0 };
    this.frames.push(frame);
    try {
      while (frame.pc < frame.cmds.length) {
        const cmd = frame.cmds[frame.pc++];
        await this.runCommand(cmd.text, cmd.line, frame);
      }
    } finally {
      this.frames.pop();
    }
  }

  /**
   * Runs one command (from the stream or from an if / run every). With partitions, an ordinary command runs on
   * every replica in turn, each parsed with its own variables (the same input runs on every partition).
   */
  async runCommand(text: string, line: number, frame: Frame | null = this.frames[this.frames.length - 1] ?? null): Promise<void> {
    // a command nested in one replica's run (run ... every) runs on that replica only
    if (this.replicas.length === 1 || this.onReplica) return this.runOne(text, line, frame);
    const first = this.parse(text)[0];
    if (first !== undefined && REPLICA_DRIVEN.has(first)) {
      // neb and the tempering commands run once, but their arguments are read per replica ($u gives each partition its own value)
      this.nebWords = this.replicas.map((sys) => { this.sys = sys; return this.parse(text); });
      this.sys = this.replicas[0];
      try { return await this.runOne(text, line, frame); } finally { this.nebWords = null; }
    }
    // a run with a coupled fix (fix pimd: every bead's forces and positions are needed each step) runs once, in lockstep
    if (first === 'run' && this.replicas.some((sys) => sys.fixes.some((f) => f.replicaCoupled))) {
      this.coupledRun = true;
      this.sys = this.replicas[0];
      try { return await this.runOne(text, line, frame); } finally { this.coupledRun = false; }
    }
    // an if whose branches run no flow or run command is evaluated on each replica (fix_alchemy.rst at b332f28: "This
    // can be initialized in the same input script by using commands which only apply to one or the other replica. The
    // example scripts use a world-style variable command along with if/then/else commands for this purpose."), so
    // each partition takes its own branch; an if with a jump, next, run or similar stays a session-level command
    const perReplicaIf = first === 'if' && !SESSION_FLOW_WORDS.test(text);
    if (first === undefined || (SESSION_LEVEL.has(first) && !perReplicaIf)) return this.runOne(text, line, frame);
    this.onReplica = true;
    try {
      for (const sys of this.replicas) {
        this.sys = sys;
        await this.runOne(text, line, frame);
      }
    } finally {
      this.onReplica = false;
      this.sys = this.replicas[0];
    }
  }

  private async runOne(text: string, line: number, frame: Frame | null): Promise<void> {
    this.line = line;
    this.text = text;
    this.command = '';
    try {
      const words = this.parse(text);
      if (words.length === 0) return;
      this.command = words[0];
      if (this.echo === 'screen' || this.echo === 'both' || this.echo === 'log') this.io.emit({ kind: 'log', text: words.join(' ') });
      await this.dispatch(words[0], words.slice(1), frame);
    } catch (e) {
      if (e instanceof RunCancelled) {
        this.io.emit({ kind: 'log', text: `Run cancelled at step ${this.sys.hasBox ? this.sys.state.step : 0}` });
        throw e;
      }
      if (e instanceof QuitSignal) throw e;
      if (e instanceof EngineError) throw e;
      const where = frame && frame.name !== 'cell' ? ` (in ${frame.name})` : '';
      const err = new EngineError((e instanceof Error ? e.message : String(e)) + where, this.line, this.command);
      this.io.emit({ kind: 'error', message: err.message, line: err.line, command: err.command });
      throw err;
    }
  }

  /** $-substitution and word splitting (Commands_parse.html steps 3-6). */
  parse(text: string): string[] {
    const env = this.sys.formulaEnv;
    const substituted = mapUnquoted(text, (part) => substituteVariables(
      part,
      (name) => this.sys.vars.text(name, env),
      (f, fmt) => formatNumber(evaluateScalar(f, env), fmt ?? '%.20g'),
    ));
    return tokenize(substituted);
  }

  private ctx(frame: Frame | null): Ctx {
    return {
      sys: this.sys,
      session: this,
      frame,
      fail: (m: string): never => { throw new StyleError(m); },
      run: RUN_API,
    };
  }

  /**
   * partition style N command — docs.lammps.org/partition.html: "This command invokes the specified command on a
   * subset of the partitions of processors you have defined via the -partition command-line switch." "If the style
   * is yes, the command is invoked on the specified partition(s). If the style is no, the command is invoked on all
   * partitions except the specified partition(s)." "Partitions are numbered from 1 to Np". As native LAMMPS does
   * (src/input.cpp Input::partition at b332f28): the style is a utils::logical word, N a utils::bounds range over
   * 1..Np, the third word may not be partition, and the rest of the line is run as one command on this partition.
   */
  private async partitionCommand(a: string[], frame: Frame | null): Promise<void> {
    if (a.length < 3) throw new StyleError('Illegal partition command: missing argument(s)');
    const w = a[0];
    let yes: boolean;
    if (w === 'yes' || w === 'on' || w === 'true' || w === '1') yes = true;
    else if (w === 'no' || w === 'off' || w === 'false' || w === '0') yes = false;
    else throw new StyleError(`Expected boolean parameter instead of '${w}' in input script or data file`);
    const np = this.replicas.length;
    const range = a[1];
    if (/[^*\-0-9]/.test(range)) throw new StyleError(`Invalid range string: ${range}`);
    const star = range.indexOf('*');
    const lo = star < 0 ? Number(range) : star === 0 ? 1 : Number(range.slice(0, star));
    const hi = star < 0 ? lo : star === range.length - 1 ? np : Number(range.slice(star + 1));
    if (!(lo > 0) || !(hi > 0)) throw new StyleError(`Invalid range string: ${range}`);
    const url = '\nFor more information see https://docs.lammps.org/err0019';
    if (lo < 1) throw new StyleError(`Numeric index ${lo} is out of bounds (1-${np})${url}`);
    if (hi > np) throw new StyleError(`Numeric index ${hi} is out of bounds (1-${np})${url}`);
    if (lo > hi) throw new StyleError(`Numeric index ${lo} is out of bounds (1-${hi})${url}`);
    if (a[2] === 'partition') throw new StyleError('Illegal partition command argument');
    const me = this.sys.vars.partition + 1;
    const inside = me >= lo && me <= hi;
    if (yes !== inside) return;
    // the rest of the line from the third word (native strstr(line, arg[2])), run on this partition only
    const rest = this.text.replace(/^\s*\S+\s+\S+\s+\S+\s+/, '');
    const was = this.onReplica;
    this.onReplica = true;
    try { await this.runOne(rest, this.line, frame); } finally { this.onReplica = was; }
  }

  private async dispatch(cmd: string, a: string[], frame: Frame | null): Promise<void> {
    // a command LAMMPS renamed (deprecated.ts) runs under its new name after native's warning
    const renamed = renamedCommand(cmd, a);
    if (renamed) {
      this.sys.warn(renamed.warning);
      return this.dispatch(renamed.cmd, renamed.args, frame);
    }
    // a formula that calls omega() (variable, print, if, fix print ...) needs the group motion helper of
    // groupMotion.ts; it is loaded here, before the switch below, which runs some commands (if) without the lookup
    if (a.some((w) => w.includes('omega('))) await ensureStyles(['omega']);
    // with partitions there is one input stream: each replica advances its own loop variables for next (the same
    // command on every replica, as lammps -partition runs it on each partition), and the jump is taken only when
    // all partitions agree that their loop has run out, so the replicas stay on the same command
    switch (cmd) {
      case 'label':
        // the messages of native's Input (src/input.cpp at b332f28)
        if (a.length !== 1) throw new StyleError(`Illegal label command: expected 1 argument but found ${a.length}`);
        return;
      case 'jump': return this.jump(a, frame);
      case 'next':
        if (!a.length) throw new StyleError('Illegal next command: missing argument(s)');
        if (this.replicas.length > 1) {
          const outs = this.replicas.map((sys) => sys.vars.next(a));
          if (outs.some((o) => o !== outs[0])) {
            throw new StyleError("'next': the partitions' loop variables run out at different times (the partitions must run the same number of loop passes)");
          }
          if (outs[0]) this.skipJump = true;
          return;
        }
        if (this.sys.vars.next(a)) this.skipJump = true;
        return;
      case 'include': {
        if (a.length !== 1) throw new StyleError('Illegal include command. Must name one file to include.');
        const text = this.readScript(a[0]);
        await this.runText(text, 1, a[0]);
        return;
      }
      case 'if': return this.ifCommand(a, frame);
      case 'partition': return this.partitionCommand(a, frame);
      case 'quit': throw new QuitSignal(a.length ? Number(a[0]) : 0);
      case 'clear':
        if (a.length) throw new StyleError('clear takes no arguments');
        this.clear();
        this.io.emit({ kind: 'log', text: 'Cleared: atoms, box, settings, fixes and computes reset (variables kept)' });
        return;
      case 'echo':
        if (a.length !== 1) throw new StyleError('Echo command expects exactly one argument');
        if (!['none', 'screen', 'log', 'both'].includes(a[0])) throw new StyleError(`Unknown echo command keyword: ${a[0]}`);
        this.echo = a[0] as Session['echo'];
        return;
    }
    // A single-file dump opens its file at the first system init after the dump command (src/dump_custom.cpp
    // init_style at b332f28: "open single file, one time only"), which these commands perform (lmp->init() in
    // src/run.cpp, minimize.cpp, rerun.cpp, write_data.cpp, write_restart.cpp, write_coeff.cpp and balance.cpp), so the
    // file name is resolved from the working directory (shell cd) at that point. Measured with native LAMMPS: a dump
    // defined in out/ then "shell cd .." before the run writes its file in the root; with a write_data in out/ between
    // them, in out/.
    if (INIT_COMMANDS.has(cmd)) for (const sys of this.replicas) for (const d of sys.dumps) d.openFile();
    // an accelerator alias (lj/cut/omp, nve/kk, ...) runs as its base style: the names are replaced here, before
    // the styles load, so every object is created with the base name and the fast paths see the base style
    a = this.baseStyleNames(cmd, a);
    // styles of the lazily loaded registries (lazyStyles.ts) are loaded before a command that names one;
    // read_restart restores the styles named in its file
    // a lazily loaded command (a COMMANDS name of lazyStyles.ts) is registered by its group before the lookup below
    await ensureStyles([cmd, ...a]);
    const handler = COMMANDS[cmd] ?? LAZY_COMMANDS[cmd];
    // a native binary restart file (output/restart.ts) is read by the lazily loaded helper, whatever its words
    if (cmd === 'read_restart') await ensureStyles(['native-restart']);
    // the words of a restart file: a native binary file has its style names between non-printable length bytes, so
    // every run of word characters counts (an over-loaded group is harmless)
    if (cmd === 'read_restart' && a.length && this.sys.files.has(normalizeFileName(a[0]))) await ensureStyles(this.sys.files.get(normalizeFileName(a[0]))!.split(/[^\w/.+*-]+/));
    if (!handler) {
      throw removedStyleError('command', cmd) ?? new StyleError(`'${cmd}' is not supported by the in-browser engine (it is not LAMMPS). Supported commands: ${SUPPORTED_COMMANDS.join(', ')}`);
    }
    await handler(this.ctx(frame), a);
  }

  /** The words of a command with each accelerator alias replaced by its base style (ACCEL_ALIASES). */
  private baseStyleNames(cmd: string, a: string[]): string[] {
    if (!Object.hasOwn(STYLE_WORDS_FROM, cmd)) return a;
    const from = STYLE_WORDS_FROM[cmd];
    let out = a;
    for (let k = from; k < a.length; k++) {
      const base = ACCEL_ALIASES.get(a[k]);
      if (base === undefined) continue;
      if (out === a) out = a.slice();
      out[k] = base;
      if (!this.accelNoted.has(a[k])) {
        this.accelNoted.add(a[k]);
        this.io.emit({ kind: 'log', text: `${a[k]}: runs as ${base} (the browser engine picks CPU threads or the GPU itself)` });
      }
    }
    return out;
  }

  /** jump file [label] */
  private jump(a: string[], frame: Frame | null): void {
    if (a.length < 1 || a.length > 2) throw new StyleError(`Illegal jump command: expected 1 or 2 argument(s) but found ${a.length}`);
    if (this.skipJump) { this.skipJump = false; return; }
    if (!frame) throw new StyleError('jump can only be used in an input script');
    const [file, label] = a;
    if (file !== 'SELF') {
      // "the original file is not returned to": the current stream becomes the new file
      const text = this.readScript(file);
      frame.name = file;
      frame.text = text;
      frame.firstLine = 1;
      frame.cmds = splitCommands(text, 1);
    }
    frame.pc = 0;
    if (label === undefined) return;
    // scan for "label ID" without executing commands
    for (let k = 0; k < frame.cmds.length; k++) {
      const w = tokenize(frame.cmds[k].text);
      if (w[0] === 'label' && w[1] === label) { frame.pc = k + 1; return; }
    }
    throw new StyleError(`Label ${label} wasn't found in input script`);
  }

  /** An input script for include or jump: native's message when the file is missing, then the engine's hint. */
  private readScript(file: string): string {
    return this.sys.readFile(file, 'Cannot open input script');
  }

  /**
   * if Boolean then t1 t2 ... elif Boolean f1 f2 ... else e1 e2 ... — as native LAMMPS does (src/input.cpp
   * Input::ifthenelse at b332f28): the first Boolean is evaluated before "then" is checked, and an elif or else branch
   * is read only when the branches before it were not taken.
   */
  private async ifCommand(a: string[], frame: Frame | null): Promise<void> {
    if (a.length < 3) throw new StyleError('Illegal if command: missing argument(s)');
    // "all variables used will be substituted for before the Boolean expression in evaluated"
    const env = this.sys.formulaEnv;
    const subst = (t: string) => substituteVariables(t, (name) => this.sys.vars.text(name, env),
      (f, fmt) => formatNumber(evaluateScalar(f, env), fmt ?? '%.20g'));
    let btest = evaluateBoolean(subst(a[0]));
    if (a[1] !== 'then') throw new StyleError(`Illegal if command: expected "then" but found "${a[1]}"`);
    const bound = (from: number) => { let k = from; while (k < a.length && a[k] !== 'elif' && a[k] !== 'else') k++; return k; };
    const run = async (cmds: string[], what: string) => {
      if (cmds.length <= 0) throw new StyleError(`Illegal ${what} command: missing argument(s)`);
      if (cmds.some((c) => c === '')) throw new StyleError(`Illegal ${what} command: execute command is empty`);
      for (const c of cmds) {
        await this.runCommand(c, this.line, frame);
        // a jump inside the if moves the enclosing stream; stop executing this branch
        if (frame && /^\s*jump\b/.test(c)) break;
      }
    };
    let iarg = bound(2);
    if (btest) { await run(a.slice(2, iarg), 'if then'); return; }
    while (iarg !== a.length) {
      if (iarg + 2 > a.length) throw new StyleError('Illegal if then command: missing argument(s)');
      let first: number;
      if (a[iarg] === 'elif') { btest = evaluateBoolean(subst(a[iarg + 1])); first = iarg + 2; } else { btest = true; first = iarg + 1; }
      iarg = bound(first);
      if (!btest) continue;
      await run(a.slice(first, iarg), 'if elif/else');
      return;
    }
  }
}

/** Commands the engine runs (recognized-but-unavailable ones such as python or shell excluded). */
export const SUPPORTED_COMMANDS = [
  'label', 'jump', 'next', 'include', 'if', 'partition', 'quit', 'clear', 'echo',
  // neb/spin runs at session level like neb; its handler is registered by a lazily loaded group (lazyStyles.ts)
  'neb/spin',
  ...Object.keys(COMMANDS).filter((c) => !UNAVAILABLE_COMMANDS.has(c)),
].sort();
