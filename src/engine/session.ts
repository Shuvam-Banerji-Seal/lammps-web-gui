import { EngineError, type EngineEvent, type ForceBackend } from './types';
import type { ForceField } from './force/forcefield';
import { System, type SessionIO } from './system';
import { StyleError } from './force/types';
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
 * Commands that run once for the session rather than on every replica (partitions): control flow, `clear`,
 * `echo`, and `neb`, which drives all replicas together (neb.html: "a NEB calculation performs a damped
 * dynamics minimization across all the replicas").
 */
const SESSION_LEVEL = new Set(['label', 'jump', 'next', 'include', 'if', 'quit', 'clear', 'echo', 'neb']);

/**
 * The commands whose words can name a style, with the index of the first such word. fix and compute take an
 * ID and a group before the style; every other word of the commands below is a style name, a sub-style of a
 * hybrid style, or a number (pair_coeff's type ranges). suffix and package name accelerator packages, not
 * styles, and the other commands take no style name, so they are not rewritten.
 * *_write (pair_write, bond_write, ...) takes a style name too; the engine does not implement those commands yet.
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

/** The I/O of a replica other than the first: its output is dropped (the first replica's log and screen speak for the session). */
const QUIET_IO: SessionIO = { emit: () => undefined };

export class Session {
  /** The active system: replica 0 between commands, the replica being run while a command runs on all of them. */
  sys: System;
  /** One System per partition (replica); a single entry unless partitions > 1. */
  private replicas: System[] = [];
  private frames: Frame[] = [];
  private skipJump = false;
  /** True while a command runs on the replicas one by one. */
  private onReplica = false;
  /** The words of the neb command being run, as each replica parsed them. */
  private nebWords: string[][] | null = null;
  private cancelled = false;
  private line = 0;
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
    const sys = new System(k === 0 ? this.io : QUIET_IO);
    sys.registries = { ...styleNames(), command: SUPPORTED_COMMANDS.slice() };
    sys.ff.pairThreads = pairThreadsOf(this.backend);
    sys.vars.partition = k;
    return sys;
  }

  /** The systems of all partitions, replica 0 first (the neb command reads and drives them). */
  get replicaSystems(): readonly System[] { return this.replicas; }

  /** Whether frames carry velocities and masses (EngineEvent 'frame', v and mass); off by default. */
  setFrameVelocities(on: boolean): void {
    for (const sys of this.replicas) sys.frameVelocities = on;
  }

  /** Sends an event through the session's output (the neb command's table and log lines). */
  emit(ev: EngineEvent): void { this.io.emit(ev); }

  /** The arguments of the running neb command as replica r parsed them (after $-substitution with its variables). */
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
    if (first === 'neb') {
      // neb runs once, but its arguments are read per replica ($u gives each partition its own value)
      this.nebWords = this.replicas.map((sys) => { this.sys = sys; return this.parse(text); });
      this.sys = this.replicas[0];
      try { return await this.runOne(text, line, frame); } finally { this.nebWords = null; }
    }
    if (first === undefined || SESSION_LEVEL.has(first)) return this.runOne(text, line, frame);
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
    };
  }

  private async dispatch(cmd: string, a: string[], frame: Frame | null): Promise<void> {
    // a jump or next moves one stream for all partitions; with partitions each replica's variables could
    // differ, so the loop controls are refused rather than guessed
    if (this.replicas.length > 1 && (cmd === 'jump' || cmd === 'next')) {
      throw new StyleError(`'${cmd}' is not supported with partitions yet (the replicas share one input stream)`);
    }
    switch (cmd) {
      case 'label':
        if (a.length !== 1) throw new StyleError('usage: label ID');
        return;
      case 'jump': return this.jump(a, frame);
      case 'next':
        if (!a.length) throw new StyleError('usage: next variable1 [variable2 ...]');
        if (this.sys.vars.next(a)) this.skipJump = true;
        return;
      case 'include': {
        if (a.length !== 1) throw new StyleError('usage: include file');
        const text = this.sys.readFile(a[0]);
        await this.runText(text, 1, a[0]);
        return;
      }
      case 'if': return this.ifCommand(a, frame);
      case 'quit': throw new QuitSignal(a.length ? Number(a[0]) : 0);
      case 'clear':
        if (a.length) throw new StyleError('clear takes no arguments');
        this.clear();
        this.io.emit({ kind: 'log', text: 'Cleared: atoms, box, settings, fixes and computes reset (variables kept)' });
        return;
      case 'echo':
        if (!['none', 'screen', 'log', 'both'].includes(a[0] ?? '')) throw new StyleError('usage: echo none|screen|log|both');
        this.echo = a[0] as Session['echo'];
        return;
    }
    // an accelerator alias (lj/cut/omp, nve/kk, ...) runs as its base style: the names are replaced here, before
    // the styles load, so every object is created with the base name and the fast paths see the base style
    a = this.baseStyleNames(cmd, a);
    // styles of the lazily loaded registries (lazyStyles.ts) are loaded before a command that names one;
    // read_restart restores the styles named in its file
    // a lazily loaded command (a COMMANDS name of lazyStyles.ts) is registered by its group before the lookup below
    await ensureStyles([cmd, ...a]);
    const handler = COMMANDS[cmd] ?? LAZY_COMMANDS[cmd];
    if (cmd === 'read_restart' && a.length && this.sys.files.has(normalizeFileName(a[0]))) await ensureStyles(this.sys.files.get(normalizeFileName(a[0]))!.split(/[\s"',:\[\]{}]+/));
    if (!handler) {
      throw new StyleError(`'${cmd}' is not supported by the in-browser engine (it is not LAMMPS). Supported commands: ${SUPPORTED_COMMANDS.join(', ')}`);
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
    if (a.length < 1 || a.length > 2) throw new StyleError('usage: jump file [label]');
    if (this.skipJump) { this.skipJump = false; return; }
    if (!frame) throw new StyleError('jump can only be used in an input script');
    const [file, label] = a;
    if (file !== 'SELF') {
      // "the original file is not returned to": the current stream becomes the new file
      const text = this.sys.readFile(file);
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
    throw new StyleError(`label ${label} not found in ${file === 'SELF' ? 'this cell' : file}`);
  }

  /** if Boolean then t1 t2 ... elif Boolean f1 f2 ... else e1 e2 ... */
  private async ifCommand(a: string[], frame: Frame | null): Promise<void> {
    if (a.length < 2 || a[1] !== 'then') throw new StyleError('usage: if "Boolean" then "command" ... [elif "Boolean" "command" ...] [else "command" ...]');
    const branches: { cond: string | null; cmds: string[] }[] = [{ cond: a[0], cmds: [] }];
    for (let k = 2; k < a.length; k++) {
      if (a[k] === 'elif') {
        if (a[k + 1] === undefined) throw new StyleError('if: elif needs a Boolean expression');
        branches.push({ cond: a[k + 1], cmds: [] });
        k++;
      } else if (a[k] === 'else') branches.push({ cond: null, cmds: [] });
      else branches[branches.length - 1].cmds.push(a[k]);
    }
    // "all variables used will be substituted for before the Boolean expression in evaluated"
    const env = this.sys.formulaEnv;
    const subst = (t: string) => substituteVariables(t, (name) => this.sys.vars.text(name, env),
      (f, fmt) => formatNumber(evaluateScalar(f, env), fmt ?? '%.20g'));
    for (const b of branches) {
      if (b.cond === null || evaluateBoolean(subst(b.cond))) {
        for (const c of b.cmds) {
          await this.runCommand(c, this.line, frame);
          // a jump inside the if moves the enclosing stream; stop executing this branch
          if (frame && /^\s*jump\b/.test(c)) break;
        }
        return;
      }
    }
  }
}

/** Commands the engine runs (recognized-but-unavailable ones such as python or shell excluded). */
export const SUPPORTED_COMMANDS = [
  'label', 'jump', 'next', 'include', 'if', 'quit', 'clear', 'echo',
  ...Object.keys(COMMANDS).filter((c) => !UNAVAILABLE_COMMANDS.has(c)),
].sort();
