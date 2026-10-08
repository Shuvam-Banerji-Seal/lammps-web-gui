import { EngineError, type ForceBackend } from './types';
import type { ForceField } from './force/forcefield';
import { System, type SessionIO } from './system';
import { StyleError } from './force/types';
import { mapUnquoted, splitCommands, substituteVariables, tokenize, formatNumber, type RawCommand } from './script';
import { evaluateScalar } from './formula';
import { COMMANDS, UNAVAILABLE_COMMANDS, type Ctx } from './commands';
import { styleNames } from './styles';
import { evaluateBoolean } from './boolean';
import { CpuForceBackend } from './cpu/forces';
import { RunCancelled, QuitSignal } from './errors';

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

export class Session {
  sys: System;
  private frames: Frame[] = [];
  private skipJump = false;
  private cancelled = false;
  private line = 0;
  private command = '';
  echo: 'none' | 'screen' | 'log' | 'both' = 'none';

  constructor(
    private io: SessionIO,
    private backend: ForceBackend = new CpuForceBackend(),
    /** Also emit a 'frame' every this many steps during a run (0 = only at dumps and at the end). */
    readonly frameEvery = 0,
  ) {
    this.sys = this.newSystem();
  }

  private newSystem(): System {
    const sys = new System(this.io);
    sys.registries = { ...styleNames(), command: SUPPORTED_COMMANDS.slice() };
    sys.ff.pairThreads = pairThreadsOf(this.backend);
    return sys;
  }

  /** The current system (null before create_box). */
  get system() { return this.sys.hasBox ? this.sys.state : null; }
  get backendLabel(): string { return this.backend.label; }
  get forceBackend(): ForceBackend { return this.backend; }

  /** Swaps the force backend between runs; the system and settings stay. */
  setBackend(backend: ForceBackend): void {
    this.backend = backend;
    this.sys.ff.pairThreads = pairThreadsOf(backend);
    this.sys.bump();
  }

  /** Requests that a running `run` stop after its current step. */
  cancel(): void { this.cancelled = true; }
  get isCancelled(): boolean { return this.cancelled; }

  /** Makes a file available to read_data, include, jump, potential files. */
  addFile(name: string, text: string): void {
    this.sys.files.set(name, text);
  }

  /** Forgets a file added with addFile (or written by the session). */
  removeFile(name: string): void {
    this.sys.files.delete(name);
  }

  /** clear: a fresh system; "input script variables" survive (clear.html). */
  clear(): void {
    const vars = this.sys.vars;
    const files = this.sys.files;
    this.sys = this.newSystem();
    for (const [k, v] of vars.vars) this.sys.vars.vars.set(k, v);
    this.sys.vars.clear();
    for (const [k, v] of files) this.sys.files.set(k, v);
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

  /** Runs one command (from the stream or from an if / run every). */
  async runCommand(text: string, line: number, frame: Frame | null = this.frames[this.frames.length - 1] ?? null): Promise<void> {
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
    const handler = COMMANDS[cmd];
    if (!handler) {
      throw new StyleError(`'${cmd}' is not supported by the in-browser engine (it is not LAMMPS). Supported commands: ${SUPPORTED_COMMANDS.join(', ')}`);
    }
    await handler(this.ctx(frame), a);
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
