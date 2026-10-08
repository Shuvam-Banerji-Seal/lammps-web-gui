import type { Handler } from './args';
import { StyleError } from '../force/types';
import { writeRestartText, readRestartText } from '../output/restart';
import type { System } from '../system';

/*
 * write_restart, read_restart and restart (wave 9). The file format is the
 * browser engine's own (output/restart.ts): a native binary restart file
 * cannot be read here.
 *
 * write_restart.html: "Write a binary restart file of the current state of the
 * simulation." Filename rule: "If a "\*" appears in the filename, it is
 * replaced with the current timestep value." Names with % write one file per
 * processor; the browser engine runs one process, so those are refused.
 *
 * read_restart.html: "Read in a previously saved system configuration from a
 * restart file." With a "\*" in the name: "the directory is searched for all
 * filenames that match the pattern where "\*" is replaced with a timestep
 * value. The file with the largest timestep value is read in." The comparison
 * is numeric (restart.100 is newer than restart.50).
 *
 * restart.html: "Write out a binary restart file with the current state of the
 * simulation on timesteps which are a multiple of N." "A value of N = 0 means
 * do not write out any restart files, which is the default." Periodic output
 * (N > 0) needs the run loop to write files on those steps; the browser engine
 * has no such hook yet, so it is refused by name.
 *
 * Measured with native LAMMPS (black box), not from the docs: fileper and nfile
 * without a "%" in the name are refused with the message Cannot use
 * write_restart nfile without % in restart file name; read_restart after a box
 * exists is refused with Cannot use read_restart after simulation box is
 * defined; restart N with a file name is accepted by native LAMMPS.
 */

/** Filename with its one wildcard replaced by the timestep (write_restart.html). */
const expandStar = (file: string, step: number): string => {
  const n = file.split('*').length - 1;
  if (n > 1) throw new StyleError(`restart file name '${file}' has ${n} '*' wildcards; use one`);
  return n === 1 ? file.replace('*', String(step)) : file;
};

const KEYWORDS = new Set(['fileper', 'nfile']);

/** Validates write_restart / restart keyword value pairs; returns the first one found (if any). */
const checkKeywords = (args: string[], what: string): string | null => {
  let first: string | null = null;
  for (let k = 0; k < args.length;) {
    const key = args[k];
    if (!KEYWORDS.has(key)) throw new StyleError(`${what}: unknown keyword '${key}' (supported: fileper, nfile)`);
    const v = Number(args[k + 1]);
    if (args[k + 1] === undefined || !Number.isInteger(v) || v < 1) throw new StyleError(`${what}: ${key} needs an integer value >= 1`);
    first ??= key;
    k += 2;
  }
  return first;
};

/** write_restart file [fileper Np | nfile Nf] — write_restart.html. */
const writeRestart: Handler = ({ sys }, a) => {
  if (!a[0]) throw new StyleError('usage: write_restart file [fileper Np | nfile Nf]');
  const file = a[0];
  const kw = checkKeywords(a.slice(1), 'write_restart');
  if (kw && !file.includes('%')) throw new StyleError(`Cannot use write_restart ${kw} without % in restart file name`);
  if (file.includes('%')) {
    throw new StyleError(`write_restart ${file}: '%' writes one file per processor, and the browser engine runs one process; give a single file name`);
  }
  if (file.endsWith('.mpiio')) throw new StyleError(`write_restart ${file}: MPI-IO restart files are not supported by the browser engine`);
  // the state must be current (atoms wrapped, as write_data does before writing)
  sys.pbc();
  sys.nb.lastBuild = -1;
  sys.bump();
  sys.forces();
  const name = expandStar(file, sys.state.step);
  sys.writeFile(name, writeRestartText(sys), false);
  sys.log(`Wrote restart file ${name} at step ${sys.state.step}`);
};

/** The file named by a "*" pattern with the largest timestep among the session's files. */
const latestMatch = (sys: System, pattern: string): string => {
  if (pattern.split('*').length - 1 > 1) throw new StyleError(`read_restart ${pattern}: use one '*' wildcard`);
  const [pre, post] = pattern.split('*');
  const esc = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`^${esc(pre)}(\\d+)${esc(post)}$`);
  let best: { name: string; step: number } | null = null;
  for (const name of sys.files.keys()) {
    const m = re.exec(name);
    if (!m) continue;
    const step = Number(m[1]);
    if (!best || step > best.step) best = { name, step };
  }
  if (!best) throw new StyleError(`read_restart ${pattern}: no file matches the pattern (write one with write_restart first)`);
  return best.name;
};

/** read_restart file — read_restart.html (before any box exists). */
const readRestart: Handler = ({ sys }, a) => {
  if (a.length !== 1) throw new StyleError('usage: read_restart file');
  if (sys.hasBox) throw new StyleError('Cannot use read_restart after simulation box is defined');
  const file = a[0];
  if (file.includes('%')) {
    throw new StyleError(`read_restart ${file}: '%' restart file sets (one file per processor) are not supported by the browser engine`);
  }
  if (file.endsWith('.mpiio')) throw new StyleError(`read_restart ${file}: MPI-IO restart files are not supported by the browser engine`);
  const name = file.includes('*') ? latestMatch(sys, file) : file;
  readRestartText(sys, sys.readFile(name), name);
};

/**
 * restart 0 | restart N root [keywords] | restart N file1 file2 [keywords] —
 * restart.html. Only "restart 0" (off) is implemented; N > 0 is refused until the
 * run loop has a restart hook (see the report: write_restart is the supported path).
 */
const restartCmd: Handler = (_ctx, a) => {
  const n = a[0];
  if (n === undefined) throw new StyleError('usage: restart N file [file2] [fileper Np | nfile Nf] | restart 0');
  if (n.startsWith('v_')) {
    throw new StyleError('restart v_name (a variable-driven restart schedule) is not supported by the browser engine: periodic restart files need a run-loop hook');
  }
  const every = Number(n);
  if (!Number.isInteger(every) || every < 0) throw new StyleError(`restart: N must be an integer >= 0, got '${n}'`);
  if (every === 0) {
    if (a.length !== 1) throw new StyleError('restart 0 takes no other arguments');
    return;
  }
  // syntax first, so a malformed command reports itself rather than the missing hook
  const rest = a.slice(1);
  let k = 0;
  while (k < rest.length && !KEYWORDS.has(rest[k])) k++;
  if (k < 1 || k > 2) throw new StyleError(`restart ${every}: give one file name, or two (toggled)`);
  checkKeywords(rest.slice(k), `restart ${every}`);
  throw new StyleError(`restart ${every}: periodic restart files are not written by the browser engine yet (the run loop has no restart hook); use write_restart before or after a run`);
};

export const RESTART_COMMANDS: Record<string, Handler> = {
  write_restart: writeRestart,
  read_restart: readRestart,
  restart: restartCmd,
};
