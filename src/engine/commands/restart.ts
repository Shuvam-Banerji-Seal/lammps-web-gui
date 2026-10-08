import type { Handler } from './args';
import { StyleError } from '../force/types';
import { writeRestartText, readRestartText } from '../output/restart';
import type { System } from '../system';

/** The file name with its (first) '*' wildcard replaced by text. */
export const fillWildcard = (file: string, text: string): string => {
  const i = file.indexOf('*');
  return i < 0 ? file : file.slice(0, i) + text + file.slice(i + 1);
};

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
 * is written by writeRestarts below, called by the run and minimize loops
 * (commands/run.ts). Measured with native LAMMPS (black box): the same file
 * names on the same steps for restart 5 r.*.eq, restart 4 a.rst b.rst,
 * restart v_s v.rst with stride(13,30,7), and restart 3 during a minimization.
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
  return n === 1 ? fillWildcard(file, String(step)) : file;
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
 * restart 0 | restart N root [keywords] | restart N file1 file2 [keywords] — restart.html:
 * "If one filename is specified, a series of filenames will be created which include the timestep
 * in the filename.  If two filenames are specified, only 2 restart files will be created, with
 * those names." "If a single filename is used with no "\*", then the timestep value is
 * appended." "Note that you can specify the restart command twice, once with a single filename
 * and once with two filenames." "Using restart 0 will turn off both modes of output."
 * "Instead of a numeric value, N can be specified as an :doc:`equal-style variable <variable>`"
 * "evaluated at the beginning of a run to determine the next timestep at which a restart file
 * will be written out.  On that timestep, the variable will be evaluated again to determine the
 * next timestep".
 */
const restartCmd: Handler = ({ sys }, a) => {
  const n = a[0];
  if (n === undefined) throw new StyleError('usage: restart N file [file2] [fileper Np | nfile Nf] | restart 0');
  let every = 0;
  let everyVar: string | null = null;
  if (n.startsWith('v_')) everyVar = n.slice(2);
  else {
    every = Number(n);
    if (!Number.isInteger(every) || every < 0) throw new StyleError(`restart: N must be an integer >= 0, got '${n}'`);
  }
  if (every === 0 && everyVar === null) {
    if (a.length !== 1) throw new StyleError('restart 0 takes no other arguments');
    sys.restartOut = [];
    return;
  }
  const rest = a.slice(1);
  let k = 0;
  while (k < rest.length && !KEYWORDS.has(rest[k])) k++;
  if (k < 1 || k > 2) throw new StyleError(`restart ${n}: give one file name, or two (toggled)`);
  const files = rest.slice(0, k);
  const kw = checkKeywords(rest.slice(k), `restart ${n}`);
  for (const f of files) {
    if (f.includes('%')) throw new StyleError(`restart ${f}: '%' writes one file per processor, and the browser engine runs one process; give a single file name`);
    if (f.endsWith('.mpiio')) throw new StyleError(`restart ${f}: MPI-IO restart files are not supported by the browser engine`);
  }
  if (kw) throw new StyleError(`restart ${n}: ${kw} needs a '%' file name, which the browser engine does not write`);
  if (everyVar !== null && !sys.vars.has(everyVar)) throw new StyleError(`restart: variable ${everyVar} does not exist`);
  const mode = files.length === 1 ? 'single' as const : 'toggle' as const;
  sys.restartOut = sys.restartOut.filter((r) => r.mode !== mode);
  sys.restartOut.push({ mode, every, everyVar, files, flip: 0, next: -1 });
};

/** Next restart step of a v_name schedule (restart.html: "the variable should return timestep values"). */
const nextVarStep = (sys: System, name: string): number => {
  const v = Math.floor(sys.equalVariable(name));
  if (!(v > sys.state.step)) throw new StyleError(`restart variable ${name} returned timestep ${v}, not after the current timestep ${sys.state.step}`);
  return v;
};

/** Start of a run or minimization: evaluate the v_name schedules. */
export const startRestarts = (sys: System): void => {
  for (const r of sys.restartOut) if (r.everyVar !== null) r.next = nextVarStep(sys, r.everyVar);
};

/** True if a restart file is due on this step (for the accelerated loop's host steps). */
export const restartDue = (sys: System, step: number): boolean =>
  sys.restartOut.some((r) => (r.everyVar !== null ? step === r.next : step % r.every === 0));

/**
 * Writes the restart files due on this step. restart.html: "Restart files are written on timesteps
 * that are a multiple of N but not on the first timestep of a run or minimization." "A restart
 * file is not written on the last timestep of a run unless it is a multiple of N.  A restart file
 * is written on the last timestep of a minimization if N > 0 and the minimization converges."
 * Callers skip the first step; `force` writes regardless of the schedule (converged minimization).
 */
export const writeRestarts = (sys: System, step: number, force = false): void => {
  for (const r of sys.restartOut) {
    const due = r.everyVar !== null ? step === r.next : step % r.every === 0;
    if (!due && !force) continue;
    let name: string;
    if (r.mode === 'single') name = r.files[0].includes('*') ? fillWildcard(r.files[0], String(step)) : `${r.files[0]}.${step}`;
    else { name = r.files[r.flip]; r.flip = 1 - r.flip; }
    sys.writeFile(name, writeRestartText(sys), false);
    if (r.everyVar !== null && due) r.next = nextVarStep(sys, r.everyVar);
  }
};

export const RESTART_COMMANDS: Record<string, Handler> = {
  write_restart: writeRestart,
  read_restart: readRestart,
  restart: restartCmd,
};
