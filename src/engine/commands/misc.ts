import type { Handler } from './args';
import { inumeric, logical, missingArgs } from './args';
import { StyleError } from '../force/types';
import { formatNumber, mapUnquoted, substituteVariables } from '../script';
import { evaluateScalar } from '../formula';
import { clearSoftHalt } from '../run/halt';
import { isDirectory, makeDirectories, removeDirectory, shellKey } from '../shellFs';

/*
 * General-purpose input commands.
 */

/**
 * variable name style args — variable.html (see variables.ts). A string variable's text is substituted here, when it
 * is defined: variable.html "string performs variable substitution even if the string parameter is quoted"; as native
 * LAMMPS does (src/variable.cpp Variable::set at b332f28), with Input::substitute's rules (quoted parts kept).
 */
const variable: Handler = ({ sys }, a) => {
  if (a.length < 2) throw new StyleError('Illegal variable command: missing argument(s)');
  if (a[1] === 'string' && a.length === 3) {
    const env = sys.formulaEnv;
    const text = mapUnquoted(a[2], (part) => substituteVariables(part, (n) => sys.vars.text(n, env), (f, fmt) => formatNumber(evaluateScalar(f, env), fmt ?? '%.20g')));
    sys.vars.define(a[0], a[1], [text]);
    return;
  }
  sys.vars.define(a[0], a[1], a.slice(2));
};

/**
 * print string [file f | append f | screen yes|no | universe yes|no] —
 * print.html: "If the text string contains variables, they will be evaluated
 * and their current values printed." Text inside quotes is substituted here
 * (the line was substituted outside quotes when it was read).
 */
const print: Handler = ({ sys }, a) => {
  // the keyword checks of native's Input::print (src/input.cpp at b332f28)
  if (!a.length) throw new StyleError('Illegal print command: missing argument(s)');
  const env = sys.formulaEnv;
  const text = substituteVariables(a[0], (n) => sys.vars.text(n, env), (f, fmt) => formatNumber(evaluateScalar(f, env), fmt ?? '%.20g'));
  let screen = true;
  for (let k = 1; k < a.length; k += 2) {
    const key = a[k], v = a[k + 1];
    if (!['file', 'append', 'screen', 'universe'].includes(key)) throw new StyleError(`Unknown print keyword: ${key}`);
    if (v === undefined) throw new StyleError(`Illegal print ${key} command: missing argument(s)`);
    if (key === 'file') sys.writeFile(v, text + '\n', false);
    else if (key === 'append') sys.writeFile(v, text + '\n', true);
    else if (key === 'screen') screen = logical(v);
    else logical(v);
  }
  if (screen) sys.log(text);
};

/**
 * log file [append] — log.html: "closes the current LAMMPS log file, opens a
 * new file with the specified name, and begins logging information to it. If
 * the specified file name is none, then no new log file is opened."
 */
const log: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/input.cpp Input::log at b332f28)
  if (a.length < 1 || a.length > 2) throw new StyleError(`Illegal log command: expected 1 or 2 argument(s) but found ${a.length}`);
  if (a.length === 2 && a[1] !== 'append') throw new StyleError(`Unknown log keyword: ${a[1]}`);
  sys.setLogFile(a[0] === 'none' ? null : a[0], a[1] === 'append');
};

/** utils::timespec2seconds: off or unlimited (-1), or up to three ':'-separated numbers; null when one is no number. */
const timespecSeconds = (spec: string): number | null => {
  if (spec === 'off' || spec === 'unlimited') return -1;
  const vals = [0, 0, 0];
  const parts = spec.split(':');
  let i = 0;
  for (; i < 3 && i < parts.length; i++) {
    if (!/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(parts[i])) return null;
    vals[i] = Number(parts[i]);
  }
  if (i === 3) return (vals[0] * 60 + vals[1]) * 60 + vals[2];
  if (i === 2) return vals[0] * 60 + vals[1];
  return vals[0];
};

/** timer args — timer.html; timing breakdowns are not collected in the browser, timeout is not supported. */
const timer: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/timer.cpp Timer::modify_params at b332f28): the styles and modes, timeout with a time
  // (utils::timespec2seconds: off, unlimited, or [[HH:]MM:]SS), every with a check frequency (utils::inumeric, > 0); the
  // messages name the keyword, not the value
  for (let k = 0; k < a.length; k++) {
    const w = a[k];
    if (['off', 'loop', 'normal', 'full', 'sync', 'nosync'].includes(w)) continue;
    if (w === 'timeout') {
      if (++k >= a.length) throw missingArgs('timer timeout');
      const t = timespecSeconds(a[k]);
      if (t === null) throw new StyleError(`Illegal timeout time: ${w}`);
      // timer.html: "timer timeout off" and "timer timeout unlimited" clear a timeout; a fix halt with error soft
      // stops later runs until one of them is given (fix_halt.html). Other timeouts need the wall clock: refused.
      if (t >= 0) throw new StyleError('timer timeout is not supported by the browser engine (use the Stop button)');
      clearSoftHalt(sys);
      continue;
    }
    if (w === 'every') {
      if (++k >= a.length) throw missingArgs('timer every');
      // the timeout check frequency; the engine has no timeout to check
      if (inumeric(a[k]) <= 0) throw new StyleError(`Illegal timer every frequency: ${w}`);
      continue;
    }
    throw new StyleError(`Unknown timer keyword ${w}`);
  }
  sys.log('timer: accepted (the browser engine reports only the loop time)');
};

/** info [categories] — info.html; prints what the browser engine knows. */
const info: Handler = ({ sys }, a) => {
  const cats = a.length ? a : ['system', 'computes', 'fixes', 'groups', 'regions', 'variables'];
  const out: string[] = ['Info-Info-Info-Info-Info-Info-Info-Info-Info-Info-Info'];
  for (const c of cats) {
    switch (c) {
      case 'system': case 'all': {
        out.push(`Units = ${sys.units.style}, Atom style = ${sys.atomStyle}, Dimension = ${sys.dimension}`);
        if (sys.hasBox) {
          const s = sys.state;
          out.push(`Atoms = ${s.n}, types = ${s.ntypes}, bonds = ${s.topo.bonds.n}, angles = ${s.topo.angles.n}, dihedrals = ${s.topo.dihedrals.n}, impropers = ${s.topo.impropers.n}`);
          out.push(`Box = (${s.box.lo.join(' ')}) to (${s.box.hi.join(' ')})${s.box.triclinic ? ` tilt (${s.box.tilt.join(' ')})` : ''}, boundary ${s.box.boundary.map((f) => f[0] + f[1]).join(' ')}`);
        } else out.push('No simulation box yet');
        out.push(`Pair style = ${sys.ff.pair?.name ?? 'none'}, Bond style = ${sys.ff.bond?.name ?? 'none'}, Angle style = ${sys.ff.angle?.name ?? 'none'}, Dihedral style = ${sys.ff.dihedral?.name ?? 'none'}, Improper style = ${sys.ff.improper?.name ?? 'none'}, Kspace style = ${sys.ff.kspace?.name ?? 'none'}`);
        if (c === 'system') break;
      }
      // falls through for 'all'
      case 'computes': out.push('Compute information:', ...sys.computes.map((x) => `  Compute ${x.id}: ${x.style} on group ${x.group}`)); break;
      case 'fixes': out.push('Fix information:', ...sys.fixes.map((x) => `  Fix ${x.id}: ${x.style} on group ${x.group}`)); break;
      case 'groups': out.push('Group information:', ...sys.groups.list().map((g) => `  ${g}`)); break;
      case 'regions': out.push('Region information:', ...[...sys.regions.values()].map((r) => `  Region ${r.id}: ${r.style}${r.interior ? '' : ' (side out)'}`)); break;
      case 'variables': out.push('Variable information:', ...[...sys.vars.vars.entries()].map(([k, v]) => `  Variable[${k}]: ${v.style} ${v.formula ?? v.values.join(' ')}`)); break;
      case 'styles': out.push(...Object.entries(sys.registries).map(([k, v]) => `${k}: ${v.join(' ')}`)); break;
      case 'out': case 'screen': case 'log': case 'append': break;
      default: throw new StyleError(`unknown info category '${c}'`);
    }
  }
  sys.log(out.join('\n'));
};

/** Commands that are recognized only to explain why the browser engine cannot run them. */
export const UNAVAILABLE_COMMANDS = new Set<string>();

const browserOnly = (name: string, why: string): Handler => {
  UNAVAILABLE_COMMANDS.add(name);
  return () => {
    throw new StyleError(`${name} ${why}`);
  };
};

/**
 * shell command args — shell.html: "A few simple file-based shell commands are supported directly, in Unix-style
 * syntax."; "*cd* arg = dir", "*mkdir* args = dir1 dir2 ...", "*mv* args = old new", "*rm* args = [-f] file1 file2 ...",
 * "*rmdir* args = dir1 dir2 ...", "*putenv* args = var1=value1 var2=value2"; "anything else is passed as a command to
 * the shell for direct execution". A browser has no operating-system shell, so every other command is refused, naming it.
 * The built-ins act on the notebook's virtual file system (shellFs.ts): cd sets the directory that file names are
 * resolved from (shell.html: "All subsequent LAMMPS commands that read/write files will use the new directory"), mkdir
 * creates the whole path ("similar to the Unix mkdir -p command"), rmdir removes the contents recursively, rm and mv act
 * on the session files, and putenv sets environment variables (LAMMPS_POTENTIALS is searched for a file that is not in
 * the working directory). Like native (src/input.cpp Input::shell at b332f28), a failed cd, mkdir, rmdir, rm, mv or
 * putenv warns and carries on.
 *
 * Measured with native LAMMPS (black box):
 *   shell rm <file>          deletes it, silently.
 *   shell rm <missing>       warns once per missing file and continues; the warning reads
 *                            Shell command 'rm <missing>' failed with error 'No such file or directory'.
 *   shell rm -f <missing>    silent (no warning); also silent with no file names.
 *   shell mv <old> <new>     renames old to new, silently overwriting an existing new.
 *   shell mv <missing> <new> warns and continues; the warning reads
 *                            Shell command 'mv <missing> <new>' failed with error 'No such file or directory'.
 *   shell                    is an error (Illegal shell command: missing argument(s)).
 *   shell rm                 is an error (Illegal shell rm command: missing argument(s)).
 *   shell mv [a]             is an error (expected 3 argument but found N).
 */
const shell: Handler = ({ sys, session }, a) => {
  const sub = a[0];
  if (sub === undefined) throw new StyleError('Illegal shell command: missing argument(s)');
  const NO_SUCH = 'No such file or directory';
  const failed = (what: string, err: string) => sys.warn(`Shell command '${what}' failed with error '${err}'`);
  if (sub === 'cd') {
    // native: narg != 2 is "Illegal shell command: expected 2 argument but found narg" (narg counts the command word)
    if (a.length !== 2) throw new StyleError(`Illegal shell command: expected 2 argument but found ${a.length}`);
    if (!isDirectory(sys, shellKey(sys, a[1]))) { failed(`cd ${a[1]}`, NO_SUCH); return; }
    session.forEachReplica((r) => { r.cwd = shellKey(r, a[1]); });
    return;
  }
  if (sub === 'mkdir') {
    if (a.length < 2) throw new StyleError('Illegal shell mkdir command: missing argument(s)');
    for (const dir of a.slice(1)) {
      if (sys.files.has(shellKey(sys, dir))) { failed(`mkdir ${dir}`, 'File exists'); continue; }
      session.forEachReplica((r) => makeDirectories(r, shellKey(r, dir)));
    }
    return;
  }
  if (sub === 'rmdir') {
    if (a.length < 2) throw new StyleError('shell rmdir: missing argument(s) (usage: shell rmdir dir1 dir2 ...)');
    for (const dir of a.slice(1)) {
      if (!isDirectory(sys, shellKey(sys, dir)) || shellKey(sys, dir) === '') { failed(`rmdir ${dir}`, NO_SUCH); continue; }
      session.forEachReplica((r) => removeDirectory(r, shellKey(r, dir)));
    }
    return;
  }
  if (sub === 'putenv') {
    if (a.length < 2) throw new StyleError('shell putenv: missing argument(s) (usage: shell putenv var1=value1 ...)');
    for (const def of a.slice(1)) {
      const eq = def.indexOf('=');
      if (eq <= 0) { failed(`putenv ${def}`, 'Invalid argument'); continue; }
      const name = def.slice(0, eq), value = def.slice(eq + 1);
      session.forEachReplica((r) => r.env.set(name, value));
    }
    return;
  }
  if (sub === 'rm') {
    let k = 1;
    let force = false;
    if (a[k] === '-f') { force = true; k++; }
    if (k >= a.length && !force) throw new StyleError('shell rm: missing argument(s) (usage: shell rm [-f] file1 file2 ...)');
    for (; k < a.length; k++) {
      const name = a[k];
      if (sys.files.has(shellKey(sys, name))) session.forEachReplica((r) => r.files.delete(shellKey(r, name)));
      else if (!force) failed(`rm ${name}`, NO_SUCH);
    }
    return;
  }
  if (sub === 'mv') {
    if (a.length !== 3) throw new StyleError(`shell mv: expected 2 arguments (old new) but found ${a.length - 1}`);
    const oldName = a[1], newName = a[2];
    const from = shellKey(sys, oldName);
    // native: a destination folder takes the file under its own base name (system("mv old dir"))
    const toDir = isDirectory(sys, shellKey(sys, newName));
    if (!sys.files.has(from)) {
      if (toDir) sys.warn(`Shell command 'mv ${oldName} ${newName}' returned with non-zero status`);
      else failed(`mv ${oldName} ${newName}`, NO_SUCH);
      return;
    }
    const dest = toDir ? `${newName}/${from.split('/').pop()}` : newName;
    let text = '';
    session.forEachReplica((r) => {
      const t = r.files.get(shellKey(r, oldName));
      if (t === undefined) return;
      r.files.delete(shellKey(r, oldName));
      r.files.set(shellKey(r, dest), t);
      text = t;
    });
    // the current replica also reports the file to the notebook, as a write does
    sys.writeFile(dest, text, false);
    return;
  }
  // every other command: the browser has no operating-system shell to pass it to
  throw new StyleError(`shell ${sub}: cannot run 'shell ${a.join(' ')}': a browser has no operating-system shell; only the built-in cd, mkdir, mv, rm, rmdir and putenv are available`);
};

/*
 * box: docs.lammps.org/Commands_removed.html: "The *box* command has been removed and the LAMMPS code
 * changed so it won't be needed.  If present, LAMMPS will ignore the command and print a warning."
 * Measured with native LAMMPS (black box): box tilt large prints WARNING: The 'box' command has been
 * removed and will be ignored, and the input carries on (examples/ELASTIC displace.mod uses it).
 */
const box: Handler = ({ sys }) => {
  sys.warn("The 'box' command has been removed and will be ignored");
};

export const MISC_COMMANDS: Record<string, Handler> = {
  variable, print, log, timer, info, shell, box,
  python: browserOnly('python', 'is not available: the browser engine has no Python interpreter'),
  plugin: browserOnly('plugin', 'is not available: plugins are native shared libraries'),
  mdi: browserOnly('mdi', 'is not available in the browser engine'),
  kim: browserOnly('kim', 'is not available: OpenKIM models are native libraries'),
  // fenix.rst: "This command initializes `Fenix <https://github.com/sandialabs/fenix>`_ for online process recovery.
  // From this point on, Fenix claims a number of ranks to be used as spares." The browser engine runs no MPI ranks.
  fenix: browserOnly('fenix', 'is not available: Fenix recovers failed MPI processes, and the browser engine runs no MPI ranks'),
  geturl: browserOnly('geturl', 'is not available: the notebook only reads files you add to it'),
};
