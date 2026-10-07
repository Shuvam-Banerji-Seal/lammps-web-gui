import type { Handler } from './args';
import { yesno } from './args';
import { StyleError } from '../force/types';
import { formatNumber, substituteVariables } from '../script';
import { evaluateScalar } from '../formula';

/*
 * General-purpose input commands.
 */

/** variable name style args — variable.html (see variables.ts). */
const variable: Handler = ({ sys }, a) => {
  if (a.length < 2) throw new StyleError('usage: variable name style args');
  sys.vars.define(a[0], a[1], a.slice(2));
};

/**
 * print string [file f | append f | screen yes|no | universe yes|no] —
 * print.html: "If the text string contains variables, they will be evaluated
 * and their current values printed." Text inside quotes is substituted here
 * (the line was substituted outside quotes when it was read).
 */
const print: Handler = ({ sys }, a) => {
  if (!a.length) throw new StyleError('usage: print "text" [keywords]');
  const env = sys.formulaEnv;
  const text = substituteVariables(a[0], (n) => sys.vars.text(n, env), (f, fmt) => formatNumber(evaluateScalar(f, env), fmt ?? '%.20g'));
  let screen = true;
  for (let k = 1; k < a.length; k += 2) {
    const key = a[k], v = a[k + 1];
    if (v === undefined) throw new StyleError(`print ${key} needs a value`);
    if (key === 'file') sys.writeFile(v, text + '\n', false);
    else if (key === 'append') sys.writeFile(v, text + '\n', true);
    else if (key === 'screen') screen = yesno(v, 'screen');
    else if (key === 'universe') yesno(v, 'universe');
    else throw new StyleError(`unknown print keyword '${key}'`);
  }
  if (screen) sys.log(text);
};

/**
 * log file [append] — log.html: "closes the current LAMMPS log file, opens a
 * new file with the specified name, and begins logging information to it. If
 * the specified file name is none, then no new log file is opened."
 */
const log: Handler = ({ sys }, a) => {
  if (!a[0] || a.length > 2 || (a[1] !== undefined && a[1] !== 'append')) throw new StyleError('usage: log file [append]');
  sys.setLogFile(a[0] === 'none' ? null : a[0], a[1] === 'append');
};

/** timer args — timer.html; timing breakdowns are not collected in the browser, timeout is not supported. */
const timer: Handler = ({ sys }, a) => {
  for (let k = 0; k < a.length; k++) {
    const w = a[k];
    if (['off', 'loop', 'normal', 'full', 'sync', 'nosync'].includes(w)) continue;
    if (w === 'timeout' || w === 'every') throw new StyleError(`timer ${w} is not supported by the browser engine (use the Stop button)`);
    throw new StyleError(`unknown timer keyword '${w}'`);
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

const browserOnly = (name: string, why: string): Handler => () => {
  throw new StyleError(`${name} ${why}`);
};

export const MISC_COMMANDS: Record<string, Handler> = {
  variable, print, log, timer, info,
  shell: browserOnly('shell', 'cannot run: a browser has no operating-system shell'),
  python: browserOnly('python', 'is not available: the browser engine has no Python interpreter'),
  plugin: browserOnly('plugin', 'is not available: plugins are native shared libraries'),
  mdi: browserOnly('mdi', 'is not available in the browser engine'),
  kim: browserOnly('kim', 'is not available: OpenKIM models are native libraries'),
  geturl: browserOnly('geturl', 'is not available: the notebook only reads files you add to it'),
};
