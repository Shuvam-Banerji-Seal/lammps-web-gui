import type { System } from '../system';
import type { ThermoRow } from '../types';
import { THERMO_KEYWORDS } from '../types';
import { StyleError } from '../force/types';
import { totalVirial } from '../force/types';
import { massOf } from '../atoms';

/*
 * Thermodynamic output — docs.lammps.org/thermo_style.html and
 * thermo_modify.html.
 *
 * "Style one prints a single line of thermodynamic info that is the
 * equivalent of "thermo_style custom step temp epair emol etotal press"."
 * "Style multi ... is the equivalent of "thermo_style custom etotal ke temp
 * pe ebond eangle edihed eimp evdwl ecoul elong press"."
 * "When you use a "thermo_style" command, all thermodynamic settings are
 * restored to their default values, including those previously set by a
 * thermo_modify command."
 * "The kinetic energy of the system ke is inferred from the temperature of
 * the system with 1/2 k_B T of energy for each degree of freedom."
 * "The etail contribution is included in evdwl, epair, pe, and etotal".
 * "enthalpy = enthalpy (etotal + press*vol)"; "econserve = pe + ke +
 * ecouple". Extensive values (energies) are divided by the number of atoms
 * when thermo_modify norm is yes ("norm = yes for unit style of lj, norm = no
 * for unit style of real and metal"); compute and fix values follow their
 * extensive flag; variables are never normalized.
 * Default computes: "compute thermo_temp all temp", "compute thermo_press
 * all pressure thermo_temp", "compute thermo_pe all pe".
 * Restricted triclinic cell constants (Howto_triclinic.html):
 *   a = lx, b^2 = ly^2 + xy^2, c^2 = lz^2 + xz^2 + yz^2,
 *   cos(alpha) = (xy*xz + ly*yz)/(b*c), cos(beta) = xz/c, cos(gamma) = xy/b.
 */

export const THERMO_ONE = ['step', 'temp', 'epair', 'emol', 'etotal', 'press'];
export const THERMO_MULTI = ['etotal', 'ke', 'temp', 'pe', 'ebond', 'eangle', 'edihed', 'eimp', 'evdwl', 'ecoul', 'elong', 'press'];

/** Keywords that scale with the number of atoms (normalized with norm yes). */
const EXTENSIVE = new Set([
  'pe', 'ke', 'etotal', 'enthalpy', 'evdwl', 'ecoul', 'epair', 'ebond', 'eangle', 'edihed', 'eimp', 'emol',
  'elong', 'etail', 'ecouple', 'econserve',
]);

const BUILTIN = new Set<string>([...THERMO_KEYWORDS, 'cpuuse', 'ecouple', 'econserve',
  'avecx', 'avecy', 'avecz', 'bvecx', 'bvecy', 'bvecz', 'cvecx', 'cvecy', 'cvecz']);

/** LAMMPS column names for the log header. */
const HEADER: Record<string, string> = {
  step: 'Step', elapsed: 'Elapsed', elaplong: 'Elaplong', dt: 'Dt', time: 'Time', cpu: 'CPU', tpcpu: 'T/CPU',
  spcpu: 'S/CPU', cpuremain: 'CPULeft', part: 'Part', timeremain: 'TimeoutLeft', atoms: 'Atoms', temp: 'Temp',
  press: 'Press', pe: 'PotEng', ke: 'KinEng', etotal: 'TotEng', enthalpy: 'Enthalpy', evdwl: 'E_vdwl',
  ecoul: 'E_coul', epair: 'E_pair', ebond: 'E_bond', eangle: 'E_angle', edihed: 'E_dihed', eimp: 'E_impro',
  emol: 'E_mol', elong: 'E_long', etail: 'E_tail', vol: 'Volume', density: 'Density', lx: 'Lx', ly: 'Ly', lz: 'Lz',
  xlo: 'Xlo', xhi: 'Xhi', ylo: 'Ylo', yhi: 'Yhi', zlo: 'Zlo', zhi: 'Zhi', xy: 'Xy', xz: 'Xz', yz: 'Yz',
  xlat: 'Xlat', ylat: 'Ylat', zlat: 'Zlat', bonds: 'Bonds', angles: 'Angles', dihedrals: 'Diheds', impropers: 'Impros',
  pxx: 'Pxx', pyy: 'Pyy', pzz: 'Pzz', pxy: 'Pxy', pxz: 'Pxz', pyz: 'Pyz', fmax: 'Fmax', fnorm: 'Fnorm',
  nbuild: 'Nbuild', ndanger: 'Ndanger', cella: 'Cella', cellb: 'Cellb', cellc: 'Cellc', cellalpha: 'CellAlpha',
  cellbeta: 'CellBeta', cellgamma: 'CellGamma', ecouple: 'Ecouple', econserve: 'Econserve', cpuuse: 'CPUuse',
};

export class Thermo {
  style: 'one' | 'multi' | 'custom' | 'yaml' = 'one';
  keywords: string[] = [...THERMO_ONE];
  /** thermo N (0 = first and last step only) or thermo v_name. */
  every = 0;
  everyVar: string | null = null;
  normUser: boolean | null = null;
  tempId = 'thermo_temp';
  pressId = 'thermo_press';
  lost: 'error' | 'warn' | 'ignore' = 'error';
  lostBond: 'error' | 'warn' | 'ignore' = 'error';
  flush = false;
  /** thermo_modify format: line / int / float / per column. */
  formatLine: string | null = null;
  formatInt: string | null = null;
  formatFloat: string | null = null;
  formatCol = new Map<number, string>();
  lineStyle: 'one' | 'multi' | 'yaml' = 'one';
  /** Last wall time / step for tpcpu, spcpu. */
  private lastCpu = { t: 0, step: 0, time: 0 };

  constructor(private sys: System) {}

  get norm(): boolean {
    return this.normUser ?? this.sys.units.normDefault;
  }

  /** thermo_style ... ; resets thermo_modify settings. */
  setStyle(style: string, args: string[]): void {
    let kw: string[];
    if (style === 'one') kw = [...THERMO_ONE];
    else if (style === 'multi') kw = [...THERMO_MULTI];
    else if (style === 'yaml') kw = [...THERMO_ONE];
    else if (style === 'custom') {
      if (!args.length) throw new StyleError('thermo_style custom needs at least one keyword');
      kw = [];
      for (const a of args) kw.push(...this.expand(a));
    } else throw new StyleError(`unknown thermo_style '${style}'; use one, multi, yaml or custom`);
    if (style !== 'custom' && args.length) throw new StyleError(`thermo_style ${style} takes no arguments`);
    for (const k of kw) this.validate(k);
    this.style = style as Thermo['style'];
    this.keywords = kw;
    this.lineStyle = style === 'multi' ? 'multi' : style === 'yaml' ? 'yaml' : 'one';
    // "all thermodynamic settings are restored to their default values"
    this.normUser = null;
    this.tempId = 'thermo_temp';
    this.pressId = 'thermo_press';
    this.lost = 'error';
    this.lostBond = 'error';
    this.flush = false;
    this.formatLine = this.formatInt = this.formatFloat = null;
    this.formatCol.clear();
  }

  /** c_ID[*] / f_ID[*] wildcards expand to every component. */
  private expand(a: string): string[] {
    const m = /^([cfv])_([A-Za-z0-9_]+)\[(\d*)\*(\d*)\]$/.exec(a);
    if (!m) return [a];
    const [, kind, id, lo, hi] = m;
    let n: number;
    if (kind === 'c') {
      const c = this.sys.compute(id);
      n = c.vectorFlag ? c.sizeVector : c.arrayFlag ? c.sizeArrayCols : 0;
    } else if (kind === 'f') {
      const f = this.sys.fix(id);
      n = f.vectorFlag ? f.sizeVector : 0;
    } else n = this.sys.vars.evalVector(id, this.sys.formulaEnv).length;
    if (!n) throw new StyleError(`${a}: ${kind}_${id} has no vector to expand`);
    const i0 = lo ? Number(lo) : 1, i1 = hi ? Number(hi) : n;
    const out: string[] = [];
    for (let i = i0; i <= i1; i++) out.push(`${kind}_${id}[${i}]`);
    return out;
  }

  private validate(k: string): void {
    if (BUILTIN.has(k)) return;
    if (/^[cfv]_[A-Za-z0-9_]+(\[\d+\]){0,2}$/.test(k)) return;
    throw new StyleError(`unknown thermo keyword '${k}'`);
  }

  /** Checks references before a run. */
  init(): void {
    for (const k of this.keywords) {
      const m = /^([cfv])_([A-Za-z0-9_]+)/.exec(k);
      if (!m) continue;
      if (m[1] === 'c') this.sys.compute(m[2]);
      else if (m[1] === 'f') this.sys.fix(m[2]);
      else if (!this.sys.vars.has(m[2])) throw new StyleError(`thermo keyword ${k}: variable ${m[2]} is not defined`);
    }
    const t = this.sys.compute(this.tempId);
    if (!t.tempFlag) throw new StyleError(`thermo_modify temp: compute ${this.tempId} does not compute a temperature`);
    const p = this.sys.compute(this.pressId);
    if (!p.pressFlag) throw new StyleError(`thermo_modify press: compute ${this.pressId} does not compute a pressure`);
    this.lastCpu = { t: performance.now(), step: this.sys.state.step, time: 0 };
  }

  header(): string[] {
    return this.keywords.map((k) => HEADER[k] ?? k);
  }

  /** The current row (fresh compute values). */
  row(): ThermoRow {
    this.sys.refreshComputes();
    const out: ThermoRow = {};
    for (const k of this.keywords) out[k] = this.value(k);
    return out;
  }

  private natoms(): number {
    return this.sys.state.n;
  }

  private normalize(v: number, extensive: boolean): number {
    if (!extensive || !this.norm) return v;
    const n = this.natoms();
    return n > 0 ? v / n : v;
  }

  /** A thermo keyword for formulas, or undefined if `name` is not a keyword. */
  keyword(name: string): number | undefined {
    if (!BUILTIN.has(name)) return undefined;
    return this.value(name);
  }

  value(k: string): number {
    const m = /^([cfv])_([A-Za-z0-9_]+)(?:\[(\d+)\])?(?:\[(\d+)\])?$/.exec(k);
    if (m) return this.reference(m[1], m[2], m[3] ? Number(m[3]) : null, m[4] ? Number(m[4]) : null);
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const u = s.units;
    const temp = () => sys.compute(this.tempId);
    const press = () => sys.compute(this.pressId);
    const acc = () => sys.forces();
    const vol = () => g.volume(s.dimension);
    const pe = () => sys.compute('thermo_pe').scalarValue();
    const ke = () => { const t = temp(); return 0.5 * t.dof * u.boltz * t.scalarValue(); };
    const n = (v: number) => this.normalize(v, EXTENSIVE.has(k));
    const now = performance.now();
    switch (k) {
      case 'step': return s.step;
      case 'elapsed': return s.step - sys.run.firstStep;
      case 'elaplong': return s.step - sys.run.beginStep;
      case 'dt': return s.dt;
      case 'time': return s.time + (s.step - s.timeStep) * s.dt;
      case 'cpu': return sys.run.inRun ? (now - sys.run.t0) / 1000 : 0;
      case 'tpcpu': case 'spcpu': {
        const dtw = (now - this.lastCpu.t) / 1000;
        const dstep = s.step - this.lastCpu.step;
        this.lastCpu = { t: now, step: s.step, time: 0 };
        if (!(dtw > 0) || dstep <= 0) return 0;
        return k === 'spcpu' ? dstep / dtw : (dstep * s.dt) / dtw;
      }
      case 'cpuremain': {
        if (!sys.run.inRun) return 0;
        const done = s.step - sys.run.firstStep;
        const left = sys.run.lastStep - s.step;
        return done > 0 ? ((now - sys.run.t0) / 1000) * left / done : 0;
      }
      case 'cpuuse': return 100;
      case 'part': return 0;
      case 'timeremain': return 0;
      case 'atoms': return s.n;
      case 'temp': return temp().scalarValue();
      case 'press': return press().scalarValue();
      case 'pe': return n(pe());
      case 'ke': return n(ke());
      case 'etotal': return n(pe() + ke());
      case 'enthalpy': {
        const e = pe() + ke();
        return n(e + press().scalarValue() * vol() / u.nktv2p);
      }
      case 'ecouple': return n(sys.ecouple());
      case 'econserve': return n(pe() + ke() + sys.ecouple());
      case 'evdwl': return n(acc().evdwl);
      case 'ecoul': return n(acc().ecoul);
      case 'epair': { const a = acc(); return n(a.evdwl + a.ecoul + a.elong); }
      case 'ebond': return n(acc().ebond);
      case 'eangle': return n(acc().eangle);
      case 'edihed': return n(acc().edihed);
      case 'eimp': return n(acc().eimp);
      case 'emol': { const a = acc(); return n(a.ebond + a.eangle + a.edihed + a.eimp); }
      case 'elong': return n(acc().elong);
      case 'etail': return n(sys.ff.etailV / vol());
      case 'vol': return vol();
      case 'density': {
        let m = 0;
        for (let i = 0; i < s.n; i++) m += massOf(s, i);
        return (m * u.mv2d) / vol();
      }
      case 'lx': return g.lx;
      case 'ly': return g.ly;
      case 'lz': return g.lz;
      case 'xlo': return s.box.lo[0];
      case 'xhi': return s.box.hi[0];
      case 'ylo': return s.box.lo[1];
      case 'yhi': return s.box.hi[1];
      case 'zlo': return s.box.lo[2];
      case 'zhi': return s.box.hi[2];
      case 'xy': return s.box.tilt[0];
      case 'xz': return s.box.tilt[1];
      case 'yz': return s.box.tilt[2];
      case 'avecx': return g.lx;
      case 'avecy': return 0;
      case 'avecz': return 0;
      case 'bvecx': return g.xy;
      case 'bvecy': return g.ly;
      case 'bvecz': return 0;
      case 'cvecx': return g.xz;
      case 'cvecy': return g.yz;
      case 'cvecz': return g.lz;
      case 'xlat': return sys.lattice?.spacing[0] ?? 1;
      case 'ylat': return sys.lattice?.spacing[1] ?? 1;
      case 'zlat': return sys.lattice?.spacing[2] ?? 1;
      case 'cella': return g.lx;
      case 'cellb': return Math.sqrt(g.ly * g.ly + g.xy * g.xy);
      case 'cellc': return Math.sqrt(g.lz * g.lz + g.xz * g.xz + g.yz * g.yz);
      case 'cellalpha': {
        const b = Math.sqrt(g.ly * g.ly + g.xy * g.xy), c = Math.sqrt(g.lz * g.lz + g.xz * g.xz + g.yz * g.yz);
        return (Math.acos((g.xy * g.xz + g.ly * g.yz) / (b * c)) * 180) / Math.PI;
      }
      case 'cellbeta': {
        const c = Math.sqrt(g.lz * g.lz + g.xz * g.xz + g.yz * g.yz);
        return (Math.acos(g.xz / c) * 180) / Math.PI;
      }
      case 'cellgamma': {
        const b = Math.sqrt(g.ly * g.ly + g.xy * g.xy);
        return (Math.acos(g.xy / b) * 180) / Math.PI;
      }
      case 'pxx': case 'pyy': case 'pzz': case 'pxy': case 'pxz': case 'pyz':
        return press().vectorValues()[['pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz'].indexOf(k)];
      case 'bonds': return s.topo.bonds.n;
      case 'angles': return s.topo.angles.n;
      case 'dihedrals': return s.topo.dihedrals.n;
      case 'impropers': return s.topo.impropers.n;
      case 'fmax': {
        sys.forces();
        let mx = 0;
        for (let i = 0; i < 3 * s.n; i++) mx = Math.max(mx, Math.abs(s.f[i]));
        return mx;
      }
      case 'fnorm': {
        sys.forces();
        let t = 0;
        for (let i = 0; i < 3 * s.n; i++) t += s.f[i] * s.f[i];
        return Math.sqrt(t);
      }
      case 'nbuild': return sys.nb.nbuild;
      case 'ndanger': return sys.nb.ndanger;
    }
    throw new StyleError(`unknown thermo keyword '${k}'`);
  }

  private reference(kind: string, id: string, i: number | null, j: number | null): number {
    const sys = this.sys;
    if (kind === 'v') {
      if (i === null) return sys.equalVariable(id);
      const v = sys.vars.evalVector(id, sys.formulaEnv);
      if (i < 1 || i > v.length) throw new StyleError(`v_${id}[${i}] is out of range`);
      return v[i - 1];
    }
    if (kind === 'c') {
      const c = sys.compute(id);
      if (i === null) return this.normalize(c.scalarValue(), c.extscalar === 1);
      if (j === null) {
        const v = c.vectorValues();
        if (i < 1 || i > v.length) throw new StyleError(`c_${id}[${i}] is out of range (1..${v.length})`);
        const ext = c.extlist ? c.extlist[i - 1] === 1 : c.extvector === 1;
        return this.normalize(v[i - 1], ext);
      }
      const a = c.arrayValues();
      return a[(i - 1) * c.sizeArrayCols + j - 1];
    }
    const f = sys.fix(id);
    if (i === null) return this.normalize(f.computeScalar(), f.extscalar === 1);
    if (j === null) return this.normalize(f.computeVector(i - 1), f.extvector === 1);
    return f.computeArray(i - 1, j - 1);
  }

  /** thermo_modify keyword list. */
  modify(args: string[]): void {
    for (let k = 0; k < args.length;) {
      const key = args[k];
      const val = args[k + 1];
      const need = (w: string | undefined) => {
        if (w === undefined) throw new StyleError(`thermo_modify ${key} needs a value`);
        return w;
      };
      switch (key) {
        case 'norm': this.normUser = yesno(need(val), key); k += 2; break;
        case 'temp': {
          const c = this.sys.compute(need(val));
          if (!c.tempFlag) throw new StyleError(`thermo_modify temp: compute ${val} does not compute a temperature`);
          this.tempId = val!;
          // "a pressure compute defines its own temperature compute as an argument when it is
          // specified. The temp keyword will override this (for the pressure compute being used
          // by thermodynamics), but only if the temp keyword comes after the press keyword."
          const p = this.sys.compute(this.pressId) as unknown as { tempId?: string | null };
          if (p.tempId !== undefined) p.tempId = val!;
          k += 2;
          break;
        }
        case 'press': {
          const c = this.sys.compute(need(val));
          if (!c.pressFlag) throw new StyleError(`thermo_modify press: compute ${val} does not compute a pressure`);
          this.pressId = val!;
          k += 2;
          break;
        }
        case 'lost': case 'lost/bond': {
          const v = need(val);
          if (v !== 'error' && v !== 'warn' && v !== 'ignore') throw new StyleError(`thermo_modify ${key} must be error, warn or ignore`);
          if (key === 'lost') this.lost = v; else this.lostBond = v;
          k += 2;
          break;
        }
        case 'warn': k += 2; break;
        case 'flush': this.flush = yesno(need(val), key); k += 2; break;
        case 'line': {
          const v = need(val);
          if (v !== 'one' && v !== 'multi' && v !== 'yaml') throw new StyleError('thermo_modify line must be one, multi or yaml');
          this.lineStyle = v;
          k += 2;
          break;
        }
        case 'format': {
          const which = need(val);
          const fmt = args[k + 2];
          if (which === 'none') { this.formatLine = this.formatInt = this.formatFloat = null; this.formatCol.clear(); k += 2; break; }
          if (fmt === undefined) throw new StyleError('thermo_modify format needs a format string');
          if (which === 'line') this.formatLine = fmt;
          else if (which === 'int') this.formatInt = fmt;
          else if (which === 'float') this.formatFloat = fmt;
          else if (/^\d+$/.test(which)) this.formatCol.set(Number(which), fmt);
          else throw new StyleError(`thermo_modify format: '${which}' must be line, int, float, a column number, or none`);
          k += 3;
          break;
        }
        case 'every': {
          // thermo_modify every was removed upstream; accept v_name like the thermo command
          throw new StyleError("thermo_modify every is not a LAMMPS keyword; use 'thermo v_name'");
        }
        case 'triclinic/general':
          if (yesno(need(val), key)) throw new StyleError('thermo_modify triclinic/general yes is not supported');
          k += 2;
          break;
        default:
          throw new StyleError(`unknown thermo_modify keyword '${key}'`);
      }
    }
  }

  /** Thermo lines due on this step? */
  due(step: number, firstStep: number, lastStep: number): boolean {
    if (step === firstStep || step === lastStep) return true;
    if (this.everyVar) return false;
    return this.every > 0 && step % this.every === 0;
  }

  /** For thermo v_name: the next output step after `step`. */
  nextVariableStep(): number {
    if (!this.everyVar) return Number.MAX_SAFE_INTEGER;
    return Math.trunc(this.sys.equalVariable(this.everyVar));
  }

  /** Pressure tensor helper (unused virial pieces). */
  virialSum(): Float64Array {
    return totalVirial(this.sys.forces(), new Float64Array(6));
  }
}

const yesno = (w: string, key: string): boolean => {
  if (w !== 'yes' && w !== 'no') throw new StyleError(`thermo_modify ${key} must be yes or no`);
  return w === 'yes';
};
