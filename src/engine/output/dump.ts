import type { System } from '../system';
import { StyleError } from '../force/types';
import { formatNumber } from '../script';
import { localDumpColumns, localColumnSource } from '../compute/local_dump';
import { generalBoxFromRestricted, toGeneralPoint, unrotateVector, type V3 } from '../triclinic_general';
import { hasChargeStyle, hasDipoleStyle, isMolecularStyle, isSphereStyle, hasRmassStyle, isEllipsoidStyle, massOf, CUSTOM_ATTR, customAttr, hasCharge, hasMolecule, nativeOrder } from '../atoms';

/*
 * Per-atom snapshots — docs.lammps.org/dump.html and dump_modify.html.
 *
 * "N = dump on timesteps which are multiples of N"; "If a "*" character
 * appears in the filename, then one file per snapshot is written and the "*"
 * character is replaced with the timestep value." atom style writes
 * "id type xs ys zs" (scaled; "scale = yes"); custom writes the listed
 * attributes; xyz writes "element x y z" with "element = "C" for every atom
 * type" unless dump_modify element is used (native LAMMPS writes the numeric
 * type by default for xyz, as observed: line "1 0 0 0"); defaults "format =
 * %d and %g for each integer or floating point value", "sort = off for dump
 * styles atom, custom, cfg, and local", "sort = id for dump styles dcd, xtc,
 * and xyz", "append = no", "first = no", "pbc = no", "scale = yes",
 * "time = no", "units = no", "unwrap = no".
 * Box bounds (native LAMMPS output): "ITEM: BOX BOUNDS pp pp pp" then
 * "%-1.16e %-1.16e" per dimension; for a triclinic box "ITEM: BOX BOUNDS xy
 * xz yz pp pp pp" with xlo_bound = xlo + MIN(0.0,xy,xz,xy+xz) etc.
 * (Howto_triclinic.html). xyz comment line: " Atoms. Timestep: N".
 * "ITEM: TIME" lines with dump_modify time yes; "ITEM: UNITS" with units yes.
 */

const INT_COLS = new Set(['id', 'mol', 'proc', 'procp1', 'type', 'ix', 'iy', 'iz']);
const ATOM_COLS = new Set([
  'id', 'mol', 'proc', 'procp1', 'type', 'element', 'mass', 'x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu',
  'xsu', 'ysu', 'zsu', 'ix', 'iy', 'iz', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'q',
  // dump.html: "radius,diameter = radius, diameter of spherical particle", "omegax,omegay,omegaz =
  // angular velocity of spherical particle", "tqx,tqy,tqz = torque on finite-size particles"
  'radius', 'diameter', 'omegax', 'omegay', 'omegaz', 'tqx', 'tqy', 'tqz',
  // dump.html: "mux,muy,muz = orientation of dipole moment of atom", "mu = magnitude of dipole moment of atom"
  'mux', 'muy', 'muz', 'mu',
  // dump.html: "The *angmomx*, *angmomy*, and *angmomz* attributes are specific to finite-size aspherical
  // particles that have an angular momentum.  Only the *ellipsoid* atom style defines this quantity."
  'angmomx', 'angmomy', 'angmomz',
]);
const SPHERE_COLS = new Set(['radius', 'diameter', 'omegax', 'omegay', 'omegaz']);
// dump.html: "The *tqx*, *tqy*, and *tqz* attributes are for finite-size particles that can sustain a rotational
// torque due to interactions with other particles."
const TORQUE_COLS = new Set(['tqx', 'tqy', 'tqz']);
const ANGMOM_COLS = new Set(['angmomx', 'angmomy', 'angmomz']);
const DIPOLE_COLS = new Set(['mux', 'muy', 'muz', 'mu']);

/** Compiled C formats (parsing a format per value is slow for large dumps). */
const fmtCache = new Map<string, (v: number) => string>();
export const fmt = (f: string): ((v: number) => string) => {
  let h = fmtCache.get(f);
  if (!h) {
    if (f === '%g') h = (v) => formatNumber(v, '%g');
    else if (f === '%d') h = (v) => String(Math.trunc(v));
    else h = (v) => formatNumber(v, f);
    fmtCache.set(f, h);
  }
  return h;
};

export type DumpStyle = 'atom' | 'custom' | 'xyz' | 'extxyz' | 'yaml' | 'local';

/** Columns dump_modify triclinic/general rotates (dump.html: "vx,vy,vz = atom velocities" and the others listed there). */
const GENERAL_ROTATED = /^(x|y|z|xu|yu|zu|vx|vy|vz|fx|fy|fz)$/;
/** Per-atom vector columns the browser engine does not rotate; they stop the dump rather than write restricted values. */
const GENERAL_UNSUPPORTED = /^(mu[xyz]|omega[xyz]|angmom[xyz]|tq[xyz]|sp[xyz]|quat[wijk]|shape[xyz])$/;

export class Dump {
  readonly groupBit: number;
  every: number;
  everyVar: string | null = null;
  append = false;
  first = false;
  delay = 0;
  pbc = false;
  scale = true;
  image = false;
  unwrap = false;
  /** dump_modify triclinic/general: write the box and per-atom vectors in the general triclinic frame. */
  triclinicGeneral = false;
  time = false;
  units = false;
  header = true;
  sort: 'off' | 'id' | { col: number; desc: boolean };
  element: string[] = [];
  formatLine: string | null = null;
  formatInt = '%d';
  formatFloat = '%g';
  formatCol = new Map<number, string>();
  colnames = new Map<number, string>();
  region: string | null = null;
  thresh: { col: string; op: string; value: number | 'LAST'; last?: Map<number, number> }[] = [];
  pad = 0;
  /** dump_modify label (dump local): the word in the header lines, default ENTRIES. */
  label = 'ENTRIES';
  lastStep = -1;
  private opened = false;
  private nextStep = -1;
  readonly columns: string[];

  constructor(private sys: System, readonly id: string, readonly group: string, readonly style: DumpStyle, every: number | string, readonly file: string, cols: string[]) {
    this.groupBit = sys.groupBit(group);
    if (typeof every === 'string') this.everyVar = every;
    this.every = typeof every === 'number' ? every : 0;
    this.sort = style === 'xyz' || style === 'extxyz' ? 'id' : 'off';
    if (style === 'atom') {
      if (cols.length) throw new StyleError('dump atom takes no attributes (use dump custom)');
      this.columns = ['id', 'type', 'xs', 'ys', 'zs'];
    } else if (style === 'xyz' || style === 'extxyz') {
      if (cols.length) throw new StyleError(`dump ${style} takes no attributes`);
      this.columns = ['type', 'x', 'y', 'z'];
    } else if (style === 'local') {
      this.columns = localDumpColumns(sys, id, cols);
    } else {
      if (!cols.length) throw new StyleError('dump custom needs a list of attributes, e.g. id type x y z');
      const out: string[] = [];
      for (const c of cols) out.push(...this.expand(c));
      for (const c of out) this.validate(c);
      this.columns = out;
    }
  }

  private expand(c: string): string[] {
    // i2_name[*] / d2_name[*]: every column of a custom array (dump.html "I can include wildcard")
    const cm = /^([id])2_([A-Za-z0-9_]+)\[\*\]$/.exec(c);
    if (cm) {
      const cp = this.sys.state.custom.get(cm[2]);
      if (!cp || cp.cols === 0) throw new StyleError(`dump ${this.id}: custom per-atom array ${c} does not exist`);
      return Array.from({ length: cp.cols }, (_, k) => `${cm[1]}2_${cm[2]}[${k + 1}]`);
    }
    const m = /^([cf])_([A-Za-z0-9_]+)\[(\d*)\*(\d*)\]$/.exec(c);
    if (!m) return [c];
    const [, kind, id, lo, hi] = m;
    const n = kind === 'c' ? this.sys.compute(id).sizePeratomCols : this.sys.fix(id).sizePeratomCols;
    if (!n) throw new StyleError(`dump ${this.id}: ${c} has no per-atom array to expand`);
    const out: string[] = [];
    for (let k = lo ? Number(lo) : 1; k <= (hi ? Number(hi) : n); k++) out.push(`${kind}_${id}[${k}]`);
    return out;
  }

  private validate(c: string): void {
    if (ATOM_COLS.has(c)) {
      const st = this.sys.atomStyle;
      const sh = this.sys.hasBox ? this.sys.state : null;
      if (c === 'q' && !(sh ? hasCharge(sh) : hasChargeStyle(st))) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (q needs atom_style charge or full)`);
      if (SPHERE_COLS.has(c) && !isSphereStyle(st)) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (${c} needs atom_style sphere)`);
      if (TORQUE_COLS.has(c) && !hasRmassStyle(st)) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (${c} needs atom_style sphere or ellipsoid)`);
      if (ANGMOM_COLS.has(c) && !isEllipsoidStyle(st)) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (${c} needs atom_style ellipsoid)`);
      if (DIPOLE_COLS.has(c) && !hasDipoleStyle(st)) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (${c} needs atom_style dipole)`);
      if (c === 'mol' && !(sh ? hasMolecule(sh) : isMolecularStyle(st))) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (mol needs a molecular atom_style)`);
      return;
    }
    if (CUSTOM_ATTR.test(c)) {
      customAttr(this.sys.state, c);
      return;
    }
    const m = /^([cfv])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(c);
    if (!m) throw new StyleError(`invalid dump custom attribute '${c}'`);
    if (m[1] === 'c') {
      const cp = this.sys.compute(m[2]);
      if (!cp.peratomFlag) throw new StyleError(`dump ${this.id}: compute ${m[2]} does not compute per-atom info`);
    } else if (m[1] === 'f') {
      const f = this.sys.fix(m[2]);
      if (!f.peratomFlag) throw new StyleError(`dump ${this.id}: fix ${m[2]} does not compute per-atom info`);
    } else {
      const v = this.sys.vars.get(m[2]);
      if (v && v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`dump ${this.id}: variable ${m[2]} is not atom-style`);
    }
  }

  /** dump_modify keywords. */
  modify(args: string[]): void {
    const yesno = (w: string | undefined, key: string) => {
      if (w !== 'yes' && w !== 'no') throw new StyleError(`dump_modify ${key} must be yes or no`);
      return w === 'yes';
    };
    for (let k = 0; k < args.length;) {
      const key = args[k];
      const v = args[k + 1];
      switch (key) {
        case 'append': this.append = yesno(v, key); k += 2; break;
        case 'first': this.first = yesno(v, key); k += 2; break;
        case 'pbc': this.pbc = yesno(v, key); k += 2; break;
        case 'scale': this.scale = yesno(v, key); this.applyScaleImage(); k += 2; break;
        case 'image': this.image = yesno(v, key); this.applyScaleImage(); k += 2; break;
        case 'unwrap': this.unwrap = yesno(v, key); k += 2; break;
        case 'time': this.time = yesno(v, key); k += 2; break;
        case 'units': this.units = yesno(v, key); k += 2; break;
        case 'header': this.header = yesno(v, key); k += 2; break;
        case 'flush': case 'buffer': case 'balance': case 'thermo': yesno(v, key); k += 2; break;
        case 'delay': this.delay = intArg(v, key); k += 2; break;
        case 'pad': this.pad = intArg(v, key); k += 2; break;
        case 'every': {
          if (v?.startsWith('v_')) { this.everyVar = v.slice(2); this.every = 0; } else {
            this.every = intArg(v, key);
            if (this.every < 1) throw new StyleError('dump_modify every must be > 0');
            this.everyVar = null;
          }
          this.nextStep = -1;
          k += 2;
          break;
        }
        case 'sort': {
          if (v === 'off') this.sort = 'off';
          else if (v === 'id') this.sort = 'id';
          else {
            const n = intArg(v, key);
            if (n === 0 || Math.abs(n) > this.columns.length) throw new StyleError('dump_modify sort column out of range');
            this.sort = { col: Math.abs(n) - 1, desc: n < 0 };
          }
          k += 2;
          break;
        }
        case 'element': {
          const n = this.sys.state.ntypes;
          const el = args.slice(k + 1, k + 1 + n);
          if (el.length !== n) throw new StyleError(`dump_modify element needs ${n} element names (one per atom type)`);
          this.element = el;
          k += 1 + n;
          break;
        }
        case 'format': {
          if (v === 'none') { this.formatLine = null; this.formatInt = '%d'; this.formatFloat = '%g'; this.formatCol.clear(); k += 2; break; }
          const f = args[k + 2];
          if (f === undefined) throw new StyleError('dump_modify format needs a format string');
          if (v === 'line') this.formatLine = f;
          else if (v === 'int') this.formatInt = f;
          else if (v === 'float') this.formatFloat = f;
          else {
            const col = this.columnIndex(v ?? '');
            this.formatCol.set(col, f);
          }
          k += 3;
          break;
        }
        case 'colname': {
          if (v === 'default') { this.colnames.clear(); k += 2; break; }
          const name = args[k + 2];
          if (name === undefined) throw new StyleError('dump_modify colname needs a column and a name');
          this.colnames.set(this.columnIndex(v ?? ''), name);
          k += 3;
          break;
        }
        case 'region': this.region = v === 'none' ? null : (this.sys.region(v ?? ''), v!); k += 2; break;
        case 'thresh': {
          if (v === 'none') { this.thresh = []; k += 2; break; }
          const op = args[k + 2], val = args[k + 3];
          if (!['<', '<=', '>', '>=', '==', '!=', '|^'].includes(op ?? '')) throw new StyleError(`dump_modify thresh: invalid operator '${op ?? ''}'`);
          this.thresh.push({ col: v!, op: op!, value: val === 'LAST' ? 'LAST' : numArg(val, key) });
          k += 4;
          break;
        }
        case 'label': if (!v) throw new StyleError('dump_modify label needs a string'); this.label = v; k += 2; break;
        case 'precision': case 'sfactor': case 'tfactor': case 'maxfiles': case 'nfile': case 'fileper': case 'at': numArg(v, key); k += 2; break;
        case 'triclinic/general': this.triclinicGeneral = yesno(v, key); k += 2; break;
        case 'types': if (v !== 'numeric') throw new StyleError('dump_modify types labels needs type labels (not supported)'); k += 2; break;
        default:
          throw new StyleError(`unsupported dump_modify keyword '${key}'`);
      }
    }
  }

  private applyScaleImage(): void {
    if (this.style !== 'atom') return;
    const c = this.scale ? ['id', 'type', 'xs', 'ys', 'zs'] : ['id', 'type', 'x', 'y', 'z'];
    if (this.image) c.push('ix', 'iy', 'iz');
    (this as unknown as { columns: string[] }).columns = c;
  }

  private columnIndex(w: string): number {
    if (/^-?\d+$/.test(w)) {
      const n = Number(w);
      const k = n > 0 ? n - 1 : this.columns.length + n;
      if (k < 0 || k >= this.columns.length) throw new StyleError(`dump_modify: column ${w} out of range`);
      return k;
    }
    const k = this.columns.indexOf(w);
    if (k < 0) throw new StyleError(`dump_modify: '${w}' is not a column of dump ${this.id}`);
    return k;
  }

  /** Should a snapshot be written on this step? */
  due(step: number, firstStepOfRun: boolean): boolean {
    if (step < this.delay) return false;
    if (this.lastStep === step) return false;
    if (firstStepOfRun && this.first) return true;
    if (this.everyVar) {
      if (this.nextStep < 0) this.nextStep = Math.trunc(this.sys.equalVariable(this.everyVar));
      return step === this.nextStep;
    }
    return step % this.every === 0;
  }

  write(): void {
    const sys = this.sys;
    const s = sys.state;
    sys.forces();
    const step = s.step;
    this.lastStep = step;
    if (this.everyVar) this.nextStep = Math.trunc(sys.equalVariable(this.everyVar));
    if (this.style === 'local') {
      this.writeLocal(step);
      return;
    }
    const g = sys.geom;
    // dump_modify.html: "It can only be used with a value of *yes* if the" simulation box was created as a general
    // triclinic box. Only the atom and custom styles take it (Howto_triclinic.html lists dump atom, dump custom).
    if (this.triclinicGeneral) {
      if (!s.box.general) throw new StyleError(`dump ${this.id}: dump_modify triclinic/general yes needs a general triclinic box (create_box NULL, or a general triclinic data file)`);
      if (this.style !== 'atom' && this.style !== 'custom') throw new StyleError(`dump ${this.id}: dump_modify triclinic/general applies only to the atom and custom styles`);
    }
    // atoms to write
    const reg = this.region ? sys.region(this.region) : null;
    const rows: number[] = [];
    // unsorted dumps list atoms in native LAMMPS's storage order (SimState.order)
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (reg && !reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
      rows.push(i);
    }
    const values = this.columns.map((c) => this.columnValues(c));
    let keep = rows;
    for (const t of this.thresh) {
      const col = this.columnValues(t.col);
      keep = keep.filter((i) => {
        const ref = t.value === 'LAST' ? (t.last?.get(s.id[i]) ?? col[i]) : t.value;
        const x = col[i];
        switch (t.op) {
          case '<': return x < ref;
          case '<=': return x <= ref;
          case '>': return x > ref;
          case '>=': return x >= ref;
          case '==': return x === ref;
          case '!=': return x !== ref;
          default: return (x !== 0) !== (ref !== 0);
        }
      });
      if (t.value === 'LAST') {
        t.last = new Map();
        for (const i of keep) t.last.set(s.id[i], col[i]);
      }
    }
    if (this.sort === 'id') keep.sort((a, b) => s.id[a] - s.id[b]);
    else if (this.sort !== 'off') {
      const { col, desc } = this.sort;
      const v = values[col];
      keep.sort((a, b) => (desc ? v[b] - v[a] : v[a] - v[b]));
    }
    const lines: string[] = [];
    const time = s.time + (s.step - s.timeStep) * s.dt;
    if (this.style === 'xyz' || this.style === 'extxyz') {
      lines.push(String(keep.length));
      if (this.style === 'extxyz') {
        lines.push(`Lattice="${g.lx} 0 0 ${g.xy} ${g.ly} 0 ${g.xz} ${g.yz} ${g.lz}" Origin="${s.box.lo.join(' ')}" Properties=species:S:1:pos:R:3 Timestep=${step}${this.time ? ` Time=${time}` : ''}`);
      } else lines.push(` Atoms. Timestep: ${step}${this.time ? ` Time: ${formatNumber(time, '%g')}` : ''}`);
      const fx = fmt(this.formatFloat);
      for (const i of keep) {
        const t = s.type[i];
        const el = this.element.length ? this.element[t - 1] : String(t);
        lines.push(`${el} ${fx(s.x[3 * i])} ${fx(s.x[3 * i + 1])} ${fx(s.x[3 * i + 2])}`);
      }
    } else if (this.style === 'yaml') {
      lines.push('---', `creator: LAMMPS web notebook`, `timestep: ${step}`, `natoms: ${keep.length}`);
      lines.push(`boundary: [ ${s.box.boundary.flat().join(', ')} ]`);
      lines.push('box:', ...[0, 1, 2].map((d) => `  - [ ${s.box.lo[d]}, ${s.box.hi[d]} ]`));
      lines.push(`keywords: [ ${this.columns.map((c, k) => this.colnames.get(k) ?? c).join(', ')} ]`, 'data:');
      for (const i of keep) lines.push(`  - [ ${values.map((v) => String(v[i])).join(', ')} ]`);
      lines.push('...');
    } else {
      if (this.units && !this.opened) lines.push('ITEM: UNITS', s.units.style);
      if (this.time) lines.push('ITEM: TIME', formatNumber(time, '%.16g'));
      if (this.header) {
        lines.push('ITEM: TIMESTEP', String(step), 'ITEM: NUMBER OF ATOMS', String(keep.length));
        const bnd = s.box.boundary.map((f) => f[0] + f[1]).join(' ');
        const e = (v: number) => formatNumber(v, '%-1.16e');
        if (this.triclinicGeneral && s.box.general) {
          // dump.html: "ITEM: BOX BOUNDS abc origin" then "ax ay az originx", "bx by bz originy", "cx cy cz originz"
          const gb = generalBoxFromRestricted(s.box.general.Q, s.box.lo, s.box.hi, s.box.tilt);
          const o = s.box.lo;
          lines.push(`ITEM: BOX BOUNDS abc origin ${bnd}`, `${e(gb.A[0])} ${e(gb.A[1])} ${e(gb.A[2])} ${e(o[0])}`, `${e(gb.B[0])} ${e(gb.B[1])} ${e(gb.B[2])} ${e(o[1])}`, `${e(gb.C[0])} ${e(gb.C[1])} ${e(gb.C[2])} ${e(o[2])}`);
        } else if (g.triclinic) {
          const [xy, xz, yz] = s.box.tilt;
          const xlob = s.box.lo[0] + Math.min(0, xy, xz, xy + xz), xhib = s.box.hi[0] + Math.max(0, xy, xz, xy + xz);
          const ylob = s.box.lo[1] + Math.min(0, yz), yhib = s.box.hi[1] + Math.max(0, yz);
          lines.push(`ITEM: BOX BOUNDS xy xz yz ${bnd}`, `${e(xlob)} ${e(xhib)} ${e(xy)}`, `${e(ylob)} ${e(yhib)} ${e(xz)}`, `${e(s.box.lo[2])} ${e(s.box.hi[2])} ${e(yz)}`);
        } else {
          lines.push(`ITEM: BOX BOUNDS ${bnd}`, ...[0, 1, 2].map((d) => `${e(s.box.lo[d])} ${e(s.box.hi[d])}`));
        }
        lines.push(`ITEM: ATOMS ${this.columns.map((c, k) => this.colnames.get(k) ?? c).join(' ')}`);
      }
      const formatters = this.columns.map((c, k) => fmt(this.formatCol.get(k) ?? (this.isInt(c) ? this.formatInt : this.formatFloat)));
      const line = this.formatLine ? this.formatLine.split(/\s+/).filter(Boolean) : null;
      // dump_modify.html, Default section: the element name is C for every atom type without dump_modify element
      // (measured the same with native LAMMPS for dump custom ... element); the column is the name of the atom's type
      const elementName = (i: number): string => (this.element.length ? this.element[s.type[i] - 1] : 'C');
      for (const i of keep) {
        lines.push(this.columns.map((c, k) => (c === 'element' ? elementName(i) : line?.[k] ? fmt(line[k])(values[k][i]) : formatters[k](values[k][i]))).join(' '));
      }
    }
    const multi = this.file.includes('*');
    const stepText = this.pad > 0 ? String(step).padStart(this.pad, '0') : String(step);
    const name = multi ? this.file.replace('*', stepText) : this.file;
    const appendNow = multi ? false : this.opened || this.append;
    sys.writeFile(name, lines.join('\n') + '\n', appendNow);
    this.opened = true;
  }

  /**
   * dump local: "ITEM: TIMESTEP", "ITEM: NUMBER OF ENTRIES", the box, "ITEM: ENTRIES ..." (the word
   * ENTRIES is replaced by dump_modify label), then one line per local datum. Measured with native
   * LAMMPS (black box): every value, the index included, is followed by one space; dump_modify header
   * no leaves out all header lines.
   */
  private writeLocal(step: number): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    if (this.thresh.length || this.region) throw new StyleError(`dump ${this.id} local: thresh and region are not supported by the browser engine`);
    const srcs = this.columns.map((c) => localColumnSource(sys, c));
    let nrows = -1;
    for (const c of this.columns) {
      if (c === 'index') continue;
      const m = /^c_([A-Za-z0-9_]+)/.exec(c)!;
      const r = sys.compute(m[1]).localRows;
      if (nrows >= 0 && r !== nrows) throw new StyleError(`dump ${this.id} local: the computes do not have the same number of local entries (${nrows} and ${r})`);
      nrows = r;
    }
    if (nrows < 0) nrows = 0;
    const lines: string[] = [];
    if (this.header) {
      lines.push('ITEM: TIMESTEP', String(step), `ITEM: NUMBER OF ${this.label}`, String(nrows));
      const bnd = s.box.boundary.map((f) => f[0] + f[1]).join(' ');
      const e = (v: number) => formatNumber(v, '%-1.16e');
      if (g.triclinic) {
        const [xy, xz, yz] = s.box.tilt;
        const xlob = s.box.lo[0] + Math.min(0, xy, xz, xy + xz), xhib = s.box.hi[0] + Math.max(0, xy, xz, xy + xz);
        const ylob = s.box.lo[1] + Math.min(0, yz), yhib = s.box.hi[1] + Math.max(0, yz);
        lines.push(`ITEM: BOX BOUNDS xy xz yz ${bnd}`, `${e(xlob)} ${e(xhib)} ${e(xy)}`, `${e(ylob)} ${e(yhib)} ${e(xz)}`, `${e(s.box.lo[2])} ${e(s.box.hi[2])} ${e(yz)}`);
      } else {
        lines.push(`ITEM: BOX BOUNDS ${bnd}`, ...[0, 1, 2].map((d) => `${e(s.box.lo[d])} ${e(s.box.hi[d])}`));
      }
      lines.push(`ITEM: ${this.label} ${this.columns.map((c, k) => this.colnames.get(k) ?? c).join(' ')}`);
    }
    const formatters = this.columns.map((c, k) => fmt(this.formatCol.get(k) ?? (c === 'index' ? this.formatInt : this.formatFloat)));
    const line = this.formatLine ? this.formatLine.split(/\s+/).filter(Boolean) : null;
    for (let r = 0; r < nrows; r++) {
      let text = '';
      this.columns.forEach((_, k) => {
        const src = srcs[k];
        const v = src ? src.data[r * src.ncol + src.col] : r + 1;
        text += (line?.[k] ? fmt(line[k])(v) : formatters[k](v)) + ' ';
      });
      lines.push(text);
    }
    const multi = this.file.includes('*');
    const stepText = this.pad > 0 ? String(step).padStart(this.pad, '0') : String(step);
    const name = multi ? this.file.replace('*', stepText) : this.file;
    const appendNow = multi ? false : this.opened || this.append;
    sys.writeFile(name, lines.join('\n') + '\n', appendNow);
    this.opened = true;
  }

  private isInt(c: string): boolean {
    return INT_COLS.has(c) || /^i2?_/.test(c);
  }

  /** Values of one column for every atom index (length n). */
  private columnValues(c: string): Float64Array {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const out = new Float64Array(s.n);
    const lam = [0, 0, 0];
    const u = [0, 0, 0];
    if (this.triclinicGeneral && GENERAL_UNSUPPORTED.test(c)) {
      throw new StyleError(`dump ${this.id}: dump_modify triclinic/general does not rotate column ${c} (the browser engine rotates x, y, z, xu, yu, zu, vx, vy, vz, fx, fy, fz)`);
    }
    const pos = (i: number, d: number) => {
      if (!this.pbc) return s.x[3 * i + d];
      const x = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
      const im = [s.image[3 * i], s.image[3 * i + 1], s.image[3 * i + 2]];
      const xx = Float64Array.from(x);
      const ii = Int32Array.from(im);
      g.remap(xx, ii, 0);
      return xx[d];
    };
    // dump_modify triclinic/general: positions (x, xu: about the box origin), velocities and forces are rotated
    // into the general frame (dump.html; Howto_triclinic.html). The rotation is the stored one (SimBox.general).
    if (this.triclinicGeneral && s.box.general && GENERAL_ROTATED.test(c)) {
      const Q = s.box.general.Q;
      const o: V3 = [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
      const d = 'xyz'.indexOf(c[c.length - 1]);
      for (let i = 0; i < s.n; i++) {
        let w: V3;
        if (c[0] === 'v') w = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
        else if (c[0] === 'f') w = [s.f[3 * i], s.f[3 * i + 1], s.f[3 * i + 2]];
        else if (c.endsWith('u') || this.unwrap) { g.unwrap(s.x, s.image, i, u); w = [u[0], u[1], u[2]]; } else w = [pos(i, 0), pos(i, 1), pos(i, 2)];
        out[i] = c[0] === 'v' || c[0] === 'f' ? unrotateVector(Q, w)[d] : toGeneralPoint(Q, o, w)[d];
      }
      return out;
    }
    switch (c) {
      case 'id': for (let i = 0; i < s.n; i++) out[i] = s.id[i]; return out;
      case 'mol': for (let i = 0; i < s.n; i++) out[i] = s.molecule[i]; return out;
      case 'proc': return out;
      case 'procp1': return out.fill(1);
      case 'type': for (let i = 0; i < s.n; i++) out[i] = s.type[i]; return out;
      // the element column is written as text (writeCustom); its numeric stand-in is the type
      case 'element': for (let i = 0; i < s.n; i++) out[i] = s.type[i]; return out;
      case 'mass': for (let i = 0; i < s.n; i++) out[i] = massOf(s, i); return out;
      case 'q': for (let i = 0; i < s.n; i++) out[i] = s.q[i]; return out;
      case 'x': case 'y': case 'z': {
        const d = 'xyz'.indexOf(c);
        for (let i = 0; i < s.n; i++) {
          if (this.unwrap) { g.unwrap(s.x, s.image, i, u); out[i] = u[d]; } else out[i] = pos(i, d);
        }
        return out;
      }
      case 'xs': case 'ys': case 'zs': {
        const d = 'xyz'.indexOf(c[0]);
        for (let i = 0; i < s.n; i++) { g.toLamda(pos(i, 0), pos(i, 1), pos(i, 2), lam); out[i] = lam[d]; }
        return out;
      }
      case 'xu': case 'yu': case 'zu': {
        const d = 'xyz'.indexOf(c[0]);
        for (let i = 0; i < s.n; i++) { g.unwrap(s.x, s.image, i, u); out[i] = u[d]; }
        return out;
      }
      case 'xsu': case 'ysu': case 'zsu': {
        const d = 'xyz'.indexOf(c[0]);
        for (let i = 0; i < s.n; i++) {
          g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
          out[i] = lam[d] + s.image[3 * i + d];
        }
        return out;
      }
      case 'ix': case 'iy': case 'iz': {
        const d = 'xyz'.indexOf(c[1]);
        for (let i = 0; i < s.n; i++) out[i] = s.image[3 * i + d];
        return out;
      }
      case 'vx': case 'vy': case 'vz': {
        const d = 'xyz'.indexOf(c[1]);
        for (let i = 0; i < s.n; i++) out[i] = s.v[3 * i + d];
        return out;
      }
      case 'fx': case 'fy': case 'fz': {
        const d = 'xyz'.indexOf(c[1]);
        for (let i = 0; i < s.n; i++) out[i] = s.f[3 * i + d];
        return out;
      }
      case 'radius': case 'diameter': {
        const f = c === 'radius' ? 1 : 2;
        for (let i = 0; i < s.n; i++) out[i] = f * s.radius![i];
        return out;
      }
      case 'omegax': case 'omegay': case 'omegaz': {
        const d = 'xyz'.indexOf(c[5]);
        for (let i = 0; i < s.n; i++) out[i] = s.omega![3 * i + d];
        return out;
      }
      case 'tqx': case 'tqy': case 'tqz': {
        const d = 'xyz'.indexOf(c[2]);
        for (let i = 0; i < s.n; i++) out[i] = s.torque![3 * i + d];
        return out;
      }
      case 'angmomx': case 'angmomy': case 'angmomz': {
        const d = 'xyz'.indexOf(c[6]);
        for (let i = 0; i < s.n; i++) out[i] = s.angmom![3 * i + d];
        return out;
      }
      case 'mux': case 'muy': case 'muz': case 'mu': {
        const d = c === 'mu' ? 3 : 'xyz'.indexOf(c[2]);
        for (let i = 0; i < s.n; i++) out[i] = s.mu![4 * i + d];
        return out;
      }
    }
    const getter = customAttr(s, c);
    if (getter) { for (let i = 0; i < s.n; i++) out[i] = getter(i); return out; }
    const m = /^([cfv])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(c);
    if (!m) throw new StyleError(`invalid dump attribute '${c}'`);
    const col = m[3] ? Number(m[3]) : 0;
    if (m[1] === 'v') return sys.atomVariable(m[2]);
    const cols = m[1] === 'c' ? sys.compute(m[2]).sizePeratomCols : sys.fix(m[2]).sizePeratomCols;
    const vals = m[1] === 'c' ? sys.compute(m[2]).peratomValues() : (cols ? sys.fix(m[2]).arrayAtom : sys.fix(m[2]).vectorAtom);
    if (cols === 0) { out.set(vals.subarray(0, s.n)); return out; }
    if (col < 1 || col > cols) throw new StyleError(`dump attribute ${c}: column out of range 1..${cols}`);
    for (let i = 0; i < s.n; i++) out[i] = vals[i * cols + col - 1];
    return out;
  }
}

const intArg = (w: string | undefined, key: string): number => {
  const n = Number(w);
  if (w === undefined || !Number.isInteger(n)) throw new StyleError(`dump_modify ${key} needs an integer`);
  return n;
};

const numArg = (w: string | undefined, key: string): number => {
  const n = Number(w);
  if (w === undefined || !Number.isFinite(n)) throw new StyleError(`dump_modify ${key} needs a number`);
  return n;
};
