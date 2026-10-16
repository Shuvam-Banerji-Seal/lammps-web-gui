/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-DUMP/dump_vtk.cpp and
 * dump_vtk.h (class DumpVTK, dump style vtk; contributing authors Daniel Queteschiner (DCS, JKU), Christoph Kloss (DCS) and
 * Richard Berger (JKU); copyright Sandia Corporation), with the writer of src/vtk_writer.cpp (vtk_writer.ts). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump_vtk.html (verbatim): "dump ID group-ID vtk N file args"; "Dump a snapshot of atom quantities to one or
 * more files every N timesteps in a format readable by the VTK visualization toolkit"; "This dump style is similar to dump
 * style custom but writes data in the VTK simple legacy or XML format, depending on the filename extension specified for the
 * dump file."
 *
 * Native behaviour kept here (dump_vtk.cpp at b332f28):
 *  - the attributes are stored by their kind, so the columns come out in a fixed order: the atom attributes in the order of
 *    the DumpVTK enum (id, mol, proc, procp1, type, element, mass, xs..., xu..., xsu..., ix..., vx..., fx..., q, mu..., radius,
 *    diameter, omega..., angmom..., tq...), then the compute, fix, variable and custom attributes in the order given. A
 *    repeated fixed attribute is written once; a repeated compute, fix, variable or custom attribute is written each time.
 *  - "x y z" are always the points and are not written as arrays.
 *  - a run of three attributes xs ys zs (also xu, xsu, ix, vx, fx, mux, omegax, angmomx, tqx) becomes one 3-component array
 *    named after the first attribute with its "x" removed (vx vy vz gives "v", xs ys zs gives "s"); c_ID[1] c_ID[2] c_ID[3]
 *    becomes one array named c_ID[1-3].
 *  - the file extension selects the format: .vtp (PolyData), .vtu (UnstructuredGrid), otherwise the legacy format. The
 *    bounding box goes to a second file named with _boundingBox before the extension: a rectilinear grid for an orthogonal
 *    box (.vtr for XML), an unstructured grid of one hexahedron for a triclinic box (.vtu for XML).
 *  - legacy output: a "%" in the file name is replaced by 0 (only processor 0 writes); atom order is the storage order, and
 *    sorting is off by default.
 *
 * Not supported (named StyleErrors): dump_modify binary yes (binary payloads), refresh, thresh LAST (the stored values of the
 * previous snapshot), sort by a column, and the other dump_modify keywords that dump vtk does not take; a "%" file name with
 * .vtp or .vtu (one piece file per processor and a .pvtp summary); compressed (.gz) files.
 */
import { StyleError } from '../../engine/force/types';
import type { DumpVtkFactory, DumpVtkHost, DumpVtkStyle } from '../../engine/output/dump';
import { VtkWriter, type VtkArray } from './vtk_writer';

/** The atom attributes in DumpVTK's enum order (x, y, z are the points and come first). */
const FIXED_ORDER = [
  'id', 'mol', 'proc', 'procp1', 'type', 'element', 'mass',
  'xs', 'ys', 'zs', 'xu', 'yu', 'zu', 'xsu', 'ysu', 'zsu', 'ix', 'iy', 'iz',
  'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'q', 'mux', 'muy', 'muz', 'mu', 'radius', 'diameter',
  'omegax', 'omegay', 'omegaz', 'angmomx', 'angmomy', 'angmomz', 'tqx', 'tqy', 'tqz',
];
const RANK = new Map(FIXED_ORDER.map((n, i) => [n, i]));
/** The attribute triples identify_vectors() turns into one 3-component array (the first name is the key). */
const VECTOR_TRIPLES: [string, string, string][] = [
  ['xs', 'ys', 'zs'], ['xu', 'yu', 'zu'], ['xsu', 'ysu', 'zsu'], ['ix', 'iy', 'iz'], ['vx', 'vy', 'vz'], ['fx', 'fy', 'fz'],
  ['mux', 'muy', 'muz'], ['omegax', 'omegay', 'omegaz'], ['angmomx', 'angmomy', 'angmomz'], ['tqx', 'tqy', 'tqz'],
];
const VECTOR_START = new Map(VECTOR_TRIPLES.map((t) => [t[0], t]));
/** DumpVTK's compute, fix, variable and custom attribute names (ArgInfo COMPUTE, FIX, VARIABLE, DNAME, INAME). */
const DYNAMIC = /^(c|f|v|d2|d|i2|i)_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/;
const THRESH_OPS = new Set(['<', '<=', '>', '>=', '==', '!=', '|^']);

/** One output array: the core column names it is made of (one, or three for a vector) and its kind. */
interface Column {
  name: string;
  parts: string[];
  kind: 'int' | 'double' | 'string';
}

interface Thresh {
  col: string;
  op: string;
  value: number;
}

/** The index in brackets of a dynamic attribute, 0 when it has none (native argindex). */
const argIndex = (c: string): number => {
  const m = /\[(\d+)\]$/.exec(c);
  return m ? Number(m[1]) : 0;
};

/** The vector name of a triple start: its first "x" removed (native identify_vectors). */
const vectorName = (start: string): string => {
  const pos = start.indexOf('x');
  return pos === 0 ? start.slice(1) : start.slice(0, pos);
};

/** A floating point dump_modify value (native utils::numeric). */
const numArg = (w: string | undefined, key: string): number => {
  const n = Number(w);
  if (w === undefined || w === '' || !Number.isFinite(n)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in dump_modify ${key}`);
  return n;
};

/** dump_modify keywords dump vtk leaves to the core dump command (its every, first, delay and pad). */
const CORE_KEYWORDS = new Set(['every', 'first', 'delay', 'pad']);

/** The dump vtk helper for one dump command (see the header of this file). */
/**
 * percent_subst of dump_vtk.cpp: only the FIRST '%' of the file name is replaced (std::string::find), as native does;
 * the legacy format writes one file, so its '%' becomes 0.
 */
const percentSubst = (name: string, subst: string): string => {
  const pos = name.indexOf('%');
  return pos < 0 ? name : name.slice(0, pos) + subst + name.slice(pos + 1);
};

class DumpVtk implements DumpVtkStyle {
  private readonly columns: Column[];
  private readonly xml: 'polydata' | 'unstructured' | null;
  private label = 'Generated by LAMMPS';
  private precision: 'single' | 'double' = 'single';
  private element: string[] = [];
  private region: string | null = null;
  private thresh: Thresh[] = [];
  private sortById = false;

  constructor(private readonly host: DumpVtkHost, cols: string[]) {
    const file = host.file;
    this.xml = file.endsWith('.vtp') ? 'polydata' : file.endsWith('.vtu') ? 'unstructured' : null;
    if (this.xml && file.includes('%')) {
      throw new StyleError(`dump vtk: a "%" file name with .vtp or .vtu (one piece file per processor and a .pvtp or .pvtu summary) is not supported by the browser engine`);
    }
    this.columns = this.parseColumns(cols);
  }

  /** The attributes of the dump command (native DumpVTK::parse_vtk_fields and identify_vectors). */
  private parseColumns(cols: string[]): Column[] {
    const fixed = new Set<string>();
    for (const c of cols) {
      if (c === 'x' || c === 'y' || c === 'z') continue;
      if (RANK.has(c)) {
        this.host.validateColumn(c);
        fixed.add(c);
      } else if (DYNAMIC.test(c)) {
        this.host.validateColumn(c);
      } else {
        throw new StyleError(`Invalid attribute in dump vtk command: ${c}`);
      }
    }
    const out: Column[] = [];
    const merged = new Set<string>();
    const sorted = [...fixed].sort((a, b) => (RANK.get(a) ?? 0) - (RANK.get(b) ?? 0));
    for (const c of sorted) {
      const triple = VECTOR_START.get(c);
      if (triple && triple.every((n) => fixed.has(n))) {
        for (const n of triple.slice(1)) merged.add(n);
        out.push({ name: vectorName(c), parts: [...triple], kind: this.host.isIntColumn(c) ? 'int' : 'double' });
      } else if (!merged.has(c)) {
        out.push({ name: c, parts: [c], kind: c === 'element' ? 'string' : this.host.isIntColumn(c) ? 'int' : 'double' });
      }
    }
    // dynamic attributes keep their order; the three of an index-range vector are one array named c_ID[1-3]
    const isDyn = cols.map((c) => DYNAMIC.test(c));
    const grouped = new Set<number>();
    for (let i = 0; i < cols.length; i++) {
      if (!isDyn[i] || grouped.has(i)) continue;
      const c = cols[i];
      const k = argIndex(c);
      if (k !== 0 && i + 2 < cols.length && isDyn[i + 1] && isDyn[i + 2]) {
        const b = cols[i + 1], d = cols[i + 2];
        const len = c.length - 3;
        if (b.slice(0, len) === c.slice(0, len) && d.slice(0, len) === c.slice(0, len)) {
          grouped.add(i + 1);
          grouped.add(i + 2);
          out.push({ name: `${c.slice(0, -1)}-${argIndex(d)}]`, parts: [c, b, d], kind: this.host.isIntColumn(c) ? 'int' : 'double' });
          continue;
        }
      }
      out.push({ name: c, parts: [c], kind: this.host.isIntColumn(c) ? 'int' : 'double' });
    }
    return out;
  }

  modify(args: string[], k: number): number {
    const key = args[k];
    const v = args[k + 1];
    switch (key) {
      case 'label':
        if (v === undefined) throw new StyleError('dump_modify label needs a string');
        this.label = v;
        return 2;
      case 'binary':
        if (v !== 'no') throw new StyleError('dump_modify binary yes is not supported by the browser engine (binary VTK output); use binary no');
        return 2;
      case 'double':
        if (v !== 'yes' && v !== 'no') throw new StyleError('dump_modify double must be yes or no');
        this.precision = v === 'yes' ? 'double' : 'single';
        return 2;
      case 'element': {
        const n = this.host.sys.state.ntypes;
        const names = args.slice(k + 1, k + 1 + n);
        if (names.length !== n) throw new StyleError('Number of dump_modify element names does not match number of atom types');
        this.element = names;
        return 1 + n;
      }
      case 'region':
        if (v === undefined) throw new StyleError('Illegal dump_modify command');
        if (v === 'none') this.region = null;
        else {
          this.host.sys.region(v);
          this.region = v;
        }
        return 2;
      case 'sort':
        if (v !== 'off' && v !== 'id') throw new StyleError('dump vtk supports dump_modify sort off or id only');
        this.sortById = v === 'id';
        return 2;
      case 'thresh': {
        if (v === 'none') {
          this.thresh = [];
          return 2;
        }
        const op = args[k + 2], val = args[k + 3];
        if (v === undefined || op === undefined || val === undefined) throw new StyleError('Illegal dump_modify thresh command');
        if (v === 'element') throw new StyleError('Invalid dump_modify thresh attribute: element');
        if (!RANK.has(v) && !DYNAMIC.test(v)) throw new StyleError(`Invalid dump_modify thresh attribute: ${v}`);
        if (!THRESH_OPS.has(op)) throw new StyleError('Invalid dump_modify thresh operator');
        if (val === 'LAST') throw new StyleError('dump_modify thresh LAST is not supported by the browser engine for dump vtk');
        this.host.validateColumn(v);
        this.thresh.push({ col: v, op, value: numArg(val, 'thresh') });
        return 4;
      }
      case 'refresh':
        throw new StyleError('dump_modify refresh is not supported by the browser engine for dump vtk');
      default:
        if (CORE_KEYWORDS.has(key)) return 0;
        throw new StyleError(`dump_modify ${key} is not supported by the browser engine for dump vtk`);
    }
  }

  write(step: number): void {
    const sys = this.host.sys;
    const s = sys.state;
    // forces are current for fx fy fz (native compute invocations happen in count(); the core dump does the same)
    sys.forces();
    const rows = this.selectRows();
    const flavor = this.xml ? 'xml' : 'legacy';
    const cache = new Map<string, Float64Array>();
    const column = (c: string): Float64Array => {
      let vals = cache.get(c);
      if (!vals) cache.set(c, (vals = this.host.columnValues(c)));
      return vals;
    };

    const writer = new VtkWriter(flavor, this.precision);
    writer.setTitle(this.label);
    const xyz: number[] = [];
    for (const i of rows) xyz.push(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
    if (this.xml === 'unstructured') writer.setUnstructured(xyz);
    else writer.setPolydata(xyz);
    for (const col of this.columns) writer.addPointArray(this.array(col, rows, column));

    const pad = this.host.pad;
    const stepText = pad > 0 ? String(step).padStart(pad, '0') : String(step);
    const multi = this.host.file.includes('*');
    const named = (base: string): string => sys.resolveFile(multi ? this.host.fillWildcard(base, stepText) : base);
    const main = flavor === 'legacy' ? percentSubst(this.host.file, '0') : this.host.file;
    sys.writeFileAt(named(main), writer.text(), false);

    // the bounding box: a second file, named before the extension
    const box = new VtkWriter(flavor, this.precision);
    box.setTitle(this.label);
    const g = sys.geom;
    const lo = s.box.lo, hi = s.box.hi;
    if (g.triclinic) box.setHexahedron(triclinicCorners(lo, hi, s.box.tilt));
    else box.setRectilinear([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]]);
    const dot = main.lastIndexOf('.');
    let domain = dot < 0 ? `${main}_boundingBox` : `${main.slice(0, dot)}_boundingBox${main.slice(dot)}`;
    if (flavor === 'xml') domain = `${domain.slice(0, -1)}${g.triclinic ? 'u' : 'r'}`;
    sys.writeFileAt(named(domain), box.text(), false);
  }

  /** The atoms written: the group, the region and the thresholds, in native storage order (native count()). */
  private selectRows(): number[] {
    const sys = this.host.sys;
    const s = sys.state;
    const reg = this.region ? sys.region(this.region) : null;
    let rows: number[] = [];
    for (const i of this.host.nativeRows()) {
      if (!(s.mask[i] & this.host.groupBit)) continue;
      if (reg && !reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
      rows.push(i);
    }
    for (const t of this.thresh) {
      const vals = this.host.columnValues(t.col);
      rows = rows.filter((i) => {
        const x = vals[i];
        switch (t.op) {
          case '<': return x < t.value;
          case '<=': return x <= t.value;
          case '>': return x > t.value;
          case '>=': return x >= t.value;
          case '==': return x === t.value;
          case '!=': return x !== t.value;
          default: return (x !== 0) !== (t.value !== 0);
        }
      });
    }
    if (this.sortById) rows.sort((a, b) => s.id[a] - s.id[b]);
    return rows;
  }

  /** One point data array: the values of the atoms in row order, ncomp per atom (interleaved). */
  private array(col: Column, rows: number[], column: (c: string) => Float64Array): VtkArray {
    const ncomp = col.parts.length;
    if (col.kind === 'string') {
      const s = this.host.sys.state;
      const strs = rows.map((i) => (this.element.length ? this.element[s.type[i] - 1] : 'C'));
      return { name: col.name, kind: 'string', ncomp: 1, nums: [], strs };
    }
    const vals = col.parts.map(column);
    const nums = new Float64Array(rows.length * ncomp);
    rows.forEach((i, r) => {
      for (let k = 0; k < ncomp; k++) nums[r * ncomp + k] = vals[k][i];
    });
    return { name: col.name, kind: col.kind, ncomp, nums, strs: [] };
  }
}

/**
 * The eight corners of a triclinic box in the order of the VTK hexahedron (native Domain::lamda_box_corners, then the
 * corner_map of dump_vtk.cpp write_domain). Lamda corner k has x from bit 0, y from bit 1 and z from bit 2 of k.
 */
const triclinicCorners = (lo: number[], hi: number[], tilt: number[]): number[][] => {
  const [xy, xz, yz] = tilt;
  const xprd = hi[0] - lo[0], yprd = hi[1] - lo[1], zprd = hi[2] - lo[2];
  const lamda2x = (l0: number, l1: number, l2: number): number[] => [
    xprd * l0 + xy * l1 + xz * l2 + lo[0],
    yprd * l1 + yz * l2 + lo[1],
    zprd * l2 + lo[2],
  ];
  const map = [0, 1, 3, 2, 4, 5, 7, 6];
  return map.map((k) => lamda2x(k & 1, (k >> 1) & 1, (k >> 2) & 1));
};

/** The helper the core dump command creates for "dump ID group vtk N file attributes" (see lazyStyles.ts group gpl_dump_vtk). */
export const createDumpVtk: DumpVtkFactory = (host, cols) => new DumpVtk(host, cols);
