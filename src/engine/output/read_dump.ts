import type { System } from '../system';
import { StyleError } from '../force/types';
import { appendAtoms, buildAtomMap, hasCharge } from '../atoms';
import type { SimState } from '../types';

/*
 * read_dump and the snapshot selection of rerun (Haiku wave 9).
 *
 * Supported: native text dump files (dump atom / dump custom), fields x y z
 * vx vy vz q ix iy iz, keywords box, timestep, replace, purge, trim, add, label,
 * scaled, wrapped, format native. Everything else is a StyleError naming it.
 *
 * read_dump.rst: "Note that a simulation box must already be defined before
 * using the read_dump command." Defaults: "The option defaults are box = yes,
 * timestep = yes, replace = yes, purge = no, trim = no, add = no, scaled = no,
 * wrapped = yes, and format = native."
 *
 * Measured with native LAMMPS (black box, 2 Sep 2026 build), where the docs
 * leave the detail open:
 *  - a field type is rejected (invalid attribute fields); the type column is
 *    read only when add is yes or keep;
 *  - purge yes with replace yes or trim yes is rejected (native message: if
 *    read_dump purges it cannot replace or trim); purge yes with add no leaves
 *    no atoms and crashes the native build, so it is rejected here;
 *  - box no keeps the current box and remaps the absolute coordinates into it;
 *  - add yes gives the new atoms IDs continuing the largest existing ID in dump
 *    file order (dump IDs 7, 8, 9 with 1..4 present become 5, 6, 7);
 *  - with x y z read and no vx..vz, velocities are not changed.
 */

export const READ_FIELDS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'q', 'ix', 'iy', 'iz'] as const;
type ReadField = (typeof READ_FIELDS)[number];
const UNSUPPORTED_FIELDS = ['fx', 'fy', 'fz', 'apip_lambda'];
const LABEL_FIELDS = new Set<string>([...READ_FIELDS, 'id', 'type']);
const COORD_CANDIDATES: Record<'x' | 'y' | 'z', string[]> = {
  x: ['x', 'xs', 'xu', 'xsu'], y: ['y', 'ys', 'yu', 'ysu'], z: ['z', 'zs', 'zu', 'zsu'],
};
/** [scaled, wrapped] implied by a coordinate column label (dump.html: xs scaled, xu unwrapped, xsu both). */
const COORD_KIND: Record<string, [boolean, boolean]> = {
  x: [false, true], xs: [true, true], xu: [false, false], xsu: [true, false],
  y: [false, true], ys: [true, true], yu: [false, false], ysu: [true, false],
  z: [false, true], zs: [true, true], zu: [false, false], zsu: [true, false],
};

/** Parsed read_dump field and keyword arguments (without the file and Nstep). */
export interface DumpReadSpec {
  fields: ReadField[];
  box: boolean;
  timestep: boolean;
  /** true when the timestep keyword was given explicitly. */
  timestepGiven: boolean;
  replace: boolean;
  purge: boolean;
  trim: boolean;
  add: 'no' | 'yes' | 'keep';
  /** label field -> column label (field is a read field, id or type). */
  labels: Map<string, string>;
  scaled: boolean;
  wrapped: boolean;
}

const yn = (w: string | undefined, what: string): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`${what} must be yes or no, got '${w ?? ''}'`);
};

/**
 * Parses "field... keyword value ..." (read_dump.rst: "field = *x* or *y* or *z*
 * or *vx* or *vy* or *vz* or *q* or *ix* or *iy* or *iz* or *fx* or *fy* or *fz*
 * or *apip_lambda*"; "keyword = *nfile* or *box* or *timestep* or *replace* or
 * *purge* or *trim* or *add* or *label* or *scaled* or *wrapped* or *format*").
 * `what` names the command in messages.
 */
export const parseDumpSpec = (words: string[], what: string): DumpReadSpec => {
  const spec: DumpReadSpec = {
    fields: [], box: true, timestep: true, timestepGiven: false, replace: true, purge: false, trim: false,
    add: 'no', labels: new Map(), scaled: false, wrapped: true,
  };
  let k = 0;
  while (k < words.length && (READ_FIELDS as readonly string[]).includes(words[k])) {
    const f = words[k] as ReadField;
    if (!spec.fields.includes(f)) spec.fields.push(f);
    k++;
  }
  if (spec.fields.length === 0) {
    const w = words[0] ?? '';
    if (UNSUPPORTED_FIELDS.includes(w)) throw new StyleError(`${what}: field ${w} is not supported by the browser engine (fields: ${READ_FIELDS.join(' ')})`);
    if (w === 'type') throw new StyleError(`${what}: 'type' is not a read field (native LAMMPS rejects it too); the type column is read for add yes or keep`);
    throw new StyleError(`${what}: needs one or more fields (${READ_FIELDS.join(' ')}), got '${w}'`);
  }
  for (; k < words.length;) {
    const key = words[k];
    switch (key) {
      case 'box': spec.box = yn(words[k + 1], `${what} box`); k += 2; break;
      case 'timestep': spec.timestep = yn(words[k + 1], `${what} timestep`); spec.timestepGiven = true; k += 2; break;
      case 'replace': spec.replace = yn(words[k + 1], `${what} replace`); k += 2; break;
      case 'purge': spec.purge = yn(words[k + 1], `${what} purge`); k += 2; break;
      case 'trim': spec.trim = yn(words[k + 1], `${what} trim`); k += 2; break;
      case 'scaled': spec.scaled = yn(words[k + 1], `${what} scaled`); k += 2; break;
      case 'wrapped': spec.wrapped = yn(words[k + 1], `${what} wrapped`); k += 2; break;
      case 'add': {
        const v = words[k + 1];
        if (v !== 'yes' && v !== 'keep' && v !== 'no') throw new StyleError(`${what} add must be yes, keep or no, got '${v ?? ''}'`);
        spec.add = v; k += 2; break;
      }
      case 'label': {
        const f = words[k + 1], col = words[k + 2];
        if (f === undefined || col === undefined) throw new StyleError(`${what}: keyword label needs a field and a column label`);
        if (!LABEL_FIELDS.has(f)) throw new StyleError(`${what}: label field must be one of ${[...LABEL_FIELDS].join(' ')}, got '${f}'`);
        spec.labels.set(f, col); k += 3; break;
      }
      case 'format': {
        const v = words[k + 1];
        if (v !== 'native') throw new StyleError(`${what}: dump format '${v ?? ''}' is not supported by the browser engine (native text dumps only)`);
        if (k + 2 < words.length) throw new StyleError(`${what}: keyword format must be the last keyword, but '${words[k + 2]}' follows`);
        k = words.length; break;
      }
      case 'nfile':
        throw new StyleError(`${what}: keyword nfile (parallel dump files) is not supported by the browser engine`);
      default:
        if ((READ_FIELDS as readonly string[]).includes(key)) throw new StyleError(`${what}: field ${key} must come before the keywords`);
        throw new StyleError(`${what}: unknown keyword or field '${key}'`);
    }
  }
  if (spec.purge && (spec.replace || spec.trim)) {
    // read_dump.rst: "If the purge keyword is specified with a yes value, then all current atoms in the system are deleted before any of the operations invoked by the replace, trim, or add keywords take place."
    throw new StyleError(`${what}: purge yes cannot be combined with replace yes or trim yes (native: "If read_dump purges it cannot replace or trim"); use replace no and trim no`);
  }
  if (spec.purge && spec.add === 'no') {
    throw new StyleError(`${what}: purge yes needs add yes or keep, otherwise no atom remains (the native build crashes here)`);
  }
  return spec;
};

/** One snapshot of a native dump file. */
export interface DumpSnapshot {
  step: number;
  n: number;
  /** Orthogonal bounds, or for triclinic the true box (Howto_triclinic.html). */
  lo: [number, number, number];
  hi: [number, number, number];
  tilt: [number, number, number];
  triclinic: boolean;
  /** "pp pp pp" as written in the file. */
  boundary: string;
  columns: string[];
  /** Atom lines of the snapshot. */
  rows: string[];
}

export interface DumpFile {
  name: string;
  snapshots: DumpSnapshot[];
}

/**
 * Splits a native text dump ("ITEM: TIMESTEP", "ITEM: NUMBER OF ATOMS",
 * "ITEM: BOX BOUNDS", "ITEM: ATOMS ...") into snapshots. dump.html gives the
 * item order; for a triclinic box the bounding box lines are "xlo_bound
 * xhi_bound xy" and the true bounds follow Howto_triclinic.html:
 * "xlo_bound = xlo + MIN(0.0,xy,xz,xy+xz)", "ylo_bound = ylo + MIN(0.0,yz)",
 * "zlo_bound = zlo".
 */
export const parseDumpFile = (name: string, text: string): DumpFile => {
  if (name.includes('%')) throw new StyleError(`${name}: '%' wildcard dump files are not supported by the browser engine`);
  if (name.endsWith('.gz')) throw new StyleError(`${name}: gzipped dump files are not available in the browser`);
  const lines = text.split('\n');
  const snapshots: DumpSnapshot[] = [];
  const bad = (ln: number, msg: string) => new StyleError(`${name}: line ${ln + 1}: ${msg}`);
  let i = 0;
  const next = (): string => (i < lines.length ? lines[i++] : '');
  while (i < lines.length) {
    let line = next();
    if (line.trim() === '') continue;
    if (!line.startsWith('ITEM: TIMESTEP')) throw bad(i - 1, `expected 'ITEM: TIMESTEP', got '${line.trim()}'`);
    const step = Number(next().trim());
    if (!Number.isInteger(step)) throw bad(i - 1, 'timestep is not an integer');
    if (!next().startsWith('ITEM: NUMBER OF ATOMS')) throw bad(i - 1, "expected 'ITEM: NUMBER OF ATOMS'");
    const n = Number(next().trim());
    if (!Number.isInteger(n) || n < 0) throw bad(i - 1, 'bad atom count');
    line = next();
    if (!line.startsWith('ITEM: BOX BOUNDS')) throw bad(i - 1, "expected 'ITEM: BOX BOUNDS'");
    const bw = line.trim().split(/\s+/).slice(3);
    const triclinic = bw[0] === 'xy';
    const boundary = (triclinic ? bw.slice(3) : bw).join(' ');
    const b: number[][] = [];
    for (let d = 0; d < 3; d++) {
      const w = next().trim().split(/\s+/).map(Number);
      if (w.length < 2 || w.some((v) => !Number.isFinite(v))) throw bad(i - 1, 'bad box bounds');
      b.push(w);
    }
    let lo: [number, number, number], hi: [number, number, number], tilt: [number, number, number] = [0, 0, 0];
    if (triclinic) {
      const xy = b[0][2], xz = b[1][2], yz = b[2][2];
      tilt = [xy, xz, yz];
      lo = [b[0][0] - Math.min(0, xy, xz, xy + xz), b[1][0] - Math.min(0, yz), b[2][0]];
      hi = [b[0][1] - Math.max(0, xy, xz, xy + xz), b[1][1] - Math.max(0, yz), b[2][1]];
    } else {
      lo = [b[0][0], b[1][0], b[2][0]];
      hi = [b[0][1], b[1][1], b[2][1]];
    }
    line = next();
    if (!line.startsWith('ITEM: ATOMS')) throw bad(i - 1, "expected 'ITEM: ATOMS'");
    const columns = line.trim().split(/\s+/).slice(2);
    const rows: string[] = [];
    for (let a = 0; a < n; a++) {
      if (i >= lines.length) throw bad(i, 'file ends inside a snapshot');
      rows.push(next());
    }
    snapshots.push({ step, n, lo, hi, tilt, triclinic, boundary, columns, rows });
  }
  return { name, snapshots };
};

/** The snapshot with timestep `step` (read_dump: "The dump file is scanned for a snapshot with a timestamp that matches the specified Nstep"). */
export const findSnapshot = (file: DumpFile, step: number): DumpSnapshot => {
  const snap = file.snapshots.find((s) => s.step === step);
  if (!snap) throw new StyleError(`read_dump: no snapshot with timestep ${step} in ${file.name}`);
  return snap;
};

interface ColumnPlan {
  id: number;
  type: number;
  /** Column index per requested coordinate, -1 when not requested. */
  coord: [number, number, number];
  scaled: boolean;
  wrapped: boolean;
  cols: Partial<Record<ReadField, number>>;
}

const planColumns = (sys: System, file: DumpFile, snap: DumpSnapshot, spec: DumpReadSpec): ColumnPlan => {
  const cols = snap.columns;
  const find = (label: string, what: string): number => {
    const c = cols.indexOf(label);
    if (c < 0) throw new StyleError(`read_dump: column ${label} (${what}) not found in ${file.name}; columns: ${cols.join(' ')}`);
    return c;
  };
  const id = find(spec.labels.get('id') ?? 'id', 'atom IDs');
  let type = -1;
  if (spec.add !== 'no') type = find(spec.labels.get('type') ?? 'type', 'atom types needed for add');
  const coord: [number, number, number] = [-1, -1, -1];
  const kinds: [boolean, boolean][] = [];
  const axes = ['x', 'y', 'z'] as const;
  axes.forEach((ax, d) => {
    if (!spec.fields.includes(ax)) return;
    if (sys.state.dimension === 2 && d === 2) throw new StyleError('read_dump: a z-dimension field (z) is an error for a 2d simulation');
    const lab = spec.labels.get(ax);
    if (lab !== undefined) {
      coord[d] = find(lab, `field ${ax}`);
      kinds.push([spec.scaled, spec.wrapped]);
      return;
    }
    const c = COORD_CANDIDATES[ax].find((l) => cols.includes(l));
    if (c === undefined) throw new StyleError(`read_dump: no column for field ${ax} in ${file.name} (labels ${COORD_CANDIDATES[ax].join(', ')}); columns: ${cols.join(' ')}`);
    coord[d] = cols.indexOf(c);
    kinds.push(COORD_KIND[c]);
  });
  const scaled = kinds.length ? kinds[0][0] : false;
  const wrapped = kinds.length ? kinds[0][1] : true;
  if (kinds.some(([s, w]) => s !== scaled || w !== wrapped)) {
    throw new StyleError('read_dump: the x, y, z fields do not have consistent scaling/wrapping (read_dump.rst: "must be identical for any of the x, y, z fields")');
  }
  if (scaled && snap.triclinic && coord.some((c) => c < 0)) {
    throw new StyleError('read_dump: scaled coordinates of a triclinic box need all three of x, y and z');
  }
  const planCols: Partial<Record<ReadField, number>> = {};
  for (const f of spec.fields) {
    if (f === 'x' || f === 'y' || f === 'z') continue;
    planCols[f] = find(spec.labels.get(f) ?? f, `field ${f}`);
  }
  return { id, type, coord, scaled, wrapped, cols: planCols };
};

/**
 * Reads one snapshot into the system (read_dump.rst, "Description"). Returns
 * the counts of replaced, added and trimmed atoms.
 */
export const readDumpSnapshot = (
  sys: System,
  file: DumpFile,
  snap: DumpSnapshot,
  spec: DumpReadSpec,
): { replaced: number; added: number; trimmed: number } => {
  const s = sys.state;
  // read_dump.rst: "an error is generated if the snapshot is for a triclinic box and the current simulation box is orthogonal or vice versa"
  if (snap.triclinic !== s.box.triclinic) {
    throw new StyleError(`read_dump: snapshot ${file.name} @${snap.step} is ${snap.triclinic ? 'triclinic' : 'orthogonal'} but the simulation box is ${s.box.triclinic ? 'triclinic' : 'orthogonal'}`);
  }
  for (let d = 0; d < 3; d++) {
    for (const side of [0, 1] as const) {
      const b = s.box.boundary[d][side];
      if (b === 's' || b === 'm' || b === 'f') {
        throw new StyleError(`read_dump: ${b === 'f' ? 'fixed' : 'shrink-wrapped'} boundaries are not supported by the browser engine (boundary ${s.box.boundary.map((x) => x.join('')).join(' ')})`);
      }
    }
  }
  const plan = planColumns(sys, file, snap, spec);
  if (spec.fields.some((f) => f === 'q') && !hasCharge(s)) {
    throw new StyleError('read_dump: field q needs per-atom charges (use an atom style with charge, or fix property/atom q)');
  }
  if (spec.box && snap.boundary !== s.box.boundary.map((x) => x.join('')).join(' ')) {
    sys.warn(`read_dump: snapshot boundary conditions (${snap.boundary}) differ from the simulation box; the boundary is kept`);
  }
  const nrow = snap.rows.length;
  const parse = (r: string[], c: number): number => {
    const v = Number(r[c]);
    if (!Number.isFinite(v)) throw new StyleError(`read_dump: non-numeric value '${r[c]}' in ${file.name} @${snap.step}`);
    return v;
  };
  // ---- the box and timestep come first: positions are remapped into the new box
  if (spec.box) sys.setBox(snap.lo, snap.hi, snap.tilt, 0);
  if (spec.timestep) {
    s.time = s.time + (s.step - s.timeStep) * s.dt;
    s.step = snap.step;
    s.timeStep = snap.step;
    sys.nb.lastBuild = -1;
  }
  // ---- per row values (absolute coordinates; scaled ones converted with the snapshot box)
  const rows = snap.rows.map((l) => l.trim().split(/\s+/));
  const pos = new Float64Array(3 * nrow);
  const b = snap;
  for (let r = 0; r < nrow; r++) {
    const w = rows[r];
    if (plan.scaled) {
      if (snap.triclinic) {
        const s0 = parse(w, plan.coord[0]), s1 = parse(w, plan.coord[1]), s2 = parse(w, plan.coord[2]);
        const lx = b.hi[0] - b.lo[0], ly = b.hi[1] - b.lo[1], lz = b.hi[2] - b.lo[2];
        const [xy, xz, yz] = b.tilt;
        pos[3 * r] = b.lo[0] + lx * s0 + xy * s1 + xz * s2;
        pos[3 * r + 1] = b.lo[1] + ly * s1 + yz * s2;
        pos[3 * r + 2] = b.lo[2] + lz * s2;
      } else {
        for (let d = 0; d < 3; d++) {
          if (plan.coord[d] < 0) continue;
          pos[3 * r + d] = b.lo[d] + parse(w, plan.coord[d]) * (b.hi[d] - b.lo[d]);
        }
      }
    } else {
      for (let d = 0; d < 3; d++) if (plan.coord[d] >= 0) pos[3 * r + d] = parse(w, plan.coord[d]);
    }
  }
  const fieldVal = (r: number, f: ReadField): number => parse(rows[r], plan.cols[f] as number);
  // ---- atoms: purge, trim, replace, add (read_dump.rst keyword order)
  let trimmed = 0, replaced = 0, added = 0;
  if (spec.purge) {
    sys.deleteAtoms(new Uint8Array(sys.state.n).fill(1));
  }
  const ids = new Int32Array(nrow);
  for (let r = 0; r < nrow; r++) ids[r] = Math.trunc(parse(rows[r], plan.id));
  if (spec.trim) {
    const inDump = new Set<number>(ids);
    const cur = sys.state;
    const del = new Uint8Array(cur.n);
    for (let i = 0; i < cur.n; i++) if (!inDump.has(cur.id[i])) del[i] = 1;
    trimmed = sys.deleteAtoms(del);
  }
  const st: SimState = sys.state;
  const map = buildAtomMap(st);
  const newRows: number[] = [];
  for (let r = 0; r < nrow; r++) {
    const id = ids[r];
    const idx = id >= 0 && id < map.length ? map[id] : -1;
    if (idx >= 0) {
      if (!spec.replace) continue;
      replaced++;
      for (let d = 0; d < 3; d++) if (plan.coord[d] >= 0) st.x[3 * idx + d] = pos[3 * r + d];
      for (const f of spec.fields) {
        if (f === 'vx' || f === 'vy' || f === 'vz') st.v[3 * idx + (f.charCodeAt(1) - 120)] = fieldVal(r, f);
        else if (f === 'ix' || f === 'iy' || f === 'iz') st.image[3 * idx + (f.charCodeAt(1) - 120)] = fieldVal(r, f);
        else if (f === 'q') st.q[idx] = fieldVal(r, f);
      }
      sys.geom.remap(st.x, st.image, idx);
    } else if (spec.add !== 'no') {
      newRows.push(r);
    }
  }
  if (newRows.length) {
    const m = newRows.length;
    const x = new Float64Array(3 * m), v = new Float64Array(3 * m), image = new Int32Array(3 * m);
    const type = new Int32Array(m), q = new Float64Array(m), keepIds = new Int32Array(m);
    newRows.forEach((r, k) => {
      for (let d = 0; d < 3; d++) x[3 * k + d] = pos[3 * r + d];
      for (const f of spec.fields) {
        if (f === 'vx' || f === 'vy' || f === 'vz') v[3 * k + (f.charCodeAt(1) - 120)] = fieldVal(r, f);
        else if (f === 'ix' || f === 'iy' || f === 'iz') image[3 * k + (f.charCodeAt(1) - 120)] = fieldVal(r, f);
        else if (f === 'q') q[k] = fieldVal(r, f);
      }
      const t = Math.trunc(parse(rows[r], plan.type));
      if (t < 1 || t > st.ntypes) throw new StyleError(`read_dump: atom type ${t} of atom ${ids[r]} is outside 1..${st.ntypes}`);
      type[k] = t;
      keepIds[k] = ids[r];
    });
    const n0 = st.n;
    appendAtoms(st, {
      x, v, image, type, q: hasCharge(st) ? q : undefined,
      id: spec.add === 'keep' ? keepIds : undefined,
    });
    // "image flags for new atoms are set to default values" unless ix/iy/iz were read (above)
    for (let i = n0; i < st.n; i++) sys.geom.remap(st.x, st.image, i);
    added = m;
  }
  if (trimmed || added) sys.atomsChanged();
  sys.bump();
  sys.log(`read_dump ${file.name} @${snap.step}: ${replaced} atoms replaced, ${added} added, ${trimmed} trimmed`);
  return { replaced, added, trimmed };
};

/** Rerun selection rule (rerun.html keywords first, last, every, skip). */
export interface RerunSelection {
  first: number;
  last: number;
  every: number;
  skip: number;
}

/**
 * The snapshots rerun reads, in order (rerun.html: "The first, last, every,
 * skip keywords determine which snapshots are read from the dump file(s)").
 * Also the rerun.html sentence "If a snapshot is encountered that is not in
 * ascending order, it will skip the snapshot until it reads one that is." drops
 * snapshots that are not ascending in time. A snapshot above `last` ends the list.
 * Measured with native LAMMPS for one file: skip counts snapshots from the first
 * one read (first 5 skip 2 on steps 0,5,..,20 reads 5 and 15), and the first
 * snapshot read is not blocked by every. Across two files with skip above 1 the
 * native counting was not reproduced, so that combination is refused.
 */
export const selectSnapshots = (files: DumpFile[], sel: RerunSelection): { file: DumpFile; snap: DumpSnapshot }[] => {
  const out: { file: DumpFile; snap: DumpSnapshot }[] = [];
  if (sel.skip > 1 && files.length > 1) {
    throw new StyleError('rerun: skip > 1 with more than one dump file is not supported by the browser engine (native counting across files was not reproduced)');
  }
  let lastRead = -Infinity;
  let anchor = -1;
  for (const file of files) {
    for (let k = 0; k < file.snapshots.length; k++) {
      const snap = file.snapshots[k];
      if (snap.step > sel.last) return out;
      if (snap.step < sel.first) continue;
      if (snap.step <= lastRead) continue;
      if (sel.skip > 1 && anchor >= 0 && (k - anchor) % sel.skip !== 0) continue;
      const firstRead = anchor < 0;
      if (!firstRead && sel.every > 0 && snap.step % sel.every !== 0) continue;
      out.push({ file, snap });
      lastRead = snap.step;
      if (firstRead) anchor = k;
    }
  }
  return out;
};

