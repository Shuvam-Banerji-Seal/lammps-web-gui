import type { System } from '../system';
import type { AtomStyle, SimState, TopoList } from '../types';
import { StyleError } from '../force/types';
import { appendAtoms, emptyState, isMolecularStyle, maxAtomId, pushTopo, sphereMass } from '../atoms';
import { makeBox } from '../domain';

/*
 * Data files — docs.lammps.org/read_data.html and write_data.html.
 *
 * Header: "Header lines can come in any order. Each keyword takes a single
 * value unless noted"; atoms, bonds, angles, dihedrals, impropers, atom
 * types, bond types, angle types, dihedral types, improper types, xlo xhi,
 * ylo yhi, zlo zhi, xy xz yz. "The default values for these 3 keywords are
 * -0.5 and 0.5 for each lo/hi pair." Body sections: Atoms, Velocities,
 * Masses, Bonds, Angles, Dihedrals, Impropers, Pair Coeffs, PairIJ Coeffs,
 * Bond Coeffs, Angle Coeffs, Dihedral Coeffs, Improper Coeffs.
 * Atoms line formats: atomic "atom-ID atom-type x y z"; charge "atom-ID
 * atom-type q x y z"; bond/angle/molecular "atom-ID molecule-ID atom-type x y
 * z"; full "atom-ID molecule-ID atom-type q x y z"; "each line can
 * optionally have 3 flags (nx,ny,nz) appended to it". "If the system is
 * periodic (in a dimension), then atom coordinates can be outside the bounds
 * (in that dimension); they will be remapped (in a periodic sense) back
 * inside the box." "all non-zero image flags for non-periodic dimensions
 * will be be reset to zero". "For 2d simulations, the atom coordinate z must
 * be specified as 0.0 ... LAMMPS will force them to zero." Velocities lines:
 * "atom-ID vx vy vz". Bonds: "ID type atom1 atom2"; Angles: "ID type atom1
 * atom2 atom3"; Dihedrals and Impropers: "ID type atom1 atom2 atom3 atom4".
 * "Pair Coeffs ... only a single type I is specified, which sets the
 * coefficients for type I interacting with type I" (pair_coeff.html).
 * write_data output format matches native LAMMPS (header line "LAMMPS data
 * file via write_data, version ..., timestep = N, units = U", shortest
 * round-trip numbers, atoms ordered by ID).
 */

const STYLE_COLS: Record<AtomStyle, string[]> = {
  atomic: ['id', 'type', 'x', 'y', 'z'],
  charge: ['id', 'type', 'q', 'x', 'y', 'z'],
  bond: ['id', 'mol', 'type', 'x', 'y', 'z'],
  angle: ['id', 'mol', 'type', 'x', 'y', 'z'],
  molecular: ['id', 'mol', 'type', 'x', 'y', 'z'],
  full: ['id', 'mol', 'type', 'q', 'x', 'y', 'z'],
  // read_data.html: "sphere | atom-ID atom-type diameter density x y z"
  sphere: ['id', 'type', 'diameter', 'density', 'x', 'y', 'z'],
};

const HEADER_KEYS: [RegExp, string][] = [
  [/^atoms$/, 'atoms'], [/^bonds$/, 'bonds'], [/^angles$/, 'angles'], [/^dihedrals$/, 'dihedrals'], [/^impropers$/, 'impropers'],
  [/^atom types$/, 'atom types'], [/^bond types$/, 'bond types'], [/^angle types$/, 'angle types'],
  [/^dihedral types$/, 'dihedral types'], [/^improper types$/, 'improper types'],
  [/^extra (bond|angle|dihedral|improper|special) per atom$/, 'extra'],
];

const SECTIONS = new Set([
  'Atoms', 'Velocities', 'Masses', 'Bonds', 'Angles', 'Dihedrals', 'Impropers',
  'Pair Coeffs', 'PairIJ Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs',
  'Ellipsoids', 'Lines', 'Triangles', 'Bodies',
  'Atom Type Labels', 'Bond Type Labels', 'Angle Type Labels', 'Dihedral Type Labels', 'Improper Type Labels',
  'BondBond Coeffs', 'BondAngle Coeffs', 'MiddleBondTorsion Coeffs', 'EndBondTorsion Coeffs', 'AngleTorsion Coeffs',
  'AngleAngleTorsion Coeffs', 'BondBond13 Coeffs', 'AngleAngle Coeffs',
]);

export interface ReadDataOptions {
  add: 'none' | 'append' | 'merge' | { id: number; mol: number };
  offset: [number, number, number, number, number];
  shift: [number, number, number];
  extraTypes: [number, number, number, number, number];
  group: string | null;
  nocoeff: boolean;
  /** read_data fix fix-ID header-string section-string (section name -> fix ID). */
  fixSections: Map<string, string>;
}

/** What read_data / write_data need from a fix that owns a data-file section (fix property/atom). */
interface DataSectionFix {
  id: string;
  nvalues: number;
  readValues(s: SimState, i: number, w: readonly string[], at: number): void;
  sectionHeader(): string;
  writeValues(s: SimState, i: number): string;
}
const isDataSectionFix = (f: unknown): f is DataSectionFix => typeof (f as DataSectionFix)?.readValues === 'function';

export const defaultReadOptions = (): ReadDataOptions => ({
  add: 'none', offset: [0, 0, 0, 0, 0], shift: [0, 0, 0], extraTypes: [0, 0, 0, 0, 0], group: null, nocoeff: false, fixSections: new Map(),
});

const intOf = (w: string | undefined, what: string, line: number): number => {
  const n = Number(w);
  if (w === undefined || !Number.isInteger(n)) throw new StyleError(`data file line ${line}: expected an integer ${what}, got '${w ?? ''}'`);
  return n;
};
const numOf = (w: string | undefined, what: string, line: number): number => {
  const n = Number(w);
  if (w === undefined || w === '' || !Number.isFinite(n)) throw new StyleError(`data file line ${line}: expected a number for ${what}, got '${w ?? ''}'`);
  return n;
};

/** read_data file [keywords] into the system. */
export const readData = (sys: System, text: string, opts: ReadDataOptions): void => {
  const lines = text.split('\n');
  // header: skip the first line (title)
  const h: Record<string, number> = {};
  let lo: [number, number, number] = [-0.5, -0.5, -0.5];
  let hi: [number, number, number] = [0.5, 0.5, 0.5];
  let tilt: [number, number, number] | null = null;
  let k = 1;
  for (; k < lines.length; k++) {
    const raw = lines[k].replace(/#.*/, '').trim();
    if (!raw) continue;
    if (SECTIONS.has(raw) || SECTIONS.has(lines[k].trim().split('#')[0].trim())) break;
    const w = raw.split(/\s+/);
    const rest = w.slice(1).join(' ');
    const rest3 = w.slice(3).join(' ');
    const rest2 = w.slice(2).join(' ');
    if (rest2 === 'xlo xhi' || rest2 === 'ylo yhi' || rest2 === 'zlo zhi') {
      const d = 'xyz'.indexOf(rest2[0]);
      lo[d] = numOf(w[0], rest2, k + 1);
      hi[d] = numOf(w[1], rest2, k + 1);
      continue;
    }
    if (rest3 === 'xy xz yz') {
      tilt = [numOf(w[0], 'xy', k + 1), numOf(w[1], 'xz', k + 1), numOf(w[2], 'yz', k + 1)];
      continue;
    }
    if (/^(avec|bvec|cvec|abc origin)$/.test(rest3) || /^(avec|bvec|cvec)$/.test(w.slice(3).join(' '))) {
      throw new StyleError('general triclinic data files (avec/bvec/cvec) are not supported; use xlo xhi ... xy xz yz');
    }
    const key = HEADER_KEYS.find(([re]) => re.test(rest));
    if (!key) {
      if (/^(ellipsoids|lines|triangles|bodies)$/.test(rest)) throw new StyleError(`data file header '${rest}' needs an atom style the browser engine does not support`);
      throw new StyleError(`unknown data file header line ${k + 1}: '${raw}'`);
    }
    if (key[1] !== 'extra') h[key[1]] = intOf(w[0], rest, k + 1);
  }
  const ntypesFile = h['atom types'] ?? 0;
  const s0 = sys.hasBox ? sys.state : null;
  const adding = opts.add !== 'none';
  if (!adding && s0) throw new StyleError('read_data: a simulation box already exists (use add append/merge, or clear first)');
  if (adding && !s0) throw new StyleError('read_data add needs an existing box');
  const [toff, boff, aoff, doff, ioff] = opts.offset;
  let s: SimState;
  if (!adding) {
    if (sys.dimension === 2 && !(lo[2] < 0 && hi[2] > 0)) throw new StyleError('read_data: for a 2d simulation zlo and zhi must straddle zero');
    const box = makeBox({ lo, hi, boundary: sys.boundary, tilt: tilt ?? undefined });
    s = emptyState(sys.units, sys.dimension, box, ntypesFile + opts.extraTypes[0], sys.atomStyle);
    s.topo.nbondtypes = (h['bond types'] ?? 0) + opts.extraTypes[1];
    s.topo.nangletypes = (h['angle types'] ?? 0) + opts.extraTypes[2];
    s.topo.ndihedraltypes = (h['dihedral types'] ?? 0) + opts.extraTypes[3];
    s.topo.nimpropertypes = (h['improper types'] ?? 0) + opts.extraTypes[4];
    sys.setState(s);
    sys.ff.bond?.allocate(s.topo.nbondtypes);
    sys.ff.angle?.allocate(s.topo.nangletypes);
    sys.ff.dihedral?.allocate(s.topo.ndihedraltypes);
    sys.ff.improper?.allocate(s.topo.nimpropertypes);
  } else {
    s = s0!;
    if (ntypesFile + toff > s.ntypes) throw new StyleError(`read_data add: atom types ${ntypesFile}+${toff} exceed the box's ${s.ntypes} types (use extra/atom/types when creating it)`);
  }
  const natoms = h.atoms ?? 0;
  const style = s.atomStyle;
  const cols = STYLE_COLS[style];
  const idBase = opts.add === 'append' ? maxAtomId(s) : typeof opts.add === 'object' ? opts.add.id : 0;
  const molBase = typeof opts.add === 'object' ? opts.add.mol : 0;
  const n0 = s.n;
  const gbit = opts.group ? (sys.groups.create(opts.group), sys.groupBit(opts.group)) : 0;
  let sawAtoms = false;
  const vel = new Map<number, number[]>();
  const coeffLines: { section: string; line: string; at: number }[] = [];
  const topo: { kind: 'bonds' | 'angles' | 'dihedrals' | 'impropers'; type: number; ids: number[] }[] = [];
  const fixRows: { fx: DataSectionFix; rows: { w: string[]; at: number }[] }[] = [];
  // sections
  while (k < lines.length) {
    const title = lines[k].split('#')[0].trim();
    const hint = (lines[k].split('#')[1] ?? '').trim();
    k++;
    if (!title) continue;
    // read_data.html: "fix values = fix-ID header-string section-string" — a section named
    // section-string carries one line per atom for that fix
    const fixId = opts.fixSections.get(title);
    if (fixId !== undefined) {
      const fx = sys.fix(fixId);
      if (!isDataSectionFix(fx)) throw new StyleError(`read_data fix ${fixId}: fix style ${fx.style} does not read data-file sections`);
      const rows: { w: string[]; at: number }[] = [];
      while (rows.length < natoms && k < lines.length) {
        const t = lines[k].replace(/#.*/, '').trim();
        k++;
        if (!t) continue;
        rows.push({ w: t.split(/\s+/), at: k });
      }
      if (rows.length < natoms) throw new StyleError(`data file section ${title}: expected ${natoms} lines, found ${rows.length}`);
      fixRows.push({ fx, rows });
      continue;
    }
    if (!SECTIONS.has(title)) throw new StyleError(`unknown data file section '${title}' (line ${k})`);
    // count of lines expected
    const count = (() => {
      switch (title) {
        case 'Atoms': case 'Velocities': return natoms;
        case 'Masses': case 'Pair Coeffs': return ntypesFile;
        case 'PairIJ Coeffs': return (ntypesFile * (ntypesFile + 1)) / 2;
        case 'Bonds': return h.bonds ?? 0;
        case 'Angles': return h.angles ?? 0;
        case 'Dihedrals': return h.dihedrals ?? 0;
        case 'Impropers': return h.impropers ?? 0;
        case 'Bond Coeffs': return h['bond types'] ?? 0;
        case 'Angle Coeffs': return h['angle types'] ?? 0;
        case 'Dihedral Coeffs': return h['dihedral types'] ?? 0;
        case 'Improper Coeffs': return h['improper types'] ?? 0;
        default: throw new StyleError(`data file section '${title}' needs features the browser engine does not support (type labels, class2 cross terms or finite-size particles)`);
      }
    })();
    if (title === 'Atoms' && hint && hint !== style) sys.warn(`atom style in data file (${hint}) differs from the current atom style (${style})`);
    const body: { w: string[]; at: number }[] = [];
    // skip the blank line after the title, then read `count` non-blank lines
    while (body.length < count && k < lines.length) {
      const t = lines[k].replace(/#.*/, '').trim();
      k++;
      if (!t) continue;
      body.push({ w: t.split(/\s+/), at: k });
    }
    if (body.length < count) throw new StyleError(`data file section ${title}: expected ${count} lines, found ${body.length}`);
    switch (title) {
      case 'Atoms': {
        sawAtoms = true;
        const x = new Float64Array(3 * count), type = new Int32Array(count), id = new Int32Array(count);
        const mol = new Int32Array(count), q = new Float64Array(count), image = new Int32Array(3 * count);
        const radius = new Float64Array(count), density = new Float64Array(count);
        body.forEach(({ w, at }, a) => {
          if (w.length !== cols.length && w.length !== cols.length + 3) {
            throw new StyleError(`data file line ${at}: Atoms # ${style} expects ${cols.length} values (+3 image flags), got ${w.length}`);
          }
          for (let c = 0; c < cols.length; c++) {
            const v = w[c];
            switch (cols[c]) {
              case 'id': id[a] = intOf(v, 'atom-ID', at) + (idBase || 0); break;
              case 'mol': mol[a] = intOf(v, 'molecule-ID', at) + (intOf(v, 'molecule-ID', at) > 0 ? molBase : 0); break;
              case 'type': {
                const t = intOf(v, 'atom-type', at) + toff;
                if (t < 1 || t > s.ntypes) throw new StyleError(`data file line ${at}: atom type ${t} is outside 1..${s.ntypes}`);
                type[a] = t;
                break;
              }
              case 'q': q[a] = numOf(v, 'charge', at); break;
              case 'diameter': radius[a] = numOf(v, 'diameter', at) / 2; break;
              case 'density': density[a] = numOf(v, 'density', at); break;
              case 'x': x[3 * a] = numOf(v, 'x', at) + opts.shift[0]; break;
              case 'y': x[3 * a + 1] = numOf(v, 'y', at) + opts.shift[1]; break;
              case 'z': x[3 * a + 2] = numOf(v, 'z', at) + opts.shift[2]; break;
            }
          }
          if (w.length === cols.length + 3) {
            for (let d = 0; d < 3; d++) {
              const f = intOf(w[cols.length + d], 'image flag', at);
              image[3 * a + d] = s.box.periodic[d] ? f : 0;
            }
          }
          if (sys.dimension === 2) {
            if (Math.abs(x[3 * a + 2]) > 1e-6) throw new StyleError(`data file line ${at}: z must be 0.0 in a 2d simulation`);
            x[3 * a + 2] = 0;
          }
        });
        if (opts.add === 'merge' || opts.add === 'none') {
          const existing = new Set(Array.from(s.id.subarray(0, s.n)));
          for (const i of id) if (existing.has(i)) throw new StyleError(`read_data: atom ID ${i} already exists`);
        }
        // read_data.html: "the density is used in conjunction with the particle volume to set the mass
        // of each particle as mass = density * volume ... If the volume is 0.0, meaning a point
        // particle, then the density value is used as the mass."
        const rmass = s.radius ? radius.map((r, a) => sphereMass(r, density[a])) : undefined;
        appendAtoms(s, { x, type, id, image, molecule: mol, q, mask: gbit, radius: s.radius ? radius : undefined, rmass });
        // periodic remap of the new atoms
        for (let i = n0; i < s.n; i++) sys.geom.remap(s.x, s.image, i);
        break;
      }
      case 'Velocities':
        // read_data.html: "sphere | atom-ID vx vy vz wx wy wz"
        for (const { w, at } of body) {
          const need = s.omega ? 7 : 4;
          if (w.length < need) throw new StyleError(`data file line ${at}: Velocities needs atom-ID vx vy vz${s.omega ? ' wx wy wz' : ''}`);
          vel.set(intOf(w[0], 'atom-ID', at) + (idBase || 0), w.slice(1, need).map((t, c) => numOf(t, ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'][c], at)));
        }
        break;
      case 'Masses':
        // measured with native LAMMPS (atom_style sphere): "Cannot set mass for atom style sphere"
        if (s.atomStyle === 'sphere') throw new StyleError(`Cannot set mass for atom style ${s.atomStyle}`);
        for (const { w, at } of body) {
          const t = intOf(w[0], 'atom type', at) + toff;
          if (t < 1 || t > s.ntypes) throw new StyleError(`data file line ${at}: atom type ${t} is outside 1..${s.ntypes}`);
          const m = numOf(w[1], 'mass', at);
          if (!(m > 0)) throw new StyleError(`data file line ${at}: mass must be > 0`);
          s.massByType[t] = m;
        }
        break;
      case 'Bonds': case 'Angles': case 'Dihedrals': case 'Impropers': {
        const kind = title.toLowerCase() as 'bonds' | 'angles' | 'dihedrals' | 'impropers';
        const width = kind === 'bonds' ? 2 : kind === 'angles' ? 3 : 4;
        const off = kind === 'bonds' ? boff : kind === 'angles' ? aoff : kind === 'dihedrals' ? doff : ioff;
        const ntype = kind === 'bonds' ? s.topo.nbondtypes : kind === 'angles' ? s.topo.nangletypes : kind === 'dihedrals' ? s.topo.ndihedraltypes : s.topo.nimpropertypes;
        for (const { w, at } of body) {
          if (w.length < 2 + width) throw new StyleError(`data file line ${at}: ${title} needs ID type and ${width} atom IDs`);
          const t = intOf(w[1], 'type', at) + off;
          if (t < 1 || t > ntype) throw new StyleError(`data file line ${at}: ${title.slice(0, -1).toLowerCase()} type ${t} is outside 1..${ntype}`);
          const ids = w.slice(2, 2 + width).map((v) => intOf(v, 'atom-ID', at) + (idBase || 0));
          topo.push({ kind, type: t, ids });
        }
        break;
      }
      default:
        for (const { w, at } of body) coeffLines.push({ section: title, line: w.join(' '), at });
    }
  }
  if (natoms > 0 && !sawAtoms) throw new StyleError('data file: an Atoms section is required when atoms > 0');
  // velocities
  if (vel.size) {
    for (let i = n0; i < s.n; i++) {
      const v = vel.get(s.id[i]);
      if (v) { s.v[3 * i] = v[0]; s.v[3 * i + 1] = v[1]; s.v[3 * i + 2] = sys.dimension === 2 ? 0 : v[2]; }
      if (v && s.omega) for (let d = 0; d < 3; d++) s.omega[3 * i + d] = v[3 + d];
    }
  }
  // fix sections: "the lines of per-atom properties can be listed in any order" (fix_property_atom.html)
  if (fixRows.length) {
    const byId = new Map<number, number>();
    for (let i = n0; i < s.n; i++) byId.set(s.id[i], i);
    for (const { fx, rows } of fixRows) {
      for (const { w, at } of rows) {
        const i = byId.get(intOf(w[0], 'atom-ID', at) + (idBase || 0));
        if (i === undefined) throw new StyleError(`data file line ${at}: fix ${fx.id} section names atom ${w[0]}, which this file does not define`);
        fx.readValues(s, i, w.slice(1), at);
      }
    }
  }
  // topology must reference existing atoms
  if (topo.length) {
    const ids = new Set(Array.from(s.id.subarray(0, s.n)));
    for (const e of topo) {
      for (const id of e.ids) if (!ids.has(id)) throw new StyleError(`data file: ${e.kind.slice(0, -1)} references atom ${id}, which does not exist`);
      pushTopo(s.topo[e.kind] as TopoList, e.type, e.ids);
    }
  }
  // force-field coefficients
  if (!opts.nocoeff) {
    for (const c of coeffLines) {
      const w = c.line.split(/\s+/);
      const ctx = sys.styleContext();
      if (c.section === 'Pair Coeffs' || c.section === 'PairIJ Coeffs') {
        if (!sys.ff.pair) throw new StyleError(`data file has ${c.section} but no pair_style is defined (define pair_style before read_data, or use nocoeff)`);
        const shiftT = (t: string) => String(Number(t) + toff);
        const args = c.section === 'Pair Coeffs' ? [shiftT(w[0]), shiftT(w[0]), ...w.slice(1)] : [shiftT(w[0]), shiftT(w[1]), ...w.slice(2)];
        sys.ff.pair.coeff(args, ctx);
      } else {
        const kind = c.section.split(' ')[0].toLowerCase() as 'bond' | 'angle' | 'dihedral' | 'improper';
        const st = sys.ff[kind];
        if (!st) throw new StyleError(`data file has ${c.section} but no ${kind}_style is defined (define it before read_data, or use nocoeff)`);
        const off = kind === 'bond' ? boff : kind === 'angle' ? aoff : kind === 'dihedral' ? doff : ioff;
        st.coeff([String(Number(w[0]) + off), ...w.slice(1)], ctx);
      }
    }
  }
  sys.atomsChanged();
};

/** Shortest round-trip text, with C-style two-digit exponents (as native write_data prints). */
export const shortest = (v: number): string => {
  const t = String(v === 0 ? 0 : v);
  const m = /^(-?[\d.]+)e([+-])(\d+)$/.exec(t);
  return m ? `${m[1]}e${m[2]}${m[3].padStart(2, '0')}` : t;
};

export interface WriteDataOptions { nocoeff: boolean; pairStyle: 'ii' | 'ij' | null; nofix?: boolean }

export const writeData = (sys: System, opts: WriteDataOptions): string => {
  const s = sys.state;
  const t = s.topo;
  const out: string[] = [
    `LAMMPS data file via write_data, version 2 Sep 2026 (LAMMPS web notebook), timestep = ${s.step}, units = ${s.units.style}`,
    '',
    `${s.n} atoms`,
    `${s.ntypes} atom types`,
  ];
  const mol = isMolecularStyle(s.atomStyle);
  if (mol) {
    const lines: [number, number, string][] = [
      [t.bonds.n, t.nbondtypes, 'bond'], [t.angles.n, t.nangletypes, 'angle'],
      [t.dihedrals.n, t.ndihedraltypes, 'dihedral'], [t.impropers.n, t.nimpropertypes, 'improper'],
    ];
    for (const [n, nt, name] of lines) {
      if (name === 'angle' && s.atomStyle === 'bond') continue;
      if ((name === 'dihedral' || name === 'improper') && (s.atomStyle === 'bond' || s.atomStyle === 'angle')) continue;
      out.push(`${n} ${name}s`, `${nt} ${name} types`);
    }
  }
  out.push('');
  out.push(...[0, 1, 2].map((d) => `${shortest(s.box.lo[d])} ${shortest(s.box.hi[d])} ${'xyz'[d]}lo ${'xyz'[d]}hi`));
  if (s.box.triclinic) out.push(`${shortest(s.box.tilt[0])} ${shortest(s.box.tilt[1])} ${shortest(s.box.tilt[2])} xy xz yz`);
  // measured with native write_data: atom_style sphere (per-atom masses) writes no Masses section
  if (s.atomStyle !== 'sphere') {
    out.push('', 'Masses', '');
    for (let k = 1; k <= s.ntypes; k++) out.push(`${k} ${shortest(s.massByType[k])}`);
  }
  if (!opts.nocoeff) {
    const p = sys.ff.pair;
    if (p) {
      const ij = opts.pairStyle === 'ij';
      const lines = ij ? p.dataCoeffsIJ() : p.dataCoeffs();
      if (lines) out.push('', `${ij ? 'PairIJ' : 'Pair'} Coeffs # ${p.name}`, '', ...lines);
    }
    for (const [kind, title] of [['bond', 'Bond'], ['angle', 'Angle'], ['dihedral', 'Dihedral'], ['improper', 'Improper']] as const) {
      const st = sys.ff[kind];
      const lines = st?.dataCoeffs();
      if (st && lines) out.push('', `${title} Coeffs # ${st.name}`, '', ...lines);
    }
  }
  out.push('', `Atoms # ${s.atomStyle}`, '');
  // Measured with native LAMMPS (black box, a 10-atom chain whose Atoms
  // section was shuffled): Atoms and Velocities come out in storage order
  // (the order read_data/create_atoms added them), not sorted by ID; each
  // topology section is grouped by its owning atom in that same order (bonds
  // by the first atom, angles, dihedrals and impropers by the second), keeps
  // the read order within one owner, and is renumbered from 1. Native's
  // periodic spatial re-sort of atoms (atom_modify sort) is not reproduced.
  const order = Array.from({ length: s.n }, (_, i) => i);
  const cols = STYLE_COLS[s.atomStyle];
  for (const i of order) {
    const v = cols.map((c) => {
      switch (c) {
        case 'id': return String(s.id[i]);
        case 'mol': return String(s.molecule[i]);
        case 'type': return String(s.type[i]);
        case 'q': return shortest(s.q[i]);
        // measured with native write_data: diameter 2r, and density = mass / volume (mass itself for r = 0)
        case 'diameter': return shortest(2 * s.radius![i]);
        case 'density': return shortest(s.radius![i] > 0 ? s.rmass![i] / sphereMass(s.radius![i], 1) : s.rmass![i]);
        default: return shortest(s.x[3 * i + 'xyz'.indexOf(c)]);
      }
    });
    out.push(`${v.join(' ')} ${s.image[3 * i]} ${s.image[3 * i + 1]} ${s.image[3 * i + 2]}`);
  }
  out.push('', 'Velocities', '');
  for (const i of order) {
    const w = s.omega ? ` ${shortest(s.omega[3 * i])} ${shortest(s.omega[3 * i + 1])} ${shortest(s.omega[3 * i + 2])}` : '';
    out.push(`${s.id[i]} ${shortest(s.v[3 * i])} ${shortest(s.v[3 * i + 1])} ${shortest(s.v[3 * i + 2])}${w}`);
  }
  let maxId = 0;
  for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
  const local = new Int32Array(maxId + 1).fill(-1);
  for (let i = 0; i < s.n; i++) local[s.id[i]] = i;
  const slot = (id: number): number => (id <= maxId ? local[id] : -1);
  for (const [list, title, owner] of [[t.bonds, 'Bonds', 0], [t.angles, 'Angles', 1], [t.dihedrals, 'Dihedrals', 1], [t.impropers, 'Impropers', 1]] as const) {
    if (!list.n) continue;
    out.push('', title, '');
    const w = list.width;
    const rows = Array.from({ length: list.n }, (_, e) => e)
      .sort((a, b) => slot(list.atoms[a * w + owner]) - slot(list.atoms[b * w + owner]) || a - b);
    rows.forEach((e, k) => {
      const ids = Array.from(list.atoms.subarray(e * w, (e + 1) * w));
      out.push(`${k + 1} ${list.type[e]} ${ids.join(' ')}`);
    });
  }
  // measured with native write_data: each fix property/atom adds a section titled
  // "<fix-ID> # <names>" after the topology, one line per atom ("nofix" leaves them out)
  if (!opts.nofix) {
    for (const f of sys.fixes) {
      if (!isDataSectionFix(f)) continue;
      out.push('', f.sectionHeader(), '');
      for (const i of order) out.push(f.writeValues(s, i));
    }
  }
  return out.join('\n') + '\n';
};
