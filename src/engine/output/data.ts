import type { System } from '../system';
import type { AtomStyle, SimState, TopoList } from '../types';
import { StyleError } from '../force/types';
import { appendAtoms, atomSubStyles, countEllipsoids, ellipsoidVolume, emptyState, isEllipsoid, isMolecularStyle, isSphereStyle, maxAtomId, nativeOrder, pushTopo, sphereMass, topologyLevel } from '../atoms';
import { makeBox } from '../domain';
import { generalBoxFromRestricted, generalFrame, rotateVector, toGeneralPoint, toRestrictedPoint, unrotateVector, type GeneralFrame, type Mat3, type V3 } from '../triclinic_general';

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

const BASE_COLS: Record<string, string[]> = {
  atomic: ['id', 'type', 'x', 'y', 'z'],
  charge: ['id', 'type', 'q', 'x', 'y', 'z'],
  bond: ['id', 'mol', 'type', 'x', 'y', 'z'],
  angle: ['id', 'mol', 'type', 'x', 'y', 'z'],
  molecular: ['id', 'mol', 'type', 'x', 'y', 'z'],
  full: ['id', 'mol', 'type', 'q', 'x', 'y', 'z'],
  // read_data.html: "sphere | atom-ID atom-type diameter density x y z"
  sphere: ['id', 'type', 'diameter', 'density', 'x', 'y', 'z'],
  // read_data.html: "dipole | atom-ID atom-type q x y z mux muy muz"
  dipole: ['id', 'type', 'q', 'x', 'y', 'z', 'mux', 'muy', 'muz'],
  // read_data.html, the Atoms-section table, row ellipsoid: "atom-ID atom-type ellipsoidflag density x y z"
  ellipsoid: ['id', 'type', 'ellipsoidflag', 'density', 'x', 'y', 'z'],
};

/**
 * Atoms-line columns of an atom style. read_data.html for hybrid: "following the 5 initial values
 * (ID,type,x,y,z), specific values for each sub-style must be listed. The order of the sub-styles is
 * the same as they were listed in the atom_style command." and "if a non-standard value is defined by
 * multiple sub-styles, it only appears once in the atom line".
 */
export const atomStyleCols = (style: AtomStyle): string[] => {
  const subs = atomSubStyles(style);
  if (!style.startsWith('hybrid ')) return BASE_COLS[style];
  const out = ['id', 'type', 'x', 'y', 'z'];
  for (const sub of subs) for (const c of BASE_COLS[sub]) if (!out.includes(c)) out.push(c);
  return out;
};

const HEADER_KEYS: [RegExp, string][] = [
  [/^atoms$/, 'atoms'], [/^bonds$/, 'bonds'], [/^angles$/, 'angles'], [/^dihedrals$/, 'dihedrals'], [/^impropers$/, 'impropers'],
  [/^atom types$/, 'atom types'], [/^bond types$/, 'bond types'], [/^angle types$/, 'angle types'],
  [/^dihedral types$/, 'dihedral types'], [/^improper types$/, 'improper types'],
  [/^extra (bond|angle|dihedral|improper|special) per atom$/, 'extra'],
  [/^ellipsoids$/, 'ellipsoids'],
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
  /** read_data fix fix-ID header-string section-string (section name -> fix ID and header-string; NULL = no header lines). */
  fixSections: Map<string, { fixId: string; header: string }>;
}

/**
 * What read_data / write_data need from a fix that owns data-file lines. Every hook is optional:
 * - per-atom sections (fix property/atom): readValues(s, i, w, at) per line, sectionLines = natoms;
 * - a whole-section reader (fix cmap): readHeader for the header lines that contain the header-string,
 *   sectionLines(natoms) for the number of section lines, readSection(rows) for the section;
 * - write_data: dataHeaderLine() (a header line, after the topology counts), dataSection() (a section,
 *   after the topology), and for per-atom fixes sectionHeader() / writeValues().
 */
interface DataFix {
  id: string;
  style: string;
  readValues?(s: SimState, i: number, w: readonly string[], at: number): void;
  readHeader?(line: string, at: number): void;
  sectionLines?(natoms: number): number;
  readSection?(rows: { w: string[]; at: number }[]): void;
  sectionHeader?(): string;
  writeValues?(s: SimState, i: number): string;
  dataHeaderLine?(): string | null;
  dataSection?(slot: (id: number) => number): { title: string; lines: string[] } | null;
}
const isDataFix = (f: unknown): f is DataFix => {
  const x = f as DataFix;
  return typeof x?.readValues === 'function' || typeof x?.readSection === 'function' || typeof x?.readHeader === 'function'
    || typeof x?.dataHeaderLine === 'function' || typeof x?.dataSection === 'function';
};

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
  let orthoLines = false;
  let lo: [number, number, number] = [-0.5, -0.5, -0.5];
  let hi: [number, number, number] = [0.5, 0.5, 0.5];
  let tilt: [number, number, number] | null = null;
  // general triclinic header (Howto_triclinic.html): avec, bvec, cvec, abc origin
  const gen: { avec?: V3; bvec?: V3; cvec?: V3; origin?: V3 } = {};
  let k = 1;
  // read_data.html: "header lines containing this string will be passed to fix"
  const headerFixes = [...opts.fixSections.values()].filter((e) => e.header !== 'NULL');
  for (; k < lines.length; k++) {
    const raw = lines[k].replace(/#.*/, '').trim();
    if (!raw) continue;
    if (SECTIONS.has(raw) || SECTIONS.has(lines[k].trim().split('#')[0].trim())) break;
    const hf = headerFixes.find((e) => raw.includes(e.header));
    if (hf) {
      const fx = sys.fix(hf.fixId) as unknown as DataFix;
      if (!isDataFix(fx) || typeof fx.readHeader !== 'function') {
        throw new StyleError(`read_data fix ${hf.fixId}: header-string must be NULL (fix ${fx.style} reads no header lines)`);
      }
      fx.readHeader(raw, k + 1);
      continue;
    }
    const w = raw.split(/\s+/);
    const rest = w.slice(1).join(' ');
    const rest3 = w.slice(3).join(' ');
    const rest2 = w.slice(2).join(' ');
    if (rest2 === 'xlo xhi' || rest2 === 'ylo yhi' || rest2 === 'zlo zhi') {
      const d = 'xyz'.indexOf(rest2[0]);
      lo[d] = numOf(w[0], rest2, k + 1);
      hi[d] = numOf(w[1], rest2, k + 1);
      orthoLines = true;
      continue;
    }
    if (rest3 === 'xy xz yz') {
      tilt = [numOf(w[0], 'xy', k + 1), numOf(w[1], 'xz', k + 1), numOf(w[2], 'yz', k + 1)];
      orthoLines = true;
      continue;
    }
    if (rest3 === 'avec' || rest3 === 'bvec' || rest3 === 'cvec') {
      gen[rest3] = [numOf(w[0], rest3, k + 1), numOf(w[1], rest3, k + 1), numOf(w[2], rest3, k + 1)];
      continue;
    }
    if (rest3 === 'abc origin') {
      gen.origin = [numOf(w[0], 'abc origin', k + 1), numOf(w[1], 'abc origin', k + 1), numOf(w[2], 'abc origin', k + 1)];
      continue;
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
  // read_data.html: "For a general triclinic box, the avec, bvec, cvec, and abc origin keywords are used.
  // The xlo xhi, ylo yhi, zlo zhi, and xy xz yz keywords are NOT used."
  const isGeneral = gen.avec !== undefined || gen.bvec !== undefined || gen.cvec !== undefined || gen.origin !== undefined;
  let genFrame: GeneralFrame | null = null;
  if (isGeneral) {
    if (orthoLines) throw new StyleError('read_data: a general triclinic header (avec/bvec/cvec/abc origin) cannot be combined with xlo/xhi or xy xz yz lines');
    if (sys.dimension === 2) throw new StyleError('read_data: general triclinic data files are not supported in 2d');
    if (adding) throw new StyleError('read_data: general triclinic data files cannot be combined with add append/merge');
    if (opts.shift.some((v) => v !== 0)) throw new StyleError('read_data: shift is not supported with a general triclinic data file');
    genFrame = generalFrame({ origin: gen.origin ?? [0, 0, 0], A: gen.avec ?? [1, 0, 0], B: gen.bvec ?? [0, 1, 0], C: gen.cvec ?? [0, 0, 1] });
    lo = genFrame.lo; hi = genFrame.hi; tilt = genFrame.tilt;
  }
  let s: SimState;
  if (!adding) {
    if (sys.dimension === 2 && !(lo[2] < 0 && hi[2] > 0)) throw new StyleError('read_data: for a 2d simulation zlo and zhi must straddle zero');
    const box = makeBox({ lo, hi, boundary: sys.boundary, tilt: tilt ?? undefined });
    // the general triclinic rotation stays on the box so write_data triclinic/general can undo it
    if (genFrame) box.general = { Q: genFrame.Q };
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
  const cols = atomStyleCols(style);
  const idBase = opts.add === 'append' ? maxAtomId(s) : typeof opts.add === 'object' ? opts.add.id : 0;
  const molBase = typeof opts.add === 'object' ? opts.add.mol : 0;
  const n0 = s.n;
  const gbit = opts.group ? (sys.groups.create(opts.group), sys.groupBit(opts.group)) : 0;
  let sawAtoms = false;
  const vel = new Map<number, number[]>();
  /** Atoms with ellipsoidflag 1 waiting for their Ellipsoids line: atom ID -> density. */
  const ellPending = new Map<number, number>();
  const coeffLines: { section: string; line: string; at: number }[] = [];
  const topo: { kind: 'bonds' | 'angles' | 'dihedrals' | 'impropers'; type: number; ids: number[] }[] = [];
  const fixRows: { fx: DataFix; rows: { w: string[]; at: number }[] }[] = [];
  // sections
  while (k < lines.length) {
    const title = lines[k].split('#')[0].trim();
    const hint = (lines[k].split('#')[1] ?? '').trim();
    k++;
    if (!title) continue;
    // read_data.html: "fix values = fix-ID header-string section-string" — a section named
    // section-string carries one line per atom for that fix
    const fixEntry = opts.fixSections.get(title);
    if (fixEntry !== undefined) {
      const fixId = fixEntry.fixId;
      const fx = sys.fix(fixId) as unknown as DataFix;
      if (!isDataFix(fx) || (typeof fx.readValues !== 'function' && typeof fx.readSection !== 'function')) {
        throw new StyleError(`read_data fix ${fixId}: fix style ${fx.style} does not read data-file sections`);
      }
      // the fix says how many lines its section holds (per-atom fixes: one line per atom)
      const nlines = fx.sectionLines ? fx.sectionLines(natoms) : natoms;
      const rows: { w: string[]; at: number }[] = [];
      while (rows.length < nlines && k < lines.length) {
        const t = lines[k].replace(/#.*/, '').trim();
        k++;
        if (!t) continue;
        rows.push({ w: t.split(/\s+/), at: k });
      }
      if (rows.length < nlines) throw new StyleError(`data file section ${title}: expected ${nlines} lines, found ${rows.length}`);
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
        case 'Ellipsoids':
          if (!s.shape) throw new StyleError('data file section Ellipsoids needs atom_style ellipsoid');
          return h.ellipsoids ?? 0;
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
        const mu = s.mu ? new Float64Array(4 * count) : null;
        const eflag = s.shape ? new Uint8Array(count) : null;
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
              case 'ellipsoidflag': {
                const f = intOf(v, 'ellipsoidflag', at);
                if (f !== 0 && f !== 1) throw new StyleError(`data file line ${at}: ellipsoidflag must be 0 or 1`);
                eflag![a] = f;
                break;
              }
              case 'mux': mu![4 * a] = numOf(v, 'mux', at); break;
              case 'muy': mu![4 * a + 1] = numOf(v, 'muy', at); break;
              case 'muz': mu![4 * a + 2] = numOf(v, 'muz', at); break;
            }
          }
          if (mu) mu[4 * a + 3] = Math.hypot(mu[4 * a], mu[4 * a + 1], mu[4 * a + 2]);
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
        // ellipsoids: the density becomes a mass once the Ellipsoids section gives the volume; a point
        // particle (ellipsoidflag 0) takes the density as its mass
        const rmass = s.radius ? radius.map((r, a) => sphereMass(r, density[a])) : s.shape ? Float64Array.from(density) : undefined;
        if (eflag) for (let a = 0; a < count; a++) if (eflag[a]) ellPending.set(id[a], density[a]);
        if (genFrame) {
          // read_data.html: coordinates "should be inside the general triclinic simulation box"; the
          // general -> restricted rotation is about the box origin (Howto_triclinic.html)
          const o = gen.origin ?? [0, 0, 0];
          for (let a = 0; a < count; a++) {
            const r = toRestrictedPoint(genFrame.Q, o, [x[3 * a], x[3 * a + 1], x[3 * a + 2]]);
            x[3 * a] = r[0]; x[3 * a + 1] = r[1]; x[3 * a + 2] = r[2];
          }
        }
        appendAtoms(s, { x, type, id, image, molecule: mol, q, mask: gbit, radius: s.radius ? radius : undefined, rmass, mu: mu ?? undefined });
        // periodic remap of the new atoms
        for (let i = n0; i < s.n; i++) sys.geom.remap(s.x, s.image, i);
        break;
      }
      // read_data.html: "line syntax: atom-ID shapex shapey shapez quatw quati quatj quatk" with "shapex,shapey,shapez
      // = 3 diameters of ellipsoid"; "They must all be non-zero values."; "LAMMPS normalizes each atom's quaternion"
      case 'Ellipsoids': {
        const where = new Map<number, number>();
        for (let i = n0; i < s.n; i++) where.set(s.id[i], i);
        for (const { w, at } of body) {
          if (w.length !== 8) throw new StyleError(`data file line ${at}: Ellipsoids needs atom-ID shapex shapey shapez quatw quati quatj quatk`);
          const aid = intOf(w[0], 'atom-ID', at) + (idBase || 0);
          const i = where.get(aid);
          const dens = ellPending.get(aid);
          if (i === undefined || dens === undefined) throw new StyleError(`data file line ${at}: atom ${aid} is not an ellipsoid (ellipsoidflag 1) of this data file`);
          const sh = [1, 2, 3].map((c) => numOf(w[c], 'shape', at));
          if (sh.some((x) => !(x > 0))) throw new StyleError(`data file line ${at}: ellipsoid shape values must all be > 0`);
          const q = [4, 5, 6, 7].map((c) => numOf(w[c], 'quaternion', at));
          const qn = Math.hypot(q[0], q[1], q[2], q[3]);
          if (!(qn > 0)) throw new StyleError(`data file line ${at}: ellipsoid quaternion is zero`);
          for (let d = 0; d < 3; d++) s.shape![3 * i + d] = sh[d] / 2;
          for (let d = 0; d < 4; d++) s.quat![4 * i + d] = q[d] / qn;
          s.rmass![i] = dens * ellipsoidVolume(s, i);
          ellPending.delete(aid);
        }
        break;
      }
      case 'Velocities':
        // read_data.html: "sphere | atom-ID vx vy vz wx wy wz"
        if (s.omega && s.angmom) throw new StyleError('data file Velocities: hybrid styles with both sphere and ellipsoid sub-styles are not supported by the browser engine');
        for (const { w, at } of body) {
          // read_data.html: "ellipsoid | atom-ID vx vy vz lx ly lz"
          const need = s.omega || s.angmom ? 7 : 4;
          if (w.length < need) throw new StyleError(`data file line ${at}: Velocities needs atom-ID vx vy vz${s.omega ? ' wx wy wz' : s.angmom ? ' lx ly lz' : ''}`);
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
  if (ellPending.size) throw new StyleError(`data file: ${ellPending.size} atoms have ellipsoidflag 1 but no Ellipsoids line`);
  // velocities
  if (vel.size) {
    for (let i = n0; i < s.n; i++) {
      const v = vel.get(s.id[i]);
      if (v) {
        // velocities (and angular velocities) rotate with the box: read_data.html "should be specified for the rotated coordinate axes"
        const lin = genFrame ? rotateVector(genFrame.Q, [v[0], v[1], v[2]]) : [v[0], v[1], v[2]];
        s.v[3 * i] = lin[0]; s.v[3 * i + 1] = lin[1]; s.v[3 * i + 2] = sys.dimension === 2 ? 0 : lin[2];
        if (s.omega) {
          const ang = genFrame ? rotateVector(genFrame.Q, [v[3], v[4], v[5]]) : [v[3], v[4], v[5]];
          for (let d = 0; d < 3; d++) s.omega[3 * i + d] = ang[d];
        }
        if (s.angmom) {
          if (genFrame) throw new StyleError('read_data: ellipsoid angular momenta with a general triclinic box are not supported by the browser engine');
          for (let d = 0; d < 3; d++) s.angmom[3 * i + d] = v[3 + d];
        }
      }
    }
  }
  // fix sections: "the lines of per-atom properties can be listed in any order" (fix_property_atom.html)
  if (fixRows.length) {
    const byId = new Map<number, number>();
    for (let i = n0; i < s.n; i++) byId.set(s.id[i], i);
    for (const { fx, rows } of fixRows) {
      // whole-section fixes (fix cmap) take their rows as they are; the IDs are checked when the fix acts
      if (fx.readSection) {
        if (adding) throw new StyleError(`read_data fix ${fx.id} (${fx.style}): add append/merge is not supported with this fix's section`);
        fx.readSection(rows);
        continue;
      }
      for (const { w, at } of rows) {
        const i = byId.get(intOf(w[0], 'atom-ID', at) + (idBase || 0));
        if (i === undefined) throw new StyleError(`data file line ${at}: fix ${fx.id} section names atom ${w[0]}, which this file does not define`);
        fx.readValues!(s, i, w.slice(1), at);
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

export interface WriteDataOptions { nocoeff: boolean; pairStyle: 'ii' | 'ij' | null; nofix?: boolean; triclinicGeneral?: boolean }

/** Maps every 3-vector of a flat array (x, v, omega, ...) through f; out is a new array. */
const mapVectors = (arr: ArrayLike<number>, f: (v: V3) => V3): Float64Array => {
  const out = Float64Array.from(arr);
  for (let i = 0; i + 2 < arr.length; i += 3) {
    const r = f([arr[i], arr[i + 1], arr[i + 2]]);
    out[i] = r[0]; out[i + 1] = r[1]; out[i + 2] = r[2];
  }
  return out;
};

/**
 * write_data triclinic/general (write_data.html: "write data file in general triclinic format"; Howto_triclinic.html:
 * "is effectively the inverse of the operation described in the" preceding bullet). The box edges come back from the restricted box
 * with the stored rotation; the origin is the box lower-left corner, which the rotation keeps fixed.
 */
const generalOutput = (s: SimState): { Q: Mat3; origin: V3; header: string[] } => {
  const Q = s.box.general?.Q;
  if (!Q) throw new StyleError('write_data triclinic/general needs a general triclinic box (create_box NULL, or a general triclinic data file)');
  if (s.shape) throw new StyleError('write_data triclinic/general: ellipsoid orientations (quaternions) are not rotated by the browser engine');
  const origin: V3 = [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
  const g = generalBoxFromRestricted(Q, s.box.lo, s.box.hi, s.box.tilt);
  const edge = (v: V3, name: string) => `${shortest(v[0])} ${shortest(v[1])} ${shortest(v[2])} ${name}`;
  return {
    Q,
    origin,
    header: [edge(g.A, 'avec'), edge(g.B, 'bvec'), edge(g.C, 'cvec'), `${shortest(origin[0])} ${shortest(origin[1])} ${shortest(origin[2])} abc origin`],
  };
};

export const writeData = (sys: System, opts: WriteDataOptions): string => {
  const s = sys.state;
  const t = s.topo;
  const gen = opts.triclinicGeneral ? generalOutput(s) : null;
  const out: string[] = [
    `LAMMPS data file via write_data, version 2 Sep 2026 (LAMMPS web notebook), timestep = ${s.step}, units = ${s.units.style}`,
    '',
    `${s.n} atoms`,
    `${s.ntypes} atom types`,
  ];
  // measured with native write_data: N ellipsoids follows the atom types
  const nEll = s.shape ? countEllipsoids(s) : 0;
  if (s.shape) out.push(`${nEll} ellipsoids`);
  const mol = isMolecularStyle(s.atomStyle);
  if (mol) {
    const lines: [number, number, string][] = [
      [t.bonds.n, t.nbondtypes, 'bond'], [t.angles.n, t.nangletypes, 'angle'],
      [t.dihedrals.n, t.ndihedraltypes, 'dihedral'], [t.impropers.n, t.nimpropertypes, 'improper'],
    ];
    for (const [n, nt, name] of lines) {
      const lv = topologyLevel(s.atomStyle);
      if (name === 'angle' && lv < 2) continue;
      if ((name === 'dihedral' || name === 'improper') && lv < 3) continue;
      out.push(`${n} ${name}s`, `${nt} ${name} types`);
    }
  }
  // fix cmap header line N crossterms (measured with native write_data: after the topology counts;
  // the nofix keyword leaves it out)
  if (!opts.nofix) {
    for (const f of sys.fixes) {
      const hl = (f as unknown as DataFix).dataHeaderLine?.();
      if (hl) out.push(hl);
    }
  }
  out.push('');
  if (gen) out.push(...gen.header);
  else {
    out.push(...[0, 1, 2].map((d) => `${shortest(s.box.lo[d])} ${shortest(s.box.hi[d])} ${'xyz'[d]}lo ${'xyz'[d]}hi`));
    if (s.box.triclinic) out.push(`${shortest(s.box.tilt[0])} ${shortest(s.box.tilt[1])} ${shortest(s.box.tilt[2])} xy xz yz`);
  }
  // measured with native write_data: atom_style sphere (per-atom masses) writes no Masses section; a
  // hybrid style with sphere writes it (per-type and per-atom masses both exist there)
  if (s.atomStyle !== 'sphere' && s.atomStyle !== 'ellipsoid') {
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
  // measured with native write_data: a hybrid style is labelled Atoms # hybrid
  out.push('', `Atoms # ${s.atomStyle.startsWith('hybrid ') ? 'hybrid' : s.atomStyle}`, '');
  // Measured with native LAMMPS (black box, a 10-atom chain whose Atoms
  // section was shuffled): Atoms and Velocities come out in storage order
  // (the order read_data/create_atoms added them), not sorted by ID; each
  // topology section is grouped by its owning atom in that same order (bonds
  // by the first atom, angles, dihedrals and impropers by the second), keeps
  // the read order within one owner, and is renumbered from 1. Storage order
  // is native's (SimState.order, including its spatial sort at run setup).
  const order = nativeOrder(s);
  const cols = atomStyleCols(s.atomStyle);
  // general output: positions, dipole directions and velocities rotated to the general frame
  const xo = gen ? mapVectors(s.x, (p) => toGeneralPoint(gen.Q, gen.origin, p)) : s.x;
  const mu = gen && s.mu ? Float64Array.from(s.mu) : s.mu;
  if (gen && mu) for (let i = 0; i < s.n; i++) {
    const m = unrotateVector(gen.Q, [mu[4 * i], mu[4 * i + 1], mu[4 * i + 2]]);
    mu[4 * i] = m[0]; mu[4 * i + 1] = m[1]; mu[4 * i + 2] = m[2];
  }
  const vo = gen ? mapVectors(s.v, (w) => unrotateVector(gen.Q, w)) : s.v;
  for (const i of order) {
    const v = cols.map((c) => {
      switch (c) {
        case 'id': return String(s.id[i]);
        case 'mol': return String(s.molecule[i]);
        case 'type': return String(s.type[i]);
        case 'q': return shortest(s.q[i]);
        // measured with native write_data: diameter 2r, and density = mass / volume (mass itself for r = 0)
        case 'diameter': return shortest(2 * s.radius![i]);
        case 'density':
          if (s.radius) return shortest(s.radius[i] > 0 ? s.rmass![i] / sphereMass(s.radius[i], 1) : s.rmass![i]);
          return shortest(isEllipsoid(s, i) ? s.rmass![i] / ellipsoidVolume(s, i) : s.rmass![i]);
        case 'ellipsoidflag': return isEllipsoid(s, i) ? '1' : '0';
        case 'mux': return shortest(mu![4 * i]);
        case 'muy': return shortest(mu![4 * i + 1]);
        case 'muz': return shortest(mu![4 * i + 2]);
        default: return shortest(xo[3 * i + 'xyz'.indexOf(c)]);
      }
    });
    out.push(`${v.join(' ')} ${s.image[3 * i]} ${s.image[3 * i + 1]} ${s.image[3 * i + 2]}`);
  }
  out.push('', 'Velocities', '');
  const angRaw = s.omega ?? s.angmom;
  const ang = angRaw && gen ? mapVectors(angRaw, (w) => unrotateVector(gen.Q, w)) : angRaw;
  for (const i of order) {
    const w = ang ? ` ${shortest(ang[3 * i])} ${shortest(ang[3 * i + 1])} ${shortest(ang[3 * i + 2])}` : '';
    out.push(`${s.id[i]} ${shortest(vo[3 * i])} ${shortest(vo[3 * i + 1])} ${shortest(vo[3 * i + 2])}${w}`);
  }
  // measured with native write_data: an Ellipsoids section (diameters and quaternion) after Velocities
  if (s.shape && nEll > 0) {
    out.push('', 'Ellipsoids', '');
    for (const i of order) {
      if (!isEllipsoid(s, i)) continue;
      const sh = [0, 1, 2].map((d) => shortest(2 * s.shape![3 * i + d]));
      const q = [0, 1, 2, 3].map((d) => shortest(s.quat![4 * i + d]));
      out.push(`${s.id[i]} ${sh.join(' ')} ${q.join(' ')}`);
    }
  }
  let maxId = 0;
  for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
  const local = new Int32Array(maxId + 1).fill(-1);
  order.forEach((i, k) => { local[s.id[i]] = k; });
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
  // fix cmap: the N crossterms header and a CMAP section after the topology (measured with native
  // write_data; the nofix keyword leaves both out)
  if (!opts.nofix) {
    for (const f of sys.fixes) {
      const sec = (f as unknown as DataFix).dataSection?.(slot);
      if (!sec) continue;
      out.push('', sec.title, '');
      for (const line of sec.lines) out.push(line);
    }
  }
  // measured with native write_data: each fix property/atom adds a section titled
  // <fix-ID> # <names> after the topology, one line per atom (the nofix keyword leaves them out)
  if (!opts.nofix) {
    for (const f of sys.fixes) {
      const df = f as unknown as DataFix;
      if (typeof df.sectionHeader !== 'function' || typeof df.writeValues !== 'function') continue;
      out.push('', df.sectionHeader(), '');
      for (const i of order) out.push(df.writeValues(s, i));
    }
  }
  return out.join('\n') + '\n';
};
