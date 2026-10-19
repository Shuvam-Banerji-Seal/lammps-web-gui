import type { System } from '../system';
import type { AtomStyle, SimState, TopoList } from '../types';
import { StyleError } from '../force/types';
import { isSmdStyle, SMD_CONTACT } from '../smd';
import { isSphStyle, SPH_CV, SPH_DESPH, SPH_DRHO, SPH_ESPH, SPH_RHO, SPH_VEST, appendAtoms, setOxdnaPartners, setSpinVector, atomSubStyles, countEllipsoids, ellipsoidVolume, emptyState, isEllipsoid, isMolecularStyle, isTemplateStyle, maxAtomId, nativeOrder, pushTopo, sphereMass, templateStyleId, topologyLevel, isLineStyle, isTriStyle, isLineAtom, isTriAtom, LINE_FLAG, LINE_BONUS, TRI_FLAG, TRI_BONUS, TRI_BONUS_COLS, ELECTRON_RADIUS, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle, isDielectricStyle, DIEL_AREA, DIEL_ED, DIEL_EM, DIEL_EPSILON, DIEL_CURVATURE, DIEL_QSCALED, isEdpdStyle, isMdpdStyle, tdpdSpecies, EDPD_TEMP, EDPD_CV, MDPD_RHO, TDPD_CC, RHEO_RHO, RHEO_STATUS, isRheoStyle, isRheoThermalStyle, RHEO_ESPH, isDpdStyle, DPD_THETA } from '../atoms';
import { lineFromEnds, lineEnds, triangleCorner, triangleAreaOfBody, type LineBonus } from '../lineTri';
import { triBodyHook } from '../styles';
import type { TriBody } from '../triBonus';

import { expandTemplateTopology } from '../template';
import { BODY_RECORD, POLY_OFF, bodyRecordOffset, bodyRecordStride, parseBodyStyle, polyRecordLength, polyStoredCounts, polyhedronFrame, principalFrame } from '../bodies';
import { makeBox } from '../domain';
import { LABEL_KINDS, LABEL_SECTION, isValidLabel, labelKindOfSection, type LabelKind } from '../labelmap';
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
  // read_data.html, the Atoms-section table, row amoeba (native: "Atoms # amoeba" lines carry the same 7 columns as full)
  amoeba: ['id', 'mol', 'type', 'q', 'x', 'y', 'z'],
  // read_data.html: "sphere | atom-ID atom-type diameter density x y z"
  sphere: ['id', 'type', 'diameter', 'density', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row bpm/sphere: "atom-ID molecule-ID atom-type diameter density x y z"
  'bpm/sphere': ['id', 'mol', 'type', 'diameter', 'density', 'x', 'y', 'z'],
  // read_data.html: "dipole | atom-ID atom-type q x y z mux muy muz"
  dipole: ['id', 'type', 'q', 'x', 'y', 'z', 'mux', 'muy', 'muz'],
  // read_data.html, Atoms-section table, row spin: atom-ID atom-type x y z spx spy spz sp
  spin: ['id', 'type', 'x', 'y', 'z', 'spx', 'spy', 'spz', 'sp'],
  // read_data.html, the Atoms-section table, row ellipsoid: "atom-ID atom-type ellipsoidflag density x y z"
  ellipsoid: ['id', 'type', 'ellipsoidflag', 'density', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row peri: "atom-ID atom-type volume density x y z"
  peri: ['id', 'type', 'volume', 'density', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row template. Measured with native LAMMPS (black box):
  // the column order is atom-ID molecule-ID template-index template-atom atom-type x y z (the docs row
  // lists atom-type before molecule-ID; native rejects that order: Invalid atom type)
  template: ['id', 'mol', 'tindex', 'tatom', 'type', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row body: "atom-ID atom-type bodyflag mass x y z"
  body: ['id', 'type', 'bodyflag', 'mass', 'x', 'y', 'z'],
  // atom_style.html, oxdna: "*atomic* + id5p" (the 5'/3' partners come from the Bonds section): no Atoms-line column of its own
  oxdna: ['id', 'type', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row line: "atom-ID molecule-ID atom-type lineflag density x y z"
  line: ['id', 'mol', 'type', 'lineflag', 'density', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row tri: "atom-ID molecule-ID atom-type triangleflag density x y z"
  tri: ['id', 'mol', 'type', 'triangleflag', 'density', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row electron: "atom-ID atom-type q espin eradius x y z"
  electron: ['id', 'type', 'q', 'espin', 'eradius', 'x', 'y', 'z'],
  // atom_style.html, sph: "The sph style is for smoothed particle hydrodynamics (SPH) particles which store a density (rho),
  // energy (esph), and heat capacity (cv)"; read_data.html, the Atoms-section table, row sph: "atom-ID atom-type rho esph cv x y z"
  sph: ['id', 'type', 'rho', 'esph', 'cv', 'x', 'y', 'z'],
  // read_data.html, Atoms section, dielectric: "atom-ID atom-type q x y z mux muy muz area ed em epsilon curvature";
  // atom_style.html: "full + mu, area, ed, em, epsilon, curvature, q_scaled". Measured with native LAMMPS (black box): a
  // dielectric data line has a molecule ID after the atom ID (15 columns, as the examples/PACKAGES/dielectric data files)
  dielectric: ['id', 'mol', 'type', 'q', 'x', 'y', 'z', 'mux', 'muy', 'muz', 'area', 'ed', 'em', 'epsilon', 'curvature'],
  // read_data.html, the Atoms-section table: "edpd: atom-ID atom-type edpd_temp edpd_cv x y z"; "mdpd: atom-ID atom-type rho x y z";
  // "tdpd: atom-ID atom-type x y z cc1 cc2 ... ccNspecies" (the cc columns are added by atomStyleCols from the species count)
  edpd: ['id', 'type', 'edpd_temp', 'edpd_cv', 'x', 'y', 'z'],
  mdpd: ['id', 'type', 'rho', 'x', 'y', 'z'],
  tdpd: ['id', 'type', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row rheo: "atom-ID atom-type status rho x y z" (atom_style.html: "atomic + rho, status")
  rheo: ['id', 'type', 'rheo_status', 'rho', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row rheo/thermal: "atom-ID atom-type status rho energy x y z"
  'rheo/thermal': ['id', 'type', 'rheo_status', 'rho', 'energy', 'x', 'y', 'z'],
  // read_data.html, Atoms section for atom_style dpd (the data files of examples/PACKAGES/dpd-react): "atom-ID atom-type
  // dpdTheta x y z"; atom_style.html: "dpd" is "atomic + rho + reactive DPD data", so the energies are not in the data line.
  dpd: ['id', 'type', 'dpdTheta', 'x', 'y', 'z'],
  // atom_style.html: apip is "*atomic* +" the APIP fields; as native LAMMPS does (src/APIP/atom_vec_apip.cpp at
  // b332f28, fields_data_atom = id, type, x), the Atoms line is that of atomic and the fields start at zero
  apip: ['id', 'type', 'x', 'y', 'z'],
  // read_data.html, the Atoms-section table, row smd: "atom-ID atom-type molecule volume mass kradius cradius x0 y0 z0 x y z"
  // (kradius = kernel radius, cradius = contact radius). The x0 columns are read and then replaced by x, as native
  // AtomVecSMD::data_atom_post does ("reset x0 b/c x may have been modified in Atom::data_atoms()").
  smd: ['id', 'type', 'mol', 'vfrac', 'rmass', 'kradius', 'cradius', 'x0', 'y0', 'z0', 'x', 'y', 'z'],
};

/**
 * Atoms-line columns of an atom style. read_data.html for hybrid: "following the 5 initial values
 * (ID,type,x,y,z), specific values for each sub-style must be listed. The order of the sub-styles is
 * the same as they were listed in the atom_style command." and "if a non-standard value is defined by
 * multiple sub-styles, it only appears once in the atom line".
 */
export const atomStyleCols = (style: AtomStyle): string[] => {
  const subs = atomSubStyles(style);
  // atom_style tdpd N: one cc column per chemical species after the coordinates (read_data.html: "ccN = chemical concentration")
  const withSpecies = (sub: string, cols: string[]): string[] => {
    if (sub !== 'tdpd') return cols;
    const nsp = tdpdSpecies(style);
    return [...cols, ...Array.from({ length: nsp }, (_, k) => `cc${k + 1}`)];
  };
  if (!style.startsWith('hybrid ')) return withSpecies(subs[0], BASE_COLS[subs[0]]);
  const out = ['id', 'type', 'x', 'y', 'z'];
  for (const sub of subs) for (const c of withSpecies(sub, BASE_COLS[sub])) if (!out.includes(c)) out.push(c);
  return out;
};

const HEADER_KEYS: [RegExp, string][] = [
  [/^atoms$/, 'atoms'], [/^bonds$/, 'bonds'], [/^angles$/, 'angles'], [/^dihedrals$/, 'dihedrals'], [/^impropers$/, 'impropers'],
  [/^atom types$/, 'atom types'], [/^bond types$/, 'bond types'], [/^angle types$/, 'angle types'],
  [/^dihedral types$/, 'dihedral types'], [/^improper types$/, 'improper types'],
  [/^extra (bond|angle|dihedral|improper|special) per atom$/, 'extra'],
  [/^ellipsoids$/, 'ellipsoids'], [/^bodies$/, 'bodies'], [/^lines$/, 'lines'], [/^triangles$/, 'triangles'],
];

/** Sections whose lines name atom or molecular types: a Type Labels section must come before them (read_data.html). */
const TYPE_SECTIONS = new Set([
  'Atoms', 'Masses', 'Bonds', 'Angles', 'Dihedrals', 'Impropers', 'Pair Coeffs', 'PairIJ Coeffs',
  'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs',
]);

const SECTIONS = new Set([
  'Atoms', 'Velocities', 'Masses', 'Bonds', 'Angles', 'Dihedrals', 'Impropers',
  'Pair Coeffs', 'PairIJ Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs',
  'Ellipsoids', 'Lines', 'Triangles', 'Bodies',
  'Atom Type Labels', 'Bond Type Labels', 'Angle Type Labels', 'Dihedral Type Labels', 'Improper Type Labels',
  'BondBond Coeffs', 'BondAngle Coeffs', 'UreyBradley Coeffs', 'MiddleBondTorsion Coeffs', 'EndBondTorsion Coeffs', 'AngleTorsion Coeffs',
  'AngleAngleTorsion Coeffs', 'BondBond13 Coeffs', 'AngleAngle Coeffs',
]);

/*
 * class 2 force-field sections (docs.lammps.org/read_data.html: "*BondBond Coeffs, BondAngle Coeffs,
 * MiddleBondTorsion Coeffs, EndBondTorsion Coeffs, AngleTorsion Coeffs, AngleAngleTorsion Coeffs,
 * BondBond13 Coeffs, AngleAngle Coeffs* = class 2 force field sections"). Each line is "ID coeffs" with
 * "coeffs = list of coeffs (see class 2 section of angle_coeff)" (dihedral_coeff, improper_coeff for the
 * others), so a line is the style's coeff command with the cross-term keyword of angle_class2.html
 * (bb, ba), dihedral_class2.html (mbt, ebt, at, aat, bb13) and improper_class2.html (aa) put after the ID.
 */
const CLASS2_SECTIONS: Record<string, ['angle' | 'dihedral' | 'improper', string]> = {
  'BondBond Coeffs': ['angle', 'bb'],
  'BondAngle Coeffs': ['angle', 'ba'],
  // AMOEBA (angle_style amoeba): "ID k r0" is the angle_coeff N ub k r0 line. Measured with native LAMMPS (black box): the
  // AMOEBA water data file reads its UreyBradley Coeffs section.
  'UreyBradley Coeffs': ['angle', 'ub'],
  'MiddleBondTorsion Coeffs': ['dihedral', 'mbt'],
  'EndBondTorsion Coeffs': ['dihedral', 'ebt'],
  'AngleTorsion Coeffs': ['dihedral', 'at'],
  'AngleAngleTorsion Coeffs': ['dihedral', 'aat'],
  'BondBond13 Coeffs': ['dihedral', 'bb13'],
  'AngleAngle Coeffs': ['improper', 'aa'],
};

for (const title of Object.keys(CLASS2_SECTIONS)) TYPE_SECTIONS.add(title);

export interface ReadDataOptions {
  add: 'none' | 'append' | 'merge' | { id: number; mol: number };
  offset: [number, number, number, number, number];
  shift: [number, number, number];
  extraTypes: [number, number, number, number, number];
  group: string | null;
  nocoeff: boolean;
  /** read_data fix fix-ID header-string section-string (section name -> fix ID and header-string; NULL = no header lines). */
  fixSections: Map<string, { fixId: string; header: string }>;
  /** read_data extra/bond/per/atom (null = keyword not given); used only by the read that creates the box. */
  extraBondPerAtom: number | null;
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
  /** title: the section name (a fix with several sections, e.g. fix amoeba/pitorsion, tells them apart by it). */
  sectionLines?(natoms: number, title?: string): number;
  readSection?(rows: { w: string[]; at: number }[], title?: string): void;
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
  extraBondPerAtom: null,
});

/**
 * The Bodies section (read_data.html): "Lines are read one at a time until Ninteger values are read." The
 * header line of each body is atom-ID Ninteger Ndouble, followed by the integer and then the double values.
 */
const readBodyLines = (lines: string[], k0: number, nbodies: number): { bodies: { id: number; ints: number[]; dbls: number[]; at: number }[]; k: number } => {
  let k = k0;
  const next = (): { w: string[]; at: number } | null => {
    while (k < lines.length) {
      const t = lines[k].replace(/#.*/, '').trim();
      k++;
      if (t) return { w: t.split(/\s+/), at: k };
    }
    return null;
  };
  const bodies: { id: number; ints: number[]; dbls: number[]; at: number }[] = [];
  for (let b = 0; b < nbodies; b++) {
    const h = next();
    if (!h) throw new StyleError(`data file section Bodies: expected ${nbodies} bodies, found ${b}`);
    if (h.w.length !== 3) throw new StyleError(`data file line ${h.at}: Bodies header needs atom-ID Ninteger Ndouble`);
    const id = intOf(h.w[0], 'atom-ID', h.at), nint = intOf(h.w[1], 'Ninteger', h.at), ndbl = intOf(h.w[2], 'Ndouble', h.at);
    const ints: number[] = [], dbls: number[] = [];
    while (ints.length < nint) {
      const l = next();
      if (!l) throw new StyleError(`data file line ${h.at}: body ${id} lists ${ints.length} of ${nint} integers`);
      for (const t of l.w) ints.push(intOf(t, 'body integer', l.at));
      if (ints.length > nint) throw new StyleError(`data file line ${l.at}: body ${id} has more than ${nint} integers`);
    }
    while (dbls.length < ndbl) {
      const l = next();
      if (!l) throw new StyleError(`data file line ${h.at}: body ${id} lists ${dbls.length} of ${ndbl} doubles`);
      for (const t of l.w) dbls.push(numOf(t, 'body double', l.at));
      if (dbls.length > ndbl) throw new StyleError(`data file line ${l.at}: body ${id} has more than ${ndbl} doubles`);
    }
    bodies.push({ id, ints, dbls, at: h.at });
  }
  return { bodies, k };
};

/** Number of atoms whose line or triangle flag (LINE_FLAG or TRI_FLAG) is 1. */
const flagCount = (s: SimState, key: string): number => {
  const f = s.custom.get(key)!.data;
  let n = 0;
  for (let i = 0; i < s.n; i++) if (f[i]) n++;
  return n;
};

/** The body-frame corners of triangle i (TRI_BONUS, 9 values). */
const triCorners = (s: SimState, i: number): Float64Array => s.custom.get(TRI_BONUS)!.data.subarray(TRI_BONUS_COLS * i, TRI_BONUS_COLS * i + 9);

/**
 * Point q moved by whole periodic box lengths so that it is the image nearest to r (orthogonal box). read_data.html:
 * "even if individual end points are outside the simulation box"; measured with native read_data (box 0 to 10, atom at
 * x 0.5, end points 9.5 and 11.5 are read as -0.5 and 1.5; refused with a non-periodic x boundary).
 */
const remapNear3 = (s: SimState, q: [number, number, number], r: [number, number, number]): [number, number, number] => {
  const out: [number, number, number] = [q[0], q[1], q[2]];
  for (let d = 0; d < 3; d++) {
    if (!s.box.periodic[d]) continue;
    const L = s.box.hi[d] - s.box.lo[d];
    while (out[d] - r[d] > 0.5 * L) out[d] -= L;
    while (out[d] - r[d] < -0.5 * L) out[d] += L;
  }
  return out;
};

/** The two end points of a line segment, each moved to the periodic image nearest atom i's position (2d: x and y). */
const remapEndsNear = (s: SimState, i: number, ends: [number, number][]): [number, number][] => {
  const r: [number, number, number] = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
  return ends.map(([x, y]) => { const q = remapNear3(s, [x, y, 0], r); return [q[0], q[1]] as [number, number]; });
};

/**
 * read_data.html: the centre of the end points or corners "should be values close to the center point" of the atom.
 * Measured with native read_data: the tolerance is 0.001 of the size, not an absolute distance. For a segment the
 * size is its length (offset 0.0015 on length 2 accepted, 0.0025 refused; 0.05 on length 100 accepted). For a
 * triangle the size is its largest edge from the first corner (offset 0.0049 accepted, 0.0051 refused, for a
 * triangle with edges 5 and 5.001 from its first corner and 7 between the other two corners).
 */
const checkCentre = (s: SimState, i: number, c: readonly number[], size: number, msg: string, at: number): void => {
  const d = Math.hypot(c[0] - s.x[3 * i], c[1] - s.x[3 * i + 1], c[2] - s.x[3 * i + 2]);
  if (d / size > 0.001) throw new StyleError(`data file line ${at}: ${msg}`);
  // the stored position becomes the exact centre (measured with native read_data: an atom at x 5.0015 on a segment
  // centred at 5 is written back at x 5)
  s.x[3 * i] = c[0]; s.x[3 * i + 1] = c[1]; s.x[3 * i + 2] = c[2];
};

/**
 * Line bonus: length, theta, radius = half the length, rmass = density * length. Measured with native read_data and
 * compute property/atom: length 2 with density 3 gives radius 1 and mass 6; length 4 with density 1 gives radius 2
 * and mass 4; a point particle (lineflag 0) has radius 0 and mass = density.
 */
const setLineBonus = (s: SimState, i: number, lb: LineBonus, density: number): void => {
  s.custom.get(LINE_FLAG)!.data[i] = 1;
  const bonus = s.custom.get(LINE_BONUS)!.data;
  bonus[2 * i] = lb.length;
  bonus[2 * i + 1] = lb.theta;
  s.radius![i] = 0.5 * lb.length;
  s.rmass![i] = density * lb.length;
};

/**
 * Triangle bonus: rmass = density * area, radius = the largest distance from the centroid to a corner, orientation
 * quaternion and principal moments (the engine's own layout). Measured with native compute property/atom: a right
 * isoceles triangle with legs 2 (centroid (2/3, 2/3)) has radius 1.49071, not its circumradius sqrt(2); corners
 * (0,0), (5,0), (0.1,5) have radius 3.69745 and area 12.5; density 1.5 on area 2 gives mass 3.
 */
const setTriBonus = (s: SimState, i: number, tb: TriBody): void => {
  s.custom.get(TRI_FLAG)!.data[i] = 1;
  const bonus = s.custom.get(TRI_BONUS)!.data;
  for (let k = 0; k < 9; k++) bonus[TRI_BONUS_COLS * i + k] = tb.corners[k];
  for (let d = 0; d < 3; d++) bonus[TRI_BONUS_COLS * i + 9 + d] = tb.moments[d];
  s.radius![i] = tb.radius;
  s.rmass![i] = s.rmass![i] * tb.area;
  for (let d = 0; d < 4; d++) s.quat![4 * i + d] = tb.quat[d];
};

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
  // a read after the box exists (add keyword) is a subsequent read: its bonds are limited by the storage
  const subsequentRead = sys.hasBox;
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
  // atom_style line and tri: the dimension is fixed. Measured with native read_data: a line style in a 3d box fails with
  // an error that says atom_style line can only be used in 2d simulations, and a tri style in a 2d box with an error that
  // says atom_style tri can only be used in 3d simulations.
  if (isLineStyle(style) && s.dimension !== 2) throw new StyleError('Atom_style line can only be used in 2d simulations');
  if (isTriStyle(style) && s.dimension !== 3) throw new StyleError('Atom_style tri can only be used in 3d simulations');
  if ((isLineStyle(style) || isTriStyle(style)) && (genFrame || s.box.triclinic)) throw new StyleError(`read_data: ${isLineStyle(style) ? 'line' : 'tri'} particles in a triclinic box are not supported by the browser engine`);
  // atom_style hybrid with a tri sub-style: the Atoms-line columns of the union are not checked against native (the
  // data_atom column order of atom_vec_hybrid.cpp for two sub-styles that share density); tri particles are read in their own style
  if (style.startsWith('hybrid ') && isTriStyle(style)) throw new StyleError('read_data: atom_style hybrid with a tri sub-style is not supported by the browser engine (use atom_style tri alone, or create the particles with fix surface/local)');
  // atom_style template: read_data.html "set the *bonds*, *angles*, etc header keywords in the data file," (see template.ts)
  const tmplId = templateStyleId(style);
  if ((h.bodies ?? 0) > 0 && !parseBodyStyle(style)) throw new StyleError(`data file header bodies needs atom_style body (current: ${style})`);
  if (parseBodyStyle(style)?.bstyle === 'rounded/polygon' && s.dimension !== 2) throw new StyleError('atom_style body rounded/polygon can only be used in 2d simulations (dimension 2)');
  if (parseBodyStyle(style) && genFrame) throw new StyleError('read_data: body particles in a general triclinic box are not supported by the browser engine');
  if (tmplId !== null) {
    for (const k of ['bonds', 'angles', 'dihedrals', 'impropers'] as const) {
      if ((h[k] ?? 0) > 0) throw new StyleError(`data file header ${k}: the topology of atom_style template comes from the molecule templates`);
    }
  }
  const idBase = opts.add === 'append' ? maxAtomId(s) : typeof opts.add === 'object' ? opts.add.id : 0;
  const molBase = typeof opts.add === 'object' ? opts.add.mol : 0;
  const n0 = s.n;
  const gbit = opts.group ? (sys.groups.create(opts.group), sys.groupBit(opts.group)) : 0;
  let sawAtoms = false;
  const vel = new Map<number, number[]>();
  /** Atoms with ellipsoidflag 1 waiting for their Ellipsoids line: atom ID -> density. */
  const ellPending = new Map<number, number>();
  /** atom_style line / tri: atoms with lineflag 1 (triangleflag 1) waiting for their Lines (Triangles) line: atom ID -> density. */
  const linePending = new Map<number, number>();
  const triPending = new Map<number, number>();
  // read_data.html: a Type Labels section must come before any section involving types ("Must read Type Labels before any section involving types")
  let typesRead = false;
  /**
   * atom_style body rounded/polyhedron: the Bodies entries (native body_rounded_polyhedron.cpp data_body). The
   * record of each body holds its counts, the principal moments, the enclosing and rounded radii, the body-frame
   * vertices, edge ends and faces (bodies.ts). Checks as native: the counts, the number of doubles, the edge
   * vertices (in range, distinct), the face vertices (in range, the first three not on a line), and the two
   * vertices of a rod (distinct).
   */
  const readPolyhedronBodies = (bodies: { id: number; ints: number[]; dbls: number[]; at: number }[]): void => {
    const parsed = bodies.map((b) => {
      if (b.ints.length !== 3) throw new StyleError(`data file line ${b.at}: Incorrect # of integer values in Bodies section of data file (body rounded/polyhedron has nvertices nedges nfaces)`);
      const [nsub, ned, nfac] = b.ints;
      if (nsub < 1) throw new StyleError(`data file line ${b.at}: Incorrect integer value in Bodies section of data file`);
      const nentries = nsub < 3 ? 6 + 3 * nsub + 1 : 6 + 3 * nsub + 2 * ned + 4 * nfac + 1;
      if (b.dbls.length !== nentries) throw new StyleError(`data file line ${b.at}: Incorrect # of floating-point values in Bodies section of data file (needs ${nentries}, got ${b.dbls.length})`);
      const st = polyStoredCounts(nsub, ned, nfac);
      return { b, nsub, ...st, need: polyRecordLength(nsub, st.nedge, st.nface) };
    });
    let need = 0;
    for (const p of parsed) need = Math.max(need, p.need);
    const rec0 = s.custom.get(BODY_RECORD)!;
    if (need > rec0.cols) {
      const data = new Float64Array(s.n * need);
      for (let i = 0; i < s.n; i++) data.set(rec0.data.subarray(rec0.cols * i, rec0.cols * (i + 1)), need * i);
      rec0.data = data;
      rec0.cols = need;
    }
    for (const { b, nsub, nedge, nface } of parsed) {
      if (!bodyPending.has(b.id)) throw new StyleError(`data file line ${b.at}: atom ${b.id} has no bodyflag 1 in the Atoms section`);
      const i = (() => { for (let q = n0; q < s.n; q++) if (s.id[q] === b.id) return q; return -1; })();
      if (i < 0) throw new StyleError(`data file line ${b.at}: atom ${b.id} is not in the Atoms section`);
      bodyPending.delete(b.id);
      const d = b.dbls;
      const ned = b.ints[1], nfac = b.ints[2];
      const vfile = (q: number): number[] => [d[6 + 3 * q], d[7 + 3 * q], d[8 + 3 * q]];
      const dsq = (a: number[], c: number[]): number => (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2 + (a[2] - c[2]) ** 2;
      const diam = nsub < 3 ? d[6 + 3 * nsub] : d[6 + 3 * nsub + 2 * ned + 4 * nfac];
      if (nsub === 2 && dsq(vfile(0), vfile(1)) === 0) throw new StyleError(`The two vertices of rod body ${b.id} in Bodies section of data file coincide`);
      // the edges of a body with three or more vertices are data-file entries (a rod's edge is 0-1, not read)
      for (let e = 0; nsub >= 3 && e < nedge; e++) {
        const na = d[6 + 3 * nsub + 2 * e], nb = d[7 + 3 * nsub + 2 * e];
        if (!(na >= 0 && na < nsub && nb >= 0 && nb < nsub) || dsq(vfile(na), vfile(nb)) === 0) {
          throw new StyleError(`Edge ${e} of body ${b.id} in Bodies section of data file has invalid vertices or zero length`);
        }
      }
      for (let f = 0; f < nface; f++) {
        const o = 6 + 3 * nsub + 2 * ned + 4 * f;
        const nv = [d[o], d[o + 1], d[o + 2]];
        for (const v of nv) if (!(v >= 0 && v < nsub)) throw new StyleError(`Face ${f} of body ${b.id} in Bodies section of data file has invalid vertices`);
        const u = [vfile(nv[1])[0] - vfile(nv[0])[0], vfile(nv[1])[1] - vfile(nv[0])[1], vfile(nv[1])[2] - vfile(nv[0])[2]];
        const v2 = [vfile(nv[2])[0] - vfile(nv[0])[0], vfile(nv[2])[1] - vfile(nv[0])[1], vfile(nv[2])[2] - vfile(nv[0])[2]];
        const n = [u[1] * v2[2] - u[2] * v2[1], u[2] * v2[0] - u[0] * v2[2], u[0] * v2[1] - u[1] * v2[0]];
        const lsq = (a: number[]): number => a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        if (lsq(n) <= 1.0e-24 * lsq(u) * lsq(v2)) throw new StyleError(`The first three vertices of face ${f} of body ${b.id} in Bodies section of data file are on a line`);
      }
      const pf = polyhedronFrame(d.slice(0, 6));
      const rec = s.custom.get(BODY_RECORD)!;
      const B = rec.cols, base = B * i;
      rec.data[base] = nsub;
      for (let c = 0; c < 3; c++) rec.data[base + 1 + c] = pf.moments[c];
      rec.data[base + 4] = nedge;
      rec.data[base + 5] = nface;
      let erad2 = 0;
      for (let q = 0; q < nsub; q++) {
        const dl = vfile(q);
        erad2 = Math.max(erad2, dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]);
        for (let c = 0; c < 3; c++) rec.data[base + POLY_OFF + 3 * q + c] = pf.E[c] * dl[0] + pf.E[3 + c] * dl[1] + pf.E[6 + c] * dl[2];
      }
      const rrad = 0.5 * diam;
      rec.data[base + 7] = rrad;
      rec.data[base + 6] = nsub === 1 ? rrad : Math.sqrt(erad2);
      const eo = base + POLY_OFF + 3 * nsub;
      if (nsub === 2) {
        rec.data[eo] = 0;
        rec.data[eo + 1] = 1;
      } else {
        for (let e = 0; e < nedge; e++) {
          rec.data[eo + 2 * e] = d[6 + 3 * nsub + 2 * e];
          rec.data[eo + 2 * e + 1] = d[7 + 3 * nsub + 2 * e];
        }
      }
      const fo = eo + 2 * nedge;
      for (let f = 0; f < nface; f++) {
        for (let m = 0; m < 4; m++) rec.data[fo + 4 * f + m] = d[6 + 3 * nsub + 2 * ned + 4 * f + m];
      }
      for (let c = 0; c < 4; c++) s.quat![4 * i + c] = pf.quat[c];
    }
    if (bodyPending.size) throw new StyleError(`data file: ${bodyPending.size} atoms have bodyflag 1 but no Bodies entry`);
  };
  /**
   * A type column of a data line: a numeric type (plus the offset) or a type label of the kind. Offsets are
   * ignored on a label (read_data.html: "the type labels will determine the actual types directly").
   */
  const typeCol = (v: string, kind: LabelKind, off: number, at: number, what: string): number => {
    if (!/^[+-]?\d/.test(v)) {
      const t = sys.labels.typeOf(kind, v);
      if (t === undefined) throw new StyleError(`data file line ${at}: Invalid line in ${what} section of data file: ${kind} type label ${v} is not defined`);
      return t;
    }
    return intOf(v, `${kind}-type`, at) + off;
  };
  /** atom_style body: atoms with bodyflag 1 waiting for their Bodies entry. */
  const bodyPending = new Set<number>();
  const coeffLines: { section: string; line: string; at: number }[] = [];
  const topo: { kind: 'bonds' | 'angles' | 'dihedrals' | 'impropers'; type: number; ids: number[] }[] = [];
  const fixRows: { fx: DataFix; rows: { w: string[]; at: number }[]; title: string }[] = [];
  // sections
  while (k < lines.length) {
    const title = lines[k].split('#')[0].trim();
    const hint = (lines[k].split('#')[1] ?? '').trim();
    k++;
    if (!title) continue;
    // read_data.html: "The *next* line is *always* skipped." Measured with native LAMMPS (black box): a
    // non-blank line there (e.g. a column legend) is ignored, and with no blank line the first value line is lost.
    k++;
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
      const nlines = fx.sectionLines ? fx.sectionLines(natoms, title) : natoms;
      const rows: { w: string[]; at: number }[] = [];
      while (rows.length < nlines && k < lines.length) {
        const t = lines[k].replace(/#.*/, '').trim();
        k++;
        if (!t) continue;
        rows.push({ w: t.split(/\s+/), at: k });
      }
      if (rows.length < nlines) throw new StyleError(`data file section ${title}: expected ${nlines} lines, found ${rows.length}`);
      fixRows.push({ fx, rows, title });
      continue;
    }
    if (!SECTIONS.has(title)) throw new StyleError(`unknown data file section '${title}' (line ${k})`);
    const labelKind = labelKindOfSection(title);
    if (labelKind && typesRead) throw new StyleError('Must read Type Labels before any section involving types');
    if (labelKind && adding) throw new StyleError(`read_data: ${title} with add append/merge is not supported by the browser engine`);
    if (TYPE_SECTIONS.has(title)) typesRead = true;
    // read_data.html: "The *Bodies* section must appear if atom_style body is used"; its entries have no fixed
    // line count (Ninteger and Ndouble values per body, read line by line), so it is read here.
    if (title === 'Bodies') {
      if (!parseBodyStyle(style)) throw new StyleError('data file section Bodies needs atom_style body');
      const r = readBodyLines(lines, k, h.bodies ?? 0);
      k = r.k;
      if (parseBodyStyle(style)!.bstyle === 'rounded/polyhedron') {
        readPolyhedronBodies(r.bodies);
        continue;
      }
      // Measured with native LAMMPS (black box): read_data accepts bodies with fewer sub-particles than Nmin or more
      // than Nmax (the two values only size internal arrays), so the record widens to the largest body read here.
      let maxN = 0;
      for (const b of r.bodies) if (b.ints.length === 1) maxN = Math.max(maxN, b.ints[0]);
      const bstyle = parseBodyStyle(style)!.bstyle;
      const rec0 = s.custom.get(BODY_RECORD)!;
      if (bodyRecordStride(maxN, bstyle) > rec0.cols) {
        const cols = bodyRecordStride(maxN, bstyle), data = new Float64Array(s.n * cols);
        for (let i = 0; i < s.n; i++) data.set(rec0.data.subarray(rec0.cols * i, rec0.cols * (i + 1)), cols * i);
        rec0.data = data;
        rec0.cols = cols;
      }
      for (const b of r.bodies) {
        if (b.ints.length !== 1) throw new StyleError(`data file line ${b.at}: body style ${bstyle} has one integer (the number of ${bstyle === 'rounded/polygon' ? 'vertices' : 'sub-particles'})`);
        const nsub = b.ints[0];
        // Howto_body.rst (rounded/polygon): "where M = 6 + 3\*N + 1, and N is the number of vertices in the body"; the last value is the diameter
        const extra = bstyle === 'rounded/polygon' ? 1 : 0;
        if (bstyle === 'rounded/polygon' && nsub < 1) throw new StyleError(`data file line ${b.at}: body rounded/polygon needs at least one vertex`);
        if (b.dbls.length !== 6 + 3 * nsub + extra) throw new StyleError(`data file line ${b.at}: body ${bstyle} needs 6 + 3N${extra ? ' + 1' : ''} = ${6 + 3 * nsub + extra} doubles (N = ${nsub}), got ${b.dbls.length}`);
        if (!bodyPending.has(b.id)) throw new StyleError(`data file line ${b.at}: atom ${b.id} has no bodyflag 1 in the Atoms section`);
        const i = (() => { for (let q = n0; q < s.n; q++) if (s.id[q] === b.id) return q; return -1; })();
        if (i < 0) throw new StyleError(`data file line ${b.at}: atom ${b.id} is not in the Atoms section`);
        bodyPending.delete(b.id);
        // read_data.html: the sub-particle coordinates are displacements from the center of mass in the box
        // frame; the engine stores them in the body frame of the principal axes (bodies.ts)
        const pf = principalFrame(b.dbls.slice(0, 6));
        const rec = s.custom.get(BODY_RECORD)!;
        const B = rec.cols, base = B * i;
        rec.data[base] = nsub;
        for (let d = 0; d < 3; d++) rec.data[base + 1 + d] = pf.moments[d];
        if (bstyle === 'rounded/polygon') {
          // Howto_body.rst: "with z = 0.0 for each" vertex (the diameter is the value after the vertices)
          for (let q = 0; q < nsub; q++) {
            if (b.dbls[8 + 3 * q] !== 0) throw new StyleError(`data file line ${b.at}: body rounded/polygon vertices must have z = 0 (2d bodies)`);
          }
          rec.data[base + 4] = b.dbls[6 + 3 * nsub];
        }
        const o0 = bodyRecordOffset(bstyle);
        for (let q = 0; q < nsub; q++) {
          const ds = [b.dbls[6 + 3 * q], b.dbls[7 + 3 * q], b.dbls[8 + 3 * q]];
          for (let d = 0; d < 3; d++) rec.data[base + o0 + 3 * q + d] = pf.E[d] * ds[0] + pf.E[3 + d] * ds[1] + pf.E[6 + d] * ds[2];
        }
        for (let d = 0; d < 4; d++) s.quat![4 * i + d] = pf.quat[d];
      }
      if (bodyPending.size) throw new StyleError(`data file: ${bodyPending.size} atoms have bodyflag 1 but no Bodies entry`);
      continue;
    }
    // count of lines expected
    const count = (() => {
      switch (title) {
        case 'Atoms': case 'Velocities': return natoms;
        case 'Masses': case 'Pair Coeffs': case 'Atom Type Labels': return ntypesFile;
        case 'Bond Type Labels': return h['bond types'] ?? 0;
        case 'Angle Type Labels': return h['angle types'] ?? 0;
        case 'Dihedral Type Labels': return h['dihedral types'] ?? 0;
        case 'Improper Type Labels': return h['improper types'] ?? 0;
        case 'PairIJ Coeffs': return (ntypesFile * (ntypesFile + 1)) / 2;
        case 'Bonds': return h.bonds ?? 0;
        case 'Angles': return h.angles ?? 0;
        case 'Dihedrals': return h.dihedrals ?? 0;
        case 'Impropers': return h.impropers ?? 0;
        case 'Bond Coeffs': return h['bond types'] ?? 0;
        case 'Angle Coeffs': return h['angle types'] ?? 0;
        case 'Dihedral Coeffs': return h['dihedral types'] ?? 0;
        case 'Improper Coeffs': return h['improper types'] ?? 0;
        case 'BondBond Coeffs': case 'BondAngle Coeffs': case 'UreyBradley Coeffs': return h['angle types'] ?? 0;
        case 'MiddleBondTorsion Coeffs': case 'EndBondTorsion Coeffs': case 'AngleTorsion Coeffs':
        case 'AngleAngleTorsion Coeffs': case 'BondBond13 Coeffs': return h['dihedral types'] ?? 0;
        case 'AngleAngle Coeffs': return h['improper types'] ?? 0;
        case 'Ellipsoids':
          if (!s.shape) throw new StyleError('data file section Ellipsoids needs atom_style ellipsoid');
          return h.ellipsoids ?? 0;
        // read_data.html: "The *Lines* section must appear if atom_style line is used and any atoms are listed in the *Atoms* section with a lineflag = 1"
        case 'Lines':
          if (!isLineStyle(style)) throw new StyleError('data file section Lines needs atom_style line');
          return h.lines ?? 0;
        case 'Triangles':
          if (!isTriStyle(style)) throw new StyleError('data file section Triangles needs atom_style tri');
          return h.triangles ?? 0;
        default: throw new StyleError(`data file section '${title}' needs features the browser engine does not support (type labels, class2 cross terms or finite-size particles)`);
      }
    })();
    if (title === 'Atoms' && hint && hint !== style) sys.warn(`atom style in data file (${hint}) differs from the current atom style (${style})`);
    const body: { w: string[]; at: number }[] = [];
    // read `count` non-blank lines (the line after the title was skipped above)
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
        const radius = new Float64Array(count), density = new Float64Array(count), vol = new Float64Array(count);
        const mu = s.mu ? new Float64Array(4 * count) : null;
        const spin = s.sp ? new Float64Array(4 * count) : null;
        const eflag = s.shape ? new Uint8Array(count) : null;
        // atom_style line: lineflag; atom_style tri: triangleflag (read_data.html: 1 for line segment or triangular particles, 0 for point particles)
        const lflag = isLineStyle(style) || isTriStyle(style) ? new Uint8Array(count) : null;
        const tIdx = new Int32Array(count), tAt = new Int32Array(count);
        const espin = new Float64Array(count), eradius = new Float64Array(count);
        const rheoStatus = isRheoStyle(style) ? new Float64Array(count) : null, rheoRho = isRheoStyle(style) ? new Float64Array(count) : null;
        const rheoEsph = isRheoThermalStyle(style) ? new Float64Array(count) : null;
        const diel = isDielectricStyle(style) ? { area: new Float64Array(count), ed: new Float64Array(count), em: new Float64Array(count), epsilon: new Float64Array(count), curvature: new Float64Array(count) } : null;
        const sph = isSphStyle(style) ? { rho: new Float64Array(count), esph: new Float64Array(count), cv: new Float64Array(count) } : null;
        const bflag = parseBodyStyle(style) ? new Uint8Array(count) : null;
        // atom_style edpd / mdpd / tdpd (read_data.html: "edpd_temp", "edpd_cv", "rho", "ccN"); the species count of tdpd
        const nsp = tdpdSpecies(style);
        const dm = isEdpdStyle(style) || isMdpdStyle(style) || nsp > 0
          ? { temp: new Float64Array(count), cv: new Float64Array(count), rho: new Float64Array(count), cc: new Float64Array(count * nsp) }
          : null;
        const dpdTh = isDpdStyle(style) ? new Float64Array(count) : null;
        // atom_style smd (read_data.html): the contact radius and the mass are columns; the kernel radius is the radius
        const smdCr = isSmdStyle(style) ? new Float64Array(count) : null;
        const smdRm = isSmdStyle(style) ? new Float64Array(count) : null;
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
                const t = typeCol(v, 'atom', toff, at, 'Atoms');
                if (t < 1 || t > s.ntypes) throw new StyleError(`data file line ${at}: atom type ${t} is outside 1..${s.ntypes}`);
                type[a] = t;
                break;
              }
              case 'q': q[a] = numOf(v, 'charge', at); break;
              case 'diameter': radius[a] = numOf(v, 'diameter', at) / 2; break;
              case 'density': density[a] = numOf(v, 'density', at); break;
              case 'volume': vol[a] = numOf(v, 'volume', at); break;
              case 'x': x[3 * a] = numOf(v, 'x', at) + opts.shift[0]; break;
              case 'y': x[3 * a + 1] = numOf(v, 'y', at) + opts.shift[1]; break;
              case 'z': x[3 * a + 2] = numOf(v, 'z', at) + opts.shift[2]; break;
              case 'ellipsoidflag': {
                const f = intOf(v, 'ellipsoidflag', at);
                if (f !== 0 && f !== 1) throw new StyleError(`data file line ${at}: ellipsoidflag must be 0 or 1`);
                eflag![a] = f;
                break;
              }
              case 'lineflag': case 'triangleflag': {
                const f = intOf(v, cols[c], at);
                if (f !== 0 && f !== 1) throw new StyleError(`data file line ${at}: ${cols[c]} must be 0 or 1`);
                lflag![a] = f;
                break;
              }
              case 'bodyflag': {
                const f = intOf(v, 'bodyflag', at);
                if (f !== 0 && f !== 1) throw new StyleError(`data file line ${at}: bodyflag must be 0 or 1`);
                bflag![a] = f;
                break;
              }
              case 'mass': density[a] = numOf(v, 'mass', at); break;
              case 'tindex': tIdx[a] = intOf(v, 'template-index', at); break;
              case 'tatom': tAt[a] = intOf(v, 'template-atom', at); break;
              case 'mux': mu![4 * a] = numOf(v, 'mux', at); break;
              case 'muy': mu![4 * a + 1] = numOf(v, 'muy', at); break;
              case 'muz': mu![4 * a + 2] = numOf(v, 'muz', at); break;
              case 'spx': spin![4 * a] = numOf(v, 'spx', at); break;
              case 'spy': spin![4 * a + 1] = numOf(v, 'spy', at); break;
              case 'spz': spin![4 * a + 2] = numOf(v, 'spz', at); break;
              case 'sp': spin![4 * a + 3] = numOf(v, 'sp', at); break;
              // read_data.html: "espin = electron spin (+1/-1), 0 = nuclei, 2 = fixed-core, 3 = pseudo-cores (i.e. ECP)"
              case 'espin': espin[a] = intOf(v, 'espin', at); break;
              // atom_style dpd: "dpdTheta" is the internal temperature of the atom (read_data: "must be > 0.0")
              case 'dpdTheta': dpdTh![a] = numOf(v, 'dpdTheta', at); break;
              // read_data.html: "eradius = electron radius (or fixed-core radius)"
              case 'eradius': eradius[a] = numOf(v, 'eradius', at); break;
              // read_data.html, row rheo: "status = status or phase of RHEO particles (unitless)", "rho = density"
              case 'rheo_status': rheoStatus![a] = intOf(v, 'status', at); break;
              // read_data.html, rheo/thermal: "energy = energy of RHEO particles (need units)"
              case 'energy': rheoEsph![a] = numOf(v, 'energy', at); break;
              case 'area': case 'ed': case 'em': case 'epsilon': case 'curvature': {
                const dk = cols[c] as 'area' | 'ed' | 'em' | 'epsilon' | 'curvature';
                diel![dk][a] = numOf(v, dk, at);
                break;
              }
              // read_data.html, sph: "rho = density (need units)", "esph = energy (need units) for SPH particles", "cv = heat capacity (need units) for SPH particles"
              case 'esph': case 'cv': {
                const sk = cols[c] as 'esph' | 'cv';
                sph![sk][a] = numOf(v, sk, at);
                break;
              }
              // rho is a column of atom_style sph, mdpd and rheo (read_data.html)
              case 'rho':
                if (rheoRho) rheoRho[a] = numOf(v, 'rho', at);
                else if (sph) sph.rho[a] = numOf(v, 'rho', at);
                else dm!.rho[a] = numOf(v, 'rho', at);
                break;
              // atom_style smd (read_data.html): "volume", "mass", "kradius", "cradius"; the x0 values are not kept (x0 = x)
              case 'vfrac': vol[a] = numOf(v, 'volume', at); break;
              case 'rmass': smdRm![a] = numOf(v, 'mass', at); break;
              case 'kradius': radius[a] = numOf(v, 'kradius', at); break;
              case 'cradius': smdCr![a] = numOf(v, 'cradius', at); break;
              case 'x0': case 'y0': case 'z0': numOf(v, cols[c], at); break;
              case 'edpd_temp': dm!.temp[a] = numOf(v, 'edpd_temp', at); break;
              case 'edpd_cv': dm!.cv[a] = numOf(v, 'edpd_cv', at); break;
              default: {
                const k = /^cc(\d+)$/.exec(cols[c]);
                if (k && dm) dm.cc[nsp * a + Number(k[1]) - 1] = numOf(v, cols[c], at);
                break;
              }
            }
          }
          if (mu) mu[4 * a + 3] = Math.hypot(mu[4 * a], mu[4 * a + 1], mu[4 * a + 2]);
          if (spin) setSpinVector(spin, a, spin[4 * a], spin[4 * a + 1], spin[4 * a + 2], spin[4 * a + 3], `data file line ${at}`);
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
        // atom_style peri. Measured with native LAMMPS (black box): read_data gives mass = density (volume is kept apart)
        const rmass = smdRm ?? (s.radius ? radius.map((r, a) => sphereMass(r, density[a])) : s.shape || s.vfrac || bflag ? Float64Array.from(density) : undefined);
        // atom_style body: the mass column is the body's mass (read_data.html); a bodyflag 1 atom needs a Bodies entry
        if (bflag) for (let a = 0; a < count; a++) if (bflag[a]) bodyPending.add(id[a]);
        if (eflag) for (let a = 0; a < count; a++) if (eflag[a]) ellPending.set(id[a], density[a]);
        // atom_style line / tri: the density is per length or per area; the mass is set when the Lines or Triangles entry gives the size
        if (lflag && isLineStyle(style)) for (let a = 0; a < count; a++) if (lflag[a]) linePending.set(id[a], density[a]);
        if (lflag && isTriStyle(style)) for (let a = 0; a < count; a++) if (lflag[a]) triPending.set(id[a], density[a]);
        if (genFrame) {
          // read_data.html: coordinates "should be inside the general triclinic simulation box"; the
          // general -> restricted rotation is about the box origin (Howto_triclinic.html)
          const o = gen.origin ?? [0, 0, 0];
          for (let a = 0; a < count; a++) {
            const r = toRestrictedPoint(genFrame.Q, o, [x[3 * a], x[3 * a + 1], x[3 * a + 2]]);
            x[3 * a] = r[0]; x[3 * a + 1] = r[1]; x[3 * a + 2] = r[2];
          }
        }
        const firstNew = s.n;
        const electronCustom = isElectronStyle(s.atomStyle) ? new Map([[ELECTRON_SPIN, espin], [ELECTRON_RADIUS, eradius]]) : undefined;
        // Measured with native LAMMPS (black box): a data line with q 2.0 and epsilon 4.0 reads as q_scaled 0.5 (q / epsilon)
        const rheoCustom = rheoStatus && rheoRho ? new Map<string, Float64Array>([[RHEO_STATUS, rheoStatus], [RHEO_RHO, rheoRho], ...(rheoEsph ? [[RHEO_ESPH, rheoEsph] as [string, Float64Array]] : [])]) : undefined;
        if (dpdTh) for (let a = 0; a < count; a++) {
          if (!(dpdTh[a] > 0)) throw new StyleError(`data file line ${body[a].at}: Internal temperature theta in Atoms section of data file must be > 0.0`);
        }
        const dpdCustom = dpdTh ? new Map<string, Float64Array>([[DPD_THETA, dpdTh]]) : undefined;
        const dielCustom = diel ? new Map<string, Float64Array>([
          [DIEL_AREA, diel.area], [DIEL_ED, diel.ed], [DIEL_EM, diel.em], [DIEL_EPSILON, diel.epsilon], [DIEL_CURVATURE, diel.curvature],
          [DIEL_QSCALED, Float64Array.from(q, (qq, a) => qq / diel.epsilon[a])],
        ]) : undefined;
        // atom_style sph: the velocity estimate and the rates start at zero
        const sphCustom = sph ? new Map<string, Float64Array>([
          [SPH_RHO, sph.rho], [SPH_ESPH, sph.esph], [SPH_CV, sph.cv], [SPH_DRHO, new Float64Array(count)],
          [SPH_DESPH, new Float64Array(count)], [SPH_VEST, new Float64Array(3 * count)],
        ]) : undefined;
        // atom_style edpd / mdpd / tdpd: the DPD-MESO props of the read atoms (appendAtoms keeps only the styles' own props)
        const dmCustom = dm ? new Map<string, Float64Array>([[EDPD_TEMP, dm.temp], [EDPD_CV, dm.cv], [MDPD_RHO, dm.rho], [TDPD_CC, dm.cc]]) : undefined;
        const smdCustom = smdCr ? new Map<string, Float64Array>([[SMD_CONTACT, smdCr]]) : undefined;
        const extraCustom = new Map<string, Float64Array>([...(electronCustom ?? []), ...(dielCustom ?? []), ...(sphCustom ?? []), ...(dmCustom ?? []), ...(rheoCustom ?? []), ...(dpdCustom ?? []), ...(smdCustom ?? [])]);
        appendAtoms(s, { x, type, id, image, molecule: mol, q, mask: gbit, radius: s.radius ? radius : undefined, rmass, mu: mu ?? undefined, sp: spin ?? undefined, vfrac: s.vfrac ? vol : undefined, tmplIndex: tmplId !== null ? tIdx : undefined, tmplAtom: tmplId !== null ? tAt : undefined, custom: extraCustom.size ? extraCustom : undefined });
        // periodic remap of the new atoms
        for (let i = n0; i < s.n; i++) sys.geom.remap(s.x, s.image, i);
        // read_data: an atom is assigned to the box only if lo <= x < hi in every dimension (native Domain::ownatom);
        // with fixed (f) faces on both sides, an atom on the upper face is not assigned. "Measured with native LAMMPS
        // (black box, boundary p p f, zhi 3.0 with atoms at z 3.0): ERROR: Did not assign all atoms correctly (read_data.cpp)".
        // Shrink-wrapped faces (s, m) keep the header box as before.
        if (!s.box.triclinic) {
          for (let d = 0; d < 3; d++) {
            if (s.box.periodic[d] || s.box.boundary[d][0] !== 'f' || s.box.boundary[d][1] !== 'f') continue;
            for (let i = n0; i < s.n; i++) {
              const xd = s.x[3 * i + d];
              if (xd < s.box.lo[d] || xd >= s.box.hi[d]) throw new StyleError('read_data: Did not assign all atoms correctly (an atom lies outside the box in a fixed dimension; native ERROR err0016)');
            }
          }
        }
        if (tmplId !== null) expandTemplateTopology(s, sys.molecules.get(tmplId) ?? [], tmplId, firstNew, s.n);
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
      // read_data.html, Lines section: "atom-ID x1 y1 x2 y2" (the end points of a line segment, centred on the atom's x)
      case 'Lines': {
        const where = new Map<number, number>();
        for (let i = n0; i < s.n; i++) where.set(s.id[i], i);
        for (const { w, at } of body) {
          if (w.length !== 5) throw new StyleError(`data file line ${at}: Lines needs atom-ID x1 y1 x2 y2`);
          const aid = intOf(w[0], 'atom-ID', at) + (idBase || 0);
          const i = where.get(aid);
          const dens = linePending.get(aid);
          if (i === undefined || dens === undefined) throw new StyleError(`data file line ${at}: atom ${aid} is not a line segment (lineflag 1) of this data file`);
          const e = [1, 2, 3, 4].map((c) => numOf(w[c], 'end point', at));
          const ends = remapEndsNear(s, i, [[e[0], e[1]], [e[2], e[3]]]);
          const lb = lineFromEnds(ends[0][0], ends[0][1], ends[1][0], ends[1][1]);
          if (!(lb.length > 0)) throw new StyleError(`data file line ${at}: line segment has zero length`);
          checkCentre(s, i, [0.5 * (ends[0][0] + ends[1][0]), 0.5 * (ends[0][1] + ends[1][1]), 0], lb.length, 'Inconsistent line segment in data file', at);
          setLineBonus(s, i, lb, dens);
          linePending.delete(aid);
        }
        break;
      }
      // read_data.html, Triangles section: "atom-ID x1 y1 z1 x2 y2 z2 x3 y3 z3" (the corner points, centred on the atom's x)
      case 'Triangles': {
        const where = new Map<number, number>();
        for (let i = n0; i < s.n; i++) where.set(s.id[i], i);
        for (const { w, at } of body) {
          if (w.length !== 10) throw new StyleError(`data file line ${at}: Triangles needs atom-ID x1 y1 z1 x2 y2 z2 x3 y3 z3`);
          const aid = intOf(w[0], 'atom-ID', at) + (idBase || 0);
          const i = where.get(aid);
          const dens = triPending.get(aid);
          if (i === undefined || dens === undefined) throw new StyleError(`data file line ${at}: atom ${aid} is not a triangle (triangleflag 1) of this data file`);
          const c0 = [1, 2, 3].map((c) => numOf(w[c], 'corner', at));
          const c1 = [4, 5, 6].map((c) => numOf(w[c], 'corner', at));
          const c2 = [7, 8, 9].map((c) => numOf(w[c], 'corner', at));
          const r: [number, number, number] = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
          const cr = [c0, c1, c2].map((p) => remapNear3(s, p as [number, number, number], r)) as [V3, V3, V3];
          const size = Math.max(
            Math.hypot(cr[1][0] - cr[0][0], cr[1][1] - cr[0][1], cr[1][2] - cr[0][2]),
            Math.hypot(cr[2][0] - cr[0][0], cr[2][1] - cr[0][1], cr[2][2] - cr[0][2]),
          );
          const tb = triBodyHook().triBody(cr, dens);
          checkCentre(s, i, tb.centroid, size, 'Inconsistent triangle in data file', at);
          setTriBonus(s, i, tb);
          triPending.delete(aid);
        }
        break;
      }
      case 'Velocities': {
        // read_data.html: "sphere | atom-ID vx vy vz wx wy wz"; "ellipsoid | atom-ID vx vy vz lx ly lz"; the tri style
        // has both (atom-ID vx vy vz wx wy wz lx ly lz); "electron | atom-ID vx vy vz ervel" ("ervel = electron radial
        // velocity (0 for fixed-core)")
        const tri = isTriStyle(style);
        const electron = isElectronStyle(s.atomStyle);
        if (s.omega && s.angmom && !tri) throw new StyleError('data file Velocities: hybrid styles with both sphere and ellipsoid sub-styles are not supported by the browser engine');
        if (electron && (s.omega || s.angmom)) throw new StyleError('data file Velocities: atom_style electron with a sphere or ellipsoid sub-style is not supported by the browser engine');
        const labels = electron ? ['vx', 'vy', 'vz', 'ervel'] : ['vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'lx', 'ly', 'lz'];
        for (const { w, at } of body) {
          const need = tri ? 10 : s.omega || s.angmom ? 7 : electron ? 5 : 4;
          if (w.length < need) throw new StyleError(`data file line ${at}: Velocities needs atom-ID vx vy vz${tri ? ' wx wy wz lx ly lz' : s.omega ? ' wx wy wz' : s.angmom ? ' lx ly lz' : electron ? ' ervel' : ''}`);
          vel.set(intOf(w[0], 'atom-ID', at) + (idBase || 0), w.slice(1, need).map((t, c) => numOf(t, labels[c], at)));
        }
        break;
      }
      case 'Masses':
        // measured with native LAMMPS (atom_style sphere): "Cannot set mass for atom style sphere"
        // read_data.html: "This section cannot be used for atom styles that define a mass for individual atoms - e.g. atom_style sphere"
        if (s.atomStyle === 'sphere' || s.atomStyle === 'bpm/sphere' || isLineStyle(s.atomStyle) || isTriStyle(s.atomStyle)) throw new StyleError(`Cannot set mass for atom style ${s.atomStyle}`);
        for (const { w, at } of body) {
          const t = typeCol(w[0], 'atom', toff, at, 'Masses');
          if (t < 1 || t > s.ntypes) throw new StyleError(`data file line ${at}: atom type ${t} is outside 1..${s.ntypes}`);
          const m = numOf(w[1], 'mass', at);
          if (!(m > 0)) throw new StyleError(`data file line ${at}: mass must be > 0`);
          s.massByType[t] = m;
        }
        break;
      case 'Bonds': case 'Angles': case 'Dihedrals': case 'Impropers': {
        if (tmplId !== null) throw new StyleError(`data file section ${title}: the topology of atom_style template comes from the molecule templates, not from a ${title} section`);
        const kind = title.toLowerCase() as 'bonds' | 'angles' | 'dihedrals' | 'impropers';
        const width = kind === 'bonds' ? 2 : kind === 'angles' ? 3 : 4;
        const off = kind === 'bonds' ? boff : kind === 'angles' ? aoff : kind === 'dihedrals' ? doff : ioff;
        const ntype = kind === 'bonds' ? s.topo.nbondtypes : kind === 'angles' ? s.topo.nangletypes : kind === 'dihedrals' ? s.topo.ndihedraltypes : s.topo.nimpropertypes;
        const lkind: LabelKind = kind === 'bonds' ? 'bond' : kind === 'angles' ? 'angle' : kind === 'dihedrals' ? 'dihedral' : 'improper';
        for (const { w, at } of body) {
          if (w.length < 2 + width) throw new StyleError(`data file line ${at}: ${title} needs ID type and ${width} atom IDs`);
          const t = typeCol(w[1], lkind, off, at, title);
          if (t < 1 || t > ntype) throw new StyleError(`data file line ${at}: ${title.slice(0, -1).toLowerCase()} type ${t} is outside 1..${ntype}`);
          const ids = w.slice(2, 2 + width).map((v) => intOf(v, 'atom-ID', at) + (idBase || 0));
          topo.push({ kind, type: t, ids });
        }
        break;
      }
      // read_data.html: the type-label sections give one label per numeric type (ID label); the sections
      // that follow must name their types numerically
      case 'Atom Type Labels': case 'Bond Type Labels': case 'Angle Type Labels': case 'Dihedral Type Labels': case 'Improper Type Labels': {
        const lk = labelKind!;
        for (const { w, at } of body) {
          if (w.length !== 2) throw new StyleError(`data file line ${at}: Invalid format in section: ${title}`);
          const t = intOf(w[0], 'type', at);
          if (t < 1 || t > count) throw new StyleError(`data file line ${at}: ${title.slice(0, -' Type Labels'.length)} type ${t} is outside 1..${count}`);
          if (!isValidLabel(w[1])) throw new StyleError(`data file line ${at}: Invalid type label ${w[1]}`);
          const other = sys.labels.typeOf(lk, w[1]);
          if (other !== undefined && other !== t) throw new StyleError(`${title} map is incomplete`);
          sys.labels.set(lk, t, w[1]);
        }
        if (!sys.labels.complete(lk, count)) throw new StyleError(`${title} map is incomplete`);
        break;
      }
      default:
        for (const { w, at } of body) {
          // read_data.html: "ID coeffs" with a numeric ID (measured with native read_data: a label there is refused)
          const nid = title === 'PairIJ Coeffs' ? 2 : 1;
          for (let c = 0; c < nid; c++) {
            if (!/^[+-]?\d+$/.test(w[c] ?? '')) throw new StyleError(`Expected integer parameter instead of '${w[c] ?? ''}' in input script or data file`);
          }
          coeffLines.push({ section: title, line: w.join(' '), at });
        }
    }
  }
  if (natoms > 0 && !sawAtoms) throw new StyleError('data file: an Atoms section is required when atoms > 0');
  if (ellPending.size) throw new StyleError(`data file: ${ellPending.size} atoms have ellipsoidflag 1 but no Ellipsoids line`);
  if (linePending.size) throw new StyleError(`data file: ${linePending.size} atoms have lineflag 1 but no Lines line`);
  if (triPending.size) throw new StyleError(`data file: ${triPending.size} atoms have triangleflag 1 but no Triangles line`);
  if (bodyPending.size) throw new StyleError(`data file: ${bodyPending.size} atoms have bodyflag 1 but no Bodies entry (read_data.html: the Bodies section must appear)`);
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
          // tri: the angular momentum follows the angular velocity (fields_data_vel id v omega angmom)
          const l0 = isTriStyle(style) ? 6 : 3;
          for (let d = 0; d < 3; d++) s.angmom[3 * i + d] = v[l0 + d];
        }
        const rvel = isElectronStyle(s.atomStyle) ? s.custom.get(ELECTRON_RVEL) : undefined;
        if (rvel) rvel.data[i] = v[3];
      }
    }
  }
  // fix sections: "the lines of per-atom properties can be listed in any order" (fix_property_atom.html)
  if (fixRows.length) {
    const byId = new Map<number, number>();
    for (let i = n0; i < s.n; i++) byId.set(s.id[i], i);
    for (const { fx, rows, title } of fixRows) {
      // whole-section fixes (fix cmap) take their rows as they are; the IDs are checked when the fix acts
      if (fx.readSection) {
        if (adding) throw new StyleError(`read_data fix ${fx.id} (${fx.style}): add append/merge is not supported with this fix's section`);
        fx.readSection(rows, title);
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
    // a bond is stored with its first atom, so the bonds per atom of this file count the bonds that start at each atom.
    // Measured with native LAMMPS (black box, 6 atoms, 2 bonds starting at each of atoms 1, 2 and 3): a read after
    // create_box bond/types 2 with extra/bond/per/atom 0 or 1 stops with "Subsequent read data induced too many bonds
    // per atom", and with 2 it reads; the same read as the first (box-making) read is accepted and sets the storage.
    const perAtom = new Map<number, number>();
    let most = 0;
    for (const e of topo) {
      if (e.kind !== 'bonds') continue;
      const n = (perAtom.get(e.ids[0]) ?? 0) + 1;
      perAtom.set(e.ids[0], n);
      if (n > most) most = n;
    }
    if (subsequentRead) {
      if (most > sys.extraBondPerAtom) throw new StyleError('Subsequent read data induced too many bonds per atom');
    } else sys.extraBondPerAtom = Math.max(opts.extraBondPerAtom ?? 0, most);

    // atom_style oxdna: each bond sets the 5'/3' partners (atoms.ts setOxdnaPartners)
    setOxdnaPartners(s, topo.filter((e) => e.kind === 'bonds').map((e) => [e.ids[0], e.ids[1]] as const));
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
      } else if (CLASS2_SECTIONS[c.section]) {
        const [kind, kw] = CLASS2_SECTIONS[c.section];
        const st = sys.ff[kind];
        if (!st) throw new StyleError(`data file has ${c.section} but no ${kind}_style is defined (define it before read_data, or use nocoeff)`);
        const off = kind === 'angle' ? aoff : kind === 'dihedral' ? doff : ioff;
        st.coeff([String(Number(w[0]) + off), kw, ...w.slice(1)], ctx);
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

export interface WriteDataOptions { nocoeff: boolean; pairStyle: 'ii' | 'ij' | null; nofix?: boolean; triclinicGeneral?: boolean; nolabelmap?: boolean; typesLabels?: boolean }

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
  if (parseBodyStyle(s.atomStyle)) throw new StyleError(`write_data is not supported for atom_style ${s.atomStyle} by the browser engine (body records are not written)`);
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
  // measured with native write_data: the header gives N lines (atom_style line) and N triangles (atom_style tri), where
  // N counts the atoms with lineflag or triangleflag 1 (three flagged atoms give 3 lines, and none gives 0 lines)
  if (isLineStyle(s.atomStyle)) out.push(`${flagCount(s, LINE_FLAG)} lines`);
  if (isTriStyle(s.atomStyle)) out.push(`${flagCount(s, TRI_FLAG)} triangles`);
  const mol = isMolecularStyle(s.atomStyle);
  if (mol && isTemplateStyle(s.atomStyle)) {
    // measured with native write_data: a template style writes no bond, angle, dihedral or improper
    // counts (its topology is in the molecule templates), only the type counts that are non-zero
    for (const [nt, name] of [[t.nbondtypes, 'bond'], [t.nangletypes, 'angle'], [t.ndihedraltypes, 'dihedral'], [t.nimpropertypes, 'improper']] as const) {
      if (nt > 0) out.push(`${nt} ${name} types`);
    }
  } else if (mol) {
    const lines: [number, number, string][] = [
      [t.bonds.n, t.nbondtypes, 'bond'], [t.angles.n, t.nangletypes, 'angle'],
      [t.dihedrals.n, t.ndihedraltypes, 'dihedral'], [t.impropers.n, t.nimpropertypes, 'improper'],
    ];
    // Measured with native LAMMPS (black box): a kind with no types is left out (a molecular box with
    // bond and angle types only writes no dihedral or improper lines), while a kind with types but no
    // topology still writes both lines (0 angles, 2 angle types).
    for (const [n, nt, name] of lines) {
      const lv = topologyLevel(s.atomStyle);
      if (name === 'angle' && lv < 2) continue;
      if ((name === 'dihedral' || name === 'improper') && lv < 3) continue;
      if (nt === 0) continue;
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
  // write_data.html: "nolabelmap = do not write out type labels". Measured with native write_data: each kind's
  // section follows the box and comes before Masses, in the order atom, bond, angle, dihedral, improper; a kind
  // is written when its label map covers every numeric type and its style is defined (atoms always have one)
  if (!opts.nolabelmap) {
    const nOf: Record<LabelKind, number> = { atom: s.ntypes, bond: t.nbondtypes, angle: t.nangletypes, dihedral: t.ndihedraltypes, improper: t.nimpropertypes };
    for (const k of LABEL_KINDS) {
      if (k !== 'atom' && !sys.ff[k]) continue;
      if (nOf[k] === 0 || !sys.labels.complete(k, nOf[k])) continue;
      out.push('', LABEL_SECTION[k], '', ...sys.labels.entries(k).map(([ty, lb]) => `${ty} ${lb}`));
    }
  }
  // measured with native write_data: atom_style sphere (per-atom masses) writes no Masses section; a
  // hybrid style with sphere writes it (per-type and per-atom masses both exist there)
  // measured with native write_data: atom_style peri writes no Masses section either
  if (s.atomStyle !== 'sphere' && s.atomStyle !== 'bpm/sphere' && s.atomStyle !== 'ellipsoid' && s.atomStyle !== 'peri' && !isLineStyle(s.atomStyle) && !isTriStyle(s.atomStyle)) {
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
      if (st && lines) {
        out.push('', `${title} Coeffs # ${st.name}`, '', ...lines);
        // Measured with native LAMMPS (black box): write_data puts the class 2 cross-term sections right
        // after the style's Coeffs section (BondBond, BondAngle after Angle Coeffs; AngleAngleTorsion,
        // EndBondTorsion, MiddleBondTorsion, BondBond13, AngleTorsion after Dihedral Coeffs; AngleAngle
        // after Improper Coeffs), titles without a style comment.
        for (const sec of st.dataCrossSections()) out.push('', sec.title, '', ...sec.lines);
      }
    }
  }
  // measured with native write_data: a hybrid style is labelled Atoms # hybrid
  out.push('', `Atoms # ${s.atomStyle.startsWith('hybrid ') ? 'hybrid' : atomSubStyles(s.atomStyle)[0]}`, '');
  // Measured with native LAMMPS (black box, a 10-atom chain whose Atoms
  // section was shuffled): Atoms and Velocities come out in storage order
  // (the order read_data/create_atoms added them), not sorted by ID; each
  // topology section is grouped by its owning atom in that same order (bonds
  // by the first atom, angles, dihedrals and impropers by the second), keeps
  // the read order within one owner, and is renumbered from 1. Storage order
  // is native's (SimState.order, including its spatial sort at run setup).
  const order = nativeOrder(s);
  const cols = atomStyleCols(s.atomStyle);
  // write_data.html: "types value = numeric or labels". Measured with native write_data: a type without a label is
  // written as an empty field, so the line has two blanks in a row
  const typeText = (k: LabelKind, ty: number): string => (opts.typesLabels ? sys.labels.labelOf(k, ty) ?? '' : String(ty));
  const nsp = tdpdSpecies(s.atomStyle);
  if ((isLineStyle(s.atomStyle) || isTriStyle(s.atomStyle)) && gen) throw new StyleError('write_data: line and tri particles in a general triclinic box are not supported by the browser engine');
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
        case 'tindex': return String(s.tmplIndex![i]);
        case 'tatom': return String(s.tmplAtom![i]);
        case 'type': return typeText('atom', s.type[i]);
        case 'q': return shortest(s.q[i]);
        // measured with native write_data: diameter 2r, and density = mass / volume (mass itself for r = 0)
        case 'diameter': return shortest(2 * s.radius![i]);
        // set.rst: density is "for a triangle (mass/distance^2 units) or line (mass/distance units) particle", so a line's
        // written density is its mass per unit length and a triangle's its mass per unit area (measured: density 3 on a
        // length-2 segment is written back as 3; density 1.5 on a triangle of area 2 as 1.5)
        case 'density':
          if (isLineAtom(s, i)) return shortest(s.rmass![i] / s.custom.get(LINE_BONUS)!.data[2 * i]);
          if (isTriAtom(s, i)) return shortest(s.rmass![i] / triangleAreaOfBody(triCorners(s, i)));
          if (s.radius) return shortest(s.radius[i] > 0 ? s.rmass![i] / sphereMass(s.radius[i], 1) : s.rmass![i]);
          return shortest(isEllipsoid(s, i) ? s.rmass![i] / ellipsoidVolume(s, i) : s.rmass![i]);
        case 'ellipsoidflag': return isEllipsoid(s, i) ? '1' : '0';
        case 'lineflag': return isLineAtom(s, i) ? '1' : '0';
        case 'triangleflag': return isTriAtom(s, i) ? '1' : '0';
        // measured with native write_data: atom_style peri writes its volume and the density (= mass) column
        case 'volume': return shortest(s.vfrac![i]);
        case 'mux': return shortest(mu![4 * i]);
        case 'muy': return shortest(mu![4 * i + 1]);
        case 'muz': return shortest(mu![4 * i + 2]);
        case 'spx': return shortest(s.sp![4 * i]);
        case 'spy': return shortest(s.sp![4 * i + 1]);
        case 'spz': return shortest(s.sp![4 * i + 2]);
        case 'sp': return shortest(s.sp![4 * i + 3]);
        // atom_style.html, electron: espin is an integer (native write_data), eradius a distance
        case 'espin': return String(s.custom.get(ELECTRON_SPIN)!.data[i]);
        case 'eradius': return shortest(s.custom.get(ELECTRON_RADIUS)!.data[i]);
        case 'area': return shortest(s.custom.get(DIEL_AREA)!.data[i]);
        case 'ed': return shortest(s.custom.get(DIEL_ED)!.data[i]);
        case 'em': return shortest(s.custom.get(DIEL_EM)!.data[i]);
        case 'epsilon': return shortest(s.custom.get(DIEL_EPSILON)!.data[i]);
        case 'curvature': return shortest(s.custom.get(DIEL_CURVATURE)!.data[i]);
        case 'esph': return shortest(s.custom.get(SPH_ESPH)!.data[i]);
        case 'cv': return shortest(s.custom.get(SPH_CV)!.data[i]);
        // atom_style edpd / mdpd / tdpd (read_data.html): edpd_temp, edpd_cv, rho and ccN; sph and mdpd share the rho prop
        case 'edpd_temp': return shortest(s.custom.get(EDPD_TEMP)!.data[i]);
        case 'edpd_cv': return shortest(s.custom.get(EDPD_CV)!.data[i]);
        // read_data.html, rheo: the status is an integer, rho a density (sph and mdpd share their rho prop)
        case 'rheo_status': return String(s.custom.get(RHEO_STATUS)!.data[i]);
        // read_data.html, rheo/thermal: energy (the RHEO energy prop)
        case 'energy': return shortest(s.custom.get(RHEO_ESPH)!.data[i]);
        case 'rho': return shortest(s.custom.get(isRheoStyle(s.atomStyle) ? RHEO_RHO : MDPD_RHO)!.data[i]);
        default: {
          const k = /^cc(\d+)$/.exec(c);
          if (k) return shortest(s.custom.get(TDPD_CC)!.data[nsp * i + Number(k[1]) - 1]);
          return shortest(xo[3 * i + 'xyz'.indexOf(c)]);
        }
      }
    });
    out.push(`${v.join(' ')} ${s.image[3 * i]} ${s.image[3 * i + 1]} ${s.image[3 * i + 2]}`);
  }
  out.push('', 'Velocities', '');
  const angRaw = s.omega ?? s.angmom;
  const ang = angRaw && gen ? mapVectors(angRaw, (w) => unrotateVector(gen.Q, w)) : angRaw;
  // the tri style writes its angular momentum after the angular velocity; electron writes ervel after v
  // (read_data.html, Velocities of electron: "atom-ID vx vy vz ervel")
  const triL = isTriStyle(s.atomStyle) ? s.angmom : null;
  const rvelOut = isElectronStyle(s.atomStyle) ? s.custom.get(ELECTRON_RVEL) : undefined;
  for (const i of order) {
    let w = ang ? ` ${shortest(ang[3 * i])} ${shortest(ang[3 * i + 1])} ${shortest(ang[3 * i + 2])}` : '';
    if (triL) w += ` ${shortest(triL[3 * i])} ${shortest(triL[3 * i + 1])} ${shortest(triL[3 * i + 2])}`;
    const er = rvelOut ? ` ${shortest(rvelOut.data[i])}` : '';
    out.push(`${s.id[i]} ${shortest(vo[3 * i])} ${shortest(vo[3 * i + 1])} ${shortest(vo[3 * i + 2])}${er}${w}`);
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
  // read_data.html: "one line per line segment" with the line syntax atom-ID x1 y1 x2 y2, and "one line per triangle" with
  // the line syntax atom-ID x1 y1 z1 x2 y2 z2 x3 y3 z3. The sections must appear after the Atoms section. Measured with
  // native write_data: a line's end points are written as its centre -/+ half its length along theta, and the triangle
  // corners as their coordinates (the atom's position is the corner centroid)
  if (isLineStyle(s.atomStyle) && flagCount(s, LINE_FLAG) > 0) {
    out.push('', 'Lines', '');
    const bonus = s.custom.get(LINE_BONUS)!.data;
    for (const i of order) {
      if (!isLineAtom(s, i)) continue;
      const e = lineEnds(s.x[3 * i], s.x[3 * i + 1], bonus[2 * i], bonus[2 * i + 1]);
      out.push(`${s.id[i]} ${e.map(shortest).join(' ')}`);
    }
  }
  if (isTriStyle(s.atomStyle) && flagCount(s, TRI_FLAG) > 0) {
    out.push('', 'Triangles', '');
    for (const i of order) {
      if (!isTriAtom(s, i)) continue;
      const cen = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
      const q = [0, 1, 2, 3].map((d) => s.quat![4 * i + d]);
      const pts = [0, 1, 2].flatMap((k) => triangleCorner(cen, q, triCorners(s, i), k));
      out.push(`${s.id[i]} ${pts.map(shortest).join(' ')}`);
    }
  }
  let maxId = 0;
  for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
  const local = new Int32Array(maxId + 1).fill(-1);
  order.forEach((i, k) => { local[s.id[i]] = k; });
  const slot = (id: number): number => (id <= maxId ? local[id] : -1);
  for (const [list, title, owner, lk] of [[t.bonds, 'Bonds', 0, 'bond'], [t.angles, 'Angles', 1, 'angle'], [t.dihedrals, 'Dihedrals', 1, 'dihedral'], [t.impropers, 'Impropers', 1, 'improper']] as const) {
    // measured with native write_data: no topology sections for atom_style template (the templates hold them)
    if (!list.n || isTemplateStyle(s.atomStyle)) continue;
    out.push('', title, '');
    const w = list.width;
    const rows = Array.from({ length: list.n }, (_, e) => e)
      .sort((a, b) => slot(list.atoms[a * w + owner]) - slot(list.atoms[b * w + owner]) || a - b);
    rows.forEach((e, k) => {
      const ids = Array.from(list.atoms.subarray(e * w, (e + 1) * w));
      // Measured with native LAMMPS (black box): an interaction turned off by delete_bonds (negative type) is
      // written with its positive type
      out.push(`${k + 1} ${typeText(lk, Math.abs(list.type[e]))} ${ids.join(' ')}`);
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
