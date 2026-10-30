import type { System } from '../system';
import type { AtomStyle, SimState, TopoList } from '../types';
import { StyleError } from '../force/types';
import { bnumeric, inumeric, numeric } from '../force/util';
import { errorUrl } from '../formula';
import { isSmdStyle, SMD_CONTACT } from '../smd';
import { isSphStyle, SPH_CV, SPH_DESPH, SPH_DRHO, SPH_ESPH, SPH_RHO, SPH_VEST, appendAtoms, setOxdnaPartners, setSpinVector, atomSubStyles, countEllipsoids, ellipsoidVolume, emptyState, isEllipsoid, isMolecularStyle, isTemplateStyle, maxAtomId, nativeOrder, pushTopo, sphereMass, templateStyleId, topologyLevel, isLineStyle, isTriStyle, isLineAtom, isTriAtom, LINE_FLAG, LINE_BONUS, TRI_FLAG, TRI_BONUS, TRI_BONUS_COLS, ELECTRON_RADIUS, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle, isDielectricStyle, DIEL_AREA, DIEL_ED, DIEL_EM, DIEL_EPSILON, DIEL_CURVATURE, DIEL_QSCALED, isEdpdStyle, isMdpdStyle, tdpdSpecies, EDPD_TEMP, EDPD_CV, MDPD_RHO, TDPD_CC, RHEO_RHO, RHEO_STATUS, isRheoStyle, isRheoThermalStyle, RHEO_ESPH, isDpdStyle, DPD_THETA } from '../atoms';
import { lineFromEnds, lineEnds, triangleCorner, triangleAreaOfBody, type LineBonus } from '../lineTri';
import { triBodyHook } from '../styles';
import type { TriBody } from '../triBonus';

import { expandTemplateTopology } from '../template';
import { BODY_RECORD, POLY_OFF, bodyRecordOffset, bodyRecordStride, parseBodyStyle, polyRecordLength, polyStoredCounts, polyhedronFrame, principalFrame } from '../bodies';
import { gplHook } from '../gplHooks';
import { makeBox, SHRINK_PAD } from '../domain';
import { LABEL_KINDS, LABEL_SECTION, labelKindOfSection, type LabelKind } from '../labelmap';
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


/** The data-file sections (native section_keywords). */
export const DATA_SECTION_NAMES: ReadonlySet<string> = new Set([
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
/** The words of a data-file line up to the first that starts a comment ('#'), as native's Tokenizer and comment check. */
const words = (l: string): string[] => {
  const out: string[] = [];
  for (const w of l.trim().split(/\s+/)) {
    if (!w) continue;
    if (w.startsWith('#')) break;
    out.push(w);
  }
  return out;
};

/**
 * native utils::is_type: 0 for a numeric type (digits, with at most one '*'), 1 for a type label (no leading digit,
 * '*' or '#', no blank), -1 otherwise.
 */
const isType = (str: string): number => {
  if (!str) return -1;
  let numericOnly = true;
  let nstar = 0;
  for (const c of str) {
    if (c >= '0' && c <= '9') continue;
    if (c === '*') { nstar++; continue; }
    numericOnly = false;
  }
  if (numericOnly && nstar < 2) return 0;
  if (/^[0-9*#]/.test(str) || /[ \t\r\n\f]/.test(str)) return -1;
  return 1;
};

export const readData = (sys: System, text: string, opts: ReadDataOptions): void => {
  // As native LAMMPS does (src/read_data.cpp ReadData::command at b332f28), before the file is read
  if (sys.dimension === 2 && sys.boundary[2][0] !== 'p') throw new StyleError('Cannot run 2d simulation with nonperiodic Z dimension');
  // As native LAMMPS does (src/read_data.cpp header, parse_keyword and the section loop of command at b332f28): the
  // file is read line by line as fgets reads it, the header lines are recognized by their pattern, and each section
  // reads its count of lines, blank ones included
  const lines = text.split('\n');
  if (lines.length && lines[lines.length - 1] === '') lines.pop();
  let k = 0;
  const next = (): string | null => (k < lines.length ? lines[k++].replace(/\r$/, '') : null);
  /** native read_lines_from_file: exactly n lines; none (n = 0) or too few is the end of the file. */
  const readLines = (n: number): string[] => {
    if (n <= 0 || k + n > lines.length) throw new StyleError('Unexpected end of data file');
    const out = lines.slice(k, k + n).map((l) => l.replace(/\r$/, ''));
    k += n;
    return out;
  };
  // a read after the box exists (add keyword) is a subsequent read: its bonds are limited by the storage
  const subsequentRead = sys.hasBox;
  const h: Record<string, number> = {};
  let orthoLines = false;
  let lo: [number, number, number] = [-0.5, -0.5, -0.5];
  let hi: [number, number, number] = [0.5, 0.5, 0.5];
  let tilt: [number, number, number] | null = null;
  // general triclinic header (Howto_triclinic.html): avec, bvec, cvec, abc origin
  const gen: { avec?: V3; bvec?: V3; cvec?: V3; origin?: V3 } = {};
  // read_data.html: "header lines containing this string will be passed to fix"
  const headerFixes = [...opts.fixSections.values()].filter((e) => e.header !== 'NULL');
  const subs = atomSubStyles(sys.atomStyle);
  if (next() === null) throw new StyleError('Unexpected end of data file');
  /** The line the header stopped at, without its comment (null: the file ended in the header). */
  let stopLine: string | null = null;
  const F = '[-+0-9.eE]+';
  const triple = (w: string[]): V3 => [numeric(w[0]), numeric(w[1]), numeric(w[2])];
  for (;;) {
    const raw = next();
    if (raw === null) break;
    const hash = raw.indexOf('#');
    const t = hash >= 0 ? raw.slice(0, hash) : raw;
    if (/^[ \t\n\r]*$/.test(t)) continue;
    const hf = headerFixes.find((e) => t.includes(e.header));
    if (hf) {
      const fx = sys.fix(hf.fixId) as unknown as DataFix;
      if (!isDataFix(fx) || typeof fx.readHeader !== 'function') {
        throw new StyleError(`read_data fix ${hf.fixId}: header-string must be NULL (fix ${fx.style} reads no header lines)`);
      }
      fx.readHeader(t.trim(), k);
      continue;
    }
    const w = t.trim().split(/\s+/);
    const L = `${t}\n`;
    const is = (re: string): boolean => new RegExp(re).test(L);
    if (is('^\\s*\\d+\\s+atoms\\s')) h.atoms = bnumeric(w[0]);
    else if (is('^\\s*\\d+\\s+ellipsoids\\s')) {
      if (!subs.includes('ellipsoid')) throw new StyleError('No ellipsoids allowed with this atom style');
      h.ellipsoids = bnumeric(w[0]);
    } else if (is('^\\s*\\d+\\s+lines\\s')) {
      if (!subs.includes('line')) throw new StyleError('No lines allowed with this atom style');
      h.lines = bnumeric(w[0]);
    } else if (is('^\\s*\\d+\\s+triangles\\s')) {
      if (!subs.includes('tri')) throw new StyleError('No triangles allowed with this atom style');
      h.triangles = bnumeric(w[0]);
    } else if (is('^\\s*\\d+\\s+bodies\\s')) {
      if (!subs.includes('body')) throw new StyleError('No bodies allowed with this atom style');
      h.bodies = bnumeric(w[0]);
    } else if (is('^\\s*\\d+\\s+bonds\\s')) h.bonds = bnumeric(w[0]);
    else if (is('^\\s*\\d+\\s+angles\\s')) h.angles = bnumeric(w[0]);
    else if (is('^\\s*\\d+\\s+dihedrals\\s')) h.dihedrals = bnumeric(w[0]);
    else if (is('^\\s*\\d+\\s+impropers\\s')) h.impropers = bnumeric(w[0]);
    else if (is('^\\s*\\d+\\s+atom\\s+types\\s')) h['atom types'] = inumeric(w[0]);
    // native's pattern for bond types has no ^
    else if (is('\\s*\\d+\\s+bond\\s+types\\s')) h['bond types'] = inumeric(w[0]);
    else if (is('^\\s*\\d+\\s+angle\\s+types\\s')) h['angle types'] = inumeric(w[0]);
    else if (is('^\\s*\\d+\\s+dihedral\\s+types\\s')) h['dihedral types'] = inumeric(w[0]);
    else if (is('^\\s*\\d+\\s+improper\\s+types\\s')) h['improper types'] = inumeric(w[0]);
    else if (/extra (bond|angle|dihedral|improper|special) per atom/.test(t)) {
      if (opts.add === 'none') inumeric(w[0]);
    } else if (is(`^\\s*${F}\\s+${F}\\s+xlo\\s+xhi\\s`)) { lo[0] = numeric(w[0]); hi[0] = numeric(w[1]); orthoLines = true; }
    else if (is(`^\\s*${F}\\s+${F}\\s+ylo\\s+yhi\\s`)) { lo[1] = numeric(w[0]); hi[1] = numeric(w[1]); orthoLines = true; }
    else if (is(`^\\s*${F}\\s+${F}\\s+zlo\\s+zhi\\s`)) { lo[2] = numeric(w[0]); hi[2] = numeric(w[1]); orthoLines = true; }
    else if (is(`^\\s*${F}\\s+${F}\\s+${F}\\s+xy\\s+xz\\s+yz\\s`)) { tilt = triple(w); orthoLines = true; }
    else if (is(`^\\s*${F}\\s+${F}\\s+${F}\\s+avec\\s`)) gen.avec = triple(w);
    else if (is(`^\\s*${F}\\s+${F}\\s+${F}\\s+bvec\\s`)) gen.bvec = triple(w);
    else if (is(`^\\s*${F}\\s+${F}\\s+${F}\\s+cvec\\s`)) gen.cvec = triple(w);
    else if (is(`^\\s*${F}\\s+${F}\\s+${F}\\s+abc\\s+origin\\s`)) gen.origin = triple(w);
    else { stopLine = t; break; }
  }
  /**
   * native parse_keyword: the section keyword (the next line that is not blank or a comment, or the given line) and
   * the style after its '#'; the line after the keyword is always skipped, and when there is none the keyword is
   * empty (the end of the file).
   */
  const parseKeyword = (given: string | null): { keyword: string; style: string } => {
    let line = given ?? next();
    while (line !== null && (/^[ \t\n\r]*$/.test(line) || /^[ \t\n\r]*#/.test(line))) line = next();
    if (line === null || next() === null) return { keyword: '', style: '' };
    const hash = line.indexOf('#');
    return { keyword: (hash >= 0 ? line.slice(0, hash) : line).trim(), style: hash >= 0 ? line.slice(hash + 1).trim() : '' };
  };
  // the header ends at the first section keyword, which must be a data-file section (a fix section is not one here)
  let kw = { keyword: '', style: '' };
  if (stopLine !== null) {
    kw = parseKeyword(stopLine);
    if (!DATA_SECTION_NAMES.has(kw.keyword)) throw new StyleError(`Unknown identifier in data file: ${kw.keyword}${errorUrl(1)}`);
    const level = topologyLevel(sys.atomStyle);
    const tmpl = subs.includes('template');
    const allow = [level >= 1 || tmpl || subs.includes('bpm/sphere'), level >= 2 || tmpl, level >= 3 || tmpl, level >= 3 || tmpl];
    (['bond', 'angle', 'dihedral', 'improper'] as const).forEach((x, i) => {
      if (((h[`${x}s`] ?? 0) || (h[`${x} types`] ?? 0)) && !allow[i]) throw new StyleError(`No ${x}s allowed with this atom style`);
    });
    (['bond', 'angle', 'dihedral', 'improper'] as const).forEach((x) => {
      if ((h[`${x}s`] ?? 0) > 0 && !((h[`${x} types`] ?? 0) + opts.extraTypes[['bond', 'angle', 'dihedral', 'improper'].indexOf(x) + 1] > 0)) {
        throw new StyleError(`${x[0].toUpperCase()}${x.slice(1)}s defined but no ${x} types`);
      }
    });
    if (tmpl && ((h.bonds ?? 0) || (h.angles ?? 0) || (h.dihedrals ?? 0) || (h.impropers ?? 0))) {
      throw new StyleError('No molecule topology allowed with atom style template');
    }
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
  // native Domain::set_initial_box: every dimension needs lo < hi
  if (!adding && !genFrame && (lo[0] >= hi[0] || lo[1] >= hi[1] || lo[2] >= hi[2])) throw new StyleError('Box bounds are invalid or missing');
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
   * An atom type of an Atoms line as native's data_atoms reads it: a numeric type plus the offset, in range, or a
   * defined type label.
   */
  const atomTypeOf = (typestr: string, off: number, loc: string, line: string, ntypes: number): number => {
    const kind = isType(typestr);
    if (kind === 0) {
      const t = inumeric(typestr) + off;
      if (t < 1 || t > ntypes) throw new StyleError(`Invalid atom type ${t} in ${loc}: ${line}`);
      return t;
    }
    const t = kind === 1 ? sys.labels.typeOf('atom', typestr) : undefined;
    if (t === undefined) throw new StyleError(`Invalid line in ${loc}: ${line}`);
    return t;
  };
  /** atom_style body: atoms with bodyflag 1 waiting for their Bodies entry. */
  const bodyPending = new Set<number>();
  const topo: { kind: 'bonds' | 'angles' | 'dihedrals' | 'impropers'; type: number; ids: number[] }[] = [];
  const fixRows: { fx: DataFix; rows: { w: string[]; at: number }[]; title: string }[] = [];
  // sections, in native's dispatch order: a data-file section, else a fix's section, else an unknown identifier
  let atomflag = false;
  const seen = new Set<string>();
  /** native settypeflag: a Masses or coefficient section was read, so no Type Labels section may follow */
  let settypeflag = false;
  const kindIndex = { Bonds: 1, Angles: 2, Dihedrals: 3, Impropers: 4 } as Record<string, number>;
  const allowed = (i: number): boolean => {
    const level = topologyLevel(sys.atomStyle);
    const tmpl = subs.includes('template');
    return [level >= 1 || tmpl || subs.includes('bpm/sphere'), level >= 2 || tmpl, level >= 3 || tmpl, level >= 3 || tmpl][i];
  };
  for (; kw.keyword !== ''; kw = parseKeyword(null)) {
    const title = kw.keyword;
    const hint = kw.style;
    seen.add(title);
    if (!DATA_SECTION_NAMES.has(title)) {
      // read_data.html: "fix values = fix-ID header-string section-string" — a section named section-string
      const fixEntry = opts.fixSections.get(title);
      if (fixEntry === undefined) throw new StyleError(`Unknown identifier in data file: ${title}${errorUrl(1)}`);
      const fixId = fixEntry.fixId;
      const fx = sys.fix(fixId) as unknown as DataFix;
      if (!isDataFix(fx) || (typeof fx.readValues !== 'function' && typeof fx.readSection !== 'function')) {
        throw new StyleError(`read_data fix ${fixId}: fix style ${fx.style} does not read data-file sections`);
      }
      // the fix says how many lines its section holds (per-atom fixes: one line per atom)
      const nlines = fx.sectionLines ? fx.sectionLines(natoms, title) : natoms;
      const at0 = k;
      const rows: { w: string[]; at: number }[] = [];
      if (nlines > 0) {
        readLines(nlines).forEach((l, i) => {
          const t = l.replace(/#.*/, '').trim();
          if (t) rows.push({ w: t.split(/\s+/), at: at0 + i + 1 });
        });
      }
      fixRows.push({ fx, rows, title });
      continue;
    }
    const labelKind = labelKindOfSection(title);
    if (labelKind) {
      if (settypeflag) throw new StyleError('Must read Type Labels before any section involving types');
      if (adding) throw new StyleError(`read_data: ${title} with add append/merge is not supported by the browser engine`);
      const before = { atom: 'Atoms', bond: 'Bonds', angle: 'Angles', dihedral: 'Dihedrals', improper: 'Impropers' }[labelKind];
      if (seen.has(before)) throw new StyleError(`Must read ${title} before ${before}`);
    }
    if (title === 'Masses' || title.endsWith(' Coeffs')) settypeflag = true;
    if (title === 'Atoms') atomflag = true;
    if (title === 'Velocities' && !atomflag) throw new StyleError('Must read Atoms before Velocities');
    if (kindIndex[title]) {
      if (!((h[title.toLowerCase()] ?? 0) > 0)) throw new StyleError(`Invalid data file section: ${title}`);
      if (!atomflag) throw new StyleError(`Must read Atoms before ${title}`);
    }
    const bonusSub = { Ellipsoids: 'ellipsoid', Lines: 'line', Triangles: 'tri', Bodies: 'body' }[title];
    if (bonusSub) {
      if (!subs.includes(bonusSub)) throw new StyleError(`Invalid data file section: ${title}`);
      if (!atomflag) throw new StyleError(`Must read Atoms before ${title}`);
    }
    if ((title === 'Pair Coeffs' || title === 'PairIJ Coeffs') && !sys.ff.pair && !opts.nocoeff) {
      throw new StyleError(`Must define pair_style before ${title}`);
    }
    const coeffKind = title === 'Bond Coeffs' ? 'bond' : title === 'Angle Coeffs' ? 'angle' : title === 'Dihedral Coeffs' ? 'dihedral'
      : title === 'Improper Coeffs' ? 'improper' : CLASS2_SECTIONS[title]?.[0];
    if (coeffKind) {
      if (!allowed(['bond', 'angle', 'dihedral', 'improper'].indexOf(coeffKind))) throw new StyleError(`Invalid data file section: ${title}`);
      if (!sys.ff[coeffKind] && !opts.nocoeff) throw new StyleError(`Must define ${coeffKind}_style before ${title}`);
    }
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
    // count of lines: each section reads its count of lines, blank and comment lines included
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
        case 'Ellipsoids': return h.ellipsoids ?? 0;
        case 'Lines': return h.lines ?? 0;
        case 'Triangles': return h.triangles ?? 0;
        default: throw new StyleError(`data file section '${title}' needs features the browser engine does not support`);
      }
    })();
    if (title === 'Atoms' && hint && hint !== style) sys.warn(`Atom style in data file ${hint} differs from currently defined atom style ${style}`);
    // Masses, Pair Coeffs and PairIJ Coeffs read their lines even for no types (native read_lines_from_file then
    // reports the end of the file); the other sections read nothing for a count of 0
    const at0 = k;
    const raw = count > 0 || title === 'Masses' || title === 'Pair Coeffs' || title === 'PairIJ Coeffs' ? readLines(count) : [];
    /** The section's lines that are not blank, their words up to a word that starts a comment. */
    const body: { w: string[]; at: number }[] = [];
    raw.forEach((l, i) => {
      const w = words(l);
      if (w.length) body.push({ w, at: at0 + i + 1 });
    });
    switch (title) {
      case 'Atoms': {
        sawAtoms = true;
        // As native LAMMPS does (src/atom.cpp data_atoms, src/atom_vec.cpp data_atom and the styles' data_atom_post at
        // b332f28): the first line sets the number of words (the style's columns, or 3 more with image flags); blank and
        // comment lines are skipped, a line with fewer words, or more that do not start a comment, is refused
        const loc = 'Atoms section of data file';
        const nwords = raw.length ? words(raw[0]).length : 0;
        if (raw.length && nwords !== cols.length && nwords !== cols.length + 3) {
          throw new StyleError(`Incorrect format in ${loc}: ${raw[0].trim()}${errorUrl(2)}`);
        }
        const rows: { w: string[]; line: string; at: number }[] = [];
        raw.forEach((l, i) => {
          const v = l.trim() ? l.trim().split(/\s+/) : [];
          if (!v.length || v[0].startsWith('#')) return;
          rows.push({ w: v, line: l.trim(), at: at0 + i + 1 });
        });
        const nrow = rows.length;
        const xcol = cols.indexOf('x');
        const resetImage = [false, false, false];
        const x = new Float64Array(3 * nrow), type = new Int32Array(nrow), id = new Int32Array(nrow);
        const mol = new Int32Array(nrow), q = new Float64Array(nrow), image = new Int32Array(3 * nrow);
        const radius = new Float64Array(nrow), density = new Float64Array(nrow), vol = new Float64Array(nrow);
        const mu = s.mu ? new Float64Array(4 * nrow) : null;
        const spin = s.sp ? new Float64Array(4 * nrow) : null;
        const eflag = s.shape ? new Uint8Array(nrow) : null;
        // atom_style line: lineflag; atom_style tri: triangleflag (read_data.html: 1 for line segment or triangular particles, 0 for point particles)
        const lflag = isLineStyle(style) || isTriStyle(style) ? new Uint8Array(nrow) : null;
        const tIdx = new Int32Array(nrow), tAt = new Int32Array(nrow);
        const espin = new Float64Array(nrow), eradius = new Float64Array(nrow);
        const rheoStatus = isRheoStyle(style) ? new Float64Array(nrow) : null, rheoRho = isRheoStyle(style) ? new Float64Array(nrow) : null;
        const rheoEsph = isRheoThermalStyle(style) ? new Float64Array(nrow) : null;
        const diel = isDielectricStyle(style) ? { area: new Float64Array(nrow), ed: new Float64Array(nrow), em: new Float64Array(nrow), epsilon: new Float64Array(nrow), curvature: new Float64Array(nrow) } : null;
        const sph = isSphStyle(style) ? { rho: new Float64Array(nrow), esph: new Float64Array(nrow), cv: new Float64Array(nrow) } : null;
        const bflag = parseBodyStyle(style) ? new Uint8Array(nrow) : null;
        // atom_style edpd / mdpd / tdpd (read_data.html: "edpd_temp", "edpd_cv", "rho", "ccN"); the species nrow of tdpd
        const nsp = tdpdSpecies(style);
        const dm = isEdpdStyle(style) || isMdpdStyle(style) || nsp > 0
          ? { temp: new Float64Array(nrow), cv: new Float64Array(nrow), rho: new Float64Array(nrow), cc: new Float64Array(nrow * nsp) }
          : null;
        const dpdTh = isDpdStyle(style) ? new Float64Array(nrow) : null;
        // atom_style smd (read_data.html): the contact radius and the mass are columns; the kernel radius is the radius
        const smdCr = isSmdStyle(style) ? new Float64Array(nrow) : null;
        const smdRm = isSmdStyle(style) ? new Float64Array(nrow) : null;
        rows.forEach(({ w: all, line, at }, a) => {
          // line by line: the format of a line is checked when it is reached
          if (all.length < nwords || (all.length > nwords && !all[nwords].startsWith('#'))) {
            throw new StyleError(`Incorrect format in ${loc}: ${line}${errorUrl(2)}`);
          }
          const w = all.slice(0, nwords);
          if (nwords > cols.length) {
            const im = [inumeric(w[nwords - 3]), inumeric(w[nwords - 2]), inumeric(w[nwords - 1])];
            if (sys.dimension === 2 && im[2] !== 0) throw new StyleError('Z-direction image flag must be 0 for 2d-systems');
            for (let d = 0; d < 3; d++) {
              if (!s.box.periodic[d] && im[d] !== 0) { resetImage[d] = true; im[d] = 0; }
              image[3 * a + d] = im[d];
            }
          }
          const xd = [numeric(w[xcol]), numeric(w[xcol + 1]), numeric(w[xcol + 2])];
          if (sys.dimension === 2) {
            if (Math.abs(xd[2]) > 1e-12) throw new StyleError('Read_data atom z coord is non-zero for 2d simulation');
            xd[2] = 0;
          }
          for (let d = 0; d < 3; d++) x[3 * a + d] = xd[d] + opts.shift[d];
          let typestr = '';
          for (let c = 0; c < cols.length; c++) {
            const v = w[c];
            switch (cols[c]) {
              // atom and molecule IDs are 32-bit (native tagint, utils::tnumeric)
              case 'id': id[a] = inumeric(v); break;
              case 'mol': { const m = inumeric(v); mol[a] = m + (m > 0 ? molBase : 0); break; }
              case 'type': typestr = v; break;
              case 'q': q[a] = numeric(v); break;
              case 'diameter': radius[a] = numeric(v) / 2; break;
              case 'density': density[a] = numeric(v); break;
              case 'volume': vol[a] = numeric(v); break;
              case 'x': case 'y': case 'z': break;
              case 'ellipsoidflag': {
                const f = inumeric(v);
                if (f !== 0 && f !== 1) throw new StyleError(`Invalid ellipsoid flag in ${loc}`);
                eflag![a] = f;
                break;
              }
              case 'lineflag': case 'triangleflag': {
                const f = inumeric(v);
                if (f !== 0 && f !== 1) throw new StyleError(`Invalid ${cols[c] === 'lineflag' ? 'line' : 'tri'} flag in ${loc}`);
                lflag![a] = f;
                break;
              }
              case 'bodyflag': {
                const f = inumeric(v);
                if (f !== 0 && f !== 1) throw new StyleError(`Invalid body flag in ${loc}`);
                bflag![a] = f;
                break;
              }
              case 'mass': density[a] = numeric(v); break;
              case 'tindex': tIdx[a] = inumeric(v); break;
              case 'tatom': tAt[a] = inumeric(v); break;
              case 'mux': mu![4 * a] = numeric(v); break;
              case 'muy': mu![4 * a + 1] = numeric(v); break;
              case 'muz': mu![4 * a + 2] = numeric(v); break;
              case 'spx': spin![4 * a] = numeric(v); break;
              case 'spy': spin![4 * a + 1] = numeric(v); break;
              case 'spz': spin![4 * a + 2] = numeric(v); break;
              case 'sp': spin![4 * a + 3] = numeric(v); break;
              // read_data.html: "espin = electron spin (+1/-1), 0 = nuclei, 2 = fixed-core, 3 = pseudo-cores (i.e. ECP)"
              case 'espin': espin[a] = inumeric(v); break;
              // atom_style dpd: "dpdTheta" is the internal temperature of the atom (read_data: "must be > 0.0")
              case 'dpdTheta': dpdTh![a] = numeric(v); break;
              // read_data.html: "eradius = electron radius (or fixed-core radius)"
              case 'eradius': eradius[a] = numeric(v); break;
              // read_data.html, row rheo: "status = status or phase of RHEO particles (unitless)", "rho = density"
              case 'rheo_status': rheoStatus![a] = inumeric(v); break;
              // read_data.html, rheo/thermal: "energy = energy of RHEO particles (need units)"
              case 'energy': rheoEsph![a] = numeric(v); break;
              case 'area': case 'ed': case 'em': case 'epsilon': case 'curvature': {
                const dk = cols[c] as 'area' | 'ed' | 'em' | 'epsilon' | 'curvature';
                diel![dk][a] = numeric(v);
                break;
              }
              // read_data.html, sph: "rho = density (need units)", "esph = energy (need units) for SPH particles", "cv = heat capacity (need units) for SPH particles"
              case 'esph': case 'cv': {
                const sk = cols[c] as 'esph' | 'cv';
                sph![sk][a] = numeric(v);
                break;
              }
              // rho is a column of atom_style sph, mdpd and rheo (read_data.html)
              case 'rho':
                if (rheoRho) rheoRho[a] = numeric(v);
                else if (sph) sph.rho[a] = numeric(v);
                else dm!.rho[a] = numeric(v);
                break;
              // atom_style smd (read_data.html): "volume", "mass", "kradius", "cradius"; the x0 values are not kept (x0 = x)
              case 'vfrac': vol[a] = numeric(v); break;
              case 'rmass': smdRm![a] = numeric(v); break;
              case 'kradius': radius[a] = numeric(v); break;
              case 'cradius': smdCr![a] = numeric(v); break;
              case 'x0': case 'y0': case 'z0': numeric(v); break;
              case 'edpd_temp': dm!.temp[a] = numeric(v); break;
              case 'edpd_cv': dm!.cv[a] = numeric(v); break;
              default: {
                const k = /^cc(\d+)$/.exec(cols[c]);
                if (k && dm) dm.cc[nsp * a + Number(k[1]) - 1] = numeric(v);
                break;
              }
            }
          }
          if (mu) mu[4 * a + 3] = Math.hypot(mu[4 * a], mu[4 * a + 1], mu[4 * a + 2]);
          if (spin) setSpinVector(spin, a, spin[4 * a], spin[4 * a + 1], spin[4 * a + 2], spin[4 * a + 3], `data file line ${at}`);
          // the ID before the offsets; native names the atom's place among the atoms (its local index)
          if (!(id[a] > 0)) throw new StyleError(`Invalid atom ID ${id[a]} in line ${s.n + a + 1} of ${loc}`);
          if (idBase) id[a] += idBase;
          if (sys.atomStyle === 'peri' && !(density[a] > 0)) throw new StyleError(`Invalid mass in ${loc}`);
          // ellipsoid, line, tri, body and sphere styles: the density (a sphere's mass: density times volume, or the
          // density of a point particle) must be > 0
          const sphereSub = subs.includes('sphere') || subs.includes('bpm/sphere');
          if ((eflag || lflag || bflag || sphereSub) && !(density[a] > 0)) {
            const m = sphereSub && radius[a] > 0 ? sphereMass(radius[a], density[a]) : density[a];
            if (!(m > 0)) throw new StyleError(`Invalid density in ${loc}`);
          }
          if (tmplId !== null) {
            // native atom_vec_template data_atom_post: the template index and atom (0 = none) of a set of the template
            const sets = sys.molecules.get(tmplId) ?? [];
            const mi = tIdx[a] - 1, ma = tAt[a] - 1;
            if (mi < -1 || mi >= sets.length) throw new StyleError(`Invalid template index in ${loc}`);
            if (ma < -1 || (mi >= 0 && ma >= sets[mi].natoms)) throw new StyleError(`Invalid template atom in ${loc}`);
          }
          type[a] = atomTypeOf(typestr, toff, loc, line, s.ntypes);
        });
        if (opts.add === 'merge' || opts.add === 'none') {
          const existing = new Set(Array.from(s.id.subarray(0, s.n)));
          for (const i of id) if (existing.has(i)) throw new StyleError('Duplicate atom IDs exist');
        }
        // read_data.html: "the density is used in conjunction with the particle volume to set the mass
        // of each particle as mass = density * volume ... If the volume is 0.0, meaning a point
        // particle, then the density value is used as the mass."
        // ellipsoids: the density becomes a mass once the Ellipsoids section gives the volume; a point
        // particle (ellipsoidflag 0) takes the density as its mass
        // atom_style peri. Measured with native LAMMPS (black box): read_data gives mass = density (volume is kept apart)
        const rmass = smdRm ?? (s.radius ? radius.map((r, a) => sphereMass(r, density[a])) : s.shape || s.vfrac || bflag ? Float64Array.from(density) : undefined);
        // atom_style body: the mass column is the body's mass (read_data.html); a bodyflag 1 atom needs a Bodies entry
        if (bflag) for (let a = 0; a < nrow; a++) if (bflag[a]) bodyPending.add(id[a]);
        if (eflag) for (let a = 0; a < nrow; a++) if (eflag[a]) ellPending.set(id[a], density[a]);
        // atom_style line / tri: the density is per length or per area; the mass is set when the Lines or Triangles entry gives the size
        if (lflag && isLineStyle(style)) for (let a = 0; a < nrow; a++) if (lflag[a]) linePending.set(id[a], density[a]);
        if (lflag && isTriStyle(style)) for (let a = 0; a < nrow; a++) if (lflag[a]) triPending.set(id[a], density[a]);
        if (genFrame) {
          // read_data.html: coordinates "should be inside the general triclinic simulation box"; the
          // general -> restricted rotation is about the box origin (Howto_triclinic.html)
          const o = gen.origin ?? [0, 0, 0];
          for (let a = 0; a < nrow; a++) {
            const r = toRestrictedPoint(genFrame.Q, o, [x[3 * a], x[3 * a + 1], x[3 * a + 2]]);
            x[3 * a] = r[0]; x[3 * a + 1] = r[1]; x[3 * a + 2] = r[2];
          }
        }
        const firstNew = s.n;
        const electronCustom = isElectronStyle(s.atomStyle) ? new Map([[ELECTRON_SPIN, espin], [ELECTRON_RADIUS, eradius]]) : undefined;
        // Measured with native LAMMPS (black box): a data line with q 2.0 and epsilon 4.0 reads as q_scaled 0.5 (q / epsilon)
        const rheoCustom = rheoStatus && rheoRho ? new Map<string, Float64Array>([[RHEO_STATUS, rheoStatus], [RHEO_RHO, rheoRho], ...(rheoEsph ? [[RHEO_ESPH, rheoEsph] as [string, Float64Array]] : [])]) : undefined;
        if (dpdTh) for (let a = 0; a < nrow; a++) {
          if (!(dpdTh[a] > 0)) throw new StyleError('Internal temperature theta in Atoms section of data file must be > 0.0');
        }
        const dpdCustom = dpdTh ? new Map<string, Float64Array>([[DPD_THETA, dpdTh]]) : undefined;
        const dielCustom = diel ? new Map<string, Float64Array>([
          [DIEL_AREA, diel.area], [DIEL_ED, diel.ed], [DIEL_EM, diel.em], [DIEL_EPSILON, diel.epsilon], [DIEL_CURVATURE, diel.curvature],
          [DIEL_QSCALED, Float64Array.from(q, (qq, a) => qq / diel.epsilon[a])],
        ]) : undefined;
        // atom_style sph: the velocity estimate and the rates start at zero
        const sphCustom = sph ? new Map<string, Float64Array>([
          [SPH_RHO, sph.rho], [SPH_ESPH, sph.esph], [SPH_CV, sph.cv], [SPH_DRHO, new Float64Array(nrow)],
          [SPH_DESPH, new Float64Array(nrow)], [SPH_VEST, new Float64Array(3 * nrow)],
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
        // native: "Non-zero imageflag(s) in x direction for non-periodic boundary reset to zero" (once per direction)
        resetImage.forEach((r, d) => { if (r) sys.warn(`Non-zero imageflag(s) in ${'xyz'[d]} direction for non-periodic boundary reset to zero`); });
        let assigned = nrow;
        if (!s.box.triclinic) {
          for (let i = n0; i < s.n; i++) {
            for (let d = 0; d < 3; d++) {
              if (s.box.periodic[d] || s.box.boundary[d][0] !== 'f' || s.box.boundary[d][1] !== 'f') continue;
              const xd = s.x[3 * i + d];
              if (xd < s.box.lo[d] || xd >= s.box.hi[d]) { assigned--; break; }
            }
          }
        }
        // native: the atoms read (blank and comment lines are none) and kept must be the header's count
        if (assigned !== natoms) throw new StyleError(`Did not assign all atoms correctly${errorUrl(16)}`);
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
        // As native LAMMPS does (src/atom.cpp data_vels at b332f28): blank and comment lines are skipped, a line has
        // exactly the style's values, and its atom ID is at most the largest one read
        const need = tri ? 10 : s.omega || s.angmom ? 7 : electron ? 5 : 4;
        raw.forEach((l) => {
          const w = l.replace(/#.*/, '').trim() ? l.replace(/#.*/, '').trim().split(/\s+/) : [];
          if (!w.length) return;
          if (w.length !== need) throw new StyleError(`Incorrect format in Velocities section of data file: ${l.trim()}${errorUrl(2)}`);
          const tag = inumeric(w[0]) + (idBase || 0);
          if (tag <= 0 || tag > maxAtomId(s)) throw new StyleError(`Invalid atom ID ${tag} in Velocities section of data file: ${l}`);
          vel.set(tag, w.slice(1, need).map((x) => numeric(x)));
        });
        break;
      }
      case 'Masses':
        // measured with native LAMMPS (atom_style sphere): "Cannot set mass for atom style sphere"
        // read_data.html: "This section cannot be used for atom styles that define a mass for individual atoms - e.g. atom_style sphere"
        if (s.atomStyle === 'sphere' || s.atomStyle === 'bpm/sphere' || isLineStyle(s.atomStyle) || isTriStyle(s.atomStyle)) throw new StyleError(`Cannot set mass for atom style ${s.atomStyle}`);
        // As native LAMMPS does (src/atom.cpp set_mass at b332f28): every line names a type and a mass
        for (const l of raw) {
          const loc = 'Masses section of data file';
          const w = words(l);
          if (w.length !== 2) throw new StyleError(`Invalid format in ${loc}: ${l}`);
          const kind = isType(w[0]);
          let t: number;
          if (kind === 0) {
            t = inumeric(w[0]);
            if (t < 1 || t > s.ntypes) throw new StyleError(`Invalid atom type ${t} in ${loc}: ${l.trim()}`);
          } else if (kind === 1) {
            if (!sys.labels.defined) throw new StyleError(`Invalid atom type in ${loc}: ${l.trim()}`);
            const lt = sys.labels.typeOf('atom', w[0]);
            if (lt === undefined) throw new StyleError(`Unknown atom type ${w[0]} in ${loc}: ${l.trim()}`);
            t = lt;
          } else throw new StyleError(`Invalid ${loc}: ${l.trim()}`);
          t += kind === 0 ? toff : 0;
          const m = numeric(w[1]);
          if (t < 1 || t > s.ntypes) throw new StyleError(`Invalid atom type ${t} in ${loc}: ${l.trim()}`);
          if (!(m > 0)) throw new StyleError(`Invalid mass value ${shortest(m)} in ${loc}: ${l.trim()}`);
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
        // As native LAMMPS does (src/atom.cpp data_bonds, data_angles, data_dihedrals, data_impropers at b332f28):
        // blank and comment lines are skipped; a line is "ID type atom-IDs"; its atom IDs are distinct and at most the
        // largest atom ID read
        const loc = `${title} section of data file`;
        const maxTag = maxAtomId(s);
        raw.forEach((l) => {
          const w = words(l);
          if (!w.length) return;
          if (w.length !== 2 + width) throw new StyleError(`Incorrect format in ${loc}: ${l.trim()}${errorUrl(2)}`);
          const ids = w.slice(2).map((v) => inumeric(v) + (idBase || 0));
          const tk = isType(w[1]);
          let t: number;
          if (tk === 0) {
            t = inumeric(w[1]) + off;
            if (t < 1 || t > ntype) throw new StyleError(`Invalid ${lkind} type ${t} in ${loc}: ${l.trim()}`);
          } else {
            const lt = tk === 1 && sys.labels.defined ? sys.labels.typeOf(lkind, w[1]) : undefined;
            if (lt === undefined) throw new StyleError(`Invalid ${loc}: ${l.trim()}`);
            t = lt;
          }
          if (ids.some((id) => id <= 0 || id > maxTag) || new Set(ids).size !== ids.length) {
            throw new StyleError(`Invalid atom ID in ${loc}: ${l.trim()}`);
          }
          topo.push({ kind, type: t, ids });
        });
        break;
      }
      // read_data.html: the type-label sections give one label per numeric type (ID label); as native LAMMPS does
      // (src/read_data.cpp typelabels at b332f28), every type must end up with its own label
      case 'Atom Type Labels': case 'Bond Type Labels': case 'Angle Type Labels': case 'Dihedral Type Labels': case 'Improper Type Labels': {
        const lk = labelKind!;
        const name = title.slice(0, -' Type Labels'.length);
        const got = new Map<number, string>();
        for (const l of raw) {
          const w = words(l);
          if (w.length !== 2) throw new StyleError(`Invalid format in section: ${title}: ${l}`);
          if (isType(w[1]) !== 1) throw new StyleError(`Invalid type label ${w[1]}`);
          const t = inumeric(w[0]);
          // native formats this message with three placeholders and two values, and its formatter stops with this text
          if (t < 1 || t > count) throw new StyleError('argument not found');
          got.set(t, w[1]);
        }
        if (count > 0 && (got.size !== count || new Set(got.values()).size !== count)) throw new StyleError(`${name} Type Labels map is incomplete`);
        for (const [t, label] of got) sys.labels.set(lk, t, label);
        break;
      }
      default: {
        // As native LAMMPS does (src/read_data.cpp parse_coeffs and the Coeffs sections at b332f28): each line is a
        // coeff command, read as it comes; a blank line is refused, the type IDs are integers
        const emptyMsg: Record<string, string> = {
          'Pair Coeffs': `Unexpected empty line in PairCoeffs section. Expected ${count} lines.`,
          'PairIJ Coeffs': `Unexpected empty line in PairIJCoeffs section. Expected ${count} lines.`,
          'Bond Coeffs': `Unexpected empty line in BondCoeffs section. Expected ${count} lines.`,
        };
        const c2 = title in CLASS2_SECTIONS ? CLASS2_SECTIONS[title] : null;
        const kind: 'pair' | 'bond' | 'angle' | 'dihedral' | 'improper' = title === 'Pair Coeffs' || title === 'PairIJ Coeffs' ? 'pair'
          : c2 ? c2[0] : (title.split(' ')[0].toLowerCase() as 'bond' | 'angle' | 'dihedral' | 'improper');
        for (const l of raw) {
          const w = l.replace(/#.*/, '').trim() ? l.replace(/#.*/, '').trim().split(/\s+/) : [];
          if (!w.length) {
            throw new StyleError(emptyMsg[title] ?? `Unexpected empty line in ${kind[0].toUpperCase()}${kind.slice(1)}Coeffs section`);
          }
          let args: string[];
          if (kind === 'pair') {
            const i = String(inumeric(w[0]) + toff);
            const j = title === 'Pair Coeffs' ? i : String(inumeric(w[1]) + toff);
            args = [i, j, ...w.slice(title === 'Pair Coeffs' ? 1 : 2)];
          } else {
            const off = kind === 'bond' ? boff : kind === 'angle' ? aoff : kind === 'dihedral' ? doff : ioff;
            args = [String(inumeric(w[0]) + off), ...w.slice(1)];
            const add = c2?.[1];
            // the class 2 keyword goes after the ID, or after a hybrid sub-style name (a word that starts lower case)
            if (add) args.splice(args.length > 1 && /^[a-z]/.test(args[1]) ? 2 : 1, 0, add);
          }
          if (opts.nocoeff) continue;
          const st = kind === 'pair' ? sys.ff.pair : sys.ff[kind];
          st!.coeff(args, sys.styleContext());
          st!.coeffGiven = true;
        }
      }
    }
  }
  // native: the header's atoms need an Atoms section, and its topology and bonus counts their sections
  if (natoms > 0 && !sawAtoms) throw new StyleError('No valid atoms found in data file');
  for (const [key, sec] of [['bonds', 'Bonds'], ['angles', 'Angles'], ['dihedrals', 'Dihedrals'], ['impropers', 'Impropers'],
    ['ellipsoids', 'Ellipsoids'], ['lines', 'Lines'], ['triangles', 'Triangles'], ['bodies', 'Bodies']]) {
    if ((h[key] ?? 0) > 0 && !seen.has(sec)) throw new StyleError(`${sec} section for ${h[key]} ${key} not found`);
  }
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
  if (topo.length || ['Bonds', 'Angles', 'Dihedrals', 'Impropers'].some((x) => seen.has(x))) {
    const ids = new Set(Array.from(s.id.subarray(0, s.n)));
    // As native LAMMPS does (src/read_data.cpp bonds and friends at b332f28): an entry is stored with its first atom (a
    // bond) or its second (the others); one whose storing atom does not exist is not stored, and the count fails
    for (const kind of ['bonds', 'angles', 'dihedrals', 'impropers'] as const) {
      const of = topo.filter((e) => e.kind === kind);
      if (of.length !== (h[kind] ?? 0) || of.some((e) => !ids.has(e.ids[kind === 'bonds' ? 0 : 1]))) {
        throw new StyleError(`${kind[0].toUpperCase()}${kind.slice(1)} assigned incorrectly`);
      }
    }
    for (const e of topo) {
      // native keeps an entry whose other atoms are missing until a run looks them up; the engine needs them now
      for (const id of e.ids) if (!ids.has(id)) throw new StyleError(`data file: ${e.kind.slice(0, -1)} references atom ${id}, which does not exist (native LAMMPS stops at the first run for it)`);
      pushTopo(s.topo[e.kind] as TopoList, e.type, e.ids);
    }
    // a bond is stored with its first atom, so the bonds per atom of this file count the bonds that start at each atom.
    // Measured with native LAMMPS (black box, 6 atoms, 2 bonds starting at each of atoms 1, 2 and 3): a read after
    // create_box bond/types 2 with extra/bond/per/atom 0 or 1 stops with "Subsequent read data induced too many bonds
    // per atom", and with 2 it reads; the same read as the first (box-making) read is accepted and sets the storage.
    // As native LAMMPS does (src/read_data.cpp bonds at b332f28): the count is at the first atom, and also at the second
    // with newton_bond off; a first read stores "maxall += atom->extra_bond_per_atom" (atom->bond_per_atom), which
    // create_bonds checks
    const perAtom = new Map<number, number>();
    let most = 0;
    const count = (id: number): void => {
      const n = (perAtom.get(id) ?? 0) + 1;
      perAtom.set(id, n);
      if (n > most) most = n;
    };
    for (const e of topo) {
      if (e.kind !== 'bonds') continue;
      count(e.ids[0]);
      if (!sys.newtonBond) count(e.ids[1]);
    }
    if (subsequentRead) {
      if (most > sys.extraBondPerAtom) throw new StyleError('Subsequent read data induced too many bonds per atom');
    } else sys.extraBondPerAtom = most + (opts.extraBondPerAtom ?? 0);

    // atom_style oxdna: each bond sets the 5'/3' partners (atoms.ts setOxdnaPartners)
    setOxdnaPartners(s, topo.filter((e) => e.kind === 'bonds').map((e) => [e.ids[0], e.ids[1]] as const));
  }
  if (!adding) {
    // As native LAMMPS does (src/read_data.cpp ReadData::command and src/domain.cpp set_initial_box at b332f28):
    // set_initial_box (expandflag 1) runs on each pass over the file, and a file with a Bonds, Angles, Dihedrals or
    // Impropers section is read twice. Each call takes small = 1e-4 of the box length as it then is and moves a
    // shrink-wrapped (s) face out by it; the last small is the margin later shrink-wrapping keeps (Geometry.pad).
    const passes = ['Bonds', 'Angles', 'Dihedrals', 'Impropers'].some((x) => seen.has(x)) ? 2 : 1;
    const box = s.box;
    let small: [number, number, number] = [0, 0, 0];
    for (let pass = 0; pass < passes; pass++) {
      small = [0, 1, 2].map((d) => SHRINK_PAD * (box.hi[d] - box.lo[d])) as [number, number, number];
      for (let d = 0; d < 3; d++) {
        if (box.boundary[d][0] === 's') box.lo[d] -= small[d];
        if (box.boundary[d][1] === 's') box.hi[d] += small[d];
      }
    }
    sys.geom.update();
    sys.geom.pad = small;
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

/** The 'body' hook (src/engine-gpl/body/write_data_body.ts, loaded with atom_style body): the Bodies section of write_data. */
export interface BodyWriteHook {
  /** The lines of the Bodies section for the atoms of `order` that are bodies (record count > 0), in that order. */
  bodiesSection(s: SimState, order: ArrayLike<number>, bstyle: string): string[];
}

export const writeData = (sys: System, opts: WriteDataOptions): string => {
  const s = sys.state;
  const bodyStyle = parseBodyStyle(s.atomStyle)?.bstyle ?? null;
  const bodyRec = bodyStyle ? s.custom.get(BODY_RECORD)! : null;
  const isBodyAtom = (i: number): boolean => bodyRec !== null && bodyRec.data[bodyRec.cols * i] > 0;
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
  // As native LAMMPS does (src/write_data.cpp header at b332f28): "{} bodies" (atom->nbodies, the atoms with a body)
  if (bodyStyle) {
    let nb = 0;
    for (let i = 0; i < s.n; i++) if (isBodyAtom(i)) nb++;
    out.push(`${nb} bodies`);
  }
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
  // As native LAMMPS does (src/write_data.cpp at b332f28, "if (atom->mass)"): atom_style body has per-atom masses only
  if (s.atomStyle !== 'sphere' && s.atomStyle !== 'bpm/sphere' && s.atomStyle !== 'ellipsoid' && s.atomStyle !== 'peri' && !isLineStyle(s.atomStyle) && !isTriStyle(s.atomStyle) && !bodyStyle) {
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
        // As native LAMMPS does (src/atom_vec_body.cpp pack_data_pre at b332f28): bodyflag 1 for an atom with a body, and
        // the mass column is rmass
        case 'bodyflag': return isBodyAtom(i) ? '1' : '0';
        case 'mass': return shortest(s.rmass![i]);
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
  // As native LAMMPS does (src/write_data.cpp at b332f28): "if (natoms && atom->body_flag) bonus(BODY);" after the
  // ellipsoid, line and triangle sections, the bodies in local order (the 'body' hook)
  if (bodyStyle && s.n > 0) out.push('', 'Bodies', '', ...gplHook<BodyWriteHook>('body').bodiesSection(s, order, bodyStyle));
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
