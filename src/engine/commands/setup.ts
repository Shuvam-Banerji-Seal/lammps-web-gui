import type { Handler } from './args';
import { int, num, yesno, latticeScale, keywords, missingArgs, logical, numeric, inumeric, bnumeric, bounds, initializing } from './args';
import { StyleError } from '../force/types';
import { unknownStyleError } from '../nativeStyles';
import { errorUrl } from '../formula';
import { UNIT_SYSTEMS, isUnitStyle } from '../units';
import { makeBox, Geometry, cloneBox } from '../domain';
import { APIP_LAMBDA, isApipStyle, DPD_THETA, isDpdStyle, EDPD_CV, EDPD_TEMP, TDPD_CC, isEdpdStyle, tdpdSpecies, isSphStyle, SPH_CV, SPH_ESPH, SPH_RHO, DIEL_EPSILON, DIEL_QSCALED, isDielectricStyle, ELECTRON_RADIUS, ELECTRON_SPIN, electronProp, HEAT_TEMPERATURE, emptyState, appendAtoms, maxAtomId, pushTopo, ALL_GROUP_BIT, hasChargeStyle, hasDipoleStyle, setSpinVector, isMolecularStyle, sphereMass, SPHERE_DEFAULT_RADIUS, massOf, isEllipsoid, isBpmSphereStyle, isTriAtom, ellipsoidVolume, gatherAtoms, hasCharge, hasMolecule, nativeOrder, isTemplateStyle, templateStyleId, isBodyStyle, isSuperEllipsoidStyle, RHEO_RHO, RHEO_STATUS, isRheoStyle, isRheoThermalStyle, RHEO_ESPH, atomSubStyles, topologyLevel } from '../atoms';
import { SUPER_BLOCK, storeSuperInertia } from '../superellipsoid';
import { isSmdStyle, SMD_CONTACT } from '../smd';
import { parseStl } from '../stl';
import { typeWord, isNumericTypeWord, parseLabelPairs, labelmapText, isLabelKind, parseCheckWhich, improperSymmatoms, type LabelKind, type CheckWhich } from '../labelmap';
import { moleculeLabelCheck } from '../labelcheck';

/** group.html and set.html logical operators of the type / id / molecule styles (not type labels) */
const LOGICAL_OPS = ['<', '<=', '>', '>=', '==', '!=', '<>'];
import { isLatticeStyle, makeLattice, checkLatticeDimension, latticeSites } from '../lattice';
import { generalAtomSites, generalBoxFromRestricted, generalCreateBox } from '../triclinic_general';
import {
  BIG, BlockRegion, CompoundRegion, ConeRegion, EllipsoidRegion, PlaneRegion, PrismRegion, SphereRegion,
  type Param, type Region,
} from '../region';
import { readData, writeData, defaultReadOptions, DATA_SECTION_NAMES, type ReadDataOptions } from '../output/data';

/** The largest atom or molecule ID (native MAXTAGINT). */
const MAXTAGINT = 2147483647;
import { RanMars, RanPark, Rng } from '../rng';
import { groupMotionOf } from '../groupMotion';
import { nativeRandom } from '../styles';
import { gzipText } from '../gzip';
import { ComputeTemp } from '../compute/temp';
import type { AtomStyle, BoundaryStyle, SimState } from '../types';
import type { System } from '../system';
import type { Fix } from '../fix/fix';
import type { Compute } from '../compute/compute';
import { bitOfIndex, type DynamicGroup } from '../group';
import { parseBalanceWeight, applyBalanceWeights, balanceNeighFactor, balanceUsesNeigh, balanceDump, type BalanceWeightAction } from '../fix/balance';
import { bodyRadiusBody, parseBodyStyle, setBodyAtom } from '../bodies';
import { parseMoleculeFile, geometricCenter, orientationMatrix, templateDipoles, type MoleculeTemplate, type MoleculeOptions } from '../molecule';
import { libm } from '../libm';

/*
 * System setup commands. Each handler cites its docs.lammps.org page.
 */

/** As native LAMMPS does (src/input.cpp at b332f28): "Units command after simulation box is defined" and the like. */
const noBox = (sys: System, cmd: string) => {
  if (sys.hasBox) throw new StyleError(`${cmd[0].toUpperCase()}${cmd.slice(1)} command after simulation box is defined\nFor more information see https://docs.lammps.org/err0034`);
};

/** units style — units.html: "This command cannot be used after the simulation box is defined". */
const units: Handler = ({ sys }, a) => {
  if (a.length !== 1) throw new StyleError(`Illegal units command: expected 1 argument but found ${a.length}`);
  noBox(sys, 'units');
  if (!isUnitStyle(a[0])) throw new StyleError(`Unknown units style ${a[0]}`);
  sys.units = UNIT_SYSTEMS[a[0]];
  // "The units command also sets the timestep size and neighbor skin distance to default values"
  sys.nb.skin = sys.units.skin;
  sys.pendingDt = null;
};

/** dimension N — dimension.html (default 3). */
const dimension: Handler = ({ sys }, a) => {
  if (a.length !== 1) throw new StyleError('Dimension command expects exactly one argument');
  noBox(sys, 'dimension');
  const d = inumeric(a[0]);
  if (d !== 2 && d !== 3) throw new StyleError(`Invalid dimension argument: ${a[0]}`);
  sys.dimension = d;
};

/**
 * boundary x y z — boundary.html: "A single letter assigns the same style to
 * both the lower and upper face of the box. Two letters assigns the first
 * style to the lower face and the second style to the upper face." "The p
 * style must be applied to both faces of a dimension. For 2d simulations the
 * z dimension must be periodic".
 */
const boundary: Handler = ({ sys }, a) => {
  noBox(sys, 'boundary');
  // a 2d simulation's z is checked when the box is made (create_box, read_data), as native does
  sys.boundary = boundaryWords(a, 'boundary');
};

/**
 * The face styles of the boundary command and of change_box boundary, as native LAMMPS reads them (src/domain.cpp
 * Domain::set_boundary at b332f28): three words, of which only the first character (both faces) or the first two
 * (lower, upper face) are read; every face is checked before a dimension with one periodic face is refused.
 */
const boundaryWords = (a: string[], cmd: 'boundary' | 'change_box'): SimState['box']['boundary'] => {
  if (a.length !== 3) throw new StyleError(`Illegal boundary command: expected 3 arguments but found ${a.length}`);
  const ok = (c: string): c is BoundaryStyle => c === 'p' || c === 'f' || c === 's' || c === 'm';
  const b = a.map((w) => {
    const lo = w[0], hi = w.length === 1 ? w[0] : w[1];
    for (const c of [lo, hi]) if (!ok(c)) throw new StyleError(`Unknown ${cmd} keyword: ${c}`);
    return [lo, hi] as [BoundaryStyle, BoundaryStyle];
  });
  for (const [lo, hi] of b) if ((lo === 'p') !== (hi === 'p')) throw new StyleError('Both sides of boundary must be periodic');
  return b as SimState['box']['boundary'];
};

const ATOM_STYLES: string[] = ['atomic', 'charge', 'bond', 'angle', 'molecular', 'full', 'sphere', 'bpm/sphere', 'dipole', 'spin', 'ellipsoid', 'peri', 'template', 'body', 'oxdna', 'line', 'tri', 'electron', 'dielectric', 'sph', 'edpd', 'mdpd', 'tdpd', 'rheo', 'rheo/thermal', 'dpd', 'smd', 'amoeba', 'apip'];

/**
 * atom_style — atom_style.html: "The default atom style is atomic." and "*hybrid* args = list of one
 * or more sub-styles, each with their args"; "atoms store and communicate the union of all quantities
 * implied by the individual styles".
 */
const atomStyle: Handler = ({ sys }, a) => {
  if (a.length < 1) throw missingArgs('atom_style');
  noBox(sys, 'atom_style');
  const known = (w: string) => {
    if (!(ATOM_STYLES as string[]).includes(w)) {
      throw unknownStyleError('atom', w, () => new StyleError(`atom_style '${w}' is not supported by the browser engine; supported: ${ATOM_STYLES.join(', ')}, hybrid`));
    }
  };
  // atom_style.html: "*sphere* arg = 0/1 (optional) for static/dynamic particle radii"; 1 is kept as sys.radvary (the
  // engine's ghosts read their owners' radii, so it changes no numbers; fixes that change radii check it)
  sys.radvary = false;
  // atom_style.html: "tdpd Nspecies": the number of chemical species is a positive integer
  const tdpdArg = (w: string | undefined): string => {
    // As native LAMMPS does (src/DPD-MESO/atom_vec_tdpd.cpp process_args at b332f28): the count is required and read
    // by utils::inumeric; the engine needs at least one species
    if (w === undefined) throw new StyleError('Invalid atom_style tdpd command');
    const n = inumeric(w);
    if (n < 1) throw new StyleError('Invalid atom_style tdpd command');
    return `tdpd:${n}`;
  };
  // atom_style.html: "*template* arg = template-ID" and "template-ID = ID of molecule template specified in a separate"
  // (the molecule command follows); the ID is kept in the style string (template:ID, see atoms.ts).
  const templateArg = (id: string | undefined): string => {
    if (!id) throw new StyleError('atom_style template needs a molecule template-ID (defined by a molecule command)');
    if (!sys.molecules.has(id)) throw new StyleError(`atom_style template: molecule template ${id} does not exist (define it with the molecule command first)`);
    return `template:${id}`;
  };
  // native Atom::create_avec and AtomVecHybrid::process_args (src/atom.cpp, src/atom_vec_hybrid.cpp at b332f28): the
  // words after a style name, up to the next known style name, are that style's own args, and the style checks them
  const isStyleWord = (w: string) => (ATOM_STYLES as string[]).includes(w) || w === 'hybrid';
  const subStyle = (w: string, args: string[]): string => {
    if (w === 'tdpd') return tdpdArg(args[0]);
    if (w === 'template') {
      // native AtomVecTemplate::process_args (src/MOLECULE/atom_vec_template.cpp at b332f28): "if (narg != 1)"
      if (args.length !== 1) throw new StyleError('Illegal atom_style template command');
      return templateArg(args[0]);
    }
    if (w === 'sphere' || w === 'bpm/sphere') {
      // native AtomVecSphere::process_args (src/atom_vec_sphere.cpp) and AtomVecBPMSphere::process_args
      // (src/BPM/atom_vec_bpm_sphere.cpp): more than one arg is illegal; sphere reads its arg with utils::logical,
      // bpm/sphere with utils::inumeric (0 or 1)
      if (args.length > 1) throw new StyleError(`Illegal atom_style ${w} command`);
      if (args.length === 1) {
        if (w === 'sphere') sys.radvary = logical(args[0]);
        else {
          const v = inumeric(args[0]);
          if (v < 0 || v > 1) throw new StyleError('Illegal atom_style bpm/sphere command');
          sys.radvary = v === 1;
        }
      }
      return w;
    }
    if (w === 'ellipsoid') {
      // native AtomVecEllipsoid::process_args (src/ASPHERE/atom_vec_ellipsoid.cpp at b332f28): each arg must be superellipsoid
      for (const x of args) if (x !== 'superellipsoid') throw new StyleError(`Unknown atom_style ellipsoid argument: ${x}`);
      if (args.length) throw new StyleError('atom_style hybrid with ellipsoid superellipsoid is not supported by the browser engine');
      return w;
    }
    // native AtomVec::process_args (src/atom_vec.cpp at b332f28): a style with no args of its own refuses any
    if (args.length) throw new StyleError('Invalid atom_style command');
    return w;
  };
  if (a[0] === 'hybrid') {
    const subs: string[] = [];
    for (let k = 1; k < a.length; k++) {
      const w = a[k];
      if (w === 'hybrid') throw new StyleError('atom_style hybrid cannot have hybrid as a sub-style');
      if (w === 'body') throw new StyleError('atom_style hybrid with a body sub-style is not supported by the browser engine');
      // atom_style hybrid tri (gransurf/in.screwfeeder.local): the triangle particles come from fix surface/local, so the
      // tri sub-style is accepted; a line sub-style is still refused
      if (w === 'line') throw new StyleError(`atom_style hybrid with a ${w} sub-style is not supported by the browser engine (use atom_style ${w} alone)`);
      // atom_style hybrid with apip: the default (lambda thermostat) mode; the conservative mode is not supported here
      if (w === 'apip' && a[k + 1] === 'conservative') throw new StyleError('atom_style hybrid with apip conservative is not supported by the browser engine');
      if (w === 'apip' && a[k + 1] === 'thermostat') {
        if (subs.includes('apip')) throw new StyleError('atom_style hybrid lists apip twice');
        subs.push('apip');
        k++;
        continue;
      }
      known(w);
      // atom_style.html: "tdpd Nspecies" and "template-ID": a sub-style is listed once (tdpd and template keep their count or ID in the name)
      const listed = w === 'tdpd' ? subs.some((x) => x.startsWith('tdpd:')) : w === 'template' ? subs.some((x) => x.startsWith('template:')) : subs.includes(w);
      if (listed) throw new StyleError(`atom_style hybrid lists ${w} twice`);
      let end = k + 1;
      while (end < a.length && !isStyleWord(a[end])) end++;
      const args = a.slice(k + 1, end);
      k = end - 1;
      subs.push(subStyle(w, args));
    }
    if (!subs.length) throw new StyleError('usage: atom_style hybrid sub-style1 sub-style2 ...');
    // atom_style.html: "When using the *hybrid* style, you cannot combine the *template* style"
    // with another molecular style that stores bond, angle, etc info on a per-atom basis
    if (subs.some((x) => x.startsWith('template:')) && subs.some((x) => ['bond', 'angle', 'molecular', 'full'].includes(x))) {
      throw new StyleError('atom_style hybrid: you cannot combine the template style with a bond, angle, molecular or full sub-style');
    }
    sys.atomStyle = `hybrid ${subs.join(' ')}`;
    return;
  }
  // atom_style.html: "body args = bstyle bstyle-args"; the engine implements the nparticle, rounded/polygon and
  // rounded/polyhedron body styles (Howto_body.rst: "atom_style body rounded/polyhedron Nmin Nmax"). Nmin and Nmax only
  // size the internal arrays (bodies.ts).
  if (a[0] === 'body') {
    if (a[1] !== 'nparticle' && a[1] !== 'rounded/polygon' && a[1] !== 'rounded/polyhedron') throw new StyleError(`atom_style body ${a[1] ?? ''} is not supported by the browser engine; only atom_style body nparticle, rounded/polygon and rounded/polyhedron Nmin Nmax are`);
    // As native LAMMPS does (src/BODY/body_nparticle.cpp, body_rounded_polygon.cpp and body_rounded_polyhedron.cpp at
    // b332f28): the body args include the style name, so exactly two integers (utils::inumeric) follow it; the
    // polyhedron copy of that count check also says "rounded/polygon"
    if (a.length !== 4) throw new StyleError(`Invalid body ${a[1] === 'nparticle' ? 'nparticle' : 'rounded/polygon'} command`);
    const nmin = inumeric(a[2]), nmax = inumeric(a[3]);
    if (nmin <= 0 || nmin > nmax) throw new StyleError(`Invalid body ${a[1]} command`);
    sys.atomStyle = `body ${a[1]} ${nmin} ${nmax}`;
    return;
  }
  // atom_style.html: "*apip* arg = *conservative*/*thermostat* (optional) for conservative APIP/lambda thermostat APIP"
  // and "If atom_style *apip* is used, its default argument is 'thermostat'." The messages are native
  // AtomVecApip::process_args (src/APIP/atom_vec_apip.cpp at b332f28).
  if (a[0] === 'apip') {
    if (a.length > 2) throw new StyleError('Invalid atom_style apip command');
    if (a.length === 2 && a[1] !== 'thermostat' && a[1] !== 'conservative') throw new StyleError(`unknown argument ${a[1]}`);
    sys.atomStyle = a[1] === 'conservative' ? 'apip conservative' : 'apip';
    return;
  }
  known(a[0]);
  // atom_style.html: "*ellipsoid* arg = superellipsoid (optional) for superellipsoids instead of ellipsoids"
  // native AtomVecEllipsoid::process_args (src/ASPHERE/atom_vec_ellipsoid.cpp at b332f28): every arg must be superellipsoid
  if (a[0] === 'ellipsoid' && a.length > 1) {
    for (const x of a.slice(1)) if (x !== 'superellipsoid') throw new StyleError(`Unknown atom_style ellipsoid argument: ${x}`);
    sys.atomStyle = 'ellipsoid superellipsoid';
    return;
  }
  // atom_style.html: "tdpd args = Nspecies" with Nspecies the number of chemical species; the count is kept in the style
  // string (tdpd:N, atoms.ts tdpdSpecies). Native AtomVecTDPD::process_args reads the first arg and ignores the rest.
  sys.atomStyle = subStyle(a[0], a.slice(1)) as AtomStyle;
};

/**
 * atom_modify id/map/first/sort — atom_modify.html; bookkeeping only in a single-process engine. The keywords are read
 * in order with native's checks and messages, as native LAMMPS does (src/atom.cpp Atom::modify_params at b332f28).
 */
const atomModify: Handler = ({ sys }, a) => {
  if (a.length === 0) throw missingArgs('atom_modify');
  // native error->all(..., utils::errorurl(34)) for id and map after the box exists
  const afterBox = (key: string) => new StyleError(`Atom_modify ${key} command after simulation box is defined\nFor more information see https://docs.lammps.org/err0034`);
  for (let k = 0; k < a.length;) {
    const key = a[k];
    if (key === 'id') {
      if (k + 2 > a.length) throw missingArgs('atom_modify id');
      if (sys.hasBox) throw afterBox('id');
      if (!logical(a[k + 1])) {
        // atom_modify.html: "The *id* keyword allows atom IDs to be assigned or not assigned to atoms." A molecular
        // atom style always builds the atom map, which native refuses without IDs when the box is created (atom.cpp
        // map_style_set: Cannot create an atom map unless atoms have IDs). The engine keeps an ID on every atom.
        if (isMolecularStyle(sys.atomStyle)) throw new StyleError('Cannot create an atom map unless atoms have IDs');
        throw new StyleError('atom_modify id no is not supported by the browser engine (atoms always have IDs)');
      }
      k += 2;
    } else if (key === 'map') {
      if (k + 2 > a.length) throw missingArgs('atom_modify map');
      if (sys.hasBox) throw afterBox('map');
      if (!['array', 'hash', 'yes'].includes(a[k + 1])) throw new StyleError(`Illegal atom_modify map command argument ${a[k + 1]}`);
      sys.atomMapSet = true;
      k += 2;
    } else if (key === 'first') {
      if (k + 2 > a.length) throw missingArgs('atom_modify first');
      // atom_modify.html: "*first* value = group-ID = group whose atoms will appear first in internal atom lists" — an
      // ordering for performance; the engine keeps its own storage order (as it does for the spatial *sort*), so say
      // so instead of reordering. "Since sorting is on by default, it will be turned off if the first keyword is used
      // with a group-ID that is not “all”." Measured with native LAMMPS (black box): with a first group, dumps before
      // and after run 0 list the atoms in the same order
      const g = a[k + 1];
      sys.log(`atom_modify first ${g}: accepted; the browser engine does not reorder its atom storage (results are unchanged, only the internal order differs)`);
      if (g === 'all') sys.firstGroup = null;
      else {
        sys.firstGroup = g;
        sys.sortEvery = 0;
      }
      k += 2;
    } else if (key === 'sort') {
      // atom_modify.html: "*sort* values = Nfreq binsize"; the order matters for per-atom random draws and unsorted
      // output (SimState.order, System.sortAtoms). "The first and sort options cannot be used together."
      if (k + 3 > a.length) throw missingArgs('atom_modify sort');
      const nfreq = inumeric(a[k + 1]);
      const binsize = numeric(a[k + 2]);
      if (nfreq < 0) throw new StyleError(`Illegal atom_modify sort frequency ${nfreq}`);
      if (binsize < 0) throw new StyleError(`Illegal atom_modify sort bin size ${binsize}`);
      if (sys.firstGroup !== null) throw new StyleError('Atom_modify sort and first options cannot be used together');
      sys.sortEvery = nfreq;
      sys.sortBinsize = binsize;
      k += 3;
    } else throw new StyleError(`Illegal atom_modify command argument: ${key}`);
  }
};

/** Single-process / no-accelerator commands: accepted, with what they mean here. */
const parallelOnly = (name: string): Handler => (({ sys }, a) => {
  // suffix.html: "style = *off* or *on* or *gpu* or *intel* or *kk* or *omp* or *opt* or *hybrid*"; the
  // accelerator names are aliases here (accelAliases.ts), so a suffix changes no style and the words are only checked
  if (name === 'suffix') {
    // As native LAMMPS does (src/input.cpp Input::suffix at b332f28): off/no/false turns suffixes off, on/yes/true turns
    // them on again (only after one was defined), any other word defines the suffix (hybrid takes two)
    if (a.length < 1) throw missingArgs('suffix');
    const style = a[0];
    if (style === 'on' || style === 'yes' || style === 'true') {
      if (sys.suffix === null) throw new StyleError('May only enable suffixes after defining one');
    } else if (style !== 'off' && style !== 'no' && style !== 'false') {
      if (style === 'hybrid') {
        if (a.length !== 3) throw new StyleError('Illegal suffix hybrid command');
      } else if (a.length !== 1) throw new StyleError('Illegal suffix command');
      sys.suffix = style === 'hybrid' ? a.slice(1).join(' ') : style;
    }
    sys.log(`suffix ${a.join(' ')}: accepted; styles named later run as their base styles (the browser engine picks CPU threads or the GPU itself)`);
    return;
  }
  // package.html: "style = *gpu* or *intel* or *kokkos* or *omp*"; the package settings do not change the engine
  if (name === 'package') {
    // As native LAMMPS does (src/input.cpp Input::package at b332f28): before the box, then the package name
    if (sys.hasBox) throw new StyleError(`Package command after simulation box is defined${errorUrl(34)}`);
    if (a.length < 1) throw missingArgs('package');
    const packages = ['gpu', 'intel', 'kokkos', 'omp'];
    if (!packages.includes(a[0])) throw new StyleError(`Unknown package keyword: ${a[0]}`);
    if (a[0] === 'omp') {
      // native FixOMP constructor (src/OPENMP/fix_omp.cpp at b332f28), which Input::package (src/input.cpp) reaches with the
      // words after omp: after the thread count only the neigh keyword is known, and it takes one value. As native
      // FixOMP::FixOMP does: "if (narg < 4) utils::missing_cmd_args(FLERR, \"package omp\", error)", and the fix line
      // is "package_omp all OMP" plus these words, so the thread count is required. The count is 0 (all threads) or
      // utils::inumeric, at least 1; neigh takes utils::logical
      if (a.length < 2) throw missingArgs('package omp');
      if (a[1] !== '0' && inumeric(a[1]) < 1) throw new StyleError('Illegal number of OpenMP threads requested');
      for (let k = 2; k < a.length;) {
        if (a[k] !== 'neigh') throw new StyleError(`Unknown package omp keyword ${a[k]}`);
        if (k + 2 > a.length) throw missingArgs('package omp neigh');
        logical(a[k + 1]);
        k += 2;
      }
    }
    sys.log(`package ${a.join(' ')}: accepted; the browser engine chooses CPU threads or the GPU in the notebook toolbar, so the settings have no effect`);
    return;
  }
  sys.log(`${name}: accepted (the browser engine runs one process; this setting has no effect)`);
}) as Handler;

/**
 * processors Px Py Pz [keyword args] — processors.html. As native LAMMPS does (src/input.cpp Input::processors and
 * src/comm.cpp Comm::set_processors at b332f28): before the box only, and the grid (each count or *) must fit the one
 * process the engine runs; the keywords choose how processes are mapped and have no effect here.
 */
const processors: Handler = ({ sys }, a) => {
  if (sys.hasBox) throw new StyleError('Processors command after simulation box is defined\nFor more information see https://docs.lammps.org/err0034');
  if (a.length < 3) throw new StyleError('Illegal processors command');
  // native Comm::set_processors (src/comm.cpp at b332f28): the three grid counts, then keywords whose values are checked in
  // order; a keyword that is not known is "Illegal processors command". The keywords have no effect in one process.
  const illegal = () => new StyleError('Illegal processors command');
  const p = a.slice(0, 3).map((w) => (w === '*' ? 0 : inumeric(w)));
  if (p.some((v) => v < 0)) throw illegal();
  if (p[0] * p[1] * p[2] !== 0 && p[0] * p[1] * p[2] !== 1) throw new StyleError('Specified processors != physical processors');
  const mapStyles = ['cart', 'cart/reorder', 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'];
  let gridflag = 'onelevel';
  let mapflag = 'cart';
  for (let k = 3; k < a.length;) {
    const kw = a[k];
    if (kw === 'grid') {
      if (k + 2 > a.length) throw illegal();
      const g = a[k + 1];
      if (g === 'twolevel') {
        // native: ncores and the three core-grid counts (* or a number) follow
        if (k + 6 > a.length) throw illegal();
        const ncores = inumeric(a[k + 2]);
        const cg = [a[k + 3], a[k + 4], a[k + 5]].map((w) => (w === '*' ? 0 : inumeric(w)));
        if (ncores <= 0 || cg.some((v) => v < 0)) throw illegal();
        k += 4;
      } else if (g === 'custom') {
        if (k + 3 > a.length) throw illegal();
        k += 1;
      } else if (g !== 'onelevel' && g !== 'numa') {
        throw illegal();
      }
      gridflag = g;
      k += 2;
    } else if (kw === 'map') {
      if (k + 2 > a.length || !mapStyles.includes(a[k + 1])) throw illegal();
      mapflag = a[k + 1];
      k += 2;
    } else if (kw === 'part') {
      if (k + 4 > a.length) throw illegal();
      // one process is one partition, and native refuses the part keyword without partitions
      throw new StyleError('Cannot use processors part command without using partitions');
    } else if (kw === 'file') {
      if (k + 2 > a.length) throw illegal();
      k += 2;
    } else if (kw === 'numa_nodes') {
      if (k + 2 > a.length) throw illegal();
      inumeric(a[k + 1]);
      k += 2;
    } else {
      throw illegal();
    }
  }
  if (gridflag === 'numa' && mapflag !== 'cart') throw new StyleError('Processors grid numa and map style are incompatible');
  sys.log('processors: accepted (the browser engine runs one process; this setting has no effect)');
};

/** comm_style brick|tiled — comm_style.html; recorded for fix balance rcb, otherwise no effect in one process. */
const commStyle: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/input.cpp Input::comm_style at b332f28)
  if (a.length < 1) throw missingArgs('comm_style');
  if (a[0] !== 'brick' && a[0] !== 'tiled') throw new StyleError(`Unknown comm_style argument: ${a[0]}`);
  sys.commStyle = a[0];
  sys.log(`comm_style ${a[0]}: accepted (the browser engine runs one process)`);
};

/*
 * balance thresh style args ... keyword args ... — docs.lammps.org/balance.html:
 * static load balancing. The engine is one process, so the subdomains are
 * always the whole box and no atom moves; the only observable effect is the
 * *weight* processing, which can compute and, with *store*, write the custom
 * d_name vector (the same weighted-balance options as fix balance, see
 * fix/balance.ts). style = x/y/z (a "grid" method, combinable) or shift or
 * rcb; "Balancing through recursive bisectioning (rcb style) requires
 * comm_style tiled."
 */
const balanceCommand: Handler = ({ sys }, a) => {
  if (!sys.hasBox) throw new StyleError('balance requires a simulation box (create_box or read_data first)');
  // As native LAMMPS does (src/balance.cpp Balance::command at b332f28): two words at least, then thresh (utils::numeric)
  if (a.length < 2) throw missingArgs('balance');
  const thresh = numeric(a[0]);
  let k = 1;
  const style = a[k];
  if (style === 'shift') {
    if (k + 4 > a.length) throw missingArgs('balance shift');
    const dims = a[k + 1];
    if (!dims || !/^[xyz]{1,3}$/.test(dims) || new Set(dims).size !== dims.length) throw new StyleError(`balance shift: dimstr '${dims ?? ''}' must contain x, y or z, each at most once`);
    if (sys.dimension === 2 && dims.includes('z')) throw new StyleError('balance shift: cannot balance in z for a 2d simulation');
    // native: utils::inumeric Niter > 0 ("Illegal balance shift value N") and utils::numeric stopthresh >= 1
    if (inumeric(a[k + 2]) <= 0) throw new StyleError(`Illegal balance shift value ${a[k + 2]}`);
    if (numeric(a[k + 3]) < 1) throw new StyleError(`Illegal balance stop threshold value ${a[k + 3]}`);
    k += 4;
  } else if (style === 'rcb') {
    if (sys.commStyle !== 'tiled') throw new StyleError('balance rcb requires comm_style tiled');
    k += 1;
  } else if (style === 'x' || style === 'y' || style === 'z') {
    while (a[k] === 'x' || a[k] === 'y' || a[k] === 'z') {
      const dim = a[k++];
      if (sys.dimension === 2 && dim === 'z') throw new StyleError('balance: cannot balance in z for a 2d simulation');
      if (a[k] === 'uniform') k += 1;
      else {
        const start = k;
        while (k < a.length && a[k] !== '' && Number.isFinite(Number(a[k]))) k += 1;
        if (k === start) throw new StyleError(`balance ${dim}: expected uniform or cut positions`);
      }
    }
  } else throw new StyleError(`balance: unknown style '${style ?? ''}' (x, y, z, shift or rcb)`);
  const actions: BalanceWeightAction[] = [];
  let outFile: string | null = null;
  while (k < a.length) {
    const key = a[k];
    if (key === 'weight') k = parseBalanceWeight(sys, a, k, actions);
    else if (key === 'sort') {
      // Balance::options (src/balance.cpp at b332f28): utils::logical
      if (k + 2 > a.length) throw missingArgs('(fix) balance sort');
      logical(a[k + 1]);
      k += 2;
    } else if (key === 'out') {
      // As native LAMMPS does (src/balance.cpp Balance::command at b332f28): "if (iarg+2 > narg) utils::missing_cmd_args(FLERR,"(fix) balance out", error)"
      if (a[k + 1] === undefined) throw new StyleError('Illegal (fix) balance out command: missing argument(s)');
      outFile = a[k + 1];
      k += 2;
    } else throw new StyleError(`Unknown (fix) balance keyword ${key}`);
  }
  applyBalanceWeights(sys, actions, balanceUsesNeigh(actions) ? balanceNeighFactor(sys) : null);
  // As native LAMMPS does (src/balance.cpp at b332f28): the out file is opened by the options, and the result is
  // written unless the imbalance (1 on the engine's one process) is below the threshold ("imbinit < thresh) return")
  if (outFile !== null) {
    sys.writeFile(outFile, '', false);
    if (!(1 < thresh)) sys.writeFile(outFile, balanceDump(sys, sys.state.step), true);
  }
  sys.log('balance: accepted (the browser engine runs one process; subdomains are unchanged; stored weights are written)');
};

/** newton on/off — newton.html; forces are always summed once per pair here, so either setting gives the same result. */
const newton: Handler = ({ sys }, a) => {
  // newton.html: "flag = on or off for both pairwise and bonded interactions"; "flag2 = on or off for bonded
  // interactions". As native LAMMPS does (src/input.cpp Input::newton at b332f28): the bonded setting cannot change
  // once the box exists
  if (a.length < 1 || a.length > 2) throw new StyleError('Illegal newton command');
  const pair = logical(a[0]);
  const bond = a.length === 2 ? logical(a[1]) : pair;
  if (sys.hasBox && bond !== sys.newtonBond) throw new StyleError('Newton bond change after simulation box is defined\nFor more information see https://docs.lammps.org/err0034');
  sys.newtonPair = pair;
  sys.newtonBond = bond;
  sys.log('newton: accepted (the engine always uses Newton\'s 3rd law within one process; results are the same)');
};

/**
 * comm_modify keyword value ... — comm_modify.html: "keyword = *mode* or *cutoff* or *cutoff/multi* or *group* or
 * *reduce/multi* or *vel*". The cutoffs set the ghost cutoff; a single process makes the ghosts of the largest one. The
 * keywords are read in order with native's checks and messages, as native LAMMPS does (src/comm.cpp
 * Comm::modify_params at b332f28).
 */
const commModify: Handler = ({ sys }, a) => {
  if (a.length < 1) throw missingArgs('comm_modify');
  const nb = sys.nb;
  for (let k = 0; k < a.length;) {
    const key = a[k];
    if (key === 'mode') {
      if (k + 2 > a.length) throw missingArgs('comm_modify mode');
      const m = a[k + 1];
      // "The *multi* communication mode is only compatible with the *multi* neighbor style."; switching the mode
      // resets the cutoff set by the other mode
      if (m === 'single') {
        if (nb.commMulti) nb.commCutoff = 0;
        nb.commMulti = false;
      } else if (m === 'multi') {
        if (!nb.multiStyle) throw new StyleError("Cannot use comm mode 'multi' without 'multi' style neighbor lists");
        if (!nb.commMulti) nb.commCutoff = 0;
        nb.commMulti = true;
      } else if (m === 'multi/old') {
        throw new StyleError("Communication mode 'multi/old' has been removed. Please use mode 'multi' and see the documentation more information about it.");
      } else throw new StyleError(`Unknown comm_modify mode argument: ${m}`);
      k += 2;
    } else if (key === 'group') {
      if (k + 2 > a.length) throw missingArgs('comm_modify group');
      const g = a[k + 1];
      if (sys.groups.find(g) < 0) throw new StyleError(`Invalid comm_modify keyword: group ${g} not found`);
      if (g !== 'all' && sys.firstGroup !== g) {
        // native formats the first group's name even when atom_modify first was not used: fmt then stops on the null
        // name with its own message (measured with native LAMMPS: string pointer is null)
        throw new StyleError(sys.firstGroup === null ? 'string pointer is null' : `Comm_modify group != atom_modify first group: ${sys.firstGroup}`);
      }
      // only the atoms of the group get ghost atoms (Neighbor.borderBit)
      nb.borderBit = g === 'all' ? 0 : sys.groupBit(g);
      nb.lastBuild = -1;
      k += 2;
    } else if (key === 'cutoff') {
      if (k + 2 > a.length) throw missingArgs('comm_modify cutoff');
      if (nb.commMulti) throw new StyleError('Use cutoff/multi keyword to set cutoff in multi mode');
      const c = numeric(a[k + 1]);
      if (c < 0) throw new StyleError(`Invalid cutoff ${a[k + 1]} in comm_modify command`);
      nb.commCutoff = c;
      k += 2;
    } else if (key === 'cutoff/multi') {
      if (!nb.commMulti) throw new StyleError('Use cutoff keyword to set cutoff in single mode');
      if (!sys.hasBox) throw new StyleError('Cannot set cutoff/multi before simulation box is defined');
      if (k + 3 > a.length) throw missingArgs('comm_modify cutoff/multi');
      // "Collections are indexed from 1 to N where N is the total number of collections" (one per atom type unless
      // neigh_modify collection/type or collection/interval defines them)
      bounds(a[k + 1], 1, nb.ncollections || sys.state.ntypes);
      const c = numeric(a[k + 2]);
      nb.commCutoff = Math.max(nb.commCutoff, c);
      if (c < 0) throw new StyleError(`Invalid cutoff ${a[k + 2]} in comm_modify command`);
      k += 3;
    } else if (key === 'reduce/multi') {
      if (!nb.commMulti) throw new StyleError('Use reduce/multi in mode multi only');
      k += 1;
    } else if (key === 'vel') {
      if (k + 2 > a.length) throw missingArgs('comm_modify vel');
      sys.ghostVelocity = logical(a[k + 1]);
      k += 2;
    } else throw new StyleError(`Unknown comm_modify keyword: ${key}`);
  }
};

/** lattice style scale [keywords] — lattice.html. */
const lattice: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/lattice.cpp Lattice::Lattice at b332f28): the style, its dimension, then the scale
  if (a.length < 1) throw missingArgs('lattice');
  if (!isLatticeStyle(a[0])) throw new StyleError(`Unknown lattice keyword: ${a[0]}`);
  if (a[0] === 'none' && a.length !== 2) throw new StyleError(`Illegal lattice command: expected 2 arguments but found ${a.length}`);
  checkLatticeDimension(a[0], sys.dimension);
  if (a.length < 2) throw missingArgs('lattice');
  const scale = numeric(a[1]);
  if (scale <= 0) throw new StyleError(`Invalid lattice ${a[0]} argument: ${a[1]}`);
  sys.lattice = makeLattice(a[0], scale, sys.units, sys.dimension, a.slice(2));
  const sp = sys.lattice.spacing.map((v) => Number(v.toPrecision(8))).join(' ');
  sys.log(`Lattice spacing in x,y,z = ${sp}`);
};

/** A region bound: number, INF, EDGE, or v_name; scaled by the lattice spacing. */
const bound = (sys: System, w: string | undefined, scale: number, d: number, side: 'lo' | 'hi'): Param => {
  if (w === undefined) throw new StyleError('region: missing bound');
  if (w === 'INF' || w === 'EDGE') {
    // As native LAMMPS does (src/region_block.cpp, region_cylinder.cpp, region_prism.cpp at b332f28)
    if (!sys.hasBox) throw new StyleError('Cannot use region INF or EDGE when box does not exist');
    if (w === 'INF') return side === 'lo' ? -BIG : BIG;
    return side === 'lo' ? sys.state.box.lo[d] : sys.state.box.hi[d];
  }
  if (w.startsWith('v_')) return { variable: w.slice(2), scale };
  return numeric(w) * scale;
};

/** region ID style args [side in|out] [units lattice|box] [move ...] [rotate ...] [open N] — region.html. */
const region: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/domain.cpp Domain::add_region, src/region.cpp Region::options and Region::init, and the
  // region styles' constructors at b332f28): the ID, the style and its argument count, the keywords, then the
  // style's own values
  if (a.length < 2) throw missingArgs('region');
  const [id, style] = a;
  if (style === 'delete') {
    if (!sys.regions.delete(id)) throw new StyleError(`Delete region ${id} does not exist`);
    return;
  }
  if (style === 'none') throw new StyleError("Unrecognized region style 'none'");
  if (sys.regions.has(id)) throw new StyleError(`Reuse of region ID ${id}`);
  const nargs: Record<string, number> = { block: 6, sphere: 4, cylinder: 6, cone: 7, ellipsoid: 6, plane: 6, prism: 9 };
  let n: number;
  if (style === 'union' || style === 'intersect') {
    if (a.length < 5) throw missingArgs(`region ${style}`);
    const nreg = inumeric(a[2]);
    if (nreg < 2) throw new StyleError(`Illegal region ${style} n: ${nreg}`);
    n = 1 + nreg;
  } else if (Object.prototype.hasOwnProperty.call(nargs, style)) n = nargs[style];
  else throw new StyleError(`Unrecognized region style '${style}'`);
  if (a.length < 2 + n) throw missingArgs('region');
  const args = a.slice(2, 2 + n);
  const rest = a.slice(2 + n);
  // keywords first: units affects scaling
  let side = true;
  let unitsW = 'lattice';
  let move: [string | null, string | null, string | null] | null = null;
  let rotate: Region['rotate'] = null;
  const open: number[] = [];
  for (let k = 0; k < rest.length;) {
    const key = rest[k];
    if (key === 'side') {
      if (k + 2 > rest.length) throw missingArgs('region side');
      if (rest[k + 1] !== 'in' && rest[k + 1] !== 'out') throw new StyleError(`Unknown region side setting: ${rest[k + 1]}`);
      side = rest[k + 1] === 'in';
      k += 2;
    } else if (key === 'units') {
      if (k + 2 > rest.length) throw missingArgs('region units');
      if (rest[k + 1] !== 'lattice' && rest[k + 1] !== 'box') throw new StyleError(`Unknown region units: ${rest[k + 1]}`);
      unitsW = rest[k + 1];
      k += 2;
    } else if (key === 'move') {
      if (k + 4 > rest.length) throw missingArgs('region move');
      move = [0, 1, 2].map((d) => {
        const w = rest[k + 1 + d];
        if (w === 'NULL') return null;
        if (!w.startsWith('v_')) throw new StyleError(`Illegal region move ${'xyz'[d]} displacement variable: ${w}`);
        return w.slice(2);
      }) as [string | null, string | null, string | null];
      k += 4;
    } else if (key === 'rotate') {
      if (k + 8 > rest.length) throw missingArgs('region rotate');
      const th = rest[k + 1];
      if (!th.startsWith('v_')) throw new StyleError('Illegal region command');
      const p = [numeric(rest[k + 2]), numeric(rest[k + 3]), numeric(rest[k + 4])];
      const r = [numeric(rest[k + 5]), numeric(rest[k + 6]), numeric(rest[k + 7])];
      const len = Math.hypot(r[0], r[1], r[2]);
      if (!(len > 0)) throw new StyleError('region rotate axis must be non-zero');
      rotate = { theta: th.slice(2), p: p as [number, number, number], r: r.map((x) => x / len) as [number, number, number] };
      k += 8;
    } else if (key === 'open') {
      if (k + 2 > rest.length) throw missingArgs('region open');
      const face = inumeric(rest[k + 1]);
      if (face < 1 || face > 6) throw new StyleError(`Illegal region open face index: ${face}`);
      open.push(face);
      k += 2;
    } else throw new StyleError(`Unknown region command argument: ${key}`);
  }
  if ((style === 'union' || style === 'intersect') && (move || rotate)) throw new StyleError('Region union or intersect cannot be dynamic');
  // Region::init at creation: the move and rotate variables must be equal-style
  for (const [k, v] of [move?.[0], move?.[1], move?.[2], rotate?.theta].entries()) {
    if (!v) continue;
    const vv = sys.vars.get(v);
    if (!vv) throw new StyleError(`Variable ${v} for region ${style} ${id} does not exist`);
    if (vv.style !== 'equal' && vv.style !== 'timer' && vv.style !== 'internal') {
      throw new StyleError(`Variable ${v} for region ${style} ${id} ${k === 0 ? 'is invalid style' : 'is not equal style'}`);
    }
  }
  const sc = latticeScale(sys, unitsW, 'region');
  if (rotate) rotate.p = rotate.p.map((v, d) => v * sc[d]) as [number, number, number];
  const env = sys.regionEnv;
  let reg: Region;
  const axis = (w: string) => {
    const d = ['x', 'y', 'z'].indexOf(w);
    if (d < 0) throw new StyleError(`Illegal region ${style} axis: ${w}`);
    return d as 0 | 1 | 2;
  };
  const p = (w: string, s: number) => (w.startsWith('v_') ? { variable: w.slice(2), scale: s } : numeric(w) * s);
  // a value known now (a number, not a variable)
  const known = (v: Param): v is number => typeof v === 'number';
  const XYZ = ['x', 'y', 'z'];
  switch (style) {
    case 'block': case 'prism': {
      const b = [0, 1, 2].flatMap((d) => [bound(sys, args[2 * d], sc[d], d, 'lo'), bound(sys, args[2 * d + 1], sc[d], d, 'hi')]);
      // block refuses lo > hi, prism lo >= hi (both say >=)
      for (let d = 0; d < 3; d++) {
        const lo = b[2 * d], hi = b[2 * d + 1];
        if (known(lo) && known(hi) && (style === 'block' ? lo > hi : lo >= hi)) {
          throw new StyleError(`Illegal region ${style} ${XYZ[d]}lo: ${lo} >= ${XYZ[d]}hi: ${hi}`);
        }
      }
      if (style === 'block') { reg = new BlockRegion(id, env, b); break; }
      // "The lattice spacing in dimension x is applied to xy and xz, and the spacing in dimension y to yz."
      b.push(p(args[6], sc[0]), p(args[7], sc[0]), p(args[8], sc[1]));
      reg = new PrismRegion(id, env, b);
      break;
    }
    case 'sphere': {
      const r = p(args[3], sc[0]);
      if (known(r) && r < 0) throw new StyleError(`Illegal region sphere radius: ${r}`);
      reg = new SphereRegion(id, env, [0, 1, 2].map((d) => p(args[d], sc[d])), r);
      break;
    }
    case 'ellipsoid': {
      const ax = [0, 1, 2].map((d) => p(args[3 + d], sc[d]));
      ax.forEach((v, d) => { if (known(v) && v < 0) throw new StyleError(`Illegal region ellipsoid ${'abc'[d]}: ${v}`); });
      reg = new EllipsoidRegion(id, env, [0, 1, 2].map((d) => p(args[d], sc[d])), ax);
      break;
    }
    case 'cylinder': case 'cone': {
      // a cylinder has 3 faces (the open keyword takes 1 to 3)
      if (style === 'cylinder') for (const f of open) if (f > 3) throw new StyleError(`Illegal region cylinder open face: ${f}`);
      const ax = axis(args[0]);
      const [d1, d2] = ax === 0 ? [1, 2] : ax === 1 ? [0, 2] : [0, 1];
      const r1 = p(args[3], sc[d1]);
      const r2 = style === 'cone' ? p(args[4], sc[d1]) : r1;
      const lo = style === 'cone' ? args[5] : args[4], hi = style === 'cone' ? args[6] : args[5];
      const blo = bound(sys, lo, sc[ax], ax, 'lo'), bhi = bound(sys, hi, sc[ax], ax, 'hi');
      if (style === 'cylinder') {
        if (known(r1) && r1 <= 0) throw new StyleError(`Illegal radius ${r1} in region cylinder command`);
      } else {
        if (known(r1) && r1 < 0) throw new StyleError('Illegal lower radius in region cone command');
        if (known(r2) && r2 < 0) throw new StyleError('Illegal upper radius in region cone command');
        if (known(r1) && known(r2) && r1 === 0 && r2 === 0) throw new StyleError('Illegal radius in region cone command');
        if (known(blo) && known(bhi) && bhi <= blo) throw new StyleError('Illegal cone length in region cone command');
      }
      reg = new ConeRegion(id, style, env, ax, p(args[1], sc[d1]), p(args[2], sc[d2]), r1, r2, blo, bhi);
      break;
    }
    case 'plane': {
      const nrm = [0, 1, 2].map((d) => p(args[3 + d], sc[d]));
      if (nrm.every(known) && nrm.every((v) => v === 0)) throw new StyleError(`Illegal region plane normal vector: ${nrm.join(' ')}`);
      reg = new PlaneRegion(id, env, [0, 1, 2].map((d) => p(args[d], sc[d])), nrm);
      break;
    }
    default: {
      const ids = args.slice(1);
      for (const r of ids) if (!sys.regions.has(r)) throw new StyleError(`Region ${style} region ${r} does not exist`);
      reg = new CompoundRegion(id, style as 'union' | 'intersect', env, ids);
    }
  }
  reg.interior = side;
  reg.move = move;
  reg.rotate = rotate;
  reg.openFaces = open;
  sys.regions.set(id, reg);
};

/**
 * create_box N region-ID [bond/types N ...] — create_box.html. A block region
 * gives an orthogonal box, a prism region a restricted triclinic one. "For
 * 2d simulations, the z-axis bounds must bracket zero" (dimension.html: 2d
 * boxes straddle z = 0).
 */
const createBox: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/create_box.cpp CreateBox::command at b332f28)
  if (a.length < 2) throw missingArgs('create_box');
  if (sys.hasBox) throw new StyleError('Cannot create_box after simulation box is defined\nFor more information see https://docs.lammps.org/err0034');
  if (sys.dimension === 2 && sys.boundary[2][0] !== 'p') throw new StyleError('Cannot run 2d simulation with nonperiodic Z dimension');
  const n = int(a[0], 'number of atom types');
  if (n < 1) throw new StyleError('create_box needs at least 1 atom type');
  const general = a[1] === 'NULL';
  const reg = general ? null : sys.region(a[1] ?? '');
  const boxRest = a.slice(general ? 8 : 2);
  let lo: [number, number, number], hi: [number, number, number];
  let tilt: [number, number, number] | undefined;
  let genQ: [[number, number, number], [number, number, number], [number, number, number]] | undefined;
  if (general) {
    ({ lo, hi, tilt, Q: genQ } = generalCreateBox(sys.lattice, a.slice(2, 8), sys.dimension));
  } else if (reg instanceof BlockRegion) {
    const b = reg.bbox();
    if (!b) throw new StyleError('create_box: the region must be a static block with side in');
    lo = b.lo as [number, number, number]; hi = b.hi as [number, number, number];
  } else if (reg instanceof PrismRegion) {
    const v = reg.values();
    lo = [v[0], v[2], v[4]]; hi = [v[1], v[3], v[5]];
    tilt = [v[6], v[7], v[8]];
  } else if (reg instanceof ConeRegion && reg.bbox()) {
    // create_box.html: the region is enclosed by its axis-aligned bounding box (cylinder and cone)
    const b = reg.bbox()!;
    lo = b.lo as [number, number, number]; hi = b.hi as [number, number, number];
  } else throw new StyleError('create_box needs a block, cylinder, cone or prism region');
  if ([...lo, ...hi].some((v) => Math.abs(v) >= BIG)) throw new StyleError('create_box needs a finite region (no INF bounds)');
  // As native LAMMPS does (src/create_box.cpp at b332f28): "if (domain->boxlo[2] >= 0.0 || domain->boxhi[2] <= 0.0)"
  if (sys.dimension === 2 && !(lo[2] < 0 && hi[2] > 0)) throw new StyleError('Create_box region zlo/zhi for 2d simulation must straddle 0.0');
  if (sys.dimension === 2 && tilt && (tilt[1] !== 0 || tilt[2] !== 0)) throw new StyleError('2d triclinic boxes must have xz = yz = 0');
  const box = makeBox({ lo, hi, boundary: sys.boundary, tilt });
  // the rotation of a general triclinic box is kept on the box (types.ts SimBox.general)
  if (genQ) box.general = { Q: genQ };
  const s = emptyState(sys.units, sys.dimension, box, n, sys.atomStyle);
  // As native LAMMPS does (src/create_box.cpp CreateBox::command at b332f28): the keywords in order, each with its word
  // count, the atom style's permission for its kind (AtomVec bonds_allow, angles_allow, ...) and its value
  // (utils::inumeric). bond and bpm/sphere allow bonds, angle bonds and angles, molecular, full, dielectric and amoeba
  // all four, template those its molecules have; a hybrid style allows those of its sub-styles.
  const tmplSets = templateStyleId(sys.atomStyle) !== null ? sys.molecules.get(templateStyleId(sys.atomStyle)!) ?? [] : [];
  const KINDS = ['bond', 'angle', 'dihedral', 'improper'] as const;
  const allows = (kind: typeof KINDS[number]): boolean => atomSubStyles(sys.atomStyle).some((x) => {
    if (x === 'molecular' || x === 'full' || x === 'dielectric' || x === 'amoeba') return true;
    if (x === 'angle') return kind === 'bond' || kind === 'angle';
    if (x === 'bond' || x === 'bpm/sphere' || x === 'oxdna') return kind === 'bond';
    if (x === 'template') return tmplSets.some((t) => ({ bond: t.bonds, angle: t.angles, dihedral: t.dihedrals, improper: t.impropers })[kind].length > 0);
    return false;
  });
  const shownStyle = (sys.atomStyle.startsWith('hybrid ') ? 'hybrid ' : '') + atomSubStyles(sys.atomStyle).join(' ');
  const ntypesOf = { bond: 0, angle: 0, dihedral: 0, improper: 0 };
  for (let i = 0; i < boxRest.length; i += 2) {
    const key = boxRest[i];
    const m = /^(bond|angle|dihedral|improper)\/types$|^extra\/(bond|angle|dihedral|improper|special)\/per\/atom$/.exec(key);
    if (!m) throw new StyleError(`Unknown create_box keyword: ${key}`);
    // native names bond/types "create_box bond/type" in this message
    if (i + 2 > boxRest.length) throw missingArgs(`create_box ${key === 'bond/types' ? 'bond/type' : key}`);
    const kind = (m[1] ?? m[2]) as typeof KINDS[number] | 'special';
    if (kind !== 'special' && !allows(kind)) throw new StyleError(`No ${kind}s allowed with atom style ${shownStyle}`);
    const v = inumeric(boxRest[i + 1]);
    if (m[1]) ntypesOf[m[1] as typeof KINDS[number]] = v;
    // create_box.html, keyword extra/bond/per/atom: the bonds per atom the storage holds for later reads (data.ts)
    else if (kind === 'bond') {
      // native sizes its storage with any integer; the engine needs a count of at least 0
      if (v < 0) throw new StyleError('create_box extra/bond/per/atom must be >= 0');
      sys.extraBondPerAtom = v;
    }
  }
  s.topo.nbondtypes = ntypesOf.bond;
  s.topo.nangletypes = ntypesOf.angle;
  s.topo.ndihedraltypes = ntypesOf.dihedral;
  s.topo.nimpropertypes = ntypesOf.improper;
  sys.setState(s);
  // src/domain.cpp Domain::set_initial_box (expandflag = 1, called by create_box at b332f28): SMALL = 1.0e-4 of the box
  // length, taken before the expansion; a shrink-wrapped face is then moved out by that amount. create_atoms lattice
  // fill uses the expanded window until the box is first reset (System.pbc).
  sys.initSmall = [0, 1, 2].map((d) => 1.0e-4 * (box.hi[d] - box.lo[d])) as [number, number, number];
  sys.ff.bond?.allocate(s.topo.nbondtypes);
  sys.ff.angle?.allocate(s.topo.nangletypes);
  sys.ff.dihedral?.allocate(s.topo.ndihedraltypes);
  sys.ff.improper?.allocate(s.topo.nimpropertypes);
  const f = (v: number[]) => v.map((x) => Number(x.toPrecision(8))).join(' ');
  sys.log(`Created ${tilt ? 'triclinic' : 'orthogonal'} box = (${f(lo)}) to (${f(hi)})${tilt ? ` with tilt (${f(tilt)})` : ''}`);
};

/**
 * create_atoms type box | region ID | single x y z | random N seed region-ID —
 * create_atoms.html: "If your box is periodic and a multiple of the lattice
 * spacing in a particular dimension, LAMMPS is careful to put exactly one
 * particle at the boundary (on either side of the box), not zero or two."
 * Native LAMMPS: a lattice point on an upper face is included only when the
 * face is shrink-wrapped (s); p, f and m faces are half-open (2x2x2 sc box:
 * 8 atoms with p/f/m, 27 with s). Keywords basis, ratio, subset, group,
 * remap, var, set, overlap, maxtry, units ("units = lattice" default).
 */
/**
 * Appends copies of a molecule template whose atom positions are given
 * (flat, natoms per copy): types offset by toff, consecutive atom and
 * molecule IDs after the existing maxima, template charges, and the
 * template's bonds / angles / dihedrals / impropers with those IDs.
 * Atoms are wrapped into periodic dimensions with image flags, so each
 * molecule stays whole when unwrapped.
 */
export const insertMolecules = (sys: System, t: MoleculeTemplate, pts: number[], toff: number, gbit: number, rotMu?: number[]): void => {
  const s = sys.state;
  const copies = pts.length / (3 * t.natoms);
  if (copies === 0) { sys.log('Created 0 atoms'); return; }
  const types = new Int32Array(copies * t.natoms);
  for (let c = 0; c < copies; c++) for (let i = 0; i < t.natoms; i++) types[c * t.natoms + i] = t.type[i] + toff;
  for (const ty of types) if (ty < 1 || ty > s.ntypes) throw new StyleError(`molecule ${t.id}: atom type ${ty} is outside 1..${s.ntypes}`);
  // create_atoms.cpp (src/create_atoms.cpp at b332f28): molecule IDs are set when atom->molecule_flag is set, which
  // includes fix property/atom mol (not only a molecular atom_style)
  const molecular = hasMolecule(s);
  const tmplStyle = isTemplateStyle(sys.atomStyle);
  const needs: [string, number[][], number, string[]][] = [
    ['bonds', t.bonds, s.topo.nbondtypes, ['bond', 'angle', 'molecular', 'full', 'amoeba', 'template']],
    ['angles', t.angles, s.topo.nangletypes, ['angle', 'molecular', 'full', 'amoeba', 'template']],
    ['dihedrals', t.dihedrals, s.topo.ndihedraltypes, ['molecular', 'full', 'amoeba', 'template']],
    ['impropers', t.impropers, s.topo.nimpropertypes, ['molecular', 'full', 'amoeba', 'template']],
  ];
  for (const [what, list, ntypes, styles] of needs) {
    if (!list.length) continue;
    const ok = tmplStyle ? styles.includes('template') : styles.includes(sys.atomStyle);
    if (!ok) throw new StyleError(`molecule ${t.id} has ${what}, which atom_style ${sys.atomStyle} cannot store`);
    for (const e of list) if (e[0] < 1 || e[0] > ntypes) throw new StyleError(`molecule ${t.id}: ${what.slice(0, -1)} type ${e[0]} is outside 1..${ntypes}`);
  }
  if (t.q && !hasChargeStyle(sys.atomStyle)) throw new StyleError(`molecule ${t.id} has charges, which atom_style ${sys.atomStyle} cannot store`);
  if (t.mu && !hasDipoleStyle(sys.atomStyle)) throw new StyleError(`molecule ${t.id} has dipoles, which atom_style ${sys.atomStyle} cannot store`);
  const x = Float64Array.from(pts);
  const image = new Int32Array(x.length);
  const g = sys.geom;
  for (let i = 0; i < x.length / 3; i++) {
    g.remap(x, image, i);
    for (let d = 0; d < 3; d++) {
      if (s.box.periodic[d] || (d === 2 && sys.dimension === 2)) continue;
      if (x[3 * i + d] < s.box.lo[d] || x[3 * i + d] > s.box.hi[d]) throw new StyleError(`molecule ${t.id}: an inserted atom lies outside the non-periodic ${'xyz'[d]} boundary`);
    }
  }
  let maxMol = 0;
  for (let i = 0; i < s.n; i++) if (s.molecule[i] > maxMol) maxMol = s.molecule[i];
  let tmplMaxMol = 1;
  if (t.mol) for (const m of t.mol) tmplMaxMol = Math.max(tmplMaxMol, m);
  const mol = new Int32Array(copies * t.natoms);
  for (let c = 0; c < copies; c++) {
    for (let i = 0; i < t.natoms; i++) mol[c * t.natoms + i] = molecular ? maxMol + c * tmplMaxMol + (t.mol ? t.mol[i] : 1) : 0;
  }
  const q = t.q ? new Float64Array(copies * t.natoms) : undefined;
  if (q) for (let c = 0; c < copies; c++) q.set(t.q!, c * t.natoms);
  // atom_style template: every copy is template index 1 (the first molecule of the template), atoms 1..Natoms
  const tIdx = tmplStyle ? new Int32Array(copies * t.natoms).fill(1) : undefined;
  const tAt = tmplStyle ? Int32Array.from({ length: copies * t.natoms }, (_, k) => (k % t.natoms) + 1) : undefined;
  const base = maxAtomId(s);
  // Diameters / Masses from the molecule template (molecule.html): a sphere-style copy gets
  // radius = diameter / 2 (default 0.5) and per-atom mass = template mass, else 4/3 pi r^3 at
  // density 1. Measured with native LAMMPS (black box, create_atoms single mol atom_style sphere):
  // Diameters 1.0 / 2.0 gives radii 0.5 / 1.0; Masses 0.5 / 1.5 gives masses 0.5 / 1.5; without
  // both, radius 0.5 and mass 0.5235987755982988.
  let radius: Float64Array | undefined;
  let rmass: Float64Array | undefined;
  // a body molecule (one atom): native Molecule sets "radius[0] = avec_body->radius_body(nibody, ndbody, ibodyparams,
  // dbodyparams)" (src/molecule.cpp at b332f28), which Atom::add_molecule_atom copies
  const bstyle = t.body ? parseBodyStyle(sys.atomStyle)?.bstyle ?? null : null;
  if (s.radius) {
    radius = new Float64Array(copies * t.natoms);
    for (let c = 0; c < copies; c++) {
      for (let i = 0; i < t.natoms; i++) {
        radius[c * t.natoms + i] = bstyle ? bodyRadiusBody(bstyle, t.body!.ints, t.body!.dbls) : t.diam ? t.diam[i] / 2 : SPHERE_DEFAULT_RADIUS;
      }
    }
  }
  if (s.rmass && (t.mass || radius)) {
    rmass = new Float64Array(copies * t.natoms);
    for (let c = 0; c < copies; c++) for (let i = 0; i < t.natoms; i++) rmass[c * t.natoms + i] = t.mass ? t.mass[i] : sphereMass(radius![c * t.natoms + i], 1);
  }
  const mu = templateDipoles(t, copies, rotMu);
  const added = appendAtoms(s, { x, image, type: types, molecule: mol, q, mask: gbit, tmplIndex: tIdx, tmplAtom: tAt, radius, rmass, mu });
  if (bstyle) {
    // As native LAMMPS does (src/atom.cpp Atom::add_molecule_atom at b332f28): "body[ilocal] = 0; // as if a body read
    // from data file", data_body() with the template's integers and doubles, and set_quat(ilocal, onemol->quat_external).
    // create_atoms sets quat_external only "if (onemol->muflag)" (src/create_atoms.cpp at b332f28), which a body template
    // never has, so the body keeps the Molecule constructor's quat_external of zeros: measured with native LAMMPS, its
    // sub-particles all sit at the body's center (compute body/local) and fix nve/body then stops with "Non-numeric atom
    // coords - simulation unstable"
    for (let c = 0; c < copies; c++) setBodyAtom(s, s.n - copies + c, bstyle, t.body!.ints, t.body!.dbls, [0, 0, 0, 0]);
  }
  for (let c = 0; c < copies; c++) {
    const id0 = base + c * t.natoms;
    for (const [what, list] of [['bonds', t.bonds], ['angles', t.angles], ['dihedrals', t.dihedrals], ['impropers', t.impropers]] as const) {
      for (const e of list) pushTopo(s.topo[what], e[0], e.slice(1).map((k) => id0 + k));
    }
  }
  sys.atomsChanged();
  sys.log(`Created ${added} atoms (${copies} molecules of template ${t.id})`);
};

/**
 * The checks native LAMMPS makes on a create_atoms command before it creates any atom (src/create_atoms.cpp
 * CreateAtoms::command at b332f28): the box and the lattice's dimension, the style and its values, every keyword in
 * order, then the atom type, the mesh units and the var/set variables.
 */
const checkCreateAtomsArgs = (sys: System, a: string[]): void => {
  if (!sys.hasBox) throw new StyleError('Create_atoms command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  if (sys.lattice) checkLatticeDimension(sys.lattice.style, sys.dimension);
  if (a.length < 2) throw missingArgs('create_atoms');
  const s = sys.state;
  const typeOf = (w: string) => inumeric(sys.labels.defined && !isNumericTypeWord(w) ? typeWord(sys.labels, 'atom', w) : w);
  const ntype = typeOf(a[0]);
  const style = a[1];
  let k: number;
  const region = (w: string) => { if (!sys.regions.has(w)) throw new StyleError(`Create_atoms region ${w} does not exist`); };
  if (style === 'box') k = 2;
  else if (style === 'region') {
    if (a.length < 3) throw missingArgs('create_atoms region');
    region(a[2]);
    k = 3;
  } else if (style === 'single') {
    if (a.length < 5) throw missingArgs('create_atoms single');
    const z = [numeric(a[2]), numeric(a[3]), numeric(a[4])][2];
    if (sys.dimension === 2 && z !== 0) throw new StyleError('Create_atoms single for 2d simulation requires z coord = 0.0');
    k = 5;
  } else if (style === 'random') {
    if (a.length < 5) throw missingArgs('create_atoms random');
    const n = inumeric(a[2]);
    if (n < 0) throw new StyleError(`Illegal create_atoms number of random atoms ${n}`);
    const seed = inumeric(a[3]);
    if (seed <= 0) throw new StyleError(`Illegal create_atoms random seed ${seed}`);
    if (a[4] !== 'NULL') region(a[4]);
    k = 5;
  } else if (style === 'mesh') {
    if (a.length < 3) throw missingArgs('create_atoms mesh');
    k = 3;
  } else throw new StyleError(`Unknown create_atoms command option ${style}`);
  let units = 'lattice', mol: string | null = null, vstr: string | null = null;
  const sets: string[] = [];
  const need = (n: number, key: string) => { if (k + n > a.length) throw missingArgs(`create_atoms ${key}`); };
  while (k < a.length) {
    const key = a[k];
    switch (key) {
      case 'basis': {
        need(3, key);
        const ib = inumeric(a[k + 1]), it = typeOf(a[k + 2]);
        const nbasis = sys.lattice?.basis.length ?? 0;
        if (ib <= 0 || ib > nbasis || it <= 0 || it > s.ntypes) throw new StyleError(`Out of range basis setting '${ib} ${it}' in create_atoms command`);
        k += 3;
        break;
      }
      case 'remap': need(2, key); logical(a[k + 1]); k += 2; break;
      case 'mol':
        need(3, key);
        if (!sys.molecules.has(a[k + 1])) throw new StyleError(`Molecule template ID ${a[k + 1]} for create_atoms does not exist`);
        mol = a[k + 1];
        inumeric(a[k + 2]);
        k += 3;
        break;
      case 'units':
        need(2, key);
        if (a[k + 1] !== 'box' && a[k + 1] !== 'lattice') throw new StyleError(`Unknown create_atoms units option ${a[k + 1]}`);
        units = a[k + 1];
        k += 2;
        break;
      case 'var':
        if (style === 'single') throw new StyleError("Cannot use 'var' keyword with 'single' style for create_atoms");
        need(2, key);
        vstr = a[k + 1];
        k += 2;
        break;
      case 'set':
        need(3, key);
        // native names the variable (the third word) when the dimension is unknown
        if (!['x', 'y', 'z'].includes(a[k + 1])) throw new StyleError(`Unknown create_atoms set option ${a[k + 2]}`);
        sets.push(a[k + 2]);
        k += 3;
        break;
      case 'rotate': {
        need(5, key);
        numeric(a[k + 1]);
        const ax = [numeric(a[k + 2]), numeric(a[k + 3]), numeric(a[k + 4])];
        if (ax.every((v) => v === 0)) throw new StyleError('Illegal create_atoms rotate arguments');
        if (sys.dimension === 2 && (ax[0] !== 0 || ax[1] !== 0)) throw new StyleError('Invalid create_atoms rotation vector for 2d model');
        k += 5;
        break;
      }
      case 'ratio': {
        need(3, key);
        const f = numeric(a[k + 1]), seed = inumeric(a[k + 2]);
        if (f <= 0 || f > 1 || seed <= 0) throw new StyleError('Illegal create_atoms ratio settings');
        k += 3;
        break;
      }
      case 'subset': {
        need(3, key);
        const n = bnumeric(a[k + 1]), seed = inumeric(a[k + 2]);
        if (n <= 0 || seed <= 0) throw new StyleError('Illegal create_atoms subset settings');
        k += 3;
        break;
      }
      case 'group': need(2, key); k += 2; break;
      case 'overlap':
        if (style !== 'random') throw new StyleError('Create_atoms overlap can only be used with random style');
        need(2, key);
        if (numeric(a[k + 1]) <= 0) throw new StyleError(`Illegal create_atoms overlap value: ${numeric(a[k + 1])}`);
        k += 2;
        break;
      case 'maxtry':
        if (style !== 'random') throw new StyleError('Create_atoms maxtry can only be used with random style');
        need(2, key);
        if (inumeric(a[k + 1]) <= 0) throw new StyleError(`Illegal create_atoms maxtry value ${inumeric(a[k + 1])}`);
        k += 2;
        break;
      case 'meshmode':
        if (style !== 'mesh') throw new StyleError('Create_atoms meshmode can only be used with mesh style');
        need(3, key);
        if (a[k + 1] !== 'bisect' && a[k + 1] !== 'qrand') throw new StyleError(`Unknown create_atoms meshmode ${a[k + 2]}`);
        numeric(a[k + 2]);
        k += 3;
        break;
      case 'radscale':
        need(2, key);
        if (style !== 'mesh') throw new StyleError('Create_atoms radscale can only be used with mesh style');
        if (!s.radius) throw new StyleError('Must have atom attribute radius to set radscale factor');
        if (numeric(a[k + 1]) <= 0) throw new StyleError(`Illegal create_atoms radscale value: ${numeric(a[k + 1])}`);
        k += 2;
        break;
      default: throw new StyleError(`Illegal create_atoms command option ${key}`);
    }
  }
  if (!mol && (ntype <= 0 || ntype > s.ntypes)) throw new StyleError('Invalid atom type in create_atoms command');
  if (mol) {
    // As native LAMMPS does (src/create_atoms.cpp at b332f28): the type offset plus the largest type of the (first)
    // molecule must be an atom type
    const onemol = sys.molecules.get(mol)![0];
    let maxType = 0;
    for (let i = 0; i < onemol.type.length; i++) if (onemol.type[i] > maxType) maxType = onemol.type[i];
    if (ntype + maxType <= 0 || ntype + maxType > s.ntypes) throw new StyleError('Invalid atom type in create_atoms mol command');
  }
  if (style === 'mesh' && mol) throw new StyleError("Create_atoms mesh is not compatible with the 'mol' option");
  if (style === 'mesh' && units !== 'box') throw new StyleError("Create_atoms mesh must use 'units box' option");
  if ((vstr === null) !== (sets.length === 0)) throw new StyleError('Incomplete use of variables in create_atoms command');
  if (vstr !== null) {
    const v = sys.vars.get(vstr);
    if (!v) throw new StyleError(`Variable ${vstr} for create_atoms does not exist`);
    if (v.style !== 'equal' && v.style !== 'timer' && v.style !== 'internal') throw new StyleError(`Variable ${vstr} for create_atoms is invalid style`);
  }
};

const createAtoms: Handler = ({ sys }, a) => {
  checkCreateAtomsArgs(sys, a);
  const s = sys.state;
  // create_atoms.html: "type = atom type (1-Ntypes or type label)"
  let type = int(typeLabelWord(sys, 'atom', a[0]), 'atom type');
  const style = a[1];
  let rest: string[];
  let regionId: string | null = null;
  let single: number[] | null = null;
  let random: { count: number; seed: number } | null = null;
  let meshFile: string | null = null;
  if (style === 'box') rest = a.slice(2);
  else if (style === 'region') { regionId = a[2]; sys.region(regionId ?? ''); rest = a.slice(3); } else if (style === 'single') {
    single = [num(a[2], 'x'), num(a[3], 'y'), num(a[4], 'z')];
    rest = a.slice(5);
  } else if (style === 'random') {
    random = { count: int(a[2], 'number of atoms'), seed: int(a[3], 'seed') };
    if (random.seed <= 0) throw new StyleError('create_atoms random: seed must be a positive integer');
    regionId = a[4] === 'NULL' ? null : a[4];
    if (a[4] === undefined) throw new StyleError('usage: create_atoms type random N seed region-ID|NULL');
    if (regionId) sys.region(regionId);
    rest = a.slice(5);
  } else if (style === 'mesh') {
    meshFile = a[2];
    if (!meshFile) throw new StyleError('usage: create_atoms type mesh STL-file keyword values ...');
    rest = a.slice(3);
  } else throw new StyleError(`unknown create_atoms style '${style ?? ''}'`);
  const kw = new Map<string, string[]>();
  const basisType = new Map<number, number>();
  for (let k = 0; k < rest.length;) {
    const key = rest[k];
    const nv: Record<string, number> = { basis: 2, ratio: 2, subset: 2, group: 1, remap: 1, var: 1, set: 2, overlap: 1, maxtry: 1, units: 1, rotate: 4, mol: 2, radscale: 1, meshmode: 2 };
    if (!(key in nv)) throw new StyleError(`unknown create_atoms keyword '${key}'`);
    const vals = rest.slice(k + 1, k + 1 + nv[key]);
    if (vals.length < nv[key]) throw new StyleError(`create_atoms ${key} needs ${nv[key]} value(s)`);
    if (key === 'basis') basisType.set(int(vals[0], 'basis index'), int(typeLabelWord(sys, 'atom', vals[1]), 'basis type'));
    else if (key === 'set') kw.set(`set_${vals[0]}`, vals);
    else kw.set(key, vals);
    k += 1 + nv[key];
  }
  // create_atoms.html: "type = atom type ... of atoms to create (offset for molecule creation)"
  const molKw = kw.get('mol');
  const tmpl = molKw ? sys.molecules.get(molKw[0])?.[0] : undefined;
  if (molKw && !tmpl) throw new StyleError(`create_atoms mol: molecule template '${molKw[0]}' does not exist`);
  // create_atoms.html: "the same molecule template-ID." (the mol template must be the atom_style template)
  const styleTmpl = templateStyleId(sys.atomStyle);
  if (tmpl && styleTmpl !== null && molKw![0] !== styleTmpl) {
    throw new StyleError(`create_atoms mol: molecule template ${molKw![0]} is not the template ${styleTmpl} of atom_style template; they must use the same molecule template-ID`);
  }
  if (!tmpl && (type < 1 || type > s.ntypes)) throw new StyleError(`atom type ${type} is outside 1..${s.ntypes}`);
  if (tmpl && type < 0) throw new StyleError('create_atoms with mol: the type (an offset) must be >= 0');
  // Measured with native LAMMPS (black box, 3d, create_atoms box mol cychex 734594 on a lattice): the
  // orientation of molecule k (k = 0, 1, ...) comes from a Marsaglia stream (RanMars) seeded with the mol
  // seed: one leading draw, then four per molecule: the axis is (u1 - 0.5, u2 - 0.5, u3 - 0.5) (normalised)
  // and the angle is 2 pi u4. Recovered from the coordinates of all 64 molecules of that run.
  const molRng = molKw ? new RanMars(int(molKw[1], 'mol seed')) : null;
  if (molRng) molRng.uniform();
  if (molKw && int(molKw[1], 'mol seed') <= 0) throw new StyleError('create_atoms mol: seed must be a positive integer');
  const rotKw = kw.get('rotate');
  if (rotKw && !tmpl) throw new StyleError('create_atoms rotate needs the mol keyword');
  if (rotKw && sys.dimension === 2 && (num(rotKw[1], 'Rx') !== 0 || num(rotKw[2], 'Ry') !== 0)) {
    throw new StyleError('create_atoms rotate: "A rotation vector specified for a single molecule must be in the z-direction for a 2d model."');
  }
  const fixedR = rotKw ? orientationMatrix((num(rotKw[0], 'theta') / 180) * Math.PI, num(rotKw[1], 'Rx'), num(rotKw[2], 'Ry'), num(rotKw[3], 'Rz')) : null;
  const molCenter = tmpl ? geometricCenter(tmpl) : [0, 0, 0];
  /** create_atoms.html: "placing the geometric center of the molecule at the lattice point, and (by default) giving the molecule a random orientation about the point" */
  const placeMol = (p: number[]): number[] => {
    let R = fixedR;
    if (!R) {
      if (sys.dimension === 2) R = orientationMatrix(molRng!.uniform() * (2 * Math.PI), 0, 0, 1);
      else {
        // rotation by 2 pi u4 about the axis (u1 - 0.5, u2 - 0.5, u3 - 0.5), see molRng above
        const u1 = molRng!.uniform(), u2 = molRng!.uniform(), u3 = molRng!.uniform(), u4 = molRng!.uniform();
        R = orientationMatrix(u4 * (2 * Math.PI), u1 - 0.5, u2 - 0.5, u3 - 0.5);
      }
    }
    const out: number[] = [];
    // the point dipoles turn with the molecule (create_atoms.html rotates the molecule about its center; the
    // dipole vectors are rotated with it): measured with native LAMMPS (black box, a 3-atom dipole molecule
    // rotated 20 deg about x: mu = (0, 0.5, 0) became (0, 0.4698, 0.1710), i.e. R mu)
    lastMu = [];
    for (let i = 0; i < tmpl!.natoms; i++) {
      const r = [0, 1, 2].map((d) => tmpl!.x[3 * i + d] - molCenter[d]);
      // matvec then add3 in native (create_atoms.cpp get_xmol): the rotated offset is summed first, then the center is added
      for (let d = 0; d < 3; d++) out.push(p[d] + (R[d][0] * r[0] + R[d][1] * r[1] + R[d][2] * r[2]));
      if (tmpl!.mu) {
        const m = [0, 1, 2].map((d) => tmpl!.mu![3 * i + d]);
        for (let d = 0; d < 3; d++) lastMu.push(R[d][0] * m[0] + R[d][1] * m[1] + R[d][2] * m[2]);
      }
    }
    return out;
  };
  /** Rotated dipoles of the molecules placed so far (3 per atom, in the order of pts); lastMu is the latest placement. */
  const molMu: number[] = [];
  let lastMu: number[] = [];
  const unitsW = kw.get('units')?.[0] ?? 'lattice';
  const g = sys.geom;
  const box = s.box;
  const reg = regionId ? sys.region(regionId) : null;
  // variable test (var + set x/y/z)
  const varName = kw.get('var')?.[0];
  const setVars = ['x', 'y', 'z'].map((d) => kw.get(`set_${d}`)?.[1] ?? null);
  const varOk = (p: number[]) => {
    if (!varName) return true;
    setVars.forEach((v, d) => { if (v) sys.vars.setInternal(v, p[d]); });
    return sys.equalVariable(varName) !== 0;
  };
  const lam = [0, 0, 0];
  const insideBox = (x: number, y: number, z: number) => {
    g.toLamda(x, y, z, lam);
    const eps = 1e-9;
    for (let d = 0; d < (sys.dimension === 2 ? 2 : 3); d++) {
      const upperInclusive = box.boundary[d][1] === 's';
      if (lam[d] < -eps) return false;
      if (upperInclusive ? lam[d] > 1 + eps : lam[d] >= 1 - eps) return false;
    }
    return true;
  };
  // Lattice fill (create_atoms box | region): the insertion window of native add_lattice (src/create_atoms.cpp at
  // b332f28, the bounds set before loop_lattice). The window bounds are the sublo/subhi of the box (src/domain.cpp
  // set_local_box with one proc: subhi = boxhi exactly). A triclinic box (a prism region, even with zero tilt) tests
  // the lamda coordinates against [0, 1) (set_lamda_box), computed by x2lamda in native's operation order; an
  // orthogonal box tests the box coordinates themselves. Only a periodic face moves: "if I am lo proc, decrement
  // lower bound by EPSILON ... if I am hi proc, decrement upper bound by 2.0*EPSILON" with EPSILON = 1.0e-6 in lamda
  // units or prd*1e-6 in box units, so exactly one of two periodic images is created. A non-periodic face has no
  // tolerance: a lattice point on it is kept only when its computed coordinate lies below the upper bound, which
  // depends on the last bit of the point (the rutile TiO2 cell of examples/PACKAGES/smtbq with boundary f f f gives
  // 60 atoms in a block box and 75 in a prism box, not 48). A shrink-wrapped face is expanded by SMALL = 1e-4 of the
  // box length when create_box makes it (src/domain.cpp set_initial_box, expandflag = 1; sys.initSmall holds those
  // amounts until the first reset of the box in pbc), so the window reaches past the nominal face.
  const winLo: [number, number, number] = [box.lo[0], box.lo[1], box.lo[2]];
  const winHi: [number, number, number] = [box.hi[0], box.hi[1], box.hi[2]];
  if (sys.initSmall) {
    for (let d = 0; d < 3; d++) {
      if (box.boundary[d][0] === 's') winLo[d] -= sys.initSmall[d];
      if (box.boundary[d][1] === 's') winHi[d] += sys.initSmall[d];
    }
  }
  const winGeom = box.triclinic ? new Geometry({ ...box, lo: winLo, hi: winHi }) : null;
  const winLam = [0, 0, 0];
  const insideLatticeBox = (x: number, y: number, z: number) => {
    if (winGeom) winGeom.toLamda(x, y, z, winLam);
    for (let d = 0; d < 3; d++) {
      const periodicLo = box.boundary[d][0] === 'p', periodicHi = box.boundary[d][1] === 'p';
      const v = winGeom ? winLam[d] : d === 0 ? x : d === 1 ? y : z;
      const subLo = winGeom ? 0 : winLo[d], subHi = winGeom ? 1 : winHi[d];
      const epsilon = winGeom ? 1e-6 : (winHi[d] - winLo[d]) * 1e-6;
      const lowBound = periodicLo ? subLo - epsilon : subLo;
      const highBound = periodicHi ? subHi - 2.0 * epsilon : subHi;
      if (v < lowBound || v >= highBound) return false;
    }
    return true;
  };
  let pts: number[] = [];
  let types: number[] = [];
  let rad: number[] | null = null;
  let meshMol = 0;
  if (single) {
    const sc = latticeScale(sys, unitsW, 'create_atoms');
    const p = single.map((v, d) => v * sc[d]);
    if (sys.dimension === 2) p[2] = 0;
    const remap = kw.has('remap') ? yesno(kw.get('remap')![0], 'remap') : false;
    if (remap) {
      const xx = Float64Array.from(p), im = new Int32Array(3);
      g.remap(xx, im, 0);
      p[0] = xx[0]; p[1] = xx[1]; p[2] = xx[2];
    }
    g.toLamda(p[0], p[1], p[2], lam);
    const inside = [0, 1, 2].every((d) => (d === 2 && sys.dimension === 2) || (lam[d] >= 0 && (box.boundary[d][1] === 's' || box.boundary[d][1] === 'm' ? lam[d] <= 1 : lam[d] < 1)));
    if (!inside) {
      sys.warn('create_atoms single: the point is outside the box; no atom created');
    } else if (varOk(p)) { pts = tmpl ? placeMol(p) : p; molMu.push(...lastMu); types = [type]; }
  } else if (random) {
    // Measured with native LAMMPS (black box, create_atoms 2 random 800 495437 box, and 3 or 5 atoms with
    // seeds 12345 and 777, with and without molecules): the coordinates are consecutive Park-Miller draws
    // (RanPark), x then y then z for each attempt, after the first 30 draws of the stream, which native
    // does not use for any atom
    const rng = new RanPark(random.seed);
    for (let k = 0; k < 30; k++) rng.uniform();
    // create_atoms.html, keyword units (default lattice) "lattice means the distance units are in lattice spacings",
    // and the overlap distance is one of the values it scales (x lattice spacing; box leaves it unscaled).
    // Native rejects an explicit overlap that is not > 0 ("Illegal create_atoms overlap value", measured with
    // native LAMMPS, black box: overlap 0.0 and -0.5 both stop the run).
    const overlapW = kw.has('overlap') ? num(kw.get('overlap')![0], 'overlap') : 0;
    if (kw.has('overlap') && !(overlapW > 0)) throw new StyleError(`Illegal create_atoms overlap value: ${kw.get('overlap')![0]}`);
    const overlap = overlapW * latticeScale(sys, unitsW, 'create_atoms')[0];
    // create_atoms.html, keyword maxtry: "It limits the number of attempts to generate valid coordinates for a
    // single new particle." Native stops with "Illegal create_atoms maxtry value 0" (measured, black box).
    // The default is 1000 as in native LAMMPS (src/create_atoms.cpp at b332f28: DEFAULT_MAXTRY = 1000), not the
    // "maxtry = 10" of the page's Default section. Measured with native LAMMPS: 300 random atoms with overlap 0.8 in
    // an 8 x 7 x 6 box are all inserted by default and 256 with maxtry 10 (the engine's count with 10).
    const maxtry = kw.has('maxtry') ? int(kw.get('maxtry')![0], 'maxtry') : 1000;
    if (maxtry < 1) throw new StyleError(`Illegal create_atoms maxtry value ${maxtry}`);
    // sample in the region's bounding box clipped to the box, accept points inside both
    const bb = reg?.bbox();
    const blo = [0, 1, 2].map((d) => Math.max(box.lo[d] + Math.min(0, d === 0 ? Math.min(box.tilt[0], box.tilt[1], box.tilt[0] + box.tilt[1]) : d === 1 ? box.tilt[2] : 0), bb ? bb.lo[d] : -Infinity));
    const bhi = [0, 1, 2].map((d) => Math.min(box.hi[d] + Math.max(0, d === 0 ? Math.max(box.tilt[0], box.tilt[1], box.tilt[0] + box.tilt[1]) : d === 1 ? box.tilt[2] : 0), bb ? bb.hi[d] : Infinity));
    const existing: number[] = Array.from(s.x.subarray(0, 3 * s.n));
    const tooClose = (p: number[]) => {
      if (overlap <= 0) return false;
      const d = [0, 0, 0];
      for (const arr of [existing, pts]) {
        for (let j = 0; j < arr.length; j += 3) {
          d[0] = p[0] - arr[j]; d[1] = p[1] - arr[j + 1]; d[2] = p[2] - arr[j + 2];
          g.minimumImage(d);
          if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < overlap * overlap) return true;
        }
      }
      return false;
    };
    let failed = 0;
    for (let c = 0; c < random.count; c++) {
      let placed = false;
      // every attempt counts against maxtry, whichever test rejects it (create_atoms.html, maxtry: "limits the number of attempts")
      for (let t = 0; t < maxtry && !placed; t++) {
        // Measured with native LAMMPS (black box, dimension 2, create_atoms 1 random 5 12345 NULL): every attempt
        // draws x, y and z; in 2d the z draw is discarded and z = 0 (native's second atom has x equal to the y the
        // engine gave it when it skipped the z draw)
        const p = [0, 1, 2].map((d) => {
          const u = rng.uniform();
          return d === 2 && sys.dimension === 2 ? 0 : blo[d] + u * (bhi[d] - blo[d]);
        });
        if (!insideBox(p[0], p[1], p[2])) continue;
        if (reg && !reg.match(p[0], p[1], p[2])) continue;
        if (!varOk(p)) continue;
        const atoms = tmpl ? placeMol(p) : p;
        let close = false;
        for (let j = 0; j < atoms.length && !close; j += 3) close = tooClose([atoms[j], atoms[j + 1], atoms[j + 2]]);
        if (close) continue;
        pts.push(...atoms);
        molMu.push(...lastMu);
        types.push(type);
        placed = true;
      }
      if (!placed) failed++;
    }
    // native's warning (src/create_atoms.cpp add_random)
    if (failed) sys.warn(`Only inserted ${random.count - failed} particles out of ${random.count}`);
  } else if (meshFile) {
    // create_atoms.html: "For the *mesh* style, a file with a triangle mesh in"
    // STL format "is read and one or more particles are placed into the area of
    // each triangle." "The use of the *units box* option is required."
    if (unitsW !== 'box') throw new StyleError('create_atoms mesh must use the "units box" option (create_atoms.html: "The use of the units box option is required.")');
    if (molKw) throw new StyleError('create_atoms mesh does not support the mol keyword');
    const tris = parseStl(sys.readFile(meshFile, 'Cannot open STL file'), meshFile);
    const mm = kw.get('meshmode');
    const mode = mm ? mm[0] : 'bisect';
    const radscale = kw.has('radscale') ? num(kw.get('radscale')![0], 'radscale') : 1.0;
    // create_atoms.html: "Its value is a prefactor (must be > 0.0, default is 1.0)"
    if (radscale <= 0) throw new StyleError('create_atoms mesh: radscale must be > 0.0');
    const dist3 = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    // create_atoms.html: "You cannot use this command to create atoms that are
    // outside the simulation box; they will just be ignored by LAMMPS." The
    // *var* and *set* keywords "apply to all styles except *single*".
    rad = [];
    const emit = (p: number[], r: number) => {
      if (!insideBox(p[0], p[1], p[2]) || !varOk(p)) return;
      pts.push(p[0], p[1], p[2]);
      types.push(type);
      rad!.push(r);
    };
    // create_atoms.html: "If the atom style allows it, the radius will be set
    // to a value depending on the algorithm and the value of the *radscale*
    // parameter ... and the atoms created from the mesh are assigned a new
    // molecule ID."
    let maxMol = 0;
    for (let i = 0; i < s.n; i++) if (s.molecule[i] > maxMol) maxMol = s.molecule[i];
    meshMol = Math.max(1, maxMol + 1);
    if (mode === 'bisect') {
      // create_atoms.html: "In *bisect* mode a particle is created at the
      // center of each triangle unless the average distance of the triangle
      // vertices from its center is larger than the *radthresh* value ... In
      // case the average distance is over the threshold, the triangle is
      // recursively split into two halves along the the longest side until the
      // threshold is reached. There will be at least one sphere per triangle."
      // Measured with native LAMMPS (black box): with the required units box
      // the default radthresh is 1.0 (it does not depend on any lattice), the
      // center is the centroid, and the average distance sets the radius.
      const radthresh = mm ? num(mm[1], 'radthresh') : 1.0;
      for (const tri of tris) {
        const stack: [number, number, number][][] = [tri.v];
        while (stack.length) {
          const t = stack.pop()!;
          const c = [0, 1, 2].map((d) => (t[0][d] + t[1][d] + t[2][d]) / 3);
          const avg = (dist3(t[0], c) + dist3(t[1], c) + dist3(t[2], c)) / 3;
          if (avg > radthresh) {
            const e = [dist3(t[0], t[1]), dist3(t[1], t[2]), dist3(t[2], t[0])];
            let idx = 0;
            if (e[1] > e[idx]) idx = 1;
            if (e[2] > e[idx]) idx = 2;
            const i = idx, j = (idx + 1) % 3, k = (idx + 2) % 3;
            const mid = [0, 1, 2].map((d) => (t[i][d] + t[j][d]) / 2) as [number, number, number];
            // children of the split edge (i,j), the shared vertex k; the
            // second child is popped (created) first, matching native's order
            stack.push([t[i], mid, t[k]]);
            stack.push([mid, t[j], t[k]]);
          } else emit(c, avg * radscale);
        }
      }
    } else if (mode === 'qrand') {
      if (!mm) throw new StyleError('create_atoms mesh meshmode qrand needs a density value');
      // create_atoms.html: "In *qrand* mode a quasi-random sequence is used to
      // distribute particles on mesh triangles using an approach by (Roberts).
      // Particles are added to the triangle until the minimum number density is
      // met or exceeded such that every triangle will have at least one
      // particle." "The radius will be set so that the sum of the area of the
      // radius of the particles created in place of a triangle will be equal to
      // the area of that triangle."
      const density = num(mm[1], 'density');
      // (Roberts) R. Roberts (2019) "Evenly Distributing Points in a Triangle":
      // the R2 sequence from the plastic number rho (x^3 = x + 1), mapped into
      // the triangle and folded. Measured with native LAMMPS (black box): the
      // barycentric sequence (u,w) is triangle-independent,
      // u=frac(frac(i alpha2)+0.5), w=frac(frac(i alpha1)+0.5), reflected through
      // (0.5,0.5) when u+w>1, with alpha1 = 1/rho and alpha2 = 1/rho^2 rounded to
      // 7 decimals: solving native's points for (u,w) gives 0.0698403, 0.2548777
      // at i = 1 and 0.6396806, 0.0097554 at i = 2 (w35mesh_qrand matches).
      const alpha1 = 0.7548777, alpha2 = 0.5698403;
      const frac = (x: number) => x - Math.floor(x);
      for (const tri of tris) {
        // Measured with native LAMMPS (black box, every point of w35mesh_qrand): the sequence starts at
        // the vertex opposite the longest edge, u runs towards the lower-numbered other vertex and w
        // towards the higher one (equal longest edges, not measured, take the first).
        const v = tri.v;
        const opp = [dist3(v[1], v[2]), dist3(v[2], v[0]), dist3(v[0], v[1])];
        let o = 0;
        if (opp[1] > opp[o]) o = 1;
        if (opp[2] > opp[o]) o = 2;
        const a = v[o], b = v[o === 0 ? 1 : 0], cc = v[o === 2 ? 1 : 2];
        const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        const wx = cc[0] - a[0], wy = cc[1] - a[1], wz = cc[2] - a[2];
        const area = 0.5 * Math.hypot(uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx);
        const n = Math.max(1, Math.ceil(density * area));
        const r = Math.sqrt(area / (n * Math.PI)) * radscale;
        for (let i = 1; i <= n; i++) {
          let u = frac(frac(i * alpha2) + 0.5);
          let w = frac(frac(i * alpha1) + 0.5);
          if (u + w > 1) { u = 1 - u; w = 1 - w; }
          emit([a[0] + u * ux + w * wx, a[1] + u * uy + w * wy, a[2] + u * uz + w * wz], r);
        }
      }
    } else throw new StyleError(`create_atoms mesh: unknown meshmode '${mode}' (bisect or qrand)`);
  } else {
    const lat = sys.lattice;
    if (!lat || lat.style === 'none') throw new StyleError(`create_atoms ${style} needs a lattice (lattice command)`);
    // bounding box of the domain (tilted boxes: extents of the parallelepiped)
    const xlo = box.lo[0] + Math.min(0, box.tilt[0], box.tilt[1], box.tilt[0] + box.tilt[1]);
    const xhi = box.hi[0] + Math.max(0, box.tilt[0], box.tilt[1], box.tilt[0] + box.tilt[1]);
    const ylo = box.lo[1] + Math.min(0, box.tilt[2]), yhi = box.hi[1] + Math.max(0, box.tilt[2]);
    const accept = (x: number, y: number, z: number) => insideLatticeBox(x, y, z) && (!reg || reg.match(x, y, z)) && varOk([x, y, z]);
    // As native LAMMPS does (src/create_atoms.cpp add_lattice at b332f28): with a region that has a bounding box, only
    // the lattice cells covering the region's extent clipped to the box are visited, so a site far from the region
    // that the periodic remap of the region test would move inside it is never tested. Measured with native LAMMPS:
    // examples/PACKAGES/ptm/in.ptm's region upper (y 0 .. 64.0361225 in a periodic box -64.0361225 .. 64.0361225)
    // gets 200 atoms; testing every box site gave 202 (two sites just below the lower face, remapped to the top).
    const bboxLo = [xlo, ylo, box.lo[2]], bboxHi = [xhi, yhi, box.hi[2]];
    const rb = !lat.general && reg ? reg.bbox() : null;
    if (rb) {
      for (let d = 0; d < 3; d++) {
        if (rb.lo[d] > bboxLo[d]) bboxLo[d] = rb.lo[d] > bboxHi[d] ? bboxHi[d] : rb.lo[d];
        if (rb.hi[d] < bboxHi[d]) bboxHi[d] = rb.hi[d] < bboxLo[d] ? bboxLo[d] : rb.hi[d];
      }
    }
    const site = lat.general
      ? generalAtomSites(lat.general, generalBoxFromRestricted(lat.general.Q, box.lo as [number, number, number], box.hi as [number, number, number], box.tilt as [number, number, number]), accept)
      : latticeSites(lat, bboxLo, bboxHi, sys.dimension, accept);
    pts = site.x;
    // create_atoms.html: basis "M = which basis atom"; the lattice defines lat.basis.length of them. Native stops
    // with "Out of range basis setting '2 2' in create_atoms command" (measured, black box: lattice sc, basis 2 2).
    for (const [b, t] of basisType) {
      if (!(b >= 1 && b <= lat.basis.length)) throw new StyleError(`Out of range basis setting '${b} ${t}' in create_atoms command`);
    }
    types = site.basis.map((b) => basisType.get(b + 1) ?? type);
    // ratio / subset: choose a random subset of the sites
    const pick = kw.get('ratio') ?? kw.get('subset');
    if (pick) {
      const nsite = types.length;
      const want = kw.has('ratio') ? Math.floor(num(pick[0], 'ratio') * nsite) : int(pick[0], 'subset');
      if (want > nsite) throw new StyleError(`create_atoms subset: ${want} is more than the ${nsite} lattice sites`);
      const rng = new Rng(int(pick[1], 'seed'));
      const idx = Array.from({ length: nsite }, (_, i) => i);
      for (let i = nsite - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      const keep = idx.slice(0, want).sort((p, q) => p - q);
      pts = keep.flatMap((i) => [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]]);
      types = keep.map((i) => types[i]);
    }
  }
  const groupName = kw.get('group')?.[0];
  const gbit = groupName ? bitOfIndex(sys.groups.create(groupName)) : 0;
  if (tmpl) {
    if (!single && !random) {
      const centers = pts;
      pts = [];
      for (let c = 0; c < centers.length; c += 3) {
        pts.push(...placeMol([centers[c], centers[c + 1], centers[c + 2]]));
        molMu.push(...lastMu);
      }
    }
    insertMolecules(sys, tmpl, pts, type, gbit, molMu);
    return;
  }
  for (const t of types) if (t < 1 || t > s.ntypes) throw new StyleError(`create_atoms basis type ${t} is outside 1..${s.ntypes}`);
  // As native LAMMPS does (src/atom_vec_body.cpp AtomVecBody::create_atom_post at b332f28): an atom of atom_style body
  // has "radius[ilocal] = 0.5; rmass[ilocal] = 1.0; body[ilocal] = -1;" (a point particle, no body record)
  const bodyAtoms = isBodyStyle(sys.atomStyle);
  const added = appendAtoms(s, {
    x: Float64Array.from(pts), type: Int32Array.from(types), mask: gbit,
    radius: rad ? Float64Array.from(rad) : bodyAtoms ? new Float64Array(types.length).fill(0.5) : undefined, molecule: meshMol || undefined,
    rmass: bodyAtoms ? new Float64Array(types.length).fill(1.0) : undefined,
  });
  type = 0;
  sys.atomsChanged();
  sys.log(`Created ${added} atoms`);
};

/**
 * molecule ID file1 keyword values ... file2 ... — molecule.html: "zero or
 * more keyword/value pairs may be appended after each file"; "keyword =
 * offset or toff or boff or aoff or doff or ioff or scale". The ID "can only
 * contain alphanumeric characters and underscores".
 */
const molecule: Handler = ({ sys }, a) => {
  const id = a[0];
  // As native LAMMPS does (src/molecule.cpp Molecule::command at b332f28): "if (index >= narg) utils::missing_cmd_args(FLERR, \"molecule\", error)"
  if (a.length < 2) throw missingArgs('molecule');
  if (!id) throw new StyleError('usage: molecule ID file1 keyword values ... file2 ...');
  if (!/^[A-Za-z0-9_]+$/.test(id)) throw new StyleError(`molecule ID '${id}' must be alphanumeric or underscore`);
  if (sys.molecules.has(id)) throw new StyleError(`molecule template ID '${id}' already exists`);
  const sets: MoleculeTemplate[] = [];
  for (let k = 1; k < a.length;) {
    const file = a[k++];
    const o: MoleculeOptions = {
      toff: 0, boff: 0, aoff: 0, doff: 0, ioff: 0, scale: 1, units: sys.units.style,
      improperSymmatoms: improperSymmatoms(sys.ff.improper?.name),
      bodyStyle: isBodyStyle(sys.atomStyle),
    };
    let checkWhich: CheckWhich | null = null;
    while (k < a.length && ['offset', 'toff', 'boff', 'aoff', 'doff', 'ioff', 'scale', 'auto', 'check_labels'].includes(a[k])) {
      const key = a[k];
      // As native LAMMPS does (src/molecule.cpp Molecule::command at b332f28): each keyword checks its own arguments
      // first, "if (iarg + 6 > narg) utils::missing_cmd_args(FLERR, \"molecule offset\", error)" for offset and
      // "if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, \"molecule <key>\", error)" for the others
      if (k + (key === 'offset' ? 6 : 2) > a.length) throw missingArgs(key === 'check_labels' ? key : `molecule ${key}`);
      if (key === 'offset') {
        [o.toff, o.boff, o.aoff, o.doff, o.ioff] = [1, 2, 3, 4, 5].map((j) => int(a[k + j], 'offset'));
        k += 6;
      } else if (key === 'scale') { o.scale = num(a[k + 1], 'scale'); k += 2; }
      else if (key === 'auto') {
        // molecule.html: "The auto value is a single string that should contain one or more of the characters 'a', 'd', and 'i'"
        const v = a[k + 1] ?? '';
        if (!/^[adi]+$/.test(v)) throw new StyleError(`Illegal auto option ${v}`);
        o.auto = v;
        k += 2;
      } else if (key === 'check_labels') {
        // As native LAMMPS does (src/molecule.cpp at b332f28): the letters b, a, d, i; the template's type labels are
        // checked once it is read (labelcheck.ts moleculeLabelCheck)
        checkWhich = parseCheckWhich(a[k + 1], (c) => `Illegal check_labels option ${c}`);
        k += 2;
      } else { o[key as 'toff'] = int(a[k + 1], key); k += 2; }
    }
    // molecule.html: type words may be type labels, resolved through the label map at read time
    const t = parseMoleculeFile(id, file, sys.readFile(file, 'Cannot open molecule file'), o, sys.labels);
    if (checkWhich) moleculeLabelCheck(sys, sys.labels, checkWhich, id, t);
    // molecule.cpp (read_molecule): "Molecule file with lines must be for 2d simulation" and "... with triangles must be for 3d simulation"
    if (t.lines && sys.dimension !== 2) throw new StyleError('Molecule file with lines must be for 2d simulation');
    if (t.tris && sys.dimension !== 3) throw new StyleError('Molecule file with triangles must be for 3d simulation');
    sets.push(t);
    const surf = t.lines ? `, ${t.lines.length} lines` : t.tris ? `, ${t.tris.length} triangles` : '';
    sys.log(`Read molecule template ${id}: ${t.natoms} atoms, ${t.bonds.length} bonds, ${t.angles.length} angles, ${t.dihedrals.length} dihedrals, ${t.impropers.length} impropers${surf} (${file})`);
  }
  sys.molecules.set(id, sets);
};

/** mass I value — mass.html: "I can be specified ... as a wildcard"; "All masses must be defined before a simulation is run." */
const mass: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/input.cpp Input::mass and src/atom.cpp Atom::set_mass at b332f28)
  if (a.length !== 2) throw new StyleError(`Illegal mass command: expected 2 arguments but found ${a.length}`);
  if (!sys.hasBox) throw new StyleError('Mass command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  const s = sys.state;
  // Measured with native LAMMPS (black box): mass is refused for atom style sphere and for atom style ellipsoid
  // (Cannot set per-type atom mass for atom style sphere, and the same with ellipsoid)
  // Measured with native LAMMPS (black box): mass is refused for atom style peri too (Cannot set per-type atom mass for atom style peri)
  // Measured with native LAMMPS (black box): the same for bpm/sphere (Cannot set per-type atom mass for atom style bpm/sphere)
  if (s.atomStyle === 'sphere' || s.atomStyle === 'bpm/sphere' || s.atomStyle === 'ellipsoid' || s.atomStyle === 'peri' || isBodyStyle(s.atomStyle)) throw new StyleError(`Cannot set per-type atom mass for atom style ${s.atomStyle}`);
  // mass.html: "I can be a type label, which is an alphanumeric string defined by the labelmap command" (looked up only
  // when a labelmap is defined); the range is checked before the value
  const w = sys.labels.defined && !isNumericTypeWord(a[0]) ? typeWord(sys.labels, 'atom', a[0]) : a[0];
  const [lo, hi] = bounds(w, 1, s.ntypes);
  const m = numeric(a[1]);
  if (m <= 0) throw new StyleError(`Invalid atom mass value ${m} for type ${a[0]}`);
  for (let t = lo; t <= hi; t++) s.massByType[t] = m;
  sys.bump();
};

/** read_data file [keywords] — read_data.html; the keywords as native LAMMPS reads them (src/read_data.cpp command at b332f28). */
const readDataCmd: Handler = ({ sys }, a) => {
  if (a.length < 1) throw missingArgs('read_data');
  const o: ReadDataOptions = defaultReadOptions();
  const style = sys.atomStyle;
  const subs = atomSubStyles(style);
  const level = topologyLevel(style);
  const tmpl = subs.includes('template');
  const allow = [level >= 1 || tmpl || subs.includes('bpm/sphere'), level >= 2 || tmpl, level >= 3 || tmpl, level >= 3 || tmpl];
  // native atom->molecular == ATOMIC: no style with bonds
  const atomic = !allow[0];
  const hasMolecule = isMolecularStyle(style);
  let offsetflag = false;
  let shiftflag = false;
  const need = (k: number, n: number, what: string): void => { if (k + n > a.length) throw missingArgs(`read_data ${what}`); };
  const kinds = ['bond', 'angle', 'dihedral', 'improper'];
  for (let k = 1; k < a.length;) {
    const key = a[k];
    const xt = /^extra\/(atom|bond|angle|dihedral|improper)\/types$/.exec(key);
    const xp = /^extra\/(bond|angle|dihedral|improper|special)\/per\/atom$/.exec(key);
    if (key === 'add') {
      need(k, 2, 'add');
      if (a[k + 1] === 'append') o.add = 'append';
      else if (a[k + 1] === 'merge') o.add = 'merge';
      else {
        if (hasMolecule && k + 3 > a.length) throw missingArgs('read_data add');
        const id = bnumeric(a[k + 1]);
        if (id > MAXTAGINT) throw new StyleError(`Read data add IDoffset ${id} is too big`);
        let mol = 0;
        if (hasMolecule) {
          mol = bnumeric(a[k + 2]);
          if (mol > MAXTAGINT) throw new StyleError(`Read data add MOLoffset ${mol} is too big`);
          k++;
        }
        o.add = { id, mol };
      }
      k += 2;
    } else if (key === 'offset') {
      need(k, 6, 'offset');
      offsetflag = true;
      o.offset = [1, 2, 3, 4, 5].map((j) => inumeric(a[k + j])) as ReadDataOptions['offset'];
      if (o.offset.some((v) => v < 0)) throw new StyleError('Illegal read_data offset value(s)');
      k += 6;
    } else if (key === 'shift') {
      need(k, 4, 'shift');
      shiftflag = true;
      o.shift = [1, 2, 3].map((j) => numeric(a[k + j])) as ReadDataOptions['shift'];
      if (sys.dimension === 2 && o.shift[2] !== 0) throw new StyleError('Non-zero read_data shift z value for 2d simulation not allowed');
      k += 4;
    } else if (key === 'nocoeff') {
      o.nocoeff = true;
      k += 1;
    } else if (xt) {
      need(k, 2, key);
      const i = ['atom', ...kinds].indexOf(xt[1]);
      if (i > 0 && !allow[i - 1]) throw new StyleError(`No ${xt[1]}s allowed with atom style ${style}`);
      const v = inumeric(a[k + 1]);
      if (v < 0) throw new StyleError(`Illegal read_data ${key} value ${v}`);
      o.extraTypes[i] = v;
      k += 2;
    } else if (xp) {
      need(k, 2, key);
      if (atomic) throw new StyleError(xp[1] === 'special' ? `No bonded interactions allowed with atom style ${style}` : `No ${xp[1]}s allowed with atom style ${style}`);
      const v = inumeric(a[k + 1]);
      if (v < 0) throw new StyleError(`Illegal read_data ${key} value ${v}`);
      // read_data.html: extra/bond/per/atom sets the bonds per atom the storage holds (bonds only are enforced; the
      // angle, dihedral, improper and special limits are read and not enforced: audit-setup.md RD-1)
      if (xp[1] === 'bond') o.extraBondPerAtom = v;
      k += 2;
    } else if (key === 'group') {
      need(k, 2, 'group');
      o.group = a[k + 1];
      k += 2;
    } else if (key === 'fix') {
      need(k, 4, 'fix');
      const [fid, header, sec] = [a[k + 1], a[k + 2], a[k + 3]];
      if (DATA_SECTION_NAMES.has(sec)) throw new StyleError(`Custom data section name ${sec} for fix ${fid} collides with existing data section`);
      if (!sys.fixes.some((f) => f.id === fid)) throw new StyleError(`Fix ID ${fid} for read_data does not exist`);
      if (header !== 'NULL') {
        // read_data.html: "header lines containing this string will be passed to fix"; only a fix that reads
        // header lines (fix cmap: "N crossterms") accepts a header-string other than NULL
        const fx = sys.fix(fid) as unknown as { style: string; readHeader?: unknown };
        if (typeof fx.readHeader !== 'function') throw new StyleError(`read_data fix ${fid}: header-string must be NULL (fix ${fx.style} reads no header lines)`);
      }
      o.fixSections.set(sec === 'NULL' ? fid : sec, { fixId: fid, header });
      k += 4;
    } else throw new StyleError(`Unknown read_data keyword ${key}`);
  }
  if (sys.dimension === 2 && sys.boundary[2][0] !== 'p') throw new StyleError('Cannot run 2d simulation with nonperiodic Z dimension');
  if (sys.hasBox && o.add === 'none') throw new StyleError(`Cannot use read_data without add keyword after simulation box is defined${errorUrl(34)}`);
  if (!sys.hasBox && o.add !== 'none') throw new StyleError(`Cannot use read_data add before simulation box is defined${errorUrl(33)}`);
  if (offsetflag && o.add === 'none') throw new StyleError('Cannot use read_data offset without add keyword');
  if (offsetflag && sys.labels.defined) sys.warn('Using read_data offset with a labelmap. Offsets will be only applied to numeric types and not to type labels');
  if (shiftflag && o.add === 'none') throw new StyleError('Cannot use read_data shift without add keyword');
  if (o.add !== 'none' && o.extraTypes.some((v) => v)) throw new StyleError('Cannot use any read_data extra/*/types keyword with add keyword');
  // the group is made when the keyword is read (native group->find_or_create)
  if (o.group) sys.groups.create(o.group);
  const text = sys.readFile(a[0]);
  readData(sys, text, o);
  const s = sys.state;
  sys.log(`Read ${s.n} atoms${s.topo.bonds.n ? `, ${s.topo.bonds.n} bonds` : ''}${s.topo.angles.n ? `, ${s.topo.angles.n} angles` : ''}${s.topo.dihedrals.n ? `, ${s.topo.dihedrals.n} dihedrals` : ''}${s.topo.impropers.n ? `, ${s.topo.impropers.n} impropers` : ''} from ${a[0]}`);
};

/**
 * labelmap option args — labelmap.html: "atom or bond or angle or dihedral or improper ... args = list of one or
 * more numeric-type/type-label pairs", "clear = no args", "write arg = filename". The check_labels keyword only
 * warns about labels that disagree with their atom types, which the browser engine does not do, so it is refused.
 */
const labelmapCmd: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/label_map.cpp LabelMap::modify_lmap at b332f28): "if ((narg < 1) || ((narg > 2) && ((narg % 2) == 0)))"
  if (a.length < 1 || (a.length > 2 && a.length % 2 === 0)) throw new StyleError('Incorrect number of arguments for labelmap command');
  const opt = a[0];
  if (opt === 'clear') {
    if (a.length !== 1) throw new StyleError('Incorrect number of arguments for labelmap clear command');
    sys.labels.clear();
    return;
  }
  if (opt === 'write') {
    if (a.length !== 2) throw new StyleError('Incorrect number of arguments for labelmap write command');
    sys.writeFile(a[1], labelmapText(sys.labels), false);
    return;
  }
  if (opt === 'check_labels') {
    // As native LAMMPS does (src/label_map.cpp at b332f28): one option word of the letters b, a, d, i; the check runs at
    // the setup of the next run or minimization (labelcheck.ts runLabelCheck)
    if (a.length !== 2) throw new StyleError('Incorrect number of arguments for labelmap check_labels command');
    sys.labels.checkWhich = parseCheckWhich(a[1], (c) => `Labelmap command: Illegal check_labels option ${c}`);
    return;
  }
  if (!isLabelKind(opt)) throw new StyleError(`Unknown labelmap keyword ${opt}`);
  if (!sys.hasBox) throw new StyleError('Labelmap command before simulation box is defined');
  const s = sys.state;
  const n = opt === 'atom' ? s.ntypes : opt === 'bond' ? s.topo.nbondtypes : opt === 'angle' ? s.topo.nangletypes : opt === 'dihedral' ? s.topo.ndihedraltypes : s.topo.nimpropertypes;
  for (const [t, label] of parseLabelPairs(a.slice(1), opt, n)) sys.labels.set(opt, t, label);
  sys.bump();
};

/** write_data file [nocoeff] [pair ii|ij] [nofix] [nolabelmap] [types numeric|labels] — write_data.html. */
const writeDataCmd: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/write_data.cpp WriteData::command at b332f28): a box, then the file name
  if (!sys.hasBox) throw new StyleError(`Write_data command before simulation box is defined${errorUrl(33)}`);
  if (!a[0]) throw missingArgs('write_data');
  let nocoeff = false, nofix = false, triclinicGeneral = false, nolabelmap = false, typesLabels = false;
  let pairStyle: 'ii' | 'ij' | null = null;
  for (let k = 1; k < a.length;) {
    // noinit (undocumented; src/write_data.cpp at b332f28 then skips lmp->init() and the remap of the atoms into the box)
    // is refused by name
    if (a[k] === 'noinit') throw new StyleError('write_data noinit is not supported by the browser engine (native skips the system init and the remap of atoms into the box)');
    if (a[k] === 'nocoeff') { nocoeff = true; k++; } else if (a[k] === 'nofix') { nofix = true; k++; } else if (a[k] === 'nolabelmap') { nolabelmap = true; k++; }
    else if (a[k] === 'pair') {
      // As native LAMMPS does (src/write_data.cpp WriteData::command at b332f28)
      if (a[k + 1] === undefined) throw missingArgs('write_data pair');
      if (a[k + 1] !== 'ii' && a[k + 1] !== 'ij') throw new StyleError(`Unknown write_data pair option: ${a[k + 1]}`);
      pairStyle = a[k + 1] as 'ii' | 'ij';
      k += 2;
    } else if (a[k] === 'types') {
      // write_data.html: "*types* value = *numeric* or *labels*"
      if (a[k + 1] === undefined) throw missingArgs('write_data types');
      if (a[k + 1] !== 'numeric' && a[k + 1] !== 'labels') throw new StyleError(`Unknown write_data types option: ${a[k + 1]}`);
      typesLabels = a[k + 1] === 'labels';
      k += 2;
    } else if (a[k] === 'triclinic/general') {
      // write_data.html: "*triclinic/general* = write data file in general triclinic format"
      triclinicGeneral = true;
      k++;
    }
    else throw new StyleError(`Unknown write_data keyword: ${a[k]}`);
  }
  // the box must be current (shrink-wrapped faces, remapped atoms): write_data "calls ... pbc" via a setup.
  // Measured with native LAMMPS: atoms that drifted out of a periodic box since the last reneighboring
  // are written wrapped back in, so remap them even when forces are current. Measured likewise: the
  // forces are not recomputed (fx of an atom after a fix langevin run is unchanged by write_data,
  // random force included), so the wrapped state keeps the forces of the last run.
  // write_data initializes the system first (native lmp->init(), src/write_data.cpp at b332f28)
  sys.markComputesInitialized();
  const wasCurrent = sys.hasForces;
  sys.pbc();
  sys.nb.lastBuild = -1;
  sys.bump();
  if (wasCurrent) sys.forcesCurrent(); else sys.forces();
  // write_data.html: "The file may also be a compressed text file (detected by its suffix)"; the browser
  // engine writes stored (uncompressed) gzip members, see gzip.ts.
  const text = writeData(sys, { nocoeff, pairStyle, nofix, triclinicGeneral, nolabelmap, typesLabels });
  sys.writeFile(a[0], a[0].endsWith('.gz') ? gzipText(text) : text, false);
  sys.log(`Wrote ${sys.state.n} atoms to ${a[0]}`);
};

/** timestep dt — timestep.html; also keeps the elapsed time ("time") cumulative. */
const timestep: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/input.cpp Input::timestep at b332f28): one number, which is not range-checked (timestep
  // 0 runs without motion, and a negative step runs backwards; measured with native LAMMPS: both run)
  if (a.length !== 1) throw new StyleError('Timestep command expects exactly one argument');
  const dt = numeric(a[0]);
  if (!sys.hasBox) { sys.pendingDt = dt; return; }
  const s = sys.state;
  s.time += (s.step - s.timeStep) * s.dt;
  s.timeStep = s.step;
  s.dt = dt;
  // native resets the fixes' step size only once a run has happened (Input::timestep: update->first_update)
  if (sys.run.ranOnce) for (const f of sys.fixes) f.resetDt?.();
};

/** reset_timestep N [time T] — reset_timestep.html. */
/** The fix styles native marks time-dependent (time_depend = 1 at b332f28): reset_timestep refuses them. */
const TIME_DEPEND_FIXES = new Set([
  'ave/correlate', 'ave/correlate/long', 'ave/time', 'ave/grid', 'ave/histo', 'ave/histo/weight', 'ave/atom', 'ave/chunk', 'ave/moments',
  'vector', 'dt/reset', 'deposit', 'pour', 'ti/spring', 'widom', 'atom/swap', 'gcmc', 'gemc', 'charge/regulation', 'lb/fluid',
  'neighbor/swap', 'mol/swap', 'alchemy', 'smd/adjust_dt', 'append/atoms', 'graphics/lines',
]);

const resetTimestep: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/update.cpp Update::reset_timestep at b332f28): the step, then the time keyword
  if (a.length < 1) throw missingArgs('reset_timestep');
  const s = sys.state;
  const n = bnumeric(a[0]);
  if (n < 0) throw new StyleError('Timestep must be >= 0');
  let t: number | null = null;
  if (a.length > 1) {
    if (a[1] !== 'time') throw new StyleError(`Unknown reset_timestep option ${a[1]}`);
    if (a.length < 3) throw missingArgs('reset_timestep time');
    t = numeric(a[2]);
  }
  sys.checkTimestepReset();
  for (const f of sys.fixes) {
    if (TIME_DEPEND_FIXES.has(f.style)) throw new StyleError(`Cannot reset timestep with time-dependent fix ${f.style} defined`);
  }
  // no energy is current after the reset (native resets update->eflag_global)
  sys.run.eflagStep = -1;
  // reset_timestep.html: "The optional time keyword allows to also set the accumulated simulation time." Without the
  // time keyword the clock is not restarted
  // unless the step goes down: measured with native LAMMPS (black box, LJ 32 atoms, dt 0.005): after run 2
  // (time 0.01), reset_timestep 100 and run 1 give step 100 time 0.5 and step 101 time 0.505; reset_timestep 50
  // (below step 101) gives time 0 at step 50; reset_timestep 7 time 3.5 then reset_timestep 10 gives time 3.515.
  if (t !== null) { s.time = t; s.timeStep = n; } else if (n < s.step) { s.time = 0; s.timeStep = n; } else {
    // the time of the current step carries on from its anchor (timeStep) to the new step
    s.time += (s.step - s.timeStep) * s.dt;
    s.timeStep = s.step;
  }
  s.step = n;
  sys.nb.lastBuild = -1;
};

// ----------------------------------------------------------------- groups

/**
 * One entry of a group type/id/molecule list, read as native LAMMPS does (src/group.cpp Group::assign at b332f28): a
 * value, A:B or A:B:C, each an integer ("Incorrect range string 'w': Not a valid integer number: 'x'" or "...: Syntax
 * error"), with a positive increment.
 */
const parseList = (w: string): [number, number, number] => {
  const bad = (why: string) => new StyleError(`Incorrect range string '${w}': ${why}`);
  const parts = w.split(':');
  const int64 = (t: string): number => {
    // std::stoll of the whole token
    if (!/^\s*[+-]?\d+$/.test(t)) throw bad(`Not a valid integer number: '${t}'`);
    return Number(t);
  };
  const start = int64(parts[0]);
  let stop: number, delta = 1;
  if (/^-?\d+$/.test(w)) stop = start;
  else if (/^-?\d+:-?\d+$/.test(w)) stop = int64(parts[1]);
  else if (/^-?\d+:-?\d+:\d+$/.test(w)) { stop = int64(parts[1]); delta = int64(parts[2]); } else throw bad('Syntax error');
  if (delta < 1) throw new StyleError('Illegal range increment value');
  return [start, stop, delta];
};

/** group ID style args — group.html. As native LAMMPS does (src/group.cpp Group::assign at b332f28). */
const group: Handler = ({ sys }, a) => {
  if (!sys.hasBox) throw new StyleError('Group command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  if (a.length < 2) throw missingArgs('group');
  const [id, style] = a;
  if (style === 'delete') {
    if (a.length !== 2) throw new StyleError('Illegal group delete command: too many arguments');
    const k0 = sys.groups.find(id);
    if (k0 < 0) throw new StyleError(`Could not find group delete group ID ${id}`);
    if (k0 === 0) throw new StyleError('Cannot delete group all');
    // a group in use cannot be deleted (its bit could be given to a new group)
    const fixUser = sys.fixes.find((f) => f.group === id);
    if (fixUser) throw new StyleError(`Cannot delete group ${id} currently used by fix ID ${fixUser.id}`);
    const computeUser = sys.computes.find((c) => c.group === id);
    if (computeUser) throw new StyleError(`Cannot delete group ${id} currently used by compute ID ${computeUser.id}`);
    const dumpUser = sys.dumps.find((d) => d.group === id);
    if (dumpUser) throw new StyleError(`Cannot delete group ${id} currently used by dump ID ${dumpUser.id}`);
    if (sys.firstGroup === id) throw new StyleError(`Cannot delete group ${id} currently used by atom_modify first`);
    const k = sys.groups.delete(id);
    const s = sys.state;
    const m = ~bitOfIndex(k);
    for (let i = 0; i < s.n; i++) s.mask[i] &= m;
    return;
  }
  const s = sys.state;
  const args = a.slice(2);
  if (style === 'clear') {
    // Measured with native LAMMPS (black box): group clear needs an existing group and logs nothing
    const k0 = sys.groups.find(id);
    if (k0 < 0) throw new StyleError(`Could not find group clear group ID ${id}`);
    if (k0 === 0) throw new StyleError('Cannot clear group all');
    const m = ~bitOfIndex(k0);
    for (let i = 0; i < s.n; i++) s.mask[i] &= m;
    sys.refreshComputes();
    return;
  }
  if (style === 'dynamic') {
    groupDynamic(sys, id, args);
    return;
  }
  // native makes the group (if new) before it reads the style's arguments
  const k = sys.groups.create(id);
  const bit = bitOfIndex(k);
  const set = (pred: (i: number) => boolean) => { for (let i = 0; i < s.n; i++) if (pred(i)) s.mask[i] |= bit; };
  switch (style) {
    case 'static':
      // Measured with native LAMMPS (black box): group static on a group that is not dynamic only reports its count
      if (a.length !== 2) throw new StyleError('Illegal group static command');
      sys.groups.dynamic.delete(bit);
      break;
    case 'empty':
      if (a.length !== 2) throw new StyleError('Illegal group empty command');
      break;
    case 'region': {
      if (a.length !== 3) throw new StyleError('Illegal group region command');
      if (!sys.regions.has(args[0])) throw new StyleError(`Region ${args[0]} for group region does not exist`);
      const r = sys.region(args[0]);
      set((i) => r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]));
      break;
    }
    case 'type': case 'id': case 'molecule': {
      if (a.length < 3) throw missingArgs(`group ${style}`);
      if (style === 'molecule' && !hasMolecule(s)) throw new StyleError('Group molecule command requires atom attribute molecule');
      const val = (i: number) => (style === 'type' ? s.type[i] : style === 'id' ? s.id[i] : s.molecule[i]);
      // group.html: for style type "type labels are converted into numeric types before being evaluated" (only when a
      // labelmap is defined; otherwise a word is read as a number)
      const typeOf = (w: string): string => (style === 'type' && sys.labels.defined && !isNumericTypeWord(w) ? typeWord(sys.labels, 'atom', w) : w);
      if (a.length > 3 && LOGICAL_OPS.includes(args[0])) {
        const op = args[0];
        const bound = (w: string) => (style === 'type' ? inumeric(typeOf(w)) : bnumeric(w));
        const v1 = bound(args[1]);
        let v2 = -1;
        if (op === '<>') {
          if (a.length !== 5) throw new StyleError('Illegal group command');
          v2 = bound(args[2]);
        } else if (a.length !== 4) throw new StyleError('Illegal group command');
        set((i) => {
          const v = val(i);
          switch (op) {
            case '<': return v < v1; case '<=': return v <= v1; case '>': return v > v1; case '>=': return v >= v1;
            case '==': return v === v1; case '!=': return v !== v1; default: return v >= v1 && v <= v2;
          }
        });
      } else {
        for (const w of args) {
          const t = typeOf(w);
          const [lo, hi, st] = t !== w ? [Number(t), Number(t), 1] : parseList(w);
          set((i) => { const v = val(i); return v >= lo && v <= hi && (v - lo) % st === 0; });
        }
      }
      break;
    }
    case 'variable': {
      // As native LAMMPS does (src/group.cpp Group::assign at b332f28): with no name the lookup fails and formatting the
      // null name into "Variable name {} for group does not exist" stops with fmt's "string pointer is null"
      if (args[0] === undefined) throw new StyleError('string pointer is null');
      const v0 = sys.vars.get(args[0]);
      if (!v0) throw new StyleError(`Variable name ${args[0]} for group does not exist`);
      if (v0.style !== 'atom' && v0.style !== 'atomfile') throw new StyleError(`Variable ${args[0]} for group is invalid style`);
      const v = sys.atomVariable(args[0]);
      set((i) => v[i] !== 0);
      break;
    }
    case 'include': {
      if (a.length !== 3) throw new StyleError('Illegal group include command');
      if (args[0] !== 'molecule') throw new StyleError(`Unknown group include keyword ${args[0]}`);
      if (!hasMolecule(s)) throw new StyleError('Group include molecule command requires atom attribute molecule');
      const mols = new Set<number>();
      for (let i = 0; i < s.n; i++) if (s.mask[i] & bit && s.molecule[i] !== 0) mols.add(s.molecule[i]);
      set((i) => mols.has(s.molecule[i]));
      break;
    }
    case 'subtract': case 'union': case 'intersect': {
      if (a.length < (style === 'union' ? 3 : 4)) throw missingArgs(`group ${style}`);
      const bits = args.map((g) => {
        if (sys.groups.find(g) < 0) throw new StyleError(`Group ID ${g} does not exist`);
        const b = sys.groupBit(g);
        // Measured with native LAMMPS (black box): a dynamic group cannot be combined
        if (sys.groups.isDynamic(b)) {
          throw new StyleError(style === 'subtract' ? 'Cannot subtract dynamic groups'
            : style === 'union' ? 'Cannot union groups from a dynamic group' : 'Cannot intersect groups using a dynamic group');
        }
        return b;
      });
      set((i) => {
        const m = s.mask[i];
        if (style === 'union') return bits.some((b) => (m & b) !== 0);
        if (style === 'intersect') return bits.every((b) => (m & b) !== 0);
        return (m & bits[0]) !== 0 && bits.slice(1).every((b) => (m & b) === 0);
      });
      break;
    }
    default: throw new StyleError(`Unknown group command keyword: ${style}`);
  }
  sys.refreshComputes();
  // Measured with native LAMMPS (black box): a dynamic group reports "dynamic group ID defined"
  // after any group command instead of its count
  if (sys.groups.isDynamic(bit)) { sys.log(`dynamic group ${id} defined`); return; }
  let count = 0;
  for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) count++;
  sys.log(`${count} atoms in group ${id}`);
};

/*
 * group ID dynamic parent-ID keyword value ... — docs.lammps.org/group.html:
 *   *dynamic* args = parent-ID keyword value ...
 *     keyword = *region* or *var* or *property* or *every* or *within* or *include* or *exclude*
 * "A group with the ID all is predefined. All atoms belong to this group.
 * This group cannot be deleted, or made dynamic." "If the *var* keyword is
 * used, the variable name must be an atom-style or atomfile-style variable."
 * "Note that the name of the custom per-atom vector is specified just as
 * *name*, not as *i_name* or *d_name*".
 * "The optional *within* keyword with its *cutoff* argument adds atoms to a
 * dynamic group that are within the given cutoff distance from any of the
 * atoms selected for the dynamic group."
 * "The optional *include* keyword with its argument *molecule* adds atoms
 * to a dynamic group that have the same molecule ID as atoms already in the
 * group." "Atoms with the molecule ID = 0 are ignored in this operation"
 * "The optional *exclude* keyword has a group ID as argument and after all
 * other atom selections for the dynamic group have been performed, it
 * removes selected atoms that are in the specified group from the dynamic
 * group."
 * Measured with native LAMMPS (black box): the group keeps its atoms until the
 * next run assigns them; the command logs "dynamic group ID defined"; a
 * missing region, variable or property and a non-positive every stop at the
 * command, the variable style and a dynamic parent at the next run; a within
 * cutoff of 0 or below stops at the command (Illegal within cutoff value 0
 * for dynamic group d); a non-numeric within value stops with the input
 * parser's floating point error; a missing argument of within, include or
 * exclude stops at the command; include with anything other than molecule
 * stops with Unknown include setting atom in dynamic group command; a
 * repeated within keeps the last cutoff; the error texts below are native's.
 */
const groupDynamic = (sys: System, id: string, args: string[]): void => {
  const parent = args[0];
  if (!parent || args.length < 2) throw new StyleError('Illegal group command: usage group ID dynamic parent-ID keyword value ...');
  if (parent === id) throw new StyleError('Group dynamic cannot reference itself');
  if (sys.groups.find(parent) < 0) throw new StyleError(`Group dynamic parent group ${parent} does not exist`);
  if (id === 'all') throw new StyleError('Group all cannot be made dynamic');
  const g: DynamicGroup = {
    parent, region: null, variable: null, property: null, every: 1, within: null, includeMolecule: false, exclude: null,
  };
  for (let k = 1; k < args.length; k += 2) {
    const key = args[k], val = args[k + 1];
    if (!['region', 'var', 'property', 'every', 'within', 'include', 'exclude'].includes(key)) throw new StyleError(`Unknown keyword ${key} in dynamic group command`);
    if (key === 'within') {
      if (val === undefined) throw new StyleError('Illegal group dynamic within command: missing argument(s)');
      const c = Number(val);
      if (val.trim() === '' || !Number.isFinite(c)) throw new StyleError(`Expected floating point parameter instead of '${val}' in input script or data file`);
      if (c <= 0) throw new StyleError(`Illegal within cutoff value ${c} for dynamic group ${id}`);
      g.within = c;
      continue;
    }
    if (key === 'include') {
      if (val === undefined) throw new StyleError('Illegal group dynamic include command: missing argument(s)');
      if (val !== 'molecule') throw new StyleError(`Unknown include setting ${val} in dynamic group command`);
      g.includeMolecule = true;
      continue;
    }
    if (key === 'exclude') {
      if (val === undefined) throw new StyleError('Illegal group dynamic exclude command: missing argument(s)');
      g.exclude = val;
      continue;
    }
    // As native LAMMPS does (src/fix_group.cpp FixGroup::FixGroup at b332f28)
    if (val === undefined) throw missingArgs(`group dynamic ${key}`);
    if (key === 'region') {
      if (!sys.regions.has(val)) throw new StyleError(`Region ${val} for dynamic group ${id} does not exist`);
      g.region = val;
    } else if (key === 'var') {
      if (!sys.vars.has(val)) throw new StyleError(`Variable '${val}' for dynamic group ${id} does not exist`);
      g.variable = val;
    } else if (key === 'property') {
      const c = sys.state.custom.get(val);
      if (!c || c.cols !== 0) throw new StyleError(`Custom per-atom vector ${val} for dynamic group ${id} does not exist`);
      g.property = val;
    } else {
      const n = inumeric(val);
      if (n <= 0) throw new StyleError(`Illegal every value ${n} for dynamic group ${id}`);
      g.every = n;
    }
  }
  sys.groups.dynamic.set(bitOfIndex(sys.groups.create(id)), g);
  sys.log(`dynamic group ${id} defined`);
};

// ----------------------------------------------------------------- set

/** A type word given as a keyword value (set type, set bond ...): a label becomes its numeric type; a missing word stays missing. */
const typeLabelWord = (sys: System, kind: LabelKind, w: string | undefined): string => (w === undefined ? '' : w.startsWith('v_') ? w : typeWord(sys.labels, kind, w));

/**
 * The atoms a set command selects, as native LAMMPS does (src/set.cpp Set::process_args and selection at b332f28):
 * atom and mol take a range of IDs (utils::bounds over 1..MAXTAGINT, so IDs no atom has select nothing), type a
 * range of types or a type label, group and region an existing ID.
 */
const selected = (sys: System, style: string, id: string): number[] => {
  const s = sys.state;
  const out: number[] = [];
  const MAXTAGINT = 0x7fffffff;
  if (style === 'atom') {
    const [lo, hi] = bounds(id, 1, MAXTAGINT);
    for (let i = 0; i < s.n; i++) if (s.id[i] >= lo && s.id[i] <= hi) out.push(i);
  } else if (style === 'mol') {
    if (!hasMolecule(s)) throw new StyleError('Cannot use set mol with no molecule IDs defined');
    const [lo, hi] = bounds(id, 1, MAXTAGINT);
    for (let i = 0; i < s.n; i++) if (s.molecule[i] >= lo && s.molecule[i] <= hi) out.push(i);
  } else if (style === 'type') {
    // set.html: "for style = type, ID = a range of numeric types or a single type label"
    const w = sys.labels.defined && !isNumericTypeWord(id) ? typeWord(sys.labels, 'atom', id) : id;
    const [lo, hi] = bounds(w, 1, s.ntypes);
    for (let i = 0; i < s.n; i++) if (s.type[i] >= lo && s.type[i] <= hi) out.push(i);
  } else if (style === 'group') {
    if (sys.groups.find(id) < 0) throw new StyleError(`Could not find set group ID ${id}`);
    const bit = sys.groupBit(id);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) out.push(i);
  } else if (style === 'region') {
    if (!sys.regions.has(id)) throw new StyleError(`Set region ${id} does not exist`);
    const r = sys.region(id);
    for (let i = 0; i < s.n; i++) if (r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) out.push(i);
  } else throw new StyleError(`Unknown set or fix set command style: ${style}`);
  return out;
};

/** The values each set keyword takes (src/set.cpp process_* at b332f28); a custom i_/d_ name takes one. */
const SET_NVALUES: Record<string, number> = {
  angle: 1, angmom: 3, 'apip/lambda': 1, block: 2, bond: 1, cc: 2, charge: 1, density: 1, 'density/disc': 1, diameter: 1,
  dihedral: 1, dipole: 3, 'dipole/random': 2, 'dpd/theta': 1, 'edpd/cv': 1, 'edpd/temp': 1, epsilon: 1, image: 3, improper: 1,
  length: 1, mass: 1, mol: 1, omega: 3, quat: 4, 'quat/random': 1, 'radius/electron': 1, 'rheo/rho': 1, 'rheo/status': 1,
  shape: 3, 'smd/contact/radius': 1, 'smd/mass/density': 1, 'sph/cv': 1, 'sph/e': 1, 'sph/rho': 1, 'spin/atom': 4, spin: 4,
  'spin/atom/random': 2, 'spin/random': 2, 'spin/electron': 1, temperature: 1, theta: 1, 'theta/random': 1, tri: 1, type: 1,
  'type/fraction': 3, 'type/ratio': 3, 'type/subset': 3, volume: 1, vx: 1, vy: 1, vz: 1, x: 1, y: 1, z: 1,
};

/**
 * How native's process_* function reads each value of a set keyword (src/set.cpp at b332f28), one letter per value:
 * i an integer (utils::inumeric), n a float (utils::numeric), b a 64-bit integer (utils::bnumeric or tnumeric); an
 * upper-case letter also takes a v_name (varparse); V is a v_name or a word read otherwise (a type), - a word read
 * otherwise. Keywords not listed (angle, bond, dihedral, improper, type) read type words.
 */
const SET_READERS: Record<string, string> = {
  angmom: 'NNN', 'apip/lambda': 'N', block: 'NN', cc: 'iN', charge: 'N', density: 'N', 'density/disc': 'N', diameter: 'N',
  dipole: 'NNN', 'dipole/random': 'in', 'dpd/theta': 'N', 'edpd/cv': 'N', 'edpd/temp': 'N', epsilon: 'N', image: 'III', length: 'N',
  mass: 'N', mol: 'B', omega: 'NNN', quat: 'NNNN', 'quat/random': 'i', 'radius/electron': 'N', 'rheo/rho': 'N', 'rheo/status': 'I',
  shape: 'NNN', 'smd/contact/radius': 'N', 'smd/mass/density': 'N', 'sph/cv': 'N', 'sph/e': 'N', 'sph/rho': 'N', spin: 'NNNN',
  'spin/atom': 'NNNN', 'spin/atom/random': 'in', 'spin/random': 'in', 'spin/electron': 'I', temperature: 'N', theta: 'N',
  'theta/random': 'i', tri: 'N', 'type/fraction': '-ni', 'type/ratio': '-ni', 'type/subset': '-bi', volume: 'N', vx: 'N', vy: 'N', vz: 'N',
  x: 'N', y: 'N', z: 'N',
};

/**
 * The checks native LAMMPS makes on a set command's keywords before it changes any atom (src/set.cpp
 * Set::process_args at b332f28): each keyword is known and has its values, a v_name value is an atom-style variable,
 * and the type keywords name a valid type, fraction, subset size and seed.
 */
const checkSetKeywords = (sys: System, a: string[]): void => {
  const s = sys.state;
  const typeArg = (w: string): number => {
    const t = inumeric(sys.labels.defined && !isNumericTypeWord(w) ? typeWord(sys.labels, 'atom', w) : w);
    if (t <= 0 || t > s.ntypes) throw new StyleError('Invalid atom type in set command');
    return t;
  };
  // the per-atom attribute a keyword needs, checked before its values as native's process_* functions do
  const has: Record<string, () => boolean> = {
    charge: () => hasCharge(s), mass: () => !!s.rmass, density: () => !!s.rmass, 'density/disc': () => !!s.rmass,
    diameter: () => !!s.radius, dipole: () => !!s.mu, 'dipole/random': () => !!s.mu, omega: () => !!s.omega,
    angmom: () => !!s.angmom, mol: () => hasMolecule(s), volume: () => !!s.vfrac,
  };
  for (let k = 2; k < a.length;) {
    const key = a[k];
    const custom = /^[id]2?_/.test(key);
    const n = custom ? 1 : SET_NVALUES[key];
    if (n === undefined) throw new StyleError(`Unrecognized set or fix set command keyword ${key}`);
    if (has[key] && !has[key]()) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle}`);
    // the message names the keyword as native's process_* function does (src/set.cpp at b332f28): density/disc is
    // density's, and the old spin and spin/random names are spin/atom's and spin/atom/random's
    const named: Record<string, string> = { 'density/disc': 'density', spin: 'spin/atom', 'spin/random': 'spin/atom/random' };
    if (k + 1 + n > a.length) throw missingArgs(custom ? 'set' : `set ${named[key] ?? key}`);
    // each value in turn, as native reads it: a v_name where the keyword takes one, else the number (SET_READERS); a
    // custom i_ name takes an integer, a d_ name a float; image and dpd/theta take NULL, apip/lambda fast and precise
    const readers = custom ? (key[0] === 'i' ? 'I' : 'N') : SET_READERS[key] ?? '';
    for (let p = 0; p < n; p++) {
      const w = a[k + 1 + p];
      const r = readers[p] ?? 'V';
      if (w.startsWith('v_') && r !== r.toLowerCase()) {
        const v = sys.vars.get(w.slice(2));
        if (!v) throw new StyleError(`Variable name ${w} for set command does not exist`);
        if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Variable ${w} for set command is invalid style`);
        continue;
      }
      if (((key === 'image' || key === 'dpd/theta') && w === 'NULL') || (key === 'apip/lambda' && (w === 'fast' || w === 'precise'))) continue;
      const kind = r.toLowerCase();
      if (kind === 'i') inumeric(w);
      else if (kind === 'n') numeric(w);
      else if (kind === 'b') bnumeric(w);
    }
    if (key === 'type' && !a[k + 1].startsWith('v_')) typeArg(a[k + 1]);
    if (key === 'type/fraction' || key === 'type/ratio' || key === 'type/subset') {
      typeArg(a[k + 1]);
      if (key === 'type/subset') {
        if (bnumeric(a[k + 2]) < 0) throw new StyleError('Invalid subset size in set command');
      } else {
        const f = numeric(a[k + 2]);
        if (f < 0 || f > 1) throw new StyleError('Invalid fraction in set command');
      }
      if (inumeric(a[k + 3]) <= 0) throw new StyleError('Invalid random number seed in set command');
    }
    k += 1 + n;
  }
};

/**
 * The random keywords of set (type/fraction, type/ratio, type/subset, dipole/random, quat/random, spin/random,
 * spin/atom/random): as native LAMMPS does (src/set.cpp setrandom at b332f28), each one also builds a Marsaglia
 * generator from its seed, so a seed above 900000000 stops even when the keyword's own draws use another generator.
 * Measured with native LAMMPS: set group all type/fraction 2 0.5 900000001 (and dipole/random, quat/random,
 * spin/atom/random) stops with Invalid seed 900000001 for the Marsaglia random # generator; 900000000 is accepted.
 */
const marsagliaSeed = (seed: number): void => {
  if (seed > 900000000) throw new StyleError(`Invalid seed ${seed} for the Marsaglia random # generator`);
};

/**
 * set style ID keyword values — set.html. Values may be atom-style variables
 * (v_name). type/fraction, type/ratio and type/subset choose atoms with the
 * per-atom and subset streams of native LAMMPS (nativeRandom.ts).
 */
const set: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/set.cpp Set::command at b332f28): the box and atoms, the selection and every keyword are
  // checked before any atom is changed
  if (!sys.hasBox) throw new StyleError('Set command before simulation box is defined\nFor more information see https://docs.lammps.org/Errors_details.html');
  const s = sys.state;
  if (s.n === 0) throw new StyleError('Set command on system without atoms');
  if (a.length < 4) throw missingArgs('set');
  const atoms = selected(sys, a[0], a[1]);
  checkSetKeywords(sys, a);
  const value = (w: string | undefined, what: string): ((i: number) => number) => {
    if (w?.startsWith('v_')) {
      const v = sys.atomVariable(w.slice(2));
      return (i) => v[i];
    }
    const x = num(w, what);
    return () => x;
  };
  let changed = 0;
  for (let k = 2; k < a.length;) {
    const key = a[k];
    switch (key) {
      case 'type': {
        const v = value(typeLabelWord(sys, 'atom', a[k + 1]), 'type');
        for (const i of atoms) {
          const t = Math.trunc(v(i));
          if (t < 1 || t > s.ntypes) throw new StyleError('Invalid atom type in set command');
          s.type[i] = t;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'type/fraction': case 'type/ratio': case 'type/subset': {
        const t = int(typeLabelWord(sys, 'atom', a[k + 1]), 'type');
        const amount = num(a[k + 2], key);
        const seed = int(a[k + 3], 'seed');
        if (t < 1 || t > s.ntypes) throw new StyleError(`set type ${t} is outside 1..${s.ntypes}`);
        marsagliaSeed(seed);
        const rnd = nativeRandom(key);
        if (key === 'type/fraction') {
          // set.html: "Random numbers are used in such a way that a particular atom is changed or not changed,
          // regardless of how many processors are being used."
          for (const i of atoms) {
            if (rnd.typeFraction(seed, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], amount)) { s.type[i] = t; changed++; }
          }
        } else {
          // Measured with native LAMMPS (black box): type/ratio 0.5 on 27 selected atoms changes 13 (the product is
          // truncated); type/subset changes exactly Nsubset.
          const want = key === 'type/ratio' ? Math.trunc(amount * atoms.length) : Math.trunc(amount);
          if (want > atoms.length) throw new StyleError('Set type/subset value exceeds eligible atoms');
          const flag = rnd.selectSubset(seed, want, atoms.length);
          for (let e = 0; e < atoms.length; e++) if (flag[e]) s.type[atoms[e]] = t;
          changed = want;
        }
        k += 4;
        break;
      }
      case 'mol': {
        if (!hasMolecule(s)) throw new StyleError(`Cannot set attribute mol for atom style ${s.atomStyle}`);
        const v = value(a[k + 1], 'mol');
        for (const i of atoms) s.molecule[i] = Math.trunc(v(i));
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'charge': {
        if (!hasCharge(s)) throw new StyleError(`Cannot set attribute charge for atom style ${s.atomStyle}`);
        const v = value(a[k + 1], 'charge');
        for (const i of atoms) s.q[i] = v(i);
        // Measured with native LAMMPS (black box), atom_style dielectric: after set epsilon 2.0, set charge 3.0 gives
        // q 3 and q_scaled 1.5 (the scaled charge follows the new charge)
        const qsc = s.custom.get(DIEL_QSCALED), epc = s.custom.get(DIEL_EPSILON);
        if (qsc && epc) for (const i of atoms) qsc.data[i] = s.q[i] / epc.data[i];
        changed = atoms.length;
        sys.nb.refreshCharges(s);
        k += 2;
        break;
      }
      case 'x': case 'y': case 'z': case 'vx': case 'vy': case 'vz': {
        const v = value(a[k + 1], key);
        const arr = key[0] === 'v' ? s.v : s.x;
        const d = 'xyz'.indexOf(key[key.length - 1]);
        for (const i of atoms) arr[3 * i + d] = v(i);
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'image': {
        for (let d = 0; d < 3; d++) {
          const w = a[k + 1 + d];
          if (w === 'NULL') continue;
          // As native LAMMPS does (src/set.cpp process_image at b332f28)
          if (!s.box.periodic[d] && w.startsWith('v_')) throw new StyleError('Cannot set variable image flag for non-periodic dimension');
          if (!s.box.periodic[d] && !w.startsWith('v_') && inumeric(w) !== 0) throw new StyleError('Cannot set non-zero image flag for non-periodic dimension');
          const v = value(w, 'image');
          for (const i of atoms) s.image[3 * i + d] = Math.trunc(v(i));
        }
        changed = atoms.length;
        k += 4;
        break;
      }
      // set.html: "Keyword *mass* sets the mass of all selected particles.  The particles must have a
      // per-atom mass attribute"; "Keyword *diameter* sets the size of the selected atoms ... this
      // command does not adjust the particle mass"; "Keyword *density* or *density/disc* also sets the
      // mass" (atoms.ts sphereMass); "Keyword *omega* sets the angular velocity of selected atoms."
      case 'mass': case 'density': case 'density/disc': {
        // errors as measured with native LAMMPS
        if (!s.rmass) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle}`);
        if (key === 'density/disc' && sys.dimension !== 2) throw new StyleError('Set density/disc requires 2d simulation');
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const x = v(i);
          if (!(x > 0)) throw new StyleError(key === 'mass' ? `Invalid mass ${x} in set command` : `Invalid density value ${x} in set command`);
          // set.html: "If the atom has a shape attribute (see :doc:`atom_style ellipsoid <atom_style>`) and its 3
          // shape parameters are non-zero, then its mass is set from the density and particle volume" and "The
          // *density/disc* keyword has no effect"; measured with native LAMMPS (black box): a point particle of
          // atom style ellipsoid gets mass = density
          if (key !== 'mass' && !(s.radius && s.radius[i] > 0) && isEllipsoid(s, i)) s.rmass[i] = x * ellipsoidVolume(s, i);
          else s.rmass[i] = key === 'mass' ? x : sphereMass(s.radius ? s.radius[i] : 0, x, key === 'density/disc');
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keyword *volume* sets the volume of all selected particles. Currently, only the atom_style peri
      // command defines particles with a volume attribute. Note that this command does not adjust the particle mass."
      // Measured with native LAMMPS (black box): a volume <= 0 is refused (Invalid volume in set command)
      // set.html: "Keyword *smd/contact/radius* ... sets an interaction radius for computing short-range interactions"
      // (atom_style smd only); "Keyword *smd/mass/density* sets the mass of all selected particles ... It assumes that
      // the particle volume has already been correctly set and calculates particle mass from the provided mass density".
      case 'smd/contact/radius': case 'smd/mass/density': {
        const contact = s.custom.get(SMD_CONTACT);
        if (!isSmdStyle(s.atomStyle) || !contact) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle}`);
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const x = v(i);
          if (key === 'smd/contact/radius') contact.data[i] = x;
          else s.rmass![i] = x * s.vfrac![i];
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'volume': {
        if (!s.vfrac) throw new StyleError(`Cannot set attribute volume for atom style ${s.atomStyle}`);
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const x = v(i);
          if (!(x > 0)) throw new StyleError(`Invalid volume in set command`);
          s.vfrac[i] = x;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keyword epsilon sets the dielectric constant of a particle to be that of the medium where the
      // particle resides as defined by the DIELECTRIC package." and "Note that the set command with this keyword will
      // rescale the particle charge accordingly so that the real charge (e.g., as read from a data file) stays
      // intact." The page says "The value for the dielectric constant must be >= 0.0"; measured with native LAMMPS
      // (black box): 0 and -2 stop with Invalid epsilon value 0 in set command, 2.0 gives q 1 and q_scaled 0.5.
      // set.html: "Keywords *edpd/temp* and *edpd/cv* set the temperature and volumetric heat capacity of an eDPD particle as
      // defined by the DPD-MESO package. Currently, only atom_style edpd defines particles with these attributes. The values
      // for the temperature and heat capacity must be positive."
      // set.html: "Keyword *dpd/theta* sets the internal temperature of a DPD particle as defined by the DPD-REACT package.
      // If the specified value is a number it must be >= 0.0. If the specified value is NULL, then the kinetic temperature
      // Tkin of each particle is computed as 3/2 k Tkin = KE = 1/2 m v^2". As native LAMMPS does (src/set.cpp
      // invoke_dpd_theta at b332f28): NULL gives tfactor m v^2 with tfactor = mvv2e / (dimension kB), and a negative value
      // stops with "Invalid dpd/theta value in set command" also for an atom-style variable (the doc says a negative
      // variable value selects Tkin; native's code refuses it, and the engine follows the code).
      // set.html: "*apip/lambda* value = fast or precise or float" and "The precise potential is used for an atom when
      // its switching parameter lambda is 0. The fast potential is used for an atom when its switching parameter
      // lambda is 1." As native LAMMPS does (src/set.cpp process_apip_lambda and invoke_apip_lambda at b332f28): fast
      // is 1, precise is 0, and a value outside [0,1] (a number, or an atom-style variable per atom) stops the run.
      case 'apip/lambda': {
        if (!isApipStyle(s.atomStyle)) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle.split(' ')[0]}`);
        if (a[k + 1] === undefined) throw new StyleError('Illegal set apip/lambda command: missing argument(s)');
        const lam = s.custom.get(APIP_LAMBDA)!.data;
        const w = a[k + 1];
        const v = w === 'fast' ? () => 1 : w === 'precise' ? () => 0 : value(w, key);
        for (const i of atoms) {
          const x = v(i);
          if (x < 0 || x > 1) throw new StyleError(`apip/lambda ${x} not in [0,1]`);
          lam[i] = x;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'dpd/theta': {
        if (!isDpdStyle(s.atomStyle)) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle}`);
        const th = s.custom.get(DPD_THETA)!.data;
        if (a[k + 1] === 'NULL') {
          const tfactor = s.units.mvv2e / (s.dimension * s.units.boltz);
          for (const i of atoms) {
            const vx = s.v[3 * i], vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
            th[i] = tfactor * massOf(s, i) * (vx * vx + vy * vy + vz * vz);
          }
        } else {
          const v = value(a[k + 1], key);
          for (const i of atoms) {
            const x = v(i);
            if (!(x >= 0)) throw new StyleError(`Invalid dpd/theta value ${x} in set command`);
            th[i] = x;
          }
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'edpd/temp': case 'edpd/cv': {
        if (!isEdpdStyle(s.atomStyle)) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle}`);
        const v = value(a[k + 1], key);
        const arr = s.custom.get(key === 'edpd/temp' ? EDPD_TEMP : EDPD_CV)!.data;
        for (const i of atoms) {
          const x = v(i);
          if (!(x > 0)) throw new StyleError(`Invalid ${key} value ${x} in set command`);
          arr[i] = x;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keywords rheo/rho and rheo/status set the density and the status of rheo particles." and
      // "rheo/rho value = density of RHEO particles (mass/distance^3)", "rheo/status value = status or phase of RHEO
      // particles (unitless)". Measured with native LAMMPS (black box): after set rheo/rho 2.5 and rheo/status 1,
      // write_data gives the density 2.5 and status 1 for every selected atom.
      case 'rheo/rho': case 'rheo/status': {
        if (!isRheoStyle(s.atomStyle)) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle}`);
        const name = key === 'rheo/rho' ? RHEO_RHO : RHEO_STATUS;
        const v = value(a[k + 1], key);
        const arr = s.custom.get(name)!.data;
        for (const i of atoms) arr[i] = v(i);
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keyword *cc* sets the chemical concentration of a tDPD particle for a specified species as defined by the
      // DPD-MESO package. ... An integer for "index" selects a chemical species (1 to Nspecies) ... The value for the chemical
      // concentration must be >= 0.0."
      case 'cc': {
        const nsp = tdpdSpecies(s.atomStyle);
        if (!nsp) throw new StyleError(`Cannot set attribute cc for atom style ${s.atomStyle}`);
        // src/set.cpp process_cc at b332f28: an index below 1 is refused; native writes outside the species for one above
        // them, which the engine refuses
        const idx = inumeric(a[k + 1]);
        if (idx < 1) throw new StyleError(`Invalid cc index ${a[k + 1]} in set command`);
        if (idx > nsp) throw new StyleError(`set cc: species index ${idx} is above the ${nsp} species (native LAMMPS writes outside its array)`);
        const v = value(a[k + 2], key);
        const cc = s.custom.get(TDPD_CC)!.data;
        for (const i of atoms) {
          const x = v(i);
          if (!(x >= 0)) throw new StyleError(`Invalid cc value ${x} in set command`);
          cc[nsp * i + idx - 1] = x;
        }
        changed = atoms.length;
        k += 3;
        break;
      }
      case 'epsilon': {
        if (!isDielectricStyle(s.atomStyle)) throw new StyleError(`Cannot set attribute epsilon for atom style ${s.atomStyle}`);
        if (!String(a[k + 1]).startsWith('v_') && !(Number(a[k + 1]) > 0)) throw new StyleError(`Invalid epsilon value ${a[k + 1]} in set command`);
        const v = value(a[k + 1], key);
        const eps = s.custom.get(DIEL_EPSILON)!.data, qsc = s.custom.get(DIEL_QSCALED)!.data;
        for (const i of atoms) {
          const e = v(i);
          if (!(e > 0)) throw new StyleError('Invalid epsilon in set command');
          eps[i] = e;
          qsc[i] = s.q[i] / e;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keywords sph/cv, sph/e, and sph/rho set the heat capacity, energy, and density of smoothed particle
      // hydrodynamics (SPH) particles."; measured with native LAMMPS (black box):
      // a negative density is accepted, and the keyword on atom_style atomic stops with Cannot set attribute sph/e for
      // atom style atomic
      case 'sph/cv': case 'sph/e': case 'sph/rho': {
        // set.html (sph/e on atom_style rheo/thermal, the energy column of the RHEO thermal particles, as native
        // LAMMPS sets esph for any style with an energy: examples/rheo/ice-cubes uses set sph/e with rheo/thermal)
        if (key === 'sph/e' && isRheoThermalStyle(s.atomStyle)) {
          const earr = s.custom.get(RHEO_ESPH)!.data;
          const ev = value(a[k + 1], key);
          for (const i of atoms) earr[i] = ev(i);
          changed = atoms.length;
          k += 2;
          break;
        }
        if (!isSphStyle(s.atomStyle)) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle}`);
        const arr = s.custom.get(key === 'sph/cv' ? SPH_CV : key === 'sph/e' ? SPH_ESPH : SPH_RHO)!.data;
        const v = value(a[k + 1], key);
        for (const i of atoms) arr[i] = v(i);
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keyword *radius/electron* uses the specified value to set the radius of electrons or fixed cores."
      // Measured with native LAMMPS (black box): the value becomes the electron radius; a negative one stops with
      // Invalid electron radius -1 in set radius/electron command.
      case 'radius/electron': {
        const er = electronProp(s, ELECTRON_RADIUS, 'radius/electron');
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const r = v(i);
          if (r < 0.0) throw new StyleError(`Invalid electron radius ${r} in set radius/electron command`);
          er[i] = r;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keyword *spin/electron* sets the spin of an electron (+/- 1) or indicates nuclei (=0), fixed-cores (=2), or pseudo-cores (= 3)."
      // Measured with native LAMMPS (black box): spin 4 or -2 stops with Invalid electron spin 4 in set command.
      case 'spin/electron': {
        const es = electronProp(s, ELECTRON_SPIN, 'spin/electron');
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const sp = v(i);
          if (sp < -1 || sp > 3) throw new StyleError(`Invalid electron spin ${sp} in set command`);
          es[i] = sp;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'diameter': {
        if (!s.radius) throw new StyleError(`Cannot set attribute diameter for atom style ${s.atomStyle}`);
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const d = v(i);
          if (!(d >= 0)) throw new StyleError(`Invalid diameter value ${d} in set command`);
          s.radius[i] = d / 2;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keyword *dipole* uses the specified x,y,z values as components of a vector to set as
      // the orientation of the dipole moment vectors of the selected atoms.  The magnitude of the dipole
      // moment is set by the length of this orientation vector."
      case 'dipole': {
        if (!s.mu) throw new StyleError(`Cannot set attribute dipole for atom style ${s.atomStyle}`);
        const vx = value(a[k + 1], 'dipole x'), vy = value(a[k + 2], 'dipole y'), vz = value(a[k + 3], 'dipole z');
        for (const i of atoms) {
          const mx = vx(i), my = vy(i), mz = vz(i);
          s.mu[4 * i] = mx; s.mu[4 * i + 1] = my; s.mu[4 * i + 2] = mz; s.mu[4 * i + 3] = Math.hypot(mx, my, mz);
        }
        changed = atoms.length;
        k += 4;
        break;
      }
      // set.html: "Keyword *spin/atom* uses the specified g value to set the magnitude of the magnetic spin
      // vectors, and the x,y,z values as components of a vector to set as the orientation of the magnetic spin
      // vectors of the selected atoms." Errors as measured with native LAMMPS (black box): a zero magnitude
      // ("Invalid spin magnitude 0 in set spin/atom command") and a zero direction.
      // set.html: "This keyword was previously called *spin*."
      case 'spin':
      case 'spin/atom': {
        if (!s.sp) throw new StyleError(`Cannot set attribute spin/atom for atom style ${s.atomStyle}`);
        const vg = value(a[k + 1], 'spin/atom g'), vx = value(a[k + 2], 'spin/atom x'), vy = value(a[k + 3], 'spin/atom y'), vz = value(a[k + 4], 'spin/atom z');
        for (const i of atoms) {
          const g = vg(i);
          if (!(g !== 0)) throw new StyleError(`Invalid spin magnitude ${g} in set spin/atom command`);
          setSpinVector(s.sp, i, vx(i), vy(i), vz(i), g, 'set spin/atom');
        }
        changed = atoms.length;
        k += 5;
        break;
      }
      // set.html: "Keyword *spin/atom/random* randomizes the orientation of the magnetic spin vectors", and
      // "spin/random" (the older name). Set spin/atom/random seed Dlen: the magnitude is Dlen, the direction is
      // random. Measured with native LAMMPS (black box): Dlen <= 0 stops with Invalid spin magnitude 0 in set
      // spin/atom/random command (and the same with spin/random).
      case 'spin/random':
      case 'spin/atom/random': {
        if (!s.sp) throw new StyleError(`Cannot set attribute ${a[k]} for atom style ${s.atomStyle}`);
        const seed = int(a[k + 1], 'seed');
        const g = num(a[k + 2], 'spin magnitude');
        if (seed <= 0) throw new StyleError(`Invalid random number seed ${seed} in set ${a[k]} command`);
        if (!(g > 0)) throw new StyleError(`Invalid spin magnitude ${g} in set ${a[k]} command`);
        marsagliaSeed(seed);
        const rnd = nativeRandom('spin/atom/random');
        const dir = new Float64Array(3);
        for (const i of atoms) {
          rnd.spinRandom(seed, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], sys.dimension, dir);
          s.sp[4 * i] = dir[0]; s.sp[4 * i + 1] = dir[1]; s.sp[4 * i + 2] = dir[2]; s.sp[4 * i + 3] = g;
        }
        changed = atoms.length;
        k += 3;
        break;
      }
      // set.html: "Keyword *shape* sets the size and shape of the selected atoms.  The particles must be
      // ellipsoids as defined by the :doc:`atom_style ellipsoid <atom_style>` command.  The *Sx*, *Sy*, *Sz*
      // settings are the 3 diameters of the ellipsoid in each direction."; "They can also all be set to 0.0
      // which means the particle will be treated as a point particle.  Note that this command does not adjust
      // the particle mass". Measured with native LAMMPS (black box): one zero diameter among non-zero ones is
      // refused (Invalid shape in set command).
      case 'shape': {
        if (!s.shape) throw new StyleError(`Cannot set attribute shape for atom style ${s.atomStyle}`);
        const vs = [1, 2, 3].map((d) => value(a[k + d], 'shape'));
        for (const i of atoms) {
          const d3 = vs.map((f) => f(i));
          const zeros = d3.filter((x) => x === 0).length;
          if (d3.some((x) => !(x >= 0)) || (zeros > 0 && zeros < 3)) throw new StyleError('Invalid shape in set command');
          for (let d = 0; d < 3; d++) s.shape[3 * i + d] = d3[d] / 2;
          if (isSuperEllipsoidStyle(s.atomStyle)) storeSuperInertia(s, i);
        }
        changed = atoms.length;
        k += 4;
        break;
      }
      // set.html: "Keyword *block* sets the blockiness of the selected atoms.  The particles must be ellipsoids as
      // defined by the atom_style ellipsoid command."; "The *block1*, *block2* settings are the 2 exponents of the
      // superellipsoid in the vertical and horizontal directions."; "If both parameters are set to a value of 2 (the
      // default), the atom is a regular ellipsoid."; "If the keyword *block* is given alone, and the *shape* has not been
      // defined, e.g., in a previous *set* command, the 3 diameters would be set to a value of 1 internally."
      // The moment of inertia is stored with the new block (superellipsoid.ts).
      case 'block': {
        if (!isSuperEllipsoidStyle(s.atomStyle)) throw new StyleError(`Cannot set attribute block for atom style ${s.atomStyle}`);
        const vs = [1, 2].map((d) => value(a[k + d], 'block'));
        const blk = s.custom.get(SUPER_BLOCK)!.data;
        for (const i of atoms) {
          const [b1, b2] = vs.map((f) => f(i));
          if (!(b1 > 0) || !(b2 > 0)) throw new StyleError('Invalid block in set command');
          if (!isEllipsoid(s, i)) for (let d = 0; d < 3; d++) s.shape![3 * i + d] = 0.5;
          blk[2 * i] = b1;
          blk[2 * i + 1] = b2;
          storeSuperInertia(s, i);
        }
        changed = atoms.length;
        k += 3;
        break;
      }
      // set.html: "If this body is rotated (via the right-hand rule) by an angle theta around a unit rotation
      // vector (a,b,c), then the quaternion that represents its new orientation is given by (cos(theta/2),
      // a\*sin(theta/2), b\*sin(theta/2), c\*sin(theta/2))."; "LAMMPS normalizes the quaternion in case (a,b,c)
      // was not specified as a unit vector."; "The 3 values must be non-zero for each particle set by this
      // command." Measured with native LAMMPS (black box): the whole 4-vector is normalized (a b c = 1 1 0,
      // theta 90 gives 0.57735 0.57735 0.57735 0); a point particle is refused (Cannot set quaternion for atom
      // that has none); in 2d a non-zero a or b is refused (Cannot set quaternion with xy components for 2d
      // system). As native LAMMPS does (src/set.cpp process_quat at b332f28): an atom with a per-atom quat array
      // (quat_flag: atom_style bpm/sphere, atom_style.html row bpm/sphere "quat") is set through that array even
      // when it is not an ellipsoid.
      case 'quat': {
        if (!s.quat) throw new StyleError(`Cannot set attribute quat for atom style ${s.atomStyle}`);
        const vs = [1, 2, 3, 4].map((d) => value(a[k + d], 'quat'));
        for (const i of atoms) {
          // set.cpp (src/set.cpp at b332f28, the quat branch): ellipsoid, then tri (tri[i] >= 0), then body, then quat_flag
          if (!isEllipsoid(s, i) && !isTriAtom(s, i) && !isBpmSphereStyle(s.atomStyle)) throw new StyleError('Cannot set quaternion for atom that has none');
          const [qa, qb, qc, th] = vs.map((f) => f(i));
          if (sys.dimension === 2 && (qa !== 0 || qb !== 0)) throw new StyleError('Cannot set quaternion with xy components for 2d system');
          const h = (th * Math.PI) / 360, sn = libm.sin(h);
          const q = [libm.cos(h), qa * sn, qb * sn, qc * sn];
          const len = Math.hypot(q[0], q[1], q[2], q[3]);
          for (let d = 0; d < 4; d++) s.quat[4 * i + d] = q[d] / len;
        }
        changed = atoms.length;
        k += 5;
        break;
      }
      // set.html: "Keyword *quat/random* randomizes the orientation of the quaternion for the selected atoms."; "As with
      // keyword *quat*, for ellipsoidal particles, the 3 shape values must be non-zero for each particle set by this
      // command."; "For 2d systems, only orientations in the xy plane are generated." The per-atom streams are
      // native's (nativeRandom.ts).
      case 'quat/random': {
        if (!s.quat) throw new StyleError(`Cannot set attribute quat/random for atom style ${s.atomStyle}`);
        const seed = int(a[k + 1], 'seed');
        if (seed <= 0) throw new StyleError('set quat/random: seed must be a positive integer');
        marsagliaSeed(seed);
        const rnd = nativeRandom('quat/random');
        const q = new Float64Array(4);
        for (const i of atoms) {
          if (!isEllipsoid(s, i)) throw new StyleError('Cannot set quaternion for atom that has none');
          rnd.quatRandom(seed, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], sys.dimension, q);
          for (let d = 0; d < 4; d++) s.quat[4 * i + d] = q[d];
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      // set.html: "Keyword *angmom* sets the angular momentum of selected atoms.  The particles must be
      // ellipsoids as defined by the :doc:`atom_style ellipsoid <atom_style>` command"; "The angular momentum
      // vector of the particles is set to the 3 specified components."
      case 'angmom': {
        if (!s.angmom) throw new StyleError(`Cannot set attribute angmom for atom style ${s.atomStyle}`);
        const vs = [1, 2, 3].map((d) => value(a[k + d], 'angmom'));
        for (const i of atoms) for (let d = 0; d < 3; d++) s.angmom[3 * i + d] = vs[d](i);
        changed = atoms.length;
        k += 4;
        break;
      }
      // set.html: "Keyword *dipole/random* randomizes the orientation of the dipole moment vectors for the selected
      // atoms and sets the magnitude of each to the specified *Dlen* value.  For 2d systems, the z component of the
      // orientation is set to 0.0." The per-atom streams are native's (nativeRandom.ts).
      case 'dipole/random': {
        if (!s.mu) throw new StyleError(`Cannot set attribute dipole/random for atom style ${s.atomStyle}`);
        const seed = int(a[k + 1], 'seed');
        const dlen = num(a[k + 2], 'Dlen');
        if (seed <= 0) throw new StyleError('set dipole/random: seed must be a positive integer');
        if (!(dlen > 0)) throw new StyleError('set dipole/random: Dlen must be > 0');
        marsagliaSeed(seed);
        const rnd = nativeRandom('dipole/random');
        const mu = new Float64Array(3);
        for (const i of atoms) {
          rnd.dipoleRandom(seed, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], dlen, sys.dimension, mu);
          s.mu[4 * i] = mu[0]; s.mu[4 * i + 1] = mu[1]; s.mu[4 * i + 2] = mu[2]; s.mu[4 * i + 3] = dlen;
        }
        changed = atoms.length;
        k += 3;
        break;
      }
      case 'omega': {
        if (!s.omega) throw new StyleError(`Cannot set attribute omega for atom style ${s.atomStyle}`);
        const vs = [1, 2, 3].map((d) => value(a[k + d], 'omega'));
        for (const i of atoms) for (let d = 0; d < 3; d++) s.omega[3 * i + d] = vs[d](i);
        changed = atoms.length;
        k += 4;
        break;
      }
      case 'bond': case 'angle': case 'dihedral': case 'improper': {
        const t = int(typeLabelWord(sys, key as LabelKind, a[k + 1]), `${key} type`);
        const list = s.topo[`${key}s` as 'bonds' | 'angles' | 'dihedrals' | 'impropers'];
        const ids = new Set(atoms.map((i) => s.id[i]));
        for (let e = 0; e < list.n; e++) {
          let all = true;
          for (let w = 0; w < list.width; w++) if (!ids.has(list.atoms[e * list.width + w])) { all = false; break; }
          if (all) { list.type[e] = t; changed++; }
        }
        k += 2;
        break;
      }
      // set.html: "Keyword *temperature* sets the temperature of a finite-size particle." "The temperature must be
      // added using an instance of fix property/atom". Measured with native LAMMPS (black box): a negative value stops
      // with Invalid temperature -1 in set command; zero is accepted.
      case 'temperature': {
        const cp = s.custom.get(HEAT_TEMPERATURE);
        if (!cp) throw new StyleError('set temperature needs fix property/atom temperature heatflow (GRANULAR heat conduction)');
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const x = v(i);
          if (x < 0) throw new StyleError(`Invalid temperature ${x} in set command`);
          cp.data[i] = x;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      default: {
        // set.html: "*i_name* value = custom integer vector with name", "*d_name* value = custom
        // floating-point vector with name", "column specified as i2_name[N] where N is 1 to Ncol"
        const m = /^(i|d)(2?)_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(key);
        const cp = m ? s.custom.get(m[3]) : undefined;
        if (!m || !cp || cp.int !== (m[1] === 'i') || (cp.cols > 0) !== (m[2] === '2')) {
          // native's message names the property without its i_/d_ prefix (src/set.cpp process_custom)
          throw new StyleError(m ? `Set keyword or custom property ${m[3]} does not exist` : `set keyword '${key}' is not supported by the browser engine`);
        }
        const col = m[4] ? Number(m[4]) : 0;
        if (cp.cols > 0 && (col < 1 || col > cp.cols)) throw new StyleError(`set ${key}: column must be 1..${cp.cols}`);
        if (cp.cols === 0 && m[4]) throw new StyleError(`set ${key}: ${m[1]}_${m[3]} is a vector, not an array`);
        const w = Math.max(cp.cols, 1), off = cp.cols ? col - 1 : 0;
        const v = value(a[k + 1], key);
        for (const i of atoms) { const x = v(i); cp.data[w * i + off] = cp.int ? Math.trunc(x) : x; }
        changed = atoms.length;
        k += 2;
        break;
      }
    }
  }
  sys.bump();
  sys.log(`Setting atom values ...\n  ${changed} settings made for ${a[2]}`);
};

// ----------------------------------------------------------------- velocity

/** velocity group style args [keywords] — velocity.html (see the module comment for defaults). */
const velocity: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/velocity.cpp Velocity::command and Velocity::options at b332f28): the box, the atoms, the
  // per-type masses, the group and the style are checked first, then the keywords in order (with the bias checks after
  // them), and the style's own arguments last
  if (a.length < 2) throw missingArgs('velocity');
  if (!sys.hasBox) throw new StyleError('Velocity command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  const s = sys.state;
  if (s.n === 0) throw new StyleError('Velocity command on a system without atoms');
  if (!s.rmass) {
    for (let t = 1; t <= s.ntypes; t++) if (!(s.massByType[t] > 0)) throw new StyleError(`Not all per-type masses are set. Type ${t} is missing.`);
  }
  const [gname, style] = a;
  if (sys.groups.find(gname) < 0) throw new StyleError(`Could not find velocity group ID ${gname}`);
  const bit = sys.groupBit(gname);
  const nargs: Record<string, number> = { create: 2, set: 3, scale: 1, ramp: 6, zero: 1 };
  if (!Object.prototype.hasOwnProperty.call(nargs, style)) throw new StyleError(`Unknown velocity command style: ${style}`);
  if (a.length < 2 + nargs[style]) throw missingArgs('velocity');
  const args = a.slice(2, 2 + nargs[style]);
  // "The keyword defaults are dist = uniform, sum = no, mom = yes, rot = no, bias = no, loop = all, and units = lattice."
  let dist = 'uniform', sum = false, mom = true, rot = false, bias = false, loop = 'all', unitsW = 'lattice';
  let tempC: Compute | null = null;
  // velocity.html: "If the rigid option is specified, then the zeroing is performed on individual rigid bodies, as
  // defined by the fix rigid or fix rigid/small commands." Only the zero style uses it
  let rigidFix: Fix | null = null;
  const opt = a.slice(2 + nargs[style]);
  for (let k = 0; k < opt.length; k += 2) {
    const key = opt[k], v = opt[k + 1];
    const need = () => { if (v === undefined) throw missingArgs(`velocity ${key}`); };
    switch (key) {
      case 'dist':
        need();
        if (v !== 'uniform' && v !== 'gaussian') throw new StyleError(`Unknown velocity dist argument: ${v}`);
        dist = v;
        break;
      case 'sum': need(); sum = logical(v); break;
      case 'mom': need(); mom = logical(v); break;
      case 'rot': need(); rot = logical(v); break;
      case 'temp': {
        need();
        const c = sys.computes.find((x) => x.id === v);
        if (!c) throw new StyleError(`Could not find velocity temperature compute ID: ${v}`);
        if (!c.tempFlag) throw new StyleError(`Velocity temperature compute ${v} does not compute temperature`);
        tempC = c;
        break;
      }
      case 'bias': need(); bias = logical(v); break;
      case 'loop':
        need();
        if (v !== 'all' && v !== 'local' && v !== 'geom') throw new StyleError(`Unknown velocity loop argument: ${v}`);
        loop = v;
        break;
      case 'rigid':
        need();
        rigidFix = sys.fixes.find((f) => f.id === v) ?? null;
        if (!rigidFix) throw new StyleError(`Fix ID ${v} for velocity does not exist`);
        break;
      case 'units':
        need();
        if (v !== 'box' && v !== 'lattice') throw new StyleError(`Unknown velocity units argument: ${v}`);
        unitsW = v;
        break;
      default: throw new StyleError(`Unknown velocity keyword: ${key}`);
    }
  }
  if (bias && tempC === null) throw new StyleError('Cannot use velocity bias command without temp keyword');
  // native's message (it says does where it means does not)
  if (bias && !tempC!.hasBias()) throw new StyleError('Velocity temperature ID does calculate a velocity bias');
  // Modify::check_rigid_group_overlap: a warning when the group has atoms of set-up rigid bodies (the first rigid fix
  // whose bodies are not set up yet ends the check), unless the zeroing is done on the bodies themselves
  if (!(style === 'zero' && rigidFix)) {
    let overlap = false;
    for (const f of sys.fixes) {
      if (!/^rigid/.test(f.style)) continue;
      const ids = f.bodyAtomIds?.() ?? null;
      if (!ids) break;
      for (let i = 0; i < s.n && !overlap; i++) if ((s.mask[i] & bit) && ids.has(s.id[i])) overlap = true;
      if (overlap) break;
    }
    if (overlap) sys.warn('Changing velocities of atoms in rigid bodies. This has no effect unless rigid bodies are rebuilt');
  }
  const members: number[] = [];
  // native order (SimState.order): loop local draws and the sums follow native's atom list
  for (const i of nativeOrder(s)) if (s.mask[i] & bit) members.push(i);
  // "If this keyword is not specified, create and scale calculate temperature using a compute ...
  //  compute velocity_temp group-ID temp"
  const tempCompute = () => {
    if (tempC) {
      tempC.init();
      return tempC;
    }
    const c = new ComputeTemp(sys, 'velocity_temp', gname, []);
    c.init();
    return c;
  };
  const scaleTo = (target: number) => {
    sys.refreshComputes();
    const c = tempCompute();
    if (bias && c.hasBias()) { c.computeBias(); c.removeBiasAll(); }
    sys.refreshComputes();
    const t = c.scalarValue();
    // Velocity::rescale: "Attempting to rescale a 0.0 temperature"
    if (t === 0) throw new StyleError('Attempting to rescale a 0.0 temperature');
    const f = Math.sqrt(target / t);
    for (const i of members) { s.v[3 * i] *= f; s.v[3 * i + 1] *= f; s.v[3 * i + 2] *= f; }
    if (bias && c.hasBias()) c.restoreBiasAll();
  };
  // docs.lammps.org/velocity.html (bias keyword) says the bias "is subtracted
  // from atom velocities before the *create* and *scale* operations are
  // performed.  After the operations, the bias is added back to the atom
  // velocities." Measured with native LAMMPS (black box) on a 4-pair core/shell
  // probe (tests/oracle/w26velbias_*.in), `create ... bias yes temp <compute>`
  // first rescales the freshly generated ensemble to the target with the
  // internal plain compute (the *temp* keyword does not enter the scale factor:
  // `create ... temp CStemp` gives c_CStemp = 1427 without bias, but with
  // `bias yes` the pair-COM field equals a plain create, c_CStemp = 1719.538
  // on the probe) and then applies the temp compute's bias to the rescaled
  // velocities without adding it back: temp/cs and temp/partial remove the bias
  // (velocities replaced by the pair COM / excluded components zeroed),
  // temp/ramp adds the ramp, and temp/com and temp/region leave the velocities
  // unchanged.
  const scalePlain = (target: number) => {
    sys.refreshComputes();
    const pc = new ComputeTemp(sys, 'velocity_temp', gname, []);
    pc.init();
    sys.refreshComputes();
    const t = pc.scalarValue();
    if (t === 0) throw new StyleError('Attempting to rescale a 0.0 temperature');
    const f = Math.sqrt(target / t);
    for (const i of members) { s.v[3 * i] *= f; s.v[3 * i + 1] *= f; s.v[3 * i + 2] *= f; }
  };
  const createBias = () => {
    const c = tempCompute();
    if (!c.hasBias()) return;
    const before = Float64Array.from(s.v);
    c.removeBiasAll();
    const removed = Float64Array.from(s.v);
    c.restoreBiasAll();
    if (c.style === 'temp/ramp') {
      for (let k = 0; k < s.v.length; k++) s.v[k] = before[k] + (before[k] - removed[k]);
    } else if (c.style === 'temp/cs' || c.style === 'temp/partial') {
      s.v.set(removed);
    }
  };
  const old = sum ? Float64Array.from(s.v) : null;
  switch (style) {
    case 'create': {
      const t = numeric(args[0]);
      const seed = inumeric(args[1]);
      if (seed <= 0) throw new StyleError(`Illegal velocity create seed argument: ${seed}`);
      // native takes the square root of a negative ratio and gives every velocity NaN
      if (t < 0) throw new StyleError('velocity create with a negative temperature is not supported by the browser engine (native sets every velocity to NaN)');
      // Measured with native LAMMPS (black box; rng.ts RanPark): each atom takes
      // three draws, uniform() - 0.5 or gaussian(), scaled by 1/sqrt(mass).
      // velocity.html: "If loop = all, then each processor loops over all
      // atoms in the simulation to create velocities, but only stores
      // velocities for atoms it owns." — draws run over atom IDs 1..N
      // (non-members draw too) and need consecutive IDs (native stops with an
      // error otherwise). "If loop = local, then
      // each processor loops over only its atoms to produce velocities.  The
      // random number seed is adjusted to give a different set of velocities
      // on each processor." — on one processor: group members in storage
      // order, after 100 discarded uniform draws. loop geom seeds every atom
      // from its coordinates ("For each atom a unique random number seed is
      // created, based on the atom's xyz coordinates"); the docs do not
      // promise identical velocities across machines ("will not necessarily
      // assign identical velocities for two simulations run on different
      // machines"), and the per-atom streams here are native's (nativeRandom.ts).
      const draw = (rng: { uniform(): number; gaussian(): number }, i: number) => {
        const c = 1 / Math.sqrt(massOf(s, i));
        for (let d = 0; d < 3; d++) {
          const r = dist === 'gaussian' ? rng.gaussian() : rng.uniform() - 0.5;
          s.v[3 * i + d] = d === 2 && s.dimension === 2 ? 0 : r * c;
        }
      };
      if (loop === 'all') {
        let maxId = 0;
        for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
        if (maxId !== s.n) throw new StyleError('Atom IDs must be consecutive for velocity create loop all');
        const slot = new Int32Array(maxId + 1).fill(-1);
        for (const i of members) slot[s.id[i]] = i;
        const rng = new RanPark(seed);
        const sink = new Float64Array(3);
        for (let id = 1; id <= maxId; id++) {
          if (slot[id] >= 0) draw(rng, slot[id]);
          else for (let d = 0; d < 3; d++) sink[d] = dist === 'gaussian' ? rng.gaussian() : rng.uniform();
        }
      } else if (loop === 'local') {
        const rng = new RanPark(seed);
        for (let k = 0; k < 100; k++) rng.uniform();
        for (const i of members) draw(rng, i);
      } else {
        const geom = nativeRandom('geom');
        const r = new Float64Array(3);
        for (const i of members) {
          geom.velocityGeom(seed, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], dist === 'gaussian', r);
          const c = 1 / Math.sqrt(massOf(s, i));
          for (let d = 0; d < 3; d++) s.v[3 * i + d] = d === 2 && s.dimension === 2 ? 0 : r[d] * c;
        }
      }
      if (mom) zeroMomentum(sys, members);
      if (rot) zeroRotation(sys, members);
      if (bias) { scalePlain(t); createBias(); } else scaleTo(t);
      break;
    }
    case 'scale': scaleTo(numeric(args[0])); break;
    case 'set': {
      const sc = latticeScale(sys, unitsW, 'velocity');
      // Velocity::set: the three values are read first (a constant is scaled by the lattice spacing, a variable is
      // not), then the variables are looked up (equal-style: equal, timer, internal; atom-style: atom, atomfile), then
      // the z component of a 2d system is checked
      const parsed = args.map((w) => (w === 'NULL' ? null : w.startsWith('v_') ? w.slice(2) : numeric(w)));
      const comp = parsed.map((p, d) => {
        if (p === null) return null;
        if (typeof p === 'number') { const x = p * sc[d]; return () => x; }
        const v = sys.vars.get(p);
        if (!v) throw new StyleError('Variable name for velocity set does not exist');
        if (v.style === 'atom' || v.style === 'atomfile') { const arr = sys.atomVariable(p); return (i: number) => arr[i]; }
        if (v.style !== 'equal' && v.style !== 'timer' && v.style !== 'internal') throw new StyleError('Variable for velocity set is invalid style');
        const x = sys.equalVariable(p);
        return () => x;
      });
      if (s.dimension === 2) {
        if (typeof parsed[2] === 'number' && parsed[2] !== 0) throw new StyleError('Cannot set non-zero z velocity for 2d simulation');
        if (typeof parsed[2] === 'string') throw new StyleError('Cannot set variable z velocity for 2d simulation');
      }
      // velocity.html, keyword sum: "The new velocities will be added to the existing ones if sum = yes". Only the
      // components the style sets are added; a NULL component keeps its old value (old is only set when sum is yes).
      for (const i of members) for (let d = 0; d < 3; d++) { const f = comp[d]; if (f) s.v[3 * i + d] = (old ? old[3 * i + d] : 0) + f(i); }
      break;
    }
    case 'ramp': {
      // Velocity::ramp: the velocity component, its range, the coordinate and its range, in that order
      const vd = ['vx', 'vy', 'vz'].indexOf(args[0]);
      if (vd < 0) throw new StyleError(`Illegal velocity ramp velocity direction ${args[0]}`);
      if (vd === 2 && s.dimension === 2) throw new StyleError('Cannot generate velocity ramp in z for a 2d system');
      const sc = latticeScale(sys, unitsW, 'velocity');
      const vlo = numeric(args[1]) * sc[vd], vhi = numeric(args[2]) * sc[vd];
      const cd = ['x', 'y', 'z'].indexOf(args[3]);
      if (cd < 0) throw new StyleError(`Illegal velocity ramp direction ${args[3]}`);
      const clo = numeric(args[4]) * sc[cd], chi = numeric(args[5]) * sc[cd];
      for (const i of members) {
        let frac = (s.x[3 * i + cd] - clo) / (chi - clo);
        frac = Math.max(0, Math.min(1, frac));
        // sum yes adds the ramp to the old velocity of the ramped component only
        s.v[3 * i + vd] = (old ? old[3 * i + vd] : 0) + (vlo + frac * (vhi - vlo));
      }
      break;
    }
    case 'zero':
      // Velocity::zero formats arg[2] of its own arguments, two words past the keyword: a stale word of an earlier
      // command or a null pointer (measured with native LAMMPS: Unknown velocity zero keyword 1 after a compute with 1
      // there, string pointer is null otherwise). The engine names the keyword given
      if (args[0] !== 'linear' && args[0] !== 'angular') throw new StyleError(`Unknown velocity zero keyword ${args[0]}`);
      if (rigidFix) {
        // zero_momentum / zero_rotation of the rigid fix (the /small styles set up their bodies first)
        if (!/^rigid/.test(rigidFix.style) || !rigidFix.zeroMomentum || !rigidFix.zeroRotation) {
          throw new StyleError(`Velocity rigid used with non-rigid fix-ID ${rigidFix.id}`);
        }
        if (args[0] === 'linear') rigidFix.zeroMomentum();
        else rigidFix.zeroRotation();
      } else if (args[0] === 'linear') zeroMomentum(sys, members);
      else zeroRotation(sys, members);
      break;
  }
  // velocity.html says sum "is used by all styles, except zero", but native's scale (measured with native LAMMPS,
  // black box: velocity scale 2.0 sum yes reaches temperature 2.0, and zero linear sum yes leaves ~0 momentum) does not
  // add the old velocities, so the engine follows the measurement: create adds them after the new ones are made, set and
  // ramp add them component by component above, and scale and zero do not add them
  if (old && style === 'create') for (const i of members) for (let d = 0; d < 3; d++) s.v[3 * i + d] += old[3 * i + d];
  sys.refreshComputes();
};

export const zeroMomentum = (sys: System, members: number[]): void => {
  const s = sys.state;
  const p = [0, 0, 0];
  let mt = 0;
  for (const i of members) {
    const m = massOf(s, i);
    mt += m;
    for (let d = 0; d < 3; d++) p[d] += m * s.v[3 * i + d];
  }
  if (!(mt > 0)) return;
  for (const i of members) for (let d = 0; d < 3; d++) s.v[3 * i + d] -= p[d] / mt;
};

/**
 * velocity zero angular and rot yes (Velocity::zero_rotation in src/velocity.cpp at b332f28): omega is the group's
 * Group::omega about its centre of mass (no 2d branch in the source), and v_i -= omega x (r_i - xcm) with the unwrapped
 * positions.
 */
export const zeroRotation = (sys: System, members: number[]): void => {
  const s = sys.state;
  const motion = groupMotionOf(sys, members, 'velocity');
  if (!(motion.masstotal > 0)) return;
  const { xcm, omega: w } = motion;
  const g = sys.geom;
  const u = [0, 0, 0];
  for (const i of members) {
    g.unwrap(s.x, s.image, i, u);
    const r = [u[0] - xcm[0], u[1] - xcm[1], u[2] - xcm[2]];
    s.v[3 * i] -= w[1] * r[2] - w[2] * r[1];
    s.v[3 * i + 1] -= w[2] * r[0] - w[0] * r[2];
    s.v[3 * i + 2] -= w[0] * r[1] - w[1] * r[0];
  }
};

// ----------------------------------------------------------------- atom edits

/**
 * The checks native LAMMPS makes on a delete_atoms command before it deletes any atom (src/delete_atoms.cpp
 * DeleteAtoms::command, the delete_* styles and options at b332f28): the style and its values, then the keywords.
 */
const checkDeleteAtomsArgs = (sys: System, a: string[]): void => {
  if (!sys.hasBox) throw new StyleError('Delete_atoms command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  if (a.length < 1) throw missingArgs('delete_atoms');
  const s = sys.state;
  const group = (g: string) => { if (sys.groups.find(g) < 0) throw new StyleError(`Group ID ${g} requested by delete_atoms does not exist`); };
  const style = a[0];
  let k: number;
  if (style === 'group') {
    if (a.length < 2) throw missingArgs('delete_atoms group');
    group(a[1]);
    k = 2;
  } else if (style === 'region') {
    if (a.length < 2) throw missingArgs('delete_atoms region');
    if (!sys.regions.has(a[1])) throw new StyleError(`Could not find delete_atoms region ID ${a[1]}`);
    k = 2;
  } else if (style === 'overlap') {
    if (a.length < 4) throw missingArgs('delete_atoms overlap');
    numeric(a[1]);
    group(a[2]);
    group(a[3]);
    k = 4;
  } else if (style === 'random') {
    if (a.length < 7) throw missingArgs('delete_atoms random');
    if (a[1] === 'fraction') {
      const f = numeric(a[2]);
      logical(a[3]);
      if (f < 0 || f > 1) throw new StyleError(`Delete_atoms random fraction has invalid value: ${f}`);
    } else if (a[1] === 'count') {
      const n = bnumeric(a[2]);
      logical(a[3]);
      if (n < 0) throw new StyleError(`Delete_atoms random count has invalid value: ${n}`);
    } else throw new StyleError(`Unknown delete_atoms random style: ${a[1]}`);
    group(a[4]);
    if (a[5] !== 'NULL' && !sys.regions.has(a[5])) throw new StyleError(`Could not find delete_atoms random region ID ${a[5]}`);
    inumeric(a[6]);
    k = 7;
  } else if (style === 'porosity') {
    throw new StyleError("The delete_atoms 'porosity' keyword has been removed.\nPlease use: delete_atoms random fraction frac exact group-ID region-ID seed");
  } else if (style === 'variable') {
    if (a.length < 2) throw missingArgs('delete_atoms variable');
    const v = sys.vars.get(a[1]);
    if (!v) throw new StyleError(`Variable name ${a[1]} for delete_atoms does not exist`);
    if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Variable ${a[1]} for delete_atoms is invalid style`);
    k = 2;
  } else throw new StyleError(`Unknown delete_atoms sub-command: ${style}`);
  for (; k < a.length; k += 2) {
    const key = a[k];
    if (!['compress', 'condense', 'bond', 'mol'].includes(key)) throw new StyleError(`Unknown delete_atoms option: ${key}`);
    if (k + 2 > a.length) throw missingArgs(`delete_atoms ${key}`);
    const on = logical(a[k + 1]);
    if (key === 'bond' && on && isTemplateStyle(s.atomStyle)) throw new StyleError('Cannot use delete_atoms bond yes with atom_style template');
    if (key === 'bond' && on && !isMolecularStyle(s.atomStyle)) throw new StyleError('Cannot use delete_atoms bond yes for non-molecular systems');
    if (key === 'mol' && on && !hasMolecule(s)) throw new StyleError('Delete_atoms mol yes requires atom attribute molecule');
  }
};

/** delete_atoms style args [compress/bond/mol] — delete_atoms.html. */
const deleteAtomsCmd: Handler = ({ sys }, a) => {
  checkDeleteAtomsArgs(sys, a);
  const s = sys.state;
  const del = new Uint8Array(s.n);
  let rest: string[];
  switch (a[0]) {
    case 'group': {
      const bit = sys.groupBit(a[1] ?? '');
      for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) del[i] = 1;
      rest = a.slice(2);
      break;
    }
    case 'region': {
      const r = sys.region(a[1] ?? '');
      for (let i = 0; i < s.n; i++) if (r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) del[i] = 1;
      rest = a.slice(2);
      break;
    }
    case 'variable': {
      const v = sys.atomVariable(a[1] ?? '');
      for (let i = 0; i < s.n; i++) if (v[i] !== 0) del[i] = 1;
      rest = a.slice(2);
      break;
    }
    case 'overlap': {
      // "delete one atom from pairs of atoms within the cutoff"
      const cut = num(a[1], 'cutoff');
      const b1 = sys.groupBit(a[2] ?? ''), b2 = sys.groupBit(a[3] ?? '');
      const g = sys.geom;
      const d = [0, 0, 0];
      // delete_atoms.html: "Since a neighbor list is used to find overlapping atom pairs, it also means that you must
      // define a pair style with the minimum force cutoff distance between any pair of atoms types (plus the neighbor
      // skin) >= the specified overlap cutoff." Only pairs within their own pair cutoff plus the skin
      // are seen (measured below).
      const pc = sys.ff.pair;
      if (!pc) throw new StyleError('Delete_atoms requires a pair style be defined');
      // the cutoff table is set when a pair style is initialized (native delete_atoms initializes its pair style too)
      if (pc.ntypes !== s.ntypes) pc.allocate(s.ntypes);
      pc.init(sys.styleContext());
      const nt = s.ntypes + 1;
      const skin = sys.nb.skin;
      // Measured with native LAMMPS (black box): an overlap cutoff above the largest pair cutoff plus the skin stops
      // with Delete_atoms cutoff > max neighbor cutoff (lj/cut 2.5 with pair_coeff 1 1 cutoff 0.5, skin 0.3, overlap
      // 1.0); with a type pair of cutoff 0.5 among larger ones, that pair is seen at 0.79 apart but not at 0.81.
      let maxReach = 0;
      for (let ti = 1; ti < nt; ti++) for (let tj = 1; tj < nt; tj++) maxReach = Math.max(maxReach, pc.cut[ti * nt + tj] + skin);
      if (cut > maxReach) throw new StyleError('Delete_atoms cutoff > max neighbor cutoff');
      // delete_atoms.html: "Only pairs where one of the two atoms is in the first group specified and the other atom is in
      // the second group are considered. The atom that is in the first group is the one that is deleted." As native
      // LAMMPS does (DeleteAtoms::delete_overlap, src/delete_atoms.cpp at b332f28): atom I of group 1 is deleted when an atom J
      // of group 2 lies within the cutoff, unless J is an owned atom that is already deleted; a periodic image of J is
      // never checked against the deletions (single processor: the image is the ghost of the owned atom)
      const raw = [0, 0, 0];
      // the atoms J that can lie within the cutoff of I: the cells of a grid at least `cut` wide (all atoms in a
      // triclinic box). Whether I is deleted depends only on the set of such J, so the result is that of the all-pairs loop.
      const ident = Array.from({ length: s.n }, (_, k) => k);
      const L = [g.hi[0] - g.lo[0], g.hi[1] - g.lo[1], g.hi[2] - g.lo[2]];
      const per = [g.periodic[0], g.periodic[1], g.periodic[2] && sys.dimension === 3];
      const near = cellCandidates(s, ident, per, g.triclinic ? 0 : cut, L, b2, false, b1);
      for (let i = 0; i < s.n; i++) {
        if (del[i] || !(s.mask[i] & b1)) continue;
        for (const j of near(i)) {
          if (!(s.mask[j] & b2)) continue;
          raw[0] = s.x[3 * i] - s.x[3 * j]; raw[1] = s.x[3 * i + 1] - s.x[3 * j + 1]; raw[2] = s.x[3 * i + 2] - s.x[3 * j + 2];
          d[0] = raw[0]; d[1] = raw[1]; d[2] = raw[2];
          g.minimumImage(d);
          const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
          const reach = pc.cut[s.type[i] * nt + s.type[j]] + skin;
          if (rsq > reach * reach) continue;
          if (rsq >= cut * cut) continue;
          const image = d[0] !== raw[0] || d[1] !== raw[1] || d[2] !== raw[2];
          if (!image && del[j]) continue;
          del[i] = 1;
          break;
        }
      }
      rest = a.slice(4);
      break;
    }
    case 'random': {
      const [, ranstyle, value, eflag, gid, rid, seed] = a;
      const bit = sys.groupBit(gid ?? '');
      const r = rid && rid !== 'NULL' ? sys.region(rid) : null;
      const elig: number[] = [];
      for (let i = 0; i < s.n; i++) if ((s.mask[i] & bit) && (!r || r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]))) elig.push(i);
      const rng = new Rng(int(seed, 'seed'));
      if (ranstyle === 'fraction') {
        const f = num(value, 'fraction');
        if (yesno(eflag, 'eflag')) {
          const want = Math.round(f * elig.length);
          for (let i = elig.length - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [elig[i], elig[j]] = [elig[j], elig[i]]; }
          for (const i of elig.slice(0, want)) del[i] = 1;
        } else {
          // delete_atoms.html, eflag: "If eflag is set to no, then the number of deleted atoms will be approximate, but
          // the operation will be fast." One draw per eligible atom; the stream is native's (nativeRandom.ts), which
          // the measured run with native LAMMPS (fcc 80 atoms, seed 4321, 0.3) matches
          const mark = nativeRandom('random').deleteFraction(int(seed, 'seed'), f, elig.length);
          elig.forEach((i, k) => { if (mark[k]) del[i] = 1; });
        }
      } else if (ranstyle === 'count') {
        let want = int(value, 'count');
        if (want > elig.length) {
          if (yesno(eflag, 'eflag')) throw new StyleError(`delete_atoms random count ${want} is more than the ${elig.length} eligible atoms`);
          sys.warn('delete_atoms random count exceeds eligible atoms; deleting all of them');
          want = elig.length;
        }
        for (let i = elig.length - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [elig[i], elig[j]] = [elig[j], elig[i]]; }
        for (const i of elig.slice(0, want)) del[i] = 1;
      } else throw new StyleError('delete_atoms random needs fraction or count');
      rest = a.slice(7);
      break;
    }
    case 'porosity': throw new StyleError('delete_atoms porosity was replaced by delete_atoms random in LAMMPS');
    default: throw new StyleError(`unknown delete_atoms style '${a[0] ?? ''}'`);
  }
  const kw = keywords(rest, { compress: 1, condense: 1, bond: 1, mol: 1 }, 'delete_atoms');
  if (kw.has('mol') && yesno(kw.get('mol')![0], 'mol')) {
    // "mol yes: delete entire molecules if any atom of the molecule is deleted"
    const mols = new Set<number>();
    for (let i = 0; i < s.n; i++) if (del[i] && s.molecule[i]) mols.add(s.molecule[i]);
    for (let i = 0; i < s.n; i++) if (mols.has(s.molecule[i])) del[i] = 1;
  }
  // delete_atoms.html: compress "is enabled by default for atomic systems"; "For molecular systems
  // (see the :doc:`atom_style <atom_style>` command), the atom ID re-assignment now calls the
  // :doc:`reset_atoms id <reset_atoms>` command internally.  For backward compatibility, the default
  // setting is *no* in this case."; "the *compress* and the *condense* keywords cannot be used at
  // the same time.  Whichever of the two is used last will be applied."
  const molecular = isMolecularStyle(s.atomStyle);
  let mode: 'compress' | 'condense' | 'none' = molecular ? 'none' : 'compress';
  for (let k = 0; k < rest.length; k += 2) {
    if (rest[k] === 'compress') mode = yesno(rest[k + 1], 'compress') ? 'compress' : 'none';
    else if (rest[k] === 'condense') mode = yesno(rest[k + 1], 'condense') ? 'condense' : 'none';
  }
  // Measured with native LAMMPS (black box, 10 atoms, delete 2 3 7): an atomic system needs an atom
  // map for condense (error below); on a molecular system condense yes leaves the IDs as they are
  // (with and without bonds), and compress yes renumbers in atom-list order like an atomic system,
  // bonded entries following the new IDs.
  if (mode === 'condense' && !molecular && !sys.atomMapSet) throw new StyleError("Using 'condense yes' option requires an atom map");
  if (mode === 'condense' && molecular) mode = 'none';
  const dropBonds = kw.has('bond') && yesno(kw.get('bond')![0], 'bond');
  const t = s.topo;
  const nTopo = t.bonds.n + t.angles.n + t.dihedrals.n + t.impropers.n;
  const n = sys.deleteAtoms(del);
  if (!dropBonds && t.bonds.n + t.angles.n + t.dihedrals.n + t.impropers.n < nTopo) {
    // delete_atoms.html: "you must be careful not to end up with bonded interactions that are stored
    // by remaining atoms but which include deleted atoms"; native keeps them (bond no) and stops at
    // the next run, the engine drops them
    sys.warn('delete_atoms removed bonded interactions that included deleted atoms (native LAMMPS keeps them with bond no and stops with a missing-atoms error at the next run; use bond yes)');
  }
  if (n > 0 && mode === 'compress') {
    // "compress yes ... atom IDs are re-assigned so that they run from 1 to N"; measured with
    // native LAMMPS (black box): in the order of its atom list (SimState.order), not of the old IDs
    const map = new Map<number, number>();
    nativeOrder(s).forEach((i, k) => { map.set(s.id[i], k + 1); s.id[i] = k + 1; });
    for (const list of [t.bonds, t.angles, t.dihedrals, t.impropers]) {
      for (let e = 0; e < list.n * list.width; e++) list.atoms[e] = map.get(list.atoms[e]) ?? list.atoms[e];
    }
    sys.atomsChanged();
  } else if (n > 0 && mode === 'condense') {
    // "If the *condense* keyword set to *yes*, then after atoms are deleted, the atom IDs are
    // re-assigned in such a way that the order of atom-IDs is preserved." Bonded entries follow.
    const byId = Array.from({ length: s.n }, (_, i) => i).sort((p, q) => s.id[p] - s.id[q]);
    const map = new Map<number, number>();
    byId.forEach((i, k) => { map.set(s.id[i], k + 1); });
    for (const i of byId) s.id[i] = map.get(s.id[i])!;
    for (const list of [s.topo.bonds, s.topo.angles, s.topo.dihedrals, s.topo.impropers]) {
      for (let e = 0; e < list.n * list.width; e++) list.atoms[e] = map.get(list.atoms[e]) ?? list.atoms[e];
    }
    sys.atomsChanged();
  }
  sys.log(`Deleted ${n} atoms, new total = ${s.n}`);
};

/** displace_atoms group style args [units] — displace_atoms.html. */
const displaceAtoms: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/displace_atoms.cpp DisplaceAtoms::command and options at b332f28): the box, the group (its
  // message says display_atoms), the style, the keywords, then the style's values
  if (!sys.hasBox) throw new StyleError('Displace_atoms command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  if (a.length < 2) throw new StyleError('Illegal displace_atoms command');
  const s = sys.state;
  if (sys.groups.find(a[0]) < 0) throw new StyleError(`Could not find display_atoms group ${a[0]}`);
  const bit = sys.groupBit(a[0]);
  const style = a[1];
  const nargs: Record<string, number> = { move: 3, ramp: 6, random: 4, rotate: 7 };
  if (!Object.prototype.hasOwnProperty.call(nargs, style)) throw new StyleError(`Unknown displace_atoms keyword ${style}`);
  if (a.length < 2 + nargs[style]) throw missingArgs('displace_atoms');
  const args = a.slice(2, 2 + nargs[style]);
  let unitsW = 'lattice';
  const opt = a.slice(2 + nargs[style]);
  for (let k = 0; k < opt.length; k += 2) {
    if (opt[k] !== 'units') throw new StyleError(`Unknown displace_atoms keyword ${opt[k]}`);
    if (k + 2 > opt.length) throw missingArgs('displace_atoms units');
    if (opt[k + 1] !== 'box' && opt[k + 1] !== 'lattice') throw new StyleError(`Unknown displace_atoms units argument ${opt[k + 1]}`);
    unitsW = opt[k + 1];
  }
  const sc = latticeScale(sys, unitsW, 'displace_atoms');
  const members: number[] = [];
  for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) members.push(i);
  switch (style) {
    case 'move': {
      // native LAMMPS evaluates the components in turn, each after the previous one moved the
      // atoms (atom-style dx=1, dy=x, dz=y on an atom at (1,1,1) gives (2,3,4)); a 2d system ignores the z value
      for (let d = 0; d < s.dimension; d++) {
        const w = args[d];
        let comp: (i: number) => number;
        if (w.startsWith('v_')) {
          const name = w.slice(2);
          const v = sys.vars.get(name);
          if (!v) throw new StyleError(`Variable ${name} for displace_atoms move does not exist`);
          if (v.style === 'atom' || v.style === 'atomfile') { const arr = sys.atomVariable(name); comp = (i) => arr[i] * sc[d]; } else if (v.style === 'equal' || v.style === 'timer' || v.style === 'internal') {
            const x = sys.equalVariable(name) * sc[d];
            comp = () => x;
          } else throw new StyleError(`Variable ${name} for displace_atoms is invalid style`);
        } else {
          const x = numeric(w) * sc[d];
          comp = () => x;
        }
        for (const i of members) s.x[3 * i + d] += comp(i);
      }
      break;
    }
    case 'ramp': {
      const dd = ['x', 'y', 'z'].indexOf(args[0]);
      if (dd < 0) throw new StyleError(`Unknown displace_atoms ramp dimension ${args[0]}`);
      if (s.dimension === 2 && dd === 2) throw new StyleError('Must not displace atoms in z-direction with 2d system');
      const dlo = numeric(args[1]) * sc[dd], dhi = numeric(args[2]) * sc[dd];
      const cd = ['x', 'y', 'z'].indexOf(args[3]);
      if (cd < 0) throw new StyleError(`Unknown displace_atoms ramp dimension ${args[3]}`);
      const clo = numeric(args[4]) * sc[cd], chi = numeric(args[5]) * sc[cd];
      for (const i of members) {
        const f = Math.max(0, Math.min(1, (s.x[3 * i + cd] - clo) / (chi - clo)));
        s.x[3 * i + dd] += dlo + f * (dhi - dlo);
      }
      break;
    }
    case 'random': {
      // displace_atoms.html: "Random numbers are used in such a way that the displacement of a particular atom is the
      // same, regardless of how many processors are being used." The per-atom streams are native's (nativeRandom.ts).
      const dx = [0, 1, 2].map((d) => numeric(args[d]) * sc[d]);
      const seed = inumeric(args[3]);
      if (seed <= 0) throw new StyleError(`Illegal displace_atoms random seed ${args[3]}`);
      if (s.dimension === 2 && dx[2] !== 0) throw new StyleError('Must not displace atoms in z-direction with 2d system');
      const rnd = nativeRandom('random');
      const disp = new Float64Array(3);
      for (const i of members) {
        rnd.displaceRandom(seed, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], dx[0], dx[1], dx[2], s.dimension, disp);
        for (let d = 0; d < 3; d++) s.x[3 * i + d] += disp[d];
      }
      break;
    }
    case 'rotate': {
      const P = [0, 1, 2].map((d) => numeric(args[d]) * sc[d]);
      let R = [numeric(args[3]), numeric(args[4]), numeric(args[5])];
      const theta = numeric(args[6]);
      if (s.dimension === 2 && (R[0] !== 0 || R[1] !== 0)) throw new StyleError('Invalid displace_atoms rotate axis for 2d system');
      const len = Math.hypot(R[0], R[1], R[2]);
      if (!(len > 0)) throw new StyleError('Zero length rotation vector with displace_atoms');
      R = R.map((x) => x / len);
      const th = (theta * Math.PI) / 180;
      const c = libm.cos(th), sn = libm.sin(th);
      const g = sys.geom;
      for (const i of members) {
        // rotate the unwrapped position, then let pbc remap it
        const u = [0, 0, 0];
        g.unwrap(s.x, s.image, i, u);
        const v = [u[0] - P[0], u[1] - P[1], u[2] - P[2]];
        const dot = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
        const cr = [R[1] * v[2] - R[2] * v[1], R[2] * v[0] - R[0] * v[2], R[0] * v[1] - R[1] * v[0]];
        for (let d = 0; d < 3; d++) s.x[3 * i + d] = P[d] + v[d] * c + cr[d] * sn + R[d] * dot * (1 - c);
        s.image[3 * i] = s.image[3 * i + 1] = s.image[3 * i + 2] = 0;
      }
      // displace_atoms.html: "If the defined :doc:`atom_style <atom_style>` assigns an orientation to each atom
      // (:doc:`atom styles <atom_style>` ellipsoid, line, tri, body), then that property is also updated
      // appropriately to correspond to the atom's rotation." The ellipsoid quaternion q becomes r q, with r the
      // rotation by theta about R (the composition measured with native LAMMPS, black box).
      if (s.quat) {
        const h = th / 2, rs = libm.sin(h);
        const r = [libm.cos(h), R[0] * rs, R[1] * rs, R[2] * rs];
        for (const i of members) {
          if (!isEllipsoid(s, i)) continue;
          const q = s.quat.subarray(4 * i, 4 * i + 4);
          const w = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3];
          const x = r[0] * q[1] + r[1] * q[0] + r[2] * q[3] - r[3] * q[2];
          const y = r[0] * q[2] - r[1] * q[3] + r[2] * q[0] + r[3] * q[1];
          const z = r[0] * q[3] + r[1] * q[2] - r[2] * q[1] + r[3] * q[0];
          const len = Math.hypot(w, x, y, z);
          q[0] = w / len; q[1] = x / len; q[2] = y / len; q[3] = z / len;
        }
      }
      break;
    }
  }
  if (s.dimension === 2) for (const i of members) s.x[3 * i + 2] = 0;
  // As native LAMMPS does (src/displace_atoms.cpp at b332f28): every atom is remapped, and in a triclinic box
  // all coordinates then go to lamda and back once more (x2lamda/lamda2x around the atom migration), which
  // can change their last bits; velocity loop geom hashes those bits
  for (let i = 0; i < s.n; i++) sys.geom.remap(s.x, s.image, i);
  if (sys.geom.triclinic) for (let i = 0; i < s.n; i++) sys.geom.lamdaRoundTrip(s.x, i);
  sys.bump();
  sys.nb.lastBuild = -1;
};

/**
 * replicate nx ny nz — replicate.html: "Replicate the current simulation
 * one or more times in each dimension." New atoms get IDs offset by the
 * old maximum ID per copy; bonds are replicated with the copies' IDs,
 * using image flags to keep periodic bonds intact.
 */
const replicate: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/replicate.cpp Replicate::command at b332f28)
  if (!sys.hasBox) throw new StyleError('Replicate command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  if (a.length < 3 || a.length > 4) throw new StyleError('Illegal number of arguments for replicate command');
  const s = sys.state;
  const rep = [inumeric(a[0]), inumeric(a[1]), inumeric(a[2])];
  if (rep.some((r) => r <= 0)) throw new StyleError(`Illegal replication grid ${rep[0]}x${rep[1]}x${rep[2]}. All replications must be > 0`);
  if (rep[0] * rep[1] * rep[2] > 0x7fffffff) {
    throw new StyleError(`Total # of replica is too large: ${rep[0]}x${rep[1]}x${rep[2]} = ${rep[0] * rep[1] * rep[2]}. Please use replicate multiple times`);
  }
  for (const w of a.slice(3)) if (w !== 'bbox' && w !== 'bond/periodic') throw new StyleError(`Unknown replicate keyword ${w}`);
  if (s.dimension === 2 && rep[2] !== 1) throw new StyleError('Cannot replicate 2d simulation in z dimension');
  // As native LAMMPS does (src/replicate.cpp at b332f28, bond_flag): without bond/periodic a copy's topology refers to the
  // atoms of the same copy; with it, each partner is taken from the copy whose image is nearest to the first atom.
  const bondPeriodic = a.slice(3).includes('bond/periodic');
  const g = sys.geom;
  const old: SimState = { ...s, x: s.x.slice(0, 3 * s.n), v: s.v.slice(0, 3 * s.n), image: s.image.slice(0, 3 * s.n), type: s.type.slice(0, s.n), id: s.id.slice(0, s.n), mask: s.mask.slice(0, s.n), molecule: s.molecule.slice(0, s.n), q: s.q.slice(0, s.n) };
  // measured with native LAMMPS: "Cannot replicate with fixes that store per-atom quantities"
  for (const f of sys.fixes) if (f.style === 'property/atom') throw new StyleError('Cannot replicate with fixes that store per-atom quantities');
  // every other per-atom field (sphere radius/mass/omega, dipoles, ellipsoid shape/quat/angmom) is copied as is
  const base = gatherAtoms(s, Array.from({ length: s.n }, (_, i) => i));
  const maxId = maxAtomId(s);
  let maxMol = 0;
  for (let i = 0; i < s.n; i++) maxMol = Math.max(maxMol, s.molecule[i]);
  const ncopy = rep[0] * rep[1] * rep[2];
  const n0 = s.n;
  const x = new Float64Array(3 * n0 * ncopy), v = new Float64Array(3 * n0 * ncopy);
  const type = new Int32Array(n0 * ncopy), id = new Int32Array(n0 * ncopy), mask = new Int32Array(n0 * ncopy);
  const mol = new Int32Array(n0 * ncopy), q = new Float64Array(n0 * ncopy), image = new Int32Array(3 * n0 * ncopy);
  let c = 0;
  const u = [0, 0, 0];
  // Measured with native LAMMPS (black box): right after replicate, a dump with sort off lists the copies with the
  // x index outermost and z innermost (replicate 2 3 2 gives copies 0, 6, 2, 8, 4, 10, 1, ...), so the copies are
  // stored in that order; the IDs of each copy are unchanged.
  for (let kx = 0; kx < rep[0]; kx++) for (let ky = 0; ky < rep[1]; ky++) for (let kz = 0; kz < rep[2]; kz++) {
    const copy = (kz * rep[1] + ky) * rep[0] + kx;
    for (let i = 0; i < n0; i++, c++) {
      g.unwrap(old.x, old.image, i, u);
      x[3 * c] = u[0] + kx * g.lx + ky * g.xy + kz * g.xz;
      x[3 * c + 1] = u[1] + ky * g.ly + kz * g.yz;
      x[3 * c + 2] = u[2] + kz * g.lz;
      for (let d = 0; d < 3; d++) v[3 * c + d] = old.v[3 * i + d];
      type[c] = old.type[i]; mask[c] = old.mask[i]; q[c] = old.q[i];
      id[c] = old.id[i] + copy * maxId;
      mol[c] = old.molecule[i] ? old.molecule[i] + copy * maxMol : 0;
    }
  }
  // new box
  const b = s.box;
  for (let d = 0; d < 3; d++) b.hi[d] = b.lo[d] + (b.hi[d] - b.lo[d]) * rep[d];
  b.tilt = [b.tilt[0] * rep[1], b.tilt[1] * rep[2], b.tilt[2] * rep[2]];
  b.minLo = [...b.lo]; b.minHi = [...b.hi];
  // topology copies: partner atoms belong to the copy whose image keeps the bond short
  const topo = s.topo;
  const lists = [topo.bonds, topo.angles, topo.dihedrals, topo.impropers].map((l) => ({ ...l, atoms: l.atoms.slice(0, l.n * l.width), type: l.type.slice(0, l.n) }));
  for (const l of [topo.bonds, topo.angles, topo.dihedrals, topo.impropers]) l.n = 0;
  s.n = 0;
  s.x = new Float64Array(0); s.v = new Float64Array(0); s.f = new Float64Array(0); s.image = new Int32Array(0);
  s.type = new Int32Array(0); s.id = new Int32Array(0); s.mask = new Int32Array(0); s.molecule = new Int32Array(0); s.q = new Float64Array(0);
  if (s.rmass) s.rmass = new Float64Array(0);
  if (s.radius) s.radius = new Float64Array(0);
  if (s.omega) s.omega = new Float64Array(0);
  if (s.torque) s.torque = new Float64Array(0);
  if (s.mu) s.mu = new Float64Array(0);
  if (s.shape) s.shape = new Float64Array(0);
  if (s.quat) s.quat = new Float64Array(0);
  if (s.angmom) s.angmom = new Float64Array(0);
  for (const cp of s.custom.values()) cp.data = new Float64Array(0);
  const tile = (a: Float64Array | undefined): Float64Array | undefined => {
    if (!a) return undefined;
    const out = new Float64Array(a.length * ncopy);
    for (let k = 0; k < ncopy; k++) out.set(a, k * a.length);
    return out;
  };
  const custom = new Map<string, Float64Array>();
  for (const [name, arr] of base.custom!) custom.set(name, tile(arr)!);
  // atom_style template: the template index and atom are copied with each atom
  const tileI = (a: Int32Array | null): Int32Array | undefined => {
    if (!a) return undefined;
    const out = new Int32Array(a.length * ncopy);
    for (let k = 0; k < ncopy; k++) out.set(a, k * a.length);
    return out;
  };
  appendAtoms(s, {
    x, v, type, id, mask: 0, molecule: mol, q, image, tmplIndex: tileI((base.tmplIndex as Int32Array | undefined) ?? null), tmplAtom: tileI((base.tmplAtom as Int32Array | undefined) ?? null),
    rmass: tile(base.rmass as Float64Array | undefined), radius: tile(base.radius as Float64Array | undefined), omega: tile(base.omega),
    // peri and smd per-atom volume and reference positions are per-atom fields copied as they are (native replicate: pack_restart)
    vfrac: tile(base.vfrac as Float64Array | undefined), x0: tile(base.x0 as Float64Array | undefined),
    mu: tile(base.mu), shape: tile(base.shape), quat: tile(base.quat), angmom: tile(base.angmom), custom,
  });
  s.mask.set(mask);
  sys.setState(s);
  const gnew = sys.geom;
  // wrap new atoms into the new box
  for (let i = 0; i < s.n; i++) gnew.remap(s.x, s.image, i);
  const oldIdx = new Map<number, number>();
  for (let i = 0; i < n0; i++) oldIdx.set(old.id[i], i);
  const ux = (i: number) => { const p = [0, 0, 0]; g.unwrap(old.x, old.image, i, p); return p; };
  [topo.bonds, topo.angles, topo.dihedrals, topo.impropers].forEach((l, li) => {
    const src = lists[li];
    for (let kz = 0; kz < rep[2]; kz++) for (let ky = 0; ky < rep[1]; ky++) for (let kx = 0; kx < rep[0]; kx++) {
      for (let e = 0; e < src.n; e++) {
        const ids = Array.from(src.atoms.subarray(e * src.width, (e + 1) * src.width));
        const p0 = ux(oldIdx.get(ids[0])!);
        const out = ids.map((aid, w) => {
          // which copy holds the partner nearest to atom 0 of this copy
          const p = ux(oldIdx.get(aid)!);
          const off = [kx, ky, kz];
          if (w > 0 && bondPeriodic) {
            const dd = [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]];
            const sh = [Math.round(dd[0] / g.lx), Math.round(dd[1] / g.ly), Math.round(dd[2] / g.lz)];
            for (let d = 0; d < 3; d++) off[d] = (((off[d] + sh[d]) % rep[d]) + rep[d]) % rep[d];
            void sh;
          }
          return aid + ((off[2] * rep[1] + off[1]) * rep[0] + off[0]) * maxId;
        });
        pushTopo(l, src.type[e], out);
      }
    }
  });
  sys.atomsChanged();
  sys.log(`Replicated to ${s.n} atoms`);
};

/** change_box group-ID parameter args ... [units] — change_box.html (x/y/z final/delta/scale/volume, xy/xz/yz, boundary, ortho, triclinic, set, remap). */
/**
 * The checks native LAMMPS makes on a change_box command before it changes the box (src/change_box.cpp
 * ChangeBox::command and options at b332f28): the group, every operation and its values (x, y, z final, delta, scale
 * or volume after an x/y/z operation; xy, xz, yz final or delta; boundary; ortho, triclinic, set, remap), then the
 * units keyword, which ends the operations.
 */
const checkChangeBoxArgs = (sys: System, a: string[]): void => {
  if (!sys.hasBox) throw new StyleError('Change_box command before simulation box is defined\nFor more information see https://docs.lammps.org/err0033');
  if (a.length < 2) throw missingArgs('change_box');
  if (sys.groups.find(a[0]) < 0) throw new StyleError(`Group ID ${a[0]} requested by change_box does not exist`);
  const two = sys.dimension === 2;
  let k = 1, nops = 0;
  // the last operation, for volume: an x/y/z operation, its dimension and the volume dimensions given so far
  let last: { xyz: boolean; dim: string; nvol: number } | null = null;
  while (k < a.length) {
    const p = a[k];
    if (p === 'x' || p === 'y' || p === 'z') {
      if (k + 2 > a.length) throw new StyleError(`Illegal change_box ${p} command: missing argument(s)`);
      if (two && p === 'z') throw new StyleError('Cannot change_box in z dimension for 2d simulation');
      const st = a[k + 1];
      if (st === 'final' || st === 'delta') {
        if (k + 4 > a.length) throw new StyleError(`Illegal change_box ${p} ${st} command: missing argument(s)`);
        numeric(a[k + 2]); numeric(a[k + 3]);
        last = { xyz: true, dim: p, nvol: 0 }; nops++; k += 4;
      } else if (st === 'scale') {
        if (k + 3 > a.length) throw new StyleError(`Illegal change_box ${p} scale command: missing argument(s)`);
        numeric(a[k + 2]);
        last = { xyz: true, dim: p, nvol: 0 }; nops++; k += 3;
      } else if (st === 'volume') {
        if (!last || !last.xyz || last.dim === p || last.nvol >= 2) throw new StyleError('Change_box volume used incorrectly');
        last.nvol++;
        k += 2;
      } else throw new StyleError(`Unknown change_box ${p} argument: ${st}`);
    } else if (p === 'xy' || p === 'xz' || p === 'yz') {
      if (k + 2 > a.length) throw new StyleError(`Illegal change_box ${p} command: missing argument(s)`);
      if (two && p !== 'xy') throw new StyleError('Cannot change_box in xz or yz for 2d simulation');
      const st = a[k + 1];
      if (st !== 'final' && st !== 'delta') throw new StyleError(`Unknown change_box ${p} argument: ${st}`);
      if (k + 3 > a.length) throw new StyleError(`Illegal change_box ${p} ${st} command: missing argument(s)`);
      numeric(a[k + 2]);
      last = { xyz: false, dim: p, nvol: 0 }; nops++; k += 3;
    } else if (p === 'boundary') {
      if (k + 4 > a.length) throw missingArgs('change_box boundary');
      last = { xyz: false, dim: p, nvol: 0 }; nops++; k += 4;
    } else if (p === 'ortho' || p === 'triclinic' || p === 'set' || p === 'remap') {
      last = { xyz: false, dim: p, nvol: 0 }; nops++; k += 1;
    } else break;
  }
  if (nops === 0) throw new StyleError(`Unknown change_box keyword: ${a[k]}`);
  for (; k < a.length; k += 2) {
    if (a[k] !== 'units') throw new StyleError(`Unknown change_box keyword: ${a[k]}`);
    if (k + 2 > a.length) throw missingArgs('change_box units');
    if (a[k + 1] !== 'box' && a[k + 1] !== 'lattice') throw new StyleError(`Invalid change_box units argument: ${a[k + 1]}`);
  }
};

const changeBox: Handler = ({ sys }, a) => {
  checkChangeBoxArgs(sys, a);
  const s = sys.state;
  const bit = sys.groupBit(a[0] ?? '');
  const ops = a.slice(1);
  // units keyword applies to all
  let unitsW = 'lattice';
  const ui = ops.indexOf('units');
  if (ui >= 0) { unitsW = ops[ui + 1]; ops.splice(ui, 2); }
  const sc = latticeScale(sys, unitsW, 'change_box');
  const b = s.box;
  // As native LAMMPS does (src/change_box.cpp at b332f28): a triclinic box first takes every atom to lamda
  // coordinates, wraps it (pbc) and back, before the box state is saved for remap
  if (sys.geom.triclinic) for (let i = 0; i < s.n; i++) sys.geom.remap(s.x, s.image, i);
  let saved = cloneBox(b);
  const geomOf = (box: SimState['box']) => new Geometry(box);
  for (let k = 0; k < ops.length;) {
    const p = ops[k];
    if (p === 'x' || p === 'y' || p === 'z') {
      const d = 'xyz'.indexOf(p);
      const st = ops[k + 1];
      // the box volume before this operation (native domain->xprd * yprd (* zprd))
      const oldvol = (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (sys.dimension === 2 ? 1 : b.hi[2] - b.lo[2]);
      // native scales about the middle as mid + factor * (bound - mid)
      const scaleDim = (dd: number, f: number): void => {
        const mid = 0.5 * (b.lo[dd] + b.hi[dd]);
        b.lo[dd] = mid + f * (b.lo[dd] - mid);
        b.hi[dd] = mid + f * (b.hi[dd] - mid);
      };
      if (st === 'final') { b.lo[d] = num(ops[k + 2], 'lo') * sc[d]; b.hi[d] = num(ops[k + 3], 'hi') * sc[d]; k += 4; } else if (st === 'delta') { b.lo[d] += num(ops[k + 2], 'dlo') * sc[d]; b.hi[d] += num(ops[k + 3], 'dhi') * sc[d]; k += 4; } else if (st === 'scale') {
        scaleDim(d, num(ops[k + 2], 'factor'));
        k += 3;
      } else if (st === 'volume') throw new StyleError('Change_box volume used incorrectly');
      else throw new StyleError(`change_box ${p}: style must be final, delta, scale or volume`);
      // As native LAMMPS does (src/change_box.cpp at b332f28, volume_preserve): x, y or z "volume" right after an x, y
      // or z operation names a dimension (at most two, not the operation's own) rescaled about its middle so that the box
      // volume is the one before the operation; with two, both are scaled by the square root of the factor.
      // change_box.html: "The volume style changes the specified dimension in such a way that the overall box volume
      // remains constant with respect to the operation performed by the preceding keyword."
      const vdims: number[] = [];
      while ((ops[k] === 'x' || ops[k] === 'y' || ops[k] === 'z') && ops[k + 1] === 'volume') {
        const d2 = 'xyz'.indexOf(ops[k]);
        if (d2 === d || vdims.length >= 2) throw new StyleError('Change_box volume used incorrectly');
        vdims.push(d2);
        k += 2;
      }
      if (vdims.length) {
        const newvol = (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (sys.dimension === 2 ? 1 : b.hi[2] - b.lo[2]);
        const factor = oldvol / newvol;
        if (vdims.length === 1) scaleDim(vdims[0], factor);
        else { scaleDim(vdims[0], Math.sqrt(factor)); scaleDim(vdims[1], Math.sqrt(factor)); }
      }
    } else if (p === 'xy' || p === 'xz' || p === 'yz') {
      // As native LAMMPS does (src/change_box.cpp at b332f28)
      if (!b.triclinic) throw new StyleError('Cannot change box tilt factors for orthogonal box');
      const t = ['xy', 'xz', 'yz'].indexOf(p);
      const scale = t === 2 ? sc[1] : sc[0];
      if (ops[k + 1] === 'final') b.tilt[t] = num(ops[k + 2], 'tilt') * scale;
      else if (ops[k + 1] === 'delta') b.tilt[t] += num(ops[k + 2], 'dtilt') * scale;
      else throw new StyleError(`change_box ${p}: style must be final or delta`);
      k += 3;
    } else if (p === 'boundary') {
      const nb = boundaryWords(ops.slice(k + 1, k + 4), 'change_box');
      // As native LAMMPS does (src/change_box.cpp at b332f28)
      if (sys.dimension === 2 && nb[2][0] !== 'p') throw new StyleError('Cannot change box z boundary to non-periodic for a 2d simulation');
      b.boundary = nb as SimState['box']['boundary'];
      b.periodic = [nb[0][0] === 'p', nb[1][0] === 'p', nb[2][0] === 'p'];
      sys.boundary = b.boundary;
      // "all non-zero image flags for non-periodic dimensions will be reset" (as read_data)
      for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) if (!b.periodic[d]) s.image[3 * i + d] = 0;
      k += 4;
    } else if (p === 'ortho') {
      if (b.tilt.some((t) => t !== 0)) throw new StyleError('change_box ortho needs all tilt factors to be 0');
      b.triclinic = false;
      k++;
    } else if (p === 'triclinic') { b.triclinic = true; k++; } else if (p === 'set') { saved = cloneBox(b); k++; } else if (p === 'remap') {
      // map group atoms from the saved box to the current one, in fractional coordinates: native x2lamda with the
      // saved boxlo and h_inv, then lamda2x, which adds boxlo last (change_box.cpp REMAP at b332f28)
      const g0 = geomOf(saved), g1 = geomOf(b);
      const lam = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & bit)) continue;
        g0.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
        s.x[3 * i] = g1.lx * lam[0] + g1.xy * lam[1] + g1.xz * lam[2] + g1.lo[0];
        s.x[3 * i + 1] = g1.ly * lam[1] + g1.yz * lam[2] + g1.lo[1];
        s.x[3 * i + 2] = g1.lz * lam[2] + g1.lo[2];
      }
      saved = cloneBox(b);
      k++;
    } else throw new StyleError(`unknown change_box parameter '${p}'`);
  }
  b.minLo = [...b.lo]; b.minHi = [...b.hi];
  sys.geom.update();
  // As native LAMMPS does (src/change_box.cpp at b332f28): every atom is remapped into the new box, and in a triclinic
  // box all coordinates then go to lamda and back once more (around the atom migration), which can change their last
  // bits; velocity loop geom hashes those bits
  for (let i = 0; i < s.n; i++) sys.geom.remap(s.x, s.image, i);
  if (sys.geom.triclinic) for (let i = 0; i < s.n; i++) sys.geom.lamdaRoundTrip(s.x, i);
  sys.checkLostBoxChange();
  for (const f of sys.fixes) f.boxChanged?.();
  sys.nb.lastBuild = -1;
  sys.bump();
};

/** create_bonds single/bond|angle|dihedral|improper type atoms... — create_bonds.html (single styles). */
/*
 * create_bonds many group1 group2 btype rmin rmax — docs.lammps.org/create_bonds.html.
 * Quoted from the page: "Pairs of atoms that are already bonded cannot appear in the
 * neighbor list, to avoid creation of duplicate bonds." and "1--3 or 1--4 atoms are
 * those which are two hops or three hops apart in the bond topology."
 * "When using periodic boundary conditions, the box length in each periodic dimension
 * must be larger than rmax, so that no bonds are created between the system and its
 * own periodic image."
 *
 * Measured with native LAMMPS (black box, 1 process), on small periodic lattices:
 *   - a bond joins a pair of atoms whose distance D lies in [rmin, rmax] (both ends
 *     inclusive: a sc lattice with rmin = rmax = 1.0 gives its 81 nearest-neighbour bonds);
 *   - each periodic image of a pair within the range is a separate bond (two images at
 *     1.4 and 1.6 in a box of length 3 gave two bonds between the same IDs); an atom is
 *     never bonded to its own image;
 *   - the stored bond lists the lower atom ID first and the lines are grouped by that
 *     first atom in ascending ID order (write_data "Bonds" section);
 *   - a pair is skipped when its special weight is 0 for its 1-2, 1-3 or 1-4 relation:
 *     a 1-3 pair was created with special_bonds lj 0 1 1 and not with lj 0 0 0 or 0 0 1;
 *     the skip applies to the closest image of the pair only: after a first many all all
 *     (81 bonds, 27 atoms, box 3) a many hi all with rmin 1.9 rmax 2.1 added 34 bonds, among
 *     them a second bond between 2 and 20 through the image at distance 2 (the NN bond uses
 *     the image at distance 1);
 *   - group1/group2 are symmetric: the set of bonds is the same with the groups swapped;
 *   - rmax above the box length of a periodic dimension is an error (rmax equal to it is
 *     accepted); rmax above the largest pair cutoff plus neighbor skin is an error;
 *   - a non-zero 1-2 special weight is an error; any optional keyword with many is an error;
 *   - non-periodic boundaries create only the shift-0 images (a 5x5x5 lattice with x
 *     non-periodic gave 350 bonds against 375 periodic).
 * Not reproduced: the order of the partners of one atom (native follows its neighbour
 * list order, e.g. 19 7 3 2 4 10 for atom 1; here partners are in ascending ID order), and
 * the limit of bonds per atom set by extra/bond/per/atom (native errors when it is exceeded;
 * create_box and read_data do not keep that value, so it is not enforced here).
 */
/**
 * create_bonds many (and delete_atoms overlap, with `later` false): for the atom of rank u (atoms in the order
 * `order`, u in the groups `queryBits`), the ranks v > u (with `later` false: every v other than u) of the atoms of the
 * groups `bits` that can lie within `rmax` of it under some periodic image, in increasing order. With rmax > 0 (orthogonal box) the atoms of the
 * groups (`bits`) are binned into cells at least rmax wide (positions wrapped into periodic dimensions, the atoms' extent
 * in the others), so a pair within rmax at any image is in two neighboring cells; with rmax = 0 every later rank is a
 * candidate. The caller tests every image of each candidate, so the bonds and their order are those of the all-pairs loop.
 */
const cellCandidates = (s: SimState, order: number[], periodic: boolean[], rmax: number, L: number[], bits: number, later = true, queryBits = bits): ((u: number) => Iterable<number>) => {
  const n = order.length;
  if (!(rmax > 0)) {
    return later
      ? function* (u: number) { for (let v = u + 1; v < n; v++) yield v; }
      : function* (u: number) { for (let v = 0; v < n; v++) if (v !== u) yield v; };
  }
  const lo = [0, 0, 0], size = [1, 1, 1], nb = [1, 1, 1];
  for (let d = 0; d < 3; d++) {
    let mn = Infinity, mx = -Infinity;
    if (periodic[d]) { mn = s.box.lo[d]; mx = mn + L[d]; } else {
      for (let k = 0; k < n; k++) {
        const i = order[k];
        if (!(s.mask[i] & bits)) continue;
        const v = s.x[3 * i + d];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      if (!(mx >= mn)) { mn = 0; mx = 0; }
    }
    const ext = mx - mn;
    // a cell a little wider than rmax, so that rounding in the cell index cannot put two atoms rmax apart two cells apart
    nb[d] = Math.max(1, Math.min(256, Math.floor(ext / (rmax * (1 + 1e-9)))));
    lo[d] = mn;
    size[d] = ext > 0 ? ext / nb[d] : 1;
  }
  const cellOf = (i: number, d: number): number => {
    let v = s.x[3 * i + d] - lo[d];
    if (periodic[d]) { v %= L[d]; if (v < 0) v += L[d]; }
    return Math.min(nb[d] - 1, Math.max(0, Math.floor(v / size[d])));
  };
  const cells = new Map<number, number[]>();
  const key = (cx: number, cy: number, cz: number): number => (cz * nb[1] + cy) * nb[0] + cx;
  const cell = new Int32Array(3 * n);
  for (let k = 0; k < n; k++) {
    const i = order[k];
    if (!(s.mask[i] & bits)) continue;
    for (let d = 0; d < 3; d++) cell[3 * k + d] = cellOf(i, d);
    const c = key(cell[3 * k], cell[3 * k + 1], cell[3 * k + 2]);
    let list = cells.get(c);
    if (!list) cells.set(c, (list = []));
    list.push(k);
  }
  // the distinct neighboring cell coordinates of c along dimension d (periodic dimensions wrap)
  const around = (c: number, d: number): number[] => {
    const out: number[] = [];
    for (let o = -1; o <= 1; o++) {
      let w = c + o;
      if (periodic[d]) w = ((w % nb[d]) + nb[d]) % nb[d];
      else if (w < 0 || w >= nb[d]) continue;
      if (!out.includes(w)) out.push(w);
    }
    return out;
  };
  return (u: number) => {
    const out: number[] = [];
    const iu = order[u];
    if (!(s.mask[iu] & queryBits)) return out;
    // the cell of u (an atom outside the binned atoms' extent of a non-periodic dimension takes the edge cell, whose
    // neighbors still hold every binned atom within rmax of it)
    for (const cz of around(cellOf(iu, 2), 2)) for (const cy of around(cellOf(iu, 1), 1)) for (const cx of around(cellOf(iu, 0), 0)) {
      for (const v of cells.get(key(cx, cy, cz)) ?? []) if (later ? v > u : v !== u) out.push(v);
    }
    return out.sort((a, b) => a - b);
  };
};

const createBondsMany = (sys: System, a: string[]): void => {
  const s = sys.state;
  if (a.length !== 6) throw new StyleError('usage: create_bonds many group1 group2 btype rmin rmax (create_bonds many takes no keywords)');
  const bit1 = sys.groupBit(a[1]), bit2 = sys.groupBit(a[2]);
  const t = int(a[3], 'bond type');
  if (t < 1 || t > s.topo.nbondtypes) throw new StyleError(`bond type ${t} is outside 1..${s.topo.nbondtypes}`);
  const rmin = num(a[4], 'rmin'), rmax = num(a[5], 'rmax');
  if (!(rmin >= 0) || !(rmax >= rmin)) throw new StyleError('create_bonds many needs 0 <= rmin <= rmax');
  const ff = sys.ff;
  if (!ff.pair) throw new StyleError('create_bonds many needs a pair_style (the neighbor list is used to find the pairs)');
  if (ff.kspace) throw new StyleError('create_bonds many cannot be used with a kspace_style (see create_bonds.html restrictions)');
  if (ff.special.lj[0] !== 0) throw new StyleError('create_bonds many requires special_bonds 1-2 (lj) weight 0.0');
  // the system must be ready for a run (create_bonds.html: "similar to what a run command would require"):
  // the force field is initialised and the neighbor cutoff (largest pair cutoff plus skin) is set
  sys.setupNeighbors();
  const cutneigh = sys.nb.cutneighmax;
  if (rmax > cutneigh) throw new StyleError(`create_bonds many: rmax ${rmax} exceeds the neighbor cutoff ${cutneigh} (largest pair cutoff plus neighbor skin)`);
  const g = sys.geom;
  const L = [g.hi[0] - g.lo[0], g.hi[1] - g.lo[1], g.hi[2] - g.lo[2]];
  // the periodic image shifts are the box vectors a = (lx, 0, 0), b = (xy, ly, 0), c = (xz, yz, lz)
  const xy = g.triclinic ? g.xy : 0, xz = g.triclinic ? g.xz : 0, yz = g.triclinic ? g.yz : 0;
  // measured with native LAMMPS (black box): in 2d the z extent is not checked (rmax 1.5 with a 1.17-thick box adds the bonds)
  for (let d = 0; d < (sys.dimension === 2 ? 2 : 3); d++) if (g.periodic[d] && rmax > L[d]) throw new StyleError(`create_bonds many: rmax ${rmax} is larger than the periodic box length ${L[d]} in ${'xyz'[d]}`);
  // relation of two atoms by bond hops (1, 2 or 3); beyond three hops there is no special weight
  const adj = new Map<number, number[]>();
  for (let e = 0; e < s.topo.bonds.n; e++) {
    const i = s.topo.bonds.atoms[2 * e], j = s.topo.bonds.atoms[2 * e + 1];
    if (!adj.has(i)) adj.set(i, []);
    if (!adj.has(j)) adj.set(j, []);
    adj.get(i)!.push(j);
    adj.get(j)!.push(i);
  }
  const hops = new Map<number, Map<number, number>>();
  for (const start of adj.keys()) {
    const dist = new Map<number, number>([[start, 0]]);
    let frontier = [start];
    for (let h = 1; h <= 3; h++) {
      const next: number[] = [];
      for (const u of frontier) for (const v of adj.get(u) ?? []) if (!dist.has(v)) { dist.set(v, h); next.push(v); }
      frontier = next;
    }
    dist.delete(start);
    hops.set(start, dist);
  }
  // 2d: the z direction has no periodic images, whatever its boundary. Measured with native LAMMPS (black box):
  // create_bonds many on the 2d RHEO balloon lattice (z periodic) makes 1263 bonds; counting z images made about 132k.
  const per = [g.periodic[0], g.periodic[1], g.periodic[2] && sys.dimension === 3];
  const lj = ff.special.lj;
  const order = Array.from({ length: s.n }, (_, i) => i).sort((p, q) => s.id[p] - s.id[q]);
  const newBonds: [number, number][] = [];
  const minimumImage = (ex: number, ey: number, ez: number): boolean =>
    !((g.periodic[0] && Math.abs(ex) > 0.5 * L[0]) || (g.periodic[1] && Math.abs(ey) > 0.5 * L[1]) || (g.periodic[2] && Math.abs(ez) > 0.5 * L[2]));
  const range = (periodic: boolean, base: number, len: number): [number, number] =>
    (periodic ? [Math.ceil((-rmax - base) / len), Math.floor((rmax - base) / len)] : [0, 0]);
  // the later atoms (by ID) that can lie within rmax of the atom of rank u, in ID order: every atom for a triclinic box,
  // else the atoms of the 27 neighboring cells of a grid whose cells are at least rmax wide (periodic dimensions wrap)
  const candidates = cellCandidates(s, order, per, g.triclinic ? 0 : rmax, L, bit1 | bit2);
  for (let u = 0; u < order.length; u++) {
    const ia = order[u];
    const ma = s.mask[ia];
    const inA1 = (ma & bit1) !== 0, inA2 = (ma & bit2) !== 0;
    if (!inA1 && !inA2) continue;
    for (const v of candidates(u)) {
      const ib = order[v];
      const mb = s.mask[ib];
      if (!((inA1 && (mb & bit2) !== 0) || (inA2 && (mb & bit1) !== 0))) continue;
      // a special pair (weight 0) is skipped at its minimum images only: a second image of a bonded pair
      // is bonded again (measured: a second many all all added the far image of an existing NN bond). As native
      // LAMMPS does (src/npair.cpp and Domain::minimum_image_check at b332f28): the neighbor build drops an excluded
      // special pair when every periodic component of the separation is at most half the box length, and keeps the
      // other images as plain neighbors
      const h = hops.get(s.id[ia])?.get(s.id[ib]);
      const special = h !== undefined && lj[h - 1] === 0;
      const dx = s.x[3 * ib] - s.x[3 * ia], dy = s.x[3 * ib + 1] - s.x[3 * ia + 1], dz = s.x[3 * ib + 2] - s.x[3 * ia + 2];
      // the images within rmax: z first, then y (shifted by the yz tilt), then x (shifted by the xy and xz tilts)
      const [kzlo, kzhi] = range(per[2], dz, L[2]);
      for (let kz = kzlo; kz <= kzhi; kz++) {
        const ez = dz + (per[2] ? kz * L[2] : 0);
        const by = dy + (per[2] ? kz * yz : 0);
        const [kylo, kyhi] = range(per[1], by, L[1]);
        for (let ky = kylo; ky <= kyhi; ky++) {
          const ey = by + (per[1] ? ky * L[1] : 0);
          const bx = dx + (per[1] ? ky * xy : 0) + (per[2] ? kz * xz : 0);
          const [kxlo, kxhi] = range(per[0], bx, L[0]);
          for (let kx = kxlo; kx <= kxhi; kx++) {
            const ex = bx + (per[0] ? kx * L[0] : 0);
            if (special && minimumImage(ex, ey, ez)) continue;
            const d = Math.sqrt(ex * ex + ey * ey + ez * ez);
            if (d >= rmin && d <= rmax) newBonds.push([s.id[ia], s.id[ib]]);
          }
        }
      }
    }
  }
  for (const [i, j] of newBonds) pushTopo(s.topo.bonds, t, [i, j]);
  sys.atomsChanged();
};

const createBonds: Handler = ({ sys }, a) => {
  // As native LAMMPS does (src/create_bonds.cpp CreateBonds::command at b332f28): "if (narg < 4) utils::missing_cmd_args(FLERR, "create_bonds", error)"
  if (a.length < 4) throw missingArgs('create_bonds');
  if (a[0] === 'many') { createBondsMany(sys, a); return; }
  const s = sys.state;
  const width: Record<string, number> = { 'single/bond': 2, 'single/angle': 3, 'single/dihedral': 4, 'single/improper': 4 };
  const style = a[0];
  if (!(style in width)) throw new StyleError('create_bonds supports many, single/bond, single/angle, single/dihedral and single/improper');
  const kind = style.split('/')[1] as 'bond' | 'angle' | 'dihedral' | 'improper';
  const t = int(a[1], `${kind} type`);
  const ntypes = { bond: s.topo.nbondtypes, angle: s.topo.nangletypes, dihedral: s.topo.ndihedraltypes, improper: s.topo.nimpropertypes }[kind];
  if (t < 1 || t > ntypes) throw new StyleError(`${kind} type ${t} is outside 1..${ntypes}`);
  const ids = a.slice(2, 2 + width[style]).map((w) => int(w, 'atom ID'));
  if (ids.length < width[style]) throw new StyleError(`create_bonds ${style} needs ${width[style]} atom IDs`);
  for (const id of ids) if (sys.indexOfId(id) < 0) throw new StyleError(`create_bonds: atom ${id} does not exist`);
  // native create_bonds (src/create_bonds.cpp at b332f28): the only optional keyword is special, and any other word is
  // "Illegal create_bonds command"
  const kwRest = a.slice(2 + width[style]);
  for (let i = 0; i < kwRest.length; i += 2) if (kwRest[i] !== 'special') throw new StyleError('Illegal create_bonds command');
  const kw = keywords(kwRest, { special: 1 }, 'create_bonds');
  // special no leaves the special list stale until a later special yes (create_bonds.html); the engine
  // always rebuilds it after a topology change, so 'no' would silently differ: rejected instead
  if (kw.has('special') && yesno(kw.get('special')![0], 'create_bonds special') === false) {
    throw new StyleError('create_bonds special no is not supported (the browser engine always rebuilds the special list after a change)');
  }
  pushTopo(s.topo[`${kind}s` as 'bonds'], t, ids);
  sys.atomsChanged();
};

/**
 * delete_bonds group-ID style arg keyword ... — docs.lammps.org/delete_bonds.html:
 *   "style = multi or atom or bond or angle or dihedral or improper or stats"
 *   "atom arg = an atom type or range of types (see below)" (bond, angle, dihedral and improper alike)
 *   "keyword = any or undo or remove or special"
 *   "The default behavior of the delete_bonds command is to turn off interactions by toggling their type to a
 *   negative value, but not to permanently remove the interaction."
 *   "Style atom is the same as style multi except that in addition, one or more of the atoms involved in the
 *   bond, angle, dihedral, or improper interaction must also be of the specified atom type."
 *   "The remove keyword is invoked at the end of the delete_bonds operation.  It causes turned-off bonds (angles,
 *   etc.) to be removed from each atom's data structure"; "Removal does not alter the pairwise 1–2, 1–3, or 1–4
 *   weighting list."; "The weighting list computation treats turned-off bonds the same as turned-on."
 * Measured with native LAMMPS (black box), 4 atoms, 3 bonds, 2 angles: any turns off an interaction with at least
 * one atom in the group; undo turns matching negative types back on; style atom and multi need every atom in the
 * group; a type range 1*2 works and a bare * stops with Invalid range string: *; remove drops every turned-off
 * interaction of every kind (a bond turned off before goes with a later delete_bonds g angle 1 remove); stats
 * changes nothing.
 */
const deleteBonds: Handler = ({ sys }, a) => {
  const s = sys.state;
  const bit = sys.groupBit(a[0] ?? '');
  const style = a[1];
  const kinds = ['bond', 'angle', 'dihedral', 'improper'] as const;
  const styles = ['multi', 'atom', 'stats', ...kinds];
  if (!styles.includes(style)) throw new StyleError(`Illegal delete_bonds command: unknown style '${style ?? ''}'`);
  const typed = style === 'atom' || (kinds as readonly string[]).includes(style);
  let lo = 0, hi = 0;
  if (typed) {
    const tok = a[2];
    if (tok === undefined) throw new StyleError('Illegal delete_bonds command');
    const ntypes = style === 'atom' ? s.ntypes : { bond: s.topo.nbondtypes, angle: s.topo.nangletypes, dihedral: s.topo.ndihedraltypes, improper: s.topo.nimpropertypes }[style as typeof kinds[number]];
    // As native LAMMPS does (src/delete_bonds.cpp at b332f28): a type label is its type (utils::expand_type), any
    // other word a range over 0..N (utils::bounds, which refuses 0 and so "*" too). Measured with native LAMMPS:
    // delete_bonds all bond 0 gives Invalid range string: 0.
    if (sys.labels.defined && !isNumericTypeWord(tok)) lo = hi = inumeric(typeWord(sys.labels, style as LabelKind, tok));
    else [lo, hi] = bounds(tok, 0, ntypes);
  }
  const flags = { any: false, undo: false, remove: false, special: false };
  for (const w of a.slice(typed ? 3 : 2)) {
    if (!(w in flags)) throw new StyleError('Illegal delete_bonds command');
    flags[w as keyof typeof flags] = true;
  }
  if (style === 'stats') return;
  const inGroup = new Set<number>();
  const typeOfId = new Map<number, number>();
  for (let i = 0; i < s.n; i++) {
    if (s.mask[i] & bit) inGroup.add(s.id[i]);
    if (style === 'atom') typeOfId.set(s.id[i], s.type[i]);
  }
  const lists = { bond: s.topo.bonds, angle: s.topo.angles, dihedral: s.topo.dihedrals, improper: s.topo.impropers };
  const targets = style === 'multi' || style === 'atom' ? kinds : [style as typeof kinds[number]];
  for (const kind of targets) {
    const l = lists[kind];
    for (let e = 0; e < l.n; e++) {
      const t = l.type[e];
      if (flags.undo ? t >= 0 : t < 0) continue;
      let all = true, some = false, ofType = style !== 'atom';
      for (let w = 0; w < l.width; w++) {
        const id = l.atoms[e * l.width + w];
        if (inGroup.has(id)) some = true; else all = false;
        if (style === 'atom') { const at = typeOfId.get(id) ?? 0; if (at >= lo && at <= hi) ofType = true; }
      }
      if (!(flags.any ? some : all) || !ofType) continue;
      if (style !== 'multi' && style !== 'atom' && (Math.abs(t) < lo || Math.abs(t) > hi)) continue;
      l.type[e] = -t;
    }
  }
  if (flags.remove) {
    // native (src/delete_bonds.cpp at b332f28): the removed entries leave the lists; the special lists are rebuilt only
    // with the special keyword ("re-compute special list if requested"), so without it they keep the old 1-2/1-3/1-4
    // entries until a later topology change rebuilds them (kept by sys.atomsChanged(true) below)
    for (const kind of kinds) {
      const l = lists[kind];
      let k = 0;
      for (let e = 0; e < l.n; e++) {
        const t = l.type[e];
        if (t < 0 || (t === 0 && kind === 'bond')) continue;
        l.type[k] = t;
        for (let w = 0; w < l.width; w++) l.atoms[k * l.width + w] = l.atoms[e * l.width + w];
        k++;
      }
      l.n = k;
    }
  }
  sys.atomsChanged(!flags.special);
};

export const SETUP_COMMANDS: Record<string, Handler> = {
  units, dimension, boundary, atom_style: atomStyle, atom_modify: atomModify, newton,
  processors, comm_style: commStyle, package: parallelOnly('package'),
  suffix: parallelOnly('suffix'), balance: initializing(balanceCommand),
  comm_modify: commModify, lattice, region, create_box: createBox, create_atoms: createAtoms, mass, molecule,
  read_data: readDataCmd, write_data: writeDataCmd, timestep, reset_timestep: resetTimestep,
  group, set: initializing(set, (a) => a.slice(2).some((w) => w === 'bond' || w === 'angle' || w === 'dihedral' || w === 'improper')),
  velocity, delete_atoms: initializing(deleteAtomsCmd, (a) => a[0] === 'overlap'), displace_atoms: displaceAtoms, replicate,
  change_box: changeBox, create_bonds: initializing(createBonds), delete_bonds: initializing(deleteBonds), labelmap: labelmapCmd,
};

void ALL_GROUP_BIT;

