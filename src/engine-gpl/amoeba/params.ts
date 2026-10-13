/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/amoeba_file.cpp (read_prmfile: the continuation-line joining and the "####" section handling;
 * file_multipole, file_atomtype, file_vdwl, file_dippolar, the keyword handling of file_ffield and read_keyfile) and
 * src/AMOEBA/pair_amoeba.cpp mix() at commit b332f28. AMOEBA package authors: the LAMMPS developers (Sandia
 * Corporation, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Reads the subset of the Tinker PRM and key files the AMOEBA terms need: the buffered 14-7 term (stage 2a), the
 * permanent atomic multipoles (stage 2b, multipole frames with their axis types), the mpole scale factors, the dipole
 * polarizabilities with their polar groups and the polarization scale factors (induced dipoles, polar.ts). Other PRM
 * keywords are skipped.
 */
import { StyleError } from '../../engine/force/types';

export const RADIUS_RULES = ['arithmetic', 'geometric', 'cubic-mean'] as const;
export const EPSILON_RULES = ['arithmetic', 'geometric', 'harmonic', 'hhg', 'w-h'] as const;

/** Local frame of a multipole site (amoeba_file.cpp file_multipole: mpaxis values). */
export type AxisKind = 'NOFRAME' | 'ZONLY' | 'ZTHENX' | 'BISECTOR' | 'ZBISECT' | 'THREEFOLD';

/** One multipole frame of an atom type. Axis atom types are stored as positive numbers (0 = unused). */
export interface MultipoleFrame {
  /** z-axis, x-axis and y-axis atom types (positive; 0 when the axis is not used). */
  z: number;
  x: number;
  y: number;
  axis: AxisKind;
  /** The 13 values of file_multipole after symmetrizing the quadrupole, in e, e-Angstrom and e-Angstrom^2 (traceless). */
  pole: Float64Array;
}

export interface AmoebaPrm {
  radiusRule: (typeof RADIUS_RULES)[number];
  epsilonRule: (typeof EPSILON_RULES)[number];
  /** radiustype: 'r-min' (the vdW radius is the minimum) or 'sigma'. */
  radiusType: 'r-min' | 'sigma';
  /** radiussize: 'diameter' or 'radius'. */
  radiusSize: 'diameter' | 'radius';
  /** atom type (1-based) -> class. */
  typeToClass: Map<number, number>;
  /** class -> [radius (as written), epsilon, reduction factor kred]. */
  vdw: Map<number, { sigma: number; eps: number; kred: number }>;
  /** vdw-1n-scale factors: index 1 = 1-2, 2 = 1-3, 3 = 1-4, 4 = 1-5 (other entries 1.0). */
  specialHal: number[];
  /** mpole-1n-scale factors, same indexing as specialHal (set by the mpole-12-scale .. mpole-15-scale keywords). */
  specialMpole: number[];
  /** Halgren buffer constants (key: delta-halgren, gamma-halgren). */
  dhal: number;
  ghal: number;
  /** dielectric keyword of the PRM file (am_dielectric). */
  dielectric: number;
  /** atom type (1-based) -> multipole frames in file order. */
  multipoles: Map<number, MultipoleFrame[]>;
  /** A non-zero multipole value was found (the permanent multipole term then runs). */
  hasMultipoles: boolean;
  /** Non-zero polarizability found in the PRM file (the induced-dipole terms run). */
  hasPolarity: boolean;
  /** vdwpr pair overrides are not ported; their presence is refused. */
  hasVdwPair: boolean;
  /** Dipole polarizability per atom type (file_dippolar): alpha, Thole width and the polarization group partner types. */
  polar: Map<number, { alpha: number; thole: number; group: number[] }>;
  /** polarization keyword of the PRM file: 'mutual', 'direct' or 'opt...' (only mutual is ported). */
  poltyp: string;
  /** Polarization scale factors (index 1..4 = 1-2..1-5 relations; index 0 unused). */
  polarScale: { p: number[]; pi: number[]; w: number[]; direct11: number; mutual11: number };
}

export interface AmoebaKey {
  /** Cutoff and taper of the buffered 14-7 term (key: cutoff, taper, vdw-cutoff). */
  vdwcut: number;
  vdwtaper: number;
  /** Cutoff of the permanent multipole term (amoeba_file.cpp: "cutoff" sets mpolecut, "mpole-cutoff" overrides). */
  mpolecut: number;
  /** Cutoff of the induced-dipole terms (amoeba_file.cpp "cutoff" and "ewald-cutoff"; the default is 7). */
  ewaldcut: number;
  /** Cutoff of the induced-dipole preconditioner (usolve-cutoff, default 4.5). */
  usolvcut: number;
  /** Diagonal preconditioner factor (usolve-diag, default 2). */
  udiag: number;
  /** Convergence criterion of the mutual induced dipoles (polar-eps, default 1e-6) and iteration limit (polar-iter). */
  poleps: number;
  politer: number;
  /** PCG options: preconditioner, initial guess and the peek factor (pcg-precond/noprecond, pcg-guess/noguess, pcg-peek). */
  pcgprec: boolean;
  pcgguess: boolean;
  pcgpeek: number;
}

/** Words of a PRM/key line, with quoted strings kept as one word (without the quotes). */
const words = (line: string): string[] => {
  const out: string[] = [];
  const re = /"([^"]*)"|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line))) out.push(m[1] ?? m[2]);
  return out;
};

const NUMBER_RE = /^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$/;
const isNumberWord = (w: string): boolean => NUMBER_RE.test(w);

/**
 * Logical lines of a PRM file (read_prmfile): "!!" comments are cut, a line whose first word is a number continues the
 * line before it, and a blank line, a "#" banner or a commented-out keyword ("#bond ...") ends a logical line. The
 * continuation lines of a commented-out keyword are dropped with it.
 */
export const logicalLines = (text: string): string[] => {
  const out: string[] = [];
  let cur: string | null = null;
  for (const raw of text.split(/\r?\n/)) {
    let t = raw.trim();
    const bang = t.indexOf('!!');
    if (bang >= 0) t = t.slice(0, bang).trim();
    const first = t.split(/\s+/)[0] ?? '';
    if (t !== '' && cur !== null && isNumberWord(first)) {
      cur += ' ' + t;
      continue;
    }
    if (cur !== null) out.push(cur);
    cur = null;
    if (t === '') continue;
    cur = t;
  }
  if (cur !== null) out.push(cur);
  // a "#" line is a banner or a commented-out keyword: it is dropped (with its continuation lines, joined above)
  return out.filter((l) => !l.startsWith('#'));
};

const num = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || !Number.isFinite(v)) throw new StyleError(`pair amoeba: expected a number for ${what}, got '${w ?? ''}'`);
  return v;
};

const BOHR = 0.52917721067; // Bohr in Angstroms (amoeba_file.cpp)

/** file_multipole: a multipole line is "multipole type [z [x [y]]] c dx dy dz qxx qyx qyy qzx qzy qzz". */
const parseMultipole = (w: string[], line: string): { type: number; frame: MultipoleFrame; nonzero: boolean } => {
  if (w.length < 12 || w.length > 15) throw new StyleError(`pair amoeba: PRM atomic multipole line has incorrect length (${w.length}): ${line}`);
  const type = num(w[1], 'multipole type');
  let zp = 0, xp = 0, yp = 0, extra: number;
  if (w.length === 12) extra = 2;
  else if (w.length === 13) { zp = num(w[2], 'multipole z'); extra = 3; }
  else if (w.length === 14) { zp = num(w[2], 'multipole z'); xp = num(w[3], 'multipole x'); extra = 4; }
  else { zp = num(w[2], 'multipole z'); xp = num(w[3], 'multipole x'); yp = num(w[4], 'multipole y'); extra = 5; }
  const f = new Float64Array(13);
  for (let i = 0; i < 10; i++) f[i] = num(w[extra + i], 'multipole value');
  // symmetrize the quadrupole: xx yx yy zx zy zz --> row-major 3x3 at 4..12 (amoeba_file.cpp file_multipole)
  const xx = f[4], yx = f[5], yy = f[6], zx = f[7], zy = f[8], zz = f[9];
  f[4] = xx; f[8] = yy; f[12] = zz;
  f[5] = f[7] = yx;
  f[6] = f[10] = zx;
  f[9] = f[11] = zy;
  for (let i = 1; i < 4; i++) f[i] *= BOHR;
  for (let i = 4; i < 13; i++) f[i] *= (BOHR * BOHR) / 3.0;
  // the axis kind uses the signed values (file_multipole), then the stored axis types are positive
  let axis: AxisKind = 'ZTHENX';
  if (zp === 0) axis = 'NOFRAME';
  if (zp !== 0 && xp === 0) axis = 'ZONLY';
  if (zp < 0 || xp < 0) axis = 'BISECTOR';
  if (xp < 0 && yp < 0) axis = 'ZBISECT';
  const xyzmax = Math.max(zp, xp, yp);
  if (xyzmax < 0) axis = 'THREEFOLD';
  const frame: MultipoleFrame = { z: Math.abs(zp), x: Math.abs(xp), y: Math.abs(yp), axis, pole: f };
  const nonzero = f.some((v) => v !== 0);
  return { type, frame, nonzero };
};

export const parseAmoebaPrm = (text: string): AmoebaPrm => {
  const prm: AmoebaPrm = {
    radiusRule: 'arithmetic', epsilonRule: 'geometric', radiusType: 'r-min', radiusSize: 'radius',
    typeToClass: new Map(), vdw: new Map(), specialHal: [1, 1, 1, 1, 1], specialMpole: [1, 1, 1, 1, 1],
    dhal: 0.07, ghal: 0.12, dielectric: 1.0, multipoles: new Map(),
    hasMultipoles: false, hasPolarity: false, hasVdwPair: false,
    polar: new Map(), poltyp: 'mutual',
    polarScale: { p: [1, 1, 1, 1, 1], pi: [1, 1, 1, 1, 1], w: [1, 1, 1, 1, 1], direct11: 0, mutual11: 0 },
  };
  let sawVdwType = false;
  for (const line of logicalLines(text)) {
    const w = words(line);
    if (!w.length) continue;
    const key = w[0].toLowerCase();
    switch (key) {
      case 'vdwtype':
        if (w[1]?.toLowerCase() !== 'buffered-14-7') throw new StyleError(`pair amoeba: vdwtype ${w[1]} is not ported (only buffered-14-7)`);
        sawVdwType = true;
        break;
      case 'radiusrule': {
        const r = w[1]?.toLowerCase();
        if (r === 'arithmetic' || r === 'geometric' || r === 'cubic-mean') prm.radiusRule = r;
        else throw new StyleError(`pair amoeba: unrecognized radiusrule ${w[1]}`);
        break;
      }
      case 'radiustype': {
        const r = w[1]?.toLowerCase();
        if (r === 'r-min' || r === 'sigma') prm.radiusType = r;
        else throw new StyleError(`pair amoeba: unrecognized radiustype ${w[1]}`);
        break;
      }
      case 'radiussize': {
        const r = w[1]?.toLowerCase();
        if (r === 'diameter' || r === 'radius') prm.radiusSize = r;
        else throw new StyleError(`pair amoeba: unrecognized radiussize ${w[1]}`);
        break;
      }
      case 'epsilonrule': {
        const r = w[1]?.toLowerCase();
        if (r === 'arithmetic' || r === 'geometric' || r === 'harmonic' || r === 'hhg' || r === 'w-h') prm.epsilonRule = r;
        else throw new StyleError(`pair amoeba: unrecognized epsilonrule ${w[1]}`);
        break;
      }
      case 'dielectric': prm.dielectric = num(w[1], key); break;
      case 'vdw-12-scale': prm.specialHal[1] = num(w[1], key); break;
      case 'vdw-13-scale': prm.specialHal[2] = num(w[1], key); break;
      case 'vdw-14-scale': prm.specialHal[3] = num(w[1], key); break;
      case 'vdw-15-scale': prm.specialHal[4] = num(w[1], key); break;
      case 'mpole-12-scale': prm.specialMpole[1] = num(w[1], key); break;
      case 'mpole-13-scale': prm.specialMpole[2] = num(w[1], key); break;
      case 'mpole-14-scale': prm.specialMpole[3] = num(w[1], key); break;
      case 'mpole-15-scale': prm.specialMpole[4] = num(w[1], key); break;
      case 'atom': {
        // atom type class symbol "description" atomic-number mass valence (amoeba_file.cpp file_atomtype)
        if (w.length < 8) throw new StyleError(`pair amoeba: PRM atom line has too few values: ${line}`);
        const itype = num(w[1], 'atom type'), iclass = num(w[2], 'atom class');
        prm.typeToClass.set(itype, iclass);
        break;
      }
      case 'vdw': {
        // vdw class radius epsilon [reduction] (amoeba_file.cpp file_vdwl)
        if (w.length !== 4 && w.length !== 5) throw new StyleError(`pair amoeba: PRM vdw line has incorrect length: ${line}`);
        const iclass = num(w[1], 'vdw class');
        prm.vdw.set(iclass, { sigma: num(w[2], 'vdw radius'), eps: num(w[3], 'vdw epsilon'), kred: w.length === 5 ? num(w[4], 'vdw reduction') : 0 });
        break;
      }
      case 'vdwpr':
        prm.hasVdwPair = true;
        break;
      case 'multipole': {
        const m = parseMultipole(w, line);
        const list = prm.multipoles.get(m.type) ?? [];
        list.push(m.frame);
        prm.multipoles.set(m.type, list);
        if (m.nonzero) prm.hasMultipoles = true;
        break;
      }
      case 'polarize': {
        // file_dippolar: polarize type alpha thole [partner types...] (AMOEBA has the Thole width)
        if (w.length < 4) throw new StyleError(`pair amoeba: PRM dipole polarizability line has too few values: ${line}`);
        const itype = num(w[1], 'polarize type');
        const alpha = num(w[2], 'polarizability');
        if (alpha !== 0) prm.hasPolarity = true;
        const group = w.slice(4).map((g) => num(g, 'polarization group type'));
        prm.polar.set(itype, { alpha, thole: num(w[3], 'Thole width'), group });
        break;
      }
      case 'polarization': {
        const v = w[1]?.toLowerCase() ?? '';
        if (v !== 'mutual' && v !== 'direct' && !v.startsWith('opt')) throw new StyleError(`pair amoeba: unrecognized polarization ${w[1]}`);
        prm.poltyp = v;
        break;
      }
      case 'polar-12-scale': prm.polarScale.p[1] = num(w[1], key); break;
      case 'polar-13-scale': prm.polarScale.p[2] = num(w[1], key); break;
      case 'polar-14-scale': prm.polarScale.p[3] = num(w[1], key); break;
      case 'polar-15-scale': prm.polarScale.p[4] = num(w[1], key); break;
      case 'polar-12-intra': prm.polarScale.pi[1] = num(w[1], key); break;
      case 'polar-13-intra': prm.polarScale.pi[2] = num(w[1], key); break;
      case 'polar-14-intra': prm.polarScale.pi[3] = num(w[1], key); break;
      case 'polar-15-intra': prm.polarScale.pi[4] = num(w[1], key); break;
      case 'induce-12-scale': prm.polarScale.w[1] = num(w[1], key); break;
      case 'induce-13-scale': prm.polarScale.w[2] = num(w[1], key); break;
      case 'induce-14-scale': prm.polarScale.w[3] = num(w[1], key); break;
      case 'induce-15-scale': prm.polarScale.w[4] = num(w[1], key); break;
      case 'direct-11-scale': prm.polarScale.direct11 = num(w[1], key); break;
      case 'mutual-11-scale': prm.polarScale.mutual11 = num(w[1], key); break;
      case 'direct-12-scale': case 'direct-13-scale': case 'direct-14-scale':
      case 'mutual-12-scale': case 'mutual-13-scale': case 'mutual-14-scale':
        if (num(w[1], key) !== 1.0) throw new StyleError(`pair amoeba: ${key} must be 1.0 (amoeba_file.cpp)`);
        break;
      default:
        // other PRM keywords belong to terms not ported yet (bonds, angles, torsions, polarization groups)
        break;
    }
  }
  if (!sawVdwType) throw new StyleError('pair amoeba: the PRM file has no vdwtype line');
  return prm;
};

/** Key file keywords the AMOEBA terms read (amoeba_file.cpp read_keyfile). */
export const parseAmoebaKey = (text: string, prm: AmoebaPrm): AmoebaKey => {
  let vdwcut = 9.0;
  let vdwtaper = 0.9 * vdwcut;
  let mpolecut = 9.0;
  let mpoleCutSet = false;
  let ewaldcut = 7.0;
  let usolvcut = 4.5;
  let udiag = 2.0;
  let poleps = 1.0e-6;
  let politer = 100;
  let pcgprec = true;
  let pcgguess = true;
  let pcgpeek = 1.0;
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim();
    if (t === '' || t.startsWith('!') || t.startsWith('#')) continue;
    const w = words(t);
    if (!w.length) continue;
    const key = w[0].toLowerCase();
    switch (key) {
      case 'cutoff': {
        // amoeba_file.cpp: cutoff sets vdwcut, repcut, dispcut, mpolecut, ctrncut and the Ewald cutoffs
        const cut = num(w[1], 'cutoff');
        vdwcut = cut;
        vdwtaper = 0.9 * vdwcut;
        if (!mpoleCutSet) mpolecut = cut;
        ewaldcut = cut;
        break;
      }
      case 'ewald-cutoff': ewaldcut = num(w[1], key); break;
      case 'usolve-cutoff': usolvcut = num(w[1], key); break;
      case 'usolve-diag': udiag = num(w[1], key); break;
      case 'polar-eps': poleps = num(w[1], key); break;
      case 'polar-iter': politer = num(w[1], key); break;
      case 'pcg-precond': pcgprec = true; break;
      case 'pcg-noprecond': pcgprec = false; break;
      case 'pcg-guess': pcgguess = true; break;
      case 'pcg-noguess': pcgguess = false; break;
      case 'pcg-peek': pcgpeek = num(w[1], key); break;
      case 'polar-predict':
        throw new StyleError('pair amoeba: polar-predict (induced dipole prediction) is not ported yet');
      case 'polarization': {
        const v = w[1]?.toLowerCase() ?? '';
        if (v !== 'mutual') throw new StyleError(`pair amoeba: polarization ${w[1]} in the key file is not ported yet (only mutual)`);
        break;
      }
      case 'taper': {
        // taper >= 1 is an absolute distance; below 1 it is a fraction of the cutoff (amoeba_file.cpp "taper")
        const tp = num(w[1], 'taper');
        vdwtaper = tp >= 1.0 ? tp : -tp * vdwcut;
        break;
      }
      case 'vdw-cutoff':
        vdwcut = num(w[1], 'vdw-cutoff');
        vdwtaper = 0.9 * vdwcut;
        break;
      case 'mpole-cutoff':
        mpolecut = num(w[1], 'mpole-cutoff');
        mpoleCutSet = true;
        break;
      case 'ewald':
        throw new StyleError('pair amoeba: the Ewald summation (key "ewald") is not ported yet; the multipole term is the direct sum only');
      case 'delta-halgren': prm.dhal = num(w[1], key); break;
      case 'gamma-halgren': prm.ghal = num(w[1], key); break;
      default:
        break;
    }
  }
  return { vdwcut, vdwtaper, mpolecut, ewaldcut, usolvcut, udiag, poleps, politer, pcgprec, pcgguess, pcgpeek };
};
