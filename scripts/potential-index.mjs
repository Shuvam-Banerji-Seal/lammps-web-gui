#!/usr/bin/env node
/**
 * Index of the LAMMPS potential files (third_party/lammps/potentials), written at build time to
 * dist/potentials/index.json by the copy plugin in vite.config.ts, and read by the notebook's potential
 * library (src/components/workbench/notebook/potentialLibrary.ts).
 *
 * Entry: { name, bytes, kind, elements, styles, units? }
 *   kind     the potential family, inferred from the file name and its header ('unknown' when unsure)
 *   elements the element symbols the file states (setfl/funcfl header line, three-body rows); [] otherwise
 *   styles   the pair styles that read the file (LAMMPS pair_style names; see the style comments below)
 *   units    the UNITS keyword of the first line, when present
 *
 * Usage: node scripts/potential-index.mjs [potentialsDir] [outFile]
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const LAMMPS_COMMIT = 'b332f28';
export const LAMMPS_LICENSE = 'GPL-2.0';

const ELEMENTS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr ' +
  'Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu ' +
  'Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr').split(' ');
const ELEMENT_SET = new Set(ELEMENTS);
/** Atomic number -> symbol (the funcfl eam header states Z). */
const symbolOfZ = (z) => (Number.isInteger(z) && z >= 1 && z <= ELEMENTS.length ? ELEMENTS[z - 1] : null);

const isNumber = (t) => /^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$/.test(t);

/** Element symbols from leading symbol tokens of data lines (tersoff, sw, edip, vashishta, snap rows). */
const rowElements = (text) => {
  const found = new Set();
  for (const raw of text.split(/\r?\n/)) {
    if (raw.trimStart().startsWith('#')) continue;
    const w = raw.replace(/#.*$/, '').trim().split(/\s+/).filter(Boolean);
    let k = 0;
    while (k < w.length && k < 3 && ELEMENT_SET.has(w[k])) k++;
    if (k === 0 || k === w.length) continue;
    if (!w.slice(k).every((t) => isNumber(t))) continue;
    for (let i = 0; i < k; i++) found.add(w[i]);
  }
  return [...found];
};

/** The element symbols a file states: setfl (eam/alloy, eam/fs, ...) line 4, funcfl line 4 (Z), or data rows. */
export const elementsOf = (kind, text) => {
  const lines = text.split(/\r?\n/);
  if (kind === 'eam/alloy' || kind === 'eam/fs' || kind === 'eam/he' || kind === 'eam/cd') {
    const w = (lines[3] ?? '').trim().split(/\s+/);
    const n = Number(w[0]);
    if (Number.isInteger(n) && n > 0 && w.length >= n + 1) return w.slice(1, 1 + n).filter((s) => ELEMENT_SET.has(s));
    return [];
  }
  if (kind === 'eam') {
    // funcfl: line 1 is the comment with the DATE keyword, line 2 is "Z mass alat lattice"
    const sym = symbolOfZ(Number((lines[1] ?? '').trim().split(/\s+/)[0]));
    return sym ? [sym] : [];
  }
  if (['tersoff', 'sw', 'edip', 'vashishta', 'snap'].includes(kind)) return rowElements(text);
  return [];
};

/** The UNITS keyword of the first line (the potential header format, potentials/README). */
const unitsOf = (text) => {
  const m = /UNITS:\s*(\S+)/.exec(text.slice(0, 400));
  return m ? m[1] : undefined;
};

/** Pair styles per kind for the bulk of the files. */
const KIND_STYLES = {
  'eam': ['eam'], 'eam/alloy': ['eam/alloy'], 'eam/fs': ['eam/fs'], 'eam/he': ['eam/he'], 'eam/cd': ['eam/cd'],
  'tersoff': ['tersoff'], 'sw': ['sw'], 'meam': ['meam'], 'meam/spline': ['meam/spline'],
  'meam/sw/spline': ['meam/sw/spline'], 'meam/ms': ['meam/ms'], 'snap': ['snap'], 'mliap': ['mliap'],
  'pace': ['pace'], 'pod': ['pod'], 'uf3': ['uf3'], 'agni': ['agni'], 'rann': ['rann'], 'adp': ['adp'],
  'bop': ['bop'], 'edip': ['edip', 'edip/multi'], 'vashishta': ['vashishta', 'vashishta/table'],
  'streitz': ['coul/streitz'], 'nb3b': ['nb3b/harmonic', 'nb3b/screened'], 'airebo': ['airebo'],
  'rebo': ['rebo'], 'drip': ['drip'], 'extep': ['extep'], 'lcbop': ['lcbop'], 'reaxff': ['reaxff'],
  'comb': ['comb'], 'comb3': ['comb3'], 'eim': ['eim'], 'smtbq': ['smtbq'], 'mgpt': ['mgpt'], 'ldd': ['ldd'],
  'polymorphic': ['polymorphic'], 'table': ['table'], 'gw': ['gw', 'gw/zbl'], 'quip': ['quip'],
  'rebomos': ['rebomos'], 'ctip': ['coul/ctip'], 'aip': ['aip/water/2dm', 'aip/water/2dm/opt'], 'cmap': ['cmap'], 'kc': ['kolmogorov/crespi/full', 'kolmogorov/crespi/z'], 'lebedeva': ['lebedeva/z'],
  'ilp': ['ilp/graphene/hbn'], 'cgdna': [], 'doc': [], 'unknown': [],
};

/** The pair styles a file is read by, from its kind and name (null when the kind does not decide them). */
const stylesFor = (name, kind) => {
  if (name.endsWith('.tersoff.modc')) return ['tersoff/mod/c'];
  if (name.endsWith('.tersoff.mod')) return ['tersoff/mod'];
  if (name.endsWith('.tersoff.zbl')) return ['tersoff/zbl'];
  if (name.endsWith('.sw.mod')) return ['sw/mod'];
  if (name.endsWith('airebo-bc')) return ['airebo/bc'];
  if (name.endsWith('airebo-m')) return ['airebo/morse'];
  if (name.endsWith('.ctip')) return ['coul/ctip'];
  if (kind === 'ilp') return /tmd|mos/i.test(name) ? ['ilp/tmd'] : ['ilp/graphene/hbn'];
  if (kind === 'cgdna') {
    const family = /^(oxdna\d?|oxrna2)_/.exec(name)?.[1] ?? 'oxdna';
    return ['coaxstk', 'dh', 'excv', 'hbond', 'stk', 'xstk'].map((s) => `${family}/${s}`);
  }
  return KIND_STYLES[kind] ?? [];
};

/**
 * The kind of a potential file, from its name first and its header second. Name rules are checked in order;
 * the first match decides. Returns 'unknown' when no rule applies.
 */
export const kindOf = (name, text = '') => {
  const n = name;
  if (n.startsWith('.') || n === 'README' || /^README[.\s]/i.test(n) || /\.(md|txt)$/i.test(n)) return 'doc';
  if (/\.eam\.alloy$/.test(n)) return 'eam/alloy';
  if (/\.eam\.fs$/.test(n)) return 'eam/fs';
  if (/\.eam\.he$/.test(n)) return 'eam/he';
  if (/\.cdeam$/.test(n)) return 'eam/cd';
  if (/\.eam$/.test(n)) return 'eam';
  if (/\.tersoff(\.mod|\.modc|\.zbl)?$/.test(n) || /tersoff$/.test(n)) return 'tersoff';
  if (/\.sw(\.mod)?$/.test(n)) return 'sw';
  if (/\.meam\.spline$/.test(n)) return 'meam/spline';
  if (/\.meam\.sw\.spline$/.test(n)) return 'meam/sw/spline';
  if (/\.msmeam$/.test(n)) return 'meam/ms';
  if (/\.meam$/.test(n)) return 'meam';
  if (/\.(snap|snapcoeff|snapparam)$/.test(n)) return 'snap';
  if (/\.mliap(\.[a-z]+)?$|\.nn\.mliap(\.model)?$|\.(model|descriptor)$/.test(n) && /mliap|\.model$|\.descriptor$/.test(n)) return 'mliap';
  if (/\.ace$/.test(n)) return 'pace';
  if (/\.pod$/.test(n)) return 'pod';
  if (/\.uf3$/.test(n)) return 'uf3';
  if (/\.agni$/.test(n)) return 'agni';
  if (/\.rann$/.test(n)) return 'rann';
  if (/\.adp$/.test(n)) return 'adp';
  if (/\.bop\.table$/.test(n)) return 'bop';
  if (/\.edip$/.test(n)) return 'edip';
  if (/\.vashishta$/.test(n)) return 'vashishta';
  if (/\.streitz(\+tersoff\.mod)?$/.test(n)) return 'streitz';
  if (/\.nb3b\./.test(n)) return 'nb3b';
  if (/^CH\.airebo/.test(n)) return 'airebo';
  if (/^CH\.rebo$/.test(n)) return 'rebo';
  if (/\.drip$/.test(n)) return 'drip';
  if (/\.extep$/.test(n)) return 'extep';
  if (/\.lcbop$/.test(n)) return 'lcbop';
  if (/^ffield\.reax|^ffield\.ci-reax|^acks2_ff|ci-reaxFF.*\.dat$/.test(n)) return 'reaxff';
  if (/^ffield\.comb$/.test(n)) return 'comb';
  if (/^(ffield|lib)\.comb3$/.test(n)) return 'comb3';
  if (/^ffield\.eim$/.test(n)) return 'eim';
  if (/^ffield\.smtbq/.test(n)) return 'smtbq';
  if (/\.mgpt\./.test(n)) return 'mgpt';
  if (/\.ldd$/.test(n)) return 'ldd';
  if (/\.poly$/.test(n)) return 'polymorphic';
  if (/\.table$/.test(n)) return 'table';
  if (/\.gw(\.zbl)?$/.test(n)) return 'gw';
  if (/\.quip$/.test(n)) return 'quip';
  if (/\.rebomos$/.test(n)) return 'rebomos';
  if (/\.KC(-full)?$/.test(n)) return 'kc';
  if (/^CC\.Lebedeva$/.test(n)) return 'lebedeva';
  if (/\.ILP$/.test(n)) return 'ilp';
  if (/\.cgdna$/.test(n)) return 'cgdna';
  if (/^(W_|Mo_|Ni_|Cu_|Si_|Li_|Ge_|Ta06A|InP_|WBe_).*(snap|snapcoeff)$/.test(n)) return 'snap';
  if (/\.ctip$/.test(n)) return 'ctip';
  if (/\.aip\.water\.2dm$/.test(n)) return 'aip';
  if (/\.cmap$/.test(n)) return 'cmap';
  return 'unknown';
};

/** One index entry for a file of the potentials directory. */
export const indexEntry = (name, text, bytes) => {
  const kind = kindOf(name, text);
  const styles = stylesFor(name, kind);
  const entry = { name, bytes, kind, elements: elementsOf(kind, text), styles };
  const units = unitsOf(text);
  if (units) entry.units = units;
  return entry;
};

/** The whole index for a directory: every regular file that is not a dotfile, sorted by name. */
export const buildIndex = (dir) => {
  const files = readdirSync(dir, { withFileTypes: true }).filter((d) => !d.name.startsWith('.') && d.isFile()).map((d) => d.name).sort();
  return {
    format: 1,
    source: {
      upstream: 'https://github.com/lammps/lammps',
      commit: LAMMPS_COMMIT,
      license: LAMMPS_LICENSE,
      note: 'Unmodified LAMMPS potential files; each file header gives its own author and citation.',
    },
    files: files.map((name) => {
      const path = join(dir, name);
      // one read: the size comes from the same bytes as the header (no stat-then-read race)
      const buf = readFileSync(path);
      const head = buf.toString('latin1', 0, Math.min(buf.length, 1 << 20));
      return indexEntry(name, head, buf.length);
    }),
  };
};

const runAsScript = () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const dir = resolve(process.argv[2] ?? join(here, '..', 'third_party', 'lammps', 'potentials'));
  const out = resolve(process.argv[3] ?? join(dir, 'index.json'));
  const index = buildIndex(dir);
  writeFileSync(out, JSON.stringify(index, null, 1) + '\n');
  console.log(`wrote ${index.files.length} entries to ${out} (${basename(dir)})`);
};

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) runAsScript();
