/**
 * The LAMMPS potential library: the potential files of third_party/lammps/potentials, shipped as static
 * assets under dist/potentials/ (vite.config.ts) and described by index.json (scripts/potential-index.mjs).
 * Nothing is bundled into JS: the index and each file are fetched on demand and cached for the page's lifetime.
 *
 * This module is shared by the notebook (src/components/workbench/notebook/potentialLibrary.ts re-exports it)
 * and the Script Builder (src/components/workbench/builder/PotentialPicker.tsx), so it has no UI imports.
 *
 * The files are GPL-2.0 (LAMMPS commit b332f28, see potentials/LICENSE and potentials/README.md).
 */

/** One file of the library, as index.json describes it. */
export interface PotentialEntry {
  name: string;
  bytes: number;
  /** The potential family, from the name and header ('unknown' when unsure, 'doc' for README files). */
  kind: string;
  /** Element symbols the file states; [] when it does not state them. */
  elements: string[];
  /** The pair styles that read the file. */
  styles: string[];
  /** The UNITS keyword of the file's header, when it has one. */
  units?: string;
}

export interface PotentialIndex {
  format: number;
  source: { upstream: string; commit: string; license: string; note: string };
  files: PotentialEntry[];
}

/** The provenance line shown in the library browser. */
export const POTENTIAL_PROVENANCE = 'LAMMPS commit b332f28, GPL-2.0 (unmodified files; each file header gives its author and citation)';

const baseUrl = (): string => import.meta.env.BASE_URL;

/** URL of a file of the library (or of LICENSE and README.md beside them). */
export const potentialUrl = (name: string): string => `${baseUrl()}potentials/${encodeURIComponent(name)}`;

let indexPromise: Promise<PotentialIndex> | null = null;
const textPromises = new Map<string, Promise<string>>();

/** Fetches index.json once. A failed fetch is not cached, so a later call retries. */
export const loadPotentialIndex = (): Promise<PotentialIndex> => {
  if (!indexPromise) {
    indexPromise = fetch(`${baseUrl()}potentials/index.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`index.json: HTTP ${r.status}`);
        return r.json() as Promise<PotentialIndex>;
      })
      .catch((e: unknown) => {
        indexPromise = null;
        throw e;
      });
  }
  return indexPromise;
};

/** Fetches one file of the library as text, once per name. A failed fetch is not cached. */
export const loadPotentialText = (name: string): Promise<string> => {
  let p = textPromises.get(name);
  if (!p) {
    p = fetch(potentialUrl(name))
      .then((r) => {
        if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
        return r.text();
      })
      .catch((e: unknown) => {
        textPromises.delete(name);
        throw e;
      });
    textPromises.set(name, p);
  }
  return p;
};

/** The words of a script that are not in comments (a '#' starts a comment). */
export const scriptWords = (script: string): string[] => {
  const out: string[] = [];
  for (const raw of script.split('\n')) {
    const line = raw.replace(/#.*$/, '');
    for (const w of line.split(/\s+/)) {
      const t = w.replace(/^["']|["']$/g, '');
      if (t) out.push(t);
    }
  }
  return out;
};

/**
 * The library files a script names (a word equal to a file name, e.g. the file of pair_coeff or of
 * read_data/include) that are not in `haveFiles` (the names the user already provides; those always win).
 */
export const missingPotentials = (index: PotentialIndex, script: string, haveFiles: ReadonlySet<string>): PotentialEntry[] => {
  const words = new Set(scriptWords(script));
  return index.files.filter((f) => f.kind !== 'doc' && words.has(f.name) && !haveFiles.has(f.name));
};

/** Async form of missingPotentials: loads the index first. */
export const resolveMissing = async (script: string, haveFiles: ReadonlySet<string>): Promise<PotentialEntry[]> => {
  if (!/\S/.test(script)) return [];
  const index = await loadPotentialIndex();
  return missingPotentials(index, script, haveFiles);
};

/** Which of a file's styles the browser engine has (pair styles), for the library browser. */
export const engineSupport = (entry: PotentialEntry, pairStyles: ReadonlySet<string>): { supported: string[]; unsupported: string[] } => {
  const supported = entry.styles.filter((s) => pairStyles.has(s));
  return { supported, unsupported: entry.styles.filter((s) => !pairStyles.has(s)) };
};

/** Library entries matching a query on the name, an element or a style (case-insensitive; empty matches all). */
export const searchPotentials = (files: readonly PotentialEntry[], query: string): PotentialEntry[] => {
  const q = query.trim().toLowerCase();
  if (!q) return [...files];
  return files.filter((f) => f.name.toLowerCase().includes(q)
    || f.elements.some((e) => e.toLowerCase() === q)
    || f.styles.some((s) => s.toLowerCase().includes(q)));
};

/* ------------------------------------------------------------------ */
/* Script Builder helpers                                              */
/* ------------------------------------------------------------------ */

/**
 * The pair style names a pair_style argument string mentions: its words (for `pair_style hybrid ...`
 * the sub-style names are among them, interleaved with numbers). Used to filter the library by style.
 */
export const pairStyleWords = (pairStyleArgs: string): string[] =>
  pairStyleArgs.trim().split(/\s+/).filter(Boolean);

/** Library files that some of the given pair style words read; all files when `words` is empty. */
export const filterByStyles = (files: readonly PotentialEntry[], words: readonly string[]): PotentialEntry[] => {
  if (words.length === 0) return [...files];
  const want = new Set(words);
  return files.filter((f) => f.styles.some((s) => want.has(s)));
};

/** Files the picker offers as the file of a pair_coeff (the .snapparam of a SNAP pair is chosen with its .snapcoeff). */
export const isChoosable = (entry: PotentialEntry): boolean => entry.kind !== 'doc' && !entry.name.endsWith('.snapparam');

/**
 * How a pair style's pair_coeff takes its file, from docs.lammps.org:
 *  - 'funcfl': `pair_coeff * * file` (one element, no element names), pair_eam.html.
 *  - 'snap': `pair_coeff * * snapcoeff snapparam El ...`, pair_snap.html.
 *  - 'elements': `pair_coeff * * file El1 El2 ...`, element names after the file in the order of the file.
 *  - 'file': only the file name is filled; the element names are for the user to write.
 */
export type CoeffForm = 'funcfl' | 'snap' | 'elements' | 'file';

/** pair_coeff forms, each one checked against the example lines of the doc page named in its comment. */
export const coeffFormOf = (style: string): CoeffForm => {
  if (style === 'eam') return 'funcfl'; // pair_eam.html: "pair_coeff * * cuu3" (funcfl file, no element names)
  if (style === 'snap') return 'snap'; // pair_snap.html: "pair_coeff * * InP.snapcoeff InP.snapparam In In P P"
  // pair_eam.html "pair_coeff * * ../potentials/NiAlH_jea.eam.alloy Ni Al Ni Ni"; pair_tersoff.html "pair_coeff * * Si.tersoff Si";
  // pair_sw.html "pair_coeff * * GaN.sw Ga N Ga"; pair_edip.html "pair_coeff * * Si.edip Si";
  // pair_vashishta.html "pair_coeff * * SiC.vashishta Si C"; pair_airebo.html "pair_coeff * * ../potentials/CH.airebo H C";
  // pair_adp.html "pair_coeff * * Ta.adp Ta"; pair_meam_spline.html "pair_coeff * * Ti.meam.spline Ti";
  // pair_reaxff.html "pair_coeff * * ffield.reax C H O N"
  const elementStyles = new Set([
    'eam/alloy', 'eam/fs', 'eam/he', 'eam/cd', 'tersoff', 'tersoff/mod', 'tersoff/mod/c', 'sw', 'sw/mod',
    'edip', 'vashishta', 'airebo', 'airebo/morse', 'airebo/bc', 'rebo', 'adp', 'meam/spline', 'reaxff',
  ]);
  return elementStyles.has(style) ? 'elements' : 'file';
};

/** The name of the .snapparam file that goes with a .snapcoeff file (same stem), or null. */
export const snapParamName = (coeffName: string): string | null =>
  coeffName.endsWith('.snapcoeff') ? `${coeffName.slice(0, -'.snapcoeff'.length)}.snapparam` : null;

/**
 * The text that goes into a pair_coeff's coefficient field when the user picks a library file.
 * `pairWords` are the words of the pair_style line; for `pair_style hybrid ...` the sub-style that reads
 * the file is written before it (`pair_coeff * * sw Si.sw Si`, pair_sw.html).
 * The element names are the file's own (entry.elements, file order) for the user to map to atom types.
 */
export const coeffsForPick = (entry: PotentialEntry, pairWords: readonly string[], index: PotentialIndex): string => {
  const hybrid = pairWords[0]?.startsWith('hybrid') ?? false;
  const style = entry.styles.find((s) => pairWords.includes(s)) ?? entry.styles[0] ?? '';
  const sub = hybrid && style ? [style] : [];
  const elements = entry.elements.join(' ');
  switch (coeffFormOf(style)) {
    case 'funcfl':
      return [...sub, entry.name].join(' ');
    case 'snap': {
      const param = snapParamName(entry.name);
      const hasParam = param !== null && index.files.some((f) => f.name === param);
      return [...sub, entry.name, ...(hasParam ? [param] : []), elements].filter(Boolean).join(' ');
    }
    case 'elements':
      return [...sub, entry.name, elements].filter(Boolean).join(' ');
    case 'file':
      return [...sub, entry.name].join(' ');
    default:
      return entry.name;
  }
};

/** The doc sentence that states the units a family of potential files is parameterized in (quoted from the pair style page). */
export interface UnitsDoc {
  page: string;
  sentence: string;
}

const UNITS_DOC_BY_STYLE: Record<string, UnitsDoc> = {
  // pair_eam.rst: "All of these files are parameterized in terms of LAMMPS metal units."
  'eam': { page: 'pair_eam.html', sentence: 'All of these files are parameterized in terms of LAMMPS metal units.' },
  'eam/alloy': { page: 'pair_eam.html', sentence: 'All of these files are parameterized in terms of LAMMPS metal units.' },
  'eam/fs': { page: 'pair_eam.html', sentence: 'All of these files are parameterized in terms of LAMMPS metal units.' },
  'eam/he': { page: 'pair_eam.html', sentence: 'All of these files are parameterized in terms of LAMMPS metal units.' },
  'eam/cd': { page: 'pair_eam.html', sentence: 'All of these files are parameterized in terms of LAMMPS metal units.' },
  // pair_meam.rst: "All of these are parameterized in terms of LAMMPS metal units."
  'meam': { page: 'pair_meam.html', sentence: 'All of these are parameterized in terms of LAMMPS metal units.' },
  // pair_tersoff.rst: "The Tersoff potential files provided with LAMMPS (see the potentials directory) are parameterized for "metal" units."
  'tersoff': { page: 'pair_tersoff.html', sentence: 'The Tersoff potential files provided with LAMMPS (see the potentials directory) are parameterized for "metal" units.' },
  'tersoff/mod': { page: 'pair_tersoff.html', sentence: 'The Tersoff potential files provided with LAMMPS (see the potentials directory) are parameterized for "metal" units.' },
  'tersoff/mod/c': { page: 'pair_tersoff.html', sentence: 'The Tersoff potential files provided with LAMMPS (see the potentials directory) are parameterized for "metal" units.' },
  // pair_sw.rst: "The Stillinger-Weber potential files provided with LAMMPS (see the potentials directory) are parameterized for metal units."
  'sw': { page: 'pair_sw.html', sentence: 'The Stillinger-Weber potential files provided with LAMMPS (see the potentials directory) are parameterized for metal units.' },
  'sw/mod': { page: 'pair_sw.html', sentence: 'The Stillinger-Weber potential files provided with LAMMPS (see the potentials directory) are parameterized for metal units.' },
  // pair_airebo.rst: "The CH.airebo, CH.airebo-m, and CH.airebo-bc potential files provided with LAMMPS (see the potentials directory) are parameterized for metal units."
  'airebo': { page: 'pair_airebo.html', sentence: 'The CH.airebo, CH.airebo-m, and CH.airebo-bc potential files provided with LAMMPS (see the potentials directory) are parameterized for metal units.' },
  // pair_reaxff.rst: "The ReaxFF potential files provided with LAMMPS in the potentials directory are parameterized for *real* units."
  'reaxff': { page: 'pair_reaxff.html', sentence: 'The ReaxFF potential files provided with LAMMPS in the potentials directory are parameterized for real units.' },
};

/** The units doc of a pair style, or null when the doc page states none for it. */
export const unitsDocFor = (style: string): UnitsDoc | null => UNITS_DOC_BY_STYLE[style] ?? null;
