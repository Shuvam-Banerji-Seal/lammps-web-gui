/**
 * The LAMMPS potential library: the potential files of third_party/lammps/potentials, shipped as static
 * assets under dist/potentials/ (vite.config.ts) and described by index.json (scripts/potential-index.mjs).
 * Nothing is bundled into JS: the index and each file are fetched on demand and cached for the page's lifetime.
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
