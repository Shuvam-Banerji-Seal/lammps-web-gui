/**
 * Notebook display of an engine error. Messages for an unknown command, style or keyword end with the
 * whole supported list (often 100+ names), e.g.
 *   line 1: 'pair_coef' is not supported by the in-browser engine (it is not LAMMPS). Supported commands: ...
 *   line 3: fix style 'bogus/style' is not supported by the browser engine; supported: adapt, ...
 *   line 2: velocity: unknown keyword 'bais' (supported: dist, sum, mom, ...)
 * The notebook shows the head, a "did you mean" for the closest supported name, and the list folded away.
 */
export interface EngineErrorView {
  head: string;
  suggestion: string | null;
  supported: string[];
}

/** Optimal-string-alignment distance (Levenshtein plus adjacent transpositions). */
export const editDistance = (a: string, b: string): number => {
  const m = a.length, n = b.length;
  const d: number[][] = Array.from({ length: m + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
    }
  }
  return d[m][n];
};

/** The closest name within max(1, floor(length / 3)) edits, or null. */
export const closestName = (name: string, candidates: readonly string[]): string | null => {
  const limit = Math.max(1, Math.floor(name.length / 3));
  let best: string | null = null, bestD = Infinity;
  for (const c of candidates) {
    const dist = editDistance(name.toLowerCase(), c.toLowerCase());
    if (dist < bestD) { best = c; bestD = dist; }
  }
  return bestD <= limit ? best : null;
};

const LIST = /[;(.]\s*(?:[Ss]upported(?: commands)?): ([^)]*)\)?\s*$/;

export const explainEngineError = (message: string): EngineErrorView => {
  const m = LIST.exec(message);
  if (!m) return { head: message, suggestion: null, supported: [] };
  const supported = m[1].split(',').map((s) => s.trim()).filter((s) => s !== '' && s !== 'none' && s !== 'none yet');
  const head = message.slice(0, m.index + (m[0].startsWith('(') ? 0 : 1)).trimEnd();
  const name = /'([^']+)'/.exec(head)?.[1] ?? null;
  return { head, suggestion: name ? closestName(name, supported) : null, supported };
};
