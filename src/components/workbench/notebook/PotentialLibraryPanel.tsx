import React, { useEffect, useMemo, useState } from 'react';
import { getThemeTokens, type Theme } from '../../../theme';
import {
  engineSupport, loadPotentialIndex, POTENTIAL_PROVENANCE, potentialUrl, searchPotentials,
  type PotentialEntry, type PotentialIndex,
} from './potentialLibrary';

/** Size in bytes as B, KB or MB (the panel's own format, kept here so the file does not depend on InputFilesPanel). */
const size = (n: number): string =>
  n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`;

/** Results shown at most; a search narrows the list further. */
const LIMIT = 60;

export interface PotentialLibraryPanelProps {
  theme: Theme;
  /** The names the notebook's inputs hold (an entry already there offers no button). */
  inputs: Record<string, number>;
  /** Adds one library file to the inputs. */
  onAdd: (name: string) => Promise<void>;
  /** The browser engine's pair_style names (the engine's ready message); null until the engine is ready. */
  pairStyles: readonly string[] | null;
}

/**
 * "Potential library" browser: the LAMMPS potential files shipped with the site (dist/potentials), searched by
 * name, element or style, with their size, provenance, whether the browser engine has a style that reads them,
 * and an "Add to inputs" button. Opening it loads the index (and the engine's style list) once.
 */
export const PotentialLibraryPanel: React.FC<PotentialLibraryPanelProps> = ({ theme, inputs, onAdd, pairStyles: enginePairStyles }) => {
  const ct = getThemeTokens(theme);
  const [open, setOpen] = useState(false);
  const [index, setIndex] = useState<PotentialIndex | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const pairStyles = useMemo(() => (enginePairStyles ? new Set(enginePairStyles) : null), [enginePairStyles]);
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [addError, setAddError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    loadPotentialIndex()
      .then((i) => { if (alive) setIndex(i); })
      .catch((e: unknown) => { if (alive) setLoadError(e instanceof Error ? e.message : String(e)); });
    return () => { alive = false; };
  }, [open]);

  const shown = useMemo(() => (index ? searchPotentials(index.files.filter((f) => f.kind !== 'doc'), query) : []), [index, query]);
  const btn = `inline-flex min-h-6 items-center justify-center rounded px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${ct.hoverSurface}`;

  const add = async (name: string) => {
    setBusy(name);
    setAddError(null);
    try {
      await onAdd(name);
    } catch (e) {
      setAddError(`${name} could not be added: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  };

  const row = (f: PotentialEntry) => {
    const support = pairStyles ? engineSupport(f, pairStyles) : null;
    const inInputs = f.name in inputs;
    return (
      <li key={f.name} className={`rounded border p-2 ${ct.card}`} data-potential={f.name}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="font-mono text-xs">{f.name}</span>
          <span className={`text-[11px] ${ct.muted}`}>{size(f.bytes)}</span>
        </div>
        <p className={`text-[11px] ${ct.muted}`}>
          {f.kind}{f.elements.length > 0 ? ` · ${f.elements.join(' ')}` : ''}{f.units ? ` · units ${f.units}` : ''}
        </p>
        <p className={`text-[11px] ${ct.muted}`}>
          {support === null ? 'Checking the browser engine...'
            : support.supported.length > 0 ? `Browser engine: supports pair_style ${support.supported.join(', ')}`
              : `Browser engine: no supported style reads this file${f.styles.length ? ` (LAMMPS style: ${f.styles.join(', ')})` : ''}`}
        </p>
        <div className="mt-1 flex items-center justify-end">
          {inInputs ? (
            <span className={`text-[11px] ${ct.muted}`}>In the inputs</span>
          ) : (
            <button className={`${btn} ${ct.button}`} disabled={busy !== null} onClick={() => void add(f.name)}>
              {busy === f.name ? 'Adding...' : 'Add to inputs'}
            </button>
          )}
        </div>
      </li>
    );
  };

  return (
    <details className="mt-3" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className={`cursor-pointer text-[11px] font-semibold ${ct.muted}`}>Potential library</summary>
      <div className="mt-2 space-y-2">
        <p className={`text-[11px] ${ct.muted}`}>
          {POTENTIAL_PROVENANCE}. A script that names one of these files (for example pair_coeff) gets it added
          automatically when it runs; a file you upload with the same name is used instead.{' '}
          <a className="underline" href={potentialUrl('LICENSE')} target="_blank" rel="noreferrer">License</a>
          {' · '}
          <a className="underline" href={potentialUrl('README.md')} target="_blank" rel="noreferrer">README</a>
        </p>
        {loadError && <p role="alert" className={`rounded border p-2 text-[11px] ${ct.danger}`}>The library index could not be loaded: {loadError}</p>}
        {addError && <p role="alert" className={`rounded border p-2 text-[11px] ${ct.danger}`}>{addError}</p>}
        {index && (
          <>
            <input type="search" value={query} onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by name, element or style (e.g. Ni, Si, eam/alloy)" aria-label="Search the potential library"
              className={`w-full rounded border px-2 py-1 text-xs ${ct.input}`} />
            <p className={`text-[11px] ${ct.muted}`}>
              {shown.length} file{shown.length === 1 ? '' : 's'}{shown.length > LIMIT ? `, showing ${LIMIT}` : ''}
            </p>
            <ul className="space-y-2" aria-label="Potential library files">
              {shown.slice(0, LIMIT).map(row)}
            </ul>
          </>
        )}
      </div>
    </details>
  );
};

export default PotentialLibraryPanel;
