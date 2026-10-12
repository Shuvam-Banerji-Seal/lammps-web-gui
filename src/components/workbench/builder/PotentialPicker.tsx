/**
 * Library-file picker for the coefficient field of a pair_coeff step in the Script Builder.
 *
 * It lists the LAMMPS potential files (src/lammps/potentialIndex.ts) that the pair style above the step reads,
 * shows each file's elements, units and size, and on a choice fills the coefficient field with the file name
 * and the element names in the file's order (coeffsForPick). The user then maps those names to atom types.
 */
import React, { useEffect, useMemo, useState } from 'react';
import { BookOpen, Search, X } from 'lucide-react';
import {
  coeffsForPick, filterByStyles, isChoosable, loadPotentialIndex, POTENTIAL_PROVENANCE, searchPotentials,
  type PotentialEntry, type PotentialIndex,
} from '../../../lammps/potentialIndex';
import type { ThemeTokens } from '../../../theme';

interface PotentialPickerProps {
  ct: ThemeTokens;
  /** The pair_style words above this step: the style, or for hybrid the mode and the sub-styles. [] when none. */
  pairWords: readonly string[];
  /** Called with the new text for the coefficient field, and the file chosen. */
  onChoose: (coeffs: string, entry: PotentialEntry) => void;
  /** Id prefix for the controls (the step uid). */
  idPrefix: string;
}

const kb = (bytes: number): string => (bytes < 1024 ? `${bytes} B` : `${Math.round(bytes / 1024)} KB`);

export const PotentialPicker: React.FC<PotentialPickerProps> = ({ ct, pairWords, onChoose, idPrefix }) => {
  const [open, setOpen] = useState(false);
  const [index, setIndex] = useState<PotentialIndex | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState('');
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    if (!open || index) return;
    let live = true;
    loadPotentialIndex()
      .then((i) => { if (live) setIndex(i); })
      .catch(() => { if (live) setFailed(true); });
    return () => { live = false; };
  }, [open, index]);

  // a stable key for the words: the caller builds a fresh array on every render
  const wordKey = pairWords.join(' ');
  const words = useMemo(() => (wordKey ? wordKey.split(' ') : []), [wordKey]);
  const hasStyle = words.length > 0;
  const styleLabel = hasStyle ? (words[0].startsWith('hybrid') ? words.slice(1).join(' ') || words[0] : words[0]) : '';

  const shown = useMemo(() => {
    if (!index) return [];
    const choosable = index.files.filter(isChoosable);
    const byStyle = showAll || !hasStyle ? choosable : filterByStyles(choosable, words);
    return searchPotentials(byStyle, query);
  }, [index, showAll, hasStyle, query, words]);

  const choose = (entry: PotentialEntry) => {
    if (!index) return;
    onChoose(coeffsForPick(entry, words, index), entry);
    setOpen(false);
    setQuery('');
  };

  const pid = `${idPrefix}-potential`;

  if (!open) {
    return (
      <button
        id={`${pid}-open`}
        type="button"
        onClick={() => setOpen(true)}
        className={`inline-flex min-h-8 items-center gap-1.5 rounded border px-2.5 py-1.5 text-[12px] font-medium ${ct.button}`}
      >
        <BookOpen size={13} /> Pick a LAMMPS library file
      </button>
    );
  }

  return (
    <section id={pid} aria-label="LAMMPS library files" className={`space-y-2 rounded-lg border p-3 ${ct.divider}`}>
      <div className="flex items-center justify-between gap-2">
        <h4 className={`text-[12px] font-bold ${ct.headerText}`}>LAMMPS library files</h4>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className={`inline-flex min-h-6 min-w-6 items-center justify-center rounded p-1 ${ct.muted} ${ct.hoverSurface}`}
          title="Close the library"
          aria-label="Close the library"
        >
          <X size={14} />
        </button>
      </div>

      <p className={`text-[11px] leading-relaxed ${ct.muted}`}>
        {hasStyle ? (
          <>Matching <code className="font-mono">{styleLabel}</code>.</>
        ) : (
          <>No <code className="font-mono">pair_style</code> above this step, so every file is listed.</>
        )}{' '}
        {POTENTIAL_PROVENANCE}.
      </p>

      <label className={`flex items-center gap-2 text-[12px] ${ct.muted}`}>
        <input
          id={`${pid}-all`}
          type="checkbox"
          checked={showAll}
          onChange={(e) => setShowAll(e.target.checked)}
          disabled={!hasStyle}
        />
        Show all styles
      </label>

      <div className="relative">
        <Search size={13} className={`pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 ${ct.muted}`} />
        <input
          id={`${pid}-search`}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by name, element or style"
          aria-label="Search library files"
          className={`w-full rounded border py-2 pl-8 pr-2.5 text-[13px] focus:border-[#7fa66b] focus:outline-none ${ct.input}`}
        />
      </div>

      {failed && (
        <p role="alert" className={`text-[11px] ${ct.warn}`}>The library index could not be loaded. Type the file name instead.</p>
      )}
      {!failed && !index && <p className={`text-[11px] ${ct.muted}`}>Loading the library…</p>}

      {index && (
        <>
          <p className={`text-[11px] ${ct.muted}`} aria-live="polite">{shown.length} file{shown.length === 1 ? '' : 's'}</p>
          <ul className="max-h-64 space-y-1 overflow-y-auto pr-1" aria-label="Matching library files">
            {shown.map((f) => (
              <li key={f.name}>
                <button
                  type="button"
                  id={`${pid}-item-${f.name}`}
                  onClick={() => choose(f)}
                  className={`w-full rounded border px-2.5 py-2 text-left ${ct.button} ${ct.hoverSurface}`}
                  aria-label={`Use ${f.name}`}
                >
                  <span className="block break-all font-mono text-[12px]">{f.name}</span>
                  <span className={`block text-[11px] ${ct.muted}`}>
                    {f.kind}
                    {f.elements.length > 0 && <> · elements {f.elements.join(' ')}</>}
                    {f.units && <> · {f.units} units</>}
                    {' · '}{kb(f.bytes)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {shown.length === 0 && <p className={`text-[11px] ${ct.muted}`}>No library file matches. Try “Show all styles” or another search.</p>}
        </>
      )}

      <p className={`text-[11px] leading-relaxed ${ct.muted}`}>
        Choosing a file fills the coefficient field with its name and its element names in the file&apos;s order.
        Check that they match the atom types of your system.
      </p>
    </section>
  );
};

export default PotentialPicker;
