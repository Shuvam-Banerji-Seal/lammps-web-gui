import React, { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import { getThemeTokens, type Theme } from '../../../theme';
import { PotentialLibraryPanel } from './PotentialLibraryPanel';

export const formatBytes = (n: number): string =>
  n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`;

export interface InputFilesPanelProps {
  theme: Theme;
  inputs: Record<string, number>;
  totalBytes: number;
  error: string | null;
  busy: boolean;
  onRemove: (name: string) => void;
  onClearAll: () => void;
  onDismissError: () => void;
  /** Names of the stored files that came from the LAMMPS potential library (marked in the list). */
  libraryNames?: ReadonlySet<string>;
  /** Adds one file of the potential library to the inputs. */
  onAddLibrary?: (name: string) => Promise<void>;
  /** The browser engine's pair_style names, from its ready message (null until ready). */
  pairStyles?: readonly string[] | null;
}

/**
 * The "Input files" block of the notebook's right panel: the stored files, their total size, Clear all, and the
 * potential library browser (always shown, so a library file can be added before any input exists).
 */
export const InputFilesPanel: React.FC<InputFilesPanelProps> = ({
  theme, inputs, totalBytes, error, busy, onRemove, onClearAll, onDismissError, libraryNames, onAddLibrary, pairStyles,
}) => {
  const ct = getThemeTokens(theme);
  /** Clear all asks for a second click (within 4 s). */
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    if (!confirming) return;
    const t = setTimeout(() => setConfirming(false), 4000);
    return () => clearTimeout(t);
  }, [confirming]);
  const names = Object.keys(inputs);
  const btn = `inline-flex min-h-6 min-w-6 items-center justify-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${ct.hoverSurface}`;

  const library = onAddLibrary
    ? <PotentialLibraryPanel theme={theme} inputs={inputs} onAdd={onAddLibrary} pairStyles={pairStyles ?? null} />
    : null;
  if (names.length === 0 && !error) return <>{library}</>;
  return (
    <div>
      <div className="flex items-center justify-between gap-2">
        <p className={`text-[11px] font-semibold ${ct.muted}`}>
          Input files{names.length > 0 ? ` (${names.length}, ${formatBytes(totalBytes)} stored in this browser)` : ''}
        </p>
        {names.length > 0 && (
          <button className={`${btn} ${ct.button}`} disabled={busy}
            onClick={() => {
              if (confirming) {
                setConfirming(false);
                onClearAll();
              } else {
                setConfirming(true);
              }
            }}>
            {confirming ? `Remove all ${names.length} files?` : 'Clear all'}
          </button>
        )}
      </div>
      {error && (
        <p role="alert" className={`mt-1 flex items-start justify-between gap-2 rounded border p-2 text-[11px] ${ct.danger}`}>
          <span>{error}</span>
          <button className={btn} aria-label="Dismiss file error" onClick={onDismissError}><X size={12} aria-hidden="true" /></button>
        </p>
      )}
      {names.length > 0 && (
        <ul className="mt-1 flex flex-wrap gap-1" aria-label="Input files">
          {names.map((name) => (
            <li key={name} className={`inline-flex items-center rounded text-xs ${ct.chip}`}>
              <span className="px-2 py-1 font-mono">{name}</span>
              {libraryNames?.has(name) && (
                <span className={`pr-1 text-[10px] ${ct.muted}`} title="A file of the LAMMPS potential library">
                  LAMMPS potential library (GPL-2.0)
                </span>
              )}
              <span className={`pr-1 ${ct.muted}`}>{formatBytes(inputs[name])}</span>
              <button className={`${btn} ${ct.hoverSurface}`} aria-label={`Remove ${name}`} title={`Remove ${name}`}
                disabled={busy} onClick={() => onRemove(name)}>
                <X size={12} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ul>
      )}
      {library}
    </div>
  );
};

export default InputFilesPanel;
