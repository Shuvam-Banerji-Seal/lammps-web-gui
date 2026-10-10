import React, { useEffect, useState } from 'react';
import type { getThemeTokens } from '../../../theme';

type Tokens = ReturnType<typeof getThemeTokens>;

/** Engine frame interval used when the user has not changed it (the notebook's long-standing default). */
export const DEFAULT_RECORD_EVERY = 25;
export const RADIUS_MIN = 0.2;
export const RADIUS_MAX = 1.5;

export const reviveRecordEvery = (raw: unknown): number | null =>
  typeof raw === 'number' && Number.isInteger(raw) && raw >= 1 ? raw : null;

export const reviveRadius = (raw: unknown): number | null =>
  typeof raw === 'number' && Number.isFinite(raw) && raw >= RADIUS_MIN && raw <= RADIUS_MAX ? raw : null;

export const reviveBool = (raw: unknown): boolean | null => (typeof raw === 'boolean' ? raw : null);

interface ViewControlsProps {
  ct: Tokens;
  recordEvery: number;
  onRecordEvery: (n: number) => void;
  radius: number;
  onRadius: (r: number) => void;
  showBox: boolean;
  onShowBox: (on: boolean) => void;
}

/**
 * View panel settings: the record interval (the engine's frameEvery, applied at the next reset or Run all),
 * the atom radius scale and the box toggle (both MoleculeCanvas config fields).
 */
const ViewControls: React.FC<ViewControlsProps> = ({ ct, recordEvery, onRecordEvery, radius, onRadius, showBox, onShowBox }) => {
  // the field is free text while typing; only a whole number >= 1 reaches the engine setting
  const [draft, setDraft] = useState(String(recordEvery));
  useEffect(() => setDraft(String(recordEvery)), [recordEvery]);
  const commit = (text: string) => {
    setDraft(text);
    const n = Number(text);
    if (/^\s*\d+\s*$/.test(text) && Number.isInteger(n) && n >= 1) onRecordEvery(n);
  };
  const invalid = !(/^\s*\d+\s*$/.test(draft) && Number(draft) >= 1);

  return (
    <div className={`flex flex-wrap items-center gap-x-3 gap-y-1 text-xs ${ct.muted}`}>
      <label className="flex items-center gap-1" title="The engine records a frame every N steps during runs. Applies at the next reset or Run all.">
        <span>Record every</span>
        <input type="number" min={1} step={1} inputMode="numeric" aria-label="Record every N steps"
          aria-invalid={invalid} value={draft} onChange={(e) => commit(e.target.value)}
          className={`w-16 min-h-6 rounded border px-1 py-0.5 font-mono text-xs ${ct.input}`} />
        <span>steps</span>
      </label>
      <label className="flex items-center gap-1" title="Scales the atom spheres in the view">
        <span>Atom size</span>
        <input type="range" min={RADIUS_MIN} max={RADIUS_MAX} step={0.05} value={radius} aria-label="Atom radius scale"
          onChange={(e) => onRadius(Number(e.target.value))} className="w-24" />
        <span className="font-mono tabular-nums">{radius.toFixed(2)}×</span>
      </label>
      <label className="flex items-center gap-1">
        <input type="checkbox" checked={showBox} onChange={(e) => onShowBox(e.target.checked)} aria-label="Show simulation box" />
        <span>Box</span>
      </label>
    </div>
  );
};

export default ViewControls;
