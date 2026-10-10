import React, { useEffect, useMemo, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { getThemeTokens, type Theme } from '../../../theme';
import { detectFileFormat, detectFormatFromContent, parseFile } from '../../../services/fileParser';
import type { FileFormat, MoleculeData } from '../../../types';
import {
  StructureConvertError,
  convertStructure,
  outputName,
  pdbAtomNames,
  pdbBox,
  type AtomStyleChoice,
  type ConvertOptions,
  type ConvertResult,
} from '../../../lammps/structureToLammps';
import type { ImportSource } from './useInputFiles';

export interface ImportStructureDialogProps {
  theme: Theme;
  source: ImportSource;
  onClose: () => void;
  /** Stores a converted file as an input file and sends it to the engine. */
  onAddText: (name: string, text: string) => Promise<void>;
  /** Appends a code cell with the text (the cell that loads the converted file). */
  onInsertCell: (text: string) => void;
}

/** The parser format of a picked file: by its name, then by its content for generic extensions. */
export const formatOfSource = (source: ImportSource): FileFormat => {
  const ext = source.name.toLowerCase().split('.').pop() ?? '';
  if (ext === 'txt' || ext === 'dump' || ext === 'lammpstrj') return detectFormatFromContent(source.text);
  return detectFileFormat(source.name);
};

interface Prepared {
  mol: MoleculeData;
  format: FileFormat;
  atomNames?: string[];
  box?: ConvertOptions['box'];
}

/** Parses the source with the app's parsers and gathers what the converter needs (cell, PDB names). */
const prepare = (source: ImportSource): Prepared | string => {
  try {
    const format = formatOfSource(source);
    const mol = parseFile(source.text, format);
    if (format === 'pdb') {
      return { mol, format, atomNames: pdbAtomNames(source.text, mol.atoms), box: pdbBox(source.text) };
    }
    return { mol, format, box: mol.box };
  } catch (e) {
    return `The file could not be read: ${e instanceof Error ? e.message : String(e)}`;
  }
};

const withExt = (name: string, ext: 'data' | 'mol'): string => {
  const base = name.replace(/\.[^.]*$/, '') || 'structure';
  return `${base}.${ext}`;
};

const ALL_STYLES: { id: AtomStyleChoice; label: string; hint: string }[] = [
  { id: 'atomic', label: 'atomic', hint: 'x y z only' },
  { id: 'charge', label: 'charge', hint: 'adds charges' },
  { id: 'full', label: 'full', hint: 'molecule IDs and bonds' },
];

export const ImportStructureDialog: React.FC<ImportStructureDialogProps> = ({ theme, source, onClose, onAddText, onInsertCell }) => {
  const ct = getThemeTokens(theme);
  const prepared = useMemo(() => prepare(source), [source]);
  const [target, setTarget] = useState<'data' | 'molecule'>('data');
  const [outName, setOutName] = useState(() => withExt(source.name, 'data'));
  const [style, setStyle] = useState<AtomStyleChoice>('full');
  const [typeBy, setTypeBy] = useState<'element' | 'pdbName'>('element');
  const [bonds, setBonds] = useState(true);
  const [padding, setPadding] = useState(10);
  const [busy, setBusy] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const dialog = useRef<HTMLDivElement>(null);

  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    dialog.current?.querySelector<HTMLElement>('button, input')?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const chooseTarget = (t: 'data' | 'molecule') => {
    setTarget(t);
    setOutName((prev) => withExt(prev, t === 'data' ? 'data' : 'mol'));
  };

  const outcome = useMemo((): { result: ConvertResult } | { error: string } => {
    if (typeof prepared === 'string') return { error: prepared };
    const { mol, format, atomNames, box } = prepared;
    const sourceKind: ConvertOptions['source'] = format === 'lammps' ? 'lammps' : 'elements';
    const opts: ConvertOptions = {
      target,
      source: sourceKind,
      atomStyle: style,
      typeBy: typeBy === 'pdbName' && atomNames ? 'pdbName' : 'element',
      bonds,
      padding: Number.isFinite(padding) && padding >= 0 ? padding : 10,
      fileName: outName.trim() || outputName(source.name, target === 'data' ? 'data' : 'mol'),
      title: `Converted from ${source.name} by the LAMMPS workbench`,
      box,
      atomNames,
    };
    try {
      return { result: convertStructure(mol, opts) };
    } catch (e) {
      if (e instanceof StructureConvertError) return { error: e.message };
      return { error: `Conversion failed: ${e instanceof Error ? e.message : String(e)}` };
    }
  }, [prepared, target, style, typeBy, bonds, padding, outName, source.name]);

  const result = 'result' in outcome ? outcome.result : null;
  const error = 'error' in outcome ? outcome.error : null;
  const atomNamesOk = typeof prepared !== 'string' && !!prepared.atomNames;

  const add = async () => {
    if (!result) return;
    setBusy(true);
    setAddError(null);
    try {
      await onAddText(result.fileName, result.text);
      onInsertCell(result.suggestion);
      onClose();
    } catch (e) {
      setAddError(e instanceof Error ? e.message : 'The file could not be added.');
    } finally {
      setBusy(false);
    }
  };

  const field = `rounded border px-2 py-1 text-xs ${ct.input}`;
  const btn = `inline-flex min-h-6 items-center justify-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${ct.hoverSurface}`;
  const label = `flex items-center gap-2 text-xs ${ct.text}`;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 sm:items-center sm:p-4" role="presentation"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby="import-structure-title"
        className={`max-h-[90vh] w-full overflow-y-auto rounded-t-xl border p-4 sm:max-w-lg sm:rounded-xl ${ct.panel}`}>
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 id="import-structure-title" className={`text-sm font-semibold ${ct.headerText}`}>Import structure</h2>
            <p className={`text-[11px] ${ct.muted}`}>
              <span className="font-mono">{source.name}</span>
              {result ? ` · ${result.natoms.toLocaleString('en-US')} atoms` : ''}
            </p>
          </div>
          <button className={btn} aria-label="Close import" onClick={onClose}><X size={14} aria-hidden="true" /></button>
        </div>

        <fieldset className="mt-3 space-y-2">
          <legend className={`text-[11px] font-semibold ${ct.muted}`}>Output</legend>
          <div className="flex flex-wrap gap-3">
            <label className={label}>
              <input type="radio" name="import-target" checked={target === 'data'} onChange={() => chooseTarget('data')} />
              LAMMPS data file (read_data)
            </label>
            <label className={label}>
              <input type="radio" name="import-target" checked={target === 'molecule'} onChange={() => chooseTarget('molecule')} />
              Molecule template (molecule)
            </label>
          </div>
          <label className={label}>
            File name
            <input className={`${field} flex-1 font-mono`} aria-label="Output file name" value={outName}
              onChange={(e) => setOutName(e.target.value)} />
          </label>
        </fieldset>

        <fieldset className="mt-3 space-y-2">
          <legend className={`text-[11px] font-semibold ${ct.muted}`}>Atom style</legend>
          <div className="flex flex-wrap gap-3">
            {ALL_STYLES.map((s) => (
              <label key={s.id} className={label} title={s.hint}>
                <input type="radio" name="import-style" checked={style === s.id} onChange={() => setStyle(s.id)} />
                {s.label}
              </label>
            ))}
          </div>
          <p className={`text-[11px] ${ct.muted}`}>
            {target === 'molecule'
              ? 'A molecule template always carries bonds; create_atoms mol needs an atom style that stores molecules (full).'
              : 'Bonds are written only with the full style.'}
          </p>
        </fieldset>

        <fieldset className="mt-3 space-y-2">
          <legend className={`text-[11px] font-semibold ${ct.muted}`}>Types and bonds</legend>
          <div className="flex flex-wrap gap-3">
            <label className={label}>
              <input type="radio" name="import-types" checked={typeBy === 'element'} onChange={() => setTypeBy('element')} />
              one type per element
            </label>
            <label className={`${label} ${atomNamesOk ? '' : 'opacity-50'}`}>
              <input type="radio" name="import-types" disabled={!atomNamesOk} checked={typeBy === 'pdbName' && atomNamesOk}
                onChange={() => setTypeBy('pdbName')} />
              one type per PDB atom name
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label className={label}>
              <input type="checkbox" checked={bonds} onChange={(e) => setBonds(e.target.checked)} />
              bonds (CONECT records, else inferred from radii)
            </label>
            <label className={label}>
              Padding (Å, no cell)
              <input type="number" min={0} step={0.5} className={`${field} w-20`} aria-label="Padding in Angstrom"
                value={padding} onChange={(e) => setPadding(parseFloat(e.target.value))} />
            </label>
          </div>
        </fieldset>

        {error && <p role="alert" className={`mt-3 rounded border p-2 text-xs ${ct.danger}`}>{error}</p>}

        {result && (
          <div className="mt-3 space-y-2">
            <p className={`text-xs ${ct.text}`}>
              {result.natoms.toLocaleString('en-US')} atoms, {result.nbonds.toLocaleString('en-US')} bonds, {result.ntypes} atom types
              {target === 'data' ? `, ${result.triclinic ? 'triclinic' : 'orthogonal'} box` : ''}.
            </p>
            <table className={`w-full text-left text-[11px] ${ct.text}`} aria-label="Atom types">
              <thead>
                <tr className={ct.muted}><th className="pr-2 font-semibold">type</th><th className="pr-2 font-semibold">label</th><th className="pr-2 font-semibold">mass</th><th className="font-semibold">atoms</th></tr>
              </thead>
              <tbody>
                {result.types.map((t) => (
                  <tr key={t.id}>
                    <td className="pr-2 font-mono">{t.id}</td>
                    <td className="pr-2">{t.label}</td>
                    <td className="pr-2 font-mono">{t.mass}</td>
                    <td>{t.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.warnings.map((w) => (
              <p key={w} className={`rounded border p-2 text-[11px] ${ct.warn}`}>{w}</p>
            ))}
            <p className={`text-[11px] ${ct.muted}`}>The cell to insert:</p>
            <pre className={`overflow-x-auto rounded border p-2 font-mono text-[11px] ${ct.input}`} aria-label="Suggested cell">{result.suggestion}</pre>
          </div>
        )}

        {addError && <p role="alert" className={`mt-3 text-xs ${ct.danger}`}>{addError}</p>}

        <div className="mt-4 flex flex-wrap justify-end gap-2">
          <button className={`${btn} ${ct.button}`} onClick={onClose}>Cancel</button>
          <button className={`${btn} ${ct.accent}`} disabled={!result || busy} onClick={() => void add()}>
            Add file and insert cell
          </button>
        </div>
      </div>
    </div>
  );
};

export default ImportStructureDialog;
