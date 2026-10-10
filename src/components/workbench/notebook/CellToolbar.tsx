import React from 'react';
import { ChevronDown, ChevronUp, ChevronsDownUp, Copy, Pencil, Play, Plus, Trash2, Type, Code, Network, Eye } from 'lucide-react';
import type { ThemeTokens } from '../../../theme';

/** Shared class for the small icon and text buttons in the notebook cells. */
export const cellBtn = (ct: ThemeTokens): string =>
  `inline-flex min-h-6 min-w-6 items-center justify-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${ct.hoverSurface}`;

export interface CellToolbarProps {
  ct: ThemeTokens;
  /** 1-based cell number, for the labels. */
  n: number;
  kind: 'code' | 'markdown';
  busy: boolean;
  isFirst: boolean;
  isLast: boolean;
  canDelete: boolean;
  /** Code cells: run status text (idle shows nothing). */
  status: string;
  statusError: boolean;
  /** Code cells: output is collapsed (hidden). */
  collapsed: boolean;
  /** Code cells: the flowchart is shown below the input. */
  flowOpen: boolean;
  /** Markdown cells: the text is in the editor (otherwise rendered). */
  editing: boolean;
  onRun: () => void;
  onToggleKind: () => void;
  onToggleEdit: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onDuplicate: () => void;
  onToggleCollapse: () => void;
  onToggleFlow: () => void;
  onDelete: () => void;
  /** Extra content after the status (the run progress bar). */
  children?: React.ReactNode;
}

/** The row of actions above a cell. Every icon button has an aria-label and a title. */
const CellToolbar: React.FC<CellToolbarProps> = (p) => {
  const { ct, n } = p;
  const btn = cellBtn(ct);
  const code = p.kind === 'code';
  return (
    <div role="toolbar" aria-label={`Cell ${n} actions`} className="mb-1 flex flex-wrap items-center gap-1">
      <span className={`font-mono text-[11px] ${ct.muted}`}>{code ? `In [${n}]` : 'Text'}</span>
      {code && (
        <button type="button" className={`${btn} ${ct.button}`} aria-label={`Run cell ${n}`} title="Run this cell (Shift+Enter)"
          disabled={p.busy} onClick={p.onRun}>
          <Play size={12} aria-hidden="true" />
        </button>
      )}
      {!code && (
        <button type="button" className={`${btn} ${ct.button}`} aria-label={p.editing ? `Done editing text cell ${n}` : `Edit text cell ${n}`}
          title={p.editing ? 'Render the Markdown (Shift+Enter)' : 'Edit the Markdown source (double-click also works)'}
          aria-pressed={p.editing} onClick={p.onToggleEdit}>
          {p.editing ? <Eye size={12} aria-hidden="true" /> : <Pencil size={12} aria-hidden="true" />}
          {p.editing ? 'Done' : 'Edit'}
        </button>
      )}
      {code && (
        <span aria-live="polite" className={`text-[11px] ${p.statusError ? ct.danger : ct.muted}`}>{p.status}</span>
      )}
      {code && p.children}
      <div className="ml-auto flex flex-wrap items-center gap-1">
        <button type="button" className={`${btn} ${ct.button}`} aria-label={`Move cell ${n} up`} title="Move up (Ctrl/Cmd+Shift+Up)"
          disabled={p.isFirst} onClick={p.onMoveUp}>
          <ChevronUp size={12} aria-hidden="true" />
        </button>
        <button type="button" className={`${btn} ${ct.button}`} aria-label={`Move cell ${n} down`} title="Move down (Ctrl/Cmd+Shift+Down)"
          disabled={p.isLast} onClick={p.onMoveDown}>
          <ChevronDown size={12} aria-hidden="true" />
        </button>
        <button type="button" className={`${btn} ${ct.button}`} aria-label={`Duplicate cell ${n}`} title="Duplicate this cell"
          onClick={p.onDuplicate}>
          <Copy size={12} aria-hidden="true" />
        </button>
        <button type="button" className={`${btn} ${ct.button}`} aria-label={`Change cell ${n} to ${code ? 'text' : 'code'}`}
          title={code ? 'Make this a Markdown text cell' : 'Make this a LAMMPS code cell'} disabled={p.busy} onClick={p.onToggleKind}>
          {code ? <Type size={12} aria-hidden="true" /> : <Code size={12} aria-hidden="true" />}
          {code ? 'Text' : 'Code'}
        </button>
        {code && (
          <button type="button" className={`${btn} ${ct.button}`} aria-label={`${p.collapsed ? 'Expand' : 'Collapse'} output of cell ${n}`}
            title={p.collapsed ? 'Show the output' : 'Hide the output'} aria-expanded={!p.collapsed} onClick={p.onToggleCollapse}>
            <ChevronsDownUp size={12} aria-hidden="true" />
          </button>
        )}
        {code && (
          <button type="button" className={`${btn} ${p.flowOpen ? ct.accentSoft : ct.button}`} aria-label={`${p.flowOpen ? 'Hide' : 'Show'} flowchart of cell ${n}`}
            title="Flowchart of this cell's commands" aria-pressed={p.flowOpen} onClick={p.onToggleFlow}>
            <Network size={12} aria-hidden="true" />
          </button>
        )}
        <button type="button" className={`${btn} ${ct.button}`} aria-label={`Delete cell ${n}`} title="Delete this cell"
          disabled={p.busy || !p.canDelete} onClick={p.onDelete}>
          <Trash2 size={12} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
};

export default CellToolbar;

export interface AddCellRowProps {
  ct: ThemeTokens;
  n: number;
  onAdd: (kind: 'code' | 'markdown') => void;
}

/** "+ Code" and "+ Markdown" below a cell: each inserts an empty cell of that kind after it. */
export const AddCellRow: React.FC<AddCellRowProps> = ({ ct, n, onAdd }) => {
  const btn = cellBtn(ct);
  return (
    <div className="mt-1 flex flex-wrap gap-1" role="group" aria-label={`Add a cell below cell ${n}`}>
      <button type="button" className={`${btn} ${ct.button}`} aria-label={`Add a code cell below cell ${n}`}
        title="Insert an empty LAMMPS code cell below" onClick={() => onAdd('code')}>
        <Plus size={12} aria-hidden="true" />Code
      </button>
      <button type="button" className={`${btn} ${ct.button}`} aria-label={`Add a Markdown cell below cell ${n}`}
        title="Insert an empty Markdown text cell below" onClick={() => onAdd('markdown')}>
        <Plus size={12} aria-hidden="true" />Markdown
      </button>
    </div>
  );
};

