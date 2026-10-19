/**
 * The notebook view's full-screen mode (useFullscreen: native Fullscreen API and the CSS overlay fallback,
 * Escape leaves the overlay) and the drop handling of the notebook menu (one notebook or script imports; every
 * other dropped file is an input file; a structure opens the import dialog). Pure classification is tested
 * directly. jsdom has no Fullscreen API and no layout, so the native path stubs the element and the document.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { useRef } from 'react';
import { useFullscreen, type Fullscreen } from '../src/components/workbench/notebook/useFullscreen';
import NotebookMenu from '../src/components/workbench/notebook/NotebookMenu';
import { classifyDropped, isNotebookName } from '../src/components/workbench/notebook/notebookFile';
import type { InputFiles } from '../src/components/workbench/notebook/useInputFiles';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  localStorage.clear();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  vi.unstubAllGlobals();
  container.remove();
});

/* ------------------------------------------------------------ useFullscreen */

/** Renders a host element and the hook; the hook's latest result is read through `api`. */
const api: { current: Fullscreen | null; el: HTMLElement | null } = { current: null, el: null };
const Probe: React.FC = () => {
  const ref = useRef<HTMLDivElement>(null);
  const fs = useFullscreen(ref);
  api.current = fs;
  return React.createElement('div', { ref, 'data-probe': '' }, fs.mode ?? 'off');
};

const mountProbe = async () => {
  await act(async () => { root.render(React.createElement(Probe)); });
  api.el = container.querySelector<HTMLElement>('[data-probe]');
};

describe('useFullscreen', () => {
  it('falls back to the overlay when the Fullscreen API is missing (jsdom), and Escape exits it', async () => {
    await mountProbe();
    expect(api.current!.active).toBe(false);

    await act(async () => { api.current!.toggle(); });
    expect(api.current!.mode).toBe('overlay');
    expect(api.current!.active).toBe(true);
    expect(api.el!.textContent).toBe('overlay');

    await act(async () => { window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })); });
    expect(api.current!.mode).toBe(null);
    expect(api.current!.active).toBe(false);
  });

  it('falls back to the overlay when requestFullscreen rejects', async () => {
    await mountProbe();
    api.el!.requestFullscreen = vi.fn(() => Promise.reject(new Error('denied')));
    await act(async () => { api.current!.toggle(); });
    expect(api.current!.mode).toBe('overlay');
  });

  it('uses native full screen when requestFullscreen resolves and follows fullscreenchange', async () => {
    await mountProbe();
    let fullscreenElement: Element | null = null;
    vi.stubGlobal('fullscreenEnabled', true);
    Object.defineProperty(document, 'fullscreenElement', { configurable: true, get: () => fullscreenElement });
    const exitFullscreen = vi.fn(() => {
      fullscreenElement = null;
      document.dispatchEvent(new Event('fullscreenchange'));
      return Promise.resolve();
    });
    Object.defineProperty(document, 'exitFullscreen', { configurable: true, value: exitFullscreen });
    api.el!.requestFullscreen = vi.fn(() => {
      fullscreenElement = api.el;
      document.dispatchEvent(new Event('fullscreenchange'));
      return Promise.resolve();
    });

    await act(async () => { api.current!.toggle(); });
    expect(api.el!.requestFullscreen).toHaveBeenCalledTimes(1);
    expect(api.current!.mode).toBe('native');

    // the browser leaves full screen on its own (Escape): the state follows the event
    await act(async () => {
      fullscreenElement = null;
      document.dispatchEvent(new Event('fullscreenchange'));
    });
    expect(api.current!.mode).toBe(null);

    // toggle again and leave through the hook: document.exitFullscreen is called
    await act(async () => { api.current!.toggle(); });
    expect(api.current!.mode).toBe('native');
    await act(async () => { api.current!.toggle(); });
    expect(exitFullscreen).toHaveBeenCalled();
    expect(api.current!.mode).toBe(null);

    Reflect.deleteProperty(document, 'fullscreenElement');
    Reflect.deleteProperty(document, 'exitFullscreen');
    vi.unstubAllGlobals();
  });
});

/* ------------------------------------------------------------ drop classification */

describe('classifyDropped', () => {
  it('marks notebooks and LAMMPS scripts, and everything else as input files', () => {
    expect(classifyDropped([
      'melt.lmpnb', 'nb.ipynb', 'in.melt', 'run.in', 'job.LMP', 'x.lammps',
      'data.melt', 'Cu.eam', 'Cu.eam.alloy', 'pot.tersoff', 'sw.sw', 'model.yace', 'pot.ace',
      'table.table', 'dump.gz', 'mol.pdb', 'cell.cif', 'mol.xyz', 'notes.txt', 'in_data.txt',
    ])).toEqual([
      'notebook', 'notebook', 'notebook', 'notebook', 'notebook', 'notebook',
      'input', 'input', 'input', 'input', 'input', 'input', 'input',
      'input', 'input', 'input', 'input', 'input', 'input', 'input',
    ]);
  });

  it('reads the file name only: a path keeps its base name for in.* scripts', () => {
    expect(isNotebookName('some/dir/in.melt')).toBe(true);
    expect(isNotebookName('dir/melt.in')).toBe(true);
    expect(isNotebookName('dir/melt.data')).toBe(false);
    expect(isNotebookName('data.in.txt')).toBe(false);
  });
});

/* ------------------------------------------------------------ drop on the notebook menu */

const stubSession = () => ({
  baseId: null, cells: [], entries: [], name: 'Untitled', saved: true,
  importNotebook: vi.fn(async () => {}), refresh: vi.fn(async () => {}),
  dismissNotice: vi.fn(), duplicate: vi.fn(), newNotebook: vi.fn(), notice: null,
  openEntry: vi.fn(), remove: vi.fn(), rename: vi.fn(), save: vi.fn(async () => true),
  saveAs: vi.fn(async () => true), setName: vi.fn(), storage: null,
}) as unknown as Parameters<typeof NotebookMenu>[0]['session'];

const stubInputFiles = (): InputFiles => ({
  inputs: {}, totalBytes: 0, error: null,
  addPicked: vi.fn(async () => null),
  addText: vi.fn(async () => {}),
  readForImport: vi.fn(async () => ({ name: '', text: '' })),
  remove: vi.fn(async () => {}), clearAll: vi.fn(async () => {}), sendToEngine: vi.fn(),
  dismissError: vi.fn(), libraryNames: new Set<string>(), addLibrary: vi.fn(async () => {}),
  fetchLibraryFor: vi.fn(async () => ({ added: [], failed: [] })),
}) as unknown as InputFiles;

/** Dispatches a drop with the given files on an element inside the notebook root (the window listener sees it). */
const drop = async (files: File[]) => {
  const host = container.querySelector<HTMLElement>('[data-notebook-root]')!;
  const ev = new Event('drop', { bubbles: true, cancelable: true });
  Object.defineProperty(ev, 'dataTransfer', { value: { files, types: ['Files'] } });
  await act(async () => { host.dispatchEvent(ev); });
  // the handlers are async: let their awaited work settle
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
};

describe('NotebookMenu drop', () => {
  it('adds a data file and a potential as input files and does not open the import error dialog', async () => {
    const session = stubSession();
    const inputFiles = stubInputFiles();
    const onStructure = vi.fn();
    // the store hands back the first structure or data file, as addPicked does for Add files
    const source = { name: 'melt.data', text: '1 atoms' };
    (inputFiles.addPicked as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce(source);
    await act(async () => {
      root.render(React.createElement('div', { 'data-notebook-root': '' },
        React.createElement(NotebookMenu, {
          theme: 'light', btn: '', session, inputFiles, busy: false, onStop: () => {}, runs: {}, onStructure,
        })));
    });
    const data = new File(['1 atoms'], 'melt.data', { type: 'text/plain' });
    const pot = new File(['pair'], 'Cu.eam', { type: 'text/plain' });
    await drop([data, pot]);

    expect(inputFiles.addPicked).toHaveBeenCalledTimes(1);
    const passed = (inputFiles.addPicked as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0] as File[];
    expect(passed.map((f) => f.name)).toEqual(['melt.data', 'Cu.eam']);
    expect(session.importNotebook).not.toHaveBeenCalled();
    expect(document.body.textContent).not.toContain('Could not import');
    // the structure source goes to the import dialog, as with Add files
    expect(onStructure).toHaveBeenCalledWith(source);
  });

  it('imports one dropped notebook and adds the other files as input files', async () => {
    const session = stubSession();
    const inputFiles = stubInputFiles();
    await act(async () => {
      root.render(React.createElement('div', { 'data-notebook-root': '' },
        React.createElement(NotebookMenu, {
          theme: 'light', btn: '', session, inputFiles, busy: false, onStop: () => {}, runs: {},
        })));
    });
    const nb = new File(['units lj\nrun 0\n'], 'in.melt', { type: 'text/plain' });
    const pot = new File(['pair'], 'Cu.eam', { type: 'text/plain' });
    await drop([nb, pot]);

    expect(session.importNotebook).toHaveBeenCalledTimes(1);
    expect(inputFiles.addPicked).toHaveBeenCalledTimes(1);
    const passed = (inputFiles.addPicked as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0] as File[];
    expect(passed.map((f) => f.name)).toEqual(['Cu.eam']);
    expect(document.body.textContent).toContain('Added 1 input file: Cu.eam.');
  });
});
