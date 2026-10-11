/**
 * Regression (notebook audit): the saved mark must describe what was written. A save or Save as that is still in
 * flight when the person edits the cells must leave the notebook marked unsaved; the old code compared against the
 * cells read after the await and so reported the newer edits as saved.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { useNotebookSession, type NotebookSession } from '../src/components/workbench/notebook/useNotebookSession';
import { createMemoryBackend, createNotebookLibrary, type NotebookLibrary } from '../src/components/workbench/notebook/notebookLibrary';
import type { Cell } from '../src/components/workbench/notebook/cells';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const starter: Cell[] = [{ id: 'a', text: 'run 10' }];

let container: HTMLDivElement;
let root: Root;
let session: NotebookSession | null;

const Probe: React.FC<{ library: NotebookLibrary }> = ({ library }) => {
  session = useNotebookSession(starter, library);
  return null;
};

/** A library whose writes wait for `gate` before they reach the backend. */
const gatedLibrary = () => {
  const base = createNotebookLibrary(createMemoryBackend());
  let open: () => void = () => {};
  let gate = new Promise<void>((r) => { open = r; });
  const hold = () => { gate = new Promise<void>((r) => { open = r; }); };
  const lib: NotebookLibrary = {
    ...base,
    create: async (...a: Parameters<NotebookLibrary['create']>) => { await gate; return base.create(...a); },
    update: async (...a: Parameters<NotebookLibrary['update']>) => { await gate; return base.update(...a); },
  };
  return { lib, release: () => open(), hold };
};

const flush = async () => {
  for (let i = 0; i < 5; i++) await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
};

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  session = null;
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

describe('notebook session: saved mark during an in-flight save', () => {
  it('an edit made while Save as is in flight stays unsaved', async () => {
    const g = gatedLibrary();
    await act(async () => { root.render(<Probe library={g.lib} />); });
    await flush();
    expect(session?.ready).toBe(true);

    let saving: Promise<boolean> | undefined;
    await act(async () => { saving = session!.saveAs('First'); });
    // the person keeps typing before the library has written the notebook
    await act(async () => { session!.setCells((cs) => cs.map((c) => ({ ...c, text: 'run 999' }))); });
    await act(async () => { g.release(); });
    await act(async () => { await saving; });
    await flush();

    expect(session!.baseId).not.toBeNull();
    expect(session!.saved).toBe(false);
  });

  it('an edit made while Save is in flight stays unsaved; a save with no edits is saved', async () => {
    const g = gatedLibrary();
    await act(async () => { root.render(<Probe library={g.lib} />); });
    await flush();
    g.release();
    await act(async () => { await session!.saveAs('Named'); });
    await flush();
    expect(session!.saved).toBe(true);

    let saving: Promise<boolean> | undefined;
    g.hold();
    await act(async () => { session!.setCells((cs) => cs.map((c) => ({ ...c, text: 'run 1' }))); });
    await act(async () => { saving = session!.save(); });
    await act(async () => { session!.setCells((cs) => cs.map((c) => ({ ...c, text: 'run 2' }))); });
    await act(async () => { g.release(); });
    await act(async () => { await saving; });
    await flush();
    expect(session!.saved).toBe(false);
  });
});
