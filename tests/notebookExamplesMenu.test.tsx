/**
 * The notebook menu's "Open an example…": the list comes from public/notebooks/index.json and an example opens as an
 * imported notebook with its input files. fetch is stubbed with the files of public/notebooks.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import NotebookMenu from '../src/components/workbench/notebook/NotebookMenu';
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

const stubSession = () => ({
  baseId: null, cells: [], entries: [], name: 'Untitled', saved: true,
  importNotebook: vi.fn(async () => {}), refresh: vi.fn(async () => {}),
  dismissNotice: vi.fn(), duplicate: vi.fn(), newNotebook: vi.fn(), notice: null,
  openEntry: vi.fn(), remove: vi.fn(), rename: vi.fn(), save: vi.fn(async () => true),
  saveAs: vi.fn(async () => true), setName: vi.fn(), storage: null,
}) as unknown as Parameters<typeof NotebookMenu>[0]['session'];

const stubInputFiles = (): InputFiles => ({
  inputs: {}, totalBytes: 0, error: null,
  addPicked: vi.fn(async () => null), addText: vi.fn(async () => {}),
  readForImport: vi.fn(async () => ({ name: '', text: '' })),
  remove: vi.fn(async () => {}), clearAll: vi.fn(async () => {}), sendToEngine: vi.fn(),
  dismissError: vi.fn(), libraryNames: new Set<string>(), addLibrary: vi.fn(async () => {}),
  fetchLibraryFor: vi.fn(async () => ({ added: [], failed: [] })),
}) as unknown as InputFiles;

const settle = async () => { await act(async () => { await new Promise((r) => setTimeout(r, 0)); }); };

describe('NotebookMenu: Open an example…', () => {
  it('lists the examples of index.json and opens one with its input files', async () => {
    const dir = join(__dirname, '..', 'public', 'notebooks');
    const fetchMock = vi.fn(async (url: string) => {
      const name = String(url).split('/').pop()!;
      return new Response(readFileSync(join(dir, name), 'utf8'), { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);
    const session = stubSession();
    const inputFiles = stubInputFiles();
    await act(async () => {
      root.render(React.createElement(NotebookMenu, { theme: 'light', btn: '', session, inputFiles, busy: false, onStop: () => {}, runs: {} }));
    });
    await act(async () => { container.querySelector<HTMLButtonElement>('button[aria-haspopup]')!.click(); });
    const item = Array.from(document.querySelectorAll<HTMLButtonElement>('[role=menuitem]')).find((b) => b.textContent?.includes('Open an example'));
    expect(item).toBeDefined();
    await act(async () => { item!.click(); });
    await settle();
    expect(String(fetchMock.mock.calls[0][0])).toMatch(/notebooks\/index\.json$/);
    const list = document.querySelector('[aria-label="Example notebooks"] ul, ul[aria-label="Example notebooks"]');
    expect(list?.querySelectorAll('li').length).toBe(6);
    const open = document.querySelector<HTMLButtonElement>('[aria-label="Open the example Ice Ih from a CIF file (rigid SPC/E water)"]');
    expect(open).not.toBeNull();
    await act(async () => { open!.click(); });
    await settle();
    expect(session.importNotebook).toHaveBeenCalledTimes(1);
    const imp = (session.importNotebook as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0] as { name: string; inputFiles: Record<string, string> };
    expect(imp.name).toBe('Ice Ih from a CIF file (rigid SPC/E water)');
    expect(Object.keys(imp.inputFiles).sort()).toEqual(['ice_ih.cif', 'ice_ih.data']);
    expect(inputFiles.addText).toHaveBeenCalledTimes(2);
  });

  it('says so when the list cannot be loaded', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })));
    await act(async () => {
      root.render(React.createElement(NotebookMenu, { theme: 'light', btn: '', session: stubSession(), inputFiles: stubInputFiles(), busy: false, onStop: () => {}, runs: {} }));
    });
    await act(async () => { container.querySelector<HTMLButtonElement>('button[aria-haspopup]')!.click(); });
    const item = Array.from(document.querySelectorAll<HTMLButtonElement>('[role=menuitem]')).find((b) => b.textContent?.includes('Open an example'))!;
    await act(async () => { item.click(); });
    await settle();
    expect(document.querySelector('[role=alert]')?.textContent).toMatch(/could not be loaded \(HTTP 404\)/);
  });
});
