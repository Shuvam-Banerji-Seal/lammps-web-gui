/**
 * The browser-storage side panel (src/components/workbench/StoragePanel.tsx): listing and search, download,
 * inline delete, the hand-off actions (Notebook, Script Builder, Viewer), adding files by the hidden input,
 * live refresh on FILESTORE_EVENT, Escape, and the pure helpers isLammpsScript / isViewable.
 */
import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import StoragePanel, { isLammpsScript, isViewable, type StoragePanelProps } from '../src/components/workbench/StoragePanel';
import { FILESTORE_EVENT, FileStoreError, createMemoryFileStore, type FileStore } from '../src/services/fileStore';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const flush = async (): Promise<void> => {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
};

const buttonNamed = (container: HTMLElement | Document, label: string): HTMLButtonElement | undefined =>
  Array.from(container.querySelectorAll('button')).find((b) => b.getAttribute('aria-label') === label || b.textContent?.trim() === label) as
    | HTMLButtonElement
    | undefined;

const click = async (el: Element | undefined): Promise<void> => {
  if (!el) throw new Error('element not found');
  await act(async () => {
    (el as HTMLElement).click();
  });
  await flush();
};

const LAMMPS_SCRIPT = 'units lj\natom_style atomic\nlattice fcc 0.8442\nrun 100\n';
const DATA_FILE = 'LAMMPS data file\n\n2 atoms\n';

describe('pure helpers', () => {
  it('isLammpsScript accepts input-script names and command-led text', () => {
    expect(isLammpsScript('in.melt', '')).toBe(true);
    expect(isLammpsScript('melt.in', '')).toBe(true);
    expect(isLammpsScript('run.lmp', '')).toBe(true);
    expect(isLammpsScript('run.LAMMPS', '')).toBe(true);
    expect(isLammpsScript('notes.txt', '# comment\n\nunits metal\n')).toBe(true);
    expect(isLammpsScript('notes.txt', 'Buy milk and eggs')).toBe(false);
    expect(isLammpsScript('data.lj', DATA_FILE)).toBe(false);
    expect(isLammpsScript('pair.setfl', 'pairstyle')).toBe(false);
  });

  it('isViewable accepts structure, data, dump and trajectory files, not gzip', () => {
    expect(isViewable({ name: 'water.pdb', kind: 'structure' })).toBe(true);
    expect(isViewable({ name: 'water.xyz', kind: 'text' })).toBe(true);
    expect(isViewable({ name: 'lj.data', kind: 'text' })).toBe(true);
    expect(isViewable({ name: 'data.lj', kind: 'text' })).toBe(true);
    expect(isViewable({ name: 'run.dump', kind: 'text' })).toBe(true);
    expect(isViewable({ name: 'traj.lammpstrj', kind: 'text' })).toBe(true);
    expect(isViewable({ name: 'x.cif', kind: 'text' })).toBe(true);
    expect(isViewable({ name: 'data.lj.gz', kind: 'gzip' })).toBe(false);
    expect(isViewable({ name: 'in.melt', kind: 'text' })).toBe(false);
  });
});

describe('StoragePanel', () => {
  let host: HTMLDivElement;
  let root: Root;
  let store: FileStore;
  const onClose = vi.fn();
  const onOpenInNotebook = vi.fn();
  const onOpenInBuilder = vi.fn();
  const onOpenInViewer = vi.fn();

  const render = async (props: Partial<StoragePanelProps> = {}): Promise<void> => {
    await act(async () => {
      root.render(
        <StoragePanel
          theme="dark"
          open
          onClose={onClose}
          onOpenInNotebook={onOpenInNotebook}
          onOpenInBuilder={onOpenInBuilder}
          onOpenInViewer={onOpenInViewer}
          store={store}
          {...props}
        />,
      );
    });
    await flush();
  };

  const rowNames = (): string[] =>
    Array.from(host.querySelectorAll('ul[aria-label="Stored files"] > li button[aria-expanded]')).map((b) => b.textContent ?? '');

  beforeAll(() => {
    // jsdom's Blob may lack arrayBuffer(), which readInputText needs for uploads
    if (typeof Blob.prototype.arrayBuffer !== 'function') {
      Object.defineProperty(Blob.prototype, 'arrayBuffer', {
        configurable: true,
        value(this: Blob): Promise<ArrayBuffer> {
          return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as ArrayBuffer);
            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(this);
          });
        },
      });
    }
  });

  beforeEach(async () => {
    vi.clearAllMocks();
    store = createMemoryFileStore();
    await store.put({ name: 'in.melt', text: LAMMPS_SCRIPT, size: LAMMPS_SCRIPT.length });
    await store.put({ name: 'data.lj', text: DATA_FILE, size: DATA_FILE.length });
    await store.put({ name: 'lj.gz', text: '\u001f\u008b\u0008', size: 3 });
    host = document.createElement('div');
    document.body.appendChild(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    host.remove();
  });

  it('renders nothing when closed', async () => {
    await render({ open: false });
    expect(host.querySelector('[role="dialog"]')).toBeNull();
  });

  it('lists the stored files with kind and size, and the header totals', async () => {
    await render();
    const dialog = host.querySelector('[role="dialog"]') as HTMLElement;
    expect(dialog.getAttribute('aria-label')).toBe('Browser storage');
    expect(rowNames()).toEqual(['data.lj', 'in.melt', 'lj.gz']);
    expect(dialog.textContent).toContain('3 files');
    // the test store is the memory fallback, so the header says so
    expect(dialog.textContent).toContain('memory only: files are lost on reload');
    const row = Array.from(host.querySelectorAll('li')).find((li) => li.textContent?.includes('in.melt'));
    expect(row?.textContent).toContain(`${LAMMPS_SCRIPT.length} B`);
    expect(row?.textContent).toContain('text');
  });

  it('filters the list by name', async () => {
    await render();
    const search = host.querySelector('input[type="search"]') as HTMLInputElement;
    await act(async () => {
      // React tracks the value, so set it through the native setter before the input event
      const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setValue?.call(search, 'DATA');
      search.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await flush();
    expect(rowNames()).toEqual(['data.lj']);
  });

  it('downloads a text file as a text Blob and a gzip file as bytes', async () => {
    const blobs: Blob[] = [];
    const createObjectURL = vi.fn((b: Blob) => {
      blobs.push(b);
      return 'blob:test';
    });
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL, revokeObjectURL }));
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    try {
      await render();
      await click(buttonNamed(host, 'Download in.melt'));
      expect(createObjectURL).toHaveBeenCalledTimes(1);
      expect(blobs[0].type).toContain('text/plain');
      expect(blobs[0].size).toBe(LAMMPS_SCRIPT.length);

      await click(buttonNamed(host, 'Download lj.gz'));
      const gz = blobs[1];
      expect(gz.type).toBe('application/gzip');
      expect(gz.size).toBe(3);
      const bytes = new Uint8Array(await gz.arrayBuffer());
      expect(Array.from(bytes)).toEqual([0x1f, 0x8b, 0x08]);
      expect(clickSpy).toHaveBeenCalledTimes(2);
    } finally {
      clickSpy.mockRestore();
      vi.unstubAllGlobals();
    }
  });

  it('deletes a file only after the inline Yes', async () => {
    await render();
    await click(buttonNamed(host, 'Delete data.lj'));
    expect(host.textContent).toContain('Delete data.lj?');
    await click(buttonNamed(host, 'No'));
    expect(await store.get('data.lj')).toBeDefined();

    await click(buttonNamed(host, 'Delete data.lj'));
    await click(buttonNamed(host, 'Yes'));
    expect(await store.get('data.lj')).toBeUndefined();
    expect(rowNames()).toEqual(['in.melt', 'lj.gz']);
  });

  it('Open in Notebook passes the script text and closes the panel', async () => {
    await render();
    await click(buttonNamed(host, 'Open in Notebook'));
    expect(onOpenInNotebook).toHaveBeenCalledWith(LAMMPS_SCRIPT);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Open in Script Builder passes the file and closes the panel', async () => {
    await render();
    await click(buttonNamed(host, 'Open in Script Builder'));
    expect(onOpenInBuilder).toHaveBeenCalledWith({ text: LAMMPS_SCRIPT, name: 'in.melt' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Open in Viewer passes a data file and closes the panel; gzip files get no viewer action', async () => {
    await render();
    await click(buttonNamed(host, 'Open in Viewer'));
    expect(onOpenInViewer).toHaveBeenCalledWith({ name: 'data.lj', text: DATA_FILE });
    expect(onClose).toHaveBeenCalledTimes(1);
    // only data.lj is viewable: the Viewer action appears once, in the list
    expect(Array.from(host.querySelectorAll('button')).filter((b) => b.textContent === 'Open in Viewer')).toHaveLength(1);
  });

  it('Copy name uses the clipboard when it exists and is silent when not', async () => {
    await render();
    await click(buttonNamed(host, 'Copy name'));
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    try {
      await click(buttonNamed(host, 'Copy name'));
      expect(writeText).toHaveBeenCalledWith('data.lj');
    } finally {
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
    }
  });

  it('adds a file through the hidden input and lists it', async () => {
    await render();
    const input = host.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['units metal\nrun 10\n'], 'in.metal', { type: 'text/plain' });
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    await act(async () => {
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await flush();
    const stored = await store.get('in.metal');
    expect(stored).toMatchObject({ text: 'units metal\nrun 10\n', size: file.size, kind: 'text' });
    expect(rowNames()).toContain('in.metal');
  });

  it('shows Browser storage is full when the store rejects a file for quota', async () => {
    const full: FileStore = {
      ...store,
      backend: 'memory',
      put: async () => {
        throw new FileStoreError('quota', 'full');
      },
    };
    await render({ store: full });
    const input = host.querySelector('input[type="file"]') as HTMLInputElement;
    Object.defineProperty(input, 'files', { configurable: true, value: [new File(['x'], 'big.data')] });
    await act(async () => {
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await flush();
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('Browser storage is full');
  });

  it('adds files dropped on the drawer', async () => {
    await render();
    const dialog = host.querySelector('[role="dialog"]') as HTMLElement;
    const file = new File(['units lj\n'], 'in.drop', { type: 'text/plain' });
    const dropEvent = new Event('drop', { bubbles: true, cancelable: true }) as Event & { dataTransfer?: unknown };
    Object.defineProperty(dropEvent, 'dataTransfer', { value: { files: [file], types: ['Files'] } });
    await act(async () => {
      dialog.dispatchEvent(dropEvent);
    });
    await flush();
    expect(await store.get('in.drop')).toBeDefined();
    expect(rowNames()).toContain('in.drop');
  });

  it('re-lists when FILESTORE_EVENT fires', async () => {
    await render();
    expect(rowNames()).not.toContain('late.data');
    await store.put({ name: 'late.data', text: DATA_FILE, size: DATA_FILE.length });
    await act(async () => {
      window.dispatchEvent(new Event(FILESTORE_EVENT));
    });
    await flush();
    expect(rowNames()).toContain('late.data');
  });

  it('clears all files after an inline confirmation', async () => {
    await render();
    await click(buttonNamed(host, 'Clear all'));
    expect(host.textContent).toContain('Remove all 3 files?');
    await click(buttonNamed(host, 'Yes'));
    expect(await store.list()).toEqual([]);
    expect(host.textContent).toContain('No files are stored yet.');
  });

  it('shows the file preview: text lines, or binary for gzip', async () => {
    await render();
    await click(buttonNamed(host, 'in.melt') ?? Array.from(host.querySelectorAll('button[aria-expanded]')).find((b) => b.textContent === 'in.melt'));
    const pre = host.querySelector('pre');
    expect(pre?.textContent).toContain('units lj');
    await click(Array.from(host.querySelectorAll('button[aria-expanded]')).find((b) => b.textContent === 'lj.gz'));
    expect(host.textContent).toContain('binary (gzip)');
  });

  it('closes on Escape and on the close button', async () => {
    await render();
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(onClose).toHaveBeenCalledTimes(1);
    await click(buttonNamed(host, 'Close browser storage'));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});

describe('the notebook input files follow the store', () => {
  it('a file stored or removed elsewhere (the panel) reaches the engine and the input list', async () => {
    const { useInputFiles } = await import('../src/components/workbench/notebook/useInputFiles');
    const store = createMemoryFileStore();
    const client = { addFile: vi.fn(), removeFile: vi.fn() };
    const getClient = () => client as unknown as import('../src/engine/client').EngineClient;
    let names: string[] = [];
    const Host: React.FC = () => {
      const inputs = useInputFiles(getClient, store);
      names = Object.keys(inputs.inputs);
      return null;
    };
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);
    await act(async () => root.render(<Host />));
    await flush();
    await act(async () => { await store.put({ name: 'data.lj', text: DATA_FILE, size: DATA_FILE.length }); });
    await flush();
    expect(client.addFile).toHaveBeenCalledWith('data.lj', DATA_FILE);
    expect(names).toEqual(['data.lj']);
    await act(async () => { await store.remove('data.lj'); });
    await flush();
    expect(client.removeFile).toHaveBeenCalledWith('data.lj');
    expect(names).toEqual([]);
    await act(async () => root.unmount());
    host.remove();
  });
});
