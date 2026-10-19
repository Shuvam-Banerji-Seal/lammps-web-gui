/**
 * Drag and drop on the Script Builder: every dropped file is kept in the browser file store (read by the MD
 * Notebook by name), each LAMMPS script also opens as a new tab, and a drop the storage drawer already took
 * (defaultPrevented) is ignored. Plus the shared helpers isLammpsScript and scriptTitle.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import ScriptBuilder from '../src/components/workbench/ScriptBuilder';
import { getFileStore } from '../src/services/fileStore';
import { isLammpsScript, scriptTitle } from '../src/lammps/scriptDetect';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const flush = async () => {
  for (let k = 0; k < 5; k++) await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
};

/** A window drop event carrying files (jsdom has no DataTransfer constructor). */
const dropEvent = (files: File[], prevented = false): Event => {
  const ev = new Event('drop', { bubbles: true, cancelable: true });
  Object.defineProperty(ev, 'dataTransfer', { value: { types: ['Files'], files } });
  if (prevented) ev.preventDefault();
  return ev;
};

const SCRIPT = 'units lj\natom_style atomic\nread_data data.melt\npair_style lj/cut 2.5\npair_coeff * * 1 1\nrun 10\n';
const DATA = 'LAMMPS data file\n\n2 atoms\n1 atom types\n';

beforeEach(async () => {
  localStorage.clear();
  await getFileStore().clear();
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  window.innerWidth = 1280;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root.render(React.createElement(ScriptBuilder, { theme: 'light', onRunInNotebook: () => {} }));
  });
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  vi.unstubAllGlobals();
  container.remove();
});

describe('Script Builder drop', () => {
  it('opens a dropped script as a new tab and keeps every dropped file in the browser store', async () => {
    await act(async () => {
      window.dispatchEvent(dropEvent([new File([SCRIPT], 'in.melt'), new File([DATA], 'data.melt')]));
    });
    await flush();
    const names = (await getFileStore().list()).map((f) => f.name).sort();
    expect(names).toEqual(['data.melt', 'in.melt']);
    const status = container.querySelector('[role="status"]')?.textContent ?? '';
    expect(status).toContain('Opened in.melt as a new tab.');
    expect(status).toContain('Kept in.melt, data.melt in browser storage');
    // the new tab carries the script's commands
    expect(container.textContent).toContain('read_data');
  });

  it('ignores a drop another handler (the storage drawer) already took', async () => {
    await act(async () => { window.dispatchEvent(dropEvent([new File([DATA], 'data.other')], true)); });
    await flush();
    expect(await getFileStore().list()).toHaveLength(0);
    expect(container.querySelector('[role="status"]')).toBeNull();
  });
});

describe('stored files in the step editor', () => {
  it('a read_data step suggests the files in browser storage, and follows the store', async () => {
    await act(async () => {
      window.dispatchEvent(dropEvent([new File([SCRIPT], 'in.melt'), new File([DATA], 'data.melt')]));
    });
    await flush();
    const step = [...container.querySelectorAll<HTMLElement>('[role="button"][aria-label$=" step"]')]
      .find(s => (s.getAttribute('aria-label') ?? '').startsWith('read_data'));
    expect(step).toBeDefined();
    await act(async () => { step!.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })); });
    await flush();
    const options = () => [...document.querySelectorAll('datalist option')].map(o => (o as HTMLOptionElement).value).sort();
    expect(options()).toEqual(['data.melt', 'in.melt']);
    await act(async () => { await getFileStore().put({ name: 'more.data', text: DATA, size: DATA.length }); });
    await flush();
    expect(options()).toEqual(['data.melt', 'in.melt', 'more.data']);
  });
});

describe('script helpers', () => {
  it('scriptTitle names in.* after what follows in. and other scripts after the base name', () => {
    expect(scriptTitle('in.melt')).toBe('melt');
    expect(scriptTitle('dir/in.lj.npt')).toBe('lj.npt');
    expect(scriptTitle('melt.in')).toBe('melt');
    expect(scriptTitle('run.lmp')).toBe('run');
    expect(scriptTitle('in.')).toBe('in.');
  });

  it('isLammpsScript accepts script names and command-led text, not data files', () => {
    expect(isLammpsScript('settings.mod', 'units real\natom_style full\n')).toBe(true);
    expect(isLammpsScript('data.melt', DATA)).toBe(false);
  });
});
