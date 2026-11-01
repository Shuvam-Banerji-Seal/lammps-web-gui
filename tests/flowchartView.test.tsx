/**
 * Script Builder flowchart fixes (measured in Chromium at 1440, 820 and 390 px before the change):
 *  - Fit (F, and the fit button) left a 19-step pipeline at 100 %: below 50 % it only fitted the width. It now fits the
 *    whole pipeline (never above 100 %).
 *  - The connector lines between cards were a hard-coded dark brown that all but vanished on the dark themes; they use
 *    the theme's edge colour.
 *  - START and END had a line on one side only.
 *  - The title field was a fixed 10rem, cutting "Lennard-Jones fluid (NVT)" while the row had room.
 * jsdom has no layout, so the canvas and column sizes are stubbed for the fit test.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import ScriptBuilder from '../src/components/workbench/ScriptBuilder';
import { SCRIPT_TEMPLATES } from '../src/lammps/templates';
import { getThemeTokens } from '../src/theme';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const click = async (el: Element) => {
  await act(async () => {
    el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
};
const buttonWithText = (text: string) =>
  [...container.querySelectorAll<HTMLButtonElement>('button')].find(b => b.textContent?.includes(text));

const open = async (theme: 'light' | 'dark') => {
  await act(async () => {
    root.render(React.createElement(ScriptBuilder, { theme, onRunInNotebook: () => {} }));
  });
  await click(buttonWithText('Templates')!);
  // the longest starter pipeline
  const longest = [...SCRIPT_TEMPLATES].sort((a, b) => b.steps.length - a.steps.length)[0];
  await click(buttonWithText(longest.label)!);
  return longest;
};

/** The zoom readout of the canvas controls (e.g. "100%"). */
const zoomText = () => [...container.querySelectorAll('button')].map(b => b.textContent ?? '').find(t => /^\d+%$/.test(t.trim()));

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  window.innerWidth = 1440;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  container.remove();
});

describe('Script Builder flowchart view', () => {
  it('Fit shows the whole pipeline: a long one zooms out below 100 %', async () => {
    // a 832 x 690 canvas and a 592-px column that is 2800 px tall (19 steps at about 140 px each)
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(832);
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(690);
    vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(592);
    vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(2800);
    await open('light');
    const fit = container.querySelector<HTMLButtonElement>('button[aria-label="Fit pipeline to view"]')!;
    await click(fit);
    // (690 - 48) / 2800 = 23 %
    expect(zoomText()).toBe('23%');
    // a short pipeline is not blown up past 100 %
    vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockReturnValue(300);
    await click(fit);
    expect(zoomText()).toBe('100%');
  });

  it('draws the connectors in the theme edge colour and START/END without a one-sided line', async () => {
    await open('dark');
    const edge = getThemeTokens('dark').edgeLine;
    const pills = [...container.querySelectorAll<HTMLButtonElement>('button[aria-label^="Connection "]')];
    expect(pills.length).toBeGreaterThan(5);
    for (const p of pills.slice(0, 3)) {
      expect(p.previousElementSibling?.className).toContain(edge);
      expect(p.nextElementSibling?.className).toContain(edge);
    }
    for (const label of ['START', 'END']) {
      const badge = [...container.querySelectorAll('span')].find(s => s.textContent?.trim() === label)!;
      expect(badge).toBeTruthy();
      expect(badge.parentElement!.children.length).toBe(1);
    }
  });

  it('gives the title the room of its row and its full text as a tooltip', async () => {
    await open('light');
    const title = container.querySelector<HTMLInputElement>('input[aria-label="Script title"]')!;
    expect(title.className).not.toMatch(/\bsm:w-40\b/);
    expect(title.className).toMatch(/\bflex-1\b/);
    expect(title.title).toBe(title.value);
  });
});
