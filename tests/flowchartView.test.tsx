/**
 * Script Builder flowchart fixes (measured in Chromium at 1440, 820 and 390 px before the change):
 *  - Fit (F, and the fit button) left a 19-step pipeline at 100 %: below 50 % it only fitted the width. It now fits the
 *    whole pipeline (never above 100 %).
 *  - The connector lines between cards were a hard-coded dark brown that all but vanished on the dark themes; they use
 *    the theme's edge colour.
 *  - START and END had a line on one side only.
 *  - The title field was a fixed 10rem, cutting "Lennard-Jones fluid (NVT)" while the row had room.
 * jsdom has no layout, so the canvas and column sizes are stubbed for the fit test.
 *
 * Node buttons (measured in Chromium before the change): the fork, move, enable, duplicate and remove buttons of a card
 * and the connector pills did nothing. The card's drag handler (and the canvas pan handler for the pills) captured the
 * pointer on every press, and a captured pointer's click goes to the capturing element, not to the button. jsdom does not
 * retarget clicks, so the test checks that a press on a control captures nothing.
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

  it('a press on a card button or a connector pill does not capture the pointer; one on the card body does', async () => {
    const proto = HTMLElement.prototype as unknown as { setPointerCapture?: (id: number) => void };
    if (!proto.setPointerCapture) proto.setPointerCapture = () => {};
    const capture = vi.spyOn(proto as { setPointerCapture: (id: number) => void }, 'setPointerCapture');
    await open('light');
    const press = async (el: Element) => {
      await act(async () => {
        el.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true, button: 0 }));
        el.dispatchEvent(new MouseEvent('pointerup', { bubbles: true, cancelable: true, button: 0 }));
      });
    };
    const cards = () => [...container.querySelectorAll<HTMLElement>('[role="button"][aria-label$=" step"]')];
    const card = cards()[1];
    for (const title of ['Move up', 'Move down', 'Disable', 'Duplicate step', 'Remove']) {
      const b = card.querySelector<HTMLButtonElement>(`button[title="${title}"]`)!;
      expect(b, title).toBeTruthy();
      await press(b);
    }
    await press(container.querySelector<HTMLButtonElement>('button[aria-label="Connection 2 actions"]')!);
    expect(capture).not.toHaveBeenCalled();
    // the remove button removes its step
    const n = cards().length;
    await click(cards()[1].querySelector<HTMLButtonElement>('button[title="Remove"]')!);
    expect(cards().length).toBe(n - 1);
    // a press on the card body still starts a drag (captures the pointer)
    await press(cards()[1].querySelector('code')!);
    expect(capture).toHaveBeenCalledTimes(1);
  });
});
