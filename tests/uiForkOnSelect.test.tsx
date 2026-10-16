/**
 * Regression 1: on desktop the flowchart's step actions (Fork a concept here,
 * move up/down, ...) were reachable only on hover, so a click that selects a
 * step left "Fork a concept here" hidden. The selected step now always shows
 * them; other steps keep the hover/focus reveal (class contract; jsdom has no
 * hover or layout).
 * Regression 2: at tablet width (768-1023 px) the step editor was a fixed
 * 320 px column beside the 288 px palette, leaving the flowchart 160 px; it is
 * now an overlay drawer there, opened by selecting a step.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import ScriptBuilder from '../src/components/workbench/ScriptBuilder';
import { SCRIPT_TEMPLATES } from '../src/lammps/templates';

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
const forkButtons = () => [...container.querySelectorAll<HTMLButtonElement>('button[aria-label="Fork a concept here"]')];

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  window.innerWidth = 1280;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  vi.unstubAllGlobals();
  container.remove();
});

describe('Script Builder flowchart: step actions on the selected step', () => {
  it('shows Fork a concept here on the clicked step without hover', async () => {
    await act(async () => {
      root.render(React.createElement(ScriptBuilder, { theme: 'light', onRunInNotebook: () => {} }));
    });
    await click(buttonWithText('Templates')!);
    await click(buttonWithText(SCRIPT_TEMPLATES[0].label)!);
    const steps = [...container.querySelectorAll<HTMLElement>('[role="button"][aria-label$=" step"]')];
    expect(steps.length).toBeGreaterThan(5);
    const target = steps[3];
    await click(target);
    expect(target.getAttribute('aria-pressed')).toBe('true');
    const rowOf = (b: HTMLButtonElement) => b.parentElement!;
    const forks = forkButtons();
    const mine = forks.find(b => target.contains(b))!;
    expect(mine).toBeTruthy();
    expect(rowOf(mine).className).not.toContain('md:opacity-0');
    // an unselected step keeps the hover/focus reveal
    const other = forks.find(b => !target.contains(b))!;
    expect(rowOf(other).className).toContain('md:opacity-0');
    expect(rowOf(other).className).toContain('md:group-hover:opacity-100');
  });

  const editor = () => [...container.querySelectorAll<HTMLElement>('div.overflow-y-auto.border-l')].pop()!;
  const loadTemplate = async () => {
    await act(async () => {
      root.render(React.createElement(ScriptBuilder, { theme: 'light', onRunInNotebook: () => {} }));
    });
    await click(buttonWithText('Templates')!);
    await click(buttonWithText(SCRIPT_TEMPLATES[0].label)!);
  };

  it('opens the step editor as a drawer at tablet width', async () => {
    window.innerWidth = 800;
    await loadTemplate();
    expect(editor().className).toContain('fixed');
    expect(editor().className).toContain('translate-x-full');
    const step = container.querySelector<HTMLElement>('[role="button"][aria-label$=" step"]')!;
    await click(step);
    expect(editor().className).toContain('translate-x-0');
    expect(container.querySelector('div.fixed.inset-0.z-30')).toBeTruthy();
    // the backdrop closes it again
    await click(container.querySelector('div.fixed.inset-0.z-30')!);
    expect(editor().className).toContain('translate-x-full');
  });

  it('keeps the step editor as a column at desktop width', async () => {
    window.innerWidth = 1280;
    await loadTemplate();
    expect(editor().className).toContain('w-80 shrink-0');
    expect(editor().className).not.toContain('fixed');
  });
});
