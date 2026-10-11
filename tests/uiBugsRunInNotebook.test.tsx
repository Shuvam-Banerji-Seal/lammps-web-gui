/**
 * Regression: the Script Builder's "Run in Notebook" button must hand the
 * EXACT active script (template or flowchart, either view) to the notebook,
 * and its toolbar group must wrap at md+ instead of overflowing the editor
 * column under the inspector sidebar (jsdom has no layout, so the wrap is
 * checked through the class contract the browser measurement relies on).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import ScriptBuilder from '../src/components/workbench/ScriptBuilder';
import { SCRIPT_TEMPLATES, buildTemplate } from '../src/lammps/templates';
import { generateScript } from '../src/lammps/generator';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const click = async (el: Element) => {
  await act(async () => {
    el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
};

const runButton = () =>
  container.querySelector<HTMLButtonElement>('button[aria-label="Run this script in the MD Notebook"]');

const buttonWithText = (text: string) =>
  [...container.querySelectorAll<HTMLButtonElement>('button')].find(b => b.textContent?.includes(text));

beforeEach(() => {
  localStorage.clear();
  // jsdom has no ResizeObserver; the builder only uses it to measure its column.
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  window.innerWidth = 1024;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  vi.unstubAllGlobals();
  container.remove();
});

describe('Script Builder: Run in Notebook hands over the active script', () => {
  it('sends the loaded template script (script view and flowchart view)', async () => {
    const onRunInNotebook = vi.fn();
    await act(async () => {
      root.render(React.createElement(ScriptBuilder, {
        theme: 'light', onRunInNotebook, onOpenViewer: () => {},
      }));
    });
    const tpl = SCRIPT_TEMPLATES[0];
    const expected = generateScript(buildTemplate(tpl)).text;
    expect(expected).toContain('units');

    await click(buttonWithText('Templates')!);
    await click(buttonWithText(tpl.label)!);
    expect(runButton()).toBeTruthy();

    // flowchart view (default)
    await click(runButton()!);
    expect(onRunInNotebook).toHaveBeenCalledTimes(1);
    expect(onRunInNotebook.mock.calls[0][0]).toBe(expected);

    // script view: the same button sends the same text
    await click(container.querySelector<HTMLButtonElement>('button[title="View the generated script"]')!);
    await click(runButton()!);
    expect(onRunInNotebook).toHaveBeenCalledTimes(2);
    expect(onRunInNotebook.mock.calls[1][0]).toBe(expected);
  });

  it('keeps the Run in Notebook button in the toolbar at desktop widths', async () => {
    await act(async () => {
      root.render(React.createElement(ScriptBuilder, {
        theme: 'light', onRunInNotebook: () => {},
      }));
    });
    const btn = runButton();
    expect(btn).toBeTruthy();
    // Its group must be allowed to wrap onto more rows at md+ (no overflow
    // under the inspector sidebar); the phone strip keeps horizontal scroll.
    const group = btn!.parentElement!;
    expect(group.className).toContain('md:flex-wrap');
    expect(group.className).toContain('max-md:overflow-x-auto');
  });

  it('hides the button when no handler is supplied (Notebook not wired)', async () => {
    await act(async () => {
      root.render(React.createElement(ScriptBuilder, { theme: 'light' }));
    });
    expect(runButton()).toBeNull();
  });

  it('closes the palette drawer when the window narrows to phone width (its backdrop covered the toolbar)', async () => {
    window.innerWidth = 1024;
    await act(async () => {
      root.render(React.createElement(ScriptBuilder, { theme: 'light', onRunInNotebook: () => {} }));
    });
    const palette = () => container.querySelector<HTMLElement>('div.border-r.transition-transform');
    // desktop: the palette column is open (flex w-72), not collapsed
    expect(palette()!.className).toContain('flex w-72');
    window.innerWidth = 390;
    await act(async () => { window.dispatchEvent(new Event('resize')); });
    expect(palette()!.className).toContain('-translate-x-full');
    expect(container.querySelector('div.fixed.inset-0.z-30')).toBeNull();
  });
});
