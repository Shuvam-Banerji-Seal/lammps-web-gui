/**
 * Accessibility/UX contract for the Compiler Helper options drawer
 * (jsdom, react-dom/client + act). Mirrors the real-browser acceptance
 * checks: the mobile drawer must leave the tab order when closed, behave
 * as a modal dialog when open, surface build-option help as visible text,
 * and clipboard failures must reach the user.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import CompilerHelper from '../src/components/workbench/CompilerHelper';
import { BUILD_OPTIONS } from '../src/lammps/compiler';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const renderComp = async (theme: 'dark' | 'light' = 'dark') => {
  await act(async () => {
    root.render(React.createElement(CompilerHelper, { theme }));
  });
};

const click = async (el: Element) => {
  await act(async () => {
    el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
};

const pressKey = async (key: string) => {
  await act(async () => {
    document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
  });
};

const buttonByLabel = (label: string) =>
  container.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);

const drawerEl = () =>
  [...container.querySelectorAll<HTMLDivElement>('div')]
    .find(d => d.className.includes('transition-transform'));

const openDrawer = async () => {
  const toggle = buttonByLabel('Toggle build options');
  expect(toggle).toBeTruthy();
  await click(toggle!);
};

beforeEach(() => {
  localStorage.clear();
  window.innerWidth = 360;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

describe('CompilerHelper drawer a11y (mobile)', () => {
  it('the closed off-canvas drawer is inert (nothing inside is focusable)', async () => {
    window.innerWidth = 360;
    await renderComp();
    const drawer = drawerEl();
    expect(drawer).toBeTruthy();
    expect(drawer!.hasAttribute('inert')).toBe(true);
    // The dialog role only exists while the drawer is open.
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    // Controls inside (Build type select) live under the inert element.
    const select = [...drawer!.querySelectorAll('select')];
    expect(select.length).toBeGreaterThan(0);
    expect(select[0].closest('[inert]')).toBe(drawer);
  });

  it('when open it is role=dialog + aria-modal with a name, and the script pane is inert', async () => {
    window.innerWidth = 360;
    await renderComp();
    await openDrawer();
    const dlg = container.querySelector<HTMLElement>('[role="dialog"][aria-modal="true"]');
    expect(dlg).toBeTruthy();
    expect(dlg!.getAttribute('aria-label')).toBe('Build options');
    expect(dlg!.hasAttribute('inert')).toBe(false);
    // Focus moved into the drawer (its close button).
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Close build options');
    // The covered script pane (contains the generated script <pre>) is inert.
    const pane = [...container.querySelectorAll<HTMLElement>('[inert]')]
      .find(el => el.querySelector('pre'));
    expect(pane).toBeTruthy();
    expect(dlg!.contains(pane!)).toBe(false);
  });

  it('Escape closes the drawer and focus returns to the toggle button', async () => {
    window.innerWidth = 360;
    await renderComp();
    await openDrawer();
    expect(container.querySelector('[role="dialog"]')).toBeTruthy();
    await pressKey('Escape');
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Toggle build options');
  });

  it('every BUILD_OPTIONS entry with help renders visible help text linked to its select', async () => {
    await renderComp();
    for (const bo of BUILD_OPTIONS) {
      if (!bo.help) continue;
      const helpEl = container.querySelector<HTMLElement>(`#opt-help-${bo.key}`);
      expect(helpEl, `missing help node for ${bo.key}`).toBeTruthy();
      expect(helpEl!.textContent).toBe(bo.help);
      const select = container.querySelector<HTMLSelectElement>(`select[aria-describedby="opt-help-${bo.key}"]`);
      expect(select, `select not described by help for ${bo.key}`).toBeTruthy();
    }
  });

  it('flag chips expose aria-pressed (false until selected)', async () => {
    await renderComp();
    const chips = container.querySelectorAll<HTMLButtonElement>('div.flex-wrap > button');
    expect(chips.length).toBeGreaterThan(0);
    expect(chips[0].getAttribute('aria-pressed')).toBe('false');
    await click(chips[0]);
    const chipsNow = container.querySelectorAll<HTMLButtonElement>('div.flex-wrap > button');
    expect(chipsNow[0].getAttribute('aria-pressed')).toBe('true');
    expect(buttonByLabel('Close flag details')).toBeTruthy();
  });

  it('a rejected clipboard write shows a role=alert message', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: () => Promise.reject(new Error('blocked')) },
    });
    await renderComp();
    const copy = [...container.querySelectorAll<HTMLButtonElement>('button')]
      .find(b => b.textContent.trim() === 'Copy');
    expect(copy).toBeTruthy();
    await click(copy!);
    await act(async () => { await Promise.resolve(); });
    const alert = container.querySelector<HTMLElement>('[role="alert"]');
    expect(alert).toBeTruthy();
    expect(alert!.textContent).toMatch(/Copy failed/);
  });
});
