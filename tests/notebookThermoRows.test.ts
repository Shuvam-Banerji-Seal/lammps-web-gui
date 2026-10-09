import { describe, expect, it } from 'vitest';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { ThermoRows } from '../src/components/workbench/Notebook';
import { getThemeTokens } from '../src/theme';
import type { ThermoRow } from '../src/engine/types';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mount = async (rows: ThermoRow[]) => {
  const table = document.createElement('table');
  document.body.appendChild(table);
  const root = createRoot(table);
  await act(async () => {
    root.render(React.createElement(ThermoRows, { ct: getThemeTokens('dark'), keywords: ['step', 'temp'], rows }));
  });
  return { table, root };
};
const rows = (n: number): ThermoRow[] => Array.from({ length: n }, (_, k) => ({ step: 10 * k, temp: 1 }) as ThermoRow);

describe('notebook thermo table rows', () => {
  it('a short table shows every row', async () => {
    const { table, root } = await mount(rows(50));
    expect(table.querySelectorAll('tbody tr').length).toBe(50);
    act(() => root.unmount());
  });

  it('a long table shows the first row, a hidden-rows line and the latest rows, and all of them on request', async () => {
    const { table, root } = await mount(rows(500));
    const trs = table.querySelectorAll('tbody tr');
    expect(trs.length).toBe(201);
    expect(trs[0].textContent).toContain('0');
    expect(trs[1].textContent).toMatch(/300 earlier rows hidden/);
    expect(trs[200].textContent).toContain('4990');
    await act(async () => { (table.querySelector('button') as HTMLButtonElement).click(); });
    expect(table.querySelectorAll('tbody tr').length).toBe(500);
    act(() => root.unmount());
  });
});
