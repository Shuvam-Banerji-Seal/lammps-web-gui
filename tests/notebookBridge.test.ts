import { describe, expect, it } from 'vitest';
import { cellsToScript } from '../src/lammps/notebookBridge';
import { parseScript } from '../src/lammps/scriptParser';
import { generateScript } from '../src/lammps/generator';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

// The notebook's starter cells (src/components/workbench/Notebook.tsx STARTER).
const MELT_CELLS = [
  [
    '# 3d Lennard-Jones melt: a small version of the LAMMPS examples/melt setup',
    'units           lj',
    'atom_style      atomic',
    'lattice         fcc 0.8442',
    'region          box block 0 5 0 5 0 5',
    'create_box      1 box',
    'create_atoms    1 box',
    'mass            1 1.0',
  ].join('\n'),
  ['velocity        all create 3.0 87287', 'pair_style      lj/cut 2.5', 'pair_coeff      1 1 1.0 1.0 2.5',
    'fix             1 all nve', 'thermo          50'].join('\n'),
  'run             250',
];

const thermoOf = async (script: string): Promise<ThermoRow[]> => {
  const rows: ThermoRow[] = [];
  const session = new Session({ emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); } });
  await session.execute(script);
  return rows;
};

describe('notebook <-> script builder bridge', () => {
  it('joins the non-empty cells in order, one blank line apart', () => {
    expect(cellsToScript(['units lj  \n', '   ', '', 'run 0\n\n'])).toBe('units lj\n\nrun 0\n');
    expect(cellsToScript([])).toBe('\n');
  });

  it('notebook cells -> Builder model -> generated script runs with the same thermo', async () => {
    const script = cellsToScript(MELT_CELLS);
    const { model, stats } = parseScript(script, 'From MD Notebook');
    expect(stats.raw).toBe(0);
    expect(stats.recognized).toBe(stats.total);
    const generated = generateScript(model).text;
    const direct = await thermoOf(script);
    expect(direct.map((r) => r.step)).toEqual([0, 50, 100, 150, 200, 250]);
    expect(await thermoOf(generated)).toEqual(direct);
  });
});
