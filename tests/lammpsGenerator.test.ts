import { describe, it, expect } from 'vitest';
import {
  generateScript,
  deriveFlowchart,
  sortStepsBySection,
  isSectionSorted,
} from '../src/lammps/generator';
import {
  COMMAND_BY_ID,
  defaultParams,
  ScriptModel,
  ScriptStep,
} from '../src/lammps/catalog';

const step = (defId: string, overrides: Record<string, string> = {}): ScriptStep => {
  const def = COMMAND_BY_ID[defId];
  return {
    uid: `${defId}-1`,
    defId,
    params: { ...defaultParams(def), ...overrides },
    enabled: true,
  };
};

const model = (steps: ScriptStep[], title = 'Test system'): ScriptModel => ({
  title,
  steps,
});

describe('LAMMPS script generator', () => {
  it('renders a canonical LJ-fluid script in section order', () => {
    const m = model([
      step('units', { style: 'lj' }),
      step('dimension'),
      step('boundary'),
      step('atom_style_cmd', { style: 'atomic' }),
      step('region_block', { id: 'box', xhi: '20', yhi: '20', zhi: '20' }),
      step('create_box', { ntypes: '1', region: 'box' }),
      step('create_atoms', { type: '1' }),
      step('mass', { type: '1', value: '1.0' }),
      step('velocity_create', { temp: '1.0', seed: '12345' }),
      step('pair_style_popular', { style: 'lj/cut', args: '2.5' }),
      step('pair_coeff', { types: '* *', coeffs: '1.0 1.0 2.5' }),
      step('thermo_style', { fields: 'step temp pe press' }),
      step('thermo', { n: '100' }),
      step('dump_custom', { file: 'traj.lammpstrj', fields: 'id type x y z' }),
      step('fix_nve'),
      step('run', { steps: '10000' }),
    ]);

    const out = generateScript(m);
    expect(out.warnings).toHaveLength(0);

    const text = out.text;
    // Section order respected even though user listed run before thermo? (they didn't here)
    const idx = (s: string) => text.indexOf(s);
    expect(idx('units lj')).toBeGreaterThan(-1);
    expect(idx('# ---- 1 · Simulation setup ----')).toBeGreaterThan(-1);
    expect(text).toContain('boundary p p p');
    expect(text).toContain('atom_style atomic');
    expect(text).toContain('region box block 0 20 0 20 0 20');
    expect(text).toContain('create_box 1 box');
    expect(text).toContain('mass 1 1.0');
    expect(text).toContain('velocity all create 1.0 12345'); // dist kw omitted = LAMMPS default (gaussian)
    expect(text).toContain('pair_style lj/cut 2.5');
    expect(text).toContain('pair_coeff * * 1.0 1.0 2.5');
    expect(text).toContain('thermo_style custom step temp pe press');
    expect(text).toContain('dump traj all custom 1000 traj.lammpstrj id type x y z');
    expect(text).toContain('fix integrate all nve');
    expect(text).toContain('run 10000');

    // setup section must appear before interactions, output before control
    expect(idx('Simulation setup')).toBeLessThan(idx('3 · Interactions'));
    expect(idx('4 · Output')).toBeLessThan(idx('5 · Run control'));
  });

  it('emits in PIPELINE order — LAMMPS executes an input file top to bottom', () => {
    const m = model([
      step('run', { steps: '5' }),          // control section first in list
      step('units', { style: 'real' }),     // setup later
    ]);
    const out = generateScript(m);
    const iUnits = out.text.indexOf('units real');
    const iRun = out.text.indexOf('run 5');
    expect(iRun).toBeLessThan(iUnits);
  });

  it('keeps multi-stage scripts correct: write_data stays AFTER run', () => {
    // Regression: section grouping used to hoist the whole `output` section
    // above `control`, so the data file captured the INITIAL state.
    const m = model([
      step('units', { style: 'lj' }),
      step('fix_nve'),
      step('run', { steps: '1000' }),
      step('write_data_out', { file: 'final.data' }),
    ]);
    const text = generateScript(m).text;
    expect(text.indexOf('run 1000')).toBeLessThan(text.indexOf('write_data final.data'));
  });

  it('keeps reset_timestep between two runs instead of hoisting it', () => {
    const m = model([
      step('run', { steps: '1000' }),
      step('reset_timestep', { n: '0' }),
      step('run', { steps: '5000' }),
    ]);
    const text = generateScript(m).text;
    const iReset = text.indexOf('reset_timestep 0');
    expect(text.indexOf('run 1000')).toBeLessThan(iReset);
    expect(iReset).toBeLessThan(text.lastIndexOf('run 5000'));
  });

  it('sortStepsBySection() physically reorders so the flowchart matches', () => {
    const steps = [step('run', { steps: '5' }), step('units', { style: 'real' })];
    expect(isSectionSorted(steps)).toBe(false);
    const sorted = sortStepsBySection(steps);
    expect(sorted.map(s => s.defId)).toEqual(['units', 'run']);
    expect(isSectionSorted(sorted)).toBe(true);
    const text = generateScript(model(sorted)).text;
    expect(text.indexOf('units real')).toBeLessThan(text.indexOf('run 5'));
  });

  it('writes a section banner each time the section changes', () => {
    const m = model([
      step('units', { style: 'lj' }),
      step('fix_nve'),
      step('run', { steps: '10' }),
    ]);
    const text = generateScript(m).text;
    expect(text).toContain('# ---- 1 · Simulation setup ----');
    expect(text).toContain('# ---- 5 · Run control ----');
  });

  it('skips disabled steps silently (flowchart shows them dashed)', () => {
    const s = step('fix_nve');
    s.enabled = false;
    const out = generateScript(model([s]));
    expect(out.text).not.toContain('fix integrate');
    expect(out.warnings).toHaveLength(0);
  });

  it('warns and skips when required params are blank', () => {
    const s = step('read_data', { file: '' });
    const out = generateScript(model([s]));
    expect(out.text).not.toContain('read_data');
    expect(out.warnings.join(' ')).toMatch(/missing/);
  });

  it('renders optional-only commands as no-ops without warnings', () => {
    const out = generateScript(model([step('min_modify')]));
    expect(out.emitted).toHaveLength(0);
    expect(out.warnings).toHaveLength(0);
  });

  it('derives a linear flowchart with start/end', () => {
    const m = model([step('units'), step('read_data'), step('fix_npt')]);
    const g = deriveFlowchart(m);
    expect(g.nodes).toHaveLength(3);
    expect(g.edges).toEqual([
      { from: 'units-1', to: 'read_data-1', fork: false },
      { from: 'read_data-1', to: 'fix_npt-1', fork: false },
    ]);
    expect(g.start).toBe(true);
    expect(g.end).toBe(true);
  });

  it('manual override emits text verbatim and warns', () => {
    const raw = '# my hand-written script\nunits real\nrun 5';
    const out = generateScript({ title: 'T', steps: [step('units')], manualText: raw });
    expect(out.text).toBe(raw);
    expect(out.emitted).toHaveLength(0);
    expect(out.warnings.join(' ')).toMatch(/manual edit mode/i);
  });

  it('empty-string manual override falls back to the builder', () => {
    const m = model([step('units', { style: 'metal' })]);
    m.manualText = '   ';
    const out = generateScript(m);
    expect(out.text).toContain('units metal');
    expect(out.warnings).toHaveLength(0);
  });
});
