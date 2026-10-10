import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { parseGranularSpec, heatFlowOf } from '../src/engine/force/pair/granular';

/*
 * GRANULAR heat conduction (docs.lammps.org/pair_granular.html heat keyword, fix_property_atom.html temperature and
 * heatflow, fix_heat_flow.html, fix_wall_gran.html temperature keyword). The parity runs are tests/oracle/w38heat_*.in
 * (engineOracle.test.ts). The numbers in this file come from black-box runs of two spheres with diameter 1 (density 1
 * and 2), hooke k_n 2000, dt 1e-4, placed 0.8 apart (overlap delta 0.2, so the contact radius a = sqrt(R_eff delta)
 * = sqrt(0.25 * 0.2) = 0.2236 with R_eff = R/2).
 */

/** Runs a script; returns its thermo rows and the error message when the engine stops. */
async function run(script: string): Promise<{ rows: ThermoRow[]; error: string | null }> {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const ev = events.find((x) => x.kind === 'error');
  if (!error && ev && 'message' in ev) error = String(ev.message);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, error };
}

/** Two spheres (types 1 and 2, temperatures 1 and 3) with the given pair_style lines and fix lines. */
function twoSpheres(opts: { pair?: string; fixes?: string; withProperty?: boolean; extra?: string; steps?: number; sep?: number } = {}): string {
  const sep = opts.sep ?? 0.8;
  const withProp = opts.withProperty !== false;
  return [
    'units lj',
    'atom_style sphere',
    'boundary f f f',
    'region b block -10 10 -10 10 -10 10',
    'create_box 2 b',
    'create_atoms 1 single 0 0 0',
    `create_atoms 2 single ${sep} 0 0`,
    'set atom 1 diameter 1.0 density 1.0',
    'set atom 2 diameter 1.0 density 2.0',
    'comm_modify vel yes',
    withProp ? 'fix pa all property/atom temperature heatflow' : '',
    withProp ? 'set atom 1 temperature 1.0' : '',
    withProp ? 'set atom 2 temperature 3.0' : '',
    `pair_style granular`,
    `pair_coeff * * hooke 2000.0 0.0 tangential linear_nohistory 0.0 0.0 ${opts.pair ?? 'heat none'}`,
    'timestep 0.0001',
    'fix 1 all nve/sphere',
    opts.fixes ?? '',
    opts.extra ?? '',
    'compute pa all property/atom temperature heatflow',
    'compute hmax all reduce max c_pa[2]',
    'compute hmin all reduce min c_pa[2]',
    'compute tmin all reduce min c_pa[1]',
    'compute tmax all reduce max c_pa[1]',
    'thermo_style custom step c_hmax c_hmin c_tmin c_tmax',
    'thermo_modify format float %.15g',
    `run ${opts.steps ?? 0}`,
  ].filter((l) => l !== '').join('\n');
}

describe('pair_style granular heat keyword: parsing', () => {
  const base = ['hooke', '2000.0', '0.0', 'tangential', 'linear_nohistory', '0.0', '0.0'];

  it('reads the heat model and its coefficient', () => {
    expect(parseGranularSpec([...base, 'heat', 'area', '0.5']).heat).toBe('area');
    expect(parseGranularSpec([...base, 'heat', 'area', '0.5']).hk).toBe(0.5);
    expect(parseGranularSpec([...base, 'heat', 'radius', '2']).heat).toBe('radius');
    expect(parseGranularSpec([...base, 'heat', 'none']).heat).toBe('none');
    expect(parseGranularSpec(base).heat).toBe('none');
  });

  it('refuses a missing, unknown or negative heat model value', () => {
    expect(() => parseGranularSpec([...base, 'heat'])).toThrow(/heat/);
    expect(() => parseGranularSpec([...base, 'heat', 'area'])).toThrow(/heat area/);
    expect(() => parseGranularSpec([...base, 'heat', 'foo', '1'])).toThrow(/unknown heat model 'foo'/);
    expect(() => parseGranularSpec([...base, 'heat', 'area', '-0.5'])).toThrow(/must be >= 0/);
    expect(() => parseGranularSpec([...base, 'heat', 'radius', 'abc'])).toThrow();
  });

  it('heatFlowOf: radius Q = 2 k a dT, area Q = h pi a^2 dT, none Q = 0', () => {
    expect(heatFlowOf('radius', 0.5, Math.sqrt(0.05), 2)).toBeCloseTo(2 * 0.5 * Math.sqrt(0.05) * 2, 15);
    expect(heatFlowOf('area', 0.5, Math.sqrt(0.05), 2)).toBeCloseTo(0.5 * Math.PI * 0.05 * 2, 15);
    expect(heatFlowOf('none', 1, 1, 1)).toBe(0);
  });
});

describe('heat conduction between two spheres', () => {
  it('conducted heat matches h_s pi a^2 dT (area) and 2 k_s a dT (radius)', async () => {
    // measured with native LAMMPS (black box): atom 1 (T 1) against atom 2 (T 3): Q_1 = +0.15708 for area 0.5
    // and +0.44721 for radius 0.5; atom 2 gets the opposite sign
    const a = Math.sqrt(0.25 * 0.2);
    const area = await run(twoSpheres({ pair: 'heat area 0.5' }));
    expect(area.error).toBeNull();
    expect(area.rows[0].c_hmax as number).toBeCloseTo(0.5 * Math.PI * a * a * 2, 9);
    expect(area.rows[0].c_hmin as number).toBeCloseTo(-0.5 * Math.PI * a * a * 2, 9);
    const radius = await run(twoSpheres({ pair: 'heat radius 0.5' }));
    expect(radius.error).toBeNull();
    expect(radius.rows[0].c_hmax as number).toBeCloseTo(2 * 0.5 * a * 2, 9);
  });

  it('no heat without the heat keyword', async () => {
    const r = await run(twoSpheres({ pair: 'heat none', withProperty: true }));
    expect(r.error).toBeNull();
    expect(r.rows[0].c_hmax).toBe(0);
  });

  it('the temperature change per step is dt Q / (m cp) with the Q of that step', async () => {
    // measured with native LAMMPS (black box): T_1 after one step = 1 + dt Q_1 / m_1 with m_1 = 4/3 pi 0.5^3 (density 1)
    // and Q_1 the heat flow of the same step (the update follows the force evaluation)
    const m1 = (4 / 3) * Math.PI * 0.125;
    const r = await run(twoSpheres({ pair: 'heat area 0.5', fixes: 'fix 2 all heat/flow constant 1.0', steps: 1 }));
    expect(r.error).toBeNull();
    const last = r.rows[r.rows.length - 1];
    const Q1 = last.c_hmax as number;
    expect((last.c_tmin as number) - 1).toBeCloseTo((1e-4 * Q1) / m1, 12);
  });

  it('the heat flow is zeroed by fix heat/flow before each step, so it does not accumulate', async () => {
    // measured with native LAMMPS (black box): without fix heat/flow the step-1 heat flow is the setup value plus the step value
    const withFix = await run(twoSpheres({ pair: 'heat area 0.5', fixes: 'fix 2 all heat/flow constant 1.0', steps: 1 }));
    const without = await run(twoSpheres({ pair: 'heat area 0.5', steps: 1 }));
    const q0 = 0.5 * Math.PI * 0.05 * 2;
    expect(withFix.error).toBeNull();
    expect(without.error).toBeNull();
    expect(withFix.rows[1].c_hmax as number).toBeLessThan(q0 * 1.01);
    expect(without.rows[1].c_hmax as number).toBeGreaterThan(q0 * 1.5);
  });
});

describe('heat conduction: bad input is refused', () => {
  it('a heat model needs the temperature and heatflow properties', async () => {
    const r = await run(twoSpheres({ pair: 'heat area 0.5', withProperty: false }));
    expect(r.error).toMatch(/temperature heatflow/);
  });

  it('set temperature refuses a negative value and accepts zero', async () => {
    const neg = await run(twoSpheres({ withProperty: true, extra: 'set atom 1 temperature -1' }));
    expect(neg.error).toMatch(/Invalid temperature -1 in set command/);
    const zero = await run(twoSpheres({ withProperty: true, extra: 'set atom 1 temperature 0' }));
    expect(zero.error).toBeNull();
  });

  it('fix heat/flow refuses a negative cp, a wrong number of type values and unknown styles', async () => {
    expect((await run(twoSpheres({ fixes: 'fix 2 all heat/flow constant -1.0' }))).error).toMatch(/cp must be >= 0/);
    expect((await run(twoSpheres({ fixes: 'fix 2 all heat/flow type 1.0' }))).error).toMatch(/values for 2 atom types/);
    expect((await run(twoSpheres({ fixes: 'fix 2 all heat/flow bogus 1.0' }))).error).toMatch(/unknown style 'bogus'/);
    expect((await run(twoSpheres({ fixes: 'fix 2 all heat/flow constant' }))).error).toMatch(/heat\/flow constant/);
  });

  it('a fix wall/gran heat model needs the temperature keyword, and temperature needs a heat model', async () => {
    // fix_wall_gran.html: the keyword temperature follows the wallstyle arguments; the heat model is in the granular part
    const wall = (heat: string, kw: string) => twoSpheres({
      extra: `fix w all wall/gran granular hooke 2000.0 0.0 tangential linear_nohistory 0.0 0.0 ${heat} zplane -5.0 NULL ${kw}`,
    });
    expect((await run(wall('heat area 0.5', ''))).error).toMatch(/needs the temperature keyword/);
    expect((await run(wall('heat area 0.5', 'temperature 2.0'))).error).toBeNull();
    expect((await run(wall('heat none', 'temperature 2.0'))).error).toMatch(/needs a heat model/);
  });

  it('compute property/atom temperature needs fix property/atom temperature heatflow', async () => {
    const r = await run(twoSpheres({ withProperty: false, extra: 'compute q all property/atom temperature' }));
    expect(r.error).toMatch(/needs fix property\/atom temperature heatflow/);
  });
});

describe('fix property/atom temperature and heatflow storage', () => {
  it('stores one value per atom that set and the heat models read', async () => {
    const r = await run([
      'units lj',
      'atom_style sphere',
      'region b block -10 10 -10 10 -10 10',
      'create_box 1 b',
      'create_atoms 1 single 0 0 0',
      'create_atoms 1 single 5 0 0',
      'set atom 1 diameter 1.0 density 1.0',
      'set atom 2 diameter 1.0 density 1.0',
      'fix pa all property/atom temperature heatflow',
      'set atom 1 temperature 1.5',
      'set atom 2 temperature 2.5',
      'compute pa all property/atom temperature heatflow',
      'compute s all reduce sum c_pa[1]',
      'thermo_style custom step c_s',
      'thermo_modify format float %.15g',
      'run 0',
    ].join('\n'));
    expect(r.error).toBeNull();
    // thermo in lj units normalizes an extensive reduce sum by the atom count (thermo_modify norm): (1.5 + 2.5) / 2
    expect(r.rows[0].c_s as number).toBeCloseTo(2, 12);
  });

  it('a second temperature or heatflow property is refused', async () => {
    const r = await run([
      'units lj',
      'atom_style sphere',
      'region b block -10 10 -10 10 -10 10',
      'create_box 1 b',
      'create_atoms 1 single 0 0 0',
      'fix pa all property/atom temperature',
      'fix pb all property/atom temperature',
    ].join('\n'));
    expect(r.error).toMatch(/already exists/);
  });
});
