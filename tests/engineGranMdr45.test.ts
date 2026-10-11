import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { parseGranularSpec } from '../src/engine/force/pair/granular';

/*
 * GRANULAR synchronized_verlet and the mdr adhesion limit (docs.lammps.org/pair_granular.html, synchronized_verlet and
 * the mdr normal model). The parity runs are tests/oracle/w45gm_*.in (engineOracle.test.ts). The numbers below come from
 * black-box runs of two spheres (diameter 1, density 1, lj units) with hooke k_n 1000, e 0.5, k_t 300 and x_gamma_t 0,
 * timestep 0.01, one step of the pair (tests/oracle/w45gm_sync_hooke_steps.in).
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

/** Two spheres with the given pair_coeff tail; spin and tangential velocity as in the oracle case. */
function twoSpheres(pairCoeff: string, dt = 0.01, steps = 1): string {
  return [
    'units lj',
    'atom_style sphere',
    'atom_modify map array',
    'comm_modify vel yes',
    'newton off',
    'boundary f f f',
    'region b block -10 10 -10 10 -10 10',
    'create_box 1 b',
    'create_atoms 1 single 0.0 0.0 0.0',
    'create_atoms 1 single 0.9 0.05 0.0',
    'set type 1 diameter 1.0 density 1.0',
    'velocity all set 0.0 0.0 0.0 units box',
    'set atom 1 vx 0.3 vy 0.2 vz 0.05',
    'set atom 2 vx -0.1 vy 0.1 vz -0.02',
    'set atom 1 omega 0.4 -0.3 0.2',
    'set atom 2 omega -0.2 0.5 0.1',
    `pair_style granular`,
    `pair_coeff * * ${pairCoeff}`,
    `timestep ${dt}`,
    'fix 1 all nve/sphere',
    'thermo_style custom step ke',
    `thermo 1`,
    `run ${steps}`,
  ].join('\n');
}

describe('pair_style granular synchronized_verlet (keyword parsing)', () => {
  it('sets the flag on the pair_coeff and keeps it off otherwise', () => {
    const on = parseGranularSpec(['hooke', '1000.0', '0.5', 'tangential', 'linear_history', '300.0', '0.0', '0.4', 'damping', 'coeff_restitution', 'synchronized_verlet']);
    expect(on.sync).toBe(true);
    const off = parseGranularSpec(['hooke', '1000.0', '0.5', 'tangential', 'linear_history', '300.0', '0.0', '0.4', 'damping', 'coeff_restitution']);
    expect(off.sync).toBe(false);
  });

  it('refuses synchronized_verlet with mdr, rolling, twisting and mindlin', async () => {
    const mdr = await run(twoSpheres('mdr 1e6 0.3 1e5 0.0 0.5 0.2 damping mdr 1 tangential linear_history 300.0 0.0 0.4 synchronized_verlet'));
    expect(mdr.error).toMatch(/synchronized_verlet with normal model 'mdr'/);
    const roll = await run(twoSpheres('hooke 1000.0 0.5 tangential linear_history 300.0 0.0 0.4 damping coeff_restitution rolling sds 100.0 0.0 0.1 synchronized_verlet'));
    expect(roll.error).toMatch(/synchronized_verlet with rolling or twisting/);
    const mind = await run(twoSpheres('hertz 1000.0 0.5 tangential mindlin 300.0 0.0 0.4 damping coeff_restitution synchronized_verlet'));
    expect(mind.error).toMatch(/synchronized_verlet with tangential mindlin/);
  });
});

describe('pair_style granular synchronized_verlet (native values)', () => {
  it('step-1 kinetic energy matches the black-box run (standard and synchronized)', async () => {
    // Measured with native LAMMPS (black box): KinEng at step 1 is 0.56683778490988 with synchronized_verlet and
    // 0.56685646... without it (the flag changes the step-1 result by about 2e-5 relative).
    const sync = await run(twoSpheres('hooke 1000.0 0.5 tangential linear_history 300.0 0.0 0.4 damping coeff_restitution synchronized_verlet'));
    expect(sync.error).toBeNull();
    expect(sync.rows[1].ke).toBeCloseTo(0.56683778490988, 9);
    const std = await run(twoSpheres('hooke 1000.0 0.5 tangential linear_history 300.0 0.0 0.4 damping coeff_restitution'));
    expect(std.rows[1].ke).not.toBeCloseTo(sync.rows[1].ke, 6);
  });
});

describe('pair_style granular mdr surface energy (refusals kept)', () => {
  it('refuses an adhesive mdr contact until it is implemented', async () => {
    const r = await run(twoSpheres('mdr 1e6 0.3 1e5 0.5 0.5 0.2 damping mdr 1 tangential linear_history 300.0 0.0 0.4'));
    expect(r.error).toMatch(/adhesive mdr/);
  });
});
