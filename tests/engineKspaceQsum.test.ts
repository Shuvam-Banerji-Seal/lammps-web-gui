import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * The kspace charge sums (qsum, qsqsum) follow the current charges at every force evaluation, so the self and
 * background terms of a run in which a fix changes the charges (fix adapt atom charge) match the energy of the
 * same charges evaluated after the run. kspace_modify gewald fixes G-ewald, so the only difference between the
 * two evaluations is the charge sums themselves. The oracle cases w70qsum_* compare whole runs with native LAMMPS.
 */

const setup = [
  'units lj',
  'atom_style charge',
  'lattice sc 0.8',
  'region b block 0 3 0 3 0 3',
  'create_box 2 b',
  'create_atoms 1 box',
  'mass * 1.0',
  'set group all charge 1.0',
  'group odd id 1:27:2',
  'set group odd type 2',
  'set group odd charge -1.0',
  'variable dx atom 0.1*sin(1.3*x+0.7*y)',
  'variable dy atom 0.1*cos(0.9*y-1.1*z)',
  'variable dz atom 0.1*sin(1.7*z+0.5*x)',
  'displace_atoms all move v_dx v_dy v_dz units box',
  'timestep 0.002',
  'thermo_style custom step pe elong ecoul press',
  'thermo_modify format float %.15g',
  'thermo 50',
  'pair_style lj/cut/coul/long 2.0',
  'pair_coeff * * 0.5 1.0',
];

const run = async (lines: string[]): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  await session.execute(lines.join('\n'));
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

const cases: [string, string[]][] = [
  ['ewald neutral', ['kspace_style ewald 1e-4', 'variable qf equal ramp(0.5,1.5)', 'fix 1 all adapt 1 atom charge v_qf scale yes']],
  ['ewald non-neutral', ['kspace_style ewald 1e-4', 'variable qf equal ramp(1.0,2.0)', 'fix 1 odd adapt 1 atom charge v_qf scale no']],
  ['pppm neutral', ['kspace_style pppm 1e-4', 'variable qf equal ramp(0.5,1.5)', 'fix 1 all adapt 1 atom charge v_qf scale yes']],
  ['pppm non-neutral', ['kspace_style pppm 1e-4', 'variable qf equal ramp(1.0,2.0)', 'fix 1 odd adapt 1 atom charge v_qf scale no']],
];

describe('kspace charge sums follow the charges within a run', () => {
  for (const [name, extra] of cases) {
    it(`${name}: the last in-run energy equals the energy of the same charges evaluated after the run`, async () => {
      // G-ewald is fixed so that run 0 (which re-inits the kspace style) keeps the same g
      const kspace = [extra[0], 'kspace_modify gewald 1.2', ...extra.slice(1)];
      const rows = await run([...setup, ...kspace, 'fix 2 all nve', 'run 100']);
      const last = rows[rows.length - 1];
      // the adapt fix leaves the charges at the ramp value; unfix keeps them, and run 0 re-inits the sums
      const after = (await run([...setup, ...kspace, 'fix 2 all nve', 'run 100', 'unfix 1', 'run 0'])).at(-1)!;
      expect(last.step).toBe(100);
      expect(after.step).toBe(100);
      for (const k of ['pe', 'elong', 'ecoul', 'press'] as const) {
        expect(Math.abs(last[k] - after[k]), k).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(after[k])));
      }
    });
  }
});
