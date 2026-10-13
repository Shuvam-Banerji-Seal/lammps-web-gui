import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * kspace_modify compute: kspace_modify.html: "The compute keyword allows Kspace computations to be turned off, even
 * though a kspace_style is defined." The kspace setup still runs; the kspace force, energy (elong) and virial are not
 * computed. Measured with native LAMMPS (black box, tests/oracle/w84lang_kspace_compute_no.in): with compute no the
 * elong column is 0 and the potential energy is the pair part alone. The numbers are checked against native by the
 * w84lang_kspace_compute_no oracle case.
 */

const SETUP = `units lj
atom_style charge
atom_modify map array sort 0 0.0
lattice sc 0.7
region box block 0 2 0 3 0 3
create_box 1 box
create_atoms 1 box
group neg id 1:17:2
group pos id 2:18:2
set group neg charge -1.0
set group pos charge 1.0
mass * 1.0
velocity all create 0.5 4928 loop all
pair_style lj/cut/coul/long 2.5
pair_coeff * * 1.0 1.0
pair_modify table 0
kspace_style pppm 1e-4
`;

const run = async (extra: string): Promise<{ rows: ThermoRow[]; errors: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  try {
    await session.execute(`${SETUP}${extra}\nthermo_style custom step pe elong press\nrun 0\n`);
  } catch (e) {
    return { rows: [], errors: [String(e)] };
  }
  const rows = events.flatMap((e) => (e.kind === 'thermo' ? [e.row] : []));
  const errors = events.flatMap((e) => (e.kind === 'error' ? [e.message] : []));
  return { rows, errors };
};

describe('kspace_modify compute', () => {
  it('compute no drops the kspace energy and virial but keeps the pair part', async () => {
    const on = await run('');
    const off = await run('kspace_modify compute no\n');
    expect(on.errors).toEqual([]);
    expect(off.errors).toEqual([]);
    expect(on.rows[0].elong).not.toBe(0);
    expect(off.rows[0].elong).toBe(0);
    // the potential energy without kspace is the pair energy: pe(on) - elong(on)
    expect(off.rows[0].pe).toBeCloseTo(on.rows[0].pe - on.rows[0].elong, 9);
    expect(off.rows[0].press).not.toBe(on.rows[0].press);
  });

  it('compute yes restores the kspace terms', async () => {
    const on = await run('');
    const back = await run('kspace_modify compute no\nkspace_modify compute yes\n');
    expect(back.rows[0].elong).toBe(on.rows[0].elong);
    expect(back.rows[0].pe).toBe(on.rows[0].pe);
  });

  it('refuses a keyword value that is not yes or no', async () => {
    const r = await run('kspace_modify compute maybe\n');
    expect(r.errors.join('\n')).toMatch(/compute/);
  });
});
