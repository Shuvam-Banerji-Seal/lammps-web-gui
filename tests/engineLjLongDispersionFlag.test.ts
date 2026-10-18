import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * The dispersion flag of the pair and kspace styles (Pair.dispersionFlag, KSpace.dispersion; native
 * KSpace::pair_check, src/kspace.cpp at b332f28) and pair_style lj/long/coul/long flag_lj off. Each row below was
 * run with native LAMMPS (b332f28) on the same input: an accepted combination ran, a refused one stopped with
 * "KSpace style X is incompatible with Pair style lj/long/coul/long". Native parity of the accepted runs is
 * tests/oracle/w252ll_*.
 */

const BASE = (flagLJ: string, kspace: string) => `units real
atom_style charge
lattice sc 3.0
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 4321
set type 1 charge 1.0
set type 2 charge -1.0
mass * 20.0
displace_atoms all random 0.2 0.2 0.2 777
pair_style lj/long/coul/long ${flagLJ} long 7.0
pair_coeff * * 0.1 2.5
kspace_style ${kspace} 1e-5
kspace_modify gewald 0.3${kspace.startsWith('pppm') ? ' mesh 10 10 10' : ''}
thermo_style custom step pe evdwl
run 0
`;

async function run(script: string) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  try {
    await session.execute(script);
  } catch {
    // the failure is reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { error: err && 'message' in err ? String(err.message) : undefined, rows };
}

describe('kspace dispersion flag (native pair_check)', () => {
  const cases: [string, string, boolean][] = [
    ['off', 'ewald', false], ['cut', 'ewald', true], ['long', 'ewald', false],
    ['off', 'pppm', false], ['cut', 'pppm', true], ['long', 'pppm', false],
    ['off', 'msm', false], ['long', 'msm', false],
    ['off', 'ewald/disp', false], ['cut', 'ewald/disp', true], ['long', 'ewald/disp', true],
    ['off', 'pppm/disp', true], ['cut', 'pppm/disp', false],
  ];
  it.each(cases)('flag_lj %s with kspace_style %s: accepted %s, as native', async (flagLJ, kspace, accepted) => {
    const r = await run(BASE(flagLJ, kspace));
    if (accepted) expect(r.error, `${flagLJ} ${kspace}`).toBeUndefined();
    else expect(r.error).toMatch(new RegExp(`KSpace style ${kspace.replace('/', '\\/')} is incompatible with Pair style lj/long/coul/long`));
  }, 60000);

  it('flag_lj off computes the LJ terms of flag_lj cut (measured with native LAMMPS: the same step-0 evdwl)', async () => {
    const off = await run(BASE('off', 'pppm/disp'));
    const cut = await run(BASE('cut', 'ewald'));
    expect(off.rows[0].evdwl).toBe(cut.rows[0].evdwl);
    expect(off.rows[0].evdwl as number).toBeCloseTo(5.00207487626398, 12);
  }, 60000);
});
