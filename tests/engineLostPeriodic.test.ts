import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * An atom that moves more than a box length between two reneighborings leaves a periodic box for good: as native LAMMPS
 * does (src/domain.cpp Domain::pbc and src/comm_brick.cpp CommBrick::exchange at b332f28), the wrap moves it by one period
 * and the exchange drops it, under thermo_modify lost (error by default, warn, ignore). The oracle case
 * tests/oracle/w720lost_periodic.in compares the warn run with native.
 */
const INPUT = (lost: string) => `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 87287 loop geom
set atom 7 vx 2000.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
fix 1 all nve
thermo 1
thermo_style custom step atoms
${lost ? `thermo_modify lost ${lost}` : ''}
run 3
`;

const run = async (lost: string) => {
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); else if (ev.kind === 'log') logs.push(ev.text); },
    writeFile: () => {},
  });
  let error = '';
  try { await s.execute(INPUT(lost)); } catch (e) { error = (e as Error).message; }
  return { rows, logs, error };
};

describe('atoms lost through a periodic boundary', () => {
  it('the default stops the run with native\'s message at the first reneighboring', async () => {
    const r = await run('');
    expect(r.error).toMatch(/Lost atoms: original 108 current 107/);
  });
  it('warn drops the atom and says so; ignore drops it silently', async () => {
    const w = await run('warn');
    expect(w.error).toBe('');
    expect(w.rows.map((row) => Number(row.atoms))).toEqual([108, 107, 107, 107]);
    expect(w.logs.some((l) => /Lost atoms: original 108 current 107/.test(l))).toBe(true);
    const i = await run('ignore');
    expect(i.error).toBe('');
    expect(i.rows.map((row) => Number(row.atoms))).toEqual([108, 107, 107, 107]);
    expect(i.logs.some((l) => /Lost atoms/.test(l))).toBe(false);
  });
});
