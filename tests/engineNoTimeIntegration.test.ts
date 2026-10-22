import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Verlet::init (src/verlet.cpp at b332f28): "No fixes with time integration, atoms won't move" when no fix has
 * time_integrate set. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28): the count of that warning for
 * each input below (run_style respa and minimize do not warn; run pre no skips the init; rerun inits a Verlet run).
 */

const BASE = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

const ROWS: [string, number][] = [
  ['run 2', 1],
  ['run 0', 1],
  ['fix 1 all nve\nrun 2', 0],
  ['fix 1 all langevin 1.0 1.0 1.0 48279\nrun 2', 1],
  ['minimize 1e-4 1e-6 10 100', 0],
  ['run_style respa 2 2\nrun 2', 0],
  ['fix 1 all rigid single\nrun 2', 0],
  ['run 2\nrun 2 pre no', 1],
  ['write_dump all atom t07.dump\nrerun t07.dump dump x y z', 1],
  ['fix 1 all nve/limit 0.1\nrun 2', 0],
  ['fix 1 all move linear 0 0 0\nrun 2', 0],
  ['fix 1 all nvt temp 1 1 1\nrun 2', 0],
];

describe("No fixes with time integration, atoms won't move", () => {
  it.each(ROWS)('%s', async (tail, count) => {
    const files = new Map<string, string>();
    const warnings: string[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const s = new Session({ emit: (e: any) => { if (e.kind === 'log' && e.text.startsWith('WARNING')) warnings.push(e.text); }, writeFile: (n: string, t: string) => { files.set(n, t); } } as any);
    await s.execute(`${BASE}${tail}\n`);
    const w = warnings.filter((x) => x.includes('No fixes with time integration'));
    expect(w.length).toBe(count);
    for (const x of w) expect(x).toBe("WARNING: No fixes with time integration, atoms won't move\nFor more information see https://docs.lammps.org/err0028");
  });
});
