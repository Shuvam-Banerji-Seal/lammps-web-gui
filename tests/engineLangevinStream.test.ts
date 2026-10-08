import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix langevin random stream. Expected forces were measured with native
 * LAMMPS (black box, not from its source): five atoms of types 1,2,3,1,2 at
 * (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0), masses 1, 2, 0.5, pair_style zero,
 * `fix 1 all langevin 1.0 1.0 1.0 12345`, no velocities, dt 0.005, no
 * integrator. Forces are (uniform() - 0.5) * sqrt(m 24 kT / (dt damp)) in
 * storage order, three draws per atom, one warm-up draw before the first
 * force evaluation, and each force evaluation (run setup or step) takes the
 * next block of 15 draws.
 */

const PRE = `units lj
atom_style atomic
atom_modify map array sort 0 0.0
region box block -50 50 -50 50 -50 50
create_box 3 box
mass * 1.0
mass 2 2.0
mass 3 0.5
create_atoms 1 single 0 0 0 units box
create_atoms 2 single 1 0 0 units box
create_atoms 3 single 2 0 0 units box
create_atoms 1 single 3 0 0 units box
create_atoms 2 single 4 0 0 units box
pair_style zero 1.0
pair_coeff * *
`;


describe('fix langevin random stream (native LAMMPS measurements)', () => {
  it('step-0 forces follow the RANMAR stream with one warm-up draw, in storage order', async () => {
    const files = new Map<string, string>();
    const session = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, (files.get(n) ?? '') + t) });
    await session.execute(`${PRE}fix 1 all langevin 1.0 1.0 1.0 12345
run 0
write_dump all custom langevin_probe.dump id type fx fy fz modify format float %.15g sort id
`);
    const lines = files.get('langevin_probe.dump')!.trim().split('\n');
    const k0 = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    const rows = lines.slice(k0 + 1).map((l) => l.trim().split(/\s+/).map(Number));
    // native: id 1..5 forces (sorted by id)
    const native = [
      [25.6311023415441, -33.9186868610869, -14.3020398348554],
      [-46.9672201135172, -41.4767369863311, -19.3355970518805],
      [7.06670079605927, 4.97533915879185, -5.39775207364108],
      [3.70015055823202, -31.4367727028336, -9.19514804899281],
      [7.56751038961162, 12.3777691761671, -8.07629455592687],
    ];
    expect(rows.length).toBe(5);
    rows.forEach((r, i) => {
      for (let d = 0; d < 3; d++) expect(Math.abs(r[2 + d] - native[i][d]) / Math.abs(native[i][d])).toBeLessThan(1e-10);
    });
  });
});
