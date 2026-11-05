import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * kspace_style pppm with an atom whose coordinate is not finite: native LAMMPS (src/KSPACE/pppm.cpp PPPM::particle_map
 * at b332f28) maps every atom to its grid point with a cast of (x - boxlo) / h, which a NaN or an infinity turns into an
 * index outside every grid brick, and stops with "Out of range atoms - cannot compute PPPM". Measured with native
 * LAMMPS (/home/roy/.local/bin/lmp, b332f28) on the input below: an infinite force (1e308*10) on one atom between
 * neighbor builds prints the step-0 row and then stops with that message. At a run's setup the periodic remap comes
 * first and stops with "Non-numeric atom coords - simulation unstable" (Domain::pbc).
 */

const BASE = `units real
atom_style charge
region box block 0 12 0 12 0 12
create_box 2 box
create_atoms 1 single 3 3 3
create_atoms 2 single 8 8 8
create_atoms 1 single 3 9 6
create_atoms 2 single 9 3 6
mass * 16.0
set type 1 charge 0.5
set type 2 charge -0.5
pair_style lj/cut/coul/long 5.0
pair_coeff * * 0.1 3.0
kspace_style pppm 1.0e-4
neigh_modify every 10 delay 0 check no
`;

const run = async (input: string): Promise<{ rows: number; error: string | null }> => {
  let rows = 0;
  const s = new Session({ emit: (e: { kind: string }) => { if (e.kind === 'thermo') rows++; }, writeFile: () => {} } as never);
  try {
    await s.execute(input);
    return { rows, error: null };
  } catch (e) {
    return { rows, error: (e as Error).message };
  }
};

describe('kspace_style pppm: an atom whose coordinate is not finite', () => {
  it('an infinite force between neighbor builds stops at the next step with native message', async () => {
    const r = await run(`${BASE}variable big equal 1e308*10
group one id 1
fix 1 all nve
fix 2 one addforce v_big 0.0 0.0
thermo 1
run 3
`);
    expect(r.rows).toBe(1);
    expect(r.error).toMatch(/Out of range atoms - cannot compute PPPM/);
  });

  it('a NaN coordinate at a run setup stops in the periodic remap first', async () => {
    const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
    await s.execute(`${BASE}run 0\n`);
    s.sys.state.x[4] = Number.NaN;
    await expect(s.execute('run 0\n')).rejects.toThrow(/Non-numeric atom coords - simulation unstable/);
  });

  it('finite coordinates run as before', async () => {
    const r = await run(`${BASE}fix 1 all nve\nthermo 1\nrun 3\n`);
    expect(r.error).toBeNull();
    expect(r.rows).toBe(4);
  });
});
