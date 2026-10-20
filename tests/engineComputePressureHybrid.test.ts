import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * compute pressure ... pair/hybrid (compute_pressure.html); numerical parity with native LAMMPS is in the oracle cases
 * w341press_pair_hybrid and w342press_hybrid_types. These tests cover the argument handling, read from
 * src/compute_pressure.cpp at b332f28.
 */

const SETUP = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
mass * 1.0
`;

const errorOf = async (script: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  try {
    await s.execute(SETUP + script);
  } catch (e) {
    return String((e as Error).message);
  }
  return null;
};

describe('compute pressure pair/hybrid', () => {
  it('needs the name of a sub-style of the hybrid pair style', async () => {
    const hyb = 'pair_style hybrid/overlay lj/cut 2.5 morse 2.0\npair_coeff * * lj/cut 1.0 1.0\npair_coeff * * morse 0.3 2.0 1.1\n';
    expect(await errorOf(`${hyb}compute p all pressure NULL pair/hybrid lj/cut\nthermo_style custom step c_p\nrun 0\n`)).toBeNull();
    expect(await errorOf(`${hyb}compute p all pressure NULL pair/hybrid soft\nthermo_style custom step c_p\nrun 0\n`))
      .toMatch(/Unrecognized pair style soft in compute pressure command/);
    // native reads no sub-style index: the number is an unknown keyword
    expect(await errorOf(`${hyb}compute p all pressure NULL pair/hybrid lj/cut 1\n`)).toMatch(/Unknown compute pressure keyword 1/);
    expect(await errorOf(`${hyb}compute p all pressure NULL pair/hybrid\n`)).toMatch(/needs a pair sub-style name/);
  });

  it('refuses a sub-style named twice (native pair_match finds no single match) and a non-hybrid pair style', async () => {
    const two = 'pair_style hybrid lj/cut 2.5 lj/cut 2.0\npair_coeff 1 1 lj/cut 1 1.0 1.0\npair_coeff 2 2 lj/cut 2 1.0 1.0\npair_coeff 1 2 lj/cut 1 1.0 1.0\n';
    expect(await errorOf(`${two}compute p all pressure NULL pair/hybrid lj/cut\nthermo_style custom step c_p\nrun 0\n`))
      .toMatch(/Unrecognized pair style lj\/cut in compute pressure command/);
    expect(await errorOf('pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\ncompute p all pressure NULL pair/hybrid lj/cut\nthermo_style custom step c_p\nrun 0\n'))
      .toMatch(/the pair style is not hybrid/);
  });
});
