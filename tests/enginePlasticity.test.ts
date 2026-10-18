import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute plasticity/atom (PERI package, src/engine-gpl/peri/compute_plasticity_atom.ts) with pair_style peri/eps:
 * the native error texts are those of src/PERI/compute_plasticity_atom.cpp (b332f28). The values are the cumulative
 * lambdaValue of fix PERI_NEIGH; the native runs are the oracle cases tests/oracle/w240pl_*.in.
 */

const newSession = () => {
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => files.set(n, t) });
  return { session, files, events };
};

const BLOCK = `units si
boundary s s s
atom_style peri
atom_modify map array
neighbor 0.0010 bin
lattice sc 0.0005
region target block 0 6 0 6 0 6 units lattice
create_box 1 target
create_atoms 1 box
`;

const PAIR_EPS = `pair_style peri/eps
pair_coeff * * 14.9e9 14.9e9 0.0015001 0.0005 0.25 10.0e8
set group all density 2200
set group all volume 1.25e-10
`;

const SINUSOIDAL = `variable vx atom 60.0*sin(2000*y+0.3)
variable vy atom 60.0*cos(2500*z)
variable vz atom 60.0*sin(1800*x)
velocity all set v_vx v_vy v_vz units box
fix 1 all nve
timestep 1.0e-7
`;

describe('compute plasticity/atom', () => {
  it('refuses an argument (Illegal compute plasticity/atom command)', async () => {
    const { session } = newSession();
    await expect(session.execute(`${BLOCK}${PAIR_EPS}compute 1 all plasticity/atom extra\n`))
      .rejects.toThrow('Illegal compute plasticity/atom command');
  });

  it('needs pair_style peri/eps (Compute plasticity/atom cannot be used with this pair style)', async () => {
    const { session } = newSession();
    await expect(session.execute(`${BLOCK}pair_style peri/pmb
pair_coeff * * 1.6863e22 0.0015001 0.0005 0.25
compute 1 all plasticity/atom
`)).rejects.toThrow('Compute plasticity/atom cannot be used with this pair style');
  });

  it('refuses without any pair style as well', async () => {
    const { session } = newSession();
    await expect(session.execute(`${BLOCK}compute 1 all plasticity/atom\n`))
      .rejects.toThrow('Compute plasticity/atom cannot be used with this pair style');
  });

  it('is zero outside its group and keeps the cumulative value, which only grows (fix PERI_NEIGH lambdaValue)', async () => {
    const { session } = newSession();
    await session.execute(`${BLOCK}${PAIR_EPS}${SINUSOIDAL}group low id 1 2 3 4 5 6 7 8 9 10
compute 1 all plasticity/atom
compute 2 low plasticity/atom
run 20
`);
    const first = Float64Array.from((session.sys.computes.find((c) => c.id === '1') as unknown as { peratomValues(): Float64Array }).peratomValues());
    const outside = Float64Array.from((session.sys.computes.find((c) => c.id === '2') as unknown as { peratomValues(): Float64Array }).peratomValues());
    const s = session.sys.state;
    let yielded = 0;
    for (let i = 0; i < s.n; i++) {
      expect(first[i]).toBeGreaterThanOrEqual(0);
      if (first[i] > 0) yielded++;
      // the group low holds the atoms with id 1..10: every other atom reads 0 (plasticity/atom, "else plasticity[i] = 0.0")
      if (!(s.id[i] <= 10)) expect(outside[i]).toBe(0);
      else expect(outside[i]).toBe(first[i]);
    }
    // the sinusoidal start at 60 m/s yields the block (the oracle block case reaches the same state)
    expect(yielded).toBeGreaterThan(0);

    await session.execute('run 20\n');
    const second = (session.sys.computes.find((c) => c.id === '1') as unknown as { peratomValues(): Float64Array }).peratomValues();
    for (let i = 0; i < s.n; i++) expect(second[i]).toBeGreaterThanOrEqual(first[i]);
  });
});
