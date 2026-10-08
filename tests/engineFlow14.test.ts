import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/* Wave 14 flow styles: fix accelerate/cos and compute viscosity/cos (docs.lammps.org/fix_accelerate_cos.html,
 * docs.lammps.org/compute_viscosity_cos.html). Parity is in tests/oracle/w14flow_*.in. */

const run = async (script: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  await session.execute(script);
  return events;
};

const base = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

describe('w14flow: fix accelerate/cos', () => {
  it('runs with a numeric amplitude', async () => {
    await expect(run(`${base}fix 1 all nve\nfix 2 all accelerate/cos 0.05\nrun 2\n`)).resolves.toBeDefined();
  });
  it('rejects a variable amplitude', async () => {
    await expect(run(`${base}variable A equal 0.1\nfix 1 all nve\nfix 2 all accelerate/cos v_A\nrun 1\n`)).rejects.toThrow(/variables are not supported/);
  });
  it('rejects a missing amplitude', async () => {
    await expect(run(`${base}fix 1 all nve\nfix 2 all accelerate/cos\nrun 1\n`)).rejects.toThrow();
  });
});

describe('w14flow: compute viscosity/cos', () => {
  it('rejects arguments', async () => {
    await expect(run(`${base}compute c all viscosity/cos 3\nrun 0\n`)).rejects.toThrow(/takes no arguments/);
  });
  it('is usable in thermo as a 7-vector', async () => {
    const ev = await run(`${base}compute c all viscosity/cos\nthermo_style custom step c_c[7]\nrun 0\n`);
    expect(ev.some((e) => e.kind === 'thermo')).toBe(true);
  });
});
