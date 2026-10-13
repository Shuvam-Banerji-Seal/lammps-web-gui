import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * atom_style rheo and the RHEO styles (src/engine-gpl/rheo): the two-atom periodic virial that fixes the ghost-pair
 * rule of pair rheo, the per-atom set keywords with write_data, and refusals that name the unsupported option.
 * The full parity cases are tests/oracle/w96rheo_*.in (engineOracle.test.ts).
 */

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => files.set(n, t) }, undefined, 0, 1);
  await session.execute(text);
  const rows: ThermoRow[] = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, files };
};

describe('atom_style rheo and the RHEO styles', () => {
  it('pair rheo: the two-atom periodic virial matches native (measured with native LAMMPS, black box: 0.00307996581884303)', async () => {
    const { rows } = await run(`
units lj
dimension 2
atom_style rheo
boundary p p p
comm_modify vel yes
region box block 0 4 0 4 -0.01 0.01
create_box 1 box
create_atoms 1 single 0.3 0.5 0
create_atoms 1 single 3.4 0.7 0
mass * 1.0
set atom 1 rheo/rho 1.0
set atom 2 rheo/rho 1.6
set atom 1 vx 0.3 vy 0.1
set atom 2 vx -0.2 vy 0.4
set atom 1 rheo/status 0
set atom 2 rheo/status 0
compute pvp all pressure NULL pair
fix 1 all rheo 3.0 quintic 0
fix 2 all rheo/viscosity * constant 0.1
fix 3 all rheo/pressure * linear
fix 4 all enforce2d
pair_style rheo 3.0 artificial/visc 1.0 rho/damp 0.15
pair_coeff * *
thermo_style custom step c_pvp
thermo_modify format float %.15g
run 0
`);
    const want = 0.00307996581884303;
    expect(Math.abs(Number(rows[0].c_pvp) - want)).toBeLessThanOrEqual(1e-12 * Math.abs(want));
  });

  it('set rheo/rho and rheo/status set each selected atom; write_data writes the rheo columns', async () => {
    const { files } = await run(`
units lj
dimension 2
atom_style rheo
region box block 0 4 0 4 -0.01 0.01
create_box 1 box
create_atoms 1 single 1.0 1.0 0
create_atoms 1 single 2.0 1.0 0
set atom 2 rheo/rho 2.5
set atom 2 rheo/status 1
write_data rheo.data
`);
    const data = [...files.values()].join('\n');
    expect(data).toMatch(/1 1 0 1 1 1 0/);
    expect(data).toMatch(/2 1 1 2\.5 2 1 0/);
  });

  it('refuses an unsupported kernel and an unsupported keyword by name', async () => {
    await expect(run(`
units lj
dimension 2
atom_style rheo
region box block 0 4 0 4 -0.01 0.01
create_box 1 box
create_atoms 1 single 1.0 1.0 0
fix 1 all rheo 3.0 wendland/c4 0
`)).rejects.toThrow(/wendland\/c4/);
    await expect(run(`
units lj
dimension 2
atom_style rheo
region box block 0 4 0 4 -0.01 0.01
create_box 1 box
create_atoms 1 single 1.0 1.0 0
fix 1 all rheo 3.0 quintic 0 thermal
`)).rejects.toThrow(/thermal/);
  });

  it('fix rheo RK1 runs with newton on and with newton off (the native-order virial of owned pairs)', async () => {
    const base = `
units lj
dimension 2
atom_style rheo
boundary p p p
comm_modify vel yes
region box block 0 4 0 4 -0.01 0.01
create_box 1 box
lattice sq 1.0
create_atoms 1 box
mass * 1.0
set group all rheo/rho 1.0
set group all rheo/status 0
timestep 0.05
pair_style rheo 3.0 artificial/visc 1 rho/damp 0.3
pair_coeff * *
fix 1 all rheo 3.0 RK1 8 shift
fix 2 all rheo/viscosity * constant 0.05
fix 3 all rheo/pressure * linear
fix 4 all enforce2d
thermo 1
thermo_style custom step ke press
run 2
`;
    for (const newton of ['on', 'off']) {
      const { rows } = await run(base.replace('boundary p p p', `boundary p p p\nnewton ${newton}`));
      expect(rows.length).toBe(3);
      expect(Number.isFinite(rows[2].press as number)).toBe(true);
    }
  });
});
