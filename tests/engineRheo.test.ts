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

  it('atom_style rheo/thermal: set sph/e writes the energy column of write_data (read_data.html: "atom-ID atom-type status rho energy x y z")', async () => {
    const { files } = await run(`
units lj
dimension 2
atom_style rheo/thermal
boundary p p p
region box block 0 4 0 4 -0.01 0.01
create_box 1 box
create_atoms 1 single 1.0 1.0 0
create_atoms 1 single 2.0 1.0 0
set atom 1 sph/e 2.5
set atom 2 rheo/status 1
write_data t.data
`);
    const lines = (files.get('t.data') ?? '').split('\n');
    const k = lines.indexOf('Atoms # atomic') >= 0 ? lines.indexOf('Atoms # atomic') : lines.findIndex((l) => l.startsWith('Atoms'));
    // the Atoms section lines come first; the Velocities section repeats the IDs
    const rows = lines.slice(k + 2).filter((l) => /^\d/.test(l)).slice(0, 2).map((l) => l.trim().split(/\s+/));
    expect(rows.length).toBe(2);
    // id type status rho energy x y z image...
    expect(Number(rows[0][4])).toBeCloseTo(2.5, 12);
    expect(Number(rows[1][2])).toBe(1);
    expect(Number(rows[1][4])).toBe(0);
  });

  it('refuses the RHEO thermal parts that the browser engine lacks, and fix rheo/thermal without its fix rheo, by name', async () => {
    await expect(run(`
units lj
dimension 2
atom_style hybrid rheo/thermal bond
region box block 0 4 0 4 -0.01 0.01
create_box 1 box bond/types 1 extra/bond/per/atom 4
create_atoms 1 single 1.0 1.0 0
bond_style hybrid bpm/spring no/such/style t/form 100
`)).rejects.toThrow(/no\/such\/style/);
    await expect(run(`
units lj
dimension 2
atom_style rheo/thermal
region box block 0 4 0 4 -0.01 0.01
create_box 1 box
create_atoms 1 single 1.0 1.0 0
mass * 1.0
pair_style rheo 3.0
pair_coeff * *
fix 4 all rheo/thermal conductivity * constant 0.1 specific/heat * constant 1.0
run 1
`)).rejects.toThrow(/Need to define fix rheo/);
  });

  it('atom_style rheo/thermal: the energy, temperature and conductivity survive write_restart / read_restart', async () => {
    const { files } = await run(`
units lj
dimension 2
atom_style rheo/thermal
boundary p p p
comm_modify vel yes
region box block 0 4 0 4 -0.01 0.01
create_box 1 box
create_atoms 1 single 1.0 1.0 0
create_atoms 1 single 2.0 1.0 0
set atom 1 sph/e 2.5
set atom 2 sph/e 0.75
set atom 2 rheo/status 1
mass * 1.0
pair_style rheo 3.0
pair_coeff * *
fix 1 all rheo 3.0 quintic 0 thermal
fix 2 all rheo/viscosity * constant 0.05
fix 3 all rheo/pressure * linear
fix 4 all rheo/thermal conductivity * constant 0.1 specific/heat * constant 1.0
run 0
write_data before.data
write_restart rt.restart
clear
read_restart rt.restart
mass * 1.0
pair_style rheo 3.0
pair_coeff * *
fix 1 all rheo 3.0 quintic 0 thermal
fix 2 all rheo/viscosity * constant 0.05
fix 3 all rheo/pressure * linear
fix 4 all rheo/thermal conductivity * constant 0.1 specific/heat * constant 1.0
run 0
write_data after.data
`);
    const atoms = (t: string | undefined) => {
      const lines = (t ?? '').split('\n');
      const k = lines.findIndex((l) => l.startsWith('Atoms'));
      return lines.slice(k + 2, k + 4);
    };
    expect(atoms(files.get('after.data'))).toEqual(atoms(files.get('before.data')));
    expect(Number(atoms(files.get('after.data'))[0].trim().split(/\s+/)[4])).toBeCloseTo(2.5, 12);
  });

  it('refuses the oxidation parts by name: rheo/shell without fix rheo/oxidation, nbond/shell without fix rheo/oxidation, add/heat variables', async () => {
    const HEAD = `
units lj
dimension 2
atom_style hybrid rheo/thermal bond
comm_modify vel yes
region box block 0 4 0 4 -0.01 0.01
create_box 1 box bond/types 1 extra/bond/per/atom 4
create_atoms 1 single 1.0 1.0 0
mass * 1.0
pair_style rheo 3.0
pair_coeff * *
fix 1 all rheo 3.0 quintic 0 thermal surface/detection coordination 22 8
fix 2 all rheo/viscosity * constant 0.05
fix 3 all rheo/pressure * linear
fix 4 all rheo/thermal conductivity * constant 0.1 specific/heat * constant 1.0
`;
    await expect(run(`${HEAD}bond_style rheo/shell t/form 10\nbond_coeff 1 0.2 0.2 0.1\nrun 1\n`)).rejects.toThrow(/Need to define fix rheo\/oxidation to use bond rheo\/shell/);
    await expect(run(`${HEAD}bond_style rheo/shell\nbond_coeff 1 0.2 0.2 0.1\n`)).rejects.toThrow(/requires the t\/form keyword/);
    await expect(run(`${HEAD}compute n all rheo/property/atom nbond/shell\nrun 1\n`)).rejects.toThrow(/Cannot request nbond\/shell without fix rheo\/oxidation/);
    await expect(run(`${HEAD}fix 5 all add/heat linear v_t 0.5\nrun 1\n`)).rejects.toThrow(/variable values/);
  });
});
