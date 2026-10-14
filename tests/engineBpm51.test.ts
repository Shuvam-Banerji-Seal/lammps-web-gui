import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

// BPM package styles (src/engine/registry/bpm51.ts): bond bpm/spring, pair bpm/spring (compute nbond/atom is compute/misc18.ts).
// The numbers checked here were measured with native LAMMPS (black box) on a dimer (see the oracle cases
// tests/oracle/w51bpm_*.in, which compare every thermo row and the final atom state with native LAMMPS).

const DIMER = `units lj
atom_style bond
newton on off
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 1.0 0.0 0.0
mass 1 1.0
create_bonds single/bond 1 1 2
pair_style bpm/spring
pair_coeff 1 1 1.0 1.0 0.5
comm_modify vel yes cutoff 2.0
special_bonds lj 0 1 1 coul 1 1 1
fix 1 all nve
compute nb all nbond/atom
compute tb all reduce sum c_nb
thermo_style custom step pe ebond epair c_tb
thermo_modify format float %.15g norm no
`;

interface Row { step: number; pe: number; ebond: number; epair: number; c_tb: number }

const run = async (script: string): Promise<{ rows: Row[]; errors: string[] }> => {
  const rows: Row[] = [];
  const errors: string[] = [];
  const s = new Session({
    emit: (e: { kind: string; row?: Row; message?: string }) => {
      if (e.kind === 'thermo' && e.row) rows.push(e.row);
      if (e.kind === 'error') errors.push(e.message ?? '');
    },
    writeFile: () => {},
  } as never);
  try {
    await s.execute(script);
  } catch (e) {
    errors.push(e instanceof Error ? e.message : String(e));
  }
  return { rows, errors };
};

describe('bond_style bpm/spring', () => {
  it('refuses the x/ref store/local values by name; volume/factor yes takes the av coefficient', async () => {
    const a = await run(`${DIMER}bond_style bpm/spring store/local brk 100 time x/ref\n`);
    expect(a.errors.join(' ')).toMatch(/x\/ref/);
    const b = await run(`${DIMER}bond_style bpm/spring volume/factor yes\nbond_coeff 1 1.0 0.1 1.0\n`);
    expect(b.errors.join(' ')).toMatch(/Incorrect args for bond coefficients/);
  });

  it('refuses break no with smoothing on, as native does', async () => {
    const r = await run(`${DIMER}bond_style bpm/spring break no\n`);
    expect(r.errors.join(' ')).toMatch(/turn off smoothing with break no/);
    const ok = await run(`${DIMER}bond_style bpm/spring break no smooth no\nbond_coeff 1 1.0 0.1 0.5\nrun 0\n`);
    expect(ok.errors).toEqual([]);
  });

  it('gives the reference-length energy k (r - r0)^2 / 2 and normalizes by r0 when asked', async () => {
    const r = await run(`${DIMER}bond_style bpm/spring normalize yes
bond_coeff 1 1.0 0.1 0.5
run 0
set atom 2 x 1.05
run 0
`);
    expect(r.errors).toEqual([]);
    // r0 = 1: E = 0.5 * 0.05^2 = 0.00125 (measured with native LAMMPS, black box: thermo norm no)
    expect(r.rows[1].ebond).toBeCloseTo(0.00125, 12);
    expect(r.rows[1].c_tb).toBe(2);
  });

  it('breaks a bond stretched past eps on a timestep, after which nbond/atom no longer counts it', async () => {
    const r = await run(`${DIMER}bond_style bpm/spring
bond_coeff 1 1.0 0.1 0.5
timestep 0.01
run 0
set atom 2 x 1.2
run 2
`);
    expect(r.errors).toEqual([]);
    // setup keeps the bond (force 51 at strain 0.2), the first timestep breaks it and applies no bond force
    expect(r.rows[1].c_tb).toBe(2);
    expect(r.rows[2].c_tb).toBe(0);
    expect(r.rows[2].ebond).toBe(0);
  });
});

describe('pair_style bpm/spring', () => {
  it('needs the anharmonic coefficient when anharmonic yes is set', async () => {
    const r = await run(`units lj
atom_style bond
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 0.8 0.0 0.0
mass 1 1.0
pair_style bpm/spring anharmonic yes
pair_coeff 1 1 1.0 1.0 0.5
`);
    expect(r.errors.join(' ')).toMatch(/usage: pair_coeff/);
  });

  it('repels unbonded atoms inside the cutoff with k (r - rc)^2/2 + ka (r - rc)^4/4', async () => {
    const r = await run(`units lj
atom_style bond
newton on off
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 0.8 0.0 0.0
mass 1 1.0
pair_style bpm/spring anharmonic yes
pair_coeff 1 1 1.0 1.0 0.5 2.0
comm_modify vel yes cutoff 2.0
fix 1 all nve
thermo_style custom step pe epair
thermo_modify format float %.15g norm no
run 0
`);
    expect(r.errors).toEqual([]);
    // dr = -0.2: 0.5 * 0.04 + 2 * 0.0016 / 4 = 0.0208 (measured with native LAMMPS, black box)
    expect(r.rows[0].pe).toBeCloseTo(0.0208, 12);
  });
});

describe('compute nbond/atom', () => {
  it('counts bonds per atom, honours bond/type, and refuses other keywords', async () => {
    const r = await run(`${DIMER}compute n2 all nbond/atom bond/type 2
thermo_style custom step c_tb
thermo_modify format float %.15g norm no
run 0
`);
    expect(r.errors).toEqual([]);
    expect(r.rows[0].c_tb).toBe(2);
    const bad = await run(`${DIMER}compute n3 all nbond/atom cutoff 2.0\n`);
    expect(bad.errors.join(' ')).toMatch(/Unknown compute nbond\/atom command cutoff/);
  });
});
