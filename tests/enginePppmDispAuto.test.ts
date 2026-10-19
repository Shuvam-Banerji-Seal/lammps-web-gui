import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Unit tests of kspace_style pppm/disp (src/engine-gpl/pppm_disp/kspace.ts): the automatic Coulomb G-ewald and grid
 * (set_grid_global, adjust_gewald), the buck/long/coul/long pair, the refusals and their native messages, and the
 * box-change path (native PPPMDisp::setup). The native parity cases are tests/oracle/w253pd_*.in (engineOracle.test.ts).
 */

const runScript = async (script: string): Promise<{ log: string }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  await session.execute(script);
  const log = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text).join('\n');
  return { log };
};

/** 108 fcc atoms, charges +-0.5 alternating by id (the w253pd_coul_auto setup without displacement). */
const FCC108 = `units lj
atom_style charge
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
mass * 1.0
group odd id 1:108:2
set group odd type 2
set type 1 charge 0.5
set type 2 charge -0.5
velocity all create 1.44 87287 loop geom
`;

const numberAfter = (log: string, label: string): number => {
  const m = new RegExp(`${label}\\s*=\\s*([-+0-9.eE]+)`).exec(log);
  if (!m) throw new Error(`no "${label}" in the log`);
  return Number(m[1]);
};

const gridAfter = (log: string, label: string): number[] => {
  const m = new RegExp(`${label}\\s*=\\s*(\\d+) (\\d+) (\\d+)`).exec(log);
  if (!m) throw new Error(`no "${label}" in the log`);
  return [Number(m[1]), Number(m[2]), Number(m[3])];
};

describe('kspace_style pppm/disp: automatic Coulomb G-ewald and grid', () => {
  it('chooses the native G-ewald and mesh for 108 charged fcc atoms at accuracy 1e-4 (Coulomb and dispersion auto)', async () => {
    const { log } = await runScript(`${FCC108}pair_style lj/long/coul/long long long 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
kspace_style pppm/disp 1.0e-4
kspace_modify disp/auto yes
run 0
`);
    // Measured with native LAMMPS: Coulomb G vector (1/distance)= 1.164842158812349, Coulomb grid = 15 15 15,
    // Dispersion G vector (1/distance)= 1.451546425444761, Dispersion grid = 36 36 36 (b332f28, same input)
    expect(numberAfter(log, 'Coulomb G vector \\(1/distance\\)')).toBeCloseTo(1.164842158812349, 12);
    expect(gridAfter(log, 'Coulomb grid')).toEqual([15, 15, 15]);
    expect(numberAfter(log, 'Dispersion G vector \\(1/distance\\)')).toBeCloseTo(1.451546425444761, 9);
    expect(gridAfter(log, 'Dispersion grid')).toEqual([36, 36, 36]);
  }, 180000);

  it('buck/long/coul/long: the dispersion G-ewald and mesh from the geometric B = sqrt(C_ii)', async () => {
    const { log } = await runScript(`${FCC108}pair_style buck/long/coul/long long long 2.5
pair_coeff 1 1 10.0 0.35 1.0
pair_coeff 1 2 8.0 0.35 0.6
pair_coeff 2 2 6.0 0.40 0.5
kspace_style pppm/disp 1.0e-4
kspace_modify disp/auto yes
run 0
`);
    // Measured with native LAMMPS (same buck input): Coulomb G 1.164842158812349 on 15 15 15; Dispersion G vector
    // (1/distance)= 1.340130406696288, Dispersion grid = 24 24 24
    expect(numberAfter(log, 'Coulomb G vector \\(1/distance\\)')).toBeCloseTo(1.164842158812349, 12);
    expect(numberAfter(log, 'Dispersion G vector \\(1/distance\\)')).toBeCloseTo(1.340130406696288, 9);
    expect(gridAfter(log, 'Dispersion grid')).toEqual([24, 24, 24]);
  }, 180000);

  it('keeps the user gewald and mesh, and takes the automatic mesh with a user gewald', async () => {
    const { log } = await runScript(`${FCC108}pair_style lj/long/coul/long long long 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
kspace_style pppm/disp 1.0e-4
kspace_modify gewald 0.5 mesh 12 12 12 gewald/disp 0.5 mesh/disp 12 12 12
run 0
`);
    expect(numberAfter(log, 'Coulomb G vector \\(1/distance\\)')).toBe(0.5);
    expect(gridAfter(log, 'Coulomb grid')).toEqual([12, 12, 12]);
  });
});

describe('kspace_style pppm/disp: refusals with native messages', () => {
  it('refuses a pair style without a long-range term', async () => {
    await expect(runScript(`${FCC108}pair_style lj/cut/coul/long 2.5
pair_coeff * * 1.0 1.0
kspace_style pppm/disp 1.0e-4
kspace_modify gewald 0.5 mesh 12 12 12
run 0
`)).rejects.toThrow(/KSpace style pppm\/disp is incompatible with Pair style lj\/cut\/coul\/long/); // native KSpace::pair_check (the pair has no dispersion term)
  });

  it('refuses buck/long with arithmetic mixing when the C matrix splits (no epsilon and sigma, native init_coeffs)', async () => {
    await expect(runScript(`${FCC108}pair_style buck/long/coul/long long long 2.5
pair_modify mix arithmetic
pair_coeff 1 1 10.0 0.35 1.0
pair_coeff 1 2 8.0 0.35 0.6
pair_coeff 2 2 6.0 0.40 0.5
kspace_style pppm/disp 1.0e-4
kspace_modify disp/auto yes
run 0
`)).rejects.toThrow(/Epsilon or sigma reference not set by pair style for PPPMDisp/);
  });

  it('refuses a dispersion without any parameter (native message)', async () => {
    await expect(runScript(`${FCC108}pair_style lj/long/coul/long long long 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
kspace_style pppm/disp 1.0e-4
kspace_modify gewald 0.5 mesh 12 12 12
run 0
`)).rejects.toThrow(/PPPMDisp used but no parameters set/);
  });
});

describe('kspace_style pppm/disp: box change', () => {
  it('runs under fix npt (the Green functions follow the box) and reports a finite energy', async () => {
    const { log } = await runScript(`${FCC108}pair_style lj/long/coul/long long long 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
kspace_style pppm/disp 1.0e-4
kspace_modify gewald/disp 1.451546425444761 mesh/disp 36 36 36
fix 1 all npt temp 1.44 1.44 0.5 iso 1.0 1.0 2.0
thermo_style custom step pe elong
thermo 2
run 4
`);
    expect(log).not.toMatch(/box changed after init/);
    expect(log).not.toMatch(/ERROR/);
  });
});
