import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/** Relative closeness for the checks against native numbers. */
const close = (got: number, want: number, rel = 1e-10) => Math.abs(got - want) <= rel * Math.max(1, Math.abs(got), Math.abs(want));

/** Runs input text in a fresh session and returns the thermo rows and the error (if any). */
const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, error };
};

/** Two atoms along x at distance r in a large box (the real-space pair energy only). */
const twoAtoms = (r: number, extra: string) => `
units lj
atom_style atomic
region box block -15 15 -15 15 -15 15
create_box 1 box
create_atoms 1 single 0 0 0
create_atoms 1 single ${r} 0 0
mass 1 1.0
pair_style lj/long/coul/long long off 2.5
pair_modify table 0 table/disp 0
pair_coeff 1 1 1.0 1.0
${extra}
kspace_style ewald/disp 1.0e-6
kspace_modify gewald/disp 0.4
thermo_style custom step evdwl
thermo_modify format float %.15g norm no
run 0
`;

/** Fcc 2x2x2 cells, two types, geometric mixing (small: runs in milliseconds). */
const FCC = `
units lj
atom_style charge
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
mass * 1.0
set type 1 type/fraction 2 0.5 4242
pair_style lj/long/coul/long long off 2.5
pair_modify table 0 table/disp 0
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.1
kspace_style ewald/disp 1.0e-6
kspace_modify gewald/disp 0.4
thermo_style custom step elong evdwl
thermo_modify format float %.15g norm no
run 0
`;

describe('kspace_style ewald/disp and pair lj/long/coul/long flag_lj long', () => {
  it('real-space dispersion kernel matches native LAMMPS (black box, gewald/disp 0.4, table/disp 0)', async () => {
    // Measured with native LAMMPS (black box): evdwl at r = 1.6 is -0.22219283725231 and at r = 2.4 is -0.0194305071976297
    const a = await run(twoAtoms(1.6, ''));
    expect(a.error).toBeNull();
    expect(close(a.rows[0].evdwl, -0.22219283725231, 1e-11)).toBe(true);
    const b = await run(twoAtoms(2.4, ''));
    expect(close(b.rows[0].evdwl, -0.0194305071976297, 1e-11)).toBe(true);
  });

  it('mix/disp geom equals the pair C6 matrix when the pair mixing is geometric', async () => {
    // geometric pair mixing gives C6_ij = sqrt(C6_ii C6_jj), so the kspace matrix is unchanged
    const pair = await run(FCC);
    const geom = await run(FCC.replace('kspace_modify gewald/disp 0.4', 'kspace_modify gewald/disp 0.4 mix/disp geom'));
    expect(pair.error).toBeNull();
    expect(geom.error).toBeNull();
    expect(close(geom.rows[0].elong, pair.rows[0].elong, 1e-13)).toBe(true);
  });

  it('the automatic G-ewald (Newton solve of the r^-6 error, g_ewald_6 = g_ewald) matches native LAMMPS', async () => {
    // Measured with native LAMMPS: G vector = 1.6965764, elong -850.512893456264, evdwl 215.769335926041
    const r = await run(FCC.replace('kspace_modify gewald/disp 0.4', ''));
    expect(r.error).toBeNull();
    expect(close(r.rows[0].elong, -850.512893456264, 1e-12)).toBe(true);
    expect(close(r.rows[0].evdwl, 215.769335926041, 1e-12)).toBe(true);
  });

  it('refuses pair_modify table/disp N below 8 bits (native: too few total bits)', async () => {
    const r = await run(FCC.replace('pair_modify table 0 table/disp 0', 'pair_modify table 0 table/disp 4'));
    expect(r.error).toMatch(/Too few total bits for bitmapped Dispersion lookup table/);
  });

  it('runs the default dispersion table (table/disp 12) with flag_lj long', async () => {
    const r = await run(FCC.replace(' table/disp 0', ''));
    expect(r.error).toBeNull();
    expect(r.rows.length).toBeGreaterThan(0);
  });

  it('splittol, disp/auto, mesh/disp, order/disp and force/disp/* have no effect on ewald/disp; mix/disp none is refused', async () => {
    // Measured with native LAMMPS: with all of them elong -20.3347008705397 and evdwl -584.183842934727, as without;
    // mix/disp none stops with: kspace_modify mix/disp none is not supported by kspace_style ewald/disp
    const r = await run(FCC.replace('gewald/disp 0.4', 'gewald/disp 0.4 splittol 1.0e-4 disp/auto yes mesh/disp 8 8 8 order/disp 5 force/disp/real 1e-4 force/disp/kspace 1e-4'));
    expect(r.error).toBeNull();
    expect(close(r.rows[0].elong, -20.3347008705397, 1e-12)).toBe(true);
    expect(close(r.rows[0].evdwl, -584.183842934727, 1e-12)).toBe(true);
    expect((await run(FCC.replace('gewald/disp 0.4', 'gewald/disp 0.4 mix/disp none'))).error)
      .toContain('kspace_modify mix/disp none is not supported by kspace_style ewald/disp');
  });

  it('accepts per-type LJ cutoffs and shift yes with flag_lj long', async () => {
    // Measured with native LAMMPS: pair_coeff 2 2 with a fifth (cutoff1) argument is accepted under flag_lj long
    // and that pair uses its own LJ cutoff (src/KSPACE/pair_lj_long_coul_long.cpp init_one at b332f28).
    expect((await run(FCC.replace('pair_coeff 2 2 0.5 1.1', 'pair_coeff 2 2 0.5 1.1 2.0'))).error).toBeNull();
    expect((await run(FCC.replace('table 0 table/disp 0', 'table 0 table/disp 0 shift yes'))).error).toBeNull();
  });

  it('refuses flag_lj off with ewald/disp, as native (the pair keeps its dispersion flag, ewald/disp has no dispersion term)', async () => {
    // Measured with native LAMMPS: lj/long/coul/long off long with kspace_style ewald/disp stops with
    // KSpace style ewald/disp is incompatible with Pair style lj/long/coul/long (src/kspace.cpp pair_check)
    const r = await run(FCC.replace(' long off 2.5', ' off long 2.5'));
    expect(r.error).toMatch(/KSpace style ewald\/disp is incompatible with Pair style lj\/long\/coul\/long/);
  });
});
