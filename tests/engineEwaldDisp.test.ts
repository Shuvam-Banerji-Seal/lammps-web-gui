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

  it('refuses the automatic dispersion G-ewald with a named StyleError', async () => {
    const r = await run(FCC.replace('kspace_modify gewald/disp 0.4', ''));
    expect(r.error).toMatch(/gewald\/disp/);
  });

  it('refuses pair_modify table/disp N > 0 with flag_lj long', async () => {
    const r = await run(FCC.replace('pair_modify table 0 table/disp 0', 'pair_modify table 0 table/disp 12'));
    expect(r.error).toMatch(/table\/disp/);
  });

  it('refuses the default dispersion table (table/disp not set)', async () => {
    const r = await run(FCC.replace(' table/disp 0', ''));
    expect(r.error).toMatch(/table\/disp/);
  });

  it('refuses splittol, disp/auto yes and mesh/disp with a named StyleError', async () => {
    expect((await run(FCC.replace('gewald/disp 0.4', 'gewald/disp 0.4 splittol 1.0e-4'))).error).toMatch(/splittol/);
    expect((await run(FCC.replace('gewald/disp 0.4', 'gewald/disp 0.4 disp/auto yes'))).error).toMatch(/disp\/auto/);
    expect((await run(FCC.replace('gewald/disp 0.4', 'gewald/disp 0.4 mesh/disp 8 8 8'))).error).toMatch(/mesh\/disp/);
  });

  it('refuses per-type LJ cutoffs and shift with flag_lj long', async () => {
    expect((await run(FCC.replace('pair_coeff 2 2 0.5 1.1', 'pair_coeff 2 2 0.5 1.1 2.0'))).error).toMatch(/cutoff1|LJ cutoff/);
    expect((await run(FCC.replace('table 0 table/disp 0', 'table 0 table/disp 0 shift yes'))).error).toMatch(/shift/);
  });

  it('flag_lj off is still refused by name', async () => {
    const r = await run(FCC.replace(' long off 2.5', ' off long 2.5'));
    expect(r.error).toMatch(/flag_lj 'off'/);
  });
});
