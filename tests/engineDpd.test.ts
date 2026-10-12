import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

// DPD-BASIC pair styles ported from LAMMPS (src/engine-gpl/dpd, commit b332f28): pair_style dpd, dpd/tstat, dpd/ext and
// dpd/ext/tstat. The numbers the oracle cases (tests/oracle/w77dpd_*.in) check against native LAMMPS are not repeated
// here; these tests cover the refusals by name and the no-noise energy bookkeeping.

const BOX = `units lj
boundary p p p
atom_style atomic
newton on
comm_modify vel yes
region simBox block 0 5 0 5 0 5
create_box 1 simBox
create_atoms 1 single 1.0 1.0 1.0
create_atoms 1 single 1.5 1.2 1.1
mass * 1.0
`;

const run = async (script: string): Promise<{ pe: number[]; errors: string[] }> => {
  const pe: number[] = [];
  const errors: string[] = [];
  const s = new Session({
    emit: (e: { kind: string; row?: { pe?: number }; message?: string }) => {
      if (e.kind === 'thermo' && e.row && e.row.pe !== undefined) pe.push(e.row.pe);
      if (e.kind === 'error') errors.push(e.message ?? '');
    },
    writeFile: () => {},
  } as never);
  try {
    await s.execute(script);
  } catch (e) {
    errors.push(e instanceof Error ? e.message : String(e));
  }
  return { pe, errors };
};

describe('DPD-BASIC styles ported from LAMMPS (engine-gpl/dpd)', () => {
  it('needs ghost velocities (comm_modify vel yes), as native does', async () => {
    const r = await run(`${BOX.replace('comm_modify vel yes\n', '')}pair_style dpd 1.0 1.0 1234\npair_coeff * * 25.0 4.5\nrun 0\n`);
    expect(r.errors.join(' ')).toMatch(/requires ghost atoms store velocity/);
  });

  it('refuses newton pair off', async () => {
    const r = await run(`${BOX.replace('newton on', 'newton off')}pair_style dpd 1.0 1.0 1234\npair_coeff * * 25.0 4.5\nrun 0\n`);
    expect(r.errors.join(' ')).toMatch(/newton pair off is not supported/);
  });

  it('refuses a temperature ramp in dpd/tstat and dpd/ext/tstat', async () => {
    const a = await run(`${BOX}pair_style dpd/tstat 1.0 2.0 1.0 1234\npair_coeff * * 4.5\nrun 0\n`);
    expect(a.errors.join(' ')).toMatch(/t_start != t_stop/);
    const b = await run(`${BOX}pair_style dpd/ext/tstat 1.0 2.0 1.0 1234\npair_coeff * * 4.5 4.5 0.5 0.5\nrun 0\n`);
    expect(b.errors.join(' ')).toMatch(/t_start != t_stop/);
  });

  it('needs a positive seed and every pair coefficient', async () => {
    const seed = await run(`${BOX}pair_style dpd 1.0 1.0 0\npair_coeff * * 25.0 4.5\nrun 0\n`);
    expect(seed.errors.join(' ')).toMatch(/Illegal pair_style command/);
    const unset = await run(`${BOX.replace('create_box 1', 'create_box 2')}pair_style dpd 1.0 1.0 1234\npair_coeff 1 1 25.0 4.5\nrun 0\n`);
    expect(unset.errors.join(' ')).toMatch(/All pair coeffs are not set/);
  });

  it('checks the pair_coeff argument count of each style', async () => {
    const dpd = await run(`${BOX}pair_style dpd 1.0 1.0 1234\npair_coeff * * 25.0\nrun 0\n`);
    expect(dpd.errors.join(' ')).toMatch(/Incorrect args for pair coefficients/);
    const ext = await run(`${BOX}pair_style dpd/ext 1.0 1.0 1234\npair_coeff * * 25.0 4.5 4.5\nrun 0\n`);
    expect(ext.errors.join(' ')).toMatch(/Incorrect args for pair coefficients/);
  });

  it('gives zero conservative energy for dpd/tstat, which has no conservative term', async () => {
    const r = await run(`${BOX}pair_style dpd/tstat 0.0 0.0 1.0 1234\npair_coeff * * 4.5\nthermo_style custom step pe\nthermo 0\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(r.pe[0]).toBe(0);
  });

  it('gives a nonzero conservative energy for dpd within the cutoff', async () => {
    const r = await run(`${BOX}pair_style dpd 0.0 1.0 1234\npair_coeff * * 25.0 4.5\nthermo_style custom step pe\nthermo 0\nrun 0\n`);
    expect(r.errors).toEqual([]);
    // pair at r = sqrt(0.25 + 0.04 + 0.01) = 0.5477 < 1.0: E = 0.5 a0 rc wd^2 with wd = 1 - r/rc. lj units print pe per atom
    // (thermo_modify norm yes is the default), so the total is twice the printed value
    const r0 = Math.sqrt(0.25 + 0.04 + 0.01);
    const wd = 1 - r0 / 1.0;
    expect(2 * r.pe[0]).toBeCloseTo(0.5 * 25.0 * 1.0 * wd * wd, 10);
  });
});
