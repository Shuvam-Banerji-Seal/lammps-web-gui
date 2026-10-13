import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Unit tests for the SHOCK package fixes ported into src/engine-gpl/shock (fix msst, fix nphug): the keyword and
 * geometry refusals and the thermo outputs. The numerical parity with native LAMMPS is in the w128shock_* oracle
 * cases (tests/oracle). docs.lammps.org/fix_msst.html: "dftb yes" needs "fix external", which the browser engine does
 * not have; docs.lammps.org/fix_nphug.html: "One and only one of the following keywords should be used: iso, aniso,
 * tri, x, y, z".
 */

const base = `units lj
atom_style atomic
boundary p p p
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 0.8 4928 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

/** Runs a script; returns the error message ('' when it ran) and the thermo rows. */
const runScript = async (text: string) => {
  const events: { kind: string; row?: Record<string, number>; message?: string }[] = [];
  const session = new Session({ emit: (e: never) => events.push(e), writeFile: () => {} } as never, undefined, 0, 1);
  let error = '';
  try {
    await session.execute(text);
  } catch (e) {
    error = String(e instanceof Error ? e.message : e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.message) error = err.message;
  const rows = events.filter((e) => e.kind === 'thermo').map((e) => e.row as Record<string, number>);
  return { rows, error };
};

describe('fix msst (SHOCK)', () => {
  it('refuses dftb yes by name, since fix external is not available', async () => {
    const { error } = await runScript(`${base}fix s all msst z 1.0 dftb yes\nrun 1\n`);
    expect(error).toContain('fix msst dftb yes');
    expect(error).toContain('fix external');
  });

  it('requires a periodic box', async () => {
    const { error } = await runScript(base.replace('boundary p p p', 'boundary f p p') + 'fix s all msst z 1.0\nrun 1\n');
    expect(error).toContain('Fix msst requires a periodic box');
  });

  it('rejects an unknown keyword and a bad direction', async () => {
    expect((await runScript(`${base}fix s all msst z 1.0 foo 2\n`)).error).toContain('Illegal fix msst command');
    expect((await runScript(`${base}fix s all msst w 1.0\n`)).error).toContain('Illegal fix msst command');
  });

  it('writes the four global outputs and the scalar in thermo', async () => {
    const { rows, error } = await runScript(`${base}fix s all nve\nfix m all msst z 1.5 q 10.0 mu 0.5\nthermo_style custom step f_m f_m[1] f_m[2] f_m[3] f_m[4]\nthermo 5\nrun 10\n`);
    expect(error).toBe('');
    expect(rows.length).toBeGreaterThan(1);
    expect(Object.keys(rows[0])).toEqual(expect.arrayContaining(['f_m', 'f_m[1]', 'f_m[2]', 'f_m[3]', 'f_m[4]']));
    // the Lagrangian position is zero at setup and moves with the cell after the first step
    expect(rows[0]['f_m[4]']).toBe(0);
    expect(rows[rows.length - 1]['f_m[4]']).not.toBe(0);
  });
});

describe('fix nphug (SHOCK)', () => {
  it('needs a temperature keyword', async () => {
    const { error } = await runScript(`${base}fix h all nphug z 1.5 1.5 20.0\nrun 1\n`);
    expect(error).toContain('Temperature control must be used with fix nphug');
  });

  it('needs a pressure keyword', async () => {
    const { error } = await runScript(`${base}fix h all nphug temp 1.0 1.0 10.0\nrun 1\n`);
    expect(error).toContain('Pressure control must be used with fix nphug');
  });

  it('requires equal Pstart and Pstop', async () => {
    const { error } = await runScript(`${base}fix h all nphug temp 1.0 1.0 10.0 z 1.5 2.0 20.0\nrun 1\n`);
    expect(error).toContain('Pstart and Pstop must have the same value');
  });

  it('is uniaxial or hydrostatic only', async () => {
    const { error } = await runScript(`${base}fix h all nphug temp 1.0 1.0 10.0 x 1.5 1.5 20.0 y 1.5 1.5 20.0\nrun 1\n`);
    expect(error).toContain('Specified target stress must be uniaxial or hydrostatic');
  });

  it('writes its vector: dhugoniot, us and up first, with the scalar in thermo', async () => {
    const { rows, error } = await runScript(`${base}fix h all nphug temp 1.0 1.0 10.0 z 1.5 1.5 20.0 tchain 1 pchain 0\nthermo_style custom step f_h f_h[1] f_h[2] f_h[3]\nthermo 5\nrun 10\n`);
    expect(error).toBe('');
    expect(rows[0]['f_h[1]']).toBe(0);
    expect(rows[0]['f_h[2]']).toBe(0);
    expect(rows[0]['f_h[3]']).toBe(0);
  });
});

describe('SHOCK fixes not ported', () => {
  it('names fix wall/piston', async () => {
    const { error } = await runScript(`${base}fix w all wall/piston xlo 0.0 0.5\nrun 1\n`);
    expect(error).toContain('fix wall/piston');
  });

  it('names fix append/atoms', async () => {
    const { error } = await runScript(`${base}fix a all append/atoms\nrun 1\n`);
    expect(error).toContain('fix append/atoms');
  });
});
