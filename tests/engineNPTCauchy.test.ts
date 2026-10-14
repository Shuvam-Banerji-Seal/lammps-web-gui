/*
 * fix npt/cauchy (src/engine-gpl/extrafix/fix_npt_cauchy.ts, ported from EXTRA-FIX at b332f28): the constructor's
 * refusals and the continuation store. Parity is checked by the oracle cases w141cy_* (tests/oracle).
 * docs.lammps.org/fix_npt_cauchy.html: "Setting alpha to zero is not permitted"; "A continue value of yes indicates
 * that the fix is subsequent to a previous run with the npt/cauchy fix".
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

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

const base = `units lj
atom_style atomic
boundary p p p
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 4928 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

describe('fix npt/cauchy (EXTRA-FIX)', () => {
  it('needs temperature and pressure control', async () => {
    expect((await runScript(`${base}fix c all npt/cauchy x 1.0 1.0 1.0\nrun 1\n`)).error).toContain('Temperature control must be used with fix npt/cauchy');
    expect((await runScript(`${base}fix c all npt/cauchy temp 1.0 1.0 1.0\nrun 1\n`)).error).toContain('Pressure control must be used with fix npt/cauchy');
  });

  it('refuses alpha zero or negative, by the documented message', async () => {
    const { error } = await runScript(`${base}fix c all npt/cauchy temp 1.0 1.0 1.0 x 1.0 1.0 1.0 alpha 0\nrun 1\n`);
    expect(error).toContain('Alpha cannot be zero or negative');
  });

  it('refuses continue yes without a previous fix npt/cauchy run', async () => {
    const { error } = await runScript(`${base}fix c all npt/cauchy temp 1.0 1.0 1.0 x 1.0 1.0 1.0 continue yes\nrun 1\n`);
    expect(error).toContain('Continuation run must follow a previously equilibrated npt/cauchy run');
  });

  it('names the keywords it does not implement', async () => {
    expect((await runScript(`${base}fix c all npt/cauchy temp 1.0 1.0 1.0 x 1.0 1.0 1.0 ptemp 1.0\nrun 1\n`)).error).toContain("'ptemp' is not a fix npt/cauchy keyword");
    expect((await runScript(`${base}fix c all npt/cauchy temp 1.0 1.0 1.0 x 1.0 1.0 1.0 nreset 5\nrun 1\n`)).error).toContain('nreset is not supported');
    expect((await runScript(`${base}fix c all npt/cauchy temp 1.0 1.0 1.0 x 1.0 1.0 1.0 drag 0.1\nrun 1\n`)).error).toContain('drag is not supported');
  });

  it('writes the CauchyStat state with write_restart (the restarted run is the oracle case w176rs_cauchy)', async () => {
    const { error } = await runScript(`${base}fix c all npt/cauchy temp 1.0 1.0 1.0 x 1.0 1.0 1.0\nrun 1\nwrite_restart r.restart\n`);
    expect(error).toBe('');
  });

  it('runs two runs, the second one continuing the stored target stress', async () => {
    const script = `${base}fix c all npt/cauchy temp 1.0 1.0 1.0 x 2.0 2.0 1.0 y 2.0 2.0 1.0 z 2.0 2.0 1.0 alpha 0.001 continue no
thermo_style custom step temp pxx pyy pzz lx
thermo 5
run 10
fix c all npt/cauchy temp 1.0 1.0 1.0 x 2.0 2.0 1.0 y 2.0 2.0 1.0 z 2.0 2.0 1.0 alpha 0.001 continue yes
run 10
`;
    const { rows, error } = await runScript(script);
    expect(error).toBe('');
    expect(rows.map((r) => r.step)).toEqual([0, 5, 10, 10, 15, 20]);
    expect(rows.every((r) => Number.isFinite(r.pxx) && Number.isFinite(r.lx))).toBe(true);
  });
});
