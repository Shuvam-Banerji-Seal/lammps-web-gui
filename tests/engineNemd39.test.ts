import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * NEMD keywords (docs.lammps.org/fix_deform.html, fix_nvt_sllod.html, fix_modify.html,
 * compute_temp_deform.html): parsing, values, and the refusals that remain.
 * Oracle parity for the accepted forms is in tests/oracle/w39nemd_*.in.
 */

const BOX = `
units lj
atom_style atomic
lattice fcc 0.8442
region box prism 0 4 0 4 0 4 0.3 0.0 0.0
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 4928459
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
timestep 0.002
`;

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  return { error, session, events };
};

const SLLOD = 'fix 1 all nvt/sllod temp 1.0 1.0 0.1';

describe('fix deform N = 0 and erate/rescale', () => {
  it('accepts N = 0 with the box updated after each position update (remap v)', async () => {
    const { error, session } = await runScript(`${BOX}\n${SLLOD}\nfix 2 all deform 0 xy erate 0.05 remap v\nrun 2\n`);
    expect(error).toBeNull();
    expect(session.sys.fixes.find((f) => f.id === '2')).toBeDefined();
  });

  it('rejects a negative N', async () => {
    const { error } = await runScript(`${BOX}\n${SLLOD}\nfix 2 all deform -1 xy erate 0.05 remap v\nrun 1\n`);
    expect(error).toMatch(/N must be zero or a positive step interval/);
  });

  it('refuses the variable style with N = 0 (native: "Fix deform cannot use the variable style with nevery = 0")', async () => {
    const { error } = await runScript(`${BOX}\nvariable ex equal 0.01\nvariable ey equal 0.01\n${SLLOD}\nfix 2 all deform 0 xy variable v_ex v_ey remap v\nrun 1\n`);
    expect(error).toMatch(/variable style with nevery = 0/);
  });

  it('accepts a lone tilt erate/rescale and refuses it on a length or with other parameters', async () => {
    const ok = await runScript(`${BOX}\n${SLLOD}\nfix 2 all deform 0 xy erate/rescale 0.05 remap v\nrun 2\n`);
    expect(ok.error).toBeNull();
    const onLength = await runScript(`${BOX}\n${SLLOD}\nfix 2 all deform 0 x erate/rescale 0.05 remap v\nrun 1\n`);
    expect(onLength.error).toMatch(/is only valid for tilt parameters/);
    const mixed = await runScript(`${BOX}\n${SLLOD}\nfix 2 all deform 0 xy erate/rescale 0.05 x trate 0.02 remap v\nrun 1\n`);
    expect(mixed.error).toMatch(/erate\/rescale together with other parameters is not supported/);
  });
});

describe('compute temp/deform temp keyword', () => {
  const withTemp = (keyword: string) => `${BOX}\n${SLLOD}\nfix 2 all deform 0 xy erate 0.05 remap v\ncompute part all temp/partial 1 1 0\ncompute tt all temp/deform ${keyword}\nrun 1\n`;

  it('accepts temp ID and stores it', async () => {
    const { error, session } = await runScript(withTemp('temp part'));
    expect(error).toBeNull();
    const tt = session.sys.computes.find((c) => c.id === 'tt') as unknown as { tempId: string | null };
    expect(tt.tempId).toBe('part');
  });

  it('defaults to no internal temperature compute', async () => {
    const { error, session } = await runScript(withTemp(''));
    expect(error).toBeNull();
    const tt = session.sys.computes.find((c) => c.id === 'tt') as unknown as { tempId: string | null };
    expect(tt.tempId).toBeNull();
  });

  it('rejects unknown keywords and a missing value', async () => {
    const bad = await runScript(withTemp('profile part'));
    expect(bad.error).toMatch(/Unknown compute temp\/deform keyword: profile/);
    const missing = await runScript(withTemp('temp'));
    expect(missing.error).toMatch(/Illegal compute temp\/deform temp command: missing argument/);
  });

  it('rejects a missing temperature compute, a non-temperature compute and a group mismatch (native messages)', async () => {
    const none = await runScript(withTemp('temp nosuch'));
    expect(none.error).toMatch(/Temperature ID nosuch for compute temp\/deform does not exist/);
    const pe = await runScript(`${BOX}\n${SLLOD}\nfix 2 all deform 0 xy erate 0.05 remap v\ncompute pe all pe\ncompute tt all temp/deform temp pe\nrun 1\n`);
    expect(pe.error).toMatch(/does not compute temperature/);
    const grp = await runScript(`${BOX}\n${SLLOD}\nfix 2 all deform 0 xy erate 0.05 remap v\ngroup g2 region box\ncompute part g2 temp\ncompute tt all temp/deform temp part\nrun 1\n`);
    expect(grp.error).toMatch(/Group of temperature compute with ID part for compute temp\/deform does not match/);
  });
});

describe('fix nvt/sllod kick, integrator, peculiar', () => {
  const deform = 'fix 2 all deform 0 xy erate 0.05 remap v';

  it('accepts kick yes and kick no and rejects other values', async () => {
    expect((await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 kick no\n${deform}\nrun 1\n`)).error).toBeNull();
    expect((await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 kick yes\n${deform}\nrun 1\n`)).error).toBeNull();
    const bad = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 kick maybe\n${deform}\nrun 1\n`);
    expect(bad.error).toMatch(/Expected boolean parameter instead of '.*' in input script or data file/);
  });

  it('consumes the queued kick at the run setup (default kick yes)', async () => {
    const kickOf = (s: Session) => (s.sys.fixes.find((f) => f.id === '1') as unknown as { kickPending: boolean }).kickPending;
    const dflt = await runScript(`${BOX}\n${SLLOD}\n${deform}\nrun 0\n`);
    const no = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 kick no\n${deform}\nrun 0\n`);
    expect(dflt.error).toBeNull();
    expect(no.error).toBeNull();
    expect(kickOf(dflt.session)).toBe(false);   // applied at the run setup
    expect(kickOf(no.session)).toBe(false);
  });

  it('fix_modify kick yes queues a kick; fix_modify kick on another fix is refused', async () => {
    const ok = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 kick no\n${deform}\nrun 0\nfix_modify 1 kick yes\nrun 0\n`);
    expect(ok.error).toBeNull();
    const other = await runScript(`${BOX}\nfix 1 all nve\nfix_modify 1 kick yes\nrun 1\n`);
    expect(other.error).toMatch(/Fix 1 nve does not support fix_modify kick command/);
    const badValue = await runScript(`${BOX}\n${SLLOD}\n${deform}\nfix_modify 1 kick maybe\n`);
    expect(badValue.error).toMatch(/Expected boolean parameter instead of 'maybe'/);
  });

  it('refuses integrator legacy and peculiar yes by name, accepts integrator reversible and peculiar no', async () => {
    const legacy = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 integrator legacy\n${deform}\nrun 1\n`);
    expect(legacy.error).toMatch(/integrator legacy is not supported/);
    const rev = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 integrator reversible\n${deform}\nrun 1\n`);
    expect(rev.error).toBeNull();
    const peculiar = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 peculiar yes\n${deform}\nrun 1\n`);
    expect(peculiar.error).toMatch(/peculiar yes is not supported/);
    const pNo = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 peculiar no\n${deform}\nrun 1\n`);
    expect(pNo.error).toBeNull();
    const badInt = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 integrator fast\n${deform}\nrun 1\n`);
    expect(badInt.error).toMatch(/integrator must be reversible or legacy/);
  });
});
