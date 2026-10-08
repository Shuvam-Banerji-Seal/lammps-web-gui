import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { Fix } from '../src/engine/fix/fix';

/*
 * fix nvt/sllod (docs.lammps.org/fix_nvt_sllod.html): argument and requirement
 * errors, and the psllod keyword. The documented requirements are quoted in
 * src/engine/fix/nvt_sllod.ts (the fix deform remap rule, psllod yes/no).
 */

/** A 4x4x4 fcc prism (a small triclinic box, as the oracle cases use). */
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

/** Runs the script in a fresh session; returns the error message (or null) and the session. */
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

describe('fix nvt/sllod', () => {
  it('needs a fix deform command', async () => {
    const { error } = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1\nrun 1\n`);
    expect(error).toMatch(/requires a fix deform command/);
  });

  it('needs fix deform to remap velocities (remap v), not positions', async () => {
    for (const remap of ['x', 'none']) {
      const { error } = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1\nfix 2 all deform 1 xy erate 0.05 remap ${remap}\nrun 1\n`);
      expect(error, remap).toMatch(/needs fix deform with remap v/);
    }
  });

  it('accepts fix deform with remap v, and fix deform may come after the fix', async () => {
    const { error, session } = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1\nfix 2 all deform 1 xy erate 0.05 remap v\nrun 2\n`);
    expect(error).toBeNull();
    expect(session.sys.fixes.find((f: Fix) => f.id === '1')).toBeDefined();
  });

  it('parses psllod yes and no, and defaults to no', async () => {
    const yes = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 psllod yes\nfix 2 all deform 1 xy erate 0.05 remap v\nrun 1\n`);
    const no = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 psllod no\nfix 2 all deform 1 xy erate 0.05 remap v\nrun 1\n`);
    const dflt = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1\nfix 2 all deform 1 xy erate 0.05 remap v\nrun 1\n`);
    expect(yes.error).toBeNull();
    expect(no.error).toBeNull();
    const flag = (s: Session) => (s.sys.fixes.find((f) => f.id === '1') as unknown as { psllod: boolean }).psllod;
    expect(flag(yes.session)).toBe(true);
    expect(flag(no.session)).toBe(false);
    expect(flag(dflt.session)).toBe(false);
  });

  it('rejects a psllod value other than yes or no', async () => {
    for (const v of ['true', 'on', '']) {
      const { error } = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 psllod ${v}\nfix 2 all deform 1 xy erate 0.05 remap v\nrun 1\n`);
      expect(error, `psllod '${v}'`).toMatch(/psllod must be yes or no/);
    }
  });

  it('still needs the temp keyword and rejects pressure keywords (fix nvt rules)', async () => {
    const noTemp = await runScript(`${BOX}\nfix 1 all nvt/sllod psllod yes\nfix 2 all deform 1 xy erate 0.05 remap v\nrun 1\n`);
    expect(noTemp.error).toMatch(/needs the temp keyword/);
    const withP = await runScript(`${BOX}\nfix 1 all nvt/sllod temp 1.0 1.0 0.1 iso 0.0 0.0 1.0\nfix 2 all deform 1 xy erate 0.05 remap v\nrun 1\n`);
    expect(withP.error).toMatch(/pressure keywords/);
  });
});
