import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix charge/regulation (src/engine-gpl/mc, a port of LAMMPS src/MC/fix_charge_regulation.cpp) and a check of
 * fix filter/corotate on this atomic system (src/engine-gpl/filtercorotate). The oracle cases w116xf_* check the moves against
 * native LAMMPS; these tests check the option parsing: accepted keywords run, unsupported ones are refused by name.
 */
const BOX = `units lj
atom_style charge
region box block 0 6 0 6 0 6
create_box 3 box
create_atoms 1 random 30 4321 box
mass * 1
pair_style lj/cut/coul/cut 2.5
pair_coeff * * 1 1
`;

/** The error message of running the input (or null when it runs without an error). */
const errorOf = async (fixLine: string): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  try {
    await session.execute(`${BOX}${fixLine}\n`);
  } catch (e) {
    return String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  return err && 'message' in err ? err.message : null;
};

describe('fix charge/regulation options', () => {
  it('accepts the box-wide acid and salt options', async () => {
    expect(await errorOf('fix cr all charge/regulation 2 3 acid_type 1 pH 7 pKa 6 pIp 3 pIm 3 lunit_nm 0.72 nevery 10 nmc 5 seed 2345 temp 1.0')).toBeNull();
  });

  it('refuses rxd > 0 by name', async () => {
    expect(await errorOf('fix cr all charge/regulation 2 3 acid_type 1 rxd 0.5 seed 1')).toMatch(/rxd/);
  });

  it('refuses onlysalt and group by name', async () => {
    expect(await errorOf('fix cr all charge/regulation 2 3 onlysalt yes 1 -1 seed 1')).toMatch(/onlysalt/);
    expect(await errorOf('fix cr all charge/regulation 2 3 group foo seed 1')).toMatch(/'group'/);
  });

  it('refuses an equal-style variable for pH', async () => {
    expect(await errorOf('fix cr all charge/regulation 2 3 acid_type 1 pH v_ph seed 1')).toMatch(/v_name/);
  });

  it('needs a seed', async () => {
    expect(await errorOf('fix cr all charge/regulation 2 3 acid_type 1')).toMatch(/Seed/);
  });
});

describe('fix filter/corotate', () => {
  it('needs a molecular system, as the native constructor checks (tests/engineFilterCorotate.test.ts has the rest)', async () => {
    const msg = await errorOf('fix cor all filter/corotate m 1.0');
    expect(msg).toMatch(/Cannot use fix filter\/corotate with non-molecular system/);
  });
});
