import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError } from '../src/engine/types';

/* fix rigid extras (src/engine/fix/rigid.ts): bodystyle custom, keyword infile, fix_modify bodyforces,
 * and the named errors for what the engine does not support. Dynamics vs native LAMMPS: oracle cases w6rigid_*. */

const BOX = `units lj
atom_style molecular
lattice sc 1.5
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
set atom 1*14 mol 1
set atom 15*27 mol 2
variable bid atom floor(x/2.0)+3*floor(y/2.5)
variable scal equal 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
`;

/** Runs BOX plus extra lines; returns the engine error, or null when the run succeeds. */
const run = async (lines: string, files: Record<string, string> = {}): Promise<EngineError | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  for (const [name, text] of Object.entries(files)) s.addFile(name, text);
  try {
    await s.execute(`${BOX}${lines}\nrun 5`);
    return null;
  } catch (e) {
    return e as EngineError;
  }
};

const msg = (e: EngineError | null) => e?.message ?? '';

describe('fix rigid bodystyle custom', () => {
  it('accepts an atom-style variable as the body ID', async () => {
    expect(await run('fix 1 all rigid custom v_bid')).toBeNull();
    expect(await run('fix 1 all rigid/small custom v_bid')).toBeNull();
  });

  it('rejects i_name (fix property/atom is not in the engine)', async () => {
    expect(msg(await run('fix 1 all rigid custom i_bodyid'))).toMatch(/i_bodyid.*fix property\/atom/);
  });

  it('names an undefined or non-atom variable', async () => {
    expect(msg(await run('fix 1 all rigid custom v_nope'))).toMatch(/variable nope does not exist/);
    expect(msg(await run('fix 1 all rigid custom v_scal'))).toMatch(/v_scal|scal.*must be atom-style/);
    expect(msg(await run('fix 1 all rigid custom bodyid'))).toMatch(/expected v_name/);
  });

  it('rejects unknown bodystyles by name', async () => {
    expect(msg(await run('fix 1 all rigid bogus'))).toMatch(/unknown bodystyle 'bogus'/);
  });
});

describe('fix rigid keyword infile', () => {
  const body = (...vals: (number | string)[]) => vals.join(' ');
  const good = `# header\n\n1\n${body(1, 16, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)}\n`;

  it('reads a well-formed file', async () => {
    expect(await run('fix 1 all rigid custom v_bid infile body.txt', { 'body.txt': good })).toBeNull();
  });

  it('names each kind of malformed file', async () => {
    const f = (text: string) => run('fix 1 all rigid custom v_bid infile body.txt', { 'body.txt': text });
    expect(msg(await f(''))).toMatch(/no count line/);
    expect(msg(await f('x\n'))).toMatch(/first line must be the number of bodies N/);
    expect(msg(await f('2\n' + body(1, 16, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + '\n'))).toMatch(/expected 2 body lines, found 1/);
    expect(msg(await f('1\n1 16 0 0\n'))).toMatch(/needs 20 values/);
    expect(msg(await f('1\n' + body(1, 16, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'a') + '\n'))).toMatch(/non-numeric/);
    expect(msg(await f('1\n' + body(0, 16, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + '\n'))).toMatch(/positive integer/);
    expect(msg(await f('1\n' + body(1, -2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + '\n'))).toMatch(/masstotal of body 1 must be positive/);
    expect(msg(await f('2\n' + body(1, 16, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + '\n' + body(1, 16, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + '\n'))).toMatch(/listed twice/);
  });

  it('refuses a blank line between the count and the body lines (measured: native LAMMPS does too)', async () => {
    expect(msg(await run('fix 1 all rigid custom v_bid infile body.txt', { 'body.txt': `1\n\n${good.split('\n')[3]}\n` }))).toMatch(/expected 1 body lines, found 0/);
  });

  it('refuses an ID that is not a body', async () => {
    const text = `1\n${body(3, 16, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)}\n`;
    expect(msg(await run('fix 1 all rigid custom v_bid infile body.txt', { 'body.txt': text }))).toMatch(/infile body ID 3 is not a rigid body/);
  });

  it('names a missing infile', async () => {
    expect(msg(await run('fix 1 all rigid custom v_bid infile nofile.txt'))).toMatch(/nofile\.txt/);
  });
});

describe('fix_modify bodyforces', () => {
  it('accepts early and late, rejects other values', async () => {
    expect(await run('fix 1 all rigid custom v_bid\nfix_modify 1 bodyforces early')).toBeNull();
    expect(await run('fix 1 all rigid custom v_bid\nfix_modify 1 bodyforces late')).toBeNull();
    expect(msg(await run('fix 1 all rigid custom v_bid\nfix_modify 1 bodyforces middle'))).toMatch(/bodyforces must be early or late/);
  });
});

describe('fix rigid/nvt and rigid/nvt/small', () => {
  it('names the style it does not support', async () => {
    expect(msg(await run('fix 1 all rigid/nvt molecule temp 1.0 1.0 1.0'))).toMatch(/fix rigid\/nvt is not supported by the browser engine/);
    expect(msg(await run('fix 1 all rigid/nvt/small molecule temp 1.0 1.0 1.0'))).toMatch(/fix rigid\/nvt\/small is not supported/);
  });

  it('keeps temp and tparam out of the NVE styles', async () => {
    expect(msg(await run('fix 1 all rigid custom v_bid temp 1.0 1.0 1.0'))).toMatch(/keyword 'temp' is not supported/);
  });
});
