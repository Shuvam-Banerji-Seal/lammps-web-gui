import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

// pair_write and bond_write (src/engine-gpl/cmds/pair_write.ts, bond_write.ts): argument errors with the texts native
// LAMMPS (b332f28) prints, and the header and row layout of the written table. The numbers themselves are checked
// against native LAMMPS by the w255pw oracle cases (tests/oracle/w255pw_*.in).
const run = async (text: string, setup: string) => {
  const files = new Map<string, string>();
  const errors: string[] = [];
  const s = new Session({
    emit: (e: { kind: string; message?: string }) => { if (e.kind === 'error') errors.push(e.message ?? ''); },
    writeFile: (n: string, t: string, ap: boolean) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  } as never);
  let error = '';
  try {
    await s.execute(setup + text);
  } catch (e) {
    error = (e as Error).message;
  }
  return { files, error: error || errors.join('|') };
};

// two types, lj/cut with 1 1 and 2 2 set (mixed 1 2 by geometric rule)
const PAIR_SETUP = `units lj
atom_style atomic
region box block 0 10 0 10 0 10
create_box 2 box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 0.5 1.2
`;

const BOND_SETUP = `units real
atom_style full
region box block 0 10 0 10 0 10
create_box 2 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 2
mass * 1.0
pair_style zero 5.0
pair_coeff * *
bond_style harmonic
bond_coeff 1 100.0 1.5
`;

describe('pair_write argument errors (native Pair::write_file texts)', () => {
  it('refuses a missing pair_style, a wrong argument count, and bad types', async () => {
    expect((await run('pair_write 1 1 10 r 1.0 3.0 t.table KEY\n', 'units lj\natom_style atomic\nregion box block 0 10 0 10 0 10\ncreate_box 2 box\n')).error).toMatch(/Pair_write command before pair_style is defined/);
    expect((await run('pair_write 1 1 10 r 1.0\n', PAIR_SETUP)).error).toMatch(/Illegal pair_write command/);
    expect((await run('pair_write 1 1 10 r 1.0 3.0 t.table KEY 1.0\n', PAIR_SETUP)).error).toMatch(/Illegal pair_write command/);
    expect((await run('pair_write 0 1 10 r 1.0 3.0 t.table KEY\n', PAIR_SETUP)).error).toMatch(/Invalid atom types in pair_write command/);
    expect((await run('pair_write 1 5 10 r 1.0 3.0 t.table KEY\n', PAIR_SETUP)).error).toMatch(/Invalid atom types in pair_write command/);
  });

  it('refuses bad N, style, cutoffs and charges with native messages', async () => {
    expect((await run('pair_write 1 1 xx r 1.0 3.0 t.table KEY\n', PAIR_SETUP)).error).toMatch(/Expected integer parameter instead of 'xx' in input script or data file/);
    expect((await run('pair_write 1 1 10 foo 1.0 3.0 t.table KEY\n', PAIR_SETUP)).error).toMatch(/Invalid style foo in pair_write command/);
    expect((await run('pair_write 1 1 1 r 1.0 3.0 t.table KEY\n', PAIR_SETUP)).error).toMatch(/Must have at least 2 table values/);
    expect((await run('pair_write 1 1 10 r 3.0 1.0 t.table KEY\n', PAIR_SETUP)).error).toMatch(/Invalid cutoffs \(3 - 1\) in pair_write command/);
    expect((await run('pair_write 1 1 10 r 0.0 1.0 t.table KEY\n', PAIR_SETUP)).error).toMatch(/Invalid cutoffs \(0 - 1\) in pair_write command/);
    expect((await run('pair_write 1 1 10 r abc 3.0 t.table KEY\n', PAIR_SETUP)).error).toMatch(/Expected floating point parameter instead of 'abc' in input script or data file/);
    expect((await run('pair_write 1 1 10 r 1.0 3.0 t.table KEY x 1.0\n', PAIR_SETUP)).error).toMatch(/Expected floating point parameter instead of 'x'/);
  });

  it('refuses style bitmap by name (its lookup is not ported) and writes nothing', async () => {
    const r = await run('pair_write 1 1 10 bitmap 1.0 3.0 t.table KEY\n', PAIR_SETUP);
    expect(r.error).toMatch(/pair_write style bitmap is not supported by the browser engine/);
    expect(r.files.has('t.table')).toBe(false);
  });

  it('writes the header and rows of style r and rsq in native layout', async () => {
    const r = await run('pair_write 1 1 3 r 1.0 2.0 t.table LJ11\n', PAIR_SETUP);
    expect(r.error).toBe('');
    const text = r.files.get('t.table') ?? '';
    expect(text).toMatch(/^# DATE: \d{4}-\d{2}-\d{2} UNITS: lj Created by pair_write\n# Pair potential lj\/cut for atom types 1 1: i,r,energy,force\n\nLJ11\nN 3 R 1 2\n\n/);
    // three rows; the first at the inner cutoff, the last at the outer
    const rows = text.trim().split('\n').slice(-3);
    expect(rows.map((l) => l.trim().split(/\s+/)[0])).toEqual(['1', '2', '3']);
    expect(Number(rows[0].trim().split(/\s+/)[1])).toBe(1);
    expect(Number(rows[2].trim().split(/\s+/)[1])).toBe(2);
    // the rsq table: its header names RSQ and the radii are evenly spaced in r^2
    const q = await run('pair_write 1 2 3 rsq 1.0 2.0 t.table KEY\n', PAIR_SETUP);
    expect(q.files.get('t.table')).toMatch(/\nKEY\nN 3 RSQ 1 2\n\n/);
    const qr = q.files.get('t.table')!.trim().split('\n').slice(-3).map((l) => Number(l.trim().split(/\s+/)[1]));
    expect(qr[1]).toBeCloseTo(Math.sqrt(2.5), 12);
  });

  it('a pair style without single() refuses by name', async () => {
    const r = await run('pair_write 1 1 10 r 1.0 3.0 t.table KEY\n', `units lj
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
mass * 1.0
pair_style tersoff
`);
    // native Pair::write_file: "Pair style tersoff does not support pair_write" (sw has single_enable 0)
    expect(r.error).toMatch(/Pair style tersoff does not support pair_write/);
  });
});

describe('bond_write argument errors (native Bond::write_file texts)', () => {
  it('refuses a missing bond_style, bonds that are not allowed, and a wrong argument count', async () => {
    expect((await run('bond_write 1 10 1.0 3.0 tb.table KEY\n', 'units real\natom_style atomic\nregion box block 0 10 0 10 0 10\ncreate_box 2 box\n')).error).toMatch(/Bond_write command when no bonds allowed/);
    expect((await run('bond_write 1 10 1.0 3.0 tb.table KEY\n', 'units real\natom_style full\nregion box block 0 10 0 10 0 10\ncreate_box 2 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 2\n')).error).toMatch(/Bond_write command before bond_style is defined/);
    expect((await run('bond_write 1 10 1.0 3.0 tb.table\n', BOND_SETUP)).error).toMatch(/Illegal bond_write command/);
    expect((await run('bond_write 1 10 1.0 3.0 tb.table KEY 1\n', BOND_SETUP)).error).toMatch(/Illegal bond_write command/);
  });

  it('refuses bad atom types, N, lengths and integers with native messages', async () => {
    expect((await run('bond_write 1 10 1.0 3.0 tb.table KEY 0 1\n', BOND_SETUP)).error).toMatch(/Invalid atom type 0 in bond_write command/);
    expect((await run('bond_write 1 10 1.0 3.0 tb.table KEY 1 9\n', BOND_SETUP)).error).toMatch(/Invalid atom type 9 in bond_write command/);
    expect((await run('bond_write 1 1 1.0 3.0 tb.table KEY\n', BOND_SETUP)).error).toMatch(/Must have at least 2 table values/);
    expect((await run('bond_write 1 10 3.0 1.0 tb.table KEY\n', BOND_SETUP)).error).toMatch(/Invalid rlo=3 \/ rhi=1 values in bond_write command\./);
    expect((await run('bond_write 1 xx 1.0 3.0 tb.table KEY\n', BOND_SETUP)).error).toMatch(/Expected integer parameter instead of 'xx' in input script or data file/);
  });

  it('writes the harmonic table with the EQ header and native row layout', async () => {
    const r = await run('bond_write 1 4 1.0 2.0 tb.table HARM\n', BOND_SETUP);
    expect(r.error).toBe('');
    const text = r.files.get('tb.table') ?? '';
    expect(text).toMatch(/^# DATE: \d{4}-\d{2}-\d{2} UNITS: real Created by bond_write\n# Bond potential harmonic for bond type 1: i,r,energy,force\n\nHARM\nN 4 EQ 1.5\n\n/);
    const rows = text.trim().split('\n').slice(-4).map((l) => l.trim().split(/\s+/).map(Number));
    expect(rows.map((w) => w[0])).toEqual([1, 2, 3, 4]);
    // harmonic: E = K (r - r0)^2 at r = 1 gives 100 * 0.25 = 25 and the force -dE/dr = -2 K (r - r0) = 100
    expect(rows[0][1]).toBeCloseTo(1, 12);
    expect(rows[0][2]).toBeCloseTo(25, 9);
    expect(rows[0][3]).toBeCloseTo(100, 9);
  });
});
