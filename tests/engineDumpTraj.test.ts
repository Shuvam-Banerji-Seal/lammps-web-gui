import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { applyFileWrite } from '../src/engine/fileWrite';

/*
 * dump dcd and xtc (the GPL port in src/engine-gpl/dump/dump_dcd.ts and dump_xtc.ts): the checks of the constructor,
 * dump_modify and init. Measured with native LAMMPS: each input below gives the message expected. The bytes of the files
 * are compared with native ones by the oracle cases dump_dcd_xtc, dump_dcd_xtc_real and dump_xtc_wide.
 */

const run = async (input: string): Promise<{ files: Map<string, string>; log: string }> => {
  const files = new Map<string, string>();
  let log = '';
  const s = new Session({
    emit: (e: { kind: string; text?: string }) => { if (e.kind === 'log') log += `${e.text}\n`; },
    writeFile: (n: string, t: string, ap: boolean, at?: number) => files.set(n, applyFileWrite(files.get(n), t, ap, at)),
  } as never);
  await s.execute(input);
  return { files, log };
};

const BASE = `units lj
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
`;

const fails = async (body: string, message: string): Promise<void> => {
  await expect(run(BASE + body)).rejects.toThrow(new RegExp(`^line \\d+: ${message.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`));
};

describe('dump dcd and xtc: native checks', () => {
  it('the dump command: extra words and file names', async () => {
    await fails('dump 1 all dcd 10 a.dcd extra\n', 'Illegal dump dcd command');
    await fails('dump 1 all dcd 10 a*.dcd\n', 'Invalid dump dcd filename a*.dcd');
    await fails('dump 1 all dcd 10 a.dcd.gz\n', 'Invalid dump dcd filename a.dcd.gz');
    await fails('dump 1 all dcd 10 a.bin\n', 'Invalid dump dcd filename a.bin');
    await fails('dump 1 all xtc 10 a.xtc.gz\n', 'Invalid dump xtc filename');
    await fails('dump 1 all xtc 10 a%.xtc\n', 'Invalid dump xtc filename');
    await fails('dump 1 all xtc 10 a.xtc x\n', 'Illegal dump xtc command');
  });

  it('init: sorting by atom ID, flush, every', async () => {
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 sort off\nrun 0\n', 'Dump dcd requires sorting by atom ID');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 sort 2\nrun 0\n', 'Dump dcd requires sorting by atom ID');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 sort 7\nrun 0\n', 'Dump dcd requires sorting by atom ID');
    await fails('dump 1 all xtc 10 a.xtc\ndump_modify 1 flush yes\nrun 0\n', 'Cannot set dump_modify flush for dump xtc');
    await fails('variable e equal 10\ndump 1 all dcd 10 a.dcd\ndump_modify 1 every v_e\nrun 0\n', 'Cannot use variable every setting for dump dcd');
    await fails('variable e equal 10\ndump 1 all xtc 10 a.xtc\ndump_modify 1 every v_e\nrun 0\n', 'Cannot use every variable setting for dump xtc');
    await fails('dump 1 all dcd 10 a.dcd\nrun 0\ndump_modify 1 every 20\nrun 0\n', 'Cannot change dump_modify every for dump dcd');
    await fails('dump 1 all xtc 10 a.xtc\nrun 0\ndump_modify 1 every 20\nrun 0\n', 'Cannot change dump_modify every for dump xtc');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 every/time 0.1\nrun 0\n', 'Cannot use every/time setting for dump dcd');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 every/time 0\n', 'Invalid dump_modify every/time argument: 0');
    await fails('dump 1 all dcd 10 a.dcd\ncreate_atoms 1 single 0.5 0.5 0.5\nrun 0\n', 'Dump dcd of non-matching # of atoms');
    await fails('write_dump all dcd w.dcd modify sort off\n', 'Dump dcd requires sorting by atom ID');
    await fails('write_dump all xtc w.xtc modify flush yes\n', 'Cannot set dump_modify flush for dump xtc');
  });

  it('dump_modify keywords', async () => {
    await fails('dump 1 all xtc 10 a.xtc\ndump_modify 1 precision 50\n', 'Illegal dump_modify command');
    await fails('dump 1 all xtc 10 a.xtc\ndump_modify 1 sfactor 0\n', 'Illegal dump_modify sfactor value (must be > 0.0)');
    await fails('dump 1 all xtc 10 a.xtc\ndump_modify 1 tfactor -1\n', 'Illegal dump_modify tfactor value (must be > 0.0)');
    await fails('dump 1 all xtc 10 a.xtc\ndump_modify 1 precision\n', 'Illegal dump_modify command');
    await fails('dump 1 all xtc 10 a.xtc\ndump_modify 1 unwrap\n', 'Illegal dump_modify command');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 precision 100\n', 'Unknown dump_modify keyword: precision');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 unwrap\n', 'Illegal dump_modify unwrap command: missing argument(s)');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 unwrap maybe\n', "Expected boolean parameter instead of 'maybe' in input script or data file");
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 buffer yes\n', 'Dump_modify buffer yes not allowed for this style');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 format int %d\n', 'Unknown dump_modify format keyword: int');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 element Ar\n', 'Unknown dump_modify keyword: element');
    await fails('dump 1 all dcd 10 a.dcd\ndump_modify 1 colname 1 foo\n', 'Incorrect dump_modify arguments: colname 1 foo');
  });

  it('accepted settings', async () => {
    for (const body of [
      'dump 1 all xtc 10 a.xtc\ndump_modify 1 flush no buffer no precision 100.000001\nrun 0\n',
      'dump 1 all dcd 10 a.dcd\ndump_modify 1 flush yes format line "%g"\nrun 0\n',
      'dump 1 all dcd 10 a.dcd\ndump_modify 1 every 20\nrun 0\n',
      'dump 1 all xtc 10 a.xtc\ncreate_atoms 1 single 0.5 0.5 0.5\nrun 0\n',
      'write_dump all dcd w.dcd modify every 3\n',
    ]) await expect(run(BASE + body)).resolves.toBeDefined();
  });

  it('the file is created at the dump command; xtc in lj units warns', async () => {
    const { files, log } = await run(`${BASE}dump 1 all dcd 10 a.dcd\ndump 2 all xtc 10 a.xtc\n`);
    expect(files.get('a.dcd')).toBe('');
    expect(files.get('a.xtc')).toBe('');
    expect(log).toContain('WARNING: No automatic unit conversion to XTC file format conventions possible for units lj');
  });

  it('dcd header: START, SKIP and the NFILE and NSTEP fields updated after each frame', async () => {
    const { files } = await run(`${BASE}dump 1 all dcd 10 a.dcd\nrun 25\n`);
    const t = files.get('a.dcd')!;
    const bytes = Uint8Array.from(t, (c) => c.charCodeAt(0));
    const v = new DataView(bytes.buffer);
    const int = (at: number) => v.getInt32(at, true);
    expect(int(0)).toBe(84);
    expect(t.slice(4, 8)).toBe('CORD');
    expect([int(8), int(12), int(16), int(20)]).toEqual([3, 0, 10, 20]); // NFILE, START, SKIP, NSTEP
    expect(t.slice(100, 180).trimEnd()).toBe('Written by LAMMPS');
    expect(t.slice(180, 260)).toMatch(/^REMARKS Created \d\d [A-Z][a-z]+,\d{4} at \d\d:\d\d +$/);
    // the header, then 3 frames of the box (6 doubles) and x, y, z (32 floats each)
    expect(bytes.length).toBe(276 + 3 * (56 + 3 * (8 + 4 * 32)));
  });
});
