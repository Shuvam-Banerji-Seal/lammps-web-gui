import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * dump cfg (GPL port of src/dump_cfg.cpp, src/engine-gpl/dump/dump_cfg.ts): the files byte for byte against the native
 * fixture of tests/oracle/w215cfg_dump.in (the oracle compares them token by token), and the native argument checks.
 */

const run = async (input: string): Promise<{ files: Map<string, string>; error?: string }> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  let error: string | undefined;
  try {
    await s.execute(input);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) error = String(err.message);
  return { files, error };
};

const SETUP = 'units lj\nlattice fcc 0.8\nregion b block 0 1 0 1 0 1\ncreate_box 1 b\ncreate_atoms 1 box\nmass 1 1\npair_style zero 1.0\npair_coeff * *\n';

describe('dump cfg', () => {
  it('writes the native files byte for byte (w215cfg_dump)', async () => {
    const input = readFileSync(join(__dirname, 'oracle', 'w215cfg_dump.in'), 'utf8');
    const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'oracle', 'w215cfg_dump.json'), 'utf8')) as { files: Record<string, string> };
    const { files, error } = await run(input);
    expect(error).toBeUndefined();
    for (const [name, text] of Object.entries(fx.files)) expect(files.get(name), name).toBe(text);
  });

  it('refuses attributes as the native constructor does', async () => {
    // Measured with native LAMMPS: each of these stops with the message below
    const bad = async (cols: string) => (await run(`${SETUP}dump 1 all cfg 1 a.*.cfg ${cols}\n`)).error;
    expect(await bad('mass type xs ysu zs')).toMatch(/Dump cfg arguments can not mix xs\|ys\|zs with xsu\|ysu\|zsu/);
    expect(await bad('mass type xsu ys zsu')).toMatch(/Dump cfg arguments can not mix xs\|ys\|zs with xsu\|ysu\|zsu/);
    expect(await bad('type mass xs ys zs')).toMatch(/Dump cfg arguments must start with 'mass type xs ys zs' or 'mass type xsu ysu zsu'/);
    expect(await bad('mass type xs ys')).toMatch(/Dump cfg arguments must start with/);
    expect((await run(`${SETUP}dump 1 all cfg 1 a.cfg\n`)).error).toMatch(/No dump cfg arguments specified/);
  });

  it('needs a "*" file name at the run, as native init_style does; write_dump does not', async () => {
    const r = await run(`${SETUP}dump 1 all cfg 1 a.cfg mass type xs ys zs\nrun 0\n`);
    expect(r.error).toMatch(/Dump cfg requires one snapshot per file/);
    const w = await run(`${SETUP}write_dump all cfg w.cfg mass type xs ys zs\n`);
    expect(w.error).toBeUndefined();
    expect(w.files.get('w.cfg')).toMatch(/^Number of particles = 4\nA = 1 Angstrom \(basic length-scale\)\n/);
  });
});
