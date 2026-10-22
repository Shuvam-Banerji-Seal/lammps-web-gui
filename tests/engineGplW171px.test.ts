import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { fmtShortest } from '../src/engine/cformat';

/*
 * Ported styles fix saed/vtk, compute podd/atom and compute slcsa/atom (LAMMPS commit b332f28). Native parity is in
 * tests/oracle/w171px_*.in (engineOracle.test.ts); this file checks the number format of the VTK writer and the refusals.
 */

const CASES = join(__dirname, 'oracle');

const run = async (script: string, inputs: Record<string, string> = {}): Promise<{ errors: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  for (const [name, text] of Object.entries(inputs)) session.addFile(name, text);
  try {
    await session.execute(script);
  } catch (e) {
    events.push({ kind: 'error', message: String(e instanceof Error ? e.message : e) } as unknown as EngineEvent);
  }
  return { errors: events.filter((e) => e.kind === 'error').map((e) => ('message' in e ? String(e.message) : '')) };
};

describe('fix saed/vtk number format (C++ fmt "{}" as the VTK writer prints it)', () => {
  it('prints shortest double forms with fixed notation for exponents in [-4, 16)', () => {
    expect(fmtShortest(1, false)).toBe('1');
    expect(fmtShortest(-1, false)).toBe('-1');
    expect(fmtShortest(2.5, false)).toBe('2.5');
    expect(fmtShortest(0.0001, false)).toBe('0.0001');
    expect(fmtShortest(1e15, false)).toBe('1000000000000000');
    expect(fmtShortest(0, false)).toBe('0');
    expect(fmtShortest(-0, false)).toBe('-0');
  });

  it('switches to exponent notation outside that range, with at least two exponent digits', () => {
    expect(fmtShortest(1e16, false)).toBe('1e+16');
    expect(fmtShortest(0.00001, false)).toBe('1e-05');
    expect(fmtShortest(-1.5e-7, false)).toBe('-1.5e-07');
    expect(fmtShortest(-0.6000000000000001, false)).toBe('-0.6000000000000001');
  });

  it('prints single precision values with the shortest form that reads back as the same float', () => {
    expect(fmtShortest(Math.fround(0.1), true)).toBe('0.1');
    expect(fmtShortest(Math.fround(1 / 3), true)).toBe('0.33333334');
    expect(fmtShortest(Math.fround(1.5e-7), true)).toBe('1.5e-07');
  });
});

describe('compute slcsa/atom and podd/atom (ported) refusals', () => {
  it('refuses a model with other than four classes, naming the limit', async () => {
    const r = await run('units metal\nlattice fcc 3.52\nregion box block 0 1 0 1 0 1\ncreate_box 1 box\ncreate_atoms 1 box\n'
      + 'compute b all sna/atom 4.0 0.99 2 0.5 1.0\ncompute s all slcsa/atom 2 3 a b c d e c_b[1]\nrun 0\n');
    expect(r.errors[0]).toMatch(/only nclasses 4 is supported/);
  });

  it('refuses four-body POD models (their descriptor derivatives are not ported)', async () => {
    const param = readFileSync(join(CASES, 'w128pod_InP_param.pod'), 'utf8');
    const coeff = readFileSync(join(CASES, 'w128pod_InP_coeff.pod'), 'utf8');
    const r = await run('units metal\natom_style atomic\nlattice diamond 5.83\nregion box block 0 1 0 1 0 1\ncreate_box 2 box\n'
      + 'create_atoms 1 box basis 5 2 basis 6 2 basis 7 2 basis 8 2\nmass * 50.0\npair_style pod\npair_coeff * * p.pod c.pod In P\n'
      + 'compute dd all podd/atom p.pod c.pod In P\nrun 0\n', { 'p.pod': param, 'c.pod': coeff });
    expect(r.errors[0]).toMatch(/four-body POD descriptors/);
  });
});
