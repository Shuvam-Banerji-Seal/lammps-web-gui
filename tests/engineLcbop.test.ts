import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { StyleError } from '../src/engine/force/types';
import { readLcbopFile, splineInit } from '../src/engine-gpl/lcbop/params';
import { F_conj, gSpline } from '../src/engine-gpl/lcbop/funcs';
import type { StyleContext } from '../src/engine/force/types';

// pair_style lcbop (ported from LAMMPS b332f28, see src/engine-gpl/lcbop/). The parity cases live in
// tests/oracle/w234lc_*; here are the potential-file reader and its errors, the refused options, the spline
// tables against values computed from the potential file, and the energy of a random carbon cluster that
// native LAMMPS gives (tests/oracle/w234lc_cl1.in, run 0, free boundaries).

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const ORACLE = join(__dirname, 'oracle');
const potential = (name: string): string => readFileSync(join(POT, name), 'utf8');

const ctx = { s: { units: { style: 'metal' } }, readFile: () => '', log: () => {} } as unknown as StyleContext;

/** Runs a script in a fresh session with the given potential files; returns thermo rows and error messages. */
const run = async (script: string, files: Record<string, string> = {}): Promise<{ rows: Record<string, number>[]; errors: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  try {
    await session.execute(script);
  } catch {
    // the error is also an event
  }
  return {
    rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as unknown as Record<string, number>),
    errors: events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error').map((e) => ('message' in e ? String(e.message) : '')),
  };
};

const box = `units metal
atom_style atomic
boundary f f f
region box block -20 20 -20 20 -20 20
create_box 1 box
mass 1 12.011
`;

describe('pair_style lcbop: potential file', () => {
  it('reads the 28 scalars and the spline tables of C.lcbop', () => {
    const { params, gX } = readLcbopFile(potential('C.lcbop'), 'C.lcbop', ctx);
    expect(params.r_1).toBe(1.7);
    expect(params.r_2).toBe(2.2);
    expect(params.gamma_1).toBe(1.5);
    expect(params.r_2_LR).toBe(6.0);
    expect(params.delta).toBe(0.5);
    expect(Array.from(gX)).toEqual([-1, -0.5, -0.33333333, 0, 0.5, 1]);
  });

  it('gives the F_conj values of the potential file (independent Python evaluation of the same cell formulas)', () => {
    const { gX, gC, F_data } = readLcbopFile(potential('C.lcbop'), 'C.lcbop', ctx);
    const T = { gX, gC, F_conj_field: splineInit(F_data) };
    const d = { dij: 0, dji: 0, dconj: 0 };
    expect(F_conj(T, 2.5, 2.5, 0.5, d)).toBeCloseTo(-0.032766375, 12);
    expect(F_conj(T, 1.2, 0.3, 0, d)).toBeCloseTo(0.0268733328, 12);
    expect(F_conj(T, 0, 0, 1, d)).toBe(0);
    const dg = { v: 0 };
    expect(Number.isFinite(gSpline(T, 0.9, dg))).toBe(true);
  });

  it('refuses a file whose UNITS tag differs from the units in use (native: PotentialFileReader)', async () => {
    const r = await run(`${box}pair_style lcbop\npair_coeff * * C.lcbop C\n`, { 'C.lcbop': potential('C.lcbop').replace('UNITS: metal', 'UNITS: real') });
    expect(r.errors.join('\n')).toContain('requires real units');
  });

  it('reports a truncated file with the native reader message', () => {
    const lines = potential('C.lcbop').split('\n');
    const cut = lines.slice(0, 40).join('\n');
    expect(() => readLcbopFile(cut, 'cut.lcbop', ctx)).toThrow(StyleError);
    expect(() => readLcbopFile(cut, 'cut.lcbop', ctx)).toThrow(/^Error reading LCBOP potential file cut\.lcbop: /);
  });

  it('reports a non-numeric value with the native tokenizer message', () => {
    const text = potential('C.lcbop').replace('1.7         r1, AA', '1.7x        r1, AA');
    expect(() => readLcbopFile(text, 'bad.lcbop', ctx)).toThrow(/Not a valid floating-point number: '1\.7x'/);
  });
});

describe('pair_style lcbop: styles and refused options', () => {
  it('takes no pair_style arguments', async () => {
    const r = await run(`${box}pair_style lcbop 3.0\n`);
    expect(r.errors.join('\n')).toContain('Illegal pair_style command');
  });

  it('accepts only element C in pair_coeff', async () => {
    const r = await run(`${box}pair_style lcbop\npair_coeff * * C.lcbop H\n`, { 'C.lcbop': potential('C.lcbop') });
    expect(r.errors.join('\n')).toContain('Incorrect args for pair coefficients');
  });

  it('refuses pair_modify shift and tail, naming them', async () => {
    const r = await run(`${box}pair_style lcbop\npair_coeff * * C.lcbop C\npair_modify tail yes\nrun 0\n`, { 'C.lcbop': potential('C.lcbop') });
    expect(r.errors.join('\n')).toContain('pair_style lcbop does not support the pair_modify shift and tail options');
  });

  it('reproduces the native energy of a random carbon cluster (oracle w234lc_cl1)', async () => {
    const text = readFileSync(join(ORACLE, 'w234lc_cl1.in'), 'utf8')
      .split('\n').filter((l) => l.trim() !== '' && !l.startsWith('#')).join('\n');
    const r = await run(text, { 'C.lcbop': potential('C.lcbop') });
    expect(r.errors).toEqual([]);
    // native LAMMPS b332f28: run 0, thermo step pe = 23.0471328235346 (oracle fixture tests/fixtures/oracle/w234lc_cl1.json)
    expect(r.rows[0].pe).toBeCloseTo(23.0471328235346, 10);
  });
});
