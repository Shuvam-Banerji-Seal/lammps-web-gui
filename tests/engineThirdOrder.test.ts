import { describe, expect, it } from 'vitest';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import { fixed8 } from '../src/engine-gpl/phonon/third_order';

// third_order (PHONON package, ported from src/PHONON/third_order.cpp at b332f28): the file it writes is compared
// with the native file of the same case in tests/fixtures/oracle (the oracle harness, engineOracle*.test.ts, does the
// number-by-number comparison; this file checks the text exactly, the output format and the refusals).

const CASES = join(__dirname, 'oracle');
const FIX = join(__dirname, 'fixtures', 'oracle');

const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^#\\s*${key}:\\s*(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

/** Runs an oracle case through the engine and returns the files it wrote (name -> text). */
const runCase = async (name: string): Promise<Map<string, string>> => {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const files = new Map<string, string>();
  const session = new Session({
    emit: () => {},
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  }, undefined, 0, 1);
  for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readFileSync(join(CASES, f), 'utf8'));
  for (const f of directive(text, 'oracle-potentials')) session.addFile(f, readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', f), 'utf8'));
  await session.execute(text);
  return files;
};

const CASE_NAMES = ['w241to_lj_regular', 'w241to_lj_group', 'w241to_lj_fold', 'w241to_sw_eskm'];

describe('fixed8 (fmt "%17.8f" of the native output)', () => {
  it('rounds half to even on the exact binary value and keeps the sign of negative zero', () => {
    expect(fixed8(0)).toBe('0.00000000');
    expect(fixed8(-0)).toBe('-0.00000000');
    expect(fixed8(-1e-12)).toBe('-0.00000000');
    expect(fixed8(0.001953125)).toBe('0.00195312'); // exact tie: 2^-9 rounds to the even eighth decimal
    expect(fixed8(0.005859375)).toBe('0.00585938'); // 3 * 2^-9: tie rounds to the even 8th digit (7 is odd, up)
    expect(fixed8(1.5)).toBe('1.50000000');
    expect(fixed8(-12.5)).toBe('-12.50000000');
    expect(fixed8(3806.85735431)).toBe('3806.85735431');
    expect(fixed8(123456.1)).toBe('123456.10000000');
  });
});

/** One line of the file: five integers, then three "%17.8f" numbers. */
const LINE = /^\d+ \d+ \d+ \d+ \d+ {1,}-?\d+\.\d{8} {1,}-?\d+\.\d{8} {1,}-?\d+\.\d{8}$/;

describe('third_order file against native LAMMPS', () => {
  for (const name of CASE_NAMES) {
    it(`${name}: the same entries in the same order, each value within the printed precision of native`, { timeout: 120_000 }, async () => {
      const mine = (await runCase(name)).get(`${name}.dat`);
      expect(mine, 'file written').toBeDefined();
      const ref = (JSON.parse(readFileSync(join(FIX, `${name}.json`), 'utf8')).files as Record<string, string>)[`${name}.dat`];
      if (process.env.W241_DUMP) {
        mkdirSync(process.env.W241_DUMP, { recursive: true });
        writeFileSync(join(process.env.W241_DUMP, `${name}.mine.dat`), mine!);
        writeFileSync(join(process.env.W241_DUMP, `${name}.native.dat`), ref);
      }
      const a = mine!.split('\n').filter(Boolean), b = ref.split('\n').filter(Boolean);
      expect(a.length, 'number of entries').toBe(b.length);
      let worst = 0;
      for (let k = 0; k < b.length; k++) {
        expect(a[k], `format of line ${k + 1}`).toMatch(LINE);
        const ta = a[k].trim().split(/\s+/), tb = b[k].trim().split(/\s+/);
        // the entry labels are exact; the values agree to one unit of the eighth decimal (the printed rounding of a
        // finite difference at this delta) or to 1e-6 relative
        expect(ta.slice(0, 5), `labels of line ${k + 1}`).toEqual(tb.slice(0, 5));
        for (let c = 5; c < 8; c++) {
          const x = Number(ta[c]), y = Number(tb[c]);
          worst = Math.max(worst, Math.abs(x - y) / Math.max(1e-8, 1e-6 * Math.abs(y)));
          expect(Math.abs(x - y) <= 1e-8 + 1e-6 * Math.abs(y), `value of line ${k + 1} column ${c - 4}: ${x} vs ${y}`).toBe(true);
        }
      }
      expect(worst).toBeLessThanOrEqual(1);
    });
  }
});

describe('third_order refusals', () => {
  const run = async (cmd: string) => {
    const base = readFileSync(join(CASES, 'w241to_lj_regular.in'), 'utf8').split('\n').filter((l) => !/^third_order/.test(l)).join('\n');
    const session = new Session({ emit: () => {}, writeFile: () => {} }, undefined, 0, 1);
    await session.execute(`${base}\n${cmd}\n`);
  };
  it('refuses binary and gzip output by name', async () => {
    await expect(run('third_order all regular 0.0001 binary yes')).rejects.toThrow(/binary yes output is native-only/);
    await expect(run('third_order all regular 0.0001 binary gzip')).rejects.toThrow(/binary gzip output is native-only/);
  });
  it('gives the native messages for bad arguments', async () => {
    await expect(run('third_order all regular')).rejects.toThrow('Illegal Third Order command');
    await expect(run('third_order nogroup regular 0.0001')).rejects.toThrow('Could not find third_order group ID');
    await expect(run('third_order all cubic 0.0001')).rejects.toThrow('Illegal Dynamical Matrix command');
    await expect(run('third_order all regular 0.0001 fold maybe')).rejects.toThrow('Illegal input for Third Order fold option');
    await expect(run('third_order all regular 0.0001 colour red')).rejects.toThrow('Illegal Third Order command');
  });
});
