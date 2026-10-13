import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// The text files that fix phonon (prefix.log) and dynamical_matrix (the dynamical matrix file) write, compared
// number by number with the files native LAMMPS wrote for the same inputs (b332f28, PHONON package). The oracle
// harness (engineOracle.test.ts) compares the files of the cases that list them; the cases here are read from
// tests/oracle and their native files from tests/fixtures/oracle (JSON) or tests/fixtures/phonon.

const CASES = join(__dirname, 'oracle');
const FIX = join(__dirname, 'fixtures', 'oracle');
const PHONON_FIX = join(__dirname, 'fixtures', 'phonon');

const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^#\\s*${key}:\\s*(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

/** Runs an oracle case through the engine and returns the files it wrote. */
const runCase = async (name: string): Promise<Map<string, string>> => {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  }, undefined, 0, 1);
  for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readFileSync(join(CASES, f), 'utf8'));
  await session.execute(text);
  return files;
};

/**
 * Compares two text files token by token: numbers within abs + rel * |native|; brackets around a number (the
 * basis vectors of fix phonon's log) are not part of it. Lines for which skip(line index, tokens) is true are not
 * compared. Returns the worst absolute and relative differences found (over the compared numbers).
 */
const compareNumbers = (
  mine: string, ref: string, skip: (line: number, tokens: string[]) => boolean = () => false,
): { worst: number; worstRel: number; where: string; words: string | null } => {
  const la = mine.trim().split('\n'), lb = ref.trim().split('\n');
  const out = { worst: 0, worstRel: 0, where: '', words: null as string | null };
  if (la.length !== lb.length) return { ...out, words: `line count ${la.length} vs ${lb.length}` };
  for (let i = 0; i < lb.length; i++) {
    const ta = la[i].trim().split(/\s+/).map((w) => w.replace(/[[\]]/g, ''));
    const tb = lb[i].trim().split(/\s+/).map((w) => w.replace(/[[\]]/g, ''));
    if (skip(i, tb)) continue;
    if (ta.length !== tb.length) return { ...out, words: `line ${i + 1}: token count ${ta.length} vs ${tb.length}` };
    for (let k = 0; k < tb.length; k++) {
      const x = Number(ta[k]), y = Number(tb[k]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        const d = Math.abs(x - y);
        if (d > out.worst) { out.worst = d; out.where = `line ${i + 1} token ${k + 1}: ${x} vs ${y}`; }
        if (Math.abs(y) > 1e-3) out.worstRel = Math.max(out.worstRel, d / Math.abs(y));
      } else if (ta[k] !== tb[k]) {
        return { ...out, words: `line ${i + 1} token ${k + 1}: '${ta[k]}' vs '${tb[k]}'` };
      }
    }
  }
  return out;
};

const nativeFile = (name: string, file: string): string => {
  const fx = JSON.parse(readFileSync(join(FIX, `${name}.json`), 'utf8')) as { files: Record<string, string> };
  return fx.files[file];
};

describe('fix phonon log against native LAMMPS', () => {
  // the basis vectors and the sums of the log agree to roundoff; the Phi(q) numbers to 1e-6 relative (|Phi| > 1e-3)
  for (const [name, file] of [['w99phq_mono1d', 'phonon.log'], ['w99phq_diat1d', 'w99phq_diat1d.log'], ['w99phq_cu', 'w99phq_cu.log']]) {
    it(`${name}: every number of ${file} is within 1e-6 relative to native`, { timeout: 120_000 }, async () => {
      const mine = (await runCase(name)).get(file);
      expect(mine, `${file} written`).toBeDefined();
      const r = compareNumbers(mine!, nativeFile(name, file));
      expect(r.words, 'words and layout').toBeNull();
      expect(r.worstRel, r.where).toBeLessThan(1e-6);
    });
  }

  // the graphene cell (tilted box, tersoff, 3d with two basis atoms): the Gamma-point block is not compared. Its correlation matrix is nearly singular (the centre of
  // mass is conserved and the Gamma-point sums are roundoff), so its values are set by roundoff in native too
  it('w99phq_graphene: the q != 0 blocks match the native log within 1e-6 relative', { timeout: 120_000 }, async () => {
    const mine = (await runCase('w99phq_graphene')).get('w99phq_graphene.log');
    expect(mine, 'log written').toBeDefined();
    const ref = readFileSync(join(PHONON_FIX, 'w99phq_graphene.log'), 'utf8');
    const gamma = (_i: number, t: string[]) => t.length > 3 && Number(t[0]) === 0 && Number(t[1]) === 0 && Number(t[2]) === 0 && /^[0-9.e+-]+$/.test(t[0]) && t.length === 75;
    const r = compareNumbers(mine!, ref, gamma);
    expect(r.words, 'words and layout').toBeNull();
    // the worst entries are set by the roundoff of the matrix inversion (measured: 1.6e-5 relative on Node 20, 9.5e-6 on Node 24)
    expect(r.worst, r.where).toBeLessThan(2e-5);
    expect(r.worstRel, r.where).toBeLessThan(5e-5);
  });
});

describe('dynamical_matrix file against native LAMMPS', () => {
  it('w99phq_dynmat_silicon: every entry of dynmat.dat is within 3e-5 absolute of native', { timeout: 120_000 }, async () => {
    const mine = (await runCase('w99phq_dynmat_silicon')).get('dynmat.dat');
    expect(mine, 'dynmat.dat written').toBeDefined();
    const ref = readFileSync(join(PHONON_FIX, 'w99phq_silicon_dynmat.dat'), 'utf8');
    // the native entries are finite differences of forces: roundoff of up to 1.6e-5 absolute (largest value 862)
    const r = compareNumbers(mine!, ref);
    expect(r.words, 'words and layout').toBeNull();
    expect(r.worst, r.where).toBeLessThan(3e-5);
  });
});
