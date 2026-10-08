import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Wave-11 per-atom structure computes (src/engine/compute/struct.ts):
 * centro/atom, cna/atom, cluster/atom, fragment/atom, aggregate/atom.
 * Oracle parity is in tests/oracle/w11struct_*.in; this file holds the
 * closed-form checks and argument errors.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

/** Perfect fcc, 2x2x2 cells, metal units, pair cutoff 4.5 A (nearest-neighbour shell at a/sqrt(2) = 2.864 A). */
const FCC = `
units metal
atom_style atomic
lattice fcc 4.05
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 26.98
pair_style lj/cut 4.5
pair_coeff * * 0.01 2.5
`;

/** Runs `text` and returns the thrown error (undefined if none). */
const errorOf = async (text: string): Promise<unknown> => {
  try {
    await runScript(text);
  } catch (e) {
    return e;
  }
  return undefined;
};

describe('compute centro/atom', () => {
  it('gives 0 for every atom of a perfect fcc crystal (all six opposite pairs are exact)', async () => {
    const s = await runScript(`${FCC}\ncompute c all centro/atom fcc\nrun 0\n`);
    const vals = s.sys.compute('c').peratomValues();
    expect(vals.length).toBe(s.sys.state.n);
    for (let i = 0; i < vals.length; i++) expect(Math.abs(vals[i])).toBeLessThan(1e-12);
  });

  it('gives 10 columns with axes yes: unit axes, the third one orthogonal to the first two', async () => {
    const s = await runScript(`${FCC}\ncompute c all centro/atom fcc axes yes\nrun 0\n`);
    const c = s.sys.compute('c');
    expect(c.peratomFlag).toBe(true);
    expect(c.sizePeratomCols).toBe(10);
    const a = c.peratomValues();
    for (let i = 0; i < s.sys.state.n; i++) {
      const o = 10 * i;
      const ax1 = [a[o + 1], a[o + 2], a[o + 3]];
      const ax2 = [a[o + 4], a[o + 5], a[o + 6]];
      const ax3 = [a[o + 7], a[o + 8], a[o + 9]];
      const dot = (u: number[], v: number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      expect(dot(ax1, ax1)).toBeCloseTo(1, 12);
      expect(dot(ax2, ax2)).toBeCloseTo(1, 12);
      expect(dot(ax3, ax3)).toBeCloseTo(1, 12);
      expect(Math.abs(dot(ax1, ax3))).toBeLessThan(1e-12);
      expect(Math.abs(dot(ax2, ax3))).toBeLessThan(1e-12);
    }
  });

  it('rejects an odd or non-positive N and an unknown lattice keyword', async () => {
    let e = await errorOf(`${FCC}\ncompute c all centro/atom 5\nrun 0\n`);
    expect((e as Error).message).toMatch(/even integer/);
    e = await errorOf(`${FCC}\ncompute c all centro/atom hcp\nrun 0\n`);
    expect((e as Error).message).toMatch(/lattice must be fcc, bcc or N/);
    e = await errorOf(`${FCC}\ncompute c all centro/atom fcc axes maybe\nrun 0\n`);
    expect((e as Error).message).toMatch(/axes must be yes or no/);
    e = await errorOf(`${FCC}\ncompute c all centro/atom fcc foo\nrun 0\n`);
    expect((e as Error).message).toMatch(/unknown keyword 'foo'/);
  });
});

describe('compute cna/atom', () => {
  it('classifies every atom of a perfect fcc crystal as fcc (1)', async () => {
    const s = await runScript(`${FCC}\ncompute n all cna/atom 3.4\nrun 0\n`);
    const vals = s.sys.compute('n').peratomValues();
    expect(vals.length).toBe(s.sys.state.n);
    for (let i = 0; i < vals.length; i++) expect(vals[i]).toBe(1);
  });

  it('gives 0 for atoms outside the group and needs exactly one cutoff argument', async () => {
    const s = await runScript(`${FCC}\ngroup half id 1:4\ncompute n half cna/atom 3.4\nrun 0\n`);
    const vals = s.sys.compute('n').peratomValues();
    const mask = s.sys.state.mask;
    const bit = s.sys.groupBit('half');
    for (let i = 0; i < s.sys.state.n; i++) {
      if (!(mask[i] & bit)) expect(vals[i]).toBe(0);
    }
    let e = await errorOf(`${FCC}\ncompute n all cna/atom\nrun 0\n`);
    expect((e as Error).message).toMatch(/usage: compute ID group-ID cna\/atom cutoff/);
    e = await errorOf(`${FCC}\ncompute n all cna/atom -1.0\nrun 0\n`);
    expect((e as Error).message).toMatch(/cutoff must be > 0/);
  });
});

describe('compute cluster/atom, fragment/atom, aggregate/atom', () => {
  it('labels a perfect fcc crystal with one cluster whose ID is the smallest atom ID', async () => {
    const s = await runScript(`${FCC}\ncompute cl all cluster/atom 3.0\nrun 0\n`);
    const vals = s.sys.compute('cl').peratomValues();
    for (let i = 0; i < vals.length; i++) expect(vals[i]).toBe(1);
  });

  it('gives one-atom clusters their own ID when the cutoff is below the nearest-neighbour distance', async () => {
    const s = await runScript(`${FCC}\ncompute cl all cluster/atom 1.0\nrun 0\n`);
    const vals = s.sys.compute('cl').peratomValues();
    const ids = s.sys.state.id;
    for (let i = 0; i < vals.length; i++) expect(vals[i]).toBe(ids[i]);
  });

  it('fragment/atom gives 0 to unbonded atoms unless single yes; aggregate/atom needs a bonded atom style', async () => {
    const bonded = `
units metal
atom_style bond
lattice fcc 4.05
region box block 0 2 0 2 0 2
create_box 1 box bond/types 1 extra/bond/per/atom 12
create_atoms 1 box
mass 1 26.98
bond_style harmonic
bond_coeff 1 1.0 2.9
pair_style lj/cut 4.5
pair_coeff * * 0.01 2.5
create_bonds single/bond 1 1 2
`;
    const s = await runScript(`${bonded}\ncompute fr all fragment/atom\ncompute fs all fragment/atom single yes\nrun 0\n`);
    const fr = s.sys.compute('fr').peratomValues();
    const fs = s.sys.compute('fs').peratomValues();
    const ids = s.sys.state.id;
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] === 1 || ids[i] === 2) expect(fr[i]).toBe(1);
      else expect(fr[i]).toBe(0);
      expect(fs[i]).toBe(ids[i] === 1 || ids[i] === 2 ? 1 : ids[i]);
    }
    const e = await errorOf(`${FCC}\ncompute ag all aggregate/atom 3.0\nrun 0\n`);
    expect((e as Error).message).toMatch(/bonds are not allowed/);
  });

  it('rejects unknown fragment keywords', async () => {
    const e = await errorOf(`${FCC}\ncompute fr all fragment/atom bogus\nrun 0\n`);
    expect((e as Error).message).toMatch(/unknown keyword 'bogus'/);
  });
});
