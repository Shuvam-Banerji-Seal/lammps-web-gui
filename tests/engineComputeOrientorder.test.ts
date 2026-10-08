import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { ComputeOrientorderAtom } from '../src/engine/compute/orientorder';
import { EngineError } from '../src/engine/types';
import { StyleError } from '../src/engine/force/types';
import type { EngineEvent } from '../src/engine/types';

/** Relative closeness for checks against documented reference values. */
const close = (got: number, want: number, rel = 1e-9) => Math.abs(got - want) <= rel * Math.max(Math.abs(got), Math.abs(want));

/** Runs input text in a fresh session. */
const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

/**
 * Perfect FCC (lattice fcc 0.8442, 2x2x2 cells, no displacement): every atom
 * has its 12 nearest neighbours at a/sqrt(2) and the next shell at a, so
 * nnn = 12 selects the ideal first shell and the documented crystal values
 * apply (docs.lammps.org/compute_orientorder_atom.html):
 *   Q_4 = sqrt(7/192) = 0.19094065395649334
 *   W_4 = -sqrt(14/143)(49/4096)pi^{-3/2} = -0.0006722136424160106
 *   What_4 = -(7/3)sqrt(2/429) = -0.15931737313308109
 */
const FCC = `
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
run 0
`;

const Q4_FCC = Math.sqrt(7 / 192);
const Q6_FCC = 0.5745242597130389; // Mickel et al. Table I value for FCC
const WINV4_FCC = -Math.sqrt(14 / 143) * (49 / 4096) * Math.PI ** -1.5; // doc W4 (invariant without the factor)
const W4_FCC = WINV4_FCC / 3; // wl column = invariant / sqrt(2l+1), measured on the oracle case
const WHAT4_FCC = -(7 / 3) * Math.sqrt(2 / 429);

describe('compute orientorder/atom', () => {
  it('reproduces the documented FCC Q4/Q6/W4/What4 values', async () => {
    const s = await runScript(`
      ${FCC}
      compute q all orientorder/atom
      compute w all orientorder/atom wl yes wl/hat yes
      run 0
    `);
    const q = s.sys.compute('q');
    expect(q.style).toBe('orientorder/atom');
    expect(q.peratomFlag).toBe(true);
    expect(q.sizePeratomCols).toBe(5);
    const vals = q.peratomValues();
    const n = s.sys.state.n;
    expect(n).toBe(32);
    for (let i = 0; i < n; i++) {
      expect(close(vals[5 * i], Q4_FCC)).toBe(true);
      expect(close(vals[5 * i + 1], Q6_FCC, 1e-7)).toBe(true);
      expect(vals[5 * i]).toBeGreaterThanOrEqual(0);
      expect(vals[5 * i]).toBeLessThanOrEqual(1);
    }
    const w = s.sys.compute('w');
    // 5 Q columns + 5 W columns + 5 What columns
    expect(w.sizePeratomCols).toBe(15);
    const wv = w.peratomValues();
    for (let i = 0; i < n; i++) {
      expect(close(wv[15 * i], Q4_FCC)).toBe(true);
      expect(close(wv[15 * i + 5], W4_FCC, 1e-7)).toBe(true);
      expect(close(wv[15 * i + 10], WHAT4_FCC, 1e-7)).toBe(true);
    }
  });

  it('honors degrees and nnn NULL with a cutoff (all neighbours within the cutoff)', async () => {
    const s = await runScript(`
      ${FCC}
      compute q2 all orientorder/atom degrees 3 4 6 8 nnn NULL cutoff 1.2
      run 0
    `);
    const q = s.sys.compute('q2');
    expect(q.sizePeratomCols).toBe(3);
    const vals = q.peratomValues();
    const n = s.sys.state.n;
    // cutoff 1.2 keeps exactly the 12 first-shell neighbours: same Q as nnn 12
    for (let i = 0; i < n; i++) {
      expect(close(vals[3 * i], Q4_FCC)).toBe(true);
      expect(close(vals[3 * i + 1], Q6_FCC, 1e-7)).toBe(true);
    }
  });

  it('zeroes atoms with fewer than nnn neighbours within the cutoff', async () => {
    const s = await runScript(`
      ${FCC}
      compute q all orientorder/atom cutoff 1.0
      compute qnull all orientorder/atom nnn NULL cutoff 1.0
      run 0
    `);
    const vals = s.sys.compute('q').peratomValues();
    const vnull = s.sys.compute('qnull').peratomValues();
    const n = s.sys.state.n;
    for (let i = 0; i < n; i++) {
      expect(vals[5 * i]).toBe(0); // 0 neighbours < nnn 12
      expect(vnull[5 * i]).toBe(0); // no neighbours at all
    }
  });

  it('zeroes rows for atoms outside the compute group', async () => {
    const s = await runScript(`
      ${FCC}
      region upper block INF INF INF INF 1 INF units box
      group half region upper
      compute q half orientorder/atom
      run 0
    `);
    const st = s.sys.state;
    const vals = s.sys.compute('q').peratomValues();
    const half = s.sys.groupBit('half');
    let nHalf = 0;
    for (let i = 0; i < st.n; i++) {
      if (st.mask[i] & half) {
        nHalf++;
        expect(close(vals[5 * i], Q4_FCC)).toBe(true);
      } else {
        expect(vals[5 * i]).toBe(0);
      }
    }
    expect(nHalf).toBeGreaterThan(0);
    expect(nHalf).toBeLessThan(st.n);
  });

  it('outputs normalized unit components in documented order', async () => {
    const s = await runScript(`
      ${FCC}
      compute c all orientorder/atom components 4
      run 0
    `);
    const c = s.sys.compute('c');
    // 5 Q columns + 2*(2*4+1) component columns
    expect(c.sizePeratomCols).toBe(5 + 18);
    const vals = c.peratomValues();
    const n = s.sys.state.n;
    for (let i = 0; i < n; i++) {
      expect(close(vals[23 * i], Q4_FCC)).toBe(true);
      let norm2 = 0;
      for (let k = 0; k < 18; k++) norm2 += vals[23 * i + 5 + k] ** 2;
      expect(close(norm2, 1, 1e-9)).toBe(true);
    }
  });

  it('accepts chunksize (KOKKOS-only, ignored)', async () => {
    const s = await runScript(`
      ${FCC}
      compute q all orientorder/atom chunksize 7
      run 0
    `);
    const vals = s.sys.compute('q').peratomValues();
    expect(close(vals[0], Q4_FCC)).toBe(true);
  });

  it('throws StyleError for bad arguments', async () => {
    const cases: [string, RegExp][] = [
      ['compute q all orientorder/atom bogus 1', /unknown keyword 'bogus'/],
      ['compute q all orientorder/atom wl maybe', /wl must be yes or no/],
      ['compute q all orientorder/atom degrees 3 4 6', /degrees needs 3 degrees/],
      ['compute q all orientorder/atom degrees 2 4 -6', /non-negative integers/],
      ['compute q all orientorder/atom nnn 0', /nnn must be a positive integer or NULL/],
      ['compute q all orientorder/atom components 5', /components degree 5 must be included/],
      ['compute q all orientorder/atom chunksize 0', /chunksize must be a positive integer/],
    ];
    for (const [cmd, re] of cases) {
      let err: unknown;
      try {
        await runScript(`${FCC}\n${cmd}\nrun 0\n`);
      } catch (e) {
        err = e;
      }
      expect(err, cmd).toBeInstanceOf(EngineError);
      expect((err as Error).message, cmd).toMatch(re);
    }
  });

  it('rejects a cutoff above the pair style cutoff', async () => {
    let err: unknown;
    try {
      const s = await runScript(`${FCC}\ncompute q all orientorder/atom cutoff 3.0\nrun 0\n`);
      s.sys.compute('q').peratomValues();
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(StyleError);
    expect((err as Error).message).toMatch(/exceeds the pair style cutoff/);
  });

  it('rejects a missing pair cutoff at evaluation', async () => {
    let err: unknown;
    try {
      const s = await runScript(`
        units lj
        atom_style atomic
        lattice sc 1.0
        region box block 0 2 0 2 0 2
        create_box 1 box
        create_atoms 1 box
        mass 1 1.0
        compute q all orientorder/atom
        run 0
      `);
      s.sys.compute('q').peratomValues();
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(StyleError);
    expect((err as Error).message).toMatch(/no pair style cutoff is defined/);
  });

  it('constructs with documented defaults', async () => {
    const session = await runScript(FCC);
    const q = new ComputeOrientorderAtom(session.sys, 'qq', 'all', []);
    expect(q.style).toBe('orientorder/atom');
    expect(q.sizePeratomCols).toBe(5); // default degrees 4 6 8 10 12
  });
});
