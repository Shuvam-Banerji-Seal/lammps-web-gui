import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { Fix } from '../src/engine/fix/fix';

/*
 * fix numdiff and fix numdiff/virial (docs.lammps.org/fix_numdiff.html,
 * docs.lammps.org/fix_numdiff_virial.html). Everything is checked on tiny
 * lj/cut systems whose analytic forces and virial can be computed by hand; the
 * native parity itself is in tests/oracle/w16numdiff_*.in.
 */

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  let error: Error | null = null;
  try { await session.execute(text); } catch (e) { error = e as Error; }
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { sys: session.sys, rows, error };
};

const HEADER = `
units           lj
atom_style      atomic
boundary        p p p
region          box block 0 8 0 8 0 8
create_box      1 box
create_atoms    1 single 3.0 4.0 4.0
create_atoms    1 single 4.5 4.0 4.0
create_atoms    1 single 4.0 5.2 4.0
create_atoms    1 single 4.3 3.3 4.7
mass            1 1.0
pair_style      lj/cut 3.0
pair_coeff      1 1 1.0 1.0
`;

const maxAbs = (a: Float64Array, b: Float64Array): number => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

describe('fix numdiff', () => {
  it('finite-difference forces equal the analytic forces with O(delta^2) error', async () => {
    const errs: number[] = [];
    for (const delta of [1e-4, 1e-5, 1e-6]) {
      const { sys, error } = await run(`${HEADER}
pair_modify tail yes
fix fd all numdiff 1 ${delta}
run 0`);
      expect(error?.message ?? '').toBe('');
      errs.push(maxAbs(sys.fix('fd').arrayAtom, sys.state.f));
    }
    // truncation shrinks as delta^2 (factor ~100 per decade of delta)
    expect(errs[1]).toBeLessThan(errs[0]);
    expect(errs[2]).toBeLessThan(errs[1]);
    expect(errs[0] / errs[1]).toBeGreaterThan(30);
    expect(errs[0] / errs[1]).toBeLessThan(300);
    expect(errs[2]).toBeLessThan(1e-8);
  });

  it('is exact to roundoff at a moderate delta and matches the sign of the analytic force', async () => {
    const { sys, error } = await run(`${HEADER}
fix fd all numdiff 1 1e-6
run 0`);
    expect(error?.message ?? '').toBe('');
    const fd = sys.fix('fd').arrayAtom;
    const analytic = sys.state.f;
    for (let i = 0; i < analytic.length; i++) expect(fd[i]).toBeCloseTo(analytic[i], 8);
  });

  it('does not perturb a run: thermo is bit-identical with and without the fix', async () => {
    const common = `${HEADER}
fix nve all nve
thermo_style custom step temp pe ke etotal press
thermo_modify format float %.15g
run 20`;
    const a = await run(common);
    const b = await run(`${HEADER}
fix fd all numdiff 1 1e-6
fix nve all nve
thermo_style custom step temp pe ke etotal press
thermo_modify format float %.15g
run 20`);
    expect(b.error?.message ?? '').toBe('');
    expect(b.rows).toEqual(a.rows);
    // the restored forces are still the analytic ones of the final step
    expect(maxAbs(b.sys.fix('fd').arrayAtom, b.sys.state.f)).toBeLessThan(1e-7);
  });

  it('sets 0.0 for atoms outside the fix group', async () => {
    const { sys, error } = await run(`${HEADER}
group g id 1 2
fix fd g numdiff 1 1e-6
run 0`);
    expect(error?.message ?? '').toBe('');
    const fd = sys.fix('fd').arrayAtom;
    const mask = sys.state.mask;
    const bit = sys.groups.bit('g');
    for (let i = 0; i < sys.state.n; i++) {
      if (mask[i] & bit) continue;
      expect([fd[3 * i], fd[3 * i + 1], fd[3 * i + 2]]).toEqual([0, 0, 0]);
    }
  });

  it('produces a per-atom array with three columns', async () => {
    const { sys } = await run(`${HEADER}
fix fd all numdiff 1 1e-6
run 0`);
    const f: Fix = sys.fix('fd');
    expect(f.peratomFlag).toBe(true);
    expect(f.sizePeratomCols).toBe(3);
    expect(f.arrayAtom.length).toBe(3 * sys.state.n);
  });

  it('rejects malformed arguments with a style error', async () => {
    expect((await run(`${HEADER}
fix fd all numdiff 1`)).error?.message).toMatch(/expected exactly Nevery and delta/);
    expect((await run(`${HEADER}
fix fd all numdiff 0 1e-6`)).error?.message).toMatch(/Nevery must be a positive integer/);
    expect((await run(`${HEADER}
fix fd all numdiff 1.5 1e-6`)).error?.message).toMatch(/Nevery must be a positive integer/);
    expect((await run(`${HEADER}
fix fd all numdiff 1 -1e-6`)).error?.message).toMatch(/delta must be a positive number/);
    expect((await run(`${HEADER}
fix fd all numdiff 1 1e-6 extra`)).error?.message).toMatch(/expected exactly Nevery and delta/);
  });
});

describe('fix numdiff/virial', () => {
  it('finite-difference virial equals compute pressure (virial) with the Voigt ordering', async () => {
    const { sys, error } = await run(`${HEADER}
pair_modify tail no
fix fd all numdiff/virial 1 1e-6
compute p all pressure NULL virial
run 0`);
    expect(error?.message ?? '').toBe('');
    const fd = sys.fix('fd') as Fix;
    const p = sys.compute('p').vectorValues();
    // fd order xx yy zz yz xz xy, pressure order xx yy zz xy xz yz
    const map = [1, 2, 3, 6, 5, 4];
    for (let i = 0; i < 6; i++) {
      expect(fd.computeVector(i)).toBeCloseTo(p[map[i] - 1], 7);
    }
  });

  it('excludes the pair tail correction (native measured behaviour)', async () => {
    const withTail = await run(`${HEADER}
pair_modify tail yes
fix fd all numdiff/virial 1 1e-6
compute p all pressure NULL virial
run 0`);
    const noTail = await run(`${HEADER}
pair_modify tail no
fix fd all numdiff/virial 1 1e-6
compute p all pressure NULL virial
run 0`);
    expect(withTail.error?.message ?? '').toBe('');
    // the finite-difference value is unchanged by tail yes/no ...
    for (let i = 0; i < 6; i++) {
      expect((withTail.sys.fix('fd') as Fix).computeVector(i))
        .toBeCloseTo((noTail.sys.fix('fd') as Fix).computeVector(i), 9);
    }
    // ... while compute pressure changes with the tail
    const a = withTail.sys.compute('p').vectorValues();
    const b = noTail.sys.compute('p').vectorValues();
    expect(Math.abs(a[0] - b[0])).toBeGreaterThan(1e-6);
  });

  it('requires group all', async () => {
    const { error } = await run(`${HEADER}
group g id 1
fix fd g numdiff/virial 1 1e-6
run 0`);
    expect(error?.message).toMatch(/requires group all/);
  });

  it('does not perturb a run and restores the analytic virial state', async () => {
    const a = await run(`${HEADER}
compute p all pressure NULL virial
fix nve all nve
thermo_style custom step pe press c_p[1] c_p[2] c_p[3]
thermo_modify format float %.15g
run 10`);
    const b = await run(`${HEADER}
compute p all pressure NULL virial
fix fd all numdiff/virial 1 1e-6
fix nve all nve
thermo_style custom step pe press c_p[1] c_p[2] c_p[3]
thermo_modify format float %.15g
run 10`);
    expect(b.error?.message ?? '').toBe('');
    expect(b.rows).toEqual(a.rows);
  });
});
