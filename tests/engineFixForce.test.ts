import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Direct checks for the force-modifying fixes (setforce, addforce, aveforce)
 * on a tiny two-atom system with no pair style (zero initial forces), so
 * every documented formula can be computed by hand.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  let error: Error | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as Error;
  }
  const rows = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { session, rows, error, events };
};

const BASE = `
units           lj
atom_style      atomic
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 1 1 1
create_atoms    1 single 4 1 1
mass            1 1.0
`;

describe('fix setforce', () => {
  it('sets the given components, leaves NULL ones alone, and tallies the pre-fix group force', async () => {
    const { session, rows, error } = await runScript(`${BASE}
fix s1 all setforce 0.0 0.0 2.0
fix s2 all setforce 0.0 1.5 NULL
thermo_style custom step f_s2[1] f_s2[2] f_s2[3]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const s = session.system!;
    // s1 ran first (fz = 2), then s2 set fy = 1.5 and left fx, fz alone (NULL)
    expect([...s.f.slice(0, 3)]).toEqual([0, 1.5, 2]);
    expect([...s.f.slice(3, 6)]).toEqual([0, 1.5, 2]);
    // f_s2 is the force BEFORE s2 changed it: fx = 0, fy = 0, fz = 2 + 2
    const r = rows[rows.length - 1];
    expect(r['f_s2[1]']).toBe(0);
    expect(r['f_s2[2]']).toBe(0);
    expect(r['f_s2[3]']).toBe(4);
  });

  it('honours the region keyword: atoms outside it keep their forces', async () => {
    const { session, error } = await runScript(`${BASE}
create_atoms    1 single 7 1 5
region lowz block INF INF INF INF INF 2 units box
fix 1 all setforce 0.0 5.0 NULL region lowz
run 0`);
    expect(error?.message ?? '').toBe('');
    const s = session.system!;
    expect([...s.f.slice(0, 3)]).toEqual([0, 5, 0]);   // z = 1, inside lowz
    expect([...s.f.slice(3, 6)]).toEqual([0, 5, 0]);   // z = 1, inside lowz
    expect([...s.f.slice(6, 9)]).toEqual([0, 0, 0]);   // z = 5, outside lowz
  });

  it('rejects bad arguments', async () => {
    expect((await runScript(`${BASE}\nfix 1 all setforce 0.0 0.0`)).error?.message).toMatch(/usage: fix ID group setforce/);
    expect((await runScript(`${BASE}\nfix 1 all setforce 0 0 0 foo 1`)).error?.message).toMatch(/unknown fix setforce keyword 'foo'/);
    expect((await runScript(`${BASE}\nfix 1 all setforce 0 0 0 region`)).error?.message).toMatch(/keyword 'region' needs a value/);
    expect((await runScript(`${BASE}\nfix 1 all setforce 0 0 v_nope`)).error?.message).toMatch(/variable nope does not exist/);
    expect((await runScript(`${BASE}\nfix 1 all setforce 0 0 0 region noregion`)).error?.message).toMatch(/region ID 'noregion' does not exist/);
    expect((await runScript(`${BASE}\nfix 1 all setforce 0 0 yes`)).error?.message).toMatch(/expected a number, NULL or v_name/);
  });

  it('errors on a non-zero force during minimization, and allows zero', async () => {
    const bad = await runScript(`${BASE}
fix 1 all setforce 0.0 1.0 0.0
minimize 1e-4 1e-4 10 10`);
    expect(bad.error?.message).toMatch(/non-zero value during minimization/);
    const ok = await runScript(`${BASE}
fix 1 all setforce 0.0 0.0 0.0
minimize 1e-4 1e-4 10 10`);
    expect(ok.error?.message ?? '').toBe('');
  });
});

describe('fix addforce', () => {
  it('adds a constant force and reports E = -x.F as its scalar / fix energy', async () => {
    const { session, rows, error } = await runScript(`${BASE}
fix 1 all addforce 1.0 -0.5 0.25
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const s = session.system!;
    expect([...s.f.slice(0, 3)]).toEqual([1, -0.5, 0.25]);
    expect([...s.f.slice(3, 6)]).toEqual([1, -0.5, 0.25]);
    // E = -(x fx + y fy + z fz) = -[(0.75) + (3.75)] = -4.5; f_1[1] is the pre-fix sum
    const r = rows[rows.length - 1];
    expect(r.pe).toBeCloseTo(-4.5, 12);
    expect(r.f_1).toBeCloseTo(-4.5, 12);
    expect(r['f_1[1]']).toBe(0);
  });

  it('fix_modify energy no keeps the energy out of pe but f_1 still reports it', async () => {
    const { rows, error } = await runScript(`${BASE}
fix 1 all addforce 1.0 -0.5 0.25
thermo_style custom step pe f_1
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    expect(r.pe).toBe(0);
    expect(r.f_1).toBeCloseTo(-4.5, 12);
  });

  it('evaluates atom-style variables per atom and tallies the energy variable', async () => {
    const { session, rows, error } = await runScript(`${BASE}
variable fxv atom 0.1*x
variable ea atom 0.05*x*x
fix 1 all addforce v_fxv 0.0 0.0 energy v_ea
thermo_style custom step f_1 f_1[1]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const s = session.system!;
    // -dE/dx = -0.1 x = fx (the documented consistency requirement)
    expect(s.f[0]).toBeCloseTo(0.1, 12);
    expect(s.f[3]).toBeCloseTo(0.4, 12);
    const r = rows[rows.length - 1];
    expect(r.f_1).toBeCloseTo(0.05 * (1 + 16), 12); // sum of v_ea over the group
    expect(r['f_1[1]']).toBe(0);
  });

  it('applies the force only every N steps with the every keyword', async () => {
    const gated = await runScript(`${BASE}
fix 1 all nve
fix 2 all addforce 0.0 0.0 1.0 every 2
run 1`);
    expect(gated.error?.message ?? '').toBe('');
    expect([...gated.session.system!.f.slice(0, 3)]).toEqual([0, 0, 0]); // step 1: 1 % 2 != 0
    const ungated = await runScript(`${BASE}
fix 1 all nve
fix 2 all addforce 0.0 0.0 1.0
run 1`);
    expect(ungated.error?.message ?? '').toBe('');
    expect([...ungated.session.system!.f.slice(0, 3)]).toEqual([0, 0, 1]);
  });

  it('rejects bad arguments', async () => {
    expect((await runScript(`${BASE}\nfix 1 all addforce 1.0 NULL 0.0`)).error?.message).toMatch(/NULL is not valid/);
    expect((await runScript(`${BASE}\nvariable e atom 0.5*x*x\nfix 1 all addforce 1.0 0.0 0.0 energy v_e`)).error?.message)
      .toMatch(/energy keyword is not allowed when all force components are constants/);
    expect((await runScript(`${BASE}\nfix 1 all addforce 1.0 0.0 0.0 every 0`)).error?.message).toMatch(/every must be a positive integer/);
    expect((await runScript(`${BASE}\nfix 1 all addforce 1.0 0.0 0.0 energy 1.0`)).error?.message).toMatch(/energy must be v_name/);
    expect((await runScript(`${BASE}\nvariable e equal 0.5\nfix 1 all addforce v_e 0.0 0.0 energy v_e`)).error?.message)
      .toMatch(/energy variable e must be atom-style/);
    expect((await runScript(`${BASE}\nfix 1 all addforce v_nope 0.0 0.0`)).error?.message).toMatch(/variable nope does not exist/);
  });

  it('requires the energy keyword for variable forces during minimization', async () => {
    const bad = await runScript(`${BASE}
variable fxv atom 0.1*x
fix 1 all addforce v_fxv 0.0 0.0
minimize 1e-4 1e-4 10 10`);
    expect(bad.error?.message).toMatch(/energy keyword is required/);
  });
});

describe('fix aveforce', () => {
  it('sets each component to the group average plus the given value; NULL leaves it alone', async () => {
    const { session, rows, error } = await runScript(`${BASE}
group a id 1
group b id 2
fix s1 a setforce 1.0 0.9 0.7
fix s2 b setforce 3.0 -0.9 -0.3
fix av all aveforce NULL NULL 0.1
thermo_style custom step f_av[1] f_av[2] f_av[3]
thermo_modify norm no
run 0`);
    expect(error?.message ?? '').toBe('');
    const s = session.system!;
    // pre-fix forces (1, 0.9, 0.7) and (3, -0.9, -0.3): average (2, 0, 0.2)
    expect(s.f[0]).toBe(1);
    expect(s.f[1]).toBeCloseTo(0.9, 12);
    expect(s.f[2]).toBeCloseTo(0.3, 12); // NULL x,y; z = 0.2 + 0.1
    expect(s.f[3]).toBe(3);
    expect(s.f[4]).toBeCloseTo(-0.9, 12);
    expect(s.f[5]).toBeCloseTo(0.3, 12);
    // f_av is the pre-fix total force on the group, not the average
    const r = rows[rows.length - 1];
    expect(r['f_av[1]']).toBeCloseTo(4, 12);
    expect(r['f_av[2]']).toBeCloseTo(0, 12);
    expect(r['f_av[3]']).toBeCloseTo(0.4, 12);
  });

  it('0.0 is not NULL: every atom gets the average (plus 0)', async () => {
    const { session, error } = await runScript(`${BASE}
group a id 1
group b id 2
fix s1 a setforce 1.0 0.9 0.7
fix s2 b setforce 3.0 -0.9 -0.3
fix av all aveforce 0.0 0.0 0.0
run 0`);
    expect(error?.message ?? '').toBe('');
    const s = session.system!;
    expect(s.f[0]).toBeCloseTo(2, 12);
    expect(s.f[1]).toBeCloseTo(0, 12);
    expect(s.f[2]).toBeCloseTo(0.2, 12);
    expect(s.f[3]).toBeCloseTo(2, 12);
    expect(s.f[4]).toBeCloseTo(0, 12);
    expect(s.f[5]).toBeCloseTo(0.2, 12);
  });

  it('rejects an atom-style variable (the page documents equal-style only)', async () => {
    const { error } = await runScript(`${BASE}
variable fxv atom 0.1*x
fix 1 all aveforce v_fxv 0.0 0.0`);
    expect(error?.message).toMatch(/must be equal-style/);
  });
});

describe('fix_modify on force-modifying fixes', () => {
  it('setforce does not support fix_modify energy', async () => {
    const { error } = await runScript(`${BASE}
fix 1 all setforce 0 0 0
fix_modify 1 energy yes`);
    expect(error?.message).toMatch(/does not support fix_modify energy/);
  });

  it('fix_modify energy needs yes or no', async () => {
    const { error } = await runScript(`${BASE}
fix 1 all addforce 0 0 0
fix_modify 1 energy maybe`);
    expect(error?.message).toMatch(/must be yes or no/);
  });
});
