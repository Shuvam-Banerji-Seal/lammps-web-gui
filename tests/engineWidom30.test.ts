import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix widom (src/engine/fix/widom.ts): Widom test-particle insertion. The
 * random stream (RanPark seeded with the seed, no discarded draws; three draws
 * per atom insertion; axis rejection plus an angle draw per molecule) is pinned
 * by the w30widom_* oracle cases. This file checks the documented vector
 * relations, the argument errors, the statistical limit and the region volume.
 */

const run = async (script: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { error, rows };
};

const rowAt = (rows: ThermoRow[], step: number): ThermoRow => {
  const r = rows.find((x) => x.step === step);
  if (!r) throw new Error(`no thermo row at step ${step}`);
  return r;
};

const DIMER = `# dimer

2 atoms
1 bonds

Coords

1 0.0 0.0 0.0
2 1.0 0.0 0.0

Types

1 1
2 2

Bonds

1 1 1 2
`;

const ATOM = `units lj
atom_style atomic
boundary p p p
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 single 1.0 1.0 1.0 units box
create_atoms 1 single 3.0 2.0 2.5 units box
mass 1 1.0
mass 2 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 1 2 1.0 1.0
pair_coeff 2 2 1.0 1.0
fix w all widom 2 3 2 12345 1.0
thermo_style custom step f_w[1] f_w[2] f_w[3]
thermo 1
`;

describe('fix widom', () => {
  it('reproduces the native atom-insertion stream (measured, seed 12345)', async () => {
    const { error, rows } = await run(`${ATOM}run 10\n`);
    expect(error).toBeNull();
    const r1 = rowAt(rows, 1);
    expect(r1['f_w[1]']).toBeCloseTo(-0.476598814576117, 12);
    expect(r1['f_w[2]']).toBeCloseTo(1.61058717032848, 12);
    expect(r1['f_w[3]']).toBe(64);
    // the event is refreshed at step 1 and held until step 1 + N = 3
    expect(rowAt(rows, 2)['f_w[2]']).toBe(r1['f_w[2]']);
    expect(rowAt(rows, 3)['f_w[2]']).not.toBe(r1['f_w[2]']);
  });

  it('keeps f_w[1] = -boltz T ln(f_w[2]) and the box volume', async () => {
    const { error, rows } = await run(`${ATOM}run 10\n`);
    expect(error).toBeNull();
    for (const r of rows) {
      if (r.step === 0) continue;
      const f2 = r['f_w[2]'];
      if (f2 > 0) expect(r['f_w[1]']).toBeCloseTo(-1.0 * Math.log(f2), 12);
      else expect(r['f_w[1]']).toBe(0);
      expect(r['f_w[3]']).toBe(64);
    }
  });

  it('reports f_w[2] = 1 and f_w[1] = 0 when the inserted atom does not interact', async () => {
    const script = `units lj
atom_style atomic
boundary p p p
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 single 0.5 0.5 0.5 units box
mass 1 1.0
mass 2 1.0
pair_style lj/cut 2.5
pair_coeff * * 0.0 1.0
fix w all widom 1 4 2 12345 2.0
thermo_style custom step f_w[1] f_w[2] f_w[3]
thermo 1
run 4
`;
    const { error, rows } = await run(script);
    expect(error).toBeNull();
    const r = rowAt(rows, 1);
    expect(r['f_w[2]']).toBeCloseTo(1, 12);
    expect(r['f_w[1]']).toBeCloseTo(0, 12);
    expect(r['f_w[3]']).toBe(27);
  });

  it('uses the region volume and holds the molecule count', async () => {
    const script = `units lj
atom_style molecular
boundary p p p
region box block 0 6 0 6 0 6
region ins block 1 3 1 3 1 3
create_box 2 box bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 8
create_atoms 1 single 2.0 2.0 2.0 units box
mass 1 1.0
mass 2 1.0
pair_style lj/cut 2.0
pair_coeff * * 1.0 1.0
bond_style harmonic
bond_coeff 1 50.0 1.0
set atom 1 mol 1
molecule m dimer
fix w all widom 1 2 0 12345 1.0 mol m region ins full_energy
thermo_style custom step f_w[1] f_w[2] f_w[3]
thermo 1
run 4
`;
    const { error, rows } = await run(script, { dimer: DIMER });
    expect(error).toBeNull();
    expect(rowAt(rows, 1)['f_w[3]']).toBe(8);
  });

  it('rejects the unsupported and inconsistent arguments with a StyleError', async () => {
    const cases: [string, RegExp][] = [
      [`fix w all widom 1 1 2 12345 1.0 bogus\n`, /unknown keyword 'bogus'/],
      [`fix w all widom 1 1 2 12345\n`, /usage: fix ID group-ID widom/],
      [`fix w all widom 0 1 2 12345 1.0\n`, /N must be a positive integer/],
      [`fix w all widom 1 1 2 0 1.0\n`, /seed must be a positive integer/],
      [`fix w all widom 1 1 2 12345 -1.0\n`, /T must be > 0/],
      [`fix w all widom 1 1 0 12345 1.0 mol nope\n`, /atom_style atomic cannot store molecule IDs/],
      [`fix w all widom 1 1 5 12345 1.0\n`, /atom type 5 is larger than ntypes 2/],
    ];
    for (const [fix, re] of cases) {
      const script = `units lj
atom_style atomic
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 single 0.5 0.5 0.5 units box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
${fix}thermo_style custom step
run 1
`;
      const { error } = await run(script);
      expect(error, fix).toMatch(re);
    }
  });

  it('requires molecule IDs on the group atoms when mol is used', async () => {
    const script = `units lj
atom_style molecular
region box block 0 3 0 3 0 3
create_box 2 box bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 8
create_atoms 1 single 0.5 0.5 0.5 units box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
bond_style harmonic
bond_coeff 1 50.0 1.0
molecule m dimer
fix w all widom 1 1 0 12345 1.0 mol m
thermo_style custom step
run 1
`;
    const { error } = await run(script, { dimer: DIMER });
    expect(error).toMatch(/All mol IDs should be set/);
  });

  it('rejects a region defined with side out', async () => {
    const script = `units lj
atom_style atomic
region box block 0 3 0 3 0 3
region ins block 0.5 2.5 0.5 2.5 0.5 2.5 side out
create_box 2 box
create_atoms 1 single 1.0 1.0 1.0 units box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
fix w all widom 1 1 2 12345 1.0 region ins
thermo_style custom step
run 1
`;
    const { error } = await run(script);
    expect(error).toMatch(/side in/);
  });
});
