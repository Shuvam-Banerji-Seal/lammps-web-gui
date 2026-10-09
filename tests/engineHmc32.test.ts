import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix hmc (src/engine/fix/hmc.ts): hybrid Monte Carlo. The velocity stream,
 * per-cycle NVE trajectory and the force kept across a rejected move are pinned
 * by the w32hmc_accept / w32hmc_reject oracle cases (tests/oracle/). This file
 * checks the deterministic accept/reject limit, the argument errors and the
 * linear-momentum handling of the mom keyword.
 */

const run = async (script: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const out = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => out.set(n, (ap ? out.get(n) ?? '' : '') + t),
  });
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
  return { error, rows, files: out };
};

const rowAt = (rows: ThermoRow[], step: number): ThermoRow => {
  const r = rows.find((x) => x.step === step);
  if (!r) throw new Error(`no thermo row at step ${step}`);
  return r;
};

/** Sum of mass * velocity from a write_dump of vx vy vz. */
const momentum = (dump: string): number[] => {
  const lines = dump.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  const p = [0, 0, 0];
  for (const l of lines.slice(k + 1)) {
    if (!l.trim()) continue;
    const w = l.trim().split(/\s+/);
    const m = Number(w[cols.indexOf('mass')] ?? 4);
    p[0] += m * Number(w[cols.indexOf('vx')]);
    p[1] += m * Number(w[cols.indexOf('vy')]);
    p[2] += m * Number(w[cols.indexOf('vz')]);
  }
  return p;
};

const LJ = `units lj
atom_style atomic
boundary p p p
region box block 0 8 0 8 0 8
create_box 1 box
create_atoms 1 single 2.0 2.0 2.0 units box
create_atoms 1 single 2.9 2.0 2.0 units box
create_atoms 1 single 2.0 5.0 2.0 units box
create_atoms 1 single 2.9 5.0 2.0 units box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
velocity all set 0.0 0.0 0.0
fix 1 all nve
`;

describe('fix hmc', () => {
  it('reproduces the deterministic T -> 0 accept/reject sequence (measured)', async () => {
    // Measured with native LAMMPS (black box): with T = 1e-9 the Metropolis test
    // is "accept iff dH < 0"; the move is rejected at steps 3 and 4 and accepted
    // at step 5, even though steps 4 and 5 start from the same restored state.
    const script = `${LJ}fix 2 all hmc 1 98765 1.0e-9 resample no
timestep 0.01
thermo_style custom step pe ke f_2[1] f_2[2]
thermo 1
run 8
`;
    const { error, rows } = await run(script);
    expect(error).toBeNull();
    expect(rowAt(rows, 3)['f_2[1]']).toBe(2);
    expect(rowAt(rows, 4)['f_2[1]']).toBe(2);
    expect(rowAt(rows, 5)['f_2[1]']).toBe(3);
    // the restored configuration's cached energy is reported on the reject rows
    expect(rowAt(rows, 3)['pe']).toBeCloseTo(rowAt(rows, 2)['pe'], 12);
    expect(rowAt(rows, 4)['pe']).toBeCloseTo(rowAt(rows, 2)['pe'], 12);
  });

  it('reproduces the always-accept limit with velocity resampling (measured)', async () => {
    // Measured with native LAMMPS (black box): T -> infinity, first move and all
    // later moves accepted; f_2[1] counts the accepted moves.
    const script = `units lj
atom_style atomic
boundary p p p
region box block 0 8 0 8 0 8
create_box 1 box
create_atoms 1 single 2.0 2.0 2.0 units box
create_atoms 1 single 3.5 2.0 2.0 units box
create_atoms 1 single 2.0 5.0 2.0 units box
create_atoms 1 single 3.5 5.0 2.0 units box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
velocity all set 0.0 0.0 0.0
fix 1 all nve
fix 2 all hmc 5 12345 1000000.0 resample yes
thermo_style custom step f_2[1] f_2[2]
thermo 5
run 20
`;
    const { error, rows } = await run(script);
    expect(error).toBeNull();
    for (const step of [5, 10, 15, 20]) {
      const r = rowAt(rows, step);
      expect(r['f_2[1]']).toBe(step / 5);
      expect(r['f_2[2]']).toBe(step / 5);
    }
  });

  it('zeroes the linear momentum with mom = yes but leaves it with mom = no', async () => {
    const base = `units lj
atom_style atomic
boundary p p p
region box block 0 8 0 8 0 8
create_box 1 box
create_atoms 1 single 2.0 2.0 2.0 units box
create_atoms 1 single 3.0 2.0 2.0 units box
mass 1 4.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
velocity all set 1.0 2.0 3.0
fix 1 all nve
`;
    const yes = await run(`${base}fix 2 all hmc 1 777 1000000.0 resample yes mom yes
thermo_style custom step
thermo 1
run 6
write_dump all custom mom_yes.dump id vx vy vz
`);
    expect(yes.error).toBeNull();
    const py = momentum(yes.files.get('mom_yes.dump')!);
    expect(Math.hypot(...py)).toBeLessThan(1e-6);

    const no = await run(`${base}fix 2 all hmc 1 777 1000000.0 resample yes mom no
thermo_style custom step
thermo 1
run 6
write_dump all custom mom_no.dump id vx vy vz
`);
    expect(no.error).toBeNull();
    const pn = momentum(no.files.get('mom_no.dump')!);
    expect(Math.hypot(...pn)).toBeGreaterThan(1);
  });

  it('rejects unsupported and inconsistent arguments with a StyleError', async () => {
    const cases: [string, RegExp][] = [
      [`fix h all hmc 1 12345 1.0 bogus\n`, /unknown keyword 'bogus'/],
      [`fix h all hmc 1 12345\n`, /usage: fix ID group-ID hmc N seed T/],
      [`fix h all hmc 0 12345 1.0\n`, /N must be a positive integer/],
      [`fix h all hmc 1 0 1.0\n`, /seed must be a positive integer/],
      [`fix h all hmc 1 12345 -1.0\n`, /T must be > 0/],
      [`fix h all hmc 1 12345 1.0 resample maybe\n`, /resample must be yes or no/],
      [`fix h all hmc 1 12345 1.0 mom maybe\n`, /mom must be yes or no/],
      [`fix h all hmc 1 12345 1.0 rigid 1\n`, /rigid 1: .*rigid HMC is not supported/],
    ];
    for (const [fix, re] of cases) {
      const script = `${LJ}${fix}thermo_style custom step
run 1
`;
      const { error } = await run(script);
      expect(error, fix).toMatch(re);
    }
  });

  it('accepts only the documented keywords', async () => {
    const { error } = await run(`${LJ}fix 2 all hmc 2 12345 1.5 resample yes mom no
thermo_style custom step
run 4
`);
    expect(error).toBeNull();
  });

  it('runs the flexible-melt example (shortened)', async () => {
    // A shortened copy of LAMMPS examples/mc/in.hmc.flexible.melt: the full run
    // is 1000 equilibrate + 50000 production steps; 100 + 300 keeps it quick.
    const script = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify every 2 delay 4 check yes
velocity all create 3.0 321654 loop geom
fix 1 all nve
thermo 50
run 100
fix 2 all hmc 100 6234 1.5 resample yes
thermo_style custom step temp pe ke etotal f_2 f_2[*]
timestep 0.005
run 300
`;
    const { error, rows } = await run(script);
    expect(error).toBeNull();
    const last = rows[rows.length - 1];
    expect(last.step).toBe(400);
    expect(last['f_2[2]']).toBeGreaterThan(0);
    expect(last['f_2[1]']).toBeLessThanOrEqual(last['f_2[2]']);
  });
});
