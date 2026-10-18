import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * compute heat/flux/tally and heat/flux/virial/tally (src/engine-gpl/tally2): the argument and group refusals with
 * native's text, and an analytic two-atom check. Native parity is in tests/oracle/w239ht_*.
 */

/** Runs an input and returns the first error message and the thermo rows. */
async function run(script: string) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  try {
    await session.execute(script);
  } catch {
    // the failure is reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { error: err && 'message' in err ? String(err.message) : undefined, rows };
}

/** Two atoms in a free (non-periodic) box, atom 1 in group g1 with velocity (1,0,0), atom 2 in group g2 at rest. */
const PAIR = (r: number, pair: string) => `units lj
atom_style atomic
boundary f f f
region box block -10 10 -10 10 -10 10
create_box 2 box
create_atoms 1 single 0 0 0
create_atoms 2 single ${r} 0 0
mass * 1.0
group g1 id 1
group g2 id 2
velocity g1 set 1 0 0 units box
velocity g2 set 0 0 0 units box
pair_style ${pair}
pair_coeff * *${pair.startsWith('zero') ? '' : ' 1.0 1.0'}
`;

describe('compute heat/flux/tally and heat/flux/virial/tally: refusals and group checks', () => {
  it('needs the group2-ID, with the native missing-argument text', async () => {
    expect((await run(`${PAIR(1.5, 'lj/cut 2.5')}compute h g1 heat/flux/tally\nrun 0\n`)).error).toMatch(/Illegal compute heat\/flux\/tally command: missing argument\(s\)/);
    expect((await run(`${PAIR(1.5, 'lj/cut 2.5')}compute v g1 heat/flux/virial/tally\nrun 0\n`)).error).toMatch(/Illegal compute heat\/flux\/virial\/tally command: missing argument\(s\)/);
  });

  it('refuses an unknown second group with the native text', async () => {
    const r = await run(`${PAIR(1.5, 'lj/cut 2.5')}compute h g1 heat/flux/tally nosuch\nrun 0\n`);
    expect(r.error).toMatch(/Could not find compute heat\/flux\/tally second group ID nosuch/);
    const v = await run(`${PAIR(1.5, 'lj/cut 2.5')}compute v g1 heat/flux/virial/tally nosuch\nrun 0\n`);
    expect(v.error).toMatch(/Could not find compute heat\/flux\/virial\/tally second group ID nosuch/);
  });

  it('refuses extra arguments by name (native ignores them)', async () => {
    const r = await run(`${PAIR(1.5, 'lj/cut 2.5')}compute h g1 heat/flux/tally g2 pair yes\nrun 0\n`);
    expect(r.error).toMatch(/heat\/flux\/tally takes one group2-ID \(no keywords\)/);
  });

  it('the virial version errors when an atom is in both groups (native text, at run start)', async () => {
    const r = await run(`${PAIR(1.5, 'lj/cut 2.5')}compute v g1 heat/flux/virial/tally g1\nthermo_style custom step c_v\nrun 0\n`);
    expect(r.error).toMatch(/Atom 1 belonging to both groups is not allowed with compute heat\/flux\/virial\/tally/);
  });

  it('refuses a pair style without single(), naming it', async () => {
    const r = await run(`${PAIR(1.5, 'zero 2.5')}compute h g1 heat/flux/tally g2\nthermo_style custom step c_h[1]\nrun 0\n`);
    expect(r.error).toMatch(/needs a pair_style that supports single\(\).*pair_style zero does not/);
  });
});

describe('compute heat/flux/tally and heat/flux/virial/tally: analytic two-atom values', () => {
  // Lennard-Jones with epsilon = sigma = 1: U(r) = 4 (r^-12 - r^-6), fpair = F(r)/r = 24 (2 r^-14 - r^-8).
  const r = 1.5;
  const U = 4 * (Math.pow(r, -12) - Math.pow(r, -6));
  const fpair = 24 * (2 * Math.pow(r, -14) - Math.pow(r, -8));
  const dx = -r; // x1 - x2 with atom 1 at the origin and atom 2 at +r
  const F1x = dx * fpair; // force on atom 1 along x (repulsive is negative here: atom 1 is pushed away)

  it('heat/flux/virial/tally scalar is 0.5 F1x (v1x + v2x) = 0.5 F1x for the pair', async () => {
    const res = await run(`${PAIR(r, 'lj/cut 2.5')}compute v g1 heat/flux/virial/tally g2\nthermo_style custom step c_v\nthermo_modify norm no format float %.15g\nrun 0\n`);
    expect(res.error).toBeUndefined();
    expect(res.rows[0]!['c_v']).toBeCloseTo(0.5 * F1x * 1, 10);
  });

  it('heat/flux/tally vector: x flux of group1 = (0.5 m v^2 + 0.5 U) v1x + stress_xx v1x with stress_xx = 0.5 fpair dx^2', async () => {
    const res = await run(`${PAIR(r, 'lj/cut 2.5')}compute h g1 heat/flux/tally g2\nthermo_style custom step c_h[1] c_h[2] c_h[4]\nthermo_modify norm no format float %.15g\nrun 0\n`);
    expect(res.error).toBeUndefined();
    const row = res.rows[0]!;
    const convective = 0.5 * 1 * 1 + 0.5 * U; // 0.5 m v1x^2 + epairhalf, times v1x
    const virial = 0.5 * fpair * dx * dx * 1; // stress_xx v1x (stress_xx = 0.5 fpair dx^2)
    expect(row['c_h[4]']).toBeCloseTo(convective, 10);
    expect(row['c_h[1]']).toBeCloseTo(convective + virial, 10);
    expect(row['c_h[2]']).toBeCloseTo(0, 10);
  });
});
