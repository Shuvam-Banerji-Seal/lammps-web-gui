import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * group dynamic / group static (docs.lammps.org/group.html): argument errors
 * with the native error texts (measured black box), logging, the fixes that
 * refuse a dynamic group, and region matching of atoms that crossed a
 * periodic face. Parity of whole runs with native LAMMPS is in
 * tests/oracle/w8group_dynamic_*.in.
 */

const BASE = `units lj
atom_modify map array
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.3 12345
mass * 1.0
velocity all create 1.5 87287 loop all
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
region left block INF 1.5 INF INF INF INF
`;

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(BASE + text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error');
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { error: err?.message ?? error, logs, thermo };
};

describe('group dynamic: argument errors (native texts)', () => {
  const cases: [string, string][] = [
    ['group all dynamic all region left', 'Group dynamic cannot reference itself'],
    ['group b type 1\ngroup all dynamic b region left', 'Group all cannot be made dynamic'],
    ['group a dynamic nope region left', 'Group dynamic parent group nope does not exist'],
    ['group a dynamic all', 'Illegal group command'],
    ['group a dynamic all region left every 0', 'Illegal every value 0 for dynamic group a'],
    ['group a dynamic all region nope', 'Region nope for dynamic group a does not exist'],
    ['group a dynamic all var nope', "Variable 'nope' for dynamic group a does not exist"],
    ['group a dynamic all property foo', 'Custom per-atom vector foo for dynamic group a does not exist'],
    ['group a dynamic all bogus 1', 'Unknown keyword bogus in dynamic group command'],
    ['group a dynamic all region left\ngroup b union a', 'Cannot union groups from a dynamic group'],
    ['group a dynamic all region left\ngroup b subtract all a', 'Cannot subtract dynamic groups'],
    ['group a dynamic all region left\ngroup b intersect all a', 'Cannot intersect groups using a dynamic group'],
    ['group b clear', 'Could not find group clear group ID b'],
    ['group a dynamic all region left\ngroup b dynamic a region left\nrun 0', 'Dynamic group parent group a cannot be dynamic'],
    ['variable v equal 1\ngroup a dynamic all var v\nrun 0', "Variable 'v' for dynamic group a is of incompatible style"],
    ['group a dynamic all region left\nfix 1 a spring/self 1', 'Fix spring/self does not allow use with a dynamic group'],
    ['group a dynamic all region left\nfix 1 a rigid single', 'Fix rigid does not allow use with a dynamic group'],
    ['group a dynamic all region left\ncompute 1 a msd', 'Compute msd is not compatible with dynamic groups'],
  ];
  for (const [text, msg] of cases) {
    it(text.split('\n').pop()!, async () => {
      const r = await run(text);
      expect(r.error).toContain(msg);
    });
  }
});

describe('group dynamic: assignment', () => {
  it('keeps the atoms until a run assigns them, logs "dynamic group ID defined"', async () => {
    const r = await run(`group a type 1
variable n equal count(a)
print "pre \${n}"
group a dynamic all region left
print "post \${n}"
run 0
print "run \${n}"
`);
    expect(r.error).toBeNull();
    const pre = r.logs.find((l) => l.startsWith('pre '))!;
    expect(r.logs.find((l) => l.startsWith('post '))).toBe(pre.replace('pre', 'post'));
    expect(r.logs).toContain('dynamic group a defined');
    const n = Number(r.logs.find((l) => l.startsWith('run '))!.slice(4));
    // atoms with x <= 1.5 lattice spacings: planes 0, 0.5, 1 and 1.5 of 6
    expect(n).toBe(4 * 18);
  });

  it('group static keeps the current atoms and reports the count', async () => {
    const r = await run(`group a dynamic all region left
group a static
run 0
variable n equal count(a)
print "n \${n}"
`);
    expect(r.error).toBeNull();
    expect(r.logs).toContain('0 atoms in group a');
    expect(r.logs).toContain('n 0');
  });

  it('every N reassigns only on multiples of N; the temperature dof follows the count', async () => {
    const r = await run(`group a dynamic all region left every 4
variable n equal count(a)
compute t a temp
compute k a ke
variable dof equal 2*c_k/c_t
fix 1 all nve
thermo_style custom step v_n v_dof
thermo 1
run 8
`);
    expect(r.error).toBeNull();
    for (const row of r.thermo) {
      expect(row.v_dof).toBeCloseTo(3 * Number(row.v_n) - 3, 6);
      if (Number(row.step) % 4 !== 0) {
        const prev = r.thermo.find((x) => Number(x.step) === Number(row.step) - 1)!;
        expect(row.v_n).toBe(prev.v_n);
      }
    }
  });

  it('a region matches an atom that crossed a periodic face at its image inside the box', async () => {
    const r = await run(`region edge block INF 0.1 INF INF INF INF
set atom 1 x -0.01
variable n equal count(all,edge)
print "n \${n}"
`);
    expect(r.error).toBeNull();
    // 18 atoms sit on the plane x = 0; atom 1 is moved to x = -0.01 (set does not remap), and its
    // image at L - 0.01 lies outside edge. Measured with native LAMMPS (black box): 17.
    const n = Number(r.logs.find((l) => l.startsWith('n '))!.slice(2));
    expect(n).toBe(17);
  });
});
