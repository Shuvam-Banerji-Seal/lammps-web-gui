/**
 * The MANIFOLD fixes (src/engine-gpl/manifold: fix nve/manifold/rattle, fix nvt/manifold/rattle, fix manifoldforce):
 * the messages and the behaviour checked against native LAMMPS b332f28 built with PKG_MANIFOLD
 * (/store/shuvam/software/lammps-build-manifold/lmp). Native runs of these inputs gave the messages below, the
 * "Outputting every 2 steps, next is 2" line and the per-interval statistics of the every keyword
 * ("nve/manifold/rattle stats for time step 2 on 3 atoms:", "  iters/atom: x = 3, v = 2, dofs removed = 3"), and a
 * temperature of 0.14 for three atoms at velocity (0.3, 0.1, -0.2) on a sphere (dof 9 - 3 - 3: one removed per
 * constrained atom). The native parity of the dynamics is the oracle cases w282mf_* to w289mf_*.
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

const base = [
  'units lj', 'boundary f f f', 'region b block -12 12 -12 12 -12 12', 'create_box 1 b', 'mass 1 1',
  'create_atoms 1 single 10 0 0', 'create_atoms 1 single 0 10 0', 'variable va atom x', 'variable ve equal 10',
];

const run = async (lines: string[]): Promise<{ error: string; rows: Record<string, number>[]; log: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  try {
    await session.execute(lines.join('\n'));
  } catch {
    // the error event carries the message
  }
  const err = events.find((e) => e.kind === 'error');
  const rows = events.filter((e) => e.kind === 'thermo').map((e) => (e as unknown as { row: Record<string, number> }).row);
  const log = events.filter((e) => e.kind === 'log').map((e) => (e as { text: string }).text);
  return { error: err && 'message' in err ? String(err.message) : '', rows, log };
};

describe('fix nve/manifold/rattle and nvt/manifold/rattle: native messages', () => {
  const cases: [string[], RegExp][] = [
    [['fix f all nve/manifold/rattle 1e-4 10'], /Illegal fix nve\/manifold\/rattle command/],
    [['fix f all nve/manifold/rattle 1e-4 10 blob 1'], /Error creating manifold pointer/],
    [['fix f all nve/manifold/rattle 1e-4 10 sphere'], /Not enough args for manifold sphere, 1 expected but got 0/],
    [['fix f all nve/manifold/rattle 1e-4 10 sphere abc'], /Expected floating point parameter instead of 'abc' in input script or data file/],
    [['fix f all nve/manifold/rattle 1e-4 x sphere 10'], /Expected integer parameter instead of 'x' in input script or data file/],
    [['fix f all nve/manifold/rattle 1e-4 10 sphere v_va', 'run 1'], /Variable for fix nve\/manifold\/rattle is invalid style/],
    [['fix f all nve/manifold/rattle 1e-4 10 sphere v_nope', 'run 1'], /Variable name for fix nve\/manifold\/rattle does not exist/],
    [['fix f all nve/manifold/rattle 1e-4 10 sphere 10 foo'], /Error parsing arg "foo"\./],
    [['fix f all nvt/manifold/rattle 1e-4 10 sphere 10'], /Fix nvt\/manifold\/rattle needs 'temp'!/],
    [['fix f all nvt/manifold/rattle 1e-4 10 sphere 10 temp 1 1 1 every 5'], /Error parsing arg "every"\./],
    [['fix f all nvt/manifold/rattle 1e-4 10 sphere 10 temp 1 1 -1'], /Fix nvt\/manifold\/rattle damping parameter must be > 0\.0/],
    [['fix f all nve/manifold/rattle 1e-4 10 thylakoid 5 10 3'], /Thylakoid expects xlo of at most -32\.00000000, but found -12\.00000000/],
    [['fix f all nve/manifold/rattle 1e-4 10 torus 2 4', 'run 1'], /Large radius < small radius!/],
  ];
  for (const [lines, msg] of cases) {
    it(lines.join(' / '), async () => {
      expect((await run([...base, ...lines])).error).toMatch(msg);
    });
  }
});

describe('fix manifoldforce: native messages', () => {
  it('needs a manifold, constant parameters, and min_style hftn or quickmin even for a run', async () => {
    expect((await run([...base, 'fix f all manifoldforce'])).error).toMatch(/Illegal fix manifoldforce! No manifold given/);
    expect((await run([...base, 'fix f all manifoldforce sphere v_ve'])).error).toMatch(/Equal-style variables not allowed with fix manifoldforce/);
    expect((await run([...base, 'fix f all manifoldforce sphere 10', 'run 1'])).error).toMatch(/Fix manifoldforce requires min_style hftn or quickmin/);
    expect((await run([...base, 'fix f all manifoldforce sphere 10', 'min_style quickmin', 'run 1'])).error).toBe('');
  });
});

describe('fix nve/manifold/rattle: dof and the every keyword', () => {
  it('removes one dof per constrained atom from every temperature compute, and prints the iteration statistics', async () => {
    const r = await run([
      ...base, 'create_atoms 1 single 0 0 10', 'velocity all set 0.3 0.1 -0.2',
      'fix f all nve/manifold/rattle 1e-8 10 sphere 10 every 2', 'compute t2 all temp',
      'thermo_style custom step temp c_t2', 'thermo 4', 'run 4',
    ]);
    expect(r.error).toBe('');
    expect(r.rows[0].temp).toBeCloseTo(0.14, 12);
    expect(r.rows[0].c_t2).toBeCloseTo(0.14, 12);
    const text = r.log.join('\n');
    expect(text).toContain('Outputting every 2 steps, next is 2');
    expect(text).toContain('nve/manifold/rattle stats for time step 2 on 3 atoms:');
    expect(text).toContain('nve/manifold/rattle stats for time step 4 on 3 atoms:');
    expect(text).toContain('  iters/atom: x = 3, v = 2, dofs removed = 3');
  });
});
