import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Direct checks for fix wall/table (docs.lammps.org/fix_wall.html). The tables
 * are chosen so the expected values are exact by hand:
 *  - HARMONIC nodes on a uniform r grid with Ntable = Nfile, so the internal
 *    table's grid coincides with the file nodes and linear interpolation of
 *    the file values is exact between them.
 *  - A quadratic E = r^2 with F = -2r on two nodes, which the two-node spline
 *    path (cubic Hermite) reproduces exactly, and whose internal spline also
 *    reproduces a quadratic exactly.
 * units lj and thermo_modify norm no, so thermo values are raw. f_1 is the
 * fix's global scalar (energy); f_1[1] is the normal force on the first wall.
 */

const runScript = async (text: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error: Error | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as Error;
  }
  const rows = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { session, rows, error };
};

/** E = 10 (r-2.5)^2, F = 20 (2.5-r) on the uniform nodes 1, 1.5, 2, 2.5, 3. */
const HARM = `HARM
N 5 FP -30 -10

1 1.0 22.5 30.0
2 1.5 10.0 20.0
3 2.0 2.5 10.0
4 2.5 0.0 0.0
5 3.0 2.5 -10.0
`;

const BASE = `
units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 1.25
mass            1 1.0
`;

describe('fix wall/table linear', () => {
  it('interpolates the table energy and force and shifts the energy to 0 at the cutoff', async () => {
    const { rows, error, session } = await runScript(`${BASE}
fix 1 all wall/table linear 5 zlo 0.0 harm.dat HARM 3.0 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`, { 'harm.dat': HARM });
    expect(error?.message ?? '').toBe('');
    // r = 1.25 is midway between the nodes 1.0 (E=22.5) and 1.5 (E=10.0)
    const r = rows[rows.length - 1];
    const eShift = 2.5; // E(3.0)
    expect(r.pe).toBeCloseTo((22.5 + 10.0) / 2 - eShift, 12);
    expect(r.f_1).toBeCloseTo((22.5 + 10.0) / 2 - eShift, 12);
    // F midway between 30 and 20; force on the lo wall is opposite the atom's
    expect(r['f_1[1]']).toBeCloseTo(-25, 12);
    expect([...session!.system!.f.slice(0, 3)]).toEqual([0, 0, 25]);
  });

  it('gives no interaction at or beyond the cutoff', async () => {
    const { rows, error } = await runScript(`units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 3.5
mass            1 1.0
fix 1 all wall/table linear 5 zlo 0.0 harm.dat HARM 3.0 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`, { 'harm.dat': HARM });
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    expect(r.pe).toBe(0);
    expect(r.f_1).toBe(0);
    expect(r['f_1[1]']).toBe(0);
  });
});

describe('fix wall/table spline', () => {
  it('reproduces a quadratic table exactly', async () => {
    // E = r^2, F = -2r between r=1 and r=3 (Nfile = 2, so the preliminary
    // splines are cubic Hermite interpolants, exact for these polynomials).
    const quad = `QUAD
N 2

1 1.0 1.0 -2.0
2 3.0 9.0 -6.0
`;
    const { rows, error } = await runScript(`units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 1.75
mass            1 1.0
fix 1 all wall/table spline 5 zlo 0.0 quad.dat QUAD 3.0 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`, { 'quad.dat': quad });
    expect(error?.message ?? '').toBe('');
    const r = rows[rows.length - 1];
    const pe = 1.75 ** 2 - 9;
    expect(r.pe).toBeCloseTo(pe, 12);
    expect(r.f_1).toBeCloseTo(pe, 12);
    expect(r['f_1[1]']).toBeCloseTo(-(-2 * 1.75), 12);
  });

  it('uses the FP end derivatives of the force spline', async () => {
    // F = 7 - (r-1.5)^2 (quadratic), E = (r-1.5)^3/3 - 7r so F = -dE/dr; FP is
    // dF/dr = -2(r-1.5) at the ends, 1 and -3, which differs from the
    // first/last-two estimate 0.5 and -5.
    const nodes = [1.0, 1.5, 2.0, 2.5, 3.0];
    const f = (r: number) => 7 - (r - 1.5) ** 2;
    const e = (r: number) => (r - 1.5) ** 3 / 3 - 7 * r;
    const table = (fp: string) => `CUB\nN 5${fp}\n\n${nodes.map((r, i) => `${i + 1} ${r} ${e(r)} ${f(r)}`).join('\n')}\n`;
    const run = (file: string) => runScript(`units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 1.1
mass            1 1.0
fix 1 all wall/table spline 101 zlo 0.0 cub.dat CUB 3.0 units box
fix_modify 1 energy yes
thermo_style custom step pe f_1 f_1[1]
thermo_modify norm no
run 0`, { 'cub.dat': file });
    const a = await run(table(' FP 1 -3'));
    const b = await run(table(''));
    expect(a.error?.message ?? '').toBe('');
    expect(b.error?.message ?? '').toBe('');
    const ra = a.rows[a.rows.length - 1], rb = b.rows[b.rows.length - 1];
    // FP makes the force spline exact (quadratic end derivatives)
    expect(ra['f_1[1]']).toBeCloseTo(-f(1.1), 10);
    // without FP the end derivatives are estimated and the force differs
    expect(rb['f_1[1]']).not.toBeCloseTo(-f(1.1), 6);
    // FP never changes the energy column
    expect(ra.pe).toBeCloseTo(e(1.1) - e(3.0), 10);
    expect(rb.pe).toBeCloseTo(ra.pe, 12);
  });
});

describe('fix wall/table errors', () => {
  const base = (extra: string, pos = 1.25) => `units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5 5 ${pos}
mass            1 1.0
${extra}
run 0`;

  it('rejects an unknown tabstyle', async () => {
    const { error } = await runScript(base('fix 1 all wall/table cubic 5 zlo 0.0 harm.dat HARM 3.0 units box'), { 'harm.dat': HARM });
    expect(error?.message ?? '').toMatch(/unknown table style 'cubic'/);
  });

  it('rejects N < 2', async () => {
    const { error } = await runScript(base('fix 1 all wall/table linear 1 zlo 0.0 harm.dat HARM 3.0 units box'), { 'harm.dat': HARM });
    expect(error?.message ?? '').toMatch(/N must be an integer >= 2/);
  });

  it('rejects an unknown table parameter by name', async () => {
    const bad = `HARM\nN 5 RSQ 1.0 3.0\n\n1 1.0 22.5 30.0\n2 1.5 10.0 20.0\n3 2.0 2.5 10.0\n4 2.5 0.0 0.0\n5 3.0 2.5 -10.0\n`;
    const { error } = await runScript(base('fix 1 all wall/table linear 5 zlo 0.0 bad.dat HARM 3.0 units box'), { 'bad.dat': bad });
    expect(error?.message ?? '').toMatch(/unknown table parameter 'RSQ'/);
  });

  it('rejects a cutoff beyond the outer tabulated distance', async () => {
    const { error } = await runScript(base('fix 1 all wall/table linear 5 zlo 0.0 harm.dat HARM 3.5 units box'), { 'harm.dat': HARM });
    expect(error?.message ?? '').toMatch(/exceeds the table outer distance/);
  });

  it('rejects a particle below the inner tabulated distance', async () => {
    const { error } = await runScript(base('fix 1 all wall/table linear 5 zlo 0.0 harm.dat HARM 3.0 units box', 0.5), { 'harm.dat': HARM });
    expect(error?.message ?? '').toMatch(/below the table inner cutoff/);
  });
});
