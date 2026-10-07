import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Direct checks of fix spring (docs.lammps.org/fix_spring.html) and
 * fix spring/self (docs.lammps.org/fix_spring_self.html) on tiny systems:
 * the documented formulas, the NULL/dim keywords and the argument errors.
 * thermo_modify norm no keeps the raw (unnormalized) fix values.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { session, thermo, error, files };
};

/** Per-atom columns of a written custom dump, keyed by atom id. */
const dumpAtoms = (text: string): Map<number, Record<string, number>> => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  const out = new Map<number, Record<string, number>>();
  for (const line of lines.slice(k + 1)) {
    const w = line.trim().split(/\s+/).map(Number);
    const a: Record<string, number> = {};
    cols.forEach((c, i) => { a[c] = w[i]; });
    out.set(a.id, a);
  }
  return out;
};

const box = (ntypes: number): string => `
units           lj
atom_style      atomic
region          box block 0 10 0 10 0 10
create_box      ${ntypes} box
`;

const THERMO = 'thermo_style custom step f_1 f_1[1] f_1[2] f_1[3] f_1[4]\nthermo_modify norm no\n';
// spring/self has only a global scalar
const THERMO_SELF = 'thermo_style custom step f_1\nthermo_modify norm no\n';

describe('fix spring tether (fix_spring.html formula)', () => {
  // two atoms, masses 1 and 3 at x = 2 and 4: COM = (2*1 + 4*3)/4 = 3.5
  const TWO = `${box(2)}
create_atoms    1 single 2 0 0
create_atoms    2 single 4 0 0
mass            1 1.0
mass            2 3.0
`;

  it('force -K(R-R0)Mi/M to the tether point, energy 0.5 K (R-R0)^2', async () => {
    const { thermo, files } = await runScript(`
${TWO}
fix             1 all spring tether 10.0 1.0 0.0 0.0 0.0
${THERMO}
run             0
write_dump      all custom spring.dump id fx fy fz
`);
    // delta = (2.5,0,0), r = 2.5, K(R-R0) = 25
    const r = thermo[thermo.length - 1];
    expect(r['f_1']).toBeCloseTo(0.5 * 10 * 2.5 * 2.5, 12);   // energy
    expect(r['f_1[1]']).toBeCloseTo(-25, 12);                 // force on group, x
    expect(r['f_1[2]']).toBeCloseTo(0, 12);
    expect(r['f_1[3]']).toBeCloseTo(0, 12);
    expect(r['f_1[4]']).toBeCloseTo(25, 12);                  // signed magnitude
    // per atom: -25 * Mi / M with M = 4
    const atoms = dumpAtoms(files.get('spring.dump') ?? '');
    expect(atoms.get(1)!.fx).toBeCloseTo(-6.25, 12);
    expect(atoms.get(2)!.fx).toBeCloseTo(-18.75, 12);
  });

  it('R0 holds the group on a sphere: r < R0 pushes away', async () => {
    const { thermo } = await runScript(`
${TWO}
fix             1 all spring tether 10.0 1.0 0.0 0.0 5.0
${THERMO}
run             0
`);
    // r = 2.5, R0 = 5: K(R-R0) = -25, energy = 0.5*10*6.25
    const r = thermo[thermo.length - 1];
    expect(r['f_1']).toBeCloseTo(31.25, 12);
    expect(r['f_1[1]']).toBeCloseTo(25, 12);
    expect(r['f_1[4]']).toBeCloseTo(-25, 12);
  });

  it('NULL drops a dimension from the distance and the force', async () => {
    const { thermo } = await runScript(`
${TWO}
fix             1 all spring tether 10.0 NULL NULL 5.0 0.2
${THERMO}
run             0
`);
    // only z: COM_z = 0, delta = -5, r = 5, K(r-R0) = 48
    const r = thermo[thermo.length - 1];
    expect(r['f_1']).toBeCloseTo(0.5 * 10 * 4.8 * 4.8, 12);
    expect(r['f_1[1]']).toBeCloseTo(0, 12);
    expect(r['f_1[2]']).toBeCloseTo(0, 12);
    expect(r['f_1[3]']).toBeCloseTo(48, 12);   // pulls +z toward z = 5
    expect(r['f_1[4]']).toBeCloseTo(48, 12);
  });

  it('wrap-around: the spring crosses a periodic boundary', async () => {
    const { thermo } = await runScript(`
${box(1)}
create_atoms    1 single 1 5 5
create_atoms    1 single 9.5 5 5
mass            1 1.0
fix             1 all spring tether 10.0 0.0 NULL NULL 0.0
${THERMO}
run             0
`);
    // COM_x = 5.25, delta = 5.25 -> minimum image -4.75: pull is +x through the boundary
    const r = thermo[thermo.length - 1];
    expect(r['f_1[1]']).toBeCloseTo(47.5, 12);
    expect(r['f_1[4]']).toBeCloseTo(47.5, 12);
  });
});

describe('fix spring couple (fix_spring.html formula)', () => {
  it('equal and opposite forces at the equilibrium displacement x,y,z', async () => {
    const { thermo, files } = await runScript(`
${box(2)}
create_atoms    1 single 2 5 5
create_atoms    2 single 4 5 5
group           ga type 1
group           gb type 2
mass            1 1.0
mass            2 3.0
fix             1 ga spring couple gb 10.0 1.0 NULL NULL 0.5
${THERMO}
run             0
write_dump      all custom couple.dump id fx fy fz
`);
    // COM2 - COM1 = (2,0,0); e = (1,0,0) (y,z NULL), r = 1, K(r-R0) = 5
    const r = thermo[thermo.length - 1];
    expect(r['f_1']).toBeCloseTo(0.5 * 10 * 0.25, 12);
    expect(r['f_1[1]']).toBeCloseTo(5, 12);    // force on the fix group (ga)
    expect(r['f_1[4]']).toBeCloseTo(5, 12);
    const atoms = dumpAtoms(files.get('couple.dump') ?? '');
    expect(atoms.get(1)!.fx).toBeCloseTo(5, 12);   // single atom: full force
    expect(atoms.get(2)!.fx).toBeCloseTo(-5, 12);
  });
});

describe('fix spring/self (fix_spring_self.html formula)', () => {
  it('force -K r to the initial position, energy sum of 0.5 K r^2', async () => {
    const { thermo, files } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5
mass            1 1.0
fix             1 all spring/self 10.0
displace_atoms  all move 0.2 -0.1 0.0 units box
${THERMO_SELF}
run             0
write_dump      all custom self.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r['f_1']).toBeCloseTo(0.5 * 10 * (0.04 + 0.01), 12);
    const atoms = dumpAtoms(files.get('self.dump') ?? '');
    expect(atoms.get(1)!.fx).toBeCloseTo(-2, 12);
    expect(atoms.get(1)!.fy).toBeCloseTo(1, 12);
    expect(atoms.get(1)!.fz).toBeCloseTo(0, 12);
  });

  it('dir z restrains the force and the energy to z', async () => {
    const { thermo, files } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5
mass            1 1.0
fix             1 all spring/self 10.0 z
displace_atoms  all move 0.2 -0.1 0.3 units box
${THERMO_SELF}
run             0
write_dump      all custom selfz.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r['f_1']).toBeCloseTo(0.5 * 10 * 0.09, 12);
    const atoms = dumpAtoms(files.get('selfz.dump') ?? '');
    expect(atoms.get(1)!.fx).toBeCloseTo(0, 12);
    expect(atoms.get(1)!.fy).toBeCloseTo(0, 12);
    expect(atoms.get(1)!.fz).toBeCloseTo(-3, 12);
  });

  it('K from an equal-style variable', async () => {
    const { thermo } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5
mass            1 1.0
variable        kk equal 10.0
fix             1 all spring/self v_kk
displace_atoms  all move 0.2 -0.1 0.0 units box
${THERMO_SELF}
run             0
`);
    expect(thermo[thermo.length - 1]['f_1']).toBeCloseTo(0.25, 12);
  });

  it('fix_modify energy yes adds the spring energy to pe', async () => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e) });
    await session.execute(`
${box(1)}
create_atoms    1 single 5 5 5
mass            1 1.0
fix             1 all spring/self 10.0
displace_atoms  all move 0.2 -0.1 0.0 units box
thermo_style    custom step pe f_1
thermo_modify   norm no
run             0
`);
    const rows = (): ThermoRow[] => events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    expect(rows().at(-1)!.pe).toBe(0);                      // energy no (default)
    await session.execute('fix_modify 1 energy yes\nrun 0');
    expect(rows().at(-1)!.pe).toBeCloseTo(0.25, 12);
  });
});

describe('fix spring / spring/self argument errors', () => {
  const bad = async (line: string): Promise<string> => {
    const { error } = await runScript(`
${box(1)}
create_atoms    1 single 5 5 5
mass            1 1.0
${line}
`);
    expect(error).toBeTruthy();
    return String((error as Error).message);
  };

  it('rejects an unknown keyword, missing/extra values and bad numbers', async () => {
    expect(await bad('fix 1 all spring bogus 1 2 3 4 5')).toMatch(/tether or couple/);
    expect(await bad('fix 1 all spring tether 10.0 1 0 0')).toMatch(/usage/);
    expect(await bad('fix 1 all spring couple 10.0 1 0 0 0')).toMatch(/unknown group|usage/);
    expect(await bad('fix 1 all spring tether 10.0 1 0 abc 0')).toMatch(/number or NULL/);
    expect(await bad('fix 1 all spring tether 10.0 1 0 0 0 9')).toMatch(/usage/);
  });

  it('rejects a bad dir and an undefined or unsupported K variable', async () => {
    expect(await bad('fix 1 all spring/self 10.0 q')).toMatch(/dir must be/);
    expect(await bad('fix 1 all spring/self v_novar')).toMatch(/not defined/);
    expect(await bad('variable s string 5\nfix 1 all spring/self v_s')).toMatch(/equal-style or atom-style/);
  });
});
