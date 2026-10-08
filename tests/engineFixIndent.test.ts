import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Direct checks of fix wall/reflect (docs.lammps.org/fix_wall_reflect.html)
 * and fix indent (docs.lammps.org/fix_indent.html) on tiny systems: the
 * documented reflection rule, the F(r) = -K (r - R)^2 repulsion and the
 * K/3 (r - R)^3 energy, the global scalar/3-vector output and the argument
 * errors. thermo_modify norm no keeps the raw (unnormalized) fix values.
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

const THERMO_F = 'thermo_style custom step f_1 f_1[1] f_1[2] f_1[3]\nthermo_modify norm no\n';

const WALL_SYS = `
units           lj
atom_style      atomic
boundary        p p f
region          box block 0 10 0 10 0 10 units box
create_box      1 box
mass            1 1.0
timestep        0.005
`;

describe('fix wall/reflect (fix_wall_reflect.html)', () => {
  it('reflects an atom that crosses the zhi EDGE: same delta back, velocity flipped', async () => {
    const { files } = await runScript(`
${WALL_SYS}
create_atoms    1 single 5 5 9.99
velocity        all set 0.0 0.0 4.0 units box
fix             1 all nve
fix             2 all wall/reflect zhi EDGE
run             1
write_dump      all custom refl_hi.dump id z vz
`);
    const a = dumpAtoms(files.get('refl_hi.dump') ?? '').get(1)!;
    // z = 9.99 + 0.005*4 = 10.01, delta = 0.01 outside: z = 2*10 - 10.01, vz = -4
    expect(a.z).toBeCloseTo(9.99, 12);
    expect(a.vz).toBeCloseTo(-4, 12);
  });

  it('reflects at the zlo EDGE back in the hi direction', async () => {
    const { files } = await runScript(`
${WALL_SYS}
create_atoms    1 single 5 5 0.01
velocity        all set 0.0 0.0 -4.0 units box
fix             1 all nve
fix             2 all wall/reflect zlo EDGE
run             1
write_dump      all custom refl_lo.dump id z vz
`);
    const a = dumpAtoms(files.get('refl_lo.dump') ?? '').get(1)!;
    // z = 0.01 - 0.02 = -0.01: z = 2*0 - (-0.01), vz = +4
    expect(a.z).toBeCloseTo(0.01, 12);
    expect(a.vz).toBeCloseTo(4, 12);
  });

  it('numeric wall with units box reflects at the given position', async () => {
    const { files } = await runScript(`
${WALL_SYS}
create_atoms    1 single 5 5 0.51
velocity        all set 0.0 0.0 -4.0 units box
fix             1 all nve
fix             2 all wall/reflect zlo 0.5 units box
run             1
write_dump      all custom refl_num.dump id z vz
`);
    const a = dumpAtoms(files.get('refl_num.dump') ?? '').get(1)!;
    expect(a.z).toBeCloseTo(0.51, 12);
    expect(a.vz).toBeCloseTo(4, 12);
  });

  it('units lattice scales a constant wall by the lattice spacing', async () => {
    const { files } = await runScript(`
units           lj
atom_style      atomic
boundary        p p f
lattice         sc 0.125
region          box block 0 10 0 10 0 10 units box
create_box      1 box
mass            1 1.0
create_atoms    1 single 5 5 1.01 units box
velocity        all set 0.0 0.0 -4.0 units box
timestep        0.005
fix             1 all nve
fix             2 all wall/reflect zlo 0.5 units lattice
run             1
write_dump      all custom refl_lat.dump id z vz
`);
    const a = dumpAtoms(files.get('refl_lat.dump') ?? '').get(1)!;
    // lattice sc 0.125 in lj units: spacing = (1/0.125)^(1/3) = 2.0, wall at 0.5*2 = 1.0
    // z = 1.01 - 0.02 = 0.99 < 1.0: reflected to 1.01 with vz = +4
    expect(a.z).toBeCloseTo(1.01, 12);
    expect(a.vz).toBeCloseTo(4, 12);
  });

  it('a variable wall position (v_name) is used as the current wall', async () => {
    const { files } = await runScript(`
${WALL_SYS}
create_atoms    1 single 5 5 9.94
velocity        all set 0.0 0.0 4.0 units box
variable        zw equal 9.95
fix             1 all nve
fix             2 all wall/reflect zhi v_zw
run             1
write_dump      all custom refl_var.dump id z vz
`);
    const a = dumpAtoms(files.get('refl_var.dump') ?? '').get(1)!;
    expect(a.z).toBeCloseTo(9.94, 12);
    expect(a.vz).toBeCloseTo(-4, 12);
  });
});

const INDENT_SYS = `
units           lj
atom_style      atomic
region          box block 0 10 0 10 0 10
create_box      1 box
mass            1 1.0
`;

describe('fix indent sphere (fix_indent.html)', () => {
  it('pushes an atom inside the sphere radially outward with K (R - r)^2, energy K/3 (R - r)^3', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 6 5 5
fix             1 all indent 10.0 sphere 5 5 5 2.0 units box
${THERMO_F}
run             0
write_dump      all custom sph.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r.f_1).toBeCloseTo(10 / 3, 12);      // energy = K/3 (R - r)^3, R - r = 1
    expect(r['f_1[1]']).toBeCloseTo(-10, 12);      // force on the indenter (reaction), -x
    expect(r['f_1[2]']).toBeCloseTo(0, 12);
    expect(r['f_1[3]']).toBeCloseTo(0, 12);
    const a = dumpAtoms(files.get('sph.dump') ?? '').get(1)!;
    expect(a.fx).toBeCloseTo(10, 12);           // repelled away from the center
    expect(a.fy).toBeCloseTo(0, 12);
    expect(a.fz).toBeCloseTo(0, 12);
  });

  it('exerts no force on atoms with r > R', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 8 5 5
fix             1 all indent 10.0 sphere 5 5 5 2.0 units box
${THERMO_F}
run             0
write_dump      all custom sph_out.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r.f_1).toBeCloseTo(0, 12);
    expect(r['f_1[1]']).toBeCloseTo(0, 12);
    const a = dumpAtoms(files.get('sph_out.dump') ?? '').get(1)!;
    expect(a.fx).toBeCloseTo(0, 12);
  });

  it('side in reverses the action: a containing wall, energy -K/3 (R - r)^3', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 6 5 5
fix             1 all indent 10.0 sphere 5 5 5 2.0 units box side in
${THERMO_F}
run             0
write_dump      all custom sph_in.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r.f_1).toBeCloseTo(-10 / 3, 12);
    expect(r['f_1[1]']).toBeCloseTo(10, 12);
    const a = dumpAtoms(files.get('sph_in.dump') ?? '').get(1)!;
    expect(a.fx).toBeCloseTo(-10, 12);          // pushed back toward the center
  });

  it('accounts for periodic boundaries when measuring the distance to the center', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 0.5 5 5
fix             1 all indent 10.0 sphere 9.5 5 5 2.0 units box
${THERMO_F}
run             0
write_dump      all custom sph_pbc.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r['f_1[1]']).toBeCloseTo(-10, 12);
    const a = dumpAtoms(files.get('sph_pbc.dump') ?? '').get(1)!;
    expect(a.fx).toBeCloseTo(10, 12);
  });

  it('geometry values can be equal-style variables', async () => {
    const { thermo } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 5 5 5
variable        zc equal 4.0
fix             1 all indent 10.0 sphere 5 5 v_zc 2.0 units box
${THERMO_F}
run             0
`);
    const r = thermo[thermo.length - 1];
    expect(r.f_1).toBeCloseTo(10 / 3, 12);
    expect(r['f_1[3]']).toBeCloseTo(-10, 12);
  });

  it('fix_modify energy yes adds the indenter energy to pe', async () => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e) });
    await session.execute(`
${INDENT_SYS}
create_atoms    1 single 6 5 5
fix             1 all indent 10.0 sphere 5 5 5 2.0 units box
thermo_style    custom step pe f_1
thermo_modify   norm no
run             0
`);
    const rows = (): ThermoRow[] => events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    expect(rows().at(-1)!.pe).toBe(0);                      // energy no (default)
    await session.execute('fix_modify 1 energy yes\nrun 0');
    expect(rows().at(-1)!.pe).toBeCloseTo(10 / 3, 12);
  });
});

describe('fix indent cylinder (fix_indent.html)', () => {
  it('force from the distance to the axis, no component along the axis', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 6 5 9
fix             1 all indent 8.0 cylinder z 5 5 2.0 units box
${THERMO_F}
run             0
write_dump      all custom cyl.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r.f_1).toBeCloseTo(8 / 3, 12);
    expect(r['f_1[1]']).toBeCloseTo(-8, 12);
    expect(r['f_1[3]']).toBeCloseTo(0, 12);
    const a = dumpAtoms(files.get('cyl.dump') ?? '').get(1)!;
    expect(a.fx).toBeCloseTo(8, 12);
    expect(a.fz).toBeCloseTo(0, 12);
  });

  it('side in pushes atoms inside the cylinder toward the axis', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 6 5 9
fix             1 all indent 8.0 cylinder z 5 5 2.0 units box side in
${THERMO_F}
run             0
write_dump      all custom cylin.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r.f_1).toBeCloseTo(-8 / 3, 12);
    expect(r['f_1[1]']).toBeCloseTo(8, 12);
    const a = dumpAtoms(files.get('cylin.dump') ?? '').get(1)!;
    expect(a.fx).toBeCloseTo(-8, 12);
  });
});

describe('fix indent plane (fix_indent.html)', () => {
  it('side lo: atoms below the plane are pushed towards the hi end', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 5 5 0.2
fix             1 all indent 8.0 plane z 0.4 lo units box
${THERMO_F}
run             0
write_dump      all custom pllo.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    // depth = 0.4 - 0.2 = 0.2: F = K*0.04 = 0.32 in +z, energy = K/3*0.008
    expect(r.f_1).toBeCloseTo((8 / 3) * 0.008, 12);
    expect(r['f_1[3]']).toBeCloseTo(-0.32, 12);
    expect(r['f_1[1]']).toBeCloseTo(0, 12);
    const a = dumpAtoms(files.get('pllo.dump') ?? '').get(1)!;
    expect(a.fz).toBeCloseTo(0.32, 12);
  });

  it('side hi: atoms above the plane are pushed towards the lo end', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 5 5 0.6
fix             1 all indent 8.0 plane z 0.4 hi units box
${THERMO_F}
run             0
write_dump      all custom plhi.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r.f_1).toBeCloseTo((8 / 3) * 0.008, 12);
    expect(r['f_1[3]']).toBeCloseTo(0.32, 12);
    const a = dumpAtoms(files.get('plhi.dump') ?? '').get(1)!;
    expect(a.fz).toBeCloseTo(-0.32, 12);
  });

  it('atoms on the far side of the plane feel no force', async () => {
    const { thermo, files } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 5 5 0.6
fix             1 all indent 8.0 plane z 0.4 lo units box
${THERMO_F}
run             0
write_dump      all custom plnone.dump id fx fy fz
`);
    const r = thermo[thermo.length - 1];
    expect(r.f_1).toBeCloseTo(0, 12);
    const a = dumpAtoms(files.get('plnone.dump') ?? '').get(1)!;
    expect(a.fz).toBeCloseTo(0, 12);
  });
});

describe('fix wall/reflect and fix indent argument errors', () => {
  const bad = async (line: string, setup = WALL_SYS): Promise<string> => {
    const { error } = await runScript(`
${setup}
${line}
`);
    expect(error).toBeTruthy();
    return String((error as Error).message);
  };

  it('wall/reflect: missing argument, bad face, bad units, duplicate face, undefined variable', async () => {
    expect(await bad('fix 1 all wall/reflect zlo')).toMatch(/usage/);
    expect(await bad('fix 1 all wall/reflect qlo EDGE')).toMatch(/unknown argument/);
    expect(await bad('fix 1 all wall/reflect zlo EDGE units both')).toMatch(/units must be/);
    expect(await bad('fix 1 all wall/reflect zlo EDGE zlo EDGE')).toMatch(/more than once/);
    expect(await bad('fix 1 all wall/reflect zhi v_novar')).toMatch(/does not exist/);
  });

  it('wall/reflect: a wall in a periodic dimension is rejected', async () => {
    const per = `
units           lj
atom_style      atomic
region          box block 0 10 0 10 0 10
create_box      1 box
mass            1 1.0
`;
    expect(await bad('fix 2 all wall/reflect zlo EDGE\nrun 0', per)).toMatch(/non-periodic/);
  });

  it('indent: bad gstyle, cone unsupported, K not a number, bad side/keyword/variable', async () => {
    expect(await bad('fix 1 all indent 10.0 ball 5 5 5 2')).toMatch(/gstyle/);
    expect(await bad('fix 1 all indent 10.0 cone z 5 5 1 2 0 3')).toMatch(/cone/);
    expect(await bad('variable kk equal 5.0\nfix 1 all indent v_kk sphere 5 5 5 2')).toMatch(/K must be a number/);
    expect(await bad('fix 1 all indent 10.0 sphere 5 5 5 2 side maybe')).toMatch(/side must be/);
    expect(await bad('fix 1 all indent 10.0 sphere 5 5 5 2 bogus')).toMatch(/unknown keyword/);
    expect(await bad('fix 1 all indent 10.0 sphere 5 5 v_novar 2')).toMatch(/does not exist/);
    expect(await bad('fix 1 all indent 8.0 plane z 0.4 middle')).toMatch(/lo or hi/);
    expect(await bad('fix 1 all indent 8.0 plane z 0.4 lo side in')).toMatch(/side keyword/);
  });

  it('indent: fix_modify virial is not supported (only energy is documented)', async () => {
    const { error } = await runScript(`
${INDENT_SYS}
create_atoms    1 single 6 5 5
fix             1 all indent 10.0 sphere 5 5 5 2.0 units box
fix_modify      1 virial yes
`);
    expect(error).toBeTruthy();
    expect(String((error as Error).message)).toMatch(/virial/);
  });
});
