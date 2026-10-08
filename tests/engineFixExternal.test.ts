import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Direct unit checks for the force_ext fix styles (viscous, gravity,
 * lineforce, planeforce, efield): the documented formulas on tiny
 * single-atom systems, plus argument errors. Oracle parity with native
 * LAMMPS for these styles lives in tests/engineOracle.test.ts
 * (w2fdamp_* / w2fefield_ cases).
 */

interface Run {
  thermo: ThermoRow[];
  files: Map<string, string>;
  error: EngineError | null;
}

const runScript = async (text: string): Promise<Run> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { thermo, files, error };
};

/** Per-atom columns of a write_dump custom file, keyed by atom id. */
const dumpAtoms = (r: Run, name: string): Map<number, Record<string, number>> => {
  const text = r.files.get(name) ?? '';
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

const errText = (r: Run): string => {
  expect(r.error, 'expected the script to fail').not.toBeNull();
  return r.error?.message ?? '';
};

/** One atom of mass m at (5 5 5) in a periodic box, velocities zeroed by default. */
const oneAtom = (extra: string, atomStyle = 'atomic', mass = 1.0): string => `
units lj
atom_style ${atomStyle}
boundary p p p
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5 5 5 units box
mass 1 ${mass}
${extra}
`;

const DUMP = 'write_dump all custom ext.dump id fx fy fz vx vy vz modify format float %.17g sort id';

describe('fix viscous (docs.lammps.org/fix_viscous.html: F_i = -gamma v_i)', () => {
  it('damps v by -gamma*dt/m per step', async () => {
    const r = await runScript(oneAtom('fix n all nve\nvelocity all set 1 0 0\ntimestep 0.001\nfix d all viscous 2.0\nthermo 10\nrun 1') + '\n' + DUMP);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    // velocity-Verlet with F = -gamma*v: v1 = v0*(1 - g*dt + (g*dt)^2/4)
    expect(a.vx).toBeCloseTo(0.998001, 12);
    expect(a.vy).toBe(0);
    expect(a.vz).toBe(0);
  });

  it('scale keyword damps each type by its ratio', async () => {
    const script = `
units lj
atom_style atomic
boundary p p p
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 5 5 5 units box
create_atoms 2 single 6 5 5 units box
mass 1 1.0
mass 2 1.0
fix n all nve
velocity all set 1 0 0
timestep 0.001
fix d all viscous 1.0 scale 2 3.0
thermo 10
run 1
${DUMP}
`;
    const r = await runScript(script);
    expect(r.error).toBeNull();
    const atoms = dumpAtoms(r, 'ext.dump');
    // gamma_eff = gamma*ratio: 1.0 for type 1, 3.0 for type 2
    expect(atoms.get(1)!.vx).toBeCloseTo(0.99900025, 12);
    expect(atoms.get(2)!.vx).toBeCloseTo(0.99700225, 12);
  });

  it('rejects bad keywords and arguments', async () => {
    expect((await runScript(oneAtom('fix d all viscous abc'))).error?.message).toContain('gamma');
    expect((await runScript(oneAtom('fix d all viscous 0.5 bogus 1 2'))).error?.message).toContain('bogus');
    expect((await runScript(oneAtom('fix d all viscous 0.5 scale 1 0'))).error?.message).toContain('ratio');
    expect((await runScript(oneAtom('fix d all viscous 0.5 scale 9 1'))).error?.message).toContain('type');
  });
});

describe('fix gravity (docs.lammps.org/fix_gravity.html)', () => {
  it('vector style: same acceleration on each atom, F = m*g', async () => {
    const r = await runScript(oneAtom('mass 1 2.0\nfix g all gravity 0.5 vector 0.0 0.0 -1.0\nthermo 10\nrun 0') + '\n' + DUMP);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBeCloseTo(0, 14);
    expect(a.fy).toBe(0);
    expect(a.fz).toBeCloseTo(-1.0, 14); // m * 0.5 * unit(0,0,-1)
  });

  it('vector style ignores the vector length', async () => {
    const r = await runScript(oneAtom('mass 1 2.0\nfix g all gravity 0.5 vector 0 0 -7.5\nthermo 10\nrun 0') + '\n' + DUMP);
    expect(r.error).toBeNull();
    expect(dumpAtoms(r, 'ext.dump').get(1)!.fz).toBeCloseTo(-1.0, 14);
  });

  it('chute style: angle in +x away from -z (3d)', async () => {
    const r = await runScript(oneAtom('fix g all gravity 1.0 chute 30.0\nthermo 10\nrun 0') + '\n' + DUMP);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBeCloseTo(Math.sin(Math.PI / 6), 14);
    expect(a.fy).toBe(0);
    expect(a.fz).toBeCloseTo(-Math.cos(Math.PI / 6), 14);
  });

  it('spherical style: theta from +z, phi from +x (3d)', async () => {
    // theta=90, phi=-90 must act in -y (doc example); check a general direction too
    const r = await runScript(oneAtom('fix g all gravity 1.0 spherical 30.0 120.0\nthermo 10\nrun 0') + '\n' + DUMP);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    const th = 120 * Math.PI / 180, ph = 30 * Math.PI / 180;
    expect(a.fx).toBeCloseTo(Math.sin(th) * Math.cos(ph), 14);
    expect(a.fy).toBeCloseTo(Math.sin(th) * Math.sin(ph), 14);
    expect(a.fz).toBeCloseTo(Math.cos(th), 14);
  });

  it('spherical style: theta 90, phi -90 acts in -y (doc example)', async () => {
    const r = await runScript(oneAtom('fix g all gravity 1.0 spherical -90.0 90.0\nthermo 10\nrun 0') + '\n' + DUMP);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBeCloseTo(0, 14);
    expect(a.fy).toBeCloseTo(-1, 14);
    expect(a.fz).toBeCloseTo(0, 14);
  });

  it('scalar is the field potential energy -m*(g.x); energy yes adds it to pe', async () => {
    const script = oneAtom('mass 1 2.0\nfix g all gravity 0.5 vector 0 0 -1.0\nfix_modify g energy yes\nthermo_style custom f_g pe\nthermo 10\nrun 0');
    const r = await runScript(script);
    expect(r.error).toBeNull();
    expect(r.thermo).toHaveLength(1);
    // U = -m*(g.x) = -2*0.5*(-1)*5 = 5; one atom -> norm divides by 1
    expect(r.thermo[0].f_g).toBeCloseTo(5, 12);
    expect(r.thermo[0].pe).toBeCloseTo(5, 12);
  });

  it('magnitude can be an equal-style variable', async () => {
    const r = await runScript(oneAtom('variable mg equal 0.25\nfix g all gravity v_mg vector 0 0 -1\nthermo 10\nrun 0') + '\n' + DUMP);
    expect(r.error).toBeNull();
    expect(dumpAtoms(r, 'ext.dump').get(1)!.fz).toBeCloseTo(-0.25, 14);
  });

  it('rejects bad styles and arguments', async () => {
    expect((await runScript(oneAtom('fix g all gravity 1.0 gradient 1 2 3 4 5 6'))).error?.message).toContain('gradient');
    expect((await runScript(oneAtom('fix g all gravity 1.0 chute'))).error?.message).toContain('chute');
    expect((await runScript(oneAtom('fix g all gravity 1.0 vector 0 0 0'))).error?.message).toContain('non-zero');
    expect((await runScript(oneAtom('fix g all gravity 1.0 vector 0 0'))).error?.message).toContain('vector');
  });
});

describe('fix lineforce (docs.lammps.org/fix_lineforce.html)', () => {
  const withGravity = (fixline: string): string =>
    oneAtom('fix g all gravity 1.0 vector 0 0 -1\n' + fixline + '\nthermo 10\nrun 0') + '\n' + DUMP;

  it('keeps only the component along the line', async () => {
    const r = await runScript(withGravity('fix l all lineforce 0.0 0.0 1.0'));
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBe(0);
    expect(a.fy).toBe(0);
    expect(a.fz).toBeCloseTo(-1, 14);
  });

  it('removes the in-plane components (line (1,0,0) kills a -z force)', async () => {
    const r = await runScript(withGravity('fix l all lineforce 1.0 0.0 0.0'));
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBe(0);
    expect(a.fy).toBe(0);
    expect(a.fz).toBe(0);
  });

  it('normalizes the direction vector', async () => {
    const r = await runScript(withGravity('fix l all lineforce 0.0 3.0 4.0'));
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    // F=(0,0,-1); n=(0,0.6,0.8); F' = (F.n)n = -0.8*n = (0,-0.48,-0.64)
    expect(a.fx).toBe(0);
    expect(a.fy).toBeCloseTo(-0.48, 14);
    expect(a.fz).toBeCloseTo(-0.64, 14);
  });

  it('rejects wrong arg counts and zero vectors', async () => {
    expect((await runScript(oneAtom('fix l all lineforce 1 0'))).error?.message).toContain('usage');
    expect((await runScript(oneAtom('fix l all lineforce 0 0 0'))).error?.message).toContain('non-zero');
    expect((await runScript(oneAtom('fix l all lineforce 1 x 0'))).error?.message).toContain('number');
  });
});

describe('fix planeforce (docs.lammps.org/fix_planeforce.html)', () => {
  const withGravity = (fixplane: string): string =>
    oneAtom('fix g all gravity 1.0 vector 0 0 -1\n' + fixplane + '\nthermo 10\nrun 0') + '\n' + DUMP;

  it('removes the component along the normal', async () => {
    const r = await runScript(withGravity('fix p all planeforce 0.0 0.0 1.0'));
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBe(0);
    expect(a.fy).toBe(0);
    expect(a.fz).toBe(0);
  });

  it('keeps in-plane force untouched', async () => {
    const r = await runScript(withGravity('fix p all planeforce 1.0 0.0 0.0'));
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fz).toBeCloseTo(-1, 14);
  });

  it('normalizes the normal vector (F=(0,0,-1), n=(1,1,1))', async () => {
    const r = await runScript(withGravity('fix p all planeforce 1.0 1.0 1.0'));
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    // F.n = -1/sqrt(3); F' = F - (F.n)n = (1/3, 1/3, -2/3)
    expect(a.fx).toBeCloseTo(1 / 3, 14);
    expect(a.fy).toBeCloseTo(1 / 3, 14);
    expect(a.fz).toBeCloseTo(-2 / 3, 14);
  });

  it('rejects wrong arg counts and zero vectors', async () => {
    expect((await runScript(oneAtom('fix p all planeforce 1 0 0 0'))).error?.message).toContain('usage');
    expect((await runScript(oneAtom('fix p all planeforce 0 0 0'))).error?.message).toContain('non-zero');
  });
});

describe('fix efield (docs.lammps.org/fix_efield.html: F = qE)', () => {
  it('adds F = q*E to each charged atom in the group', async () => {
    const script = oneAtom('set group all charge 2.0\nfix e all efield 0.5 0.0 0.0\nthermo 10\nrun 0', 'charge') + '\n' + DUMP;
    const r = await runScript(script);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBeCloseTo(1.0, 14); // q * E * qe2f = 2 * 0.5
    expect(a.fy).toBe(0);
    expect(a.fz).toBe(0);
  });

  it('group-restricted and constant-vector energy U = -q(x.E)', async () => {
    const script = `
units lj
atom_style charge
boundary p p p
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5 5 5 units box
create_atoms 1 single 1 5 5 units box
mass 1 1.0
set group all charge 1.0
group far id 1
fix e far efield 0.5 0.0 0.0
fix_modify e energy yes
thermo_style custom f_e f_e[1] f_e[2] f_e[3] pe
thermo 10
run 0
`;
    const r = await runScript(script);
    expect(r.error).toBeNull();
    expect(r.thermo).toHaveLength(1);
    const t = r.thermo[0] as Record<string, number | undefined>;
    // atom 1 (id 1) at x=5, q=1: fx = 0.5, U = -q*x*Ex = -2.5; atom 2 not in
    // the group. thermo normalizes extensive values by BOTH atoms (norm yes).
    expect(t.f_e).toBeCloseTo(-1.25, 12);
    expect(t.pe).toBeCloseTo(-1.25, 12);
    expect(t['f_e[1]']).toBeCloseTo(0.25, 14); // total added force / natoms
    expect(t['f_e[2]']).toBe(0);
    expect(t['f_e[3]']).toBe(0);
  });

  it('region keyword: atoms outside the region get no force', async () => {
    const script = oneAtom('region half block 0 1 0 10 0 10 units box\nfix e all efield 0.5 0 0 region half\nthermo 10\nrun 0', 'charge') + '\n' + DUMP;
    const r = await runScript(script);
    expect(r.error).toBeNull();
    const a = dumpAtoms(r, 'ext.dump').get(1)!;
    expect(a.fx).toBe(0); // atom at x=5 is outside the region
  });

  it('variable components are evaluated each step', async () => {
    const script = oneAtom('set group all charge 2.0\nvariable ez equal 0.1\nfix e all efield 0.0 0.0 v_ez\nthermo 10\nrun 0', 'charge') + '\n' + DUMP;
    const r = await runScript(script);
    expect(r.error).toBeNull();
    expect(dumpAtoms(r, 'ext.dump').get(1)!.fz).toBeCloseTo(0.2, 14);
  });

  it('energy keyword with a variable field uses the atom-style variable', async () => {
    const script = oneAtom(
      'set group all charge 2.0\n' +
      'variable exq equal 0.5\n' +
      'variable uu atom -0.5*q*x\n' +
      'fix e all efield v_exq 0 0 energy v_uu\n' +
      'fix_modify e energy yes\nthermo_style custom f_e pe\nthermo 10\nrun 0', 'charge');
    const r = await runScript(script);
    expect(r.error).toBeNull();
    // U = -0.5*q*x = -0.5*2*5 = -5 from the variable; force from v_exq is q*0.5 = 1
    expect(r.thermo[0].f_e).toBeCloseTo(-5, 12);
    expect(r.thermo[0].pe).toBeCloseTo(-5, 12);
  });

  it('rejects missing charges, bad keywords and bad variable styles', async () => {
    expect((await runScript(oneAtom('fix e all efield 1 0 0'))).error?.message).toContain('charge');
    expect(errText(await runScript(oneAtom('set group all charge 1.0\nfix e all efield 1 0 0 energy v_u', 'charge')))).toContain('constant vector');
    expect(errText(await runScript(oneAtom('set group all charge 1.0\nvariable u equal 1.0\nfix e all efield 1 0 0 energy v_u', 'charge')))).toContain('constant vector');
    expect(errText(await runScript(oneAtom('set group all charge 1.0\nvariable exq equal 0.5\nvariable u equal 1.0\nfix e all efield v_exq 0 0 energy v_u', 'charge')))).toContain('atom-style');
    expect(errText(await runScript(oneAtom('set group all charge 1.0\nvariable exq equal 0.5\nvariable uu atom -0.5*q*x\nvariable pp atom 0.5*x\nfix e all efield v_exq 0 0 energy v_uu potential v_pp', 'charge')))).toContain('potential');
    expect(errText(await runScript(oneAtom('set group all charge 1.0\nfix e all efield 1 0 0 dipole 1 0 0', 'charge')))).toContain('dipole');
    expect(errText(await runScript(oneAtom('set group all charge 1.0\nfix e all efield 1 0', 'charge')))).toContain('usage');
  });
});
