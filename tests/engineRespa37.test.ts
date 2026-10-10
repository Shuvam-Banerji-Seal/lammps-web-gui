import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { parseRunStyleRespa } from '../src/engine/run/respa';
import { EngineError, type EngineEvent, type ThermoRow } from '../src/engine/types';

/*
 * run_style respa (run/respa.ts): argument parsing and defaults from docs.lammps.org/run_style.html,
 * the refusals (inner/middle/outer, unsupported fixes), and integrator checks. The 3-atom dimer
 * numbers below were measured with native LAMMPS (black box): a bond on level 1 and lj/cut on level 2,
 * lj units (thermo output is per atom), timestep 0.005, 20 steps.
 */

interface Run {
  thermo: ThermoRow[];
  files: Map<string, string>;
  error: EngineError | null;
}

const runScript = async (text: string, inputs: Record<string, string> = {}): Promise<Run> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (name, body, append) => files.set(name, (append ? files.get(name) ?? '' : '') + body),
  });
  for (const [name, body] of Object.entries(inputs)) session.addFile(name, body);
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { thermo, files, error };
};

const DIMER3 = `three atoms

3 atoms
1 bonds
1 atom types
1 bond types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Masses

1 1.0

Atoms # bond

1 1 1 0.0 0.0 0.0
2 1 1 1.3 0.2 0.0
3 1 1 2.0 -1.0 0.5

Velocities

1 0.1 0.0 0.05
2 -0.1 0.0 0.02
3 0.0 0.1 0.0

Bonds

1 1 1 2
`;

const dimerScript = (runStyle: string, extra = '') => `
units lj
atom_style bond
read_data dimer3.data
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
bond_style harmonic
bond_coeff 1 50.0 1.0
timestep 0.005
thermo_style custom step pe ke etotal
thermo_modify format float %.15g
thermo 1
${runStyle}
fix 1 all nve
run 20
${extra}
`;

const MELT = `
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
timestep 0.003
thermo_style custom step pe ke etotal press
thermo_modify format float %.15g
thermo 5
RUNSTYLE
fix 1 all nve
run 20
`;

describe('run_style respa parsing (run_style.html)', () => {
  it('assigns the documented defaults: bonds on level 1, pair and kspace on level N', () => {
    // docs: "respa 2 8 ... bonds, angles, dihedrals will be computed every 0.5 fs ... pair and kspace ... once every 4 fs"
    const p = parseRunStyleRespa(['respa', '2', '8']);
    expect(p).toMatchObject({ levels: 2, loops: [8], bond: 1, angle: 1, dihedral: 1, improper: 1, pair: 2, kspace: 2, hybrid: null });
  });

  it('chains the defaults: angle follows bond, dihedral follows angle, improper follows dihedral, kspace follows pair', () => {
    const p = parseRunStyleRespa(['respa', '2', '2', 'pair', '1', 'kspace', '2']);
    expect(p).toMatchObject({ loops: [2], bond: 1, angle: 1, dihedral: 1, improper: 1, pair: 1, kspace: 2 });
    const q = parseRunStyleRespa(['respa', '4', '2', '2', '2', 'bond', '1', 'dihedral', '2', 'pair', '3', 'kspace', '4']);
    expect(q).toMatchObject({ levels: 4, loops: [2, 2, 2], bond: 1, angle: 1, dihedral: 2, improper: 2, pair: 3, kspace: 4 });
  });

  it('reads one level per hybrid sub-style', () => {
    const p = parseRunStyleRespa(['respa', '3', '4', '2', 'bond', '1', 'hybrid', '2', '2', '1', 'kspace', '3']);
    expect(p.hybrid).toEqual([2, 2, 1]);
    expect(p.kspace).toBe(3);
    expect(p.pair).toBe(3);
  });

  it('refuses the pair splitting keywords by name', () => {
    for (const kw of ['inner', 'middle', 'outer']) {
      const args = kw === 'outer' ? ['respa', '2', '2', 'outer', '2'] : ['respa', '2', '2', kw, '1', '5.0', '6.0'];
      expect(() => parseRunStyleRespa(args)).toThrow(new RegExp(`'${kw}'`));
    }
  });

  it('refuses more than 4 levels, a missing loop factor, a level out of range, and hybrid with pair', () => {
    expect(() => parseRunStyleRespa(['respa', '5', '2', '2', '2', '2'])).toThrow(/2 to 4 levels/);
    expect(() => parseRunStyleRespa(['respa', '1'])).toThrow(/2 to 4 levels/);
    expect(() => parseRunStyleRespa(['respa', '3', '2'])).toThrow(/loop factor 2/);
    expect(() => parseRunStyleRespa(['respa', '2', '2', 'pair', '3'])).toThrow(/level must be an integer 1 to 2/);
    expect(() => parseRunStyleRespa(['respa', '2', '2', 'hybrid', '1', 'pair', '2'])).toThrow(/mutually exclusive/);
    expect(() => parseRunStyleRespa(['respa', '2', '2', 'frob', '1'])).toThrow(/unknown keyword 'frob'/);
  });
});

describe('run_style respa in runs', () => {
  it('refuses inner/middle/outer during a run, naming the keyword', async () => {
    const r = await runScript(`${dimerScript('run_style respa 2 2 inner 1 4.0 5.0 outer 2')}`, { 'dimer3.data': DIMER3 });
    expect(r.error?.message).toMatch(/'inner'.*not supported/);
  });

  it('refuses a fix it does not support, naming the fix and run_style respa', async () => {
    const r = await runScript(`
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
run_style respa 2 2
fix 1 all nve
fix 2 all langevin 1.0 1.0 1.0 4928
run 2
`);
    expect(r.error?.message).toMatch(/fix langevin \(2\) is not supported with run_style respa/);
  });

  it('refuses fix npt (a barostat) under respa', async () => {
    const r = await runScript(MELT.replace('RUNSTYLE', 'run_style respa 2 2').replace('fix 1 all nve', 'fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 2.0'));
    expect(r.error?.message).toMatch(/fix npt \(1\) is not supported with run_style respa/);
  });

  it('refuses a hybrid keyword whose count does not match the pair_style sub-styles', async () => {
    const r = await runScript(`
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style hybrid/overlay lj/cut 2.5 morse 3.0
pair_coeff 1 1 lj/cut 1.0 1.0 2.5
pair_coeff 1 1 morse 0.2 1.6 1.1 3.0
run_style respa 2 2 hybrid 1
fix 1 all nve
run 1
`);
    expect(r.error?.message).toMatch(/hybrid gives 1 levels but pair_style hybrid\/overlay has 2 sub-styles/);
  });

  it('with every term on level 1 and one loop, respa is velocity Verlet: the same thermo as run_style verlet', async () => {
    const verlet = await runScript(MELT.replace('RUNSTYLE', 'run_style verlet'));
    const respa = await runScript(MELT.replace('RUNSTYLE', 'run_style respa 2 1 pair 1'));
    expect(verlet.error).toBeNull();
    expect(respa.error).toBeNull();
    expect(respa.thermo.length).toBe(verlet.thermo.length);
    for (let r = 0; r < verlet.thermo.length; r++) {
      for (const k of ['pe', 'ke', 'etotal', 'press'] as const) {
        const a = verlet.thermo[r][k] as number, b = respa.thermo[r][k] as number;
        expect(Math.abs(a - b)).toBeLessThanOrEqual(1e-10 * Math.max(1, Math.abs(a)));
      }
    }
  });

  it('reproduces the native dimer probe (bond on level 1, pair on level 2)', async () => {
    const r = await runScript(dimerScript('run_style respa 2 2 pair 2', 'write_dump all custom dump.txt id x y z modify format float %.15g sort id'), { 'dimer3.data': DIMER3 });
    expect(r.error).toBeNull();
    // thermo is per atom in lj units; native: step 1 PotEng 1.5127924758368 KinEng 0.0241046869094777
    const step1 = r.thermo[1];
    expect(step1.pe).toBeCloseTo(1.5127924758368, 9);
    expect(step1.ke).toBeCloseTo(0.0241046869094777, 9);
    // native: step 20 PotEng -0.0926548971449025 KinEng 1.62905786756732
    const step20 = r.thermo[20];
    expect(step20.pe).toBeCloseTo(-0.0926548971449025, 9);
    expect(step20.ke).toBeCloseTo(1.62905786756732, 9);
    // native final positions (dump sorted by id): atom 1 and atom 2
    const rows = r.files.get('dump.txt')!.trim().split('\n').slice(-3).map((l) => l.trim().split(/\s+/).map(Number));
    expect(rows[0][1]).toBeCloseTo(0.139064237062646, 9);
    expect(rows[1][1]).toBeCloseTo(1.16430953489853, 9);
  });

  it('accepts fix temp/rescale next to fix nve', async () => {
    const r = await runScript(MELT.replace('RUNSTYLE', 'run_style respa 2 2 pair 1').replace('fix 1 all nve', 'fix 1 all nve\nfix 2 all temp/rescale 2 1.0 1.0 0.05 1.0'));
    expect(r.error).toBeNull();
    expect(r.thermo.every((row) => Number.isFinite(row.etotal as number))).toBe(true);
  });
});
