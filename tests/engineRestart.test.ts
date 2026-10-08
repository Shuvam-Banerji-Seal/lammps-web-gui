import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, SimState } from '../src/engine/types';

/*
 * write_restart / read_restart / restart (output/restart.ts, commands/restart.ts).
 * Round trips must reproduce the stored state exactly; forces are not stored
 * (the next run computes them), so they are excluded from the comparison.
 */

const newSession = () => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  return { session, events, files };
};

const LJ_BOX = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.5 1234 loop geom
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
pair_modify     shift yes
timestep        0.004
fix             1 all nve
`;

const MOL_DATA = `molecular test

4 atoms
3 bonds
2 angles
3 atom types
2 bond types
1 angle types

0 6 xlo xhi
0 6 ylo yhi
0 6 zlo zhi

Masses

1 1.0
2 1.0
3 1.0

Atoms # full

1 1 1 0.2 1 1 1
2 1 2 -0.2 2 1.2 1
3 1 2 0.1 3 1.1 2
4 2 3 -0.1 1 3 3

Velocities

1 0.1 0 0
2 0 0.1 0
3 0 0 0.1
4 -0.1 0 0

Bonds

1 1 1 2
2 1 2 3
3 2 3 4

Angles

1 1 1 2 3
2 1 2 3 4
`;

const MOL_BOX = `
units           real
atom_style      full
bond_style      harmonic
angle_style     harmonic
pair_style      lj/cut/coul/cut 2.5 3.0
read_data       mol.data
pair_coeff      * * 0.1 1.0
pair_coeff      1 2 0.2 1.1
pair_modify     shift yes mix arithmetic
bond_coeff      1 100.0 1.0
bond_coeff      2 80.0 1.2
angle_coeff     1 50.0 110.0
special_bonds   lj/coul 0.0 0.5 0.8333
comm_modify     cutoff 3.2
`;

/** Every stored field of the state, except forces (not stored). */
const storedState = (session: Session): Omit<SimState, 'f'> => {
  const { f: _f, ...rest } = session.sys.state;
  return rest;
};

describe('write_restart / read_restart round trip', () => {
  it('reproduces the LJ state (atoms, velocities, images, box, step, time, dt, masses, pair style)', async () => {
    const { session, files } = newSession();
    await session.execute(`${LJ_BOX}run 5\nwrite_restart rt.restart\n`);
    const before = storedState(session);
    const text = files.get('rt.restart');
    expect(text?.startsWith('LAMMPS-WEB-RESTART 1\n')).toBe(true);

    await session.execute('clear\nread_restart rt.restart\n');
    const after = storedState(session);
    expect(after).toEqual(before);
    expect(session.sys.state.step).toBe(5);
    expect(session.sys.ff.pair?.name).toBe('lj/cut');
    expect((session.sys.ff.pair as unknown as { p: { get(n: string, i: number, j: number): number } }).p.get('epsilon', 1, 1)).toBe(1);
    // pair_modify shift was stored with the style
    expect(session.sys.ff.pair?.shift).toBe(true);
  });

  it('keeps the molecular topology, bonded coefficients, special_bonds and comm settings', async () => {
    const { session, files } = newSession();
    session.addFile('mol.data', MOL_DATA);
    await session.execute(`${MOL_BOX}run 0\nwrite_restart mol.restart\n`);
    const before = storedState(session);
    const pairBefore = session.sys.ff.pair!.dataCoeffs();
    const bondBefore = session.sys.ff.bond!.dataCoeffs();
    const angleBefore = session.sys.ff.angle!.dataCoeffs();
    const special = structuredClone(session.sys.ff.special);
    expect(files.get('mol.restart')).toContain('"$ta":"Int32Array"');

    await session.execute('clear\nread_restart mol.restart\n');
    expect(storedState(session)).toEqual(before);
    expect(session.sys.ff.pair!.dataCoeffs()).toEqual(pairBefore);
    expect(session.sys.ff.bond!.dataCoeffs()).toEqual(bondBefore);
    expect(session.sys.ff.angle!.dataCoeffs()).toEqual(angleBefore);
    expect(session.sys.ff.special).toEqual(special);
    expect(session.sys.ff.pair!.shift).toBe(true);
    expect(session.sys.ff.pair!.mix).toBe('arithmetic');
    expect(session.sys.commStyle).toBe('brick');
    expect(session.sys.nb.commCutoff).toBe(3.2);
    expect(session.sys.state.topo.bonds.n).toBe(3);
    expect(session.sys.state.topo.angles.n).toBe(2);
  });

  it('stores the unset mass slot (NaN) and restores it as NaN', async () => {
    const { session, files } = newSession();
    session.addFile('mol.data', MOL_DATA);
    await session.execute(`${MOL_BOX}run 0\nwrite_restart mol.restart\n`);
    // index 0 of massByType is never used and stays NaN, which JSON cannot hold as a number
    expect(files.get('mol.restart')).toContain('{"$n":"NaN"}');
    await session.execute('clear\nread_restart mol.restart\n');
    expect(Number.isNaN(session.sys.state.massByType[0])).toBe(true);
    expect(session.sys.state.massByType[3]).toBe(1);
  });
});

describe('read_restart errors', () => {
  it('refuses to run after a box exists (native message)', async () => {
    const { session, files } = newSession();
    await session.execute(`${LJ_BOX}write_restart rt.restart\n`);
    expect(files.has('rt.restart')).toBe(true);
    await expect(session.execute('read_restart rt.restart\n')).rejects.toThrow(/Cannot use read_restart after simulation box is defined/);
  });

  it('rejects a file that is not a restart file from this engine, naming the conversion path', async () => {
    const { session } = newSession();
    session.addFile('native.restart', 'LAMMPS\u0000\u0001 binary bytes');
    await expect(session.execute('read_restart native.restart\n')).rejects.toThrow(/not a restart file written by this browser engine.*restart2data.*read_data/s);
  });

  it('rejects a missing file', async () => {
    const { session } = newSession();
    await expect(session.execute('read_restart missing.restart\n')).rejects.toThrow(/cannot open file missing\.restart/);
  });

  it('rejects a damaged body and an unknown format version', async () => {
    const { session } = newSession();
    session.addFile('bad.restart', 'LAMMPS-WEB-RESTART 1\n{"format":');
    await expect(session.execute('read_restart bad.restart\n')).rejects.toThrow(/damaged/);
    session.addFile('v2.restart', 'LAMMPS-WEB-RESTART 2\n{}');
    await expect(session.execute('read_restart v2.restart\n')).rejects.toThrow(/not supported/);
  });

  it('refuses % file sets and .mpiio names', async () => {
    const { session } = newSession();
    await expect(session.execute('read_restart save.%\n')).rejects.toThrow(/'%' restart file sets/);
    await expect(session.execute('read_restart save.mpiio\n')).rejects.toThrow(/MPI-IO/);
  });

  it('reports a wildcard with no matching file', async () => {
    const { session } = newSession();
    await expect(session.execute('read_restart none.*\n')).rejects.toThrow(/no file matches the pattern/);
  });
});

describe('write_restart options and unsupported states', () => {
  it('refuses % names, fileper/nfile without %, and .mpiio names', async () => {
    const { session } = newSession();
    await session.execute(LJ_BOX);
    await expect(session.execute('write_restart r.%\n')).rejects.toThrow(/one file per processor/);
    await expect(session.execute('write_restart r.restart nfile 4\n')).rejects.toThrow(/Cannot use write_restart nfile without % in restart file name/);
    await expect(session.execute('write_restart r.restart fileper 2\n')).rejects.toThrow(/Cannot use write_restart fileper without %/);
    await expect(session.execute('write_restart r.mpiio\n')).rejects.toThrow(/MPI-IO/);
  });

  it('keeps fix property/atom data for a fix re-specified with the same ID after read_restart', async () => {
    const { session, files } = newSession();
    await session.execute(`${LJ_BOX}fix p all property/atom i_flag d_val\nset atom 5 i_flag 7\nset atom 6 d_val 2.5\nwrite_restart p.restart\n`);
    const again = newSession();
    again.session.addFile('p.restart', files.get('p.restart')!);
    await again.session.execute('read_restart p.restart\nfix p all property/atom i_flag d_val\nvariable f equal i_flag[5]\nvariable d equal d_val[6]\nprint "f=${f} d=${d}"\n');
    expect(again.events.some((e) => e.kind === 'log' && e.text === 'f=7 d=2.5')).toBe(true);
    // a different layout under the same ID is refused (native may corrupt the data)
    const third = newSession();
    third.session.addFile('p.restart', files.get('p.restart')!);
    await expect(third.session.execute('read_restart p.restart\nfix p all property/atom i_flag\n')).rejects.toThrow(/same properties/);
  });

  it('refuses hybrid pair styles (only the sub-style list is stored natively)', async () => {
    const { session } = newSession();
    await session.execute(`${LJ_BOX}pair_style hybrid lj/cut 2.5\npair_coeff 1 1 lj/cut 1.0 1.0 2.5\n`);
    await expect(session.execute('write_restart h.restart\n')).rejects.toThrow(/pair_style hybrid is not stored/);
  });

  it('needs a box', async () => {
    const { session } = newSession();
    await expect(session.execute('write_restart empty.restart\n')).rejects.toThrow(/simulation box/);
  });
});

describe('restart command', () => {
  it('restart 0 turns output off and is accepted', async () => {
    const { session } = newSession();
    await expect(session.execute('restart 0\n')).resolves.toBeUndefined();
  });

  it('writes periodic restart files with the names and steps native LAMMPS uses', async () => {
    // Measured with native LAMMPS (black box) on this input: the same file names
    const { session, files } = newSession();
    await session.execute(`${LJ_BOX}
restart 5 r.*.eq
restart 4 a.rst b.rst
run 12
variable s equal stride(13,30,7)
restart v_s v.rst
run 15
restart 0
run 5
restart 3 m.rst
minimize 1e-12 1e-12 7 100
`);
    expect([...files.keys()].sort()).toEqual(['a.rst', 'b.rst', 'm.rst.33', 'm.rst.36', 'm.rst.39', 'r.10.eq', 'r.5.eq', 'v.rst.13', 'v.rst.20', 'v.rst.27']);
    // the toggled pair keeps alternating in the second run (a 4, b 8, a 12, b 16, a 20, b 24)
    const stepOf = (name: string) => JSON.parse(files.get(name)!.split('\n')[1]).step;
    expect([stepOf('a.rst'), stepOf('b.rst'), stepOf('r.10.eq'), stepOf('v.rst.27')]).toEqual([20, 24, 10, 27]);
  });

  it('refuses per-processor files and malformed arguments', async () => {
    const { session } = newSession();
    await expect(session.execute('restart 100 a.restart b.restart nfile 2\n')).rejects.toThrow(/nfile needs a '%' file name/);
    await expect(session.execute('restart 100 a.%.restart\n')).rejects.toThrow(/one file per processor/);
    await expect(session.execute('restart v_nope poly.restart\n')).rejects.toThrow(/variable nope does not exist/);
    await expect(session.execute('restart -5 x\n')).rejects.toThrow(/integer >= 0/);
    await expect(session.execute('restart 0 extra\n')).rejects.toThrow(/takes no other arguments/);
    await expect(session.execute('restart 10 a.restart c.restart d.restart\n')).rejects.toThrow(/one file name, or two/);
  });
});

describe('wildcard file names', () => {
  it('writes one file per timestep and reads the largest timestep', async () => {
    const { session, files } = newSession();
    await session.execute(`${LJ_BOX}write_restart w.*\nrun 3\nwrite_restart w.*\n`);
    expect([...files.keys()].filter((k) => k.startsWith('w.')).sort()).toEqual(['w.0', 'w.3']);
    const at3 = storedState(session);
    await session.execute('clear\nread_restart w.*\n');
    expect(session.sys.state.step).toBe(3);
    expect(storedState(session)).toEqual(at3);
  });

  it('compares timesteps numerically (w.100 is newer than w.50)', async () => {
    const { session } = newSession();
    session.addFile('w.50', 'LAMMPS-WEB-RESTART 1\n{}');
    session.addFile('w.100', 'LAMMPS-WEB-RESTART 1\n{}');
    // the chosen file is the one parsed: its body is not a valid restart, so the error names it
    await expect(session.execute('read_restart w.*\n')).rejects.toThrow(/w\.100/);
  });
});
