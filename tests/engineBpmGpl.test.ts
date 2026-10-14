import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { quatrotvec, normalize3, quatquat, qconjugate } from '../src/engine-gpl/bpm/quat';

// BPM styles ported from LAMMPS (src/engine-gpl/bpm, commit b332f28): bond_style bpm/rotational (full), bpm/zero,
// bpm/spring/plastic and bpm/peri (PMB), pair_style bpm/peri, and the fix nve/bpm/sphere update dipole keyword.
// The numbers the oracle cases (tests/oracle/w59bpm_*.in) check against native LAMMPS are not repeated here; these
// tests cover the refusals by name, the quaternion helpers and a small break.

const DIMER = `units lj
newton on off
atom_style bpm/sphere
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 1.0 0.0 0.0
create_bonds single/bond 1 1 2
comm_modify vel yes cutoff 2.6
pair_style zero 2.0
pair_coeff * *
special_bonds lj 0 1 1 coul 1 1 1
`;

interface Row { step: number; pe: number; ebond: number; c_tb?: number; [k: string]: number | undefined }

const run = async (script: string): Promise<{ rows: Row[]; errors: string[] }> => {
  const rows: Row[] = [];
  const errors: string[] = [];
  const s = new Session({
    emit: (e: { kind: string; row?: Row; message?: string }) => {
      if (e.kind === 'thermo' && e.row) rows.push(e.row);
      if (e.kind === 'error') errors.push(e.message ?? '');
    },
    writeFile: () => {},
  } as never);
  try {
    await s.execute(script);
  } catch (e) {
    errors.push(e instanceof Error ? e.message : String(e));
  }
  return { rows, errors };
};

describe('bpm styles ported from LAMMPS (engine-gpl)', () => {
  it('refuses store/local for the bond styles whose breaks write no rows, by name', async () => {
    for (const style of ['bpm/zero', 'bpm/spring/plastic', 'bpm/peri']) {
      const r = await run(`${DIMER}bond_style ${style} store/local brk 100 time\n`);
      expect(r.errors.join(' '), style).toMatch(/store\/local/);
    }
  });

  it('accepts store/local for bpm/rotational and bpm/spring (the broken bonds go to fix store/local)', async () => {
    for (const style of ['bpm/rotational', 'bpm/spring']) {
      const r = await run(`${DIMER}bond_style ${style} store/local brk 100 time id1 id2\n`);
      expect(r.errors, style).toEqual([]);
    }
  });

  it('refuses an unknown peridynamic model by name (pmb, lps, ves and eps are implemented)', async () => {
    const r = await run(`${DIMER}bond_style bpm/peri\nbond_coeff 1 foo 1e10 1e10 0.0015 0.0005 0.25\n`);
    expect(r.errors.join(' ')).toMatch(/model: foo/);
  });

  it('needs the per-atom vfrac property for bond_style bpm/peri', async () => {
    const r = await run(`${DIMER}bond_style bpm/peri\nbond_coeff 1 pmb 1e10 1.5 0.0005 0.25\nlattice sc 0.0005\nrun 0\n`);
    expect(r.errors.join(' ')).toMatch(/vfrac/);
  });

  it('refuses manybody yes for bpm/zero and unknown keywords for bpm/spring/plastic', async () => {
    const z = await run(`${DIMER}bond_style bpm/zero manybody yes\n`);
    expect(z.errors.join(' ')).toMatch(/manybody yes/);
    const p = await run(`${DIMER}bond_style bpm/spring/plastic foo yes\n`);
    expect(p.errors.join(' ')).toMatch(/invalid argument foo/);
  });

  it('refuses the dipole/dlm integrator of fix nve/bpm/sphere, by name', async () => {
    const r = await run(`${DIMER}fix 1 all nve/bpm/sphere update dipole/dlm\nrun 0\n`);
    expect(r.errors.join(' ')).toMatch(/dipole\/dlm is not supported/);
  });

  it('breaks a bpm/zero bond stretched beyond ecrit, after the setup run, and reports it gone', async () => {
    const r = await run(`${DIMER}bond_style bpm/zero\nbond_coeff 1 0.2\ncompute nb all nbond/atom\ncompute tb all reduce sum c_nb\nthermo_style custom step pe ebond c_tb\nthermo_modify format float %.15g norm no\nfix 1 all nve/bpm/sphere\nrun 0\nset atom 2 x 1.4\nrun 0\nrun 3\n`);
    expect(r.errors).toEqual([]);
    expect(r.rows[1].c_tb).toBe(2);
    expect(r.rows[r.rows.length - 1].c_tb).toBe(0);
  });
});

describe('quaternion helpers (MathExtra ports)', () => {
  it('rotates a vector by a unit quaternion (90 degrees about z maps x to y)', () => {
    const h = Math.SQRT1_2;
    const out = quatrotvec([h, 0, 0, h], [1, 0, 0]);
    expect(out[0]).toBeCloseTo(0, 14);
    expect(out[1]).toBeCloseTo(1, 14);
    expect(out[2]).toBeCloseTo(0, 14);
  });

  it('keeps a zero vector zero in normalize3 (MathExtra::norm3 skips a zero length)', () => {
    expect(normalize3([0, 0, 0])).toEqual([0, 0, 0]);
    const u = normalize3([0, 3, 4]);
    expect(u[1]).toBeCloseTo(0.6, 14);
    expect(u[2]).toBeCloseTo(0.8, 14);
  });

  it('composes a quaternion with its conjugate to the identity', () => {
    const q = [0.5, 0.5, 0.5, 0.5];
    const id = quatquat(q, qconjugate(q));
    expect(id[0]).toBeCloseTo(1, 14);
    expect(Math.hypot(id[1], id[2], id[3])).toBeCloseTo(0, 14);
  });
});
