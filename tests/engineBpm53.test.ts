import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

// BPM wave 53: atom_style bpm/sphere, fix nve/bpm/sphere, bond_style bpm/rotational (ported to src/engine-gpl/bpm), and the
// special-bond weight of granular contacts. The numbers checked here were measured with native LAMMPS (black box);
// the oracle cases tests/oracle/w53bpm_*.in compare every thermo row and the final atom state with native LAMMPS.

const DIMER = (x2 = 1.0) => `units lj
newton on off
atom_style bpm/sphere
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single ${x2} 0.0 0.0
create_bonds single/bond 1 1 2
comm_modify vel yes cutoff 2.6
pair_style zero 2.0
pair_coeff * *
special_bonds lj 0 1 1 coul 1 1 1
`;

interface Row { step: number; pe: number; ebond: number; ke?: number; c_tb?: number; [k: string]: number | undefined }

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

const TH = 'thermo_style custom step pe ebond ke c_tb\nthermo_modify format float %.15g norm no\ncompute nb all nbond/atom\ncompute tb all reduce sum c_nb\n';

describe('atom_style bpm/sphere', () => {
  it('create_atoms gives a sphere of radius 0.5 with the unit quaternion 1 0 0 0', async () => {
    const r = await run(`units lj
atom_style bpm/sphere
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
compute q all property/atom quatw quati radius
compute qs all reduce sum c_q[1] c_q[2] c_q[3]
thermo_style custom step c_qs[1] c_qs[2] c_qs[3]
thermo_modify format float %.15g norm no
run 0
`);
    expect(r.errors).toEqual([]);
    const row = r.rows[0] as unknown as Record<string, number>;
    expect(row['c_qs[1]']).toBe(1);
    expect(row['c_qs[2]']).toBe(0);
    expect(row['c_qs[3]']).toBe(0.5);
  });

  it('set quat works on bpm/sphere atoms (hybrid with dipole too), as native sets the per-atom quat array', async () => {
    // Measured with native LAMMPS (black box, atom_style hybrid bpm/sphere dipole, in.bpm.wires): "set group all quat 1 0 0 0"
    // is accepted. set.html: the quaternion of a rotation theta about (a,b,c) is (cos(theta/2), a sin, b sin, c sin);
    // "set group all quat 0 0 1 90" gives (cos 45, 0, 0, sin 45). src/set.cpp process_quat, quat_flag branch, b332f28.
    const r = await run(`units lj
atom_style hybrid bpm/sphere dipole
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
compute q all property/atom quatw quati quatj quatk
compute qs all reduce sum c_q[1] c_q[2] c_q[3] c_q[4]
thermo_style custom step c_qs[1] c_qs[2] c_qs[3] c_qs[4]
thermo_modify format float %.15g norm no
set group all quat 1 0 0 0
run 0
`);
    expect(r.errors).toEqual([]);
    const r2 = await run(`units lj
atom_style hybrid bpm/sphere dipole
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
compute q all property/atom quatw quati quatj quatk
compute qs all reduce sum c_q[1] c_q[2] c_q[3] c_q[4]
thermo_style custom step c_qs[1] c_qs[2] c_qs[3] c_qs[4]
thermo_modify format float %.15g norm no
set group all quat 0 0 1 90
run 0
`);
    expect(r2.errors).toEqual([]);
    const row = r2.rows[0] as unknown as Record<string, number>;
    expect(row['c_qs[1]']).toBeCloseTo(Math.SQRT1_2, 12);
    expect(row['c_qs[4]']).toBeCloseTo(Math.SQRT1_2, 12);
  });

  it('accepts the optional 0/1 argument and rejects a bad one with the usage text', async () => {
    expect((await run('atom_style bpm/sphere 1\n')).errors).toEqual([]);
    const bad = await run('atom_style bpm/sphere 2\n');
    expect(bad.errors.join(' ')).toMatch(/usage: atom_style bpm\/sphere/);
  });
});

describe('fix nve/bpm/sphere', () => {
  it('refuses update dipole without a dipole moment, disc in 3d, and needs atom_style bpm/sphere', async () => {
    const up = await run(`${DIMER()}${TH}fix 1 all nve/bpm/sphere update dipole\nrun 0\n`);
    expect(up.errors.join(' ')).toMatch(/update/);
    const disc = await run(`${DIMER()}${TH}fix 1 all nve/bpm/sphere disc\nrun 0\n`);
    expect(disc.errors.join(' ')).toMatch(/disc requires 2d/);
    const sph = await run(`units lj
atom_style sphere
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0
fix 1 all nve/bpm/sphere
run 0
`);
    expect(sph.errors.join(' ')).toMatch(/requires atom style bpm\/sphere/);
  });
});

describe('bond_style bpm/rotational', () => {
  it('refuses store/local, unknown keywords and bad values by name', async () => {
    expect((await run(`${DIMER()}bond_style bpm/rotational store/local brk 100 time\n`)).errors.join(' ')).toMatch(/store\/local/);
    expect((await run(`${DIMER()}bond_style bpm/rotational frame sideways\n`)).errors.join(' ')).toMatch(/frame must be/);
    expect((await run(`${DIMER()}bond_style bpm/rotational foo yes\n`)).errors.join(' ')).toMatch(/illegal bond_style bpm\/rotational keyword 'foo'/);
  });

  it('refuses break no with smoothing on, as native does, and accepts break no smooth no', async () => {
    expect((await run(`${DIMER()}bond_style bpm/rotational break no\n`)).errors.join(' ')).toMatch(/turn off smoothing with break no/);
    const ok = await run(`${DIMER()}bond_style bpm/rotational break no smooth no\nbond_coeff 1 1.0 0 0 0 1.0 1.0 1.0 1.0 0.1 0 0 0\nrun 0\n`);
    expect(ok.errors).toEqual([]);
  });

  it('accepts the shear, twisting and bending terms (full bpm/rotational, see engineBpmGpl.test.ts) and accepts zero critical values as native does', async () => {
    const shear = await run(`${DIMER()}bond_style bpm/rotational\nbond_coeff 1 1.0 0.2 0 0 1.0 1.0 1.0 1.0 0.1 0.1 0.1 0.1\nrun 0\n`);
    expect(shear.errors).toEqual([]);
    // native BondBPMRotational::coeff takes fcr, fcs, tct, tcb as given (no range check); zero is accepted
    // (oracle w105cf_bpm_rot_zero)
    const zero = await run(`${DIMER()}bond_style bpm/rotational\nbond_coeff 1 1.0 0 0 0 1.0 0.0 1.0 1.0 0.1 0 0 0\nrun 0\n`);
    expect(zero.errors).toEqual([]);
  });

  it('keeps the bond at a setup strain below the critical value and breaks it at 1.01 (B = 1.01)', async () => {
    // measured with native LAMMPS (black box): a stretch of 0.99 gives force -0.039009950100 on atom 2 and energy 0.49005;
    // a stretch of 1.01 breaks at setup (zero force, zero energy, compute nbond/atom reports the bond gone)
    const kept = await run(`${DIMER(1.0)}bond_style bpm/rotational\nbond_coeff 1 1.0 0 0 0 1.0 1.0 1.0 1.0 0.5 0 0 0\n${TH}fix 1 all nve/bpm/sphere\nrun 0\nset atom 2 x 1.99\nrun 0\n`);
    expect(kept.errors).toEqual([]);
    expect(kept.rows[1].ebond).toBeCloseTo(0.49005, 12);
    const broke = await run(`${DIMER(1.0)}bond_style bpm/rotational\nbond_coeff 1 1.0 0 0 0 1.0 1.0 1.0 1.0 0.5 0 0 0\n${TH}fix 1 all nve/bpm/sphere\nrun 0\nset atom 2 x 2.01\nrun 0\n`);
    expect(broke.errors).toEqual([]);
    expect(broke.rows[1].ebond).toBe(0);
    expect(broke.rows[1].c_tb).toBe(0);
  });
});

describe('pair gran with a 1-2 partner (special_bonds lj 0)', () => {
  // measured with native LAMMPS (black box): two overlapping spheres (diameter 1, 0.8 apart) with pair gran/hooke:
  // without a bond they push apart (ke 11.4823881560248 after 20 steps); with a bond and special_bonds lj 0 1 1 they
  // stay at rest (ke 0).
  const OVERLAP = (bond: boolean) => `units lj
newton on off
atom_style bpm/sphere
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 0.8 0.0 0.0
${bond ? 'create_bonds single/bond 1 1 2' : ''}
comm_modify vel yes cutoff 2.6
pair_style gran/hooke 2000.0 NULL 50.0 NULL 0.5 0
pair_coeff 1 1
special_bonds lj 0 1 1 coul 1 1 1
fix 1 all nve
thermo_style custom step ke
thermo_modify format float %.15g norm no
run 20
`;
  it('applies the lj weight 0 of a 1-2 partner to the contact force', async () => {
    const free = await run(OVERLAP(false));
    expect(free.errors).toEqual([]);
    expect(free.rows[1].ke).toBeCloseTo(11.4823881560248, 8);
    const bonded = await run(OVERLAP(true));
    expect(bonded.errors).toEqual([]);
    expect(bonded.rows[1].ke).toBe(0);
  });
});
