import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/session';

/*
 * compute born/matrix (GPL port, src/engine-gpl/extra_compute/compute_born_matrix.ts): the analytic bonded terms checked
 * against the independent numdiff route and against a hand computation, and native's argument errors. Native parity is
 * in the oracle cases w26cgrid_born*, w417born_pair, w418born_bonded and w419born_cross.
 */

const io = { emit: () => {}, writeFile: () => {} };
const run = async (text: string, files: Record<string, string> = {}): Promise<Session> => {
  const s = new Session(io);
  for (const [name, body] of Object.entries(files)) s.addFile(name, body);
  await s.execute(`${text}\n`);
  return s;
};
const DATA = readFileSync(join(__dirname, 'oracle', 'w418born_bonded.data'), 'utf8');
const CHAINS = `units real
atom_style full
bond_style harmonic
angle_style harmonic
dihedral_style opls
read_data chains.data
pair_style lj/cut 8.0
pair_coeff 1 1 0.10 3.2
pair_coeff 2 2 0.15 3.0
special_bonds lj 0.0 0.0 0.5
bond_coeff 1 300.0 1.52
bond_coeff 2 250.0 1.50
angle_coeff 1 60.0 110.0
angle_coeff 2 50.0 115.0
dihedral_coeff 1 1.3 -0.05 0.2 0.1
compute myvirial all pressure NULL virial
`;

describe('compute born/matrix analytic bonded terms', () => {
  // Measured with native LAMMPS on these chains: the analytic bond and angle terms equal the numdiff values to 1e-10,
  // the analytic opls dihedral term differs from them by about 13% of its size, and lj/cut adds the jump of its force
  // at the cutoff to the finite differences only
  it('equal the finite differences of the virial (numdiff) for bond and angle terms', async () => {
    const BONDED = CHAINS.replace('dihedral_style opls', 'dihedral_style zero').replace('dihedral_coeff 1 1.3 -0.05 0.2 0.1', 'dihedral_coeff *')
      .replace('pair_style lj/cut 8.0\npair_coeff 1 1 0.10 3.2\npair_coeff 2 2 0.15 3.0', 'pair_style zero 8.0\npair_coeff * *');
    const s = await run(`${BONDED}compute ba all born/matrix bond angle
compute bm all born/matrix numdiff 1.0e-5 myvirial
run 0`, { 'chains.data': DATA });
    const ba = s.sys.compute('ba').vectorValues(), bm = s.sys.compute('bm').vectorValues();
    let scale = 0;
    for (let m = 0; m < 21; m++) scale = Math.max(scale, Math.abs(ba[m]));
    expect(scale).toBeGreaterThan(100);
    for (let m = 0; m < 21; m++) expect(Math.abs(ba[m] - bm[m]) / scale, `C[${m + 1}]`).toBeLessThan(1e-9);
  });

  it('a stretched harmonic bond along x: C11 = (2K - 2K (r - r0) / r) r^2, nothing in the shear-only components', async () => {
    const s = await run(`units lj
atom_style bond
region box block 0 10 0 10 0 10
create_box 1 box bond/types 1 extra/bond/per/atom 1
create_atoms 1 single 4.0 5.0 5.0
create_atoms 1 single 5.2 5.0 5.0
mass 1 1.0
create_bonds single/bond 1 1 2
bond_style harmonic
bond_coeff 1 100.0 1.0
pair_style zero 1.0
pair_coeff * *
compute bb all born/matrix bond
run 0`);
    const v = s.sys.compute('bb').vectorValues();
    expect(v[0]).toBeCloseTo((200 - 200 * 0.2 / 1.2) * 1.44, 10);
    for (let m = 1; m < 21; m++) expect(v[m]).toBe(0);
  });

  it('runs in 2d (native has no dimension check)', async () => {
    const s = await run(`units lj
dimension 2
atom_style atomic
lattice sq 0.8
region box block 0 4 0 4 -0.5 0.5
create_box 1 box
create_atoms 1 box
displace_atoms all random 0.05 0.05 0.0 2741
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
compute ba all born/matrix
compute myvirial all pressure NULL virial
compute bm all born/matrix numdiff 1.0e-6 myvirial
run 0`);
    const ba = s.sys.compute('ba').vectorValues(), bm = s.sys.compute('bm').vectorValues();
    expect(Math.abs(ba[0])).toBeGreaterThan(1);
    // z never enters a pair in the plane
    for (const m of [2, 3, 4, 7, 11, 15, 16, 17]) expect(ba[m]).toBe(0);
    for (const m of [0, 1, 5, 6, 10, 14]) expect(bm[m]).toBeCloseTo(ba[m], 5);
  });
});

describe('compute born/matrix native argument errors', () => {
  it('a bonded style without born_matrix, an improper style and hybrid/molecular after a run stop the compute command', async () => {
    await expect(run(`${CHAINS}bond_style zero\nbond_coeff *\ncompute ba all born/matrix bond`, { 'chains.data': DATA }))
      .rejects.toThrow(/Bond style zero does not support compute born\/matrix/);
    await expect(run(`${CHAINS}improper_style harmonic\ncompute ba all born/matrix`, { 'chains.data': DATA }))
      .rejects.toThrow(/Improper style harmonic does not support compute born\/matrix/);
    // the improper keyword is accepted without an improper style, the default set too
    await expect(run(`${CHAINS}compute ba all born/matrix improper\ncompute bb all born/matrix\nrun 0`, { 'chains.data': DATA })).resolves.toBeDefined();
    await expect(run(`${CHAINS}angle_style class2\nangle_coeff * 110.0 10.0 0.0 0.0\nangle_coeff * bb 0.0 1.5 1.5\nangle_coeff * ba 0.0 0.0 1.5 1.5\ncompute ba all born/matrix angle`, { 'chains.data': DATA }))
      .rejects.toThrow(/Angle style class2 does not support compute born\/matrix/);
    await expect(run(`${CHAINS}dihedral_style harmonic\ndihedral_coeff * 1.0 1 2\ncompute ba all born/matrix dihedral`, { 'chains.data': DATA }))
      .rejects.toThrow(/Dihedral style harmonic does not support compute born\/matrix/);
    const MOLEC = CHAINS.replace('pair_style lj/cut 8.0\npair_coeff 1 1 0.10 3.2\npair_coeff 2 2 0.15 3.0', 'pair_style hybrid/molecular lj/cut 8.0 lj/cut 8.0\npair_coeff * * lj/cut 1 0.10 3.2\npair_coeff * * lj/cut 2 0.05 3.2');
    await expect(run(`${MOLEC}compute ba all born/matrix pair\nrun 0`, { 'chains.data': DATA })).resolves.toBeDefined();
    await expect(run(`${MOLEC}run 0\ncompute ba all born/matrix pair`, { 'chains.data': DATA }))
      .rejects.toThrow(/Pair style hybrid\/molecular does not support compute born\/matrix/);
  });

  it('numdiff needs an existing pressure compute', async () => {
    await expect(run(`${CHAINS}compute bm all born/matrix numdiff 1.0e-5 nope`, { 'chains.data': DATA }))
      .rejects.toThrow(/Could not find compute born\/matrix pressure ID nope/);
    await expect(run(`${CHAINS}compute bm all born/matrix numdiff 0 myvirial`, { 'chains.data': DATA }))
      .rejects.toThrow(/Illegal compute born\/matrix command/);
  });
});
