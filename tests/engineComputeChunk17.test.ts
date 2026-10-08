import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Wave-17 chunk and shape computes (src/engine/compute/chunk17.ts). The
 * expected numbers below are either closed-form (worked by hand from the
 * definitions on docs.lammps.org) or were measured with native LAMMPS (black
 * box) on the same input; the oracle cases w17chunk_* hold the full native
 * comparisons.
 */

/** Five atoms: chunk 1 = atoms 1-3 (masses 1, 1, 2), chunk 2 = atom 4, chunk 3 empty, chunk 4 = atom 5. */
const DATA = `five atoms
5 atoms
2 atom types
0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0
2 2.0

Atoms # full

1 1 1 0.5 0.0 0.0 0.0 0 0 0
2 1 1 -0.2 1.0 0.0 0.0 0 0 0
3 1 2 0.3 0.0 2.0 0.5 0 0 0
4 2 1 0.0 5.0 5.0 5.0 0 0 0
5 4 2 0.0 6.0 5.0 5.0 0 0 0

Velocities

1 0.1 0.2 0.3
2 -0.2 0.1 0.0
3 0.0 0.4 -0.1
4 1.0 0.0 0.5
5 0.3 -0.6 0.2
`;

const HEAD = `units lj
atom_style full
boundary f f f
read_data five.data
timestep 0.001
compute cm all chunk/atom molecule compress no
`;

/** Runs a script; returns thermo rows or the engine's error message. */
const run = async (text: string, files: Record<string, string> = { 'five.data': DATA }) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  try {
    await session.execute(text);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return { rows: [] as ThermoRow[], error: err && 'message' in err ? String(err.message) : String(e) };
  }
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, error: null as string | null };
};

const close = (got: number, want: number, abs = 1e-9) => Math.abs(got - want) <= abs + 1e-9 * Math.abs(want);

describe('chunk computes (wave 17)', () => {
  it('angmom/chunk, inertia/chunk and gyration/chunk tensor agree with the hand formulas', async () => {
    const { rows, error } = await run(`${HEAD}
compute a all angmom/chunk cm
compute ine all inertia/chunk cm
compute gt all gyration/chunk cm tensor
thermo_modify format float %.15g norm no
thermo_style custom step c_ine[1][1] c_ine[1][2] c_ine[1][3] c_ine[1][4] c_ine[1][5] c_ine[1][6] c_gt[1][1] c_gt[1][2] c_gt[1][3] c_gt[1][4] c_gt[1][5] c_gt[1][6]
run 0
`);
    expect(error).toBeNull();
    const r = rows[0];
    // Hand values for chunk 1: masses 1, 1, 2 at (0,0,0), (1,0,0), (0,2,0.5); M = 4, r_cm = (0.25, 1, 0.25).
    // Ixx = sum m (dy^2 + dz^2) = 4.25, Iyy = 1, Izz = 4.75; Ixy = -sum m dx dy = 1, Iyz = -sum m dy dz = -1, Ixz = -sum m dx dz = 0.25.
    const ine = [4.25, 1, 4.75, 1, -1, 0.25];
    ine.forEach((v, j) => expect(close(r[`c_ine[1][${j + 1}]`] as number, v)).toBe(true));
    // Gyration tensor = (1/M) sum m dr_a dr_b: (0.1875, 1, 0.0625, -0.25, -0.0625, 0.25).
    const gt = [0.1875, 1, 0.0625, -0.25, -0.0625, 0.25];
    gt.forEach((v, j) => expect(close(r[`c_gt[1][${j + 1}]`] as number, v)).toBe(true));
  });

  it('omega/chunk and angmom/chunk: angular momentum about the centre of mass', async () => {
    const { rows, error } = await run(`${HEAD}
compute a all angmom/chunk cm
compute o all omega/chunk cm
thermo_modify format float %.15g norm no
thermo_style custom step c_a[1][1] c_a[1][2] c_a[1][3] c_o[1][1] c_o[1][2] c_o[1][3] c_o[2][1] c_o[3][1] c_o[4][2]
run 0
`);
    expect(error).toBeNull();
    const r = rows[0];
    // Measured with native LAMMPS (black box): angmom of chunk 1 = (-0.625, 0.05, -0.275); omega of chunk 1 = (-0.21176471, 0.27235294, 0.010588235);
    // single-atom and empty chunks give zero omega.
    expect(close(r['c_a[1][1]'] as number, -0.625)).toBe(true);
    expect(close(r['c_a[1][2]'] as number, 0.05)).toBe(true);
    expect(close(r['c_a[1][3]'] as number, -0.275)).toBe(true);
    expect(close(r['c_o[1][1]'] as number, -0.21176471, 1e-7)).toBe(true);
    expect(close(r['c_o[1][2]'] as number, 0.27235294, 1e-7)).toBe(true);
    expect(close(r['c_o[1][3]'] as number, 0.010588235, 1e-7)).toBe(true);
    expect(r['c_o[2][1]']).toBe(0);
    expect(r['c_o[3][1]']).toBe(0);
    expect(r['c_o[4][2]']).toBe(0);
  });

  it('omega/chunk of a linear chunk is the minimum-norm solution (measured with native LAMMPS, black box)', async () => {
    const LINE = `two atoms along (1,1,0)
2 atoms
1 atom types
0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # full

1 1 1 0.0 3.0 3.0 3.0 0 0 0
2 1 1 0.0 4.0 4.0 3.0 0 0 0

Velocities

1 0.0 0.0 0.5
2 0.0 0.0 -0.5
`;
    const { rows, error } = await run(`units lj
atom_style full
boundary f f f
read_data line.data
compute cm all chunk/atom molecule compress no
compute o all omega/chunk cm
thermo_modify format float %.15g norm no
thermo_style custom step c_o[1][1] c_o[1][2] c_o[1][3]
run 0
`, { 'line.data': LINE });
    expect(error).toBeNull();
    expect(close(rows[0]['c_o[1][1]'] as number, -0.5)).toBe(true);
    expect(close(rows[0]['c_o[1][2]'] as number, 0.5)).toBe(true);
    expect(close(rows[0]['c_o[1][3]'] as number, 0)).toBe(true);
  });

  it('dipole and dipole/chunk: mass and geometry net-charge corrections', async () => {
    const { rows, error } = await run(`${HEAD}
compute dip all dipole
compute dipg all dipole geometry
compute dm all dipole/chunk cm
compute dg all dipole/chunk cm geometry
thermo_modify format float %.15g norm no
thermo_style custom step c_dip[1] c_dip[2] c_dip[3] c_dip c_dipg[1] c_dipg[2] c_dipg[3] c_dm[1][1] c_dm[1][2] c_dm[1][4] c_dg[1][1] c_dg[1][2] c_dg[1][3] c_dg[2][4]
run 0
`);
    expect(error).toBeNull();
    const r = rows[0];
    // Measured with native LAMMPS (black box): system dipole (-1.7428571, -1.0285714, -1.2214286), |d| = 2.3637677;
    // geometric centre (-1.64, -0.84, -1.11).
    expect(close(r['c_dip[1]'] as number, -1.7428571, 1e-7)).toBe(true);
    expect(close(r['c_dip[2]'] as number, -1.0285714, 1e-7)).toBe(true);
    expect(close(r['c_dip[3]'] as number, -1.2214286, 1e-7)).toBe(true);
    expect(close(r.c_dip as number, 2.3637677, 1e-7)).toBe(true);
    expect(close(r['c_dipg[1]'] as number, -1.64)).toBe(true);
    expect(close(r['c_dipg[2]'] as number, -0.84)).toBe(true);
    expect(close(r['c_dipg[3]'] as number, -1.11)).toBe(true);
    // Chunk 1 (net charge 0.6): mass centre gives (-0.35, 0, 0); geometric centre gives (-0.4, 0.2, 0.05).
    expect(close(r['c_dm[1][1]'] as number, -0.35)).toBe(true);
    expect(close(r['c_dm[1][2]'] as number, 0)).toBe(true);
    expect(close(r['c_dm[1][4]'] as number, 0.35)).toBe(true);
    expect(close(r['c_dg[1][1]'] as number, -0.4)).toBe(true);
    expect(close(r['c_dg[1][2]'] as number, 0.2)).toBe(true);
    expect(close(r['c_dg[1][3]'] as number, 0.05)).toBe(true);
    // Chunk 2 has a single neutral atom: zero dipole.
    expect(close(r['c_dg[2][4]'] as number, 0)).toBe(true);
  });

  it('reduce/chunk: sum, min and max; empty chunks give 0, 1e20 and -1e20 (measured with native LAMMPS, black box)', async () => {
    const { rows, error } = await run(`${HEAD}
compute pa all property/atom x y
compute rsum all reduce/chunk cm sum c_pa[1]
compute rmin all reduce/chunk cm min c_pa[1] c_pa[2]
compute rmax all reduce/chunk cm max c_pa[1]
thermo_modify format float %.15g norm no
thermo_style custom step c_rsum[1] c_rsum[3] c_rmin[1][1] c_rmin[3][1] c_rmin[1][2] c_rmax[1] c_rmax[3]
run 0
`);
    expect(error).toBeNull();
    const r = rows[0];
    expect(r['c_rsum[1]']).toBe(1);
    expect(r['c_rsum[3]']).toBe(0);
    expect(r['c_rmin[1][1]']).toBe(0);
    expect(r['c_rmin[3][1]']).toBe(1e20);
    expect(r['c_rmin[1][2]']).toBe(0);
    expect(r['c_rmax[1]']).toBe(1);
    expect(r['c_rmax[3]']).toBe(-1e20);
  });

  it('chunk/spread/atom feeds per-chunk values back to atoms; atoms outside a chunk get 0', async () => {
    const { rows, error } = await run(`${HEAD}
compute com all com/chunk cm
compute sp all chunk/spread/atom cm c_com[1] c_com[3]
compute cnt all property/chunk cm count
compute spm all chunk/spread/atom cm c_cnt
compute ps all reduce/chunk cm sum c_spm
dump d all custom 1 spread.dump id c_sp[1] c_sp[2] c_spm
thermo_modify format float %.15g norm no
thermo_style custom step c_ps[1] c_ps[2] c_ps[3] c_ps[4]
run 0
`, { 'five.data': DATA });
    expect(error).toBeNull();
    // Chunk 1 holds 3 atoms, so spread counts sum to 3 * 3 = 9 for chunk 1 and 1 for chunks 2 and 4; chunk 3 is empty.
    expect(close(rows[0]['c_ps[1]'] as number, 9)).toBe(true);
    expect(close(rows[0]['c_ps[2]'] as number, 1)).toBe(true);
    expect(close(rows[0]['c_ps[3]'] as number, 0)).toBe(true);
    expect(close(rows[0]['c_ps[4]'] as number, 1)).toBe(true);
  });

  it('momentum is extensive: sum of m v (thermo norm off)', async () => {
    const { rows, error } = await run(`${HEAD}
compute mom all momentum
thermo_style custom step c_mom[1] c_mom[2] c_mom[3]
thermo_modify format float %.15g norm no
run 0
`);
    expect(error).toBeNull();
    // Hand sum: x 1.5, y -0.1, z 1.0.
    expect(close(rows[0]['c_mom[1]'] as number, 1.5)).toBe(true);
    expect(close(rows[0]['c_mom[2]'] as number, -0.1)).toBe(true);
    expect(close(rows[0]['c_mom[3]'] as number, 1.0)).toBe(true);
  });

  it('gyration/shape: eigenvalues in descending order, then b, c and k (measured with native LAMMPS, black box)', async () => {
    const { rows, error } = await run(`${HEAD}
compute gyr all gyration
compute gsh all gyration/shape gyr
compute gt all gyration/chunk cm tensor
compute gshc all gyration/shape/chunk gt
thermo_modify format float %.15g norm no
thermo_style custom step c_gsh[1] c_gsh[2] c_gsh[3] c_gsh[4] c_gsh[5] c_gsh[6] c_gshc[1][1] c_gshc[1][4] c_gshc[1][5] c_gshc[1][6] c_gshc[3][1]
run 0
`);
    expect(error).toBeNull();
    const r = rows[0];
    const want = [16.827296, 0.57774222, 0.033737272, 16.521556, 0.54400495, 0.89830317];
    want.forEach((v, j) => expect(close(r[`c_gsh[${j + 1}]`] as number, v, 1e-6)).toBe(true));
    // Chunk 1 (three atoms, a plane): eigenvalues 1.1327524, 0.1172476, 0.
    expect(close(r['c_gshc[1][1]'] as number, 1.1327524, 1e-6)).toBe(true);
    expect(close(r['c_gshc[1][4]'] as number, 1.0741286, 1e-6)).toBe(true);
    expect(close(r['c_gshc[1][5]'] as number, 0.1172476, 1e-6)).toBe(true);
    expect(close(r['c_gshc[1][6]'] as number, 0.745, 1e-6)).toBe(true);
    // Empty chunk 3: eigenvalues zero (k is not a number, as measured).
    expect(r['c_gshc[3][1]']).toBe(0);
  });

  it('property/chunk: count and id (id needs compress yes)', async () => {
    const ok = await run(`${HEAD}
compute cc all chunk/atom molecule compress yes
compute cnt all property/chunk cc count id
thermo_modify format float %.15g norm no
thermo_style custom step c_cnt[1][1] c_cnt[2][1] c_cnt[3][1] c_cnt[1][2] c_cnt[3][2]
run 0
`);
    expect(ok.error).toBeNull();
    // Compressed chunks: molecule 1 -> chunk 1 (3 atoms), molecule 2 -> chunk 2 (1 atom), molecule 4 -> chunk 3 (1 atom).
    expect(ok.rows[0]['c_cnt[1][1]']).toBe(3);
    expect(ok.rows[0]['c_cnt[2][1]']).toBe(1);
    expect(ok.rows[0]['c_cnt[3][1]']).toBe(1);
    expect(ok.rows[0]['c_cnt[1][2]']).toBe(1);
    expect(ok.rows[0]['c_cnt[3][2]']).toBe(4);
  });

  it('unsupported keywords and styles name themselves (StyleError)', async () => {
    const tip4p = await run(`${HEAD}\ncompute d all dipole/tip4p\nrun 0\n`);
    expect(tip4p.error).toMatch(/dipole\/tip4p/);
    const coord = await run(`${HEAD}\ncompute p all property/chunk cm coord1\nrun 0\n`);
    expect(coord.error).toMatch(/coord1/);
    const noTensor = await run(`${HEAD}\ncompute g all gyration/chunk cm\ncompute s all gyration/shape/chunk g\nrun 0\n`);
    expect(noTensor.error).toMatch(/tensor/);
    const kw = await run(`${HEAD}\ncompute g all gyration/chunk cm bogus\nrun 0\n`);
    expect(kw.error).toMatch(/bogus/);
    const id = await run(`${HEAD}\ncompute p all property/chunk cm id\nthermo_style custom step c_p[1]\nrun 0\n`);
    expect(id.error).toMatch(/compress/);
    const reduceMode = await run(`${HEAD}\ncompute r all reduce/chunk cm avg c_pa[1]\nrun 0\n`);
    expect(reduceMode.error).toMatch(/mode/);
  });
});
