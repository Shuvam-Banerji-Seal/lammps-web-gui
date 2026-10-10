import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute gaussian/grid/local (wave 38). Parity with native LAMMPS is in tests/oracle/w38gauss_*.in (run by
 * tests/engineOracle.test.ts); this file checks the layout, one analytic value, the example of
 * examples/snap/in.gaussian.grid and the argument errors.
 *
 * Docs: docs.lammps.org/compute_gaussian_grid_local.html ("For each LAMMPS type, a separate sum of Gaussians
 * is calculated, using a separate Gaussian broadening per type."; "looping over the global index ix fastest").
 */

const runScript = async (text: string): Promise<Session> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

/** One type-1 atom at the origin of a 4 x 4 x 4 periodic box; grid 4 4 4 has spacing 1. */
const ONE_ATOM = `units metal
atom_modify map hash
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 single 0 0 0 units box
mass * 180.88
pair_style zero 2.5
pair_coeff * *
`;

/** Normalisation 1 / (sigma sqrt(2 pi))^3 of the Gaussian form. */
const norm = (sigma: number) => 1 / Math.pow(sigma * Math.sqrt(2 * Math.PI), 3);

describe('compute gaussian/grid/local', () => {
  it('has ntypes + 6 columns and one row per grid point, ix fastest, then iy, iz slowest', async () => {
    const s = await runScript(`${ONE_ATOM}compute g all gaussian/grid/local grid 4 3 2 1.0 1.0 1.0 0.5 0.5\nrun 0\n`);
    const c = s.sys.compute('g');
    const v = c.localValues();
    expect(c.sizeLocalCols).toBe(2 + 6);
    expect(c.localRows).toBe(4 * 3 * 2);
    const cols = c.sizeLocalCols;
    for (let r = 0; r < c.localRows; r++) {
      const ix = r % 4, iy = Math.floor(r / 4) % 3, iz = Math.floor(r / 12);
      expect(v[r * cols]).toBe(ix);
      expect(v[r * cols + 1]).toBe(iy);
      expect(v[r * cols + 2]).toBe(iz);
      expect(v[r * cols + 3]).toBeCloseTo(ix / 4 * 4, 12);
      expect(v[r * cols + 4]).toBeCloseTo(iy / 3 * 4, 12);
      expect(v[r * cols + 5]).toBeCloseTo(iz / 2 * 4, 12);
    }
  });

  it('gives the Gaussian of the atom with the cutoff 2 rcutfac R and periodic images (one atom)', async () => {
    const sigma = 0.5;
    const s = await runScript(`${ONE_ATOM}compute g all gaussian/grid/local grid 4 4 4 1.0 1.0 1.0 ${sigma} ${sigma}\nrun 0\n`);
    const c = s.sys.compute('g');
    const v = c.localValues();
    const cols = c.sizeLocalCols;
    const at = (ix: number, iy: number, iz: number) => v[(ix + 4 * (iy + 4 * iz)) * cols + 6];
    // on the atom: the normalisation itself (Measured with native LAMMPS: 0.507949087473928 for sigma 0.5)
    expect(at(0, 0, 0)).toBeCloseTo(norm(sigma), 12);
    // distance 1 along x: norm exp(-r^2 / (2 sigma^2)) with r = 1
    expect(at(1, 0, 0)).toBeCloseTo(norm(sigma) * Math.exp(-1 / (2 * sigma * sigma)), 12);
    // distance 1 across the periodic face: the image at x = 4 is the neighbour of grid point x = 3
    expect(at(3, 0, 0)).toBeCloseTo(norm(sigma) * Math.exp(-1 / (2 * sigma * sigma)), 12);
    // distance exactly 2 = cutoff: zero (strict inequality)
    expect(at(2, 0, 0)).toBe(0);
  });

  it('gives zero for the other type columns when only type 1 atoms exist', async () => {
    const s = await runScript(`${ONE_ATOM}compute g all gaussian/grid/local grid 2 2 2 1.0 1.0 1.0 0.5 0.5\nrun 0\n`);
    const c = s.sys.compute('g');
    const v = c.localValues();
    for (let r = 0; r < c.localRows; r++) expect(v[r * 8 + 7]).toBe(0);
  });

  it('rejects the argument errors that native LAMMPS refuses', async () => {
    const fails = (line: string, re: RegExp) => expect(runScript(`${ONE_ATOM}${line}\nrun 0\n`)).rejects.toThrow(re);
    // native message: Illegal compute grid/local command (the grid keyword is missing)
    await fails('compute g all gaussian/grid/local 2 2 2 1.0 1.0 1.0 0.5 0.5', /expects 'grid nx ny nz/);
    // native message: Illegal compute gaussian/grid/local command (one width missing for two types)
    await fails('compute g all gaussian/grid/local grid 2 2 2 1.0 1.0 1.0 0.5', /expects 'grid nx ny nz/);
    // native message: All grid/local dimensions must be positive
    await fails('compute g all gaussian/grid/local grid 0 2 2 1.0 1.0 1.0 0.5 0.5', /grid counts must be positive integers/);
    // native message: Expected integer parameter instead of '2.5'
    await fails('compute g all gaussian/grid/local grid 2.5 2 2 1.0 1.0 1.0 0.5 0.5', /expected an integer/);
    // native message: Gaussian width for type 1 must be > 0
    await fails('compute g all gaussian/grid/local grid 2 2 2 1.0 1.0 1.0 0 0.5', /Gaussian width for type 1 must be > 0/);
    await fails('compute g all gaussian/grid/local grid 2 2 2 1.0 1.0 1.0 0.5 -0.1', /Gaussian width for type 2 must be > 0/);
    // native message: Expected floating point parameter instead of 'abc'
    await fails('compute g all gaussian/grid/local grid 2 2 2 1.0 abc 1.0 0.5 0.5', /expected a number/);
  });

  it('refuses a cutoff longer than the ghost cutoff (pair cutoff + skin) instead of dropping images', async () => {
    // Measured with native LAMMPS (black box): native does not refuse this (ghost cutoff 3 with pair zero 1.0
    // and skin 2, compute cutoff 2 rcutfac R = 4); its value equals the sum over all images within 4.
    // The engine refuses it with an explicit error rather than summing only the ghosts it has.
    const s = await runScript(`units metal\natom_modify map hash\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 single 0.1 0.1 0.1 units box\nmass * 1.0\npair_style zero 1.0\npair_coeff * *\ncompute g all gaussian/grid/local grid 2 2 2 2.0 1.0 0.5\nrun 0\n`);
    // the compute is evaluated lazily, when its values are requested
    expect(() => s.sys.compute('g').localValues()).toThrow(/ghost cutoff/);
  });
});
