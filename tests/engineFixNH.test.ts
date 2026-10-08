import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { EngineError, type EngineEvent } from '../src/engine/types';

/*
 * fix nvt / npt / nph keyword consistency — docs.lammps.org/fix_nh.html.
 * "Pstart, Pstop, Pdamp parameters for any coupled dimensions must be identical."
 * "Note that in order to use the xy, xz, or yz keywords, the simulation box must
 * be triclinic, even if its initial tilt factors are 0.0."
 * "X, y, z cannot be barostatted if the associated dimension is not periodic."
 * "For the temp keyword, the final Tstop cannot be 0.0 ..."
 */

const BASE = `
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

const errorOf = async (fix: string, extra = ''): Promise<string | null> => {
  const session = new Session({ emit: (_e: EngineEvent) => {} });
  try {
    await session.execute(`${BASE}${extra}\n${fix}\nrun 0\n`);
  } catch (e) {
    return e instanceof EngineError ? e.message : String(e);
  }
  return null;
};

describe('fix nvt / npt / nph keywords', () => {
  it('npt needs both temp and a pressure keyword', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5')).toMatch(/needs both temp and pressure/);
    expect(await errorOf('fix 1 all npt iso 1.0 1.0 1.0')).toMatch(/needs both temp and pressure/);
  });

  it('nvt rejects pressure keywords and nph rejects temp', async () => {
    expect(await errorOf('fix 1 all nvt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0')).toMatch(/no pressure keywords/);
    expect(await errorOf('fix 1 all nph iso 1.0 1.0 1.0 temp 1.0 1.0 0.5')).toMatch(/no temp keyword/);
  });

  it('Pdamp and Tdamp must be positive', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 0.0')).toMatch(/Pdamp must be > 0/);
    expect(await errorOf('fix 1 all nvt temp 1.0 1.0 0.0')).toMatch(/Tdamp must be > 0/);
  });

  it('couple accepts only none, xyz, xy, yz, xz', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0 couple abc')).toMatch(/couple must be none, xyz, xy, yz or xz/);
  });

  it('coupled dimensions need identical Pstart, Pstop and Pdamp', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 x 1.0 1.0 1.0 y 2.0 2.0 1.0 couple xy')).toMatch(/identical Pstart, Pstop and Pdamp/);
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 x 1.0 1.0 1.0 y 1.0 1.0 2.0 couple xy')).toMatch(/identical Pstart, Pstop and Pdamp/);
  });

  it('tilt keywords need a triclinic box', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 tri 1.0 1.0 1.0')).toMatch(/tilt keywords need a triclinic box/);
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 xy 0.0 0.0 1.0')).toMatch(/tilt keywords need a triclinic box/);
  });

  it('a triclinic box accepts the tilt keywords', async () => {
    const tri = BASE.replace('region box block 0 2 0 2 0 2', 'region box prism 0 2 0 2 0 2 0.1 0.1 0.1');
    const session = new Session({ emit: (_e: EngineEvent) => {} });
    await session.execute(`${tri}\nfix 1 all npt temp 1.0 1.0 0.5 tri 1.0 1.0 1.0\nrun 0\n`);
  });

  it('couple xyz is accepted for iso', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0 couple xyz')).toBeNull();
  });

  it('mtk and the nonzero keyword values are validated', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0 mtk maybe')).toMatch(/mtk must be yes or no/);
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0 pchain -1')).toMatch(/pchain must be >= 0/);
  });

  it('barostatting a non-periodic dimension is refused', async () => {
    const np = BASE.replace('units lj', 'units lj\nboundary p p f');
    const session = new Session({ emit: (_e: EngineEvent) => {} });
    await expect(session.execute(`${np}\nfix 1 all npt temp 1.0 1.0 0.5 z 1.0 1.0 1.0\nrun 0\n`)).rejects.toThrow(/non-periodic dimension/);
  });
});
