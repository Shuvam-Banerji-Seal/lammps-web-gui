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
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5')).toMatch(/Pressure control must be used with fix npt/);
    expect(await errorOf('fix 1 all npt iso 1.0 1.0 1.0')).toMatch(/Temperature control must be used with fix npt/);
  });

  it('nvt rejects pressure keywords and nph rejects temp', async () => {
    expect(await errorOf('fix 1 all nvt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0')).toMatch(/Pressure control can not be used with fix nvt/);
    expect(await errorOf('fix 1 all nph iso 1.0 1.0 1.0 temp 1.0 1.0 0.5')).toMatch(/Temperature control can not be used with fix nph/);
  });

  it('Pdamp and Tdamp must be positive (native texts)', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 0.0')).toMatch(/Fix npt damping parameters must be > 0.0/);
    expect(await errorOf('fix 1 all nvt temp 1.0 1.0 0.0')).toMatch(/Temperature damping for fix nvt cannot be <= 0.0/);
  });

  it('couple accepts only none, xyz, xy, yz, xz', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0 couple abc')).toMatch(/Illegal fix npt couple option: abc/);
  });

  it('coupled dimensions need identical Pstart, Pstop and Pdamp', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 x 1.0 1.0 1.0 y 2.0 2.0 1.0 couple xy')).toMatch(/Invalid fix npt pressure settings/);
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 x 1.0 1.0 1.0 y 1.0 1.0 2.0 couple xy')).toMatch(/Invalid fix npt pressure settings/);
  });

  it('tilt keywords need a triclinic box', async () => {
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 tri 1.0 1.0 1.0')).toMatch(/Can not specify Pxy\/Pxz\/Pyz in fix npt with non-triclinic box/);
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 xy 0.0 0.0 1.0')).toMatch(/Can not specify Pxy\/Pxz\/Pyz in fix npt with non-triclinic box/);
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
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0 mtk maybe')).toMatch(/Expected boolean parameter instead of 'maybe' in input script or data file/);
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 1.0 pchain -1')).toMatch(/Invalid fix npt pchain argument: -1/);
    expect(await errorOf('fix 1 all npt temp 1.0 1.0 0.5 x 1.0 1.0 1.0 nreset -2')).toMatch(/Invalid fix npt nreset argument: -2/);
  });

  it('has the PE_strain vector entry only with a deviatoric keyword (x, y, z, xy, yz, xz), as native compute_vector', async () => {
    const size = async (fix: string): Promise<number> => {
      const session = new Session({ emit: (_e: EngineEvent) => {} });
      await session.execute(`${BASE}\n${fix}\nrun 0\n`);
      return session.sys.fix('1').sizeVector;
    };
    // nph pchain 3: omega, omega_dot (3), etap, etap_dot (3), PE_omega, KE_omega_dot (3), PE_etap, KE_etap_dot (3)
    expect(await size('fix 1 all nph aniso 1.0 1.0 1.0 ptemp 1.0')).toBe(24);
    expect(await size('fix 1 all nph x 1.0 1.0 1.0 y 1.0 1.0 1.0 z 1.0 1.0 1.0 ptemp 1.0')).toBe(25);
    expect(await size('fix 1 all nph iso 1.0 1.0 1.0 ptemp 1.0')).toBe(16);
  });

  it('barostatting a non-periodic dimension is refused', async () => {
    const np = BASE.replace('units lj', 'units lj\nboundary p p f');
    const session = new Session({ emit: (_e: EngineEvent) => {} });
    await expect(session.execute(`${np}\nfix 1 all npt temp 1.0 1.0 0.5 z 1.0 1.0 1.0\nrun 0\n`)).rejects.toThrow('Cannot use fix npt on a non-periodic z dimension');
  });
});
