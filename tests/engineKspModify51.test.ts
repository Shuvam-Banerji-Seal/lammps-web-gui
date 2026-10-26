import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/**
 * kspace_modify keywords that the engine refuses (rule 7: an unsupported keyword is an EngineError naming it,
 * never a silent no-op), and the run_style refusals. The accepted keywords are checked against native LAMMPS
 * in tests/oracle/w51km_*.in.
 */

/** Runs input text in a fresh session and returns the error message (null when it runs). */
const run = async (text: string): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  try {
    await session.execute(text);
    return null;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
};

/** Two charged atoms in a periodic box; the kspace style and kspace_modify line are given by the caller. */
const system = (kspace: string, modify: string, extra = '') => `
units lj
atom_style charge
region box block -6 6 -6 6 -6 6
create_box 1 box
create_atoms 1 single 0 0 0
create_atoms 1 single 1.5 0.5 -0.5
set atom 1 charge 1.0
set atom 2 charge -1.0
mass 1 1.0
pair_style lj/cut/coul/long 2.5
pair_coeff 1 1 0.0 1.0
kspace_style ${kspace}
kspace_modify ${modify}
${extra}
thermo_style custom step pe
run 0
`;

describe('kspace_modify keywords refused by the engine', () => {
  it('refuses tol, which is not a kspace_modify keyword (native rejects it for ewald and pppm)', async () => {
    for (const style of ['ewald 1e-4', 'pppm 1e-4']) {
      const err = await run(system(style, 'gewald 0.4 tol 1e-3'));
      expect(err).toMatch(/kspace_modify tol/);
    }
  });

  it('pressure/scalar yes with ewald: press is unchanged, a tensor component is refused (measured)', async () => {
    // Measured with native LAMMPS: thermo press is the same as with pressure/scalar no, and pxx stops the run with
    // Must use 'kspace_modify pressure/scalar no' for tensor components with kspace_style msm
    expect(await run(system('ewald 1e-4', 'gewald 0.4 kmax/ewald 5 5 5 pressure/scalar yes'))).toBeNull();
    const err = await run(system('ewald 1e-4', 'gewald 0.4 kmax/ewald 5 5 5 pressure/scalar yes').replace('run 0', 'thermo_style custom step press pxx\nrun 0'));
    expect(err).toMatch(/Must use 'kspace_modify pressure\/scalar no' for tensor components with kspace_style msm/);
  });

  it('accepts pressure/scalar no and the no-effect keywords without error', async () => {
    const err = await run(system('ewald 1e-4', 'gewald 0.4 kmax/ewald 5 5 5 pressure/scalar no collective yes nonblocking no fftbench no overlap yes cutoff/adjust no minorder 2'));
    expect(err).toBeNull();
  });

  it('rejects a non-boolean value of the yes/no keywords as utils::logical does', async () => {
    for (const kw of ['collective maybe', 'nonblocking maybe', 'overlap maybe', 'fftbench maybe', 'cutoff/adjust maybe', 'pressure/scalar maybe']) {
      const err = await run(system('ewald 1e-4', `gewald 0.4 kmax/ewald 5 5 5 ${kw}`));
      expect(err, kw).toContain("Expected boolean parameter instead of 'maybe' in input script or data file");
    }
  });

  it('rejects minorder below 2 and a non-integer minorder', async () => {
    expect(await run(system('ewald 1e-4', 'gewald 0.4 kmax/ewald 5 5 5 minorder 1'))).toMatch(/minorder must be at least 2/);
    expect(await run(system('ewald 1e-4', 'gewald 0.4 kmax/ewald 5 5 5 minorder abc'))).toMatch(/Expected integer parameter instead of 'abc' in input script or data file/);
  });

  it('refuses pppm minorder above the order at init (native: PPPM order < minimum allowed order)', async () => {
    const err = await run(system('pppm 1e-4', 'gewald 0.4 mesh 8 8 8 order 5 minorder 6'));
    expect(err).toMatch(/PPPM order < minimum allowed order/);
    const ok = await run(system('pppm 1e-4', 'gewald 0.4 mesh 8 8 8 order 7 minorder 7'));
    expect(ok).toBeNull();
  });
});

describe('run_style refusals name the keyword and say why', () => {
  it('verlet/split on one partition: native Verlet/split requires 2 partitions', async () => {
    const err = await run('units lj\natom_style atomic\nregion box block 0 5 0 5 0 5\ncreate_box 1 box\nrun_style verlet/split\n');
    expect(err).toMatch(/Verlet\/split requires 2 partitions/);
  });

  it('verlet/split/rk without a kspace style: native A KSpace style must be defined with verlet/split/rk', async () => {
    // measured with native LAMMPS: run_style verlet/split/rk itself is accepted on one partition; the run stops
    const err = await run('units lj\natom_style atomic\nregion box block 0 5 0 5 0 5\ncreate_box 1 box\nmass 1 1\nrun_style verlet/split/rk\nrun 0\n');
    expect(err).toMatch(/A KSpace style must be defined with verlet\/split\/rk/);
  });

  it('names a run_style that does not exist as native LAMMPS does', async () => {
    const err = await run('units lj\natom_style atomic\nregion box block 0 5 0 5 0 5\ncreate_box 1 box\nrun_style foo\n');
    expect(err).toMatch(/Unknown integrate style foo/);
  });
});
