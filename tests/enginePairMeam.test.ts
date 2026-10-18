import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { runMeam, meamPe, worstForceError, fccCellSetup, jittered, oracleFiles, relErr, type MeamSetup } from './meamHarness';

/*
 * pair_style meam, single element (the Cu-like synthetic entry below), through the engine. Forces are checked
 * against central finite differences of the engine energy; the file handling and the StyleError paths are the
 * ones of the GPL port (src/engine-gpl/meam/pairMeamMs.ts, the parse and coeff routines of src/MEAM/pair_meam.cpp).
 */

const LIB = "# comment\n'Cu' 'fcc' 12 29 63.546 4.95 2 1 2 4 3.615 3.54 1 1 1 1 1 1 0\n";
const LIB_SPLIT = "'Ni' 'fcc' 12 28 58.69 4.6 2 1\n 2 4 3.52 4.45 1 1 1 1 1 1 0\n";
const LIB_NI_ONE = "'Ni' 'fcc' 12 28 58.69 4.6 2 1 2 4 3.52 4.45 1 1 1 1 1 1 0\n";
const PAR = 'rc = 4.0\ndelr = 0.1\nEc(1,1) = 3.54\nre(1,1) = 2.55\nalpha(1,1) = 4.95\nlattce(1,1) = fcc\nCmin(1,1,1) = 2.0\nCmax(1,1,1) = 2.8\nzbl(1,1) = 0\n';
const FILES = { 'lib.meam': LIB, 'par.meam': PAR };
const TWO = [0, 0, 0, 2.5, 0, 0];

/** A single-element setup of the given coordinates. */
const single = (files: Record<string, string>, x: ArrayLike<number>, extra: Partial<MeamSetup> = {}): MeamSetup => ({
  files,
  coeff: 'lib.meam Cu par.meam Cu',
  types: Array.from({ length: x.length / 3 }, () => 1),
  x,
  ...extra,
});

describe('pair_style meam: forces equal -grad E (finite differences of the engine energy)', () => {
  it('perturbed fcc cell (32 atoms): forces equal -grad E to 1e-6 relative', async () => {
    const c = fccCellSetup(3.615, 2, 0.08, 7);
    expect(await worstForceError({ files: FILES, coeff: 'lib.meam Cu par.meam Cu', types: c.types, x: c.x, box: c.box, offset: 0 })).toBeLessThan(1e-6);
  }, 120000);

  it('perturbed fcc cell with a different seed and a larger distortion', async () => {
    const c = fccCellSetup(3.615, 2, 0.15, 99);
    expect(await worstForceError({ files: FILES, coeff: 'lib.meam Cu par.meam Cu', types: c.types, x: c.x, box: c.box, offset: 0 })).toBeLessThan(1e-6);
  }, 120000);

  it('trimer (partial screening by the third atom)', async () => {
    const x = jittered([[0, 0, 0], [2.5, 0, 0], [1.3, 2.2, 0.3]], 0.05, 3);
    expect(await worstForceError(single(FILES, x))).toBeLessThan(1e-6);
  });

  it('4-atom cluster with partial screening (three screeners on one bond)', async () => {
    const x = jittered([[0, 0, 0], [2.6, 0.1, 0], [1.2, 2.4, 0.2], [1.5, 0.9, 2.3]], 0.1, 11);
    expect(await worstForceError(single(FILES, x))).toBeLessThan(1e-6);
  });

  it('4-atom cluster with a 2nd-neighbour bond (r in the screened 2nd shell)', async () => {
    const x = jittered([[0, 0, 0], [3.4, 0.2, 0], [1.6, 2.7, 0.4], [1.8, 1.2, 2.9]], 0.05, 5);
    expect(await worstForceError(single(FILES, x))).toBeLessThan(1e-6);
  });
});

describe('pair_style meam: library and parameter files', () => {
  it('library: fields may span lines and give the same element as the one-line entry', async () => {
    const split = await meamPe(single({ 'lib.meam': LIB_SPLIT, 'par.meam': PAR }, TWO, { coeff: 'lib.meam Ni par.meam Ni' }));
    const one = await meamPe(single({ 'lib.meam': LIB_NI_ONE, 'par.meam': PAR }, TWO, { coeff: 'lib.meam Ni par.meam Ni' }));
    expect(split).toBe(one);
    expect(Number.isFinite(split)).toBe(true);
  });

  it('library: an element that the file does not contain is an error naming it', async () => {
    await expect(runMeam(single(FILES, TWO, { coeff: 'lib.meam Au par.meam Au' }))).rejects.toThrow(
      /Did not find all elements in MEAM library file.*missing: Au/,
    );
  });

  it('parameters: an unknown keyword is an error naming it', async () => {
    const files = { 'lib.meam': LIB, 'par.meam': PAR + 'foo_keyword = 1\n' };
    await expect(runMeam(single(files, TWO))).rejects.toThrow(/Keyword foo_keyword in MEAM parameter file par\.meam not recognized/);
  });

  it('parameters: rho0 must be positive', async () => {
    const files = { 'lib.meam': LIB, 'par.meam': PAR + 'rho0(1) = 0.0\n' };
    await expect(runMeam(single(files, TWO))).rejects.toThrow(/rho0 must be > 0/);
  });

  it('parameters: an element index outside the potential is an error (Ec(2,2) in a one-element file)', async () => {
    const files = { 'lib.meam': LIB, 'par.meam': PAR + 'Ec(2,2) = 1\n' };
    await expect(runMeam(single(files, TWO))).rejects.toThrow(/Ec.*out of range element index/);
  });

  it('parameters: keywords that the native parser reads run through the engine (zbl, gsmooth_factor, theta, erose_form)', async () => {
    const cases = ['zbl(1,1) = 1\n', 'gsmooth_factor = 0.5\n', 'theta(1,1) = 170\n', 'erose_form = 2\nattrac(1,1) = 0.1\nrepuls(1,1) = 0.1\n'];
    for (const extra of cases) {
      const files = { 'lib.meam': LIB, 'par.meam': PAR.replace('zbl(1,1) = 0\n', '') + extra };
      expect(Number.isFinite(await meamPe(single(files, TWO)))).toBe(true);
    }
  });

  it('a NULL parameter file is accepted (native: no user file is read)', async () => {
    expect(Number.isFinite(await meamPe(single(FILES, TWO, { coeff: 'lib.meam Cu NULL Cu' })))).toBe(true);
  });
});

describe('pair_style meam: pair_style and pair_coeff syntax', () => {
  it('pair_style meam takes no arguments', async () => {
    const s = new Session({ emit: () => {}, writeFile: () => {} });
    s.addFile('lib.meam', LIB);
    await expect(
      s.execute('units metal\natom_style atomic\nregion box block -5 5 -5 5 -5 5\ncreate_box 1 box\npair_style meam 1.0\n'),
    ).rejects.toThrow(/Illegal pair_style meam command/);
  });

  it('pair_modify shift yes is a generic pair option that native accepts; the shifted dimer energy matches native', async () => {
    // Measured with native LAMMPS (black box, bcc W entry of tests/oracle/w15meam_bcc*, dimer at 2.6 A, pair_modify shift yes): -8.81128900636432 eV.
    const events: EngineEvent[] = [];
    const s = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
    for (const [n, t] of Object.entries(oracleFiles('w15meam_bcc_lib.meam', 'w15meam_bcc.meam'))) s.addFile(n, t);
    await s.execute(
      'units metal\natom_style atomic\nboundary p p p\nregion box block 0 40 0 40 0 40\ncreate_box 1 box\n' +
        'create_atoms 1 single 20 20 20 units box\ncreate_atoms 1 single 22.6 20 20 units box\nmass * 1.0\n' +
        'pair_style meam\npair_coeff * * w15meam_bcc_lib.meam W w15meam_bcc.meam W\npair_modify shift yes\n' +
        'thermo_style custom step pe\nthermo_modify format float %.15g\nrun 0\n',
    );
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo');
    expect(relErr(Number(rows[rows.length - 1].row.pe), -8.81128900636432)).toBeLessThan(1e-9);
  });

  it('pair_coeff without a parameter file (no file or NULL among the element names) is an error', async () => {
    await expect(runMeam(single(FILES, TWO, { coeff: 'lib.meam Cu Cu Cu' }))).rejects.toThrow(/No MEAM parameter file in pair coefficients/);
  });

  it('pair_coeff mapping an element that the library does not contain is an error', async () => {
    await expect(runMeam(single(FILES, TWO, { coeff: 'lib.meam Ni par.meam Ni' }))).rejects.toThrow(/Ni/);
  });
});
