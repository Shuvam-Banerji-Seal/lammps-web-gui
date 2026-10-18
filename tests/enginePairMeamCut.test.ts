import { describe, expect, it } from 'vitest';
import { runMeam, meamPe, worstForceError, oracleFiles, relErr, type MeamSetup } from './meamHarness';

/*
 * pair_style meam: the smoothing window [rc - delr, rc], partially screened clusters, the periodic diamond cell,
 * the Rose energy forms (erose_form 0, 1, 2 with attrac and repuls) and the tabulated pair term, all through the
 * engine. The synthetic bcc W entry is tests/oracle/w15meam_bcc_lib.meam + w15meam_bcc.meam (the same numbers as
 * w16meamcut_*); every energy below was measured with native LAMMPS as a black box, on the inputs named in the
 * comments, and is a number, not LAMMPS text.
 */

const BCC = oracleFiles('w15meam_bcc_lib.meam', 'w15meam_bcc.meam');
const BCC_CMD = 'w15meam_bcc_lib.meam W w15meam_bcc.meam W';
const ROSE_LIB = oracleFiles('w16meamcut_bcc_lib.meam');
const DIA = oracleFiles('w15meam_dia_lib.meam', 'w15meam_dia.meam');
const DIA_CMD = 'w15meam_dia_lib.meam W w15meam_dia.meam W';

/** bcc W setup of the given coordinates, box 60 A (no periodic image within rc). */
const bcc = (x: number[], files: Record<string, string> = BCC, coeff = BCC_CMD): MeamSetup => ({
  files,
  coeff,
  types: Array.from({ length: x.length / 3 }, () => 1),
  x,
});

/** Rose-form parameter file: the bcc W entry of w16meamcut_bcc.meam with a given erose_form and attrac/repuls. */
const roseFiles = (form: number): Record<string, string> => ({
  ...ROSE_LIB,
  'par.meam': oracleFiles('w16meamcut_bcc.meam')['w16meamcut_bcc.meam'] +
    `erose_form = ${form}\nattrac(1,1) = 0.4\nrepuls(1,1) = 0.9\n`,
});
const ROSE_CMD = 'w16meamcut_bcc_lib.meam W par.meam W';

describe('pair_style meam: smoothing window [rc - delr, rc]', () => {
  // Measured with native LAMMPS (black box): dimer pe along x for bcc W (dimers at 3.9 ... 3.999 A).
  const native: Array<[number, number]> = [
    [3.9, -4.65069307426565],
    [3.92, -4.5907421718859],
    [3.95, -4.21873921729463],
    [3.96, -3.85736275330339],
    [3.98, -2.32815752424894],
    [3.99, -1.05457803891509],
    [3.999, -0.0278618695728573],
  ];
  for (const [r, pe] of native) {
    it(`dimer at ${r} A matches native pe to 1e-10 relative`, async () => {
      expect(relErr(await meamPe(bcc([0, 0, 0, r, 0, 0])), pe)).toBeLessThan(1e-10);
    });
  }

  it('dimer forces inside the window equal -grad E', async () => {
    expect(await worstForceError(bcc([0, 0, 0, 3.95, 0.1, 0]), 1e-6, 1)).toBeLessThan(1e-6);
  });
});

describe('pair_style meam: partially screened clusters with a neighbour inside the window', () => {
  // Measured with native LAMMPS (black box): the screening atom 2 at 3.918 A lies inside the window; atom 1 at 3.8 A.
  const cases: Array<[number[], number]> = [
    [[0, 0, 0, 3.8, 0, 0, 1.95, 3.4, 0], -9.14206625906409],
    [[0, 0, 0, 3.8, 0, 0, 1.95, 3.4, 0.3], -9.08734716841828],
    [[0, 0, 0, 3.85, 0, 0, 1.9, 3.42, 0.2], -8.86808803339698],
    [[0, 0, 0, 3.8, 0, 0, 1.9, 3.5, 0], -6.74256212707885],
    [[0, 0, 0, 3.6, 0, 0, 1.8, 3.6, 0], -5.52071413702904],
    [[0, 0, 0, 3.7, 0, 0, 1.85, 3.55, 0.4], -5.2126427823761],
    [[0, 0, 0, 3.7, 0, 0, 1.85, 3.55, 0.4, 0.2, 1.3, 1.5], -12.3842905845873],
  ];
  for (const [x, pe] of cases) {
    it(`cluster of ${x.length / 3} atoms matches native pe to 1e-9 relative`, async () => {
      expect(relErr(await meamPe(bcc(x)), pe)).toBeLessThan(1e-9);
    });
  }
});

describe('pair_style meam: periodic distorted diamond cell', () => {
  it('1-cell diamond (8 atoms, L = 5 A, 0.06 A distortion) matches native pe to 1e-9 relative', async () => {
    // Measured with native LAMMPS (black box, periodic p p p): pe -36.6157506731495 eV (press 5118.16596 bar).
    const x = [4.968932738527656, 4.994925902681425, 0.026488956045358947, 2.457889558551833, 2.4457766466215247, 4.981045851120725, 2.553664881978184, 0.006002893438562751, 2.454528980404138, 4.979435355486348, 2.523419070597738, 2.5168148566316813, 1.227563006095588, 1.2210494157019998, 1.3069995287247007, 3.7588786383625123, 3.773848517015576, 1.2511135089863092, 3.7218735930137328, 1.2107393417041745, 3.7510782984271636, 1.2930176328215746, 3.7036004885472362, 3.7115272442903375];
    const E = await meamPe({ files: DIA, coeff: DIA_CMD, types: Array.from({ length: 8 }, () => 1), x, box: 5, offset: 0 });
    expect(relErr(E, -36.6157506731495)).toBeLessThan(1e-9);
  });
});

describe('pair_style meam: Rose energy forms (erose_form 0, 1, 2 with attrac and repuls)', () => {
  // Measured with native LAMMPS (black box): bcc W entry of tests/oracle/w16meamcut_bcc_lib.meam with
  // attrac(1,1) = 0.4 and repuls(1,1) = 0.9 (tests/oracle/w17meam_erose1.meam, w17meam_erose2.meam; form 0 with the same numbers).
  const EROSE: Record<number, Array<[number, number]>> = {
    0: [[2.2, -1.66227435023832], [2.7, -8.61120313931507], [3.2, -7.10997982705863], [3.95, -4.86202032965563]],
    1: [[2.2, -7.87019867137249], [2.7, -8.61190178278414], [3.2, -6.84316133263483], [3.95, -3.81913317082445]],
    2: [[2.2, -2.88958008591648], [2.7, -8.61121185605331], [3.2, -7.14351524619321], [3.95, -5.14723561938785]],
  };
  const FORM_FILES: Record<number, Record<string, string>> = {
    0: roseFiles(0),
    1: { ...ROSE_LIB, 'par.meam': oracleFiles('w17meam_erose1.meam')['w17meam_erose1.meam'] },
    2: { ...ROSE_LIB, 'par.meam': oracleFiles('w17meam_erose2.meam')['w17meam_erose2.meam'] },
  };
  for (const form of [0, 1, 2]) {
    it(`erose_form ${form} dimers match native pe to 1e-10 relative`, async () => {
      for (const [r, pe] of EROSE[form]) {
        expect(relErr(await meamPe(bcc([0, 0, 0, r, 0, 0], FORM_FILES[form], ROSE_CMD)), pe)).toBeLessThan(1e-10);
      }
    });
  }

  it('erose_form 1 and 2 forces equal -grad E (bcc dimer and cluster)', async () => {
    for (const form of [1, 2]) {
      expect(await worstForceError(bcc([0, 0, 0, 2.6, 0.2, 0, 1.4, 2.2, 0.3], FORM_FILES[form], ROSE_CMD), 1e-6, 1)).toBeLessThan(1e-6);
    }
  });

  it('an erose_form other than 1 or 2 uses the form-0 Rose function (native erose(): the else branch)', async () => {
    // Measured with native LAMMPS (black box): erose_form = 3 gives the same dimer energies as erose_form = 0.
    const x = [0, 0, 0, 2.7, 0, 0];
    const f3 = await meamPe(bcc(x, roseFiles(3), ROSE_CMD));
    const f0 = await meamPe(bcc(x, roseFiles(0), ROSE_CMD));
    expect(f3).toBe(f0);
  });
});

describe('pair_style meam: tabulated pair term forces against native', () => {
  it('bcc dimer forces match native to 1e-10 relative, also inside the window', async () => {
    // Measured with native LAMMPS (black box, dump of the force on atom 2): x-force -0.14251957938563331 eV/A at 2.5 A
    // and -27.148276609005627 eV/A at 3.95 A (the analytic pair term differs by 6e-7 relative at 2.5 A).
    const fx = async (r: number) => (await runMeam(bcc([0, 0, 0, r, 0, 0]))).F[1][0];
    expect(relErr(await fx(2.5), -0.14251957938563331)).toBeLessThan(1e-10);
    expect(relErr(await fx(3.95), -27.148276609005627)).toBeLessThan(1e-10);
  });
});
