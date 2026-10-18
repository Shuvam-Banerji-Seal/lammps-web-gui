import { describe, expect, it } from 'vitest';
import { meamPe, oracleFiles, relErr, worstForceError, type MeamSetup } from './meamHarness';

/*
 * pair_style meam, meam35 scope, through the engine: the element density scaling rozero (library column 18) and its
 * parameter-file override rho0(I) (docs.lammps.org/pair_meam.html: "The *rozero* parameter is an element-dependent
 * density scaling that weights the reference background density"; "rho0(I) = relative density for element I
 * (overwrites value read from meamf file)"), and the embedding reference density of a diamond reference with
 * ibar = 1 / 3 (the real SiC potential needs it). Native LAMMPS (black box) values; the periodic SiC parity is in
 * tests/engineOracle*.test.ts (w35meamrho_sic, w35meamrho_si).
 */

const A_LINE = "'A' 'fcc' 12 29 63.546 5.0 2.0 1.5 2.5 4.0 3.6 3.5 1.0 1.0 2.0 1.0 1.0 1.0 0\n";
/** The B entry of tests/oracle/w15meam_alloy_lib.meam with its rozero (column 18) replaced. */
const bLine = (rozero: number) => `'B' 'fcc' 12 28 58.69 4.0 3.0 1.0 2.0 3.0 3.5 4.0 1.0 1.0 1.5 0.5 2.0 ${rozero} 0\n`;
const PAR_B1 = oracleFiles('w15meam_alloy_b1.meam')['w15meam_alloy_b1.meam'];

const alloy = (rozeroB: number, extraPar = ''): MeamSetup => ({
  files: { 'lib.meam': A_LINE + bLine(rozeroB), 'par.meam': PAR_B1 + extraPar },
  coeff: 'lib.meam A B par.meam A B',
  types: [1, 2],
  x: [],
  ntypes: 2,
});
const withX = (s: MeamSetup, types: number[], x: number[]): MeamSetup => ({ ...s, types, x });
const dimer = (rozeroB: number, r: number, extraPar = '') => withX(alloy(rozeroB, extraPar), [1, 2], [0, 0, 0, r, 0, 0]);
const trimer = (rozeroB: number) => withX(alloy(rozeroB), [1, 2, 1], [0, 0, 0, 2.4, 0, 0, 1.2, 2.0, 0.3]);

describe('pair_style meam: element density scaling rozero (meam35)', () => {
  // Measured with native LAMMPS (black box): A-B dimer at r A with rozero(B) = rz, pe in eV.
  const DIMER: Array<[number, number, number]> = [
    [1.0, 2.0, -2.46023965769572], [1.0, 2.2, -3.09974618373519], [1.0, 2.4, -3.25526687222797],
    [1.0, 3.0, -2.64815172340702], [1.0, 3.5, -1.98901594647507],
    [2.0, 2.0, -2.72321261663502], [2.0, 2.4, -3.36918255365997], [2.0, 3.0, -2.68830698326347], [2.0, 3.5, -2.03376827670451],
    [3.5, 2.0, -3.07665733636415], [3.5, 2.4, -3.5366621943715], [3.5, 3.0, -2.73883384026872], [3.5, 3.5, -2.07643864716689],
    [2.25, 2.0, -2.79204347960128], [2.25, 2.4, -3.40492086964905], [2.25, 3.0, -2.70400952839182], [2.25, 3.5, -2.048145083756],
  ];
  for (const [rz, r, pe] of DIMER) {
    it(`A-B dimer at ${r} A with rozero(B) = ${rz} matches native pe to 1e-9 relative`, async () => {
      expect(relErr(await meamPe(dimer(rz, r)), pe)).toBeLessThan(1e-9);
    });
  }

  // Measured with native LAMMPS (black box): A-B-A trimer at (0,0,0), (2.4,0,0), (1.2,2.0,0.3), pe in eV.
  const TRIMER: Array<[number, number]> = [
    [1.0, -6.23612331013911],
    [2.0, -6.48046168353576],
    [3.5, -6.76079986877474],
    [2.25, -6.54402789282861],
  ];
  for (const [rz, pe] of TRIMER) {
    it(`A-B-A trimer with rozero(B) = ${rz} matches native pe to 1e-9 relative`, async () => {
      expect(relErr(await meamPe(trimer(rz)), pe)).toBeLessThan(1e-9);
    });
  }

  it('rho0(2) = 2.25 in the parameter file on a library rozero of 1 equals the rozero = 2.25 entry', async () => {
    // Measured with native LAMMPS (black box): the same dimer at 2.4 A gives -3.40492086964905 eV both ways.
    const override = await meamPe(dimer(1.0, 2.4, 'rho0(2) = 2.25\n'));
    expect(relErr(override, -3.40492086964905)).toBeLessThan(1e-9);
    expect(override).toBe(await meamPe(dimer(2.25, 2.4)));
  });

  it('a quoted keyword value is read (lattce(1,2) = \'b1\')', async () => {
    const quoted = withX({ ...alloy(1.0), files: { 'lib.meam': A_LINE + bLine(1.0), 'par.meam': PAR_B1.replace("lattce(1,2) = b1", "lattce(1,2) = 'b1'") } }, [1, 2], [0, 0, 0, 2.4, 0, 0]);
    expect(await meamPe(quoted)).toBe(await meamPe(dimer(1.0, 2.4)));
  });

  it('a single element is invariant under rozero (the scaling cancels in a one-element model)', async () => {
    // tests/oracle/w35meamrho_single_lib.meam (rozero 1) and w35meamrho_single_lib2.meam (rozero 2.5)
    const lib = oracleFiles('w35meamrho_single_lib.meam', 'w35meamrho_single_lib2.meam', 'w35meamrho_single.meam');
    const at = (libName: string) => ({
      files: { 'lib.meam': lib[libName], 'par.meam': lib['w35meamrho_single.meam'] },
      coeff: 'lib.meam S par.meam S',
      types: [1, 1],
      x: [0, 0, 0, 2.5, 0, 0],
    });
    const a = await meamPe(at('w35meamrho_single_lib.meam'));
    const b = await meamPe(at('w35meamrho_single_lib2.meam'));
    expect(relErr(a, b)).toBeLessThan(1e-12);
  });

  it('forces equal -grad E to 1e-6 relative for a mixed cluster with rozero(B) = 2.25', async () => {
    const s = withX(alloy(2.25), [1, 2, 1, 2], [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2]);
    expect(await worstForceError(s, 1e-5, 1e-2)).toBeLessThan(1e-6);
  });
});

describe('pair_style meam: diamond reference with ibar = 1 / 3 (meam35)', () => {
  /** Single-element S setup: library entry of the given lattice, ibar and third moment t3; parameters Ec 4, re 2, alpha 5. */
  const sElement = (lat: 'dia' | 'fcc', ibar: number, t3 = 0.3): MeamSetup => ({
    files: {
      'lib.meam': `'S' '${lat}' ${lat === 'dia' ? 4 : 12} 14 20.0 5.0 3.0 2.0 2.0 2.0 4.6188 4.0 1.0 1.0 1.0 0.5 ${t3} 1.0 ${ibar}\n`,
      'par.meam': 'rc = 4.0\ndelr = 0.1\nEc(1,1) = 4.0\nre(1,1) = 2.0\nalpha(1,1) = 5.0\nzbl(1,1) = 0\naugt1 = 1\n',
    },
    coeff: 'lib.meam S par.meam S',
    types: [1, 1],
    x: [],
  });
  const dimerS = (s: MeamSetup, r: number) => withX(s, [1, 1], [0, 0, 0, r, 0, 0]);

  // Measured with native LAMMPS (black box): S-S diamond dimer, pe in eV.
  const DIA: Array<[number, number, number]> = [
    [0, 2.0, -4.99465376351151], [0, 2.4, -3.62126140055328], [0, 3.0, -2.01016261772884],
    [1, 2.0, -4.63946910167441], [1, 2.4, -3.69698941716855], [1, 3.0, -2.67332894409027],
  ];
  for (const [ibar, r, pe] of DIA) {
    it(`diamond dimer ibar = ${ibar} at ${r} A matches native pe to 1e-12 relative`, async () => {
      expect(relErr(await meamPe(dimerS(sElement('dia', ibar), r)), pe)).toBeLessThan(1e-12);
    });
  }

  // Measured with native LAMMPS (black box): large third moment (t3 = 3) makes Gamma_ref large.
  const DIA_BIG: Array<[number, number, number]> = [
    [0, 2.0, -5.19660944713074], [0, 2.4, -4.01906138084711], [0, 3.0, -2.41116852436907],
    [3, 2.0, -4.94269943182461], [3, 2.4, -3.47450974876774], [3, 3.0, -1.52596762559122],
  ];
  for (const [ibar, r, pe] of DIA_BIG) {
    it(`diamond dimer with t3 = 3, ibar = ${ibar} at ${r} A matches native pe to 1e-12 relative`, async () => {
      expect(relErr(await meamPe(dimerS(sElement('dia', ibar, 3.0), r)), pe)).toBeLessThan(1e-12);
    });
  }

  it('an fcc reference has Gamma_ref = 0, so ibar 1 gives the native fcc dimer energies', async () => {
    // Measured with native LAMMPS (black box): fcc dimer ibar = 1, pe in eV.
    const NAT: Array<[number, number]> = [[2.0, -3.18342255666494], [2.4, -2.5803355638014], [3.0, -2.38558355907171]];
    for (const [r, pe] of NAT) expect(relErr(await meamPe(dimerS(sElement('fcc', 1), r)), pe)).toBeLessThan(1e-12);
  });
});
