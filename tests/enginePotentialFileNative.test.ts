import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * Pair styles given a potential file the notebook does not have, read as native LAMMPS reads them: the words checked
 * before the file, then the file opened with native's message for its reader (PotentialFileReader's "cannot open
 * <name> potential file <file>: <reason>", src/potential_file_reader.cpp at b332f28, and the styles' own readers).
 * Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28): each row is the unit style, the atom style, the
 * pair_style and pair_coeff lines after a one-atom box, and the first line of native's error message (null when native
 * accepts the input). Not compared here: the styles of packages the native binary lacks.
 */

const BOX = 'region box block 0 4 0 4 0 4\ncreate_box 1 box\ncreate_atoms 1 single 1 1 1\nmass 1 1.0\n';

const firstError = async (input: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(input); } catch (e) { return (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return null;
};

const ROWS: [string, string, string, string, string | null][] = [
  ["metal", "atomic", "pair_style tersoff", "pair_coeff * * nofile.txt Si", "cannot open tersoff potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style tersoff/mod", "pair_coeff * * nofile.txt Si", "cannot open tersoff/mod potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style tersoff/mod/c", "pair_coeff * * nofile.txt Si", "cannot open tersoff/mod/c potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style tersoff/zbl", "pair_coeff * * nofile.txt Si", "cannot open tersoff/zbl potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style tersoff/table", "pair_coeff * * nofile.txt Si", "cannot open tersoff/table potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style sw", "pair_coeff * * nofile.txt Si", "cannot open sw potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style sw/mod", "pair_coeff * * nofile.txt Si", "cannot open sw potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style gw", "pair_coeff * * nofile.txt Si", "cannot open gw potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style gw/zbl", "pair_coeff * * nofile.txt Si", "cannot open gw/zbl potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style bop", "pair_coeff * * nofile.txt Si", "cannot open BOP potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style edip", "pair_coeff * * nofile.txt Si", "cannot open edip potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style edip/multi", "pair_coeff * * nofile.txt Si", "cannot open edip potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style extep", "pair_coeff * * nofile.txt Si", "cannot open ExTeP potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style polymorphic", "pair_coeff * * nofile.txt Si", "cannot open polymorphic potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style vashishta", "pair_coeff * * nofile.txt Si", "cannot open vashishta potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style meam/spline", "pair_coeff * * nofile.txt Si", "cannot open meam/spline potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style meam/sw/spline", "pair_coeff * * nofile.txt Si", "cannot open meam/sw/spline potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style adp", "pair_coeff * * nofile.txt Si", "cannot open adp potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style eam/alloy", "pair_coeff * * nofile.txt Si", "cannot open eam/alloy potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style eam/fs", "pair_coeff * * nofile.txt Si", "cannot open eam/fs potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style eam/he", "pair_coeff * * nofile.txt Si", "cannot open eam/he potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style eam/cd", "pair_coeff * * nofile.txt Si", "cannot open eam/alloy potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style eam/cd/old", "pair_coeff * * nofile.txt Si", "cannot open eam/alloy potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style nb3b/harmonic", "pair_coeff * * nofile.txt Si", "cannot open nb3n/harmonic potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style nb3b/screened", "pair_coeff * * nofile.txt Si", "cannot open nb3b/screened potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style threebody/table", "pair_coeff * * nofile.txt Si", "cannot open threebody potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style sw/angle/table", "pair_coeff * * nofile.txt Si", "cannot open sw potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style rebo", "pair_coeff * * nofile.txt Si", "Element Si not supported by pair style rebo"],
  ["metal", "atomic", "pair_style comb", "pair_coeff * * nofile.txt Si", "cannot open comb potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style ldd", "pair_coeff * * nofile.txt Si", "Cannot open ldd potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style eam", "pair_coeff 1 1 nofile.txt", "cannot open eam potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style lcbop", "pair_coeff * * nofile.txt C", "cannot open LCBOP potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style eim", "pair_coeff * * Si nofile.txt Si", "cannot open eim potential file nofile.txt"],
  ["metal", "atomic", "pair_style rebomos", "pair_coeff * * nofile.txt Mo", "cannot open rebomos potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style local/density", "pair_coeff * * nofile.txt", "cannot open local/density potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style pod", "pair_coeff * * nofile.txt nofile2.txt Si", "Cannot open POD coefficient file nofile.txt: "],
  ["metal", "atomic", "pair_style mesocnt", "pair_coeff * * nofile.txt 1 1 1", "Illegal pair_coeff command: missing argument(s)"],
  ["metal", "atomic", "pair_style airebo 3.0", "pair_coeff * * nofile.txt C", "cannot open airebo potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style airebo/morse 3.0", "pair_coeff * * nofile.txt C", "cannot open airebo/morse potential file nofile.txt: No such file or directory"],
  ["metal", "charge", "pair_style comb3 polar_off", "pair_coeff * * nofile.txt Si", "cannot open COMB3 potential file nofile.txt: No such file or directory"],
  ["metal", "charge", "pair_style comb", "pair_coeff * * nofile.txt Si", "cannot open comb potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style meam", "pair_coeff * * nofile.txt Si NULL Si", "Cannot open MEAM library file "],
  ["metal", "atomic", "pair_style meam", "pair_coeff * * nofile.txt Si nofile2.txt Si", "Cannot open MEAM library file "],
  ["metal", "atomic", "pair_style meam/ms", "pair_coeff * * nofile.txt Si NULL Si", "Cannot open MS-MEAM library file "],
  ["metal", "atomic", "pair_style snap", "pair_coeff * * nofile.txt nofile2.txt Si", "Cannot open SNAP coefficient file nofile.txt: "],
  ["metal", "atomic", "pair_style snap", "pair_coeff * * nofile.txt Si", "Incorrect args for pair coefficients"],
  ["metal", "charge", "pair_style coul/streitz 10.0 wolf 0.3", "pair_coeff * * nofile.txt Si", "cannot open coul/streitz potential file nofile.txt: No such file or directory"],
  ["metal", "charge", "pair_style coul/ctip 0.3 10.0", "pair_coeff * * nofile.txt Si", "cannot open coul/ctip potential file nofile.txt: No such file or directory"],
  ["metal", "charge", "pair_style reaxff NULL", "pair_coeff * * nofile.txt Si", "The ReaxFF parameter file nofile.txt cannot be opened: No such file or directory"],
  ["metal", "atomic", "pair_style table linear 100", "pair_coeff 1 1 nofile.txt KEY", "cannot open pair table potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style table/rx linear 100", "pair_coeff 1 1 nofile.txt KEY", "Incorrect args for pair coefficients"],
  ["metal", "atomic", "pair_style hybrid/overlay kolmogorov/crespi/z 14.0", "pair_coeff * * kolmogorov/crespi/z nofile.txt C", "cannot open kolmogorov/crespi/z potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style hybrid/overlay lebedeva/z 14.0", "pair_coeff * * lebedeva/z nofile.txt C", "cannot open lebedeva/z potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style hybrid/overlay kolmogorov/crespi/full 14.0 0", "pair_coeff * * kolmogorov/crespi/full nofile.txt C", "cannot open kolmogorov/crespi/full potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style hybrid/overlay ilp/graphene/hbn 14.0 0", "pair_coeff * * ilp/graphene/hbn nofile.txt C", "cannot open ilp/graphene/hbn potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style hybrid/overlay drip", "pair_coeff * * drip nofile.txt C", "cannot open drip potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style mliap model linear nofile.txt descriptor sna nofile2.txt", "", "Cannot open MLIAPModel coeff file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style mliap model linear nofile.txt descriptor so3 nofile2.txt", "", "Cannot open MLIAPModel coeff file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style uf3 2", "pair_coeff * * nofile.txt", "Invalid number of arguments uf3 in pair coeffs."],
  ["metal", "atomic", "pair_style hybrid/overlay ilp/tmd 14.0 0", "pair_coeff * * ilp/tmd nofile.txt Mo", "cannot open ilp/tmd potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style hybrid/overlay saip/metal 14.0 0", "pair_coeff * * saip/metal nofile.txt Au", "cannot open saip/metal potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style hybrid/overlay aip/water/2dm 14.0 0", "pair_coeff * * aip/water/2dm nofile.txt C", "cannot open aip/water/2dm potential file nofile.txt: No such file or directory"],
  ["metal", "charge", "pair_style comb3 polar_off", "pair_coeff * * nofile.txt C", "cannot open comb3 potential file lib.comb3: No such file or directory"],
  ["metal", "amoeba", "pair_style amoeba", "pair_coeff * * nofile.prm nofile.key", "Cannot open AMOEBA PRM file nofile.prm: No such file or directory"],
  ["metal", "amoeba", "pair_style hippo", "pair_coeff * * nofile.prm nofile.key", "Cannot open HIPPO PRM file nofile.prm: No such file or directory"],
  ["metal", "atomic", "pair_style table/rx linear 100", "pair_coeff 1 1 nofile.txt KEY fractional", "Incorrect args for pair coefficients"],
  ["metal", "atomic", "pair_style meam/spline", "pair_coeff * * nofile.txt Si", "cannot open meam/spline potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style meam", "pair_coeff * * nofile.txt Si NULL", "Illegal pair style meam coefficients command: missing argument(s)"],
  ["metal", "atomic", "pair_style hybrid/overlay lj/cut 2.5 sw", "pair_coeff * * sw nofile.txt Si", "cannot open sw potential file nofile.txt: No such file or directory"],
  ["metal", "dpd", "pair_style multi/lucy linear 100", "pair_coeff 1 1 nofile.txt KEY", "cannot open multi/lucy table potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style lj/smooth/linear 2.5", "pair_coeff 1 1 1.0 1.0", null],
  ["metal", "atomic", "pair_style bop", "pair_coeff * * nofile.txt Si", "cannot open BOP potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style tracker 2.5 fix_ID FIX type/name 1", "pair_coeff * * nofile.txt", "Expected integer parameter instead of 'fix_ID' in input script or data file"],
  ["metal", "atomic", "pair_style meam/spline", "pair_coeff * * nofile.txt Si", "cannot open meam/spline potential file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style dpd/fdt/energy 2.5 1 34387", "pair_coeff 1 1 1 1", "Illegal pair_style command"],
  ["metal", "atomic", "pair_style e3b 1", "pair_coeff * * preset 2015", null],
  ["metal", "atomic", "pair_style tracker fixID 1000 id1 bogus", "", "Illegal pair_style command"],
  ["metal", "atomic", "pair_style tracker fixID 1000 id1 time/min", "", "Invalid keyword in pair tracker command"],
  ["metal", "atomic", "pair_style tracker fixID 1000", "", "Must request at least one value to output"],
  ["metal", "atomic", "pair_style list nofile.txt 2.5", "", "Error opening pair list coeffs file nofile.txt: No such file or directory"],
  ["metal", "atomic", "pair_style lepton 2.5", "pair_coeff 1 1 \"r^2\"", null],
  ["metal", "dpd", "pair_style exp6/rx 2.5", "pair_coeff 1 1 nofile.txt a a 1.0", "Fix rx not defined or not compatible with pair style"],
];

describe('pair styles with a missing potential file as native LAMMPS', () => {
  it.each(ROWS)('%s %s: %s / %s', async (units, atom, style, coeff, error) => {
    expect(await firstError(`units ${units}\natom_style ${atom}\n${BOX}${style}\n${coeff}\n`)).toBe(error);
  });
});

/*
 * oxDNA styles read their files as an "oxdna potential" (PotentialFileReader, src/CG-DNA at b332f28), so native's
 * message says "potential" twice. Measured with native LAMMPS on the oxDNA duplex1 example (lj units).
 */
const OXDNA_DATA = readFileSync(join(__dirname, 'oracle', 'w57oxdna_duplex1.data'), 'utf8');
const OXDNA = "units lj\n\ndimension 3\n\nnewton on\n\nboundary  p p p\n\natom_style hybrid bond ellipsoid oxdna\natom_modify sort 0 1.0\n\n# Pair interactions require lists of neighbours to be calculated\nneighbor 2.0 bin\nneigh_modify every 1 delay 0 check yes\n\nread_data duplex1.data\n\nset atom * mass 3.1575\n\ngroup all type 1 4\n\n# oxDNA bond interactions - FENE backbone\nbond_style oxdna/fene\nbond_coeff * 2.0 0.25 0.7525\nspecial_bonds lj 0 1 1\n";

const oxdnaError = async (input: string): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  s.addFile('duplex1.data', OXDNA_DATA);
  try { await s.execute(input); } catch (e) { return (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return null;
};

const OXDNA_PAIRS: [string, string][] = [
  ["pair_coeff * * oxdna/excv    nofile.txt", "cannot open oxdna potential potential file nofile.txt: No such file or directory"],
  ["pair_coeff * * oxdna/stk     seqav 0.1 nofile.txt", "cannot open oxdna potential potential file nofile.txt: No such file or directory"],
  ["pair_coeff * * oxdna/hbond   seqav nofile.txt", "cannot open oxdna potential potential file nofile.txt: No such file or directory"],
  ["pair_coeff * * oxdna/xstk    nofile.txt", "cannot open oxdna potential potential file nofile.txt: No such file or directory"],
  ["pair_coeff * * oxdna/coaxstk nofile.txt", "cannot open oxdna potential potential file nofile.txt: No such file or directory"],
];

describe('oxDNA styles with a missing potential file as native LAMMPS', () => {
  it.each(OXDNA_PAIRS)('%s', async (coeff, error) => {
    expect(await oxdnaError(OXDNA + "pair_style hybrid/overlay oxdna/excv oxdna/stk oxdna/hbond oxdna/xstk oxdna/coaxstk\n" + coeff + '\n')).toBe(error);
  });
  it('bond_style oxdna/fene', async () => {
    expect(await oxdnaError(OXDNA.replace('bond_coeff * 2.0 0.25 0.7525', "bond_coeff * nofile.txt"))).toBe("cannot open oxdna potential potential file nofile.txt: No such file or directory");
  });
});
