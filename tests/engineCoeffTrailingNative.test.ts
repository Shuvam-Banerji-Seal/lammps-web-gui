import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * An extra word after a pair_coeff, bond_coeff, angle_coeff, dihedral_coeff or improper_coeff line's coefficients,
 * refused with native LAMMPS's message: each row takes an oracle case up to one of its coefficient lines, adds the word
 * "bogus" to that line and runs 0 steps. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28; PACKAGE_ROWS
 * with b332f28 builds that have the package); the messages come from each style's coeff() in the C++ source at b332f28.
 */

const ORACLE = join(__dirname, 'oracle');
const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const EXAMPLES = join(__dirname, '..', 'third_party', 'lammps', 'examples');
const ORACLE_FILES = new Set(readdirSync(ORACLE).filter((f) => !f.endsWith('.in')));
const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^# ${key}:(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter((w) => w !== '') : [];
};
const readInput = (path: string): string => readFileSync(path, path.endsWith('.gz') ? 'latin1' : 'utf8');

/** The case's input up to line k, which gets the word "bogus", then run 0, with the files it reads. */
const withTrailingWord = (name: string, k: number): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const body = lines.slice(0, k).filter((l) => !l.startsWith('# oracle')).join('\n') + '\n' + lines[k].split('#')[0].trimEnd() + ' bogus\nrun 0\n';
  const files: Record<string, string> = {};
  for (const f of directive(text, 'oracle-inputs')) files[f] = readInput(join(ORACLE, f));
  for (const f of directive(text, 'oracle-potentials')) files[f] = readFileSync(join(POTENTIALS, f), 'utf8');
  for (const f of directive(text, 'oracle-examples')) files[f.split('/').pop()!] = readInput(join(EXAMPLES, f));
  // files of the oracle directory that the lines name without a directive
  for (const w of body.split(/\s+/)) if (ORACLE_FILES.has(w) && !(w in files)) files[w] = readInput(join(ORACLE, w));
  return [body, files];
};

const outcome = async (input: string, files: Record<string, string>): Promise<string> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  for (const [name, text] of Object.entries(files)) s.addFile(name, text);
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(input); } catch (e) { return 'ERROR: ' + (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return 'OK';
};

/** [oracle case, index of the coefficient line that gets " bogus" (0-based), native result] */
const ROWS: [string, number, string][] = [
  ["w108xf_elfit_sisi", 11, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w10nb3b_harmonic", 22, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w115p_rebomos", 11, "ERROR: Incorrect args for pair coefficients"],
  ["w115p_ylz_vesicle", 21, "ERROR: Incorrect args for pair coefficients"],
  ["w119amoeba_hippo_hexamer", 16, "ERROR: Incorrect args for pair coefficients"],
  ["w119amoeba_mpole_rspace", 15, "ERROR: Incorrect args for pair coefficients"],
  ["w128pod_ta_static", 18, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w142mx_multi_lucy_lookup", 26, "ERROR: Illegal pair_coeff command"],
  ["w145pr_pair_extep", 13, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w147ab_airebo_bc", 10, "ERROR: Incorrect number of args for pair coefficient."],
  ["w157mt_bond_angle", 12, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w157mt_viscous", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w158cb_cu", 15, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w15gran_mdr_pair", 18, "ERROR: Illegal pair_coeff command bogus"],
  ["w16lepton_bonded", 14, "ERROR: Incorrect number of args for angle coefficients"],
  ["w16lepton_bonded", 16, "ERROR: Incorrect number of args for dihedral coefficients"],
  ["w16lepton_vref", 10, "ERROR: Incorrect number of args for bond coefficients"],
  ["w17gaussian", 12, "ERROR: Illegal bond_coeff command: missing argument(s)"],
  ["w17gaussian", 14, "ERROR: Illegal angle_coeff command: missing argument(s)"],
  ["w1dih_zero", 16, "ERROR: Incorrect args for dihedral coefficients"],
  ["w1imp_zero", 17, "ERROR: Incorrect args for improper coefficients"],
  ["w204sg_fs_vcsgc", 12, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w209ms_charmm_water", 9, "ERROR: Illegal pair_coeff command"],
  ["w22pot_sw_mod_mos", 21, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w22pot_tersoff_mod_si", 18, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w22pot_tersoff_modc_si", 18, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w234lc_diamond", 13, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w2tsw_tersoff", 19, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w38nb3b_screened_random", 14, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w43st_dsf_alone", 19, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w450dyngroup_tdpd", 18, "ERROR: Incorrect args for pair tdpd coefficients"],
  ["w472vashtable_runs", 13, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w47ms_si_min", 15, "ERROR: Incorrect args for pair coefficients"],
  ["w47ms_sw_ti_run", 19, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w47pkg_dtreset_metal", 12, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w47spin_moving2", 14, "ERROR: Incorrect args for pair coefficients"],
  ["w48sp2_biquad", 21, "ERROR: Incorrect args for pair coefficients"],
  ["w48sp2_dmi", 21, "ERROR: Incorrect args in pair_style command"],
  ["w48sp2_magelec", 21, "ERROR: Incorrect args in pair_style command"],
  ["w49mt_swang_spline", 16, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w49mt_tri_edge", 14, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w49pb_coul_diel_dimer", 14, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w49pb_momb_md", 23, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w51bpm_spring_compress", 10, "ERROR: Incorrect args for bond coefficients"],
  ["w51cs_sdk_angle", 14, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w51cs_spica_angle", 14, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w52airebo_airebo_00", 10, "ERROR: Incorrect number of args for pair coefficient."],
  ["w52airebo_morse_11", 10, "ERROR: Incorrect number of args for pair coefficient."],
  ["w52airebo_rebo_ethane", 9, "ERROR: Incorrect number of args for pair coefficient."],
  ["w53bpm_rot_chain", 13, "ERROR: Incorrect args for bond coefficients: expected Kr Ks Kt Kb Fcr Fcs Tct Tcb gr gs gt gb"],
  ["w58comb_si0", 13, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w59bpm_peri_pmb", 27, "ERROR: Incorrect args for bond_style bpm/peri pmb model (expected: <type> pmb c horizon s00 alpha)"],
  ["w5vash_si", 19, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w69fep_charmmsoft", 18, "ERROR: Illegal pair_coeff command"],
  ["w76dih_fine_fourier", 12, "ERROR: Incorrect number of arguments for dihedral coefficients"],
  ["w80superell_ellipsoid_gran", 25, "ERROR: Illegal pair_coeff command bogus"],
  ["w83sph_heat3d", 20, "ERROR: Incorrect number of args for pair_style sph/heatconduction coefficients"],
];

/** As ROWS, measured with a native binary built with the package named last (the installed one lacks it). */
const PACKAGE_ROWS: [string, number, string, string][] = [
  ["w189sq_al2o3", 21, "ERROR: Number of element to type mappings does not match number of atom types", "SMTBQ"],
  ["w270ya_pace_yace_cu", 15, "ERROR: Number of element to type mappings does not match number of atom types", "ML-PACE"],
];

describe('an extra word after a coefficient line, as native LAMMPS', () => {
  it.each(ROWS)('%s line %i', async (name, k, native) => {
    const [input, files] = withTrailingWord(name, k);
    expect(await outcome(input, files)).toBe(native);
  });

  it.each(PACKAGE_ROWS)('%s line %i gives %s (native built with %s)', async (name, k, native) => {
    const [input, files] = withTrailingWord(name, k);
    expect(await outcome(input, files)).toBe(native);
  });
});
