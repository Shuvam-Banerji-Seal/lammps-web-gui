import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * A fix, compute, input command or coefficient line with its last word removed, so an argument or a keyword value is
 * missing, refused with native LAMMPS's message: each row takes an oracle case up to one of its lines, drops that line's
 * last word and runs 0 steps. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28; PACKAGE_ROWS with b332f28
 * builds that have the package); the messages come from each command's or style's parser in the C++ source at b332f28.
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

/** The case's input up to line k, whose last word is removed, then run 0, with the files it reads. */
const withoutLastWord = (name: string, k: number): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const body = lines.slice(0, k).filter((l) => !l.startsWith('# oracle')).join('\n') + '\n' + lines[k].split('#')[0].trim().split(/\s+/).slice(0, -1).join(' ') + '\nrun 0\n';
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

/** [oracle case, index of the fix or compute line whose last word is removed (0-based), native result] */
const ROWS: [string, number, string][] = [
  ["w133mf_baoab_lj", 17, "ERROR: Illegal fix baoab zero command: missing argument(s)"],
  ["w13gran_region_hertz", 24, "ERROR: Illegal fix wall/gran command"],
  ["w141cy_cauchy_basic", 18, "ERROR: Illegal fix npt/cauchy command"],
  ["w14ttm_stop", 19, "ERROR: Illegal fix ttm infile command: missing argument(s)"],
  ["w155mx_fepta_tail", 19, "ERROR: Illegal optional keyword in compute fep/ta"],
  ["w15snap_grid", 17, "ERROR: Illegal compute sna/grid command"],
  ["w15snap_grid_local", 18, "ERROR: Illegal compute sna/grid/local command"],
  ["w15snap_group", 17, "ERROR: Illegal compute snap command"],
  ["w171px_saed_vtk_full", 18, "ERROR: Illegal fix saed/vtk file command: missing argument(s)"],
  ["w179rg_nh_deform", 15, "ERROR: Illegal fix rigid/nvt command"],
  ["w224mc_bondcreateangle", 15, "ERROR: Illegal fix bond/create/angle prob command: missing argument(s)"],
  ["w228gs_addtorque", 22, "ERROR: Illegal fix addtorque/atom every command: missing argument(s)"],
  ["w229xf_corrlong_auto", 18, "ERROR: Illegal fix ave/correlate/long file command: missing argument(s)"],
  ["w229xf_nonaffine", 16, "ERROR: Illegal fix nonaffine/displacement command: missing argument(s)"],
  ["w229xf_set", 21, "ERROR: Illegal set vx command: missing argument(s)"],
  ["w230re_eff_langevin", 15, "ERROR: Illegal fix langevin scale command: missing argument(s)"],
  ["w230re_eff_sllod", 15, "ERROR: Illegal fix nvt/sllod/eff temp command: missing argument(s)"],
  ["w230re_rigid_nvt_small_erotate", 21, "ERROR: Illegal fix rigid/nvt/small temp command: missing argument(s)"],
  ["w230re_temp_deform_eff", 15, "ERROR: Illegal fix deform remap command: missing argument(s)"],
  ["w349chreg_onlysalt_anion", 19, "ERROR: Illegal fix charge/regulation command"],
  ["w352gjf_method8", 20, "ERROR: Expected integer parameter instead of NULL or empty string in input script or data file"],
  ["w355dipole_nph_sphere", 21, "ERROR: Illegal fix nph/sphere update command: missing argument(s)"],
  ["w356dlm_npt_sphere", 22, "ERROR: Illegal fix npt/sphere update command: missing argument(s)"],
  ["w36pour2d", 19, "ERROR: Illegal fix pour region command: missing argument(s)"],
  ["w37grav_double", 12, "ERROR: Illegal fix rigid/small gravity command: missing argument(s)"],
  ["w38gauss_example", 11, "ERROR: Illegal compute gaussian/grid/local command"],
  ["w390rheo_ideal_gas", 22, "ERROR: Need to define fix rheo/pressure to use pair rheo"],
  ["w390rheo_ideal_gas", 23, "ERROR: Illegal fix rheo/viscosity constant command: missing argument(s)"],
  ["w390rheo_ideal_gas", 24, "ERROR: Illegal fix rheo/pressure ideal/gas command: missing argument(s)"],
  ["w390rheo_ideal_gas", 25, "ERROR: Illegal fix rheo/thermal specific/heat constant command: missing argument(s)"],
  ["w399dp_box_volume", 15, "ERROR: Illegal fix deform/pressure units command: missing argument(s)"],
  ["w41ttm_cas_cetab", 20, "ERROR: Illegal fix ttm/cascade offset command: missing argument(s)"],
  ["w41ttm_thermal4", 21, "ERROR: Illegal fix ttm/thermal source command: missing argument(s)"],
  ["w421dump_grid_ttm", 20, "ERROR: Illegal fix ttm/grid infile command: missing argument(s)"],
  ["w431rigidnh_ellipsoid", 21, "ERROR: Illegal fix rigid/npt/small iso command: missing argument(s)"],
  ["w43tm_si", 18, "ERROR: Illegal fix ttm/mod set command: missing argument(s)"],
  ["w456dyngroup_reaxff_out", 22, "ERROR: Illegal fix reaxff/species element command: missing argument(s)"],
  ["w459dyngroup_bondcreate", 21, "ERROR: Illegal fix bond/create prob command: missing argument(s)"],
  ["w460dyngroup_neighborswap", 26, "ERROR: Illegal fix neighbor/swap types command: missing argument(s)"],
  ["w47pkg_dtreset_metal", 17, "ERROR: Illegal fix dt/reset command"],
  ["w48cb_tp_twobins", 14, "ERROR: Illegal compute temp/profile command"],
  ["w48fb_ti_spring_tiny", 14, "ERROR: Expected integer parameter instead of NULL or empty string in input script or data file"],
  ["w48sp2_langevin0", 13, "ERROR: Expected floating point parameter instead of NULL or empty string in input script or data file"],
  ["w49hy_halt_min", 20, "ERROR: Illegal fix halt message command: missing argument(s)"],
  ["w50il_chain", 14, "ERROR: Fix ilves selector 'b' needs one or more values"],
  ["w56rnh_nph_iso", 48, "ERROR: Illegal fix rigid/nph command"],
  ["w56rnh_nph_small", 48, "ERROR: Illegal fix rigid/nph/small iso command: missing argument(s)"],
  ["w56rnh_npt_iso", 48, "ERROR: Illegal fix rigid/npt command"],
  ["w62brown_asphere3d", 13, "ERROR: Illegal fix brownian/asphere rotation_temp command: missing argument(s)"],
  ["w62brown_point3d", 12, "ERROR: Illegal fix brownian gamma_t command: missing argument(s)"],
  ["w62brown_propelvel", 16, "ERROR: Compute pressure requires temperature ID to include kinetic energy"],
  ["w69fep_adapt", 23, "ERROR: Illegal fix adapt/fep atom command: missing argument(s)"],
  ["w69fep_compute", 23, "ERROR: Illegal atom attribute in compute fep"],
  ["w6prop_read", 6, "ERROR: Illegal fix property/atom command"],
  ["w79srd_pure", 29, "ERROR: Illegal fix srd collision command: missing argument(s)"],
  ["w7bal_forced", 12, "ERROR: Illegal fix balance shift command: missing argument(s)"],
  ["w7pour_fewer", 15, "ERROR: Illegal fix wall/gran command"],
  ["w8snap_sna", 18, "ERROR: Illegal compute sna/atom command"],
  ["w8snap_snad", 18, "ERROR: Illegal compute snad/atom command"],
  ["w8snap_snav", 18, "ERROR: Illegal compute snav/atom command"],
  ["w94react_nylon_unstab", 24, "ERROR: Illegal fix bond/react command: 'rescale_charges' has too few arguments"],
  ["w99phq_cu", 22, "ERROR: Illegal fix phonon nasr command: missing argument(s)"],
  ["w9adapt_soft", 27, "ERROR: Illegal fix adapt pair command: missing argument(s)"],
];

/** As ROWS, for input commands and coefficient lines (pair_coeff, bond_coeff, angle_coeff, dihedral_coeff). */
const COMMAND_ROWS: [string, number, string][] = [
  ["dump_formats", 12, "ERROR: Illegal dump command: missing argument(s)"],
  ["dump_formats", 14, "ERROR: Illegal dump command: missing argument(s)"],
  ["w104cf_reset_image_all", 10, "ERROR: Illegal reset_atoms image command: missing argument(s)"],
  ["w117cf_tip4p_pppm_water", 15, "ERROR: Illegal kspace_style pppm/tip4p command: missing argument(s)"],
  ["w119amoeba_hippo_hexamer", 16, "OK"],
  ["w119amoeba_mpole_rspace", 15, "OK"],
  ["w128pod_ta_static", 18, "ERROR: Illegal pair_coeff command: missing argument(s)"],
  ["w131cf_bpm_bond_extra", 9, "ERROR: Illegal create_bonds command: missing argument(s)"],
  ["w157mt_bond_angle", 12, "ERROR: Illegal pair_coeff command: missing argument(s)"],
  ["w158cb_cu", 15, "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w15gran_mdr_pair", 18, "ERROR: Insufficient arguments provided for linear_history model"],
  ["w165mc_pour_body_molfrac", 19, "ERROR: Illegal molecule toff command: missing argument(s)"],
  ["w16lepton_bonded", 14, "ERROR: Unmatched double quote in command: angle_coeff 1 109.5 \"40.0*theta^2 +"],
  ["w16lepton_bonded", 16, "ERROR: Unmatched double quote in command: dihedral_coeff 1 \"1.5*(1+cos(2*phi)) +"],
  ["w16lepton_vref", 10, "ERROR: Unmatched double quote in command: bond_coeff 1 1.05 \"120.0*r^2 +"],
  ["w19trigen_general", 13, "ERROR: Illegal dump_modify triclinic/general command: missing argument(s)"],
  ["w217xb_dihedral_spherical", 19, "ERROR: Expected floating point parameter instead of NULL or empty string in input script or data file"],
  ["w225cm_write_molecule", 8, "ERROR: Illegal molecule command: missing argument(s)"],
  ["w242tp_ljlong_shake_tables", 16, "ERROR: Illegal kspace_style pppm/disp/tip4p command"],
  ["w250pd_ljlong_arith", 18, "ERROR: Illegal kspace_style pppm/disp/dielectric command"],
  ["w29dump_image", 11, "ERROR: Illegal dump image view command: missing argument(s)"],
  ["w378balance_out_tri2d", 13, "ERROR: Illegal (fix) balance out command: missing argument(s)"],
  ["w37respa_bond", 17, "ERROR: Not enough rRESPA loop factors provided"],
  ["w390rheo_ideal_gas", 20, "ERROR: Illegal pair rheo artificial/visc command: missing argument(s)"],
  ["w43st_dsf_alone", 18, "ERROR: Illegal pair_style coul/streitz dsf command: missing argument(s)"],
  ["w50bd_cosbuck6d", 15, "ERROR: Incorrect args for pair coefficients"],
  ["w50bd_cosbuck6d_long", 17, "ERROR: Incorrect args for pair coefficients"],
  ["w54accel_charged", 29, "ERROR: Illegal kspace_style pppm/omp command: missing argument(s)"],
  ["w54accel_melt", 2, "ERROR: Illegal package omp command: missing argument(s)"],
  ["w55reax_water", 8, "ERROR: Illegal pair_style reaxff command"],
  ["w76dih_fine_fourier", 12, "ERROR: Expected floating point parameter instead of NULL or empty string in input script or data file"],
  ["w88pair_shift_eam", 13, "ERROR: Illegal thermo_modify format command: missing argument(s)"],
  ["w90label_mol", 7, "ERROR: Incorrect number of arguments for labelmap command"],
  ["w95dreact_shardlow_dpde_1000", 16, "ERROR: pair_style dpd/fdt/energy requires an EOS fix to be specified"],
  ["w96rheo_balloon", 61, "ERROR: Illegal fix rheo surface/detection command: missing argument(s)"],
  ["w9rerun_boxadd", 13, "ERROR: Illegal read_dump add command: missing argument(s)"],
];

/** As ROWS, measured with a native binary built with the package named last (the installed one lacks it). */
const PACKAGE_ROWS: [string, number, string, string][] = [
  ["w276cp_compute_pace_ace", 17, "ERROR: Expected integer parameter instead of NULL or empty string in input script or data file", "ML-PACE"],
  ["w277ma_compute_mliap_ace", 16, "ERROR: Expected boolean parameter instead of NULL or empty string in input script or data file", "ML-PACE and ML-IAP"],
  ["w278ap_atom_weight", 21, "OK", "APIP"],
  ["w279ap_thermostat_shuffle", 16, "OK", "APIP"],
  ["w279ap_thermostat_shuffle", 17, "ERROR: fix lambda_thermostat/apip: seed requires one argument", "APIP"],
  ["w280ap_la_csp_static", 16, "ERROR: the store_stats option of fix lambda/la/csp/apip requires an additional argument", "APIP"],
  ["w282mf_sphere_energy", 9, "ERROR: Not enough args for manifold sphere, 1 expected but got 0", "MANIFOLD"],
  ["w286mf_nvt_ramp", 14, "ERROR: Keyword 'tchain' needs 1 argument", "MANIFOLD"],
  ["w289mf_manifoldforce_min", 13, "ERROR: Illegal fix manifoldforce! No manifold given", "MANIFOLD"],
  ["w189sq_al2o3", 21, "ERROR: Number of element to type mappings does not match number of atom types", "SMTBQ"],
  ["w257pace_product", 17, "ERROR: All pair coeffs are not set. Status:", "ML-PACE"],
  ["w270ya_pace_yace_cu", 15, "ERROR: Number of element to type mappings does not match number of atom types", "ML-PACE"],
  ["w278ap_funcfl_ace", 7, "ERROR: Run command before simulation box is defined", "APIP"],
];

describe('a fix, compute, input command or coefficient line without its last word, as native LAMMPS', () => {
  it.each(ROWS)('%s line %i', async (name, k, native) => {
    const [input, files] = withoutLastWord(name, k);
    expect(await outcome(input, files)).toBe(native);
  });

  it.each(COMMAND_ROWS)('%s line %i', async (name, k, native) => {
    const [input, files] = withoutLastWord(name, k);
    expect(await outcome(input, files)).toBe(native);
  });

  it.each(PACKAGE_ROWS)('%s line %i gives %s (native built with %s)', async (name, k, native) => {
    const [input, files] = withoutLastWord(name, k);
    expect(await outcome(input, files)).toBe(native);
  });
});
