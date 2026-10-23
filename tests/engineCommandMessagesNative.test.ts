import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * Trailing words after an input command (atom_style, neighbor, kspace_style, dump, pair_style, run_style, ...), refused
 * or ignored with native LAMMPS's message: each row takes an oracle case up to one of its lines, adds the words
 * "bogus 1" to that line and runs 0 steps. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28); the
 * messages come from each command's or style's parser in the C++ source at b332f28.
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

/** The case's input up to line k, which gets the words "bogus 1", then run 0, with the files it reads. */
const withTrailingWords = (name: string, k: number): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const body = lines.slice(0, k).filter((l) => !l.startsWith('# oracle')).join('\n') + '\n' + lines[k].split('#')[0].trimEnd() + ' bogus 1\nrun 0\n';
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

/** [oracle case, index of the line that gets " bogus 1" (0-based), native result] */
const ROWS: [string, number, string][] = [
  ["dump_formats", 12, "ERROR: Illegal dump atom command"],
  ["dump_formats", 13, "ERROR: Invalid attribute bogus in dump custom command"],
  ["dump_formats", 14, "ERROR: Illegal dump xyz command"],
  ["propl_newtonoff", 24, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w100smd_aluminum_strip", 120, "ERROR: Invalid attribute bogus in dump custom command"],
  ["w104cf_reset_id", 4, "ERROR: Invalid atom_style command"],
  ["w104cf_reset_id", 10, "ERROR: Unknown reset_atoms id keyword: bogus"],
  ["w104cf_sds_coul_alone", 13, "ERROR: Illegal kspace_style ewald command"],
  ["w105cf_erotate_line", 6, "ERROR: Invalid atom_style command"],
  ["w105cf_erotate_tri", 6, "ERROR: Invalid atom_style command"],
  ["w106gs_tri_rot", 17, "ERROR: Illegal pair_coeff command bogus"],
  ["w113cf_benzene_table_spline", 9, "ERROR: Unknown pair_style table keyword: bogus"],
  ["w116xf_chreg_acid_real", 8, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w117cf_tip4p_pppm_water", 15, "OK"],
  ["w119amoeba_angle", 6, "ERROR: Invalid atom_style command"],
  ["w11local_bond", 12, "ERROR: Invalid attribute bogus in dump local command"],
  ["w131cf_bpm_bond_extra", 9, "ERROR: Illegal create_bonds command"],
  ["w135pd_fixed_fcc", 17, "OK"],
  ["w137cf_delbonds_stale", 18, "ERROR: Illegal delete_bonds command"],
  ["w145pr_pair_extep", 10, "ERROR: Illegal processors command"],
  ["w149cp_rxatom", 9, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w156mx_bocs", 16, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w167om_si_cluster", 17, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w16lepton_coul", 18, "ERROR: Unknown pair_style lepton/coul keyword: bogus"],
  ["w174mx_pn_gran_hooke_history", 8, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w17chunk_pointdipole", 2, "ERROR: Invalid atom_style command"],
  ["w183md_tableting_split", 11, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w196vtk_xml", 14, "ERROR: Invalid attribute bogus in dump vtk command"],
  ["w19trigen_thermo", 7, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w215cfg_dump", 19, "ERROR: Invalid attribute bogus in dump cfg command"],
  ["w225cm_write_molecule", 5, "ERROR: Invalid atom_style command"],
  ["w225cm_write_molecule", 9, "ERROR: Not all per-type masses are set. Type 1 is missing."],
  ["w226cs_diel_debye", 4, "ERROR: Invalid atom_style command"],
  ["w228gs_sph_lj", 6, "ERROR: Invalid atom_style command"],
  ["w22peri_pmb_data", 5, "ERROR: Invalid atom_style command"],
  ["w22peri_pmb_data", 7, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w230re_smd_wall_linear", 15, "ERROR: Invalid atom_style command"],
  ["w231co_lubricate_nolog_fld", 18, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w231co_lubricate_poly", 21, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w237pk_stagger_acc5_order4", 28, "OK"],
  ["w243de_efield_pair_kw_pppm", 13, "OK"],
  ["w24tmpl_read", 7, "ERROR: Illegal atom_style template command"],
  ["w250pd_ljlong_arith", 18, "ERROR: PPPMDisp used but no parameters set, for further information please see the pppm/disp documentation"],
  ["w268el_pppm_peratom", 6, "ERROR: Run command before simulation box is defined"],
  ["w272sp_pppm_spin_peatom", 20, "OK"],
  ["w303vsplit_pppm", 23, "ERROR: Verlet/split requires 2 partitions"],
  ["w304vsplit_rk", 17, "ERROR: The Integrate type must support an rk decomposition (e.g., run_style verlet/split/rk) that is compatible with kspace style pppm/rk"],
  ["w304vsplit_rk", 23, "ERROR: pppm/rk requires 2 partitions"],
  ["w310gneb_cg", 14, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w310gneb_cg", 25, "ERROR: Cannot use NEBSpin with a single replica"],
  ["w312prd_vel_local", 33, "ERROR: Illegal prd command"],
  ["w319dpd_ext_tstat_ramp", 14, "OK"],
  ["w35c2imp", 4, "ERROR: Invalid atom_style command"],
  ["w35mesh_tri", 4, "ERROR: Illegal atom_style sphere command"],
  ["w367eff_nph_couple_xy", 6, "ERROR: Invalid atom_style command"],
  ["w378balance_out_tri2d", 13, "ERROR: Unknown (fix) balance keyword bogus"],
  ["w37respa_bond", 17, "ERROR: Unknown run_style respa keyword bogus"],
  ["w390rheo_ideal_gas", 4, "ERROR: Invalid atom_style command"],
  ["w40core_restart_lj", 15, "ERROR: Unknown write_restart keyword: bogus"],
  ["w40ksp_zerocoul_zero", 21, "ERROR: Unknown pair style zero option bogus"],
  ["w423dump_grid_2d", 20, "ERROR: Invalid attribute bogus in dump grid/vtk command"],
  ["w426avegrid_discard", 13, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w43st_dsf_alone", 18, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w449dyngroup_edpd", 8, "ERROR: Invalid atom_style command"],
  ["w449dyngroup_edpd", 19, "ERROR: Illegal pair edpd coefficients"],
  ["w450dyngroup_tdpd", 7, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w450dyngroup_tdpd", 9, "ERROR: Run command before simulation box is defined"],
  ["w451dyngroup_mdpd", 8, "ERROR: Invalid atom_style command"],
  ["w46neb_tiny", 24, "ERROR: Cannot use NEB with a single replica"],
  ["w47ls_cucg", 10, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w47qm_fire", 7, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w48fb_rhok_npt", 21, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w49pb_ewald_dielectric", 13, "ERROR: Illegal dielectric command"],
  ["w50bc_static", 5, "ERROR: Invalid body rounded/polygon command"],
  ["w50bd_pppmcg", 25, "ERROR: Illegal kspace_style pppm/cg command"],
  ["w51cs_sdk_angle", 4, "ERROR: Invalid atom_style command"],
  ["w51cs_sdk_peg", 12, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w52rng_spin3d", 3, "ERROR: Invalid atom_style command"],
  ["w53bpm_rot_chain", 5, "ERROR: Illegal atom_style bpm/sphere command"],
  ["w54accel_bonded", 14, "ERROR: Illegal bond_style harmonic/omp argument: bogus"],
  ["w54accel_bonded", 17, "ERROR: Illegal angle_style harmonic/omp argument: bogus"],
  ["w54accel_charged", 29, "OK"],
  ["w54accel_melt", 2, "ERROR: Unknown package omp keyword bogus"],
  ["w54accel_suffix", 3, "ERROR: Illegal suffix command"],
  ["w55reax_water", 5, "ERROR: Invalid atom_style command"],
  ["w55reax_water", 8, "ERROR: Illegal pair_style reaxff command"],
  ["w55reax_water20", 10, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w57oxdna_excv_pair2", 4, "ERROR: Invalid atom_style command"],
  ["w58comb_si0", 14, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w62brown_asphere3d", 3, "ERROR: Unknown atom_style ellipsoid argument: bogus"],
  ["w66units_gravity_chute_si", 16, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w68elec_madelung_cg", 5, "ERROR: Illegal kspace_style ewald/electrode command"],
  ["w75pppmslab_madelung_lowacc", 3, "ERROR: Run command before simulation box is defined"],
  ["w78oxdna3_real_duplex2", 11, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w83sph_cavity", 52, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w83sph_heat3d", 21, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w87bond_angle_fourier_linear", 8, "ERROR: Unknown pair style zero option bogus"],
  ["w88pair_shift_eam", 3, "ERROR: Invalid atom_style command"],
  ["w88pair_shift_eam", 6, "ERROR: Unknown create_box keyword: bogus"],
  ["w95dreact_dpdvv_1000", 8, "ERROR: Invalid atom_style command"],
  ["w96rheo_pair_solid", 4, "ERROR: Invalid atom_style command"],
  ["w99phq_dynmat_silicon", 11, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w99phq_dynmat_silicon", 16, "ERROR: Illegal dynamical_matrix command"],
  ["w99phq_graphene", 18, "ERROR: Illegal neighbor command: expected 2 arguments but found 4"],
  ["w9rerun_boxadd", 13, "ERROR: Unknown read_dump keyword: bogus"],
];

describe('trailing words after an input command, as native LAMMPS', () => {
  it.each(ROWS)('%s line %i', async (name, k, native) => {
    const [input, files] = withTrailingWords(name, k);
    expect(await outcome(input, files)).toBe(native);
  });
});
