import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * A fix or compute line whose first numeric argument after the style name is replaced by the word "bogus", refused with
 * native LAMMPS's message: the value is read by utils::numeric, inumeric, logical or bounds, or a style's own check comes
 * first. Each row takes an oracle case up to one of its lines, replaces the word and runs 0 steps. Measured with native
 * LAMMPS (/home/roy/.local/bin/lmp, b332f28; PACKAGE_ROWS with b332f28 builds that have the package).
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

const NUMBER = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

/** The line with its first numeric word after the style name (word 4 on) replaced by "bogus". */
const badNumber = (line: string): string => {
  const w = line.split('#')[0].trim().split(/\s+/);
  const i = w.findIndex((x, j) => j >= 4 && NUMBER.test(x));
  if (i >= 0) w[i] = 'bogus';
  return w.join(' ');
};

/** The case's input up to line k, whose first numeric argument after the style becomes "bogus", then run 0. */
const withBadNumber = (name: string, k: number): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const body = lines.slice(0, k).filter((l) => !l.startsWith('# oracle')).join('\n') + '\n' + badNumber(lines[k]) + '\nrun 0\n';
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

/** [oracle case, index of the fix or compute line (0-based), native result] */
const ROWS: [string, number, string][] = [
  ["deposit_rate_lattice", 17, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w108xf_elfit_sisi", 18, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w108xf_gle_lj_eq", 15, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w108xf_grem_nvt", 13, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w10boxrelax_iso", 16, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w11restrain_springrg", 16, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w126body_cubes_wall", 17, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w126body_wall2d", 19, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w133mf_baoab_lj", 17, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w133mf_flow_gauss_mod", 29, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w133mf_orient_eco", 15, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w133mf_region_ees", 15, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w133mf_viscous_sphere_var", 25, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w134cp_entropy_fe", 25, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w134cp_saed_ni", 12, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w134cp_xrd_ni", 12, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w139bc_break_prob", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w13misc_controller", 21, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w13misc_controller", 23, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w142mx_append_atoms", 17, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w149cp_mop_setup", 24, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w149cp_sph", 15, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w14ttm_stop", 19, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w155mx_pafi_od", 33, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w162rx_table_lookup", 12, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w162rx_table_lookup", 44, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w171px_saed_vtk_full", 18, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w171px_slcsa", 20, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w179rg_nh_deform", 15, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w18fix_drag", 18, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w18fix_oneway", 18, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w218ck_fix_momentum", 28, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w218ck_inertia_slice", 24, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w224mc_bondcreateangle", 15, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w227xc_ti_single", 21, "ERROR: Invalid range string: bogus"],
  ["w229xf_efield_tip4p", 20, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w229xf_set", 21, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w22rwall_region_union", 17, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w230re_eff_sllod", 15, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w230re_eos_table", 23, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w230re_rigid_nvt_small_erotate", 21, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w230re_smd_wall_linear", 40, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w230re_smd_wall_linear", 44, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w230re_smd_wall_linear", 48, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w26cgrid_propgrid", 15, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w26cgrid_propgrid2d", 18, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w2cred_coord", 29, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w2fmom_recenter", 28, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w2fset_aveforce", 28, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w2fspring_tether", 28, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w30widom_atom", 14, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w323spring_chunk_wrap", 27, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w33gcmc_real_ins", 11, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w340deposit_tail", 20, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w349chreg_onlysalt_anion", 19, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w36pour2d", 19, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w38wflow_seam_single", 16, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w39cfs_wrap", 12, "ERROR: Expected boolean parameter instead of 'bogus' in input script or data file"],
  ["w416mop_bonded", 24, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w41nptp_bias", 12, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w41ttm_cas_cetab", 20, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w41ttm_thermal4", 21, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w421dump_grid_ttm", 20, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w431rigidnh_ellipsoid", 21, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w436dyngroup_sphere", 21, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w43tm_si", 18, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w456dyngroup_reaxff_out", 21, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w456dyngroup_reaxff_out", 22, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w457dyngroup_smd_setvel", 64, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w459dyngroup_bondcreate", 21, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w460dyngroup_neighborswap", 26, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w47sp_cart1d", 19, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w48fb_tmd_mass", 24, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w48sp2_langevin0", 13, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w48sp2_langevin0", 14, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w48sp2_setforce", 19, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w49hy_halt_min", 20, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w56rnh_nph_iso", 48, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w56rnh_nph_small", 48, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w56rnh_npt_iso", 48, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w58comb_cu2o", 21, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w63eff_ch4_rescale", 22, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w66units_efield_const_real", 23, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w66units_lineplane_real", 24, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w66units_lineplane_real", 25, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w69fep_adapt", 23, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w6prop_read", 6, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w72drude_butane_lang", 36, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w79srd_pure", 29, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w7bal_forced", 12, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w89fix_rescale_onedof", 15, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w89fix_storestate_every", 14, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w94react_nylon_unstab", 24, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w9adapt_soft", 27, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
];

/** As ROWS, measured with a native binary built with the package named last (the installed one lacks it). */
const PACKAGE_ROWS: [string, number, string, string][] = [
  ["w276cp_compute_pace_ace", 17, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file", "ML-PACE"],
  ["w277ma_compute_mliap_ace", 16, "ERROR: Expected boolean parameter instead of 'bogus' in input script or data file", "ML-PACE and ML-IAP"],
  ["w278ap_atom_weight", 21, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file", "APIP"],
  ["w279ap_thermostat_shuffle", 16, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file", "APIP"],
  ["w279ap_thermostat_shuffle", 17, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file", "APIP"],
  ["w280ap_la_csp_static", 16, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file", "APIP"],
  ["w282mf_sphere_energy", 9, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file", "MANIFOLD"],
  ["w286mf_nvt_ramp", 14, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file", "MANIFOLD"],
  ["w289mf_manifoldforce_min", 13, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file", "MANIFOLD"],
];

describe('a fix or compute line with a word where a number belongs, as native LAMMPS', () => {
  it.each(ROWS)('%s line %i', async (name, k, native) => {
    const [input, files] = withBadNumber(name, k);
    expect(await outcome(input, files)).toBe(native);
  });

  it.each(PACKAGE_ROWS)('%s line %i gives %s (native built with %s)', async (name, k, native) => {
    const [input, files] = withBadNumber(name, k);
    expect(await outcome(input, files)).toBe(native);
  });
});
