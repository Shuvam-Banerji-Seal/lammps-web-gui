import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * The last numeric word of a command, coefficient line or fix/compute line replaced by the word "bogus" (usually a
 * keyword value: a seed, a damping time, a flag), refused with native LAMMPS's message: the value is read by
 * utils::numeric, inumeric or bounds, or a check of the command comes first. Each row takes an oracle case up to one
 * of its lines, replaces the word (searching back to word 4 of a fix or compute, word 2 of a coefficient line, word 1
 * of a command) and runs 0 steps. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28; PACKAGE_ROWS with
 * b332f28 builds that have the package).
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

/** The line with its last numeric word (from word 4 of a fix or compute, 2 of a coefficient line, 1 else) replaced. */
const badLastNumber = (line: string): string => {
  const w = line.split('#')[0].trim().split(/\s+/);
  const first = w[0] === 'fix' || w[0] === 'compute' ? 4 : w[0].endsWith('_coeff') ? 2 : 1;
  for (let i = w.length - 1; i >= first; i--) {
    if (NUMBER.test(w[i])) { w[i] = 'bogus'; break; }
  }
  return w.join(' ');
};

/** The case's input up to line k, whose last numeric word becomes "bogus", then run 0. */
const withBadNumber = (name: string, k: number): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const body = lines.slice(0, k).filter((l) => !l.startsWith('# oracle')).join('\n') + '\n' + badLastNumber(lines[k]) + '\nrun 0\n';
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

/** [oracle case, index of the line (0-based), native result] */
const ROWS: [string, number, string][] = [
  ["w15snap_grid", 17, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w15snap_grid_local", 18, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w225cm_write_molecule", 7, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w88pair_amber_coul", 12, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w21bhyb_mol_hybrid", 17, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w29dump_image", 11, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w196vtk_modify", 18, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w43ad_snap_single_scale", 25, "ERROR: Invalid range string: bogus"],
  ["w7bal_forced", 12, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w139bc_break_prob", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w459dyngroup_bondcreate", 21, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w224mc_bondcreateangle", 15, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w68elec_thermo", 19, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w352gjf_method8", 20, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w89fix_rigid_langevin", 14, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w230re_eff_langevin", 15, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w141cy_cauchy_basic", 18, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w229xf_pair", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w89fix_pberend_iso", 14, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w456dyngroup_reaxff_out", 22, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w36rigid_lgv2d", 15, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w165mc_store_com", 18, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w89fix_berendsen_onedof", 15, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w89fix_csld_const", 14, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w89fix_csvr_2d", 15, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w41ttm_cas_cetab", 20, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w41ttm_thermal4", 21, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w133mf_viscous_sphere_type", 29, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w38wflow_seam_single", 16, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w13gran_region_hertz", 24, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w16disp_ljcoul", 10, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w52airebo_airebo_00", 9, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w147ab_airebo_bc", 9, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w52airebo_morse_11", 9, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w77dpd_noise_dpd3d_nosort", 14, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w77dpd_noise_ext3d", 13, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w319dpd_ext_tstat_ramp", 11, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w95dreact_dpdvv_1000", 14, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w95dreact_shardlow_dpde_1000", 15, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w77dpd_noise_tstat_spce", 11, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w235mx_dsmc_gas", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w449dyngroup_edpd", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w230re_eff_langevin", 9, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w4gran_hertz_history", 25, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w7pour_fewer", 10, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w36pour2d", 10, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w231co_lubricateU_flaglog0", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w231co_lubricateU_poly", 22, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w450dyngroup_tdpd", 17, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
];

/** [oracle case, line index, result of a native build with the package, the package] */
const PACKAGE_ROWS: [string, number, string, string][] = [
  ["w276cp_compute_pace_ace", 17, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file", "ML-PACE"],
  ["w278ap_atom_weight", 21, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file", "APIP"],
  ["w279ap_thermostat_shuffle", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file", "APIP"],
  ["w280ap_la_csp_static", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file", "APIP"],
  ["w279ap_thermostat_shuffle", 17, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file", "APIP"],
  ["w282mf_sphere_energy", 9, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file", "MANIFOLD"],
  ["w286mf_nvt_ramp", 14, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file", "MANIFOLD"],
];

/*
 * No native build here has the LATBOLTZ or QTB package; the expected messages follow from the C++ source at b332f28:
 * fix_lb_fluid.cpp reads the npits values with utils::inumeric, fix_lb_momentum.cpp the linear flags with
 * utils::inumeric, fix_qbmsst.cpp T_init with utils::numeric and fix_qtb.cpp N_f with utils::inumeric.
 */
const SOURCE_ROWS: [string, number, string][] = [
  ["w181px_lb_pits_sw", 20, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w132lb_momentum", 22, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
  ["w99phq_qbmsst_lj", 16, "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w99phq_qtb_lj", 16, "ERROR: Expected integer parameter instead of 'bogus' in input script or data file"],
];

describe('a bad last number (a keyword value) as native LAMMPS', () => {
  it.each(ROWS)('%s line %i: %s', async (name, k, expected) => {
    const [input, files] = withBadNumber(name, k);
    expect(await outcome(input, files)).toBe(expected);
  });

  it.each(PACKAGE_ROWS)('%s line %i: %s (%s build)', async (name, k, expected) => {
    const [input, files] = withBadNumber(name, k);
    expect(await outcome(input, files)).toBe(expected);
  });

  it.each(SOURCE_ROWS)('%s line %i: %s (from the C++ source)', async (name, k, expected) => {
    const [input, files] = withBadNumber(name, k);
    expect(await outcome(input, files)).toBe(expected);
  });
});
