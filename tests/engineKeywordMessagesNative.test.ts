import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * A word after a fix's or compute's arguments, refused (or ignored) with native LAMMPS's message: each row takes an
 * oracle case up to its first line of that style and adds the words "bogus 1" to that line. Measured with native
 * LAMMPS (/home/roy/.local/bin/lmp, b332f28; PACKAGE_ROWS with b332f28 builds that have the package). The messages
 * come from each style's constructor (its keyword loop or argument count check) in the C++ source at b332f28.
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

/** The case's input up to its first kind/style line (without include, jump or variable references), plus "bogus 1". */
const withTrailingWord = (name: string, kind: string, style: string): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const k = lines.findIndex((l, i) => {
    const w = l.split('#')[0].trim().split(/\s+/);
    if (w[0] !== kind || w[3] !== style || l.includes('&') || l.includes('$')) return false;
    return !lines.slice(0, i + 1).some((x) => /^\s*(include|jump|label|next|read_restart)\b/.test(x));
  });
  expect(k, `${name}: no ${kind} ${style} line`).toBeGreaterThanOrEqual(0);
  const body = lines.slice(0, k).filter((l) => !l.startsWith('# oracle')).join('\n') + '\n' + lines[k].split('#')[0].trimEnd() + ' bogus 1\n';
  const files: Record<string, string> = {};
  for (const f of directive(text, 'oracle-inputs')) files[f] = readInput(join(ORACLE, f));
  for (const f of directive(text, 'oracle-potentials')) files[f] = readFileSync(join(POTENTIALS, f), 'utf8');
  for (const f of directive(text, 'oracle-examples')) files[f.split('/').pop()!] = readInput(join(EXAMPLES, f));
  // files of the oracle directory that the lines name without a directive (molecule and target files)
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

/** [oracle case, fix or compute, style, native result] */
const ROWS: [string, string, string, string][] = [
  ["w21adf_lj", "compute", "adf", "ERROR: Unknown compute adf keyword bogus"],
  ["w17chunk_bin", "compute", "angmom/chunk", "ERROR: Illegal compute angmom/chunk command"],
  ["w17chunk_bin", "compute", "chunk/spread/atom", "ERROR: Illegal compute chunk/spread/atom argument: bogus"],
  ["w39cfs_wrap_tri", "compute", "com/chunk", "ERROR: Unknown compute com/chunk keyword bogus"],
  ["w2cred_coord", "compute", "coord/atom", "ERROR: Invalid range string: bogus"],
  ["w134cp_entropy_fe", "compute", "entropy/atom", "ERROR: Illegal compute entropy/atom command"],
  ["w69fep_compute", "compute", "fep", "ERROR: Illegal optional keyword in compute fep"],
  ["w155mx_fepta_tail", "compute", "fep/ta", "ERROR: Illegal optional keyword in compute fep/ta"],
  ["w17chunk_bin", "compute", "gyration/chunk", "ERROR: Illegal compute gyration/chunk command"],
  ["w39cfs_ell", "compute", "inertia/chunk", "ERROR: Illegal compute inertia/chunk command"],
  ["w7chunk_msd", "compute", "msd/chunk", "ERROR: Incorrect number of arguments for compute msd/chunk"],
  ["w39cfs_ell", "compute", "omega/chunk", "ERROR: Illegal compute omega/chunk command"],
  ["w155mx_podatom_ta", "compute", "pod/atom", "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w177px_pod_example", "compute", "pod/global", "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w177px_pod_example", "compute", "pod/local", "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w171px_podd_ta", "compute", "podd/atom", "ERROR: Number of element to type mappings does not match number of atom types"],
  ["w17chunk_bin", "compute", "property/chunk", "ERROR: Unkown keyword bogus in compute property/chunk command"],
  ["w26cgrid_propgrid2d", "compute", "property/grid", "ERROR: Unknown compute property/grid keyword: bogus"],
  ["w2cglob_rdf", "compute", "rdf", "ERROR: Unknown compute rdf keyword bogus"],
  ["w17chunk_bin", "compute", "reduce/chunk", "ERROR: Illegal compute reduce/chunk argument: bogus"],
  ["w134cp_saed_ni", "compute", "saed", "ERROR: Illegal Compute SAED Command"],
  ["w218ck_inertia_slice", "compute", "slice", "ERROR: Illegal compute slice argument: bogus"],
  ["w8snap_sna", "compute", "sna/atom", "ERROR: Illegal compute sna/atom command"],
  ["w15snap_grid", "compute", "sna/grid", "ERROR: Illegal compute sna/grid command"],
  ["w15snap_grid_local", "compute", "sna/grid/local", "ERROR: Illegal compute sna/grid/local command"],
  ["w8snap_snad", "compute", "snad/atom", "ERROR: Illegal compute snad/atom command"],
  ["w15snap_group", "compute", "snap", "ERROR: Illegal compute snap command"],
  ["w8snap_snav", "compute", "snav/atom", "ERROR: Illegal compute snav/atom command"],
  ["w47sp_cart1d", "compute", "stress/cartesian", "ERROR: Unknown compute stress/cartesian keyword: bogus"],
  ["w149cp_mop_setup", "compute", "stress/mop", "ERROR: Unknown compute stress/mop keyword bogus"],
  ["w416mop_bonded", "compute", "stress/mop/profile", "ERROR: Unknown compute stress/mop/profile keyword bogus"],
  ["w14temp_chunk_nvt", "compute", "temp/chunk", "ERROR: Illegal compute temp/chunk command"],
  ["w230re_eff_langevin", "compute", "temp/eff", "OK"],
  ["w48cb_tp_twobins", "compute", "temp/profile", "ERROR: Illegal compute temp/profile command"],
  ["w14temp_ramp", "compute", "temp/ramp", "ERROR: Illegal compute temp/ramp command"],
  ["w134cp_xrd_ni", "compute", "xrd", "ERROR: Illegal Compute XRD Command"],
  ["w65reaxq_acks2_field", "fix", "acks2/reaxff", "ERROR: Illegal fix acks2/reaxff command"],
  ["w9adapt_soft", "fix", "adapt", "ERROR: Unknown fix adapt keyword bogus"],
  ["w69fep_adapt", "fix", "adapt/fep", "ERROR: Illegal fix adapt/fep command"],
  ["w437dyngroup_bpm_heat", "fix", "add/heat", "ERROR: Illegal fix add/heat command: missing argument(s)"],
  ["w13misc_controller", "fix", "addforce", "ERROR: Unknown fix addforce keyword: bogus"],
  ["w142mx_append_atoms", "fix", "append/atoms", "ERROR: Illegal fix append/atoms command"],
  ["w26cgrid_propgrid", "fix", "ave/grid", "ERROR: Unknown fix ave/grid keyword: bogus"],
  ["w2fset_aveforce", "fix", "aveforce", "ERROR: Illegal fix aveforce command"],
  ["w7bal_forced", "fix", "balance", "ERROR: Unknown (fix) balance keyword bogus"],
  ["w139bc_break_prob", "fix", "bond/break", "ERROR: Illegal fix bond/break command"],
  ["w224mc_bondcreateangle", "fix", "bond/create/angle", "ERROR: Unknown fix bond/create/angle keyword bogus"],
  ["w94react_nylon_unstab", "fix", "bond/react", "ERROR: Illegal fix bond/react command: unknown keyword"],
  ["w10boxrelax_iso", "fix", "box/relax", "ERROR: Unknown fix box/relax keyword bogus"],
  ["w349chreg_onlysalt_anion", "fix", "charge/regulation", "ERROR: Illegal fix charge/regulation command"],
  ["w230re_temp_deform_eff", "fix", "deform", "ERROR: Unknown fix deform keyword: bogus"],
  ["w399dp_box_volume", "fix", "deform/pressure", "ERROR: Unknown fix deform/pressure keyword: bogus"],
  ["w47pkg_dtreset_metal", "fix", "dt/reset", "ERROR: Illegal fix dt/reset command"],
  ["w449dyngroup_edpd", "fix", "edpd/source", "ERROR: Illegal fix edpd/source command (6 args for cuboid)"],
  ["w66units_efield_const_real", "fix", "efield", "ERROR: Unknown keyword for fix efield command: bogus"],
  ["w229xf_efield_tip4p", "fix", "efield/tip4p", "ERROR: Unknown keyword for fix efield/tip4p command: bogus"],
  ["w68elec_conq_cg", "fix", "electrode/conq", "ERROR: Unknown keyword bogus for fix electrode/conq command"],
  ["w68elec_thermo", "fix", "electrode/thermo", "ERROR: Unknown keyword bogus for fix electrode/thermo command"],
  ["w47pkg_elstop_cu", "fix", "electron/stopping", "ERROR: Illegal fix electron/stopping command: unknown argument"],
  ["w108xf_elfit_sisi", "fix", "electron/stopping/fit", "OK"],
  ["w162rx_table_lookup", "fix", "eos/table/rx", "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w340deposit_tail", "fix", "evaporate", "ERROR: Illegal fix evaporate command"],
  ["w133mf_flow_gauss_mod", "fix", "flow/gauss", "ERROR: Illegal fix flow/gauss command"],
  ["w33gcmc_real_ins", "fix", "gcmc", "ERROR: Unknown fix gcmc keyword bogus"],
  ["w352gjf_method8", "fix", "gjf", "ERROR: Illegal fix gjf command"],
  ["w108xf_gle_lj_eq", "fix", "gle", "OK"],
  ["w36pour2d", "fix", "gravity", "ERROR: Illegal fix gravity command"],
  ["w108xf_grem_nvt", "fix", "grem", "OK"],
  ["w49hy_halt_min", "fix", "halt", "ERROR: Unknown fix halt keyword bogus"],
  ["w89fix_rigid_langevin", "fix", "langevin", "ERROR: Unknown fix langevin keyword bogus"],
  ["w72drude_butane_lang", "fix", "langevin/drude", "ERROR: Illegal fix langevin/drude command"],
  ["w30molswap_keno", "fix", "mol/swap", "ERROR: Illegal fix mol/swap command"],
  ["w451dyngroup_mdpd", "fix", "mvv/dpd", "ERROR: Expected floating point parameter instead of 'bogus' in input script or data file"],
  ["w449dyngroup_edpd", "fix", "mvv/edpd", "OK"],
  ["w450dyngroup_tdpd", "fix", "mvv/tdpd", "OK"],
  ["w16numdiff_forces", "fix", "numdiff", "OK"],
  ["w16numdiff_virial", "fix", "numdiff/virial", "OK"],
  ["w54accel_melt", "fix", "nve/omp", "OK"],
  ["w18fix_oneway", "fix", "oneway", "OK"],
  ["w155mx_pafi_od", "fix", "pafi", "ERROR: Illegal fix pafi command"],
  ["w36pour2d", "fix", "pour", "ERROR: Illegal fix pour command argument: bogus"],
  ["w48sp2_langevin0", "fix", "precession/spin", "ERROR: Illegal precession/spin command"],
  ["w89fix_pberend_iso", "fix", "press/berendsen", "ERROR: Illegal fix press/berendsen command"],
  ["w58comb_cu2o", "fix", "qeq/comb", "ERROR: Illegal fix qeq/comb command"],
  ["w55reax_qeq_rdx_tight", "fix", "qeq/reax", "ERROR: Illegal fix qeq/reax command"],
  ["w55reax_qeq_water", "fix", "qeq/reaxff", "ERROR: Illegal fix qeq/reaxff command"],
  ["w65reaxq_qeqr_tight", "fix", "qeq/rel/reaxff", "ERROR: Illegal fix qeq/rel/reaxff command"],
  ["w65reaxq_qtpie_tight", "fix", "qtpie/reaxff", "ERROR: Illegal fix qtpie/reaxff command"],
  ["w2fmom_recenter", "fix", "recenter", "ERROR: Unknown fix recenter keyword bogus"],
  ["w89fix_rigid_langevin", "fix", "rigid", "ERROR: Illegal fix rigid command"],
  ["w123q_meso_disk", "fix", "rigid/meso", "ERROR: Illegal fix rigid/meso command"],
  ["w56rnh_nph_iso", "fix", "rigid/nph", "ERROR: Illegal fix rigid/nph command"],
  ["w56rnh_nph_small", "fix", "rigid/nph/small", "ERROR: Unknown fix rigid/nph/small keyword bogus"],
  ["w56rnh_npt_iso", "fix", "rigid/npt", "ERROR: Illegal fix rigid/npt command"],
  ["w431rigidnh_ellipsoid", "fix", "rigid/npt/small", "ERROR: Unknown fix rigid/npt/small keyword bogus"],
  ["w474rigid_nve_ellipsoids", "fix", "rigid/nve", "ERROR: Illegal fix rigid/nve command"],
  ["w251rg_nve_small", "fix", "rigid/nve/small", "ERROR: Unknown fix rigid/nve/small keyword bogus"],
  ["w179rg_nh_deform", "fix", "rigid/nvt", "ERROR: Illegal fix rigid/nvt command"],
  ["w230re_rigid_nvt_small_erotate", "fix", "rigid/nvt/small", "ERROR: Unknown fix rigid/nvt/small keyword bogus"],
  ["w37grav_double", "fix", "rigid/small", "ERROR: Unknown fix rigid/small keyword bogus"],
  ["w162rx_table_lookup", "fix", "rx", "ERROR: Too many arguments (10) for RK4 solver"],
  ["w171px_saed_vtk_full", "fix", "saed/vtk", "ERROR: Illegal fix saed/vtk command"],
  ["w229xf_set", "fix", "set", "ERROR: Unrecognized set or fix set command keyword bogus"],
  ["deposit_rate_lattice", "fix", "setforce", "ERROR: Unknown fix setforce keyword: bogus"],
  ["w48sp2_setforce", "fix", "setforce/spin", "ERROR: Unknown fix setforce keyword: bogus"],
  ["w230re_smd_wall_linear", "fix", "smd/integrate_ulsph", "ERROR: Unknown fix smd/integrate_ulsph keyword: bogus"],
  ["w457dyngroup_smd_setvel", "fix", "smd/setvel", "ERROR: Illegal fix setvelocity command"],
  ["w2fspring_tether", "fix", "spring", "ERROR: Incorrect number of arguments for tether mode"],
  ["w89fix_storestate_every", "fix", "store/state", "ERROR: Unknown fix store/state keyword: bogus"],
  ["w450dyngroup_tdpd", "fix", "tdpd/source", "ERROR: Illegal fix tdpd/source command (7 args for cuboid)"],
  ["w89fix_rescale_onedof", "fix", "temp/rescale", "OK"],
  ["w63eff_ch4_rescale", "fix", "temp/rescale/eff", "OK"],
  ["w11gk_thermcond", "fix", "thermal/conductivity", "ERROR: Illegal fix thermal/conductivity command"],
  ["w48fb_tmd_mass", "fix", "tmd", "OK"],
  ["w14ttm_stop", "fix", "ttm", "ERROR: Unknown fix ttm keyword bogus"],
  ["w41ttm_cas_cetab", "fix", "ttm/cascade", "ERROR: Unknown fix ttm/cascade keyword bogus"],
  ["w421dump_grid_ttm", "fix", "ttm/grid", "ERROR: Unknown fix ttm/grid keyword bogus"],
  ["w43tm_si", "fix", "ttm/mod", "ERROR: Unknown fix ttm/mod keyword bogus"],
  ["w41ttm_thermal4", "fix", "ttm/thermal", "ERROR: Unknown fix ttm/thermal keyword bogus"],
  ["w11gk_viscosity", "fix", "viscosity", "ERROR: Illegal fix viscosity command"],
  ["w133mf_viscous_sphere_var", "fix", "viscous/sphere", "ERROR: Illegal fix viscous/sphere command"],
  ["w38wflow_seam_single", "fix", "wall/flow", "ERROR: Wrong fix wall/flow wall count"],
  ["w7pour_fewer", "fix", "wall/gran", "ERROR: Illegal fix wall/gran command"],
  ["w13gran_region_hertz", "fix", "wall/gran/region", "ERROR: Illegal fix wall/gran command"],
  ["w22rwall_region_union", "fix", "wall/region", "ERROR: Illegal fix wall/region command"],
  ["w30widom_atom", "fix", "widom", "ERROR: Illegal fix widom command"],
];

/** As ROWS, measured with a native binary built with the package named last (the installed one lacks it). */
const PACKAGE_ROWS: [string, string, string, string, string][] = [
  ["w277ma_compute_mliap_ace", "compute", "mliap", "ERROR: Unknown compute mliap keyword bogus", "ML-PACE and ML-IAP"],
  ["w278ap_atom_weight", "fix", "atom_weight/apip", "ERROR: Fix atom_weight/apip: unknown argument bogus", "APIP"],
  ["w279ap_thermostat_shuffle", "fix", "lambda/apip", "ERROR: fix lambda/apip: unknown argument bogus", "APIP"],
  ["w280ap_la_csp_static", "fix", "lambda/la/csp/apip", "ERROR: unknown argument bogus", "APIP"],
  ["w279ap_thermostat_shuffle", "fix", "lambda_thermostat/apip", "ERROR: fix lambda_thermostat/apip: unknown argument bogus", "APIP"],
  ["w289mf_manifoldforce_min", "fix", "manifoldforce", "OK", "MANIFOLD"],
  ["w282mf_sphere_energy", "fix", "nve/manifold/rattle", "ERROR: Error parsing arg \"bogus\".", "MANIFOLD"],
  ["w286mf_nvt_ramp", "fix", "nvt/manifold/rattle", "ERROR: Error parsing arg \"bogus\".", "MANIFOLD"],
];

describe('a trailing word after a fix or compute, as native LAMMPS', () => {
  it.each(ROWS)('%s: %s %s', async (name, kind, style, native) => {
    const [input, files] = withTrailingWord(name, kind, style);
    expect(await outcome(input, files)).toBe(native);
  });

  it.each(PACKAGE_ROWS)('%s: %s %s gives %s (native built with %s)', async (name, kind, style, native) => {
    const [input, files] = withTrailingWord(name, kind, style);
    expect(await outcome(input, files)).toBe(native);
  });
});
