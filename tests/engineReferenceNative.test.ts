import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * A line of an oracle case whose group ID, compute / fix / variable reference or type number is replaced (by
 * "nogroup", c_nosuch, f_nosuch, v_nosuch or 99), refused with native LAMMPS's message: the group, compute, fix or
 * variable is looked up where native looks it up (when the command is read, or at init for most fix variables), with the
 * style's own message. Each row takes the case up to the line, puts the given line in its place and runs 0 steps.
 * Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28; PACKAGE_ROWS with b332f28 builds that have the
 * package).
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

/** The case's input up to line k, then the given line in its place, then run 0. */
const withLine = (name: string, k: number, line: string): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const body = lines.slice(0, k).filter((l) => !l.startsWith('# oracle')).join('\n') + '\n' + line + '\nrun 0\n';
  const files: Record<string, string> = {};
  for (const f of directive(text, 'oracle-inputs')) files[f] = readInput(join(ORACLE, f));
  for (const f of directive(text, 'oracle-potentials')) files[f] = readFileSync(join(POTENTIALS, f), 'utf8');
  for (const f of directive(text, 'oracle-examples')) files[f.split('/').pop()!] = readInput(join(EXAMPLES, f));
  // files of the oracle directory that the lines name without a directive
  for (const w of body.split(/\s+/)) if (ORACLE_FILES.has(w) && !(w in files)) files[w] = readInput(join(ORACLE, w));
  return [body, files];
};

const outcome = async (input: string, files: Record<string, string> = {}): Promise<string> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  for (const [name, text] of Object.entries(files)) s.addFile(name, text);
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(input); } catch (e) { return 'ERROR: ' + (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return 'OK';
};

/** [oracle case, index of the line (0-based), the line put in its place, native result] */
const ROWS: [string, number, string, string][] = [
  ["dump_formats", 12, "dump 1 nogroup atom 10 d.atom", "ERROR: Could not find dump group ID: nogroup"],
  ["w215cfg_dump", 19, "dump 1 nogroup cfg 20 s.*.cfg mass type xs ys zs id vx c_pe c_s[2]", "ERROR: Could not find dump group ID: nogroup"],
  ["dump_formats", 13, "dump 2 nogroup custom 10 d.custom id type x y z xu vx fx mass ix", "ERROR: Could not find dump group ID: nogroup"],
  ["w422dump_grid_tri", 11, "dump 1 nogroup grid 1 tri.dump c_pg:grid:data[*]", "ERROR: Could not find dump group ID: nogroup"],
  ["w423dump_grid_2d", 20, "dump 2 nogroup grid/vtk 40 v2d.*.vtr f_ag:grid:data[1] f_ag:grid:data[2] f_ag:grid:data[3]", "ERROR: Could not find dump group ID: nogroup"],
  ["w29dump_image", 11, "dump im nogroup image 1 img.*.ppm type type size 64 64 box no 0.02 view 90 0", "ERROR: Could not find dump group ID: nogroup"],
  ["w11local_bond", 12, "dump 1 nogroup local 1 w11local_bond.local index c_bl[*] c_bp[*]", "ERROR: Could not find dump group ID: nogroup"],
  ["w196vtk_xml", 14, "dump d nogroup vtk 10 dump*.vtp id type vx vy vz fx fy fz", "ERROR: Could not find dump group ID: nogroup"],
  ["dump_formats", 14, "dump 3 nogroup xyz 10 d.xyz", "ERROR: Could not find dump group ID: nogroup"],
  ["w17chunk_bin", 16, "compute spr all chunk/spread/atom cb c_nosuch[1] c_rs[2]", "ERROR: Compute ID nosuch for compute chunk/spread/atom does not exist"],
  ["w69fep_compute", 23, "compute FEP all fep 300.0 pair lj/cut/coul/long/soft lambda * * v_nosuch atom charge 1 v_dq atom charge 2 v_mdq", "ERROR: Variable name nosuch for compute fep does not exist"],
  ["w322ga_wildcard", 26, "compute gat all global/atom c_nosuch c_vcm[*]", "ERROR: Compute ID nosuch for compute global/atom index"],
  ["w17chunk_bin", 13, "compute rs all reduce/chunk cb sum c_nosuch[1] c_pa[2]", "ERROR: Compute ID nosuch for compute reduce/chunk does not exist"],
  ["w171px_slcsa", 20, "compute slcsa all slcsa/atom 8 4 w171px_slcsa_mean_descriptor.dat w171px_slcsa_lda_scalings.dat w171px_slcsa_lr_decision.dat w171px_slcsa_lr_bias.dat w171px_slcsa_mahalanobis.dat c_nosuch[1]", "ERROR: Compute ID nosuch for compute slcsa/atom does not exist"],
  ["w215cfg_dump", 19, "dump 1 all cfg 20 s.*.cfg mass type xs ys zs id vx c_nosuch c_s[2]", "ERROR: Could not find dump cfg compute ID: nosuch"],
  ["w31voronp_single", 15, "dump d all custom 1 w31voronp_single.dump id type x y z c_nosuch[1] c_v[2] c_v[3]", "ERROR: Could not find dump custom compute ID: nosuch"],
  ["w16numdiff_forces", 18, "dump d all custom 1 w16numdiff_forces.dump id fx fy fz f_nosuch[1] f_fd[2] f_fd[3]", "ERROR: Could not find dump custom fix ID: nosuch"],
  ["w471dump_thresh_last", 18, "dump d all custom 1 last.dump id type v_nosuch", "ERROR: Could not find dump custom variable name nosuch"],
  ["w11local_bond", 12, "dump 1 all local 1 w11local_bond.local index c_nosuch[*] c_bp[*]", "ERROR: Invalid attribute c_nosuch[*] in dump local command"],
  ["w142mx_tracker", 26, "dump 1 all local 100 w142mx_contact_history.dat f_nosuch[1] f_trackfix[2] f_trackfix[3] f_trackfix[4]", "ERROR: Could not find dump local fix ID nosuch"],
  ["w215cfg_dump", 20, "dump_modify 1 element Ar Kr colname c_nosuch epot format float %.10g", "ERROR: Incorrect dump_modify arguments: colname c_nosuch epot"],
  ["w471dump_thresh_last", 19, "dump_modify d thresh v_nosuch > LAST", "ERROR: Could not find dump modify variable name: nosuch"],
  ["w471dump_thresh_last", 29, "dump_modify e append yes thresh v_nosuch < LAST thresh type == 1", "ERROR: Could not find dump modify variable name: nosuch"],
  ["w9adapt_soft", 27, "fix 1 all adapt 1 pair soft a * * v_nosuch", "ERROR: Variable name nosuch for fix adapt does not exist"],
  ["w69fep_adapt", 23, "fix ADAPT all adapt/fep 20 pair lj/cut/coul/long/soft lambda * * v_nosuch atom charge 1 v_q1 atom charge 2 v_q2", "ERROR: Variable name nosuch for fix adapt/fep does not exist"],
  ["w13misc_controller", 21, "fix push all addforce v_nosuch 0.0 0.0", "ERROR: Variable nosuch for fix addforce does not exist"],
  ["w48fb_addtorque_pair", 24, "fix 4 pair addtorque/group 2.0 0.5 v_nosuch", "ERROR: Variable name for fix addtorque/group does not exist"],
  ["w229xf_corrlong_auto", 18, "fix JJ all ave/correlate/long 2 40 c_nosuch[4] c_p[5] type auto/upper file corrl_auto.dat", "ERROR: Compute ID nosuch for fix ave/correlate/long does not exist"],
  ["w13misc_controller", 23, "fix ctl all controller 10 1.0 0.5 0.2 0.1 v_nosuch 0.5 fc", "ERROR: Variable name nosuch for fix controller does not exist"],
  ["w3deform_wiggle_var", 23, "fix 2 all deform 1 x wiggle 0.2 0.08 z variable v_nosuch v_dv", "ERROR: Variable name nosuch for fix deform does not exist"],
  ["w66units_efield_atomvar_real", 24, "fix 2 all efield v_nosuch -0.01 0.03", "ERROR: Variable nosuch for x-field in fix efield does not exist"],
  ["w68elec_planar_ffield_ewald", 24, "fix conp bot electrode/conp 0 1.979 couple top v_nosuch symm on ffield yes", "ERROR: Variable 'nosuch' for fix electrode/conp does not exist"],
  ["w49hy_halt_min", 20, "fix h all halt 1 v_nosuch >= 21 error soft message no", "ERROR: Could not find fix halt variable name nosuch"],
  ["w266rg_custom_trunc", 13, "fix 1 all rigid custom v_nosuch", "ERROR: Variable nosuch for fix rigid custom does not exist"],
  ["w171px_saed_vtk_full", 18, "fix 2 all saed/vtk 2 2 4 c_nosuch file w171px_saed_vtk_full", "ERROR: Compute ID nosuch for fix saed/vtk does not exist or is of incorrect style"],
  ["w229xf_set", 21, "fix 2 all set 10 1 atom 1*108 type v_nosuch vx v_t", "ERROR: Variable name v_nosuch for set command does not exist"],
  ["w2fset_setforce", 29, "fix 2 lower setforce 0.0 NULL v_nosuch", "ERROR: Variable nosuch for fix setforce does not exist"],
  ["w457dyngroup_smd_setvel", 64, "fix veltop_fix top smd/setvel 0 v_nosuch 0", "ERROR: Variable name for fix setvelocity does not exist"],
  ["w269ss_history_cont", 19, "fix hist all store/state 0 vx c_nosuch history 5 4 0", "ERROR: Compute ID nosuch for fix store/state does not exist"],
  ["w41sm_wallgran_var", 19, "fix 3 all wall/gran hooke/history 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 v_nosuch", "ERROR: Variable nosuch for fix wall/gran wall position does not exist"],
  ["w38heat_pour", 69, "fix 5 all wall/gran/region granular hertz/material 1e5 0.1 0.3 tangential mindlin NULL 1.0 0.5 damping tsuji heat area 5.0 region bottom_wall temperature v_nosuch", "ERROR: Variable nosuch for fix wall/gran does not exist"],
  ["w48sp2_langevin0", 19, "write_dump all custom w48sp2_langevin0.dump id c_nosuch[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6] c_p[7] c_p[8] c_p[9] c_p[10] modify format float %.15g", "ERROR: Could not find dump custom compute ID: nosuch"],
  ["w229xf_pair", 22, "write_dump all custom w229xf_pair.dump id f_nosuch[1] f_pr[2] modify format float %.15g sort id", "ERROR: Could not find dump custom fix ID: nosuch"],
  ["w59bpm_peri_pmb", 27, "bond_coeff 99 pmb 1.6863e13 0.0015001 0.0005 0.25", "ERROR: Numeric index 99 is out of bounds (1-1)"],
];

/** [oracle case, line index, line, result of a native build with the package, the package] */
const PACKAGE_ROWS: [string, number, string, string, string][] = [
  ["w276cp_compute_pace_ace", 17, "compute pace nogroup pace Cu-PBE-core-rep.ace 0 0", "ERROR: Group ID nogroup requested by compute pace does not exist", "ML-PACE"],
  ["w278ap_atom_weight", 21, "fix w nogroup atom_weight/apip 2 1e-05 0.0004 3e-06 0 grp no_rescale", "ERROR: Could not find fix group ID nogroup", "APIP"],
  ["w279ap_thermostat_shuffle", 16, "fix lambda nogroup lambda/apip 0.5 1.5 time_averaged_zone 2.0 3.0 3 3 store_atomic_stats", "ERROR: Could not find fix group ID nogroup", "APIP"],
  ["w280ap_la_csp_static", 16, "fix la nogroup lambda/la/csp/apip 0.7 2.0 3.0 5.0 fcc store_peratom 5", "ERROR: Could not find fix group ID nogroup", "APIP"],
  ["w279ap_thermostat_shuffle", 17, "fix lambda_thermostat nogroup lambda_thermostat/apip N_rescaling 3 seed 1234", "ERROR: Could not find fix group ID nogroup", "APIP"],
  ["w286mf_nvt_ramp", 14, "fix step all nvt/manifold/rattle 1e-8 20 sphere v_nosuch temp 1.0 1.2 0.1 tchain 4", "ERROR: Variable name for fix nve/manifold/rattle does not exist", "MANIFOLD"],
];

/*
 * No native build here has the PTM, LATBOLTZ or QTB package; the expected messages follow from the C++ source at b332f28:
 * a compute's group is looked up by Group::get_bitmask_by_id ("Group ID {} requested by {} does not exist", src/group.cpp)
 * and a fix's by Modify::add_fix ("Could not find fix group ID {}", src/modify.cpp); fix qbmsst and fix qtb refuse an
 * unknown word with "Illegal fix qbmsst command" and "Illegal fix qtb command".
 */
const SOURCE_ROWS: [string, number, string, string][] = [
  ["w201pt_rmsd_group2", 18, "compute ptm nogroup ptm/atom fcc-hcp-ico 0.05 lower", "ERROR: Group ID nogroup requested by compute ptm/atom does not exist"],
  ["w132lb_momentum", 22, "fix 4 nogroup lb/momentum 50 linear 1 1 1", "ERROR: Could not find fix group ID nogroup"],
  ["w181px_lb_pits_sw", 21, "fix 2 nogroup lb/viscous", "ERROR: Could not find fix group ID nogroup"],
  ["w99phq_qbmsst_lj", 16, "fix shock all qbmsst z 1.5 q 10.0 tscale 0.1 damp 1.0 f_nosuch 4.0 N_f 20 seed 35082 eta 1.0 beta 5 T_init 0.8", "ERROR: Illegal fix qbmsst command"],
  ["w99phq_qtb_lj", 16, "fix 2 all qtb temp 0.8 damp 0.5 seed 35082 f_nosuch 2.0 N_f 50", "ERROR: Illegal fix qtb command"],
];

describe('a missing group, compute, fix or variable, or a type past the type count, as native LAMMPS', () => {
  it.each(ROWS)('%s line %i: %s -> %s', async (name, k, line, expected) => {
    const [input, files] = withLine(name, k, line);
    expect(await outcome(input, files)).toBe(expected);
  });

  it.each(PACKAGE_ROWS)('%s line %i: %s -> %s (%s build)', async (name, k, line, expected) => {
    const [input, files] = withLine(name, k, line);
    expect(await outcome(input, files)).toBe(expected);
  });

  it.each(SOURCE_ROWS)('%s line %i: %s -> %s (from the C++ source)', async (name, k, line, expected) => {
    const [input, files] = withLine(name, k, line);
    expect(await outcome(input, files)).toBe(expected);
  });
});

/*
 * The dump command checks the dump ID, the group and the frequency before the style, as native LAMMPS does
 * (src/output.cpp Output::add_dump at b332f28). Measured with native LAMMPS: the four lines below give these messages
 * (a v_name frequency is a dump_modify every value, not a dump command one).
 */
describe('the dump command check order, as native LAMMPS', () => {
  const head = 'units lj\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\nvariable n equal 10\n';
  it.each([
    ['dump 1 all atom v_n d.atom', "ERROR: Expected integer parameter instead of 'v_n' in input script or data file"],
    ['dump 1 all atom 0 d.atom', 'ERROR: Invalid dump frequency 0'],
    ['dump 1 all bogusstyle 10 d.x', "ERROR: Unrecognized dump style 'bogusstyle'"],
    ['dump 1 nogroup bogusstyle 10 d.x', 'ERROR: Could not find dump group ID: nogroup'],
  ])('%s', async (line, expected) => {
    const got = await outcome(`${head}${line}\nrun 0\n`);
    expect(got.startsWith(expected)).toBe(true);
  });
});
