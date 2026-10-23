import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * A line of an oracle case with one word replaced: a yes/no value by "maybe", the ID after a region keyword by
 * "noregion", or an input file name by "nofile.txt", refused with native LAMMPS's message (utils::logical, the style's
 * missing-region message, or the style's message for a file it cannot open). Each row takes the case up to the line,
 * puts the given line in its place and runs 0 steps. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28;
 * PACKAGE_ROWS with a b332f28 build that has the package).
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
  ["w16mliap_compute", 18, "compute g all mliap model linear descriptor sna nofile.txt gradgradflag 1", "ERROR: Cannot open SNAP parameter file nofile.txt: No such file or directory"],
  ["w155mx_podatom_ta", 20, "compute ld all pod/atom nofile.txt w128pod_Ta_coeff.pod Ta", "ERROR: Cannot open POD coefficient file nofile.txt: "],
  ["w177px_pod_example", 19, "compute gdd all pod/global nofile.txt w128pod_Ta_coeff.pod Ta Ta", "ERROR: Cannot open POD coefficient file nofile.txt: "],
  ["w177px_pod_example", 21, "compute ldd all pod/local nofile.txt w128pod_Ta_coeff.pod Ta Ta", "ERROR: Cannot open POD coefficient file nofile.txt: "],
  ["w171px_podd_ta", 17, "compute dd all podd/atom nofile.txt w128pod_Ta_coeff.pod Ta", "ERROR: Cannot open POD coefficient file nofile.txt: "],
  ["w171px_slcsa", 20, "compute slcsa all slcsa/atom 8 4 nofile.txt w171px_slcsa_lda_scalings.dat w171px_slcsa_lr_decision.dat w171px_slcsa_lr_bias.dat w171px_slcsa_mahalanobis.dat c_bnnn[1]", "ERROR: cannot open database mean descriptor file potential file nofile.txt: No such file or directory"],
  ["w35mesh_tri", 7, "create_atoms 1 mesh nofile.txt meshmode bisect 1.0 units box", "ERROR: Cannot open STL file nofile.txt: No such file or directory"],
  ["w19trigen_general", 13, "dump_modify 1 triclinic/general maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w196vtk_modify", 18, "dump_modify d element Ar Kr label cells double yes pad 3 region noregion thresh type == 2", "ERROR: Dump_modify region noregion does not exist"],
  ["w99phq_dynmat_silicon", 16, "dynamical_matrix all eskm 0.000001 file dynmat.dat binary maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w9adapt_charge", 28, "fix 1 all adapt 1 atom charge v_qf scale maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w384adaptfep_diameter", 19, "fix 1 all adapt/fep 1 atom diameter 1 v_d reset maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w96rheo_oxidation", 71, "fix 10 floor add/heat constant 0 overwrite maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w426avegrid_discard", 17, "fix ak all ave/grid 1 1 1 2 3 3 density/number vx discard maybe", "ERROR: Illegal fix ave/grid discard argument: maybe"],
  ["w94react_nylon_unstab", 24, "fix myrxns all bond/react stabilization no react rxn1 all 1 0.0 2.9 mol1 mol2 nofile.txt react rxn2 all 1 0.0 5.0 mol3 mol4 w94react_nylon_stp2.map rescale_charges yes", "ERROR: Fix bond/react: Cannot open map file nofile.txt"],
  ["w94react_nylon_unstab", 24, "fix myrxns all bond/react stabilization maybe react rxn1 all 1 0.0 2.9 mol1 mol2 w94react_nylon_stp1.map react rxn2 all 1 0.0 5.0 mol3 mol4 w94react_nylon_stp2.map rescale_charges yes", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w229xf_deform_pressure_couple", 15, "fix 2 all deform/pressure 10 x pressure 0.5 0.05 couple xyz max/rate 0.02 normalize/pressure maybe units box", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w93meso_edpd_region", 38, "fix upper all edpd/source region noregion 0.011", "ERROR: Region noregion for fix edpd/source does not exist"],
  ["w68elec_madelung_cg", 16, "fix conp bot electrode/conp 0 2 couple top 1 symm maybe algo cg 1e-6", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w68elec_conq_cg", 19, "fix q bot electrode/conq -0.2 2 couple top 0.2 symm maybe algo cg 1e-6", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w47pkg_elstop_cu", 18, "fix fel all electron/stopping 10.0 nofile.txt", "ERROR: cannot open electron stopping data table potential file nofile.txt: No such file or directory"],
  ["w230re_eos_table", 23, "fix 3 all eos/table linear nofile.txt 200 EOS_TABLE", "ERROR: cannot open eos/table table potential file nofile.txt: No such file or directory"],
  ["w162rx_table_lookup", 44, "fix 3 all eos/table/rx linear nofile.txt 201 KEYWORD w160rx_thermo.dpdrx", "ERROR: cannot open eos/table/rx table potential file nofile.txt: No such file or directory"],
  ["w11restrain_evaporate", 32, "fix EV all2 evaporate 4 2 ev 20243 molecule maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w133mf_flow_gauss_kw", 29, "fix GD left flow/gauss 1 1 0 energy maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w198wr_gcmc_region", 14, "fix g all gcmc 1 4 0 1 81723 1.3 -2.0 0.3 region noregion", "ERROR: Region noregion for fix gcmc does not exist"],
  ["w108xf_gle_lj_eq", 15, "fix 1 all gle 6 1.2 1.4 31415 nofile.txt", "ERROR: cannot open fix gle A matrix potential file nofile.txt: No such file or directory"],
  ["w199vo_mc_rates", 25, "fix mc all neighbor/swap 3 4 777 1.0 1.2 v diff 2 region right rates 1.0 0.7 ke maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w199vo_mc_rates", 25, "fix mc all neighbor/swap 3 4 777 1.0 1.2 v diff 2 region noregion rates 1.0 0.7 ke yes", "ERROR: Region ID noregion for fix neighbor/swap does not exist"],
  ["w39nemd_n0_nocheck", 22, "fix 1 all nvt/sllod temp 1.0 1.0 0.1 kick maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w230re_eff_sllod_peculiar", 15, "fix 2 all nvt/sllod/eff temp 300.0 300.0 10.0 peculiar maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w236or_bcc", 14, "fix or all orient/bcc 0 0 2.88 0.02 0.2 0.8 nofile.txt w236or_bcc_grain1.ori", "ERROR: cannot open fix orient/bcc potential file nofile.txt: No such file or directory"],
  ["w133mf_orient_eco", 15, "fix eco all orient/eco 0.08 0.25 3.6 nofile.txt", "ERROR: cannot open fix orient/eco potential file nofile.txt: No such file or directory"],
  ["w236or_fcc", 14, "fix or all orient/fcc 0 1 3.6 0.01 0.25 0.75 nofile.txt w236or_fcc_grain2.ori", "ERROR: cannot open fix orient/fcc potential file nofile.txt: No such file or directory"],
  ["w155mx_pafi_od", 33, "fix hp all pafi pa 500.0 0.01 434 overdamped maybe com no", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w99phq_cu", 22, "fix 2 all phonon 10 100 1000 nofile.txt w99phq_cu nasr 20", "ERROR: Cannot open input map file nofile.txt: No such file or directory"],
  ["w36qeq_dyn_loose", 23, "fix q all qeq/dynamic 1 8 1.0e-3 200 nofile.txt", "ERROR: Cannot open fix qeq parameter file nofile.txt: No such file or directory"],
  ["w36qeq_fire_tight", 23, "fix q all qeq/fire 1 8 1.0e-5 200 nofile.txt", "ERROR: Cannot open fix qeq parameter file nofile.txt: No such file or directory"],
  ["w32qeq_point2", 16, "fix q all qeq/point 1 20 1.0e-14 200 nofile.txt", "ERROR: Cannot open fix qeq parameter file nofile.txt: No such file or directory"],
  ["w55reax_qeq_rdx_tight", 10, "fix 2 all qeq/reax 1 0.0 10.0 1e-12 nofile.txt", "ERROR: Unknown fix qeq/reaxff keyword nofile.txt"],
  ["w65reaxq_qeqr_tight", 10, "fix 1 all qeq/rel/reaxff 1 0.0 10.0 1e-12 reaxff nofile.txt", "ERROR: Fix qtpie/reaxff: could not open gauss file: 'nofile.txt'"],
  ["w32qeq_shield4", 22, "fix q all qeq/shielded 1 8 1.0e-14 200 nofile.txt", "ERROR: Cannot open fix qeq parameter file nofile.txt: No such file or directory"],
  ["w40qslat_qfile", 27, "fix 1 all qeq/slater 1 12.0 1.0e-12 500 nofile.txt", "ERROR: Cannot open fix qeq parameter file nofile.txt: No such file or directory"],
  ["w65reaxq_qtpie_tight", 10, "fix 1 all qtpie/reaxff 1 0.0 10.0 1e-12 reaxff nofile.txt", "ERROR: Fix qtpie/reaxff: could not open gauss file: 'nofile.txt'"],
  ["w6rigid_infile", 11, "fix 1 all rigid custom v_bid infile nofile.txt", "ERROR: Cannot open fix rigid infile nofile.txt: No such file or directory"],
  ["rigid_force_torque", 48, "fix 1 all rigid molecule force 1*9 maybe on on torque 10* on off off", "ERROR: Illegal fix rigid command"],
  ["w162rx_table_lookup", 12, "fix 4 all rx nofile.txt none dense lammps_rk4 1", "ERROR: cannot open rx potential file nofile.txt: No such file or directory"],
  ["w166px_settorque_wire", 56, "fix 4 loaded settorque/atom NULL v_ty v_tzat region noregion", "ERROR: Region noregion for fix settorque/atom does not exist"],
  ["w41sm_shake_store", 46, "fix 2 all shake 1.0e-12 200 0 b 1 2 3 4 a 1 store maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w230re_smd_wall_linear", 40, "fix stl_surface_fix all smd/wall_surface nofile.txt 2 65535", "ERROR: Error reading triangles from file nofile.txt: Cannot open STL file nofile.txt: No such file or directory"],
  ["w79srd_tstat", 29, "fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip tstat maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w165mc_store_com", 17, "fix c all store/state 0 xu yu zu com maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w261gs_screw_press", 12, "fix 1 all surface/local input stl 3 2 nofile.txt flat 10.0", "ERROR: Cannot open STL file nofile.txt: No such file or directory"],
  ["w14ttm_stop", 19, "fix 2 all ttm 4242 1.0 1.0 1.0 20.0 3.0 2.5 2 2 2 infile nofile.txt", "ERROR: cannot open electron temperature grid potential file nofile.txt: No such file or directory"],
  ["w41ttm_cas_cetab", 20, "fix 2 all ttm/cascade 77 3.0 2.0 0.5 20.0 0.0 2.0 2 2 2 infile nofile.txt cetab w41ttm_ce.tab offset 0.015", "ERROR: Cannot open grid file: nofile.txt: No such file or directory"],
  ["w421dump_grid_ttm", 20, "fix 2 all ttm/grid 777 0.001 1.0 0.01 20.0 0.0 2.0 2 2 2 infile nofile.txt", "ERROR: Cannot open grid file: nofile.txt: No such file or directory"],
  ["w43tm_si", 18, "fix twotemp all ttm/mod 1354684 nofile.txt 4 4 4 set 1000.0", "ERROR: cannot open ttm/mod parameter potential file nofile.txt: No such file or directory"],
  ["w41ttm_thermal4", 21, "fix 2 all ttm/thermal 99 nofile.txt 4 4 4 infile w41ttm_thermal4.init source 0.02", "ERROR: cannot open electron property grid potential file nofile.txt: No such file or directory"],
  ["w13gran_region_hertz", 24, "fix 3 all wall/gran/region hertz/history 2000.0 1000.0 0.4 0.3 0.6 1 region noregion", "ERROR: Region noregion for fix wall/gran/region does not exist"],
  ["w16walltable_linear", 14, "fix 2 all wall/table linear 201 zlo EDGE nofile.txt HARMONIC 2.5 zhi EDGE w16walltable.dat LJ126 2.5", "ERROR: cannot open fix wall/table table potential file nofile.txt: No such file or directory"],
  ["w30widom_region", 16, "fix w all widom 1 2 2 12345 1.0 region noregion", "ERROR: Region noregion for fix widom does not exist"],
  ["w51km_disp_noeffect", 10, "kspace_modify gewald 0.4 gewald/disp 0.4 collective maybe fftbench yes overlap no minorder 2 cutoff/adjust yes", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w130pr_dpd_coul_slater", 19, "pair_coeff 1 1 78 4.5 maybe 1", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w57oxdna2_dh_half", 19, "pair_coeff * * oxdna2/dh 0.1 0.2 0.815 half_charged_ends maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w47spin_moving2", 14, "pair_coeff * * exchange 4.0 0.02726 0.2171 1.841 offset maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w48sp2_biquad", 21, "pair_coeff * * biquadratic 4.0 0.05 0.03 1.48 0.07 0.02 1.9 offset maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w16disp_lj", 5, "pair_style lj/long/coul/long long maybe 2.5", "ERROR: Illegal pair_style lj/long/coul/long command"],
  ["w143cf_reset_sort", 19, "reset_atoms id sort maybe", "ERROR: Expected boolean parameter instead of 'maybe' in input script or data file"],
  ["w49hy_halt_min", 23, "timer timeout maybe", "ERROR: Illegal timeout time: timeout"],
];

/** [oracle case, line index, line, result of a native build with the package, the package] */
const PACKAGE_ROWS: [string, number, string, string, string][] = [
  ["w189sq_al2o3", 21, "pair_coeff * * nofile.txt O Al", "ERROR: Cannot open SMTBQ potential file nofile.txt", "SMTBQ"],
];

describe('a bad yes/no value, region ID or file name, as native LAMMPS', () => {
  it.each(ROWS)('%s line %i: %s -> %s', async (name, k, line, expected) => {
    const [input, files] = withLine(name, k, line);
    expect(await outcome(input, files)).toBe(expected);
  });

  it.each(PACKAGE_ROWS)('%s line %i: %s -> %s (%s build)', async (name, k, line, expected) => {
    const [input, files] = withLine(name, k, line);
    expect(await outcome(input, files)).toBe(expected);
  });
});

/*
 * timer, as native LAMMPS reads it (src/timer.cpp Timer::modify_params at b332f28; the messages name the keyword, not
 * the value): a timeout of off or unlimited is accepted, a bad time or every value is refused natively, and a real
 * timeout is refused by the browser engine (it needs the wall clock).
 */
describe('the timer command, as native LAMMPS', () => {
  it.each([
    ['timer timeout maybe', 'ERROR: Illegal timeout time: timeout'],
    ['timer every 0', 'ERROR: Illegal timer every frequency: every'],
    ['timer timeout', 'ERROR: Illegal timer timeout command: missing argument(s)'],
    ['timer frob', 'ERROR: Unknown timer keyword frob'],
    ['timer timeout off', 'OK'],
    ['timer every 10 full sync', 'OK'],
  ])('%s', async (line, expected) => {
    expect(await outcome(`units lj\n${line}\n`)).toBe(expected);
  });
});
