import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * The pair style's coefficients and charges checked at init as native LAMMPS does (src/pair.cpp Pair::init and each
 * style's init_style and init_one, src/force.cpp Force::init at b332f28): "All pair coeffs are not set. Status:" when
 * pair_coeff never ran or an I,I pair is unset, each style's init_one wording for an I,J pair it cannot mix, and the
 * charge requirement of the kspace and pair styles. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28).
 */

const ORACLE = join(__dirname, 'oracle');
const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const EXAMPLES = join(__dirname, '..', 'third_party', 'lammps', 'examples');
const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^# ${key}:(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter((w) => w !== '') : [];
};
const readInput = (path: string): string => readFileSync(path, path.endsWith('.gz') ? 'latin1' : 'utf8');

const outcome = async (input: string, files: Record<string, string> = {}): Promise<string> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  for (const [name, text] of Object.entries(files)) s.addFile(name, text);
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(input); } catch (e) { return 'ERROR: ' + (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return 'OK';
};

/** An oracle case's input up to its pair_style line (no pair_coeff), then run 0, with the files it reads. */
const withoutCoeffs = (name: string, style: string): [string, Record<string, string>] => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const files: Record<string, string> = {};
  for (const f of directive(text, 'oracle-inputs')) files[f] = readInput(join(ORACLE, f));
  for (const f of directive(text, 'oracle-potentials')) files[f] = readFileSync(join(POTENTIALS, f), 'utf8');
  for (const f of directive(text, 'oracle-examples')) files[f.split('/').pop()!] = readInput(join(EXAMPLES, f));
  const lines = text.split('\n');
  const k = lines.findIndex((l) => {
    const w = l.trim().split(/\s+/);
    return w[0] === 'pair_style' && w[1] === style;
  });
  return [lines.slice(0, k + 1).filter((l) => !l.startsWith('# oracle')).join('\n') + '\nrun 0\n', files];
};

/** [oracle case, its pair style, native's result] */
const NO_COEFFS: [string, string, string][] = [
  ["w222me_adp_ni", "adp", "ERROR: All pair coeffs are not set. Status:"],
  ["w52airebo_airebo_00", "airebo", "ERROR: All pair coeffs are not set. Status:"],
  ["w147ab_airebo_bc", "airebo/bc", "ERROR: All pair coeffs are not set. Status:"],
  ["w52airebo_morse_11", "airebo/morse", "ERROR: All pair coeffs are not set. Status:"],
  ["w119amoeba_mpole_rspace", "amoeba", "ERROR: All pair coeffs are not set. Status:"],
  ["w10atm_pure", "atm", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_beck", "beck", "ERROR: All pair coeffs are not set. Status:"],
  ["w48bd_free", "body/nparticle", "ERROR: All pair coeffs are not set. Status:"],
  ["w50bc_static", "body/rounded/polygon", "ERROR: All pair coeffs are not set. Status:"],
  ["w126body_cubes_wall", "body/rounded/polyhedron", "ERROR: All pair coeffs are not set. Status:"],
  ["w233bo_alcu", "bop", "ERROR: All pair coeffs are not set. Status:"],
  ["w1buck_born", "born", "ERROR: All pair coeffs are not set. Status:"],
  ["w18borncoul_dsf", "born/coul/dsf", "ERROR: All pair coeffs are not set. Status:"],
  ["w21borncs_dsf", "born/coul/dsf/cs", "ERROR: All pair coeffs are not set. Status:"],
  ["w3coullong_born", "born/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w9pair_born_cs", "born/coul/long/cs", "ERROR: All pair coeffs are not set. Status:"],
  ["w209ms_born", "born/coul/msm", "ERROR: All pair coeffs are not set. Status:"],
  ["w18borncoul_wolf", "born/coul/wolf", "ERROR: All pair coeffs are not set. Status:"],
  ["w21borncs_wolf", "born/coul/wolf/cs", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_born_gauss", "born/gauss", "ERROR: All pair coeffs are not set. Status:"],
  ["w59bpm_peri_pmb", "bpm/peri", "ERROR: All pair coeffs are not set. Status:"],
  ["w51bpm_pair_anh", "bpm/spring", "ERROR: All pair coeffs are not set. Status:"],
  ["w231co_brownian_nohi", "brownian", "ERROR: All pair coeffs are not set. Status:"],
  ["w231co_brownian_poly", "brownian/poly", "ERROR: All pair coeffs are not set. Status:"],
  ["w1buck_buck", "buck", "ERROR: All pair coeffs are not set. Status:"],
  ["w1buck_buck_coul_cut", "buck/coul/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w3coullong_buck", "buck/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w9pair_buck_cs", "buck/coul/long/cs", "ERROR: All pair coeffs are not set. Status:"],
  ["w209ms_buck", "buck/coul/msm", "ERROR: All pair coeffs are not set. Status:"],
  ["w226cs_buck_ewald_t0", "buck/long/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_buck_mdf", "buck/mdf", "ERROR: All pair coeffs are not set. Status:"],
  ["w50bd_cosbuck6d", "buck6d/coul/gauss/dsf", "ERROR: All pair coeffs are not set. Status:"],
  ["w50bd_cosbuck6d_long", "buck6d/coul/gauss/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w5colloid_mix", "colloid", "ERROR: All pair coeffs are not set. Status:"],
  ["w58comb_si0", "comb", "ERROR: All pair coeffs are not set. Status:"],
  ["w158cb_cu", "comb3", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_cosine_squared", "cosine/squared", "ERROR: All pair coeffs are not set. Status:"],
  ["w32qeq_point2", "coul/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w226cs_diel_cutcoul", "coul/cut/dielectric", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_coul_cut_global", "coul/cut/global", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_coulcutsoft", "coul/cut/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_gapsys_coul", "coul/cut/soft/gapsys", "ERROR: All pair coeffs are not set. Status:"],
  ["w1coul_debye", "coul/debye", "ERROR: All pair coeffs are not set. Status:"],
  ["w49pb_coul_diel_dimer", "coul/diel", "ERROR: All pair coeffs are not set. Status:"],
  ["w1coul_special_dsf", "coul/dsf", "ERROR: Run command before simulation box is defined"],
  ["w244es_ortho", "coul/esp", "ERROR: All pair coeffs are not set. Status:"],
  ["w117cf_pppm_mesh8_real", "coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w9pair_coul_cs_short", "coul/long/cs", "ERROR: All pair coeffs are not set. Status:"],
  ["w226cs_diel_coullong_t12", "coul/long/dielectric", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_coullongsoft", "coul/long/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_coul_slater_cut", "coul/slater/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w130pr_coulslater_ewald", "coul/slater/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w43st_dsf_alone", "coul/streitz", "ERROR: All pair coeffs are not set. Status:"],
  ["w1coul_special_wolf", "coul/wolf", "ERROR: Run command before simulation box is defined"],
  ["w226cs_wolf", "coul/wolf/cs", "ERROR: All pair coeffs are not set. Status:"],
  ["w250d3_zerom", "dispersion/d3", "ERROR: All pair coeffs are not set. Status:"],
  ["w77dpd_noise_dpd3d_nosort", "dpd", "ERROR: All pair coeffs are not set. Status:"],
  ["w130pr_dpd_coul_slater", "dpd/coul/slater/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w77dpd_noise_ext3d", "dpd/ext", "ERROR: All pair coeffs are not set. Status:"],
  ["w319dpd_ext_tstat_ramp", "dpd/ext/tstat", "ERROR: All pair coeffs are not set. Status:"],
  ["w95dreact_dpdvv_1000", "dpd/fdt", "ERROR: All pair coeffs are not set. Status:"],
  ["w95dreact_shardlow_dpde_1000", "dpd/fdt/energy", "ERROR: All pair coeffs are not set. Status:"],
  ["w77dpd_noise_tstat_spce", "dpd/tstat", "ERROR: All pair coeffs are not set. Status:"],
  ["w235mx_dsmc_gas", "dsmc", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_shift_eam", "eam", "ERROR: All pair coeffs are not set. Status:"],
  ["w47pkg_dtreset_metal", "eam/alloy", "ERROR: All pair coeffs are not set. Status:"],
  ["w222me_eamcd_fecr", "eam/cd", "ERROR: All pair coeffs are not set. Status:"],
  ["w222me_eamcdold_fecr", "eam/cd/old", "ERROR: All pair coeffs are not set. Status:"],
  ["w204sg_fs_vcsgc", "eam/fs", "ERROR: All pair coeffs are not set. Status:"],
  ["w222me_eamhe_pdhhe", "eam/he", "ERROR: All pair coeffs are not set. Status:"],
  ["w99phq_cu", "eam/opt", "ERROR: All pair coeffs are not set. Status:"],
  ["w48pb_edip_diamond", "edip", "ERROR: All pair coeffs are not set. Status:"],
  ["w48pb_edip_sic_example", "edip/multi", "ERROR: All pair coeffs are not set. Status:"],
  ["w449dyngroup_edpd", "edpd", "ERROR: All pair coeffs are not set. Status:"],
  ["w367eff_nph_couple_xy", "eff/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w14eim_one", "eim", "ERROR: All pair coeffs are not set. Status:"],
  ["w145pr_pair_extep", "extep", "ERROR: All pair coeffs are not set. Status:"],
  ["w1misc_gauss", "gauss", "ERROR: All pair coeffs are not set. Status:"],
  ["w49pb_gauss_cut_dimer", "gauss/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w431rigidnh_ellipsoid", "gayberne", "ERROR: All pair coeffs are not set. Status:"],
  ["w4gran_hertz_history", "gran/hertz/history", "ERROR: All pair coeffs are not set. Status:"],
  ["w7pour_fewer", "gran/hooke", "ERROR: All pair coeffs are not set. Status:"],
  ["w36pour2d", "gran/hooke/history", "ERROR: All pair coeffs are not set. Status:"],
  ["w15gran_mdr_pair", "granular", "ERROR: All pair coeffs are not set. Status:"],
  ["w80superell_ellipsoid_gran", "granular/superellipsoid", "ERROR: All pair coeffs are not set. Status:"],
  ["w223mb_gw", "gw", "ERROR: All pair coeffs are not set. Status:"],
  ["w223mb_gw_zbl", "gw/zbl", "ERROR: All pair coeffs are not set. Status:"],
  ["w17harmcut", "harmonic/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w119amoeba_hippo_hexamer", "hippo", "ERROR: All pair coeffs are not set. Status:"],
  ["w234lc_diamond", "lcbop", "ERROR: All pair coeffs are not set. Status:"],
  ["w112ldd_molecule_ld", "ldd", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_lennard_mdf", "lennard/mdf", "ERROR: All pair coeffs are not set. Status:"],
  ["w16lepton_shift", "lepton", "ERROR: All pair coeffs are not set. Status:"],
  ["w16lepton_coul", "lepton/coul", "ERROR: All pair coeffs are not set. Status:"],
  ["w16lepton_sphere", "lepton/sphere", "ERROR: All pair coeffs are not set. Status:"],
  ["w105cf_erotate_line", "line/lj", "ERROR: All pair coeffs are not set. Status:"],
  ["w76dih_fine_charmm", "lj/charmm/coul/charmm", "ERROR: Run command before simulation box is defined"],
  ["w3charmm_implicit", "lj/charmm/coul/charmm/implicit", "ERROR: All pair coeffs are not set. Status:"],
  ["w50il_peptide_run0", "lj/charmm/coul/long", "ERROR: Run command before simulation box is defined"],
  ["w69fep_charmmsoft", "lj/charmm/coul/long/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w209ms_charmm_water", "lj/charmm/coul/msm", "ERROR: All pair coeffs are not set. Status:"],
  ["w10charmmfsw_fsh", "lj/charmmfsw/coul/charmmfsh", "ERROR: All pair coeffs are not set. Status:"],
  ["w10charmmfsw_long", "lj/charmmfsw/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w1ljx_class2", "lj/class2", "ERROR: All pair coeffs are not set. Status:"],
  ["w51cs_c2cut_nylon", "lj/class2/coul/cut", "ERROR: Run command before simulation box is defined"],
  ["w69fep_class2cutsoft", "lj/class2/coul/cut/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w51cs_c2long_nylon", "lj/class2/coul/long", "ERROR: Run command before simulation box is defined"],
  ["w226cs_class2", "lj/class2/coul/long/cs", "ERROR: All pair coeffs are not set. Status:"],
  ["w17soft_class2", "lj/class2/coul/long/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_class2soft", "lj/class2/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_shift_lj_cubic", "lj/cubic", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_nofdotr", "lj/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_amber_coul", "lj/cut/coul/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w243de_efield_cutcoul", "lj/cut/coul/cut/dielectric", "ERROR: All pair coeffs are not set. Status:"],
  ["w17soft_coul", "lj/cut/coul/cut/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w1lcoul_debye", "lj/cut/coul/debye", "ERROR: All pair coeffs are not set. Status:"],
  ["w226cs_diel_debye", "lj/cut/coul/debye/dielectric", "ERROR: All pair coeffs are not set. Status:"],
  ["w1lcoul_special_dsf", "lj/cut/coul/dsf", "ERROR: Run command before simulation box is defined"],
  ["w244es_ljfluid", "lj/cut/coul/esp", "ERROR: All pair coeffs are not set. Status:"],
  ["w75pppmslab_madelung_lowacc", "lj/cut/coul/long", "ERROR: Run command before simulation box is defined"],
  ["w226cs_ljcoul_t12", "lj/cut/coul/long/cs", "ERROR: All pair coeffs are not set. Status:"],
  ["w243de_efield_ljlong_pppm", "lj/cut/coul/long/dielectric", "ERROR: All pair coeffs are not set. Status:"],
  ["w54accel_charged", "lj/cut/coul/long/omp", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_adapt", "lj/cut/coul/long/soft", "ERROR: Run command before simulation box is defined"],
  ["w209ms_water_flex", "lj/cut/coul/msm", "ERROR: All pair coeffs are not set. Status:"],
  ["w243de_efield_ljcoulmsm", "lj/cut/coul/msm/dielectric", "ERROR: All pair coeffs are not set. Status:"],
  ["w1lcoul_special_wolf", "lj/cut/coul/wolf", "ERROR: Run command before simulation box is defined"],
  ["w41sm_dipole_mol", "lj/cut/dipole/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w44dl_static_auto", "lj/cut/dipole/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w54accel_melt", "lj/cut/omp", "ERROR: All pair coeffs are not set. Status:"],
  ["w17soft_tail", "lj/cut/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_gapsys_lj", "lj/cut/soft/gapsys", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_ljcutsphere", "lj/cut/sphere", "ERROR: All pair coeffs are not set. Status:"],
  ["w72drude_toluene_lang", "lj/cut/thole/long", "ERROR: Run command before simulation box is defined"],
  ["w12tip4p_cut", "lj/cut/tip4p/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w117cf_tip4p_pppm_water", "lj/cut/tip4p/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_tip4plj", "lj/cut/tip4p/long/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_expand_shift01", "lj/expand", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_ljexpandcoullong_special", "lj/expand/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_ljexpandsphere", "lj/expand/sphere", "ERROR: All pair coeffs are not set. Status:"],
  ["w1ljx_gromacs", "lj/gromacs", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_gromacs", "lj/gromacs/coul/gromacs", "ERROR: All pair coeffs are not set. Status:"],
  ["w16disp_lj", "lj/long/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w226cs_diel_ljlong", "lj/long/coul/long/dielectric", "ERROR: All pair coeffs are not set. Status:"],
  ["w211ed_dipole_only", "lj/long/dipole/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w242tp_ljcut_pppm_tip4p", "lj/long/tip4p/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_lj_mdf", "lj/mdf", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_lj_pirani", "lj/pirani", "ERROR: All pair coeffs are not set. Status:"],
  ["w6relres_mixed", "lj/relres", "ERROR: All pair coeffs are not set. Status:"],
  ["w51cs_sdk_angle", "lj/sdk", "ERROR: All pair coeffs are not set. Status:"],
  ["w51cs_sdklong_dimer", "lj/sdk/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w9pair_ljsf", "lj/sf", "ERROR: All pair coeffs are not set. Status:"],
  ["w18dipole_sf", "lj/sf/dipole/sf", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_tail_lj_smooth", "lj/smooth", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_shift_lj_smooth_linear", "lj/smooth/linear", "ERROR: All pair coeffs are not set. Status:"],
  ["w51cs_spica_angle", "lj/spica", "ERROR: All pair coeffs are not set. Status:"],
  ["w51cs_spicalong_run", "lj/spica/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w209ms_spica", "lj/spica/coul/msm", "ERROR: All pair coeffs are not set. Status:"],
  ["w118yaff_mil53al_switch3_t0", "lj/switch3/coulgauss/long", "ERROR: Run command before simulation box is defined"],
  ["w1ljv_lj96", "lj96/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w112ld_benzene_ld", "local/density", "ERROR: Run command before simulation box is defined"],
  ["w231co_lubricate_nolog_fld", "lubricate", "ERROR: All pair coeffs are not set. Status:"],
  ["w231co_lubricate_poly", "lubricate/poly", "ERROR: All pair coeffs are not set. Status:"],
  ["w231co_lubricateU_flaglog0", "lubricateU", "ERROR: All pair coeffs are not set. Status:"],
  ["w231co_lubricateU_poly", "lubricateU/poly", "ERROR: All pair coeffs are not set. Status:"],
  ["w33meamtrip_dim", "meam", "ERROR: All pair coeffs are not set. Status:"],
  ["w128ms_msmeam", "meam/ms", "ERROR: All pair coeffs are not set. Status:"],
  ["w47ms_si_min", "meam/spline", "ERROR: All pair coeffs are not set. Status:"],
  ["w47ms_sw_ti_run", "meam/sw/spline", "ERROR: All pair coeffs are not set. Status:"],
  ["w157mt_bond_angle", "mesocnt", "ERROR: All pair coeffs are not set. Status:"],
  ["w157mt_viscous", "mesocnt/viscous", "ERROR: All pair coeffs are not set. Status:"],
  ["w1ljx_mie", "mie/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w37so3_dimer", "mliap", "ERROR: All pair coeffs are not set. Status:"],
  ["w115p_mm3_mof5", "mm3/switch3/coulgauss/long", "ERROR: Run command before simulation box is defined"],
  ["w49pb_momb_md", "momb", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_tail_morse", "morse", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_morse_smooth_linear", "morse/smooth/linear", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_morse", "morse/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w142mx_multi_lucy_lookup", "multi/lucy", "ERROR: All pair coeffs are not set. Status:"],
  ["w10nb3b_harmonic", "nb3b/harmonic", "ERROR: All pair coeffs are not set. Status:"],
  ["w38nb3b_screened_random", "nb3b/screened", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_nm_cut", "nm/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_nm_cut_coul_cut", "nm/cut/coul/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_nm_cut_coul_long", "nm/cut/coul/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w220xp_nm_cut_split_tail", "nm/cut/split", "ERROR: All pair coeffs are not set. Status:"],
  ["w57oxdna_excv_pair2", "oxdna/excv", "ERROR: All pair coeffs are not set. Status:"],
  ["w57oxdna2_excv_bond", "oxdna2/excv", "ERROR: All pair coeffs are not set. Status:"],
  ["w48pb_pedone_triclinic", "pedone", "ERROR: All pair coeffs are not set. Status:"],
  ["w107peri_eps_block", "peri/eps", "ERROR: All pair coeffs are not set. Status:"],
  ["w22peri_lps_data", "peri/lps", "ERROR: All pair coeffs are not set. Status:"],
  ["w22peri_pmb_data", "peri/pmb", "ERROR: All pair coeffs are not set. Status:"],
  ["w107peri_ves_block", "peri/ves", "ERROR: All pair coeffs are not set. Status:"],
  ["w128pod_ta_static", "pod", "ERROR: All pair coeffs are not set. Status:"],
  ["w223mb_poly_cuta_eam", "polymorphic", "ERROR: All pair coeffs are not set. Status:"],
  ["w55reax_water", "reaxff", "ERROR: All pair coeffs are not set. Status:"],
  ["w52airebo_rebo_ethane", "rebo", "ERROR: All pair coeffs are not set. Status:"],
  ["w115p_rebomos", "rebomos", "ERROR: All pair coeffs are not set. Status:"],
  ["w133mf_region_ees", "resquared", "ERROR: All pair coeffs are not set. Status:"],
  ["w390rheo_ideal_gas", "rheo", "ERROR: All pair coeffs are not set. Status:"],
  ["w96rheo_pair_solid", "rheo/solid", "ERROR: All pair coeffs are not set. Status:"],
  ["w112sdpd_equipartition", "sdpd/taitwater/isothermal", "ERROR: All pair coeffs are not set. Status:"],
  ["w457dyngroup_smd_setvel", "smd/tlsph", "ERROR: All pair coeffs are not set. Status:"],
  ["w33snapchem_pair", "snap", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_shift_soft", "soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w83sph_heat3d", "sph/heatconduction", "ERROR: All pair coeffs are not set. Status:"],
  ["w228gs_sph_lj", "sph/lj", "ERROR: All pair coeffs are not set. Status:"],
  ["w267ed_spin_ptensor", "spin/dipole/long", "ERROR: All pair coeffs are not set. Status:"],
  ["w48sp2_dmi", "spin/dmi", "ERROR: All pair coeffs are not set. Status:"],
  ["w47spin_moving2", "spin/exchange", "ERROR: All pair coeffs are not set. Status:"],
  ["w48sp2_biquad", "spin/exchange/biquadratic", "ERROR: All pair coeffs are not set. Status:"],
  ["w48sp2_magelec", "spin/magelec", "ERROR: All pair coeffs are not set. Status:"],
  ["w2tsw_sw", "sw", "ERROR: All pair coeffs are not set. Status:"],
  ["w49mt_swang_spline", "sw/angle/table", "ERROR: All pair coeffs are not set. Status:"],
  ["w22pot_sw_mod_mos", "sw/mod", "ERROR: All pair coeffs are not set. Status:"],
  ["w113cf_benzene_table_spline", "table", "ERROR: All pair coeffs are not set. Status:"],
  ["w450dyngroup_tdpd", "tdpd", "ERROR: All pair coeffs are not set. Status:"],
  ["w2tsw_tersoff", "tersoff", "ERROR: All pair coeffs are not set. Status:"],
  ["w22pot_tersoff_mod_si", "tersoff/mod", "ERROR: All pair coeffs are not set. Status:"],
  ["w22pot_tersoff_modc_si", "tersoff/mod/c", "ERROR: All pair coeffs are not set. Status:"],
  ["w223mb_tersoff_table_si", "tersoff/table", "ERROR: All pair coeffs are not set. Status:"],
  ["w108xf_elfit_sisi", "tersoff/zbl", "ERROR: All pair coeffs are not set. Status:"],
  ["w49mt_tri_edge", "threebody/table", "ERROR: All pair coeffs are not set. Status:"],
  ["w69fep_tip4p", "tip4p/long/soft", "ERROR: All pair coeffs are not set. Status:"],
  ["w61tri_pair", "tri/lj", "ERROR: All pair coeffs are not set. Status:"],
  ["w146uf3_nb3b", "uf3", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_ufm", "ufm", "ERROR: All pair coeffs are not set. Status:"],
  ["w5vash_si", "vashishta", "ERROR: All pair coeffs are not set. Status:"],
  ["w472vashtable_runs", "vashishta/table", "ERROR: All pair coeffs are not set. Status:"],
  ["w219xp_wf_cut", "wf/cut", "ERROR: All pair coeffs are not set. Status:"],
  ["w115p_ylz_vesicle", "ylz", "ERROR: All pair coeffs are not set. Status:"],
  ["w88pair_tail_yukawa", "yukawa", "ERROR: All pair coeffs are not set. Status:"],
  ["w5yukawa_colloid", "yukawa/colloid", "ERROR: All pair coeffs are not set. Status:"],
  ["w15snap_grid_local", "zbl", "ERROR: All pair coeffs are not set. Status:"],
  ["w87bond_angle_fourier_linear", "zero", "ERROR: All pair coeffs are not set. Status:"],
  ["w40ksp_zerocoul_zero", "zero/coul", "ERROR: All pair coeffs are not set. Status:"],
];

/** The potential files the inputs below read. */
const POTS: Record<string, string> = Object.fromEntries(['Si.tersoff', 'Si.sw'].map((f) => [f, readFileSync(join(POTENTIALS, f), 'utf8')]));

/** [input, native's result] */
const INPUTS: [string, string][] = [
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style buck 5.0\npair_coeff 1 1 100.0 0.3 1.0\npair_coeff 2 2 100.0 0.3 1.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style born 5.0\npair_coeff 1 1 100.0 0.3 1.0 1.0 1.0\npair_coeff 2 2 100.0 0.3 1.0 1.0 1.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style morse 5.0\npair_coeff 1 1 1.0 1.0 2.0\npair_coeff 2 2 1.0 1.0 2.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style lj/cut 5.0\npair_coeff 1 1 1.0 2.0\npair_coeff 2 2 1.0 2.0\nrun 0\n", "OK"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style lj/cut 5.0\npair_coeff 1 1 1.0 2.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style lj/cut 5.0\npair_coeff 2 2 1.0 2.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style soft 5.0\npair_coeff 1 1 1.0\npair_coeff 2 2 1.0\nrun 0\n", "OK"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style yukawa 1.0 5.0\npair_coeff 1 1 1.0\npair_coeff 2 2 1.0\nrun 0\n", "OK"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style gauss 5.0\npair_coeff 1 1 1.0 1.0\npair_coeff 2 2 1.0 1.0\nrun 0\n", "OK"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style dpd 1.0 5.0 1234\npair_coeff 1 1 1.0 1.0\npair_coeff 2 2 1.0 1.0\nrun 0\n", "ERROR: Pair dpd requires ghost atoms store velocity"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style table linear 100\npair_coeff 1 1 nofile.txt K\nrun 0\n", "ERROR: cannot open pair table potential file nofile.txt: No such file or directory"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style tersoff\npair_coeff * * Si.tersoff Si NULL\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style sw\npair_coeff * * Si.sw Si NULL\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style hybrid lj/cut 5.0 morse 5.0\npair_coeff 1 1 lj/cut 1.0 2.0\npair_coeff 2 2 morse 1.0 1.0 2.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style hybrid lj/cut 5.0 morse 5.0\npair_coeff 1 1 lj/cut 1.0 2.0\npair_coeff 1 2 morse 1.0 1.0 2.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style hybrid lj/cut 5.0 morse 5.0\npair_coeff * * lj/cut 1.0 2.0\nrun 0\n", "ERROR: Pair hybrid sub-style morse is not used"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style hybrid/overlay lj/cut 5.0 morse 5.0\npair_coeff 1 1 lj/cut 1.0 2.0\npair_coeff 2 2 lj/cut 1.0 2.0\npair_coeff 1 1 morse 1.0 1.0 2.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style coul/cut 5.0\npair_coeff 1 1\npair_coeff 2 2\nrun 0\n", "ERROR: Pair style coul/cut requires atom attribute q"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style beck 5.0\npair_coeff 1 1 1.0 1.0 1.0 1.0 1.0\npair_coeff 2 2 1.0 1.0 1.0 1.0 1.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style mie/cut 5.0\npair_coeff 1 1 1.0 2.0 12 6\npair_coeff 2 2 1.0 2.0 12 6\nrun 0\n", "OK"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style lj96/cut 5.0\npair_coeff 1 1 1.0 2.0\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style zero 5.0\npair_coeff 1 1\nrun 0\n", "ERROR: All pair coeffs are not set. Status:"],
  ["units metal\nlattice diamond 5.431\nregion box block 0 2 0 2 0 2\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 28.0\npair_style lj/cut 5.0\npair_modify tail yes shift yes\npair_coeff * * 1.0 2.0\nrun 0\n", "ERROR: Cannot have both pair_modify shift and tail set to yes"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1e-4\nrun 0\n", "ERROR: Kspace style requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style coul/long 2.5\npair_coeff * *\nkspace_style ewald 1e-4\nrun 0\n", "ERROR: Kspace style requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style born/coul/wolf 0.2 2.5\npair_coeff * * 1.0 1.0 1.0 1.0 1.0\nrun 0\n", "ERROR: Pair style born/coul/wolf requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style buck/coul/cut 2.5\npair_coeff * * 1.0 1.0 1.0\nrun 0\n", "ERROR: Pair style buck/coul/cut requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut/coul/debye 1.0 2.5\npair_coeff * * 1.0 1.0\nrun 0\n", "ERROR: Pair style lj/cut/coul/cut requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style coul/dsf 0.2 2.5\npair_coeff * *\nrun 0\n", "ERROR: Pair style coul/dsf requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut/coul/dsf 0.2 2.5\npair_coeff * * 1.0 1.0\nrun 0\n", "ERROR: Pair style lj/cut/coul/dsf requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style coul/wolf 0.2 2.5\npair_coeff * *\nrun 0\n", "ERROR: Pair coul/wolf requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut/coul/cut 2.5\npair_coeff * * 1.0 1.0\nrun 0\n", "ERROR: Pair style lj/cut/coul/cut requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/charmm/coul/charmm 2.0 2.5\npair_coeff * * 1.0 1.0\nrun 0\n", "ERROR: Pair style lj/charmm/coul/charmm requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style hybrid/overlay lj/cut 2.5 coul/cut 2.5\npair_coeff * * lj/cut 1.0 1.0\npair_coeff * * coul/cut\nrun 0\n", "ERROR: Pair style coul/cut requires atom attribute q"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut/coul/wolf 0.2 2.5\npair_coeff * * 1.0 1.0\nrun 0\n", "ERROR: Pair style lj/cut/coul/wolf requires atom attribute q"],
];

describe('pair styles without pair_coeff as native LAMMPS', () => {
  it.each(NO_COEFFS)('%s (%s)', async (name, style, native) => {
    const [input, files] = withoutCoeffs(name, style);
    expect(await outcome(input, files)).toBe(native);
  });
});

describe('unset pairs and charges at the pair init as native LAMMPS', () => {
  it.each(INPUTS.map(([input, native], k) => [k + 1, input, native] as const))('input %i', async (_k, input, native) => {
    expect(await outcome(input, POTS)).toBe(native);
  });
});
