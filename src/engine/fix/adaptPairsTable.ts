/*
 * The pair parameters fix adapt can name, as native LAMMPS does (src/fix_adapt.cpp FixAdapt::init at b332f28): the
 * attribute's pstyle pparam is passed to the pair style's extract(), which returns a per-type-pair array (dim 2), a
 * global scalar (dim 0) or something else; only dim 2 and 0 can be adapted. PAIR_EXTRACT lists, for every pair style of
 * this engine that native gives an extract(), the names it accepts and their dims, read from the extract() of the file
 * named above each entry (src/ at b332f28; a style without its own extract() inherits its parent class's). A third
 * element names the engine's PairParams table where it differs from native's name (case aside). Not listed: eam/cd and
 * eam/cd/old (MANYBODY/pair_eam_cd.h overrides extract() to return nothing), and the peri styles, whose extract("theta")
 * returns the per-atom array that the first force evaluation allocates (null when fix adapt looks it up).
 */

/**
 * native extract() names: [name, dim, engine table name if different]. Plain data (no imports): the parity matrix
 * runner (scripts/parity/run-parity.mjs) reads it under Node's type stripping.
 */
export const PAIR_EXTRACT: Record<string, ReadonlyArray<readonly [string, number, string?]>> = {
  // AMOEBA/pair_amoeba.cpp
  'amoeba': [
    ['amtype', 0], ['atomic_num', 0], ['bond_flag', 0], ['angle_flag', 0], ['dihedral_flag', 0],
    ['improper_flag', 0], ['urey_flag', 0], ['pitorsion_flag', 0], ['bitorsion_flag', 0], ['opbend_cubic', 0],
    ['opbend_quartic', 0], ['opbend_pentic', 0], ['opbend_sextic', 0]
  ],
  // AMOEBA/pair_amoeba.cpp
  'hippo': [
    ['amtype', 0], ['atomic_num', 0], ['bond_flag', 0], ['angle_flag', 0], ['dihedral_flag', 0],
    ['improper_flag', 0], ['urey_flag', 0], ['pitorsion_flag', 0], ['bitorsion_flag', 0], ['opbend_cubic', 0],
    ['opbend_quartic', 0], ['opbend_pentic', 0], ['opbend_sextic', 0]
  ],
  // pair_born.cpp
  'born': [['a', 2], ['c', 2], ['d', 2]],
  // KSPACE/pair_born_coul_long.cpp
  'born/coul/long': [['cut_coul', 0]],
  // KSPACE/pair_born_coul_long.cpp
  'born/coul/long/cs': [['cut_coul', 0]],
  // KSPACE/pair_born_coul_msm.cpp
  'born/coul/msm': [['cut_coul', 0]],
  // EXTRA-PAIR/pair_born_gauss.cpp
  'born/gauss': [['biga0', 2], ['biga1', 2], ['r0', 2]],
  // pair_buck.cpp
  'buck': [['a', 2], ['c', 2]],
  // pair_buck_coul_cut.cpp
  'buck/coul/cut': [['a', 2], ['c', 2]],
  // KSPACE/pair_buck_coul_long.cpp
  'buck/coul/long': [['cut_coul', 0], ['a', 2], ['c', 2]],
  // KSPACE/pair_buck_coul_long.cpp
  'buck/coul/long/cs': [['cut_coul', 0], ['a', 2], ['c', 2]],
  // KSPACE/pair_buck_coul_msm.cpp
  'buck/coul/msm': [['cut_coul', 0], ['a', 2], ['c', 2]],
  // KSPACE/pair_buck_long_coul_long.cpp
  'buck/long/coul/long': [
    ['B', 2], ['ewald_order', 0], ['ewald_cut', 0], ['ewald_mix', 0], ['cut_coul', 0], ['cut_vdwl', 0],
    ['cut_LJ', 0]
  ],
  // EXTRA-PAIR/pair_buck_mdf.cpp
  'buck/mdf': [['a', 2], ['c', 2]],
  // MOFFF/pair_buck6d_coul_gauss_dsf.cpp
  'buck6d/coul/gauss/dsf': [
    ['cut_ljsq', 2], ['buck6d1', 2], ['buck6d2', 2], ['buck6d3', 2], ['buck6d4', 2], ['rsmooth_sq', 2], ['c0', 2],
    ['c1', 2], ['c2', 2], ['c3', 2], ['c4', 2], ['c5', 2], ['offset', 2], ['cut_coul', 0]
  ],
  // MOFFF/pair_buck6d_coul_gauss_long.cpp
  'buck6d/coul/gauss/long': [
    ['cut_ljsq', 2], ['buck6d1', 2], ['buck6d2', 2], ['buck6d3', 2], ['buck6d4', 2], ['rsmooth_sq', 2], ['c0', 2],
    ['c1', 2], ['c2', 2], ['c3', 2], ['c4', 2], ['c5', 2], ['offset', 2], ['cut_coul', 0]
  ],
  // COLLOID/pair_colloid.cpp
  'colloid': [['a12', 2, 'a'], ['sigma', 2], ['d1', 2], ['d2', 2], ['diameter', 2]],
  // EXTRA-PAIR/pair_coul_ctip.cpp
  'coul/ctip': [
    ['cut_coul', 0], ['chi', 1], ['eta', 1], ['gamma', 1], ['zeta', 1], ['zcore', 1], ['qmin', 1], ['qmax', 1],
    ['omega', 1]
  ],
  // pair_coul_cut.cpp
  'coul/cut': [['cut_coul', 2, 'cut'], ['scale', 2]],
  // pair_coul_cut.cpp
  'coul/cut/dielectric': [['cut_coul', 2, 'cut'], ['scale', 2]],
  // pair_coul_cut.cpp
  'coul/debye': [['cut_coul', 2, 'cut'], ['scale', 2]],
  // pair_coul_cut.cpp
  'coul/slater/cut': [['cut_coul', 2, 'cut'], ['scale', 2]],
  // EXTRA-PAIR/pair_coul_cut_global.cpp
  'coul/cut/global': [['cut_coul', 0], ['scale', 2]],
  // FEP/pair_coul_cut_soft.cpp
  'coul/cut/soft': [['lambda', 2]],
  // FEP/pair_coul_cut_soft_gapsys.cpp
  'coul/cut/soft/gapsys': [['lambda', 2]],
  // pair_coul_dsf.cpp
  'coul/dsf': [['cut_coul', 0]],
  // KSPACE/pair_coul_esp.cpp
  'coul/esp': [['cut_coul', 0], ['scale', 2]],
  // EXTRA-PAIR/pair_coul_exclude.cpp
  'coul/exclude': [['cut_coul', 0]],
  // KSPACE/pair_coul_long.cpp
  'coul/long': [['cut_coul', 0], ['scale', 2]],
  // KSPACE/pair_coul_long.cpp
  'coul/long/cs': [['cut_coul', 0], ['scale', 2]],
  // KSPACE/pair_coul_long.cpp
  'coul/long/dielectric': [['cut_coul', 0], ['scale', 2]],
  // FEP/pair_coul_long_soft.cpp
  'coul/long/soft': [['cut_coul', 0], ['scale', 2], ['lambda', 2]],
  // KSPACE/pair_coul_msm.cpp
  'coul/msm': [['cut_coul', 0], ['scale', 2]],
  // EXTRA-PAIR/pair_coul_slater_long.cpp
  'coul/slater/long': [['cut_coul', 0], ['lamda', 0], ['scale', 2]],
  // KSPACE/pair_coul_streitz.cpp
  'coul/streitz': [
    ['cut_coul', 0], ['scale', 2], ['chi', 1], ['eta', 1], ['gamma', 1], ['zeta', 1], ['zcore', 1],
    ['kspacetype', 0], ['alpha', 0]
  ],
  // DRUDE/pair_coul_tt.cpp
  'coul/tt': [['scale', 2], ['b', 2], ['c', 2], ['ntt', 2]],
  // DPD-BASIC/pair_dpd_coul_slater_long.cpp
  'dpd/coul/slater/long': [['cut_coul', 0], ['lamda', 0], ['a0', 2], ['gamma', 2]],
  // MANYBODY/pair_eam.cpp
  'eam': [['scale', 2]],
  // MANYBODY/pair_eam.cpp
  'eam/alloy': [['scale', 2]],
  // MANYBODY/pair_eam.cpp
  'eam/fs': [['scale', 2]],
  // MANYBODY/pair_eam.cpp
  'eam/he': [['scale', 2]],
  // APIP/pair_eam_apip.cpp
  'eam/apip': [['scale', 2]],
  // APIP/pair_eam_apip.cpp
  'eam/fs/apip': [['scale', 2]],
  // EXTRA-PAIR/pair_gauss.cpp
  'gauss': [['a', 2]],
  // EXTRA-PAIR/pair_harmonic_cut.cpp
  'harmonic/cut': [['k', 2], ['cut', 2]],
  // EXTRA-PAIR/pair_lennard_mdf.cpp
  'lennard/mdf': [['a', 2, 'aparm'], ['b', 2, 'bparm']],
  // LEPTON/pair_lepton_coul.cpp
  'lepton/coul': [['cut_coul', 0]],
  // MOLECULE/pair_lj_charmm_coul_charmm.cpp
  'lj/charmm/coul/charmm': [
    ['lj14_1', 2], ['lj14_2', 2], ['lj14_3', 2], ['lj14_4', 2], ['sigma', 2], ['epsilon', 2], ['implicit', 0]
  ],
  // MOLECULE/pair_lj_charmm_coul_charmm.cpp
  'lj/charmm/coul/charmm/implicit': [
    ['lj14_1', 2], ['lj14_2', 2], ['lj14_3', 2], ['lj14_4', 2], ['sigma', 2], ['epsilon', 2], ['implicit', 0]
  ],
  // KSPACE/pair_lj_charmm_coul_long.cpp
  'lj/charmm/coul/long': [
    ['lj14_1', 2], ['lj14_2', 2], ['lj14_3', 2], ['lj14_4', 2], ['sigma', 2], ['epsilon', 2], ['implicit', 0],
    ['cut_coul', 0]
  ],
  // FEP/pair_lj_charmm_coul_long_soft.cpp
  'lj/charmm/coul/long/soft': [
    ['lj14_1', 2], ['lj14_2', 2], ['lj14_3', 2], ['lj14_4', 2], ['epsilon', 2], ['sigma', 2], ['lambda', 2],
    ['implicit', 0], ['cut_coul', 0]
  ],
  // KSPACE/pair_lj_charmm_coul_msm.cpp
  'lj/charmm/coul/msm': [
    ['lj14_1', 2], ['lj14_2', 2], ['lj14_3', 2], ['lj14_4', 2], ['sigma', 2], ['epsilon', 2], ['implicit', 0],
    ['cut_coul', 0]
  ],
  // MOLECULE/pair_lj_charmmfsw_coul_charmmfsh.cpp
  'lj/charmmfsw/coul/charmmfsh': [
    ['lj14_1', 2], ['lj14_2', 2], ['lj14_3', 2], ['lj14_4', 2], ['implicit', 0], ['cut_coul', 0],
    ['cut_lj_inner', 0], ['cut_lj', 0], ['dihedflag', 0]
  ],
  // KSPACE/pair_lj_charmmfsw_coul_long.cpp
  'lj/charmmfsw/coul/long': [
    ['lj14_1', 2], ['lj14_2', 2], ['lj14_3', 2], ['lj14_4', 2], ['sigma', 2], ['epsilon', 2], ['implicit', 0],
    ['cut_coul', 0], ['cut_lj_inner', 0], ['cut_lj', 0], ['dihedflag', 0]
  ],
  // CLASS2/pair_lj_class2.cpp
  'lj/class2': [['epsilon', 2], ['sigma', 2]],
  // CLASS2/pair_lj_class2_coul_cut.cpp
  'lj/class2/coul/cut': [['cut_coul', 2], ['epsilon', 2], ['sigma', 2]],
  // FEP/pair_lj_class2_coul_cut_soft.cpp
  'lj/class2/coul/cut/soft': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2], ['lambda', 2]],
  // CLASS2/pair_lj_class2_coul_long.cpp
  'lj/class2/coul/long': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2]],
  // CLASS2/pair_lj_class2_coul_long.cpp
  'lj/class2/coul/long/cs': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2]],
  // FEP/pair_lj_class2_coul_long_soft.cpp
  'lj/class2/coul/long/soft': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2], ['lambda', 2]],
  // FEP/pair_lj_class2_soft.cpp
  'lj/class2/soft': [['epsilon', 2], ['sigma', 2], ['lambda', 2]],
  // EXTRA-PAIR/pair_lj_cubic.cpp
  'lj/cubic': [['epsilon', 2], ['sigma', 2]],
  // pair_lj_cut.cpp
  'lj/cut': [['epsilon', 2], ['sigma', 2]],
  // pair_lj_cut_coul_cut.cpp
  'lj/cut/coul/cut': [['cut_coul', 2], ['epsilon', 2], ['sigma', 2]],
  // pair_lj_cut_coul_cut.cpp
  'lj/cut/coul/cut/dielectric': [['cut_coul', 2], ['epsilon', 2], ['sigma', 2]],
  // pair_lj_cut_coul_cut.cpp
  'lj/cut/coul/debye': [['cut_coul', 2], ['epsilon', 2], ['sigma', 2]],
  // pair_lj_cut_coul_cut.cpp
  'lj/cut/coul/debye/dielectric': [['cut_coul', 2], ['epsilon', 2], ['sigma', 2]],
  // FEP/pair_lj_cut_coul_cut_soft.cpp
  'lj/cut/coul/cut/soft': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2], ['lambda', 2]],
  // EXTRA-PAIR/pair_lj_cut_coul_dsf.cpp
  'lj/cut/coul/dsf': [['cut_coul', 0]],
  // KSPACE/pair_lj_cut_coul_esp.cpp
  'lj/cut/coul/esp': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2]],
  // KSPACE/pair_lj_cut_coul_long.cpp
  'lj/cut/coul/long': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2]],
  // KSPACE/pair_lj_cut_coul_long.cpp
  'lj/cut/coul/long/cs': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2]],
  // KSPACE/pair_lj_cut_coul_long.cpp
  'lj/cut/coul/long/dielectric': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2]],
  // FEP/pair_lj_cut_coul_long_soft.cpp
  'lj/cut/coul/long/soft': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2], ['lambda', 2]],
  // KSPACE/pair_lj_cut_coul_msm.cpp
  'lj/cut/coul/msm': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2]],
  // DIELECTRIC/pair_lj_cut_coul_msm_dielectric.cpp
  'lj/cut/coul/msm/dielectric': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2]],
  // DIPOLE/pair_lj_cut_dipole_cut.cpp
  'lj/cut/dipole/cut': [['cut_coul', 2, 'cutc'], ['epsilon', 2], ['sigma', 2]],
  // DIPOLE/pair_lj_cut_dipole_long.cpp
  'lj/cut/dipole/long': [['cut_coul', 0], ['ewald_order', 0], ['ewald_mix', 0], ['epsilon', 2], ['sigma', 2]],
  // FEP/pair_lj_cut_soft.cpp
  'lj/cut/soft': [['epsilon', 2], ['sigma', 2], ['lambda', 2]],
  // FEP/pair_lj_cut_soft_gapsys.cpp
  'lj/cut/soft/gapsys': [['epsilon', 2], ['sigma', 2], ['lambda', 2]],
  // EXTRA-PAIR/pair_lj_cut_sphere.cpp
  'lj/cut/sphere': [['epsilon', 2]],
  // DRUDE/pair_lj_cut_thole_long.cpp
  'lj/cut/thole/long': [
    ['cut_coul', 0], ['epsilon', 2], ['sigma', 2], ['scale', 2], ['polar', 2], ['thole', 2], ['ascreen', 2]
  ],
  // MOLECULE/pair_lj_cut_tip4p_cut.cpp
  'lj/cut/tip4p/cut': [
    ['qdist', 0], ['typeO', 0], ['typeH', 0], ['typeA', 0], ['typeB', 0], ['cut_coul', 0], ['epsilon', 2],
    ['sigma', 2]
  ],
  // KSPACE/pair_lj_cut_tip4p_long.cpp
  'lj/cut/tip4p/long': [
    ['qdist', 0], ['typeO', 0], ['typeH', 0], ['typeA', 0], ['typeB', 0], ['cut_coul', 0], ['epsilon', 2],
    ['sigma', 2]
  ],
  // FEP/pair_lj_cut_tip4p_long_soft.cpp
  'lj/cut/tip4p/long/soft': [
    ['qdist', 0], ['typeO', 0], ['typeH', 0], ['typeA', 0], ['typeB', 0], ['cut_coul', 0], ['epsilon', 2],
    ['sigma', 2], ['lambda', 2]
  ],
  // pair_lj_expand.cpp
  'lj/expand': [['epsilon', 2], ['sigma', 2], ['delta', 2]],
  // EXTRA-PAIR/pair_lj_expand_coul_long.cpp
  'lj/expand/coul/long': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2], ['delta', 2]],
  // EXTRA-PAIR/pair_lj_expand_sphere.cpp
  'lj/expand/sphere': [['epsilon', 2], ['sigma', 2]],
  // EXTRA-PAIR/pair_lj_gromacs.cpp
  'lj/gromacs': [['epsilon', 2], ['sigma', 2]],
  // EXTRA-PAIR/pair_lj_gromacs_coul_gromacs.cpp
  'lj/gromacs/coul/gromacs': [['epsilon', 2], ['sigma', 2]],
  // KSPACE/pair_lj_long_coul_long.cpp
  'lj/long/coul/long': [
    ['B', 2], ['sigma', 2], ['epsilon', 2], ['ewald_order', 0], ['ewald_cut', 0], ['ewald_mix', 0], ['cut_coul', 0],
    ['cut_vdwl', 0], ['cut_LJ', 0]
  ],
  // KSPACE/pair_lj_long_coul_long.cpp
  'lj/long/coul/long/dielectric': [
    ['B', 2], ['sigma', 2], ['epsilon', 2], ['ewald_order', 0], ['ewald_cut', 0], ['ewald_mix', 0], ['cut_coul', 0],
    ['cut_vdwl', 0], ['cut_LJ', 0]
  ],
  // DIPOLE/pair_lj_long_dipole_long.cpp
  'lj/long/dipole/long': [
    ['B', 2], ['sigma', 2], ['epsilon', 2], ['ewald_order', 0], ['ewald_cut', 0], ['ewald_mix', 0], ['cut_coul', 0],
    ['cut_vdwl', 0]
  ],
  // KSPACE/pair_lj_long_tip4p_long.cpp
  'lj/long/tip4p/long': [
    ['qdist', 0], ['typeO', 0], ['typeH', 0], ['typeA', 0], ['typeB', 0], ['cut_coul', 0], ['B', 2], ['sigma', 2],
    ['epsilon', 2], ['ewald_order', 0], ['ewald_cut', 0], ['ewald_mix', 0], ['cut_vdwl', 0], ['cut_LJ', 0]
  ],
  // EXTRA-PAIR/pair_lj_mdf.cpp
  'lj/mdf': [['epsilon', 2], ['sigma', 2]],
  // EXTRA-PAIR/pair_lj_pirani.cpp
  'lj/pirani': [['alpha', 2], ['beta', 2], ['gamma', 2], ['rm', 2], ['epsilon', 2]],
  // CG-SPICA/pair_lj_spica.cpp
  'lj/sdk': [
    ['epsilon', 2], ['sigma', 2], ['lj_type', 2], ['lj1', 2], ['lj2', 2], ['lj3', 2], ['lj4', 2], ['rminsq', 2],
    ['emin', 2]
  ],
  // CG-SPICA/pair_lj_spica.cpp
  'lj/spica': [
    ['epsilon', 2], ['sigma', 2], ['lj_type', 2], ['lj1', 2], ['lj2', 2], ['lj3', 2], ['lj4', 2], ['rminsq', 2],
    ['emin', 2]
  ],
  // CG-SPICA/pair_lj_spica_coul_long.cpp
  'lj/sdk/coul/long': [
    ['epsilon', 2], ['sigma', 2], ['lj_type', 2], ['lj1', 2], ['lj2', 2], ['lj3', 2], ['lj4', 2], ['rminsq', 2],
    ['emin', 2], ['cut_coul', 0]
  ],
  // CG-SPICA/pair_lj_spica_coul_long.cpp
  'lj/spica/coul/long': [
    ['epsilon', 2], ['sigma', 2], ['lj_type', 2], ['lj1', 2], ['lj2', 2], ['lj3', 2], ['lj4', 2], ['rminsq', 2],
    ['emin', 2], ['cut_coul', 0]
  ],
  // CG-SPICA/pair_lj_spica_coul_msm.cpp
  'lj/sdk/coul/msm': [
    ['epsilon', 2], ['sigma', 2], ['lj_type', 2], ['lj1', 2], ['lj2', 2], ['lj3', 2], ['lj4', 2], ['rminsq', 2],
    ['emin', 2], ['cut_coul', 0], ['cut_msm', 0]
  ],
  // CG-SPICA/pair_lj_spica_coul_msm.cpp
  'lj/spica/coul/msm': [
    ['epsilon', 2], ['sigma', 2], ['lj_type', 2], ['lj1', 2], ['lj2', 2], ['lj3', 2], ['lj4', 2], ['rminsq', 2],
    ['emin', 2], ['cut_coul', 0], ['cut_msm', 0]
  ],
  // EXTRA-PAIR/pair_lj_smooth_linear.cpp
  'lj/sf': [['epsilon', 2], ['sigma', 2]],
  // EXTRA-PAIR/pair_lj_smooth_linear.cpp
  'lj/smooth/linear': [['epsilon', 2], ['sigma', 2]],
  // DIPOLE/pair_lj_sf_dipole_sf.cpp
  'lj/sf/dipole/sf': [['epsilon', 2], ['sigma', 2], ['scale', 2], ['cut_coul', 2, 'cutc']],
  // EXTRA-PAIR/pair_lj_smooth.cpp
  'lj/smooth': [['epsilon', 2], ['sigma', 2]],
  // YAFF/pair_lj_switch3_coulgauss_long.cpp
  'lj/switch3/coulgauss/long': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2], ['gamma', 2]],
  // EXTRA-PAIR/pair_lj96_cut.cpp
  'lj96/cut': [['epsilon', 2], ['sigma', 2]],
  // MEAM/pair_meam.cpp
  'meam': [['scale', 2]],
  // MEAM/pair_meam.cpp
  'meam/ms': [['scale', 2]],
  // EXTRA-PAIR/pair_mie_cut.cpp
  'mie/cut': [['epsilon', 2], ['sigma', 2], ['gamR', 2, 'gammaR'], ['gamA', 2, 'gammaA']],
  // YAFF/pair_mm3_switch3_coulgauss_long.cpp
  'mm3/switch3/coulgauss/long': [['cut_coul', 0], ['epsilon', 2], ['sigma', 2], ['gamma', 2]],
  // pair_morse.cpp
  'morse': [['d0', 2], ['r0', 2], ['alpha', 2]],
  // EXTRA-PAIR/pair_morse_smooth_linear.cpp
  'morse/smooth/linear': [['d0', 2], ['r0', 2], ['alpha', 2]],
  // FEP/pair_morse_soft.cpp
  'morse/soft': [['d0', 2], ['r0', 2], ['alpha', 2], ['lambda', 2]],
  // EXTRA-PAIR/pair_nm_cut.cpp
  'nm/cut': [['e0', 2], ['r0', 2], ['nn', 2], ['mm', 2]],
  // EXTRA-PAIR/pair_nm_cut.cpp
  'nm/cut/split': [['e0', 2], ['r0', 2], ['nn', 2], ['mm', 2]],
  // EXTRA-PAIR/pair_nm_cut_coul_cut.cpp
  'nm/cut/coul/cut': [['cut_coul', 2], ['e0', 2], ['r0', 2], ['nn', 2], ['mm', 2]],
  // EXTRA-PAIR/pair_nm_cut_coul_long.cpp
  'nm/cut/coul/long': [['cut_coul', 0], ['e0', 2], ['r0', 2], ['nn', 2], ['mm', 2]],
  // CG-DNA/pair_oxdna_coaxstk.cpp
  'oxdna/coaxstk': [
    ['k_cxst', 2], ['cut_cxst_0', 2], ['cut_cxst_c', 2], ['cut_cxst_lo', 2], ['cut_cxst_hi', 2], ['cut_cxst_lc', 2],
    ['cut_cxst_hc', 2], ['b_cxst_lo', 2], ['b_cxst_hi', 2], ['a_cxst1', 2], ['theta_cxst1_0', 2],
    ['dtheta_cxst1_ast', 2], ['b_cxst1', 2], ['dtheta_cxst1_c', 2], ['a_cxst4', 2], ['theta_cxst4_0', 2],
    ['dtheta_cxst4_ast', 2], ['b_cxst4', 2], ['dtheta_cxst4_c', 2], ['a_cxst5', 2], ['theta_cxst5_0', 2],
    ['dtheta_cxst5_ast', 2], ['b_cxst5', 2], ['dtheta_cxst5_c', 2], ['a_cxst6', 2], ['theta_cxst6_0', 2],
    ['dtheta_cxst6_ast', 2], ['b_cxst6', 2], ['dtheta_cxst6_c', 2], ['a_cxst3p', 2], ['cosphi_cxst3p_ast', 2],
    ['b_cxst3p', 2], ['cosphi_cxst3p_c', 2], ['a_cxst4p', 2], ['cosphi_cxst4p_ast', 2], ['b_cxst4p', 2],
    ['cosphi_cxst4p_c', 2]
  ],
  // CG-DNA/pair_oxdna_coaxstk.cpp
  'oxrna2/coaxstk': [
    ['k_cxst', 2], ['cut_cxst_0', 2], ['cut_cxst_c', 2], ['cut_cxst_lo', 2], ['cut_cxst_hi', 2], ['cut_cxst_lc', 2],
    ['cut_cxst_hc', 2], ['b_cxst_lo', 2], ['b_cxst_hi', 2], ['a_cxst1', 2], ['theta_cxst1_0', 2],
    ['dtheta_cxst1_ast', 2], ['b_cxst1', 2], ['dtheta_cxst1_c', 2], ['a_cxst4', 2], ['theta_cxst4_0', 2],
    ['dtheta_cxst4_ast', 2], ['b_cxst4', 2], ['dtheta_cxst4_c', 2], ['a_cxst5', 2], ['theta_cxst5_0', 2],
    ['dtheta_cxst5_ast', 2], ['b_cxst5', 2], ['dtheta_cxst5_c', 2], ['a_cxst6', 2], ['theta_cxst6_0', 2],
    ['dtheta_cxst6_ast', 2], ['b_cxst6', 2], ['dtheta_cxst6_c', 2], ['a_cxst3p', 2], ['cosphi_cxst3p_ast', 2],
    ['b_cxst3p', 2], ['cosphi_cxst3p_c', 2], ['a_cxst4p', 2], ['cosphi_cxst4p_ast', 2], ['b_cxst4p', 2],
    ['cosphi_cxst4p_c', 2]
  ],
  // CG-DNA/pair_oxdna_excv.cpp
  'oxdna/excv': [
    ['epsilon_bkbk', 2], ['sigma_bkbk', 2], ['cut_bkbk_ast', 2], ['b_bkbk', 2], ['cut_bkbk_c', 2],
    ['epsilon_bkbs', 2], ['sigma_bkbs', 2], ['cut_bkbs_ast', 2], ['b_bkbs', 2], ['cut_bkbs_c', 2],
    ['sigma4_bkbs', 2], ['epsilon_bsbs', 2], ['sigma_bsbs', 2], ['cut_bsbs_ast', 2], ['b_bsbs', 2],
    ['cut_bsbs_c', 2]
  ],
  // CG-DNA/pair_oxdna_excv.cpp
  'oxdna2/excv': [
    ['epsilon_bkbk', 2], ['sigma_bkbk', 2], ['cut_bkbk_ast', 2], ['b_bkbk', 2], ['cut_bkbk_c', 2],
    ['epsilon_bkbs', 2], ['sigma_bkbs', 2], ['cut_bkbs_ast', 2], ['b_bkbs', 2], ['cut_bkbs_c', 2],
    ['sigma4_bkbs', 2], ['epsilon_bsbs', 2], ['sigma_bsbs', 2], ['cut_bsbs_ast', 2], ['b_bsbs', 2],
    ['cut_bsbs_c', 2]
  ],
  // CG-DNA/pair_oxdna_excv.cpp
  'oxdna3/excv': [
    ['epsilon_bkbk', 2], ['sigma_bkbk', 2], ['cut_bkbk_ast', 2], ['b_bkbk', 2], ['cut_bkbk_c', 2],
    ['epsilon_bkbs', 2], ['sigma_bkbs', 2], ['cut_bkbs_ast', 2], ['b_bkbs', 2], ['cut_bkbs_c', 2],
    ['sigma4_bkbs', 2], ['epsilon_bsbs', 2], ['sigma_bsbs', 2], ['cut_bsbs_ast', 2], ['b_bsbs', 2],
    ['cut_bsbs_c', 2]
  ],
  // CG-DNA/pair_oxdna_excv.cpp
  'oxrna2/excv': [
    ['epsilon_bkbk', 2], ['sigma_bkbk', 2], ['cut_bkbk_ast', 2], ['b_bkbk', 2], ['cut_bkbk_c', 2],
    ['epsilon_bkbs', 2], ['sigma_bkbs', 2], ['cut_bkbs_ast', 2], ['b_bkbs', 2], ['cut_bkbs_c', 2],
    ['sigma4_bkbs', 2], ['epsilon_bsbs', 2], ['sigma_bsbs', 2], ['cut_bsbs_ast', 2], ['b_bsbs', 2],
    ['cut_bsbs_c', 2]
  ],
  // CG-DNA/pair_oxdna_hbond.cpp
  'oxdna/hbond': [
    ['epsilon_hb', 2], ['a_hb', 2], ['cut_hb_0', 2], ['cut_hb_c', 2], ['cut_hb_lo', 2], ['cut_hb_hi', 2],
    ['cut_hb_lc', 2], ['cut_hb_hc', 2], ['b_hb_lo', 2], ['b_hb_hi', 2], ['shift_hb', 2], ['a_hb1', 2],
    ['theta_hb1_0', 2], ['dtheta_hb1_ast', 2], ['b_hb1', 2], ['dtheta_hb1_c', 2], ['a_hb2', 2], ['theta_hb2_0', 2],
    ['dtheta_hb2_ast', 2], ['b_hb2', 2], ['dtheta_hb2_c', 2], ['a_hb3', 2], ['theta_hb3_0', 2],
    ['dtheta_hb3_ast', 2], ['b_hb3', 2], ['dtheta_hb3_c', 2], ['a_hb4', 2], ['theta_hb4_0', 2],
    ['dtheta_hb4_ast', 2], ['b_hb4', 2], ['dtheta_hb4_c', 2], ['a_hb7', 2], ['theta_hb7_0', 2],
    ['dtheta_hb7_ast', 2], ['b_hb7', 2], ['dtheta_hb7_c', 2], ['a_hb8', 2], ['theta_hb8_0', 2],
    ['dtheta_hb8_ast', 2], ['b_hb8', 2], ['dtheta_hb8_c', 2]
  ],
  // CG-DNA/pair_oxdna_hbond.cpp
  'oxdna2/hbond': [
    ['epsilon_hb', 2], ['a_hb', 2], ['cut_hb_0', 2], ['cut_hb_c', 2], ['cut_hb_lo', 2], ['cut_hb_hi', 2],
    ['cut_hb_lc', 2], ['cut_hb_hc', 2], ['b_hb_lo', 2], ['b_hb_hi', 2], ['shift_hb', 2], ['a_hb1', 2],
    ['theta_hb1_0', 2], ['dtheta_hb1_ast', 2], ['b_hb1', 2], ['dtheta_hb1_c', 2], ['a_hb2', 2], ['theta_hb2_0', 2],
    ['dtheta_hb2_ast', 2], ['b_hb2', 2], ['dtheta_hb2_c', 2], ['a_hb3', 2], ['theta_hb3_0', 2],
    ['dtheta_hb3_ast', 2], ['b_hb3', 2], ['dtheta_hb3_c', 2], ['a_hb4', 2], ['theta_hb4_0', 2],
    ['dtheta_hb4_ast', 2], ['b_hb4', 2], ['dtheta_hb4_c', 2], ['a_hb7', 2], ['theta_hb7_0', 2],
    ['dtheta_hb7_ast', 2], ['b_hb7', 2], ['dtheta_hb7_c', 2], ['a_hb8', 2], ['theta_hb8_0', 2],
    ['dtheta_hb8_ast', 2], ['b_hb8', 2], ['dtheta_hb8_c', 2]
  ],
  // CG-DNA/pair_oxdna_hbond.cpp
  'oxdna3/hbond': [
    ['epsilon_hb', 2], ['a_hb', 2], ['cut_hb_0', 2], ['cut_hb_c', 2], ['cut_hb_lo', 2], ['cut_hb_hi', 2],
    ['cut_hb_lc', 2], ['cut_hb_hc', 2], ['b_hb_lo', 2], ['b_hb_hi', 2], ['shift_hb', 2], ['a_hb1', 2],
    ['theta_hb1_0', 2], ['dtheta_hb1_ast', 2], ['b_hb1', 2], ['dtheta_hb1_c', 2], ['a_hb2', 2], ['theta_hb2_0', 2],
    ['dtheta_hb2_ast', 2], ['b_hb2', 2], ['dtheta_hb2_c', 2], ['a_hb3', 2], ['theta_hb3_0', 2],
    ['dtheta_hb3_ast', 2], ['b_hb3', 2], ['dtheta_hb3_c', 2], ['a_hb4', 2], ['theta_hb4_0', 2],
    ['dtheta_hb4_ast', 2], ['b_hb4', 2], ['dtheta_hb4_c', 2], ['a_hb7', 2], ['theta_hb7_0', 2],
    ['dtheta_hb7_ast', 2], ['b_hb7', 2], ['dtheta_hb7_c', 2], ['a_hb8', 2], ['theta_hb8_0', 2],
    ['dtheta_hb8_ast', 2], ['b_hb8', 2], ['dtheta_hb8_c', 2]
  ],
  // CG-DNA/pair_oxdna_hbond.cpp
  'oxrna2/hbond': [
    ['epsilon_hb', 2], ['a_hb', 2], ['cut_hb_0', 2], ['cut_hb_c', 2], ['cut_hb_lo', 2], ['cut_hb_hi', 2],
    ['cut_hb_lc', 2], ['cut_hb_hc', 2], ['b_hb_lo', 2], ['b_hb_hi', 2], ['shift_hb', 2], ['a_hb1', 2],
    ['theta_hb1_0', 2], ['dtheta_hb1_ast', 2], ['b_hb1', 2], ['dtheta_hb1_c', 2], ['a_hb2', 2], ['theta_hb2_0', 2],
    ['dtheta_hb2_ast', 2], ['b_hb2', 2], ['dtheta_hb2_c', 2], ['a_hb3', 2], ['theta_hb3_0', 2],
    ['dtheta_hb3_ast', 2], ['b_hb3', 2], ['dtheta_hb3_c', 2], ['a_hb4', 2], ['theta_hb4_0', 2],
    ['dtheta_hb4_ast', 2], ['b_hb4', 2], ['dtheta_hb4_c', 2], ['a_hb7', 2], ['theta_hb7_0', 2],
    ['dtheta_hb7_ast', 2], ['b_hb7', 2], ['dtheta_hb7_c', 2], ['a_hb8', 2], ['theta_hb8_0', 2],
    ['dtheta_hb8_ast', 2], ['b_hb8', 2], ['dtheta_hb8_c', 2]
  ],
  // CG-DNA/pair_oxdna_stk.cpp
  'oxdna/stk': [
    ['epsilon_st', 2], ['a_st', 2], ['cut_st_0', 2], ['cut_st_c', 2], ['cut_st_lo', 2], ['cut_st_hi', 2],
    ['cut_st_lc', 2], ['cut_st_hc', 2], ['b_st_lo', 2], ['b_st_hi', 2], ['shift_st', 2], ['a_st4', 2],
    ['theta_st4_0', 2], ['dtheta_st4_ast', 2], ['b_st4', 2], ['dtheta_st4_c', 2], ['a_st5', 2], ['theta_st5_0', 2],
    ['dtheta_st5_ast', 2], ['b_st5', 2], ['dtheta_st5_c', 2], ['a_st6', 2], ['theta_st6_0', 2],
    ['dtheta_st6_ast', 2], ['b_st6', 2], ['dtheta_st6_c', 2], ['a_st1', 2], ['cosphi_st1_ast', 2], ['b_st1', 2],
    ['cosphi_st1_c', 2], ['a_st2', 2], ['cosphi_st2_ast', 2], ['b_st2', 2], ['cosphi_st2_c', 2]
  ],
  // CG-DNA/pair_oxdna_stk.cpp
  'oxdna2/stk': [
    ['epsilon_st', 2], ['a_st', 2], ['cut_st_0', 2], ['cut_st_c', 2], ['cut_st_lo', 2], ['cut_st_hi', 2],
    ['cut_st_lc', 2], ['cut_st_hc', 2], ['b_st_lo', 2], ['b_st_hi', 2], ['shift_st', 2], ['a_st4', 2],
    ['theta_st4_0', 2], ['dtheta_st4_ast', 2], ['b_st4', 2], ['dtheta_st4_c', 2], ['a_st5', 2], ['theta_st5_0', 2],
    ['dtheta_st5_ast', 2], ['b_st5', 2], ['dtheta_st5_c', 2], ['a_st6', 2], ['theta_st6_0', 2],
    ['dtheta_st6_ast', 2], ['b_st6', 2], ['dtheta_st6_c', 2], ['a_st1', 2], ['cosphi_st1_ast', 2], ['b_st1', 2],
    ['cosphi_st1_c', 2], ['a_st2', 2], ['cosphi_st2_ast', 2], ['b_st2', 2], ['cosphi_st2_c', 2]
  ],
  // CG-DNA/pair_oxdna_stk.cpp
  'oxdna3/stk': [
    ['epsilon_st', 2], ['a_st', 2], ['cut_st_0', 2], ['cut_st_c', 2], ['cut_st_lo', 2], ['cut_st_hi', 2],
    ['cut_st_lc', 2], ['cut_st_hc', 2], ['b_st_lo', 2], ['b_st_hi', 2], ['shift_st', 2], ['a_st4', 2],
    ['theta_st4_0', 2], ['dtheta_st4_ast', 2], ['b_st4', 2], ['dtheta_st4_c', 2], ['a_st5', 2], ['theta_st5_0', 2],
    ['dtheta_st5_ast', 2], ['b_st5', 2], ['dtheta_st5_c', 2], ['a_st6', 2], ['theta_st6_0', 2],
    ['dtheta_st6_ast', 2], ['b_st6', 2], ['dtheta_st6_c', 2], ['a_st1', 2], ['cosphi_st1_ast', 2], ['b_st1', 2],
    ['cosphi_st1_c', 2], ['a_st2', 2], ['cosphi_st2_ast', 2], ['b_st2', 2], ['cosphi_st2_c', 2]
  ],
  // CG-DNA/pair_oxdna_xstk.cpp
  'oxdna/xstk': [
    ['k_xst', 2], ['cut_xst_0', 2], ['cut_xst_c', 2], ['cut_xst_lo', 2], ['cut_xst_hi', 2], ['cut_xst_lc', 2],
    ['cut_xst_hc', 2], ['b_xst_lo', 2], ['b_xst_hi', 2], ['a_xst1', 2], ['theta_xst1_0', 2], ['dtheta_xst1_ast', 2],
    ['b_xst1', 2], ['dtheta_xst1_c', 2], ['a_xst2', 2], ['theta_xst2_0', 2], ['dtheta_xst2_ast', 2], ['b_xst2', 2],
    ['dtheta_xst2_c', 2], ['a_xst3', 2], ['theta_xst3_0', 2], ['dtheta_xst3_ast', 2], ['b_xst3', 2],
    ['dtheta_xst3_c', 2], ['a_xst4', 2], ['theta_xst4_0', 2], ['dtheta_xst4_ast', 2], ['b_xst4', 2],
    ['dtheta_xst4_c', 2], ['a_xst7', 2], ['theta_xst7_0', 2], ['dtheta_xst7_ast', 2], ['b_xst7', 2],
    ['dtheta_xst7_c', 2], ['a_xst8', 2], ['theta_xst8_0', 2], ['dtheta_xst8_ast', 2], ['b_xst8', 2],
    ['dtheta_xst8_c', 2]
  ],
  // CG-DNA/pair_oxdna_xstk.cpp
  'oxdna2/xstk': [
    ['k_xst', 2], ['cut_xst_0', 2], ['cut_xst_c', 2], ['cut_xst_lo', 2], ['cut_xst_hi', 2], ['cut_xst_lc', 2],
    ['cut_xst_hc', 2], ['b_xst_lo', 2], ['b_xst_hi', 2], ['a_xst1', 2], ['theta_xst1_0', 2], ['dtheta_xst1_ast', 2],
    ['b_xst1', 2], ['dtheta_xst1_c', 2], ['a_xst2', 2], ['theta_xst2_0', 2], ['dtheta_xst2_ast', 2], ['b_xst2', 2],
    ['dtheta_xst2_c', 2], ['a_xst3', 2], ['theta_xst3_0', 2], ['dtheta_xst3_ast', 2], ['b_xst3', 2],
    ['dtheta_xst3_c', 2], ['a_xst4', 2], ['theta_xst4_0', 2], ['dtheta_xst4_ast', 2], ['b_xst4', 2],
    ['dtheta_xst4_c', 2], ['a_xst7', 2], ['theta_xst7_0', 2], ['dtheta_xst7_ast', 2], ['b_xst7', 2],
    ['dtheta_xst7_c', 2], ['a_xst8', 2], ['theta_xst8_0', 2], ['dtheta_xst8_ast', 2], ['b_xst8', 2],
    ['dtheta_xst8_c', 2]
  ],
  // CG-DNA/pair_oxdna2_coaxstk.cpp
  'oxdna2/coaxstk': [
    ['k_cxst', 2], ['cut_cxst_0', 2], ['cut_cxst_c', 2], ['cut_cxst_lo', 2], ['cut_cxst_hi', 2], ['cut_cxst_lc', 2],
    ['cut_cxst_hc', 2], ['b_cxst_lo', 2], ['b_cxst_hi', 2], ['a_cxst1', 2], ['theta_cxst1_0', 2],
    ['dtheta_cxst1_ast', 2], ['b_cxst1', 2], ['dtheta_cxst1_c', 2], ['a_cxst4', 2], ['theta_cxst4_0', 2],
    ['dtheta_cxst4_ast', 2], ['b_cxst4', 2], ['dtheta_cxst4_c', 2], ['a_cxst5', 2], ['theta_cxst5_0', 2],
    ['dtheta_cxst5_ast', 2], ['b_cxst5', 2], ['dtheta_cxst5_c', 2], ['a_cxst6', 2], ['theta_cxst6_0', 2],
    ['dtheta_cxst6_ast', 2], ['b_cxst6', 2], ['dtheta_cxst6_c', 2], ['AA_cxst1', 2], ['BB_cxst1', 2]
  ],
  // CG-DNA/pair_oxdna2_coaxstk.cpp
  'oxdna3/coaxstk': [
    ['k_cxst', 2], ['cut_cxst_0', 2], ['cut_cxst_c', 2], ['cut_cxst_lo', 2], ['cut_cxst_hi', 2], ['cut_cxst_lc', 2],
    ['cut_cxst_hc', 2], ['b_cxst_lo', 2], ['b_cxst_hi', 2], ['a_cxst1', 2], ['theta_cxst1_0', 2],
    ['dtheta_cxst1_ast', 2], ['b_cxst1', 2], ['dtheta_cxst1_c', 2], ['a_cxst4', 2], ['theta_cxst4_0', 2],
    ['dtheta_cxst4_ast', 2], ['b_cxst4', 2], ['dtheta_cxst4_c', 2], ['a_cxst5', 2], ['theta_cxst5_0', 2],
    ['dtheta_cxst5_ast', 2], ['b_cxst5', 2], ['dtheta_cxst5_c', 2], ['a_cxst6', 2], ['theta_cxst6_0', 2],
    ['dtheta_cxst6_ast', 2], ['b_cxst6', 2], ['dtheta_cxst6_c', 2], ['AA_cxst1', 2], ['BB_cxst1', 2]
  ],
  // CG-DNA/pair_oxdna2_dh.cpp
  'oxdna2/dh': [['kappa_dh', 2], ['qeff_dh_pf', 2], ['b_dh', 2], ['cut_dh_ast', 2], ['cut_dh_c', 2]],
  // CG-DNA/pair_oxdna2_dh.cpp
  'oxdna3/dh': [['kappa_dh', 2], ['qeff_dh_pf', 2], ['b_dh', 2], ['cut_dh_ast', 2], ['cut_dh_c', 2]],
  // CG-DNA/pair_oxdna2_dh.cpp
  'oxrna2/dh': [['kappa_dh', 2], ['qeff_dh_pf', 2], ['b_dh', 2], ['cut_dh_ast', 2], ['cut_dh_c', 2]],
  // CG-DNA/pair_oxdna3_xstk.cpp
  'oxdna3/xstk': [
    ['k_xst', 2], ['cut_xst_0_33', 2], ['cut_xst_c_33', 2], ['cut_xst_lo_33', 2], ['cut_xst_hi_33', 2],
    ['cut_xst_lc_33', 2], ['cut_xst_hc_33', 2], ['cut_xst_0_55', 2], ['cut_xst_c_55', 2], ['cut_xst_lo_55', 2],
    ['cut_xst_hi_55', 2], ['cut_xst_lc_55', 2], ['cut_xst_hc_55', 2], ['b_xst_lo', 2], ['b_xst_hi', 2],
    ['a_xst1', 2], ['theta_xst1_0', 2], ['dtheta_xst1_ast', 2], ['b_xst1', 2], ['dtheta_xst1_c', 2], ['a_xst2', 2],
    ['theta_xst2_0', 2], ['dtheta_xst2_ast', 2], ['b_xst2', 2], ['dtheta_xst2_c', 2], ['a_xst3', 2],
    ['theta_xst3_0', 2], ['dtheta_xst3_ast', 2], ['b_xst3', 2], ['dtheta_xst3_c', 2], ['a_xst7', 2],
    ['theta_xst7_0_33', 2], ['theta_xst7_0_55', 2], ['dtheta_xst7_ast', 2], ['b_xst7', 2], ['dtheta_xst7_c', 2],
    ['a_xst8', 2], ['theta_xst8_0_33', 2], ['theta_xst8_0_55', 2], ['dtheta_xst8_ast', 2], ['b_xst8', 2],
    ['dtheta_xst8_c', 2]
  ],
  // CG-DNA/pair_oxrna2_stk.cpp
  'oxrna2/stk': [
    ['epsilon_st', 2], ['a_st', 2], ['cut_st_0', 2], ['cut_st_c', 2], ['cut_st_lo', 2], ['cut_st_hi', 2],
    ['cut_st_lc', 2], ['cut_st_hc', 2], ['b_st_lo', 2], ['b_st_hi', 2], ['shift_st', 2], ['a_st5', 2],
    ['theta_st5_0', 2], ['dtheta_st5_ast', 2], ['b_st5', 2], ['dtheta_st5_c', 2], ['a_st6', 2], ['theta_st6_0', 2],
    ['dtheta_st6_ast', 2], ['b_st6', 2], ['dtheta_st6_c', 2], ['a_st9', 2], ['theta_st9_0', 2],
    ['dtheta_st9_ast', 2], ['b_st9', 2], ['dtheta_st9_c', 2], ['a_st10', 2], ['theta_st10_0', 2],
    ['dtheta_st10_ast', 2], ['b_st10', 2], ['dtheta_st10_c', 2], ['a_st1', 2], ['cosphi_st1_ast', 2], ['b_st1', 2],
    ['cosphi_st1_c', 2], ['a_st2', 2], ['cosphi_st2_ast', 2], ['b_st2', 2], ['cosphi_st2_c', 2]
  ],
  // CG-DNA/pair_oxrna2_xstk.cpp
  'oxrna2/xstk': [
    ['k_xst', 2], ['cut_xst_0', 2], ['cut_xst_c', 2], ['cut_xst_lo', 2], ['cut_xst_hi', 2], ['cut_xst_lc', 2],
    ['cut_xst_hc', 2], ['b_xst_lo', 2], ['b_xst_hi', 2], ['a_xst1', 2], ['theta_xst1_0', 2], ['dtheta_xst1_ast', 2],
    ['b_xst1', 2], ['dtheta_xst1_c', 2], ['a_xst2', 2], ['theta_xst2_0', 2], ['dtheta_xst2_ast', 2], ['b_xst2', 2],
    ['dtheta_xst2_c', 2], ['a_xst3', 2], ['theta_xst3_0', 2], ['dtheta_xst3_ast', 2], ['b_xst3', 2],
    ['dtheta_xst3_c', 2], ['a_xst7', 2], ['theta_xst7_0', 2], ['dtheta_xst7_ast', 2], ['b_xst7', 2],
    ['dtheta_xst7_c', 2], ['a_xst8', 2], ['theta_xst8_0', 2], ['dtheta_xst8_ast', 2], ['b_xst8', 2],
    ['dtheta_xst8_c', 2]
  ],
  // ML-PACE/pair_pace.cpp
  'pace': [['corerep_flag', 0], ['scale', 2]],
  // APIP/pair_pace_apip.cpp
  'pace/apip': [['corerep_flag', 0], ['scale', 2]],
  // APIP/pair_pace_fast_apip.cpp
  'pace/fast/apip': [['scale', 2]],
  // APIP/pair_pace_precise_apip.cpp
  'pace/precise/apip': [['scale', 2]],
  // EXTRA-PAIR/pair_pedone.cpp
  'pedone': [['c0', 2, 'B0'], ['d0', 2], ['r0', 2], ['alpha', 2]],
  // REAXFF/pair_reaxff.cpp
  'reaxff': [['chi', 1], ['eta', 1], ['gamma', 1], ['bcut_acks2', 1], ['bond_softness', 1]],
  // ML-SNAP/pair_snap.cpp
  'snap': [['scale', 2]],
  // pair_soft.cpp
  'soft': [['a', 2]],
  // SPIN/pair_spin_dipole_cut.cpp
  'spin/dipole/cut': [['cut', 0], ['cut_coul', 0], ['ewald_order', 0], ['ewald_mix', 0]],
  // SPIN/pair_spin_dipole_long.cpp
  'spin/dipole/long': [['cut', 0], ['cut_coul', 0], ['ewald_order', 0], ['ewald_mix', 0]],
  // SPIN/pair_spin_dmi.cpp
  'spin/dmi': [['cut', 0]],
  // SPIN/pair_spin_exchange.cpp
  'spin/exchange': [['cut', 0]],
  // SPIN/pair_spin_exchange_biquadratic.cpp
  'spin/exchange/biquadratic': [['cut', 0]],
  // SPIN/pair_spin_magelec.cpp
  'spin/magelec': [['cut', 0]],
  // SPIN/pair_spin_neel.cpp
  'spin/neel': [['cut', 0]],
  // DRUDE/pair_thole.cpp
  'thole': [['scale', 2], ['polar', 2], ['thole', 2], ['ascreen', 2]],
  // MOLECULE/pair_tip4p_cut.cpp
  'tip4p/cut': [['qdist', 0], ['typeO', 0], ['typeH', 0], ['typeA', 0], ['typeB', 0], ['cut_coul', 0]],
  // KSPACE/pair_tip4p_long.cpp
  'tip4p/long': [['qdist', 0], ['typeO', 0], ['typeH', 0], ['typeA', 0], ['typeB', 0], ['cut_coul', 0]],
  // FEP/pair_tip4p_long_soft.cpp
  'tip4p/long/soft': [
    ['qdist', 0], ['typeO', 0], ['typeH', 0], ['typeA', 0], ['typeB', 0], ['cut_coul', 0], ['lambda', 2]
  ],
  // EXTRA-PAIR/pair_ufm.cpp
  'ufm': [['epsilon', 2], ['sigma', 2], ['scale', 2]],
  // EXTRA-PAIR/pair_wf_cut.cpp
  'wf/cut': [['epsilon', 2], ['sigma', 2], ['nu', 2], ['mu', 2]],
  // pair_yukawa.cpp
  'yukawa': [['alpha', 2, 'a']],
  // pair_yukawa.cpp
  'yukawa/colloid': [['alpha', 2, 'A']],
  // pair_zero_coul.cpp
  'zero/coul': [['cut_coul', 0]],
};
