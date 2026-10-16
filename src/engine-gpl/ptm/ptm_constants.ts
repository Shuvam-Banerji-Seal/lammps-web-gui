/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_constants.h at
 * LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm).
 * The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Preprocessor constants of ptm_constants.h, ptm_fundamental_mappings.h and ptm_graph_data.h, with the C names.
 */

export const PTM_NO_ERROR = 0;

export const PTM_CHECK_FCC = 1 << 0;
export const PTM_CHECK_HCP = 1 << 1;
export const PTM_CHECK_BCC = 1 << 2;
export const PTM_CHECK_ICO = 1 << 3;
export const PTM_CHECK_SC = 1 << 4;
export const PTM_CHECK_DCUB = 1 << 5;
export const PTM_CHECK_DHEX = 1 << 6;
export const PTM_CHECK_GRAPHENE = 1 << 7;
export const PTM_CHECK_DEFAULT = PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO | PTM_CHECK_BCC;
export const PTM_CHECK_ALL = PTM_CHECK_SC | PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO | PTM_CHECK_BCC | PTM_CHECK_DCUB | PTM_CHECK_DHEX | PTM_CHECK_GRAPHENE;

export const PTM_MATCH_NONE = 0;
export const PTM_MATCH_FCC = 1;
export const PTM_MATCH_HCP = 2;
export const PTM_MATCH_BCC = 3;
export const PTM_MATCH_ICO = 4;
export const PTM_MATCH_SC = 5;
export const PTM_MATCH_DCUB = 6;
export const PTM_MATCH_DHEX = 7;
export const PTM_MATCH_GRAPHENE = 8;

export const PTM_ALLOY_NONE = 0;
export const PTM_ALLOY_PURE = 1;
export const PTM_ALLOY_L10 = 2;
export const PTM_ALLOY_L12_CU = 3;
export const PTM_ALLOY_L12_AU = 4;
export const PTM_ALLOY_B2 = 5;
export const PTM_ALLOY_SIC = 6;
export const PTM_ALLOY_BN = 7;

export const PTM_MAX_INPUT_POINTS = 19;
export const PTM_MAX_NBRS = 16;
export const PTM_MAX_POINTS = PTM_MAX_NBRS + 1;
export const PTM_MAX_FACETS = 28;
export const PTM_MAX_EDGES = 42;

export const PTM_NUM_NBRS_FCC = 12;
export const PTM_NUM_NBRS_HCP = 12;
export const PTM_NUM_NBRS_BCC = 14;
export const PTM_NUM_NBRS_ICO = 12;
export const PTM_NUM_NBRS_SC = 6;
export const PTM_NUM_NBRS_DCUB = 16;
export const PTM_NUM_NBRS_DHEX = 16;
export const PTM_NUM_NBRS_GRAPHENE = 9;

export const PTM_NUM_POINTS_FCC = PTM_NUM_NBRS_FCC + 1;
export const PTM_NUM_POINTS_HCP = PTM_NUM_NBRS_HCP + 1;
export const PTM_NUM_POINTS_BCC = PTM_NUM_NBRS_BCC + 1;
export const PTM_NUM_POINTS_ICO = PTM_NUM_NBRS_ICO + 1;
export const PTM_NUM_POINTS_SC = PTM_NUM_NBRS_SC + 1;
export const PTM_NUM_POINTS_DCUB = PTM_NUM_NBRS_DCUB + 1;
export const PTM_NUM_POINTS_DHEX = PTM_NUM_NBRS_DHEX + 1;
export const PTM_NUM_POINTS_GRAPHENE = PTM_NUM_NBRS_GRAPHENE + 1;

export const NUM_SC_GRAPHS = 1;
export const NUM_ICO_GRAPHS = 1;
export const NUM_FCC_GRAPHS = 8;
export const NUM_HCP_GRAPHS = 16;
export const NUM_BCC_GRAPHS = 218;
export const NUM_DCUB_GRAPHS = 12;
export const NUM_DHEX_GRAPHS = 24;

export const NUM_CUBIC_MAPPINGS = 24;
export const NUM_ICO_MAPPINGS = 60;
export const NUM_HEX_MAPPINGS = 6;
export const NUM_DCUB_MAPPINGS = 12;
export const NUM_DHEX_MAPPINGS = 3;
export const NUM_GRAPHENE_MAPPINGS = 6;
export const NUM_CONVENTIONAL_GRAPHENE_MAPPINGS = 12;
export const NUM_CONVENTIONAL_HEX_MAPPINGS = 12;
export const NUM_CONVENTIONAL_DCUB_MAPPINGS = 24;
export const NUM_CONVENTIONAL_DHEX_MAPPINGS = 12;

/** ptm_constants.cpp: ptm_num_nbrs[9] indexed by PTM_MATCH_* */
export const ptm_num_nbrs = [0, PTM_NUM_NBRS_FCC, PTM_NUM_NBRS_HCP, PTM_NUM_NBRS_BCC, PTM_NUM_NBRS_ICO, PTM_NUM_NBRS_SC, PTM_NUM_NBRS_DCUB, PTM_NUM_NBRS_DHEX, PTM_NUM_NBRS_GRAPHENE];

/** ptm_constants.cpp: MYSQRT* macros (C literals, same digits) */
export const MYSQRT2 = 1.41421356237309504880;
export const MYSQRT3 = 1.73205080756887729352;
export const MYSQRT5 = 2.23606797749978969640;
export const MYSQRT6 = 2.44948974278317809819;
export const MY5MINUSSQRT5BY10 = 0.52573111211913360602;
export const MY5PLUSSQRT5BY10 = 0.85065080835203993218;
