/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the ACE descriptor of the ML-IAP
 * package (src/ML-IAP/mliap_descriptor_ace.cpp) with the ML-PACE library v.2025.12.4.p1 (see ../pace/mliap_ace.ts).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_mliap_ace, kind HOOKS, word 'ace'):
 * pair_style mliap ... descriptor ace file reads it with gplHook('ace').
 */
export { HOOKS } from '../pace/mliap_ace';
