/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the DPD-MESO package's style registry
 * (src/DPD-MESO: pair_edpd, pair_mdpd, pair_mdpd_rhosum, pair_tdpd, fix_mvv_dpd, fix_mvv_edpd, fix_mvv_tdpd, fix_edpd_source,
 * fix_tdpd_source, compute_edpd_temp_atom, compute_tdpd_cc_atom) at commit b332f28. Contributing author: Zhen Li (Clemson
 * University). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_dpdmeso; the names listed there must match these keys).
 */
import type { FixFactory, ComputeFactory } from '../../engine/styles';
import type { Pair } from '../../engine/force/types';
import { PairEDPD } from '../dpdmeso/pair_edpd';
import { PairMDPD, PairMDPDRhoSum } from '../dpdmeso/pair_mdpd';
import { PairTDPD } from '../dpdmeso/pair_tdpd';
import { FixMvvDPD, FixMvvEDPD, FixMvvTDPD } from '../dpdmeso/fix_mvv';
import { FixEDPDSource, FixTDPDSource } from '../dpdmeso/fix_source';
import { ComputeEDPDTempAtom, ComputeTDPDCCAtom } from '../dpdmeso/compute_meso';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  edpd: () => new PairEDPD(),
  mdpd: () => new PairMDPD(),
  'mdpd/rhosum': () => new PairMDPDRhoSum(),
  tdpd: () => new PairTDPD(),
};

/** fix styles. */
export const FIXES: Record<string, FixFactory> = {
  'mvv/dpd': (sys, id, group, args) => new FixMvvDPD(sys, id, group, args),
  'mvv/edpd': (sys, id, group, args) => new FixMvvEDPD(sys, id, group, args),
  'mvv/tdpd': (sys, id, group, args) => new FixMvvTDPD(sys, id, group, args),
  'edpd/source': (sys, id, group, args) => new FixEDPDSource(sys, id, group, args),
  'tdpd/source': (sys, id, group, args) => new FixTDPDSource(sys, id, group, args),
};

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'edpd/temp/atom': (sys, id, group, args) => new ComputeEDPDTempAtom(sys, id, group, args),
  'tdpd/cc/atom': (sys, id, group, args) => new ComputeTDPDCCAtom(sys, id, group, args),
};
