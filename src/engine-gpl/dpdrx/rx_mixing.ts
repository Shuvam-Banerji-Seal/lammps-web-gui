/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): PairTableRX::getMixingWeights in
 * src/DPD-REACT/pair_table_rx.cpp, and the same function in PairMultiLucyRX::getMixingWeights (src/DPD-REACT/
 * pair_multi_lucy_rx.cpp; the two are identical). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the
 * GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing
 * author (from pair_table_rx.cpp): Paul Crozier (SNL); from pair_multi_lucy_rx.cpp: James Larentzos and Joshua Moore
 * (U.S. Army Research Laboratory).
 *
 * docs.lammps.org/pair_table_rx.html: "The interaction potential is weighted by the geometric average of either the mole
 * fraction concentrations or the number of molecules associated with the interacting coarse-grained particles (see the
 * fractional or molecular weighting pair style options)." The site "1fluid" is the one-fluid approximation: its weight is
 * the sum over the species that are not the other site, with 1 minus that sum for the number of molecules.
 */
import { MY_EPSILON } from './fix_rx';
import { StyleError } from '../../engine/force/types';

/** Number of mixing weights per atom: site1 old, site2 old, site1 new, site2 new. */
export const MIX_WEIGHTS = 4;

/**
 * The four weights of atom id (native mixWtSite1old, mixWtSite2old, mixWtSite1, mixWtSite2), written to out[b..b+3].
 * cur and old are the per-species concentrations (current and at the start of the step), indexed by species; isite1 and
 * isite2 are species indices or -1 for 1fluid. fractional selects the mole fractions (true) or the molecule numbers.
 */
export const mixingWeights = (
  id: number, cur: Float64Array[], old: Float64Array[], isite1: number, isite2: number, fractional: boolean,
  out: Float64Array, b: number,
): void => {
  const ns = cur.length;
  let nTotal = 0.0, nTotalOld = 0.0;
  for (let k = 0; k < ns; k++) {
    nTotal += cur[k][id];
    nTotalOld += old[k][id];
  }
  if (nTotal < MY_EPSILON || nTotalOld < MY_EPSILON) {
    throw new StyleError('The number of molecules in CG particle is less than 10*DBL_EPSILON.');
  }
  let nMoleculesOld1 = 0.0, nMolecules1 = 0.0, fractionOld1 = 0.0, fraction1 = 0.0;
  let nMoleculesOld2 = 0.0, nMolecules2 = 0.0, fractionOld2 = 0.0, fraction2 = 0.0;
  if (isite1 !== -1) {
    nMoleculesOld1 = old[isite1][id];
    nMolecules1 = cur[isite1][id];
    fractionOld1 = nMoleculesOld1 / nTotalOld;
    fraction1 = nMolecules1 / nTotal;
  }
  if (isite2 !== -1) {
    nMoleculesOld2 = old[isite2][id];
    nMolecules2 = cur[isite2][id];
    fractionOld2 = nMoleculesOld2 / nTotalOld;
    fraction2 = nMolecules2 / nTotal;
  }
  if (isite1 === -1 || isite2 === -1) {
    // the species other than the named sites make up the one-fluid mixture
    let nOFAold = 0.0, nOFA = 0.0, fOFAold = 0.0, fOFA = 0.0;
    for (let k = 0; k < ns; k++) {
      if (isite1 === k || isite2 === k) continue;
      nOFAold += old[k][id];
      nOFA += cur[k][id];
      fOFAold += old[k][id] / nTotalOld;
      fOFA += cur[k][id] / nTotal;
    }
    if (isite1 === -1) {
      nMoleculesOld1 = 1.0 - (nTotalOld - nOFAold);
      nMolecules1 = 1.0 - (nTotal - nOFA);
      fractionOld1 = fOFAold;
      fraction1 = fOFA;
    }
    if (isite2 === -1) {
      nMoleculesOld2 = 1.0 - (nTotalOld - nOFAold);
      nMolecules2 = 1.0 - (nTotal - nOFA);
      fractionOld2 = fOFAold;
      fraction2 = fOFA;
    }
  }
  if (fractional) {
    out[b] = fractionOld1;
    out[b + 1] = fractionOld2;
    out[b + 2] = fraction1;
    out[b + 3] = fraction2;
  } else {
    out[b] = nMoleculesOld1;
    out[b + 1] = nMoleculesOld2;
    out[b + 2] = nMolecules1;
    out[b + 3] = nMolecules2;
  }
};
