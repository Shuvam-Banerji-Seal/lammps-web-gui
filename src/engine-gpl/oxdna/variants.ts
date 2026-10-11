/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the oxDNA2 and oxRNA2 subclasses of the
 * CG-DNA bond and excluded-volume styles, src/CG-DNA/bond_oxdna2_fene.h, bond_oxrna2_fene.h, pair_oxdna2_excv.h
 * and pair_oxrna2_excv.h. Each class only overrides compute_backbone_site with the backbone rule of the
 * nucleotide model (NucleotideOxdna2::backbone_site, NucleotideOxrna2::backbone_site in nucleotide_oxdna.h).
 * Original authors (from the headers): Oliver Henrich (University of Strathclyde, Glasgow) and Kierran Falloon
 * (University of Strathclyde, Glasgow); described by Snodin et al. (oxDNA2) and Sulc et al. (oxRNA2). LAMMPS is
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { BondOxdnaFene } from './bond_fene';
import { PairOxdnaExcv } from './pair_excv';
import { type OxdnaConstants, backboneOxdna2, backboneOxrna2 } from './common';

/** bond_style oxdna2/fene (BondOxdna2Fene). */
export class BondOxdna2Fene extends BondOxdnaFene {
  override readonly name = 'oxdna2/fene';
  protected override backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxdna2(c, e1, e2, e3, out);
  }
}

/** bond_style oxrna2/fene (BondOxrna2Fene). */
export class BondOxrna2Fene extends BondOxdnaFene {
  override readonly name = 'oxrna2/fene';
  protected override backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxrna2(c, e1, e2, e3, out);
  }
}

/** pair_style oxdna2/excv (PairOxdna2Excv). */
export class PairOxdna2Excv extends PairOxdnaExcv {
  override readonly name = 'oxdna2/excv';
  protected override backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxdna2(c, e1, e2, e3, out);
  }
}

/** pair_style oxdna2/excv (PairOxrna2Excv). */
export class PairOxrna2Excv extends PairOxdnaExcv {
  override readonly name = 'oxrna2/excv';
  protected override backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxrna2(c, e1, e2, e3, out);
  }
}
