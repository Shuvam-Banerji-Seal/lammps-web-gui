/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_saip_metal.cpp and src/INTERLAYER/pair_saip_metal.h, at commit b332f28 (the metal and
 * 2D-atom branches of calc_FRep; the normals and the attractive part are inherited from ilp/graphene/hbn, as native
 * inherits them). Contributing author: Wengen Ouyang (Wuhan University). Copyright (2003) Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { PairILPGrapheneHBN } from './ilp_graphene_hbn';

/**
 * pair_style saip/metal (docs.lammps.org/pair_saip_metal.html): the registry-dependent potential for gold on graphitic
 * systems. Atoms of an element other than C, H, B and N (the metal) interact through the isotropic repulsion; the 2D
 * atoms keep the transverse (normal) form of ilp/graphene/hbn. Usage: pair_style saip/metal cutoff [tap_flag] inside
 * pair_style hybrid/overlay.
 */
export class PairSAIPMetal extends PairILPGrapheneHBN {
  override readonly name: string = 'saip/metal';

  constructor() {
    super();
    this.metalTest = (element) => element !== 'C' && element !== 'H' && element !== 'B' && element !== 'N';
  }
}
