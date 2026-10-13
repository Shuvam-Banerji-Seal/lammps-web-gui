/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_saip_metal_tmd.cpp and src/INTERLAYER/pair_saip_metal_tmd.h, at commit b332f28 (the
 * metal and 2D-atom branches of calc_FRep, on the TMD normals of ./ilp_tmd.ts). Contributing author: Wengen Ouyang
 * (Wuhan University). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 */
import { PairILPTMD } from './ilp_tmd';

/**
 * pair_style saip/metal/tmd (docs.lammps.org/pair_saip_metal.html): the saip/metal potential for the interface of
 * metals (Au, Cu, Ag, Ru, Pt, Ni) with transition-metal dichalcogenides. The metal atoms use the isotropic repulsion,
 * the other atoms the TMD normals of ilp/tmd. Usage: pair_style saip/metal/tmd cutoff [tap_flag] inside
 * pair_style hybrid/overlay.
 */
export class PairSAIPMetalTMD extends PairILPTMD {
  override readonly name: string = 'saip/metal/tmd';

  constructor() {
    super();
    const metals = new Set(['Au', 'Cu', 'Ag', 'Ru', 'Pt', 'Ni']);
    this.metalTest = (element) => metals.has(element);
  }
}
