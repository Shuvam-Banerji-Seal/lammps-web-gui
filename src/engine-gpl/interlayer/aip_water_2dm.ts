/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_aip_water_2dm.cpp and src/INTERLAYER/pair_aip_water_2dm.h, at commit b332f28 (the style
 * is the TMD variant of the shared ILP code with six neighbors and the water normals of ./ilp_tmd.ts). Contributing
 * author: Wengen Ouyang (Wuhan University). Copyright (2003) Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { PairILPTMD } from './ilp_tmd';

/**
 * pair_style aip/water/2dm (docs.lammps.org/pair_aip_water_2dm.html): the anisotropic interlayer potential for water on
 * 2D materials (graphene, hBN). The 2D atoms take the normals of ilp/tmd and the water O and H atoms their own normals.
 * Usage: pair_style aip/water/2dm cutoff [tap_flag] inside pair_style hybrid/overlay.
 */
export class PairAIPWater2DM extends PairILPTMD {
  override readonly name: string = 'aip/water/2dm';
}
