/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/fix_tgnvt_drude.cpp and
 * src/DRUDE/fix_tgnvt_drude.h (class FixTGNVTDrude, fix tgnvt/drude). Copyright (2003) Sandia Corporation; GPL-2.0
 * (src/engine-gpl/LICENSE). DRUDE package authors: Alain Dequidt, Julien Devemy, Agilio Padua; contributors
 * Kateryna Goloviznina and Zheng Gong.
 *
 * fix tgnvt/drude: the temperature-grouped NVT integrator of the Drude model (fix_tgnh_drude.ts holds the chains).
 */
import type { System } from '../../engine/system';
import { FixTGNHDrude } from './fix_tgnh_drude';

export class FixTGNVTDrude extends FixTGNHDrude {
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'tgnvt/drude');
  }
}
