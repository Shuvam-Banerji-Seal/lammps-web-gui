/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/fix_tgnpt_drude.cpp and
 * src/DRUDE/fix_tgnpt_drude.h (class FixTGNPTDrude, fix tgnpt/drude). Copyright (2003) Sandia Corporation; GPL-2.0
 * (src/engine-gpl/LICENSE). DRUDE package authors: Alain Dequidt, Julien Devemy, Agilio Padua; contributors
 * Kateryna Goloviznina and Zheng Gong.
 *
 * fix tgnpt/drude: the temperature-grouped NPT integrator of the Drude model with the iso barostat (fix_tgnh_drude.ts
 * holds the chains and the barostat).
 */
import type { System } from '../../engine/system';
import { FixTGNHDrude } from './fix_tgnh_drude';

export class FixTGNPTDrude extends FixTGNHDrude {
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'tgnpt/drude');
  }
}
