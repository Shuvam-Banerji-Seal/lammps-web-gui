/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/fix_drude.cpp and
 * src/DRUDE/fix_drude.h (class FixDrude, fix drude). Copyright (2003) Sandia Corporation; GPL-2.0 (src/engine-gpl/LICENSE).
 * DRUDE package authors: Alain Dequidt (Clermont University, CNRS), Julien Devemy and Agilio Padua (ENS de Lyon);
 * contributors Kateryna Goloviznina and Zheng Gong.
 *
 * fix drude ID group drude type1 ... typeN: tags the types as non-polarizable (n), cores (c) or Drude particles (d),
 * and rebuilds the special neighbor lists for them (see drude_model.ts). Registered through registry/drude.ts.
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';
import type { SpecialList } from '../../engine/atoms';
import { StyleError } from '../../engine/force/types';
import { DrudeModel, parseDrudeTypes, setDrudeModel } from './drude_model';

export class FixDrude extends Fix {
  readonly style = 'drude';
  readonly model: DrudeModel;
  private readonly hook: (list: SpecialList, s: SimState) => SpecialList;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (sys.fixes.some((f) => f.style === 'drude')) throw new StyleError('More than one fix drude');
    const s = sys.state;
    const kind = parseDrudeTypes(args, s.ntypes);
    this.model = new DrudeModel(s, kind);
    setDrudeModel(s, this.model);
    this.hook = (list, st) => this.model.transformSpecial(list, st);
    sys.ff.specialHook = this.hook;
  }

  override destroy(): void {
    if (this.sys.ff.specialHook === this.hook) this.sys.ff.specialHook = null;
    setDrudeModel(this.sys.state, null);
  }
}
