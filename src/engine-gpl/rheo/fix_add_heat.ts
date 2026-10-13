/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/fix_add_heat.cpp and fix_add_heat.h (FixAddHeat: fix add/heat), at commit b332f28, with the power
 * of four of src/math_special.cpp (MathSpecial::powint). Contributing author (from fix_add_heat.cpp): Joel Clemmer (SNL).
 * Copyright Sandia Corporation (GNU General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_add_heat.html): "fix ID group-ID add/heat style value keyword ..." with the styles "constant"
 * (value = heat added per unit time), "linear" (value = target temperature, prefactor = heat per unit temperature
 * difference) and "quartic" (the same with the fourth powers of the temperatures); the option "overwrite yes/no" resets the
 * heat flow of the group before the heat is added. Variable values (v_name) are refused by name (not supported by the
 * browser engine yet).
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { RHEO_HEATFLOW, RHEO_TEMP } from '../../engine/atoms';
import { rheoProp } from './model';

const ADD = 0, LINEAR = 1, QUARTIC = 2;

/** math_special.cpp powint(x, 4): the squares multiplied, as the native loop does. */
const pow4 = (x: number): number => {
  if (x === 0.0) return 0.0;
  const w = x * x;
  return 1.0 * (w * w);
};

export class FixAddHeat extends Fix {
  readonly style: string = 'add/heat';
  private readonly kind: number;
  private readonly value: number;
  private readonly prefactor: number;
  private readonly overwrite: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal fix add/heat command: missing arguments');
    const style = args[0];
    if (style === 'constant') this.kind = ADD;
    else if (style === 'linear') this.kind = LINEAR;
    else if (style === 'quartic') this.kind = QUARTIC;
    else throw new StyleError(`Invalid option ${style}`);
    if (args[1].startsWith('v_')) throw new StyleError('fix add/heat variable values (v_name) are not supported by the browser engine yet');
    this.value = parseNum(args[1], 'value');
    let iarg = 2;
    let prefactor = 0.0;
    if (this.kind !== ADD) {
      // fix_add_heat.cpp: "if (narg != 6) utils::missing_cmd_args" (the prefactor is the last argument)
      if (args.length !== 3) throw new StyleError('Illegal fix add/heat command: missing prefactor');
      prefactor = parseNum(args[2], 'prefactor');
      iarg = 3;
    }
    let overwrite = false;
    while (iarg < args.length) {
      if (args[iarg] === 'overwrite') {
        if (iarg + 1 >= args.length) throw new StyleError('Illegal fix add/heat command: missing overwrite value');
        if (args[iarg + 1] !== 'yes' && args[iarg + 1] !== 'no') throw new StyleError(`Expected boolean parameter instead of ${args[iarg + 1]} in fix add/heat`);
        overwrite = args[iarg + 1] === 'yes';
        iarg += 2;
      } else {
        throw new StyleError(`Illegal fix add/heat command, invalid argument ${args[iarg]}`);
      }
    }
    this.prefactor = prefactor;
    this.overwrite = overwrite;
  }

  init(): void {
    const s = this.sys.state;
    // "Fix add/heat requires atom style with temperature property" and "... with heatflow property"
    if (!s.custom.get(RHEO_TEMP)) throw new StyleError('Fix add/heat requires atom style with temperature property');
    if (!s.custom.get(RHEO_HEATFLOW)) throw new StyleError('Fix add/heat requires atom style with heatflow property');
  }

  /**
   * Base Fix::setup() is empty in native and FixAddHeat does not override it, so the heat is not added at the setup of a
   * run (only on the steps of the run).
   */
  setup(): void { /* no setup post_force, as native */ }

  postForce(): void {
    const s = this.sys.state;
    const heatflow = rheoProp(s, RHEO_HEATFLOW), temperature = rheoProp(s, RHEO_TEMP);
    if (this.overwrite) {
      for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) heatflow[i] = 0.0;
    }
    const vtmp = this.value;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (this.kind === ADD) heatflow[i] += vtmp;
      else if (this.kind === LINEAR) heatflow[i] += this.prefactor * (vtmp - temperature[i]);
      else heatflow[i] += this.prefactor * (pow4(vtmp) - pow4(temperature[i]));
    }
  }
}
