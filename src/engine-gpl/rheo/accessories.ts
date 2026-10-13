/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/RHEO/fix_rheo_viscosity.cpp (FixRHEOViscosity: fix rheo/viscosity) and src/RHEO/fix_rheo_pressure.cpp
 * (FixRHEOPressure: fix rheo/pressure), at commit b332f28. RHEO package authors: Joel Clemmer (SNL), Thomas O'Connor
 * (CMU), Eric Palermo (CMU). Copyright Sandia Corporation (GNU General Public License); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_rheo_viscosity.html): "fix ID group-ID rheo/viscosity type1 pstyle1 args1 ... typeN pstyleN
 * argsN" with "constant args = eta" (the viscosity of each type) and "power args = eta, gd0, K, n" (refused by name:
 * not supported by the browser engine yet). Docs (docs.lammps.org/fix_rheo_pressure.html): "fix ID group-ID
 * rheo/pressure type1 pstyle1 args1 ... typeN pstyleN argsN" with "linear" (the other pressure styles are refused by
 * name). The linear pressure is csq (rho - rho0).
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { StyleError, typeBounds } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { RHEO_PRESSURE, RHEO_RHO, RHEO_VISC } from '../../engine/atoms';
import { rheoModelOf, rheoProp, type RheoModel } from './model';

/** The shared state of fix rheo, which must be defined before an accessory fix (fix rheo comes first in the input). */
const needModel = (sys: System, style: string): RheoModel => {
  const m = rheoModelOf(sys.state);
  if (!m) throw new StyleError(`Need to define fix rheo to use fix ${style}`);
  return m;
};

/** fix rheo/viscosity: the viscosity of each type (constant), set on every atom. */
export class FixRHEOViscosity extends Fix {
  readonly style: string = 'rheo/viscosity';
  private readonly eta: Float64Array;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError('fix rheo/viscosity command requires group all');
    if (sys.fixes.some((f) => f.style === 'rheo/viscosity')) throw new StyleError('Can only specify one instance of fix rheo/viscosity');
    const n = sys.state.ntypes;
    this.eta = new Float64Array(n + 1);
    const set = new Uint8Array(n + 1);
    let iarg = 0;
    while (iarg < args.length) {
      const [lo, hi] = typeBounds(args[iarg], n);
      if (iarg + 1 >= args.length) throw new StyleError('Illegal fix rheo/viscosity command: missing style');
      if (args[iarg + 1] === 'constant') {
        if (iarg + 2 >= args.length) throw new StyleError('Illegal fix rheo/viscosity constant: missing viscosity');
        const eta = parseNum(args[iarg + 2], 'viscosity');
        if (eta < 0.0) throw new StyleError('The viscosity must be positive');
        for (let t = lo; t <= hi; t++) { this.eta[t] = eta; set[t] = 1; }
        iarg += 3;
      } else if (args[iarg + 1] === 'power') {
        throw new StyleError('fix rheo/viscosity power is not supported by the browser engine yet (constant only)');
      } else {
        throw new StyleError(`Illegal fix rheo/viscosity style: ${args[iarg + 1]}`);
      }
    }
    for (let t = 1; t <= n; t++) {
      if (!set[t]) throw new StyleError(`Must specify viscosity for atom type ${t} in fix/rheo/viscosity`);
    }
    needModel(sys, this.style);
  }

  setupPreForce(): void { this.preForce(); }

  /** Viscosity of each atom from its type (constant styles only; the value does not change during a run). */
  preForce(): void {
    const s = this.sys.state;
    const visc = rheoProp(s, RHEO_VISC);
    for (let i = 0; i < s.n; i++) visc[i] = this.eta[s.type[i]];
  }
}

/** fix rheo/pressure: the linear equation of state p = csq (rho - rho0) for each type. */
export class FixRHEOPressure extends Fix {
  readonly style: string = 'rheo/pressure';
  private readonly model: RheoModel;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError('fix rheo/pressure command requires group all');
    if (sys.fixes.some((f) => f.style === 'rheo/pressure')) throw new StyleError('Can only specify one instance of fix rheo/pressure');
    const n = sys.state.ntypes;
    const set = new Uint8Array(n + 1);
    let iarg = 0;
    while (iarg < args.length) {
      const [lo, hi] = typeBounds(args[iarg], n);
      if (iarg + 1 >= args.length) throw new StyleError('Illegal fix rheo/pressure command: missing style');
      if (args[iarg + 1] === 'linear') {
        for (let t = lo; t <= hi; t++) set[t] = 1;
        iarg += 2;
      } else {
        throw new StyleError(`fix rheo/pressure style ${args[iarg + 1]} is not supported by the browser engine yet (linear only)`);
      }
    }
    for (let t = 1; t <= n; t++) {
      if (!set[t]) throw new StyleError(`Must specify pressure for atom type ${t} in fix/rheo/pressure`);
    }
    this.model = needModel(sys, this.style);
  }

  setupPreForce(): void { this.preForce(); }

  /** Pressure of each group atom from its density (linear equation of state). */
  preForce(): void {
    const s = this.sys.state;
    const m = this.model;
    const rho = rheoProp(s, RHEO_RHO), pressure = rheoProp(s, RHEO_PRESSURE);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const t = s.type[i];
      pressure[i] = m.csq[t] * (rho[i] - m.rho0[t]);
    }
  }
}
