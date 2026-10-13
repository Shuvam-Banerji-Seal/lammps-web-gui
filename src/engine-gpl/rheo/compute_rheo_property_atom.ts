/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/RHEO/compute_rheo_property_atom.cpp and compute_rheo_property_atom.h (ComputeRHEOPropertyAtom: compute
 * rheo/property/atom) at commit b332f28. RHEO package authors: Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo
 * (CMU). Copyright Sandia Corporation (GNU General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/compute_rheo_property_atom.html): "compute ID group-ID rheo/property/atom input1 input2 ..."
 * with the attributes "viscosity, pressure, rho" (the browser engine also accepts status and rheo_status for the phase
 * status) and "surface, surface/n/a" (surface: "atom surface status"; surface/n/a: "a-component of surface normal
 * vector"). The surface values are 1 for a surface atom, 2 for a splash atom and 0 otherwise, on the group atoms.
 * Other attributes (phase, surface/r, surface/divr, coordination, energy, temperature, cv and the tensors) are not
 * supported by the browser engine yet and are refused by name.
 */
import { rheoModelOf } from './model';
import { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { RHEO_PRESSURE, RHEO_RHO, RHEO_STATUS, RHEO_VISC, isRheoStyle } from '../../engine/atoms';
import { STATUS_SPLASH, STATUS_SURFACE } from './compute_rheo_surface';

/** Keyword -> per-atom internal property (status is the short name of rheo_status). */
const KEYWORDS = new Map<string, string>([
  ['rho', RHEO_RHO], ['viscosity', RHEO_VISC], ['rheo_status', RHEO_STATUS], ['status', RHEO_STATUS],
  ['pressure', RHEO_PRESSURE],
]);

/** The normal component of the surface attribute ("surface/n/a", a = x, y, z). */
const NORMAL = new Map<string, number>([['surface/n/x', 0], ['surface/n/y', 1], ['surface/n/z', 2]]);

export class ComputeRHEOPropertyAtom extends Compute {
  readonly style: string = 'rheo/property/atom';
  /** Per attribute: an internal property name, 'surface' or a surface normal component index. */
  private readonly names: (string | 'surface' | number)[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!isRheoStyle(sys.atomStyle)) throw new StyleError('compute rheo/property/atom requires atom_style rheo');
    if (args.length < 1) throw new StyleError('Illegal compute rheo/property/atom command');
    for (const a of args) {
      if (!KEYWORDS.has(a) && a !== 'surface' && !NORMAL.has(a)) {
        throw new StyleError(`Invalid keyword ${a} for atom style rheo in compute rheo/property/atom command (the browser engine supports ${[...KEYWORDS.keys(), 'surface', 'surface/n/x', 'surface/n/y', 'surface/n/z'].join(', ')})`);
      }
    }
    this.names = args.map((a) => (a === 'surface' ? 'surface' : NORMAL.has(a) ? NORMAL.get(a)! : KEYWORDS.get(a)!));
    this.peratomFlag = true;
    this.sizePeratomCols = args.length > 1 ? args.length : 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const m = this.names.length;
    const n = s.n;
    const out = new Float64Array(m > 1 ? m * n : n);
    const model = rheoModelOf(s);
    if (this.names.some((k) => typeof k === 'number' || k === 'surface')) {
      // fix rheo surface/detection must be defined, as native (the surface values come from its status)
      if (!model || !model.surfaceFlag) throw new StyleError('Cannot request surface property without corresponding option in fix rheo');
    }
    for (let k = 0; k < m; k++) {
      const name = this.names[k];
      for (let i = 0; i < n; i++) {
        let val = 0;
        if (s.mask[i] & this.groupBit) {
          if (name === 'surface') {
            const st = s.custom.get(RHEO_STATUS)!.data[i];
            val = (st & STATUS_SURFACE) ? 1.0 : (st & STATUS_SPLASH) ? 2.0 : 0.0;
          } else if (typeof name === 'number') {
            val = model!.nsurface[i * model!.dim + name];
          } else {
            val = s.custom.get(name)!.data[i];
          }
        }
        if (m > 1) out[m * i + k] = val;
        else out[i] = val;
      }
    }
    if (m > 1) this.arrayAtom = out;
    else this.vectorAtom = out;
  }
}
