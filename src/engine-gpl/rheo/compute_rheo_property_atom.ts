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
 * phase (status & PHASECHECK) and coordination are supported; the thermal attributes (temperature, heatflow,
 * conductivity, energy on atom_style rheo/thermal, cv with fix rheo/thermal) read the RHEO thermal props.
 * surface/r (the surface distance, compute_rheo_surface.cpp) and nbond/shell (the formed shell bonds, needs fix
 * rheo/oxidation) are supported too: "surface/r" and "nbond/shell" are the native attribute names.
 * Other attributes (chi, surface/divr, shift/v, grad/v, stress/v, stress/t) are not supported
 * by the browser engine yet and are refused by name.
 */
import { PHASECHECK, rheoModelOf } from './model';
import { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { RHEO_COND, RHEO_ESPH, RHEO_HEATFLOW, RHEO_PRESSURE, RHEO_RHO, RHEO_STATUS, RHEO_TEMP, RHEO_VISC, isRheoStyle, isRheoThermalStyle } from '../../engine/atoms';
import { STATUS_SPLASH, STATUS_SURFACE } from './compute_rheo_surface';
import { shellNbond } from './model';

/** Keyword -> per-atom internal property (status is the short name of rheo_status). */
const KEYWORDS = new Map<string, string>([
  ['rho', RHEO_RHO], ['viscosity', RHEO_VISC], ['rheo_status', RHEO_STATUS], ['status', RHEO_STATUS],
  ['pressure', RHEO_PRESSURE],
]);
/** The thermal keywords (atom_style rheo/thermal): their internal property, and the energy keyword (esph of native). */
const THERMAL_KEYWORDS = new Map<string, string>([
  ['temperature', RHEO_TEMP], ['heatflow', RHEO_HEATFLOW], ['conductivity', RHEO_COND], ['energy', RHEO_ESPH],
]);

/** The normal component of the surface attribute ("surface/n/a", a = x, y, z). */
const NORMAL = new Map<string, number>([['surface/n/x', 0], ['surface/n/y', 1], ['surface/n/z', 2]]);

export class ComputeRHEOPropertyAtom extends Compute {
  readonly style: string = 'rheo/property/atom';
  /** Per attribute: an internal property name, 'surface', 'phase', 'coordination', 'cv' or a surface normal component index. */
  private readonly names: (string | 'surface' | 'cv' | 'phase' | 'coordination' | 'surface/r' | 'nbond/shell' | number)[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!isRheoStyle(sys.atomStyle)) throw new StyleError('compute rheo/property/atom requires atom_style rheo');
    if (args.length < 1) throw new StyleError('Illegal compute rheo/property/atom command');
    for (const a of args) {
      const thermalOk = THERMAL_KEYWORDS.has(a) && isRheoThermalStyle(sys.atomStyle);
      if (!KEYWORDS.has(a) && a !== 'surface' && a !== 'surface/r' && a !== 'nbond/shell' && !NORMAL.has(a) && !thermalOk && a !== 'cv' && a !== 'phase' && a !== 'coordination') {
        throw new StyleError(`Invalid keyword ${a} for atom style ${sys.atomStyle} in compute rheo/property/atom command (the browser engine supports ${[...KEYWORDS.keys(), ...(isRheoThermalStyle(sys.atomStyle) ? [...THERMAL_KEYWORDS.keys(), 'cv'] : []), 'surface', 'surface/r', 'nbond/shell', 'surface/n/x', 'surface/n/y', 'surface/n/z'].join(', ')})`);
      }
    }
    // compute_rheo_property_atom.cpp: the thermal attributes need fix rheo/thermal (checked at init in native; the cv
    // keyword is the specific heat of the type, read from the thermal fix when the values are computed)
    this.names = args.map((a) => (a === 'surface' || a === 'surface/r' || a === 'nbond/shell' || a === 'phase' || a === 'coordination' || a === 'cv' ? a : NORMAL.has(a) ? NORMAL.get(a)! : KEYWORDS.get(a) ?? THERMAL_KEYWORDS.get(a)!));
    this.peratomFlag = true;
    this.sizePeratomCols = args.length > 1 ? args.length : 0;
  }

  /** compute_rheo_property_atom.cpp init(): the attributes need their fixes (checked at each run, as native). */
  init(): void {
    const model = rheoModelOf(this.sys.state);
    // compute_rheo_property_atom.cpp: "Cannot request nbond/shell without fix rheo/oxidation"
    if (this.names.includes('nbond/shell') && !(model && model.oxidation)) throw new StyleError('Cannot request nbond/shell without fix rheo/oxidation');
  }

  /** The fix rheo/pressure whose equation of state gives the pressure (compute_rheo_property_atom.cpp fix_pressure). */
  private pressureFix(): { calcPressure(rho: number, i: number): number } {
    const f = this.sys.fixes.find((x) => x.style === 'rheo/pressure');
    if (!f) throw new StyleError('Cannot request pressure without fix rheo/pressure');
    return f as unknown as { calcPressure(rho: number, i: number): number };
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const m = this.names.length;
    const n = s.n;
    const out = new Float64Array(m > 1 ? m * n : n);
    const model = rheoModelOf(s);
    if (this.names.includes('cv') && !(model && model.thermal)) throw new StyleError('Cannot request thermal property without fix rheo/thermal');
    // compute_rheo_property_atom.cpp: "Cannot request nbond/shell without fix rheo/oxidation"
    if (this.names.includes('nbond/shell') && !(model && model.oxidation)) throw new StyleError('Cannot request nbond/shell without fix rheo/oxidation');
    if (this.names.some((k) => typeof k === 'number' || k === 'surface' || k === 'surface/r')) {
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
          } else if (name === 'surface/r') {
            // pack_surface_r: the surface distance of the group atoms (compute_rheo_surface.cpp rsurface)
            val = model!.rsurface[i];
          } else if (name === 'nbond/shell') {
            // pack_nbond_shell: the formed shell bonds (the internal per-atom property of bond rheo/shell)
            val = shellNbond(s)[i];
          } else if (name === 'cv') {
            val = model!.thermal!.calcCv(s.type[i]);
          } else if (name === 'phase') {
            // compute_rheo_property_atom.cpp pack_phase: "buf[n] = (status[i] & PHASECHECK)"
            val = s.custom.get(RHEO_STATUS)!.data[i] & PHASECHECK;
          } else if (name === 'coordination') {
            // pack_coordination: the neighbor count of the kernel (fix rheo computes it with RK1 or surface detection)
            val = model!.coord[i];
          } else if (typeof name === 'number') {
            val = model!.nsurface[i * model!.dim + name];
          } else if (name === RHEO_PRESSURE) {
            // pack_pressure: "fix_pressure->calc_pressure(rho[i], i)", from the density at the time of the output
            val = this.pressureFix().calcPressure(s.custom.get(RHEO_RHO)!.data[i], i);
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
