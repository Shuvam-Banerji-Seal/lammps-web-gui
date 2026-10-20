/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/RHEO/compute_rheo_property_atom.cpp and compute_rheo_property_atom.h (ComputeRHEOPropertyAtom: compute
 * rheo/property/atom) at commit b332f28. RHEO package authors: Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo
 * (CMU). Copyright Sandia Corporation (GNU General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/compute_rheo_property_atom.html): "compute ID group-ID rheo/property/atom input1 input2 ...".
 * Every native attribute is read: phase, status (rheo_status), chi, surface, surface/r, surface/divr, surface/n/a,
 * coordination, pressure, cv, shift/v/a, grad/v/ab, stress/v/ab, stress/t/ab, energy, nbond/shell and the per-atom
 * properties of the atom style (rho, drho, viscosity, and temperature, esph, heatflow, conductivity on rheo/thermal).
 * A vector or tensor attribute ending in * gives all its components (dim or dim*dim columns). The surface values are
 * 1 for a surface atom, 2 for a splash atom and 0 otherwise. chi needs fix rheo interface/reconstruct.
 */
import { PHASECHECK, rheoModelOf } from './model';
import { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { RHEO_COND, RHEO_DRHO, RHEO_ESPH, RHEO_HEATFLOW, RHEO_PRESSURE, RHEO_RHO, RHEO_STATUS, RHEO_TEMP, RHEO_VISC, isRheoStyle, isRheoThermalStyle } from '../../engine/atoms';
import { STATUS_SPLASH, STATUS_SURFACE } from './compute_rheo_surface';
import { computeRheoCoordination } from './compute_rheo_kernel';
import { shellNbond } from './model';

/** AtomVecRHEO::property_atom and AtomVecRHEOThermal::property_atom: the per-atom properties of the atom style. */
const AVEC = new Map<string, string>([
  ['rheo_status', RHEO_STATUS], ['pressure', RHEO_PRESSURE], ['rho', RHEO_RHO], ['drho', RHEO_DRHO], ['viscosity', RHEO_VISC],
]);
const AVEC_THERMAL = new Map<string, string>([
  ['temperature', RHEO_TEMP], ['esph', RHEO_ESPH], ['heatflow', RHEO_HEATFLOW], ['conductivity', RHEO_COND],
]);

type Pack =
  | 'phase' | 'status' | 'chi' | 'surface' | 'surface/r' | 'surface/divr' | 'coordination' | 'pressure' | 'cv'
  | 'surface/n' | 'shift/v' | 'grad/v' | 'stress/v' | 'stress/t' | 'nbond/shell' | 'avec';

/** One output column: its packing function, its component (col_index) and its per-atom property (avec_index). */
interface Column { pack: Pack; index: number; prop: string }

export class ComputeRHEOPropertyAtom extends Compute {
  readonly style: string = 'rheo/property/atom';
  private readonly cols: Column[] = [];
  private readonly interfaceFlag: boolean;
  private readonly surfaceFlag: boolean;
  private readonly shiftFlag: boolean;
  private readonly thermalFlag: boolean;
  private readonly pressureFlag: boolean;
  private readonly shellFlag: boolean;
  private readonly coordinationFlag: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!isRheoStyle(sys.atomStyle)) throw new StyleError('compute rheo/property/atom requires atom_style rheo');
    if (args.length < 1) throw new StyleError('Illegal compute rheo/property/atom command: missing argument(s)');
    const dim = sys.state.dimension;
    let interfaceFlag = false, surfaceFlag = false, shiftFlag = false, thermalFlag = false, pressureFlag = false;
    let shellFlag = false, coordinationFlag = false;
    const invalid = (a: string): StyleError =>
      new StyleError(`Invalid keyword ${a} for atom style ${sys.atomStyle} in compute rheo/property/atom command `);
    const one = (pack: Pack, prop = ''): void => { this.cols.push({ pack, index: 0, prop }); };
    for (const a of args) {
      if (a === 'phase') one('phase');
      else if (a === 'status') one('status');
      else if (a === 'chi') { interfaceFlag = true; one('chi'); }
      else if (a === 'surface') { surfaceFlag = true; one('surface'); }
      else if (a === 'surface/r') { surfaceFlag = true; one('surface/r'); }
      else if (a === 'surface/divr') { surfaceFlag = true; one('surface/divr'); }
      else if (a === 'coordination') { coordinationFlag = true; one('coordination'); }
      else if (a === 'pressure') { pressureFlag = true; one('pressure'); }
      else if (a === 'cv') { thermalFlag = true; one('cv'); }
      else if (a.startsWith('surface/n/')) { surfaceFlag = true; this.addVector(a, 'surface/n', dim); }
      else if (a.startsWith('shift/v/')) { shiftFlag = true; this.addVector(a, 'shift/v', dim); }
      else if (a.startsWith('grad/v/')) this.addTensor(a, 'grad/v', dim);
      else if (a.startsWith('stress/v/')) { pressureFlag = true; this.addTensor(a, 'stress/v', dim); }
      else if (a.startsWith('stress/t/')) { pressureFlag = true; this.addTensor(a, 'stress/t', dim); }
      else if (a === 'energy') {
        // the energy keyword is the esph property of atom_style rheo/thermal
        if (!isRheoThermalStyle(sys.atomStyle)) throw invalid(a);
        thermalFlag = true;
        one('avec', RHEO_ESPH);
      } else if (a === 'nbond/shell') { shellFlag = true; one('nbond/shell'); }
      else {
        const prop = AVEC.get(a) ?? (isRheoThermalStyle(sys.atomStyle) ? AVEC_THERMAL.get(a) : undefined);
        if (prop === undefined) throw invalid(a);
        one('avec', prop);
        if (a === 'temperature' || a === 'heatflow' || a === 'conductivity') thermalFlag = true;
      }
    }
    this.interfaceFlag = interfaceFlag;
    this.surfaceFlag = surfaceFlag;
    this.shiftFlag = shiftFlag;
    this.thermalFlag = thermalFlag;
    this.pressureFlag = pressureFlag;
    this.shellFlag = shellFlag;
    this.coordinationFlag = coordinationFlag;
    this.peratomFlag = true;
    this.sizePeratomCols = this.cols.length > 1 ? this.cols.length : 0;
  }

  /** add_vector_component: "a/*" gives the dim components, "a/x", "a/y", "a/z" one (z is an error in 2d). */
  private addVector(option: string, pack: Pack, dim: number): void {
    if (option.endsWith('*')) {
      for (let a = 0; a < dim; a++) this.cols.push({ pack, index: a, prop: '' });
      return;
    }
    let index: number;
    if (option.endsWith('x')) index = 0;
    else if (option.endsWith('y')) index = 1;
    else if (option.endsWith('z')) {
      if (dim === 2) throw new StyleError(`Invalid compute rheo/property/atom property ${option} in 2D`);
      index = 2;
    } else throw new StyleError(`Invalid compute rheo/property/atom property ${option}`);
    this.cols.push({ pack, index, prop: '' });
  }

  /**
   * add_tensor_component: "a/*" gives the dim*dim components (row-major), "a/xx" ... "a/zz" one, at index a*dim + b
   * (in 2d yx and yy are 2 and 3; a z component is an error).
   */
  private addTensor(option: string, pack: Pack, dim: number): void {
    if (option.endsWith('*')) {
      for (let a = 0; a < dim; a++) for (let b = 0; b < dim; b++) this.cols.push({ pack, index: a * dim + b, prop: '' });
      return;
    }
    let index: number;
    let dimError = false;
    if (option.endsWith('xx')) index = 0;
    else if (option.endsWith('xy')) index = 1;
    else if (option.endsWith('xz')) { index = 2; dimError = dim === 2; }
    else if (option.endsWith('yx')) index = dim === 2 ? 2 : 3;
    else if (option.endsWith('yy')) index = dim === 2 ? 3 : 4;
    else if (option.endsWith('yz')) { index = 5; dimError = dim === 2; }
    else if (option.endsWith('zx')) { index = 6; dimError = dim === 2; }
    else if (option.endsWith('zy')) { index = 7; dimError = dim === 2; }
    else if (option.endsWith('zz')) { index = 8; dimError = dim === 2; }
    else throw new StyleError(`Invalid compute rheo/property/atom property ${option}`);
    if (dimError) throw new StyleError(`Invalid compute rheo/property/atom property ${option} in 2D`);
    this.cols.push({ pack, index, prop: '' });
  }

  /** compute_rheo_property_atom.cpp init() and setup(): the attributes need their fix rheo options and fixes. */
  init(): void {
    const model = rheoModelOf(this.sys.state);
    if (!model || !this.sys.fixes.some((f) => f.style === 'rheo')) throw new StyleError('Need to define fix rheo to use compute rheo/property/atom');
    if (this.interfaceFlag && !model.interfaceFlag) throw new StyleError('Cannot request interfacial property without corresponding option in fix rheo');
    if (this.surfaceFlag && !model.surfaceFlag) throw new StyleError('Cannot request surface property without corresponding option in fix rheo');
    if (this.shiftFlag && !model.shiftFlag) throw new StyleError('Cannot request velocity shifting property without corresponding option in fix rheo');
    if (this.thermalFlag && !(model.thermalFlag && this.sys.fixes.some((f) => f.style === 'rheo/thermal'))) throw new StyleError('Cannot request thermal property without fix rheo/thermal');
    if (this.pressureFlag && !this.sys.fixes.some((f) => f.style === 'rheo/pressure')) throw new StyleError('Cannot request pressure property without fix rheo/pressure');
    if (this.shellFlag && !model.oxidation) throw new StyleError('Cannot request nbond/shell without fix rheo/oxidation');
  }

  /** The fix rheo/pressure whose equation of state gives the pressure (compute_rheo_property_atom.cpp fix_pressure). */
  private pressureFix(): { calcPressure(rho: number, i: number): number } {
    const f = this.sys.fixes.find((x) => x.style === 'rheo/pressure');
    if (!f) throw new StyleError('Cannot request pressure property without fix rheo/pressure');
    return f as unknown as { calcPressure(rho: number, i: number): number };
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const nv = this.cols.length;
    const n = s.n;
    const dim = s.dimension, dd = dim * dim;
    const out = new Float64Array(nv > 1 ? nv * n : n);
    const model = rheoModelOf(s)!;
    // compute_peratom(): "if (coordination_flag && !(fix_rheo->coordination_flag)) compute_kernel->compute_coordination()"
    // (fix rheo counts the neighbors itself for a corrected kernel or surface detection)
    if (this.coordinationFlag && !(model.correctionOrder >= 0 || model.surfaceFlag)) computeRheoCoordination(s, this.sys.nb, model);
    const status = s.custom.get(RHEO_STATUS)!.data;
    const rho = s.custom.get(RHEO_RHO)!.data;
    const visc = s.custom.get(RHEO_VISC)!.data;
    const pfix = this.pressureFlag ? this.pressureFix() : null;
    const gradv = model.gradv;
    for (let k = 0; k < nv; k++) {
      const { pack, index, prop } = this.cols[k];
      const a = Math.trunc(index / dim), b = index % dim;
      const transpose = b * dim + a;
      const data = pack === 'avec' ? s.custom.get(prop)!.data : null;
      for (let i = 0; i < n; i++) {
        let val = 0.0;
        if (s.mask[i] & this.groupBit) {
          switch (pack) {
            // pack_phase: "buf[n] = (status[i] & PHASECHECK)"
            case 'phase': val = status[i] & PHASECHECK; break;
            case 'status': val = status[i]; break;
            case 'surface': val = (status[i] & STATUS_SURFACE) ? 1.0 : (status[i] & STATUS_SPLASH) ? 2.0 : 0.0; break;
            case 'surface/r': val = model.rsurface[i]; break;
            case 'surface/divr': val = model.divr[i]; break;
            case 'surface/n': val = model.nsurface[i * dim + index]; break;
            case 'coordination': val = model.coord[i]; break;
            // pack_pressure: "fix_pressure->calc_pressure(rho[i], i)", from the density at the time of the output
            case 'pressure': val = pfix!.calcPressure(rho[i], i); break;
            case 'cv': val = model.thermal!.calcCv(s.type[i]); break;
            case 'shift/v': val = model.vshift[i * dim + index]; break;
            case 'grad/v': val = gradv[i * dd + index]; break;
            // pack_viscous_stress: "viscosity[i] * (gradv[i][index] + gradv[i][index_transpose])"
            case 'stress/v': val = visc[i] * (gradv[i * dd + index] + gradv[i * dd + transpose]); break;
            // pack_total_stress: the viscous stress plus the pressure on the diagonal
            case 'stress/t': {
              const p = index === transpose ? pfix!.calcPressure(rho[i], i) : 0.0;
              val = visc[i] * (gradv[i * dd + index] + gradv[i * dd + transpose]) + p;
              break;
            }
            // pack_nbond_shell: the formed shell bonds (the internal per-atom property of bond rheo/shell)
            case 'nbond/shell': val = shellNbond(s)[i]; break;
            case 'avec': val = data![i]; break;
            // pack_chi: compute rheo/interface's chi
            case 'chi': val = model.chi[i]; break;
          }
        }
        if (nv > 1) out[nv * i + k] = val;
        else out[i] = val;
      }
    }
    if (nv > 1) this.arrayAtom = out;
    else this.vectorAtom = out;
  }
}
