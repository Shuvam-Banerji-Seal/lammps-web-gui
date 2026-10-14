/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BOCS/fix_bocs.cpp and fix_bocs.h at commit b332f28 (class FixBocs, fix bocs; Nicholas J. H. Dunn and
 * Michael R. DeLyser, Pennsylvania State University). Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The Nose-Hoover machinery (thermostat and barostat chains, the MTK terms, the box remap) is the core FixNH
 * (src/engine/fix/nh.ts), as in native fix_bocs.cpp, which copies fix_npt.cpp. The BOCS-specific part is here: the
 * cgiso pressure target keyword with the analytic coarse-grained basis, the compute pressure/bocs whose scalar carries
 * the basis correction (compute_pressure_bocs.ts), and the barostat reading that corrected isotropic pressure
 * (native couple(): the ISO pressure is the scalar).
 *
 * Documented behaviour (docs.lammps.org/fix_bocs.html, doc/src/fix_bocs.rst at b332f28): "keyword = *temp* or
 * *cgiso* or *tchain* or *pchain* or *mtk* or *tloop* or *ploop*"; "*cgiso* values = Pstart Pstop Pdamp basis_set
 * args", "basis_set = *analytic* or *linear_spline* or *cubic_spline*". "These commands incorporate a pressure
 * correction ... to the standard MTK barostat"; "The two differences are replacing *npt* with *bocs*, and replacing
 * the iso, aniso and other pressure keywords with cgiso". "The pressure correction has only been tested for use with an isotropic pressure
 * coupling in 3 dimensions." "The fix_modify temp option is supported by this fix." "In order to have LAMMPS report
 * the modified pressure, you must include the thermo_modify command given in the examples" (thermo_modify press
 * 1_press). The keywords tchain, pchain, mtk, tloop and ploop are those of fix npt. The linear_spline and
 * cubic_spline bases read a table file, which the browser engine does not read yet: they are refused by name.
 * Refused by name (StyleError): iso (native message "Pressure fix must be cgiso"), any other keyword, and non-3d
 * systems.
 *
 * Measured with native LAMMPS (w156mx_bocs, 20 steps of the in.methanol system): the modified thermo press, pe,
 * ecouple and econserve match the engine (see the case header).
 */
import { FixNH } from '../../engine/fix/nh';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { ComputePressure } from '../../engine/compute/pressure';
import { ComputePressureBocs } from './compute_pressure_bocs';

/** Keywords of native fix bocs's constructor (fix_bocs.cpp, process keywords loop). */
const BOCS_KEYWORDS = ['temp', 'cgiso', 'tchain', 'pchain', 'mtk', 'tloop', 'ploop'];

/** Translate the BOCS argument list into fix npt keywords plus the analytic basis (fix_bocs.cpp constructor). */
function translate(sys: System, args: string[]): { npt: string[]; basis: { phi: number[]; nMol: number; vavg: number } | null } {
  const npt: string[] = [];
  let basis: { phi: number[]; nMol: number; vavg: number } | null = null;
  let cgiso = false;
  for (let k = 0; k < args.length;) {
    const key = args[k];
    const need = (count: number, msg: string) => {
      if (k + count > args.length) throw new StyleError(msg);
    };
    switch (key) {
      case 'temp':
        need(4, 'Illegal fix bocs command: temp needs Tstart Tstop Tdamp');
        npt.push(...args.slice(k, k + 4));
        k += 4;
        break;
      case 'iso':
        throw new StyleError('Illegal fix bocs command. Pressure fix must be cgiso . Use regular fix bocs for iso');
      case 'cgiso': {
        need(4, 'Illegal fix bocs command. cgiso must be followed by: P_0 P_f P_coupl');
        const p0 = parseNum(args[k + 1], 'fix bocs cgiso P_0');
        const pf = parseNum(args[k + 2], 'fix bocs cgiso P_f');
        const pp = parseNum(args[k + 3], 'fix bocs cgiso P_coupl');
        npt.push('iso', String(p0), String(pf), String(pp));
        cgiso = true;
        k += 4;
        const basisKey = args[k];
        if (basisKey === 'analytic') {
          need(4, 'Illegal fix bocs command. Basis type analytic must be followed by: avg_vol n_mol n_pmatch_coeff');
          const vavg = parseNum(args[k + 1], 'fix bocs avg_vol');
          const nMol = Math.trunc(parseNum(args[k + 2], 'fix bocs n_mol'));
          const nCoeff = Math.trunc(parseNum(args[k + 3], 'fix bocs n_pmatch_coeff'));
          k += 4;
          if (k + nCoeff > args.length) throw new StyleError('Illegal fix bocs command. Missing coeffs.');
          const phi: number[] = [];
          for (let i = 0; i < nCoeff; i++) phi.push(parseNum(args[k + i], 'fix bocs coefficient'));
          k += nCoeff;
          basis = { phi, nMol, vavg };
        } else if (basisKey === 'linear_spline' || basisKey === 'cubic_spline') {
          throw new StyleError(`fix bocs ${basisKey} basis is not supported by the browser engine yet (the table is not read)`);
        } else {
          throw new StyleError(`CG basis type ${basisKey} is not recognized\nSupported basis types: analytic linear_spline cubic_spline`);
        }
        break;
      }
      case 'tchain': case 'pchain': case 'mtk': case 'tloop': case 'ploop':
        need(2, 'Illegal fix bocs command');
        npt.push(key, args[k + 1]);
        k += 2;
        break;
      default:
        throw new StyleError(`Illegal fix bocs command: unrecognized keyword ${key}`);
    }
  }
  if (!cgiso) throw new StyleError('Pressure control must be used with fix bocs');
  if (sys.dimension !== 3) throw new StyleError('Invalid fix bocs command. Must use 3 dimensions');
  return { npt, basis };
}

export class FixBocs extends FixNH {
  private readonly basis: { phi: number[]; nMol: number; vavg: number } | null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const t = translate(sys, args);
    // the fix npt machinery with the cgiso target as an isotropic iso keyword (fix_bocs.cpp keeps the same P_start,
    // P_stop and P_period for x, y and z, and couples them: pcouple XYZ)
    super(sys, id, group, t.npt, 'npt');
    this.basis = t.basis;
    // fix_bocs.cpp constructor: compute ID_press is "pressure/bocs ID_temp" in place of the fix npt pressure compute
    const pid = `${id}_press`;
    const pc = new ComputePressureBocs(sys, pid, 'all', [`${id}_temp`]);
    const k = sys.computes.findIndex((c) => c.id === pid);
    if (k >= 0) sys.computes[k] = pc; else sys.computes.push(pc);
    this.pressCompute = pc;
    if (this.basis) pc.setCgBasis(this.basis);
  }

  /** couple() of fix_bocs.cpp (ISO): the barostat reads the basis-corrected pressure on every diagonal component. */
  protected pressureTensor(pc: ComputePressure): ArrayLike<number> {
    const v = Float64Array.from(pc.vectorValues());
    if (pc instanceof ComputePressureBocs) {
      const corr = pc.correction(this.sys.geom.volume(this.sys.dimension));
      v[0] += corr; v[1] += corr; v[2] += corr;
    }
    return v;
  }
}
