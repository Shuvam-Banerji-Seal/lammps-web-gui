/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/RHEO/fix_rheo.cpp and fix_rheo.h (FixRHEO: fix rheo, the time integration of the RHEO particles
 * and the density evolution), at commit b332f28. Contributing authors (from fix_rheo.cpp): Joel Clemmer (SNL), Thomas
 * O'Connor (CMU), Eric Palermo (CMU). Copyright Sandia Corporation (GNU General Public License); this file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_rheo.html): "fix ID group-ID rheo cut kstyle zmin keyword values..." and "kstyle = quintic
 * or RK0 or RK1 or RK2". The browser engine supports the quintic and RK1 kernels with the shift, density, speed/sound,
 * thermal (energy gradient for fix rheo/thermal), surface/detection and rho/sum keywords and refuses the others by name
 * (RK0, RK2; interface/reconstruct).
 *
 * Integration (native order): initial_integrate kicks v by dt/2 F/m, computes the gradients, drifts x by dt v, evolves
 * rho by dt/2 (drho - rho div v) on fluid atoms, then applies the shift (x, v, rho) of the previous force evaluation;
 * pre_force computes the gradients and the shift velocity and clears the temporary status options; final_integrate
 * kicks v by dt/2 F/m and evolves rho the same way. Solid atoms (status bits of PHASECHECK) keep their density.
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { StyleError, typeBounds } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { massOf, RHEO_DRHO, RHEO_RHO, RHEO_STATUS, isRheoStyle, isRheoThermalStyle } from '../../engine/atoms';
import { computeRheoCoordination, computeRheoCorrections } from './compute_rheo_kernel';
import { computeRheoRhoSum, computeRheoSurface } from './compute_rheo_surface';
import {
  RheoModel, computeRheoGrad, computeRheoVShift, correctRheoSurfaces, OPTIONSMASK, PHASECHECK, STATUS_NO_INTEGRATION, STATUS_NO_SHIFT,
  rheoProp, setRheoModel,
} from './model';

export class FixRHEO extends Fix {
  readonly style: string = 'rheo';
  readonly model: RheoModel;
  private dtv = 0;
  private dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.timeIntegrate = true;
    // fix_rheo.cpp: "fix rheo command requires group all"
    if (group !== 'all') throw new StyleError('fix rheo command requires group all');
    const s = sys.state;
    if (!isRheoStyle(s.atomStyle)) throw new StyleError('fix rheo command requires atom_style rheo');
    if (args.length < 3) throw new StyleError('Illegal fix rheo command: needs cutoff, kernel and zmin');
    // fix_rheo.cpp: "Fix RHEO must be defined before all other RHEO fixes" (the accessory fixes check the order at setup)
    if (sys.fixes.some((f) => f.style === 'rheo/viscosity' || f.style === 'rheo/pressure' || f.style === 'rheo/thermal')) {
      throw new StyleError('Fix RHEO must be defined before all other RHEO fixes');
    }
    const n = s.ntypes;
    const cut = parseNum(args[0], 'cutoff');
    const kernel = args[1];
    if (kernel !== 'quintic' && kernel !== 'RK1') {
      if (['RK0', 'RK2'].includes(kernel)) {
        throw new StyleError(`fix rheo kernel ${kernel} is not supported by the browser engine yet (quintic and RK1 only)`);
      }
      throw new StyleError(`Unknown kernel style ${kernel} in fix rheo`);
    }
    // zmin is the coordination threshold of the corrected kernels (fix_rheo.cpp: utils::inumeric); the quintic kernel
    // has no correction, so there the value only has to be an integer
    if (!Number.isInteger(Number(args[2]))) throw new StyleError(`Expected integer parameter in fix rheo: ${args[2]}`);
    const dim = s.dimension;
    this.model = new RheoModel(dim, cut, n);
    this.model.rk1 = kernel === 'RK1';
    this.model.zmin = Number(args[2]);
    let shiftFlag = false;
    const shiftType = new Uint8Array(n + 1).fill(1);
    let iarg = 3;
    while (iarg < args.length) {
      const w = args[iarg];
      if (w === 'shift') {
        shiftFlag = true;
        while (iarg + 1 < args.length) {
          if (args[iarg + 1] === 'exclude/type') {
            if (iarg + 2 >= args.length) throw new StyleError('Illegal fix rheo shift exclude/type: missing types');
            const [lo, hi] = typeBounds(args[iarg + 2], n);
            for (let t = lo; t <= hi; t++) shiftType[t] = 0;
            iarg += 2;
          } else if (args[iarg + 1] === 'scale/cross/type') {
            throw new StyleError('fix rheo shift scale/cross/type is not supported by the browser engine yet');
          } else {
            break;
          }
        }
      } else if (w === 'surface/detection') {
        // fix_rheo.cpp: "surface/detection coordination zmin_surface zmin_splash" (divergence is refused by name)
        if (iarg + 3 >= args.length) throw new StyleError('Illegal fix rheo surface/detection: missing values');
        if (args[iarg + 1] !== 'coordination') {
          throw new StyleError(`fix rheo surface/detection ${args[iarg + 1]} is not supported by the browser engine yet (coordination only)`);
        }
        if (!Number.isInteger(Number(args[iarg + 2])) || !Number.isInteger(Number(args[iarg + 3]))) {
          throw new StyleError('Expected integer parameter in fix rheo surface/detection');
        }
        this.model.surfaceFlag = true;
        this.model.zminSurface = Number(args[iarg + 2]);
        this.model.zminSplash = Number(args[iarg + 3]);
        iarg += 3;
      } else if (w === 'rho/sum') {
        // fix_rheo.cpp: optional "self/mass" (a particle uses its own mass in the summation)
        this.model.rhosumFlag = true;
        if (iarg + 1 < args.length && args[iarg + 1] === 'self/mass') {
          this.model.rhosumSelfMass = true;
          iarg += 1;
        }
      } else if (w === 'thermal') {
        // fix_rheo.cpp: "thermal" sets thermal_flag (energy gradients, and fix rheo/thermal is required in setup)
        this.model.thermalFlag = true;
      } else if (w === 'interface/reconstruct') {
        throw new StyleError(`fix rheo keyword ${w} is not supported by the browser engine yet`);
      } else if (w === 'density') {
        if (iarg + n >= args.length) throw new StyleError('Illegal fix rheo density: missing values');
        for (let t = 1; t <= n; t++) {
          const r0 = parseNum(args[iarg + t], 'density');
          if (!(r0 > 0)) throw new StyleError('The equilibrium density must be greater than zero');
          this.model.rho0[t] = r0;
        }
        iarg += n;
      } else if (w === 'speed/sound') {
        if (iarg + n >= args.length) throw new StyleError('Illegal fix rheo speed/sound: missing values');
        for (let t = 1; t <= n; t++) {
          const c = parseNum(args[iarg + t], 'speed of sound');
          this.model.csq[t] = c * c;
        }
        iarg += n;
      } else {
        throw new StyleError(`Illegal fix rheo command: ${w}`);
      }
      iarg += 1;
    }
    // fix_rheo.cpp init(): "fix rheo command requires atom property esph with thermal setting" (and the other thermal properties)
    if (this.model.thermalFlag && !isRheoThermalStyle(s.atomStyle)) {
      throw new StyleError('fix rheo command requires atom property esph with thermal setting');
    }
    this.model.shiftFlag = shiftFlag;
    this.model.shiftType = shiftFlag ? shiftType : null;
    setRheoModel(s, this.model);
  }

  init(): void { this.resetDt(); }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  /** fix_rheo.cpp setup_pre_force(): the pre_force evaluation of the RHEO state. */
  setupPreForce(): void { this.preForce(); }

  /**
   * fix_rheo.cpp setup(): "Missing fix rheo/viscosity" and "Missing fix rheo/pressure" when an accessory fix is not
   * defined (native sets the flags in the accessories' setup_pre_force).
   */
  setup(): void {
    if (!this.sys.fixes.some((f) => f.style === 'rheo/viscosity')) throw new StyleError('Missing fix rheo/viscosity');
    if (!this.sys.fixes.some((f) => f.style === 'rheo/pressure')) throw new StyleError('Missing fix rheo/pressure');
    // fix_rheo.cpp setup(): "Missing fix rheo/thermal" when the thermal keyword has no fix rheo/thermal
    if (this.model.thermalFlag && !this.sys.fixes.some((f) => f.style === 'rheo/thermal')) throw new StyleError('Missing fix rheo/thermal');
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const m = this.model;
    const dim = m.dim, dd = dim * dim;
    const n = s.n, bit = this.groupBit;
    const x = s.x, v = s.v, f = s.f, mask = s.mask;
    const status = rheoProp(s, RHEO_STATUS), rho = rheoProp(s, RHEO_RHO), drho = rheoProp(s, RHEO_DRHO);
    const dtf = this.dtf, dtv = this.dtv;
    // velocity half-step kick
    for (let i = 0; i < n; i++) {
      if (status[i] & STATUS_NO_INTEGRATION) continue;
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(s, i);
      for (let a = 0; a < 3; a++) v[3 * i + a] += dtfm * f[3 * i + a];
    }
    // gradients of the current velocities and densities (the positions of the neighbor list are the old ones)
    computeRheoGrad(s, this.sys.nb, m, this.sys.geom);
    // position half-step
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & bit)) continue;
      for (let a = 0; a < dim; a++) x[3 * i + a] += dtv * v[3 * i + a];
    }
    const gradv = m.gradv;
    // density from div(u), fluid atoms only
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & bit)) continue;
      if (status[i] & STATUS_NO_INTEGRATION) continue;
      if (status[i] & PHASECHECK) continue;
      let divu = 0;
      for (let a = 0; a < dim; a++) divu += gradv[i * dd + a * (1 + dim)];
      if (!m.rhosumFlag) rho[i] += dtf * (drho[i] - rho[i] * divu);
    }
    // shifting atoms (the shift velocity of the last pre_force)
    if (m.shiftFlag) {
      const vs = m.vshift, gradr = m.gradr;
      for (let i = 0; i < n; i++) {
        if (status[i] & STATUS_NO_SHIFT) continue;
        if (status[i] & PHASECHECK) continue;
        if (!(mask[i] & bit)) continue;
        for (let a = 0; a < dim; a++) {
          x[3 * i + a] += dtv * vs[i * dim + a];
          for (let b = 0; b < dim; b++) v[3 * i + a] += dtv * vs[i * dim + b] * gradv[i * dd + a * dim + b];
        }
        if (!m.rhosumFlag) for (let a = 0; a < dim; a++) rho[i] += dtv * vs[i * dim + a] * gradr[i * dim + a];
      }
    }
  }

  /**
   * fix_rheo.cpp pre_force(): coordination and kernel corrections (RK1), gradients, shift velocity, then the temporary
   * status options are cleared.
   */
  preForce(): void {
    const s = this.sys.state;
    const m = this.model;
    // coordination (kernel corrections and surface detection), density by summation, then the corrections
    if (m.rk1 || m.surfaceFlag) computeRheoCoordination(s, this.sys.nb, m);
    if (m.rhosumFlag) computeRheoRhoSum(s, this.sys.nb, m);
    if (m.rk1) computeRheoCorrections(s, this.sys.nb, m);
    computeRheoGrad(s, this.sys.nb, m, this.sys.geom);
    if (m.shiftFlag) computeRheoVShift(s, this.sys.nb, m, this.groupBit, this.sys.geom);
    const status = rheoProp(s, RHEO_STATUS);
    for (let i = 0; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) status[i] = (status[i] & OPTIONSMASK) >>> 0;
    }
    // free-surface detection (last in pre_force, after the status options are cleared)
    if (m.surfaceFlag) computeRheoSurface(s, this.sys.nb, m, this.groupBit, this.sys.geom);
    // compute_rheo_vshift.cpp: shift and surface detection together (native: correct_surfaces after the surface update)
    if (m.shiftFlag && m.surfaceFlag) correctRheoSurfaces(s, m, this.groupBit);
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const m = this.model;
    const dim = m.dim, dd = dim * dim;
    const n = s.n, bit = this.groupBit;
    const v = s.v, f = s.f, mask = s.mask;
    const status = rheoProp(s, RHEO_STATUS), rho = rheoProp(s, RHEO_RHO), drho = rheoProp(s, RHEO_DRHO);
    const dtf = this.dtf;
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & bit)) continue;
      if (status[i] & STATUS_NO_INTEGRATION) continue;
      const dtfm = dtf / massOf(s, i);
      for (let a = 0; a < dim; a++) v[3 * i + a] += dtfm * f[3 * i + a];
    }
    const gradv = m.gradv;
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & bit)) continue;
      if (status[i] & STATUS_NO_INTEGRATION) continue;
      if (status[i] & PHASECHECK) continue;
      let divu = 0;
      for (let a = 0; a < dim; a++) divu += gradv[i * dd + a * (1 + dim)];
      if (!m.rhosumFlag) rho[i] += dtf * (drho[i] - rho[i] * divu);
    }
  }
}
