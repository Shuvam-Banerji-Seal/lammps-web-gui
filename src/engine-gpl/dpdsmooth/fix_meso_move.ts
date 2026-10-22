/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-SMOOTH/fix_meso_move.cpp and
 * fix_meso_move.h (FixMesoMove: fix meso/move, styles linear, wiggle, rotate and variable).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author (from the source): Morteza
 * Jalalvand (IASBS).
 *
 * The walls of the SPH and SDPD examples: a group of SPH particles moved by a prescribed motion, which also carries
 * their internal energy and density (half-step updates as fix sph) and their estimated velocity vest, the velocity
 * the pair styles read for the next step. docs.lammps.org/fix_meso_move.html describes the styles as those of fix
 * move; the positions are remapped with domain->remap_near (the image nearest the old position), so the image flags
 * change only at the next reneighboring, as native does.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf } from '../../engine/atoms';
import { numeric } from '../../engine/force/util';
import { dropSphGhostVest, keepSphGhostVest, requireSph, sphArrays } from '../sph/common';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { cosFma as libmCosFma } from '../libm/sincos_fma';

type Style = 'linear' | 'wiggle' | 'rotate' | 'variable';

/** Arguments after the group, style included (fix_meso_move.cpp: narg < 7, 8, 11 and 10 for the styles). */
const MIN_ARGS: Record<Style, number> = { linear: 4, wiggle: 5, rotate: 8, variable: 7 };

/** fix_meso_move.cpp's message for every malformed argument */
const ILLEGAL = 'Illegal fix meso/move command';

/** variable: the variable of a component (x, y, z, vx, vy, vz), atom-style or not (set at init); null for NULL. */
type MoveVar = { name: string; atom: boolean } | null;

/**
 * domain.cpp Domain::remap_near (orthogonal boxes): the periodic image of the new coordinate that is nearest the old
 * one, in one dimension.
 */
const remapNear1 = (cnew: number, cold: number, period: number, half: number): number => {
  let c = cnew;
  if (c - cold > period) {
    const n = Math.trunc((c - cold) / period);
    c -= n * period;
  }
  while (c - cold > half) c -= period;
  if (cold - c > period) {
    const n = Math.trunc((cold - c) / period);
    c += n * period;
  }
  while (cold - c > half) c += period;
  return c;
};

export class FixMesoMove extends Fix {
  readonly style = 'meso/move';
  private readonly mstyle: Style;
  /** Per-dimension velocity (linear), amplitude (wiggle); null marks a NULL component (integrated by nve). */
  private readonly comp: (number | null)[] = [null, null, null];
  /** variable: the variables of x, y, z, vx, vy, vz. */
  private readonly vars: MoveVar[] = [];
  /** rotate: the point P (scaled to lattice units by default) and the unit rotation axis. */
  private readonly point: number[] = [0, 0, 0];
  private readonly runit: number[] = [0, 0, 1];
  private readonly omega: number;
  /** Unwrapped positions when the fix was specified ("xoriginal"); zero for atoms outside the group. */
  private readonly xorig: Float64Array;
  private readonly captured: Uint8Array;
  /** Timestep at which the fix was specified (time_origin). */
  private readonly origin: number;
  private dtv = 0;
  private dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const s = sys.state;
    requireSph(s, 'fix meso/move command requires atom_style with both energy and density');
    if (sys.geom.triclinic) throw new StyleError('fix meso/move on a triclinic box is not supported by the browser engine');
    // fix_meso_move.cpp FixMesoMove::FixMesoMove: the style's words, the units keyword, the 2d checks, the lattice
    // scaling and then the rotation axis
    if (args.length < 1) throw new StyleError(ILLEGAL);
    const st = args[0];
    if (!(st in MIN_ARGS)) throw new StyleError(ILLEGAL);
    this.mstyle = st as Style;
    if (args.length < MIN_ARGS[this.mstyle]) throw new StyleError(ILLEGAL);
    const numOrNull = (w: string): number | null => (w === 'NULL' ? null : numeric(w));
    let iarg: number;
    let period = 0;
    const axis = [0, 0, 0];
    if (st === 'linear') {
      for (let d = 0; d < 3; d++) this.comp[d] = numOrNull(args[1 + d]);
      iarg = 4;
    } else if (st === 'wiggle') {
      for (let d = 0; d < 3; d++) this.comp[d] = numOrNull(args[1 + d]);
      period = numeric(args[4]);
      if (period <= 0) throw new StyleError(ILLEGAL);
      iarg = 5;
    } else if (st === 'rotate') {
      for (let d = 0; d < 3; d++) this.point[d] = numeric(args[1 + d]);
      for (let d = 0; d < 3; d++) axis[d] = numeric(args[4 + d]);
      period = numeric(args[7]);
      if (period <= 0) throw new StyleError(ILLEGAL);
      iarg = 8;
    } else {
      for (let k = 0; k < 6; k++) {
        const w = args[1 + k];
        if (w === 'NULL') this.vars.push(null);
        else if (w.startsWith('v_')) this.vars.push({ name: w.slice(2), atom: false });
        else throw new StyleError(ILLEGAL);
      }
      iarg = 7;
    }
    // units: lattice (the default) scales the velocities, amplitudes and the rotation point by the lattice spacings
    let scaleflag = true;
    while (iarg < args.length) {
      if (args[iarg] !== 'units' || iarg + 2 > args.length) throw new StyleError(ILLEGAL);
      const u = args[iarg + 1];
      if (u === 'box') scaleflag = false;
      else if (u === 'lattice') scaleflag = true;
      else throw new StyleError(ILLEGAL);
      iarg += 2;
    }
    if (s.dimension === 2) {
      if (st === 'linear' && this.comp[2] !== null && this.comp[2] !== 0) throw new StyleError('Fix meso/move cannot set linear z motion for 2d problem');
      if (st === 'wiggle' && this.comp[2] !== null && this.comp[2] !== 0) throw new StyleError('Fix meso/move cannot set wiggle z motion for 2d problem');
      // native's typo "aroung"
      if (st === 'rotate' && (axis[0] !== 0 || axis[1] !== 0)) throw new StyleError('Fix meso/move cannot rotate aroung non z-axis for 2d problem');
      if (st === 'variable' && (this.vars[2] || this.vars[5])) throw new StyleError('Fix meso/move cannot define z or vz variable for 2d problem');
    }
    if (scaleflag && st !== 'variable') {
      const sc = [0, 1, 2].map((d) => sys.lattice?.spacing[d] ?? 1);
      if (st === 'rotate') for (let d = 0; d < 3; d++) this.point[d] *= sc[d];
      else for (let d = 0; d < 3; d++) if (this.comp[d] !== null) this.comp[d] = this.comp[d]! * sc[d];
    }
    if (st === 'rotate') {
      const len = Math.hypot(axis[0], axis[1], axis[2]);
      if (len === 0) throw new StyleError('Zero length rotation vector with fix meso/move');
      for (let d = 0; d < 3; d++) this.runit[d] = axis[d] / len;
    }
    this.omega = st === 'wiggle' || st === 'rotate' ? (2 * Math.PI) / period : 0;
    this.timeIntegrate = true;
    this.origin = s.step;
    const n = s.n;
    const xorig = new Float64Array(3 * n);
    this.xorig = xorig;
    this.captured = new Uint8Array(n);
    // fix_meso_move.cpp: peratom_flag with 3 columns, the original unwrapped coordinates (xoriginal)
    this.peratomFlag = true;
    this.sizePeratomCols = 3;
    this.arrayAtom = xorig;
    const bit = this.groupBit;
    const r = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & bit)) continue;
      this.captured[i] = 1;
      sys.geom.unwrap(s.x, s.image, i, r);
      this.xorig[3 * i] = r[0]; this.xorig[3 * i + 1] = r[1]; this.xorig[3 * i + 2] = r[2];
    }
  }

  init(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    // fix_meso_move.cpp FixMesoMove::init: each variable exists and is equal-style or atom-style
    for (const mv of this.vars) {
      if (!mv) continue;
      const v = this.sys.vars.get(mv.name);
      if (!v) throw new StyleError('Variable name for fix meso/move does not exist');
      if (v.style === 'atom' || v.style === 'atomfile') mv.atom = true;
      else if (v.style === 'equal' || v.style === 'timer' || v.style === 'internal') mv.atom = false;
      else throw new StyleError('Variable for fix meso/move is invalid style');
    }
  }

  /** fix_meso_move.cpp FixMesoMove::reset_dt: a timestep command after a run refuses fix meso/move. */
  resetDt(): void {
    throw new StyleError('Resetting timestep size is not allowed with fix meso/move');
  }

  setupPreForce(): void {
    // fix_meso_move.cpp setup_pre_force: "set vest equal to v"
    const s = this.sys.state;
    const { vest } = sphArrays(s, 'Fix meso/move');
    // the owned atoms only: their ghosts keep the vest of before the setup (sph/common.ts sphGhostVest)
    keepSphGhostVest(s, this.sys.run, vest);
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(s.mask[i] & this.groupBit)) continue;
      for (let d = 0; d < 3; d++) vest[3 * i + d] = s.v[3 * i + d];
    }
  }

  /** Half-step update of the internal energy and the density (fix_meso_move.cpp: esph and rho, as fix sph). */
  private halfStepSph(): void {
    const s = this.sys.state;
    const a = sphArrays(s, 'Fix meso/move');
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(s.mask[i] & this.groupBit)) continue;
      a.esph[i] += this.dtf * a.desph[i];
      a.rho[i] += this.dtf * a.drho[i];
    }
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    dropSphGhostVest(s);
    if (s.n !== this.captured.length) {
      // fix_meso_move.cpp set_arrays: an atom added to the group refuses the variable style
      if (this.mstyle === 'variable') throw new StyleError('Cannot add atoms to fix meso/move variable');
      throw new StyleError('fix meso/move: the number of atoms changed after the fix was specified (not supported by the browser engine)');
    }
    // fix_meso_move.cpp initial_integrate: the linear, wiggle and rotate branches advance esph and rho by a half step;
    // the variable branch does not (final_integrate does for every style)
    if (this.mstyle !== 'variable') this.halfStepSph();
    const a = sphArrays(s, 'Fix meso/move');
    const { x, v, f, mask } = s;
    const vest = a.vest;
    const bit = this.groupBit;
    const dtf = this.dtf, dtv = this.dtv;
    const delta = (s.step - this.origin) * dtv;
    const deltaNext = (s.step - this.origin + 1) * dtv;
    const g = this.sys.geom;
    const per = [g.lx, g.ly, g.lz];
    const xold = [0, 0, 0];
    // variable: the variables of x, y, z, vx, vy, vz, evaluated in that order each step (fix_meso_move.cpp)
    const val = this.mstyle === 'variable'
      ? this.vars.map((mv) => (mv === null ? null : mv.atom ? this.sys.atomVariable(mv.name) : this.sys.equalVariable(mv.name)))
      : null;
    const updNve = (i: number, d: number): void => {
      const c = dtf / massOf(s, i);
      vest[3 * i + d] = v[3 * i + d] + 2.0 * c * f[3 * i + d];
      v[3 * i + d] += c * f[3 * i + d];
      x[3 * i + d] += dtv * v[3 * i + d];
    };
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(mask[i] & bit)) continue;
      for (let d = 0; d < 3; d++) xold[d] = x[3 * i + d];
      if (this.mstyle === 'linear') {
        for (let d = 0; d < 3; d++) {
          const val = this.comp[d];
          if (val === null) { updNve(i, d); continue; }
          vest[3 * i + d] = v[3 * i + d] = val;
          x[3 * i + d] = this.xorig[3 * i + d] + val * delta;
        }
      } else if (this.mstyle === 'wiggle') {
        const arg = this.omega * delta, argNext = this.omega * deltaNext;
        const sine = libmSin(arg), cosine = libmCos(arg), cosineNext = libmCosFma(argNext);
        for (let d = 0; d < 3; d++) {
          const val = this.comp[d];
          if (val === null) { updNve(i, d); continue; }
          v[3 * i + d] = val * this.omega * cosine;
          vest[3 * i + d] = val * this.omega * cosineNext;
          x[3 * i + d] = this.xorig[3 * i + d] + val * sine;
        }
      } else if (this.mstyle === 'rotate') {
        this.rotateAtom(i, delta, deltaNext);
      } else {
        // fix_meso_move.cpp variable: a velocity variable sets v, and vest = 2 v_new - v_old ("vest (velocity in next
        // step) could be different from v in the next step, but this is the best we could do")
        for (let d = 0; d < 3; d++) {
          const dx = val![d], vv = val![3 + d];
          if (dx === null && vv === null) { updNve(i, d); continue; }
          if (vv !== null) {
            const nv = typeof vv === 'number' ? vv : vv[i];
            vest[3 * i + d] = 2 * nv - v[3 * i + d];
            v[3 * i + d] = nv;
          }
          if (dx !== null) x[3 * i + d] = this.xorig[3 * i + d] + (typeof dx === 'number' ? dx : dx[i]);
          else x[3 * i + d] += dtv * v[3 * i + d];
        }
      }
      for (let d = 0; d < 3; d++) {
        if (!g.periodic[d]) continue;
        x[3 * i + d] = remapNear1(x[3 * i + d], xold[d], per[d], 0.5 * per[d]);
      }
    }
  }

  /** rotate: x = P + c + disp, v and vest from the rotation at this step and the next (fix_meso_move.cpp). */
  private rotateAtom(i: number, delta: number, deltaNext: number): void {
    const s = this.sys.state;
    const a = sphArrays(s, 'Fix meso/move');
    const { x, v } = s;
    const vest = a.vest;
    const R = this.runit, P = this.point;
    const w = this.omega;
    const arg = w * delta, argNext = w * deltaNext;
    const sine = libmSin(arg), cosine = libmCos(arg);
    const sineNext = libmSin(argNext), cosineNext = libmCos(argNext);
    const d = [0, 1, 2].map((k) => this.xorig[3 * i + k] - P[k]);
    const ddotr = d[0] * R[0] + d[1] * R[1] + d[2] * R[2];
    const c = [ddotr * R[0], ddotr * R[1], ddotr * R[2]];
    const aa = [d[0] - c[0], d[1] - c[1], d[2] - c[2]];
    const b = [R[1] * aa[2] - R[2] * aa[1], R[2] * aa[0] - R[0] * aa[2], R[0] * aa[1] - R[1] * aa[0]];
    const disp = [0, 1, 2].map((k) => aa[k] * cosine + b[k] * sine);
    const dispNext = [0, 1, 2].map((k) => aa[k] * cosineNext + b[k] * sineNext);
    for (let k = 0; k < 3; k++) x[3 * i + k] = P[k] + c[k] + disp[k];
    v[3 * i] = w * (R[1] * disp[2] - R[2] * disp[1]);
    v[3 * i + 1] = w * (R[2] * disp[0] - R[0] * disp[2]);
    v[3 * i + 2] = w * (R[0] * disp[1] - R[1] * disp[0]);
    vest[3 * i] = w * (R[1] * dispNext[2] - R[2] * dispNext[1]);
    vest[3 * i + 1] = w * (R[2] * dispNext[0] - R[0] * dispNext[2]);
    vest[3 * i + 2] = w * (R[0] * dispNext[1] - R[1] * dispNext[0]);
  }

  finalIntegrate(): void {
    this.halfStepSph();
    if (this.mstyle === 'rotate') return;
    const s = this.sys.state;
    const { v, f, mask } = s;
    const dtf = this.dtf;
    // fix_meso_move.cpp final_integrate: the components that neither a value nor a variable sets
    const free = this.mstyle === 'variable' ? [0, 1, 2].map((d) => this.vars[d] === null && this.vars[3 + d] === null) : this.comp.map((c) => c === null);
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(mask[i] & this.groupBit)) continue;
      const c = dtf / massOf(s, i);
      for (let d = 0; d < 3; d++) if (free[d]) v[3 * i + d] += c * f[3 * i + d];
    }
  }
}
