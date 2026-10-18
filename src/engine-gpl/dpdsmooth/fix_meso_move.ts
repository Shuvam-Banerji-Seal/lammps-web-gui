/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-SMOOTH/fix_meso_move.cpp and
 * fix_meso_move.h (FixMesoMove: fix meso/move, styles linear, wiggle and rotate; the variable style is refused).
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
import { sphArrays } from '../sph/common';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { cosFma as libmCosFma } from '../libm/sincos_fma';

type Style = 'linear' | 'wiggle' | 'rotate';

const USAGE: Record<Style, string> = {
  linear: 'linear Vx Vy Vz', wiggle: 'wiggle Ax Ay Az period', rotate: 'rotate Px Py Pz Rx Ry Rz period',
};
/** Arguments after the group, style included (fix_meso_move.cpp: narg < 4, 7, 8 and 11 for the styles). */
const MIN_ARGS: Record<Style, number> = { linear: 4, wiggle: 5, rotate: 8 };

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
    sphArrays(s, 'Fix meso/move');
    if (sys.geom.triclinic) throw new StyleError('fix meso/move on a triclinic box is not supported by the browser engine');
    const st = args[0];
    if (st === 'variable') {
      throw new StyleError('fix meso/move variable style is not supported by the browser engine (it needs the variable API); supported: linear, wiggle, rotate');
    }
    if (st !== 'linear' && st !== 'wiggle' && st !== 'rotate') {
      throw new StyleError(`fix meso/move style '${st ?? ''}' is not supported by the browser engine; supported: linear, wiggle, rotate`);
    }
    this.mstyle = st;
    if (args.length < MIN_ARGS[st]) throw new StyleError(`Illegal fix meso/move command: usage fix ID group meso/move ${USAGE[st]} [units box|lattice]`);
    const num = (w: string | undefined, what: string): number => {
      const v = Number(w);
      if (w === undefined || w === 'NULL' || !Number.isFinite(v)) throw new StyleError(`fix meso/move ${st}: ${what} must be a number, got '${w ?? ''}'`);
      return v;
    };
    const numOrNull = (w: string | undefined, what: string): number | null => (w === 'NULL' ? null : num(w, what));
    let iarg: number;
    let period = 0;
    if (st === 'linear') {
      for (let d = 0; d < 3; d++) this.comp[d] = numOrNull(args[1 + d], 'velocity components');
      iarg = 4;
    } else if (st === 'wiggle') {
      for (let d = 0; d < 3; d++) this.comp[d] = numOrNull(args[1 + d], 'amplitude components');
      period = num(args[4], 'period');
      if (!(period > 0)) throw new StyleError('Illegal fix meso/move command: period must be > 0');
      iarg = 5;
    } else {
      for (let d = 0; d < 3; d++) this.point[d] = num(args[1 + d], 'point');
      const axis = [0, 1, 2].map((d) => num(args[4 + d], 'rotation axis'));
      period = num(args[7], 'period');
      if (!(period > 0)) throw new StyleError('Illegal fix meso/move command: period must be > 0');
      const len = Math.hypot(axis[0], axis[1], axis[2]);
      if (len === 0) throw new StyleError('Zero length rotation vector with fix meso/move');
      for (let d = 0; d < 3; d++) this.runit[d] = axis[d] / len;
      iarg = 8;
      if (s.dimension === 2 && (axis[0] !== 0 || axis[1] !== 0)) {
        throw new StyleError('Fix meso/move cannot rotate aroung non z-axis for 2d problem');
      }
    }
    // units: lattice (the default) scales the velocities, amplitudes and the rotation point by the lattice spacings
    let scaleflag = true;
    while (iarg < args.length) {
      if (args[iarg] !== 'units') throw new StyleError(`Illegal fix meso/move command: unknown keyword '${args[iarg]}'`);
      const u = args[iarg + 1];
      if (u === 'box') scaleflag = false;
      else if (u === 'lattice') scaleflag = true;
      else throw new StyleError(`Illegal fix meso/move command: units must be box or lattice, got '${u ?? ''}'`);
      iarg += 2;
    }
    if (s.dimension === 2) {
      if (st === 'linear' && this.comp[2] !== null && this.comp[2] !== 0) throw new StyleError('Fix meso/move cannot set linear z motion for 2d problem');
      if (st === 'wiggle' && this.comp[2] !== null && this.comp[2] !== 0) throw new StyleError('Fix meso/move cannot set wiggle z motion for 2d problem');
    }
    if (scaleflag) {
      const sc = [0, 1, 2].map((d) => sys.lattice?.spacing[d] ?? 1);
      if (st === 'rotate') for (let d = 0; d < 3; d++) this.point[d] *= sc[d];
      else for (let d = 0; d < 3; d++) if (this.comp[d] !== null) this.comp[d] = this.comp[d]! * sc[d];
    }
    this.omega = st === 'linear' ? 0 : (2 * Math.PI) / period;
    this.timeIntegrate = true;
    this.origin = s.step;
    const n = s.n;
    this.xorig = new Float64Array(3 * n);
    this.captured = new Uint8Array(n);
    const bit = this.groupBit;
    const r = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & bit)) continue;
      this.captured[i] = 1;
      sys.geom.unwrap(s.x, s.image, i, r);
      this.xorig[3 * i] = r[0]; this.xorig[3 * i + 1] = r[1]; this.xorig[3 * i + 2] = r[2];
    }
  }

  init(): void { this.resetDt(); }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  setupPreForce(): void {
    // fix_meso_move.cpp setup_pre_force: "set vest equal to v"
    const s = this.sys.state;
    const { vest } = sphArrays(s, 'Fix meso/move');
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
    if (s.n !== this.captured.length) {
      throw new StyleError('fix meso/move: the number of atoms changed after the fix was specified (not supported by the browser engine)');
    }
    this.halfStepSph();
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
      } else {
        this.rotateAtom(i, delta, deltaNext);
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
    for (let i = 0; i < s.n; i++) {
      if (!this.captured[i] || !(mask[i] & this.groupBit)) continue;
      const c = dtf / massOf(s, i);
      for (let d = 0; d < 3; d++) if (this.comp[d] === null) v[3 * i + d] += c * f[3 * i + d];
    }
  }
}
