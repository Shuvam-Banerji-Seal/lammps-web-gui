/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM/meam_funcs.cpp (G_gam, dG_gam,
 * zbl, embedding, erose, get_shpfcn, get_Zij, get_Zij2, get_Zij2_b2nn), the inline cutoff helpers of src/MEAM/meam.h,
 * and the MathSpecial helpers they use (src/math_special.cpp: fm_exp, exp2_x86; src/math_special.h: powint). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE). Original MEAM authors (from meam_funcs.cpp and meam.h): Greg Wagner
 * (Sandia), Sungkwang Mun, Stephen Foiles, Michael I. Baskes and Mike Baskes (MS-MEAM: Steve Valone, LANL).
 */
import { pow as libmPow } from '../libm/pow';

/** MEAM lattice names (MEAM::lattice_t). */
export type Lattice = 'fcc' | 'bcc' | 'hcp' | 'dim' | 'dia' | 'dia3' | 'b1' | 'c11' | 'l12' | 'b2' | 'ch4' | 'lin' | 'zig' | 'tri' | 'sc';

const LOG2OFE = 1.4426950408889634074;
const Q0 = 2.33184211722314911771e2;
const Q1 = 4.36821166879210612817e3;
const P0 = 2.30933477057345225087e-2;
const P1 = 2.02020656693165307700e1;
const P2 = 1.51390680115615096133e3;
const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer);
const LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** MathSpecial::exp2_x86: Pade approximant of 2^x as in LAMMPS (little endian path). */
export const exp2x86 = (x: number): number => {
  const ipart = Math.floor(x + 0.5);
  const fpart = x - ipart;
  // epart.s.i0 = 0; epart.s.i1 = (ipart + 1023) << 20 (the high word of the double)
  const hi = ((ipart + 1023) << 20) >>> 0;
  if (LITTLE) {
    u32[0] = 0;
    u32[1] = hi;
  } else {
    u32[1] = 0;
    u32[0] = hi;
  }
  const epart = f64[0];
  let x2 = fpart * fpart;
  let px = P0;
  px = px * x2 + P1;
  let qx = x2 + Q0;
  px = px * x2 + P2;
  qx = qx * x2 + Q1;
  px = px * fpart;
  x2 = 1.0 + 2.0 * (px / (qx - px));
  return epart * x2;
};

/** MathSpecial::fm_exp: fast exp(x) (LAMMPS math_special.cpp). */
export const fmExp = (x: number): number => {
  if (x < -1022.0 / LOG2OFE) return 0;
  if (x > 1023.0 / LOG2OFE) return Infinity;
  return exp2x86(LOG2OFE * x);
};

/** MathSpecial::powint (repeated squaring, as in LAMMPS math_special.h). */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

export const iszero = (f: number): boolean => Math.abs(f) < 1e-20;
export const isone = (f: number): boolean => Math.abs(f - 1.0) < 1e-20;

/** MEAM::fcut: cutoff function (1 - (1 - xi)^4)^2 on (0, 1). */
export const fcut = (xi: number): number => {
  if (xi >= 1.0) return 1.0;
  if (xi <= 0.0) return 0.0;
  let a = 1.0 - xi;
  a *= a;
  a *= a;
  a = 1.0 - a;
  return a * a;
};

/** MEAM::dfcut: value and derivative (dfc) of fcut. */
export const dfcut = (xi: number): { f: number; df: number } => {
  if (xi >= 1.0) return { f: 1.0, df: 0.0 };
  if (xi <= 0.0) return { f: 0.0, df: 0.0 };
  const a = 1.0 - xi;
  const a3 = a * a * a;
  const a4 = a * a3;
  const a1m4 = 1.0 - a4;
  return { f: a1m4 * a1m4, df: 8 * a1m4 * a3 };
};

/** MEAM::dCfunc: derivative of Cikj with respect to rij. */
export const dCfunc = (rij2: number, rik2: number, rjk2: number): number => {
  const rij4 = rij2 * rij2;
  const a = rik2 - rjk2;
  const b = rik2 + rjk2;
  const asq = a * a;
  let denom = rij4 - asq;
  denom = denom * denom;
  return (-4 * (-2 * rij2 * asq + rij4 * b + asq * b)) / denom;
};

/** MEAM::dCfunc2: derivatives of Cikj with respect to rik and rjk. */
export const dCfunc2 = (rij2: number, rik2: number, rjk2: number): [number, number] => {
  const rij4 = rij2 * rij2;
  const rik4 = rik2 * rik2;
  const rjk4 = rjk2 * rjk2;
  const a = rik2 - rjk2;
  let denom = rij4 - a * a;
  denom = denom * denom;
  const d1 = (4 * rij2 * (rij4 + rik4 + 2 * rik2 * rjk2 - 3 * rjk4 - 2 * rij2 * a)) / denom;
  const d2 = (4 * rij2 * (rij4 - 3 * rik4 + 2 * rik2 * rjk2 + rjk4 + 2 * rij2 * a)) / denom;
  return [d1, d2];
};

/** MEAM::get_Zij: number of first neighbours of the reference lattice. */
export const getZij = (latt: Lattice): number => {
  switch (latt) {
    case 'fcc': return 12;
    case 'bcc': return 8;
    case 'hcp': return 12;
    case 'dia':
    case 'dia3': return 4;
    case 'dim': return 1;
    case 'b1':
    case 'sc': return 6;
    case 'c11': return 10;
    case 'l12': return 12;
    case 'b2': return 8;
    case 'ch4': return 4;
    case 'lin':
    case 'zig':
    case 'tri': return 2;
    default: return 0;
  }
};

/** MEAM::get_shpfcn: shape factors s[0..2] of the reference lattice. */
export const getShpfcn = (latt: Lattice, sthe: number, cthe: number): [number, number, number] => {
  switch (latt) {
    case 'fcc':
    case 'bcc':
    case 'b1':
    case 'b2':
    case 'sc':
      return [0.0, 0.0, 0.0];
    case 'hcp':
      return [0.0, 0.0, 1.0 / 3.0];
    case 'ch4':
    case 'dia':
    case 'dia3':
      return [0.0, 0.0, 32.0 / 9.0];
    case 'dim':
      return [1.0, 2.0 / 3.0, 0.4];
    case 'lin':
      return [0.0, 8.0 / 3.0, 0.0];
    case 'zig':
    case 'tri': {
      const s0 = 4.0 * Math.pow(cthe, 2);
      const s1 = 4.0 * (libmPow(cthe, 4) + libmPow(sthe, 4) - 1.0 / 3.0);
      let s2 = 4.0 * (Math.pow(cthe, 2) * (3 * libmPow(sthe, 4) + libmPow(cthe, 4)));
      s2 = s2 - 0.6 * s0;
      return [s0, s1, s2];
    }
    default:
      return [0.0, 0.0, 0.0];
  }
};

/** MEAM::get_Zij2: second-neighbour count, distance ratio a and screening S. */
export const getZij2 = (latt: Lattice, cmin: number, cmax: number, stheta: number): { z: number; a: number; S: number } => {
  let Zij2 = 0;
  let numscr = 0;
  let a = 0;
  switch (latt) {
    case 'fcc': Zij2 = 6; a = Math.sqrt(2.0); numscr = 4; break;
    case 'bcc': Zij2 = 6; a = 2.0 / Math.sqrt(3.0); numscr = 4; break;
    case 'hcp': Zij2 = 6; a = Math.sqrt(2.0); numscr = 4; break;
    case 'b1':
    case 'sc': Zij2 = 12; a = Math.sqrt(2.0); numscr = 2; break;
    case 'dia': Zij2 = 12; a = Math.sqrt(8.0 / 3.0); numscr = 1; break;
    case 'dia3': Zij2 = 12; a = Math.sqrt(11.0 / 3.0); numscr = 4; break;
    case 'ch4':
    case 'lin':
    case 'dim':
      return { z: 0, a: 1.0, S: 0.0 };
    case 'tri': Zij2 = 1; a = 2.0 * stheta; numscr = 2; break;
    case 'zig': Zij2 = 2; a = 2.0 * stheta; numscr = 1; break;
    case 'l12': Zij2 = 6; a = Math.sqrt(2.0); numscr = 4; break;
    case 'b2': Zij2 = 6; a = 2.0 / Math.sqrt(3.0); numscr = 4; break;
    default: break;
  }
  const C = latt === 'dia3' ? 1.0 : 4.0 / (a * a) - 1.0;
  const x = (C - cmin) / (cmax - cmin);
  const sijk = fcut(x);
  return { z: Zij2, a, S: powint(sijk, numscr) };
};

/** MEAM::get_Zij2_b2nn: zig-zag and trimer b2nn terms. */
export const getZij2b2nn = (latt: Lattice, cmin: number, cmax: number): { z: number; S: number } => {
  let numscr = 0;
  let Zij2 = 0;
  if (latt === 'zig' || latt === 'tri') {
    Zij2 = 2;
    numscr = 1;
  }
  const C = 1.0;
  const x = (C - cmin) / (cmax - cmin);
  return { z: Zij2, S: powint(fcut(x), numscr) };
};

/** MEAM::zbl: universal screened Coulomb potential. */
export const zblPot = (r: number, z1: number, z2: number): number => {
  const c = [0.028171, 0.28022, 0.50986, 0.18175];
  const d = [0.20162, 0.40290, 0.94229, 3.1998];
  const azero = 0.4685;
  const cc = 14.3997;
  const a = azero / (libmPow(z1, 0.23) + libmPow(z2, 0.23));
  let result = 0.0;
  const x = r / a;
  for (let i = 0; i <= 3; i++) result = result + c[i] * fmExp(-d[i] * x);
  if (r > 0.0) result = (result * z1 * z2) / r * cc;
  return result;
};

/** MEAM::erose: Rose energy with the erose_form selection (0, 1 or 2). */
export const erosePot = (r: number, re: number, alpha: number, Ec: number, repuls: number, attrac: number, form: number): number => {
  let result = 0.0;
  if (r > 0.0) {
    const astar = alpha * (r / re - 1.0);
    let a3 = 0.0;
    if (astar >= 0) a3 = attrac;
    else if (astar < 0) a3 = repuls;
    if (form === 1) result = -Ec * (1 + astar + (-attrac + repuls / r) * astar * astar * astar) * fmExp(-astar);
    else if (form === 2) result = -Ec * (1 + astar + a3 * astar * astar * astar) * fmExp(-astar);
    else result = -Ec * (1 + astar + (a3 * astar * astar * astar) / (r / re)) * fmExp(-astar);
  }
  return result;
};
