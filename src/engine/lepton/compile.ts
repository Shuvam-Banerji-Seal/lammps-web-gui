import type { RNode } from './resolve';
import { libm } from '../libm';

/*
 * Closure compiler for resolved Lepton trees. A compiled node reads its
 * variables from a Float64Array env: slots of builtin variables and of
 * equal-style references (v_name) are given by `slotOf`.
 *
 * Special functions: erf and erfc (the C library's, through libm), and the ZBL screened repulsion
 * zbl(zi, zj, r) of pair_style lepton (pair_lepton.rst defers to pair_zbl.rst):
 *   E = qqr2e zi zj / r * sum_m c_m exp(-d_m k r),  k = angstrom (zi^0.23 + zj^0.23) / 0.46850,
 *   c = (0.18175, 0.50986, 0.28022, 0.02817), d = (3.19980, 0.94229, 0.40290, 0.20162)
 * with the Coulomb constant qqr2e and the Angstroms per distance unit of the
 * units style ("The numerical values of the exponential decay constants in
 * the screening function depend on the unit of distance"). The pair style
 * keeps qqr2e and angstrom in a ZblConst object before each evaluation.
 * The derivative dE/dr = qqr2e zi zj sum_m c_m exp(-g_m r) (-g_m/r - 1/r^2),
 * g_m = d_m k.
 */

export interface ZblConst {
  qqr2e: number;
  angstrom: number;
}

const SCREEN = [0.18175, 0.50986, 0.28022, 0.02817];
const DECAY = [3.19980, 0.94229, 0.40290, 0.20162];

/**
 * erf and erfc are the C library's: Lepton's Erf and Erfc operations evaluate erf(args[0]) and erfc(args[0])
 * (lib/lepton/src/Operation.cpp at b332f28), and the engine's libm table holds glibc's in a session.
 */
export const erf = (x: number): number => libm.erf(x);
export const erfc = (x: number): number => libm.erfc(x);

const zblScale = (zi: number, zj: number, c: ZblConst): number =>
  (c.angstrom * (zi ** 0.23 + zj ** 0.23)) / 0.4685;

/** E(zi, zj, r) of the ZBL repulsion (see the file comment). */
export const zblEnergy = (zi: number, zj: number, r: number, c: ZblConst): number => {
  const k = zblScale(zi, zj, c);
  let phi = 0;
  for (let m = 0; m < 4; m++) phi += SCREEN[m] * libm.exp(-DECAY[m] * k * r);
  return (c.qqr2e * zi * zj * phi) / r;
};

/** dE/dr of zblEnergy. */
export const zblDerivative = (zi: number, zj: number, r: number, c: ZblConst): number => {
  const k = zblScale(zi, zj, c);
  let s = 0;
  for (let m = 0; m < 4; m++) {
    const g = DECAY[m] * k;
    s += SCREEN[m] * libm.exp(-g * r) * (-g / r - 1 / (r * r));
  }
  return c.qqr2e * zi * zj * s;
};

export type Fn = (env: Float64Array) => number;

/** Compiles a resolved tree. slotOf(kind, name) gives the env index of a variable (kind 'var' or 'vref'). */
export const compileNode = (n: RNode, slotOf: (kind: 'var' | 'vref', name: string) => number, zbl: ZblConst | null): Fn => {
  const go = (m: RNode): Fn => {
    switch (m.t) {
      case 'num': { const v = m.v; return () => v; }
      case 'var': case 'vref': { const k = slotOf(m.t, m.name); return (e) => e[k]; }
      case 'neg': { const a = go(m.a); return (e) => -a(e); }
      case 'bin': {
        const a = go(m.a), b = go(m.b);
        switch (m.op) {
          case '+': return (e) => a(e) + b(e);
          case '-': return (e) => a(e) - b(e);
          case '*': return (e) => a(e) * b(e);
          case '/': return (e) => a(e) / b(e);
          case '^': return (e) => a(e) ** b(e);
        }
        break;
      }
      case 'call': return callFn(m.fn, m.args.map(go), m.args, zbl);
    }
    throw new Error('lepton: bad node');
  };
  return go(n);
};

const callFn = (fn: string, f: Fn[], args: RNode[], zbl: ZblConst | null): Fn => {
  const [a, b, c] = f;
  const needZbl = (): ZblConst => {
    if (!zbl) throw new Error('lepton: zbl used without constants');
    return zbl;
  };
  switch (fn) {
    case 'sqrt': return (e) => Math.sqrt(a(e));
    case 'exp': return (e) => libm.exp(a(e));
    case 'log': return (e) => libm.log(a(e));
    case 'sin': return (e) => libm.sin(a(e));
    case 'cos': return (e) => libm.cos(a(e));
    case 'sec': return (e) => 1 / libm.cos(a(e));
    case 'csc': return (e) => 1 / libm.sin(a(e));
    case 'tan': return (e) => Math.tan(a(e));
    case 'cot': return (e) => 1 / Math.tan(a(e));
    case 'asin': return (e) => Math.asin(a(e));
    case 'acos': return (e) => Math.acos(a(e));
    case 'atan': return (e) => Math.atan(a(e));
    case 'sinh': return (e) => Math.sinh(a(e));
    case 'cosh': return (e) => Math.cosh(a(e));
    case 'tanh': return (e) => Math.tanh(a(e));
    case 'erf': return (e) => erf(a(e));
    case 'erfc': return (e) => erfc(a(e));
    case 'abs': return (e) => Math.abs(a(e));
    case 'min': return (e) => { const x = a(e), y = b(e); return x <= y ? x : y; };
    case 'max': return (e) => { const x = a(e), y = b(e); return x >= y ? x : y; };
    case 'delta': return (e) => (a(e) === 0 ? 1 : 0);
    case 'step': return (e) => (a(e) < 0 ? 0 : 1);
    case 'zbl': {
      const k = needZbl();
      return (e) => zblEnergy(a(e), b(e), c(e), k);
    }
    case '__zbl_d': {
      const k = needZbl();
      return (e) => zblDerivative(a(e), b(e), c(e), k);
    }
    case '__sgn': return (e) => (a(e) < 0 ? -1 : 1);
    case '__sel_le': {
      const x = f[2], y = f[3];
      return (e) => (a(e) <= b(e) ? x(e) : y(e));
    }
    case '__sel_ge': {
      const x = f[2], y = f[3];
      return (e) => (a(e) >= b(e) ? x(e) : y(e));
    }
  }
  throw new Error(`lepton: unknown function ${fn} (${args.length} args)`);
};
