/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANIFOLD/manifold.h,
 * manifold_factory.cpp, manifold_cylinder.cpp, manifold_cylinder_dent.cpp, manifold_dumbbell.cpp,
 * manifold_ellipsoid.cpp, manifold_gaussian_bump.cpp, manifold_plane.cpp, manifold_plane_wiggle.cpp,
 * manifold_sphere.h, manifold_spine.cpp, manifold_supersphere.h, manifold_thylakoid.cpp,
 * manifold_thylakoid_shared.cpp, manifold_torus.cpp and their headers at commit b332f28. MANIFOLD package by Stefan
 * Paquay (Eindhoven University of Technology), Copyright (2013-2014), GNU General Public License. This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * A manifold is a constraint g(x) = 0 on one particle with its gradient n = grad g; fix nve/manifold/rattle and
 * fix nvt/manifold/rattle keep the particles on it with RATTLE, fix manifoldforce removes the normal force component.
 * Each manifold reads its parameters from `params` (set by the fix before postParamInit, as native).
 */
import { StyleError } from '../../engine/force/types';
import { libm } from '../../engine/libm';

export type Vec3 = Float64Array | number[];

/** The manifold base class (manifold.h): g, n, g_and_n, the parameter count and post_param_init. */
export abstract class Manifold {
  params: Float64Array = new Float64Array(0);
  abstract readonly id: string;
  abstract readonly nparams: number;
  abstract g(x: Vec3): number;
  abstract n(x: Vec3, nn: Vec3): void;
  /** Variant of g that computes n at the same time (manifold.h default: n, then g). */
  gAndN(x: Vec3, nn: Vec3): number {
    this.n(x, nn);
    return this.g(x);
  }
  /** Overloaded where initialization depends on params. */
  postParamInit(): void {}
}

/** infnorm<size> of manifold.h. */
export const infnorm = (v: Vec3, size: number): number => {
  let largest = Math.abs(v[0]);
  for (let i = 1; i < size; ++i) {
    const c = Math.abs(v[i]);
    largest = c > largest ? c : largest;
  }
  return largest;
};

/** MathSpecial::powint. */
const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

const MY_2PI = 6.28318530717958647692;

class ManifoldSphere extends Manifold {
  readonly id = 'sphere';
  readonly nparams = 1;
  g(x: Vec3): number {
    const R = this.params[0];
    const r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    return r2 - R * R;
  }
  override gAndN(x: Vec3, nn: Vec3): number {
    const R = this.params[0];
    const r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    nn[0] = 2 * x[0];
    nn[1] = 2 * x[1];
    nn[2] = 2 * x[2];
    return r2 - R * R;
  }
  n(x: Vec3, nn: Vec3): void {
    nn[0] = 2 * x[0];
    nn[1] = 2 * x[1];
    nn[2] = 2 * x[2];
  }
}

class ManifoldPlane extends Manifold {
  readonly id = 'plane';
  readonly nparams = 6;
  g(x: Vec3): number {
    const p = this.params;
    return p[0] * (x[0] - p[3]) + p[1] * (x[1] - p[4]) + p[2] * (x[2] - p[5]);
  }
  n(_x: Vec3, nn: Vec3): void {
    nn[0] = this.params[0];
    nn[1] = this.params[1];
    nn[2] = this.params[2];
  }
}

class ManifoldPlaneWiggle extends Manifold {
  readonly id = 'plane/wiggle';
  readonly nparams = 2;
  g(x: Vec3): number {
    const a = this.params[0], w = this.params[1];
    return x[2] - a * libm.sinAlone(w * x[0]);
  }
  n(x: Vec3, nn: Vec3): void {
    const a = this.params[0], w = this.params[1];
    nn[2] = 1;
    nn[1] = 0.0;
    nn[0] = -a * w * libm.cosAlone(w * x[0]);
  }
}

class ManifoldCylinder extends Manifold {
  readonly id = 'cylinder';
  readonly nparams = 1;
  g(x: Vec3): number {
    const R = this.params[0];
    const r2 = x[0] * x[0] + x[1] * x[1];
    return R * R - r2;
  }
  n(x: Vec3, nn: Vec3): void {
    nn[0] = -2 * x[0];
    nn[1] = -2 * x[1];
    nn[2] = 0.0;
  }
}

class ManifoldCylinderDent extends Manifold {
  readonly id = 'cylinder/dent';
  readonly nparams = 3;
  g(x: Vec3): number {
    const l = this.params[1], R = this.params[0], a = this.params[2];
    const r2 = x[1] * x[1] + x[0] * x[0];
    if (Math.abs(x[2]) < 0.5 * l) {
      const k = MY_2PI / l;
      const c = R - 0.5 * a * (1.0 + libm.cosAlone(k * x[2]));
      return c * c - r2;
    }
    return R * R - r2;
  }
  n(x: Vec3, nn: Vec3): void {
    const l = this.params[1], R = this.params[0], a = this.params[2];
    if (Math.abs(x[2]) < 0.5 * l) {
      const k = MY_2PI / l;
      // cos and sin of the same argument: sincos in the native build
      const c = R - 0.5 * a * (1.0 + libm.cos(k * x[2]));
      nn[0] = -2 * x[0];
      nn[1] = -2 * x[1];
      nn[2] = c * a * k * libm.sin(k * x[2]);
    } else {
      nn[0] = -2 * x[0];
      nn[1] = -2 * x[1];
      nn[2] = 0.0;
    }
  }
}

class ManifoldEllipsoid extends Manifold {
  readonly id = 'ellipsoid';
  readonly nparams = 3;
  g(x: Vec3): number {
    const ai = 1.0 / this.params[0], bi = 1.0 / this.params[1], ci = 1.0 / this.params[2];
    return x[0] * x[0] * ai * ai + x[1] * x[1] * bi * bi + x[2] * x[2] * ci * ci - 1.0;
  }
  n(x: Vec3, nn: Vec3): void {
    const ai = 1.0 / this.params[0], bi = 1.0 / this.params[1], ci = 1.0 / this.params[2];
    nn[0] = 2 * x[0] * ai * ai;
    nn[1] = 2 * x[1] * bi * bi;
    nn[2] = 2 * x[2] * ci * ci;
  }
}

class ManifoldTorus extends Manifold {
  readonly id = 'torus';
  readonly nparams = 2;
  g(x: Vec3): number {
    const R = this.params[0], r = this.params[1];
    if (R < r) throw new StyleError('Large radius < small radius!');
    const rad = Math.sqrt(x[0] * x[0] + x[1] * x[1]);
    const c = R - rad;
    return c * c + x[2] * x[2] - r * r;
  }
  n(x: Vec3, nn: Vec3): void {
    const R = this.params[0], r = this.params[1];
    if (R < r) throw new StyleError('Large radius < small radius!');
    const rad = Math.sqrt(x[0] * x[0] + x[1] * x[1]);
    const c = R - rad;
    const fac = c / rad;
    nn[0] = -2.0 * fac * x[0];
    nn[1] = -2.0 * fac * x[1];
    nn[2] = 2.0 * x[2];
  }
}

/** -x^2 - y^2 + (a^2 - (z/c)^2) (1 + (A sin(B z^2))^4); params { a, A, B, c }. */
class ManifoldDumbbell extends Manifold {
  readonly id = 'dumbbell';
  readonly nparams = 4;
  g(x: Vec3): number {
    const [a, A, B, c] = this.params;
    return -1.0 * (x[0] * x[0] + x[1] * x[1]) + (a * a - x[2] * x[2] / (c * c)) * (1.0 + libm.pow(A * libm.sinAlone(B * x[2] * x[2]), 4));
  }
  n(x: Vec3, nn: Vec3): void {
    const [a, A, B, c] = this.params;
    nn[0] = -2.0 * x[0];
    nn[1] = -2.0 * x[1];
    // cos and sin of the same argument: sincos in the native build
    const arg = B * x[2] * x[2];
    nn[2] = 8.0 * A * A * A * A * B * x[2] * (a * a - (x[2] * x[2] / (c * c))) * libm.cos(arg)
      * libm.pow(libm.sin(arg), 3) - 2 * x[2] * (1 + libm.pow(A * libm.sin(arg), 4)) / (c * c);
  }
}

/** -x^2 - y^2 + (a^2 - (z/c)^2) (1 + (A sin(const(z) z^2))^power), const(z) = z > 0 ? B : B2; params { a, A, B, B2, c }. */
class ManifoldSpine extends Manifold {
  readonly id: string = 'spine';
  readonly nparams = 5;
  protected power = 4;
  private shape(x: Vec3): { BB: number; c2: number } {
    const B = this.params[2], B2 = this.params[3], c = this.params[4];
    let BB: number, cc: number;
    if (x[2] > 0) { BB = B; cc = c; } else { BB = B2; cc = 1.0; }
    return { BB, c2: 1.0 / (cc * cc) };
  }
  override gAndN(x: Vec3, nn: Vec3): number {
    const a = this.params[0], A = this.params[1];
    const { BB, c2 } = this.shape(x);
    const xy2 = x[0] * x[0] + x[1] * x[1];
    const z2 = x[2] * x[2];
    const azc = a * a - z2 * c2;
    let As = libm.sin(BB * z2);
    let Ac = libm.cos(BB * z2);
    As *= A;
    Ac *= A;
    const Apart = powint(As, this.power - 1);
    const AMs = Apart * As;
    const AMc = Apart * Ac;
    const dAMs = this.power * AMc * 2.0 * BB * x[2];
    const dazc = -2.0 * x[2] * c2;
    nn[0] = -2 * x[0];
    nn[1] = -2 * x[1];
    nn[2] = azc * dAMs + (1.0 + AMs) * dazc;
    return -xy2 + azc * (1.0 + AMs);
  }
  n(x: Vec3, nn: Vec3): void {
    const a = this.params[0], A = this.params[1];
    const { BB, c2 } = this.shape(x);
    const z2 = x[2] * x[2];
    const azc = a * a - z2 * c2;
    let As = libm.sin(BB * z2);
    let Ac = libm.cos(BB * z2);
    As *= A;
    Ac *= A;
    const Apart = powint(As, this.power - 1);
    const AMs = Apart * As;
    const AMc = Apart * Ac;
    const dAMs = this.power * AMc * 2.0 * BB * x[2];
    const dazc = -2.0 * x[2] * c2;
    nn[0] = -2 * x[0];
    nn[1] = -2 * x[1];
    nn[2] = azc * dAMs + (1.0 + AMs) * dazc;
  }
  g(x: Vec3): number {
    const a = this.params[0], A = this.params[1];
    const { BB, c2 } = this.shape(x);
    const xy2 = x[0] * x[0] + x[1] * x[1];
    const z2 = x[2] * x[2];
    const azc = a * a - z2 * c2;
    let As = libm.sinAlone(BB * z2);
    As *= A;
    const Apart = powint(As, this.power - 1);
    const AMs = Apart * As;
    return -xy2 + azc * (1.0 + AMs);
  }
}

class ManifoldSpineTwo extends ManifoldSpine {
  override readonly id: string = 'spine/two';
  constructor() {
    super();
    this.power = 2;
  }
}

class ManifoldSupersphere extends Manifold {
  readonly id = 'supersphere';
  readonly nparams = 2;
  private static sign(a: number): number { return (a > 0 ? 1 : 0) - (a < 0 ? 1 : 0); }
  g(x: Vec3): number {
    const R = this.params[0], q = this.params[1];
    const rr = libm.pow(Math.abs(x[0]), q) + libm.pow(Math.abs(x[1]), q) + libm.pow(Math.abs(x[2]), q);
    return rr - libm.pow(R, q);
  }
  n(x: Vec3, nn: Vec3): void {
    const q = this.params[1];
    nn[0] = q * ManifoldSupersphere.sign(x[0]) * libm.pow(Math.abs(x[0]), q - 1);
    nn[1] = q * ManifoldSupersphere.sign(x[1]) * libm.pow(Math.abs(x[1]), q - 1);
    nn[2] = q * ManifoldSupersphere.sign(x[2]) * libm.pow(Math.abs(x[2]), q - 1);
  }
}

/** The cubic Hermite taper of manifold_gaussian_bump.cpp (x and y polynomials in t, with Newton inversion of x(t)). */
class CubicHermite {
  readonly a: number; readonly b: number; readonly c: number; readonly d: number;
  readonly s: number; readonly u: number; readonly v: number; readonly w: number;
  constructor(readonly x0: number, readonly x1: number, readonly y0: number, readonly y1: number, yp0: number, yp1: number,
    private readonly warn: (m: string) => void) {
    this.a = 2 * x0 + 2 - 2 * x1;
    this.b = -3 * x0 - 3 + 3 * x1;
    this.c = 1.0;
    this.d = x0;
    this.s = 2 * y0 - 2 * y1 + yp0 + yp1;
    this.u = -3 * y0 + 3 * y1 - 2 * yp0 - yp1;
    this.v = yp0;
    this.w = y0;
    if (Math.abs(this.x(0) - x0) > 1e-8) throw new StyleError('x0 wrong');
    if (Math.abs(this.x(1) - x1) > 1e-8) throw new StyleError('x1 wrong');
    if (Math.abs(this.y(0) - y0) > 1e-8) throw new StyleError('y0 wrong');
    if (Math.abs(this.y(1) - y1) > 1e-8) throw new StyleError('y1 wrong');
  }
  tFromX(xx: number): number {
    if (xx < this.x0 || xx > this.x1) throw new StyleError(`x ( ${xx} ) out of bounds [${this.x0}, ${this.x1}]`);
    let t = xx - this.x0;
    const dx = this.x1 - this.x0;
    t /= dx;
    const tol = 1e-8;
    const maxit = 500;
    let ff = this.x(t) - xx;
    let ffp = this.xp(t);
    for (let i = 0; i < maxit; ++i) {
      t -= ff / ffp;
      ff = this.x(t) - xx;
      ffp = this.xp(t);
      if (Math.abs(ff) < tol) return t;
    }
    this.warn('Convergence failed');
    return t;
  }
  x(t: number): number { const t2 = t * t, t3 = t2 * t; return this.a * t3 + this.b * t2 + this.c * t + this.d; }
  y(t: number): number { const t2 = t * t, t3 = t2 * t; return this.s * t3 + this.u * t2 + this.v * t + this.w; }
  xp(t: number): number { const t2 = t * t; return 3 * this.a * t2 + 2 * this.b * t + this.c; }
  yp(t: number): number { const t2 = t * t; return 3 * t2 * this.s + 2 * this.u * t + this.v; }
}

/**
 * z = f(r), f = A exp(-r^2 / 2 l^2) below rc1, a cubic Hermite taper (from a 1024-bin look-up table) up to rc2, 0
 * beyond; params { A, l, rc1, rc2 }. The table is built in post_param_init, which the fixes call once with the
 * constant parameters (a v_ parameter is still 0 then, as native).
 */
class ManifoldGaussianBump extends Manifold {
  readonly id = 'gaussian_bump';
  readonly nparams = 4;
  private AA = 0; private ll = 0; private ll2 = 0;
  private rc12 = 0; private rc22 = 0;
  private lutX0 = 0; private lutDx = 0;
  private lutZ = new Float64Array(0);
  private lutZp = new Float64Array(0);
  constructor(private readonly warn: (m: string) => void) { super(); }
  private bumpX2(x2: number): number { return this.AA * libm.exp(-x2 / this.ll2); }
  g(x: Vec3): number {
    const x2 = x[0] * x[0] + x[1] * x[1] + 0.0 * 0.0;
    if (x2 < this.rc12) return x[2] - this.bumpX2(x2);
    if (x2 < this.rc22) return x[2] - this.lut(this.lutZ, Math.sqrt(x2));
    return x[2];
  }
  n(x: Vec3, nn: Vec3): void {
    nn[2] = 1.0;
    const x2 = x[0] * x[0] + x[1] * x[1] + 0.0 * 0.0;
    if (x2 < this.rc12) {
      let factor = this.bumpX2(x2);
      factor /= this.ll * this.ll;
      nn[0] = factor * x[0];
      nn[1] = factor * x[1];
    } else if (x2 < this.rc22) {
      const rr = Math.sqrt(x2);
      const derPart = this.lut(this.lutZp, rr) * (1.0 / rr);
      nn[0] = derPart * x[0];
      nn[1] = derPart * x[1];
    } else {
      nn[0] = nn[1] = 0.0;
    }
  }
  override gAndN(x: Vec3, nn: Vec3): number {
    nn[2] = 1.0;
    const x2 = x[0] * x[0] + x[1] * x[1] + 0.0 * 0.0;
    if (x2 < this.rc12) {
      const gb = this.bumpX2(x2);
      const factor = gb / (this.ll * this.ll);
      nn[0] = factor * x[0];
      nn[1] = factor * x[1];
      return x[2] - gb;
    }
    if (x2 < this.rc22) {
      const rr = Math.sqrt(x2);
      const z = this.lut(this.lutZ, rr);
      const derPart = this.lut(this.lutZp, rr) * (1.0 / rr);
      nn[0] = derPart * x[0];
      nn[1] = derPart * x[1];
      return x[2] - z;
    }
    nn[0] = nn[1] = 0.0;
    return x[2];
  }
  override postParamInit(): void {
    this.AA = this.params[0];
    this.ll = this.params[1];
    const rc1 = this.params[2], rc2 = this.params[3];
    this.ll2 = 2.0 * this.ll * this.ll;
    this.rc12 = rc1 * rc1;
    this.rc22 = rc2 * rc2;
    const fAtRc = this.bumpX2(this.rc12);
    const fpAtRc = this.bumpX2(rc1 * rc1) * (-rc1 / (this.ll * this.ll));
    // make_lut
    this.lutX0 = rc1;
    const nbins = 1024;
    this.lutZ = new Float64Array(nbins + 1);
    this.lutZp = new Float64Array(nbins + 1);
    this.lutDx = (rc2 - rc1) / nbins;
    const pchip = new CubicHermite(rc1, rc2, fAtRc, 0.0, fpAtRc, 0.0, this.warn);
    let xx = rc1;
    for (let i = 0; i <= nbins; ++i) {
      const t = pchip.tFromX(xx);
      this.lutZ[i] = pchip.y(t);
      this.lutZp[i] = pchip.yp(pchip.tFromX(xx));
      xx += this.lutDx;
    }
  }
  /** lut_get_z / lut_get_zp: linear interpolation in the table. */
  private lut(table: Float64Array, rr: number): number {
    const xs = rr - this.lutX0;
    const xss = xs / this.lutDx;
    const bin = Math.trunc(xss);
    const frac = xss - bin;
    return table[bin] * (1 - frac) + frac * table[bin + 1];
  }
}

const THYLA_PLANE = 1, THYLA_SPHERE = 2, THYLA_CYL = 3, THYLA_CYL_TO_PLANE = 4;

/** thyla_part (manifold_thylakoid_shared.cpp): one analytic piece of the thylakoid and its domain box. */
class ThylaPart {
  readonly params = new Float64Array(7);
  xlo = 0; xhi = 0; ylo = 0; yhi = 0; zlo = 0; zhi = 0;
  constructor(readonly type: number, args: number[]) {
    for (let i = 0; i < 7; i++) this.params[i] = args[i] ?? 0;
  }
  g(x: Vec3): number {
    const p = this.params;
    switch (this.type) {
      case THYLA_PLANE: return p[0] * (x[0] - p[3]) + p[1] * (x[1] - p[4]) + p[2] * (x[2] - p[5]);
      case THYLA_SPHERE: {
        const R2 = p[0] * p[0];
        const dx = x[0] - p[1], dy = x[1] - p[2], dz = x[2] - p[3];
        return dx * dx + dy * dy + dz * dz - R2;
      }
      case THYLA_CYL: {
        const dx = x[0] - p[3], dy = x[1] - p[4], dz = x[2] - p[5];
        return p[0] * dx * dx + p[1] * dy * dy + p[2] * dz * dz - p[6] * p[6];
      }
      default: {
        const X0 = p[0], R0 = p[1], R = p[2];
        const dx = x[0] - X0;
        const dyz = Math.sqrt(x[1] * x[1] + x[2] * x[2]);
        const rdyz = dyz - (R0 + R);
        const norm = 1.0 / (2.0 * R);
        return norm * (dx * dx + rdyz * rdyz - R * R);
      }
    }
  }
  n(x: Vec3, nn: Vec3): void {
    const p = this.params;
    switch (this.type) {
      case THYLA_PLANE: nn[0] = p[0]; nn[1] = p[1]; nn[2] = p[2]; return;
      case THYLA_SPHERE: nn[0] = 2 * (x[0] - p[1]); nn[1] = 2 * (x[1] - p[2]); nn[2] = 2 * (x[2] - p[3]); return;
      case THYLA_CYL:
        nn[0] = 2 * p[0] * (x[0] - p[3]);
        nn[1] = 2 * p[1] * (x[1] - p[4]);
        nn[2] = 2 * p[2] * (x[2] - p[5]);
        return;
      default: {
        const X0 = p[0], R0 = p[1], R = p[2];
        const s = p[6] > 0.0 ? 1.0 : -1.0;
        const dx = s * (x[0] - X0);
        const ryz = Math.sqrt(x[1] * x[1] + x[2] * x[2]);
        const norm = 1.0 / (2.0 * R);
        nn[0] = s * 2 * dx * norm;
        const constPart = 1.0 - (R0 + R) / ryz;
        nn[1] = 2 * x[1] * constPart * norm;
        nn[2] = 2 * x[2] * constPart * norm;
      }
    }
  }
}

interface Geom { pt: number[]; lo: number[]; hi: number[] }
const geom = (): Geom => ({ pt: [0, 0, 0], lo: [0, 0, 0], hi: [0, 0, 0] });
const copyGeom = (g: Geom): Geom => ({ pt: [...g.pt], lo: [...g.lo], hi: [...g.hi] });
/** thyla_part_geom::mirror along x. */
const mirrorX = (o: Geom): Geom => {
  const m = copyGeom(o);
  m.lo[0] = -o.hi[0];
  m.hi[0] = -o.lo[0];
  m.pt[0] = -o.pt[0];
  return m;
};
const fmt8 = (v: number): string => v.toFixed(8);

/** Two grana stacks joined by a bridge (manifold_thylakoid.cpp); params { wB, LB, lB }. */
class ManifoldThylakoid extends Manifold {
  readonly id = 'thylakoid';
  readonly nparams = 3;
  private parts: ThylaPart[] = [];
  constructor(private readonly box: { lo: ArrayLike<number>; hi: ArrayLike<number> }) { super(); }

  override postParamInit(): void {
    const lT = 3, LT = 15, pad = 3;
    const wB = this.params[0], LB = this.params[1], lB = this.params[2];
    if (wB + 2 * lB > LT) {
      throw new StyleError(`LT = ${LT} not large enough to accommodate bridge with wB = ${wB} and lB = ${lB}! ${wB + 2 * lB} > ${LT}\n`);
    }
    const x0 = -(0.5 * LB + lB + lT + LT + lT + pad);
    const y0 = -(0.5 * LT + lT + pad);
    let z0 = -15;
    if (x0 < this.box.lo[0]) throw new StyleError(`Thylakoid expects xlo of at most ${fmt8(x0)}, but found ${fmt8(this.box.lo[0])}`);
    if (y0 < this.box.lo[1]) throw new StyleError(`Thylakoid expects ylo of at most ${fmt8(y0)}, but found ${fmt8(this.box.lo[1])}`);
    z0 -= pad;
    const x1 = -x0, y1 = -y0, z1 = -z0;
    if (x1 > this.box.hi[0]) throw new StyleError(`Expected xhi larger than current box has: ${fmt8(x1)} > ${fmt8(this.box.hi[0])}`);
    if (y1 > this.box.hi[1]) throw new StyleError(`Expected yhi larger than current box has: ${fmt8(y1)} > ${fmt8(this.box.hi[1])}`);
    const parts: ThylaPart[] = [];
    const add = (p: ThylaPart, g: Geom) => { this.setDomain(p, g.lo, g.hi); parts.push(p); };
    const plane = (a: number, b: number, c: number, pt: number[]) => new ThylaPart(THYLA_PLANE, [a, b, c, pt[0], pt[1], pt[2]]);
    const cyl = (a: number, b: number, c: number, pt: number[], R: number) => new ThylaPart(THYLA_CYL, [a, b, c, pt[0], pt[1], pt[2], R]);
    const cylToPlane = (X0: number, R0: number, R: number, s: number, pt: number[]) => new ThylaPart(THYLA_CYL_TO_PLANE, [X0, R0, R, pt[0], pt[1], pt[2], s]);

    const rB = 0.5 * wB;
    const Reff = rB + lB;
    const safety = 1.0;
    const bl = geom();
    bl.pt = [-0.5 * LB, 0, 0];
    bl.lo = [bl.pt[0] - safety * lB, -(1.0 + safety) * Reff, -(1.0 + safety) * Reff];
    bl.hi = [bl.pt[0], (1.0 + safety) * Reff, (1.0 + safety) * Reff];
    add(cylToPlane(bl.pt[0], rB, lB, -1, bl.pt), bl);
    const br = geom();
    br.pt = [0.5 * LB, 0, 0];
    br.lo = [br.pt[0], -(1.0 + safety) * Reff, -(1.0 + safety) * Reff];
    br.hi = [br.pt[0] + safety * lB, (1.0 + safety) * Reff, (1.0 + safety) * Reff];
    add(cylToPlane(br.pt[0], rB, lB, 1, br.pt), br);
    const bc = geom();
    bc.lo = [bl.pt[0], -Reff, -Reff];
    bc.hi = [br.pt[0], Reff, Reff];
    add(cyl(0, 1, 1, bc.pt, rB), bc);

    const cllb = geom();
    cllb.lo = [x0, y0, z0];
    cllb.pt = [x0 + pad + lT, y0 + pad + lT, 0];
    cllb.hi = [cllb.pt[0], cllb.pt[1], z1];
    add(cyl(1, 1, 0, cllb.pt, lT), cllb);
    const cllt = copyGeom(cllb);
    cllt.lo[1] = y1 - pad - lT;
    cllt.hi[1] = y1;
    cllt.pt[1] = cllb.pt[1] + LT;
    add(cyl(1, 1, 0, cllt.pt, lT), cllt);
    const clrb = copyGeom(cllb);
    clrb.pt[0] += LT;
    clrb.lo[0] = clrb.pt[0];
    clrb.hi[0] = clrb.lo[0] + lT + lB;
    add(cyl(1, 1, 0, clrb.pt, lT), clrb);
    const clrt = copyGeom(clrb);
    clrt.pt[1] += LT;
    clrt.lo[1] = y1 - pad - lT;
    clrt.hi[1] = y1;
    add(cyl(1, 1, 0, clrt.pt, lT), clrt);
    const pll = geom();
    pll.pt = [x0 + pad, 0, 0];
    pll.lo = [x0, cllb.pt[1], z0];
    pll.hi = [pll.lo[0] + pad + lT, pll.lo[1] + LT, z1];
    add(plane(1, 0, 0, pll.pt), pll);
    const plb = geom();
    plb.pt = [x0 + pad + lT + 0.5 * LT, y0 + pad, 0];
    plb.lo = [x0 + pad + lT, y0, z0];
    plb.hi = [plb.lo[0] + LT, plb.lo[1] + pad + lT, z1];
    add(plane(0, 1, 0, plb.pt), plb);
    const plt = copyGeom(plb);
    plt.lo[1] = cllb.pt[1] + LT;
    plt.hi[1] = y1;
    plt.pt[1] = y1 - pad;
    add(plane(0, 1, 0, plt.pt), plt);
    const plr = copyGeom(pll);
    plr.lo[0] = bl.lo[0] - 0.5;
    plr.lo[1] = y0 - pad;
    plr.hi[0] = bl.lo[0] + 0.5;
    plr.hi[1] = y1 + pad;
    plr.pt = [bl.pt[0] - lB, 0.0, 0.0];
    plr.hi[2] = z1 + pad;
    plr.lo[2] = z0 - pad;
    add(plane(1, 0, 0, plr.pt), plr);
    if (Math.abs(plr.pt[0] - bl.pt[0] + lB) > 1e-8) {
      throw new StyleError(`Origins of plane left right and bridge left misaligned! ${plr.pt[0]} != ${bl.pt[0] - lB}!\n`);
    }
    for (const [g, isCyl, a, b] of [[cllb, true, 1, 1], [cllt, true, 1, 1], [clrb, true, 1, 1], [clrt, true, 1, 1]] as const) {
      const m = mirrorX(g);
      add(isCyl ? cyl(a, b, 0, m.pt, lT) : plane(a, b, 0, m.pt), m);
    }
    const prl = mirrorX(pll);
    add(plane(1, 0, 0, prl.pt), prl);
    const prb = mirrorX(plb);
    add(plane(0, 1, 0, prb.pt), prb);
    const prt = mirrorX(plt);
    add(plane(0, 1, 0, prt.pt), prt);
    const prr = mirrorX(plr);
    add(plane(1, 0, 0, prr.pt), prr);
    if (Math.abs(prr.pt[0] - br.pt[0] - lB) > 1e-8) {
      throw new StyleError(`Origins of plane left right and bridge left misaligned! ${prr.pt[0]} != ${br.pt[0] + lB}!\n`);
    }
    this.parts = parts;
  }

  private setDomain(p: ThylaPart, lo: number[], hi: number[]): void {
    if (lo[0] >= hi[0]) throw new StyleError(`xlo >= xhi (${fmt8(lo[0])} >= ${fmt8(hi[0])})`);
    if (lo[1] >= hi[1]) throw new StyleError(`ylo >= yhi (${fmt8(lo[1])} >= ${fmt8(hi[1])})`);
    if (lo[2] >= hi[2]) throw new StyleError(`zlo >= zhi (${fmt8(lo[2])} >= ${fmt8(hi[2])})`);
    p.xlo = lo[0]; p.ylo = lo[1]; p.zlo = lo[2];
    p.xhi = hi[0]; p.yhi = hi[1]; p.zhi = hi[2];
  }

  private inDomain(p: ThylaPart, x: Vec3): boolean {
    if (!(x[0] >= p.xlo && x[0] <= p.xhi && x[1] >= p.ylo && x[1] <= p.yhi && x[2] >= p.zlo && x[2] <= p.zhi)) return false;
    if (p.type === THYLA_CYL_TO_PLANE) {
      const RR = p.params[1] + p.params[2];
      return x[1] * x[1] + x[2] * x[2] < RR * RR;
    }
    return true;
  }

  private part(x: Vec3): ThylaPart {
    for (const p of this.parts) if (this.inDomain(p, x)) return p;
    throw new StyleError(`Could not find thyla_part for x = (${x[0]},${x[1]},${x[2]})`);
  }

  g(x: Vec3): number {
    const p = this.part(x);
    const v = p.g(x);
    if (Number.isFinite(v)) return v;
    throw new StyleError(`Error, thyla_part of type ${p.type} returned ${v} as constraint val!`);
  }

  n(x: Vec3, nn: Vec3): void {
    const p = this.part(x);
    p.n(x, nn);
    if (!(Number.isFinite(nn[0]) && Number.isFinite(nn[1]) && Number.isFinite(nn[2]))) {
      throw new StyleError(`thyla_part of type ${p.type} returned (${nn[0]},${nn[1]},${nn[2]}) as gradient!`);
    }
  }
}

/** What a manifold needs from the system when it is created: the box (thylakoid) and the warning channel. */
export interface ManifoldContext {
  box: { lo: ArrayLike<number>; hi: ArrayLike<number> };
  warn: (message: string) => void;
}

/** create_manifold (manifold_factory.cpp): the manifold of that name, or null. */
export const createManifold = (name: string, ctx: ManifoldContext): Manifold | null => {
  switch (name) {
    case 'cylinder': return new ManifoldCylinder();
    case 'cylinder/dent': return new ManifoldCylinderDent();
    case 'dumbbell': return new ManifoldDumbbell();
    case 'ellipsoid': return new ManifoldEllipsoid();
    case 'gaussian_bump': return new ManifoldGaussianBump(ctx.warn);
    case 'plane': return new ManifoldPlane();
    case 'plane/wiggle': return new ManifoldPlaneWiggle();
    case 'sphere': return new ManifoldSphere();
    case 'supersphere': return new ManifoldSupersphere();
    case 'spine': return new ManifoldSpine();
    case 'spine/two': return new ManifoldSpineTwo();
    case 'thylakoid': return new ManifoldThylakoid(ctx.box);
    case 'torus': return new ManifoldTorus();
    default: return null;
  }
};
