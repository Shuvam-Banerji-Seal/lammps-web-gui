import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { libm } from '../libm';

/*
 * fix rhok — written from docs.lammps.org/fix_rhok.html (plans/lammps-docs-live/fix_rhok.rst). Syntax quoted there:
 *   "fix ID group-ID rhok nx ny nz K a"
 * "nx, ny, nz = k-vector of collective density field"; "K = spring constant of bias potential";
 * "a = anchor point of bias potential".
 *
 * The doc's potential (the math in the rst, not quoted): U = 1/2 K (|rho_k| - a)^2 with
 * rho_k = sum_j exp(-i k . r_j) / sqrt(N) and k = (2 pi n_x / L_x, 2 pi n_y / L_y, 2 pi n_z / L_z). The force is -dU/dr_j with the gradient
 *   d|rho|/dr_j = Re( conj(rho)/|rho| * (-i k exp(-i k.r_j) / sqrt(N)) ),
 * which the engine evaluates directly (checked against the native forces to 5e-16 on eight atoms).
 *
 * Measured with native LAMMPS (black box):
 *   - N in the sum and in sqrt(N) is the number of atoms in the fix group (a group of 4 atoms gives
 *     U = 0.000701337888711706 for the value the python reference gives with sqrt(4)).
 *   - The vector f_ID[1..3] is (Re rho_k, Im rho_k, |rho_k|), printed unnormalised.
 *   - The scalar f_ID is U, printed unnormalised (no thermo division by the atom count). The doc says the
 *     scalar is extensive; the printed value does not show that division, so the engine follows the measurement.
 *   - fix_modify energy yes adds U to pe; the thermo pe then divides it by the atom count (the lj-units
 *     normalisation of pe): pe changes by U/8 for eight atoms.
 *
 * "This fix is not invoked during energy minimization" (doc): no minimization hook.
 * The k vector uses the current box lengths (npt changes them each step). A tilted box is refused.
 */

export class FixRhok extends Fix {
  readonly style = 'rhok';
  private readonly n: [number, number, number];
  private readonly K: number;
  private readonly a: number;
  private u = 0;
  private vec = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 5) throw new StyleError('fix rhok: usage: fix ID group-ID rhok nx ny nz K a');
    const ints = args.slice(0, 3).map((w, d) => {
      const v = Number(w);
      if (!Number.isInteger(v)) throw new StyleError(`fix rhok: n${'xyz'[d]} must be an integer, got '${w}'`);
      return v;
    });
    this.n = [ints[0], ints[1], ints[2]];
    // K and a: numbers only (the doc does not give variable forms)
    this.K = Number(args[3]);
    this.a = Number(args[4]);
    if (!Number.isFinite(this.K) || !Number.isFinite(this.a)) throw new StyleError(`fix rhok: K and a must be numbers, got '${args[3]}' and '${args[4]}'`);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 0;
    this.extvector = 0;
    this.energyGlobal = true;
  }

  postForce(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    if (g.xy !== 0 || g.xz !== 0 || g.yz !== 0) throw new StyleError('fix rhok: a tilted (triclinic) box is not supported');
    const kx = (2 * Math.PI * this.n[0]) / g.lx;
    const ky = (2 * Math.PI * this.n[1]) / g.ly;
    const kz = (2 * Math.PI * this.n[2]) / g.lz;
    // members of the group: N = their count
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) if (this.inGroup(i)) idx.push(i);
    const N = idx.length;
    if (N === 0) { this.u = 0; this.vec.fill(0); return; }
    const sq = Math.sqrt(N);
    const ph = new Float64Array(N);
    let re = 0, im = 0;
    for (let m = 0; m < N; m++) {
      const i = idx[m];
      const p = kx * s.x[3 * i] + ky * s.x[3 * i + 1] + kz * s.x[3 * i + 2];
      ph[m] = p;
      re += libm.cos(p);
      im += -libm.sin(p);
    }
    re /= sq;
    im /= sq;
    const absr = Math.sqrt(re * re + im * im);
    const dev = absr - this.a;
    this.u = 0.5 * this.K * dev * dev;
    this.vec[0] = re;
    this.vec[1] = im;
    this.vec[2] = absr;
    if (absr === 0) return;
    // F_j = -K (|rho| - a) d|rho|/dr_j; d|rho|/dr_j = Re(conj(rho)/|rho| * (-i k) exp(-i k.r_j) / sqrt(N))
    // with conj(rho)/|rho| = (re - i im)/absr; the factor -i k exp(-i p) = -i k (cos p - i sin p) = k(-sin p - i cos p)
    const cr = re / absr, ci = -im / absr;
    for (let m = 0; m < N; m++) {
      const i = idx[m];
      const p = ph[m];
      const ex = -libm.sin(p), ey = -libm.cos(p); // -i exp(-i p) = -sin p - i cos p
      // (cr + i ci)(ex + i ey) real part = cr ex - ci ey
      const realFac = (cr * ex - ci * ey) / sq;
      const fac = -this.K * dev * realFac;
      s.f[3 * i] += fac * kx;
      s.f[3 * i + 1] += fac * ky;
      s.f[3 * i + 2] += fac * kz;
    }
  }

  energy(): number { return this.u; }
  computeScalar(): number { return this.u; }
  computeVector(i: number): number {
    if (i < 0 || i > 2) throw new StyleError(`fix ${this.id} rhok vector index ${i + 1} out of range`);
    return this.vec[i];
  }
}
