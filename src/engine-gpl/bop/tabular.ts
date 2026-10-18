/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/tabular_function.cpp and
 * src/tabular_function.h (TabularFunction, used by pair_style bop). Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Cubic-spline table on a uniform grid: the same coefficient arrays and evaluation order as the native class.
 */

/** std::lround: halves round away from zero (Math.round rounds them towards +infinity). */
const lround = (v: number): number => (v < 0 ? -Math.round(-v) : Math.round(v));

export class TabularFunction {
  private size = 0;
  private xmin = 0;
  private xmax = 0;
  private xmaxsq = 0;
  private rdx = 0;
  private vmax = 0;
  private xs = new Float64Array(0);
  private ys = new Float64Array(0);
  private ys1 = new Float64Array(0);
  private ys2 = new Float64Array(0);
  private ys3 = new Float64Array(0);
  private ys4 = new Float64Array(0);
  private ys5 = new Float64Array(0);
  private ys6 = new Float64Array(0);
  /** Derivative of the last value() call (native y1 output). */
  dy = 0;

  /** TabularFunction::set_values: n samples of values on [x1, x2]. */
  setValues(n: number, x1: number, x2: number, values: ArrayLike<number>): void {
    this.resetSize(n);
    this.xmin = x1;
    this.xmax = x2;
    this.xmaxsq = x2 * x2;
    for (let i = 0; i < n; i++) this.ys[i] = values[i];
    this.initialize();
  }

  get xMax(): number { return this.xmax; }

  private resetSize(n: number): void {
    if (n !== this.size) {
      this.size = n;
      this.xs = new Float64Array(n);
      this.ys = new Float64Array(n);
      this.ys1 = new Float64Array(n);
      this.ys2 = new Float64Array(n);
      this.ys3 = new Float64Array(n);
      this.ys4 = new Float64Array(n);
      this.ys5 = new Float64Array(n);
      this.ys6 = new Float64Array(n);
    }
  }

  private initialize(): void {
    const size = this.size, ys = this.ys, ys1 = this.ys1, ys2 = this.ys2, ys3 = this.ys3;
    this.rdx = (this.xmax - this.xmin) / (size - 1.0);
    this.vmax = 0.0;
    for (let i = 0; i < size; i++) if (Math.abs(ys[i]) > this.vmax) this.vmax = Math.abs(ys[i]);
    for (let i = 0; i < size; i++) this.xs[i] = this.xmin + i * this.rdx;
    this.rdx = 1.0 / this.rdx;
    ys1[0] = ys[1] - ys[0];
    ys1[1] = 0.5 * (ys[2] - ys[0]);
    ys1[size - 2] = 0.5 * (ys[size - 1] - ys[size - 3]);
    ys1[size - 1] = ys[size - 1] - ys[size - 2];
    for (let i = 2; i < size - 2; i++) ys1[i] = ((ys[i - 2] - ys[i + 2]) + 8.0 * (ys[i + 1] - ys[i - 1])) / 12.0;
    for (let i = 0; i < size - 1; i++) {
      ys2[i] = 3.0 * (ys[i + 1] - ys[i]) - 2.0 * ys1[i] - ys1[i + 1];
      ys3[i] = ys1[i] + ys1[i + 1] - 2.0 * (ys[i + 1] - ys[i]);
    }
    ys2[size - 1] = 0.0;
    ys3[size - 1] = 0.0;
    for (let i = 0; i < size; i++) {
      this.ys4[i] = ys1[i] * this.rdx;
      this.ys5[i] = 2.0 * ys2[i] * this.rdx;
      this.ys6[i] = 3.0 * ys3[i] * this.rdx;
    }
  }

  /** TabularFunction::value: the spline value at x; the derivative is left in this.dy. */
  value(x: number): number {
    let ps = (x - this.xmin) * this.rdx;
    let ks = lround(ps);
    if (ks > this.size - 1) ks = this.size - 1;
    if (ks < 0) ks = 0;
    ps = ps - ks;
    const y = ((this.ys3[ks] * ps + this.ys2[ks]) * ps + this.ys1[ks]) * ps + this.ys[ks];
    this.dy = (this.ys6[ks] * ps + this.ys5[ks]) * ps + this.ys4[ks];
    return y;
  }
}
