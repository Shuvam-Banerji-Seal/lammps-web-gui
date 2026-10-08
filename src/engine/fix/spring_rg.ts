import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * fix spring/rg — written from docs.lammps.org/fix_spring_rg.html
 * (plans/lammps-docs/fix_spring_rg.rst). Syntax: "fix ID group-ID spring/rg
 * K RG0"; "RG0 = target radius of gyration to constrain to (distance units)";
 * "if RG0 = NULL, use the current RG as the target value".
 *
 * Energy and force from the page:
 *   E = K (R_G - R_G0)^2
 * "Apply a harmonic restraining force to atoms in the group to affect their
 * central moment about the center of mass (radius of gyration)." "Note that K
 * is thus a force constant for the aggregate force on the group of atoms, not
 * a per-atom force." The page's force line is
 *   F_i = 2K (m_i / M) (1 - R_G0 / R_G) (x_i - x_cm)
 * with the sign as printed. The engine uses the force that equals -grad E of
 * the energy above (the printed sign would push atoms away from the centre
 * for R_G > R_G0); the oracle case checks the trajectory against native.
 * The (x_i - x_cm) term uses unwrapped coordinates ("taking into account
 * periodic boundary conditions").
 *
 * "If R_{G0} is specified as NULL, then the RG of the group is computed at the
 * time the fix is specified, and that value is used as the target."
 *
 * "This fix is not invoked during" energy minimization (so there is no minPostForce).
 *
 * Output: "The scalar is the reference" radius of gyration R_G0 used by the fix. The scalar is intensive. No
 * fix_modify option applies, so the energy does not enter the thermo pe.
 */

const num = (w: string | undefined, what: string): number => {
  if (w === undefined || w.trim() === '') throw new StyleError(`fix spring/rg: missing ${what}`);
  const v = Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`fix spring/rg: expected a number for ${what}, got '${w}'`);
  return v;
};

export class FixSpringRG extends Fix {
  readonly style = 'spring/rg';
  private readonly k: number;
  private readonly rg0: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 2) throw new StyleError('usage: fix ID group-ID spring/rg K RG0');
    this.k = num(args[0], 'K');
    this.scalarFlag = true;
    this.extscalar = 0;
    this.energyGlobal = false;
    if (args[1] === 'NULL') {
      const rg = this.radiusOfGyration();
      if (rg === null) throw new StyleError('fix spring/rg: group has no atoms (RG0 NULL)');
      this.rg0 = rg;
    } else {
      this.rg0 = num(args[1], 'RG0');
    }
  }

  /** Radius of gyration of the group about its centre of mass (unwrapped coordinates). */
  private radiusOfGyration(): number | null {
    const s = this.sys.state;
    const u = [0, 0, 0];
    let M = 0;
    const c = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      M += m;
      for (let d = 0; d < 3; d++) c[d] += m * u[d];
    }
    if (M === 0) return null;
    for (let d = 0; d < 3; d++) c[d] /= M;
    let r2 = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      for (let d = 0; d < 3; d++) r2 += m * (u[d] - c[d]) * (u[d] - c[d]);
    }
    return Math.sqrt(r2 / M);
  }

  postForce(): void {
    const s = this.sys.state;
    const u = [0, 0, 0];
    let M = 0;
    const c = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      M += m;
      for (let d = 0; d < 3; d++) c[d] += m * u[d];
    }
    if (M === 0) return;
    for (let d = 0; d < 3; d++) c[d] /= M;
    const rg = this.radiusOfGyration();
    if (rg === null || rg === 0) return;
    // E = K (rg - rg0)^2; dE/dx_i = 2K (1 - rg0/rg) (m_i/M)(x_i - c); F = -dE/dx_i
    const pref = -2 * this.k * (1 - this.rg0 / rg);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const w = (pref * massOf(s, i)) / M;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      for (let d = 0; d < 3; d++) s.f[3 * i + d] += w * (u[d] - c[d]);
    }
  }

  computeScalar(): number { return this.rg0; }
}
