import { takeFixRecord } from './restartRecord';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';
import { numeric } from '../force/util';

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
 * time the fix is specified, and that value is used as the target." As native LAMMPS does
 * (src/EXTRA-FIX/fix_spring_rg.cpp init and post_force at b332f28), the NULL target is the RG at the first
 * init (the first run), 0 for a group without mass (a dynamic group is empty before its first update), and
 * the group mass is that of every init, recomputed at every step for a dynamic group. RG and the force follow
 * Group::xcm and Group::gyration: rg^2 = sum (dx^2 + dy^2 + dz^2) m / M, F_i -= 2K (1 - rg0/rg) dx (m_i / M).
 *
 * "This fix is not invoked during" energy minimization (so there is no minPostForce).
 *
 * Output: "The scalar is the reference" radius of gyration R_G0 used by the fix. The scalar is intensive. No
 * fix_modify option applies, so the energy does not enter the thermo pe.
 */

// As native LAMMPS does (src/EXTRA-FIX/fix_spring_rg.cpp FixSpringRG::FixSpringRG at b332f28): K and RG0 are read with utils::numeric
const num = (w: string | undefined, _what: string): number => numeric(w);

export class FixSpringRG extends Fix {
  readonly style = 'spring/rg';
  private readonly k: number;
  private rg0: number;
  /** RG0 NULL: the target is taken at the first init. */
  private rg0Pending: boolean;
  /** Group mass of the last init (Group::mass at init; every step for a dynamic group). */
  private masstotal = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 2) throw new StyleError('usage: fix ID group-ID spring/rg K RG0');
    this.k = num(args[0], 'K');
    this.scalarFlag = true;
    this.extscalar = 0;
    this.energyGlobal = false;
    this.rg0Pending = args[1] === 'NULL';
    this.rg0 = this.rg0Pending ? 0 : num(args[1], 'RG0');
    // fix_spring_rg.cpp restart(): the reference rg0 of the run that wrote the file
    const saved = takeFixRecord(sys, id, this.style)?.state as { rg0: number } | undefined;
    if (saved) { this.rg0 = saved.rg0; this.rg0Pending = false; }
  }

  init(): void {
    this.masstotal = this.groupMass();
    if (this.rg0Pending) {
      this.rg0 = this.gyration(this.masstotal);
      this.rg0Pending = false;
    }
  }

  private groupMass(): number {
    const s = this.sys.state;
    let M = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) M += massOf(s, i);
    return M;
  }

  /** Group::xcm with the given mass: sum x_unwrapped m / M (0 when M is 0). */
  private xcm(M: number): number[] {
    const s = this.sys.state;
    const u = [0, 0, 0];
    const c = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      for (let d = 0; d < 3; d++) c[d] += u[d] * m;
    }
    if (M > 0.0) for (let d = 0; d < 3; d++) c[d] /= M;
    return c;
  }

  /** Group::gyration about the group's xcm with the given mass (0 when M is 0). */
  private gyration(M: number): number {
    const s = this.sys.state;
    const c = this.xcm(M);
    const u = [0, 0, 0];
    let rg = 0.0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const dx = u[0] - c[0], dy = u[1] - c[1], dz = u[2] - c[2];
      rg += (dx * dx + dy * dy + dz * dz) * massOf(s, i);
    }
    return M > 0.0 ? Math.sqrt(rg / M) : 0.0;
  }

  /** fix_spring_rg.cpp write_restart: rg0. */
  restartState(): { rg0: number } { return { rg0: this.rg0 }; }

  postForce(): void {
    const s = this.sys.state;
    if (this.sys.groups.isDynamic(this.groupBit)) this.masstotal = this.groupMass();
    const M = this.masstotal;
    const c = this.xcm(M);
    const rg = this.gyration(M);
    // rg == 0: no atoms in the group, or all of them on top of each other
    if (rg === 0.0 || M === 0.0) return;
    const term1 = 2.0 * this.k * (1.0 - this.rg0 / rg);
    const u = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const massfrac = massOf(s, i) / M;
      for (let d = 0; d < 3; d++) s.f[3 * i + d] -= term1 * (u[d] - c[d]) * massfrac;
    }
  }

  computeScalar(): number { return this.rg0; }
}
