/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/COLLOID/fix_wall_colloid.cpp and
 * src/COLLOID/fix_wall_colloid.h (Contributing author: Jeremy Lechman (SNL)); the base class is src/fix_wall.cpp at the
 * same commit (Copyright (2003) Sandia Corporation, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * fix wall/colloid (docs.lammps.org/fix_wall.html, wall/colloid): the interaction of finite-size particles (their
 * radius is the particle size) with a flat wall, integrated over the particle's sphere; the energy is shifted to zero at
 * the cutoff for each particle size. Arguments: face coord epsilon sigma cutoff, as the other fix wall styles.
 *
 * Native formulas (fix_wall_colloid.cpp precompute and wall_particle, operation order kept):
 *   coeff1 = 4/315 eps sigma^6, coeff2 = 2/3 eps, coeff3 = eps sigma^6 / 7560, coeff4 = eps / 6,
 *   with delta the distance of the particle centre from the wall and rad its radius, the force on the wall-side
 *   interaction and the energy are the two closed forms of fix_wall_colloid.cpp (wall_particle).
 * Errors: "Fix wall/colloid requires atom attribute radius", "Fix wall/colloid requires extended particles" (a group
 * particle with radius 0), "Particle on or inside fix wall surface" (native error->one text).
 *
 * The energy, the per-wall force and the virial are tallied here (the base class keeps its own tallies with the
 * point-particle formula, which does not depend on the particle radius).
 */
import { FixWallBase } from '../../engine/fix/walls';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { valueOf } from '../../engine/fix/util';
import { log as libmLog } from '../libm/log';

/** MathSpecial::powint (math_special.h): integer power by repeated squaring, as native (exact operation order). */
const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

export class FixWallColloid extends FixWallBase {
  readonly style = 'wall/colloid';
  private readonly coeff1 = new Float64Array(6);
  private readonly coeff2 = new Float64Array(6);
  private readonly coeff3 = new Float64Array(6);
  private readonly coeff4 = new Float64Array(6);
  /** per wall: energy sum (native ewall[0]), force on each wall (native ewall[m+1]) */
  private colloidEnergy = 0;
  private readonly colloidForce = new Float64Array(7);
  private colloidVAtom = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'wall/colloid', 4);
  }

  /** Native FixWallColloid::init(): radius attribute, extended particles in the group, then FixWall::init(). */
  init(): void {
    const s = this.sys.state;
    if (!s.radius) throw new StyleError('Fix wall/colloid requires atom attribute radius');
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && s.radius[i] === 0.0) throw new StyleError('Fix wall/colloid requires extended particles');
    }
    super.init();
  }

  /** The base class's point-particle energy and force are not used: the colloid wall has its own loop (apply). */
  protected energyAt(_r: number, _w: unknown): number {
    throw new StyleError('fix wall/colloid: internal error (the point-particle energy is not used)');
  }

  protected forceAt(_r: number, _w: unknown): number {
    throw new StyleError('fix wall/colloid: internal error (the point-particle force is not used)');
  }

  /** Native precompute(m): the four coefficients from epsilon and sigma of wall m. */
  private precompute(m: number, epsilon: number, sigma: number): void {
    this.coeff1[m] = 4.0 / 315.0 * epsilon * powint(sigma, 6);
    this.coeff2[m] = 2.0 / 3.0 * epsilon;
    this.coeff3[m] = epsilon * powint(sigma, 6) / 7560.0;
    this.coeff4[m] = epsilon / 6.0;
  }

  /** Native post_force(): each wall in turn, variable epsilon/sigma re-evaluated (error if negative), then wall_particle. */
  protected apply(): void {
    const s = this.sys.state;
    const walls = this.walls;
    const coord = this.coords();
    this.colloidEnergy = 0;
    this.colloidForce.fill(0);
    if (this.thermoVirial) this.virial.fill(0);
    if (this.colloidVAtom.length !== 6 * s.n) this.colloidVAtom = new Float64Array(6 * s.n);
    else this.colloidVAtom.fill(0);
    for (let m = 0; m < walls.length; m++) {
      const w = walls[m];
      const epsilon = valueOf(this.sys, w.params[0]);
      const sigma = valueOf(this.sys, w.params[1]);
      if (epsilon < 0.0 || sigma < 0.0) throw new StyleError('Variable evaluation in fix wall gave bad value');
      this.precompute(m, epsilon, sigma);
      this.wallParticle(m, w, coord[m]);
    }
  }

  /**
   * Native FixWallColloid::wall_particle(m, which, coord): the force and energy of every group particle with the wall,
   * and the error for a particle touching or penetrating the wall (after the loop, as native).
   */
  private wallParticle(m: number, w: { dim: number; lo: boolean; cutoff: number }, coord: number): void {
    const s = this.sys.state;
    const radius = s.radius!;
    const dim = w.dim;
    const side = w.lo ? -1 : 1;
    const cutoff = w.cutoff;
    const coeff1 = this.coeff1[m], coeff2 = this.coeff2[m], coeff3 = this.coeff3[m], coeff4 = this.coeff4[m];
    let onflag = false;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const delta = side < 0 ? s.x[3 * i + dim] - coord : coord - s.x[3 * i + dim];
      if (delta >= cutoff) continue;
      const rad = radius[i];
      if (rad >= delta) {
        onflag = true;
        continue;
      }
      const newCoeff2 = coeff2 * rad * rad * rad;
      const diam = 2.0 * rad;
      const rad2 = rad * rad;
      const rad4 = rad2 * rad2;
      const rad8 = rad4 * rad4;
      const delta2 = rad2 - delta * delta;
      const rinv = 1.0 / delta2;
      const r2inv = rinv * rinv;
      const r4inv = r2inv * r2inv;
      const r8inv = r4inv * r4inv;
      // clang-format off
      const fwall = side * (coeff1 * (rad8 * rad + 27.0 * rad4 * rad2 * rad * powint(delta, 2)
                                 + 63.0 * rad4 * rad * powint(delta, 4)
                                 + 21.0 * rad2 * rad * powint(delta, 6)) * r8inv -
                      newCoeff2 * r2inv);
      // clang-format on
      s.f[3 * i + dim] -= fwall;
      // clang-format off
      let r2 = rad - delta;
      let rinv2 = 1.0 / r2;
      let r2inv2 = rinv2 * rinv2;
      let r4inv2 = r2inv2 * r2inv2;
      let r3 = delta + rad;
      let rinv3 = 1.0 / r3;
      let r2inv3 = rinv3 * rinv3;
      let r4inv3 = r2inv3 * r2inv3;
      this.colloidEnergy += coeff3 * ((-3.5 * diam + delta) * r4inv2 * r2inv2 * rinv2
                             + (3.5 * diam + delta) * r4inv3 * r2inv3 * rinv3) -
        coeff4 * ((diam * delta - r2 * r3 * (libmLog(-r2) - libmLog(r3))) *
                   (-rinv2) * rinv3);
      // clang-format on
      // offset depends on particle size
      r2 = rad - cutoff;
      rinv2 = 1.0 / r2;
      r2inv2 = rinv2 * rinv2;
      r4inv2 = r2inv2 * r2inv2;
      r3 = cutoff + rad;
      rinv3 = 1.0 / r3;
      r2inv3 = rinv3 * rinv3;
      r4inv3 = r2inv3 * r2inv3;
      // clang-format off
      const eoffset = coeff3 * ((-3.5 * diam + cutoff) * r4inv2 * r2inv2 * rinv2
                           + (3.5 * diam + cutoff) * r4inv3 * r2inv3 * rinv3) -
        coeff4 * ((diam * cutoff - r2 * r3 * (libmLog(-r2) - libmLog(r3))) *
                   (-rinv2) * rinv3);
      // clang-format on
      this.colloidEnergy -= eoffset;
      this.colloidForce[m + 1] += fwall;
      // v_tally(dim, i, vn): the virial component of this wall (native fix v_tally, global and per atom)
      const vn = side < 0 ? -fwall * delta : fwall * delta;
      if (this.thermoVirial) this.virial[dim] += vn;
      this.colloidVAtom[6 * i + dim] += vn;
    }
    if (onflag) throw new StyleError('Particle on or inside fix wall surface');
  }

  energy(): number { return this.colloidEnergy; }

  virialAtom(out: Float64Array): void {
    for (let k = 0; k < this.colloidVAtom.length && k < out.length; k++) out[k] += this.colloidVAtom[k];
  }

  /** f_ID: the energy sum (native compute_scalar: ewall_all[0]). */
  computeScalar(): number { return this.colloidEnergy; }

  /** f_ID[n]: the force on wall n (native compute_vector(n): ewall_all[n+1]). */
  computeVector(i: number): number { return this.colloidForce[i + 1]; }
}
