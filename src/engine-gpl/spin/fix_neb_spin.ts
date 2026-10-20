/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/fix_neb_spin.cpp and
 * src/SPIN/fix_neb_spin.h (FixNEBSpin, fix neb/spin: the geodesic nudged elastic band force on the spins of the
 * replicas, Julien Tranchida (SNL)). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The fix's min_post_force is split here: the neb/spin command (neb_spin.ts) runs the replicas in lockstep and calls
 * postForceReplica(r, round) once per replica after every replica's forces and energies are known, which is what
 * native's inter-replica messages give the fix. Native's fix acts only inside the neb/spin command: a minimize with
 * fix neb/spin is refused (native would compute a meaningless single-replica force there).
 *
 * Quote, docs.lammps.org/fix_neb_spin.html: "Add nudging forces to spins in the group for a multi-replica simulation
 * run via the neb/spin command to perform a geodesic nudged elastic band (GNEB) calculation for finding the transition
 * state." As native's constructor does: parallel and perp are errors ("Illegal fix neb/spin command"), and end (two
 * values) and lattice (one value) are read past and change nothing (SpinLattice stays false, "no spin-lattice neb for
 * now", and the free-end switches stay off).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { spinContribute } from '../../engine/force/spin_book';
import type { System } from '../../engine/system';

/** Geodesic distance between two unit spins (Vincenty's formula, fix_neb_spin.cpp geodesic_distance). */
export const geodesicDistance = (spi: readonly number[], spj: readonly number[]): number => {
  const cx = spi[1] * spj[2] - spi[2] * spj[1];
  const cy = spi[2] * spj[0] - spi[0] * spj[2];
  const cz = spi[0] * spj[1] - spi[1] * spj[0];
  const normcross = Math.sqrt(cx * cx + cy * cy + cz * cz);
  const dots = spi[0] * spj[0] + spi[1] * spj[1] + spi[2] * spj[2];
  if (normcross === 0 && dots === 0) throw new StyleError('Incorrect calc. of geodesic_distance in Fix NEB/spin');
  return Math.atan2(normcross, dots);
};

/** What one neb/spin evaluation gives every replica's fix: the replicas in order and their energies (raw). */
export interface NebSpinRound {
  readonly systems: readonly System[];
  readonly energies: readonly number[];
}

export class FixNEBSpin extends Fix {
  readonly style = 'neb/spin';
  readonly kspring: number;
  /** climbing replica index (0-based), -1 while the regular stage runs (fix_neb_spin.cpp FixNEBSpin::init sets -1) */
  rclimber = -1;
  /** values of the last postForceReplica call (fix_neb_spin.h: veng, plen, nlen, gradlen, dotpath, dottangrad) */
  veng = 0;
  plen = 0;
  nlen = 0;
  gradlen = 0;
  dotpath = 0;
  dottangrad = 0;
  /** Set by the neb/spin command while it runs (the nudging is applied by the driver, not by minimize hooks). */
  active = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError('Illegal fix neb/spin command');
    if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(args[0])) throw new StyleError(`Expected floating point parameter instead of '${args[0]}' in input script or data file`);
    this.kspring = Number(args[0]);
    // native's text names fix neb here (fix_neb_spin.cpp at b332f28)
    if (!(this.kspring > 0)) throw new StyleError('Illegal fix neb command');
    for (let k = 1; k < args.length;) {
      const key = args[k];
      if (key === 'end') { k += 3; continue; }
      if (key === 'lattice') { k += 2; continue; }
      throw new StyleError('Illegal fix neb/spin command');
    }
  }

  // minimize calls this hook; the geodesic force exists only inside the neb/spin command, so anywhere else the fix
  // is refused rather than ignored
  minPostForce(): void {
    if (!this.active) throw new StyleError(`fix neb/spin ${this.id} acts only inside a neb/spin command; minimize with fix neb/spin is not supported`);
  }

  /**
   * FixNEBSpin::min_post_force for replica r (fix_neb_spin.cpp at b332f28): the tangent of the best path from the
   * neighbouring replicas' spins and energies, the geodesic lengths, the spring and climbing forces, and the projection
   * of the nudged force onto the spin's tangent plane. The initial and final replicas keep their forces.
   */
  postForceReplica(r: number, round: NebSpinRound): void {
    const systems = round.systems;
    const nrep = systems.length;
    const s = systems[r].state;
    const sp = s.sp!, fm = s.fm!;
    // the engine keeps fm as the sum of its contributors' last values (force/spin_book.ts), where native clears fm at
    // every force evaluation (Min::force_clear): the previous nudging is withdrawn first, so fm is this evaluation's
    // field of the pair styles and fixes, and the new nudging is added below as this fix's contribution
    spinContribute(s, this, new Float64Array(3 * s.n));
    const E = round.energies;
    const veng = E[r];
    // vprev and vnext are the energies of the neighbouring replicas (native: vprev = vnext = veng at the ends)
    const vprev = r > 0 ? E[r - 1] : veng;
    const vnext = r < nrep - 1 ? E[r + 1] : veng;
    const group: number[] = [];
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) group.push(i);
    // the group atoms of the neighbouring replicas, matched by atom ID (native: the neb atoms sent with their IDs)
    const spOf = (other: System, id: number): number[] => {
      const j = other.indexOfId(id);
      if (j < 0) throw new StyleError(`neb/spin: atom ${id} of replica ${r + 1} is missing from a neighbouring replica`);
      const o = other.state.sp!;
      return [o[4 * j], o[4 * j + 1], o[4 * j + 2]];
    };
    const prevSys = r > 0 ? systems[r - 1] : null;
    const nextSys = r < nrep - 1 ? systems[r + 1] : null;

    let plen = 0, nlen = 0, tlen = 0, gradlen = 0, dotpath = 0, dottangrad = 0;
    const tangent = new Float64Array(3 * s.n);
    for (const i of group) {
      const si = [sp[4 * i], sp[4 * i + 1], sp[4 * i + 2]];
      const f = [fm[3 * i], fm[3 * i + 1], fm[3 * i + 2]];
      const spPrev = prevSys ? spOf(prevSys, s.id[i]) : null;
      const spNext = nextSys ? spOf(nextSys, s.id[i]) : null;
      if (r === nrep - 1) {
        // final replica: the difference to the previous replica, projected on the tangent plane
        const dp = [si[0] - spPrev![0], si[1] - spPrev![1], si[2] - spPrev![2]];
        const dd = dp[0] * si[0] + dp[1] * si[1] + dp[2] * si[2];
        for (let k = 0; k < 3; k++) dp[k] -= dd * si[k];
        const t = geodesicDistance(si, spPrev!);
        plen += t * t;
        dottangrad += dp[0] * f[0] + dp[1] * f[1] + dp[2] * f[2];
        gradlen += f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
      } else if (r === 0) {
        // initial replica: the difference to the next replica, projected
        const dn = [spNext![0] - si[0], spNext![1] - si[1], spNext![2] - si[2]];
        const dd = dn[0] * si[0] + dn[1] * si[1] + dn[2] * si[2];
        for (let k = 0; k < 3; k++) dn[k] -= dd * si[k];
        const t = geodesicDistance(si, spNext!);
        nlen += t * t;
        dottangrad += dn[0] * f[0] + dn[1] * f[1] + dn[2] * f[2];
        gradlen += f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
      } else {
        // intermediate replica. Native does not project the previous-replica difference here (its delpdots stays 0
        // in this branch), so dp is used unprojected; dn is projected
        const dp = [si[0] - spPrev![0], si[1] - spPrev![1], si[2] - spPrev![2]];
        const tp = geodesicDistance(si, spPrev!);
        plen += tp * tp;
        const dn = [spNext![0] - si[0], spNext![1] - si[1], spNext![2] - si[2]];
        const dd = dn[0] * si[0] + dn[1] * si[1] + dn[2] * si[2];
        for (let k = 0; k < 3; k++) dn[k] -= dd * si[k];
        const vmax = Math.max(Math.abs(vnext - veng), Math.abs(vprev - veng));
        const vmin = Math.min(Math.abs(vnext - veng), Math.abs(vprev - veng));
        const tan = [0, 0, 0];
        if (vnext > veng && veng > vprev) {
          for (let k = 0; k < 3; k++) tan[k] = dn[k];
        } else if (vnext < veng && veng < vprev) {
          for (let k = 0; k < 3; k++) tan[k] = dp[k];
        } else if (vnext > vprev) {
          for (let k = 0; k < 3; k++) tan[k] = vmax * dn[k] + vmin * dp[k];
        } else if (vnext < vprev) {
          for (let k = 0; k < 3; k++) tan[k] = vmin * dn[k] + vmax * dp[k];
        } else {
          for (let k = 0; k < 3; k++) tan[k] = dn[k] + dp[k];
        }
        const sdottan = si[0] * tan[0] + si[1] * tan[1] + si[2] * tan[2];
        for (let k = 0; k < 3; k++) tan[k] -= sdottan * si[k];
        const tn = geodesicDistance(si, spNext!);
        nlen += tn * tn;
        for (let k = 0; k < 3; k++) {
          tangent[3 * i + k] = tan[k];
          tlen += tan[k] * tan[k];
        }
        gradlen += f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        dotpath += dp[0] * dn[0] + dp[1] * dn[1] + dp[2] * dn[2];
        dottangrad += tan[0] * f[0] + tan[1] * f[1] + tan[2] * f[2];
      }
    }
    nlen = Math.sqrt(nlen);
    plen = Math.sqrt(plen);
    tlen = Math.sqrt(tlen);
    gradlen = Math.sqrt(gradlen);

    // the tangent is projected on the spin's tangent plane once more and normalized (intermediate replicas use it)
    for (const i of group) {
      const tandots = tangent[3 * i] * sp[4 * i] + tangent[3 * i + 1] * sp[4 * i + 1] + tangent[3 * i + 2] * sp[4 * i + 2];
      for (let k = 0; k < 3; k++) {
        tangent[3 * i + k] -= tandots * sp[4 * i + k];
        if (tlen > 0) tangent[3 * i + k] *= 1 / tlen;
      }
    }

    // the lengths and gradients of this replica, as native's print_status reads them
    if (r > 0 && r < nrep - 1) dottangrad = dottangrad / (tlen * gradlen);
    if (r === 0) dottangrad = dottangrad / (nlen * gradlen);
    if (r === nrep - 1) dottangrad = dottangrad / (plen * gradlen);
    this.veng = veng;
    this.plen = plen;
    this.nlen = nlen;
    this.gradlen = gradlen;
    this.dottangrad = dottangrad;
    // the end replicas get no nudging force (native returns here)
    if (r === 0 || r === nrep - 1) { this.dotpath = dotpath; return; }
    this.dotpath = dotpath / (plen * nlen);

    let dot = 0;
    for (const i of group) dot += fm[3 * i] * tangent[3 * i] + fm[3 * i + 1] * tangent[3 * i + 1] + fm[3 * i + 2] * tangent[3 * i + 2];
    // the climbing replica's tangential force is inverted; the others get the spring along the path
    const prefactor = r === this.rclimber ? -2 * dot : -dot + this.kspring * (nlen - plen);
    // the nudged force, projected on the spin's tangent plane; fm receives the difference to the field
    const delta = new Float64Array(3 * s.n);
    for (const i of group) {
      const v = [0, 1, 2].map((k) => fm[3 * i + k] + prefactor * tangent[3 * i + k]);
      const fdots = v[0] * sp[4 * i] + v[1] * sp[4 * i + 1] + v[2] * sp[4 * i + 2];
      for (let k = 0; k < 3; k++) delta[3 * i + k] = v[k] - fdots * sp[4 * i + k] - fm[3 * i + k];
    }
    spinContribute(s, this, delta);
  }

  /** Withdraws the nudging from fm (the neb/spin command calls it when it ends; fm is then the plain field again). */
  clearNudging(): void {
    spinContribute(this.sys.state, this, new Float64Array(3 * this.sys.state.n));
  }
}
