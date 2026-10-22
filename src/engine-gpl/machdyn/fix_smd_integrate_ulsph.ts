/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/fix_smd_integrate_ulsph.cpp and fix_smd_integrate_ulsph.h (FixSMDIntegrateUlsph: fix
 * smd/integrate_ulsph) at commit b332f28. MACHDYN author (from the sources): Georg C. Ganzenmueller, Fraunhofer EMI.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Velocity Verlet for updated Lagrangian SPH (docs.lammps.org/fix_smd_integrate_ulsph.html): the velocities and
 * positions advance, the estimated velocity vest is set for the pair style, the internal energy advances with the
 * energy rate, the volume of each atom changes with the trace of the velocity gradient, and the smoothing length can
 * be adapted to the neighbor count (keyword adjust_radius factor min max, clamped between 1.25 and 4 contact radii).
 * The keyword xsph is refused: native LAMMPS stops with "XSPH is currently not available".
 */
import { StyleError, type Pair } from '../../engine/force/types';
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { SMD_CONTACT, SMD_DESPH, SMD_ESPH, SMD_VEST } from '../../engine/smd';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import { PairSMDUlsph } from './pair_smd_ulsph';

/** The ULSPH pair of the force field (directly or as a hybrid sub-style). */
const findUlsph = (sys: System): PairSMDUlsph | null => {
  const visit = (p: Pair | null | undefined): PairSMDUlsph | null => {
    if (p instanceof PairSMDUlsph) return p;
    if (p instanceof PairHybrid) {
      for (const sub of p.subs) {
        const found = visit(sub.style);
        if (found) return found;
      }
    }
    return null;
  };
  return visit(sys.ff.pair);
};

export class FixSMDIntegrateUlsph extends Fix {
  readonly style = 'smd/integrate_ulsph';
  timeIntegrate = true;
  private adjustRadius: { factor: number; minNn: number; maxNn: number } | null = null;
  private vlimit = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_smd_integrate_ulsph.cpp constructor: the atom attributes first
    if (!sys.state.custom.has(SMD_ESPH) || !sys.state.vfrac) {
      throw new StyleError('fix smd/integrate_ulsph command requires atom_style with both energy and volume');
    }
    for (let k = 0; k < args.length; k++) {
      const kw = args[k];
      if (kw === 'xsph') throw new StyleError('fix smd/integrate_ulsph keyword xsph is not available (native LAMMPS: "XSPH is currently not available")');
      if (kw === 'adjust_radius') {
        if (k + 3 >= args.length) throw new StyleError('expected three numbers following adjust_radius: factor, min, max');
        const factor = Number(args[k + 1]), minNn = Number(args[k + 2]), maxNn = Number(args[k + 3]);
        if (![factor, minNn, maxNn].every(Number.isFinite)) throw new StyleError('expected three numbers following adjust_radius: factor, min, max');
        this.adjustRadius = { factor, minNn: Math.trunc(minNn), maxNn: Math.trunc(maxNn) };
        k += 3;
      } else if (kw === 'limit_velocity') {
        if (k + 1 >= args.length) throw new StyleError('expected number following limit_velocity');
        this.vlimit = Number(args[k + 1]);
        if (!Number.isFinite(this.vlimit)) throw new StyleError(`illegal limit_velocity value '${args[k + 1]}'`);
        k += 1;
      } else {
        throw new StyleError(`Unknown fix smd/integrate_ulsph keyword: ${kw}`);
      }
    }
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const dtv = s.dt;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const vest = s.custom.get(SMD_VEST)!.data;
    const vlimitsq = this.vlimit * this.vlimit;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const dtfm = dtf / s.rmass![i];
      for (let d = 0; d < 3; d++) s.v[3 * i + d] += dtfm * s.f[3 * i + d];
      if (this.vlimit > 0.0) {
        const vsq = s.v[3 * i] * s.v[3 * i] + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2];
        if (vsq > vlimitsq) {
          const scale = Math.sqrt(vlimitsq / vsq);
          for (let d = 0; d < 3; d++) s.v[3 * i + d] *= scale;
        }
      }
      for (let d = 0; d < 3; d++) vest[3 * i + d] = s.v[3 * i + d] + dtfm * s.f[3 * i + d];
      for (let d = 0; d < 3; d++) s.x[3 * i + d] += dtv * s.v[3 * i + d];
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const esph = s.custom.get(SMD_ESPH)!.data;
    const desph = s.custom.get(SMD_DESPH)!.data;
    const contact = s.custom.get(SMD_CONTACT)!.data;
    const vlimitsq = this.vlimit * this.vlimit;
    const pair = findUlsph(this.sys);
    if (!pair) throw new StyleError('fix smd/integrate_ulsph failed to access the velocity gradient: pair smd/ulsph is missing');
    const L = pair.L, nn = pair.numNeighs;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const dtfm = dtf / s.rmass![i];
      for (let d = 0; d < 3; d++) s.v[3 * i + d] += dtfm * s.f[3 * i + d];
      if (this.vlimit > 0.0) {
        const vsq = s.v[3 * i] * s.v[3 * i] + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2];
        if (vsq > vlimitsq) {
          const scale = Math.sqrt(vlimitsq / vsq);
          for (let d = 0; d < 3; d++) s.v[3 * i + d] *= scale;
        }
      }
      esph[i] += dtf * desph[i];
      if (this.adjustRadius) {
        const { factor, minNn, maxNn } = this.adjustRadius;
        if (nn[i] < minNn) s.radius![i] *= factor;
        else if (nn[i] > maxNn) s.radius![i] /= factor;
        s.radius![i] = Math.max(s.radius![i], 1.25 * contact[i]);
        s.radius![i] = Math.min(s.radius![i], 4.0 * contact[i]);
      }
      // D = (L + L^T) / 2; the volume changes with the trace of D (Jacobian of the deformation, native vol_increment)
      const trace = 0.5 * (L[9 * i] + L[9 * i]) + 0.5 * (L[9 * i + 4] + L[9 * i + 4]) + 0.5 * (L[9 * i + 8] + L[9 * i + 8]);
      s.vfrac![i] += s.vfrac![i] * s.dt * trace;
    }
  }
}

