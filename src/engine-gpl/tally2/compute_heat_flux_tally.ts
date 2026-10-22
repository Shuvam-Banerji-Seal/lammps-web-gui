/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/TALLY/compute_heat_flux_tally.cpp and src/TALLY/compute_heat_flux_tally.h at commit b332f28 (the
 * compute heat/flux/tally of the TALLY package; docs.lammps.org/compute_tally.html). Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Pair-tally heat flux (native: the pair_tally_callback of the pair style, reverse communication of the per-atom
 * energy and stress, and compute_vector()). Only the pairs of the pair style are tallied: the pairs with one atom in
 * group1 and the other in group2 (either order), through the pair's single() (see walkTallyPairs in
 * src/engine/compute/tally.ts). Native "heat flux vector = jc[3] + jv[3]": jc is the convective part sum_i
 * (ke_i + pe_i) v_i and jv the virial part sum_i (stress_i . v_i), summed over the atoms of group1 only; the vector
 * is six values (total flux x, y, z, then the convective part x, y, z). No volume normalisation (native: "normalization
 * by volume is not included"); the energy and stress are in the energy units of the style, as in native.
 *
 * Native's tally callback (fpair *= 0.5 and the six products dx*dx*fpair ... dy*dz*fpair, the half pair energy
 * epairhalf = 0.5*(evdwl + ecoul)) adds the same values to both atoms of a counted pair, whichever group they are
 * in; ghost contributions go to their owner (native reverse_comm with newton on, which the engine does not vary).
 *
 * Native errors and warnings: the missing-argument text is "Illegal compute heat/flux/tally command: missing
 * argument(s)" and the group text is "Could not find compute heat/flux/tally second group ID <ID>". Native ignores
 * extra arguments; the engine refuses them by name instead (no silent option). Native warns when the pair style is
 * not single-enabled or is many-body, and when bonded or kspace styles are defined ("only called from pair style");
 * the engine refuses the first case through walkTallyPairs and reports the second through the log.
 */
import { Compute } from '../../engine/compute/compute';
import { walkTallyPairs, warnUntallied } from '../../engine/compute/tally';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

export class ComputeHeatFluxTally extends Compute {
  readonly style = 'heat/flux/tally';
  private readonly group2Bit: number;
  private cacheEpoch = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length === 0) throw new StyleError('Illegal compute heat/flux/tally command: missing argument(s)');
    if (sys.groups.find(args[0]) < 0) throw new StyleError(`Could not find compute heat/flux/tally second group ID ${args[0]}`);
    // compute_heat_flux_tally.cpp constructor: only arg[3] (group2-ID) is read; words after it are ignored
    this.group2Bit = sys.groups.bit(args[0]);
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extvector = 1;
    this.vector = new Float64Array(6);
  }

  init(): void {
    const sys = this.sys;
    if (sys.ff.pair && sys.ff.pair.manybody) {
      sys.log('WARNING: Compute heat/flux/tally used with incompatible pair style');
    }
    if (sys.ff.bond || sys.ff.angle || sys.ff.dihedral || sys.ff.improper || sys.ff.kspace) {
      sys.log('WARNING: Compute heat/flux/tally only called from pair style');
    }
  }

  /** Per-atom energy (eatom) and stress (6 columns, xx yy zz xy xz yz) of the counted pairs, ghosts folded into owners. */
  private evaluate(): void {
    if (this.cacheEpoch === this.sys.epoch) return;
    const s = this.sys.state;
    const n = s.n;
    const eatom = new Float64Array(n);
    const stress = new Float64Array(6 * n);
    walkTallyPairs(this.sys, this.id, this.groupBit, this.group2Bit, 'off', (i, j, evdwl, ecoul, fforce, dx, dy, dz) => {
      const oj = this.sys.nb.owner[j];
      const epairhalf = 0.5 * (evdwl + ecoul);
      const f = fforce * 0.5;
      const v0 = dx * dx * f, v1 = dy * dy * f, v2 = dz * dz * f;
      const v3 = dx * dy * f, v4 = dx * dz * f, v5 = dy * dz * f;
      eatom[i] += epairhalf;
      stress[6 * i] += v0; stress[6 * i + 1] += v1; stress[6 * i + 2] += v2;
      stress[6 * i + 3] += v3; stress[6 * i + 4] += v4; stress[6 * i + 5] += v5;
      eatom[oj] += epairhalf;
      stress[6 * oj] += v0; stress[6 * oj + 1] += v1; stress[6 * oj + 2] += v2;
      stress[6 * oj + 3] += v3; stress[6 * oj + 4] += v4; stress[6 * oj + 5] += v5;
    });
    // native compute_vector(): J = sum_i (0.5 m v^2 + eatom_i) v_i + stress_i . v_i over the atoms of group1
    const pfactor = 0.5 * s.units.mvv2e;
    const v = s.v;
    let jc0 = 0, jc1 = 0, jc2 = 0, jv0 = 0, jv1 = 0, jv2 = 0;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      const si0 = stress[6 * i], si1 = stress[6 * i + 1], si2 = stress[6 * i + 2];
      const si3 = stress[6 * i + 3], si4 = stress[6 * i + 4], si5 = stress[6 * i + 5];
      let keI = s.rmass ? pfactor * s.rmass[i] : pfactor * s.massByType[s.type[i]];
      keI *= vx * vx + vy * vy + vz * vz;
      keI += eatom[i];
      jc0 += keI * vx;
      jc1 += keI * vy;
      jc2 += keI * vz;
      jv0 += si0 * vx + si3 * vy + si4 * vz;
      jv1 += si3 * vx + si1 * vy + si5 * vz;
      jv2 += si4 * vx + si5 * vy + si2 * vz;
    }
    this.vector[0] = jc0 + jv0;
    this.vector[1] = jc1 + jv1;
    this.vector[2] = jc2 + jv2;
    this.vector[3] = jc0;
    this.vector[4] = jc1;
    this.vector[5] = jc2;
    this.cacheEpoch = this.sys.epoch;
  }

  protected computeVector(): void {
    warnUntallied(this.sys, 'Stress');
    this.evaluate();
  }
}
