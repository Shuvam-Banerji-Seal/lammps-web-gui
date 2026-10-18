/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/TALLY/compute_heat_flux_virial_tally.cpp and src/TALLY/compute_heat_flux_virial_tally.h at commit
 * b332f28 (the compute heat/flux/virial/tally of the TALLY package; docs.lammps.org/compute_tally.html). Copyright
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Virial-only pair-tally heat flux (native: pair_tally_callback, compute_peratom() and compute_scalar()). The per-atom
 * value fatom_i = sum over the counted pairs of s * dx * fpair, with s = +0.5 when the first atom of the pair is in
 * group1 and -0.5 otherwise, the same value added to both atoms of the pair. The scalar is sum over all local atoms
 * of fatom_i . v_i (no group restriction). Because the groups are disjoint (native init() errors otherwise), the value
 * of a pair does not depend on which of its atoms the pair list calls i.
 *
 * Native errors: the atoms-in-both-groups check runs in init() (every run) with the text "Atom <ID> belonging to both
 * groups is not allowed with compute heat/flux/virial/tally". The missing-argument and group texts are as for
 * compute heat/flux/tally (src/TALLY/compute_heat_flux_virial_tally.cpp). Extra arguments are refused by name (native
 * ignores them).
 */
import { Compute } from '../../engine/compute/compute';
import { walkTallyPairs } from '../../engine/compute/tally';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

export class ComputeHeatFluxVirialTally extends Compute {
  readonly style = 'heat/flux/virial/tally';
  private readonly group2Bit: number;
  private cacheEpoch = -1;
  private total = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length === 0) throw new StyleError('Illegal compute heat/flux/virial/tally command: missing argument(s)');
    if (sys.groups.find(args[0]) < 0) throw new StyleError(`Could not find compute heat/flux/virial/tally second group ID ${args[0]}`);
    if (args.length !== 1) throw new StyleError(`compute ${id} heat/flux/virial/tally takes one group2-ID (no keywords)`);
    this.group2Bit = sys.groups.bit(args[0]);
    this.scalarFlag = true;
    this.extscalar = 1;
    this.peratomFlag = true;
    this.sizePeratomCols = 3;
  }

  init(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && (s.mask[i] & this.group2Bit)) {
        throw new StyleError(`Atom ${s.id[i]} belonging to both groups is not allowed with compute heat/flux/virial/tally`);
      }
    }
    const sys = this.sys;
    if (sys.ff.pair && sys.ff.pair.manybody) {
      sys.log('WARNING: Compute heat/flux/virial/tally used with incompatible pair style');
    }
    if (sys.ff.bond || sys.ff.angle || sys.ff.dihedral || sys.ff.improper || sys.ff.kspace) {
      sys.log('WARNING: Compute heat/flux/virial/tally only called from pair style');
    }
  }

  /** Per-atom fatom (3 columns) of the counted pairs, ghosts folded into owners; the scalar is sum fatom . v. */
  private evaluate(): void {
    if (this.cacheEpoch === this.sys.epoch) return;
    const s = this.sys.state;
    const n = s.n;
    const fatom = new Float64Array(3 * n);
    walkTallyPairs(this.sys, this.id, this.groupBit, this.group2Bit, 'off', (i, j, _evdwl, _ecoul, fforce, dx, dy, dz) => {
      const oj = this.sys.nb.owner[j];
      // signs set to calculate heat flux from group2 into group1 (native: ingroup1 is the mask of the first atom)
      const ingroup1 = (s.mask[i] & this.groupBit) !== 0;
      const sg = ingroup1 ? 0.5 : -0.5;
      const fx = sg * dx * fforce, fy = sg * dy * fforce, fz = sg * dz * fforce;
      fatom[3 * i] += fx; fatom[3 * i + 1] += fy; fatom[3 * i + 2] += fz;
      fatom[3 * oj] += fx; fatom[3 * oj + 1] += fy; fatom[3 * oj + 2] += fz;
    });
    const v = s.v;
    let hflux = 0;
    for (let i = 0; i < n; i++) {
      hflux += fatom[3 * i] * v[3 * i] + fatom[3 * i + 1] * v[3 * i + 1] + fatom[3 * i + 2] * v[3 * i + 2];
    }
    this.arrayAtom = fatom;
    this.total = hflux;
    this.cacheEpoch = this.sys.epoch;
  }

  protected computeScalar(): number {
    this.evaluate();
    return this.total;
  }

  protected computePeratom(): void {
    this.evaluate();
  }
}
