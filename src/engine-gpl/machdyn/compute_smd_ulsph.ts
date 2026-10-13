/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/compute_smd_ulsph_num_neighs.cpp, compute_smd_ulsph_strain.cpp, compute_smd_ulsph_strain_rate.cpp,
 * compute_smd_ulsph_stress.cpp, compute_smd_ulsph_effm.cpp and compute_smd_vol.cpp (ComputeSMDULSPHNumNeighs, ...,
 * ComputeSMDVol: compute smd/ulsph/num/neighs, smd/ulsph/strain, smd/ulsph/strain/rate, smd/ulsph/stress, smd/ulsph/effm,
 * smd/vol) at commit b332f28. MACHDYN author (from the sources): Georg C. Ganzenmueller, Fraunhofer EMI. LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The per-atom values are zero outside the group, as the native computes (which check the group mask). smd/ulsph/strain
 * is the accumulated strain increment of the pair style (the first six entries of the deformation prop); smd/ulsph/stress
 * gives the xx yy zz xy xz yz components of the Cauchy stress and its von Mises value (column 7).
 * smd/vol is the sum of the volume fractions of the group (docs.lammps.org/compute_smd_vol.html).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { SMD_DEFGRAD } from '../../engine/smd';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import type { Pair } from '../../engine/force/types';
import { PairSMDUlsph } from './pair_smd_ulsph';

const ulsphPair = (sys: System, style: string): PairSMDUlsph => {
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
  const found = visit(sys.ff.pair);
  if (!found) throw new StyleError(`compute ${style} failed to access the ulsph arrays: pair smd/ulsph is missing`);
  return found;
};

/** A per-atom array with a fixed column count, zero outside the group; fill writes the group atoms. */
abstract class UlsphPeratom extends Compute {
  constructor(sys: System, id: string, group: string, args: string[], readonly style: string, cols: number) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`Illegal compute ${style} command`);
    this.peratomFlag = true;
    this.sizePeratomCols = cols;
  }

  protected abstract fill(out: Float64Array, i: number): void;

  protected computePeratom(): void {
    const s = this.sys.state;
    const cols = this.sizePeratomCols;
    const out = new Float64Array(cols * s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.fill(out, i);
    this.arrayAtom = out;
  }
}

/** compute ID group-ID smd/ulsph/num/neighs: the number of neighbors within the kernel radius (native numNeighs). */
export class ComputeSMDULSPHNumNeighs extends Compute {
  readonly style = 'smd/ulsph/num/neighs';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute smd/ulsph/num/neighs command');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = ulsphPair(this.sys, 'smd/ulsph/num/neighs');
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) out[i] = pair.numNeighs[i];
    this.vectorAtom = out;
  }
}

/** compute ID group-ID smd/ulsph/strain: the accumulated strain increments xx yy zz xy xz yz. */
export class ComputeSMDULSPHStrain extends UlsphPeratom {
  constructor(sys: System, id: string, group: string, args: string[]) { super(sys, id, group, args, 'smd/ulsph/strain', 6); }
  protected fill(out: Float64Array, i: number): void {
    const F = this.sys.state.custom.get(SMD_DEFGRAD)!.data;
    for (let c = 0; c < 6; c++) out[6 * i + c] = F[9 * i + c];
  }
}

/** compute ID group-ID smd/ulsph/strain/rate: D = (L + L^T) / 2 of each atom (xx yy zz xy xz yz). */
export class ComputeSMDULSPHStrainRate extends UlsphPeratom {
  constructor(sys: System, id: string, group: string, args: string[]) { super(sys, id, group, args, 'smd/ulsph/strain/rate', 6); }
  protected fill(out: Float64Array, i: number): void {
    const L = ulsphPair(this.sys, 'smd/ulsph/strain/rate').L;
    const d = (a: number, b: number) => 0.5 * (L[9 * i + 3 * a + b] + L[9 * i + 3 * b + a]);
    out.set([d(0, 0), d(1, 1), d(2, 2), d(0, 1), d(0, 2), d(1, 2)], 6 * i);
  }
}

/** compute ID group-ID smd/ulsph/stress: the Cauchy stress xx yy zz xy xz yz and its von Mises value (column 7). */
export class ComputeSMDULSPHStress extends UlsphPeratom {
  constructor(sys: System, id: string, group: string, args: string[]) { super(sys, id, group, args, 'smd/ulsph/stress', 7); }
  protected fill(out: Float64Array, i: number): void {
    const T = ulsphPair(this.sys, 'smd/ulsph/stress').stress;
    const t = (a: number, b: number) => T[9 * i + 3 * a + b];
    const diag = [t(0, 0), t(1, 1), t(2, 2)];
    const mean = (diag[0] + diag[1] + diag[2]) / 3.0;
    let devSq = 0;
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        const dab = t(a, b) - (a === b ? mean : 0);
        devSq += dab * dab;
      }
    }
    out.set([diag[0], diag[1], diag[2], t(0, 1), t(0, 2), t(1, 2), Math.sqrt(3.0 / 2.0) * Math.sqrt(devSq)], 7 * i);
  }
}

/** compute ID group-ID smd/ulsph/effm: the effective shear modulus of each atom (native effm, zero for fluids). */
export const makeSmdUlsphEffm = (sys: System, id: string, group: string, args: string[]): Compute => {
  if (args.length) throw new StyleError('Illegal compute smd/ulsph/effm command');
  return new UlsphScalarPeratom(sys, id, group, args, 'smd/ulsph/effm', (i, pair) => pair.effm[i]);
};

class UlsphScalarPeratom extends Compute {
  constructor(sys: System, id: string, group: string, args: string[], readonly style: string, private readonly value: (i: number, pair: PairSMDUlsph) => number) {
    super(sys, id, group, args);
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = ulsphPair(this.sys, this.style);
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) out[i] = this.value(i, pair);
    this.vectorAtom = out;
  }
}

/** compute ID group-ID smd/vol: the total volume (sum of vfrac over the group), and per atom vfrac. */
export class ComputeSMDVol extends Compute {
  readonly style = 'smd/vol';
  scalarFlag = true;
  extscalar = 1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute smd/vol command');
    if (!sys.state.vfrac) throw new StyleError('compute smd/vol command requires atom_style with volume (e.g. smd)');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computeScalar(): number {
    const s = this.sys.state;
    let total = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) total += s.vfrac![i];
    return total;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) out[i] = s.vfrac![i];
    this.vectorAtom = out;
  }
}
