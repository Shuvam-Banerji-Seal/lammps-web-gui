/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/compute_smd_tlsph_stress.cpp, compute_smd_tlsph_strain.cpp and compute_smd_tlsph_num_neighs.cpp
 * (ComputeSMDTLSPHStress, ComputeSMDTLSPHstrain, ComputeSMDTLSPHNumNeighs) at commit b332f28. MACHDYN author (from the
 * sources): Georg C. Ganzenmueller, Fraunhofer EMI. LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Per-atom outputs of the tlsph style (docs.lammps.org/compute_smd_tlsph_stress.html etc.):
 *   smd/tlsph/stress: 7 columns, the Cauchy stress xx yy zz xy xz yz and the von Mises stress
 *   smd/tlsph/strain: 6 columns, the Green-Lagrange strain E = 1/2 (F^T F - I) with F = F0 Fincr (xx yy zz xy xz yz)
 *   smd/tlsph/num/neighs: the number of reference-configuration neighbors of each atom
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { SMD_CONTACT, SMD_DAMAGE, SMD_DEFGRAD, SMD_EPS, SMD_EPSRATE, SMD_ESPH } from '../../engine/smd';
import { PairSMDTlsph } from './pair_smd_tlsph';
import { deviator3, identity3, mul3, norm3, scale3, sub3, transpose3, type Mat3 } from './smd_math';

const tlsphPair = (sys: System, style: string): PairSMDTlsph => {
  const p = sys.ff.pair;
  if (!(p instanceof PairSMDTlsph)) throw new StyleError(`compute ${style} could not access the stress tensors. Are the matching pair styles present?`);
  return p;
};

/** Symmetric entries of a 3x3 matrix in the order xx yy zz xy xz yz. */
const sym6 = (m: Mat3): number[] => [m[0], m[4], m[8], m[1], m[2], m[5]];

/** compute ID group-ID smd/tlsph/stress (docs.lammps.org/compute_smd_tlsph_stress.html). */
export class ComputeSMDTLSPHStress extends Compute {
  readonly style = 'smd/tlsph/stress';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute smd/tlsph/stress command');
    this.peratomFlag = true;
    this.sizePeratomCols = 7;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = tlsphPair(this.sys, 'smd/tlsph/stress');
    const out = new Float64Array(7 * s.n);
    const T = pair.work.CauchyStress;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit) || !T[i]) continue;
      const m = T[i];
      const vm = Math.sqrt(3 / 2) * norm3(deviator3(m));
      out.set([m[0], m[4], m[8], m[1], m[2], m[5], vm], 7 * i);
    }
    this.arrayAtom = out;
  }
}

/** compute ID group-ID smd/tlsph/strain (docs.lammps.org/compute_smd_tlsph_strain.html). */
export class ComputeSMDTLSPHStrain extends Compute {
  readonly style = 'smd/tlsph/strain';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute smd/tlsph/strain command');
    this.peratomFlag = true;
    this.sizePeratomCols = 6;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = tlsphPair(this.sys, 'smd/tlsph/strain');
    const out = new Float64Array(6 * s.n);
    const F0all = s.custom.get(SMD_DEFGRAD)!.data;
    const eye = identity3();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const F0 = Array.from(F0all.subarray(9 * i, 9 * i + 9));
      const Ftotal = mul3(F0, pair.work.Fincr[i]);
      const E = scale3(sub3(mul3(transpose3(Ftotal), Ftotal), eye), 0.5);
      out.set(sym6(E), 6 * i);
    }
    this.arrayAtom = out;
  }
}

/** compute ID group-ID smd/tlsph/num/neighs (docs.lammps.org/compute_smd_tlsph_num_neighs.html). */
export class ComputeSMDTLSPHNumNeighs extends Compute {
  readonly style = 'smd/tlsph/num/neighs';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute smd/tlsph/num/neighs command');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = tlsphPair(this.sys, 'smd/tlsph/num/neighs');
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) out[i] = pair.work.numNeighs[i];
    this.vectorAtom = out;
  }
}

/** Per-atom vector of an owned-atom quantity, zero outside the group (the native computes that check the mask). */
class PeratomVector extends Compute {
  readonly style: string;

  constructor(sys: System, id: string, group: string, args: string[], what: string, private readonly fill: (i: number, pair: PairSMDTlsph | null) => number) {
    super(sys, id, group, args);
    this.style = what;
    if (args.length) throw new StyleError(`Illegal compute ${what} command`);
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const p = this.sys.ff.pair instanceof PairSMDTlsph ? this.sys.ff.pair : null;
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) out[i] = this.fill(i, p);
    this.vectorAtom = out;
  }
}

/** compute ID group-ID smd/tlsph/dt (docs.lammps.org/compute_smd_tlsph_dt.html): the stable time step of each atom. */
export const makeSmdTlsphDt = (sys: System, id: string, group: string, args: string[]): Compute =>
  new PeratomVector(sys, id, group, args, 'smd/tlsph/dt', (i, p) => (p ? p.work.particleDt[i] ?? 0 : 0));
/** compute ID group-ID smd/plastic/strain: the effective plastic strain of each atom. */
export const makeSmdPlasticStrain = (sys: System, id: string, group: string, args: string[]): Compute =>
  new PeratomVector(sys, id, group, args, 'smd/plastic/strain', (i, p) => (p ? sys.state.custom.get(SMD_EPS)!.data[i] : 0));
/** compute ID group-ID smd/plastic/strain/rate: the rate of the effective plastic strain of each atom. */
export const makeSmdPlasticStrainRate = (sys: System, id: string, group: string, args: string[]): Compute =>
  new PeratomVector(sys, id, group, args, 'smd/plastic/strain/rate', (i, p) => (p ? sys.state.custom.get(SMD_EPSRATE)!.data[i] : 0));
/** compute ID group-ID smd/damage: the damage of each atom. */
export const makeSmdDamage = (sys: System, id: string, group: string, args: string[]): Compute =>
  new PeratomVector(sys, id, group, args, 'smd/damage', (i, p) => (p ? sys.state.custom.get(SMD_DAMAGE)!.data[i] : 0));

/** compute ID group-ID smd/tlsph/strain/rate: the rate of deformation D (xx yy zz xy xz yz) of each atom, for all atoms. */
export class ComputeSMDTLSPHStrainRate extends Compute {
  readonly style = 'smd/tlsph/strain/rate';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute smd/tlsph/strain/rate command');
    this.peratomFlag = true;
    this.sizePeratomCols = 6;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = tlsphPair(this.sys, 'smd/tlsph/strain/rate');
    const out = new Float64Array(6 * s.n);
    for (let i = 0; i < s.n; i++) {
      const D = pair.work.D[i];
      if (D) out.set(sym6(D), 6 * i);
    }
    this.arrayAtom = out;
  }
}

/** Eigen's conversion of a rotation matrix to a quaternion (w, x, y, z), the branch choice as Eigen's Quaternion(matrix). */
export const quatFromRotation = (m: Mat3): [number, number, number, number] => {
  const tr = m[0] + m[4] + m[8];
  if (tr > 0) {
    let t = Math.sqrt(tr + 1.0);
    const w = 0.5 * t;
    t = 0.5 / t;
    return [(m[7] - m[5]) * t, (m[2] - m[6]) * t, (m[3] - m[1]) * t, w];
  }
  let i = 0;
  if (m[4] > m[0]) i = 1;
  if (m[8] > m[3 * i + i]) i = 2;
  const j = (i + 1) % 3, k = (j + 1) % 3;
  let t = Math.sqrt(m[4 * i] - m[4 * j] - m[4 * k] + 1.0);
  const q = [0, 0, 0, 0];
  q[i] = 0.5 * t;
  t = 0.5 / t;
  q[3] = (m[3 * k + j] - m[3 * j + k]) * t;
  q[j] = (m[3 * j + i] + m[3 * i + j]) * t;
  q[k] = (m[3 * k + i] + m[3 * i + k]) * t;
  return [q[0], q[1], q[2], q[3]];
};

/**
 * compute ID group-ID smd/tlsph/shape: per atom, the contact radius scaled by 1 + E_aa (the diagonal of the Green-Lagrange
 * strain of Fincr), then the quaternion (w, x, y, z) of the rotation of the atom (docs.lammps.org/compute_smd_tlsph_shape.html).
 */
export class ComputeSMDTLSPHShape extends Compute {
  readonly style = 'smd/tlsph/shape';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute smd/tlsph/shape command');
    this.peratomFlag = true;
    this.sizePeratomCols = 7;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = tlsphPair(this.sys, 'smd/tlsph/shape');
    const out = new Float64Array(7 * s.n);
    const contact = s.custom.get(SMD_CONTACT)!.data;
    const eye = identity3();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const F = pair.work.Fincr[i];
      if (!F) continue;
      const E = scale3(sub3(mul3(transpose3(F), F), eye), 0.5);
      const q = quatFromRotation(pair.work.R[i]);
      out.set([contact[i] * (1.0 + E[0]), contact[i] * (1.0 + E[4]), contact[i] * (1.0 + E[8]), q[3], q[0], q[1], q[2]], 7 * i);
    }
    this.arrayAtom = out;
  }
}

/**
 * Per-atom values of the SMD atom props (native compute_smd_contact_radius.cpp, compute_smd_internal_energy.cpp,
 * compute_smd_rho.cpp, compute_smd_vol.cpp): zero outside the group, as the native computes check the mask.
 */
class SmdAtomProp extends Compute {
  constructor(sys: System, id: string, group: string, args: string[], readonly style: string, private readonly value: (i: number) => number) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`Illegal compute ${style} command`);
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) out[i] = this.value(i);
    this.vectorAtom = out;
  }
}

/** compute ID group-ID smd/contact/radius (docs.lammps.org/compute_smd_contact_radius.html): the contact radius of each atom. */
export const makeSmdContactRadius = (sys: System, id: string, group: string, args: string[]): Compute =>
  new SmdAtomProp(sys, id, group, args, 'smd/contact/radius', (i) => sys.state.custom.get(SMD_CONTACT)!.data[i]);
/** compute ID group-ID smd/internal/energy: the internal energy per unit mass (esph) of each atom. */
export const makeSmdInternalEnergy = (sys: System, id: string, group: string, args: string[]): Compute =>
  new SmdAtomProp(sys, id, group, args, 'smd/internal/energy', (i) => sys.state.custom.get(SMD_ESPH)!.data[i]);
/** compute ID group-ID smd/rho: the mass density rmass / vfrac of each atom. */
export const makeSmdRho = (sys: System, id: string, group: string, args: string[]): Compute =>
  new SmdAtomProp(sys, id, group, args, 'smd/rho', (i) => sys.state.rmass![i] / sys.state.vfrac![i]);
