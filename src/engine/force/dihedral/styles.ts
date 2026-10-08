import { SimpleBonded, atomIndex, dihedralGeometry, applyDihedral } from '../bonded_util';
import { parseNum, parseInt_, fmtCoeff } from '../util';
import { StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';

/*
 * Dihedral styles (wave 1). Every class below implements exactly the formula
 * and coefficient list of its docs.lammps.org page (quoted in its header);
 * angle-like styles share dihedralGeometry/applyDihedral from bonded_util.ts
 * (docs.lammps.org/dihedral_style.html: "phi is the torsional angle defined
 * by the quadruplet of atoms", trans = 180 degrees).
 */

const DEG2RAD = Math.PI / 180;

/** Shared compute loop: phi and its analytic gradient per dihedral, energy into edihed. */
const eachDihedral = (bc: BondedCompute, term: (t: number, phi: number) => { e: number; dEdphi: number }): void => {
  const D = bc.s.topo.dihedrals;
  const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
  let e = 0;
  for (let q = 0; q < D.n; q++) {
    const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, D.atoms[4 * q + w], 'dihedral'));
    const phi = dihedralGeometry(bc, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
    const { e: ed, dEdphi } = term(D.type[q], phi);
    e += ed;
    applyDihedral(bc, atoms, dEdphi, grad, rel, ed);
  }
  bc.acc.edihed += e;
};

/*
 * dihedral_style opls — docs.lammps.org/dihedral_opls.html:
 *
 *    E = & \frac{1}{2} K_1 [1 + \cos(\phi)] + \frac{1}{2} K_2 [1 - \cos(2 \phi)] + \\
 *        & \frac{1}{2} K_3 [1 + \cos(3 \phi)] + \frac{1}{2} K_4 [1 - \cos(4 \phi)]
 *
 * "Note that the usual 1/2 factor is not included in the K values."
 *
 * * :math:`K_1` (energy)
 * * :math:`K_2` (energy)
 * * :math:`K_3` (energy)
 * * :math:`K_4` (energy)
 */
export class DihedralOpls extends SimpleBonded {
  readonly name = 'opls';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['K1', 'K2', 'K3', 'K4'];

  compute(bc: BondedCompute): void {
    const K1 = this.params.p('K1'), K2 = this.params.p('K2'), K3 = this.params.p('K3'), K4 = this.params.p('K4');
    eachDihedral(bc, (t, phi) => {
      const c = Math.cos(phi), s = Math.sin(phi);
      const c2 = Math.cos(2 * phi), s2 = Math.sin(2 * phi);
      const c3 = Math.cos(3 * phi), s3 = Math.sin(3 * phi);
      const c4 = Math.cos(4 * phi), s4 = Math.sin(4 * phi);
      return {
        e: 0.5 * (K1[t] * (1 + c) + K2[t] * (1 - c2) + K3[t] * (1 + c3) + K4[t] * (1 - c4)),
        dEdphi: -0.5 * K1[t] * s + K2[t] * s2 - 1.5 * K3[t] * s3 + 2 * K4[t] * s4,
      };
    });
  }
}

/*
 * dihedral_style multi/harmonic — docs.lammps.org/dihedral_multi_harmonic.html:
 *
 *    E = \sum_{n=1,5} A_n  \cos^{n-1}(\phi)
 *
 * * :math:`A_1` (energy)
 * * :math:`A_2` (energy)
 * * :math:`A_3` (energy)
 * * :math:`A_4` (energy)
 * * :math:`A_5` (energy)
 */
export class DihedralMultiHarmonic extends SimpleBonded {
  readonly name = 'multi/harmonic';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['A1', 'A2', 'A3', 'A4', 'A5'];

  compute(bc: BondedCompute): void {
    const A1 = this.params.p('A1'), A2 = this.params.p('A2'), A3 = this.params.p('A3'), A4 = this.params.p('A4'), A5 = this.params.p('A5');
    eachDihedral(bc, (t, phi) => {
      const c = Math.cos(phi), s = Math.sin(phi);
      const c2 = c * c, c3 = c2 * c, c4 = c3 * c;
      return {
        e: A1[t] + A2[t] * c + A3[t] * c2 + A4[t] * c3 + A5[t] * c4,
        dEdphi: -s * (A2[t] + 2 * A3[t] * c + 3 * A4[t] * c2 + 4 * A5[t] * c3),
      };
    });
  }
}

/**
 * Base for styles with a variable number of coefficients per type (fourier,
 * nharmonic): TypeParams holds a fixed list of named values, so these keep
 * their own packed per-type arrays. The type word still goes through
 * typeBounds, so "", "n", "n" and "mn" work (docs.lammps.org/
 * dihedral_coeff.html: "a wild-card asterisk can be used to set the
 * coefficients for multiple dihedral types").
 */
abstract class DihedralVarLen extends SimpleBonded {
  readonly paramNames: readonly string[] = [];
  /** Packed coefficient values per type; null until dihedral_coeff sets them. */
  protected vals: (Float64Array | null)[] = [];

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.vals = new Array(ntypes + 1).fill(null);
  }

  /** Stores one packed value list for every type in the (wildcard) type word. */
  protected setCoeffs(word: string, values: number[]): void {
    const [lo, hi] = typeBounds(word, this.ntypes);
    const packed = Float64Array.from(values);
    for (let t = lo; t <= hi; t++) this.vals[t] = packed;
  }

  override init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.vals[t]) throw new StyleError(`all dihedral coeffs are not set (type ${t})`);
    }
  }

  /** One type's values as dihedral_coeff takes them (the data file re-reads them). */
  protected dataValues(v: Float64Array): number[] { return Array.from(v); }

  override dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      const v = this.vals[t];
      out.push(`${t} ${v ? this.dataValues(v).map(fmtCoeff).join(' ') : ''}`.trimEnd());
    }
    return out;
  }
}

/*
 * dihedral_style fourier — docs.lammps.org/dihedral_fourier.html:
 *
 *    E = \sum_{i=1,m} K_i  [ 1.0 + \cos ( n_i \phi - d_i ) ]
 *
 * * :math:`m` (integer >=1)
 * * :math:`K_1` (energy)
 * * :math:`n_1` (integer >= 0)
 * * :math:`d_1` (degrees)
 * [...]
 * * :math:`K_m` (energy)
 * * :math:`n_m` (integer >= 0)
 * * :math:`d_m` (degrees)
 *
 * d_i is read in degrees and converted to radians (as with quadratic:
 * "specified in degrees, but LAMMPS converts it to radians internally").
 */
export class DihedralFourier extends DihedralVarLen {
  readonly name = 'fourier';
  readonly kind = 'dihedral' as const;

  override coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 2) throw new StyleError('usage: dihedral_coeff N m K1 n1 d1 [K2 n2 d2 ...]');
    const m = parseInt_(args[1], 'm');
    if (m < 1) throw new StyleError('dihedral fourier m must be an integer >= 1');
    const vals = args.slice(2);
    if (vals.length !== 3 * m) {
      throw new StyleError(`dihedral_coeff fourier needs m K n d triplets: m=${m} requires ${3 * m} values, got ${vals.length}`);
    }
    const packed: number[] = [];
    for (let i = 0; i < m; i++) {
      const K = parseNum(vals[3 * i], `K${i + 1}`);
      const n = parseInt_(vals[3 * i + 1], `n${i + 1}`);
      if (n < 0) throw new StyleError(`dihedral fourier n${i + 1} must be an integer >= 0`);
      const d = parseNum(vals[3 * i + 2], `d${i + 1}`) * DEG2RAD;
      packed.push(K, n, d);
    }
    this.setCoeffs(args[0], packed);
  }

  // write_data, measured with native LAMMPS: "1 2 1 1 0 0.6 3 180" — m first, d in degrees
  protected dataValues(v: Float64Array): number[] {
    const out = [v.length / 3];
    for (let i = 0; i < v.length; i += 3) out.push(v[i], v[i + 1], v[i + 2] / DEG2RAD);
    return out;
  }

  compute(bc: BondedCompute): void {
    eachDihedral(bc, (t, phi) => {
      const v = this.vals[t]!;
      let e = 0, dEdphi = 0;
      for (let i = 0; i < v.length; i += 3) {
        const arg = v[i + 1] * phi - v[i + 2];
        e += v[i] * (1 + Math.cos(arg));
        dEdphi -= v[i] * v[i + 1] * Math.sin(arg);
      }
      return { e, dEdphi };
    });
  }
}

/*
 * dihedral_style quadratic — docs.lammps.org/dihedral_quadratic.html:
 *
 *    E = K (\phi - \phi_0)^2
 *
 * * :math:`K` (energy)
 * * :math:`\phi_0` (degrees)
 *
 * ":math:`\phi_0` is specified in degrees, but LAMMPS converts it to
 * radians internally; hence :math:`K` is effectively energy per
 * radian\^2."
 */
export class DihedralQuadratic extends SimpleBonded {
  readonly name = 'quadratic';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['K', 'phi0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('dihedral_coeff quadratic needs K phi0');
    return [parseNum(args[0], 'K'), parseNum(args[1], 'phi0') * DEG2RAD];
  }

  // write_data, measured with native LAMMPS: phi0 back in degrees ("1 2 120")
  dataCoeffs(): string[] {
    const K = this.params.p('K'), phi0 = this.params.p('phi0');
    return Array.from({ length: this.ntypes }, (_, k) => `${k + 1} ${fmtCoeff(K[k + 1])} ${fmtCoeff(phi0[k + 1] / DEG2RAD)}`);
  }

  compute(bc: BondedCompute): void {
    const K = this.params.p('K'), phi0 = this.params.p('phi0');
    eachDihedral(bc, (t, phi) => {
      const d = phi - phi0[t];
      return { e: K[t] * d * d, dEdphi: 2 * K[t] * d };
    });
  }
}

/*
 * dihedral_style nharmonic — docs.lammps.org/dihedral_nharmonic.html:
 *
 *    E = \sum_{i=1,n} A_i  \cos^{i-1}(\phi)
 *
 * * :math:`n` (integer >=1)
 * * :math:`A_1` (energy)
 * * :math:`A_2` (energy)
 * ...
 * * :math:`A_n` (energy)
 */
export class DihedralNHarmonic extends DihedralVarLen {
  readonly name = 'nharmonic';
  readonly kind = 'dihedral' as const;

  override coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 2) throw new StyleError('usage: dihedral_coeff N n A1 [A2 ... An]');
    const n = parseInt_(args[1], 'n');
    if (n < 1) throw new StyleError('dihedral nharmonic n must be an integer >= 1');
    const vals = args.slice(2);
    if (vals.length !== n) {
      throw new StyleError(`dihedral_coeff nharmonic needs n A values: n=${n} requires ${n} values, got ${vals.length}`);
    }
    this.setCoeffs(args[0], vals.map((w, i) => parseNum(w, `A${i + 1}`)));
  }

  // write_data, measured with native LAMMPS: "1 4 1 -0.5 0.8 0.2" — n first
  protected dataValues(v: Float64Array): number[] { return [v.length, ...v]; }

  compute(bc: BondedCompute): void {
    eachDihedral(bc, (t, phi) => {
      const A = this.vals[t]!;
      const c = Math.cos(phi), s = Math.sin(phi);
      // E = sum_k A_k c^(k-1);  dE/dphi = -sin(phi) * sum_k (k-1) A_k c^(k-2)
      let e = A[0], cp = 1, S = 0;
      for (let i = 1; i < A.length; i++) {
        const cprev = cp;
        cp *= c;
        e += A[i] * cp;
        S += i * A[i] * cprev;
      }
      return { e, dEdphi: -s * S };
    });
  }
}

/*
 * dihedral_style cosine/shift/exp — docs.lammps.org/dihedral_cosine_shift_exp.html:
 *
 *    E = -U_{min}\frac{e^{-a U(\theta,\theta_0)}-1}{e^a-1} \quad\mbox{with}\quad U(\theta,\theta_0)=-0.5 \left(1+\cos(\theta-\theta_0) \right)
 *
 * * :math:`U_{min}` (energy)
 * * :math:`\theta` (angle)
 * * :math:`a` (real number)
 *
 * theta0 is an angle coefficient and is read in degrees, converted to
 * radians internally (as documented for quadratic). "The potential is
 * furthermore well behaved in the limit :math:`a \rightarrow 0`, where it
 * has been implemented to linear order in :math:`a` for :math:`a < 0.001`."
 */
export class DihedralCosineShiftExp extends SimpleBonded {
  readonly name = 'cosine/shift/exp';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['Umin', 'theta0', 'a'];

  protected parse(args: string[]): number[] {
    if (args.length !== 3) throw new StyleError('dihedral_coeff cosine/shift/exp needs Umin theta0 a');
    return [parseNum(args[0], 'Umin'), parseNum(args[1], 'theta0') * DEG2RAD, parseNum(args[2], 'a')];
  }

  // write_data, measured with native LAMMPS: theta0 back in degrees ("1 1.5 45 2")
  dataCoeffs(): string[] {
    const U = this.params.p('Umin'), th = this.params.p('theta0'), a = this.params.p('a');
    return Array.from({ length: this.ntypes }, (_, k) => `${k + 1} ${fmtCoeff(U[k + 1])} ${fmtCoeff(th[k + 1] / DEG2RAD)} ${fmtCoeff(a[k + 1])}`);
  }

  compute(bc: BondedCompute): void {
    const Umin = this.params.p('Umin'), th0 = this.params.p('theta0'), a = this.params.p('a');
    eachDihedral(bc, (t, phi) => {
      const dth = phi - th0[t];
      const U = -0.5 * (1 + Math.cos(dth));
      // dU/dphi = 0.5 sin(phi - theta0); dE/dphi = dE/dU * dU/dphi
      let e: number, dEdU: number;
      if (a[t] < 0.001) {
        // linear order in a: E = Umin*U - (a/2) Umin (U + U^2)
        e = Umin[t] * U - 0.5 * a[t] * Umin[t] * (U + U * U);
        dEdU = Umin[t] * (1 - 0.5 * a[t] * (1 + 2 * U));
      } else {
        const ea = Math.exp(a[t]);
        e = -Umin[t] * (Math.exp(-a[t] * U) - 1) / (ea - 1);
        dEdU = a[t] * Umin[t] * Math.exp(-a[t] * U) / (ea - 1);
      }
      return { e, dEdphi: dEdU * 0.5 * Math.sin(dth) };
    });
  }
}

/*
 * dihedral_style helix — docs.lammps.org/dihedral_helix.html:
 *
 *    E = A [1 - \cos(\theta)] + B [1 + \cos(3 \theta)] +
 *        C [1 + \cos(\theta + \frac{\pi}{4})]
 *
 * * :math:`A` (energy)
 * * :math:`B` (energy)
 * * :math:`C` (energy)
 */
export class DihedralHelix extends SimpleBonded {
  readonly name = 'helix';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['A', 'B', 'C'];

  compute(bc: BondedCompute): void {
    const A = this.params.p('A'), B = this.params.p('B'), C = this.params.p('C');
    eachDihedral(bc, (t, phi) => {
      return {
        e: A[t] * (1 - Math.cos(phi)) + B[t] * (1 + Math.cos(3 * phi)) + C[t] * (1 + Math.cos(phi + Math.PI / 4)),
        dEdphi: A[t] * Math.sin(phi) - 3 * B[t] * Math.sin(3 * phi) - C[t] * Math.sin(phi + Math.PI / 4),
      };
    });
  }
}

/*
 * dihedral_style zero — docs.lammps.org/dihedral_zero.html:
 *
 * "Using a dihedral style of zero means dihedral forces and energies are
 * not computed, but the geometry of dihedral quadruplets is still
 * accessible to other commands."
 *
 * "The optional nocoeff flag allows to read data files with a
 * DihedralCoeff section for any dihedral style. Similarly, any
 * dihedral_coeff commands will only be checked for the dihedral type number
 * and the rest ignored."
 *
 * "Note that the :doc:`dihedral_coeff <dihedral_coeff>` command must be
 * used for all dihedral types, though no additional values are specified."
 */
export class DihedralZero extends SimpleBonded {
  readonly name = 'zero';
  readonly kind = 'dihedral' as const;
  readonly paramNames: readonly string[] = [];
  private nocoeff = false;

  override settings(args: string[], _ctx?: StyleContext): void {
    for (const w of args) {
      if (w !== 'nocoeff') throw new StyleError(`dihedral_style zero: unknown keyword '${w}' (only nocoeff)`);
      this.nocoeff = true;
    }
  }

  override coeff(args: string[], _ctx?: StyleContext): void {
    if (!args.length) throw new StyleError('usage: dihedral_coeff N');
    if (!this.nocoeff && args.length > 1) {
      throw new StyleError("dihedral_coeff zero takes no coefficient values (use 'dihedral_style zero nocoeff' to ignore them)");
    }
    this.params.setRange(args[0], []);
  }

  compute(_bc: BondedCompute): void {}
}
