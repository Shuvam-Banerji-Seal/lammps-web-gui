/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/dihedral_class2.cpp and src/CLASS2/dihedral_class2.h (DihedralClass2::compute, coeff,
 * allocate, write_data, Dihedral::problem and Dihedral::init) at commit b332f28. Contributing author: Eric Simon
 * (Cray). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * dihedral_style class2 (docs.lammps.org/dihedral_class2.html): the COMPASS class-2 torsion, E_d plus the
 * middle-bond, end-bond, angle-torsion, angle-angle-torsion and bond-bond-13 cross terms. The C++ compute is kept
 * in its own operation order. Its dcosphidr, dphidr, dbonddr and dthetadr arrays and the fabcd accumulator are scalars
 * per atom and coordinate, each entry computed with native's expression. The dbonddr and dthetadr entries that native
 * holds as 0 are left out of the force sums: adding sumbte * 0 changes no value, only the sign of an exact zero. The
 * energy terms are added in native's order (dihedral, mbt, ebt (bond 1), ebt (bond 3), at (angle 1), at (angle 2), aat,
 * bb13). The bond vectors are the copies nearest to the second atom of the dihedral (bonded_util.ts delta). The
 * cos, sin, acos and exp calls use the glibc ports of src/engine-gpl/libm. The virial and per-atom tallies are
 * native's Dihedral::ev_tally (src/engine-gpl/xmolecule/ev.ts evTallyFour) with fabcd[0], fabcd[2], fabcd[3] and
 * vb1, vb2, vb3.
 */
import { SimpleBonded, atomIndex, delta, TypeParams } from '../../engine/force/bonded_util';
import { fmtCoeff } from '../../engine/force/util';
import { StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { formatNumber } from '../../engine/cformat';
import { sincos as libmSincos, SINCOS } from '../libm/sincos';
import { acos as libmAcos } from '../libm/acos';
import { evTallyFour } from '../xmolecule/ev';

/** dihedral_class2.cpp: TOLERANCE and SMALL (static constexpr). */
const TOLERANCE = 0.05;
const SMALL = 0.0000001;
/** math_const.h MY_PI */
const MY_PI = Math.PI;

/**
 * The coefficient names of each keyword, in the order of the C++ coeff() (which also sets the degree conversions).
 * deg lists the positions converted from degrees to radians (at: theta0_1, theta0_2; aat: theta0_1, theta0_2).
 */
const CROSS: Record<'mbt' | 'ebt' | 'at' | 'aat' | 'bb13', { names: readonly string[]; deg: readonly number[]; n: number }> = {
  mbt: { names: ['A1', 'A2', 'A3', 'r2'], deg: [], n: 6 },
  ebt: { names: ['B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'r1', 'r3'], deg: [], n: 10 },
  at: { names: ['D1', 'D2', 'D3', 'E1', 'E2', 'E3', 'theta1', 'theta2'], deg: [6, 7], n: 10 },
  aat: { names: ['M', 'theta1', 'theta2'], deg: [1, 2], n: 5 },
  bb13: { names: ['N', 'r1', 'r3'], deg: [], n: 5 },
};
type CrossKind = keyof typeof CROSS;

/** The numbers of the main (Dihedral Coeffs) line, in the order of coeff(): K1 phi1 K2 phi2 K3 phi3 (phi in degrees). */
const MAIN_NAMES = ['K1', 'phi1', 'K2', 'phi2', 'K3', 'phi3'];
const DECIMAL = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** utils::numeric (utils.cpp) with do_abort: the native text for a value that is not a floating point number. */
const numeric = (w: string): number => {
  if (w === '') throw new StyleError('Expected floating point parameter instead of NULL or empty string in input script or data file');
  if (!DECIMAL.test(w)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  return Number(w);
};

/** error->all(FLERR, "Incorrect args for dihedral coefficients" + utils::errorurl(21)). */
const badArgs = (): StyleError =>
  new StyleError('Incorrect args for dihedral coefficients\nFor more information see https://docs.lammps.org/err0021');

/** utils::bounds errors and the type loop of coeff() are shared with the other styles (typeBounds). */
const isKind = (kw: string): kw is CrossKind => kw === 'mbt' || kw === 'ebt' || kw === 'at' || kw === 'aat' || kw === 'bb13';

export class DihedralClass2 extends SimpleBonded {
  readonly name = 'class2';
  readonly kind = 'dihedral' as const;
  readonly paramNames = MAIN_NAMES;

  private mbt!: TypeParams;
  private ebt!: TypeParams;
  private at!: TypeParams;
  private aat!: TypeParams;
  private bb13!: TypeParams;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.mbt = new TypeParams(ntypes, CROSS.mbt.names);
    this.ebt = new TypeParams(ntypes, CROSS.ebt.names);
    this.at = new TypeParams(ntypes, CROSS.at.names);
    this.aat = new TypeParams(ntypes, CROSS.aat.names);
    this.bb13 = new TypeParams(ntypes, CROSS.bb13.names);
  }

  private group(kw: CrossKind): TypeParams {
    if (kw === 'mbt') return this.mbt;
    if (kw === 'ebt') return this.ebt;
    if (kw === 'at') return this.at;
    if (kw === 'aat') return this.aat;
    return this.bb13;
  }

  /** Dihedral::settings: no arguments. */
  override settings(args: string[], _ctx?: StyleContext): void {
    if (args.length > 0) throw new StyleError(`Illegal dihedral_style class2 argument: ${args[0]}`);
  }

  /**
   * DihedralClass2::coeff (args include the type, as in the C++ narg/arg): "dihedral_coeff N K1 phi1 K2 phi2 K3 phi3"
   * (phi in degrees), or "N mbt|ebt|at|aat|bb13 ..." with the coefficient count of that keyword (the angles of at and
   * aat in degrees). The type range is checked first (utils::bounds), then the argument count, then each number.
   */
  override coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 2) throw new StyleError('Invalid coeffs for this dihedral style');
    const word = args[0];
    typeBounds(word, this.ntypes);
    const kw = args[1];
    if (isKind(kw)) {
      const g = CROSS[kw];
      if (args.length !== g.n) throw badArgs();
      const grp = this.group(kw);
      const nums = args.slice(2).map(numeric);
      // As native LAMMPS does (src/CLASS2/dihedral_class2.cpp at b332f28): at_theta0_1[i] = theta0_1_one/180.0 * MY_PI
      for (const k of g.deg) nums[k] = (nums[k] / 180.0) * MY_PI;
      grp.setRange(word, nums);
      return;
    }
    if (args.length !== 7) throw badArgs();
    const nums = args.slice(1).map(numeric);
    // As native LAMMPS does (src/CLASS2/dihedral_class2.cpp at b332f28): phi1[i] = phi1_one/180.0 * MY_PI
    nums[1] = (nums[1] / 180.0) * MY_PI; nums[3] = (nums[3] / 180.0) * MY_PI; nums[5] = (nums[5] / 180.0) * MY_PI;
    this.params.setRange(word, nums);
  }

  /** Native setflag[t]: the main line and the five cross-term lines are all set (DihedralClass2::coeff). */
  override coeffSet(t: number): boolean | null {
    if (this.params.set.length !== this.ntypes + 1) return null;
    return this.params.set[t] === 1 && this.mbt.set[t] === 1 && this.ebt.set[t] === 1 && this.at.set[t] === 1 &&
      this.aat.set[t] === 1 && this.bb13.set[t] === 1;
  }

  /**
   * Dihedral::init: every type needs the main line and the five cross-term lines (setflag is set only when all six
   * are). The status block is the native Info::get_dihedral_coeff_status text.
   */
  override init(_ctx?: StyleContext): void {
    const ok = (t: number) => this.params.set[t] === 1 && this.mbt.set[t] === 1 && this.ebt.set[t] === 1 &&
      this.at.set[t] === 1 && this.aat.set[t] === 1 && this.bb13.set[t] === 1;
    let bad = false;
    for (let t = 1; t <= this.ntypes; t++) if (!ok(t)) bad = true;
    if (!bad) return;
    const lines: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) lines.push(`${String(t).padStart(6)}: is${ok(t) ? ' ' : ' not '}set`);
    throw new StyleError(`All dihedral coeffs are not set. Status:\n${lines.join('\n')}`);
  }

  /** write_data: "%d %g %g %g %g %g %g" with the phi values in degrees (phi1[i]*180.0/MY_PI). */
  override dataCoeffs(): string[] {
    const K1 = this.params.p('K1'), P1 = this.params.p('phi1'), K2 = this.params.p('K2'), P2 = this.params.p('phi2');
    const K3 = this.params.p('K3'), P3 = this.params.p('phi3');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${fmtCoeff(K1[t])} ${fmtCoeff((P1[t] * 180.0) / MY_PI)} ${fmtCoeff(K2[t])} ${fmtCoeff((P2[t] * 180.0) / MY_PI)} ` +
        `${fmtCoeff(K3[t])} ${fmtCoeff((P3[t] * 180.0) / MY_PI)}`);
    }
    return out;
  }

  /**
   * write_data cross sections, in native's order: AngleAngleTorsion, EndBondTorsion, MiddleBondTorsion, BondBond13,
   * AngleTorsion (the angles in degrees, the distances unchanged). data.ts maps the titles back to the keywords.
   */
  override dataCrossSections(): { title: string; lines: string[] }[] {
    const sec = (title: string, kw: CrossKind) => {
      const g = this.group(kw);
      const lines: string[] = [];
      for (let t = 1; t <= this.ntypes; t++) {
        const vals = CROSS[kw].names.map((nm, k) => {
          const v = g.p(nm)[t];
          // As native LAMMPS does (src/CLASS2/dihedral_class2.cpp at b332f28): write_data: aat_theta0_1[i]*180.0/MY_PI
          return fmtCoeff(CROSS[kw].deg.includes(k) ? (v * 180.0) / MY_PI : v);
        });
        lines.push(`${t} ${vals.join(' ')}`);
      }
      return { title, lines };
    };
    return [
      sec('AngleAngleTorsion Coeffs', 'aat'), sec('EndBondTorsion Coeffs', 'ebt'), sec('MiddleBondTorsion Coeffs', 'mbt'),
      sec('BondBond13 Coeffs', 'bb13'), sec('AngleTorsion Coeffs', 'at'),
    ];
  }

  compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const k1 = this.params.p('K1'), k2 = this.params.p('K2'), k3 = this.params.p('K3');
    const phi1 = this.params.p('phi1'), phi2 = this.params.p('phi2'), phi3 = this.params.p('phi3');
    const mbt_f1 = this.mbt.p('A1'), mbt_f2 = this.mbt.p('A2'), mbt_f3 = this.mbt.p('A3'), mbt_r0 = this.mbt.p('r2');
    const ebt_f1_1 = this.ebt.p('B1'), ebt_f2_1 = this.ebt.p('B2'), ebt_f3_1 = this.ebt.p('B3'), ebt_r0_1 = this.ebt.p('r1');
    const ebt_f1_2 = this.ebt.p('C1'), ebt_f2_2 = this.ebt.p('C2'), ebt_f3_2 = this.ebt.p('C3'), ebt_r0_2 = this.ebt.p('r3');
    const at_f1_1 = this.at.p('D1'), at_f2_1 = this.at.p('D2'), at_f3_1 = this.at.p('D3'), at_theta0_1 = this.at.p('theta1');
    const at_f1_2 = this.at.p('E1'), at_f2_2 = this.at.p('E2'), at_f3_2 = this.at.p('E3'), at_theta0_2 = this.at.p('theta2');
    const aat_k = this.aat.p('M'), aat_theta0_1 = this.aat.p('theta1'), aat_theta0_2 = this.aat.p('theta2');
    const bb13t_k = this.bb13.p('N'), bb13t_r10 = this.bb13.p('r1'), bb13t_r30 = this.bb13.p('r3');

    // fabcd[0], fabcd[2] and fabcd[3] of native compute() for the tally
    const fab0 = new Float64Array(3), fab2 = new Float64Array(3), fab3 = new Float64Array(3);
    const vb1 = [0, 0, 0], vb2 = [0, 0, 0], vb3 = [0, 0, 0];
    const pI = [0, 0, 0], pK = [0, 0, 0], pL = [0, 0, 0];

    // the four atom indices of the current dihedral (one array, refilled per dihedral)
    const atoms = [0, 0, 0, 0];
    for (let n = 0; n < D.n; n++) {
      for (let w = 0; w < 4; w++) atoms[w] = atomIndex(bc, D.atoms[4 * n + w], 'dihedral');
      const i1 = atoms[0], i2 = atoms[1], i3 = atoms[2], i4 = atoms[3];
      const type = D.type[n];

      // 1st bond: vb1 = x1 - x2; 2nd bond: vb2 = x3 - x2; 3rd bond: vb3 = x4 - x3 (copies nearest to atom 2)
      delta(bc, i2, i1, pI);
      delta(bc, i2, i3, pK);
      delta(bc, i2, i4, pL);
      for (let d = 0; d < 3; d++) { vb1[d] = pI[d]; vb2[d] = pK[d]; vb3[d] = pL[d] - pK[d]; }
      const vb1x = vb1[0], vb1y = vb1[1], vb1z = vb1[2];
      const vb2x = vb2[0], vb2y = vb2[1], vb2z = vb2[2];
      const vb3x = vb3[0], vb3y = vb3[1], vb3z = vb3[2];
      const vb2xm = -vb2x, vb2ym = -vb2y, vb2zm = -vb2z;

      // distances
      const r1mag2 = vb1x * vb1x + vb1y * vb1y + vb1z * vb1z;
      const r1 = Math.sqrt(r1mag2);
      const r2mag2 = vb2x * vb2x + vb2y * vb2y + vb2z * vb2z;
      const r2 = Math.sqrt(r2mag2);
      const r3mag2 = vb3x * vb3x + vb3y * vb3y + vb3z * vb3z;
      const r3 = Math.sqrt(r3mag2);

      const sb1 = 1.0 / r1mag2;
      const rb1 = 1.0 / r1;
      const sb2 = 1.0 / r2mag2;
      const rb2 = 1.0 / r2;
      const sb3 = 1.0 / r3mag2;
      const rb3 = 1.0 / r3;

      let c0 = (vb1x * vb3x + vb1y * vb3y + vb1z * vb3z) * rb1 * rb3;

      // angles
      const r12c1 = rb1 * rb2;
      const r12c2 = rb2 * rb3;
      let costh12 = (vb1x * vb2x + vb1y * vb2y + vb1z * vb2z) * r12c1;
      let costh13 = c0;
      let costh23 = (vb2xm * vb3x + vb2ym * vb3y + vb2zm * vb3z) * r12c2;

      costh12 = Math.max(Math.min(costh12, 1.0), -1.0);
      costh13 = Math.max(Math.min(costh13, 1.0), -1.0);
      costh23 = Math.max(Math.min(costh23, 1.0), -1.0);
      c0 = costh13;

      // cos and sin of 2 angles and final c
      let sin2 = Math.max(1.0 - costh12 * costh12, 0.0);
      let sc1 = Math.sqrt(sin2);
      if (sc1 < SMALL) sc1 = SMALL;
      sc1 = 1.0 / sc1;

      sin2 = Math.max(1.0 - costh23 * costh23, 0.0);
      let sc2 = Math.sqrt(sin2);
      if (sc2 < SMALL) sc2 = SMALL;
      sc2 = 1.0 / sc2;

      const s1 = sc1 * sc1;
      const s2 = sc2 * sc2;
      const s12 = sc1 * sc2;
      let c = (c0 + costh12 * costh23) * s12;

      // error check (Dihedral::problem: a warning, and the term is still evaluated)
      if (c > 1.0 + TOLERANCE || c < -1.0 - TOLERANCE) problem(bc, atoms, Array.from(D.atoms.slice(4 * n, 4 * n + 4)));

      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      const cosphi = c;
      let phi = libmAcos(c);

      let sinphi = Math.sqrt(1.0 - c * c);
      sinphi = Math.max(sinphi, SMALL);

      // n123 = vb1 x vb2
      const n123x = vb1y * vb2z - vb1z * vb2y;
      const n123y = vb1z * vb2x - vb1x * vb2z;
      const n123z = vb1x * vb2y - vb1y * vb2x;
      const n123_dot_vb3 = n123x * vb3x + n123y * vb3y + n123z * vb3z;
      if (n123_dot_vb3 > 0.0) {
        phi = -phi;
        sinphi = -sinphi;
      }

      const a11 = -c * sb1 * s1;
      const a22 = sb2 * (2.0 * costh13 * s12 - c * (s1 + s2));
      const a33 = -c * sb3 * s2;
      const a12 = r12c1 * (costh12 * c * s1 + costh23 * s12);
      const a13 = rb1 * rb3 * s12;
      const a23 = r12c2 * (-costh23 * c * s2 - costh12 * s12);

      const sx1 = a11 * vb1x + a12 * vb2x + a13 * vb3x;
      const sx2 = a12 * vb1x + a22 * vb2x + a23 * vb3x;
      const sx12 = a13 * vb1x + a23 * vb2x + a33 * vb3x;
      const sy1 = a11 * vb1y + a12 * vb2y + a13 * vb3y;
      const sy2 = a12 * vb1y + a22 * vb2y + a23 * vb3y;
      const sy12 = a13 * vb1y + a23 * vb2y + a33 * vb3y;
      const sz1 = a11 * vb1z + a12 * vb2z + a13 * vb3z;
      const sz2 = a12 * vb1z + a22 * vb2z + a23 * vb3z;
      const sz12 = a13 * vb1z + a23 * vb2z + a33 * vb3z;

      // d(cos(phi))/d(r) (dcosphidr), d(phi)/d(r) = -dcosphidr / sinphi (dphidr) and the torsion forces (fabcd), as
      // scalars per atom and coordinate: P<atom><coordinate>, F<atom><coordinate>
      const P0x = sx1 / sinphi;
      const P0y = sy1 / sinphi;
      const P0z = sz1 / sinphi;
      const P1x = -(sx2 + sx1) / sinphi;
      const P1y = -(sy2 + sy1) / sinphi;
      const P1z = -(sz2 + sz1) / sinphi;
      const P2x = -(sx12 - sx2) / sinphi;
      const P2y = -(sy12 - sy2) / sinphi;
      const P2z = -(sz12 - sz2) / sinphi;
      const P3x = sx12 / sinphi;
      const P3y = sy12 / sinphi;
      const P3z = sz12 / sinphi;

      // energy
      const dphi1 = phi - phi1[type];
      const dphi2 = 2.0 * phi - phi2[type];
      const dphi3 = 3.0 * phi - phi3[type];

      // sin and cos of each argument from one sincos (the same values as sin() and cos(), bit for bit); 2 phi and 3 phi
      // reuse those of dphi2 and dphi3 when they are the same number (phi2 = phi3 = 0, as in most force fields)
      libmSincos(dphi1);
      const sn1 = SINCOS[0], cs1 = SINCOS[1];
      libmSincos(dphi2);
      const sn2 = SINCOS[0], cs2 = SINCOS[1];
      libmSincos(dphi3);
      const sn3 = SINCOS[0], cs3 = SINCOS[1];

      let edihedral = k1[type] * (1.0 - cs1) + k2[type] * (1.0 - cs2) + k3[type] * (1.0 - cs3);

      const de_dihedral = k1[type] * sn1 + 2.0 * k2[type] * sn2 + 3.0 * k3[type] * sn3;

      // torsion forces on all 4 atoms
      let F0x = de_dihedral * P0x; let F0y = de_dihedral * P0y; let F0z = de_dihedral * P0z;
      let F1x = de_dihedral * P1x; let F1y = de_dihedral * P1y; let F1z = de_dihedral * P1z;
      let F2x = de_dihedral * P2x; let F2y = de_dihedral * P2y; let F2z = de_dihedral * P2z;
      let F3x = de_dihedral * P3x; let F3y = de_dihedral * P3y; let F3z = de_dihedral * P3z;

      // d(bond)/d(r): bond 1 on atoms 1, 2; bond 2 on atoms 2, 3; bond 3 on atoms 3, 4 (dbonddr; the other entries are 0)
      const B00x = vb1x / r1, B01x = -vb1x / r1;
      const B00y = vb1y / r1, B01y = -vb1y / r1;
      const B00z = vb1z / r1, B01z = -vb1z / r1;
      const B11x = vb2x / r2, B12x = -vb2x / r2;
      const B11y = vb2y / r2, B12y = -vb2y / r2;
      const B11z = vb2z / r2, B12z = -vb2z / r2;
      const B22x = vb3x / r3, B23x = -vb3x / r3;
      const B22y = vb3y / r3, B23y = -vb3y / r3;
      const B22z = vb3z / r3, B23z = -vb3z / r3;

      // d(theta)/d(r) (dthetadr): angle12 on atoms 1-3, angle23 on atoms 2-4 (the other entries are 0)
      const t1 = costh12 / r1mag2;
      const t2 = costh23 / r2mag2;
      const t3 = costh12 / r2mag2;
      const t4 = costh23 / r3mag2;
      const T00x = sc1 * ((t1 * vb1x) - (vb2x * r12c1));
      const T00y = sc1 * ((t1 * vb1y) - (vb2y * r12c1));
      const T00z = sc1 * ((t1 * vb1z) - (vb2z * r12c1));
      const T01x = sc1 * ((-t1 * vb1x) + (vb2x * r12c1) + (-t3 * vb2x) + (vb1x * r12c1));
      const T01y = sc1 * ((-t1 * vb1y) + (vb2y * r12c1) + (-t3 * vb2y) + (vb1y * r12c1));
      const T01z = sc1 * ((-t1 * vb1z) + (vb2z * r12c1) + (-t3 * vb2z) + (vb1z * r12c1));
      const T02x = sc1 * ((t3 * vb2x) - (vb1x * r12c1));
      const T02y = sc1 * ((t3 * vb2y) - (vb1y * r12c1));
      const T02z = sc1 * ((t3 * vb2z) - (vb1z * r12c1));
      const T11x = sc2 * ((t2 * vb2x) + (vb3x * r12c2));
      const T11y = sc2 * ((t2 * vb2y) + (vb3y * r12c2));
      const T11z = sc2 * ((t2 * vb2z) + (vb3z * r12c2));
      const T12x = sc2 * ((-t2 * vb2x) - (vb3x * r12c2) + (t4 * vb3x) + (vb2x * r12c2));
      const T12y = sc2 * ((-t2 * vb2y) - (vb3y * r12c2) + (t4 * vb3y) + (vb2y * r12c2));
      const T12z = sc2 * ((-t2 * vb2z) - (vb3z * r12c2) + (t4 * vb3z) + (vb2z * r12c2));
      const T13x = -sc2 * ((t4 * vb3x) + (vb2x * r12c2));
      const T13y = -sc2 * ((t4 * vb3y) + (vb2y * r12c2));
      const T13z = -sc2 * ((t4 * vb3z) + (vb2z * r12c2));

      // mid-bond/torsion coupling: energy on bond2 (middle bond)
      // cos and sin of 2 phi and 3 phi and the two bond angles, which every cross term below reads (the C++ calls sin()
      // and acos() again for each; the same arguments give the same values)
      const phi2x = 2.0 * phi, phi3x = 3.0 * phi;
      let cos2phi = cs2, sin2phi = sn2, cos3phi = cs3, sin3phi = sn3;
      if (!Object.is(phi2x, dphi2)) { libmSincos(phi2x); sin2phi = SINCOS[0]; cos2phi = SINCOS[1]; }
      if (!Object.is(phi3x, dphi3)) { libmSincos(phi3x); sin3phi = SINCOS[0]; cos3phi = SINCOS[1]; }
      const acos12 = libmAcos(costh12);
      const acos23 = libmAcos(costh23);

      let bt1 = mbt_f1[type] * cosphi;
      let bt2 = mbt_f2[type] * cos2phi;
      let bt3 = mbt_f3[type] * cos3phi;
      let sumbte = bt1 + bt2 + bt3;
      let db = r2 - mbt_r0[type];
      edihedral += db * sumbte;

      // force on bond2
      bt1 = -mbt_f1[type] * sinphi;
      bt2 = -2.0 * mbt_f2[type] * sin2phi;
      bt3 = -3.0 * mbt_f3[type] * sin3phi;
      let sumbtf = bt1 + bt2 + bt3;

      // fabcd += db * sumbtf * dphidr + sumbte * dbonddr[1]
      let g = db * sumbtf;
      F0x += g * P0x; F0y += g * P0y; F0z += g * P0z;
      F1x += g * P1x + sumbte * B11x; F1y += g * P1y + sumbte * B11y; F1z += g * P1z + sumbte * B11z;
      F2x += g * P2x + sumbte * B12x; F2y += g * P2y + sumbte * B12y; F2z += g * P2z + sumbte * B12z;
      F3x += g * P3x; F3y += g * P3y; F3z += g * P3z;

      // end-bond/torsion coupling: energy on bond1 (first bond)
      bt1 = ebt_f1_1[type] * cosphi;
      bt2 = ebt_f2_1[type] * cos2phi;
      bt3 = ebt_f3_1[type] * cos3phi;
      sumbte = bt1 + bt2 + bt3;

      db = r1 - ebt_r0_1[type];
      edihedral += db * (bt1 + bt2 + bt3);

      // force on bond1
      bt1 = ebt_f1_1[type] * sinphi;
      bt2 = 2.0 * ebt_f2_1[type] * sin2phi;
      bt3 = 3.0 * ebt_f3_1[type] * sin3phi;
      sumbtf = bt1 + bt2 + bt3;

      // fabcd -= db * sumbtf * dphidr + sumbte * dbonddr[0]
      g = db * sumbtf;
      F0x -= g * P0x + sumbte * B00x; F0y -= g * P0y + sumbte * B00y; F0z -= g * P0z + sumbte * B00z;
      F1x -= g * P1x + sumbte * B01x; F1y -= g * P1y + sumbte * B01y; F1z -= g * P1z + sumbte * B01z;
      F2x -= g * P2x; F2y -= g * P2y; F2z -= g * P2z;
      F3x -= g * P3x; F3y -= g * P3y; F3z -= g * P3z;

      // end-bond/torsion coupling: energy on bond3 (last bond)
      bt1 = ebt_f1_2[type] * cosphi;
      bt2 = ebt_f2_2[type] * cos2phi;
      bt3 = ebt_f3_2[type] * cos3phi;
      sumbte = bt1 + bt2 + bt3;

      db = r3 - ebt_r0_2[type];
      edihedral += db * (bt1 + bt2 + bt3);

      // force on bond3
      bt1 = -ebt_f1_2[type] * sinphi;
      bt2 = -2.0 * ebt_f2_2[type] * sin2phi;
      bt3 = -3.0 * ebt_f3_2[type] * sin3phi;
      sumbtf = bt1 + bt2 + bt3;

      // fabcd += db * sumbtf * dphidr + sumbte * dbonddr[2]
      g = db * sumbtf;
      F0x += g * P0x; F0y += g * P0y; F0z += g * P0z;
      F1x += g * P1x; F1y += g * P1y; F1z += g * P1z;
      F2x += g * P2x + sumbte * B22x; F2y += g * P2y + sumbte * B22y; F2z += g * P2z + sumbte * B22z;
      F3x += g * P3x + sumbte * B23x; F3y += g * P3y + sumbte * B23y; F3z += g * P3z + sumbte * B23z;

      // angle/torsion coupling: energy on angle1
      let at1 = at_f1_1[type] * cosphi;
      let at2 = at_f2_1[type] * cos2phi;
      let at3 = at_f3_1[type] * cos3phi;
      sumbte = at1 + at2 + at3;

      let da = acos12 - at_theta0_1[type];
      edihedral += da * (at1 + at2 + at3);

      // force on angle1
      bt1 = at_f1_1[type] * sinphi;
      bt2 = 2.0 * at_f2_1[type] * sin2phi;
      bt3 = 3.0 * at_f3_1[type] * sin3phi;
      sumbtf = bt1 + bt2 + bt3;

      // fabcd -= da * sumbtf * dphidr + sumbte * dthetadr[0]
      g = da * sumbtf;
      F0x -= g * P0x + sumbte * T00x; F0y -= g * P0y + sumbte * T00y; F0z -= g * P0z + sumbte * T00z;
      F1x -= g * P1x + sumbte * T01x; F1y -= g * P1y + sumbte * T01y; F1z -= g * P1z + sumbte * T01z;
      F2x -= g * P2x + sumbte * T02x; F2y -= g * P2y + sumbte * T02y; F2z -= g * P2z + sumbte * T02z;
      F3x -= g * P3x; F3y -= g * P3y; F3z -= g * P3z;

      // energy on angle2
      at1 = at_f1_2[type] * cosphi;
      at2 = at_f2_2[type] * cos2phi;
      at3 = at_f3_2[type] * cos3phi;
      sumbte = at1 + at2 + at3;

      da = acos23 - at_theta0_2[type];
      edihedral += da * (at1 + at2 + at3);

      // force on angle2
      bt1 = -at_f1_2[type] * sinphi;
      bt2 = -2.0 * at_f2_2[type] * sin2phi;
      bt3 = -3.0 * at_f3_2[type] * sin3phi;
      sumbtf = bt1 + bt2 + bt3;

      // fabcd += da * sumbtf * dphidr + sumbte * dthetadr[1]
      g = da * sumbtf;
      F0x += g * P0x; F0y += g * P0y; F0z += g * P0z;
      F1x += g * P1x + sumbte * T11x; F1y += g * P1y + sumbte * T11y; F1z += g * P1z + sumbte * T11z;
      F2x += g * P2x + sumbte * T12x; F2y += g * P2y + sumbte * T12y; F2z += g * P2z + sumbte * T12z;
      F3x += g * P3x + sumbte * T13x; F3y += g * P3y + sumbte * T13y; F3z += g * P3z + sumbte * T13z;

      // angle/angle/torsion coupling
      const da1 = acos12 - aat_theta0_1[type];
      const da2 = acos23 - aat_theta0_2[type];

      edihedral += aat_k[type] * da1 * da2 * cosphi;

      // fabcd -= aat_k * (cosphi * (da2 * dthetadr[0] - da1 * dthetadr[1]) + sinphi * da1 * da2 * dphidr)
      const ak = aat_k[type];
      const k = sinphi * da1 * da2;
      F0x -= ak * (cosphi * (da2 * T00x) + k * P0x); F0y -= ak * (cosphi * (da2 * T00y) + k * P0y); F0z -= ak * (cosphi * (da2 * T00z) + k * P0z);
      F1x -= ak * (cosphi * (da2 * T01x - da1 * T11x) + k * P1x); F1y -= ak * (cosphi * (da2 * T01y - da1 * T11y) + k * P1y); F1z -= ak * (cosphi * (da2 * T01z - da1 * T11z) + k * P1z);
      F2x -= ak * (cosphi * (da2 * T02x - da1 * T12x) + k * P2x); F2y -= ak * (cosphi * (da2 * T02y - da1 * T12y) + k * P2y); F2z -= ak * (cosphi * (da2 * T02z - da1 * T12z) + k * P2z);
      F3x -= ak * (cosphi * (0.0 - da1 * T13x) + k * P3x); F3y -= ak * (cosphi * (0.0 - da1 * T13y) + k * P3y); F3z -= ak * (cosphi * (0.0 - da1 * T13z) + k * P3z);

      // bond1/bond3 coupling (BondBond13)
      if (Math.abs(bb13t_k[type]) > SMALL) {
        const r1_0 = bb13t_r10[type];
        const r3_0 = bb13t_r30[type];
        const dr1 = r1 - r1_0;
        const dr2 = r3 - r3_0;
        const tk1 = -bb13t_k[type] * dr1 / r3;
        const tk2 = -bb13t_k[type] * dr2 / r1;

        edihedral += bb13t_k[type] * dr1 * dr2;

        F0x += tk2 * vb1x;
        F0y += tk2 * vb1y;
        F0z += tk2 * vb1z;

        F1x -= tk2 * vb1x;
        F1y -= tk2 * vb1y;
        F1z -= tk2 * vb1z;

        F2x -= tk1 * vb3x;
        F2y -= tk1 * vb3y;
        F2z -= tk1 * vb3z;

        F3x += tk1 * vb3x;
        F3y += tk1 * vb3y;
        F3z += tk1 * vb3z;
      }

      // apply force to each of 4 atoms
      const f = bc.f;
      f[3 * i1] += F0x; f[3 * i1 + 1] += F0y; f[3 * i1 + 2] += F0z;
      f[3 * i2] += F1x; f[3 * i2 + 1] += F1y; f[3 * i2 + 2] += F1z;
      f[3 * i3] += F2x; f[3 * i3 + 1] += F2y; f[3 * i3 + 2] += F2z;
      f[3 * i4] += F3x; f[3 * i4 + 1] += F3y; f[3 * i4 + 2] += F3z;

      // Dihedral::ev_tally with f1 = fabcd[0], f3 = fabcd[2], f4 = fabcd[3] and the bond vectors vb1, vb2, vb3
      fab0[0] = F0x; fab0[1] = F0y; fab0[2] = F0z;
      fab2[0] = F2x; fab2[1] = F2y; fab2[2] = F2z;
      fab3[0] = F3x; fab3[1] = F3y; fab3[2] = F3z;
      evTallyFour(bc, atoms, 'dihed', edihedral, fab0, fab2, fab3, vb1, vb2, vb3);

    }
  }
}

/**
 * Dihedral::problem (dihedral.cpp at b332f28): a warning naming the step, the four atom IDs and their positions.
 * The warning goes through bc.warn, which prefixes the first line with WARNING: as native error->warning does.
 */
const problem = (bc: BondedCompute, atoms: readonly number[], ids: readonly number[]): void => {
  if (!bc.warn) return;
  const x = bc.s.x;
  const fmt = (v: number) => formatNumber(v, '%.8g');
  const names = ['1st', '2nd', '3rd', '4th'];
  let text = `Dihedral problem: 0 ${bc.s.step} ${ids.join(' ')}`;
  for (let a = 0; a < 4; a++) {
    const i = atoms[a];
    text += `\nWARNING:   ${names[a]} atom: 0 ${fmt(x[3 * i])} ${fmt(x[3 * i + 1])} ${fmt(x[3 * i + 2])}`;
  }
  bc.warn(text);
};
