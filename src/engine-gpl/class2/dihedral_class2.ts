/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/dihedral_class2.cpp and src/CLASS2/dihedral_class2.h (DihedralClass2::compute, coeff,
 * allocate, write_data, Dihedral::problem and Dihedral::init) at commit b332f28. Contributing author: Eric Simon
 * (Cray). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * dihedral_style class2 (docs.lammps.org/dihedral_class2.html): the COMPASS class-2 torsion, E_d plus the
 * middle-bond, end-bond, angle-torsion, angle-angle-torsion and bond-bond-13 cross terms. The C++ compute is kept
 * in its own operation order: the dcosphidr, dphidr, dbonddr and dthetadr arrays and the fabcd accumulator, with the
 * energy terms added in native's order (dihedral, mbt, ebt (bond 1), ebt (bond 3), at (angle 1), at (angle 2), aat,
 * bb13). The bond vectors are the copies nearest to the second atom of the dihedral (bonded_util.ts delta). The
 * cos, sin, acos and exp calls use the glibc ports of src/engine-gpl/libm. The virial and per-atom tallies are
 * native's Dihedral::ev_tally (src/engine-gpl/xmolecule/ev.ts evTallyFour) with fabcd[0], fabcd[2], fabcd[3] and
 * vb1, vb2, vb3.
 */
import { SimpleBonded, atomIndex, delta, TypeParams } from '../../engine/force/bonded_util';
import { fmtCoeff } from '../../engine/force/util';
import { StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { formatNumber } from '../../engine/cformat';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
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

    const dcosphidr = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const dphidr = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const dbonddr = [
      [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ];
    const dthetadr = [
      [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ];
    const fabcd = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const vb1 = [0, 0, 0], vb2 = [0, 0, 0], vb3 = [0, 0, 0];
    const pI = [0, 0, 0], pK = [0, 0, 0], pL = [0, 0, 0];

    for (let n = 0; n < D.n; n++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, D.atoms[4 * n + w], 'dihedral'));
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

      // set up d(cos(phi))/d(r) array
      dcosphidr[0][0] = -sx1;
      dcosphidr[0][1] = -sy1;
      dcosphidr[0][2] = -sz1;
      dcosphidr[1][0] = sx2 + sx1;
      dcosphidr[1][1] = sy2 + sy1;
      dcosphidr[1][2] = sz2 + sz1;
      dcosphidr[2][0] = sx12 - sx2;
      dcosphidr[2][1] = sy12 - sy2;
      dcosphidr[2][2] = sz12 - sz2;
      dcosphidr[3][0] = -sx12;
      dcosphidr[3][1] = -sy12;
      dcosphidr[3][2] = -sz12;

      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) dphidr[i][j] = -dcosphidr[i][j] / sinphi;

      // energy
      const dphi1 = phi - phi1[type];
      const dphi2 = 2.0 * phi - phi2[type];
      const dphi3 = 3.0 * phi - phi3[type];

      let edihedral = k1[type] * (1.0 - libmCos(dphi1)) + k2[type] * (1.0 - libmCos(dphi2)) + k3[type] * (1.0 - libmCos(dphi3));

      const de_dihedral = k1[type] * libmSin(dphi1) + 2.0 * k2[type] * libmSin(dphi2) + 3.0 * k3[type] * libmSin(dphi3);

      // torsion forces on all 4 atoms
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) fabcd[i][j] = de_dihedral * dphidr[i][j];

      // d(bond)/d(r): dbonddr[bond][atom][coordinate]
      for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 3; k++) dbonddr[i][j][k] = 0.0;

      // bond1
      dbonddr[0][0][0] = vb1x / r1;
      dbonddr[0][0][1] = vb1y / r1;
      dbonddr[0][0][2] = vb1z / r1;
      dbonddr[0][1][0] = -vb1x / r1;
      dbonddr[0][1][1] = -vb1y / r1;
      dbonddr[0][1][2] = -vb1z / r1;

      // bond2
      dbonddr[1][1][0] = vb2x / r2;
      dbonddr[1][1][1] = vb2y / r2;
      dbonddr[1][1][2] = vb2z / r2;
      dbonddr[1][2][0] = -vb2x / r2;
      dbonddr[1][2][1] = -vb2y / r2;
      dbonddr[1][2][2] = -vb2z / r2;

      // bond3
      dbonddr[2][2][0] = vb3x / r3;
      dbonddr[2][2][1] = vb3y / r3;
      dbonddr[2][2][2] = vb3z / r3;
      dbonddr[2][3][0] = -vb3x / r3;
      dbonddr[2][3][1] = -vb3y / r3;
      dbonddr[2][3][2] = -vb3z / r3;

      // d(theta)/d(r) array: dthetadr[angle][atom][coordinate]
      for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 3; k++) dthetadr[i][j][k] = 0.0;

      const t1 = costh12 / r1mag2;
      const t2 = costh23 / r2mag2;
      const t3 = costh12 / r2mag2;
      const t4 = costh23 / r3mag2;

      // angle12
      dthetadr[0][0][0] = sc1 * ((t1 * vb1x) - (vb2x * r12c1));
      dthetadr[0][0][1] = sc1 * ((t1 * vb1y) - (vb2y * r12c1));
      dthetadr[0][0][2] = sc1 * ((t1 * vb1z) - (vb2z * r12c1));

      dthetadr[0][1][0] = sc1 * ((-t1 * vb1x) + (vb2x * r12c1) + (-t3 * vb2x) + (vb1x * r12c1));
      dthetadr[0][1][1] = sc1 * ((-t1 * vb1y) + (vb2y * r12c1) + (-t3 * vb2y) + (vb1y * r12c1));
      dthetadr[0][1][2] = sc1 * ((-t1 * vb1z) + (vb2z * r12c1) + (-t3 * vb2z) + (vb1z * r12c1));

      dthetadr[0][2][0] = sc1 * ((t3 * vb2x) - (vb1x * r12c1));
      dthetadr[0][2][1] = sc1 * ((t3 * vb2y) - (vb1y * r12c1));
      dthetadr[0][2][2] = sc1 * ((t3 * vb2z) - (vb1z * r12c1));

      // angle23
      dthetadr[1][1][0] = sc2 * ((t2 * vb2x) + (vb3x * r12c2));
      dthetadr[1][1][1] = sc2 * ((t2 * vb2y) + (vb3y * r12c2));
      dthetadr[1][1][2] = sc2 * ((t2 * vb2z) + (vb3z * r12c2));

      dthetadr[1][2][0] = sc2 * ((-t2 * vb2x) - (vb3x * r12c2) + (t4 * vb3x) + (vb2x * r12c2));
      dthetadr[1][2][1] = sc2 * ((-t2 * vb2y) - (vb3y * r12c2) + (t4 * vb3y) + (vb2y * r12c2));
      dthetadr[1][2][2] = sc2 * ((-t2 * vb2z) - (vb3z * r12c2) + (t4 * vb3z) + (vb2z * r12c2));

      dthetadr[1][3][0] = -sc2 * ((t4 * vb3x) + (vb2x * r12c2));
      dthetadr[1][3][1] = -sc2 * ((t4 * vb3y) + (vb2y * r12c2));
      dthetadr[1][3][2] = -sc2 * ((t4 * vb3z) + (vb2z * r12c2));

      // mid-bond/torsion coupling: energy on bond2 (middle bond)
      const cos2phi = libmCos(2.0 * phi);
      const cos3phi = libmCos(3.0 * phi);

      let bt1 = mbt_f1[type] * cosphi;
      let bt2 = mbt_f2[type] * cos2phi;
      let bt3 = mbt_f3[type] * cos3phi;
      let sumbte = bt1 + bt2 + bt3;
      let db = r2 - mbt_r0[type];
      edihedral += db * sumbte;

      // force on bond2
      bt1 = -mbt_f1[type] * sinphi;
      bt2 = -2.0 * mbt_f2[type] * libmSin(2.0 * phi);
      bt3 = -3.0 * mbt_f3[type] * libmSin(3.0 * phi);
      let sumbtf = bt1 + bt2 + bt3;

      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] += db * sumbtf * dphidr[i][j] + sumbte * dbonddr[1][i][j];
      }

      // end-bond/torsion coupling: energy on bond1 (first bond)
      bt1 = ebt_f1_1[type] * cosphi;
      bt2 = ebt_f2_1[type] * cos2phi;
      bt3 = ebt_f3_1[type] * cos3phi;
      sumbte = bt1 + bt2 + bt3;

      db = r1 - ebt_r0_1[type];
      edihedral += db * (bt1 + bt2 + bt3);

      // force on bond1
      bt1 = ebt_f1_1[type] * sinphi;
      bt2 = 2.0 * ebt_f2_1[type] * libmSin(2.0 * phi);
      bt3 = 3.0 * ebt_f3_1[type] * libmSin(3.0 * phi);
      sumbtf = bt1 + bt2 + bt3;

      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] -= db * sumbtf * dphidr[i][j] + sumbte * dbonddr[0][i][j];
      }

      // end-bond/torsion coupling: energy on bond3 (last bond)
      bt1 = ebt_f1_2[type] * cosphi;
      bt2 = ebt_f2_2[type] * cos2phi;
      bt3 = ebt_f3_2[type] * cos3phi;
      sumbte = bt1 + bt2 + bt3;

      db = r3 - ebt_r0_2[type];
      edihedral += db * (bt1 + bt2 + bt3);

      // force on bond3
      bt1 = -ebt_f1_2[type] * sinphi;
      bt2 = -2.0 * ebt_f2_2[type] * libmSin(2.0 * phi);
      bt3 = -3.0 * ebt_f3_2[type] * libmSin(3.0 * phi);
      sumbtf = bt1 + bt2 + bt3;

      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] += db * sumbtf * dphidr[i][j] + sumbte * dbonddr[2][i][j];
      }

      // angle/torsion coupling: energy on angle1
      let at1 = at_f1_1[type] * cosphi;
      let at2 = at_f2_1[type] * cos2phi;
      let at3 = at_f3_1[type] * cos3phi;
      sumbte = at1 + at2 + at3;

      let da = libmAcos(costh12) - at_theta0_1[type];
      edihedral += da * (at1 + at2 + at3);

      // force on angle1
      bt1 = at_f1_1[type] * sinphi;
      bt2 = 2.0 * at_f2_1[type] * libmSin(2.0 * phi);
      bt3 = 3.0 * at_f3_1[type] * libmSin(3.0 * phi);
      sumbtf = bt1 + bt2 + bt3;

      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] -= da * sumbtf * dphidr[i][j] + sumbte * dthetadr[0][i][j];
      }

      // energy on angle2
      at1 = at_f1_2[type] * cosphi;
      at2 = at_f2_2[type] * cos2phi;
      at3 = at_f3_2[type] * cos3phi;
      sumbte = at1 + at2 + at3;

      da = libmAcos(costh23) - at_theta0_2[type];
      edihedral += da * (at1 + at2 + at3);

      // force on angle2
      bt1 = -at_f1_2[type] * sinphi;
      bt2 = -2.0 * at_f2_2[type] * libmSin(2.0 * phi);
      bt3 = -3.0 * at_f3_2[type] * libmSin(3.0 * phi);
      sumbtf = bt1 + bt2 + bt3;

      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] += da * sumbtf * dphidr[i][j] + sumbte * dthetadr[1][i][j];
      }

      // angle/angle/torsion coupling
      const da1 = libmAcos(costh12) - aat_theta0_1[type];
      const da2 = libmAcos(costh23) - aat_theta0_2[type];

      edihedral += aat_k[type] * da1 * da2 * cosphi;

      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] -= aat_k[type] *
          (cosphi * (da2 * dthetadr[0][i][j] - da1 * dthetadr[1][i][j]) + sinphi * da1 * da2 * dphidr[i][j]);
      }

      // bond1/bond3 coupling (BondBond13)
      if (Math.abs(bb13t_k[type]) > SMALL) {
        const r1_0 = bb13t_r10[type];
        const r3_0 = bb13t_r30[type];
        const dr1 = r1 - r1_0;
        const dr2 = r3 - r3_0;
        const tk1 = -bb13t_k[type] * dr1 / r3;
        const tk2 = -bb13t_k[type] * dr2 / r1;

        edihedral += bb13t_k[type] * dr1 * dr2;

        fabcd[0][0] += tk2 * vb1x;
        fabcd[0][1] += tk2 * vb1y;
        fabcd[0][2] += tk2 * vb1z;

        fabcd[1][0] -= tk2 * vb1x;
        fabcd[1][1] -= tk2 * vb1y;
        fabcd[1][2] -= tk2 * vb1z;

        fabcd[2][0] -= tk1 * vb3x;
        fabcd[2][1] -= tk1 * vb3y;
        fabcd[2][2] -= tk1 * vb3z;

        fabcd[3][0] += tk1 * vb3x;
        fabcd[3][1] += tk1 * vb3y;
        fabcd[3][2] += tk1 * vb3z;
      }

      // apply force to each of 4 atoms
      for (let a = 0; a < 4; a++) for (let d = 0; d < 3; d++) bc.f[3 * atoms[a] + d] += fabcd[a][d];

      // Dihedral::ev_tally with f1 = fabcd[0], f3 = fabcd[2], f4 = fabcd[3] and the bond vectors vb1, vb2, vb3
      evTallyFour(bc, atoms, 'dihed', edihedral, fabcd[0], fabcd[2], fabcd[3], vb1, vb2, vb3);
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
