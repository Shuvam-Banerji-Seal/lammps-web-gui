/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/dihedral_class2xe.cpp and dihedral_class2xe.h (DihedralClass2xe::compute,
 * coeff, allocate), at commit b332f28. The ClassII-xe force field of Kemppainen, Heinz and
 * Odegard, npj Computational Materials 11, 341 (2025), https://doi.org/10.1038/s41524-025-01838-5.
 * Authored by Josh Kemppainen (Michigan Technological University) per the C++ file header.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The C++ compute is kept in its own operation order (the vb1, vb2, vb3 bond vectors, the
 * dthetadr and dbonddr arrays and the fabcd force accumulator). The bond vectors are taken from the
 * copies nearest to the second atom of the dihedral (bonded_util.ts delta), as the other dihedral styles do.
 */
import { SimpleBonded, atomIndex, delta, TypeParams } from '../../engine/force/bonded_util';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { bondedVirial, StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** docs.lammps.org/dihedral_class2.html (class2xe section): "The class2xe dihedral style uses the potential". */
const SMALL = 0.0000001;
const TOLERANCE = 0.05;
const DEG2RAD = Math.PI / 180;

/** The cross-term keywords with their coefficient names in the order of dihedral_coeff (C++ coeff). */
const CROSS: Record<string, { names: readonly string[]; deg: readonly number[] }> = {
  mbt: { names: ['alpha2', 'A1', 'A2', 'A3', 'r2'], deg: [] },
  ebt: { names: ['alpha1', 'alpha3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'r1', 'r3'], deg: [] },
  at: { names: ['D1', 'D2', 'D3', 'E1', 'E2', 'E3', 'theta1', 'theta2'], deg: [6, 7] },
  aat: { names: ['k', 'theta1', 'theta2'], deg: [1, 2] },
  bb13: { names: ['D', 'alpha', 'r1', 'r3'], deg: [] },
};

const MAIN_NAMES = ['K1', 'phi1', 'K2', 'phi2', 'K3', 'phi3'];

/** Vector helpers on 3-element arrays (scratch, the function is not re-entered). */
const v3 = (): number[] => [0, 0, 0];

export class DihedralClass2xe extends SimpleBonded {
  readonly name = 'class2xe';
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

  private group(kw: string): TypeParams {
    return { mbt: this.mbt, ebt: this.ebt, at: this.at, aat: this.aat, bb13: this.bb13 }[kw]!;
  }

  /**
   * DihedralClass2xe::coeff: "dihedral_coeff N K1 phi1 K2 phi2 K3 phi3" (degrees), or one of the
   * keywords mbt, ebt, at, aat, bb13 with its coefficients (the angles of at and aat in degrees).
   */
  override coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 2) throw new StyleError('Invalid coeffs for this dihedral style');
    const word = args[0];
    const kw = args[1];
    if (kw in CROSS) {
      const g = this.group(kw);
      const vals = args.slice(2);
      if (vals.length !== g.names.length) throw new StyleError('Incorrect args for dihedral coefficients');
      const nums = vals.map((w, k) => parseNum(w, g.names[k]));
      for (const k of CROSS[kw].deg) nums[k] *= DEG2RAD;
      g.setRange(word, nums);
      return;
    }
    if (/[a-zA-Z]/.test(kw)) {
      throw new StyleError(`unknown dihedral_coeff class2xe keyword '${kw}' (expected a numeric line or mbt, ebt, at, aat, bb13)`);
    }
    if (args.length !== 7) throw new StyleError('Incorrect args for dihedral coefficients');
    const nums = args.slice(1).map((w, k) => parseNum(w, MAIN_NAMES[k]));
    nums[1] *= DEG2RAD; nums[3] *= DEG2RAD; nums[5] *= DEG2RAD;
    this.params.setRange(word, nums);
  }

  /** The six sets are required for every type (C++ coeff: setflag once all six are set). */
  override init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.params.set[t] || !this.mbt.set[t] || !this.ebt.set[t] || !this.at.set[t] || !this.aat.set[t] || !this.bb13.set[t]) {
        throw new StyleError(`All dihedral coeffs are not set (type ${t})`);
      }
    }
  }

  override dataCoeffs(): string[] {
    const K = this.params.p('K1'), P1 = this.params.p('phi1'), K2 = this.params.p('K2'), P2 = this.params.p('phi2');
    const K3 = this.params.p('K3'), P3 = this.params.p('phi3');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff(P1[t] / DEG2RAD)} ${fmtCoeff(K2[t])} ${fmtCoeff(P2[t] / DEG2RAD)} ${fmtCoeff(K3[t])} ${fmtCoeff(P3[t] / DEG2RAD)}`);
    }
    return out;
  }

  /** write_data cross sections in the order of the class2 style (data.ts maps the titles back to coeff keywords). */
  override dataCrossSections(): { title: string; lines: string[] }[] {
    const sec = (title: string, kw: string) => {
      const g = this.group(kw);
      const lines: string[] = [];
      for (let t = 1; t <= this.ntypes; t++) {
        lines.push(`${t} ${CROSS[kw].names.map((nm, k) => fmtCoeff(CROSS[kw].deg.includes(k) ? g.p(nm)[t] / DEG2RAD : g.p(nm)[t])).join(' ')}`);
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
    const K = [this.params.p('K1'), this.params.p('K2'), this.params.p('K3')];
    const PH = [this.params.p('phi1'), this.params.p('phi2'), this.params.p('phi3')];
    const mf1 = this.mbt.p('A1'), mf2 = this.mbt.p('A2'), mf3 = this.mbt.p('A3');
    const mAlpha2 = this.mbt.p('alpha2'), mR0 = this.mbt.p('r2');
    const e1f1 = this.ebt.p('B1'), e1f2 = this.ebt.p('B2'), e1f3 = this.ebt.p('B3');
    const e1Alpha = this.ebt.p('alpha1'), e1R0 = this.ebt.p('r1');
    const e2f1 = this.ebt.p('C1'), e2f2 = this.ebt.p('C2'), e2f3 = this.ebt.p('C3');
    const e2Alpha = this.ebt.p('alpha3'), e2R0 = this.ebt.p('r3');
    const a1f1 = this.at.p('D1'), a1f2 = this.at.p('D2'), a1f3 = this.at.p('D3'), a1Theta = this.at.p('theta1');
    const a2f1 = this.at.p('E1'), a2f2 = this.at.p('E2'), a2f3 = this.at.p('E3'), a2Theta = this.at.p('theta2');
    const aatK = this.aat.p('k'), aatT1 = this.aat.p('theta1'), aatT2 = this.aat.p('theta2');
    const bbD = this.bb13.p('D'), bbAlpha = this.bb13.p('alpha'), bbR1 = this.bb13.p('r1'), bbR3 = this.bb13.p('r3');

    const fabcd = [v3(), v3(), v3(), v3()];
    const dphidr = [v3(), v3(), v3(), v3()];
    const dcosphidr = [v3(), v3(), v3(), v3()];
    const dbonddr = [[v3(), v3(), v3(), v3()], [v3(), v3(), v3(), v3()], [v3(), v3(), v3(), v3()]];
    const dthetadr = [[v3(), v3(), v3(), v3()], [v3(), v3(), v3(), v3()]];
    const rel = new Array(12).fill(0);
    const fk = new Array(12).fill(0);
    const vb1 = v3(), vb2 = v3(), vb3 = v3();

    for (let n = 0; n < D.n; n++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, D.atoms[4 * n + w], 'dihedral'));
      const [i1, i2, i3, i4] = atoms;
      const type = D.type[n];
      // the bond vectors of the C++ compute: vb1 = x1 - x2, vb2 = x3 - x2, vb3 = x4 - x3 (nearest copies to atom 2)
      const pI = v3(), pK = v3(), pL = v3();
      delta(bc, i2, i1, pI);
      delta(bc, i2, i3, pK);
      delta(bc, i2, i4, pL);
      for (let d = 0; d < 3; d++) {
        vb1[d] = pI[d];
        vb2[d] = pK[d];
        vb3[d] = pL[d] - pK[d];
      }
      const [vb1x, vb1y, vb1z] = vb1;
      const [vb2x, vb2y, vb2z] = vb2;
      const [vb3x, vb3y, vb3z] = vb3;
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

      // cos and sin of the two angles, and the cosine of phi
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

      // Dihedral::problem: a cosine outside [-1 - TOLERANCE, 1 + TOLERANCE] is a warning, and the term is still evaluated
      if ((c > 1.0 + TOLERANCE || c < -1.0 - TOLERANCE) && bc.warn) {
        bc.warn(`Dihedral problem: ${D.atoms[4 * n]} ${D.atoms[4 * n + 1]} ${D.atoms[4 * n + 2]} ${D.atoms[4 * n + 3]}`);
      }
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      const cosphi = c;
      let phi = Math.acos(c);

      let sinphi = Math.sqrt(1.0 - c * c);
      sinphi = Math.max(sinphi, SMALL);

      // n123 = vb1 x vb2; the sign of phi follows n123 . vb3
      const n123x = vb1y * vb2z - vb1z * vb2y;
      const n123y = vb1z * vb2x - vb1x * vb2z;
      const n123z = vb1x * vb2y - vb1y * vb2x;
      const n123DotVb3 = n123x * vb3x + n123y * vb3y + n123z * vb3z;
      if (n123DotVb3 > 0.0) {
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

      // d(cos phi)/dr and dphi/dr
      dcosphidr[0][0] = -sx1; dcosphidr[0][1] = -sy1; dcosphidr[0][2] = -sz1;
      dcosphidr[1][0] = sx2 + sx1; dcosphidr[1][1] = sy2 + sy1; dcosphidr[1][2] = sz2 + sz1;
      dcosphidr[2][0] = sx12 - sx2; dcosphidr[2][1] = sy12 - sy2; dcosphidr[2][2] = sz12 - sz2;
      dcosphidr[3][0] = -sx12; dcosphidr[3][1] = -sy12; dcosphidr[3][2] = -sz12;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) dphidr[i][j] = -dcosphidr[i][j] / sinphi;

      // the dihedral energy and its torsion forces
      const dphi1 = phi - PH[0][type];
      const dphi2 = 2.0 * phi - PH[1][type];
      const dphi3 = 3.0 * phi - PH[2][type];
      let edihedral = K[0][type] * (1.0 - libmCos(dphi1)) + K[1][type] * (1.0 - libmCos(dphi2)) + K[2][type] * (1.0 - libmCos(dphi3));
      const deDihedral = K[0][type] * libmSin(dphi1) + 2.0 * K[1][type] * libmSin(dphi2) + 3.0 * K[2][type] * libmSin(dphi3);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) fabcd[i][j] = deDihedral * dphidr[i][j];

      // d(bond)/dr: dbonddr[bond][atom][coordinate]
      for (let b = 0; b < 3; b++) for (let a = 0; a < 4; a++) for (let k = 0; k < 3; k++) dbonddr[b][a][k] = 0.0;
      const vb = [vb1, vb2, vb3], rr = [r1, r2, r3];
      // bond b joins atoms b and b+1 (0-based): bond 1 = atoms 0-1, bond 2 = atoms 1-2, bond 3 = atoms 2-3
      for (let b = 0; b < 3; b++) {
        for (let k = 0; k < 3; k++) {
          dbonddr[b][b][k] = vb[b][k] / rr[b];
          dbonddr[b][b + 1][k] = -vb[b][k] / rr[b];
        }
      }

      // d(theta)/dr: dthetadr[angle][atom][coordinate]
      for (let a = 0; a < 2; a++) for (let j = 0; j < 4; j++) for (let k = 0; k < 3; k++) dthetadr[a][j][k] = 0.0;
      const t1 = costh12 / r1mag2;
      const t2 = costh23 / r2mag2;
      const t3 = costh12 / r2mag2;
      const t4 = costh23 / r3mag2;
      for (let k = 0; k < 3; k++) {
        // angle 12
        dthetadr[0][0][k] = sc1 * ((t1 * vb1[k]) - (vb2[k] * r12c1));
        dthetadr[0][1][k] = sc1 * ((-t1 * vb1[k]) + (vb2[k] * r12c1) + (-t3 * vb2[k]) + (vb1[k] * r12c1));
        dthetadr[0][2][k] = sc1 * ((t3 * vb2[k]) - (vb1[k] * r12c1));
        // angle 23
        dthetadr[1][1][k] = sc2 * ((t2 * vb2[k]) + (vb3[k] * r12c2));
        dthetadr[1][2][k] = sc2 * ((-t2 * vb2[k]) - (vb3[k] * r12c2) + (t4 * vb3[k]) + (vb2[k] * r12c2));
        dthetadr[1][3][k] = -sc2 * ((t4 * vb3[k]) + (vb2[k] * r12c2));
      }

      const cos2phi = libmCos(2.0 * phi);
      const cos3phi = libmCos(3.0 * phi);

      // middle bond / torsion coupling (energy and force on bond 2)
      let bt1 = mf1[type] * cosphi;
      let bt2 = mf2[type] * cos2phi;
      let bt3 = mf3[type] * cos3phi;
      let sumbte = bt1 + bt2 + bt3;
      const mbtDr2 = r2 - mR0[type];
      const mbtRalpha2 = Math.exp(-mAlpha2[type] * mbtDr2);
      let db = 1 - mbtRalpha2;
      edihedral += db * sumbte;
      bt1 = -mf1[type] * sinphi;
      bt2 = -2.0 * mf2[type] * libmSin(2.0 * phi);
      bt3 = -3.0 * mf3[type] * libmSin(3.0 * phi);
      let sumbtf = bt1 + bt2 + bt3;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] += db * sumbtf * dphidr[i][j] + sumbte * (mAlpha2[type] * mbtRalpha2) * dbonddr[1][i][j];
      }

      // end bond / torsion coupling: bond 1 (first bond)
      bt1 = e1f1[type] * cosphi;
      bt2 = e1f2[type] * cos2phi;
      bt3 = e1f3[type] * cos3phi;
      sumbte = bt1 + bt2 + bt3;
      const ebtDr1 = r1 - e1R0[type];
      const ebtRalpha1 = Math.exp(-e1Alpha[type] * ebtDr1);
      db = 1 - ebtRalpha1;
      edihedral += db * (bt1 + bt2 + bt3);
      bt1 = e1f1[type] * sinphi;
      bt2 = 2.0 * e1f2[type] * libmSin(2.0 * phi);
      bt3 = 3.0 * e1f3[type] * libmSin(3.0 * phi);
      sumbtf = bt1 + bt2 + bt3;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] -= db * sumbtf * dphidr[i][j] + sumbte * (e1Alpha[type] * ebtRalpha1) * dbonddr[0][i][j];
      }

      // end bond / torsion coupling: bond 3 (last bond)
      bt1 = e2f1[type] * cosphi;
      bt2 = e2f2[type] * cos2phi;
      bt3 = e2f3[type] * cos3phi;
      sumbte = bt1 + bt2 + bt3;
      const ebtDr3 = r3 - e2R0[type];
      const ebtRalpha3 = Math.exp(-e2Alpha[type] * ebtDr3);
      db = 1 - ebtRalpha3;
      edihedral += db * (bt1 + bt2 + bt3);
      bt1 = -e2f1[type] * sinphi;
      bt2 = -2.0 * e2f2[type] * libmSin(2.0 * phi);
      bt3 = -3.0 * e2f3[type] * libmSin(3.0 * phi);
      sumbtf = bt1 + bt2 + bt3;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] += db * sumbtf * dphidr[i][j] + sumbte * (e2Alpha[type] * ebtRalpha3) * dbonddr[2][i][j];
      }

      // angle / torsion coupling: angle 1
      let at1 = a1f1[type] * cosphi;
      let at2 = a1f2[type] * cos2phi;
      let at3 = a1f3[type] * cos3phi;
      sumbte = at1 + at2 + at3;
      let da = Math.acos(costh12) - a1Theta[type];
      edihedral += da * (at1 + at2 + at3);
      bt1 = a1f1[type] * sinphi;
      bt2 = 2.0 * a1f2[type] * libmSin(2.0 * phi);
      bt3 = 3.0 * a1f3[type] * libmSin(3.0 * phi);
      sumbtf = bt1 + bt2 + bt3;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] -= da * sumbtf * dphidr[i][j] + sumbte * dthetadr[0][i][j];
      }

      // angle / torsion coupling: angle 2
      at1 = a2f1[type] * cosphi;
      at2 = a2f2[type] * cos2phi;
      at3 = a2f3[type] * cos3phi;
      sumbte = at1 + at2 + at3;
      da = Math.acos(costh23) - a2Theta[type];
      edihedral += da * (at1 + at2 + at3);
      bt1 = -a2f1[type] * sinphi;
      bt2 = -2.0 * a2f2[type] * libmSin(2.0 * phi);
      bt3 = -3.0 * a2f3[type] * libmSin(3.0 * phi);
      sumbtf = bt1 + bt2 + bt3;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] += da * sumbtf * dphidr[i][j] + sumbte * dthetadr[1][i][j];
      }

      // angle / angle / torsion coupling
      const da1 = Math.acos(costh12) - aatT1[type];
      const da2 = Math.acos(costh23) - aatT2[type];
      edihedral += aatK[type] * da1 * da2 * cosphi;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
        fabcd[i][j] -= aatK[type] * (cosphi * (da2 * dthetadr[0][i][j] - da1 * dthetadr[1][i][j]) + sinphi * da1 * da2 * dphidr[i][j]);
      }

      // bond 1 / bond 3 coupling (BondBond13)
      if (Math.abs(bbD[type]) > SMALL) {
        const dr1 = r1 - bbR1[type];
        const dr2 = r3 - bbR3[type];
        const bb13Ralpha1 = Math.exp(-bbAlpha[type] * dr1);
        const bb13Ralpha2 = Math.exp(-bbAlpha[type] * dr2);
        const tk1 = -bbD[type] * bbAlpha[type] * bb13Ralpha2 * (1 - bb13Ralpha1) / r3;
        const tk2 = -bbD[type] * bbAlpha[type] * bb13Ralpha1 * (1 - bb13Ralpha2) / r1;
        edihedral += bbD[type] * (1 - bb13Ralpha1) * (1 - bb13Ralpha2);
        for (let k = 0; k < 3; k++) {
          fabcd[0][k] += tk2 * vb1[k];
          fabcd[1][k] -= tk2 * vb1[k];
          fabcd[2][k] -= tk1 * vb3[k];
          fabcd[3][k] += tk1 * vb3[k];
        }
      }

      // the four forces, the bond-vector positions (relative to atom 1) for the virial
      for (let a = 0; a < 4; a++) for (let d = 0; d < 3; d++) {
        fk[3 * a + d] = fabcd[a][d];
        bc.f[3 * atoms[a] + d] += fabcd[a][d];
      }
      rel.fill(0);
      for (let d = 0; d < 3; d++) {
        rel[3 + d] = -vb1[d];
        rel[6 + d] = -vb1[d] + vb2[d];
        rel[9 + d] = -vb1[d] + vb2[d] + vb3[d];
      }
      bondedVirial(bc, atoms, rel, fk, edihedral);
      bc.acc.edihed += edihedral;
    }
  }
}
