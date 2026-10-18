/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-MOLECULE/dihedral_table_cut.cpp
 * and dihedral_table_cut.h (DihedralTableCut: coeff, compute, allocate; the table itself is DihedralTable's). Contributing
 * author: K. Michael Salerno (NRL), based on the tabulated dihedral by Andrew Jewett. Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dihedral_table.html (the table/cut section): "dihedral_style table/cut", with the coefficients
 * "dihedral_coeff N k theta1 theta2 file keyword": an angle-angle-torsion cutoff function (k, theta1 and theta2 in
 * degrees) multiplies the tabulated torsion. Reference: Salerno and Bernstein, J Chem Theory Comput 14, 2018,
 * https://doi.org/10.1021/acs.jctc.7b01229.
 *
 * compute() takes its bond vectors from the copies nearest to the second atom (the raw differences of the
 * neighbour-list images in C++), without a minimum image of the torsion vectors.
 */
import { StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { DihedralTable, numArg, tallyDihedral } from './dihedral_table';
import { fmtG } from './tablefile';

const MY_PI = 3.14159265358979323846;
const MY_2PI = 6.28318530717958647692;
const TOLERANCE = 0.05;
const SMALL = 0.0000001;

export class DihedralTableCut extends DihedralTable {
  override readonly name: string = 'table/cut';
  private aatK = new Float64Array(0);
  private aatTheta1 = new Float64Array(0);
  private aatTheta2 = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.aatK = new Float64Array(ntypes + 1);
    this.aatTheta1 = new Float64Array(ntypes + 1);
    this.aatTheta2 = new Float64Array(ntypes + 1);
  }

  /** dihedral_coeff N-range k theta1 theta2 file keyword (DihedralTableCut::coeff). */
  override coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 7) throw new StyleError('Incorrect args for dihedral coefficients\nFor more information see https://docs.lammps.org/err0021');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    // docs.lammps.org/dihedral_table.html: "style (= aat)" is the word after the type: dihedral_coeff 1 aat 1.0 177 180 file.table DIH_TABLE1
    if (args[1] !== 'aat') throw new StyleError(`dihedral_coeff table/cut: the style word after the type must be aat, got '${args[1]}'`);
    const kOne = numArg(args[2]);
    const theta01One = numArg(args[3]);
    const theta02One = numArg(args[4]);
    // convert theta0's from degrees to radians
    for (let i = ilo; i <= ihi; i++) {
      this.aatK[i] = kOne;
      this.aatTheta1[i] = (theta01One / 180.0) * MY_PI;
      this.aatTheta2[i] = (theta02One / 180.0) * MY_PI;
    }
    this.addTable(ilo, ihi, args[5], args[6], ctx, true);
  }

  /** The angle-angle-torsion cutoff function is the term of compute() below; the native errors name the file. */
  override compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const f = bc.f;
    for (let n = 0; n < D.n; n++) {
      const i1 = atomIndex(bc, D.atoms[4 * n], 'dihedral');
      const i2 = atomIndex(bc, D.atoms[4 * n + 1], 'dihedral');
      const i3 = atomIndex(bc, D.atoms[4 * n + 2], 'dihedral');
      const i4 = atomIndex(bc, D.atoms[4 * n + 3], 'dihedral');
      const type = D.type[n];
      // 1st bond: x[i1] - x[i2]; 2nd bond: x[i3] - x[i2]; 3rd bond: x[i4] - x[i3]
      // each bond is the raw difference of the two positions (the rounding of the native difference is kept)
      const p1 = [0, 0, 0], p2 = [0, 0, 0], p3 = [0, 0, 0];
      delta(bc, i2, i1, p1);
      delta(bc, i2, i3, p2);
      delta(bc, i3, i4, p3);
      const vb1x = p1[0], vb1y = p1[1], vb1z = p1[2];
      const vb2x = p2[0], vb2y = p2[1], vb2z = p2[2];
      const vb2xm = -vb2x, vb2ym = -vb2y, vb2zm = -vb2z;
      const vb3x = p3[0], vb3y = p3[1], vb3z = p3[2];
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
      const c0 = (vb1x * vb3x + vb1y * vb3y + vb1z * vb3z) * rb1 * rb3;
      // angles
      const r12c1 = rb1 * rb2;
      const r12c2 = rb2 * rb3;
      const costh12 = (vb1x * vb2x + vb1y * vb2y + vb1z * vb2z) * r12c1;
      const costh13 = c0;
      const costh23 = (vb2xm * vb3x + vb2ym * vb3y + vb2zm * vb3z) * r12c2;
      // cos and sin of 2 angles and final c
      let sin2 = maxZero(1.0 - costh12 * costh12);
      let sc1 = Math.sqrt(sin2);
      if (sc1 < SMALL) sc1 = SMALL;
      sc1 = 1.0 / sc1;
      sin2 = maxZero(1.0 - costh23 * costh23);
      let sc2 = Math.sqrt(sin2);
      if (sc2 < SMALL) sc2 = SMALL;
      sc2 = 1.0 / sc2;
      const s1 = sc1 * sc1;
      const s2 = sc2 * sc2;
      const s12 = sc1 * sc2;
      let c = (c0 + costh12 * costh23) * s12;
      // error check
      if (c > 1.0 + TOLERANCE || c < -1.0 - TOLERANCE) this.problem(bc, [D.atoms[4 * n], D.atoms[4 * n + 1], D.atoms[4 * n + 2], D.atoms[4 * n + 3]], [i1, i2, i3, i4]);
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      let phil = Math.acos(c);
      let sinphi = Math.sqrt(1.0 - c * c);
      sinphi = sinphi > SMALL ? sinphi : SMALL;
      // n123 = vb1 x vb2
      const n123x = vb1y * vb2z - vb1z * vb2y;
      const n123y = vb1z * vb2x - vb1x * vb2z;
      const n123z = vb1x * vb2y - vb1y * vb2x;
      const n123DotVb3 = n123x * vb3x + n123y * vb3y + n123z * vb3z;
      if (n123DotVb3 > 0.0) {
        phil = -phil;
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
      // d(cos(phi))/d(r) and dphi/dr arrays
      const dcosphidr = [
        [-sx1, -sy1, -sz1],
        [sx2 + sx1, sy2 + sy1, sz2 + sz1],
        [sx12 - sx2, sy12 - sy2, sz12 - sz2],
        [-sx12, -sy12, -sz12],
      ];
      const dphidr: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) dphidr[i][j] = -dcosphidr[i][j] / sinphi;
      const fabcd: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
      // d(theta)/d(r): dthetadr[angle][atom][coordinate]
      const dthetadr: number[][][] = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      ];
      const t1 = costh12 / r1mag2;
      const t2 = costh23 / r2mag2;
      const t3 = costh12 / r2mag2;
      const t4 = costh23 / r3mag2;
      const vb1 = [vb1x, vb1y, vb1z], vb2 = [vb2x, vb2y, vb2z], vb3 = [vb3x, vb3y, vb3z];
      for (let k = 0; k < 3; k++) {
        // angle12
        dthetadr[0][0][k] = sc1 * ((t1 * vb1[k]) - (vb2[k] * r12c1));
        dthetadr[0][1][k] = sc1 * ((-t1 * vb1[k]) + (vb2[k] * r12c1) + (-t3 * vb2[k]) + (vb1[k] * r12c1));
        dthetadr[0][2][k] = sc1 * ((t3 * vb2[k]) - (vb1[k] * r12c1));
        // angle23
        dthetadr[1][1][k] = sc2 * ((t2 * vb2[k]) + (vb3[k] * r12c2));
        dthetadr[1][2][k] = sc2 * ((-t2 * vb2[k]) - (vb3[k] * r12c2) + (t4 * vb3[k]) + (vb2[k] * r12c2));
        dthetadr[1][3][k] = -sc2 * ((t4 * vb3[k]) + (vb2[k] * r12c2));
      }
      // angle/angle/torsion cutoff
      const k0 = this.aatK[type];
      const theta1 = this.aatTheta1[type];
      const da1 = Math.acos(costh12) - theta1;
      const da2 = Math.acos(costh23) - theta1;
      const dtheta = this.aatTheta2[type] - theta1;
      if (phil < 0) phil += MY_2PI;
      const { u: fphi, f: fpphi } = this.ufLookup(type, phil);
      let gt = k0;
      let gtt = k0;
      let gpt = 0;
      let gptt = 0;
      if (Math.acos(costh12) > theta1) {
        gt *= 1 - da1 * da1 / dtheta / dtheta;
        gpt = -k0 * 2 * da1 / dtheta / dtheta;
      }
      if (Math.acos(costh23) > theta1) {
        gtt *= 1 - da2 * da2 / dtheta / dtheta;
        gptt = -k0 * 2 * da2 / dtheta / dtheta;
      }
      const edihedral = gt * gtt * fphi;
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 3; j++) {
          fabcd[i][j] -= gt * gtt * fpphi * dphidr[i][j]
            - gt * gptt * fphi * dthetadr[1][i][j] + gpt * gtt * fphi * dthetadr[0][i][j];
        }
      }
      // apply force to each of 4 atoms
      f[3 * i1] += fabcd[0][0]; f[3 * i1 + 1] += fabcd[0][1]; f[3 * i1 + 2] += fabcd[0][2];
      f[3 * i2] += fabcd[1][0]; f[3 * i2 + 1] += fabcd[1][1]; f[3 * i2 + 2] += fabcd[1][2];
      f[3 * i3] += fabcd[2][0]; f[3 * i3 + 1] += fabcd[2][1]; f[3 * i3 + 2] += fabcd[2][2];
      f[3 * i4] += fabcd[3][0]; f[3 * i4 + 1] += fabcd[3][1]; f[3 * i4 + 2] += fabcd[3][2];
      tallyDihedral(bc, [i1, i2, i3, i4], edihedral, fabcd[0], fabcd[2], fabcd[3],
        vb1x, vb1y, vb1z, vb2x, vb2y, vb2z, vb3x, vb3y, vb3z);
    }
  }

  /** Dihedral::problem: a warning for a torsion whose cosine is out of range (the run goes on). */
  private problem(bc: BondedCompute, ids: number[], idx: number[]): void {
    const x = bc.s.x;
    const pos = (i: number) => `${fmtG(x[3 * i], 8)} ${fmtG(x[3 * i + 1], 8)} ${fmtG(x[3 * i + 2], 8)}`;
    bc.warn?.(`Dihedral problem: 0 ${bc.s.step} ${ids.join(' ')} (1st atom ${pos(idx[0])}; 2nd atom ${pos(idx[1])}; 3rd atom ${pos(idx[2])}; 4th atom ${pos(idx[3])})`);
  }
}

/** maxZero is the MAX(a, 0.0) of the C++ macro (a > b ? a : b). */
const maxZero = (v: number): number => (v > 0.0 ? v : 0.0);
