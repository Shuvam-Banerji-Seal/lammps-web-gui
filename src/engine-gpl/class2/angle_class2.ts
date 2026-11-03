/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/angle_class2.cpp (AngleClass2::compute) at commit b332f28. Contributing author: Eric Simon
 * (Cray). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * angle_style class2 (docs.lammps.org/angle_class2.html): the angle, bond-bond and bond-angle terms of the C++
 * compute in its own operation order. The coefficients, the data file sections and the restart record are those of
 * the engine's AngleClass2 (src/engine/force/angle/class2.ts), which this class extends. The bond vectors are the
 * copies nearest to the centre atom (bonded_util.ts delta, native's closest_image in the angle list), acos is the
 * glibc port of src/engine-gpl/libm, and the tally is native's Angle::ev_tally (src/engine-gpl/xmolecule/ev.ts).
 */
import { AngleClass2 } from '../../engine/force/angle/class2';
import { mapIndex, delta } from '../../engine/force/bonded_util';
import type { BondedCompute } from '../../engine/force/types';
import { acos } from '../libm/acos';
import { evTallyAngle } from '../xmolecule/ev';

/** angle_class2.cpp: "static constexpr double SMALL = 0.001;" */
const SMALL = 0.001;

export class AngleClass2Native extends AngleClass2 {
  override compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const map = bc.map;
    const theta0 = this.th0, k2 = this.K2, k3 = this.K3, k4 = this.K4;
    const bb_k = this.M, bb_r1 = this.r1bb, bb_r2 = this.r2bb;
    const ba_k1 = this.N1, ba_k2 = this.N2, ba_r1 = this.r1ba, ba_r2 = this.r2ba;
    // delx1, dely1, delz1 and delx2, dely2, delz2 (doubles from the start, so the stores keep one element kind)
    const d1 = [0.5, 0.5, 0.5], d2 = [0.5, 0.5, 0.5];
    const f1 = [0.5, 0.5, 0.5], f3 = [0.5, 0.5, 0.5];
    const f = bc.f;

    for (let n = 0; n < A.n; n++) {
      const i1 = mapIndex(map, A.atoms[3 * n], 'angle');
      const i2 = mapIndex(map, A.atoms[3 * n + 1], 'angle');
      const i3 = mapIndex(map, A.atoms[3 * n + 2], 'angle');
      const type = A.type[n];

      // 1st bond

      delta(bc, i2, i1, d1);
      const delx1 = d1[0], dely1 = d1[1], delz1 = d1[2];

      const rsq1 = delx1 * delx1 + dely1 * dely1 + delz1 * delz1;
      const r1 = Math.sqrt(rsq1);

      // 2nd bond

      delta(bc, i2, i3, d2);
      const delx2 = d2[0], dely2 = d2[1], delz2 = d2[2];

      const rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;
      const r2 = Math.sqrt(rsq2);

      // angle (cos and sin)

      let c = delx1 * delx2 + dely1 * dely2 + delz1 * delz2;
      c /= r1 * r2;

      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;

      let s = Math.sqrt(1.0 - c * c);
      if (s < SMALL) s = SMALL;
      s = 1.0 / s;

      // force & energy for angle term

      const dtheta = acos(c) - theta0[type];
      const dtheta2 = dtheta * dtheta;
      const dtheta3 = dtheta2 * dtheta;
      const dtheta4 = dtheta3 * dtheta;

      const de_angle = 2.0 * k2[type] * dtheta + 3.0 * k3[type] * dtheta2 + 4.0 * k4[type] * dtheta3;

      const a = -de_angle * s;
      const a11 = a * c / rsq1;
      const a12 = -a / (r1 * r2);
      const a22 = a * c / rsq2;

      let f10 = a11 * delx1 + a12 * delx2;
      let f11 = a11 * dely1 + a12 * dely2;
      let f12 = a11 * delz1 + a12 * delz2;

      let f30 = a22 * delx2 + a12 * delx1;
      let f31 = a22 * dely2 + a12 * dely1;
      let f32 = a22 * delz2 + a12 * delz1;

      let eangle = k2[type] * dtheta2 + k3[type] * dtheta3 + k4[type] * dtheta4;

      // force & energy for bond-bond term

      let dr1 = r1 - bb_r1[type];
      let dr2 = r2 - bb_r2[type];
      const tk1 = bb_k[type] * dr1;
      const tk2 = bb_k[type] * dr2;

      f10 -= delx1 * tk2 / r1;
      f11 -= dely1 * tk2 / r1;
      f12 -= delz1 * tk2 / r1;

      f30 -= delx2 * tk1 / r2;
      f31 -= dely2 * tk1 / r2;
      f32 -= delz2 * tk1 / r2;

      eangle += bb_k[type] * dr1 * dr2;

      // force & energy for bond-angle term

      dr1 = r1 - ba_r1[type];
      dr2 = r2 - ba_r2[type];
      const aa1 = s * dr1 * ba_k1[type];
      const aa2 = s * dr2 * ba_k2[type];

      let aa11 = aa1 * c / rsq1;
      const aa12 = -aa1 / (r1 * r2);
      let aa21 = aa2 * c / rsq1;
      const aa22 = -aa2 / (r1 * r2);

      const vx11 = (aa11 * delx1) + (aa12 * delx2);
      const vx12 = (aa21 * delx1) + (aa22 * delx2);
      const vy11 = (aa11 * dely1) + (aa12 * dely2);
      const vy12 = (aa21 * dely1) + (aa22 * dely2);
      const vz11 = (aa11 * delz1) + (aa12 * delz2);
      const vz12 = (aa21 * delz1) + (aa22 * delz2);

      aa11 = aa1 * c / rsq2;
      aa21 = aa2 * c / rsq2;

      const vx21 = (aa11 * delx2) + (aa12 * delx1);
      const vx22 = (aa21 * delx2) + (aa22 * delx1);
      const vy21 = (aa11 * dely2) + (aa12 * dely1);
      const vy22 = (aa21 * dely2) + (aa22 * dely1);
      const vz21 = (aa11 * delz2) + (aa12 * delz1);
      const vz22 = (aa21 * delz2) + (aa22 * delz1);

      const b1 = ba_k1[type] * dtheta / r1;
      const b2 = ba_k2[type] * dtheta / r2;

      f10 -= vx11 + b1 * delx1 + vx12;
      f11 -= vy11 + b1 * dely1 + vy12;
      f12 -= vz11 + b1 * delz1 + vz12;

      f30 -= vx21 + b2 * delx2 + vx22;
      f31 -= vy21 + b2 * dely2 + vy22;
      f32 -= vz21 + b2 * delz2 + vz22;

      eangle += ba_k1[type] * dr1 * dtheta + ba_k2[type] * dr2 * dtheta;

      // apply force to each of 3 atoms

      f[3 * i1] += f10;
      f[3 * i1 + 1] += f11;
      f[3 * i1 + 2] += f12;

      f[3 * i2] -= f10 + f30;
      f[3 * i2 + 1] -= f11 + f31;
      f[3 * i2 + 2] -= f12 + f32;

      f[3 * i3] += f30;
      f[3 * i3 + 1] += f31;
      f[3 * i3 + 2] += f32;

      f1[0] = f10; f1[1] = f11; f1[2] = f12;
      f3[0] = f30; f3[1] = f31; f3[2] = f32;
      evTallyAngle(bc, i1, i2, i3, eangle, f1, f3, d1, d2);
    }
  }
}
