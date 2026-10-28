/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-MOLECULE/improper_ring.cpp and improper_ring.h at commit b332f28 (the LAMMPS developers; Copyright
 * (2003) Sandia Corporation, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * improper_style ring (docs.lammps.org/improper_ring.html): E = (1/6) K (D_ijl + D_ijk + D_kjl)^6 with
 * D = cos(theta_abc) - cos(theta0) for the three angles of the quadruplet (i, j, k, l) with j, the second atom, the
 * atom of symmetry: "all other atoms are considered interchangeable". Coefficients: "K theta0" (theta0 in degrees).
 * The sums are native's: the three angle terms, the bend cosines clamped by SMALL = 0.001, and the powers from
 * MathSpecial::powint. The bond vectors of the three angles are taken from the copies of i, k and l nearest to j
 * (the owner of the improper). Native writedata = 0: no Impropers Coeffs section.
 */
import { Bonded, StyleError, typeBounds, type BondedAdaptAccess, type BondedCompute, type StyleContext, NoSettingsError } from '../../engine/force/types';
import { atomIndex, delta, vectorAccess } from '../../engine/force/bonded_util';
import { parseNum } from '../../engine/force/util';
import { cosFma as cos } from '../libm/sincos_fma';
import { evTallyFour } from './ev';

// LAMMPS: static constexpr double SMALL = 0.001 (improper_ring.cpp)
const SMALL = 0.001;
// MathConst::MY_PI (math_const.h)
const MY_PI = 3.14159265358979323846;

/** MathSpecial::powint (math_special.h): integer power by repeated squaring, the native rounding. */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

export class ImproperRing extends Bonded {
  readonly name = 'ring';
  /** fix adapt: as native extract() does (src/EXTRA-MOLECULE/improper_ring.cpp at b332f28), k and "theta0", which is chi = cos(theta0). */
  override adaptParam(name: string): BondedAdaptAccess | null {
    return vectorAccess({ k: this.k, theta0: this.chi }, name);
  }
  readonly kind = 'improper' as const;
  private k!: Float64Array;
  private chi!: Float64Array;
  private set!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new NoSettingsError(args[0]);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.k = new Float64Array(ntypes + 1);
    this.chi = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
  }

  /** improper_ring.cpp coeff(): narg must be 3 ("Incorrect args for improper coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for improper coefficients');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const k_one = parseNum(args[1], 'K');
    const chi_one = parseNum(args[2], 'theta0');
    for (let i = lo; i <= hi; i++) {
      this.k[i] = k_one;
      // "Convert it to radians and store its cosine": chi = cos((chi_one/180.0)*MY_PI)
      this.chi[i] = cos((chi_one / 180.0) * MY_PI);
      this.set[i] = 1;
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`All improper coeffs are not set (type ${t})`);
    }
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const f = bc.f;
    // positions of the four atoms relative to the owner i2 (the copy of each atom nearest to it); p2 is the owner
    const p1 = [0, 0, 0], p2 = [0, 0, 0], p3 = [0, 0, 0], p4 = [0, 0, 0];
    const P = [p1, p2, p3, p4];
    const vb1 = [0, 0, 0], vb2 = [0, 0, 0], vb3 = [0, 0, 0];
    const f1 = [0, 0, 0], f3 = [0, 0, 0], f4 = [0, 0, 0];
    const bvec1 = [0, 0, 0], bvec2 = [0, 0, 0];
    const bvec1n = [0, 0, 0], bvec2n = [0, 0, 0], bend = [0, 0, 0];
    const fi = [0, 0, 0], fj = [0, 0, 0], fk = [0, 0, 0];
    for (let n = 0; n < I.n; n++) {
      const i1 = atomIndex(bc, I.atoms[4 * n], 'improper');
      const i2 = atomIndex(bc, I.atoms[4 * n + 1], 'improper');
      const i3 = atomIndex(bc, I.atoms[4 * n + 2], 'improper');
      const i4 = atomIndex(bc, I.atoms[4 * n + 3], 'improper');
      const type = I.type[n];
      delta(bc, i2, i1, p1);
      p2[0] = 0; p2[1] = 0; p2[2] = 0;
      delta(bc, i2, i3, p3);
      delta(bc, i2, i4, p4);
      // vb1 = x1 - x2, vb2 = x3 - x2, vb3 = x4 - x3 (native improper_ring.cpp, the virial vectors)
      for (let c = 0; c < 3; c++) {
        vb1[c] = p1[c] - p2[c];
        vb2[c] = p3[c] - p2[c];
        vb3[c] = p4[c] - p3[c];
      }
      // the three triads (at1, at2, at3): 1-2-9 with atoms (i1, i2, i4), 1-2-3 with (i1, i2, i3), 9-2-3 with (i4, i2, i3)
      const at1 = [i1, i1, i4], at2 = [i2, i2, i2], at3 = [i4, i3, i3];
      let angle_summer = 0.0;
      for (let icomb = 0; icomb < 3; icomb++) {
        // bvec1 = x[at2] - x[at1], bvec2 = x[at3] - x[at2]
        const P1 = P[indexOf(at1[icomb], i1, i2, i3, i4)], P2 = P[indexOf(at2[icomb], i1, i2, i3, i4)];
        const P3 = P[indexOf(at3[icomb], i1, i2, i3, i4)];
        for (let c = 0; c < 3; c++) {
          bvec1[c] = P2[c] - P1[c];
          bvec2[c] = P3[c] - P2[c];
        }
        bvec1n[icomb] = Math.sqrt(bvec1[0] * bvec1[0] + bvec1[1] * bvec1[1] + bvec1[2] * bvec1[2]);
        bvec2n[icomb] = Math.sqrt(bvec2[0] * bvec2[0] + bvec2[1] * bvec2[1] + bvec2[2] * bvec2[2]);
        let b = bvec2[0] * bvec1[0] + bvec2[1] * bvec1[1] + bvec2[2] * bvec1[2];
        b /= bvec1n[icomb] * bvec2n[icomb];
        if (b > 1.0) b -= SMALL;
        if (b < -1.0) b += SMALL;
        bend[icomb] = b;
        angle_summer += bend[icomb] - this.chi[type];
      }
      const eimproper = (1.0 / 6.0) * this.k[type] * powint(angle_summer, 6);
      const angfac = this.k[type] * powint(angle_summer, 5);
      f1[0] = 0.0; f1[1] = 0.0; f1[2] = 0.0;
      f3[0] = 0.0; f3[1] = 0.0; f3[2] = 0.0;
      f4[0] = 0.0; f4[1] = 0.0; f4[2] = 0.0;
      for (let icomb = 0; icomb < 3; icomb++) {
        const P1 = P[indexOf(at1[icomb], i1, i2, i3, i4)], P2 = P[indexOf(at2[icomb], i1, i2, i3, i4)];
        const P3 = P[indexOf(at3[icomb], i1, i2, i3, i4)];
        for (let c = 0; c < 3; c++) {
          bvec1[c] = P2[c] - P1[c];
          bvec2[c] = P3[c] - P2[c];
        }
        const cjiji = bvec1n[icomb] * bvec1n[icomb];
        const ckjkj = bvec2n[icomb] * bvec2n[icomb];
        const ckjji = bvec2[0] * bvec1[0] + bvec2[1] * bvec1[1] + bvec2[2] * bvec1[2];
        const cfact1 = angfac / Math.sqrt(ckjkj * cjiji);
        const cfact2 = ckjji / ckjkj;
        const cfact3 = ckjji / cjiji;
        for (let c = 0; c < 3; c++) {
          fk[c] = cfact2 * bvec2[c] - bvec1[c];
          fi[c] = bvec2[c] - cfact3 * bvec1[c];
        }
        for (let c = 0; c < 3; c++) {
          fj[c] = -fi[c] - fk[c];
        }
        for (let c = 0; c < 3; c++) {
          fi[c] *= cfact1;
          fj[c] *= cfact1;
          fk[c] *= cfact1;
        }
        // forces on atoms 1, 3 and 4 for the virial (improper_ring.cpp: at == i1, i3, i4)
        const a1 = at1[icomb], a2 = at2[icomb], a3 = at3[icomb];
        for (let c = 0; c < 3; c++) {
          if (a1 === i1) f1[c] += fi[c]; else if (a2 === i1) f1[c] += fj[c]; else if (a3 === i1) f1[c] += fk[c];
          if (a1 === i3) f3[c] += fi[c]; else if (a2 === i3) f3[c] += fj[c]; else if (a3 === i3) f3[c] += fk[c];
          if (a1 === i4) f4[c] += fi[c]; else if (a2 === i4) f4[c] += fj[c]; else if (a3 === i4) f4[c] += fk[c];
        }
        for (let c = 0; c < 3; c++) {
          f[3 * a1 + c] += fi[c];
          f[3 * a2 + c] += fj[c];
          f[3 * a3 + c] += fk[c];
        }
      }
      evTallyFour(bc, [i1, i2, i3, i4], 'imp', eimproper, f1, f3, f4, vb1, vb2, vb3);
    }
  }
}

/** Position of atom index a among the four atoms of the quadruplet (1, 2, 3, 4), or 1 for the owner. */
const indexOf = (a: number, i1: number, i2: number, i3: number, i4: number): number => {
  if (a === i1) return 0;
  if (a === i2) return 1;
  if (a === i3) return 2;
  if (a === i4) return 3;
  throw new StyleError('improper ring: atom outside the quadruplet');
};
