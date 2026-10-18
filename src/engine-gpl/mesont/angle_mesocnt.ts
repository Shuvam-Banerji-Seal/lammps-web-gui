/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MESONT/angle_mesocnt.cpp and src/MESONT/angle_mesocnt.h at commit b332f28 (contributing author Philipp
 * Kloza, University of Cambridge). Copyright (2003) Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * angle_style mesocnt: the bending of a carbon nanotube segment pair, harmonic up to the buckling angle thetab and
 * linear beyond it (buckling keyword), E = kh dt^2 for |dt| < thetab, else kb |dt| + thetab (kh thetab - kb), with
 * dt = theta - 180 degrees. The vertex atom is flagged in the per-atom property "buckled" (fix property/atom
 * i_buckled ghost yes, created here as native's init_style does when it is missing); the pair style reads the flags
 * of the chain atoms of the previous evaluation.
 * angle_coeff: "type buckling|harmonic custom kh [kb thetab]" or "type buckling|harmonic C n m l" (the C preset of
 * the C++ coeff(): kh = 0.5 k / l, kb = 0.7 k / (275 ang), thetab = atan(l / (275 ang)) with
 * k = 63.80 r^2.93 eunit ang). A harmonic type sets kb = 0 and thetab = 180 degrees.
 */
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { Bonded, StyleError, bondedVirial, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { parseNum, fmtCoeff } from '../../engine/force/util';
import { ANGSTROM } from '../../engine/force/kspace/ewald';
import type { CustomProp } from '../../engine/types';
import { chiralRadius, eunitOf } from './units';
import { pow as libmPow } from '../libm/pow';
import { acos as libmAcos } from '../libm/acos';

/** LAMMPS: static constexpr double SMALL = 0.001 (angle_mesocnt.cpp). */
const SMALL = 0.001;
/** Name of the per-atom property the angle style keeps (angle_mesocnt_buckled in the C++ init_style). */
export const BUCKLED = 'buckled';

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

export class AngleMesoCNT extends Bonded {
  readonly name = 'mesocnt';
  readonly kind = 'angle' as const;
  private buckling!: Uint8Array;
  private kh!: Float64Array;
  private kb!: Float64Array;
  private thetab!: Float64Array;
  private theta0!: Float64Array;
  private set!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError('angle_style mesocnt takes no arguments');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.buckling = new Uint8Array(ntypes + 1);
    this.kh = new Float64Array(ntypes + 1);
    this.kb = new Float64Array(ntypes + 1);
    this.thetab = new Float64Array(ntypes + 1);
    this.theta0 = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
  }

  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length < 3) throw new StyleError('Illegal angle_coeff command for angle_style mesocnt');
    let buckling: number;
    if (args[1] === 'buckling') buckling = 1;
    else if (args[1] === 'harmonic') buckling = 0;
    else throw new StyleError(`Unknown first argument ${args[1]} for angle coefficients, must be 'buckling' or 'harmonic'`);
    const style = ctx?.s?.units.style;
    if (style === undefined) throw new StyleError('angle_style mesocnt needs the box and units before angle_coeff');
    const eunit = eunitOf(style, 'Angle style mesocnt');
    const ang = ANGSTROM[style];
    let kh: number, kb = 0, thetab = 180;
    if (args[2] === 'custom') {
      if (buckling) {
        if (args.length !== 6) throw new StyleError("Incorrect number of args for 'custom' angle coefficients");
        kb = parseNum(args[4], 'kb');
        thetab = parseNum(args[5], 'thetab');
      } else if (args.length !== 4) throw new StyleError("Incorrect number of args for 'custom' angle coefficients");
      kh = parseNum(args[3], 'kh');
    } else if (args[2] === 'C') {
      if (args.length !== 6) throw new StyleError("Incorrect number of args for 'C' preset in angle coefficients");
      const n = Math.trunc(parseNum(args[3], 'n'));
      const m = Math.trunc(parseNum(args[4], 'm'));
      const l = parseNum(args[5], 'l');
      const rAng = chiralRadius(n, m);
      const k = 63.80 * libmPow(rAng, 2.93) * eunit * ang;
      kh = 0.5 * k / l;
      if (buckling) {
        kb = 0.7 * k / (275.0 * ang);
        thetab = RAD2DEG * Math.atan(l / (275.0 * ang));
      }
    } else throw new StyleError(`Unknown ${args[2]} preset in angle coefficients`);
    if (!buckling) {
      kb = 0.0;
      thetab = 180.0;
    }
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    for (let t = lo; t <= hi; t++) {
      this.buckling[t] = buckling;
      this.kh[t] = kh;
      this.kb[t] = kb;
      this.thetab[t] = DEG2RAD * thetab;
      this.theta0[t] = DEG2RAD * 180.0;
      this.set[t] = 1;
    }
  }

  /** init_style: the per-atom "buckled" property (native adds fix angle_mesocnt_buckled when it is missing). */
  init(ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) if (!this.set[t]) throw new StyleError(`all angle coeffs are not set (type ${t})`);
    const s = ctx?.s;
    if (s && !s.custom.has(BUCKLED)) s.custom.set(BUCKLED, { int: true, cols: 0, data: new Float64Array(s.n) } satisfies CustomProp);
  }

  equilibrium(_type: number): number { return 180.0; }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${this.buckling[t]} ${fmtCoeff(this.kh[t])} ${fmtCoeff(this.kb[t])} ${fmtCoeff(this.thetab[t] * RAD2DEG)}`);
    }
    return out;
  }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const buck = bc.s.custom.get(BUCKLED)?.data ?? null;
    const f = bc.f;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    const f1 = [0, 0, 0], f3 = [0, 0, 0];
    let e = 0;
    for (let n = 0; n < A.n; n++) {
      const i1 = atomIndex(bc, A.atoms[3 * n], 'angle');
      const i2 = atomIndex(bc, A.atoms[3 * n + 1], 'angle');
      const i3 = atomIndex(bc, A.atoms[3 * n + 2], 'angle');
      const type = A.type[n];
      // the two bond vectors from the central atom i2 (copies nearest to i2, as the other angle styles)
      delta(bc, i2, i1, d1);
      delta(bc, i2, i3, d2);
      const rsq1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
      const r1 = Math.sqrt(rsq1);
      const rsq2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
      const r2 = Math.sqrt(rsq2);
      let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      let s = Math.sqrt(1.0 - c * c);
      if (s < SMALL) s = SMALL;
      s = 1.0 / s;
      const dtheta = libmAcos(c) - this.theta0[type];
      let a: number, eangle: number;
      if (!this.buckling[type] || Math.abs(dtheta) < this.thetab[type]) {
        const tk = this.kh[type] * dtheta;
        eangle = tk * dtheta;
        a = -2.0 * tk * s;
        if (buck) buck[i2] = 0;
      } else {
        eangle = this.kb[type] * Math.abs(dtheta) + this.thetab[type] * (this.kh[type] * this.thetab[type] - this.kb[type]);
        a = this.kb[type] * s;
        if (buck) buck[i2] = 1;
      }
      e += eangle;
      const a11 = a * c / rsq1;
      const a12 = -a / (r1 * r2);
      const a22 = a * c / rsq2;
      for (let k = 0; k < 3; k++) {
        f1[k] = a11 * d1[k] + a12 * d2[k];
        f3[k] = a22 * d2[k] + a12 * d1[k];
      }
      // forces on the three atoms (i1 gets f1, i2 gets -(f1 + f3), i3 gets f3)
      for (let k = 0; k < 3; k++) {
        f[3 * i1 + k] += f1[k];
        f[3 * i2 + k] -= f1[k] + f3[k];
        f[3 * i3 + k] += f3[k];
      }
      // virial: positions relative to i2 (the vertex, at the origin of d1 and d2), forces as above
      bondedVirial(bc, [i1, i2, i3], [d1[0], d1[1], d1[2], 0, 0, 0, d2[0], d2[1], d2[2]],
        [f1[0], f1[1], f1[2], -(f1[0] + f3[0]), -(f1[1] + f3[1]), -(f1[2] + f3[2]), f3[0], f3[1], f3[2]], eangle);
    }
    bc.acc.eangle += e;
  }
}
