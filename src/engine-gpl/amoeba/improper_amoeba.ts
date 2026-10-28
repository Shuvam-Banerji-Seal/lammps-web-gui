/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/improper_amoeba.cpp (compute, coeff, init_style, extract) and src/AMOEBA/improper_amoeba.h at
 * commit b332f28, adapted from Tinker eopbend1(). AMOEBA package authors: the LAMMPS developers (Sandia Corporation,
 * GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * improper_style amoeba: the Allinger out-of-plane bending term of an sp2 centre. For the atoms I, J (centre), K, L the
 * angle between the bond J-I and the plane J-K-L is chi (in degrees, asin of the sine); the energy is
 * k chi^2 (1 + c chi + q chi^2 + p chi^3 + s chi^4) with the opbend-cubic/quartic/pentic/sextic weights of the PRM file
 * (improper_amoeba.cpp: "set opbend higher-order term weights from PairAmoeba"). The term is disabled when the pair
 * style's improper flag is off (pair amoeba settings). Degenerate geometry (zero bond or zero cross product) gives no
 * energy and no force, as native.
 * Ported as native: the energy and the derivatives are the same expressions, with the angle in degrees in the
 * prefactors. docs.lammps.org/improper_amoeba.html: "improper_coeff N k" (energy coefficient).
 */
import { Bonded, StyleError, typeBounds, type BondedAdaptAccess, type BondedCompute, type Pair, type StyleContext, NoSettingsError } from '../../engine/force/types';
import { atomIndex, delta, vectorAccess } from '../../engine/force/bonded_util';
import { parseNum } from '../../engine/force/util';

const RAD2DEG = 180.0 / Math.PI;

export class ImproperAmoeba extends Bonded {
  readonly name: string = 'amoeba';
  /** fix adapt: as native extract() does (src/AMOEBA/improper_amoeba.cpp at b332f28), k. */
  override adaptParam(name: string): BondedAdaptAccess | null {
    return vectorAccess({ k: this.k }, name);
  }
  readonly kind = 'improper' as const;
  private k!: Float64Array;
  private setK!: Uint8Array;
  private pairStyle: Pair | null = null;
  private disable = false;
  private cubic = 0;
  private quartic = 0;
  private pentic = 0;
  private sextic = 0;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new NoSettingsError(args[0]);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.k = new Float64Array(ntypes + 1);
    this.setK = new Uint8Array(ntypes + 1);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 2) throw new StyleError('Incorrect args for improper coefficients');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const k = parseNum(args[1], 'k');
    for (let t = lo; t <= hi; t++) {
      this.k[t] = k;
      this.setK[t] = 1;
    }
  }

  /** improper_amoeba.cpp init_style: the pair style supplies the flag and the opbend weights. */
  linkForceField(link: { pair: Pair | null; special: { lj: readonly number[]; coul: readonly number[] } }): void {
    this.pairStyle = link.pair;
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) if (!this.setK[t]) throw new StyleError(`All improper coeffs are not set (type ${t})`);
    const p = this.pairStyle;
    const pe = p as unknown as { extract?(name: string): unknown } | null;
    if (!pe || typeof pe.extract !== 'function' || pe.extract('improper_flag') === undefined) {
      throw new StyleError('Improper amoeba could not find pair amoeba/hippo');
    }
    this.disable = !(pe.extract('improper_flag') as number);
    this.cubic = Number(pe.extract('opbend_cubic') ?? 0);
    this.quartic = Number(pe.extract('opbend_quartic') ?? 0);
    this.pentic = Number(pe.extract('opbend_pentic') ?? 0);
    this.sextic = Number(pe.extract('opbend_sextic') ?? 0);
  }

  compute(bc: BondedCompute): void {
    if (this.disable) return;
    const I = bc.s.topo.impropers;
    const f = bc.f;
    const eprefactor = 1.0 / (RAD2DEG * RAD2DEG);
    const fprefactor = 1.0 / RAD2DEG;
    let e = 0;
    const xab = [0, 0, 0], xcb = [0, 0, 0], xdb = [0, 0, 0];
    for (let q = 0; q < I.n; q++) {
      // native improper_amoeba.cpp: id = list[0], ib = list[1] (the centre), ia = list[2], ic = list[3]
      const id = atomIndex(bc, I.atoms[4 * q], 'improper');
      const ib = atomIndex(bc, I.atoms[4 * q + 1], 'improper');
      const ia = atomIndex(bc, I.atoms[4 * q + 2], 'improper');
      const ic = atomIndex(bc, I.atoms[4 * q + 3], 'improper');
      const t = I.type[q];
      // positions relative to the centre ib (the nearest copies, as the other bonded styles)
      delta(bc, ib, ia, xab);
      delta(bc, ib, ic, xcb);
      delta(bc, ib, id, xdb);
      const xad0 = xab[0] - xdb[0], xad1 = xab[1] - xdb[1], xad2 = xab[2] - xdb[2];
      const xcd0 = xcb[0] - xdb[0], xcd1 = xcb[1] - xdb[1], xcd2 = xcb[2] - xdb[2];
      const rad2 = xad0 * xad0 + xad1 * xad1 + xad2 * xad2;
      const rcd2 = xcd0 * xcd0 + xcd1 * xcd1 + xcd2 * xcd2;
      const dot = xad0 * xcd0 + xad1 * xcd1 + xad2 * xcd2;
      const cc = rad2 * rcd2 - dot * dot;
      const ee = xdb[0] * (xab[1] * xcb[2] - xab[2] * xcb[1]) + xdb[1] * (xab[2] * xcb[0] - xab[0] * xcb[2]) +
        xdb[2] * (xab[0] * xcb[1] - xab[1] * xcb[0]);
      const rdb2 = xdb[0] * xdb[0] + xdb[1] * xdb[1] + xdb[2] * xdb[2];
      if (rdb2 === 0.0 || cc === 0.0) continue;
      let sine = Math.abs(ee) / Math.sqrt(cc * rdb2);
      sine = Math.min(1.0, sine);
      const angle = RAD2DEG * Math.asin(sine);
      const dt = angle, dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
      const kt = this.k[t];
      const eq = eprefactor * kt * dt2 * (1.0 + this.cubic * dt + this.quartic * dt2 + this.pentic * dt3 + this.sextic * dt4);
      const deddt = fprefactor * kt * dt * (2.0 + 3.0 * this.cubic * dt + 4.0 * this.quartic * dt2 +
        5.0 * this.pentic * dt3 + 6.0 * this.sextic * dt4);
      const sign = ee >= 0.0 ? 1.0 : -1.0;
      const dedcos = -deddt * sign / Math.sqrt(cc * rdb2 - ee * ee);
      const term1 = ee / cc;
      const dccdxia = (xad0 * rcd2 - xcd0 * dot) * term1;
      const dccdyia = (xad1 * rcd2 - xcd1 * dot) * term1;
      const dccdzia = (xad2 * rcd2 - xcd2 * dot) * term1;
      const dccdxic = (xcd0 * rad2 - xad0 * dot) * term1;
      const dccdyic = (xcd1 * rad2 - xad1 * dot) * term1;
      const dccdzic = (xcd2 * rad2 - xad2 * dot) * term1;
      const dccdxid = -dccdxia - dccdxic, dccdyid = -dccdyia - dccdyic, dccdzid = -dccdzia - dccdzic;
      const term2 = ee / rdb2;
      const deedxia = xdb[1] * xcb[2] - xdb[2] * xcb[1];
      const deedyia = xdb[2] * xcb[0] - xdb[0] * xcb[2];
      const deedzia = xdb[0] * xcb[1] - xdb[1] * xcb[0];
      const deedxic = xab[1] * xdb[2] - xab[2] * xdb[1];
      const deedyic = xab[2] * xdb[0] - xab[0] * xdb[2];
      const deedzic = xab[0] * xdb[1] - xab[1] * xdb[0];
      const deedxid = xcb[1] * xab[2] - xcb[2] * xab[1] + xdb[0] * term2;
      const deedyid = xcb[2] * xab[0] - xcb[0] * xab[2] + xdb[1] * term2;
      const deedzid = xcb[0] * xab[1] - xcb[1] * xab[0] + xdb[2] * term2;
      const fa = [dedcos * (dccdxia + deedxia), dedcos * (dccdyia + deedyia), dedcos * (dccdzia + deedzia)];
      const fc = [dedcos * (dccdxic + deedxic), dedcos * (dccdyic + deedyic), dedcos * (dccdzic + deedzic)];
      const fd = [dedcos * (dccdxid + deedxid), dedcos * (dccdyid + deedyid), dedcos * (dccdzid + deedzid)];
      const fb = [-fa[0] - fc[0] - fd[0], -fa[1] - fc[1] - fd[1], -fa[2] - fc[2] - fd[2]];
      e += eq;
      // native applies f -= fd (atom 4), fb (atom 2), fa (atom 1), fc (atom 3); the virial is sum r_k F_k over k = 1, 3, 4
      // relative to the centre (sum of forces is zero, so the origin does not matter)
      const pairs: [number[], number[]][] = [[fd, xdb], [fa, xab], [fc, xcb]];
      for (const [atom, g] of [[id, fd], [ib, fb], [ia, fa], [ic, fc]] as [number, number[]][]) {
        f[3 * atom] -= g[0];
        f[3 * atom + 1] -= g[1];
        f[3 * atom + 2] -= g[2];
      }
      for (const [g, r] of pairs) {
        // actual force = -g (native sign): virial += r (x) (-g)
        bc.virial[0] -= r[0] * g[0];
        bc.virial[1] -= r[1] * g[1];
        bc.virial[2] -= r[2] * g[2];
        bc.virial[3] -= 0.5 * (r[0] * g[1] + r[1] * g[0]);
        bc.virial[4] -= 0.5 * (r[0] * g[2] + r[2] * g[0]);
        bc.virial[5] -= 0.5 * (r[1] * g[2] + r[2] * g[1]);
      }
    }
    bc.acc.eimp += e;
  }
}
