/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/fix_amoeba_pitorsion.cpp (read_data_header, read_data_section, pre_neighbor, post_force) at commit
 * b332f28, adapted from Tinker epitors1(). AMOEBA package authors: the LAMMPS developers (Sandia Corporation, GNU
 * General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix amoeba/pitorsion: the pi-torsion term of the AMOEBA force field, between the six atoms of an sp2 pair (atoms 1
 * and 2 of the central bond, their in-plane neighbours 3 and 4 and 5 and 6). The energy is v (1 - cos 2 phi) with the
 * PiTorsion Coeffs value v of the type, phi the angle between the two planes; the forces are the native analytic
 * derivatives. Read from the data file: "N pitorsions" and "N pitorsion types" headers, the PiTorsions section (index,
 * type and six atom IDs per term) and the PiTorsion Coeffs section (type and value). docs.lammps.org/fix_amoeba_pitorsion
 * .html describes the fix (with the pair style amoeba). Native details kept: each term is evaluated once, from its
 * lowest-index atom (pre_neighbor). The energy, forces and virial are the full sums over the six atoms: the engine
 * wraps atoms into the periodic box, so a nearest copy is not told apart from the owned atom by its index, and the
 * native energy of a wrapped protein is the full sum (measured on examples/amoeba ubiquitin). The fix is off when the
 * pair style's pitorsion flag is off.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { BondedCompute } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';
import { buildAtomMap } from '../../engine/atoms';
import { delta } from '../../engine/force/bonded_util';

interface PiTerm {
  type: number;
  ids: number[];
}

export class FixAmoebaPiTorsion extends Fix {
  readonly style = 'amoeba/pitorsion';
  private npit: number | null = null;
  private npitTypes: number | null = null;
  private readonly terms: PiTerm[] = [];
  private kpit = new Float64Array(0);
  private etotal = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError(`fix amoeba/pitorsion group ${group}: only group all is supported`);
    if (args.length) throw new StyleError('usage: fix ID group-ID amoeba/pitorsion');
    this.energyGlobal = true;
    this.virialGlobal = true;
    this.thermoEnergy = true;
    this.thermoVirial = true;
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  /** read_data header lines: "N pitorsions" and "N pitorsion types". */
  readHeader(line: string, at: number): void {
    const m = /^\s*(\d+)\s+(pitorsions|pitorsion types)\s*$/.exec(line);
    if (!m) throw new StyleError(`data file line ${at}: invalid read data header line for amoeba/fix pitorsion: '${line.trim()}'`);
    if (m[2] === 'pitorsions') this.npit = Number(m[1]);
    else this.npitTypes = Number(m[1]);
  }

  /** Lines of a section: the PiTorsion Coeffs section has one line per type, PiTorsions one per term. */
  sectionLines(_natoms: number, title?: string): number {
    if (title === 'PiTorsion Coeffs') {
      if (this.npitTypes === null) throw new StyleError('read_data fix amoeba/pitorsion: the header must give the "pitorsion types" count');
      return this.npitTypes;
    }
    if (this.npit === null) throw new StyleError('read_data fix amoeba/pitorsion: the header must give the "pitorsions" count');
    return this.npit;
  }

  /** The PiTorsions rows (index, type, six atom IDs) and the PiTorsion Coeffs rows (type, value), by the section title. */
  readSection(rows: { w: string[]; at: number }[], title?: string): void {
    if (title === 'PiTorsion Coeffs') {
      const ntypes = this.npitTypes ?? 0;
      this.kpit = new Float64Array(ntypes + 1);
      for (const { w, at } of rows) {
        const itype = Number(w[0]), value = Number(w[1]);
        if (!Number.isInteger(itype) || itype <= 0 || itype > ntypes || !Number.isFinite(value)) {
          throw new StyleError(`data file line ${at}: incorrect args for fix amoeba/pitorsion coeffs`);
        }
        this.kpit[itype] = value;
      }
      return;
    }
    if (title !== 'PiTorsions' && title !== undefined) throw new StyleError(`read_data: invalid section for fix amoeba/pitorsion (${title})`);
    for (const { w, at } of rows) {
      if (w.length !== 8) throw new StyleError(`data file line ${at}: PiTorsions line needs 8 values (index, type, six atom IDs)`);
      const nums = w.map((t) => Number(t));
      if (nums.some((v) => !Number.isInteger(v))) throw new StyleError(`data file line ${at}: PiTorsions values must be integers`);
      this.terms.push({ type: nums[1], ids: nums.slice(2) });
    }
  }

  private pairFlag(): boolean {
    const pair = this.sys.ff.pair as unknown as { extract?(name: string): unknown } | null;
    if (!pair || typeof pair.extract !== 'function') throw new StyleError('Cannot use fix amoeba/pitorsion w/out pair amoeba/hippo');
    const flag = pair.extract('pitorsion_flag');
    if (flag === undefined) throw new StyleError('Cannot use fix amoeba/pitorsion w/out pair amoeba/hippo');
    return Boolean(flag);
  }

  postForce(): void {
    const s: SimState = this.sys.state;
    const f = s.f;
    const v = this.virial;
    v.fill(0);
    this.etotal = 0;
    if (!this.pairFlag()) return;
    const shim = { s, geom: this.sys.geom } as unknown as BondedCompute;
    const map = buildAtomMap(s);
    const idx = (id: number): number => {
      const i = id < map.length ? map[id] : -1;
      if (i < 0) throw new StyleError(`fix amoeba/pitorsion: atom ${id} missing`);
      return i;
    };
    const d = [0, 0, 0];
    for (const term of this.terms) {
      const a = term.ids.map(idx);
      // the owner is the lowest-index atom of the term (pre_neighbor lists each term once, from it)
      let owner = a[0];
      for (const k of a) if (k < owner) owner = k;
      // positions relative to the owner (nearest copies); every atom of the term takes its share
      const rel: number[][] = [];
      for (let k = 0; k < 6; k++) {
        delta(shim, owner, a[k], d);
        rel.push([d[0], d[1], d[2]]);
      }
      const r = this.pitorsionTerm(rel, term.type);
      if (r === null) continue;
      this.etotal += r.e;
      for (let k = 0; k < 6; k++) {
        const g = r.g[k];
        f[3 * a[k]] -= g[0];
        f[3 * a[k] + 1] -= g[1];
        f[3 * a[k] + 2] -= g[2];
      }
      for (let c = 0; c < 6; c++) v[c] += r.v[c];
    }
  }

  /**
   * The pi-torsion energy and forces of one term (fix_amoeba_pitorsion.cpp post_force) with the positions rel of the six
   * atoms relative to the owner. Returns the energy, the forces g[k] per atom in native order (ia, ib, ic, id, ie, ig)
   * and the virial (xx, yy, zz, xy, xz, yz) from the native tally; null for a degenerate geometry (no energy).
   */
  private pitorsionTerm(rel: number[][], ptype: number): { e: number; g: number[][]; v: number[] } | null {
    return pitorsionGeometry(rel, this.kpit[ptype]);
  }

  energy(): number { return this.etotal; }
  computeScalar(): number { return this.etotal; }
  minPostForce(): void { this.postForce(); }
}

/** The pi-torsion energy, forces and virial tally of one term (module-level so that it can be checked on its own). */
export const pitorsionGeometry = (rel: number[][], v2: number): { e: number; g: number[][]; v: number[] } | null => {
    const [ia, ib, ic, id, ie, ig] = rel;
    const xia = ia[0], yia = ia[1], zia = ia[2];
    const xib = ib[0], yib = ib[1], zib = ib[2];
    const xic = ic[0], yic = ic[1], zic = ic[2];
    const xid = id[0], yid = id[1], zid = id[2];
    const xie = ie[0], yie = ie[1], zie = ie[2];
    const xig = ig[0], yig = ig[1], zig = ig[2];
    const xad = xia - xid, yad = yia - yid, zad = zia - zid;
    const xbd = xib - xid, ybd = yib - yid, zbd = zib - zid;
    const xec = xie - xic, yec = yie - yic, zec = zie - zic;
    const xgc = xig - xic, ygc = yig - yic, zgc = zig - zic;
    const xip = yad * zbd - ybd * zad + xic;
    const yip = zad * xbd - zbd * xad + yic;
    const zip = xad * ybd - xbd * yad + zic;
    const xiq = yec * zgc - ygc * zec + xid;
    const yiq = zec * xgc - zgc * xec + yid;
    const ziq = xec * ygc - xgc * yec + zid;
    const xcp = xic - xip, ycp = yic - yip, zcp = zic - zip;
    const xdc = xid - xic, ydc = yid - yic, zdc = zid - zic;
    const xqd = xiq - xid, yqd = yiq - yid, zqd = ziq - zid;
    const xt = ycp * zdc - ydc * zcp, yt = zcp * xdc - zdc * xcp, zt = xcp * ydc - xdc * ycp;
    const xu = ydc * zqd - yqd * zdc, yu = zdc * xqd - zqd * xdc, zu = xdc * yqd - xqd * ydc;
    const xtu = yt * zu - yu * zt, ytu = zt * xu - zu * xt, ztu = xt * yu - xu * yt;
    const rt2 = xt * xt + yt * yt + zt * zt;
    const ru2 = xu * xu + yu * yu + zu * zu;
    const rtru = Math.sqrt(rt2 * ru2);
    if (rtru <= 0.0) return null;
    const rdc = Math.sqrt(xdc * xdc + ydc * ydc + zdc * zdc);
    const cosine = (xt * xu + yt * yu + zt * zu) / rtru;
    const sine = (xdc * xtu + ydc * ytu + zdc * ztu) / (rdc * rtru);
    const c2 = -1.0, s2 = 0.0;
    const cosine2 = cosine * cosine - sine * sine;
    const sine2 = 2.0 * cosine * sine;
    const phi2 = 1.0 + (cosine2 * c2 + sine2 * s2);
    const dphi2 = 2.0 * (cosine2 * s2 - sine2 * c2);
    const e = v2 * phi2;
    const dedphi = v2 * dphi2;
    const xdp = xid - xip, ydp = yid - yip, zdp = zid - zip;
    const xqc = xiq - xic, yqc = yiq - yic, zqc = ziq - zic;
    const dedxt = dedphi * (yt * zdc - ydc * zt) / (rt2 * rdc);
    const dedyt = dedphi * (zt * xdc - zdc * xt) / (rt2 * rdc);
    const dedzt = dedphi * (xt * ydc - xdc * yt) / (rt2 * rdc);
    const dedxu = -dedphi * (yu * zdc - ydc * zu) / (ru2 * rdc);
    const dedyu = -dedphi * (zu * xdc - zdc * xu) / (ru2 * rdc);
    const dedzu = -dedphi * (xu * ydc - xdc * yu) / (ru2 * rdc);
    const dedxip = zdc * dedyt - ydc * dedzt;
    const dedyip = xdc * dedzt - zdc * dedxt;
    const dedzip = ydc * dedxt - xdc * dedyt;
    let dedxic = ydp * dedzt - zdp * dedyt + zqd * dedyu - yqd * dedzu;
    let dedyic = zdp * dedxt - xdp * dedzt + xqd * dedzu - zqd * dedxu;
    let dedzic = xdp * dedyt - ydp * dedxt + yqd * dedxu - xqd * dedyu;
    let dedxid = zcp * dedyt - ycp * dedzt + yqc * dedzu - zqc * dedyu;
    let dedyid = xcp * dedzt - zcp * dedxt + zqc * dedxu - xqc * dedzu;
    let dedzid = ycp * dedxt - xcp * dedyt + xqc * dedyu - yqc * dedxu;
    const dedxiq = zdc * dedyu - ydc * dedzu;
    const dedyiq = xdc * dedzu - zdc * dedxu;
    const dedziq = ydc * dedxu - xdc * dedyu;
    const dedxia = ybd * dedzip - zbd * dedyip;
    const dedyia = zbd * dedxip - xbd * dedzip;
    const dedzia = xbd * dedyip - ybd * dedxip;
    const dedxib = zad * dedyip - yad * dedzip;
    const dedyib = xad * dedzip - zad * dedxip;
    const dedzib = yad * dedxip - xad * dedyip;
    const dedxie = ygc * dedziq - zgc * dedyiq;
    const dedyie = zgc * dedxiq - xgc * dedziq;
    const dedzie = xgc * dedyiq - ygc * dedxiq;
    const dedxig = zec * dedyiq - yec * dedziq;
    const dedyig = xec * dedziq - zec * dedxiq;
    const dedzig = yec * dedxiq - xec * dedyiq;
    dedxic = dedxic + dedxip - dedxie - dedxig;
    dedyic = dedyic + dedyip - dedyie - dedyig;
    dedzic = dedzic + dedzip - dedzie - dedzig;
    dedxid = dedxid + dedxiq - dedxia - dedxib;
    dedyid = dedyid + dedyiq - dedyia - dedyib;
    dedzid = dedzid + dedziq - dedzia - dedzib;
    const g = [
      [dedxia, dedyia, dedzia], [dedxib, dedyib, dedzib], [dedxic, dedyic, dedzic],
      [dedxid, dedyid, dedzid], [dedxie, dedyie, dedzie], [dedxig, dedyig, dedzig],
    ];
    const vxterm = dedxid + dedxia + dedxib;
    const vyterm = dedyid + dedyia + dedyib;
    const vzterm = dedzid + dedzia + dedzib;
    // ev_tally with nctotal 6 (the caller scales by the owned count): v as native
    const v = [
      -xdc * vxterm - xcp * dedxip + xqd * dedxiq,
      -ydc * vyterm - ycp * dedyip + yqd * dedyiq,
      -zdc * vzterm - zcp * dedzip + zqd * dedziq,
      -ydc * vxterm - ycp * dedxip + yqd * dedxiq,
      -zdc * vxterm - zcp * dedxip + zqd * dedxiq,
      -zdc * vyterm - zcp * dedyip + zqd * dedyiq,
    ];
    return { e, g, v };
  }
