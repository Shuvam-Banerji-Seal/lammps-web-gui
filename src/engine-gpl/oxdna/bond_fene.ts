/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/bond_oxdna_fene.cpp and
 * bond_oxdna_fene.h (class BondOxdnaFene, bond_style oxdna/fene; the oxDNA backbone FENE bond with the
 * backbone-site offsets and the 3'->5' direction of the strand). Original authors (from the headers): Oliver
 * Henrich (University of Strathclyde, Glasgow) and others of the CG-DNA package; described by Ouldridge, Louis
 * and Doye (J Chem Phys 2011). LAMMPS is Copyright (2003) Sandia Corporation, distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The C++ looks up Delta and r0 per tetramer type (a 3' neighbour, a, b, b's 5' neighbour); coeff() writes the
 * same value to every entry, so one value per bond type is kept here (the same numbers, the same arithmetic).
 */
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext, NoSettingsError, potentialFile } from '../../engine/force/types';
import { atomIndex, delta as nearestCopy } from '../../engine/force/bonded_util';
import { OXDNA_3P, OXDNA_5P, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { type OxdnaConstants, backboneOxdna1, halfVirial, lrfOf, oxdnaConstants, potentialLines } from './common';
import { log as libmLog } from '../libm/log';

/** bond_oxdna_fene.cpp: "const double rlogarg_min = 0.2;" */
const RLOGARG_MIN = 0.2;

export interface Pair2 { k: number; delta: number; r0: number }

export class BondOxdnaFene extends Bonded {
  readonly name: string = 'oxdna/fene';
  /** backbone site of the nucleotide model (oxDNA1 here; the oxDNA2 and oxRNA2 subclasses override it) */
  protected backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxdna1(c, e1, e2, e3, out);
  }
  /**
   * The parameters of a bond of type `type` between a (the 3' side, atom type atype, 3' neighbour type a3p) and b
   * (5' side, type btype, 5' neighbour type b5p); a neighbour type 0 means "none". Here one set per bond type.
   */
  protected params(type: number, _a3p: number, _atype: number, _btype: number, _b5p: number): Pair2 {
    return this.par[type]!;
  }
  readonly kind = 'bond' as const;
  protected par: (Pair2 | null)[] = [];
  /** true for the per-tetramer parameters of oxdna3/fene (Delta and r0 depend on the neighbour types) */
  protected tetramer = false;
  private special: readonly number[] | null = null;
  private lrf = new Float64Array(0);

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new NoSettingsError(args[0]);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.par = new Array(ntypes + 1).fill(null);
  }

  /**
   * bond_oxdna_fene.cpp coeff(): "bond_coeff N K Delta r0" (narg 4) or "bond_coeff N potential-file" (narg 2, the
   * line whose first word is N and second word "fene" gives K Delta r0). Other counts: "Incorrect args for bond
   * coefficients in oxdna/fene, oxdna2/fene or oxrna2/fene".
   */
  coeff(args: string[], ctx?: StyleContext): void {
    const bad = () => new StyleError('Incorrect args for bond coefficients in oxdna/fene, oxdna2/fene or oxrna2/fene');
    if (args.length !== 4 && args.length !== 2) throw bad();
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    let k: number, delta: number, r0: number;
    if (args.length === 4) {
      k = parseNum(args[1], 'K');
      delta = parseNum(args[2], 'Delta');
      r0 = parseNum(args[3], 'r0');
    } else {
      // bond_oxdna_fene.cpp: "No corresponding fene potential found in file {} for bond type {}"
      const text = ctx?.readFile(args[1], potentialFile('oxdna potential')) ?? '';
      let found: number[] | null = null;
      for (const w of potentialLines(text, [args[0]], 'fene')) {
        if (w.length < 2) continue;
        if (w.length < 5) throw new StyleError(`Problem parsing oxdna, oxdna2 or oxrna2 potential file: fene needs K Delta r0 in ${args[1]}`);
        found = [parseNum(w[2], 'K'), parseNum(w[3], 'Delta'), parseNum(w[4], 'r0')];
        break;
      }
      if (!found) throw new StyleError(`No corresponding fene potential found in file ${args[1]} for bond type ${args[0]}`);
      [k, delta, r0] = found;
    }
    for (let t = lo; t <= hi; t++) this.par[t] = { k, delta, r0 };
  }

  /** bond_oxdna_fene.cpp init_style(): the pair needs "special_bonds lj 0 1 1" and fix OXDNA/LRF (pair oxdna/excv). */
  linkForceField(link: { pair: unknown; special: { lj: readonly number[]; coul: readonly number[] } }): void {
    this.special = link.special.lj;
  }

  init(_ctx?: StyleContext): void {
    // special_bonds lj weights of the engine are [1-2, 1-3, 1-4] (the C++ special_lj[1..3])
    const sp = this.special;
    if (sp && (sp[0] !== 0.0 || sp[1] !== 1.0 || sp[2] !== 1.0)) {
      throw new StyleError("Must use 'special_bonds lj 0 1 1' with bond style oxdna/fene, oxdna2/fene, oxdna3/fene or oxrna2/fene");
    }
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.par[t]) throw new StyleError(`All bond coeffs are not set (bond type ${t})`);
    }
  }

  /** bond_oxdna_fene.cpp equilibrium_distance(): r0. */
  equilibrium(type: number): number {
    return this.par[type]?.r0 ?? Number.NaN;
  }

  compute(bc: BondedCompute): void {
    const s = bc.s;
    const bonds = s.topo.bonds;
    const constants = oxdnaConstants(s.units.style);
    const f = bc.f, x = s.x;
    const torque = s.torque!;
    const id5p = s.custom.get(OXDNA_5P)!.data;
    // tetramer types: the type of the atom with a given tag (0 = no neighbour), built once per call
    const id3p = this.tetramer ? s.custom.get(OXDNA_3P)!.data : null;
    const tagType = this.tetramer ? new Map<number, number>() : null;
    if (tagType) for (let i = 0; i < s.n; i++) tagType.set(s.id[i], s.type[i]);
    if (!s.custom.has('#id5p') || !s.quat || !s.torque) throw new StyleError('bond_style oxdna/fene requires atom_style hybrid bond ellipsoid oxdna');
    const ntotal = s.n;
    // local reference frames of all atoms (fix OXDNA/LRF, compute_lrf); index by atom index
    if (this.lrf.length !== 9 * ntotal) this.lrf = new Float64Array(9 * ntotal);
    for (let i = 0; i < ntotal; i++) {
      if (isEllipsoid(s, i)) lrfOf(s.quat!, 4 * i, this.lrf, 9 * i);
      else this.lrf.fill(0, 9 * i, 9 * i + 9);
    }
    const lrf = this.lrf;
    const ra = [0, 0, 0], rb = [0, 0, 0];
    const dbb = [0, 0, 0], delf = [0, 0, 0], d = [0, 0, 0];
    const image = [0, 0, 0];
    const rel = [0, 0, 0];
    let ebondTotal = 0;
    for (let k = 0; k < bonds.n; k++) {
      // newton_bond on: the bond is listed on its first atom (bondlist[][0] = atom1, [1] = atom2)
      let b = atomIndex(bc, bonds.atoms[2 * k], 'bond');
      let a = atomIndex(bc, bonds.atoms[2 * k + 1], 'bond');
      const type = bonds.type[k];
      // directionality test: a -> b is 3' -> 5' when tag[b] == id5p[a]
      if (s.id[b] !== id5p[a]) {
        const tmp = b;
        b = a;
        a = tmp;
      }
      const ax = lrf.subarray(9 * a, 9 * a + 3), ay = lrf.subarray(9 * a + 3, 9 * a + 6), az = lrf.subarray(9 * a + 6, 9 * a + 9);
      const bx = lrf.subarray(9 * b, 9 * b + 3), by = lrf.subarray(9 * b + 3, 9 * b + 6), bz = lrf.subarray(9 * b + 6, 9 * b + 9);
      this.backbone(constants, ax, ay, az, ra);
      this.backbone(constants, bx, by, bz, rb);
      // closest image of b relative to a (the C++ reads the ghost image of b that is nearest to a)
      nearestCopy(bc, a, b, d);
      image[0] = x[3 * a] + d[0]; image[1] = x[3 * a + 1] + d[1]; image[2] = x[3 * a + 2] + d[2];
      dbb[0] = x[3 * a] + ra[0] - image[0] - rb[0];
      dbb[1] = x[3 * a + 1] + ra[1] - image[1] - rb[1];
      dbb[2] = x[3 * a + 2] + ra[2] - image[2] - rb[2];
      const rsqBkbk = dbb[0] * dbb[0] + dbb[1] * dbb[1] + dbb[2] * dbb[2];
      const rBkbk = Math.sqrt(rsqBkbk);
      let a3p = 0, b5p = 0;
      if (tagType && id3p) { a3p = tagType.get(id3p[a]) ?? 0; b5p = tagType.get(id5p[b]) ?? 0; }
      const p = this.params(type, a3p, s.type[a], s.type[b], b5p);
      const delta = p.delta, r0 = p.r0, kk = p.k;
      let rr0 = rBkbk - r0;
      const rr0sq = rr0 * rr0;
      const deltasq = delta * delta;
      let rlogarg = 1.0 - rr0sq / deltasq;
      let ebond = -0.5 * kk * libmLog(rlogarg);
      // capped force for r - r0 beyond Delta (bond_oxdna_fene.cpp, "switching to capped force")
      if (rlogarg < RLOGARG_MIN) {
        bc.warn?.(`FENE bond too long: ${s.step} ${s.id[a]} ${s.id[b]} ${rBkbk}`);
        rlogarg = RLOGARG_MIN;
        if (rBkbk > r0) {
          rr0 = delta * Math.sqrt(1.0 - rlogarg);
          ebond = -0.5 * kk * libmLog(rlogarg) + kk * Math.sqrt(1.0 - rlogarg) / rlogarg / delta *
            (rBkbk - r0 - delta * Math.sqrt(1.0 - rlogarg));
        } else if (rBkbk < r0) {
          rr0 = -delta * Math.sqrt(1.0 - rlogarg);
          ebond = -0.5 * kk * libmLog(rlogarg) + kk * Math.sqrt(1.0 - rlogarg) / rlogarg / delta *
            (r0 - delta * Math.sqrt(1.0 - rlogarg) - rBkbk);
        }
      }
      const fbond = -kk * rr0 / rlogarg / deltasq / rBkbk;
      delf[0] = dbb[0] * fbond; delf[1] = dbb[1] * fbond; delf[2] = dbb[2] * fbond;
      // force and torque on a (cross(ra, delf)) and on b (minus cross(rb, delf))
      f[3 * a] += delf[0]; f[3 * a + 1] += delf[1]; f[3 * a + 2] += delf[2];
      torque[3 * a] += ra[1] * delf[2] - ra[2] * delf[1];
      torque[3 * a + 1] += ra[2] * delf[0] - ra[0] * delf[2];
      torque[3 * a + 2] += ra[0] * delf[1] - ra[1] * delf[0];
      f[3 * b] -= delf[0]; f[3 * b + 1] -= delf[1]; f[3 * b + 2] -= delf[2];
      const dbx = rb[1] * delf[2] - rb[2] * delf[1];
      const dby = rb[2] * delf[0] - rb[0] * delf[2];
      const dbz = rb[0] * delf[1] - rb[1] * delf[0];
      torque[3 * b] -= dbx; torque[3 * b + 1] -= dby; torque[3 * b + 2] -= dbz;
      // energy (per-atom split) and the molecular virial (x_a - x_b) (x) delf
      ebondTotal += ebond;
      if (bc.eatom) { bc.eatom[a] += 0.5 * ebond; bc.eatom[b] += 0.5 * ebond; }
      rel[0] = x[3 * a] - image[0]; rel[1] = x[3 * a + 1] - image[1]; rel[2] = x[3 * a + 2] - image[2];
      const v = bc.virial;
      v[0] += rel[0] * delf[0]; v[1] += rel[1] * delf[1]; v[2] += rel[2] * delf[2];
      v[3] += rel[0] * delf[1]; v[4] += rel[0] * delf[2]; v[5] += rel[1] * delf[2];
      if (bc.vatom) halfVirial(bc.vatom, a, b, rel, delf);
    }
    bc.acc.ebond += ebondTotal;
  }
}
