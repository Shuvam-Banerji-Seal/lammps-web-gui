/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: pair_style bop
 * (src/MANYBODY/pair_bop.cpp and pair_bop.h). Contributing authors: D.K. Ward (donward@sandia.gov) and X.W. Zhou
 * (Sandia). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_bop.html. "pair_style bop" takes the optional keyword save ("pre-compute and save some
 * values"); "pair_coeff * * file elem1 ... elemN" reads the tabulated potential. The cutoffs come from the file, and
 * comm_modify cutoff must be at least 3 times the largest cutoff ("Pair style bop requires setting a communication
 * cutoff of at least").
 *
 * The native bookkeeping is kept: pairlist1 (cutB neighbours with r, dis, betaS, betaP, rep and their derivatives),
 * pairlist2 (cutL neighbours with the cphi term), BOP_index / BOP_total per atom, and the angular triplets (computed
 * on the fly, or pre-computed with "save"; both give the same values). The neighbour lists of ghost atoms are
 * translated from their owners' full lists (the image key of the owner and the integer image offset), so j and k
 * can be ghosts. The sums run in native order; forces are accumulated into the owned+ghost array and folded back
 * by the force field.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { checkOneCoeffArgs, mapElementsToTypes } from '../manybody2/common';
import { pow as libmPow } from '../libm/pow';
import { parseBopPotential, type BopPotential } from './potential';

/** Growable typed array (the native memory->grow calls). */
const grow = <T extends Float64Array | Int32Array>(a: T, n: number): T => {
  if (a.length >= n) return a;
  const next = Math.max(n, 2 * a.length) as number;
  const out = (a instanceof Float64Array ? new Float64Array(next) : new Int32Array(next)) as T;
  out.set(a);
  return out;
};

/** The formatted value of "{:.4}" (4 significant digits, general notation) used in the communication-cutoff error. */
const fmt4 = (v: number): string => {
  const s = Number(v.toPrecision(4)).toString();
  return s;
};

/** One angle triplet's values: cosine, G and dG/dcos, and the derivatives for the two neighbours (a, b). */
class Trip {
  cos = 0;
  g = 0;
  dg = 0;
  readonly a = new Float64Array(3);
  readonly b = new Float64Array(3);
}

/** One entry of the B_SG list (the derivative bookkeeping of SigmaBo). */
class BSg {
  i = -1;
  j = -1;
  temp = -1;
  readonly dAA = new Float64Array(3);
  readonly dBB = new Float64Array(3);
  readonly dEE1 = new Float64Array(3);
  readonly dFF = new Float64Array(3);
  readonly dAAC = new Float64Array(3);
  readonly dSigB1 = new Float64Array(3);
  readonly dSigB = new Float64Array(3);
  reset(): void {
    this.i = -1; this.j = -1; this.temp = -1;
    for (const v of [this.dAA, this.dBB, this.dEE1, this.dFF, this.dAAC, this.dSigB1, this.dSigB]) v.fill(0);
  }
}

/** One entry of the B_PI list (the derivative bookkeeping of PiBo). */
class BPi {
  i = -1;
  j = -1;
  temp = -1;
  readonly dAA = new Float64Array(3);
  readonly dBB = new Float64Array(3);
  readonly dPiB = new Float64Array(3);
  reset(): void {
    this.i = -1; this.j = -1; this.temp = -1;
    for (const v of [this.dAA, this.dBB, this.dPiB]) v.fill(0);
  }
}

/** Key of an atom from its owner and image offset (owner atoms have image 0). */
const imageKey = (owner: number, ix: number, iy: number, iz: number): number =>
  owner * 1073741824 + (ix + 512) * 1048576 + (iy + 512) * 1024 + (iz + 512);

export class PairBOP extends Pair {
  override readonly name: string = 'bop';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  /** save keyword: the angular triplets are computed once per neighbour build instead of on the fly. */
  protected save = false;
  protected pot: BopPotential | null = null;
  /** Atom type -> element index of the potential file (-1 for NULL). */
  protected map: Int32Array = new Int32Array(0);

  // neighbour bookkeeping (native BOP_index etc.), sized per evaluation
  protected bopIndex = new Int32Array(0);
  protected bopTotal = new Int32Array(0);
  protected bopIndex2 = new Int32Array(0);
  protected bopTotal2 = new Int32Array(0);
  protected cosIndex = new Int32Array(0);
  protected nbr1 = new Int32Array(0);
  protected r1 = new Float64Array(0);
  protected d1 = new Float64Array(0);
  protected bS1 = new Float64Array(0);
  protected dbS1 = new Float64Array(0);
  protected bP1 = new Float64Array(0);
  protected dbP1 = new Float64Array(0);
  protected rep1 = new Float64Array(0);
  protected drep1 = new Float64Array(0);
  protected nbr2 = new Int32Array(0);
  protected r2 = new Float64Array(0);
  protected d2 = new Float64Array(0);
  protected rep2 = new Float64Array(0);
  protected drep2 = new Float64Array(0);
  protected tCos = new Float64Array(0);
  protected tG = new Float64Array(0);
  protected tDG = new Float64Array(0);
  protected tDj = new Float64Array(0);
  protected tDk = new Float64Array(0);

  // CSR full neighbour lists of owned and ghost atoms (translated for ghosts)
  protected csrStart = new Int32Array(0);
  protected csrNbr = new Int32Array(0);

  // per-evaluation context
  protected x: Float64Array = new Float64Array(0);
  protected f: Float64Array = new Float64Array(0);
  protected type: Int32Array = new Int32Array(0);

  // angle scratch
  protected angDj = new Float64Array(3);
  protected angDk = new Float64Array(3);
  protected readonly tJik = new Trip();
  protected readonly tIjk = new Trip();
  protected readonly tIkj = new Trip();
  protected readonly tJikp = new Trip();
  protected readonly tKikp = new Trip();
  protected readonly tIjkp = new Trip();
  protected readonly tKjkp = new Trip();

  // derivative bookkeeping lists
  protected sg: BSg[] = [];
  protected nSg = 0;
  protected bi: BPi[] = [];
  protected nBi = 0;

  override settings(args: string[], _ctx?: StyleContext): void {
    this.save = false;
    let k = 0;
    while (k < args.length) {
      if (args[k] === 'save') {
        this.save = true;
        k++;
      } else {
        throw new StyleError('Illegal pair_style command');
      }
    }
  }

  override coeff(args: string[], ctx: StyleContext): void {
    checkOneCoeffArgs(args, this.ntypes, this.name);
    const { elements, map } = mapElementsToTypes(args.slice(3), this.ntypes);
    const pot = parseBopPotential(ctx.readFile(args[2]), args[2], ctx);
    // replace element indices in the map with the potential file's indices (native coeff, on rank 0)
    for (let i = 1; i <= this.ntypes; i++) {
      if (map[i] < 0) continue;
      const name = elements[map[i]];
      const j = pot.names.indexOf(name);
      if (j < 0) throw new StyleError(`Element ${name} not found in bop potential file ${args[2]}`);
      map[i] = j;
      this.setMass(ctx, i, pot.masses[j]);
    }
    this.pot = pot;
    this.map = map;
  }

  /** atom->set_mass (atom.cpp): a value <= 0 is ignored with a warning. */
  protected setMass(ctx: StyleContext, type: number, value: number): void {
    const s = ctx.s;
    if (!s) return;
    if (s.rmass) throw new StyleError('Cannot set per-type atom mass for this atom style');
    if (!(value > 0)) {
      ctx.log(`WARNING: Ignoring invalid mass value ${value} for atom type ${type}`);
      return;
    }
    s.massByType[type] = value;
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError('Pair style BOP requires newton pair on');
    if (this.hybridSub || this.hybridOverlay) {
      throw new StyleError('Pair style BOP is not compatible with hybrid pair styles');
    }
  }

  override initOne(i: number, j: number): number {
    const pot = this.pot;
    if (!pot || this.map.length === 0 || this.map[i] < 0 || this.map[j] < 0) {
      throw new StyleError('All pair coeffs are not set');
    }
    const p = pot.pairs[pot.elem2param[this.map[i] * pot.nb + this.map[j]]];
    return p.cutL > p.cutB ? p.cutL : p.cutB;
  }

  override compute(pc: PairCompute): void {
    const pot = this.pot;
    if (!pot) throw new StyleError('All pair coeffs are not set');
    if (!pc.full) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    if (pc.nb.multiStyle) throw new StyleError('Pair style BOP is not compatible with multi-cutoff neighbor lists');
    if (pc.vatom) throw new StyleError('Pair style bop does not support per-atom virial (compute stress/atom)');
    if (pc.nb.commCutoff - 0.001 < 3.0 * pot.cutmax) {
      throw new StyleError(
        `Pair style bop requires setting a communication cutoff of at least ${fmt4(3.0 * pot.cutmax)}`,
      );
    }
    this.x = pc.x;
    this.f = pc.f;
    this.type = pc.type;
    const nlocal = pc.nlocal;
    this.buildCsr(pc, pot);
    this.gneigh(pc, pot);

    const x = pc.x, f = pc.f;
    const tagOf = (j: number): number => pc.s.id[pc.nb.owner[j]];
    for (let i = 0; i < nlocal; i++) {
      const itype = this.map[this.type[i]];
      const itag = tagOf(i);
      const nlisti = this.bopTotal[i];
      for (let jj = 0; jj < nlisti; jj++) {
        const tij = this.bopIndex[i] + jj;
        const j = this.nbr1[tij];
        const jtag = tagOf(j);
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < x[3 * i + 2]) continue;
          if (x[3 * j + 2] === x[3 * i + 2] && x[3 * j + 1] < x[3 * i + 1]) continue;
          if (x[3 * j + 2] === x[3 * i + 2] && x[3 * j + 1] === x[3 * i + 1] && x[3 * j] < x[3 * i]) continue;
        }
        const jtype = this.map[this.type[j]];
        const paramIJ = pot.elem2param[itype * pot.nb + jtype];
        const sigB0 = this.sigmaBo(pot, i, jj);
        const piB0 = pot.piA[paramIJ] === 0 ? 0.0 : this.piBo(pot, i, jj);
        const dpr1 = (this.drep1[tij] - 2.0 * this.dbS1[tij] * sigB0 - 2.0 * this.dbP1[tij] * piB0) / this.r1[tij];
        const d0 = this.d1[3 * tij], d1 = this.d1[3 * tij + 1], d2 = this.d1[3 * tij + 2];
        const ftmp1 = dpr1 * d0, ftmp2 = dpr1 * d1, ftmp3 = dpr1 * d2;
        f[3 * i] += ftmp1;
        f[3 * i + 1] += ftmp2;
        f[3 * i + 2] += ftmp3;
        f[3 * j] -= ftmp1;
        f[3 * j + 1] -= ftmp2;
        f[3 * j + 2] -= ftmp3;
        const dE = this.rep1[tij] - 2.0 * this.bS1[tij] * sigB0 - 2.0 * this.bP1[tij] * piB0;
        this.tallyEnergy(pc, i, j, dE);
      }
      const nlisti2 = this.bopTotal2[i];
      for (let jj = 0; jj < nlisti2; jj++) {
        const tij = this.bopIndex2[i] + jj;
        const j = this.nbr2[tij];
        const jtag = tagOf(j);
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < x[3 * i + 2]) continue;
          if (x[3 * j + 2] === x[3 * i + 2] && x[3 * j + 1] < x[3 * i + 1]) continue;
          if (x[3 * j + 2] === x[3 * i + 2] && x[3 * j + 1] === x[3 * i + 1] && x[3 * j] < x[3 * i]) continue;
        }
        const dpr2 = -this.drep2[tij] / this.r2[tij];
        const d0 = this.d2[3 * tij], d1 = this.d2[3 * tij + 1], d2 = this.d2[3 * tij + 2];
        const ftmp1 = dpr2 * d0, ftmp2 = dpr2 * d1, ftmp3 = dpr2 * d2;
        f[3 * i] += ftmp1;
        f[3 * i + 1] += ftmp2;
        f[3 * i + 2] += ftmp3;
        f[3 * j] -= ftmp1;
        f[3 * j + 1] -= ftmp2;
        f[3 * j + 2] -= ftmp3;
        this.tallyEnergy(pc, i, j, -this.rep2[tij]);
      }
    }
  }

  /** Pair::ev_tally with newton on: the global energy, and half of it per atom when eatom is kept. */
  protected tallyEnergy(pc: PairCompute, i: number, j: number, evdwl: number): void {
    pc.acc.evdwl += evdwl;
    if (pc.eatom) {
      pc.eatom[i] += 0.5 * evdwl;
      pc.eatom[j] += 0.5 * evdwl;
    }
  }

  /**
   * Full neighbour lists in CSR form for every owned and ghost atom. An owned atom uses the engine's full list. A ghost
   * uses its owner's list translated by the ghost's image: the neighbour n of the owner is the atom whose owner is
   * owner(n) and whose image is image(n) + image(ghost).
   */
  protected buildCsr(pc: PairCompute, pot: BopPotential): void {
    const nall = pc.nall, nlocal = pc.nlocal;
    const list = pc.full!;
    const owner = pc.nb.owner, gimage = pc.nb.gimage;
    const keyOf = new Map<number, number>();
    for (let a = 0; a < nall; a++) {
      const o = owner[a];
      keyOf.set(imageKey(o, gimage[3 * a], gimage[3 * a + 1], gimage[3 * a + 2]), a);
    }
    let start = new Int32Array(nall + 1);
    let nbr = new Int32Array(0);
    let n = 0;
    for (let a = 0; a < nall; a++) {
      start[a] = n;
      if (a < nlocal) {
        const k0 = list.firstneigh[a], cnt = list.numneigh[a];
        nbr = grow(nbr, n + cnt);
        for (let k = 0; k < cnt; k++) nbr[n++] = list.neighbors[k0 + k] & NEIGHMASK;
      } else {
        const o = owner[a];
        const k0 = list.firstneigh[o], cnt = list.numneigh[o];
        nbr = grow(nbr, n + cnt);
        for (let k = 0; k < cnt; k++) {
          const m = list.neighbors[k0 + k] & NEIGHMASK;
          const key = imageKey(
            owner[m], gimage[3 * m] + gimage[3 * a], gimage[3 * m + 1] + gimage[3 * a + 1],
            gimage[3 * m + 2] + gimage[3 * a + 2],
          );
          const hit = keyOf.get(key);
          if (hit !== undefined) {
            nbr[n++] = hit;
          }
          // A translated neighbour outside the ghost shell is left out: native gneigh builds the lists of the outermost
          // ghosts from the atoms it holds, which truncates them the same way. The atoms whose lists are used (the
          // neighbours of the owned atoms, and their neighbours) lie within 3 cutmax, which comm_modify cutoff must cover.
        }
      }
    }
    start[nall] = n;
    this.csrStart = start;
    this.csrNbr = nbr;
  }

  /** gneigh: the BOP lists (pairlist1 within cutB, pairlist2 within cutL) and, with save, the triplets. */
  protected gneigh(pc: PairCompute, pot: BopPotential): void {
    const nall = pc.nall;
    const x = pc.x, type = this.type, map = this.map, nb = pot.nb;
    const csrS = this.csrStart, csrN = this.csrNbr;
    this.bopIndex = grow(this.bopIndex, nall);
    this.bopTotal = grow(this.bopTotal, nall);
    this.bopIndex2 = grow(this.bopIndex2, nall);
    this.bopTotal2 = grow(this.bopTotal2, nall);
    let total1 = 0, total2 = 0;
    for (let i = 0; i < nall; i++) {
      this.bopIndex[i] = total1;
      this.bopIndex2[i] = total2;
      const itype = map[type[i]];
      let max1 = 0, max2 = 0;
      for (let q = csrS[i]; q < csrS[i + 1]; q++) {
        const j = csrN[q];
        const jtype = map[type[j]];
        const paramIJ = pot.elem2param[itype * nb + jtype];
        const p = pot.pairs[paramIJ];
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < p.cutBsq) {
          const r = Math.sqrt(rsq);
          const t = total1 + max1;
          this.grow1(t + 1);
          this.nbr1[t] = j;
          this.r1[t] = r;
          this.d1[3 * t] = dx; this.d1[3 * t + 1] = dy; this.d1[3 * t + 2] = dz;
          this.bS1[t] = p.betaS.value(r);
          this.dbS1[t] = p.betaS.dy;
          if (pot.piA[paramIJ] !== 0) {
            this.bP1[t] = p.betaP.value(r);
            this.dbP1[t] = p.betaP.dy;
          } else {
            this.bP1[t] = 0.0;
            this.dbP1[t] = 0.0;
          }
          this.rep1[t] = p.rep.value(r);
          this.drep1[t] = p.rep.dy;
          max1++;
        }
        if (rsq < p.cutLsq) {
          const r = Math.sqrt(rsq);
          const t = total2 + max2;
          this.grow2(t + 1);
          this.nbr2[t] = j;
          this.r2[t] = r;
          this.d2[3 * t] = dx; this.d2[3 * t + 1] = dy; this.d2[3 * t + 2] = dz;
          this.rep2[t] = p.cphi!.value(r);
          this.drep2[t] = p.cphi!.dy;
          max2++;
        }
      }
      this.bopTotal[i] = max1;
      this.bopTotal2[i] = max2;
      total1 += max1;
      total2 += max2;
    }

    if (this.save) {
      this.cosIndex = grow(this.cosIndex, nall);
      let cosTotal = 0;
      for (let i = 0; i < nall; i++) {
        this.cosIndex[i] = cosTotal;
        const itype = map[type[i]];
        const nlisti = this.bopTotal[i];
        const base = this.bopIndex[i];
        let count = 0;
        for (let jj = 0; jj < nlisti; jj++) {
          const j = this.nbr1[base + jj];
          const jtype = map[type[j]];
          for (let kk = jj + 1; kk < nlisti; kk++) {
            const k = this.nbr1[base + kk];
            const ktype = map[type[k]];
            const paramJIK = pot.elem3param[(jtype * nb + itype) * nb + ktype];
            const t = cosTotal + count;
            this.growT(t + 1);
            const cos = this.angleSlots(base + jj, base + kk);
            this.tCos[t] = cos;
            this.tDj.set(this.angDj, 3 * t);
            this.tDk.set(this.angDk, 3 * t);
            const tab = pot.triplets[paramJIK];
            this.tG[t] = tab.value(cos);
            this.tDG[t] = tab.dy;
            count++;
          }
        }
        cosTotal += count;
      }
    }
  }

  protected grow1(n: number): void {
    if (this.nbr1.length >= n) return;
    const m = Math.max(n, 2 * this.nbr1.length);
    this.nbr1 = grow(this.nbr1, m);
    this.r1 = grow(this.r1, m);
    this.bS1 = grow(this.bS1, m);
    this.dbS1 = grow(this.dbS1, m);
    this.bP1 = grow(this.bP1, m);
    this.dbP1 = grow(this.dbP1, m);
    this.rep1 = grow(this.rep1, m);
    this.drep1 = grow(this.drep1, m);
    this.d1 = grow(this.d1, 3 * m);
  }

  protected grow2(n: number): void {
    if (this.nbr2.length >= n) return;
    const m = Math.max(n, 2 * this.nbr2.length);
    this.nbr2 = grow(this.nbr2, m);
    this.r2 = grow(this.r2, m);
    this.rep2 = grow(this.rep2, m);
    this.drep2 = grow(this.drep2, m);
    this.d2 = grow(this.d2, 3 * m);
  }

  protected growT(n: number): void {
    if (this.tCos.length >= n) return;
    const m = Math.max(n, 2 * this.tCos.length);
    this.tCos = grow(this.tCos, m);
    this.tG = grow(this.tG, m);
    this.tDG = grow(this.tDG, m);
    this.tDj = grow(this.tDj, 3 * m);
    this.tDk = grow(this.tDk, 3 * m);
  }

  /**
   * PairBOP::angle: the cosine of the angle at the centre between two neighbour slots (given as indices into the
   * pairlist1 arrays) and its derivatives with respect to the two neighbour positions (angDj, angDk).
   */
  protected angleSlots(ia: number, ib: number): number {
    const r1 = this.r1[ia], r2 = this.r1[ib];
    const d1 = this.d1, d2 = this.d1;
    const a0 = d1[3 * ia], a1 = d1[3 * ia + 1], a2 = d1[3 * ia + 2];
    const b0 = d2[3 * ib], b1 = d2[3 * ib + 1], b2 = d2[3 * ib + 2];
    const rj1k1 = r1 * r2;
    const rj2k1 = rj1k1 * r1;
    const rj1k2 = rj1k1 * r2;
    const ang = (a0 * b0 + a1 * b1 + a2 * b2) / rj1k1;
    this.angDj[0] = (b0 * r1 - ang * a0 * r2) / rj2k1;
    this.angDj[1] = (b1 * r1 - ang * a1 * r2) / rj2k1;
    this.angDj[2] = (b2 * r1 - ang * a2 * r2) / rj2k1;
    this.angDk[0] = (a0 * r2 - ang * b0 * r1) / rj1k2;
    this.angDk[1] = (a1 * r2 - ang * b1 * r1) / rj1k2;
    this.angDk[2] = (a2 * r2 - ang * b2 * r1) / rj1k2;
    return ang;
  }

  /**
   * The triplet centred on atom c with the neighbours at BOP slots sa and sb: the cosine, G and dG (when needG), and
   * the derivatives for sa (out.a) and sb (out.b). With save the values come from the table of gneigh, which stores
   * each pair once in the native index order; the other order swaps the two derivatives.
   */
  protected trip(pot: BopPotential, c: number, sa: number, sb: number, out: Trip, needG: boolean): void {
    const nb = pot.nb;
    const base = this.bopIndex[c];
    const ia = base + sa, ib = base + sb;
    const tc = this.map[this.type[c]];
    if (!this.save) {
      out.cos = this.angleSlots(ia, ib);
      out.a.set(this.angDj);
      out.b.set(this.angDk);
      if (needG) {
        const ta = this.map[this.type[this.nbr1[ia]]];
        const tb = this.map[this.type[this.nbr1[ib]]];
        const tab = pot.triplets[pot.elem3param[(ta * nb + tc) * nb + tb]];
        out.g = tab.value(out.cos);
        out.dg = tab.dy;
      }
      return;
    }
    const nl = this.bopTotal[c];
    const lo = sa < sb ? sa : sb, hi = sa < sb ? sb : sa;
    const n = this.cosIndex[c] + (lo * (2 * nl - lo - 1)) / 2 + (hi - lo) - 1;
    out.cos = this.tCos[n];
    out.g = this.tG[n];
    out.dg = this.tDG[n];
    if (sa < sb) {
      out.a.set(this.tDj.subarray(3 * n, 3 * n + 3));
      out.b.set(this.tDk.subarray(3 * n, 3 * n + 3));
    } else {
      out.a.set(this.tDk.subarray(3 * n, 3 * n + 3));
      out.b.set(this.tDj.subarray(3 * n, 3 * n + 3));
    }
  }

  /** Lazily sized derivative-bookkeeping lists (native memory_sg / memory_pi). */
  protected newSg(i: number, j: number, temp: number): BSg {
    if (this.nSg >= this.sg.length) this.sg.push(new BSg());
    const e = this.sg[this.nSg++];
    e.reset();
    e.i = i; e.j = j; e.temp = temp;
    return e;
  }

  protected newPi(i: number, j: number, temp: number): BPi {
    if (this.nBi >= this.bi.length) this.bi.push(new BPi());
    const e = this.bi[this.nBi++];
    e.reset();
    e.i = i; e.j = j; e.temp = temp;
    return e;
  }

  /** The index of the first sg entry of this call is 0 (the native nb_t = 0 at each call). */
  protected sigmaBo(pot: BopPotential, i: number, jtmp: number): number {
    const f = this.f, type = this.type, map = this.map, nb = pot.nb;
    const bopIndex = this.bopIndex, bopTotal = this.bopTotal, nbr1 = this.nbr1, r1 = this.r1, d1 = this.d1;
    const bS1 = this.bS1, dbS1 = this.dbS1;
    this.nSg = 0;
    let sigB = 0.0;

    const itype = map[type[i]];
    const nlisti = bopTotal[i];
    const tIJ = bopIndex[i] + jtmp;
    const j = nbr1[tIJ];
    const jtype = map[type[j]];
    const nlistj = bopTotal[j];
    const paramIJ = pot.elem2param[itype * nb + jtype];
    const p = pot.pairs[paramIJ];
    const sigmaF = pot.sigmaF[paramIJ], sigmaK = pot.sigmaK[paramIJ];
    const sigmaC = pot.sigmaC[paramIJ], sigmaDelta = pot.sigmaDelta[paramIJ];
    const small4 = pot.small[3], small5 = pot.small[4];
    const nbIJ = this.newSg(i, j, tIJ);

    let nJi = -1;
    for (let loop = 0; loop < nlistj; loop++) {
      const nei = nbr1[bopIndex[j] + loop];
      if (nei === i) {
        nJi = loop;
        break;
      }
    }
    if (nJi < 0) throw new StyleError('BOP neighbor list is inconsistent');

    const dis0 = d1[3 * tIJ], dis1 = d1[3 * tIJ + 1], dis2 = d1[3 * tIJ + 2];
    const rIJ = r1[tIJ];
    const betaSIJ = bS1[tIJ];
    const dBetaSIJ = dbS1[tIJ];

    let AA = 0.0;
    let BB = 0.0;
    let EE1 = 0.0;

    const FF = betaSIJ * betaSIJ;
    if (FF <= 0.000001) return sigB;

    let agpdpr1 = (2.0 * betaSIJ * dBetaSIJ) / rIJ;
    nbIJ.dFF[0] = agpdpr1 * dis0;
    nbIJ.dFF[1] = agpdpr1 * dis1;
    nbIJ.dFF[2] = agpdpr1 * dis2;

    for (let ktmp = 0; ktmp < nlisti; ktmp++) {
      if (ktmp === jtmp) continue;
      const tIK = bopIndex[i] + ktmp;
      const k = nbr1[tIK];
      const ktype = map[type[k]];
      const nlistk = bopTotal[k];
      const disIK0 = d1[3 * tIK], disIK1 = d1[3 * tIK + 1], disIK2 = d1[3 * tIK + 2];
      const rIK = r1[tIK];
      const betaSIK = bS1[tIK];
      const dBetaSIK = dbS1[tIK];

      let nfound = 0;
      let passJk = 0;
      let nKi = -1;
      let nKj = -1;
      for (let loop = 0; loop < nlistk; loop++) {
        const nei = nbr1[bopIndex[k] + loop];
        if (nei === i) {
          nKi = loop;
          nfound++;
          if (nfound === 2) break;
        }
        if (nei === j) {
          nKj = loop;
          passJk = 1;
          nfound++;
          if (nfound === 2) break;
        }
      }
      if (nKi < 0) throw new StyleError('BOP neighbor list is inconsistent');

      const nbIK = this.newSg(i, k, tIK);
      let tJK = -1;
      let nJk = -1;
      let nbJK: BSg | null = null;
      if (passJk) {
        for (let loop = 0; loop < nlistj; loop++) {
          const tl = bopIndex[j] + loop;
          if (nbr1[tl] === k) {
            tJK = tl;
            nJk = loop;
            break;
          }
        }
        if (tJK < 0 || nJk < 0) throw new StyleError('BOP neighbor list is inconsistent');
        nbJK = this.newSg(j, k, tJK);
      }

      // triplet j-i-k, centred on i (jtmp, ktmp)
      const tJik = this.tJik;
      this.trip(pot, i, jtmp, ktmp, tJik, true);
      const gfactor1 = tJik.g, gprime1 = tJik.dg;
      const gfactorsq = gfactor1 * gfactor1;
      const gsqprime = 2.0 * gfactor1 * gprime1;
      AA += gfactorsq * betaSIK * betaSIK;
      agpdpr1 = (2.0 * gfactorsq * betaSIK * dBetaSIK) / rIK;
      const app1 = betaSIK * betaSIK * gsqprime;
      for (let c = 0; c < 3; c++) {
        nbIJ.dAA[c] += app1 * tJik.a[c];
        nbIK.dAA[c] += app1 * tJik.b[c] + agpdpr1 * (c === 0 ? disIK0 : c === 1 ? disIK1 : disIK2);
      }

      if (sigmaF === 0.5 || sigmaK === 0.0 || !passJk) continue;
      const disJk0 = d1[3 * tJK], disJk1 = d1[3 * tJK + 1], disJk2 = d1[3 * tJK + 2];
      const rJk = r1[tJK];
      const betaSJK = bS1[tJK];
      const dBetaSJK = dbS1[tJK];

      // triplet i-j-k centred on j (n_ji, n_jk); triplet i-k-j centred on k (n_ki, n_kj)
      const tIjk = this.tIjk;
      this.trip(pot, j, nJi, nJk, tIjk, true);
      const gfactor2 = tIjk.g, gprime2 = tIjk.dg;
      const tIkj = this.tIkj;
      this.trip(pot, k, nKi, nKj, tIkj, true);
      const gfactor3 = tIkj.g, gprime3 = tIkj.dg;
      const gfactor = gfactor1 * gfactor2 * gfactor3;
      const rfactor = betaSIK * betaSJK;

      EE1 += gfactor * rfactor;

      const rcm1 = (gfactor * betaSJK * dBetaSIK) / rIK;
      const rcm2 = (gfactor * betaSIK * dBetaSJK) / rJk;
      const gcm1 = rfactor * gprime1 * gfactor2 * gfactor3;
      const gcm2 = rfactor * gfactor1 * gprime2 * gfactor3;
      const gcm3 = rfactor * gfactor1 * gfactor2 * gprime3;
      for (let c = 0; c < 3; c++) {
        const disIK = c === 0 ? disIK0 : c === 1 ? disIK1 : disIK2;
        const disJK = c === 0 ? disJk0 : c === 1 ? disJk1 : disJk2;
        nbIJ.dEE1[c] += gcm1 * tJik.a[c] - gcm2 * tIjk.a[c];
        nbIK.dEE1[c] += gcm1 * tJik.b[c] + rcm1 * disIK - gcm3 * tIkj.a[c];
        nbJK!.dEE1[c] += gcm2 * tIjk.b[c] + rcm2 * disJK - gcm3 * tIkj.b[c];
      }
    }

    for (let ktmp = 0; ktmp < nlistj; ktmp++) {
      if (ktmp === nJi) continue;
      const tJK = bopIndex[j] + ktmp;
      const k = nbr1[tJK];
      const disJk0 = d1[3 * tJK], disJk1 = d1[3 * tJK + 1], disJk2 = d1[3 * tJK + 2];
      const rJk = r1[tJK];
      const betaSJK = bS1[tJK];
      const dBetaSJK = dbS1[tJK];
      const nbJK = this.newSg(j, k, tJK);

      const tIjk = this.tIjk;
      this.trip(pot, j, nJi, ktmp, tIjk, true);
      const gfactor1 = tIjk.g, gprime1 = tIjk.dg;
      const gfactorsq = gfactor1 * gfactor1;
      const gsqprime = 2.0 * gfactor1 * gprime1;
      BB += gfactorsq * betaSJK * betaSJK;
      agpdpr1 = (2.0 * gfactorsq * betaSJK * dBetaSJK) / rJk;
      const app1 = betaSJK * betaSJK * gsqprime;
      for (let c = 0; c < 3; c++) {
        nbIJ.dBB[c] -= app1 * tIjk.a[c];
        nbJK.dBB[c] += app1 * tIjk.b[c] + agpdpr1 * (c === 0 ? disJk0 : c === 1 ? disJk1 : disJk2);
      }
    }

    const AAC = AA + BB;
    for (let loop = 0; loop < this.nSg; loop++) {
      const e = this.sg[loop];
      for (let c = 0; c < 3; c++) e.dAAC[c] = e.dAA[c] + e.dBB[c];
    }
    let bndtmp = FF + sigmaDelta * sigmaDelta + sigmaC * AAC + small4;
    const bndtmp0 = 1.0 / Math.sqrt(bndtmp);
    const sigB1 = betaSIJ * bndtmp0;

    bndtmp = -0.5 * bndtmp0 * bndtmp0 * bndtmp0;
    const bndtmp1 = ((bndtmp0 + betaSIJ * bndtmp * 2.0 * betaSIJ) * dBetaSIJ) / rIJ;
    const bndtmp2 = betaSIJ * bndtmp * sigmaC;
    for (let loop = 0; loop < this.nSg; loop++) {
      const e = this.sg[loop];
      for (let c = 0; c < 3; c++) e.dSigB1[c] = bndtmp2 * e.dAAC[c];
      if (e.temp === tIJ) {
        e.dSigB1[0] += bndtmp1 * dis0;
        e.dSigB1[1] += bndtmp1 * dis1;
        e.dSigB1[2] += bndtmp1 * dis2;
      }
    }

    // sigB is the final expression for Eq. 6 (a) and Eq. 11 (b)
    sigB = p.bo.value(sigB1);
    const dsigB = p.bo.dy;
    let part2 = 0, part3 = 0, part4 = 0;
    const partsOn = sigmaF !== 0.5 && sigmaK !== 0.0;
    if (partsOn) {
      const part0 = FF + 0.5 * AAC + small5;
      const part1 = (sigmaF - 0.5) * sigmaK;
      part2 = 1.0 - (part1 * EE1) / part0;
      part3 = (sigB * part1) / part0;
      part4 = (part3 / part0) * EE1;
      sigB *= part2;
    }

    const pp1 = 2.0 * betaSIJ;
    for (let loop = 0; loop < this.nSg; loop++) {
      const e = this.sg[loop];
      const bi = e.i, bj = e.j;
      if (!partsOn) {
        for (let c = 0; c < 3; c++) e.dSigB[c] = dsigB * e.dSigB1[c];
      } else {
        for (let c = 0; c < 3; c++) {
          e.dSigB[c] = dsigB * part2 * e.dSigB1[c] - part3 * e.dEE1[c] + part4 * (e.dFF[c] + 0.5 * e.dAAC[c]);
        }
      }
      for (let c = 0; c < 3; c++) {
        const ftmp = pp1 * e.dSigB[c];
        f[3 * bi + c] -= ftmp;
        f[3 * bj + c] += ftmp;
      }
    }
    return sigB;
  }

  /** PiBo: Theta_pi,ij of Eq. 36 (a) and Eq. 18 (b), with the forces it adds. */
  protected piBo(pot: BopPotential, i: number, jtmp: number): number {
    const f = this.f, type = this.type, map = this.map, nb = pot.nb;
    const bopIndex = this.bopIndex, bopTotal = this.bopTotal, nbr1 = this.nbr1, r1 = this.r1, d1 = this.d1;
    const bS1 = this.bS1, dbS1 = this.dbS1, bP1 = this.bP1, dbP1 = this.dbP1;
    this.nBi = 0;

    const itype = map[type[i]];
    const tIJ = bopIndex[i] + jtmp;
    const j = nbr1[tIJ];
    const jtype = map[type[j]];
    const nlisti = bopTotal[i];
    const nlistj = bopTotal[j];
    const paramIJ = pot.elem2param[itype * nb + jtype];
    const piP = pot.piP, piA = pot.piA[paramIJ], piC = pot.piC[paramIJ], piDelta = pot.piDelta[paramIJ];
    const small6 = pot.small[5], small7 = pot.small[6];
    const nbIJ = this.newPi(i, j, tIJ);

    let nJi = -1;
    for (let loop = 0; loop < nlistj; loop++) {
      if (nbr1[bopIndex[j] + loop] === i) {
        nJi = loop;
        break;
      }
    }
    if (nJi < 0) throw new StyleError('BOP neighbor list is inconsistent');

    const dis0 = d1[3 * tIJ], dis1 = d1[3 * tIJ + 1], dis2 = d1[3 * tIJ + 2];
    const rIJ = r1[tIJ];
    const betaPIJ = bP1[tIJ];
    const dBetaPIJ = dbP1[tIJ];

    let AA = 0.0;
    let BB = 0.0;

    const pI = piP[itype];
    const tJik = this.tJik, tJikp = this.tJikp, tKikp = this.tKikp;
    for (let ktmp = 0; ktmp < nlisti; ktmp++) {
      if (ktmp === jtmp) continue;
      const tIK = bopIndex[i] + ktmp;
      const k = nbr1[tIK];
      const disIK0 = d1[3 * tIK], disIK1 = d1[3 * tIK + 1], disIK2 = d1[3 * tIK + 2];
      const rIK = r1[tIK];
      const betaSIK = bS1[tIK];
      const dBetaSIK = dbS1[tIK];
      const betaPIK = bP1[tIK];
      const dBetaPIK = dbP1[tIK];

      const nbIK = this.newPi(i, k, tIK);

      this.trip(pot, i, jtmp, ktmp, tJik, false);
      const cosAngJik = tJik.cos;
      const cosSq = cosAngJik * cosAngJik;
      const sinFactor = 0.5 * (1.0 - cosSq) * pI * betaSIK;
      const cosFactor = 0.5 * (1.0 + cosSq) * betaPIK;
      const betaCapSq1 = pI * betaSIK * betaSIK - betaPIK * betaPIK;
      const dbetaCapSq1 = 2.0 * pI * betaSIK * dBetaSIK - 2.0 * betaPIK * dBetaPIK;

      AA = AA + sinFactor * betaSIK + cosFactor * betaPIK;
      BB = BB + 0.25 * (1.0 - cosSq) * (1.0 - cosSq) * betaCapSq1 * betaCapSq1;

      let agpdpr1 = (2.0 * sinFactor * dBetaSIK + 2.0 * cosFactor * dBetaPIK) / rIK;
      const agpdpr2 = (0.5 * (1.0 - cosSq) * (1.0 - cosSq) * betaCapSq1 * dbetaCapSq1) / rIK;
      const app1 = cosAngJik * (betaPIK * betaPIK - pI * betaSIK * betaSIK);
      const app2 = (cosSq - 1.0) * cosAngJik * betaCapSq1 * betaCapSq1;
      for (let c = 0; c < 3; c++) {
        const disIK = c === 0 ? disIK0 : c === 1 ? disIK1 : disIK2;
        nbIJ.dAA[c] += app1 * tJik.a[c];
        nbIJ.dBB[c] += app2 * tJik.a[c];
        nbIK.dAA[c] += agpdpr1 * disIK + app1 * tJik.b[c];
        nbIK.dBB[c] += agpdpr2 * disIK + app2 * tJik.b[c];
      }

      for (let ltmp = 0; ltmp < ktmp; ltmp++) {
        if (ltmp === jtmp) continue;
        const tIKp = bopIndex[i] + ltmp;
        const kp = nbr1[tIKp];
        const disIKp0 = d1[3 * tIKp], disIKp1 = d1[3 * tIKp + 1], disIKp2 = d1[3 * tIKp + 2];
        const rIKp = r1[tIKp];
        const betaSIKp = bS1[tIKp];
        const dBetaSIKp = dbS1[tIKp];
        const betaPIKp = bP1[tIKp];
        const dBetaPIKp = dbP1[tIKp];

        const nbIKp = this.newPi(i, kp, tIKp);

        this.trip(pot, i, ktmp, ltmp, tKikp, false);
        this.trip(pot, i, jtmp, ltmp, tJikp, false);
        const cosAngKikp = tKikp.cos;
        const cosAngJikp = tJikp.cos;

        const betaCapSq2 = pI * betaSIKp * betaSIKp - betaPIKp * betaPIKp;
        const dbetaCapSq2 = 2.0 * pI * betaSIKp * dBetaSIKp - 2.0 * betaPIKp * dBetaPIKp;
        const cosSq1 = cosAngJikp * cosAngJikp;
        const angFactor = cosAngKikp - cosAngJikp * cosAngJik;
        const angFactor1 = 4.0 * angFactor;
        const angFactor2 = -angFactor1 * cosAngJikp + 2.0 * cosAngJik * (1.0 - cosSq1);
        const angFactor3 = -angFactor1 * cosAngJik + 2.0 * cosAngJikp * (1.0 - cosSq);
        const angFactor4 = 2.0 * angFactor * angFactor - (1.0 - cosSq) * (1.0 - cosSq1);
        const betaCapSum = 0.5 * betaCapSq1 * betaCapSq2;

        BB += betaCapSum * angFactor4;

        const agpdpr1 = (0.5 * angFactor4 * dbetaCapSq1 * betaCapSq2) / rIK;
        const agpdpr2b = (0.5 * angFactor4 * betaCapSq1 * dbetaCapSq2) / rIKp;
        const app1b = betaCapSum * angFactor1;
        const app2b = betaCapSum * angFactor2;
        const app3 = betaCapSum * angFactor3;
        for (let c = 0; c < 3; c++) {
          const disIK = c === 0 ? disIK0 : c === 1 ? disIK1 : disIK2;
          const disIKp = c === 0 ? disIKp0 : c === 1 ? disIKp1 : disIKp2;
          nbIJ.dBB[c] += app2b * tJik.a[c] + app3 * tJikp.a[c];
          nbIK.dBB[c] += agpdpr1 * disIK + app1b * tKikp.a[c] + app2b * tJik.b[c];
          nbIKp.dBB[c] += agpdpr2b * disIKp + app1b * tKikp.b[c] + app3 * tJikp.b[c];
        }
      }
    }

    for (let ktmp = 0; ktmp < nlistj; ktmp++) {
      if (ktmp === nJi) continue;
      const tJK = bopIndex[j] + ktmp;
      const k = nbr1[tJK];
      const disJk0 = d1[3 * tJK], disJk1 = d1[3 * tJK + 1], disJk2 = d1[3 * tJK + 2];
      const rJk = r1[tJK];
      const betaSJK = bS1[tJK];
      const dBetaSJK = dbS1[tJK];
      const betaPJK = bP1[tJK];
      const dBetaPJK = dbP1[tJK];

      const nbJK = this.newPi(j, k, tJK);

      const tIjk = this.tIjk;
      this.trip(pot, j, nJi, ktmp, tIjk, false);
      const cosAngIjk = tIjk.cos;
      const cosSq = cosAngIjk * cosAngIjk;
      const pJ = piP[jtype];
      const sinFactor = 0.5 * (1.0 - cosSq) * pJ * betaSJK;
      const cosFactor = 0.5 * (1.0 + cosSq) * betaPJK;
      const betaCapSq1 = pJ * betaSJK * betaSJK - betaPJK * betaPJK;
      const dbetaCapSq1 = 2.0 * pJ * betaSJK * dBetaSJK - 2.0 * betaPJK * dBetaPJK;

      AA += sinFactor * betaSJK + cosFactor * betaPJK;
      BB += 0.25 * (1.0 - cosSq) * (1.0 - cosSq) * betaCapSq1 * betaCapSq1;

      const agpdpr1 = (2.0 * sinFactor * dBetaSJK + 2.0 * cosFactor * dBetaPJK) / rJk;
      const agpdpr2 = (0.5 * (1.0 - cosSq) * (1.0 - cosSq) * betaCapSq1 * dbetaCapSq1) / rJk;
      const app1 = cosAngIjk * (betaPJK * betaPJK - pJ * betaSJK * betaSJK);
      const app2 = (cosSq - 1.0) * cosAngIjk * betaCapSq1 * betaCapSq1;
      for (let c = 0; c < 3; c++) {
        const disJK = c === 0 ? disJk0 : c === 1 ? disJk1 : disJk2;
        nbIJ.dAA[c] -= app1 * tIjk.a[c];
        nbIJ.dBB[c] -= app2 * tIjk.a[c];
        nbJK.dAA[c] += agpdpr1 * disJK + app1 * tIjk.b[c];
        nbJK.dBB[c] += agpdpr2 * disJK + app2 * tIjk.b[c];
      }

      for (let ltmp = 0; ltmp < ktmp; ltmp++) {
        if (ltmp === nJi) continue;
        const tJKp = bopIndex[j] + ltmp;
        const kp = nbr1[tJKp];
        const disJkp0 = d1[3 * tJKp], disJkp1 = d1[3 * tJKp + 1], disJkp2 = d1[3 * tJKp + 2];
        const rJkp = r1[tJKp];
        const betaSJKp = bS1[tJKp];
        const dBetaSJKp = dbS1[tJKp];
        const betaPJKp = bP1[tJKp];
        const dBetaPJKp = dbP1[tJKp];

        const nbJKp = this.newPi(j, kp, tJKp);

        this.trip(pot, j, ktmp, ltmp, this.tKjkp, false);
        this.trip(pot, j, nJi, ltmp, this.tIjkp, false);
        const cosAngKjkp = this.tKjkp.cos;
        const cosAngIjkp = this.tIjkp.cos;

        const betaCapSq2 = pJ * betaSJKp * betaSJKp - betaPJKp * betaPJKp;
        const dbetaCapSq2 = 2.0 * pJ * betaSJKp * dBetaSJKp - 2.0 * betaPJKp * dBetaPJKp;
        const cosSq1 = cosAngIjkp * cosAngIjkp;
        const angFactor = cosAngKjkp - cosAngIjkp * cosAngIjk;
        const angFactor1 = 4.0 * angFactor;
        const angFactor2 = -angFactor1 * cosAngIjkp + 2.0 * cosAngIjk * (1.0 - cosSq1);
        const angFactor3 = -angFactor1 * cosAngIjk + 2.0 * cosAngIjkp * (1.0 - cosSq);
        const angFactor4 = 2.0 * angFactor * angFactor - (1.0 - cosSq) * (1.0 - cosSq1);
        const betaCapSum = 0.5 * betaCapSq1 * betaCapSq2;

        BB += betaCapSum * angFactor4;

        const agpdprA = (0.5 * angFactor4 * dbetaCapSq1 * betaCapSq2) / rJk;
        const agpdprB = (0.5 * angFactor4 * betaCapSq1 * dbetaCapSq2) / rJkp;
        const app1b = betaCapSum * angFactor1;
        const app2b = betaCapSum * angFactor2;
        const app3 = betaCapSum * angFactor3;
        for (let c = 0; c < 3; c++) {
          const disJKp = c === 0 ? disJkp0 : c === 1 ? disJkp1 : disJkp2;
          const disJK = c === 0 ? disJk0 : c === 1 ? disJk1 : disJk2;
          nbIJ.dBB[c] -= app3 * this.tIjkp.a[c] + app2b * tIjk.a[c];
          nbJK.dBB[c] += agpdprA * disJK + app1b * this.tKjkp.a[c] + app2b * tIjk.b[c];
          nbJKp.dBB[c] += agpdprB * disJKp + app1b * this.tKjkp.b[c] + app3 * this.tIjkp.b[c];
        }
      }

      for (let ltmp = 0; ltmp < nlisti; ltmp++) {
        if (ltmp === jtmp) continue;
        const tIKp = bopIndex[i] + ltmp;
        const kp = nbr1[tIKp];
        const disIKp0 = d1[3 * tIKp], disIKp1 = d1[3 * tIKp + 1], disIKp2 = d1[3 * tIKp + 2];
        const rIKp = r1[tIKp];
        const betaSIKp = bS1[tIKp];
        const dBetaSIKp = dbS1[tIKp];
        const betaPIKp = bP1[tIKp];
        const dBetaPIKp = dbP1[tIKp];

        const nbIKp = this.newPi(i, kp, tIKp);

        this.trip(pot, i, jtmp, ltmp, tJikp, false);
        const cosAngJikp = tJikp.cos;

        const betaCapSq2 = pI * betaSIKp * betaSIKp - betaPIKp * betaPIKp;
        const dbetaCapSq2 = 2.0 * pI * betaSIKp * dBetaSIKp - 2.0 * betaPIKp * dBetaPIKp;
        const dotV = (disJk0 * disIKp0 + disJk1 * disIKp1 + disJk2 * disIKp2) / (rJk * rIKp);
        const cosSq1 = cosAngJikp * cosAngJikp;
        const angFactor = dotV + cosAngJikp * cosAngIjk;
        const angRfactor = 4.0 * angFactor * dotV;
        const dAngR1 = -angRfactor / rJk;
        const dAngR2 = -angRfactor / rIKp;
        const angFactor1 = 4.0 * angFactor * cosAngJikp + 2.0 * cosAngIjk * (1.0 - cosSq1);
        const angFactor2 = 4.0 * angFactor * cosAngIjk + 2.0 * cosAngJikp * (1.0 - cosSq);
        const angFactor3 = 2.0 * angFactor * angFactor - (1.0 - cosSq) * (1.0 - cosSq1);
        const betaCapSum = 0.5 * betaCapSq1 * betaCapSq2;

        BB += betaCapSum * angFactor3;

        const agpdpr1 = (0.5 * angFactor3 * dbetaCapSq1 * betaCapSq2 + betaCapSum * dAngR1) / rJk;
        const agpdpr2 = (0.5 * angFactor3 * betaCapSq1 * dbetaCapSq2 + betaCapSum * dAngR2) / rIKp;
        const agpdpr3 = (4.0 * betaCapSum * angFactor) / (rIKp * rJk);
        const app1 = betaCapSum * angFactor1;
        const app2 = betaCapSum * angFactor2;
        for (let c = 0; c < 3; c++) {
          const disIKp = c === 0 ? disIKp0 : c === 1 ? disIKp1 : disIKp2;
          const disJK = c === 0 ? disJk0 : c === 1 ? disJk1 : disJk2;
          nbIJ.dBB[c] += app2 * tJikp.a[c] - app1 * tIjk.a[c];
          nbIKp.dBB[c] += agpdpr2 * disIKp + agpdpr3 * disJK + app2 * tJikp.b[c];
          nbJK.dBB[c] += agpdpr1 * disJK + agpdpr3 * disIKp + app1 * tIjk.b[c];
        }
      }
    }

    const pp2 = 2.0 * betaPIJ;
    const CC = betaPIJ * betaPIJ + piDelta * piDelta;
    const BBrt = Math.sqrt(BB + small6);
    const AB1 = CC + piC * (AA + BBrt) + small7;
    const AB2 = CC + piC * (AA - BBrt + Math.sqrt(small6)) + small7;
    const BBrtR = 1.0 / BBrt;
    const ABrtR1 = 1.0 / Math.sqrt(AB1);
    const ABrtR2 = 1.0 / Math.sqrt(AB2);
    const piB = (ABrtR1 + ABrtR2) * piA * betaPIJ;

    const dPiB1 = -0.5 * (libmPow(ABrtR1, 3) + libmPow(ABrtR2, 3)) * piC * piA * betaPIJ;
    const dPiB2 = 0.25 * BBrtR * (libmPow(ABrtR2, 3) - libmPow(ABrtR1, 3)) * piC * piA * betaPIJ;
    const dPiB3 = (((ABrtR1 + ABrtR2) * piA - (libmPow(ABrtR1, 3) + libmPow(ABrtR2, 3)) * piA * betaPIJ * betaPIJ) *
      dBetaPIJ) / rIJ;
    for (let loop = 0; loop < this.nBi; loop++) {
      const e = this.bi[loop];
      for (let c = 0; c < 3; c++) e.dPiB[c] = dPiB1 * e.dAA[c] + dPiB2 * e.dBB[c];
      if (e.temp === tIJ) {
        e.dPiB[0] += dPiB3 * dis0;
        e.dPiB[1] += dPiB3 * dis1;
        e.dPiB[2] += dPiB3 * dis2;
      }
    }
    for (let loop = 0; loop < this.nBi; loop++) {
      const e = this.bi[loop];
      for (let c = 0; c < 3; c++) {
        const ftmp = pp2 * e.dPiB[c];
        f[3 * e.i + c] -= ftmp;
        f[3 * e.j + c] += ftmp;
      }
    }
    return piB;
  }
}
