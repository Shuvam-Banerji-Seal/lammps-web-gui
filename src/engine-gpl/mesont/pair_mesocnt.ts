/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MESONT/pair_mesocnt.cpp and src/MESONT/pair_mesocnt.h at commit b332f28 (contributing author Philipp
 * Kloza, University of Cambridge; the mesoscopic force field of Volkov and Zhigilei, J Phys Chem C 114, 5513 (2010)).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * pair_style mesocnt cutoff [chain|segment] [id|topology]: the van der Waals interaction between the flexible CNT
 * segments. Each bond of the CNT chains is one segment; for every segment, the connected chains of atoms within the
 * neighbor list of its two atoms (the chain is the run of consecutive atom IDs of one molecule among the full
 * neighbors of the segment atoms) are replaced by their local straight infinite or semi-infinite CNT, and the
 * segment pair interaction is summed over the chain segments (with the weights of PairMesoCNT::weight).
 *
 * The topology is rebuilt on each neighbor-list build (native: when update->ntimestep == neighbor->lastcall), the
 * chain atoms are stored as indices of owned or ghost atoms (as the C++ neighbor-list indices) and their positions
 * follow the ghosts every step. The per-atom "buckled" flag of angle_style mesocnt marks the chains that are always
 * computed segment by segment. The Gauss-Legendre quadrature, splines and geometry are in ./tables and ./forces.
 *
 * pair_coeff * * FILE end-type...: the potential file (its tables are read at pair_coeff, native read_file) and the
 * atom types that end a chain. The keyword "topology" (bond topology instead of atom IDs for the chains) is refused.
 */
import { Pair, PairCompute, StyleError, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { ANGSTROM } from '../../engine/force/kspace/ewald';
import { parseNum } from '../../engine/force/util';
import { BUCKLED } from './angle_mesocnt';
import { eunitOf } from './units';
import { finf, fsemi, geometry, mesoConst, newGeoOut, newWeightOut, weight, type GeoOut, type MesoConst, type WeightOut } from './forces';
import { readMesocntFile, sCubic, type MesocntTables } from './tables';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** LAMMPS: static constexpr int SELF_CUTOFF = 3 (pair_mesocnt.cpp). */
const SELF_CUTOFF = 3;

/** One connected chain of a segment: indices of owned or ghost atoms (in order) and its end flag (native end[j]). */
interface Chain {
  idx: Int32Array;
  end: number;
}

/** Per bond (segment) topology: its two atoms and the chains of its neighborhood (native bondlist, chainlist). */
interface Segment {
  i1: number;
  i2: number;
  chains: Chain[];
  selfId: number;
  selfPos0: number;
  selfPos1: number;
}

/** Vector of three (scratch). */
type V3 = Float64Array;
const v3 = (): V3 => new Float64Array(3);

const cross = (a: ArrayLike<number>, b: ArrayLike<number>, out: V3): void => {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x; out[1] = y; out[2] = z;
};

/** matvec(ex, ey, ez, v): sum_a v[a] * basis[a] (the basis vectors are rows of the 9-array). */
const matvecBasis = (basis: Float64Array, v: ArrayLike<number>, out: V3): void => {
  for (let k = 0; k < 3; k++) out[k] = basis[k] * v[0] + basis[3 + k] * v[1] + basis[6 + k] * v[2];
};

/** outer3(a, b): the 3x3 matrix a b^T, row-major. */
const outer = (a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array): void => {
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[3 * r + c] = a[r] * b[c];
};

/** transpose_matvec(M, v) = M^T v for the row-major 3x3 matrix M. */
const transposeMatvec = (M: Float64Array, v: ArrayLike<number>, out: V3): void => {
  for (let r = 0; r < 3; r++) out[r] = M[r] * v[0] + M[3 + r] * v[1] + M[6 + r] * v[2];
};

/** Friction parameters of pair_style mesocnt/viscous (pair_coeff: fvisc_max kvisc vvisc, and fvisc_shift). */
export interface ViscParams {
  fviscMax: number;
  kvisc: number;
  vvisc: number;
  fviscShift: number;
}

/**
 * The friction magnitude of a chain (native: fvisc_tot in the viscous compute): fvisc_max / (1 + exp(-kvisc (|v| - vvisc)))
 * - fvisc_shift, scaled by 0.25 (1 - s(sumw)) (param[3] - param[2]), signed by the relative velocity; zero for vtot = 0.
 */
const frictionTotal = (v: ViscParams, vtot: number, sumw: number, param: Float64Array): number => {
  if (vtot === 0.0) return 0.0;
  let ft = v.fviscMax / (1.0 + Math.exp(-v.kvisc * (Math.abs(vtot) - v.vvisc))) - v.fviscShift;
  ft *= 0.25 * (1 - sCubic(sumw)) * (param[3] - param[2]);
  if (vtot < 0) ft = -ft;
  return ft;
};

/** PairMesoCNTViscous::weight2: the friction weight of a chain segment (the cubic switch of the mid-distance). */
const weight2 = (c: MesoConst, r1: ArrayLike<number>, r2: ArrayLike<number>, p1: ArrayLike<number>, p2: ArrayLike<number>): number => {
  const distsq = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  };
  const r = [r1[0] + r2[0], r1[1] + r2[1], r1[2] + r2[2]];
  const p = [p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]];
  const dr = Math.sqrt(0.25 * distsq(r1, r2) + c.rsq);
  const dp = Math.sqrt(0.25 * distsq(p1, p2) + c.rsq);
  const rhoc = dr + dp + c.rc;
  const rhomin = 10.0 * c.ang;
  const rho = 0.5 * Math.sqrt(distsq(r, p));
  return sCubic((rho - rhomin) / (rhoc - rhomin));
};

export class PairMesoCNT extends Pair {
  readonly name: string = 'mesocnt';
  /** The global virial is the dot product of positions and forces (native: no_virial_fdotr_compute = 0). */
  virialFdotr = true;
  /** Native: manybody_flag 1 (a full neighbor list; special bonds are not applied to the chains). */
  needsFull = true;
  needsHalf = false;
  manybody = true;

  private neighCutoff = 0;
  private segmentFlag = false;
  private tables: MesocntTables | null = null;
  private consts: MesoConst | null = null;
  private endTypes: number[] = [];
  /** Segments of the last topology build, and the neighbor-build count it was made at. */
  private segments: Segment[] = [];
  private topoBuild = -1;
  private readonly geo: GeoOut = newGeoOut();
  /** Viscous friction parameters (pair_style mesocnt/viscous); null for the plain style. */
  protected visc: ViscParams | null = null;
  /** Persistent per-chain weights (native member w): the viscous friction reads entries the chain loop skipped. */
  protected wArr = new Float64Array(0);
  /** Longest chain of the last topology build. */
  private clenMax = 0;
  private readonly fl = new Float64Array(6);
  private readonly fend = { fend: 0 };
  private readonly wo: WeightOut = newWeightOut();

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length < 1) throw new StyleError('Illegal pair_style mesocnt command: needs a cutoff');
    if (args.length > 3) throw new StyleError('Too many arguments in pair_style mesocnt command');
    this.neighCutoff = parseNum(args[0], 'cutoff');
    this.segmentFlag = false;
    if (args.length > 1) {
      if (args[1] === 'segment') this.segmentFlag = true;
      else if (args[1] === 'chain') this.segmentFlag = false;
      else throw new StyleError(`Unknown second argument ${args[1]} in pair_style mesocnt command, must be 'chain' or 'segment'`);
    }
    if (args.length > 2) {
      if (args[2] === 'topology') throw new StyleError("pair_style mesocnt keyword 'topology' is not supported by the browser engine (use 'id')");
      else if (args[2] !== 'id') throw new StyleError(`Unknown third argument ${args[2]} in pair_style mesocnt command, must be 'id' or 'topology'`);
    }
  }

  /** pair_coeff * * FILE end-type ... (the I and J of a manybody style are '*'). */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 4) throw new StyleError('Illegal pair_coeff command: pair_coeff * * FILE end-types');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('pair_coeff for pair style mesocnt needs * * (manybody)');
    this.visc = null;
    this.setPotential(args[2], args.slice(3), ctx);
  }

  /** Reads the potential file and sets the end atom types (the shared part of pair_coeff). */
  protected setPotential(file: string, endWords: string[], ctx: StyleContext): void {
    const style = ctx.s?.units.style;
    if (style === undefined) throw new StyleError('pair_style mesocnt needs the box and units before pair_coeff');
    const eunit = eunitOf(style, `Pair style ${this.name}`);
    const ang = ANGSTROM[style];
    this.tables = readMesocntFile(ctx.readFile(file));
    this.endTypes = endWords.map((w) => Math.trunc(parseNum(w, 'end type')));
    this.consts = mesoConst(this.tables, ang, eunit, (text) => ctx.log(`WARNING: ${text}`));
    this.topoBuild = -1;
  }

  /** init_style: lj special bonds must be non-zero and newton pair on (checked in compute and here). */
  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError('Pair style mesocnt requires newton pair on');
  }

  /** PairMesoCNT::init_one: the neighbor cutoff of the style. */
  initOne(_i: number, _j: number): number {
    return this.neighCutoff;
  }

  /**
   * Rebuilds the segments and their chains from the full neighbor list (native bond_neigh_id). Chain atoms are indices
   * of owned or ghost atoms; the chains are the consecutive IDs of one molecule in the union of the neighbor lists of
   * the two atoms of the segment, sorted by ID.
   */
  private buildTopology(pc: PairCompute): void {
    const s = pc.s, nb = pc.nb, full = pc.full;
    if (!full) throw new StyleError('pair mesocnt needs a full neighbor list');
    const nlocal = pc.nlocal;
    const owner = nb.owner, x = pc.x;
    const tagOf = (k: number): number => s.id[owner[k]];
    const molOf = (k: number): number => s.molecule[owner[k]];
    const typeOf = (k: number): number => pc.type[k];
    // owned atom index of each ID
    let maxId = 0;
    for (let i = 0; i < nlocal; i++) if (s.id[i] > maxId) maxId = s.id[i];
    const idx = new Int32Array(maxId + 1).fill(-1);
    for (let i = 0; i < nlocal; i++) idx[s.id[i]] = i;
    const nbrs = (i: number): number[] => {
      const out: number[] = [];
      const k0 = full.firstneigh[i], n = full.numneigh[i];
      for (let k = 0; k < n; k++) out.push(full.neighbors[k0 + k] & NEIGHMASK);
      return out;
    };
    const segs: Segment[] = [];
    const topo = s.topo.bonds;
    for (let b = 0; b < topo.n; b++) {
      if (topo.type[b] <= 0) continue;
      const id1 = topo.atoms[2 * b], id2 = topo.atoms[2 * b + 1];
      const i1 = idx[id1];
      const j0 = idx[id2];
      if (i1 < 0 || j0 < 0) throw new StyleError(`mesocnt bond atom ${i1 < 0 ? id1 : id2} missing`);
      // the partner: the owned atom or its periodic ghost nearest to i1
      let i2 = j0;
      let best = Infinity;
      const cand = [j0];
      for (let g = nb.ghostStart[j0]; g < nb.ghostStart[j0 + 1]; g++) cand.push(g);
      for (const c of cand) {
        const dx = x[3 * c] - x[3 * i1], dy = x[3 * c + 1] - x[3 * i1 + 1], dz = x[3 * c + 2] - x[3 * i1 + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < best) {
          best = r2;
          i2 = c;
        }
      }
      // reduced neighborhood (native neigh_common): the union of both lists, or i1 with its list when i2 is a ghost
      let red: number[];
      if (i2 >= nlocal) red = [i1, ...nbrs(i1)];
      else red = unionSorted(nbrs(i1), nbrs(i2));
      red.sort((a, c) => tagOf(a) - tagOf(c));
      const chains = splitChains(red, tagOf, molOf, typeOf, this.endTypes);
      const seg: Segment = { i1, i2, chains, selfId: 0, selfPos0: 0, selfPos1: 0 };
      // the chain and positions of the two segment atoms (native selfid, selfpos)
      const tag1 = tagOf(i1), tag2 = tagOf(i2);
      let found1 = false, found2 = false;
      for (let j = 0; j < chains.length && !(found1 && found2); j++) {
        const ch = chains[j].idx;
        for (let k = 0; k < ch.length; k++) {
          const t = tagOf(ch[k]);
          if (tag1 === t) {
            seg.selfId = j;
            seg.selfPos0 = k;
            found1 = true;
          } else if (tag2 === t) {
            seg.selfId = j;
            seg.selfPos1 = k;
            found2 = true;
          }
          if (found1 && found2) break;
        }
      }
      segs.push(seg);
    }
    this.segments = segs;
    this.clenMax = 0;
    for (const sg of segs) for (const ch of sg.chains) if (ch.idx.length > this.clenMax) this.clenMax = ch.idx.length;
  }

  compute(pc: PairCompute): void {
    if (!this.tables || !this.consts) throw new StyleError('pair_style mesocnt needs pair_coeff * * FILE');
    if (pc.vatom) throw new StyleError('compute stress/atom is not supported by pair style mesocnt');
    const sp = pc.specialLJ;
    if (sp[1] === 0 || sp[2] === 0 || sp[3] === 0) throw new StyleError('Pair mesocnt requires special_bond lj x y z to have non-zero x, y and z');
    if (this.topoBuild !== pc.nb.nbuild) {
      this.buildTopology(pc);
      this.topoBuild = pc.nb.nbuild;
    }
    const c = this.consts;
    const x = pc.x, f = pc.f;
    const buck = pc.s.custom.get(BUCKLED)?.data ?? null;
    const visc = this.visc;
    const vel = pc.s.v;
    const velOf = (k: number, d: number): number => vel[3 * owner[k] + d];
    if (this.wArr.length < this.clenMax) {
      const grown = new Float64Array(this.clenMax);
      grown.set(this.wArr);
      this.wArr = grown;
    }
    const wArr = this.wArr;
    const wp1 = v3(), wp2 = v3(), vrBond = v3(), vrel = v3(), fvisc = v3(), vmean = v3();
    const owner = pc.nb.owner;
    const eatom = pc.eatom;
    const acc = pc.acc;
    const cutoff = c.cutoff, cutoffsq = c.cutoffsq;
    const geo = this.geo;
    const param = geo.param;
    const fl = this.fl;
    const fglobal: V3[] = [v3(), v3(), v3(), v3()];
    const ftotal = v3(), ftorque = v3(), torque = v3(), t1 = v3(), t2 = v3(), t3 = v3();
    const delr1 = v3(), delr2 = v3(), delqe = v3();
    const dr1Sumw = v3(), dr2Sumw = v3();
    const p1 = v3(), p2 = v3();
    const dr1P1 = new Float64Array(9), dr1P2 = new Float64Array(9), dr2P1 = new Float64Array(9), dr2P2 = new Float64Array(9);
    const dqP1 = new Float64Array(9), dqP2 = new Float64Array(9);
    const q1dr1 = new Float64Array(9), q2dr1 = new Float64Array(9), q1dr2 = new Float64Array(9), q2dr2 = new Float64Array(9);
    const tmp = new Float64Array(9);
    const fgradR1P1 = v3(), fgradR1P2 = v3(), fgradR2P1 = v3(), fgradR2P2 = v3();
    const fgradQP1 = v3(), fgradQP2 = v3();
    const addTo = (i: number, v: ArrayLike<number>, s: number): void => {
      f[3 * i] += s * v[0];
      f[3 * i + 1] += s * v[1];
      f[3 * i + 2] += s * v[2];
    };
    const addEatom = (i: number, e: number): void => {
      if (eatom) eatom[i] += e;
    };
    const seg = this.segments;
    for (const sg of seg) {
      const i1 = sg.i1, i2 = sg.i2;
      const r1 = [x[3 * i1], x[3 * i1 + 1], x[3 * i1 + 2]];
      const r2 = [x[3 * i2], x[3 * i2 + 1], x[3 * i2 + 2]];
      if (visc) for (let d = 0; d < 3; d++) vrBond[d] = 0.5 * (velOf(i1, d) + velOf(i2, d));
      for (let j = 0; j < sg.chains.length; j++) {
        const ch = sg.chains[j];
        const clen = ch.idx.length;
        if (clen < 2) continue;
        let endflag = ch.end;
        let buckled = 0;
        if (buck) {
          for (let k = 0; k < clen; k++) {
            if (buck[owner[ch.idx[k]]]) {
              buckled = 1;
              break;
            }
          }
        }
        if (this.segmentFlag || buckled || j === sg.selfId || endflag === 3) {
          // segment-segment interaction
          let sumwS = 0;
          wp1.fill(0);
          wp2.fill(0);
          for (let k = 0; k < clen - 1; k++) {
            if (j === sg.selfId) {
              const min11 = Math.abs(k - sg.selfPos0);
              const min12 = Math.abs(k - sg.selfPos1);
              const min21 = Math.abs(k + 1 - sg.selfPos0);
              const min22 = Math.abs(k + 1 - sg.selfPos1);
              let min = min11;
              if (min12 < min) min = min12;
              if (min21 < min) min = min21;
              if (min22 < min) min = min22;
              if (min < SELF_CUTOFF) continue;
            }
            const j1 = ch.idx[k], j2 = ch.idx[k + 1];
            const q1 = [x[3 * j1], x[3 * j1 + 1], x[3 * j1 + 2]];
            const q2 = [x[3 * j2], x[3 * j2 + 1], x[3 * j2 + 2]];
            if (visc) {
              // viscous: the friction weight of the chain segment, before the geometry (native weight2)
              wArr[k] = weight2(c, r1, r2, q1, q2);
              if (wArr[k] === 0.0) continue;
              sumwS += wArr[k];
              for (let d = 0; d < 3; d++) {
                wp1[d] += wArr[k] * velOf(j1, d);
                wp2[d] += wArr[k] * velOf(j2, d);
              }
            }
            geometry(c, r1, r2, q1, q2, q1, geo);
            if (param[0] > cutoff) continue;
            const calpha = libmCos(param[1]);
            const salpha = libmSin(param[1]);
            const hsq = param[0] * param[0];
            let ceta = calpha * param[4];
            let seta = salpha * param[4];
            let dsq1 = hsq + seta * seta;
            if (ceta < param[2]) {
              const dceta = ceta - param[2];
              dsq1 += dceta * dceta;
            } else if (ceta > param[3]) {
              const dceta = ceta - param[3];
              dsq1 += dceta * dceta;
            }
            ceta = calpha * param[5];
            seta = salpha * param[5];
            let dsq2 = hsq + seta * seta;
            if (ceta < param[2]) {
              const dceta = ceta - param[2];
              dsq2 += dceta * dceta;
            } else if (ceta > param[3]) {
              const dceta = ceta - param[3];
              dsq2 += dceta * dceta;
            }
            if (dsq1 > cutoffsq && dsq2 > cutoffsq) continue;
            let jj1: number, jj2: number;
            if (dsq1 < dsq2) {
              jj1 = j1;
              jj2 = j2;
            } else {
              if (param[1] > Math.PI) param[1] -= Math.PI;
              else param[1] += Math.PI;
              const tmpP = -param[5];
              param[5] = -param[4];
              param[4] = tmpP;
              param[6] = tmpP;
              geo.m[0] = -geo.m[0];
              geo.m[1] = -geo.m[1];
              geo.m[2] = -geo.m[2];
              jj1 = j2;
              jj2 = j1;
            }
            const m = geo.m, p = geo.p;
            // first force contribution
            let evdwl = fsemi(c, param, fl, this.fend);
            let fend = this.fend.fend;
            if (evdwl === 0.0) continue;
            matvecBasis(geo.basis, fl.subarray(0, 3), fglobal[0]);
            matvecBasis(geo.basis, fl.subarray(3, 6), fglobal[1]);
            for (let k = 0; k < 3; k++) ftotal[k] = fglobal[0][k] + fglobal[1][k] + fend * m[k];
            for (let k = 0; k < 3; k++) ftotal[k] *= -0.5;
            for (let k = 0; k < 3; k++) {
              delr1[k] = r1[k] - p[k];
              delr2[k] = r2[k] - p[k];
            }
            cross(delr1, fglobal[0], t1);
            cross(delr2, fglobal[1], t2);
            for (let k = 0; k < 3; k++) torque[k] = t1[k] + t2[k];
            cross(torque, m, ftorque);
            const lp = param[5] - param[4];
            for (let k = 0; k < 3; k++) ftorque[k] *= 1.0 / lp;
            for (let k = 0; k < 3; k++) {
              fglobal[2][k] = ftotal[k] + ftorque[k];
              fglobal[3][k] = ftotal[k] - ftorque[k];
            }
            // add forces to nodes
            addTo(i1, fglobal[0], 0.5);
            addTo(i2, fglobal[1], 0.5);
            addTo(jj1, fglobal[2], 0.5);
            addTo(jj2, fglobal[3], 0.5);
            addTo(jj1, m, 0.5 * fend);
            // energy
            evdwl = 0.5 * evdwl;
            const evdwlAtom = 0.25 * evdwl;
            acc.evdwl += evdwl;
            addEatom(i1, evdwlAtom);
            addEatom(i2, evdwlAtom);
            addEatom(jj1, evdwlAtom);
            addEatom(jj2, evdwlAtom);
            // second force contribution
            param[6] += lp;
            evdwl = fsemi(c, param, fl, this.fend);
            fend = this.fend.fend;
            if (evdwl === 0.0) continue;
            matvecBasis(geo.basis, fl.subarray(0, 3), fglobal[0]);
            matvecBasis(geo.basis, fl.subarray(3, 6), fglobal[1]);
            for (let k = 0; k < 3; k++) ftotal[k] = fglobal[0][k] + fglobal[1][k] + fend * m[k];
            for (let k = 0; k < 3; k++) ftotal[k] *= -0.5;
            for (let k = 0; k < 3; k++) p[k] = lp * m[k] + p[k];
            for (let k = 0; k < 3; k++) {
              delr1[k] = r1[k] - p[k];
              delr2[k] = r2[k] - p[k];
            }
            cross(delr1, fglobal[0], t1);
            cross(delr2, fglobal[1], t2);
            for (let k = 0; k < 3; k++) torque[k] = t1[k] + t2[k];
            cross(torque, m, ftorque);
            for (let k = 0; k < 3; k++) ftorque[k] *= 1.0 / lp;
            for (let k = 0; k < 3; k++) {
              fglobal[2][k] = ftotal[k] + ftorque[k];
              fglobal[3][k] = ftotal[k] - ftorque[k];
            }
            addTo(i1, fglobal[0], -0.5);
            addTo(i2, fglobal[1], -0.5);
            addTo(jj2, fglobal[2], -0.5);
            addTo(jj1, fglobal[3], 0.5);
            // f[jj2] -= fglobal[3]
            for (let k = 0; k < 3; k++) f[3 * jj2 + k] -= fglobal[3][k];
            addTo(jj2, m, -0.5 * fend);
            evdwl = -0.5 * evdwl;
            const evdwlAtom2 = 0.25 * evdwl;
            acc.evdwl += evdwl;
            addEatom(i1, evdwlAtom2);
            addEatom(i2, evdwlAtom2);
            addEatom(jj1, evdwlAtom2);
            addEatom(jj2, evdwlAtom2);
          }
          if (visc && sumwS !== 0.0) {
            const sumwInvS = 1.0 / sumwS;
            // mean chain velocity and relative velocity; friction along the segment normal (basis z)
            for (let d = 0; d < 3; d++) vmean[d] = (wp1[d] + wp2[d]) * (0.5 * sumwInvS);
            for (let d = 0; d < 3; d++) vrel[d] = vmean[d] - vrBond[d];
            const vtot = vrel[0] * geo.basis[6] + vrel[1] * geo.basis[7] + vrel[2] * geo.basis[8];
            const fv = frictionTotal(visc, vtot, sumwS, geo.param);
            for (let d = 0; d < 3; d++) fvisc[d] = fv * geo.basis[6 + d];
            for (let d = 0; d < 3; d++) {
              f[3 * i1 + d] += fvisc[d];
              f[3 * i2 + d] += fvisc[d];
            }
            for (let k = 0; k < clen - 1; k++) {
              if (wArr[k] === 0.0) continue;
              const j1 = ch.idx[k], j2 = ch.idx[k + 1];
              const scale = wArr[k] * sumwInvS;
              addTo(j1, fvisc, -scale);
              addTo(j2, fvisc, -scale);
            }
          }
          continue;
        }
        // segment-chain interaction: the chain is replaced by a substitute straight (semi-)infinite CNT
        let endindex = -1;
        let qe: number[] | null = null;
        if (endflag === 1) {
          endindex = ch.idx[0];
          qe = [x[3 * endindex], x[3 * endindex + 1], x[3 * endindex + 2]];
        } else if (endflag === 2) {
          endindex = ch.idx[clen - 1];
          qe = [x[3 * endindex], x[3 * endindex + 1], x[3 * endindex + 2]];
        }
        p1.fill(0);
        p2.fill(0);
        dr1Sumw.fill(0);
        dr2Sumw.fill(0);
        q1dr1.fill(0);
        q2dr1.fill(0);
        q1dr2.fill(0);
        q2dr2.fill(0);
        const w = wArr;
        const wnode = new Float64Array(clen);
        const dqw: V3[] = [];
        const q1dq: Float64Array[] = [];
        const q2dq: Float64Array[] = [];
        for (let k = 0; k < clen; k++) {
          dqw.push(v3());
          q1dq.push(new Float64Array(9));
          q2dq.push(new Float64Array(9));
        }
        let sumw = 0;
        const wo = this.wo;
        wp1.fill(0);
        wp2.fill(0);
        for (let k = 0; k < clen - 1; k++) {
          const j1 = ch.idx[k], j2 = ch.idx[k + 1];
          const q1 = [x[3 * j1], x[3 * j1 + 1], x[3 * j1 + 2]];
          const q2 = [x[3 * j2], x[3 * j2 + 1], x[3 * j2 + 2]];
          weight(c, r1, r2, q1, q2, wo);
          w[k] = wo.w;
          if (w[k] === 0.0) {
            if (endflag === 1 && k === 0) endflag = 0;
            else if (endflag === 2 && k === clen - 2) endflag = 0;
            continue;
          }
          sumw += w[k];
          wnode[k] += w[k];
          wnode[k + 1] += w[k];
          for (let d = 0; d < 3; d++) {
            p1[d] = w[k] * q1[d] + p1[d];
            p2[d] = w[k] * q2[d] + p2[d];
            dr1Sumw[d] += wo.dr1[d];
            dr2Sumw[d] += wo.dr2[d];
            if (visc) {
              wp1[d] += w[k] * velOf(j1, d);
              wp2[d] += w[k] * velOf(j2, d);
            }
          }
          outer(q1, wo.dr1, tmp);
          for (let q = 0; q < 9; q++) q1dr1[q] += tmp[q];
          outer(q2, wo.dr1, tmp);
          for (let q = 0; q < 9; q++) q2dr1[q] += tmp[q];
          outer(q1, wo.dr2, tmp);
          for (let q = 0; q < 9; q++) q1dr2[q] += tmp[q];
          outer(q2, wo.dr2, tmp);
          for (let q = 0; q < 9; q++) q2dr2[q] += tmp[q];
          for (let d = 0; d < 3; d++) {
            dqw[k][d] += wo.dp1[d];
            dqw[k + 1][d] += wo.dp2[d];
          }
          outer(q1, wo.dp1, tmp);
          for (let q = 0; q < 9; q++) q1dq[k][q] += tmp[q];
          outer(q1, wo.dp2, tmp);
          for (let q = 0; q < 9; q++) q1dq[k + 1][q] += tmp[q];
          outer(q2, wo.dp1, tmp);
          for (let q = 0; q < 9; q++) q2dq[k][q] += tmp[q];
          outer(q2, wo.dp2, tmp);
          for (let q = 0; q < 9; q++) q2dq[k + 1][q] += tmp[q];
        }
        if (sumw === 0.0) continue;
        const sumwInv = 1.0 / sumw;
        for (let d = 0; d < 3; d++) {
          p1[d] *= sumwInv;
          p2[d] *= sumwInv;
        }
        let evdwl: number;
        let fend = 0;
        const P1 = [p1[0], p1[1], p1[2]], P2 = [p2[0], p2[1], p2[2]];
        if (endflag === 0) {
          // infinite CNT case
          geometry(c, r1, r2, P1, P2, null, geo);
          if (param[0] > cutoff) continue;
          if (param[2] >= 0 || param[3] <= 0) {
            const salpha = libmSin(param[1]);
            const sxi1 = salpha * param[2];
            const sxi2 = salpha * param[3];
            const hsq = param[0] * param[0];
            if (sxi1 * sxi1 + hsq > cutoffsq && sxi2 * sxi2 + hsq > cutoffsq) continue;
          }
          evdwl = finf(c, param, fl);
        } else {
          // semi-infinite CNT case: endflag 1 = the chain end is its first atom, 2 = its last
          if (endflag === 1) geometry(c, r1, r2, P1, P2, qe, geo);
          else geometry(c, r1, r2, P2, P1, qe, geo);
          if (param[0] > cutoff) continue;
          if (param[2] >= 0 || param[3] <= 0) {
            const hsq = param[0] * param[0];
            const calpha = libmCos(param[1]);
            let etamin = calpha * param[2];
            let dsq1: number;
            if (etamin < param[6]) {
              dsq1 = hsq + Math.pow(libmSin(param[1]) * param[6], 2);
              dsq1 += Math.pow(param[2] - calpha * param[6], 2);
            } else dsq1 = hsq + Math.pow(libmSin(param[1]) * param[2], 2);
            etamin = calpha * param[3];
            let dsq2: number;
            if (etamin < param[6]) {
              dsq2 = hsq + Math.pow(libmSin(param[1]) * param[6], 2);
              dsq2 += Math.pow(param[3] - calpha * param[6], 2);
            } else dsq2 = hsq + Math.pow(libmSin(param[1]) * param[3], 2);
            if (dsq1 > cutoffsq && dsq2 > cutoffsq) continue;
          }
          evdwl = fsemi(c, param, fl, this.fend);
          fend = this.fend.fend;
        }
        if (evdwl === 0.0) continue;
        evdwl *= 0.5;
        const m = geo.m, p = geo.p;
        // transform to global coordinates; forces on the approximate chain
        matvecBasis(geo.basis, fl.subarray(0, 3), fglobal[0]);
        matvecBasis(geo.basis, fl.subarray(3, 6), fglobal[1]);
        if (visc) {
          // viscous friction on the segment and its chain (along the segment normal, basis z)
          for (let d = 0; d < 3; d++) vmean[d] = (wp1[d] + wp2[d]) * (0.5 * sumwInv);
          for (let d = 0; d < 3; d++) vrel[d] = vmean[d] - vrBond[d];
          const vtot = vrel[0] * geo.basis[6] + vrel[1] * geo.basis[7] + vrel[2] * geo.basis[8];
          const fv = frictionTotal(visc, vtot, sumw, geo.param);
          for (let d = 0; d < 3; d++) fvisc[d] = fv * geo.basis[6 + d];
          for (let d = 0; d < 3; d++) {
            f[3 * i1 + d] += fvisc[d];
            f[3 * i2 + d] += fvisc[d];
          }
        }
        for (let k = 0; k < 3; k++) ftotal[k] = fglobal[0][k] + fglobal[1][k];
        if (endflag) for (let k = 0; k < 3; k++) ftotal[k] += fend * m[k];
        for (let k = 0; k < 3; k++) ftotal[k] *= -0.5;
        for (let k = 0; k < 3; k++) {
          delr1[k] = r1[k] - p[k];
          delr2[k] = r2[k] - p[k];
        }
        cross(delr1, fglobal[0], t1);
        cross(delr2, fglobal[1], t2);
        for (let k = 0; k < 3; k++) torque[k] = t1[k] + t2[k];
        if (endflag && qe) {
          for (let k = 0; k < 3; k++) delqe[k] = qe[k] - p[k];
          cross(delqe, m, t3);
          for (let k = 0; k < 3; k++) t3[k] *= fend;
          for (let k = 0; k < 3; k++) torque[k] += t3[k];
        }
        cross(torque, m, ftorque);
        const lp = param[5] - param[4];
        for (let k = 0; k < 3; k++) ftorque[k] *= 1.0 / lp;
        if (endflag === 2) {
          for (let k = 0; k < 3; k++) {
            fglobal[3][k] = ftotal[k] + ftorque[k];
            fglobal[2][k] = ftotal[k] - ftorque[k];
          }
        } else {
          for (let k = 0; k < 3; k++) {
            fglobal[2][k] = ftotal[k] + ftorque[k];
            fglobal[3][k] = ftotal[k] - ftorque[k];
          }
        }
        for (let a = 0; a < 4; a++) for (let k = 0; k < 3; k++) fglobal[a][k] *= 0.5;
        // weight gradient terms acting on the current segment
        outer(P1, dr1Sumw, tmp);
        for (let q = 0; q < 9; q++) dr1P1[q] = q1dr1[q] - tmp[q];
        outer(P2, dr1Sumw, tmp);
        for (let q = 0; q < 9; q++) dr1P2[q] = q2dr1[q] - tmp[q];
        outer(P1, dr2Sumw, tmp);
        for (let q = 0; q < 9; q++) dr2P1[q] = q1dr2[q] - tmp[q];
        outer(P2, dr2Sumw, tmp);
        for (let q = 0; q < 9; q++) dr2P2[q] = q2dr2[q] - tmp[q];
        transposeMatvec(dr1P1, fglobal[2], fgradR1P1);
        transposeMatvec(dr1P2, fglobal[3], fgradR1P2);
        transposeMatvec(dr2P1, fglobal[2], fgradR2P1);
        transposeMatvec(dr2P2, fglobal[3], fgradR2P2);
        // add forces to the nodes of the current segment
        for (let k = 0; k < 3; k++) {
          f[3 * i1 + k] += fglobal[0][k];
          f[3 * i2 + k] += fglobal[1][k];
        }
        for (let k = 0; k < 3; k++) {
          f[3 * i1 + k] += sumwInv * fgradR1P1[k];
          f[3 * i1 + k] += sumwInv * fgradR1P2[k];
          f[3 * i2 + k] += sumwInv * fgradR2P1[k];
          f[3 * i2 + k] += sumwInv * fgradR2P2[k];
        }
        // forces on the approximate chain
        for (let k = 0; k < clen - 1; k++) {
          if (w[k] === 0.0) continue;
          const j1 = ch.idx[k], j2 = ch.idx[k + 1];
          const scale = w[k] * sumwInv;
          addTo(j1, fglobal[2], scale);
          addTo(j2, fglobal[3], scale);
          if (visc) {
            addTo(j1, fvisc, -scale);
            addTo(j2, fvisc, -scale);
          }
        }
        // weight gradient terms acting on the approximate chain (over the nodes)
        for (let k = 0; k < clen; k++) {
          if (wnode[k] === 0.0) continue;
          const j1 = ch.idx[k];
          outer(P1, dqw[k], tmp);
          for (let q = 0; q < 9; q++) dqP1[q] = q1dq[k][q] - tmp[q];
          outer(P2, dqw[k], tmp);
          for (let q = 0; q < 9; q++) dqP2[q] = q2dq[k][q] - tmp[q];
          transposeMatvec(dqP1, fglobal[2], fgradQP1);
          transposeMatvec(dqP2, fglobal[3], fgradQP2);
          for (let d = 0; d < 3; d++) {
            f[3 * j1 + d] += sumwInv * fgradQP1[d];
            f[3 * j1 + d] += sumwInv * fgradQP2[d];
          }
        }
        // force on the node at the chain end
        if (endflag && endindex >= 0) addTo(endindex, m, 0.5 * fend);
        // energy
        acc.evdwl += evdwl;
        addEatom(i1, 0.25 * evdwl);
        addEatom(i2, 0.25 * evdwl);
        for (let k = 0; k < clen - 1; k++) {
          if (w[k] === 0.0) continue;
          const j1 = ch.idx[k], j2 = ch.idx[k + 1];
          const evdwlChain = 0.5 * w[k] * sumwInv * evdwl;
          addEatom(j1, evdwlChain);
          addEatom(j2, evdwlChain);
        }
      }
    }
  }
}

/** set_union of two sorted-by-index lists (the unique union, sorted), as the C++ std::set_union over sorted inputs. */
const unionSorted = (a: number[], b: number[]): number[] => {
  const sa = [...a].sort((p, q) => p - q), sb = [...b].sort((p, q) => p - q);
  const out: number[] = [];
  let ia = 0, ib = 0;
  while (ia < sa.length && ib < sb.length) {
    if (sa[ia] < sb[ib]) out.push(sa[ia++]);
    else if (sb[ib] < sa[ia]) out.push(sb[ib++]);
    else {
      out.push(sa[ia]);
      ia++;
      ib++;
    }
  }
  while (ia < sa.length) out.push(sa[ia++]);
  while (ib < sb.length) out.push(sb[ib++]);
  return out;
};

/** count_chains / chain_split: the consecutive-ID runs of one molecule, with the end flag of each (native end[j]). */
const splitChains = (
  red: number[], tagOf: (k: number) => number, molOf: (k: number) => number, typeOf: (k: number) => number, endTypes: number[],
): Chain[] => {
  if (red.length === 0) return [];
  const runs: number[][] = [];
  let cur: number[] = [];
  for (let j = 0; j < red.length - 1; j++) {
    const j1 = red[j], j2 = red[j + 1];
    cur.push(j1);
    if (tagOf(j2) - tagOf(j1) !== 1 || molOf(j1) !== molOf(j2)) {
      runs.push(cur);
      cur = [];
    }
  }
  cur.push(red[red.length - 1]);
  runs.push(cur);
  const matches = (t: number): boolean => endTypes.includes(t);
  return runs.map((r) => {
    const estart = matches(typeOf(r[0])), eend = matches(typeOf(r[r.length - 1]));
    let end = 0;
    if (estart && eend) end = 3;
    else if (estart) end = 1;
    else if (eend) end = 2;
    return { idx: Int32Array.from(r), end };
  });
};

/**
 * pair_style mesocnt/viscous (docs.lammps.org/pair_mesocnt.html): the mesocnt interaction plus a friction between the
 * segments and their chains that depends on the relative velocity. pair_coeff * * FILE fvisc_max kvisc vvisc end-type...
 * The atoms need ghost velocities (comm_modify vel yes), as the C++ init_style requires.
 */
export class PairMesoCNTViscous extends PairMesoCNT {
  override readonly name: string = 'mesocnt/viscous';

  override initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    if (!ctx.ghostVelocity) throw new StyleError('Pair mesocnt/viscous requires ghost atoms store velocity (comm_modify vel yes)');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 6) throw new StyleError('Illegal pair_coeff command: pair_coeff * * FILE fvisc_max kvisc vvisc end-types');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('pair_coeff for pair style mesocnt/viscous needs * * (manybody)');
    const fviscMax = parseNum(args[3], 'fvisc_max');
    const kvisc = parseNum(args[4], 'kvisc');
    const vvisc = parseNum(args[5], 'vvisc');
    this.visc = { fviscMax, kvisc, vvisc, fviscShift: fviscMax / (1.0 + Math.exp(kvisc * vvisc)) };
    this.setPotential(args[2], args.slice(6), ctx);
  }
}
