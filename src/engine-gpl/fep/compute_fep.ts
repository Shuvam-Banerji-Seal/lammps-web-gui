/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/compute_fep.cpp and src/FEP/compute_fep.h (class ComputeFEP, compute fep) at commit b332f28, by
 * Agilio Padua (ENS de Lyon & CNRS) and the LAMMPS developers. Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * compute ID group-ID fep T keyword args ... (docs.lammps.org/compute_fep.html). Keywords: pair pstyle pparam I J
 * v_name (type-pair parameter perturbed by delta = v_name, added to the original value), atom charge type v_name
 * (charges of the group's atoms of the type range perturbed by delta), tail yes|no, volume yes|no.
 *
 * Vector (3): [0] = U(perturbed) - U(current) in energy units, [1] = exp(-[0] / (k_B T)) (times the box volume
 * with volume yes), [2] = box volume. U is the pair energy (evdwl + ecoul, the tail term only with tail yes),
 * plus the kspace energy when charges are perturbed. The perturbed energy is evaluated on the current positions
 * and the forces, charges and pair parameters are restored afterwards, as the native compute restores its
 * backups (backup_qfev / restore_qfev).
 *
 * In this engine the pair tail correction is part of evdwl, so tail no removes it (etail / V) from both
 * energies; tail yes requires pair_modify tail yes, as in the native init().
 */
import { Compute } from '../../engine/compute/compute';
import { Accum, Pair, StyleError, typeBounds } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { hasChargeStyle } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { hasTail, refreshChargeSums, reinitPair, resolvePairTable, typePairs, type PairTable } from './pair_table';
import { exp as libmExp } from '../libm/exp';

const USAGE = 'usage: compute ID group-ID fep T keyword args ... (pair pstyle pparam I J v_name | atom charge I v_name | tail yes/no | volume yes/no)';

type Pert =
  | { kind: 'pair'; pstyle: string; pparam: string; iw: string; jw: string; vname: string; table: PairTable | null; pairs: Array<[number, number]>; orig: Float64Array }
  | { kind: 'charge'; iw: string; vname: string };

const logical = (key: string, w: string | undefined): boolean => {
  if (w !== 'yes' && w !== 'no') throw new StyleError(`compute fep keyword ${key} must be yes or no, got '${w ?? ''}'`);
  return w === 'yes';
};

export class ComputeFEP extends Compute {
  readonly style = 'fep';
  private readonly temp: number;
  private readonly perts: Pert[] = [];
  private tailFlag = false;
  private volumeFlag = false;
  private chgFlag = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError(`Illegal number of arguments in compute fep (${USAGE})`);
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extvector = 0;
    this.vector = new Float64Array(3);
    this.temp = parseNum(args[0], 'temperature');
    let k = 1;
    while (k < args.length) {
      const w = args[k];
      if (w === 'pair') {
        const vw = args[k + 5];
        if (vw === undefined || !vw.startsWith('v_')) throw new StyleError(`Illegal variable in compute fep (${USAGE})`);
        this.perts.push({ kind: 'pair', pstyle: args[k + 1], pparam: args[k + 2], iw: args[k + 3], jw: args[k + 4], vname: vw.slice(2), table: null, pairs: [], orig: new Float64Array(0) });
        k += 6;
      } else if (w === 'atom') {
        if (args[k + 1] !== 'charge') throw new StyleError(`Illegal atom argument in compute fep (${USAGE})`);
        const vw = args[k + 3];
        if (vw === undefined || !vw.startsWith('v_')) throw new StyleError(`Illegal variable in compute fep (${USAGE})`);
        this.perts.push({ kind: 'charge', iw: args[k + 2], vname: vw.slice(2) });
        this.chgFlag = true;
        k += 4;
      } else break;
    }
    if (this.perts.length === 0) throw new StyleError(`Illegal syntax in compute fep (${USAGE})`);
    while (k < args.length) {
      const w = args[k];
      if (w === 'tail') this.tailFlag = logical(w, args[k + 1]);
      else if (w === 'volume') this.volumeFlag = logical(w, args[k + 1]);
      else throw new StyleError(`Illegal optional keyword in compute fep: '${w}' (${USAGE})`);
      k += 2;
    }
  }

  init(): void {
    const s = this.sys.state;
    for (const p of this.perts) {
      if (p.kind === 'pair') {
        p.table = resolvePairTable(this.sys, `compute ${this.id} (fep)`, p.pstyle, p.pparam);
        p.pairs = typePairs(p.iw, p.jw, s.ntypes, false);
        p.orig = new Float64Array(p.pairs.length);
      } else {
        if (!hasChargeStyle(s.atomStyle)) throw new StyleError(`compute fep requires atom attribute charge`);
        typeBounds(p.iw, s.ntypes);
      }
    }
    if (this.tailFlag && !this.pairTail()) throw new StyleError(`Compute fep tail when pair style does not compute tail corrections`);
  }

  private pairTail(): boolean {
    const owner = this.sys.ff.pair;
    return owner ? hasTail(owner) : false;
  }

  /**
   * Pair energy, with the tail term only when tail yes, plus the kspace energy when charges are perturbed.
   * acc is the force field accumulator of a state whose neighbor lists are current.
   */
  private energyOf(acc: Accum): number {
    let e = acc.evdwl + acc.ecoul;
    if (!this.tailFlag) e -= this.sys.ff.etailV / this.sys.geom.volume(this.sys.state.dimension);
    if (this.chgFlag) e += acc.elong;
    return e;
  }

  /**
   * Energy of the current state without the neighbor rebuild and force-field init of sys.forces() (that init
   * would re-choose the kspace g-ewald from the new charges, which native compute fep does not do). The
   * forces and accumulator are overwritten, so computeVector restores them.
   */
  private perturbedEnergy(): number {
    return this.energyOf(this.sys.computeForcesInRun(this.sys.computeFlags()));
  }

  protected computeVector(): void {
    const sys = this.sys, s = sys.state;
    const kT = s.units.boltz * this.temp;
    const pe0 = this.energyOf(sys.forces());
    // backups: the forces, the force-field accumulator, the charges and the pair tables the perturbation touches
    const fSave = s.f.slice();
    const qSave = s.q.slice();
    const accSave = snapshotAccum(sys.ff.acc);
    const owners = new Set<Pair>();
    for (const p of this.perts) {
      if (p.kind !== 'pair' || !p.table) continue;
      p.pairs.forEach(([i, j], k) => { p.orig[k] = p.table!.table.get(p.table!.name, i, j); });
      owners.add(p.table.owner);
    }
    try {
      for (const p of this.perts) {
        const delta = sys.equalVariable(p.vname);
        if (p.kind === 'pair' && p.table) {
          const arr = p.table.table.p(p.table.name);
          p.pairs.forEach(([i, j], k) => {
            arr[p.table!.table.idx(i, j)] = p.orig[k] + delta;
            arr[p.table!.table.idx(j, i)] = p.orig[k] + delta;
          });
        } else if (p.kind === 'charge') {
          const [ilo, ihi] = typeBounds(p.iw, s.ntypes);
          for (let i = 0; i < s.n; i++) {
            if (s.type[i] >= ilo && s.type[i] <= ihi && (s.mask[i] & this.groupBit)) s.q[i] += delta;
          }
        }
      }
      for (const owner of owners) reinitPair(sys, owner);
      if (this.chgFlag) refreshChargeSums(sys);
      const pe1 = this.perturbedEnergy();
      const dU = pe1 - pe0;
      this.vector[0] = dU;
      this.vector[1] = libmExp(-dU / kT);
      this.vector[2] = sys.geom.volume(s.dimension);
      if (this.volumeFlag) this.vector[1] *= this.vector[2];
    } finally {
      // restore the charges, the pair tables (their original values), the forces and the accumulator
      s.q.set(qSave);
      for (const p of this.perts) {
        if (p.kind !== 'pair' || !p.table) continue;
        const arr = p.table.table.p(p.table.name);
        p.pairs.forEach(([i, j], k) => {
          arr[p.table!.table.idx(i, j)] = p.orig[k];
          arr[p.table!.table.idx(j, i)] = p.orig[k];
        });
      }
      for (const owner of owners) reinitPair(sys, owner);
      if (this.chgFlag) refreshChargeSums(sys);
      s.f.set(fSave);
      restoreAccum(sys.ff.acc, accSave);
    }
  }
}

/** Copies of an accumulator's values (scalars and the virial arrays). */
export const snapshotAccum = (a: Accum): Record<string, number | Float64Array> => {
  const out: Record<string, number | Float64Array> = {};
  for (const [k, v] of Object.entries(a)) out[k] = typeof v === 'number' ? v : (v as Float64Array).slice();
  return out;
};

export const restoreAccum = (a: Accum, saved: Record<string, number | Float64Array>): void => {
  const target = a as unknown as Record<string, number | Float64Array>;
  for (const [k, v] of Object.entries(saved)) {
    if (typeof v === 'number') target[k] = v;
    else (target[k] as Float64Array).set(v);
  }
};
