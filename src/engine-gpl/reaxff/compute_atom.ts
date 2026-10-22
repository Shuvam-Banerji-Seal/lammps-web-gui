/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/compute_reaxff_atom.cpp (ComputeReaxFFAtom: constructor keywords pair and bonds, init, FindBond,
 * compute_bonds, compute_local, compute_peratom) at commit b332f28. Authors credited in the C++ file header: Hasan Metin
 * Aktulga (Michigan State University / Purdue University) and the LAMMPS ReaxFF authors (Sandia). Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * compute reaxff/atom (docs.lammps.org/compute_reaxff_atom.html): per atom, the total bond order, the number of lone
 * pairs and the number of bonds (bond order above the bond graph cutoff, both atoms in the group); with the bonds
 * keyword, a local array of (atom ID, bonded atom ID, bond order) rows for those bonds. The bond list, the native
 * order of the atoms and the ReaxFF workspace come from the pair style of the last force evaluation (the same hooks
 * as fix reaxff/bonds, reports.ts).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { inumeric, logical, missingArgs } from '../../engine/force/util';
import { nativeOrder } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { PairReaxFF } from './pair';
import type { RxSystem } from './core';
import { nativeRegion, tagOf } from './reports';

/** One bond of the bond list of atom i that compute reaxff/atom counts: partner index and bond order. */
interface Bond { pj: number; partner: number; bo: number }

export class ComputeReaxFFAtom extends Compute {
  readonly style = 'reaxff/atom';
  override peratomFlag = true;
  override sizePeratomCols = 3;
  override localFlag = false;
  override sizeLocalCols = 3;
  private readonly storeBonds: boolean;
  private readonly pairIndex: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    let storeBonds = false;
    let pairIndex = 0;
    let iarg = 0;
    while (iarg < args.length) {
      const w = args[iarg];
      // compute_reaxff_atom.cpp: pair N (N a word starting with a digit, > 0) and bonds yes|no
      if (w === 'pair') {
        if (iarg + 2 > args.length) throw missingArgs('compute reaxff/atom pair');
        const v = args[iarg + 1];
        const nsub = /^\d/.test(v) ? inumeric(v) : 0;
        if (nsub <= 0) throw new StyleError('Illegal compute reaxff/atom command');
        pairIndex = nsub;
        iarg += 2;
      } else if (w === 'bonds') {
        if (iarg + 2 > args.length) throw missingArgs('compute reaxff/atom bonds');
        storeBonds = logical(args[iarg + 1]);
        iarg += 2;
      } else {
        throw new StyleError('Illegal compute reaxff/atom command');
      }
    }
    this.storeBonds = storeBonds;
    this.pairIndex = pairIndex;
    this.localFlag = storeBonds;
  }

  /** init(): needs pair_style reaxff; the native far-list order is kept for the bond order of the rows. */
  init(): void {
    const p = this.pairOf();
    if (this.pairIndex > 1) throw new StyleError(`compute ${this.id} reaxff/atom: pair ${this.pairIndex} is not supported (the engine has one reaxff pair style)`);
    p.wantNative();
    const s = this.sys.state;
    // native: "Atom IDs must be consecutive for compute reaxff/atom" (tag_consecutive)
    const seen = new Uint8Array(s.n + 1);
    for (let i = 0; i < s.n; i++) {
      const id = s.id[i];
      if (!Number.isInteger(id) || id < 1 || id > s.n || seen[id]) throw new StyleError(`Atom IDs must be consecutive for compute ${this.id} reaxff/atom`);
      seen[id] = 1;
    }
  }

  private pairOf(): PairReaxFF {
    const p = this.sys.ff.pair;
    if (!(p instanceof PairReaxFF)) throw new StyleError(`Cannot use compute ${this.id} reaxff/atom without pair_style reaxff`);
    return p;
  }

  /** The ReaxFF state of the current step (the last force evaluation), with the native far order. */
  private current(): { st: RxSystem; cut: number; nat: NonNullable<PairReaxFF['nativeInfo']> } {
    const p = this.pairOf();
    this.sys.forces();
    if (!p.lastBonds || !p.nativeInfo) throw new StyleError(`compute ${this.id} reaxff/atom: no reaxff evaluation to read`);
    return { st: p.lastBonds, cut: p.bondGraphCutoff(), nat: p.nativeInfo };
  }

  /** The bonds of owned atom i counted by native FindBond (BO above the cutoff; i and the partner in the group). */
  private bondsOf(st: RxSystem, nat: NonNullable<PairReaxFF['nativeInfo']>, cut: number, i: number): Bond[] {
    const s = this.sys.state;
    const n = st.n;
    const out: Bond[] = [];
    if (!(s.mask[i] & this.groupBit)) return out;
    for (const pj of nativeRegion(st.bonds, nat, i, n)) {
      const partner = st.bonds.nbr[pj];
      const mp = s.mask[partner < n ? partner : this.sys.nb.owner[partner]];
      if (!(mp & this.groupBit)) continue;
      const bo = st.bonds.BO[pj];
      if (bo > cut) out.push({ pj, partner, bo });
    }
    return out;
  }

  protected computePeratom(): void {
    const { st, cut, nat } = this.current();
    const s = this.sys.state;
    const out = new Float64Array(s.n * 3);
    for (let i = 0; i < s.n; i++) {
      // native: the total bond order and nlp of every atom; bondcount is zero outside the group
      out[3 * i] = st.ws.total_bond_order[i];
      out[3 * i + 1] = st.ws.nlp[i];
      out[3 * i + 2] = this.bondsOf(st, nat, cut, i).length;
    }
    this.arrayAtom = out;
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    if (!this.storeBonds) throw new StyleError(`compute ${this.id} reaxff/atom: local values need the bonds yes keyword`);
    const { st, cut, nat } = this.current();
    const s = this.sys.state;
    const order = nativeOrder(s);
    const rows: number[] = [];
    // native compute_local: native local order, the bonds of each atom in the order of its bond list
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      for (const b of this.bondsOf(st, nat, cut, i)) rows.push(tagOf(this.sys, i, st.n), tagOf(this.sys, b.partner, st.n), b.bo);
    }
    this.localRows = rows.length / 3;
    return Float64Array.from(rows);
  }
}
