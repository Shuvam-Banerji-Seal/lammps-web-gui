/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm.cpp and src/BPM/bond_bpm.h (the common base of the BPM bond styles: keyword
 * parsing, bond breaking, the per-bond history and the special-bond update), at commit b332f28.
 * Contributing author of the BPM package: Joel Clemmer (SNL); copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Browser-engine differences: the per-bond history (the reference state of LAMMPS' fix BOND_HISTORY) is a map
 * keyed by the two atom IDs; a broken bond gets bond type 0 and the force field rebuilds the 1-2 special list
 * (ForceField, takeTopologyChange). LAMMPS' fix update_special_bonds is that rebuild.
 */
import { SimpleBonded, atomIndex, delta } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type LocalStoreRequest } from '../../engine/force/types';
import { logical } from '../../engine/force/util';
import { FixStoreLocal } from '../tracker/fix_store_local';
import type { System } from '../../engine/system';
import { nativeOrder } from '../../engine/atoms';

/** Key of a bond by its two atom IDs (IDs stay below 2^26 in the browser engine). */
export const bondKey = (a: number, b: number): number => (a < b ? a * 0x4000000 + b : b * 0x4000000 + a);

/** A yes/no setting, read with utils::logical as bond_bpm*.cpp reads them (`cmd` and `key` document the call site). */
export const yesno = (w: string, _cmd: string, _key: string): boolean => logical(w);

/** The value keywords of store/local (bond_bpm.cpp settings); the x/ref, y/ref and z/ref keywords are refused by name. */
const LOCAL_VALUES = new Set(['id1', 'id2', 'time', 'x', 'y', 'z']);

/**
 * Base class of the BPM bond styles. Subclasses set their own coefficients; this class holds the flags of the
 * common keywords and the bond-breaking hook. The doc (bond_bpm_rotational.html and bond_bpm_spring.html) says:
 * overlay/pair and break take yes or no. The doc says: "If the store/local keyword is used, an internal fix will track
 * bonds that break during the simulation." The fix is fix store/local (src/fix_store_local.cpp), read by a dump local.
 */
export abstract class BondBPMBase extends SimpleBonded {
  overlay = false;
  breakOn = true;
  /** Set when a compute() call broke bonds; read and cleared by takeTopologyChange(). */
  protected changed = false;
  /** store/local request (fix ID, every N steps, value keywords), or null. */
  protected localSpec: { id: string; freq: number; names: string[] } | null = null;
  /** The fix store/local's addRow, set by bond_style (attachLocalSink). */
  protected localSink: ((row: number[]) => void) | null = null;
  /** True for the styles whose break path calls emitBreak (bond_bpm_*.cpp process_broken). */
  protected readonly localStoreOk: boolean = false;

  localStoreRequest(): LocalStoreRequest | null {
    const spec = this.localSpec;
    if (!spec) return null;
    return { ...spec, create: (sys) => new FixStoreLocal(sys as System, spec.id, spec.freq, spec.names.length) };
  }

  attachLocalSink(sink: (row: number[]) => void): void {
    this.localSink = sink;
  }

  /** Parses the common keywords (overlay/pair, break, store/local); returns the other key/value words for the subclass. */
  protected takeCommon(args: string[]): string[] {
    const rest: string[] = [];
    const seen = new Set<string>();
    let k = 0;
    while (k < args.length) {
      const key = args[k];
      if (key === 'store/local') {
        if (!this.localStoreOk) throw new StyleError(`bond_style ${this.name} keyword store/local is not supported by the browser engine (its breaks write no fix store/local)`);
        if (seen.has(key)) throw new StyleError(`bond_style ${this.name}: keyword 'store/local' is given twice`);
        seen.add(key);
        const id = args[k + 1], freqWord = args[k + 2];
        if (id === undefined || freqWord === undefined) throw new StyleError('Illegal bond bpm command, store/local needs a fix ID and N');
        const freq = Number(freqWord);
        if (!Number.isInteger(freq) || freq <= 0) throw new StyleError(`Illegal bond bpm command, store/local N must be a positive integer (got '${freqWord}')`);
        k += 3;
        const names: string[] = [];
        while (k < args.length && LOCAL_VALUES.has(args[k])) names.push(args[k++]);
        if (k < args.length && /^[xyz]\/ref$/.test(args[k])) {
          throw new StyleError(`bond_style ${this.name} store/local value ${args[k]} is not supported by the browser engine (no property/atom reference positions)`);
        }
        if (!names.length) throw new StyleError('Storing local data must include at least one value to output');
        this.localSpec = { id, freq, names };
        continue;
      }
      const val = args[k + 1];
      if (key === 'overlay/pair' || key === 'break') {
        if (val === undefined) throw new StyleError(`Illegal bond bpm command, missing option for ${key}`);
        if (seen.has(key)) throw new StyleError(`bond_style ${this.name}: keyword '${key}' is given twice`);
        seen.add(key);
        if (key === 'overlay/pair') this.overlay = yesno(val, `bond_style ${this.name}`, key);
        else this.breakOn = yesno(val, `bond_style ${this.name}`, key);
      } else {
        rest.push(key);
        if (val !== undefined) rest.push(val);
      }
      k += 2;
    }
    return rest;
  }

  /** True once after a compute() that broke bonds (the force field then rebuilds the special lists). */
  takeTopologyChange(): boolean {
    const c = this.changed;
    this.changed = false;
    // overlay/pair yes: the special lists are not updated when bonds break (bond_bpm.cpp init_style: no fix update_special_bonds)
    return c && !this.overlay;
  }

  /** Breaks bond k: its type is set to 0 (the doc: "setting the bond type to 0"). */
  protected breakBond(type: Int32Array, k: number): void {
    type[k] = 0;
    this.changed = true;
  }

  /**
   * Writes the broken bond between atoms with IDs ia and ib to the store/local fix (bond_bpm.cpp process_broken):
   * id1 is the atom with the lower ID, id2 the higher, time the current step, x/y/z the mid point of the bond.
   * The break path of a style calls it once per bond, after breakBond.
   */
  protected emitBreak(bc: BondedCompute, ia: number, ib: number): void {
    this.unlist(ia, ib);
    if (!this.localSink || !this.localSpec) return;
    const { i1, i2, a1, a2 } = this.pairOf(bc, ia, ib);
    const d = [0, 0, 0];
    delta(bc, i1, i2, d);
    const x = bc.s.x;
    const row = this.localSpec.names.map((w) => {
      switch (w) {
        case 'id1': return a1;
        case 'id2': return a2;
        case 'time': return bc.s.step;
        case 'x': return x[3 * i1] + 0.5 * d[0];
        case 'y': return x[3 * i1 + 1] + 0.5 * d[1];
        default: return x[3 * i1 + 2] + 0.5 * d[2];
      }
    });
    this.localSink(row);
  }

  /**
   * Native per-atom bond lists (bond_atom[i][m]), kept only while a store/local fix is attached. Entries are bond keys
   * (bondKey of the two IDs), not topology indices: the force field may hand a style a topology without the switched
   * off bonds, so indices change between calls. As native LAMMPS (src/BPM/bond_bpm.cpp process_broken): a broken bond
   * is removed from both atoms' lists by moving the last entry into its slot, so the lists change between neighbor
   * builds; the neighbor list (src/ntopo_bond_partial.cpp) is built from them at each build.
   */
  private lists: Map<number, number[]> | null = null;
  private listBuild = -2;
  private listOrder: Int32Array = new Int32Array(0);

  /**
   * The order in which a style visits its bonds when it writes store/local rows, or null (any order) when it writes
   * none. As native LAMMPS (src/ntopo_bond_partial.cpp at b332f28, newton bond off): the bond list has each bond once,
   * under its owner (the atom with the lower local index), the owners in local-index order, then each owner's list
   * in its current order. The local index is the native atom order (nativeOrder). The list is rebuilt at each
   * neighbor build (bc.nbuild), as native does.
   */
  protected visitOrder(bc: BondedCompute): Int32Array | null {
    if (!this.localSink || !this.localSpec) return null;
    const s = bc.s, b = s.topo.bonds;
    if (!this.lists) {
      this.lists = new Map();
      for (let i = 0; i < s.n; i++) this.lists.set(s.id[i], []);
      for (let k = 0; k < b.n; k++) {
        if (b.type[k] <= 0) continue;
        const key = bondKey(b.atoms[2 * k], b.atoms[2 * k + 1]);
        this.lists.get(b.atoms[2 * k])?.push(key);
        this.lists.get(b.atoms[2 * k + 1])?.push(key);
      }
    }
    if (this.listBuild === bc.nbuild) return this.listOrder;
    this.listBuild = bc.nbuild ?? -1;
    const at = new Map<number, number>();
    for (let k = 0; k < b.n; k++) if (b.type[k] > 0) at.set(bondKey(b.atoms[2 * k], b.atoms[2 * k + 1]), k);
    const ord = nativeOrder(s);
    const rank = new Int32Array(s.n);
    for (let p = 0; p < s.n; p++) rank[ord[p]] = p;
    const out: number[] = [];
    for (let p = 0; p < s.n; p++) {
      const i = ord[p], id = s.id[i];
      for (const key of this.lists.get(id) ?? []) {
        const k = at.get(key);
        if (k === undefined) continue;
        const other = b.atoms[2 * k] === id ? b.atoms[2 * k + 1] : b.atoms[2 * k];
        if (rank[i] < rank[atomIndex(bc, other, 'bond')]) out.push(k);
      }
    }
    this.listOrder = Int32Array.from(out);
    return this.listOrder;
  }

  /** Removes the bond between IDs ia and ib from both atoms' native lists (the last entry moves into its slot). */
  private unlist(ia: number, ib: number): void {
    const key = bondKey(ia, ib);
    for (const id of [ia, ib]) {
      const list = this.lists?.get(id);
      if (!list) continue;
      const m = list.indexOf(key);
      if (m < 0) continue;
      list[m] = list[list.length - 1];
      list.pop();
    }
  }

  /** Index pair of a bond (atoms i1 = smaller ID, i2 = larger ID, as LAMMPS' bondlist ordering). */
  protected pairOf(bc: BondedCompute, ia: number, ib: number): { i1: number; i2: number; a1: number; a2: number } {
    const lo = Math.min(ia, ib), hi = Math.max(ia, ib);
    return { i1: atomIndex(bc, lo, 'bond'), i2: atomIndex(bc, hi, 'bond'), a1: lo, a2: hi };
  }
}
