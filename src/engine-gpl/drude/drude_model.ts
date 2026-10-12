/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/fix_drude.cpp and
 * src/DRUDE/fix_drude.h (class FixDrude: the Drude atom types, the core-Drude partner lists and the special
 * neighbor lists rebuilt so that a Drude particle is equivalent to its core atom). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is
 * GPL-2.0 too (src/engine-gpl/LICENSE). The DRUDE package was written by Alain Dequidt (Clermont University, CNRS)
 * with Julien Devemy and Agilio Padua (ENS de Lyon); contributors Kateryna Goloviznina and Zheng Gong.
 *
 * The Drude model shared by the DRUDE styles of this directory: the type kinds, the partner of every core and
 * Drude atom (by atom ID, fixed when fix drude is defined, as LAMMPS builds drudeid once), and the special list
 * transform. docs.lammps.org/fix_drude.html: "The fix drude command is required ... to define the Drude particles"
 * and the special neighbors are built so each Drude particle carries the special list of its core.
 */
import type { SimState } from '../../engine/types';
import type { SpecialList } from '../../engine/atoms';
import { StyleError } from '../../engine/force/types';

export const NOPOL_TYPE = 0;
export const CORE_TYPE = 1;
export const DRUDE_TYPE = 2;

/** Parses the per-type words of fix drude: n/0 non-polarizable, c/1 core, d/2 Drude (fix drude.html). */
export const parseDrudeTypes = (words: string[], ntypes: number): Uint8Array => {
  if (words.length !== ntypes) {
    throw new StyleError(`Incorrect number of arguments (${words.length} instead of ${ntypes}) for fix drude command`);
  }
  const kind = new Uint8Array(ntypes + 1);
  words.forEach((w, k) => {
    const c = w[0];
    if (c === 'n' || c === 'N' || c === '0') kind[k + 1] = NOPOL_TYPE;
    else if (c === 'c' || c === 'C' || c === '1') kind[k + 1] = CORE_TYPE;
    else if (c === 'd' || c === 'D' || c === '2') kind[k + 1] = DRUDE_TYPE;
    else throw new StyleError(`Unknown drude type ${w} for atom type ${k + 1}`);
  });
  return kind;
};

/** A Drude model: the kind of each type and the partner IDs of the cores and Drude particles. */
export class DrudeModel {
  /** Partner ID of each core atom (its Drude particle) and of each Drude particle (its core). */
  readonly drudeOf = new Map<number, number>();
  readonly coreOf = new Map<number, number>();

  constructor(readonly s: SimState, readonly kind: Uint8Array) {
    const idIndex = new Map<number, number>();
    for (let i = 0; i < s.n; i++) idIndex.set(s.id[i], i);
    const partners = new Map<number, number[]>();
    const b = s.topo.bonds;
    for (let e = 0; e < b.n; e++) {
      const a = b.atoms[2 * e], c = b.atoms[2 * e + 1];
      if (!partners.has(a)) partners.set(a, []);
      if (!partners.has(c)) partners.set(c, []);
      partners.get(a)!.push(c);
      partners.get(c)!.push(a);
    }
    for (let i = 0; i < s.n; i++) {
      const k = kind[s.type[i]];
      if (k === NOPOL_TYPE) continue;
      const id = s.id[i];
      const ps = (partners.get(id) ?? []).slice().sort((p, q) => p - q);
      if (k === CORE_TYPE) {
        // fix_drude.cpp build_drudeid: the core takes the smallest bonded partner that is a Drude type
        const d = ps.find((p) => { const j = idIndex.get(p); return j !== undefined && kind[s.type[j]] === DRUDE_TYPE; });
        if (d === undefined) throw new StyleError(`Core atom ID ${id} has no drude atom`);
        this.drudeOf.set(id, d);
      } else {
        // a Drude particle takes its (only) bonded partner, the core
        if (ps.length === 0) throw new StyleError(`Drude atom ID ${id} has no core atom`);
        this.coreOf.set(id, ps[0]);
      }
    }
  }

  /** Owned index of the Drude partner of each owned core, the core of each Drude; -1 otherwise. */
  partnerIndex(s: SimState): Int32Array {
    let maxId = 0;
    for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
    const pos = new Int32Array(maxId + 1).fill(-1);
    for (let i = 0; i < s.n; i++) pos[s.id[i]] = i;
    const out = new Int32Array(s.n).fill(-1);
    for (let i = 0; i < s.n; i++) {
      const id = s.id[i];
      const p = this.kind[s.type[i]] === CORE_TYPE ? this.drudeOf.get(id) : this.kind[s.type[i]] === DRUDE_TYPE ? this.coreOf.get(id) : undefined;
      if (p !== undefined && p <= maxId) out[i] = pos[p];
    }
    return out;
  }

  /**
   * LAMMPS rebuild_special (fix_drude.cpp): the special list of every non-Drude atom gains the Drude particle of
   * each core in its list (at the same 1-2, 1-3 or 1-4 level as the core, the core's own Drude first at 1-2), and a
   * Drude particle takes its core as first 1-2 neighbor and the rest of its core's special list.
   * The input list holds the bond-topology special neighbors (partners by ID with order 1, 2 or 3); the Drude
   * particles of the input are dropped first, as LAMMPS removes them with ring_remove_drude.
   */
  transformSpecial(list: SpecialList, s: SimState): SpecialList {
    const n = s.n;
    const kindOfId = new Map<number, number>();
    for (let i = 0; i < n; i++) kindOfId.set(s.id[i], this.kind[s.type[i]]);
    const entries: Array<Array<[number, number]>> = new Array(n);
    for (let i = 0; i < n; i++) {
      const out: Array<[number, number]> = [];
      if (this.kind[s.type[i]] !== DRUDE_TYPE) {
        for (let k = list.offset[i]; k < list.offset[i + 1]; k++) {
          const p = list.partner[k];
          if (kindOfId.get(p) === DRUDE_TYPE) continue;
          out.push([p, list.order[k]]);
        }
      }
      entries[i] = out;
    }
    const idIndex = new Map<number, number>();
    for (let i = 0; i < n; i++) idIndex.set(s.id[i], i);
    const final: Array<Array<[number, number]>> = new Array(n);
    for (let i = 0; i < n; i++) {
      const k = this.kind[s.type[i]];
      if (k === DRUDE_TYPE) continue;
      const base = entries[i];
      const out: Array<[number, number]> = [];
      if (k === CORE_TYPE) out.push([this.drudeOf.get(s.id[i])!, 1]);
      for (const e of base) out.push(e);
      for (const [j, o] of base) {
        if (kindOfId.get(j) === CORE_TYPE) out.push([this.drudeOf.get(j)!, o]);
      }
      final[i] = out;
    }
    for (let i = 0; i < n; i++) {
      if (this.kind[s.type[i]] !== DRUDE_TYPE) continue;
      const core = this.coreOf.get(s.id[i]);
      const c = core === undefined ? undefined : idIndex.get(core);
      if (c === undefined || final[c] === undefined) {
        final[i] = entries[i];
        continue;
      }
      const own = this.drudeOf.get(core!);
      const out: Array<[number, number]> = [[core!, 1]];
      for (const e of final[c]) if (e[0] !== own) out.push(e);
      final[i] = out;
    }
    let total = 0;
    const offset = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) {
      offset[i] = total;
      total += final[i].length;
    }
    offset[n] = total;
    const partner = new Int32Array(total);
    const order = new Int8Array(total);
    for (let i = 0; i < n; i++) {
      let k = offset[i];
      for (const [p, o] of final[i]) {
        partner[k] = p;
        order[k] = o;
        k++;
      }
    }
    return { offset, partner, order };
  }
}

/** The Drude model of each state, so the DRUDE styles find the fix drude that defined it. */
const MODELS = new WeakMap<SimState, DrudeModel>();

export const setDrudeModel = (s: SimState, m: DrudeModel | null): void => {
  if (m) MODELS.set(s, m);
  else MODELS.delete(s);
};

/** The Drude model of a state, or a StyleError naming the style that needs fix drude. */
export const drudeModelOf = (s: SimState | null, who: string): DrudeModel => {
  const m = s ? MODELS.get(s) : undefined;
  if (!m) throw new StyleError(`${who} requires fix drude`);
  return m;
};
