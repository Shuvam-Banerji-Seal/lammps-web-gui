/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MISC/fix_srp.cpp and src/MISC/fix_srp.h at commit b332f28 (fix SRP: the bond particles of pair style
 * srp; Timothy Sirk, ARL, and Pieter in 't Veld, BASF). LAMMPS is Copyright (2003) Sandia Corporation and distributed
 * under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The bond particles are real atoms, as in native: setup_pre_force deletes the old ones (native's delete loop), makes
 * one per bond of the neighbor bond list (native NTopoBondAll order: the owned atoms in native order, each atom's
 * bonds in storage order, the partner as its closest image) at the bond midpoint with the type bptype, appends them
 * (new tags in that order, as tag_extend gives them), wraps them into the box, sorts the atoms and rebuilds the
 * neighbor lists; then the bond particles leave every group (mask 0) and stand still. Before each neighbor build
 * (pre_exchange) they move to the midpoint of the closest images of their two atoms; after the run (post_run) they are
 * deleted again (native's delete loop), the atoms wrapped and sorted. Each bond particle keeps the tags of its two
 * atoms in the per-atom custom array '#srp' (native: the fix's per-atom array).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { appendAtoms, buildAtomMap, deleteAtoms, nativeOrder } from '../../engine/atoms';

/** Per-atom custom array of the bond particles: the tags of the two atoms of the bond (2 columns; internal, '#'). */
export const SRP_TAGS = '#srp';

/** Engine index of the copy of an owned atom (the atom or one of its ghosts) closest to position p (Domain::closest_image). */
export const closestImage = (
  xall: Float64Array, head: Int32Array, next: Int32Array, owner: number, px: number, py: number, pz: number,
): number => {
  if (owner < 0) return owner;
  let j = head[owner];
  let closest = j;
  let dx = px - xall[3 * j], dy = py - xall[3 * j + 1], dz = pz - xall[3 * j + 2];
  let rsqmin = dx * dx + dy * dy + dz * dz;
  while (next[j] >= 0) {
    j = next[j];
    dx = px - xall[3 * j]; dy = py - xall[3 * j + 1]; dz = pz - xall[3 * j + 2];
    const rsq = dx * dx + dy * dy + dz * dz;
    if (rsq < rsqmin) { rsqmin = rsq; closest = j; }
  }
  return closest;
};

/** The copies of each owned atom as a chain from its owned index up (native map_set: map = smallest index, sametag). */
export const imageChains = (owner: Int32Array, nall: number, nlocal: number): { head: Int32Array; next: Int32Array } => {
  const head = new Int32Array(nlocal).fill(-1);
  const next = new Int32Array(nall).fill(-1);
  for (let k = nall - 1; k >= 0; k--) {
    const o = owner[k];
    next[k] = head[o];
    head[o] = k;
  }
  return { head, next };
};

export class FixSRP extends Fix {
  readonly style: string = 'SRP';
  /** the pair style named in the messages (native pair_name) */
  protected pairName = 'srp';
  /** the bond type the bond particles represent (0: every bond) and the bond particle type, set by pair srp */
  btype = -1;
  bptype = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const s = sys.state;
    if (!s.custom.has(SRP_TAGS)) s.custom.set(SRP_TAGS, { int: false, cols: 2, data: new Float64Array(2 * s.n) });
  }

  /** modify_param: btype, bptype (from pair srp init). */
  setTypes(btype: number, bptype: number): void {
    this.btype = btype;
    this.bptype = bptype;
  }

  init(): void {
    const sys = this.sys;
    const pair = sys.ff.pair;
    if (!pair || !pair.name.startsWith('hybrid')) throw new StyleError(`Cannot use pair ${this.pairName} without pair_style hybrid`);
    if (sys.fixes.some((f) => f.style.startsWith('rigid'))) throw new StyleError(`Pair ${this.pairName} is not compatible with rigid fixes.`);
    const ntypes = sys.state.ntypes;
    if (this.bptype < 1 || this.bptype > ntypes) throw new StyleError('Illegal bond particle type');
    // bond particles only interact with each other: exclude every other type (native: types 1 .. ntypes-1)
    const nb = sys.nb;
    for (let z = 1; z < ntypes; z++) {
      if (z === this.bptype) continue;
      if (!nb.excludes.some((e) => e.kind === 'type' && e.a === z && e.b === this.bptype)) nb.excludes.push({ kind: 'type', a: z, b: this.bptype });
    }
    // pair srp init_style: "bond particles do not belong to group all but thermo normalization is by nall"
    sys.thermo.normUser = false;
    sys.log(`Thermo normalization turned off by pair ${this.pairName}`);
  }

  /** Native has no setup(): nothing after the setup forces. */
  override setup(): void {}

  /** setup_pre_force: insert the bond particles. */
  setupPreForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const bptype = this.bptype;
    const nlocalOld = s.n;
    const nall = nb.nall;
    const owner = nb.owner;
    const xall = nb.xall;
    // the neighbor bond list (NTopoBondAll::build): owned atoms in native order, their bonds in storage order
    const map = buildAtomMap(s);
    const bonds = s.topo.bonds;
    const perAtom: number[][] = Array.from({ length: s.n }, () => []);
    for (let k = 0; k < bonds.n; k++) {
      const a = map[bonds.atoms[2 * k]], b = map[bonds.atoms[2 * k + 1]];
      if (a >= 0) perAtom[a].push(2 * k);
      if (!sys.newtonBond && b >= 0) perAtom[b].push(2 * k + 1);
    }
    const { head, next } = imageChains(owner, nall, nb.nlocal);
    const ord = nativeOrder(s);
    // native local index of each owned atom (bondlist's "i < atom1" compares native indices; ghosts come after)
    const nativeIndex = new Int32Array(s.n);
    for (let p = 0; p < ord.length; p++) nativeIndex[ord[p]] = p;
    const xs: number[] = [];
    const tags: number[] = [];
    let rsqold = 0.0;
    for (let p = 0; p < ord.length; p++) {
      const i = ord[p];
      for (const code of perAtom[i]) {
        const k = code >> 1;
        const btypeK = bonds.type[k];
        // the partner: the other atom of the bond, as its closest image to atom i
        const other = bonds.atoms[2 * k + 1 - (code & 1)];
        const om = map[other];
        if (om < 0) throw new StyleError(`Bond atoms ${s.id[i]} ${other} missing`);
        const j = closestImage(xall, head, next, om, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
        if (!sys.newtonBond && !(p < (j < nlocalOld ? nativeIndex[j] : nlocalOld + j))) continue;
        if (this.btype > 0 && btypeK !== this.btype) continue;
        const xi = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
        const xj = [xall[3 * j], xall[3 * j + 1], xall[3 * j + 2]];
        const dx = xj[0] - xi[0], dy = xj[1] - xi[1], dz = xj[2] - xi[2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq > rsqold) rsqold = rsq;
        // one particle per bond: always with newton bond; otherwise when j is owned or from the tag order
        if (sys.newtonBond || j < nlocalOld || s.id[i] > s.id[owner[j]]) {
          xs.push((xi[0] + xj[0]) * 0.5, (xi[1] + xj[1]) * 0.5, (xi[2] + xj[2]) * 0.5);
          tags.push(s.id[i], s.id[owner[j]]);
        }
      }
    }
    // delete the old bond particles (native's loop over the owned atoms)
    const del = new Uint8Array(s.n);
    let ndel = 0;
    for (let i = 0; i < s.n; i++) if (s.type[i] === bptype) { del[i] = 1; ndel++; }
    if (ndel) deleteAtoms(s, del);
    const nadd = tags.length / 2;
    if (nadd) {
      appendAtoms(s, {
        x: Float64Array.from(xs),
        type: bptype,
        custom: new Map([[SRP_TAGS, Float64Array.from(tags)]]),
      });
    }
    sys.log(`Removed/inserted ${ndel}/${nadd} bond particles.`);
    // ghost atoms must be present for bonds on the edge of the neighbor cutoff: cutghost >= cutneighmax + 0.51 rmax
    const rmax = Math.sqrt(rsqold);
    const cutneighmaxSrp = nb.cutneighmax + 0.51 * rmax;
    const cutghostmin = nb.cutghost;
    if (cutneighmaxSrp > cutghostmin) {
      throw new StyleError(`Communication cutoff too small for fix ${this.style}. Need ${cutneighmaxSrp.toPrecision(8)}, current ${cutghostmin.toPrecision(8)}`);
    }
    // the new atoms into the box, sorted, ghosts and neighbor lists again (pbc, exchange, sort, borders, build)
    sys.atomsChanged();
    sys.pbc();
    sys.sortAtoms(true);
    nb.build(s, sys.geom, s.step);
    s.f.fill(0);
    // bond particles are in no group and do not move
    for (let i = 0; i < s.n; i++) {
      if (s.type[i] !== bptype) continue;
      s.mask[i] = 0;
      s.v[3 * i] = s.v[3 * i + 1] = s.v[3 * i + 2] = 0.0;
    }
  }

  /** pre_exchange: the bond particles move to the midpoint of their bonds before the neighbor build. */
  preExchange(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    // update ghosts
    nb.forwardComm(s, sys.geom);
    const tags = s.custom.get(SRP_TAGS)!.data;
    const map = buildAtomMap(s);
    const { head, next } = imageChains(nb.owner, nb.nall, nb.nlocal);
    const xall = nb.xall;
    for (let ii = 0; ii < s.n; ii++) {
      if (s.type[ii] !== this.bptype) continue;
      const px = s.x[3 * ii], py = s.x[3 * ii + 1], pz = s.x[3 * ii + 2];
      const a = map[tags[2 * ii]], b = map[tags[2 * ii + 1]];
      if (!(a >= 0) || !(b >= 0)) throw new StyleError(`Fix ${this.style} failed to map atom`);
      const i = closestImage(xall, head, next, a, px, py, pz);
      const j = closestImage(xall, head, next, b, px, py, pz);
      s.x[3 * ii] = (xall[3 * i] + xall[3 * j]) * 0.5;
      s.x[3 * ii + 1] = (xall[3 * i + 1] + xall[3 * j + 1]) * 0.5;
      s.x[3 * ii + 2] = (xall[3 * i + 2] + xall[3 * j + 2]) * 0.5;
    }
  }

  /** post_run: all bond particles are removed after each run (useful for write_data and write_restart). */
  postRun(): void {
    const sys = this.sys;
    const s = sys.state;
    const before = s.n;
    // AtomVec::copy (the delete loop and Atom::sort) moves x, v and the other per-atom data but not f: native's
    // forces stay with the storage slots, so after the run an atom shows the force of the slot it now occupies
    const ord0 = nativeOrder(s);
    const fSlot = new Float64Array(3 * before);
    for (let p = 0; p < before; p++) {
      const e = ord0[p];
      fSlot[3 * p] = s.f[3 * e]; fSlot[3 * p + 1] = s.f[3 * e + 1]; fSlot[3 * p + 2] = s.f[3 * e + 2];
    }
    const del = new Uint8Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.type[i] === this.bptype) del[i] = 1;
    deleteAtoms(s, del);
    sys.log(`Deleted ${before - s.n} atoms, new total = ${s.n}`);
    sys.atomsChanged();
    sys.pbc();
    sys.sortAtoms(true);
    const ord1 = nativeOrder(s);
    for (let p = 0; p < s.n; p++) {
      const e = ord1[p];
      s.f[3 * e] = fSlot[3 * p]; s.f[3 * e + 1] = fSlot[3 * p + 1]; s.f[3 * e + 2] = fSlot[3 * p + 2];
    }
    // output after the run (write_dump) shows these forces as native does; the next run's setup computes new ones
    sys.forcesCurrent();
  }
}
