import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { Neighbor, SBBITS, type NeighList, type NeighborNeeds } from '../src/engine/neighbor';
import type { SpecialList } from '../src/engine/atoms';
import type { SimState } from '../src/engine/types';

/*
 * Neighbor.buildList (wave: row-based bin enumeration) must hold exactly the pairs the
 * previous per-atom stencil scan held. refBuildList below is that previous algorithm copied
 * verbatim from this project's history (HEAD of src/engine/neighbor.ts, before the rewrite),
 * with `this` replaced by `self`. The membership rules it encodes are the ones in the class
 * comment of neighbor.ts: half lists keep an owned-owned pair once (lower index), an owned-ghost
 * pair only when the ghost lies lex-above; full lists keep every neighbor; special bits and
 * exclusions are applied per entry. Lists are compared as sets per owned atom, including the
 * special-bond bits carried in the top bits of each neighbor code.
 */

const emptyList = (n: number): NeighList => ({
  inum: n, numneigh: new Int32Array(n), firstneigh: new Int32Array(n + 1), neighbors: new Int32Array(0),
});

function refBuildList(self: any, s: SimState, needs: NeighborNeeds, fullList: boolean, skin: number): NeighList {
  const nlocal = self.nlocal, nall = self.nall;
  const xa = self.xall, ta = self.typeall, owner = self.owner;
  const nt = needs.ntypes + 1;
  const cutsq = new Float64Array(nt * nt);
  let cmax = 0;
  for (let k = 0; k < nt * nt; k++) {
    const c = needs.cutoff[k];
    if (c > 0) { cutsq[k] = (c + skin) * (c + skin); if (c + skin > cmax) cmax = c + skin; }
  }
  if (cmax <= 0) return emptyList(nlocal);
  // bins over the bounding box of all atoms, padded by the stencil reach so
  // the stencil is a flat list of index offsets with no bounds checks
  const binsize = self.binsize > 0 ? self.binsize : 0.5 * cmax;
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let k = 0; k < nall; k++) {
    const x = xa[3 * k], y = xa[3 * k + 1], z = xa[3 * k + 2];
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
    if (z < z0) z0 = z; if (z > z1) z1 = z;
  }
  const mx = Math.max(1, Math.min(512, Math.floor((x1 - x0) / binsize) + 1));
  const my = Math.max(1, Math.min(512, Math.floor((y1 - y0) / binsize) + 1));
  const mz = Math.max(1, Math.min(512, Math.floor((z1 - z0) / binsize) + 1));
  // bins are never narrower than binsize: a tiny but nonzero extent (atoms almost in a line or a
  // plane) would otherwise make the stencil reach ~cmax/extent bins and exhaust memory
  const bx = Math.max(binsize, (x1 - x0) / mx), by = Math.max(binsize, (y1 - y0) / my), bz = Math.max(binsize, (z1 - z0) / mz);
  const sx = Math.ceil(cmax / bx), sy = Math.ceil(cmax / by), sz = Math.ceil(cmax / bz);
  const nbx = mx + 2 * sx, nby = my + 2 * sy, nbz = mz + 2 * sz;
  const nbins = nbx * nby * nbz;
  const binOf = new Int32Array(nall);
  const binCount = new Int32Array(nbins + 1);
  for (let k = 0; k < nall; k++) {
    let ix = Math.floor((xa[3 * k] - x0) / bx); if (ix >= mx) ix = mx - 1; if (ix < 0) ix = 0;
    let iy = Math.floor((xa[3 * k + 1] - y0) / by); if (iy >= my) iy = my - 1; if (iy < 0) iy = 0;
    let iz = Math.floor((xa[3 * k + 2] - z0) / bz); if (iz >= mz) iz = mz - 1; if (iz < 0) iz = 0;
    const b = ((iz + sz) * nby + iy + sy) * nbx + ix + sx;
    binOf[k] = b;
    binCount[b + 1]++;
  }
  for (let b = 0; b < nbins; b++) binCount[b + 1] += binCount[b];
  const binStart = binCount;
  const fill = binStart.slice(0, nbins);
  // atoms in bin order, with their positions and types copied contiguously
  const binAtoms = new Int32Array(nall);
  for (let k = 0; k < nall; k++) binAtoms[fill[binOf[k]]++] = k;
  const bxs = new Float64Array(3 * nall);
  const bts = new Int32Array(nall);
  for (let p = 0; p < nall; p++) {
    const k = binAtoms[p];
    bxs[3 * p] = xa[3 * k]; bxs[3 * p + 1] = xa[3 * k + 1]; bxs[3 * p + 2] = xa[3 * k + 2];
    bts[p] = ta[k];
  }
  // stencil: bins whose closest point can be within cmax, as linear offsets
  const sten: number[] = [];
  for (let dz = -sz; dz <= sz; dz++) {
    for (let dy = -sy; dy <= sy; dy++) {
      for (let dx = -sx; dx <= sx; dx++) {
        const gx = Math.max(0, Math.abs(dx) - 1) * bx, gy = Math.max(0, Math.abs(dy) - 1) * by, gz = Math.max(0, Math.abs(dz) - 1) * bz;
        if (gx * gx + gy * gy + gz * gz < cmax * cmax) sten.push((dz * nby + dy) * nbx + dx);
      }
    }
  }
  const stencil = Int32Array.from(sten);
  const ns = stencil.length;
  const special = needs.special;
  const ss = needs.specialSettings;
  const exclude = needs.exclude ?? self.excludeFn(s);
  const incl = self.includeBit;
  const plain = !special && !exclude && !incl;
  const numneigh = new Int32Array(nlocal);
  const firstneigh = new Int32Array(nlocal + 1);
  let cap = Math.max(64, nlocal * (fullList ? 96 : 48));
  let neigh = new Int32Array(cap);
  let count = 0;
  const id = s.id;
  for (let i = 0; i < nlocal; i++) {
    firstneigh[i] = count;
    if (incl && !(s.mask[i] & incl)) { numneigh[i] = 0; continue; }
    if (count + 2048 > cap) {
      cap = Math.max(2 * cap, count + 4096);
      const nn = new Int32Array(cap);
      nn.set(neigh.subarray(0, count));
      neigh = nn;
    }
    const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
    const ti = ta[i] * nt;
    const b = binOf[i];
    const sp0 = special ? special.offset[i] : 0;
    const sp1 = special ? special.offset[i + 1] : 0;
    for (let q = 0; q < ns; q++) {
      const jb = b + stencil[q];
      const pEnd = binStart[jb + 1];
      for (let p = binStart[jb]; p < pEnd; p++) {
        const j = binAtoms[p];
        if (j === i) continue;
        const dx = bxs[3 * p] - xi, dy = bxs[3 * p + 1] - yi, dz = bxs[3 * p + 2] - zi;
        if (!fullList) {
          if (j < nlocal) { if (j < i) continue; }
          else if (dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0)))) continue;
        }
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= cutsq[ti + bts[p]]) continue;
        if (plain) {
          if (count === cap) {
            cap *= 2;
            const nn = new Int32Array(cap);
            nn.set(neigh);
            neigh = nn;
          }
          neigh[count++] = j;
          continue;
        }
        const jo = owner[j];
        if (incl && !(s.mask[jo] & incl)) continue;
        if (exclude && exclude(i, jo)) continue;
        let enc = j;
        if (sp1 > sp0) {
          const jid = id[jo];
          let skip = false;
          for (let k = sp0; k < sp1; k++) {
            if (special!.partner[k] !== jid) continue;
            const o = special!.order[k];
            const lj = ss.lj[o], cl = ss.coul[o];
            if (lj === 1 && cl === 1) break;
            if (lj === 0 && cl === 0 && !ss.keepExcluded) { skip = true; break; }
            enc = j | (o << SBBITS);
            break;
          }
          if (skip) continue;
        }
        if (count === cap) {
          cap *= 2;
          const nn = new Int32Array(cap);
          nn.set(neigh);
          neigh = nn;
        }
        neigh[count++] = enc;
      }
    }
    numneigh[i] = count - firstneigh[i];
  }
  firstneigh[nlocal] = count;
  return { inum: nlocal, numneigh, firstneigh, neighbors: neigh.subarray(0, count) };
}

/** Small deterministic PRNG (mulberry32). */
const mulberry32 = (seed: number) => {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** Per owned atom, the sorted encoded neighbors of a list. */
const rowsOf = (l: NeighList): number[][] => {
  const out: number[][] = [];
  for (let i = 0; i < l.inum; i++) {
    const row: number[] = [];
    for (let k = l.firstneigh[i]; k < l.firstneigh[i] + l.numneigh[i]; k++) row.push(l.neighbors[k]);
    out.push(row.sort((a, b) => a - b));
  }
  return out;
};

const expectSameLists = (got: NeighList, ref: NeighList, what: string): void => {
  expect(got.inum, what).toBe(ref.inum);
  const a = rowsOf(got), b = rowsOf(ref);
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length || a[i].some((v, k) => v !== b[i][k])) {
      throw new Error(`${what}: owned atom ${i} differs: got [${a[i]}] want [${b[i]}]`);
    }
  }
  expect(got.neighbors.length, what).toBe(ref.neighbors.length);
};

/** Random system on a fcc block, with a random box, types, specials, exclusions and include group. */
const randomCase = async (seed: number) => {
  const rng = mulberry32(seed * 7919 + 13);
  const pick = <T,>(xs: T[]): T => xs[Math.floor(rng() * xs.length)];
  const cells = 2 + Math.floor(rng() * 4);
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  await session.execute(`units lj
atom_style atomic
lattice fcc ${(0.6 + rng() * 0.6).toFixed(3)}
region box block 0 ${cells} 0 ${cells} 0 ${cells}
create_box 3 box
create_atoms 1 box
mass * 1.0
`);
  const sys = session.sys;
  const s: SimState = sys.state;
  const g = sys.geom;
  const box = g.box;
  const n = s.n;
  const ntypes = 1 + Math.floor(rng() * 3);
  // per-atom data: ids, types, chains of four (for specials), include-group bits, positions
  const chain = 4;
  for (let i = 0; i < n; i++) {
    s.id[i] = i + 1;
    s.type[i] = 1 + Math.floor(rng() * ntypes);
    s.molecule[i] = Math.floor(i / chain);
    s.mask[i] = rng() < 0.6 ? 1 | (rng() < 0.5 ? 2 : 0) : 2;
  }
  const lx = box.hi[0] - box.lo[0], ly = box.hi[1] - box.lo[1];
  for (let i = 0; i < n; i++) {
    if (i > 0 && rng() < 0.03) {
      // a coincident atom (zero separation) exercises the tie rules
      const j = Math.floor(rng() * i);
      for (let d = 0; d < 3; d++) s.x[3 * i + d] = s.x[3 * j + d];
      continue;
    }
    for (let d = 0; d < 3; d++) s.x[3 * i + d] += (rng() - 0.5) * 0.7;
  }
  // box: periodic or not per dimension, orthogonal or with tilt factors
  box.periodic = [rng() < 0.75, rng() < 0.75, rng() < 0.75];
  if (rng() < 0.4) {
    box.tilt = [(rng() - 0.5) * lx * 0.5, (rng() - 0.5) * lx * 0.5, (rng() - 0.5) * ly * 0.5];
    box.triclinic = true;
  } else {
    box.tilt = [0, 0, 0];
    box.triclinic = false;
  }
  g.update();

  // cutoffs per type pair, possibly asymmetric, possibly zero
  const nt = ntypes + 1;
  const symmetric = rng() < 0.6;
  const cutoff = new Float64Array(nt * nt);
  for (let a = 1; a <= ntypes; a++) {
    for (let b = 1; b <= ntypes; b++) {
      if (b < a && symmetric) { cutoff[a * nt + b] = cutoff[b * nt + a]; continue; }
      cutoff[a * nt + b] = rng() < 0.12 ? 0 : 0.8 + rng() * 1.8;
    }
  }
  cutoff[1 * nt + 1] = 1.2 + rng() * 1.5;

  // special bonds: 1-2, 1-3 and 1-4 partners inside each chain of four
  const offset = new Int32Array(n + 1);
  const partners: number[] = [], orders: number[] = [];
  const useSpecial = rng() < 0.7;
  for (let i = 0; i < n; i++) {
    offset[i] = partners.length;
    if (!useSpecial) continue;
    const m = Math.floor(i / chain), start = m * chain, end = Math.min(n, start + chain);
    for (let j = start; j < end; j++) {
      if (j === i) continue;
      const d = Math.abs(j - i);
      if (d === 1) { partners.push(s.id[j]); orders.push(1); }
      else if (d === 2) { partners.push(s.id[j]); orders.push(2); }
      else if (d === 3) { partners.push(s.id[j]); orders.push(3); }
    }
  }
  offset[n] = partners.length;
  const special: SpecialList | null = useSpecial
    ? { offset, partner: Int32Array.from(partners), order: Int8Array.from(orders) }
    : null;
  const weights = [0, 0.5, 1];
  const specialSettings = {
    lj: [1, pick(weights), pick(weights), pick(weights)] as [number, number, number, number],
    coul: [1, pick(weights), pick(weights), pick(weights)] as [number, number, number, number],
    keepExcluded: rng() < 0.5,
  };

  const exclFn = rng() < 0.5 ? (i: number, j: number) => (i + j) % 11 === 0 : null;
  const needs: NeighborNeeds = {
    half: rng() < 0.8,
    full: rng() < 0.4,
    cutoff,
    ntypes,
    special,
    specialSettings,
    exclude: exclFn,
  };
  if (!needs.half && !needs.full) needs.half = true;

  const nb: Neighbor = sys.nb;
  nb.skin = pick([0, 0.3, 0.6]);
  nb.binsize = pick([0, 0, 0.7, 1.9]);
  nb.includeBit = rng() < 0.3 ? 2 : 0;
  nb.excludes = rng() < 0.3 && !exclFn ? [{ kind: 'type', a: 1, b: 2 }] : [];
  nb.init(needs);
  nb.build(s, g, 0);
  return { nb, s, needs, rng, ntypes };
};

describe('neighbor list build matches the reference enumeration', () => {
  for (let seed = 1; seed <= 60; seed++) {
    it(`random configuration ${seed}`, async () => {
      const { nb, s, needs } = await randomCase(seed);
      if (needs.half) {
        const ref = refBuildList(nb, s, needs, false, nb.skin);
        expect(nb.half).not.toBeNull();
        expectSameLists(nb.half!, ref, `half seed ${seed}`);
      }
      if (needs.full) {
        const ref = refBuildList(nb, s, needs, true, nb.skin);
        expect(nb.full).not.toBeNull();
        expectSameLists(nb.full!, ref, `full seed ${seed}`);
      }
    });
  }

  it('occasional lists (skin 0, arbitrary cutoff) match the reference', async () => {
    for (let seed = 101; seed <= 110; seed++) {
      const { nb, s, rng, ntypes } = await randomCase(seed);
      const r = 0.6 + rng() * (nb.cutghost - 0.6);
      const full = rng() < 0.5;
      const nt = ntypes + 1;
      const needs: NeighborNeeds = {
        half: !full, full, cutoff: new Float64Array(nt * nt).fill(r), ntypes, special: null,
        specialSettings: { lj: [1, 1, 1, 1], coul: [1, 1, 1, 1], keepExcluded: true },
      };
      const got = nb.occasionalList(s, r, full, ntypes);
      const ref = refBuildList(nb, s, needs, full, 0);
      expectSameLists(got, ref, `occasional seed ${seed}`);
    }
  });

  it('an empty configuration gives empty lists', async () => {
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    await session.execute('units lj\natom_style atomic\nregion box block 0 4 0 4 0 4\ncreate_box 1 box\nmass * 1.0\n');
    const nb: Neighbor = session.sys.nb;
    nb.init({ half: true, full: false, cutoff: new Float64Array(4).fill(2.5), ntypes: 1, special: null, specialSettings: { lj: [1, 1, 1, 1], coul: [1, 1, 1, 1], keepExcluded: false } });
    nb.build(session.sys.state, session.sys.geom, 0);
    expect(nb.half!.neighbors.length).toBe(0);
    expect(nb.half!.inum).toBe(0);
  });
});
