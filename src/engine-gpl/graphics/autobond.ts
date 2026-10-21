/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/dump_image.cpp (DumpImage::create_image, the AUTO branch "render dynamic bonds for my atoms",
 * and the autobond keyword of DumpImage::DumpImage), src/GRAPHICS/image.cpp (Image::draw_cylinder, through draw.ts),
 * src/neighbor.cpp (Neighbor::get_best_pair_list), src/nbin_standard.cpp (NBinStandard::setup_bins), src/nbin.cpp
 * (NBin::coord2bin), src/nstencil_bin.cpp (the half stencil), src/npair_bin.cpp and src/npair_nsq.cpp (which side of
 * an owned/ghost pair a half list stores it on) and src/update.cpp (force->angstrom of each unit style).
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License (the LAMMPS file headers). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump_image.html, the autobond keyword: "The implicit bonds are found by searching the pair-wise
 * neighbor list for pairs of atoms that are closer than the bond cutoff; the second argument is the bond diameter. The
 * color of the bond is derived from the color of the atoms forming the implicit bond. For unit styles metal and real an
 * additional condition is applied: if the mass of both atoms of a pair within the bond cutoff is lower than 3 atomic
 * mass units, a bond is **not** drawn".
 *
 * Native walks the neighbor list get_best_pair_list returns (the first half list, else the first full list): for each
 * chosen owned atom1 and each chosen neighbor atom2, owned or ghost, closer than the cutoff it draws two cylinders, from
 * atom1 to the middle and from the middle to atom2, each in the colour and opacity of its own atom. A neighbor that is a
 * ghost is drawn at the ghost's position, so a bond that crosses a periodic boundary is drawn whole on the side of the
 * atom whose list holds the pair, reaching out of the box. With newton_pair off the pair is drawn only from the atom with
 * the smaller ID.
 *
 * The core hands over the engine's list (neighbor.ts), which holds the same pairs, and the positions of the build. The
 * engine's half list keeps an owned/ghost pair on the side whose ghost lies above (larger z, then y, then x); native's
 * side is set by the bins of the build (half/bin/newton, orthogonal) or by the parity of the two IDs (triclinic boxes and
 * neighbor nsq), which this file recomputes. (Native refuses autobond with neighbor style multi, as the core does.)
 */
import type { AutoBondSpec, RasterFrame, Vec3 } from '../../engine/output/image_objects';
import { drawCylinder } from './draw';

/** force->angstrom of each unit style (Update::set_units). */
const ANGSTROM: Record<string, number> = { lj: 1.0, real: 1.0, metal: 1.0, si: 1.0e-10, cgs: 1.0e-8, electron: 1.88972612, micro: 1.0e-4, nano: 1.0e-1 };

/** The standard bins of a list build (NBinStandard::setup_bins and NBin::coord2bin for an orthogonal box). */
function binner(lo: Vec3, hi: Vec3, binsizeUser: number, cutneighmax: number, dimension: 2 | 3): (x: Float64Array, k: number) => [number, number, number] {
  const bbox = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  let binsizeOptimal = binsizeUser > 0 ? binsizeUser : 0.5 * cutneighmax;
  if (binsizeOptimal === 0.0) binsizeOptimal = bbox[0];
  const binsizeinv = 1.0 / binsizeOptimal;
  const nbin = [Math.trunc(bbox[0] * binsizeinv), Math.trunc(bbox[1] * binsizeinv), dimension === 3 ? Math.trunc(bbox[2] * binsizeinv) : 1];
  for (let d = 0; d < 3; d++) if (nbin[d] === 0) nbin[d] = 1;
  const bininv = [1.0 / (bbox[0] / nbin[0]), 1.0 / (bbox[1] / nbin[1]), 1.0 / (bbox[2] / nbin[2])];
  const one = (v: number, d: number): number => {
    if (v >= hi[d]) return Math.trunc((v - hi[d]) * bininv[d]) + nbin[d];
    if (v >= lo[d]) return Math.min(Math.trunc((v - lo[d]) * bininv[d]), nbin[d] - 1);
    return Math.trunc((v - lo[d]) * bininv[d]) - 1;
  };
  return (x, k) => [one(x[3 * k], 0), one(x[3 * k + 1], 1), one(x[3 * k + 2], 2)];
}

/** Whether ghost g lies "above and to the right" of atom i (npair_bin.cpp, the same-bin ghost test). */
function above(x: Float64Array, i: number, g: number): boolean {
  if (x[3 * g + 2] < x[3 * i + 2]) return false;
  if (x[3 * g + 2] === x[3 * i + 2]) {
    if (x[3 * g + 1] < x[3 * i + 1]) return false;
    if (x[3 * g + 1] === x[3 * i + 1] && x[3 * g] < x[3 * i]) return false;
  }
  return true;
}

/** One bond to draw: atom1 (owned) and atom2 (owned, or a ghost at the position of its image). */
interface Pair { a1: number; a2: number; x2: Vec3 }

/** DumpImage::create_image (AUTO branch): bonds between chosen atom pairs of the neighbor list closer than the cutoff. */
export function drawAutoBonds(f: RasterFrame, b: AutoBondSpec): void {
  const L = b.list;
  if (!L) return;
  const cutsq = b.cutoff * b.cutoff;
  const x = b.x;
  const n = b.nlocal;
  const rank = new Int32Array(n);
  for (let k = 0; k < n; k++) rank[b.order[k]] = k;
  const xb = L.xBuilt;
  const bins = L.triclinic || L.style === 'nsq' ? null : binner(L.boxLo, L.boxHi, L.binsize, L.cutneighmax, L.dimension);
  const delta = 0.01 * (ANGSTROM[b.units] ?? 1.0);
  // half list, newton on: whether native's list holds the owned/ghost pair (i, ghost g) at i (npair_bin.cpp and
  // npair_nsq.cpp, built positions)
  const storedAt = (i: number, g: number, o: number): boolean => {
    if (bins) {
      const bi = bins(xb, i), bg = bins(xb, g);
      const dk = bg[2] - bi[2], dj = bg[1] - bi[1], di = bg[0] - bi[0];
      // nstencil_bin.cpp, half stencil: "k > 0 || j > 0 || (j == 0 && i > 0)", the own bin first
      if (dk !== 0 || dj !== 0 || di !== 0) return dk > 0 || (dk === 0 && (dj > 0 || (dj === 0 && di > 0)));
      return above(xb, i, g);
    }
    const itag = b.tag[i], jtag = b.tag[o];
    if (itag > jtag) return (itag + jtag) % 2 !== 0;
    if (itag < jtag) return (itag + jtag) % 2 !== 1;
    if (!L.triclinic) return above(xb, i, g);
    if (Math.abs(xb[3 * g + 2] - xb[3 * i + 2]) > delta) return !(xb[3 * g + 2] < xb[3 * i + 2]);
    if (Math.abs(xb[3 * g + 1] - xb[3 * i + 1]) > delta) return !(xb[3 * g + 1] < xb[3 * i + 1]);
    return !(xb[3 * g] < xb[3 * i]);
  };
  const ghostAt = (o: number, g: number, sign: 1 | -1): Vec3 => {
    const out: Vec3 = [0, 0, 0];
    b.imagePosition(o, sign * L.image[3 * g], sign * L.image[3 * g + 1], sign * L.image[3 * g + 2], out);
    return out;
  };
  const pairs: Pair[] = [];
  for (let i = 0; i < n; i++) {
    const start = L.firstneigh[i], end = start + L.numneigh[i];
    for (let k = start; k < end; k++) {
      const j = L.neighbors[k] & L.mask;
      const o = L.owner[j];
      if (L.full) {
        // a full list holds every pair from both sides; with newton_pair off native draws it from the smaller ID only
        if (!b.newtonPair && b.tag[i] > b.tag[o]) continue;
        pairs.push({ a1: i, a2: o, x2: j < n ? [x[3 * j], x[3 * j + 1], x[3 * j + 2]] : ghostAt(o, j, 1) });
      } else if (j < n) {
        // owned pair: newton on stores it once; newton off stores it at the lower index and draws it only when that atom
        // has the smaller ID
        const [a1, a2] = rank[i] < rank[j] ? [i, j] : [j, i];
        if (!b.newtonPair && b.tag[a1] > b.tag[a2]) continue;
        pairs.push({ a1, a2, x2: [x[3 * a2], x[3 * a2 + 1], x[3 * a2 + 2]] });
      } else if (!b.newtonPair) {
        // newton off: the list holds the pair on both sides and the smaller ID draws it (both when it is an image of itself)
        if (b.tag[i] <= b.tag[o]) pairs.push({ a1: i, a2: o, x2: ghostAt(o, j, 1) });
        if (b.tag[o] <= b.tag[i]) pairs.push({ a1: o, a2: i, x2: ghostAt(i, j, -1) });
      } else if (storedAt(i, j, o)) {
        pairs.push({ a1: i, a2: o, x2: ghostAt(o, j, 1) });
      } else {
        pairs.push({ a1: o, a2: i, x2: ghostAt(i, j, -1) });
      }
    }
  }
  // native's order: the chosen atoms in storage order, each with its list
  const order = pairs.map((_, k) => k).sort((p, q) => rank[pairs[p].a1] - rank[pairs[q].a1] || p - q);
  const del: Vec3 = [0, 0, 0];
  for (const k of order) {
    const { a1, a2, x2 } = pairs[k];
    if (!b.selected(a1) || !b.selected(a2)) continue;
    // the hydrogen rule of the docs: both masses below 3 means no bond (only in units metal and real)
    if (b.mass && b.mass(a1) < 3.0 && b.mass(a2) < 3.0) continue;
    const x1: Vec3 = [x[3 * a1], x[3 * a1 + 1], x[3 * a1 + 2]];
    del[0] = x2[0] - x1[0];
    del[1] = x2[1] - x1[1];
    del[2] = x2[2] - x1[2];
    const rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
    if (!(rsq < cutsq)) continue;
    // native: domain->minimum_image(dx), xmid = x1 + 0.5 dx for atom 1, xmid = x2 - 0.5 dx for atom 2
    b.minimumImage(del);
    const mid1: Vec3 = [x1[0] + 0.5 * del[0], x1[1] + 0.5 * del[1], x1[2] + 0.5 * del[2]];
    const mid2: Vec3 = [x2[0] - 0.5 * del[0], x2[1] - 0.5 * del[1], x2[2] - 0.5 * del[2]];
    drawCylinder(f, x1, mid1, b.colorOf(a1), b.width, 3, b.opacityOf(a1));
    drawCylinder(f, mid2, x2, b.colorOf(a2), b.width, 3, b.opacityOf(a2));
  }
}
