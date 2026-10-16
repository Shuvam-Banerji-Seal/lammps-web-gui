/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0, which bundles voro++ 0.4.6.
 * Source: voro++ 0.4.6 src/container.cc and src/container.hh (container: the non-periodic monodisperse container
 * with put and compute_cell through the block search), src/v_base.cc (voro_base: the mrad table), src/v_compute.cc
 * (voro_compute::compute_cell, the plane tests corner_test ... face_z_test, compute_min_max_radius, add_to_mask,
 * scan_bits_mask_add, add_list_memory), src/c_loops.hh (c_loop_all: block order i fastest, then j, then k).
 * The radius policy is the monodisperse one of rad_option.hh (radius_mono) as in the LAMMPS build of voro++.
 *
 * voro++ is Copyright (c) 2008, The Regents of the University of California, through Lawrence Berkeley National
 * Laboratory, and Copyright (c) 2008-2011 Chris H. Rycroft, BSD three-clause licence (LICENSE.voro++).
 * Ported as C: the block lists, the worklist symmetry bits, the mask counter and the queue ring are the same.
 */
import { VoroCellNeighbor } from './cell';
import { WL, WL_FGRID, WL_HGRID, WL_HGRIDCU, WL_SEQ_LENGTH } from './worklist';

const LARGE_NUMBER = 1e30;

/** C int() conversion: truncation toward zero. */
const cint = (a: number): number => Math.trunc(a);

/** voro_base::step_int */
const stepInt = (a: number): number => (a < 0 ? Math.trunc(a) - 1 : Math.trunc(a));

const B1 = 1 << 21, B2 = 1 << 22, B3 = 1 << 24, B4 = 1 << 25, B5 = 1 << 27, B6 = 1 << 28;

/** One block's particle list: ids and coordinates (stride 3). */
interface Block {
  id: Int32Array;
  p: Float64Array;
}

/**
 * Non-periodic monodisperse voro++ container (container class with xperiodic = yperiodic = zperiodic = false).
 * Holds the particles in blocks and computes each particle's Voronoi cell with the worklist search.
 */
export class VoroContainer {
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly nxy: number;
  readonly nxyz: number;
  readonly boxx: number;
  readonly boxy: number;
  readonly boxz: number;
  readonly xsp: number;
  readonly ysp: number;
  readonly zsp: number;
  readonly ax: number;
  readonly ay: number;
  readonly az: number;
  readonly bx: number;
  readonly by: number;
  readonly bz: number;
  /** blocks (nxyz), particle counts and capacities */
  readonly blocks: Block[];
  readonly co: Int32Array;
  readonly mem: Int32Array;
  readonly mrad: Float64Array;
  private readonly bxsq: number;
  private mv = 0;
  private quSize: number;
  private qu: Int32Array;
  private readonly mask: Uint32Array;
  private crs = 0;

  constructor(ax: number, bx: number, ay: number, by: number, az: number, bz: number, nx: number, ny: number, nz: number) {
    this.ax = ax; this.bx = bx; this.ay = ay; this.by = by; this.az = az; this.bz = bz;
    this.nx = nx; this.ny = ny; this.nz = nz;
    this.nxy = nx * ny;
    this.nxyz = this.nxy * nz;
    this.boxx = (bx - ax) / nx;
    this.boxy = (by - ay) / ny;
    this.boxz = (bz - az) / nz;
    this.xsp = 1 / this.boxx;
    this.ysp = 1 / this.boxy;
    this.zsp = 1 / this.boxz;
    this.bxsq = this.boxx * this.boxx + this.boxy * this.boxy + this.boxz * this.boxz;
    this.co = new Int32Array(this.nxyz);
    this.mem = new Int32Array(this.nxyz).fill(8);
    this.blocks = [];
    for (let l = 0; l < this.nxyz; l++) this.blocks.push({ id: new Int32Array(8), p: new Float64Array(3 * 8) });
    this.mrad = new Float64Array(WL_HGRIDCU * WL_SEQ_LENGTH);
    this.buildMrad();
    this.quSize = 3 * (3 + this.nx * this.ny + this.nz * (this.nx + this.ny));
    this.qu = new Int32Array(this.quSize);
    this.mask = new Uint32Array(this.nx * this.ny * this.nz);
  }

  /** voro_base::voro_base: the minimum squared distance from each worklist block to the particle's region. */
  private buildMrad(): void {
    const xstep = this.boxx / WL_FGRID, ystep = this.boxy / WL_FGRID, zstep = this.boxz / WL_FGRID;
    let e = 0;
    let radp = 0;
    let zlo = 0, zhi = zstep;
    for (let lz = 0; lz < WL_HGRID; zlo = zhi, zhi += zstep, lz++) {
      let ylo = 0, yhi = ystep;
      for (let ly = 0; ly < WL_HGRID; ylo = yhi, yhi += ystep, ly++) {
        let xlo = 0, xhi = xstep;
        for (let lx = 0; lx < WL_HGRID; xlo = xhi, xhi += xstep, lx++) {
          const minr = { v: LARGE_NUMBER };
          let q: number;
          for (q = WL[e] + 1; q < WL_SEQ_LENGTH; q++) {
            const f = WL[e + q];
            let i = (f & 127) - 64;
            const j = ((f >>> 7) & 127) - 64;
            const k = ((f >>> 14) & 127) - 64;
            if ((f & B2) === B2) {
              this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i - 1, j, k);
              if ((f & B1) === 0) this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i + 1, j, k);
            } else if ((f & B1) === B1) this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i + 1, j, k);
            if ((f & B4) === B4) {
              this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i, j - 1, k);
              if ((f & B3) === 0) this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i, j + 1, k);
            } else if ((f & B3) === B3) this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i, j + 1, k);
            if ((f & B6) === B6) {
              this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i, j, k - 1);
              if ((f & B5) === 0) this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i, j, k + 1);
            } else if ((f & B5) === B5) this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i, j, k + 1);
          }
          q--;
          while (q > 0) {
            this.mrad[radp + q] = minr.v;
            const f = WL[e + q];
            const i = (f & 127) - 64;
            const j = ((f >>> 7) & 127) - 64;
            const k = ((f >>> 14) & 127) - 64;
            this.computeMinimum(minr, xlo, xhi, ylo, yhi, zlo, zhi, i, j, k);
            q--;
          }
          this.mrad[radp] = minr.v;
          e += WL_SEQ_LENGTH;
          radp += WL_SEQ_LENGTH;
        }
      }
    }
  }

  private computeMinimum(minr: { v: number }, xlo: number, xhi: number, ylo: number, yhi: number, zlo: number, zhi: number, ti: number, tj: number, tk: number): void {
    let radsq: number, temp: number;
    if (ti > 0) { temp = this.boxx * ti - xhi; radsq = temp * temp; }
    else if (ti < 0) { temp = xlo - this.boxx * (1 + ti); radsq = temp * temp; }
    else radsq = 0;
    if (tj > 0) { temp = this.boxy * tj - yhi; radsq += temp * temp; }
    else if (tj < 0) { temp = ylo - this.boxy * (1 + tj); radsq += temp * temp; }
    if (tk > 0) { temp = this.boxz * tk - zhi; radsq += temp * temp; }
    else if (tk < 0) { temp = zlo - this.boxz * (1 + tk); radsq += temp * temp; }
    if (radsq < minr.v) minr.v = radsq;
  }

  /** container::put (put_remap and put_locate_block, monodisperse); particles outside the bounds are dropped. */
  put(n: number, x: number, y: number, z: number): void {
    const i = stepInt((x - this.ax) * this.xsp);
    if (i < 0 || i >= this.nx) return;
    const j = stepInt((y - this.ay) * this.ysp);
    if (j < 0 || j >= this.ny) return;
    const k = stepInt((z - this.az) * this.zsp);
    if (k < 0 || k >= this.nz) return;
    const ijk = i + this.nx * j + this.nxy * k;
    const b = this.blocks[ijk];
    const c = this.co[ijk];
    if (c === this.mem[ijk]) {
      const cap = this.mem[ijk] * 2;
      const nid = new Int32Array(cap);
      nid.set(b.id.subarray(0, c));
      const np = new Float64Array(3 * cap);
      np.set(b.p.subarray(0, 3 * c));
      b.id = nid;
      b.p = np;
      this.mem[ijk] = cap;
    }
    b.id[c] = n;
    b.p[3 * c] = x;
    b.p[3 * c + 1] = y;
    b.p[3 * c + 2] = z;
    this.co[ijk]++;
  }

  /**
   * voro_compute::compute_cell for the particle q of block ijk at block (ci, cj, ck). Returns false when the
   * cell is cut away (the caller then skips the particle, as native does).
   */
  computeCell(c: VoroCellNeighbor, ijk0: number, s: number, ci: number, cj: number, ck: number): boolean {
    const nx = this.nx, ny = this.ny, nz = this.nz;
    const ax = this.ax, ay = this.ay, az = this.az, bx = this.bx, by = this.by, bz = this.bz;
    const boxx = this.boxx, boxy = this.boxy, boxz = this.boxz;
    const own = this.blocks[ijk0];
    const x = own.p[3 * s], y = own.p[3 * s + 1], z = own.p[3 * s + 2];
    // initialize_voronoicell (non-periodic): the box bounds relative to the particle
    const x1b = ax - x, x2b = bx - x, y1b = ay - y, y2b = by - y, z1b = az - z, z2b = bz - z;
    const ii = ci, jj = cj, kk = ck;
    c.init(x1b, x2b, y1b, y2b, z1b, z2b);
    let x1: number, y1: number, z1: number, rs: number;
    let qx = 0, qy = 0, qz = 0;
    // test all particles in the particle's local block first
    let l = 0;
    for (l = 0; l < s; l++) {
      x1 = own.p[3 * l] - x;
      y1 = own.p[3 * l + 1] - y;
      z1 = own.p[3 * l + 2] - z;
      rs = x1 * x1 + y1 * y1 + z1 * z1;
      if (!c.nplane(x1, y1, z1, rs, own.id[l])) return false;
    }
    l++;
    while (l < this.co[ijk0]) {
      x1 = own.p[3 * l] - x;
      y1 = own.p[3 * l + 1] - y;
      z1 = own.p[3 * l + 2] - z;
      rs = x1 * x1 + y1 * y1 + z1 * z1;
      if (!c.nplane(x1, y1, z1, rs, own.id[l])) return false;
      l++;
    }
    let mrs = c.maxRadiusSquared();
    // fractional position within the block and the worklist index
    const fx = x - ax - boxx * ci;
    const fy = y - ay - boxy * cj;
    const fz = z - az - boxz * ck;
    let di = cint(fx * this.xsp * WL_FGRID);
    let dj = cint(fy * this.ysp * WL_FGRID);
    let dk = cint(fz * this.zsp * WL_FGRID);
    let m1: number, m2: number;
    let gxs: number, gys: number, gzs: number;
    if (di >= WL_HGRID) {
      gxs = fx;
      m1 = 127 + (3 << 21);
      m2 = 1 + (1 << 21);
      di = WL_FGRID - 1 - di;
      if (di < 0) di = 0;
    } else {
      m1 = 0; m2 = 0;
      gxs = boxx - fx;
    }
    if (dj >= WL_HGRID) {
      gys = fy;
      m1 |= (127 << 7) + (3 << 24);
      m2 |= (1 << 7) + (1 << 24);
      dj = WL_FGRID - 1 - dj;
      if (dj < 0) dj = 0;
    } else gys = boxy - fy;
    if (dk >= WL_HGRID) {
      gzs = fz;
      m1 |= (127 << 14) + (3 << 27);
      m2 |= (1 << 14) + (1 << 27);
      dk = WL_FGRID - 1 - dk;
      if (dk < 0) dk = 0;
    } else gzs = boxz - fz;
    gxs *= gxs;
    gys *= gys;
    gzs *= gzs;
    const wlIdx = di + WL_HGRID * (dj + WL_HGRID * dk);
    const radBase = wlIdx * WL_SEQ_LENGTH;
    const eBase = wlIdx * WL_SEQ_LENGTH;
    const radp = (g: number) => this.mrad[radBase + g];
    const e = (g: number) => WL[eBase + g];
    const f = e(0);
    let g = 0;
    let q: number;
    let ei: number, ej: number, ek: number;
    const hx = nx, hy = ny, hz = nz;
    const hxy = nx * ny;
    const countList = [7, 11, 15, 19, 26, 35, 45, 59];
    let countIdx = 0;
    let nextCount = 3;
    // the first worklist part
    do {
      if (g === nextCount) {
        mrs = c.maxRadiusSquared();
        if (countIdx !== countList.length) nextCount = countList[countIdx++];
      }
      if (radp(g) > mrs) return true;
      g++;
      q = (e(g) ^ m1) >>> 0;
      q = (q + m2) >>> 0;
      di = q & 127; di -= 64;
      dj = (q >>> 7) & 127; dj -= 64;
      dk = (q >>> 14) & 127; dk -= 64;
      ei = di + ii;
      if (ei < 0 || ei >= hx) continue;
      ej = dj + jj;
      if (ej < 0 || ej >= hy) continue;
      ek = dk + kk;
      if (ek < 0 || ek >= hz) continue;
      if (this.computeMinMaxRadius(di, dj, dk, fx, fy, fz, gxs, gys, gzs, mrs)) continue;
      const crs = this.crs;
      const bi = ei + nx * (ej + ny * ek);
      if (this.co[bi] > 0) {
        const bl = this.blocks[bi];
        l = 0;
        const x2 = x - qx, y2 = y - qy, z2 = z - qz;
        if (!(crs > mrs)) {
          do {
            x1 = bl.p[3 * l] - x2;
            y1 = bl.p[3 * l + 1] - y2;
            z1 = bl.p[3 * l + 2] - z2;
            rs = x1 * x1 + y1 * y1 + z1 * z1;
            if (!c.nplane(x1, y1, z1, rs, bl.id[l])) return false;
            l++;
          } while (l < this.co[bi]);
        } else {
          do {
            x1 = bl.p[3 * l] - x2;
            y1 = bl.p[3 * l + 1] - y2;
            z1 = bl.p[3 * l + 2] - z2;
            rs = x1 * x1 + y1 * y1 + z1 * z1;
            if (rs < mrs && !c.nplane(x1, y1, z1, rs, bl.id[l])) return false;
            l++;
          } while (l < this.co[bi]);
        }
      }
    } while (g < f);
    // the rest of the worklist marks the mask and queues the neighbours not on it
    this.mv++;
    if (this.mv === 0) {
      this.mask.fill(0);
      this.mv = 1;
    }
    let qs = 0, qe = 0;
    const mv = this.mv;
    while (g < WL_SEQ_LENGTH - 1) {
      if (g === nextCount) {
        mrs = c.maxRadiusSquared();
        if (countIdx !== countList.length) nextCount = countList[countIdx++];
      }
      if (radp(g) > mrs) return true;
      g++;
      q = (e(g) ^ m1) >>> 0;
      q = (q + m2) >>> 0;
      di = q & 127; di -= 64;
      dj = (q >>> 7) & 127; dj -= 64;
      dk = (q >>> 14) & 127; dk -= 64;
      ei = di + ii;
      if (ei < 0 || ei >= hx) continue;
      ej = dj + jj;
      if (ej < 0 || ej >= hy) continue;
      ek = dk + kk;
      if (ek < 0 || ek >= hz) continue;
      const mi = ei + hx * (ej + hy * ek);
      this.mask[mi] = mv;
      if (this.computeMinMaxRadius(di, dj, dk, fx, fy, fz, gxs, gys, gzs, mrs)) continue;
      const crs = this.crs;
      const bi = ei + nx * (ej + ny * ek);
      if (this.co[bi] > 0) {
        const bl = this.blocks[bi];
        l = 0;
        const x2 = x - qx, y2 = y - qy, z2 = z - qz;
        if (!(crs > mrs)) {
          do {
            x1 = bl.p[3 * l] - x2;
            y1 = bl.p[3 * l + 1] - y2;
            z1 = bl.p[3 * l + 2] - z2;
            rs = x1 * x1 + y1 * y1 + z1 * z1;
            if (!c.nplane(x1, y1, z1, rs, bl.id[l])) return false;
            l++;
          } while (l < this.co[bi]);
        } else {
          do {
            x1 = bl.p[3 * l] - x2;
            y1 = bl.p[3 * l + 1] - y2;
            z1 = bl.p[3 * l + 2] - z2;
            rs = x1 * x1 + y1 * y1 + z1 * z1;
            if (rs < mrs && !c.nplane(x1, y1, z1, rs, bl.id[l])) return false;
            l++;
          } while (l < this.co[bi]);
        }
      }
      if (qe > this.quSize - 18) [qs, qe] = this.addListMemory(qs, qe);
      qe = this.scanBitsMaskAdd(q, mi, ei, ej, ek, mv, qe);
    }
    if (radp(g) > mrs) return true;
    // block by block over the queue
    while (qs !== qe) {
      if (qs === this.quSize) qs = 0;
      ei = this.qu[qs++];
      ej = this.qu[qs++];
      ek = this.qu[qs++];
      const xlo = (ei - ii) * boxx - fx, xhi = xlo + boxx;
      const ylo = (ej - jj) * boxy - fy, yhi = ylo + boxy;
      const zlo = (ek - kk) * boxz - fz, zhi = zlo + boxz;
      if (ei > ii) {
        if (ej > jj) {
          if (ek > kk) { if (this.cornerTest(c, xlo, ylo, zlo, xhi, yhi, zhi)) continue; }
          else if (ek < kk) { if (this.cornerTest(c, xlo, ylo, zhi, xhi, yhi, zlo)) continue; }
          else { if (this.edgeZTest(c, xlo, ylo, zlo, xhi, yhi, zhi)) continue; }
        } else if (ej < jj) {
          if (ek > kk) { if (this.cornerTest(c, xlo, yhi, zlo, xhi, ylo, zhi)) continue; }
          else if (ek < kk) { if (this.cornerTest(c, xlo, yhi, zhi, xhi, ylo, zlo)) continue; }
          else { if (this.edgeZTest(c, xlo, yhi, zlo, xhi, ylo, zhi)) continue; }
        } else {
          if (ek > kk) { if (this.edgeYTest(c, xlo, ylo, zlo, xhi, yhi, zhi)) continue; }
          else if (ek < kk) { if (this.edgeYTest(c, xlo, ylo, zhi, xhi, yhi, zlo)) continue; }
          else { if (this.faceXTest(c, xlo, ylo, zlo, yhi, zhi)) continue; }
        }
      } else if (ei < ii) {
        if (ej > jj) {
          if (ek > kk) { if (this.cornerTest(c, xhi, ylo, zlo, xlo, yhi, zhi)) continue; }
          else if (ek < kk) { if (this.cornerTest(c, xhi, ylo, zhi, xlo, yhi, zlo)) continue; }
          else { if (this.edgeZTest(c, xhi, ylo, zlo, xlo, yhi, zhi)) continue; }
        } else if (ej < jj) {
          if (ek > kk) { if (this.cornerTest(c, xhi, yhi, zlo, xlo, ylo, zhi)) continue; }
          else if (ek < kk) { if (this.cornerTest(c, xhi, yhi, zhi, xlo, ylo, zlo)) continue; }
          else { if (this.edgeZTest(c, xhi, yhi, zlo, xlo, ylo, zhi)) continue; }
        } else {
          if (ek > kk) { if (this.edgeYTest(c, xhi, ylo, zlo, xlo, yhi, zhi)) continue; }
          else if (ek < kk) { if (this.edgeYTest(c, xhi, ylo, zhi, xlo, yhi, zlo)) continue; }
          else { if (this.faceXTest(c, xhi, ylo, zlo, yhi, zhi)) continue; }
        }
      } else {
        if (ej > jj) {
          if (ek > kk) { if (this.edgeXTest(c, xlo, ylo, zlo, xhi, yhi, zhi)) continue; }
          else if (ek < kk) { if (this.edgeXTest(c, xlo, ylo, zhi, xhi, yhi, zlo)) continue; }
          else { if (this.faceYTest(c, xlo, ylo, zlo, xhi, zhi)) continue; }
        } else if (ej < jj) {
          if (ek > kk) { if (this.edgeXTest(c, xlo, yhi, zlo, xhi, ylo, zhi)) continue; }
          else if (ek < kk) { if (this.edgeXTest(c, xlo, yhi, zhi, xhi, ylo, zlo)) continue; }
          else { if (this.faceYTest(c, xlo, yhi, zlo, xhi, zhi)) continue; }
        } else {
          if (ek > kk) { if (this.faceZTest(c, xlo, ylo, zlo, xhi, yhi)) continue; }
          else if (ek < kk) { if (this.faceZTest(c, xlo, ylo, zhi, xhi, yhi)) continue; }
          else throw new Error('voro++ internal error: compute cell routine revisiting central block');
        }
      }
      const bi = ei + nx * (ej + ny * ek);
      if (this.co[bi] > 0) {
        const bl = this.blocks[bi];
        l = 0;
        const x2 = x - qx, y2 = y - qy, z2 = z - qz;
        do {
          x1 = bl.p[3 * l] - x2;
          y1 = bl.p[3 * l + 1] - y2;
          z1 = bl.p[3 * l + 2] - z2;
          rs = x1 * x1 + y1 * y1 + z1 * z1;
          if (!c.nplane(x1, y1, z1, rs, bl.id[l])) return false;
          l++;
        } while (l < this.co[bi]);
      }
      if ((qs <= qe ? this.quSize - qe + qs : qs - qe) < 18) [qs, qe] = this.addListMemory(qs, qe);
      // add_to_mask
      const mi = ei + hx * (ej + hy * ek);
      if (ek > 0 && this.mask[mi - hxy] !== mv) {
        if (qe === this.quSize) qe = 0;
        this.mask[mi - hxy] = mv;
        this.qu[qe++] = ei; this.qu[qe++] = ej; this.qu[qe++] = ek - 1;
      }
      if (ej > 0 && this.mask[mi - hx] !== mv) {
        if (qe === this.quSize) qe = 0;
        this.mask[mi - hx] = mv;
        this.qu[qe++] = ei; this.qu[qe++] = ej - 1; this.qu[qe++] = ek;
      }
      if (ei > 0 && this.mask[mi - 1] !== mv) {
        if (qe === this.quSize) qe = 0;
        this.mask[mi - 1] = mv;
        this.qu[qe++] = ei - 1; this.qu[qe++] = ej; this.qu[qe++] = ek;
      }
      if (ei < hx - 1 && this.mask[mi + 1] !== mv) {
        if (qe === this.quSize) qe = 0;
        this.mask[mi + 1] = mv;
        this.qu[qe++] = ei + 1; this.qu[qe++] = ej; this.qu[qe++] = ek;
      }
      if (ej < hy - 1 && this.mask[mi + hx] !== mv) {
        if (qe === this.quSize) qe = 0;
        this.mask[mi + hx] = mv;
        this.qu[qe++] = ei; this.qu[qe++] = ej + 1; this.qu[qe++] = ek;
      }
      if (ek < hz - 1 && this.mask[mi + hxy] !== mv) {
        if (qe === this.quSize) qe = 0;
        this.mask[mi + hxy] = mv;
        this.qu[qe++] = ei; this.qu[qe++] = ej; this.qu[qe++] = ek + 1;
      }
    }
    return true;
  }

  /** scan_bits_mask_add: marks the neighbours of a worklist block that are not on the worklist. */
  private scanBitsMaskAdd(q: number, mi: number, ei: number, ej: number, ek: number, mv: number, qe0: number): number {
    const hx = this.nx, hy = this.ny, hz = this.nz;
    const hxy = hx * hy;
    let qe = qe0;
    const push = (a: number, b: number, c: number) => { this.qu[qe++] = a; this.qu[qe++] = b; this.qu[qe++] = c; };
    if ((q & B2) === B2) {
      if (ei > 0) { this.mask[mi - 1] = mv; push(ei - 1, ej, ek); }
      if ((q & B1) === 0 && ei < hx - 1) { this.mask[mi + 1] = mv; push(ei + 1, ej, ek); }
    } else if ((q & B1) === B1 && ei < hx - 1) { this.mask[mi + 1] = mv; push(ei + 1, ej, ek); }
    if ((q & B4) === B4) {
      if (ej > 0) { this.mask[mi - hx] = mv; push(ei, ej - 1, ek); }
      if ((q & B3) === 0 && ej < hy - 1) { this.mask[mi + hx] = mv; push(ei, ej + 1, ek); }
    } else if ((q & B3) === B3 && ej < hy - 1) { this.mask[mi + hx] = mv; push(ei, ej + 1, ek); }
    if ((q & B6) === B6) {
      if (ek > 0) { this.mask[mi - hxy] = mv; push(ei, ej, ek - 1); }
      if ((q & B5) === 0 && ek < hz - 1) { this.mask[mi + hxy] = mv; push(ei, ej, ek + 1); }
    } else if ((q & B5) === B5 && ek < hz - 1) { this.mask[mi + hxy] = mv; push(ei, ej, ek + 1); }
    return qe;
  }

  /** add_list_memory: doubles the queue and moves its ring contents to the start; returns the new (qs, qe). */
  private addListMemory(qs: number, qe: number): [number, number] {
    this.quSize <<= 1;
    const qn = new Int32Array(this.quSize);
    let c = 0;
    if (qs <= qe) {
      for (let s = qs; s < qe; s++) qn[c++] = this.qu[s];
    } else {
      for (let s = qs; s < this.qu.length; s++) qn[c++] = this.qu[s];
      for (let s = 0; s < qe; s++) qn[c++] = this.qu[s];
    }
    this.qu = qn;
    return [0, c];
  }

  /** compute_min_max_radius: the minimum (crs, stored in this.crs) and maximum distances; true when the block is beyond mrs. */
  private computeMinMaxRadius(di: number, dj: number, dk: number, fx: number, fy: number, fz: number, gxs: number, gys: number, gzs: number, mrs: number): boolean {
    const boxx = this.boxx, boxy = this.boxy, boxz = this.boxz, bxsq = this.bxsq;
    let xlo: number, ylo: number, zlo: number;
    let crs: number;
    const ct = (v: number) => v > mrs;
    if (di > 0) {
      xlo = di * boxx - fx;
      crs = xlo * xlo;
      if (dj > 0) {
        ylo = dj * boxy - fy;
        crs += ylo * ylo;
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += bxsq + 2 * (boxx * xlo + boxy * ylo + boxz * zlo);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += bxsq + 2 * (boxx * xlo + boxy * ylo - boxz * zlo);
        } else {
          if (ct(crs)) return this.ret(crs, true);
          crs += boxx * (2 * xlo + boxx) + boxy * (2 * ylo + boxy) + gzs;
        }
      } else if (dj < 0) {
        ylo = (dj + 1) * boxy - fy;
        crs += ylo * ylo;
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += bxsq + 2 * (boxx * xlo - boxy * ylo + boxz * zlo);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += bxsq + 2 * (boxx * xlo - boxy * ylo - boxz * zlo);
        } else {
          if (ct(crs)) return this.ret(crs, true);
          crs += boxx * (2 * xlo + boxx) + boxy * (-2 * ylo + boxy) + gzs;
        }
      } else {
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (2 * zlo + boxz);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (-2 * zlo + boxz);
        } else {
          if (ct(crs)) return this.ret(crs, true);
          crs += gzs;
        }
        crs += gys + boxx * (2 * xlo + boxx);
      }
    } else if (di < 0) {
      xlo = (di + 1) * boxx - fx;
      crs = xlo * xlo;
      if (dj > 0) {
        ylo = dj * boxy - fy;
        crs += ylo * ylo;
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += bxsq + 2 * (-boxx * xlo + boxy * ylo + boxz * zlo);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += bxsq + 2 * (-boxx * xlo + boxy * ylo - boxz * zlo);
        } else {
          if (ct(crs)) return this.ret(crs, true);
          crs += boxx * (-2 * xlo + boxx) + boxy * (2 * ylo + boxy) + gzs;
        }
      } else if (dj < 0) {
        ylo = (dj + 1) * boxy - fy;
        crs += ylo * ylo;
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += bxsq + 2 * (-boxx * xlo - boxy * ylo + boxz * zlo);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += bxsq + 2 * (-boxx * xlo - boxy * ylo - boxz * zlo);
        } else {
          if (ct(crs)) return this.ret(crs, true);
          crs += boxx * (-2 * xlo + boxx) + boxy * (-2 * ylo + boxy) + gzs;
        }
      } else {
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (2 * zlo + boxz);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (-2 * zlo + boxz);
        } else {
          if (ct(crs)) return this.ret(crs, true);
          crs += gzs;
        }
        crs += gys + boxx * (-2 * xlo + boxx);
      }
    } else {
      if (dj > 0) {
        ylo = dj * boxy - fy;
        crs = ylo * ylo;
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (2 * zlo + boxz);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (-2 * zlo + boxz);
        } else {
          if (ct(crs)) return this.ret(crs, true);
          crs += gzs;
        }
        crs += boxy * (2 * ylo + boxy);
      } else if (dj < 0) {
        ylo = (dj + 1) * boxy - fy;
        crs = ylo * ylo;
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (2 * zlo + boxz);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs += zlo * zlo; if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (-2 * zlo + boxz);
        } else {
          if (ct(crs)) return this.ret(crs, true);
          crs += gzs;
        }
        crs += boxy * (-2 * ylo + boxy);
      } else {
        if (dk > 0) {
          zlo = dk * boxz - fz;
          crs = zlo * zlo;
          if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (2 * zlo + boxz);
        } else if (dk < 0) {
          zlo = (dk + 1) * boxz - fz;
          crs = zlo * zlo;
          if (ct(crs)) return this.ret(crs, true);
          crs += boxz * (-2 * zlo + boxz);
        } else {
          throw new Error('voro++ internal error: min/max radius function called for central block');
        }
        crs += gys;
      }
      crs += gxs;
    }
    return this.ret(crs, false);
  }

  /** Stores the running minimum distance in this.crs and returns whether the block lies beyond mrs. */
  private ret(crs: number, beyond: boolean): boolean {
    this.crs = crs;
    return beyond;
  }

  private cornerTest(c: VoroCellNeighbor, xl: number, yl: number, zl: number, xh: number, yh: number, zh: number): boolean {
    if (c.planeIntersectsGuess(xh, yl, zl, xl * xh + yl * yl + zl * zl)) return false;
    if (c.planeIntersects(xh, yh, zl, xl * xh + yl * yh + zl * zl)) return false;
    if (c.planeIntersects(xl, yh, zl, xl * xl + yl * yh + zl * zl)) return false;
    if (c.planeIntersects(xl, yh, zh, xl * xl + yl * yh + zl * zh)) return false;
    if (c.planeIntersects(xl, yl, zh, xl * xl + yl * yl + zl * zh)) return false;
    if (c.planeIntersects(xh, yl, zh, xl * xh + yl * yl + zl * zh)) return false;
    return true;
  }

  private edgeXTest(c: VoroCellNeighbor, x0: number, yl: number, zl: number, x1: number, yh: number, zh: number): boolean {
    if (c.planeIntersectsGuess(x0, yl, zh, yl * yl + zl * zh)) return false;
    if (c.planeIntersects(x1, yl, zh, yl * yl + zl * zh)) return false;
    if (c.planeIntersects(x1, yl, zl, yl * yl + zl * zl)) return false;
    if (c.planeIntersects(x0, yl, zl, yl * yl + zl * zl)) return false;
    if (c.planeIntersects(x0, yh, zl, yl * yh + zl * zl)) return false;
    if (c.planeIntersects(x1, yh, zl, yl * yh + zl * zl)) return false;
    return true;
  }

  private edgeYTest(c: VoroCellNeighbor, xl: number, y0: number, zl: number, xh: number, y1: number, zh: number): boolean {
    if (c.planeIntersectsGuess(xl, y0, zh, xl * xl + zl * zh)) return false;
    if (c.planeIntersects(xl, y1, zh, xl * xl + zl * zh)) return false;
    if (c.planeIntersects(xl, y1, zl, xl * xl + zl * zl)) return false;
    if (c.planeIntersects(xl, y0, zl, xl * xl + zl * zl)) return false;
    if (c.planeIntersects(xh, y0, zl, xl * xh + zl * zl)) return false;
    if (c.planeIntersects(xh, y1, zl, xl * xh + zl * zl)) return false;
    return true;
  }

  private edgeZTest(c: VoroCellNeighbor, xl: number, yl: number, z0: number, xh: number, yh: number, z1: number): boolean {
    if (c.planeIntersectsGuess(xl, yh, z0, xl * xl + yl * yh)) return false;
    if (c.planeIntersects(xl, yh, z1, xl * xl + yl * yh)) return false;
    if (c.planeIntersects(xl, yl, z1, xl * xl + yl * yl)) return false;
    if (c.planeIntersects(xl, yl, z0, xl * xl + yl * yl)) return false;
    if (c.planeIntersects(xh, yl, z0, xl * xh + yl * yl)) return false;
    if (c.planeIntersects(xh, yl, z1, xl * xh + yl * yl)) return false;
    return true;
  }

  private faceXTest(c: VoroCellNeighbor, xl: number, y0: number, z0: number, y1: number, z1: number): boolean {
    if (c.planeIntersectsGuess(xl, y0, z0, xl * xl)) return false;
    if (c.planeIntersects(xl, y0, z1, xl * xl)) return false;
    if (c.planeIntersects(xl, y1, z1, xl * xl)) return false;
    if (c.planeIntersects(xl, y1, z0, xl * xl)) return false;
    return true;
  }

  private faceYTest(c: VoroCellNeighbor, x0: number, yl: number, z0: number, x1: number, z1: number): boolean {
    if (c.planeIntersectsGuess(x0, yl, z0, yl * yl)) return false;
    if (c.planeIntersects(x0, yl, z1, yl * yl)) return false;
    if (c.planeIntersects(x1, yl, z1, yl * yl)) return false;
    if (c.planeIntersects(x1, yl, z0, yl * yl)) return false;
    return true;
  }

  private faceZTest(c: VoroCellNeighbor, x0: number, y0: number, zl: number, x1: number, y1: number): boolean {
    if (c.planeIntersectsGuess(x0, y0, zl, zl * zl)) return false;
    if (c.planeIntersects(x0, y1, zl, zl * zl)) return false;
    if (c.planeIntersects(x1, y1, zl, zl * zl)) return false;
    if (c.planeIntersects(x1, y0, zl, zl * zl)) return false;
    return true;
  }
}
