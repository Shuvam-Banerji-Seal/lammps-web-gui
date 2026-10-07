import type { BoundaryStyle, SimBox, SimState } from './types';

/*
 * Box geometry: orthogonal and restricted triclinic boxes, periodic wrapping
 * with image flags, minimum image, and shrink-wrapping.
 *
 * docs.lammps.org/Howto_triclinic.html: the restricted triclinic box has
 * edge vectors "A = (xhi-xlo,0,0), B = (xy,yhi-ylo,0), C = (xz,yz,zhi-zlo)".
 * docs.lammps.org/boundary.html: "for y periodicity a particle which exits
 * the lower y boundary is displaced in the x-direction by xy before it
 * re-enters the upper y boundary" — so an atom's unwrapped position is
 * x + ix*A + iy*B + iz*C.
 * Shrink-wrapping: "For style s, the position of the face is set so as to
 * encompass the atoms in that dimension (shrink-wrapping), no matter how far
 * they move." "For style m, shrink-wrapping occurs, but is bounded by the
 * value specified in the data or restart file or set by the create_box
 * command." Native LAMMPS leaves a margin of 1e-4 of the box length the box
 * had when it was defined (measured: boxes 10 and 1000 wide give margins
 * 0.001 and 0.1, unchanged by later shrinking) — see SHRINK_PAD.
 */

const SHRINK_PAD = 1e-4;

export interface BoxInit {
  lo: [number, number, number];
  hi: [number, number, number];
  periodic?: [boolean, boolean, boolean];
  tilt?: [number, number, number];
  boundary?: SimBox['boundary'];
}

/** A complete SimBox from bounds; boundary defaults to periodic (p p p). */
export const makeBox = (b: BoxInit): SimBox => {
  const boundary = b.boundary
    ?? (b.periodic ?? [true, true, true]).map((p) => (p ? ['p', 'p'] : ['f', 'f'])) as SimBox['boundary'];
  const tilt: [number, number, number] = b.tilt ? [...b.tilt] : [0, 0, 0];
  return {
    lo: [...b.lo],
    hi: [...b.hi],
    periodic: [boundary[0][0] === 'p', boundary[1][0] === 'p', boundary[2][0] === 'p'],
    tilt,
    triclinic: !!b.tilt,
    boundary: boundary.map((f) => [f[0], f[1]]) as SimBox['boundary'],
    minLo: [...b.lo],
    minHi: [...b.hi],
  };
};

export const cloneBox = (b: SimBox): SimBox => ({
  lo: [...b.lo], hi: [...b.hi], periodic: [...b.periodic], tilt: [...b.tilt], triclinic: b.triclinic,
  boundary: b.boundary.map((f) => [f[0], f[1]]) as SimBox['boundary'], minLo: [...b.minLo], minHi: [...b.minHi],
});

/** Parses a boundary spec word: 'p', 'f', 's', 'm' or two of f/s/m. */
export const parseBoundary = (w: string): [BoundaryStyle, BoundaryStyle] | null => {
  const ok = (c: string): c is BoundaryStyle => c === 'p' || c === 'f' || c === 's' || c === 'm';
  if (w.length === 1 && ok(w)) return [w, w];
  if (w.length === 2 && ok(w[0]) && ok(w[1]) && w[0] !== 'p' && w[1] !== 'p') return [w[0], w[1]];
  return null;
};

/**
 * Cached geometry of a box: lengths, tilt, inverse h. Rebuild with `update`
 * whenever the box changes (shrink-wrap, deform, barostat, change_box).
 */
export class Geometry {
  lx = 0; ly = 0; lz = 0;
  xy = 0; xz = 0; yz = 0;
  lo: [number, number, number] = [0, 0, 0];
  hi: [number, number, number] = [0, 0, 0];
  periodic: [boolean, boolean, boolean] = [true, true, true];
  triclinic = false;
  /** h^-1 entries for x = lo + h * lambda, h = [[lx, xy, xz], [0, ly, yz], [0, 0, lz]]. */
  private i00 = 0; private i01 = 0; private i02 = 0; private i11 = 0; private i12 = 0; private i22 = 0;
  /** Per-dimension shrink-wrap margin (1e-4 of the defined length). */
  pad: [number, number, number] = [0, 0, 0];

  constructor(public box: SimBox) {
    this.update();
    this.pad = [0, 1, 2].map((d) => SHRINK_PAD * (box.hi[d] - box.lo[d])) as [number, number, number];
  }

  update(): void {
    const b = this.box;
    this.lo = [...b.lo];
    this.hi = [...b.hi];
    this.periodic = [...b.periodic];
    this.lx = b.hi[0] - b.lo[0];
    this.ly = b.hi[1] - b.lo[1];
    this.lz = b.hi[2] - b.lo[2];
    [this.xy, this.xz, this.yz] = b.tilt;
    this.triclinic = b.triclinic;
    // inverse of the upper-triangular h
    this.i00 = 1 / this.lx;
    this.i11 = 1 / this.ly;
    this.i22 = 1 / this.lz;
    this.i01 = -this.xy / (this.lx * this.ly);
    this.i12 = -this.yz / (this.ly * this.lz);
    this.i02 = (this.xy * this.yz - this.ly * this.xz) / (this.lx * this.ly * this.lz);
  }

  /** Volume (3d) or area (2d) — docs.lammps.org/thermo_style.html "vol = volume". */
  volume(dimension: 2 | 3): number {
    return dimension === 3 ? this.lx * this.ly * this.lz : this.lx * this.ly;
  }

  /** Fractional coordinates of x (out[0..2]). */
  toLamda(x: number, y: number, z: number, out: number[]): void {
    const dx = x - this.lo[0], dy = y - this.lo[1], dz = z - this.lo[2];
    out[0] = this.i00 * dx + this.i01 * dy + this.i02 * dz;
    out[1] = this.i11 * dy + this.i12 * dz;
    out[2] = this.i22 * dz;
  }

  fromLamda(l0: number, l1: number, l2: number, out: number[]): void {
    out[0] = this.lo[0] + this.lx * l0 + this.xy * l1 + this.xz * l2;
    out[1] = this.lo[1] + this.ly * l1 + this.yz * l2;
    out[2] = this.lo[2] + this.lz * l2;
  }

  /** Edge vector d (0 = A, 1 = B, 2 = C) times k, added to x[3i..]. */
  addEdge(x: Float64Array, i: number, d: number, k: number): void {
    if (d === 0) { x[3 * i] += k * this.lx; return; }
    if (d === 1) { x[3 * i] += k * this.xy; x[3 * i + 1] += k * this.ly; return; }
    x[3 * i] += k * this.xz; x[3 * i + 1] += k * this.yz; x[3 * i + 2] += k * this.lz;
  }

  /**
   * Wraps atom i back into the periodic box, updating its image flags.
   * Dimensions are handled z, y, x so a tilted shift is applied before the
   * coordinates it moves are tested.
   */
  remap(x: Float64Array, image: Int32Array, i: number): void {
    for (let d = 2; d >= 0; d--) {
      if (!this.periodic[d]) continue;
      // fractional coordinate along d (only components d.. matter: h is upper triangular)
      const dx = x[3 * i] - this.lo[0], dy = x[3 * i + 1] - this.lo[1], dz = x[3 * i + 2] - this.lo[2];
      const lam = d === 2 ? this.i22 * dz
        : d === 1 ? this.i11 * dy + this.i12 * dz
          : this.i00 * dx + this.i01 * dy + this.i02 * dz;
      if (lam >= 0 && lam < 1) continue;
      const k = Math.floor(lam);
      this.addEdge(x, i, d, -k);
      image[3 * i + d] += k;
      // floating-point edge: a coordinate that lands exactly on hi goes to lo
      const lam2 = d === 2 ? this.i22 * (x[3 * i + 2] - this.lo[2])
        : d === 1 ? this.i11 * (x[3 * i + 1] - this.lo[1]) + this.i12 * (x[3 * i + 2] - this.lo[2])
          : this.i00 * (x[3 * i] - this.lo[0]) + this.i01 * (x[3 * i + 1] - this.lo[1]) + this.i02 * (x[3 * i + 2] - this.lo[2]);
      if (lam2 >= 1) { this.addEdge(x, i, d, -1); image[3 * i + d] += 1; }
      else if (lam2 < 0) { this.addEdge(x, i, d, 1); image[3 * i + d] -= 1; }
    }
  }

  /** Unwrapped position of atom i: x + ix*A + iy*B + iz*C. */
  unwrap(x: Float64Array, image: Int32Array, i: number, out: number[]): void {
    const ix = image[3 * i], iy = image[3 * i + 1], iz = image[3 * i + 2];
    out[0] = x[3 * i] + ix * this.lx + iy * this.xy + iz * this.xz;
    out[1] = x[3 * i + 1] + iy * this.ly + iz * this.yz;
    out[2] = x[3 * i + 2] + iz * this.lz;
  }

  /**
   * Minimum-image convention on a displacement d (in place), periodic
   * dimensions only; for a triclinic box, z then y then x, each shift using
   * the full edge vector.
   */
  minimumImage(d: number[] | Float64Array): void {
    if (this.periodic[2]) {
      if (Math.abs(d[2]) > 0.5 * this.lz) {
        const k = Math.round(d[2] / this.lz);
        d[2] -= k * this.lz; d[1] -= k * this.yz; d[0] -= k * this.xz;
      }
    }
    if (this.periodic[1]) {
      if (Math.abs(d[1]) > 0.5 * this.ly) {
        const k = Math.round(d[1] / this.ly);
        d[1] -= k * this.ly; d[0] -= k * this.xy;
      }
    }
    if (this.periodic[0]) {
      if (Math.abs(d[0]) > 0.5 * this.lx) {
        const k = Math.round(d[0] / this.lx);
        d[0] -= k * this.lx;
      }
    }
  }

  /** True if (x, y, z) is inside the box (half-open on hi). */
  inside(x: number, y: number, z: number): boolean {
    const l = [0, 0, 0];
    this.toLamda(x, y, z, l);
    return l[0] >= 0 && l[0] < 1 && l[1] >= 0 && l[1] < 1 && l[2] >= 0 && l[2] < 1;
  }
}

/**
 * Re-fits shrink-wrapped faces around the atoms (Developer_flow: "The box
 * boundaries are then reset (if needed) via the reset_box() method ... e.g. if
 * box boundaries are shrink-wrapped to current particle coordinates").
 * Returns true if the box changed. Orthogonal boxes; for a triclinic box the
 * extents are measured in fractional coordinates of the tilted faces.
 */
export const shrinkWrap = (s: SimState, g: Geometry): boolean => {
  const b = s.box;
  let changed = false;
  const lam = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    const [flo, fhi] = b.boundary[d];
    const slo = flo === 's' || flo === 'm';
    const shi = fhi === 's' || fhi === 'm';
    if (!slo && !shi) continue;
    if (s.dimension === 2 && d === 2) continue;
    if (s.n === 0) continue;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < s.n; i++) {
      let c: number;
      if (g.triclinic) {
        g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
        c = lam[d];
      } else c = s.x[3 * i + d];
      if (c < mn) mn = c;
      if (c > mx) mx = c;
    }
    if (g.triclinic) {
      // convert the fractional extents back to box coordinates along d
      const len = d === 0 ? g.lx : d === 1 ? g.ly : g.lz;
      mn = b.lo[d] + mn * len;
      mx = b.lo[d] + mx * len;
    }
    if (slo) {
      let v = mn - g.pad[d];
      if (flo === 'm') v = Math.min(v, b.minLo[d]);
      if (v !== b.lo[d]) { b.lo[d] = v; changed = true; }
    }
    if (shi) {
      let v = mx + g.pad[d];
      if (fhi === 'm') v = Math.max(v, b.minHi[d]);
      if (v !== b.hi[d]) { b.hi[d] = v; changed = true; }
    }
  }
  if (changed) g.update();
  return changed;
};
