/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/fix_graphics_isosurface.cpp and fix_graphics_isosurface.h at commit b332f28 (class
 * FixGraphicsIsosurface: "fix ID group-ID graphics/isosurface Nevery isovalue radius keyword args ..."), with
 * MathExtra::cross3 and norm3 (src/math_extra.h) and utils::star_subst (src/utils.cpp) that it calls. Copyright (2003)
 * Sandia Corporation, distributed under the GNU General Public License (the LAMMPS file headers); the GRAPHICS source
 * headers name no contributing author. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The per-atom property of the group atoms (1.0, the mass, or a compute, fix or atom-style variable value) is spread on
 * a grid over the box with a truncated gaussian; marching cubes on that grid at the isovalue gives the triangles of the
 * isosurface, with per-vertex normals from the grid gradient and the atom type nearest to each vertex. They are drawn
 * by dump image through its "fix" keyword (triangles with normals, as fix graphics/chunk) and written, if asked, as STL
 * files. The grid sums run over the owned and ghost atoms in native order (../neighbor/nativeGhostFrame.ts), so the
 * triangles are native's.
 *
 * Difference: the per-atom property is collected in the post-force stage of the steps whose end of step builds the
 * surface (every Nevery steps, and at setup), where native collects it at every step; only those values are used.
 */
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import type { ImageObject } from '../../engine/output/image_objects';
import { parseRef, peratomValues, type Ref } from '../../engine/refs';
import { formatNumber } from '../../engine/cformat';
import { toBase64 } from '../../engine/output/dump_image';
import { exp } from '../libm/exp';
import { cbrt } from '../libm/cbrt';
import { NativeGhostFrame } from '../neighbor/nativeGhostFrame';
import { EDGETABLE, TRITABLE } from './isosurface_tables';
import type { TriNormObject, Vec3 } from './objects';
import { drawGraphicsObject } from './draw';

/** fix_graphics_isosurface.cpp: "constexpr double BIG = 1.0e200;" */
const BIG = 1.0e200;
/** GRIDPOINTS[quality]: grid points along the cube root of the box volume (min, low, med, high, max). */
const GRIDPOINTS = [0, 8.0, 16.0, 64.0, 128.0, 256.0];
const SURFMIN = 1, SURFLOW = 2, SURFMED = 3, SURFHIGH = 4, SURFMAX = 5;
/** "for truncating gaussian spreading": CUTVAL = -log(0.0001) */
const CUTVAL = 9.210340371976182;
/** "extra grid points at the sides of the subbox grid" */
const GRIDEXTRA = 4;

/** The edge of each EDGETABLE bit: its two cell corners (get_vertex calls of end_of_step). */
const EDGE_CORNERS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];

type Property = { kind: 'none' } | { kind: 'mass' } | { kind: 'ref'; ref: Ref };

/** platform::has_compress_extension: the suffixes of LAMMPS's compressed output. */
const COMPRESSED = ['.gz', '.bz2', '.zst', '.xz', '.lzma', '.lz4'];

/** C's (int) conversion: truncation toward zero. */
const toInt = (v: number): number => Math.trunc(v);

export class FixGraphicsIsosurface extends Fix {
  readonly style = 'graphics/isosurface';
  private readonly iso: number;
  private readonly rad: number;
  private quality = SURFLOW;
  private property: Property = { kind: 'none' };
  private filename = '';
  private binary = false;
  private pad = 0;
  private readonly frame = new NativeGhostFrame();
  /** the property of the owned atoms (engine order), collected for the next end of step */
  private pdata: Float64Array | null = null;
  private triangles: TriNormObject[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal fix graphics/isosurface command: missing argument(s)');
    const s = sys.state;
    if (s.box.triclinic) throw new StyleError('Fix graphics/isosurface is currently not compatible with triclinic cells');
    if (s.dimension === 2) throw new StyleError('Fix graphics/isosurface is currently not compatible with 2d systems');

    const intArg = (w: string): number => {
      if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
      return Number(w);
    };
    const numArg = (w: string): number => {
      const v = Number(w);
      if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
      return v;
    };

    this.nevery = intArg(args[0]);
    if (this.nevery <= 0) throw new StyleError(`Illegal fix graphics/isosurface nevery value ${this.nevery}`);
    this.iso = numArg(args[1]);
    this.rad = 2.0 * numArg(args[2]);
    if (this.rad <= 0.0) throw new StyleError(`Illegal fix graphics/isosurface radius value ${this.rad}`);

    let iarg = 3;
    while (iarg < args.length) {
      const kw = args[iarg];
      const value = (): string => {
        if (iarg + 2 > args.length) throw new StyleError(`Illegal fix graphics/isosurface ${kw} command: missing argument(s)`);
        return args[iarg + 1];
      };
      if (kw === 'quality') {
        const v = value();
        const q = ['min', 'low', 'med', 'high', 'max'].indexOf(v);
        if (q < 0) throw new StyleError(`Unknown fix graphics/isosurface quality setting ${v}`);
        this.quality = [SURFMIN, SURFLOW, SURFMED, SURFHIGH, SURFMAX][q];
      } else if (kw === 'property') {
        const v = value();
        if (v === 'none') this.property = { kind: 'none' };
        else if (v === 'mass') this.property = { kind: 'mass' };
        else this.property = { kind: 'ref', ref: this.checkRef(v) };
      } else if (kw === 'filename') {
        const v = value();
        if (!v.includes('*')) throw new StyleError("Output to STL file requires file name with '*'");
        this.filename = v;
      } else if (kw === 'binary') {
        const v = value();
        if (['yes', 'on', 'true', '1'].includes(v)) this.binary = true;
        else if (['no', 'off', 'false', '0'].includes(v)) this.binary = false;
        else throw new StyleError(`Expected boolean parameter instead of '${v}' in input script or data file`);
      } else if (kw === 'pad') {
        this.pad = intArg(value());
      } else {
        throw new StyleError(`Unknown fix graphics/isosurface keyword ${kw}`);
      }
      iarg += 2;
    }
    if (this.filename && COMPRESSED.some((x) => this.filename.endsWith(x))) {
      throw new StyleError(`fix graphics/isosurface: compressed STL output (${this.filename}) is not supported by the browser engine`);
    }
  }

  /** The property argument's checks (the constructor's ArgInfo branch). */
  private checkRef(w: string): Ref {
    if (/\]\[/.test(w)) throw new StyleError(`Invalid fix graphics/isosurface property ${w}`);
    let ref: Ref;
    try {
      ref = parseRef(w);
    } catch {
      throw new StyleError(`Invalid fix graphics/isosurface property ${w}`);
    }
    const sys = this.sys;
    const pindex = ref.index ?? 0;
    if (ref.kind === 'c') {
      const c = sys.computes.find((x) => x.id === ref.id);
      if (!c) throw new StyleError(`Compute ID ${ref.id} for fix graphics/isosurface does not exist`);
      if (!c.peratomFlag) throw new StyleError(`Fix graphics/isosurface compute ${ref.id} does not calculate per-atom values`);
      if (pindex === 0 && c.sizePeratomCols !== 0) throw new StyleError(`Fix graphics/isosurface compute ${ref.id} does not calculate a per-atom vector`);
      if (pindex && c.sizePeratomCols === 0) throw new StyleError(`Fix graphics/isosurface compute ${ref.id} does not calculate a per-atom array`);
      if (pindex && pindex > c.sizePeratomCols) throw new StyleError(`Fix graphics/isosurface compute ${ref.id} array is accessed out-of-range`);
    } else if (ref.kind === 'f') {
      const f = sys.fixes.find((x) => x.id === ref.id);
      if (!f) throw new StyleError(`Fix ID ${ref.id} for fix graphics/isosurface does not exist`);
      if (!f.peratomFlag) throw new StyleError(`Fix graphics/isosurface fix ${ref.id} does not calculate per-atom values`);
      if (pindex === 0 && f.sizePeratomCols !== 0) throw new StyleError(`Fix graphics/isosurface fix ${ref.id} does not calculate a per-atom vector`);
      if (pindex && f.sizePeratomCols === 0) throw new StyleError(`Fix graphics/isosurface fix ${ref.id} does not calculate a per-atom array`);
      if (pindex && pindex > f.sizePeratomCols) throw new StyleError(`Fix graphics/isosurface fix ${ref.id} array is accessed out-of-range`);
    } else if (ref.kind === 'v') {
      if (ref.index !== null) throw new StyleError(`Invalid fix graphics/isosurface property ${w}`);
      const v = sys.vars.get(ref.id);
      if (!v) throw new StyleError(`Variable name ${ref.id} for fix graphics/isosurface does not exist`);
      if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Fix graphics/isosurface variable ${ref.id} is not atom-style variable`);
    } else {
      throw new StyleError(`Invalid fix graphics/isosurface property ${w}`);
    }
    return ref;
  }

  /** FixGraphicsIsosurface::setup: post_force, then end_of_step. */
  override setup(): void {
    this.collect();
    this.build();
  }

  override postForce(): void {
    if (this.sys.state.step % this.nevery === 0) this.collect();
  }

  override endOfStep(): void {
    this.build();
  }

  /** FixGraphicsIsosurface::post_force: the property of every owned atom. */
  private collect(): void {
    const sys = this.sys;
    const s = sys.state;
    const p = new Float64Array(s.n);
    const prop = this.property;
    if (prop.kind === 'none') {
      p.fill(1.0);
    } else if (prop.kind === 'mass') {
      for (let i = 0; i < s.n; i++) p[i] = s.rmass ? s.rmass[i] : s.massByType[s.type[i]];
    } else {
      p.set(peratomValues(sys, prop.ref).subarray(0, s.n));
    }
    this.pdata = p;
  }

  /** FixGraphicsIsosurface::end_of_step: the grid, the marching cubes and the STL file. */
  private build(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    if (!this.pdata || this.pdata.length !== s.n) this.collect();
    const fr = this.frame;
    fr.update(s, sys.nb, g);
    const nall = fr.nall;
    const pd = new Float64Array(nall);
    fr.spread(this.pdata!, pd);

    // determine grid dimensions
    const sublo = g.lo, subhi = g.hi;

    // get grid spacing and sub-domain lengths
    const delta = cbrt(g.lx * g.ly * g.lz) / GRIDPOINTS[this.quality];
    const sublen = [subhi[0] - sublo[0], subhi[1] - sublo[1], subhi[2] - sublo[2]];

    // get grid dims for subdomains, add extra points on each side
    const nx = 2 * GRIDEXTRA + toInt(Math.ceil(sublen[0] / delta));
    const ny = 2 * GRIDEXTRA + toInt(Math.ceil(sublen[1] / delta));
    const nz = 2 * GRIDEXTRA + toInt(Math.ceil(sublen[2] / delta));

    // determine cutoff for spreading and number of gridpoints for spreading
    const rcutsq = this.rad * CUTVAL;
    const nrange = toInt(Math.ceil(Math.sqrt(rcutsq) / delta));

    const ncell = nx * ny * nz;
    const at = (ix: number, iy: number, iz: number): number => (ix * ny + iy) * nz + iz;
    const isogrid = new Float64Array(ncell);
    const distgrid = new Float64Array(ncell).fill(BIG);
    const typegrid = new Int32Array(ncell);

    // loop over local and ghost atoms, match group and check if within grid range
    const X = fr.X;
    const sigma = this.rad;
    for (let i = 0; i < nall; ++i) {
      const o = fr.owner[i];
      if (!(s.mask[o] & this.groupBit)) continue;
      // distribute(): spread out atom data across the grid
      const px = X[3 * i], py = X[3 * i + 1], pz = X[3 * i + 2];
      const type = s.type[o];
      const val = pd[i];
      const ix = GRIDEXTRA + toInt(Math.floor((px - sublo[0]) / delta));
      const iy = GRIDEXTRA + toInt(Math.floor((py - sublo[1]) / delta));
      const iz = GRIDEXTRA + toInt(Math.floor((pz - sublo[2]) / delta));
      const dx = px - (sublo[0] + (ix - GRIDEXTRA) * delta);
      const dy = py - (sublo[1] + (iy - GRIDEXTRA) * delta);
      const dz = pz - (sublo[2] + (iz - GRIDEXTRA) * delta);
      for (let jx = -nrange; jx <= nrange; ++jx) {
        const kx = ix + jx;
        if (kx < 0 || kx >= nx) continue;
        for (let jy = -nrange; jy <= nrange; ++jy) {
          const ky = iy + jy;
          if (ky < 0 || ky >= ny) continue;
          for (let jz = -nrange; jz <= nrange; ++jz) {
            const kz = iz + jz;
            if (kz < 0 || kz >= nz) continue;
            const xpos = jx * delta - dx;
            const ypos = jy * delta - dy;
            const zpos = jz * delta - dz;
            const distsq = xpos * xpos + ypos * ypos + zpos * zpos;
            if (distsq < rcutsq) {
              const c = at(kx, ky, kz);
              isogrid[c] += val * exp(-distsq / sigma);
              if (distsq < distgrid[c]) {
                distgrid[c] = distsq;
                typegrid[c] = type;
              }
            }
          }
        }
      }
    }

    // subtract the isovalue from grid data so that the isosurface would be drawn for a value of < 0.0
    for (let c = 0; c < ncell; c++) isogrid[c] -= this.iso;

    // the gradient at a grid point by central differences, clamped at the boundaries (compute_gradient)
    const gradient = (cx: number, cy: number, cz: number, out: Float64Array, o: number): void => {
      const xm = cx > 0 ? cx - 1 : 0;
      const xp = cx < nx - 1 ? cx + 1 : nx - 1;
      const ym = cy > 0 ? cy - 1 : 0;
      const yp = cy < ny - 1 ? cy + 1 : ny - 1;
      const zm = cz > 0 ? cz - 1 : 0;
      const zp = cz < nz - 1 ? cz + 1 : nz - 1;
      out[o] = isogrid[at(xp, cy, cz)] - isogrid[at(xm, cy, cz)];
      out[o + 1] = isogrid[at(cx, yp, cz)] - isogrid[at(cx, ym, cz)];
      out[o + 2] = isogrid[at(cx, cy, zp)] - isogrid[at(cx, cy, zm)];
    };

    // the grid cell: corner positions, values, gradients and types (gridcell)
    const gpos = new Float64Array(24), giso = new Float64Array(8), ggrad = new Float64Array(24), gtype = new Int32Array(8);
    // the 12 possible vertices of a cell, their normals and types
    const vert = new Float64Array(36), vnorm = new Float64Array(36), vtype = new Int32Array(12);
    // corner c is cell point (ix + CX[c], iy + CY[c], iz + CZ[c])
    const CX = [0, 1, 1, 0, 0, 1, 1, 0], CY = [0, 0, 1, 1, 0, 0, 1, 1], CZ = [0, 0, 0, 0, 1, 1, 1, 1];

    const tris: TriNormObject[] = [];
    for (let ix = 0; ix < nx - 1; ++ix) {
      for (let iy = 0; iy < ny - 1; ++iy) {
        for (let iz = 0; iz < nz - 1; ++iz) {
          // get lower edge coordinates of the grid cell
          const gx = sublo[0] + (ix - GRIDEXTRA) * delta;
          const gy = sublo[1] + (iy - GRIDEXTRA) * delta;
          const gz = sublo[2] + (iz - GRIDEXTRA) * delta;
          for (let c = 0; c < 8; c++) {
            const p = at(ix + CX[c], iy + CY[c], iz + CZ[c]);
            giso[c] = isogrid[p];
            gtype[c] = typegrid[p];
            gpos[3 * c] = CX[c] ? gx + delta : gx;
            gpos[3 * c + 1] = CY[c] ? gy + delta : gy;
            gpos[3 * c + 2] = CZ[c] ? gz + delta : gz;
          }

          // determine edge table index
          let idx = 0;
          for (let c = 0; c < 8; c++) if (giso[c] < 0.0) idx |= 1 << c;

          // gridcube is not crossed by isosurface
          const edges = EDGETABLE[idx];
          if (edges === 0) continue;

          // compute gradient at each corner for smooth normals
          for (let c = 0; c < 8; c++) gradient(ix + CX[c], iy + CY[c], iz + CZ[c], ggrad, 3 * c);

          // compute the possible 12 triangle vertices and normals (get_vertex)
          for (let e = 0; e < 12; e++) {
            if (!(edges & (1 << e))) continue;
            const c1 = EDGE_CORNERS[e][0], c2 = EDGE_CORNERS[e][1];
            const diff = giso[c2] - giso[c1];
            const fraction = Math.abs(diff) > 0.0 ? -giso[c1] / diff : 0.0;
            // assign type closest to vertex, if available
            const t1 = gtype[c1], t2 = gtype[c2];
            vtype[e] = t1 && !t2 ? t1 : !t1 && t2 ? t2 : t1 && t2 ? (fraction < 0.5 ? t1 : t2) : 0;
            for (let d = 0; d < 3; ++d) {
              vert[3 * e + d] = gpos[3 * c1 + d] + fraction * (gpos[3 * c2 + d] - gpos[3 * c1 + d]);
              vnorm[3 * e + d] = ggrad[3 * c1 + d] + fraction * (ggrad[3 * c2 + d] - ggrad[3 * c1 + d]);
            }
            const n0 = vnorm[3 * e], n1 = vnorm[3 * e + 1], n2 = vnorm[3 * e + 2];
            const len = Math.sqrt(n0 * n0 + n1 * n1 + n2 * n2);
            if (len > 0.0) {
              vnorm[3 * e] = n0 / len;
              vnorm[3 * e + 1] = n1 / len;
              vnorm[3 * e + 2] = n2 / len;
            }
          }

          // compute the triangles for this grid cell and add them to the list
          for (let i = 0; TRITABLE[16 * idx + i] !== -1; i += 3) {
            const e0 = TRITABLE[16 * idx + i], e1 = TRITABLE[16 * idx + i + 1], e2 = TRITABLE[16 * idx + i + 2];
            const v = (e: number): Vec3 => [vert[3 * e], vert[3 * e + 1], vert[3 * e + 2]];
            const nrm = (e: number): Vec3 => [vnorm[3 * e], vnorm[3 * e + 1], vnorm[3 * e + 2]];
            tris.push({ kind: 'trinorm', types: [vtype[e0], vtype[e1], vtype[e2]], vertices: [v(e0), v(e1), v(e2)], normals: [nrm(e0), nrm(e1), nrm(e2)] });
          }
        }
      }
    }

    // skip triangles with any part outside the subdomain
    const kept: TriNormObject[] = [];
    for (const tri of tris) {
      let addme = true;
      for (let i = 0; i < 3; ++i) {
        for (let j = 0; j < 3; ++j) if (tri.vertices[i][j] < sublo[j] || tri.vertices[i][j] > subhi[j]) addme = false;
      }
      if (addme) kept.push(tri);
    }
    this.triangles = kept;

    if (this.filename) this.writeStl(kept);
  }

  /** The STL file of this step (end_of_step, rank 0 output with one process). */
  private writeStl(tris: TriNormObject[]): void {
    const sys = this.sys;
    const step = sys.state.step;
    // convert to single precision data and compute the facet normals (MathExtra::cross3 and norm3)
    const data = new Float32Array(12 * tris.length);
    tris.forEach((tri, n) => {
      const [a, b, c] = tri.vertices;
      const d1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const d2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const nrm = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]];
      const val = nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2];
      if (val > 0.0) {
        const scale = 1.0 / Math.sqrt(val);
        nrm[0] *= scale; nrm[1] *= scale; nrm[2] *= scale;
      }
      data.set([nrm[0], nrm[1], nrm[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]], 12 * n);
    });

    // utils::star_subst: the step padded with zeroes to pad digits
    const star = this.filename.indexOf('*');
    const stepText = String(Math.abs(step)).padStart(Math.max(0, this.pad - (step < 0 ? 1 : 0)), '0');
    const name = this.filename.slice(0, star) + (step < 0 ? '-' : '') + stepText + this.filename.slice(star + 1);

    // the title is resized to 80 characters (NUL-padded); "%s" prints it up to the first NUL
    let title = `STL isosurface from fix ${this.id} graphics/isosurface on step ${step}`;
    if (title.length > 80) title = title.slice(0, 80);

    if (this.binary) {
      const head = new Uint8Array(84 + 50 * tris.length);
      for (let k = 0; k < title.length; k++) head[k] = title.charCodeAt(k) & 0xff;
      const dv = new DataView(head.buffer);
      dv.setUint32(80, tris.length, true);
      for (let n = 0; n < tris.length; n++) {
        for (let k = 0; k < 12; k++) dv.setFloat32(84 + 50 * n + 4 * k, data[12 * n + k], true);
        dv.setUint16(84 + 50 * n + 48, 0, true);
      }
      // binary session files are data URLs, as dump image writes its PNG, JPEG and PPM files
      sys.writeFile(name, 'data:model/stl;base64,' + toBase64(head), false);
      return;
    }
    const e = (v: number): string => formatNumber(v, '%e');
    const out: string[] = [`solid ${title}`];
    for (let n = 0; n < tris.length; n++) {
      const q = 12 * n;
      out.push(`  facet normal ${e(data[q])} ${e(data[q + 1])} ${e(data[q + 2])}`);
      out.push('    outer loop');
      out.push(`      vertex ${e(data[q + 3])} ${e(data[q + 4])} ${e(data[q + 5])}`);
      out.push(`      vertex ${e(data[q + 6])} ${e(data[q + 7])} ${e(data[q + 8])}`);
      out.push(`      vertex ${e(data[q + 9])} ${e(data[q + 10])} ${e(data[q + 11])}`);
      out.push('    endloop');
      out.push('  endfacet');
    }
    out.push(`endsolid ${title}`);
    sys.writeFile(name, out.join('\n') + '\n', false);
  }

  /** FixGraphicsIsosurface::image: the triangles of the last end of step, drawn by dump image. */
  imageObjects(): ImageObject[] {
    return this.triangles.map((o): ImageObject => ({ kind: 'draw', draw: (f, st) => drawGraphicsObject(f, st, o) }));
  }
}

export const makeFixGraphicsIsosurface: FixFactory = (sys, id, group, args) => new FixGraphicsIsosurface(sys, id, group, args);
