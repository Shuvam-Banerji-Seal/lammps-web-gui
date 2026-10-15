/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/image_objects.cpp at b332f28 (ConeObj: the constructor that builds the triangle mesh of a cone
 * or cylinder from its length, top and bottom widths, face flags and resolution; the file-local transform and
 * transform_normals that orient the mesh along an axis; ConeObj::draw), src/GRAPHICS/image_objects.h (RESOLUTION = 36,
 * the CONE_TOP, CONE_BOT and CONE_SIDE flags of src/GRAPHICS/graphics.h) and the cylinder and cone branch of the region
 * rendering in src/GRAPHICS/dump_image.cpp (DumpImage::create_image, "render regions, if supported"). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Scope: regions that do not move or rotate and have no open faces (the core refuses the rest). The extra uncapped
 * cylinder that native draws for filled cylinders has diameter 2 * radius, and radius stays 0 in that branch, so it
 * draws nothing and is not ported.
 */
import type { RasterFrame, Rgb255, RegionConeSpec } from '../../engine/output/image_objects';
import { drawCylinder, drawTrinorm } from './draw';

type V = [number, number, number];
type Tri = [V, V, V];

/** Graphics::RESOLUTION of image_objects.h: the default resolution for cylindrical objects. */
const RESOLUTION = 36;
/** RADOVERLAP of image_objects.cpp: a small angular overlap between neighbouring cap wedges. */
const RADOVERLAP = 0.01;
const SMALL = 1.0e-10;
/** Graphics::CONE_TOP, CONE_BOT and CONE_SIDE (graphics.h). */
const CONE_TOP = 1, CONE_BOT = 2, CONE_SIDE = 4;

const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vlen = (a: V): number => Math.hypot(a[0], a[1], a[2]);
/** vec3norm of image_objects.cpp: a zero vector stays zero. */
const vnorm = (a: V): V => {
  const l = vlen(a);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
};

/** The mesh of ConeObj: triangles and their per-vertex normals (the template along x, before any orientation). */
export interface ConeMesh { tris: Tri[]; norms: Tri[] }

/**
 * ConeObj::ConeObj: a cone (or cylinder) of the given length along x centred on the origin, with top width (radius at
 * +length/2) and bottom width (radius at -length/2). flag selects the caps and the side; resolution is the number of
 * segments around the axis (fewer than 2 gives an empty mesh). Side triangles carry radial normals (smooth shading).
 */
export function coneMesh(length: number, topwidth: number, botwidth: number, flag: number, resolution: number): ConeMesh {
  const tris: Tri[] = [], norms: Tri[] = [];
  if (resolution < 2) return { tris, norms };
  const dotop = (flag & CONE_TOP) > 0, dobot = (flag & CONE_BOT) > 0, doside = (flag & CONE_SIDE) > 0;
  const top: V = [0.5 * length, 0, 0], bot: V = [-0.5 * length, 0, 0];
  const radinc = (2 * Math.PI) / resolution;
  const p1top: V = [...top], p2top: V = [...top], p1bot: V = [...bot], p2bot: V = [...bot];
  const ntop: V = [1, 0, 0], nbot: V = [-1, 0, 0];
  for (let i = 0; i < resolution; i++) {
    if (topwidth > 0) {
      p1top[1] = topwidth * Math.sin(radinc * i - RADOVERLAP);
      p1top[2] = topwidth * Math.cos(radinc * i - RADOVERLAP);
      p2top[1] = topwidth * Math.sin(radinc * (i + 1));
      p2top[2] = topwidth * Math.cos(radinc * (i + 1));
      if (dotop) {
        tris.push([[...p1top], [...top], [...p2top]]);
        norms.push([ntop, ntop, ntop]);
      }
    }
    if (botwidth > 0) {
      p1bot[1] = botwidth * Math.sin(radinc * i - RADOVERLAP);
      p1bot[2] = botwidth * Math.cos(radinc * i - RADOVERLAP);
      p2bot[1] = botwidth * Math.sin(radinc * (i + 1));
      p2bot[2] = botwidth * Math.cos(radinc * (i + 1));
      if (dobot) {
        tris.push([[...p1bot], [...bot], [...p2bot]]);
        norms.push([nbot, nbot, nbot]);
      }
    }
    if (doside) {
      const n1top = vnorm([0, p1top[1], p1top[2]]), n2top = vnorm([0, p2top[1], p2top[2]]);
      const n1bot = vnorm([0, p1bot[1], p1bot[2]]), n2bot = vnorm([0, p2bot[1], p2bot[2]]);
      if (topwidth > 0) {
        tris.push([[...p1top], [...p1bot], [...p2top]]);
        norms.push([n1top, n1bot, n2top]);
      }
      if (botwidth > 0) {
        tris.push([[...p1bot], [...p2bot], [...p2top]]);
        norms.push([n1bot, n2bot, n2top]);
      }
    }
  }
  return { tris, norms };
}

/** The orthonormal basis of image_objects.cpp transform and transform_normals around the unit direction (null: too short). */
const basisOf = (dir: V): { u: V; v: V; w: V } | null => {
  const u = vnorm(dir);
  if (vlen(u) < SMALL) return null;
  const a: V = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const v = vnorm(cross(u, a));
  const w = cross(u, v);
  return { u, v, w };
};

const comb = (t: V, u: V, v: V, w: V, s: number): V => [
  t[0] * s * u[0] + t[1] * s * v[0] + t[2] * s * w[0],
  t[0] * s * u[1] + t[1] * s * v[1] + t[2] * s * w[1],
  t[0] * s * u[2] + t[1] * s * v[2] + t[2] * s * w[2],
];

/** transform (len = width = 1): the mesh turned so that +x points along dir and moved to offs. */
export function orientMesh(tris: Tri[], dir: V, offs: V): Tri[] {
  const b = basisOf(dir);
  if (!b) return [];
  const { u, v, w } = b;
  const at = (p: V): V => {
    const q = comb(p, u, v, w, 1);
    return [q[0] + offs[0], q[1] + offs[1], q[2] + offs[2]];
  };
  return tris.map(([p1, p2, p3]) => [at(p1), at(p2), at(p3)] as Tri);
}

/** transform_normals: the normals turned along dir (rotation only), each renormalised. */
export function orientNormals(norms: Tri[], dir: V): Tri[] {
  const b = basisOf(dir);
  if (!b) return [];
  const { u, v, w } = b;
  return norms.map(([n1, n2, n3]) => [vnorm(comb(n1, u, v, w, 1)), vnorm(comb(n2, u, v, w, 1)), vnorm(comb(n3, u, v, w, 1))] as Tri);
}

/**
 * ConeObj::draw: the mesh oriented along dir and centred on mid, drawn in colour col. flag 1 draws the triangles
 * (draw_trinorm with the per-vertex normals), flag 2 the wireframe: a cylinder of `diameter` for every triangle's
 * first edge and, for every other triangle, its second edge (the top rim, native's n & 1 test).
 */
export function drawConeMesh(f: RasterFrame, mesh: ConeMesh, flag: number, dir: V, mid: V, col: V, diameter: number, opacity: number): void {
  if (mesh.tris.length === 0) return;
  const cone = orientMesh(mesh.tris, dir, mid);
  const norms = orientNormals(mesh.norms, dir);
  if (cone.length === 0) return;
  let n = 0;
  for (let k = 0; k < cone.length; k++) {
    const [p0, p1, p2] = cone[k];
    const [n0, n1, n2] = norms[k];
    if (flag & 1) drawTrinorm(f, p0, p1, p2, n0, n1, n2, col, col, col, opacity);
    if (flag & 2) {
      drawCylinder(f, p0, p1, col, diameter, 3, opacity);
      n++;
      if (n & 1) drawCylinder(f, p0, p2, col, diameter, 3, opacity);
    }
  }
}

const unit = (c: Rgb255): V => [c[0] / 255, c[1] / 255, c[2] / 255];

/**
 * The cylinder and cone branch of DumpImage::create_image: the axis unit vector, the box-clipped lo and hi centres
 * (the region's c1 and c2 are the two coordinates other than the axis), the cap and side flags (frame draws the
 * side only, 2 * RESOLUTION segments for filled and transparent, RESOLUTION for frame), and the drawing style.
 */
export function drawConeRegion(f: RasterFrame, spec: RegionConeSpec, boxlo: V, boxhi: V): void {
  const ax = spec.axis;
  const [d1, d2] = ax === 0 ? [1, 2] : ax === 1 ? [0, 2] : [0, 1];
  const xlo = Math.max(spec.lo, boxlo[ax]), xhi = Math.min(spec.hi, boxhi[ax]);
  const length = xhi - xlo;
  const lo: V = [0, 0, 0], hi: V = [0, 0, 0];
  lo[d1] = spec.c1; lo[d2] = spec.c2; lo[ax] = xlo;
  hi[d1] = spec.c1; hi[d2] = spec.c2; hi[ax] = xhi;
  const dir: V = [0, 0, 0];
  dir[ax] = 1;
  const mid: V = [0.5 * (hi[0] + lo[0]), 0.5 * (hi[1] + lo[1]), 0.5 * (hi[2] + lo[2])];
  const frame = spec.drawstyle === 'frame';
  const faceflag = frame ? 4 : CONE_TOP | CONE_BOT | CONE_SIDE;
  const drawflag = frame ? 2 : 1;
  const opacity = spec.drawstyle === 'transparent' ? spec.opacity : 1;
  const mesh = coneMesh(length, spec.radhi, spec.radlo, faceflag, frame ? RESOLUTION : 2 * RESOLUTION);
  drawConeMesh(f, mesh, drawflag, dir, mid, unit(spec.color), spec.framediam, opacity);
}
