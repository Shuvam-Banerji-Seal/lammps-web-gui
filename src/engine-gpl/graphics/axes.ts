/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/image.cpp (Image::draw_axes, xpm2pix, rgb2yuv), src/GRAPHICS/image_objects.cpp and
 * image_objects.h (ImageObjects::ArrowObj: the arrow template, transform, transform_normals and both draw overloads;
 * RESOLUTION = 36) and the axis letter bitmaps of image.cpp (axes_letters.ts). Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License (the LAMMPS file headers). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * dump image "axes" draws a grey sphere at the origin (its diameter is the radius of the axes), three arrows (red,
 * green, blue; blue only in 3d) and the letters X, Y and Z (32x32 bitmaps, scaled by the smaller image side / 1440).
 * An arrow is a triangle mesh (draw_trinorm, Phong-shaded) for the head and the base, and a draw_cylinder shaft.
 */
import type { RasterFrame, Vec3 } from '../../engine/output/image_objects';
import { drawCylinder, drawSphere, drawTrinorm, drawPixmapBox } from './draw';
import { color2rgb, byteOf, type Col } from './colors';
import { LETTER_X, LETTER_Y, LETTER_Z } from './axes_letters';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

type V = [number, number, number];
type Tri = [V, V, V];

/** ImageObjects::RESOLUTION (image_objects.h): the triangles per set of the arrow template. */
const RESOLUTION = 36;
/** image_objects.cpp: the small angular overlap of the rim triangles. */
const RADOVERLAP = 0.01;
const SMALL = 1.0e-10;
/** The default arguments of ImageObjects::ArrowObj (image_objects.h): tip length, tip width and radius. */
const TIPL = 0.2, TIPW = 0.1, RADIUS = 0.1;
/** image.cpp draw_axes: the letters are offset from the arrow tips by 5% of the arrow length. */
const DIROFFS = 0.05;

const dot3 = (a: V, b: V): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len3 = (a: V): number => Math.sqrt(dot3(a, a));
/** vec3norm: (1 / n) * v, or the zero vector. */
const norm3 = (a: V): V => {
  const n = len3(a);
  return n > 0 ? [(1.0 / n) * a[0], (1.0 / n) * a[1], (1.0 / n) * a[2]] : [0, 0, 0];
};

interface ArrowTemplate { triangles: Tri[]; normals: Tri[]; diameter: number; resolution: number }

/** ArrowObj::ArrowObj: the template of the unit arrow from -0.5 to 0.5 along x (tip cone, flat disc at its base, bottom cap). */
const buildTemplate = (tiplength: number, tipwidth: number, radius: number, res: number): ArrowTemplate => {
  const triangles: Tri[] = [], normals: Tri[] = [];
  const tip: V = [0.5, 0.0, 0.0];
  const mid: V = [0.5 - tiplength, 0.0, 0.0];
  const bot: V = [-0.5, 0.0, 0.0];
  const radinc = (2 * Math.PI) / res;
  const p1: V = [0.5 - tiplength, 0.0, 0.0];
  const p2: V = [0.5 - tiplength, 0.0, 0.0];
  const nflat: V = [-1.0, 0.0, 0.0];
  for (let i = 0; i < res; i++) {
    p1[1] = (radius + tipwidth) * libmSin(radinc * i - RADOVERLAP);
    p1[2] = (radius + tipwidth) * libmCos(radinc * i - RADOVERLAP);
    p2[1] = (radius + tipwidth) * libmSin(radinc * (i + 1));
    p2[2] = (radius + tipwidth) * libmCos(radinc * (i + 1));
    // tip cone: radial normals at the rim, the mid-point rim normal tilted for a mostly pointy tip
    const n1 = norm3([0.0, p2[1], p2[2]]);
    const n2 = norm3([0.25, 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2])]);
    const n3 = norm3([0.0, p1[1], p1[2]]);
    triangles.push([[p2[0], p2[1], p2[2]], [tip[0], tip[1], tip[2]], [p1[0], p1[1], p1[2]]]);
    normals.push([n1, n2, n3]);
    // flat disc at the base of the tip: normal -x
    triangles.push([[p2[0], p2[1], p2[2]], [mid[0], mid[1], mid[2]], [p1[0], p1[1], p1[2]]]);
    normals.push([nflat, nflat, nflat]);
  }
  // bottom cap: normal -x
  p1[0] = -0.5;
  p2[0] = -0.5;
  for (let i = 0; i < res; i++) {
    p1[1] = radius * libmSin(radinc * i - RADOVERLAP);
    p1[2] = radius * libmCos(radinc * i - RADOVERLAP);
    p2[1] = radius * libmSin(radinc * (i + 1));
    p2[2] = radius * libmCos(radinc * (i + 1));
    triangles.push([[p2[0], p2[1], p2[2]], [bot[0], bot[1], bot[2]], [p1[0], p1[1], p1[2]]]);
    normals.push([nflat, nflat, nflat]);
  }
  return { triangles, normals, diameter: 2.0 * radius, resolution: res };
};

/** The templates by tip length (ArrowObj's tiplength; the axes use TIPL, the graphics objects their ratio). */
const templates = new Map<number, ArrowTemplate>();
const arrowTemplate = (tiplength: number = TIPL): ArrowTemplate => {
  let t = templates.get(tiplength);
  if (!t) {
    t = buildTemplate(tiplength, TIPW, RADIUS, RESOLUTION);
    templates.set(tiplength, t);
  }
  return t;
};

/** The orthonormal basis around the unit direction u (ArrowObj transform): v and w. */
const basisOf = (u: V): { v: V; w: V } => {
  const a: V = Math.abs(u[0]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0];
  const v = norm3(cross3(u, a));
  return { v, w: cross3(u, v) };
};

/** ArrowObj's transform: the template re-oriented along dir, scaled by len (along dir) and width, moved to offs. */
const transform = (tpl: ArrowTemplate, dir: V, offs: V, len: number, width: number): Tri[] => {
  const u = norm3(dir);
  if (len3(u) < SMALL) return [];
  const { v, w } = basisOf(u);
  const out: Tri[] = [];
  const pt = (t: V): V => {
    const a = (len * t[0]), b = (width * t[1]), c = (width * t[2]);
    return [(a * u[0] + b * v[0] + c * w[0]) + offs[0], (a * u[1] + b * v[1] + c * w[1]) + offs[1], (a * u[2] + b * v[2] + c * w[2]) + offs[2]];
  };
  for (const tri of tpl.triangles) out.push([pt(tri[0]), pt(tri[1]), pt(tri[2])]);
  return out;
};

/** ArrowObj's transform_normals: the template normals rotated along dir (no scaling). */
const transformNormals = (tpl: ArrowTemplate, dir: V): Tri[] => {
  const u = norm3(dir);
  if (len3(u) < SMALL) return [];
  const { v, w } = basisOf(u);
  const rot = (n: V): V => norm3([n[0] * u[0] + n[1] * v[0] + n[2] * w[0], n[0] * u[1] + n[1] * v[1] + n[2] * w[1], n[0] * u[2] + n[1] * v[2] + n[2] * w[2]]);
  return tpl.normals.map((n): Tri => [rot(n[0]), rot(n[1]), rot(n[2])]);
};

/**
 * ImageObjects::ArrowObj::draw (bottom and tip overload, image_objects.cpp at b332f28): the arrow from bottom to tip in
 * colour col, with the given scale (the shaft diameter); the head and base are draw_trinorm triangles, the shaft a
 * draw_cylinder between the middle of the second set of triangles and the bottom of the last.
 */
export function drawArrow(f: RasterFrame, col: V, bottom: Vec3, tip: Vec3, scale: number, opacity: number): void {
  const tpl = arrowTemplate();
  if (tpl.triangles.length === 0) return;
  const dir: V = [tip[0] - bottom[0], tip[1] - bottom[1], tip[2] - bottom[2]];
  const center: V = [0.5 * dir[0] + bottom[0], 0.5 * dir[1] + bottom[1], 0.5 * dir[2] + bottom[2]];
  const lscale = len3(dir);
  const wscale = scale / tpl.diameter;
  const arrow = transform(tpl, dir, center, lscale, wscale);
  const norms = transformNormals(tpl, dir);
  if (arrow.length === 0) return;
  for (let i = 0; i < arrow.length; i++) {
    drawTrinorm(f, arrow[i][0], arrow[i][1], arrow[i][2], norms[i][0], norms[i][1], norms[i][2], col, col, col, opacity);
  }
  if (arrow.length > tpl.resolution + 2) {
    drawCylinder(f, arrow[1][1], arrow[arrow.length - 1][1], col, scale, 0, opacity);
  }
}

/**
 * ImageObjects::ArrowObj::draw (centre, length and direction overload, image_objects.cpp at b332f28, the call of
 * dump image's ARROW objects): the template of tip length `tiplength` is oriented along dir, scaled by |dir| * length
 * and by scale / 0.2 across, centred on centre; the same triangles and shaft cylinder as drawArrow.
 */
export function drawArrowObj(f: RasterFrame, col: V, center: Vec3, length: number, dir: Vec3, scale: number, opacity: number, tiplength: number): void {
  const tpl = arrowTemplate(tiplength);
  if (tpl.triangles.length === 0) return;
  const dirv: V = [dir[0], dir[1], dir[2]];
  const lscale = len3(dirv) * length;
  const wscale = scale / tpl.diameter;
  const arrow = transform(tpl, dirv, [center[0], center[1], center[2]], lscale, wscale);
  const norms = transformNormals(tpl, dirv);
  if (arrow.length === 0) return;
  for (let i = 0; i < arrow.length; i++) {
    drawTrinorm(f, arrow[i][0], arrow[i][1], arrow[i][2], norms[i][0], norms[i][1], norms[i][2], col, col, col, opacity);
  }
  if (arrow.length > tpl.resolution + 2) {
    drawCylinder(f, arrow[1][1], arrow[arrow.length - 1][1], col, scale, 0, opacity);
  }
}

/**
 * xpm2pix (image.cpp at b332f28): the 32x32 bitmap as an RGB pixmap, rows from the bottom; '#' is the foreground colour,
 * anything else the background, each channel truncated from 0..1 to bytes.
 */
const xpmToPixmap = (rows: readonly string[], fg: Col, bg: Col): Uint8Array => {
  const out = new Uint8Array(32 * 32 * 3);
  for (let row = 0; row < 32; row++) {
    const src = rows[31 - row];
    for (let i = 0; i < 32; i++) {
      const c = src[i] === '#' ? fg : bg;
      const o = row * 32 * 3 + 3 * i;
      out[o] = byteOf(c[0]);
      out[o + 1] = byteOf(c[1]);
      out[o + 2] = byteOf(c[2]);
    }
  }
  return out;
};

/** rgb2yuv (image.cpp): the luma of an integer RGB colour, truncated as native's static_cast<int> does. */
const lumaOf = (rgb: V): number => Math.trunc(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]);

/**
 * Image::draw_axes (image.cpp at b332f28): the origin sphere of diameter radius = diameter / 2 in grey, the arrows from
 * axes[0] to axes[1] (red), axes[2] (green) and, in 3d, axes[3] (blue), then the letters X, Y and Z (in white, or in
 * black on the darker grey when the background is bright: luma above 192). axes are world points; the letters are
 * scaled by min(width, height) / 1440, halved with fsaa (the frame is the doubled image then).
 */
export function drawAxes(f: RasterFrame, axes: Vec3[], diameter: number, opacity: number, background: [number, number, number], fsaa: boolean, dimension: 2 | 3): void {
  if (!(diameter > 0) || !(opacity > 0)) return;
  const radius = 0.5 * diameter;
  drawSphere(f, axes[0], color2rgb('gray')!, radius, opacity);
  drawArrow(f, color2rgb('red')!, axes[0], axes[1], radius, opacity);
  drawArrow(f, color2rgb('green')!, axes[0], axes[2], radius, opacity);
  if (dimension === 3) drawArrow(f, color2rgb('blue')!, axes[0], axes[3], radius, opacity);

  let scale = Math.min(f.w, f.h) / 1440.0;
  if (fsaa) scale *= 0.5;

  let fontcolor = color2rgb('white')!;
  let backcolor = color2rgb('silver')!;
  if (lumaOf(background) > 192) {
    fontcolor = color2rgb('black')!;
    backcolor = color2rgb('darkgray')!;
  }
  // the transparent colour is the background colour in bytes, as draw_pixmap compares it
  const trans: [number, number, number] = [backcolor[0] * 255, backcolor[1] * 255, backcolor[2] * 255];

  const a = axes;
  const shifted: [V, V, V] = [
    [a[1][0] + DIROFFS * (a[1][0] - a[0][0]), a[1][1] + radius, a[1][2] - radius],
    [a[2][0] + radius, a[2][1] + DIROFFS * (a[2][1] - a[0][1]), a[2][2] + radius],
    [a[3][0] + radius, a[3][1] + radius, a[3][2] + DIROFFS * (a[3][2] - a[0][2])],
  ];
  drawPixmapBox(f, shifted[0], 32, 32, xpmToPixmap(LETTER_X, fontcolor, backcolor), trans, scale);
  drawPixmapBox(f, shifted[1], 32, 32, xpmToPixmap(LETTER_Y, fontcolor, backcolor), trans, scale);
  if (dimension === 3) drawPixmapBox(f, shifted[2], 32, 32, xpmToPixmap(LETTER_Z, fontcolor, backcolor), trans, scale);
}
