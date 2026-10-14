/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/image.cpp (Image::draw_sphere, draw_cylinder, draw_trinorm, draw_pixmap (both overloads),
 * draw_pixel, scale_pixmap, view_params and the light defaults of the Image constructor) and the object dispatch of
 * src/GRAPHICS/dump_image.cpp (the SPHERE, TRINORM, CYLINDER, PIXMAP and BOND branches of DumpImage::create_image).
 * The label bitmap is the create_label() pixmap of src/GRAPHICS/scalable_font.cpp, here rasterised from the strokes
 * of scalable_font.ts. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Conventions. The functions work in the native image coordinates: ix is the column, iy the row counted from the
 * bottom (dump image writes its top row first). RasterFrame keeps rows from the top, so a native row iy is stored at
 * row h - 1 - iy. The native depth of a point is the distance from the camera, dist = zdist - (x - ctr) . camDir;
 * the frame buffer keeps the same ordering as dump_image.ts (larger = nearer), so the buffer value is zdist - depth.
 * Pixel acceptance is draw_pixel's: a depth below zero (behind the camera) or a depth not nearer than the buffer is
 * rejected.
 *
 * Simplifications, all fixed by the engine's dump image keywords: the view is orthographic (dump image's persp is not
 * supported), opacity is 1 (atrans and the transparency thresholds are not used), metallic is 0, gamma is 1, shiny is
 * 1 and ssao is off (the ambient colour is 0), so draw_pixel reduces to the diffuse key, fill and back lights plus
 * the key specular highlight.
 */
import type { LabelObject, GraphicsObject, PixmapObject, Rgb, CylinderObject, SphereObject, BondObject, TriNormObject } from './objects';
import type { ObjectStyle, RasterFrame, Rgb255 } from '../../engine/output/image_objects';
import { shadeSurface, DEFAULT_LIGHTS, type LightColors } from './image_shading';
import { TRANK, TRANSTHRESH } from './transthresh';

type V = [number, number, number];

/**
 * The light set of one draw_pixel call: Image::ambientColor and the key, fill and back colours. Image::draw_pixel reads
 * them from the image; draw_triangle changes them for its faces with reset_lighting (dump_image.cpp).
 */
export interface PixelLight { ambient: number; lights: LightColors }

/**
 * Image::draw_sphere, draw_cylinder, draw_triangle and draw_trinorm skip a pixel when opacity < 1 and transthresh[ix % 16][iy % 16]
 * > opacity (screen-door transparency); ix is the column and iy the native row from the bottom.
 */
export const transparentAt = (opacity: number, ix: number, iy: number): boolean => opacity < 1.0 && TRANSTHRESH[ix % TRANK][iy % TRANK] > opacity;

const sat = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
const dot = (a: V, b: V): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: V): number => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V): V => {
  const l = len(a);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [a[0], a[1], a[2]];
};
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V, s: number): V => [a[0] * s, a[1] * s, a[2] * s];
/** std::lround for the non-negative values used here (rounds half away from zero). */
const lround = (x: number): number => (x < 0 ? -Math.round(-x) : Math.round(x));
const trunc = Math.trunc;

/** The transparency test of draw_pixmap: a pixel within this of the transcolor (0..1 units) is skipped. */
const TRANS_DELTA = 0.01;

/**
 * draw_pixel: shade one pixel of a surface with camera-space normal `surface` and colour `col` (0..1 per channel),
 * native row iy from the bottom. `depth` is the native distance from the camera.
 */
export function drawPixel(f: RasterFrame, ix: number, iy: number, depth: number, surface: V, col: V, light?: PixelLight): void {
  if (!Number.isFinite(depth)) return;
  if (depth < 0) return;
  if (ix < 0 || iy < 0 || ix >= f.w || iy >= f.h) return;
  const row = f.h - 1 - iy;
  const idx = row * f.w + ix;
  const buf = f.zdist - depth;
  // depth not nearer than the buffer: rejected (larger buffer value = nearer)
  if (buf <= f.depth[idx]) return;
  f.depth[idx] = buf;
  // the tangent of the surface, kept for the ssao pass (native surfaceBuffer: surface[1], -surface[0])
  if (f.surface) {
    f.surface[2 * idx] = surface[1];
    f.surface[2 * idx + 1] = -surface[0];
  }
  const c = light
    ? shadeSurface(surface, col, f.shiny ?? 1, light.ambient, light.lights)
    : shadeSurface(surface, col, f.shiny ?? 1, f.ambient ?? 0, DEFAULT_LIGHTS);
  const o = idx * 3;
  f.rgb[o] = trunc(c[0] * 255);
  f.rgb[o + 1] = trunc(c[1] * 255);
  f.rgb[o + 2] = trunc(c[2] * 255);
}

/** The orthographic projection of a box point: native image coordinates (xf, yf) before truncation, and the depth. */
function project(f: RasterFrame, x: V): { xf: number; yf: number; dist: number } {
  const xl = sub(x, f.ctr);
  return {
    xf: dot(f.camRight, xl) / f.pixelWidth,
    yf: dot(f.camUp, xl) / f.pixelWidth,
    dist: f.zdist - dot(xl, f.camDir),
  };
}

/** Image::draw_sphere: a sphere of the given full diameter at x (box coordinates). */
export function drawSphere(f: RasterFrame, x: V, col: V, diameter: number, opacity = 1): void {
  if (!(diameter > 0) || !(opacity > 0)) return;
  const xl = sub(x, f.ctr);
  const xmap = dot(f.camRight, xl);
  const ymap = dot(f.camUp, xl);
  const dist = f.zdist - dot(xl, f.camDir);
  const radius = 0.5 * diameter;
  const radsq = radius * radius;
  const pw = f.pixelWidth;
  const pixelRadius = lround(radius / pw) + 1;
  const xf = xmap / pw, yf = ymap / pw;
  const xi = trunc(xf), yi = trunc(yf);
  const widthError = xf - xi, heightError = yf - yi;
  const xc = xi + trunc(f.w / 2), yc = yi + trunc(f.h / 2);
  for (let iy = yc - pixelRadius; iy <= yc + pixelRadius; iy++) {
    for (let ix = xc - pixelRadius; ix <= xc + pixelRadius; ix++) {
      if (iy < 0 || iy >= f.h || ix < 0 || ix >= f.w) continue;
      if (transparentAt(opacity, ix, iy)) continue;
      const sy = ((iy - yc) - heightError) * pw;
      const sx = ((ix - xc) - widthError) * pw;
      const projRad = sx * sx + sy * sy;
      if (projRad > radsq) continue;
      const sz = Math.sqrt(radsq - projRad);
      drawPixel(f, ix, iy, dist - sz, [sx / radius, sy / radius, sz / radius], col);
    }
  }
}

/**
 * Image::draw_cylinder: a cylinder of full diameter between the box points x and y. sflag bit 0 adds a sphere cap at
 * x, bit 1 one at y (the flag1 of dump image's "fix" keyword).
 */
export function drawCylinder(f: RasterFrame, x: V, y: V, col: V, diameter: number, sflag: number, opacity = 1): void {
  if (!(diameter > 0) || !(opacity > 0)) return;
  if (sflag % 2 === 1) drawSphere(f, x, col, diameter, opacity);
  if (trunc(sflag / 2) > 0) drawSphere(f, y, col, diameter, opacity);
  const radius = 0.5 * diameter;
  const radsq = radius * radius;

  let zaxis: V = sub(y, x);
  const rasterWidth = Math.abs(dot(zaxis, f.camRight)) + diameter;
  const rasterHeight = Math.abs(dot(zaxis, f.camUp)) + diameter;
  const mid = sub(scale(add(y, x), 0.5), f.ctr);
  let half = len(zaxis);
  if (half === 0) return;
  zaxis = scale(zaxis, 1 / half);
  half *= 0.5;
  const zmax = half, zmin = -half;

  const xmap = dot(f.camRight, mid);
  const ymap = dot(f.camUp, mid);
  const dist = f.zdist - dot(mid, f.camDir);
  const pw = f.pixelWidth;
  const xf = xmap / pw, yf = ymap / pw;
  const xi = trunc(xf), yi = trunc(yf);
  const widthError = xf - xi, heightError = yf - yi;
  const xc = xi + trunc(f.w / 2), yc = yi + trunc(f.h / 2);
  const pixelHalfWidth = lround((rasterWidth * 0.5) / pw);
  const pixelHalfHeight = lround((rasterHeight * 0.5) / pw);

  if (zaxis[0] === f.camDir[0] && zaxis[1] === f.camDir[1] && zaxis[2] === f.camDir[2]) return;
  if (zaxis[0] === -f.camDir[0] && zaxis[1] === -f.camDir[1] && zaxis[2] === -f.camDir[2]) return;

  const yaxis = norm(cross(zaxis, f.camDir));
  const xaxis = norm(cross(yaxis, zaxis));
  const camLDir: V = [dot(f.camDir, xaxis), 0, dot(f.camDir, zaxis)];
  const camLRight = norm([dot(f.camRight, xaxis), dot(f.camRight, yaxis), dot(f.camRight, zaxis)]);
  const camLUp = norm([dot(f.camUp, xaxis), dot(f.camUp, yaxis), dot(f.camUp, zaxis)]);
  const a = camLDir[0] * camLDir[0];

  for (let iy = yc - pixelHalfHeight; iy <= yc + pixelHalfHeight; iy++) {
    for (let ix = xc - pixelHalfWidth; ix <= xc + pixelHalfWidth; ix++) {
      if (iy < 0 || iy >= f.h || ix < 0 || ix >= f.w) continue;
      if (transparentAt(opacity, ix, iy)) continue;
      const sy = ((iy - yc) - heightError) * pw;
      const sx = ((ix - xc) - widthError) * pw;
      const surface: V = [
        camLRight[0] * sx + camLUp[0] * sy,
        camLRight[1] * sx + camLUp[1] * sy,
        camLRight[2] * sx + camLUp[2] * sy,
      ];
      const b = 2 * camLDir[0] * surface[0];
      const c = surface[0] * surface[0] + surface[1] * surface[1] - radsq;
      let partial = b * b - 4 * a * c;
      if (partial < 0 || a === 0) continue;
      partial = Math.sqrt(partial);
      let t = (-b + partial) / (2 * a);
      const t2 = (-b - partial) / (2 * a);
      if (t2 > t) t = t2;
      const s: V = [surface[0] + t * camLDir[0], surface[1] + t * camLDir[1], surface[2] + t * camLDir[2]];
      if (s[2] > zmax || s[2] < zmin) continue;
      const normal: V = [s[0] / radius, s[1] / radius, 0];
      const cn: V = [dot(normal, camLRight), dot(normal, camLUp), dot(normal, camLDir)];
      drawPixel(f, ix, iy, dist - t, cn, col);
    }
  }
}

/**
 * Image::draw_trinorm: a triangle with per-vertex normals and colours, Phong-style interpolated over the pixels of the
 * triangle. Vertices x, y, z; normals nx, ny, nz; colours cx, cy, cz (0..1).
 */
export function drawTrinorm(f: RasterFrame, x: V, y: V, z: V, nx: V, ny: V, nz: V, cx: V, cy: V, cz: V, opacity = 1): void {
  if (!(opacity > 0)) return;
  const xl = sub(x, f.ctr), yl = sub(y, f.ctr), zl = sub(z, f.ctr);
  const d1l = len(sub(xl, yl));
  if (d1l === 0) return;
  const d1 = scale(sub(xl, yl), 1 / d1l);
  const d2l = len(sub(zl, yl));
  if (d2l === 0) return;
  const d2 = scale(sub(zl, yl), 1 / d2l);
  const normal = norm(cross(d1, d2));
  let invndotd = dot(normal, f.camDir);
  if (invndotd === 0) return;
  invndotd = 1 / invndotd;

  const r: V = [dot(f.camRight, xl), dot(f.camRight, yl), dot(f.camRight, zl)];
  const u: V = [dot(f.camUp, xl), dot(f.camUp, yl), dot(f.camUp, zl)];
  const rasterLeft = r[0] - Math.min(r[0], r[1], r[2]);
  const rasterRight = Math.max(r[0], r[1], r[2]) - r[0];
  const rasterDown = u[0] - Math.min(u[0], u[1], u[2]);
  const rasterUp = Math.max(u[0], u[1], u[2]) - u[0];

  const xmap = dot(f.camRight, xl);
  const ymap = dot(f.camUp, xl);
  const dist = f.zdist - dot(xl, f.camDir);
  const pw = f.pixelWidth;
  const xf = xmap / pw;
  const yf = ymap / pw;
  const xc0 = Math.floor(xf), yc0 = Math.floor(yf);
  const widthError = xf - xc0, heightError = yf - yc0;
  const xc = xc0 + trunc(f.w / 2), yc = yc0 + trunc(f.h / 2);

  const pixelLeft = Math.ceil(rasterLeft / pw);
  const pixelRight = Math.ceil(rasterRight / pw);
  const pixelDown = Math.ceil(rasterDown / pw);
  const pixelUp = Math.ceil(rasterUp / pw);

  const v0 = sub(yl, xl), v1 = sub(zl, xl);
  const d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
  const denom = d00 * d11 - d01 * d01;
  if (denom === 0) return;
  const invDenom = 1 / denom;

  for (let iy = yc - pixelDown; iy <= yc + pixelUp; iy++) {
    for (let ix = xc - pixelLeft; ix <= xc + pixelRight; ix++) {
      if (iy < 0 || iy >= f.h || ix < 0 || ix >= f.w) continue;
      if (transparentAt(opacity, ix, iy)) continue;
      const sy = ((iy - yc) - heightError) * pw;
      const sx = ((ix - xc) - widthError) * pw;
      const surface: V = [f.camRight[0] * sx + f.camUp[0] * sy, f.camRight[1] * sx + f.camUp[1] * sy, f.camRight[2] * sx + f.camUp[2] * sy];
      const t = -dot(normal, surface) * invndotd;
      const p: V = [xl[0] + surface[0] + f.camDir[0] * t, xl[1] + surface[1] + f.camDir[1] * t, xl[2] + surface[2] + f.camDir[2] * t];
      const v2 = sub(p, xl);
      const d20 = dot(v2, v0), d21 = dot(v2, v1);
      const lambdaY = (d11 * d20 - d01 * d21) * invDenom;
      const lambdaZ = (d00 * d21 - d01 * d20) * invDenom;
      const lambdaX = 1 - lambdaY - lambdaZ;
      if (lambdaX < 0 || lambdaY < 0 || lambdaZ < 0) continue;

      const inormal = norm([
        lambdaX * nx[0] + lambdaY * ny[0] + lambdaZ * nz[0],
        lambdaX * nx[1] + lambdaY * ny[1] + lambdaZ * nz[1],
        lambdaX * nx[2] + lambdaY * ny[2] + lambdaZ * nz[2],
      ]);
      const icolor: V = [
        lambdaX * cx[0] + lambdaY * cy[0] + lambdaZ * cz[0],
        lambdaX * cx[1] + lambdaY * cy[1] + lambdaZ * cz[1],
        lambdaX * cx[2] + lambdaY * cy[2] + lambdaZ * cz[2],
      ];
      const cNormal: V = [dot(f.camRight, inormal), dot(f.camUp, inormal), dot(f.camDir, inormal)];
      drawPixel(f, ix, iy, dist - t, cNormal, icolor);
    }
  }
}

/**
 * Image::draw_triangle: a flat triangle with corner points x, y, z (box coordinates) in colour col (0..1). A pixel
 * is inside when it lies on the same side of each edge as the opposite corner (three dot-product tests in native
 * order). The face normal is shaded per pixel with the frame's light set, or with `light` when given (dump_image.cpp
 * reset_lighting for tri faces).
 */
export function drawTriangle(f: RasterFrame, x: V, y: V, z: V, col: V, opacity = 1, light?: PixelLight): void {
  if (!(opacity > 0)) return;
  const xl = sub(x, f.ctr), yl = sub(y, f.ctr), zl = sub(z, f.ctr);
  const d1l = len(sub(xl, yl));
  if (d1l === 0) return;
  const d1 = scale(sub(xl, yl), 1 / d1l);
  const d2l = len(sub(zl, yl));
  if (d2l === 0) return;
  const d2 = scale(sub(zl, yl), 1 / d2l);
  const normal = norm(cross(d1, d2));
  let invndotd = dot(normal, f.camDir);
  // a triangle parallel to the camera is invisible
  if (invndotd === 0) return;
  invndotd = 1 / invndotd;

  const r: V = [dot(f.camRight, xl), dot(f.camRight, yl), dot(f.camRight, zl)];
  const u: V = [dot(f.camUp, xl), dot(f.camUp, yl), dot(f.camUp, zl)];
  const rasterLeft = r[0] - Math.min(r[0], r[1], r[2]);
  const rasterRight = Math.max(r[0], r[1], r[2]) - r[0];
  const rasterDown = u[0] - Math.min(u[0], u[1], u[2]);
  const rasterUp = Math.max(u[0], u[1], u[2]) - u[0];

  const xmap = dot(f.camRight, xl), ymap = dot(f.camUp, xl);
  const dist = f.zdist - dot(xl, f.camDir);
  const pw = f.pixelWidth;
  const xf = xmap / pw, yf = ymap / pw;
  const xc0 = Math.floor(xf), yc0 = Math.floor(yf);
  const widthError = xf - xc0, heightError = yf - yc0;
  const xc = xc0 + trunc(f.w / 2), yc = yc0 + trunc(f.h / 2);

  const pixelLeft = Math.ceil(rasterLeft / pw);
  const pixelRight = Math.ceil(rasterRight / pw);
  const pixelDown = Math.ceil(rasterDown / pw);
  const pixelUp = Math.ceil(rasterUp / pw);
  const cNormal: V = [dot(f.camRight, normal), dot(f.camUp, normal), dot(f.camDir, normal)];

  for (let iy = yc - pixelDown; iy <= yc + pixelUp; iy++) {
    for (let ix = xc - pixelLeft; ix <= xc + pixelRight; ix++) {
      if (iy < 0 || iy >= f.h || ix < 0 || ix >= f.w) continue;
      if (transparentAt(opacity, ix, iy)) continue;
      const sy = ((iy - yc) - heightError) * pw;
      const sx = ((ix - xc) - widthError) * pw;
      const surface: V = [f.camRight[0] * sx + f.camUp[0] * sy, f.camRight[1] * sx + f.camUp[1] * sy, f.camRight[2] * sx + f.camUp[2] * sy];
      const t = -dot(normal, surface) * invndotd;
      // the point on the face plane, in box-local coordinates
      const p: V = [xl[0] + surface[0] + f.camDir[0] * t, xl[1] + surface[1] + f.camDir[1] * t, xl[2] + surface[2] + f.camDir[2] * t];
      // inside test: for each edge, p and the opposite corner lie on the same side (native order: z-x-y, x-y-z, y-z-x)
      if (!sameSide(sub(zl, xl), sub(yl, xl), sub(p, xl))) continue;
      if (!sameSide(sub(xl, yl), sub(zl, yl), sub(p, yl))) continue;
      if (!sameSide(sub(yl, zl), sub(xl, zl), sub(p, zl))) continue;
      drawPixel(f, ix, iy, dist - t, cNormal, col, light);
    }
  }
}

/** The side test of draw_triangle: cross(s1, s2) and cross(s1, s3) point the same way (dot >= 0). */
const sameSide = (s1: V, s2: V, s3: V): boolean => dot(cross(s1, s2), cross(s1, s3)) >= 0;

/** Image::scale_pixmap: bilinear resampling of an RGB pixmap (rows from the bottom) to nw x nh. */
export function scalePixmap(ow: number, oh: number, opix: Uint8Array, nw: number, nh: number): Uint8Array {
  const out = new Uint8Array(nw * nh * 3);
  const xRatio = (ow - 1) / nw;
  const yRatio = (oh - 1) / nh;
  const at = (x: number, y: number, k: number) => opix[(y * ow + x) * 3 + k];
  for (let i = 0; i < nh; i++) {
    for (let j = 0; j < nw; j++) {
      const x = trunc(xRatio * j);
      const y = trunc(yRatio * i);
      const xd = xRatio * j - x;
      const yd = yRatio * i - y;
      // the C++ reads x+1 and y+1 too; their weights are zero at the last column or row, so clamp the index
      const x1 = Math.min(x + 1, ow - 1), y1 = Math.min(y + 1, oh - 1);
      for (let k = 0; k < 3; k++) {
        const a = at(x, y, k), b = at(x1, y, k), c = at(x, y1, k), d = at(x1, y1, k);
        out[(i * nw + j) * 3 + k] = trunc(a * (1 - xd) * (1 - yd) + b * xd * (1 - yd) + c * yd * (1 - xd) + d * (xd * yd));
      }
    }
  }
  return out;
}

/**
 * Image::draw_pixmap at native pixel (xc, yc): the pixmap (rows from the bottom, pwidth x pheight) is scaled by
 * `scale`, centred on (xc, yc), and each pixel that is not the transparent colour is drawn with the normal (0, 0, 1)
 * so that the light shades it. `dist` is the depth of every pixel.
 */
function drawPixmapAt(f: RasterFrame, xc0: number, yc0: number, pwidth0: number, pheight0: number, pixmap: Uint8Array, trans: V, sc: number, dist: number): void {
  if (pwidth0 === 0 || pheight0 === 0 || sc <= 0) return;
  let pix = pixmap;
  let pwidth = pwidth0, pheight = pheight0;
  if (sc !== 1) {
    const nw = lround(sc * pwidth + 0.5), nh = lround(sc * pheight + 0.5);
    pix = scalePixmap(pwidth, pheight, pixmap, nw, nh);
    pwidth = nw;
    pheight = nh;
  }
  const normal: V = [0, 0, 1];
  for (let j = 0; j < pheight; j++) {
    for (let i = 0; i < pwidth; i++) {
      const iy = yc0 + j - trunc(pheight / 2);
      const ix = xc0 + i - trunc(pwidth / 2);
      if (iy < 0 || iy >= f.h || ix < 0 || ix >= f.w) continue;
      const o = 3 * j * pwidth + 3 * i;
      const pc: V = [pix[o] / 255, pix[o + 1] / 255, pix[o + 2] / 255];
      if (Math.abs(pc[0] - trans[0]) < TRANS_DELTA && Math.abs(pc[1] - trans[1]) < TRANS_DELTA && Math.abs(pc[2] - trans[2]) < TRANS_DELTA) continue;
      drawPixel(f, ix, iy, dist, normal, pc);
    }
  }
}

/** The transparent colour of a pixmap in 0..1 units; a colour of -1 never matches (no transparent colour). */
const transOf = (t: Rgb | null): V => (t ? [t[0] / 255, t[1] / 255, t[2] / 255] : [-1, -1, -1]);

/** Image::draw_pixmap with box coordinates (the default coordinate mode of dump image's fix pixmaps). */
export function drawPixmapBox(f: RasterFrame, x: V, pwidth: number, pheight: number, pixmap: Uint8Array, trans: Rgb | null, sc: number): void {
  if (pwidth === 0 || pheight === 0 || sc <= 0) return;
  const p = project(f, x);
  const xc = trunc(p.xf) + trunc(f.w / 2);
  const yc = trunc(p.yf) + trunc(f.h / 2);
  drawPixmapAt(f, xc, yc, pwidth, pheight, pixmap, transOf(trans), sc, p.dist);
}

/**
 * Image::draw_pixmap with image coordinates (dump image flag1 != 0): the pixmap is centred on the native pixel (xc, yc)
 * and its depth is 0 (the default of the overload), so it is drawn in front of the objects.
 */
export function drawPixmapImage(f: RasterFrame, xc: number, yc: number, pwidth: number, pheight: number, pixmap: Uint8Array, trans: Rgb | null, sc: number): void {
  if (pwidth === 0 || pheight === 0 || sc <= 0) return;
  drawPixmapAt(f, trunc(xc), trunc(yc), pwidth, pheight, pixmap, transOf(trans), sc, 0);
}

/**
 * The label bitmap of fix graphics/labels: the background and a frame of `frame` pixels, and the glyph strokes in the
 * font colour (create_label() of scalable_font.cpp fills the pixmap the same way; the strokes replace its bitmap).
 * Rows are counted from the bottom, as the native pixmap is.
 */
export function rasterLabel(l: LabelObject): PixmapObject {
  const W = l.width, H = l.height;
  const rgb = new Uint8Array(W * H * 3);
  const xh = l.frame;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const c = y < xh || y >= H - xh || x < xh || x >= W - xh ? l.framecolor : l.backcolor;
      const o = 3 * (y * W + x);
      rgb[o] = c[0];
      rgb[o + 1] = c[1];
      rgb[o + 2] = c[2];
    }
  }
  const put = (x: number, y: number) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const o = 3 * (y * W + x);
    rgb[o] = l.fontcolor[0];
    rgb[o + 1] = l.fontcolor[1];
    rgb[o + 2] = l.fontcolor[2];
  };
  for (let k = 0; k + 3 < l.strokes.length; k += 4) {
    const x0 = l.strokes[k], y0 = l.strokes[k + 1], x1 = l.strokes[k + 2], y1 = l.strokes[k + 3];
    const n = Math.max(1, Math.ceil(2 * Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0))));
    for (let s = 0; s <= n; s++) {
      const t = s / n;
      put(Math.floor(x0 + (x1 - x0) * t), Math.floor(y0 + (y1 - y0) * t));
    }
  }
  return { kind: 'pixmap', pos: l.pos, width: W, height: H, scale: l.scale, transcolor: l.transcolor, rgb };
}

/** Colour of an object's atom type, 0..1 per channel (the colour tables are 0..255). */
const unit = (c: Rgb255): V => [c[0] / 255, c[1] / 255, c[2] / 255];

/**
 * Draws one graphics object the way dump image's create_image dispatches its native kind, with the colours and
 * diameters the keyword's style gives (style.colorOf, style.diameterOf, style.bondDiameter, flag1, flag2).
 */
export function drawGraphicsObject(f: RasterFrame, st: ObjectStyle, o: GraphicsObject): void {
  const flag1 = trunc(st.flag1);
  const flag2 = st.flag2;
  switch (o.kind) {
    case 'sphere': {
      const s: SphereObject = o;
      // a negative diameter asks for the dump image's default size of the type (dump_image.cpp: objarray[4] < 0)
      const d = s.diameter < 0 ? st.diameterOf(s.type) : s.diameter;
      drawSphere(f, [s.x, s.y, s.z], unit(st.colorOf(s.type)), d + flag2, st.opacity);
      return;
    }
    case 'cylinder': {
      const c: CylinderObject = o;
      drawCylinder(f, [c.x1, c.y1, c.z1], [c.x2, c.y2, c.z2], unit(st.colorOf(c.type)), c.diameter + flag2, flag1, st.opacity);
      return;
    }
    case 'bond': {
      // BOND: two cylinders, each half in the colour of its atom type, capped when flag1 is set
      const b: BondObject = o;
      const p1: V = [b.x1, b.y1, b.z1], p2: V = [b.x2, b.y2, b.z2];
      const half = scale(sub(p2, p1), 0.5);
      const mid = add(p1, half);
      const d = st.bondDiameter(b.types[0], b.types[1]) + flag2;
      const cap = flag1 !== 0 ? 3 : 0;
      drawCylinder(f, p1, mid, unit(st.colorOf(b.types[0])), d, cap);
      drawCylinder(f, sub(p2, half), p2, unit(st.colorOf(b.types[1])), d, cap);
      return;
    }
    case 'trinorm': {
      const t: TriNormObject = o;
      const c = t.types.map((ty) => unit(st.colorOf(ty))) as V[];
      const [v0, v1, v2] = t.vertices;
      const [n0, n1, n2] = t.normals;
      if (flag1 % 2 !== 0) {
        drawTrinorm(f, v0, v1, v2, n0, n1, n2, c[0], c[1], c[2], st.opacity);
      } else {
        // the wireframe: three cylinders in the colour of the first vertex, diameter flag2 (dump_image.cpp TRINORM)
        drawCylinder(f, v0, v1, c[0], flag2, 3, st.opacity);
        drawCylinder(f, v1, v2, c[0], flag2, 3, st.opacity);
        drawCylinder(f, v2, v0, c[0], flag2, 3, st.opacity);
      }
      return;
    }
    case 'pixmap': {
      drawPixmapObject(f, o, flag1);
      return;
    }
    case 'label': {
      drawPixmapObject(f, rasterLabel(o), flag1);
      return;
    }
  }
}

function drawPixmapObject(f: RasterFrame, p: PixmapObject, flag1: number): void {
  if (flag1 === 0) drawPixmapBox(f, p.pos, p.width, p.height, p.rgb, p.transcolor, p.scale);
  else drawPixmapImage(f, p.pos[0], p.pos[1], p.width, p.height, p.rgb, p.transcolor, p.scale);
}
