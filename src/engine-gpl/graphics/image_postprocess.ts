/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/image.cpp at b332f28 (contributing authors: Nathan Fabian (Sandia) and Axel Kohlmeyer (Temple)): Image::compute_outline (the depth-jump edges and the outline widening),
 * Image::depth_minmax, Image::box_depth_minmax, Image::compute_depthcue (the fog blend and the background gradient
 * rows of Image::clear) and Image::compute_defocus (the disk-sample blur with the interleaved-gradient rotation), in
 * the order Image::merge calls them: outline, depthcue, defocus, all before the fsaa reduction. LAMMPS is Copyright
 * (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Conventions: the native image is walked in native order, rows counted from the bottom (iy = 0 is the bottom row),
 * as the C++ does; the RasterFrame keeps rows from the top, so the pass copies the buffers to native order and back.
 * The native depth buffer holds zdist - (camera-space offset along camDir), with -1 for an empty pixel; the frame's
 * depth is the offset itself (-Infinity when empty), so the pass converts it.
 */
import type { PostProcessOptions, RasterFrame, Rgb255 } from '../../engine/output/image_objects';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** EPSILON of image.cpp (depth ranges closer than this have no depth variation). */
const EPSILON = 1.0e-6;
/** The golden angle of compute_defocus (radians). */
const GOLDEN_ANGLE = 2.39996322972865332;
const TWO_PI = 2 * Math.PI;

/** saturate() of image.h. */
const saturate = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
/** C lround(): halves round away from zero. */
const lround = (x: number): number => (x < 0 ? -Math.round(-x) : Math.round(x));

/** Native frame data for one pass: rows from the bottom, depth as native (-1 empty). */
interface NativeBuffers {
  w: number;
  h: number;
  rgb: Uint8Array;
  depth: Float64Array;
}

/** Copies the frame into native order (row 0 at the bottom) with native depths. */
function toNative(f: RasterFrame): NativeBuffers {
  const { w, h } = f;
  const rgb = new Uint8Array(w * h * 3);
  const depth = new Float64Array(w * h);
  for (let iy = 0; iy < h; iy++) {
    const row = h - 1 - iy;
    for (let x = 0; x < w; x++) {
      const src = row * w + x, dst = iy * w + x;
      rgb[3 * dst] = f.rgb[3 * src];
      rgb[3 * dst + 1] = f.rgb[3 * src + 1];
      rgb[3 * dst + 2] = f.rgb[3 * src + 2];
      const d = f.depth[src];
      // a pixel the GPL draw wrote keeps native's own depth (exact); any other pixel is converted from the buffer
      const dist = f.dist ? f.dist[src] : -1;
      const nd = dist >= 0 && d === f.zdist - dist ? dist : Number.isFinite(d) ? f.zdist - d : -1;
      depth[dst] = nd < 0 ? -1 : nd;
    }
  }
  return { w, h, rgb, depth };
}

/** Copies the native colour buffer back into the frame (rows from the top). */
function fromNative(f: RasterFrame, n: NativeBuffers): void {
  const { w, h } = f;
  for (let iy = 0; iy < h; iy++) {
    const row = h - 1 - iy;
    for (let x = 0; x < w; x++) {
      const src = iy * w + x, dst = row * w + x;
      f.rgb[3 * dst] = n.rgb[3 * src];
      f.rgb[3 * dst + 1] = n.rgb[3 * src + 1];
      f.rgb[3 * dst + 2] = n.rgb[3 * src + 2];
    }
  }
}

/**
 * Image::compute_outline: a drawn pixel with a neighbour (4-connected) that is background or more than the threshold
 * further away is an edge; drawn pixels within `width` pixels of an edge, on the near side of its depth jump, take the
 * outline colour. `width` is the outline width, doubled with fsaa.
 */
function computeOutline(n: NativeBuffers, threshold: number, width: number, color: Rgb255): void {
  const { w, h, depth, rgb } = n;
  const edges = new Uint8Array(w * h);
  const xoff = [-1, 1, 0, 0], yoff = [0, 0, -1, 1];
  for (let iy = 0; iy < h; iy++) {
    for (let ix = 0; ix < w; ix++) {
      const d = depth[iy * w + ix];
      if (d < 0) continue;
      for (let k = 0; k < 4; k++) {
        const jx = ix + xoff[k], jy = iy + yoff[k];
        if (jx < 0 || jx >= w || jy < 0 || jy >= h) continue;
        const dj = depth[jy * w + jx];
        if (dj < 0 || dj - d > threshold) {
          edges[iy * w + ix] = 1;
          break;
        }
      }
    }
  }
  for (let iy = 0; iy < h; iy++) {
    for (let ix = 0; ix < w; ix++) {
      const d = depth[iy * w + ix];
      if (d < 0) continue;
      let paint = false;
      for (let dy = -width + 1; dy < width && !paint; dy++) {
        const jy = iy + dy;
        if (jy < 0 || jy >= h) continue;
        for (let dx = -width + 1; dx < width; dx++) {
          const jx = ix + dx;
          if (jx < 0 || jx >= w) continue;
          if (!edges[jy * w + jx]) continue;
          if (d - depth[jy * w + jx] < threshold) {
            paint = true;
            break;
          }
        }
      }
      if (paint) {
        const i = iy * w + ix;
        rgb[3 * i] = color[0];
        rgb[3 * i + 1] = color[1];
        rgb[3 * i + 2] = color[2];
      }
    }
  }
}

/** Image::depth_minmax: the depth range of the drawn pixels, or null when nothing is drawn. */
function depthMinmax(depth: Float64Array): [number, number] | null {
  let first = true, dmin = 0, dmax = 0;
  for (let i = 0; i < depth.length; i++) {
    const d = depth[i];
    if (d < 0) continue;
    if (first) {
      dmin = dmax = d;
      first = false;
    } else {
      dmin = Math.min(dmin, d);
      dmax = Math.max(dmax, d);
    }
  }
  return first ? null : [dmin, dmax];
}

/**
 * Image::box_depth_minmax: the near and far side of the box seen from the camera, by projecting its eight corners on
 * the view direction. The camera sits at camPos = camDir * zdist, and the corners are taken relative to the box centre.
 */
function boxDepthMinmax(o: PostProcessOptions, f: RasterFrame): [number, number] {
  const V = f.camDir;
  const camPos = [V[0] * f.zdist, V[1] * f.zdist, V[2] * f.zdist];
  const dcam = camPos[0] * V[0] + camPos[1] * V[1] + camPos[2] * V[2];
  let dnear = 0, dfar = 0;
  for (let ic = 0; ic < 8; ic++) {
    const c = [
      ((ic & 1) ? o.boxHi[0] : o.boxLo[0]) - f.ctr[0],
      ((ic & 2) ? o.boxHi[1] : o.boxLo[1]) - f.ctr[1],
      ((ic & 4) ? o.boxHi[2] : o.boxLo[2]) - f.ctr[2],
    ];
    const d = dcam - (c[0] * V[0] + c[1] * V[1] + c[2] * V[2]);
    if (ic === 0) {
      dnear = dfar = d;
    } else {
      dnear = Math.min(dnear, d);
      dfar = Math.max(dfar, d);
    }
  }
  return [dnear, dfar];
}

/**
 * Image::compute_depthcue: drawn pixels fade towards the fog colour with distance; the fade runs from the start
 * (the nearest drawn pixel, or a fraction of the box depth) to the farthest drawn pixel. Rows of the auto colour
 * follow the backcolor2 gradient of Image::clear (native row iy from the bottom).
 */
function computeDepthcue(n: NativeBuffers, f: RasterFrame, o: PostProcessOptions, dc: NonNullable<PostProcessOptions['depthcue']>): void {
  const { w, h, depth, rgb } = n;
  const mm = depthMinmax(depth);
  if (!mm) return;
  const [dmin, dmax] = mm;
  if (dmax - dmin < EPSILON) return;

  let dstart = dmin;
  if (dc.start !== null) {
    const [dnear, dfar] = boxDepthMinmax(o, f);
    dstart = dnear + dc.start * (dfar - dnear);
    if (dstart >= dmax - EPSILON) return;
  }
  const dscale = dc.strength / (dmax - dstart);

  for (let iy = 0; iy < h; iy++) {
    let red: number, green: number, blue: number;
    const b1 = dc.background, b2 = dc.background2;
    if (dc.color) {
      [red, green, blue] = dc.color;
    } else if (b2) {
      const fraction = iy / h;
      red = Math.trunc(fraction * b2[0] + (1 - fraction) * b1[0]);
      green = Math.trunc(fraction * b2[1] + (1 - fraction) * b1[1]);
      blue = Math.trunc(fraction * b2[2] + (1 - fraction) * b1[2]);
    } else {
      [red, green, blue] = b1;
    }
    for (let ix = 0; ix < w; ix++) {
      const i = iy * w + ix;
      const d = depth[i];
      if (d < 0 || d <= dstart) continue;
      const fr = Math.min(1, (d - dstart) * dscale);
      rgb[3 * i] = Math.trunc((1 - fr) * rgb[3 * i] + fr * red);
      rgb[3 * i + 1] = Math.trunc((1 - fr) * rgb[3 * i + 1] + fr * green);
      rgb[3 * i + 2] = Math.trunc((1 - fr) * rgb[3 * i + 2] + fr * blue);
    }
  }
}

/**
 * Image::compute_defocus: each pixel behind the blur start is averaged over a disk of samples on a golden-angle spiral,
 * with a radius that grows with depth (background: the largest radius). A sample contributes only when its own blur
 * circle reaches the pixel. Pixels closer than half a pixel of radius are left alone.
 */
function computeDefocus(n: NativeBuffers, f: RasterFrame, o: PostProcessOptions, df: NonNullable<PostProcessOptions['defocus']>): void {
  const { w, h, depth, rgb } = n;
  const maxradius = df.strength * 0.01 * h;
  if (maxradius < 1.0) return;
  const mm = depthMinmax(depth);
  if (!mm) return;
  const [dmin, dmax] = mm;
  if (dmax - dmin < EPSILON) return;

  let dstart = dmin;
  if (df.start !== null) {
    const [dnear, dfar] = boxDepthMinmax(o, f);
    dstart = dnear + df.start * (dfar - dnear);
    if (dstart >= dmax - EPSILON) return;
  }
  const dscale = maxradius / (dmax - dstart);

  const npixels = w * h;
  const radiusBuffer = new Float64Array(npixels);
  for (let i = 0; i < npixels; i++) {
    const d = depth[i];
    if (d < 0) radiusBuffer[i] = maxradius;
    else if (d <= dstart) radiusBuffer[i] = 0;
    else radiusBuffer[i] = (d - dstart) * dscale;
  }

  let nsamples = Math.trunc(4.0 * maxradius);
  nsamples = Math.max(nsamples, 16);
  nsamples = Math.min(nsamples, 64);
  const sample = new Float64Array(3 * nsamples);
  for (let s = 0; s < nsamples; s++) {
    const r = Math.sqrt((s + 0.5) / nsamples);
    const angle = s * GOLDEN_ANGLE;
    sample[3 * s] = r * libmCos(angle);
    sample[3 * s + 1] = r * libmSin(angle);
    sample[3 * s + 2] = r;
  }

  const source = rgb.slice();
  for (let iy = 0; iy < h; iy++) {
    for (let ix = 0; ix < w; ix++) {
      const i = iy * w + ix;
      const radius = radiusBuffer[i];
      if (radius < 0.5) continue;

      const noise = (0.06711056 * ix + 0.00583715 * iy) % 1.0;
      const angle = ((52.9829189 * noise) % 1.0) * TWO_PI;
      const cosa = libmCos(angle), sina = libmSin(angle);

      const c = [source[3 * i], source[3 * i + 1], source[3 * i + 2]];
      let wsum = 1.0;
      for (let s = 0; s < nsamples; s++) {
        const dx = radius * (cosa * sample[3 * s] - sina * sample[3 * s + 1]);
        const dy = radius * (sina * sample[3 * s] + cosa * sample[3 * s + 1]);
        const jx = ix + lround(dx), jy = iy + lround(dy);
        if (jx < 0 || jx >= w || jy < 0 || jy >= h) continue;
        const j = jy * w + jx;
        const weight = saturate(radiusBuffer[j] - radius * sample[3 * s + 2]);
        if (weight <= 0) continue;
        c[0] += weight * source[3 * j];
        c[1] += weight * source[3 * j + 1];
        c[2] += weight * source[3 * j + 2];
        wsum += weight;
      }
      rgb[3 * i] = lround(c[0] / wsum);
      rgb[3 * i + 1] = lround(c[1] / wsum);
      rgb[3 * i + 2] = lround(c[2] / wsum);
    }
  }
}

/**
 * The composite passes of Image::merge after SSAO and before the fsaa reduction: compute_outline, compute_depthcue,
 * compute_defocus, each only when on. Works on the frame in place; the frame's rows are from the top.
 */
export function postProcess(f: RasterFrame, o: PostProcessOptions): void {
  if (!o.outline && !o.depthcue && !o.defocus) return;
  const n = toNative(f);
  if (o.outline) {
    // compute_outline: the outline width follows the internal image size with FSAA (w *= 2)
    const width = o.outline.width * (o.fsaa ? 2 : 1);
    computeOutline(n, 0.02 * o.maxdel, width, o.outline.color);
  }
  if (o.depthcue) computeDepthcue(n, f, o, o.depthcue);
  if (o.defocus) computeDefocus(n, f, o, o.defocus);
  fromNative(f, n);
}
