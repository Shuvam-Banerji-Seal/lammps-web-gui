/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/image.cpp at b332f28: Image::draw_pixel (the light sum with metallic 0, gamma 1 and the key
 * specular highlight of dump_image "shiny"), Image::view_params (SSAORadius, SSAOSamples, SSAOJitter, the ambient
 * colour 0.5 with ssao) and Image::compute_SSAO (the horizon-angle occlusion with Bresenham marching over the depth
 * buffer, the interleaved-gradient jitter and the seed shift). The fsaa downsampling of Image::merge is in the core
 * dump image (src/engine/output/dump_image.ts), since it is a plain 2x2 average. LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Conventions: a camera-space normal has x along the image's right, y along its up and z towards the viewer. The
 * frame keeps rows from the top (RasterFrame); compute_SSAO works on native rows counted from the bottom, so the
 * occlusion pass converts the buffers to that order and back.
 */
import type { RasterFrame } from '../../engine/output/image_objects';

type V = [number, number, number];

const sat = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
const dot = (a: V, b: V): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V): V => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [a[0], a[1], a[2]];
};

const DEG = Math.PI / 180;
/** Image constructor light defaults (setup_lights, camera space, z towards the viewer): theta, phi in degrees. */
const dirOf = (theta: number, phi: number): V => [Math.cos(theta) * Math.sin(phi), Math.sin(theta), Math.cos(theta) * Math.cos(phi)];
const KEY_DIR = dirOf(30 * DEG, -45 * DEG);
const KEY_COLOR = 0.9;
const FILL_DIR = dirOf(0, 30 * DEG);
const FILL_COLOR = 0.45;
const BACK_DIR = dirOf(15 * DEG, 180 * DEG);
const BACK_COLOR = 0.9;
/** keyHalfDir: the half vector between the key light and the viewer (0, 0, 1). */
const KEY_HALF = norm([KEY_DIR[0], KEY_DIR[1], 1 + KEY_DIR[2]]);
/** SSAOJitter (view_params): pi / 12. */
const SSAO_JITTER = Math.PI / 12;
/** ambientColor with ssao (view_params): 0.5 per channel; without ssao the ambient colour is 0. */
export const SSAO_AMBIENT = 0.5;

/** The key, fill and back light colours of Image (keyLightColor, fillLightColor, backLightColor; each 0..1 per channel). */
export interface LightColors { key: number; fill: number; back: number }
/** Image's default light colours (Image constructor, setup_lights). */
export const DEFAULT_LIGHTS: LightColors = { key: KEY_COLOR, fill: FILL_COLOR, back: BACK_COLOR };

/**
 * Image::draw_pixel's colour for one surface point: `surface` is the camera-space unit normal, `col` the surface colour
 * (0..1 per channel), `shiny` the dump image shiny value (specular intensity = shiny, hardness = 16 * shiny), `ambient`
 * the ambient colour (0, or 0.5 with ssao). Metallic is 0 and gamma is 1 (the dump image defaults), so the reflect term
 * is 1 and the colour is the sum of the ambient, key diffuse, key specular, fill and back terms, saturated to 0..1.
 */
export function shadeSurface(surface: V, col: V, shiny: number, ambient: number, lights: LightColors = DEFAULT_LIGHTS): V {
  const diffuseKey = sat(dot(surface, KEY_DIR));
  const diffuseFill = sat(dot(surface, FILL_DIR));
  const diffuseBack = sat(dot(surface, BACK_DIR));
  const specularKey = shiny > 0 ? Math.pow(sat(dot(surface, KEY_HALF)), 16 * shiny) * shiny : 0;
  const out: V = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    let c = col[k] * ambient;
    c += col[k] * lights.key * diffuseKey;
    if (shiny > 0) c += lights.key * 1 * specularKey;
    c += col[k] * lights.fill * diffuseFill;
    c += col[k] * lights.back * diffuseBack;
    out[k] = sat(c);
  }
  return out;
}

/**
 * Image::compute_SSAO over a finished frame. `radius` is SSAORadius in world units (maxdel * 0.05 * intensity, with
 * maxdel the twice longest box edge), `samples` is SSAOSamples (int(8 + 32 * intensity)), `seed` the dump image ssao
 * seed. The frame's `surface` (2 values per pixel, set by draw_pixel) gives each pixel's tangent, and the depth buffer
 * holds zdist - native depth. Pixels without a surface are left as they are.
 */
export function applySsao(f: RasterFrame, radius: number, samples: number, seed: number): void {
  const { w, h } = f;
  const surface = f.surface;
  if (!surface) return;
  const n = w * h;
  const delTheta = (2 * Math.PI) / samples;
  const pixelWidth = f.pixelWidth;
  const pixelRadius = Math.trunc(radius / pixelWidth + 0.5);
  const seedshift = (0.618033988749895 * seed) % 1;

  // native order: index = iy * w + ix with iy counted from the bottom row
  const depth = new Float64Array(n);
  const sx = new Float64Array(n), sy = new Float64Array(n);
  const rgbN = new Uint8Array(n * 3);
  for (let iy = 0; iy < h; iy++) {
    const row = h - 1 - iy;
    for (let x = 0; x < w; x++) {
      const src = row * w + x;
      const dst = iy * w + x;
      depth[dst] = Number.isFinite(f.depth[src]) ? f.zdist - f.depth[src] : -1;
      sx[dst] = surface[2 * src];
      sy[dst] = surface[2 * src + 1];
      rgbN[3 * dst] = f.rgb[3 * src];
      rgbN[3 * dst + 1] = f.rgb[3 * src + 1];
      rgbN[3 * dst + 2] = f.rgb[3 * src + 2];
    }
  }

  // horizon directions, evenly spaced; each pixel rotates the table by its jitter angle
  const dirX = new Float64Array(samples), dirY = new Float64Array(samples);
  for (let s = 0; s < samples; s++) {
    dirX[s] = Math.cos(s * delTheta);
    dirY[s] = Math.sin(s * delTheta);
  }

  for (let index = 0; index < n; index++) {
    const x = index % w;
    const y = Math.floor(index / w);
    const cdepth = depth[index];
    if (cdepth < 0) continue;

    const sxp = sx[index], syp = sy[index];
    const sinT = -Math.sqrt(sxp * sxp + syp * syp);

    // interleaved gradient noise: a deterministic jitter per pixel
    const ign = (0.06711056 * x + 0.00583715 * y + seedshift) % 1;
    const mytheta = ((52.9829189 * ign) % 1) * SSAO_JITTER;
    const cosj = Math.cos(mytheta), sinj = Math.sin(mytheta);
    let ao = 0;

    for (let s = 0; s < samples; s++) {
      const hx = cosj * dirX[s] - sinj * dirY[s];
      const hy = sinj * dirX[s] + cosj * dirY[s];
      // multiply by z cross surface tangent so that dot (cos) works here
      const scaledSinT = sinT * (hx * syp + hy * sxp);

      // Bresenham's line algorithm to march over the depth buffer
      const dx = Math.trunc(hx * pixelRadius);
      const dy = Math.trunc(hy * pixelRadius);
      let ex = x + dx;
      if (ex < 0) ex = 0;
      if (ex >= w) ex = w - 1;
      let ey = y + dy;
      if (ey < 0) ey = 0;
      if (ey >= h) ey = h - 1;
      let delta: number, small: number, large: number;
      if (Math.abs(hx) > Math.abs(hy)) {
        small = hx > 0 ? 1 : -1;
        large = hy > 0 ? w : -w;
        delta = Math.abs(hy / hx);
      } else {
        small = hy > 0 ? w : -w;
        large = hx > 0 ? 1 : -1;
        delta = Math.abs(hx / hy);
      }
      const lenIncr = Math.sqrt(1 + delta * delta) * pixelWidth;

      // initialise with one step: the centre point needs no test
      const end = ex + ey * w;
      let ind = index + small;
      let len = lenIncr;
      let err = delta;
      if (err >= 1.0) {
        ind += large;
        err -= 1.0;
      }

      let minPeak = -1;
      let peakLen = 0;
      while ((small > 0 && ind <= end) || (small < 0 && ind >= end)) {
        if (ind < 0 || ind >= n) break;
        if (minPeak < 0 || (depth[ind] >= 0 && depth[ind] < minPeak)) {
          minPeak = depth[ind];
          peakLen = len;
        }
        ind += small;
        len += lenIncr;
        err += delta;
        if (err >= 1.0) {
          ind += large;
          err -= 1.0;
        }
      }

      if (peakLen > 0) {
        const hh = Math.atan((cdepth - minPeak) / peakLen);
        ao += sat(Math.sin(hh) - scaledSinT);
      } else {
        ao += sat(-scaledSinT);
      }
    }
    ao /= samples;

    rgbN[3 * index] = Math.trunc(rgbN[3 * index] * (1 - ao));
    rgbN[3 * index + 1] = Math.trunc(rgbN[3 * index + 1] * (1 - ao));
    rgbN[3 * index + 2] = Math.trunc(rgbN[3 * index + 2] * (1 - ao));
  }

  // back to rows from the top
  for (let iy = 0; iy < h; iy++) {
    const row = h - 1 - iy;
    for (let x = 0; x < w; x++) {
      const src = iy * w + x;
      const dst = row * w + x;
      f.rgb[3 * dst] = rgbN[3 * src];
      f.rgb[3 * dst + 1] = rgbN[3 * src + 1];
      f.rgb[3 * dst + 2] = rgbN[3 * src + 2];
    }
  }
}

/** The helpers dump image reads through gplHook('image') (registry/graphics.ts). */
export interface ImageShadingHook {
  shadeSurface: typeof shadeSurface;
  applySsao: typeof applySsao;
}

export const imageShadingHook: ImageShadingHook = { shadeSurface, applySsao };
