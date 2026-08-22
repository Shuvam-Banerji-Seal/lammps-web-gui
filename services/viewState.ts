import { VisualizationConfig, MaterialType, VisualizationMode, LightingPreset } from '../types';

/**
 * Shareable view state: encodes the visualization configuration into a
 * compact URL-safe string (?s=...). Compact keys keep URLs short; every
 * decoded field is validated against its whitelist and clamped so a crafted
 * URL can never produce an invalid runtime state.
 */

const MATERIALS: MaterialType[] = ['realistic', 'plastic', 'metallic', 'toon'];
const MODES: VisualizationMode[] = ['ball-and-stick', 'space-fill', 'wireframe', 'licorice'];
const LIGHTING: LightingPreset[] = ['studio', 'lab', 'outdoor', 'space', 'soft'];

const HEX6 = /^[0-9a-f]{6}$/;

const bytesToB64Url = (bytes: Uint8Array): string => {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};

const b64UrlToBytes = (s: string): Uint8Array => {
  const b = s.replace(/-/g, '+').replace(/_/g, '/');
  const bin = atob(b);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
};

const clamp = (v: number, lo: number, hi: number): number =>
  Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : lo;

/** Encode a config subset into a URL-safe token. */
export const encodeViewState = (c: VisualizationConfig): string => {
  const view = {
    v: c.visualizationMode,
    m: c.materialType,
    l: c.lightingPreset,
    b: c.showBonds ? 1 : 0,
    x: c.showBox ? 1 : 0,
    t: c.showLabels ? 1 : 0,
    a: c.showAxes ? 1 : 0,
    sh: c.shadowsEnabled ? 1 : 0,
    as: Math.round(c.atomScale * 100),
    bs: Math.round(c.bondScale * 100),
    f: c.fov,
    r: Math.round(c.autoRotateSpeed * 10),
    bg: c.backgroundColor.replace('#', '').toLowerCase(),
    cc: Object.entries(c.customColors)
      .map(([k, hex]) => `${Number(k)}${hex.replace('#', '').toLowerCase()}`)
      .join(','),
  };
  return bytesToB64Url(new TextEncoder().encode(JSON.stringify(view)));
};

/**
 * Decode a token into a partial config. Returns null for malformed input.
 * Unknown fields inside valid JSON are ignored (forward compatibility).
 */
export const decodeViewState = (token: string): Partial<VisualizationConfig> | null => {
  try {
    if (!token || token.length > 4096) return null;
    const json = new TextDecoder().decode(b64UrlToBytes(token));
    const raw = JSON.parse(json) as Record<string, unknown>;
    const out: Partial<VisualizationConfig> = {};

    if (typeof raw.v === 'string' && MODES.includes(raw.v as VisualizationMode)) {
      out.visualizationMode = raw.v as VisualizationMode;
    }
    if (typeof raw.m === 'string' && MATERIALS.includes(raw.m as MaterialType)) {
      out.materialType = raw.m as MaterialType;
    }
    if (typeof raw.l === 'string' && LIGHTING.includes(raw.l as LightingPreset)) {
      out.lightingPreset = raw.l as LightingPreset;
    }
    for (const [key, prop] of [
      ['b', 'showBonds'], ['x', 'showBox'], ['t', 'showLabels'],
      ['a', 'showAxes'], ['sh', 'shadowsEnabled'],
    ] as const) {
      if (raw[key] === 0 || raw[key] === 1) {
        (out as Record<string, unknown>)[prop] = raw[key] === 1;
      }
    }
    if (typeof raw.as === 'number') out.atomScale = clamp(raw.as / 100, 0.1, 3);
    if (typeof raw.bs === 'number') out.bondScale = clamp(raw.bs / 100, 0.1, 3);
    if (typeof raw.f === 'number') out.fov = clamp(Math.round(raw.f), 15, 90);
    if (typeof raw.r === 'number') out.autoRotateSpeed = clamp(raw.r / 10, 0.1, 6);
    if (typeof raw.bg === 'string' && HEX6.test(raw.bg)) {
      out.backgroundColor = `#${raw.bg}`;
    }
    if (typeof raw.cc === 'string' && raw.cc.length > 0) {
      const colors: Record<number, string> = {};
      for (const entry of raw.cc.split(',').slice(0, 256)) {
        const m = entry.match(/^(\d{1,4})([0-9a-f]{6})$/);
        if (m) colors[parseInt(m[1], 10)] = `#${m[2]}`;
      }
      if (Object.keys(colors).length > 0) out.customColors = colors;
    }

    return Object.keys(out).length > 0 ? out : null;
  } catch {
    return null;
  }
};

/** Read ?s= from a location search string. */
export const viewStateFromSearch = (search: string): Partial<VisualizationConfig> | null => {
  try {
    const s = new URLSearchParams(search).get('s');
    return s ? decodeViewState(s) : null;
  } catch {
    return null;
  }
};
