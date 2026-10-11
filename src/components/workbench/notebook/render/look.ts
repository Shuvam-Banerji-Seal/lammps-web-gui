/**
 * Look options of a render: background, atom radius scale, box, and the transparent background.
 *
 * The 3D canvas is opaque (its context is created with alpha off, shared with the Structure Viewer), so a
 * transparent frame is made by difference matting: the same frame is drawn over black and over white, and the
 * alpha is the difference (1 - (white - black)) with the colour recovered from the black pass. Pure.
 */

export type BackgroundChoice = 'theme' | 'black' | 'white' | 'custom' | 'transparent';
export const BACKGROUND_LABEL: Record<BackgroundChoice, string> = {
  theme: 'Theme (as the view)',
  black: 'Black',
  white: 'White',
  custom: 'Custom colour',
  transparent: 'Transparent (PNG only)',
};

/** The background colour a pass draws, or null to keep the view's own background. */
export const backgroundHex = (choice: BackgroundChoice, custom: string): string | null => {
  if (choice === 'black') return '#000000';
  if (choice === 'white') return '#ffffff';
  if (choice === 'custom') return /^#[0-9a-f]{6}$/i.test(custom) ? custom : '#000000';
  return null;
};

/** Formats whose output has an alpha channel (only PNG frames and snapshots keep a transparent background). */
export const supportsTransparency = (format: 'png' | 'gif' | 'video'): boolean => format === 'png';

export const RADIUS_SCALE_CHOICES = [0.5, 0.75, 1, 1.25, 1.5, 2] as const;

/**
 * The simulation box in the render: as the view, or hidden. The box is only drawn while the view shows it, so a
 * render cannot add one.
 */
export type BoxChoice = 'view' | 'off';
export const BOX_LABEL: Record<BoxChoice, string> = { view: 'As the view', off: 'Off' };

/**
 * Difference matting: black and white passes of the same frame to RGBA. For a pixel drawn with alpha a and colour
 * c over black the pass gives a·c, over white a·c + (1 − a), so the difference gives a. Each channel gives the
 * same a; the three estimates are averaged. Colour is the black pass divided by a. Both inputs are RGBA of the
 * same length (only their RGB is read).
 */
export const unmatte = (black: Uint8ClampedArray, white: Uint8ClampedArray): Uint8ClampedArray<ArrayBuffer> => {
  const n = black.length >> 2;
  const out = new Uint8ClampedArray(n * 4);
  for (let p = 0; p < n; p++) {
    const o = 4 * p;
    const dr = white[o] - black[o];
    const dg = white[o + 1] - black[o + 1];
    const db = white[o + 2] - black[o + 2];
    const a = 1 - (dr + dg + db) / (3 * 255);
    if (a <= 1 / 255) continue; // fully transparent: RGBA stays 0
    const alpha = Math.min(1, a);
    out[o] = Math.min(255, Math.round(black[o] / alpha));
    out[o + 1] = Math.min(255, Math.round(black[o + 1] / alpha));
    out[o + 2] = Math.min(255, Math.round(black[o + 2] / alpha));
    out[o + 3] = Math.round(alpha * 255);
  }
  return out;
};
