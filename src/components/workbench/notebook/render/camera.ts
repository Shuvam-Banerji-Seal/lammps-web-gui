/**
 * Camera paths over a clip. Every pose is a pure function of the output frame index k of K frames, so a render
 * is repeatable and no frame depends on the frames before it. The path turns the camera about the vertical (y)
 * axis through the scene's origin, and/or scales its distance from that origin.
 */

export type CameraMode = 'static' | 'orbit' | 'zoom';

export const CAMERA_LABEL: Record<CameraMode, string> = {
  static: 'Static (current view)',
  orbit: 'Orbit around the vertical axis',
  zoom: 'Slow zoom in and out',
};

/** Fraction of the clip (0 at the first frame, 1 at the last) for output frame k of count. */
export const clipFraction = (k: number, count: number): number =>
  count <= 1 ? 0 : Math.min(1, Math.max(0, k / (count - 1)));

/** Turn angle (radians about +y) and distance scale for output frame k of count. */
export const cameraFrame = (
  mode: CameraMode,
  k: number,
  count: number,
  orbitDeg: number,
): { angle: number; scale: number } => {
  const t = clipFraction(k, count);
  if (mode === 'orbit') {
    const total = Number.isFinite(orbitDeg) ? (orbitDeg * Math.PI) / 180 : 0;
    return { angle: total * t, scale: 1 };
  }
  if (mode === 'zoom') {
    // dips to 75% of the start distance half way through and returns: in, then out, ending where it began
    return { angle: 0, scale: 1 - 0.25 * Math.sin(Math.PI * t) };
  }
  return { angle: 0, scale: 1 };
};

/**
 * The camera position for a turn and a scale: the start position rotated about the y axis, then scaled.
 * The height (y) is kept, so a turn keeps the view's elevation.
 */
export const orbitPose = (
  start: readonly [number, number, number],
  angle: number,
  scale: number,
): [number, number, number] => {
  const [x, y, z] = start;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [(x * c + z * s) * scale, y * scale, (-x * s + z * c) * scale];
};

/** Degrees-per-clip input: a finite number between 0 and 3600, else the default 360. */
export const reviveOrbitDegrees = (raw: string): number => {
  const n = Number(raw);
  return raw.trim() !== '' && Number.isFinite(n) && n >= 0 && n <= 3600 ? n : 360;
};
