import { CameraPreset } from '../types';

/**
 * Tiny typed event bus so UI chrome (keyboard shortcuts, buttons) can drive
 * the 3D camera inside the R3F Canvas without prop drilling or context.
 * Commands are consumed by <CameraRig /> inside the scene.
 */
export type CameraCommand =
  | { type: 'fit' }
  | { type: 'preset'; preset: CameraPreset }
  | { type: 'zoom'; delta: number }        // +1 zoom in, -1 zoom out
  | { type: 'orbit'; dx: number; dy: number }; // radians

type Listener = (cmd: CameraCommand) => void;

const listeners = new Set<Listener>();

export const emitCameraCommand = (cmd: CameraCommand): void => {
  listeners.forEach(fn => fn(cmd));
};

export const onCameraCommand = (fn: Listener): (() => void) => {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
};

/** Unit direction vectors for each named camera preset. */
export const CAMERA_PRESET_DIRECTIONS: Record<CameraPreset, [number, number, number]> = {
  front:  [0, 0, 1],
  back:   [0, 0, -1],
  left:   [-1, 0, 0],
  right:  [1, 0, 0],
  top:    [0, 1, 0.0001],
  bottom: [0, -1, 0.0001],
  iso:    [1, 0.8, 1],
};
