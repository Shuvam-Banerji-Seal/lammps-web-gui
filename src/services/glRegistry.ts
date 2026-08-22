import * as THREE from 'three';

/**
 * Registry for the live WebGL renderer/scene/camera so imperative actions
 * (screenshot capture, video recording) can force a render right before
 * reading pixels — removing the need for preserveDrawingBuffer, which
 * costs performance on every frame.
 */
interface ActiveGL {
  gl: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.Camera;
}

let active: ActiveGL | null = null;

export const registerActiveGL = (entry: ActiveGL | null): void => {
  active = entry;
};

export const getActiveGL = (): ActiveGL | null => active;

/**
 * Force-render the current frame and return the canvas data URL.
 * Returns null when no scene is registered or capture fails.
 */
export const captureActiveCanvas = (): string | null => {
  if (!active) return null;
  try {
    const { gl, scene, camera } = active;
    gl.render(scene, camera);
    const domElement = gl.domElement as HTMLCanvasElement;
    return domElement.toDataURL('image/png');
  } catch {
    return null;
  }
};
