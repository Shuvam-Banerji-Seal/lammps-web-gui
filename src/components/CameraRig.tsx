import React, { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { onCameraCommand, CAMERA_PRESET_DIRECTIONS } from '../services/cameraBus';

interface CameraRigProps {
  /** Radius of the scene bounding sphere (world units) for fit/preset framing. */
  boundingRadius: number;
  autoRotate: boolean;
  autoRotateSpeed: number;
  fov: number;
}

/**
 * Lives inside the Canvas; translates bus commands into camera moves.
 * All framing math assumes the molecule group is centered at the origin.
 */
const CameraRig: React.FC<CameraRigProps> = ({
  boundingRadius,
  autoRotate,
  autoRotateSpeed,
  fov,
}) => {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const camera = useThree(s => s.camera);
  const invalidate = useThree(s => s.invalidate);

  const distanceFor = React.useCallback(
    (direction: THREE.Vector3) => {
      // frame the bounding sphere with margin; fov-aware
      const fovRad = (fov * Math.PI) / 180;
      const dist = boundingRadius / Math.sin(Math.min(fovRad, Math.PI / 3) / 2);
      return Math.max(dist * 1.05, boundingRadius + 1);
    },
    [boundingRadius, fov]
  );

  useEffect(() => {
    const moveCameraTo = (direction: THREE.Vector3, dist: number) => {
      const target = new THREE.Vector3(0, 0, 0);
      const pos = direction.clone().normalize().multiplyScalar(dist);
      camera.position.copy(pos);
      camera.lookAt(target);
      if (controlsRef.current) {
        controlsRef.current.target.copy(target);
        controlsRef.current.update();
      }
      invalidate();
    };

    const off = onCameraCommand(cmd => {
      switch (cmd.type) {
        case 'fit':
        case 'preset': {
          const dir =
            cmd.type === 'fit'
              ? new THREE.Vector3(...CAMERA_PRESET_DIRECTIONS.iso)
              : new THREE.Vector3(...CAMERA_PRESET_DIRECTIONS[cmd.preset]);
          moveCameraTo(dir, distanceFor(dir));
          break;
        }
        case 'zoom': {
          if (!controlsRef.current) return;
          const offset = camera.position.clone().sub(controlsRef.current.target);
          const dolly = cmd.delta > 0 ? 0.82 : 1.22;
          const newLen = THREE.MathUtils.clamp(
            offset.length() * dolly,
            boundingRadius * 0.05,
            boundingRadius * 20
          );
          camera.position.copy(controlsRef.current.target).add(offset.setLength(newLen));
          controlsRef.current.update();
          invalidate();
          break;
        }
        case 'orbit': {
          if (!controlsRef.current) return;
          // Rotate around the target via spherical coords (no private APIs)
          const controls = controlsRef.current;
          const offset = camera.position.clone().sub(controls.target);
          const spherical = new THREE.Spherical().setFromVector3(offset);
          spherical.theta -= cmd.dx;
          spherical.phi = THREE.MathUtils.clamp(
            spherical.phi - cmd.dy,
            0.02,
            Math.PI - 0.02
          );
          const rotated = new THREE.Vector3().setFromSpherical(spherical);
          camera.position.copy(controls.target).add(rotated);
          controls.update();
          invalidate();
          break;
        }
      }
    });
    return off;
  }, [camera, boundingRadius, distanceFor, invalidate]);

  // Re-frame when the molecule changes size drastically
  useEffect(() => {
    const dir = camera.position.clone();
    if (dir.lengthSq() < 1e-6) dir.set(1, 0.8, 1);
    const dist = distanceFor(dir.normalize());
    camera.position.copy(dir.multiplyScalar(dist));
    camera.updateProjectionMatrix();
    invalidate();
  }, [boundingRadius]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (camera instanceof THREE.PerspectiveCamera && camera.fov !== fov) {
      camera.fov = fov;
      camera.updateProjectionMatrix();
      invalidate();
    }
  }, [fov, camera, invalidate]);

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      autoRotate={autoRotate}
      autoRotateSpeed={autoRotateSpeed}
      onChange={() => invalidate()}
    />
  );
};

export default CameraRig;
