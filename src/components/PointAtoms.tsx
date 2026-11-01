import React, { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { Atom, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';
import { atomDisplayRadius, radiusKeyOf } from '../services/atomStyle';
import { colorKeyOf, makeAtomColorer } from '../services/colorMaps';

/*
 * The 'points', 'lines' and 'flat' representations: every atom one point of a single THREE.Points draw call. 'points'
 * and 'lines' draw a shaded sphere impostor (the fragment shader builds the sphere normal from the point coordinate and
 * lights it from a fixed view-space direction), much cheaper than tessellated spheres, for systems of 10^5 atoms and
 * more. 'flat' draws an unlit disc with a darker rim, which keeps touching particles apart (2D simulations, 2D views).
 * The point size is the atom's on-screen diameter, capped by the GPU's largest point size.
 */
const VERTEX = `
attribute float radius;
uniform float scale;
varying vec3 vColor;
#include <fog_pars_vertex>
void main() {
  vColor = color;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  gl_PointSize = 2.0 * radius * scale / -mvPosition.z;
  #include <fog_vertex>
}`;

const FRAGMENT = `
varying vec3 vColor;
#include <fog_pars_fragment>
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  vec3 n = vec3(p.x, -p.y, sqrt(1.0 - r2));
  vec3 l = normalize(vec3(0.4, 0.6, 0.7));
  float diffuse = max(dot(n, l), 0.0);
  float specular = pow(max(dot(n, normalize(l + vec3(0.0, 0.0, 1.0))), 0.0), 48.0);
  gl_FragColor = vec4(vColor * (0.3 + 0.7 * diffuse) + 0.25 * specular, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
}`;

/** A flat disc: the atom's colour, darkened over the outer rim (r^2 from 0.72 to 0.86 of the disc). */
const FRAGMENT_FLAT = `
varying vec3 vColor;
#include <fog_pars_fragment>
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float rim = smoothstep(0.72, 0.86, r2);
  gl_FragColor = vec4(mix(vColor, vColor * 0.45, rim), 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
}`;

const PointAtoms: React.FC<{ atoms: Atom[]; config: VisualizationConfig; flat?: boolean }> = ({ atoms, config, flat = false }) => {
  const invalidate = useThree((s) => s.invalidate);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * atoms.length), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(3 * atoms.length), 3));
    g.setAttribute('radius', new THREE.BufferAttribute(new Float32Array(atoms.length), 1));
    return g;
  }, [atoms.length]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  const material = useMemo(
    () => new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: flat ? FRAGMENT_FLAT : FRAGMENT,
      vertexColors: true,
      fog: true,
      uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog, { scale: { value: 1 } }]),
    }),
    [flat],
  );
  useEffect(() => () => material.dispose(), [material]);

  const radiusKey = radiusKeyOf(config);
  useEffect(() => {
    const pos = geometry.getAttribute('position') as THREE.BufferAttribute;
    const rad = geometry.getAttribute('radius') as THREE.BufferAttribute;
    const p = pos.array as Float32Array, r = rad.array as Float32Array;
    for (let i = 0; i < atoms.length; i++) {
      const a = atoms[i];
      p[3 * i] = a.x; p[3 * i + 1] = a.y; p[3 * i + 2] = a.z;
      r[i] = atomDisplayRadius(a, config);
    }
    pos.needsUpdate = true;
    rad.needsUpdate = true;
    geometry.computeBoundingSphere();
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, geometry, radiusKey]);

  const colorKey = colorKeyOf(config);
  useEffect(() => {
    const col = geometry.getAttribute('color') as THREE.BufferAttribute;
    const c = col.array as Float32Array;
    const colorOf = makeAtomColorer(config, DEFAULT_ATOM_COLOR);
    const rgb = new Float32Array(3);
    const lin = new THREE.Color();
    for (let i = 0; i < atoms.length; i++) {
      colorOf(atoms[i], rgb);
      lin.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
      c[3 * i] = lin.r; c[3 * i + 1] = lin.g; c[3 * i + 2] = lin.b;
    }
    col.needsUpdate = true;
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, geometry, colorKey]);

  // the point size of a sphere of radius r at depth z is 2 r scale / z pixels, scale = drawing-buffer height / (2 tan(fov/2))
  const scale = useRef(material.uniforms.scale);
  scale.current = material.uniforms.scale;
  useFrame(({ camera, size, viewport }) => {
    const fov = (camera as THREE.PerspectiveCamera).fov ?? 40;
    scale.current.value = (size.height * viewport.dpr) / (2 * Math.tan((fov * Math.PI) / 360));
  });

  return atoms.length > 0 ? <points geometry={geometry} material={material} /> : null;
};

export default PointAtoms;
