import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { writeInstanceTransform, writeInstanceSegment } from '../src/services/instanceMatrix';

const lcg = (seed: number) => () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

/**
 * The direct writes replace Object3D.updateMatrix() in the instancing hot
 * loop. They are only safe if they are byte-for-byte what three.js would have
 * produced, so these tests diff them against three.js for random inputs.
 */
describe('writeInstanceTransform', () => {
  it('matches THREE.Object3D for a translation + uniform scale', () => {
    const rnd = lcg(19);
    const out = new Float32Array(16 * 40);
    const dummy = new THREE.Object3D();
    const ref = new Float32Array(16 * 40);

    for (let i = 0; i < 40; i++) {
      const x = (rnd() - 0.5) * 200;
      const y = (rnd() - 0.5) * 200;
      const z = (rnd() - 0.5) * 200;
      const s = rnd() * 5 + 0.01;
      writeInstanceTransform(out, i, x, y, z, s);
      dummy.position.set(x, y, z);
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      dummy.matrix.toArray(ref, i * 16);
    }
    expect(Array.from(out)).toEqual(Array.from(ref));
  });

  it('writes only the requested instance slot', () => {
    const out = new Float32Array(32).fill(-1);
    writeInstanceTransform(out, 1, 1, 2, 3, 4);
    expect(Array.from(out.slice(0, 16))).toEqual(new Array(16).fill(-1));
    expect(out[16]).toBe(4);
    expect(out[28]).toBe(1);
    expect(out[31]).toBe(1);
  });

  it('round-trips through InstancedMesh.getMatrixAt', () => {
    const mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 4, 3), undefined, 3);
    writeInstanceTransform(mesh.instanceMatrix.array as Float32Array, 2, 7, -8, 9, 1.5);
    const m = new THREE.Matrix4();
    mesh.getMatrixAt(2, m);
    const pos = new THREE.Vector3();
    const scale = new THREE.Vector3();
    m.decompose(pos, new THREE.Quaternion(), scale);
    expect(pos.toArray()).toEqual([7, -8, 9]);
    expect(scale.x).toBeCloseTo(1.5, 6);
    expect(scale.y).toBeCloseTo(1.5, 6);
    expect(scale.z).toBeCloseTo(1.5, 6);
  });
});

describe('writeInstanceSegment', () => {
  const basis = new Float32Array(9);

  it('places the instance at the midpoint with the right length', () => {
    const out = new Float32Array(16);
    writeInstanceSegment(out, 0, 0, 0, 0, 0, 10, 0, 0.5, basis);
    const m = new THREE.Matrix4().fromArray(out);
    const pos = new THREE.Vector3();
    const scale = new THREE.Vector3();
    m.decompose(pos, new THREE.Quaternion(), scale);
    expect(pos.toArray()).toEqual([0, 5, 0]);
    expect(scale.y).toBeCloseTo(10, 5);   // unit cylinder along +Y
    expect(scale.x).toBeCloseTo(0.5, 5);
    expect(scale.z).toBeCloseTo(0.5, 5);
  });

  it('maps the cylinder +Y axis onto the segment direction', () => {
    const rnd = lcg(71);
    const out = new Float32Array(16);
    for (let k = 0; k < 25; k++) {
      const p1 = new THREE.Vector3((rnd() - 0.5) * 20, (rnd() - 0.5) * 20, (rnd() - 0.5) * 20);
      const p2 = new THREE.Vector3((rnd() - 0.5) * 20, (rnd() - 0.5) * 20, (rnd() - 0.5) * 20);
      writeInstanceSegment(out, 0, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, 0.3, basis);
      const m = new THREE.Matrix4().fromArray(out);
      // +Y through the instance transform must land on the segment direction
      const mapped = new THREE.Vector3(0, 1, 0).applyMatrix4(
        new THREE.Matrix4().extractRotation(m),
      );
      const want = p2.clone().sub(p1).normalize();
      expect(mapped.dot(want)).toBeCloseTo(1, 5);
      // and the transform must not mirror (a negative determinant flips
      // normals and lights the bond from the wrong side)
      expect(m.determinant()).toBeGreaterThan(0);
    }
  });

  it('handles a near-vertical segment without a degenerate basis', () => {
    const out = new Float32Array(16);
    writeInstanceSegment(out, 0, 0, 0, 0, 0, 0, 5, 0.2, basis);
    const m = new THREE.Matrix4().fromArray(out);
    expect(Number.isFinite(m.determinant())).toBe(true);
    expect(m.determinant()).toBeGreaterThan(0);
  });

  it('collapses a zero-length segment instead of emitting NaNs', () => {
    const out = new Float32Array(16).fill(9);
    writeInstanceSegment(out, 0, 1, 1, 1, 1, 1, 1, 0.3, basis);
    expect(Array.from(out).every(Number.isFinite)).toBe(true);
    expect(Array.from(out.slice(0, 15))).toEqual(new Array(15).fill(0));
    expect(out[15]).toBe(1);
  });
});
