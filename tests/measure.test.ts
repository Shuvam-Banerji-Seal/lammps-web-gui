import { describe, it, expect } from 'vitest';
import { distance, angleDeg, dihedralDeg, measureSelection } from '../services/measure';
import { Atom } from '../types';

const at = (id: number, x: number, y: number, z: number): Atom =>
  ({ id, molId: 1, type: 1, charge: 0, x, y, z });

describe('measurement math', () => {
  it('computes distances', () => {
    expect(distance(at(1, 0, 0, 0), at(2, 3, 4, 0))).toBeCloseTo(5, 10);
    // real H2 bond length
    expect(distance(at(1, 0, 0, 0), at(2, 0.74, 0, 0))).toBeCloseTo(0.74, 10);
  });

  it('computes the water H-O-H angle as ~104.5°', () => {
    const o = at(1, 0, 0, 0);
    const h1 = at(2, 0.7575, 0.5865, 0);
    const h2 = at(3, -0.7575, 0.5865, 0);
    expect(angleDeg(h1, o, h2)).toBeCloseTo(104.5, 1);
  });

  it('angle degenerates safely on zero-length arms', () => {
    const a = at(1, 0, 0, 0);
    expect(angleDeg(a, a, a)).toBe(0);
  });

  it('dihedral: exact tetrahedral butane — anti 180°, eclipsed 0°, gauche +60°', () => {
    // Idealized backbone: b→c along +x (1.54 Å), CCC angle exactly 112°.
    const S = 1.54;
    const cos112 = Math.cos((112 * Math.PI) / 180); // −0.3746
    const sin112 = Math.sin((112 * Math.PI) / 180); // +0.9272

    const b = at(2, 0, 0, 0);
    const c = at(3, S, 0, 0);
    const a = at(1, S * cos112, S * sin112, 0);          // at vertex b
    const dAnti = at(4, S + S * (-cos112), -S * sin112, 0);  // planar trans
    const dCis = at(5, S + S * (-cos112), S * sin112, 0);    // planar eclipsed

    const anti = dihedralDeg(a, b, c, dAnti);
    expect(Math.abs(Math.abs(anti) - 180)).toBeLessThan(1e-6);

    const cis = dihedralDeg(a, b, c, dCis);
    expect(Math.abs(cis)).toBeLessThan(1e-6);

    // Gauche: rotate the eclipsed substituent +60° about the b→c axis (+x).
    const phi = (60 * Math.PI) / 180;
    const px = -cos112;
    const py = sin112 * Math.cos(phi);
    const pz = sin112 * Math.sin(phi);
    const dGauche = at(6, S + S * px, S * py, S * pz);
    expect(dihedralDeg(a, b, c, dGauche)).toBeCloseTo(60, 6);
  });

  it('dihedral sign follows the canonical (praxeolitic/RDKit) convention', () => {
    // Hand-computed reference: axis along +x, v=+z, w=(y+z)/√2 → −45°.
    const a = at(1, 0, 0, 1);
    const b = at(2, 0, 0, 0);
    const c = at(3, 1, 0, 0);
    const d = at(4, 1, 1, 1);
    expect(dihedralDeg(a, b, c, d)).toBeCloseTo(-45, 6);
  });

  it('measureSelection dispatches by count and rejects others', () => {
    const pts = [at(1, 0, 0, 0), at(2, 1, 0, 0), at(3, 1, 1, 0), at(4, 1, 1, 1)];
    expect(measureSelection(pts.slice(0, 2))!.kind).toBe('distance');
    expect(measureSelection(pts.slice(0, 3))!.kind).toBe('angle');
    expect(measureSelection(pts)!.kind).toBe('dihedral');
    expect(measureSelection([pts[0]])).toBeNull();
    expect(measureSelection([])).toBeNull();
    expect(measureSelection(pts.slice(0, 2))!.label).toMatch(/Å$/);
  });
});
