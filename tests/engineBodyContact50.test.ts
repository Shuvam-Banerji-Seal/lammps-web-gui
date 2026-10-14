import { describe, expect, it } from 'vitest';
import { bodyGeom, roundedPolygonPair, type BodyGeom, type RPParams, type V3 } from '../src/engine-gpl/body/rounded_polygon_law';
import { PairBodyRoundedPolygon } from '../src/engine-gpl/body/pair_body_rounded_polygon';
import { StyleError } from '../src/engine/force/types';

/*
 * pair_style body/rounded/polygon (src/engine-gpl/body/rounded_polygon_law.ts, ported from native src/BODY; the
 * style in pair_body_rounded_polygon.ts). Expected values are native LAMMPS measurements (black box): two rounded squares
 * (edge 4, vertices at +-2, diameter 0.5), dimension 2, `pair_style body/rounded/polygon 0 0 0 dua rc`, `pair_coeff * * kn kna`,
 * `run 0`. Forces are on body 2 (the native fx[2]), torques are tqz of body 2, energies are the total pair energy (native thermo
 * pe per atom times the 2 atoms).
 * Measured with native LAMMPS (black box): the face contact (two vertex-edge contacts), the vertex-vertex contact
 * (both squares rotated 0.785), a tilted pair with two overlapping contacts (body 2 rotated 0.05 at (4.3, 3.0)),
 * and the delta_ua scaling of the cohesion in that pair.
 */

interface Sq {
  c: [number, number];
  /** World-frame vertices (x, y). */
  v: [number, number][];
  vel?: [number, number];
  w?: number;
}
interface Pairing {
  fA: V3;
  fB: V3;
  tzA: number;
  tzB: number;
  energy: number;
  ja: number;
  contacts: { separation: number; n: V3 }[];
}

/** Bodies of the pair as geometry (identity orientation, the vertices relative to the centre). */
const geomOf = (b: Sq): BodyGeom => {
  const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const body = b.v.map(([x, y]) => [x - b.c[0], y - b.c[1], 0] as V3);
  return bodyGeom([b.c[0], b.c[1], 0], [b.vel?.[0] ?? 0, b.vel?.[1] ?? 0, 0], [0, 0, b.w ?? 0], body, I, 0.5);
};
const pairOf = (A: Sq, B: Sq, kn: number, kna: number, dua: number, rc: number, damp = { cn: 0, ct: 0, mu: 0 }): Pairing => {
  const p: RPParams = { kn, kna, dua, cut: rc, ...damp };
  const acc = roundedPolygonPair([geomOf(A), geomOf(B)], p);
  return { fA: acc.f[0], fB: acc.f[1], tzA: acc.t[0][2], tzB: acc.t[1][2], energy: acc.energy, ja: acc.ja, contacts: acc.contacts };
};
const roundedPairForce = (A: Sq, B: Sq, _diameter: number, rc: number, kn: number, kna: number, dua: number, damp = { cn: 0, ct: 0, mu: 0 }) =>
  pairOf(A, B, kn, kna, dua, rc, damp);
const contactsOf = (A: Sq, B: Sq) => pairOf(A, B, 0, 0, 0.5, 1.0).contacts;

const SQ: [number, number][] = [[-2, -2], [-2, 2], [2, 2], [2, -2]];
const square = (cx: number, cy: number, t: number): Sq => {
  const c = Math.cos(t), s = Math.sin(t);
  const v = SQ.map(([x, y]) => [cx + c * x - s * y, cy + s * x + c * y] as [number, number]);
  return { c: [cx, cy], v };
};
const DIAM = 0.5;
const rel = (a: number, b: number): number => Math.abs(a - b) / Math.max(1, Math.abs(b));

describe('body/rounded/polygon contact law (native numbers)', () => {
  it('face contact: two vertex-edge contacts, elastic force k_n ov per contact, energy k_n ov^2 / 2 each (per atom thermo value is half)', () => {
    const A = square(0, 0, 0), B = square(4.4, 3.5, 0);
    const cs = contactsOf(A, B);
    expect(cs.length).toBe(2); // two vertex-edge contacts
    expect(cs.map((c) => +c.separation.toFixed(12))).toEqual([-0.1, -0.1]);
    const r = roundedPairForce(A, B, DIAM, 1.0, 100, 0, 0.3);
    // native: pair energy 1.0 (k_n 100, ov 0.1; the thermo value 0.5 is per atom), force on body 2 +20 (2 k_n ov)
    expect(rel(r.energy, 1.0)).toBeLessThan(1e-12);
    expect(rel(r.fB[0], 20)).toBeLessThan(1e-12);
  });

  it('cohesion: 2 contacts at gap 0.1 give -3.6 force and -1.62 energy (k_na 2, rc 1, total)', () => {
    const A = square(0, 0, 0), B = square(4.6, 3.5, 0);
    const r = roundedPairForce(A, B, DIAM, 1.0, 0, 2, 0.5);
    expect(rel(r.fB[0], -3.6)).toBeLessThan(1e-12);
    expect(rel(r.energy, -1.62)).toBeLessThan(1e-12);
    expect(r.ja).toBe(1);
  });

  it('vertex-vertex: one contact with cohesion k_na (rc - g), energy -k_na (rc - g)^2 / 2 (native fx 1.9136818078184822944)', () => {
    const A = square(0, 0, 0.785), B = square(6.2, 0, 0.785);
    const cs = contactsOf(A, B);
    expect(cs.length).toBe(0); // a vertex-vertex cohesion without overlap is not a contact (no damping, no friction)
    const r = roundedPairForce(A, B, DIAM, 1.0, 0, 2, 0.5);
    expect(rel(r.fB[0], -1.9136818078184822944)).toBeLessThan(2e-4); // the 0.785 rotation is not exactly 45 degrees
    expect(rel(r.energy, -0.91556025952136887990)).toBeLessThan(2e-4);
  });

  it('tilted pair (body 2 rotated 0.05, at (4.3, 3.0)): force and tqz match native with delta_ua 0.3', () => {
    const A = square(0, 0, 0), B = square(4.3, 3.0, 0.05);
    const r = roundedPairForce(A, B, DIAM, 1.0, 100, 2, 0.3);
    expect(rel(r.fB[0], 16.656853108097614)).toBeLessThan(1e-9);
    expect(rel(r.fB[1], 0.54976972716821)).toBeLessThan(1e-9);
    // native tqz of body 2 (dump custom tqz) 21.6156580982125
    expect(rel(r.tzB, 21.6156580982125)).toBeLessThan(1e-9);
    expect(rel(r.energy, -0.88919479580120519558)).toBeLessThan(1e-9);
  });

  it('delta_ua scaling of the cohesion of two overlapping contacts: j_a = L/delta_ua (native -251.03 and -8.368)', () => {
    const A = square(0, 0, 0), B = square(4.3, 3.0, 0.05);
    const small = roundedPairForce(A, B, DIAM, 1.0, 0, 2, 0.01);
    expect(rel(small.fB[0], -251.03341128563891971)).toBeLessThan(1e-9);
    const large = roundedPairForce(A, B, DIAM, 1.0, 0, 2, 0.3);
    expect(rel(large.fB[0], -8.367780376187965885)).toBeLessThan(1e-9);
    expect(rel(large.ja, 0.5581 / 0.3)).toBeLessThan(1e-3);
  });

  it('a single overlapping contact is not scaled by delta_ua (native -2.5853 at delta_ua 0.01 and 0.3)', () => {
    const A = square(0, 0, 0), B = square(4.4, 3.0, 0.1);
    expect(rel(roundedPairForce(A, B, DIAM, 1.0, 100, 2, 0.01).fB[0], -2.585297595052200581)).toBeLessThan(1e-9);
    expect(rel(roundedPairForce(A, B, DIAM, 1.0, 100, 2, 0.3).fB[0], -2.585297595052200581)).toBeLessThan(1e-9);
  });

  it('forces of the pair sum to zero (Newton third law); settings accept c_n and c_t, accept mu and the history keyword, refuse an unknown keyword', () => {
    const A = square(0, 0, 0), B = square(4.3, 3.0, 0.05);
    const r = roundedPairForce(A, B, DIAM, 1.0, 100, 2, 0.3);
    expect(r.fA[0] + r.fB[0]).toBeCloseTo(0, 12);
    expect(r.fA[1] + r.fB[1]).toBeCloseTo(0, 12);
    const st = new PairBodyRoundedPolygon();
    expect(() => st.settings(['1', '0', '0', '0.5', '1.0'])).not.toThrow();
    expect(() => st.settings(['0', '0', '0.1', '0.5', '1.0'])).not.toThrow(); // friction (mu != 0) is ported
    expect(() => st.settings(['0', '0', '0', '0.5', '1.0', 'history'])).not.toThrow();
    expect(() => st.settings(['0', '0', '0', '0.5', '1.0', 'bogus'])).toThrow(StyleError);
  });
});

describe('body/rounded/polygon damping and friction (native numbers, body 2 tilted 0.1 at (4.4, 3.0))', () => {
  // one overlapping contact (delta_n -0.0122) and one separated one: only the overlapping one carries damping
  const A = square(0, 0, 0);
  const base = (B: Sq) => roundedPairForce(A, B, DIAM, 1.0, 100, 2, 0.3);
  const Bt = (vel: [number, number], w = 0): Sq => ({ ...square(4.4, 3.0, 0.1), vel, w });

  it('c_n: normal relative velocity, -c_n v_n (native d fx2 +0.49502 per c_n at v_x -0.5; 0.99003 at c_n 2)', () => {
    const b0 = base(Bt([0, 0]));
    const r1 = roundedPairForce(A, Bt([-0.5, 0]), DIAM, 1.0, 100, 2, 0.3, { cn: 1, ct: 0, mu: 0 });
    expect(rel(r1.fB[0] - b0.fB[0], 0.4950166)).toBeLessThan(1e-6);
    const r2 = roundedPairForce(A, Bt([-0.5, 0]), DIAM, 1.0, 100, 2, 0.3, { cn: 2, ct: 0, mu: 0 });
    expect(rel(r2.fB[0] - b0.fB[0], 0.990033)).toBeLessThan(1e-5);
    // native: a separating velocity +0.5 gives -0.49502 (antisymmetric)
    const r3 = roundedPairForce(A, Bt([0.5, 0]), DIAM, 1.0, 100, 2, 0.3, { cn: 1, ct: 0, mu: 0 });
    expect(rel(r3.fB[0] - b0.fB[0], -0.4950166)).toBeLessThan(1e-6);
  });

  it('c_t: tangential relative velocity, -c_t v_t (native dF (-0.049667, 0.495017), dtqz -1.116356 at v_y -0.5)', () => {
    const b0 = base(Bt([0, 0]));
    const r = roundedPairForce(A, Bt([0, -0.5]), DIAM, 1.0, 100, 2, 0.3, { cn: 0, ct: 1, mu: 0 });
    expect(rel(r.fB[0] - b0.fB[0], -0.049667)).toBeLessThan(1e-5);
    expect(rel(r.fB[1] - b0.fB[1], 0.495017)).toBeLessThan(1e-5);
    expect(rel(r.tzB - b0.tzB, -1.116356)).toBeLessThan(1e-5);
  });

  it('friction: min(mu k_n |delta_n|, c_t |v_t|) opposing v_t, from the elastic force (native -0.012156 at mu 0.01; -0.497498 at mu 10)', () => {
    const b0 = roundedPairForce(A, Bt([0, -0.5]), DIAM, 1.0, 100, 2, 0.3, { cn: 0, ct: 1, mu: 0 });
    const n = [-0.995004165278, -0.099833416647];
    const t = [-n[1], n[0]];
    const along = (r: { fB: V3 }): number => (r.fB[0] - b0.fB[0]) * t[0] + (r.fB[1] - b0.fB[1]) * t[1];
    const r1 = roundedPairForce(A, Bt([0, -0.5]), DIAM, 1.0, 100, 2, 0.3, { cn: 0, ct: 1, mu: 0.01 });
    expect(rel(along(r1), -0.012156)).toBeLessThan(1e-4);
    // the limit is independent of the cohesion: k_na 0 gives the same friction
    const r1b = roundedPairForce(A, Bt([0, -0.5]), DIAM, 1.0, 100, 0, 0.3, { cn: 0, ct: 1, mu: 0.01 });
    const b0b = roundedPairForce(A, Bt([0, -0.5]), DIAM, 1.0, 100, 0, 0.3, { cn: 0, ct: 1, mu: 0 });
    expect(rel((r1b.fB[0] - b0b.fB[0]) * t[0] + (r1b.fB[1] - b0b.fB[1]) * t[1], -0.012156)).toBeLessThan(1e-4);
    // sliding limit: c_t |v_t| = 0.497498 when mu k_n |delta| is larger
    const r10 = roundedPairForce(A, Bt([0, -0.5]), DIAM, 1.0, 100, 2, 0.3, { cn: 0, ct: 1, mu: 10 });
    expect(rel(along(r10), -0.497498)).toBeLessThan(1e-4);
  });

  it('spin: contact velocity v + w x r at the contact midpoint (native dF (-0.365411, 0.807976), dtqz -2.099558 at L 1, I 2.67)', () => {
    const b0 = base(Bt([0, 0]));
    const r = roundedPairForce(A, Bt([0, 0], 1 / 2.67), DIAM, 1.0, 100, 2, 0.3, { cn: 1, ct: 1, mu: 0 });
    expect(rel(r.fB[0] - b0.fB[0], -0.365411)).toBeLessThan(1e-5);
    expect(rel(r.fB[1] - b0.fB[1], 0.807976)).toBeLessThan(1e-5);
    expect(rel(r.tzB - b0.tzB, -2.099558)).toBeLessThan(1e-5);
  });
});
