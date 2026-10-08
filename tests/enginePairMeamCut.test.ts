import { describe, expect, it } from 'vitest';
import { DEFAULT_MEAM_OPTIONS, meamEnergy, meamEnergyForces, type MeamElement } from '../src/engine/force/pair/meam';

/*
 * MEAM smoothing window [rc - delr, rc] and partially screened clusters (meamcut scope). The synthetic bcc
 * entry is tests/oracle/w16meamcut_bcc_lib.meam + w16meamcut_bcc.meam (same numbers as w15meam_bcc). Every
 * energy below was measured with native LAMMPS as a black box on non-periodic clusters (box 200 A) or on the
 * periodic diamond cell of enginePairMeam15; they are numbers, not LAMMPS text.
 *
 * Model verified here (see the notes in meam.ts):
 *  - the radial weight fc(r) = [1-(1-x)^4]^2, x = (rc - r)/delr multiplies each partial density term and the
 *    pair term of a bond inside the window;
 *  - the reference structure used for the pair term carries no fc;
 *  - the screening factor S uses the C function only (no fc of the screening atom), with the same pre-filter.
 */

const W_BCC: MeamElement = { z: 8, re: 2.73664, alpha: 5.0, Ec: 8.9, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0], ibar: 0, lat: 'bcc' };
const W_DIA: MeamElement = { ...W_BCC, z: 4, re: 2.165064, Ec: 4.6, lat: 'dia' };
const OPTS = { ...DEFAULT_MEAM_OPTIONS };
const BOX: [number, number, number] = [1e4, 1e4, 1e4];

const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

describe('MEAM smoothing window: dimers inside [rc - delr, rc]', () => {
  // Measured with native LAMMPS (black box): dimer pe along x for bcc W (dimers at 3.9 ... 3.999 A).
  const native: Array<[number, number]> = [
    [3.9, -4.65069307426565],
    [3.92, -4.5907421718859],
    [3.95, -4.21873921729463],
    [3.96, -3.85736275330339],
    [3.98, -2.32815752424894],
    [3.99, -1.05457803891509],
    [3.999, -0.0278618695728573],
  ];
  for (const [r, pe] of native) {
    it(`dimer at ${r} A matches native pe to 1e-10 relative`, () => {
      const E = meamEnergy(W_BCC, OPTS, Float64Array.from([0, 0, 0, r, 0, 0]), BOX);
      expect(rel(E, pe)).toBeLessThan(1e-10);
    });
  }

  it('dimer forces inside the window equal -grad E', () => {
    const x = Float64Array.from([0, 0, 0, 3.95, 0.1, 0]);
    const { E, F } = meamEnergyForces(W_BCC, OPTS, x, BOX);
    const h = 1e-6;
    for (const q of [3, 4]) {
      const xp = Float64Array.from(x), xm = Float64Array.from(x);
      xp[q] += h;
      xm[q] -= h;
      const fd = -(meamEnergy(W_BCC, OPTS, xp, BOX) - meamEnergy(W_BCC, OPTS, xm, BOX)) / (2 * h);
      expect(Math.abs(F[q] - fd)).toBeLessThan(1e-6 * Math.max(1, Math.abs(fd)));
    }
    expect(E).toBeCloseTo(meamEnergy(W_BCC, OPTS, x, BOX), 12);
  });
});

describe('MEAM smoothing window: partially screened clusters with a neighbour inside the window', () => {
  // Measured with native LAMMPS (black box): the screening atom 2 at 3.918 A lies inside the window; atom 1 at 3.8 A.
  const cases: Array<[number[], number]> = [
    [[0, 0, 0, 3.8, 0, 0, 1.95, 3.4, 0], -9.14206625906409],
    [[0, 0, 0, 3.8, 0, 0, 1.95, 3.4, 0.3], -9.08734716841828],
    [[0, 0, 0, 3.85, 0, 0, 1.9, 3.42, 0.2], -8.86808803339698],
    [[0, 0, 0, 3.8, 0, 0, 1.9, 3.5, 0], -6.74256212707885],
    [[0, 0, 0, 3.6, 0, 0, 1.8, 3.6, 0], -5.52071413702904],
    [[0, 0, 0, 3.7, 0, 0, 1.85, 3.55, 0.4], -5.2126427823761],
    [[0, 0, 0, 3.7, 0, 0, 1.85, 3.55, 0.4, 0.2, 1.3, 1.5], -12.3842905845873],
  ];
  for (const [pos, pe] of cases) {
    it(`cluster of ${pos.length / 3} atoms matches native pe to 1e-9 relative`, () => {
      const E = meamEnergy(W_BCC, OPTS, Float64Array.from(pos), BOX);
      expect(rel(E, pe)).toBeLessThan(1e-9);
    });
  }
});

describe('MEAM periodic distorted diamond cell: helper against native', () => {
  it('1-cell diamond (8 atoms, L = 5 A, 0.06 A distortion) matches native pe to 1e-9 relative', () => {
    // Measured with native LAMMPS (black box, periodic p p p): pe -36.6157506731495 eV (press 5118.16596 bar).
    const x = Float64Array.from([4.968932738527656, 4.994925902681425, 0.026488956045358947, 2.457889558551833, 2.4457766466215247, 4.981045851120725, 2.553664881978184, 0.006002893438562751, 2.454528980404138, 4.979435355486348, 2.523419070597738, 2.5168148566316813, 1.227563006095588, 1.2210494157019998, 1.3069995287247007, 3.7588786383625123, 3.773848517015576, 1.2511135089863092, 3.7218735930137328, 1.2107393417041745, 3.7510782984271636, 1.2930176328215746, 3.7036004885472362, 3.7115272442903375]);
    const E = meamEnergy(W_DIA, OPTS, x, [5, 5, 5]);
    expect(rel(E, -36.6157506731495)).toBeLessThan(1e-9);
    expect(meamEnergyForces(W_DIA, OPTS, x, [5, 5, 5]).E).toBeCloseTo(E, 12);
  });
});
