import { describe, expect, it } from 'vitest';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { Rng } from '../src/engine/rng';
import { latticePoints, makeLattice } from '../src/engine/lattice';
import { newPairTable, pairArrays, resolvePairs, setPairCoeff, typeRange, getPair } from '../src/engine/pairs';
import { CpuForceBackend, allImages } from '../src/engine/cpu/forces';
import { FixEnforce2d, FixNve, wrapPositions } from '../src/engine/integrate';
import { addAtoms, emptyState, run } from '../src/engine/md';
import { createVelocities, zeroAngularMomentum } from '../src/engine/velocity';
import { kineticEnergy, temperature, thermoRow } from '../src/engine/observables';
import type { SimState, ThermoRow } from '../src/engine/types';

const lj = UNIT_SYSTEMS.lj;

/** The documented examples/melt setup, built directly on the engine API. */
const meltState = (cells = 10): SimState => {
  const lat = makeLattice('fcc', 0.8442, lj, 3);
  const hi: [number, number, number] = [cells * lat.spacing[0], cells * lat.spacing[1], cells * lat.spacing[2]];
  const s = emptyState(lj, 3, { lo: [0, 0, 0], hi, periodic: [true, true, true] }, 1);
  addAtoms(s, latticePoints(lat, [0, 0, 0], hi, 3), 1);
  s.massByType[1] = 1.0;
  return s;
};

const meltPairs = () => {
  const t = newPairTable(1, 2.5);
  setPairCoeff(t, '1', '1', 1.0, 1.0, 2.5);
  expect(resolvePairs(t)).toEqual([]);
  return t;
};

/** Uniform random positions in a cubic periodic box, no two closer than dmin. */
const randomGas = (n: number, L: number, seed: number, dmin = 0.85): SimState => {
  const s = emptyState(lj, 3, { lo: [0, 0, 0], hi: [L, L, L], periodic: [true, true, true] }, 2);
  const rng = new Rng(seed);
  const pts: number[] = [];
  while (pts.length < 3 * n) {
    const p = [rng.uniform() * L, rng.uniform() * L, rng.uniform() * L];
    let ok = true;
    for (let j = 0; j < pts.length && ok; j += 3) {
      let r2 = 0;
      for (let d = 0; d < 3; d++) {
        let dx = p[d] - pts[j + d];
        dx -= L * Math.round(dx / L);
        r2 += dx * dx;
      }
      ok = r2 > dmin * dmin;
    }
    if (ok) pts.push(...p);
  }
  const half = Math.floor(n / 2);
  addAtoms(s, Float64Array.from(pts.slice(0, 3 * half)), 1);
  addAtoms(s, Float64Array.from(pts.slice(3 * half)), 2);
  s.massByType[1] = 1;
  s.massByType[2] = 2;
  return s;
};

describe('units (derived from exact SI constants)', () => {
  it('matches the known conversion factors', () => {
    expect(UNIT_SYSTEMS.metal.mvv2e).toBeCloseTo(1.0364269e-4, 10);
    expect(UNIT_SYSTEMS.metal.boltz).toBeCloseTo(8.617333e-5, 10);
    expect(UNIT_SYSTEMS.metal.nktv2p).toBeCloseTo(1.602176634e6, 3);
    expect(UNIT_SYSTEMS.real.boltz).toBeCloseTo(0.0019872, 7);
    expect(UNIT_SYSTEMS.real.mvv2e).toBeCloseTo(2390.0574, 3);
    expect(UNIT_SYSTEMS.real.nktv2p / 68568.4).toBeCloseTo(1, 5);
    for (const u of Object.values(UNIT_SYSTEMS)) expect(u.mvv2e * u.ftm2v).toBeCloseTo(1, 12);
  });
});

describe('rng', () => {
  it('is deterministic per seed with sane moments', () => {
    const a = new Rng(87287);
    const b = new Rng(87287);
    const c = new Rng(87288);
    const sa = Array.from({ length: 5 }, () => a.nextU32());
    expect(sa).toEqual(Array.from({ length: 5 }, () => b.nextU32()));
    expect(sa).not.toEqual(Array.from({ length: 5 }, () => c.nextU32()));
    const r = new Rng(1);
    let m = 0, m2 = 0, g = 0, g2 = 0;
    const N = 200_000;
    for (let i = 0; i < N; i++) {
      const u = r.uniform(); m += u; m2 += u * u;
      const z = r.gaussian(); g += z; g2 += z * z;
    }
    expect(m / N).toBeCloseTo(0.5, 2);
    expect(m2 / N - (m / N) ** 2).toBeCloseTo(1 / 12, 3);
    expect(g / N).toBeCloseTo(0, 2);
    expect(g2 / N).toBeCloseTo(1, 2);
  });
});

describe('lattice', () => {
  it('fcc at rho* 0.8442 in lj units: a = (4/0.8442)^(1/3), 4 atoms per cell', () => {
    const lat = makeLattice('fcc', 0.8442, lj, 3);
    expect(lat.spacing[0]).toBeCloseTo(Math.cbrt(4 / 0.8442), 12);
    expect(meltState().n).toBe(4000);
  });

  it('counts basis atoms per cell for every 3d style and rejects a dimension mismatch', () => {
    const counts = { sc: 1, bcc: 2, fcc: 4, hcp: 4, diamond: 8 } as const;
    for (const [style, per] of Object.entries(counts)) {
      const lat = makeLattice(style as keyof typeof counts, 3.0, UNIT_SYSTEMS.metal, 3);
      const hi = lat.spacing.map((a) => 3 * a);
      expect(latticePoints(lat, [0, 0, 0], hi, 3).length / 3).toBe(27 * per);
    }
    expect(() => makeLattice('hex', 1, lj, 3)).toThrow(/2d/);
    expect(() => makeLattice('fcc', 1, lj, 2)).toThrow(/3d/);
  });

  it('lj density is the requested reduced density for every style', () => {
    for (const style of ['sc', 'bcc', 'fcc', 'hcp', 'diamond'] as const) {
      const lat = makeLattice(style, 0.7, lj, 3);
      const hi = lat.spacing.map((a) => 4 * a);
      const n = latticePoints(lat, [0, 0, 0], hi, 3).length / 3;
      expect(n / (hi[0] * hi[1] * hi[2])).toBeCloseTo(0.7, 10);
    }
    for (const style of ['sq', 'sq2', 'hex'] as const) {
      const lat = makeLattice(style, 0.7, lj, 2);
      const hi = [5 * lat.spacing[0], 5 * lat.spacing[1], 0.5];
      const pts = latticePoints(lat, [0, 0, -0.5], hi, 2);
      expect(pts.length / 3 / (hi[0] * hi[1])).toBeCloseTo(0.7, 10);
      for (let k = 2; k < pts.length; k += 3) expect(pts[k]).toBe(0);
    }
  });

  it('hcp nearest neighbours are all at the lattice constant (ideal c/a)', () => {
    const lat = makeLattice('hcp', 1.0, UNIT_SYSTEMS.metal, 3);
    const hi = lat.spacing.map((a) => 3 * a);
    const p = latticePoints(lat, [0, 0, 0], hi, 3);
    let near = 0;
    for (let j = 3; j < p.length; j += 3) {
      let r2 = 0;
      for (let d = 0; d < 3; d++) {
        let dx = p[d] - p[j + d];
        dx -= hi[d] * Math.round(dx / hi[d]);
        r2 += dx * dx;
      }
      const r = Math.sqrt(r2);
      expect(r).toBeGreaterThan(1 - 1e-9);
      if (r < 1 + 1e-9) near++;
    }
    expect(near).toBe(12);
  });
});

describe('pair coefficients', () => {
  it('parses wildcards with I <= J only', () => {
    expect(typeRange('*', 4)).toEqual([1, 4]);
    expect(typeRange('2*', 4)).toEqual([2, 4]);
    expect(typeRange('*3', 4)).toEqual([1, 3]);
    expect(typeRange('2*3', 4)).toEqual([2, 3]);
    expect(() => typeRange('5', 4)).toThrow();
    const t = newPairTable(3, 2.5);
    expect(setPairCoeff(t, '*', '*', 1, 1)).toBe(6);
  });

  it('mixes geometric by default, cutoff like sigma', () => {
    const t = newPairTable(2, 3.0);
    setPairCoeff(t, '1', '1', 1.0, 1.0, 2.5);
    setPairCoeff(t, '2', '2', 4.0, 2.0, 5.0);
    expect(resolvePairs(t)).toEqual([]);
    const p = getPair(t, 1, 2)!;
    expect(p.epsilon).toBeCloseTo(2, 12);
    expect(p.sigma).toBeCloseTo(Math.SQRT2, 12);
    expect(p.cutoff).toBeCloseTo(Math.sqrt(12.5), 12);
    expect(getPair(t, 2, 1)).toBe(p);
    const u = newPairTable(2, 3.0);
    setPairCoeff(u, '1', '1', 1, 1);
    expect(resolvePairs(u)).toEqual(['2 2', '1 2']);
  });
});

describe('cpu forces', () => {
  it('two atoms: analytic LJ energy, force and virial', () => {
    const s = emptyState(lj, 3, { lo: [0, 0, 0], hi: [20, 20, 20], periodic: [true, true, true] }, 1);
    addAtoms(s, Float64Array.from([5, 5, 5, 6.5, 5, 5]), 1);
    s.massByType[1] = 1;
    const t = newPairTable(1, 2.5);
    setPairCoeff(t, '1', '1', 1.0, 1.0);
    resolvePairs(t);
    const res = new CpuForceBackend().compute(s, t);
    const r = 1.5;
    expect(res.pe).toBeCloseTo(4 * (r ** -12 - r ** -6), 13);
    const F = 24 * (2 * r ** -13 - r ** -7);   // -dE/dr, > 0 repulsive
    expect(s.f[0]).toBeCloseTo(-F, 13);
    expect(s.f[3]).toBeCloseTo(F, 13);
    expect(res.virial).toBeCloseTo(F * r, 13);
  });

  it('shift yes moves the energy, not the force', () => {
    const s = emptyState(lj, 3, { lo: [0, 0, 0], hi: [20, 20, 20], periodic: [true, true, true] }, 1);
    addAtoms(s, Float64Array.from([5, 5, 5, 6.5, 5, 5]), 1);
    s.massByType[1] = 1;
    const t = newPairTable(1, 2.5);
    setPairCoeff(t, '1', '1', 1.0, 1.0);
    resolvePairs(t);
    const b = new CpuForceBackend();
    const plain = b.compute(s, t);
    const f0 = s.f[0];
    t.shift = true;
    const shifted = b.compute(s, t);
    expect(s.f[0]).toBe(f0);
    expect(plain.pe - shifted.pe).toBeCloseTo(4 * (2.5 ** -12 - 2.5 ** -6), 13);
  });

  it('cell list equals the all-images sum on a mixed two-type gas', () => {
    const s = randomGas(600, 10.5, 7);
    const t = newPairTable(2, 2.5);
    setPairCoeff(t, '1', '1', 1.0, 1.0);
    setPairCoeff(t, '2', '2', 0.5, 1.2, 3.0);
    resolvePairs(t);
    const cell = new CpuForceBackend().compute(s, t);
    const fc = Float64Array.from(s.f);
    s.f.fill(0);   // allImages accumulates, like the backend after its own fill
    const brute = allImages(s, pairArrays(t), [10.5, 10.5, 10.5]);
    expect(cell.pe).toBeCloseTo(brute.pe, 9);
    expect(cell.virial).toBeCloseTo(brute.virial, 9);
    let maxErr = 0;
    for (let k = 0; k < fc.length; k++) maxErr = Math.max(maxErr, Math.abs(fc[k] - s.f[k]));
    expect(maxErr).toBeLessThan(1e-10);
    const net = [0, 0, 0];
    for (let k = 0; k < fc.length; k++) net[k % 3] += fc[k];
    for (const c of net) expect(Math.abs(c)).toBeLessThan(1e-9);
  });

  it('a box smaller than the cutoff sums every image (independent direct sum)', () => {
    const s = meltState(2);            // 32 atoms, L ≈ 3.36 < 2.5 * 2
    const L = s.box.hi[0];
    s.x[0] += 0.13; s.x[4] -= 0.07;    // break the symmetry so forces are non-zero
    const t = meltPairs();
    const res = new CpuForceBackend().compute(s, t);
    let pe = 0;
    const f = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) {
      for (let j = 0; j < s.n; j++) {
        for (let a = -3; a <= 3; a++) for (let b = -3; b <= 3; b++) for (let c = -3; c <= 3; c++) {
          if (i === j && a === 0 && b === 0 && c === 0) continue;
          const d = [0, 1, 2].map((k) => s.x[3 * i + k] - s.x[3 * j + k] + [a, b, c][k] * L);
          const r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
          if (r2 >= 6.25) continue;
          pe += 0.5 * 4 * (r2 ** -6 - r2 ** -3);
          const fr = 24 * (2 * r2 ** -7 - r2 ** -4);
          for (let k = 0; k < 3; k++) f[3 * i + k] += fr * d[k];
        }
      }
    }
    expect(res.pe).toBeCloseTo(pe, 9);
    for (let k = 0; k < f.length; k++) expect(s.f[k]).toBeCloseTo(f[k], 9);
  });
});

describe('examples/melt reproduction (LAMMPS log.8Apr21.melt.g++.1)', () => {
  const thermo = async (steps: number) => {
    const s = meltState();
    createVelocities(s, 3.0, 87287);
    const rows: ThermoRow[] = [];
    const momentum0 = [0, 1, 2].map((d) => { let p = 0; for (let i = 0; i < s.n; i++) p += s.v[3 * i + d]; return p; });
    await run(s, meltPairs(), new CpuForceBackend(), [new FixNve('1')], steps, {
      thermoEvery: 50, keywords: ['step', 'temp', 'epair', 'etotal', 'press'], onThermo: (r) => rows.push(r),
    });
    const momentum = [0, 1, 2].map((d) => { let p = 0; for (let i = 0; i < s.n; i++) p += s.v[3 * i + d]; return p; });
    return { rows, momentum0, momentum, s };
  };

  it('step 0 matches to the printed precision', async () => {
    const { rows } = await thermo(0);
    expect(rows).toHaveLength(1);
    const r = rows[0];
    expect(r.step).toBe(0);
    expect(r.temp).toBeCloseTo(3, 12);
    expect(r.epair).toBeCloseTo(-6.7733681, 6);
    expect(r.etotal).toBeCloseTo(-2.2744931, 6);
    expect(r.press).toBeCloseTo(-3.7033504, 6);
  });

  it('250 NVE steps land in the published range and conserve energy and momentum', async () => {
    const { rows, momentum0, momentum } = await thermo(250);
    expect(rows.map((r) => r.step)).toEqual([0, 50, 100, 150, 200, 250]);
    for (const r of rows.slice(1)) {
      expect(r.temp).toBeGreaterThan(1.5);
      expect(r.temp).toBeLessThan(1.8);
      expect(r.press).toBeGreaterThan(5.0);
      expect(r.press).toBeLessThan(6.4);
    }
    const drift = rows[rows.length - 1].etotal! - rows[0].etotal!;
    expect(Math.abs(drift)).toBeLessThan(0.02);
    for (let d = 0; d < 3; d++) {
      expect(Math.abs(momentum0[d])).toBeLessThan(1e-9);
      expect(Math.abs(momentum[d])).toBeLessThan(1e-9);
    }
  }, 60_000);
});

describe('velocity', () => {
  it('create gives exactly T with zero momentum; rot yes removes angular momentum', () => {
    const s = meltState(3);
    s.massByType[1] = 2.5;
    createVelocities(s, 1.7, 5, { dist: 'gaussian', rot: true });
    expect(temperature(s)).toBeCloseTo(1.7, 12);
    const p = [0, 0, 0];
    const L = [0, 0, 0];
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < s.n; i++) { cx += s.x[3 * i]; cy += s.x[3 * i + 1]; cz += s.x[3 * i + 2]; }
    cx /= s.n; cy /= s.n; cz /= s.n;
    for (let i = 0; i < s.n; i++) {
      const [x, y, z] = [s.x[3 * i] - cx, s.x[3 * i + 1] - cy, s.x[3 * i + 2] - cz];
      const [vx, vy, vz] = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
      p[0] += vx; p[1] += vy; p[2] += vz;
      L[0] += y * vz - z * vy; L[1] += z * vx - x * vz; L[2] += x * vy - y * vx;
    }
    for (const c of [...p, ...L]) expect(Math.abs(c)).toBeLessThan(1e-9);
    zeroAngularMomentum(s);   // idempotent
    expect(temperature(s)).toBeCloseTo(1.7, 9);
  });
});

describe('2d', () => {
  it('sq2 lattice with enforce2d stays in the plane and conserves energy', async () => {
    const lat = makeLattice('sq2', 0.7, lj, 2);
    const hi: [number, number, number] = [10 * lat.spacing[0], 10 * lat.spacing[1], 0.5];
    const s = emptyState(lj, 2, { lo: [0, 0, -0.5], hi, periodic: [true, true, true] }, 1);
    addAtoms(s, latticePoints(lat, [0, 0, -0.5], hi, 2), 1);
    s.massByType[1] = 1;
    expect(s.n).toBe(200);
    createVelocities(s, 1.0, 9);
    expect(temperature(s)).toBeCloseTo(1.0, 12);
    const rows: ThermoRow[] = [];
    await run(s, meltPairs(), new CpuForceBackend(), [new FixNve('1'), new FixEnforce2d('2')], 400, {
      thermoEvery: 100, keywords: ['etotal', 'temp'], onThermo: (r) => rows.push(r),
    });
    for (let i = 0; i < s.n; i++) expect(s.x[3 * i + 2]).toBe(0);
    expect(Math.abs(rows[rows.length - 1].etotal! - rows[0].etotal!)).toBeLessThan(0.02);
  });
});

describe('wrapping and thermo keywords', () => {
  it('keeps unwrapped positions exact through image flags', () => {
    const s = meltState(3);
    const L = s.box.hi[0];
    const before = Float64Array.from(s.x);
    s.x[0] += 2.6 * L;
    s.x[1] -= 1.2 * L;
    wrapPositions(s);
    expect(s.x[0]).toBeGreaterThanOrEqual(0);
    expect(s.x[0]).toBeLessThan(L);
    expect(s.x[0] + s.image[0] * L).toBeCloseTo(before[0] + 2.6 * L, 10);
    expect(s.x[1] + s.image[1] * L).toBeCloseTo(before[1] - 1.2 * L, 10);
  });

  it('normalises energies in lj units only', () => {
    const s = meltState(3);
    createVelocities(s, 2.0, 3);
    const res = { pe: -100, virial: 0 };
    const ke = kineticEnergy(s);
    const row = thermoRow(s, ['pe', 'ke', 'etotal', 'atoms', 'density', 'vol'], res);
    expect(row.pe).toBeCloseTo(-100 / s.n, 12);
    expect(row.ke).toBeCloseTo(ke / s.n, 12);
    expect(row.density).toBeCloseTo(0.8442, 10);
    expect(thermoRow(s, ['pe'], res, { norm: false }).pe).toBe(-100);
  });
});
