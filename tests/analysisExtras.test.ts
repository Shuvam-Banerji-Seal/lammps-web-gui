/**
 * Exact checks on hand-made inputs for the analysis extras: coordination, bond angles, S(q) (services/trajectoryAnalysis),
 * block averages, autocorrelation and histograms (services/thermoAnalysis), and the PDB / LAMMPS data / CSV writers.
 */
import { describe, expect, it } from 'vitest';
import {
  computeBondAngleDistribution, computeCoordination, computeStructureFactor, defaultCutoff, type RDFPoint,
} from '../src/services/trajectoryAnalysis';
import { autocorrelation, blockAverage, histogram } from '../src/services/thermoAnalysis';
import {
  framesToCsv, framesToPdb, frameToLammpsData, type TrajectoryRecord,
} from '../src/components/workbench/notebook/trajectory';
import type { SimBox } from '../src/engine/types';
import type { Atom, BoxBounds, TrajectoryFrame } from '../src/types';

/** Simple cubic lattice of n^3 atoms with spacing a in a periodic box of side n*a. */
const cubic = (n: number, a = 1): TrajectoryFrame => {
  const atoms: Atom[] = [];
  let id = 1;
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    atoms.push({ id: id++, molId: 0, type: 1, charge: 0, x: i * a, y: j * a, z: k * a });
  }
  const L = n * a;
  const box: BoxBounds = { xlo: 0, xhi: L, ylo: 0, yhi: L, zlo: 0, zhi: L };
  return { atoms, box };
};

describe('computeCoordination', () => {
  it('gives 6 neighbours per atom on simple cubic for a cutoff between a and sqrt(2) a', () => {
    const r = computeCoordination(cubic(4), 1.2);
    expect(r.perAtom.every((c) => c === 6)).toBe(true);
    expect(r.atomsWith).toEqual([0, 0, 0, 0, 0, 0, 64]);
    expect(r.mean).toBe(6);
  });

  it('adds the 12 second neighbours at sqrt(2) a once the cutoff passes them (18 in all)', () => {
    const r = computeCoordination(cubic(4), 1.5);
    expect(r.mean).toBe(18);
    expect(r.atomsWith[18]).toBe(64);
  });

  it('counts nothing below the nearest-neighbour distance', () => {
    const r = computeCoordination(cubic(4), 0.9);
    expect(r.mean).toBe(0);
    expect(r.atomsWith).toEqual([64]);
  });
});

describe('computeBondAngleDistribution', () => {
  it('has peaks only at 90 and 180 degrees on simple cubic (12 right angles and 3 straight angles per atom)', () => {
    const bins = computeBondAngleDistribution(cubic(4), 1.2, 90);
    expect(bins).toHaveLength(90);
    const total = bins.reduce((s, b) => s + b.count, 0);
    expect(total).toBe(15 * 64);
    // 90 degrees falls in bin 45 ([90, 92)); 180 degrees in the last bin (clamped)
    expect(bins[45].count).toBe(12 * 64);
    expect(bins[89].count).toBe(3 * 64);
    const others = bins.filter((_, b) => b !== 45 && b !== 89);
    expect(others.every((b) => b.count === 0)).toBe(true);
  });

  it('is a probability density per degree: sum(density * width) = 1', () => {
    const bins = computeBondAngleDistribution(cubic(4), 1.2, 90);
    const integral = bins.reduce((s, b) => s + b.density * (b.x1 - b.x0), 0);
    expect(integral).toBeCloseTo(1, 12);
  });

  it('is empty when no atom has two neighbours', () => {
    expect(computeBondAngleDistribution(cubic(4), 0.9)).toEqual([]);
  });
});

describe('computeStructureFactor', () => {
  it('is exactly 1 for an ideal gas (g = 1) at every q', () => {
    const r = Array.from({ length: 50 }, (_, i) => (i + 0.5) * 0.1);
    const g = r.map(() => 1);
    const s = computeStructureFactor({ r, g }, 0.7, 20, 200);
    expect(s).toHaveLength(200);
    expect(s[0].q).toBe(0);
    expect(s[199].q).toBe(20);
    for (const p of s) expect(p.s).toBe(1);
  });

  it('uses the trapezoid rule and the sin(x)/x = 1 limit at q = 0', () => {
    // f(r) = r^2 (g - 1) = [0, 1] at r = [0, 1]; integral = 0.5 * (0 + 1) * 1 = 0.5 at every q for x -> 0
    // S(0) = 1 + 4 pi rho * 0.5 with rho = 1/(4 pi) gives 1.5
    const s = computeStructureFactor({ r: [0, 1], g: [1, 2] }, 1 / (4 * Math.PI), 1, 2);
    expect(s[0].s).toBeCloseTo(1.5, 12);
  });

  it('applies the Lorch window sin(pi r/R)/(pi r/R), which vanishes at the last r', () => {
    // M(0) = 1 and M(R) = 0 with R = 1: the only non-zero integrand value (at r = 1) is removed
    expect(computeStructureFactor({ r: [0, 1], g: [1, 2] }, 1 / (4 * Math.PI), 1, 2, true)[0].s).toBeCloseTo(1, 12);
    // at r = R/2, M = sin(pi/2)/(pi/2) = 2/pi: f = [0, 1 * 2/pi, 0] on r = [0, 0.5, 1] (g - 1 = 4 at r = 0.5)
    const s = computeStructureFactor({ r: [0, 0.5, 1], g: [1, 5, 1] }, 1 / (4 * Math.PI), 1, 2, true);
    expect(s[0].s).toBeCloseTo(1 + 0.5 * (2 / Math.PI), 12);
  });
});

describe('defaultCutoff', () => {
  const rdf = (r: number[], g: number[]): RDFPoint[] => r.map((x, i) => ({ r: x, g: g[i], count: 0 }));

  /** A liquid-like g(r) on a 0.01 grid: first shell at 1.1, first minimum near 1.5, second shell at 2.1, with ripple. */
  const liquid = (noise: number): RDFPoint[] => {
    const out: RDFPoint[] = [];
    for (let k = 1; k <= 400; k++) {
      const r = k * 0.01;
      const shells = 1.8 * Math.exp(-(((r - 1.1) / 0.08) ** 2)) - 0.5 * Math.exp(-(((r - 1.5) / 0.12) ** 2))
        + 0.3 * Math.exp(-(((r - 2.1) / 0.15) ** 2));
      const g = r < 0.9 ? 0 : 1 + shells + noise * Math.sin(37 * k);
      out.push({ r, g, count: 0 });
    }
    return out;
  };

  it('is the first minimum after the first peak, also with sampling noise', () => {
    expect(defaultCutoff(liquid(0))).toBeCloseTo(1.5, 1);
    expect(Math.abs(defaultCutoff(liquid(0.04)) - 1.5)).toBeLessThan(0.08);
  });

  it('is 1.25 times the peak when no minimum follows it, and 3.0 with no peak above 1', () => {
    const decaying = Array.from({ length: 100 }, (_, k) => ({ r: (k + 1) * 0.05, g: k < 20 ? k / 10 : 2 * Math.exp(-(k - 20) / 30), count: 0 }));
    // the smoothed peak is within a bin of the raw one (r = 1.05); g(r) still falls at twice that distance
    expect(defaultCutoff(decaying)).toBeGreaterThan(1.2);
    expect(defaultCutoff(decaying)).toBeLessThan(1.4);
    expect(defaultCutoff(rdf([1, 2, 3, 4], [0, 0.5, 0.9, 0.95]))).toBe(3.0);
  });
});

describe('blockAverage', () => {
  it('gives block means and standard errors of 1..8 for block sizes 1 and 2 (minBlocks 4)', () => {
    const b = blockAverage([1, 2, 3, 4, 5, 6, 7, 8], 4);
    expect(b.map((p) => p.blockSize)).toEqual([1, 2]);
    expect(b[0].mean).toBeCloseTo(4.5, 12);
    // size 1: block means 1..8, sample variance 42/7 = 6, standard error sqrt(6 / 8)
    expect(b[0].stdError).toBeCloseTo(Math.sqrt(0.75), 12);
    // size 2: block means 1.5, 3.5, 5.5, 7.5, sample variance 20/3, standard error sqrt((20/3) / 4)
    expect(b[1].mean).toBeCloseTo(4.5, 12);
    expect(b[1].stdError).toBeCloseTo(Math.sqrt(5 / 3), 12);
  });

  it('keeps block size 4 when two blocks are allowed (minBlocks 2)', () => {
    expect(blockAverage([1, 2, 3, 4, 5, 6, 7, 8], 2).map((p) => p.blockSize)).toEqual([1, 2, 4]);
  });

  it('is empty for too few values', () => {
    expect(blockAverage([1, 2, 3], 4)).toEqual([]);
  });
});

describe('autocorrelation', () => {
  it('is C(k)/C(0) with the mean removed: [1, 0.25, -0.3] for 1..4', () => {
    // mean 2.5, deviations -1.5 -0.5 0.5 1.5; C(0) = 5, C(1) = 1.25, C(2) = -1.5 (both over n)
    const acf = autocorrelation([1, 2, 3, 4], 2);
    expect(acf).toHaveLength(3);
    expect(acf[0]).toBe(1);
    expect(acf[1]).toBeCloseTo(0.25, 12);
    expect(acf[2]).toBeCloseTo(-0.3, 12);
  });

  it('clamps the lag to n - 1 and gives [1, 0, 0] for a constant series', () => {
    expect(autocorrelation([1, 2, 3, 4], 10)).toHaveLength(4);
    expect(autocorrelation([5, 5, 5, 5], 2)).toEqual([1, 0, 0]);
  });
});

describe('histogram', () => {
  it('counts 0..3 into two bins with width 1.5 and densities that integrate to 1', () => {
    const h = histogram([0, 1, 2, 3], 2);
    expect(h.width).toBe(1.5);
    expect(h.counts).toEqual([2, 2]);
    expect(h.centers[0]).toBeCloseTo(0.75, 12);
    expect(h.centers[1]).toBeCloseTo(2.25, 12);
    expect(h.density[0]).toBeCloseTo(1 / 3, 12);
    expect(h.density.reduce((s, d) => s + d * h.width, 0)).toBeCloseTo(1, 12);
  });

  it('is empty for no finite values', () => {
    expect(histogram([NaN], 5).counts).toEqual([]);
  });
});

/* ---- writers ---- */

const box = (lx: number, ly: number, lz: number, tilt: [number, number, number] = [0, 0, 0]): SimBox => ({
  lo: [0, 0, 0], hi: [lx, ly, lz], periodic: [true, true, true], tilt,
  triclinic: tilt.some((t) => t !== 0), boundary: [['p', 'p'], ['p', 'p'], ['p', 'p']],
  minLo: [0, 0, 0], minHi: [lx, ly, lz],
});

const record = (step: number, x: number[], types: number[], sim: SimBox, v?: number[]): TrajectoryRecord => ({
  step,
  x: Float32Array.from(x),
  box: sim,
  topo: { id: Int32Array.from(types.map((_, i) => i + 1)), type: Int32Array.from(types) },
  ...(v ? { v: Float32Array.from(v) } : {}),
});

describe('framesToPdb', () => {
  it('writes CRYST1 in its columns, MODEL/ENDMDL per frame and HETATM lines of 78 columns', () => {
    const f = record(0, [1.5, 2.25, 3, 4, 5, 6], [6, 1], box(10, 10, 10));
    const g = record(10, [1.5, 2.25, 3, 4, 5, 6], [6, 1], box(10, 10, 10));
    const lines = framesToPdb([f, g]).trimEnd().split('\n');
    const cryst = lines[0];
    expect(cryst).toHaveLength(70);
    expect(cryst.slice(0, 6)).toBe('CRYST1');
    expect(cryst.slice(6, 15)).toBe('   10.000');
    expect(cryst.slice(15, 24)).toBe('   10.000');
    expect(cryst.slice(24, 33)).toBe('   10.000');
    expect(cryst.slice(33, 40)).toBe('  90.00');
    expect(cryst.slice(55, 66).trimEnd()).toBe('P 1');
    expect(lines[1]).toBe('MODEL        1');
    expect(lines.filter((l) => l === 'ENDMDL')).toHaveLength(2);
    expect(lines.filter((l) => l.startsWith('MODEL'))).toHaveLength(2);
    expect(lines[lines.length - 1]).toBe('END');
    const atom = lines[2];
    expect(atom).toHaveLength(78);
    expect(atom.slice(0, 6)).toBe('HETATM');
    expect(atom.slice(6, 11)).toBe('    1');
    // atoms are named T<type>, as in the extended XYZ writer; an engine type is not an element
    expect(atom.slice(12, 16)).toBe('T6  ');
    expect(atom.slice(30, 38)).toBe('   1.500');
    expect(atom.slice(38, 46)).toBe('   2.250');
    expect(atom.slice(46, 54)).toBe('   3.000');
    expect(atom.slice(76, 78)).toBe('  ');
    expect(lines[3].slice(12, 16)).toBe('T1  ');
  });

  it('omits CRYST1 for a tilted box', () => {
    const f = record(0, [1, 1, 1], [1], box(10, 10, 10, [1, 0, 0]));
    expect(framesToPdb([f]).startsWith('MODEL')).toBe(true);
  });
});

describe('frameToLammpsData', () => {
  it('writes the header counts, the box lines, Atoms and Velocities', () => {
    const f = record(7, [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 2], box(10, 12, 14), [0.5, 0, 0, 0, 1, 0, 0, 0, -1]);
    const text = frameToLammpsData(f);
    const lines = text.split('\n');
    expect(lines[0]).toContain('step 7');
    expect(lines).toContain('3 atoms');
    expect(lines).toContain('2 atom types');
    expect(lines).toContain('0 10 xlo xhi');
    expect(lines).toContain('0 12 ylo yhi');
    expect(lines).toContain('0 14 zlo zhi');
    expect(lines.some((l) => l.endsWith('xy xz yz'))).toBe(false);
    expect(lines).toContain('Atoms # atomic');
    expect(lines).toContain('1 1 1 2 3');
    expect(lines).toContain('3 2 7 8 9');
    expect(lines).toContain('Velocities');
    expect(lines).toContain('2 0 1 0');
    expect(lines).toContain('3 0 0 -1');
  });

  it('adds the tilt line for a triclinic box', () => {
    const f = record(0, [1, 1, 1], [1], box(10, 10, 10, [1, 0, 0.5]));
    expect(frameToLammpsData(f).split('\n')).toContain('1 0 0.5 xy xz yz');
  });
});

describe('framesToCsv', () => {
  it('writes a header and one row per atom of every frame', () => {
    const f = record(0, [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 2], box(10, 10, 10));
    const g = record(5, [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 2], box(10, 10, 10));
    const lines = framesToCsv([f, g]).trimEnd().split('\n');
    expect(lines[0]).toBe('frame,step,id,type,x,y,z');
    expect(lines).toHaveLength(1 + 6);
    expect(lines[1]).toBe('0,0,1,1,1,2,3');
    expect(lines[6]).toBe('1,5,3,2,7,8,9');
  });

  it('adds velocity columns when a frame carries velocities', () => {
    const f = record(0, [1, 2, 3], [1], box(10, 10, 10), [0.5, 0.25, 0.125]);
    const g = record(1, [1, 2, 3], [1], box(10, 10, 10));
    const lines = framesToCsv([f, g]).trimEnd().split('\n');
    expect(lines[0]).toBe('frame,step,id,type,x,y,z,vx,vy,vz');
    expect(lines[1]).toBe('0,0,1,1,1,2,3,0.5,0.25,0.125');
    expect(lines[2]).toBe('1,1,1,1,1,2,3,,,');
  });
});
