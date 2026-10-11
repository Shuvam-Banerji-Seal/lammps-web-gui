import { describe, expect, it } from 'vitest';
import {
  boltzmannConstant, columnLabel, compressibility, energyDrift, ensembleOf, fitDiffusion, heatCapacity,
  linearFit, maxwellBoltzmannSpeed, numericColumns, pickEvenly, populationVariance, runningMean, segmentDt,
  seriesPoints, summarize, timeUnitLabel, timeUnitName, toCsv, type ThermoSegment,
} from '../src/components/workbench/notebook/chartData';
import { prepareTrajectory, boxAndCountSeries } from '../src/components/workbench/notebook/charts/trajectoryData';
import { computeMSD, msdIsExact } from '../src/services/trajectoryAnalysis';
import type { TrajectoryRecord } from '../src/components/workbench/notebook/trajectory';
import type { SimBox } from '../src/engine/types';

const seg = (rows: Record<string, number>[], extra: Partial<ThermoSegment> = {}): ThermoSegment => ({
  label: 'cell 1',
  keywords: Object.keys(rows[0] ?? {}),
  units: 'lj',
  rows,
  dt: null,
  ensemble: 'unknown',
  ...extra,
});

describe('summarize', () => {
  it('gives mean, sample std (n-1), min and max of known values', () => {
    const s = summarize([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(s.n).toBe(8);
    expect(s.mean).toBeCloseTo(5, 12);
    expect(s.std).toBeCloseTo(Math.sqrt(32 / 7), 12);
    expect(s.min).toBe(2);
    expect(s.max).toBe(9);
  });

  it('skips NaN and infinite values and reports NaN when nothing is left', () => {
    const s = summarize([1, NaN, 3, Infinity]);
    expect(s.n).toBe(2);
    expect(s.mean).toBe(2);
    expect(summarize([]).n).toBe(0);
    expect(Number.isNaN(summarize([NaN]).mean)).toBe(true);
    expect(Number.isNaN(summarize([5]).std)).toBe(true);
  });

  it('handles a very long series without spreading it into a function call', () => {
    const n = 1_000_000;
    const values = new Array<number>(n);
    for (let i = 0; i < n; i++) values[i] = i % 2 === 0 ? 1 : 3;
    const s = summarize(values);
    expect(s.mean).toBeCloseTo(2, 9);
    expect(s.min).toBe(1);
    expect(s.max).toBe(3);
  });
});

describe('populationVariance and linear fits', () => {
  it('population variance of known values', () => {
    expect(populationVariance([1, 2, 3, 4])).toBeCloseTo(1.25, 12);
    expect(Number.isNaN(populationVariance([NaN]))).toBe(true);
  });

  it('fits a known line exactly', () => {
    const pts = [0, 1, 2, 3, 4].map((x) => ({ x, y: 3 * x - 1 }));
    const f = linearFit(pts);
    expect(f?.slope).toBeCloseTo(3, 12);
    expect(f?.intercept).toBeCloseTo(-1, 12);
    expect(f?.r2).toBeCloseTo(1, 12);
  });

  it('returns null with fewer than two finite points or no spread in x', () => {
    expect(linearFit([])).toBeNull();
    expect(linearFit([{ x: 1, y: 1 }])).toBeNull();
    expect(linearFit([{ x: 1, y: 1 }, { x: 1, y: 2 }])).toBeNull();
    expect(linearFit([{ x: 0, y: NaN }, { x: 1, y: 2 }])).toBeNull();
  });
});

describe('fluctuation formulas', () => {
  it('heat capacity from known energies and temperatures (lj: k_B = 1)', () => {
    const energies = [-1, 1, -1, 1]; // variance 1
    const temps = [2, 2, 2, 2];
    const r = heatCapacity(energies, temps, 4, 'lj');
    expect('error' in r).toBe(false);
    if ('error' in r) return;
    expect(r.total).toBeCloseTo(1 / 4, 12); // 1 / (1 * 2^2)
    expect(r.perAtom).toBeCloseTo(1 / 16, 12);
    expect(r.kB).toBe(1);
  });

  it('heat capacity uses the real-units Boltzmann constant in kcal/(mol K)', () => {
    const r = heatCapacity([0, 2], [300, 300], 1, 'real');
    if ('error' in r) throw new Error(r.error);
    expect(r.kB).toBeCloseTo(0.0019872, 6);
    expect(r.total).toBeCloseTo(1 / (r.kB * 300 * 300), 9);
  });

  it('reports why C_V cannot be computed', () => {
    expect('error' in heatCapacity([1, 2], [0, 0], 2, 'lj')).toBe(true);
    expect('error' in heatCapacity([1, 2], [1, 1], 2, 'si-foo' as string)).toBe(true);
  });

  it('compressibility and bulk modulus from known volumes (lj)', () => {
    const r = compressibility([9, 11], [2, 2], 'lj'); // var 1, mean 10
    if ('error' in r) throw new Error(r.error);
    expect(r.kappa).toBeCloseTo(1 / (2 * 10), 12);
    expect(r.bulk).toBeCloseTo(20, 12);
    expect('error' in compressibility([10, 10], [2, 2], 'lj')).toBe(true);
  });

  it('energy drift per atom per time is the slope divided by the atom count', () => {
    const pts = [0, 1, 2, 3].map((x) => ({ x, y: -100 + 0.5 * x }));
    const d = energyDrift(pts, 10);
    expect(d?.perAtomPerTime).toBeCloseTo(0.05, 12);
    expect(energyDrift(pts, 0)).toBeNull();
  });

  it('Boltzmann constants: lj is 1, metal eV/K, unknown style null', () => {
    expect(boltzmannConstant('lj')).toBe(1);
    expect(boltzmannConstant('metal')).toBeCloseTo(8.617333e-5, 10);
    expect(boltzmannConstant('micro')).toBeNull();
  });
});

describe('series extraction', () => {
  const rows = [
    { step: 0, temp: 1, pe: -2, time: 0 },
    { step: 50, temp: 1.2, pe: NaN, time: 0.25 },
    { step: 100, temp: 0.9, pe: -3, time: 0.5 },
  ];
  it('takes finite points, in step or time mode', () => {
    const s = seg(rows, { dt: 0.005 });
    expect(seriesPoints(s, 'pe', 'step')).toEqual([{ x: 0, y: -2 }, { x: 100, y: -3 }]);
    expect(seriesPoints(s, 'temp', 'time').map((p) => p.x)).toEqual([0, 0.25, 0.5]);
  });

  it('derives the timestep from the time column when present', () => {
    expect(segmentDt(seg(rows))).toBeCloseTo(0.005, 12);
    expect(segmentDt(seg([{ step: 0, temp: 1 }], { dt: 0.01 }))).toBe(0.01);
    expect(segmentDt(seg([{ step: 0, temp: 1 }]))).toBeNull();
    expect(seriesPoints(seg([{ step: 0, temp: 1 }]), 'temp', 'time')).toEqual([]);
  });

  it('lists numeric columns without step and skips all-NaN ones', () => {
    const cols = numericColumns([seg([{ step: 0, temp: 1, press: NaN }, { step: 10, temp: 2, press: NaN }])]);
    expect(cols).toEqual(['temp']);
  });

  it('running mean is trailing and keeps x', () => {
    const pts = [1, 2, 3, 4].map((y, i) => ({ x: i, y }));
    expect(runningMean(pts, 2).map((p) => p.y)).toEqual([1, 1.5, 2.5, 3.5]);
    expect(runningMean(pts, 0)).toEqual(pts);
  });

  it('pickEvenly samples a long range without copying it', () => {
    expect(pickEvenly(10, 3)).toEqual([0, 4, 9]);
    expect(pickEvenly(3, 10)).toEqual([0, 1, 2]);
    expect(pickEvenly(0, 5)).toEqual([]);
    expect(pickEvenly(1_000_000, 100)).toHaveLength(100);
  });

  it('labels columns and time axes with the units table', () => {
    expect(columnLabel('temp', 'real')).toBe('temp [Kelvin]');
    expect(columnLabel('c_foo', 'real')).toBe('c_foo');
    expect(timeUnitLabel('real')).toBe('time [femtoseconds]');
    expect(timeUnitName('lj')).toBe('lj time');
  });

  it('infers the ensemble from the fix commands', () => {
    expect(ensembleOf('fix 1 all nve\nrun 10')).toBe('NVE');
    expect(ensembleOf('fix 1 all nvt temp 1 1 0.1')).toBe('NVT');
    expect(ensembleOf('fix 1 all npt temp 1 1 0.1 iso 1 1 1')).toBe('NPT');
    expect(ensembleOf('fix 1 all langevin 1 1 0.5 42')).toBe('NVT');
    expect(ensembleOf('run 10')).toBe('unknown');
  });
});

describe('fits and diffusion', () => {
  it('diffusion coefficient is slope / (2 dim) on the fitted window', () => {
    // MSD = 6 D t with D = 0.5 in 3D
    const msd = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((t) => ({ x: t, y: 3 * t }));
    const f = fitDiffusion(msd, 0.2, 1, 3);
    expect(f?.slope).toBeCloseTo(3, 12);
    expect(f?.D).toBeCloseTo(0.5, 12);
    expect(f?.tFrom).toBeCloseTo(2, 12);
    expect(f?.dim).toBe(3);
    expect(fitDiffusion(msd.slice(0, 1), 0, 1, 3)).toBeNull();
  });
});

describe('CSV and statistics helpers', () => {
  it('quotes fields that need it and blanks missing or NaN values', () => {
    expect(toCsv(['a', 'b'], [['x,y', 1], ['say "hi"', NaN], [undefined, 2]])).toBe('a,b\n"x,y",1\n"say ""hi""",\n,2\n');
  });

  it('maxwell-boltzmann speed density integrates to one (3D and 2D)', () => {
    const integrate = (dim: 2 | 3) => {
      let sum = 0;
      const h = 0.001;
      for (let v = 0; v < 12; v += h) sum += maxwellBoltzmannSpeed(v + h / 2, 1, 1, dim) * h;
      return sum;
    };
    expect(integrate(3)).toBeCloseTo(1, 3);
    expect(integrate(2)).toBeCloseTo(1, 3);
  });
});

/** A record whose positions are wrapped into a periodic box of edge L. */
const record = (step: number, xs: number[], L = 10, ids?: number[]): TrajectoryRecord => {
  const n = xs.length;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { x[3 * i] = xs[i]; x[3 * i + 1] = 5; x[3 * i + 2] = 5; }
  const box = {
    lo: [0, 0, 0], hi: [L, L, L], periodic: [true, true, true], tilt: [0, 0, 0], triclinic: false,
    boundary: [['p', 'p'], ['p', 'p'], ['p', 'p']],
  } as unknown as SimBox;
  const id = Int32Array.from(ids ?? xs.map((_, i) => i + 1));
  return { step, x, box, topo: { id, type: new Int32Array(n).fill(1) } };
};

describe('trajectory preparation', () => {
  it('counts image crossings so the MSD of a steadily moving atom is exact', () => {
    // one atom moves +1.5 per recorded frame in a box of 10: it wraps after frames 4 and 7
    const recs: TrajectoryRecord[] = [];
    for (let k = 0; k < 10; k++) {
      const true_x = 1 + 1.5 * k;
      recs.push(record(k * 10, [((true_x % 10) + 10) % 10]));
    }
    const prep = prepareTrajectory(recs, { maxFrames: 100, maxAtoms: 100 });
    expect(prep.frames).toHaveLength(10);
    expect(msdIsExact(prep.frames, prep.frames[0].box)).toBe(true);
    const msd = computeMSD(prep.frames, prep.frames[0].box, { timeOriginStride: 1 });
    // lag k in frames: displacement 1.5 k, so MSD = (1.5 k)^2 exactly, far beyond L/2 wrapping
    expect(msd[1].msd).toBeCloseTo(2.25, 6);
    expect(msd[4].msd).toBeCloseTo(36, 6);
    expect(msd[9].msd).toBeCloseTo(182.25, 6);
  });

  it('samples frames and atoms and reports it', () => {
    const recs = Array.from({ length: 50 }, (_, k) => record(k, Array.from({ length: 30 }, (_, i) => (i + k) % 10)));
    const prep = prepareTrajectory(recs, { maxFrames: 10, maxAtoms: 10 });
    expect(prep.sampledFrames).toBe(true);
    expect(prep.frameStride).toBe(5);
    expect(prep.frames).toHaveLength(10);
    expect(prep.sampledAtoms).toBe(true);
    expect(prep.atomsUsed).toBe(10);
    expect(prep.stepsPerRecord).toBeCloseTo(1, 12);
  });

  it('follows atoms by id when the topology layout changes between frames', () => {
    const a = record(0, [1, 2], 10, [1, 2]);
    const b = record(1, [2.5, 1.5], 10, [2, 1]); // same atoms, reversed order
    const prep = prepareTrajectory([a, b], { maxFrames: 10, maxAtoms: 10 });
    const first = prep.frames[0].atoms.find((q) => q.id === 1);
    const second = prep.frames[1].atoms.find((q) => q.id === 1);
    expect(first?.x).toBeCloseTo(1, 5);
    expect(second?.x).toBeCloseTo(1.5, 5);
  });

  it('box and atom count series cover every record', () => {
    const series = boxAndCountSeries([record(0, [1, 2], 10), record(5, [1, 2, 3], 4)]);
    expect(series).toEqual([{ step: 0, volume: 1000, atoms: 2 }, { step: 5, volume: 64, atoms: 3 }]);
  });

  it('empty input gives an empty preparation', () => {
    const prep = prepareTrajectory([], { maxFrames: 10, maxAtoms: 10 });
    expect(prep.frames).toEqual([]);
    expect(prep.recorded).toBe(0);
  });
});
