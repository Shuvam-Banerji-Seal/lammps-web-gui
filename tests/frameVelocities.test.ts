import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { EngineHost } from '../src/engine/host';
import type { FromEngine } from '../src/engine/protocol';
import type { EngineEvent } from '../src/engine/types';
import {
  TrajectoryBuffer, dumpToRecords, kineticTemperature, maxwellBoltzmannDensity, recordSpeeds,
  speedHistogram, type TrajectoryRecord,
} from '../src/components/workbench/notebook/trajectory';
import { boltzmannConstant } from '../src/components/workbench/notebook/chartData';
import { scalarValues, type ColourFrame } from '../src/components/workbench/notebook/render/colour';
import type { FrameEvent } from '../src/engine/view';

const MELT = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 4 0 4 0 4
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 3.0 87287 loop geom
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
fix             1 all nve
run             20`;

/** Runs the melt in a session that emits a frame every 10 steps, with or without velocities. */
const runMelt = async (withVelocities: boolean) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) }, undefined, 10);
  if (withVelocities) session.setFrameVelocities(true);
  await session.execute(MELT);
  const frames = events.filter((e): e is FrameEvent => e.kind === 'frame');
  return { session, frames };
};

describe('engine frames carry velocities only when asked', () => {
  it('omits v and mass by default', async () => {
    const { frames } = await runMelt(false);
    expect(frames.length).toBeGreaterThan(1);
    for (const f of frames) {
      expect(f.v).toBeUndefined();
      expect(f.mass).toBeUndefined();
    }
  });

  it('sends v (3 per atom, the state velocities at that step) and the masses on the first frame of the run', async () => {
    const { session, frames } = await runMelt(true);
    const n = frames[0].id.length;
    expect(n).toBe(4 * 4 * 4 * 4);
    for (const f of frames) expect(f.v?.length).toBe(3 * n);
    // the masses go with the first frame of the run and are not repeated while they stay the same
    expect(frames[0].mass?.length).toBe(n);
    expect(Array.from(frames[0].mass!).every((m) => m === 1)).toBe(true);
    for (const f of frames.slice(1)) expect(f.mass).toBeUndefined();
    // the last frame is the end of the run: its velocities are the state's, atom by atom
    const last = frames[frames.length - 1];
    const state = session.system!;
    expect(last.step).toBe(state.step);
    for (let i = 0; i < 3 * n; i++) expect(last.v![i]).toBe(state.v[i]);
  });

  it('the host sends v and mass through the worker protocol when reset asks for them', async () => {
    const out: FromEngine[] = [];
    const host = new EngineHost((m) => out.push(m));
    await host.handle({ type: 'reset', backend: 'cpu', threads: 1, frameEvery: 10, frameVelocities: true });
    await host.handle({ type: 'exec', id: 1, text: MELT, firstLine: 1 });
    const frames = out.filter((m): m is Extract<FromEngine, { type: 'event' }> => m.type === 'event' && m.event.kind === 'frame').map((m) => m.event as FrameEvent);
    expect(frames.length).toBeGreaterThan(0);
    expect(frames[0].v?.length).toBe(3 * frames[0].id.length);
    expect(frames[0].mass?.length).toBe(frames[0].id.length);
  });
});

describe('notebook records keep velocities', () => {
  it('the buffer copies v and mass from frame events, counts their bytes, and returns them from toFrameEvent', async () => {
    const { frames } = await runMelt(true);
    const buf = new TrajectoryBuffer();
    for (const f of frames) buf.append(f);
    const rec = buf.get(0)!;
    expect(rec.v?.length).toBe(3 * rec.topo.id.length);
    expect(rec.mass?.length).toBe(rec.topo.id.length);
    const plain = new TrajectoryBuffer();
    plain.append({ ...frames[0], v: undefined, mass: undefined });
    expect(buf.bytes).toBeGreaterThan(plain.bytes);
    const ev = buf.toFrameEvent(0)!;
    expect(ev.v?.[0]).toBeCloseTo(frames[0].v![0], 5);
  });

  it('dumpToRecords keeps vx vy vz when every atom has them, and no v without them', () => {
    const dump = [
      'ITEM: TIMESTEP', '5', 'ITEM: NUMBER OF ATOMS', '2',
      'ITEM: BOX BOUNDS pp pp pp', '0 10', '0 10', '0 10',
      'ITEM: ATOMS id type x y z vx vy vz',
      '1 1 1 2 3 0.5 -0.25 0.125',
      '2 1 4 5 6 1 2 3',
    ].join('\n') + '\n';
    const [rec] = dumpToRecords(dump);
    expect(rec.step).toBe(5);
    expect(rec.v).toBeInstanceOf(Float32Array);
    expect(Array.from(rec.v!)).toEqual([0.5, -0.25, 0.125, 1, 2, 3]);
    expect(rec.mass).toBeUndefined();

    const noVel = [
      'ITEM: TIMESTEP', '5', 'ITEM: NUMBER OF ATOMS', '2',
      'ITEM: BOX BOUNDS pp pp pp', '0 10', '0 10', '0 10',
      'ITEM: ATOMS id type x y z',
      '1 1 1 2 3', '2 1 4 5 6',
    ].join('\n') + '\n';
    const [plain] = dumpToRecords(noVel);
    expect(plain.v).toBeUndefined();
  });
});

/** Gaussian sample from a seeded linear congruential generator and Box-Muller. */
const gaussians = (count: number, seed: number): Float64Array => {
  let s = seed >>> 0;
  const uniform = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return (s + 1) / 4294967296;
  };
  const out = new Float64Array(count);
  for (let i = 0; i < count; i += 2) {
    const r = Math.sqrt(-2 * Math.log(uniform()));
    const t = 2 * Math.PI * uniform();
    out[i] = r * Math.cos(t);
    if (i + 1 < count) out[i + 1] = r * Math.sin(t);
  }
  return out;
};

describe('speed distribution and Maxwell-Boltzmann', () => {
  const N = 20000;
  const T = 1.7;
  const m = 2.0;
  const kB = boltzmannConstant('lj') ?? 1;
  const g = gaussians(3 * N, 2024);
  const v = new Float64Array(3 * N);
  for (let i = 0; i < 3 * N; i++) v[i] = Math.sqrt(kB * T / m) * g[i];
  const speed = new Float64Array(N);
  const mass = new Float64Array(N).fill(m);
  for (let i = 0; i < N; i++) speed[i] = Math.hypot(v[3 * i], v[3 * i + 1], v[3 * i + 2]);

  it('the histogram density integrates to 1 (its bins sum to 1)', () => {
    const bins = speedHistogram(speed, 40)!;
    const total = bins.reduce((acc, b) => acc + b.density * (b.x1 - b.x0), 0);
    expect(total).toBeCloseTo(1, 10);
    expect(bins.reduce((acc, b) => acc + b.count, 0)).toBe(N);
  });

  it('the Maxwell-Boltzmann density integrates to 1 and has the analytic mean sqrt(8kT/(pi m))', () => {
    const vmax = 12;
    const steps = 20000;
    const h = vmax / steps;
    let norm = 0;
    let mean = 0;
    for (let i = 0; i <= steps; i++) {
      const x = i * h;
      const f = maxwellBoltzmannDensity(x, m, T, kB);
      const w = i === 0 || i === steps ? 0.5 : 1;
      norm += w * f * h;
      mean += w * x * f * h;
    }
    expect(norm).toBeCloseTo(1, 5);
    expect(mean).toBeCloseTo(Math.sqrt(8 * kB * T / (Math.PI * m)), 5);
  });

  it('the temperature from the velocities is T (within sampling error), and the sampled mean speed matches the MB mean', () => {
    const Tk = kineticTemperature(speed, mass, kB)!;
    expect(Math.abs(Tk - T) / T).toBeLessThan(0.02);
    const sampledMean = speed.reduce((a, b) => a + b, 0) / N;
    const analytic = Math.sqrt(8 * kB * T / (Math.PI * m));
    expect(Math.abs(sampledMean - analytic) / analytic).toBeLessThan(0.02);
  });

  it('recordSpeeds reads the record velocities, and a record without v has no speeds', () => {
    const rec: TrajectoryRecord = {
      step: 0, x: new Float32Array(6), box: { lo: [0, 0, 0], hi: [1, 1, 1], periodic: [true, true, true], tilt: [0, 0, 0], triclinic: false, boundary: [['p', 'p'], ['p', 'p'], ['p', 'p']], minLo: [0, 0, 0], minHi: [1, 1, 1] },
      topo: { id: Int32Array.from([1, 2]), type: Int32Array.from([1, 1]) },
      v: Float32Array.from([3, 4, 0, 0, 0, 2]),
    };
    expect(Array.from(recordSpeeds(rec)!.speed)).toEqual([5, 2]);
    expect(recordSpeeds({ ...rec, v: undefined })).toBeNull();
  });
});

describe('colour by speed and kinetic energy uses the velocities', () => {
  it('speed and kinetic energy come from v (with mass) and need no timestep', () => {
    const frame: ColourFrame = {
      step: 0, x: new Float64Array(6), v: Float64Array.from([3, 4, 0, 0, 0, 2]), mass: Float64Array.from([1, 2]),
      box: { lo: [0, 0, 0], hi: [10, 10, 10] }, topo: { id: Int32Array.from([1, 2]) },
    };
    expect(Array.from(scalarValues('speed', undefined, frame, undefined, null)!)).toEqual([5, 2]);
    expect(Array.from(scalarValues('ke', undefined, frame, undefined, null)!)).toEqual([12.5, 4]);
  });
});

describe('TrajectoryBuffer masses', () => {
  const frameOf = (step: number, mass?: number[]): FrameEvent => ({
    kind: 'frame', step, x: new Float64Array(6), image: new Int32Array(6), type: Int32Array.from([1, 2]), id: Int32Array.from([1, 2]),
    box: { lo: [0, 0, 0], hi: [1, 1, 1], periodic: [true, true, true], tilt: [0, 0, 0], triclinic: false } as never,
    v: new Float64Array(6), ...(mass ? { mass: Float64Array.from(mass) } : {}),
  } as FrameEvent);

  it('shares the last masses sent with the records that follow, also across frames the stride skips', () => {
    const buf = new TrajectoryBuffer();
    buf.append(frameOf(0, [1, 2]));
    buf.append(frameOf(10));
    expect(Array.from(buf.get(1)!.mass!)).toEqual([1, 2]);
    expect(buf.get(1)!.mass).toBe(buf.get(0)!.mass);
    buf.append(frameOf(20, [3, 4]));
    expect(Array.from(buf.get(2)!.mass!)).toEqual([3, 4]);
    buf.clear();
    buf.append(frameOf(30));
    expect(buf.get(0)!.mass).toBeUndefined();
  });
});
