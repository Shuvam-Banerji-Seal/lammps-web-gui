import { describe, it, expect } from 'vitest';
import { TrajectoryBuffer, framesToDump, framesToExtXYZ } from '../src/components/workbench/notebook/trajectory';
import type { FrameEvent } from '../src/engine/view';
import type { SimBox } from '../src/engine/types';
import { parseDumpFile } from '../src/services/dumpParser';
import { parseXYZFile } from '../src/services/xyzParser';
import { frameToMoleculeData } from '../src/engine/view';

const box = (l = 10, tilt: [number, number, number] = [0, 0, 0]): SimBox => ({
  lo: [0, 0, 0],
  hi: [l, l, l],
  periodic: [true, true, true],
  tilt,
  triclinic: tilt.some((t) => t !== 0),
  boundary: [['p', 'p'], ['p', 'p'], ['p', 'p']],
  minLo: [0, 0, 0],
  minHi: [l, l, l],
});

/** An engine frame of n atoms; positions depend on `step` so frames differ. */
const frame = (step: number, n = 4, b: SimBox = box(), types?: number[]): FrameEvent => {
  const x = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    x[3 * i] = (i + 0.25 * step) % 10;
    x[3 * i + 1] = 1.5 + i;
    x[3 * i + 2] = 2.125 + 0.5 * step;
  }
  return {
    kind: 'frame',
    step,
    x,
    image: new Int32Array(3 * n),
    type: Int32Array.from(types ?? Array.from({ length: n }, (_, i) => (i % 2) + 1)),
    id: Int32Array.from(Array.from({ length: n }, (_, i) => i + 1)),
    box: b,
  };
};

describe('TrajectoryBuffer: recording', () => {
  it('keeps every frame in order, copies positions to float32 and shares topology', () => {
    const buf = new TrajectoryBuffer();
    const f0 = frame(0);
    const f1 = frame(10);
    expect(buf.append(f0)).toBe(true);
    expect(buf.append(f1)).toBe(true);
    expect(buf.length).toBe(2);
    const r0 = buf.get(0)!;
    const r1 = buf.get(1)!;
    expect(r0.step).toBe(0);
    expect(r1.step).toBe(10);
    expect(r0.x).toBeInstanceOf(Float32Array);
    expect(Array.from(r1.x.slice(0, 3))).toEqual([Math.fround(f1.x[0]), Math.fround(f1.x[1]), Math.fround(f1.x[2])]);
    // same atoms: one topology object, not a copy per frame
    expect(r1.topo).toBe(r0.topo);
    // the engine's arrays are copied, so later mutation does not reach the buffer
    f0.x[0] = 999;
    expect(buf.get(0)!.x[0]).not.toBe(999);
  });

  it('starts a new topology when the atoms change', () => {
    const buf = new TrajectoryBuffer();
    buf.append(frame(0, 4));
    buf.append(frame(1, 4, box(), [2, 2, 2, 2]));
    expect(buf.get(1)!.topo).not.toBe(buf.get(0)!.topo);
    expect(Array.from(buf.get(1)!.topo.type)).toEqual([2, 2, 2, 2]);
  });

  it('looks up the nearest frame by step and converts a frame back to an engine frame', () => {
    const buf = new TrajectoryBuffer();
    [0, 25, 50, 75].forEach((s) => buf.append(frame(s)));
    expect(buf.indexOfStep(0)).toBe(0);
    expect(buf.indexOfStep(30)).toBe(1);
    expect(buf.indexOfStep(1000)).toBe(3);
    expect(new TrajectoryBuffer().indexOfStep(5)).toBe(-1);
    const ev = buf.toFrameEvent(2)!;
    expect(ev.step).toBe(50);
    expect(ev.id.length).toBe(4);
    expect(buf.toFrameEvent(9)).toBeNull();
    // the view's converter accepts it as it does an engine frame
    const data = frameToMoleculeData(ev);
    expect(data.atoms.length).toBe(4);
    expect(data.box?.xhi).toBe(10);
  });

  it('clear() empties the buffer and restores the full rate', () => {
    const buf = new TrajectoryBuffer({ capBytes: 1500 });
    for (let s = 0; s < 30; s++) buf.append(frame(s));
    expect(buf.thinned).toBeGreaterThan(0);
    buf.clear();
    expect(buf.length).toBe(0);
    expect(buf.stride).toBe(1);
    expect(buf.thinned).toBe(0);
    expect(buf.bytes).toBe(0);
  });
});

describe('TrajectoryBuffer: memory cap and thinning', () => {
  it('drops every other frame when over the cap, doubles the stride and says so', () => {
    // 4 atoms: 48 B of positions + 256 B bookkeeping per frame, 32 B of ids and types for the topology
    const cap = 1000;
    const buf = new TrajectoryBuffer({ capBytes: cap });
    for (let s = 0; s < 40; s++) buf.append(frame(s));
    expect(buf.thinned).toBeGreaterThan(0);
    expect(buf.stride).toBe(2 ** buf.thinned);
    expect(buf.bytes).toBeLessThanOrEqual(cap);
    // kept frames are spaced by the stride, starting at the first frame
    const steps = buf.list().map((r) => r.step);
    expect(steps[0]).toBe(0);
    for (const s of steps) expect(s % buf.stride).toBe(0);
    for (let k = 1; k < steps.length; k++) expect(steps[k] - steps[k - 1]).toBe(buf.stride);
  });

  it('keeps the most recent frames reachable after thinning', () => {
    const buf = new TrajectoryBuffer({ capBytes: 1000 });
    for (let s = 0; s < 40; s++) buf.append(frame(s));
    const last = buf.get(buf.length - 1)!;
    expect(last.step).toBeGreaterThanOrEqual(40 - 2 * buf.stride);
  });

  it('a larger cap keeps every frame', () => {
    const buf = new TrajectoryBuffer({ capBytes: 1e9 });
    for (let s = 0; s < 40; s++) buf.append(frame(s));
    expect(buf.length).toBe(40);
    expect(buf.thinned).toBe(0);
    expect(buf.stride).toBe(1);
  });
});

describe('trajectory writers', () => {
  const build = (n: number, b = box()) => {
    const buf = new TrajectoryBuffer();
    for (let s = 0; s < n; s++) buf.append(frame(s * 10, 4, b));
    return buf.list();
  };

  it('writes a LAMMPS dump that parseDumpFile reads back', () => {
    const frames = build(3);
    const text = framesToDump(frames);
    expect(text.startsWith('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n4\nITEM: BOX BOUNDS pp pp pp\n')).toBe(true);
    expect(text).toContain('ITEM: ATOMS id type x y z');
    const parsed = parseDumpFile(text);
    expect(parsed.atoms.length).toBe(4);
    expect(parsed.frames?.length).toBe(3);
    const f2 = parsed.frames![2];
    expect(f2.atoms.length).toBe(4);
    const src = frames[2];
    for (let i = 0; i < 4; i++) {
      expect(f2.atoms[i].id).toBe(src.topo.id[i]);
      expect(f2.atoms[i].type).toBe(src.topo.type[i]);
      expect(f2.atoms[i].x).toBeCloseTo(src.x[3 * i], 5);
      expect(f2.atoms[i].y).toBeCloseTo(src.x[3 * i + 1], 5);
      expect(f2.atoms[i].z).toBeCloseTo(src.x[3 * i + 2], 5);
    }
    expect(f2.box?.xhi).toBe(10);
  });

  it('writes a tilted box as a bounding box with tilt factors the dump parser inverts', () => {
    const tilted = box(10, [2, -1, 3]);
    const frames = build(1, tilted);
    const text = framesToDump(frames);
    expect(text).toContain('ITEM: BOX BOUNDS xy xz yz pp pp pp');
    const parsed = parseDumpFile(text);
    const b = parsed.frames?.[0]?.box ?? parsed.box!;
    // the true box is back: lo = 0, hi = 10 in x, y and z
    expect(b.xlo).toBeCloseTo(0, 6);
    expect(b.xhi).toBeCloseTo(10, 6);
    expect(b.ylo).toBeCloseTo(0, 6);
    expect(b.yhi).toBeCloseTo(10, 6);
  });

  it('writes extended XYZ that parseXYZFile reads back frame by frame', () => {
    const frames = build(3);
    const text = framesToExtXYZ(frames);
    const comment = text.split('\n')[1];
    expect(comment).toContain('Step=0');
    expect(comment).toContain('Lattice="10 0 0 0 10 0 0 0 10"');
    expect(comment).toContain('pbc="T T T"');
    const parsed = parseXYZFile(text);
    expect(parsed.frames?.length).toBe(3);
    const f1 = parsed.frames![1];
    expect(f1.atoms.length).toBe(4);
    const src = frames[1];
    for (let i = 0; i < 4; i++) {
      expect(f1.atoms[i].x).toBeCloseTo(src.x[3 * i], 5);
      expect(f1.atoms[i].y).toBeCloseTo(src.x[3 * i + 1], 5);
      expect(f1.atoms[i].z).toBeCloseTo(src.x[3 * i + 2], 5);
    }
  });

  it('writes the box origin and tilt into the XYZ lattice', () => {
    const frames = build(1, box(10, [2, 0, 0]));
    const comment = framesToExtXYZ(frames).split('\n')[1];
    expect(comment).toContain('Lattice="10 0 0 2 10 0 0 0 10"');
    expect(comment).toContain('Origin="0 0 0"');
  });
});
