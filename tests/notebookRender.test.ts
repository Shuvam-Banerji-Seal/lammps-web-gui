import { describe, it, expect } from 'vitest';
import {
  bitrateFor, frameFileName, framesZip, orbitRadians, overlayText, repeatIndices, resolveSize, selectFrames,
} from '../src/components/workbench/notebook/renderFrames';
import { TrajectoryBuffer, dumpToRecords, framesToDump, type TrajectoryRecord } from '../src/components/workbench/notebook/trajectory';
import { recordingFormats } from '../src/services/recorder';
import type { FrameEvent } from '../src/engine/view';
import type { SimBox } from '../src/engine/types';

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

const frame = (step: number, n = 4, b: SimBox = box(), types: number[] = [1, 2, 1, 2]): FrameEvent => {
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
    type: Int32Array.from(types.slice(0, n)),
    id: Int32Array.from({ length: n }, (_, i) => i + 1),
    box: b,
  };
};

/** Reads the entries of a stored zip back: name and data, from the local file headers. */
const readZip = (zip: Uint8Array): { name: string; data: Uint8Array }[] => {
  const dv = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  const out: { name: string; data: Uint8Array }[] = [];
  let off = 0;
  while (off + 4 <= zip.length && dv.getUint32(off, true) === 0x04034b50) {
    const method = dv.getUint16(off + 8, true);
    expect(method).toBe(0);
    const size = dv.getUint32(off + 18, true);
    const nameLen = dv.getUint16(off + 26, true);
    const extraLen = dv.getUint16(off + 28, true);
    const name = new TextDecoder().decode(zip.subarray(off + 30, off + 30 + nameLen));
    const dataStart = off + 30 + nameLen + extraLen;
    out.push({ name, data: zip.slice(dataStart, dataStart + size) });
    off = dataStart + size;
  }
  return out;
};

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const fakePng = (tag: number, n = 16): Uint8Array => {
  const b = new Uint8Array(PNG_SIG.length + n);
  b.set(PNG_SIG, 0);
  for (let i = 0; i < n; i++) b[PNG_SIG.length + i] = (tag + i) & 0xff;
  return b;
};

describe('frame range selection', () => {
  it('takes start, start+stride, ... up to end (1-based in, 0-based out)', () => {
    const r = selectFrames(20, { start: 1, end: 10, stride: 2 });
    expect(r).toEqual({ ok: true, indices: [0, 2, 4, 6, 8] });
  });

  it('includes the last frame when the stride lands on it, and a single frame range', () => {
    expect(selectFrames(10, { start: 3, end: 9, stride: 3 })).toEqual({ ok: true, indices: [2, 5, 8] });
    expect(selectFrames(10, { start: 4, end: 4, stride: 5 })).toEqual({ ok: true, indices: [3] });
  });

  it('rejects an empty recording, out-of-range frames, a reversed range and a bad stride', () => {
    expect(selectFrames(0, { start: 1, end: 1, stride: 1 }).ok).toBe(false);
    expect(selectFrames(5, { start: 0, end: 3, stride: 1 }).ok).toBe(false);
    expect(selectFrames(5, { start: 1, end: 6, stride: 1 }).ok).toBe(false);
    expect(selectFrames(5, { start: 4, end: 2, stride: 1 }).ok).toBe(false);
    expect(selectFrames(5, { start: 1, end: 5, stride: 0 }).ok).toBe(false);
    expect(selectFrames(5, { start: 1.5, end: 5, stride: 1 }).ok).toBe(false);
    expect(selectFrames(5, { start: NaN, end: 5, stride: 1 }).ok).toBe(false);
  });

  it('repeats the selection for loops', () => {
    expect(repeatIndices([0, 2], 3)).toEqual([0, 2, 0, 2, 0, 2]);
    expect(repeatIndices([0, 2], 0)).toEqual([0, 2]);
  });
});

describe('file names', () => {
  it('zero-pads the position and keeps the step', () => {
    expect(frameFileName(1, 100)).toBe('frame_0001_step_100.png');
    expect(frameFileName(12345, 0)).toBe('frame_12345_step_0.png');
  });
});

describe('overlay text', () => {
  it('shows the step and the time in the run units (lj is reduced, tau)', () => {
    expect(overlayText(100, 0.005, 'lj', null)).toBe('Step 100 · t = 0.5 τ');
  });

  it('uses the thermo time unit abbreviation for other unit styles', () => {
    expect(overlayText(2000, 1, 'real', 'femtoseconds')).toBe('Step 2000 · t = 2000 fs');
    expect(overlayText(10, 0.001, 'metal', 'picoseconds')).toBe('Step 10 · t = 0.01 ps');
  });

  it('shows only the step without a timestep, and no unit when none is known', () => {
    expect(overlayText(50, null, 'lj', null)).toBe('Step 50');
    expect(overlayText(3, 0.5, undefined, null)).toBe('Step 3 · t = 1.5');
  });
});

describe('output sizes, bitrate and orbit', () => {
  it('keeps the current view for "current" and 16:9 for the presets', () => {
    expect(resolveSize('current')).toBeNull();
    expect(resolveSize('720p')).toEqual({ width: 1280, height: 720 });
    expect(resolveSize('1080p')).toEqual({ width: 1920, height: 1080 });
  });

  it('maps quality presets to bitrates', () => {
    expect(bitrateFor('low')).toBe(6_000_000);
    expect(bitrateFor('medium')).toBe(12_000_000);
    expect(bitrateFor('high')).toBe(24_000_000);
  });

  it('converts degrees per frame to radians and treats off or bad values as 0', () => {
    expect(orbitRadians(180)).toBeCloseTo(Math.PI, 12);
    expect(orbitRadians(0)).toBe(0);
    expect(orbitRadians(-3)).toBe(0);
    expect(orbitRadians(NaN)).toBe(0);
  });
});

describe('video formats', () => {
  it('lists every candidate the browser supports, best first', () => {
    const list = recordingFormats((m) => m.startsWith('video/webm'));
    expect(list.map((f) => f.label)).toEqual(['WebM · VP9', 'WebM · VP8', 'WebM']);
    expect(recordingFormats(() => false)).toEqual([]);
  });

  it('skips a candidate whose check throws', () => {
    const list = recordingFormats((m) => {
      if (m.includes('vp9')) throw new Error('boom');
      return m.startsWith('video/webm');
    });
    expect(list.map((f) => f.label)).toEqual(['WebM · VP8', 'WebM']);
  });
});

describe('the zip of frames', () => {
  it('stores each PNG under its frame name, byte for byte, in order', () => {
    const a = fakePng(1);
    const b = fakePng(9, 40);
    const zip = framesZip([
      { position: 1, step: 100, png: a },
      { position: 3, step: 300, png: b },
    ], new Date(2026, 0, 2, 3, 4, 6));
    const entries = readZip(zip);
    expect(entries.map((e) => e.name)).toEqual(['frame_0001_step_100.png', 'frame_0003_step_300.png']);
    expect(Array.from(entries[0].data)).toEqual(Array.from(a));
    expect(Array.from(entries[1].data)).toEqual(Array.from(b));
    for (const e of entries) expect(Array.from(e.data.slice(0, 8))).toEqual(PNG_SIG);
  });

  it('writes an end-of-central-directory record', () => {
    const zip = framesZip([{ position: 1, step: 0, png: fakePng(0) }]);
    const dv = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
    expect(dv.getUint32(zip.length - 22, true)).toBe(0x06054b50);
    expect(dv.getUint16(zip.length - 22 + 8, true)).toBe(1);
  });
});

describe('dump to buffer loading', () => {
  const records = (): TrajectoryRecord[] => {
    const buf = new TrajectoryBuffer();
    buf.append(frame(0));
    buf.append(frame(10));
    buf.append(frame(20, 4, box(10, [1, 0, 0.5])));
    return buf.list();
  };

  it('reads back the frames, steps, ids, types and positions that framesToDump wrote', () => {
    const src = records();
    const back = dumpToRecords(framesToDump(src));
    expect(back.map((r) => r.step)).toEqual([0, 10, 20]);
    for (let f = 0; f < src.length; f++) {
      expect(Array.from(back[f].topo.id)).toEqual(Array.from(src[f].topo.id));
      expect(Array.from(back[f].topo.type)).toEqual(Array.from(src[f].topo.type));
      expect(back[f].x.length).toBe(src[f].x.length);
      for (let k = 0; k < src[f].x.length; k++) {
        expect(Math.abs(back[f].x[k] - src[f].x[k])).toBeLessThan(1e-5);
      }
    }
  });

  it('keeps each frame cell, including a tilted one in the dump bounding box form', () => {
    const back = dumpToRecords(framesToDump(records()));
    expect(back[0].box.hi).toEqual([10, 10, 10]);
    expect(back[2].box.tilt[0]).toBeCloseTo(1, 6);
    expect(back[2].box.tilt[2]).toBeCloseTo(0.5, 6);
    expect(back[2].box.hi[0]).toBeCloseTo(10, 6);
  });

  it('shares the topology between frames with the same atoms', () => {
    const back = dumpToRecords(framesToDump(records()));
    expect(back[1].topo).toBe(back[0].topo);
  });

  it('throws the parser error for a file with no frames', () => {
    expect(() => dumpToRecords('not a dump')).toThrow();
  });

  it('restores a buffer from records and keeps its stride', () => {
    const buf = new TrajectoryBuffer();
    const r = records();
    buf.restore(r, 2);
    expect(buf.length).toBe(3);
    expect(buf.stride).toBe(2);
    expect(buf.get(2)?.step).toBe(20);
    buf.restore([]);
    expect(buf.length).toBe(0);
    expect(buf.stride).toBe(1);
  });
});
