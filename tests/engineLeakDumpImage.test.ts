import { describe, expect, it } from 'vitest';
import v8 from 'node:v8';
import vm from 'node:vm';
import { Session } from '../src/engine/interpreter';
import { toBase64 } from '../src/engine/output/dump_image';

/*
 * Retained-heap checks for dump image output (src/engine/output/dump_image.ts). Each snapshot is kept in the
 * session's file store as a data: URL, so its heap cost must be about its text, not a multiple of it.
 * vitest does not pass --expose-gc, so the collector is switched on at runtime and read from a new vm context.
 */

v8.setFlagsFromString('--expose-gc');
const gc = vm.runInNewContext('gc') as () => void;
/** Heap in use after a full collection: what the test still holds. */
const retainedHeap = (): number => { gc(); gc(); return process.memoryUsage().heapUsed; };

/** The base64 encoder as first written (repeated `+=`): the reference the flat encoder must match byte for byte. */
const referenceBase64 = (bytes: Uint8Array): string => {
  const T = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i], b1 = bytes[i + 1], b2 = bytes[i + 2];
    out += T[b0 >> 2];
    out += T[((b0 & 3) << 4) | (b1 === undefined ? 0 : b1 >> 4)];
    out += b1 === undefined ? '=' : T[((b1 & 15) << 2) | (b2 === undefined ? 0 : b2 >> 6)];
    out += b2 === undefined ? '=' : T[b2 & 63];
  }
  return out;
};

const randomBytes = (n: number, seed: number): Uint8Array => {
  const out = new Uint8Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; out[i] = s >>> 24; }
  return out;
};

describe('dump image base64 encoding', () => {
  it('produces exactly the same text as the reference encoder for every length mod 3', () => {
    for (let n = 0; n <= 40; n++) {
      const b = randomBytes(n, n + 1);
      expect(toBase64(b)).toBe(referenceBase64(b));
    }
    const big = randomBytes(600 * 600 * 3 + 20, 7);
    expect(toBase64(big)).toBe(referenceBase64(big));
  });

  it('retains about its text per frame, not a rope many times its size', () => {
    const frames = 10;
    const bytes = randomBytes(600 * 600 * 3, 3);
    const before = retainedHeap();
    const kept: string[] = [];
    for (let k = 0; k < frames; k++) kept.push(toBase64(bytes));
    const after = retainedHeap();
    const textBytes = kept[0].length * frames;
    expect(kept[frames - 1].length).toBeGreaterThan(1_000_000);
    // one-byte strings: the flat text costs about its length; a rope cost about 30 times that
    expect(after - before).toBeLessThan(2 * textBytes + 1_000_000);
  });
});

describe('dump image snapshots kept in the session', () => {
  it('retains about the text of the frames written, when a run is doubled', async () => {
    const input = (steps: number) => `units lj
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 87287
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
dump d all image 5 snap.*.ppm type type size 96 96 box no 0.02 view 60 30
run ${steps}
`;
    let addedText = 0;
    const sess = new Session({ emit: () => undefined, writeFile: (_n, t) => { addedText += t.length; } });
    await sess.execute(input(50));
    const heapAfterN = retainedHeap();
    const textAfterN = addedText;
    await sess.execute('run 50\n');
    const heapAfter2N = retainedHeap();
    const textAdded = addedText - textAfterN;
    expect(textAfterN).toBeGreaterThan(0);
    expect(textAdded).toBeGreaterThan(0);
    // the second run keeps its own frames (about their text) and nothing else of the first run
    expect(heapAfter2N - heapAfterN).toBeLessThan(2 * textAdded + 2_000_000);
  });
});
