import { describe, expect, it } from 'vitest';
import { crc32, deflateRawSync, gzipSync } from 'node:zlib';
import { gunzip } from '../src/engine/gunzip';
import { StyleError } from '../src/engine/force/types';
import { Session, SUPPORTED_COMMANDS } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/** Bytes of a node Buffer/Uint8Array as a fresh Uint8Array (no shared offset). */
const u8 = (b: Uint8Array): Uint8Array => new Uint8Array(b);

const expectBytes = (got: Uint8Array, want: Uint8Array): void => {
  expect(got.length).toBe(want.length);
  expect(Buffer.compare(Buffer.from(got), Buffer.from(want))).toBe(0);
};

/** A gzip member with hand-built RFC 1952 header, wrapping a raw DEFLATE stream of `plain`. */
function gzMember(plain: Buffer, opts: { fname?: string; fcomment?: string; fextra?: Buffer } = {}): Buffer {
  const raw = deflateRawSync(plain);
  const flg = (opts.fextra ? 0x04 : 0) | (opts.fname ? 0x08 : 0) | (opts.fcomment ? 0x10 : 0);
  const head: number[] = [0x1f, 0x8b, 8, flg, 0, 0, 0, 0, 0, 0xff];
  if (opts.fextra) head.push(opts.fextra.length & 0xff, (opts.fextra.length >> 8) & 0xff, ...opts.fextra);
  if (opts.fname) for (const ch of Buffer.from(opts.fname + '\0')) head.push(ch);
  if (opts.fcomment) for (const ch of Buffer.from(opts.fcomment + '\0')) head.push(ch);
  const trailer = Buffer.alloc(8);
  trailer.writeUInt32LE(crc32(plain) >>> 0, 0);
  trailer.writeUInt32LE(plain.length >>> 0, 4);
  return Buffer.concat([Buffer.from(head), raw, trailer]);
}

// A deterministic pseudo-random byte string (no dependency on Math.random).
function prngBytes(n: number): Uint8Array {
  const out = new Uint8Array(n);
  let s = 0x12345678;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s >>> 24) & 0xff;
  }
  return out;
}

function lammpsDataText(approxBytes: number): string {
  const lines: string[] = ['generated data file', '', '40000 atoms', '', '0 40 xlo xhi', '0 40 ylo yhi', '0 40 zlo zhi', '', 'Atoms # atomic', ''];
  let bytes = 120;
  let id = 1;
  while (bytes < approxBytes) {
    const l = `${id} 1 ${(id % 40) + 0.5} ${((id * 7) % 40) + 0.25} ${((id * 13) % 40) + 0.125}`;
    lines.push(l);
    bytes += l.length + 1;
    id++;
  }
  return lines.join('\n') + '\n';
}

describe('gunzip (RFC 1952 + RFC 1951)', () => {
  it('decodes an empty stream', () => {
    expectBytes(gunzip(u8(gzipSync(Buffer.alloc(0)))), Buffer.alloc(0));
  });

  it('decodes a single byte', () => {
    expectBytes(gunzip(u8(gzipSync(Buffer.from([0x42])))), Buffer.from([0x42]));
  });

  it('decodes text with long repeats (overlapping window copies)', () => {
    const text = 'the quick brown fox jumps over the lazy dog\n'.repeat(50000);
    const plain = Buffer.from(text);
    expectBytes(gunzip(u8(gzipSync(plain))), plain);
  });

  it('decodes random bytes stored with level 0 (stored blocks)', () => {
    const plain = Buffer.from(prngBytes(200000));
    const gz = gzipSync(plain, { level: 0 });
    expectBytes(gunzip(u8(gz)), plain);
  });

  it('decodes level 1 and level 9 output', () => {
    const plain = Buffer.from(('LAMMPS input script line with some structure 0123456789\n').repeat(20000));
    for (const level of [1, 9] as const) {
      expectBytes(gunzip(u8(gzipSync(plain, { level }))), plain);
    }
  });

  it('decodes 5 MB of generated LAMMPS data text in well under a second', () => {
    const text = lammpsDataText(5 * 1024 * 1024);
    const plain = Buffer.from(text);
    expect(plain.length).toBeGreaterThan(4.5 * 1024 * 1024);
    const gz = gzipSync(plain, { level: 6 });
    const t0 = Date.now();
    const got = gunzip(u8(gz));
    const ms = Date.now() - t0;
    expectBytes(got, plain);
    expect(ms).toBeLessThan(1000);
  });

  it('decodes two concatenated members and joins them', () => {
    const a = Buffer.from('first member\n'.repeat(1000));
    const b = Buffer.from('second member with different content\n'.repeat(1000));
    const both = Buffer.concat([gzipSync(a), gzipSync(b)]);
    expectBytes(gunzip(u8(both)), Buffer.concat([a, b]));
  });

  it('skips FEXTRA, FNAME and FCOMMENT header fields', () => {
    const plain = Buffer.from('header fields must be skipped exactly\n'.repeat(500));
    const cases: Buffer[] = [
      gzMember(plain, { fname: 'data.x' }),
      gzMember(plain, { fcomment: 'made by the test suite' }),
      gzMember(plain, { fextra: Buffer.from([1, 2, 3, 4, 5, 6, 7]) }),
      gzMember(plain, { fname: 'a.data', fcomment: 'note', fextra: Buffer.from('ABCD') }),
    ];
    for (const gz of cases) expectBytes(gunzip(u8(gz)), plain);
  });

  it('throws StyleError on a corrupted CRC-32', () => {
    const gz = Buffer.from(gzipSync(Buffer.from('payload to corrupt')));
    gz[gz.length - 8] ^= 0xff; // first trailer byte = low byte of CRC-32
    expect(() => gunzip(u8(gz))).toThrow(StyleError);
    expect(() => gunzip(u8(gz))).toThrow(/CRC-32/);
  });

  it('throws StyleError on a truncated stream', () => {
    const gz = Buffer.from(gzipSync(Buffer.from('payload that gets cut off'.repeat(100))));
    expect(() => gunzip(u8(gz.subarray(0, gz.length - 4)))).toThrow(StyleError);
    expect(() => gunzip(u8(gz.subarray(0, 12)))).toThrow(StyleError);
  });

  it('throws StyleError on a bad magic number', () => {
    const gz = Buffer.from(gzipSync(Buffer.from('hello')));
    gz[0] = 0x00;
    expect(() => gunzip(u8(gz))).toThrow(StyleError);
  });
});

// ---------------------------------------------------------------- end to end (read_data .gz)

const DATA = `small molecular system

4 atoms
2 bonds
2 atom types
1 bond types

0 10.0 xlo xhi
0 10.0 ylo yhi
0 10.0 zlo zhi

Masses

1 1.0
2 1.2

Atoms # molecular

1 1 1 1.0 1.0 1.0
2 1 2 2.0 1.0 1.0
3 1 1 5.0 5.0 5.0
4 1 2 6.0 5.0 5.0

Bonds

1 1 1 2
2 1 3 4
`;

const SCRIPT = (file: string): string => `
units           lj
atom_style      molecular
bond_style      harmonic
pair_style      lj/cut 2.5
read_data       ${file}
pair_coeff      * * 0.5 0.9
bond_coeff      1 120.0 1.05
thermo_style    custom step atoms pe ebond etotal
thermo_modify   format float %.15g
thermo          1
run             0
`;

/** One char per byte, exactly how the notebook uploads a .gz file. */
function bytesToString(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

async function run(script: string, files: Record<string, string>) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  await session.execute(script);
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { session, thermo };
}

describe('read_data of a gzipped file (end to end)', () => {
  it('gives the same atoms and thermo as the plain file', async () => {
    const plain = await run(SCRIPT('small.data'), { 'small.data': DATA });
    const gz = await run(SCRIPT('small.data.gz'), { 'small.data.gz': bytesToString(u8(gzipSync(Buffer.from(DATA)))) });

    expect(gz.thermo).toEqual(plain.thermo);
    expect(gz.thermo.length).toBeGreaterThan(0);

    const a = plain.session.system!;
    const b = gz.session.system!;
    expect(b.n).toBe(a.n);
    expect(Array.from(b.id)).toEqual(Array.from(a.id));
    expect(Array.from(b.type)).toEqual(Array.from(a.type));
    expect(Array.from(b.x)).toEqual(Array.from(a.x));
    expect(Array.from(b.molecule)).toEqual(Array.from(a.molecule));
    expect(SUPPORTED_COMMANDS).toContain('read_data');
  });
});
