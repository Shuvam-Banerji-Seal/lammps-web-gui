import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { xxh64 } from '../src/engine/zstd';

/*
 * Zstandard dump files (src/engine/zstd.ts): a .zst name with any text dump style (native pipes it through the zstd
 * tool), and the COMPRESS styles atom/zstd ... xyz/zstd with dump_modify checksum and compression_level
 * (src/COMPRESS/dump_atom_zstd.cpp and zstd_file_writer.cpp at b332f28). The frames are decoded here with an RFC 8878
 * reader for Raw blocks, and the content is compared with native's decompressed output.
 */

const LJ = 'units lj\nlattice fcc 0.8442\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\ncreate_atoms 1 box\nmass 1 1\npair_style lj/cut 2.5\npair_coeff 1 1 1 1';

const exec = async (text: string) => {
  const files = new Map<string, string>();
  const errors: string[] = [];
  const s = new Session({
    emit: (e: EngineEvent) => { if (e.kind === 'error') errors.push(e.message); },
    writeFile: (name, t, append) => files.set(name, (append ? files.get(name) ?? '' : '') + t),
  });
  try { await s.execute(text); } catch { /* the error event carries the message */ }
  return { files, errors };
};

/** RFC 8878 frames of Raw blocks: the content, and per frame whether it carried a (correct) checksum. */
const unzstd = (latin1: string): { text: string; checksums: boolean[] } => {
  const b = Uint8Array.from(latin1, (c) => c.charCodeAt(0));
  const parts: Uint8Array[] = [];
  const checksums: boolean[] = [];
  let o = 0;
  while (o < b.length) {
    expect([...b.subarray(o, o + 4)]).toEqual([0x28, 0xb5, 0x2f, 0xfd]);
    const fhd = b[o + 4];
    expect(fhd & 0xfb).toBe(0); // no content size, not single-segment, no dictionary
    o += 6; // magic, descriptor, window descriptor
    const start = parts.length;
    for (;;) {
      const hdr = b[o] | (b[o + 1] << 8) | (b[o + 2] << 16);
      expect((hdr >> 1) & 3).toBe(0); // Raw block
      const size = hdr >>> 3;
      parts.push(b.subarray(o + 3, o + 3 + size));
      o += 3 + size;
      if (hdr & 1) break;
    }
    const has = (fhd & 4) !== 0;
    checksums.push(has);
    if (has) {
      const content = Buffer.concat(parts.slice(start));
      const want = xxh64(new Uint8Array(content))[1];
      expect((b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0).toBe(want);
      o += 4;
    }
  }
  return { text: new TextDecoder().decode(Buffer.concat(parts)), checksums };
};

describe('XXH64 (the Zstandard content checksum)', () => {
  // Reference digests (seed 0) from the python xxhash package for bytes (31 i + 7) mod 256
  const VECTORS: [number, string][] = [
    [0, 'ef46db3751d8e999'], [1, 'a96c7f0ce858bbb7'], [4, 'c60d15b1e3ff8f04'], [8, '3da5c7aa269683e0'],
    [31, '4a74f3a1a39ad4a1'], [32, '8d57d6a4671cc43d'], [33, '62c9fd21ed857664'], [100, 'efa0ad2d3e70c151'],
    [1000, '99594f4828043d35'],
  ];
  for (const [n, hex] of VECTORS) {
    it(`${n} bytes`, () => {
      const b = new Uint8Array(n);
      for (let i = 0; i < n; i++) b[i] = (i * 31 + 7) & 255;
      const [h, l] = xxh64(b);
      expect(h.toString(16).padStart(8, '0') + l.toString(16).padStart(8, '0')).toBe(hex);
    });
  }
});

describe('Zstandard dump files', () => {
  it('a .zst custom dump decompresses to native\'s (native writes it through the zstd tool, with a checksum)', async () => {
    // Measured with native LAMMPS: c.dump.zst is one Zstandard frame with an XXH64 check; zstd -dc gives the fixture
    const r = await exec(`${LJ}\ndump 1 all custom 1 c.dump.zst id type x y z vx\ndump_modify 1 sort id\nrun 3\n`);
    expect(r.errors).toEqual([]);
    const z = unzstd(r.files.get('c.dump.zst')!);
    expect(z.text).toBe(readFileSync(join(__dirname, 'fixtures', 'zstd', 'native_custom_dump.txt'), 'utf8'));
    expect(z.checksums.every(Boolean)).toBe(true);
  });

  it('atom/zstd writes the atom dump, with or without the checksum; a large snapshot spans several blocks', async () => {
    const plain = await exec(`${LJ}\ndump 1 all atom 1 d.atom\nrun 2\n`);
    const on = await exec(`${LJ}\ndump 1 all atom/zstd 1 d.atom.zst\nrun 2\n`);
    const off = await exec(`${LJ}\ndump 1 all atom/zstd 1 d.atom.zst\ndump_modify 1 checksum no compression_level 22\nrun 2\n`);
    expect(on.errors).toEqual([]);
    expect(off.errors).toEqual([]);
    expect(unzstd(on.files.get('d.atom.zst')!)).toEqual({ text: plain.files.get('d.atom'), checksums: [true, true, true] });
    expect(unzstd(off.files.get('d.atom.zst')!)).toEqual({ text: plain.files.get('d.atom'), checksums: [false, false, false] });
    const big = await exec(`${LJ.replace('block 0 2 0 2 0 2', 'block 0 12 0 12 0 12')}\nwrite_dump all custom/zstd big.zst id x y z vx vy vz\n`);
    const bigPlain = await exec(`${LJ.replace('block 0 2 0 2 0 2', 'block 0 12 0 12 0 12')}\nwrite_dump all custom big.txt id x y z vx vy vz\n`);
    expect(bigPlain.files.get('big.txt')!.length).toBeGreaterThan(128 * 1024);
    expect(unzstd(big.files.get('big.zst')!).text).toBe(bigPlain.files.get('big.txt'));
  });

  it('refuses as native: no compressed name, a level out of range, checksum on a style that is not /zstd', async () => {
    expect((await exec(`${LJ}\ndump 1 all custom/zstd 1 d.dump id x\n`)).errors[0]).toContain('Dump custom/zstd only writes compressed files');
    expect((await exec(`${LJ}\ndump 1 all xyz/zstd 1 d.zst\ndump_modify 1 compression_level 23\n`)).errors[0])
      .toContain('Compression level must in the range of [-131072, 22]');
    // Measured with native LAMMPS: ERROR: Unknown dump_modify keyword: checksum (dump atom/gz)
    expect((await exec(`${LJ}\ndump 1 all atom/gz 1 d.gz\ndump_modify 1 checksum yes\n`)).errors[0]).toContain('Unknown dump_modify keyword: checksum');
    expect((await exec(`${LJ}\ndump 1 all atom/zstd 1 d.zst\nrun 0\ndump_modify 1 checksum no\n`)).errors[0])
      .toContain('Checksum flag can not be changed while file is open');
  });
});
