import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildZip, crc32, dosDateTime } from '../src/lammps/zip';

const enc = new TextEncoder();
const scratch = () => mkdtempSync(join(tmpdir(), 'zip-test-'));

/** Write the archive to a scratch file and return its path (the caller removes the directory). */
const writeArchive = (dir: string, entries: { name: string; data: Uint8Array }[], when?: Date): string => {
  const path = join(dir, 'out.zip');
  writeFileSync(path, buildZip(entries, when));
  return path;
};

// Info-ZIP 6.00 reads names in the charset given by -O (it ignores the UTF-8 flag without it)
const unzip = (args: string[]) => spawnSync('unzip', ['-O', 'UTF-8', ...args], { encoding: 'buffer' });

describe('zip writer', () => {
  it('computes the standard CRC-32 check value', () => {
    expect(crc32(enc.encode('123456789'))).toBe(0xcbf43926);
    expect(crc32(new Uint8Array(0))).toBe(0);
  });

  it('round-trips text entries through the system unzip', () => {
    const dir = scratch();
    try {
      const entries = [
        { name: 'dump.lammpstrj', data: enc.encode('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n') },
        { name: 'out.data', data: enc.encode('LAMMPS data file\n\n2 atoms\n') },
      ];
      const zip = writeArchive(dir, entries);

      const list = unzip(['-l', zip]);
      expect(list.status).toBe(0);
      const listing = list.stdout.toString();
      expect(listing).toContain('dump.lammpstrj');
      expect(listing).toContain('out.data');

      expect(unzip(['-t', zip]).status).toBe(0); // checks every CRC
      for (const e of entries) {
        const got = unzip(['-p', zip, e.name]);
        expect(got.status).toBe(0);
        expect(Buffer.from(got.stdout).equals(Buffer.from(e.data))).toBe(true);
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('stores binary content byte for byte, with the CRC in the central directory', () => {
    const dir = scratch();
    try {
      const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 255, 128, 0]);
      const zip = writeArchive(dir, [{ name: 'frame.png', data: png }]);
      const got = unzip(['-p', zip, 'frame.png']);
      expect(Array.from(got.stdout)).toEqual(Array.from(png));
      // the central directory record (signature 0x02014b50) carries the CRC of the data
      const bytes = new Uint8Array(readFileSync(zip));
      const cd = bytes.findIndex((_, i) => bytes[i] === 0x50 && bytes[i + 1] === 0x4b && bytes[i + 2] === 0x01 && bytes[i + 3] === 0x02);
      const crc = new DataView(bytes.buffer, bytes.byteOffset).getUint32(cd + 16, true);
      expect(crc).toBe(crc32(png));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('keeps empty files and UTF-8 names (flag bit 11)', () => {
    const dir = scratch();
    try {
      const name = 'résumé-μ-日本.txt';
      const zip = writeArchive(dir, [
        { name: 'empty.txt', data: new Uint8Array(0) },
        { name, data: enc.encode('héllo\n') },
      ]);
      expect(unzip(['-t', zip]).status).toBe(0);
      expect(unzip(['-l', zip]).stdout.toString('utf8')).toContain(name);
      expect(unzip(['-p', zip, name]).stdout.toString('utf8')).toBe('héllo\n');
      expect(unzip(['-p', zip, 'empty.txt']).stdout.length).toBe(0);
      // the name bytes are the UTF-8 encoding (no escaping)
      expect(new TextDecoder().decode(new Uint8Array(readFileSync(zip))).includes(name)).toBe(true);
      const bytes = new Uint8Array(readFileSync(zip));
      // general-purpose flag at offset 6 of the first local header has bit 11 set
      expect(new DataView(bytes.buffer, bytes.byteOffset).getUint16(6, true) & 0x0800).toBe(0x0800);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('writes the DOS date and time of the given moment', () => {
    const when = new Date(2026, 9, 11, 14, 30, 58); // 11 Oct 2026 14:30:58 local
    expect(dosDateTime(when)).toEqual({
      date: ((2026 - 1980) << 9) | (10 << 5) | 11,
      time: (14 << 11) | (30 << 5) | 29,
    });
    expect(dosDateTime(new Date(1970, 0, 1)).date).toBe((0 << 9) | (1 << 5) | 1); // clamps to 1980
  });

  it('writes an empty archive that unzip accepts', () => {
    const dir = scratch();
    try {
      const zip = writeArchive(dir, []);
      expect(readFileSync(zip).length).toBe(22); // just the end-of-central-directory record
      // Info-ZIP reports an archive with no entries as "zipfile is empty" (status 1); the file is still valid
      const t = unzip(['-t', zip]);
      expect(t.stdout.toString() + t.stderr.toString()).toMatch(/empty/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
