import { describe, expect, it } from 'vitest';
import { dataUrlBytes, isImageDataUrl } from '../src/lammps/exporter';

describe('engine files stored as data URLs (dump image)', () => {
  it('decodes a base64 data URL to its MIME type and bytes', () => {
    const png = 'data:image/png;base64,' + btoa(String.fromCharCode(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a));
    const d = dataUrlBytes(png)!;
    expect(d.mime).toBe('image/png');
    expect(Array.from(d.bytes)).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    expect(isImageDataUrl(png)).toBe(true);
    expect(isImageDataUrl('data:image/x-portable-pixmap;base64,UDYK')).toBe(true);
  });

  it('leaves text files alone', () => {
    expect(dataUrlBytes('ITEM: TIMESTEP\n0\n')).toBeNull();
    expect(isImageDataUrl('data: not an image')).toBe(false);
    expect(isImageDataUrl('LAMMPS data file')).toBe(false);
  });
});
