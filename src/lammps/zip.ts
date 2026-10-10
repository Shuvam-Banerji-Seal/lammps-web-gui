/**
 * A minimal ZIP writer for run downloads: method 0 (stored, no compression), local file headers, a central
 * directory and the end-of-central-directory record. Names are UTF-8 with the general-purpose flag bit 11 set.
 * No ZIP64: the archive and its entry count must fit the classic limits (checked below).
 */

export interface ZipEntry {
  name: string;
  data: Uint8Array;
}

const CRC_TABLE: Uint32Array = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32 (IEEE 802.3, as used by ZIP). */
export const crc32 = (data: Uint8Array): number => {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};

/** MS-DOS date and time fields for a local time (years before 1980 clamp to 1980). */
export const dosDateTime = (when: Date): { date: number; time: number } => {
  const year = Math.max(1980, when.getFullYear());
  const date = ((year - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate();
  const time = (when.getHours() << 11) | (when.getMinutes() << 5) | Math.floor(when.getSeconds() / 2);
  return { date, time };
};

const UTF8_FLAG = 0x0800;
const LOCAL_SIG = 0x04034b50;
const CENTRAL_SIG = 0x02014b50;
const END_SIG = 0x06054b50;

/** Build a stored (uncompressed) ZIP archive holding the given entries, in order. */
export const buildZip = (entries: ZipEntry[], when: Date = new Date()): Uint8Array<ArrayBuffer> => {
  if (entries.length > 0xffff) throw new RangeError('zip: more than 65535 entries');
  const enc = new TextEncoder();
  const { date, time } = dosDateTime(when);

  const prepared = entries.map((e) => {
    const name = enc.encode(e.name);
    if (name.length > 0xffff) throw new RangeError(`zip: name too long: ${e.name.slice(0, 40)}`);
    if (e.data.length > 0xffffffff) throw new RangeError(`zip: entry too large: ${e.name}`);
    return { name, data: e.data, crc: crc32(e.data) };
  });

  let total = 0;
  for (const p of prepared) total += 30 + p.name.length + p.data.length + 46 + p.name.length;
  total += 22;
  if (total > 0xffffffff) throw new RangeError('zip: archive too large for the classic format');

  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  const offsets: number[] = [];
  let pos = 0;

  for (const p of prepared) {
    offsets.push(pos);
    view.setUint32(pos, LOCAL_SIG, true);
    view.setUint16(pos + 4, 20, true); // version needed: 2.0
    view.setUint16(pos + 6, UTF8_FLAG, true);
    view.setUint16(pos + 8, 0, true); // method: stored
    view.setUint16(pos + 10, time, true);
    view.setUint16(pos + 12, date, true);
    view.setUint32(pos + 14, p.crc, true);
    view.setUint32(pos + 18, p.data.length, true);
    view.setUint32(pos + 22, p.data.length, true);
    view.setUint16(pos + 26, p.name.length, true);
    view.setUint16(pos + 28, 0, true); // extra length
    out.set(p.name, pos + 30);
    out.set(p.data, pos + 30 + p.name.length);
    pos += 30 + p.name.length + p.data.length;
  }

  const cdStart = pos;
  prepared.forEach((p, k) => {
    view.setUint32(pos, CENTRAL_SIG, true);
    view.setUint16(pos + 4, 20, true); // version made by
    view.setUint16(pos + 6, 20, true); // version needed
    view.setUint16(pos + 8, UTF8_FLAG, true);
    view.setUint16(pos + 10, 0, true); // method: stored
    view.setUint16(pos + 12, time, true);
    view.setUint16(pos + 14, date, true);
    view.setUint32(pos + 16, p.crc, true);
    view.setUint32(pos + 20, p.data.length, true);
    view.setUint32(pos + 24, p.data.length, true);
    view.setUint16(pos + 28, p.name.length, true);
    view.setUint16(pos + 30, 0, true); // extra length
    view.setUint16(pos + 32, 0, true); // comment length
    view.setUint16(pos + 34, 0, true); // disk number start
    view.setUint16(pos + 36, 0, true); // internal attributes
    view.setUint32(pos + 38, 0, true); // external attributes
    view.setUint32(pos + 42, offsets[k], true);
    out.set(p.name, pos + 46);
    pos += 46 + p.name.length;
  });
  const cdSize = pos - cdStart;

  view.setUint32(pos, END_SIG, true);
  view.setUint16(pos + 4, 0, true);
  view.setUint16(pos + 6, 0, true);
  view.setUint16(pos + 8, prepared.length, true);
  view.setUint16(pos + 10, prepared.length, true);
  view.setUint32(pos + 12, cdSize, true);
  view.setUint32(pos + 16, cdStart, true);
  view.setUint16(pos + 20, 0, true); // comment length
  return out;
};
