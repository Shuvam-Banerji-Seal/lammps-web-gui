/*
 * A small HDF5 reader for the oracle tests of dump h5md: the file forms the HDF5 library writes with its default
 * (earliest) format settings, which native LAMMPS's ch5md calls produce, and which src/engine-gpl/h5md/hdf5.ts writes.
 * Written from the HDF5 File Format Specification (The HDF Group): superblock version 0, version-1 object headers (with
 * continuation chunks), groups with a symbol table (version-1 B-tree of group nodes, symbol table nodes, local heap),
 * version-1 attribute messages, version-1 dataspace and datatype messages, and version-3 data layout messages
 * (contiguous or chunked, chunks indexed by a version-1 B-tree of chunk nodes). It returns a canonical description:
 * every path with its attributes, and for a dataset its type, dims, maximum dims, chunk dims and values; paths that
 * are hard links to one object are listed together.
 */

export interface H5Value { type: string; dims: number[] | null; value: (number | string)[] }
export interface H5Node {
  kind: 'group' | 'dataset';
  attrs: Record<string, H5Value>;
  type?: string;
  dims?: number[];
  maxdims?: (number | null)[] | null;
  chunk?: number[] | null;
  values?: (number | string)[];
}
export interface H5Canon { nodes: Record<string, H5Node>; hardlinks: string[][] }

const UNDEF = 2 ** 64 - 1;

class R {
  readonly v: DataView;
  constructor(readonly b: Uint8Array) { this.v = new DataView(b.buffer, b.byteOffset, b.byteLength); }
  u8(o: number): number { return this.b[o]; }
  u16(o: number): number { return this.v.getUint16(o, true); }
  u32(o: number): number { return this.v.getUint32(o, true); }
  /** an 8-byte unsigned value (exact below 2^53; all bits set reads as UNDEF) */
  u64(o: number): number {
    const lo = this.v.getUint32(o, true), hi = this.v.getUint32(o + 4, true);
    if (lo === 0xffffffff && hi === 0xffffffff) return UNDEF;
    return hi * 0x100000000 + lo;
  }
  ascii(o: number, n: number): string { let s = ''; for (let i = 0; i < n; i++) s += String.fromCharCode(this.b[o + i]); return s; }
  cstr(o: number): string { let s = ''; while (this.b[o]) s += String.fromCharCode(this.b[o++]); return s; }
}

interface Type { cls: number; size: number; signed: boolean; text: string }
const parseType = (r: R, o: number): Type => {
  const cls = r.u8(o) & 0x0f, bits = r.u8(o + 1), size = r.u32(o + 4);
  if (cls === 0) return { cls, size, signed: (bits & 0x08) !== 0, text: `${(bits & 0x08) ? 'i' : 'u'}${size}` };
  if (cls === 1) return { cls, size, signed: true, text: `f${size}` };
  if (cls === 3) return { cls, size, signed: false, text: `S${size}` };
  throw new Error(`datatype class ${cls} not read`);
};

const parseSpace = (r: R, o: number): { dims: number[] | null; maxdims: (number | null)[] | null } => {
  const version = r.u8(o), rank = r.u8(o + 1), flags = r.u8(o + 2);
  if (version !== 1) throw new Error(`dataspace version ${version} not read`);
  if (rank === 0) return { dims: null, maxdims: null };
  const p = o + 8;
  const dims = Array.from({ length: rank }, (_, i) => r.u64(p + 8 * i));
  const maxdims = flags & 1 ? Array.from({ length: rank }, (_, i) => { const m = r.u64(p + 8 * rank + 8 * i); return m === UNDEF ? null : m; }) : null;
  return { dims, maxdims };
};

/** Elements of `n` items of a type from bytes at o. */
const readValues = (r: R, o: number, t: Type, n: number): (number | string)[] => {
  const out: (number | string)[] = [];
  for (let i = 0; i < n; i++) {
    const p = o + i * t.size;
    if (t.cls === 0) out.push(t.size === 4 ? (t.signed ? r.v.getInt32(p, true) : r.u32(p)) : t.signed ? Number(r.v.getBigInt64(p, true)) : r.u64(p));
    else if (t.cls === 1) out.push(t.size === 8 ? r.v.getFloat64(p, true) : r.v.getFloat32(p, true));
    else out.push(r.ascii(p, t.size).replace(/\0[\s\S]*$/, ''));
  }
  return out;
};

interface Msg { type: number; at: number; size: number }

/** The messages of a version-1 object header, following continuation chunks; and its reference count. */
const headerMessages = (r: R, addr: number): { msgs: Msg[]; refcount: number } => {
  if (r.u8(addr) !== 1) throw new Error(`object header version ${r.u8(addr)} at ${addr} not read`);
  const nmsgs = r.u16(addr + 2), refcount = r.u32(addr + 4), size = r.u32(addr + 8);
  const msgs: Msg[] = [];
  const chunks: [number, number][] = [[addr + 16, size]];
  while (chunks.length && msgs.length < nmsgs) {
    const [c0, len] = chunks.shift()!;
    let p = c0;
    while (p + 8 <= c0 + len && msgs.length < nmsgs) {
      const type = r.u16(p), msize = r.u16(p + 2);
      const at = p + 8;
      if (type === 0x0010) chunks.push([r.u64(at), r.u64(at + 8)]);
      msgs.push({ type, at, size: msize });
      p = at + msize;
    }
  }
  return { msgs, refcount };
};

const parseAttr = (r: R, o: number): [string, H5Value] => {
  const version = r.u8(o);
  if (version !== 1) throw new Error(`attribute message version ${version} not read`);
  const nameSize = r.u16(o + 2), dtSize = r.u16(o + 4), dsSize = r.u16(o + 6);
  const pad = (n: number) => (n + 7) & ~7;
  let p = o + 8;
  const name = r.cstr(p); p += pad(nameSize);
  const t = parseType(r, p); p += pad(dtSize);
  const sp = parseSpace(r, p); p += pad(dsSize);
  const n = sp.dims ? sp.dims.reduce((a, b) => a * b, 1) : 1;
  return [name, { type: t.text, dims: sp.dims, value: readValues(r, p, t, n) }];
};

/** The chunks of a chunked dataset: [offsets, address] from its chunk B-tree. */
const chunkList = (r: R, addr: number, ndims: number, out: [number[], number][]): void => {
  if (r.ascii(addr, 4) !== 'TREE' || r.u8(addr + 4) !== 1) throw new Error(`not a chunk B-tree node at ${addr}`);
  const level = r.u8(addr + 5), entries = r.u16(addr + 6);
  const keySize = 8 + 8 * ndims;
  let p = addr + 24;
  for (let i = 0; i < entries; i++) {
    const offsets = Array.from({ length: ndims }, (_, d) => r.u64(p + 8 + 8 * d));
    const child = r.u64(p + keySize);
    if (level === 0) out.push([offsets, child]);
    else chunkList(r, child, ndims, out);
    p += keySize + 8;
  }
};

/** The members of a group: name and object header address, from its B-tree, symbol table nodes and heap. */
const groupMembers = (r: R, btree: number, heap: number): [string, number][] => {
  if (r.ascii(heap, 4) !== 'HEAP') throw new Error(`no local heap at ${heap}`);
  const data = r.u64(heap + 24);
  const out: [string, number][] = [];
  const visit = (node: number): void => {
    if (node === UNDEF) return;
    if (r.ascii(node, 4) !== 'TREE' || r.u8(node + 4) !== 0) throw new Error(`not a group B-tree node at ${node}`);
    const level = r.u8(node + 5), entries = r.u16(node + 6);
    let p = node + 24 + 8;
    for (let i = 0; i < entries; i++) {
      const child = r.u64(p);
      if (level > 0) visit(child);
      else {
        if (r.ascii(child, 4) !== 'SNOD') throw new Error(`no symbol table node at ${child}`);
        const n = r.u16(child + 6);
        for (let k = 0; k < n; k++) {
          const e = child + 8 + 40 * k;
          out.push([r.cstr(data + r.u64(e)), r.u64(e + 8)]);
        }
      }
      p += 16;
    }
  };
  visit(btree);
  return out;
};

/** Reads an HDF5 file (bytes, or one character per byte) into its canonical description. */
export const readH5 = (input: Uint8Array | string): H5Canon => {
  const bytes = typeof input === 'string' ? Uint8Array.from(input, (c) => c.charCodeAt(0) & 0xff) : input;
  const r = new R(bytes);
  if (r.ascii(1, 3) !== 'HDF' || r.u8(8) !== 0) throw new Error('not an HDF5 file with a version-0 superblock');
  const rootAddr = r.u64(64);
  const nodes: Record<string, H5Node> = {};
  const byAddr = new Map<number, string[]>();
  const walk = (addr: number, path: string): void => {
    if (path) {
      const seen = byAddr.get(addr);
      byAddr.set(addr, [...(seen ?? []), path]);
    }
    const { msgs } = headerMessages(r, addr);
    const attrs: Record<string, H5Value> = {};
    for (const m of msgs) if (m.type === 0x000c) { const [k, v] = parseAttr(r, m.at); attrs[k] = v; }
    const stab = msgs.find((m) => m.type === 0x0011);
    if (stab) {
      if (path) nodes[path] = { kind: 'group', attrs };
      const members = groupMembers(r, r.u64(stab.at), r.u64(stab.at + 8)).sort((a, b) => (a[0] < b[0] ? -1 : 1));
      for (const [name, child] of members) walk(child, path ? `${path}/${name}` : name);
      return;
    }
    const sp = parseSpace(r, msgs.find((m) => m.type === 0x0001)!.at);
    const t = parseType(r, msgs.find((m) => m.type === 0x0003)!.at);
    const lay = msgs.find((m) => m.type === 0x0008)!.at;
    if (r.u8(lay) !== 3) throw new Error(`layout version ${r.u8(lay)} not read`);
    const dims = sp.dims ?? [];
    const n = dims.reduce((a, b) => a * b, 1);
    let values: (number | string)[];
    let chunk: number[] | null = null;
    if (r.u8(lay + 1) === 1) {
      const at = r.u64(lay + 2);
      values = at === UNDEF ? [] : readValues(r, at, t, n);
    } else if (r.u8(lay + 1) === 2) {
      const nd = r.u8(lay + 2);
      const btree = r.u64(lay + 3);
      chunk = Array.from({ length: nd - 1 }, (_, i) => r.u32(lay + 11 + 4 * i));
      values = new Array(n).fill(0);
      if (btree !== UNDEF && n > 0) {
        const list: [number[], number][] = [];
        chunkList(r, btree, nd, list);
        const cn = chunk.reduce((a, b) => a * b, 1);
        for (const [off, at] of list) {
          const cv = readValues(r, at, t, cn);
          // each element of the chunk at its place in the dataset (edge chunks are cut at the dims)
          for (let k = 0; k < cn; k++) {
            let rem = k, flat = 0, inside = true;
            for (let d = 0; d < dims.length; d++) {
              const stride = chunk.slice(d + 1).reduce((a, b) => a * b, 1);
              const idx = off[d] + Math.floor(rem / stride);
              rem %= stride;
              if (idx >= dims[d]) { inside = false; break; }
              flat = flat * dims[d] + idx;
            }
            if (inside) values[flat] = cv[k];
          }
        }
      }
    } else throw new Error(`layout class ${r.u8(lay + 1)} not read`);
    nodes[path] = { kind: 'dataset', attrs, type: t.text, dims, maxdims: sp.maxdims, chunk, values };
  };
  walk(rootAddr, '');
  const hardlinks = [...byAddr.values()].filter((v) => v.length > 1).map((v) => v.slice().sort()).sort();
  return { nodes, hardlinks };
};
