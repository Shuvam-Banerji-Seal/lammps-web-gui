/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the HDF5 objects that the ch5md
 * library bundled with LAMMPS (lib/h5md/src/ch5md.c, Copyright (C) 2013-2014 Pierre de Buyl, BSD three-clause licence,
 * kept in LICENSE.ch5md here) creates through the HDF5 C library for dump h5md. This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The encoding follows the HDF5 File Format Specification (The HDF Group, version 2 of the specification), in the
 * forms the HDF5 library writes with its default property lists: superblock version 0, version-1 object headers,
 * groups with a symbol table (a version-1 B-tree of group nodes, symbol table nodes and a local heap for the names),
 * attributes in version-1 attribute messages, and datasets with a version-3 data layout message. A dataset is either
 * contiguous (written once) or chunked with an unlimited first dimension, whose chunks are indexed by a version-1
 * B-tree of chunk nodes (node type 1) and appended one record at a time, as H5Dset_extent and H5Dwrite of one
 * hyperslab do. Hard links are further symbol table entries pointing at the same object header, whose reference
 * count says how many there are.
 *
 * The file lives in the session's file store as one character per byte. Each flush appends the bytes allocated since
 * the last one and patches the metadata written earlier (dataspace sizes, B-tree nodes, the end-of-file address).
 */

/** The undefined address of the format (all bits set). */
const UNDEF = -1;
/** Size of offsets and of lengths: 8 bytes (the library's default). */
const O = 8;
/**
 * Group leaf node K and group internal node K of the superblock. The library's defaults are 4 and 16; the writer keeps
 * every group in one symbol table node, so its leaf K bounds the members of a group (2K = 64).
 */
const GROUP_LEAF_K = 32;
const GROUP_INTERNAL_K = 16;
/** Indexed storage internal node K: the library's default 32 (a version-0 superblock does not store it). */
const CHUNK_K = 32;
/** Local heap: H5HL_FREE_NULL, the offset meaning "no free block". */
const HEAP_FREE_NULL = 1;

const align8 = (n: number): number => (n + 7) & ~7;

/** Little-endian byte writer with growable storage. */
class Bytes {
  buf = new Uint8Array(256);
  view = new DataView(this.buf.buffer);
  length = 0;
  private need(n: number): void {
    if (this.length + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.length + n) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.length));
    this.buf = next;
    this.view = new DataView(next.buffer);
  }
  u8(v: number): this { this.need(1); this.buf[this.length++] = v & 0xff; return this; }
  u16(v: number): this { this.need(2); this.view.setUint16(this.length, v, true); this.length += 2; return this; }
  u32(v: number): this { this.need(4); this.view.setUint32(this.length, v >>> 0, true); this.length += 4; return this; }
  i32(v: number): this { this.need(4); this.view.setInt32(this.length, v | 0, true); this.length += 4; return this; }
  f64(v: number): this { this.need(8); this.view.setFloat64(this.length, v, true); this.length += 8; return this; }
  /** An 8-byte unsigned value (an address or a length); UNDEF writes all bits set. */
  u64(v: number): this {
    this.need(8);
    if (v === UNDEF) {
      this.view.setUint32(this.length, 0xffffffff, true);
      this.view.setUint32(this.length + 4, 0xffffffff, true);
    } else {
      this.view.setUint32(this.length, v % 0x100000000, true);
      this.view.setUint32(this.length + 4, Math.floor(v / 0x100000000), true);
    }
    this.length += 8;
    return this;
  }
  ascii(s: string): this { this.need(s.length); for (let i = 0; i < s.length; i++) this.buf[this.length++] = s.charCodeAt(i) & 0xff; return this; }
  bytes(b: Uint8Array): this { this.need(b.length); this.buf.set(b, this.length); this.length += b.length; return this; }
  zeros(n: number): this { this.need(n); this.buf.fill(0, this.length, this.length + n); this.length += n; return this; }
  pad8(): this { return this.zeros(align8(this.length) - this.length); }
  out(): Uint8Array { return this.buf.slice(0, this.length); }
}

/** The element types the H5MD dump uses (H5T_NATIVE_INT, H5T_NATIVE_DOUBLE, H5T_C_S1 of a set size). */
export type H5Type = { kind: 'i32' } | { kind: 'f64' } | { kind: 'str'; size: number };

const typeSize = (t: H5Type): number => (t.kind === 'i32' ? 4 : t.kind === 'f64' ? 8 : t.size);

/** The datatype message body (version 1). */
const datatypeMsg = (t: H5Type): Uint8Array => {
  const b = new Bytes();
  if (t.kind === 'i32') {
    // class 0 (fixed point), version 1; little-endian, signed (bit 3); size 4; bit offset 0, precision 32
    b.u8(0x10).u8(0x08).u8(0).u8(0).u32(4).u16(0).u16(32);
  } else if (t.kind === 'f64') {
    // class 1 (floating point), version 1; little-endian, implied mantissa normalization (2 << 4), sign at bit 63; size 8;
    // bit offset 0, precision 64, exponent at bit 52 of size 11, mantissa at 0 of size 52, exponent bias 1023
    b.u8(0x11).u8(0x20).u8(63).u8(0).u32(8).u16(0).u16(64).u8(52).u8(11).u8(0).u8(52).u32(1023);
  } else {
    // class 3 (string), version 1; null-terminated padding, ASCII; the set size
    b.u8(0x13).u8(0).u8(0).u8(0).u32(t.size);
  }
  return b.out();
};

/** The dataspace message body (version 1): scalar with no dims, else dims and, when given, the maximum dims. */
const dataspaceMsg = (dims: number[] | null, maxdims: number[] | null): Uint8Array => {
  const b = new Bytes();
  const rank = dims ? dims.length : 0;
  b.u8(1).u8(rank).u8(maxdims ? 1 : 0).u8(0).u32(0);
  for (const d of dims ?? []) b.u64(d);
  for (const d of maxdims ?? []) b.u64(d);
  return b.out();
};

/** An attribute value: its type, its dims (null: scalar) and its raw bytes. */
export interface H5Attr { name: string; type: H5Type; dims: number[] | null; data: Uint8Array }

/** The attribute message body (version 1): name, datatype and dataspace each padded to 8 bytes, then the data. */
const attributeMsg = (a: H5Attr): Uint8Array => {
  const dt = datatypeMsg(a.type), ds = dataspaceMsg(a.dims, null);
  const b = new Bytes();
  b.u8(1).u8(0).u16(a.name.length + 1).u16(dt.length).u16(ds.length);
  b.ascii(a.name).u8(0).pad8();
  b.bytes(dt).pad8();
  b.bytes(ds).pad8();
  b.bytes(a.data);
  return b.out();
};

/** Attribute data helpers: a scalar or an array of 32-bit integers, a fixed-size string (or strings). */
export const intAttr = (name: string, values: number[], scalar = false): H5Attr => {
  const b = new Bytes();
  for (const v of values) b.i32(v);
  return { name, type: { kind: 'i32' }, dims: scalar ? null : [values.length], data: b.out() };
};
export const stringAttr = (name: string, value: string): H5Attr => {
  const b = new Bytes().ascii(value);
  return { name, type: { kind: 'str', size: value.length }, dims: null, data: b.out() };
};
/** An array of strings in fixed-size elements of `size` bytes (each null-terminated and zero-padded). */
export const stringArrayAttr = (name: string, values: string[], size: number): H5Attr => {
  const b = new Bytes();
  for (const v of values) b.ascii(v).zeros(size - v.length);
  return { name, type: { kind: 'str', size }, dims: [values.length], data: b.out() };
};

/** Header message types. */
const MSG_DATASPACE = 0x0001, MSG_DATATYPE = 0x0003, MSG_FILL = 0x0005, MSG_LAYOUT = 0x0008, MSG_ATTRIBUTE = 0x000c;
const MSG_STAB = 0x0011;

/** A version-1 object header from its messages; returns the bytes and the offset of each message body. */
const objectHeader = (msgs: { type: number; body: Uint8Array; flags?: number }[], refcount: number): { bytes: Uint8Array; at: number[] } => {
  const b = new Bytes();
  let size = 0;
  for (const m of msgs) size += 8 + align8(m.body.length);
  // version 1, reserved, number of messages, object reference count, header size, then 4 bytes to align to 8
  b.u8(1).u8(0).u16(msgs.length).u32(refcount).u32(size).u32(0);
  const at: number[] = [];
  for (const m of msgs) {
    b.u16(m.type).u16(align8(m.body.length)).u8(m.flags ?? 0).zeros(3);
    at.push(b.length);
    b.bytes(m.body).pad8();
  }
  return { bytes: b.out(), at };
};

/** An object in a group: its name, its object header address, and for a group its B-tree and heap (the scratch pad). */
interface Entry { name: string; addr: number; group: H5Group | null }

export class H5Group {
  readonly entries: Entry[] = [];
  heapAddr = 0;
  heapData = 0;
  heapSize = 0;
  btreeAddr = 0;
  snodAddr = 0;
  /** heap offsets of the names (0: the empty name) */
  readonly nameOff = new Map<string, number>();
  heapBytes = new Bytes();
  constructor(readonly addr: number) {}
}

/** One chunk node of a dataset's chunk B-tree. */
interface ChunkNode { addr: number; level: number; keys: number[][]; children: number[]; sizes: number[]; right: H5ChunkNodeRef | null }
type H5ChunkNodeRef = ChunkNode;

export class H5Dataset {
  /** current dims (the first grows by one per record for a chunked dataset) */
  dims: number[];
  /** address of the dataspace message body, rewritten when the dataset grows */
  spaceAt = 0;
  /** address of the layout message body (the chunk B-tree's root address is in it) */
  layoutAt = 0;
  readonly levels: ChunkNode[][] = [];
  constructor(
    readonly addr: number, readonly type: H5Type, dims: number[], readonly maxdims: number[] | null,
    readonly chunk: number[] | null, private refs = 1,
  ) { this.dims = dims.slice(); }
  get elem(): number { return typeSize(this.type); }
  addRef(): number { return ++this.refs; }
  get refcount(): number { return this.refs; }
}

/** An HDF5 file written through the session's byte writes. */
export class H5File {
  private eof = 0;
  /** bytes from `flushed` to `eof` not yet sent */
  private tail = new Bytes();
  private flushed = 0;
  /** patches below `flushed`, sent at the next flush */
  private patches: { at: number; bytes: Uint8Array }[] = [];
  readonly root: H5Group;
  private readonly datasets = new Map<number, H5Dataset>();
  private readonly headers = new Map<number, { msgs: { type: number; body: Uint8Array }[]; refcount: number }>();

  /** write(bytes, at): at undefined appends. */
  constructor(private readonly write: (bytes: Uint8Array, at?: number) => void) {
    // the superblock (96 bytes), written at the first flush with the final end-of-file address
    this.alloc(96);
    this.root = this.newGroup([]);
    this.flush();
  }

  private alloc(n: number): number {
    const at = this.eof;
    this.eof += align8(n);
    this.tail.zeros(align8(n));
    return at;
  }

  /** Puts bytes at an address: into the unflushed tail, or as a patch of what was already sent. */
  private put(at: number, bytes: Uint8Array): void {
    if (at >= this.flushed) {
      this.tail.buf.set(bytes, at - this.flushed);
      return;
    }
    // a patch may straddle the flushed end (never in practice: objects are written whole)
    this.patches.push({ at, bytes });
  }

  private superblock(): Uint8Array {
    const b = new Bytes();
    b.u8(0x89).ascii('HDF\r\n').u8(0x1a).u8(0x0a);
    // superblock version 0, free-space version 0, root group symbol table entry version 0, reserved, shared header
    // message format version 0, size of offsets, size of lengths, reserved
    b.u8(0).u8(0).u8(0).u8(0).u8(0).u8(O).u8(O).u8(0);
    b.u16(GROUP_LEAF_K).u16(GROUP_INTERNAL_K).u32(0);
    // base address, free-space info (undefined), end of file, driver information block (undefined)
    b.u64(0).u64(UNDEF).u64(this.eof).u64(UNDEF);
    // root group symbol table entry: link name offset 0, object header, cache type 1 with the B-tree and the heap
    b.u64(0).u64(this.root.addr).u32(1).u32(0).u64(this.root.btreeAddr).u64(this.root.heapAddr);
    return b.out();
  }

  /** Sends the tail and the patches (and the superblock with the new end of file). */
  flush(): void {
    if (this.root) this.put(0, this.superblock());
    for (const p of this.patches) this.write(p.bytes, p.at);
    this.patches = [];
    if (this.tail.length) this.write(this.tail.out());
    this.flushed = this.eof;
    this.tail = new Bytes();
  }

  // ---------------------------------------------------------------- groups

  private writeHeader(addr: number): void {
    const h = this.headers.get(addr)!;
    this.put(addr, objectHeader(h.msgs, h.refcount).bytes);
  }

  /** A new group's objects: its header (symbol table message and attributes), heap, B-tree and symbol table node. */
  private newGroup(attrs: H5Attr[]): H5Group {
    const msgs = [{ type: MSG_STAB, body: new Uint8Array(16) }, ...attrs.map((a) => ({ type: MSG_ATTRIBUTE, body: attributeMsg(a) }))];
    const hdrSize = objectHeader(msgs, 1).bytes.length;
    const g = new H5Group(this.alloc(hdrSize));
    g.heapAddr = this.alloc(32);
    g.btreeAddr = this.alloc(24 + (2 * GROUP_INTERNAL_K + 1) * O + 2 * GROUP_INTERNAL_K * O);
    g.snodAddr = this.alloc(8 + 2 * GROUP_LEAF_K * 40);
    g.heapBytes.zeros(8); // the empty name at offset 0
    g.nameOff.set('', 0);
    const stab = new Bytes().u64(g.btreeAddr).u64(g.heapAddr).out();
    msgs[0].body = stab;
    this.headers.set(g.addr, { msgs, refcount: 1 });
    this.writeHeader(g.addr);
    this.writeGroupTables(g);
    return g;
  }

  /** Rewrites a group's heap (moved to the end when it grows), its B-tree node and its symbol table node. */
  private writeGroupTables(g: H5Group): void {
    const data = g.heapBytes.out();
    if (data.length !== g.heapSize) {
      g.heapData = this.alloc(data.length);
      g.heapSize = data.length;
    }
    this.put(g.heapData, data);
    const h = new Bytes();
    h.ascii('HEAP').u8(0).zeros(3).u64(g.heapSize).u64(HEAP_FREE_NULL).u64(g.heapData);
    this.put(g.heapAddr, h.out());
    // symbol table node: entries sorted by name
    const sorted = g.entries.slice().sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    const s = new Bytes();
    s.ascii('SNOD').u8(1).u8(0).u16(sorted.length);
    for (const e of sorted) {
      s.u64(g.nameOff.get(e.name)!).u64(e.addr);
      if (e.group) s.u32(1).u32(0).u64(e.group.btreeAddr).u64(e.group.heapAddr);
      else s.u32(0).u32(0).zeros(16);
    }
    s.zeros(8 + 2 * GROUP_LEAF_K * 40 - s.length);
    this.put(g.snodAddr, s.out());
    // group B-tree: one child (the symbol table node), keys: the empty name and the greatest name
    const b = new Bytes();
    b.ascii('TREE').u8(0).u8(0).u16(sorted.length ? 1 : 0).u64(UNDEF).u64(UNDEF);
    if (sorted.length) b.u64(0).u64(g.snodAddr).u64(g.nameOff.get(sorted[sorted.length - 1].name)!);
    b.zeros(24 + (2 * GROUP_INTERNAL_K + 1) * O + 2 * GROUP_INTERNAL_K * O - b.length);
    this.put(g.btreeAddr, b.out());
  }

  private link(parent: H5Group, name: string, addr: number, group: H5Group | null): void {
    if (parent.entries.some((e) => e.name === name)) throw new Error(`HDF5: ${name} exists`);
    if (parent.entries.length >= 2 * GROUP_LEAF_K) throw new Error('HDF5: too many members in one group');
    if (!parent.nameOff.has(name)) {
      parent.nameOff.set(name, parent.heapBytes.length);
      parent.heapBytes.ascii(name).u8(0).pad8();
    }
    parent.entries.push({ name, addr, group });
    this.writeGroupTables(parent);
  }

  /** H5Gcreate. */
  createGroup(parent: H5Group, name: string, attrs: H5Attr[] = []): H5Group {
    const g = this.newGroup(attrs);
    this.link(parent, name, g.addr, g);
    return g;
  }

  /** The member group `name` of a group, or null. */
  findGroup(parent: H5Group, name: string): H5Group | null {
    return parent.entries.find((e) => e.name === name)?.group ?? null;
  }

  has(parent: H5Group, name: string): boolean { return parent.entries.some((e) => e.name === name); }

  // ---------------------------------------------------------------- datasets

  private datasetHeader(d: H5Dataset, layout: Uint8Array): { type: number; body: Uint8Array }[] {
    const chunked = d.chunk !== null;
    // fill value message version 2: allocation time (incremental for chunked, late for contiguous), fill time "if set",
    // and the library's default fill value: defined, of size 0 (the reader's H5D_FILL_VALUE_DEFAULT; "not defined" would
    // read as H5D_FILL_VALUE_UNDEFINED)
    const fill = new Bytes().u8(2).u8(chunked ? 3 : 2).u8(2).u8(1).u32(0).out();
    return [
      { type: MSG_DATASPACE, body: dataspaceMsg(d.dims, d.maxdims) },
      { type: MSG_DATATYPE, body: datatypeMsg(d.type) },
      { type: MSG_FILL, body: fill },
      { type: MSG_LAYOUT, body: layout },
    ];
  }

  private layoutChunked(d: H5Dataset, btree: number): Uint8Array {
    const b = new Bytes().u8(3).u8(2).u8(d.chunk!.length + 1).u64(btree);
    for (const c of d.chunk!) b.u32(c);
    b.u32(d.elem);
    return b.out();
  }

  /**
   * H5Dcreate of a dataset with an unlimited first dimension (current size 0) in chunks of `chunk` elements: its
   * object header only; the chunk B-tree starts with the first record.
   */
  createChunked(parent: H5Group, name: string, type: H5Type, dims: number[], chunk: number[]): H5Dataset {
    const maxdims = [UNDEF, ...dims.slice(1)];
    const probe = new H5Dataset(0, type, dims, maxdims, chunk);
    const msgs = this.datasetHeader(probe, this.layoutChunked(probe, UNDEF));
    const hdr = objectHeader(msgs, 1);
    const d = new H5Dataset(this.alloc(hdr.bytes.length), type, dims, maxdims, chunk);
    d.spaceAt = d.addr + hdr.at[0];
    d.layoutAt = d.addr + hdr.at[3];
    this.headers.set(d.addr, { msgs, refcount: 1 });
    this.datasets.set(d.addr, d);
    this.writeHeader(d.addr);
    this.link(parent, name, d.addr, null);
    return d;
  }

  /** H5Dcreate and H5Dwrite of a contiguous dataset with its whole data. */
  createContiguous(parent: H5Group, name: string, type: H5Type, dims: number[], data: Uint8Array): H5Dataset {
    const dataAt = this.alloc(data.length);
    this.put(dataAt, data);
    const probe = new H5Dataset(0, type, dims, null, null);
    const layout = new Bytes().u8(3).u8(1).u64(dataAt).u64(data.length).out();
    const msgs = this.datasetHeader(probe, layout);
    const hdr = objectHeader(msgs, 1);
    const d = new H5Dataset(this.alloc(hdr.bytes.length), type, dims, null, null);
    this.headers.set(d.addr, { msgs, refcount: 1 });
    this.writeHeader(d.addr);
    this.link(parent, name, d.addr, null);
    return d;
  }

  /** H5Lcreate_hard: a further link to a dataset (its header's reference count goes up). */
  hardLink(parent: H5Group, name: string, d: H5Dataset): void {
    const h = this.headers.get(d.addr)!;
    h.refcount = d.addRef();
    this.writeHeader(d.addr);
    this.link(parent, name, d.addr, null);
  }

  /**
   * h5md_append: extends the first dimension by one (H5Dset_extent) and writes the new record (H5Dwrite of one
   * hyperslab), chunk by chunk: the record's chunks along the second dimension, each allocated whole.
   */
  appendRecord(d: H5Dataset, data: Uint8Array): void {
    const chunk = d.chunk!;
    const rec = d.dims[0];
    d.dims[0] = rec + 1;
    // the dataspace message with the new size (same length: rewritten in place)
    this.put(d.spaceAt, dataspaceMsg(d.dims, d.maxdims));
    const inner = d.dims.slice(1);
    // elements of one record and per chunk
    const per = inner.reduce((a, b) => a * b, 1);
    const chunkElems = chunk.slice(1).reduce((a, b) => a * b, 1);
    const nchunks = inner.length ? Math.ceil(inner[0] / chunk[1]) : 1;
    const rowElems = inner.slice(1).reduce((a, b) => a * b, 1);
    const elem = d.elem;
    for (let c = 0; c < nchunks; c++) {
      const bytes = new Uint8Array(chunkElems * elem);
      if (inner.length) {
        const r0 = c * chunk[1], r1 = Math.min(inner[0], r0 + chunk[1]);
        bytes.set(data.subarray(r0 * rowElems * elem, r1 * rowElems * elem), 0);
      } else {
        bytes.set(data.subarray(0, per * elem), 0);
      }
      const at = this.alloc(bytes.length);
      this.put(at, bytes);
      const offsets = [rec, ...(inner.length ? [c * chunk[1], ...inner.slice(1).map(() => 0)] : [])];
      this.addChunk(d, offsets, at, bytes.length);
    }
  }

  // ---------------------------------------------------------------- chunk B-tree

  private chunkNodeSize(d: H5Dataset): number {
    const key = 8 + 8 * (d.chunk!.length + 1);
    return 24 + (2 * CHUNK_K + 1) * key + 2 * CHUNK_K * O;
  }

  /** Appends a chunk (offsets in elements of each dimension) to the dataset's chunk B-tree, in order. */
  private addChunk(d: H5Dataset, offsets: number[], addr: number, size: number): void {
    const key = [...offsets, 0];
    let level = 0;
    let child = addr, csize = size, ckey = key;
    for (;;) {
      if (d.levels.length <= level) d.levels.push([]);
      const nodes = d.levels[level];
      let node = nodes[nodes.length - 1];
      if (!node || node.children.length >= 2 * CHUNK_K) {
        const fresh: ChunkNode = { addr: this.alloc(this.chunkNodeSize(d)), level, keys: [ckey], children: [child], sizes: [csize], right: null };
        // the full node now ends where the new one starts (its right key and right sibling)
        if (node) { node.right = fresh; this.writeChunkNode(d, node); }
        nodes.push(fresh);
        node = fresh;
        this.writeChunkNode(d, node);
        if (nodes.length === 1 && level === d.levels.length - 1) break;
        // the new node goes to its parent (a new root level when this level had one node)
        if (nodes.length === 2 && level === d.levels.length - 1) {
          d.levels.push([]);
          const root: ChunkNode = {
            addr: this.alloc(this.chunkNodeSize(d)), level: level + 1, keys: [nodes[0].keys[0], nodes[1].keys[0]],
            children: [nodes[0].addr, nodes[1].addr], sizes: [nodes[0].sizes[0], nodes[1].sizes[0]], right: null,
          };
          d.levels[level + 1].push(root);
          this.writeChunkNode(d, root);
          break;
        }
        // an internal key is its child's first key, with that chunk's size
        child = node.addr; csize = node.sizes[0]; ckey = node.keys[0];
        level++;
        continue;
      }
      node.keys.push(ckey);
      node.children.push(child);
      node.sizes.push(csize);
      this.writeChunkNode(d, node);
      break;
    }
    // the rightmost nodes above the leaf carry the new right key
    for (let l = 1; l < d.levels.length; l++) this.writeChunkNode(d, d.levels[l][d.levels[l].length - 1]);
    const root = d.levels[d.levels.length - 1][0];
    this.put(d.layoutAt, this.layoutChunked(d, root.addr));
  }

  /** The key just past the last chunk of the subtree under a node: its last chunk's offsets plus one chunk. */
  private rightKey(d: H5Dataset, node: ChunkNode): number[] {
    if (node.right) return node.right.keys[0];
    let n = node;
    while (n.level > 0) n = d.levels[n.level - 1][d.levels[n.level - 1].length - 1];
    const last = n.keys[n.keys.length - 1];
    return [...d.chunk!, d.elem].map((c, i) => last[i] + c);
  }

  private writeChunkNode(d: H5Dataset, node: ChunkNode): void {
    const b = new Bytes();
    const left = (() => {
      const nodes = d.levels[node.level];
      const i = nodes.indexOf(node);
      return i > 0 ? nodes[i - 1].addr : UNDEF;
    })();
    b.ascii('TREE').u8(1).u8(node.level).u16(node.children.length).u64(left).u64(node.right ? node.right.addr : UNDEF);
    const keyOut = (k: number[], size: number) => { b.u32(size).u32(0); for (const v of k) b.u64(v); };
    for (let i = 0; i < node.children.length; i++) {
      keyOut(node.keys[i], node.sizes[i]);
      b.u64(node.children[i]);
    }
    keyOut(this.rightKey(d, node), 0);
    b.zeros(this.chunkNodeSize(d) - b.length);
    this.put(node.addr, b.out());
  }
}

/** Little-endian bytes of numbers as 32-bit integers or doubles. */
export const i32Bytes = (v: ArrayLike<number>): Uint8Array => {
  const b = new Bytes();
  for (let i = 0; i < v.length; i++) b.i32(v[i]);
  return b.out();
};
export const f64Bytes = (v: ArrayLike<number>): Uint8Array => {
  const b = new Bytes();
  for (let i = 0; i < v.length; i++) b.f64(v[i]);
  return b.out();
};
