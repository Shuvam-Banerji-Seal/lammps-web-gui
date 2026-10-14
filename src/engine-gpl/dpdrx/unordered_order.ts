/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): fix_rx.cpp post_constructor (src/DPD-REACT/fix_rx.cpp),
 * which fills its species set as a std::unordered_set<std::string> and numbers the species in its iteration order. This file
 * reproduces that iteration order as libstdc++ (GCC 11, the compiler of the native binary) produces it: std::hash<std::string>
 * is _Hash_bytes (MurmurHash64A, seed 0xc70f6907, libstdc++-v3/src/c++11/hash_bytes.cc); buckets follow _Prime_rehash_policy
 * (max load factor 1, growth factor 2, the small bucket table __fast_bkt of hashtable_c++0x.cc); nodes sit in one singly linked
 * list with the bucket-begin rule of _Hashtable::_M_insert_bucket_begin and the rehash of _M_rehash_aux. LAMMPS is Copyright
 * (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Native sums over the species in this order, so the order sets the rounding of the species sums.
 *
 * Limit: the small bucket table covers up to 13 names; above that the caller keeps the order of first appearance.
 */

// the MurmurHash64A multiplier (built from a string: CodeQL reads the hex bigint literal as undefined)
const M = BigInt('0xc6a4a7935bd1e995');
const R = 47n;

/** libstdc++ _Hash_bytes (64-bit size_t), the hash of std::hash<std::string>. */
export const hashBytes = (s: string, seed = 0xc70f6907n): bigint => {
  const buf = new TextEncoder().encode(s);
  const len = buf.length;
  let h = BigInt.asUintN(64, seed ^ BigInt.asUintN(64, BigInt(len) * M));
  const blocks = len >> 3;
  for (let b = 0; b < blocks; b++) {
    let k = 0n;
    for (let j = 7; j >= 0; j--) k = (k << 8n) | BigInt(buf[b * 8 + j]);
    k = BigInt.asUintN(64, k * M);
    k ^= k >> R;
    k = BigInt.asUintN(64, k * M);
    h ^= k;
    h = BigInt.asUintN(64, h * M);
  }
  const rem = len & 7;
  if (rem > 0) {
    const t = blocks * 8;
    for (let j = rem - 1; j >= 1; j--) h ^= BigInt(buf[t + j]) << BigInt(8 * j);
    h ^= BigInt(buf[t]);
    h = BigInt.asUintN(64, h * M);
  }
  h ^= h >> R;
  h = BigInt.asUintN(64, h * M);
  h ^= h >> R;
  return h;
};

/** __fast_bkt of libstdc++ hashtable_c++0x.cc: the bucket counts of the first growth steps. */
const FAST_BKT = [2, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13];

interface Node {
  h: bigint;
  value: string;
  next: Node | null;
}

/** Iteration order of an unordered_set after inserting the names in order (duplicates are no-ops). */
export const unorderedSetOrder = (names: readonly string[]): string[] => {
  const before: { next: Node | null } = { next: null };
  let nBkt = 1;
  let nextResize = 0;
  // buckets[k] is the node before the first node of bucket k (before for the sentinel), or null when empty
  type Before = { next: Node | null } | Node;
  let buckets: (Before | null)[] = [null];
  let nElt = 0;
  const seen = new Set<string>();
  const nextBkt = (n: number): number => {
    nextResize = Math.floor(FAST_BKT[n]);
    return FAST_BKT[n];
  };
  for (const name of names) {
    if (seen.has(name)) continue;
    seen.add(name);
    const h = hashBytes(name);
    // _Prime_rehash_policy::_M_need_rehash(n_bkt, n_elt, 1)
    if (nElt + 1 > nextResize) {
      // libstdc++ _M_need_rehash: the first growth starts from 11 expected elements
      const minBkts = Math.max(nElt + 1, nextResize ? 0 : 11);
      if (minBkts >= nBkt) {
        const want = Math.max(Math.floor(minBkts) + 1, nBkt * 2);
        if (want >= FAST_BKT.length) throw new Error('unordered_set order: more names than the bucket table covers');
        nBkt = nextBkt(want);
        // _M_rehash_aux (unique keys): the list is rebuilt bucket by bucket
        const newBuckets: (Before | null)[] = new Array(nBkt).fill(null);
        let p = before.next;
        before.next = null;
        let bbegin = -1;
        while (p) {
          const next: Node | null = p.next;
          const bkt = Number(p.h % BigInt(nBkt));
          if (!newBuckets[bkt]) {
            p.next = before.next;
            before.next = p;
            newBuckets[bkt] = before;
            if (p.next) newBuckets[bbegin] = p;
            bbegin = bkt;
          } else {
            const bb = newBuckets[bkt] as Node | { next: Node | null };
            p.next = bb.next;
            bb.next = p;
          }
          p = next;
        }
        buckets = newBuckets;
      } else {
        nextResize = Math.floor(nBkt);
      }
    }
    // _M_insert_bucket_begin
    const node: Node = { h, value: name, next: null };
    const bkt = Number(h % BigInt(nBkt));
    const cur = buckets[bkt];
    if (cur) {
      node.next = (cur as { next: Node | null }).next;
      (cur as { next: Node | null }).next = node;
    } else {
      node.next = before.next;
      before.next = node;
      if (node.next) buckets[Number(node.next.h % BigInt(nBkt))] = node;
      buckets[bkt] = before;
    }
    nElt++;
  }
  const out: string[] = [];
  for (let p = before.next; p; p = p.next) out.push(p.value);
  return out;
};
