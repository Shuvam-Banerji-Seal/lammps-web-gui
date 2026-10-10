import { describe, expect, it } from 'vitest';
import {
  FileStoreError,
  createIndexedDbFileStore,
  createMemoryFileStore,
  kindOf,
  totalBytes,
} from '../src/services/fileStore';

/*
 * Input-file store of the notebook (src/services/fileStore.ts). The IndexedDB path needs a browser;
 * jsdom has no indexedDB, so these tests cover the in-memory fallback that the store uses there.
 */

describe('file store (memory fallback)', () => {
  it('puts, gets, lists sorted by name, removes and clears', async () => {
    const store = createMemoryFileStore();
    expect(store.backend).toBe('memory');
    await store.put({ name: 'zeta.data', text: 'abc', size: 3 });
    await store.put({ name: 'alpha.pdb', text: 'PDB', size: 3 });
    await store.put({ name: 'lj.gz', text: '\u001f\u008b', size: 2 });

    expect((await store.get('zeta.data'))?.text).toBe('abc');
    expect(await store.get('missing')).toBeUndefined();
    expect((await store.list()).map((f) => f.name)).toEqual(['alpha.pdb', 'lj.gz', 'zeta.data']);

    await store.remove('lj.gz');
    expect((await store.list()).map((f) => f.name)).toEqual(['alpha.pdb', 'zeta.data']);

    await store.clear();
    expect(await store.list()).toEqual([]);
  });

  it('replaces a file of the same name and keeps size, kind and mtime', async () => {
    const store = createMemoryFileStore();
    await store.put({ name: 'x.data', text: 'one', size: 3, mtime: 100 });
    await store.put({ name: 'x.data', text: 'three', size: 5, mtime: 200 });
    const f = await store.get('x.data');
    expect(f).toMatchObject({ name: 'x.data', text: 'three', size: 5, mtime: 200, kind: 'text' });
    expect(await store.list()).toHaveLength(1);
  });

  it('sums the stored sizes', async () => {
    const store = createMemoryFileStore();
    await store.put({ name: 'a.data', text: '12345', size: 5 });
    await store.put({ name: 'b.pdb', text: '1234567890', size: 10 });
    expect(totalBytes(await store.list())).toBe(15);
    await store.remove('a.data');
    expect(totalBytes(await store.list())).toBe(10);
  });

  it('reports a full store as a quota error and keeps the files it has', async () => {
    const store = createMemoryFileStore(10);
    await store.put({ name: 'a.data', text: 'x'.repeat(6), size: 6 });
    await expect(store.put({ name: 'b.data', text: 'y'.repeat(6), size: 6 })).rejects.toBeInstanceOf(FileStoreError);
    await expect(store.put({ name: 'b.data', text: 'y'.repeat(6), size: 6 })).rejects.toMatchObject({ code: 'quota' });
    expect((await store.list()).map((f) => f.name)).toEqual(['a.data']);
    // replacing a file counts only its new size against the limit
    await store.put({ name: 'a.data', text: 'z'.repeat(9), size: 9 });
    expect((await store.get('a.data'))?.size).toBe(9);
  });

  it('names the kind of a file from its extension', () => {
    expect(kindOf('lj.data.gz')).toBe('gzip');
    expect(kindOf('protein.PDB')).toBe('structure');
    expect(kindOf('cell.cif')).toBe('structure');
    expect(kindOf('water.xyz')).toBe('structure');
    expect(kindOf('in.potential')).toBe('text');
  });
});

describe('file store without indexedDB', () => {
  it('falls back to memory when there is no IndexedDB factory', async () => {
    for (const factory of [null, undefined]) {
      const store = createIndexedDbFileStore(factory);
      expect(store.backend).toBe('memory');
      await store.put({ name: 'x.data', text: 'hello', size: 5 });
      expect((await store.get('x.data'))?.text).toBe('hello');
      expect((await store.list()).map((f) => f.name)).toEqual(['x.data']);
      await store.remove('x.data');
      expect(await store.list()).toEqual([]);
    }
  });
});
