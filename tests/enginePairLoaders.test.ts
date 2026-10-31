import { describe, expect, it } from 'vitest';
import { THREADED_NAMES } from '../src/engine/cpu/threadedPairs';
import { THREADED_PAIRS } from '../src/engine/cpu/threadedPairFactories';
import { loadThreadedPair } from '../src/engine/cpu/threadedPairLoaders';
import { CTL_WORDS, PAIR_READY, PairThreadWorker, cloneableFields, type PairLoader } from '../src/engine/cpu/pairThreadsCore';
import type { Pair } from '../src/engine/force/types';
import { GPL_THREADED_PAIRS, loadGplThreadedPair } from '../src/engine/lazyStyles';
import { THREADED as GPL_THREADED } from '../src/engine-gpl/registry/threaded_pairs';

/*
 * The pair worker loads each style's module when the engine first sends it (cpu/threadedPairLoaders.ts), and counts
 * itself in PAIR_READY once the style is built; the engine runs the pair term itself until every worker has.
 */

describe('pair worker style loaders', () => {
  it('build the same class as the in-process factories for every threaded style', async () => {
    for (const name of THREADED_NAMES) {
      // a GPL port comes with its registry module's URL, which the engine thread gets from the lazy registries
      const url = GPL_THREADED_PAIRS.includes(name) ? (await loadGplThreadedPair(name))!.url : undefined;
      const make = await loadThreadedPair(name, url);
      expect(make, name).not.toBeNull();
      // a GPL port: the class of the GPL registry the worker loads through lazyStyles.ts
      const ref = GPL_THREADED_PAIRS.includes(name) ? GPL_THREADED[name] : THREADED_PAIRS[name];
      expect(make!().constructor, name).toBe(ref().constructor);
    }
    expect(await loadThreadedPair('nm/cut')).toBeNull();
  });

  it('count a worker in PAIR_READY only once its style is built, and ignore a stale load', async () => {
    const ctl = new Int32Array(new SharedArrayBuffer(4 * CTL_WORDS));
    const releases: (() => void)[] = [];
    // a loader whose loads finish when the test says so
    const loader: PairLoader = (name) => new Promise((resolve) => {
      releases.push(() => resolve(() => THREADED_PAIRS[name]()));
    });
    const w = new PairThreadWorker(loader);
    w.handle({ type: 'init', ctl: ctl.buffer as SharedArrayBuffer, index: 0 });
    const fields = (p: Pair) => cloneableFields(p);
    w.handle({ type: 'pair', name: 'lj/cut', fields: fields(THREADED_PAIRS['lj/cut']()) });
    w.handle({ type: 'pair', name: 'morse', fields: fields(THREADED_PAIRS.morse()) });
    await Promise.resolve();
    expect(Atomics.load(ctl, PAIR_READY)).toBe(0);
    // the older (lj/cut) load finishing late does not count: only the last style sent does
    releases[0]();
    await new Promise((r) => setTimeout(r, 0));
    expect(Atomics.load(ctl, PAIR_READY)).toBe(0);
    releases[1]();
    await new Promise((r) => setTimeout(r, 0));
    expect(Atomics.load(ctl, PAIR_READY)).toBe(1);
  });
});
