import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import { PairThreadWorker, type PairThreadMessage, type PairWorkerLike } from '../src/engine/cpu/pairThreadsCore';
import type { NeighList } from '../src/engine/neighbor';

/*
 * The pair search of the common half neighbor list on the shared-memory threads (cpu/pairThreads.ts halfList,
 * cpu/neighborKernel.ts): the list must hold the same entries in the same order as the one-thread build (the forces
 * then differ only by the order of the threaded pair term's additions, see enginePairThreads.test.ts). The workers
 * run in-process (each message is handled at once), so the first worker claims every chunk and runs out of room, and
 * the engine thread searches the chunks it could not take: both paths are exercised.
 */

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  readonly worker = new PairThreadWorker();
  neighs = 0;
  postMessage(msg: unknown): void {
    const m = msg as PairThreadMessage;
    if (m.type === 'neigh') this.neighs++;
    this.worker.handle(m);
  }
  terminate(): void {}
}

const checksum = (L: NeighList | null): number => {
  if (!L) return 0;
  let h = 0;
  for (let i = 0; i < L.inum; i++) {
    h = (h * 31 + L.numneigh[i]) | 0;
    for (let k = L.firstneigh[i]; k < L.firstneigh[i] + L.numneigh[i]; k++) h = (h * 31 + L.neighbors[k]) | 0;
  }
  return h;
};

const SYSTEMS: Record<string, string> = {
  'two types, orthogonal box': 'units lj\natom_style atomic\nlattice fcc 0.8442\nregion box block 0 6 0 6 0 6\ncreate_box 2 box\ncreate_atoms 1 box\nset type 1 type/fraction 2 0.3 7\nmass * 1.0\nvelocity all create 3.0 87287 loop geom\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 1.0 1.0 2.8\n',
  'triclinic box': 'units lj\natom_style atomic\nlattice fcc 0.8442\nregion box prism 0 6 0 6 0 6 1.3 0.6 -0.9\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nvelocity all create 3.0 87287 loop geom\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\n',
  'shrink-wrapped box, a type pair without interaction': 'units lj\natom_style atomic\nboundary p s p\nlattice fcc 0.8442\nregion box block 0 6 0 6 0 6\ncreate_box 2 box\ncreate_atoms 1 box\nset type 1 type/fraction 2 0.5 7\nmass * 1.0\nvelocity all create 3.0 87287 loop geom\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 1.0 1.0 0.0\n',
  '2d': 'units lj\ndimension 2\natom_style atomic\nlattice sq2 0.8442\nregion box block 0 20 0 20 -0.5 0.5\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nvelocity all create 3.0 87287 loop geom\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix 2 all enforce2d\n',
};

const build = async (text: string, threads: number): Promise<{ sums: number[]; neighs: number }> => {
  const workers: InProcessWorker[] = [];
  const backend = threads > 1 ? new SharedThreadsBackend(threads, () => { const w = new InProcessWorker(); workers.push(w); return w; }) : undefined;
  if (backend) backend.pairThreads.minAtoms = 1;
  const s = new Session({ emit: () => {}, writeFile: () => {} }, backend);
  await s.execute(`${text}neighbor 0.3 bin\nfix 1 all nve\nthermo_style custom step pe press\nthermo 10\nrun 30\n`);
  const nb = (s as unknown as { sys: { nb: { half: NeighList | null; full: NeighList | null } } }).sys.nb;
  return { sums: [checksum(nb.half), checksum(nb.full)], neighs: workers.reduce((a, w) => a + w.neighs, 0) };
};

describe('threaded neighbor search: the same list as one thread', () => {
  for (const [name, text] of Object.entries(SYSTEMS)) {
    it(name, async () => {
      const one = await build(text, 1);
      const four = await build(text, 4);
      expect(four.neighs).toBeGreaterThan(0);
      expect(four.sums).toEqual(one.sums);
    });
  }
});
