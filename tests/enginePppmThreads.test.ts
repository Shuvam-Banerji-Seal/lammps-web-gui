import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import { PairThreadWorker, type PairThreadMessage, type PairWorkerLike } from '../src/engine/cpu/pairThreadsCore';
import { THREADED_PAIRS } from '../src/engine/cpu/threadedPairFactories';

/*
 * PPPM's per-atom loops on the pair threads (force/kspace/pppmKernel.ts, cpu/pairThreads.ts pppmWeights / pppmField):
 * each atom's weights and field force come from the same code whichever thread computes them, so a run with threads
 * equals the run without them bit for bit. The pair style here is not a threaded one (lj/spica/coul/long), so the
 * mesh loops are the only threaded part. The workers run in-process, as in enginePairThreads.test.ts.
 */

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  readonly worker = new PairThreadWorker(THREADED_PAIRS);
  pppm = 0;
  postMessage(msg: unknown): void {
    const m = msg as PairThreadMessage;
    if (m.type === 'pppm') this.pppm++;
    this.worker.handle(m);
  }
  terminate(): void {}
}

const INPUT = (kspace: string) => `units real
atom_style charge
lattice sc 3.1
region box block 0 9 0 9 0 9
create_box 2 box
create_atoms 1 box
mass * 18.0
set group all type/fraction 2 0.5 4321
set type 1 charge 0.42
set type 2 charge -0.42
displace_atoms all random 0.3 0.3 0.3 8191
velocity all create 300.0 87287 loop geom
pair_style lj/spica/coul/long 8.0
pair_coeff * * lj9_6 0.15 3.0
${kspace}
neighbor 1.0 bin
fix 1 all nve
timestep 1.0
thermo_style custom step pe ecoul elong press pxx pyz
thermo_modify format float %.17g
thermo 5
run 20
write_dump all custom out.dump id x y z fx fy fz modify sort id format float %.17g
`;

const run = async (kspace: string, threads: number) => {
  const workers: InProcessWorker[] = [];
  const backend = threads > 1 ? new SharedThreadsBackend(threads, () => { const w = new InProcessWorker(); workers.push(w); return w; }) : undefined;
  if (backend) backend.pairThreads.minAtoms = 1;
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => files.set(n, t) }, backend);
  await s.execute(INPUT(kspace));
  backend?.dispose();
  const err = events.find((e) => e.kind === 'error');
  if (err) throw new Error(JSON.stringify(err));
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, dump: files.get('out.dump'), pppmCalls: workers.reduce((a, w) => a + w.pppm, 0) };
};

describe('PPPM on the pair threads', () => {
  for (const kspace of ['kspace_style pppm 1e-5', 'kspace_style pppm 1e-4\nkspace_modify order 4', 'kspace_style pppm/cg 1e-5']) {
    it(`${kspace.replaceAll('\n', ', ')}: the same run, bit for bit, with the mesh loops on 3 threads`, async () => {
      const serial = await run(kspace, 1);
      const threaded = await run(kspace, 3);
      // the weights and the field forces of every step ran on the workers (two loops per force evaluation)
      expect(threaded.pppmCalls).toBeGreaterThan(40);
      expect(threaded.rows).toEqual(serial.rows);
      expect(threaded.dump).toBe(serial.dump);
    });
  }
});
