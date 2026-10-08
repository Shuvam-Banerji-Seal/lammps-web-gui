import { PairThreadWorker, type PairThreadMessage } from '../engine/cpu/pairThreadsCore';

// One pair-force thread of the notebook engine (see src/engine/cpu/pairThreads.ts).
const worker = new PairThreadWorker();
self.onmessage = (ev: MessageEvent<PairThreadMessage>) => worker.handle(ev.data);
