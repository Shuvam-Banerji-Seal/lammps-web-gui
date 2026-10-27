import { PairThreadWorker, type PairThreadMessage } from '../engine/cpu/pairThreadsCore';
import { THREADED_PAIRS } from '../engine/cpu/threadedPairFactories';

// One pair-force thread of the notebook engine (see src/engine/cpu/pairThreads.ts).
const worker = new PairThreadWorker(THREADED_PAIRS);
self.onmessage = (ev: MessageEvent<PairThreadMessage>) => worker.handle(ev.data);
