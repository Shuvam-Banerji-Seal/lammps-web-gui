import { PairThreadWorker, type PairThreadMessage } from '../engine/cpu/pairThreadsCore';
import { loadThreadedPair } from '../engine/cpu/threadedPairLoaders';

// One pair-force thread of the notebook engine (see src/engine/cpu/pairThreads.ts).
// each style's module is loaded when the engine first sends that style (cpu/threadedPairLoaders.ts)
const worker = new PairThreadWorker(loadThreadedPair);
self.onmessage = (ev: MessageEvent<PairThreadMessage>) => worker.handle(ev.data);
