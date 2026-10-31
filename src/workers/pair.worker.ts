import { PairThreadWorker, type PairThreadMessage } from '../engine/cpu/pairThreadsCore';
import { loadThreadedPair } from '../engine/cpu/threadedPairLoaders';

// One pair-force thread of the notebook engine (see src/engine/cpu/pairThreads.ts).
// each style's module is loaded when the engine first sends that style (cpu/threadedPairLoaders.ts)
// a GPL style's module may import the engine worker's chunk (engine.worker.ts): it checks this mark and stays inert
(self as unknown as { lammpsPairWorker?: boolean }).lammpsPairWorker = true;
const worker = new PairThreadWorker(loadThreadedPair);
self.onmessage = (ev: MessageEvent<PairThreadMessage>) => worker.handle(ev.data);
