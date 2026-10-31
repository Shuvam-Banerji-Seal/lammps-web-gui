import { EngineHost } from '../engine/host';
import type { ToEngine } from '../engine/protocol';

// The notebook's MD engine runs here so a long `run` never blocks the page.
// Messages are handled in order; `cancel` is delivered while a run yields
// between steps (Session.run yields to the event loop every few steps).
// A pair worker can import this chunk: a GPL-ported pair style's module (lazyStyles.ts loadGplThreadedPair, imported by
// URL in cpu/threadedPairLoaders.ts) may need engine code the bundler placed here. It marks itself first
// (pair.worker.ts), and must not become an engine host, which would take over its messages.
if (!(self as unknown as { lammpsPairWorker?: boolean }).lammpsPairWorker) {
  const host = new EngineHost((msg, transfer) => (self as unknown as Worker).postMessage(msg, transfer ?? []));
  self.onmessage = (ev: MessageEvent<ToEngine>) => {
    void host.handle(ev.data);
  };
}
