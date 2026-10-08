import { EngineHost } from '../engine/host';
import type { ToEngine } from '../engine/protocol';

// The notebook's MD engine runs here so a long `run` never blocks the page.
// Messages are handled in order; `cancel` is delivered while a run yields
// between steps (Session.run yields to the event loop every few steps).
const host = new EngineHost((msg, transfer) => (self as unknown as Worker).postMessage(msg, transfer ?? []));

self.onmessage = (ev: MessageEvent<ToEngine>) => {
  void host.handle(ev.data);
};
