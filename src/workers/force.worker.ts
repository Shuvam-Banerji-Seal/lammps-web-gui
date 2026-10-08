import { computeCellRange, type RangeTask } from '../engine/cpu/rangeKernel';

// One CPU force thread of the notebook engine: computes the forces on the
// atoms of a range of cells (see src/engine/cpu/parallel.ts).
self.onmessage = (ev: MessageEvent<{ id: number; task: RangeTask }>) => {
  const { id, task } = ev.data;
  const r = computeCellRange(task);
  (self as unknown as Worker).postMessage({ id, ...r }, [r.f.buffer, r.ghostSlots.buffer, r.ghostF.buffer]);
};
