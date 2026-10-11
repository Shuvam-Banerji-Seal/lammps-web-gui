/**
 * Trajectory preparation for the notebook's analysis charts. Turns the recorded frames (TrajectoryRecord:
 * wrapped float32 positions and a box per frame) into TrajectoryFrame[] that the existing analysis functions
 * (services/trajectoryAnalysis) can take, with LAMMPS-style image flags counted from the recorded positions:
 * a coordinate that jumps by more than half a box between two recorded frames is taken to have crossed a
 * periodic face, so the unwrapped position is wrapped + image * L (docs.lammps.org/dump.html, "A value of 2 means
 * add 2 box lengths"). MSD then uses exact displacements, as for an `ix iy iz` dump.
 * Limits, reported in the UI: an atom that moves more than half a box between two recorded frames is unwrapped
 * wrongly; triclinic cells use the orthogonal crossing test (approximate). Pure: no React, no DOM.
 */
import type { Atom, BoxBounds, TrajectoryFrame } from '../../../../types';
import type { TrajectoryRecord } from '../trajectory';

export interface PreparedTrajectory {
  /** Sampled frames (uniform in recorded frames), atoms sampled uniformly by index, image flags counted. */
  frames: TrajectoryFrame[];
  /** Recorded frames between two consecutive sampled frames (the MSD lag unit). */
  frameStride: number;
  atomsTotal: number;
  atomsUsed: number;
  sampledFrames: boolean;
  sampledAtoms: boolean;
  recorded: number;
  /** Mean steps between consecutive recorded frames, or null with fewer than two. */
  stepsPerRecord: number | null;
  triclinic: boolean;
}

export const emptyPrepared = (): PreparedTrajectory => ({
  frames: [], frameStride: 1, atomsTotal: 0, atomsUsed: 0, sampledFrames: false, sampledAtoms: false,
  recorded: 0, stepsPerRecord: null, triclinic: false,
});

const boxOf = (r: TrajectoryRecord): BoxBounds => ({
  xlo: r.box.lo[0], xhi: r.box.hi[0],
  ylo: r.box.lo[1], yhi: r.box.hi[1],
  zlo: r.box.lo[2], zhi: r.box.hi[2],
  xy: r.box.tilt[0], xz: r.box.tilt[1], yz: r.box.tilt[2],
});

/**
 * Samples and unwraps the records. `maxFrames` and `maxAtoms` bound the work: the frames are every
 * ceil(R / maxFrames)-th recorded frame, the atoms every ceil(N / maxAtoms)-th atom index (a uniform random-like
 * subset of a melt, which keeps g(r), density and MSD unbiased in their normalised forms).
 */
export const prepareTrajectory = (
  records: readonly TrajectoryRecord[], opts: { maxFrames: number; maxAtoms: number },
): PreparedTrajectory => {
  const R = records.length;
  if (R === 0) return emptyPrepared();
  const first = records[0];
  const N = first.topo.id.length;
  const atomStride = Math.max(1, Math.ceil(N / Math.max(1, opts.maxAtoms)));
  const frameStride = Math.max(1, Math.ceil(R / Math.max(1, opts.maxFrames)));
  const slots: number[] = [];
  for (let k = 0; k < N; k += atomStride) slots.push(k);
  const S = slots.length;
  const slotIds = Int32Array.from(slots, (k) => first.topo.id[k]);
  const counts = new Float64Array(3 * S);
  const last = new Float64Array(3 * S).fill(NaN);
  const frames: TrajectoryFrame[] = [];
  let lastTopo = first.topo;
  let mapIdx = Int32Array.from(slots); // record atom index for each slot (-1 = absent)
  let triclinic = false;

  const stepsFirst = first.step, stepsLast = records[R - 1].step;
  for (let r = 0; r < R; r++) {
    const rec = records[r];
    if (rec.box.triclinic) triclinic = true;
    const n = rec.topo.id.length;
    // Slot -> atom index in this record. The same topology object, or the same ids at the sampled slots,
    // means the identity map; otherwise look the ids up once for this record.
    if (rec.topo !== lastTopo) {
      lastTopo = rec.topo;
      const identity = n === N && slots.every((k, s) => rec.topo.id[k] === slotIds[s]);
      if (identity) {
        mapIdx = Int32Array.from(slots);
      } else {
        const where = new Map<number, number>();
        for (let s = 0; s < S; s++) where.set(slotIds[s], -1);
        for (let k = 0; k < n; k++) if (where.has(rec.topo.id[k])) where.set(rec.topo.id[k], k);
        mapIdx = Int32Array.from(slotIds, (id) => where.get(id) ?? -1);
      }
    }
    const { lo, hi } = rec.box;
    const half = [0, 1, 2].map((d) => (hi[d] - lo[d]) / 2);
    const L = [0, 1, 2].map((d) => hi[d] - lo[d]);
    for (let s = 0; s < S; s++) {
      const j = mapIdx[s];
      if (j < 0 || j >= n) { last[3 * s] = NaN; continue; }
      for (let d = 0; d < 3; d++) {
        const v = rec.x[3 * j + d];
        const prev = last[3 * s + d];
        if (!Number.isNaN(prev) && L[d] > 0) {
          const dx = v - prev;
          if (dx > half[d]) counts[3 * s + d] -= 1;
          else if (dx < -half[d]) counts[3 * s + d] += 1;
        }
        last[3 * s + d] = v;
      }
    }
    if (r % frameStride === 0) {
      const atoms: Atom[] = new Array(S);
      for (let s = 0; s < S; s++) {
        const j = mapIdx[s];
        const ok = j >= 0 && j < n;
        atoms[s] = {
          id: slotIds[s], molId: 0, type: ok ? rec.topo.type[j] : 0, charge: 0,
          x: ok ? rec.x[3 * j] : NaN, y: ok ? rec.x[3 * j + 1] : NaN, z: ok ? rec.x[3 * j + 2] : NaN,
          ix: counts[3 * s], iy: counts[3 * s + 1], iz: counts[3 * s + 2],
        };
      }
      frames.push({ atoms, box: boxOf(rec) });
    }
  }
  // Atoms missing from a frame carry NaN positions; drop them from every frame so the MSD sees whole frames.
  const complete = frames.map((f) => ({ ...f, atoms: f.atoms.filter((a) => Number.isFinite(a.x)) }));
  const used = complete.length ? complete[0].atoms.length : 0;
  return {
    frames: complete,
    frameStride,
    atomsTotal: N,
    atomsUsed: used,
    sampledFrames: frameStride > 1,
    sampledAtoms: atomStride > 1,
    recorded: R,
    stepsPerRecord: R > 1 ? (stepsLast - stepsFirst) / (R - 1) : null,
    triclinic,
  };
};

/** Per recorded frame: step, box volume and atom count (all records, no sampling). */
export const boxAndCountSeries = (
  records: readonly TrajectoryRecord[],
): { step: number; volume: number; atoms: number }[] =>
  records.map((r) => {
    const { lo, hi } = r.box;
    const volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
    return { step: r.step, volume, atoms: r.topo.id.length };
  });
