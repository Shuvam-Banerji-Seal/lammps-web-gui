/**
 * Trajectory buffer for the MD notebook: every frame the engine streams during runs is kept
 * (positions as Float32Array; ids and types shared between frames with the same atoms), under a memory cap.
 * Over the cap the buffer drops every other frame and doubles the recording stride, and says so.
 * Also the writers for extended XYZ and LAMMPS dump text. Pure: no React, no DOM.
 */
import type { SimBox } from '../../../engine/types';
import type { FrameEvent } from '../../../engine/view';

/** Default memory cap for recorded positions and topology (bytes). */
export const DEFAULT_CAP_BYTES = 256 * 1024 * 1024;

/** Rough per-frame bookkeeping cost (step, box, object headers) added to the position bytes. */
const FRAME_OVERHEAD = 256;

/** Atom ids and types, shared by consecutive frames whose atoms are the same. */
export interface Topology {
  id: Int32Array;
  type: Int32Array;
}

export interface TrajectoryRecord {
  step: number;
  /** Wrapped positions, x0 y0 z0 x1 ... (float32: enough for viewing and export). */
  x: Float32Array;
  box: SimBox;
  topo: Topology;
}

const sameInts = (a: Int32Array, b: Int32Array): boolean => {
  if (a.length !== b.length) return false;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false;
  return true;
};

export interface TrajectoryBufferOptions {
  capBytes?: number;
}

export class TrajectoryBuffer {
  private frames: TrajectoryRecord[] = [];
  private seen = 0;
  private strideValue = 1;
  private thinnings = 0;
  private topoBytes = 0;
  private topoCounted = new Map<Topology, number>();
  private bytesValue = 0;
  private readonly capBytes: number;

  constructor(options: TrajectoryBufferOptions = {}) {
    this.capBytes = options.capBytes ?? DEFAULT_CAP_BYTES;
  }

  /** Frames kept. */
  get length(): number {
    return this.frames.length;
  }

  /** Bytes held (positions, bookkeeping and distinct topologies). */
  get bytes(): number {
    return this.bytesValue + this.topoBytes;
  }

  /** Recording stride: 1 keeps every frame; 2 after the first thinning, and so on. */
  get stride(): number {
    return this.strideValue;
  }

  /** How many times the buffer dropped every other frame to stay under the cap. */
  get thinned(): number {
    return this.thinnings;
  }

  get cap(): number {
    return this.capBytes;
  }

  /** Empties the buffer and restores the full recording rate. */
  clear(): void {
    this.frames = [];
    this.seen = 0;
    this.strideValue = 1;
    this.thinnings = 0;
    this.bytesValue = 0;
    this.topoBytes = 0;
    this.topoCounted = new Map();
  }

  /**
   * Keeps a copy of an engine frame. Returns false when the stride skipped it.
   * The engine's arrays are copied, so the engine may reuse them.
   */
  append(frame: FrameEvent): boolean {
    const idx = this.seen;
    this.seen++;
    if (idx % this.strideValue !== 0) return false;
    const last = this.frames[this.frames.length - 1];
    const topo: Topology = last && sameInts(last.topo.id, frame.id) && sameInts(last.topo.type, frame.type)
      ? last.topo
      : { id: Int32Array.from(frame.id), type: Int32Array.from(frame.type) };
    const x = Float32Array.from(frame.x);
    this.frames.push({ step: frame.step, x, box: frame.box, topo });
    this.bytesValue += x.byteLength + FRAME_OVERHEAD;
    this.countTopo(topo, 1);
    if (this.bytes > this.capBytes && this.frames.length > 1) this.thin();
    return true;
  }

  /** The frame at index i (0 = first recorded). */
  get(i: number): TrajectoryRecord | undefined {
    return this.frames[i];
  }

  /** Every recorded frame, in order (the array is a copy). */
  list(): TrajectoryRecord[] {
    return this.frames.slice();
  }

  /** Index of the frame whose step is nearest to `step` (the first one on ties); -1 when empty. */
  indexOfStep(step: number): number {
    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < this.frames.length; i++) {
      const d = Math.abs(this.frames[i].step - step);
      if (d < bestDist) {
        best = i;
        bestDist = d;
      }
    }
    return best;
  }

  /** The frame at index i as the engine's frame event (what frameToMoleculeData takes). */
  toFrameEvent(i: number): FrameEvent | null {
    const r = this.frames[i];
    if (!r) return null;
    return {
      kind: 'frame',
      step: r.step,
      x: Float64Array.from(r.x),
      image: new Int32Array(r.x.length),
      type: r.topo.type,
      id: r.topo.id,
      box: r.box,
    };
  }

  private countTopo(topo: Topology, delta: number): void {
    const n = (this.topoCounted.get(topo) ?? 0) + delta;
    if (delta > 0 && n === delta) this.topoBytes += topo.id.byteLength + topo.type.byteLength;
    if (n <= 0) {
      this.topoCounted.delete(topo);
      this.topoBytes -= topo.id.byteLength + topo.type.byteLength;
    } else this.topoCounted.set(topo, n);
  }

  /** Drops every other frame (keeps positions 0, 2, 4, ...) and doubles the stride. */
  private thin(): void {
    const kept: TrajectoryRecord[] = [];
    this.bytesValue = 0;
    this.topoBytes = 0;
    this.topoCounted = new Map();
    for (let k = 0; k < this.frames.length; k += 2) {
      const r = this.frames[k];
      kept.push(r);
      this.bytesValue += r.x.byteLength + FRAME_OVERHEAD;
      this.countTopo(r.topo, 1);
    }
    this.frames = kept;
    this.strideValue *= 2;
    this.thinnings++;
  }
}

/** Shortest decimal that reads back the float32 value (keeps files small). */
const num = (v: number): string => String(Number(v.toPrecision(9)));

const PBC = (periodic: [boolean, boolean, boolean]): string => periodic.map((p) => (p ? 'T' : 'F')).join(' ');

/**
 * Extended XYZ, one block per frame. The comment line carries Lattice (the box vectors, tilt included), Origin
 * (the box's lower corner), Properties (species, pos, type, id), Step and pbc. The species token is `T<type>` so
 * each engine type reads as a separate species.
 */
export const framesToExtXYZ = (frames: readonly TrajectoryRecord[]): string => {
  const out: string[] = [];
  for (const f of frames) {
    const n = f.topo.id.length;
    const { lo, hi, tilt, periodic } = f.box;
    const lx = hi[0] - lo[0];
    const ly = hi[1] - lo[1];
    const lz = hi[2] - lo[2];
    const [xy, xz, yz] = tilt;
    const lattice = `${num(lx)} 0 0 ${num(xy)} ${num(ly)} 0 ${num(xz)} ${num(yz)} ${num(lz)}`;
    const origin = `${num(lo[0])} ${num(lo[1])} ${num(lo[2])}`;
    out.push(String(n));
    out.push(
      `Lattice="${lattice}" Origin="${origin}" Properties=species:S:1:pos:R:3:type:I:1:id:I:1 ` +
        `Step=${f.step} pbc="${PBC(periodic)}"`,
    );
    for (let i = 0; i < n; i++) {
      const t = f.topo.type[i];
      out.push(`T${t} ${num(f.x[3 * i])} ${num(f.x[3 * i + 1])} ${num(f.x[3 * i + 2])} ${t} ${f.topo.id[i]}`);
    }
  }
  return out.join('\n') + '\n';
};

/**
 * LAMMPS dump atom text (dump_modify format default style), one block per frame, columns id type x y z.
 * A tilted box uses the documented "ITEM: BOX BOUNDS xy xz yz" bounding box, the inverse of the one dumpParser reads.
 */
export const framesToDump = (frames: readonly TrajectoryRecord[]): string => {
  const out: string[] = [];
  for (const f of frames) {
    const n = f.topo.id.length;
    const { lo, hi, tilt, periodic } = f.box;
    const [xy, xz, yz] = tilt;
    // boundary flags are two letters per dimension in the dump header: pp (periodic) or ff (fixed)
    const pbc = periodic.map((p) => (p ? 'pp' : 'ff')).join(' ');
    out.push('ITEM: TIMESTEP', String(f.step), 'ITEM: NUMBER OF ATOMS', String(n));
    if (xy !== 0 || xz !== 0 || yz !== 0) {
      const xmin = Math.min(0, xy, xz, xy + xz);
      const xmax = Math.max(0, xy, xz, xy + xz);
      const ymin = Math.min(0, yz);
      const ymax = Math.max(0, yz);
      out.push(
        `ITEM: BOX BOUNDS xy xz yz ${pbc}`,
        `${num(lo[0] + xmin)} ${num(hi[0] + xmax)} ${num(xy)}`,
        `${num(lo[1] + ymin)} ${num(hi[1] + ymax)} ${num(xz)}`,
        `${num(lo[2])} ${num(hi[2])} ${num(yz)}`,
      );
    } else {
      out.push(
        `ITEM: BOX BOUNDS ${pbc}`,
        `${num(lo[0])} ${num(hi[0])}`,
        `${num(lo[1])} ${num(hi[1])}`,
        `${num(lo[2])} ${num(hi[2])}`,
      );
    }
    out.push('ITEM: ATOMS id type x y z');
    for (let i = 0; i < n; i++) {
      out.push(`${f.topo.id[i]} ${f.topo.type[i]} ${num(f.x[3 * i])} ${num(f.x[3 * i + 1])} ${num(f.x[3 * i + 2])}`);
    }
  }
  return out.join('\n') + '\n';
};
