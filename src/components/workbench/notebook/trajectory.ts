/**
 * Trajectory buffer for the MD notebook: every frame the engine streams during runs is kept
 * (positions as Float32Array; ids and types shared between frames with the same atoms), under a memory cap.
 * Over the cap the buffer drops every other frame and doubles the recording stride, and says so.
 * Also the writers for extended XYZ, LAMMPS dump, multi-model PDB, LAMMPS data (one frame) and CSV text.
 * Pure: no React, no DOM.
 */
import type { SimBox } from '../../../engine/types';
import type { FrameEvent } from '../../../engine/view';
import type { BoxBounds } from '../../../types';
import { parseDumpFile } from '../../../services/dumpParser';

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
  /**
   * Velocities, vx0 vy0 vz0 vx1 ... (float32), when the engine sent them (frameVelocities) or a dump file has
   * vx vy vz columns. Absent otherwise: the speed views then fall back to displacements or say so.
   */
  v?: Float32Array;
  /**
   * Mass of each atom (float32, one per atom), with v; one array is shared by the records that follow the frame that
   * sent it. Absent for a dump file: its masses are not in the file.
   */
  mass?: Float32Array;
}

/** Bytes a record holds: positions, velocities when present, and the bookkeeping (the shared masses are not counted). */
const recordBytes = (r: TrajectoryRecord): number =>
  r.x.byteLength + (r.v?.byteLength ?? 0) + FRAME_OVERHEAD;

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
  /** The masses of the latest frame that carried them (shared by the records that follow). */
  private lastMass: Float32Array | null = null;
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
    this.lastMass = null;
    this.seen = 0;
    this.strideValue = 1;
    this.thinnings = 0;
    this.bytesValue = 0;
    this.topoBytes = 0;
    this.topoCounted = new Map();
  }

  /**
   * Replaces the contents with the given records (their positions are shared, not copied) and the recording
   * stride. Used to show a loaded file's frames and to put the run's recorded frames back.
   */
  restore(records: readonly TrajectoryRecord[], stride = 1): void {
    this.clear();
    this.strideValue = Math.max(1, Math.floor(stride));
    for (const r of records) {
      this.frames.push(r);
      this.bytesValue += recordBytes(r);
      this.countTopo(r.topo, 1);
    }
    while (this.bytes > this.capBytes && this.frames.length > 1) this.thin();
  }

  /**
   * Keeps a copy of an engine frame. Returns false when the stride skipped it.
   * The engine's arrays are copied, so the engine may reuse them.
   */
  append(frame: FrameEvent): boolean {
    // a frame repeats the masses only when they change: keep the last ones, also from frames the stride skips
    if (frame.mass) this.lastMass = Float32Array.from(frame.mass);
    const idx = this.seen;
    this.seen++;
    if (idx % this.strideValue !== 0) return false;
    const last = this.frames[this.frames.length - 1];
    const topo: Topology = last && sameInts(last.topo.id, frame.id) && sameInts(last.topo.type, frame.type)
      ? last.topo
      : { id: Int32Array.from(frame.id), type: Int32Array.from(frame.type) };
    const x = Float32Array.from(frame.x);
    const record: TrajectoryRecord = { step: frame.step, x, box: frame.box, topo };
    if (frame.v) record.v = Float32Array.from(frame.v);
    if (frame.v && this.lastMass && this.lastMass.length === frame.id.length) record.mass = this.lastMass;
    this.frames.push(record);
    this.bytesValue += recordBytes(record);
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
    const ev: FrameEvent = {
      kind: 'frame',
      step: r.step,
      x: Float64Array.from(r.x),
      image: new Int32Array(r.x.length),
      type: r.topo.type,
      id: r.topo.id,
      box: r.box,
    };
    if (r.v) ev.v = Float64Array.from(r.v);
    if (r.mass) ev.mass = Float64Array.from(r.mass);
    return ev;
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
      this.bytesValue += recordBytes(r);
      this.countTopo(r.topo, 1);
    }
    this.frames = kept;
    this.strideValue *= 2;
    this.thinnings++;
  }
}

const sameTopo = (a: Topology, b: Topology): boolean => sameInts(a.id, b.id) && sameInts(a.type, b.type);

/** A dump cell (BOX BOUNDS, tilts as the file's bounding box is converted by dumpParser) as the buffer's SimBox. */
const simBoxOf = (b: BoxBounds): SimBox => {
  const xy = b.xy ?? 0;
  const xz = b.xz ?? 0;
  const yz = b.yz ?? 0;
  const tilt: [number, number, number] = [xy, xz, yz];
  return {
    lo: [b.xlo, b.ylo, b.zlo],
    hi: [b.xhi, b.yhi, b.zhi],
    periodic: [true, true, true],
    tilt,
    triclinic: tilt.some((t) => t !== 0),
    boundary: [['p', 'p'], ['p', 'p'], ['p', 'p']],
    minLo: [b.xlo, b.ylo, b.zlo],
    minHi: [b.xhi, b.yhi, b.zhi],
  };
};

/**
 * Frames of a LAMMPS dump text as buffer records, in file order, through the same parser as the File import
 * (dumpParser: triclinic bounds, scaled and unwrapped coordinates). The steps come from the ITEM: TIMESTEP lines.
 * Throws the parser's error for a file it cannot read.
 */
export const dumpToRecords = (text: string): TrajectoryRecord[] => {
  const data = parseDumpFile(text);
  const steps = Array.from(text.matchAll(/^ITEM: TIMESTEP[ \t]*\r?\n[ \t]*(-?\d+)/gm), (m) => Number(m[1]));
  const blocks = data.frames ?? [{ atoms: data.atoms, box: data.box }];
  const out: TrajectoryRecord[] = [];
  for (let k = 0; k < blocks.length; k++) {
    const atoms = blocks[k].atoms;
    const n = atoms.length;
    const x = new Float32Array(3 * n);
    const id = new Int32Array(n);
    const type = new Int32Array(n);
    // velocities only when every atom of the frame has vx vy vz (a dump written with them)
    const hasV = n > 0 && atoms.every((a) => a.vx !== undefined && a.vy !== undefined && a.vz !== undefined);
    const v = hasV ? new Float32Array(3 * n) : undefined;
    for (let i = 0; i < n; i++) {
      const a = atoms[i];
      x[3 * i] = a.x;
      x[3 * i + 1] = a.y;
      x[3 * i + 2] = a.z;
      id[i] = a.id;
      type[i] = a.type;
      if (v) {
        v[3 * i] = a.vx ?? 0;
        v[3 * i + 1] = a.vy ?? 0;
        v[3 * i + 2] = a.vz ?? 0;
      }
    }
    const last = out[out.length - 1];
    const topo: Topology = last && sameTopo(last.topo, { id, type }) ? last.topo : { id, type };
    const box = blocks[k].box ?? data.box;
    if (!box) throw new Error('Invalid LAMMPS dump: a frame has no box');
    const record: TrajectoryRecord = { step: Number.isFinite(steps[k]) ? steps[k] : k, x, box: simBoxOf(box), topo };
    if (v) record.v = v;
    out.push(record);
  }
  return out;
};

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

/** Fixed-width number for a PDB column: %w.df as the format spells it, right-justified. */
const pdbNum = (v: number, width: number, decimals: number): string => v.toFixed(decimals).padStart(width);

/**
 * Multi-model PDB, one MODEL / ENDMDL block per frame. A CRYST1 record (box lengths, 90-degree angles, P 1) is
 * written from the first frame when its box is orthogonal; a tilted cell has none. Each atom is a HETATM named
 * T<type> as in the extended XYZ writer (an engine type is not an element, so the element columns are blank),
 * coordinates in the %8.3f columns 31-54, residue MOL, chain A. Serials wrap after 99999, the PDB limit.
 * Columns follow the wwPDB format guide (CRYST1 1-70, ATOM/HETATM 1-78).
 */
export const framesToPdb = (frames: readonly TrajectoryRecord[]): string => {
  const out: string[] = [];
  const first = frames[0];
  if (first && first.box.tilt.every((t) => t === 0)) {
    const { lo, hi } = first.box;
    out.push(
      'CRYST1' + pdbNum(hi[0] - lo[0], 9, 3) + pdbNum(hi[1] - lo[1], 9, 3) + pdbNum(hi[2] - lo[2], 9, 3) +
        pdbNum(90, 7, 2) + pdbNum(90, 7, 2) + pdbNum(90, 7, 2) + ' ' + 'P 1'.padEnd(11) + '1'.padStart(4),
    );
  }
  frames.forEach((f, k) => {
    out.push('MODEL'.padEnd(10) + String(k + 1).padStart(4));
    for (let i = 0; i < f.topo.id.length; i++) {
      const name = `T${f.topo.type[i]}`.slice(0, 4).padEnd(4);
      const serial = String((i % 99999) + 1).padStart(5);
      out.push(
        'HETATM' + serial + ' ' + name + ' MOL A' + '1'.padStart(4) + '    ' +
          pdbNum(f.x[3 * i], 8, 3) + pdbNum(f.x[3 * i + 1], 8, 3) + pdbNum(f.x[3 * i + 2], 8, 3) +
          '  1.00' + '  0.00' + ' '.repeat(12),
      );
    }
    out.push('ENDMDL');
  });
  out.push('END');
  return out.join('\n') + '\n';
};

/**
 * LAMMPS data file of ONE frame (read_data, atom_style atomic): the header counts (atoms, and atom types as the
 * largest type id), the box bounds with the xy xz yz tilt line when triclinic, a Masses section when the record
 * carries masses (the first atom's mass of each type), then Atoms # atomic (id type x y z) and Velocities (id vx vy
 * vz) when the frame has velocities. Positions are the wrapped ones the frame holds.
 */
export const frameToLammpsData = (f: TrajectoryRecord): string => {
  const n = f.topo.id.length;
  const { lo, hi, tilt } = f.box;
  let ntypes = 0;
  for (let i = 0; i < n; i++) if (f.topo.type[i] > ntypes) ntypes = f.topo.type[i];
  const lines: string[] = [
    `LAMMPS data file written by the MD Notebook, step ${f.step}`,
    '',
    `${n} atoms`,
    `${ntypes} atom types`,
    '',
    `${num(lo[0])} ${num(hi[0])} xlo xhi`,
    `${num(lo[1])} ${num(hi[1])} ylo yhi`,
    `${num(lo[2])} ${num(hi[2])} zlo zhi`,
  ];
  if (tilt.some((t) => t !== 0)) lines.push(`${num(tilt[0])} ${num(tilt[1])} ${num(tilt[2])} xy xz yz`);
  if (f.mass) {
    const massOf = new Array<number | undefined>(ntypes + 1);
    for (let i = 0; i < n; i++) {
      const t = f.topo.type[i];
      if (massOf[t] === undefined) massOf[t] = f.mass[i];
    }
    lines.push('', 'Masses', '');
    for (let t = 1; t <= ntypes; t++) lines.push(`${t} ${num(massOf[t] ?? 1)}`);
  }
  lines.push('', 'Atoms # atomic', '');
  for (let i = 0; i < n; i++) {
    lines.push(`${f.topo.id[i]} ${f.topo.type[i]} ${num(f.x[3 * i])} ${num(f.x[3 * i + 1])} ${num(f.x[3 * i + 2])}`);
  }
  if (f.v) {
    lines.push('', 'Velocities', '');
    for (let i = 0; i < n; i++) {
      lines.push(`${f.topo.id[i]} ${num(f.v[3 * i])} ${num(f.v[3 * i + 1])} ${num(f.v[3 * i + 2])}`);
    }
  }
  return lines.join('\n') + '\n';
};

/**
 * CSV of every atom of every frame: header frame,step,id,type,x,y,z (plus vx,vy,vz when any frame carries
 * velocities; frames without them leave those cells empty). frame is the 0-based index in the list.
 */
export const framesToCsv = (frames: readonly TrajectoryRecord[]): string => {
  const withV = frames.some((f) => f.v !== undefined);
  const lines: string[] = [withV ? 'frame,step,id,type,x,y,z,vx,vy,vz' : 'frame,step,id,type,x,y,z'];
  frames.forEach((f, k) => {
    for (let i = 0; i < f.topo.id.length; i++) {
      let row = `${k},${f.step},${f.topo.id[i]},${f.topo.type[i]},${num(f.x[3 * i])},${num(f.x[3 * i + 1])},${num(f.x[3 * i + 2])}`;
      if (withV) row += f.v ? `,${num(f.v[3 * i])},${num(f.v[3 * i + 1])},${num(f.v[3 * i + 2])}` : ',,,';
      lines.push(row);
    }
  });
  return lines.join('\n') + '\n';
};

/** Speeds and masses of the atoms of one record (null when the record has no velocities). */
export const recordSpeeds = (r: TrajectoryRecord): { speed: Float64Array; mass: Float64Array } | null => {
  if (!r.v) return null;
  const n = r.topo.id.length;
  const speed = new Float64Array(n);
  const mass = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const vx = r.v[3 * i], vy = r.v[3 * i + 1], vz = r.v[3 * i + 2];
    speed[i] = Math.sqrt(vx * vx + vy * vy + vz * vz);
    mass[i] = r.mass ? r.mass[i] : 1; // a dump file has no masses: every atom has mass 1
  }
  return { speed, mass };
};

/**
 * Temperature from the kinetic energy of the atoms: T = sum(m v^2) / (3 N k_B) (three degrees of freedom per atom,
 * no centre-of-mass correction). Null for no atoms or a non-positive k_B.
 */
export const kineticTemperature = (speed: Float64Array, mass: Float64Array, kB: number): number | null => {
  const n = speed.length;
  if (n === 0 || !(kB > 0)) return null;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += mass[i] * speed[i] * speed[i];
  return sum / (3 * n * kB);
};

/**
 * Maxwell-Boltzmann speed density f(v) = 4 pi (m / (2 pi k T))^(3/2) v^2 exp(-m v^2 / (2 k T)); integrates to 1 over
 * v in [0, inf). Zero for T <= 0.
 */
export const maxwellBoltzmannDensity = (v: number, m: number, T: number, kB: number): number => {
  if (!(T > 0) || !(m > 0) || !(kB > 0) || v < 0) return 0;
  const a = m / (2 * Math.PI * kB * T);
  return 4 * Math.PI * a * Math.sqrt(a) * v * v * Math.exp(-m * v * v / (2 * kB * T));
};

/**
 * Speed histogram as a probability density: bins from 0 to the largest speed, density = count / (N * width), so
 * the bins sum to 1 (density times width). Null for no speeds.
 */
export const speedHistogram = (speed: Float64Array, bins: number): { x0: number; x1: number; count: number; density: number }[] | null => {
  const n = speed.length;
  if (n === 0 || bins < 1) return null;
  let max = 0;
  for (let i = 0; i < n; i++) if (speed[i] > max) max = speed[i];
  const width = max > 0 ? max / bins : 1;
  const counts = new Array<number>(bins).fill(0);
  for (let i = 0; i < n; i++) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor(speed[i] / width)));
    counts[b]++;
  }
  return counts.map((count, i) => ({
    x0: i * width, x1: (i + 1) * width, count, density: count / (n * width),
  }));
};
