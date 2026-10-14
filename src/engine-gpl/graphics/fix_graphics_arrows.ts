/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/fix_graphics_arrows.cpp and fix_graphics_arrows.h (class FixGraphicsArrows, fix
 * graphics/arrows: "fix ID group-ID graphics/arrows Nevery mode args ... keyword value ..."). Copyright (2003) Sandia
 * Corporation, distributed under the GNU General Public License (the LAMMPS file headers). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_graphics_arrows.html): one arrow per atom of the group, centred on the
 * atom, pointing along the vector of the chosen mode (force, velocity or a variable) with length |vector| * scale
 * and a shaft of diameter 2 * radius; "autoscale value" sets scale so that the mean arrow length is value. The
 * C++ ARROW objects keep the tip ratio 0.2 (fix_graphics_arrows.cpp: imgparms[n][9] = 0.2).
 *
 * Mode chunk (src/GRAPHICS/fix_graphics_arrows.cpp, FixGraphicsArrows::end_of_step, the CHUNK branch): one arrow per
 * chunk, centred on the chunk's position array (the per-chunk compute of the first column group, columns 1..3) and
 * pointing along the vector array (columns 1..3 of the second compute); every arrow has type 1, the scale is the
 * "scale" argument or, with autoscale, the mean vector length over the chunks, and the shaft is 2 * radius.
 *
 * Refused (named StyleError): mode dipole (needs the dipole attribute and its arrows are not ported).
 */
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import type { ImageObject, Vec3 } from '../../engine/output/image_objects';
import { parseInteger, parseNumber } from './common';
import { chunkCompute, type ComputeChunkAtom } from '../../engine/compute/chunk';
import type { Compute } from '../../engine/compute/compute';

type Vector = { kind: 'force' } | { kind: 'velocity' } | { kind: 'variable'; comp: [string | null, string | null, string | null]; fixed: Vec3 };
/** Per-chunk arrows (mode chunk): the chunk/atom compute and the per-chunk position and vector computes. */
interface ChunkSource {
  chunk: ComputeChunkAtom;
  pos: Compute;
  vec: Compute;
}

export class FixGraphicsArrows extends Fix {
  readonly style = 'graphics/arrows';
  private readonly mode: Vector | { kind: 'chunk' };
  private readonly chunkSource: ChunkSource | null = null;
  private scale = 1;
  private readonly radius: number;
  private autoscale: number | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError(`usage: fix ${id} group-ID graphics/arrows Nevery mode args ...`);
    this.nevery = parseInteger(args[0], `fix ${id} graphics/arrows Nevery`);
    if (this.nevery <= 0) throw new StyleError(`fix ${id} graphics/arrows: illegal Nevery value`);
    const mode = args[1];
    let k: number;
    if (mode === 'force' || mode === 'velocity') {
      this.mode = { kind: mode };
      this.scale = parseNumber(args[2], `fix ${id} graphics/arrows scale`);
      this.radius = parseNumber(args[3], `fix ${id} graphics/arrows radius`);
      k = 4;
    } else if (mode === 'variable') {
      if (args.length < 6) throw new StyleError(`fix ${id} graphics/arrows variable: needs three components and a radius`);
      const comp: [string | null, string | null, string | null] = [null, null, null];
      const fixed: Vec3 = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        const w = args[2 + d];
        if (w.startsWith('v_')) comp[d] = w.slice(2);
        else fixed[d] = parseNumber(w, `fix ${id} graphics/arrows variable component`);
      }
      this.mode = { kind: 'variable', comp, fixed };
      this.radius = parseNumber(args[5], `fix ${id} graphics/arrows radius`);
      k = 6;
    } else if (mode === 'dipole') {
      throw new StyleError(`fix ${id} graphics/arrows dipole mode is not supported by the browser engine`);
    } else if (mode === 'chunk') {
      // fix_graphics_arrows.cpp: "fix ID group-ID graphics/arrows Nevery chunk chunkID posID vecID scale radius keyword value ..."
      if (args.length < 7) throw new StyleError(`fix ${id} graphics/arrows chunk: needs chunkID, posID, vecID, scale and radius`);
      const chunk = chunkCompute(sys, args[2], `fix ${id} graphics/arrows`);
      const pos = sys.compute(args[3]);
      const vec = sys.compute(args[4]);
      for (const [c, cid] of [[pos, args[3]], [vec, args[4]]] as const) {
        if (!c.arrayFlag || c.sizeArrayCols < 3) throw new StyleError(`fix ${id} graphics/arrows: per-chunk compute ${cid} is not compatible (it needs a 3-column global array)`);
      }
      this.mode = { kind: 'chunk' };
      this.chunkSource = { chunk, pos, vec };
      this.scale = parseNumber(args[5], `fix ${id} graphics/arrows scale`);
      this.radius = parseNumber(args[6], `fix ${id} graphics/arrows radius`);
      k = 7;
    } else {
      throw new StyleError(`fix ${id} graphics/arrows: unknown mode '${mode}' (force, velocity, variable)`);
    }
    if (!(this.radius > 0)) throw new StyleError(`fix ${id} graphics/arrows: arrow radius must be > 0`);
    while (k < args.length) {
      if (args[k] !== 'autoscale') throw new StyleError(`fix ${id} graphics/arrows: unknown keyword '${args[k]}'`);
      const v = parseNumber(args[k + 1], `fix ${id} graphics/arrows autoscale`);
      if (!(v > 0)) throw new StyleError(`fix ${id} graphics/arrows: autoscale value must be > 0`);
      this.autoscale = v;
      k += 2;
    }
  }

  /** The vector of each group atom: [ [atom index, vector] ] for the current state. */
  private vectors(): { i: number; v: Vec3 }[] {
    const s = this.sys.state;
    const out: { i: number; v: Vec3 }[] = [];
    if (this.mode.kind === 'variable') {
      const m = this.mode;
      // v_ components are atom-style variables (evaluated per atom); a number is the same for every atom
      const ax = m.comp[0] !== null ? this.sys.atomVariable(m.comp[0]) : null;
      const ay = m.comp[1] !== null ? this.sys.atomVariable(m.comp[1]) : null;
      const az = m.comp[2] !== null ? this.sys.atomVariable(m.comp[2]) : null;
      for (let i = 0; i < s.n; i++) {
        if (!this.inGroup(i)) continue;
        const v: Vec3 = [
          ax ? ax[i] : m.fixed[0],
          ay ? ay[i] : m.fixed[1],
          az ? az[i] : m.fixed[2],
        ];
        out.push({ i, v });
      }
      return out;
    }
    const src = this.mode.kind === 'force' ? s.f : s.v;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      out.push({ i, v: [src[3 * i], src[3 * i + 1], src[3 * i + 2]] });
    }
    return out;
  }

  imageObjects(): ImageObject[] {
    if (this.chunkSource) return chunkObjectsOf(this.chunkSource, this.scale, this.autoscale, this.radius);
    const s = this.sys.state;
    const vs = this.vectors();
    let scale = this.scale;
    if (this.autoscale !== null && vs.length > 0) {
      // the mean vector length over the group sets the scale (fix_graphics_arrows.cpp end_of_step)
      let sum = 0;
      for (const { v } of vs) sum += Math.hypot(v[0], v[1], v[2]);
      scale = sum !== 0 ? this.autoscale / (sum / vs.length) : 1;
    }
    const out: ImageObject[] = [];
    for (const { i, v } of vs) {
      const len = Math.hypot(v[0], v[1], v[2]) * scale;
      if (!(len > 0)) continue;
      out.push({ kind: 'arrow', type: s.type[i], center: [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]], dir: v, length: len, diameter: 2 * this.radius, ratio: 0.2 });
    }
    return out;
  }
}

/** Mode chunk: one arrow per chunk (fix_graphics_arrows.cpp, the CHUNK branch of end_of_step). */
const chunkObjectsOf = (src: ChunkSource, scale0: number, autoscale: number | null, radius: number): ImageObject[] => {
  src.chunk.ensure();
  const n = src.chunk.nchunk;
  const pdata = src.pos.arrayValues();
  const pc = src.pos.sizeArrayCols;
  const vdata = src.vec.arrayValues();
  const vc = src.vec.sizeArrayCols;
  let scale = scale0;
  if (autoscale !== null) {
    // native: the mean vector length over the chunks sets the scale; a zero sum gives scale 1
    let sum = 0;
    for (let c = 0; c < n; c++) sum += Math.hypot(vdata[c * vc], vdata[c * vc + 1], vdata[c * vc + 2]);
    scale = sum !== 0 ? autoscale / (sum / n) : 1;
  }
  const out: ImageObject[] = [];
  for (let c = 0; c < n; c++) {
    const v: Vec3 = [vdata[c * vc], vdata[c * vc + 1], vdata[c * vc + 2]];
    const len = Math.hypot(v[0], v[1], v[2]) * scale;
    if (!(len > 0)) continue;
    out.push({ kind: 'arrow', type: 1, center: [pdata[c * pc], pdata[c * pc + 1], pdata[c * pc + 2]], dir: v, length: len, diameter: 2 * radius, ratio: 0.2 });
  }
  return out;
};

export const makeFixGraphicsArrows: FixFactory = (sys, id, group, args) => new FixGraphicsArrows(sys, id, group, args);
