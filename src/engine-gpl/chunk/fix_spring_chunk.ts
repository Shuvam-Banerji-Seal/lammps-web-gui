/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/fix_spring_chunk.cpp and
 * src/fix_spring_chunk.h at commit b332f28 (fix spring/chunk). Copyright (2003) Sandia Corporation (LAMMPS development
 * team). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_spring_chunk.html: "fix ID group-ID spring/chunk K chunkID comID"; "This fix connects each chunk
 * to its own spring. The spring is between the centre of mass of the chunk now and its centre of mass at the first
 * step of the fix (the reference is stored when the fix first runs)." The force on each atom is the chunk's spring force
 * divided by the chunk mass and multiplied by the atom's mass, and it acts on every atom with a chunk ID (no group).
 * The energy of the springs is the scalar output, which fix_modify energy yes adds to the potential energy.
 *
 * The centres of mass come from a compute com/chunk with the same chunk/atom compute (comID); with its wrap yes the
 * centres are mapped back into the periodic box (Domain::remap) before the springs are evaluated, as native reads the
 * compute's array. Native's chunk lock (the reference COMs keep the chunk IDs from changing) is not modelled.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { ComputeComChunk } from '../../engine/compute/chunk';
import type { ComputeChunkAtom } from '../../engine/compute/chunk';
import { chunkOf, comSums, refreshChunk } from './chunk_native';

export class FixSpringChunk extends Fix {
  readonly style = 'spring/chunk';
  private readonly k: number;
  private readonly idChunk: string;
  private readonly idCom: string;
  private cchunk: ComputeChunkAtom | null = null;
  private ccom: ComputeComChunk | null = null;
  /** The reference centres of mass (com0 of native), 3 per chunk, set at the first evaluation. */
  private com0: Float64Array | null = null;
  private nchunk = 0;
  private fcom = new Float64Array(0);
  private esprings = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('Illegal fix spring/chunk command: missing argument(s)');
    this.scalarFlag = true;
    this.globalFreq = 1;
    this.extscalar = 1;
    this.energyGlobal = true;
    this.k = Number(args[0]);
    if (!Number.isFinite(this.k)) throw new StyleError(`Expected floating point parameter instead of '${args[0]}' for fix spring/chunk K`);
    this.idChunk = args[1];
    this.idCom = args[2];
    this.cchunk = chunkOf(sys, this.idChunk, 'spring/chunk');
  }

  init(): void {
    this.cchunk = chunkOf(this.sys, this.idChunk, 'spring/chunk');
    const c = this.sys.computes.find((x) => x.id === this.idCom);
    if (!(c instanceof ComputeComChunk)) throw new StyleError(`Com/chunk compute ${this.idCom} does not exist or is not com/chunk style`);
    const comChunk = (c as unknown as { chunk: ComputeChunkAtom }).chunk;
    if (comChunk !== this.cchunk) {
      throw new StyleError(`Fix spring/chunk chunk ID ${this.idChunk} not the same as compute com/chunk chunk ID ${comChunk.id}`);
    }
    this.ccom = c;
  }

  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const chunk = this.cchunk!;
    const N = refreshChunk(chunk);
    const ichunk = chunk.ids;
    const sums = comSums(sys, this.ccom!.groupBit, ichunk, N);
    const masstotal = sums.masstotal;
    const com = new Float64Array(3 * N);
    const wrap = (this.ccom as unknown as { wrap: boolean }).wrap;
    const p = [0, 0, 0];
    for (let m = 0; m < N; m++) {
      for (let d = 0; d < 3; d++) com[3 * m + d] = masstotal[m] > 0.0 ? sums.sum[3 * m + d] / masstotal[m] : 0.0;
      if (wrap && masstotal[m] > 0.0) {
        p[0] = com[3 * m]; p[1] = com[3 * m + 1]; p[2] = com[3 * m + 2];
        sys.geom.remapPoint(p);
        com[3 * m] = p[0]; com[3 * m + 1] = p[1]; com[3 * m + 2] = p[2];
      }
    }
    if (this.com0 === null) {
      // native: the reference COMs are stored the first time the fix is evaluated
      this.nchunk = N;
      this.com0 = com.slice();
      this.fcom = new Float64Array(3 * N);
    } else if (this.nchunk !== N) {
      throw new StyleError(`Fix spring/chunk: the number of chunks changed from ${this.nchunk} to ${N}`);
    }
    const com0 = this.com0;
    const fcom = this.fcom;
    const k = this.k;
    this.esprings = 0;
    for (let m = 0; m < N; m++) {
      const dx = com[3 * m] - com0[3 * m];
      const dy = com[3 * m + 1] - com0[3 * m + 1];
      const dz = com[3 * m + 2] - com0[3 * m + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (masstotal[m] !== 0.0) {
        fcom[3 * m] = k * dx / masstotal[m];
        fcom[3 * m + 1] = k * dy / masstotal[m];
        fcom[3 * m + 2] = k * dz / masstotal[m];
        this.esprings += 0.5 * k * rsq;
      }
    }
    for (const i of nativeOrder(s)) {
      const m = ichunk[i] - 1;
      if (m < 0) continue;
      const mass = massOf(s, i);
      s.f[3 * i] -= fcom[3 * m] * mass;
      s.f[3 * i + 1] -= fcom[3 * m + 1] * mass;
      s.f[3 * i + 2] -= fcom[3 * m + 2] * mass;
    }
  }

  minPostForce(): void { this.postForce(); }

  energy(): number { return this.esprings; }
  computeScalar(): number { return this.esprings; }
}
