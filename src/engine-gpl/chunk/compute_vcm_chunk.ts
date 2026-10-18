/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/compute_vcm_chunk.cpp and
 * src/compute_vcm_chunk.h at commit b332f28 (compute vcm/chunk). Copyright (2003) Sandia Corporation (LAMMPS development
 * team). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_vcm_chunk.html: "Define a computation that calculates the velocity of center-of-mass for each
 * chunk of atoms." Output: "This compute calculates a global array where the number of rows is the number of chunks
 * and the number of columns is 3 for the x,y,z components of the center-of-mass velocity."
 *
 * Simplification: the native massneed flag (chunk masses kept after the first call when chunk IDs are "once") is not
 * modelled; the chunk masses are summed at every evaluation, which gives the same values unless masses change.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { ComputeChunkAtom } from '../../engine/compute/chunk';
import { chunkOf, refreshChunk, vcmSums } from './chunk_native';

export class ComputeVCMChunk extends Compute {
  readonly style = 'vcm/chunk';
  private readonly chunk: ComputeChunkAtom;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Illegal compute vcm/chunk command');
    this.chunk = chunkOf(sys, args[0], 'vcm/chunk');
    this.arrayFlag = true;
    this.sizeArrayCols = 3;
  }

  protected computeArray(): void {
    const N = refreshChunk(this.chunk);
    const { sum, masstotal } = vcmSums(this.sys, this.groupBit, this.chunk.ids, N);
    const out = new Float64Array(3 * N);
    for (let c = 0; c < N; c++) {
      for (let d = 0; d < 3; d++) out[3 * c + d] = masstotal[c] > 0.0 ? sum[3 * c + d] / masstotal[c] : 0.0;
    }
    this.sizeArrayRows = N;
    this.array = out;
  }
}
