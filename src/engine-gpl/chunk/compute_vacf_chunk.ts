/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/compute_vacf_chunk.cpp and
 * src/compute_vacf_chunk.h at commit b332f28 (compute vacf/chunk). Copyright (2003) Sandia Corporation (LAMMPS
 * development team). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_vacf_chunk.html: "Define a computation that calculates the velocity auto-correlation function
 * (VACF) for each chunk of atoms." Output: "This compute calculates a global array where the number of rows is the
 * number of chunks and the number of columns is 4: the three x,y,z components of the VACF and the total VACF."
 * The VACF of a chunk is the dot product of its centre-of-mass velocity now and at the first evaluation (the setup of
 * the first run, when the reference velocities are stored).
 *
 * Native: the reference velocities live in a fix STORE/GLOBAL created with the compute (id_COMPUTE_STORE); here they are
 * kept by the compute itself. The native setup() runs at the first run: it stores the chunk centre-of-mass velocities
 * as the reference, and the next evaluation (the first output) forms the dot products.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { ComputeChunkAtom } from '../../engine/compute/chunk';
import { chunkOf, refreshChunk, vcmSums } from './chunk_native';

export class ComputeVACFChunk extends Compute {
  readonly style = 'vacf/chunk';
  private readonly chunk: ComputeChunkAtom;
  /** firstflag of native: the reference velocities are stored at the first setup. */
  private firstflag = true;
  /** vacfnchunk of native: the number of chunks, which must stay constant. */
  private vacfnchunk = 0;
  /** The reference centre-of-mass velocities (vcminit of native), 3 per chunk. */
  private ref: Float64Array | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Incorrect number of arguments for compute vacf/chunk');
    this.chunk = chunkOf(sys, args[0], 'vacf/chunk');
    this.arrayFlag = true;
    this.sizeArrayCols = 4;
  }

  protected computeArray(): void {
    const N = refreshChunk(this.chunk);
    const first = this.firstflag;
    if (first) this.vacfnchunk = N;
    else if (this.vacfnchunk !== N) throw new StyleError('Compute vacf/chunk nchunk is not static');
    const { sum, masstotal } = vcmSums(this.sys, this.groupBit, this.chunk.ids, N);
    const vcmall = new Float64Array(3 * N);
    for (let c = 0; c < N; c++) {
      for (let d = 0; d < 3; d++) {
        vcmall[3 * c + d] = masstotal[c] > 0.0 ? sum[3 * c + d] / masstotal[c] : sum[3 * c + d];
      }
    }
    // native setup(): store the reference velocities once, at the first evaluation
    if (first) {
      this.ref = vcmall.slice();
      this.firstflag = false;
    }
    const ref = this.ref!;
    const vacf = new Float64Array(4 * N);
    for (let c = 0; c < N; c++) {
      const vxsq = vcmall[3 * c] * ref[3 * c];
      const vysq = vcmall[3 * c + 1] * ref[3 * c + 1];
      const vzsq = vcmall[3 * c + 2] * ref[3 * c + 2];
      vacf[4 * c] = vxsq;
      vacf[4 * c + 1] = vysq;
      vacf[4 * c + 2] = vzsq;
      vacf[4 * c + 3] = vxsq + vysq + vzsq;
    }
    this.sizeArrayRows = N;
    this.array = vacf;
  }
}
