/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/compute_torque_chunk.cpp and
 * src/compute_torque_chunk.h at commit b332f28 (compute torque/chunk). Copyright (2003) Sandia Corporation (LAMMPS
 * development team). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_torque_chunk.html: "Define a computation that calculates the torque for each chunk of atoms."
 * Output: "This compute calculates a global array where the number of rows is the number of chunks and the number of
 * columns is 3 for the x,y,z components of the torque." The torque of each atom is taken about the centre of mass of its
 * chunk, with unwrapped coordinates and the current force on the atom.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { ComputeChunkAtom } from '../../engine/compute/chunk';
import { nativeOrder } from '../../engine/atoms';
import { chunkOf, comCentres, comSums, refreshChunk } from './chunk_native';

export class ComputeTorqueChunk extends Compute {
  readonly style = 'torque/chunk';
  private readonly chunk: ComputeChunkAtom;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Illegal compute torque/chunk command');
    this.chunk = chunkOf(sys, args[0], 'torque/chunk');
    this.arrayFlag = true;
    this.sizeArrayCols = 3;
  }

  protected computeArray(): void {
    const s = this.sys.state;
    const N = refreshChunk(this.chunk);
    const ichunk = this.chunk.ids;
    const comall = comCentres(comSums(this.sys, this.groupBit, ichunk, N), N);
    const torque = new Float64Array(3 * N);
    const u = [0, 0, 0];
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const c = ichunk[i] - 1;
      if (c < 0) continue;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const dx = u[0] - comall[3 * c];
      const dy = u[1] - comall[3 * c + 1];
      const dz = u[2] - comall[3 * c + 2];
      const fx = s.f[3 * i], fy = s.f[3 * i + 1], fz = s.f[3 * i + 2];
      torque[3 * c] += dy * fz - dz * fy;
      torque[3 * c + 1] += dz * fx - dx * fz;
      torque[3 * c + 2] += dx * fy - dy * fx;
    }
    this.sizeArrayRows = N;
    this.array = torque;
  }
}
