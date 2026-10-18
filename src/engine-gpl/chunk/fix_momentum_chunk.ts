/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-FIX/fix_momentum_chunk.cpp and
 * src/EXTRA-FIX/fix_momentum_chunk.h at commit b332f28 (fix momentum/chunk). Contributing author: Jiang Xiao (Hong Kong
 * Polytechnic University). Copyright (2003) Sandia Corporation (LAMMPS development team). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_momentum_chunk.html: "fix ID group-ID momentum/chunk N chunkID keyword values ..."; "Zero the
 * linear and/or angular momentum of each chunk of atoms every N steps by adjusting their velocities." Keywords: linear
 * (three 0/1 flags for x, y, z), angular, rescale. "The rescale keyword ... rescales the velocities of each chunk so the
 * kinetic energy of the chunk is the same as before the adjustment."
 *
 * The per-chunk centres of mass, velocities and angular velocities are the ones of compute com/chunk, vcm/chunk and
 * omega/chunk on the fix group, computed here from the same sums (chunk_native.ts) without creating those computes.
 * The sums run in native atom order. Native's rescale tests ke_chunk_new[0] (not [m]) for the factor; that is kept.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import type { ComputeChunkAtom } from '../../engine/compute/chunk';
import { chunkOf, comCentres, comSums, omegaChunk, refreshChunk, vcmSums } from './chunk_native';

const intArg = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || !Number.isInteger(v)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input`);
  return v;
};

export class FixMomentumChunk extends Fix {
  readonly style = 'momentum/chunk';
  private readonly idChunk: string;
  private cchunk: ComputeChunkAtom | null = null;
  private linear = false;
  private angular = false;
  private rescale = false;
  private xflag = 0;
  private yflag = 0;
  private zflag = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal fix momentum/chunk command');
    this.nevery = intArg(args[0]);
    if (this.nevery <= 0) throw new StyleError('Illegal fix momentum/chunk command');
    this.idChunk = args[1];
    if (!sys.computes.some((c) => c.id === this.idChunk)) {
      throw new StyleError('Chunk/atom compute does not exist for fix momentum/chunk');
    }
    let k = 2;
    while (k < args.length) {
      const w = args[k];
      if (w === 'linear') {
        if (k + 4 > args.length) throw new StyleError('Illegal fix momentum command');
        this.linear = true;
        this.xflag = intArg(args[k + 1]);
        this.yflag = intArg(args[k + 2]);
        this.zflag = intArg(args[k + 3]);
        k += 4;
      } else if (w === 'angular') {
        this.angular = true;
        k += 1;
      } else if (w === 'rescale') {
        this.rescale = true;
        k += 1;
      } else {
        throw new StyleError('Illegal fix momentum/chunk command');
      }
    }
    if (!this.linear && !this.angular) throw new StyleError('Illegal fix momentum/chunk command');
    if (this.linear) {
      if (this.xflag < 0 || this.xflag > 1 || this.yflag < 0 || this.yflag > 1 || this.zflag < 0 || this.zflag > 1) {
        throw new StyleError('Illegal fix momentum/chunk command');
      }
    }
  }

  init(): void {
    const c = this.sys.computes.find((x) => x.id === this.idChunk);
    if (!c) throw new StyleError('Chunk/atom compute does not exist for fix momentum/chunk');
    this.cchunk = chunkOf(this.sys, this.idChunk, 'momentum/chunk');
  }

  /** KE of each chunk: sum over the group of m (vx^2 + vy^2 + vz^2), in native order. */
  private keChunk(N: number, ichunk: Int32Array): Float64Array {
    const s = this.sys.state;
    const ke = new Float64Array(N);
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = ichunk[i] - 1;
      if (m < 0) continue;
      const vx = s.v[3 * i], vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
      ke[m] += massOf(s, i) * (vx * vx + vy * vy + vz * vz);
    }
    return ke;
  }

  endOfStep(): void {
    const sys = this.sys;
    const s = sys.state;
    const chunk: ComputeChunkAtom = this.cchunk!;
    const N = refreshChunk(chunk);
    const ichunk = chunk.ids;
    // per-chunk properties, from the fix group (compute com/chunk, vcm/chunk and omega/chunk of native)
    const com = comCentres(comSums(sys, this.groupBit, ichunk, N), N);
    const vs = vcmSums(sys, this.groupBit, ichunk, N);
    const vcm = new Float64Array(3 * N);
    for (let m = 0; m < N; m++) {
      for (let d = 0; d < 3; d++) vcm[3 * m + d] = vs.masstotal[m] > 0.0 ? vs.sum[3 * m + d] / vs.masstotal[m] : 0.0;
    }
    const omega = this.angular ? omegaChunk(sys, this.groupBit, ichunk, N) : null;
    const keOld = this.rescale ? this.keChunk(N, ichunk) : null;
    const inGroupChunk = (i: number): number => (s.mask[i] & this.groupBit) ? ichunk[i] - 1 : -1;
    if (this.linear) {
      // adjust velocities by vcm to zero linear momentum; only the flagged components
      for (let i = 0; i < s.n; i++) {
        const m = inGroupChunk(i);
        if (m < 0) continue;
        if (this.xflag) s.v[3 * i] -= vcm[3 * m];
        if (this.yflag) s.v[3 * i + 1] -= vcm[3 * m + 1];
        if (this.zflag) s.v[3 * i + 2] -= vcm[3 * m + 2];
      }
    }
    if (omega) {
      // vnew_i = v_i - w x r_i, with unwrapped coordinates
      const u = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        const m = inGroupChunk(i);
        if (m < 0) continue;
        sys.geom.unwrap(s.x, s.image, i, u);
        const dx = u[0] - com[3 * m];
        const dy = u[1] - com[3 * m + 1];
        const dz = u[2] - com[3 * m + 2];
        const w0 = omega[3 * m], w1 = omega[3 * m + 1], w2 = omega[3 * m + 2];
        s.v[3 * i] -= w1 * dz - w2 * dy;
        s.v[3 * i + 1] -= w2 * dx - w0 * dz;
        s.v[3 * i + 2] -= w0 * dy - w1 * dx;
      }
    }
    if (keOld) {
      const keNew = this.keChunk(N, ichunk);
      // native: the test uses ke_chunk_new[0], not the chunk's own value (src/EXTRA-FIX/fix_momentum_chunk.cpp)
      const factor = new Float64Array(N);
      for (let m = 0; m < N; m++) factor[m] = keNew[0] > 0.0 ? Math.sqrt(keOld[m] / keNew[m]) : 1.0;
      for (let i = 0; i < s.n; i++) {
        const m = inGroupChunk(i);
        if (m < 0) continue;
        s.v[3 * i] *= factor[m];
        s.v[3 * i + 1] *= factor[m];
        s.v[3 * i + 2] *= factor[m];
      }
    }
  }
}
