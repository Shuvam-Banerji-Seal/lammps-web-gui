import { Compute } from './compute';
import { ComputeChunkAtom, chunkCompute } from './chunk';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * compute ID group-ID temp/chunk chunkID value1 value2 ... keyword value ...
 * — docs.lammps.org/compute_temp_chunk.html. Quoted lines are verbatim from
 * plans/lammps-docs/compute_temp_chunk.rst:
 *
 *   |   compute ID group-ID temp/chunk chunkID value1 value2 ... keyword value ...
 *   |* value = *temp* or *kecom* or *internal*
 *   |* keyword = *com* or *bias* or *adof* or *cdof*
 *   |*com* value = *yes* or *no*
 *   |*bias* value = bias-ID
 *   |*adof* value = dof_per_atom
 *   |*cdof* value = dof_per_chunk
 *
 *   |The temperature is calculated by the formula
 *   |  KE = DOF/2 k_B T, DOF is the the total number of degrees of freedom
 *   |The DOF is calculated as :math:`N\times`\ *adof*
 *   |+ :math:`N_\text{chunk}\times`\ *cdof*,
 *   |where :math:`N` is the number of atoms contributing to the kinetic energy,
 *   |*adof* is the number of degrees of freedom per atom, and
 *   |*cdof* is the number of degrees of freedom per chunk.
 *   |By default, *adof* = 2 or 3 = dimensionality of system, as set
 *   |via the :doc:`dimension <dimension>` command, and *cdof* = 0.0.
 *
 * The per-chunk temp value uses DOF_c = N_c adof + cdof for the atoms of
 * chunk c. "The *kecom* value calculates the kinetic energy of each chunk as
 * if all its atoms were moving with the velocity of the center-of-mass of the
 * chunk." "The *internal* value calculates the internal kinetic energy of each
 * chunk. ... its velocity minus the center-of-mass velocity of the chunk."
 *
 * Bias handling (docs.lammps.org/compute_temp_chunk.html):
 *   - com yes: "subtract center-of-mass velocity from each chunk before
 *     calculating temperature"; as a thermostat temperature the per-chunk VCM is
 *     the bias (computeBias/removeBiasAll subtract it, restoreBiasAll adds it).
 *   - bias bias-ID: the bias compute removes its velocity bias for the whole
 *     system while this compute evaluates, and the bias is restored afterwards.
 *   - "The *com* and *bias* keywords cannot be used together."
 *
 * No degrees of freedom are subtracted for fix constraints or velocity biases
 * (docs: "no degrees of freedom are subtracted for any velocity bias or
 * constraints that are applied"), so the global dof is adof N + cdof Nchunk, with
 * N the atoms of the group that are in a chunk (chunk ID > 0). Measured with
 * native LAMMPS (black box): the global scalar of a bin/1d chunk with empty bins
 * (cdof 0.5 per chunk) matches this count with Nchunk the number of bins.
 *
 * Measured with native LAMMPS (black box), not in the docs: a fix nvt
 * thermostat whose temperature is a temp/chunk compute with per-chunk values
 * (temp, kecom or internal) referenced by thermo leaves the trajectory identical
 * to NVE; the engine does not reproduce this (see the final report).
 */

const VALUES = ['temp', 'kecom', 'internal'] as const;
type Value = (typeof VALUES)[number];

export class ComputeTempChunk extends Compute {
  readonly style = 'temp/chunk';
  private readonly chunk: ComputeChunkAtom;
  private readonly values: Value[];
  private readonly com: boolean;
  private readonly biasId: string | null;
  private adof: number;
  private cdof = 0;
  /** Set while this compute is the thermostat's bias owner (its bias removed by removeBiasAll). */
  private nestedOn = false;
  /** Per-chunk VCM subtracted while the com bias is removed (restored by +=). */
  private vcmBias: Float64Array<ArrayBuffer> = new Float64Array(0);
  private biasOn = false;
  private biasCompute: Compute | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError(`usage: compute ${id} group-ID temp/chunk chunkID value ... keyword value ...`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (temp/chunk)`);
    this.adof = sys.dimension;
    const values: Value[] = [];
    let com = false;
    let biasId: string | null = null;
    let k = 1;
    while (k < args.length) {
      const w = args[k];
      if ((VALUES as readonly string[]).includes(w)) {
        values.push(w as Value);
        k++;
        continue;
      }
      if (k + 1 >= args.length) throw new StyleError(`compute ${id} (temp/chunk): keyword ${w} needs a value`);
      const v = args[k + 1];
      if (w === 'com') {
        if (v !== 'yes' && v !== 'no') throw new StyleError(`compute ${id} (temp/chunk): com must be yes or no (got '${v}')`);
        com = v === 'yes';
      } else if (w === 'bias') {
        biasId = v;
      } else if (w === 'adof') {
        this.adof = parseNum(v, 'compute temp/chunk adof');
      } else if (w === 'cdof') {
        this.cdof = parseNum(v, 'compute temp/chunk cdof');
      } else {
        throw new StyleError(`compute ${id} (temp/chunk): unknown value or keyword '${w}'`);
      }
      k += 2;
    }
    if (com && biasId !== null) throw new StyleError(`compute ${id} (temp/chunk): The com and bias keywords cannot be used together.`);
    this.values = values;
    this.com = com;
    this.biasId = biasId;
    this.tempFlag = true;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.vector = new Float64Array(6);
    this.extscalar = 0;
    this.extvector = 1;
    this.arrayFlag = values.length > 0;
    this.sizeArrayCols = values.length;
  }

  init(): void {
    this.biasCompute = this.biasId === null ? null : this.resolveBias();
    this.dofCompute();
  }

  private resolveBias(): Compute {
    const c = this.sys.compute(this.biasId!);
    if (!c.tempFlag || !c.hasBias()) throw new StyleError(`compute ${this.id} (temp/chunk): bias compute ${this.biasId} does not remove a velocity bias`);
    return c;
  }

  /** Atoms of the group that are in a chunk (chunk ID > 0 within the chunk's range). */
  private counted(i: number, nch: number): number {
    const c = this.chunk.ids[i];
    return (this.sys.state.mask[i] & this.groupBit) && c > 0 && c <= nch ? c : 0;
  }

  /** Global dof = adof N + cdof Nchunk (doc formula). */
  dofCompute(): void {
    this.chunk.ensure();
    const s = this.sys.state;
    const nch = this.chunk.nchunk;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.counted(i, nch)) n++;
    this.dof = this.adof * n + this.cdof * nch;
  }

  /** Runs fn with the biased compute's velocity bias removed (unless the thermostat already did). */
  private withBias<T>(fn: () => T): T {
    if (this.biasId === null || this.nestedOn) return fn();
    this.ensureBiasCompute();
    const b = this.biasCompute!;
    b.computeBias();
    b.removeBiasAll();
    try {
      return fn();
    } finally {
      b.restoreBiasAll();
    }
  }

  /** Per-chunk centre-of-mass velocity (mass-weighted) and mass of the counted atoms. */
  private chunkVcm(nch: number): { vcm: Float64Array<ArrayBuffer>; mass: Float64Array<ArrayBuffer> } {
    const s = this.sys.state;
    const vcm: Float64Array<ArrayBuffer> = new Float64Array(3 * nch);
    const mass: Float64Array<ArrayBuffer> = new Float64Array(nch);
    for (let i = 0; i < s.n; i++) {
      const c = this.counted(i, nch);
      if (c === 0) continue;
      const m = massOf(s, i);
      const k = c - 1;
      vcm[3 * k] += m * s.v[3 * i];
      vcm[3 * k + 1] += m * s.v[3 * i + 1];
      vcm[3 * k + 2] += m * s.v[3 * i + 2];
      mass[k] += m;
    }
    for (let k = 0; k < nch; k++) {
      for (let d = 0; d < 3; d++) vcm[3 * k + d] = mass[k] > 0 ? vcm[3 * k + d] / mass[k] : 0;
    }
    return { vcm, mass };
  }

  /** Global temperature and tensor on the thermal velocities (com subtracted when com yes). */
  private globalThermal(): { t: number; tensor: number[] } {
    const s = this.sys.state;
    this.chunk.ensure();
    const nch = this.chunk.nchunk;
    this.dofCompute();
    const { vcm } = this.com ? this.chunkVcm(nch) : { vcm: null };
    const tensor = [0, 0, 0, 0, 0, 0];
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      const c = this.counted(i, nch);
      if (c === 0) continue;
      const m = massOf(s, i);
      const u0 = s.v[3 * i] - (vcm ? vcm[3 * (c - 1)] : 0);
      const u1 = s.v[3 * i + 1] - (vcm ? vcm[3 * (c - 1) + 1] : 0);
      const u2 = s.v[3 * i + 2] - (vcm ? vcm[3 * (c - 1) + 2] : 0);
      t += m * (u0 * u0 + u1 * u1 + u2 * u2);
      tensor[0] += m * u0 * u0; tensor[1] += m * u1 * u1; tensor[2] += m * u2 * u2;
      tensor[3] += m * u0 * u1; tensor[4] += m * u0 * u2; tensor[5] += m * u1 * u2;
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return { t: t * tfactor, tensor: tensor.map((x) => x * s.units.mvv2e) };
  }

  protected computeScalar(): number {
    return this.withBias(() => this.globalThermal().t);
  }

  protected computeVector(): void {
    const { tensor } = this.withBias(() => this.globalThermal());
    for (let c = 0; c < 6; c++) this.vector[c] = tensor[c];
  }

  protected computeArray(): void {
    this.withBias(() => {
      const s = this.sys.state;
      this.chunk.ensure();
      const nch = this.chunk.nchunk;
      const cols = this.values.length;
      const out = new Float64Array(nch * cols);
      const adof = this.adof, cdof = this.cdof;
      const { vcm, mass } = this.chunkVcm(nch); // centre of mass of each chunk (bias applied)
      const thermalV = this.com ? vcm : null;
      const nAtoms = new Int32Array(nch);
      for (let i = 0; i < s.n; i++) {
        const c = this.counted(i, nch);
        if (c > 0) nAtoms[c - 1]++;
      }
      for (let col = 0; col < cols; col++) {
        const what = this.values[col];
        const acc = new Float64Array(nch);
        for (let i = 0; i < s.n; i++) {
          const c = this.counted(i, nch);
          if (c === 0) continue;
          const k = c - 1;
          const m = massOf(s, i);
          if (what === 'temp') {
            const u0 = s.v[3 * i] - (thermalV ? thermalV[3 * k] : 0);
            const u1 = s.v[3 * i + 1] - (thermalV ? thermalV[3 * k + 1] : 0);
            const u2 = s.v[3 * i + 2] - (thermalV ? thermalV[3 * k + 2] : 0);
            acc[k] += m * (u0 * u0 + u1 * u1 + u2 * u2);
          } else if (what === 'internal') {
            const u0 = s.v[3 * i] - vcm[3 * k], u1 = s.v[3 * i + 1] - vcm[3 * k + 1], u2 = s.v[3 * i + 2] - vcm[3 * k + 2];
            acc[k] += m * (u0 * u0 + u1 * u1 + u2 * u2);
          }
        }
        for (let k = 0; k < nch; k++) {
          let val: number;
          if (what === 'temp') {
            const dofC = adof * nAtoms[k] + cdof;
            val = dofC > 0 ? (s.units.mvv2e / (dofC * s.units.boltz)) * acc[k] : 0;
          } else if (what === 'kecom') {
            const vx = vcm[3 * k], vy = vcm[3 * k + 1], vz = vcm[3 * k + 2];
            val = 0.5 * s.units.mvv2e * mass[k] * (vx * vx + vy * vy + vz * vz);
          } else {
            val = 0.5 * s.units.mvv2e * acc[k];
          }
          out[k * cols + col] = val;
        }
      }
      this.sizeArrayRows = nch;
      this.array = out;
    });
  }

  // ---- velocity bias for thermostats: the com yes per-chunk VCM, or the bias-ID compute's bias
  hasBias(): boolean { return this.com || this.biasId !== null; }

  computeBias(): void {
    if (this.biasId !== null) {
      this.ensureBiasCompute();
      this.biasCompute!.computeBias();
    }
  }

  private ensureBiasCompute(): void {
    if (this.biasCompute === null) this.biasCompute = this.resolveBias();
  }

  removeBiasAll(): void {
    const s = this.sys.state;
    if (this.biasId !== null) {
      this.ensureBiasCompute();
      this.biasCompute!.computeBias();
      this.biasCompute!.removeBiasAll();
      this.nestedOn = true;
      return;
    }
    if (this.com) {
      this.chunk.ensure();
      const nch = this.chunk.nchunk;
      const { vcm } = this.chunkVcm(nch);
      this.vcmBias = vcm;
      this.biasOn = true;
      for (let i = 0; i < s.n; i++) {
        const c = this.counted(i, nch);
        if (c === 0) continue;
        for (let d = 0; d < 3; d++) s.v[3 * i + d] -= vcm[3 * (c - 1) + d];
      }
    }
  }

  restoreBiasAll(): void {
    const s = this.sys.state;
    if (this.biasId !== null) {
      this.biasCompute?.restoreBiasAll();
      this.nestedOn = false;
      return;
    }
    if (this.com && this.biasOn) {
      this.chunk.ensure();
      const nch = this.chunk.nchunk;
      for (let i = 0; i < s.n; i++) {
        const c = this.counted(i, nch);
        if (c === 0) continue;
        for (let d = 0; d < 3; d++) s.v[3 * i + d] += this.vcmBias[3 * (c - 1) + d];
      }
      this.biasOn = false;
    }
  }
}
