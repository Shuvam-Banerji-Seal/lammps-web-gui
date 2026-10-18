/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EFF/compute_temp_deform_eff.cpp
 * and compute_temp_deform_eff.h (class ComputeTempDeformEff, compute temp/deform/eff): the temperature of an electron
 * force field system after the streaming velocity of the box deformation (fix deform) is removed from every atom. It
 * wraps a temperature compute (by default its own temp/eff, ID fixID_temp), and its scalar and vector add the electron
 * radial kinetic energy of the wrapped compute's dof. Contributing author (from the source): Andres Jaramillo-Botero
 * (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The box rates come from the fix deform of the system (domain->h_rate and h_ratelo of fix_deform.cpp init); the
 * inverse box matrix is domain.cpp set_global_box, and x2lamda is domain.cpp Domain::x2lamda. The arithmetic follows
 * the C++ order.
 */
import { Compute } from '../../engine/compute/compute';
import { ComputeTemp } from '../../engine/compute/temp';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';
import { ComputeTempEff } from '../eff/compute_temp_eff';
import { exp as libmExp } from '../libm/exp';

/** domain.h h_rate and h_ratelo (zero without a fix deform). */
export interface DeformRates { hRate: number[]; hRatelo: number[] }

/** The box rates of the system's fix deform (fix_deform.cpp init: h_rate[i] = dhi_dt - dlo_dt, h_ratelo[i] = dlo_dt). */
export const deformRates = (sys: System): DeformRates => {
  const f = sys.fixes.find((x) => x.style === 'deform') as unknown as { lenDot: number[]; tiltDot: number[]; loDot: number[] } | undefined;
  if (!f) return { hRate: [0, 0, 0, 0, 0, 0], hRatelo: [0, 0, 0] };
  // domain.h order: the length rates, then yz, xz and xy (the tilt rates)
  return {
    hRate: [f.lenDot[0], f.lenDot[1], f.lenDot[2], f.tiltDot[2], f.tiltDot[1], f.tiltDot[0]],
    hRatelo: [f.loDot[0], f.loDot[1], f.loDot[2]],
  };
};

/** domain.cpp set_global_box: h (xprd, yprd, zprd, yz, xz, xy) and the inverse h_inv. */
export const boxH = (sys: System): { h: number[]; hInv: number[] } => {
  const box = sys.state.box;
  const h = [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2], 0, 0, 0];
  if (box.triclinic) {
    h[3] = box.tilt[2]; // yz
    h[4] = box.tilt[1]; // xz
    h[5] = box.tilt[0]; // xy
  }
  const hInv = [1.0 / h[0], 1.0 / h[1], 1.0 / h[2], 0.0, 0.0, 0.0];
  if (box.triclinic) {
    hInv[3] = -h[3] / (h[1] * h[2]);
    hInv[4] = (h[3] * h[5] - h[1] * h[4]) / (h[0] * h[1] * h[2]);
    hInv[5] = -h[5] / (h[0] * h[1]);
  }
  return { h, hInv };
};

/** math_extra.h multiply_shape_shape: the velocity gradient grad_u = h_rate h_inv (upper-triangular products). */
export const multiplyShapeShape = (one: readonly number[], two: readonly number[]): number[] => [
  one[0] * two[0],
  one[1] * two[1],
  one[2] * two[2],
  one[1] * two[3] + one[3] * two[2],
  one[0] * two[4] + one[5] * two[3] + one[4] * two[2],
  one[0] * two[5] + one[5] * two[1],
];

export class ComputeTempDeformEff extends Compute {
  readonly style: string = 'temp/deform/eff';
  /** keyword temp: the ID of the wrapped temperature compute, or null for the compute's own temp/eff (ID id_temp) */
  private readonly tempId: string | null;
  /** the compute's own temp/eff (created when there is no temp keyword; native post_constructor) */
  private readonly own: ComputeTempEff | null;
  private temperature: Compute | null = null;
  private tcomputeEff = true;
  private dynamicEff = false;
  private tfactor = 1;
  /** the bias removed by the last remove_deform_bias (member vbias of the C++ class) */
  private vbias = [0, 0, 0];
  private vbiasall = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    let tempId: string | null = null;
    for (let k = 0; k < args.length;) {
      if (args[k] !== 'temp') throw new StyleError(`Unknown compute temp/deform/eff keyword: ${args[k]}`);
      if (args[k + 1] === undefined) throw new StyleError('Illegal compute temp/deform/eff command: missing argument for temp');
      tempId = args[k + 1];
      k += 2;
    }
    if (!isElectronStyle(sys.state.atomStyle)) throw new StyleError('Compute temp/deform/eff requires atom style electron');
    this.tempId = tempId;
    this.own = tempId === null ? new ComputeTempEff(sys, `${id}_temp`, group, []) : null;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.vector = new Float64Array(6);
    this.extscalar = 0;
    this.extvector = 1;
    this.tempFlag = true;
  }

  /** compute_temp_deform_eff.cpp init: the wrapped compute must exist, compute a temperature and share the group. */
  override init(): void {
    const id = this.tempId ?? this.own!.id;
    const t = this.tempId === null ? this.own! : this.sys.computes.find((c) => c.id === id);
    if (!t) throw new StyleError(`Temperature ID ${id} for compute temp/deform/eff does not exist`);
    if (!t.tempFlag) throw new StyleError(`Compute temp/deform/eff temperature ID ${id} does not compute temperature`);
    if (t.groupBit !== this.groupBit) throw new StyleError(`Group of temperature compute with ID ${id} for compute temp/deform/eff does not match`);
    if (/^temp\/deform/.test(t.style)) throw new StyleError('Compute temp/deform/eff internal temperature compute cannot be of style temp/deform');
    if (t.hasBias()) throw new StyleError('compute temp/deform/eff with a velocity-bias temperature compute is not supported by the browser engine yet');
    this.temperature = t;
    this.tcomputeEff = /\/eff$/.test(t.style);
    this.dynamicEff = this.dynamicDof;
    t.init();
    if (t instanceof ComputeTemp) t.dofCompute();
    this.dofCompute();
  }

  /** compute_temp_deform_eff.cpp dof_compute: the wrapped dof, less dimension per electron for a non-eff temperature. */
  private dofCompute(): void {
    const t = this.temperature!;
    let dof = t.dof;
    if (!this.tcomputeEff) {
      const s = this.sys.state;
      const spin = s.custom.get(ELECTRON_SPIN)!.data;
      let electrons = 0;
      for (let i = 0; i < s.n; i++) if ((s.mask[i] & this.groupBit) && Math.abs(spin[i]) === 1) electrons++;
      dof -= this.sys.dimension * electrons;
    }
    this.tfactor = dof > 0 ? t.dof / dof : 0.0;
    this.dof = dof;
  }

  /** The electron radial term of the scalar and vector (compute_temp_deform_eff.cpp: "tspin" for a non-eff temperature). */
  private electronTerm(): number {
    const s = this.sys.state;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const mefactor = this.sys.dimension / 4.0;
    let tspin = 0;
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && Math.abs(spin[i]) === 1) tspin += mefactor * massOf(s, i) * ervel[i] * ervel[i];
    }
    return tspin;
  }

  /** compute_temp_deform_eff.cpp compute_scalar: the wrapped scalar of the thermal velocities, times tfactor. */
  protected computeScalar(): number {
    const s = this.sys.state;
    if (!s.massByType) return 0;
    this.removeDeformBiasAll();
    const t = this.temperature!;
    t.invalidate();
    let scalar = t.scalarValue();
    if (this.dynamicEff) this.dofCompute();
    scalar *= this.tfactor;
    if (!this.tcomputeEff) {
      const tspin = this.electronTerm();
      scalar += tspin * s.units.mvv2e / (this.dof * s.units.boltz);
    }
    this.restoreDeformBiasAll();
    return scalar;
  }

  /** compute_temp_deform_eff.cpp compute_vector: the wrapped tensor, the electron radial term on the diagonal. */
  protected computeVector(): void {
    const s = this.sys.state;
    this.removeDeformBiasAll();
    const t = this.temperature!;
    t.invalidate();
    const src = t.vectorValues();
    for (let c = 0; c < 6; c++) this.vector[c] = src[c];
    if (this.dynamicEff) this.dofCompute();
    if (!this.tcomputeEff) {
      const tspin = this.electronTerm();
      for (let c = 0; c < 3; c++) this.vector[c] += tspin * s.units.mvv2e;
    }
    this.restoreDeformBiasAll();
  }

  /** remove_deform_bias: the streaming velocity of the box at the position of atom i, lamda from the inverse box. */
  private biasAt(i: number, out: number[]): void {
    const s = this.sys.state;
    const { hInv } = boxH(this.sys);
    const { hRate, hRatelo } = deformRates(this.sys);
    const box = s.box;
    const dx = s.x[3 * i] - box.lo[0], dy = s.x[3 * i + 1] - box.lo[1], dz = s.x[3 * i + 2] - box.lo[2];
    // domain.cpp Domain::x2lamda
    const l0 = hInv[0] * dx + hInv[5] * dy + hInv[4] * dz;
    const l1 = hInv[1] * dy + hInv[3] * dz;
    const l2 = hInv[2] * dz;
    out[0] = hRate[0] * l0 + hRate[5] * l1 + hRate[4] * l2 + hRatelo[0];
    out[1] = hRate[1] * l1 + hRate[3] * l2 + hRatelo[1];
    out[2] = hRate[2] * l2 + hRatelo[2];
  }

  /** remove_deform_bias(i, v): v minus the streaming velocity of atom i (the bias is kept for restoreDeformBiasVec). */
  removeDeformBiasVec(i: number, v: number[]): void {
    this.biasAt(i, this.vbias);
    v[0] -= this.vbias[0];
    v[1] -= this.vbias[1];
    v[2] -= this.vbias[2];
  }

  /** restore_deform_bias(i, v): adds back the bias of the last removeDeformBiasVec. */
  restoreDeformBiasVec(_i: number, v: number[]): void {
    v[0] += this.vbias[0];
    v[1] += this.vbias[1];
    v[2] += this.vbias[2];
  }

  /** remove_deform_bias_all: the streaming velocity of every group atom, stored for restoreDeformBiasAll. */
  removeDeformBiasAll(): void {
    const s = this.sys.state;
    if (this.vbiasall.length < 3 * s.n) this.vbiasall = new Float64Array(3 * s.n);
    const b = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.biasAt(i, b);
      this.vbiasall[3 * i] = b[0]; this.vbiasall[3 * i + 1] = b[1]; this.vbiasall[3 * i + 2] = b[2];
      s.v[3 * i] -= b[0]; s.v[3 * i + 1] -= b[1]; s.v[3 * i + 2] -= b[2];
    }
  }

  /** restore_deform_bias_all: adds back the streaming velocities removed by removeDeformBiasAll. */
  restoreDeformBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      s.v[3 * i] += this.vbiasall[3 * i];
      s.v[3 * i + 1] += this.vbiasall[3 * i + 1];
      s.v[3 * i + 2] += this.vbiasall[3 * i + 2];
    }
  }

  /**
   * apply_deform_bias_all(dtv): adds the streaming velocity at the half-step positions (the boundary lo moved by dtv with
   * the flow exp(grad_u dtv); the box may not be updated yet, so the gradient is taken from the rates).
   */
  applyDeformBiasAll(dtv: number): void {
    const s = this.sys.state;
    const { hInv } = boxH(this.sys);
    const { hRate } = deformRates(this.sys);
    const gradU = multiplyShapeShape(hRate, hInv);
    const box = s.box;
    const xmid = [0, 1, 2].map((d) => (box.hi[d] + box.lo[d]) / 2.0);
    const ylo = xmid[1] + (box.lo[1] - xmid[1]) * libmExp(gradU[1] * dtv);
    const zlo = xmid[2] + (box.lo[2] - xmid[2]) * libmExp(gradU[2] * dtv);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const x0 = s.x[3 * i], x1 = s.x[3 * i + 1], x2 = s.x[3 * i + 2];
      s.v[3 * i] += (x0 - xmid[0]) * gradU[0] + (x1 - ylo) * gradU[5] + (x2 - zlo) * gradU[4];
      s.v[3 * i + 1] += (x1 - xmid[1]) * gradU[1] + (x2 - zlo) * gradU[3];
      s.v[3 * i + 2] += (x2 - xmid[2]) * gradU[2];
    }
  }

  // ---- the velocity bias of the Compute interface (index based; used by fixes that thermostat the thermal velocity)
  override hasBias(): boolean { return true; }
  override removeBias(i: number): void {
    const s = this.sys.state;
    const v = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
    this.removeDeformBiasVec(i, v);
    s.v[3 * i] = v[0]; s.v[3 * i + 1] = v[1]; s.v[3 * i + 2] = v[2];
  }
  override restoreBias(i: number): void {
    const s = this.sys.state;
    const v = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
    this.restoreDeformBiasVec(i, v);
    s.v[3 * i] = v[0]; s.v[3 * i + 1] = v[1]; s.v[3 * i + 2] = v[2];
  }
  override removeBiasAll(): void { this.removeDeformBiasAll(); }
  override restoreBiasAll(): void { this.restoreDeformBiasAll(); }
}
