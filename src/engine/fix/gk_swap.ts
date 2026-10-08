import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { massOf } from '../atoms';

/*
 * fix viscosity and fix thermal/conductivity (Muller-Plathe reverse NEMD), implemented
 * only from docs.lammps.org/fix_viscosity.html and docs.lammps.org/fix_thermal_conductivity.html
 * (sources plans/lammps-docs/fix_viscosity.rst, fix_thermal_conductivity.rst). Quotes below
 * are copied character for character from those files.
 *
 * fix viscosity: "fix ID group-ID viscosity N vdim pdim Nbin keyword value ..."
 * fix thermal/conductivity: "fix ID group-ID thermal/conductivity N edim Nbin keyword value ..."
 * "N = perform momentum exchange every N steps"; "Nbin = # of layers in pdim direction (must be
 * even number)"; "*swap* value = Nswap = number of swaps to perform every N steps";
 * "*vtarget* value = V or INF = target velocity of swap partners (velocity units)".
 * Defaults: "The option defaults are swap = 1 and vtarget = INF."
 *
 * Layers: "The simulation box is divided into *Nbin* layers in the *pdim* direction, where the
 * layer 1 is at the low end of that dimension and the layer *Nbin* is at the high end."
 * "The "middle" layer for momenta swapping is defined as the *Nbin*\ /2 + 1"
 * Measured with native LAMMPS (black box): a layer is [lo + k*w, lo + (k+1)*w), so an atom
 * exactly on an interior boundary belongs to the upper layer (z = 2.0 is not in layer 1 of
 * an 8-long box with Nbin 4) and the lower wall belongs to layer 1 (z = 0.0 is in layer 1).
 *
 * Selection (viscosity): "Nswap atoms in layer 1 with positive velocity components in the
 * *vdim* direction closest to the target value *V* are selected. Similarly, Nswap atoms in the
 * "middle" layer ... with negative velocity components in the *vdim* direction closest to the
 * negative of the target value *V* are selected." "When vtarget = INF, one or more atoms with
 * the most positive and negative velocity components are selected."
 * Selection (thermal): "The hottest Nswap atoms in layer 1 are selected. Similarly, the coldest
 * Nswap atoms in the "middle" layer ... are selected."
 * Measured with native LAMMPS (black box): "hottest" and "coldest" are by kinetic energy (an atom
 * of mass 2 at speed 1.5 is hotter than one of mass 1 at speed 2.0); the viscosity ranking is by
 * the velocity component alone (not momentum). Ties go to the first atom in atom order (lowest
 * id on a fresh system). The i-th ranked atom of layer 1 is paired with the i-th ranked atom of
 * the middle layer (Nswap = 2 and 3 measured).
 * Measured: a viscosity swap is skipped when the middle layer has no negative velocity component
 * (no candidates), and swaps happen at end-of-step multiples of N only, never in the setup of a run.
 *
 * Exchange: "This resets their velocities, typically in opposite" (viscosity) and "an
 * exchange of velocities relative to center of mass motion of the two atoms is performed, to
 * conserve kinetic energy" (thermal). Measured with native LAMMPS (black box): with unequal masses
 * both fixes reflect the pair's velocity about the pair's centre-of-mass velocity,
 * v' = 2 vcm - v, which conserves momentum and kinetic energy. viscosity reflects only the
 * vdim component; thermal reflects the full velocity vector. Equal masses reduce to swapping.
 *
 * Scalar: "The scalar is the cumulative momentum transferred between the bottom and middle of the
 * simulation box" (viscosity) and "The scalar is the cumulative kinetic energy transferred between
 * the bottom and middle of the simulation box" (thermal). Measured with native LAMMPS (black box):
 * viscosity adds the momentum change of the layer-1 atom, m (v' - v) (negative when momentum flows
 * down the box); thermal adds the kinetic energy gained by the middle-layer atom.
 *
 * Restrictions: "Some features or combination of settings in LAMMPS do not support non-orthogonal
 * boxes." -> this engine refuses triclinic boxes with a StyleError.
 */

type Dim = 0 | 1 | 2;
const DIMS: Record<string, Dim> = { x: 0, y: 1, z: 2 };

const parseCommon = (what: string, args: string[], nWord: string, dimWord: string, nbinWord: string) => {
  if (args.length < 3) throw new StyleError(`usage: ${what} ${nWord} ${dimWord} Nbin [keyword value ...]`);
  const n = Number(args[0]);
  if (!Number.isInteger(n) || n < 1) throw new StyleError(`${what}: ${nWord} must be a positive integer, got '${args[0]}'`);
  const d = DIMS[args[1]];
  if (d === undefined) throw new StyleError(`${what}: ${dimWord} must be x, y or z, got '${args[1]}'`);
  const nbin = Number(args[2]);
  if (!Number.isInteger(nbin) || nbin < 2) throw new StyleError(`${what}: ${nbinWord} must be an integer >= 2, got '${args[2]}'`);
  if (nbin % 2 !== 0) throw new StyleError(`${what}: ${nbinWord} must be an even number, got ${nbin}`);
  return { n, d, nbin };
};

const parseSwap = (what: string, args: string[], from: number, allowVtarget: boolean) => {
  let nswap = 1;
  let vtarget: number | null = null; // null = INF
  for (let k = from; k < args.length; k += 2) {
    const key = args[k];
    const val = args[k + 1];
    if (val === undefined) throw new StyleError(`${what}: keyword ${key} needs a value`);
    if (key === 'swap') {
      const m = Number(val);
      if (!Number.isInteger(m) || m < 1) throw new StyleError(`${what}: swap Nswap must be a positive integer, got '${val}'`);
      nswap = m;
    } else if (key === 'vtarget' && allowVtarget) {
      if (val === 'INF') vtarget = null;
      else {
        const v = Number(val);
        if (!Number.isFinite(v)) throw new StyleError(`${what}: vtarget must be a number or INF, got '${val}'`);
        vtarget = v;
      }
    } else {
      throw new StyleError(`${what}: unknown keyword '${key}'`);
    }
  }
  return { nswap, vtarget };
};

/** Layer index (0-based) of coordinate x in dimension d; periodic dims are wrapped first. */
const layerOf = (s: { box: { lo: number[]; hi: number[]; periodic: boolean[] } }, d: number, nbin: number, x: number): number => {
  const lo = s.box.lo[d];
  const L = s.box.hi[d] - lo;
  let rel = x - lo;
  if (s.box.periodic[d]) rel -= L * Math.floor(rel / L);
  const k = Math.floor((rel * nbin) / L);
  return Math.min(Math.max(k, 0), nbin - 1);
};

/** Atoms of the fix group in one layer, sorted by key (ascending), ties by atom index. */
const rank = (cands: number[], key: (i: number) => number): number[] =>
  cands.map((i) => ({ i, k: key(i) })).sort((a, b) => a.k - b.k || a.i - b.i).map((e) => e.i);

/** Velocity reflected about the pair's centre-of-mass velocity: v' = 2 vcm - v (conserves p and KE). */
const reflectPair = (ma: number, mb: number, va: number, vb: number): [number, number] => {
  const vcm = (ma * va + mb * vb) / (ma + mb);
  return [2 * vcm - va, 2 * vcm - vb];
};

abstract class FixLayerSwap extends Fix {
  protected readonly nEvery: number;
  protected readonly dim: Dim;
  protected readonly nbin: number;
  protected readonly nswap: number;
  protected scalarTotal = 0;

  constructor(sys: System, id: string, group: string, args: string[], nEvery: number, dim: Dim, nbin: number, nswap: number) {
    super(sys, id, group, args);
    this.nEvery = nEvery;
    this.nevery = nEvery;
    this.dim = dim;
    this.nbin = nbin;
    this.nswap = nswap;
    this.scalarFlag = true;
    this.extscalar = 0; // "The scalar value calculated by this fix is "intensive"."
  }

  computeScalar(): number { return this.scalarTotal; }

  /** Atoms of the fix group in layer k (0-based). */
  protected inLayer(k: number): number[] {
    const s = this.sys.state;
    const out: number[] = [];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (layerOf(s, this.dim, this.nbin, s.x[3 * i + this.dim]) === k) out.push(i);
    }
    return out;
  }

  protected checkBox(): void {
    if (this.sys.state.box.triclinic) throw new StyleError(`fix ${this.id}: triclinic boxes are not supported (fix viscosity keeps the box orthogonal)`);
  }
}

/** fix viscosity: momentum exchange of the vdim component (Muller-Plathe 1999). */
export class FixViscosity extends FixLayerSwap {
  readonly style = 'viscosity';
  private readonly vdim: Dim;
  private readonly vtarget: number | null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const what = `fix ${id} (viscosity)`;
    // fix ID group-ID viscosity N vdim pdim Nbin keyword value ...
    if (args.length < 4) throw new StyleError(`usage: fix ${id} group-ID viscosity N vdim pdim Nbin [keyword value ...]`);
    const n = Number(args[0]);
    if (!Number.isInteger(n) || n < 1) throw new StyleError(`${what}: N must be a positive integer, got '${args[0]}'`);
    const vdim = DIMS[args[1]];
    if (vdim === undefined) throw new StyleError(`${what}: vdim must be x, y or z, got '${args[1]}'`);
    const pdim = DIMS[args[2]];
    if (pdim === undefined) throw new StyleError(`${what}: pdim must be x, y or z, got '${args[2]}'`);
    const nbin = Number(args[3]);
    if (!Number.isInteger(nbin) || nbin < 2) throw new StyleError(`${what}: Nbin must be an integer >= 2, got '${args[3]}'`);
    if (nbin % 2 !== 0) throw new StyleError(`${what}: Nbin must be an even number, got ${nbin}`);
    const { nswap, vtarget } = parseSwap(what, args, 4, true);
    super(sys, id, group, args, n, pdim, nbin, nswap);
    this.vdim = vdim;
    this.vtarget = vtarget;
  }

  endOfStep(): void {
    if (this.sys.state.step % this.nEvery !== 0) return;
    this.checkBox();
    const s = this.sys.state;
    const vd = this.vdim;
    const mid = this.nbin / 2; // "Nbin/2 + 1" layer, 1-based
    const low = this.inLayer(0);
    const high = this.inLayer(mid);
    const V = this.vtarget;
    // layer 1: positive components, closest to V (INF: largest); middle: negative, closest to -V (INF: most negative)
    const lowCands = low.filter((i) => s.v[3 * i + vd] > 0);
    const midCands = high.filter((i) => s.v[3 * i + vd] < 0);
    const lowOrder = rank(lowCands, (i) => (V === null ? -s.v[3 * i + vd] : Math.abs(s.v[3 * i + vd] - V)));
    const midOrder = rank(midCands, (i) => (V === null ? s.v[3 * i + vd] : Math.abs(s.v[3 * i + vd] + V)));
    const pairs = Math.min(this.nswap, lowOrder.length, midOrder.length);
    for (let k = 0; k < pairs; k++) {
      const a = lowOrder[k];
      const b = midOrder[k];
      const ma = massOf(s, a);
      const mb = massOf(s, b);
      const va = s.v[3 * a + vd];
      const vb = s.v[3 * b + vd];
      const [va2, vb2] = reflectPair(ma, mb, va, vb);
      s.v[3 * a + vd] = va2;
      s.v[3 * b + vd] = vb2;
      this.scalarTotal += ma * (va2 - va);
    }
  }
}

/** fix thermal/conductivity: kinetic-energy exchange (Muller-Plathe 1997). */
export class FixThermalConductivity extends FixLayerSwap {
  readonly style = 'thermal/conductivity';

  constructor(sys: System, id: string, group: string, args: string[]) {
    const what = `fix ${id} (thermal/conductivity)`;
    const { n, d, nbin } = parseCommon(what, args, 'N', 'edim', 'Nbin');
    const { nswap } = parseSwap(what, args, 3, false);
    super(sys, id, group, args, n, d, nbin, nswap);
  }

  endOfStep(): void {
    if (this.sys.state.step % this.nEvery !== 0) return;
    this.checkBox();
    const s = this.sys.state;
    const mvv2e = s.units.mvv2e;
    const ke = (i: number): number => 0.5 * massOf(s, i) * (s.v[3 * i] ** 2 + s.v[3 * i + 1] ** 2 + s.v[3 * i + 2] ** 2);
    const mid = this.nbin / 2;
    const hot = rank(this.inLayer(0), (i) => -ke(i));
    const cold = rank(this.inLayer(mid), (i) => ke(i));
    const pairs = Math.min(this.nswap, hot.length, cold.length);
    for (let k = 0; k < pairs; k++) {
      const a = hot[k];
      const b = cold[k];
      const ma = massOf(s, a);
      const mb = massOf(s, b);
      const keBefore = 0.5 * mb * (s.v[3 * b] ** 2 + s.v[3 * b + 1] ** 2 + s.v[3 * b + 2] ** 2);
      const cm = [0, 1, 2].map((c) => (ma * s.v[3 * a + c] + mb * s.v[3 * b + c]) / (ma + mb));
      for (let c = 0; c < 3; c++) {
        const va = s.v[3 * a + c];
        const vb = s.v[3 * b + c];
        s.v[3 * a + c] = 2 * cm[c] - va;
        s.v[3 * b + c] = 2 * cm[c] - vb;
      }
      const keAfter = 0.5 * mb * (s.v[3 * b] ** 2 + s.v[3 * b + 1] ** 2 + s.v[3 * b + 2] ** 2);
      this.scalarTotal += (keAfter - keBefore) * mvv2e;
    }
  }
}
