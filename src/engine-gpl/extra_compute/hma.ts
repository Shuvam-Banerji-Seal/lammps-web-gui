/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_hma.cpp and compute_hma.h (class ComputeHMA: the constructor, init, setup,
 * compute_vector, virial_compute, set_arrays), src/EXTRA-PAIR/pair_lj_smooth_linear.cpp
 * (PairLJSmoothLinear::single_hessian) and src/pair.cpp (Pair::hessian_twobody, the default single_hessian) at commit
 * b332f28. Contributing authors of compute hma (from the source): Apoorva Purohit, Andrew Schultz and David Kofke
 * (University at Buffalo). Copyright the LAMMPS development team (Sandia Corporation, GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * compute ID all hma temp-ID keyword ... (docs.lammps.org/compute_hma.html): the harmonically mapped averages of the
 * potential energy (u), the pressure (p Pharm) and the heat capacity (cv) of a crystal, from the displacements of
 * the atoms from their lattice sites (their unwrapped positions when the compute is defined) and the forces; with
 * anharmonic the first evaluation's lattice energy and pressure are subtracted. The set temperature is the thermostat
 * fix's t_target at the start of each run. cv sums the pair Hessian over the neighbor pairs: pair lj/smooth/linear
 * has it (single_hessian); for another pair style with a single() native's default single_hessian gives zeros.
 * Registered through src/engine/lazyStyles.ts (group gpl_hma, compute hma).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { nativeOrder } from '../../engine/atoms';
import type { CustomProp } from '../../engine/types';

/** Hessian tables of pair lj/smooth/linear (pair.lj1, lj2, native dljcut; index itype * nt + jtype). */
export interface LjSmoothLinearHessianTables {
  lj1: Float64Array;
  lj2: Float64Array;
  dljcut: Float64Array;
  nt: number;
}

/** The inputs of the phiSum loop of compute hma cv (the half neighbour list, ghosts mapped to their owners). */
export interface HmaPhiSumInput {
  inum: number;
  numneigh: ArrayLike<number>;
  firstneigh: ArrayLike<number>;
  neighbors: ArrayLike<number>;
  xall: ArrayLike<number>;
  typeall: ArrayLike<number>;
  owner: ArrayLike<number>;
  deltaR: ArrayLike<number>;
  cutsq: ArrayLike<number>;
  specialLJ: ArrayLike<number>;
  sbBits: number;
  neighMask: number;
  tables: LjSmoothLinearHessianTables;
}

/**
 * PairLJSmoothLinear::single_hessian at b332f28. Returns fforce (the force divided by r, times factor_lj) and writes the
 * six Hessian components (xx, xy, xz, yy, yz, zz) into d2u, as Pair::hessian_twobody does.
 */
export function ljSmoothLinearSingleHessian(
  t: LjSmoothLinearHessianTables, itype: number, jtype: number, rsq: number,
  delr: ArrayLike<number>, factorLJ: number, d2u: Float64Array,
): number {
  const k = itype * t.nt + jtype;
  const r2inv = 1.0 / rsq;
  const r6inv = r2inv * r2inv * r2inv;
  const rinv = Math.sqrt(r2inv);
  let forcelj = r6inv * (t.lj1[k] * r6inv - t.lj2[k]);
  forcelj = rinv * forcelj - t.dljcut[k];
  const fforce = factorLJ * forcelj * rinv;
  const d2r = factorLJ * r6inv * (13.0 * t.lj1[k] * r6inv - 7.0 * t.lj2[k]) / rsq;
  const dfac = -(fforce + d2r) / rsq;
  // Pair::hessian_twobody(fforce, dfac, delr, d2u)
  let m = 0;
  for (let a = 0; a < 3; a++) {
    d2u[m] = fforce;
    for (let b = a; b < 3; b++) {
      if (b > a) d2u[m] = 0;
      d2u[m++] += delr[a] * delr[b] * dfac;
    }
  }
  return fforce;
}

/** The phiSum of compute hma cv (compute_hma.cpp, computeCv branch), for a half list (native newton_pair on). */
export function hmaPhiSum(p: HmaPhiSumInput): number {
  const d2u = new Float64Array(6);
  const delr = [0, 0, 0];
  let phiSum = 0;
  for (let i = 0; i < p.inum; i++) {
    const k0 = p.firstneigh[i], k1 = k0 + p.numneigh[i];
    const itype = p.typeall[i];
    for (let kk = k0; kk < k1; kk++) {
      const entry = p.neighbors[kk];
      const factorLJ = p.specialLJ[entry >>> p.sbBits];
      const j = entry & p.neighMask;
      for (let c = 0; c < 3; c++) delr[c] = p.xall[3 * i + c] - p.xall[3 * j + c];
      const rsq = delr[0] * delr[0] + delr[1] * delr[1] + delr[2] * delr[2];
      const jtype = p.typeall[j];
      if (!(rsq < p.cutsq[itype * p.tables.nt + jtype])) continue;
      ljSmoothLinearSingleHessian(p.tables, itype, jtype, rsq, delr, factorLJ, d2u);
      const o = p.owner[j];
      let m = 0;
      for (let k = 0; k < 3; k++) {
        let a = 1.0;
        for (let l = k; l < 3; l++) {
          const idk = p.deltaR[3 * i + k], idl = p.deltaR[3 * i + l];
          const jdk = p.deltaR[3 * o + k], jdl = p.deltaR[3 * o + l];
          phiSum += a * (idk * jdl + jdk * idl) * d2u[m];
          phiSum -= a * (idk * idl * d2u[m] + jdk * jdl * d2u[m]);
          m++;
          if (k === l) a *= 2;
        }
      }
    }
  }
  return phiSum;
}

/** compute_hma.cpp: the unwrapped lattice site of an atom (Domain::unmap with its image flags). */
const unmap = (sys: System, i: number, out: number[]): void => {
  const s = sys.state, g = sys.geom;
  const xbox = s.image[3 * i], ybox = s.image[3 * i + 1], zbox = s.image[3 * i + 2];
  const x = s.x[3 * i], y = s.x[3 * i + 1], z = s.x[3 * i + 2];
  if (!s.box.triclinic) {
    out[0] = x + xbox * g.lx; out[1] = y + ybox * g.ly; out[2] = z + zbox * g.lz;
  } else {
    out[0] = x + g.lx * xbox + g.xy * ybox + g.xz * zbox;
    out[1] = y + g.ly * ybox + g.yz * zbox;
    out[2] = z + g.lz * zbox;
  }
};

export class ComputeHMA extends Compute {
  readonly style = 'hma';
  private readonly idTemp: string;
  private readonly key: string;
  private computeU = -1;
  private computeP = -1;
  private computeCv = -1;
  /** 0: plain; -1: anharmonic, the next evaluation takes the lattice energy and pressure; 1: anharmonic. */
  private returnAnharmonic = 0;
  private deltaPcap = 0;
  private finaltemp = 0;
  private uLat = 0;
  private pLat = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal compute hma command: missing argument(s)');
    if (group !== 'all') throw new StyleError('Compute hma must use group all');
    if (args[0] === 'NULL') throw new StyleError('fix ID specifying the set temperature of canonical simulation is required');
    this.idTemp = args[0];
    this.extscalar = 1;
    // the lattice sites: the STORE/ATOM fix of the native compute (x, y, z and a set flag, following the atoms)
    this.key = `#hma:${id}`;
    const s = sys.state;
    const store: CustomProp = { int: false, cols: 4, data: new Float64Array(4 * s.n) };
    s.custom.set(this.key, store);
    const site = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      unmap(sys, i, site);
      store.data[4 * i] = site[0]; store.data[4 * i + 1] = site[1]; store.data[4 * i + 2] = site[2]; store.data[4 * i + 3] = 1;
    }
    this.vectorFlag = true;
    this.extvector = -1;
    const extlist: number[] = [];
    let size = 0;
    for (let iarg = 1; iarg < args.length; iarg++) {
      const w = args[iarg];
      if (w === 'u') {
        if (this.computeU > -1) continue;
        this.computeU = size;
        extlist[size++] = 1;
      } else if (w === 'p') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal compute hma command');
        // a repeated p is skipped without its value, which is then read as a keyword (native)
        if (this.computeP > -1) continue;
        this.computeP = size;
        const v = Number(args[iarg + 1]);
        if (!Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${args[iarg + 1]}' in input script or data file`);
        this.deltaPcap = v;
        extlist[size++] = 0;
        iarg++;
      } else if (w === 'cv') {
        if (this.computeCv > -1) continue;
        this.computeCv = size;
        extlist[size++] = 1;
      } else if (w === 'anharmonic') {
        // the first time it is evaluated, the lattice pressure and energy are taken
        this.returnAnharmonic = -1;
      } else throw new StyleError(`Unknown compute hma keyword ${w}`);
    }
    if (size === 0) throw new StyleError('Illegal compute hma command');
    this.sizeVector = size;
    this.extlist = extlist;
    this.vector = new Float64Array(size);
  }

  /** init and setup: the pair checks of cv, and the set temperature (the thermostat's t_target at the start of the run). */
  init(): void {
    if (this.computeCv > -1) {
      const pair = this.sys.ff.pair;
      if (!pair) throw new StyleError('No pair style is defined for compute hma cv');
      if (!pair.single && pair.name !== 'lj/smooth/linear') throw new StyleError('Pair style does not support compute hma cv');
    }
    const f = this.sys.fixes.find((x) => x.id === this.idTemp);
    if (!f) throw new StyleError(`Could not find compute hma temperature fix ID ${this.idTemp}`);
    const ex = (f as unknown as { extract?: (n: string) => unknown }).extract;
    const tt = typeof ex === 'function' ? ex.call(f, 't_target') : null;
    if (typeof tt !== 'number') throw new StyleError(`Fix ID ${this.idTemp} is not a thermostat`);
    this.finaltemp = tt;
  }

  /** The sites store, with the sites of atoms created after the compute set to their positions (set_arrays). */
  private sites(): Float64Array {
    const s = this.sys.state;
    let c = s.custom.get(this.key);
    if (!c || c.data.length !== 4 * s.n) {
      const old = c?.data;
      c = { int: false, cols: 4, data: new Float64Array(4 * s.n) };
      if (old) c.data.set(old.subarray(0, Math.min(old.length, 4 * s.n)));
      s.custom.set(this.key, c);
    }
    const d = c.data;
    for (let i = 0; i < s.n; i++) {
      if (d[4 * i + 3] === 1) continue;
      d[4 * i] = s.x[3 * i]; d[4 * i + 1] = s.x[3 * i + 1]; d[4 * i + 2] = s.x[3 * i + 2]; d[4 * i + 3] = 1;
    }
    return d;
  }

  protected computeVector(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const a = sys.forces();
    const xorig = this.sites();
    const n = s.n;
    const dimension = s.dimension;
    const boltz = s.units.boltz, nktv2p = s.units.nktv2p;
    let u = 0.0;
    if (this.computeU > -1 || this.computeCv > -1) u = a.evdwl + a.ecoul + a.ebond + a.eangle + a.edihed + a.eimp;
    let p = 0, vol = 0;
    if (this.computeP > -1) {
      // virial_compute: pair, bonded and fix virials, then the kspace virial
      const w = new Float64Array(6);
      const add = (v: Float64Array) => { for (let c = 0; c < 6; c++) w[c] += v[c]; };
      add(a.virial); add(a.vbond); add(a.vangle); add(a.vdihed); add(a.vimp);
      sys.fixVirial(w);
      let virial = 0;
      for (let c = 0; c < dimension; c++) virial += w[c];
      for (let c = 0; c < dimension; c++) virial += a.vlong[c];
      p = virial;
      vol = g.lx * g.ly;
      if (dimension === 3) vol *= g.lz;
      p *= nktv2p / (dimension * vol);
      if (this.returnAnharmonic === -1) this.pLat = p;
    }
    // F . dr over the atoms (native order), dr the displacement from the lattice site
    const deltaR = new Float64Array(3 * n);
    let fdr = 0.0;
    const site = [0, 0, 0];
    for (const i of nativeOrder(s)) {
      unmap(sys, i, site);
      const dx = site[0] - xorig[4 * i], dy = site[1] - xorig[4 * i + 1], dz = site[2] - xorig[4 * i + 2];
      deltaR[3 * i] = dx; deltaR[3 * i + 1] = dy; deltaR[3 * i + 2] = dz;
      fdr += dx * s.f[3 * i] + dy * s.f[3 * i + 1] + dz * s.f[3 * i + 2];
    }
    let phiSum = 0.0;
    if (this.computeCv > -1) {
      const pair = sys.ff.pair!;
      if (pair.name === 'lj/smooth/linear') {
        const lj = pair as unknown as { cutsq: Float64Array; ntypes: number; lj1: Float64Array; lj2: Float64Array; dphiRc: Float64Array };
        const list = sys.nb.half;
        if (!list) throw new StyleError(`compute ${this.id} (hma): no half neighbour list is available`);
        const sLJ = (sys.ff as unknown as { specialLJ: Float64Array }).specialLJ;
        phiSum = hmaPhiSum({
          inum: list.inum, numneigh: list.numneigh, firstneigh: list.firstneigh, neighbors: list.neighbors,
          xall: sys.nb.xall, typeall: sys.nb.typeall, owner: sys.nb.owner, deltaR, cutsq: lj.cutsq, specialLJ: sLJ,
          sbBits: SBBITS, neighMask: NEIGHMASK,
          // the engine stores dphiRc = -dljcut (native dljcut = cutinv cut6inv (lj1 cut6inv - lj2), pair_lj_smooth_linear.cpp)
          tables: { lj1: lj.lj1, lj2: lj.lj2, dljcut: lj.dphiRc.map((v) => -v), nt: lj.ntypes + 1 },
        });
      }
      // another pair style: Pair::single_hessian's default sets d2u to zero, so the sum stays 0
    }
    const natoms = n;
    const uTotal = u;
    if (this.computeU > -1 || this.computeCv > -1) {
      if (this.returnAnharmonic === -1) this.uLat = uTotal;
      if (this.computeU > -1) {
        if (this.returnAnharmonic) this.vector[this.computeU] = uTotal - this.uLat + 0.5 * fdr;
        else this.vector[this.computeU] = uTotal + 0.5 * fdr + 0.5 * dimension * (natoms - 1) * boltz * this.finaltemp;
      }
    }
    if (this.computeP > -1) {
      const fv = ((this.deltaPcap) - (boltz * this.finaltemp * nktv2p * natoms / vol)) / (boltz * this.finaltemp * dimension * (natoms - 1));
      if (this.returnAnharmonic) this.vector[this.computeP] = p - this.pLat + (fv * fdr);
      else this.vector[this.computeP] = p + (fv * fdr) + this.deltaPcap;
    }
    if (this.computeCv > -1) {
      let buTot: number;
      if (this.returnAnharmonic) buTot = (uTotal - this.uLat + 0.5 * fdr) / this.finaltemp;
      else buTot = (uTotal + 0.5 * fdr) / this.finaltemp + 0.5 * dimension * (natoms - 1) * boltz;
      const one = -0.25 * (fdr + phiSum) / this.finaltemp;
      this.vector[this.computeCv] = one + buTot * buTot;
      if (!this.returnAnharmonic) this.vector[this.computeCv] += 0.5 * dimension * (natoms - 1);
    }
    // we have our lattice properties
    if (this.returnAnharmonic === -1) this.returnAnharmonic = 1;
  }
}

