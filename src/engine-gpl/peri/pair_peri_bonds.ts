/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/PERI/pair_peri.cpp and pair_peri.h (PairPeri base: compute_dilatation, influence_function, the
 * short-range contact term), src/PERI/fix_peri_neigh.cpp (the bond family: partners, r0, wvolume, vinter, the
 * per-bond history arrays) and src/PERI/compute_damage_atom.cpp at commit b332f28. Copyright Sandia Corporation
 * (GNU General Public License); original authors Mike Parks (SNL), Rezwanur Rahman and J.T. Foster (UTSA).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Shared machinery of pair_style peri/ves and peri/eps (the viscoelastic and elastoplastic state-based peridynamic
 * styles). docs.lammps.org/pair_peri.html: "pair_style peri/ves" and "pair_style peri/eps" are the Peridynamic
 * viscoelastic and elastic-plastic models. The bond family is stored per directed partner entry, as native LAMMPS
 * stores it per partner[i][jj]: a bond broken on one side only is never seen again from the other side, and the
 * per-bond deviator histories belong to the directed entry. Partner positions are taken through the image offset
 * of the ghost the neighbor list gave (the bond vector of the minimum image).
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';

/** Numerical zero of LAMMPS pair_peri.h (NEAR_ZERO); the influence function is capped at 1 / NEAR_ZERO. */
export const PERI_NEAR_ZERO = 2.2204e-16;
/** DBL_MAX of the C code: the no-breaking sentinel of smin and the unset s0. */
const DBL_MAX = Number.MAX_VALUE;

/** pair_peri.h influence_function: omega(xi) = 1 / |xi| (capped at 1 / NEAR_ZERO). */
export const periInfluence = (dx: number, dy: number, dz: number): number => {
  const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
  return Math.abs(r) < PERI_NEAR_ZERO ? 1 / PERI_NEAR_ZERO : 1 / r;
};

/**
 * Near-horizon volume scaling of pair_peri.cpp (compute_dilatation and the force loops): a partner at distance
 * r0 from a horizon delta within half a lattice constant has its volume scaled linearly down to 0.
 */
export const periVfracScale = (r0: number, delta: number, halfLc: number): number => {
  if (Math.abs(r0 - delta) <= halfLc) return (-1.0 / (2 * halfLc)) * r0 + (1.0 + (delta - halfLc) / (2 * halfLc));
  return 1.0;
};

/** Per-step data shared by the bond loops of one compute call. */
export interface PeriStep {
  pc: PairCompute;
  nl: number;
  x: Float64Array;
  x0: Float64Array;
  f: Float64Array;
  V: Float64Array;
  type: Int32Array;
  /** update->dt (the viscoelastic relaxation uses it). */
  dt: number;
  /** domain->lattice->xlattice. */
  lc: number;
}

/** The geometry of one directed partner entry in the current configuration (the scratch of bondGeom). */
export const PERI_GEOM_LEN = 12;
export const G_DX = 0, G_DY = 1, G_DZ = 2, G_R = 3, G_DR = 4, G_R0 = 5, G_OM = 6, G_VS = 7, G_J = 8, G_DX0 = 9, G_DY0 = 10, G_DZ0 = 11;

export abstract class PeriBondPair extends Pair {
  abstract readonly name: string;
  /** The coefficient names in pair_coeff order (after I and J). */
  protected abstract readonly names: readonly string[];
  needsHalf = true;
  needsFull = false;
  /** The virial is tallied bond by bond, as in LAMMPS (no_virial_fdotr_compute = 1). */
  virialFdotr = false;

  protected pp!: PairParams;
  protected xlattice = 1;

  // --- the bond family (fix PERI_NEIGH): CSR over the owned atoms, one entry per directed partner ---
  protected built = false;
  protected nBuilt = 0;
  protected idsBuilt = new Int32Array(0);
  /** pStart[i] .. pStart[i+1]: the partner entries of owned atom i. */
  protected pStart = new Int32Array(1);
  /** partner's owned index (its ghost image is resolved through pSh). */
  protected pJ = new Int32Array(0);
  /** image offset of the partner in current coordinates (3 per entry). */
  protected pSh = new Float64Array(0);
  /** reference bond length r0 (the neighbor distance at the first evaluation). */
  protected pR0 = new Float64Array(0);
  /** 1 when partner[i][jj] is 0 (the bond is broken). */
  protected pBr = new Uint8Array(0);
  /** per-entry histories: VES deviatorextention / deviatorBackextention; EPS deviatorPlasticextension in devA. */
  protected devA = new Float64Array(0);
  protected devB = new Float64Array(0);
  /** per owned atom: vinter (sum of the partner volumes at setup), wvolume, dilatation theta. */
  protected vinter = new Float64Array(0);
  protected wvol = new Float64Array(0);
  protected theta = new Float64Array(0);
  /** per owned atom: the minimum stretch of the last step (smin, the no-breaking value -DBL_MAX) and s0. */
  protected smin = new Float64Array(0);
  protected s0 = new Float64Array(0);

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.pp = new PairParams(ntypes, this.names);
  }

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length) throw new StyleError(`pair_style ${this.name} takes no arguments`);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    const usage = `usage: pair_coeff I J ${this.names.join(' ')}`;
    if (args.length !== this.names.length + 2) throw new StyleError(usage);
    const vals = this.names.map((nm, k) => parseNum(args[k + 2], nm));
    if (!(vals[this.names.indexOf('cut')] > 0)) throw new StyleError(`${this.name}: cut (horizon) must be > 0`);
    this.pp.setRange(args[0], args[1], vals);
    this.built = false;
  }

  initStyle(ctx: StyleContext): void {
    // LAMMPS PairPeri::init_style: an atom map, a lattice, and the same lattice constant in x, y, z
    this.xlattice = ctx.xlattice ?? 1;
  }

  initOne(i: number, j: number): number {
    if (!this.pp.isSet(i, j)) throw new StyleError(`All pair coeffs are not set for ${this.name} (types ${i} ${j})`);
    return this.pp.get('cut', i, j);
  }

  /** The coefficient of the type pair (I, J) of a name. */
  protected c(name: string, i: number, j: number): number {
    return this.pp.get(name, i, j);
  }

  /**
   * The bond family from the neighbor list at the first evaluation (fix PERI_NEIGH setup): every pair within the
   * horizon (rsq <= cutsq of the type pair) gives one partner entry on each side, with r0 from the current
   * positions, and wvolume and vinter from the reference positions. The setup sum uses the owner's x0 without the
   * image offset, as src/PERI/fix_peri_neigh.cpp (b332f28) reads; the periodic case is not measured (the oracle
   * cases are non-periodic).
   */
  protected buildFamily(pc: PairCompute): void {
    const s = pc.s;
    const list = pc.half;
    if (!list) throw new StyleError(`${this.name} needs a half neighbor list`);
    const nl = pc.nlocal, x = pc.x, x0 = s.x0!, V = s.vfrac!, owner = pc.nb.owner, nbr = list.neighbors, type = pc.type;
    const perI: Array<Array<[number, number, number, number, number]>> = [];
    for (let i = 0; i < nl; i++) perI.push([]);
    for (let i = 0; i < list.inum; i++) {
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbr[k] & NEIGHMASK;
        const o = j < nl ? j : owner[j];
        let sx = 0, sy = 0, sz = 0;
        if (j >= nl) {
          // a periodic image: the offset of the ghost from its owner in the current configuration
          sx = x[3 * j] - x[3 * o]; sy = x[3 * j + 1] - x[3 * o + 1]; sz = x[3 * j + 2] - x[3 * o + 2];
        }
        if (o === i && sx === 0 && sy === 0 && sz === 0) continue;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const cut = this.c('cut', type[i], type[o]);
        // LAMMPS fix_peri_neigh setup: "if (rsq <= cutsq[itype][jtype]) partner"
        if (rsq > cut * cut) continue;
        const r0 = Math.sqrt(rsq);
        perI[i].push([o, sx, sy, sz, r0]);
        perI[o].push([i, -sx, -sy, -sz, r0]);
      }
    }
    let np = 0;
    for (let i = 0; i < nl; i++) np += perI[i].length;
    this.pStart = new Int32Array(nl + 1);
    this.pJ = new Int32Array(np);
    this.pSh = new Float64Array(3 * np);
    this.pR0 = new Float64Array(np);
    this.pBr = new Uint8Array(np);
    this.devA = new Float64Array(np);
    this.devB = new Float64Array(np);
    this.vinter = new Float64Array(nl);
    this.wvol = new Float64Array(nl);
    this.theta = new Float64Array(nl);
    this.smin = new Float64Array(nl).fill(-DBL_MAX);
    this.s0 = new Float64Array(nl).fill(DBL_MAX);
    let e = 0;
    for (let i = 0; i < nl; i++) {
      this.pStart[i] = e;
      for (const [o, sx, sy, sz, r0] of perI[i]) {
        this.pJ[e] = o;
        this.pSh[3 * e] = sx; this.pSh[3 * e + 1] = sy; this.pSh[3 * e + 2] = sz;
        this.pR0[e] = r0;
        this.vinter[i] += V[o];
        e++;
      }
    }
    this.pStart[nl] = e;
    const lc = this.xlattice, halfLc = 0.5 * lc;
    for (let i = 0; i < nl; i++) {
      for (let k = this.pStart[i]; k < this.pStart[i + 1]; k++) {
        const o = this.pJ[k];
        const dx0 = x0[3 * i] - x0[3 * o], dy0 = x0[3 * i + 1] - x0[3 * o + 1], dz0 = x0[3 * i + 2] - x0[3 * o + 2];
        const rsq0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
        const delta = this.c('cut', type[i], type[o]);
        const vs = periVfracScale(this.pR0[k], delta, halfLc);
        this.wvol[i] += periInfluence(dx0, dy0, dz0) * rsq0 * V[o] * vs;
      }
    }
    this.idsBuilt = s.id.slice(0, nl);
    this.nBuilt = nl;
    this.built = true;
  }

  /** Fills scratch with the geometry of directed entry k of owned atom i (the minimum-image-equivalent offsets). */
  protected bondGeom(st: PeriStep, i: number, k: number, g: Float64Array): void {
    const { x, x0, type, lc } = st;
    const j = this.pJ[k];
    const sx = this.pSh[3 * k], sy = this.pSh[3 * k + 1], sz = this.pSh[3 * k + 2];
    const dx = x[3 * i] - (x[3 * j] + sx), dy = x[3 * i + 1] - (x[3 * j + 1] + sy), dz = x[3 * i + 2] - (x[3 * j + 2] + sz);
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const r0 = this.pR0[k];
    const ti = type[i], tj = type[j];
    const dx0 = x0[3 * i] - (x0[3 * j] + sx), dy0 = x0[3 * i + 1] - (x0[3 * j + 1] + sy), dz0 = x0[3 * i + 2] - (x0[3 * j + 2] + sz);
    let dr = r - r0;
    // avoid roundoff errors (pair_peri_ves.cpp: "if (fabs(dr) < NEAR_ZERO) dr = 0.0")
    if (Math.abs(dr) < PERI_NEAR_ZERO) dr = 0.0;
    g[G_DX] = dx; g[G_DY] = dy; g[G_DZ] = dz; g[G_R] = r; g[G_DR] = dr; g[G_R0] = r0;
    g[G_OM] = periInfluence(dx0, dy0, dz0);
    g[G_VS] = periVfracScale(r0, this.c('cut', ti, tj), 0.5 * lc);
    g[G_J] = j;
    g[G_DX0] = dx0; g[G_DY0] = dy0; g[G_DZ0] = dz0;
  }

  /** Dilatation theta_i of every owned atom over its unbroken partners (compute_dilatation). */
  protected updateDilatation(st: PeriStep): void {
    const { nl, V } = st;
    const g = new Float64Array(PERI_GEOM_LEN);
    for (let i = 0; i < nl; i++) {
      let th = 0;
      for (let k = this.pStart[i]; k < this.pStart[i + 1]; k++) {
        if (this.pBr[k]) continue;
        this.bondGeom(st, i, k, g);
        const j = this.pJ[k];
        th += g[G_OM] * g[G_R0] * g[G_DR] * V[j] * g[G_VS];
      }
      this.theta[i] = this.wvol[i] !== 0 ? (3.0 / this.wvol[i]) * th : 0;
    }
  }

  /** Short-range contact forces (pair_peri.cpp / pair_peri_ves.cpp, the half list) and their energy and virial. */
  protected contacts(st: PeriStep): void {
    const { pc, x, x0, f, V, type, lc } = st;
    const list = pc.half;
    if (!list) throw new StyleError(`${this.name} needs a half neighbor list`);
    const acc = pc.acc, vir = acc.virial, owner = pc.nb.owner, nl = st.nl, nbr = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbr[k] & NEIGHMASK;
        const o = j < nl ? j : owner[j];
        const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        // reference separation through the image of the partner (the image offset of the ghost)
        const sx = j >= nl ? x[3 * j] - x[3 * o] : 0, sy = j >= nl ? x[3 * j + 1] - x[3 * o + 1] : 0, sz = j >= nl ? x[3 * j + 2] - x[3 * o + 2] : 0;
        const dx0 = x0[3 * i] - (x0[3 * o] + sx), dy0 = x0[3 * i + 1] - (x0[3 * o + 1] + sy), dz0 = x0[3 * i + 2] - (x0[3 * o + 2] + sz);
        const rsq0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
        const ti = type[i], tj = type[j];
        const cut = this.c('cut', ti, tj);
        const r = Math.sqrt(rsq);
        // short-range interaction distance from the initial positions (0.9 and 1.35 are constants)
        const dij = Math.min(0.9 * Math.sqrt(rsq0), 1.35 * lc);
        if (!(r < dij)) continue;
        const dr = r - dij;
        // kshort: the PMB short-range constant (15 * 18 K / (pi cut^4)), with the K of the owned atom's type
        const kshort = (15.0 * 18.0 * this.c('K', ti, ti)) / (Math.PI * (cut * cut) * (cut * cut));
        const rk = (kshort * V[o]) * (dr / cut);
        const fpair = r > 0 ? -(rk / r) : 0;
        f[3 * i] += delx * fpair; f[3 * i + 1] += dely * fpair; f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair; f[3 * j + 1] -= dely * fpair; f[3 * j + 2] -= delz * fpair;
        acc.evdwl += 0.5 * rk * dr;
        const fv = fpair * V[i];
        vir[0] += delx * delx * fv; vir[1] += dely * dely * fv; vir[2] += delz * delz * fv;
        vir[3] += delx * dely * fv; vir[4] += delx * delz * fv; vir[5] += dely * delz * fv;
      }
    }
  }

  /** Volumetric energy 0.5 K theta^2 of every owned atom (the eflag part of LAMMPS compute). */
  protected volumetricEnergy(st: PeriStep): void {
    let e = 0;
    for (let i = 0; i < st.nl; i++) {
      const K = this.c('K', st.type[i], st.type[i]);
      e += 0.5 * K * this.theta[i] * this.theta[i];
    }
    st.pc.acc.evdwl += e;
  }

  /** Tallies a directed bond term of LAMMPS ev_tally(i, i, nlocal, 0, 0.5 * evdwl, 0, 0.5 * fbond * V_i, del). */
  protected tallyBond(st: PeriStep, i: number, evdwl: number, fbond: number, g: Float64Array): void {
    st.pc.acc.evdwl += 0.5 * evdwl;
    const fp = 0.5 * fbond * st.V[i];
    const vir = st.pc.acc.virial;
    const dx = g[G_DX], dy = g[G_DY], dz = g[G_DZ];
    vir[0] += dx * dx * fp; vir[1] += dy * dy * fp; vir[2] += dz * dz * fp;
    vir[3] += dx * dy * fp; vir[4] += dx * dz * fp; vir[5] += dy * dz * fp;
  }

  /** Bond breaking test and the minimum-stretch bookkeeping of one directed entry (the shared end of the loops). */
  protected breakAndTrack(i: number, k: number, stretch: number, smin_new: Float64Array, s0_new: Float64Array, first: boolean, ti: number, tj: number, j: number): void {
    const s00 = this.c('s00', ti, tj), alpha = this.c('alpha', ti, tj);
    if (stretch > s00 - alpha * Math.max(this.smin[i], this.smin[j])) this.pBr[k] = 1;
    if (stretch < smin_new[i]) smin_new[i] = stretch;
    const cand = s00 - alpha * stretch;
    s0_new[i] = first ? cand : Math.max(s0_new[i], cand);
  }

  /** Subclass bond loop: the force on every owned atom from its unbroken partners (and the history update). */
  protected abstract bondForces(st: PeriStep, smin_new: Float64Array, s0_new: Float64Array): void;

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.vfrac || !s.x0) throw new StyleError(`pair_style ${this.name} needs atom_style peri`);
    const nl = pc.nlocal;
    if (!this.built) this.buildFamily(pc);
    else {
      let same = nl === this.nBuilt;
      for (let i = 0; same && i < nl; i++) same = s.id[i] === this.idsBuilt[i];
      if (!same) throw new StyleError(`pair_style ${this.name}: the peridynamic bonds do not match the atoms (atoms were added or deleted after the first run)`);
    }
    const st: PeriStep = { pc, nl, x: pc.x, x0: s.x0, f: pc.f, V: s.vfrac, type: pc.type, dt: s.dt, lc: this.xlattice };
    this.contacts(st);
    this.updateDilatation(st);
    this.volumetricEnergy(st);
    const smin_new = new Float64Array(nl).fill(DBL_MAX);
    const s0_new = new Float64Array(nl);
    this.bondForces(st, smin_new, s0_new);
    // store the new minimum stretch (breaking in the next step) and s0 (diagnostic); an atom with no surviving
    // bonds keeps the no-breaking sentinel (pair_peri_ves.cpp, the last loop)
    // Measured with native LAMMPS (black box, tests/oracle/w107peri_ves_block): the minimum stretch of the run setup
    // is not kept, so no bond breaks in the first timestep and bonds break from the second one (the same rule as the
    // core peri/lps, historyUpdate false at the run setup).
    if (pc.historyUpdate) {
      for (let i = 0; i < nl; i++) {
        this.s0[i] = s0_new[i];
        this.smin[i] = smin_new[i] === DBL_MAX ? -DBL_MAX : smin_new[i];
      }
    }
  }

  /** Per owned atom: the fraction of its family volume whose bonds are broken (compute damage/atom). */
  damage(vfrac: Float64Array, nlocal: number): Float64Array {
    const out = new Float64Array(nlocal);
    if (!this.built) return out;
    for (let i = 0; i < nlocal; i++) {
      let intact = 0;
      for (let k = this.pStart[i]; k < this.pStart[i + 1]; k++) if (!this.pBr[k]) intact += vfrac[this.pJ[k]];
      out[i] = this.vinter[i] !== 0 ? 1.0 - intact / this.vinter[i] : 0;
    }
    return out;
  }

  /** The dilatation of the last evaluation (compute dilatation/atom), per owned atom. */
  dilatation(nlocal: number): Float64Array {
    const out = new Float64Array(nlocal);
    if (this.built) out.set(this.theta.subarray(0, Math.min(nlocal, this.theta.length)));
    return out;
  }

  /** Whether the bond family has been built (after the first evaluation). */
  isBuilt(): boolean { return this.built; }
}
