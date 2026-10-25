/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/APIP/pair_eam_apip.cpp
 * (pair_style eam/apip: settings, coeff, read_file and file2array of DYNAMO funcfl files, init_one, setup and the
 * lambda-weighted compute) and src/APIP/pair_eam_fs_apip.cpp (pair_style eam/fs/apip: coeff and file2array of the
 * Finnis-Sinclair file), at commit b332f28. Authors named in the sources: Stephen Foiles and Murray Daw (SNL, EAM) and
 * David Immel (FZ Juelich) for APIP, with the LAMMPS Sandia copyright notice (GNU General Public License). The FS reader,
 * array2spline and interpolate are the ports in ../manyeam/potential.ts (identical in pair_eam.cpp). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_eam_apip.html. The EAM energy of atom i is weighted with lambda_i (the fast potential).
 * The compute walks a full neighbour list and handles each pair of owned atoms once (from the atom with the lower
 * index); a ghost neighbour contributes only the terms of atom i (the owner of the ghost adds the others). An atom
 * whose lambda and whose neighbours' lambdas are all zero is skipped (lambda_required NO_SIMPLE).
 */
import { Pair, potentialFile, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { buildTables, interpolate, openPotential, parseInteger, parseReal, readFs, type EamFile, type EamTables, type Spline } from '../manyeam/potential';
import { NO_SIMPLE, SIMPLE, apipFields, apipModes, checkApipAtomStyle, reverseAdd3 } from './common';

/** One DYNAMO funcfl file, arrays 1-based. */
interface Funcfl {
  file: string;
  mass: number;
  nrho: number;
  drho: number;
  nr: number;
  dr: number;
  cut: number;
  frho: Float64Array;
  rhor: Float64Array;
  zr: Float64Array;
}

/** PairEAMAPIP::read_file: comment line, "ielem mass", "nrho drho nr dr cut", then F(rho), Z(r) and rho(r). */
const readFuncfl = (text: string, filename: string, ctx: StyleContext): Funcfl => {
  const t = openPotential(text, 'eam', filename, ctx);
  t.skipLine();
  let v = t.nextValues(2);
  parseInteger(v[0], 'eam', filename);
  const mass = parseReal(v[1], 'eam', filename);
  v = t.nextValues(5);
  const nrho = parseInteger(v[0], 'eam', filename);
  const drho = parseReal(v[1], 'eam', filename);
  const nr = parseInteger(v[2], 'eam', filename);
  const dr = parseReal(v[3], 'eam', filename);
  const cut = parseReal(v[4], 'eam', filename);
  if (nrho <= 0 || nr <= 0 || !(dr > 0)) throw new StyleError('Invalid EAM potential file');
  const frho = new Float64Array(nrho + 1), zr = new Float64Array(nr + 1), rhor = new Float64Array(nr + 1);
  t.nextDvector(nrho, frho, 1);
  t.nextDvector(nr, zr, 1);
  t.nextDvector(nr, rhor, 1);
  return { file: filename, mass, nrho, drho, nr, dr, cut, frho, rhor, zr };
};

/** The four-point Lagrange interpolation of file2array: value of table a (spacing d, n points) at r. */
const lagrange = (a: Float64Array, n: number, d: number, r: number): number => {
  const sixth = 1.0 / 6.0;
  let p = r / d + 1.0;
  let k = Math.trunc(p);
  k = Math.min(k, n - 2);
  k = Math.max(k, 2);
  p -= k;
  p = Math.min(p, 2.0);
  const cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
  const cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
  const cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
  const cof4 = sixth * p * (p * p - 1.0);
  return cof1 * a[k - 1] + cof2 * a[k] + cof3 * a[k + 1] + cof4 * a[k + 2];
};

/** PairEAMAPIP::file2array + array2spline for funcfl files: every file on one grid and cutoff. */
const funcflTables = (funcfl: Funcfl[], map: Int32Array, ntypes: number): EamTables & { rhomax: number } => {
  let dr = 0, drho = 0, rmax = 0, rhomax = 0;
  for (let i = 0; i < funcfl.length; i++) {
    let active = false;
    for (let j = 1; j <= ntypes; j++) if (map[j] === i) active = true;
    if (!active) continue;
    const file = funcfl[i];
    dr = Math.max(dr, file.dr);
    drho = Math.max(drho, file.drho);
    rmax = Math.max(rmax, (file.nr - 1) * file.dr);
    rhomax = Math.max(rhomax, (file.nrho - 1) * file.drho);
  }
  // std::lround
  const lround = (v: number): number => (v < 0 ? -Math.round(-v) : Math.round(v));
  const nr = lround(rmax / dr);
  const nrho = lround(rhomax / drho);
  const nfuncfl = funcfl.length;
  const nfrho = nfuncfl + 1;
  const frho: Float64Array[] = [];
  for (const file of funcfl) {
    const a = new Float64Array(nrho + 1);
    for (let m = 1; m <= nrho; m++) a[m] = lagrange(file.frho, file.nrho, file.drho, (m - 1) * drho);
    frho.push(a);
  }
  frho.push(new Float64Array(nrho + 1));
  const nt = ntypes + 1;
  const type2frho = new Int32Array(nt);
  for (let i = 1; i <= ntypes; i++) type2frho[i] = map[i] >= 0 ? map[i] : nfrho - 1;
  const rhor: Float64Array[] = [];
  for (const file of funcfl) {
    const a = new Float64Array(nr + 1);
    for (let m = 1; m <= nr; m++) a[m] = lagrange(file.rhor, file.nr, file.dr, (m - 1) * dr);
    rhor.push(a);
  }
  const type2rhor = new Int32Array(nt * nt);
  for (let i = 1; i <= ntypes; i++) for (let j = 1; j <= ntypes; j++) type2rhor[i * nt + j] = map[i];
  const z2r: Float64Array[] = [];
  for (let i = 0; i < nfuncfl; i++) {
    for (let j = 0; j <= i; j++) {
      const a = new Float64Array(nr + 1);
      for (let m = 1; m <= nr; m++) {
        const r = (m - 1) * dr;
        const zri = lagrange(funcfl[i].zr, funcfl[i].nr, funcfl[i].dr, r);
        const zrj = lagrange(funcfl[j].zr, funcfl[j].nr, funcfl[j].dr, r);
        a[m] = 27.2 * 0.529 * zri * zrj;
      }
      z2r.push(a);
    }
  }
  const type2z2r = new Int32Array(nt * nt);
  for (let i = 1; i <= ntypes; i++) {
    for (let j = 1; j <= ntypes; j++) {
      let irow = map[i], icol = map[j];
      if (irow === -1 || icol === -1) continue;
      if (irow < icol) {
        irow = map[j];
        icol = map[i];
      }
      let n = 0;
      for (let m = 0; m < irow; m++) n += m + 1;
      type2z2r[i * nt + j] = n + icol;
    }
  }
  return {
    nrho, nr, drho, dr, rdr: 1.0 / dr, rdrho: 1.0 / drho,
    frho: frho.map((a) => interpolate(nrho, drho, a)),
    rhor: rhor.map((a) => interpolate(nr, dr, a)),
    z2r: z2r.map((a) => interpolate(nr, dr, a)),
    u2r: [], w2r: [],
    type2frho, type2rhor, type2z2r, type2u2r: new Int32Array(0), type2w2r: new Int32Array(0),
    rhomax,
  };
};

export class PairEAMAPIP extends Pair {
  readonly name: string = 'eam/apip';
  manybody = true;
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;
  protected map = new Int32Array(0);
  protected funcfl: Funcfl[] = [];
  protected fsFile: EamFile | null = null;
  protected tabs: (EamTables & { rhomax: number }) | null = null;
  protected cutmax = 0;
  protected readonly coeffed = new Set<number>();
  private cutforcesq = 0;
  private lambdaThermostat = false;
  private lambdaLa = false;
  private readonly natOrder = new NativeFullOrder();
  /** The extract name of the time per atom (eam/apip and eam/fs/apip answer the same name). */
  protected readonly timeName: string = 'eam/apip:time_per_atom';

  /** Wall time and counted atoms since the last time_per_atom extraction, and the total wall time (seconds). */
  private timeWallAccumulated = 0;
  private nAccumulated = 0;
  private wallTotal = 0;

  /** pair->extract("<style>:time_per_atom") through calculate_time_per_atom: the mean, then a reset (-1 without atoms). */
  apipTimePerAtom(name: string): number | null {
    if (name !== this.timeName) return null;
    const t = this.nAccumulated > 0 ? this.timeWallAccumulated / this.nAccumulated : -1;
    this.timeWallAccumulated = 0;
    this.nAccumulated = 0;
    return t;
  }

  apipWallTotal(): number {
    return this.wallTotal;
  }

  private addTiming(t0: number, n: number): void {
    const dt = (performance.now() - t0) / 1000;
    this.timeWallAccumulated += dt;
    this.wallTotal += dt;
    this.nAccumulated += n;
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.map = new Int32Array(ntypes + 1).fill(-1);
    this.funcfl = [];
    this.fsFile = null;
    this.tabs = null;
    this.coeffed.clear();
  }

  override settings(args: string[]): void {
    if (args.length > 0) throw new StyleError('Illegal pair_style command');
  }

  /** PairEAMAPIP::coeff: "pair_coeff I J file" for I = J, one funcfl file per type; the type mass comes from the file. */
  override coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for pair coefficients');
    const [ilo, ihi] = this.bounds(args[0]);
    const [jlo, jhi] = this.bounds(args[1]);
    let ifuncfl = this.funcfl.findIndex((f) => f.file === args[2]);
    if (ifuncfl < 0) {
      // PairEAMAPIP::read_file: PotentialFileReader(lmp, filename, "eam")
      this.funcfl.push(readFuncfl(ctx.readFile(args[2], potentialFile('eam')), args[2], ctx));
      ifuncfl = this.funcfl.length - 1;
    }
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        if (i === j) {
          this.coeffed.add(i);
          this.map[i] = ifuncfl;
          this.setMass(ctx, i, this.funcfl[ifuncfl].mass);
          count++;
        }
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** setflag[i][j]: the I,I types given a funcfl file (eam/apip), or mapped element pairs (eam/fs/apip). */
  override setsPair(i: number, j: number): boolean {
    if (this.fsFile) return this.map[i] >= 0 && this.map[j] >= 0;
    return i === j && this.coeffed.has(i);
  }

  /** atom->set_mass (atom.cpp): a value <= 0 is ignored with a warning. */
  protected setMass(ctx: StyleContext, type: number, value: number): void {
    const s = ctx.s;
    if (!s) return;
    if (s.rmass) throw new StyleError('Cannot set per-type atom mass for this atom style');
    if (!(value > 0)) {
      ctx.log(`WARNING: Ignoring invalid mass value ${value} for atom type ${type}`);
      return;
    }
    s.massByType[type] = value;
  }

  protected bounds(w: string): [number, number] {
    const n = this.ntypes;
    if (w === '*') return [1, n];
    const m = /^(\d*)\*(\d*)$/.exec(w);
    if (m) return [m[1] ? Number(m[1]) : 1, m[2] ? Number(m[2]) : n];
    if (!/^\d+$/.test(w) || Number(w) < 1 || Number(w) > n) throw new StyleError(`Numeric index ${w} is out of bounds (1-${n})`);
    return [Number(w), Number(w)];
  }

  protected buildTables(): EamTables & { rhomax: number } {
    return funcflTables(this.funcfl, this.map, this.ntypes);
  }

  override initStyle(ctx: StyleContext): void {
    checkApipAtomStyle(ctx, this.name);
    this.tabs = this.buildTables();
    const m = apipModes(ctx, this.name);
    this.lambdaThermostat = m.thermostat;
    this.lambdaLa = m.la;
  }

  /** init_one: the largest file cutoff (funcfl) or the file cutoff (FS). */
  override initOne(i: number, j: number): number {
    void i; void j;
    if (this.funcfl.length) {
      this.cutmax = 0;
      for (const f of this.funcfl) this.cutmax = Math.max(this.cutmax, f.cut);
    } else if (this.fsFile) {
      this.cutmax = this.fsFile.cut;
    }
    this.cutforcesq = this.cutmax * this.cutmax;
    return this.cutmax;
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    const T = this.tabs;
    if (!T) throw new StyleError('All pair coeffs are not set');
    const t0 = performance.now();
    let nNonComplex = 0;
    const ap = apipFields(pc);
    const lambda = ap.lambda, req = ap.lambdaRequired;
    const thermo = this.lambdaThermostat;
    const lambdaConst = thermo ? ap.lambdaConst! : null;
    const eSimple = thermo || this.lambdaLa ? ap.eFast : null;
    const fDyn = thermo ? new Float64Array(3 * pc.nall) : null;
    const fConst = thermo ? new Float64Array(3 * pc.nall) : null;
    const x = pc.x, f = pc.f, type = pc.type;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const nr = T.nr, nrho = T.nrho, rdr = T.rdr, rdrho = T.rdrho, rhomax = T.rhomax;
    const nbrs = this.natOrder.neighbors(pc.s, pc.nb, pc.geom, list);
    const ilo = list.ilo ?? 0, inum = list.inum;
    const rho = new Float64Array(nlocal);
    const fp = new Float64Array(nlocal);
    const cutforcesq = this.cutforcesq;
    const sp = (s: Spline, m: number, k: number): number => s[m * 7 + k];

    // rho = density at each owned atom; a pair of owned atoms is handled once (j < i is skipped)
    for (let i = ilo; i < inum; i++) {
      const xt = x[3 * i], yt = x[3 * i + 1], zt = x[3 * i + 2];
      const itype = type[i];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        if (j < i && i < nlocal) continue;
        const dx = xt - x[3 * j], dy = yt - x[3 * j + 1], dz = zt - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < cutforcesq) {
          const jtype = type[j];
          let p = Math.sqrt(rsq) * rdr + 1.0;
          let m = Math.trunc(p);
          m = Math.min(m, nr - 1);
          p -= m;
          p = Math.min(p, 1.0);
          let c = T.rhor[T.type2rhor[jtype * nt + itype]];
          rho[i] += ((sp(c, m, 3) * p + sp(c, m, 4)) * p + sp(c, m, 5)) * p + sp(c, m, 6);
          if (j < nlocal) {
            c = T.rhor[T.type2rhor[itype * nt + jtype]];
            rho[j] += ((sp(c, m, 3) * p + sp(c, m, 4)) * p + sp(c, m, 5)) * p + sp(c, m, 6);
          }
        }
      }
    }

    // fp = derivative of the embedding energy; the embedding energy weighted with lambda
    for (let i = ilo; i < inum; i++) {
      let p = rho[i] * rdrho + 1.0;
      let m = Math.trunc(p);
      m = Math.max(1, Math.min(m, nrho - 1));
      p -= m;
      p = Math.min(p, 1.0);
      const c = T.frho[T.type2frho[type[i]]];
      fp[i] = (sp(c, m, 0) * p + sp(c, m, 1)) * p + sp(c, m, 2);
      let phi = ((sp(c, m, 3) * p + sp(c, m, 4)) * p + sp(c, m, 5)) * p + sp(c, m, 6);
      if (rho[i] > rhomax) phi += fp[i] * (rho[i] - rhomax);
      // scale[itype][itype] = 1
      if (eSimple) eSimple[i] = phi;
      pc.acc.evdwl += lambda[i] * phi;
      if (pc.eatom) pc.eatom[i] += lambda[i] * phi;
    }

    // forces
    for (let i = ilo; i < inum; i++) {
      const xt = x[3 * i], yt = x[3 * i + 1], zt = x[3 * i + 2];
      const itype = type[i];
      const k0: number = list.firstneigh[i], k1: number = k0 + list.numneigh[i];
      if (req[i] & NO_SIMPLE) continue;
      if (!(req[i] & SIMPLE)) {
        if (lambda[i] !== 0 || (thermo && lambdaConst![i] !== 0)) {
          req[i] |= SIMPLE;
          for (let k = k0; k < k1; k++) {
            const j = nbrs[k] & NEIGHMASK;
            if (j < nlocal) req[j] |= SIMPLE;
          }
        } else {
          for (let k: number = k0; k < k1; k++) {
            const j: number = nbrs[k] & NEIGHMASK;
            if (lambda[j] !== 0 || (thermo && lambdaConst![j] !== 0)) {
              req[i] |= SIMPLE;
              if (j < nlocal) req[j] |= SIMPLE;
              break;
            }
          }
        }
        if (!(req[i] & SIMPLE)) {
          req[i] |= NO_SIMPLE;
          continue;
        }
      }
      nNonComplex++;
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        if (j < i && i < nlocal) continue;
        const dx = xt - x[3 * j], dy = yt - x[3 * j + 1], dz = zt - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (!(rsq < cutforcesq)) continue;
        const jtype = type[j];
        const r = Math.sqrt(rsq);
        let p = r * rdr + 1.0;
        let m = Math.trunc(p);
        m = Math.min(m, nr - 1);
        p -= m;
        p = Math.min(p, 1.0);
        let c = T.rhor[T.type2rhor[jtype * nt + itype]];
        const rhojp = (sp(c, m, 0) * p + sp(c, m, 1)) * p + sp(c, m, 2);
        c = T.z2r[T.type2z2r[itype * nt + jtype]];
        const z2p = (sp(c, m, 0) * p + sp(c, m, 1)) * p + sp(c, m, 2);
        const z2 = ((sp(c, m, 3) * p + sp(c, m, 4)) * p + sp(c, m, 5)) * p + sp(c, m, 6);
        const recip = 1.0 / r;
        const phi = z2 * recip;
        const phip = z2p * recip - phi * recip;
        let psip: number, rhoip = 0;
        if (j < nlocal) {
          c = T.rhor[T.type2rhor[itype * nt + jtype]];
          rhoip = (sp(c, m, 0) * p + sp(c, m, 1)) * p + sp(c, m, 2);
          psip = lambda[i] * fp[i] * rhojp + lambda[j] * fp[j] * rhoip + phip * (lambda[i] + lambda[j]) / 2;
        } else {
          psip = lambda[i] * fp[i] * rhojp + phip * 0.5 * lambda[i];
        }
        // scale[itype][jtype] = 1
        const fpair = -1.0 * psip * recip;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (fDyn && fConst && lambdaConst) {
          fDyn[3 * i] += dx * fpair; fDyn[3 * i + 1] += dy * fpair; fDyn[3 * i + 2] += dz * fpair;
          fDyn[3 * j] -= dx * fpair; fDyn[3 * j + 1] -= dy * fpair; fDyn[3 * j + 2] -= dz * fpair;
          const psipCl = j < nlocal
            ? lambdaConst[i] * fp[i] * rhojp + lambdaConst[j] * fp[j] * rhoip + phip * (lambdaConst[i] + lambdaConst[j]) / 2
            : lambdaConst[i] * fp[i] * rhojp + phip * 0.5 * lambdaConst[i];
          const fpairCl = -1.0 * psipCl * recip;
          fConst[3 * i] += dx * fpairCl; fConst[3 * i + 1] += dy * fpairCl; fConst[3 * i + 2] += dz * fpairCl;
          fConst[3 * j] -= dx * fpairCl; fConst[3 * j + 1] -= dy * fpairCl; fConst[3 * j + 2] -= dz * fpairCl;
        }
        const evdwl = 0.5 * 1.0 * phi;
        if (eSimple) {
          eSimple[i] += evdwl;
          if (j < nlocal) eSimple[j] += evdwl;
        }
        pc.acc.evdwl += lambda[i] * evdwl;
        if (pc.eatom) pc.eatom[i] += lambda[i] * evdwl;
        if (j < nlocal) {
          pc.acc.evdwl += lambda[j] * evdwl;
          if (pc.eatom) pc.eatom[j] += lambda[j] * evdwl;
        }
        if (pc.vatom) {
          // ev_tally with newton on: the pair virial, half to each atom
          const va = pc.vatom;
          const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
          const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
          for (const a of [i, j]) {
            va[6 * a] += 0.5 * v0; va[6 * a + 1] += 0.5 * v1; va[6 * a + 2] += 0.5 * v2;
            va[6 * a + 3] += 0.5 * v3; va[6 * a + 4] += 0.5 * v4; va[6 * a + 5] += 0.5 * v5;
          }
        }
      }
    }
    if (fDyn && fConst) {
      reverseAdd3(pc, fDyn, ap.fDynLambda!);
      reverseAdd3(pc, fConst, ap.fConstLambda!);
    }
    this.addTiming(t0, nNonComplex);
  }
}

/** pair_style eam/fs/apip: one Finnis-Sinclair file, "pair_coeff * * file elem1 ... elemN". */
export class PairEAMFSAPIP extends PairEAMAPIP {
  override readonly name: string = 'eam/fs/apip';

  /** PairEAMFSAPIP::coeff (native messages): the element of each type, NULL skips a type. */
  override coeff(args: string[], ctx: StyleContext): void {
    const ntypes = this.ntypes;
    if (args.length !== 3 + ntypes) throw new StyleError('Number of element to type mappings does not match number of atom types');
    // PairEAMFSAPIP::read_file: PotentialFileReader(lmp, filename, "eam/fs")
    const file = readFs(ctx.readFile(args[2], potentialFile('eam/fs')), args[2], ctx, false);
    this.fsFile = file;
    this.funcfl = [];
    const map = new Int32Array(ntypes + 1).fill(-1);
    for (let i = 3; i < args.length; i++) {
      if (args[i] === 'NULL') continue;
      const j = file.names.indexOf(args[i]);
      if (j < 0) throw new StyleError('No matching element in EAM potential file');
      map[i - 2] = j;
    }
    this.map = map;
    let count = 0;
    for (let i = 1; i <= ntypes; i++) {
      for (let j = i; j <= ntypes; j++) {
        if (map[i] >= 0 && map[j] >= 0) {
          if (i === j) this.setMass(ctx, i, file.mass[map[i]]);
          count++;
        }
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** file2array of the FS file: the file's own grid; rhomax = (nrho - 1) drho. */
  protected override buildTables(): EamTables & { rhomax: number } {
    const file = this.fsFile!;
    const t = buildTables(file, this.map, this.ntypes, true);
    return { ...t, rhomax: (file.nrho - 1) * file.drho };
  }
}
