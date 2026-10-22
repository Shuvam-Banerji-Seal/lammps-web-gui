/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EXTRA-PAIR/pair_e3b.cpp and pair_e3b.h
 * (PairE3B: pair_style e3b, the explicit three-body water potential of Kumar and Skinner, with the parameter presets of
 * Tainter et al.). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The potential is used with a TIP4P water style in pair_style hybrid/overlay (docs.lammps.org/pair_e3b.html:
 * "pair_style hybrid/overlay e3b Otype"; the pair_coeff form is "pair_coeff * * keyword values" or "pair_coeff * * preset
 * 2011|2015"). It sums a two-body O-O exponential and, for each O-O pair within rc3 plus a dangling-H allowance, the
 * three-body terms A, B and C built from the O-H contacts of the pair, switched between rs and rc3. The H atoms are the
 * atoms with tag O+1 and O+2 (the molecule order of the data file); their periodic copy is the one nearest the partner O.
 * Sums are kept by atom tag, as the source does. Parity: oracle case tests/oracle/w130pr_e3b_*.in.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { inumeric, parseNum } from '../../engine/force/util';
import { exp as libmExp } from '../libm/exp';

/** pair_e3b.cpp: NOT_SET marks a keyword that was not given. */
const NOT_SET = 1.024e300;
/** pair_e3b.cpp: BOND_DELTA, the buffer for OH bonds that are not perfectly rigid. */
const BOND_DELTA = 1.01;
/** pair_e3b.cpp: NUMH hydrogens per water, NUMO oxygens per O-O pair. */
const NUMH = 2;
const NUMO = 2;

export class PairE3B extends Pair {
  readonly name: string = 'e3b';
  virialFdotr = true;
  /** pvector (pair_e3b.cpp): e2, ea, eb, ec energies of the last evaluation (compute pair e3b). */
  private readonly pvector = new Float64Array(4);
  private typeO = 0;
  /** The Otype word of pair_style (native typeO_str), read at pair_coeff. */
  private typeOStr = '';
  private ea = NOT_SET;
  private eb = NOT_SET;
  private ec = NOT_SET;
  private k3 = NOT_SET;
  private e2 = NOT_SET;
  private k2 = NOT_SET;
  private rs = 0;
  private rc3 = 0;
  private rc2 = 0;
  private cutmax = 0;
  private rc2sq = 0;
  private rc3sq = 0;
  private rc3deltaSq = 0;
  private scDenom = 0;
  private scNum = 0;
  /** pair_coeff sets the O-O type pair only (setflag[typeO][typeO]). */
  private setflag = new Uint8Array(0);
  private nt = 0;
  private natoms = -1;

  settings(args: string[]): void {
    // pair_e3b.cpp settings(): if (narg != 1) error->all(FLERR, "Illegal pair_style command");
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.typeOStr = args[0];
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
    this.setflag = new Uint8Array(this.nt * this.nt);
  }

  /** pair_coeff * * keyword value ...: Ea Eb Ec K3 Rs Rc3 Rc2 bondL E2 K2 neigh, or preset 2011|2015. */
  coeff(args: string[], ctx: StyleContext): void {
    // pair_e3b.cpp coeff(): "There must be at least one keyword given to pair_coeff"
    if (args.length < 4) throw new StyleError('There must be at least one keyword given to pair_coeff');
    if (this.setflag.length === 0) throw new StyleError('pair_coeff e3b before the box exists');
    // the Otype word of pair_style is read here, a type label or a number within 1-ntypes (utils::expand_type_int)
    if (this.typeOStr) {
      const t = inumeric(ctx.atomTypeWord ? ctx.atomTypeWord(this.typeOStr) : this.typeOStr);
      if (t <= 0 || t > this.ntypes) throw new StyleError(`Atom type ${t} is out of bounds (1-${this.ntypes})`);
      this.typeO = t;
    }
    // clear setflag, then set the O-O pair only
    this.setflag.fill(0);
    this.setflag[this.typeO * this.nt + this.typeO] = 1;
    this.ea = this.eb = this.ec = this.k3 = this.k2 = this.e2 = NOT_SET;
    this.rs = this.rc3 = this.rc2 = 0;
    let bondL = 0;
    let presetSeen = false;
    const need = (k: string, i: number): string => {
      if (i >= args.length) throw new StyleError(`Too few arguments to '${k}' keyword.`);
      return args[i];
    };
    let iarg = 2;
    while (iarg < args.length) {
      const key = args[iarg++];
      switch (key) {
        case 'Ea': this.ea = parseNum(need(key, iarg++), key); break;
        case 'Eb': this.eb = parseNum(need(key, iarg++), key); break;
        case 'Ec': this.ec = parseNum(need(key, iarg++), key); break;
        case 'K3': this.k3 = parseNum(need(key, iarg++), key); break;
        case 'Rs': this.rs = parseNum(need(key, iarg++), key); break;
        case 'Rc3': this.rc3 = parseNum(need(key, iarg++), key); break;
        case 'Rc2': this.rc2 = parseNum(need(key, iarg++), key); break;
        case 'bondL': bondL = parseNum(need(key, iarg++), key); break;
        case 'E2': this.e2 = parseNum(need(key, iarg++), key); break;
        case 'K2': this.k2 = parseNum(need(key, iarg++), key); break;
        case 'neigh': parseNum(need(key, iarg++), key); break; // neighbor-storage size of the source; no effect here
        case 'preset': {
          const flag = parseNum(need(key, iarg++), key);
          if (presetSeen) throw new StyleError('Cannot request two different sets of preset parameters');
          presetSeen = true;
          const p = this.presetParams(flag, ctx);
          this.ea = p.ea; this.eb = p.eb; this.ec = p.ec; this.k3 = p.k3; this.e2 = p.e2; this.k2 = p.k2;
          this.rs = p.rs; this.rc3 = p.rc3; this.rc2 = p.rc2; bondL = p.bondL;
          break;
        }
        default:
          throw new StyleError(`Keyword ${key} is unknown`);
      }
    }
    this.checkInputs(bondL);
    // cutmax for neighbor listing
    this.cutmax = Math.max(this.rc2, this.rc3);
    this.rc2sq = this.rc2 * this.rc2;
    this.rc3sq = this.rc3 * this.rc3;
    this.rc3deltaSq = (this.rc3 + bondL) * (this.rc3 + bondL);
    const tmpfact = 1.0 / (this.rc3 - this.rs);
    this.scDenom = tmpfact * tmpfact * tmpfact;
    this.scNum = this.rc3 - 3 * this.rs;
  }

  /**
   * pair_e3b.cpp presetParam(): the Tainter et al. parameters in kJ/mol and Angstrom, converted to the units style.
   * preset 2008 is refused as the source does (it needs distinct k3 values).
   */
  private presetParams(flag: number, ctx: StyleContext): { ea: number; eb: number; ec: number; k3: number; e2: number; k2: number; rs: number; rc3: number; rc2: number; bondL: number } {
    const unit = ctx.s?.units.style;
    let econv: number, lconv: number;
    if (unit === 'real') { econv = 1.0 / 4.184; lconv = 1.0; }
    else if (unit === 'metal') { econv = 0.103653271; lconv = 1.0; }
    else if (unit === 'si') { econv = 1.660578e-21; lconv = 1e-10; }
    else if (unit === 'cgs') { econv = 1.660578e-14; lconv = 1e-8; }
    else throw new StyleError(`Pre-defined E3B parameters have not been set for ${unit ?? 'unknown'} units`);
    let p: { ea: number; eb: number; ec: number; k3: number; e2: number; k2: number; rs: number; rc3: number; rc2: number; bondL: number };
    if (flag === 2008) {
      throw new StyleError("'preset 2008' is not yet supported, because this would require distinct k3 coefficients, use 'preset 2011' or 'preset 2015'");
    } else if (flag === 2011) {
      p = { ea: 1745.7, eb: -4565.0, ec: 7606.8, k3: 1.907, e2: 2.349e6, k2: 4.872, rs: 5.0, rc3: 5.2, rc2: 5.2, bondL: 0.9572 };
    } else if (flag === 2015) {
      p = { ea: 150.0, eb: -1005.0, ec: 1880.0, k3: 1.907, e2: 0.453e6, k2: 4.872, rs: 5.0, rc3: 5.2, rc2: 5.2, bondL: 0.9572 };
    } else {
      throw new StyleError('Unknown argument: preset only takes 2011 or 2015 as arguments');
    }
    // "here parameters are defined in kJ/mol and A; they will be converted to the lammps units after"
    return {
      ea: p.ea * econv, eb: p.eb * econv, ec: p.ec * econv, e2: p.e2 * econv,
      k3: p.k3 / lconv, k2: p.k2 / lconv, rs: p.rs * lconv, rc2: p.rc2 * lconv, rc3: p.rc3 * lconv,
      bondL: p.bondL * lconv * BOND_DELTA,
    };
  }

  /** pair_e3b.cpp checkInputs(): every keyword must be given and in range. */
  private checkInputs(bondL: number): void {
    if (this.rc2 === 0) throw new StyleError('rc2 keyword missing');
    if (this.rs === 0) throw new StyleError('Rs keyword missing');
    if (this.rc3 === 0) throw new StyleError('Rc3 keyword missing');
    if (bondL === 0) throw new StyleError('bondL keyword missing');
    if (this.ea === NOT_SET) throw new StyleError('Ea keyword missing');
    if (this.eb === NOT_SET) throw new StyleError('Eb keyword missing');
    if (this.ec === NOT_SET) throw new StyleError('Ec keyword missing');
    if (this.k3 === NOT_SET) throw new StyleError('K3 keyword missing');
    if (this.e2 === NOT_SET) throw new StyleError('E2 keyword missing');
    if (this.k2 === NOT_SET) throw new StyleError('K2 keyword missing');
    if (this.k2 < 0 || this.k3 < 0) throw new StyleError('exponential decay is negative');
    if (bondL < 0) throw new StyleError('OH bond length is negative');
    if (this.rc2 < 0 || this.rc3 < 0 || this.rs < 0) throw new StyleError('potential cutoff is negative');
    if (this.rs > this.rc3) throw new StyleError('potential switching distance is larger than cutoff');
  }

  /** pair_e3b.cpp init_style(): atom IDs and newton on. The O-O pair is the only one this style sets. */
  initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError('Pair style E3B requires newton pair on');
    if (this.typeO < 1 || this.typeO > this.ntypes) throw new StyleError('Invalid Otype: out of bounds');
    this.natoms = ctx.s ? ctx.s.n : -1;
  }

  /** The O-O pairs only (the hybrid assigns the sub-style to them, not to every type pair). */
  setsPair(i: number, j: number): boolean {
    return this.setflag[i * this.nt + j] === 1;
  }

  initOne(i: number, j: number): number {
    // pair_e3b.cpp init_one(): "All pair coeffs are not set"
    if (this.setflag[i * this.nt + j] === 0) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    return this.cutmax;
  }

  compute(pc: PairCompute): void {
    const list = pc.half;
    if (!list) throw new StyleError('pair e3b needs a half neighbor list');
    if (this.natoms >= 0 && pc.s.n !== this.natoms) throw new StyleError('pair E3B requires a fixed number of atoms');
    const { x, f, type } = pc;
    const nbr = list.neighbors;
    const nb = pc.nb;
    const owner = nb.owner;
    const gs = nb.ghostStart;
    const id = pc.s.id;
    const nlocal = pc.nlocal;
    const typeO = this.typeO;
    // maxID: the largest atom ID (the sums are indexed by ID)
    let maxID = 0;
    for (let i = 0; i < nlocal; i++) if (id[i] > maxID) maxID = id[i];
    const idxOfTag = new Int32Array(maxID + 2).fill(-1);
    for (let i = 0; i < nlocal; i++) idxOfTag[id[i]] = i;
    const tagOf = (k: number): number => id[owner[k]];
    const sumExp = new Float64Array(maxID + 2);
    const dist2 = (a: number, b: number): number => {
      const dx = x[3 * a] - x[3 * b], dy = x[3 * a + 1] - x[3 * b + 1], dz = x[3 * a + 2] - x[3 * b + 2];
      return dx * dx + dy * dy + dz * dz;
    };
    // domain->closest_image(i, h): the copy of owned atom h (or h itself) nearest to atom i
    const closest = (i: number, h0: number): number => {
      let best = h0, bestR = dist2(i, h0);
      for (let g = gs[h0]; g < gs[h0 + 1]; g++) {
        const r = dist2(i, g);
        if (r < bestR) { bestR = r; best = g; }
      }
      return best;
    };
    // per O-O pair (npair entries): the two O indices, the H indices [kk][hh], exps, fpair3 and del3
    const pairO: number[] = [];
    const pairH: number[] = [];
    const exps: number[] = [];
    const fpair3: number[] = [];
    const del3: number[] = [];
    const sc = this.scNum, scDenom = this.scDenom, rs = this.rs, rc3 = this.rc3, rc3sq = this.rc3sq, k3 = this.k3;
    const pv = this.pvector;
    pv.fill(0);
    let evdwlTotal = 0;
    const tally = (i: number, j: number, e: number, fpair: number, dx: number, dy: number, dz: number): void => {
      evdwlTotal += e;
      tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
    };
    let npair = 0;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      if (type[i] !== typeO) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let fix = 0, fiy = 0, fiz = 0;
      const k0: number = list.firstneigh[i], k1: number = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbr[k] & NEIGHMASK;
        if (type[j] !== typeO) continue;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        // two-body O-O term, not shifted: k2 = 4.87/A so e^(-k r) is 1e-11 at the cutoff
        if (rsq < this.rc2sq) {
          const tmpr = Math.sqrt(rsq);
          const tmpexp = this.e2 * libmExp(-this.k2 * tmpr);
          const fpair = this.k2 * tmpexp / tmpr;
          const fxtmp = delx * fpair, fytmp = dely * fpair, fztmp = delz * fpair;
          fix += fxtmp; fiy += fytmp; fiz += fztmp;
          f[3 * j] -= fxtmp; f[3 * j + 1] -= fytmp; f[3 * j + 2] -= fztmp;
          tally(i, j, tmpexp, fpair, delx, dely, delz);
          pv[0] += tmpexp;
        }
        // O-O pairs within rc3 plus the allowance for the OH bond length (dangling hydrogens)
        const rc3deltaSq = this.rc3deltaSq;
        if (rsq < rc3deltaSq) {
          const po = [i, j];
          const ph = [0, 0, 0, 0];
          const ex = [0, 0, 0, 0];
          const f3 = [0, 0, 0, 0];
          const d3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
          let addedH = false;
          for (let kk = 0; kk < NUMO; kk++) {
            const kAtom = po[kk];
            const otherO = po[(kk + 1) % 2];
            for (let hh = 0; hh < NUMH; hh++) {
              const hOwned = idxOfTag[tagOf(otherO) + hh + 1];
              // "hydrogen atom missing" (native leaves it to bond/shake to catch; the engine stops here)
              if (hOwned < 0) throw new StyleError(`pair e3b: hydrogen atom ${tagOf(otherO) + hh + 1} of O ${tagOf(otherO)} is missing`);
              const h = closest(otherO, hOwned);
              const slot = kk * NUMH + hh;
              ph[slot] = h;
              const delxh = x[3 * kAtom] - x[3 * h];
              const delyh = x[3 * kAtom + 1] - x[3 * h + 1];
              const delzh = x[3 * kAtom + 2] - x[3 * h + 2];
              const rsqh = delxh * delxh + delyh * delyh + delzh * delzh;
              if (rsqh < rc3sq) {
                const tmpr = Math.sqrt(rsqh);
                let tmpexp = libmExp(-k3 * tmpr);
                let scEng: number, scDer: number;
                if (tmpr > rs) {
                  const scFact1 = rc3 - tmpr;
                  const scFact2 = sc + 2 * tmpr;
                  scEng = scFact1 * scFact1 * scFact2 * scDenom;
                  scDer = k3 * scEng - 6 * scFact1 * (rs - tmpr) * scDenom;
                } else {
                  scDer = k3;
                  scEng = 1.0;
                }
                // fpair3 is kept apart from del3 for the virial
                f3[slot] = scDer * tmpexp / tmpr;
                tmpexp *= scEng;
                ex[slot] = tmpexp;
                d3[3 * slot] = delxh; d3[3 * slot + 1] = delyh; d3[3 * slot + 2] = delzh;
                // sums by atom ID (tags start at 1)
                sumExp[tagOf(kAtom)] += tmpexp;
                sumExp[tagOf(h)] += tmpexp;
                addedH = true;
              } else {
                ex[slot] = 0;
                f3[slot] = 0;
              }
            }
          }
          if (addedH) {
            pairO.push(po[0], po[1]);
            pairH.push(...ph);
            exps.push(...ex);
            fpair3.push(...f3);
            del3.push(...d3);
            npair++;
          }
        }
      }
      // the two-body force on i
      f[3 * i] += fix; f[3 * i + 1] += fiy; f[3 * i + 2] += fiz;
    }
    // three-body forces from the summed exponentials of each O-O pair
    for (let p = 0; p < npair; p++) {
      for (let kk = 0; kk < NUMO; kk++) {
        const i = pairO[2 * p + kk];
        const otherO = (kk + 1) % 2;
        const oth = pairO[2 * p + otherO];
        const hOth0 = pairH[4 * p + otherO * NUMH], hOth1 = pairH[4 * p + otherO * NUMH + 1];
        const partB = this.eb * (sumExp[tagOf(oth)] + sumExp[tagOf(hOth0)] + sumExp[tagOf(hOth1)]
          - 2 * (exps[4 * p + otherO * NUMH] + exps[4 * p + otherO * NUMH + 1]));
        const partC = this.ec * (sumExp[tagOf(i)] - exps[4 * p + kk * NUMH] - exps[4 * p + kk * NUMH + 1]);
        for (let hh = 0; hh < NUMH; hh++) {
          const slot = kk * NUMH + hh;
          const j = pairH[4 * p + slot];
          // type A
          const otherH = (hh + 1) % 2;
          const j2 = pairH[4 * p + kk * NUMH + otherH];
          const partA = this.ea * (sumExp[tagOf(j2)] - exps[4 * p + kk * NUMH + otherH]);
          const fA = partA * fpair3[4 * p + slot];
          const ax = fA * del3[12 * p + 3 * slot], ay = fA * del3[12 * p + 3 * slot + 1], az = fA * del3[12 * p + 3 * slot + 2];
          f[3 * i] += ax; f[3 * i + 1] += ay; f[3 * i + 2] += az;
          f[3 * j] -= ax; f[3 * j + 1] -= ay; f[3 * j + 2] -= az;
          const eA = partA * exps[4 * p + slot] * 0.5;
          tally(i, j, eA, fA, del3[12 * p + 3 * slot], del3[12 * p + 3 * slot + 1], del3[12 * p + 3 * slot + 2]);
          pv[1] += eA;
          // type B
          const fB = partB * fpair3[4 * p + slot];
          const bx = fB * del3[12 * p + 3 * slot], by = fB * del3[12 * p + 3 * slot + 1], bz = fB * del3[12 * p + 3 * slot + 2];
          f[3 * i] += bx; f[3 * i + 1] += by; f[3 * i + 2] += bz;
          f[3 * j] -= bx; f[3 * j + 1] -= by; f[3 * j + 2] -= bz;
          const eB = partB * exps[4 * p + slot] * 0.5;
          tally(i, j, eB, fB, del3[12 * p + 3 * slot], del3[12 * p + 3 * slot + 1], del3[12 * p + 3 * slot + 2]);
          pv[2] += eB;
          // type C
          const fC = partC * fpair3[4 * p + slot];
          const cx = fC * del3[12 * p + 3 * slot], cy = fC * del3[12 * p + 3 * slot + 1], cz = fC * del3[12 * p + 3 * slot + 2];
          f[3 * i] += cx; f[3 * i + 1] += cy; f[3 * i + 2] += cz;
          f[3 * j] -= cx; f[3 * j + 1] -= cy; f[3 * j + 2] -= cz;
          const eC = partC * exps[4 * p + slot] * 0.5;
          tally(i, j, eC, fC, del3[12 * p + 3 * slot], del3[12 * p + 3 * slot + 1], del3[12 * p + 3 * slot + 2]);
          pv[3] += eC;
        }
      }
    }
    pc.acc.evdwl += evdwlTotal;
  }

  /** pair_e3b.cpp: pvector (e2, ea, eb, ec) for compute pair e3b. */
  extract(name: string): unknown {
    if (name === 'pvector') return this.pvector;
    return undefined;
  }
}
