/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/pair_exp6_rx.cpp and pair_exp6_rx.h
 * (PairExp6rx: pair_style exp6/rx, the exp-6 potential of a reacting DPD particle, mixed over the species of the particle
 * by the fix rx species concentrations). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author
 * (from pair_exp6_rx.cpp): James Larentzos (U.S. Army Research Laboratory).
 *
 * docs.lammps.org/pair_exp6_rx.html: "pair_style exp6/rx cutoff [fractional|molecular]" and "pair_coeff I J file site1
 * site2 scaling [...] [cutoff]". The particle's potential is the mixture of its species' exp-6 potentials: the
 * concentrations of the species (fix rx) weight the Lorentz-Berthelot mixed pair potentials of the two sites ("1fluid" is
 * the one-fluid approximation: the mixed rm, epsilon and alpha of the whole particle). The force uses the old
 * concentrations (the species at the start of the step) and the energy change of the reactions is accumulated as
 * uCG (old) and uCGnew (new), which fix eos/table/rx turns into the internal chemical energy uChem.
 *
 * Per-atom mixing (native getMixingWeights) is computed for each owned atom; a ghost atom uses its owner's values, which
 * is what native's forward communication gives it. The 'exp(value)' guard (native DBL_MIN_EXP) and the 10*DBL_EPSILON
 * tolerance are as native's.
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { DPD_UCG, DPD_UCGNEW } from '../../engine/atoms';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { tallyXYZ } from '../dpd/pair_dpd';
import { FixRX, MY_EPSILON, powint } from './fix_rx';
import { TextLines, Tokens } from './rx_table_reader';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

/** Exponent of the repulsive wall (pair_exp6_rx.cpp compute: nRep = 12); the wall starts at shift * rm * f(alpha). */
const N_REP = 12;
const SHIFT = 1.05;
/** DBL_MIN_EXP: exp of a value below it is taken as zero (pair_exp6_rx.cpp expValue). */
const DBL_MIN_EXP = -1021;

const expValue = (value: number): number => (value < DBL_MIN_EXP ? 0.0 : libmExp(value));

/** pair_exp6_rx.cpp func_rin: the inner point of the repulsive wall as a function of alpha. */
const funcRin = (alpha: number): number => expValue(3.7682065 + -1.4308614 * Math.sqrt(alpha));

/** An exp-6 parameter set of one species (pair_exp6_rx.cpp Param). */
interface Exp6Param {
  epsilon: number;
  rm: number;
  alpha: number;
  ispecies: number;
}

/** The per-atom mixing of one atom: 8 values for the current and 8 for the old concentrations (see MIX_*). */
const MIX = 16;
const EPS1 = 0, ALPHA1 = 1, RM1 = 2, WT1 = 3, EPS2 = 4, ALPHA2 = 5, RM2 = 6, WT2 = 7;
const EPS1O = 8, ALPHA1O = 9, RM1O = 10, WT1O = 11, EPS2O = 12, ALPHA2O = 13, RM2O = 14, WT2O = 15;

/** The energy (out[0]) and the radial force factor (out[1]) of one exp-6 pair (pair_exp6_rx.cpp compute, both sites). */
const exp6Pair = (r: number, rCut: number, rCutInv: number, rCut6inv: number, r6inv: number,
  eps: number, alpha: number, rm: number, out: Float64Array): void => {
  const rminv = 1.0 / rm;
  const buck1 = eps / (alpha - 6.0);
  const buck2 = 6.0 * alpha;
  const rexp = expValue(alpha * (1.0 - r * rminv));
  const rm2ij = rm * rm;
  const rm6ij = rm2ij * rm2ij * rm2ij;
  // the shifted potential: zero energy and force at the cutoff
  const rCutExp = expValue(alpha * (1.0 - rCut * rminv));
  const urc = buck1 * (6.0 * rCutExp - alpha * rm6ij * rCut6inv);
  const durc = -buck1 * buck2 * (rCutExp * rminv - rCutInv * rm6ij * rCut6inv);
  const rin1 = SHIFT * rm * funcRin(alpha);
  if (r < rin1) {
    const rin6 = rin1 * rin1 * rin1 * rin1 * rin1 * rin1;
    const rin6inv = 1.0 / rin6;
    const rin1exp = expValue(alpha * (1.0 - rin1 * rminv));
    const uin1 = buck1 * (6.0 * rin1exp - alpha * rm6ij * rin6inv) - urc - durc * (rin1 - rCut);
    const win1 = buck1 * buck2 * (rin1 * rin1exp * rminv - rm6ij * rin6inv) + rin1 * durc;
    const aRep = win1 * powint(rin1, N_REP) / N_REP;
    const uin1rep = aRep / powint(rin1, N_REP);
    out[1] = N_REP * aRep / powint(r, N_REP);
    out[0] = uin1 - uin1rep + aRep / powint(r, N_REP);
  } else {
    out[1] = buck1 * buck2 * (r * rexp * rminv - rm6ij * r6inv) + r * durc;
    out[0] = buck1 * (6.0 * rexp - alpha * rm6ij * r6inv) - urc - durc * (r - rCut);
  }
};

/** A keyword-argument number (utils::numeric). */
const numArg = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating-point parameter instead of '${w ?? ''}'`);
  return v;
};

export class PairExp6rx extends Pair {
  readonly name = 'exp6/rx';
  /** pair_exp6_rx.cpp compute: "if (vflag_fdotr) virial_fdotr_compute();" (the global virial by the fdotr sum). */
  virialFdotr = true;
  private cutGlobal = 0;
  /** fractional (default) or molecular weighting of the sites (pair_style exp6/rx keyword). */
  private fractional = true;
  private rx: FixRX | null = null;
  private nspecies = 0;
  private params: Exp6Param[] = [];
  private mol2param = new Int32Array(0);
  private isite1 = -2;
  private isite2 = -2;
  private scaling: 'none' | 'exponent' | 'polynomial' = 'none';
  private exponentR = 0;
  private exponentEpsilon = 0;
  private coeffAlpha = new Float64Array(6);
  private coeffEps = new Float64Array(6);
  private coeffRm = new Float64Array(6);
  private setflag = new Uint8Array(0);
  private cutSet = new Float64Array(0);
  /** Per owned atom, MIX values (see the index constants above). */
  private W = new Float64Array(0);
  private readonly order = new NativePairOrder();
  private readonly out = new Float64Array(2);

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.setflag = new Uint8Array(nt * nt);
    this.cutSet = new Float64Array(nt * nt);
  }

  settings(args: string[]): void {
    if (args.length < 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = numArg(args[0]);
    for (const w of args.slice(1)) {
      if (w === 'fractional') this.fractional = true;
      else if (w === 'molecular') this.fractional = false;
      else throw new StyleError('Illegal pair_style command');
    }
    // pair_exp6_rx.cpp settings: the cutoff of the pairs already set becomes the global one
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) for (let j = i; j < nt; j++) if (this.setflag[i * nt + j]) this.cutSet[i * nt + j] = this.cutGlobal;
  }

  /** pair_exp6_rx.cpp coeff: "I J file site1 site2 scaling [args] [cutoff]" (6 to 9 words). */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 6 || args.length > 9) throw new StyleError('Incorrect args for pair coefficients');
    const fx = ctx.fixOf?.('rx');
    if (!(fx instanceof FixRX)) throw new StyleError('Fix rx not defined or not compatible with pair style');
    this.rx = fx;
    const nt = this.ntypes + 1;
    if (this.setflag.length !== nt * nt) this.allocate(this.ntypes);
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    this.nspecies = fx.nspecies;
    if (this.nspecies === 0) throw new StyleError('There are no rx species specified.');
    this.readFile(ctx.readFile(args[2]), args[2], fx);
    this.isite1 = this.siteOf(args[3], 'Site1', args[2]);
    this.isite2 = this.siteOf(args[4], 'Site2', args[2]);
    this.setup();
    let cutOne = this.cutGlobal;
    if (args[5] === 'exponent') {
      this.scaling = 'exponent';
      this.exponentR = numArg(args[6]);
      this.exponentEpsilon = numArg(args[7]);
      if (args.length === 9) cutOne = numArg(args[8]);
    } else if (args[5] === 'polynomial') {
      this.scaling = 'polynomial';
      this.readFile2(ctx.readFile(args[6]), args[6]);
      if (args.length > 8) throw new StyleError('Incorrect args for pair coefficients');
      if (args.length === 8) cutOne = numArg(args[7]);
    } else if (args[5] === 'none') {
      this.scaling = 'none';
      if (args.length > 7) throw new StyleError('Incorrect args for pair coefficients');
      if (args.length === 7) cutOne = numArg(args[6]);
    } else {
      throw new StyleError('Incorrect args for pair coefficients');
    }
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.cutSet[i * nt + j] = cutOne;
        this.setflag[i * nt + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** The site of a pair_coeff word: 1fluid (-1) or a species of the rx fix. */
  private siteOf(word: string, which: string, file: string): number {
    if (word === '1fluid') return -1;
    const k = this.rx!.speciesIndexOf(word);
    if (k === undefined) throw new StyleError(`${which} name not recognized in pair coefficients`);
    if (!this.params.some((p) => p.ispecies === k)) throw new StyleError(`${which} species ${word} has no exp6 parameters in file ${file}`);
    return k;
  }

  /** pair_exp6_rx.cpp read_file: "species exp6 alpha epsilon rm" for the species of the rx fix. */
  private readFile(text: string, file: string, fx: FixRX): void {
    this.params = [];
    const lines = new TextLines(text, file);
    for (let words = lines.next(5); words; words = lines.next(5)) {
      const tok = new Tokens(words, words.join(' '));
      const species = tok.next();
      const ispecies = fx.speciesIndexOf(species);
      if (ispecies === undefined) continue;
      const potential = tok.next();
      if (potential !== 'exp6') throw new StyleError(`Illegal exp6/rx parameters in file ${file}. Interaction potential does not exist.`);
      const alpha = tok.double(), epsilon = tok.double(), rm = tok.double();
      if (epsilon <= 0.0 || rm <= 0.0 || alpha < 0.0) {
        throw new StyleError(`Illegal exp6/rx parameters in file ${file}. Rm and Epsilon must be greater than zero. Alpha cannot be negative.`);
      }
      if (tok.hasNext()) throw new StyleError(`Misplaced characters at end of line in file ${file}. Full line: ${words.join(' ')}`);
      this.params.push({ epsilon, rm, alpha, ispecies });
    }
  }

  /** pair_exp6_rx.cpp read_file2: the polynomial scaling coefficients ("alpha", "epsilon" and "rm" lines of six). */
  private readFile2(text: string, file: string): void {
    const lines = new TextLines(text, file);
    for (let words = lines.next(7); words; words = lines.next(7)) {
      const tok = new Tokens(words, words.join(' '));
      const kind = tok.next();
      let target: Float64Array | null = null;
      if (kind === 'alpha') target = this.coeffAlpha;
      else if (kind === 'epsilon') target = this.coeffEps;
      else if (kind === 'rm') target = this.coeffRm;
      if (target) for (let ii = 0; ii < 6; ii++) target[ii] = tok.double();
      if (tok.hasNext()) throw new StyleError(`Misplaced characters at end of line in file ${file}. Full line: ${words.join(' ')}`);
    }
  }

  /** pair_exp6_rx.cpp setup: the parameter set of each species (mol2param), a duplicate is an error. */
  private setup(): void {
    const ns = this.nspecies;
    this.mol2param = new Int32Array(ns);
    for (let i = 0; i < ns; i++) {
      let n = -1;
      for (let j = 0; j < this.params.length; j++) {
        if (i === this.params[j].ispecies) {
          if (n >= 0) throw new StyleError('Potential file has duplicate entry');
          n = j;
        }
      }
      this.mol2param[i] = n;
    }
  }

  initStyle(ctx: StyleContext): void {
    // pair_exp6_rx.cpp: the per-atom uCG and uCGnew exist for atom_style dpd only (the native per-atom energies)
    if (!ctx.s || !ctx.s.custom.has(DPD_UCG)) throw new StyleError('pair_style exp6/rx requires atom_style with internal temperature and energies (e.g. dpd)');
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.setflag[i * nt + j]) throw new StyleError(`All pair coeffs are not set. Status: pair ${i} ${j}`);
    return this.cutSet[i * nt + j];
  }

  /** pair_exp6_rx.cpp getMixingWeights for atom id, written to W[16 id ...]. */
  private mixing(id: number, cur: Float64Array[], old: Float64Array[]): void {
    const ns = this.nspecies;
    const W = this.W;
    const b = 16 * id;
    const isite1 = this.isite1, isite2 = this.isite2;
    const oneFluid = isite1 === -1 || isite2 === -1;
    let rm3 = 0.0, epsilon = 0.0, alpha = 0.0, epsilonOld = 0.0, rm3Old = 0.0, alphaOld = 0.0;
    let fractionOFA = 0.0, fractionOFAold = 0.0, nMoleculesOFA = 0.0, nMoleculesOFAold = 0.0;
    let nTotal = 0.0, nTotalOld = 0.0;
    for (let k = 0; k < ns; k++) {
      nTotal += cur[k][id];
      nTotalOld += old[k][id];
      const ip = this.mol2param[k];
      if (ip < 0) continue;
      if (oneFluid) {
        if (isite1 === this.params[ip].ispecies || isite2 === this.params[ip].ispecies) continue;
        nMoleculesOFAold += old[k][id];
        nMoleculesOFA += cur[k][id];
      }
    }
    if (nTotal < MY_EPSILON || nTotalOld < MY_EPSILON) {
      throw new StyleError('The number of molecules in CG particle is less than 10*DBL_EPSILON.');
    }
    fractionOFAold = nMoleculesOFAold / nTotalOld;
    fractionOFA = nMoleculesOFA / nTotal;
    // the site values of the two named species (rm, epsilon, alpha and the fractions of the species)
    let rm1 = 0.0, epsilon1 = 0.0, alpha1 = 0.0, rm2 = 0.0, epsilon2 = 0.0, alpha2 = 0.0;
    let rm1Old = 0.0, epsilon1Old = 0.0, alpha1Old = 0.0, rm2Old = 0.0, epsilon2Old = 0.0, alpha2Old = 0.0;
    let nMolecules1 = 0.0, nMoleculesOld1 = 0.0, nMolecules2 = 0.0, nMoleculesOld2 = 0.0;
    let fraction1 = 0.0, fractionOld1 = 0.0, fraction2 = 0.0, fractionOld2 = 0.0;
    for (let k = 0; k < ns; k++) {
      const ip = this.mol2param[k];
      if (ip < 0) continue;
      const p = this.params[ip];
      if (isite1 === p.ispecies) {
        rm1Old = p.rm; rm1 = p.rm;
        epsilon1Old = p.epsilon; epsilon1 = p.epsilon;
        alpha1Old = p.alpha; alpha1 = p.alpha;
        nMoleculesOld1 = old[k][id];
        nMolecules1 = cur[k][id];
        fractionOld1 = nMoleculesOld1 / nTotalOld;
        fraction1 = nMolecules1 / nTotal;
      }
      if (isite2 === p.ispecies) {
        rm2Old = p.rm; rm2 = p.rm;
        epsilon2Old = p.epsilon; epsilon2 = p.epsilon;
        alpha2Old = p.alpha; alpha2 = p.alpha;
        nMoleculesOld2 = old[k][id];
        nMolecules2 = cur[k][id];
        fractionOld2 = old[k][id] / nTotalOld;
        fraction2 = cur[k][id] / nTotal;
      }
      if (oneFluid) {
        if (isite1 === p.ispecies || isite2 === p.ispecies) continue;
        const rmi = p.rm, epsiloni = p.epsilon, alphai = p.alpha;
        const xMolei = nMoleculesOFA < MY_EPSILON ? 0.0 : cur[k][id] / nMoleculesOFA;
        const xMoleiOld = nMoleculesOFAold < MY_EPSILON ? 0.0 : old[k][id] / nMoleculesOFAold;
        for (let kk = 0; kk < ns; kk++) {
          const jp = this.mol2param[kk];
          if (jp < 0) continue;
          const q = this.params[jp];
          if (isite1 === q.ispecies || isite2 === q.ispecies) continue;
          const rmj = q.rm, epsilonj = q.epsilon, alphaj = q.alpha;
          const xMolej = nMoleculesOFA < MY_EPSILON ? 0.0 : cur[kk][id] / nMoleculesOFA;
          const xMolejOld = nMoleculesOFAold < MY_EPSILON ? 0.0 : old[kk][id] / nMoleculesOFAold;
          const rmij = (rmi + rmj) / 2.0;
          const rm3ij = rmij * rmij * rmij;
          const epsilonij = Math.sqrt(epsiloni * epsilonj);
          const alphaij = Math.sqrt(alphai * alphaj);
          if (fractionOFAold > 0.0) {
            rm3Old += xMoleiOld * xMolejOld * rm3ij;
            epsilonOld += xMoleiOld * xMolejOld * rm3ij * epsilonij;
            alphaOld += xMoleiOld * xMolejOld * rm3ij * epsilonij * alphaij;
          }
          if (fractionOFA > 0.0) {
            rm3 += xMolei * xMolej * rm3ij;
            epsilon += xMolei * xMolej * rm3ij * epsilonij;
            alpha += xMolei * xMolej * rm3ij * epsilonij * alphaij;
          }
        }
      }
    }
    // the one-fluid site values (the same for both sites)
    if (isite1 === -1) {
      rm1 = Math.cbrt(rm3);
      if (rm1 < MY_EPSILON) {
        rm1 = 0.0; epsilon1 = 0.0; alpha1 = 0.0;
      } else {
        epsilon1 = epsilon / rm3;
        alpha1 = alpha / epsilon1 / rm3;
      }
      nMolecules1 = 1.0 - (nTotal - nMoleculesOFA);
      fraction1 = fractionOFA;
      rm1Old = Math.cbrt(rm3Old);
      if (rm1Old < MY_EPSILON) {
        rm1Old = 0.0; epsilon1Old = 0.0; alpha1Old = 0.0;
      } else {
        epsilon1Old = epsilonOld / rm3Old;
        alpha1Old = alphaOld / epsilon1Old / rm3Old;
      }
      nMoleculesOld1 = 1.0 - (nTotalOld - nMoleculesOFAold);
      fractionOld1 = fractionOFAold;
      if (this.scaling === 'exponent') {
        [epsilon1, rm1] = this.exponentScaling(nMoleculesOFA, epsilon1, rm1);
        [epsilon1Old, rm1Old] = this.exponentScaling(nMoleculesOFAold, epsilon1Old, rm1Old);
      } else if (this.scaling === 'polynomial') {
        [alpha1, epsilon1, rm1] = this.polynomialScaling(nMoleculesOFA, alpha1, epsilon1, rm1);
        [alpha1Old, epsilon1Old, rm1Old] = this.polynomialScaling(nMoleculesOFAold, alpha1Old, epsilon1Old, rm1Old);
      }
    }
    if (isite2 === -1) {
      rm2 = Math.cbrt(rm3);
      if (rm2 < MY_EPSILON) {
        rm2 = 0.0; epsilon2 = 0.0; alpha2 = 0.0;
      } else {
        epsilon2 = epsilon / rm3;
        alpha2 = alpha / epsilon2 / rm3;
      }
      nMolecules2 = 1.0 - (nTotal - nMoleculesOFA);
      fraction2 = fractionOFA;
      rm2Old = Math.cbrt(rm3Old);
      if (rm2Old < MY_EPSILON) {
        rm2Old = 0.0; epsilon2Old = 0.0; alpha2Old = 0.0;
      } else {
        epsilon2Old = epsilonOld / rm3Old;
        alpha2Old = alphaOld / epsilon2Old / rm3Old;
      }
      nMoleculesOld2 = 1.0 - (nTotalOld - nMoleculesOFAold);
      fractionOld2 = fractionOFAold;
      if (this.scaling === 'exponent') {
        [epsilon2, rm2] = this.exponentScaling(nMoleculesOFA, epsilon2, rm2);
        [epsilon2Old, rm2Old] = this.exponentScaling(nMoleculesOFAold, epsilon2Old, rm2Old);
      } else if (this.scaling === 'polynomial') {
        [alpha2, epsilon2, rm2] = this.polynomialScaling(nMoleculesOFA, alpha2, epsilon2, rm2);
        [alpha2Old, epsilon2Old, rm2Old] = this.polynomialScaling(nMoleculesOFAold, alpha2Old, epsilon2Old, rm2Old);
      }
    }
    // negative fractions are rounding (clamped), anything below -10 DBL_EPSILON is an error
    const clampFraction = (frac: number, n: number): [number, number] => {
      if (frac < 0.0 || n < 0.0) {
        if (frac < -MY_EPSILON || n < -MY_EPSILON) throw new StyleError('Computed fraction less than -10*DBL_EPSILON');
        return [0.0, 0.0];
      }
      return [frac, n];
    };
    [fraction1, nMolecules1] = clampFraction(fraction1, nMolecules1);
    [fraction2, nMolecules2] = clampFraction(fraction2, nMolecules2);
    [fractionOld1, nMoleculesOld1] = clampFraction(fractionOld1, nMoleculesOld1);
    [fractionOld2, nMoleculesOld2] = clampFraction(fractionOld2, nMoleculesOld2);
    W[b + EPS1] = epsilon1; W[b + ALPHA1] = alpha1; W[b + RM1] = rm1;
    W[b + EPS2] = epsilon2; W[b + ALPHA2] = alpha2; W[b + RM2] = rm2;
    W[b + EPS1O] = epsilon1Old; W[b + ALPHA1O] = alpha1Old; W[b + RM1O] = rm1Old;
    W[b + EPS2O] = epsilon2Old; W[b + ALPHA2O] = alpha2Old; W[b + RM2O] = rm2Old;
    if (this.fractional) {
      W[b + WT1] = fraction1; W[b + WT2] = fraction2; W[b + WT1O] = fractionOld1; W[b + WT2O] = fractionOld2;
    } else {
      W[b + WT1] = nMolecules1; W[b + WT2] = nMolecules2; W[b + WT1O] = nMoleculesOld1; W[b + WT2O] = nMoleculesOld2;
    }
  }

  /** pair_exp6_rx.cpp exponentScaling: epsilon and rm times phi^exponent (negative exponents divide). */
  private exponentScaling(phi: number, epsilon: number, rm: number): [number, number] {
    let eps = epsilon, r = rm;
    if (this.exponentEpsilon < 0.0) {
      const powfuch = libmPow(phi, -this.exponentEpsilon);
      if (powfuch < MY_EPSILON) eps = 0.0;
      else eps *= 1.0 / powfuch;
    } else {
      eps *= libmPow(phi, this.exponentEpsilon);
    }
    if (this.exponentR < 0.0) {
      const powfuch = libmPow(phi, -this.exponentR);
      if (powfuch < MY_EPSILON) r = 0.0;
      else r *= 1.0 / powfuch;
    } else {
      r *= libmPow(phi, this.exponentR);
    }
    return [eps, r];
  }

  /** pair_exp6_rx.cpp polynomialScaling: alpha, epsilon and rm from the fifth-degree polynomials of phi. */
  private polynomialScaling(phi: number, alpha: number, epsilon: number, rm: number): [number, number, number] {
    const phi2 = phi * phi, phi3 = phi2 * phi, phi4 = phi2 * phi2, phi5 = phi2 * phi3;
    const c = this.coeffAlpha, e = this.coeffEps, m = this.coeffRm;
    const a = c[0] * phi5 + c[1] * phi4 + c[2] * phi3 + c[3] * phi2 + c[4] * phi + c[5];
    const eps = epsilon * (e[0] * phi5 + e[1] * phi4 + e[2] * phi3 + e[3] * phi2 + e[4] * phi + e[5]);
    const r = rm * (m[0] * phi5 + m[1] * phi4 + m[2] * phi3 + m[3] * phi2 + m[4] * phi + m[5]);
    return [a, eps, r];
  }

  compute(pc: PairCompute): void {
    const rx = this.rx;
    if (!rx) throw new StyleError('pair exp6/rx used before pair_coeff');
    const s = pc.s;
    const cur = rx.speciesData(s);
    const old = rx.speciesNames.map((n) => s.custom.get(`${n}Old`)!.data);
    const uCG = s.custom.get(DPD_UCG)!.data, uCGnew = s.custom.get(DPD_UCGNEW)!.data;
    const nl = pc.nlocal;
    if (this.W.length !== MIX * nl) this.W = new Float64Array(MIX * nl);
    for (let i = 0; i < nl; i++) this.mixing(i, cur, old);
    const W = this.W;
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const sLJ = pc.specialLJ;
    const owner = pc.nb.owner;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const isite1 = this.isite1, isite2 = this.isite2;
    const same = isite1 === isite2;
    const out = this.out;
    let evdwlOld = 0.0;
    let evAcc = 0.0;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const jj = pairs[p + 1];
      const j = jj & NEIGHMASK;
      const factorLJ = sLJ[jj >>> SBBITS];
      const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const t = type[i] * nt + type[j];
      const csq = this.cutsq[t];
      if (!(rsq < csq)) continue;
      const r2inv = 1.0 / rsq;
      const r6inv = r2inv * r2inv * r2inv;
      const r = Math.sqrt(rsq);
      const rCut2inv = 1.0 / csq;
      const rCut6inv = rCut2inv * rCut2inv * rCut2inv;
      const rCut = Math.sqrt(csq);
      const rCutInv = 1.0 / rCut;
      const oi = i, oj = owner[j];
      const bi = MIX * oi, bj = MIX * oj;
      const eps1i = W[bi + EPS1], alpha1i = W[bi + ALPHA1], rm1i = W[bi + RM1], wt1i = W[bi + WT1];
      const eps2i = W[bi + EPS2], alpha2i = W[bi + ALPHA2], rm2i = W[bi + RM2], wt2i = W[bi + WT2];
      const eps1Oi = W[bi + EPS1O], alpha1Oi = W[bi + ALPHA1O], rm1Oi = W[bi + RM1O], wt1Oi = W[bi + WT1O];
      const eps2Oi = W[bi + EPS2O], alpha2Oi = W[bi + ALPHA2O], rm2Oi = W[bi + RM2O], wt2Oi = W[bi + WT2O];
      const eps1j = W[bj + EPS1], alpha1j = W[bj + ALPHA1], rm1j = W[bj + RM1], wt1j = W[bj + WT1];
      const eps2j = W[bj + EPS2], alpha2j = W[bj + ALPHA2], rm2j = W[bj + RM2], wt2j = W[bj + WT2];
      const eps1Oj = W[bj + EPS1O], alpha1Oj = W[bj + ALPHA1O], rm1Oj = W[bj + RM1O], wt1Oj = W[bj + WT1O];
      const eps2Oj = W[bj + EPS2O], alpha2Oj = W[bj + ALPHA2O], rm2Oj = W[bj + RM2O], wt2Oj = W[bj + WT2O];
      // Lorentz-Berthelot mixing of the two sites (old concentrations, then new)
      const alphaOld12 = Math.sqrt(alpha1Oi * alpha2Oj);
      const rmOld12 = 0.5 * (rm1Oi + rm2Oj);
      const epsOld12 = Math.sqrt(eps1Oi * eps2Oj);
      const alphaOld21 = Math.sqrt(alpha2Oi * alpha1Oj);
      const rmOld21 = 0.5 * (rm2Oi + rm1Oj);
      const epsOld21 = Math.sqrt(eps2Oi * eps1Oj);
      const alpha12 = Math.sqrt(alpha1i * alpha2j);
      const rm12 = 0.5 * (rm1i + rm2j);
      const eps12 = Math.sqrt(eps1i * eps2j);
      const alpha21 = Math.sqrt(alpha2i * alpha1j);
      const rm21 = 0.5 * (rm2i + rm1j);
      const eps21 = Math.sqrt(eps2i * eps1j);
      let evOld12 = 0.0, evOld21 = 0.0, fOld12 = 0.0, fOld21 = 0.0;
      let evNew12 = 0.0, evNew21 = 0.0;
      if (rmOld12 !== 0.0 && rmOld21 !== 0.0) {
        if (alphaOld21 === 6.0 || alphaOld12 === 6.0) throw new StyleError('alpha_ij is 6.0 in pair exp6');
        exp6Pair(r, rCut, rCutInv, rCut6inv, r6inv, epsOld12, alphaOld12, rmOld12, out);
        evOld12 = out[0];
        fOld12 = factorLJ * out[1] * r2inv;
        exp6Pair(r, rCut, rCutInv, rCut6inv, r6inv, epsOld21, alphaOld21, rmOld21, out);
        evOld21 = out[0];
        fOld21 = factorLJ * out[1] * r2inv;
        evdwlOld = same ? Math.sqrt(wt1Oi * wt2Oj) * evOld12
          : Math.sqrt(wt1Oi * wt2Oj) * evOld12 + Math.sqrt(wt2Oi * wt1Oj) * evOld21;
        evdwlOld *= factorLJ;
        uCG[oi] += 0.5 * evdwlOld;
        uCG[oj] += 0.5 * evdwlOld;
      }
      if (rm12 !== 0.0 && rm21 !== 0.0) {
        if (alpha21 === 6.0 || alpha12 === 6.0) throw new StyleError('alpha_ij is 6.0 in pair exp6');
        exp6Pair(r, rCut, rCutInv, rCut6inv, r6inv, eps12, alpha12, rm12, out);
        evNew12 = out[0];
        exp6Pair(r, rCut, rCutInv, rCut6inv, r6inv, eps21, alpha21, rm21, out);
        evNew21 = out[0];
      }
      // the force is the mixed old-concentration force; the energy of the new concentrations goes to uCGnew
      const fpair = same ? Math.sqrt(wt1Oi * wt2Oj) * fOld12
        : Math.sqrt(wt1Oi * wt2Oj) * fOld12 + Math.sqrt(wt2Oi * wt1Oj) * fOld21;
      const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      let evdwl = same ? Math.sqrt(wt1i * wt2j) * evNew12
        : Math.sqrt(wt1i * wt2j) * evNew12 + Math.sqrt(wt2i * wt1j) * evNew21;
      evdwl *= factorLJ;
      uCGnew[oi] += 0.5 * evdwl;
      uCGnew[oj] += 0.5 * evdwl;
      // the energy tallied is the old-concentration one (native: evdwl = evdwlOld before ev_tally)
      evAcc += evdwlOld;
      if (tally) tallyXYZ(pc, i, j, evdwlOld, fx, fy, fz, delx, dely, delz);
    }
    pc.acc.evdwl += evAcc;
  }
}
