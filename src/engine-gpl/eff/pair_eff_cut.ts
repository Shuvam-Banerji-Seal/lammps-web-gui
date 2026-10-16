/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/pair_eff_cut.cpp and
 * pair_eff_cut.h (class PairEffCut, pair_style eff/cut: the electron force field pair style with the
 * limit/eradius, pressure/evirials and ecp keywords, the fixed-core (espin 2), electron (espin +-1) and
 * pseudo-core (ECP, espin 3) interactions, the radial electron forces erforce and the radial virial, the ECP
 * Pauli parameters of pair_coeff, and the pvector of compute pair: KE, Pauli, Coulomb and the radius restraint).
 * Contributing author (from the source): Andres Jaramillo-Botero (Caltech). espin 4 pseudo-cores are refused.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { Pair, PairParams, StyleError, mixDistance, type PairCompute, type PairMinPerAtom, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import { ELECTRON_RADIUS, ELECTRON_RFORCE, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';
import type { SimState } from '../../engine/types';
import {
  kinElec, elecNucNuc, elecNucElec, elecElecElec, pauliElecElec, smallRForce, cutoff, dcutoff,
  elecCoreNuc, elecCoreCore, elecCoreElec, pauliCoreElec, pauliCorePElec,
} from './math';
import { log as libmLog } from '../libm/log';

/** units real: qqr2e and the Hartree/Bohr conversions used by pair eff/cut (initStyle, "Check your units" otherwise). */
const QQR2E_REAL = 332.06371;
const QQR2E_ELECTRON = 1.0;
const HHMSS2E_REAL = 175.72044219620075;
/** hartree -> kcal/mol (h2e of pair_eff_cut.cpp settings), applied to the ECP Pauli terms. */
const H2E_REAL = 627.509;
/** Largest atomic number with ECP parameters (the arrays are indexed by element Z, as in pair_eff_cut.cpp). */
const ECP_MAX_Z = 100;

/** Default ECP parameters by element Z (pair_eff_cut.cpp settings): s-type (D = E = 0) or p-type (D, E). */
const ECP_DEFAULTS: Record<number, [number, number, number, number, number]> = {
  6: [22.721015, 0.728733, 1.103199, 17.695345, 6.693621],
  7: [16.242367, 0.602818, 1.081856, 7.150803, 5.351936],
  8: [29.5185, 0.32995, 1.21676, 11.98757, 3.073417],
  13: [0.486, 1.049, 0.207, 0.0, 0.0],
  14: [0.320852, 2.283269, 0.814857, 0.0, 0.0],
};
/** The element symbols of the ecp keyword and their Z (pair_eff_cut.cpp settings). */
const ECP_ELEMENTS: Record<string, number> = { C: 6, N: 7, O: 8, Al: 13, Si: 14 };

/** The per-atom electron property stored in the state (created by atom_style electron). */
const electronArray = (s: SimState, name: string): Float64Array => {
  const c = s.custom.get(name);
  if (!c) throw new StyleError('Pair eff/cut requires atom attributes q, spin, eradius, erforce (atom_style electron)');
  return c.data;
};

export class PairEffCut extends Pair {
  readonly name = 'eff/cut';
  needsHalf = true;
  needsFull = false;
  /** The global virial is tallied per pair (equal to the sum x.f of the pair forces). */
  virialFdotr = false;

  cutGlobal = 0;
  limitEradius = false;
  pressureEvirials = false;
  /** Unit conversion: hartree -> energy units of the style (1 for units electron, see initStyle). */
  private hhmss2e = 1;
  /** hartree -> energy units of the ECP Pauli terms (h2e in pair_eff_cut.cpp: 1 for units electron). */
  private h2e = 1;
  private p!: PairParams;
  /** ecp keyword: the element Z of each atom type (pair_eff_cut.cpp ecp_type, indexed by type). */
  private ecpType: number[] = [];
  /** ECP Pauli parameters A..E indexed by element Z (pair_eff_cut.cpp PAULI_CORE_A..E). */
  private pauliA = new Float64Array(ECP_MAX_Z);
  private pauliB = new Float64Array(ECP_MAX_Z);
  private pauliC = new Float64Array(ECP_MAX_Z);
  private pauliD = new Float64Array(ECP_MAX_Z);
  private pauliE = new Float64Array(ECP_MAX_Z);
  private ecpFound = false;
  /** compute pair vector: [KE, Pauli, ecoul, radial restraint] (pair_eff_cut.cpp pvector). */
  private pvector = new Float64Array(4);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length < 1) throw new StyleError('Illegal pair_style command');
    // pair_eff_cut.cpp settings: the default ECP parameters of C, N, O, Al and Si are set on every pair_style
    for (const z of Object.keys(ECP_DEFAULTS).map(Number)) {
      const d = ECP_DEFAULTS[z];
      this.pauliA[z] = d[0]; this.pauliB[z] = d[1]; this.pauliC[z] = d[2]; this.pauliD[z] = d[3]; this.pauliE[z] = d[4];
    }
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.limitEradius = false;
    this.pressureEvirials = false;
    this.ecpFound = false;
    this.ecpType = [];
    let iarg = 1;
    while (iarg < args.length) {
      const w = args[iarg];
      if (w === 'limit/eradius') { this.limitEradius = true; iarg += 1; }
      else if (w === 'pressure/evirials') { this.pressureEvirials = true; iarg += 1; }
      else if (w === 'ecp') {
        // pair_eff_cut.cpp settings: "ecp" takes pairs "atomtype element" up to the end of the command
        iarg += 1;
        while (iarg < args.length) {
          const t = Number(args[iarg]);
          if (!Number.isInteger(t) || t < 1) throw new StyleError(`Illegal pair_style eff/cut ecp atom type: ${args[iarg]}`);
          const el = args[iarg + 1];
          const z = el !== undefined && Object.hasOwn(ECP_ELEMENTS, el) ? ECP_ELEMENTS[el] : undefined;
          if (z === undefined) throw new StyleError(`No default parameters for this atom ECP (ecp ${args[iarg]} ${el ?? ''})`);
          this.ecpType[t] = z;
          iarg += 2;
          this.ecpFound = true;
        }
      }
      else throw new StyleError('Illegal pair style command');
    }
    // pair_eff_cut.cpp settings: the cutoffs that were set explicitly are reset to the new global cutoff
    if (this.p) {
      for (let i = 1; i <= this.ntypes; i++) {
        for (let j = i; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.set(i, j, [this.cutGlobal]);
      }
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['cut']);
  }

  /** The ECP lines "I s ..." and "I p ..." keep their words (pair_eff_cut.cpp coeff: arg[1] is "s" or "p"). */
  rawCoeff(args: string[]): boolean {
    return args.length >= 2 && args[0] !== '*' && args[1] !== '*' && (args[1] === 's' || args[1] === 'p');
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 2) throw new StyleError('Incorrect args for pair coefficients');
    if (args[0] !== '*' && args[1] !== '*') {
      // pair_eff_cut.cpp coeff, else branch: "pair_coeff <atom type> s A B C" or "... p A B C D E" (ECP parameters)
      const t = Number(args[0]);
      if (!Number.isInteger(t) || t < 1) throw new StyleError(`Illegal pair_coeff ECP atom type: ${args[0]}`);
      const z = this.ecpElement(t);
      if (args[1] === 's') {
        if (args.length !== 5) throw new StyleError('Illegal pair_coeff command: "s" takes A B C');
        this.setPauli(z, [parseNum(args[2], 'A'), parseNum(args[3], 'B'), parseNum(args[4], 'C'), 0, 0]);
      } else if (args[1] === 'p') {
        if (args.length !== 7) throw new StyleError('Illegal pair_coeff command: "p" takes A B C D E');
        this.setPauli(z, [parseNum(args[2], 'A'), parseNum(args[3], 'B'), parseNum(args[4], 'C'), parseNum(args[5], 'D'), parseNum(args[6], 'E')]);
      } else throw new StyleError('Illegal pair_coeff command');
      return;
    }
    if (args.length > 3) throw new StyleError('Incorrect args for pair coefficients');
    const cutOne = args.length === 3 ? parseNum(args[2], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [cutOne]);
  }

  /** The element Z of an atom type set by the ecp keyword of pair_style (refused when the type has none). */
  private ecpElement(type: number): number {
    const z = this.ecpType[type];
    if (z === undefined) throw new StyleError(`pair_coeff ECP: atom type ${type} has no element (pair_style eff/cut ecp ${type} <element>)`);
    return z;
  }

  /** Stores the ECP parameters A..E of element z (pair_eff_cut.cpp coeff: PAULI_CORE_*[ecp_type[ecp]]). */
  private setPauli(z: number, v: number[]): void {
    this.pauliA[z] = v[0]; this.pauliB[z] = v[1]; this.pauliC[z] = v[2]; this.pauliD[z] = v[3]; this.pauliE[z] = v[4];
  }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) throw new StyleError('pair_style eff/cut needs the simulation box (create_box) first');
    if (!isElectronStyle(s.atomStyle)) throw new StyleError('Pair eff/cut requires atom attributes q, spin, eradius, erforce (atom_style electron)');
    // pair_eff_cut.cpp init_style: the q, spin, eradius and erforce attributes are required
    electronArray(s, ELECTRON_SPIN);
    electronArray(s, ELECTRON_RADIUS);
    electronArray(s, ELECTRON_RFORCE);
    // pair_eff_cut.cpp settings: real units and electron units only, with their Hartree conversions
    if (s.units.qqr2e === QQR2E_REAL) { this.hhmss2e = HHMSS2E_REAL; this.h2e = H2E_REAL; }
    else if (s.units.qqr2e === QQR2E_ELECTRON) { this.hhmss2e = 1; this.h2e = 1; }
    else throw new StyleError('Check your units');
    // pair_eff_cut.cpp init_style: espin 3 (pseudo-core) atoms need the ecp keyword of pair_style
    const spin = electronArray(s, ELECTRON_SPIN);
    let anyCore = false;
    for (let i = 0; i < s.n; i++) {
      if (spin[i] === 4) throw new StyleError('pair_style eff/cut: atoms with espin 4 (pseudo-core) are not ported yet');
      if (spin[i] === 3) {
        anyCore = true;
        this.ecpElement(s.type[i]);
      }
    }
    if (anyCore && !this.ecpFound) throw new StyleError('Need to specify ECP type on pair_style command');
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) {
      // pair_eff_cut.cpp init_one: the cutoff of an unset pair is mixed from its two like pairs
      this.p.setMixed(i, j, 'cut', mixDistance(this.mix, this.p.get('cut', i, i), this.p.get('cut', j, j)));
    }
    return this.p.get('cut', i, j);
  }

  /**
   * The minimizer's per-atom degrees of freedom are the logarithm of each electron radius (pair_eff_cut.cpp
   * min_xf_get and min_x_set); the force on that coordinate is eradius erforce. Nuclei and fixed cores with
   * spin 0 have no radius to move: their coordinate and force are 0. extra_max is the 0.01 of init_style.
   */
  minPerAtom(): PairMinPerAtom {
    return {
      maxStep: 0.01,
      pull: (st, x, f) => {
        const spin = electronArray(st, ELECTRON_SPIN), er = electronArray(st, ELECTRON_RADIUS), rf = electronArray(st, ELECTRON_RFORCE);
        for (let i = 0; i < st.n; i++) {
          if (spin[i]) {
            x[i] = libmLog(er[i]);
            f[i] = er[i] * rf[i];
          } else {
            x[i] = 0;
            f[i] = 0;
          }
        }
      },
      push: (st, x) => {
        const spin = electronArray(st, ELECTRON_SPIN), er = electronArray(st, ELECTRON_RADIUS);
        for (let i = 0; i < st.n; i++) if (spin[i]) er[i] = Math.exp(x[i]);
      },
    };
  }

  extract(name: string): unknown {
    return name === 'pvector' ? this.pvector : undefined;
  }

  /** Pair::ev_tally_xyz with the electron conventions of ev_tally_eff (pair_eff_cut.cpp): energy to ecoul, pair virial. */
  private tallyXYZ(pc: PairCompute, i: number, j: number, energy: number, fx: number, fy: number, fz: number, dx: number, dy: number, dz: number): void {
    // the six virial terms are written out (no array per pair: this runs for every neighbour pair)
    const a = pc.acc;
    a.ecoul += energy;
    const v0 = dx * fx, v1 = dy * fy, v2 = dz * fz, v3 = dx * fy, v4 = dx * fz, v5 = dy * fz;
    a.virial[0] += v0; a.virial[1] += v1; a.virial[2] += v2; a.virial[3] += v3; a.virial[4] += v4; a.virial[5] += v5;
    if (pc.eatom) { pc.eatom[i] += 0.5 * energy; pc.eatom[j] += 0.5 * energy; }
    if (pc.vatom) {
      const va = pc.vatom;
      va[6 * i] += 0.5 * v0; va[6 * i + 1] += 0.5 * v1; va[6 * i + 2] += 0.5 * v2;
      va[6 * i + 3] += 0.5 * v3; va[6 * i + 4] += 0.5 * v4; va[6 * i + 5] += 0.5 * v5;
      va[6 * j] += 0.5 * v0; va[6 * j + 1] += 0.5 * v1; va[6 * j + 2] += 0.5 * v2;
      va[6 * j + 3] += 0.5 * v3; va[6 * j + 4] += 0.5 * v4; va[6 * j + 5] += 0.5 * v5;
    }
  }

  /**
   * pair_eff_cut.cpp ev_tally_eff: the energy of an electron term goes to eng_coul (newton on), and its radial
   * virial (eVirial / 3 per diagonal component) only to the per-atom virial of the electrons with spin != 0.
   */
  private tallyEff(pc: PairCompute, i: number, j: number, energy: number, eVirial: number, spin: Float64Array): void {
    pc.acc.ecoul += energy;
    if (pc.eatom) { pc.eatom[i] += 0.5 * energy; pc.eatom[j] += 0.5 * energy; }
    if (pc.vatom) {
      const partial = eVirial / 3.0;
      const half = partial / 2;
      const va = pc.vatom;
      if (spin[i]) for (let c = 0; c < 3; c++) va[6 * i + c] += half;
      if (spin[j]) for (let c = 0; c < 3; c++) va[6 * j + c] += half;
    }
  }

  compute(pc: PairCompute): void {
    const list = pc.half;
    if (!list) throw new Error('pair style eff/cut needs a half neighbor list');
    const s = pc.s;
    const nb = pc.nb;
    const owner = nb.owner;
    const nall = pc.nall;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const qqrd2e = pc.qqrd2e;
    const hh = this.hhmss2e;
    const pv = this.pvector;
    pv.fill(0);
    // owned electron properties; ghost copies (owner image) are built here, as the ghosts carry them in LAMMPS
    const spinO = electronArray(s, ELECTRON_SPIN), eradO = electronArray(s, ELECTRON_RADIUS), erfO = electronArray(s, ELECTRON_RFORCE);
    const spin = new Float64Array(nall), erad = new Float64Array(nall), erf = new Float64Array(nall);
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      spin[k] = spinO[o];
      erad[k] = eradO[o];
    }
    const box = s.box;

    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const si = spin[i];
      const ei = erad[i];
      // electron wavefunction kinetic energy (not pairwise); fixed core (spin 2): KE(2s) + Coul(1s-1s) + Coul(2s-nuclei) + Pauli(2s)
      if (Math.abs(si) === 1 || si === 2) {
        let eke = 0, epauli = 0, ecoul = 0;
        let e1rforce = 0;
        const kin = kinElec(ei);
        eke += kin.e;
        e1rforce += kin.f;
        if (si === 2) {
          // the fixed core does not contribute to the radial virial: its radial forces are zeroed (pair_eff_cut.cpp)
          eke *= 2;
          const n1 = elecNucElec(q[i], 0.0, ei);
          ecoul += n1.e;
          ecoul += n1.e;
          ecoul += elecElecElec(0.0, ei, ei).e;
          epauli += pauliElecElec(0, 0.0, ei, ei).ep;
          e1rforce = 0;
        }
        // apply unit conversion factors
        eke *= hh;
        ecoul *= qqrd2e;
        epauli *= hh;
        e1rforce *= hh;
        const energy = eke + epauli + ecoul;
        erf[i] += e1rforce;
        this.tallyEff(pc, i, i, energy, 0.0, spin);
        if (this.pressureEvirials) this.tallyEff(pc, i, i, 0.0, e1rforce * ei, spin);
        pv[0] += eke;
        pv[1] += epauli;
        pv[2] += ecoul;
      }

      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const jj = list.neighbors[kk];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const rc = Math.sqrt(rsq);
        const tj = ti + type[j];
        if (!(rsq < cutsq[tj])) continue;
        const sj = spin[j];
        const ej = erad[j];
        const qi = q[i], qj = q[j];
        // pair_eff_cut.cpp: the spline cutoff of the pair (taper = the pair cutoff)
        const taper = Math.sqrt(cutsq[tj]);
        const dist = rc / taper;
        const spline = cutoff(dist);
        const dspline = dcutoff(dist) / taper;

        let ecoul = 0, epauli = 0, fpair = 0, sFpair = 0;
        let e1rforce = 0, e2rforce = 0, sE1 = 0, sE2 = 0;
        // ECP (pseudo-core) Pauli energy and force, kept apart as in pair_eff_cut.cpp (ecp_epauli, ecp_fpair)
        let ecpEpauli = 0, ecpFpair = 0;
        const absI = Math.abs(si), absJ = Math.abs(sj);
        if (si === 0 && sj === 0) {
          // nucleus (i) - nucleus (j) Coul interaction
          const r = elecNucNuc(qi * qj, rc);
          ecoul += r.e; fpair += r.f;
        } else if (si === 2 && sj === 0) {
          // fixed-core (i) - nucleus (j) nuclear Coul interaction (the electron term is added twice, as in LAMMPS)
          const qxq = qi * qj;
          const r = elecNucNuc(qxq, rc);
          ecoul += r.e; fpair += r.f;
          for (let t = 0; t < 2; t++) {
            const n = elecNucElec(qj, rc, ei);
            ecoul += n.e; fpair += n.f;
          }
        } else if (si === 0 && sj === 2) {
          // nucleus (i) - fixed-core (j) nuclear Coul interaction
          const qxq = qi * qj;
          const r = elecNucNuc(qxq, rc);
          ecoul += r.e; fpair += r.f;
          for (let t = 0; t < 2; t++) {
            const n = elecNucElec(qi, rc, ej);
            ecoul += n.e; fpair += n.f;
          }
        } else if (si === 0 && absJ === 1) {
          // nucleus (i) - electron (j) Coul interaction
          const n = elecNucElec(qi, rc, ej);
          ecoul += n.e; fpair += n.f; e1rforce += n.f1;
          const er = spline * qqrd2e * e1rforce;
          erf[j] += er;
          if (this.pressureEvirials) this.tallyEff(pc, j, j, 0.0, ej * er, spin);
        } else if (absI === 1 && sj === 0) {
          // electron (i) - nucleus (j) Coul interaction
          const n = elecNucElec(qj, rc, ei);
          ecoul += n.e; fpair += n.f; e1rforce += n.f1;
          const er = spline * qqrd2e * e1rforce;
          erf[i] += er;
          if (this.pressureEvirials) this.tallyEff(pc, i, i, 0.0, ei * er, spin);
        } else if (absI === 1 && absJ === 1) {
          // electron (i) - electron (j) interactions
          const ee = elecElecElec(rc, ei, ej);
          ecoul += ee.e; fpair += ee.f; e1rforce += ee.f1; e2rforce += ee.f2;
          const pa = pauliElecElec(si === sj ? 1 : 0, rc, ei, ej);
          epauli += pa.ep; sFpair -= pa.f; sE1 -= pa.f1; sE2 -= pa.f2;
          epauli *= hh;
          sFpair *= hh;
          const er1 = spline * (qqrd2e * e1rforce + hh * sE1);
          erf[i] += er1;
          const er2 = spline * (qqrd2e * e2rforce + hh * sE2);
          erf[j] += er2;
          if (this.pressureEvirials) this.tallyEff(pc, i, j, 0.0, ei * er1 + ej * er2, spin);
        } else if (si === 2 && absJ === 1) {
          // fixed-core (i) - electron (j) interactions: only the radial force of j is kept
          const n = elecNucElec(qi, rc, ej);
          ecoul += n.e; fpair += n.f; e2rforce += n.f1;
          for (let t = 0; t < 2; t++) {
            const ee = elecElecElec(rc, ei, ej);
            ecoul += ee.e; fpair += ee.f; e1rforce += ee.f1; e2rforce += ee.f2;
          }
          for (let t = 0; t < 2; t++) {
            const pa = pauliElecElec(t === 0 ? 0 : 1, rc, ei, ej);
            epauli += pa.ep; sFpair -= pa.f; sE1 -= pa.f1; sE2 -= pa.f2;
          }
          epauli *= hh;
          sFpair *= hh;
          const er = spline * (qqrd2e * e2rforce + hh * sE2);
          erf[j] += er;
          if (this.pressureEvirials) this.tallyEff(pc, j, j, 0.0, ej * er, spin);
        } else if (absI === 1 && sj === 2) {
          // electron (i) - fixed-core (j) interactions: only the radial force of i is kept
          const n = elecNucElec(qj, rc, ei);
          ecoul += n.e; fpair += n.f; e2rforce += n.f1;
          for (let t = 0; t < 2; t++) {
            const ee = elecElecElec(rc, ej, ei);
            ecoul += ee.e; fpair += ee.f; e1rforce += ee.f1; e2rforce += ee.f2;
          }
          for (let t = 0; t < 2; t++) {
            const pa = pauliElecElec(t === 0 ? 0 : 1, rc, ej, ei);
            epauli += pa.ep; sFpair -= pa.f; sE1 -= pa.f1; sE2 -= pa.f2;
          }
          epauli *= hh;
          sFpair *= hh;
          const er = spline * (qqrd2e * e2rforce + hh * sE2);
          erf[i] += er;
          if (this.pressureEvirials) this.tallyEff(pc, i, i, 0.0, ei * er, spin);
        } else if (si === 2 && sj === 2) {
          // fixed-core (i) - fixed-core (j) interactions: energies and the pair force only (no radial force)
          const qxq = qi * qj;
          const r = elecNucNuc(qxq, rc);
          ecoul += r.e; fpair += r.f;
          for (let t = 0; t < 2; t++) {
            const n = elecNucElec(qi, rc, ej);
            ecoul += n.e; fpair += n.f;
          }
          for (let t = 0; t < 2; t++) {
            const n = elecNucElec(qj, rc, ei);
            ecoul += n.e; fpair += n.f;
          }
          for (let t = 0; t < 4; t++) {
            const ee = elecElecElec(rc, ei, ej);
            ecoul += ee.e; fpair += ee.f;
          }
          for (let t = 0; t < 2; t++) {
            const pa = pauliElecElec(t === 0 ? 0 : 1, rc, ei, ej);
            epauli += pa.ep; sFpair -= pa.f;
          }
          epauli *= 2;
          sFpair *= 2;
          epauli *= hh;
          sFpair *= hh;
        }
        // pseudo-core nucleus (i) - nucleus (j) interaction (ElecCoreNuc, the core is a Gaussian of radius eradius[i])
        else if (si === 3 && sj === 0) {
          const r = elecCoreNuc(qi * qj, rc, ei);
          ecoul += r.e; fpair += r.f;
        }

        // nucleus (i) - pseudo-core nucleus (j) interaction
        else if (si === 0 && sj === 3) {
          const r = elecCoreNuc(qi * qj, rc, ej);
          ecoul += r.e; fpair += r.f;
        }

        // pseudo-core (i) - electron/fixed-core electrons (j) interactions
        else if (si === 3 && (absJ === 1 || sj === 2)) {
          const zc = this.ecpElement(type[i]);
          const A = this.pauliA[zc], B = this.pauliB[zc], C = this.pauliC[zc], D = this.pauliD[zc], Ep = this.pauliE[zc];
          const sType = D === 0 && Ep === 0;
          let ecpE2 = 0;
          if (absJ === 1) {
            const n = elecCoreElec(qi, rc, ei, ej);
            ecoul += n.e; fpair += n.f; e2rforce += n.f2;
            const pa = sType ? pauliCoreElec(rc, ej, A, B, C) : pauliCorePElec(rc, ej, A, B, C, D, Ep);
            ecpEpauli += pa.ep; ecpFpair += pa.f; ecpE2 += pa.f2;
          } else {
            // add the second s electron contribution of the fixed core (qxq = q[i] * q[j], as in pair_eff_cut.cpp)
            const nn = elecCoreNuc(qi * qj, rc, ej);
            ecoul += nn.e; fpair += nn.f;
            for (let t = 0; t < 2; t++) {
              const n = elecCoreElec(qi, rc, ei, ej);
              ecoul += n.e; fpair += n.f; e2rforce += n.f2;
            }
            for (let t = 0; t < 2; t++) {
              const pa = sType ? pauliCoreElec(rc, ej, A, B, C) : pauliCorePElec(rc, ej, A, B, C, D, Ep);
              ecpEpauli += pa.ep; ecpFpair += pa.f; ecpE2 += pa.f2;
            }
          }
          ecpEpauli *= this.h2e;
          ecpFpair *= this.h2e;
          // only the radial force of the electron j is kept
          e2rforce = spline * (qqrd2e * e2rforce + this.h2e * ecpE2);
          erf[j] += e2rforce;
          if (this.pressureEvirials) this.tallyEff(pc, j, j, 0.0, ej * e2rforce, spin);
        }

        // electron/fixed-core electrons (i) - pseudo-core (j) interactions
        else if ((absI === 1 || si === 2) && sj === 3) {
          const zc = this.ecpElement(type[j]);
          const A = this.pauliA[zc], B = this.pauliB[zc], C = this.pauliC[zc], D = this.pauliD[zc], Ep = this.pauliE[zc];
          const sType = D === 0 && Ep === 0;
          let ecpE1 = 0;
          if (absI === 1) {
            const n = elecCoreElec(qj, rc, ej, ei);
            ecoul += n.e; fpair += n.f; e1rforce += n.f2;
            const pa = sType ? pauliCoreElec(rc, ei, A, B, C) : pauliCorePElec(rc, ei, A, B, C, D, Ep);
            ecpEpauli += pa.ep; ecpFpair += pa.f; ecpE1 += pa.f2;
          } else {
            const nn = elecCoreNuc(qi * qj, rc, ei);
            ecoul += nn.e; fpair += nn.f;
            for (let t = 0; t < 2; t++) {
              const n = elecCoreElec(qj, rc, ej, ei);
              ecoul += n.e; fpair += n.f; e1rforce += n.f2;
            }
            for (let t = 0; t < 2; t++) {
              const pa = sType ? pauliCoreElec(rc, ei, A, B, C) : pauliCorePElec(rc, ei, A, B, C, D, Ep);
              ecpEpauli += pa.ep; ecpFpair += pa.f; ecpE1 += pa.f2;
            }
          }
          ecpEpauli *= this.h2e;
          ecpFpair *= this.h2e;
          // only the radial force of the electron i is kept
          e1rforce = spline * (qqrd2e * e1rforce + this.h2e * ecpE1);
          erf[i] += e1rforce;
          if (this.pressureEvirials) this.tallyEff(pc, i, i, 0.0, ei * e1rforce, spin);
        }

        // pseudo-core (i) - pseudo-core (j) interactions
        else if (si === 3 && sj === 3) {
          const r = elecCoreCore(qi * qj, rc, ei, ej);
          ecoul += r.e; fpair += r.f;
        } else {
          // espin 4 is refused in initStyle
          throw new StyleError('pair_style eff/cut: pseudo-core (espin 4) pairs are not ported yet');
        }

        // Apply Coulomb conversion factor for all cases
        ecoul *= qqrd2e;
        fpair *= qqrd2e;
        // sum up energy and force contributions
        // the ECP Pauli terms join the pair sums (epauli += ecp_epauli; fpair += ecp_fpair in pair_eff_cut.cpp)
        epauli += ecpEpauli;
        const energy0 = ecoul + epauli;
        fpair = fpair + sFpair + ecpFpair;
        // apply the cutoff spline
        fpair = fpair * spline - energy0 * dspline;
        const energy = spline * energy0;
        // smallRForce (pair_eff_inline.h) without a tuple per pair; the rc <= 1e-6 branch is the rare one
        let fx: number, fy: number, fz: number;
        if (rc > 0.000001) {
          const ff = fpair / rc;
          fx = ff * dx; fy = ff * dy; fz = ff * dz;
        } else [fx, fy, fz] = smallRForce(dx, dy, dz, rc, fpair);
        f[3 * i] += fx;
        f[3 * i + 1] += fy;
        f[3 * i + 2] += fz;
        f[3 * j] -= fx;
        f[3 * j + 1] -= fy;
        f[3 * j + 2] -= fz;
        this.tallyXYZ(pc, i, j, energy, fx, fy, fz, dx, dy, dz);
        pv[1] += spline * epauli;
        pv[2] += spline * ecoul;
      }

      // limit electron size (stiffness) for periodic systems, to max = half the box size (limit/eradius)
      if (Math.abs(si) === 1 && this.limitEradius) {
        const kfactor = hh * 1.0;
        const periodic = box.periodic[0] || box.periodic[1] || box.periodic[2];
        if (periodic) {
          const lx = box.hi[0] - box.lo[0], ly = box.hi[1] - box.lo[1], lz = box.hi[2] - box.lo[2];
          const halfBoxLength = 0.5 * Math.min(lx, Math.min(ly, lz));
          if (ei > halfBoxLength) {
            const dr = ei - halfBoxLength;
            const errestrain = 0.5 * kfactor * dr * dr;
            const e1 = -kfactor * dr;
            pv[3] += errestrain;
            erf[i] += e1;
            this.tallyEff(pc, i, i, errestrain, 0.0, spin);
            if (this.pressureEvirials) this.tallyEff(pc, i, i, 0.0, ei * e1, spin);
          }
        }
      }
    }

    // the radial forces of the owned electrons are the sum over their owned and ghost copies (reverse communication)
    erfO.fill(0);
    nb.reverseSum(erf, 1, erfO);
    // virial_eff_compute: the radial electron virial, summed over owned and ghost atoms
    if (this.pressureEvirials) {
      for (let k = 0; k < nall; k++) {
        if (spin[k]) {
          const ev = erf[k] * erad[k] / 3;
          pc.acc.virial[0] += ev;
          pc.acc.virial[1] += ev;
          pc.acc.virial[2] += ev;
        }
      }
    }
  }
}
