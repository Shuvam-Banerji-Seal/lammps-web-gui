/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/pair_eff_cut.cpp and
 * pair_eff_cut.h (class PairEffCut, pair_style eff/cut: the electron force field pair style with the
 * limit/eradius and pressure/evirials keywords, the fixed-core (espin 2) and electron (espin +-1) interactions,
 * the radial electron forces erforce and the radial virial, and the pvector of compute pair: KE, Pauli, Coulomb
 * and the radius restraint). Contributing author (from the source): Andres Jaramillo-Botero (Caltech). The
 * pseudo-core (ECP, espin 3) terms and the pair_coeff ECP parameters are not ported (a StyleError names them).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { Pair, PairParams, StyleError, mixDistance, type PairCompute, type PairMinPerAtom, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import { ELECTRON_RADIUS, ELECTRON_RFORCE, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';
import type { SimState } from '../../engine/types';
import { kinElec, elecNucNuc, elecNucElec, elecElecElec, pauliElecElec, smallRForce, cutoff, dcutoff } from './math';

/** units real: qqr2e and the Hartree/Bohr conversions used by pair eff/cut (initStyle, "Check your units" otherwise). */
const QQR2E_REAL = 332.06371;
const QQR2E_ELECTRON = 1.0;
const HHMSS2E_REAL = 175.72044219620075;

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
  private p!: PairParams;
  /** compute pair vector: [KE, Pauli, ecoul, radial restraint] (pair_eff_cut.cpp pvector). */
  private pvector = new Float64Array(4);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length < 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.limitEradius = false;
    this.pressureEvirials = false;
    let iarg = 1;
    while (iarg < args.length) {
      const w = args[iarg];
      if (w === 'limit/eradius') { this.limitEradius = true; iarg += 1; }
      else if (w === 'pressure/evirials') { this.pressureEvirials = true; iarg += 1; }
      else if (w === 'ecp') throw new StyleError('pair_style eff/cut ecp (pseudo-core electrons, espin 3) is not ported yet');
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

  coeff(args: string[], _ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 2 || args.length > 3) throw new StyleError('Incorrect args for pair coefficients');
    // pair_coeff I J with I, J numbers are the ECP parameter lines (pair_eff_cut.cpp coeff, else branch)
    if (args[0] !== '*' && args[1] !== '*') throw new StyleError('pair_coeff for electron types (ECP parameters s and p) is not ported yet; use pair_coeff * *');
    const cutOne = args.length === 3 ? parseNum(args[2], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [cutOne]);
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
    if (s.units.qqr2e === QQR2E_REAL) this.hhmss2e = HHMSS2E_REAL;
    else if (s.units.qqr2e === QQR2E_ELECTRON) this.hhmss2e = 1;
    else throw new StyleError('Check your units');
    const spin = electronArray(s, ELECTRON_SPIN);
    for (let i = 0; i < s.n; i++) {
      if (spin[i] === 3 || spin[i] === 4) throw new StyleError('pair_style eff/cut: atoms with espin 3 or 4 (pseudo-core, ECP) are not ported yet');
    }
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
            x[i] = Math.log(er[i]);
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
    const a = pc.acc;
    a.ecoul += energy;
    const v = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
    for (let c = 0; c < 6; c++) a.virial[c] += v[c];
    if (pc.eatom) { pc.eatom[i] += 0.5 * energy; pc.eatom[j] += 0.5 * energy; }
    if (pc.vatom) {
      const va = pc.vatom;
      for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * v[c]; va[6 * j + c] += 0.5 * v[c]; }
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
        } else {
          // espin 3 or 4 (pseudo-cores) are refused in initStyle
          throw new StyleError('pair_style eff/cut: pseudo-core (espin 3 or 4) pairs are not ported yet');
        }

        // Apply Coulomb conversion factor for all cases
        ecoul *= qqrd2e;
        fpair *= qqrd2e;
        // sum up energy and force contributions
        const energy0 = ecoul + epauli;
        fpair = fpair + sFpair;
        // apply the cutoff spline
        fpair = fpair * spline - energy0 * dspline;
        const energy = spline * energy0;
        const [fx, fy, fz] = smallRForce(dx, dy, dz, rc, fpair);
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
