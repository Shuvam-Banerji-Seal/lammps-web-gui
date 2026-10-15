/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SMTBQ/pair_smtbq.cpp and src/SMTBQ/pair_smtbq.h (PairSMTBQ: settings, coeff, init_style, init_one,
 * compute, self, qfo_self, potqeq, pot_ES, rep_OO, Attr_OO, repulsive, attractive, f_att, pot_COV, QForce_charge,
 * Charge), at commit b332f28. Not ported: Init_charge and CheckEnergyVSForce, which the C++ does not call (the
 * call of Init_charge in Charge is commented out, and CheckEnergyVSForce is a debugging routine). Authors credited in the C++ file headers: N. Salles, O. Politano,
 * E. Amzallag, R. Tetot (SMTBQ) and E. Maras (Aalto University). Copyright (2015) Universite de Bourgogne
 * (SMTBQ) and Sandia Corporation (LAMMPS). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org (pair_smtbq, doc/src/pair_smtbq.rst at b332f28) says: "The SMTB-Q potential is a variable
 * charge potential. The equilibrium charge on each atom is calculated by the electronegativity equalization (QEq)
 * method." and "Note that atom type 1 must always correspond to oxygen atoms." (the element order of pair_coeff
 * is therefore checked against the potential file, see coeff). The file layout matters: "the lines beginning with
 * '#' are comments (note that the number of comment lines matter)" (potfile.ts reads it line by line).
 *
 * Serial path (one process), as the C++ runs on one process. Refused with a StyleError: the writeenerg and
 * writepot output of the potential file (true: the C++ writes Energy_component.txt / Electroneg_component.txt),
 * pair_modify shift and tail, compute stress/atom, NULL element names. The C++ Verbose output is not printed here:
 * read_file reads the file's "Verbose" line one line off (the line after the QEq block is skipped), so the C++
 * Bavard value is "Verbose" for the shipped files and its debug lines are printed; they are diagnostics, not results.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { readSmtbqFile, type SmtbqFile, type SmtbqInter, type SmtbqParam } from './potfile';
import { SlaterIntegrals } from './slater';
import { buildTables, fcoup2, fcoupured, fcoupure, intfcoup2, type SmtbqTables } from './tables';
import { QEqGroups, type GroupContext } from './groups';

/** Cut-off width of the oxygen coordination switch (the 0.15 of potqeq and pot_ES). */
const OX_SWITCH = 0.15;

/** PairSMTBQ: second-moment tight-binding potential with the Coulomb charge equilibration of the oxide. */
export class PairSMTBQ extends Pair {
  readonly name = 'smtbq';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private file: SmtbqFile | null = null;
  private tables: SmtbqTables | null = null;
  private readonly slater = new SlaterIntegrals();
  private readonly groups = new QEqGroups();
  /** map[type] = element index (params index); -1 for an unmapped type. */
  private map = new Int32Array(0);
  /** QInit value (native: a member that the initial charge set may raise). */
  private QOxInit = 0;
  private cutmax = 0;
  private ncov = 0;
  private Vself = 0;
  private dc2 = new Float64Array(0);
  /** Per-atom arrays on owned + ghost atoms (nall), resized with nall. */
  private nmax = 0;
  private sbcov = new Float64Array(0);
  private coord = new Float64Array(0);
  /** intfcoup2(coord, coordOxBulk, 0.15) of every owned and ghost atom, refreshed with coord (loop 0 of qforce). */
  private ifc = new Float64Array(0);
  /** 1 for an element that is oxygen (the element index of params). */
  private isOEl = new Uint8Array(0);
  /**
   * pot_ES of every neighbor pair (by position in the full list) and whether the pair is inside cutmax. The
   * positions and the coordinations do not change between the charge iterations of one compute, so loop 0 of
   * qforce fills these and the later iterations only re-weight them by the new charges.
   */
  private esVal = new Float64Array(0);
  private esOk = new Uint8Array(0);
  private sbmet = new Float64Array(0);
  private ecov = new Float64Array(0);
  private potmad = new Float64Array(0);
  private potself = new Float64Array(0);
  private potcov = new Float64Array(0);
  private qf = new Float64Array(0);
  private q1 = new Float64Array(0);
  private q2 = new Float64Array(0);
  /** The pair's current context (read by the charge routines between compute calls). */
  private pc: PairCompute | null = null;
  /** Outputs of potqeq, pot_ES, repulsive, attractive: avoid allocation in the pair loops. */
  private fOut = 0;
  private eOut = 0;

  settings(args: string[]): void {
    if (args.length > 0) throw new StyleError('Illegal pair_style smtbq command: no arguments are accepted');
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 4 || args[0] !== '*' || args[1] !== '*') {
      throw new StyleError('Incorrect args for pair coefficients: usage pair_coeff * * file.smtbq element1 ... elementN');
    }
    const names = args.slice(3);
    if (names.length !== this.ntypes) {
      throw new StyleError(`Number of element to type mappings does not match number of atom types (${names.length} elements, ${this.ntypes} types)`);
    }
    if (names.includes('NULL')) throw new StyleError('pair_style smtbq does not support NULL in pair_coeff (every atom type is an element)');
    const file = readSmtbqFile(ctx.readFile(args[2]));
    if (file.params.length !== names.length || file.params.some((p, k) => p.nom !== names[k])) {
      const order = file.params.map((p: SmtbqParam) => p.nom).join(' ');
      throw new StyleError(`pair_coeff element names must follow the element order of the potential file (${order})`);
    }
    if (file.writeenerg === 'true') {
      throw new StyleError("pair_style smtbq: the potential file asks for the energy component output ('Print.Energy.components' true), which the engine does not write");
    }
    if (file.writepot === 'true') {
      throw new StyleError("pair_style smtbq: the potential file asks for the electronegativity output ('Print.electroneg...components' true), which the engine does not write");
    }
    this.file = file;
    this.cutmax = file.cutmax;
    this.ncov = file.ncov;
    this.QOxInit = file.QOxInit;
    this.isOEl = Uint8Array.from(file.params.map((p) => (p.nom === 'O' ? 1 : 0)));
    this.map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) this.map[t] = t - 1;
    this.tables = buildTables(file, this.slater);
    this.Vself = this.tables.Vself;
    this.dc2 = new Float64Array(file.maxintparam + 1);
    for (let m = 0; m <= file.maxintparam; m++) this.dc2[m] = file.intparams[m].dc2;
    ctx.log('Pair SMTBQ: generating Coulomb integral lookup table ...');
  }

  requireFile(): void {
    if (!this.file || !this.tables) throw new StyleError('pair_style smtbq needs a pair_coeff command with an SMTBQ potential file');
  }

  initStyle(): void {
    this.requireFile();
    if (this.shift || this.tail) throw new StyleError('pair_modify shift and tail are not supported for pair style smtbq');
  }

  initOne(): number {
    return this.cutmax;
  }

  /** Resizes the per-atom arrays to nall (owned + ghost). */
  private ensure(nall: number): void {
    if (this.nmax >= nall) return;
    this.nmax = nall;
    this.sbcov = new Float64Array(nall);
    this.coord = new Float64Array(nall);
    this.ifc = new Float64Array(nall);
    this.sbmet = new Float64Array(nall);
    this.ecov = new Float64Array(nall);
    this.potmad = new Float64Array(nall);
    this.potself = new Float64Array(nall);
    this.potcov = new Float64Array(nall);
    this.qf = new Float64Array(nall);
    this.q1 = new Float64Array(nall);
    this.q2 = new Float64Array(nall);
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style smtbq needs a full neighbor list');
    if (!this.file || !this.tables) throw new StyleError('pair_style smtbq needs a pair_coeff command first');
    if (pc.vatom) throw new StyleError('compute stress/atom with pair_style smtbq is not supported by the browser engine');
    const file = this.file;
    const tb = this.tables;
    const nb = pc.nb;
    const s = pc.s;
    this.pc = pc;
    this.ensure(pc.nall);
    this.groups.reserve(pc.nall);
    const x = pc.x, f = pc.f, type = pc.type, qAll = pc.q;
    const tag = s.id;
    const acc = pc.acc;
    const eatom = pc.eatom;
    const { firstneigh, numneigh, neighbors } = list;
    const inum = list.inum;
    const sbcov = this.sbcov, sbmet = this.sbmet, ecov = this.ecov;

    // tally of a half pair (ev_tally) and of a single atom's self term (ev_tally_full with 2E: E)
    const tallyPair = (i: number, j: number, e: number): void => {
      acc.evdwl += e;
      if (eatom) { eatom[i] += 0.5 * e; eatom[j] += 0.5 * e; }
    };
    const tallySelf = (i: number, e: number): void => {
      acc.evdwl += e;
      if (eatom) eatom[i] += e;
    };

    const step = s.step;
    const Qstep = file.Qstep;
    if (step === 0 || (Qstep > 0 && step % Qstep === 0)) this.charge(pc);
    // this is necessary to get sbcov or sbmet table in order to calculate the covalent or metal bonding
    if (Qstep === 0 || step % Qstep !== 0) this.qforce(pc, 0);
    // charges communication
    nb.refreshCharges(s);

    for (let i = 0; i < inum; i++) {
      const itype = this.map[type[i]];
      const iq = qAll[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itag = tag[i];

      // self energy of i
      const Eself = this.self(file.params[itype], iq);
      tallySelf(i, Eself);

      // N-body energy of i (covalent)
      const dq = Math.abs(file.params[itype].qform) - Math.abs(iq);
      const dqcov = dq * (2.0 * this.ncov / file.params[itype].sto - dq);
      ecov[i] = -Math.sqrt(sbcov[i] * dqcov + sbmet[i]);
      tallySelf(i, ecov[i]);

      // Coulombian self term
      tallySelf(i, this.Vself * iq * iq);

      for (let jj = 0; jj < numneigh[i]; jj++) {
        const j = neighbors[firstneigh[i] + jj] & NEIGHMASK;
        const jtype = this.map[type[j]];
        const jtag = tag[j];
        const jq = qAll[j];

        // each pair once: the parity of the tags (periodic images with equal tags: the coordinates)
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < x[3 * i + 2]) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }

        const m = file.intype[itype * file.ntypes + jtype];
        const delx = xtmp - x[3 * j];
        const dely = ytmp - x[3 * j + 1];
        const delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (Math.sqrt(rsq) > this.cutmax) continue;

        // Coulombian energy and force
        this.potqeq(itype, jtype, i, j, iq, jq, rsq);
        let fpair = this.fOut;
        tallyPair(i, j, this.eOut);
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;

        if (m === 0) continue;
        const it: SmtbqInter = file.intparams[m];

        // pure two-body oxygen-oxygen repulsion (and attraction)
        if (it.typepot === 'buck' || it.typepot === 'buckPlusAttr') {
          this.repOO(it, rsq);
          fpair = this.fOut;
          tallyPair(i, j, this.eOut);
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
        }
        if (it.typepot === 'buckPlusAttr') {
          this.attrOO(it, rsq);
          fpair = this.fOut;
          tallyPair(i, j, this.eOut);
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
        }

        if (it.typepot !== 'second_moment') continue;
        if (Math.sqrt(rsq) > it.dc2) continue;

        // repulsion: energy (counted twice in the C++ tally) and force
        this.repulsive(it, rsq);
        fpair = this.fOut;
        tallyPair(i, j, 2.0 * this.eOut);
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;

        // attraction: force only
        fpair = this.fAtt(it, i, j, rsq);
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
      }
    }
  }

  /** self (pair_smtbq.cpp): the on-site electrostatic energy of a charge. */
  private self(param: SmtbqParam, qi: number): number {
    const s1 = param.chi, s2 = param.dj;
    return qi * (s1 + 0.5 * qi * s2);
  }

  /** qfo_self: the charge force of the on-site term. */
  private qfoSelf(param: SmtbqParam, qi: number): number {
    return param.chi + qi * param.dj;
  }

  /**
   * potqeq (pair_smtbq.cpp): the Coulombian pair energy (eOut) and force (fOut) of i and j, with the oxygen
   * coordination interpolation between the bulk, surface and bulk-to-bulk tables.
   */
  private potqeq(itype: number, jtype: number, i: number, j: number, qi: number, qj: number, rsq: number): void {
    const tb = this.tables!;
    const file = this.file!;
    const coord = this.coord;
    const m = tb.coultype[itype * file.ntypes + jtype];
    const nt = tb.nntype;
    const r = rsq;
    const sds = r / tb.ds;
    const l = Math.trunc(sds);
    const xi = sds - l;
    const fafb = tb.fafb, dfafb = tb.dfafb;

    let iCoord = coord[i];
    let jCoord = coord[j];
    let iIntfCoup2 = this.ifc[i];
    let jIntfCoup2 = this.ifc[j];

    // energies interpolation
    const a0 = fafb[l * nt + m], a1 = fafb[(l + 1) * nt + m], a2 = fafb[(l + 2) * nt + m];
    let t1 = a0 + (a1 - a0) * xi;
    let t2 = a1 + (a2 - a1) * (xi - 1.0);
    const engBulk = qi * qj * (t1 + (t2 - t1) * xi / 2.0);
    let eng = engBulk;

    // forces interpolation
    const d0 = dfafb[l * nt + m], d1 = dfafb[(l + 1) * nt + m], d2 = dfafb[(l + 2) * nt + m];
    t1 = d0 + (d1 - d0) * xi;
    t2 = d1 + (d2 - d1) * (xi - 1);
    let fforce = -qi * qj * (t1 + (t2 - t1) * xi / 2.0);

    const isOi = this.isOEl[itype] === 1;
    const isOj = this.isOEl[jtype] === 1;
    if (isOi || isOj) {
      const r2 = tb.fafbTiOxSurf;
      if (isOi && isOj) {
        // between two oxygens
        const engSurf = qi * qj * this.interp(tb.fafbOxOxSurf, l, xi);
        const engBB = qi * qj * this.interp(tb.fafbOxOxBB, l, xi);
        eng = engBulk + (iCoord + jCoord - 2 * file.coordOxBulk) / (2 * (file.coordOxBB - file.coordOxBulk)) * (engBB - engBulk)
          + (iIntfCoup2 + jIntfCoup2) * ((engBulk - engSurf) / (2 * (file.coordOxBulk - file.coordOxSurf))
            - (engBB - engBulk) / (2 * (file.coordOxBB - file.coordOxBulk)));

        // interpolation of the forces
        const fforceBulk = fforce;
        const fforceSurf = -qi * qj * this.interpD(tb.dfafbOxOxSurf, l, xi);
        const fforceBB = -qi * qj * this.interpD(tb.dfafbOxOxBB, l, xi);
        fforce = fforceBulk + (iCoord + jCoord - 2 * file.coordOxBulk) / (2 * (file.coordOxBB - file.coordOxBulk)) * (fforceBB - fforceBulk)
          + (iIntfCoup2 + jIntfCoup2) * ((fforceBulk - fforceSurf) / (2 * (file.coordOxBulk - file.coordOxSurf))
            - (fforceBB - fforceBulk) / (2 * (file.coordOxBB - file.coordOxBulk)));
      } else {
        // between metal and oxygen
        const engSurf = qi * qj * this.interp(r2, l, xi);
        const engBB = qi * qj * this.interp(tb.fafbTiOxBB, l, xi);
        if (isOj) {  // the atom j is an oxygen
          iIntfCoup2 = jIntfCoup2;
          iCoord = jCoord;
        }
        eng = engBulk + (engBulk - engSurf) / (file.coordOxBulk - file.coordOxSurf) * iIntfCoup2
          + (engBB - engBulk) / (file.coordOxBB - file.coordOxBulk) * (iCoord - file.coordOxBulk - iIntfCoup2);

        // interpolation of the forces
        const fforceBulk = fforce;
        const fforceSurf = -qi * qj * this.interpD(tb.dfafbTiOxSurf, l, xi);
        const fforceBB = -qi * qj * this.interpD(tb.dfafbTiOxBB, l, xi);
        const dcoordloc = fcoupured(Math.sqrt(r), file.r1Coord, file.r2Coord);
        const dcoupureloc = fcoupured(iCoord, file.coordOxSurf, file.coordOxBulk);
        const dIntfcoup2loc = fcoup2(iCoord, file.coordOxBulk, OX_SWITCH) * dcoupureloc;
        fforce = fforceBulk + 1 / (file.coordOxBulk - file.coordOxSurf) * ((fforceBulk - fforceSurf) * iIntfCoup2
          - (engBulk - engSurf) * dIntfcoup2loc)
          + 1 / (file.coordOxBB - file.coordOxBulk) * ((fforceBB - fforceBulk) * (iCoord - file.coordOxBulk - iIntfCoup2)
            - (engBB - engBulk) * (dcoordloc - dIntfcoup2loc));
      }
    }
    this.fOut = fforce;
    this.eOut = eng;
  }

  /** The table interpolation of an energy-type table (t1 + (t2 - t1) xi / 2), at index l. */
  private interp(tab: Float64Array, l: number, xi: number): number {
    const t1 = tab[l] + (tab[l + 1] - tab[l]) * xi;
    const t2 = tab[l + 1] + (tab[l + 2] - tab[l + 1]) * (xi - 1.0);
    return t1 + (t2 - t1) * xi / 2.0;
  }

  /** The table interpolation of a force-type table (the same form as interp, for the derivative tables). */
  private interpD(tab: Float64Array, l: number, xi: number): number {
    const t1 = tab[l] + (tab[l + 1] - tab[l]) * xi;
    const t2 = tab[l + 1] + (tab[l + 2] - tab[l + 1]) * (xi - 1);
    return t1 + (t2 - t1) * xi / 2.0;
  }

  /** pot_ES: the Coulombian (1/r with the electrostatic table) energy of i and j, for the Madelung sums. */
  private potES(itype: number, jtype: number, i: number, j: number, rsq: number): number {
    const tb = this.tables!;
    const file = this.file!;
    const coord = this.coord;
    const m = tb.coultype[itype * file.ntypes + jtype];
    const nt = tb.nntype;
    const r = rsq;
    const sds = r / tb.ds;
    const l = Math.trunc(sds);
    const xi = sds - l;
    let iCoord = coord[i];
    let jCoord = coord[j];
    let iIntfCoup2 = this.ifc[i];
    const jIntfCoup2 = this.ifc[j];

    const a0 = tb.fafb[l * nt + m], a1 = tb.fafb[(l + 1) * nt + m], a2 = tb.fafb[(l + 2) * nt + m];
    const t1 = a0 + (a1 - a0) * xi;
    const t2 = a1 + (a2 - a1) * (xi - 1.0);
    let eng = t1 + (t2 - t1) * xi / 2.0;
    const engBulk = eng;

    const isOi = this.isOEl[itype] === 1;
    const isOj = this.isOEl[jtype] === 1;
    if (isOi || isOj) {
      if (isOi && isOj) {
        const engSurf = this.interp(tb.fafbOxOxSurf, l, xi);
        const engBB = this.interp(tb.fafbOxOxBB, l, xi);
        eng = engBulk + (iCoord + jCoord - 2 * file.coordOxBulk) / (2 * (file.coordOxBB - file.coordOxBulk)) * (engBB - engBulk)
          + (iIntfCoup2 + jIntfCoup2) * ((engBulk - engSurf) / (2 * (file.coordOxBulk - file.coordOxSurf))
            - (engBB - engBulk) / (2 * (file.coordOxBB - file.coordOxBulk)));
      } else {
        const engSurf = this.interp(tb.fafbTiOxSurf, l, xi);
        const engBB = this.interp(tb.fafbTiOxBB, l, xi);
        if (isOj) {  // the atom j is an oxygen
          iIntfCoup2 = jIntfCoup2;
          iCoord = jCoord;
        }
        eng = engBulk + (engBulk - engSurf) / (file.coordOxBulk - file.coordOxSurf) * iIntfCoup2
          + (engBB - engBulk) / (file.coordOxBB - file.coordOxBulk) * (iCoord - file.coordOxBulk - iIntfCoup2);
      }
    }
    return eng;
  }

  /** rep_OO: the Buckingham repulsion of the oxygen pair (fOut force, eOut energy). */
  private repOO(it: SmtbqInter, rsq: number): void {
    const A = it.abuck;
    const rho = it.rhobuck;
    const r = Math.sqrt(rsq);
    const tmp = -r / rho;
    const tmpExp = Math.exp(tmp);
    this.eOut = A * tmpExp;
    this.fOut = A / rho * tmpExp / r;
  }

  /** Attr_OO: the oxygen-oxygen attraction with its cut-off (fOut force, eOut energy). */
  private attrOO(it: SmtbqInter, rsq: number): void {
    const aOO = it.aOO, bOO = it.bOO, r1OO = it.r1OO, r2OO = it.r2OO;
    const r = Math.sqrt(rsq);
    const tmpExp = Math.exp(bOO * r);
    this.eOut = aOO * tmpExp * fcoupure(r, r1OO, r2OO);
    this.fOut = -(aOO * bOO * tmpExp * fcoupure(r, r1OO, r2OO) + aOO * tmpExp * fcoupured(r, r1OO, r2OO)) / r;
  }

  /** repulsive: the second-moment repulsion of the pair (fOut force, eOut energy); mode oxide or metal. */
  private repulsive(it: SmtbqInter, rsq: number): void {
    const tb = this.tables!;
    const sm = it.intsm;
    const cols = tb.smCols;
    const rrcs = it.dc2;
    this.fOut = 0;
    this.eOut = 0;
    if (Math.sqrt(rsq) > rrcs) return;
    const sds = rsq / tb.ds;
    const l = Math.trunc(sds);
    const xi = sds - l;
    const t1 = tb.tabsmr[l * cols + sm] + (tb.tabsmr[(l + 1) * cols + sm] - tb.tabsmr[l * cols + sm]) * xi;
    const t2 = tb.tabsmr[(l + 1) * cols + sm] + (tb.tabsmr[(l + 2) * cols + sm] - tb.tabsmr[(l + 1) * cols + sm]) * (xi - 1.0);
    const dt1 = tb.dtabsmr[l * cols + sm] + (tb.dtabsmr[(l + 1) * cols + sm] - tb.dtabsmr[l * cols + sm]) * xi;
    const dt2 = tb.dtabsmr[(l + 1) * cols + sm] + (tb.dtabsmr[(l + 2) * cols + sm] - tb.dtabsmr[(l + 1) * cols + sm]) * (xi - 1.0);
    if (it.mode === 'oxide') {
      this.fOut = -2.0 * (dt1 + (dt2 - dt1) * xi / 2.0);
      this.eOut = t1 + (t2 - t1) * xi / 2.0;
    } else if (it.mode === 'metal') {
      const sweet = 1.0;
      this.fOut = -2.0 * (dt1 + (dt2 - dt1) * xi / 2.0) * sweet;
      this.eOut = (t1 + (t2 - t1) * xi / 2.0) * sweet;
    }
  }

  /**
   * attractive: the bond-order sums of atom i (sbcov and coord for an oxide, sbmet for a metal). Writes the
   * per-atom sums; the pair energy is not involved.
   */
  private attractive(it: SmtbqInter, rsq: number, i: number): void {
    const file = this.file!;
    const tb = this.tables!;
    const sm = it.intsm;
    const cols = tb.smCols;
    const rrcs = it.dc2;
    const itype = this.map[this.pc!.type[i]];
    const r = rsq;
    if (Math.sqrt(r) > rrcs) return;
    const sds = r / tb.ds;
    const l = Math.trunc(sds);
    const xi = sds - l;
    const t1 = tb.tabsmb[l * cols + sm] + (tb.tabsmb[(l + 1) * cols + sm] - tb.tabsmb[l * cols + sm]) * xi;
    const t2 = tb.tabsmb[(l + 1) * cols + sm] + (tb.tabsmb[(l + 2) * cols + sm] - tb.tabsmb[(l + 1) * cols + sm]) * (xi - 1.0);
    if (it.mode === 'oxide') {
      const mu = 0.5 * (Math.sqrt(file.params[1].sto) + Math.sqrt(file.params[0].sto));
      this.sbcov[i] += (t1 + (t2 - t1) * xi / 2.0) * file.params[itype].sto * mu * mu;
      if (Math.sqrt(r) < file.r1Coord) {
        this.coord[i] += 1;
      } else if (Math.sqrt(r) < file.r2Coord) {
        this.coord[i] += fcoupure(Math.sqrt(r), file.r1Coord, file.r2Coord);
      }
    } else if (it.mode === 'metal') {
      const sweet = 1.0;
      this.sbmet[i] += (t1 + (t2 - t1) * xi / 2.0) * sweet;
    }
  }

  /** f_att: the force of the bond-order (covalent or metal) term on the pair (returns the pair force). */
  private fAtt(it: SmtbqInter, i: number, j: number, rsq: number): number {
    const file = this.file!;
    const tb = this.tables!;
    const pc = this.pc!;
    const sm = it.intsm;
    const cols = tb.smCols;
    const itype = this.map[pc.type[i]];
    const jtype = this.map[pc.type[j]];
    const iq = pc.q[i];
    const jq = pc.q[j];
    const r = rsq;
    const sds = r / tb.ds;
    const l = Math.trunc(sds);
    const xi = sds - l;
    const dt1 = tb.dtabsmb[l * cols + sm] + (tb.dtabsmb[(l + 1) * cols + sm] - tb.dtabsmb[l * cols + sm]) * xi;
    const dt2 = tb.dtabsmb[(l + 1) * cols + sm] + (tb.dtabsmb[(l + 2) * cols + sm] - tb.dtabsmb[(l + 1) * cols + sm]) * (xi - 1.0);

    let dq = Math.abs(file.params[itype].qform) - Math.abs(iq);
    const dqcovi = dq * (2.0 * this.ncov / file.params[itype].sto - dq);
    dq = Math.abs(file.params[jtype].qform) - Math.abs(jq);
    const dqcovj = dq * (2.0 * this.ncov / file.params[jtype].sto - dq);

    const sbcov = this.sbcov, sbmet = this.sbmet;
    if (it.mode === 'oxide') {
      const mu = 0.5 * (Math.sqrt(file.params[1].sto) + Math.sqrt(file.params[0].sto));
      const fcovIj = (dt1 + (dt2 - dt1) * xi / 2.0) * dqcovi * file.params[itype].sto * mu * mu;
      const fcovJi = (dt1 + (dt2 - dt1) * xi / 2.0) * dqcovj * file.params[jtype].sto * mu * mu;
      return 0.5 * (fcovIj / Math.sqrt(sbcov[i] * dqcovi + sbmet[i]) + fcovJi / Math.sqrt(sbcov[j] * dqcovj + sbmet[j]));
    }
    // metal
    const sweet = 1.0;
    const fcovIj = (dt1 + (dt2 - dt1) * xi / 2.0) * sweet;
    return 0.5 * fcovIj * (1.0 / Math.sqrt(sbcov[i] * dqcovi + sbmet[i]) + 1.0 / Math.sqrt(sbcov[j] * dqcovj + sbmet[j]));
  }

  /** pot_COV: the charge force of the covalent term of atom i (returns it). */
  private potCov(param: SmtbqParam, i: number): number {
    const pc = this.pc!;
    const qform = param.qform;
    const sto = param.sto;
    const sign = qform / Math.abs(qform);
    const iq = pc.q[i];
    const dq = Math.abs(qform) - Math.abs(iq);
    const DQ = dq * (2.0 * this.ncov / sto - dq);
    if (Math.abs(iq) < 1.0e-7 || Math.abs(this.sbcov[i]) < 1.0e-7) return 0.0;
    return sign * this.sbcov[i] / Math.sqrt(this.sbcov[i] * DQ + this.sbmet[i]) * (this.ncov / sto - dq);
  }

  /**
   * QForce_charge (pair_smtbq.cpp): loop 0 computes the bond-order sums (sbcov, coord, sbmet) from the
   * geometry and forwards them to the ghosts; the later loops compute the charge forces potself, potmad and
   * potcov of the equilibrated atoms.
   */
  private qforce(pc: PairCompute, loop: number): void {
    const file = this.file!;
    const nb = pc.nb;
    const s = pc.s;
    const list = pc.full!;
    const x = pc.x, type = pc.type, qAll = pc.q;
    const step = s.step;
    const inum = list.inum;
    const { firstneigh, numneigh, neighbors } = list;
    const flag = this.groups.flag;
    const sbcov = this.sbcov, sbmet = this.sbmet, coord = this.coord;

    if (loop === 0) {
      sbcov.fill(0);
      coord.fill(0);
      sbmet.fill(0);
      // two-body interactions
      for (let i = 0; i < inum; i++) {
        const itype = this.map[type[i]];
        const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
        for (let jj = 0; jj < numneigh[i]; jj++) {
          const j = neighbors[firstneigh[i] + jj] & NEIGHMASK;
          const jtype = this.map[type[j]];
          const m = file.intype[itype * file.ntypes + jtype];
          const it = file.intparams[m];
          if (it.intsm === 0) continue;
          const delx = x[3 * j] - xtmp, dely = x[3 * j + 1] - ytmp, delz = x[3 * j + 2] - ztmp;
          const rsq = delx * delx + dely * dely + delz * delz;
          // covalent charge forces: sbcov initialization
          if (Math.sqrt(rsq) > it.dc2) continue;
          this.attractive(it, rsq, i);
        }
      }
      // communicates the tables sbcov, coord and sbmet to the ghosts
      nb.forwardCopy(sbcov, 1);
      nb.forwardCopy(coord, 1);
      nb.forwardCopy(sbmet, 1);
      for (let k = 0; k < pc.nall; k++) this.ifc[k] = intfcoup2(coord[k], file.coordOxBulk, OX_SWITCH);

      if (this.groups.nteam === 0) return;  // no oxide
      if (file.Qstep === 0 || step % file.Qstep !== 0) return;
    }

    const potmad = this.potmad, potself = this.potself, potcov = this.potcov;
    if (loop === 0 && this.esVal.length < list.neighbors.length) {
      this.esVal = new Float64Array(list.neighbors.length);
      this.esOk = new Uint8Array(list.neighbors.length);
    }
    for (let i = 0; i < inum; i++) {
      const itype = this.map[type[i]];
      const gp = flag[i];
      const iq = qAll[i];
      potmad[i] = 0.0;
      potself[i] = 0.0;
      potcov[i] = 0.0;
      if (gp === 0) continue;
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];

      // Madelung potential
      potmad[i] += 2.0 * this.Vself * iq;
      // charge force from the self energy
      potself[i] = this.qfoSelf(file.params[itype], iq);

      // long-range charge-dependent 1/r forces over the full list (the 1/r terms are cached by loop 0)
      for (let jj = 0; jj < numneigh[i]; jj++) {
        const k = firstneigh[i] + jj;
        const j = neighbors[k] & NEIGHMASK;
        if (loop === 0) {
          const jtype = this.map[type[j]];
          const delx = x[3 * j] - xtmp, dely = x[3 * j + 1] - ytmp, delz = x[3 * j + 2] - ztmp;
          const rsq = delx * delx + dely * dely + delz * delz;
          if (Math.sqrt(rsq) > this.cutmax) {
            this.esOk[k] = 0;
          } else {
            this.esOk[k] = 1;
            this.esVal[k] = this.potES(itype, jtype, i, j, rsq);
          }
        }
        if (this.esOk[k] === 0) continue;
        const jq = qAll[j];
        potmad[i] += jq * this.esVal[k];
      }

      potcov[i] = this.potCov(file.params[itype], i);
    }
  }

  /**
   * Charge (pair_smtbq.cpp): the charge equilibration of the groups, iterated until the electronegativity
   * deviation of each group is below the precision (the C++ damped charge dynamics with heatpq, qmass, dtq).
   */
  private charge(pc: PairCompute): void {
    const file = this.file!;
    const s = pc.s;
    const nb = pc.nb;
    const list = pc.full!;
    const inum = list.inum;
    const step = s.step;
    const q = s.q;
    const type = pc.type;

    const ctx: GroupContext = {
      nlocal: pc.nlocal, nall: pc.nall, inum, el: this.elementOf(pc), x: pc.x, owner: nb.owner, list, intype: file.intype,
      ntypes: file.ntypes, dc2: this.dc2, cutmax: this.cutmax, zlim1QEq: file.zlim1QEq, zlim2QEq: file.zlim2QEq,
    };
    switch (file.QEqMode) {
      case 'BulkFromSlab': this.groups.groupBulkFromSlab(ctx); break;
      case 'QEqAll': this.groups.groupQEqAll(ctx); break;
      case 'QEqAllParallel': this.groups.groupQEqAllParallel(ctx); break;
      case 'Surface': this.groups.groupSurface(ctx); break;
      default: throw new StyleError(`pair_style smtbq: unknown QEq mode ${file.QEqMode}`);
    }
    const flagAll = this.groups.flag;
    const nteam = this.groups.nteam;
    const G = nteam + 1;
    const cluster = G;

    const nQEqall = new Int32Array(G + 1);
    const nQEqcall = new Int32Array(G + 1);
    const nQEqaall = new Int32Array(G + 1);
    const q1 = this.q1, q2 = this.q2, qf = this.qf;
    for (let ii = 0; ii < inum; ii++) {
      const i = ii;
      const itype = this.map[type[i]];
      const gp = flagAll[i];
      q1[i] = 0.0;
      q2[i] = 0.0;
      qf[i] = 0.0;
      nQEqall[gp] += 1;
      if (itype !== 0) nQEqcall[gp] += 1;
      if (itype === 0) nQEqaall[gp] += 1;
    }

    // initial charges (QInit true): oxygens at QOxInit, the cations balance them
    if (step === 0 && file.QInitMode === 'true') {
      let gpLast = 0;
      if (inum > 0) gpLast = flagAll[inum - 1];
      this.QOxInit = Math.max(this.QOxInit, -0.98 * file.params[1].qform * nQEqcall[gpLast] / nQEqaall[gpLast]);
      for (let ii = 0; ii < inum; ii++) {
        const i = ii;
        const itype = this.map[type[i]];
        const gp = flagAll[i];
        if (itype === 0) q[i] = this.QOxInit;
        if (itype > 0) q[i] = -this.QOxInit * nQEqaall[gp] / nQEqcall[gp];
      }
      nb.refreshCharges(s);
    }

    if (nteam === 0 || file.Qstep === 0) return;
    if (step % file.Qstep !== 0) return;

    // Transf layout (C++): [gp] potentials, [gp + cluster] cation charges, [gp + 2 cluster] oxygen charges, [3 cluster] total
    const Transf = new Float64Array(3 * cluster + 1);
    const enegchk = new Float64Array(G);
    const enegmax = new Float64Array(G);
    const end = new Int32Array(G);

    const heatpq = 0.07;
    const qmass = 0.000548580;
    const dtq = 0.0006;
    const dtq2 = 0.5 * dtq * dtq / qmass;
    const precision = file.precision;
    const MAXIT = file.loopmax;

    for (let iloop = 0; iloop < MAXIT; iloop++) {
      Transf.fill(0);
      for (let ii = 0; ii < inum; ii++) {
        const i = ii;
        q1[i] += qf[i] * dtq2 - heatpq * q1[i];
        q[i] += q1[i];
        Transf[3 * cluster] += q[i];
        const itype = this.map[type[i]];
        const gp = flagAll[i];
        if (itype === 0) Transf[gp + 2 * cluster] += q[i];
        if (itype !== 0) Transf[gp + cluster] += q[i];
      }
      // charge communication
      nb.refreshCharges(s);

      // potentials
      this.qforce(pc, iloop);

      for (let ii = 0; ii < inum; ii++) {
        const i = ii;
        const gp = flagAll[i];
        qf[i] = 0.0;
        qf[i] = this.potself[i] + this.potmad[i] + this.potcov[i];
        Transf[gp] += qf[i];
      }

      // the average electronegativity of each group
      for (let i = 0; i < G; i++) {
        if (nQEqall[i] !== 0) Transf[i] /= nQEqall[i];
        enegchk[i] = 0.0;
        enegmax[i] = 0.0;
      }
      for (let ii = 0; ii < inum; ii++) {
        const i = ii;
        const gp = flagAll[i];
        if (gp === 0) continue;
        q2[i] = Transf[gp] - qf[i];
        enegmax[gp] = Math.max(enegmax[gp], Math.abs(q2[i]));
        enegchk[gp] += Math.abs(q2[i]);
        qf[i] = q2[i];
      }
      for (let gp = 0; gp < G; gp++) {
        if (nQEqall[gp] !== 0) {
          enegchk[gp] = enegchk[gp] / nQEqall[gp];
        }
      }

      // convergence test (groups 1 .. nteam)
      let m = 0;
      for (let gp = 1; gp < G; gp++) {
        if (enegchk[gp] <= precision && enegmax[gp] <= 100.0 * precision) end[gp] = 1;
      }
      for (let gp = 1; gp < G; gp++) m += end[gp];
      if (m === nteam) break;

      for (let ii = 0; ii < inum; ii++) {
        const i = ii;
        q1[i] += qf[i] * dtq2 - heatpq * q1[i];
      }
    }

    nb.refreshCharges(s);
  }

  /** Element index of every owned and ghost atom (map[type]). */
  private elementOf(pc: PairCompute): Int32Array {
    const el = new Int32Array(pc.nall);
    for (let k = 0; k < pc.nall; k++) el[k] = this.map[pc.type[k]];
    return el;
  }
}
