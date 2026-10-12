/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ELECTRODE/fix_electrode_conp.cpp and fix_electrode_conp.h (the constant-potential fix: charges from the
 * elastance matrix (algo mat_inv or mat_cg) or from conjugate gradients (algo cg), symmetrised for charge neutrality,
 * with the Gaussian-smearing correction and self energy), and the matrix and vector routines it calls
 * (electrode_matrix.cpp, electrode_vector.cpp) at commit b332f28. ELECTRODE contributing authors: Ludwig Ahrens-Iwers
 * (TUHH), Shern Tee (UQ), Robert Meissner (TUHH), and the LAMMPS developers. Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Supported: potentials as numbers or equal-style variables (v_name), couple groups, algo mat_inv / mat_cg / cg with
 * the CG threshold, symm on/off, eta as a number or as a property/atom vector (NULL), write_inv / write_mat /
 * write_vec files, and kspace_style ewald/electrode with boundary corrections. Not ported, refused with a StyleError:
 * qtotal, ffield yes, read_inv / read_mat, fix_modify tf, and the fix electrode/conq and thermo styles. etypes yes is
 * accepted: it selects type-based neighbour lists, which change the Gaussian correction (electrode-electrode pairs are
 * left out of the vector list), and that is reproduced here; the other lists give the same results.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { parseNum } from '../../engine/force/util';
import { invertDense } from './electrode_math';
import { MY_2PI, safeDerfcr } from './electrode_math';
import { eachImagePair, matrixPairTerms, matrixSelfTerms, vectorPairTerms, vectorSelfTerms } from './electrode_pairs';
import type { ElectrodeKSpace } from './ewald_electrode';

/** Smallest determinant of the two-electrode macro matrix before the inversion is refused (fix_electrode_conp.cpp SMALL). */
const SMALL = 1e-16;

type Algo = 'mat_inv' | 'mat_cg' | 'cg';

/** C printf %.11e (two-digit exponent at least) of x. */
const cExp = (x: number): string => {
  const [m, e] = x.toExponential(11).split('e');
  const digits = e.replace(/^[+-]/, '').padStart(2, '0');
  return `${m}e${e.startsWith('-') ? '-' : '+'}${digits}`;
};

const isElectrodeKSpace = (k: unknown): k is ElectrodeKSpace =>
  typeof k === 'object' && k !== null && typeof (k as ElectrodeKSpace).computeMatrix === 'function'
  && typeof (k as ElectrodeKSpace).computeVector === 'function';

export class FixElectrodeConp extends Fix {
  readonly style = 'electrode/conp';
  scalarFlag = true;
  vectorFlag = true;
  arrayFlag = false;
  extscalar = 1;
  extvector = 0;
  virialGlobal = true;
  energyGlobal = true;
  thermoVirial = true;
  thermoEnergy = true;

  /** Electrode group bits: the fix group first, then the couple groups; elecBit = their union. */
  private readonly groupBits: number[] = [];
  private elecBit = 0;
  private readonly psiConst: number[] = [];
  /** Equal-style variable name of each group potential (null = constant psiConst). */
  private readonly psiVar: (string | null)[] = [];
  private readonly groupPsi: number[] = [];
  private readonly sbCharges: number[] = [];
  private eta = 0;
  private etaNull = false;
  private etaName: string | null = null;
  private algo: Algo = 'mat_inv';
  private cgThreshold = 0;
  private symm = false;
  private nCgStep = 0;
  /** etypes yes: the electrode and electrolyte types must not overlap (checked at init). */
  private etypes = false;
  /** write_inv / write_mat / write_vec file names (null = not written). */
  private writeInv: string | null = null;
  private writeMat: string | null = null;
  private writeVec: string | null = null;
  private nCall = 0;

  // matrix state, rebuilt on every setup (positions change only between runs)
  private elec: Int32Array = new Int32Array(0);     // atom indices of the electrode atoms, in atom order
  private ieleOfElec: Int32Array = new Int32Array(0);  // matrix index (tag order) of each electrode atom
  private groupOfElec: Int32Array = new Int32Array(0);  // coupling group of each electrode atom
  private ieleToGroup: Int32Array = new Int32Array(0);  // coupling group of each matrix index
  private imat: Int32Array = new Int32Array(0);     // per atom: matrix index or -1
  private ngroup = 0;
  private elastance: Float64Array = new Float64Array(0);
  private capacitance: Float64Array = new Float64Array(0);
  private sdVectors: Float64Array[] = [];
  private macroCapacitance: number[][] = [];
  private macroElastance: number[][] = [];
  private evscale = 1;

  /** Energy and virial of the fix from the last force evaluation (the Gaussian correction and self energy). */
  private eCorr = 0;
  private readonly virialSum = new Float64Array(6);
  private kspaceOf: ElectrodeKSpace | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal fix electrode/conp command (usage: fix ID group electrode/conp pot eta ...)');
    this.groupBits.push(this.groupBit);
    const names: string[] = [group];
    // the potential: a number, or an equal-style variable v_name (re-evaluated at every charge update)
    this.psiVar.push(args[0].startsWith('v_') ? args[0].slice(2) : null);
    this.psiConst.push(args[0].startsWith('v_') ? 0 : parseNum(args[0], 'pot'));
    if (args[1] === 'NULL') this.etaNull = true;
    else this.eta = parseNum(args[1], 'eta');
    let iarg = 2;
    let defaultAlgo = true;
    while (iarg < args.length) {
      const key = args[iarg];
      switch (key) {
        case 'couple': {
          if (iarg + 2 >= args.length) throw new StyleError('Need two arguments after couple keyword');
          const gname = args[iarg + 1];
          if (sys.groups.find(gname) < 0) throw new StyleError(`Group ${gname} does not exist`);
          names.push(gname);
          this.groupBits.push(sys.groups.bit(gname));
          this.psiVar.push(args[iarg + 2].startsWith('v_') ? args[iarg + 2].slice(2) : null);
          this.psiConst.push(args[iarg + 2].startsWith('v_') ? 0 : parseNum(args[iarg + 2], 'couple potential'));
          iarg += 3;
          break;
        }
        case 'algo': {
          if (!defaultAlgo) throw new StyleError('Algorithm can be set only once');
          defaultAlgo = false;
          if (iarg + 1 >= args.length) throw new StyleError('Need at least one argument after algo command');
          const a = args[iarg + 1];
          if (a === 'mat_inv') this.algo = 'mat_inv';
          else if (a === 'mat_cg') this.algo = 'mat_cg';
          else if (a === 'cg') this.algo = 'cg';
          else throw new StyleError(`Unknown algo keyword ${a}`);
          iarg += 2;
          if (this.algo !== 'mat_inv') {
            if (iarg >= args.length) throw new StyleError('Need one argument after algo cg command');
            this.cgThreshold = parseNum(args[iarg], 'cg threshold');
            iarg++;
          }
          break;
        }
        case 'eta': {
          if (iarg + 1 >= args.length) throw new StyleError('Need one argument after eta command');
          const name = args[iarg + 1].replace(/^d_/, '');
          if (!sys.state.custom.has(name)) throw new StyleError('eta keyword requires name of previously defined property (fix property/atom d_name with ghost on)');
          this.etaName = name;
          iarg += 2;
          break;
        }
        case 'symm':
          this.symm = this.yesNo(args[iarg + 1], 'symm');
          iarg += 2;
          break;
        case 'etypes':
          // etypes selects type-based neighbour lists in the C++; the pairs used here are enumerated directly, so the
          // results do not change (the check that the electrode and electrolyte types are disjoint is kept)
          this.etypes = this.yesNo(args[iarg + 1], 'etypes');
          iarg += 2;
          break;
        case 'ffield':
          if (this.yesNo(args[iarg + 1], 'ffield')) throw new StyleError('fix electrode/conp ffield yes (finite-field electrolyte) is not supported by the browser engine');
          iarg += 2;
          break;
        case 'write_inv': case 'write_mat': case 'write_vec':
          if (iarg + 1 >= args.length) throw new StyleError(`Need one argument after ${key} command`);
          if (key === 'write_inv') this.writeInv = args[iarg + 1];
          else if (key === 'write_mat') this.writeMat = args[iarg + 1];
          else this.writeVec = args[iarg + 1];
          iarg += 2;
          break;
        case 'read_inv': case 'read_mat':
          throw new StyleError(`fix electrode/conp ${key} (reading matrices from files) is not supported by the browser engine`);
        case 'qtotal':
          throw new StyleError('fix electrode/conp qtotal is not supported by the browser engine');
        case 'temp':
          throw new StyleError('temp keyword not available for fix electrode/conp');
        default:
          throw new StyleError(`Unknown keyword ${key} for fix electrode/conp command`);
      }
    }
    if (this.etaNull && this.etaName === null) throw new StyleError('If eta is NULL the eta keyword must be used');
    this.groupPsi = this.psiConst.map(() => 0);
    this.sbCharges = this.psiConst.map(() => 0);
    this.sizeVector = this.groupBits.length;
    // array output of algo mat_inv (compute_array): sb charges, macro capacitance and macro elastance
    this.arrayFlag = this.algo === 'mat_inv';
    this.sizeArrayRows = this.groupBits.length;
    this.sizeArrayCols = 2 + 2 * this.groupBits.length;
    // groups may not overlap
    const mask = sys.state.mask;
    for (let i = 0; i < sys.state.n; i++) {
      let matches = 0;
      for (const b of this.groupBits) if (mask[i] & b) matches++;
      if (matches > 1) throw new StyleError('Groups may not overlap');
    }
    this.elecBit = this.groupBits.reduce((a, b) => a | b, 0);
    if (this.algo !== 'mat_inv' && this.writeInv !== null) {
      throw new StyleError('Selected algorithm does not use inverted matrix. Cannot read/write inverted matrix.');
    }
    if (this.algo === 'cg' && (this.writeMat !== null || this.writeVec !== null)) {
      throw new StyleError('Selected algorithm does not use matrix. Cannot read/write matrix or vector.');
    }
  }

  /** utils::logical: yes/on/true/1 and no/off/false/0. */
  private yesNo(w: string | undefined, key: string): boolean {
    if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
    if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
    throw new StyleError(`Expected yes or no after ${key}`);
  }

  modify(key: string, values: string[]): number {
    if (key === 'tf') throw new StyleError('fix_modify tf (Thomas-Fermi) is not supported by the browser engine (fix electrode/conp)');
    if (key === 'energy') throw new StyleError('fix_modify energy is not used by fix electrode/conp (its energy is always part of pe)');
    return super.modify(key, values);
  }

  /** Fix init: the Coulomb pair style and the k-space style must support the electrode interface. */
  init(): void {
    const pair = this.sys.ff.pair;
    if (!pair || !pair.name.includes('coul')) throw new StyleError("Fix electrode couldn't find a Coulombic pair style");
    const k = this.sys.ff.kspace;
    if (!isElectrodeKSpace(k)) throw new StyleError('KSpace does not implement ElectrodeKSpace (use kspace_style ewald/electrode)');
    this.kspaceOf = k;
    if (this.etypes) {
      // fix_electrode_conp.cpp request_etypes_neighlists(): a type must belong wholly to the electrodes or wholly to the electrolyte
      const s = this.sys.state;
      const elec = new Int32Array(s.ntypes + 1), elyt = new Int32Array(s.ntypes + 1);
      for (let i = 0; i < s.n; i++) {
        if (s.mask[i] & this.elecBit) elec[s.type[i]]++;
        else elyt[s.type[i]]++;
      }
      for (let t = 1; t <= s.ntypes; t++) {
        if (!elec[t] === !elyt[t]) throw new StyleError('Types overlap, cannot use etypes keyword');
      }
    }
  }

  /** Tag-sorted matrix index per electrode atom and the coupling group of each atom (fix_electrode_conp.cpp create_taglist). */
  private buildLists(): void {
    const s = this.sys.state;
    const n = s.n;
    const elecIdx: number[] = [];
    for (let i = 0; i < n; i++) if (s.mask[i] & this.elecBit) elecIdx.push(i);
    const sorted = [...elecIdx].sort((a, b) => s.id[a] - s.id[b]);
    const tagToIele = new Map<number, number>();
    sorted.forEach((i, k) => tagToIele.set(s.id[i], k));
    this.ngroup = elecIdx.length;
    this.elec = Int32Array.from(elecIdx);
    this.ieleOfElec = Int32Array.from(elecIdx.map((i) => tagToIele.get(s.id[i])!));
    this.groupOfElec = Int32Array.from(elecIdx.map((i) => this.groupBits.findIndex((b) => (s.mask[i] & b) !== 0)));
    this.ieleToGroup = new Int32Array(this.ngroup);
    elecIdx.forEach((_, k) => { this.ieleToGroup[this.ieleOfElec[k]] = this.groupOfElec[k]; });
    this.imat = new Int32Array(n).fill(-1);
    elecIdx.forEach((i, k) => { this.imat[i] = this.ieleOfElec[k]; });
    // create_taglist: the IDs group by group (each group sorted), and the matrix index of each (group_idx)
    this.taglistBygroup = [];
    for (const b of this.groupBits) {
      const tags: number[] = [];
      for (let i = 0; i < n; i++) if (s.mask[i] & b) tags.push(s.id[i]);
      this.taglistBygroup.push(...tags.sort((a, c) => a - c));
    }
    this.groupIdx = this.taglistBygroup.map((t) => tagToIele.get(t)!);
  }

  /** eta of atom i: the constant, or the per-atom property. */
  private etaOf = (i: number): number => (this.etaName !== null ? this.sys.state.custom.get(this.etaName)!.data[i] : this.eta);

  /** Elastance matrix (Coulomb between electrode atoms, with the k-space part and boundary corrections). */
  private computeElastance(): void {
    const s = this.sys.state, ks = this.kspaceOf!;
    const ng = this.ngroup;
    this.elastance = new Float64Array(ng * ng);
    const geom = this.sys.geom;
    const nt = this.sys.ff.pair!.ntypes + 1;
    const cutsq = this.sys.ff.pair!.cutsq;
    ks.computeMatrix(s, this.imat, this.elastance, ng);
    // etypes yes: the matrix list keeps only the pairs of equal electrode type (ijskip_mat[etype][etype])
    matrixPairTerms(s, geom, cutsq, nt, this.elecBit, this.imat, ng, this.elastance, ks.gEwald, this.etaOf, this.etypes);
    matrixSelfTerms(s, this.elecBit, this.imat, ng, this.elastance, ks.gEwald, this.etaOf);
    ks.computeMatrixCorr(s, this.imat, this.elastance, ng);
  }

  /** Dense symmetrisation: C -= (C 1)(C 1)^T / (1^T C 1) (fix_electrode_conp.cpp symmetrize). */
  private symmetrize(): void {
    const ng = this.ngroup, C = this.capacitance;
    const AinvE = new Float64Array(ng);
    let EAinvE = 0;
    for (let i = 0; i < ng; i++) {
      let t = 0;
      for (let j = 0; j < ng; j++) t += C[i * ng + j];
      AinvE[i] = t;
      EAinvE += AinvE[i];
    }
    for (let i = 0; i < ng; i++) {
      const iAinvE = AinvE[i];
      for (let j = 0; j < ng; j++) C[i * ng + j] -= AinvE[j] * iAinvE / EAinvE;
    }
  }

  /** compute_sd_vectors(): the response of the electrode charges to each group potential. */
  private computeSdVectors(): void {
    const ng = this.ngroup, C = this.capacitance, ev = this.evscale;
    this.sdVectors = [];
    for (let g = 0; g < this.groupBits.length; g++) {
      const sd = new Float64Array(ng);
      for (let j = 0; j < ng; j++) {
        if (this.ieleToGroup[j] !== g) continue;
        for (let k = 0; k < ng; k++) sd[k] += C[k * ng + j] * ev;
      }
      this.sdVectors.push(sd);
    }
  }

  /** compute_macro_matrices(): the group-group capacitance and its inverse (fix_electrode_conp.cpp). */
  private computeMacroMatrices(): void {
    const G = this.groupBits.length;
    const mc: number[][] = Array.from({ length: G }, () => new Array(G).fill(0));
    for (let g = 0; g < G; g++) {
      for (let k = 0; k < this.ngroup; k++) mc[this.ieleToGroup[k]][g] += this.sdVectors[g][k];
    }
    if (this.symm) {
      // scaling with C[0][0] improves numerical stability
      const scalar = mc[0][0];
      mc[G - 1] = new Array(G).fill(scalar);
    }
    this.macroCapacitance = mc;
    const me: number[][] = Array.from({ length: G }, () => new Array(G).fill(0));
    if (G === 1) {
      me[0][0] = 1 / mc[0][0];
    } else if (G === 2) {
      const det = mc[0][0] * mc[1][1] - mc[0][1] * mc[1][0];
      if (Math.abs(det) < SMALL) throw new StyleError('ELECTRODE macro matrix inversion failed!');
      const detinv = 1 / det;
      me[0][0] = mc[1][1] * detinv;
      me[1][1] = mc[0][0] * detinv;
      me[0][1] = -mc[0][1] * detinv;
      me[1][0] = -mc[1][0] * detinv;
    } else {
      const flat = new Float64Array(G * G);
      for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) flat[i * G + j] = mc[i][j];
      if (!invertDense(flat, G)) throw new StyleError('ELECTRODE macro matrix inversion failed!');
      for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) me[i][j] = flat[i * G + j];
    }
    this.macroElastance = me;
  }

  /** Charges, b vector and matrices at the start of a run (setup_post_neighbor), then the fix's energy and forces. */
  setup(): void {
    this.buildLists();
    const s = this.sys.state;
    const ks = this.kspaceOf!;
    this.evscale = s.units.qe2f / ks.qqrd2e;
    if (this.algo !== 'cg') {
      this.computeElastance();
      if (this.writeMat !== null) this.writeTable(this.writeMat, this.orderMatrix(this.elastance));
      if (this.algo === 'mat_inv') {
        this.capacitance = Float64Array.from(this.elastance);
        if (!invertDense(this.capacitance, this.ngroup)) throw new StyleError('CONP matrix inversion failed!');
        if (this.symm) this.symmetrize();
        this.computeSdVectors();
        this.computeMacroMatrices();
      }
    }
    this.updateCharges();
    if (this.writeVec !== null) {
      const pot = this.electrodeVector(this.elecBit, this.elecBit, true);
      const potIele = new Float64Array(this.ngroup);
      for (let k = 0; k < this.elec.length; k++) potIele[this.ieleOfElec[k]] = pot[this.elec[k]];
      const ng = this.ngroup;
      const vec = Array.from({ length: ng }, () => [0]);
      for (let i = 0; i < ng; i++) vec[this.groupIdx[i]][0] = potIele[i];
      this.writeTable(this.writeVec, vec);
    }
    if (this.writeInv !== null) this.writeTable(this.writeInv, this.orderMatrix(this.capacitance));
    this.sys.computeForcesInRun(this.sys.computeFlags());
    this.sys.forcesCurrent();
    this.preReverseTerms();
  }

  /** Atom IDs by coupling group (each group sorted by ID) and the matrix index of each of them (group_idx). */
  private groupIdx: number[] = [];
  private taglistBygroup: number[] = [];

  /** fix_electrode_conp.cpp order_matrix(): the matrix re-indexed by the group-sorted ID list. */
  private orderMatrix(mat: Float64Array): number[][] {
    const ng = this.ngroup;
    const ordered: number[][] = Array.from({ length: ng }, () => new Array(ng).fill(0));
    for (let i = 0; i < ng; i++) {
      const gi = this.groupIdx[i];
      for (let j = 0; j < ng; j++) ordered[gi][this.groupIdx[j]] = mat[i * ng + j];
    }
    return ordered;
  }

  /** write_to_file(): the tag header and one row per matrix row, C-style %20 and %20.11e columns. */
  private writeTable(name: string, rows: number[][]): void {
    let out = this.taglistBygroup.map((t) => String(t).padStart(20)).join('') + '\n';
    for (const r of rows) out += r.map((x) => cExp(x).padStart(20)).join('') + '\n';
    this.sys.writeFile(name, out, false);
  }

  /** Charges are set before the force evaluation of every step (fix electrode/conp pre_force). */
  preForce(): void {
    this.updateCharges();
  }

  /** After the pair/k-space forces (pre_reverse): the Gaussian correction and the self energy. */
  preReverse(): void {
    this.preReverseTerms();
  }

  private preReverseTerms(): void {
    this.virialSum.fill(0);
    const eSr = this.gausscorr();
    const eSelf = this.selfEnergy();
    this.eCorr = eSr + eSelf;
    this.virial.set(this.virialSum);
  }

  /** The charge update of the chosen algorithm (fix_electrode_conp.cpp update_charges). */
  private updateCharges(): void {
    this.nCall++;
    const s = this.sys.state;
    const ng = this.ngroup, n = this.elec.length;
    const qLocal = new Float64Array(n);
    if (this.algo === 'mat_inv') {
      this.sbCharges.fill(0);
      const pot = this.electrodeVector(this.elecBit, this.elecBit, true);
      const potIele = new Float64Array(ng);
      for (let k = 0; k < n; k++) potIele[this.ieleOfElec[k]] = pot[this.elec[k]];
      const C = this.capacitance;
      for (let k = 0; k < n; k++) {
        const iele = this.ieleOfElec[k];
        let qtmp = 0;
        for (let j = 0; j < ng; j++) qtmp -= C[iele * ng + j] * potIele[j];
        qLocal[k] = qtmp;
        this.sbCharges[this.ieleToGroup[iele]] += qtmp;
      }
      this.updatePsi();
      for (let g = 0; g < this.groupBits.length; g++) {
        for (let k = 0; k < n; k++) qLocal[k] += this.sdVectors[g][this.ieleOfElec[k]] * this.groupPsi[g];
      }
    } else {
      // conjugate gradients (mat_cg with the elastance, cg with the electrode vector)
      this.updatePsi();
      const pot = this.electrodeVector(this.elecBit, this.elecBit, true);
      const b = new Float64Array(n);
      for (let k = 0; k < n; k++) {
        b[k] = pot[this.elec[k]] - this.evscale * this.groupPsi[this.groupOfElec[k]];
        qLocal[k] = s.q[this.elec[k]];
      }
      let q = this.constraintCorrection(qLocal);
      let a = this.eleEle(q);
      let r = this.add(b, a);
      let d = this.constraintProjection(r);
      let dotOld = this.dot(r, d);
      let delta = dotOld;
      for (let k = 0; k < ng && delta > this.cgThreshold; k++, this.nCgStep++) {
        const y = this.eleEle(d);
        const alpha = dotOld / -this.dot(d, y);
        q = this.add(q, this.scale(alpha, d));
        // prepare next step
        if ((k + 1) % 20 === 0) {
          // avoid shifting residual. This rarely happens.
          q = this.constraintCorrection(q);
          a = this.eleEle(q);
          r = this.add(b, a);
        } else {
          r = this.add(r, this.scale(alpha, y));
        }
        const p = this.constraintProjection(r);
        const dotNew = this.dot(r, p);
        d = this.add(p, this.scale(dotNew / dotOld, d));
        delta = this.dot(r, d);
        dotOld = dotNew;
      }
      if (delta > this.cgThreshold) this.sys.warn('CG threshold not reached');
      for (let k = 0; k < n; k++) qLocal[k] = q[k];
    }
    this.setCharges(qLocal);
  }

  /**
   * electrode_vector: the potential at every atom from the source charges (pair term, self term, k-space, correction).
   * electrodeOnly: the electrode-electrode list of the CG algorithm (elec_vector), which etypes restricts to equal types.
   */
  private electrodeVector(sensorBit: number, sourceBit: number, invert: boolean, electrodeOnly = false): Float64Array {
    const s = this.sys.state, ks = this.kspaceOf!;
    const pot = new Float64Array(s.n);
    const sameType = electrodeOnly && this.etypes;
    vectorPairTerms(s, this.sys.geom, this.sys.ff.pair!.cutsq, this.sys.ff.pair!.ntypes + 1, sensorBit, sourceBit, invert, pot, ks.gEwald, this.etaOf, sameType);
    vectorSelfTerms(s, sensorBit, sourceBit, invert, pot, ks.gEwald, this.etaOf);
    ks.computeVector(s, pot, sensorBit, sourceBit, invert);
    ks.computeVectorCorr(s, pot, sensorBit, sourceBit, invert);
    return pot;
  }

  /** ele_ele_interaction(): A q at the electrode atoms (the electrode vector with the charges set, or the elastance). */
  private eleEle(q: Float64Array): Float64Array {
    if (this.algo === 'cg') {
      this.setCharges(q);
      const pot = this.electrodeVector(this.elecBit, this.elecBit, false, true);
      const out = new Float64Array(this.elec.length);
      for (let k = 0; k < this.elec.length; k++) out[k] = pot[this.elec[k]];
      return out;
    }
    // times_elastance(gather_ngroup(q))
    const ng = this.ngroup;
    const x = new Float64Array(ng);
    for (let k = 0; k < this.elec.length; k++) x[this.ieleOfElec[k]] = q[k];
    const out = new Float64Array(this.elec.length);
    for (let k = 0; k < this.elec.length; k++) {
      const row = this.ieleOfElec[k];
      let oi = 0;
      for (let j = 0; j < ng; j++) oi += this.elastance[row * ng + j] * x[j];
      out[k] = oi;
    }
    return out;
  }

  /** constraint_correction and constraint_projection (symm on: total electrode charge 0). */
  private constraintCorrection(x: Float64Array): Float64Array {
    if (!this.symm) return x;
    let sum = 0;
    for (let k = 0; k < x.length; k++) sum += x[k];
    sum /= this.ngroup;
    const out = Float64Array.from(x);
    for (let k = 0; k < out.length; k++) out[k] -= sum;
    return out;
  }

  private constraintProjection(x: Float64Array): Float64Array {
    return this.constraintCorrection(x);
  }

  private add(a: Float64Array, b: Float64Array): Float64Array {
    const out = Float64Array.from(a);
    for (let k = 0; k < out.length; k++) out[k] += b[k];
    return out;
  }

  private scale(alpha: number, x: Float64Array): Float64Array {
    const out = Float64Array.from(x);
    for (let k = 0; k < out.length; k++) out[k] *= alpha;
    return out;
  }

  private dot(a: Float64Array, b: Float64Array): number {
    let out = 0;
    for (let k = 0; k < a.length; k++) out += a[k] * b[k];
    return out;
  }

  /** set_charges(): writes the electrode charges and refreshes the ghost copies the pair style reads (comm->forward_comm). */
  private setCharges(q: Float64Array): void {
    const s = this.sys.state;
    for (let k = 0; k < this.elec.length; k++) s.q[this.elec[k]] = q[k];
    this.sys.nb.refreshCharges(s);
  }

  /** update_psi(): the constant potentials, or the current value of their equal-style variables. */
  private updatePsi(): void {
    for (let g = 0; g < this.groupBits.length; g++) {
      const v = this.psiVar[g];
      this.groupPsi[g] = v === null ? this.psiConst[g] : this.sys.vars.scalar(v, this.sys.formulaEnv);
    }
  }

  /** gausscorr(): the short-range correction to the Coulomb pair term, for electrode-containing pairs (ordered pairs, half weight). */
  private gausscorr(): number {
    const s = this.sys.state, ks = this.kspaceOf!;
    const qqrd2e = ks.qqrd2e;
    const f = s.f, q = s.q, mask = s.mask;
    const pair = this.sys.ff.pair!;
    const nt = pair.ntypes + 1;
    const cutsq = pair.cutsq;
    const out = { erfc: 0 };
    let energy = 0;
    const vir = this.virialSum;
    eachImagePair(s, this.sys.geom, cutsq, nt, (i, j, rsq, dx, dy, dz) => {
      const iEle = (mask[i] & this.elecBit) !== 0, jEle = (mask[j] & this.elecBit) !== 0;
      if (!(iEle || jEle)) return;
      // etypes yes: the vector neighbour list of the C++ skips electrode-electrode pairs (ijskip_vec), and gausscorr reads it
      if (this.etypes && iEle && jEle) return;
      const etaI = this.etaOf(i), etaJ = this.etaOf(j);
      let etaIj: number;
      if (iEle && jEle) etaIj = etaI * etaJ / Math.sqrt(etaI * etaI + etaJ * etaJ);
      else if (iEle) etaIj = etaI;
      else etaIj = etaJ;
      const r2inv = 1.0 / rsq;
      const r = Math.sqrt(rsq);
      const derfcr = safeDerfcr(etaIj * r, out);
      const prefactor = qqrd2e * q[i] * q[j] / r;
      const half = 0.5;   // each unordered pair is visited twice (both orders)
      energy -= half * prefactor * out.erfc;
      const fpair = prefactor * derfcr * r2inv;
      f[3 * i] += half * dx * fpair;
      f[3 * i + 1] += half * dy * fpair;
      f[3 * i + 2] += half * dz * fpair;
      f[3 * j] -= half * dx * fpair;
      f[3 * j + 1] -= half * dy * fpair;
      f[3 * j + 2] -= half * dz * fpair;
      // virial of the pair (v_tally, half list weights)
      vir[0] += half * dx * dx * fpair;
      vir[1] += half * dy * dy * fpair;
      vir[2] += half * dz * dz * fpair;
      vir[3] += half * dx * dy * fpair;
      vir[4] += half * dx * dz * fpair;
      vir[5] += half * dy * dz * fpair;
    });
    return energy;
  }

  /** self_energy(): the eta-dependent self interaction of every electrode atom. */
  private selfEnergy(): number {
    const s = this.sys.state, ks = this.kspaceOf!;
    const qqrd2e = ks.qqrd2e;
    const pre = 1.0 / Math.sqrt(MY_2PI) * qqrd2e;
    let energy = 0;
    for (let k = 0; k < this.elec.length; k++) {
      const i = this.elec[k];
      const q2 = s.q[i] * s.q[i];
      const ieta = this.etaOf(i);
      energy += ieta * pre * q2;
    }
    return energy;
  }

  /** Pair energy part of the fix (in pe, fix_modify energy is always on). */
  energy(): number {
    return this.eCorr;
  }

  /** f_ID: the electrode potential energy term -sum q_i psi_g evscale (potential_energy()). */
  computeScalar(): number {
    const s = this.sys.state, ks = this.kspaceOf!;
    let e = 0;
    for (let k = 0; k < this.elec.length; k++) {
      const i = this.elec[k];
      e -= ks.qqrd2e * s.q[i] * this.groupPsi[this.groupOfElec[k]] * this.evscale;
    }
    return e;
  }

  /** f_ID[i][j] (algo mat_inv): j = 0 the coupling-group charge, then the macro capacitance, then the macro elastance. */
  computeArray(i: number, j: number): number {
    const G = this.groupBits.length;
    if (j === 0) return this.sbCharges[i];
    if (j <= G) return this.macroCapacitance[i][j - 1];
    if (j <= 2 * G) return this.macroElastance[i][j - G - 1];
    return 0;
  }

  /** f_ID[i]: the group potential. */
  computeVector(i: number): number {
    return this.groupPsi[i];
  }

}
