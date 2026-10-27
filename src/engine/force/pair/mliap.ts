import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { missingArgs } from '../util';
import { NEIGHMASK } from '../../neighbor';
import { WignerTables, rawBispectrum, adjointBispectrum, newZStore, type Cmat, type Grad } from '../../compute/sna';
import { parseMliapSnaDescriptor, type MliapSnaDescriptor } from '../../mliap/descriptor';
import { parseMliapModel, type MliapModel, type ModelKind } from '../../mliap/model';
import { parseMliapSo3Descriptor, So3Engine, type So3Descriptor } from '../../mliap/so3';
import { ChemSna } from '../../compute/snapChem';
import { libm } from '../../libm';
import { gplHook } from '../../gplHooks';
import type { MliapAceDescriptor, MliapAceHook } from '../../mliap/ace';

/*
 * pair_style mliap (wave 16) — docs.lammps.org/pair_mliap.html (plans/lammps-docs/pair_mliap.rst).
 *
 * Syntax (verbatim): "pair_style mliap ... keyword values ..."; "one or two keyword/value pairs must
 * be appended"; "keyword = model or descriptor or unified". This engine implements the pair
 * model plus descriptor combinations: model linear, quadratic or nn with descriptor sna, model
 * linear or nn with descriptor so3 (src/engine/mliap/so3.ts), and model linear, quadratic or nn with descriptor
 * ace (the ACE projections of a .ace or .yace file, src/engine/mliap/ace.ts). The other styles are reported as
 * StyleError: model mliappy and the unified keyword (they need the Python package or a serialized Python object).
 *
 * descriptor ace: as native LAMMPS does (src/ML-IAP/mliap_descriptor_ace.cpp and mliap_data.cpp at b332f28): the
 * neighbours of atom i are those of the full list with r^2 < cutsq of the two elements (computed in single
 * precision), and the descriptors are the projections of atom type t as ACE species t - 1 (the pair_coeff names map
 * the types to the model's element blocks and the cutoffs). The force on atom i is the sum over neighbours j and
 * descriptors k of beta_k dB_k/dr_j, with the opposite force on j; the per-atom virial tallies -r_ij (x) f_ij, half to
 * each atom (PairMLIAP::v_tally), and the global virial is the f.r sum. The doc lists linear and mliappy as the
 * models with ace; quadratic and nn use the same descriptor gradients through dE/dB, as natively.
 *
 * pair_coeff: the first 2 arguments must be * *, followed by N element names mapping the MLIAP elements
 * to the N LAMMPS atom types (paraphrase of the doc). NULL mapping is refused by the native reader in
 * this setup (measured with native LAMMPS, black box), so it is a StyleError here.
 *
 * Energy: each atom i of element e has E_i = model_e(B^i), the same bispectrum B^i as pair_style
 * snap (src/engine/mliap/descriptor.ts; the bispectrum and its derivatives are the sna.ts
 * machinery). Measured with native LAMMPS (black box): "model linear" with the coefficients of an
 * existing snap file gives the same potential energy and pressure as pair_style snap, and "model
 * quadratic" equals pair_style snap with quadraticflag 1. The engine's forces of the two styles agree
 * to 1e-12 (tests/engineMliap.test.ts).
 *
 * Force loop: the same analytic chain rule as pair_style snap (see src/engine/force/pair/snap.ts for
 * the derivation): dE_i/dd is accumulated from the adjoint of the bispectrum with respect to the
 * model's dE_i/dB, gam = dE_i/dB of the model (energy(e, B, gam) in src/engine/mliap/model.ts).
 *
 * "This pair style does not support the pair_modify shift, table, and tail options." The per-atom
 * virial (stress/atom) is tallied as for pair_style snap (see the force loop below).
 *
 * chemflag 1 in the sna descriptor (explicit multi-element SNAP, Cusentino, Wood, Thompson 2020): the
 * descriptor is the multi-element bispectrum of src/engine/compute/snapChem.ts, shared with pair_style snap,
 * and the model is linear with nparams = 1 + K N_elem^3 per element. Doc (docs.lammps.org/compute_mliap.html,
 * plans/lammps-docs-live/compute_mliap.rst, see the block comment above DESCRIPTOR_STYLES): the SNAP descriptor
 * style sna is the same as that used by pair_style snap, including the linear, quadratic, and chem variants.
 * Measured with native LAMMPS (black box):
 * with the InP_JCPA2020 mliap model and descriptor files and the pair_coeff line that maps type 1 to In and
 * type 2 to P, energy and forces equal pair_style snap with the InP_JCPA2020 snapcoeff and snapparam files to
 * the last printed digit. With the same files and the pair_coeff line that maps type 1 to P and type 2 to In,
 * native gives an energy of -370.904 for the 64-atom displaced cell where pair_style snap gives -163.032, and
 * the pair_coeff line that maps both types to In runs natively where pair_style snap refuses it (unique element
 * count). The engine therefore accepts chem only when the pair_coeff names are the descriptor's elements in the
 * descriptor's order, one per atom type (the measured case); the other mappings are StyleErrors. Model quadratic
 * and nn with chem are StyleErrors (not measured), and compute mliap refuses chem.
 */

/*
 * Doc (docs.lammps.org/compute_mliap.html, plans/lammps-docs-live/compute_mliap.rst), the sna descriptor of
 * compute mliap: "The SNAP descriptor style *sna* is the same as that used by :doc:`pair_style snap <pair_snap>`, including the linear, quadratic, and chem variants."
 */
const DESCRIPTOR_STYLES = ['sna', 'so3', 'ace'];

export class PairMliap extends Pair {
  readonly name = 'mliap';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private modelKind: ModelKind | null = null;
  private modelFile = '';
  private modelText = '';
  private descFile = '';
  private desc: MliapSnaDescriptor | null = null;
  /** Descriptor so3 (wave 37, src/engine/mliap/so3.ts); null for sna. */
  private so3: So3Descriptor | null = null;
  private so3Engine: So3Engine | null = null;
  private ace: MliapAceDescriptor | null = null;
  private model: MliapModel | null = null;
  /** Per atom type: element index in the descriptor, or -1 for NULL. */
  private elemOf = new Int32Array(0);
  private tables: WignerTables | null = null;
  /** chemflag 1 bispectrum (created on the first compute after pair_coeff). */
  private chemEng: ChemSna | null = null;

  settings(args: string[], ctx: StyleContext): void {
    // As native LAMMPS does (src/ML-IAP/pair_mliap.cpp PairMLIAP::settings at b332f28): keywords in order, each model
    // or descriptor read from its file when its keyword is reached, at most one of each; python-backed options
    // (model mliappy, unified) need a python build, which the browser has not.
    const n = args.length;
    if (n < 2) throw missingArgs('pair_style mliap');
    let kind: string | null = null, mfile = '', modelText = '', dstyle: string | null = null, dfile = '', descText = '';
    let k = 0;
    while (k < n) {
      const kw = args[k];
      if (kw === 'model') {
        if (k + 2 > n) throw missingArgs('pair_style mliap model');
        if (kind !== null) throw new StyleError('Illegal multiple pair_style mliap model definition');
        const style = args[k + 1];
        if (style === 'linear' || style === 'quadratic' || style === 'nn') {
          if (k + 3 > n) throw missingArgs(`pair_style mliap model ${style}`);
          kind = style; mfile = args[k + 2];
          modelText = ctx.readFile(mfile, 'Cannot open MLIAPModel coeff file');
          k += 3;
        } else if (style === 'mliappy') {
          throw new StyleError('Using pair_style mliap model mliappy requires ML-IAP with python support');
        } else {
          throw new StyleError(`Unknown pair_style mliap model keyword: ${kw}`);
        }
      } else if (kw === 'descriptor') {
        if (k + 2 > n) throw missingArgs('pair_style mliap descriptor');
        if (dstyle !== null) throw new StyleError('Illegal multiple pair_style mliap descriptor definition');
        const style = args[k + 1];
        if (!DESCRIPTOR_STYLES.includes(style)) throw new StyleError('Illegal pair_style mliap command');
        if (k + 3 > n) throw style === 'ace' ? new StyleError('Illegal pair_style mliap command') : missingArgs(`pair_style mliap descriptor ${style}`);
        dstyle = style; dfile = args[k + 2];
        descText = ctx.readFile(dfile, style === 'sna' ? 'Cannot open SNAP parameter file' : style === 'so3' ? 'Cannot open SO3 parameter file' : undefined);
        k += 3;
      } else if (kw === 'unified') {
        throw new StyleError('Using pair_style mliap unified requires ML-IAP with python support');
      } else {
        throw new StyleError(`Unknown pair_style mliap keyword: ${kw}`);
      }
    }
    if (kind === null || dstyle === null) throw new StyleError('Incomplete pair_style mliap setup: need model and descriptor, or unified');
    if (dstyle === 'so3' && kind === 'quadratic') {
      // doc: "The available models are *linear* and *nn*." (so3)
      throw new StyleError(`pair_style mliap: model quadratic with descriptor so3 is not implemented (the doc lists linear and nn for so3)`);
    }
    this.modelKind = kind as ModelKind;
    this.modelFile = mfile;
    this.ace = null;
    if (dstyle === 'ace') {
      this.ace = gplHook<MliapAceHook>('ace').load(descText, dfile);
      this.desc = null;
      this.so3 = null;
      this.so3Engine = null;
    } else if (dstyle === 'so3') {
      this.so3 = parseMliapSo3Descriptor(descText, dfile);
      this.so3Engine = new So3Engine(this.so3);
      this.desc = null;
    } else {
      this.desc = parseMliapSnaDescriptor(descText, dfile);
      this.so3 = null;
      this.so3Engine = null;
    }
    this.descFile = dfile;
    this.modelText = modelText;
    this.model = null;
  }

  coeff(args: string[]): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    const base = this.desc ?? this.so3;
    if ((!base && !this.ace) || !this.modelKind) throw new StyleError('pair_coeff for style mliap needs pair_style mliap first');
    const descElems: readonly string[] = this.ace ? this.ace.elements : base!.elems;
    if (args.length < 2 || args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style mliap must be * *');
    // Measured with native LAMMPS (black box): the Si nn example's "pair_coeff * * Si Si" runs with one atom type,
    // so extra element names are accepted and only the first N (N = atom types) are mapped. Fewer names: StyleError.
    const elems = args.slice(2);
    if (elems.length < this.ntypes) throw new StyleError(`pair_coeff for style mliap needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    const chem = this.desc !== null && this.desc.chem;
    if (chem && this.modelKind !== 'linear') {
      throw new StyleError(`pair_style mliap: descriptor sna with chemflag 1 is implemented for model linear only (got model ${this.modelKind})`);
    }
    const nmodel = this.ace ? this.ace.ndescriptors : this.desc ? this.desc.nmodel : base!.K;
    this.model = parseMliapModel(this.modelKind, this.modelText, this.modelFile, nmodel, descElems.length);
    const nt = this.ntypes + 1;
    this.elemOf = new Int32Array(nt).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') throw new StyleError('NULL element mapping for pair_style mliap is not implemented in this engine');
      const idx = descElems.indexOf(name);
      if (idx < 0) throw new StyleError(`element '${name}' is not in the mliap descriptor file ${this.descFile} (elements: ${descElems.join(' ')})`);
      this.elemOf[t] = idx;
    }
    if (chem) {
      // Measured with native LAMMPS (black box): see the header. The chem index follows the descriptor order.
      for (let t = 1; t <= this.ntypes; t++) {
        if (this.elemOf[t] !== t - 1 || this.ntypes !== descElems.length) {
          throw new StyleError(`pair_style mliap with descriptor sna chemflag 1: pair_coeff must name the ${descElems.length} descriptor elements (${descElems.join(' ')}) in that order, one per atom type (got ${elems.slice(0, this.ntypes).join(' ')})`);
        }
      }
    }
    this.chemEng = null;
  }

  /** Cutoff rcutfac (R_i + R_j) between mapped types. */
  initOne(i: number, j: number): number {
    if (!this.desc && !this.so3 && !this.ace) throw new StyleError('pair_coeff for style mliap has not been given');
    const ei = this.elemOf[i], ej = this.elemOf[j];
    if (ei < 0 || ej < 0) return 0;
    // ace: PairMLIAP::init_one, sqrt of the descriptor's cutsq of the two elements
    if (this.ace) return Math.sqrt(this.ace.cutsq(ei, ej));
    // so3: every pair is cut at rcutfac (measured with native LAMMPS, black box; see so3.ts)
    if (this.so3) return this.so3.rcutfac;
    return this.desc!.rcutfac * (this.desc!.radius[ei] + this.desc!.radius[ej]);
  }

  compute(pc: PairCompute): void {
    if (this.ace && this.model) {
      this.computeAce(pc, this.ace, this.model);
      return;
    }
    if (this.so3Engine && this.so3 && this.model) {
      this.computeSo3(pc, this.so3Engine, this.so3, this.model);
      return;
    }
    if (!this.desc || !this.model) throw new StyleError('pair_coeff for style mliap has not been given');
    if (this.desc.chem) {
      this.computeChem(pc, this.desc, this.model);
      return;
    }
    const d = this.desc, model = this.model;
    const list = pc.full;
    if (!list) throw new Error('pair style mliap needs a full neighbor list');
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const K = d.K, tj = d.twojmax, triples = d.triples;
    const zs = newZStore(triples);
    const rfac0 = d.rfac0, rmin0 = d.rmin0, rc = d.rcutfac;

    const u: Cmat[] = [];
    const gr: Grad[] = [], gi: Grad[] = [];
    for (let J = 0; J <= tj; J++) {
      const nn = (J + 1) * (J + 1);
      u.push({ re: new Float64Array(nn), im: new Float64Array(nn) });
      gr.push({ r: new Float64Array(nn), i: new Float64Array(nn) });
      gi.push({ r: new Float64Array(nn), i: new Float64Array(nn) });
    }
    const WT = this.tables ?? (this.tables = new WignerTables(tj));
    const gpR = new Float64Array(WT.size), gpI = new Float64Array(WT.size), cs5 = new Float64Array(5);
    const B = new Float64Array(K), Bf = new Float64Array(K), gam = new Float64Array(K), gbuf = new Float64Array(K);
    let evdwl = 0;
    let mx = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > mx) mx = list.numneigh[i];
    const nJ = new Int32Array(mx), nDx = new Float64Array(mx), nDy = new Float64Array(mx), nDz = new Float64Array(mx);
    const nR = new Float64Array(mx), nRc = new Float64Array(mx), nTh = new Float64Array(mx);
    const nSc = new Float64Array(mx), nDsc = new Float64Array(mx);
    const dp = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
    // per-atom virial (compute stress/atom): each bond term d = x_j - x_i with G0 = dE_i/dd gives the pair
    // virial -d (x) G0, half to i and half to j. Measured with native LAMMPS (black box): the stress/atom values
    // of a 3-atom, 2-type triclinic system match this split atom by atom to 1e-15, and they sum to the global
    // virial (oracle cases w19vatom_snap and w19vatom_nn).
    const va = pc.vatom;

    for (let i = 0; i < nlocal; i++) {
      const ei = this.elemOf[type[i]];
      if (ei < 0) continue;
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const ej = this.elemOf[type[j]];
        if (ej < 0) continue;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const Rii = rc * (d.radius[ei] + d.radius[ej]);
        if (!(r < Rii) || r === 0) continue;
        const span = Rii - rmin0;
        const th = (rfac0 * Math.PI * (r - rmin0)) / span;
        const sw = d.switchflag && r >= rmin0;
        const fc = sw ? 0.5 * (libm.cos((Math.PI * (r - rmin0)) / span) + 1) : 1;
        const dfc = sw ? -0.5 * libm.sin((Math.PI * (r - rmin0)) / span) * (Math.PI / span) : 0;
        nJ[m] = j; nDx[m] = dx; nDy[m] = dy; nDz[m] = dz;
        nR[m] = r; nRc[m] = Rii; nTh[m] = th;
        nSc[m] = fc * d.weight[ej];
        nDsc[m] = dfc * d.weight[ej];
        m++;
      }

      // pass 1: u^J = identity (central atom) + sum_j f_c w_j U^J(a_j, b_j), then B
      for (let J = 0; J <= tj; J++) {
        const nn = J + 1;
        u[J].re.fill(0); u[J].im.fill(0);
        for (let q = 0; q < nn; q++) u[J].re[q * nn + q] = 1;
      }
      for (let a = 0; a < m; a++) {
        const r = nR[a];
        const sn = libm.sin(nTh[a]), cs = libm.cos(nTh[a]);
        const sg = sn < 0 ? -1 : 1;
        const S = sg * sn, C = sg * cs;
        const g = S / r;
        const ar = C, ai = g * nDz[a], br = g * nDy[a], bi = g * nDx[a];
        WT.addScaled(ar, ai, br, bi, nSc[a], u);
      }
      WignerTables.mirror(u);
      rawBispectrum(triples, u, B, zs);
      for (let c = 0; c < K; c++) Bf[c] = (B[c] - d.b0[c]) / d.norm[c];

      const E = model.energy(ei, Bf, gam);
      evdwl += E;
      if (pc.eatom) pc.eatom[i] += E;
      if (m === 0) continue;

      // adjoint of E_i with respect to the real and imaginary parts of every u^J entry
      for (let c = 0; c < K; c++) gbuf[c] = gam[c] / d.norm[c];
      for (let J = 0; J <= tj; J++) {
        gr[J].r.fill(0); gr[J].i.fill(0); gi[J].r.fill(0); gi[J].i.fill(0);
      }
      adjointBispectrum(triples, u, gbuf, gr, gi, zs);
      WT.polyGradient(gr, gi, gpR, gpI);

      // pass 2: dE_i/dd for every neighbour (d = x_j - x_i); F_j = -G, F_i = +G
      for (let a = 0; a < m; a++) {
        const j = nJ[a];
        const dx = nDx[a], dy = nDy[a], dz = nDz[a];
        const r = nR[a], Rii = nRc[a];
        const sn = libm.sin(nTh[a]), cs = libm.cos(nTh[a]);
        const sg = sn < 0 ? -1 : 1;
        const S = sg * sn, C = sg * cs;
        const g = S / r;
        const nx = dx / r, ny = dy / r, nz = dz / r;
        const nm = [nx, ny, nz];
        const thp = (rfac0 * Math.PI) / (Rii - rmin0);
        const gp = (C * thp * r - S) / (r * r);
        const vz = [0, 0, 1], vy = [0, 1, 0], vx = [1, 0, 0];
        for (let mm = 0; mm < 3; mm++) {
          dp[0][mm] = -S * thp * nm[mm];
          dp[1][mm] = gp * nm[mm] * dz + g * vz[mm];
          dp[2][mm] = gp * nm[mm] * dy + g * vy[mm];
          dp[3][mm] = gp * nm[mm] * dx + g * vx[mm];
        }
        const ar = C, ai = g * dz, br = g * dy, bi = g * dx;
        const sc = nSc[a], dsc = nDsc[a];
        // as pair snap: the gradient contracted over every entry first (WignerTables.contract)
        WT.contract(ar, ai, br, bi, gpR, gpI, cs5);
        const G0 = [0, 0, 0];
        for (let mm = 0; mm < 3; mm++) {
          G0[mm] = dsc * nm[mm] * cs5[0] + sc * (cs5[1] * dp[0][mm] + cs5[2] * dp[1][mm] + cs5[3] * dp[2][mm] + cs5[4] * dp[3][mm]);
        }
        f[3 * j] -= G0[0];
        f[3 * j + 1] -= G0[1];
        f[3 * j + 2] -= G0[2];
        f[3 * i] += G0[0];
        f[3 * i + 1] += G0[1];
        f[3 * i + 2] += G0[2];
        if (va) {
          const w0 = -0.5 * dx * G0[0], w1 = -0.5 * dy * G0[1], w2 = -0.5 * dz * G0[2];
          const w3 = -0.5 * dx * G0[1], w4 = -0.5 * dx * G0[2], w5 = -0.5 * dy * G0[2];
          va[6 * i] += w0; va[6 * i + 1] += w1; va[6 * i + 2] += w2; va[6 * i + 3] += w3; va[6 * i + 4] += w4; va[6 * i + 5] += w5;
          va[6 * j] += w0; va[6 * j + 1] += w1; va[6 * j + 2] += w2; va[6 * j + 3] += w3; va[6 * j + 4] += w4; va[6 * j + 5] += w5;
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }

  /**
   * Descriptor sna with chemflag 1: the multi-element bispectrum of src/engine/compute/snapChem.ts (the code
   * of pair_style snap's chemflag path), and the same energy and force conventions as the non-chem sna path.
   * The model gives E_i and gam = dE_i/dB (linear: beta_0 + beta . B, as in the pair_style snap chem energy).
   */
  private computeChem(pc: PairCompute, d: MliapSnaDescriptor, model: MliapModel): void {
    const list = pc.full;
    if (!list) throw new Error('pair style mliap needs a full neighbor list');
    const { x, f, type } = pc;
    const nbase = d.nmodel;
    if (!this.chemEng) {
      this.chemEng = new ChemSna({
        triples: d.triples, twojmax: d.twojmax, nelem: d.elems.length, rfac0: d.rfac0, rmin0: d.rmin0,
        switchflag: d.switchflag, wselfall: d.wselfall, normOf: d.norm, b0: d.b0,
      }, this.tables ?? (this.tables = new WignerTables(d.twojmax)));
    }
    const chem = this.chemEng;
    const Bf = new Float64Array(nbase), gam = new Float64Array(nbase);
    const G0 = [0, 0, 0];
    let evdwl = 0;
    let mx = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > mx) mx = list.numneigh[i];
    chem.reserve(mx);
    const va = pc.vatom;
    for (let i = 0; i < pc.nlocal; i++) {
      const ei = this.elemOf[type[i]];
      if (ei < 0) continue;
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      chem.m = 0;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const ej = this.elemOf[type[j]];
        if (ej < 0) continue;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        chem.add(j, ej, dx, dy, dz, d.rcutfac * (d.radius[ei] + d.radius[ej]), d.weight[ej]);
      }
      chem.bispectrum(ei, Bf);
      const E = model.energy(ei, Bf, gam);
      evdwl += E;
      if (pc.eatom) pc.eatom[i] += E;
      const m = chem.m;
      if (m === 0) continue;
      chem.adjoint(gam);
      for (let a = 0; a < m; a++) {
        const j = chem.j[a];
        chem.force(a, G0);
        f[3 * j] -= G0[0];
        f[3 * j + 1] -= G0[1];
        f[3 * j + 2] -= G0[2];
        f[3 * i] += G0[0];
        f[3 * i + 1] += G0[1];
        f[3 * i + 2] += G0[2];
        if (va) {
          const dx = chem.dx[a], dy = chem.dy[a], dz = chem.dz[a];
          const w0 = -0.5 * dx * G0[0], w1 = -0.5 * dy * G0[1], w2 = -0.5 * dz * G0[2];
          const w3 = -0.5 * dx * G0[1], w4 = -0.5 * dx * G0[2], w5 = -0.5 * dy * G0[2];
          va[6 * i] += w0; va[6 * i + 1] += w1; va[6 * i + 2] += w2; va[6 * i + 3] += w3; va[6 * i + 4] += w4; va[6 * i + 5] += w5;
          va[6 * j] += w0; va[6 * j + 1] += w1; va[6 * j + 2] += w2; va[6 * j + 3] += w3; va[6 * j + 4] += w4; va[6 * j + 5] += w5;
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }

  /** Descriptor so3 path: same energy and force conventions as the sna path (F_j = -G_j, F_i = +sum G_j). */
  private computeAce(pc: PairCompute, ace: MliapAceDescriptor, model: MliapModel): void {
    const list = pc.full;
    if (!list) throw new Error('pair style mliap needs a full neighbor list');
    const { x, f, type } = pc;
    const K = ace.ndescriptors;
    const gam = new Float64Array(K);
    const va = pc.vatom;
    const nbrs = ace.nativeNeighbors(pc, list);
    let jl = new Int32Array(64);
    let evdwl = 0;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const ei = this.elemOf[type[i]];
      if (ei < 0) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      if (jl.length < k1 - k0) jl = new Int32Array(k1 - k0);
      // MLIAPData::generate_neighdata: the neighbours inside cutsq of the two elements, in list order
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        const ej = this.elemOf[type[j]];
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (ej >= 0 && rsq < ace.cutsq(ei, ej)) jl[m++] = j;
      }
      const { B, dB } = ace.evaluate(i, x, type, m, jl);
      const E = model.energy(ei, B, gam);
      evdwl += E;
      if (pc.eatom) pc.eatom[i] += E;
      // compute_forces: beta_k dB_k/dr_j to atom i and the opposite to j, then v_tally of -r_ij (x) f_ij
      for (let jj = 0; jj < m; jj++) {
        const j = jl[jj];
        let fx = 0, fy = 0, fz = 0;
        for (let k = 0; k < K; k++) {
          const o = (k * m + jj) * 3;
          const g = gam[k];
          const gx = dB[o] * g, gy = dB[o + 1] * g, gz = dB[o + 2] * g;
          fx += gx; fy += gy; fz += gz;
          f[3 * i] += gx; f[3 * i + 1] += gy; f[3 * i + 2] += gz;
          f[3 * j] -= gx; f[3 * j + 1] -= gy; f[3 * j + 2] -= gz;
        }
        if (va) {
          const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
          const v0 = -dx * fx, v1 = -dy * fy, v2 = -dz * fz, v3 = -dx * fy, v4 = -dx * fz, v5 = -dy * fz;
          va[6 * i] += 0.5 * v0; va[6 * i + 1] += 0.5 * v1; va[6 * i + 2] += 0.5 * v2;
          va[6 * i + 3] += 0.5 * v3; va[6 * i + 4] += 0.5 * v4; va[6 * i + 5] += 0.5 * v5;
          va[6 * j] += 0.5 * v0; va[6 * j + 1] += 0.5 * v1; va[6 * j + 2] += 0.5 * v2;
          va[6 * j + 3] += 0.5 * v3; va[6 * j + 4] += 0.5 * v4; va[6 * j + 5] += 0.5 * v5;
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }

  private computeSo3(pc: PairCompute, eng: So3Engine, d: So3Descriptor, model: MliapModel): void {
    const list = pc.full;
    if (!list) throw new Error('pair style mliap needs a full neighbor list');
    const { x, f, type } = pc;
    const K = eng.K;
    const B = new Float64Array(K), gam = new Float64Array(K);
    let nb = new Float64Array(3 * 64), wt = new Float64Array(64), G = new Float64Array(3 * 64);
    let evdwl = 0;
    const va = pc.vatom;
    for (let i = 0; i < pc.nlocal; i++) {
      if (this.elemOf[type[i]] < 0) continue;
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const js: number[] = [];
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const ej = this.elemOf[type[j]];
        if (ej < 0) continue;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (!(r < d.rcutfac) || r === 0) continue;
        if (3 * (m + 1) > nb.length) {
          const nb2 = new Float64Array(2 * nb.length); nb2.set(nb); nb = nb2;
          const wt2 = new Float64Array(2 * wt.length); wt2.set(wt); wt = wt2;
          G = new Float64Array(2 * G.length);
        }
        nb[3 * m] = dx; nb[3 * m + 1] = dy; nb[3 * m + 2] = dz;
        wt[m] = d.weight[ej];
        js.push(j);
        m++;
      }
      const ei = this.elemOf[type[i]];
      const nbv = nb.subarray(0, 3 * m), wv = wt.subarray(0, m);
      eng.evaluate(nbv, wv, B);
      const E = model.energy(ei, B, gam);
      evdwl += E;
      if (pc.eatom) pc.eatom[i] += E;
      if (m === 0) continue;
      eng.evaluate(nbv, wv, B, gam, G.subarray(0, 3 * m));
      for (let a = 0; a < m; a++) {
        const j = js[a];
        const g0 = G[3 * a], g1 = G[3 * a + 1], g2 = G[3 * a + 2];
        f[3 * j] -= g0; f[3 * j + 1] -= g1; f[3 * j + 2] -= g2;
        f[3 * i] += g0; f[3 * i + 1] += g1; f[3 * i + 2] += g2;
        if (va) {
          const dx = nb[3 * a], dy = nb[3 * a + 1], dz = nb[3 * a + 2];
          const w0 = -0.5 * dx * g0, w1 = -0.5 * dy * g1, w2 = -0.5 * dz * g2;
          const w3 = -0.5 * dx * g1, w4 = -0.5 * dx * g2, w5 = -0.5 * dy * g2;
          va[6 * i] += w0; va[6 * i + 1] += w1; va[6 * i + 2] += w2; va[6 * i + 3] += w3; va[6 * i + 4] += w4; va[6 * i + 5] += w5;
          va[6 * j] += w0; va[6 * j + 1] += w1; va[6 * j + 2] += w2; va[6 * j + 3] += w3; va[6 * j + 4] += w4; va[6 * j + 5] += w5;
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
