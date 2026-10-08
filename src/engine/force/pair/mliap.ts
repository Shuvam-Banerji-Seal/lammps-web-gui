import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { WignerTables, rawBispectrum, adjointBispectrum, type Cmat, type Grad } from '../../compute/sna';
import { parseMliapSnaDescriptor, type MliapSnaDescriptor } from '../../mliap/descriptor';
import { parseMliapModel, type MliapModel, type ModelKind } from '../../mliap/model';

/*
 * pair_style mliap (wave 16) — docs.lammps.org/pair_mliap.html (plans/lammps-docs/pair_mliap.rst).
 *
 * Syntax (verbatim): "pair_style mliap ... keyword values ..."; "one or two keyword/value pairs must
 * be appended"; "keyword = model or descriptor or unified". This engine implements the pair
 * "model" plus "descriptor sna" combination. "model" with style linear, quadratic or nn, and
 * "descriptor" with style sna are implemented. The other styles are reported as StyleError:
 * model mliappy, descriptor so3 and ace, and the unified keyword (they need the Python or ML-PACE
 * packages or a serialized Python object).
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
 * "This pair style does not support the pair_modify shift, table, and tail options." Per-atom
 * virial (stress/atom) is a StyleError, as for pair_style snap.
 */

const DESCRIPTOR_STYLES = ['sna'];

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
  private model: MliapModel | null = null;
  /** Per atom type: element index in the descriptor, or -1 for NULL. */
  private elemOf = new Int32Array(0);
  private tables: WignerTables | null = null;

  settings(args: string[], ctx: StyleContext): void {
    if (args[0] === 'unified') throw new StyleError(`pair_style mliap: keyword 'unified' (serialized Python object) is not implemented in this engine`);
    if (args.length !== 4 && args.length !== 6) {
      throw new StyleError(`pair_style mliap needs the keywords model and descriptor (got ${args.length} arguments)`);
    }
    let kind: string | null = null, mfile = '', dstyle: string | null = null, dfile = '';
    for (let k = 0; k < args.length; k += 2) {
      const kw = args[k], v = args[k + 1];
      if (v === undefined) throw new StyleError(`pair_style mliap: keyword '${kw}' needs a style and a filename`);
      if (kw === 'model') {
        const style = args[k + 1];
        const file = args[k + 2];
        if (file === undefined) throw new StyleError(`pair_style mliap: model needs a style and a filename`);
        if (style === 'mliappy') throw new StyleError(`pair_style mliap: model mliappy needs the Python module and is not implemented in this engine`);
        if (style !== 'linear' && style !== 'quadratic' && style !== 'nn') throw new StyleError(`pair_style mliap: unknown model style '${style}'`);
        kind = style; mfile = file;
        k += 1;
      } else if (kw === 'descriptor') {
        const style = args[k + 1];
        const file = args[k + 2];
        if (file === undefined) throw new StyleError(`pair_style mliap: descriptor needs a style and a filename`);
        if (!DESCRIPTOR_STYLES.includes(style)) {
          throw new StyleError(`pair_style mliap: descriptor style '${style}' is not implemented in this engine (sna is)`);
        }
        dstyle = style; dfile = file;
        k += 1;
      } else if (kw === 'unified') {
        throw new StyleError(`pair_style mliap: keyword 'unified' (serialized Python object) is not implemented in this engine`);
      } else {
        throw new StyleError(`pair_style mliap: unknown keyword '${kw}'`);
      }
    }
    if (kind === null || dstyle === null) throw new StyleError(`pair_style mliap needs both model and descriptor keywords`);
    this.modelKind = kind as ModelKind;
    this.modelFile = mfile;
    this.desc = parseMliapSnaDescriptor(ctx.readFile(dfile), dfile);
    this.descFile = dfile;
    this.modelText = ctx.readFile(mfile);
    this.model = null;
  }

  coeff(args: string[]): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (!this.desc || !this.modelKind) throw new StyleError('pair_coeff for style mliap needs pair_style mliap first');
    if (args.length < 2 || args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style mliap must be * *');
    const elems = args.slice(2);
    if (elems.length !== this.ntypes) throw new StyleError(`pair_coeff for style mliap needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    const d = this.desc;
    this.model = parseMliapModel(this.modelKind, this.modelText, this.modelFile, d.K, d.elems.length);
    const nt = this.ntypes + 1;
    this.elemOf = new Int32Array(nt).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') throw new StyleError('NULL element mapping for pair_style mliap is not implemented in this engine');
      const idx = d.elems.indexOf(name);
      if (idx < 0) throw new StyleError(`element '${name}' is not in the mliap descriptor file ${this.descFile} (elements: ${d.elems.join(' ')})`);
      this.elemOf[t] = idx;
    }
  }

  /** Cutoff rcutfac (R_i + R_j) between mapped types. */
  initOne(i: number, j: number): number {
    if (!this.desc) throw new StyleError('pair_coeff for style mliap has not been given');
    const ei = this.elemOf[i], ej = this.elemOf[j];
    if (ei < 0 || ej < 0) return 0;
    return this.desc.rcutfac * (this.desc.radius[ei] + this.desc.radius[ej]);
  }

  compute(pc: PairCompute): void {
    if (!this.desc || !this.model) throw new StyleError('pair_coeff for style mliap has not been given');
    const d = this.desc, model = this.model;
    const list = pc.full;
    if (!list) throw new Error('pair style mliap needs a full neighbor list');
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const K = d.K, tj = d.twojmax, triples = d.triples;
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
        const fc = sw ? 0.5 * (Math.cos((Math.PI * (r - rmin0)) / span) + 1) : 1;
        const dfc = sw ? -0.5 * Math.sin((Math.PI * (r - rmin0)) / span) * (Math.PI / span) : 0;
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
        const sn = Math.sin(nTh[a]), cs = Math.cos(nTh[a]);
        const sg = sn < 0 ? -1 : 1;
        const S = sg * sn, C = sg * cs;
        const g = S / r;
        const ar = C, ai = g * nDz[a], br = g * nDy[a], bi = g * nDx[a];
        const sc = nSc[a];
        WT.compute(ar, ai, br, bi, false);
        for (let J = 0; J <= tj; J++) {
          const Ur = WT.ur[J], Ui = WT.ui[J];
          const uj = u[J];
          for (let q = 0; q < Ur.length; q++) {
            uj.re[q] += sc * Ur[q];
            uj.im[q] += sc * Ui[q];
          }
        }
      }
      rawBispectrum(triples, u, B);
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
      adjointBispectrum(triples, u, gbuf, gr, gi);

      // pass 2: dE_i/dd for every neighbour (d = x_j - x_i); F_j = -G, F_i = +G
      for (let a = 0; a < m; a++) {
        const j = nJ[a];
        const dx = nDx[a], dy = nDy[a], dz = nDz[a];
        const r = nR[a], Rii = nRc[a];
        const sn = Math.sin(nTh[a]), cs = Math.cos(nTh[a]);
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
        const G0 = [0, 0, 0];
        WT.compute(ar, ai, br, bi, true);
        for (let J = 0; J <= tj; J++) {
          const nn = (J + 1) * (J + 1);
          const Ur = WT.ur[J], Ui = WT.ui[J];
          const Dr = WT.dur[J], Di = WT.dui[J];
          const GR = gr[J], GI = gi[J];
          for (let q = 0; q < nn; q++) {
            const gR = GR.r[q], gI = GI.r[q];
            if (gR === 0 && gI === 0) continue;
            for (let mm = 0; mm < 3; mm++) {
              let dre = dsc * nm[mm] * Ur[q], dim = dsc * nm[mm] * Ui[q];
              for (let pp = 0; pp < 4; pp++) {
                dre += sc * Dr[pp][q] * dp[pp][mm];
                dim += sc * Di[pp][q] * dp[pp][mm];
              }
              G0[mm] += gR * dre + gI * dim;
            }
          }
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
}
