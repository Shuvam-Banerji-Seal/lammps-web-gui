import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { buildTriples, WignerTables, rawBispectrum, adjointBispectrum, type Triple, type Cmat, type Grad } from '../../compute/sna';
import { parseNum } from '../util';

/*
 * pair_style snap — docs.lammps.org/pair_snap.html
 * (source: plans/lammps-docs/pair_snap.rst)
 *
 * Syntax (verbatim): "pair_style snap"
 * Usage (verbatim): "pair_coeff * * InP.snapcoeff InP.snapparam In In P P"
 *
 * "Pair style *snap* defines the spectral neighbor analysis potential
 * (SNAP), a machine-learning interatomic potential". "In SNAP, the total
 * energy is decomposed into a sum over atom energies. The energy of atom *i*
 * is expressed as a weighted sum over bispectrum components."
 *
 *   E^i_{SNAP}(B_1^i,...,B_K^i) = \beta^{\mu_i}_0 + \sum_{k=1}^K \beta_k^{\mu_i} B_k^i
 *
 * "The mathematical definition of the bispectrum calculation and its
 * derivatives w.r.t. atom positions is identical to that used by compute snap"
 * (i.e. compute sna/atom, see src/engine/compute/sna.ts for its own citations).
 *
 * Quadratic form (quadraticflag 1): "E^i_{SNAP}(\mathbf{B}^i) = \beta^{\mu_i}_0 +
 * \boldsymbol{\beta}^{\mu_i} \cdot \mathbf{B}_i + \frac{1}{2}\mathbf{B}^t_i
 * \cdot \boldsymbol{\alpha}^{\mu_i} \cdot \mathbf{B}_i". "The SNAP coefficient
 * file should contain *K*\ (\ *K*\ +1)/2 additional coefficients in each element
 * block, the upper-triangular elements of" alpha. With the native coefficient
 * order measured (below), the quadratic coefficient c_kl (k <= l) multiplies the
 * quadratic column Q_kl of compute sna/atom (Q_kk = B_k^2/2, Q_kl = B_k B_l).
 *
 * Coefficient file: "The first non-blank non-comment line must contain two
 * integers:" nelem (number of elements) and ncoeff (number of coefficients).
 * "This is followed by one block for each of the *nelem* elements." "The first
 * line of each block contains three entries:" the element name, its radius R
 * and its weight w. "This line is followed by *ncoeff* coefficients, one per
 * line."
 * Measured with native LAMMPS (black box): the first coefficient is beta_0,
 * the next K are beta_1..beta_K (multiplying B_k after the B0 subtraction),
 * then the quadratic block when quadraticflag is 1 (ncoeff = 1 + K + K(K+1)/2).
 *
 * Parameter file (verbatim): "The SNAP parameter file can contain blank and
 * comment lines (start with #) anywhere. Each non-blank non-comment line must
 * contain one keyword/value pair. The required keywords are *rcutfac* and
 * *twojmax*." The defaults listed on that page are rfac0 0.99363, rmin0 0.0,
 * switchflag 1, bzeroflag 1, quadraticflag 0, chemflag 0, bnormflag 0,
 * wselfallflag 0, switchinnerflag 0, chunksize 32768 and parallelthresh 8192.
 * "chunksize" and "parallelthresh" "are only
 * applicable when using the pair style *snap* with the KOKKOS package ... and
 * are ignored otherwise." chemflag 1 and switchinnerflag 1 are not implemented
 * here and throw StyleError.
 *
 * Cutoffs: "cutoffs for SNAP potentials are not set in the pair_style or
 * pair_coeff command; they are specified in the SNAP potential files
 * themselves." The pair cutoff between types i, j is rcutfac (R_i + R_j).
 *
 * "This pair style does not support the pair_modify shift, table, and tail
 * options." NULL element mapping: "If a SNAP mapping value is specified as NULL,
 * the mapping is not performed."
 *
 * Conventions for neighbours closer than rmin0 (theta0 < 0), measured with
 * native LAMMPS (black box, oracle w8snap_quadratic, step 11, per-atom energies
 * agree to 1e-6 only with these two rules): the Cayley-Klein parameters use the
 * positive radial normalization sqrt(r^2 + z0^2) = r / |sin theta0|, i.e.
 * a = sign(sin theta0) cos(theta0) + i z |sin theta0| / r (and likewise b), and
 * the switching factor is f_c = 1 for r < rmin0 (the cosine is only applied for
 * r >= rmin0). The same two conventions are not yet applied in compute sna/atom.
 *
 * Forces: the energy is E = sum_i E_i. The derivative of each bispectrum
 * component with respect to a neighbor displacement d = x_j - x_i is
 * evaluated analytically: the Cayley-Klein parameters of the 3-sphere point,
 * a = cos(theta0) + i sin(theta0) z/r, b = sin(theta0) (y + i x)/r, with
 * theta0(r) = rfac0 pi (r - rmin0)/(R - rmin0) and f_c(r), the Wigner matrices
 * U^j(a, b) (polynomial in a, b, conj(a), conj(b), differentiated by the product
 * rule), and the adjoint of E_i with respect to the real and imaginary parts of
 * each u^j entry. Then F_j = -dE/dd and F_i = +dE/dd, summed over the pairs.
 */

/** Parsed coefficient file: element names, radii, weights and coefficient vectors. */
interface SnapFile {
  elems: string[];
  radius: number[];
  weight: number[];
  coeff: Float64Array[];
  ncoeff: number;
}

/** Parsed parameter file. */
interface SnapParam {
  rcutfac: number;
  twojmax: number;
  rfac0: number;
  rmin0: number;
  switchflag: boolean;
  bzeroflag: boolean;
  quadraticflag: boolean;
  bnormflag: boolean;
}

const dataLines = (text: string): string[] =>
  text.split('\n').map((l) => l.replace(/#.*$/, '').trim()).filter((l) => l.length > 0);

export const parseSnapParam = (text: string, filename: string): SnapParam => {
  const p: SnapParam = { rcutfac: NaN, twojmax: NaN, rfac0: 0.99363, rmin0: 0, switchflag: true, bzeroflag: true, quadraticflag: false, bnormflag: false };
  const seen = new Set<string>();
  for (const line of dataLines(text)) {
    const w = line.split(/\s+/);
    if (w.length !== 2) throw new StyleError(`SNAP parameter file ${filename}: each line must hold one keyword/value pair (got '${line}')`);
    const [kw, v] = w;
    seen.add(kw);
    const flag = (): boolean => {
      if (v !== '0' && v !== '1') throw new StyleError(`SNAP parameter file ${filename}: ${kw} must be 0 or 1 (got '${v}')`);
      return v === '1';
    };
    switch (kw) {
      case 'rcutfac': p.rcutfac = parseNum(v, `${filename} rcutfac`); break;
      case 'twojmax': p.twojmax = parseNum(v, `${filename} twojmax`); break;
      case 'rfac0': p.rfac0 = parseNum(v, `${filename} rfac0`); break;
      case 'rmin0': p.rmin0 = parseNum(v, `${filename} rmin0`); break;
      case 'switchflag': p.switchflag = flag(); break;
      case 'bzeroflag': p.bzeroflag = flag(); break;
      case 'quadraticflag': p.quadraticflag = flag(); break;
      case 'bnormflag': p.bnormflag = flag(); break;
      case 'wselfallflag': flag(); break; // acts only with chemflag
      case 'chunksize': case 'parallelthresh': break; // KOKKOS-only, ignored
      case 'chemflag':
        if (v !== '0') throw new StyleError(`pair style snap: chemflag ${v} is not implemented in this engine`);
        break;
      case 'switchinnerflag':
        if (v !== '0') throw new StyleError(`pair style snap: switchinnerflag ${v} is not implemented in this engine`);
        break;
      default:
        throw new StyleError(`SNAP parameter file ${filename}: unknown keyword '${kw}'`);
    }
  }
  if (!seen.has('rcutfac') || !seen.has('twojmax')) throw new StyleError(`SNAP parameter file ${filename}: rcutfac and twojmax are required`);
  if (!(p.rcutfac > 0)) throw new StyleError(`SNAP parameter file ${filename}: rcutfac must be positive`);
  if (!Number.isInteger(p.twojmax) || p.twojmax < 0) throw new StyleError(`SNAP parameter file ${filename}: twojmax must be a non-negative integer`);
  return p;
};

export const parseSnapCoeff = (text: string, filename: string, K: number, quadratic: boolean): SnapFile => {
  const lines = dataLines(text);
  if (lines.length === 0) throw new StyleError(`SNAP coefficient file ${filename} is empty`);
  const head = lines[0].split(/\s+/).map(Number);
  if (head.length !== 2 || !Number.isInteger(head[0]) || !Number.isInteger(head[1]) || head[0] < 1 || head[1] < 1) {
    throw new StyleError(`SNAP coefficient file ${filename}: first line must hold nelem and ncoeff (got '${lines[0]}')`);
  }
  const [nelem, ncoeff] = head;
  const want = 1 + K + (quadratic ? (K * (K + 1)) / 2 : 0);
  if (ncoeff !== want) {
    throw new StyleError(`SNAP coefficient file ${filename}: ncoeff ${ncoeff} does not match twojmax (K = ${K} bispectrum components${quadratic ? ', quadratic terms included' : ''}): expected ${want}`);
  }
  const out: SnapFile = { elems: [], radius: [], weight: [], coeff: [], ncoeff };
  let at = 1;
  for (let e = 0; e < nelem; e++) {
    const hdr = lines[at++];
    if (hdr === undefined) throw new StyleError(`SNAP coefficient file ${filename}: missing header of element ${e + 1}`);
    const hw = hdr.split(/\s+/);
    if (hw.length !== 3) throw new StyleError(`SNAP coefficient file ${filename}: element header must be 'name R w' (got '${hdr}')`);
    out.elems.push(hw[0]);
    out.radius.push(parseNum(hw[1], `${filename} radius of ${hw[0]}`));
    out.weight.push(parseNum(hw[2], `${filename} weight of ${hw[0]}`));
    const c = new Float64Array(ncoeff);
    for (let k = 0; k < ncoeff; k++) {
      const l = lines[at++];
      if (l === undefined) throw new StyleError(`SNAP coefficient file ${filename}: element ${hw[0]} has fewer than ${ncoeff} coefficients`);
      c[k] = parseNum(l, `${filename} coefficient ${k + 1} of ${hw[0]}`);
    }
    out.coeff.push(c);
  }
  return out;
};

export class PairSnap extends Pair {
  readonly name = 'snap';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private param: SnapParam | null = null;
  private file: SnapFile | null = null;
  private triples: Triple[] = [];
  private K = 0;
  private quadratic = false;
  /** Per atom type: element index in file, or -1 for NULL. */
  private elemOf = new Int32Array(0);
  private b0: Float64Array = new Float64Array(0);
  private normOf: Float64Array = new Float64Array(0);
  private setupDone = false;
  private tables: WignerTables | null = null;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError(`pair_style snap takes no arguments (got ${args.length})`);
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError('usage: pair_coeff * * coefficient-file parameter-file elem1 ... elemN (style snap)');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style snap must be * *');
    const coefFile = args[2];
    const paramFile = args[3];
    if (paramFile === undefined) throw new StyleError('pair_coeff for style snap needs a coefficient file and a parameter file');
    const elems = args.slice(4);
    if (elems.length !== this.ntypes) throw new StyleError(`pair_coeff for style snap needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    this.param = parseSnapParam(ctx.readFile(paramFile), paramFile);
    const p = this.param;
    this.triples = buildTriples(p.twojmax);
    this.K = this.triples.length;
    this.quadratic = p.quadraticflag;
    this.file = parseSnapCoeff(ctx.readFile(coefFile), coefFile, this.K, p.quadraticflag);
    const nt = this.ntypes + 1;
    this.elemOf = new Int32Array(nt).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') { this.elemOf[t] = -1; continue; }
      const idx = this.file.elems.indexOf(name);
      if (idx < 0) throw new StyleError(`element '${name}' is not in SNAP coefficient file ${coefFile} (elements: ${this.file.elems.join(' ')})`);
      this.elemOf[t] = idx;
    }
    this.setupDone = false;
  }

  /** Cutoff rcutfac (R_i + R_j) between mapped types; 0 for NULL-mapped types. */
  initOne(i: number, j: number): number {
    if (!this.file || !this.param) throw new StyleError('pair_coeff for style snap has not been given');
    const ei = this.elemOf[i], ej = this.elemOf[j];
    if (ei < 0 || ej < 0) return 0;
    return this.param.rcutfac * (this.file.radius[ei] + this.file.radius[ej]);
  }

  override initStyle(): void {
    if (!this.param || !this.file) return;
    const p = this.param;
    // bispectrum of an atom with no neighbours (identity u matrices), raw
    const K = this.K;
    this.b0 = new Float64Array(K);
    const id: Cmat[] = [];
    for (let J = 0; J <= p.twojmax; J++) {
      const n = J + 1;
      const re = new Float64Array(n * n), im = new Float64Array(n * n);
      for (let k = 0; k < n; k++) re[k * n + k] = 1;
      id.push({ re, im });
    }
    rawBispectrum(this.triples, id, this.b0);
    if (!p.bzeroflag) this.b0.fill(0);
    this.normOf = new Float64Array(K);
    for (let c = 0; c < K; c++) this.normOf[c] = p.bnormflag ? this.triples[c].J + 1 : 1;
    this.setupDone = true;
  }

  override compute(pc: PairCompute): void {
    if (!this.param || !this.file) throw new StyleError('pair_coeff for style snap has not been given');
    if (pc.vatom) throw new StyleError('per-atom virial (compute stress/atom) is not implemented for pair style snap');
    if (!this.setupDone) this.initStyle();
    const p = this.param, file = this.file;
    const list = pc.full;
    if (!list) throw new Error('pair style snap needs a full neighbor list');
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const K = this.K, tj = p.twojmax, triples = this.triples;
    const quad = this.quadratic;
    const rfac0 = p.rfac0, rmin0 = p.rmin0, rc = p.rcutfac;

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

    for (let i = 0; i < nlocal; i++) {
      const ei = this.elemOf[type[i]];
      if (ei < 0) continue;
      const coef = file.coeff[ei];
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      // neighbours inside the element pair cutoff rcutfac (R_i + R_j)
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const ej = this.elemOf[type[j]];
        if (ej < 0) continue;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const Rii = rc * (file.radius[ei] + file.radius[ej]);
        if (!(r < Rii) || r === 0) continue;
        const span = Rii - rmin0;
        const th = (rfac0 * Math.PI * (r - rmin0)) / span;
        const sw = p.switchflag && r >= rmin0;
        const fc = sw ? 0.5 * (Math.cos((Math.PI * (r - rmin0)) / span) + 1) : 1;
        const dfc = sw ? -0.5 * Math.sin((Math.PI * (r - rmin0)) / span) * (Math.PI / span) : 0;
        nJ[m] = j; nDx[m] = dx; nDy[m] = dy; nDz[m] = dz;
        nR[m] = r; nRc[m] = Rii; nTh[m] = th;
        nSc[m] = fc * file.weight[ej];
        nDsc[m] = dfc * file.weight[ej];
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
        // Cayley-Klein parameters with the positive radial normalization (see header)
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
      for (let c = 0; c < K; c++) Bf[c] = (B[c] - this.b0[c]) / this.normOf[c];

      // energy E_i = beta_0 + beta . B + quadratic, and gam = dE_i/dBf
      let E = coef[0];
      for (let c = 0; c < K; c++) {
        E += coef[1 + c] * Bf[c];
        gam[c] = coef[1 + c];
      }
      if (quad) {
        let q = 1 + K;
        for (let kk = 0; kk < K; kk++) {
          for (let ll = kk; ll < K; ll++) {
            const cq = coef[q++];
            if (kk === ll) {
              E += cq * 0.5 * Bf[kk] * Bf[kk];
              gam[kk] += cq * Bf[kk];
            } else {
              E += cq * Bf[kk] * Bf[ll];
              gam[kk] += cq * Bf[ll];
              gam[ll] += cq * Bf[kk];
            }
          }
        }
      }
      evdwl += E;
      if (pc.eatom) pc.eatom[i] += E;
      if (m === 0) continue;

      // adjoint of E_i with respect to the real and imaginary parts of every u^J entry
      for (let c = 0; c < K; c++) gbuf[c] = gam[c] / this.normOf[c];
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
        const gp = (C * thp * r - S) / (r * r); // d(S / r)/dr with dS/dr = C theta0'
        // dp[p][m] = d(parameter p)/d(d_m), parameters (ar, ai, br, bi)
        const vz = [0, 0, 1], vy = [0, 1, 0], vx = [1, 0, 0];
        for (let mm = 0; mm < 3; mm++) {
          dp[0][mm] = -S * thp * nm[mm]; // d(C)/dr = -S theta0'
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
              // d u / d d_m = dsc n_m U + sc sum_p dU_p dp_p/dd_m
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
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
