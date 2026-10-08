import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style vashishta — docs.lammps.org/pair_vashishta.html
 * (source: plans/lammps-docs/pair_vashishta.rst)
 *
 * Quoted sentences below are copied from the .rst with its reST role markup
 * (:math:, :doc:) removed and line breaks joined.
 *
 * Syntax (verbatim):
 *
 *    pair_style style args
 *
 * * style = *vashishta* or *vashishta/table*
 * * args = list of arguments for a particular style
 *
 *   *vashishta* args = none
 *   *vashishta/table* args = Ntable cutinner
 *     Ntable = # of tabulation points
 *     cutinner = tablulate from cutinner to cutoff
 *
 * Energy (verbatim LaTeX from the doc page):
 *
 *    U & =  \sum_i^N \sum_{j > i}^N U_{ij}^{(2)} (r_{ij}) + \sum_i^N \sum_{j \neq i}^N \sum_{k > j, k \neq i}^N U_{ijk}^{(3)} (r_{ij}, r_{ik}, \theta_{ijk}) \\
 *    U_{ij}^{(2)} (r) & =   \frac{H_{ij}}{r^{\eta_{ij}}} + \frac{Z_i Z_j}{r}\exp(-r/\lambda_{1,ij}) - \frac{D_{ij}}{r^4}\exp(-r/\lambda_{4,ij}) - \frac{W_{ij}}{r^6}, r < r_{c,{ij}} \\
 *    U_{ijk}^{(3)}(r_{ij},r_{ik},\theta_{ijk}) & =  B_{ijk} \frac{\left[ \cos \theta_{ijk} - \cos \theta_{0ijk} \right]^2} {1+C_{ijk}\left[ \cos \theta_{ijk} - \cos \theta_{0ijk} \right]^2} \times \\
 *                     &  \exp \left( \frac{\gamma_{ij}}{r_{ij} - r_{0,ij}} \right) \exp \left( \frac{\gamma_{ik}}{r_{ik} - r_{0,ik}} \right), r_{ij} < r_{0,ij}, r_{ik} < r_{0,ik}
 *
 * "The summation over two-body terms is over all neighbors J within
 * a cutoff distance = :math:`r_c`.  The twobody terms are shifted and
 * tilted by a linear function so that the energy and force are
 * both zero at :math:`r_c`. The summation over three-body terms
 * is over all neighbors *i* and *k* within a cut-off distance :math:`= r_0`,
 * where the exponential screening function becomes zero."
 *
 * Potential file (verbatim): "Lines that are not blank or comments
 * (starting with #) define parameters for a triplet of elements."  One entry
 * is "element 1 (the center atom in a 3-body interaction)", element 2,
 * element 3, then H, eta, Zi, Zj, lambda1, D, lambda4, W, rc, B, gamma, r0,
 * C, costheta0.  Two-body parameters for atoms I and J "are taken from the
 * IJJ entry, where the second and third elements are the same" ("the
 * two-body parameters for Si interacting with C come from the SiCC entry");
 * "the three-body function U3 above contains the two-body parameters
 * :math:`\gamma` and :math:`r_0`. So U3 for a central C atom bonded to
 * an Si atom and a second C atom will take three-body parameters from
 * the CSiC entry, but two-body parameters from the CCC and CSiSi entries."  Three-body parameters "for a
 * central atom I and two neighbors J and K are taken from the IJK entry".
 *
 * pair_coeff (verbatim): "Only a single pair_coeff command is used with
 * either style which specifies a Vashishta potential file with parameters
 * for all needed elements."  "The first 2 arguments must be \* \* so as to
 * span all LAMMPS atom types."  "If a mapping value is specified as NULL,
 * the mapping is not performed."
 *
 * "This pair style does not support the pair_modify shift, table, and tail
 * options."  "These pair styles requires the newton setting to be "on" for
 * pair interactions" (the engine always evaluates each physical pair once).
 * The vashishta/table variant is not implemented: constructing it throws a
 * StyleError naming the style instead of silently approximating the
 * tabulated evaluation.
 *
 * Coulomb prefactor: the doc lists Z_i and Z_j in "(electron charge units)",
 * so the screened Coulomb term carries the e^2/(4 pi eps_0) prefactor of the
 * metal unit system, qqr2e = 14.399645 eV*Angstrom (the metal row of the
 * unit table in src/engine/units.ts).  The doc says the potential files are
 * "parameterized for metal units" and gives no conversion, so the constant is
 * used as is.  Measured with native LAMMPS (black box): the oracle cases
 * tests/oracle/w5vash_si and w5vash_six agree to the printed precision with
 * this value.
 *
 * Cutoffs: the neighbor-list cutoff of a type pair covers the two-body rc of
 * both ordered entries (EI,EJ,EJ) and (EJ,EI,EI) and the three-body leg
 * cutoffs r0 of both legs (center I -> neighbor J and center J -> neighbor
 * I); beyond r0 the leg exponential is zero (between r0 and r0 + skin it
 * would overflow, so the leg is dead past r0 exactly).
 *
 * Full-list scheme (as in pair_style sw): the two-body term is summed over
 * the full neighbor list with half the pair energy and half the pair force
 * per visit (every physical pair is visited from both ends, each side using
 * its own ordered (EI,EJ,EJ) entry); the three-body term is summed per owned
 * center atom over unordered neighbor pairs and counted exactly once, with
 * its forces on the two neighbors added at their (possibly ghost) indices so
 * the force field folds them back to the owners.  Per-atom energy and virial
 * follow compute_stress_atom.html: "The total contribution for the cluster
 * interaction is divided evenly among those atoms."
 */

/** One entry (element triplet) of a Vashishta potential file. */
interface VEntry {
  e1: string;
  e2: string;
  e3: string;
  H: number;
  eta: number;
  Zi: number;
  Zj: number;
  lam1: number;
  D: number;
  lam4: number;
  W: number;
  rc: number;
  B: number;
  gam: number;
  r0: number;
  C: number;
  costheta0: number;
}

/**
 * e^2/(4 pi eps_0) in eV*Angstrom (metal units, qqr2e of src/engine/units.ts);
 * see the file header comment.
 */
const QQ = 14.399645;

const key3 = (e1: string, e2: string, e3: string): string => `${e1} ${e2} ${e3}`;

/** Parses a .vashishta potential file: entry lines of 3 element names + 14 numbers. */
const parseVFile = (text: string, fileName: string): { entries: Map<string, VEntry>; elems: Set<string> } => {
  const lines = text.split(/\r?\n/);
  const entries = new Map<string, VEntry>();
  const elems = new Set<string>();
  for (let ln = 0; ln < lines.length; ln++) {
    const raw = lines[ln];
    const hash = raw.indexOf('#');
    const line = (hash >= 0 ? raw.slice(0, hash) : raw).trim();
    if (line === '') continue;
    const t = line.split(/\s+/);
    if (t.length !== 17) {
      throw new StyleError(
        `Vashishta potential file ${fileName} line ${ln + 1}: expected 'element1 element2 element3 H eta Zi Zj lambda1 D lambda4 W rc B gamma r0 C costheta0' (17 values), got ${t.length}`,
      );
    }
    const k = key3(t[0], t[1], t[2]);
    if (entries.has(k)) throw new StyleError(`Vashishta potential file ${fileName} line ${ln + 1}: duplicate entry for elements ${k}`);
    const nums: number[] = [];
    for (let c = 3; c < 17; c++) nums.push(parseNum(t[c], `parameter ${c - 2} of the Vashishta file entry ${k}`));
    const e: VEntry = {
      e1: t[0], e2: t[1], e3: t[2],
      H: nums[0], eta: nums[1], Zi: nums[2], Zj: nums[3], lam1: nums[4],
      D: nums[5], lam4: nums[6], W: nums[7], rc: nums[8],
      B: nums[9], gam: nums[10], r0: nums[11], C: nums[12], costheta0: nums[13],
    };
    if (e.rc < 0 || e.r0 < 0) {
      throw new StyleError(`Vashishta potential file ${fileName} entry ${k}: rc and r0 must be >= 0`);
    }
    entries.set(k, e);
    elems.add(t[0]);
    elems.add(t[1]);
    elems.add(t[2]);
  }
  if (entries.size === 0) throw new StyleError(`Vashishta potential file ${fileName} contains no parameter entries`);
  return { entries, elems };
};

/**
 * pair_style vashishta — the analytic Vashishta potential (2-body
 * repulsion + screened Coulomb + screened charge-dipole + dispersion, with a
 * Stillinger-Weber-like 3-body bond-angle term), shifted and tilted to zero
 * energy and force at the two-body cutoff.
 */
export class PairVashishta extends Pair {
  readonly name = 'vashishta';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private fileName = '';
  private fileRead = false;
  private entries = new Map<string, VEntry>();
  private fileElems = new Set<string>();
  private elemOf = new Int32Array(0);
  private elemNames: string[] = [];
  private mappingSet = false;

  // per ordered type pair (i,j): two-body parameters of the (EI,EJ,EJ) entry
  // plus the shift/tilt of U2 at its rc, plus the leg screening (gamma, r0)
  private tbH = new Float64Array(0);
  private tbEta = new Float64Array(0);
  private tbZpq = new Float64Array(0);
  private tbLam1 = new Float64Array(0);
  private tbD = new Float64Array(0);
  private tbLam4 = new Float64Array(0);
  private tbW = new Float64Array(0);
  private tbRc = new Float64Array(0);
  private tbU2rc = new Float64Array(0);
  private tbDuRc = new Float64Array(0);
  private tbGam = new Float64Array(0);
  private tbR0 = new Float64Array(0);
  // three-body parameters of the (EI,EJ,EK) entry, tables of size nt^3
  private tB = new Float64Array(0);
  private tC = new Float64Array(0);
  private tCos0 = new Float64Array(0);

  // per-center scratch for the gathered three-body legs
  private cap = 0;
  private sIdx = new Int32Array(0);
  private sDx = new Float64Array(0);
  private sDy = new Float64Array(0);
  private sDz = new Float64Array(0);
  private sR = new Float64Array(0);
  private sInvR = new Float64Array(0);
  private sES = new Float64Array(0);
  private sW = new Float64Array(0);

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) {
      throw new StyleError(`usage: pair_style ${this.name} (no arguments; got ${args.length})`);
    }
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.elemOf = new Int32Array(nt).fill(-1);
    this.elemNames = [];
    this.mappingSet = false;
    this.tbH = new Float64Array(nt * nt);
    this.tbEta = new Float64Array(nt * nt);
    this.tbZpq = new Float64Array(nt * nt);
    this.tbLam1 = new Float64Array(nt * nt);
    this.tbD = new Float64Array(nt * nt);
    this.tbLam4 = new Float64Array(nt * nt);
    this.tbW = new Float64Array(nt * nt);
    this.tbRc = new Float64Array(nt * nt);
    this.tbU2rc = new Float64Array(nt * nt);
    this.tbDuRc = new Float64Array(nt * nt);
    this.tbGam = new Float64Array(nt * nt);
    this.tbR0 = new Float64Array(nt * nt);
    this.tB = new Float64Array(nt * nt * nt);
    this.tC = new Float64Array(nt * nt * nt);
    this.tCos0 = new Float64Array(nt * nt * nt);
    this.fileRead = false;
    this.entries.clear();
    this.fileElems.clear();
    this.fileName = '';
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 2 + 1 + this.ntypes) {
      throw new StyleError(
        `usage: pair_coeff * * filename elem1 ... elem${this.ntypes} (style ${this.name} needs one element name per atom type)`,
      );
    }
    if (args[0] !== '*' || args[1] !== '*') {
      throw new StyleError(`the first 2 arguments of pair_coeff for style ${this.name} must be * *`);
    }
    const filename = args[2];
    if (!this.fileRead) {
      const text = ctx.readFile(filename);
      const parsed = parseVFile(text, filename);
      this.entries = parsed.entries;
      this.fileElems = parsed.elems;
      this.fileName = filename;
      this.fileRead = true;
    } else if (filename !== this.fileName) {
      throw new StyleError(`all pair_coeff commands for style ${this.name} must use the same potential file (first: ${this.fileName})`);
    }
    for (let t = 1; t <= this.ntypes; t++) {
      const name = args[2 + t];
      let idx: number;
      if (name === 'NULL') {
        idx = -1;
      } else {
        if (!this.fileElems.has(name)) {
          throw new StyleError(
            `element '${name}' is not in Vashishta potential file ${this.fileName} (elements: ${[...this.fileElems].sort().join(' ')})`,
          );
        }
        idx = this.elemNames.indexOf(name);
        if (idx < 0) {
          idx = this.elemNames.length;
          this.elemNames.push(name);
        }
      }
      this.elemOf[t] = idx;
    }
    this.mappingSet = true;
  }

  private entry(k: string, what: string): VEntry {
    const e = this.entries.get(k);
    if (!e) {
      throw new StyleError(`Vashishta potential file ${this.fileName} has no entry for elements ${k} (${what})`);
    }
    return e;
  }

  /** U2(r) of one ordered entry with the Coulomb prefactor folded in. */
  private u2Raw(e: VEntry, r: number): number {
    const coul = e.Zi * e.Zj * QQ * Math.exp(-r / e.lam1) / r;
    return e.H / Math.pow(r, e.eta) + coul - e.D / Math.pow(r, 4) * Math.exp(-r / e.lam4) - e.W / Math.pow(r, 6);
  }

  /** dU2/dr of one ordered entry (same prefactor). */
  private du2Raw(e: VEntry, r: number): number {
    const coul = -e.Zi * e.Zj * QQ * Math.exp(-r / e.lam1) * (1 / (r * r) + 1 / (e.lam1 * r));
    return (
      -e.eta * e.H / Math.pow(r, e.eta + 1) + coul +
      e.D * Math.exp(-r / e.lam4) * (4 / Math.pow(r, 5) + 1 / (e.lam4 * Math.pow(r, 4))) +
      6 * e.W / Math.pow(r, 7)
    );
  }

  override initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify shift and tail options`);
    }
    if (this.table !== 12) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify table option`);
    }
    if (!this.fileRead) {
      throw new StyleError(`pair_style ${this.name} needs a pair_coeff command with a Vashishta potential file`);
    }
    // three-body parameters for every type triple with mapped elements; the
    // entry is "taken from the IJK entry" (center element first)
    const nt = this.ntypes + 1;
    for (let ti = 1; ti <= this.ntypes; ti++) {
      if (this.elemOf[ti] < 0) continue;
      for (let tj = 1; tj <= this.ntypes; tj++) {
        if (this.elemOf[tj] < 0) continue;
        for (let tk = 1; tk <= this.ntypes; tk++) {
          if (this.elemOf[tk] < 0) continue;
          const k = key3(this.elemNames[this.elemOf[ti]], this.elemNames[this.elemOf[tj]], this.elemNames[this.elemOf[tk]]);
          const e = this.entry(k, `needed by the three-body combination of types ${ti} ${tj} ${tk}`);
          const t3 = (ti * nt + tj) * nt + tk;
          this.tB[t3] = e.B;
          this.tC[t3] = e.C;
          this.tCos0[t3] = e.costheta0;
        }
      }
    }
  }

  override initOne(i: number, j: number): number {
    const ei = this.elemOf[i];
    const ej = this.elemOf[j];
    if (ei < 0 || ej < 0) return 0;
    const nt = this.ntypes + 1;
    const e1 = this.elemNames[ei];
    const e2 = this.elemNames[ej];
    // ordered entries: two body of (i,j) from (EI,EJ,EJ), of (j,i) from (EJ,EI,EI);
    // the leg (center i -> neighbor j) takes gamma and r0 from the (EI,EJ,EJ) entry
    const eIJ = this.entry(key3(e1, e2, e2), `needed by the two-body interaction of types ${i} ${j}`);
    const eJI = this.entry(key3(e2, e1, e1), `needed by the two-body interaction of types ${j} ${i}`);
    const fill = (a: number, b: number, e: VEntry): void => {
      const t = a * nt + b;
      this.tbH[t] = e.H;
      this.tbEta[t] = e.eta;
      this.tbZpq[t] = e.Zi * e.Zj * QQ;
      this.tbLam1[t] = e.lam1;
      this.tbD[t] = e.D;
      this.tbLam4[t] = e.lam4;
      this.tbW[t] = e.W;
      this.tbRc[t] = e.rc;
      this.tbGam[t] = e.gam;
      this.tbR0[t] = e.r0;
      if (e.rc > 0) {
        this.tbU2rc[t] = this.u2Raw(e, e.rc);
        this.tbDuRc[t] = this.du2Raw(e, e.rc);
      } else {
        this.tbU2rc[t] = 0;
        this.tbDuRc[t] = 0;
      }
    };
    fill(i, j, eIJ);
    fill(j, i, eJI);
    // the neighbor list must cover both two-body cutoffs and both legs' r0
    return Math.max(eIJ.rc, eJI.rc, eIJ.r0, eJI.r0);
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const elemOf = this.elemOf;
    const eatom = pc.eatom;
    const vatom = pc.vatom;
    let evdwl = 0;

    let maxn = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.sIdx = new Int32Array(maxn);
      this.sDx = new Float64Array(maxn);
      this.sDy = new Float64Array(maxn);
      this.sDz = new Float64Array(maxn);
      this.sR = new Float64Array(maxn);
      this.sInvR = new Float64Array(maxn);
      this.sES = new Float64Array(maxn);
      this.sW = new Float64Array(maxn);
    }
    const nIdx = this.sIdx, nDx = this.sDx, nDy = this.sDy, nDz = this.sDz;
    const nR = this.sR, nInvR = this.sInvR, nES = this.sES, nW = this.sW;

    for (let i = 0; i < nlocal; i++) {
      const tiv = type[i];
      if (elemOf[tiv] < 0) continue;
      const ti = tiv * nt;
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      let m = 0;
      let fxi = 0, fyi = 0, fzi = 0;
      // pass 1: two-body terms (half per visit; the pair is seen again from the other
      // end, using its own ordered entry) and gather of the three-body legs
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dxx = x[3 * j] - x[3 * i];
        const dyy = x[3 * j + 1] - x[3 * i + 1];
        const dzz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dxx * dxx + dyy * dyy + dzz * dzz;
        const tj = type[j];
        const t = ti + tj;
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        const rc = this.tbRc[t];
        if (r < rc) {
          // shifted and tilted two-body: U2(r) - U2(rc) - (r - rc) * U2'(rc)
          const H = this.tbH[t], eta = this.tbEta[t];
          const zpq = this.tbZpq[t], lam1 = this.tbLam1[t];
          const D = this.tbD[t], lam4 = this.tbLam4[t], W = this.tbW[t];
          const u2rc = this.tbU2rc[t], duRc = this.tbDuRc[t];
          const scr = Math.exp(-r / lam1);
          const phi2 =
            H / Math.pow(r, eta) + zpq * scr / r -
            D / Math.pow(r, 4) * Math.exp(-r / lam4) - W / Math.pow(r, 6);
          const du =
            -eta * H / Math.pow(r, eta + 1) - zpq * scr * (1 / (r * r) + 1 / (lam1 * r)) +
            D * Math.exp(-r / lam4) * (4 / Math.pow(r, 5) + 1 / (lam4 * Math.pow(r, 4))) +
            6 * W / Math.pow(r, 7) - duRc;
          const phi2s = phi2 - u2rc - (r - rc) * duRc;
          const fpair = -du / r;
          const hx = 0.5 * fpair * dxx;
          const hy = 0.5 * fpair * dyy;
          const hz = 0.5 * fpair * dzz;
          fxi -= hx;
          fyi -= hy;
          fzi -= hz;
          f[3 * j] += hx;
          f[3 * j + 1] += hy;
          f[3 * j + 2] += hz;
          evdwl += 0.5 * phi2s;
          if (eatom) {
            const e4 = 0.25 * phi2s;
            eatom[i] += e4;
            eatom[j] += e4;
          }
          if (vatom) {
            const vc = 0.25 * fpair;
            const v0 = vc * dxx * dxx, v1 = vc * dyy * dyy, v2 = vc * dzz * dzz;
            const v3 = vc * dxx * dyy, v4 = vc * dxx * dzz, v5 = vc * dyy * dzz;
            vatom[6 * i] += v0; vatom[6 * i + 1] += v1; vatom[6 * i + 2] += v2;
            vatom[6 * i + 3] += v3; vatom[6 * i + 4] += v4; vatom[6 * i + 5] += v5;
            vatom[6 * j] += v0; vatom[6 * j + 1] += v1; vatom[6 * j + 2] += v2;
            vatom[6 * j + 3] += v3; vatom[6 * j + 4] += v4; vatom[6 * j + 5] += v5;
          }
        }
        // three-body leg center i -> neighbor j: screening exp(gamma/(r - r0)) of
        // the ordered (EI,EJ,EJ) entry; zero at and beyond r0
        const r0 = this.tbR0[t];
        if (r < r0) {
          const g = this.tbGam[t];
          const d2 = r - r0;
          nIdx[m] = j;
          nDx[m] = dxx; nDy[m] = dyy; nDz[m] = dzz;
          nR[m] = r;
          nInvR[m] = 1 / r;
          nES[m] = Math.exp(g / d2);
          nW[m] = g / (d2 * d2);
          m++;
        }
      }
      // pass 2: three-body terms, once per (center i; unordered neighbor pair)
      if (m > 1) {
        const t3base = tiv * nt * nt;
        for (let p1 = 0; p1 < m; p1++) {
          const j = nIdx[p1];
          const tjv = type[j];
          const d1x = nDx[p1], d1y = nDy[p1], d1z = nDz[p1];
          const invrj = nInvR[p1], esj = nES[p1], wj = nW[p1];
          const base1 = t3base + tjv * nt;
          for (let p2 = p1 + 1; p2 < m; p2++) {
            const k = nIdx[p2];
            const t3 = base1 + type[k];
            const B = this.tB[t3];
            if (B === 0) continue;
            const d2x = nDx[p2], d2y = nDy[p2], d2z = nDz[p2];
            const invrk = nInvR[p2], esk = nES[p2], wk = nW[p2];
            const c = (d1x * d2x + d1y * d2y + d1z * d2z) * invrj * invrk;
            const dl = c - this.tCos0[t3];
            const C = this.tC[t3];
            const den = 1 + C * dl * dl;
            const ang = dl * dl / den;
            const P = B * esj * esk;
            const e3v = P * ang;
            evdwl += e3v;
            // fj = aj*d1 - bk*d2, fk = ak*d2 - bk*d1, fi = -(fj + fk), with
            // dU3/dcos = B * 2*delta / den^2 * esj * esk
            const zf = P * 2 * dl / (den * den);
            const bk = zf * invrj * invrk;
            const aj = zf * c * invrj * invrj + e3v * wj * invrj;
            const ak = zf * c * invrk * invrk + e3v * wk * invrk;
            const fxj = aj * d1x - bk * d2x;
            const fyj = aj * d1y - bk * d2y;
            const fzj = aj * d1z - bk * d2z;
            const fxk = ak * d2x - bk * d1x;
            const fyk = ak * d2y - bk * d1y;
            const fzk = ak * d2z - bk * d1z;
            f[3 * j] += fxj;
            f[3 * j + 1] += fyj;
            f[3 * j + 2] += fzj;
            f[3 * k] += fxk;
            f[3 * k + 1] += fyk;
            f[3 * k + 2] += fzk;
            fxi -= fxj + fxk;
            fyi -= fyj + fyk;
            fzi -= fzj + fzk;
            if (eatom) {
              const e9 = e3v / 3;
              eatom[i] += e9;
              eatom[j] += e9;
              eatom[k] += e9;
            }
            if (vatom) {
              const va = vatom;
              const w0 = (d1x * fxj + d2x * fxk) / 3;
              const w1 = (d1y * fyj + d2y * fyk) / 3;
              const w2 = (d1z * fzj + d2z * fzk) / 3;
              const w3 = (d1x * fyj + d2x * fyk) / 3;
              const w4 = (d1x * fzj + d2x * fzk) / 3;
              const w5 = (d1y * fzj + d2y * fzk) / 3;
              va[6 * i] += w0; va[6 * i + 1] += w1; va[6 * i + 2] += w2;
              va[6 * i + 3] += w3; va[6 * i + 4] += w4; va[6 * i + 5] += w5;
              va[6 * j] += w0; va[6 * j + 1] += w1; va[6 * j + 2] += w2;
              va[6 * j + 3] += w3; va[6 * j + 4] += w4; va[6 * j + 5] += w5;
              va[6 * k] += w0; va[6 * k + 1] += w1; va[6 * k + 2] += w2;
              va[6 * k + 3] += w3; va[6 * k + 4] += w4; va[6 * k + 5] += w5;
            }
          }
        }
      }
      f[3 * i] += fxi;
      f[3 * i + 1] += fyi;
      f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }
}

/**
 * pair_style vashishta/table — the tabulated variant of the Vashishta
 * potential ("*vashishta/table* args = Ntable cutinner").  Tabulated
 * evaluation is not implemented in this engine: constructing and using it
 * throws a StyleError naming the style instead of silently falling back to
 * the analytic form.
 */
export class PairVashishtaTable extends Pair {
  readonly name = 'vashishta/table';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 2) {
      throw new StyleError(`usage: pair_style ${this.name} Ntable cutinner`);
    }
    throw new StyleError(`pair_style ${this.name} is not supported by this engine (only the analytic vashishta style is)`);
  }

  override coeff(_args: string[], _ctx: StyleContext): void {
    throw new StyleError(`pair_style ${this.name} is not supported by this engine (only the analytic vashishta style is)`);
  }

  override initOne(_i: number, _j: number): number {
    throw new StyleError(`pair_style ${this.name} is not supported by this engine (only the analytic vashishta style is)`);
  }

  override compute(_pc: PairCompute): void {
    throw new StyleError(`pair_style ${this.name} is not supported by this engine (only the analytic vashishta style is)`);
  }
}
