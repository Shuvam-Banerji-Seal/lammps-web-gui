import { WignerTables, bispectrumComponent, adjointComponentChem, mirrorGradient, newZStore, type Triple, type Cmat, type ZStore } from './sna';
import { libm } from '../libm';

/*
 * Explicit multi-element bispectrum (chemflag 1) shared by pair_style snap (src/engine/force/pair/snap.ts,
 * computeChem) and the sna descriptor of pair_style mliap with chemflag 1 (src/engine/force/pair/mliap.ts).
 * The kernels are those of the single-element form (sna.ts): the Wigner tables summed over the columns k <= J/2 and
 * mirrored, the products Z kept by the bispectrum for the adjoint, and one contraction per neighbour. Documentation: docs.lammps.org/pair_snap.html, "If *chemflag* is set to 1, then the
 * energy expression is written in terms of explicit multi-element bispectrum components indexed on ordered
 * triplets of elements"; docs.lammps.org/pair_mliap.html, the sna descriptor "is the same as that used by
 * pair_style snap", including the chem variant.
 *
 * Per atom i (element ei): the caller adds its neighbours (add), asks for the descriptor vector
 * (bispectrum), applies its own energy model and gradient, then asks for the adjoint of the model with respect
 * to the multi-element components (adjoint) and for the force on each neighbour (force).
 *
 *   u^mu = wself_{mu_i mu} U(0,0,0) + sum_{j: elem(j)=mu} f_c w_{mu_j} U(theta0, theta, phi)
 *   B^{kappa lambda mu} = sum conj(u^mu) H u^kappa u^lambda   (components in bispectrum order, see sna.ts)
 *
 * with the element blocks (kappa, lambda, mu) ordered with mu fastest, and the self term bzeroflag subtracting
 * the identity bispectrum (wselfall applies it to every block, otherwise only to the (ei, ei, ei) block).
 */

export interface ChemSnaSetup {
  triples: Triple[];
  twojmax: number;
  /** Number of elements N_elem (the descriptor has K N_elem^3 columns). */
  nelem: number;
  rfac0: number;
  rmin0: number;
  switchflag: boolean;
  wselfall: boolean;
  /** Per-component division (2j+1 with bnormflag, else 1). */
  normOf: Float64Array;
  /** Identity bispectrum subtracted with bzeroflag (zeros when bzeroflag is 0). */
  b0: Float64Array;
}

export class ChemSna {
  readonly K: number;
  /** Number of columns K N_elem^3. */
  readonly nbase: number;
  private readonly tj: number;
  private readonly triples: Triple[];
  private readonly nelem: number;
  private readonly rfac0: number;
  private readonly rmin0: number;
  private readonly switchflag: boolean;
  private readonly wselfall: boolean;
  private readonly normOf: Float64Array;
  private readonly b0: Float64Array;
  private readonly WT: WignerTables;
  private ue: Cmat[][] = [];
  private gd: Cmat[][] = [];
  /** The products Z of each element block (kappa, lambda, mu), mu fastest. */
  private readonly zs: ZStore[] = [];
  /** Each element's gradient in the layout of WignerTables.contract (set by adjoint), and one contraction. */
  private readonly gpR: Float64Array[] = [];
  private readonly gpI: Float64Array[] = [];
  private readonly cs5 = new Float64Array(5);
  private cap = 0;
  /** Number of accepted neighbours of the current atom. */
  m = 0;
  /** Per accepted neighbour: atom index, element index, displacement d = x_j - x_i, distance, cutoff, theta0, f_c * w, f_c' * w. */
  j = new Int32Array(0);
  ele = new Int32Array(0);
  dx = new Float64Array(0);
  dy = new Float64Array(0);
  dz = new Float64Array(0);
  r = new Float64Array(0);
  rc = new Float64Array(0);
  th = new Float64Array(0);
  sc = new Float64Array(0);
  dsc = new Float64Array(0);
  private readonly dp = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];

  constructor(s: ChemSnaSetup, tables: WignerTables) {
    this.K = s.triples.length;
    this.tj = s.twojmax;
    this.triples = s.triples;
    this.nelem = s.nelem;
    this.nbase = this.K * s.nelem * s.nelem * s.nelem;
    this.rfac0 = s.rfac0; this.rmin0 = s.rmin0; this.switchflag = s.switchflag; this.wselfall = s.wselfall;
    this.normOf = s.normOf; this.b0 = s.b0;
    this.WT = tables;
    for (let e = 0; e < s.nelem; e++) {
      const um: Cmat[] = [], gm: Cmat[] = [];
      for (let J = 0; J <= this.tj; J++) {
        const nn = (J + 1) * (J + 1);
        um.push({ re: new Float64Array(nn), im: new Float64Array(nn) });
        gm.push({ re: new Float64Array(nn), im: new Float64Array(nn) });
      }
      this.ue.push(um); this.gd.push(gm);
      this.gpR.push(new Float64Array(tables.size));
      this.gpI.push(new Float64Array(tables.size));
    }
    for (let b = 0; b < s.nelem ** 3; b++) this.zs.push(newZStore(s.triples));
  }

  /** Make room for mx neighbours (call once per atom before add). */
  reserve(mx: number): void {
    if (mx <= this.cap) return;
    this.cap = mx;
    this.j = new Int32Array(mx); this.ele = new Int32Array(mx);
    this.dx = new Float64Array(mx); this.dy = new Float64Array(mx); this.dz = new Float64Array(mx);
    this.r = new Float64Array(mx); this.rc = new Float64Array(mx); this.th = new Float64Array(mx);
    this.sc = new Float64Array(mx); this.dsc = new Float64Array(mx);
  }

  /**
   * Add neighbour j (element ej, displacement d = x_j - x_i, cutoff rc = rcutfac (R_i + R_j), weight w_ej).
   * Neighbours with r >= rc or r = 0 are skipped (as pair snap has always done).
   */
  add(j: number, ej: number, dx: number, dy: number, dz: number, Rii: number, weight: number): void {
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(r < Rii) || r === 0) return;
    const span = Rii - this.rmin0;
    const th = (this.rfac0 * Math.PI * (r - this.rmin0)) / span;
    const sw = this.switchflag && r >= this.rmin0;
    const fc = sw ? 0.5 * (libm.cos((Math.PI * (r - this.rmin0)) / span) + 1) : 1;
    const dfc = sw ? -0.5 * libm.sin((Math.PI * (r - this.rmin0)) / span) * (Math.PI / span) : 0;
    const m = this.m;
    this.j[m] = j; this.ele[m] = ej; this.dx[m] = dx; this.dy[m] = dy; this.dz[m] = dz;
    this.r[m] = r; this.rc[m] = Rii; this.th[m] = th;
    this.sc[m] = fc * weight;
    this.dsc[m] = dfc * weight;
    this.m = m + 1;
  }

  /** Multi-element bispectrum of the current atom (element ei), normalised, into Bf (length nbase). */
  bispectrum(ei: number, Bf: Float64Array): void {
    const m = this.m, tj = this.tj, triples = this.triples, Ne = this.nelem, K = this.K;
    const WT = this.WT, ue = this.ue;
    // self term per element: identity times wself_{mu_i,e}
    for (let e = 0; e < Ne; e++) {
      const wself = this.wselfall || e === ei ? 1 : 0;
      for (let J = 0; J <= tj; J++) {
        const nn = J + 1;
        ue[e][J].re.fill(0); ue[e][J].im.fill(0);
        if (wself !== 0) for (let q = 0; q < nn; q++) ue[e][J].re[q * nn + q] = 1;
      }
    }
    for (let a = 0; a < m; a++) {
      const r = this.r[a];
      const sn = libm.sin(this.th[a]), cs = libm.cos(this.th[a]);
      const sg = sn < 0 ? -1 : 1;
      const S = sg * sn, C = sg * cs;
      const g = S / r;
      const ar = C, ai = g * this.dz[a], br = g * this.dy[a], bi = g * this.dx[a];
      WT.addScaled(ar, ai, br, bi, this.sc[a], ue[this.ele[a]]);
    }
    for (let e = 0; e < Ne; e++) WignerTables.mirror(ue[e]);

    // raw multi-element bispectrum, blocks (κ,λ,μ) with μ fastest
    let b = 0;
    for (let k1e = 0; k1e < Ne; k1e++) {
      for (let k2e = 0; k2e < Ne; k2e++) {
        for (let k3e = 0; k3e < Ne; k3e++) {
          const bo = b * K, z = this.zs[b];
          const useB0 = this.wselfall || (k1e === ei && k2e === ei && k3e === ei);
          for (let c = 0; c < K; c++) {
            const t = triples[c];
            let v = bispectrumComponent(t, ue[k1e][t.J1], ue[k2e][t.J2], ue[k3e][t.J], z.r[c], z.i[c]);
            if (useB0) v -= this.b0[c];
            Bf[bo + c] = v / this.normOf[c];
          }
          b++;
        }
      }
    }
  }

  /** Adjoint of the model energy with respect to every element's u entries; gam = dE/dBf (length nbase). */
  adjoint(gam: Float64Array): void {
    const tj = this.tj, triples = this.triples, Ne = this.nelem, K = this.K;
    const ue = this.ue, gd = this.gd;
    for (let e = 0; e < Ne; e++) {
      for (let J = 0; J <= tj; J++) { gd[e][J].re.fill(0); gd[e][J].im.fill(0); }
    }
    let b = 0;
    for (let k1e = 0; k1e < Ne; k1e++) {
      for (let k2e = 0; k2e < Ne; k2e++) {
        for (let k3e = 0; k3e < Ne; k3e++) {
          const bo = b * K, z = this.zs[b];
          for (let c = 0; c < K; c++) {
            const g = gam[bo + c] / this.normOf[c];
            if (g === 0) continue;
            const t = triples[c];
            adjointComponentChem(t, ue[k1e][t.J1], ue[k2e][t.J2], ue[k3e][t.J], g, gd[k1e], gd[k2e], gd[k3e], z.r[c], z.i[c]);
          }
          b++;
        }
      }
    }
    for (let e = 0; e < Ne; e++) {
      for (let J = 0; J <= tj; J++) mirrorGradient(J, gd[e][J].re, gd[e][J].im);
      this.WT.polyGradient(gd[e].map((x) => x.re), gd[e].map((x) => x.im), this.gpR[e], this.gpI[e]);
    }
  }

  /**
   * dE_i/dd for neighbour a (d = x_j - x_i), after adjoint(): writes G0 (length 3). The caller applies
   * F_j = -G0 and F_i = +G0.
   */
  force(a: number, G0: number[]): void {
    const rfac0 = this.rfac0, rmin0 = this.rmin0, WT = this.WT, dp = this.dp;
    const dx = this.dx[a], dy = this.dy[a], dz = this.dz[a];
    const r = this.r[a], Rii = this.rc[a];
    const sn = libm.sin(this.th[a]), cs = libm.cos(this.th[a]);
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
    const sc = this.sc[a], dsc = this.dsc[a];
    // the element's gradient contracted with U and dU/dp over every entry first (WignerTables.contract)
    const cs5 = this.cs5, e = this.ele[a];
    WT.contract(ar, ai, br, bi, this.gpR[e], this.gpI[e], cs5);
    for (let mm = 0; mm < 3; mm++) {
      G0[mm] = dsc * nm[mm] * cs5[0] + sc * (cs5[1] * dp[0][mm] + cs5[2] * dp[1][mm] + cs5[3] * dp[2][mm] + cs5[4] * dp[3][mm]);
    }
  }
}
