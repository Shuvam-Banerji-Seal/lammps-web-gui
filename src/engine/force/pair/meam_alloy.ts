import { referenceVectors, type ReferenceLattice } from './meam_lattice';

/*
 * Multi-element MEAM (pair_style meam, two or more elements). Energy and analytic forces for the
 * subset verified against native LAMMPS (black box):
 *   - fcc and dia single-element references (the homonuclear pair term phi_ii and the embedding
 *     reference rho_ref,i use the element's own lattice, as in meam.ts; z is the library coordination);
 *   - heteronuclear pairs with lattce(I,J) = b1 (rock salt) or dia (diamond/zincblende);
 *   - ibar = 0, t0 = 1, rozero = 1, zbl = 0, default Cmin/Cmax, no nn2/delta;
 *   - erose_form 0 (with attrac = repuls = 0), 1 and 2 with per-pair attrac(I,J)/repuls(I,J)
 *     (pairErose below; erose_form 0 with nonzero attrac/repuls is refused by the parser).
 *
 * Documented model (docs.lammps.org/pair_meam.html, plans/lammps-docs/pair_meam.rst):
 *   "E = \sum_i \left\{ F_i(\bar{\rho}_i)"   (energy: embedding F_i plus half the pair sum, as on that page)
 *   "lattce(I,J) = lattice structure of I-J reference structure:"
 *   "b1  = rock salt (NaCl structure)"
 *   "dia = diamond (interlaced fcc for alloy)"
 *   "Ec(I,J)     = cohesive energy of reference structure for I-J mixture"
 *
 * Measured with native LAMMPS (black box), all with the synthetic entries of tests/oracle/w15meam_alloy_*:
 *  - B1 crystal (8 atoms, lattce(1,2) = b1): pe per atom equals the Rose energy of the B1 pair at the
 *    nearest-neighbour distance re(1,2) to 1e-11 relative, for three lattice constants;
 *  - A-B dimers agree with the model below at distances 2.2, 2.6 and 3.0 A to 4e-12 (eV);
 *  - A-A and B-B dimers agree with the single-element model to 1e-12 (eV);
 *  - A-B-A, B-A-A-B and A-A-B-B clusters agree to 1e-10 (eV); the averaging of t uses the weights w_j a0_j
 *    of each neighbour (unweighted averaging differs by about 1e-5 eV on these clusters);
 *  - the partial density of a neighbour of element j uses beta_j and re(j,j) (the element's own re).
 *
 * Measured with native LAMMPS (black box) for the w29meama_erose2 entries (B1 alloy, erose_form = 2,
 * per-pair attrac/repuls): the pair term uses the same erose expression as the single-element path
 * (pairErose reproduces eroseE/eroseDeriv of meam.ts exactly), and A-B, A-A and B-B dimers, the A-B-A
 * trimer and an A-B-A-A cluster agree with native to 1e-13 (eV); the displaced B1 crystal (tests/oracle/
 * w29meama_erose2) agrees over 40 nve steps at rel = 1e-6 including the forces.
 *
 * Measured with native LAMMPS (black box) for the w29meama_dia entries (zincblende alloy, lattce(1,2) = dia,
 * fcc/dia elements): phi_ij(r) = (2/4)(erose_ij(r) - (F_i(rho_bar_i) + F_j(rho_bar_j))/2), with rho_bar_i the
 * background of element i in the diamond reference (4 unlike neighbours at r, the same-sublattice shell at
 * r sqrt(8/3) screened to zero); A-B, A-A and B-B dimers, an A-B-A trimer and an A-B-A-A cluster agree with
 * native to 1e-13 (eV), and the displaced zincblende crystal (tests/oracle/w29meama_dia) agrees over 40 nve
 * steps at rel = 1e-6 including the forces.
 *
 * Not verified (and therefore rejected with a StyleError in meam.ts): lattce(I,J) = l12 and other names
 * (the L12 pair term is not the B1/dia form, see the remaining issues of the meam15 report), bcc/hcp/sc
 * elements in an alloy, non-default Cmin/Cmax (per-triplet entries), delta, nn2, and erose_form 0 with
 * nonzero attrac/repuls.
 */

export interface AlloyElement {
  /** library coordination z of the element's own reference structure */
  z: number;
  lat: ReferenceLattice;
  /** single-element equilibrium distance re(i,i) */
  re: number;
  /** alpha(i,i) */
  alpha: number;
  /** Ec(i,i) */
  Ec: number;
  /** asub */
  A: number;
  /** beta0..beta3 of the element */
  beta: [number, number, number, number];
  /** t0..t3 of the element, t0 = 1, t1 already augmented by 3/5 t3 when augt1 = 1 */
  t: [number, number, number, number];
  /** ibar of the element (library entry; default 0); selects G(Gamma) through gOfIbar */
  ibar?: number;
}

/**
 * G(Gamma) and dG/dGamma of the ibar forms. Docs (pair_meam.rst): "0 => G = sqrt(1+Gamma)", "1 => G = exp(Gamma/2)",
 * "3 => G = 2/(1+exp(-Gamma))", "-5 => G = +-sqrt(abs(1+Gamma))". The sign of the -5 form is measured in the tests.
 */
export const gOfIbar = (ibar: number, gamma: number): number => {
  if (ibar === 0 || ibar === 4) return Math.sqrt(1 + gamma);
  if (ibar === 1) return Math.exp(gamma / 2);
  if (ibar === 3) return 2 / (1 + Math.exp(-gamma));
  // The -5 form is signed; a symmetric reference lattice has Gamma = 0 only up to rounding, so a
  // tiny negative value must not flip the sign (the reference density would go negative and the
  // embedding would vanish). Measured with native LAMMPS (black box): the sc Bi reference crystal
  // (Gamma = 0) has the +sqrt branch, giving the Rose reference energy.
  if (ibar === -5) return (gamma > -1e-12 ? 1 : -1) * Math.sqrt(Math.abs(1 + gamma));
  throw new Error(`MEAM ibar ${ibar} is not supported`);
};

/** dG/dGamma for the same ibar forms as gOfIbar. */
export const gPrimeOfIbar = (ibar: number, gamma: number): number => {
  if (ibar === 0 || ibar === 4) return 1 / (2 * Math.sqrt(1 + gamma));
  if (ibar === 1) return Math.exp(gamma / 2) / 2;
  if (ibar === 3) {
    // 2 exp(-Gamma) / (1 + exp(-Gamma))^2 is even in Gamma; with exp(-|Gamma|) neither factor overflows, so a
    // large |Gamma| (a far reference shell of the 2NN series, where rho0 is tiny) gives 0, not Infinity/Infinity.
    const e = Math.exp(-Math.abs(gamma));
    return (2 * e) / ((1 + e) * (1 + e));
  }
  if (ibar === -5) {
    const s = gamma > -1e-12 ? 1 : -1;
    const s2 = 1 + gamma >= 0 ? 1 : -1;
    return (s * s2) / (2 * Math.sqrt(Math.abs(1 + gamma)));
  }
  throw new Error(`MEAM ibar ${ibar} is not supported`);
};

export interface AlloyPair {
  Ec: number;
  re: number;
  alpha: number;
  /** 'self' for i = j (the element's own lattice), 'b1' (rock salt) or 'dia' (diamond/zincblende) for i != j */
  lat: 'self' | 'b1' | 'dia';
  /** attrac(I,J) and repuls(I,J) of the I-J pair (docs pair_meam.rst); default 0 */
  attrac?: number;
  repuls?: number;
}

export interface AlloyOptions {
  rc: number;
  delr: number;
  Cmin: number;
  Cmax: number;
  /** erose_form of the potential (docs pair_meam.rst); default 0. Only 0, 1 and 2 are supported. */
  eroseForm?: number;
}

export interface AlloyModel {
  elements: AlloyElement[];
  /** pairs[i][j] (symmetric) */
  pairs: AlloyPair[][];
  opts: AlloyOptions;
  /** element-wise t vectors used in the density, precomputed */
  tEff: Array<[number, number, number, number]>;
  /** rho_ref,i: own-lattice background density at re(i,i) (embedding normalisation) */
  rhoRef: number[];
}

/** A neighbour of the central atom: element index, atom index, separation vector and length. */
export interface AlloyNeighbor {
  e: number;
  j: number;
  dx: number;
  dy: number;
  dz: number;
  r: number;
}

const polyW = (x: number): number => {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const q = (1 - x) ** 4;
  return (1 - q) * (1 - q);
};
const polyP = (x: number): number => {
  if (x <= 0 || x >= 1) return 0;
  const q = (1 - x) ** 4;
  return 8 * (1 - x) ** 3 * (1 - q);
};
const fcW = (r: number, o: AlloyOptions): number => polyW((o.rc - r) / o.delr);
const fcP = (r: number, o: AlloyOptions): number => polyP((o.rc - r) / o.delr) * (-1 / o.delr);
const screenW = (C: number, o: AlloyOptions): number => (C >= o.Cmax ? 1 : polyW((C - o.Cmin) / (o.Cmax - o.Cmin)));
const screenP = (C: number, o: AlloyOptions): number =>
  C >= o.Cmax || C <= o.Cmin ? 0 : polyP((C - o.Cmin) / (o.Cmax - o.Cmin)) / (o.Cmax - o.Cmin);
const ebound = (o: AlloyOptions): number => (o.Cmax * o.Cmax) / (4 * (o.Cmax - 1));

/** Screening S_m and the pair factors f_mk with their C-derivatives (same formulas as meam.ts). */
interface Screen {
  S: Float64Array;
  f: Float64Array;
  cA: Float64Array;
  cB: Float64Array;
}

const screenAll = (nb: Array<{ dx: number; dy: number; dz: number; r: number }>, o: AlloyOptions): Screen => {
  const N = nb.length;
  const eb = ebound(o);
  const f = new Float64Array(N * N).fill(1);
  const cA = new Float64Array(N * N), cB = new Float64Array(N * N);
  for (let m = 0; m < N; m++) {
    const p = nb[m];
    const rm2 = p.r * p.r;
    for (let k = 0; k < N; k++) {
      if (k === m) continue;
      const q = nb[k];
      const A = (q.r * q.r) / rm2;
      const ex = q.dx - p.dx, ey = q.dy - p.dy, ez = q.dz - p.dz;
      const B = (ex * ex + ey * ey + ez * ez) / rm2;
      if (A > eb || B > eb) continue;
      const dd = A - B;
      const Dn = 1 - dd * dd;
      const Nn = 2 * (A + B) - dd * dd - 1;
      const C = Nn / Dn;
      const sp = screenP(C, o);
      f[m * N + k] = screenW(C, o);
      cA[m * N + k] = (sp * ((2 - 2 * dd) * Dn + 2 * dd * Nn)) / (Dn * Dn);
      cB[m * N + k] = (sp * ((2 + 2 * dd) * Dn - 2 * dd * Nn)) / (Dn * Dn);
    }
  }
  const S = new Float64Array(N);
  for (let m = 0; m < N; m++) {
    let s = 1;
    for (let k = 0; k < N; k++) if (k !== m) s *= f[m * N + k];
    S[m] = s;
  }
  return { S, f, cA, cB };
};

/** Prefix/suffix products R[m,k] = prod_{l != m,k} f[m,l] (used by the screening chain rule). */
const otherScreens = (N: number, f: Float64Array): Float64Array => {
  const R = new Float64Array(N * N);
  for (let m = 0; m < N; m++) {
    const pre = new Float64Array(N);
    let p = 1;
    for (let k = 0; k < N; k++) {
      pre[k] = p;
      if (k !== m) p *= f[m * N + k];
    }
    let s = 1;
    for (let k = N - 1; k >= 0; k--) {
      R[m * N + k] = pre[k] * s;
      if (k !== m) s *= f[m * N + k];
    }
  }
  return R;
};

/** Per-neighbour density term: weight W = fc S, atomic densities A_n = a_n(r) of the neighbour's element. */
interface Term {
  e: number;
  W: number;
  A: [number, number, number, number];
  u: [number, number, number];
}

/**
 * Background density rho_bar = rho0 G(Gamma) of one atom from its density terms, with the partial derivatives
 * needed for forces, for an embedding weight P = dF/drho_bar:
 *   rho0 = sum W A0,  T_l = sum W A0 t_l(e),  v1 = sum W A1 u,  s2 = sum W A2,  V2 = sum W A2 uu,
 *   v3 = sum W A3 u,  V3 = sum W A3 uuu,  q1 = |v1|^2, q2 = |V2|^2 - s2^2/3, q3 = |V3|^2 - 3/5 |v3|^2,
 *   Gamma = (T1 q1 + T2 q2 + T3 q3) / rho0^3 (t_l = T_l / rho0 is the density-weighted average),  G = sqrt(1+Gamma).
 * Returns gW_m = d(P rho_bar)/dW_m, gA_m[n] = d(P rho_bar)/dA_n,m and gU_m = d(P rho_bar)/du_m.
 */
export const densityPartials = (terms: Term[], tEff: Array<[number, number, number, number]>, P: number, ibar: number) => {
  const N = terms.length;
  const out = { rb: 0, rho0: 0, gW: new Float64Array(N), gA: new Float64Array(4 * N), gU: new Float64Array(3 * N) };
  let S0 = 0, s2 = 0;
  const ST = [0, 0, 0, 0];
  const v1 = [0, 0, 0], v3 = [0, 0, 0], V2 = new Float64Array(9), V3 = new Float64Array(27);
  for (const t of terms) {
    const te = tEff[t.e];
    const { W, A, u } = t;
    S0 += W * A[0];
    for (let l = 1; l < 4; l++) ST[l] += W * A[0] * te[l];
    for (let c = 0; c < 3; c++) { v1[c] += W * A[1] * u[c]; v3[c] += W * A[3] * u[c]; }
    s2 += W * A[2];
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) V2[3 * p + q] += W * A[2] * u[p] * u[q];
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) for (let s = 0; s < 3; s++) V3[9 * p + 3 * q + s] += W * A[3] * u[p] * u[q] * u[s];
  }
  out.rho0 = S0;
  if (S0 <= 0) return out;
  const q1 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
  let V2sq = 0;
  for (let q = 0; q < 9; q++) V2sq += V2[q] * V2[q];
  const q2 = V2sq - (s2 * s2) / 3;
  let V3sq = 0;
  for (let q = 0; q < 27; q++) V3sq += V3[q] * V3[q];
  const q3 = V3sq - (3 / 5) * (v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2]);
  const qs = [0, q1, q2, q3];
  const Nn = ST[1] * q1 + ST[2] * q2 + ST[3] * q3;
  const g = Nn / (S0 * S0 * S0);
  const G = gOfIbar(ibar, g), Gp = gPrimeOfIbar(ibar, g);
  out.rb = S0 * G;
  const PG = P * Gp;
  const dS0 = P * (G - 3 * g * Gp);
  const dST = [0, 0, 0, 0], dq = [0, 0, 0, 0];
  for (let l = 1; l < 4; l++) {
    dST[l] = (PG * qs[l]) / (S0 * S0);
    dq[l] = (PG * ST[l]) / (S0 * S0);
  }
  // derivatives of the moment sums (dE/dv1, dE/ds2, dE/dV2, dE/dv3, dE/dV3)
  const M1 = [0, 1, 2].map((c) => dq[1] * 2 * v1[c]);
  const ms2 = dq[2] * (-2 * s2) / 3;
  const M2 = new Float64Array(9);
  for (let q = 0; q < 9; q++) M2[q] = dq[2] * 2 * V2[q];
  const M3v = [0, 1, 2].map((c) => dq[3] * (-6 / 5) * v3[c]);
  const M3 = new Float64Array(27);
  for (let q = 0; q < 27; q++) M3[q] = dq[3] * 2 * V3[q];
  for (let m = 0; m < N; m++) {
    const { e, W, A, u } = terms[m];
    const te = tEff[e];
    const uu = (p: number, q: number) => u[p] * u[q];
    let uM2u = 0; // sum M2_pq u_p u_q
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) uM2u += M2[3 * p + q] * uu(p, q);
    let uM3uu = 0; // sum M3_pqs u_p u_q u_s
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) for (let s = 0; s < 3; s++) uM3uu += M3[9 * p + 3 * q + s] * u[p] * u[q] * u[s];
    let M1u = 0, M3vu = 0;
    for (let c = 0; c < 3; c++) { M1u += M1[c] * u[c]; M3vu += M3v[c] * u[c]; }
    // dE/dW_m (through rho0, T, v1, s2, V2, v3, V3)
    let gWm = dS0 * A[0] + dST[1] * A[0] * te[1] + dST[2] * A[0] * te[2] + dST[3] * A[0] * te[3];
    gWm += M1u * A[1] + ms2 * A[2] + M3vu * A[3];
    gWm += A[2] * uM2u + A[3] * uM3uu;
    out.gW[m] = gWm;
    // dE/dA_n,m
    out.gA[4 * m] = dS0 * W + W * (dST[1] * te[1] + dST[2] * te[2] + dST[3] * te[3]);
    out.gA[4 * m + 1] = W * M1u;
    out.gA[4 * m + 2] = ms2 * W + W * uM2u;
    out.gA[4 * m + 3] = W * M3vu + W * uM3uu;
    // dE/du_m
    for (let c = 0; c < 3; c++) {
      let gc = W * A[1] * M1[c] + W * A[3] * M3v[c];
      let m2 = 0, m3 = 0;
      for (let q = 0; q < 3; q++) m2 += M2[3 * c + q] * u[q];
      for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) m3 += M3[9 * c + 3 * p + q] * u[p] * u[q];
      gc += 2 * W * A[2] * m2 + 3 * W * A[3] * m3;
      out.gU[3 * m + c] = gc;
    }
  }
  return out;
};

/** Embedding F(rho) = A Ec x ln x with x = rho / rho_ref (zero for rho <= 0, as meam.ts embedding()). */
const embedF = (el: AlloyElement, rhoRef: number, rb: number): number => {
  if (rb <= 0) return 0;
  const x = rb / rhoRef;
  return el.A * el.Ec * x * Math.log(x);
};
const embedFp = (el: AlloyElement, rhoRef: number, rb: number): number => {
  if (rb <= 0) return 0;
  const x = rb / rhoRef;
  return (el.A * el.Ec * (Math.log(x) + 1)) / rhoRef;
};

/**
 * Element-i density terms for a neighbour list at distance scale: W = fc S, A_n = exp(-beta_n (r/re_e - 1)).
 * radial = false gives the reference-structure weights W = S (no radial cutoff; see the reference note in meam.ts).
 */
const termsOf = (model: AlloyModel, list: AlloyNeighbor[], radial = true): Term[] => {
  const sc = screenAll(list, model.opts);
  return list.map((p, m) => {
    const el = model.elements[p.e];
    const W = (radial ? fcW(p.r, model.opts) : 1) * sc.S[m];
    const A: [number, number, number, number] = [0, 0, 0, 0];
    for (let n = 0; n < 4; n++) A[n] = Math.exp(-el.beta[n] * (p.r / el.re - 1));
    return { e: p.e, W, A, u: [p.dx / p.r, p.dy / p.r, p.dz / p.r] };
  });
};

/** Background density of a central element from a list; returns rho_bar and d rho_bar / d r (lists scale with r). */
const scaledRho = (
  model: AlloyModel,
  list: AlloyNeighbor[],
  r: number,
  ibar: number,
  radial = true,
): { rho: number; drho: number; rho0: number } => {
  const terms = termsOf(model, list, radial);
  const part = densityPartials(terms, model.tEff, 1, ibar);
  let drho = 0;
  const sc = screenAll(list, model.opts);
  for (let m = 0; m < list.length; m++) {
    const p = list[m];
    const el = model.elements[p.e];
    const sm = p.r / r;
    // d(fc S)/dr = fc'(r_m) s_m S_m (screening depends on ratios only); d a_n/dr = -(beta_n/re) a_n s_m
    if (radial) drho += part.gW[m] * fcP(p.r, model.opts) * sc.S[m] * sm;
    for (let n = 0; n < 4; n++) drho += part.gA[4 * m + n] * (-(el.beta[n] / el.re)) * terms[m].A[n] * sm;
  }
  return { rho: part.rb, drho, rho0: part.rho0 };
};

/** Neighbour list of the element c's own reference lattice at nearest-neighbour distance r (all neighbours are c). */
const ownList = (model: AlloyModel, c: number, r: number): AlloyNeighbor[] =>
  referenceVectors(model.elements[c].lat, r, model.opts.rc).map((v) => ({ e: c, j: -1, dx: v.dx, dy: v.dy, dz: v.dz, r: v.r }));

/** B1 (rock salt) list for central element c with partner p at nearest-neighbour distance r. */
const b1List = (model: AlloyModel, c: number, p: number, r: number): AlloyNeighbor[] => {
  const out: AlloyNeighbor[] = [];
  const m = Math.ceil(model.opts.rc / r) + 2;
  for (let i = -m; i <= m; i++)
    for (let j = -m; j <= m; j++)
      for (let k = -m; k <= m; k++) {
        if (!i && !j && !k) continue;
        const dx = i * r, dy = j * r, dz = k * r, rr = Math.hypot(dx, dy, dz);
        if (rr >= model.opts.rc) continue;
        const e = (i + j + k) & 1 ? p : c;
        out.push({ e, j: -1, dx, dy, dz, r: rr });
      }
  return out;
};

/**
 * Diamond (zincblende) list for central element c with partner p at nearest-neighbour distance r: the two
 * interpenetrating fcc sublattices of diamond, the central atom on the c sublattice, the four nearest
 * neighbours on the p sublattice (the diamond reference of the I-J pair, "dia = diamond (interlaced fcc for
 * alloy)" on docs.lammps.org/pair_meam.html). Geometry as in referenceVectors('dia') of meam_lattice.ts.
 */
const diaList = (model: AlloyModel, c: number, p: number, r: number): AlloyNeighbor[] => {
  const out: AlloyNeighbor[] = [];
  const a = (4 * r) / Math.sqrt(3);
  const h = a / 2;
  const s = a / 4;
  const m = Math.ceil(model.opts.rc / h) + 1;
  for (let i = -m; i <= m; i++)
    for (let j = -m; j <= m; j++)
      for (let k = -m; k <= m; k++) {
        if (((i + j + k) & 1) !== 0 || (i === 0 && j === 0 && k === 0)) continue;
        const dx = i * h, dy = j * h, dz = k * h;
        const rr = Math.hypot(dx, dy, dz);
        if (rr < model.opts.rc) out.push({ e: c, j: -1, dx, dy, dz, r: rr });
      }
  for (let i = -m; i <= m; i++)
    for (let j = -m; j <= m; j++)
      for (let k = -m; k <= m; k++) {
        if (((i + j + k) & 1) !== 0) continue;
        const dx = i * h + s, dy = j * h + s, dz = k * h + s;
        const rr = Math.hypot(dx, dy, dz);
        if (rr < model.opts.rc) out.push({ e: p, j: -1, dx, dy, dz, r: rr });
      }
  return out;
};

/**
 * Rose reference energy erose(r) and its r-derivative of one I-J pair, with the I-J attrac/repuls. Docs
 * (docs.lammps.org/pair_meam.html, plans/lammps-docs/pair_meam.rst):
 *   "astar = alpha \* (r/re - 1.d0)"
 *   "if erose_form = 0: erose = -Ec\*(1+astar+a3\*(astar\*\*3)/(r/re))\*exp(-astar)"
 *   "if erose_form = 1: erose = -Ec\*(1+astar+(-attrac+repuls/r)\*(astar\*\*3))\*exp(-astar)"
 *   "if erose_form = 2: erose = -Ec\*(1 +astar + a3\*(astar\*\*3))\*exp(-astar)"
 *   "a3 = repuls, astar < 0"
 *   "a3 = attrac, astar >= 0"
 * form 0 with attrac = repuls = 0 is the plain -Ec (1 + astar) exp(-astar) used by the verified alloy subset.
 */
export const pairErose = (pr: AlloyPair, form: number, r: number): { E: number; dE: number } => {
  const q = r / pr.re;
  const s = pr.alpha * (q - 1);
  const sp = pr.alpha / pr.re;
  const attrac = pr.attrac ?? 0;
  const repuls = pr.repuls ?? 0;
  const a3 = s < 0 ? repuls : attrac;
  let T: number, Tp: number;
  if (form === 0) {
    T = (a3 * s ** 3) / q;
    Tp = a3 * ((3 * s * s * sp) / q - s ** 3 / (pr.re * q * q));
  } else if (form === 1) {
    T = (-attrac + repuls / r) * s ** 3;
    Tp = (-repuls / (r * r)) * s ** 3 + (-attrac + repuls / r) * 3 * s * s * sp;
  } else {
    T = a3 * s ** 3;
    Tp = a3 * 3 * s * s * sp;
  }
  return { E: -pr.Ec * (1 + s + T) * Math.exp(-s), dE: -pr.Ec * Math.exp(-s) * (Tp - (s + T) * sp) };
};

/** Pair term phi_ij(r) and its derivative (homonuclear: fcc/bcc/dia reference; heteronuclear: B1 or dia). */
export const alloyPair = (model: AlloyModel, i: number, j: number, r: number): { phi: number; dphi: number } => {
  const pr = model.pairs[i][j];
  const { E: Eu, dE: dEu } = pairErose(pr, model.opts.eroseForm ?? 0, r);
  if (pr.lat === 'self') {
    const el = model.elements[i];
    const { rho, drho } = scaledRho(model, ownList(model, i, r), r, model.elements[i].ibar ?? 0, false);
    const Fv = embedF(el, model.rhoRef[i], rho);
    const Fp = embedFp(el, model.rhoRef[i], rho);
    return { phi: (2 / el.z) * (Eu - Fv), dphi: (2 / el.z) * (dEu - Fp * drho) };
  }
  if (pr.lat === 'dia') {
    const ri = scaledRho(model, diaList(model, i, j, r), r, model.elements[i].ibar ?? 0, false);
    const rj = scaledRho(model, diaList(model, j, i, r), r, model.elements[j].ibar ?? 0, false);
    const Fi = embedF(model.elements[i], model.rhoRef[i], ri.rho);
    const Fj = embedF(model.elements[j], model.rhoRef[j], rj.rho);
    const Fip = embedFp(model.elements[i], model.rhoRef[i], ri.rho);
    const Fjp = embedFp(model.elements[j], model.rhoRef[j], rj.rho);
    // Diamond reference: z = 4 unlike neighbours, two atoms per cell, so phi = (2/4)(Eu - (Fi+Fj)/2).
    return {
      phi: (2 / 4) * (Eu - (Fi + Fj) / 2),
      dphi: (2 / 4) * (dEu - (Fip * ri.drho + Fjp * rj.drho) / 2),
    };
  }
  const ri = scaledRho(model, b1List(model, i, j, r), r, model.elements[i].ibar ?? 0, false);
  const rj = scaledRho(model, b1List(model, j, i, r), r, model.elements[j].ibar ?? 0, false);
  const Fi = embedF(model.elements[i], model.rhoRef[i], ri.rho), Fj = embedF(model.elements[j], model.rhoRef[j], rj.rho);
  const Fip = embedFp(model.elements[i], model.rhoRef[i], ri.rho), Fjp = embedFp(model.elements[j], model.rhoRef[j], rj.rho);
  return {
    phi: (2 / 6) * (Eu - (Fi + Fj) / 2),
    dphi: (2 / 6) * (dEu - (Fip * ri.drho + Fjp * rj.drho) / 2),
  };
};

/** Builds the model: t-vectors (t1 augmented when augt1) and the own-lattice reference densities. */
export const makeAlloyModel = (elements: AlloyElement[], pairs: AlloyPair[][], opts: AlloyOptions, augt1: boolean): AlloyModel => {
  const tEff = elements.map((el) => [1, augt1 ? el.t[1] + 0.6 * el.t[3] : el.t[1], el.t[2], el.t[3]] as [number, number, number, number]);
  const model: AlloyModel = { elements, pairs, opts, tEff, rhoRef: [] };
  // embedding normalisation: rho0 at re without G(Gamma), as in meam.ts referenceBackground
  model.rhoRef = elements.map((el, c) => scaledRho({ ...model, rhoRef: [] }, ownList(model, c, el.re), el.re, el.ibar ?? 0).rho0);
  return model;
};

/**
 * Energy of one atom ci with neighbour list nb (within rc) and dE_ci / d(d_m) for each neighbour vector d_m
 * (written to g, length 3N). Includes the atom's half pair sum; forces on the neighbours are -g, on ci +g.
 */
export function alloyAtomEnergyGrad(model: AlloyModel, ci: number, nb: AlloyNeighbor[], g: Float64Array): number {
  const N = nb.length;
  g.fill(0, 0, 3 * N);
  if (N === 0) return 0;
  const o = model.opts;
  const el = model.elements[ci];
  const terms = termsOf(model, nb);
  const sc = screenAll(nb, o);
  const rb0 = densityPartials(terms, model.tEff, 1, el.ibar ?? 0).rb;
  const rhoRef = model.rhoRef[ci];
  const Fv = embedF(el, rhoRef, rb0);
  const P = embedFp(el, rhoRef, rb0);
  const part = densityPartials(terms, model.tEff, P, el.ibar ?? 0);
  // pair terms
  const phi = new Float64Array(N), dphi = new Float64Array(N);
  let pairE = 0;
  for (let m = 0; m < N; m++) {
    const pv = alloyPairTab(model, ci, nb[m].e, nb[m].r);
    phi[m] = pv.phi;
    dphi[m] = pv.dphi;
    pairE += 0.5 * terms[m].W * pv.phi;
  }
  // total energy derivative wrt W_m (density plus half pair term)
  const gWt = new Float64Array(N);
  for (let m = 0; m < N; m++) gWt[m] = part.gW[m] + 0.5 * phi[m];
  // direct (radial and angular) contributions for each neighbour vector
  for (let m = 0; m < N; m++) {
    const p = nb[m], W = terms[m].W, rm = p.r;
    const u = [p.dx / rm, p.dy / rm, p.dz / rm];
    const fcp = fcP(rm, o) * sc.S[m];
    let dEdr = gWt[m] * fcp;
    const elm = model.elements[p.e];
    for (let n = 0; n < 4; n++) dEdr += part.gA[4 * m + n] * (-(elm.beta[n] / elm.re)) * terms[m].A[n];
    dEdr += 0.5 * W * dphi[m];
    // angular: (I - u u) / r applied to dE/du
    const gu = [part.gU[3 * m], part.gU[3 * m + 1], part.gU[3 * m + 2]];
    const uDotG = u[0] * gu[0] + u[1] * gu[1] + u[2] * gu[2];
    for (let c = 0; c < 3; c++) g[3 * m + c] = dEdr * u[c] + (gu[c] - u[c] * uDotG) / rm;
  }
  // screening chain: dE/dS_m = gWt_m fc_m; S_m = prod_k f_mk; dS_m/d d_k through f_mk(A, B)
  const R = otherScreens(N, sc.f);
  for (let m = 0; m < N; m++) {
    const cm = gWt[m] * fcW(nb[m].r, o);
    if (cm === 0) continue;
    const rm2 = nb[m].r * nb[m].r;
    for (let k = 0; k < N; k++) {
      if (k === m) continue;
      const idx = m * N + k;
      if (sc.cA[idx] === 0 && sc.cB[idx] === 0) continue;
      const cw = cm * R[idx];
      const pk = nb[k], pm = nb[m];
      const A = (pk.r * pk.r) / rm2;
      const ex = pk.dx - pm.dx, ey = pk.dy - pm.dy, ez = pk.dz - pm.dz;
      const B = (ex * ex + ey * ey + ez * ez) / rm2;
      const dvec = [ex, ey, ez];
      const dk = [pk.dx, pk.dy, pk.dz], dm = [pm.dx, pm.dy, pm.dz];
      for (let c = 0; c < 3; c++) {
        const dAk = (2 * dk[c]) / rm2;
        const dBk = (2 * dvec[c]) / rm2;
        const dAm = (-2 * A * dm[c]) / rm2;
        const dBm = (-2 * dvec[c]) / rm2 - (2 * B * dm[c]) / rm2;
        g[3 * k + c] += cw * (sc.cA[idx] * dAk + sc.cB[idx] * dBk);
        g[3 * m + c] += cw * (sc.cA[idx] * dAm + sc.cB[idx] * dBm);
      }
    }
  }
  return Fv + pairE;
}

/*
 * Tabulated pair term (see meam.ts): 1000 uniform intervals over [0, 1.1 rc] with the cubic of PhiTable, nodes
 * from 1 A, reference shells summed up to 1.1 rc. The alloy path reads the same table as the single element.
 */
export const PHI_INTERVALS = 1000;
export const PHI_SPAN = 1.1;
export const PHI_LO = 1.0;

/** Cubic Hermite interpolation of tabulated values over a uniform grid (x_k = k dx; the eam.ts scheme). */
export class PhiTable {
  readonly n: number;
  readonly dx: number;
  readonly y: Float64Array;
  private readonly s: Float64Array;
  private readonly c2: Float64Array;
  private readonly c3: Float64Array;

  constructor(y: Float64Array, dx: number) {
    const n = y.length;
    this.n = n;
    this.dx = dx;
    this.y = y;
    const s = new Float64Array(n);
    s[0] = y[1] - y[0];
    s[n - 1] = y[n - 1] - y[n - 2];
    s[1] = 0.5 * (y[2] - y[0]);
    s[n - 2] = 0.5 * (y[n - 1] - y[n - 3]);
    for (let k = 2; k < n - 2; k++) s[k] = ((y[k - 2] - y[k + 2]) + 8 * (y[k + 1] - y[k - 1])) / 12;
    this.s = s;
    this.c2 = new Float64Array(n - 1);
    this.c3 = new Float64Array(n - 1);
    for (let k = 0; k < n - 1; k++) {
      const dy = y[k + 1] - y[k];
      this.c2[k] = 3 * dy - 2 * s[k] - s[k + 1];
      this.c3[k] = s[k] + s[k + 1] - 2 * dy;
    }
  }

  private at(x: number): [number, number] {
    let p = x / this.dx;
    let k = Math.floor(p);
    if (k < 0) k = 0;
    else if (k > this.n - 2) k = this.n - 2;
    p -= k;
    if (p > 1) p = 1;
    return [k, p];
  }

  eval(x: number): number {
    const [k, p] = this.at(x);
    return ((this.c3[k] * p + this.c2[k]) * p + this.s[k]) * p + this.y[k];
  }

  deriv(x: number): number {
    const [k, p] = this.at(x);
    return ((3 * this.c3[k] * p + 2 * this.c2[k]) * p + this.s[k]) / this.dx;
  }
}

const alloyTables = new WeakMap<AlloyModel, Map<number, { tab: PhiTable }>>();

/** Tabulated alloy pair term phi_ij and its derivative for pair (i, j) at distance r. */
export const alloyPairTab = (model: AlloyModel, i: number, j: number, r: number): { phi: number; dphi: number } => {
  const rc1 = PHI_SPAN * model.opts.rc;
  const dr = rc1 / PHI_INTERVALS;
  const kLo = Math.ceil(PHI_LO / dr);
  const tmodel = (): AlloyModel => ({ ...model, opts: { ...model.opts, rc: rc1 } });
  if (r < (kLo + 2) * dr) return alloyPair(tmodel(), i, j, r);
  let per = alloyTables.get(model);
  if (!per) alloyTables.set(model, (per = new Map()));
  const key = i * 64 + j; // element pairs (i, j) of a model
  let ent = per.get(key);
  if (!ent) {
    const tm = tmodel();
    const y = new Float64Array(PHI_INTERVALS + 1);
    for (let k = kLo; k <= PHI_INTERVALS; k++) y[k] = alloyPair(tm, i, j, k * dr).phi;
    for (let k = 0; k < kLo; k++) y[k] = y[kLo];
    ent = { tab: new PhiTable(y, dr) };
    per.set(key, ent);
  }
  return { phi: ent.tab.eval(r), dphi: ent.tab.deriv(r) };
};
