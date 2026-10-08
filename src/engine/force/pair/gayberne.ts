import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, typeBounds, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style gayberne — docs.lammps.org/pair_gayberne.html (mirror: plans/lammps-docs/pair_gayberne.rst).
 *   pair_style gayberne gamma upsilon mu cutoff
 *   pair_coeff I J epsilon sigma epsilon_ia epsilon_ib epsilon_ic epsilon_ja epsilon_jb epsilon_jc [cutoff]
 * The page: "The *gayberne* styles compute a Gay-Berne anisotropic LJ interaction" between ellipsoidal particles
 * or an ellipsoidal and spherical particle, with U = U_r . eta . chi and U_r = 4 epsilon (rho^12 - rho^6),
 * rho = sigma / (h12 + gamma sigma). The energy functions are the Everaers and Ejtehadi (PRE 67, 041710, 2003)
 * forms of the Berardi, Fava and Zannoni product ansatz, with the distance of closest approach h12 = r12 - sigma12
 * (their eq. 8-10):
 *   sigma12 = [ 1/2 rhat^T G12^-1 rhat ]^-1/2,  G12 = R1 S1^2 R1^T + R2 S2^2 R2^T   (S = radii)
 *   eta12   = [ 2 s1 s2 / det G12 ]^(upsilon/2),  s_i = (a_i b_i + c_i c_i) (a_i b_i)^1/2
 *   chi12   = [ 2 rhat^T B12^-1 rhat ]^mu,        B12 = R1 E1 R1^T + R2 E2 R2^T,  E_i = diag(e_ia, e_ib, e_ic)^(-1/mu)
 * The page says "The distance-of-closest-approach approximation used by LAMMPS becomes less accurate when" about
 * this h12 (eq. 8, the Everaers sigma12 form). The engine keeps the rotation R (body to space); the paper's A
 * (lab to body) is R^T, which gives the same G12 and B12.
 * Point particles: "if the 3 shape parameters are set to 0.0, which is a valid way in LAMMPS to specify a point
 * particle, then the Gay-Berne potential will treat that as shape parameters of 1.0". Measured with native LAMMPS
 * (black box): a point particle takes radius 1.0 in each direction (a sphere of diameter 2.0), and a pair with a
 * point particle matches the same pair with shape 2.0.
 * LJ spheres: "then the particle is treated as an LJ sphere by the Gay-Berne potential". Measured with native
 * LAMMPS (black box): the LJ-sphere test is spherical shape plus equal epsilon a,b,c; two LJ spheres use the plain
 * Lennard-Jones formula in r (with pair_modify shift), and a sphere with epsilon 1 1 2 is a Gay-Berne particle.
 * Epsilon per type: "coefficients are actually defined for atom types, not for pairs of atom types." Measured
 * with native LAMMPS (black box): the three epsilon_i values of a pair_coeff line go to type I and the three
 * epsilon_j values to type J, and a later line replaces the earlier values of a type (last setting wins).
 * Shift: "but only for sphere-sphere interactions." "There is no shifting performed for ellipsoidal interactions"
 * (the doc); measured with native LAMMPS (black box): the shift is applied to LJ-sphere pairs only.
 * Cutoff: pairs with r12 at or beyond the cutoff have no energy (measured with native LAMMPS, black box).
 * Quaternions are normalized before they become rotation matrices (measured with native LAMMPS, black box: an
 * unnormalized read_data quaternion gives the energy of its normalized form).
 *
 * Torques: tau = -dU/dphi with phi the space-frame rotation of each body; torque is an owned per-atom quantity,
 * folded from ghosts with nb.reverseSum (as in pair dipole). The derivatives are written out next to the code;
 * tests/engineGayBerne.test.ts checks forces and torques by central finite differences.
 *
 * Refusals (StyleError, each measured with native LAMMPS, black box, see the comments at the check):
 * an atom style without ellipsoids; two atoms of one type with different shapes; a type without epsilon a,b,c;
 * an anisotropic-epsilon point particle; an LJ sphere paired with a Gay-Berne particle when mu is not 1; a special
 * neighbour with a factor other than 1 (the page describes no special-bond factor for this style).
 */

/** Row-major 3x3 helpers (9 entries). */
const det3 = (m: ArrayLike<number>): number =>
  m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);

/** Inverse of a symmetric 3x3 matrix (adjugate / det) into out. */
const inv3 = (m: ArrayLike<number>, out: Float64Array): number => {
  const d = det3(m);
  const id = 1 / d;
  out[0] = (m[4] * m[8] - m[5] * m[7]) * id;
  out[1] = (m[2] * m[7] - m[1] * m[8]) * id;
  out[2] = (m[1] * m[5] - m[2] * m[4]) * id;
  out[3] = (m[5] * m[6] - m[3] * m[8]) * id;
  out[4] = (m[0] * m[8] - m[2] * m[6]) * id;
  out[5] = (m[2] * m[3] - m[0] * m[5]) * id;
  out[6] = (m[3] * m[7] - m[4] * m[6]) * id;
  out[7] = (m[1] * m[6] - m[0] * m[7]) * id;
  out[8] = (m[0] * m[4] - m[1] * m[3]) * id;
  return d;
};

const mv = (m: ArrayLike<number>, v: ArrayLike<number>, o: number[]): number[] => {
  o[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  o[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  o[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
  return o;
};

/** Rotation matrix (body to space) of a unit quaternion (w, i, j, k). */
const quatToMat = (w: number, x: number, y: number, z: number, out: Float64Array): void => {
  const n = 1 / Math.sqrt(w * w + x * x + y * y + z * z);
  w *= n; x *= n; y *= n; z *= n;
  out[0] = 1 - 2 * (y * y + z * z); out[1] = 2 * (x * y - w * z); out[2] = 2 * (x * z + w * y);
  out[3] = 2 * (x * y + w * z); out[4] = 1 - 2 * (x * x + z * z); out[5] = 2 * (y * z - w * x);
  out[6] = 2 * (x * z - w * y); out[7] = 2 * (y * z + w * x); out[8] = 1 - 2 * (x * x + y * y);
};

/** Symmetric product R diag(d) R^T for diagonal entries d. */
const rdrt = (R: ArrayLike<number>, d: ArrayLike<number>, out: Float64Array): void => {
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += R[3 * a + k] * d[k] * R[3 * b + k];
      out[3 * a + b] = s;
    }
  }
};

/** Matrix product a b (row-major). */
const mm = (a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array): void => {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) out[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
  }
};

export class PairGayBerne extends Pair {
  readonly name = 'gayberne';
  virialFdotr = true;
  gamma = 1;
  upsilon = 1;
  mu = 1;
  cutGlobal = 0;
  p!: PairParams;
  /** Per type: epsilon a,b,c (index 3 t + d), and whether any were set. */
  eT = new Float64Array(0);
  eSet = new Uint8Array(0);
  offset = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 4) throw new StyleError('Illegal pair_style command');
    this.gamma = parseNum(args[0], 'gamma');
    this.upsilon = parseNum(args[1], 'upsilon');
    this.mu = parseNum(args[2], 'mu');
    this.cutGlobal = parseNum(args[3], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style gayberne cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut']);
    this.eT = new Float64Array(3 * (ntypes + 1));
    this.eSet = new Uint8Array(ntypes + 1);
  }

  coeff(args: string[]): void {
    if (args.length < 10 || args.length > 11) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const ei = [4, 5, 6].map((k) => parseNum(args[k], 'epsilon_i'));
    const ej = [7, 8, 9].map((k) => parseNum(args[k], 'epsilon_j'));
    const cut = args.length === 11 ? parseNum(args[10], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, cut]);
    // Per-type epsilon a,b,c: non-zero triples are assigned to the type (the three values are assigned to atom type I).
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const assign = (lo: number, hi: number, e: number[]) => {
      if (!e.some((v) => v !== 0)) return;
      for (let t = lo; t <= hi; t++) {
        for (let d = 0; d < 3; d++) this.eT[3 * t + d] = e[d];
        this.eSet[t] = 1;
      }
    };
    for (let i = ilo; i <= ihi; i++) for (let j = jlo; j <= jhi; j++) {
      assign(i, i, ei);
      assign(j, j, ej);
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    for (const t of [i, j]) {
      // Measured with native LAMMPS (black box): init fails when a type has no epsilon a,b,c.
      if (!this.eSet[t]) throw new StyleError('Pair gayberne epsilon a,b,c coeffs are not all set');
    }
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'epsilon', mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    const nt = this.ntypes + 1;
    if (this.offset.length !== nt * nt) this.offset = new Float64Array(nt * nt);
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    // pair_modify shift: the LJ offset is used by sphere-sphere (LJ sphere) pairs only.
    if (this.shift && cut > 0) {
      const r = sig / cut;
      this.offset[k1] = this.offset[k2] = 4 * eps * (r ** 12 - r ** 6);
    } else this.offset[k1] = this.offset[k2] = 0;
    return cut;
  }

  init(ctx: StyleContext): void {
    // The page says that this style does not support the pair_modify tail option. Native LAMMPS ignores tail yes for
    // this style (measured, black box: the pair energy is unchanged); the engine refuses it instead.
    if (this.tail) throw new StyleError('pair_modify tail is not supported by pair_style gayberne');
    this.offset = new Float64Array(0);
    super.init(ctx);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.shape || !s.quat || !s.torque) throw new StyleError('Pair gayberne requires atom style ellipsoid');
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const mu = this.mu, nu = this.upsilon, gam = this.gamma;
    const invMu = 1 / mu;
    const tq = new Float64Array(3 * pc.nall);
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const cutsq = this.cutsq;
    const shape = s.shape, quat = s.quat;

    // Per owned atom: space-frame G and B contributions, radii, the eta factor s_i and the LJ-sphere flag.
    const Mo = new Float64Array(9 * nlocal), No = new Float64Array(9 * nlocal);
    const sf = new Float64Array(nlocal), ljs = new Uint8Array(nlocal), pt = new Uint8Array(nlocal);
    const R = new Float64Array(9), D2 = new Float64Array(3), Dn = new Float64Array(3);
    const rad = new Float64Array(3);
    // Measured with native LAMMPS (black box): every atom of one type must have the same shape (same raw diameters).
    const tShape = new Float64Array(3 * (nt + 1)), tSeen = new Uint8Array(nt + 1);
    for (let o = 0; o < nlocal; o++) {
      const t = type[o];
      const ellip = shape[3 * o] > 0;
      if (!tSeen[t]) {
        tSeen[t] = 1;
        for (let d = 0; d < 3; d++) tShape[3 * t + d] = shape[3 * o + d];
      } else if (shape[3 * o] !== tShape[3 * t] || shape[3 * o + 1] !== tShape[3 * t + 1] || shape[3 * o + 2] !== tShape[3 * t + 2]) {
        throw new StyleError('Pair gayberne requires atoms with same type have same shape');
      }
      if (ellip) { rad[0] = shape[3 * o]; rad[1] = shape[3 * o + 1]; rad[2] = shape[3 * o + 2]; } else { rad[0] = rad[1] = rad[2] = 1; }
      quatToMat(quat[4 * o], quat[4 * o + 1], quat[4 * o + 2], quat[4 * o + 3], R);
      for (let d = 0; d < 3; d++) {
        D2[d] = rad[d] * rad[d];
        Dn[d] = Math.pow(this.eT[3 * t + d], -invMu);
      }
      const Mtmp = new Float64Array(9), Ntmp = new Float64Array(9);
      rdrt(R, D2, Mtmp);
      rdrt(R, Dn, Ntmp);
      Mo.set(Mtmp, 9 * o);
      No.set(Ntmp, 9 * o);
      sf[o] = (rad[0] * rad[1] + rad[2] * rad[2]) * Math.sqrt(rad[0] * rad[1]);
      pt[o] = ellip ? 0 : 1;
      ljs[o] = rad[0] === rad[1] && rad[1] === rad[2] && this.eT[3 * t] === this.eT[3 * t + 1] && this.eT[3 * t + 1] === this.eT[3 * t + 2] ? 1 : 0;
    }

    const G = new Float64Array(9), Gi = new Float64Array(9), B = new Float64Array(9), Bi = new Float64Array(9);
    const tmp = new Float64Array(9), C = new Float64Array(9);
    const v = [0, 0, 0], w = [0, 0, 0], o3 = [0, 0, 0];
    let evdwl = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const dxi = xi - x[3 * j], dyi = yi - x[3 * j + 1], dzi = zi - x[3 * j + 2];
        const rsq = dxi * dxi + dyi * dyi + dzi * dzi;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const sLj = sL[jj >>> SBBITS], sCj = sC[jj >>> SBBITS];
        if (jj >>> SBBITS !== 0 && (sLj !== 1 || sCj !== 1)) {
          throw new StyleError('pair_style gayberne does not support special_bonds factors other than 1');
        }
        const ti_ = type[i], tj_ = type[j];
        const eps = this.p.get('epsilon', ti_, tj_), sig = this.p.get('sigma', ti_, tj_);
        const cut = this.p.get('cut', ti_, tj_);
        if (ljs[oi] && ljs[oj]) {
          // LJ sphere pair: plain Lennard-Jones in r, shifted by the pair_modify shift offset.
          const r2inv = 1 / rsq;
          const sr2 = sig * sig * r2inv, sr6 = sr2 * sr2 * sr2, sr12 = sr6 * sr6;
          const eLJ = 4 * eps * (sr12 - sr6) - this.offset[t];
          evdwl += eLJ;
          const fljr = (48 * eps * sr12 - 24 * eps * sr6) * r2inv;
          f[3 * i] += fljr * dxi; f[3 * i + 1] += fljr * dyi; f[3 * i + 2] += fljr * dzi;
          f[3 * j] -= fljr * dxi; f[3 * j + 1] -= fljr * dyi; f[3 * j + 2] -= fljr * dzi;
          if (pc.eatom) { pc.eatom[i] += 0.5 * eLJ; pc.eatom[j] += 0.5 * eLJ; }
          if (pc.vatom) {
            const va = pc.vatom;
            const ww = [dxi * fljr * dxi, dyi * fljr * dyi, dzi * fljr * dzi, dxi * fljr * dyi, dxi * fljr * dzi, dyi * fljr * dzi];
            for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * ww[c]; va[6 * j + c] += 0.5 * ww[c]; }
          }
          continue;
        }

        // Measured with native LAMMPS (black box): a point particle (shape 0 0 0) whose epsilon a,b,c are not equal
        // gives an energy that differs from the Gay-Berne form with radius 1.0 (the same pair with an isotropic
        // epsilon matches), and two such point particles make the native solver fail. Refused rather than reproduced.
        if ((pt[oi] && !ljs[oi]) || (pt[oj] && !ljs[oj])) {
          throw new StyleError('pair_style gayberne: point particles need equal epsilon a,b,c (anisotropic point particles are not supported)');
        }
        // Measured with native LAMMPS (black box): when one partner is an LJ sphere (spherical shape, equal epsilon
        // a,b,c) and the other is not, the energy and forces match this form, but the native torque on the other
        // particle differs from the gradient of its energy unless mu = 1. Refused rather than reproduced.
        if (ljs[oi] !== ljs[oj] && mu !== 1) {
          throw new StyleError('pair_style gayberne: an LJ sphere paired with an ellipsoid needs mu = 1 (the native torques differ otherwise)');
        }

        // Gay-Berne pair. r = x_j - x_i (the energy is even in r, so the sign only matters for the force).
        const rx = -dxi, ry = -dyi, rz = -dzi;
        const rr = Math.sqrt(rsq);
        const rhat = [rx / rr, ry / rr, rz / rr];
        const Mi = Mo.subarray(9 * oi, 9 * oi + 9), Mj = Mo.subarray(9 * oj, 9 * oj + 9);
        const Ni = No.subarray(9 * oi, 9 * oi + 9), Nj = No.subarray(9 * oj, 9 * oj + 9);
        for (let q = 0; q < 9; q++) { G[q] = Mi[q] + Mj[q]; B[q] = Ni[q] + Nj[q]; }
        const detG = inv3(G, Gi);
        inv3(B, Bi);
        // sigma12 = |r| P^-1/2 with P = 1/2 r^T G^-1 r ; v = G^-1 r
        mv(Gi, [rx, ry, rz], v);
        const P = 0.5 * (rx * v[0] + ry * v[1] + rz * v[2]);
        const sqrtP = Math.sqrt(P);
        const sigma12 = rr / sqrtP;
        const h = rr - sigma12;
        // chi = q^mu, q = 2 rhat^T B^-1 rhat ; w = B^-1 r
        mv(Bi, [rx, ry, rz], w);
        const rw = rx * w[0] + ry * w[1] + rz * w[2];
        const qv = 2 * rw / rsq;
        const chi = Math.pow(qv, mu);
        const eta = Math.pow(2 * sf[oi] * sf[oj] / detG, 0.5 * nu);
        const hg = h + gam * sig;
        const rho = sig / hg;
        const rho6 = rho ** 6, rho12 = rho6 * rho6;
        const Ur = 4 * eps * (rho12 - rho6);
        const dUr = 4 * eps * (-12 * rho12 + 6 * rho6) / hg; // dUr/dh
        const eC = eta * chi;
        const e = Ur * eC;
        evdwl += e;

        // dU/dr12 (vector). dh/dr = rhat - dsigma/dr, dsigma/dr = rhat sigma/rr - 1/2 rr P^-3/2 v ;
        // dq/dr = 4 w/rr^2 - 2 q r/rr^2.
        const P32 = P * sqrtP; // P^3/2
        const dsdr = [0, 1, 2].map((d) => rhat[d] * sigma12 / rr - 0.5 * rr / P32 * v[d]);
        const dhdr = [0, 1, 2].map((d) => rhat[d] - dsdr[d]);
        const dqdr = [0, 1, 2].map((d) => 4 * w[d] / rsq - 2 * qv * [rx, ry, rz][d] / rsq);
        const g = [0, 1, 2].map((d) =>
          eta * (dUr * chi * dhdr[d] + Ur * chi * mu * dqdr[d] / qv));
        // force on i is +g (U depends on x_j - x_i), on j is -g
        f[3 * i] += g[0]; f[3 * i + 1] += g[1]; f[3 * i + 2] += g[2];
        f[3 * j] -= g[0]; f[3 * j + 1] -= g[1]; f[3 * j + 2] -= g[2];

        // Torques: tau_b = -dU/dphi_b. For a rotation of body b with space matrix M_b (G part), N_b (B part):
        //   dsigma/dphi_b = 1/2 rr P^-3/2 (M_b v x v)
        //   dq/dphi_b     = -4 (N_b w x w)/rr^2
        //   dln(eta)/dphi_b = -nu axial(M_b G^-1 - G^-1 M_b)
        // so dU/dphi_b = eta chi dUr (-dsigma/dphi_b) + Ur eta chi mu (dq/dphi_b)/q + Ur chi d eta/dphi_b.
        for (let side = 0; side < 2; side++) {
          const b = side === 0 ? i : j;
          const Mb = side === 0 ? Mi : Mj;
          const Nb = side === 0 ? Ni : Nj;
          mv(Mb, v, o3);
          const cvx = o3[1] * v[2] - o3[2] * v[1];
          const cvy = o3[2] * v[0] - o3[0] * v[2];
          const cvz = o3[0] * v[1] - o3[1] * v[0];
          const dsx = 0.5 * rr / P32 * cvx, dsy = 0.5 * rr / P32 * cvy, dsz = 0.5 * rr / P32 * cvz;
          mv(Nb, w, o3);
          const nwx = o3[1] * w[2] - o3[2] * w[1];
          const nwy = o3[2] * w[0] - o3[0] * w[2];
          const nwz = o3[0] * w[1] - o3[1] * w[0];
          // C = Mb Gi - Gi Mb
          mm(Mb, Gi, tmp);
          const T2 = new Float64Array(9);
          mm(Gi, Mb, T2);
          for (let q = 0; q < 9; q++) C[q] = tmp[q] - T2[q];
          const ax = C[5], ay = C[6], az = C[1]; // axial (C23, C31, C12), 0-based C[1][2], C[2][0], C[0][1]
          const dUx = eC * dUr * (-dsx) + Ur * eC * mu * (-4 * nwx / rsq) / qv - Ur * chi * nu * eta * ax;
          const dUy = eC * dUr * (-dsy) + Ur * eC * mu * (-4 * nwy / rsq) / qv - Ur * chi * nu * eta * ay;
          const dUz = eC * dUr * (-dsz) + Ur * eC * mu * (-4 * nwz / rsq) / qv - Ur * chi * nu * eta * az;
          // Measured with native LAMMPS (black box): a point particle (shape 0 0 0) gets no torque.
          if (!pt[side === 0 ? oi : oj]) {
            tq[3 * b] -= dUx; tq[3 * b + 1] -= dUy; tq[3 * b + 2] -= dUz;
          }
        }

        if (pc.eatom) { pc.eatom[i] += 0.5 * e; pc.eatom[j] += 0.5 * e; }
        if (pc.vatom) {
          const va = pc.vatom;
          const fx = g[0], fy = g[1], fz = g[2];
          const dx = -rx, dy = -ry, dz = -rz; // x_i - x_j
          const wv = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
          for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * wv[c]; va[6 * j + c] += 0.5 * wv[c]; }
        }
      }
    }
    nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
