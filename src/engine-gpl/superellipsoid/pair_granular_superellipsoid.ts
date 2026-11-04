/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/ASPHERE/pair_granular_superellipsoid.cpp and
 * pair_granular_superellipsoid.h, GPL-2.0 (src/engine-gpl/LICENSE). Contributing authors: Jacopo Bilotto (EPFL), Jibril B.
 * Coulibaly. pair_style granular/superellipsoid: the normal models hooke and hertz, the damping models mass_velocity and
 * viscoelastic, the tangential models linear_history and classic, limit_damping, the curvature_gaussian and no_bounding_box
 * settings, and the geometric contact formulation.
 *
 * Differences from the C++ that the engine needs (see the report of the port):
 * - the engine's half neighbour list is used with the pair always computed from the atom of the lower ID (the C++ uses the
 * neighbour list order); the tangential history and the contact point are kept per atom-ID pair, and the forces of both
 * atoms are tallied in the engine's ghost-force array (reverse-summed by the force field) instead of the C++ newton-off
 * ownership rule. The physics and the stored history are the same.
 * - the contact radius of the C++ code is computed twice; the second value (the Hertzian approximation) is the one used,
 * and it is what this port uses.
 * - the radius used for the neighbour cutoff and the bounding sphere test is sqrt(a^2 + b^2 + c^2) for every particle (a
 * bound on the surface, not the C++ radius); the contact set does not depend on the bound, only the neighbour list does.
 *
 * Functions: compute and single (PairGranularSuperellipsoid::single at b332f28). single returns the energy 0.0 and
 * fforce 0 as native does (native sets fforce only on its early returns and never on a contact); the contact forces
 * native writes to svector (single_extra) are not reproduced here.
 */
import { Pair, StyleError, pairCoeffArgs, tallyAtomXyz, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { bounds, missingArgs, numeric, parseNum } from '../../engine/force/util';
import { SUPER_BLOCK, SUPER_INERTIA } from '../../engine/superellipsoid';
import type { SimState } from '../../engine/types';
import {
  FORMULATION_ALGEBRAIC, FORMULATION_GEOMETRIC, checkOrientedBoundingBoxes, contactParticle, determineContactPoint,
  gaussianCurvature, meanCurvature, overlapDistance, type ContactParticle,
} from './math';

const HOOKE = 0, HERTZ = 1;
const MASS_VELOCITY = 0, VISCOELASTIC = 1;
const CLASSIC = 0, LINEAR_HISTORY = 1;
const NUMSTEP_INITIAL_GUESS = 5;
const EPSILON = 1e-10;

/** A pair_coeff record (one type pair, after mixing). */
interface Spec {
  normal: number;
  kn: number;
  gamman: number;
  damping: number;
  tangential: number;
  kt: number;
  xt: number;
  xmu: number;
  limitDamping: boolean;
}

/**
 * Contact history of one atom-ID pair: the flag of the last step, the contact point (4) and the shear and axis entries,
 * and the stamp of the last compute that met the pair in the neighbour list (a record it did not meet is dropped).
 */
interface Hist {
  touch: number;
  h: Float64Array;
  stamp: number;
}

/** The key of an atom-ID pair (lo < hi): a number while the IDs fit, else a string. */
const histKey = (lo: number, hi: number): number | string => (hi < 67108864 ? lo * 67108864 + hi : `${lo}:${hi}`);

/** Rotation matrix (row-major) of a unit quaternion (w i j k), the MathExtra quat_to_mat. */
const quatToMat = (w: number, i: number, j: number, k: number, R: Float64Array): void => {
  R[0] = 1 - 2 * (j * j + k * k); R[1] = 2 * (i * j - w * k); R[2] = 2 * (i * k + w * j);
  R[3] = 2 * (i * j + w * k); R[4] = 1 - 2 * (i * i + k * k); R[5] = 2 * (j * k - w * i);
  R[6] = 2 * (i * k - w * j); R[7] = 2 * (j * k + w * i); R[8] = 1 - 2 * (i * i + j * j);
};

/**
 * Angular velocity from the angular momentum L and the principal moments I (the MathExtra angmom_to_omega): the body
 * components wb[k] = L . e_k / I[k] with e_k the column k of R, then w = sum wb[k] e_k.
 */
const angmomToOmega = (L0: number, L1: number, L2: number, R: Float64Array, I: Float64Array, o: number, w: Float64Array): void => {
  const wb0 = I[3 * o] === 0 ? 0 : (L0 * R[0] + L1 * R[3] + L2 * R[6]) / I[3 * o];
  const wb1 = I[3 * o + 1] === 0 ? 0 : (L0 * R[1] + L1 * R[4] + L2 * R[7]) / I[3 * o + 1];
  const wb2 = I[3 * o + 2] === 0 ? 0 : (L0 * R[2] + L1 * R[5] + L2 * R[8]) / I[3 * o + 2];
  w[0] = wb0 * R[0] + wb1 * R[1] + wb2 * R[2];
  w[1] = wb0 * R[3] + wb1 * R[4] + wb2 * R[5];
  w[2] = wb0 * R[6] + wb1 * R[7] + wb2 * R[8];
};

/** Rotate v so that it is perpendicular to the unit vector n, keeping its magnitude (GranSubMod rotate_rescale). */
const rotateRescale = (v: Float64Array, n: Float64Array): void => {
  const rsht = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
  const shrmag = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  for (let k = 0; k < 3; k++) v[k] -= rsht * n[k];
  const prjmag = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const f = prjmag > 0 ? shrmag / prjmag : 0;
  for (let k = 0; k < 3; k++) v[k] *= f;
};

/** Centre, rotation, half-diameters, exponents and block type of the particle of owned atom o at (x0, x1, x2). */
const setParticle = (p: ContactParticle, x0: number, x1: number, x2: number, o: number, shape: Float64Array, block: Float64Array, quat: Float64Array): void => {
  p.x[0] = x0; p.x[1] = x1; p.x[2] = x2;
  quatToMat(quat[4 * o], quat[4 * o + 1], quat[4 * o + 2], quat[4 * o + 3], p.R);
  p.shape[0] = shape[3 * o]; p.shape[1] = shape[3 * o + 1]; p.shape[2] = shape[3 * o + 2];
  const b1 = block[2 * o], b2 = block[2 * o + 1];
  p.block[0] = b1; p.block[1] = b2;
  p.flag = blockFlag(b1, b2);
};

/** compute's scratch: the two particles, their morphed copies for a new contact's initial guess, and contact vectors. */
const PA = contactParticle(), PB = contactParticle(), QA = contactParticle(), QB = contactParticle();
const X0 = new Float64Array(4);
const NIJ = new Float64Array(3);
const OMI = new Float64Array(3), OMJ = new Float64Array(3);
const SHEAR = new Float64Array(3), FS = new Float64Array(3), FDAMP = new Float64Array(3);

/** Bound on the distance of the surface from the centre: sqrt(a^2 + b^2 + c^2) with the half-diameters. */
const boundRadius = (sh: Float64Array, o: number): number => Math.sqrt(sh[3 * o] ** 2 + sh[3 * o + 1] ** 2 + sh[3 * o + 2] ** 2);

/** Block type of a particle: 0 ellipsoid (2, 2), 1 equal exponents, 2 superquadric (the C++ BlockType). */
const blockFlag = (b1: number, b2: number): number => (b1 === 2 && b2 === 2 ? 0 : b1 === b2 ? 1 : 2);

export class PairGranularSuperellipsoid extends Pair {
  readonly name = 'granular/superellipsoid';
  private globalCut: number | null = null;
  private curvatureGaussian = false;
  private boundingBox = true;
  private geometric = false;
  /** pair_coeff specs per type pair (i*nt+j, i <= j) and the resolved parameters (both orders) after init. */
  private specs: (Spec | null)[] = [];
  private pm: (Spec | null)[] = [];
  private cutType: number[] = [];
  private state: SimState | null = null;
  private freezeBit = 0;
  private maxRad: number[] = [];
  private hist = new Map<number | string, Hist>();
  private stampN = 0;
  private tqBuf = new Float64Array(0);
  /** Size of the per-pair history: X0 (4), the cached axis (bounding box only), and the 3 tangential entries. */
  private sizeHistory = 8;
  private defaultHist = 5;

  settings(args: string[], _ctx: StyleContext): void {
    this.globalCut = null;
    this.curvatureGaussian = false;
    this.boundingBox = true;
    this.geometric = false;
    args.forEach((w, iarg) => {
      if (w === 'no_bounding_box') this.boundingBox = false;
      else if (w === 'geometric') this.geometric = true;
      else if (w === 'curvature_gaussian') this.curvatureGaussian = true;
      else if (iarg === 0) this.globalCut = parseNum(w, 'cutoff');
      else throw new StyleError(`Illegal pair_style granular/superellipsoid keyword '${w}'`);
    });
    this.defaultHist = this.boundingBox ? 5 : 4;
    this.sizeHistory = this.defaultHist + 3;
    this.hist.clear();
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.specs = new Array(nt * nt).fill(null);
    this.pm = new Array(nt * nt).fill(null);
    this.cutType = new Array(nt * nt).fill(-1);
  }

  coeff(args: string[]): void {
    if (args.length < 3) throw pairCoeffArgs('usage: pair_coeff I J normal-model args tangential ...');
    const [ilo, ihi] = bounds(args[0], 1, this.ntypes);
    const [jlo, jhi] = bounds(args[1], 1, this.ntypes);
    const sp = parseSpec(args.slice(2));
    const nt = this.ntypes + 1;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.specs[i * nt + j] = { ...sp.spec };
        this.cutType[i * nt + j] = sp.cutoff;
      }
    }
  }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    this.state = s;
    this.freezeBit = ctx.freezeGroupBit ?? 0;
    if (!s || !s.rmass || !s.angmom || !s.shape || !s.quat || !s.custom.get(SUPER_BLOCK) || !s.custom.get(SUPER_INERTIA)) {
      throw new StyleError('Pair granular/superellipsoid requires atom attributes rmass, angmom and superellipsoid flag (atom_style ellipsoid superellipsoid)');
    }
    if (!ctx.ghostVelocity) throw new StyleError('Pair granular/superellipsoid requires ghost atoms store velocity\n(use comm_modify vel yes)');
    if (s.rigidBodyMass && s.rigidBodyMass.size > 0) {
      throw new StyleError('pair_style granular/superellipsoid with atoms in rigid bodies is not supported by the browser engine');
    }
    const blk = s.custom.get(SUPER_BLOCK)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.shape![3 * i] > 0)) throw new StyleError('Pair granular/superellipsoid requires all atoms are ellipsoids (a point particle was found)');
      // pair_granular_superellipsoid.html: "the blockiness parameters must be n1 >= 2.0 and n2 >= 2.0"
      if (blk[2 * i] < 2 || blk[2 * i + 1] < 2) throw new StyleError('Pair granular/superellipsoid requires blockiness exponents block1 and block2 >= 2 (concave particles are not supported)');
    }
    // the bounding radius per type (the C++ onerad / maxrad with the particles present at init)
    const nt = this.ntypes + 1;
    this.maxRad = new Array(nt).fill(0);
    for (let i = 0; i < s.n; i++) {
      const t = s.type[i];
      this.maxRad[t] = Math.max(this.maxRad[t], boundRadius(s.shape!, i));
    }
  }

  /** Mixed or explicit parameters of a type pair (i <= j); mixing as the C++ init_one (geometric means). */
  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    let spec = this.specs[i * nt + j];
    if (!spec) {
      const si = this.specs[i * nt + i], sj = this.specs[j * nt + j];
      if (!si || !sj) throw new StyleError(`All pair coeffs are not set for types ${i} and ${j} (granular/superellipsoid)`);
      if (si.normal !== sj.normal || si.tangential !== sj.tangential || si.damping !== sj.damping) {
        throw new StyleError(`Granular pair style functional forms are different, cannot mix coefficients for types ${i} and ${j}. This combination must be set explicitly via a pair_coeff command`);
      }
      spec = {
        normal: si.normal, damping: si.damping, tangential: si.tangential,
        kn: Math.sqrt(si.kn * sj.kn), gamman: Math.sqrt(si.gamman * sj.gamman),
        kt: Math.sqrt(si.kt * sj.kt), xt: Math.sqrt(si.xt * sj.xt), xmu: Math.sqrt(si.xmu * sj.xmu),
        limitDamping: si.limitDamping || sj.limitDamping,
      };
      this.specs[i * nt + j] = spec;
    }
    this.pm[i * nt + j] = this.pm[j * nt + i] = spec;
    const cut = this.cutType[i * nt + j] ?? -1;
    if (cut > 0) return cut;
    if (this.globalCut !== null && this.globalCut > 0) return this.globalCut;
    if (this.maxRad[i] > 0 && this.maxRad[j] > 0) return this.maxRad[i] + this.maxRad[j];
    let cutmax = 0;
    for (let k = 1; k < nt; k++) cutmax = Math.max(cutmax, 2 * this.maxRad[k]);
    return cutmax;
  }

  compute(pc: PairCompute): void {
    const s = this.state!;
    const list = pc.half!;
    const { x, f, nb } = pc;
    const owner = nb.owner;
    const update = pc.historyUpdate === true;
    const dt = s.dt;
    const ids = s.id, type = s.type, mask = s.mask, v = s.v, rmass = s.rmass!;
    const shape = s.shape!, quat = s.quat!, angmom = s.angmom!;
    const block = s.custom.get(SUPER_BLOCK)!.data, inertia = s.custom.get(SUPER_INERTIA)!.data;
    const nt = this.ntypes + 1;
    if (this.tqBuf.length < 3 * pc.nall) this.tqBuf = new Float64Array(3 * pc.nall);
    const tq = this.tqBuf;
    tq.fill(0, 0, 3 * pc.nall);
    const hist = this.hist;
    const stamp = ++this.stampN;
    const size = this.sizeHistory, nh = this.defaultHist;
    const acc = pc.acc;
    const formulation = this.geometric ? FORMULATION_GEOMETRIC : FORMULATION_ALGEBRAIC;
    const sLJ = pc.specialLJ;
    const nij = NIJ;

    for (let i = 0; i < list.inum; i++) {
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jraw = list.neighbors[k];
        let a = i, b = jraw & NEIGHMASK;
        let oa = owner[a], ob = owner[b];
        // compute: "factor_lj = special_lj[sbmask(j)]; ... if (factor_lj == 0) continue;" ahead of check_contact, so the
        // pair's touch flag and history stay as they were
        const factorLJ = sLJ[jraw >>> SBBITS];
        if (ids[oa] > ids[ob]) {
          let t = a; a = b; b = t;
          t = oa; oa = ob; ob = t;
        }
        const pm = this.pm[type[oa] * nt + type[ob]];
        if (!pm) continue;
        const key = histKey(ids[oa], ids[ob]);
        let rec = hist.get(key);
        if (!rec) {
          rec = { touch: 0, h: new Float64Array(size), stamp };
          hist.set(key, rec);
        } else rec.stamp = stamp;
        if (factorLJ === 0) continue;
        const h = rec.h;

        const xi0 = x[3 * a], xi1 = x[3 * a + 1], xi2 = x[3 * a + 2];
        const xj0 = x[3 * b], xj1 = x[3 * b + 1], xj2 = x[3 * b + 2];
        const dx0 = xi0 - xj0, dx1 = xi1 - xj1, dx2 = xi2 - xj2;
        const rsq = dx0 * dx0 + dx1 * dx1 + dx2 * dx2;
        const radsum = boundRadius(shape, oa) + boundRadius(shape, ob);
        const touchjj = rec.touch;

        // check_contact: bounding sphere, oriented bounding box, then the midway contact point
        let touching = false;
        // the exponents of the contact (the morphed ones after a new contact's initial guess)
        let blockI = PA.block, blockJ = PB.block;
        if (rsq < radsum * radsum) {
          setParticle(PA, xi0, xi1, xi2, oa, shape, block, quat);
          setParticle(PB, xj0, xj1, xj2, ob, shape, block, quat);
          let skip = false;
          if (this.boundingBox) {
            const axis = Math.trunc(h[4]);
            const newAxis = checkOrientedBoundingBoxes(PA, PB, axis);
            if (newAxis !== -1) {
              skip = true;
              if (update) h[4] = newAxis;
            }
          }
          if (!skip) {
            nij[0] = nij[1] = nij[2] = 0;
            if (touchjj === 1) {
              X0[0] = h[0] + xi0;
              X0[1] = h[1] + xi1;
              X0[2] = h[2] + xi2;
              X0[3] = h[3];
              touching = determineContactPoint(PA, PB, X0, nij, formulation) === 0;
            } else {
              // new contact: initial guess by morphing the particles from spheres to their shapes (C++ new contact branch)
              const shape0I = PA.shape, shape0J = PB.shape, block0I = PA.block, block0J = PB.block;
              const reqi = Math.cbrt(shape0I[0] * shape0I[1] * shape0I[2]);
              const reqj = Math.cbrt(shape0J[0] * shape0J[1] * shape0J[2]);
              const rsuminv = 1 / (reqi + reqj);
              X0[0] = (reqj * rsuminv) * xi0 + (reqi * rsuminv) * xj0;
              X0[1] = (reqj * rsuminv) * xi1 + (reqi * rsuminv) * xj1;
              X0[2] = (reqj * rsuminv) * xi2 + (reqi * rsuminv) * xj2;
              X0[3] = reqj / reqi;
              QA.x.set(PA.x); QA.R.set(PA.R);
              QB.x.set(PB.x); QB.R.set(PB.R);
              const shapeI = QA.shape, shapeJ = QB.shape;
              for (let iter = 1; iter <= NUMSTEP_INITIAL_GUESS; iter++) {
                const frac = iter / NUMSTEP_INITIAL_GUESS;
                for (let d = 0; d < 3; d++) {
                  shapeI[d] = reqi;
                  shapeJ[d] = reqj;
                }
                for (let d = 0; d < 3; d++) {
                  shapeI[d] = (1 - frac) * shapeI[d] + frac * shape0I[d];
                  shapeJ[d] = (1 - frac) * shapeJ[d] + frac * shape0J[d];
                }
                QA.block[0] = 2 + frac * (block0I[0] - 2); QA.block[1] = 2 + frac * (block0I[1] - 2);
                QB.block[0] = 2 + frac * (block0J[0] - 2); QB.block[1] = 2 + frac * (block0J[1] - 2);
                // the first guess uses the ellipsoid form for both particles (C++ iter_ig == 1)
                QA.flag = iter === 1 ? 0 : PA.flag;
                QB.flag = iter === 1 ? 0 : PB.flag;
                const status = determineContactPoint(QA, QB, X0, nij, formulation);
                if (status === 0) touching = true;
                else if (status === 1) touching = false;
              }
              // after the last guess the morphed shape is the shape itself (frac = 1); the exponents stay the morphed ones
              blockI = QA.block;
              blockJ = QB.block;
            }
          }
        }

        if (!touching) {
          rec.touch = 0;
          for (let kk = 0; kk < size; kk++) {
            if (this.boundingBox && kk === 4) continue;
            h[kk] = 0;
          }
          continue;
        }
        rec.touch = 1;

        // meff: a frozen partner (fix freeze) takes the other mass
        const mi = rmass[oa], mj = rmass[ob];
        let meff = (mi * mj) / (mi + mj);
        if (this.freezeBit) {
          if (mask[oa] & this.freezeBit) meff = mj;
          if (mask[ob] & this.freezeBit) meff = mi;
        }

        // the rotation matrices of the quaternions (quat_to_mat, as in the contact test)
        const RI = PA.R, RJ = PB.R;
        const xi = PA.x, xj = PB.x;
        const shapeI = PA.shape, shapeJ = PB.shape;
        angmomToOmega(angmom[3 * oa], angmom[3 * oa + 1], angmom[3 * oa + 2], RI, inertia, oa, OMI);
        angmomToOmega(angmom[3 * ob], angmom[3 * ob + 1], angmom[3 * ob + 2], RJ, inertia, ob, OMJ);
        const sp = pm;
        const normalModel = sp.normal, dampModel = sp.damping, tangModel = sp.tangential;
        const contactRadiusFlag = normalModel === HERTZ || dampModel === VISCOELASTIC;
        const flagI = blockFlag(blockI[0], blockI[1]), flagJ = blockFlag(blockJ[0], blockJ[1]);
        const n0 = nij[0], n1 = nij[1], n2 = nij[2];
        const m0 = -n0, m1 = -n1, m2 = -n2;
        const c0 = X0[0], c1 = X0[1], c2 = X0[2];

        // calculate_forces: store the contact point relative to atom i for the next step
        if (update) {
          h[0] = c0 - xi0;
          h[1] = c1 - xi1;
          h[2] = c2 - xi2;
          h[3] = X0[3];
        }
        const overlapI = overlapDistance(shapeI, blockI, RI, flagI, c0, c1, c2, n0, n1, n2, xi);
        const overlapJ = overlapDistance(shapeJ, blockJ, RJ, flagJ, c0, c1, c2, m0, m1, m2, xj);
        const crI0 = c0 - xi0, crI1 = c1 - xi1, crI2 = c2 - xi2;
        const crJ0 = c0 - xj0, crJ1 = c1 - xj1, crJ2 = c2 - xj2;
        // contact point velocities v + omega x r
        const cvI0 = v[3 * oa] + (OMI[1] * crI2 - OMI[2] * crI1);
        const cvI1 = v[3 * oa + 1] + (OMI[2] * crI0 - OMI[0] * crI2);
        const cvI2 = v[3 * oa + 2] + (OMI[0] * crI1 - OMI[1] * crI0);
        const cvJ0 = v[3 * ob] + (OMJ[1] * crJ2 - OMJ[2] * crJ1);
        const cvJ1 = v[3 * ob + 1] + (OMJ[2] * crJ0 - OMJ[0] * crJ2);
        const cvJ2 = v[3 * ob + 2] + (OMJ[0] * crJ1 - OMJ[1] * crJ0);
        const vr0 = cvI0 - cvJ0, vr1 = cvI1 - cvJ1, vr2 = cvI2 - cvJ2;
        const vnnr = vr0 * n0 + vr1 * n1 + vr2 * n2;
        const vtr0 = vr0 - vnnr * n0, vtr1 = vr1 - vnnr * n1, vtr2 = vr2 - vnnr * n2;

        let contactRadius = 0;
        if (contactRadiusFlag) {
          const sI0 = overlapI * n0 + c0, sI1 = overlapI * n1 + c1, sI2 = overlapI * n2 + c2;
          const sJ0 = overlapJ * m0 + c0, sJ1 = overlapJ * m1 + c1, sJ2 = overlapJ * m2 + c2;
          const curvI = this.curvatureGaussian ? gaussianCurvature(shapeI, blockI, flagI, RI, sI0, sI1, sI2, xi) : meanCurvature(shapeI, blockI, flagI, RI, sI0, sI1, sI2, xi);
          const curvJ = this.curvatureGaussian ? gaussianCurvature(shapeJ, blockJ, flagJ, RJ, sJ0, sJ1, sJ2, xj) : meanCurvature(shapeJ, blockJ, flagJ, RJ, sJ0, sJ1, sJ2, xj);
          // the Hertzian contact radius approximation (the C++ overwrites its capped value with this one)
          contactRadius = Math.sqrt((overlapI + overlapJ) / (curvI + curvJ));
        }

        let Fnormal = 0;
        if (normalModel === HOOKE) Fnormal = sp.kn * (overlapI + overlapJ);
        else if (normalModel === HERTZ) Fnormal = sp.kn * (overlapI + overlapJ) * contactRadius;

        const damp = sp.gamman;
        let dampPrefactor: number, Fdamp: number;
        if (dampModel === MASS_VELOCITY) {
          dampPrefactor = damp * meff;
          Fdamp = dampPrefactor * vnnr;
        } else {
          dampPrefactor = damp * meff * contactRadius;
          Fdamp = dampPrefactor * vnnr;
        }
        let Fntot = Fnormal + Fdamp;
        if (sp.limitDamping && Fntot < 0) Fntot = 0;
        const Fncrit = Math.abs(Fntot);

        const Fscrit = Fncrit * sp.xmu;
        const dampt = sp.xt * dampPrefactor;
        const shear = SHEAR, fs = FS, fdamp = FDAMP;
        shear[0] = h[nh]; shear[1] = h[nh + 1]; shear[2] = h[nh + 2];
        fdamp[0] = -dampt * vtr0; fdamp[1] = -dampt * vtr1; fdamp[2] = -dampt * vtr2;
        if (tangModel === LINEAR_HISTORY) {
          if (update) {
            const rsht = shear[0] * n0 + shear[1] * n1 + shear[2] * n2;
            const frameUpdate = Math.abs(rsht) * sp.kt > EPSILON * Fscrit;
            if (frameUpdate) rotateRescale(shear, nij);
            shear[0] += dt * vtr0; shear[1] += dt * vtr1; shear[2] += dt * vtr2;
          }
          for (let kk = 0; kk < 3; kk++) fs[kk] = -sp.kt * shear[kk] + fdamp[kk];
          const magfs = Math.sqrt(fs[0] * fs[0] + fs[1] * fs[1] + fs[2] * fs[2]);
          if (magfs > Fscrit) {
            const shrmag = Math.sqrt(shear[0] * shear[0] + shear[1] * shear[1] + shear[2] * shear[2]);
            if (shrmag !== 0) {
              const magInv = 1 / magfs;
              for (let kk = 0; kk < 3; kk++) {
                shear[kk] = Fscrit * magInv * fs[kk];
                shear[kk] -= fdamp[kk];
                shear[kk] *= -1 / sp.kt;
              }
              for (let kk = 0; kk < 3; kk++) fs[kk] = Fscrit * magInv * fs[kk];
            } else {
              fs[0] = fs[1] = fs[2] = 0;
            }
          }
        } else {
          // classic: shear history with the rotation of the history into the tangent plane
          if (update) {
            shear[0] += dt * vtr0; shear[1] += dt * vtr1; shear[2] += dt * vtr2;
          }
          const shrmag = Math.sqrt(shear[0] * shear[0] + shear[1] * shear[1] + shear[2] * shear[2]);
          if (update) {
            const rsht = shear[0] * n0 + shear[1] * n1 + shear[2] * n2;
            for (let kk = 0; kk < 3; kk++) shear[kk] -= rsht * nij[kk];
          }
          const ks = contactRadiusFlag ? sp.kt * contactRadius : sp.kt;
          for (let kk = 0; kk < 3; kk++) fs[kk] = -ks * shear[kk] + fdamp[kk];
          const magfs = Math.sqrt(fs[0] * fs[0] + fs[1] * fs[1] + fs[2] * fs[2]);
          if (magfs > Fscrit) {
            if (shrmag !== 0) {
              const sc = Fscrit / magfs;
              for (let kk = 0; kk < 3; kk++) fs[kk] = fs[kk] * sc;
              const inv = contactRadiusFlag ? -1 / (sp.kt * contactRadius) : -1 / sp.kt;
              for (let kk = 0; kk < 3; kk++) shear[kk] = (fs[kk] - fdamp[kk]) * inv;
            } else {
              fs[0] = fs[1] = fs[2] = 0;
            }
          }
        }
        h[nh] = shear[0]; h[nh + 1] = shear[1]; h[nh + 2] = shear[2];

        let F0 = Fntot * m0 + fs[0], F1 = Fntot * m1 + fs[1], F2 = Fntot * m2 + fs[2];
        // torques r x F on i and F x r on j
        let tI0 = crI1 * F2 - crI2 * F1, tI1 = crI2 * F0 - crI0 * F2, tI2 = crI0 * F1 - crI1 * F0;
        let tJ0 = F1 * crJ2 - F2 * crJ1, tJ1 = F2 * crJ0 - F0 * crJ2, tJ2 = F0 * crJ1 - F1 * crJ0;
        // compute: "scale3(factor_lj, forces)", "scale3(factor_lj, torquesi)", "scale3(factor_lj, torquesj)" on the
        // torques calculate_forces made from the unscaled forces
        if (factorLJ !== 1) {
          F0 *= factorLJ; F1 *= factorLJ; F2 *= factorLJ;
          tI0 *= factorLJ; tI1 *= factorLJ; tI2 *= factorLJ;
          tJ0 *= factorLJ; tJ1 *= factorLJ; tJ2 *= factorLJ;
        }

        f[3 * a] += F0; f[3 * a + 1] += F1; f[3 * a + 2] += F2;
        f[3 * b] -= F0; f[3 * b + 1] -= F1; f[3 * b + 2] -= F2;
        tq[3 * a] += tI0; tq[3 * a + 1] += tI1; tq[3 * a + 2] += tI2;
        tq[3 * b] += tJ0; tq[3 * b + 1] += tJ1; tq[3 * b + 2] += tJ2;
        // ev_tally_xyz: virial of the pair with r = x_i - x_j and the force on i
        const vv = acc.virial;
        vv[0] += dx0 * F0; vv[1] += dx1 * F1; vv[2] += dx2 * F2;
        vv[3] += dx0 * F1; vv[4] += dx0 * F2; vv[5] += dx1 * F2;
        if (pc.vatom) tallyAtomXyz(pc, a, b, 0, F0, F1, F2, dx0, dx1, dx2);
      }
    }
    // the pairs this call did not meet in the neighbour list leave the store
    for (const [key, rec] of hist) if (rec.stamp !== stamp) hist.delete(key);
    nb.reverseSum(tq, 3, s.torque!);
  }

  /**
   * As native LAMMPS does (src/ASPHERE/pair_granular_superellipsoid.cpp single at b332f28): the energy is 0.0 and fforce is
   * 0 (native assigns fforce only on the factor_lj == 0 and no-contact early returns, and returns 0.0 for the energy).
   */
  single(_i: number, _j: number, _itype: number, _jtype: number, _rsq: number, _factorCoul: number, _factorLJ: number, _qi = 0, _qj = 0): { eng: number; fforce: number } {
    return { eng: 0, fforce: 0 };
  }

  dataCoeffs(): string[] | null { return null; }
}

/**
 * The pair_coeff arguments after I and J: normal model, optional damping, tangential, cutoff and limit_damping. As
 * native LAMMPS does (src/ASPHERE/pair_granular_superellipsoid.cpp PairGranularSuperellipsoid::coeff at b332f28): each
 * model's word count, its values (utils::numeric) and checks, with native's messages.
 */
const parseSpec = (a: string[]): { spec: Spec; cutoff: number } => {
  const spec: Spec = { normal: HOOKE, kn: 0, gamman: 0, damping: -1, tangential: -1, kt: 0, xt: 0, xmu: 0, limitDamping: false };
  let cutoff = -1;
  let iarg = 0;
  const need = (n: number) => {
    if (iarg + n > a.length) throw missingArgs('pair granular/superellipsoid');
  };
  if (a[iarg] === 'hooke' || a[iarg] === 'hertz') {
    spec.normal = a[iarg] === 'hooke' ? HOOKE : HERTZ;
    need(3);
    spec.kn = numeric(a[iarg + 1]);
    spec.gamman = numeric(a[iarg + 2]);
    if (spec.kn < 0 || spec.gamman < 0) throw new StyleError('Illegal linear normal model');
    iarg += 3;
  } else {
    throw new StyleError(`Unknown normal model ${a[iarg]}`);
  }
  while (iarg < a.length) {
    const w = a[iarg];
    if (w === 'tangential') {
      iarg++;
      const t = a[iarg];
      if (t !== 'linear_history' && t !== 'classic') throw new StyleError(`Unknown tangential model ${t ?? ''}`);
      spec.tangential = t === 'classic' ? CLASSIC : LINEAR_HISTORY;
      need(4);
      spec.kt = numeric(a[iarg + 1]);
      spec.xt = numeric(a[iarg + 2]);
      spec.xmu = numeric(a[iarg + 3]);
      if (spec.kt < 0 || spec.xt < 0 || spec.xmu < 0) throw new StyleError('Illegal linear tangential model');
      iarg += 4;
    } else if (w === 'damping') {
      iarg++;
      const d = a[iarg];
      if (d === 'mass_velocity') spec.damping = MASS_VELOCITY;
      else if (d === 'viscoelastic') spec.damping = VISCOELASTIC;
      // native's message for an unknown damping model says "normal model"
      else throw new StyleError(`Unknown normal model ${d ?? ''}`);
      iarg += 1;
    } else if (w === 'rolling' || w === 'twisting' || w === 'heat') {
      throw new StyleError(`${w[0].toUpperCase()}${w.slice(1)} models not yet implemented for superellipsoids`);
    } else if (w === 'cutoff') {
      if (iarg + 1 >= a.length) throw new StyleError('Illegal pair_coeff command, not enough parameters for cutoff keyword');
      cutoff = numeric(a[iarg + 1]);
      iarg += 2;
    } else if (w === 'limit_damping') {
      spec.limitDamping = true;
      iarg += 1;
    } else {
      throw new StyleError(`Illegal pair_coeff command ${w}`);
    }
  }
  // native leaves the tangential model unset (uninitialized) without the keyword; the engine requires it
  if (spec.tangential < 0) throw new StyleError('pair_coeff granular/superellipsoid needs the tangential keyword');
  if (spec.damping < 0) spec.damping = VISCOELASTIC;
  return { spec, cutoff };
};
