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
 */
import { Pair, StyleError, tallyAtomXyz, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { SUPER_BLOCK, SUPER_INERTIA } from '../../engine/superellipsoid';
import type { SimState } from '../../engine/types';
import {
  FORMULATION_ALGEBRAIC, FORMULATION_GEOMETRIC, checkOrientedBoundingBoxes, determineContactPoint, gaussianCurvature,
  meanCurvature, overlapDistance, type M3, type V3,
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

/** Contact history of one atom-ID pair: the flag of the last step, the contact point (4) and the shear and axis entries. */
interface Hist {
  touch: number;
  h: Float64Array;
}

const sub3 = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len3 = (a: V3): number => Math.sqrt(dot3(a, a));

/** Rotation matrix of a unit quaternion (w i j k), the MathExtra quat_to_mat. */
const quatToMat = (w: number, i: number, j: number, k: number): M3 => [
  [1 - 2 * (j * j + k * k), 2 * (i * j - w * k), 2 * (i * k + w * j)],
  [2 * (i * j + w * k), 1 - 2 * (i * i + k * k), 2 * (j * k - w * i)],
  [2 * (i * k - w * j), 2 * (j * k + w * i), 1 - 2 * (i * i + j * j)],
];

/** Angular velocity from the angular momentum and the principal moments (the MathExtra angmom_to_omega). */
const angmomToOmega = (L: V3, R: M3, I: V3): V3 => {
  const e = [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]];
  const wb = [0, 0, 0];
  for (let k = 0; k < 3; k++) wb[k] = I[k] === 0 ? 0 : dot3(L, e[k] as V3) / I[k];
  return [
    wb[0] * e[0][0] + wb[1] * e[1][0] + wb[2] * e[2][0],
    wb[0] * e[0][1] + wb[1] * e[1][1] + wb[2] * e[2][1],
    wb[0] * e[0][2] + wb[1] * e[1][2] + wb[2] * e[2][2],
  ];
};

/** Rotate v so that it is perpendicular to the unit vector n, keeping its magnitude (GranSubMod rotate_rescale). */
const rotateRescale = (v: number[], n: V3): void => {
  const rsht = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
  const shrmag = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  for (let k = 0; k < 3; k++) v[k] -= rsht * n[k];
  const prjmag = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const f = prjmag > 0 ? shrmag / prjmag : 0;
  for (let k = 0; k < 3; k++) v[k] *= f;
};

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
  private hist = new Map<string, Hist>();
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
    if (args.length < 3) throw new StyleError('Incorrect args for pair coefficients (usage: pair_coeff I J normal-model args tangential ...)');
    const [ilo, ihi] = boundsOf(args[0], this.ntypes);
    const [jlo, jhi] = boundsOf(args[1], this.ntypes);
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
    if (!ctx.ghostVelocity) throw new StyleError('Pair granular/superellipsoid requires ghost atoms store velocity (use comm_modify vel yes)');
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
    const tq = new Float64Array(3 * pc.nall);
    const next = new Map<string, Hist>();
    const size = this.sizeHistory, nh = this.defaultHist;
    const acc = pc.acc;
    const formulation = this.geometric ? FORMULATION_GEOMETRIC : FORMULATION_ALGEBRAIC;

    for (let i = 0; i < list.inum; i++) {
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        let a = i, b = list.neighbors[k] & NEIGHMASK;
        let oa = owner[a], ob = owner[b];
        if (ids[oa] > ids[ob]) {
          [a, b] = [b, a];
          [oa, ob] = [ob, oa];
        }
        const pm = this.pm[type[oa] * nt + type[ob]];
        if (!pm) continue;
        const key = `${ids[oa]}:${ids[ob]}`;
        let rec = this.hist.get(key);
        if (!rec) rec = { touch: 0, h: new Float64Array(size) };
        next.set(key, rec);
        const h = rec.h;

        const xi: V3 = [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
        const xj: V3 = [x[3 * b], x[3 * b + 1], x[3 * b + 2]];
        const dx = sub3(xi, xj);
        const rsq = dot3(dx, dx);
        const radsum = boundRadius(shape, oa) + boundRadius(shape, ob);
        const pa = this.particle(oa, shape, block, quat);
        const pb = this.particle(ob, shape, block, quat);
        const touchjj = rec.touch;

        // check_contact: bounding sphere, oriented bounding box, then the midway contact point
        let touching = false;
        let shapeI = pa.shape, blockI = pa.block, shapeJ = pb.shape, blockJ = pb.block;
        const X0 = [0, 0, 0, 0];
        const nij: V3 = [0, 0, 0];
        if (rsq < radsum * radsum) {
          let skip = false;
          if (this.boundingBox) {
            const axis = Math.trunc(h[4]);
            const newAxis = checkOrientedBoundingBoxes(xi, pa.R, shapeI, xj, pb.R, shapeJ, axis);
            if (newAxis !== -1) {
              skip = true;
              if (update) h[4] = newAxis;
            }
          }
          if (!skip) {
            if (touchjj === 1) {
              X0[0] = h[0] + xi[0];
              X0[1] = h[1] + xi[1];
              X0[2] = h[2] + xi[2];
              X0[3] = h[3];
              const status = determineContactPoint(
                { x: xi, R: pa.R, shape: shapeI, block: blockI, flag: pa.flag },
                { x: xj, R: pb.R, shape: shapeJ, block: blockJ, flag: pb.flag },
                X0, nij, formulation);
              touching = status === 0;
            } else {
              // new contact: initial guess by morphing the particles from spheres to their shapes (C++ new contact branch)
              const shape0I = [pa.shape[0], pa.shape[1], pa.shape[2]] as V3;
              const shape0J = [pb.shape[0], pb.shape[1], pb.shape[2]] as V3;
              const block0I: [number, number] = [pa.block[0], pa.block[1]];
              const block0J: [number, number] = [pb.block[0], pb.block[1]];
              const reqi = Math.cbrt(shape0I[0] * shape0I[1] * shape0I[2]);
              const reqj = Math.cbrt(shape0J[0] * shape0J[1] * shape0J[2]);
              const rsuminv = 1 / (reqi + reqj);
              X0[0] = (reqj * rsuminv) * xi[0] + (reqi * rsuminv) * xj[0];
              X0[1] = (reqj * rsuminv) * xi[1] + (reqi * rsuminv) * xj[1];
              X0[2] = (reqj * rsuminv) * xi[2] + (reqi * rsuminv) * xj[2];
              X0[3] = reqj / reqi;
              shapeI = [0, 0, 0];
              shapeJ = [0, 0, 0];
              blockI = [0, 0];
              blockJ = [0, 0];
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
                blockI = [2 + frac * (block0I[0] - 2), 2 + frac * (block0I[1] - 2)];
                blockJ = [2 + frac * (block0J[0] - 2), 2 + frac * (block0J[1] - 2)];
                // the first guess uses the ellipsoid form for both particles (C++ iter_ig == 1)
                const flagI = iter === 1 ? 0 : pa.flag, flagJ = iter === 1 ? 0 : pb.flag;
                const status = determineContactPoint(
                  { x: xi, R: pa.R, shape: shapeI, block: blockI, flag: flagI },
                  { x: xj, R: pb.R, shape: shapeJ, block: blockJ, flag: flagJ },
                  X0, nij, formulation);
                if (status === 0) touching = true;
                else if (status === 1) touching = false;
              }
              // after the last guess the morphed shape is the shape itself (frac = 1)
              shapeI = shape0I;
              shapeJ = shape0J;
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

        const quatI = [quat[4 * oa], quat[4 * oa + 1], quat[4 * oa + 2], quat[4 * oa + 3]];
        const quatJ = [quat[4 * ob], quat[4 * ob + 1], quat[4 * ob + 2], quat[4 * ob + 3]];
        const RI = quatToMat(quatI[0], quatI[1], quatI[2], quatI[3]);
        const RJ = quatToMat(quatJ[0], quatJ[1], quatJ[2], quatJ[3]);
        const omegaI = angmomToOmega([angmom[3 * oa], angmom[3 * oa + 1], angmom[3 * oa + 2]], RI, [inertia[3 * oa], inertia[3 * oa + 1], inertia[3 * oa + 2]]);
        const omegaJ = angmomToOmega([angmom[3 * ob], angmom[3 * ob + 1], angmom[3 * ob + 2]], RJ, [inertia[3 * ob], inertia[3 * ob + 1], inertia[3 * ob + 2]]);
        const vI: V3 = [v[3 * oa], v[3 * oa + 1], v[3 * oa + 2]];
        const vJ: V3 = [v[3 * ob], v[3 * ob + 1], v[3 * ob + 2]];
        const sp = pm;
        const normalModel = sp.normal, dampModel = sp.damping, tangModel = sp.tangential;
        const contactRadiusFlag = normalModel === HERTZ || dampModel === VISCOELASTIC;
        const flagI = blockFlag(blockI[0], blockI[1]), flagJ = blockFlag(blockJ[0], blockJ[1]);
        const nji: V3 = [-nij[0], -nij[1], -nij[2]];

        // calculate_forces: store the contact point relative to atom i for the next step
        if (update) {
          h[0] = X0[0] - xi[0];
          h[1] = X0[1] - xi[1];
          h[2] = X0[2] - xi[2];
          h[3] = X0[3];
        }
        const overlapI = overlapDistance(shapeI, blockI, RI, flagI, X0.slice(0, 3) as V3, nij, xi);
        const overlapJ = overlapDistance(shapeJ, blockJ, RJ, flagJ, X0.slice(0, 3) as V3, nji, xj);
        const crI = sub3(X0.slice(0, 3) as V3, xi);
        const crJ = sub3(X0.slice(0, 3) as V3, xj);
        const cvI = add3(vI, cross3(omegaI, crI));
        const cvJ = add3(vJ, cross3(omegaJ, crJ));
        const vr = sub3(cvI, cvJ);
        const vnnr = dot3(vr, nij);
        const vn: V3 = [vnnr * nij[0], vnnr * nij[1], vnnr * nij[2]];
        const vtr = sub3(vr, vn);

        let contactRadius = 0;
        if (contactRadiusFlag) {
          const surfI: V3 = [overlapI * nij[0] + X0[0], overlapI * nij[1] + X0[1], overlapI * nij[2] + X0[2]];
          const surfJ: V3 = [overlapJ * nji[0] + X0[0], overlapJ * nji[1] + X0[1], overlapJ * nji[2] + X0[2]];
          const curvI = this.curvatureGaussian ? gaussianCurvature(shapeI, blockI, flagI, RI, surfI, xi) : meanCurvature(shapeI, blockI, flagI, RI, surfI, xi);
          const curvJ = this.curvatureGaussian ? gaussianCurvature(shapeJ, blockJ, flagJ, RJ, surfJ, xj) : meanCurvature(shapeJ, blockJ, flagJ, RJ, surfJ, xj);
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
        const hist = h.subarray(nh, nh + 3);
        const shear = [hist[0], hist[1], hist[2]];
        let fs: V3;
        if (tangModel === LINEAR_HISTORY) {
          if (update) {
            const rsht = shear[0] * nij[0] + shear[1] * nij[1] + shear[2] * nij[2];
            const frameUpdate = Math.abs(rsht) * sp.kt > EPSILON * Fscrit;
            if (frameUpdate) rotateRescale(shear, nij);
            for (let kk = 0; kk < 3; kk++) shear[kk] += dt * vtr[kk];
          }
          fs = [-sp.kt * shear[0], -sp.kt * shear[1], -sp.kt * shear[2]];
          const fdamp: V3 = [-dampt * vtr[0], -dampt * vtr[1], -dampt * vtr[2]];
          fs = add3(fs, fdamp);
          const magfs = len3(fs);
          if (magfs > Fscrit) {
            const shrmag = len3(shear as V3);
            if (shrmag !== 0) {
              const magInv = 1 / magfs;
              for (let kk = 0; kk < 3; kk++) {
                shear[kk] = Fscrit * magInv * fs[kk];
                shear[kk] -= fdamp[kk];
                shear[kk] *= -1 / sp.kt;
              }
              fs = [Fscrit * magInv * fs[0], Fscrit * magInv * fs[1], Fscrit * magInv * fs[2]];
            } else {
              fs = [0, 0, 0];
            }
          }
          for (let kk = 0; kk < 3; kk++) hist[kk] = shear[kk];
        } else {
          // classic: shear history with the rotation of the history into the tangent plane
          if (update) {
            for (let kk = 0; kk < 3; kk++) shear[kk] += dt * vtr[kk];
          }
          const shrmag = len3(shear as V3);
          if (update) {
            const rsht = shear[0] * nij[0] + shear[1] * nij[1] + shear[2] * nij[2];
            for (let kk = 0; kk < 3; kk++) shear[kk] -= rsht * nij[kk];
          }
          const ks = contactRadiusFlag ? sp.kt * contactRadius : sp.kt;
          fs = [-ks * shear[0], -ks * shear[1], -ks * shear[2]];
          const fdamp: V3 = [-dampt * vtr[0], -dampt * vtr[1], -dampt * vtr[2]];
          fs = add3(fs, fdamp);
          const magfs = len3(fs);
          if (magfs > Fscrit) {
            if (shrmag !== 0) {
              const sc = Fscrit / magfs;
              fs = [fs[0] * sc, fs[1] * sc, fs[2] * sc];
              const inv = contactRadiusFlag ? -1 / (sp.kt * contactRadius) : -1 / sp.kt;
              for (let kk = 0; kk < 3; kk++) shear[kk] = (fs[kk] - fdamp[kk]) * inv;
            } else {
              fs = [0, 0, 0];
            }
          }
          for (let kk = 0; kk < 3; kk++) hist[kk] = shear[kk];
        }

        const forces: V3 = [Fntot * nji[0] + fs[0], Fntot * nji[1] + fs[1], Fntot * nji[2] + fs[2]];
        const torqueI = cross3(crI, forces);
        const torqueJ = cross3(forces, crJ);

        f[3 * a] += forces[0]; f[3 * a + 1] += forces[1]; f[3 * a + 2] += forces[2];
        f[3 * b] -= forces[0]; f[3 * b + 1] -= forces[1]; f[3 * b + 2] -= forces[2];
        tq[3 * a] += torqueI[0]; tq[3 * a + 1] += torqueI[1]; tq[3 * a + 2] += torqueI[2];
        tq[3 * b] += torqueJ[0]; tq[3 * b + 1] += torqueJ[1]; tq[3 * b + 2] += torqueJ[2];
        // ev_tally_xyz: virial of the pair with r = x_i - x_j and the force on i
        const vv = acc.virial;
        vv[0] += dx[0] * forces[0]; vv[1] += dx[1] * forces[1]; vv[2] += dx[2] * forces[2];
        vv[3] += dx[0] * forces[1]; vv[4] += dx[0] * forces[2]; vv[5] += dx[1] * forces[2];
        if (pc.vatom) tallyAtomXyz(pc, a, b, 0, forces[0], forces[1], forces[2], dx[0], dx[1], dx[2]);
      }
    }
    this.hist = next;
    nb.reverseSum(tq, 3, s.torque!);
  }

  /** Rotation, half-diameters, exponents and block type of owned atom o. */
  private particle(o: number, shape: Float64Array, block: Float64Array, quat: Float64Array): { R: M3; shape: V3; block: [number, number]; flag: number } {
    const b1 = block[2 * o], b2 = block[2 * o + 1];
    const R = quatToMat(quat[4 * o], quat[4 * o + 1], quat[4 * o + 2], quat[4 * o + 3]);
    return { R, shape: [shape[3 * o], shape[3 * o + 1], shape[3 * o + 2]], block: [b1, b2], flag: blockFlag(b1, b2) };
  }

  dataCoeffs(): string[] | null { return null; }
}

const add3 = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

/** Type bounds "I", "*", "*n", "n*", "m*n" as in pair_coeff. */
const boundsOf = (word: string, n: number): [number, number] => {
  const bad = () => new StyleError(`invalid type '${word}' (expected 1..${n}, or a wildcard *, *n, n*, m*n)`);
  if (word === '*') return [1, n];
  const star = word.indexOf('*');
  const num = (w: string) => {
    if (!/^\d+$/.test(w)) throw bad();
    return Number(w);
  };
  let lo: number, hi: number;
  if (star < 0) lo = hi = num(word);
  else {
    const a = word.slice(0, star), b = word.slice(star + 1);
    lo = a === '' ? 1 : num(a);
    hi = b === '' ? n : num(b);
  }
  if (lo < 1 || hi > n || lo > hi) throw bad();
  return [lo, hi];
};

/** The pair_coeff arguments after I and J: normal model, optional damping, tangential, cutoff and limit_damping. */
const parseSpec = (a: string[]): { spec: Spec; cutoff: number } => {
  const spec: Spec = { normal: HOOKE, kn: 0, gamman: 0, damping: -1, tangential: -1, kt: 0, xt: 0, xmu: 0, limitDamping: false };
  let cutoff = -1;
  let iarg = 0;
  const need = (n: number, what: string) => {
    if (iarg + n > a.length) throw new StyleError(`pair_coeff: missing arguments for ${what}`);
  };
  const num = (w: string | undefined, what: string): number => {
    if (w === undefined || !Number.isFinite(Number(w))) throw new StyleError(`pair_coeff: ${what} needs a number`);
    return Number(w);
  };
  if (a[iarg] === 'hooke' || a[iarg] === 'hertz') {
    spec.normal = a[iarg] === 'hooke' ? HOOKE : HERTZ;
    need(3, a[iarg]);
    spec.kn = num(a[iarg + 1], 'k_n');
    spec.gamman = num(a[iarg + 2], 'eta_n0');
    if (spec.kn < 0 || spec.gamman < 0) throw new StyleError('Illegal linear normal model');
    iarg += 3;
  } else {
    throw new StyleError(`Unknown normal model '${a[iarg] ?? ''}' for pair granular/superellipsoid (hooke or hertz)`);
  }
  while (iarg < a.length) {
    const w = a[iarg];
    if (w === 'tangential') {
      const t = a[iarg + 1];
      if (t !== 'linear_history' && t !== 'classic') throw new StyleError(`Unknown tangential model '${t ?? ''}' for pair granular/superellipsoid (linear_history or classic)`);
      spec.tangential = t === 'classic' ? CLASSIC : LINEAR_HISTORY;
      need(4, t);
      spec.kt = num(a[iarg + 2], 'k_t');
      spec.xt = num(a[iarg + 3], 'x_gamma_t');
      spec.xmu = num(a[iarg + 4], 'mu_s');
      if (spec.kt < 0 || spec.xt < 0 || spec.xmu < 0) throw new StyleError('Illegal linear tangential model');
      iarg += 5;
    } else if (w === 'damping') {
      const d = a[iarg + 1];
      if (d === 'mass_velocity') spec.damping = MASS_VELOCITY;
      else if (d === 'viscoelastic') spec.damping = VISCOELASTIC;
      else throw new StyleError(`Unknown damping model '${d ?? ''}' for pair granular/superellipsoid (mass_velocity or viscoelastic)`);
      iarg += 2;
    } else if (w === 'rolling' || w === 'twisting' || w === 'heat') {
      throw new StyleError(`${w[0].toUpperCase()}${w.slice(1)} models are not implemented for pair granular/superellipsoid (the LAMMPS style refuses them too)`);
    } else if (w === 'cutoff') {
      cutoff = num(a[iarg + 1], 'cutoff');
      iarg += 2;
    } else if (w === 'limit_damping') {
      spec.limitDamping = true;
      iarg += 1;
    } else {
      throw new StyleError(`Illegal pair_coeff command keyword '${w}' for pair granular/superellipsoid`);
    }
  }
  if (spec.tangential < 0) throw new StyleError('pair_coeff granular/superellipsoid needs the tangential keyword');
  if (spec.damping < 0) spec.damping = VISCOELASTIC;
  return { spec, cutoff };
};
