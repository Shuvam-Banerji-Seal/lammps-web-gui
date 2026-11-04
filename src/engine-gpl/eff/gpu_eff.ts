/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/pair_eff_cut.cpp (PairEffCut::compute:
 * the electron kinetic, fixed-core, electron, pseudo-core and Pauli terms, the cutoff spline and limit/eradius) and
 * src/EFF/pair_eff_inline.h (the Erf polynomials and the interaction functions), as a GPU kernel for the engine's WebGPU
 * path (src/engine/gpu/atomKernel.ts). Contributing authors (from the sources): Andres Jaramillo-Botero, Hai Xiao, Julius
 * Su (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The kernel takes each atom's partners from the cells around it (a full list: every pair twice) and adds, for the atom
 * i of the pair (i, j), what the C++ half loop gives i: the pair force on i, the radial force the pair's branch gives i
 * (the C++ code assigns each branch's radial forces to i or to j), half of the pair's energy, virial and compute-pair
 * terms, and once the atom's own terms (the kinetic energy of an electron, the fixed-core self terms, the limit/eradius
 * restraint). The pair functions are symmetric under the exchange of i and j where the C++ code calls them with the
 * roles swapped, so the two rows of a pair agree. fp32 on the GPU; the per-atom sums, the radial virial of
 * pressure/evirials and the totals are fp64 in finish(). fp64() states the same arithmetic in double precision with the
 * functions of ./math.ts (the tests run the accelerated loop with it in place of the GPU).
 */
import { ATOM_KERNEL_OUT, ATOM_KERNEL_WORKGROUP, WGSL_ATOM_KERNEL_HEAD, type GpuAtomKernel } from '../../engine/gpu/atomKernel';
import { NO_IMAGE } from '../../engine/gpu/cellSort';
import { ELECTRON_RADIUS, ELECTRON_RFORCE, ELECTRON_SPIN } from '../../engine/atoms';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';
import type { PairEffCut } from './pair_eff_cut';
import {
  kinElec, elecNucNuc, elecNucElec, elecElecElec, pauliElecElec, smallRForce, cutoff, dcutoff,
  elecCoreNuc, elecCoreCore, elecCoreElec, pauliCoreElec, pauliCorePElec,
} from './math';

/** coef: the cutoff^2 table (stride ntypes + 1), then six values per type: the ECP A..E and 1 for an s-type core. */
const ECP_VALUES = 6;
/** P.u0.w flags. */
const LIMIT_ERADIUS = 1, PRESSURE_EVIRIALS = 2;

/** The fixed parameters of a run (gpuKernel). */
interface EffParams {
  nt: number;
  cutsq: Float64Array;
  /** per type: A..E and s-type (1/0), or null */
  ecp: (number[] | null)[];
  qqrd2e: number;
  hh: number;
  h2e: number;
  limitEradius: boolean;
  pressureEvirials: boolean;
}

/** One ordered pair (i, j) as the half loop's row of i gives it, for i alone. */
interface PairOut {
  /** the pair force after the spline (fpair of pair_eff_cut.cpp) */
  fpair: number;
  /** the pair energy after the spline */
  energy: number;
  /** the radial force the branch gives i (with the spline) */
  erfI: number;
  /** spline * epauli and spline * ecoul (pvector) */
  pauli: number;
  coul: number;
}

const PO: PairOut = { fpair: 0, energy: 0, erfI: 0, pauli: 0, coul: 0 };

/** pair_eff_cut.cpp compute, one pair of the row of i (fp64; the C++ statement order). */
const pairForI = (P: EffParams, si: number, sj: number, qi: number, qj: number, ei: number, ej: number, ti: number, tj: number, rc: number): PairOut => {
  const { qqrd2e, hh, h2e } = P;
  const taper = Math.sqrt(P.cutsq[ti * P.nt + tj]);
  const dist = rc / taper;
  const spline = cutoff(dist);
  const dspline = dcutoff(dist) / taper;
  let ecoul = 0, epauli = 0, fpair = 0, sFpair = 0;
  let e1rforce = 0, e2rforce = 0, sE1 = 0, sE2 = 0;
  let ecpEpauli = 0, ecpFpair = 0;
  let erfI = 0;
  const absI = Math.abs(si), absJ = Math.abs(sj);
  if (si === 0 && sj === 0) {
    const r = elecNucNuc(qi * qj, rc);
    ecoul += r.e; fpair += r.f;
  } else if (si === 2 && sj === 0) {
    const r = elecNucNuc(qi * qj, rc);
    ecoul += r.e; fpair += r.f;
    for (let t = 0; t < 2; t++) { const n = elecNucElec(qj, rc, ei); ecoul += n.e; fpair += n.f; }
  } else if (si === 0 && sj === 2) {
    const r = elecNucNuc(qi * qj, rc);
    ecoul += r.e; fpair += r.f;
    for (let t = 0; t < 2; t++) { const n = elecNucElec(qi, rc, ej); ecoul += n.e; fpair += n.f; }
  } else if (si === 0 && absJ === 1) {
    const n = elecNucElec(qi, rc, ej);
    ecoul += n.e; fpair += n.f;
  } else if (absI === 1 && sj === 0) {
    const n = elecNucElec(qj, rc, ei);
    ecoul += n.e; fpair += n.f; e1rforce += n.f1;
    erfI = spline * qqrd2e * e1rforce;
  } else if (absI === 1 && absJ === 1) {
    const ee = elecElecElec(rc, ei, ej);
    ecoul += ee.e; fpair += ee.f; e1rforce += ee.f1;
    const pa = pauliElecElec(si === sj ? 1 : 0, rc, ei, ej);
    epauli += pa.ep; sFpair -= pa.f; sE1 -= pa.f1;
    epauli *= hh;
    sFpair *= hh;
    erfI = spline * (qqrd2e * e1rforce + hh * sE1);
  } else if (si === 2 && absJ === 1) {
    const n = elecNucElec(qi, rc, ej);
    ecoul += n.e; fpair += n.f;
    for (let t = 0; t < 2; t++) { const ee = elecElecElec(rc, ei, ej); ecoul += ee.e; fpair += ee.f; }
    for (let t = 0; t < 2; t++) { const pa = pauliElecElec(t === 0 ? 0 : 1, rc, ei, ej); epauli += pa.ep; sFpair -= pa.f; }
    epauli *= hh;
    sFpair *= hh;
  } else if (absI === 1 && sj === 2) {
    const n = elecNucElec(qj, rc, ei);
    ecoul += n.e; fpair += n.f; e2rforce += n.f1;
    for (let t = 0; t < 2; t++) { const ee = elecElecElec(rc, ej, ei); ecoul += ee.e; fpair += ee.f; e2rforce += ee.f2; }
    for (let t = 0; t < 2; t++) { const pa = pauliElecElec(t === 0 ? 0 : 1, rc, ej, ei); epauli += pa.ep; sFpair -= pa.f; sE2 -= pa.f2; }
    epauli *= hh;
    sFpair *= hh;
    erfI = spline * (qqrd2e * e2rforce + hh * sE2);
  } else if (si === 2 && sj === 2) {
    const r = elecNucNuc(qi * qj, rc);
    ecoul += r.e; fpair += r.f;
    for (let t = 0; t < 2; t++) { const n = elecNucElec(qi, rc, ej); ecoul += n.e; fpair += n.f; }
    for (let t = 0; t < 2; t++) { const n = elecNucElec(qj, rc, ei); ecoul += n.e; fpair += n.f; }
    for (let t = 0; t < 4; t++) { const ee = elecElecElec(rc, ei, ej); ecoul += ee.e; fpair += ee.f; }
    for (let t = 0; t < 2; t++) { const pa = pauliElecElec(t === 0 ? 0 : 1, rc, ei, ej); epauli += pa.ep; sFpair -= pa.f; }
    epauli *= 2;
    sFpair *= 2;
    epauli *= hh;
    sFpair *= hh;
  } else if (si === 3 && sj === 0) {
    const r = elecCoreNuc(qi * qj, rc, ei);
    ecoul += r.e; fpair += r.f;
  } else if (si === 0 && sj === 3) {
    const r = elecCoreNuc(qi * qj, rc, ej);
    ecoul += r.e; fpair += r.f;
  } else if (si === 3 && (absJ === 1 || sj === 2)) {
    // the radial force goes to the electron j only
    const [A, B, C, D, Ep, sType] = P.ecp[ti]!;
    const pauli = (re: number) => (sType ? pauliCoreElec(rc, re, A, B, C) : pauliCorePElec(rc, re, A, B, C, D, Ep));
    if (absJ === 1) {
      const n = elecCoreElec(qi, rc, ei, ej);
      ecoul += n.e; fpair += n.f;
      const pa = pauli(ej);
      ecpEpauli += pa.ep; ecpFpair += pa.f;
    } else {
      const nn = elecCoreNuc(qi * qj, rc, ej);
      ecoul += nn.e; fpair += nn.f;
      for (let t = 0; t < 2; t++) { const n = elecCoreElec(qi, rc, ei, ej); ecoul += n.e; fpair += n.f; }
      for (let t = 0; t < 2; t++) { const pa = pauli(ej); ecpEpauli += pa.ep; ecpFpair += pa.f; }
    }
    ecpEpauli *= h2e;
    ecpFpair *= h2e;
  } else if ((absI === 1 || si === 2) && sj === 3) {
    const [A, B, C, D, Ep, sType] = P.ecp[tj]!;
    const pauli = (re: number) => (sType ? pauliCoreElec(rc, re, A, B, C) : pauliCorePElec(rc, re, A, B, C, D, Ep));
    let ecpE1 = 0;
    if (absI === 1) {
      const n = elecCoreElec(qj, rc, ej, ei);
      ecoul += n.e; fpair += n.f; e1rforce += n.f2;
      const pa = pauli(ei);
      ecpEpauli += pa.ep; ecpFpair += pa.f; ecpE1 += pa.f2;
    } else {
      const nn = elecCoreNuc(qi * qj, rc, ei);
      ecoul += nn.e; fpair += nn.f;
      for (let t = 0; t < 2; t++) { const n = elecCoreElec(qj, rc, ej, ei); ecoul += n.e; fpair += n.f; e1rforce += n.f2; }
      for (let t = 0; t < 2; t++) { const pa = pauli(ei); ecpEpauli += pa.ep; ecpFpair += pa.f; ecpE1 += pa.f2; }
    }
    ecpEpauli *= h2e;
    ecpFpair *= h2e;
    erfI = spline * (qqrd2e * e1rforce + h2e * ecpE1);
  } else if (si === 3 && sj === 3) {
    const r = elecCoreCore(qi * qj, rc, ei, ej);
    ecoul += r.e; fpair += r.f;
  }
  ecoul *= qqrd2e;
  fpair *= qqrd2e;
  epauli += ecpEpauli;
  const energy0 = ecoul + epauli;
  fpair = fpair + sFpair + ecpFpair;
  PO.fpair = fpair * spline - energy0 * dspline;
  PO.energy = spline * energy0;
  PO.erfI = erfI;
  PO.pauli = spline * epauli;
  PO.coul = spline * ecoul;
  return PO;
};

/** The kernel's per-atom outputs in fp64 (ATOM_KERNEL_OUT per atom, atom order). */
const effFp64 = (P: EffParams, s: SimState): Float64Array => {
  const n = s.n, x = s.x, type = s.type, q = s.q!;
  const spin = s.custom.get(ELECTRON_SPIN)!.data, erad = s.custom.get(ELECTRON_RADIUS)!.data;
  const out = new Float64Array(ATOM_KERNEL_OUT * n);
  const box = s.box;
  const L = [0, 1, 2].map((d) => (box.periodic[d] || (d === 2 && s.dimension === 2) ? box.hi[d] - box.lo[d] : NO_IMAGE));
  const [xy, xz, yz] = box.triclinic ? box.tilt : [0, 0, 0];
  const two = s.dimension === 2;
  const lim = effLimit(P, s);
  for (let i = 0; i < n; i++) {
    const si = spin[i], ei = erad[i], qi = q[i], ti = type[i];
    let fx = 0, fy = 0, fz = 0, erf = 0, e = 0;
    let ke = 0, pauli = 0, coul = 0, restrain = 0;
    let vxx = 0, vyy = 0, vzz = 0, vxy = 0, vxz = 0, vyz = 0;
    if (Math.abs(si) === 1 || si === 2) {
      let eke = 0, epauli = 0, ecoul = 0, e1rforce = 0;
      const kin = kinElec(ei);
      eke += kin.e;
      e1rforce += kin.f;
      if (si === 2) {
        eke *= 2;
        const n1 = elecNucElec(qi, 0.0, ei);
        ecoul += n1.e;
        ecoul += n1.e;
        ecoul += elecElecElec(0.0, ei, ei).e;
        epauli += pauliElecElec(0, 0.0, ei, ei).ep;
        e1rforce = 0;
      }
      eke *= P.hh; ecoul *= P.qqrd2e; epauli *= P.hh; e1rforce *= P.hh;
      erf += e1rforce;
      e += eke + epauli + ecoul;
      ke += eke; pauli += epauli; coul += ecoul;
    }
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      // the minimum image of xj - xi (cellSort.ts WGSL_BOX_IMAGE), then dx = xi - xj
      let ddx = x[3 * j] - x[3 * i], ddy = x[3 * j + 1] - x[3 * i + 1], ddz = two ? 0 : x[3 * j + 2] - x[3 * i + 2];
      const fzc = ddz / L[2];
      const fyc = (ddy - yz * fzc) / L[1];
      const fxc = (ddx - xy * fyc - xz * fzc) / L[0];
      const kx = Math.round(fxc), ky = Math.round(fyc), kz = Math.round(fzc);
      ddx -= kx * L[0] + ky * xy + kz * xz; ddy -= ky * L[1] + kz * yz; ddz -= kz * L[2];
      const dx = -ddx, dy = -ddy, dz = -ddz;
      const rsq = dx * dx + dy * dy + dz * dz;
      if (!(rsq < P.cutsq[ti * P.nt + type[j]])) continue;
      const rc = Math.sqrt(rsq);
      const r = pairForI(P, si, spin[j], qi, q[j], ei, erad[j], ti, type[j], rc);
      let gx: number, gy: number, gz: number;
      if (rc > 0.000001) { const ff = r.fpair / rc; gx = ff * dx; gy = ff * dy; gz = ff * dz; } else [gx, gy, gz] = smallRForce(dx, dy, dz, rc, r.fpair);
      fx += gx; fy += gy; fz += gz;
      erf += r.erfI;
      e += 0.5 * r.energy;
      pauli += 0.5 * r.pauli; coul += 0.5 * r.coul;
      vxx += 0.5 * dx * gx; vyy += 0.5 * dy * gy; vzz += 0.5 * dz * gz;
      vxy += 0.5 * dx * gy; vxz += 0.5 * dx * gz; vyz += 0.5 * dy * gz;
    }
    if (Math.abs(si) === 1 && lim.on && ei > lim.half) {
      const kfactor = P.hh * 1.0;
      const dr = ei - lim.half;
      const errestrain = 0.5 * kfactor * dr * dr;
      restrain += errestrain;
      erf += -kfactor * dr;
      e += errestrain;
    }
    const o = ATOM_KERNEL_OUT * i;
    out[o] = fx; out[o + 1] = fy; out[o + 2] = fz; out[o + 3] = erf; out[o + 4] = e;
    out[o + 5] = ke; out[o + 6] = pauli; out[o + 7] = coul; out[o + 8] = restrain;
    out[o + 9] = vxx; out[o + 10] = vyy; out[o + 11] = vzz; out[o + 12] = vxy; out[o + 13] = vxz; out[o + 14] = vyz;
  }
  return out;
};

/** limit/eradius: on with any periodic dimension, the half of the shortest box edge (pair_eff_cut.cpp compute). */
const effLimit = (P: EffParams, s: SimState): { on: boolean; half: number; lx: number; ly: number; lz: number } => {
  const box = s.box;
  const lx = box.hi[0] - box.lo[0], ly = box.hi[1] - box.lo[1], lz = box.hi[2] - box.lo[2];
  const periodic = box.periodic[0] || box.periodic[1] || box.periodic[2];
  return { on: P.limitEradius && periodic, half: 0.5 * Math.min(lx, Math.min(ly, lz)), lx, ly, lz };
};

/** The WGSL of the kernel: the Erf polynomials, the interaction functions and the per-atom loop. */
const KERNEL_EFF = /* wgsl */ `${WGSL_ATOM_KERNEL_HEAD}
var<private> E1 : array<f32, 13> = array<f32, 13>(1.4831105640848036, -0.30107107338659494, 0.068994830689831566,
  -0.013916271264722188, 0.0024207995224334637, -0.00036586396858480864, 4.8620984432319048e-5, -5.7492565580356848e-6,
  6.1132435784347647e-7, -5.8991015312958434e-8, 5.2070090920686482e-9, -4.2329758799655433e-10, 3.188113506649175e-11);
var<private> E2 : array<f32, 8> = array<f32, 8>(1.0779778520723832, -0.026559890409148673, -0.0014870731466980995,
  -0.00013804014541414386, -1.1280303332287491e-5, -1.1728698427437252e-6, -1.0347615039330462e-7, -1.1899114085892438e-8);
var<private> DE1 : array<f32, 14> = array<f32, 14>(-0.6893799748484185, 0.29593905685116177, -0.087237828075228616,
  0.01995973409183551, -0.0037402004868954903, 0.00059333791236780046, -8.1560801047403878e-5, 9.886099179971884e-6,
  -1.0712092349042906e-6, 1.049094544762605e-7, -9.3709592710387467e-9, 7.6927263488753842e-10,
  -5.841233511455152e-11, 4.1253932917364248e-12);

fn ipoly01(x0 : f32) -> f32 {
  var b0 = 0.0; var b1 = 0.0; var b2 = 0.0;
  let x = x0 * 2.0;
  for (var i = 12; i >= 0; i = i - 1) { b2 = b1; b1 = b0; b0 = x * b1 - b2 + E1[i]; }
  return 0.5 * (b0 - b2);
}
fn ipoly02(x0 : f32) -> f32 {
  var b0 = 0.0; var b1 = 0.0; var b2 = 0.0;
  let x = x0 * 2.0;
  for (var i = 7; i >= 0; i = i - 1) { b2 = b1; b1 = b0; b0 = x * b1 - b2 + E2[i]; }
  return 0.5 * (b0 - b2);
}
fn ipoly1(x0 : f32) -> f32 {
  var b0 = 0.0; var b1 = 0.0; var b2 = 0.0;
  let x = x0 * 2.0;
  for (var i = 13; i >= 0; i = i - 1) { b2 = b1; b1 = b0; b0 = x * b1 - b2 + DE1[i]; }
  return 0.5 * (b0 - b2);
}
// Erf(x)/x and its derivative (ierfoverx1)
fn ierf(x : f32) -> vec2<f32> {
  if (x < 2.0) {
    let t = 0.5 * x * x - 1.0;
    return vec2<f32>(ipoly01(t), ipoly1(t) * x);
  }
  let x2 = x * x;
  let t = (10.5 - x2) / (2.5 + x2);
  let ex = exp(-x2);
  let fv = 1.0 / x - (ex / x2) * ipoly02(t);
  return vec2<f32>(fv, (1.1283791670955126 * ex - fv) / x);
}
fn kinElec(r : f32) -> vec2<f32> { return vec2<f32>(1.5 / (r * r), 3.0 / (r * r * r)); }
fn nucNuc(q : f32, rc : f32) -> vec2<f32> { return vec2<f32>(q / rc, q / (rc * rc)); }
// (E, dE/dr, dE/dr1)
fn nucElec(q : f32, rc : f32, re1 : f32) -> vec3<f32> {
  let a = 1.4142135623730951 / re1;
  let arc = a * rc;
  let g = ierf(arc);
  return vec3<f32>(-q * a * g.x, q * a * a * g.y, -q * (a / re1) * (g.x + arc * g.y));
}
// (E, dE/dr, dE/dr1, dE/dr2)
fn elecElec(rc : f32, re1 : f32, re2 : f32) -> vec4<f32> {
  let re = sqrt(re1 * re1 + re2 * re2);
  let a = 1.4142135623730951 / re;
  let arc = a * rc;
  let g = ierf(arc);
  let fre = a * (g.x + arc * g.y) / (re * re);
  return vec4<f32>(a * g.x, -a * a * g.y, fre * re1, fre * re2);
}
// (Ep, f, f1, f2): the C++ code subtracts the three forces
fn pauliEE(same : bool, rc0 : f32, re10 : f32, re20 : f32) -> vec4<f32> {
  let re1 = re10 * 0.9; let re2 = re20 * 0.9; let rc = rc0 * 1.125;
  let ree = re1 * re1 + re2 * re2;
  let rem = re1 * re1 - re2 * re2;
  let S = (2.82842712474619 / pow(re2 / re1 + re1 / re2, 1.5)) * exp(-rc * rc / ree);
  let t1 = 1.5 * (1.0 / (re1 * re1) + 1.0 / (re2 * re2));
  let t2 = 2.0 * (3.0 * ree - 2.0 * rc * rc) / (ree * ree);
  let tt = t1 - t2;
  let dSdr1 = (-1.5 / re1) * (rem / ree) + 2.0 * re1 * rc * rc / (ree * ree);
  let dSdr2 = (1.5 / re2) * (rem / ree) + 2.0 * re2 * rc * rc / (ree * ree);
  let dSdr = -2.0 * rc / ree;
  let dTdr1 = -3.0 / (re1 * re1 * re1) - 12.0 * re1 / (ree * ree) + 8.0 * re1 * (-2.0 * rc * rc + 3.0 * ree) / (ree * ree * ree);
  let dTdr2 = -3.0 / (re2 * re2 * re2) - 12.0 * re2 / (ree * ree) + 8.0 * re2 * (-2.0 * rc * rc + 3.0 * ree) / (ree * ree * ree);
  let dTdr = 8.0 * rc / (ree * ree);
  var O = 0.0; var dOdS = 0.0;
  let S2 = S * S;
  if (same) {
    O = S2 / (1.0 - S2) + 1.2 * S2 / (1.0 + S2);
    dOdS = 2.0 * S / ((1.0 - S2) * (1.0 - S2)) + 1.2 * 2.0 * S / ((1.0 + S2) * (1.0 + S2));
  } else {
    O = 0.2 * S2 / (1.0 + S2);
    dOdS = 0.2 * 2.0 * S / ((1.0 + S2) * (1.0 + S2));
  }
  let ratio = tt * dOdS * S;
  return vec4<f32>(tt * O, 1.125 * (dTdr * O + ratio * dSdr), 0.9 * (dTdr1 * O + ratio * dSdr1), 0.9 * (dTdr2 * O + ratio * dSdr2));
}
fn splineOf(x : f32) -> f32 { return (((20.0 * x - 70.0) * x + 84.0) * x - 35.0) * x * x * x * x + 1.0; }
fn dsplineOf(x : f32) -> f32 { return (((140.0 * x - 420.0) * x + 420.0) * x - 140.0) * x * x * x; }
// (E, f)
fn coreNuc(q : f32, rc : f32, re1 : f32) -> vec2<f32> {
  let a = 1.4142135623730951 / re1;
  let g = ierf(a * rc);
  return vec2<f32>(q * a * g.x, -q * a * a * g.y);
}
fn coreCore(q : f32, rc : f32, re1 : f32, re2 : f32) -> vec2<f32> {
  let re = sqrt(re1 * re1 + re2 * re2);
  let a = 1.4142135623730951 / re;
  let g = ierf(a * rc);
  return vec2<f32>(q * a * g.x, -q * a * a * g.y);
}
// (E, f, f2), the forces already negated (ElecCoreElec)
fn coreElec(q : f32, rc : f32, re1 : f32, re2 : f32) -> vec3<f32> {
  let re = sqrt(re1 * re1 + re2 * re2);
  let a = 1.4142135623730951 / re;
  let arc = a * rc;
  let g = ierf(arc);
  let fre = q * a * (g.x + arc * g.y) / (re * re);
  return vec3<f32>(-q * a * g.x, q * a * g.y * a, -fre * re2);
}
// (Ep, f, f2) of the ECP Pauli term of the core whose parameters start at coef[o], with an electron of radius re2
fn pauliCore(o : u32, rc : f32, re2 : f32) -> vec3<f32> {
  let A = coef[o]; let B = coef[o + 1u]; let C = coef[o + 2u]; let D = coef[o + 3u]; let Ep = coef[o + 4u];
  if (coef[o + 5u] != 0.0) {
    let rcsq = rc * rc; let ssq = re2 * re2;
    let ex = exp(-B * rcsq / (ssq + C));
    return vec3<f32>(A * ex, 2.0 * A * B * rc * ex / (ssq + C), -2.0 * A * B * re2 * rcsq * ex / ((C + ssq) * (C + ssq)));
  }
  let u = rc - C * re2;
  let w = B / re2 + re2 / B;
  let g = pow(2.0 / w, 5.0);
  let den = Ep + re2 * re2;
  let ex = exp(-D * u * u / den);
  let en = A * g * u * u * ex;
  let dEdrc = A * g * 2.0 * u * ex + en * (-D * 2.0 * u / den);
  let dEdre2 = en * (-5.0 / w * (-B / (re2 * re2) + 1.0 / B)) + A * g * 2.0 * u * (-C) * ex
    + en * (2.0 * D * u * (C * Ep + rc * re2) / (den * den));
  return vec3<f32>(en, -dEdrc, -dEdre2);
}

@compute @workgroup_size(${ATOM_KERNEL_WORKGROUP})
fn force(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) { return; }
  let pi = pos[i];
  let ti = u32(pi.w);
  let xi = ext[i];
  let qi = xi.x; let si = xi.y; let ei = xi.z;
  let qqrd2e = P.u0.x; let hh = P.u0.y; let h2e = P.u0.z; let flags = u32(P.u0.w);
  let nt = P.stride;
  let ecpBase = nt * nt;
  var f = vec3<f32>(0.0);
  var erf = 0.0; var e = 0.0;
  var pv = vec4<f32>(0.0);   // ke pauli coul restrain
  var vd = vec3<f32>(0.0);   // xx yy zz
  var vo = vec3<f32>(0.0);   // xy xz yz
  let absI = abs(si);
  if (absI == 1.0 || si == 2.0) {
    let kin = kinElec(ei);
    var eke = kin.x; var ep = 0.0; var ec = 0.0; var e1 = kin.y;
    if (si == 2.0) {
      eke = eke * 2.0;
      let n1 = nucElec(qi, 0.0, ei);
      ec = ec + n1.x + n1.x + elecElec(0.0, ei, ei).x;
      ep = ep + pauliEE(false, 0.0, ei, ei).x;
      e1 = 0.0;
    }
    eke = eke * hh; ec = ec * qqrd2e; ep = ep * hh; e1 = e1 * hh;
    erf = erf + e1;
    e = e + eke + ep + ec;
    pv = pv + vec4<f32>(eke, ep, ec, 0.0);
  }
  let nc = P.nc;
  let c = ints[P.cellOfBase + i];
  let cx = i32(c % nc.x);
  let cy = i32((c / nc.x) % nc.y);
  let cz = i32(c / (nc.x * nc.y));
  let lo = vec3<i32>(select(-1, 0, nc.x < 3u), select(-1, 0, nc.y < 3u), select(-1, 0, nc.z < 3u));
  let hi = vec3<i32>(select(1, i32(nc.x) - 1, nc.x < 3u), select(1, i32(nc.y) - 1, nc.y < 3u), select(1, i32(nc.z) - 1, nc.z < 3u));
  for (var dzc = lo.z; dzc <= hi.z; dzc = dzc + 1) {
    for (var dyc = lo.y; dyc <= hi.y; dyc = dyc + 1) {
      for (var dxc = lo.x; dxc <= hi.x; dxc = dxc + 1) {
        let c2 = (wrapCell(cz + dzc, nc.z) * nc.y + wrapCell(cy + dyc, nc.y)) * nc.x + wrapCell(cx + dxc, nc.x);
        let jEnd = ints[c2 + 1u];
        for (var j = ints[c2]; j < jEnd; j = j + 1u) {
          if (j == i) { continue; }
          let pj = pos[j];
          let tj = u32(pj.w);
          let dx = -minImage(pi.xyz, pj.xyz);
          let rsq = dot(dx, dx);
          let cs = coef[ti * nt + tj];
          if (!(rsq < cs)) { continue; }
          let xj = ext[j];
          let qj = xj.x; let sj = xj.y; let ej = xj.z;
          let absJ = abs(sj);
          let rc = sqrt(rsq);
          let taper = sqrt(cs);
          let dist = rc / taper;
          let spline = splineOf(dist);
          let dspline = dsplineOf(dist) / taper;
          var ecoul = 0.0; var epauli = 0.0; var fpair = 0.0; var sF = 0.0;
          var e1r = 0.0; var e2r = 0.0; var sE1 = 0.0; var sE2 = 0.0;
          var ecpE = 0.0; var ecpF = 0.0;
          var erfI = 0.0;
          if (si == 0.0 && sj == 0.0) {
            let r = nucNuc(qi * qj, rc); ecoul = ecoul + r.x; fpair = fpair + r.y;
          } else if (si == 2.0 && sj == 0.0) {
            let r = nucNuc(qi * qj, rc); let nn = nucElec(qj, rc, ei);
            ecoul = ecoul + r.x + 2.0 * nn.x; fpair = fpair + r.y + 2.0 * nn.y;
          } else if (si == 0.0 && sj == 2.0) {
            let r = nucNuc(qi * qj, rc); let nn = nucElec(qi, rc, ej);
            ecoul = ecoul + r.x + 2.0 * nn.x; fpair = fpair + r.y + 2.0 * nn.y;
          } else if (si == 0.0 && absJ == 1.0) {
            let nn = nucElec(qi, rc, ej); ecoul = ecoul + nn.x; fpair = fpair + nn.y;
          } else if (absI == 1.0 && sj == 0.0) {
            let nn = nucElec(qj, rc, ei); ecoul = ecoul + nn.x; fpair = fpair + nn.y; e1r = e1r + nn.z;
            erfI = spline * qqrd2e * e1r;
          } else if (absI == 1.0 && absJ == 1.0) {
            let ee = elecElec(rc, ei, ej); ecoul = ecoul + ee.x; fpair = fpair + ee.y; e1r = e1r + ee.z;
            let pa = pauliEE(si == sj, rc, ei, ej); epauli = epauli + pa.x; sF = sF - pa.y; sE1 = sE1 - pa.z;
            epauli = epauli * hh; sF = sF * hh;
            erfI = spline * (qqrd2e * e1r + hh * sE1);
          } else if (si == 2.0 && absJ == 1.0) {
            let nn = nucElec(qi, rc, ej); let ee = elecElec(rc, ei, ej);
            ecoul = ecoul + nn.x + 2.0 * ee.x; fpair = fpair + nn.y + 2.0 * ee.y;
            let p0 = pauliEE(false, rc, ei, ej); let p1 = pauliEE(true, rc, ei, ej);
            epauli = epauli + p0.x + p1.x; sF = sF - p0.y - p1.y;
            epauli = epauli * hh; sF = sF * hh;
          } else if (absI == 1.0 && sj == 2.0) {
            let nn = nucElec(qj, rc, ei); let ee = elecElec(rc, ej, ei);
            ecoul = ecoul + nn.x + 2.0 * ee.x; fpair = fpair + nn.y + 2.0 * ee.y; e2r = e2r + nn.z + 2.0 * ee.w;
            let p0 = pauliEE(false, rc, ej, ei); let p1 = pauliEE(true, rc, ej, ei);
            epauli = epauli + p0.x + p1.x; sF = sF - p0.y - p1.y; sE2 = sE2 - p0.w - p1.w;
            epauli = epauli * hh; sF = sF * hh;
            erfI = spline * (qqrd2e * e2r + hh * sE2);
          } else if (si == 2.0 && sj == 2.0) {
            let r = nucNuc(qi * qj, rc); let n1 = nucElec(qi, rc, ej); let n2 = nucElec(qj, rc, ei); let ee = elecElec(rc, ei, ej);
            ecoul = ecoul + r.x + 2.0 * n1.x + 2.0 * n2.x + 4.0 * ee.x; fpair = fpair + r.y + 2.0 * n1.y + 2.0 * n2.y + 4.0 * ee.y;
            let p0 = pauliEE(false, rc, ei, ej); let p1 = pauliEE(true, rc, ei, ej);
            epauli = 2.0 * (p0.x + p1.x) * hh; sF = -2.0 * (p0.y + p1.y) * hh;
          } else if (si == 3.0 && sj == 0.0) {
            let r = coreNuc(qi * qj, rc, ei); ecoul = ecoul + r.x; fpair = fpair + r.y;
          } else if (si == 0.0 && sj == 3.0) {
            let r = coreNuc(qi * qj, rc, ej); ecoul = ecoul + r.x; fpair = fpair + r.y;
          } else if (si == 3.0 && (absJ == 1.0 || sj == 2.0)) {
            let o = ecpBase + ${ECP_VALUES}u * ti;
            if (absJ == 1.0) {
              let nn = coreElec(qi, rc, ei, ej); ecoul = ecoul + nn.x; fpair = fpair + nn.y;
              let pa = pauliCore(o, rc, ej); ecpE = ecpE + pa.x; ecpF = ecpF + pa.y;
            } else {
              let r = coreNuc(qi * qj, rc, ej); let nn = coreElec(qi, rc, ei, ej); let pa = pauliCore(o, rc, ej);
              ecoul = ecoul + r.x + 2.0 * nn.x; fpair = fpair + r.y + 2.0 * nn.y;
              ecpE = ecpE + 2.0 * pa.x; ecpF = ecpF + 2.0 * pa.y;
            }
            ecpE = ecpE * h2e; ecpF = ecpF * h2e;
          } else if ((absI == 1.0 || si == 2.0) && sj == 3.0) {
            let o = ecpBase + ${ECP_VALUES}u * tj;
            var ecpE1 = 0.0;
            if (absI == 1.0) {
              let nn = coreElec(qj, rc, ej, ei); ecoul = ecoul + nn.x; fpair = fpair + nn.y; e1r = e1r + nn.z;
              let pa = pauliCore(o, rc, ei); ecpE = ecpE + pa.x; ecpF = ecpF + pa.y; ecpE1 = ecpE1 + pa.z;
            } else {
              let r = coreNuc(qi * qj, rc, ei); let nn = coreElec(qj, rc, ej, ei); let pa = pauliCore(o, rc, ei);
              ecoul = ecoul + r.x + 2.0 * nn.x; fpair = fpair + r.y + 2.0 * nn.y; e1r = e1r + 2.0 * nn.z;
              ecpE = ecpE + 2.0 * pa.x; ecpF = ecpF + 2.0 * pa.y; ecpE1 = ecpE1 + 2.0 * pa.z;
            }
            ecpE = ecpE * h2e; ecpF = ecpF * h2e;
            erfI = spline * (qqrd2e * e1r + h2e * ecpE1);
          } else if (si == 3.0 && sj == 3.0) {
            let r = coreCore(qi * qj, rc, ei, ej); ecoul = ecoul + r.x; fpair = fpair + r.y;
          }
          ecoul = ecoul * qqrd2e;
          fpair = fpair * qqrd2e;
          epauli = epauli + ecpE;
          let energy0 = ecoul + epauli;
          fpair = fpair + sF + ecpF;
          fpair = fpair * spline - energy0 * dspline;
          var g = vec3<f32>(0.0);
          if (rc > 0.000001) {
            g = (fpair / rc) * dx;
          } else {
            let d2 = dx * dx;
            if (dx.x != 0.0) { g.x = fpair / sqrt(1.0 + (d2.y + d2.z) / d2.x); }
            if (dx.y != 0.0) { g.y = fpair / sqrt(1.0 + (d2.x + d2.z) / d2.y); }
            if (dx.z != 0.0) { g.z = fpair / sqrt(1.0 + (d2.x + d2.y) / d2.z); }
          }
          f = f + g;
          erf = erf + erfI;
          e = e + 0.5 * spline * energy0;
          pv = pv + vec4<f32>(0.0, 0.5 * spline * epauli, 0.5 * spline * ecoul, 0.0);
          vd = vd + 0.5 * dx * g;
          vo = vo + 0.5 * vec3<f32>(dx.x * g.y, dx.x * g.z, dx.y * g.z);
        }
      }
    }
  }
  // limit/eradius (u1: any periodic dimension, the edge lengths)
  if (absI == 1.0 && (flags & ${LIMIT_ERADIUS}u) != 0u && P.u1.x != 0.0) {
    let half = 0.5 * min(P.u1.y, min(P.u1.z, P.u1.w));
    if (ei > half) {
      let dr = ei - half;
      let er = 0.5 * hh * dr * dr;
      pv.w = pv.w + er;
      erf = erf - hh * dr;
      e = e + er;
    }
  }
  let o = ${ATOM_KERNEL_OUT}u * i;
  outv[o] = f.x; outv[o + 1u] = f.y; outv[o + 2u] = f.z; outv[o + 3u] = erf; outv[o + 4u] = e;
  outv[o + 5u] = pv.x; outv[o + 6u] = pv.y; outv[o + 7u] = pv.z; outv[o + 8u] = pv.w;
  outv[o + 9u] = vd.x; outv[o + 10u] = vd.y; outv[o + 11u] = vd.z; outv[o + 12u] = vo.x; outv[o + 13u] = vo.y; outv[o + 14u] = vo.z;
  outv[o + 15u] = 0.0;
}
`;

/**
 * The GPU kernel of pair eff/cut for the run (run/accel.ts through the gplHook eff/cut), or why it cannot run there.
 * The pair is initialized (accelerator() runs init first).
 */
export const gpuEffKernel = (pair: PairEffCut, sys: System): GpuAtomKernel | string => {
  const s = sys.state;
  if (!s.q) return 'pair eff/cut without per-atom charges';
  const g = pair.gpuParams();
  const nt = s.ntypes + 1;
  const ecp: (number[] | null)[] = g.ecp.map((v) => (v ? [...v, v[3] === 0 && v[4] === 0 ? 1 : 0] : null));
  const qqrd2e = s.units.qqr2e / sys.ff.dielectric;
  const P: EffParams = {
    nt, cutsq: Float64Array.from(pair.cutsq), ecp, qqrd2e, hh: g.hhmss2e, h2e: g.h2e,
    limitEradius: g.limitEradius, pressureEvirials: g.pressureEvirials,
  };
  if (P.cutsq.length < nt * nt) return 'pair eff/cut is not initialized';
  const coef = new Float64Array(nt * nt + ECP_VALUES * nt);
  coef.set(P.cutsq.subarray(0, nt * nt));
  for (let t = 0; t < nt; t++) { const v = ecp[t]; if (v) coef.set(v, nt * nt + ECP_VALUES * t); }
  let maxCut = 0;
  for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) maxCut = Math.max(maxCut, Math.sqrt(P.cutsq[i * nt + j]));
  const flags = (g.limitEradius ? LIMIT_ERADIUS : 0) | (g.pressureEvirials ? PRESSURE_EVIRIALS : 0);
  return {
    name: 'eff/cut',
    code: KERNEL_EFF,
    key: `eff:${Array.from(coef).join(',')}`,
    coef,
    stride: nt,
    maxCutoff: maxCut,
    scalars: (st) => {
      const lim = effLimit(P, st);
      return [qqrd2e, P.hh, P.h2e, flags, lim.on ? 1 : 0, lim.lx, lim.ly, lim.lz];
    },
    extra: (st, out) => {
      const spin = st.custom.get(ELECTRON_SPIN)!.data, erad = st.custom.get(ELECTRON_RADIUS)!.data, q = st.q!;
      for (let i = 0; i < st.n; i++) { out[4 * i] = q[i]; out[4 * i + 1] = spin[i]; out[4 * i + 2] = erad[i]; out[4 * i + 3] = 0; }
    },
    finish: (st, out) => {
      const n = st.n;
      const spin = st.custom.get(ELECTRON_SPIN)!.data, erad = st.custom.get(ELECTRON_RADIUS)!.data;
      const erforce = st.custom.get(ELECTRON_RFORCE)!.data;
      const pv = pair.extract('pvector') as Float64Array;
      pv.fill(0);
      const vt = [0, 0, 0, 0, 0, 0];
      let e = 0;
      for (let i = 0; i < n; i++) {
        const o = ATOM_KERNEL_OUT * i;
        st.f[3 * i] = out[o]; st.f[3 * i + 1] = out[o + 1]; st.f[3 * i + 2] = out[o + 2];
        erforce[i] = out[o + 3];
        e += out[o + 4];
        for (let c = 0; c < 4; c++) pv[c] += out[o + 5 + c];
        for (let c = 0; c < 6; c++) vt[c] += out[o + 9 + c];
      }
      // pressure/evirials: the radial virial of the electrons (pair_eff_cut.cpp virial_eff_compute), once per atom
      if (P.pressureEvirials) {
        for (let i = 0; i < n; i++) {
          if (!spin[i]) continue;
          const ev = erforce[i] * erad[i] / 3;
          vt[0] += ev; vt[1] += ev; vt[2] += ev;
        }
      }
      return { pe: e, virial: vt[0] + vt[1] + vt[2], virialTensor: vt, evdwl: 0, ecoul: e };
    },
    fp64: (st) => effFp64(P, st),
  };
};
