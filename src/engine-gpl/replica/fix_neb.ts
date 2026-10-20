/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/REPLICA/fix_neb.cpp
 * (FixNEB::min_post_force and calculate_ideal_positions) and the orthogonal and triclinic branches of
 * Domain::minimum_image in src/domain.cpp at commit b332f28. Authors named in the source: Rochus Schmid (Ruhr-Uni
 * Bochum) and Aidan Thompson (Sandia), free end and perpendicular spring by Emile Maras (CEA); LAMMPS is Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The nudged forces of every replica of a NEB, computed together: native runs one FixNEB per partition and exchanges the
 * neighbouring replicas' coordinates, forces and energies over MPI; here every replica's NEB atoms are given in the same
 * (atom-ID) order, so replica r's xprev and xnext are the arrays of replicas r - 1 and r + 1.
 */
import { libm } from '../../engine/libm';
import { StyleError } from '../../engine/force/types';

/** One replica's NEB atoms: coordinates and forces (3 per atom, the same atoms in the same order in every replica). */
export interface NebReplicaInput {
  X: Float64Array;
  /** the forces before the nudging; replaced by the nudged forces */
  F: Float64Array;
  /** potential energy (compute pe, not normalized) */
  E: number;
}

/** The fix neb keywords (FixNEB constructor). */
export interface NebSettings {
  kspring: number;
  kspringPerp: number;
  mode: 'neigh' | 'ideal' | 'equal';
  freeEndIni: boolean;
  freeEndFinal: boolean;
  freeEndFinalWithRespToEIni: boolean;
  finalAndInterWithRespToEIni: boolean;
  kspringIni: number;
  kspringFinal: number;
}

/** What the fixes keep between calls: the end energies recorded on timestep 0 (EIniIni, EFinalIni). */
export interface NebEndEnergies {
  eIniIni: number;
  eFinalIni: number;
}

/** The box as Domain::minimum_image sees it. */
export interface NebBox {
  prd: [number, number, number];
  periodic: [boolean, boolean, boolean];
  triclinic: boolean;
  /** xy, xz, yz */
  tilt: [number, number, number];
}

/** One replica's FixNEB members after min_post_force (what neb print_status reads). */
export interface NebReplicaOutput {
  veng: number;
  plen: number;
  nlen: number;
  gradlen: number;
  dotpath: number;
  dottangrad: number;
  dotgrad: number;
}

const MAXIMGCOUNT = 16;

/** Domain::minimum_image: the error text is native's ("Atoms have moved too far apart ({}) for minimum image"). */
const minimumImage = (box: NebBox, d: Float64Array): void => {
  const [xprd, yprd, zprd] = box.prd;
  const [xperiodic, yperiodic, zperiodic] = box.periodic;
  const far = (v: number): never => { throw new StyleError(`Atoms have moved too far apart (${v}) for minimum image`); };
  if (!box.triclinic) {
    if (xperiodic) {
      if (Math.abs(d[0]) > MAXIMGCOUNT * xprd) far(d[0]);
      while (Math.abs(d[0]) > xprd / 2) d[0] += d[0] < 0 ? xprd : -xprd;
    }
    if (yperiodic) {
      if (Math.abs(d[1]) > MAXIMGCOUNT * yprd) far(d[1]);
      while (Math.abs(d[1]) > yprd / 2) d[1] += d[1] < 0 ? yprd : -yprd;
    }
    if (zperiodic) {
      if (Math.abs(d[2]) > MAXIMGCOUNT * zprd) far(d[2]);
      while (Math.abs(d[2]) > zprd / 2) d[2] += d[2] < 0 ? zprd : -zprd;
    }
    return;
  }
  const [xy, xz, yz] = box.tilt;
  if (zperiodic) {
    if (Math.abs(d[2]) > MAXIMGCOUNT * zprd) far(d[2]);
    while (Math.abs(d[2]) > zprd / 2) {
      if (d[2] < 0) { d[2] += zprd; d[1] += yz; d[0] += xz; } else { d[2] -= zprd; d[1] -= yz; d[0] -= xz; }
    }
  }
  if (yperiodic) {
    if (Math.abs(d[1]) > MAXIMGCOUNT * yprd) far(d[1]);
    while (Math.abs(d[1]) > yprd / 2) {
      if (d[1] < 0) { d[1] += yprd; d[0] += xy; } else { d[1] -= yprd; d[0] -= xy; }
    }
  }
  if (xperiodic) {
    if (Math.abs(d[0]) > MAXIMGCOUNT * xprd) far(d[0]);
    while (Math.abs(d[0]) > xprd / 2) d[0] += d[0] < 0 ? xprd : -xprd;
  }
};

/**
 * FixNEB::min_post_force of every replica (then calculate_ideal_positions): replaces each replica's F with its nudged
 * force. rclimber is the climbing replica (-1 before the climbing stage); ntimestep is update->ntimestep (the end
 * energies are recorded when it is 0, into ends).
 */
export const nebForces = (reps: NebReplicaInput[], p: NebSettings, ends: NebEndEnergies, rclimber: number, ntimestep: number, box: NebBox): NebReplicaOutput[] => {
  const nreplica = reps.length;
  const out: NebReplicaOutput[] = [];
  const tangents: Float64Array[] = [];
  const springFs: Float64Array[] = [];
  const dots: number[] = [];
  const tlens: number[] = [];
  const dp = new Float64Array(3), dn = new Float64Array(3);

  // the first part of min_post_force, up to the end forces: every replica reads its neighbours' unmodified forces
  // (inter_replica_comm runs before any replica changes its own), so all replicas are measured before any is nudged
  const fnexts = reps.map((rep) => Float64Array.from(rep.F));
  for (let ireplica = 0; ireplica < nreplica; ireplica++) {
    const { X: x, F: f, E: veng } = reps[ireplica];
    const vprev = ireplica > 0 ? reps[ireplica - 1].E : veng;
    const vnext = ireplica < nreplica - 1 ? reps[ireplica + 1].E : veng;
    if (p.freeEndFinal && ireplica === nreplica - 1 && ntimestep === 0) ends.eFinalIni = veng;
    if (p.freeEndIni && ireplica === 0 && ntimestep === 0) ends.eIniIni = veng;
    const xprev = ireplica > 0 ? reps[ireplica - 1].X : x;
    const xnext = ireplica < nreplica - 1 ? reps[ireplica + 1].X : x;
    const fnext = ireplica < nreplica - 1 ? fnexts[ireplica + 1] : f;
    const m = x.length / 3;
    const tangent = new Float64Array(3 * m);
    const springF = new Float64Array(3 * m);
    let plen = 0, nlen = 0, tlen = 0, gradnextlen = 0;
    let dotgrad = 0, gradlen = 0, dotpath = 0, dottangrad = 0, dot = 0;
    if (ireplica === nreplica - 1) {
      for (let i = 0; i < m; i++) {
        const k = 3 * i;
        dp[0] = x[k] - xprev[k]; dp[1] = x[k + 1] - xprev[k + 1]; dp[2] = x[k + 2] - xprev[k + 2];
        minimumImage(box, dp);
        plen += dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
        dottangrad += dp[0] * f[k] + dp[1] * f[k + 1] + dp[2] * f[k + 2];
        gradlen += f[k] * f[k] + f[k + 1] * f[k + 1] + f[k + 2] * f[k + 2];
        if (p.freeEndFinal || p.freeEndFinalWithRespToEIni) {
          tangent[k] = dp[0]; tangent[k + 1] = dp[1]; tangent[k + 2] = dp[2];
          tlen += tangent[k] * tangent[k] + tangent[k + 1] * tangent[k + 1] + tangent[k + 2] * tangent[k + 2];
          dot += f[k] * tangent[k] + f[k + 1] * tangent[k + 1] + f[k + 2] * tangent[k + 2];
        }
      }
    } else if (ireplica === 0) {
      for (let i = 0; i < m; i++) {
        const k = 3 * i;
        dn[0] = xnext[k] - x[k]; dn[1] = xnext[k + 1] - x[k + 1]; dn[2] = xnext[k + 2] - x[k + 2];
        minimumImage(box, dn);
        nlen += dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2];
        gradnextlen += fnext[k] * fnext[k] + fnext[k + 1] * fnext[k + 1] + fnext[k + 2] * fnext[k + 2];
        dotgrad += f[k] * fnext[k] + f[k + 1] * fnext[k + 1] + f[k + 2] * fnext[k + 2];
        dottangrad += dn[0] * f[k] + dn[1] * f[k + 1] + dn[2] * f[k + 2];
        gradlen += f[k] * f[k] + f[k + 1] * f[k + 1] + f[k + 2] * f[k + 2];
        if (p.freeEndIni) {
          tangent[k] = dn[0]; tangent[k + 1] = dn[1]; tangent[k + 2] = dn[2];
          tlen += tangent[k] * tangent[k] + tangent[k + 1] * tangent[k + 1] + tangent[k + 2] * tangent[k + 2];
          dot += f[k] * tangent[k] + f[k + 1] * tangent[k + 1] + f[k + 2] * tangent[k + 2];
        }
      }
    } else {
      // not the first or last replica
      const vmax = Math.max(Math.abs(vnext - veng), Math.abs(vprev - veng));
      const vmin = Math.min(Math.abs(vnext - veng), Math.abs(vprev - veng));
      for (let i = 0; i < m; i++) {
        const k = 3 * i;
        dp[0] = x[k] - xprev[k]; dp[1] = x[k + 1] - xprev[k + 1]; dp[2] = x[k + 2] - xprev[k + 2];
        minimumImage(box, dp);
        plen += dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
        dn[0] = xnext[k] - x[k]; dn[1] = xnext[k + 1] - x[k + 1]; dn[2] = xnext[k + 2] - x[k + 2];
        minimumImage(box, dn);
        for (let d = 0; d < 3; d++) {
          if (vnext > veng && veng > vprev) tangent[k + d] = dn[d];
          else if (vnext < veng && veng < vprev) tangent[k + d] = dp[d];
          else if (vnext > vprev) tangent[k + d] = vmax * dn[d] + vmin * dp[d];
          else if (vnext < vprev) tangent[k + d] = vmin * dn[d] + vmax * dp[d];
          // vnext == vprev, e.g. for potentials that do not compute an energy
          else tangent[k + d] = dn[d] + dp[d];
        }
        nlen += dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2];
        tlen += tangent[k] * tangent[k] + tangent[k + 1] * tangent[k + 1] + tangent[k + 2] * tangent[k + 2];
        gradlen += f[k] * f[k] + f[k + 1] * f[k + 1] + f[k + 2] * f[k + 2];
        dotpath += dp[0] * dn[0] + dp[1] * dn[1] + dp[2] * dn[2];
        dottangrad += tangent[k] * f[k] + tangent[k + 1] * f[k + 1] + tangent[k + 2] * f[k + 2];
        gradnextlen += fnext[k] * fnext[k] + fnext[k + 1] * fnext[k + 1] + fnext[k + 2] * fnext[k + 2];
        dotgrad += f[k] * fnext[k] + f[k + 1] * fnext[k + 1] + f[k + 2] * fnext[k + 2];
        springF[k] = p.kspringPerp * (dn[0] - dp[0]);
        springF[k + 1] = p.kspringPerp * (dn[1] - dp[1]);
        springF[k + 2] = p.kspringPerp * (dn[2] - dp[2]);
      }
    }
    nlen = Math.sqrt(nlen);
    plen = Math.sqrt(plen);
    tlen = Math.sqrt(tlen);
    gradlen = Math.sqrt(gradlen);
    gradnextlen = Math.sqrt(gradnextlen);

    // normalize tangent vector
    if (tlen > 0) {
      const tleninv = 1 / tlen;
      for (let k = 0; k < 3 * m; k++) tangent[k] *= tleninv;
    }
    if (ireplica > 0 && ireplica < nreplica - 1) dottangrad = dottangrad / (tlen * gradlen);
    if (ireplica === 0) dottangrad = dottangrad / (nlen * gradlen);
    if (ireplica === nreplica - 1) dottangrad = dottangrad / (plen * gradlen);
    if (ireplica < nreplica - 1) dotgrad = dotgrad / (gradlen * gradnextlen);
    out.push({ veng, plen, nlen, gradlen, dotpath, dottangrad, dotgrad });
    tangents.push(tangent);
    springFs.push(springF);
    dots.push(dot);
    tlens.push(tlen);
  }

  const vIni = reps[0].E;
  const nlenall = out.map((o) => o.nlen);
  const vengall = out.map((o) => o.veng);
  for (let ireplica = 0; ireplica < nreplica; ireplica++) {
    const f = reps[ireplica].F;
    const tangent = tangents[ireplica];
    const springF = springFs[ireplica];
    const o = out[ireplica];
    const veng = o.veng;
    const tlen = tlens[ireplica];
    const m = f.length / 3;
    let dot = dots[ireplica];
    let prefactor = 0;
    const addAlongTangent = (): void => { for (let k = 0; k < 3 * m; k++) f[k] += prefactor * tangent[k]; };

    if (p.freeEndIni && ireplica === 0 && tlen > 0) {
      dot = dot / tlen;
      if (dot < 0) prefactor = -dot - p.kspringIni * (veng - ends.eIniIni);
      else prefactor = -dot + p.kspringIni * (veng - ends.eIniIni);
      addAlongTangent();
    }
    if (p.freeEndFinal && ireplica === nreplica - 1 && tlen > 0) {
      dot = dot / tlen;
      if (veng < ends.eFinalIni) {
        if (dot < 0) prefactor = -dot - p.kspringFinal * (veng - ends.eFinalIni);
        else prefactor = -dot + p.kspringFinal * (veng - ends.eFinalIni);
      }
      addAlongTangent();
    }
    if (p.freeEndFinalWithRespToEIni && ireplica === nreplica - 1 && tlen > 0) {
      dot = dot / tlen;
      if (veng < vIni) {
        if (dot < 0) prefactor = -dot - p.kspringFinal * (veng - vIni);
        else prefactor = -dot + p.kspringFinal * (veng - vIni);
      }
      addAlongTangent();
    }

    // calculate_ideal_positions (parallel ideal and equal)
    let actualPos = 0, idealPos = 0;
    if (p.mode === 'ideal' || p.mode === 'equal') {
      const lens = nlenall.slice();
      if (p.mode === 'equal' && rclimber > 0) {
        for (let i = 0; i < nreplica - 1; i++) lens[i] = Math.abs(vengall[i + 1] - vengall[i]);
        lens[nreplica - 1] = 0;
      }
      let lentot = 0;
      for (let i = 0; i < nreplica; i++) lentot += lens[i];
      for (let i = 0; i < ireplica; i++) actualPos += lens[i];
      const meanDist = lentot / (nreplica - 1);
      let lenuntilClimber = 0;
      if (rclimber > 0) {
        for (let i = 0; i < rclimber; i++) lenuntilClimber += lens[i];
        const meanDistBeforeClimber = lenuntilClimber / rclimber;
        const meanDistAfterClimber = (lentot - lenuntilClimber) / (nreplica - rclimber - 1);
        if (ireplica < rclimber) idealPos = ireplica * meanDistBeforeClimber;
        else idealPos = lenuntilClimber + (ireplica - rclimber) * meanDistAfterClimber;
      } else idealPos = ireplica * meanDist;
      idealPos /= meanDist;
      actualPos /= meanDist;
    }

    if (ireplica === 0 || ireplica === nreplica - 1) continue;

    o.dotpath = o.dotpath / (o.plen * o.nlen);
    let AngularContr = 0.5 * (1 + libm.cosAlone(Math.PI * o.dotpath));
    let dotSpringTangent = 0;
    for (let i = 0; i < m; i++) {
      const k = 3 * i;
      dot += f[k] * tangent[k] + f[k + 1] * tangent[k + 1] + f[k + 2] * tangent[k + 2];
      dotSpringTangent += springF[k] * tangent[k] + springF[k + 1] * tangent[k + 1] + springF[k + 2] * tangent[k + 2];
    }
    if (ireplica === rclimber) prefactor = -2 * dot;
    else {
      if (p.mode === 'ideal' || p.mode === 'equal') prefactor = -dot - p.kspring * (actualPos - idealPos) / 2;
      else prefactor = -dot + p.kspring * (o.nlen - o.plen);
      if (p.finalAndInterWithRespToEIni && veng < vIni) {
        f.fill(0);
        prefactor = p.kspring * (o.nlen - o.plen);
        AngularContr = 0;
      }
    }
    for (let k = 0; k < 3 * m; k++) f[k] += prefactor * tangent[k] + AngularContr * (springF[k] - dotSpringTangent * tangent[k]);
  }
  return out;
};
