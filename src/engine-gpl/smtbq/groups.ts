/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SMTBQ/pair_smtbq.cpp (PairSMTBQ::groupQEqAll_QEq, groupSurface_QEq, groupBulkFromSlab_QEq,
 * groupQEqAllParallel_QEq, the team assembly of the last part of that routine) and src/SMTBQ/pair_smtbq.h, at
 * commit b332f28. Authors credited in the C++ file headers: N. Salles, O. Politano, E. Amzallag, R. Tetot
 * (SMTBQ) and E. Maras (Aalto University). Copyright (2015) Universite de Bourgogne and Sandia Corporation
 * (LAMMPS). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Serial (one process) path only: the C++ routine assembles the groups of several processes with MPI_Allgather,
 * MPI_Allreduce and forward_int ghost copies; with one process those reduce to the local tables, which is what
 * is kept here. The ghost copy of the group flags (forward_int) keeps a ghost's old value when its owner's flag
 * is zero, as the C++ routine does (only values with |v| > 0.1 are copied).
 */
import { StyleError } from '../../engine/force/types';
import { NEIGHMASK, type NeighList } from '../../engine/neighbor';

/** The geometry and topology the group routines read. */
export interface GroupContext {
  nlocal: number;
  nall: number;
  /** Owned atoms with lists (ilist is 0 .. inum-1). */
  inum: number;
  /** Element index of every owned and ghost atom (map[type]). */
  el: Int32Array;
  /** Owned and ghost positions (3 per atom). */
  x: Float64Array;
  /** Owner of each ghost (owned atoms own themselves). */
  owner: Int32Array;
  /** The full neighbor list. */
  list: NeighList;
  /** intype[itype * ntypes + jtype]: interaction number (0 = none). */
  intype: Int32Array;
  ntypes: number;
  /** Second-moment cut-off dc2 per interaction number (index 0 = none). */
  dc2: Float64Array;
  cutmax: number;
  zlim1QEq: number;
  zlim2QEq: number;
}

/** Number of group rows of the C++ tab_gp array. */
const MAXGROUPS = 10;

/** The QEq group state: flag_QEq per atom and nteam (the number of charge-equilibrated groups). */
export class QEqGroups {
  /** flag_QEq of every owned and ghost atom: 0 = not charge-equilibrated; else the group number (1 .. nteam). */
  flag = new Int32Array(0);
  /** Number of groups (nteam). Kept between calls: the parallel grouping returns without changing it when no oxide is present. */
  nteam = 0;

  private ensure(nall: number): void {
    if (this.flag.length < nall) this.flag = new Int32Array(nall);
  }

  /** Sizes the flag array for nall atoms (before any charge equilibration reads it). */
  reserve(nall: number): void {
    this.ensure(nall);
  }

  /** groupQEqAll_QEq: every owned atom is in group 1. */
  groupQEqAll(c: GroupContext): void {
    this.ensure(c.nall);
    this.nteam = 1;
    for (let ii = 0; ii < c.inum; ii++) this.flag[ii] = 1;
  }

  /** groupSurface_QEq: owned atoms above zlim1 (the z limit). */
  groupSurface(c: GroupContext): void {
    this.ensure(c.nall);
    for (let ii = 0; ii < c.inum; ii++) {
      const z = c.x[3 * ii + 2];
      this.flag[ii] = z > c.zlim1QEq ? 1 : 0;
      this.nteam = 1;
    }
  }

  /** groupBulkFromSlab_QEq: owned atoms between zlim1 and zlim2. */
  groupBulkFromSlab(c: GroupContext): void {
    this.ensure(c.nall);
    for (let ii = 0; ii < c.inum; ii++) {
      const z = c.x[3 * ii + 2];
      this.flag[ii] = z > c.zlim1QEq && z < c.zlim2QEq ? 1 : 0;
      this.nteam = 1;
    }
  }

  /** groupQEqAllParallel_QEq (serial): the groups of the cations and the oxygens they touch. */
  groupQEqAllParallel(c: GroupContext): void {
    const { nlocal, nall, inum, el, x, list, intype, ntypes, cutmax, dc2 } = c;
    this.ensure(nall);
    const flag = this.flag;
    const flagGp = new Int32Array(nall);
    const nelt = new Int32Array(MAXGROUPS + 1);
    const tabGp: Int32Array[] = [];
    for (let g = 0; g < MAXGROUPS; g++) tabGp.push(new Int32Array(nall));
    const { firstneigh, numneigh, neighbors } = list;
    const nbr = (i: number, jj: number): number => neighbors[firstneigh[i] + jj] & NEIGHMASK;

    for (let i = 0; i < nall; i++) flag[i] = 0;

    // identify the atoms entering the QEq scheme
    let QEq = 0;
    for (let i = 0; i < inum; i++) {
      const itype = el[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      for (let jj = 0; jj < numneigh[i]; jj++) {
        const j = nbr(i, jj);
        const jtype = el[j];
        if (jtype === itype) continue;
        const m = intype[itype * ntypes + jtype];
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (Math.sqrt(rsq) <= dc2[m]) {
          flag[i] = 1;
          flag[j] = 1;
        }
      }
      if (flag[i] === 1) QEq = 1;
    }

    // no oxide: the groups (and nteam) are left as they are
    if (QEq === 0) return;

    // the oxygens entering the QEq scheme, grouped by the cations' neighbors
    flagGp.fill(0);
    let ngp = 0;
    let igp = 0;
    nelt[0] = 0;
    for (let ii = 0; ii < inum; ii++) {
      const i = ii;
      const itype = el[i];
      if (itype !== 0 || flag[i] === 0) continue;
      if (ngp !== 0 && flagGp[i] === ngp) continue;

      // grouping initialisation
      if (flagGp[i] === 0) {
        ngp += 1;
        if (ngp >= MAXGROUPS) throw new StyleError('pair_style smtbq: too many QEq groups (the native limit is 9)');
        nelt[ngp] = 0;
        tabGp[ngp][nelt[ngp]] = i;
        flagGp[i] = ngp;
        nelt[ngp] += 1;
      }

      // loop on the group members
      for (let kk = 0; kk < nelt[ngp]; kk++) {
        const k = tabGp[ngp][kk];
        const ktype = el[k];
        if (k >= nlocal) continue;
        const xk = x[3 * k], yk = x[3 * k + 1], zk = x[3 * k + 2];
        // loop on all owned and ghost atoms (the C++ loop runs over nall, not over the list)
        for (let j = 0; j < nall; j++) {
          const jtype = el[j];
          if (jtype === ktype) continue;
          if (jtype === 0 && flag[j] === 0) continue;
          if (flagGp[j] === ngp) continue;
          const dx = x[3 * j] - xk, dy = x[3 * j + 1] - yk, dz = x[3 * j + 2] - zk;
          const rsq = dx * dx + dy * dy + dz * dz;
          if (Math.sqrt(rsq) <= cutmax) {
            flag[j] = 1;
            // meeting of two groups in the same process
            if (flagGp[j] !== 0 && flagGp[j] !== ngp && nelt[flagGp[j]] !== 0) {
              igp = flagGp[j];
              const z = Math.min(igp, ngp);
              if (z === igp) {
                igp = z;
              } else if (z === ngp) {
                ngp = igp;
                igp = z;
                flagGp[j] = ngp;
              }
              for (let zz = 0; zz < nelt[ngp]; zz++) {
                const zm = tabGp[ngp][zz];
                tabGp[igp][nelt[igp]] = zm;
                nelt[igp] += 1;
                flagGp[zm] = igp;
                tabGp[ngp][zz] = 0;
              }
              nelt[ngp] = 0;
              for (let z2 = nlocal; z2 < nall; z2++) {
                if (flagGp[z2] === ngp) flagGp[z2] = igp;
              }
              kk = 0;
              ngp = igp;
              break;
            }
            flagGp[j] = ngp;
            if (j < nlocal) {
              tabGp[ngp][nelt[ngp]] = j;
              nelt[ngp] += 1;
            }
          }
        }
      }
    }

    // group communication (forward_int): a ghost takes its owner's group when that is nonzero
    this.forwardFlags(flagGp, c);

    // the cations join the group of the oxygens they interact with
    igp = 0;
    for (let ii = 0; ii < inum; ii++) {
      const i = ii;
      const itype = el[i];
      if (itype === 0) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      for (let jj = 0; jj < numneigh[i]; jj++) {
        const j = nbr(i, jj);
        const jtype = el[j];
        if (jtype !== 0) continue;
        let m = 0;
        if (flagGp[j] !== 0) m = flagGp[j];
        if (m === 0) continue;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (Math.sqrt(rsq) <= cutmax) {
          flag[i] = 1;
          igp = flagGp[j];
          if (flagGp[i] === 0) flagGp[i] = igp;
          if (flagGp[i] !== igp && igp !== 0) {
            igp = Math.min(flagGp[i], flagGp[j]);
            const kgp = igp === flagGp[i] ? flagGp[j] : flagGp[i];
            for (let k = 0; k < nelt[kgp]; k++) {
              const z = tabGp[kgp][k];
              tabGp[igp][nelt[igp]] = z;
              nelt[igp] += 1;
              flagGp[z] = igp;
              tabGp[kgp][k] = 0;
            }
            nelt[kgp] = 0;
            for (let k = 0; k < nall; k++) {
              if (flagGp[k] === kgp) flagGp[k] = igp;
            }
          }
          m = 0;
          for (let k = 0; k < nelt[igp]; k++) {
            if (tabGp[igp][k] === i) m = 1;
          }
          if (i >= nlocal || m === 1) continue;
          tabGp[igp][nelt[igp]] = i;
          nelt[igp] += 1;
          break;
        }
      }
    }
    this.forwardFlags(flagGp, c);

    // the groups present (group numbers 1 .. 9) and the teams: one team per group in serial
    const gptmp = new Int32Array(MAXGROUPS);
    for (let i = 0; i < nall; i++) {
      const mm = flagGp[i];
      if (mm === 0) continue;
      gptmp[mm] = 1;
    }
    const teams: number[][] = [[]];
    let nt = 0;
    for (let g = 0; g < MAXGROUPS; g++) {
      if (gptmp[g] === 0) continue;
      if (nt === 0) {
        nt += 1;
        teams[nt] = [g];
      } else {
        let found = false;
        for (let t = 1; t <= nt; t++) {
          if (teams[t].includes(g)) found = true;
        }
        if (found) continue;
        nt += 1;
        teams[nt] = [g];
      }
    }

    // the QEq flag of each atom becomes the number of its team
    for (let ii = 0; ii < inum; ii++) {
      const i = ii;
      if (flag[i] === 0) continue;
      const gp = flagGp[i];
      for (let j = 1; j <= nt; j++) {
        if (teams[j].includes(gp)) {
          flag[i] = j;
          break;
        }
      }
    }
    this.nteam = nt;
  }

  /** forward_int of the group flags: a ghost takes its owner's value when that is nonzero. */
  private forwardFlags(flagGp: Int32Array, c: GroupContext): void {
    for (let k = c.nlocal; k < c.nall; k++) {
      const v = flagGp[c.owner[k]];
      if (Math.abs(v) > 0.1) flagGp[k] = v;
    }
  }
}
