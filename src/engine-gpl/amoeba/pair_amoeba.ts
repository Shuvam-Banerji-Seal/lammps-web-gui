/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/pair_amoeba.cpp (settings, coeff, mix(), choose(), compute(), kewald(), ewaldcof(), the pvector
 * energies) and src/AMOEBA/amoeba_utils.cpp (add_onefive_neighbors, find_hydrogen_neighbors, find_multipole_neighbors)
 * at commit b332f28. AMOEBA package authors: the LAMMPS developers (Sandia Corporation, GNU General Public License). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style amoeba: the buffered 14-7 van der Waals term (hal), the permanent multipole term (mpole; direct sum and the
 * PME reciprocal sum, kspace.ts), the induced-dipole polarization (mutual solver, polarization energy, forces and torques;
 * polar.ts, with the reciprocal fields and polar_kspace of kspace.ts) on periodic boxes (ghost images, closest-image
 * frame and hydrogen maps as native's neighbor-build copies), with the Ewald real-space damping and self terms.
 * `pair_style amoeba exclude mpole/kspace polar/kspace` drops the reciprocal parts (native). Refused by name: the
 * dispersion Ewald (dewald), polar-predict, polarization DIRECT and OPT, the vdwpr overrides, the triclinic box, per-atom
 * energy and virial, and the other pair_style components (include, hal/repulse/qxfer off).
 * docs.lammps.org/pair_amoeba.html: "pair_style amoeba" takes no arguments and "pair_coeff * * file.prm file.key"
 * reads the Tinker PRM and key files.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, sbOf } from '../../engine/neighbor';
import { buildOneFive } from '../../engine/atoms';
import {
  checkPmeOrder, cmpFromRpole, defaultGridSize, factorable, influence, makePmeGrid, mutualRecipField, multipoleKspace,
  permanentRecipField, pmeFill, polarKspaceForces, type PmeFill, type PmeGrid,
} from './kspace';
import { halTerm, powint, type HalTables, type HalInput } from './hal';
import { parseAmoebaKey, parseAmoebaPrm, type AmoebaKey, type AmoebaPrm } from './params';
import { chkpole, ELECTRIC, kmpole, mapAxes, multipoleReal, rotmat, rotsite, type MpoleSites, type MpoleTopo } from './multipole';
import {
  assignPolarGroups, buildPrecondTensors, induceDipoles, polarEnergy, polarReal, precondPairs,
  type PolarAtoms, type PolarSettings, type PrecondPairs,
} from './polar';
import { chargeTransferTerm, dispersionReal, repulsionTerm, type HippoSiteData, type TermPairs } from './hippo_terms';
import { dispersionGridSize, dispersionKspace, makeDispersionGrid } from './dispersion_pme';
import { multipoleRealHippo, type HippoMpoleSites } from './hippo_mpole';
import type { HippoPolarAtoms } from './hippo_polar';
import { pow as libmPow } from '../libm/pow';
import { erfc as libmErfc } from '../libm/erf';

/** pvector order of pair amoeba (pair_amoeba.cpp): hal, repulse, disp, mpole, polar, qxfer. */
const NVECTOR = 6;
/** sqrt(pi) (pair_amoeba.cpp MY_PIS). */
const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)

/**
 * ewaldcof (amoeba_kspace.cpp): the Ewald coefficient for which erfc(a cutoff)/cutoff falls below 1e-8, found by
 * doubling and then a bisection of 60 + the doubling count steps.
 */
export const ewaldcof = (cutoff: number): number => {
  const eps = 1.0e-8;
  let ratio = eps + 1.0;
  let x = 0.5;
  let i = 0;
  while (ratio >= eps) {
    i++;
    x *= 2.0;
    const y = x * cutoff;
    ratio = libmErfc(y) / cutoff;
  }
  const k = i + 60;
  let xlo = 0.0, xhi = x;
  for (let m = 0; m < k; m++) {
    x = (xlo + xhi) / 2.0;
    const y = x * cutoff;
    ratio = libmErfc(y) / cutoff;
    if (ratio >= eps) xlo = x;
    else xhi = x;
  }
  return x;
};

export class PairAmoeba extends Pair {
  readonly name: string;
  /** pair hippo (native PairHippo is PairAmoeba with amoeba == false): no hal, HIPPO repulsion, dispersion, charge
   * transfer and the charge-penetration multipoles. */
  readonly hippo: boolean;
  needsHalf = true;
  needsFull = false;
  /** Every special pair stays in the list (the force field's own weights apply: pair_amoeba.cpp add_onefive_neighbors). */
  keepExcluded = true;
  /** The virial is tallied here (amoeba_hal.cpp vflag_global), not by the fdotr sum. */
  explicitVirial = true;
  private prm: AmoebaPrm | null = null;
  private vdwcut = 9.0;
  private vdwtaper = 8.0;
  private tables: HalTables | null = null;
  /** mpolecut of the key file (the multipole term's cutoff); the neighbor list reaches the largest of the term cutoffs. */
  private mpolecut = 9.0;
  private key: AmoebaKey | null = null;
  /** Preconditioner pairs of the induced-dipole solver, rebuilt on each neighbor build (precond_neigh). */
  private precond: PrecondPairs | null = null;
  private precondBuild = -1;
  /** pvector of the last compute (energies per term, native order). */
  private pvec = new Float64Array(NVECTOR);
  /** Reciprocal-space switches (pair_style amoeba exclude mpole/kspace polar/kspace; both on by default). */
  private mpoleKspace = true;
  private polarKspace = true;
  /** Copy choices of the current neighbor build (closest images), and the build they belong to. */
  private memo = new Map<string, number>();
  private memoBuild = -1;
  /** Ewald coefficients of the current step: aeew for the permanent multipoles, apew for the induced dipoles (0 without key ewald). */
  private aeew = 0;
  private apew = 0;

  constructor(hippo = false) {
    super();
    this.hippo = hippo;
    this.name = hippo ? 'hippo' : 'amoeba';
  }

  /**
   * pair_style amoeba [exclude mpole/kspace polar/kspace]: native takes the component switches (pair_amoeba.cpp
   * settings). Only the reciprocal-space switches are accepted here; the real-space and bonded components are not ported.
   */
  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length === 0) return;
    if (args[0] !== 'exclude') {
      throw new StyleError('pair amoeba: only "pair_style amoeba exclude mpole/kspace polar/kspace" is accepted (include and the other components are not ported)');
    }
    if (args.length === 1) throw new StyleError('pair amoeba: exclude needs a component (mpole/kspace or polar/kspace)');
    for (const c of args.slice(1)) {
      if (c === 'mpole/kspace') this.mpoleKspace = false;
      else if (c === 'polar/kspace') this.polarKspace = false;
      else throw new StyleError(`pair amoeba: exclude ${c} is not ported (only mpole/kspace and polar/kspace)`);
    }
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 4 || args[0] !== '*' || args[1] !== '*') throw new StyleError('usage: pair_coeff * * file.prm file.key');
    const prm = parseAmoebaPrm(ctx.readFile(args[2]), this.hippo);
    const key = parseAmoebaKey(ctx.readFile(args[3]), prm);
    this.prm = prm;
    this.key = key;
    this.vdwcut = key.vdwcut;
    this.vdwtaper = key.vdwtaper;
    this.mpolecut = key.mpolecut;
    this.precond = null;
    this.tables = null;
    this.version++;
  }

  /** Per class mixing (pair_amoeba.cpp mix()): radius and epsilon rules on the class values. */
  private mixTables(prm: AmoebaPrm, specialHal: readonly number[], dhal: number, ghal: number): HalTables {
    let nclass = 0;
    for (const c of prm.vdw.keys()) nclass = Math.max(nclass, c);
    for (const c of prm.typeToClass.values()) nclass = Math.max(nclass, c);
    const nt = nclass + 1;
    const radmin = new Float64Array(nt * nt), epsilon = new Float64Array(nt * nt);
    const kred = new Float64Array(nt);
    const TWOSIX = libmPow(2.0, 1.0 / 6.0);
    for (let c = 1; c <= nclass; c++) kred[c] = prm.vdw.get(c)?.kred ?? 0;
    for (let i = 1; i <= nclass; i++) {
      for (let j = i; j <= nclass; j++) {
        const vi = prm.vdw.get(i), vj = prm.vdw.get(j);
        if (!vi || !vj) throw new StyleError(`pair amoeba: no vdw parameters for class ${vi ? j : i}`);
        let ei = vi.eps, ej = vj.eps;
        let ri = vi.sigma, rj = vj.sigma;
        if (prm.radiusType === 'sigma') {
          ri *= TWOSIX;
          rj *= TWOSIX;
        }
        if (prm.radiusSize === 'diameter') {
          ri *= 0.5;
          rj *= 0.5;
        }
        const sri = Math.sqrt(ri);
        ei = Math.abs(ei);
        const sei = Math.sqrt(ei);
        const srj = Math.sqrt(rj);
        ej = Math.abs(ej);
        const sej = Math.sqrt(ej);
        let rij: number, eij: number;
        if (ri === 0.0 && rj === 0.0) rij = 0.0;
        else if (prm.radiusRule === 'arithmetic') rij = ri + rj;
        else if (prm.radiusRule === 'geometric') rij = 2.0 * sri * srj;
        else rij = 2.0 * (ri * ri * ri + rj * rj * rj) / (ri * ri + rj * rj);
        if (ei === 0.0 && ej === 0.0) eij = 0.0;
        else if (prm.epsilonRule === 'arithmetic') eij = 0.5 * (ei + ej);
        else if (prm.epsilonRule === 'geometric') eij = sei * sej;
        else if (prm.epsilonRule === 'harmonic') eij = 2.0 * (ei * ej) / (ei + ej);
        else if (prm.epsilonRule === 'hhg') eij = 4.0 * (ei * ej) / ((sei + sej) * (sei + sej));
        else eij = 2.0 * (sei * sej) * powint(ri * rj, 3) / (powint(ri, 6) + powint(rj, 6));
        radmin[i * nt + j] = radmin[j * nt + i] = rij;
        epsilon[i * nt + j] = epsilon[j * nt + i] = eij;
      }
    }
    // mix() (pair_amoeba.cpp): the vdwpr entries override the mixed radius and well depth of their class pair (the radius
    // is scaled by 2^(1/6) for radiustype sigma, and no diameter or radius scaling applies)
    for (const pr of prm.vdwPairs) {
      const rij = prm.radiusType === 'sigma' ? pr.sigma * TWOSIX : pr.sigma;
      radmin[pr.j * nt + pr.i] = radmin[pr.i * nt + pr.j] = rij;
      epsilon[pr.j * nt + pr.i] = epsilon[pr.i * nt + pr.j] = pr.eps;
    }
    return {
      off: this.vdwcut, cut: this.vdwtaper, dhal, ghal,
      specialHal: [1.0, specialHal[1], specialHal[2], specialHal[3], specialHal[4]],
      kred, nclass, radmin, epsilon,
      // no vdwl14 terms in AMOEBA force fields: the 1-4 radius and epsilon equal the 1-2 ones (mix() sets both)
      radmin4: radmin, epsilon4: epsilon,
    };
  }

  initOne(_i: number, _j: number): number {
    if (!this.prm || !this.key) throw new StyleError('pair_coeff * * file.prm file.key is required for pair style amoeba');
    if (!this.hippo) return Math.max(this.vdwcut, this.mpolecut, this.key.ewaldcut);
    // pair_amoeba.cpp init_one (not amoeba): the largest of the repulsion, dispersion, multipole, polarization and
    // charge transfer cutoffs
    const k = this.key;
    return Math.max(k.repcut, k.dispcut, k.mpolecut, k.ewaldcut, k.ctrncut);
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.prm || this.hippo) return;
    this.tables = this.mixTables(this.prm, this.prm.specialHal, this.prm.dhal, this.prm.ghal);
  }

  extract(name: string): unknown {
    if (name === 'pvector') return this.pvec;
    // the flags and weights the bonded styles of AMOEBA read from the pair (pair_amoeba.cpp extract)
    if (name === 'improper_flag' || name === 'angle_flag' || name === 'bond_flag' || name === 'dihedral_flag') return 1;
    if (name === 'urey_flag' || name === 'pitorsion_flag' || name === 'bitorsion_flag') return 1;
    if (name === 'opbend_cubic') return this.prm?.opbend[0] ?? 0;
    if (name === 'opbend_quartic') return this.prm?.opbend[1] ?? 0;
    if (name === 'opbend_pentic') return this.prm?.opbend[2] ?? 0;
    if (name === 'opbend_sextic') return this.prm?.opbend[3] ?? 0;
    if (name === 'atomic_num') return this.prm?.atomicNum;
    return undefined;
  }

  compute(pc: PairCompute): void {
    if (!this.prm || (!this.tables && !this.hippo) || !this.key) throw new StyleError('pair_coeff * * file.prm file.key is required for pair style amoeba');
    const prm = this.prm;
    const s = pc.s;
    // as native LAMMPS does (pair_amoeba.cpp compute at b332f28): per-atom energy and virial are not computed
    if (pc.eatom || pc.vatom) throw new StyleError(`Cannot (yet) compute per-atom energy/virial with pair_style ${this.name}`);
    if (prm.hasPolarity && prm.poltyp !== 'mutual') {
      throw new StyleError(`pair amoeba: polarization ${prm.poltyp} is not ported yet (only mutual)`);
    }
    if (pc.geom.triclinic) throw new StyleError('pair amoeba: triclinic boxes are not ported yet');
    // dewald is the dispersion Ewald sum, which only the HIPPO dispersion has (native computes no dispersion for AMOEBA)
    if (this.key.useDewald && !this.hippo) throw new StyleError('pair amoeba: the dispersion Ewald summation (key "dewald") is HIPPO only and is not ported for AMOEBA');
    // kewald (amoeba_kspace.cpp): aeewald is the key value or ewaldcof(ewaldcut); apewald is the key value or aeewald,
    // raised for boxes smaller than 6 A
    this.aeew = 0;
    this.apew = 0;
    if (this.key.useEwald) {
      this.aeew = this.key.aeewaldKey ?? ewaldcof(this.key.ewaldcut);
      this.apew = this.key.apewaldKey ?? this.aeew;
      if (this.key.apewaldKey === null) {
        const size = Math.min(pc.geom.lx, pc.geom.ly, pc.geom.lz);
        if (size < 6.0) this.apew += ((1.0 - this.apew) / 2.0) * (6.0 - size);
      }
    }
    const amtype = s.custom.get('amtype');
    if (!amtype) throw new StyleError('pair amoeba needs the per-atom amtype (fix property/atom i_amtype with read_data fix ... "Tinker Types")');
    const n = pc.nlocal, nall = pc.nall;
    // copies: own[k] is the owned atom of copy k (owned atoms are their own copies; ghosts are periodic images)
    const own = new Int32Array(nall);
    const cls = new Int32Array(nall);
    for (let k = 0; k < nall; k++) {
      own[k] = k < n ? k : pc.nb.owner[k];
      const c = prm.typeToClass.get(amtype.data[own[k]]);
      if (c === undefined) throw new StyleError(`pair amoeba: no atom class for amtype ${amtype.data[own[k]]}`);
      cls[k] = c;
    }
    const idOf = (k: number): number => s.id[own[k]];
    const idIndex = new Map<number, number>();
    for (let i = 0; i < n; i++) idIndex.set(s.id[i], i);
    // every copy (owned and ghost) of each atom ID; closest(id, ref) is native domain->closest_image(ref, map(id))
    const copies = new Map<number, number[]>();
    for (let k = 0; k < nall; k++) {
      const id = idOf(k);
      const list = copies.get(id);
      if (list) list.push(k);
      else copies.set(id, [k]);
    }
    const x = pc.x;
    // The copy choices (red2local, the axis atoms) are made on reneighboring steps and kept until the next one
    // (neighbor->ago == 0 in find_hydrogen_neighbors and find_multipole_neighbors), so they are memoized per build.
    if (this.memoBuild !== pc.nb.nbuild) {
      this.memo.clear();
      this.memoBuild = pc.nb.nbuild;
    }
    const closest = (id: number, ref: number): number => {
      const memoKey = `${ref}|${id}`;
      const known = this.memo.get(memoKey);
      if (known !== undefined) return known;
      const list = copies.get(id);
      if (!list) throw new StyleError(`pair amoeba: atom ${id} is not in the system`);
      let best = list[0], bd = Infinity;
      for (const c of list) {
        const dx = x[3 * c] - x[3 * ref], dy = x[3 * c + 1] - x[3 * ref + 1], dz = x[3 * c + 2] - x[3 * ref + 2];
        const d = dx * dx + dy * dy + dz * dz;
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      this.memo.set(memoKey, best);
      return best;
    };
    // 1-2 partners by ID (the bonds of the topology)
    const bonds = s.topo.bonds;
    const oneTwo = new Map<number, number[]>();
    for (let e = 0; e < bonds.n; e++) {
      if (bonds.type[e] === 0) continue;
      const a = bonds.atoms[2 * e], b = bonds.atoms[2 * e + 1];
      if (!oneTwo.has(a)) oneTwo.set(a, []);
      if (!oneTwo.has(b)) oneTwo.set(b, []);
      if (!oneTwo.get(a)!.includes(b)) oneTwo.get(a)!.push(b);
      if (!oneTwo.get(b)!.includes(a)) oneTwo.get(b)!.push(a);
    }
    // hydrogen reduction (find_hydrogen_neighbors): a copy with kred > 0 and one 1-2 partner is reduced toward the
    // copy of that partner nearest to it (native red2local = domain->closest_image(i, map(id)))
    const red2 = new Int32Array(nall);
    for (let k = 0; k < nall; k++) {
      red2[k] = k;
      if (this.hippo) continue;
      const partners = oneTwo.get(idOf(k)) ?? [];
      if (this.tables!.kred[cls[k]] === 0.0 || partners.length !== 1) continue;
      red2[k] = closest(partners[0], k);
    }
    // reduced positions (amoeba xred): owned atoms from their partner copy; a ghost is its owner's xred moved by the
    // image offset, then wrapped to within half a box of the ghost (pbc_xred), as native does
    const xred = new Float64Array(3 * nall);
    for (let k = 0; k < n; k++) {
      const j = red2[k];
      const rdn = this.hippo ? 0 : this.tables!.kred[cls[k]];
      for (let d = 0; d < 3; d++) xred[3 * k + d] = rdn * (x[3 * k + d] - x[3 * j + d]) + x[3 * j + d];
    }
    const prd = [pc.geom.lx, pc.geom.ly, pc.geom.lz];
    for (let k = n; k < nall; k++) {
      const o = own[k];
      for (let d = 0; d < 3; d++) {
        let v = xred[3 * o + d] + pc.nb.gimage[3 * k + d] * prd[d];
        const half = 0.5 * prd[d];
        if (pc.geom.periodic[d]) {
          let delta = v - x[3 * k + d];
          while (Math.abs(delta) > half) {
            v += delta < 0.0 ? prd[d] : -prd[d];
            delta = v - x[3 * k + d];
          }
        }
        xred[3 * k + d] = v;
      }
    }
    // 1-5 neighbours by ID (special.cpp onefive); the list codes them as special 4
    const fifth = buildOneFive(s);
    const fifthOf = new Map<number, Set<number>>();
    for (let i = 0; i < n; i++) {
      const set = new Set<number>();
      for (let k = fifth.offset[i]; k < fifth.offset[i + 1]; k++) set.add(fifth.partner[k]);
      fifthOf.set(i, set);
    }
    // pairs from the half list; special codes 1..3 come from the list, 4 from the 1-5 set
    const list = pc.half;
    if (!list) throw new StyleError('pair amoeba needs a half neighbor list');
    const pi: number[] = [], pj: number[] = [], ps: number[] = [];
    for (let ii = list.ilo ?? 0; ii < list.inum; ii++) {
      const i = ii;
      const start = list.firstneigh[i];
      for (let k = 0; k < list.numneigh[i]; k++) {
        const enc = list.neighbors[start + k];
        const j = enc & NEIGHMASK;
        let sb = sbOf(enc);
        if (sb === 0 && fifthOf.get(i)?.has(idOf(j))) sb = 4;
        pi.push(i);
        pj.push(j);
        ps.push(sb);
      }
    }
    const pairsArr = { pi: Int32Array.from(pi), pj: Int32Array.from(pj), ps: Int8Array.from(ps) };
    const inp: HalInput = {
      xred, cls, red2local: red2,
      pi: pairsArr.pi, pj: pairsArr.pj, pspecial: pairsArr.ps,
      npairs: pi.length,
    };
    this.pvec.fill(0);
    if (!this.hippo) {
      const r = halTerm(this.tables!, inp, pc.f);
      pc.acc.evdwl += r.e;
      for (let k = 0; k < 6; k++) pc.acc.virial[k] += r.virial[k];
      this.pvec[0] = r.e;
    }
    const needSites = prm.hasMultipoles || prm.hasPolarity || this.hippo;
    if (!needSites) return;
    const sites = this.mpoleSites(pc, prm, oneTwo, idIndex, amtype, n, closest);
    if (this.hippo) {
      this.hippoTerms(pc, prm, sites, { pi: pairsArr.pi, pj: pairsArr.pj, ps: pairsArr.ps, npairs: pi.length }, own, cls, amtype, n);
    }
    if (prm.hasMultipoles || this.hippo) {
      // HIPPO always has the charge-penetration term (the core charges interact even with zero permanent multipoles)
      const rr = this.hippo
        ? this.hippoMpoleReal(pc, prm, sites, pairsArr, own, cls)
        : multipoleReal(sites, x, pc.f, pairsArr.pi, pairsArr.pj, pairsArr.ps, pairsArr.pi.length, prm.specialMpole, prm.dielectric, this.mpolecut, own, this.aeew);
      for (let k = 0; k < 6; k++) pc.acc.virial[k] += rr.virial[k];
      // the Ewald self-energy of the permanent multipoles (amoeba_multipole.cpp multipole(), over the owned atoms)
      const felec = ELECTRIC / prm.dielectric;
      const term = 2.0 * this.aeew * this.aeew;
      const fterm = -felec * this.aeew / MY_PIS;
      const rp = sites.rpole;
      let eself = 0.0;
      for (let i = 0; i < n; i++) {
        const o = 13 * i;
        const ci = rp[o], dix = rp[o + 1], diy = rp[o + 2], diz = rp[o + 3];
        const qixx = rp[o + 4], qixy = rp[o + 5], qixz = rp[o + 6], qiyy = rp[o + 8], qiyz = rp[o + 9], qizz = rp[o + 12];
        const cii = ci * ci;
        const dii = dix * dix + diy * diy + diz * diz;
        const qii = 2.0 * (qixy * qixy + qixz * qixz + qiyz * qiyz) + qixx * qixx + qiyy * qiyy + qizz * qizz;
        eself += fterm * (cii + term * (dii / 3.0 + 2.0 * term * qii / 5.0));
      }
      rr.e += eself;
      // the PME reciprocal sum (multipole_kspace), with the grid of kewald: pme-grid or the box size, pme-order
      if (this.key.useEwald && this.mpoleKspace) {
        const grid = this.pmeGridOf(pc, this.key.bseorder);
        const ks = multipoleKspace(sites, n, x, pc.f, grid, this.aeew, prm.dielectric);
        rr.e += ks.e;
        for (let k = 0; k < 6; k++) pc.acc.virial[k] += ks.virial[k];
      }
      pc.acc.ecoul += rr.e;
      this.pvec[3] = rr.e;
    }
    if (prm.hasPolarity) {
      const e = this.polarizationTerm(pc, prm, sites, pairsArr, oneTwo, idIndex, amtype, n, own);
      pc.acc.ecoul += e;
      this.pvec[4] = e;
    }
  }

  /**
   * The HIPPO pair terms (pair_amoeba.cpp compute() for amoeba == false): the Pauli repulsion, the damped dispersion, the
   * charge-penetration multipoles (core and valence charges of each site) and the charge transfer. Energies go to the
   * pvector slots of native (repulse 1, disp 2, mpole 3, qxfer 5); the dispersion is evdwl and the rest ecoul.
   */
  private hippoTerms(
    pc: PairCompute, prm: AmoebaPrm, sites: MpoleSites, pairs: TermPairs,
    own: Int32Array, cls: Int32Array, amtype: { data: ArrayLike<number> }, n: number,
  ): void {
    const key = this.key!;
    const x = pc.x, f = pc.f;
    // per-type and per-class tables (zero where the PRM file has no entry, as native's zeroed arrays)
    let maxType = 0, maxClass = 0;
    const types = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      types[i] = amtype.data[i];
      if (types[i] > maxType) maxType = types[i];
    }
    for (let k = 0; k < cls.length; k++) if (cls[k] > maxClass) maxClass = cls[k];
    const sizpr = new Float64Array(maxType + 1), dmppr = new Float64Array(maxType + 1), elepr = new Float64Array(maxType + 1);
    for (const [t, r] of prm.repulsion) {
      if (t > maxType) continue;
      sizpr[t] = r.sizpr;
      dmppr[t] = r.dmppr;
      elepr[t] = r.elepr;
    }
    const csix = new Float64Array(maxClass + 1), adisp = new Float64Array(maxClass + 1);
    for (const [c, d] of prm.dispersion) {
      if (c > maxClass) continue;
      csix[c] = d.csix;
      adisp[c] = d.adisp;
    }
    const chgct = new Float64Array(maxClass + 1), dmpct = new Float64Array(maxClass + 1);
    for (const [c, t] of prm.chgtrn) {
      if (c > maxClass) continue;
      chgct[c] = t.chgct;
      dmpct[c] = t.dmpct;
    }
    const sd: HippoSiteData = { own, types, cls };
    // Pauli repulsion (repulse), damped dispersion (disp), charge transfer (qxfer)
    const rep = repulsionTerm(sites, x, f, pairs, sd, { sizpr, dmppr, elepr }, prm.specialRepel, key.repcut, key.reptaper);
    // the Ewald coefficient of the dispersion sum (key dewald-alpha, else ewaldcof(dispcut)); aewald = 0 without dewald
    const adew = key.useDewald ? (key.adewaldKey ?? ewaldcof(key.dispcut)) : 0.0;
    const disp = dispersionReal(x, f, pairs, own, cls, n, { csix, adisp }, prm.specialDisp, key.dispcut, adew);
    if (key.useDewald) {
      // csixpr (pair_amoeba.cpp): sum over class pairs of csix_i csix_j n_i n_j (first step; the counts do not change)
      const counts = new Float64Array(maxClass + 1);
      for (let i = 0; i < n; i++) counts[cls[i]] += 1;
      let csixpr = 0.0;
      for (let i = 1; i <= maxClass; i++) {
        for (let j = i + 1; j <= maxClass; j++) csixpr += csix[i] * csix[j] * counts[i] * counts[j];
      }
      csixpr *= 2.0;
      for (let i = 1; i <= maxClass; i++) csixpr += csix[i] * csix[i] * counts[i] * counts[i];
      const nfft: [number, number, number] = key.dpmeGrid
        ? [key.dpmeGrid[0], key.dpmeGrid[1], key.dpmeGrid[2]]
        : [dispersionGridSize(pc.geom.lx), dispersionGridSize(pc.geom.ly), dispersionGridSize(pc.geom.lz)];
      for (let d = 0; d < 3; d++) if (!factorable(nfft[d])) throw new StyleError('pair hippo: dpme-grid must factor into 2, 3 and 5');
      const grid = makeDispersionGrid(nfft, key.bsdorder, [pc.geom.lo[0], pc.geom.lo[1], pc.geom.lo[2]], [pc.geom.lx, pc.geom.ly, pc.geom.lz]);
      disp.e += dispersionKspace(x, f, n, cls, csix, grid, adew, csixpr);
    }

    const felec = ELECTRIC / prm.dielectric;
    const qx = chargeTransferTerm(x, f, pairs, cls, { chgct, dmpct }, prm.specialMpole, key.ctrncut, key.ctrntaper, felec);
    pc.acc.evdwl += disp.e;
    pc.acc.ecoul += rep.e + qx.e;
    for (let k = 0; k < 6; k++) pc.acc.virial[k] += rep.virial[k] + disp.virial[k] + qx.virial[k];
    this.pvec[1] = rep.e;
    this.pvec[2] = disp.e;
    this.pvec[5] = qx.e;
  }

  /**
   * The HIPPO multipole real-space term (multipole_real, HIPPO branch) with the charge penetration of each site; its
   * result has the same shape as multipoleReal. The pval and pcore arrays are per owned atom and per copy.
   */
  private hippoMpoleReal(
    pc: PairCompute, prm: AmoebaPrm, sites: MpoleSites, pairsArr: { pi: Int32Array; pj: Int32Array; ps: Int8Array },
    own: Int32Array, cls: Int32Array,
  ): { e: number; virial: Float64Array } {
    const pcore = new Float64Array(this.maxClassOf(cls) + 1), palpha = new Float64Array(pcore.length);
    for (const [c, t] of prm.chgpen) {
      if (c > this.maxClassOf(cls)) continue;
      pcore[c] = t.pcore;
      palpha[c] = t.palpha;
    }
    const nall = cls.length;
    const hs: HippoMpoleSites = { core: new Float64Array(nall), alpha: new Float64Array(nall), val: new Float64Array(nall) };
    for (let k = 0; k < nall; k++) {
      hs.core[k] = pcore[cls[k]];
      hs.alpha[k] = palpha[cls[k]];
    }
    // pval = pole - pcore per owned atom (pair_amoeba.cpp compute, first call); copies take their owner's value
    for (let k = 0; k < nall; k++) hs.val[k] = sites.pole[13 * own[k]] - pcore[cls[own[k]]];
    const mp = multipoleRealHippo(sites, pc.x, pc.f, { pi: pairsArr.pi, pj: pairsArr.pj, ps: pairsArr.ps, npairs: pairsArr.pi.length }, own, hs, prm.specialMpole, prm.dielectric, this.mpolecut, this.aeew);
    return { e: mp.e, virial: mp.virial };
  }

  private maxClassOf(cls: Int32Array): number {
    let m = 0;
    for (let k = 0; k < cls.length; k++) if (cls[k] > m) m = cls[k];
    return m;
  }

  /**
   * The induced-dipole polarization (pair_amoeba.cpp polar(), no Ewald): the polarization groups, the mutual solver
   * (amoeba_induce.cpp induce()), the polarization energy (polar_energy) and the real-space forces and torques (polar_real).
   * The energy goes to ecoul, as native eng_coul = erepulse + empole + epolar + eqxfer.
   */
  private polarizationTerm(
    pc: PairCompute, prm: AmoebaPrm, sites: MpoleSites,
    pairsArr: { pi: Int32Array; pj: Int32Array; ps: Int8Array },
    oneTwo: Map<number, number[]>, idIndex: Map<number, number>, amtype: { data: ArrayLike<number> }, n: number,
    own: Int32Array,
  ): number {
    const key = this.key!;
    const types = new Int32Array(n);
    for (let i = 0; i < n; i++) types[i] = amtype.data[i];
    const alpha = new Float64Array(n), pdamp = new Float64Array(n), thole = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const pol = prm.polar.get(types[i]);
      if (!pol) continue;
      alpha[i] = pol.alpha;
      pdamp[i] = libmPow(pol.alpha, 1.0 / 6.0);
      thole[i] = pol.thole;
    }
    // 1-2 partners as local indices (polar groups, assign_groups in pair_amoeba.cpp)
    const oneTwoLocal: number[][] = [];
    for (let i = 0; i < n; i++) {
      oneTwoLocal.push((oneTwo.get(pc.s.id[i]) ?? []).map((id) => idIndex.get(id) ?? -1));
    }
    const group = assignPolarGroups(n, oneTwoLocal, types, (t) => prm.polar.get(t)?.group ?? []);
    // HIPPO (pair hippo): core charge, damping exponent and valence charge of each copy (pcore, palpha of its class;
    // pval = pole - pcore of the owner, the same as the charge-penetration term)
    let hippo: HippoPolarAtoms | undefined;
    if (this.hippo) {
      const nall = own.length;
      hippo = { core: new Float64Array(nall), alphaP: new Float64Array(nall), val: new Float64Array(nall) };
      const pcoreOf = (c: number): number => prm.chgpen.get(c)?.pcore ?? 0;
      const palphaOf = (c: number): number => prm.chgpen.get(c)?.palpha ?? 0;
      const clsOf = (t: number): number => prm.typeToClass.get(t) ?? 0;
      for (let k = 0; k < nall; k++) {
        const c = clsOf(types[own[k]]);
        hippo.core[k] = pcoreOf(c);
        hippo.alphaP[k] = palphaOf(c);
      }
      for (let k = 0; k < nall; k++) hippo.val[k] = sites.pole[13 * own[k]] - pcoreOf(clsOf(types[own[k]]));
    }
    const atoms: PolarAtoms = {
      n, x: pc.x, own, rpole: sites.rpole, alpha, pdamp, thole, group,
      pi: pairsArr.pi, pj: pairsArr.pj, ps: pairsArr.ps, hippo,
    };
    // the reciprocal-space polarization (kewald: the grid of bsporder; udirect1, umutual1, polar_kspace)
    const useRecip = this.key!.useEwald && this.polarKspace && this.apew >= 1.0e-6;
    let gP: PmeGrid | null = null, fillP: PmeFill | null = null, wP: Float64Array | null = null, cmpP: Float64Array | null = null;
    if (useRecip) {
      gP = this.pmeGridOf(pc, this.key!.bsporder);
      fillP = pmeFill(pc.x, n, gP);
      wP = influence(gP, this.apew);
      cmpP = cmpFromRpole(sites.rpole, n);
    }
    const gridP = gP, fillPole = fillP, wPole = wP, cmpPole = cmpP;
    const recipOn = gridP !== null && fillPole !== null && wPole !== null && cmpPole !== null;
    const settings: PolarSettings = {
      aewald: this.apew,
      recipDirect: recipOn ? permanentRecipField(cmpPole!, n, fillPole!, gridP!, wPole!) : null,
      recipMutual: recipOn ? (u, up) => mutualRecipField(u, up, n, fillPole!, gridP!, wPole!) : null,
      pscale: prm.polarScale.p, piscale: prm.polarScale.pi,
      dscale: prm.polarScale.direct11, uscale: prm.polarScale.mutual11, dielectric: prm.dielectric,
      ewaldcut: key.ewaldcut, usolvcut: key.usolvcut, udiag: key.udiag, poleps: key.poleps, politer: key.politer,
      pcgprec: key.pcgprec, pcgguess: key.pcgguess, pcgpeek: key.pcgpeek,
      hippo: this.hippo, wscale: prm.polarScale.w,
    };
    // precond_neigh: rebuilt on each neighbor build (neighbor->ago == 0 in native)
    if (settings.pcgprec && (this.precond === null || this.precondBuild !== pc.nb.nbuild)) {
      this.precond = precondPairs(atoms, settings.usolvcut);
      this.precondBuild = pc.nb.nbuild;
    }
    const pre: PrecondPairs = settings.pcgprec ? this.precond! : { pi: new Int32Array(0), pj: new Int32Array(0), ps: new Int8Array(0) };
    const tens = settings.pcgprec ? buildPrecondTensors(atoms, pre, settings) : new Float64Array(0);
    const pairs = Array.from({ length: pairsArr.pi.length }, (_, k) => k);
    const ind = induceDipoles(atoms, settings, pairs, pre, tens);
    const e = polarEnergy(atoms, settings, ind);
    const vir = polarReal(atoms, settings, sites, ind, pc.f, pairs).virial;
    for (let k = 0; k < 6; k++) pc.acc.virial[k] += vir[k];
    if (recipOn) {
      const vpol = polarKspaceForces(sites, n, pc.x, pc.f, gridP, wPole, fillPole, cmpPole, ind.uind, ind.uinp, prm.dielectric, this.apew);
      for (let k = 0; k < 6; k++) pc.acc.virial[k] += vpol[k];
    }
    return e;
  }

  /** The PME grid of kewald: pme-grid or the box-size default, the B-spline order given (bseorder or bsporder). */
  private pmeGridOf(pc: PairCompute, order: number): PmeGrid {
    const lo: [number, number, number] = [pc.geom.lo[0], pc.geom.lo[1], pc.geom.lo[2]];
    const prd: [number, number, number] = [pc.geom.lx, pc.geom.ly, pc.geom.lz];
    const key = this.key!;
    const nfft: [number, number, number] = key.pmeGrid
      ? [key.pmeGrid[0], key.pmeGrid[1], key.pmeGrid[2]]
      : [defaultGridSize(prd[0]), defaultGridSize(prd[1]), defaultGridSize(prd[2])];
    for (let d = 0; d < 3; d++) if (!factorable(nfft[d])) throw new StyleError('pair amoeba: pme-grid must factor into 2, 3 and 5');
    checkPmeOrder(order, Math.min(nfft[0], nfft[1], nfft[2]));
    return makePmeGrid(nfft, order, lo, prd);
  }

  /** Per-atom multipole sites of one step: frames (kmpole), axis atoms, rotated moments (rpole) and rotation matrices. */
  private mpoleSites(
    pc: PairCompute, prm: AmoebaPrm,
    oneTwo: Map<number, number[]>, idIndex: Map<number, number>,
    amtype: { data: ArrayLike<number> }, n: number,
    closest: (id: number, ref: number) => number,
  ): MpoleSites {
    const s = pc.s;
    const ids = new Int32Array(n);
    const amt = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      ids[i] = s.id[i];
      amt[i] = amtype.data[i];
    }
    // 1-3 partners: partners of the 1-2 partners, without the atom itself and its 1-2 partners (sorted by ID)
    const oneThree = new Map<number, number[]>();
    for (const id of ids) {
      const direct = new Set(oneTwo.get(id) ?? []);
      const set = new Set<number>();
      for (const p of direct) for (const q of oneTwo.get(p) ?? []) if (q !== id && !direct.has(q)) set.add(q);
      oneThree.set(id, [...set].sort((a, b) => a - b));
    }
    const typeOfId = (id: number): number => {
      const li = idIndex.get(id);
      if (li === undefined) throw new StyleError(`pair amoeba: multipole neighbor ${id} is not a local atom`);
      return amt[li];
    };
    const topo: MpoleTopo = {
      nb12: (id) => [...(oneTwo.get(id) ?? [])].sort((a, b) => a - b),
      nb13: (id) => oneThree.get(id) ?? [],
      bonded: (a, b) => (oneTwo.get(a) ?? []).includes(b),
    };
    const sites = kmpole((t) => prm.multipoles.get(t) ?? [], ids, amt, typeOfId, topo);
    // axis atoms: the copy (owned or image) nearest to the site (native closest_image)
    mapAxes(sites, (i, id) => closest(id, i));
    chkpole(sites, pc.x);
    for (let i = 0; i < n; i++) {
      rotmat(sites, pc.x, i);
      rotsite(sites, i);
    }
    return sites;
  }
}
