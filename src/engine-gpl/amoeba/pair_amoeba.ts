/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/pair_amoeba.cpp (settings, coeff, mix(), choose(), compute(), the pvector energies) and
 * src/AMOEBA/amoeba_utils.cpp (add_onefive_neighbors, find_hydrogen_neighbors) at commit b332f28. AMOEBA package
 * authors: the LAMMPS developers (Sandia Corporation, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * pair_style amoeba: the buffered 14-7 van der Waals term (hal), the permanent multipole term (mpole, direct sum;
 * src/engine-gpl/amoeba/multipole.ts) and the induced-dipole polarization (mutual solver, polarization energy, forces
 * and torques; src/engine-gpl/amoeba/polar.ts), for the non-periodic case with no Ewald summation. DIRECT and OPT
 * polarization, polar-predict, periodic images (ghost atoms) and the Ewald summation are refused by name. An all-zero
 * term contributes exactly zero energy and force, so it does not block the run.
 * docs.lammps.org/pair_amoeba.html: "pair_style amoeba" takes no arguments and "pair_coeff * * file.prm file.key"
 * reads the Tinker PRM and key files.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, sbOf } from '../../engine/neighbor';
import { buildOneFive } from '../../engine/atoms';
import { halTerm, powint, type HalTables, type HalInput } from './hal';
import { parseAmoebaKey, parseAmoebaPrm, type AmoebaKey, type AmoebaPrm } from './params';
import { chkpole, kmpole, mapAxes, multipoleReal, rotmat, rotsite, type MpoleSites, type MpoleTopo } from './multipole';
import {
  assignPolarGroups, buildPrecondTensors, induceDipoles, polarEnergy, polarReal, precondPairs,
  type PolarAtoms, type PolarSettings, type PrecondPairs,
} from './polar';

/** pvector order of pair amoeba (pair_amoeba.cpp): hal, repulse, disp, mpole, polar, qxfer. */
const NVECTOR = 6;

export class PairAmoeba extends Pair {
  readonly name: string = 'amoeba';
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

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('usage: pair_style amoeba (no arguments)');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 4 || args[0] !== '*' || args[1] !== '*') throw new StyleError('usage: pair_coeff * * file.prm file.key');
    const prm = parseAmoebaPrm(ctx.readFile(args[2]));
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
    const TWOSIX = Math.pow(2.0, 1.0 / 6.0);
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
    if (prm.hasVdwPair) throw new StyleError('pair amoeba: vdwpr pair overrides are not ported yet');
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
    return Math.max(this.vdwcut, this.mpolecut, this.key.ewaldcut);
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.prm) return;
    this.tables = this.mixTables(this.prm, this.prm.specialHal, this.prm.dhal, this.prm.ghal);
  }

  extract(name: string): unknown {
    if (name === 'pvector') return this.pvec;
    return undefined;
  }

  compute(pc: PairCompute): void {
    if (!this.prm || !this.tables || !this.key) throw new StyleError('pair_coeff * * file.prm file.key is required for pair style amoeba');
    const prm = this.prm;
    const s = pc.s;
    if (pc.nall > pc.nlocal) throw new StyleError('pair amoeba: periodic images (ghost atoms) are not ported yet (stage 2a: hal only)');
    if (pc.eatom || pc.vatom) throw new StyleError('pair amoeba: per-atom energy and virial are not ported yet');
    if (prm.hasPolarity && prm.poltyp !== 'mutual') {
      throw new StyleError(`pair amoeba: polarization ${prm.poltyp} is not ported yet (only mutual)`);
    }
    const amtype = s.custom.get('amtype');
    if (!amtype) throw new StyleError('pair amoeba needs the per-atom amtype (fix property/atom i_amtype with read_data fix ... "Tinker Types")');
    const n = pc.nlocal;
    const cls = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const c = prm.typeToClass.get(amtype.data[i]);
      if (c === undefined) throw new StyleError(`pair amoeba: no atom class for amtype ${amtype.data[i]}`);
      cls[i] = c;
    }
    // hydrogen reduction: an atom with kred > 0 and one 1-2 partner is reduced toward it (amoeba_utils/pair_amoeba init)
    const idIndex = new Map<number, number>();
    for (let i = 0; i < n; i++) idIndex.set(s.id[i], i);
    const oneTwo = new Map<number, number[]>();
    const bonds = s.topo.bonds;
    for (let e = 0; e < bonds.n; e++) {
      if (bonds.type[e] === 0) continue;
      const a = bonds.atoms[2 * e], b = bonds.atoms[2 * e + 1];
      if (!oneTwo.has(a)) oneTwo.set(a, []);
      if (!oneTwo.has(b)) oneTwo.set(b, []);
      if (!oneTwo.get(a)!.includes(b)) oneTwo.get(a)!.push(b);
      if (!oneTwo.get(b)!.includes(a)) oneTwo.get(b)!.push(a);
    }
    const red2local = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      red2local[i] = i;
      const partners = oneTwo.get(s.id[i]) ?? [];
      if (this.tables.kred[cls[i]] === 0.0 || partners.length !== 1) continue;
      const p = idIndex.get(partners[0]);
      if (p === undefined) throw new StyleError('pair amoeba: a hydrogen partner is not a local atom (periodic images are not ported yet)');
      red2local[i] = p;
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
        if (sb === 0 && fifthOf.get(i)?.has(s.id[j])) sb = 4;
        pi.push(i);
        pj.push(j);
        ps.push(sb);
      }
    }
    const pairsArr = { pi: Int32Array.from(pi), pj: Int32Array.from(pj), ps: Int8Array.from(ps) };
    const inp: HalInput = {
      x: Float64Array.from(pc.x.subarray(0, 3 * n)),
      cls, red2local,
      pi: pairsArr.pi, pj: pairsArr.pj, pspecial: pairsArr.ps,
      npairs: pi.length,
    };
    const r = halTerm(this.tables, inp, pc.f);
    pc.acc.evdwl += r.e;
    for (let k = 0; k < 6; k++) pc.acc.virial[k] += r.virial[k];
    this.pvec.fill(0);
    this.pvec[0] = r.e;
    const needSites = prm.hasMultipoles || prm.hasPolarity;
    if (!needSites) return;
    const sites = this.mpoleSites(pc, prm, idIndex, oneTwo, amtype, n);
    if (prm.hasMultipoles) {
      const rr = multipoleReal(sites, pc.x, pc.f, pairsArr.pi, pairsArr.pj, pairsArr.ps, pairsArr.pi.length, prm.specialMpole, prm.dielectric, this.mpolecut);
      for (let k = 0; k < 6; k++) pc.acc.virial[k] += rr.virial[k];
      pc.acc.ecoul += rr.e;
      this.pvec[3] = rr.e;
    }
    if (prm.hasPolarity) {
      const e = this.polarizationTerm(pc, prm, sites, pairsArr, oneTwo, idIndex, amtype, n);
      pc.acc.ecoul += e;
      this.pvec[4] = e;
    }
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
  ): number {
    const key = this.key!;
    const types = new Int32Array(n);
    for (let i = 0; i < n; i++) types[i] = amtype.data[i];
    const alpha = new Float64Array(n), pdamp = new Float64Array(n), thole = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const pol = prm.polar.get(types[i]);
      if (!pol) continue;
      alpha[i] = pol.alpha;
      pdamp[i] = Math.pow(pol.alpha, 1.0 / 6.0);
      thole[i] = pol.thole;
    }
    // 1-2 partners as local indices (polar groups, assign_groups in pair_amoeba.cpp)
    const oneTwoLocal: number[][] = [];
    for (let i = 0; i < n; i++) {
      oneTwoLocal.push((oneTwo.get(pc.s.id[i]) ?? []).map((id) => idIndex.get(id) ?? -1));
    }
    const group = assignPolarGroups(n, oneTwoLocal, types, (t) => prm.polar.get(t)?.group ?? []);
    const x = Float64Array.from(pc.x.subarray(0, 3 * n));
    const atoms: PolarAtoms = {
      n, x, rpole: sites.rpole, alpha, pdamp, thole, group,
      pi: pairsArr.pi, pj: pairsArr.pj, ps: pairsArr.ps,
    };
    const settings: PolarSettings = {
      pscale: prm.polarScale.p, piscale: prm.polarScale.pi,
      dscale: prm.polarScale.direct11, uscale: prm.polarScale.mutual11, dielectric: prm.dielectric,
      ewaldcut: key.ewaldcut, usolvcut: key.usolvcut, udiag: key.udiag, poleps: key.poleps, politer: key.politer,
      pcgprec: key.pcgprec, pcgguess: key.pcgguess, pcgpeek: key.pcgpeek,
    };
    // precond_neigh: rebuilt on each neighbor build (neighbor->ago == 0 in native)
    if (settings.pcgprec && (this.precond === null || this.precondBuild !== pc.nb.nbuild)) {
      this.precond = precondPairs(atoms, settings.usolvcut);
      this.precondBuild = pc.nb.nbuild;
    }
    const pre: PrecondPairs = settings.pcgprec ? this.precond! : { pi: new Int32Array(0), pj: new Int32Array(0) };
    const tens = settings.pcgprec ? buildPrecondTensors(atoms, pre, settings) : new Float64Array(0);
    const pairs = Array.from({ length: pairsArr.pi.length }, (_, k) => k);
    const ind = induceDipoles(atoms, settings, pairs, pre, tens);
    const e = polarEnergy(atoms, settings, ind);
    const vir = polarReal(atoms, settings, sites, ind, pc.f, pairs).virial;
    for (let k = 0; k < 6; k++) pc.acc.virial[k] += vir[k];
    return e;
  }

  /** Per-atom multipole sites of one step: frames (kmpole), axis atoms, rotated moments (rpole) and rotation matrices. */
  private mpoleSites(
    pc: PairCompute, prm: AmoebaPrm,
    idIndex: Map<number, number>, oneTwo: Map<number, number[]>,
    amtype: { data: ArrayLike<number> }, n: number,
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
      if (li === undefined) throw new StyleError(`pair amoeba: multipole neighbor ${id} is not a local atom (periodic images are not ported yet)`);
      return amt[li];
    };
    const topo: MpoleTopo = {
      nb12: (id) => [...(oneTwo.get(id) ?? [])].sort((a, b) => a - b),
      nb13: (id) => oneThree.get(id) ?? [],
      bonded: (a, b) => (oneTwo.get(a) ?? []).includes(b),
    };
    const sites = kmpole((t) => prm.multipoles.get(t) ?? [], ids, amt, typeOfId, topo);
    mapAxes(sites, (id) => idIndex.get(id) ?? -1);
    chkpole(sites, pc.x);
    for (let i = 0; i < n; i++) {
      rotmat(sites, pc.x, i);
      rotsite(sites, i);
    }
    return sites;
  }
}
