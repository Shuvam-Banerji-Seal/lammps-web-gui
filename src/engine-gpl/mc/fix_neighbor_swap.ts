/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MC/fix_neighbor_swap.cpp and src/MC/fix_neighbor_swap.h at commit b332f28 (Daniel Schwen's
 * contributors and the LAMMPS developers: fix neighbor/swap, Monte Carlo swaps of atom types between a centre atom
 * and one of its Voronoi neighbours, docs.lammps.org/fix_neighbor_swap.html). Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Native structure kept here. The event runs where native's pre_exchange runs: after the neighbour build of a step
 * that the fix asked for (native force_reneighbor, next_reneighbor = step + 1 at creation, then + nevery), so the
 * engine calls it from postNeighbor (after the build of the same step). Per event: energy_stored = energy_full();
 * update_iswap_atoms_list(); ncycles attempts, each drawing (1) the centre atom index int(niswap * uniform), (2) the
 * neighbour selected_prob = global * uniform (only when the centre has neighbours), and (3) the acceptance uniform only
 * when the chosen types differ. The neighbour rows come from compute voronoi/atom neighbors yes (its local array, in
 * voro++ order, see src/engine-gpl/voro), taken once per event as native's compute_local is cached per timestep.
 *
 * Quirks kept as native has them (measured with native LAMMPS in the case headers): the 'types' list stores the
 * index of each swappable type (local_swap_type_list = jswaptype), and attempt_swap compares that index with the
 * centre's type (itype == jtype), so a no-op counts as an attempt and a success; the 'diff' option uses the actual
 * neighbour type. The distance weight is min over the centre's own position and its periodic images (ghost copies).
 *
 * Not supported (refused by name): charges (atom_style charge; the qtype bookkeeping), per-atom masses (rmass), and
 * the 'fix neighbor/swap' restart file (the counters are not restored across a write_restart).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { missingArgs } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { RanPark } from '../random_park';
import { nativeOrder } from '../../engine/atoms';
import { exp as libmExp } from '../libm/exp';

const num = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || w === '' || !Number.isFinite(v)) throw new StyleError(`Illegal fix neighbor/swap command: ${what} must be a number, got '${w ?? ''}'`);
  return v;
};
const int = (w: string | undefined, what: string): number => {
  const v = num(w, what);
  if (!Number.isInteger(v)) throw new StyleError(`Illegal fix neighbor/swap command: ${what} must be an integer, got '${w ?? ''}'`);
  return v;
};

/** The keywords that end a 'types' or 'rates' list (native is_keyword). */
const KEYWORDS = new Set(['region', 'ke', 'types', 'diff', 'rates']);

/** One candidate neighbour of the centre: local index, weight, and the type value the swap uses (index or type). */
interface Candidate { j: number; prob: number; typeValue: number }

export class FixNeighborSwap extends Fix {
  readonly style = 'neighbor/swap';
  private readonly mcEvery: number;
  private readonly ncycles: number;
  private readonly beta: number;
  private readonly invR0: number;
  private readonly voroId: string;
  private readonly rng: RanPark;
  private readonly typeList: number[] = [];
  private diffFlag = false;
  private ratesFlag = false;
  private readonly rateList: number[] = [];
  private keFlag = true;
  private regionId: string | null = null;
  private nextReneighbor: number;
  private energyStored = 0;
  private nattempts = 0;
  private nsuccesses = 0;
  /** swappable centre atoms (local indices, native local_swap_iatom_list) */
  private iatoms: number[] = [];
  private rows: Float64Array = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 6) throw new StyleError('Illegal fix neighbor/swap command: usage fix ID group neighbor/swap N ncycles seed T R0 voronoi-ID keyword values ...');
    this.vectorFlag = true;
    this.sizeVector = 2;
    this.globalFreq = 1;
    this.extvector = 0;
    this.mcEvery = int(args[0], 'nevery');
    if (this.mcEvery <= 0) throw new StyleError(`Illegal fix neighbor/swap command nevery value: ${this.mcEvery}`);
    this.ncycles = int(args[1], 'ncycles');
    if (this.ncycles < 0) throw new StyleError(`Illegal fix neighbor/swap command ncycles value: ${this.ncycles}`);
    const seed = int(args[2], 'seed');
    if (seed <= 0) throw new StyleError(`Illegal fix neighbor/swap command seed value: ${seed}`);
    const temperature = num(args[3], 'temperature');
    if (temperature <= 0) throw new StyleError(`Illegal fix neighbor/swap command temperature value: ${temperature}`);
    const r0 = num(args[4], 'R0');
    if (r0 <= 0) throw new StyleError(`Illegal fix neighbor/swap command R0 value: ${r0}`);
    this.beta = 1.0 / (sys.units.boltz * temperature);
    this.invR0 = 1.0 / r0;
    this.voroId = args[5];
    const comp = sys.compute(this.voroId);
    if (!comp.localFlag) throw new StyleError(`Voronoi compute ${this.voroId} does not compute local info`);
    if (comp.sizeLocalCols !== 3) throw new StyleError(`Voronoi compute ${this.voroId} does not give i, j, size as expected`);
    this.options(args.slice(6), sys.state.ntypes);
    this.rng = new RanPark(seed);
    // native: force_reneighbor = 1, next_reneighbor = ntimestep + 1
    this.nextReneighbor = sys.state.step + 1;
    if (sys.geom.box.general) throw new StyleError('fix neighbor/swap: general triclinic boxes are not supported');
  }

  private options(args: string[], ntypes: number): void {
    const where = 'fix neighbor/swap';
    let i = 0;
    while (i < args.length) {
      const key = args[i];
      if (key === 'region') {
        if (i + 2 > args.length) throw new StyleError(`Illegal ${where} region: missing argument`);
        this.regionId = args[i + 1];
        i += 2;
      } else if (key === 'ke') {
        if (i + 2 > args.length) throw new StyleError(`Illegal ${where} ke: missing argument`);
        const v = args[i + 1];
        if (v !== 'yes' && v !== 'no') throw new StyleError(`Illegal ${where} ke value '${v}' (yes or no)`);
        this.keFlag = v === 'yes';
        i += 2;
      } else if (key === 'types') {
        if (i + 3 > args.length) throw missingArgs(`${where} types`);
        if (this.diffFlag) throw new StyleError(`${where}: cannot use 'diff' and 'types' keywords together`);
        i++;
        this.typeList.length = 0;
        while (i < args.length && !KEYWORDS.has(args[i])) {
          if (this.typeList.length >= ntypes) throw new StyleError(`Too many arguments to ${where} types`);
          this.typeList.push(int(args[i], 'types'));
          i++;
        }
      } else if (key === 'diff') {
        if (this.diffFlag) throw new StyleError(`${where}: cannot use 'diff' keyword multiple times`);
        if (this.typeList.length !== 0) throw new StyleError(`${where}: cannot use 'diff' and 'types' keywords together`);
        this.typeList.push(int(args[i + 1], 'diff'));
        this.diffFlag = true;
        i += 2;
      } else if (key === 'rates') {
        if (i + ntypes >= args.length) throw missingArgs(`${where} rates`);
        i++;
        let n = 0;
        while (i < args.length && !KEYWORDS.has(args[i])) {
          if (n >= ntypes) throw new StyleError(`Too many values (> ${ntypes}) for ${where} rates`);
          this.rateList[n] = num(args[i], 'rates');
          n++;
          i++;
        }
        this.ratesFlag = true;
        if (n !== ntypes) throw new StyleError(`${where} rates keyword must have exactly ${ntypes} arguments`);
      } else {
        throw new StyleError(`Unknown ${where} keyword: ${key}`);
      }
    }
    if (this.typeList.length === 0 && !this.diffFlag) throw new StyleError(`Must specify at either 'types' or 'diff' keyword with ${where}`);
    if (this.typeList.length < 2 && !this.diffFlag) throw new StyleError(`Must specify at least 2 atom types in ${where} 'types' keyword`);
    const s = this.sys.state;
    if (s.rmass !== null) throw new StyleError(`${where}: per-atom masses (atom_style with rmass) are not supported by the browser engine`);
    for (const t of this.typeList) if (t <= 0 || t > ntypes) throw new StyleError(`Invalid atom type in ${where} command`);
  }

  /** compute_vector (native): f_ID[1] attempts, f_ID[2] successes. */
  computeVector(n: number): number {
    if (n === 0) return this.nattempts;
    if (n === 1) return this.nsuccesses;
    return 0;
  }

  /** fix neighbor/swap runs at the reneighbouring steps it asks for (native force_reneighbor). */
  reneighborDue(step: number): boolean {
    return step === this.nextReneighbor;
  }

  /** The event (native pre_exchange), after the neighbour build of the step. */
  postNeighbor(): void {
    if (this.sys.state.step !== this.nextReneighbor) return;
    this.event();
  }

  private event(): void {
    const s = this.sys.state;
    this.energyStored = this.energyFull();
    this.updateIswapList();
    const comp = this.sys.compute(this.voroId);
    this.rows = comp.localValues();
    let nsuccess = 0;
    for (let c = 0; c < this.ncycles; c++) nsuccess += this.attemptSwap();
    this.nattempts += this.ncycles;
    this.nsuccesses += nsuccess;
    this.nextReneighbor = s.step + this.mcEvery;
    // comm->forward_comm (native, the swap's comm_forward): every ghost takes its owner's type. The last trial may
    // have built the neighbour cache for a swapped configuration, and a rejection restores only the owned types.
    const nb = this.sys.nb;
    for (let k = 0; k < nb.nall; k++) nb.typeall[k] = s.type[nb.owner[k]];
  }

  /** Full potential energy of the current configuration (native energy_full: pre_force, the force styles, post_force). */
  private energyFull(): number {
    const sys = this.sys;
    sys.atomsChanged();
    for (const f of sys.fixes) f.preForce?.();
    const a = sys.forces();
    // modify->post_force: the post_force_group fixes (a dynamic group's FixGroup) before the post_force fixes
    sys.assignDynamicGroups(false);
    for (const f of sys.fixes) f.postForce?.();
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp;
  }

  private inRegion(i: number): boolean {
    if (this.regionId === null) return true;
    const s = this.sys.state;
    return this.sys.region(this.regionId).match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
  }

  /** update_iswap_atoms_list: the local atoms of the group (and region) with the first type of the list. */
  private updateIswapList(): void {
    const s = this.sys.state;
    this.iatoms = [];
    // native loops its local atoms in storage order (after the atom sort): nativeOrder
    const ord = nativeOrder(s);
    for (let k = 0; k < s.n; k++) {
      const i = ord[k];
      if ((s.mask[i] & this.groupBit) === 0) continue;
      if (s.type[i] !== this.typeList[0]) continue;
      if (!this.inRegion(i)) continue;
      this.iatoms.push(i);
    }
  }

  /** attempt_swap (native): returns 1 for an accepted or no-op swap, 0 otherwise. */
  private attemptSwap(): number {
    if (this.iatoms.length === 0) return 0;
    const s = this.sys.state;
    const energyBefore = this.energyStored;
    // pick_i_swap_atom: one draw
    const iwhich = Math.trunc(this.iatoms.length * this.rng.uniform());
    const i = this.iatoms[iwhich];
    const idCenter = s.id[i];
    const cand = this.buildINeighborList(i, idCenter);
    if (cand.length === 0) return 0;
    // pick_j_swap_neighbor: one draw; the candidates are picked by their weights in list order
    let localProb = 0;
    for (const c of cand) localProb += c.prob;
    const globalProb = localProb;
    const selected = globalProb * this.rng.uniform();
    let j = -1;
    let jtype = -1;
    if (selected >= 0 && selected < localProb) {
      let search = selected;
      for (const c of cand) {
        if (search > c.prob) search -= c.prob;
        else { j = c.j; jtype = c.typeValue; break; }
      }
      if (j < 0) throw new StyleError('fix neighbor/swap: did not select a local neighbor swap atom');
    }
    const itype = this.typeList[0];
    if (itype === jtype) {
      // native: the swap is a no-op and counts as a success (no acceptance draw)
      return 1;
    }
    if (j < 0) throw new StyleError('fix neighbor/swap: no neighbour was selected for the swap');
    // swap the types
    s.type[i] = jtype;
    s.type[j] = itype;
    const energyAfter = this.energyFull();
    if (this.rng.uniform() < libmExp(this.beta * (energyBefore - energyAfter))) {
      this.updateIswapList();
      if (this.keFlag) {
        const ri = this.sqrtMassRatio(itype, jtype);
        s.v[3 * i] *= ri; s.v[3 * i + 1] *= ri; s.v[3 * i + 2] *= ri;
        const rj = this.sqrtMassRatio(jtype, itype);
        s.v[3 * j] *= rj; s.v[3 * j + 1] *= rj; s.v[3 * j + 2] *= rj;
      }
      this.energyStored = energyAfter;
      return 1;
    }
    // rejected: restore the types
    s.type[i] = itype;
    s.type[j] = jtype;
    return 0;
  }

  /** sqrt(mass[a] / mass[b]) by type (native sqrt_mass_ratio without per-atom masses). */
  private sqrtMassRatio(a: number, b: number): number {
    const m = this.sys.state.massByType;
    return Math.sqrt(m[a] / m[b]);
  }

  /** build_i_neighbor_list: the neighbour candidates of centre atom i from the voronoi rows of its owner ID. */
  private buildINeighborList(i: number, idCenter: number): Candidate[] {
    const s = this.sys.state;
    const out: Candidate[] = [];
    const rows = this.rows;
    const nrows = rows.length / 3;
    for (let n = 0; n < nrows; n++) {
      if (rows[3 * n] !== idCenter) continue;
      const tempId = rows[3 * n + 1];
      // the first local atom with that ID (native loops over the owned atoms)
      let tj = -1;
      for (let j = 0; j < s.n; j++) if (s.id[j] === tempId) { tj = j; break; }
      if (tj < 0) continue;
      if (this.regionId !== null) {
        if (!this.inRegion(tj)) continue;
        if ((s.mask[tj] & this.groupBit) === 0) continue;
      } else if ((s.mask[tj] & this.groupBit) === 0) continue;
      if (this.diffFlag) {
        out.push(this.candidate(tj, i, idCenter, s.type[tj], s.type[tj]));
      } else {
        for (let js = 1; js < this.typeList.length; js++) {
          if (s.type[tj] === this.typeList[js]) out.push(this.candidate(tj, i, idCenter, s.type[tj], js));
        }
      }
    }
    return out;
  }

  /** The weight of one candidate: exp(-(r/R0)^2), times the rate of its type; r the nearest centre image. */
  private candidate(tj: number, i: number, idCenter: number, typeOfJ: number, typeValue: number): Candidate {
    const s = this.sys.state;
    const nb = this.sys.nb;
    const xj = s.x[3 * tj], yj = s.x[3 * tj + 1], zj = s.x[3 * tj + 2];
    const dist = (x: number, y: number, z: number): number => {
      const dx = xj - x, dy = yj - y, dz = zj - z;
      return dx * dx + dy * dy + dz * dz;
    };
    // distsq3(x[temp_j], x[i_center]); then the ghost copies of the centre, nlocal .. nlocal+nghost
    let r = Math.sqrt(dist(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]));
    const nlocal = s.n;
    const nall = nlocal + nb.nghost;
    for (let g = nlocal; g < nall; g++) {
      const rtmp = Math.sqrt(dist(nb.xall[3 * g], nb.xall[3 * g + 1], nb.xall[3 * g + 2]));
      if (s.id[nb.owner[g]] === idCenter && rtmp < r) r = rtmp;
    }
    const rr = r * this.invR0;
    const weight = libmExp(-(rr * rr));
    const prob = this.ratesFlag ? this.rateList[typeOfJ - 1] * weight : weight;
    return { j: tj, prob, typeValue };
  }
}
