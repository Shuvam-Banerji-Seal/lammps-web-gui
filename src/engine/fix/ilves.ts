import { Fix } from './fix';
import { harmonicRestraint } from './harmonicRestraint';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { TopoList } from '../types';
import { int, num } from '../commands/args';
import { typeWord } from '../labelmap';
import { isMolecularStyle, massOf } from '../atoms';
import { missingArgs } from '../force/util';
import { libm } from '../libm';

/*
 * fix ID group ilves tol iter N selectors ... [mode|linearangle|kbond|store values] — docs.lammps.org/fix_ilves.html:
 * "ILVES enforces holonomic distance constraints with Newton's method on a sparse system of nonlinear equations."
 * "Unlike fix shake, ILVES handles arbitrarily large connected constraint clusters" — the selected bonds and the A-C
 * virtual bond of every constrained angle form one constraint network, solved as a whole (no cluster limit).
 *
 * Selection: the fix shake rules (docs.lammps.org/fix_shake.html) with the MASSDELTA 0.1 mass fudge. An angle is
 * constrained when its three atoms are in the group, its type is in the a list and both flanking bonds are
 * constrained; its A-C target is the law-of-cosines distance of the two bond equilibrium lengths and the angle.
 *
 * Measured with native LAMMPS (black box; scratch probes: a stretched dimer, a three-atom chain, a three-atom angle,
 * a four-atom chain, a three-bond star, two independent dimers, weak and strong fix nvt on a dimer):
 *  - setup velocities: the relative velocity along every constrained separation is removed by Jacobi sweeps on the
 *    mass-weighted impulses (see projectVelocities). With tol 1e-12 this is the exact projection; with tol 4e-4 a
 *    four-atom chain takes three sweeps and keeps a residual of 1e-4 (its v3 = 0.00834808 -0.00600587 -0.00517863).
 *    Sweeps do not converge for a star of three bonds with two or three angles (native gives an unconverged iterate
 *    there; the engine reproduces the sweeps, so it does too). Coordinates are not moved at setup.
 *  - setup and every step: the constraint solve predicts the next separations with the full step,
 *    s_k = r_k + dt (v_a - v_b) + dt^2 ftm2v (f_a/m_a - f_b/m_b), f including the restraint forces; the factor is
 *    dt^2 ftm2v at setup as in every step (fix shake uses 0.5 at setup). A stretched dimer (1.2 to 1.0, no external
 *    force) gives a constraint force 2868.0688 = 1.2 / (dt^2 ftm2v), and the star of three bonds with one angle and
 *    the four-atom chain match the solve to 1e-11 in 50 steps (oracle cases w50il_*).
 *  - Newton: each step solves J dlam = -F with F_k = |s_k|^2 - d_k^2 and J_kj = 2 s_k . (dC_kj r_j), the Jacobian at the
 *    current separations, one banded LU per network. Before each step the largest relative error
 *    |g_k| / d_k^2 with g_k = (|s_k|^2 - d_k^2) / 2 is checked against tol; the check is global over all constraints
 *    (two independent dimers: one stretched 1.2 and one 1.01 is iterated to convergence too, force 143.403441691981
 *    against 142.6935 at its own tolerance). Dimer forces: tol 1e-4 gives 2866.1097704291 after two steps and
 *    2868.0687 after three; tol 1e-2 stops after two, tol 3e-2 after one.
 *  - mode fixed performs exactly iter steps with no check (a dimer with iter 2 gives 2866.1097704291).
 *  - constraint force on atom a is lam_j r_j and -lam_j r_j on atom b (r_j the current bond vector); its virial is the
 *    sum of lam_j r_j (x) r_j, which reproduces native press at step 0 and in 50 steps.
 *  - restrain: the A-C distance of a near-linear angle is a harmonic restraint E = k (r - d)^2 with forces on A and C
 *    that take part in the prediction; the energy is in pe by default (fix_modify energy no excludes it); its virial is
 *    not in the pressure; the angle potential is switched off (eangle 0). The default k = 5e5 kB (kbond 2000 gave
 *    f_c 175.046; the default gave 86.963, which is 175.046 * 993.6 / 2000).
 *  - skip: the A-C distance is not constrained and the angle potential stays on.
 *  - error: a selected angle type whose equilibrium angle is at or above the threshold is refused (native: threshold
 *    170 refuses 170.0 and accepts 169.9).
 *  - store yes: f_ID[1..3] is the constraint force of each atom from the latest solve (setup or step).
 *
 * Known gap, measured with native LAMMPS (not in this file): fix nvt takes its first thermostat update from the
 * temperature of its own setup, before the setup velocity projection of fix ilves. The engine's fix nh reads the
 * current temperature, so with fix nvt and fix ilves the trajectories differ from step 1 (a dimer with
 * temp 300 300 100: f_1[1] 0.00028430 native against 0.00025062 engine); the engine matches native with the setup-time
 * temperature in fix nh (checked by a temporary change that was reverted). Oracle cases therefore use fix nve.
 *
 * Minimization (fix_ilves.cpp at b332f28, min_setup, min_post_force, min_harmonic_bond): every constraint is a
 * harmonic bond k (r - d)^2 with the stiff k = KBOND_MIN_FACTOR (2e3) times the MD constant (kbond, default
 * 5e5 * boltz, so 1e9 * boltz by default, as fix shake); the linear-angle A-C restraints keep the MD constant. The
 * energy is the fix's scalar and enters the potential energy. A separation below 1e-10 gives neither force nor energy.
 * Refused: a second fix ilves.
 * Type words of the b, a and t keywords may be type labels (fix_ilves.html: "may use type labels"), resolved through the label map.
 */

/** Atom-mass match fudge of the b/t/m selectors (fix_ilves.html, MASSDELTA). */
const MASSDELTA = 0.1;
const KEYWORDS = new Set(['mode', 'linearangle', 'kbond', 'store']);

/** A constraint between two atoms (atom IDs) with its squared target length. */
interface Con {
  a: number;
  b: number;
  d2: number;
  /** A bond constraint (false: the A-C distance of an angle constraint). */
  bond: boolean;
}

/** A harmonic restraint E = k (r - d)^2 between two atoms (atom IDs). */
interface Restraint {
  a: number;
  c: number;
  d: number;
  k: number;
}

/** Sparse network of constraints: the coupling of constraint k to constraint j with c = dC_kj (mass factor). */
interface Coupling {
  j: number;
  c: number;
}

/**
 * Solves the linear system with a banded LU (partial pivoting inside the band). Row i of the band holds columns
 * i - kl .. i + kl + ku, so the banded storage fits the fill of the pivoting. Used for each connected constraint
 * network and for the setup velocity projection; returns null when the matrix is singular.
 */
export const solveBanded = (n: number, kl: number, ku: number, entries: { row: number; col: number; v: number }[], rhs: number[]): number[] | null => {
  const W = 2 * kl + ku + 1;
  const A = new Float64Array(n * W);
  const b = rhs.slice();
  // band index of (i, j): j - i + kl, valid for j in [i - kl, i + kl + ku]
  const at = (i: number, j: number) => {
    const q = j - i + kl;
    return q >= 0 && q < W ? i * W + q : -1;
  };
  for (const e of entries) {
    const p = at(e.row, e.col);
    if (p < 0) {
      if (e.v !== 0) throw new StyleError('fix ilves: constraint coupling outside the band');
      continue;
    }
    A[p] += e.v;
  }
  const get = (i: number, j: number) => { const p = at(i, j); return p < 0 ? 0 : A[p]; };
  const set = (i: number, j: number, v: number) => { const p = at(i, j); if (p >= 0) A[p] = v; };
  for (let k = 0; k < n; k++) {
    const last = Math.min(n - 1, k + kl);
    let piv = k;
    for (let i = k + 1; i <= last; i++) if (Math.abs(get(i, k)) > Math.abs(get(piv, k))) piv = i;
    if (Math.abs(get(piv, k)) < 1e-300) return null;
    if (piv !== k) {
      const hi = Math.min(n - 1, k + kl + ku);
      for (let j = k; j <= hi; j++) {
        const t = get(k, j);
        set(k, j, get(piv, j));
        set(piv, j, t);
      }
      const t = b[k];
      b[k] = b[piv];
      b[piv] = t;
    }
    const d = get(k, k);
    const hi = Math.min(n - 1, k + kl + ku);
    for (let i = k + 1; i <= last; i++) {
      const f = get(i, k) / d;
      if (f === 0) continue;
      for (let j = k + 1; j <= hi; j++) set(i, j, get(i, j) - f * get(k, j));
      b[i] -= f * b[k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    const hi = Math.min(n - 1, i + kl + ku);
    for (let j = i + 1; j <= hi; j++) s -= get(i, j) * x[j];
    x[i] = s / get(i, i);
  }
  return x;
};

/** Hooks each constraint network (connected component) keeps: its constraints in banded order and its bandwidth. */
interface Network {
  /** constraint indices in banded order */
  order: number[];
  /** half bandwidth of the coupling graph in that order */
  bw: number;
}

export class FixIlves extends Fix {
  readonly style = 'ilves';
  private tol: number;
  private maxIter: number;
  private fixedMode = false;
  private bondTypes = new Set<number>();
  private angleTypes = new Set<number>();
  private atomTypes = new Set<number>();
  private masses: number[] = [];
  private linearStyle: 'error' | 'skip' | 'restrain' = 'error';
  private linearThreshold = 175;
  private kbond: number | null = null;
  private storeForces = false;
  private cons: Con[] = [];
  private restraints: Restraint[] = [];
  /** per-constraint coupling list (includes the constraint itself) and mass factors, built at run setup */
  private adj: Coupling[][] = [];
  /** constraint networks (connected components of the coupling graph) */
  private networks: Network[] = [];
  private builtTopology = false;
  /** keyword skip: warned once per angle type */
  private warnedSkip = new Set<number>();
  private lastEnergy = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/EXTRA-FIX/fix_ilves.cpp FixIlves::FixIlves at b332f28): the word count, a molecular atom
    // style, then tol, iter and N, then their ranges
    if (args.length < 4) throw missingArgs('fix ilves');
    if (!isMolecularStyle(sys.state.atomStyle)) throw new StyleError('Fix ilves requires a molecular atom style with bond topology');
    this.tol = num(args[0], 'tol');
    this.maxIter = int(args[1], 'iter');
    int(args[2], 'N');
    // As native LAMMPS does (src/EXTRA-FIX/fix_ilves.cpp FixIlves::FixIlves at b332f28)
    if (this.tol <= 0.0) throw new StyleError('Fix ilves tolerance must be > 0');
    if (this.maxIter <= 0) throw new StyleError('Fix ilves iteration count must be > 0');
    let mode = '';
    for (let k = 3; k < args.length; k++) {
      const w = args[k];
      if (KEYWORDS.has(w)) {
        mode = '';
        const v = args[++k];
        if (w === 'mode') {
          if (v !== 'converge' && v !== 'fixed') throw new StyleError(`fix ilves: mode must be converge or fixed, got '${v ?? ''}'`);
          this.fixedMode = v === 'fixed';
        } else if (w === 'linearangle') {
          if (v !== 'error' && v !== 'skip' && v !== 'restrain') throw new StyleError(`fix ilves: linearangle style must be error, skip or restrain, got '${v ?? ''}'`);
          this.linearStyle = v;
          this.linearThreshold = num(args[++k], 'linearangle threshold');
        } else if (w === 'kbond') {
          this.kbond = num(v, 'kbond');
          if (!(this.kbond > 0)) throw new StyleError('fix ilves: kbond must be > 0');
        } else {
          if (v !== 'yes' && v !== 'no') throw new StyleError(`fix ilves: store must be yes or no, got '${v ?? ''}'`);
          this.storeForces = v === 'yes';
        }
        continue;
      }
      if (w === 'b' || w === 'a' || w === 't' || w === 'm') {
        // As native LAMMPS does (src/RIGID/fix_ilves.cpp FixIlves::FixIlves at b332f28): a selector must be followed by one or more values before the next selector or keyword
        const nxt = args[k + 1];
        if (nxt === undefined || nxt === 'b' || nxt === 'a' || nxt === 't' || nxt === 'm' || KEYWORDS.has(nxt)) {
          throw new StyleError(`Fix ilves selector '${w}' needs one or more values`);
        }
        mode = w;
        continue;
      }
      if (!mode) throw new StyleError(`fix ilves: '${w}' must follow a constraint keyword b, a, t or m`);
      if (mode === 'm') this.masses.push(num(w, 'mass'));
      else (mode === 'b' ? this.bondTypes : mode === 'a' ? this.angleTypes : this.atomTypes).add(int(typeWord(sys.labels, mode === 'b' ? 'bond' : mode === 'a' ? 'angle' : 'atom', w), `${mode} type`));
    }
    if (!this.bondTypes.size && !this.angleTypes.size && !this.atomTypes.size && !this.masses.length) {
      throw new StyleError('fix ilves: no bond, angle, atom type or mass selected');
    }
    if (sys.fixes.some((f) => f instanceof FixIlves && f.id !== id)) throw new StyleError('"Only one fix ilves may be defined at a time"');
    this.virialGlobal = true;
    this.thermoVirial = true;
    this.energyGlobal = true;
    this.thermoEnergy = true;
    this.scalarFlag = true;
    this.extscalar = 1;
    if (this.storeForces) {
      this.peratomFlag = true;
      this.sizePeratomCols = 3;
    }
  }

  // ---------------------------------------------------------------- topology

  init(): void {
    const ff = this.sys.ff;
    if (!ff.bond) throw new StyleError('fix ilves needs a bond style for the bond lengths');
    if (this.angleTypes.size && !ff.angle) throw new StyleError('fix ilves a needs an angle style for the angles');
    this.buildTopology(true);
  }

  /** Bond-length targets come from the bond style, the A-C targets from the angle style (law of cosines). */
  private buildTopology(report: boolean): void {
    const sys = this.sys;
    const s = sys.state;
    const ff = sys.ff;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    const inGroup = (id: number) => (s.mask[index.get(id)!] & this.groupBit) !== 0;
    const massMatch = (i: number) => this.masses.some((m) => Math.abs(massOf(s, i) - m) <= MASSDELTA);
    const bonds = s.topo.bonds;
    const bondOn: boolean[] = [];
    const constrainedLen = new Map<string, number>();
    const key = (p: number, q: number) => (p < q ? `${p}:${q}` : `${q}:${p}`);
    const cons: Con[] = [];
    const restraints: Restraint[] = [];
    const bondLength = (e: number) => {
      const r0 = ff.bond!.equilibrium(bonds.type[e]);
      if (!Number.isFinite(r0)) throw new StyleError(`fix ilves: bond style ${ff.bond!.name} has no equilibrium length`);
      return r0;
    };
    for (let e = 0; e < bonds.n; e++) {
      const a = bonds.atoms[2 * e], b = bonds.atoms[2 * e + 1], t = bonds.type[e];
      const ia = index.get(a)!, ib = index.get(b)!;
      const on = inGroup(a) && inGroup(b) && (this.bondTypes.has(t) || this.atomTypes.has(s.type[ia]) || this.atomTypes.has(s.type[ib]) || massMatch(ia) || massMatch(ib));
      bondOn.push(on);
      if (!on) continue;
      const r0 = bondLength(e);
      cons.push({ a, b, d2: r0 * r0, bond: true });
      constrainedLen.set(key(a, b), r0);
    }
    const angles = s.topo.angles;
    const keepA: boolean[] = new Array(angles.n).fill(true);
    for (let e = 0; e < angles.n; e++) {
      const x1 = angles.atoms[3 * e], x2 = angles.atoms[3 * e + 1], x3 = angles.atoms[3 * e + 2];
      const t = angles.type[e];
      if (!this.angleTypes.has(t)) continue;
      const th = ff.angle ? ff.angle.equilibriumRadians(t) : NaN;
      if (!Number.isFinite(th)) throw new StyleError(`fix ilves: angle style ${ff.angle!.name} has no equilibrium angle`);
      // native fix_ilves.cpp: thresh_rad = linear_threshold * MY_PI / 180.0; th >= thresh_rad
      const nearLinear = th >= (this.linearThreshold * Math.PI) / 180;
      if (nearLinear && this.linearStyle === 'error') {
        throw new StyleError(`fix ilves: angle type ${t} has an equilibrium angle at or above the linearangle threshold of ${this.linearThreshold} degrees and cannot be rigidly constrained; use linearangle skip or restrain`);
      }
      if (![x1, x2, x3].every(inGroup)) continue;
      const r1 = constrainedLen.get(key(x1, x2)), r2 = constrainedLen.get(key(x2, x3));
      if (r1 === undefined || r2 === undefined) continue;
      // native FixIlves::init calls cos() alone (glibc's FMA cos, libm.ts)
      const dac2 = r1 * r1 + r2 * r2 - 2 * r1 * r2 * libm.cosAlone(th);
      if (!nearLinear) {
        // rigid angle: the A-C distance is a constraint and the angle potential is switched off
        cons.push({ a: x1, b: x3, d2: dac2, bond: false });
        keepA[e] = false;
      } else if (this.linearStyle === 'restrain') {
        restraints.push({ a: x1, c: x3, d: Math.sqrt(dac2), k: this.kbond ?? 5e5 * s.units.boltz });
        keepA[e] = false;
      } else if (!this.warnedSkip.has(t)) {
        // skip: the angle keeps its bending potential, no A-C constraint
        this.warnedSkip.add(t);
        sys.warn(`fix ilves treating near-linear angle type(s) ${t} with linearangle mode skip`);
      }
    }
    this.cons = cons;
    this.restraints = restraints;
    const keepBonds = bondOn.map((on) => !on);
    this.buildCouplings(index);
    this.builtTopology = true;
    if (!report) return;
    ff.setTopologyOverride({
      bonds: filterList(bonds, keepBonds),
      angles: filterList(angles, keepA),
      bondsN: bonds.n,
      anglesN: angles.n,
    });
    let bwMax = 0;
    for (const net of this.networks) bwMax = Math.max(bwMax, net.bw);
    sys.log(`fix ${this.id} ilves: ${cons.length} constraints in ${this.networks.length} network(s), largest bandwidth ${bwMax}, ${restraints.length} restraint(s)`);
  }

  /**
   * Coupling lists: constraint k is coupled to constraint j when they share an atom; c_kj = sg(a_k, j)/m_a - sg(b_k, j)/m_b
   * with sg = +1 at the first atom of j, -1 at the second (the same coefficient fix shake uses per cluster).
   * Networks are the connected components, ordered with Cuthill-McKee so the banded solve is narrow.
   */
  private buildCouplings(index: Map<number, number>): void {
    const s = this.sys.state;
    const n = this.cons.length;
    const byAtom = new Map<number, number[]>();
    this.cons.forEach((c, k) => {
      for (const id of [c.a, c.b]) {
        if (!byAtom.has(id)) byAtom.set(id, []);
        byAtom.get(id)!.push(k);
      }
    });
    const mass = (id: number) => massOf(s, index.get(id)!);
    this.adj = this.cons.map((ck, k) => {
      const set = new Set<number>([k]);
      for (const id of [ck.a, ck.b]) for (const j of byAtom.get(id)!) set.add(j);
      return [...set].map((j) => {
        const cj = this.cons[j];
        const sg = (atom: number) => (atom === cj.a ? 1 : atom === cj.b ? -1 : 0);
        return { j, c: sg(ck.a) / mass(ck.a) - sg(ck.b) / mass(ck.b) };
      }).filter((e) => e.c !== 0 || e.j === k);
    });
    const deg = (k: number) => this.adj[k].length - 1;
    const seen = new Uint8Array(n);
    this.networks = [];
    const nodesByDeg = Array.from({ length: n }, (_, k) => k).sort((p, q) => deg(p) - deg(q));
    for (const start of nodesByDeg) {
      if (seen[start]) continue;
      // Cuthill-McKee from the lowest-degree node of the component
      const order: number[] = [];
      seen[start] = 1;
      order.push(start);
      for (let h = 0; h < order.length; h++) {
        const k = order[h];
        const nb = this.adj[k].map((e) => e.j).filter((j) => j !== k && !seen[j]).sort((p, q) => deg(p) - deg(q));
        for (const j of nb) {
          if (seen[j]) continue;
          seen[j] = 1;
          order.push(j);
        }
      }
      const pos = new Map<number, number>();
      order.forEach((k, p) => pos.set(k, p));
      let bw = 0;
      for (const k of order) for (const e of this.adj[k]) bw = Math.max(bw, Math.abs(pos.get(k)! - pos.get(e.j)!));
      this.networks.push({ order, bw });
    }
  }

  dofRemoved(groupBit: number): number {
    // the constraints count against a temperature compute when their first atom is in its group
    if (!this.builtTopology && this.sys.ff.bond) this.buildTopology(false);
    const s = this.sys.state;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    let n = 0;
    for (const c of this.cons) if (s.mask[index.get(c.a)!] & groupBit) n++;
    return n;
  }

  destroy(): void {
    this.sys.ff.setTopologyOverride(null);
  }

  /** fix_ilves.cpp min_setup: the log line that counts the constraints replaced by restraints. */
  minSetup(): void {
    let nb = 0, na = 0;
    for (const c of this.cons) { if (c.bond) nb++; else na++; }
    this.sys.log(`Fix ilves: replacing ${nb} bond and ${na} angle constraint(s) with harmonic restraints for minimization`);
  }

  /**
   * fix_ilves.cpp min_post_force: the constraints are harmonic restraints with k = kbond times KBOND_MIN_FACTOR (the
   * default MD constant 5e5 * boltz gives 1e9 * boltz), the linear-angle restraints with the MD constant.
   */
  minPostForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const kMd = this.kbond ?? 5e5 * sys.units.boltz;
    const kBond = kMd * 2e3;
    const index = this.indexOf();
    let e = 0;
    for (const c of this.cons) e += harmonicRestraint(s, sys.geom, index.get(c.a)!, index.get(c.b)!, Math.sqrt(c.d2), kBond, 1e-10, null);
    for (const rs of this.restraints) e += harmonicRestraint(s, sys.geom, index.get(rs.a)!, index.get(rs.c)!, rs.d, kMd, 1e-10, null);
    this.lastEnergy = e;
  }

  // ---------------------------------------------------------------- dynamics

  setup(): void {
    if (this.storeForces) {
      if (this.arrayAtom.length !== 3 * this.sys.state.n) this.arrayAtom = new Float64Array(3 * this.sys.state.n);
      this.arrayAtom.fill(0);
    }
    this.projectVelocities();
    this.solve();
  }

  postForce(): void {
    this.solve();
  }

  energy(): number {
    return this.thermoEnergy ? this.lastEnergy : 0;
  }

  computeScalar(): number {
    return this.lastEnergy;
  }

  /** Index of atom IDs in the current state. */
  private indexOf(): Map<number, number> {
    const s = this.sys.state;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    return index;
  }

  /**
   * Setup velocity projection (see the header). The relative velocity along each constrained separation is removed
   * by Jacobi sweeps: every sweep moves each multiplier by its own residual over its diagonal coefficient, the
   * residuals of all constraints are checked before a sweep against tol (relative to the current length squared),
   * and the sweeps stop at iter. Measured with native LAMMPS (black box): a four-atom bond chain at tol 4e-4 takes
   * three sweeps and gives v3 = (0.00834808, -0.00600587, -0.00517863); at tol 1e-12 it is the exact projection.
   * The sweeps do not converge when angle constraints share atoms (a star of three bonds with two or three angles
   * diverges), and the native result of such a case is the unconverged iterate.
   */
  private projectVelocities(): void {
    const nc = this.cons.length;
    if (!nc) return;
    const s = this.sys.state;
    const index = this.indexOf();
    const g = this.sys.geom;
    const d = [0, 0, 0];
    const at = this.cons.map((c) => [index.get(c.a)!, index.get(c.b)!]);
    const r = this.cons.map((_c, k) => {
      for (let q = 0; q < 3; q++) d[q] = s.x[3 * at[k][0] + q] - s.x[3 * at[k][1] + q];
      g.minimumImage(d);
      return [d[0], d[1], d[2]];
    });
    const r2 = r.map((rk) => rk[0] * rk[0] + rk[1] * rk[1] + rk[2] * rk[2]);
    const v0 = Float64Array.from(s.v);
    const mu = new Float64Array(nc);
    const vc = new Float64Array(v0.length);
    const residuals = (): Float64Array => {
      vc.set(v0);
      for (let j = 0; j < nc; j++) {
        const ia = at[j][0], ib = at[j][1];
        for (let q = 0; q < 3; q++) {
          vc[3 * ia + q] += (mu[j] * r[j][q]) / massOf(s, ia);
          vc[3 * ib + q] -= (mu[j] * r[j][q]) / massOf(s, ib);
        }
      }
      const res = new Float64Array(nc);
      for (let k = 0; k < nc; k++) {
        let acc = 0;
        for (let q = 0; q < 3; q++) acc += (vc[3 * at[k][0] + q] - vc[3 * at[k][1] + q]) * r[k][q];
        res[k] = acc;
      }
      return res;
    };
    let sweeps = 0;
    for (;;) {
      const res = residuals();
      let worst = 0;
      for (let k = 0; k < nc; k++) if (r2[k] > 0) worst = Math.max(worst, Math.abs(res[k]) / r2[k]);
      if (worst <= this.tol || sweeps >= this.maxIter) {
        if (worst > this.tol) this.sys.warn(`fix ilves: velocity projection did not converge in ${this.maxIter} sweeps`);
        break;
      }
      for (let k = 0; k < nc; k++) {
        const diag = this.adj[k].find((e) => e.j === k)!.c * r2[k];
        if (diag !== 0) mu[k] -= res[k] / diag;
      }
      sweeps++;
    }
    s.v.set(vc);
  }

  /**
   * Constraint force of this step: the predicted separations s_k = r_k + dt (v_a - v_b) + dt^2 ftm2v (f_a/m_a - f_b/m_b)
   * (f includes the restraint forces), then Newton steps on |s_k|^2 - d_k^2 until the largest |g_k|/d_k^2 is within tol
   * (converge) or exactly iter steps (fixed). Adds the constraint and restraint forces to the atoms and the virial.
   */
  private solve(): void {
    const s = this.sys.state;
    const dt = s.dt;
    const factor = dt * dt * s.units.ftm2v;
    const vir = this.virial;
    vir.fill(0);
    if (this.storeForces) this.arrayAtom.fill(0);
    const index = this.indexOf();
    const g = this.sys.geom;
    const d = [0, 0, 0];
    const nc = this.cons.length;
    // restraint forces (part of the prediction, added after the solve)
    const R = new Float64Array(3 * s.n);
    this.lastEnergy = 0;
    for (const rs of this.restraints) {
      const ia = index.get(rs.a)!, ic = index.get(rs.c)!;
      for (let q = 0; q < 3; q++) d[q] = s.x[3 * ia + q] - s.x[3 * ic + q];
      g.minimumImage(d);
      const rr = Math.hypot(d[0], d[1], d[2]);
      if (!(rr > 0)) continue;
      const stretch = rr - rs.d;
      this.lastEnergy += rs.k * stretch * stretch;
      for (let q = 0; q < 3; q++) {
        const f = (-2 * rs.k * stretch * d[q]) / rr;
        R[3 * ia + q] += f;
        R[3 * ic + q] -= f;
      }
    }
    if (nc === 0) {
      this.addRestraintForces(R);
      return;
    }
    const at = this.cons.map((c) => [index.get(c.a)!, index.get(c.b)!]);
    const r: number[][] = this.cons.map((_c, k) => {
      for (let q = 0; q < 3; q++) d[q] = s.x[3 * at[k][0] + q] - s.x[3 * at[k][1] + q];
      g.minimumImage(d);
      return [d[0], d[1], d[2]];
    });
    const s0: number[][] = this.cons.map((_c, k) => [0, 1, 2].map((q) => {
      const ia = at[k][0], ib = at[k][1];
      const ma = massOf(s, ia), mb = massOf(s, ib);
      return r[k][q] + dt * (s.v[3 * ia + q] - s.v[3 * ib + q]) + (factor * ((s.f[3 * ia + q] + R[3 * ia + q]) / ma - (s.f[3 * ib + q] + R[3 * ib + q]) / mb));
    }));
    const lam = new Float64Array(nc);
    const sep = (k: number): number[] => {
      const out = [s0[k][0], s0[k][1], s0[k][2]];
      for (const e of this.adj[k]) {
        const w = factor * e.c * lam[e.j];
        out[0] += w * r[e.j][0];
        out[1] += w * r[e.j][1];
        out[2] += w * r[e.j][2];
      }
      return out;
    };
    const d2 = this.cons.map((c) => c.d2);
    const seps: number[][] = new Array(nc);
    const residual = (): number => {
      let worst = 0;
      for (let k = 0; k < nc; k++) {
        const v = sep(k);
        seps[k] = v;
        const F = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - d2[k];
        worst = Math.max(worst, Math.abs(F / (2 * d2[k])));
      }
      return worst;
    };
    let steps = 0;
    let converged = false;
    for (;;) {
      const worst = residual();
      if (!this.fixedMode && worst <= this.tol) { converged = true; break; }
      if (steps >= this.maxIter) break;
      this.newtonStep(seps, r, lam, factor, d2);
      steps++;
    }
    if (!converged && !this.fixedMode) {
      this.sys.warn(`fix ilves: Newton iteration did not converge to tol in ${this.maxIter} iterations`);
    }
    // constraint forces and virial from the converged multipliers and the start-of-step bond vectors
    for (let k = 0; k < nc; k++) {
      const ia = at[k][0], ib = at[k][1];
      const rr = r[k];
      const l = lam[k];
      for (let q = 0; q < 3; q++) {
        const fq = l * rr[q];
        s.f[3 * ia + q] += fq;
        s.f[3 * ib + q] -= fq;
        if (this.storeForces) {
          this.arrayAtom[3 * ia + q] += fq;
          this.arrayAtom[3 * ib + q] -= fq;
        }
      }
      vir[0] += l * rr[0] * rr[0]; vir[1] += l * rr[1] * rr[1]; vir[2] += l * rr[2] * rr[2];
      vir[3] += l * rr[0] * rr[1]; vir[4] += l * rr[0] * rr[2]; vir[5] += l * rr[1] * rr[2];
    }
    this.addRestraintForces(R);
  }

  private addRestraintForces(R: Float64Array): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      for (let q = 0; q < 3; q++) s.f[3 * i + q] += R[3 * i + q];
    }
  }

  /**
   * One Newton step: J_kj = 2 s_k . (dC_kj r_j) at the current separations, F_k = |s_k|^2 - d_k^2, solve J delta = -F per
   * network (banded LU) and add delta to lam.
   */
  private newtonStep(seps: number[][], r: number[][], lam: Float64Array, factor: number, d2: number[]): void {
    for (const net of this.networks) {
      const order = net.order;
      const pos = new Map<number, number>();
      order.forEach((k, p) => pos.set(k, p));
      const entries: { row: number; col: number; v: number }[] = [];
      const rhs: number[] = new Array(order.length);
      order.forEach((k, p) => {
        const sk = seps[k];
        for (const e of this.adj[k]) {
          const rj = r[e.j];
          entries.push({ row: p, col: pos.get(e.j)!, v: 2 * factor * e.c * (sk[0] * rj[0] + sk[1] * rj[1] + sk[2] * rj[2]) });
        }
        rhs[p] = -(sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2] - d2[k]);
      });
      const delta = solveBanded(order.length, net.bw, net.bw, entries, rhs);
      if (!delta) throw new StyleError('fix ilves: singular constraint system');
      order.forEach((k, p) => { lam[k] += delta[p]; });
    }
  }
}

const filterList = (l: TopoList, keep: boolean[]): TopoList => {
  const n = keep.filter(Boolean).length;
  const out: TopoList = { n, width: l.width, type: new Int32Array(n), atoms: new Int32Array(n * l.width) };
  let k = 0;
  for (let e = 0; e < l.n; e++) {
    if (!keep[e]) continue;
    out.type[k] = l.type[e];
    for (let w = 0; w < l.width; w++) out.atoms[k * l.width + w] = l.atoms[e * l.width + w];
    k++;
  }
  return out;
};
