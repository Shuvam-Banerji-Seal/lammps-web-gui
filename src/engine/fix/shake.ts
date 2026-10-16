import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { TopoList } from '../types';
import { int, num } from '../commands/args';
import { typeWord } from '../labelmap';
import { massOf } from '../atoms';
import { harmonicRestraint } from './harmonicRestraint';
import { libm } from '../libm';

/*
 * fix ID group shake|rattle tol iter N constraint values ... —
 * docs.lammps.org/fix_shake.html: "Apply bond and angle constraints to
 * specified bonds and angles in the simulation by either the SHAKE or RATTLE
 * algorithms." "constraint = b or a or t or m"; "keyword = mol or kbond".
 *
 * "This command works by using the current forces on atoms to calculate an
 * additional constraint force which when added will leave the atoms in
 * positions that satisfy the SHAKE constraints (e.g. bond length) after the
 * next time integration step." With velocity Verlet that means: at post_force
 * of step n the velocity is the half-step one, so the next positions are
 * x + dt v + dt^2 (f + G)/m; at setup v is the full-step velocity and they are
 * x + dt v + dt^2/2 (f + G)/m. The constraint force G acts along the current
 * bond vectors (Ryckaert, Ciccotti and Berendsen 1977) and is solved per
 * cluster by iteration to the requested tolerance. Each cluster iterates on its
 * own with the Jacobian frozen at the predicted separations, and stops after the
 * sweep whose largest lambda update, times its prefactor (dt^2 ftm2v for the
 * step, 0.5 dt^2 ftm2v for the setup force solve, 1 for the setup coordinate
 * correction), is within tol (see solveClusters). RATTLE positions use the same
 * per-cluster solve with the 0.5 dt^2 ftm2v prefactor.
 * "A cluster is defined as a central atom connected to others in the cluster
 * by constrained bonds. LAMMPS allows for the following kinds of clusters to
 * be constrained: one central atom bonded to 1 or 2 or 3 atoms, or one
 * central atom bonded to 2 others and the angle between the three atoms also
 * constrained." An angle constraint is the distance between the two outer
 * atoms implied by the bond lengths and the angle. "For all constraints, a
 * particular bond is only constrained if both atoms in the bond are in the
 * group specified with the SHAKE fix."
 * "The degrees-of-freedom removed by SHAKE bonds and angles are accounted for
 * in temperature and pressure computations." "The default setting for this
 * fix is fix_modify virial yes."
 * RATTLE: "The velocity constraints lead to a linear system of equations
 * which can be solved analytically"; "The fix rattle command modifies forces
 * and velocities and thus should be defined after all other integration
 * fixes" — the velocity constraint (v_a - v_b) . r_ab = 0 is solved at setup
 * and at final_integrate.
 *
 * Measured with native LAMMPS (2 Sep 2026; oracle cases shake_clusters,
 * rattle_clusters):
 *  - constrained bonds and angles contribute no energy or force (ebond and
 *    eangle are 0 although the starting lengths differ from r0);
 *  - a cluster's constraints count against a temperature compute's degrees
 *    of freedom when the cluster's central atom is in the compute's group
 *    (a compute on the 6 central atoms of 4 water-like and 2 CH3-like
 *    clusters stops with "degrees of freedom < 0": 15 - 18).
 * Minimization (fix_shake.cpp at b332f28, min_setup, min_post_force; docs.lammps.org/fix_shake.html: "kbond value =
 * spring constant for springs that replace rigid constraints during minimization"): the constraints are not solved;
 * each constrained bond (and the A-C distance of an angle constraint) is a harmonic restraint k (r - r0)^2 with
 * k = kbond, default 1.0e9 * boltz ("kbond = 1.0e9*force->boltz" in the constructor). The restraint energy is the
 * fix's scalar and enters the potential energy (thermo_energy is 1 by default in the constructor); the forces are
 * added after the pair and bond forces at every minimizer evaluation. Its energy is 0 in dynamics (post_force
 * resets it, as native does).
 *
 * Measured with native LAMMPS (black box, 2 Sep 2026 probes: a 30-atom water/methyl/chain
 * box and a two-atom bond stretched from 1.2 to 1.0 with mass 12, dt 1 fs):
 *  - the stored array is the 3-column per-atom array f_ID[1..3]; compute reduce on f_ID[k]
 *    with store no stops with the native message: does not calculate per-atom values;
 *  - the values are set at setup only: they do not change during the run (the sumsq of the
 *    stored forces is the same at every step of a 6-step run);
 *  - a second run starting from satisfied constraints stores ~1e-12 (the setup correction is
 *    zero there);
 *  - the setup correction stored in the dimer is m dx / (0.5 dt^2 ftm2v) per atom (5736.1 for
 *    dx = 0.1, m = 12, dt = 1 fs, real units), equal and opposite on the two atoms, and the
 *    velocity-driven setup constraint (velocity along the bond, positions satisfied) stores zero;
 *  - the stored forces of each cluster sum to zero (internal forces).
 *
 * Measured with native LAMMPS (black box, 2 Sep 2026 probes on the 4500-atom SPC/E box of examples/PACKAGES/tally
 * with fix shake 0.0001 20 0 b 1 a 1, run 0, dumps at step 0):
 *  - the setup coordinates are not solved exactly at tol 1e-4: the bond-length error of the native dump is 3e-9 A,
 *    and the step-0 press depends on the iteration count (press -1216.307 after one sweep, -1289.832 after two,
 *    -1290.306 after three; iter 20 with tol 1e-4 and iter 5 with tol 1e-3 give the two-sweep value);
 *  - the step-0 press needs the frozen-Jacobian force solve at setup (one sweep gives -1216.307, two give
 *    -1289.832, the native value);
 *  - the setup solves use the same per-cluster update-size stop as the per-step solve (see below): the coordinate
 *    correction (factor 1) and the force solve (factor 0.5 dt^2 ftm2v), each cluster on its own. Measured with
 *    native LAMMPS (black box) on tests/oracle/w48sh_run0 at tol 1e-3, 1e-4, 1e-5 and 1e-7: the engine matches
 *    the step-0 positions to 1e-14 and the forces to 2e-8 at every tolerance, where the earlier global sweeps with a
 *    fixed minimum sweep count were off by 2e-9 A and 3e-3 at 1e-5 and 1e-7.
 *
 * Measured with native LAMMPS (black box, per-atom dumps of tests/oracle/w48sh_run20 after steps 1 to 3, with
 * the iteration limit set to k so that the native solve stops after exactly k sweeps at tol 1e-12):
 *  - the solve is per cluster: at tol 1e-4 the 1500 water clusters split into 868 that are already
 *    converged after two sweeps and 632 that need three, so a global sweep (every cluster swept while any is
 *    above tol) gives the wrong forces;
 *  - the Jacobian is frozen at the predicted separations (the sweep-k native results match the engine's
 *    frozen-Jacobian sweeps to 1e-7 relative, and the re-linearised ones do not);
 *  - the stop test is on the lambda update: a cluster stops after the sweep whose largest update |lambda|
 *    times dt^2 ftm2v is within tol; the largest update of the second sweep separates the 868 from the 632
 *    clusters exactly (0.23899 against 0.23918 in lambda units, which is 1e-4 / ftm2v dt^2 at 1 fs);
 *  - the residual of the constraints after the same sweep does not separate them (a relative or absolute,
 *    squared or linear, max or sum measure all overlap), so the residual is not the stop test.
 *
 * Measured with native LAMMPS (black box, per-atom dumps of the 4500-atom SPC/E box with fix rattle in place of
 * fix shake, nve defined before the fix, dumps at steps 1 and 2):
 *  - a cluster is crossing when one of its constrained separations (bond or angle) straddles a periodic
 *    boundary in the wrapped coordinates (a minimum-image difference that is not the raw difference). Clusters
 *    without a crossing follow the RATTLE rules of this file exactly (velocities to 1e-11, forces to 5e-8);
 *  - with n crossing separations the velocity correction of the projection impulse dmu is multiplied by n at
 *    the end of the step (v = u + n c, c the projection of u; exact to 1e-11 for n = 2 and 3), and the constraint
 *    force carries the extra term -(n - 1) m c / (dt/2 ftm2v) (per atom, the same multipliers as dmu on the
 *    separations); the extra term is in the dumped forces and in the virial of the pressure (without the virial
 *    term the step-1 pressure is off by about 2150 bar; with it, by 0.04);
 *  - the position solve on crossing clusters still differs from native by a constraint-type force of relative
 *    size 1e-5 to 5e-5 (not a sweep-count effect: the converged solve differs), so the forces of the crossing
 *    atoms are open at that level;
 *  - with fix rattle defined before nve, the velocity stage is set so that the nve half kick that follows gives
 *    the full-step velocity u + c (within 1e-5 of the native value at step 1), and the extra force is
 *    -n m c / (dt/2 ftm2v) for every cluster, n >= 1 (a non-crossing cluster then has no constraint force in the
 *    dumped forces); the step-2 forces of that ordering do not follow this rule and are not matched;
 *  - with fix rattle defined before nve the native run loses atoms at about step 12 on this box (temperature
 *    grows to 265 K at step 11), so that ordering cannot be checked over 20 steps.
 *
 * Measured with native LAMMPS (black box, tests/oracle/w73shake_gil: the graph-il ionic liquid, 320 BMI three-atom
 * clusters with a stretched start, fix shake 1e-4 20 0 b 1 2 a 1, run 0, the pppm mesh and g_ewald pinned):
 *  - the setup force solve runs for every cluster. Skipping the clusters whose residual was already inside the
 *    tolerance after the coordinate correction (residuals 1e-6 to 1e-4 here) moved the step-0 press by 555 bar
 *    (45953131 against 45953686.037) and the forces of the electrolyte by up to 12.7 (forces up to 4e4); with every
 *    cluster solved, press is 45953686.03704 (native 45953686.03705) and the forces agree to 2e-8;
 *  - the one-sweep solve of a two-atom cluster is exact: on the stretched bond of tests/oracle/w26shake_molb (tol
 *    1e-12) native gives the same converged positions at iteration limits 1 to 8, so one sweep of the quadratic in
 *    the multiplier (the separation is quadratic in lambda) is the native solve. The frozen-Jacobian step needed
 *    about 8 sweeps for those positions, and its 6e-14 residual made the setup force solve give press -2.5e-9
 *    where native gives -1.4e-11;
 *  - tests/oracle/w73shake_eth: 250 ethanol molecules of examples/PACKAGES/drude/ethanol without the drude atoms and
 *    dihedrals (2-, 3- and 4-atom clusters, fix shake 1e-4 20 0 b 2 3 5, cutoff Coulomb): the frozen-Jacobian
 *    two-atom solve left per-atom forces off by 0.0155 on a force of 1069 (1.4e-5 relative) and press 7467535.02
 *    against 7467534.5656; the solve as above matches the forces to 4e-12 and press to 1e-9 relative.
 */

interface Cluster {
  /** Atom IDs, central atom first. */
  ids: number[];
  /** Constraints: [index a, index b] into ids and the squared target distance. */
  cons: { a: number; b: number; d2: number }[];
}

/*
 * keyword store (docs.lammps.org/fix_shake.html): "The *store* keyword controls whether the fix stores the constraint
 * (or restraint) forces as a per-atom property." "By using *store yes* the original
 * constraint forces on all atoms can be accessed as a per-atom array of the fix."
 */
export class FixShake extends Fix {
  readonly style: 'shake' | 'rattle';
  private tol: number;
  private maxIter: number;
  private bondTypes = new Set<number>();
  private angleTypes = new Set<number>();
  private atomTypes = new Set<number>();
  private masses: number[] = [];
  private clusters: Cluster[] = [];
  /** Per-bond and per-angle "constrained" flags of the last build, indexed like s.topo (see extendTopology). */
  private keepBonds: boolean[] = [];
  private keepAngles: boolean[] = [];
  /** keyword mol: molecule template-ID whose molecules may be added during the run (fix deposit shake). */
  private molTemplateId: string | null = null;
  /** keyword store yes: per-atom constraint forces (f_ID[1..3]) set at setup (see the header). */
  private storeForces = false;
  /**
   * RATTLE: per cluster, from the last post-force solve: the atoms, the full-step velocity u = v + dt/2 F/m
   * (F without the constraint force), the velocity correction dv of the projection of u, and the number of
   * constrained separations that cross a periodic boundary (see rattleFinal).
   */
  private velStore: { at: number[]; u: number[][]; dv: number[][]; nCross: number }[] = [];
  /** keyword kbond: restraint spring constant during minimization (null = 1.0e9 * boltz). */
  private kbond: number | null = null;
  /** Energy of the minimization restraints of the last minimizer evaluation (fix_shake.cpp ebond, compute_scalar). */
  private restraintEnergy = 0;

  constructor(sys: System, id: string, group: string, args: string[], style: 'shake' | 'rattle') {
    super(sys, id, group, args);
    this.style = style;
    if (args.length < 4) throw new StyleError(`usage: fix ID group ${style} tol iter N constraint values ...`);
    this.tol = num(args[0], 'tol');
    this.maxIter = int(args[1], 'iter');
    int(args[2], 'N');
    if (!(this.tol > 0) || this.maxIter < 1) throw new StyleError(`fix ${style}: tol must be > 0 and iter >= 1`);
    let mode = '';
    for (let k = 3; k < args.length; k++) {
      const w = args[k];
      if (w === 'b' || w === 'a' || w === 't' || w === 'm') { mode = w; continue; }
      if (w === 'mol') {
        const t = args[++k];
        if (!t) throw new StyleError(`fix ${style}: keyword mol needs a molecule template-ID`);
        this.molTemplateId = t;
        continue;
      }
      if (w === 'kbond') {
        const kb = num(args[++k], 'kbond');
        if (!(kb >= 0)) throw new StyleError(`Illegal fix ${style} kbond value ${kb}. Must be >= 0.0`);
        this.kbond = kb;
        continue;
      }
      if (w === 'store') {
        const v = args[++k];
        if (v !== 'yes' && v !== 'no') throw new StyleError(`fix ${style}: store must be yes or no, got '${v ?? ''}'`);
        this.storeForces = v === 'yes';
        continue;
      }
      if (!mode) throw new StyleError(`fix ${style}: '${w}' must follow a constraint keyword b, a, t or m`);
      if (mode === 'm') this.masses.push(num(w, 'mass'));
      // fix_shake.html: "*b* values = one or more bond types (may use type labels)"
      else (mode === 'b' ? this.bondTypes : mode === 'a' ? this.angleTypes : this.atomTypes).add(int(typeWord(sys.labels, mode === 'b' ? 'bond' : mode === 'a' ? 'angle' : 'atom', w), `${mode} type`));
    }
    if (sys.fixes.some((f) => f instanceof FixShake && f.id !== id)) throw new StyleError('"there can only be one shake or rattle fix defined in a simulation"');
    this.virialGlobal = true;
    this.thermoVirial = true;
    // fix_shake.cpp (constructor): energy_global_flag = 1, thermo_energy = 1, scalar_flag = 1
    this.energyGlobal = true;
    this.thermoEnergy = true;
    this.scalarFlag = true;
    if (this.storeForces) {
      this.peratomFlag = true;
      this.sizePeratomCols = 3;
    }
  }

  // ---------------------------------------------------------------- clusters

  private massOf(i: number): number {
    const s = this.sys.state;
    return massOf(s, i);
  }

  init(): void {
    const ff = this.sys.ff;
    if (!ff.bond) throw new StyleError(`fix ${this.style} needs a bond style for the bond lengths`);
    if (this.angleTypes.size && !ff.angle) throw new StyleError(`fix ${this.style} a needs an angle style for the angles`);
    if (this.molTemplateId && !this.sys.molecules.has(this.molTemplateId)) {
      throw new StyleError(`fix ${this.style}: mol molecule template '${this.molTemplateId}' does not exist`);
    }
    this.buildClusters();
  }

  /**
   * Rebuilds the clusters and the force-field topology override. Called at run setup and, for the
   * mol keyword, by fix deposit after it adds a molecule
   * (docs.lammps.org/fix_shake.html: "The mol keyword should be used when other commands, such as
   * fix deposit or fix pour, add molecules on-the-fly during a simulation, and you wish to
   * constrain the new molecules via SHAKE.").
   */
  rebuildClusters(): void {
    const ff = this.sys.ff;
    if (!ff.bond) throw new StyleError(`fix ${this.style} needs a bond style for the bond lengths`);
    this.buildClusters();
  }

  /** Whether clusters exist for the current topology (set by the first build). */
  private clustersBuilt = false;

  /**
   * @param report false for the early build of dofRemoved: no log line and no force-field override yet
   *   (run setup builds again and does both).
   */
  private buildClusters(report = true): void {
    const sys = this.sys;
    const s = sys.state;
    const ff = sys.ff;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    const inGroup = (id: number) => (s.mask[index.get(id)!] & this.groupBit) !== 0;
    // the m constraint matches masses within 0.1 (measured: m 1.099 and 0.901 constrain a mass-1.0 atom, 1.101 and 0.899 do not)
    const massMatch = (i: number) => this.masses.some((m) => Math.abs(this.massOf(i) - m) <= 0.1);
    const bonds = s.topo.bonds;
    const constrained: boolean[] = [];
    const partners = new Map<number, { other: number; bond: number }[]>();
    for (let e = 0; e < bonds.n; e++) {
      const a = bonds.atoms[2 * e], b = bonds.atoms[2 * e + 1], t = bonds.type[e];
      const ia = index.get(a)!, ib = index.get(b)!;
      const on = inGroup(a) && inGroup(b) && (this.bondTypes.has(t) || this.atomTypes.has(s.type[ia]) || this.atomTypes.has(s.type[ib]) || massMatch(ia) || massMatch(ib));
      constrained.push(on);
      if (!on) continue;
      for (const [p, q] of [[a, b], [b, a]]) {
        if (!partners.has(p)) partners.set(p, []);
        partners.get(p)!.push({ other: q, bond: e });
      }
    }
    const clusters: Cluster[] = [];
    const used = new Set<number>();
    const bondLen = (e: number) => {
      const r0 = ff.bond!.equilibrium(bonds.type[e]);
      if (!Number.isFinite(r0)) throw new StyleError(`fix ${this.style}: bond style ${ff.bond!.name} has no equilibrium length`);
      return r0;
    };
    const ids = [...partners.keys()].sort((p, q) => p - q);
    // angle index by central atom (a scan of every angle per cluster is quadratic on large topologies)
    const anglesByCentre = new Map<number, number[]>();
    if (this.angleTypes.size) {
      const A = s.topo.angles;
      for (let e = 0; e < A.n; e++) {
        if (!this.angleTypes.has(A.type[e])) continue;
        const centre = A.atoms[3 * e + 1];
        const list = anglesByCentre.get(centre);
        if (list) list.push(e);
        else anglesByCentre.set(centre, [e]);
      }
    }
    for (const id of ids) {
      const mine = partners.get(id)!;
      const isCentral = mine.length > 1 || (mine.length === 1 && partners.get(mine[0].other)!.length === 1 && id < mine[0].other);
      if (!isCentral) continue;
      if (mine.length > 3) throw new StyleError(`fix ${this.style}: "Shake cluster of more than 4 atoms" (atom ${id})`);
      for (const p of mine) {
        if (partners.get(p.other)!.length > 1 && mine.length > 1) throw new StyleError(`fix ${this.style}: "Shake clusters are connected" (atoms ${id} and ${p.other})`);
      }
      const c: Cluster = { ids: [id, ...mine.map((p) => p.other)], cons: [] };
      mine.forEach((p, k) => { const r0 = bondLen(p.bond); c.cons.push({ a: 0, b: k + 1, d2: r0 * r0 }); });
      if (mine.length === 2 && this.angleTypes.size) {
        const A = s.topo.angles;
        // the angles centred on this atom, in topology order (the same visiting order as a scan of every angle)
        for (const e of anglesByCentre.get(id) ?? []) {
          const [x1, x2, x3] = [A.atoms[3 * e], A.atoms[3 * e + 1], A.atoms[3 * e + 2]];
          if (x2 !== id || !this.angleTypes.has(A.type[e])) continue;
          const ends = new Set([x1, x3]);
          if (!(ends.has(c.ids[1]) && ends.has(c.ids[2]))) continue;
          const th = (ff.angle!.equilibrium(A.type[e]) * Math.PI) / 180;
          const d1 = Math.sqrt(c.cons[0].d2), d2 = Math.sqrt(c.cons[1].d2);
          c.cons.push({ a: 1, b: 2, d2: d1 * d1 + d2 * d2 - 2 * d1 * d2 * libm.cos(th) });
        }
      }
      for (const x of c.ids) {
        if (used.has(x)) throw new StyleError(`fix ${this.style}: "Shake clusters are connected" (atom ${x})`);
        used.add(x);
      }
      clusters.push(c);
    }
    this.clusters = clusters;
    this.clustersBuilt = true;
    if (!report) return;
    // constrained bonds and angles are switched off in the force field
    const keepB: boolean[] = constrained.map((on) => !on);
    this.keepBonds = keepB;
    const angleOff = new Set<string>();
    for (const c of clusters) if (c.cons.length === 3 && c.ids.length === 3) angleOff.add(`${c.ids[0]}:${Math.min(c.ids[1], c.ids[2])}:${Math.max(c.ids[1], c.ids[2])}`);
    const A = s.topo.angles;
    const keepA: boolean[] = [];
    for (let e = 0; e < A.n; e++) {
      const key = `${A.atoms[3 * e + 1]}:${Math.min(A.atoms[3 * e], A.atoms[3 * e + 2])}:${Math.max(A.atoms[3 * e], A.atoms[3 * e + 2])}`;
      keepA.push(!(angleOff.has(key) && this.angleTypes.has(A.type[e])));
    }
    this.keepAngles = keepA;
    ff.setTopologyOverride({ bonds: filterList(bonds, keepB), angles: filterList(A, keepA), bondsN: bonds.n, anglesN: A.n });
    sys.log(`fix ${this.id} ${this.style}: ${clusters.length} clusters (${clusters.filter((c) => c.ids.length === 2).length} of 2 atoms, ${clusters.filter((c) => c.ids.length === 3 && c.cons.length === 2).length} of 3, ${clusters.filter((c) => c.ids.length === 4).length} of 4, ${clusters.filter((c) => c.cons.length === 3 && c.ids.length === 3).length} with an angle)`);
  }

  destroy(): void {
    this.sys.ff.setTopologyOverride(null);
  }

  /** The molecule template-ID of the mol keyword, or null (fix gcmc checks it matches its own mol). */
  moleculeTemplateId(): string | null {
    return this.molTemplateId;
  }

  /**
   * Keeps the constraints of the last build while bonds and angles were appended to (or removed from
   * the end of) the topology after it: the appended entries stay unconstrained. fix gcmc uses it for a
   * trial insertion, whose energy is evaluated before the new molecule is registered (measured with
   * native LAMMPS, black box: the trial energy includes the inserted molecule's bonds, see fix_gcmc.ts).
   */
  extendTopology(): void {
    const s = this.sys.state;
    const ff = this.sys.ff;
    const B = s.topo.bonds, A = s.topo.angles;
    const pad = (keep: boolean[], n: number) => keep.slice(0, n).concat(new Array(Math.max(0, n - keep.length)).fill(true));
    const keepB = pad(this.keepBonds, B.n), keepA = pad(this.keepAngles, A.n);
    ff.setTopologyOverride({ bonds: filterList(B, keepB), angles: filterList(A, keepA), bondsN: B.n, anglesN: A.n });
  }

  dofRemoved(groupBit: number): number {
    // Native LAMMPS finds the clusters when the fix is defined, so a temperature compute used before the
    // first run (velocity create, velocity scale) already excludes the constraints. Measured with native
    // LAMMPS (black box): 64 SPC/E waters (192 atoms, fix shake on bonds and angles, velocity create 300.0)
    // give temp 300 at run 0, i.e. dof = 3 N - 3 - 3 * 64 = 381; the engine's first build was run setup.
    if (!this.clustersBuilt && this.sys.ff.bond) this.buildClusters(false);
    const s = this.sys.state;
    let n = 0;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    for (const c of this.clusters) if (s.mask[index.get(c.ids[0])!] & groupBit) n += c.cons.length;
    return n;
  }

  /** fix_shake.cpp min_setup (the warning of init() for a minimization run); the forces come from minPostForce. */
  minSetup(): void {
    const k = this.kbond ?? 1.0e9 * this.sys.units.boltz;
    this.sys.warn(`Using fix ${this.style} with minimization.\n  Substituting constraints with harmonic restraint forces using kbond=${k.toPrecision(8)}`);
  }

  /**
   * fix_shake.cpp min_post_force: every constrained bond of a cluster (and the A-C distance of an angle constraint)
   * is a harmonic restraint with kbond; the energy is stored for energy() and computeScalar().
   */
  minPostForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const kbond = this.kbond ?? 1.0e9 * sys.units.boltz;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    if (this.storeForces) {
      if (this.arrayAtom.length !== 3 * s.n) this.arrayAtom = new Float64Array(3 * s.n);
      this.arrayAtom.fill(0);
    }
    let e = 0;
    for (const c of this.clusters) {
      for (const p of c.cons) {
        e += harmonicRestraint(s, sys.geom, index.get(c.ids[p.a])!, index.get(c.ids[p.b])!, Math.sqrt(p.d2), kbond, 0, this.storeForces ? this.arrayAtom : null);
      }
    }
    this.restraintEnergy = e;
  }

  energy(): number {
    return this.restraintEnergy;
  }

  computeScalar(): number {
    return this.restraintEnergy;
  }

  // ---------------------------------------------------------------- SHAKE

  setup(): void {
    this.restraintEnergy = 0;
    // measured: after setup the coordinates satisfy the constraints exactly (velocities unchanged)
    if (this.storeForces) {
      if (this.arrayAtom.length !== 3 * this.sys.state.n) this.arrayAtom = new Float64Array(3 * this.sys.state.n);
      this.arrayAtom.fill(0);
    }
    this.correctCoordinates();
    if (this.style === 'rattle') this.rattleVelocities();
    this.shake(0.5);
  }

  /** Moves each cluster onto its constraints along the current bond vectors, displacements weighted by 1/m. */
  private correctCoordinates(): void {
    // store yes: the correction displacement m dx of each atom, expressed as the force that moves it
    // by dx in the half-step kinematics 0.5 dt^2 ftm2v (see the header; measured 5736.1 for dx 0.1, m 12)
    const st = this.sys.state;
    const toForce = 1 / (0.5 * st.dt * st.dt * st.units.ftm2v);
    this.solveClusters((at, m, r, cons) => {
      const s0 = cons.map((_k, j) => r[j].slice());
      return { s0, factor: 1 };
    }, (at, m, r, lam) => {
      const s = this.sys.state;
      for (let j = 0; j < lam.length; j++) {
        const k = this.current!.cons[j];
        for (let q = 0; q < 3; q++) {
          s.x[3 * at[k.a] + q] += (lam[j] * r[j][q]) / m[k.a];
          s.x[3 * at[k.b] + q] -= (lam[j] * r[j][q]) / m[k.b];
          if (this.storeForces) {
            this.arrayAtom[3 * at[k.a] + q] += lam[j] * r[j][q] * toForce;
            this.arrayAtom[3 * at[k.b] + q] -= lam[j] * r[j][q] * toForce;
          }
        }
      }
    }, 1);
    // the setup forces and energies stay those of the uncorrected coordinates, as in native LAMMPS
    // (its step-0 thermo pe is the uncorrected one); the next step's integration starts from here
    for (let i = 0; i < this.sys.state.n; i++) this.sys.geom.remap(this.sys.state.x, this.sys.state.image, i);
  }

  postForce(): void {
    this.restraintEnergy = 0;
    if (this.style === 'rattle') this.rattlePositions();
    else this.shake(1);
  }

  finalIntegrate(): void {
    if (this.style === 'rattle') this.rattleFinal();
  }

  /**
   * Re-applies the position constraint as a force correction at the end of a
   * step, for fix ehex with the constrain keyword (docs.lammps.org/fix_ehex.html:
   * "apply the constraint algorithm (SHAKE or RATTLE) again at the end of the
   * timestep"). Positions and velocities are left untouched here; the added
   * force reaches the next step's integration, as measured with native LAMMPS
   * (black box): with constrain only the dumped forces change on the step the
   * ehex fix acts, while positions and velocities are identical to a run
   * without constrain on that step. The solve uses dt^2/2 because the added
   * force enters only the next step's first half kick, and it is the same
   * position solve for SHAKE and RATTLE (measured: the correction force of a
   * RATTLE fix equals the SHAKE one to 1e-7).
   */
  applyConstraint(): void {
    // The constraint force added here acts on the next step; native LAMMPS keeps
    // the virial of the step's post-force solve for thermo, so the re-application
    // must not replace it (measured with native LAMMPS (black box): the pressure
    // on the step the ehex fix acts equals the one without constrain).
    const saved = this.virial.slice();
    this.shake(0.5);
    this.virial.set(saved);
  }

  private current: Cluster | null = null;

  /**
   * Solves every cluster for multipliers lambda_j such that the separations s0_k + sum_j C_kj lambda_j r_j
   * have the target lengths, then hands them to apply(). Each cluster iterates on its own: every sweep solves
   * the linearised system for the multiplier update, with the Jacobian frozen at the predicted separations s0,
   * and applies it. The cluster stops after the sweep whose largest update times stepScale is within the
   * tolerance (at least one sweep). stepScale is the prefactor of C: dt^2 ftm2v for the per-step SHAKE solve,
   * 0.5 dt^2 ftm2v for the setup force solve and RATTLE positions, 1 for the setup coordinate correction.
   * A cluster with one constraint (a two-atom bond) is solved exactly in one sweep, since its separation is
   * quadratic in the multiplier; every cluster takes its first sweep, even when its residual is already inside the
   * tolerance (see the header).
   */
  private solveClusters(
    start: (at: number[], m: number[], r: number[][], cons: Cluster['cons']) => { s0: number[][]; factor: number },
    apply: (at: number[], m: number[], r: number[][], lam: number[]) => void,
    stepScale: number,
  ): void {
    if (!this.clusters.length) return;
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    const d = [0, 0, 0];
    interface Solve {
      c: Cluster;
      at: number[];
      m: number[];
      r: number[][];
      s0: number[][];
      /** Coefficient of lambda_j in the separation of constraint k. */
      C: number[][];
      /** Jacobian of the residuals in lambda, frozen at s0. */
      J: number[][];
      lam: number[];
    }
    const todo: Solve[] = this.clusters.map((c) => {
      const at = c.ids.map((id) => index.get(id)!);
      const m = at.map((i) => massOf(s, i));
      // current bond vectors (minimum image)
      const r = c.cons.map((k) => {
        for (let q = 0; q < 3; q++) d[q] = s.x[3 * at[k.a] + q] - s.x[3 * at[k.b] + q];
        g.minimumImage(d);
        return [d[0], d[1], d[2]];
      });
      const { s0, factor } = start(at, m, r, c.cons);
      const nc = c.cons.length;
      const C = Array.from({ length: nc }, (_, k) => Array.from({ length: nc }, (_, j) => {
        const ck = c.cons[k], cj = c.cons[j];
        const sg = (atom: number) => (atom === cj.a ? 1 : atom === cj.b ? -1 : 0);
        return factor * (sg(ck.a) / m[ck.a] - sg(ck.b) / m[ck.b]);
      }));
      const J = Array.from({ length: nc }, (_, k) => Array.from({ length: nc }, (_, j) =>
        2 * C[k][j] * (s0[k][0] * r[j][0] + s0[k][1] * r[j][1] + s0[k][2] * r[j][2])));
      return { c, at, m, r, s0, C, J, lam: new Array(nc).fill(0) };
    });
    for (const t of todo) {
      const nc = t.c.cons.length;
      for (let k = 0; ; k++) {
        const sep = t.s0.map((sk, kk) => [0, 1, 2].map((q) => {
          let v = sk[q];
          for (let j = 0; j < nc; j++) v += t.C[kk][j] * t.lam[j] * t.r[j][q];
          return v;
        }));
        const F = sep.map((v, q) => v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - t.c.cons[q].d2);
        let delta: number[] | null;
        if (nc === 1) {
          const x = quadraticRoot(t.C[0][0] * t.C[0][0] * dot3(t.r[0], t.r[0]), 2 * t.C[0][0] * dot3(t.s0[0], t.r[0]), F[0]);
          delta = x === null ? null : [x];
        } else delta = solve(t.J, F.map((x) => -x));
        if (!delta) { sys.warn(`fix ${this.style}: singular SHAKE system for cluster at atom ${t.c.ids[0]}`); break; }
        let big = 0;
        for (let j = 0; j < nc; j++) {
          t.lam[j] += delta[j];
          big = Math.max(big, Math.abs(delta[j]) * stepScale);
        }
        if (nc === 1 || big <= this.tol) break;
        if (k + 1 >= this.maxIter) {
          sys.warn(`fix ${this.style}: SHAKE did not converge in ${this.maxIter} iterations for the cluster at atom ${t.c.ids[0]}`);
          break;
        }
      }
    }
    for (const t of todo) {
      this.current = t.c;
      apply(t.at, t.m, t.r, t.lam);
    }
    this.current = null;
  }

  /** Adds constraint forces so the next positions satisfy the constraints; the prediction uses factor * dt^2. */
  private shake(dtfFactor: number): void {
    const s = this.sys.state;
    const dt = s.dt;
    const dtfsq = dtfFactor * dt * dt * s.units.ftm2v;
    const vir = this.virial;
    vir.fill(0);
    this.solveClusters((at, m, r, cons) => {
      const p = at.map((i, n) => [0, 1, 2].map((q) => dt * s.v[3 * i + q] + (dtfsq * s.f[3 * i + q]) / m[n]));
      return { s0: cons.map((k, j) => [0, 1, 2].map((q) => r[j][q] + p[k.a][q] - p[k.b][q])), factor: dtfsq };
    }, (at, _m, r, lam) => {
      for (let j = 0; j < lam.length; j++) {
        const k = this.current!.cons[j];
        for (let q = 0; q < 3; q++) {
          const fq = lam[j] * r[j][q];
          s.f[3 * at[k.a] + q] += fq;
          s.f[3 * at[k.b] + q] -= fq;
        }
        const rr = r[j];
        vir[0] += lam[j] * rr[0] * rr[0]; vir[1] += lam[j] * rr[1] * rr[1]; vir[2] += lam[j] * rr[2] * rr[2];
        vir[3] += lam[j] * rr[0] * rr[1]; vir[4] += lam[j] * rr[0] * rr[2]; vir[5] += lam[j] * rr[1] * rr[2];
      }
    }, dtfsq);
  }

  /**
   * RATTLE post-force stage: the position constraint G is solved with the
   * full-step velocity u = v + dt/2 F/m projected onto the velocity constraints
   * (impulse dmu), x_next = x + dt (u + dmu/m) + dt^2/2 (F + G)/m. The
   * constraint force of a cluster with n > 1 separations crossing a periodic
   * boundary carries an extra term (see the header and rattleFinal).
   */
  private rattlePositions(): void {
    const s = this.sys.state;
    const dt = s.dt;
    const ftm2v = s.units.ftm2v;
    const half = 0.5 * dt * dt * ftm2v;
    const vir = this.virial;
    vir.fill(0);
    this.velStore = [];
    const pending: { at: number[]; acc: number[][]; nCross: number; dmu: number[]; r: number[][]; cons: Cluster['cons'] }[] = [];
    this.solveClusters((at, m, r, cons) => {
      // full-step velocity without G, then its projection impulse dmu (linear solve)
      const u = at.map((i, n) => [0, 1, 2].map((q) => s.v[3 * i + q] + (0.5 * dt * ftm2v * s.f[3 * i + q]) / m[n]));
      const nc = cons.length;
      const sg = (atom: number, j: number) => (atom === cons[j].a ? 1 : atom === cons[j].b ? -1 : 0);
      const A = Array.from({ length: nc }, (_, k) => Array.from({ length: nc }, (_, j) => {
        const ck = cons[k];
        return (sg(ck.a, j) / m[ck.a] - sg(ck.b, j) / m[ck.b]) * (r[j][0] * r[k][0] + r[j][1] * r[k][1] + r[j][2] * r[k][2]);
      }));
      const rhs = cons.map((k, kk) => -[0, 1, 2].reduce((acc, q) => acc + (u[k.a][q] - u[k.b][q]) * r[kk][q], 0));
      const dmu = solve(A, rhs) ?? new Array(nc).fill(0);
      // the impulse of the projection, per atom of the cluster: sum_j dmu_j sg r_j (momentum-like, = m dv)
      const acc = at.map((_i, n) => [0, 1, 2].map((q) => {
        let v = 0;
        for (let j = 0; j < nc; j++) v += dmu[j] * sg(n, j) * r[j][q];
        return v;
      }));
      const dv = acc.map((a, n) => a.map((x) => x / m[n]));
      const nCross = this.imageCrossings(cons, at);
      this.velStore.push({ at, u, dv, nCross });
      // the constraint force carries the velocity correction of a cluster with more than one crossing
      // separation: -(nCross - 1) m dv / (dt/2 ftm2v) is added to f after the position solve (see below)
      pending.push({ at, acc, nCross, dmu, r, cons });
      const p = at.map((_i, n) => [0, 1, 2].map((q) => {
        let v = u[n][q];
        for (let j = 0; j < nc; j++) v += (dmu[j] * sg(n, j) * r[j][q]) / m[n];
        return dt * v + (half * s.f[3 * at[n] + q]) / m[n];
      }));
      return { s0: cons.map((k, j) => [0, 1, 2].map((q) => r[j][q] + p[k.a][q] - p[k.b][q])), factor: half };
    }, (at, _m, r, lam) => {
      for (let j = 0; j < lam.length; j++) {
        const k = this.current!.cons[j];
        for (let q = 0; q < 3; q++) {
          const fq = lam[j] * r[j][q];
          s.f[3 * at[k.a] + q] += fq;
          s.f[3 * at[k.b] + q] -= fq;
        }
        const rr = r[j];
        vir[0] += lam[j] * rr[0] * rr[0]; vir[1] += lam[j] * rr[1] * rr[1]; vir[2] += lam[j] * rr[2] * rr[2];
        vir[3] += lam[j] * rr[0] * rr[1]; vir[4] += lam[j] * rr[0] * rr[2]; vir[5] += lam[j] * rr[1] * rr[2];
      }
    }, half);
    const kickLater = this.integratorAfter();
    for (const e of pending) {
      const k = kickLater ? Math.max(e.nCross, 1) : Math.max(e.nCross - 1, 0);
      if (k === 0) continue;
      // the extra force is a constraint-type force: multiplier -k dmu_j / (dt/2 ftm2v) on the separation r_j
      const w = -k / (0.5 * dt * ftm2v);
      for (let j = 0; j < e.cons.length; j++) {
        const lam = w * e.dmu[j], rr = e.r[j], ck = e.cons[j];
        for (let q = 0; q < 3; q++) {
          s.f[3 * e.at[ck.a] + q] += lam * rr[q];
          s.f[3 * e.at[ck.b] + q] -= lam * rr[q];
        }
        vir[0] += lam * rr[0] * rr[0]; vir[1] += lam * rr[1] * rr[1]; vir[2] += lam * rr[2] * rr[2];
        vir[3] += lam * rr[0] * rr[1]; vir[4] += lam * rr[0] * rr[2]; vir[5] += lam * rr[1] * rr[2];
      }
    }
  }

  /** Whether an integrator fix comes after this one in the list (its final half kick is still to come). */
  private integratorAfter(): boolean {
    const fixes = this.sys.fixes;
    return fixes.slice(fixes.indexOf(this) + 1).some((f) => typeof f.initialIntegrate === 'function');
  }

  /** Number of the cluster's constrained separations whose wrapped coordinates straddle a periodic boundary. */
  private imageCrossings(cons: Cluster['cons'], at: number[]): number {
    const s = this.sys.state, g = this.sys.geom;
    let n = 0;
    for (const con of cons) {
      const d = [0, 0, 0], dm = [0, 0, 0];
      for (let q = 0; q < 3; q++) { d[q] = s.x[3 * at[con.a] + q] - s.x[3 * at[con.b] + q]; dm[q] = d[q]; }
      g.minimumImage(dm);
      for (let q = 0; q < 3; q++) if (Math.abs(d[q] - dm[q]) > 1e-9) { n++; break; }
    }
    return n;
  }

  /**
   * RATTLE velocity stage of the step, after the post-force solve (see the header for the measured rules). The
   * full-step velocity is u + c, c the projection impulse of u; a cluster with n > 1 crossing separations ends at
   * u + n c (measured with native LAMMPS). When the integrator comes after fix rattle its final half kick
   * has not happened yet, so the stored velocity is set so that the kick adds back to the same full-step velocity.
   */
  private rattleFinal(): void {
    const s = this.sys.state;
    const dt = s.dt;
    const ftm2v = s.units.ftm2v;
    const kickLater = this.integratorAfter();
    for (const e of this.velStore) {
      const k = kickLater ? 1 : Math.max(e.nCross, 1);
      for (let n = 0; n < e.at.length; n++) {
        const i = e.at[n], m = massOf(s, i);
        for (let q = 0; q < 3; q++) {
          let v = e.u[n][q] + k * e.dv[n][q];
          if (kickLater) v -= (0.5 * dt * ftm2v * s.f[3 * i + q]) / m;
          s.v[3 * i + q] = v;
        }
      }
    }
    this.velStore = [];
  }

  /** RATTLE: removes the relative velocity along each constrained separation (exact linear solve per cluster). */
  private rattleVelocities(addVirial = false): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    const d = [0, 0, 0];
    for (const c of this.clusters) {
      const at = c.ids.map((id) => index.get(id)!);
      const m = at.map((i) => massOf(s, i));
      const nc = c.cons.length;
      const r = c.cons.map((k) => {
        for (let q = 0; q < 3; q++) d[q] = s.x[3 * at[k.a] + q] - s.x[3 * at[k.b] + q];
        g.minimumImage(d);
        return [d[0], d[1], d[2]];
      });
      const sg = (atom: number, j: number) => (atom === c.cons[j].a ? 1 : atom === c.cons[j].b ? -1 : 0);
      const A = Array.from({ length: nc }, (_, k) => Array.from({ length: nc }, (_, j) => {
        const ck = c.cons[k];
        const w = sg(ck.a, j) / m[ck.a] - sg(ck.b, j) / m[ck.b];
        return w * (r[j][0] * r[k][0] + r[j][1] * r[k][1] + r[j][2] * r[k][2]);
      }));
      const rhs = c.cons.map((k, kk) => -[0, 1, 2].reduce((acc, q) => acc + (s.v[3 * at[k.a] + q] - s.v[3 * at[k.b] + q]) * r[kk][q], 0));
      const mu = solve(A, rhs);
      if (!mu) continue;
      if (addVirial) {
        // the velocity constraint as a force over the half step of final_integrate: F = mu r / (dt/2 ftm2v)
        const w = 1 / (0.5 * s.dt * s.units.ftm2v);
        for (let j = 0; j < nc; j++) {
          const rr = r[j], f = mu[j] * w;
          this.virial[0] += f * rr[0] * rr[0]; this.virial[1] += f * rr[1] * rr[1]; this.virial[2] += f * rr[2] * rr[2];
          this.virial[3] += f * rr[0] * rr[1]; this.virial[4] += f * rr[0] * rr[2]; this.virial[5] += f * rr[1] * rr[2];
        }
      }
      for (let j = 0; j < nc; j++) {
        const k = c.cons[j];
        for (let q = 0; q < 3; q++) {
          s.v[3 * at[k.a] + q] += (mu[j] * r[j][q]) / m[k.a];
          s.v[3 * at[k.b] + q] -= (mu[j] * r[j][q]) / m[k.b];
        }
      }
    }
  }
}

const dot3 = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Small root of a x^2 + b x + c = 0 (the root that vanishes with c), by the cancellation-free form; a = 0 gives
 * the linear root. Returns null when the equation has no real root.
 */
const quadraticRoot = (a: number, b: number, c: number): number | null => {
  if (a === 0) return b === 0 ? null : -c / b;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return null;
  const q = -0.5 * (b + (b >= 0 ? 1 : -1) * Math.sqrt(disc));
  return q === 0 ? 0 : c / q;
};

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

/** Gaussian elimination with partial pivoting for the small (<= 3x3) cluster systems. */
const solve = (A: number[][], b: number[]): number[] | null => {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-300) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let q = c; q <= n; q++) M[r][q] -= f * M[c][q];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
};
