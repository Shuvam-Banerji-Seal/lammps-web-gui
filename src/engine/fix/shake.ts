import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { TopoList } from '../types';
import { int, num } from '../commands/args';
import { massOf } from '../atoms';

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
 * cluster by Newton iteration to the requested tolerance.
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
 * Minimization (restraints with kbond) is not implemented: it is an error.
 */

interface Cluster {
  /** Atom IDs, central atom first. */
  ids: number[];
  /** Constraints: [index a, index b] into ids and the squared target distance. */
  cons: { a: number; b: number; d2: number }[];
}

export class FixShake extends Fix {
  readonly style: 'shake' | 'rattle';
  private tol: number;
  private maxIter: number;
  private bondTypes = new Set<number>();
  private angleTypes = new Set<number>();
  private atomTypes = new Set<number>();
  private masses: number[] = [];
  private clusters: Cluster[] = [];
  /** keyword mol: molecule template-ID whose molecules may be added during the run (fix deposit shake). */
  private molTemplateId: string | null = null;

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
      if (w === 'kbond') { num(args[++k], 'kbond'); continue; }
      if (!mode) throw new StyleError(`fix ${style}: '${w}' must follow a constraint keyword b, a, t or m`);
      if (mode === 'm') this.masses.push(num(w, 'mass'));
      else (mode === 'b' ? this.bondTypes : mode === 'a' ? this.angleTypes : this.atomTypes).add(int(w, `${mode} type`));
    }
    if (sys.fixes.some((f) => f instanceof FixShake && f.id !== id)) throw new StyleError('"there can only be one shake or rattle fix defined in a simulation"');
    this.virialGlobal = true;
    this.thermoVirial = true;
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

  private buildClusters(): void {
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
        for (let e = 0; e < A.n; e++) {
          const [x1, x2, x3] = [A.atoms[3 * e], A.atoms[3 * e + 1], A.atoms[3 * e + 2]];
          if (x2 !== id || !this.angleTypes.has(A.type[e])) continue;
          const ends = new Set([x1, x3]);
          if (!(ends.has(c.ids[1]) && ends.has(c.ids[2]))) continue;
          const th = (ff.angle!.equilibrium(A.type[e]) * Math.PI) / 180;
          const d1 = Math.sqrt(c.cons[0].d2), d2 = Math.sqrt(c.cons[1].d2);
          c.cons.push({ a: 1, b: 2, d2: d1 * d1 + d2 * d2 - 2 * d1 * d2 * Math.cos(th) });
        }
      }
      for (const x of c.ids) {
        if (used.has(x)) throw new StyleError(`fix ${this.style}: "Shake clusters are connected" (atom ${x})`);
        used.add(x);
      }
      clusters.push(c);
    }
    this.clusters = clusters;
    // constrained bonds and angles are switched off in the force field
    const keepB: boolean[] = constrained.map((on) => !on);
    const angleOff = new Set<string>();
    for (const c of clusters) if (c.cons.length === 3 && c.ids.length === 3) angleOff.add(`${c.ids[0]}:${Math.min(c.ids[1], c.ids[2])}:${Math.max(c.ids[1], c.ids[2])}`);
    const A = s.topo.angles;
    const keepA: boolean[] = [];
    for (let e = 0; e < A.n; e++) {
      const key = `${A.atoms[3 * e + 1]}:${Math.min(A.atoms[3 * e], A.atoms[3 * e + 2])}:${Math.max(A.atoms[3 * e], A.atoms[3 * e + 2])}`;
      keepA.push(!(angleOff.has(key) && this.angleTypes.has(A.type[e])));
    }
    ff.setTopologyOverride({ bonds: filterList(bonds, keepB), angles: filterList(A, keepA), bondsN: bonds.n, anglesN: A.n });
    sys.log(`fix ${this.id} ${this.style}: ${clusters.length} clusters (${clusters.filter((c) => c.ids.length === 2).length} of 2 atoms, ${clusters.filter((c) => c.ids.length === 3 && c.cons.length === 2).length} of 3, ${clusters.filter((c) => c.ids.length === 4).length} of 4, ${clusters.filter((c) => c.cons.length === 3 && c.ids.length === 3).length} with an angle)`);
  }

  destroy(): void {
    this.sys.ff.setTopologyOverride(null);
  }

  dofRemoved(groupBit: number): number {
    const s = this.sys.state;
    let n = 0;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    for (const c of this.clusters) if (s.mask[index.get(c.ids[0])!] & groupBit) n += c.cons.length;
    return n;
  }

  minPostForce(): void {
    throw new StyleError(`fix ${this.style} during minimization (harmonic restraints with kbond) is not supported by the browser engine`);
  }

  // ---------------------------------------------------------------- SHAKE

  setup(): void {
    // measured: after setup the coordinates satisfy the constraints exactly (velocities unchanged)
    this.correctCoordinates();
    if (this.style === 'rattle') this.rattleVelocities();
    this.shake(0.5);
  }

  /** Moves each cluster onto its constraints along the current bond vectors, displacements weighted by 1/m. */
  private correctCoordinates(): void {
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
        }
      }
    });
    // the setup forces and energies stay those of the uncorrected coordinates, as in native LAMMPS
    // (its step-0 thermo pe is the uncorrected one); the next step's integration starts from here
    for (let i = 0; i < this.sys.state.n; i++) this.sys.geom.remap(this.sys.state.x, this.sys.state.image, i);
  }

  postForce(): void {
    if (this.style === 'rattle') this.rattlePositions();
    else this.shake(1);
  }

  finalIntegrate(): void {
    if (this.style === 'rattle') this.rattleVelocities();
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
   * Solves every cluster for multipliers lambda_j such that the separations
   * s0_k + factor * sum_j (sign/m) lambda_j r_j have the target lengths
   * (Newton iteration to tol), then hands them to apply().
   */
  private solveClusters(
    start: (at: number[], m: number[], r: number[][], cons: Cluster['cons']) => { s0: number[][]; factor: number },
    apply: (at: number[], m: number[], r: number[][], lam: number[]) => void,
  ): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const index = new Map<number, number>();
    for (let i = 0; i < s.n; i++) index.set(s.id[i], i);
    const d = [0, 0, 0];
    for (const c of this.clusters) {
      this.current = c;
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
      const lam = new Array(nc).fill(0);
      // coefficient of lambda_j in the separation of constraint k
      const C = Array.from({ length: nc }, (_, k) => Array.from({ length: nc }, (_, j) => {
        const ck = c.cons[k], cj = c.cons[j];
        const sg = (atom: number) => (atom === cj.a ? 1 : atom === cj.b ? -1 : 0);
        return factor * (sg(ck.a) / m[ck.a] - sg(ck.b) / m[ck.b]);
      }));
      let it = 0;
      let prevDev = Infinity;
      for (; it < this.maxIter; it++) {
        const sep = s0.map((sk, k) => [0, 1, 2].map((q) => {
          let v = sk[q];
          for (let j = 0; j < nc; j++) v += C[k][j] * lam[j] * r[j][q];
          return v;
        }));
        const F = sep.map((v, k) => v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - c.cons[k].d2);
        let dev = 0;
        for (let k = 0; k < nc; k++) { const a = Math.abs(F[k]) / c.cons[k].d2; if (a > dev) dev = a; }
        if (dev <= this.tol) break;
        // The position residual saturates at a floating-point floor well above the
        // user tolerance on ill-conditioned angle clusters; once it stops falling the
        // constraint force has converged to machine precision, so further Newton
        // sweeps only add noise to the virial. Stop there (only at that floor: a residual that rises
        // while still large keeps iterating up to the iteration limit).
        if (dev >= prevDev && dev < 1e-8) break;
        prevDev = dev;
        const J = Array.from({ length: nc }, (_, k) => Array.from({ length: nc }, (_, j) => 2 * C[k][j] * (sep[k][0] * r[j][0] + sep[k][1] * r[j][1] + sep[k][2] * r[j][2])));
        const delta = solve(J, F.map((x) => -x));
        if (!delta) { sys.warn(`fix ${this.style}: singular SHAKE system for cluster at atom ${c.ids[0]}`); break; }
        for (let j = 0; j < nc; j++) lam[j] += delta[j];
      }
      if (it === this.maxIter) sys.warn(`fix ${this.style}: SHAKE did not converge in ${this.maxIter} iterations for the cluster at atom ${c.ids[0]}`);
      apply(at, m, r, lam);
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
    });
  }

  /**
   * RATTLE position stage. The velocity constraint at final_integrate
   * projects the full-step velocity u = v + dt/2 f/m onto the constraint
   * manifold, independently of the constraint force G; so G reaches the next
   * positions only through the next half kick:
   *   x_next = x + dt (u + d/m) + dt^2/2 (f + G)/m,
   * with d the velocity impulse of the projection of u. Solving G for that
   * keeps both the distances and v . r exact every step, as measured with
   * native LAMMPS (rattle_clusters: lengths to 1e-16 and v . r ~ 1e-19 at
   * step 10).
   */
  private rattlePositions(): void {
    const s = this.sys.state;
    const dt = s.dt;
    const ftm2v = s.units.ftm2v;
    const half = 0.5 * dt * dt * ftm2v;
    const vir = this.virial;
    vir.fill(0);
    this.solveClusters((at, m, r, cons) => {
      // full-step velocity without G, then its projection impulse d (linear solve)
      const u = at.map((i, n) => [0, 1, 2].map((q) => s.v[3 * i + q] + (0.5 * dt * ftm2v * s.f[3 * i + q]) / m[n]));
      const nc = cons.length;
      const sg = (atom: number, j: number) => (atom === cons[j].a ? 1 : atom === cons[j].b ? -1 : 0);
      const A = Array.from({ length: nc }, (_, k) => Array.from({ length: nc }, (_, j) => {
        const ck = cons[k];
        return (sg(ck.a, j) / m[ck.a] - sg(ck.b, j) / m[ck.b]) * (r[j][0] * r[k][0] + r[j][1] * r[k][1] + r[j][2] * r[k][2]);
      }));
      const rhs = cons.map((k, kk) => -[0, 1, 2].reduce((acc, q) => acc + (u[k.a][q] - u[k.b][q]) * r[kk][q], 0));
      const dmu = solve(A, rhs) ?? new Array(nc).fill(0);
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
    });
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
