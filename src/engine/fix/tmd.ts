import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { ramp } from './util';
import { massOf } from '../atoms';

/*
 * fix tmd — targeted molecular dynamics, written from docs.lammps.org/fix_tmd.html
 * (plans/lammps-docs-live/fix_tmd.rst) and measured with native LAMMPS (black box). Syntax quoted there:
 *   "fix ID group-ID tmd rho_final file1 N file2"
 * "Perform targeted molecular dynamics (TMD) on a group of atoms. A holonomic constraint is used to force the
 * atoms to move towards (or away from) the target configuration. The parameter "rho" is monotonically decreased
 * (or increased) from its initial value to rho_final at the end of the run." "Rho has distance units and is a
 * measure of the root-mean-squared distance (RMSD) between the current configuration of the atoms in the group
 * and the target coordinates listed in file1."
 *
 * Measured with native LAMMPS (black box), 8-atom and 4-of-8 group tests:
 *   rho(x) = sqrt( sum_i m_i |x_i - y_i|^2 / M ) over the group atoms, M = their total mass. With equal masses this is
 *     the RMSD with N = group count. Measured with masses 1 and 12 (lj and real) and 1 and 3: rho_old matches to 1e-15.
 *   The constraint quadratic is mass weighted the same way (sum m |e + gamma d|^2 = M rho^2), while the position
 *   correction gamma d is NOT mass weighted (measured: dx = gamma d per atom, residual 1e-14 for every mass).
 *   lambda = gamma_back M rho_old mvv2e / dt^2 (mvv2e = 1 in lj); the peptide example's first step matches native to 1e-11.
 *   rho_target(s) is a linear ramp from the rho at the start of the run to rho_final over the run (the run's
 *     start..stop span); the logged value at the last step equals rho_final.
 *   After each step the group satisfies rho(x_s) = rho_target(s) to 1e-12 (the constraint is met exactly).
 *
 * Constraint (per step, after the integrator has moved the atoms; x_free = the moved positions, x_old = the
 * positions at the start of the step, d_i = x_old_i - y_i, e_i = x_free_i - y_i):
 *   x_new_i = x_free_i + gamma d_i,   gamma from the quadratic  sum |e + gamma d|^2 = N rho_t^2 .
 * gamma_forward is the larger root with rho_t = rho_target(s); gamma_back is the same root with rho_t replaced by
 * rho_old = rho(x_old) (the previous step's rho). Both agree with the logged columns to 1e-12 (eight atoms,
 * dump precision limited the earlier check). The velocity after the constraint is (x_new - x_old)/dt (the
 * half-step velocity); the final half kick then adds dt/2 f_new. Measured: v_native - (x_new - x_old)/dt = dt/2 f_new
 * to 8e-10 with zero pair forces.
 *
 * Output columns (file2, written every N steps; first row at step 1 for N = 1): a header line with the
 * columns Step, rho_target, rho_old, gamma_back, gamma_forward, lambda, work_lambda, work_analytical. Measured:
 *   lambda as above (equal-mass 8-atom case: exact to 1e-16 on 10 steps);
 *   work_lambda = sum over steps of lambda * (rho_target(s) - rho_old(s)) (exact);
 *   work_analytical: the increment for step s is N gamma_forward rho_old d (1 - (d / rho_old)^2) / dt^2 with
 *   d = rho_target(s) - rho_old(s): measured to 2e-9 relative for steps >= 2 (8 atoms and the 4-atom group).
 *   The first step's increment is NOT reproduced: the native value is about 0.004 (8 atoms) or 0.0035 (group)
 *   times this formula, with no closed form found, so the first row and every later row of work_analytical
 *   differ from native by that step's difference. The engine applies the formula at every step.
 * "The scalar..." none: "No global or per-atom quantities are stored by this fix for access by various output
 * commands." Not invoked during energy minimization (no minimize hook).
 *
 * Target file (file1, ASCII or plain text): optional box lines (lo hi xlo xhi, ylo yhi, zlo zhi; only when image
 * flags are given), then one line per atom with id x y z, or id x y z nx ny nz; comments with a hash sign and
 * blank lines are allowed.
 * The doc: "The atom lines (all or none of them) can optionally be followed by 3 integer values" and "every atom
 * in the group must be listed in the file." Image flags add n * box length to the coordinate.
 *
 * Restrictions per the doc: TMD fixes must come after the integrators (the engine applies the constraint in
 * postIntegrate, after initialIntegrate of every fix). Restarts and multiple runs ramp from the rho measured at
 * the start of each run (the engine's run start/stop ramp).
 */

const BOX_RE = /^(\S+)\s+(\S+)\s+([xyz])lo\s+\3hi$/;

/**
 * Parses the target file: optional three box lines (lo hi xlo xhi, ylo yhi, zlo zhi), then one line per atom,
 * id x y z, or when the box lines are given id x y z nx ny nz. Returns the target coordinates by atom ID and
 * the box lengths used for the image flags (from the box lines when present).
 */
const parseTarget = (text: string, file: string): { targets: Map<number, [number, number, number]>; box: [number, number, number] | null } => {
  const lines = text.split('\n').map((l) => l.replace(/#.*/, '').trim()).filter((l) => l.length > 0);
  const boxLines = lines.slice(0, 3);
  const hasBox = boxLines.length === 3 && boxLines.every((l) => BOX_RE.test(l));
  let box: [number, number, number] | null = null;
  let atoms = lines;
  if (hasBox) {
    box = [0, 0, 0];
    boxLines.forEach((l, d) => {
      const w = l.split(/\s+/);
      const lo = Number(w[0]), hi = Number(w[1]);
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) throw new StyleError(`fix tmd: bad box line in ${file}: '${l}'`);
      box![d] = hi - lo;
    });
    atoms = lines.slice(3);
  }
  if (atoms.length === 0) throw new StyleError(`fix tmd: no atom lines in ${file}`);
  const nw = atoms[0].split(/\s+/).length;
  if (nw !== 4 && nw !== 7) throw new StyleError(`fix tmd: atom lines in ${file} need 4 values (id x y z) or 7 (with image flags), got ${nw}`);
  const withImages = nw === 7;
  if (withImages && !hasBox) throw new StyleError(`fix tmd: ${file} has image flags but no box lines (the first 3 lines must give the box)`);
  if (!withImages && hasBox) throw new StyleError(`fix tmd: ${file} has box lines but no image flags (the box lines must not appear)`);
  const targets = new Map<number, [number, number, number]>();
  for (const l of atoms) {
    const w = l.split(/\s+/);
    if (w.length !== nw) throw new StyleError(`fix tmd: malformed atom line in ${file}: '${l}'`);
    const id = Number(w[0]);
    if (!Number.isInteger(id) || id <= 0) throw new StyleError(`fix tmd: bad atom ID in ${file}: '${w[0]}'`);
    const p: [number, number, number] = [Number(w[1]), Number(w[2]), Number(w[3])];
    if (p.some((c) => !Number.isFinite(c))) throw new StyleError(`fix tmd: bad coordinates in ${file}: '${l}'`);
    if (withImages) {
      for (let d = 0; d < 3; d++) {
        const n = Number(w[4 + d]);
        if (!Number.isInteger(n)) throw new StyleError(`fix tmd: bad image flag in ${file}: '${l}'`);
        p[d] += n * box![d];
      }
    }
    targets.set(id, p);
  }
  return { targets, box };
};

export class FixTmd extends Fix {
  readonly style = 'tmd';
  private readonly rhoFinal: number;
  private readonly every: number;
  private readonly file2: string | null;
  private targets: Map<number, [number, number, number]>;
  /** group atoms (indices), their target coordinates, and positions at the start of the step */
  private members: number[] = [];
  private tgt: Float64Array = new Float64Array(0);
  /** masses of the group members and their sum */
  private mass: Float64Array = new Float64Array(0);
  private massSum = 0;
  /**
   * Output factor on lambda and work_analytical: mvv2e (real: 48.88821291^2 = 2390.06; measured on two mass ratios
   * and the peptide), 1 in lj units. Other unit systems are not measured and refused in the constructor.
   */
  private unitFactor = 1;
  private xold: Float64Array = new Float64Array(0);
  private rhoInit = 0;
  private rhoOld = 0;
  private rhoTarget = 0;
  private lambda = 0;
  private workLambda = 0;
  private workAnalytical = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4 && args.length !== 3) throw new StyleError('fix tmd: usage: fix ID group-ID tmd rho_final file1 N file2');
    this.rhoFinal = Number(args[0]);
    if (!Number.isFinite(this.rhoFinal)) throw new StyleError(`fix tmd: rho_final must be a number, got '${args[0]}'`);
    const fileName = args[1];
    this.every = Number(args[2]);
    if (!Number.isInteger(this.every) || this.every < 0) throw new StyleError(`fix tmd: N must be a non-negative integer, got '${args[2]}'`);
    // "file2 = filename to write TMD statistics to (only needed if N > 0)"
    if (this.every > 0 && args.length !== 4) throw new StyleError('fix tmd: file2 is needed when N > 0');
    this.file2 = this.every > 0 ? args[3] : null;
    this.nevery = 1;
    this.targets = parseTarget(sys.readFile(fileName), fileName).targets;
    // every group atom must be listed (the doc: "every atom in the group must be listed in the file")
    const s = sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const tag = s.id[i];
      if (!this.targets.has(tag)) throw new StyleError(`fix tmd: atom ${tag} of the group is not listed in ${fileName}`);
      this.members.push(i);
    }
    if (this.members.length === 0) throw new StyleError('fix tmd: group has no atoms');
    this.tgt = new Float64Array(3 * this.members.length);
    this.mass = Float64Array.from(this.members, (i) => massOf(s, i));
    this.massSum = this.mass.reduce((a, b) => a + b, 0);
    this.members.forEach((i, k) => {
      const p = this.targets.get(s.id[i])!;
      this.tgt[3 * k] = p[0]; this.tgt[3 * k + 1] = p[1]; this.tgt[3 * k + 2] = p[2];
    });
    if (this.file2 !== null) {
      const u = sys.units.style;
      if (u === 'lj') this.unitFactor = 1;
      else if (u === 'real') this.unitFactor = sys.units.mvv2e;
      else throw new StyleError(`fix tmd: the lambda and work columns of the N > 0 output are measured only in lj and real units (this is ${u})`);
    }
    if (this.file2 !== null) sys.writeFile!(this.file2, '# Step rho_target rho_old gamma_back gamma_forward lambda work_lambda work_analytical\n', false);
  }

  /** Unwrapped positions of the group members (3 per member). */
  private positions(): Float64Array {
    const s = this.sys.state;
    const out = new Float64Array(3 * this.members.length);
    const p = [0, 0, 0];
    this.members.forEach((i, k) => {
      this.sys.geom.unwrap(s.x, s.image, i, p);
      out[3 * k] = p[0]; out[3 * k + 1] = p[1]; out[3 * k + 2] = p[2];
    });
    return out;
  }

  /** rho over the group: sqrt(sum |x - y|^2 / N) with N the group count. */
  private rhoOf(x: Float64Array): number {
    let sum = 0;
    for (let k = 0; k < this.members.length; k++) {
      let d2 = 0;
      for (let c = 0; c < 3; c++) { const d = x[3 * k + c] - this.tgt[3 * k + c]; d2 += d * d; }
      sum += this.mass[k] * d2;
    }
    return Math.sqrt(sum / this.massSum);
  }

  setup(): void {
    this.xold = this.positions();
    this.rhoInit = this.rhoOf(this.xold);
    this.rhoOld = this.rhoInit;
  }

  endOfStep(): void {
    this.xold = this.positions();
  }

  postIntegrate(): void {
    const s = this.sys.state;
    const dt = s.dt;
    const nGroup = this.members.length;
    const xfree = this.positions();
    const rhoT = ramp(this.sys, this.rhoInit, this.rhoFinal);
    // d = x_old - y, e = x_free - y; the constraint sum |e + gamma d|^2 = N rho^2 gives a quadratic in gamma
    // the constraint is mass weighted: rho^2 = sum m |x - y|^2 / M; the correction itself is not (native, black box)
    let A = 0, B = 0, E2 = 0;
    for (let k = 0; k < 3 * nGroup; k++) {
      const mk = this.mass[k / 3 | 0];
      const d = this.xold[k] - this.tgt[k];
      const e = xfree[k] - this.tgt[k];
      A += mk * d * d;
      B += mk * e * d;
      E2 += mk * e * e;
    }
    B *= 2;
    const root = (c: number): number => {
      const disc = Math.sqrt(B * B - 4 * A * c);
      return (-B + disc) / (2 * A);
    };
    const N = this.massSum;
    const gf = root(E2 - N * rhoT * rhoT);
    const gb = root(E2 - N * this.rhoOld * this.rhoOld);
    // constrained positions and the half-step velocity (x_new - x_old)/dt
    for (let k = 0; k < nGroup; k++) {
      const i = this.members[k];
      for (let c = 0; c < 3; c++) {
        const xn = xfree[3 * k + c] + gf * (this.xold[3 * k + c] - this.tgt[3 * k + c]);
        // positions are stored wrapped: shift by the image offset of this atom
        const dx = xn - xfree[3 * k + c];
        s.x[3 * i + c] += dx;
        s.v[3 * i + c] = (xn - this.xold[3 * k + c]) / dt;
      }
    }
    this.lambda = (this.unitFactor * gb * N * this.rhoOld) / (dt * dt);
    const dRho = rhoT - this.rhoOld;
    this.workLambda += this.lambda * dRho;
    this.workAnalytical += (this.unitFactor * N * gf * this.rhoOld * dRho * (1 - (dRho / this.rhoOld) ** 2)) / (dt * dt);
    this.rhoTarget = rhoT;
    if (this.file2 !== null && s.step % this.every === 0) {
      this.sys.writeFile!(this.file2, [s.step, this.rhoTarget, this.rhoOld, gb, gf, this.lambda, this.workLambda, this.workAnalytical].map(fmt).join(' ') + '\n', true);
    }
    this.rhoOld = rhoT;
  }
}

/** %.17g, as the native output prints it (shortest exact digits are not needed; 17 significant digits). */
const fmt = (v: number): string => {
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  if (e < -4 || e >= 17) {
    const s = v.toExponential(16).replace(/\.?0+e/, 'e');
    return s.replace(/e([+-])(\d)$/, 'e$10$2');
  }
  return String(Number(v.toPrecision(17)));
};
