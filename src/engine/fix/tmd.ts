import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { rampFraction } from './util';
import { fmtShortest } from '../cformat';
import { inumeric, numeric } from '../force/util';
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
 * The step, as native LAMMPS does (src/EXTRA-FIX/fix_tmd.cpp at b332f28): the fix keeps x_old, the unwrapped
 * positions of the group, from its definition (the constructor, which also sets rho_start = rho(x_old) and rho_old =
 * rho_start; neither is reset by a later run) and after each constraint. In initial_integrate (here postIntegrate, after
 * every integrator; native requires tmd after them), with d = x_old - y and e = unwrapped x - y:
 *   rho_target = rho_start + delta (rho_final - rho_start), delta = (step - beginstep) / (endstep - beginstep);
 *   a = sum m d.d / M, b = 2 sum m e.d / M, e2 = sum m e.e / M; for rho in (rho_old, rho_target) the root
 *   gamma = c / (a gamma_max), c = e2 - rho^2, gamma_max = (-b -/+ sqrt(b^2 - 4 a c)) / (2 a) by the sign of b (a
 *   negative discriminant counts as 0, a = 0 gives 0): gamma_back for rho_old, gamma_forward for rho_target;
 *   x += gamma_forward d, v += gamma_forward d / dt and f += gamma_forward d m / (dt dtf) (dtf = dt ftm2v), then
 *   x_old = unwrap(x).
 * Output (N > 0): fr = sum f.d (the forces at that point, those of the previous step) and kt = sum m |x + gamma_back d
 * - x_old|^2; work_analytical += (-fr - kt / (dt dtf)) (rho_target - rho_old) / rho_old, lambda = gamma_back rho_old M /
 * (dt dtf), work_lambda += lambda (rho_target - rho_old); a row every N steps (once per step) with {fmt}'s "{}" numbers.
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
  private xold: Float64Array = new Float64Array(0);
  private rhoStart = 0;
  /** The step of the last output row (native previous_stat). */
  private previousStat = 0;
  private rhoOld = 0;
  private rhoTarget = 0;
  private lambda = 0;
  private workLambda = 0;
  private workAnalytical = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/EXTRA-FIX/fix_tmd.cpp FixTMD::FixTMD at b332f28): fewer than the three required words is
    // "Illegal fix tmd command"; with N = 0 any words after N are not read.
    if (args.length < 3) throw new StyleError('Illegal fix tmd command');
    // As native LAMMPS does (src/EXTRA-FIX/fix_tmd.cpp FixTMD::FixTMD at b332f28): rho_final is read with utils::numeric
    // and N with utils::inumeric; a negative value of either is "Illegal fix tmd command".
    this.rhoFinal = numeric(args[0]);
    const fileName = args[1];
    this.every = inumeric(args[2]);
    if (this.rhoFinal < 0 || this.every < 0) throw new StyleError('Illegal fix tmd command');
    // "file2 = filename to write TMD statistics to (only needed if N > 0)"
    // native: "if (nfileevery && narg != 7)": the file2 word is required (and only then) when N > 0
    if (this.every > 0 && args.length !== 4) throw new StyleError('Illegal fix tmd command');
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
    if (this.file2 !== null) sys.writeFile!(this.file2, '# Step rho_target rho_old gamma_back gamma_forward lambda work_lambda work_analytical\n', false);
    // x_old, rho_start and rho_old from the positions at the definition (native constructor)
    this.xold = this.positions();
    this.rhoStart = this.rhoOf(this.xold);
    this.rhoOld = this.rhoStart;
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

  postIntegrate(): void {
    const s = this.sys.state;
    const dtv = s.dt, dtf = s.dt * s.units.ftm2v;
    const nGroup = this.members.length;
    const M = this.massSum;
    const x = this.positions();
    const rhoT = this.rhoStart + rampFraction(this.sys) * (this.rhoFinal - this.rhoStart);
    let a = 0, b = 0, e = 0;
    for (let k = 0; k < nGroup; k++) {
      let dd = 0, ed = 0, ee = 0;
      for (let c = 0; c < 3; c++) {
        const dOld = this.xold[3 * k + c] - this.tgt[3 * k + c];
        const dNow = x[3 * k + c] - this.tgt[3 * k + c];
        dd += dOld * dOld; ed += dNow * dOld; ee += dNow * dNow;
      }
      a += this.mass[k] * dd; b += this.mass[k] * ed; e += this.mass[k] * ee;
    }
    a /= M; b = (2.0 * b) / M; e /= M;
    const root = (rho: number): number => {
      const c = e - rho * rho;
      let d = b * b - 4 * a * c;
      if (d < 0) d = 0;
      const gmax = b >= 0 ? (-b - Math.sqrt(d)) / (2 * a) : (-b + Math.sqrt(d)) / (2 * a);
      return a === 0.0 ? 0 : c / (a * gmax);
    };
    const gb = root(this.rhoOld);
    const gf = root(rhoT);
    if (this.file2 !== null) {
      let fr = 0, kt = 0;
      for (let k = 0; k < nGroup; k++) {
        const i = this.members[k];
        let k2 = 0;
        for (let c = 0; c < 3; c++) {
          const dOld = this.xold[3 * k + c] - this.tgt[3 * k + c];
          const dkt = x[3 * k + c] + gb * dOld - this.xold[3 * k + c];
          k2 += dkt * dkt;
          fr += s.f[3 * i + c] * dOld;
        }
        kt += this.mass[k] * k2;
      }
      this.workAnalytical += ((-fr - kt / dtv / dtf) * (rhoT - this.rhoOld)) / this.rhoOld;
      this.lambda = (gb * this.rhoOld * M) / dtv / dtf;
      this.workLambda += this.lambda * (rhoT - this.rhoOld);
      if (s.step % this.every === 0 && this.previousStat !== s.step) {
        const row = [this.rhoTarget = rhoT, this.rhoOld, gb, gf, this.lambda, this.workLambda, this.workAnalytical];
        this.sys.writeFile!(this.file2, [String(s.step), ...row.map((v) => fmtShortest(v, false))].join(' ') + '\n', true);
        this.previousStat = s.step;
      }
    }
    this.rhoTarget = rhoT;
    this.rhoOld = rhoT;
    // the constraint, and x_old for the next step
    for (let k = 0; k < nGroup; k++) {
      const i = this.members[k];
      const dtfm = dtf / this.mass[k];
      for (let c = 0; c < 3; c++) {
        const dOld = this.xold[3 * k + c] - this.tgt[3 * k + c];
        s.x[3 * i + c] += gf * dOld;
        s.v[3 * i + c] += (gf * dOld) / dtv;
        s.f[3 * i + c] += (gf * dOld) / dtv / dtfm;
      }
    }
    this.xold = this.positions();
  }
}
