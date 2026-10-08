import { hasChargeStyle } from '../atoms';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNumOrVar, valueOf, type NumOrVar } from './util';
import { compileLepton, fillVrefs, type LeptonProgram } from '../lepton';
import { parseLepton } from '../lepton/parse';
import { resolveNames, dependsOn, diff } from '../lepton/resolve';

/*
 * fix wall/lepton (docs.lammps.org/fix_wall.html, style wall/lepton) and
 * fix efield/lepton (docs.lammps.org/fix_efield_lepton.html).
 *
 * wall/lepton: args are coord, expression and cutoff. The energy E is the Lepton
 * expression in r (the distance to the wall); the expression may use rc, the
 * cutoff of that wall, and definitions after a semicolon. The energy is shifted so
 * that it is 0 at the cutoff, as for the other wall styles (measured with native
 * LAMMPS, black box: r^2 at x = 1 with cutoff 4 gives -15). The force is -dE/dr.
 * Native LAMMPS aborts with an unknown-variable error when the derivative does
 * not depend on r (e.g. the expression r or 2*r); this engine evaluates those
 * exactly instead.
 *
 * efield/lepton: the expression V is in x, y, z (unwrapped coordinates). Charged
 * atoms get F = q E with E = -grad V; the energy is q V (fix_modify energy), and
 * the virial is that of the added forces. The keywords are region and step.
 *
 * Point dipoles (docs.lammps.org/fix_efield_lepton.html): the page gives
 * F = q E on charges and F = (p . grad) E, T = p x E on dipoles with E = -grad V.
 * "It creates an analytical representation of :math:`V` and :math:`\vec{E}`, while the gradient force
 * is computed using a central difference scheme", so the dipole force is
 * |p|/(2h) [ E(x + h p) - E(x - h p) ] with E the analytic gradient.
 * "The step keyword is required when atom_style dipole is used and the electric
 * field is non-uniform." Measured with native LAMMPS (black box): atom_style
 * dipole without the sphere sub-style aborts with Dipoles must be finite-sized
 * to rotate; the torque and the energy use the analytic E at the atom; the force
 * and the energy carry the same qe2f factor as fix efield; and the virial of the
 * added forces is sum f (x) r (off-diagonals f_x r_y, f_x r_z, f_y r_z) in
 * unwrapped coordinates.
 *
 * The wall logic follows fix/walls.ts (copied, since that class is not exported),
 * except for the virial sign (see apply()).
 */

const FACES = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'] as const;
type Face = (typeof FACES)[number];
const isFace = (w: string): w is Face => (FACES as readonly string[]).includes(w);

interface LWall {
  dim: 0 | 1 | 2;
  lo: boolean;
  coord: { edge: true } | { edge: false; v: NumOrVar };
  prog: LeptonProgram;
  env: Float64Array;
  cutoff: number;
}

export class FixWallLepton extends Fix {
  readonly style = 'wall/lepton';
  private readonly walls: LWall[] = [];
  private unitsMode: 'lattice' | 'box' = 'lattice';
  private fld = false;
  private pbcOk = false;
  private wallForce = new Float64Array(0);
  private wallEnergy = 0;
  private vAtom = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError(`usage: fix ID group-ID wall/lepton face args ... [face args ...] [keyword value ...]`);
    for (let k = 0; k < args.length;) {
      const w = args[k];
      if (isFace(w)) {
        if (this.walls.some((x) => FACES[x.dim * 2 + (x.lo ? 0 : 1)] === w)) {
          throw new StyleError(`fix ${id} (wall/lepton): face ${w} is specified more than once`);
        }
        const words = args.slice(k + 1, k + 4);
        if (words.length < 3) throw new StyleError(`fix ${id} (wall/lepton): face ${w} needs coord, expression and cutoff`);
        this.walls.push(this.parseWall(id, w, words));
        k += 4;
      } else if (w === 'units') {
        const v = args[k + 1];
        if (v !== 'lattice' && v !== 'box') throw new StyleError(`fix ${id} (wall/lepton): units must be lattice or box, got '${v ?? ''}'`);
        this.unitsMode = v;
        k += 2;
      } else if (w === 'fld' || w === 'pbc') {
        const v = args[k + 1];
        if (v !== 'yes' && v !== 'no') throw new StyleError(`fix ${id} (wall/lepton): ${w} value must be yes or no, got '${v ?? ''}'`);
        if (w === 'fld') this.fld = v === 'yes'; else this.pbcOk = v === 'yes';
        k += 2;
      } else {
        throw new StyleError(`fix ${id} (wall/lepton): unknown argument '${w}' (expected a face xlo..zhi or the keyword units, fld or pbc)`);
      }
    }
    if (!this.walls.length) throw new StyleError(`fix ${id} (wall/lepton): no wall face (xlo, xhi, ylo, yhi, zlo or zhi) was specified`);
    this.wallForce = new Float64Array(this.walls.length);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = this.walls.length;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
    this.virialGlobal = true;
  }

  private parseWall(id: string, face: Face, words: string[]): LWall {
    const dim = 'xyz'.indexOf(face[0]) as 0 | 1 | 2;
    const lo = face[1] === 'l';
    const coordW = words[0];
    const coord: LWall['coord'] = coordW === 'EDGE'
      ? { edge: true }
      : { edge: false, v: parseNumOrVar(coordW, `${face} wall coord`) };
    if (!coord.edge && typeof coord.v === 'object' && !this.sys.vars.has(coord.v.variable)) {
      throw new StyleError(`fix ${id} (wall/lepton): variable ${coord.v.variable} does not exist`);
    }
    const cutW = words[2];
    const cutoff = Number(cutW);
    if (cutW.startsWith('v_') || !Number.isFinite(cutoff) || !(cutoff > 0)) {
      throw new StyleError(`fix ${id} (wall/lepton): cutoff must be a number > 0 (distance units), got '${cutW}'`);
    }
    const prog = compileLepton(words[1], { builtins: ['r', 'rc'], wrt: ['r'] });
    for (const name of prog.vrefs) {
      if (!this.sys.vars.has(name)) throw new StyleError(`fix ${id} (wall/lepton): variable ${name} does not exist`);
    }
    const env = new Float64Array(prog.builtins.length + prog.vrefs.length);
    env[1] = cutoff;
    return { dim, lo, coord, prog, env, cutoff };
  }

  init(): void {
    const s = this.sys.state;
    if (s.dimension === 2 && this.walls.some((w) => w.dim === 2)) {
      throw new StyleError(`fix ${this.id} (wall/lepton): cannot use a z wall (zlo/zhi) in a 2d simulation`);
    }
    for (const w of this.walls) {
      if (!this.pbcOk && s.box.periodic[w.dim]) {
        throw new StyleError(`fix ${this.id} (wall/lepton): wall in a periodic dimension requires pbc yes (the default is pbc no)`);
      }
    }
  }

  preForce(): void { this.apply(); }
  postForce(): void { this.apply(); }
  setup(): void { if (this.fld) this.preForce(); else this.postForce(); }
  minPostForce(): void { this.apply(); }

  private coords(): Float64Array {
    const g = this.sys.geom;
    const sp = this.sys.lattice?.spacing;
    const out = new Float64Array(this.walls.length);
    for (let k = 0; k < this.walls.length; k++) {
      const w = this.walls[k];
      out[k] = w.coord.edge
        ? (w.lo ? g.lo[w.dim] : g.hi[w.dim])
        : valueOf(this.sys, w.coord.v) * (this.unitsMode === 'lattice' ? sp?.[w.dim] ?? 1 : 1);
    }
    return out;
  }

  private apply(): void {
    const s = this.sys.state;
    const coord = this.coords();
    const bit = this.groupBit;
    const walls = this.walls;
    this.wallEnergy = 0;
    this.wallForce.fill(0);
    if (this.thermoVirial) this.virial.fill(0);
    if (this.vAtom.length !== 6 * s.n) this.vAtom = new Float64Array(6 * s.n);
    else this.vAtom.fill(0);
    // equal-style variable slots (v_name) are refreshed once per application
    for (const w of walls) fillVrefs(w.env, w.prog, (n) => this.sys.equalVariable(n));
    const eCut = new Float64Array(walls.length);
    for (let k = 0; k < walls.length; k++) {
      const w = walls[k];
      w.env[0] = w.cutoff;
      eCut[k] = w.prog.value(w.env);
    }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & bit)) continue;
      for (let k = 0; k < walls.length; k++) {
        const w = walls[k];
        const d = w.dim;
        const r = w.lo ? s.x[3 * i + d] - coord[k] : coord[k] - s.x[3 * i + d];
        if (r <= 0) throw new StyleError(`Particle on or inside fix ${this.id} wall/lepton surface`);
        if (r >= w.cutoff) continue;
        w.env[0] = r;
        const dEdr = w.prog.deriv[0](w.env);
        const e = w.prog.value(w.env) - eCut[k];
        // F = -dE/dr, pointing away from the wall; sign flips the axis for hi walls
        const f = -dEdr;
        const sign = w.lo ? 1 : -1;
        s.f[3 * i + d] += sign * f;
        this.wallEnergy += e;
        this.wallForce[k] -= sign * f;
        // Measured with native LAMMPS (black box): for the same forces the wall/lepton
        // pressure contribution is the negative of wall/harmonic's (r f tallied with the
        // opposite sign), so the virial here is -r f.
        if (this.thermoVirial) this.virial[d] -= r * f;
        this.vAtom[6 * i + d] -= r * f;
      }
    }
  }

  energy(): number { return this.wallEnergy; }
  virialAtom(out: Float64Array): void {
    for (let k = 0; k < this.vAtom.length && k < out.length; k++) out[k] += this.vAtom[k];
  }
  computeScalar(): number { return this.wallEnergy; }
  computeVector(i: number): number { return this.wallForce[i]; }
}

/**
 * True if E = -grad V depends on x, y or z, i.e. the field is non-uniform.
 * The page requires the step keyword in that case ("The step keyword is
 * required when atom_style dipole is used and the electric field is
 * non-uniform."). The rule was measured with native LAMMPS (black box): it
 * rejects -x^2, -x*y and -abs(x) without step, while -x, a tumbling uniform
 * field -(x cos + y sin) and a constant are accepted.
 */
const fieldIsNonUniform = (text: string): boolean => {
  const tree = resolveNames(parseLepton(text, { zbl: false }), new Set(['x', 'y', 'z']));
  for (const c of ['x', 'y', 'z']) {
    const d = diff(tree, c);
    for (const q of ['x', 'y', 'z']) if (dependsOn(d, q)) return true;
  }
  return false;
};

/**
 * fix efield/lepton V [region ID] [step h] — docs.lammps.org/fix_efield_lepton.html.
 * Charges get F = q E; dipoles get the central-difference gradient force
 * F = |p|/(2h) [E(x + h p) - E(x - h p)], the analytic torque T = p x E and the
 * energy -p . E, all with the qe2f factor fix efield uses (measured, see the
 * file comment).
 */
export class FixEfieldLepton extends Fix {
  readonly style = 'efield/lepton';
  private prog: LeptonProgram;
  private regionId: string | null = null;
  /** step h: the central-difference displacement for the dipole force (distance units). */
  private stepH = 0;
  /** Point dipoles present (atom_style dipole): also apply torque and grad E force. */
  private dipoles = false;
  private fadd: Float64Array;
  private pot: Float64Array;
  /** grad V at the atom, for the dipole energy -p . E = p . grad V. */
  private gx: Float64Array;
  private gy: Float64Array;
  private gz: Float64Array;
  private unwrapped = [0, 0, 0];
  private envBuf: Float64Array;
  private envP: Float64Array;
  private envM: Float64Array;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const s = sys.state;
    if (!hasChargeStyle(s.atomStyle)) {
      throw new StyleError(`fix efield/lepton requires atoms that store a charge; atom_style ${s.atomStyle} does not`);
    }
    if (s.box.triclinic) throw new StyleError(`fix ${id} efield/lepton: triclinic boxes are not supported`);
    if (args.length < 1) throw new StyleError('usage: fix ID group-ID efield/lepton V [region ID] [step h]');
    this.prog = compileLepton(args[0], { builtins: ['x', 'y', 'z'], wrt: ['x', 'y', 'z'] });
    for (const name of this.prog.vrefs) {
      if (!sys.vars.has(name)) throw new StyleError(`fix ${id} efield/lepton: variable ${name} does not exist`);
    }
    for (let k = 1; k < args.length; k += 2) {
      const key = args[k];
      const val = args[k + 1];
      if (val === undefined) throw new StyleError(`fix efield/lepton keyword '${key}' needs a value`);
      if (key === 'region') {
        sys.region(val);
        this.regionId = val;
      } else if (key === 'step') {
        const h = Number(val);
        if (!(h > 0)) throw new StyleError(`fix efield/lepton: step must be > 0, got '${val}'`);
        this.stepH = h;
      } else {
        throw new StyleError(`fix efield/lepton: unknown keyword '${key}' (expected region or step)`);
      }
    }
    if (s.mu) {
      // Measured with native LAMMPS (black box): atom_style dipole without the sphere
      // sub-style is rejected (Dipoles must be finite-sized to rotate); hybrid sphere dipole works.
      if (!s.radius || !s.torque) {
        throw new StyleError(`fix ${id} efield/lepton: point dipoles require finite-size particles (use atom_style hybrid sphere dipole)`);
      }
      this.dipoles = true;
      if (this.stepH <= 0 && fieldIsNonUniform(args[0])) {
        throw new StyleError(`fix ${id} efield/lepton requires keyword 'step' for dipoles in a non-uniform electric field`);
      }
    }
    // The page says the fix "computes a global scalar and a global 3-vector of forces"; a native
    // build (2 Sep 2026) rejects f_ID[i] for this fix (does not compute the requested property),
    // measured in a variable formula, so only the scalar is exposed here.
    this.scalarFlag = true;
    this.extscalar = 1;
    this.energyGlobal = true;
    this.virialGlobal = true;
    this.fadd = new Float64Array(3 * s.n);
    this.pot = new Float64Array(s.n);
    this.gx = new Float64Array(s.n);
    this.gy = new Float64Array(s.n);
    this.gz = new Float64Array(s.n);
    const nv = this.prog.builtins.length + this.prog.vrefs.length;
    this.envBuf = new Float64Array(nv);
    this.envP = new Float64Array(nv);
    this.envM = new Float64Array(nv);
  }

  /** Fills the env (unwrapped x, y, z of atom i; then the v_name values). */
  private env(i: number): Float64Array {
    const s = this.sys.state;
    this.sys.geom.unwrap(s.x, s.image, i, this.unwrapped);
    const e = this.envBuf;
    e[0] = this.unwrapped[0];
    e[1] = this.unwrapped[1];
    e[2] = this.unwrapped[2];
    return e;
  }

  private inRegion(i: number): boolean {
    if (this.regionId === null) return true;
    const s = this.sys.state;
    return this.sys.region(this.regionId).match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
  }

  postForce(): void {
    const s = this.sys.state;
    const k = s.units.qe2f / s.units.ftm2v;
    const kt = s.units.qe2f;
    const { f, q, mask } = s;
    const bit = this.groupBit;
    const mu = s.mu, tq = s.torque;
    const useDip = this.dipoles && !!mu && !!tq;
    const h = this.stepH;
    const nb = this.prog.builtins.length;
    this.fadd.fill(0);
    this.gx.fill(0); this.gy.fill(0); this.gz.fill(0);
    // v_name slots are refreshed once per force evaluation
    const vals = new Float64Array(this.prog.vrefs.length);
    this.prog.vrefs.forEach((name, m) => { vals[m] = this.sys.equalVariable(name); });
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit) || !this.inRegion(i)) continue;
      const env = this.env(i);
      env.set(vals, nb);
      const dx = this.prog.deriv[0](env), dy = this.prog.deriv[1](env), dz = this.prog.deriv[2](env);
      // F = -q grad V (times the units factor)
      const fx = -q[i] * dx * k, fy = -q[i] * dy * k, fz = -q[i] * dz * k;
      this.fadd[3 * i] += fx; this.fadd[3 * i + 1] += fy; this.fadd[3 * i + 2] += fz;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      this.pot[i] = this.prog.value(env);
      if (!useDip) continue;
      const px = mu![4 * i], py = mu![4 * i + 1], pz = mu![4 * i + 2], len = mu![4 * i + 3];
      this.gx[i] = dx; this.gy[i] = dy; this.gz[i] = dz;
      // Torque T = p x E with E = -grad V (analytic, measured with native LAMMPS)
      tq![3 * i] += kt * (pz * dy - py * dz);
      tq![3 * i + 1] += kt * (px * dz - pz * dx);
      tq![3 * i + 2] += kt * (py * dx - px * dy);
      if (!(len > 0) || !(h > 0)) continue;
      // F = |p|/(2h) [E(x + h p_hat) - E(x - h p_hat)] with E = -grad V (central difference)
      const ox = (h * px) / len, oy = (h * py) / len, oz = (h * pz) / len;
      const ep = this.envP, em = this.envM;
      ep[0] = this.unwrapped[0] + ox; ep[1] = this.unwrapped[1] + oy; ep[2] = this.unwrapped[2] + oz;
      em[0] = this.unwrapped[0] - ox; em[1] = this.unwrapped[1] - oy; em[2] = this.unwrapped[2] - oz;
      ep.set(vals, nb); em.set(vals, nb);
      const c = (len / (2 * h)) * kt / s.units.ftm2v;
      const gx2 = c * (-this.prog.deriv[0](ep) + this.prog.deriv[0](em));
      const gy2 = c * (-this.prog.deriv[1](ep) + this.prog.deriv[1](em));
      const gz2 = c * (-this.prog.deriv[2](ep) + this.prog.deriv[2](em));
      this.fadd[3 * i] += gx2; this.fadd[3 * i + 1] += gy2; this.fadd[3 * i + 2] += gz2;
      f[3 * i] += gx2; f[3 * i + 1] += gy2; f[3 * i + 2] += gz2;
    }
    if (this.thermoVirial) this.tallyVirial();
  }

  /**
   * Virial of the added forces: sum f (x) r (the transpose of the pair r (x) f
   * convention), with the unwrapped position (the fix evaluates V and its
   * gradient in unwrapped coordinates). Measured with native LAMMPS (black box):
   * a charge at unwrapped x = -0.5 (wrapped 9.5, image -1) under V = -x gives
   * Pxx = -0.5/V, i.e. the unwrapped coordinate, and pxy = f_x r_y.
   */
  private tallyVirial(): void {
    const s = this.sys.state;
    const v = this.virial;
    v.fill(0);
    const u = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      const fx = this.fadd[3 * i], fy = this.fadd[3 * i + 1], fz = this.fadd[3 * i + 2];
      if (fx === 0 && fy === 0 && fz === 0) continue;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const px = u[0], py = u[1], pz = u[2];
      v[0] += fx * px; v[1] += fy * py; v[2] += fz * pz;
      v[3] += fx * py; v[4] += fx * pz; v[5] += fy * pz;
    }
  }

  /** U = sum q V + sum (-p . E) (energy units), for fix_modify energy yes. */
  energy(): number {
    const s = this.sys.state;
    const k = s.units.qe2f;
    const mu = s.mu;
    let e = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit) || !this.inRegion(i)) continue;
      e += s.q[i] * this.pot[i] * k;
      // -p . E = p . grad V (measured with native LAMMPS, black box)
      if (this.dipoles && mu) e += (mu[4 * i] * this.gx[i] + mu[4 * i + 1] * this.gy[i] + mu[4 * i + 2] * this.gz[i]) * k;
    }
    return e;
  }

  computeScalar(): number { return this.energy(); }
}
