import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { buildAtomMap } from '../atoms';
import { dihedralGeometry } from '../force/bonded_util';
import type { BondedCompute } from '../force/types';
import { ramp } from './util';

/*
 * fix restrain — written from docs.lammps.org/fix_restrain.html
 * (plans/lammps-docs/fix_restrain.rst). Syntax quoted there:
 *   "fix ID group-ID restrain keyword args ..."
 * "keyword = *bond* or *lbound* or *angle* or *dihedral*". The group-ID is
 * ignored: "The group-ID specified by this fix is ignored."
 *
 * Coefficients ramp over the run: "Kstart,Kstop = restraint coefficients at
 * start/end of run (energy units)"; "r0stop = equilibrium bond distance at end
 * of run (optional)". The ramp is linear in the step inside the run, measured
 * with native LAMMPS (black box): a bond restraint with K from 10 to 30 over
 * run 100 has K = 10 + 0.2 step at every printed step (K inferred from the
 * thermo energy and the atom distance).
 *
 * bond: "E = K (r - r_0)^2" with "Note that the usual 1/2 factor is included
 * in K." lbound: the same form only when r < r0, else zero.
 * angle: "E = K (\theta - \theta_0)^2"; "\theta_0 is specified in degrees, but
 * LAMMPS converts it to radians internally". Measured with native LAMMPS
 * (black box): atoms 1-2-4 at 45 degrees with K = 1 and theta0 = 90 give
 * 0.61685 = (pi/4)^2.
 * dihedral: "E = K [ 1 + \cos (n \phi - d) ]" with "d (degrees) = \phi_0 + 180"
 * as the doc page writes it. Measured with native LAMMPS (black box): the
 * energy is K [1 + cos(n phi - phi0)] with phi the usual dihedral (cis = 0,
 * the convention of the engine's bonded_util.ts). Fits over six geometries
 * for mult 1, 2, 3 and phi0 = 0, 60 give A = K, cos and sin coefficients
 * matching cos(n phi - phi0) to 2e-8; mult 0 gives K (1 + cos phi0). The doc
 * sentence "Also note that the energy will be a minimum when the" (the doc goes on to say
 * the current dihedral angle is equal to phi0 at the minimum) is therefore not what native does for
 * this sign convention (native's minimum for mult 1 is at phi0 + 180); the
 * engine follows the measured form.
 *
 * Keyword parsing (measured with native LAMMPS as a black box): the optional
 * r0stop of bond/lbound is read whenever another argument follows, so a bond
 * line followed by another keyword fails with the native message Expected floating point
 * parameter instead of 'angle' (the doc's example
 * "fix holdem all restrain bond 45 48 2000.0 2000.0 2.75 dihedral 1 2 3 4 2000.0 2000.0 120.0"
 * fails the same way). The engine
 * throws the same StyleError. dihedral takes an optional "mult n" after phi0.
 *
 * Vector outputs. The doc: "This fix computes a global scalar and a global
 * vector of length 3," and "The vector values are the sum of contributions to the
 * following individual categories:" bond energy, angle energy, dihedral energy.
 * Measured with native LAMMPS (black box): f_ID[1] holds the bond energy,
 * f_ID[2] holds the lbound energy, and f_ID[3] is zero; angle and dihedral
 * energies appear in the scalar (total) but in none of the three vector slots
 * (checked with angle-only, dihedral-only, angle+dihedral and two-angle
 * restraints, 1 and 3 steps). The engine reports the same slots.
 *
 * "The scalar and vector values calculated by this fix are "extensive"."
 * fix_modify energy: "The default setting for this fix is" fix_modify energy no.
 * (fix_modify.html: "Energy yes will add a contribution to the potential
 * energy of the system".)
 *
 * Force: the doc says the restraint "is functionally similar to creating a bond
 * or angle or dihedral" so the forces are -grad E of the energies above, using
 * minimum-image geometry as the bonded styles do. Native atom-ID lookup needs
 * an atom map (native message: Fix restrain requires an atom map, see atom_modify; measured).
 * Minimization: the doc page does not say how the ramp applies during
 * minimize; the engine applies the same ramp through minPostForce (not
 * measured).
 */

const num = (w: string | undefined, what: string): number => {
  if (w === undefined || w.trim() === '') throw new StyleError(`fix restrain: missing ${what}`);
  const v = Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`fix restrain: expected a floating point parameter for ${what}, got '${w}'`);
  return v;
};

const KEYWORDS = ['bond', 'lbound', 'angle', 'dihedral'] as const;
type Keyword = (typeof KEYWORDS)[number];

interface Term {
  kind: Keyword;
  atoms: number[]; // atom IDs
  kstart: number;
  kstop: number;
  /** bond/lbound: r0 at start and end; angle/dihedral: phi0 (radians) in r0s. */
  r0start: number;
  r0stop: number;
  mult: number;
}

export class FixRestrain extends Fix {
  readonly style = 'restrain';
  private readonly terms: Term[] = [];
  private evec = new Float64Array(3);
  private etotal = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.energyGlobal = true;
    let k = 0;
    while (k < args.length) {
      const kw = args[k] as Keyword;
      if (!KEYWORDS.includes(kw)) throw new StyleError(`fix restrain: unknown keyword '${args[k]}' (bond, lbound, angle or dihedral)`);
      k++;
      if (kw === 'bond' || kw === 'lbound') {
        if (args.length - k < 5) throw new StyleError(`fix restrain ${kw} needs atom1 atom2 Kstart Kstop r0start (r0stop)`);
        const atoms = [num(args[k], 'atom1'), num(args[k + 1], 'atom2')];
        const kstart = num(args[k + 2], 'Kstart'), kstop = num(args[k + 3], 'Kstop');
        const r0start = num(args[k + 4], 'r0start');
        k += 5;
        let r0stop = r0start;
        // the optional r0stop is read whenever another argument follows (native behaviour)
        if (k < args.length) {
          r0stop = num(args[k], 'r0stop');
          k++;
        }
        this.terms.push({ kind: kw, atoms, kstart, kstop, r0start, r0stop, mult: 1 });
      } else if (kw === 'angle') {
        if (args.length - k < 6) throw new StyleError('fix restrain angle needs atom1 atom2 atom3 Kstart Kstop theta0');
        const atoms = [num(args[k], 'atom1'), num(args[k + 1], 'atom2'), num(args[k + 2], 'atom3')];
        const kstart = num(args[k + 3], 'Kstart'), kstop = num(args[k + 4], 'Kstop');
        const theta0 = (num(args[k + 5], 'theta0') * Math.PI) / 180;
        k += 6;
        this.terms.push({ kind: kw, atoms, kstart, kstop, r0start: theta0, r0stop: theta0, mult: 1 });
      } else {
        if (args.length - k < 7) throw new StyleError('fix restrain dihedral needs atom1 atom2 atom3 atom4 Kstart Kstop phi0');
        const atoms = [0, 1, 2, 3].map((o) => num(args[k + o], `atom${o + 1}`));
        const kstart = num(args[k + 4], 'Kstart'), kstop = num(args[k + 5], 'Kstop');
        const phi0 = (num(args[k + 6], 'phi0') * Math.PI) / 180;
        k += 7;
        let mult = 1;
        if (k < args.length && args[k] === 'mult') {
          if (k + 1 >= args.length) throw new StyleError('fix restrain dihedral: mult needs a value');
          mult = num(args[k + 1], 'mult');
          if (!Number.isInteger(mult) || mult < 0) throw new StyleError('fix restrain dihedral: mult must be an integer >= 0');
          k += 2;
        }
        this.terms.push({ kind: kw, atoms, kstart, kstop, r0start: phi0, r0stop: phi0, mult });
      }
    }
    if (!this.terms.length) throw new StyleError('fix restrain needs at least one of bond, lbound, angle, dihedral');
    for (const t of this.terms) {
      if (t.atoms.some((a) => !Number.isInteger(a) || a <= 0)) throw new StyleError(`fix restrain ${t.kind}: atom IDs must be positive integers`);
      if (t.kind === 'bond' || t.kind === 'lbound') {
        if (t.atoms[0] === t.atoms[1]) throw new StyleError(`fix restrain ${t.kind}: the two atoms must differ`);
      }
    }
  }

  /** K and r0 (or theta0 / phi0 in radians) at the current step of the run. */
  private current(t: Term): { k: number; r0: number } {
    return { k: ramp(this.sys, t.kstart, t.kstop), r0: ramp(this.sys, t.r0start, t.r0stop) };
  }

  postForce(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const map = buildAtomMap(s);
    const idx = (id: number, kind: string): number => {
      const i = id < map.length ? map[id] : -1;
      if (i < 0) throw new StyleError(`fix restrain ${kind}: atom ${id} missing`);
      return i;
    };
    const f = s.f, x = s.x;
    const d = [0, 0, 0];
    this.evec.fill(0);
    let etot = 0;
    for (const t of this.terms) {
      const { k, r0 } = this.current(t);
      if (t.kind === 'bond' || t.kind === 'lbound') {
        const i = idx(t.atoms[0], t.kind), j = idx(t.atoms[1], t.kind);
        for (let c = 0; c < 3; c++) d[c] = x[3 * j + c] - x[3 * i + c];
        g.minimumImage(d);
        const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (t.kind === 'lbound' && r >= r0) continue;
        const dr = r - r0;
        const e = k * dr * dr;
        etot += e;
        this.evec[t.kind === 'bond' ? 0 : 1] += e;
        if (r > 0) {
          // E = K dr^2: dE/dx_j = 2 K dr d / r; the force on j is -dE/dx_j
          const fac = (-2 * k * dr) / r;
          for (let c = 0; c < 3; c++) {
            f[3 * j + c] += fac * d[c];
            f[3 * i + c] -= fac * d[c];
          }
        }
      } else if (t.kind === 'angle') {
        const i = idx(t.atoms[0], 'angle'), j = idx(t.atoms[1], 'angle'), kk = idx(t.atoms[2], 'angle');
        const u = [0, 0, 0], v = [0, 0, 0];
        for (let c = 0; c < 3; c++) { u[c] = x[3 * i + c] - x[3 * j + c]; v[c] = x[3 * kk + c] - x[3 * j + c]; }
        g.minimumImage(u);
        g.minimumImage(v);
        const ru = Math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        const rv = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (ru === 0 || rv === 0) continue;
        let cth = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (ru * rv);
        if (cth > 1) cth = 1;
        if (cth < -1) cth = -1;
        const theta = Math.acos(cth);
        const dth = theta - r0;
        etot += k * dth * dth;
        let sn = Math.sqrt(1 - cth * cth);
        if (sn < 1e-3) sn = 1e-3;
        // dE/dtheta = 2 K dth; dtheta/dc = -1/sin; F_i = -dE/dc dc/dx_i
        const dEdc = (-2 * k * dth) / sn;
        for (let c = 0; c < 3; c++) {
          const fi = -dEdc * (v[c] / (ru * rv) - cth * u[c] / (ru * ru));
          const fk = -dEdc * (u[c] / (ru * rv) - cth * v[c] / (rv * rv));
          f[3 * i + c] += fi;
          f[3 * kk + c] += fk;
          f[3 * j + c] -= fi + fk;
        }
      } else {
        // dihedral: geometry from the engine's bonded helpers (import only)
        const atoms = t.atoms.map((id, o) => idx(id, `dihedral atom ${o + 1}`));
        const shim = { s, geom: g } as unknown as BondedCompute;
        const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
        const phi = dihedralGeometry(shim, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
        const n = t.mult;
        const e = k * (1 + Math.cos(n * phi - r0));
        etot += e;
        // dE/dphi = -K n sin(n phi - phi0); F = -dE/dphi grad
        const dEdphi = -k * n * Math.sin(n * phi - r0);
        for (let a = 0; a < 4; a++) {
          for (let c = 0; c < 3; c++) f[3 * atoms[a] + c] += -dEdphi * grad[3 * a + c];
        }
      }
    }
    this.etotal = etot;
  }

  minPostForce(): void { this.postForce(); }

  energy(): number { return this.etotal; }
  computeScalar(): number { return this.etotal; }
  /** The vector slots follow the measured native layout: [bond, lbound, 0]. */
  computeVector(i: number): number {
    if (i < 0 || i > 2) throw new StyleError(`fix restrain vector index ${i + 1} out of range`);
    return i === 2 ? 0 : this.evec[i];
  }
}
