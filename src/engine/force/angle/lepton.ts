import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';
import { atomIndex } from '../bonded_util';
import { angleTerm } from './harmonic';
import { parseNum } from '../util';
import { compileLepton, fillVrefs, type LeptonProgram } from '../../lepton';

/*
 * angle_style lepton [auto_offset|no_offset] — docs.lammps.org/angle_lepton.html.
 * The expression uses theta, the angle relative to the reference angle theta_0
 * (an angle coefficient, in degrees): theta = theta_i - theta_0 in radians, so
 * the expression must assume radians. auto_offset (the default) shifts the
 * energy so that it is 0.0 at theta = theta_0; no_offset does not.
 * angle_coeff N theta0 "expression" (theta0 in degrees).
 * Measured with native LAMMPS (black box): angle_coeff 1 90.0 "theta" gives
 * -0.5236 for an angle of 60 degrees (theta = theta_i - theta_0 in radians),
 * and angle_coeff 1 60.0 "theta+1" with no_offset gives 1 at 60 degrees.
 * The force uses the exact dE/dtheta; dtheta/dc = -1/sin(theta) with sin(theta)
 * floored at 1e-10 for exactly collinear atoms.
 */

export class AngleLepton extends Bonded {
  readonly name = 'lepton';
  readonly kind = 'angle' as const;
  private autoOffset = true;
  private th0 = new Float64Array(0);
  private set = new Uint8Array(0);
  private progs: (LeptonProgram | null)[] = [];
  private env: Float64Array[] = [];
  private offset = new Float64Array(0);
  private ctx: StyleContext | null = null;

  settings(args: string[]): void {
    if (args.length > 1) throw new StyleError('usage: angle_style lepton [auto_offset|no_offset]');
    if (args.length === 1) {
      if (args[0] === 'auto_offset') this.autoOffset = true;
      else if (args[0] === 'no_offset') this.autoOffset = false;
      else throw new StyleError(`angle_style lepton: unknown argument '${args[0]}' (expected auto_offset or no_offset)`);
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.th0 = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
    this.progs = new Array(ntypes + 1).fill(null);
    this.env = new Array(ntypes + 1).fill(null);
    this.offset = new Float64Array(ntypes + 1);
  }

  coeff(args: string[], ctx?: StyleContext): void {
    if (ctx) this.ctx = ctx;
    if (args.length !== 3) throw new StyleError('usage: angle_coeff N theta0 "expression"');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const th0 = (parseNum(args[1], 'theta0') * Math.PI) / 180;
    const prog = compileLepton(args[2], { builtins: ['theta'], wrt: ['theta'] });
    for (let t = lo; t <= hi; t++) {
      this.th0[t] = th0;
      this.set[t] = 1;
      this.progs[t] = prog;
      this.env[t] = new Float64Array(prog.builtins.length + prog.vrefs.length);
    }
  }

  init(ctx?: StyleContext): void {
    if (ctx) this.ctx = ctx;
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`all angle coeffs are not set (type ${t})`);
    }
  }

  private prepare(): void {
    const fn = (this.ctx as { equalVariable?: (n: string) => number } | null)?.equalVariable;
    for (let t = 1; t <= this.ntypes; t++) {
      const p = this.progs[t]!;
      const env = this.env[t]!;
      if (p.vrefs.length) {
        if (!fn) throw new StyleError('angle_style lepton: v_ references need the equal-style variable hook (not available)');
        fillVrefs(env, p, fn);
      }
      // the offset is the value at theta = theta_0, i.e. at the displacement theta - theta_0 = 0
      env[0] = 0;
      this.offset[t] = this.autoOffset ? p.value(env) : 0;
    }
  }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    this.prepare();
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      const env = this.env[t]!;
      const p = this.progs[t]!;
      angleTerm(bc, i, j, k, d1, d2, (theta, c) => {
        env[0] = theta - this.th0[t];
        const ea = p.value(env) - this.offset[t];
        const dEdtheta = p.deriv[0](env);
        e += ea;
        // dE/dc = dE/dtheta * dtheta/dc, with dtheta/dc = -1/sin(theta)
        const s = Math.max(Math.sqrt(1 - c * c), 1e-10);
        return { e: ea, dEdc: -dEdtheta / s };
      });
    }
    bc.acc.eangle += e;
  }
}
