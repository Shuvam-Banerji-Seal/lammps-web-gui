import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';
import { atomIndex, delta } from '../bonded_util';
import { parseNum } from '../util';
import { compileLepton, fillVrefs, type LeptonProgram } from '../../lepton';

/*
 * bond_style lepton [auto_offset|no_offset] — docs.lammps.org/bond_lepton.html.
 * The expression uses r as the distance variable relative to the reference
 * distance r_0 (a bond coefficient): U = K (r_i - r_0)^2 = K r^2 with
 * r = r_i - r_0 is the page's example. auto_offset (the default, since
 * 7Feb2024) shifts the energy so that it is 0.0 at r = r_0; no_offset does not.
 * bond_coeff N r0 "expression": the expression is in energy units, r0 in distance units.
 * Measured with native LAMMPS (black box): with bond_coeff 1 1.0 "r^2+1" and a
 * bond length of 1.2 the energy is 0.04 with auto_offset and 1.04 with no_offset,
 * so the expression sees r = r_i - r_0 = 0.2 and the offset is its value at r = 0.
 * Forces use the exact derivative of the expression (lepton/index.ts).
 */

export class BondLepton extends Bonded {
  readonly name = 'lepton';
  readonly kind = 'bond' as const;
  private autoOffset = true;
  private r0 = new Float64Array(0);
  private set = new Uint8Array(0);
  private progs: (LeptonProgram | null)[] = [];
  private texts: string[] = [];
  private env: Float64Array[] = [];
  private offset = new Float64Array(0);
  private ctx: StyleContext | null = null;

  settings(args: string[]): void {
    if (args.length > 1) throw new StyleError('usage: bond_style lepton [auto_offset|no_offset]');
    if (args.length === 1) {
      if (args[0] === 'auto_offset') this.autoOffset = true;
      else if (args[0] === 'no_offset') this.autoOffset = false;
      else throw new StyleError(`bond_style lepton: unknown argument '${args[0]}' (expected auto_offset or no_offset)`);
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.r0 = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
    this.progs = new Array(ntypes + 1).fill(null);
    this.texts = new Array(ntypes + 1).fill('');
    this.env = new Array(ntypes + 1).fill(null);
    this.offset = new Float64Array(ntypes + 1);
  }

  coeff(args: string[], ctx?: StyleContext): void {
    if (ctx) this.ctx = ctx;
    if (args.length !== 3) throw new StyleError('usage: bond_coeff N r0 "expression"');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const r0 = parseNum(args[1], 'r0');
    const prog = compileLepton(args[2], { builtins: ['r'], wrt: ['r'] });
    for (let t = lo; t <= hi; t++) {
      this.r0[t] = r0;
      this.set[t] = 1;
      this.progs[t] = prog;
      this.texts[t] = args[2];
      this.env[t] = new Float64Array(prog.builtins.length + prog.vrefs.length);
    }
  }

  init(ctx?: StyleContext): void {
    if (ctx) this.ctx = ctx;
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`all bond coeffs are not set (type ${t})`);
    }
  }

  /** Fills v_name slots and the offsets (value at r = 0) for the current time. */
  private prepare(): void {
    for (let t = 1; t <= this.ntypes; t++) {
      const p = this.progs[t]!;
      const env = this.env[t]!;
      this.fill(p, env);
      env[0] = 0;
      this.offset[t] = this.autoOffset ? p.value(env) : 0;
    }
  }

  private fill(p: LeptonProgram, env: Float64Array): void {
    if (!p.vrefs.length) return;
    const fn = (this.ctx as { equalVariable?: (n: string) => number } | null)?.equalVariable;
    if (!fn) throw new StyleError(`bond_style lepton: v_ references need the equal-style variable hook (not available)`);
    fillVrefs(env, p, fn);
  }

  compute(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    this.prepare();
    const d = [0, 0, 0];
    const f = bc.f;
    const v = bc.virial;
    let e = 0;
    for (let k = 0; k < b.n; k++) {
      const i = atomIndex(bc, b.atoms[2 * k], 'bond');
      const j = atomIndex(bc, b.atoms[2 * k + 1], 'bond');
      const t = b.type[k];
      delta(bc, j, i, d);   // r_i - r_j
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const env = this.env[t]!;
      const p = this.progs[t]!;
      env[0] = r - this.r0[t];
      const eb = p.value(env) - this.offset[t];
      const dEdr = p.deriv[0](env);
      e += eb;
      const fbond = r > 0 ? -dEdr / r : 0;
      f[3 * i] += d[0] * fbond; f[3 * i + 1] += d[1] * fbond; f[3 * i + 2] += d[2] * fbond;
      f[3 * j] -= d[0] * fbond; f[3 * j + 1] -= d[1] * fbond; f[3 * j + 2] -= d[2] * fbond;
      const w = [d[0] * d[0] * fbond, d[1] * d[1] * fbond, d[2] * d[2] * fbond, d[0] * d[1] * fbond, d[0] * d[2] * fbond, d[1] * d[2] * fbond];
      for (let c = 0; c < 6; c++) v[c] += w[c];
      if (bc.eatom) { bc.eatom[i] += 0.5 * eb; bc.eatom[j] += 0.5 * eb; }
      if (bc.vatom) for (let c = 0; c < 6; c++) { bc.vatom[6 * i + c] += 0.5 * w[c]; bc.vatom[6 * j + c] += 0.5 * w[c]; }
    }
    bc.acc.ebond += e;
  }

  equilibrium(type: number): number { return this.r0[type]; }
}
