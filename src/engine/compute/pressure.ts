import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * compute ID all pressure temp-ID [ke pair bond angle dihedral improper
 * kspace fix virial] — docs.lammps.org/compute_pressure.html:
 *   P = N k_B T / V + 1/(V d) sum r_i . f_i
 * "where N is the number of atoms in the system (see discussion of DOF
 * below)" — the kinetic term uses the temperature compute's degrees of
 * freedom: dof k_B T / (d V).
 *   P_IJ = 1/V sum m v_I v_J + 1/V sum r_I f_J
 * "If no extra keywords are listed, the entire equations above are
 * calculated. ... If any extra keywords are listed, then only those
 * components are summed ... The virial keyword means include all terms
 * except the kinetic energy ke." "temp-ID = ID of compute that calculates
 * temperature, can be NULL if not needed". Tail corrections
 * (pair_modify tail yes) add to the pressure (pair_modify.html).
 */

const KEYS = ['ke', 'pair', 'bond', 'angle', 'dihedral', 'improper', 'kspace', 'fix', 'virial'] as const;

export class ComputePressure extends Compute {
  readonly style = 'pressure';
  tempId: string | null;
  private terms: Set<string>;
  private w = new Float64Array(6);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError('compute pressure must use group all');
    if (args.length < 1) throw new StyleError('usage: compute ID all pressure temp-ID [keywords]');
    this.tempId = args[0] === 'NULL' ? null : args[0];
    const rest = args.slice(1);
    for (const a of rest) if (!(KEYS as readonly string[]).includes(a)) throw new StyleError(`unknown compute pressure keyword '${a}'`);
    if (rest.includes('virial')) rest.push('pair', 'bond', 'angle', 'dihedral', 'improper', 'kspace', 'fix');
    this.terms = new Set(rest.length ? rest : KEYS);
    if (this.terms.has('ke') && !this.tempId) throw new StyleError('compute pressure with temp-ID NULL cannot include the ke term');
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extscalar = 0;
    this.extvector = 0;
    this.pressFlag = true;
    this.vector = new Float64Array(6);
  }

  init(): void {
    if (this.tempId) {
      const c = this.sys.compute(this.tempId);
      if (!c.tempFlag) throw new StyleError(`compute pressure: compute ${this.tempId} does not compute a temperature`);
    }
  }

  private virial(): Float64Array {
    const a = this.sys.forces();
    const w = this.w;
    w.fill(0);
    const add = (v: Float64Array) => { for (let c = 0; c < 6; c++) w[c] += v[c]; };
    if (this.terms.has('pair')) add(a.virial);
    if (this.terms.has('bond')) add(a.vbond);
    if (this.terms.has('angle')) add(a.vangle);
    if (this.terms.has('dihedral')) add(a.vdihed);
    if (this.terms.has('improper')) add(a.vimp);
    if (this.terms.has('kspace')) add(a.vlong);
    if (this.terms.has('fix')) this.sys.fixVirial(w);
    return w;
  }

  protected computeScalar(): number {
    const s = this.sys.state;
    const d = this.sys.dimension;
    const vol = this.sys.geom.volume(d);
    const w = this.virial();
    const trace = d === 3 ? w[0] + w[1] + w[2] : w[0] + w[1];
    let p = trace / (d * vol);
    if (this.terms.has('ke') && this.tempId) {
      const t = this.sys.compute(this.tempId);
      const temp = t.scalarValue();
      p += (t.dof * s.units.boltz * temp) / (d * vol);
    }
    if (this.terms.has('pair')) p += this.sys.ff.ptail(vol);
    return p * s.units.nktv2p;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const d = this.sys.dimension;
    const vol = this.sys.geom.volume(d);
    const w = this.virial();
    const ke = this.terms.has('ke') && this.tempId ? this.sys.compute(this.tempId).vectorValues() : null;
    const tail = this.terms.has('pair') ? this.sys.ff.ptail(vol) : 0;
    for (let c = 0; c < 6; c++) {
      let p = w[c] + (ke ? ke[c] : 0);
      p /= vol;
      if (c < 3) p += tail;
      this.vector[c] = p * s.units.nktv2p;
    }
    if (d === 2) { this.vector[2] = 0; this.vector[4] = 0; this.vector[5] = 0; }
  }
}
