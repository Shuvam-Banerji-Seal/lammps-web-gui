import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { PairHybrid } from '../force/pair/hybrid';

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
 *
 * pair/hybrid style — the docs: "The *pair/hybrid* keyword means to only include contribution from a sub-style in
 * a *hybrid* or *hybrid/overlay* pair style." As native LAMMPS does (src/compute_pressure.cpp at b332f28): the named
 * sub-style's own virial is summed (the hybrid switches to per-sub-style virials, PairHybrid.perSubVirial), with no
 * tail term; a name that matches no sub-style, or several without an index, stops with "Unrecognized pair style X in
 * compute pressure command". Native reads no sub-style index here: its test for one looks at the style name, so a
 * number after the name is an unknown keyword.
 */

const KEYS = ['ke', 'pair', 'bond', 'angle', 'dihedral', 'improper', 'kspace', 'fix', 'virial'] as const;

export class ComputePressure extends Compute {
  readonly style: string = 'pressure';
  tempId: string | null;
  /**
   * Kinetic tensor used in place of the temperature compute's current one (fix nh sets it, see fix/nh.ts
   * computePressCurrent). null = the temperature compute's current tensor.
   */
  keOverride: Float64Array | null = null;
  private terms: Set<string>;
  private w = new Float64Array(6);
  /** pair/hybrid: the sub-style name, and its entry in the hybrid (found at init). */
  private hybridStyle: string | null = null;
  private hybridSub: { virial: Float64Array } | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError('Compute pressure must use group all');
    if (args.length < 1) throw new StyleError('usage: compute ID all pressure temp-ID [keywords]');
    this.tempId = args[0] === 'NULL' ? null : args[0];
    // As native LAMMPS does (src/compute_pressure.cpp ComputePressure::ComputePressure at b332f28)
    if (this.tempId !== null) {
      const t = sys.computes.find((c) => c.id === this.tempId);
      if (!t) throw new StyleError(`Could not find compute pressure temperature ID ${this.tempId}`);
      if (!t.tempFlag) throw new StyleError(`Compute pressure temperature ID ${this.tempId} does not compute temperature`);
    }
    const rest: string[] = [];
    for (let k = 1; k < args.length; k++) {
      const a = args[k];
      if (a === 'pair/hybrid') {
        if (k + 1 >= args.length) throw new StyleError('compute pressure pair/hybrid needs a pair sub-style name');
        this.hybridStyle = args[++k];
        rest.push('pair/hybrid');
        continue;
      }
      if (!(KEYS as readonly string[]).includes(a)) throw new StyleError(`Unknown compute pressure keyword ${a}`);
      rest.push(a);
    }
    if (rest.includes('virial')) rest.push('pair', 'bond', 'angle', 'dihedral', 'improper', 'kspace', 'fix');
    this.terms = new Set(rest.length ? rest : KEYS);
    // As native LAMMPS does (src/compute_pressure.cpp ComputePressure::ComputePressure at b332f28): the ke term needs a temperature ID
    if (this.terms.has('ke') && !this.tempId) throw new StyleError('Compute pressure requires temperature ID to include kinetic energy');
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extscalar = 0;
    this.extvector = 0;
    this.pressFlag = true;
    this.vector = new Float64Array(6);
  }

  init(): void {
    if (this.hybridStyle !== null) {
      const pair = this.sys.ff.pair;
      const name = this.hybridStyle;
      const subs = pair instanceof PairHybrid ? pair.subs.filter((q) => q.name === name) : [];
      if (subs.length !== 1) {
        if (pair && !(pair instanceof PairHybrid) && pair.name === name) {
          throw new StyleError(`compute pressure pair/hybrid ${name}: the pair style is not hybrid (native LAMMPS fails here)`);
        }
        throw new StyleError(`Unrecognized pair style ${name} in compute pressure command`);
      }
      (pair as PairHybrid).enablePerSubVirial();
      this.hybridSub = subs[0];
    }
    // ComputePressure::init: the temperature compute may have been deleted since
    if (this.tempId) {
      const c = this.sys.computes.find((x) => x.id === this.tempId);
      if (!c) throw new StyleError(`Could not find compute pressure temperature ID ${this.tempId}`);
      if (!c.tempFlag) throw new StyleError(`Compute pressure temperature ID ${this.tempId} does not compute temperature`);
    }
  }

  /** The raw virial sum of the selected terms (energy units, no volume, kinetic or tail terms), as a copy. */
  rawVirial(): Float64Array {
    return Float64Array.from(this.virial());
  }

  protected virial(): Float64Array {
    const a = this.sys.forces();
    const w = this.w;
    w.fill(0);
    const add = (v: Float64Array) => { for (let c = 0; c < 6; c++) w[c] += v[c]; };
    if (this.hybridSub) add(this.hybridSub.virial);
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

  /**
   * The pressure tensor for a raw virial w (see rawVirial) in a box of volume vol: the kinetic term and the pair tail
   * are evaluated at that volume. Used by compute born/matrix numdiff, which evaluates the virial in a strained cell
   * but keeps the undeformed volume (as native LAMMPS does).
   */
  vectorAt(w: Float64Array, vol: number): Float64Array {
    const s = this.sys.state;
    const d = this.sys.dimension;
    const ke = this.keOverride ?? (this.terms.has('ke') && this.tempId ? this.sys.compute(this.tempId).vectorValues() : null);
    const tail = this.terms.has('pair') ? this.sys.ff.ptail(vol) : 0;
    const out = new Float64Array(6);
    for (let c = 0; c < 6; c++) {
      let p = w[c] + (ke ? ke[c] : 0);
      p /= vol;
      if (c < 3) p += tail;
      out[c] = p * s.units.nktv2p;
    }
    if (d === 2) { out[2] = 0; out[4] = 0; out[5] = 0; }
    return out;
  }

  protected computeVector(): void {
    this.vector.set(this.vectorAt(this.virial(), this.sys.geom.volume(this.sys.dimension)));
  }

  /**
   * The pressure tensor. With kspace_modify pressure/scalar yes (MSM) a tensor that includes the kspace term is
   * refused, as native LAMMPS does (src/compute_pressure.cpp compute_vector at b332f28: "Must use 'kspace_modify
   * pressure/scalar no' for tensor components with kspace_style msm"); kspace_modify.html: the option "cannot be used
   * to access individual components of the pressure tensor".
   */
  vectorValues(): Float64Array {
    if (this.terms.has('kspace') && this.sys.ff.kspace?.scalarPressure) {
      throw new StyleError("Must use 'kspace_modify pressure/scalar no' for tensor components with kspace_style msm");
    }
    return super.vectorValues();
  }

  /**
   * The tensor for an isotropic barostat, which uses only its diagonal mean (the scalar pressure). Native fix nh and
   * fix press/berendsen call compute_scalar() for an iso barostat (pstyle ISO), so pressure/scalar yes is allowed there.
   */
  isoVectorValues(): Float64Array {
    return super.vectorValues();
  }
}
