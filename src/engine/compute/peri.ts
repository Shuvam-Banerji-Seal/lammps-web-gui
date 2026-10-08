import type { System } from '../system';
import { StyleError } from '../force/types';
import { Compute } from './compute';
import { PairPeri } from '../force/pair/peri';

/*
 * Per-atom peridynamic computes (PERI package). docs.lammps.org/compute_damage_atom.html:
 * "Define a computation that calculates the per-atom damage for each atom". The damage is the
 * volume-weighted fraction of broken bonds of the family (Howto_peri.rst, eq. 10).
 */

const periPair = (sys: System, style: string): PairPeri => {
  const p = sys.ff.pair;
  if (!(p instanceof PairPeri)) throw new StyleError(`compute ${style} needs a peridynamic pair style (peri/pmb or peri/lps)`);
  return p;
};

/** compute ID group-ID damage/atom — docs.lammps.org/compute_damage_atom.html. */
export class ComputeDamageAtom extends Compute {
  readonly style = 'damage/atom';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('compute damage/atom takes no arguments');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = periPair(this.sys, 'damage/atom');
    const d = pair.damage(s.vfrac ?? new Float64Array(s.n).fill(1), s.n);
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.vectorAtom[i] = d[i];
  }
}

/**
 * compute ID group-ID dilatation/atom (docs.lammps.org/compute_dilatation_atom.html) works with the
 * peri/lps pair style here; peri/ves and peri/eps are not available in this engine.
 */
export class ComputeDilatationAtom extends Compute {
  readonly style = 'dilatation/atom';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('compute dilatation/atom takes no arguments');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = periPair(this.sys, 'dilatation/atom');
    const th = pair.dilatation(s.n);
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.vectorAtom[i] = th[i];
  }
}
