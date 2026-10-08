import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * compute ID all pe [pair bond angle dihedral improper kspace fix] —
 * docs.lammps.org/compute_pe.html: "The specified group must be "all"."
 * "If no extra keywords are listed, then the potential energy is the sum of
 * pair, bond, angle, dihedral, improper, k-space (long-range), and fix
 * energy (i.e., it is as though all the keywords were listed). If any extra
 * keywords are listed, then only those components are summed". "The
 * fix_modify energy yes command must also be specified if a fix is to
 * contribute potential energy to this command."
 */

const KEYS = ['pair', 'bond', 'angle', 'dihedral', 'improper', 'kspace', 'fix'] as const;

export class ComputePE extends Compute {
  readonly style = 'pe';
  private terms: Set<string>;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError('compute pe must use group all');
    for (const a of args) if (!(KEYS as readonly string[]).includes(a)) throw new StyleError(`unknown compute pe keyword '${a}'`);
    this.terms = new Set(args.length ? args : KEYS);
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  protected computeScalar(): number {
    const a = this.sys.forces();
    let e = 0;
    if (this.terms.has('pair')) e += a.evdwl + a.ecoul;
    if (this.terms.has('bond')) e += a.ebond;
    if (this.terms.has('angle')) e += a.eangle;
    if (this.terms.has('dihedral')) e += a.edihed;
    if (this.terms.has('improper')) e += a.eimp;
    if (this.terms.has('kspace')) e += a.elong;
    if (this.terms.has('fix')) e += this.sys.fixEnergy();
    return e;
  }
}
