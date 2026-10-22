/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the coefficient handling and init of the
 * SPH pair styles (src/SPH/pair_sph_taitwater.cpp, pair_sph_taitwater_morris.cpp, pair_sph_idealgas.cpp,
 * pair_sph_heatconduction.cpp, pair_sph_rhosum.cpp: allocate, coeff, init_one). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 */
import { Pair, StyleError, typeBounds, type PairConstructorWhere, type StyleContext } from '../../engine/force/types';
import { isSphStyle } from '../../engine/atoms';
import { atomHasFlag, type AtomFlag } from '../atom_flags';

/**
 * Base of the SPH pair styles. Pair cutoffs (the kernel support h) are per type pair, symmetric, with I <= J set by
 * pair_coeff; per-type values are set for every type of the I range (as the source does).
 */
export abstract class PairSPHBase extends Pair {
  /** The virial comes from x.f over owned and ghost atoms (the source's virial_fdotr_compute); hybrid requires it. */
  override virialFdotr = true;
  protected nt = 0;
  /** Kernel support h per type pair (index i * nt + j). */
  protected h = new Float64Array(0);
  protected setflag = new Uint8Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
    this.h = new Float64Array(this.nt * this.nt);
    this.setflag = new Uint8Array(this.nt * this.nt);
  }

  /** Type ranges of a pair_coeff line: I and J (wildcards allowed), the values after them. */
  protected coeffRange(args: string[], count: number, style: string): { ilo: number; ihi: number; jlo: number; jhi: number; values: string[] } {
    if (args.length !== count + 2) throw new StyleError(`Incorrect args for pair_coeff ${style} coefficients`);
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    return { ilo, ihi, jlo, jhi, values: args.slice(2) };
  }

  /** Sets the pairs I <= J of a range through set(i, j); throws when no pair is in the range. */
  protected forPairs(ilo: number, ihi: number, jlo: number, jhi: number, set: (i: number, j: number) => void): void {
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        set(i, j);
        this.setflag[i * this.nt + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** The constructor's check of the atom attribute flags the style needs. */
  override constructorCheck(where: PairConstructorWhere): void {
    if (!this.attributeFlags.every((f) => atomHasFlag(where.atomStyle, f))) throw new StyleError(this.attributesMessage());
  }

  /** Newton check at init; the engine's SPH arrays come with atom_style sph only. */
  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError(`pair_style ${this.name} with newton pair off is not supported by the browser engine`);
    if (ctx.s && !isSphStyle(ctx.s.atomStyle)) throw new StyleError(this.attributesMessage());
  }

  protected attributesMessage(): string {
    return `Pair ${this.name} requires atom ${this.attributesText}, e.g. in atom_style sph`;
  }

  /** The atom attribute flags the constructor tests, and how its message names them. */
  protected abstract readonly attributeFlags: readonly AtomFlag[];
  protected abstract readonly attributesText: string;

  override initOne(i: number, j: number): number {
    const nt = this.nt;
    if (!this.setflag[i * nt + j]) throw new StyleError(`All pair ${this.name} coeffs are not set`);
    this.h[j * nt + i] = this.h[i * nt + j];
    this.mirror(i, j);
    return this.h[i * nt + j];
  }

  /** Copies the per-pair coefficients of (i, j) to (j, i) (init_one). */
  protected mirror(_i: number, _j: number): void {}
}
