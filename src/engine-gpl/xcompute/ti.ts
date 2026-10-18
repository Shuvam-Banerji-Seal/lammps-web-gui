/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMPUTE/compute_ti.cpp and
 * compute_ti.h (ComputeTI: compute ti, the derivative dU/dlambda of a thermodynamic integration). Contributing author:
 * Sai Jayaraman (University of Notre Dame). Copyright (2003) Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_ti.html: "compute ID group-ID ti keyword args ..." with the keywords pair style, tail and kspace,
 * each with the arguments atype v_name1 v_name2: the energy of the atom types atype (per-atom energies, summed over the
 * atoms of those types) times f(lambda) and its derivative, dU/dlambda = U / f * df/dlambda. The scalar is extensive.
 *
 * Engine scope: the pair term is the per-atom energy of the pair style itself (a hybrid's total included). Refused with
 * a StyleError, because the engine does not keep their per-atom energies: a hybrid sub-style, the kspace term (the
 * engine's kspace styles tally no per-atom energy) and the tail term (no per-type tail corrections). The "total" branch of the native code (ihi - ilo == ntypes) can not be reached with
 * a type range 1..ntypes, so every term takes the per-type branch, as in native LAMMPS.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { nativeOrder } from '../../engine/atoms';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import { argBounds } from './common';

type TermKind = 'pair' | 'tail' | 'kspace';

interface Term {
  kind: TermKind;
  pstyle: string;
  ilo: number;
  ihi: number;
  var1: string;
  var2: string;
}

export class ComputeTI extends Compute {
  readonly style = 'ti';
  private readonly terms: Term[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal compute ti command');
    // the per-atom energies of the pair style and the kspace style are needed on every evaluation
    this.needsEatom = true;
    this.scalarFlag = true;
    this.extscalar = 1;
    // terms come in quadruplets: keyword, atom types, v_name1, v_name2
    if (args.length % 4 !== 0) throw new StyleError('Illegal compute ti command');
    const ntypes = sys.state.ntypes;
    for (let iarg = 0; iarg < args.length; iarg += 4) {
      const kw = args[iarg];
      const kind: TermKind = kw === 'kspace' ? 'kspace' : kw === 'tail' ? 'tail' : 'pair';
      const [ilo, ihi] = argBounds(args[iarg + 1], ntypes, 'compute ti');
      if (!args[iarg + 2].startsWith('v_') || !args[iarg + 3].startsWith('v_')) throw new StyleError('Illegal compute ti command');
      this.terms.push({ kind, pstyle: kw, ilo, ihi, var1: args[iarg + 2].slice(2), var2: args[iarg + 3].slice(2) });
    }
  }

  /** The checks of ComputeTI::init(): variables, pair style, tail corrections and kspace style. */
  init(): void {
    this.check();
  }

  private check(): void {
    const sys = this.sys;
    const pf = sys.ff.pair;
    for (const t of this.terms) {
      if (!sys.vars.has(t.var1) || !sys.vars.has(t.var2)) throw new StyleError('Variable name for compute ti does not exist');
      if (sys.vars.get(t.var1)!.style !== 'equal' || sys.vars.get(t.var2)!.style !== 'equal') {
        throw new StyleError('Variable for compute ti is invalid style');
      }
      if (t.kind === 'pair') {
        if (!pf) throw new StyleError('Compute ti pair style does not exist');
        if (pf.name === t.pstyle) continue;
        if (pf instanceof PairHybrid && pf.subs.some((x) => x.name === t.pstyle)) {
          throw new StyleError(`compute ti: the per-atom energy of the hybrid sub-style ${t.pstyle} is not available in the engine`);
        }
        throw new StyleError('Compute ti pair style does not exist');
      }
      if (t.kind === 'tail') {
        if (!pf || !pf.tail) throw new StyleError('Compute ti tail when pair style does not compute tail corrections');
        throw new StyleError('compute ti tail: the per-type tail correction of each pair is not available in the engine');
      }
      if (!sys.ff.kspace) throw new StyleError('Compute ti kspace style does not exist');
      throw new StyleError(`compute ti kspace: the per-atom energy of kspace style ${sys.ff.kspace.name} is not available in the engine`);
    }
  }

  protected computeScalar(): number {
    this.check();
    const sys = this.sys;
    const s = sys.state;
    const order = nativeOrder(s);
    sys.peratomEnergy();
    const pairEatom = sys.ff.eatomTerm.pair ?? null;
    let dUdl = 0.0;
    for (const t of this.terms) {
      const value1 = sys.vars.scalar(t.var1, sys.formulaEnv);
      const value2 = sys.vars.scalar(t.var2, sys.formulaEnv);
      if (value1 === 0.0) continue;
      // the per-atom energies summed over the atoms of the type range (owned atoms: the ghosts of an atom are already
      // added to it, so this equals the native sum over local and ghost atoms of those types). check() refused every
      // term but the pair one.
      let eng = 0.0;
      if (pairEatom) {
        for (const i of order) if (t.ilo <= s.type[i] && t.ihi >= s.type[i]) eng += pairEatom[i];
      }
      dUdl += eng / value1 * value2;
    }
    return dUdl;
  }
}
