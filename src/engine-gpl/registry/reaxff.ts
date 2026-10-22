/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/pair_reaxff.cpp and src/REAXFF/fix_qeq_reaxff.cpp (style registration), at
 * commit b332f28. Authors credited in the C++ file headers: Hasan Metin Aktulga (Michigan State
 * University / Purdue University); Ray Shan (Sandia). Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { ComputeFactory } from '../../engine/styles';
import type { Pair } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairReaxFF } from '../reaxff/pair';
import { FixQEqReaxFF } from '../reaxff/qeq';
import { FixACKS2ReaxFF } from '../reaxff/acks2';
import { FixQEqRelReaxFF, FixQtpieReaxFF } from '../reaxff/qtpie';
import { FixReaxFFBonds, FixReaxFFSpecies, reaxffPairOf } from '../reaxff/reports';
import { ComputeReaxFFAtom } from '../reaxff/compute_atom';

/** ReaxFF styles ported from LAMMPS (src/REAXFF); merged into styles.ts through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  reaxff: () => new PairReaxFF(),
};

/** A factory for the style name given in the input (a fix's messages name its style as invoked, alias or not). */
const qeqFactory = (invoked: string) => (sys: System, id: string, group: string, args: string[]): Fix =>
  new FixQEqReaxFF(sys, id, group, args, () => (sys.ff.pair instanceof PairReaxFF ? sys.ff.pair : null), invoked);

const qtpieFactory = (sys: System, id: string, group: string, args: string[]): Fix =>
  new FixQtpieReaxFF(sys, id, group, args, () => (sys.ff.pair instanceof PairReaxFF ? sys.ff.pair : null));

const qeqRelFactory = (sys: System, id: string, group: string, args: string[]): Fix =>
  new FixQEqRelReaxFF(sys, id, group, args, () => (sys.ff.pair instanceof PairReaxFF ? sys.ff.pair : null));

const acks2Factory = (invoked: string) => (sys: System, id: string, group: string, args: string[]): Fix =>
  new FixACKS2ReaxFF(sys, id, group, args, () => (sys.ff.pair instanceof PairReaxFF ? sys.ff.pair : null), invoked);

const bondsFactory = (sys: System, id: string, group: string, args: string[]): Fix =>
  new FixReaxFFBonds(sys, id, group, args, () => reaxffPairOf(sys));

const speciesFactory = (sys: System, id: string, group: string, args: string[]): Fix =>
  new FixReaxFFSpecies(sys, id, group, args, () => reaxffPairOf(sys));

/**
 * fix qeq/reaxff (and its alias qeq/reax), fix acks2/reaxff (alias acks2/reax), fix qtpie/reaxff,
 * fix qeq/rel/reaxff, fix reaxff/bonds and fix reaxff/species, all in LAMMPS' FixStyle lists.
 */
export const FIXES: Record<string, typeof qtpieFactory> = {
  'qeq/reaxff': qeqFactory('qeq/reaxff'),
  'qeq/reax': qeqFactory('qeq/reax'),
  'acks2/reaxff': acks2Factory('acks2/reaxff'),
  'acks2/reax': acks2Factory('acks2/reax'),
  'qtpie/reaxff': qtpieFactory,
  'qeq/rel/reaxff': qeqRelFactory,
  'reaxff/bonds': bondsFactory as unknown as typeof qtpieFactory,
  'reaxff/species': speciesFactory as unknown as typeof qtpieFactory,
};

/** compute reaxff/atom (src/REAXFF/compute_reaxff_atom.cpp), in reaxff/compute_atom.ts. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'reaxff/atom': (sys, id, group, args) => new ComputeReaxFFAtom(sys, id, group, args),
};
