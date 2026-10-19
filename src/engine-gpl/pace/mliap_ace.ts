/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ML-IAP/mliap_descriptor_ace.cpp
 * and mliap_descriptor_ace.h (MLIAPDescriptorACE: the constructor's element list, descriptor count and cutsq, and the
 * per-atom evaluation of compute_descriptors and compute_forces), at commit b332f28. Authors named in the source: James
 * Goff (Sandia National Laboratories), with the LAMMPS Sandia copyright notice (GNU General Public License). The ACE
 * projections come from ./projections.ts (ported from the ML-PACE library). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (kind HOOKS, word 'ace'): pair_style mliap
 * (src/engine/force/pair/mliap.ts) reads it through gplHook('ace'); the contract is MliapAceHook
 * (src/engine/mliap/ace.ts).
 */
import { StyleError } from '../../engine/force/types';
import type { MliapAceDescriptor, MliapAceHook } from '../../engine/mliap/ace';
import type { PairCompute } from '../../engine/force/types';
import type { NeighList } from '../../engine/neighbor';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { CTildeBasis } from './cbasis';
import { loadYace } from './yace';
import { AceProjections } from './projections';
import { ComputeMliapAce } from './compute_mliap_ace';
import type { System } from '../../engine/system';

class AceDescriptor implements MliapAceDescriptor {
  readonly elements: readonly string[];
  readonly nelements: number;
  readonly ndescriptors: number;
  private readonly cutsqTable: Float64Array;
  private readonly proj: AceProjections;
  private readonly natOrder = new NativeFullOrder();

  constructor(text: string, filename: string) {
    // ACECTildeBasisSet(filename): .ace with load, .yace with load_yaml
    let basis: CTildeBasis;
    if (filename.endsWith('.ace')) basis = CTildeBasis.fromText(text, filename);
    else if (filename.endsWith('.yace')) basis = loadYace(text, filename);
    else throw new StyleError(`Unrecognized file type of '${filename}'. Accept only .ace or .yace`);
    const nel = basis.nelements;
    this.nelements = nel;
    this.elements = basis.elementsName.slice(0, nel);
    // ndescriptors: the largest rank-1 plus higher-rank count of the species
    let maxNum = 0;
    for (let mu = 0; mu < nel; mu++) {
      const num = basis.totalBasisSizeRank1[mu] + basis.totalBasisSize[mu];
      if (maxNum < num) maxNum = num;
    }
    this.ndescriptors = maxNum;
    // cutsq in single precision as the constructor computes it: cuti = cut(mui, mui), cutj = cut(mui, muj), and
    // cutsq = (2 cuti cutfac)(2 cutj cutfac) with cutfac = 0.5, all floats
    const f = Math.fround;
    const cut = basis.radial.cut;
    const cutsq = new Float64Array(nel * nel);
    const cutfac = f(0.5);
    for (let mui = 0; mui < nel; mui++) {
      const cuti = f(cut[mui * nel + mui]);
      const ci = f(f(2 * cuti) * cutfac);
      cutsq[mui * nel + mui] = f(ci * ci);
      for (let muj = mui + 1; muj < nel; muj++) {
        const cutj = f(cut[mui * nel + muj]);
        const v = f(ci * f(f(2 * cutj) * cutfac));
        cutsq[mui * nel + muj] = v;
        cutsq[muj * nel + mui] = v;
      }
    }
    this.cutsqTable = cutsq;
    // element_type_mapping(ik) = ik - 1 for the types 1..nelements
    const map = new Int32Array(nel + 1);
    for (let ik = 1; ik <= nel; ik++) map[ik] = ik - 1;
    this.proj = new AceProjections(basis, map);
  }

  nativeNeighbors(pc: PairCompute, full: NeighList): Int32Array {
    return this.natOrder.neighbors(pc.s, pc.nb, pc.geom, full);
  }

  cutsq(ei: number, ej: number): number {
    return this.cutsqTable[ei * this.nelements + ej];
  }

  evaluate(i: number, x: Float64Array, type: Int32Array, jnum: number, jlist: Int32Array): { B: Float64Array; dB: Float64Array } {
    // compute_atom maps the atom types themselves (type t is species t - 1); a type beyond the elements has no species
    const ti = type[i];
    if (ti < 1 || ti > this.nelements) throw new StyleError(`pair_style mliap descriptor ace: atom type ${ti} has no ACE species (type t is species t-1 of the ${this.nelements} in the file)`);
    for (let jj = 0; jj < jnum; jj++) {
      const tj = type[jlist[jj]];
      if (tj < 1 || tj > this.nelements) throw new StyleError(`pair_style mliap descriptor ace: atom type ${tj} has no ACE species (type t is species t-1 of the ${this.nelements} in the file)`);
    }
    this.proj.computeAtom(i, x, type, jnum, jlist);
    if (this.proj.projections.length < this.ndescriptors) {
      throw new StyleError(`pair_style mliap descriptor ace: species ${ti - 1} has fewer than the ${this.ndescriptors} descriptors of the largest species`);
    }
    return { B: this.proj.projections, dB: this.proj.neighboursDB };
  }
}

/** The HOOKS entry 'ace' (the contract is MliapAceHook, src/engine/mliap/ace.ts). */
export const HOOKS: Record<string, MliapAceHook> = {
  ace: {
    load: (text: string, filename: string): MliapAceDescriptor => new AceDescriptor(text, filename),
    compute: (sys: System, id: string, group: string, args: string[]) => {
      const at = args.findIndex((w, k) => w === 'descriptor' && args[k + 1] === 'ace');
      const file = at >= 0 ? args[at + 2] : undefined;
      if (file === undefined) throw new StyleError('Illegal compute mliap descriptor ace command: missing argument(s)');
      return new ComputeMliapAce(sys, id, group, args, new AceDescriptor(sys.readFile(file), file));
    },
  },
};
