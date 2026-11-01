/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ML-PACE/pair_pace.cpp and
 * pair_pace.h (pair_style pace: settings, coeff, init_style, init_one, compute with the full neighbor list and the
 * energy, force and per-atom tallies). Copyright 2021 Yury Lysogorskiy, Cas van der Oord, Anton Bochkarev, Sarath Menon,
 * Matteo Rinaldi, Thomas Hammerschmidt, Matous Mrovec, Aidan Thompson, Gabor Csanyi, Christoph Ortner, Ralf Drautz
 * (Ruhr-University Bochum, University of Cambridge, Sandia National Laboratories, University of British Columbia), and
 * the LAMMPS Sandia copyright notice (GNU General Public License). The ACE evaluator it calls is ported from the
 * ML-PACE library (see ./cbasis.ts, ./radial.ts, ./spherical.ts, ./dag.ts and ./evaluator.ts). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_pace.html. "pair_style pace [recursive|product] [chunksize N] [neigh auto|shared|global]":
 * the default evaluator is recursive; "product" selects the product evaluator; chunksize and neigh are parsed and have no
 * effect here (the KOKKOS-only neigh keyword is accepted so that an input for pace/kk also runs). The docs say the
 * ACE potentials "are parameterized in metal units", and "pair_coeff * * file elem1 ... elemN" maps the atom types to
 * elements ("NULL" skips a type).
 */
import { Pair, StyleError, type AdaptAccess, type PairCompute, type StyleContext } from '../../engine/force/types';
import type { NeighList } from '../../engine/neighbor';
import { scaleAccess, unitScaleTable } from '../../engine/force/pair/scaleTable';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { BBasis } from './bbasis';
import { CTildeBasis } from './cbasis';
import { loadYace } from './yace';
import { ACEPaceEvaluator } from './evaluator';

/** The elements of ML-PACE's elements_pace list (pair_pace.cpp), index = atomic number. */
export const ELEMENTS_PACE = [
  'X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
  'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
  'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
  'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
  'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
  'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
];

/**
 * The basis and the evaluator of each PairPace (large, class-typed): kept outside the style's own fields, so that the
 * pair threads' structured copy of the style (engine cpu/pairThreadsCore.ts cloneableFields) leaves them out; a
 * worker's copy builds its own from the potential text (threadAdopt).
 */
const heavy = new WeakMap<object, { basis: CTildeBasis | null; ace: ACEPaceEvaluator | null }>();

/** The last potential parsed in this realm (a worker rebuilds its copy at each run's init): text and file -> basis. */
let lastParsed: { text: string; fileName: string; basis: CTildeBasis } | null = null;

/** The C-tilde basis of a .ace, .yace or .yaml (B-basis) potential file. */
const parseBasis = (text: string, fileName: string): CTildeBasis => {
  if (lastParsed && lastParsed.fileName === fileName && lastParsed.text === text) return lastParsed.basis;
  const basis = /\.yaml$/.test(fileName) ? BBasis.fromYaml(text, fileName).toCTilde()
    : fileName.endsWith('.yace') ? loadYace(text, fileName) : CTildeBasis.fromText(text, fileName);
  lastParsed = { text, fileName, basis };
  return basis;
};

export class PairPace extends Pair {
  override readonly name: string = 'pace';
  manybody = true;
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;

  /** recursive evaluator (default) or product */
  protected recursive = true;
  protected chunksize = 4096;
  /** LAMMPS type -> ACE species (-1 for NULL), index 1..ntypes */
  protected map = new Int32Array(0);
  protected get basis(): CTildeBasis | null { return heavy.get(this)?.basis ?? null; }
  protected set basis(b: CTildeBasis | null) { this.heavyOf().basis = b; }
  protected get ace(): ACEPaceEvaluator | null { return heavy.get(this)?.ace ?? null; }
  protected set ace(a: ACEPaceEvaluator | null) { this.heavyOf().ace = a; }
  private heavyOf(): { basis: CTildeBasis | null; ace: ACEPaceEvaluator | null } {
    let h = heavy.get(this);
    if (!h) { h = { basis: null, ace: null }; heavy.set(this, h); }
    return h;
  }
  /** The potential file read at pair_coeff, kept for a pair thread's copy to rebuild the basis (threadAdopt). */
  private potFile = '';
  private potText = '';
  protected readonly natOrder = new NativeFullOrder();
  private setflagAll = false;
  /**
   * fix adapt's scale by type pair (pair_pace.cpp extract "scale"): an atom's neighbour forces and energy times
   * scale[itype][itype]; coeff sets every pair to 1 and init_one mirrors it.
   */
  scale: Float64Array = new Float64Array(0);
  /** flag_corerep_factor: the trigger fix pair sets (extract "corerep_flag"); corerep_factor by owned atom */
  flagCorerepFactor = 0;
  corerep: Float64Array | null = null;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.scale = unitScaleTable(ntypes);
    this.map = new Int32Array(ntypes + 1).fill(-1);
    this.basis = null;
    this.ace = null;
    this.setflagAll = false;
  }

  /** PairPACE::settings: [recursive|product] [chunksize N] [neigh auto|shared|global] (messages as native pair_pace.cpp). */
  override settings(args: string[], ctx: StyleContext): void {
    if (args.length > 5) throw new StyleError('Illegal pair_style pace command: missing argument(s)');
    if (ctx.s && ctx.s.units.style !== 'metal') throw new StyleError("ACE potentials require 'metal' units");
    this.recursive = true;
    let iarg = 0;
    while (iarg < args.length) {
      const w = args[iarg];
      if (w === 'recursive') {
        this.recursive = true;
        iarg += 1;
      } else if (w === 'product') {
        this.recursive = false;
        iarg += 1;
      } else if (w === 'chunksize') {
        const v = args[iarg + 1];
        if (v === undefined || !/^[+-]?\d+$/.test(v)) {
          throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
        }
        this.chunksize = parseInt(v, 10);
        iarg += 2;
      } else if (w === 'neigh') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal pair_style pace neigh command: missing argument(s)');
        const v = args[iarg + 1];
        if (v !== 'auto' && v !== 'shared' && v !== 'global') throw new StyleError(`Unknown pair_style pace neigh keyword: ${v}`);
        iarg += 2;
      } else {
        throw new StyleError(`Unknown pair_style pace keyword: ${w}`);
      }
    }
    void this.chunksize;
  }

  /** PairPACE::coeff: "pair_coeff * * file elem1 ... elemN" (one_coeff style). */
  override coeff(args: string[], ctx: StyleContext): void {
    const nt = this.ntypes;
    if (nt === 0) throw new StyleError('Pair_coeff command before simulation box is defined');
    // pair.cpp map_element2type: "Number of element to type mappings does not match number of atom types"
    if (args.length !== 3 + nt) throw new StyleError('Number of element to type mappings does not match number of atom types');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Incorrect args for pair coefficients');
    const fileName = args[2];
    // pair_pace.cpp coeff: a .yaml file is a B-basis (ACEBBasisSet) converted to the C-tilde basis; ACECTildeBasisSet's
    // constructor reads .ace with load and .yace with load_yaml, and refuses other names
    // a missing file: utils::get_potential_file_path gives '', which the C-tilde loader refuses by its (empty) type
    const text = ctx.readFile(fileName, () => "Unrecognized file type of ''. Accept only .ace or .yace");
    const isYaml = /\.yaml$/.test(fileName);
    if (!isYaml && !fileName.endsWith('.ace') && !fileName.endsWith('.yace')) throw new StyleError(`Unrecognized file type of '${fileName}'. Accept only .ace or .yace`);
    void isYaml;
    const basis = parseBasis(text, fileName);
    this.basis = basis;
    this.potFile = fileName;
    this.potText = text;
    // element map: LAMMPS type i -> species index (NULL skips the type)
    const map = new Int32Array(nt + 1).fill(-1);
    for (let i = 1; i <= nt; i++) {
      const elem = args[2 + i];
      if (elem === 'NULL') {
        map[i] = -1;
        continue;
      }
      if (ELEMENTS_PACE.indexOf(elem) < 1) throw new StyleError(`'${elem}' is not a valid element`);
      const mu = basis.speciesIndexByName(elem);
      if (mu === -1) throw new StyleError(`Element ${elem} is not supported by ACE-potential from file ${fileName}`);
      map[i] = mu;
    }
    this.map = map;
    this.ace = new ACEPaceEvaluator(basis, this.recursive, map);
    this.setflagAll = true;
    this.scale.fill(1);
  }

  /** On a pair thread's copy (Pair.threadAdopt): the basis and the evaluator, built from the potential text. */
  override threadAdopt(): void {
    if (!this.potFile || this.ace) return;
    const basis = parseBasis(this.potText, this.potFile);
    this.basis = basis;
    this.ace = new ACEPaceEvaluator(basis, this.recursive, this.map);
  }

  /** On the pair threads the rows take their neighbors in native full-list order (the evaluator's sums follow it). */
  override threadNeighbors(pc: PairCompute, list: NeighList): Int32Array {
    return this.natOrder.neighbors(pc.s, pc.nb, pc.geom, list);
  }

  /** Not on the threads while fix pair reads the per-atom corerep factor (each thread's copy would hold its own). */
  override threadable(): boolean {
    return !this.flagCorerepFactor;
  }

  /** extract("corerep_flag"): the scalar trigger fix pair sets. */
  hasTrigger(name: string): boolean {
    return name === 'corerep_flag';
  }

  setTrigger(name: string, value: number): boolean {
    if (name !== 'corerep_flag') return false;
    this.flagCorerepFactor = value;
    return true;
  }

  /** extract_peratom("corerep"): 1 - ace_fcut by owned atom (null before the first step that computes it). */
  extractPeratom(name: string): { values: Float64Array | null; columns: number } | undefined {
    return name === 'corerep' ? { values: this.corerep, columns: 0 } : undefined;
  }

  override adaptParam(name: string): AdaptAccess | null {
    return name === 'scale' ? scaleAccess(this) : null;
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.s && ctx.s.units.style !== 'metal') throw new StyleError("ACE potentials require 'metal' units");
    if (ctx.newtonPair === false) throw new StyleError('Pair style pace requires newton pair on');
  }

  /** PairPACE::init_one: the radial cutoff of the species pair. */
  override initOne(i: number, j: number): number {
    if (!this.setflagAll || !this.basis) throw new StyleError('All pair coeffs are not set');
    const nt = this.ntypes + 1;
    this.scale[j * nt + i] = this.scale[i * nt + j];
    const mi = this.map[i], mj = this.map[j];
    if (mi < 0 || mj < 0) return 0;
    return this.basis.radial.cut[mi * this.basis.nelements + mj];
  }

  /** PairPACE::compute: the atomic energy of each owned atom, and the forces on its neighbours (full list). */
  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    if (!this.ace) throw new StyleError('All pair coeffs are not set');
    const ace = this.ace;
    if (this.flagCorerepFactor && pc.nlocal > (this.corerep?.length ?? 0)) this.corerep = new Float64Array(pc.nlocal);
    const x = pc.x, f = pc.f, type = pc.type;
    // the neighbours in native full-list order (the evaluator's sums follow this order)
    // on the pair threads the list arrives in native order (threadNeighbors)
    const nbrs = list.nativeOrdered ? list.neighbors : this.natOrder.neighbors(pc.s, pc.nb, pc.geom, list);
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      const jl = new Int32Array(jnum);
      for (let jj = 0; jj < jnum; jj++) jl[jj] = nbrs[k0 + jj] & NEIGHMASK;
      ace.computeAtom(i, x, type, jnum, jl, 0);
      if (this.flagCorerepFactor) this.corerep![i] = 1 - ace.aceFcut;
      const nf = ace.neighboursForces;
      const sc = this.scale[type[i] * (this.ntypes + 1) + type[i]];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let jj = 0; jj < jnum; jj++) {
        const j = jl[jj];
        const fx = sc * nf[3 * jj], fy = sc * nf[3 * jj + 1], fz = sc * nf[3 * jj + 2];
        f[3 * i] += fx;
        f[3 * i + 1] += fy;
        f[3 * i + 2] += fz;
        f[3 * j] -= fx;
        f[3 * j + 1] -= fy;
        f[3 * j + 2] -= fz;
        if (pc.vatom) {
          // ev_tally_xyz: the separation x_i - x_j times the force, half to each atom
          const dx = -(x[3 * j] - xtmp), dy = -(x[3 * j + 1] - ytmp), dz = -(x[3 * j + 2] - ztmp);
          const v = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
          const va = pc.vatom;
          for (let c = 0; c < 6; c++) {
            va[6 * i + c] += 0.5 * v[c];
            va[6 * j + c] += 0.5 * v[c];
          }
        }
      }
      // ev_tally_full with 2 * scale * e_atom: energy scale * e_atom to the total and to the atom
      const evdwl = sc * ace.eAtom;
      pc.acc.evdwl += evdwl;
      if (pc.eatom) pc.eatom[i] += evdwl;
    }
  }
}
