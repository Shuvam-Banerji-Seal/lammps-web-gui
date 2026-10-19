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
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { CTildeBasis } from './cbasis';
import { ACEPaceEvaluator } from './evaluator';

/** The elements of ML-PACE's elements_pace list (pair_pace.cpp), index = atomic number. */
const ELEMENTS_PACE = [
  'X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
  'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
  'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
  'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
  'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
  'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
];

export class PairPace extends Pair {
  override readonly name: string = 'pace';
  manybody = true;
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;

  /** recursive evaluator (default) or product */
  private recursive = true;
  private chunksize = 4096;
  /** LAMMPS type -> ACE species (-1 for NULL), index 1..ntypes */
  private map = new Int32Array(0);
  private basis: CTildeBasis | null = null;
  private ace: ACEPaceEvaluator | null = null;
  private readonly natOrder = new NativeFullOrder();
  private setflagAll = false;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
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
    if (/\.yaml$/.test(fileName) || /\.yace$/.test(fileName)) {
      throw new StyleError(`pair_style pace: the YAML .yace potential format (${fileName}) is not supported yet; use a .ace file`);
    }
    if (!fileName.endsWith('.ace')) throw new StyleError(`Unrecognized file type of '${fileName}'. Accept only .ace or .yace`);
    const text = ctx.readFile(fileName);
    const basis = CTildeBasis.fromText(text, fileName);
    this.basis = basis;
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
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.s && ctx.s.units.style !== 'metal') throw new StyleError("ACE potentials require 'metal' units");
    if (ctx.newtonPair === false) throw new StyleError('Pair style pace requires newton pair on');
  }

  /** PairPACE::init_one: the radial cutoff of the species pair. */
  override initOne(i: number, j: number): number {
    if (!this.setflagAll || !this.basis) throw new StyleError('All pair coeffs are not set');
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
    const x = pc.x, f = pc.f, type = pc.type;
    // the neighbours in native full-list order (the evaluator's sums follow this order)
    const nbrs = this.natOrder.neighbors(pc.s, pc.nb, pc.geom, list);
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      const jl = new Int32Array(jnum);
      for (let jj = 0; jj < jnum; jj++) jl[jj] = nbrs[k0 + jj] & NEIGHMASK;
      ace.computeAtom(i, x, type, jnum, jl, 0);
      const nf = ace.neighboursForces;
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let jj = 0; jj < jnum; jj++) {
        const j = jl[jj];
        const fx = nf[3 * jj], fy = nf[3 * jj + 1], fz = nf[3 * jj + 2];
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
      // ev_tally_full with 2 * scale * e_atom: energy e_atom to the total and to the atom (scale = 1)
      const evdwl = 1.0 * ace.eAtom;
      pc.acc.evdwl += evdwl;
      if (pc.eatom) pc.eatom[i] += evdwl;
    }
  }
}
