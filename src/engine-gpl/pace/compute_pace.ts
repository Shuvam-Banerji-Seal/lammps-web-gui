/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ML-PACE/compute_pace.cpp and
 * compute_pace.h (compute pace: the global array of the ACE descriptors, their gradients and the reference energy,
 * forces and virial), at commit b332f28. Authors named in the source: James Goff and Drew Rohskopf (Sandia National
 * Laboratories), with the LAMMPS Sandia copyright notice (GNU General Public License). The ACE projections come from
 * ./projections.ts (ported from the ML-PACE library). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/compute_pace.html, "compute ID group-ID pace ace_potential_filename ... bikflag ... dgradflag".
 * The array is sized when the compute is defined (N = the atom count then) and laid out as compute_array writes it:
 * - dgradflag 0: bik rows (1, or N with bikflag 1), then 3N gradient rows at 3 (id-1) + 1 + d (from row 1 whatever
 *   bikflag is, as native does), then 6 virial rows at bik_rows + 3N in the order xx yy zz yz xz xy. The columns are
 *   the descriptors of each atom type in turn (the block of the central atom's type), then one reference column: the
 *   potential energy in row 0 (the other bik rows zeroed), the force components in the gradient rows and the pressure
 *   (compute pressure NULL virial) in the virial rows.
 * - dgradflag 1: N rows with the force components in columns 0..2 and the descriptors from column 3, then 3N^2 rows
 *   (bik_rows + 3N (id_j - 1) + 3 (id_i - 1) + d) holding id_i - 1, id_j - 1, d and -dB_i/dr_j (dB_i/dr_i on the
 *   diagonal), then the potential energy in the first column of the last row.
 * As native, type t uses ACE species t - 1 (species 0 for a type beyond the file's elements) and the descriptor count
 * is that of species 0.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { CTildeBasis } from './cbasis';
import { loadYace } from './yace';
import { AceProjections } from './projections';

const inumeric = (w: string): number => {
  if (w === undefined || w === '') throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  return parseInt(w, 10);
};

export class ComputePACE extends Compute {
  readonly style = 'pace';
  private readonly basis: CTildeBasis;
  private readonly proj: AceProjections;
  private readonly bikflag: boolean;
  private readonly dgradflag: boolean;
  private readonly natoms: number;
  private readonly nvalues: number;
  private readonly bikRows: number;
  private readonly natOrder = new NativeFullOrder();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.arrayFlag = true;
    this.extarray = 0;
    if (args.length < 1) throw new StyleError('Illegal compute pace command');
    this.bikflag = inumeric(args[1]) !== 0;
    this.dgradflag = inumeric(args[2]) !== 0;
    if (this.dgradflag && !this.bikflag) throw new StyleError('Illegal compute pace command: dgradflag=1 requires bikflag=1');
    const fileName = args[0];
    const ntypes = sys.state.ntypes;
    // ACECTildeBasisSet(filename): .ace with load, .yace with load_yaml
    let basis: CTildeBasis;
    if (fileName.endsWith('.ace')) basis = CTildeBasis.fromText(sys.readFile(fileName), fileName);
    else if (fileName.endsWith('.yace')) basis = loadYace(sys.readFile(fileName), fileName);
    else throw new StyleError(`Unrecognized file type of '${fileName}'. Accept only .ace or .yace`);
    this.basis = basis;
    // element_type_mapping(ik) = ik - 1 for the types that name a species; the others keep the array's zero
    const map = new Int32Array(ntypes + 1);
    for (let ik = 1; ik <= ntypes; ik++) map[ik] = ik - 1 < basis.nelements ? ik - 1 : 0;
    this.proj = new AceProjections(basis, map);
    this.nvalues = basis.totalBasisSizeRank1[0] + basis.totalBasisSize[0];
    this.natoms = sys.state.n;
    this.bikRows = this.bikflag ? this.natoms : 1;
    if (this.dgradflag) {
      this.sizeArrayRows = this.bikRows + 3 * this.natoms * this.natoms + 1;
      this.sizeArrayCols = this.nvalues + 3;
      sys.warn('dgradflag=1 creates a N^2 array, beware of large systems.');
    } else {
      this.sizeArrayRows = this.bikRows + 3 * this.natoms + 6;
      this.sizeArrayCols = this.nvalues * ntypes + 1;
    }
  }

  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute pace requires a pair style be defined');
    if (this.sys.computes.filter((c) => c.style === 'pace').length > 1) this.sys.warn('More than one compute pace');
  }

  protected computeArray(): void {
    const sys = this.sys, s = sys.state, nb = sys.nb;
    const acc = sys.forces();
    // init: "Compute pace cutoff is longer than pairwise cutoff" (cutoffmax against the pair's cutforce)
    const paircut = nb.cutneighmax > 0 ? nb.cutneighmax - nb.skin : 0;
    if (this.basis.cutoffmax > paircut) throw new StyleError('Compute pace cutoff is longer than pairwise cutoff');
    const n = s.n, natoms = this.natoms;
    if (n !== natoms) throw new StyleError(`compute pace: the atom count changed since the compute was defined (${natoms} then, ${n} now)`);
    const ntypes = s.ntypes, nvalues = this.nvalues, bikRows = this.bikRows;
    const rows = this.sizeArrayRows, cols = this.sizeArrayCols, lastcol = cols - 1;
    const dgrad = this.dgradflag;
    const nall = nb.nall, xa = nb.xall, ta = nb.typeall, owner = nb.owner;
    const tag = (k: number): number => {
      const t = s.id[owner[k]];
      if (t < 1 || t > natoms) throw new StyleError(`compute pace: atom ID ${t} is outside 1..${natoms} (the array is indexed by atom ID)`);
      return t;
    };
    const pace = new Float64Array(rows * cols);
    const sizePeratom = 3 * nvalues * ntypes;
    const peratom = dgrad ? new Float64Array(0) : new Float64Array(nall * sizePeratom);
    const yoffset = nvalues, zoffset = 2 * nvalues;
    const list = nb.occasionalList(s, nb.cutneighmax, true, ntypes);
    const nbrs = this.natOrder.neighbors(s, nb, sys.geom, list);
    const gb = this.groupBit;
    const proj = this.proj;
    for (let i = 0; i < n; i++) {
      const irow = this.bikflag ? tag(i) - 1 : 0;
      if (!(s.mask[i] & gb)) continue;
      const itype = ta[i];
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      const jl = new Int32Array(jnum);
      for (let jj = 0; jj < jnum; jj++) jl[jj] = nbrs[k0 + jj] & NEIGHMASK;
      const ti = tag(i) - 1;
      if (dgrad) {
        for (let d = 0; d < 3; d++) {
          const r = bikRows + ti * 3 * natoms + 3 * ti + d;
          pace[r * cols] = ti; pace[r * cols + 1] = ti; pace[r * cols + 2] = d;
        }
        for (let j = 0; j < natoms; j++) {
          for (let d = 0; d < 3; d++) {
            const r = bikRows + j * 3 * natoms + 3 * ti + d;
            pace[r * cols] = ti; pace[r * cols + 1] = j; pace[r * cols + 2] = d;
          }
        }
      }
      proj.computeAtom(i, xa, ta, jnum, jl);
      if (proj.projections.length < nvalues) {
        throw new StyleError(`compute pace: species ${proj.basis.elementsName[0]} has more descriptors than the species of atom type ${itype}`);
      }
      const nDB = proj.neighboursDB;
      if (!dgrad) {
        const tl = 3 * nvalues * (itype - 1);
        const pi = i * sizePeratom + tl;
        for (let jj = 0; jj < jnum; jj++) {
          const pj = jl[jj] * sizePeratom + tl;
          for (let f = 0; f < nvalues; f++) {
            const o = (f * jnum + jj) * 3;
            const fx = nDB[o], fy = nDB[o + 1], fz = nDB[o + 2];
            peratom[pi + f] += fx;
            peratom[pi + f + yoffset] += fy;
            peratom[pi + f + zoffset] += fz;
            peratom[pj + f] -= fx;
            peratom[pj + f + yoffset] -= fy;
            peratom[pj + f + zoffset] -= fz;
          }
        }
        const tg = nvalues * (itype - 1);
        for (let f = 0; f < nvalues; f++) pace[irow * cols + tg + f] += proj.projections[f];
      } else {
        for (let jj = 0; jj < jnum; jj++) {
          const tj = tag(jl[jj]) - 1;
          const rj = bikRows + tj * 3 * natoms + 3 * ti;
          const ri = bikRows + ti * 3 * natoms + 3 * ti;
          for (let f = 0; f < nvalues; f++) {
            const o = (f * jnum + jj) * 3;
            for (let d = 0; d < 3; d++) {
              pace[(rj + d) * cols + f + 3] -= nDB[o + d];
              pace[(ri + d) * cols + f + 3] += nDB[o + d];
            }
          }
        }
        for (let f = 0; f < nvalues; f++) pace[irow * cols + 3 + f] += proj.projections[f];
      }
    }

    if (!dgrad) {
      // the gradients of every atom and ghost, folded onto the rows of its atom ID
      for (let it = 0; it < ntypes; it++) {
        const tl = 3 * nvalues * it, tg = nvalues * it;
        for (let c = 0; c < nvalues; c++) {
          for (let i = 0; i < nall; i++) {
            const p = i * sizePeratom + tl + c;
            const r = 3 * (tag(i) - 1) + 1;
            pace[r * cols + c + tg] += peratom[p];
            pace[(r + 1) * cols + c + tg] += peratom[p + yoffset];
            pace[(r + 2) * cols + c + tg] += peratom[p + zoffset];
          }
        }
      }
      for (let i = 0; i < n; i++) {
        const r = 3 * (tag(i) - 1) + 1;
        pace[r * cols + lastcol] = s.f[3 * i];
        pace[(r + 1) * cols + lastcol] = s.f[3 * i + 1];
        pace[(r + 2) * cols + lastcol] = s.f[3 * i + 2];
      }
      // dbdotr_compute: sum over atoms and ghosts of x times the gradient
      const irow0 = bikRows + 3 * natoms;
      for (let i = 0; i < nall; i++) {
        const x0 = xa[3 * i], x1 = xa[3 * i + 1], x2 = xa[3 * i + 2];
        for (let it = 0; it < ntypes; it++) {
          const tl = 3 * nvalues * it, tg = nvalues * it;
          for (let c = 0; c < nvalues; c++) {
            const p = i * sizePeratom + tl + c;
            const dbdx = peratom[p], dbdy = peratom[p + yoffset], dbdz = peratom[p + zoffset];
            const col = c + tg;
            pace[irow0 * cols + col] += dbdx * x0;
            pace[(irow0 + 1) * cols + col] += dbdy * x1;
            pace[(irow0 + 2) * cols + col] += dbdz * x2;
            pace[(irow0 + 3) * cols + col] += dbdz * x1;
            pace[(irow0 + 4) * cols + col] += dbdz * x0;
            pace[(irow0 + 5) * cols + col] += dbdy * x0;
          }
        }
      }
    } else {
      for (let i = 0; i < n; i++) {
        const r = tag(i) - 1;
        pace[r * cols] = s.f[3 * i];
        pace[r * cols + 1] = s.f[3 * i + 1];
        pace[r * cols + 2] = s.f[3 * i + 2];
      }
    }

    // the reference energy (thermo_pe) and virial (compute pressure NULL virial)
    const pe = acc.evdwl + acc.ecoul + acc.ebond + acc.eangle + acc.edihed + acc.eimp + acc.elong + sys.fixEnergy();
    if (!dgrad) {
      for (let i = 0; i < bikRows; i++) pace[i * cols + lastcol] = 0;
      pace[lastcol] = pe;
      const vol = sys.geom.volume(sys.dimension);
      const w = new Float64Array(6);
      const addv = (v: Float64Array): void => { for (let c = 0; c < 6; c++) w[c] += v[c]; };
      addv(acc.virial); addv(acc.vbond); addv(acc.vangle); addv(acc.vdihed); addv(acc.vimp); addv(acc.vlong);
      sys.fixVirial(w);
      const tail = sys.ff.ptail(vol);
      const press = new Float64Array(6);
      for (let c = 0; c < 6; c++) {
        let v = w[c] / vol;
        if (c < 3) v += tail;
        press[c] = v * s.units.nktv2p;
      }
      // pressure vector xx yy zz xy xz yz -> Voigt rows xx yy zz yz xz xy
      const voigt = [0, 1, 2, 5, 4, 3];
      const r0 = 3 * natoms + bikRows;
      for (let c = 0; c < 6; c++) pace[(r0 + c) * cols + lastcol] = press[voigt[c]];
    } else {
      pace[(bikRows + 3 * natoms * natoms) * cols] = pe;
    }
    this.array = pace;
  }
}
