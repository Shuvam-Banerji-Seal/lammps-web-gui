/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/PTM/compute_ptm_atom.cpp and
 * compute_ptm_atom.h (compute ptm/atom, contributing author PM Larsen, MIT PTM library). Copyright (2003) Sandia
 * Corporation, GNU General Public License version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Documentation: docs.lammps.org/compute_ptm_atom.html. Argument grammar as the C constructor: structures (fcc, hcp,
 * bcc, ico, sc, dcub, dhex, graphene, all, default, joined by '-'), the RMSD threshold (0 means no threshold), and an
 * optional group2-ID that restricts the neighbours. Output: 7 columns per atom (type, rmsd, interatomic distance,
 * quaternion w x y z), with -1 for atoms outside the group (and atoms with no neighbours in the list), and 0 (other)
 * for atoms whose rmsd exceeds the threshold or that match nothing.
 *
 * Neighbours: the C get_neighbours() reads the full neighbour list (pair cutoff + skin, per type pair) and sorts it
 * by squared distance with std::sort on the key d alone; that sort is reproduced with libstdcxx_sort.ts. The
 * candidate order before the sort is ascending xall index (the engine's cell list), not native's list order, so
 * equal-distance neighbours at the cutoff boundary can differ from native (measured in the w201pt oracle header).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { CellList } from '../../engine/cellList';
import {
  PTM_CHECK_BCC, PTM_CHECK_DCUB, PTM_CHECK_DHEX, PTM_CHECK_FCC, PTM_CHECK_GRAPHENE, PTM_CHECK_HCP, PTM_CHECK_ICO,
  PTM_CHECK_SC, PTM_CHECK_ALL, PTM_CHECK_DEFAULT, PTM_MATCH_NONE,
} from './ptm_constants';
import type { GetNeighbours } from './ptm_types';
import { ptm_index } from './ptm_index';
import { ptm_initialize_global, ptm_initialize_local, ptm_uninitialize_local } from './ptm_initialize_data';
import { std_sort } from './libstdcxx_sort';

/** compute_ptm_atom.cpp: NUM_COLUMNS */
const NUM_COLUMNS = 7;
/** compute_ptm_atom.cpp: PTM_LAMMPS_UNKNOWN and PTM_LAMMPS_OTHER */
const PTM_LAMMPS_UNKNOWN = -1;
const PTM_LAMMPS_OTHER = 0;

/** The C structure-name table (compute_ptm_atom.cpp constructor), in the C order of the prefix test. */
const STRINGS = ['fcc', 'hcp', 'bcc', 'ico', 'sc', 'dcub', 'dhex', 'graphene', 'all', 'default'];
const FLAGS = [
  PTM_CHECK_FCC,
  PTM_CHECK_HCP,
  PTM_CHECK_BCC,
  PTM_CHECK_ICO,
  PTM_CHECK_SC,
  PTM_CHECK_DCUB,
  PTM_CHECK_DHEX,
  PTM_CHECK_GRAPHENE,
  PTM_CHECK_ALL,
  PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_BCC | PTM_CHECK_ICO,
];

/** The structure keyword list ('-' separated, each matched by prefix as in the C loop). */
export function parsePtmStructures(text: string): number {
  let flags = 0;
  let ptr = 0;
  while (ptr < text.length) {
    let found = false;
    for (let i = 0; i < STRINGS.length; i++) {
      const name = STRINGS[i];
      if (text.startsWith(name, ptr)) {
        flags |= FLAGS[i];
        ptr += name.length;
        found = true;
        break;
      }
    }
    if (!found) throw new StyleError('Illegal compute ptm/atom command (invalid structure type)');
    if (ptr >= text.length) break;
    if (text[ptr] !== '-') throw new StyleError('Illegal compute ptm/atom command (invalid structure type)');
    ptr++;
  }
  return flags;
}

interface NbrRec { index: number; d: number }

export class ComputePtmAtom extends Compute {
  readonly style = 'ptm/atom';
  peratomFlag = true;
  private readonly inputFlags: number;
  private readonly rmsdThreshold: number;
  private readonly group2Bit: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2 || args.length > 3) throw new StyleError('Illegal compute ptm/atom command');
    this.inputFlags = parsePtmStructures(args[0]);
    const threshold = parseNum(args[1], `compute ${id} (ptm/atom) threshold`);
    if (threshold < 0.0) throw new StyleError('Illegal compute ptm/atom command (threshold is negative)');
    this.rmsdThreshold = threshold === 0 ? Infinity : threshold;
    this.group2Bit = args.length > 2 ? sys.groups.bit(args[2]) : sys.groups.bit('all');
    this.sizePeratomCols = NUM_COLUMNS;
  }

  protected computePeratom(): void {
    const sys = this.sys;
    if (!sys.ff.pair) throw new StyleError('Compute ptm/atom requires a pair style be defined');
    sys.forces();
    const s = sys.state;
    const nb = sys.nb;
    const pair = sys.ff.pair;
    const nt = pair.ntypes + 1;
    const skin = nb.skin;

    // largest neighbour-list cutoff (pair cutoff + skin) over the type pairs: the cell size and the ghost check
    let cutmax = 0;
    for (let k = 0; k < pair.cut.length; k++) if (pair.cut[k] > cutmax) cutmax = pair.cut[k];
    const cutlist = cutmax + skin;
    if (nb.cutghost > 0 && nb.cutghost < cutlist - 1e-9) {
      throw new StyleError(`compute ${this.id} (ptm/atom): cutoff ${cutlist} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
    }

    const n = s.n;
    const xa = nb.xall;
    const nall = nb.nall;
    const cells = new CellList(xa, nall, cutlist);
    const out = (this.arrayAtom = new Float64Array(NUM_COLUMNS * n));

    /** The full-list neighbours of xall atom c (self excluded), membership by the per-type-pair neighbour cutoff. */
    const listOf = (c: number): number[] => {
      const res: number[] = [];
      const xc = xa[3 * c], yc = xa[3 * c + 1], zc = xa[3 * c + 2];
      const tc = s.type[nb.owner[c]];
      const cand = cells.candidates(xc, yc, zc);
      for (let t = 0; t < cand.length; t++) {
        const k = cand[t];
        if (k === c) continue;
        const dx = xa[3 * k] - xc, dy = xa[3 * k + 1] - yc, dz = xa[3 * k + 2] - zc;
        const rsq = dx * dx + dy * dy + dz * dz;
        const tk = s.type[nb.owner[k]];
        const cl = pair.cut[tc * nt + tk] + skin;
        if (rsq < cl * cl) res.push(k);
      }
      return res;
    };

    // get_neighbours(): the central atom's list for ghosts, the atom's own list otherwise; sorted by distance
    const get_neighbours: GetNeighbours = (_vdata, central_index, atom_index, num, nbr_indices, numbers, nbr_pos) => {
      const src = atom_index < n ? atom_index : central_index;
      const px = xa[3 * atom_index], py = xa[3 * atom_index + 1], pz = xa[3 * atom_index + 2];
      const order: NbrRec[] = [];
      const jlist = listOf(src);
      for (let jj = 0; jj < jlist.length; jj++) {
        const j = jlist[jj];
        if (!(s.mask[nb.owner[j]] & this.group2Bit)) continue;
        if (j === atom_index) continue;
        const dx = px - xa[3 * j], dy = py - xa[3 * j + 1], dz = pz - xa[3 * j + 2];
        order.push({ index: j, d: dx * dx + dy * dy + dz * dz });
      }
      std_sort(order, 0, order.length, (a, b) => a.d < b.d);
      const numNbrs = Math.min(num - 1, order.length);

      nbr_pos[0][0] = nbr_pos[0][1] = nbr_pos[0][2] = 0;
      nbr_indices[0] = atom_index;
      numbers[0] = 0;
      for (let jj = 0; jj < numNbrs; jj++) {
        const j = order[jj].index;
        nbr_pos[jj + 1][0] = xa[3 * j] - px;
        nbr_pos[jj + 1][1] = xa[3 * j + 1] - py;
        nbr_pos[jj + 1][2] = xa[3 * j + 2] - pz;
        nbr_indices[jj + 1] = j;
        numbers[jj + 1] = 0;
      }
      return numNbrs + 1;
    };

    ptm_initialize_global();
    const handle = ptm_initialize_local();
    for (let i = 0; i < n; i++) {
      out[i * NUM_COLUMNS] = PTM_LAMMPS_UNKNOWN;
      if (!(s.mask[i] & this.groupBit)) continue;
      if (listOf(i).length <= 0) continue;

      const r = ptm_index(handle, i, get_neighbours, null, this.inputFlags);
      let type = r.p_type;
      const rmsd = r.p_rmsd;
      if (rmsd > this.rmsdThreshold) type = PTM_MATCH_NONE;
      if (type === PTM_MATCH_NONE) type = PTM_LAMMPS_OTHER;

      out[i * NUM_COLUMNS] = type;
      out[i * NUM_COLUMNS + 1] = rmsd;
      out[i * NUM_COLUMNS + 2] = r.p_interatomic_distance;
      out[i * NUM_COLUMNS + 3] = r.q[0];
      out[i * NUM_COLUMNS + 4] = r.q[1];
      out[i * NUM_COLUMNS + 5] = r.q[2];
      out[i * NUM_COLUMNS + 6] = r.q[3];
    }
    ptm_uninitialize_local(handle);
  }
}
