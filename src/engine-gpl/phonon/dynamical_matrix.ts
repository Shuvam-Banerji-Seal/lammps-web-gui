/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/PHONON/dynamical_matrix.cpp and
 * dynamical_matrix.h (class DynamicalMatrix, command dynamical_matrix group-ID regular|eskm del keyword values: the
 * dynamical matrix of a group of atoms by central finite differences of the forces, one displacement of +del and one
 * of -del per Cartesian direction, written as text). Contributing author: Charlie Sievers (UC Davis).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/dynamical_matrix.html): "the dynamical matrix is computed from the forces
 * of displaced atoms", the eskm style converts the result to the units of the dynamical matrix in
 * "conversion = conv_energy/conv_distance/conv_mass", and the keywords are file, fold and binary. The text output
 * holds, for each atom of the group, three lines per Cartesian direction of the atom, each with 3 * (group size)
 * values in "%4.8f" format.
 *
 * Browser scope: the binary and gzip-compressed outputs are native-only and are refused by name; the eskm
 * conversion is given for the metal and real unit styles (the ones the examples use).
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { massOf } from '../../engine/atoms';
import type { SimState } from '../../engine/types';

/** Energy, distance and mass conversions of the eskm style, in the units of the dynamical matrix. */
const ESKM_CONVERSION: Record<string, [number, number, number]> = {
  real: [418.4, 1, 1],
  metal: [9648.5, 1, 1],
};

/** Output text: "%4.8f" values, three per line, each Cartesian row of an atom on its own lines. */
const fmt = (v: number): string => v.toFixed(8);

export const dynamicalMatrixCmd: Handler = ({ sys }, a) => {
  if (a.length < 3) throw new StyleError('Illegal dynamical_matrix command');
  const [group, style] = a;
  if (style !== 'regular' && style !== 'eskm') throw new StyleError('Illegal Dynamical_matrix command');
  const del = parseNum(a[2], 'dynamical_matrix del');
  let file = 'dynmat.dyn';
  let folded = false;
  for (let k = 3; k < a.length;) {
    const key = a[k];
    const val = a[k + 1];
    // PHONON/dynamical_matrix.cpp options (native at b332f28): a keyword without its value, or an unknown word, is 'Illegal dynamical_matrix command'
    if (val === undefined) throw new StyleError('Illegal dynamical_matrix command');
    if (key === 'binary') {
      if (val === 'gzip') throw new StyleError('dynamical_matrix binary gzip output is native-only; the browser engine writes text only');
      if (val === 'yes') throw new StyleError('dynamical_matrix binary yes output is native-only; the browser engine writes text only (binary no)');
      if (val !== 'no') throw new StyleError(`Illegal dynamical_matrix binary value ${val}`);
    } else if (key === 'file') {
      file = val;
    } else if (key === 'fold') {
      if (val !== 'yes' && val !== 'no') throw new StyleError('Illegal input for dynamical_matrix fold option');
      folded = val === 'yes';
    } else {
      throw new StyleError('Illegal dynamical_matrix command');
    }
    k += 2;
  }
  // conversion: 1 for regular; for eskm the unit style's conversion (dynamical_matrix.cpp convert_units)
  let conversion = 1;
  if (style === 'eskm') {
    const ust = sys.units.style;
    if (ust === 'lj') throw new StyleError('Conversion Not Set');
    const c = ESKM_CONVERSION[ust];
    if (!c) throw new StyleError(`Units Type Conversion Not Found for dynamical_matrix eskm in unit style ${ust} (the browser engine has metal and real)`);
    conversion = c[0] / c[1] / c[2];
  }
  const bit = sys.groups.bit(group);
  // the state object is read again after the neighbour setup (it may be replaced there)
  let s: SimState = sys.state;
  const natoms = s.n;
  // the group map: the rank of each group atom among the group atoms in ID order (gm[id - 1], -1 outside the group)
  const idxOf = new Map<number, number>();
  for (let i = 0; i < natoms; i++) idxOf.set(s.id[i], i);
  const ids = Array.from(idxOf.keys()).sort((p, q) => p - q);
  const gm = new Int32Array(natoms).fill(-1);
  let gcount = 0;
  for (const id of ids) {
    const i = idxOf.get(id)!;
    if (s.mask[i] & bit) gm[id - 1] = gcount++;
  }
  if (gcount < 1) throw new StyleError(`Could not find dynamical matrix group ID ${group}`);
  const dynlen = gcount * 3;
  const dynlenb = folded ? natoms * 3 : dynlen;
  sys.setupNeighbors();
  s = sys.state;
  // the periodic images of a displaced atom move with it (native displace_atom walks the image chain): the ghost
  // coordinates are refreshed from the local ones before the forces are evaluated
  const force = (): Float64Array => {
    sys.nb.forwardComm(sys.state, sys.geom);
    sys.computeForcesInRun(sys.computeFlags());
    return sys.state.f;
  };
  const displace = (li: number, dir: number, magnitude: number): void => {
    if (li < 0) return;
    sys.state.x[3 * li + dir] += del * magnitude;
  };
  const lines: string[] = [];
  for (let i = 1; i <= natoms; i++) {
    const li = idxOf.get(i) ?? -1;
    if (gm[i - 1] < 0) continue;
    const dyn = [new Float64Array(dynlenb), new Float64Array(dynlenb), new Float64Array(dynlenb)];
    for (let alpha = 0; alpha < 3; alpha++) {
      displace(li, alpha, 1);
      let f = force();
      for (let j = 1; j <= natoms; j++) {
        const lj = idxOf.get(j) ?? -1;
        if (li >= 0 && lj >= 0 && (gm[j - 1] >= 0 || folded)) {
          const col = folded ? (j - 1) * 3 : gm[j - 1] * 3;
          for (let beta = 0; beta < 3; beta++) dyn[alpha][col + beta] -= f[3 * lj + beta];
        }
      }
      displace(li, alpha, -2);
      f = force();
      for (let j = 1; j <= natoms; j++) {
        const lj = idxOf.get(j) ?? -1;
        if (li >= 0 && lj >= 0 && (gm[j - 1] >= 0 || folded)) {
          const imass = Math.sqrt(massOf(s, li) * massOf(s, lj));
          const col = folded ? (j - 1) * 3 : gm[j - 1] * 3;
          for (let beta = 0; beta < 3; beta++) {
            dyn[alpha][col + beta] -= -f[3 * lj + beta];
            dyn[alpha][col + beta] /= 2 * del * imass;
            dyn[alpha][col + beta] *= conversion;
          }
        }
      }
      displace(li, alpha, 1);
    }
    for (let k = 0; k < 3; k++) {
      let line = '';
      for (let j = 0; j < dynlenb; j++) {
        line += (j + 1) % 3 === 0 ? `${fmt(dyn[k][j])}\n` : `${fmt(dyn[k][j])} `;
      }
      lines.push(line);
    }
  }
  // As native LAMMPS does (dynamical_matrix.cpp): the positions are back in place, but the stored forces stay those of
  // the last update_force (the last atom displaced by -del); a write_dump right after the command prints them
  sys.bump();
  sys.forcesCurrent();
  sys.writeFile(file, lines.join(''), false);
};
