/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MACHDYN/compute_smd_triangle_vertices.cpp
 * and .h (ComputeSMDTriangleVertices), compute_smd_tlsph_defgrad.cpp and .h (ComputeSMDTLSPHDefgrad) and
 * compute_smd_hourglass_error.cpp and .h (ComputeSMDHourglassError). MACHDYN author (from the sources): Georg C.
 * Ganzenmueller, Fraunhofer EMI. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The triangle vertices and the deformation gradient are the 9 per-atom values of atom_style smd (SMD_DEFGRAD, the
 * smd_data_9 of the source); the hourglass error is the per-atom array of the tlsph pair style (its extract
 * "smd/tlsph/hourglass_error_ptr").
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isSmdStyle, SMD_DEFGRAD } from '../../engine/smd';
import { PairSMDTlsph } from '../machdyn/pair_smd_tlsph';
import { det3 } from '../machdyn/smd_math';

/** The triangle particles of atom_style smd have a molecule ID of at least 65535 (compute_smd_triangle_vertices.cpp). */
const TRI_MOL_MIN = 65535;

/** compute ID group-ID smd/triangle/vertices: the 9 vertex coordinates of each triangle atom, zeros for the others. */
export class ComputeSMDTriangleVertices extends Compute {
  readonly style: string = 'smd/triangle/vertices';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 0) throw new StyleError('Illegal compute smd/triangle/vertices command');
    this.peratomFlag = true;
    this.sizePeratomCols = 9;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const sm = s.custom.get(SMD_DEFGRAD)!.data;
    const out = new Float64Array(9 * s.n);
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && s.molecule[i] >= TRI_MOL_MIN) {
        for (let k = 0; k < 9; k++) out[9 * i + k] = sm[9 * i + k];
      }
    }
    this.arrayAtom = out;
  }
}

/** compute ID group-ID smd/tlsph/defgrad: F (9 values, row-major) and det F per atom; identity for atoms outside the group. */
export class ComputeSMDTLSPHDefgrad extends Compute {
  readonly style: string = 'smd/tlsph/defgrad';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 0) throw new StyleError('Illegal compute smd/tlsph/defgrad command');
    this.peratomFlag = true;
    this.sizePeratomCols = 10;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const F = s.custom.get(SMD_DEFGRAD)!.data;
    const out = new Float64Array(10 * s.n);
    for (let i = 0; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) {
        const m = [F[9 * i], F[9 * i + 1], F[9 * i + 2], F[9 * i + 3], F[9 * i + 4], F[9 * i + 5], F[9 * i + 6], F[9 * i + 7], F[9 * i + 8]];
        for (let k = 0; k < 9; k++) out[10 * i + k] = m[k];
        out[10 * i + 9] = det3(m);
      } else {
        out.set([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], 10 * i);
      }
    }
    this.arrayAtom = out;
  }
}

/** compute ID group-ID smd/hourglass/error: the hourglass error of the tlsph pair style per atom (zero outside the group). */
export class ComputeSMDHourglassError extends Compute {
  readonly style: string = 'smd/hourglass/error';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 0) throw new StyleError('Illegal compute smd/hourglass_error command');
    if (!isSmdStyle(sys.state.atomStyle)) throw new StyleError('compute smd/hourglass/error command requires atom_style with hourglass_error');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const p = this.sys.ff.pair;
    if (!(p instanceof PairSMDTlsph)) throw new StyleError('compute smd/hourglass/error failed to access hourglass_error array');
    const hg = p.work.hourglass;
    const out = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) out[i] = (s.mask[i] & this.groupBit) ? hg[i] : 0.0;
    this.vectorAtom = out;
  }
}
