import { Compute } from './compute';
import { ComputeSnap } from './snap_global';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseMliapSnaDescriptor, type MliapSnaDescriptor } from '../mliap/descriptor';

/*
 * compute mliap (wave 16) — docs.lammps.org/compute_mliap.html (plans/lammps-docs/compute_mliap.rst).
 *
 * Syntax (verbatim): "compute ID group-ID mliap ... keyword values ..."; "two or more keyword/value
 * pairs must be appended"; keyword = model, descriptor or gradgradflag. Implemented: "model linear"
 * and "model quadratic" (the doc lists linear, quadratic and mliappy; mliappy needs Python and is a
 * StyleError), "descriptor sna <file>" (the doc lists sna, ace; ace is a StyleError), and
 * "gradgradflag 0/1" (see the note below).
 *
 * Doc, Output info (paraphrase): the array has nelems blocks of nparams columns, in element order, and a
 * final column with the energy, force component or virial stress component. Rows: 1 row of derivatives
 * of the potential energy; then 3N force-derivative rows (x, y, z of each atom, atoms sorted by ID); then
 * 6 virial-derivative rows in Voigt order pxx, pyy, pzz, pyz, pxz, pxy.
 *
 * Implementation: the array is the compute snap array (src/engine/compute/snap_global.ts) of the same
 * descriptor, with one constant column per element block inserted in front of the descriptor
 * columns. Its value is the derivative of the model with respect to its constant term beta_0, that
 * is the number of atoms of the element in the group (measured with native LAMMPS, black box: row 1
 * starts with 1 for a single atom). The descriptor columns are the compute snap rows (bispectrum,
 * quadratic terms when model quadratic) and the last column is the reference potential energy,
 * force or virial of the pair style, as in compute snap. The element of atom type t is the element
 * t of the descriptor file, so nelems must equal the number of atom types (doc, Note). Measured with
 * native LAMMPS (black box): the virial rows are the derivatives of the virial sum r.f itself (no
 * division by the volume and no nktv2p factor), which is the convention of compute snap's snav rows.
 *
 * gradgradflag (doc): "A value of 1 requires that the model provide the matrix of double gradients
 * ... A value of 0 requires that the descriptor provide the derivative of the descriptors with
 * respect to the position of every neighbor atom." For the linear and quadratic models both
 * formulations give the same array; this engine uses the descriptor derivative for both values.
 * The measured arrays of both gradgradflag values agree with native LAMMPS on the probes used.
 */

const MODEL_STYLES = ['linear', 'quadratic'];

export class ComputeMliap extends Compute {
  readonly style = 'mliap';
  private readonly inner: ComputeSnap;
  private readonly blk: number;
  private readonly ntypes: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const nt = sys.state.ntypes;
    this.ntypes = nt;
    let kind: string | null = null, desc: MliapSnaDescriptor | null = null;
    for (let k = 0; k < args.length; k += 2) {
      const kw = args[k], v = args[k + 1];
      if (v === undefined) throw new StyleError(`compute ${id} (mliap): keyword '${kw}' needs a value`);
      if (kw === 'model') {
        if (v === 'mliappy') throw new StyleError(`compute ${id} (mliap): model mliappy needs the Python module and is not implemented in this engine`);
        if (!MODEL_STYLES.includes(v)) throw new StyleError(`compute ${id} (mliap): model style '${v}' is not implemented (linear, quadratic)`);
        kind = v;
      } else if (kw === 'descriptor') {
        const file = args[k + 2];
        if (file === undefined) throw new StyleError(`compute ${id} (mliap): descriptor needs a style and a filename`);
        if (v !== 'sna') throw new StyleError(`compute ${id} (mliap): descriptor style '${v}' is not implemented in this engine (sna is)`);
        desc = parseMliapSnaDescriptor(sys.readFile(file), file);
        k += 1;
      } else if (kw === 'gradgradflag') {
        if (v !== '0' && v !== '1') throw new StyleError(`compute ${id} (mliap): gradgradflag must be 0 or 1 (got '${v}')`);
      } else {
        throw new StyleError(`compute ${id} (mliap): unknown keyword '${kw}'`);
      }
    }
    if (kind === null || desc === null) throw new StyleError(`compute ${id} (mliap) needs the keywords model and descriptor`);
    if (desc.elems.length !== nt) {
      throw new StyleError(`compute ${id} (mliap): nelems ${desc.elems.length} of the descriptor must match the ${nt} atom types`);
    }
    const quadratic = kind === 'quadratic';
    const snapArgs: string[] = [String(desc.rcutfac), String(desc.rfac0), String(desc.twojmax)];
    for (let t = 0; t < nt; t++) snapArgs.push(String(desc.radius[t]));
    for (let t = 0; t < nt; t++) snapArgs.push(String(desc.weight[t]));
    snapArgs.push('rmin0', String(desc.rmin0), 'switchflag', desc.switchflag ? '1' : '0', 'bzeroflag', desc.bzeroflag ? '1' : '0', 'bnormflag', desc.bnormflag ? '1' : '0', 'quadraticflag', quadratic ? '1' : '0');
    this.inner = new ComputeSnap(sys, id, group, snapArgs);
    const K = desc.K;
    const Q = quadratic ? (K * (K + 1)) / 2 : 0;
    this.blk = K + Q + 1; // constant term, bispectrum, quadratic terms
    this.arrayFlag = true;
    this.sizeArrayCols = nt * this.blk + 1;
  }

  protected computeArray(): void {
    const src = this.inner.arrayValues();
    const rows = this.inner.sizeArrayRows, scols = this.inner.sizeArrayCols;
    const nt = this.ntypes, blk = this.blk, ocols = nt * blk + 1;
    const s = this.sys.state;
    const count = new Float64Array(nt);
    for (let i = 0; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) count[s.type[i] - 1]++;
    }
    const cb = blk - 1; // descriptor columns per element block
    const out = new Float64Array(rows * ocols);
    for (let r = 0; r < rows; r++) {
      for (let t = 0; t < nt; t++) {
        out[r * ocols + t * blk] = r === 0 ? count[t] : 0;
        for (let c = 0; c < cb; c++) out[r * ocols + t * blk + 1 + c] = src[r * scols + t * cb + c];
      }
      out[r * ocols + ocols - 1] = src[r * scols + scols - 1];
    }
    this.array = out;
    this.sizeArrayRows = rows;
    this.sizeArrayCols = ocols;
  }
}
