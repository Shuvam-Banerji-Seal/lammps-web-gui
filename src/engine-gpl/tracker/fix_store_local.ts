/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/STORE/fix_store_local.cpp and fix_store_local.h at commit b332f28 (the LAMMPS developers; Copyright
 * (2003) Sandia Corporation). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The internal fix STORE/LOCAL that pair_style tracker creates for its local output (the fix ID the style names).
 */
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';

/**
 * fix STORE/LOCAL N M (fix_store_local.cpp): add_data appends one row of M values from the pair style during the
 * force evaluation; post_force (every step) takes the row count at steps that are multiples of N and starts the
 * next window, so the local output is the rows of the last N steps. The rows keep their order of insertion.
 */
export class FixStoreLocal extends Fix {
  readonly style = 'STORE/LOCAL';
  readonly nreset: number;
  readonly nvalues: number;
  private ncount = 0;
  private data = new Float64Array(0);

  constructor(sys: System, id: string, nreset: number, nvalues: number) {
    super(sys, id, 'all', []);
    this.nreset = nreset;
    this.nvalues = nvalues;
    this.localFlag = true;
    this.sizeLocalCols = nvalues === 1 ? 0 : nvalues;
  }

  /** Appends one row (length nvalues) (fix_store_local.cpp add_data). */
  addData(row: Float64Array): void {
    const need = (this.ncount + 1) * this.nvalues;
    if (need > this.data.length) {
      const grown = new Float64Array(Math.max(need, 2 * this.data.length));
      grown.set(this.data);
      this.data = grown;
    }
    this.data.set(row.subarray(0, this.nvalues), this.ncount * this.nvalues);
    this.ncount++;
  }

  /** Appends one row given as plain numbers (the bpm bond styles' broken-bond rows, bond_bpm.cpp store_data). */
  addRow(values: readonly number[]): void {
    this.addData(Float64Array.from(values));
  }

  /** post_force: "if (update->ntimestep % nreset == 0) { size_local_rows = ncount; ncount = 0; }". */
  postForce(): void {
    if (this.sys.state.step % this.nreset === 0) {
      this.localRows = this.ncount;
      this.ncount = 0;
    }
  }

  localValues(): Float64Array<ArrayBuffer> {
    return this.data.subarray(0, this.localRows * this.nvalues) as Float64Array<ArrayBuffer>;
  }
}
