import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseInt_ } from '../force/util';
import { rawBispectrum, WignerTables, type Cmat } from './sna';
import {
  parseSnaArgs, cutoffFactor, innerSwitch, cayleyKlein, quadraticTerms, checkCutoff, type SnaParams,
} from './snap_global';

/*
 * compute ID group-ID sna/grid grid nx ny nz rcutfac rfac0 twojmax R_1 ... w_1 ... keyword values ...
 * compute ID group-ID sna/grid/local grid nx ny nz rcutfac rfac0 twojmax R_1 ... w_1 ... keyword values ...
 * — docs.lammps.org/compute_sna_atom.html:
 *   "The compute sna/grid and sna/grid/local commands calculate bispectrum components for a regular grid
 *   of points. These are calculated from the local density of nearby atoms i' around each grid point, as
 *   if there was a central atom i at the grid point." "Neighbor atoms not in the group do not contribute
 *   to the bispectrum components of the grid points." The distance cutoff R_ii' assumes that i has the same type as the neighbor atom i' (doc).
 *   "The grid is aligned with the current box dimensions, with the first point at the box origin, and
 *   forming a regular 3d array with nx, ny, and nz points in the x, y, and z directions. For triclinic
 *   boxes, the array is congruent with the periodic lattice vectors a, b, and c."
 *   The array has one row for each of the nx x ny x nz grid points, with ix fastest, then iy, and iz slowest.
 *   sna/grid/local: "Each row of the array contains the global indexes ix, iy, and iz first, followed by
 *   the x, y, and z coordinates of the grid point, followed by the bispectrum components."
 *
 * Measured with native LAMMPS (black box): indices ix, iy, iz are 0-based; a grid point that coincides
 * with an atom reproduces the compute sna/atom values of that atom (the central self term is the
 * identity, and the atom at distance zero is not a neighbour); the grid is the same for the local and
 * the global array.
 *
 * Keywords other than the grid arguments are those of compute snap that apply to grid points: rmin0,
 * switchflag, bzeroflag, quadraticflag, bnormflag, wselfallflag, switchinnerflag, sinner, dinner.
 * bikflag, dgradflag are only for compute snap (doc); chem, nnn, wmode and delta are not implemented
 * here (StyleError, see snap_global.ts).
 */

interface GridSpec {
  nx: number;
  ny: number;
  nz: number;
  p: SnaParams;
}

const parseGrid = (id: string, style: string, args: string[], ntypes: number): GridSpec => {
  if (args.length < 4 || args[0] !== 'grid') {
    throw new StyleError(`compute ${id} (${style}): expects 'grid nx ny nz rcutfac rfac0 twojmax ...'`);
  }
  const dims = [1, 2, 3].map((k) => parseInt_(args[k], `compute ${id} (${style}) grid ${'nxyz'[k - 1]}`));
  for (const d of dims) if (!(d >= 1)) throw new StyleError(`compute ${id} (${style}): grid counts must be positive integers (got ${dims.join(' ')})`);
  const o = parseSnaArgs(id, style, args.slice(4), ntypes, false);
  return { nx: dims[0], ny: dims[1], nz: dims[2], p: o.p };
};

/** Bispectrum values (blk columns) of every grid point, row-major with ix fastest. */
class GridEvaluator {
  readonly points: Float64Array<ArrayBuffer>;
  readonly nPoints: number;
  constructor(private readonly sys: System, private readonly id: string, private readonly style: string, private readonly groupBit: number, private readonly spec: GridSpec) {
    this.nPoints = spec.nx * spec.ny * spec.nz;
    this.points = new Float64Array(this.nPoints * (3 + spec.p.blk)) as Float64Array<ArrayBuffer>;
  }

  /** Fills the rows: x y z followed by the blk bispectrum columns. */
  evaluate(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const p = this.spec.p;
    checkCutoff(this.id, this.style, p, nb);
    const { nx, ny, nz } = this.spec;
    const nall = nb.nall, xa = nb.xall, ta = nb.typeall, owner = nb.owner;
    const K = p.K, Q = p.Q, blk = p.blk, tj = p.twojmax, triples = p.triples;
    const T = new WignerTables(tj);
    const u: Cmat[] = [];
    for (let J = 0; J <= tj; J++) {
      const nn = (J + 1) * (J + 1);
      u.push({ re: new Float64Array(nn), im: new Float64Array(nn) });
    }
    const raw = new Float64Array(K), Bf = new Float64Array(K);
    const vals = new Float64Array(blk);
    const pos = [0, 0, 0];
    const row = 3 + blk;
    const gb = this.groupBit;
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          sys.geom.fromLamda(ix / nx, iy / ny, iz / nz, pos);
          const x = pos[0], y = pos[1], z = pos[2];
          for (let J = 0; J <= tj; J++) {
            u[J].re.fill(0); u[J].im.fill(0);
            for (let q = 0; q < J + 1; q++) u[J].re[q * (J + 1) + q] = 1;
          }
          for (let k = 0; k < nall; k++) {
            if (!(s.mask[owner[k]] & gb)) continue;
            const dx = xa[3 * k] - x, dy = xa[3 * k + 1] - y, dz = xa[3 * k + 2] - z;
            const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
            // the grid point has the type of its neighbour (doc: "i has the same type as the neighbor atom")
            const tk = ta[k] - 1;
            const Rii = p.rcutfac * (p.radius[tk] + p.radius[tk]);
            if (!(r < Rii) || r === 0) continue;
            const [fc] = cutoffFactor(p, r, Rii);
            let fin = 1;
            if (p.switchinner) fin = innerSwitch(r, p.sinner[tk], p.dinner[tk])[0];
            const sc = fc * fin * p.weight[tk];
            const ck = cayleyKlein(p, r, Rii, dx, dy, dz);
            T.compute(ck.ar, ck.ai, ck.br, ck.bi, false);
            for (let J = 0; J <= tj; J++) {
              const nn = (J + 1) * (J + 1);
              const Ur = T.ur[J], Ui = T.ui[J];
              for (let q = 0; q < nn; q++) {
                u[J].re[q] += sc * Ur[q];
                u[J].im[q] += sc * Ui[q];
              }
            }
          }
          rawBispectrum(triples, u, raw);
          for (let c = 0; c < K; c++) Bf[c] = (raw[c] - p.b0[c]) / p.norm[c];
          vals.fill(0);
          for (let c = 0; c < K; c++) vals[c] = Bf[c];
          if (Q) quadraticTerms(Bf, K, vals, K);
          const idx = ix + nx * (iy + ny * iz);
          const base = idx * row;
          this.points[base] = x;
          this.points[base + 1] = y;
          this.points[base + 2] = z;
          for (let c = 0; c < blk; c++) this.points[base + 3 + c] = vals[c];
        }
      }
    }
  }
}

export class ComputeSnaGrid extends Compute {
  readonly style = 'sna/grid';
  private readonly spec: GridSpec;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.spec = parseGrid(id, 'sna/grid', args, sys.state.ntypes);
    this.arrayFlag = true;
    this.sizeArrayRows = this.spec.nx * this.spec.ny * this.spec.nz;
    this.sizeArrayCols = 3 + this.spec.p.blk;
  }

  protected computeArray(): void {
    const ev = new GridEvaluator(this.sys, this.id, this.style, this.groupBit, this.spec);
    ev.evaluate();
    this.array = ev.points;
  }
}

export class ComputeSnaGridLocal extends Compute {
  readonly style = 'sna/grid/local';
  private readonly spec: GridSpec;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.spec = parseGrid(id, 'sna/grid/local', args, sys.state.ntypes);
    this.localFlag = true;
    this.sizeLocalCols = 6 + this.spec.p.blk;
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    const ev = new GridEvaluator(this.sys, this.id, this.style, this.groupBit, this.spec);
    ev.evaluate();
    // global indexes ix iy iz, then the coordinates and the bispectrum components of the sna/grid row
    const { nx, ny } = this.spec;
    const blk = this.spec.p.blk;
    const cols = 6 + blk;
    const out = new Float64Array(ev.nPoints * cols);
    for (let idx = 0; idx < ev.nPoints; idx++) {
      const ix = idx % nx;
      const iy = Math.floor(idx / nx) % ny;
      const iz = Math.floor(idx / (nx * ny));
      const o = idx * cols;
      out[o] = ix;
      out[o + 1] = iy;
      out[o + 2] = iz;
      for (let c = 0; c < 3 + blk; c++) out[o + 3 + c] = ev.points[idx * (3 + blk) + c];
    }
    this.localRows = ev.nPoints;
    return out;
  }
}
