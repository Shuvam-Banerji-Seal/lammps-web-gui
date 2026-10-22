import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * compute ID group-ID property/grid Nx Ny Nz input1 input2 ...
 * — docs.lammps.org/compute_property_grid.html (plans/lammps-docs/compute_property_grid.rst).
 * Quoted sentences are copied character for character from that page.
 *
 *   "Define a computation that stores the specified attributes of a distributed grid."
 *   "Nx, Ny, and Nz define the size of the grid.  For a 2d simulation Nz must be 1."
 *   "The id attribute is the grid ID for each grid cell.  For a global grid of
 *    size Nx by Ny by Nz (in 3d simulations) the grid IDs range from 1 to
 *    Nx*Ny*Nz.  They are ordered with the X index of the 3d grid varying
 *    fastest, then Y, then Z slowest.  For 2d grids (in 2d simulations), the
 *    grid IDs range from 1 to Nx*Ny, with X varying fastest and Y slowest."
 *   "The proc attribute is the ID of the processor which owns the grid cell.
 *    Processor IDs range from 0 to Nprocs - 1" — the browser engine is a single
 *    process, so proc is 0 for every cell.
 *   "The ix, iy, iz attributes are the indices of a grid cell in each
 *    dimension.  They range from 1 to Nx inclusive in the X dimension, and
 *    similar for Y and Z."
 *   "The x, y, z attributes are the coordinates of the lower left corner point
 *    of each grid cell."
 *   "The xs, ys, zs attributes are also coordinates of the lower left corner
 *    point of each grid cell, except in scaled coordinates, where the
 *    lower-left corner of the entire simulation box is (0,0,0) and the upper
 *    right corner is (1,1,1)."
 *   "The xc, yc, zc attributes are the coordinates of the center point of each
 *    grid cell."
 *   "The xsc, ysc, zsc attributes are also coordinates of the center point each
 *    grid cell, except in scaled coordinates"
 *   "For triclinic simulation boxes ... the grid point coordinates for (x,y,z)
 *    and (xc,yc,zc) will reflect the triclinic geometry.  For (xs,yz,zs) and
 *    (xsc,ysc,zsc), the coordinates are the same for orthogonal versus
 *    triclinic boxes."
 *   "This compute calculates a per-grid vector or array depending on the
 *    number of input values.  The length of the vector or number of array rows
 *    (distributed across all processors) is Nx * Ny * Nz.  For access by other
 *    commands, the name of the single grid produced by this command is "grid".
 *    The name of its per-grid data is "data"."
 *   "For 2d simulations, the attributes which refer to the Z dimension cannot
 *    be used."
 *
 * Measured with native LAMMPS (black box, plans/scratch/w26cgrid/probe_grid.in,
 * box 0..2 x 0..4 x 0..6, grid 2x2x3, dumped with dump grid sort 1): cell 1 is
 * id 1, ix 1, iy 1, iz 1, lower-left (0,0,0), center (0.5,1,1); cell 2 (x
 * fastest) is ix 2, lower-left (1,0,0), center (1.5,1,1); the scaled lower-left
 * corner is (ix-1)/Nx and the scaled center is (ix-0.5)/Nx; the grid id is
 * ix + Nx*(iy-1) + Nx*Ny*(iz-1).
 *
 * The per-grid data is read by dump grid and fix ave/grid as c_ID:grid:data (grid.ts), and here through
 * gridValue(cell, col) / gridVector(), with the cell index 0..Nx*Ny*Nz-1 in the same x-fastest order as the grid id.
 * As native LAMMPS does (src/compute_property_grid.cpp compute_pergrid and pack_coords at b332f28): the values are
 * filled at every invocation (the box may have changed); an unscaled coordinate of an orthogonal box is
 * boxlo + i * (prd / N) (the center (i + 0.5) times that spacing), a scaled one i * (1 / N), and a triclinic
 * unscaled one maps the scaled point through h (h x lamda, then boxlo added last).
 */

/** Every attribute the page documents, in the syntax order. */
const ATTRS = ['id', 'proc', 'ix', 'iy', 'iz', 'x', 'y', 'z', 'xs', 'ys', 'zs', 'xc', 'yc', 'zc', 'xsc', 'ysc', 'zsc'] as const;
type Attr = (typeof ATTRS)[number];
/** "For 2d simulations, the attributes which refer to the Z dimension cannot be used." */
const Z_ATTRS = new Set<Attr>(['iz', 'z', 'zs', 'zc', 'zsc']);

export class ComputePropertyGrid extends Compute {
  readonly style = 'property/grid';
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly ncell: number;
  readonly attrs: Attr[];
  private readonly data: Float64Array;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `compute ${id} property/grid`;
    if (args.length < 4) throw new StyleError(`${what}: missing argument(s): Nx Ny Nz input1 ...`);
    const dim = (w: string | undefined, name: string): number => {
      const v = Number(w);
      if (!Number.isInteger(v) || v < 1) throw new StyleError(`${what}: ${name} must be a positive integer, got '${w}'`);
      return v;
    };
    this.nx = dim(args[0], 'Nx');
    this.ny = dim(args[1], 'Ny');
    this.nz = dim(args[2], 'Nz');
    if (sys.dimension === 2 && this.nz !== 1) throw new StyleError(`${what}: for 2d simulations Nz must be 1`);
    this.ncell = this.nx * this.ny * this.nz;

    const attrs: Attr[] = [];
    for (let k = 3; k < args.length; k++) {
      const w = args[k] as Attr;
      if (!(ATTRS as readonly string[]).includes(w)) {
        // As native LAMMPS does (src/compute_property_grid.cpp ComputePropertyGrid::ComputePropertyGrid at b332f28): "Unknown compute property/grid keyword: {}"
        throw new StyleError(`Unknown compute property/grid keyword: ${args[k]}`);
      }
      if (sys.dimension === 2 && Z_ATTRS.has(w)) {
        throw new StyleError(`${what}: attribute '${w}' refers to the Z dimension and cannot be used in a 2d simulation`);
      }
      attrs.push(w);
    }
    this.attrs = attrs;
    this.data = new Float64Array(this.ncell * attrs.length);
    this.pergridFlag = true;
    this.fill();
  }

  /** "the name of the single grid produced by this command is "grid". The name of its per-grid data is "data"." */
  gridByName(name: string): number { return name === 'grid' ? 0 : -1; }
  gridSize(_igrid: number): [number, number, number] { return [this.nx, this.ny, this.nz]; }
  griddataByName(igrid: number, name: string): { index: number; ncol: number } | null {
    return igrid === 0 && name === 'data' ? { index: 0, ncol: this.attrs.length === 1 ? 0 : this.attrs.length } : null;
  }
  griddata(_index: number): Float64Array { return this.data; }
  protected computePergrid(): void { this.fill(); }

  /** Number of per-grid columns (1 = per-grid vector, >1 = per-grid array). */
  get ncols(): number { return this.attrs.length; }

  /** The whole per-grid data, row-major cell x column (cell x fastest). */
  gridData(): Float64Array { return this.data; }

  /** One per-grid value (cell index 0-based, column 0-based). */
  gridValue(cell: number, col: number): number {
    if (cell < 0 || cell >= this.ncell || col < 0 || col >= this.attrs.length) {
      throw new StyleError(`compute ${this.id} property/grid: grid index out of range`);
    }
    return this.data[cell * this.attrs.length + col];
  }

  /** The single column as a per-grid vector (ncols must be 1). */
  gridVector(): Float64Array {
    if (this.attrs.length !== 1) throw new StyleError(`compute ${this.id} property/grid calculates a per-grid array, not a vector`);
    return this.data;
  }

  private fill(): void {
    const sys = this.sys;
    const g = sys.geom;
    const { nx, ny, nz } = this;
    const n = [nx, ny, nz];
    const tri = g.triclinic;
    const prd = [g.lx, g.ly, g.lz];
    const lo = sys.state.box.lo;
    const idx = [0, 0, 0];
    const lamLow = [0, 0, 0], lamCtr = [0, 0, 0];
    for (let c = 0; c < this.ncell; c++) {
      idx[0] = c % nx;
      idx[1] = Math.floor(c / nx) % ny;
      idx[2] = Math.floor(c / (nx * ny));
      for (let d = 0; d < 3; d++) {
        const dl = 1.0 / n[d];
        lamLow[d] = idx[d] * dl;
        lamCtr[d] = (idx[d] + 0.5) * dl;
      }
      // an unscaled coordinate: orthogonal boxlo + i * delta, triclinic h x lamda + boxlo
      const coord = (lam: number[], ctr: boolean, d: number): number => {
        if (!tri) return lo[d] + (idx[d] + (ctr ? 0.5 : 0)) * (prd[d] / n[d]);
        if (d === 0) return g.lx * lam[0] + g.xy * lam[1] + g.xz * lam[2] + lo[0];
        if (d === 1) return g.ly * lam[1] + g.yz * lam[2] + lo[1];
        return g.lz * lam[2] + lo[2];
      };
      for (let j = 0; j < this.attrs.length; j++) {
        const a = this.attrs[j];
        let v: number;
        switch (a) {
          case 'id': v = c + 1; break;
          case 'proc': v = 0; break;
          case 'ix': v = idx[0] + 1; break;
          case 'iy': v = idx[1] + 1; break;
          case 'iz': v = idx[2] + 1; break;
          case 'x': v = coord(lamLow, false, 0); break;
          case 'y': v = coord(lamLow, false, 1); break;
          case 'z': v = coord(lamLow, false, 2); break;
          case 'xc': v = coord(lamCtr, true, 0); break;
          case 'yc': v = coord(lamCtr, true, 1); break;
          case 'zc': v = coord(lamCtr, true, 2); break;
          case 'xs': v = lamLow[0]; break;
          case 'ys': v = lamLow[1]; break;
          case 'zs': v = lamLow[2]; break;
          case 'xsc': v = lamCtr[0]; break;
          case 'ysc': v = lamCtr[1]; break;
          case 'zsc': v = lamCtr[2]; break;
        }
        this.data[c * this.attrs.length + j] = v;
      }
    }
  }
}
