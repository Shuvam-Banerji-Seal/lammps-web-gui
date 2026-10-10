import { Compute } from './compute';
import { StyleError } from '../force/types';
import { parseInt_, parseNum } from '../force/util';
import type { System } from '../system';

/*
 * compute ID group-ID gaussian/grid/local grid nx ny nz rcutfac R_1 ... R_ntypes sigma_1 ... sigma_ntypes
 * — docs.lammps.org/compute_gaussian_grid_local.html:
 *   "Define a computation that calculates a Gaussian representation of the ionic structure."
 *   "For each LAMMPS type, a separate sum of Gaussians is calculated, using a separate Gaussian
 *   broadening per type."
 *   "The computation is always performed on the numerical grid, no atom-based version of this compute
 *   exists."
 *   "The layout of the grid is the same as for the see sna/grid/local command." (see
 *   docs.lammps.org/compute_sna_atom.html; the grid layout is the one of ComputeSnaGridLocal)
 *   "looping over the global index ix fastest, then iy, and iz slowest" (rows of the local array).
 *   Output: "The array contains math ntypes+6 columns": ix iy iz, then x y z, then one value per type.
 *   Arguments: "number of grid points in x, y, and z directions (positive integer)";
 *   "scale factor applied to all cutoff radii"; "Gaussian widths, one for each type (distance units)".
 *
 * Form (own words): for grid point g and type t, value_t(g) = sum over the atoms j of type t in the
 * compute group and over all their periodic images of norm_t * exp(-r^2 / (2 sigma_t^2)), where r is the
 * distance from g and norm_t = 1 / (sigma_t sqrt(2 pi))^3, and the sum only includes r inside the cutoff.
 * The sum is hard-cut (no taper). The grid is the one of the sna/grid/local compute: the grid point with
 * fractional coordinates (ix/nx, iy/ny, iz/nz) of the (possibly triclinic) cell.
 *
 * Measured with native LAMMPS (black box): a single atom at the origin of a 10 x 10 x 10 periodic box gives
 * at grid point 0 the value 0.507949087473928 = 1 / (0.5 sqrt(2 pi))^3 (sigma 0.5), and at distances
 * 0.5, 1, 1.5 the values equal norm * exp(-r^2 / (2 sigma^2)) to round-off (relative deviation 3e-15).
 * Measured with native LAMMPS (black box): the cutoff is |2 rcutfac R_t| (for example rcutfac 1, R 1 gives
 * 2; rcutfac 1, R 0.6 gives 1.2; rcutfac 1.5, R 0.5 gives 1.5; rcutfac 1, R 0.9 gives 1.8): the value is zero
 * at r equal to the cutoff (strict inequality) and non-zero just inside it.
 * Measured with native LAMMPS (black box): with rcutfac -1 (R 0.5), rcutfac 1 (R -0.5), and rcutfac -1 with
 * R -0.5 the cutoff is 1 (the absolute value of 2 rcutfac R); with R_1 = 0 the value of type 1 is zero
 * everywhere, including the grid point on the atom.
 * Measured with native LAMMPS (black box): each type column is built from atoms of that type only, with its
 * own cutoff and width (two types, R 0.5 and 1.0, sigma 0.3 and 0.6, atoms of types 1 and 2 placed 3 apart:
 * the columns match the per-type reference to 1e-12).
 * Measured with native LAMMPS (black box): atoms outside the compute group do not contribute (group sub holds
 * atom 1 only; atom 2 of the same type at distance 3 is absent from the values).
 * Measured with native LAMMPS (black box): all periodic images inside the cutoff are summed (box 2, cutoff 2,
 * one atom: the values equal the sum over images in [-4, 4]^3 to round-off).
 * Measured with native LAMMPS (black box): in a triclinic box (prism 0 10 0 10 0 10 xy 3 xz -2 yz 1.5) the
 * grid point (ix, iy, iz) sits at ix/nx a + iy/ny b + iz/nz c (error 4e-15), and the values equal the
 * image sum with the lattice vectors a, b, c.
 *
 * Measured with native LAMMPS (black box), error cases (the engine refuses the same inputs):
 *   'grid' missing (compute gaussian/grid/local 2 2 2 ...): native message ERROR: Illegal compute grid/local command
 *   fewer arguments than 5 + 2 ntypes: native message ERROR: Illegal compute gaussian/grid/local command
 *   grid count 0: native message ERROR: All grid/local dimensions must be positive
 *   grid count 2.5: native message ERROR: Expected integer parameter instead of '2.5' in input script or data file
 *   sigma 0 or -0.1: native message ERROR: Gaussian width for type 1 must be > 0
 *   a non-number such as abc: native message ERROR: Expected floating point parameter instead of 'abc' in input script or data file
 * Measured with native LAMMPS (black box), accepted: rcutfac 0 or -1, R 0 or negative (zero or |cutoff| as
 * above), and trailing arguments beyond the 2 ntypes radii and widths (they are ignored, no error).
 *
 * Ghost atoms: the grid points need every periodic image within the cutoff. The engine requires the ghost
 * cutoff to cover max |2 rcutfac R_t| and throws otherwise (the same rule as the sna grid computes).
 */

interface GaussSpec {
  nx: number;
  ny: number;
  nz: number;
  rcutfac: number;
  radius: number[];
  sigma: number[];
}

const parseSpec = (id: string, style: string, args: string[], ntypes: number): GaussSpec => {
  // Measured with native LAMMPS (black box): a missing 'grid' keyword is an "Illegal" command.
  if (args.length < 5 + 2 * ntypes || args[0] !== 'grid') {
    throw new StyleError(`compute ${id} (${style}): expects 'grid nx ny nz rcutfac R_1 .. R_${ntypes} sigma_1 .. sigma_${ntypes}' (${ntypes} types)`);
  }
  const dims = [1, 2, 3].map((k) => parseInt_(args[k], `compute ${id} (${style}) grid ${'nxyz'[k - 1]}`));
  for (const d of dims) if (!(d >= 1)) throw new StyleError(`compute ${id} (${style}): grid counts must be positive integers (got ${dims.join(' ')})`);
  const rcutfac = parseNum(args[4], `compute ${id} (${style}) rcutfac`);
  const radius: number[] = [];
  const sigma: number[] = [];
  for (let t = 0; t < ntypes; t++) radius.push(parseNum(args[5 + t], `compute ${id} (${style}) R_${t + 1}`));
  for (let t = 0; t < ntypes; t++) {
    const s = parseNum(args[5 + ntypes + t], `compute ${id} (${style}) sigma_${t + 1}`);
    if (!(s > 0)) throw new StyleError(`compute ${id} (${style}): Gaussian width for type ${t + 1} must be > 0`);
    sigma.push(s);
  }
  return { nx: dims[0], ny: dims[1], nz: dims[2], rcutfac, radius, sigma };
};

export class ComputeGaussianGridLocal extends Compute {
  readonly style = 'gaussian/grid/local';
  private readonly spec: GaussSpec;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.spec = parseSpec(id, 'gaussian/grid/local', args, sys.state.ntypes);
    this.localFlag = true;
    this.sizeLocalCols = 6 + sys.state.ntypes;
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const sp = this.spec;
    const nt = sp.sigma.length;
    const cols = 6 + nt;

    const cutOf = (t: number): number => Math.abs(2 * sp.rcutfac * sp.radius[t]);
    let cutmax = 0;
    const cut = new Float64Array(nt), norm = new Float64Array(nt), inv2s2 = new Float64Array(nt);
    for (let t = 0; t < nt; t++) {
      cut[t] = cutOf(t);
      if (cut[t] > cutmax) cutmax = cut[t];
      const sg = sp.sigma[t];
      norm[t] = 1 / Math.pow(sg * Math.sqrt(2 * Math.PI), 3);
      inv2s2[t] = 1 / (2 * sg * sg);
    }
    if (nb.cutghost < cutmax - 1e-12) {
      throw new StyleError(`compute ${this.id} (${this.style}): cutoff up to ${cutmax} exceeds the ghost cutoff ${nb.cutghost} (set a pair style with a larger cutoff)`);
    }

    const { nx, ny, nz } = sp;
    const out = new Float64Array(nx * ny * nz * cols);
    const nall = nb.nall, xa = nb.xall, ta = nb.typeall, owner = nb.owner;
    const gb = this.groupBit;
    // contributing atoms (group members with a non-zero cutoff, incl. ghosts) binned into cubic cells of
    // edge cutmax, so each grid point only visits the 27 cells around it
    const sel: number[] = [];
    let lo0 = Infinity, lo1 = Infinity, lo2 = Infinity, hi0 = -Infinity, hi1 = -Infinity, hi2 = -Infinity;
    for (let k = 0; k < nall; k++) {
      if (!(s.mask[owner[k]] & gb) || !(cut[ta[k] - 1] > 0)) continue;
      sel.push(k);
      const x = xa[3 * k], y = xa[3 * k + 1], z = xa[3 * k + 2];
      if (x < lo0) lo0 = x; if (x > hi0) hi0 = x;
      if (y < lo1) lo1 = y; if (y > hi1) hi1 = y;
      if (z < lo2) lo2 = z; if (z > hi2) hi2 = z;
    }
    const cell = cutmax > 0 ? cutmax : 1;
    const nc0 = sel.length ? Math.floor((hi0 - lo0) / cell) + 1 : 1;
    const nc1 = sel.length ? Math.floor((hi1 - lo1) / cell) + 1 : 1;
    const nc2 = sel.length ? Math.floor((hi2 - lo2) / cell) + 1 : 1;
    const head = new Int32Array(nc0 * nc1 * nc2).fill(-1);
    const next = new Int32Array(sel.length);
    for (let m = 0; m < sel.length; m++) {
      const k = sel[m];
      const c = Math.floor((xa[3 * k] - lo0) / cell) + nc0 * (Math.floor((xa[3 * k + 1] - lo1) / cell) + nc1 * Math.floor((xa[3 * k + 2] - lo2) / cell));
      next[m] = head[c];
      head[c] = m;
    }
    const pos = [0, 0, 0];
    const vals = new Float64Array(nt);
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          sys.geom.fromLamda(ix / nx, iy / ny, iz / nz, pos);
          const x = pos[0], y = pos[1], z = pos[2];
          vals.fill(0);
          const c0 = Math.floor((x - lo0) / cell), c1 = Math.floor((y - lo1) / cell), c2 = Math.floor((z - lo2) / cell);
          for (let b2 = Math.max(c2 - 1, 0); b2 <= Math.min(c2 + 1, nc2 - 1); b2++) {
            for (let b1 = Math.max(c1 - 1, 0); b1 <= Math.min(c1 + 1, nc1 - 1); b1++) {
              for (let b0 = Math.max(c0 - 1, 0); b0 <= Math.min(c0 + 1, nc0 - 1); b0++) {
                for (let m = head[b0 + nc0 * (b1 + nc1 * b2)]; m >= 0; m = next[m]) {
                  const k = sel[m];
                  const t = ta[k] - 1;
                  const dx = xa[3 * k] - x, dy = xa[3 * k + 1] - y, dz = xa[3 * k + 2] - z;
                  const r2 = dx * dx + dy * dy + dz * dz;
                  // strict inequality: the atom at distance exactly the cutoff does not contribute
                  if (!(Math.sqrt(r2) < cut[t])) continue;
                  vals[t] += norm[t] * Math.exp(-r2 * inv2s2[t]);
                }
              }
            }
          }
          const o = (ix + nx * (iy + ny * iz)) * cols;
          out[o] = ix;
          out[o + 1] = iy;
          out[o + 2] = iz;
          out[o + 3] = x;
          out[o + 4] = y;
          out[o + 5] = z;
          for (let t = 0; t < nt; t++) out[o + 6 + t] = vals[t];
        }
      }
    }
    this.localRows = nx * ny * nz;
    return out;
  }
}
