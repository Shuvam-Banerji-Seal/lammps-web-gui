/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SHOCK/fix_append_atoms.cpp and fix_append_atoms.h at commit b332f28 (Laurence Fried and Evan Reed, for
 * the shock examples; the LAMMPS developers). Copyright (2003) Sandia Corporation (GNU General Public License). This
 * file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix append/atoms zhi keywords: every freq steps a layer of atoms is added on the upper z face of a shrink-wrapped
 * (minimum) box, one atom per lattice basis site inside the layer [zhi, zhi + size). The new atoms take the basis
 * types (keyword basis) and zero velocity, and the box is reset (shrink-wrap) after the atoms are added. The top layer
 * of the box (within size of boxhi) is held still (post_force zeroes its forces and velocities). The atoms
 * are appended through the engine's atom insertion path (appendAtoms, as fix deposit does). The random, spatial and
 * temp keywords and the other faces are not supported (StyleError).
 */
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { appendAtoms } from '../../engine/atoms';

export class FixAppendAtoms extends Fix {
  readonly style = 'append/atoms';
  private freq = 0;
  private size = 0;
  private scaleflag = true;
  private zhiflag = false;
  private basisType: number[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const lat = sys.lattice;
    if (!lat || lat.basis.length === 0) throw new StyleError('Fix append/atoms requires a lattice be defined');
    this.basisType = lat.basis.map(() => 1);
    let k = 0;
    const need = (n: number) => { if (k + n >= args.length) throw new StyleError('Illegal fix append/atoms command'); };
    while (k < args.length) {
      const w = args[k];
      if (w === 'xlo' || w === 'xhi' || w === 'ylo' || w === 'yhi' || w === 'zlo') {
        throw new StyleError('Only zhi currently implemented for fix append/atoms');
      } else if (w === 'zhi') {
        this.zhiflag = true;
        k++;
        if (sys.state.box.boundary[2][1] !== 'm') throw new StyleError('Append boundary must be shrink/minimum');
      } else if (w === 'freq') {
        need(1);
        const n = parseNum(args[k + 1], 'freq');
        if (!Number.isInteger(n) || n <= 0) throw new StyleError('Illegal fix append/atoms command: freq must be a positive integer');
        this.freq = n;
        k += 2;
      } else if (w === 'basis') {
        need(2);
        const ib = parseNum(args[k + 1], 'basis');
        const it = parseNum(args[k + 2], 'basis type');
        if (!Number.isInteger(ib) || !Number.isInteger(it) || ib <= 0 || ib > this.basisType.length || it <= 0 || it > sys.state.ntypes) {
          throw new StyleError('Invalid basis setting in fix append/atoms command');
        }
        this.basisType[ib - 1] = it;
        k += 3;
      } else if (w === 'size') {
        need(1);
        this.size = parseNum(args[k + 1], 'size');
        k += 2;
      } else if (w === 'units') {
        need(1);
        const u = args[k + 1];
        if (u === 'box') this.scaleflag = false;
        else if (u === 'lattice') this.scaleflag = true;
        else throw new StyleError('Illegal fix append/atoms command');
        k += 2;
      } else if (w === 'random' || w === 'spatial' || w === 'temp') {
        throw new StyleError(`fix append/atoms keyword ${w} is not supported by the browser engine`);
      } else {
        throw new StyleError(`Illegal fix append/atoms command: unknown keyword '${w}' (zhi, freq, basis, size, units)`);
      }
    }
    if (!this.zhiflag) throw new StyleError('fix append/atoms needs the zhi face');
    if (this.freq <= 0) throw new StyleError('Illegal fix append/atoms command: freq is required');
    // units lattice (the default): the size is in lattice spacings along z
    if (this.scaleflag) this.size *= lat.spacing[2];
  }

  /**
   * post_force (fix_append_atoms.cpp, also at setup): the atoms at or above boxhi - size hold still, their forces and
   * velocities are zeroed (the temp keyword, which would thermostat them, is not supported).
   */
  postForce(): void {
    const s = this.sys.state;
    const hiz = s.box.hi[2] - this.size;
    for (let i = 0; i < s.n; i++) {
      if (s.x[3 * i + 2] >= hiz) {
        s.f.fill(0, 3 * i, 3 * i + 3);
        s.v.fill(0, 3 * i, 3 * i + 3);
      }
    }
  }

  /** The new atoms come on the steps that are multiples of freq (initial_integrate forces a neighbor list build). */
  reneighborDue(step: number): boolean {
    return step % this.freq === 0;
  }

  /**
   * pre_exchange (fix_append_atoms.cpp): on the steps that are multiples of freq, the lattice sites (all basis sites
   * of the lattice cells that the layer's bounding box touches) inside the layer [zhi, zhi + size) in x and y within the
   * box are appended, in the loop order k, j, i, basis; then the box is reset.
   */
  preExchange(): void {
    const sys = this.sys;
    const s = sys.state;
    if (s.step % this.freq !== 0) return;
    const lat = sys.lattice!;
    const lo = s.box.lo, hi = s.box.hi;
    const zlo = hi[2], zhi = hi[2] + this.size;
    // lattice bounding box of the layer: the 8 corners in lattice coordinates (lattice.cpp bbox with flag 1)
    const corners: [number, number, number][] = [];
    for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [zlo, zhi]) corners.push(lat.fromBox([x, y, z]));
    let xmin = Infinity, ymin = Infinity, zmin = Infinity, xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
    for (const c of corners) {
      if (c[0] < xmin) xmin = c[0];
      if (c[0] > xmax) xmax = c[0];
      if (c[1] < ymin) ymin = c[1];
      if (c[1] > ymax) ymax = c[1];
      if (c[2] < zmin) zmin = c[2];
      if (c[2] > zmax) zmax = c[2];
    }
    let ilo = Math.trunc(xmin), jlo = Math.trunc(ymin), klo = Math.trunc(zmin);
    const ihi = Math.trunc(xmax), jhi = Math.trunc(ymax), khi = Math.trunc(zmax);
    if (xmin < 0) ilo--;
    if (ymin < 0) jlo--;
    if (zmin < 0) klo--;
    const xs: number[] = [];
    const types: number[] = [];
    for (let k = klo; k <= khi; k++) {
      for (let j = jlo; j <= jhi; j++) {
        for (let i = ilo; i <= ihi; i++) {
          for (let m = 0; m < lat.basis.length; m++) {
            const b = lat.basis[m];
            const p = lat.toBox([i + b[0], j + b[1], k + b[2]]);
            // fix_append_atoms.cpp: sublo <= x < subhi in x and y; subhi[2] <= z < subhi[2] + size
            if (p[0] >= lo[0] && p[0] < hi[0] && p[1] >= lo[1] && p[1] < hi[1] && p[2] >= hi[2] && p[2] < hi[2] + this.size) {
              xs.push(p[0], p[1], p[2]);
              types.push(this.basisType[m]);
            }
          }
        }
      }
    }
    if (types.length === 0) return;
    appendAtoms(s, { x: Float64Array.from(xs), type: Int32Array.from(types), mask: 1 });
    sys.atomsChanged();
    // domain->reset_box(): wrap and shrink-wrap the box around the atoms (the minimum boundary grows it)
    sys.pbc();
    sys.setupNeighbors();
  }
}
