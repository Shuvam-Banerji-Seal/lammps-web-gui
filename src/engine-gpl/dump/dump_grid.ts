/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/dump_grid.cpp and dump_grid.h (DumpGrid: parse_fields, init_style, header_item, header_item_triclinic,
 * count, pack_grid2d / pack_grid3d, write_lines, modify_param) and src/dump_grid_vtk.cpp and dump_grid_vtk.h
 * (DumpGridVTK) at commit b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public License);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * dump ID group grid N file c_ID:gname:dname[I] f_ID:gname:dname[I] ... (docs.lammps.org/dump.html): one line per
 * grid cell, cells in grid-ID order (x fastest), after the header ITEM: TIMESTEP, BOX BOUNDS, DIMENSION, GRID SIZE
 * nx ny nz and GRID CELLS with the column names. The core dump keeps the schedule, the files and the dump_modify
 * keywords common to every style (format, colname, sort, header, units, time, append ...); this helper resolves the
 * references (src/engine/grid.ts), checks that every field has the same grid, and writes the snapshot.
 *
 * dump grid/vtk: one or three fields (a "Scalar" or "Vector" cell array) on a rectilinear grid (.vtr, the default, or
 * .vtk legacy) or image data (.vti), in single precision unless dump_modify double yes; one snapshot per file, sorted
 * by grid ID, orthogonal boxes only. Not written: dump_modify binary yes (refused, as for dump vtk).
 */
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { gridColumn, isGridRef, resolveGridRef, type GridRef } from '../../engine/grid';
import { formatNumber } from '../../engine/cformat';
import type { DumpGridFactory, DumpGridHost, DumpGridStyle } from '../../engine/output/dump';
import { ANGSTROM_PER_UNIT } from '../../engine/lepton/units';
import { VtkWriter, singlePrecisionResolution, type VtkPrecision } from './vtk_writer';


/** Resolves the fields and checks that their grids have one size (init_style: "Dump grid field grid sizes do not match"). */
const resolveFields = (host: DumpGridHost, style: string, fields: string[]): { refs: GridRef[]; size: [number, number, number] } => {
  const sys = host.sys;
  if (!fields.length) throw new StyleError(`No dump ${style} arguments specified`);
  const refs: GridRef[] = [];
  for (const f of fields) {
    // parse_fields: a word that is not a compute or fix reference ends the fields, and only dump image takes more
    if (!isGridRef(f)) throw new StyleError(`Invalid attribute ${f} in dump ${style} command`);
    refs.push(resolveGridRef(sys, 'Dump grid', f, host.every));
  }
  let size: [number, number, number] = [0, 0, 0];
  refs.forEach((r, k) => {
    const src = r.kind === 'c' ? sys.compute(r.id) : sys.fix(r.id);
    const sz = src.gridSize(r.igrid);
    if (sys.dimension === 2) sz[2] = 1;
    if (k === 0) size = sz;
    else if (sz[0] !== size[0] || sz[1] !== size[1] || (sys.dimension === 3 && sz[2] !== size[2])) {
      throw new StyleError('Dump grid field grid sizes do not match');
    }
  });
  return { refs, size };
};

/** The fields' values per cell, in grid-ID order: data[cell * nfield + field]. */
const packCells = (sys: System, refs: GridRef[], ncell: number): Float64Array => {
  const nfield = refs.length;
  const buf = new Float64Array(ncell * nfield);
  refs.forEach((r, k) => {
    const col = gridColumn(sys, r);
    for (let c = 0; c < ncell; c++) buf[c * nfield + k] = col[c];
  });
  return buf;
};

/** The cell order of a snapshot: grid ID (also the unsorted order of one process) or one column's values. */
const sortedCells = (sort: DumpGridHost['sort'], buf: Float64Array, nfield: number, ncell: number): number[] => {
  const order = Array.from({ length: ncell }, (_, c) => c);
  if (sort === 'off' || sort === 'id') return order;
  const { col, desc } = sort;
  order.sort((a, b) => (desc ? buf[b * nfield + col] - buf[a * nfield + col] : buf[a * nfield + col] - buf[b * nfield + col]));
  return order;
};

class DumpGridText implements DumpGridStyle {
  private readonly refs: GridRef[];
  private readonly size: [number, number, number];

  constructor(private readonly host: DumpGridHost, private readonly fields: string[]) {
    ({ refs: this.refs, size: this.size } = resolveFields(host, 'grid', fields));
  }

  modify(args: string[], k: number): number {
    // DumpGrid::modify_param region: kept and checked, the cells written are not filtered by it
    if (args[k] === 'region') {
      const v = args[k + 1];
      if (v === undefined) throw new StyleError('Illegal dump_modify command');
      if (v !== 'none') this.host.sys.region(v);
      return 2;
    }
    return 0;
  }

  write(step: number): void {
    const host = this.host;
    const sys = host.sys;
    const s = sys.state;
    const [nx, ny, nz] = this.size;
    const ncell = nx * ny * nz;
    const nfield = this.refs.length;
    const buf = packCells(sys, this.refs, ncell);
    const lines: string[] = [];
    if (host.header) {
      if (host.units && !host.opened) lines.push('ITEM: UNITS', s.units.style);
      if (host.time) lines.push('ITEM: TIME', formatNumber(s.time + (s.step - s.timeStep) * s.dt, '%.16g'));
      lines.push('ITEM: TIMESTEP', String(step));
      const bnd = s.box.boundary.map((f) => f[0] + f[1]).join(' ');
      const e = (v: number) => formatNumber(v, '%1.16e');
      if (sys.geom.triclinic) {
        const [xy, xz, yz] = s.box.tilt;
        const xlob = s.box.lo[0] + Math.min(0, xy, xz, xy + xz), xhib = s.box.hi[0] + Math.max(0, xy, xz, xy + xz);
        const ylob = s.box.lo[1] + Math.min(0, yz), yhib = s.box.hi[1] + Math.max(0, yz);
        lines.push(`ITEM: BOX BOUNDS xy xz yz ${bnd}`, `${e(xlob)} ${e(xhib)} ${e(xy)}`, `${e(ylob)} ${e(yhib)} ${e(xz)}`,
          `${e(s.box.lo[2])} ${e(s.box.hi[2])} ${e(yz)}`);
      } else {
        lines.push(`ITEM: BOX BOUNDS ${bnd}`, ...[0, 1, 2].map((d) => `${e(s.box.lo[d])} ${e(s.box.hi[d])}`));
      }
      lines.push('ITEM: DIMENSION', String(sys.dimension));
      lines.push('ITEM: GRID SIZE nx ny nz', `${nx} ${ny} ${nz}`);
      lines.push(`ITEM: GRID CELLS ${this.fields.map((f, k) => host.colnames.get(k) ?? f).join(' ')}`);
    }
    const formatters = host.formatters();
    for (const c of sortedCells(host.sort, buf, nfield, ncell)) {
      const vals: string[] = [];
      for (let k = 0; k < nfield; k++) vals.push(formatters[k](buf[c * nfield + k]));
      lines.push(vals.join(' '));
    }
    host.writeText(lines.join('\n') + '\n');
  }
}

class DumpGridVtk implements DumpGridStyle {
  private readonly refs: GridRef[];
  private readonly size: [number, number, number];
  private precision: VtkPrecision = 'single';
  private precisionWarned = false;
  private readonly flavor: 'legacy' | 'xml';
  private readonly dataset: 'rectilinear' | 'image';

  constructor(private readonly host: DumpGridHost, fields: string[]) {
    if (!fields.length) throw new StyleError('No dump grid/vtk arguments specified');
    // As native LAMMPS does (src/dump_grid.cpp at b332f28, DumpGrid::DumpGrid parse_fields): the base constructor
    // rejects a word that is not a grid field before DumpGridVTK::DumpGridVTK counts the fields
    for (const f of fields) {
      if (!isGridRef(f)) throw new StyleError(`Invalid attribute ${f} in dump grid/vtk command`);
    }
    // DumpGridVTK: "Dump grid/vtk requires one or three fields"
    if (fields.length !== 1 && fields.length !== 3) throw new StyleError('Dump grid/vtk requires one or three fields');
    ({ refs: this.refs, size: this.size } = resolveFields(host, 'grid/vtk', fields));
    // the file name extension selects the format; an unrecognized one keeps the XML rectilinear grid
    const dot = host.file.lastIndexOf('.');
    const ext = dot < 0 ? '' : host.file.slice(dot);
    this.flavor = ext === '.vtk' ? 'legacy' : 'xml';
    this.dataset = ext === '.vti' ? 'image' : 'rectilinear';
  }

  modify(args: string[], k: number): number {
    // DumpGridVTK::modify_param takes binary and double only (not dump grid's region)
    const key = args[k], v = args[k + 1];
    if (key !== 'binary' && key !== 'double') return 0;
    if (v === undefined) throw new StyleError(`Illegal dump_modify ${key} command: missing argument(s)`);
    const on = v === 'yes' || v === 'on' || v === 'true' || v === '1';
    if (!on && v !== 'no' && v !== 'off' && v !== 'false' && v !== '0') throw new StyleError(`Expected boolean parameter instead of '${v}' in input script or data file`);
    if (key === 'binary') {
      if (on) throw new StyleError('dump_modify binary yes is not supported by the browser engine (binary VTK output); use binary no');
    } else this.precision = on ? 'double' : 'single';
    return 2;
  }

  write(step: number): void {
    const host = this.host;
    const sys = host.sys;
    const s = sys.state;
    // init_style checks
    if (!host.file.includes('*')) throw new StyleError('Dump grid/vtk requires one snapshot per file');
    if (host.sort !== 'id') throw new StyleError('Dump grid/vtk requires sorting on IDs');
    if (sys.geom.triclinic) throw new StyleError('Dump grid/vtk does not support triclinic simulation boxes, use dump grid instead');
    const [nx, ny, nz] = this.size;
    const ncell = nx * ny * nz;
    const nfield = this.refs.length;
    const values = packCells(sys, this.refs, ncell);
    const writer = new VtkWriter(this.flavor, this.precision);
    const lo = s.box.lo;
    const prd = [s.box.hi[0] - lo[0], s.box.hi[1] - lo[1], s.box.hi[2] - lo[2]];
    const n = [nx, ny, nz];
    if (this.dataset === 'image') {
      writer.setImageData([nx + 1, ny + 1, nz + 1], [lo[0], lo[1], lo[2]], [prd[0] / nx, prd[1] / ny, prd[2] / nz]);
    } else {
      // xyz_grid: the cell boundaries, boxlo + i * (prd / N)
      const axis = (d: number): number[] => {
        const delta = prd[d] / n[d];
        return Array.from({ length: n[d] + 1 }, (_, i) => lo[d] + i * delta);
      };
      writer.setRectilinear(axis(0), axis(1), axis(2));
    }
    writer.addCellArray({ name: nfield === 1 ? 'Scalar' : 'Vector', kind: 'double', ncomp: nfield, nums: values, strs: [] });
    const stepText = host.pad > 0 ? String(step).padStart(host.pad, '0') : String(step);
    sys.writeFileAt(sys.resolveFile(host.fillWildcard(host.file, stepText)), writer.text(), false);
    // force->angstrom is one Angstrom in the distance unit
    const maxcoord = writer.maxSinglePrecisionValue();
    const resolution = singlePrecisionResolution(maxcoord, 1 / (ANGSTROM_PER_UNIT[s.units.style] ?? 1));
    if (!this.precisionWarned && resolution > 0.0) {
      this.precisionWarned = true;
      sys.warn(`Dump grid/vtk writes grid coordinates in single precision, which resolves the largest coordinate of this grid, ${formatNumber(maxcoord, '%.4g')}, to only about ${formatNumber(resolution, '%.2g')} length units. Use dump_modify double yes if your analysis needs more resolution than that.`);
    }
  }
}

export const dumpGrid: DumpGridFactory = (host, fields) => new DumpGridText(host, fields);
export const dumpGridVtk: DumpGridFactory = (host, fields) => new DumpGridVtk(host, fields);
