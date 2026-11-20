/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the Colvars library it bundles (lib/colvars,
 * Colvars version 2026-07-27) and its LAMMPS proxy (src/COLVARS/colvarproxy_lammps.cpp, part of Colvars).
 * Licence: Colvars (https://github.com/Colvars/colvars; Giacomo Fiorin, Jerome Henin and the Colvars developers) is
 * distributed under the GNU Lesser General Public License, version 3 (LGPL-3.0). This file translates library code and
 * is LGPL-3.0 too (the licence texts are colvars/LICENSE.lgpl-3.0 and colvars/LICENSE.gpl-3.0; see
 * src/engine-gpl/README.md). The rest of this directory, LAMMPS ported code, is GPL-2.0.
 *
 * Sources: colvarmodule.cpp (calc, calc_biases, update_colvar_forces, write_traj_files, write_traj_label, write_traj,
 * write_state, read_state, setup_input, setup_output, parse_global_params, parse_colvars, parse_biases), colvar.cpp
 * (init, collect_cvc_values, update_forces_energy, communicate_forces, dist2, dist2_lgrad, write_traj_label, write_traj,
 * get_state_params), colvarcomp.cpp (cvc::init, cvc::dist2, cvc::dist2_lgrad, cvc::apply_force),
 * colvarcomp_distances.cpp (distance, distance_vec, distance_z, distance_xy, cartesian, gyration),
 * colvarcomp_angles.cpp (angle, dihedral), colvarcomp_coordnums.cpp/.h (coordnum, selfcoordnum, groupcoordnum, the
 * switching function and pair list), colvaratoms.cpp (atom_group parse, update_total_mass,
 * calc_center_of_mass, set_weighted_gradient, apply_colvar_force, apply_force), colvarbias.cpp and
 * colvarbias_restraint.cpp (the harmonic and harmonic_walls restraints), colvarbias_meta.cpp/.h (metadynamics: hills,
 * grids, well-tempered, the PMF and hills trajectory files), colvargrid.h, colvargrid_def.h and colvargrid.cpp (the
 * scalar and gradient grids: init_from_colvars, setup, map_grid, write_restart, write_multicol), colvar.cpp
 * init_grid_parameters and check_grid_parameters, colvars_system.h (system_boundary_conditions), colvartypes.h/.cpp
 * and colvarvalue.cpp (vector arithmetic and stream formats), colvarscript_commands.h (config, configfile),
 * colvarproxy.cpp, colvarproxy_io.cpp (output streams, backup_file) and colvarproxy_lammps.cpp (init, setup, compute).
 *
 * The supported subset: the components distance, distanceVec, distanceZ, distanceXY, angle, dihedral, cartesian,
 * gyration, coordNum, selfCoordNum and groupCoord (atom groups from atomNumbers, atomNumbersRange or a dummyAtom), linear or polynomial combinations of components
 * (componentCoeff, componentExp), periodic components (dihedral; distanceZ with period), the harmonic and
 * harmonicWalls restraints, metadynamics on one replica (hillWeight, hillWidth or gaussianSigmas, newHillFrequency,
 * useGrids with gridsUpdateFrequency, expandBoundaries, keepHills, wellTempered with biasTemperature, stepZeroData,
 * outputFreq, writeFreeEnergyFile, writeHillsTrajectory), the trajectory file (colvarsTrajFrequency) and the state
 * file (colvarsRestartFrequency, the end of each run, the input state; a metadynamics state is written but not read
 * back). Every other component, bias or keyword stops with an error naming it. The
 * arithmetic follows the library's order of operations, so the values match native LAMMPS to the last bit.
 */
import { compileCFormat } from '../../engine/cformat';
import { Conf, colvarsError, parseBool, parseInt10, parseReal, splitValues } from './config';
import { acos } from '../libm/acos';
import { exp } from '../libm/exp';

/** colvarmodule.h widths and precisions of the trajectory and state files. */
const IT_WIDTH = 12;
const CV_WIDTH = 21;
const EN_WIDTH = 21;
/** colvars_version.h at b332f28. */
export const COLVARS_VERSION = '2026-07-27';

const fmtSci = compileCFormat('%21.14e', 'float');
const fmtGen = compileCFormat('%21.14g', 'float');
const fmtDt = compileCFormat('%e', 'float');

/** colvarmodule::wrap_string: pads with spaces or cuts to nchars. */
const wrapString = (s: string, nchars: number): string =>
  s.length === 0 ? ' '.repeat(nchars) : s.length <= nchars ? s + ' '.repeat(nchars - s.length) : s.slice(0, nchars);

const padLeft = (s: string, w: number): string => (s.length >= w ? s : ' '.repeat(w - s.length) + s);

/** cvm::integer_power (from LAMMPS math_special.h). */
const integerPower = (x: number, n: number): number => {
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

type ValueType = 'scalar' | '3vector' | 'vector';

const TYPE_DESC: Record<ValueType, string> = { scalar: 'scalar number', '3vector': '3-dimensional vector', vector: 'n-dimensional vector' };

/** The proxy's atoms: positions, masses and the forces the colvars apply (colvarproxy_atoms). */
export class ColvarsAtoms {
  readonly ids: number[] = [];
  readonly refcount: number[] = [];
  pos = new Float64Array(0);
  mass = new Float64Array(0);
  force = new Float64Array(0);
  /** The list changed since the fix last built its tag list (modified_atom_list). */
  modified = false;

  /** colvarproxy_lammps::init_atom: the slot of an atom ID, added on first use. */
  initAtom(id: number): number {
    for (let i = 0; i < this.ids.length; i++) {
      if (this.ids[i] === id) { this.refcount[i] += 1; return i; }
    }
    if (id < 0) throw colvarsError(`Error: invalid atom number specified, ${id}`);
    this.ids.push(id);
    this.refcount.push(1);
    this.modified = true;
    const n = this.ids.length;
    const grow = (a: Float64Array, m: number): Float64Array<ArrayBuffer> => { const b = new Float64Array(m * n); b.set(a); return b; };
    this.pos = grow(this.pos, 3);
    this.mass = grow(this.mass, 1);
    this.force = grow(this.force, 3);
    return n - 1;
  }

  /** proxy->apply_atom_force: atoms_new_colvar_forces[index] += f. */
  addForce(index: number, fx: number, fy: number, fz: number): void {
    this.force[3 * index] += fx;
    this.force[3 * index + 1] += fy;
    this.force[3 * index + 2] += fz;
  }
}

/** cvm::system_boundary_conditions: the minimum-image convention of the Colvars library. */
export class Boundaries {
  type: 'non_periodic' | 'mixed' | 'pbc_orthogonal' | 'pbc_triclinic' = 'non_periodic';
  private px = false; private py = false; private pz = false;
  private ux = [0, 0, 0]; private uy = [0, 0, 0]; private uz = [0, 0, 0];
  private rx = [0, 0, 0]; private ry = [0, 0, 0]; private rz = [0, 0, 0];

  reset(): void {
    this.type = 'non_periodic';
    this.ux = [0, 0, 0]; this.uy = [0, 0, 0]; this.uz = [0, 0, 0];
    this.rx = [0, 0, 0]; this.ry = [0, 0, 0]; this.rz = [0, 0, 0];
  }

  copyFrom(b: Boundaries): void {
    this.type = b.type; this.px = b.px; this.py = b.py; this.pz = b.pz;
    this.ux = [...b.ux]; this.uy = [...b.uy]; this.uz = [...b.uz];
    this.rx = [...b.rx]; this.ry = [...b.ry]; this.rz = [...b.rz];
  }

  /** set_boundaries(periodic flags, A, B, C) as colvarproxy_lammps::compute passes the LAMMPS box. */
  set(px: boolean, py: boolean, pz: boolean, A: number[], B: number[], C: number[]): void {
    const tol2 = 1.0e-10;
    this.px = px; this.py = py; this.pz = pz;
    this.rx = [0, 0, 0]; this.ry = [0, 0, 0]; this.rz = [0, 0, 0];
    if (px === py && px === pz) {
      if (px) this.type = 'pbc_orthogonal';
      else { this.reset(); return; }
    } else this.type = 'mixed';
    const n2 = (v: number[]): number => v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const div = (v: number[], a: number): number[] => [v[0] / a, v[1] / a, v[2] / a];
    let offDiagonal = false;
    if (px) {
      this.ux = [...A];
      if (A[1] * A[1] > tol2 || A[2] * A[2] > tol2) offDiagonal = true;
      else this.rx = div(this.ux, n2(this.ux));
    } else { this.ux = [0, 0, 0]; this.rx = [0, 0, 0]; }
    if (py) {
      this.uy = [...B];
      if (B[0] * B[0] > tol2 || B[2] * B[2] > tol2) offDiagonal = true;
      else this.ry = div(this.uy, n2(this.uy));
    } else { this.uy = [0, 0, 0]; this.ry = [0, 0, 0]; }
    if (pz) {
      this.uz = [...C];
      if (C[0] * C[0] > tol2 || C[1] * C[1] > tol2) offDiagonal = true;
      else this.rz = div(this.uz, n2(this.uz));
    } else { this.uz = [0, 0, 0]; this.rz = [0, 0, 0]; }
    if (this.type === 'pbc_orthogonal' && offDiagonal) this.type = 'pbc_triclinic';
    if (this.type === 'pbc_triclinic') {
      // rvector::outer and the dot product, in the library's order
      const outer = (a: number[], b: number[]): number[] => [a[1] * b[2] - b[1] * a[2], -a[0] * b[2] + b[0] * a[2], a[0] * b[1] - b[0] * a[1]];
      const dot = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
      const vyz = outer(this.uy, this.uz);
      this.rx = div(vyz, dot(vyz, this.ux));
      const vzx = outer(this.uz, this.ux);
      this.ry = div(vzx, dot(vzx, this.uy));
      const vxy = outer(this.ux, this.uy);
      this.rz = div(vxy, dot(vxy, this.uz));
    }
  }

  /** position_distance(pos1, pos2): pos2 - pos1, shifted to the nearest image. */
  distance(p1: ArrayLike<number>, p2: ArrayLike<number>, out: number[]): void {
    let dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
    if (this.type === 'non_periodic') { out[0] = dx; out[1] = dy; out[2] = dz; return; }
    const { rx, ry, rz, ux, uy, uz } = this;
    const xs = Math.floor(rx[0] * dx + rx[1] * dy + rx[2] * dz + 0.5);
    const ys = Math.floor(ry[0] * dx + ry[1] * dy + ry[2] * dz + 0.5);
    const zs = Math.floor(rz[0] * dx + rz[1] * dy + rz[2] * dz + 0.5);
    dx -= xs * ux[0] + ys * uy[0] + zs * uz[0];
    dy -= xs * ux[1] + ys * uy[1] + zs * uz[1];
    dz -= xs * ux[2] + ys * uy[2] + zs * uz[2];
    if (this.type !== 'pbc_orthogonal') {
      // get_triclinic_shift: the neighbouring cell with the shortest distance
      let min2 = dx * dx + dy * dy + dz * dz;
      let sx = 0, sy = 0, sz = 0;
      const nx = this.px ? 1 : 0, ny = this.py ? 1 : 0, nz = this.pz ? 1 : 0;
      for (let ix = -nx; ix <= nx; ix++) {
        for (let iy = -ny; iy <= ny; iy++) {
          for (let iz = -nz; iz <= nz; iz++) {
            const hx = ix * ux[0] + iy * uy[0] + iz * uz[0];
            const hy = ix * ux[1] + iy * uy[1] + iz * uz[1];
            const hz = ix * ux[2] + iy * uy[2] + iz * uz[2];
            const ex = dx + hx, ey = dy + hy, ez = dz + hz;
            const d2 = ex * ex + ey * ey + ez * ez;
            if (d2 < min2) { sx = hx; sy = hy; sz = hz; min2 = d2; }
          }
        }
      }
      dx += sx; dy += sy; dz += sz;
    }
    out[0] = dx; out[1] = dy; out[2] = dz;
  }
}

/** cvm::atom_group without fitting: the atoms of a component, their centre of mass and gradients. */
class AtomGroup {
  readonly index: number[] = [];
  readonly ids: number[] = [];
  dummy: number[] | null = null;
  noforce = false;
  weight = new Float64Array(0);
  totalMass = 0;
  px = new Float64Array(0); py = new Float64Array(0); pz = new Float64Array(0);
  gx = new Float64Array(0); gy = new Float64Array(0); gz = new Float64Array(0);
  readonly com = [0, 0, 0];

  constructor(readonly atoms: ColvarsAtoms, readonly key: string, readonly description: string, conf: string) {
    const c = new Conf(conf);
    c.get('name');
    for (const k of ['centerToOrigin', 'centerReference', 'centerToReference', 'rotateReference', 'rotateToReference']) {
      const v = c.get(k);
      if (v !== undefined && parseBool(v, k)) throw colvarsError(`Error: the atom group keyword "${k}" (fitting to reference positions) is not supported by the browser engine.`);
    }
    for (const k of ['atomsOfGroup', 'indexGroup', 'psfSegID', 'atomNameResidueRange', 'atomsFile', 'refPositions', 'refPositionsFile', 'fittingGroup']) {
      if (c.get(k) !== undefined) throw colvarsError(`Error: the atom group keyword "${k}" is not supported by the browser engine (supported: atomNumbers, atomNumbersRange, dummyAtom, enableForces).`);
    }
    const seen = new Set<number>();
    const add = (id: number): void => {
      if (id < 0) throw colvarsError(`Error: invalid atom number ${id}`);
      // atom_modifier::add_atom: an atom listed twice is counted once
      if (seen.has(id)) return;
      seen.add(id);
      this.ids.push(id);
      this.index.push(atoms.initAtom(id));
    };
    for (const v of c.all('atomNumbers')) {
      const nums = v.trim() === '' ? [] : v.trim().split(/\s+/);
      if (nums.length === 0) throw colvarsError('Error: no numbers provided for "atomNumbers".');
      for (const t of nums) add(parseInt10(t, 'atomNumbers'));
    }
    for (const v of c.all('atomNumbersRange')) {
      const m = /^\s*(\d+)\s*-\s*(\d+)\s*$/.exec(v);
      if (!m || Number(m[1]) <= 0 || Number(m[2]) <= 0) throw colvarsError(`Error: no valid definition for "atomNumbersRange", "${v.trim()}".`);
      for (let a = Number(m[1]); a <= Number(m[2]); a++) add(a);
    }
    const dummy = c.get('dummyAtom');
    if (dummy !== undefined) {
      const v = splitValues(dummy);
      if (v.length !== 1 || !Array.isArray(v[0]) || v[0].length !== 3) throw colvarsError(`Error: could not read the position of "dummyAtom" from "${dummy.trim()}".`);
      if (this.ids.length > 0) throw colvarsError('Error: cannot set up group "' + key + '" as a dummy atom and provide it with atom definitions.');
      this.dummy = [...v[0]];
    } else if (this.ids.length === 0) {
      throw colvarsError(`Error: no atoms defined for atom group "${key}".`);
    }
    const ef = c.get('enableForces');
    if (ef !== undefined) this.noforce = !parseBool(ef, 'enableForces');
    c.get('printAtomIDs');
    c.check(`atom group "${key}"`);
    const n = this.ids.length;
    this.px = new Float64Array(n); this.py = new Float64Array(n); this.pz = new Float64Array(n);
    this.gx = new Float64Array(n); this.gy = new Float64Array(n); this.gz = new Float64Array(n);
  }

  /** update_total_mass: the sum of the masses (std::accumulate) and each atom's weight m/M. */
  setup(): void {
    if (this.dummy) { this.totalMass = 1.0; return; }
    const n = this.ids.length;
    let total = 0.0;
    for (let i = 0; i < n; i++) total += this.atoms.mass[this.index[i]];
    this.totalMass = total;
    this.weight = new Float64Array(n);
    for (let i = 0; i < n; i++) this.weight[i] = this.atoms.mass[this.index[i]] / total;
    if (total < 1e-15) throw colvarsError(`Error: ${this.description} has zero total mass.`);
  }

  /** read_positions and calc_center_of_mass. */
  calc(): void {
    if (this.dummy) { this.com[0] = this.dummy[0]; this.com[1] = this.dummy[1]; this.com[2] = this.dummy[2]; return; }
    const p = this.atoms.pos;
    let cx = 0.0, cy = 0.0, cz = 0.0;
    for (let i = 0; i < this.ids.length; i++) {
      const k = 3 * this.index[i];
      const m = this.atoms.mass[this.index[i]];
      this.px[i] = p[k]; this.py[i] = p[k + 1]; this.pz[i] = p[k + 2];
      cx += m * p[k]; cy += m * p[k + 1]; cz += m * p[k + 2];
    }
    this.com[0] = cx / this.totalMass; this.com[1] = cy / this.totalMass; this.com[2] = cz / this.totalMass;
  }

  /** set_weighted_gradient: grad_i = (m_i / M) grad. */
  setWeightedGradient(g: number[]): void {
    if (this.dummy) return;
    for (let i = 0; i < this.ids.length; i++) {
      const w = this.weight[i];
      this.gx[i] = w * g[0]; this.gy[i] = w * g[1]; this.gz[i] = w * g[2];
    }
  }

  private checkForces(): void {
    if (this.noforce) throw colvarsError('Error: sending a force to a group that has "enableForces" set to off.');
  }

  /** apply_colvar_force: each atom gets force times its gradient. */
  applyColvarForce(force: number): void {
    if (this.dummy) return;
    this.checkForces();
    for (let i = 0; i < this.ids.length; i++) this.atoms.addForce(this.index[i], force * this.gx[i], force * this.gy[i], force * this.gz[i]);
  }

  /** apply_force: each atom gets its weight times the vector force. */
  applyForce(f: number[]): void {
    if (this.dummy) return;
    this.checkForces();
    for (let i = 0; i < this.ids.length; i++) {
      const w = this.weight[i];
      this.atoms.addForce(this.index[i], w * f[0], w * f[1], w * f[2]);
    }
  }
}

/** colvar::cvc: one component of a colvar. */
abstract class Component {
  abstract readonly functionType: string;
  abstract readonly type: ValueType;
  x = new Float64Array(1);
  name = '';
  coeff = 1.0;
  exponent = 1;
  readonly groups: AtomGroup[] = [];
  readonly bc = new Boundaries();
  private pbc = true;
  /** f_cvc_periodic: enabled, its period and wrap centre (cvc::init_as_periodic_angle); available per style. */
  periodic = false;
  period = 0.0;
  wrapCenter = 0.0;
  /**
   * f_cvc_lower_boundary / f_cvc_upper_boundary: the intrinsic boundaries of a scalar component (cvc::init_as_distance
   * sets lower 0; init_as_angle 0 and 180; init_as_periodic_angle -180 and 180; init_scalar_boundaries), null if none.
   */
  lowerBoundary: number | null = null;
  upperBoundary: number | null = null;

  /** cvc::init: the keywords every component reads. */
  protected initBase(c: Conf): void {
    const name = c.get('name');
    if (name !== undefined) this.name = name.trim();
    const coeff = c.get('componentCoeff');
    if (coeff !== undefined) this.coeff = parseReal(coeff, 'componentCoeff');
    const exp = c.get('componentExp');
    if (exp !== undefined) this.exponent = parseInt10(exp, 'componentExp');
    const period = c.get('period');
    if (period !== undefined) this.period = parseReal(period, 'period');
    const wrap = c.get('wrapAround');
    if (wrap !== undefined) this.wrapCenter = parseReal(wrap, 'wrapAround');
    if (this.period !== 0.0) {
      if (!this.periodicAvailable()) throw colvarsError(`Error: invalid use of period and/or wrapAround in a "${this.functionType}" component.`);
      this.periodic = true;
    }
    if (this.wrapCenter !== 0.0 && !this.periodic) throw colvarsError('Error: wrapAround was defined for a non-periodic component.');
    c.get('debugGradients');
    const nopbc = c.get('forceNoPBC');
    if (nopbc !== undefined) this.pbc = !parseBool(nopbc, 'forceNoPBC');
    const scalable = c.get('scalable');
    if (scalable !== undefined) parseBool(scalable, 'scalable');
  }

  /** provide(f_cvc_periodic): whether period and wrapAround may be given. */
  protected periodicAvailable(): boolean { return false; }

  /** cvc::parse_group. */
  protected group(atoms: ColvarsAtoms, c: Conf, key: string, cvName: string): AtomGroup {
    const conf = c.get(key);
    if (conf === undefined) throw colvarsError(`Error: definition for atom group "${key}" not found.`);
    const g = new AtomGroup(atoms, key, `atom group "${key}" of colvar "${cvName}"`, conf);
    this.groups.push(g);
    return g;
  }

  /** cvc::read_data: the boundaries of the proxy, or none with forceNoPBC. */
  readData(bc: Boundaries): void {
    if (this.pbc) this.bc.copyFrom(bc); else this.bc.reset();
    for (const g of this.groups) g.calc();
  }

  /** cvc::calc_value; stepRelative is the module's step since its last restart (coordNum's pair list). */
  abstract calcValue(stepRelative: number): void;
  calcGradients(): void {}
  /** cvc::apply_force: the colvar force on each atom group. */
  applyForce(f: Float64Array): void {
    for (const g of this.groups) if (!g.noforce) g.applyColvarForce(f[0]);
  }

  /** cvc::dist2 (the scalar default; a periodic component takes the nearest period). */
  dist2(a: Float64Array, b: Float64Array): number {
    let d = a[0] - b[0];
    if (this.periodic) d -= Math.floor(d / this.period + 0.5) * this.period;
    return d * d;
  }

  /** cvc::dist2_lgrad. */
  dist2Lgrad(a: Float64Array, b: Float64Array): Float64Array {
    let d = a[0] - b[0];
    if (this.periodic) d -= Math.floor(d / this.period + 0.5) * this.period;
    return Float64Array.of(2.0 * d);
  }

  /** cvc::wrap: a periodic value into [wrapCenter - period/2, wrapCenter + period/2). */
  wrap(x: Float64Array): void {
    if (!this.periodic) return;
    x[0] -= Math.floor((x[0] - this.wrapCenter) / this.period + 0.5) * this.period;
  }
}

/** colvar::distance: |COM2 - COM1| (minimum image). */
class Distance extends Component {
  readonly functionType: string = 'distance';
  // cvc::init_as_distance: lower boundary 0
  lowerBoundary: number | null = 0.0;
  readonly type: ValueType = 'scalar';
  protected g1: AtomGroup;
  protected g2: AtomGroup;
  protected readonly dist = [0, 0, 0];

  constructor(atoms: ColvarsAtoms, c: Conf, cvName: string) {
    super();
    this.initBase(c);
    this.g1 = this.group(atoms, c, 'group1', cvName);
    this.g2 = this.group(atoms, c, 'group2', cvName);
    const ostf = c.get('oneSiteTotalForce');
    if (ostf !== undefined) parseBool(ostf, 'oneSiteTotalForce');
  }

  calcValue(): void {
    this.bc.distance(this.g1.com, this.g2.com, this.dist);
    const d = this.dist;
    this.x[0] = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  }

  calcGradients(): void {
    // rvector::unit: v / |v|, or (1, 0, 0) for a zero vector
    const d = this.dist;
    const n = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const u = n > 0.0 ? [d[0] / n, d[1] / n, d[2] / n] : [1.0, 0.0, 0.0];
    this.g1.setWeightedGradient([-1.0 * u[0], -1.0 * u[1], -1.0 * u[2]]);
    this.g2.setWeightedGradient(u);
  }
}

/** colvar::distance_vec: COM2 - COM1 (minimum image), a 3-vector. */
class DistanceVec extends Distance {
  readonly functionType: string = 'distanceVec';
  readonly type: ValueType = '3vector';
  private readonly tmp = [0, 0, 0];

  constructor(atoms: ColvarsAtoms, c: Conf, cvName: string) {
    super(atoms, c, cvName);
    this.x = new Float64Array(3);
  }

  calcValue(): void {
    this.bc.distance(this.g1.com, this.g2.com, this.dist);
    this.x[0] = this.dist[0]; this.x[1] = this.dist[1]; this.x[2] = this.dist[2];
  }

  calcGradients(): void {}

  applyForce(f: Float64Array): void {
    if (!this.g1.noforce) this.g1.applyForce([-1.0 * f[0], -1.0 * f[1], -1.0 * f[2]]);
    if (!this.g2.noforce) this.g2.applyForce([f[0], f[1], f[2]]);
  }

  /** |position_distance(x1, x2)|^2. */
  dist2(a: Float64Array, b: Float64Array): number {
    this.bc.distance(a, b, this.tmp);
    const t = this.tmp;
    return t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
  }

  /** 2 position_distance(x2, x1). */
  dist2Lgrad(a: Float64Array, b: Float64Array): Float64Array {
    this.bc.distance(b, a, this.tmp);
    return Float64Array.of(2.0 * this.tmp[0], 2.0 * this.tmp[1], 2.0 * this.tmp[2]);
  }
}

/** colvar::cartesian: the selected coordinates of the atoms, atom by atom. */
class Cartesian extends Component {
  readonly functionType = 'cartesian';
  readonly type: ValueType = 'vector';
  private readonly atomsGroup: AtomGroup;
  private readonly axes: number[] = [];

  constructor(atoms: ColvarsAtoms, c: Conf, cvName: string) {
    super();
    this.initBase(c);
    this.atomsGroup = this.group(atoms, c, 'atoms', cvName);
    const use = (k: string): boolean => { const v = c.get(k); return v === undefined ? true : parseBool(v, k); };
    if (use('useX')) this.axes.push(0);
    if (use('useY')) this.axes.push(1);
    if (use('useZ')) this.axes.push(2);
    if (this.axes.length === 0) throw colvarsError('Error: a "cartesian" component was defined with all three axes disabled.');
    this.x = new Float64Array(this.atomsGroup.ids.length * this.axes.length);
  }

  calcValue(): void {
    const g = this.atomsGroup, dim = this.axes.length;
    for (let ia = 0; ia < g.ids.length; ia++) {
      const pos = [g.px[ia], g.py[ia], g.pz[ia]];
      for (let j = 0; j < dim; j++) this.x[dim * ia + j] = pos[this.axes[j]];
    }
  }

  applyForce(f: Float64Array): void {
    const g = this.atomsGroup, dim = this.axes.length;
    if (g.noforce) return;
    // as the library does, one force vector is reused from atom to atom (axes not used keep their earlier values)
    const fv = [0.0, 0.0, 0.0];
    for (let ia = 0; ia < g.ids.length; ia++) {
      for (let j = 0; j < dim; j++) fv[this.axes[j]] = f[dim * ia + j];
      g.atoms.addForce(g.index[ia], fv[0], fv[1], fv[2]);
    }
  }

  /** (x1 - x2).norm2() of a vector1d. */
  dist2(a: Float64Array, b: Float64Array): number {
    let r = 0.0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; r += d * d; }
    return r;
  }

  dist2Lgrad(a: Float64Array, b: Float64Array): Float64Array {
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = 2.0 * (a[i] - b[i]);
    return out;
  }
}

/** 180/PI of colvarcomp_angles.cpp (cvm's PI is 3.14159265358979323846). */
const DEG = 180.0 / Math.PI;

const dot3 = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a: number[]): number => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
/** rvector::outer (the cross product, in the library's order of operations). */
const outer3 = (a: number[], b: number[]): number[] => [a[1] * b[2] - b[1] * a[2], -a[0] * b[2] + b[0] * a[2], a[0] * b[1] - b[0] * a[1]];
/** rvector::unit: v / |v|, or (1, 0, 0) for a zero vector. */
const unit3 = (v: number[]): number[] => { const n = norm3(v); return n > 0.0 ? [v[0] / n, v[1] / n, v[2] / n] : [1.0, 0.0, 0.0]; };

/** colvar::angle: the angle at the centre of mass of group2, in degrees. */
class Angle extends Component {
  readonly functionType = 'angle';
  // cvc::init_as_angle: boundaries 0 and 180
  lowerBoundary: number | null = 0.0;
  upperBoundary: number | null = 180.0;
  readonly type: ValueType = 'scalar';
  private readonly g1: AtomGroup;
  private readonly g2: AtomGroup;
  private readonly g3: AtomGroup;
  private r21 = [0, 0, 0];
  private r23 = [0, 0, 0];
  private r21l = 0;
  private r23l = 0;

  constructor(atoms: ColvarsAtoms, c: Conf, cvName: string) {
    super();
    this.initBase(c);
    this.g1 = this.group(atoms, c, 'group1', cvName);
    this.g2 = this.group(atoms, c, 'group2', cvName);
    this.g3 = this.group(atoms, c, 'group3', cvName);
    const ostf = c.get('oneSiteTotalForce');
    if (ostf !== undefined) parseBool(ostf, 'oneSiteTotalForce');
  }

  calcValue(): void {
    const r21 = [0, 0, 0], r23 = [0, 0, 0];
    this.bc.distance(this.g2.com, this.g1.com, r21);
    this.bc.distance(this.g2.com, this.g3.com, r23);
    this.r21 = r21; this.r23 = r23;
    this.r21l = norm3(r21); this.r23l = norm3(r23);
    const cosTheta = dot3(r21, r23) / (this.r21l * this.r23l);
    this.x[0] = DEG * acos(cosTheta);
  }

  calcGradients(): void {
    const { r21, r23, r21l, r23l } = this;
    const cosTheta = dot3(r21, r23) / (r21l * r23l);
    const dxdcos = -1.0 / Math.sqrt(1.0 - cosTheta * cosTheta);
    const c1 = DEG * dxdcos * (1.0 / r21l), c3 = DEG * dxdcos * (1.0 / r23l);
    const mc = -1.0 * cosTheta;
    const d1 = [0, 1, 2].map((k) => c1 * (r23[k] / r23l + (mc * r21[k]) / r21l));
    const d3 = [0, 1, 2].map((k) => c3 * (r21[k] / r21l + (mc * r23[k]) / r23l));
    this.g1.setWeightedGradient(d1);
    this.g2.setWeightedGradient([0, 1, 2].map((k) => (d1[k] + d3[k]) * -1.0));
    this.g3.setWeightedGradient(d3);
  }
}

/** colvar::dihedral: the dihedral angle of four centres of mass, in degrees, periodic over 360. */
class Dihedral extends Component {
  readonly functionType = 'dihedral';
  // cvc::init_as_periodic_angle: boundaries -180 and 180
  lowerBoundary: number | null = -180.0;
  upperBoundary: number | null = 180.0;
  readonly type: ValueType = 'scalar';
  protected periodicAvailable(): boolean { return true; }
  private readonly g: AtomGroup[];
  private r12 = [0, 0, 0];
  private r23 = [0, 0, 0];
  private r34 = [0, 0, 0];

  constructor(atoms: ColvarsAtoms, c: Conf, cvName: string) {
    super();
    // cvc::init_as_periodic_angle: periodic, period 360
    this.periodic = true;
    this.period = 360.0;
    this.initBase(c);
    this.g = ['group1', 'group2', 'group3', 'group4'].map((k) => this.group(atoms, c, k, cvName));
    const ostf = c.get('oneSiteTotalForce');
    if (ostf !== undefined) parseBool(ostf, 'oneSiteTotalForce');
  }

  calcValue(): void {
    const [g1, g2, g3, g4] = this.g;
    const r12 = [0, 0, 0], r23 = [0, 0, 0], r34 = [0, 0, 0];
    // the usual sign convention: r12 = r2 - r1
    this.bc.distance(g1.com, g2.com, r12);
    this.bc.distance(g2.com, g3.com, r23);
    this.bc.distance(g3.com, g4.com, r34);
    this.r12 = r12; this.r23 = r23; this.r34 = r34;
    const n1 = outer3(r12, r23), n2 = outer3(r23, r34);
    const cosPhi = dot3(n1, n2);
    const sinPhi = dot3(n1, r34) * norm3(r23);
    // std::atan2: JavaScript's Math.atan2 may differ from the C library's in the last bit
    this.x[0] = DEG * Math.atan2(sinPhi, cosPhi);
    this.wrap(this.x);
  }

  calcGradients(): void {
    const { r12, r23, r34 } = this;
    const A = outer3(r12, r23), B = outer3(r23, r34);
    const nG = norm3(r23);
    const A2 = dot3(A, A), B2 = dot3(B, B);
    const K = DEG;
    const a1 = K * nG / A2, a3 = K * nG / B2;
    const fa = dot3(r12, r23) / (A2 * nG), fb = dot3(r34, r23) / (B2 * nG);
    const f1 = A.map((v) => a1 * v);
    const f2 = [0, 1, 2].map((k) => K * (fa * A[k] + fb * B[k]));
    const f3 = B.map((v) => a3 * v);
    const [g1, g2, g3, g4] = this.g;
    g1.setWeightedGradient(f1.map((v) => -v));
    g2.setWeightedGradient([0, 1, 2].map((k) => f2[k] + f1[k]));
    g3.setWeightedGradient([0, 1, 2].map((k) => -f3[k] - f2[k]));
    g4.setWeightedGradient(f3);
  }
}

/** colvar::distance_z: the projection of main - ref on a fixed axis, or on the axis from ref to ref2. */
class DistanceZ extends Component {
  readonly functionType: string = 'distanceZ';
  readonly type: ValueType = 'scalar';
  protected periodicAvailable(): boolean { return true; }
  protected readonly main: AtomGroup;
  protected readonly ref1: AtomGroup;
  protected readonly ref2: AtomGroup | null;
  protected readonly fixedAxis: boolean;
  protected axis = [0.0, 0.0, 1.0];
  protected axisNorm = 0;
  protected distV = [0, 0, 0];

  constructor(atoms: ColvarsAtoms, c: Conf, cvName: string) {
    super();
    this.initBase(c);
    this.main = this.group(atoms, c, 'main', cvName);
    this.ref1 = this.group(atoms, c, 'ref', cvName);
    this.ref2 = c.all('ref2').length ? this.group(atoms, c, 'ref2', cvName) : null;
    const ax = c.get('axis');
    if (this.ref2) {
      this.fixedAxis = false;
    } else {
      this.fixedAxis = true;
      if (ax !== undefined) {
        const v = splitValues(ax);
        if (v.length !== 1 || !Array.isArray(v[0]) || v[0].length !== 3) throw colvarsError(`Error: could not read the vector of "axis" from "${ax.trim()}".`);
        let a = [...v[0]];
        const n2 = dot3(a, a);
        if (n2 === 0.0) throw colvarsError('Axis vector is zero!');
        if (n2 !== 1.0) a = unit3(a);
        this.axis = a;
      }
    }
    const ostf = c.get('oneSiteTotalForce');
    if (ostf !== undefined) parseBool(ostf, 'oneSiteTotalForce');
  }

  calcValue(): void {
    const d = [0, 0, 0];
    if (this.fixedAxis) {
      this.bc.distance(this.ref1.com, this.main.com, d);
    } else {
      const R1 = this.ref1.com, R2 = this.ref2!.com;
      const C = [0.5 * (R1[0] + R2[0]), 0.5 * (R1[1] + R2[1]), 0.5 * (R1[2] + R2[2])];
      this.bc.distance(C, this.main.com, d);
      const ax = [0, 0, 0];
      this.bc.distance(R1, R2, ax);
      this.axisNorm = norm3(ax);
      this.axis = unit3(ax);
    }
    this.distV = d;
    this.x[0] = dot3(this.axis, d);
    this.wrap(this.x);
  }

  calcGradients(): void {
    this.main.setWeightedGradient(this.axis);
    if (this.fixedAxis) {
      this.ref1.setWeightedGradient(this.axis.map((v) => -1.0 * v));
    } else {
      const x = this.x[0];
      const perp = [0, 1, 2].map((k) => this.distV[k] - this.axis[k] * x);
      this.ref1.setWeightedGradient([0, 1, 2].map((k) => -perp[k] / this.axisNorm - 0.5 * this.axis[k]));
      this.ref2!.setWeightedGradient([0, 1, 2].map((k) => perp[k] / this.axisNorm - 0.5 * this.axis[k]));
    }
  }
}

/** colvar::distance_xy: the length of the part of main - ref perpendicular to the axis. */
class DistanceXY extends DistanceZ {
  readonly functionType: string = 'distanceXY';
  // distance_xy: cvc::init_as_distance, lower boundary 0
  lowerBoundary: number | null = 0.0;
  // provide(f_cvc_periodic, false): the inherited distanceZ flag is withdrawn
  protected periodicAvailable(): boolean { return false; }
  private ortho = [0, 0, 0];

  calcValue(): void {
    const d = [0, 0, 0];
    this.bc.distance(this.ref1.com, this.main.com, d);
    if (!this.fixedAxis) {
      const v12 = [0, 0, 0];
      this.bc.distance(this.ref1.com, this.ref2!.com, v12);
      this.axisNorm = norm3(v12);
      this.axis = unit3(v12);
    }
    this.distV = d;
    const p = dot3(d, this.axis);
    this.ortho = [0, 1, 2].map((k) => d[k] - p * this.axis[k]);
    this.x[0] = norm3(this.ortho);
  }

  calcGradients(): void {
    // as the library does, a zero value leaves the gradients of the step before
    if (this.x[0] === 0.0) return;
    const xInv = 1.0 / this.x[0];
    const o = this.ortho;
    if (this.fixedAxis) {
      const c = -1.0 * xInv;
      this.ref1.setWeightedGradient(o.map((v) => c * v));
      this.main.setWeightedGradient(o.map((v) => xInv * v));
    } else {
      const A = dot3(this.distV, this.axis) / this.axisNorm;
      const c1 = (A - 1.0) * xInv, c2 = -A * xInv, c3 = 1.0 * xInv;
      this.ref1.setWeightedGradient(o.map((v) => c1 * v));
      this.ref2!.setWeightedGradient(o.map((v) => c2 * v));
      this.main.setWeightedGradient(o.map((v) => c3 * v));
    }
  }
}

/** colvar::gyration: the radius of gyration of the atoms about their own centre of geometry. */
class Gyration extends Component {
  readonly functionType = 'gyration';
  // cvc::init_as_distance: lower boundary 0
  lowerBoundary: number | null = 0.0;
  readonly type: ValueType = 'scalar';
  private readonly atomsGroup: AtomGroup;

  constructor(atoms: ColvarsAtoms, c: Conf, cvName: string) {
    super();
    this.initBase(c);
    this.atomsGroup = this.group(atoms, c, 'atoms', cvName);
    if (this.atomsGroup.dummy) throw colvarsError('Error: the "atoms" group of a gyration component cannot be a dummy atom.');
  }

  calcValue(): void {
    // gyration::init enables f_ag_center on a reference of one point at the origin: the positions are moved by
    // -COG (calc_apply_roto_translation), then by the reference centre (0, 0, 0); fit gradients stay off
    const g = this.atomsGroup, n = g.ids.length;
    let cx = 0.0, cy = 0.0, cz = 0.0;
    for (let i = 0; i < n; i++) { cx += g.px[i]; cy += g.py[i]; cz += g.pz[i]; }
    cx /= n; cy /= n; cz /= n;
    const tx = -1.0 * cx, ty = -1.0 * cy, tz = -1.0 * cz;
    for (let i = 0; i < n; i++) {
      g.px[i] = (g.px[i] + tx) + 0.0; g.py[i] = (g.py[i] + ty) + 0.0; g.pz[i] = (g.pz[i] + tz) + 0.0;
    }
    // the positions are stored x of every atom, then y, then z, and summed in that order
    let r = 0.0;
    for (let i = 0; i < n; i++) r += g.px[i] * g.px[i];
    for (let i = 0; i < n; i++) r += g.py[i] * g.py[i];
    for (let i = 0; i < n; i++) r += g.pz[i] * g.pz[i];
    this.x[0] = Math.sqrt(r / n);
  }

  calcGradients(): void {
    const g = this.atomsGroup, n = g.ids.length;
    const drdx = 1.0 / (n * this.x[0]);
    for (let i = 0; i < n; i++) { g.gx[i] = drdx * g.px[i]; g.gy[i] = drdx * g.py[i]; g.gz[i] = drdx * g.pz[i]; }
  }
}

/** coordnum::switching_function: (1 - l2^(n/2)) / (1 - l2^(d/2)), with its log-derivative, the pair-list tolerance. */
const switchingFunction = (l2: number, en: number, ed: number, usePairlist: boolean, gradients: boolean, tol: number, out: { dFdl2: number }): number => {
  const en2 = Math.trunc(en / 2), ed2 = Math.trunc(ed / 2);
  const xn = integerPower(l2, en2), xd = integerPower(l2, ed2);
  const epsL2 = 1.0e-7;
  const h = l2 - 1.0;
  const en2r = en2, ed2r = ed2;
  let f0: number;
  if (Math.abs(h) < epsL2) {
    // order-2 Taylor expansion about l2 = 1
    const c0 = en2r / ed2r;
    const c1 = (en2r * (en2r - ed2r)) / (2.0 * ed2r);
    const c2 = (en2r * (en2r - ed2r) * (2.0 * en2r - ed2r - 3.0)) / (12.0 * ed2r);
    f0 = c0 + h * (c1 + h * c2);
  } else {
    f0 = (1.0 - xn) / (1.0 - xd);
  }
  let func: number, inv = 0;
  if (usePairlist) {
    inv = 1 / (1.0 - tol);
    func = (f0 - tol) * inv;
  } else {
    func = f0;
  }
  if (func < 0) return 0;
  if (gradients) {
    let logDeriv: number;
    if (Math.abs(h) < epsL2) {
      const g0 = 0.5 * (en2r - ed2r);
      const g1 = ((en2r - ed2r) * (en2r + ed2r - 6.0)) / 12.0;
      logDeriv = g0 + h * g1;
    } else {
      logDeriv = (ed2r * xd / ((1.0 - xd) * l2)) - (en2r * xn / ((1.0 - xn) * l2));
    }
    out.dFdl2 = usePairlist ? f0 * inv * logDeriv : func * logDeriv;
  }
  return func;
};

/**
 * colvar::coordnum and its kinds: coordNum (the sum over atom pairs of group1 and group2 of the switching function of
 * their distance), selfCoordNum (over the pairs i < j of group1) and groupCoord (between the two centres of mass).
 */
class CoordNum extends Component {
  readonly functionType: string;
  readonly type: ValueType = 'scalar';
  private readonly g1: AtomGroup;
  private readonly g2: AtomGroup | null;
  private readonly center1: boolean;
  private readonly center2: boolean;
  private r0 = [0, 0, 0];
  private invR0 = [0, 0, 0];
  private invR0sq = [0, 0, 0];
  private en = 6;
  private ed = 12;
  private tolerance = 0.0;
  private tolL2Max = 1.0e20;
  private pairlistFreq = 100;
  private pairlist: Uint8Array | null = null;

  constructor(atoms: ColvarsAtoms, c: Conf, cvName: string, ctx: ComponentContext, kind: 'coordNum' | 'selfCoordNum' | 'groupCoord' = 'coordNum') {
    super();
    this.functionType = kind;
    // the default cutoff, 4 A in the units of the input (proxy->angstrom_to_internal)
    const r0 = 4.0 * ctx.angstrom;
    this.updateCutoffs([r0, r0, r0]);
    this.initBase(c);
    this.g1 = this.group(atoms, c, 'group1', cvName);
    if (this.g1.dummy) throw colvarsError('Error: group1 may not be a dummy atom');
    if (kind === 'selfCoordNum') {
      this.g2 = null;
      this.center1 = false;
      this.center2 = false;
    } else {
      const g2 = this.group(atoms, c, 'group2', cvName);
      this.g2 = g2;
      const shared = this.g1.ids.find((id) => g2.ids.includes(id));
      if (shared !== undefined) throw colvarsError(`Error: group1 and group2 share a common atom (number: ${shared})`);
      if (kind === 'coordNum') {
        const b1 = c.get('group1CenterOnly'), b2 = c.get('group2CenterOnly');
        const dummy2 = g2.dummy !== null;
        this.center1 = b1 !== undefined ? parseBool(b1, 'group1CenterOnly') : dummy2;
        this.center2 = b2 !== undefined ? parseBool(b2, 'group2CenterOnly') : dummy2;
      } else {
        // groupCoord: hard-coded centres of mass
        this.center1 = true;
        this.center2 = true;
      }
    }
    const cut = c.get('cutoff');
    const cut3 = c.get('cutoff3');
    if (cut3 !== undefined) {
      if (cut !== undefined) throw colvarsError('Error: cannot specify "cutoff" and "cutoff3" at the same time.');
      const v = splitValues(cut3);
      if (v.length !== 1 || !Array.isArray(v[0]) || v[0].length !== 3) throw colvarsError(`Error: could not read the vector of "cutoff3" from "${cut3.trim()}".`);
      // remove meaningless negative signs
      this.updateCutoffs(v[0].map((r) => (r < 0.0 ? r * -1.0 : r)));
    } else if (cut !== undefined) {
      const r = parseReal(cut, 'cutoff');
      this.updateCutoffs([r, r, r]);
    }
    const en = c.get('expNumer'), ed = c.get('expDenom');
    if (en !== undefined) this.en = parseInt10(en, 'expNumer');
    if (ed !== undefined) this.ed = parseInt10(ed, 'expDenom');
    if (this.en % 2 || this.ed % 2) throw colvarsError('Error: odd exponent(s) provided, can only use even ones.');
    if (this.en <= 0 || this.ed <= 0) throw colvarsError('Error: negative exponent(s) provided.');
    // every kind but groupCoord may use a pair list
    const tol = kind !== 'groupCoord' ? c.get('tolerance') : undefined;
    if (tol !== undefined) this.tolerance = parseReal(tol, 'tolerance');
    if (this.tolerance > 0) {
      this.computeTolL2Max();
      const pf = c.get('pairListFrequency');
      if (pf !== undefined) this.pairlistFreq = parseInt10(pf, 'pairListFrequency');
      if (!(this.pairlistFreq > 0)) throw colvarsError('Error: non-positive pairlistfrequency provided.');
      const n = this.g1.ids.length;
      const pairs = this.g2 === null ? (n * (n - 1)) / 2 : (this.center1 ? 1 : n) * (this.center2 ? 1 : this.g2.ids.length);
      this.pairlist = new Uint8Array(pairs).fill(1);
    }
    // coordnum::init: init_scalar_boundaries(0.0, num_pairs)
    const n1 = this.g1.ids.length;
    this.lowerBoundary = 0.0;
    this.upperBoundary = this.g2 === null ? Math.floor((n1 * (n1 - 1)) / 2) : (this.center1 ? 1 : n1) * (this.center2 ? 1 : this.g2.ids.length);
  }

  private updateCutoffs(r0: number[]): void {
    this.r0 = r0;
    this.invR0 = [1.0 / r0[0], 1.0 / r0[1], 1.0 / r0[2]];
    this.invR0sq = [this.invR0[0] * this.invR0[0], this.invR0[1] * this.invR0[1], this.invR0[2] * this.invR0[2]];
  }

  /** compute_tolerance_l2_max: the l2 where the tolerance-shifted switching function reaches 0 (Newton). */
  private computeTolL2Max(): void {
    let l2 = 1.001;
    const d = { dFdl2: 0.0 };
    for (let i = 0; i < 1000000; i++) {
      const F = switchingFunction(l2, this.en, this.ed, true, true, this.tolerance, d);
      if (Math.abs(F) < 1.0e-6 || Math.abs(d.dFdl2) < 1.0e-9) break;
      l2 -= F / d.dFdl2;
    }
    this.tolL2Max = l2;
  }

  /** The pair term of compute_pair_coordnum: F, with the gradient G added to (g1, i) negated and to (g2, j). */
  private pair(p1: number[], p2: number[], usePl: boolean, diff: number[], d: { dFdl2: number }, G: number[]): number {
    this.bc.distance(p1, p2, diff);
    const sx = diff[0] * this.invR0[0], sy = diff[1] * this.invR0[1], sz = diff[2] * this.invR0[2];
    const l2 = sx * sx + sy * sy + sz * sz;
    if (usePl && l2 > this.tolL2Max) return -1;
    d.dFdl2 = 0.0;
    const F = switchingFunction(l2, this.en, this.ed, usePl, true, this.tolerance, d);
    if (F > 0.0) {
      G[0] = d.dFdl2 * ((2.0 * this.invR0sq[0]) * diff[0]);
      G[1] = d.dFdl2 * ((2.0 * this.invR0sq[1]) * diff[1]);
      G[2] = d.dFdl2 * ((2.0 * this.invR0sq[2]) * diff[2]);
    }
    return F;
  }

  calcValue(stepRelative: number): void {
    const usePl = this.pairlist !== null;
    const rebuild = usePl && stepRelative % this.pairlistFreq === 0;
    if (this.g2 === null) { this.selfLoop(usePl, rebuild); return; }
    const g1 = this.g1, g2 = this.g2;
    // cvc::read_data reset the gradients; the loop sums the pair terms into them
    g1.gx.fill(0); g1.gy.fill(0); g1.gz.fill(0);
    g2.gx.fill(0); g2.gy.fill(0); g2.gz.fill(0);
    const n1 = this.center1 ? 1 : g1.ids.length, n2 = this.center2 ? 1 : g2.ids.length;
    const com1 = g1.com, com2 = g2.com;
    const cg1 = [0.0, 0.0, 0.0], cg2 = [0.0, 0.0, 0.0];
    const diff = [0, 0, 0];
    const p1 = [0, 0, 0], p2 = [0, 0, 0];
    const d = { dFdl2: 0.0 };
    let x = 0.0;
    let k = 0;
    for (let i = 0; i < n1; i++) {
      if (this.center1) { p1[0] = com1[0]; p1[1] = com1[1]; p1[2] = com1[2]; } else { p1[0] = g1.px[i]; p1[1] = g1.py[i]; p1[2] = g1.pz[i]; }
      for (let j = 0; j < n2; j++) {
        if (this.center2) { p2[0] = com2[0]; p2[1] = com2[1]; p2[2] = com2[2]; } else { p2[0] = g2.px[j]; p2[1] = g2.py[j]; p2[2] = g2.pz[j]; }
        const within = usePl ? (this.pairlist![k] === 1 || rebuild) : true;
        let partial = 0.0;
        if (within) {
          this.bc.distance(p1, p2, diff);
          const sx = diff[0] * this.invR0[0], sy = diff[1] * this.invR0[1], sz = diff[2] * this.invR0[2];
          const l2 = sx * sx + sy * sy + sz * sz;
          if (!(usePl && l2 > this.tolL2Max)) {
            d.dFdl2 = 0.0;
            const F = switchingFunction(l2, this.en, this.ed, usePl, true, this.tolerance, d);
            if (F > 0.0) {
              const Gx = d.dFdl2 * ((2.0 * this.invR0sq[0]) * diff[0]);
              const Gy = d.dFdl2 * ((2.0 * this.invR0sq[1]) * diff[1]);
              const Gz = d.dFdl2 * ((2.0 * this.invR0sq[2]) * diff[2]);
              if (this.center1) { cg1[0] += -1.0 * Gx; cg1[1] += -1.0 * Gy; cg1[2] += -1.0 * Gz; } else { g1.gx[i] += -1.0 * Gx; g1.gy[i] += -1.0 * Gy; g1.gz[i] += -1.0 * Gz; }
              if (this.center2) { cg2[0] += Gx; cg2[1] += Gy; cg2[2] += Gz; } else { g2.gx[j] += Gx; g2.gy[j] += Gy; g2.gz[j] += Gz; }
            }
            partial = F;
          }
        }
        if (usePl && rebuild) this.pairlist![k] = partial > 0.0 ? 1 : 0;
        x += partial;
        if (usePl) k++;
      }
    }
    this.x[0] = x;
    if (this.center1) g1.setWeightedGradient(cg1);
    if (this.center2) g2.setWeightedGradient(cg2);
  }

  /** selfcoordnum_sequential_loop: the pairs i < j of group1, in order. */
  private selfLoop(usePl: boolean, rebuild: boolean): void {
    const g = this.g1, n = g.ids.length;
    g.gx.fill(0); g.gy.fill(0); g.gz.fill(0);
    const diff = [0, 0, 0], p1 = [0, 0, 0], p2 = [0, 0, 0], G = [0, 0, 0];
    const d = { dFdl2: 0.0 };
    let x = 0.0;
    let k = 0;
    for (let i = 0; i < n - 1; i++) {
      p1[0] = g.px[i]; p1[1] = g.py[i]; p1[2] = g.pz[i];
      for (let j = i + 1; j < n; j++) {
        p2[0] = g.px[j]; p2[1] = g.py[j]; p2[2] = g.pz[j];
        const within = usePl ? (this.pairlist![k] === 1 || rebuild) : true;
        let partial = 0.0;
        if (within) {
          const F = this.pair(p1, p2, usePl, diff, d, G);
          if (F > 0.0) {
            g.gx[i] += -1.0 * G[0]; g.gy[i] += -1.0 * G[1]; g.gz[i] += -1.0 * G[2];
            g.gx[j] += G[0]; g.gy[j] += G[1]; g.gz[j] += G[2];
          }
          partial = F < 0 ? 0.0 : F;
        }
        if (usePl && rebuild) this.pairlist![k] = partial > 0.0 ? 1 : 0;
        x += partial;
        if (usePl) k++;
      }
    }
    this.x[0] = x;
  }
}

/** What a component needs from the module: the unit of length (proxy->angstrom_to_internal). */
export interface ComponentContext {
  /** force->angstrom of the unit style. */
  angstrom: number;
}

/** The components of this port, and the ones of the library it refuses. */
const COMPONENTS: Record<string, (atoms: ColvarsAtoms, c: Conf, cvName: string, ctx: ComponentContext) => Component> = {
  distance: (a, c, n) => new Distance(a, c, n),
  distancevec: (a, c, n) => new DistanceVec(a, c, n),
  cartesian: (a, c, n) => new Cartesian(a, c, n),
  gyration: (a, c, n) => new Gyration(a, c, n),
  coordnum: (a, c, n, x) => new CoordNum(a, c, n, x),
  selfcoordnum: (a, c, n, x) => new CoordNum(a, c, n, x, 'selfCoordNum'),
  groupcoord: (a, c, n, x) => new CoordNum(a, c, n, x, 'groupCoord'),
  angle: (a, c, n) => new Angle(a, c, n),
  dihedral: (a, c, n) => new Dihedral(a, c, n),
  distancez: (a, c, n) => new DistanceZ(a, c, n),
  distancexy: (a, c, n) => new DistanceXY(a, c, n),
};
const LIBRARY_COMPONENTS = [
  'distanceDir', 'distanceInv', 'distancePairs', 'dipoleMagnitude', 'dipoleAngle',
  'hBond', 'rmsd', 'eigenvector', 'inertia',
  'inertiaZ', 'orientation', 'orientationAngle', 'orientationProj', 'spinAngle', 'tilt', 'alpha', 'dihedralPC',
  'polarTheta', 'polarPhi', 'linearCombination', 'customColvar', 'neuralNetwork', 'mapTotal', 'aspathCV', 'azpathCV',
  'gspath', 'gzpath', 'gspathCV', 'gzpathCV', 'euler_phi', 'euler_psi', 'euler_theta', 'dipoleDirection',
  'alchLambda', 'alchFLambda', 'componentDistance', 'distanceEuler',
];
const COMPONENT_NAMES = 'distance, distanceVec, distanceZ, distanceXY, angle, dihedral, cartesian, gyration, coordNum, selfCoordNum, groupCoord';

/** One collective variable. */
export class Colvar {
  name: string;
  width = 1.0;
  readonly cvcs: Component[] = [];
  type: ValueType = 'scalar';
  x = new Float64Array(1);
  /** The bias forces summed (fb) and the force applied (f). */
  fb = new Float64Array(1);
  f = new Float64Array(1);
  homogeneous = true;
  /** f_cv_periodic. */
  periodic = false;
  /** A bias applies forces to this colvar (f_cv_apply_force). */
  applyForceFlag = false;
  outputValue = true;
  outputAppliedForce = false;
  /** The value last read from an input state file (x_restart), checked once (after_restart). */
  xRestart: Float64Array | null = null;
  /** colvar::period of a periodic colvar (the components' period), 0 otherwise. */
  period = 0.0;
  /**
   * Grid parameters (colvar::init_grid_parameters): the boundaries, default 0 and width ("Default to 1-wide grids"),
   * f_cv_lower_boundary / f_cv_upper_boundary, the hard boundaries and expandBoundaries.
   */
  lowerBoundary = 0.0;
  upperBoundary = 1.0;
  lowerBoundaryOn = false;
  upperBoundaryOn = false;
  hardLower = false;
  hardUpper = false;
  expandBoundaries = false;

  constructor(atoms: ColvarsAtoms, conf: string, index: number, existing: (name: string) => boolean, ctx: ComponentContext) {
    const c = new Conf(conf);
    const nm = c.get('name');
    this.name = nm !== undefined ? nm.trim() : `colvar${index}`;
    if (existing(this.name)) throw colvarsError(`Error: this colvar cannot have the same name, "${this.name}", as another colvar.`);
    const w = c.get('width');
    if (w !== undefined) {
      this.width = parseReal(w, 'width');
      if (!(this.width > 0.0)) throw colvarsError('Error: "width" must be positive.');
    }
    // the grid parameters are read here and applied once the components are known (init_grid_parameters)
    const lbText = c.get('lowerBoundary'), ubText = c.get('upperBoundary');
    const lbIn = lbText !== undefined ? parseReal(lbText, 'lowerBoundary') : undefined;
    const ubIn = ubText !== undefined ? parseReal(ubText, 'upperBoundary') : undefined;
    const hlText = c.get('hardLowerBoundary'), huText = c.get('hardUpperBoundary'), exText = c.get('expandBoundaries');
    const hlIn = hlText !== undefined ? parseBool(hlText, 'hardLowerBoundary') : undefined;
    const huIn = huText !== undefined ? parseBool(huText, 'hardUpperBoundary') : undefined;
    if (exText !== undefined) this.expandBoundaries = parseBool(exText, 'expandBoundaries');
    const ov = c.get('outputValue');
    if (ov !== undefined) this.outputValue = parseBool(ov, 'outputValue');
    const oa = c.get('outputAppliedForce');
    if (oa !== undefined) this.outputAppliedForce = parseBool(oa, 'outputAppliedForce');
    for (const k of ['outputVelocity', 'outputEnergy', 'outputTotalForce', 'extendedLagrangian', 'subtractAppliedForce', 'runAve', 'corrFunc', 'outputSystemForce', 'hardLowerBoundary_']) {
      const v = c.get(k);
      if (v !== undefined && parseBool(v, k)) throw colvarsError(`Error: the colvar option "${k}" is not supported by the browser engine.`);
    }
    for (const k of ['timeStepFactor']) {
      const v = c.get(k);
      if (v !== undefined && parseInt10(v, k) !== 1) throw colvarsError(`Error: the colvar option "${k}" other than 1 is not supported by the browser engine.`);
    }
    for (const k of ['scriptedFunction', 'customFunction', 'lowerWall', 'upperWall', 'lowerWallConstant', 'upperWallConstant']) {
      if (c.get(k) !== undefined) throw colvarsError(`Error: the colvar option "${k}" is not supported by the browser engine.`);
    }
    // colvar::init_components: every component keyword of the block, in the library's order of types
    for (const key of c.blockKeys()) {
      if (Object.hasOwn(COMPONENTS, key.toLowerCase())) continue;
      if (LIBRARY_COMPONENTS.some((n) => n.toLowerCase() === key.toLowerCase())) {
        throw colvarsError(`Error: the Colvars component "${key}" is not supported by the browser engine (supported: ${COMPONENT_NAMES}).`);
      }
    }
    for (const [lower, make] of Object.entries(COMPONENTS)) {
      for (const v of c.all(lower)) this.cvcs.push(make(atoms, new Conf(v), this.name, ctx));
    }
    c.check(`colvar "${this.name}"`);
    if (this.cvcs.length === 0) throw colvarsError(`Error: no valid components were provided for this collective variable.`);
    const t0 = this.cvcs[0].type;
    for (const cvc of this.cvcs) {
      if (cvc.type !== t0) throw colvarsError(`Error: you are combining components of different types ("${TYPE_DESC[t0]}" and "${TYPE_DESC[cvc.type]}") in colvar "${this.name}".`);
      if (t0 !== 'scalar' && cvc.exponent !== 1) throw colvarsError('Error: cannot use a non-unit exponent (componentExp) for a non-scalar component.');
    }
    this.type = t0;
    // f_cv_linear and f_cv_homogeneous
    let linear = true;
    for (const cvc of this.cvcs) if (cvc.exponent !== 1) linear = false;
    let homogeneous = linear;
    for (const cvc of this.cvcs) if (Math.abs(Math.abs(cvc.coeff) - 1.0) > 1.0e-10) homogeneous = false;
    this.homogeneous = homogeneous;
    // f_cv_periodic: a homogeneous colvar whose components are all periodic with one period
    if (homogeneous && this.cvcs[0].periodic) {
      const period = this.cvcs[0].period;
      this.periodic = this.cvcs.every((cvc) => cvc.periodic && cvc.period === period);
    }
    if (this.periodic) this.period = this.cvcs[0].period;
    const size = this.cvcs[0].x.length;
    if (this.type === 'vector') for (const cvc of this.cvcs) if (cvc.x.length !== size) throw colvarsError(`Error: the components of colvar "${this.name}" have different sizes.`);
    this.x = new Float64Array(size);
    this.initGridParameters(lbIn, ubIn, hlIn, huIn);
    this.fb = new Float64Array(size);
    this.f = new Float64Array(size);
  }

  /**
   * colvar::init_grid_parameters for a scalar colvar: a single component (f_cv_single_cvc: one homogeneous component
   * with coefficient 1 and exponent 1) gives its intrinsic boundaries, which are hard; a user boundary farther than
   * colvar_boundaries_tol (1e-10 widths) from the intrinsic one is not hard; hardLowerBoundary and hardUpperBoundary
   * set the features; then check_grid_parameters.
   */
  private initGridParameters(lbIn: number | undefined, ubIn: number | undefined, hlIn: boolean | undefined, huIn: boolean | undefined): void {
    this.lowerBoundary = 0.0;
    this.upperBoundary = this.width;
    if (this.type !== 'scalar') return;
    const single = this.cvcs.length === 1 && this.homogeneous && Math.abs(this.cvcs[0].coeff - 1.0) < 1.0e-10 && this.cvcs[0].exponent === 1;
    let cvcLb = 0.0, cvcUb = 0.0;
    if (single) {
      const cvc = this.cvcs[0];
      if (cvc.lowerBoundary !== null) { this.lowerBoundaryOn = true; this.hardLower = true; this.lowerBoundary = cvcLb = cvc.lowerBoundary; }
      if (cvc.upperBoundary !== null) { this.upperBoundaryOn = true; this.hardUpper = true; this.upperBoundary = cvcUb = cvc.upperBoundary; }
    }
    const tol = 1.0e-10;
    if (lbIn !== undefined) {
      this.lowerBoundary = lbIn;
      this.lowerBoundaryOn = true;
      if (single && this.hardLower && Math.sqrt(this.dist2(Float64Array.of(lbIn), Float64Array.of(cvcLb))) / this.width > tol) this.hardLower = false;
    }
    if (ubIn !== undefined) {
      this.upperBoundary = ubIn;
      this.upperBoundaryOn = true;
      if (single && this.hardUpper && Math.sqrt(this.dist2(Float64Array.of(ubIn), Float64Array.of(cvcUb))) / this.width > tol) this.hardUpper = false;
    }
    if (hlIn !== undefined) this.hardLower = hlIn;
    if (huIn !== undefined) this.hardUpper = huIn;
    if (this.hardLower && !this.lowerBoundaryOn) throw colvarsError(`Error: feature "hard_lower_boundary" of colvar "${this.name}" requires "lower_boundary" (lowerBoundary).`);
    if (this.hardUpper && !this.upperBoundaryOn) throw colvarsError(`Error: feature "hard_upper_boundary" of colvar "${this.name}" requires "upper_boundary" (upperBoundary).`);
    // check_grid_parameters
    if (this.lowerBoundaryOn && this.upperBoundaryOn && this.lowerBoundary >= this.upperBoundary) {
      throw colvarsError(`Error: the upper boundary, ${this.upperBoundary}, is not higher than the lower boundary, ${this.lowerBoundary}.`);
    }
    if (this.expandBoundaries && this.periodicBoundaries()) throw colvarsError('Error: trying to expand boundaries that already cover a whole period of a periodic colvar.');
    if (this.expandBoundaries && this.hardLower && this.hardUpper) {
      throw colvarsError('Error: inconsistent configuration (trying to expand boundaries, but both hardLowerBoundary and hardUpperBoundary are enabled).');
    }
  }

  /** colvar::periodic_boundaries(lb, ub): a periodic colvar whose interval spans one period (within 1e-10 widths). */
  periodicBoundariesOf(lb: number, ub: number): boolean {
    return this.period > 0.0 && Math.sqrt(this.dist2(Float64Array.of(lb), Float64Array.of(ub))) / this.width < 1.0e-10;
  }

  /** colvar::periodic_boundaries(): false unless both boundaries are defined. */
  periodicBoundaries(): boolean {
    if (!this.lowerBoundaryOn || !this.upperBoundaryOn) return false;
    return this.periodicBoundariesOf(this.lowerBoundary, this.upperBoundary);
  }

  /** colvar::calc: component values and gradients, then collect_cvc_values. */
  calc(bc: Boundaries, simulationRunning: boolean, stepRelative: number): void {
    for (const cvc of this.cvcs) {
      cvc.readData(bc);
      cvc.calcValue(stepRelative);
      cvc.calcGradients();
    }
    this.x.fill(0);
    if (this.type === 'scalar') {
      for (const cvc of this.cvcs) this.x[0] += cvc.coeff * (cvc.exponent !== 1 ? integerPower(cvc.x[0], cvc.exponent) : cvc.x[0]);
    } else {
      for (const cvc of this.cvcs) for (let k = 0; k < this.x.length; k++) this.x[k] += cvc.coeff * cvc.x[k];
    }
    if (this.xRestart) {
      const xr = this.xRestart;
      this.xRestart = null;
      if (simulationRunning) {
        const jump2 = this.dist2(this.x, xr) / (this.width * this.width);
        if (jump2 > 0.25) {
          throw colvarsError(`Error: the calculated value of colvar "${this.name}":\n${this.valueText(this.x)}\n differs greatly from the value last read from the state file:\n${this.valueText(xr)}\nPossible causes are changes in configuration, wrong state file, or how PBC wrapping is handled.`);
        }
      }
    }
  }

  private valueText(v: Float64Array): string {
    return this.type === 'scalar' ? String(v[0]) : `( ${[...v].join(' , ')} )`;
  }

  /** colvar::dist2. */
  dist2(a: Float64Array, b: Float64Array): number {
    if (this.homogeneous) return this.cvcs[0].dist2(a, b);
    let r = 0.0;
    if (this.type === 'scalar') return (a[0] - b[0]) * (a[0] - b[0]);
    if (this.type === '3vector') { const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2]; return dx * dx + dy * dy + dz * dz; }
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; r += d * d; }
    return r;
  }

  /** colvar::dist2_lgrad. */
  dist2Lgrad(a: Float64Array, b: Float64Array): Float64Array {
    if (this.homogeneous) return this.cvcs[0].dist2Lgrad(a, b);
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = 2.0 * (a[i] - b[i]);
    return out;
  }

  /** colvar::update_forces_energy: f = fb (no extended Lagrangian, no walls). */
  updateForces(): void {
    this.f.fill(0);
    for (let k = 0; k < this.f.length; k++) this.f[k] += this.fb[k];
  }

  /** colvar::communicate_forces. */
  communicateForces(): void {
    if (this.type === 'scalar') {
      for (const cvc of this.cvcs) {
        cvc.applyForce(Float64Array.of(this.f[0] * cvc.coeff * cvc.exponent * integerPower(cvc.x[0], cvc.exponent - 1)));
      }
    } else {
      for (const cvc of this.cvcs) {
        const g = new Float64Array(this.f.length);
        for (let k = 0; k < g.length; k++) g[k] = this.f[k] * cvc.coeff;
        cvc.applyForce(g);
      }
    }
  }

  /** colvarvalue::output_width(cv_width). */
  outputWidth(): number {
    if (this.type === 'scalar') return CV_WIDTH;
    if (this.type === '3vector') return 3 * CV_WIDTH + 10;
    return CV_WIDTH * this.x.length + 3 * (this.x.length - 1) + 4;
  }
}

/** The stream form of a colvar value with a field width: a number, or "( a , b , ... )". */
const valueStream = (v: Float64Array, type: ValueType, fmt: (x: number) => string): string =>
  type === 'scalar' ? fmt(v[0]) : `( ${[...v].map(fmt).join(' , ')} )`;

/**
 * colvarbias: names, colvars, output, the state block. Each bias type adds its update, its state data and its output
 * files.
 */
abstract class Bias {
  abstract readonly biasType: string;
  abstract readonly stateKeyword: string;
  name = '';
  readonly colvars: Colvar[] = [];
  outputEnergy = false;
  energy = 0.0;
  forces: Float64Array[] = [];
  /** outputFreq: default the module's colvarsRestartFrequency when the bias is defined (colvarbias constructor). */
  outputFreq = 0;
  /** f_cvb_step_zero_data (stepZeroData). */
  stepZeroData = false;
  protected module!: ColvarsModule;

  /** colvarbias::init: the keywords every bias reads. */
  protected initBias(c: Conf, rank: number, module: ColvarsModule, unsupported: string[]): void {
    this.module = module;
    this.outputFreq = module.restartFreq;
    // colvarbias::init: name = bias_type + rank (bias_type is the keyword in lower case)
    this.name = `${this.biasType.toLowerCase()}${rank}`;
    const nm = c.get('name');
    if (nm !== undefined) this.name = nm.trim();
    if (module.biases.some((b) => b.name === this.name)) throw colvarsError(`Error: this bias cannot have the same name, "${this.name}", as another bias.`);
    const names = c.get('colvars');
    if (names !== undefined) {
      for (const n of names.trim().split(/\s+/).filter((t) => t)) {
        const cv = module.colvars.find((x) => x.name === n);
        if (!cv) throw colvarsError(`Error: cannot find a colvar named "${n}".`);
        this.colvars.push(cv);
        cv.applyForceFlag = true;
      }
    }
    if (this.colvars.length === 0) throw colvarsError('Error: no collective variables specified.');
    const sz = c.get('stepZeroData');
    if (sz !== undefined) this.stepZeroData = parseBool(sz, 'stepZeroData');
    const oe = c.get('outputEnergy');
    if (oe !== undefined) this.outputEnergy = parseBool(oe, 'outputEnergy');
    const of = c.get('outputFreq');
    if (of !== undefined) this.outputFreq = parseInt10(of, 'outputFreq');
    for (const k of ['writeTISamples', 'writeTIPMF', 'scaledBiasingForce', 'timeStepFactor', ...unsupported]) {
      const v = c.get(k);
      if (v !== undefined && !/^\s*(off|no|false)\s*$/i.test(v)) throw colvarsError(`Error: the ${this.biasType} option "${k}" is not supported by the browser engine.`);
    }
    this.forces = this.colvars.map((cv) => new Float64Array(cv.x.length));
  }

  /** colvarbias::can_accumulate_data. */
  protected canAccumulateData(): boolean {
    return (this.module.stepRelative() > 0 && !this.module.continuing) || this.stepZeroData;
  }

  /** The update of the bias: the energy and the force on each colvar. */
  abstract update(): void;

  /** colvarbias::communicate_forces: add to each colvar's bias force. */
  communicateForces(): void {
    this.colvars.forEach((cv, i) => { for (let k = 0; k < cv.fb.length; k++) cv.fb[k] += 1.0 * this.forces[i][k]; });
  }

  trajLabel(): string {
    return ' ' + (this.outputEnergy ? ' E_' + wrapString(this.name, EN_WIDTH - 2) : '');
  }

  trajLine(): string {
    return ' ' + (this.outputEnergy ? ' ' + fmtSci(this.energy) : '');
  }

  /** colvarbias::write_state: the configuration block (get_state_params), then the bias's own data. */
  stateText(step: number): string {
    return `${this.stateKeyword} {\n  configuration {\n${this.stateParams(step)}  }\n${this.stateData()}}\n\n`;
  }

  /** get_state_params. */
  protected stateParams(step: number): string {
    return `    step ${step}\n    name ${this.name}\n`;
  }

  /** write_state_data. */
  protected stateData(): string { return ''; }

  /** setup_output (each run setup). */
  setupOutput(): void {}

  /** write_output_files. */
  writeOutputFiles(): void {}
}

/** colvarbias_restraint and colvarbias_restraint_k. */
abstract class Restraint extends Bias {
  readonly stateKeyword = 'restraint';
  forceK = 1.0;

  /** colvarbias::init, colvarbias_restraint::init and colvarbias_restraint_k::init. */
  protected initBase(c: Conf, rank: number, module: ColvarsModule, unsupported: string[]): void {
    this.initBias(c, rank, module, ['targetForceConstant', 'targetNumSteps', 'targetNumStages', 'lambdaSchedule', 'targetEquilSteps', 'decoupling', 'outputAccumulatedWork', ...unsupported]);
    // colvarbias_restraint_k::init: forceConstant, default 1
    const k = c.get('forceConstant');
    if (k !== undefined) this.forceK = parseReal(k, 'forceConstant');
    if (this.forceK < 0.0) throw colvarsError('Error: undefined or invalid force constant.');
  }
}

/** One value per colvar of a bias (centers, walls): scalars or parenthesized vectors. */
const parseColvarValues = (text: string, key: string, colvars: Colvar[]): Float64Array[] => {
  const vals = splitValues(text);
  if (vals.length !== colvars.length) throw colvarsError(`Error: the number of values of "${key}" does not match that of collective variables.`);
  return vals.map((v, i) => {
    const cv = colvars[i];
    if (cv.type === 'scalar') {
      if (Array.isArray(v)) throw colvarsError(`Error: the value of "${key}" for colvar "${cv.name}" must be a scalar number.`);
      return Float64Array.of(v);
    }
    if (!Array.isArray(v) || v.length !== cv.x.length) throw colvarsError(`Error: the value of "${key}" for colvar "${cv.name}" must be a vector of ${cv.x.length} components.`);
    return Float64Array.from(v);
  });
};

/** colvarbias_restraint_harmonic. */
class Harmonic extends Restraint {
  readonly biasType = 'harmonic';
  centers: Float64Array[] = [];

  constructor(conf: string, rank: number, module: ColvarsModule) {
    super();
    const c = new Conf(conf);
    this.initBase(c, rank, module, ['targetCenters', 'centersFile', 'bypassExtendedLagrangian', 'outputCenters']);
    const centers = c.get('centers');
    if (centers === undefined) throw colvarsError('Error: must define the initial centers of the restraints.');
    if (splitValues(centers).length !== this.colvars.length) throw colvarsError('Error: number of centers does not match that of collective variables.');
    this.centers = parseColvarValues(centers, 'centers', this.colvars);
    c.check(`harmonic "${this.name}"`);
  }

  /** restraint_potential and restraint_force of colvarbias_restraint_harmonic. */
  update(): void {
    this.energy = 0.0;
    this.colvars.forEach((cv, i) => {
      const w = cv.width;
      this.energy += 0.5 * this.forceK / (w * w) * cv.dist2(cv.x, this.centers[i]);
      const g = cv.dist2Lgrad(cv.x, this.centers[i]);
      const c = -0.5 * this.forceK / (w * w);
      for (let k = 0; k < g.length; k++) this.forces[i][k] = c * g[k];
    });
  }
}

/** colvarbias_restraint_harmonic_walls: half-harmonic walls below lowerWalls and above upperWalls (scalar colvars). */
class HarmonicWalls extends Restraint {
  readonly biasType = 'harmonicWalls';
  private lowerWalls: Float64Array[] = [];
  private upperWalls: Float64Array[] = [];
  private lowerK = -1.0;
  private upperK = -1.0;

  constructor(conf: string, rank: number, module: ColvarsModule) {
    super();
    const c = new Conf(conf);
    this.initBase(c, rank, module, ['targetLowerWalls', 'targetUpperWalls']);
    // bypassExtendedLagrangian defaults to on here; without extended Lagrangian colvars it changes nothing
    const bel = c.get('bypassExtendedLagrangian');
    if (bel !== undefined) parseBool(bel, 'bypassExtendedLagrangian');
    for (const cv of this.colvars) {
      if (cv.type !== 'scalar') throw colvarsError(`Error: the harmonicWalls bias "${this.name}" requires scalar colvars ("${cv.name}" is not).`);
    }
    const lw = c.get('lowerWalls');
    const uw = c.get('upperWalls');
    if (lw !== undefined) this.lowerWalls = parseColvarValues(lw, 'lowerWalls', this.colvars);
    if (uw !== undefined) this.upperWalls = parseColvarValues(uw, 'upperWalls', this.colvars);
    if (!this.lowerWalls.length && !this.upperWalls.length) throw colvarsError('Error: no walls provided.');
    const lk = c.get('lowerWallConstant');
    const uk = c.get('upperWallConstant');
    if (this.lowerWalls.length) this.lowerK = lk !== undefined ? parseReal(lk, 'lowerWallConstant') : this.forceK;
    else if (lk !== undefined) parseReal(lk, 'lowerWallConstant');
    if (this.upperWalls.length) this.upperK = uk !== undefined ? parseReal(uk, 'upperWallConstant') : this.forceK;
    else if (uk !== undefined) parseReal(uk, 'upperWallConstant');
    if (!this.lowerWalls.length || !this.upperWalls.length) {
      for (const cv of this.colvars) if (cv.periodic) throw colvarsError('Error: at least one variable is periodic, both walls must be provided.');
    }
    if (this.lowerWalls.length && this.upperWalls.length) {
      this.colvars.forEach((cv, i) => {
        if (this.lowerWalls[i][0] >= this.upperWalls[i][0]) {
          throw colvarsError(`Error: one upper wall, ${this.upperWalls[i][0]}, is not higher than the lower wall, ${this.lowerWalls[i][0]}.`);
        }
        if (cv.dist2(this.lowerWalls[i], this.upperWalls[i]) < 1.0e-12) {
          throw colvarsError(`Error: lower wall and upper wall are equal in the domain of the variable "${cv.name}".`);
        }
      });
      if (this.lowerK * this.upperK === 0.0) throw colvarsError('Error: lowerWallConstant and upperWallConstant, when defined, must both be positive.');
      // the two constants relative to their geometric mean, which becomes the force constant
      this.forceK = Math.sqrt(this.lowerK * this.upperK);
      this.lowerK /= this.forceK;
      this.upperK /= this.forceK;
    } else {
      if (this.lowerWalls.length) { this.forceK = this.lowerK; this.lowerK = 1.0; }
      if (this.upperWalls.length) { this.forceK = this.upperK; this.upperK = 1.0; }
    }
    c.check(`harmonicWalls "${this.name}"`);
  }

  /** colvar_distance: how far the colvar is beyond the wall it touches (0 between the walls). */
  private distance(i: number): number {
    const cv = this.colvars[i];
    const x = cv.x;
    if (cv.periodic) {
      // both walls may apply to a periodic colvar: the closer one counts
      if (cv.dist2(x, this.lowerWalls[i]) < cv.dist2(x, this.upperWalls[i])) {
        const g = cv.dist2Lgrad(x, this.lowerWalls[i])[0];
        if (g < 0.0) return 0.5 * g;
      } else {
        const g = cv.dist2Lgrad(x, this.upperWalls[i])[0];
        if (g > 0.0) return 0.5 * g;
      }
      return 0.0;
    }
    if (this.lowerWalls.length) {
      const g = cv.dist2Lgrad(x, this.lowerWalls[i])[0];
      if (g < 0.0) return 0.5 * g;
    }
    if (this.upperWalls.length) {
      const g = cv.dist2Lgrad(x, this.upperWalls[i])[0];
      if (g > 0.0) return 0.5 * g;
    }
    return 0.0;
  }

  /** restraint_potential and restraint_force of colvarbias_restraint_harmonic_walls. */
  update(): void {
    this.energy = 0.0;
    this.colvars.forEach((cv, i) => {
      const w = cv.width;
      const dist = this.distance(i);
      const scale = dist > 0.0 ? this.upperK : this.lowerK;
      this.energy += 0.5 * this.forceK * scale / (w * w) * dist * dist;
      this.forces[i][0] = -this.forceK * scale / (w * w) * dist;
    });
  }
}

/** The state stream's std::scientific precision 14 with no field width (write_state: "os.precision(cv_prec)"). */
const sci14 = compileCFormat('%.14e', 'float');
/** A std::ostringstream's default floating-point format (precision 6): %g. */
const fmtDefault = compileCFormat('%g', 'float');
/** std::setw(cv_width or en_width) with precision 14: %21.14e. */
const sci21 = fmtSci;

/**
 * colvar_grid<cvm::real> with colvar_grid_scalar (mult 1) and colvar_grid_gradient (mult = number of colvars):
 * per colvar the lower and upper boundaries, the width, the number of points nx and periodicity; the data in row-major
 * order (the last index the fastest), nxc the strides (colvar_grid::setup, which counts the multiplicity in).
 */
class ColvarGrid {
  nd: number;
  mult: number;
  nx: number[] = [];
  nxc: number[] = [];
  lower: number[];
  upper: number[];
  widths: number[];
  periodic: boolean[] = [];
  hardLower: boolean[];
  hardUpper: boolean[];
  data = new Float64Array(0);

  /** colvar_grid::init_from_colvars (no extra bin, no template) then init_from_boundaries and setup. */
  constructor(readonly cv: Colvar[], mult: number) {
    this.nd = cv.length;
    this.mult = mult;
    for (const c of cv) {
      if (c.type !== 'scalar') throw colvarsError('Error: Colvar grids can only be automatically constructed for scalar variables.  ABF and histogram can not be used; metadynamics can be used with useGrids disabled.');
      if (c.width <= 0.0) throw colvarsError('Error: Tried to initialize a grid on a variable with negative or zero width.');
    }
    this.widths = cv.map((c) => c.width);
    this.hardLower = cv.map((c) => c.hardLower);
    this.hardUpper = cv.map((c) => c.hardUpper);
    this.lower = cv.map((c) => c.lowerBoundary);
    this.upper = cv.map((c) => c.upperBoundary);
    this.periodic = cv.map((c, i) => c.periodicBoundariesOf(this.lower[i], this.upper[i]));
    this.initFromBoundaries();
    this.setup(this.nx, this.mult);
  }

  /** A copy with the same parameters and data (the copy constructor). */
  clone(): ColvarGrid {
    const g = Object.create(ColvarGrid.prototype) as ColvarGrid;
    Object.assign(g, this, { nx: this.nx.slice(), nxc: this.nxc.slice(), lower: this.lower.slice(), upper: this.upper.slice(), widths: this.widths.slice(), periodic: this.periodic.slice(), data: this.data.slice() });
    return g;
  }

  /** init_from_boundaries: nx = (ub - lb) / width rounded; an incommensurate interval moves the upper boundary. */
  initFromBoundaries(): void {
    this.nx = [];
    for (let i = 0; i < this.lower.length; i++) {
      this.periodic[i] = this.cv[i].periodicBoundariesOf(this.lower[i], this.upper[i]);
      const nbins = (this.upper[i] - this.lower[i]) / this.widths[i];
      const nbinsRound = Math.trunc(nbins + 0.5);
      if (Math.abs(nbins - nbinsRound) > 1.0e-10) this.upper[i] = this.lower[i] + nbinsRound * this.widths[i];
      this.nx.push(nbinsRound);
    }
  }

  /** colvar_grid::setup: the strides and zeroed data. */
  setup(nx: number[], mult: number): void {
    this.mult = mult;
    this.nx = nx.slice();
    this.nxc = new Array(this.nd).fill(0);
    let nt = mult;
    for (let i = this.nd - 1; i >= 0; i--) {
      if (this.nx[i] <= 0) throw colvarsError(`Error: providing an invalid number of grid points, ${this.nx[i]}.`);
      this.nxc[i] = nt;
      nt *= this.nx[i];
    }
    this.data = new Float64Array(nt);
  }

  address(ix: number[]): number {
    let a = 0;
    for (let i = 0; i < this.nd; i++) a += ix[i] * this.nxc[i];
    return a;
  }

  indexOk(ix: number[]): boolean {
    for (let i = 0; i < this.nd; i++) if (ix[i] < 0 || ix[i] >= this.nx[i]) return false;
    return true;
  }

  /** incr: the last index runs fastest; past the end the first index becomes nx[0]. */
  incr(ix: number[]): void {
    for (let i = ix.length - 1; i >= 0; i--) {
      ix[i]++;
      if (ix[i] >= this.nx[i]) {
        if (i > 0) { ix[i] = 0; continue; }
        ix[0] = this.nx[0];
        return;
      }
      return;
    }
  }

  /** value_to_bin_scalar: floor((x - lb) / width). */
  valueToBin(x: number, i: number): number {
    return Math.floor((x - this.lower[i]) / this.widths[i]);
  }

  /** bin_to_value_scalar: the centre of the bin. */
  binToValue(ib: number, i: number): number {
    return this.lower[i] + this.widths[i] * (0.5 + ib);
  }

  /** get_colvars_index(values). */
  index(values: readonly number[]): number[] {
    return values.map((v, i) => this.valueToBin(v, i));
  }

  /** bin_distance_from_boundaries (skip_hard_boundaries): the smallest distance in bins, negative off the grid. */
  binDistanceFromBoundaries(values: readonly number[], skipHard: boolean): number {
    let minimum = 1.0e16;
    for (let i = 0; i < this.nd; i++) {
      if (this.periodic[i]) continue;
      const cv = this.cv[i];
      let dl = Math.sqrt(cv.dist2(Float64Array.of(values[i]), Float64Array.of(this.lower[i]))) / this.widths[i];
      let du = Math.sqrt(cv.dist2(Float64Array.of(values[i]), Float64Array.of(this.upper[i]))) / this.widths[i];
      if (values[i] < this.lower[i]) dl *= -1.0;
      if (values[i] > this.upper[i]) du *= -1.0;
      if ((!skipHard || !this.hardLower[i]) && dl < minimum) minimum = dl;
      if ((!skipHard || !this.hardUpper[i]) && du < minimum) minimum = du;
    }
    return minimum;
  }

  /** map_grid: the values of the bins of `other` that hold this grid's bin centres. */
  mapGrid(other: ColvarGrid): void {
    const ix = new Array(this.nd).fill(0);
    const oix = new Array(this.nd).fill(0);
    for (; this.indexOk(ix); this.incr(ix)) {
      for (let i = 0; i < this.nd; i++) oix[i] = Math.floor((this.binToValue(ix[i], i) - other.lower[i]) / other.widths[i]);
      if (!other.indexOk(oix)) continue;
      const a = this.address(ix), oa = other.address(oix);
      for (let im = 0; im < this.mult; im++) this.data[a + im] = other.data[oa + im];
    }
  }

  /** get_state_params: its own std::ostringstream, so the default format (%g). */
  stateParams(): string {
    let s = `  n_colvars ${this.nd}\n`;
    s += '  lower_boundaries ' + this.lower.map((v) => ' ' + fmtDefault(v)).join('') + '\n';
    s += '  upper_boundaries ' + this.upper.map((v) => ' ' + fmtDefault(v)).join('') + '\n';
    s += '  widths ' + this.widths.map((v) => ' ' + fmtDefault(v)).join('') + '\n';
    s += '  sizes ' + this.nx.map((v) => ' ' + v).join('') + '\n';
    return s;
  }

  /** write_restart: the parameters block, then write_raw (three values per line). */
  restartText(): string {
    let s = `grid_parameters {\n${this.stateParams()}}\n`;
    let count = 0;
    for (let k = 0; k < this.data.length; k++) {
      s += ' ' + sci14(this.data[k]);
      if (++count % 3 === 0) s += '\n';
    }
    if (count % 3 !== 0) s += '\n';
    return s;
  }

  /** write_multicol: the header (dimensions, then per colvar lower boundary, width, points, periodic) and the points. */
  multicolText(): string {
    let s = `# ${this.nd}\n`;
    for (let i = 0; i < this.nd; i++) s += `# ${sci21(this.lower[i])} ${sci21(this.widths[i])} ${padLeft(String(this.nx[i]), 10)}  ${this.periodic[i] ? 1 : 0}\n`;
    const ix = new Array(this.nd).fill(0);
    for (; this.indexOk(ix); this.incr(ix)) {
      if (ix[this.nd - 1] === 0) s += '\n';
      for (let i = 0; i < this.nd; i++) s += ' ' + sci21(this.binToValue(ix[i], i));
      s += ' ';
      const a = this.address(ix);
      for (let im = 0; im < this.mult; im++) s += ' ' + sci21(this.data[a + im]);
      s += '\n';
    }
    return s;
  }
}

/** colvarbias_meta::hill: step, weight W (sW = 1), centres, sigmas and the last Gaussian value. */
interface Hill {
  it: number;
  W: number;
  centers: Float64Array[];
  sigmas: number[];
  value: number;
}

const copyHill = (h: Hill): Hill => ({ it: h.it, W: h.W, centers: h.centers.map((c) => c.slice()), sigmas: h.sigmas.slice(), value: 0.0 });

/**
 * colvarbias_meta (one replica): Gaussian hills of weight hillWeight every newHillFrequency steps, with half-widths
 * sigma = width * hillWidth / 2 (or gaussianSigmas), projected every gridsUpdateFrequency steps onto an energy grid
 * and a gradient grid (useGrids, default on); hills near or beyond the grid edges stay off-grid and are summed
 * analytically. Well-tempered metadynamics scales each new hill by exp(-V / (k biasTemperature)). Output: the PMF
 * (writeFreeEnergyFile, default on with grids) and the hills trajectory (writeHillsTrajectory).
 */
class Metadynamics extends Bias {
  readonly biasType = 'metadynamics';
  readonly stateKeyword = 'metadynamics';
  private hillWeight = 0.0;
  private hillWidth = 0.0;
  private newHillFreq = 1000;
  private sigmas: number[] = [];
  private useGrids = true;
  private gridsFreq = 0;
  private expandGrids = false;
  private dumpFes = true;
  private keepHills = false;
  private hillsTraj = false;
  private wellTempered = false;
  private biasTemperature = -1.0;
  private hills: Hill[] = [];
  /** new_hills_begin: the first hill not yet projected onto the grids. */
  private newHillsBegin = 0;
  private hillsOffGrid: Hill[] = [];
  private hillsEnergy: ColvarGrid | null = null;
  private hillsGradients: ColvarGrid | null = null;
  private hillsTrajBuf = '';
  private outputPrefix = '';
  /** The colvar values cached at update (colvar_values). */
  private values: Float64Array[] = [];

  constructor(conf: string, rank: number, module: ColvarsModule) {
    super();
    const c = new Conf(conf);
    this.initBias(c, rank, module, ['multipleReplicas', 'ebMeta', 'rebinGrids', 'keepFreeEnergyFiles', 'writePartialFreeEnergyFile', 'dumpPartialFreeEnergyFile']);
    for (const k of ['replicaID', 'replicasRegistry', 'replicaUpdateFrequency', 'targetDistFile', 'ebMetaEquilSteps', 'grid']) {
      if (c.get(k) !== undefined) throw colvarsError(`Error: the metadynamics option "${k}" is not supported by the browser engine.`);
    }
    const hw = c.get('hillWeight');
    if (hw !== undefined) this.hillWeight = parseReal(hw, 'hillWeight');
    if (!(this.hillWeight > 0.0)) throw colvarsError('Error: hillWeight must be provided, and a positive number.');
    const nf = c.get('newHillFrequency');
    if (nf !== undefined) this.newHillFreq = parseInt10(nf, 'newHillFrequency');
    const gs = c.get('gaussianSigmas');
    if (gs !== undefined) this.sigmas = splitValues(gs).map((v) => { if (Array.isArray(v)) throw colvarsError('Error: gaussianSigmas must be numbers.'); return v; });
    const hwid = c.get('hillWidth');
    if (hwid !== undefined) this.hillWidth = parseReal(hwid, 'hillWidth');
    if (this.sigmas.length > 0 && this.hillWidth > 0.0) throw colvarsError('Error: hillWidth and gaussianSigmas are mutually exclusive.');
    if (this.hillWidth > 0.0) {
      this.sigmas = this.colvars.map((cv) => cv.width * this.hillWidth / 2.0);
      module.log('Half-widths of the Gaussian hills (sigma\'s) for each colvar:');
      this.colvars.forEach((cv, i) => module.log(`  ${cv.name}: ${this.sigmas[i]}`));
    }
    if (this.sigmas.length === 0) throw colvarsError('either hillWidth or gaussianSigmas must be set to positive values.');
    const ug = c.get('useGrids');
    if (ug !== undefined) this.useGrids = parseBool(ug, 'useGrids');
    if (this.useGrids && this.sigmas.length === this.colvars.length) {
      if (this.gridsFreq === 0) this.gridsFreq = this.newHillFreq;
      this.colvars.forEach((cv, i) => {
        if (2.0 * this.sigmas[i] < cv.width) module.log(`Warning: gaussianSigmas is too narrow for the grid spacing along ${cv.name}.`);
      });
      const gf = c.get('gridsUpdateFrequency');
      if (gf !== undefined) this.gridsFreq = parseInt10(gf, 'gridsUpdateFrequency');
      for (const cv of this.colvars) {
        if (cv.expandBoundaries) {
          this.expandGrids = true;
          module.log(`Metadynamics bias "${this.name}": Will expand grids when the colvar "${cv.name}" approaches its boundaries.`);
        }
      }
      const wf = c.get('writeFreeEnergyFile');
      if (wf !== undefined) this.dumpFes = parseBool(wf, 'writeFreeEnergyFile');
      const kh = c.get('keepHills');
      if (kh !== undefined) this.keepHills = parseBool(kh, 'keepHills');
      this.hillsEnergy = new ColvarGrid(this.colvars, 1);
      this.hillsGradients = new ColvarGrid(this.colvars, this.colvars.length);
    } else {
      this.useGrids = false;
      this.dumpFes = false;
    }
    const ht = c.get('writeHillsTrajectory');
    if (ht !== undefined) this.hillsTraj = parseBool(ht, 'writeHillsTrajectory');
    // init_well_tempered_params
    const wt = c.get('wellTempered');
    if (wt !== undefined) this.wellTempered = parseBool(wt, 'wellTempered');
    const bt = c.get('biasTemperature');
    if (bt !== undefined) this.biasTemperature = parseReal(bt, 'biasTemperature');
    if (this.biasTemperature === -1.0 && this.wellTempered) throw colvarsError('Error: biasTemperature must be set to a positive value.');
    if (this.wellTempered) {
      module.log('Well-tempered metadynamics is used.');
      module.log(`The bias temperature is ${this.biasTemperature}.`);
    }
    c.check(`metadynamics "${this.name}"`);
    this.values = this.colvars.map((cv) => new Float64Array(cv.x.length));
  }

  /** f_cvb_calc_pmf (every metadynamics bias). */
  readonly calcPmf = true;

  /** add_hill: append; with grids, a hill within 3 floor(hillWidth) + 1 bins of a soft boundary is also kept off-grid. */
  private addHill(h: Hill): void {
    this.hills.push(h);
    if (this.useGrids) {
      const minDist = this.hillsEnergy!.binDistanceFromBoundaries(h.centers.map((c) => c[0]), true);
      if (minDist < 3.0 * Math.floor(this.hillWidth) + 1.0) this.hillsOffGrid.push(copyHill(h));
    }
    if (this.hillsTraj) this.hillsTrajBuf += this.hillTraj(h);
  }

  /** colvarbias_meta::update. */
  update(): void {
    // colvarbias::update: cache the colvar values, then calc_energy and calc_forces
    this.colvars.forEach((cv, i) => this.values[i].set(cv.x));
    this.calcEnergy(null);
    this.calcForces(null);
    this.updateGridParams();
    this.updateBias();
    if (this.useGrids) this.updateGridData();
    this.calcEnergy(null);
    this.calcForces(null);
  }

  /** update_grid_params: expand the grids by whole bins when a colvar with expandBoundaries nears a soft boundary. */
  private updateGridParams(): void {
    if (!this.useGrids || !this.expandGrids) return;
    const he = this.hillsEnergy!;
    const currBin = he.index(this.values.map((v) => v[0]));
    let changed = false;
    const minBuffer = 3 * Math.floor(this.hillWidth) + 1;
    const newSizes = he.nx.slice();
    const newLb = he.lower.slice();
    const newUb = he.upper.slice();
    this.colvars.forEach((cv, i) => {
      if (!cv.expandBoundaries) return;
      let chLb = false, chUb = false;
      if (!cv.hardLower && currBin[i] < minBuffer) {
        const extra = minBuffer - currBin[i];
        newLb[i] -= extra * cv.width;
        newSizes[i] += extra;
        currBin[i] += extra;
        chLb = true;
        this.module.log(`Metadynamics bias "${this.name}": new lower boundary for colvar "${cv.name}", at ${fmtDefault(newLb[i])}.`);
      }
      if (!cv.hardUpper && currBin[i] > newSizes[i] - minBuffer - 1) {
        const extra = currBin[i] - (newSizes[i] - 1) + minBuffer;
        newUb[i] += extra * cv.width;
        newSizes[i] += extra;
        chUb = true;
        this.module.log(`Metadynamics bias "${this.name}": new upper boundary for colvar "${cv.name}", at ${fmtDefault(newUb[i])}.`);
      }
      if (chLb || chUb) changed = true;
    });
    if (!changed) return;
    const ne = he.clone();
    ne.lower = newLb.slice(); ne.upper = newUb.slice();
    ne.setup(newSizes, 1);
    const ng = this.hillsGradients!.clone();
    ng.lower = newLb.slice(); ng.upper = newUb.slice();
    ng.setup(newSizes, this.colvars.length);
    ne.mapGrid(he);
    ng.mapGrid(this.hillsGradients!);
    this.hillsEnergy = ne;
    this.hillsGradients = ng;
  }

  /** update_bias: a new hill every newHillFrequency steps once data can be accumulated. */
  private updateBias(): void {
    const it = this.module.it;
    if (!(this.newHillFreq > 0 && it % this.newHillFreq === 0 && this.canAccumulateData())) return;
    let scale = 1.0;
    if (this.wellTempered) {
      let here = 0.0;
      if (this.useGrids) {
        const bin = this.hillsEnergy!.index(this.values.map((v) => v[0]));
        if (this.hillsEnergy!.indexOk(bin)) here += this.hillsEnergy!.data[this.hillsEnergy!.address(bin)];
        else here = this.calcHills(this.keepHills ? this.hills : this.hillsOffGrid, 0, here, null);
      } else {
        here = this.calcHills(this.hills, 0, here, null);
      }
      scale *= exp(-1.0 * here / (this.biasTemperature * this.module.boltzmann));
    }
    this.addHill({ it, W: this.hillWeight * scale, centers: this.values.map((v) => v.slice()), sigmas: this.sigmas.slice(), value: 0.0 });
  }

  /** update_grid_data: every gridsUpdateFrequency steps, project the new hills onto the grids. */
  private updateGridData(): void {
    if (this.module.it % this.gridsFreq === 0) this.projectNewHills();
  }

  private projectNewHills(): void {
    this.projectHills(this.newHillsBegin, this.hills.length);
    this.newHillsBegin = this.hills.length;
  }

  /** calc_hills: each hill's Gaussian at the values (0 past an exponent of 23), summed into energy. */
  private calcHills(list: Hill[], first: number, energy: number, values: Float64Array[] | null): number {
    const x = values ?? this.values;
    for (let k = first; k < list.length; k++) {
      const h = list[k];
      let sq = 0.0;
      for (let i = 0; i < this.colvars.length; i++) {
        const sigma = h.sigmas[i];
        sq += this.colvars[i].dist2(x[i], h.centers[i]) / (sigma * sigma);
      }
      // cvm::exp is the C library's exp (the glibc port)
      h.value = sq > 23.0 ? 0.0 : exp(-0.5 * sq);
      energy += h.W * 1.0 * h.value;
    }
    return energy;
  }

  /** calc_hills_force for colvar i: W v (0.5 / sigma^2) dist2_lgrad(x, centre), skipping hills of value 0. */
  private calcHillsForce(i: number, list: Hill[], first: number, forces: Float64Array[], values: Float64Array[] | null): void {
    const x = (values ?? this.values)[i];
    const cv = this.colvars[i];
    for (let k = first; k < list.length; k++) {
      const h = list[k];
      if (h.value === 0.0) continue;
      const sigma = h.sigmas[i];
      const g = cv.dist2Lgrad(x, h.centers[i]);
      const a = h.W * 1.0 * h.value * (0.5 / (sigma * sigma));
      for (let d = 0; d < g.length; d++) forces[i][d] += a * g[d];
    }
  }

  /** calc_energy: the grid value at the current bin (or the off-grid hills), then the hills not projected yet. */
  private calcEnergy(values: Float64Array[] | null): void {
    this.energy = 0.0;
    let ok = false;
    let bin: number[] = [];
    if (this.useGrids) {
      bin = this.hillsEnergy!.index((values ?? this.values).map((v) => v[0]));
      ok = this.hillsEnergy!.indexOk(bin);
    }
    if (ok) this.energy += this.hillsEnergy!.data[this.hillsEnergy!.address(bin)];
    else this.energy = this.calcHills(this.hillsOffGrid, 0, this.energy, values);
    this.energy = this.calcHills(this.hills, this.newHillsBegin, this.energy, values);
  }

  /** calc_forces: minus the grid gradient at the current bin (or the off-grid hills), then the new hills. */
  private calcForces(values: Float64Array[] | null): void {
    for (const f of this.forces) f.fill(0);
    let ok = false;
    let bin: number[] = [];
    if (this.useGrids) {
      bin = this.hillsEnergy!.index((values ?? this.values).map((v) => v[0]));
      ok = this.hillsEnergy!.indexOk(bin);
    }
    if (ok) {
      const g = this.hillsGradients!;
      const a = g.address(bin);
      for (let ic = 0; ic < this.colvars.length; ic++) this.forces[ic][0] += -1.0 * g.data[a + ic];
    } else {
      for (let ic = 0; ic < this.colvars.length; ic++) this.calcHillsForce(ic, this.hillsOffGrid, 0, this.forces, values);
    }
    for (let ic = 0; ic < this.colvars.length; ic++) this.calcHillsForce(ic, this.hills, this.newHillsBegin, this.forces, values);
  }

  /**
   * project_hills: at every grid point, the energy and the forces of the given hills are accumulated (acc_value; the
   * gradient grid sums minus the forces, acc_force); unless keepHills, all hills are then forgotten.
   */
  private projectHills(first: number, last: number): void {
    const he = this.hillsEnergy!, hg = this.hillsGradients!;
    const nv = this.colvars.length;
    const list = this.hills.slice(first, last);
    const vals = this.colvars.map(() => new Float64Array(1));
    const f = this.colvars.map(() => new Float64Array(1));
    const ix = new Array(he.nd).fill(0);
    const gx = new Array(hg.nd).fill(0);
    for (; he.indexOk(ix) && hg.indexOk(gx);) {
      for (let i = 0; i < nv; i++) vals[i][0] = he.binToValue(ix[i], i);
      const e = this.calcHills(list, 0, 0.0, vals);
      he.data[he.address(ix)] += e;
      const ga = hg.address(gx);
      for (let i = 0; i < nv; i++) {
        f[i][0] = 0.0;
        this.calcHillsForce(i, list, 0, f, vals);
      }
      for (let i = 0; i < nv; i++) hg.data[ga + i] -= f[i][0];
      he.incr(ix);
      hg.incr(gx);
    }
    if (!this.keepHills) {
      this.hills = [];
      this.newHillsBegin = 0;
    }
  }

  /** output_traj: one line of the hills trajectory. */
  private hillTraj(h: Hill): string {
    let s = padLeft(String(h.it), IT_WIDTH) + ' ';
    s += '  ';
    for (const c of h.centers) s += ' ' + valueStream(c, c.length === 1 ? 'scalar' : '3vector', sci21);
    s += '  ';
    for (const sg of h.sigmas) s += ' ' + sci21(sg);
    s += '  ' + sci21(h.W) + '\n';
    return s;
  }

  /** write_hill (formatted): the step, the weight, the centres and the widths (twice the sigmas). */
  private hillState(h: Hill): string {
    let s = `hill {\nstep ${padLeft(String(h.it), IT_WIDTH)}\nweight ${sci21(h.W)}\ncenters `;
    for (const c of h.centers) s += ' ' + valueStream(c, c.length === 1 ? 'scalar' : '3vector', sci21);
    s += '\nwidths ';
    for (const sg of h.sigmas) s += ' ' + sci21(2.0 * sg);
    return s + '\n}\n';
  }

  /** colvarbias_meta::get_state_params: "keepHills on" (unindented) when the hills are kept. */
  protected stateParams(step: number): string {
    return super.stateParams(step) + (this.keepHills ? 'keepHills on\n' : '');
  }

  /** write_state_data: with grids, project the pending hills first, then both grids and the off-grid hills. */
  protected stateData(): string {
    let s = '';
    if (this.useGrids) {
      this.projectNewHills();
      s += '\nhills_energy\n' + this.hillsEnergy!.restartText();
      s += '\nhills_energy_gradients\n' + this.hillsGradients!.restartText();
    }
    const list = !this.useGrids || this.keepHills ? this.hills : this.hillsOffGrid;
    for (const h of list) s += this.hillState(h);
    return s;
  }

  /** setup_output: the PMF prefix (with the bias name when several biases write one) and the hills trajectory file. */
  setupOutput(): void {
    this.outputPrefix = this.module.outputPrefixValue();
    if (this.module.biases.filter((b) => (b as Partial<Metadynamics>).calcPmf).length > 1) this.outputPrefix += '.' + this.name;
    if (this.hillsTraj) this.module.openOutput(this.hillsTrajName());
  }

  private hillsTrajName(): string {
    return `${this.module.outputPrefixValue()}.colvars.${this.name}.hills.traj`;
  }

  /** write_output_files: the PMF (write_pmf) and the buffered hills trajectory. */
  writeOutputFiles(): void {
    if (this.dumpFes) {
      // write_pmf: -(V - max V), times (T_bias + T) / T_bias when well-tempered
      const pmf = this.hillsEnergy!.clone();
      pmf.data.fill(0);
      for (let k = 0; k < pmf.data.length; k++) pmf.data[k] += this.hillsEnergy!.data[k];
      let max = pmf.data[0];
      for (let k = 1; k < pmf.data.length; k++) if (pmf.data[k] > max) max = pmf.data[k];
      for (let k = 0; k < pmf.data.length; k++) pmf.data[k] += -1.0 * max;
      for (let k = 0; k < pmf.data.length; k++) pmf.data[k] *= -1.0;
      if (this.wellTempered) {
        const s = (this.biasTemperature + this.module.targetTemperature) / this.biasTemperature;
        for (let k = 0; k < pmf.data.length; k++) pmf.data[k] *= s;
      }
      this.module.writeWholeFile(this.outputPrefix + '.pmf', pmf.multicolText());
    }
    if (this.hillsTraj) {
      const name = this.hillsTrajName();
      this.module.openOutput(name);
      if (this.hillsTrajBuf) this.module.appendOutput(name, this.hillsTrajBuf);
      this.hillsTrajBuf = '';
    }
  }
}

/** The bias keywords of the library, in its order of parsing (colvarmodule::parse_biases). */
const BIAS_TYPES = ['abf', 'abmd', 'ALB', 'harmonic', 'harmonicWalls', 'histogram', 'histogramRestraint', 'linear', 'metadynamics', 'reweightaMD', 'opes_metad'];

/** What the module needs from the engine: files and the log. */
export interface ColvarsHost {
  readFile(name: string): string | undefined;
  writeFile(name: string, text: string, append: boolean): void;
  /** Renames a file of the session, if it exists. */
  rename(from: string, to: string): void;
  exists(name: string): boolean;
  log(text: string): void;
}

/** colvarmodule with the colvarproxy_lammps state that drives it. */
export class ColvarsModule {
  readonly atoms = new ColvarsAtoms();
  readonly colvars: Colvar[] = [];
  readonly biases: Bias[] = [];
  private readonly configQueue: { cmd: 'config' | 'configfile'; text: string }[] = [];
  private biasCount = new Map<string, number>();
  trajFreq = 100;
  restartFreq = 0;
  it = 0;
  itRestart = 0;
  private firstTimestep = true;
  private previousStep = 0;
  /** proxy->simulation_running(): false before the first run's setup. */
  private running = false;
  private outputPrefix = '';
  private trajName = '';
  private trajWriteLabels = true;
  /** Output files this module opened (colvarproxy_io output streams). */
  private readonly open = new Set<string>();
  inputPrefix = '';
  /** force->angstrom of the unit style (proxy->angstrom_to_internal), set by the fix. */
  angstrom = 1.0;
  /** integration timestep in fs (dt * force->femtosecond). */
  dt = 1.0;
  readonly bc = new Boundaries();
  biasEnergy = 0.0;
  /** proxy->boltzmann(): force->boltz of the unit style. */
  boltzmann = 1.0;
  /** proxy->target_temperature(): the t_target of the tstat fix when fix colvars was defined, else 0. */
  targetTemperature = 0.0;
  /** proxy->simulation_continuing(): the step did not advance by one since the last compute (run 0, a new run). */
  continuing = false;

  /** cvm::step_relative(). */
  stepRelative(): number { return this.it - this.itRestart; }

  /** cvm::log. */
  log(text: string): void { this.host.log(text); }

  /** cvmodule->output_prefix(). */
  outputPrefixValue(): string { return this.outputPrefix; }

  /** output_stream on first use (backup of an existing file), kept open. */
  openOutput(name: string): void { this.openStream(name); }

  appendOutput(name: string, text: string): void { this.host.writeFile(name, text, true); }

  /** output_stream, write, close_output_stream: an existing file is backed up each time. */
  writeWholeFile(name: string, text: string): void {
    if (this.host.exists(name)) this.host.rename(name, name + (name.includes('.colvars.state') ? '.old' : '.BAK'));
    this.host.writeFile(name, text, false);
  }

  constructor(private readonly host: ColvarsHost) {}

  /** colvarproxy::add_config. */
  addConfig(cmd: 'config' | 'configfile', text: string): void {
    this.configQueue.push({ cmd, text });
  }

  /** colvarproxy_lammps::init: the engine's step, when it is not 0, starts the count; the timestep in fs. */
  initFromEngine(step: number, dtFs: number): void {
    if (step !== 0) { this.it = step; this.itRestart = step; }
    this.dt = dtFs;
  }

  /** colvarscript::run with the commands of this port, config and configfile (colvarscript_commands.h), which queue configuration. */
  runScript(words: string[]): void {
    const cmd = words[0];
    if (cmd !== 'config' && cmd !== 'configfile') throw colvarsError(`Error: the Colvars script command "${cmd}" is not supported by the browser engine (supported: config, configfile).`);
    if (words.length !== 2) throw colvarsError(`Error: wrong number of arguments to command "${cmd}"; 1 expected, ${words.length - 1} given.`);
    this.addConfig(cmd, words[1]);
  }

  /** colvarproxy::parse_module_config: reads the queued configuration. */
  parseModuleConfig(): void {
    while (this.configQueue.length) {
      const p = this.configQueue.shift()!;
      if (p.cmd === 'config') this.readConfig(p.text);
      else {
        const text = this.host.readFile(p.text);
        if (text === undefined) throw colvarsError(`Error: cannot read from file "${p.text}".`);
        this.host.log(`Reading new configuration from file "${p.text}":`);
        this.readConfig(text);
      }
    }
  }

  /** colvarmodule::parse_config: global parameters, colvars and then the biases by type. */
  private readConfig(text: string): void {
    const c = new Conf(text);
    // parse_global_params
    c.get('logLevel'); c.get('debugGradientsStepSize'); c.get('monitorEigenvalueCrossing'); c.get('eigenvalueCrossingThreshold');
    const units = c.get('units');
    if (units !== undefined) throw colvarsError('Error: the Colvars keyword "units" is not supported by the browser engine (the colvars use the units of the LAMMPS input).');
    if (c.all('indexFile').length) throw colvarsError('Error: the Colvars keyword "indexFile" is not supported by the browser engine.');
    c.get('smp');
    const an = c.get('analysis');
    if (an !== undefined) parseBool(an, 'analysis');
    const tf = c.get('colvarsTrajFrequency');
    if (tf !== undefined) this.trajFreq = parseInt10(tf, 'colvarsTrajFrequency');
    const rf = c.get('colvarsRestartFrequency');
    if (rf !== undefined) this.restartFreq = parseInt10(rf, 'colvarsRestartFrequency');
    if (this.trajFreq < 0 || this.restartFreq < 0) throw colvarsError('Error: colvarsTrajFrequency and colvarsRestartFrequency must not be negative.');
    for (const k of ['scriptedColvarForces', 'scriptingAfterBiases']) {
      const v = c.get(k);
      if (v !== undefined && parseBool(v, k)) throw colvarsError(`Error: "${k}" (scripted forces) is not supported by the browser engine.`);
    }
    if (c.get('sourceTclFile') !== undefined) throw colvarsError('Error: Tcl scripting (sourceTclFile) is not supported by the browser engine.');
    // parse_colvars
    let added = false;
    for (const v of c.all('colvar')) {
      if (v.trim() === '') throw colvarsError('Error: "colvar" keyword found without any configuration.');
      const cv = new Colvar(this.atoms, v, this.colvars.length + 1, (n) => this.colvars.some((x) => x.name === n), { angstrom: this.angstrom });
      this.colvars.push(cv);
      added = true;
    }
    // parse_biases
    for (const type of BIAS_TYPES) {
      const confs = c.all(type);
      if (confs.length === 0) continue;
      if (type !== 'harmonic' && type !== 'harmonicWalls' && type !== 'metadynamics') throw colvarsError(`Error: the Colvars bias "${type}" is not supported by the browser engine (supported: harmonic, harmonicWalls, metadynamics).`);
      for (const v of confs) {
        if (v.trim() === '') throw colvarsError(`Error: keyword "${type}" found without configuration.`);
        const lower = type.toLowerCase();
        const rank = (this.biasCount.get(lower) ?? 0) + 1;
        this.biasCount.set(lower, rank);
        this.biases.push(type === 'harmonic' ? new Harmonic(v, rank, this) : type === 'harmonicWalls' ? new HarmonicWalls(v, rank, this) : new Metadynamics(v, rank, this));
        added = true;
      }
    }
    // time_dependent_biases: more than one bias whose forces change with time (here the metadynamics biases)
    const timeBiases = this.biases.filter((b) => b instanceof Metadynamics).map((b) => b.name);
    if (timeBiases.length > 1) {
      this.host.log(`WARNING: there are ${timeBiases.length} time-dependent biases with non-zero force parameters:\n{${timeBiases.join(', ')}}\nPlease ensure that their forces do not counteract each other.`);
    }
    c.check('the global configuration');
    if (added) this.trajWriteLabels = true;
  }

  /** colvarproxy_lammps::setup: the timestep, atom groups (masses), the input state and the output files. */
  setup(dtFs: number): void {
    this.dt = dtFs;
    this.running = true;
    for (const cv of this.colvars) for (const cvc of cv.cvcs) for (const g of cvc.groups) g.setup();
    this.setupInput();
    this.setupOutput();
  }

  /** colvarmodule::setup_input: reads the input state once. */
  private setupInput(): void {
    if (this.inputPrefix === '') return;
    let name = this.inputPrefix + '.colvars.state';
    if (!this.host.exists(name)) {
      name = this.inputPrefix;
      if (!this.host.exists(name)) throw colvarsError(`Error: cannot open file "${name}" for reading (input state file).`);
    }
    this.inputPrefix = '';
    this.host.log(`Loading state from text file "${name}".`);
    this.readState(this.host.readFile(name) ?? '', name);
  }

  /** colvarmodule::read_state (formatted): the configuration block, then each colvar and bias. */
  private readState(text: string, name: string): void {
    const c = new Conf(text);
    const conf = c.all('configuration');
    if (conf.length) {
      const cc = new Conf(conf[0]);
      const step = cc.get('step');
      this.itRestart = step !== undefined ? parseInt10(step, 'step') : 0;
      this.it = this.itRestart;
    }
    for (const v of c.all('colvar')) {
      const cc = new Conf(v);
      const nm = (cc.get('name') ?? '').trim();
      const cv = this.colvars.find((x) => x.name === nm);
      if (!cv) { this.host.log(`Warning: input state file contains unknown colvar object "${nm}", discarding.`); continue; }
      const xs = cc.get('x');
      if (xs === undefined) throw colvarsError(`Error: restart file does not contain the value of the colvar "${nm}".`);
      const vals = splitValues(xs);
      const v0 = vals[0];
      cv.xRestart = Array.isArray(v0) ? Float64Array.from(v0) : Float64Array.of(v0 as number);
      if (cv.xRestart.length !== cv.x.length) throw colvarsError(`Error: in reading the state of colvar "${nm}" from "${name}": wrong number of values.`);
    }
    for (const v of c.all('metadynamics')) {
      const inner = new Conf(new Conf(v).all('configuration')[0] ?? '');
      const nm = (inner.get('name') ?? '').trim();
      if (this.biases.some((b) => b.name === nm)) throw colvarsError(`Error: reading the state of the metadynamics bias "${nm}" from "${name}" is not supported by the browser engine.`);
      this.host.log(`Warning: input state file contains unknown bias object "${nm}", discarding.`);
    }
    for (const key of ['restraint', 'harmonic']) {
      for (const v of c.all(key)) {
        const cc = new Conf(v);
        const inner = cc.all('configuration');
        const ic = new Conf(inner[0] ?? '');
        const nm = (ic.get('name') ?? '').trim();
        if (!this.biases.some((b) => b.name === nm)) this.host.log(`Warning: input state file contains unknown bias object "${nm}", discarding.`);
      }
    }
  }

  /** colvarmodule::setup_output. */
  private setupOutput(): void {
    if (this.outputPrefix !== this.pendingOutputPrefix) {
      this.outputPrefix = this.pendingOutputPrefix;
      if (this.open.has(this.trajName)) { this.open.delete(this.trajName); this.trajWriteLabels = true; }
      this.trajName = this.outputPrefix ? this.outputPrefix + '.colvars.traj' : '';
      for (const b of this.biases) b.setupOutput();
    }
  }

  /** The output prefix the fix sets at each setup (proxy->set_output_prefix). */
  pendingOutputPrefix = '';

  /** colvarproxy_lammps::compute: the step counter, the box, then colvarmodule::calc. Returns the bias energy. */
  compute(ntimestep: number, periodic: [boolean, boolean, boolean], A: number[], B: number[], C: number[]): number {
    if (this.firstTimestep) this.firstTimestep = false;
    else if (ntimestep - this.previousStep === 1) { this.it++; this.continuing = false; } else this.continuing = true;
    this.previousStep = ntimestep;
    this.bc.set(periodic[0], periodic[1], periodic[2], A, B, C);
    this.atoms.force.fill(0);
    this.biasEnergy = 0.0;
    this.calc();
    return this.biasEnergy;
  }

  /** colvarmodule::calc. */
  private calc(): void {
    // calc_colvars
    for (const cv of this.colvars) cv.calc(this.bc, this.running, this.it - this.itRestart);
    // calc_biases
    for (const cv of this.colvars) cv.fb.fill(0);
    let total = 0.0;
    for (const b of this.biases) b.update();
    for (const b of this.biases) total += b.energy;
    // update_colvar_forces
    for (const b of this.biases) b.communicateForces();
    this.biasEnergy += total;
    let colvarEnergy = 0.0;
    for (const cv of this.colvars) { cv.updateForces(); colvarEnergy += 0.0; }
    this.biasEnergy += colvarEnergy;
    for (const cv of this.colvars) if (cv.applyForceFlag) cv.communicateForces();
    // trajectory and restart files
    if (this.trajFreq && this.trajName) this.writeTrajFiles();
    if (this.restartFreq && this.it - this.itRestart > 0 && this.it % this.restartFreq === 0) {
      if (this.outputPrefix) this.writeRestartFile(this.outputPrefix + '.colvars.state');
    }
    // the output files of each bias at its outputFreq
    for (const b of this.biases) {
      if (b.outputFreq > 0 && this.it - this.itRestart > 0 && this.it % b.outputFreq === 0) b.writeOutputFiles();
    }
  }

  /** colvarproxy_io::output_stream on first use: an existing file is kept as NAME.old (state) or NAME.BAK. */
  private openStream(name: string): void {
    if (this.open.has(name)) return;
    if (this.host.exists(name)) this.host.rename(name, name + (name.includes('.colvars.state') ? '.old' : '.BAK'));
    this.host.writeFile(name, '', false);
    this.open.add(name);
  }

  /** colvarmodule::write_traj_files. */
  private writeTrajFiles(): void {
    this.openStream(this.trajName);
    let text = '';
    if (this.it - this.itRestart === 0 || this.trajWriteLabels || this.it % (this.trajFreq * 1000) === 0) {
      text += this.trajLabel();
      this.trajWriteLabels = false;
    }
    if (this.it % this.trajFreq === 0) text += this.trajLine();
    if (text) this.host.writeFile(this.trajName, text, true);
  }

  private trajLabel(): string {
    let s = '# ' + wrapString('step', IT_WIDTH - 2) + ' ';
    for (const cv of this.colvars) {
      const w = cv.outputWidth();
      s += ' ';
      if (cv.outputValue) s += ' ' + wrapString(cv.name, w);
      if (cv.outputAppliedForce) s += ' fa_' + wrapString(cv.name, w - 3);
    }
    for (const b of this.biases) s += b.trajLabel();
    return s + '\n';
  }

  private trajLine(): string {
    let s = padLeft(String(this.it), IT_WIDTH) + ' ';
    for (const cv of this.colvars) {
      s += ' ';
      if (cv.outputValue) s += ' ' + valueStream(cv.x, cv.type, fmtSci);
      if (cv.outputAppliedForce) s += ' ' + valueStream(cv.f, cv.type, fmtSci);
    }
    for (const b of this.biases) s += b.trajLine();
    return s + '\n';
  }

  /** colvarmodule::write_restart_file: the formatted state. */
  writeRestartFile(name: string): void {
    this.host.log(`Saving collective variables state to "${name}".`);
    if (this.host.exists(name)) this.host.rename(name, name + (name.includes('.colvars.state') ? '.old' : '.BAK'));
    this.host.writeFile(name, this.stateText(), false);
  }

  /** write_state_template_: the configuration block, each colvar, each bias. */
  stateText(): string {
    let s = `configuration {\n  step ${padLeft(String(this.it), IT_WIDTH)}\n  dt ${fmtDt(this.dt)}\n  version ${COLVARS_VERSION}\n}\n\n`;
    for (const cv of this.colvars) s += `colvar {\n  name ${cv.name}\n  x ${valueStream(cv.x, cv.type, fmtGen)}\n}\n\n`;
    for (const b of this.biases) s += b.stateText(this.it);
    return s;
  }

  /** colvarproxy::post_run: the final state file. */
  postRun(): void {
    if (!this.outputPrefix) return;
    this.writeRestartFile(this.outputPrefix + '.colvars.state');
    // colvarmodule::write_output_files: the biases whose files were not written at this step already
    for (const b of this.biases) {
      if (b.outputFreq === 0 || this.it - this.itRestart === 0 || this.it % b.outputFreq !== 0) b.writeOutputFiles();
    }
  }

  /** The number of array columns (the largest colvar size). */
  arrayCols(): number {
    let n = 0;
    for (const cv of this.colvars) n = Math.max(n, cv.x.length);
    return n;
  }
}
