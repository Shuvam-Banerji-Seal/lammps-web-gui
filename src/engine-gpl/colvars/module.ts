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
 * colvarcomp_distances.cpp (distance, distance_vec, distance_z, distance_xy, cartesian), colvarcomp_angles.cpp
 * (angle, dihedral), colvaratoms.cpp (atom_group parse, update_total_mass,
 * calc_center_of_mass, set_weighted_gradient, apply_colvar_force, apply_force), colvarbias.cpp and
 * colvarbias_restraint.cpp (the harmonic and harmonic_walls restraints), colvars_system.h (system_boundary_conditions), colvartypes.h/.cpp
 * and colvarvalue.cpp (vector arithmetic and stream formats), colvarscript_commands.h (config, configfile),
 * colvarproxy.cpp, colvarproxy_io.cpp (output streams, backup_file) and colvarproxy_lammps.cpp (init, setup, compute).
 *
 * The supported subset: the components distance, distanceVec, distanceZ, distanceXY, angle, dihedral and cartesian
 * (atom groups from atomNumbers, atomNumbersRange or a dummyAtom), linear or polynomial combinations of components
 * (componentCoeff, componentExp), periodic components (dihedral; distanceZ with period), the harmonic and
 * harmonicWalls restraints, the trajectory file (colvarsTrajFrequency) and the state file (colvarsRestartFrequency, the
 * end of each run, the input state). Every other component, bias or keyword stops with an error naming it. The
 * arithmetic follows the library's order of operations, so the values match native LAMMPS to the last bit.
 */
import { compileCFormat } from '../../engine/cformat';
import { Conf, colvarsError, parseBool, parseInt10, parseReal, splitValues } from './config';
import { acos } from '../libm/acos';

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

  abstract calcValue(): void;
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

/** The components of this port, and the ones of the library it refuses. */
const COMPONENTS: Record<string, (atoms: ColvarsAtoms, c: Conf, cvName: string) => Component> = {
  distance: (a, c, n) => new Distance(a, c, n),
  distancevec: (a, c, n) => new DistanceVec(a, c, n),
  cartesian: (a, c, n) => new Cartesian(a, c, n),
  angle: (a, c, n) => new Angle(a, c, n),
  dihedral: (a, c, n) => new Dihedral(a, c, n),
  distancez: (a, c, n) => new DistanceZ(a, c, n),
  distancexy: (a, c, n) => new DistanceXY(a, c, n),
};
const LIBRARY_COMPONENTS = [
  'distanceDir', 'distanceInv', 'distancePairs', 'dipoleMagnitude', 'dipoleAngle',
  'coordNum', 'selfCoordNum', 'groupCoord', 'hBond', 'rmsd', 'eigenvector', 'gyration', 'inertia',
  'inertiaZ', 'orientation', 'orientationAngle', 'orientationProj', 'spinAngle', 'tilt', 'alpha', 'dihedralPC',
  'polarTheta', 'polarPhi', 'linearCombination', 'customColvar', 'neuralNetwork', 'mapTotal', 'aspathCV', 'azpathCV',
  'gspath', 'gzpath', 'gspathCV', 'gzpathCV', 'euler_phi', 'euler_psi', 'euler_theta', 'dipoleDirection',
  'alchLambda', 'alchFLambda', 'componentDistance', 'distanceEuler',
];
const COMPONENT_NAMES = 'distance, distanceVec, distanceZ, distanceXY, angle, dihedral, cartesian';

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

  constructor(atoms: ColvarsAtoms, conf: string, index: number, existing: (name: string) => boolean) {
    const c = new Conf(conf);
    const nm = c.get('name');
    this.name = nm !== undefined ? nm.trim() : `colvar${index}`;
    if (existing(this.name)) throw colvarsError(`Error: this colvar cannot have the same name, "${this.name}", as another colvar.`);
    const w = c.get('width');
    if (w !== undefined) {
      this.width = parseReal(w, 'width');
      if (!(this.width > 0.0)) throw colvarsError('Error: "width" must be positive.');
    }
    for (const k of ['lowerBoundary', 'upperBoundary']) { const v = c.get(k); if (v !== undefined) parseReal(v, k); }
    for (const k of ['hardLowerBoundary', 'hardUpperBoundary', 'expandBoundaries']) { const v = c.get(k); if (v !== undefined) parseBool(v, k); }
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
      for (const v of c.all(lower)) this.cvcs.push(make(atoms, new Conf(v), this.name));
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
    const size = this.cvcs[0].x.length;
    if (this.type === 'vector') for (const cvc of this.cvcs) if (cvc.x.length !== size) throw colvarsError(`Error: the components of colvar "${this.name}" have different sizes.`);
    this.x = new Float64Array(size);
    this.fb = new Float64Array(size);
    this.f = new Float64Array(size);
  }

  /** colvar::calc: component values and gradients, then collect_cvc_values. */
  calc(bc: Boundaries, simulationRunning: boolean): void {
    for (const cvc of this.cvcs) {
      cvc.readData(bc);
      cvc.calcValue();
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

/** colvarbias_restraint_harmonic. */
/** The restraint biases: colvarbias and colvarbias_restraint (names, colvars, output, the state block). */
abstract class Restraint {
  abstract readonly biasType: string;
  readonly stateKeyword = 'restraint';
  name = '';
  readonly colvars: Colvar[] = [];
  forceK = 1.0;
  outputEnergy = false;
  energy = 0.0;
  forces: Float64Array[] = [];

  /** colvarbias::init and colvarbias_restraint_k::init: the keywords every restraint reads. */
  protected initBase(c: Conf, rank: number, module: ColvarsModule, unsupported: string[]): void {
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
    const oe = c.get('outputEnergy');
    if (oe !== undefined) this.outputEnergy = parseBool(oe, 'outputEnergy');
    const of = c.get('outputFreq');
    if (of !== undefined) parseInt10(of, 'outputFreq');
    const sz = c.get('stepZeroData');
    if (sz !== undefined) parseBool(sz, 'stepZeroData');
    for (const k of ['targetForceConstant', 'targetNumSteps', 'targetNumStages', 'lambdaSchedule', 'targetEquilSteps', 'writeTIPMF', 'writeTISamples', 'decoupling', 'scaledBiasingForce', 'timeStepFactor', 'outputAccumulatedWork', ...unsupported]) {
      const v = c.get(k);
      if (v !== undefined && !/^\s*(off|no|false)\s*$/i.test(v)) throw colvarsError(`Error: the ${this.biasType} option "${k}" is not supported by the browser engine.`);
    }
    // colvarbias_restraint_k::init: forceConstant, default 1
    const k = c.get('forceConstant');
    if (k !== undefined) this.forceK = parseReal(k, 'forceConstant');
    if (this.forceK < 0.0) throw colvarsError('Error: undefined or invalid force constant.');
    this.forces = this.colvars.map((cv) => new Float64Array(cv.x.length));
  }

  /** colvarbias_restraint::update: the energy and the force on each colvar. */
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

  stateText(step: number): string {
    return `${this.stateKeyword} {\n  configuration {\n    step ${step}\n    name ${this.name}\n  }\n}\n\n`;
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
  readonly biases: Restraint[] = [];
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
  /** integration timestep in fs (dt * force->femtosecond). */
  dt = 1.0;
  readonly bc = new Boundaries();
  biasEnergy = 0.0;

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
      const cv = new Colvar(this.atoms, v, this.colvars.length + 1, (n) => this.colvars.some((x) => x.name === n));
      this.colvars.push(cv);
      added = true;
    }
    // parse_biases
    for (const type of BIAS_TYPES) {
      const confs = c.all(type);
      if (confs.length === 0) continue;
      if (type !== 'harmonic' && type !== 'harmonicWalls') throw colvarsError(`Error: the Colvars bias "${type}" is not supported by the browser engine (supported: harmonic, harmonicWalls).`);
      for (const v of confs) {
        if (v.trim() === '') throw colvarsError(`Error: keyword "${type}" found without configuration.`);
        const lower = type.toLowerCase();
        const rank = (this.biasCount.get(lower) ?? 0) + 1;
        this.biasCount.set(lower, rank);
        this.biases.push(type === 'harmonic' ? new Harmonic(v, rank, this) : new HarmonicWalls(v, rank, this));
        added = true;
      }
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
    }
  }

  /** The output prefix the fix sets at each setup (proxy->set_output_prefix). */
  pendingOutputPrefix = '';

  /** colvarproxy_lammps::compute: the step counter, the box, then colvarmodule::calc. Returns the bias energy. */
  compute(ntimestep: number, periodic: [boolean, boolean, boolean], A: number[], B: number[], C: number[]): number {
    if (this.firstTimestep) this.firstTimestep = false;
    else if (ntimestep - this.previousStep === 1) this.it++;
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
    for (const cv of this.colvars) cv.calc(this.bc, this.running);
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
    if (this.outputPrefix) this.writeRestartFile(this.outputPrefix + '.colvars.state');
  }

  /** The number of array columns (the largest colvar size). */
  arrayCols(): number {
    let n = 0;
    for (const cv of this.colvars) n = Math.max(n, cv.x.length);
    return n;
  }
}
