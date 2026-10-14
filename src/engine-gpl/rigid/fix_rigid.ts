/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/RIGID/fix_rigid.cpp and
 * src/RIGID/fix_rigid.h (class FixRigid: the body setup of setup_bodies_static and setup_bodies_dynamic, init,
 * setup, set_xv, set_v, compute_forces_and_torques, pre_neighbor, dof, compute_scalar, compute_array), for the
 * bodystyles single, molecule and group with point particles in 3d.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). The source carries no contributing-author line.
 *
 * This is the base class of the Nose-Hoover rigid-body fixes (fix_rigid_nh.ts). It is not a registered style by
 * itself: fix rigid, rigid/nve and the other NVE rigid styles keep their own core implementation (src/engine/fix/rigid.ts).
 *
 * Body state follows the source: principal axes ex, ey, ez from jacobi3 (ascending moments), the quaternion
 * from exyz_to_q, the image flags of the body centre of mass (imagebody) remapped in pre_neighbor, and an atom's
 * image relative to its body (xcmimage = image - imagebody) used wherever the source uses xcmimage.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError, typeBounds } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf } from '../../engine/atoms';
import { exyzToQ, transposeMatvec, matvec, angmomToOmega, jacobi3, quatToMat, type V3, type Q4 } from './math_extra';

/** rigid_const.h: EPSILON = 1.0e-7 */
export const EPSILON = 1.0e-7;

export interface Body {
  /** atom IDs of the members, in atom order */
  atoms: number[];
  nrigid: number;
  mass: number;
  xcm: V3;
  vcm: V3;
  fcm: V3;
  torque: V3;
  angmom: V3;
  omega: V3;
  inertia: V3;
  ex: V3;
  ey: V3;
  ez: V3;
  quat: Q4;
  /** image flags of the body centre of mass (domain->remap of xcm) */
  imagebody: V3;
  /** body-frame coordinates of each member */
  displace: V3[];
  /** force and torque flags (fix rigid force / torque keywords), 1 or 0 per space-frame component */
  fflag: number[];
  tflag: number[];
  /** Nose-Hoover conjugate quaternion momentum (fix rigid/nvt and the other NH styles) */
  conjqm: Q4;
}

const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Thermostat settings of the Nose-Hoover styles (keywords temp and tparam, fix_rigid.html). */
export interface ThermostatKeywords {
  tstat: boolean;
  tStart: number;
  tStop: number;
  tPeriod: number;
  tChain: number;
  tIter: number;
  tOrder: number;
}

/** Barostat settings of the pressure-coupled styles (fix_rigid.html: iso, aniso, x, y, z, couple, dilate, pchain). */
export interface PressureKeywords {
  pstat: boolean;
  pFlag: number[];
  pStart: number[];
  pStop: number[];
  pPeriod: number[];
  /** couple: none, xyz, xy, yz or xz (fix_rigid.cpp: pcouple) */
  pcouple: 'none' | 'xyz' | 'xy' | 'yz' | 'xz';
  /** dilate all (true) or the group named by dilate */
  allremap: boolean;
  dilate: string | null;
  pChain: number;
}

/**
 * Parses the bodystyle and every keyword of the rigid styles handled here: force, torque, reinit (all styles),
 * temp, tparam (the Nose-Hoover styles) and iso, aniso, x, y, z, couple, dilate, pchain (the barostats).
 * Other keywords are refused with a StyleError naming them.
 * Defaults (fix_rigid.cpp constructor): tparam 10 1 3, pchain 10, dilate all, couple none.
 */
export const parseRigidArgs = (style: string, args: string[], dimension: 2 | 3) => {
  const bodystyle = args[0];
  const out = {
    bodystyle: 'single' as 'single' | 'molecule' | 'group',
    groupList: [] as string[],
    forceSpecs: [] as [string, number[]][],
    torqueSpecs: [] as [string, number[]][],
    reinit: true,
    mol: null as string | null,
    thermo: { tstat: false, tStart: 0, tStop: 0, tPeriod: 0, tChain: 10, tIter: 1, tOrder: 3 } as ThermostatKeywords,
    press: { pstat: false, pFlag: [0, 0, 0], pStart: [0, 0, 0], pStop: [0, 0, 0], pPeriod: [0, 0, 0], pcouple: 'none', allremap: true, dilate: null, pChain: 10 } as PressureKeywords,
  };
  const num = (w: string | undefined, what: string): number => {
    const v = Number(w);
    if (w === undefined || w === '' || !Number.isFinite(v)) throw new StyleError(`fix ${style} ${what}: expected a number, got '${w ?? ''}'`);
    return v;
  };
  const pr = out.press;
  const pset = (d: number, a: string | undefined, b: string | undefined, c: string | undefined) => {
    pr.pStart[d] = num(a, `${style} Pstart`);
    pr.pStop[d] = num(b, `${style} Pstop`);
    pr.pPeriod[d] = num(c, `${style} Pperiod`);
    pr.pFlag[d] = 1;
  };
  let k = 1;
  // fix_rigid_small.cpp: the /small styles take molecule bodies only (bodystyle single and group are for the rigid styles)
  const small = style.includes('small');
  if (small && bodystyle !== 'molecule') throw new StyleError(`fix ${style}: bodystyle ${bodystyle ?? ''} is only allowed for the rigid styles, not ${style} (bodystyle molecule)`);
  if (bodystyle === 'single' || bodystyle === 'molecule') out.bodystyle = bodystyle;
  else if (bodystyle === 'group') {
    out.bodystyle = 'group';
    const n = Number(args[1]);
    if (!Number.isInteger(n) || n < 1) throw new StyleError(`fix ${style} group: N must be a positive integer`);
    out.groupList = args.slice(2, 2 + n);
    if (out.groupList.length !== n) throw new StyleError(`fix ${style} group: expected ${n} group IDs`);
    k = 2 + n;
  } else if (bodystyle === 'custom') {
    throw new StyleError(`fix ${style}: bodystyle custom is not supported by the browser engine's native-parity rigid/nvt`);
  } else throw new StyleError(`fix ${style}: unknown bodystyle '${bodystyle ?? ''}' (single, molecule or group)`);
  const onoff = (w: string | undefined) => {
    if (w !== 'on' && w !== 'off') throw new StyleError(`fix ${style}: force/torque flags must be on or off`);
    return w === 'on' ? 1 : 0;
  };
  while (k < args.length) {
    const key = args[k];
    if ((key === 'force' || key === 'torque') && small) {
      throw new StyleError(`fix ${style}: "The force and torque keywords discussed next are only allowed for the rigid styles."`);
    }
    if (key === 'force' || key === 'torque') {
      if (args[k + 4] === undefined) throw new StyleError(`fix ${style} ${key}: expected M xflag yflag zflag`);
      const flags = [onoff(args[k + 2]), onoff(args[k + 3]), onoff(args[k + 4])];
      (key === 'force' ? out.forceSpecs : out.torqueSpecs).push([args[k + 1], flags]);
      k += 5;
    } else if (key === 'reinit') {
      if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError(`fix ${style}: reinit must be yes or no`);
      out.reinit = args[k + 1] === 'yes';
      k += 2;
    } else if (key === 'temp') {
      // fix_rigid.html: "temp values = Tstart Tstop Tperiod"
      out.thermo.tstat = true;
      out.thermo.tStart = num(args[k + 1], 'temp Tstart');
      out.thermo.tStop = num(args[k + 2], 'temp Tstop');
      out.thermo.tPeriod = num(args[k + 3], 'temp Tperiod');
      k += 4;
    } else if (key === 'iso' || key === 'aniso') {
      // fix_rigid.cpp: iso sets pcouple = XYZ and the same settings on x, y and z (z dropped in 2d)
      if (key === 'iso') pr.pcouple = 'xyz';
      const dims = dimension === 2 ? 2 : 3;
      for (let d = 0; d < dims; d++) pset(d, args[k + 1], args[k + 2], args[k + 3]);
      k += 4;
    } else if (key === 'x' || key === 'y' || key === 'z') {
      pset(key === 'x' ? 0 : key === 'y' ? 1 : 2, args[k + 1], args[k + 2], args[k + 3]);
      k += 4;
    } else if (key === 'couple') {
      const c = args[k + 1];
      if (c !== 'xyz' && c !== 'xy' && c !== 'yz' && c !== 'xz' && c !== 'none') throw new StyleError(`fix ${style} couple: must be xyz, xy, yz, xz or none, got '${c ?? ''}'`);
      pr.pcouple = c;
      k += 2;
    } else if (key === 'dilate') {
      const g = args[k + 1];
      if (g === undefined) throw new StyleError(`fix ${style}: keyword dilate needs all or a group ID`);
      if (g === 'all') { pr.allremap = true; pr.dilate = null; } else { pr.allremap = false; pr.dilate = g; }
      k += 2;
    } else if (key === 'pchain') {
      pr.pChain = Math.trunc(num(args[k + 1], 'pchain'));
      k += 2;
    } else if (key === 'mol' && small) {
      // fix_rigid.html: "The *mol* keyword can only be used with the *rigid/small* styles." (the template of fix deposit rigid)
      if (args[k + 1] === undefined) throw new StyleError(`fix ${style}: keyword mol needs a molecule template-ID`);
      out.mol = args[k + 1];
      k += 2;
    } else if (key === 'tparam') {
      // fix_rigid.html: "tparam values = Tchain Titer Torder"
      out.thermo.tChain = Math.trunc(num(args[k + 1], 'tparam Tchain'));
      out.thermo.tIter = Math.trunc(num(args[k + 2], 'tparam Titer'));
      out.thermo.tOrder = Math.trunc(num(args[k + 3], 'tparam Torder'));
      k += 4;
    } else {
      throw new StyleError(`fix ${style}: keyword '${key}' is not supported by the browser engine's native-parity ${style} (supported: bodystyle, force, torque, reinit, temp, tparam, iso, aniso, x, y, z, couple, dilate, pchain, and mol for rigid/small)`);
    }
  }
  // fix_rigid.cpp: pstat_flag = 1 if any p_flag is set
  pr.pstat = pr.pFlag[0] + pr.pFlag[1] + pr.pFlag[2] > 0;
  return out;
};

export abstract class FixRigidBase extends Fix {
  readonly style: string;
  bodystyle: 'single' | 'molecule' | 'group';
  /** the /small styles (fix_rigid_nh_small.cpp): molecule bodies, and the Kamberaj energy in compute_scalar */
  readonly small: boolean;
  protected groupList: string[];
  protected reinitFlag: boolean;
  protected bodies: Body[] = [];
  protected nbody = 0;
  /** linear bodies counted by the last dof() call (persistent, as in the source) */
  protected nlinear = 0;
  protected setupFlag = false;
  protected builtOnce = false;
  protected tfactor = 0;
  protected dtv = 0;
  protected dtf = 0;
  protected dtq = 0;
  protected thermo: ThermostatKeywords;
  protected press: PressureKeywords;
  protected forceSpecs: [string, number[]][];
  protected torqueSpecs: [string, number[]][];
  /** keyword mol (rigid/small only): the molecule template-ID that fix deposit rigid must name too */
  protected molTemplateId: string | null = null;

  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args);
    this.style = style;
    this.timeIntegrate = true;
    this.virialGlobal = true;
    this.thermoVirial = true;
    this.scalarFlag = true;
    this.arrayFlag = true;
    this.sizeArrayCols = 15;
    const kw = parseRigidArgs(style, args, sys.dimension);
    this.small = style.includes('small');
    this.bodystyle = kw.bodystyle;
    this.groupList = kw.groupList;
    this.reinitFlag = kw.reinit;
    this.forceSpecs = kw.forceSpecs;
    this.torqueSpecs = kw.torqueSpecs;
    this.thermo = kw.thermo;
    this.press = kw.press;
    this.molTemplateId = kw.mol;
    if (this.molTemplateId && !sys.molecules.has(this.molTemplateId)) throw new StyleError(`fix ${style}: molecule template ${this.molTemplateId} does not exist`);
    if (this.molTemplateId && this.bodystyle !== 'molecule') throw new StyleError(`fix ${style}: the mol keyword requires bodystyle molecule`);
    for (const g of this.groupList) sys.groups.bit(g);
    if (this.press.dilate) sys.groups.bit(this.press.dilate);
  }

  /** The mol template-ID of this fix (null without the keyword); fix deposit rigid must name the same one. */
  moleculeTemplateId(): string | null { return this.molTemplateId; }

  /**
   * Adds the body of a molecule deposited by fix deposit with the rigid keyword (fix_rigid_small.cpp set_molecule,
   * which the deposit calls with the new atom IDs). The body is built from the atoms as they were inserted; the
   * source takes the template's geometry instead, which is the same rigid molecule. As in set_molecule, the new body
   * starts at rest: its angular momentum, omega and conjugate quaternion momentum are zero.
   */
  addMolecule(atomIds: readonly number[]): void {
    if (this.bodystyle !== 'molecule') throw new StyleError(`fix ${this.style}: the mol keyword needs bodystyle molecule`);
    const idx = this.indexById();
    const ii = atomIds.map((id) => {
      const k = id < idx.length ? idx[id] : -1;
      if (k < 0) throw new StyleError(`fix ${this.style}: atom ${id} does not exist`);
      return k;
    });
    const b = this.staticBody(ii);
    b.angmom = [0, 0, 0];
    b.omega = [0, 0, 0];
    b.conjqm = [0, 0, 0, 0];
    this.bodies.push(b);
    this.nbody = this.bodies.length;
    this.sizeArrayRows = this.bodies.length;
  }

  /** Index of each atom ID (atoms are looked up by ID every time, as the atoms may be reordered). */
  protected indexById(): Int32Array {
    const s = this.sys.state;
    let max = 0;
    for (let i = 0; i < s.n; i++) if (s.id[i] > max) max = s.id[i];
    const idx = new Int32Array(max + 1).fill(-1);
    for (let i = 0; i < s.n; i++) idx[s.id[i]] = i;
    return idx;
  }

  /** Box edge length of dimension d. */
  protected edge(d: number): number {
    const b = this.sys.state.box;
    return b.hi[d] - b.lo[d];
  }

  /** Unwrapped position of atom i relative to its body's image: x + xcmimage * L (xcmimage = image - imagebody). */
  protected unmapRel(i: number, b: Body, out: V3): void {
    const s = this.sys.state;
    for (let d = 0; d < 3; d++) out[d] = s.x[3 * i + d] + (s.image[3 * i + d] - b.imagebody[d]) * this.edge(d);
  }

  /** Builds the bodies from the current atoms (setup_bodies_static + setup_bodies_dynamic). */
  protected buildBodies(): void {
    const sys = this.sys;
    const s = sys.state;
    const members = new Map<number, number[]>();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      let key: number | null = null;
      if (this.bodystyle === 'single') key = 1;
      else if (this.bodystyle === 'molecule') key = s.molecule[i];
      else {
        const k = this.groupList.findIndex((gname) => (s.mask[i] & sys.groups.bit(gname)) !== 0);
        if (k >= 0) key = k + 1;
      }
      if (key === null) continue;
      if (!members.has(key)) members.set(key, []);
      members.get(key)!.push(i);
    }
    const keys = [...members.keys()].sort((a, b) => a - b);
    for (const key of keys) {
      if (members.get(key)!.length < 2) throw new StyleError(`fix ${this.style}: "Each rigid body must have two or more atoms." (body ${key})`);
    }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (s.radius && s.radius[i] > 0) throw new StyleError(`fix ${this.style}: finite-size particles are not supported by the browser engine's native-parity ${this.style}`);
    }
    this.bodies = keys.map((key) => this.staticBody(members.get(key)!));
    const nb = this.bodies.length;
    this.nbody = nb;
    for (const [spec, flags] of this.forceSpecs) { const [lo, hi] = typeBounds(spec, nb); for (let k = lo; k <= hi; k++) this.bodies[k - 1].fflag = flags.slice(); }
    for (const [spec, flags] of this.torqueSpecs) { const [lo, hi] = typeBounds(spec, nb); for (let k = lo; k <= hi; k++) this.bodies[k - 1].tflag = flags.slice(); }
    this.sizeArrayRows = nb;
  }

  /**
   * setup_bodies_static and setup_bodies_dynamic for one body. Order of the source: the centre of mass from the
   * unwrapped positions (imagebody = 0), remapped into the box by pre_neighbor(), then the atoms' offsets from it
   * (through xcmimage = image - imagebody), the inertia tensor, principal axes (jacobi3, ascending), the quaternion,
   * the body-frame displacements, and the velocities and angular momentum from the atoms.
   */
  private staticBody(idx: number[]): Body {
    const s = this.sys.state;
    // the NH port has no finite-size ellipsoid or point dipole bodies: refused by name, never treated as points
    for (const i of idx) {
      if (s.shape && s.shape[3 * i] > 0) throw new StyleError(`fix ${this.style}: finite-size ellipsoid atom ${s.id[i]} is not supported by the browser engine's native-parity ${this.style}`);
      if (s.mu && s.mu[4 * i + 3] > 0) throw new StyleError(`fix ${this.style}: point dipole atom ${s.id[i]} is not supported by the browser engine's native-parity ${this.style}`);
    }
    const box = s.box;
    const xcm: V3 = [0, 0, 0];
    let M = 0;
    const unw = idx.map((i) => {
      for (let d = 0; d < 3; d++) if (!box.periodic[d] && s.image[3 * i + d] !== 0) throw new StyleError(`fix ${this.style}: an atom of a body has a non-zero image flag in a non-periodic dimension`);
      return [0, 1, 2].map((d) => s.x[3 * i + d] + s.image[3 * i + d] * this.edge(d));
    });
    idx.forEach((i, n) => {
      const m = massOf(s, i);
      M += m;
      for (let d = 0; d < 3; d++) xcm[d] += unw[n][d] * m;
    });
    for (let d = 0; d < 3; d++) xcm[d] = xcm[d] / M;
    const body: Body = {
      atoms: idx.map((i) => s.id[i]),
      nrigid: idx.length,
      mass: M,
      xcm,
      vcm: [0, 0, 0],
      fcm: [0, 0, 0],
      torque: [0, 0, 0],
      angmom: [0, 0, 0],
      omega: [0, 0, 0],
      inertia: [0, 0, 0],
      ex: [1, 0, 0], ey: [0, 1, 0], ez: [0, 0, 1],
      quat: [1, 0, 0, 0],
      imagebody: [0, 0, 0],
      displace: [],
      fflag: [1, 1, 1],
      tflag: [1, 1, 1],
      conjqm: [0, 0, 0, 0],
    };
    this.remapBody(body);
    // offsets of the atoms from the centre of mass, unwrapped relative to the body image
    const delta = idx.map((i) => {
      const u: V3 = [0, 0, 0];
      this.unmapRel(i, body, u);
      return [u[0] - body.xcm[0], u[1] - body.xcm[1], u[2] - body.xcm[2]];
    });
    // inertia tensor about the centre of mass (sum[0..5] of setup_bodies_static)
    const tensor = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    idx.forEach((i, n) => {
      const m = massOf(s, i);
      const [dx, dy, dz] = delta[n];
      tensor[0][0] += m * (dy * dy + dz * dz);
      tensor[1][1] += m * (dx * dx + dz * dz);
      tensor[2][2] += m * (dx * dx + dy * dy);
      tensor[1][2] -= m * dy * dz;
      tensor[0][2] -= m * dx * dz;
      tensor[0][1] -= m * dx * dy;
    });
    tensor[2][1] = tensor[1][2];
    tensor[2][0] = tensor[0][2];
    tensor[1][0] = tensor[0][1];
    const { eval: ev, evec } = jacobi3(tensor);
    const inertia: V3 = [ev[0], ev[1], ev[2]];
    const ex: V3 = [evec[0][0], evec[1][0], evec[2][0]];
    const ey: V3 = [evec[0][1], evec[1][1], evec[2][1]];
    const ez: V3 = [evec[0][2], evec[1][2], evec[2][2]];
    const max = Math.max(inertia[0], inertia[1], inertia[2]);
    for (let k = 0; k < 3; k++) if (inertia[k] < EPSILON * max) inertia[k] = 0.0;
    // right-handed frame: flip ez if needed
    if (dot(cross(ex, ey), ez) < 0.0) for (let d = 0; d < 3; d++) ez[d] = -ez[d];
    body.inertia = inertia;
    body.ex = ex;
    body.ey = ey;
    body.ez = ez;
    body.quat = exyzToQ(ex, ey, ez);
    body.displace = delta.map((dl) => {
      const out: V3 = [0, 0, 0];
      transposeMatvec(ex, ey, ez, dl, out);
      return out;
    });
    this.dynamicBody(body, idx);
    return body;
  }

  /** domain->remap of the centre of mass into the periodic box, with its image flags (imagebody). */
  protected remapBody(b: Body): void {
    const box = this.sys.state.box;
    for (let d = 0; d < 3; d++) {
      if (!box.periodic[d]) continue;
      const lo = box.lo[d], hi = box.hi[d], L = hi - lo;
      while (b.xcm[d] < lo) { b.xcm[d] += L; b.imagebody[d] -= 1; }
      while (b.xcm[d] >= hi) { b.xcm[d] -= L; b.imagebody[d] += 1; }
      b.xcm[d] = Math.max(b.xcm[d], lo);
    }
  }

  /** setup_bodies_dynamic: vcm = sum m v / M and angmom = sum m (dr x v), from the current atoms. */
  protected dynamicBody(b: Body, idx: number[]): void {
    const s = this.sys.state;
    const vcm: V3 = [0, 0, 0], angmom: V3 = [0, 0, 0];
    const ur: V3 = [0, 0, 0];
    idx.forEach((i) => {
      const m = massOf(s, i);
      vcm[0] += s.v[3 * i] * m; vcm[1] += s.v[3 * i + 1] * m; vcm[2] += s.v[3 * i + 2] * m;
    });
    // dr = unwrapped position minus centre of mass (xcmimage relative)
    idx.forEach((i) => {
      const m = massOf(s, i);
      this.unmapRel(i, b, ur);
      const dx = ur[0] - b.xcm[0], dy = ur[1] - b.xcm[1], dz = ur[2] - b.xcm[2];
      const vx = s.v[3 * i], vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
      angmom[0] += dy * m * vz - dz * m * vy;
      angmom[1] += dz * m * vx - dx * m * vz;
      angmom[2] += dx * m * vy - dy * m * vx;
    });
    for (let d = 0; d < 3; d++) b.vcm[d] = vcm[d] / b.mass;
    b.angmom = angmom;
    angmomToOmega(b.angmom, b.ex, b.ey, b.ez, b.inertia, b.omega);
  }

  /** FixRigid::init: bodies at the first init (and at every init with reinit yes), dt factors, temperature factor. */
  init(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    this.dtq = 0.5 * s.dt;
    if (this.reinitFlag || !this.setupFlag) {
      this.buildBodies();
      this.setupFlag = true;
    }
    this.builtOnce = true;
    // nlinear is the count of the last dof() call (FixRigid::dof); the source sets it in dof(), which runs after
    // the fix init, so the first run subtracts 0 here. Measured with native LAMMPS (black box): the first run of
    // fix rigid/nvt on 9 dimers among 27 bodies has ndof 162 (f_1 = 2 KE / 162), later runs 153.
    let ndof = 0;
    for (const b of this.bodies) {
      ndof += b.fflag[0] + b.fflag[1] + b.fflag[2];
      ndof += b.tflag[0] + b.tflag[1] + b.tflag[2];
    }
    ndof -= this.nlinear;
    this.tfactor = ndof > 0 ? s.units.mvv2e / (ndof * s.units.boltz) : 0;
  }

  /**
   * compute_forces_and_torques: total force and torque on each body. Native: sums f and the torque about the centre
   * of mass with the unwrapped positions relative to xcmimage.
   */
  protected computeForcesAndTorques(): void {
    const s = this.sys.state;
    const idx = this.indexById();
    const ur: V3 = [0, 0, 0];
    for (const b of this.bodies) {
      const F: V3 = [0, 0, 0], T: V3 = [0, 0, 0];
      for (const id of b.atoms) {
        const i = idx[id];
        const fx = s.f[3 * i], fy = s.f[3 * i + 1], fz = s.f[3 * i + 2];
        F[0] += fx; F[1] += fy; F[2] += fz;
        this.unmapRel(i, b, ur);
        const dx = ur[0] - b.xcm[0], dy = ur[1] - b.xcm[1], dz = ur[2] - b.xcm[2];
        T[0] += dy * fz - dz * fy;
        T[1] += dz * fx - dx * fz;
        T[2] += dx * fy - dy * fx;
      }
      b.fcm = F;
      b.torque = T;
    }
  }

  /**
   * set_xv (setX = true) or set_v (setX = false): atom velocities (and positions) from the body state, with the
   * constraint-force virial of the source, factor 0.5 per call (the setup call adds 2 * 0.5 = 1 in FixRigid::setup).
   */
  protected setXV(setX: boolean, virialFactor: number): void {
    const s = this.sys.state;
    const idx = this.indexById();
    const dtf = this.dtf;
    const vir = this.virial;
    const L = [this.edge(0), this.edge(1), this.edge(2)];
    const xbox: V3 = [0, 0, 0];
    const delta: V3 = [0, 0, 0];
    for (const b of this.bodies) {
      b.atoms.forEach((id, n) => {
        const i = idx[id];
        const m = massOf(s, i);
        for (let d = 0; d < 3; d++) xbox[d] = s.image[3 * i + d] - b.imagebody[d];
        const x0 = [0, 1, 2].map((d) => s.x[3 * i + d] + xbox[d] * L[d]);
        const v0 = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
        matvec(b.ex, b.ey, b.ez, b.displace[n], delta);
        const w = b.omega;
        for (let d = 0; d < 3; d++) {
          const c = [w[1] * delta[2] - w[2] * delta[1], w[2] * delta[0] - w[0] * delta[2], w[0] * delta[1] - w[1] * delta[0]];
          s.v[3 * i + d] = c[d] + b.vcm[d];
        }
        if (setX) {
          for (let d = 0; d < 3; d++) s.x[3 * i + d] = delta[d] + (b.xcm[d] - xbox[d] * L[d]);
        }
        if (virialFactor) {
          const fc = [0, 1, 2].map((d) => (m * (s.v[3 * i + d] - v0[d])) / dtf - s.f[3 * i + d]);
          vir[0] += virialFactor * x0[0] * fc[0];
          vir[1] += virialFactor * x0[1] * fc[1];
          vir[2] += virialFactor * x0[2] * fc[2];
          vir[3] += virialFactor * x0[0] * fc[1];
          vir[4] += virialFactor * x0[0] * fc[2];
          vir[5] += virialFactor * x0[1] * fc[2];
        }
      });
    }
  }

  /** FixRigid::setup: forces and torques, angmom -> omega, set_v, and the virial (set_v at factor 1/2, doubled). */
  setup(): void {
    this.computeForcesAndTorques();
    for (const b of this.bodies) angmomToOmega(b.angmom, b.ex, b.ey, b.ez, b.inertia, b.omega);
    this.virial.fill(0);
    this.setXV(false, 0.5);
    for (let n = 0; n < 6; n++) this.virial[n] *= 2.0;
  }

  /** pre_neighbor: remap each body's centre of mass into the periodic box (domain->remap with imagebody). */
  preNeighbor(): void {
    for (const b of this.bodies) this.remapBody(b);
  }

  /**
   * FixRigid::dof: the degrees of freedom the bodies take away from a temperature compute (3 per atom minus 6, or
   * 5 for a linear body, counted only for bodies entirely in the group).
   */
  dofRemoved(groupBit: number): number {
    if (!this.setupFlag) return 0;
    const s = this.sys.state;
    const idx = this.indexById();
    let n = 0;
    this.nlinear = 0;
    for (const b of this.bodies) {
      if (!b.atoms.every((id) => (s.mask[idx[id]] & groupBit) !== 0)) continue;
      n += 3 * b.atoms.length - 6;
      if (b.inertia[0] === 0.0 || b.inertia[1] === 0.0 || b.inertia[2] === 0.0) { n++; this.nlinear++; }
    }
    return n;
  }

  /**
   * FixRigid::compute_scalar: temperature of the rigid bodies, (sum m v_cm^2 + sum I w_body^2) * tfactor.
   * The Nose-Hoover subclasses add their extended-system energy to it.
   */
  computeScalar(): number {
    let t = 0.0;
    for (const b of this.bodies) {
      t += b.mass * (b.fflag[0] * b.vcm[0] * b.vcm[0] + b.fflag[1] * b.vcm[1] * b.vcm[1] + b.fflag[2] * b.vcm[2] * b.vcm[2]);
      // body-frame angular momentum: the columns of the rotation matrix are the principal axes
      const rot = quatToMat(b.quat);
      const cols = [0, 1, 2].map((c) => [rot[0][c], rot[1][c], rot[2][c]]);
      const wb: V3 = [0, 0, 0];
      transposeMatvec(cols[0], cols[1], cols[2], b.angmom, wb);
      for (let k = 0; k < 3; k++) wb[k] = b.inertia[k] === 0.0 ? 0.0 : wb[k] / b.inertia[k];
      for (let k = 0; k < 3; k++) t += b.tflag[k] * b.inertia[k] * wb[k] * wb[k];
    }
    return t * this.tfactor;
  }

  /** compute_array: xcm, vcm, fcm, torque and the body image flags, 15 columns per body (0-based i, j). */
  computeArray(i: number, j: number): number {
    const b = this.bodies[i];
    if (!b) throw new StyleError(`fix ${this.id}: rigid body ${i + 1} does not exist`);
    if (j < 3) return b.xcm[j];
    if (j < 6) return b.vcm[j - 3];
    if (j < 9) return b.fcm[j - 6];
    if (j < 12) return b.torque[j - 9];
    return b.imagebody[j - 12];
  }
}
