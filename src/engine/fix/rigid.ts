import { Fix } from './fix';
import { StyleError, typeBounds } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * fix ID group rigid|rigid/nve|rigid/small|rigid/nve/small bodystyle ... —
 * docs.lammps.org/fix_rigid.html: "Treat one or more sets of atoms as
 * independent rigid bodies." "bodystyle = single or molecule or group";
 * "For bodystyle single the entire fix group of atoms is treated as one
 * rigid body"; "For bodystyle molecule, atoms are grouped into rigid bodies
 * by their respective molecule IDs"; group "N groupID1 groupID2 ...".
 * "the image flags for each atom in the body are used to "unwrap" the atom
 * coordinates". "Each rigid body must have two or more atoms."
 * "The aggregate properties of each rigid body are calculated at the start
 * of a simulation run": mass, center of mass, principal moments and axes of
 * inertia, angular momentum; "Positions, velocities, and orientations of the
 * constituent particles are regenerated from the rigid body data structures
 * in every time step."
 * "Components of the external center-of-mass force and torque can be turned
 * off by the force and torque keywords" (M = 1..Nbody with * wildcards).
 * "A 3d rigid body has 6 degrees of freedom (3 translational, 3
 * rotational), except for a collection of point particles lying on a
 * straight line, which has only 5"; they are removed "only if the
 * temperature group includes all the particles in a particular rigid body".
 * "The scalar is the current temperature of the collection of rigid
 * bodies"; "The number of columns is 15": COM xyz, COM velocity, force,
 * torque, COM image flags.
 *
 * Integration (velocity Verlet for the centers of mass and angular momenta):
 * half kicks of v_cm by F/M and of the space-frame angular momentum L by the
 * torque; the orientation quaternion is advanced over dt with Richardson
 * extrapolation (one full step against two half steps of dq/dt = 1/2 w q,
 * with w recomputed from L at the midpoint). rigid/nve names the method of
 * Miller et al.; on the oracle cases native rigid and rigid/nve agree to
 * 1e-12, and this integrator matches both.
 */

interface Body {
  atoms: number[];          // atom IDs
  mass: number;
  xcm: number[];            // unwrapped center of mass
  vcm: number[];
  image: number[];          // COM image flags (xcm wrapped into the box + image * L)
  inertia: number[];        // principal moments
  q: number[];              // quaternion (w, x, y, z): body -> space
  angmom: number[];         // space frame
  omega: number[];
  displace: number[][];     // body-frame coordinates of the atoms
  fcm: number[];
  torque: number[];
  fflag: boolean[];
  tflag: boolean[];
  dof: number;
}

/** One line of an infile (fix rigid infile): attributes that override the computed ones. */
interface InfileBody {
  mass: number;
  xcm: number[];
  /** box-frame inertia tensor (ixx iyy izz ixy ixz iyz), as a symmetric 3x3 */
  I: number[][];
  vcm: number[];
  lam: number[];
  image: number[];
}

const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const qnorm = (q: number[]) => { const n = Math.hypot(q[0], q[1], q[2], q[3]); return q.map((v) => v / n); };
/** Rotation matrix (columns = body axes in the space frame) of a unit quaternion. */
const qmat = (q: number[]): number[][] => {
  const [w, x, y, z] = q;
  return [
    [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
  ];
};
/** Quaternion from a proper rotation matrix (columns = axes). */
const matq = (R: number[][]): number[] => {
  const tr = R[0][0] + R[1][1] + R[2][2];
  let q: number[];
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    q = [0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s];
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    const s = 2 * Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]);
    q = [(R[2][1] - R[1][2]) / s, 0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s];
  } else if (R[1][1] > R[2][2]) {
    const s = 2 * Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]);
    q = [(R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s];
  } else {
    const s = 2 * Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]);
    q = [(R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s];
  }
  return qnorm(q);
};
/** Symmetric 3x3 eigen-decomposition by cyclic Jacobi rotations: values and column eigenvectors. */
const jacobi = (A: number[][]): { values: number[]; vectors: number[][] } => {
  const a = A.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 100; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-300) break;
    for (const [p, r] of [[0, 1], [0, 2], [1, 2]]) {
      if (a[p][r] === 0) continue;
      const theta = (a[r][r] - a[p][p]) / (2 * a[p][r]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akr = a[k][r];
        a[k][p] = c * akp - s * akr; a[k][r] = s * akp + c * akr;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], ark = a[r][k];
        a[p][k] = c * apk - s * ark; a[r][k] = s * apk + c * ark;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkr = v[k][r];
        v[k][p] = c * vkp - s * vkr; v[k][r] = s * vkp + c * vkr;
      }
    }
  }
  return { values: [a[0][0], a[1][1], a[2][2]], vectors: v };
};

/**
 * Principal moments and the body -> space quaternion of a box-frame inertia tensor.
 * docs.lammps.org/fix_rigid.html: "The values are with respect to the simulation box XYZ axes,
 * not with respect to the principal axes of the rigid body itself. LAMMPS performs the latter
 * calculation internally."
 */
const principalFrame = (I: number[][]): { inertia: number[]; q: number[] } => {
  const { values, vectors } = jacobi(I);
  const maxI = Math.max(values[0], values[1], values[2]);
  const inertia = values.map((v) => (v < 1e-7 * maxI ? 0 : v));
  // right-handed axes
  const e = [0, 1, 2].map((c) => [vectors[0][c], vectors[1][c], vectors[2][c]]);
  const e3 = cross(e[0], e[1]);
  if (e3[0] * e[2][0] + e3[1] * e[2][1] + e3[2] * e[2][2] < 0) e[2] = e[2].map((x) => -x);
  const R = [[e[0][0], e[1][0], e[2][0]], [e[0][1], e[1][1], e[2][1]], [e[0][2], e[1][2], e[2][2]]];
  return { inertia, q: matq(R) };
};

/**
 * Parses an infile: "The file can contain initial blank lines or comment lines starting with "#"
 * which are ignored. The first non-blank, non-comment line should list N = the number of lines
 * to follow." Each line: "ID1 masstotal xcm ycm zcm ixx iyy izz ixy ixz iyz vxcm vycm vzcm lx ly lz ixcm iycm izcm".
 * Measured with native LAMMPS (black box): a blank line between the count line and the body lines is an
 * error, so only the lines before the count are skipped.
 */
const parseInfileText = (text: string, style: string, fname: string): Map<string, InfileBody> => {
  const where = `fix ${style} infile ${fname}`;
  // only initial blank and comment lines are skipped; the N body lines follow the count line directly
  const all = text.split('\n').map((l) => l.trim());
  let start = 0;
  while (start < all.length && (all[start] === '' || all[start].startsWith('#'))) start++;
  if (start >= all.length) throw new StyleError(`${where}: file has no count line`);
  const n = Number(all[start]);
  if (!Number.isInteger(n) || n < 0) throw new StyleError(`${where}: first line must be the number of bodies N, got '${all[start]}'`);
  // the N body lines must follow the count line without a blank line in between
  let m = 0;
  while (m < n && start + 1 + m < all.length && all[start + 1 + m] !== '') m++;
  if (m < n) throw new StyleError(`${where}: expected ${n} body lines, found ${m}`);
  const lines = all.slice(start + 1, start + 1 + n);
  const out = new Map<string, InfileBody>();
  for (let j = 0; j < n; j++) {
    const w = lines[j].split(/\s+/);
    if (w.length !== 20) throw new StyleError(`${where}: body line ${j + 1} needs 20 values (ID masstotal xcm ycm zcm ixx iyy izz ixy ixz iyz vxcm vycm vzcm lx ly lz ixcm iycm izcm), got ${w.length}`);
    const v = w.map(Number);
    if (v.some((x) => !Number.isFinite(x))) throw new StyleError(`${where}: body line ${j + 1} has a non-numeric value: '${lines[j]}'`);
    if (!Number.isInteger(v[0]) || v[0] < 1) throw new StyleError(`${where}: body ID on line ${j + 1} must be a positive integer, got ${w[0]}`);
    if (!(v[1] > 0)) throw new StyleError(`${where}: masstotal of body ${v[0]} must be positive`);
    const key = String(v[0]);
    if (out.has(key)) throw new StyleError(`${where}: body ID ${v[0]} listed twice`);
    out.set(key, {
      mass: v[1],
      xcm: v.slice(2, 5),
      I: [[v[5], v[8], v[9]], [v[8], v[6], v[10]], [v[9], v[10], v[7]]],
      vcm: v.slice(11, 14),
      lam: v.slice(14, 17),
      image: v.slice(17, 20),
    });
  }
  return out;
};

export class FixRigid extends Fix {
  readonly style: string;
  private bodies: Body[] = [];
  private bodystyle: string;
  private groupList: string[] = [];
  private forceSpecs: [string, boolean[]][] = [];
  private torqueSpecs: [string, boolean[]][] = [];
  private reinit = true;
  private built = false;
  /** bodystyle custom: name of the atom-style variable (v_name) that gives each atom's body ID. */
  private customVar: string | null = null;
  /** keyword mol: molecule template-ID whose molecules may be added during the run (fix deposit rigid). */
  private molTemplateId: string | null = null;
  /** keyword infile: per-body attributes keyed by body ID (see parseInfile). */
  private infile: Map<string, InfileBody> | null = null;
  /** fix_modify bodyforces early: forces and torques are summed in post_force, not final_integrate. */
  private early = false;
  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args);
    this.style = style;
    this.timeIntegrate = true;
    this.virialGlobal = true;
    this.thermoVirial = true;
    this.scalarFlag = true;
    this.extscalar = 0;
    this.arrayFlag = true;
    this.sizeArrayCols = 15;
    const small = style.includes('small');
    this.bodystyle = args[0];
    let k = 1;
    if (this.bodystyle === 'single') {
      if (small) throw new StyleError(`fix ${style}: bodystyle single is only allowed for the rigid styles, not rigid/small`);
    } else if (this.bodystyle === 'molecule') {
      /* ok */
    } else if (this.bodystyle === 'group') {
      if (small) throw new StyleError(`fix ${style}: bodystyle group is only allowed for the rigid styles`);
      const n = Number(args[1]);
      if (!Number.isInteger(n) || n < 1) throw new StyleError(`fix ${style} group: N must be a positive integer`);
      this.groupList = args.slice(2, 2 + n);
      if (this.groupList.length !== n) throw new StyleError(`fix ${style} group: expected ${n} group IDs`);
      for (const gname of this.groupList) sys.groups.bit(gname);
      k = 2 + n;
    } else if (this.bodystyle === 'custom') {
      /* docs.lammps.org/fix_rigid.html: "*custom* args = *i_propname* or *v_varname*" and
       * "v_varname = an atom-style or atomfile-style variable"; "the floating-point value
       * produced by the variable is rounded to an integer". */
      const w = args[1] ?? '';
      if (w.startsWith('i_')) {
        throw new StyleError(`fix ${style} custom ${w}: integer per-atom properties (i_name) need fix property/atom, which the browser engine does not have; use v_name with an atom-style variable`);
      }
      if (!w.startsWith('v_') || w.length < 3) throw new StyleError(`fix ${style} custom: expected v_name (or i_name), got '${w}'`);
      const vname = w.slice(2);
      const v = sys.vars.get(vname);
      if (!v) throw new StyleError(`fix ${style} custom: variable ${vname} does not exist`);
      if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`fix ${style} custom: variable ${vname} must be atom-style or atomfile-style, not ${v.style}`);
      this.customVar = vname;
      k = 2;
    } else throw new StyleError(`fix ${style}: unknown bodystyle '${this.bodystyle ?? ''}' (single, molecule or group)`);
    const onoff = (w: string | undefined) => {
      if (w !== 'on' && w !== 'off') throw new StyleError(`fix ${style}: force/torque flags must be on or off`);
      return w === 'on';
    };
    for (; k < args.length;) {
      const key = args[k];
      if (key === 'force' || key === 'torque') {
        if (small) throw new StyleError(`fix ${style}: "The force and torque keywords discussed next are only allowed for the rigid styles."`);
        const spec: [string, boolean[]] = [args[k + 1], [onoff(args[k + 2]), onoff(args[k + 3]), onoff(args[k + 4])]];
        (key === 'force' ? this.forceSpecs : this.torqueSpecs).push(spec);
        k += 5;
      } else if (key === 'reinit') {
        if (args[k + 1] !== 'yes' && args[k + 1] !== 'no') throw new StyleError(`fix ${style}: reinit must be yes or no`);
        this.reinit = args[k + 1] === 'yes';
        k += 2;
      } else if (key === 'infile') {
        const fname = args[k + 1];
        if (!fname) throw new StyleError(`fix ${style}: keyword infile needs a filename`);
        this.infile = parseInfileText(sys.readFile(fname), style, fname);
        // docs.lammps.org/fix_rigid.html: "When using the *infile* keyword, the *reinit* option is automatically set to *no*\ ."
        this.reinit = false;
        k += 2;
      } else if (key === 'mol') {
        // docs.lammps.org/fix_rigid.html: "The *mol* keyword can only be used with the *rigid/small* styles."
        // Measured with native LAMMPS (black box): fix rigid molecule mol <template> stops with Illegal fix rigid command.
        if (!small) throw new StyleError(`fix ${style}: the mol keyword can only be used with the rigid/small styles (Illegal fix ${style} command)`);
        const t = args[k + 1];
        if (!t) throw new StyleError(`fix ${style}: keyword mol needs a molecule template-ID`);
        this.molTemplateId = t;
        k += 2;
      } else {
        throw new StyleError(`fix ${style}: keyword '${key}' is not supported by the browser engine (supported: force, torque, reinit, infile, mol)`);
      }
    }
    if (this.molTemplateId && this.bodystyle !== 'molecule') {
      throw new StyleError(`fix ${style}: the mol keyword requires bodystyle molecule`);
    }
  }

  // ---------------------------------------------------------------- setup

  private index(): Map<number, number> {
    const s = this.sys.state;
    const m = new Map<number, number>();
    for (let i = 0; i < s.n; i++) m.set(s.id[i], i);
    return m;
  }

  private buildBodies(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const members = new Map<string, number[]>();
    // bodystyle custom: the atom-style variable's value, rounded to an integer (per-atom, all atoms)
    const custom = this.customVar ? sys.atomVariable(this.customVar) : null;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      let key: string | null = null;
      if (this.bodystyle === 'single') key = '1';
      else if (this.bodystyle === 'molecule') key = String(s.molecule[i]);
      else if (custom) key = String(Math.round(custom[i]));
      else {
        const k = this.groupList.findIndex((gname) => (s.mask[i] & sys.groups.bit(gname)) !== 0);
        if (k >= 0) key = String(k + 1);
      }
      if (key === null) continue;
      if (!members.has(key)) members.set(key, []);
      members.get(key)!.push(i);
    }
    // body order: molecule IDs ascending, group order, single
    const keys = [...members.keys()].sort((a, b) => Number(a) - Number(b));
    const bodies: Body[] = [];
    // infile body IDs: the molecule/group/single ID, or for bodystyle custom the value minus the
    // smallest body value plus 1. Measured with native LAMMPS (black box): custom values 0,1,3,4 take
    // infile IDs 1,2,4,5, and infile ID 3 (no body with value 2) is an error.
    const infileId = (key: string) => (this.customVar ? String(Number(key) - Number(keys[0]) + 1) : key);
    for (const key of keys) bodies.push(this.constructBody(key, members.get(key)!, infileId(key)));
    if (this.infile) {
      const valid = new Set(keys.map(infileId));
      for (const key of this.infile.keys()) {
        if (!valid.has(key)) throw new StyleError(`fix ${this.style}: infile body ID ${key} is not a rigid body (bodystyle ${this.bodystyle})`);
      }
    }
    const nb = bodies.length;
    for (const [spec, flags] of this.forceSpecs) { const [lo, hi] = typeBounds(spec, nb); for (let k = lo; k <= hi; k++) bodies[k - 1].fflag = flags.slice(); }
    for (const [spec, flags] of this.torqueSpecs) { const [lo, hi] = typeBounds(spec, nb); for (let k = lo; k <= hi; k++) bodies[k - 1].tflag = flags.slice(); }
    this.bodies = bodies;
    this.sizeArrayRows = nb;
    for (const b of bodies) this.omegaFromAngmom(b);
    sys.log(`fix ${this.id} ${this.style}: ${nb} rigid bodies with ${bodies.reduce((a, b) => a + b.atoms.length, 0)} atoms`);
  }

  /** Builds one rigid body from the current state of the atoms at indices `idx` (see buildBodies). */
  private constructBody(key: string, idx: number[], infileKey: string): Body {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const u = [0, 0, 0];
    if (idx.length < 2) throw new StyleError(`fix ${this.style}: "Each rigid body must have two or more atoms." (body ${key})`);
    for (const i of idx) for (let d = 0; d < 3; d++) if (!s.box.periodic[d] && s.image[3 * i + d] !== 0) throw new StyleError(`fix ${this.style}: an atom of body ${key} has a non-zero image flag in a non-periodic dimension`);
    let M = 0;
    const xcm = [0, 0, 0], vcm = [0, 0, 0];
    const pos = idx.map((i) => { g.unwrap(s.x, s.image, i, u); return [u[0], u[1], u[2]]; });
    idx.forEach((i, n) => {
      const m = massOf(s, i);
      M += m;
      for (let d = 0; d < 3; d++) { xcm[d] += m * pos[n][d]; vcm[d] += m * s.v[3 * i + d]; }
    });
    for (let d = 0; d < 3; d++) { xcm[d] /= M; vcm[d] /= M; }
    const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const angmom = [0, 0, 0];
    idx.forEach((i, n) => {
      const m = massOf(s, i);
      const r = [pos[n][0] - xcm[0], pos[n][1] - xcm[1], pos[n][2] - xcm[2]];
      const r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) I[a][b] += m * ((a === b ? r2 : 0) - r[a] * r[b]);
      const L = cross(r, [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]]);
      for (let d = 0; d < 3; d++) angmom[d] += m * L[d];
    });
    // infile attributes replace the computed ones; the atoms keep their positions relative to the body
    const over = this.infile?.get(infileKey) ?? null;
    const L = (d: number) => s.box.hi[d] - s.box.lo[d];
    const xcmB = over ? over.xcm.map((c, d) => c + (s.box.periodic[d] ? over.image[d] * L(d) : 0)) : xcm;
    const frame = principalFrame(over ? over.I : I);
    const Rq = qmat(frame.q);
    const displace = pos.map((p) => {
      const r = [p[0] - xcmB[0], p[1] - xcmB[1], p[2] - xcmB[2]];
      return [0, 1, 2].map((c) => Rq[0][c] * r[0] + Rq[1][c] * r[1] + Rq[2][c] * r[2]);
    });
    const inertia = frame.inertia;
    const ncollinear = inertia.filter((v) => v === 0).length;
    const b: Body = {
      atoms: idx.map((i) => s.id[i]),
      mass: over ? over.mass : M,
      xcm: xcmB,
      vcm: over ? over.vcm.slice() : vcm,
      image: [0, 0, 0],
      inertia,
      q: frame.q,
      angmom: over ? over.lam.slice() : angmom,
      omega: [0, 0, 0],
      displace,
      fcm: [0, 0, 0], torque: [0, 0, 0], fflag: [true, true, true], tflag: [true, true, true],
      dof: sys.dimension === 2 ? 3 : ncollinear >= 1 ? 5 : 6,
    };
    this.wrapCom(b);
    return b;
  }

  /**
   * Adds a rigid body for the molecule given by its new atom IDs (fix deposit with the rigid keyword).
   * docs.lammps.org/fix_rigid.html: "It must be used when other commands, such as fix deposit or fix pour,
   * add rigid bodies on-the-fly during a simulation." The body is built from the atoms as deposited, so it
   * keeps the deposited positions and velocities.
   */
  addMolecule(atomIds: readonly number[]): void {
    if (this.bodystyle !== 'molecule') throw new StyleError(`fix ${this.style}: the mol keyword needs bodystyle molecule`);
    const idx = this.index();
    const ii: number[] = [];
    for (const idNum of atomIds) {
      const k = idx.get(idNum);
      if (k === undefined) throw new StyleError(`fix ${this.style}: atom ${idNum} does not exist`);
      ii.push(k);
    }
    const key = String(this.sys.state.molecule[ii[0]]);
    const b = this.constructBody(key, ii, key);
    this.bodies.push(b);
    this.sizeArrayRows = this.bodies.length;
    this.omegaFromAngmom(b);
  }

  /** COM image flags: xcm stays the unwrapped position; image counts box periods. */
  private wrapCom(b: Body): void {
    const box = this.sys.state.box;
    for (let d = 0; d < 3; d++) {
      const L = box.hi[d] - box.lo[d];
      b.image[d] = box.periodic[d] ? Math.floor((b.xcm[d] - box.lo[d]) / L) : 0;
    }
  }

  private omegaFromAngmom(b: Body): void {
    const R = qmat(b.q);
    const w = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      if (b.inertia[k] === 0) continue;
      const ek = [R[0][k], R[1][k], R[2][k]];
      const lk = (b.angmom[0] * ek[0] + b.angmom[1] * ek[1] + b.angmom[2] * ek[2]) / b.inertia[k];
      for (let d = 0; d < 3; d++) w[d] += lk * ek[d];
    }
    b.omega = w;
  }

  init(): void {
    if (this.molTemplateId && !this.sys.molecules.has(this.molTemplateId)) {
      throw new StyleError(`fix ${this.style}: mol molecule template '${this.molTemplateId}' does not exist`);
    }
    if (!this.built || this.reinit) { this.buildBodies(); this.built = true; }
  }

  dofRemoved(groupBit: number): number {
    const s = this.sys.state;
    const idx = this.index();
    let n = 0;
    for (const b of this.bodies) {
      if (!b.atoms.every((id) => (s.mask[idx.get(id)!] & groupBit) !== 0)) continue;
      n += 3 * b.atoms.length - b.dof;
    }
    return n;
  }

  // ---------------------------------------------------------------- integration

  private sumForces(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const idx = this.index();
    const u = [0, 0, 0];
    for (const b of this.bodies) {
      const F = [0, 0, 0], T = [0, 0, 0];
      for (const id of b.atoms) {
        const i = idx.get(id)!;
        const f = [s.f[3 * i], s.f[3 * i + 1], s.f[3 * i + 2]];
        g.unwrap(s.x, s.image, i, u);
        const r = [u[0] - b.xcm[0], u[1] - b.xcm[1], u[2] - b.xcm[2]];
        const t = cross(r, f);
        for (let d = 0; d < 3; d++) { F[d] += f[d]; T[d] += t[d]; }
      }
      // stored unflagged (the global array reports them so, measured); the flags act on the kicks
      for (let d = 0; d < 3; d++) { b.fcm[d] = F[d]; b.torque[d] = T[d]; }
    }
  }

  /**
   * Atom positions (with image flags) and velocities from the body state.
   * Virial (measured with native LAMMPS, oracle rigid_molecule): each
   * regeneration adds factor * sum x_u (x) fc with the constraint force
   * fc = m (v_new - v_old) / (dt/2) - f, x_u the unwrapped position before
   * the update (xy = x fc_y, xz = x fc_z, yz = y fc_z); factor 1 at setup
   * (where v_old are the atoms' own velocities) and 1/2 in each of
   * initial_integrate and final_integrate. This reproduces native pressure
   * components to ~1e-9 absolute (the scalar pressure to 1e-12).
   */
  private setXV(setX: boolean, virialFactor: number): void {
    const s = this.sys.state;
    const box = s.box;
    const g = this.sys.geom;
    const idx = this.index();
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const vir = this.virial;
    const u = [0, 0, 0];
    for (const b of this.bodies) {
      const R = qmat(b.q);
      b.atoms.forEach((id, n) => {
        const i = idx.get(id)!;
        const dsp = b.displace[n];
        const r = [0, 1, 2].map((d) => R[d][0] * dsp[0] + R[d][1] * dsp[1] + R[d][2] * dsp[2]);
        const w = cross(b.omega, r);
        const vold = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
        const xold = [0, 0, 0];
        g.unwrap(s.x, s.image, i, xold);
        for (let d = 0; d < 3; d++) s.v[3 * i + d] = b.vcm[d] + w[d];
        if (setX) {
          // unwrapped position, written back as wrapped position + image flags
          for (let d = 0; d < 3; d++) {
            const xu = b.xcm[d] + r[d];
            const L = box.hi[d] - box.lo[d];
            const im = box.periodic[d] ? Math.floor((xu - box.lo[d]) / L) : 0;
            s.x[3 * i + d] = xu - im * L;
            s.image[3 * i + d] = im;
          }
        }
        if (virialFactor) {
          const m = massOf(s, i);
          if (setX) { u[0] = xold[0]; u[1] = xold[1]; u[2] = xold[2]; } else g.unwrap(s.x, s.image, i, u);
          const fc = [0, 1, 2].map((d) => (m * (s.v[3 * i + d] - vold[d])) / dtf - s.f[3 * i + d]);
          vir[0] += virialFactor * u[0] * fc[0]; vir[1] += virialFactor * u[1] * fc[1]; vir[2] += virialFactor * u[2] * fc[2];
          vir[3] += virialFactor * u[0] * fc[1]; vir[4] += virialFactor * u[0] * fc[2]; vir[5] += virialFactor * u[1] * fc[2];
        }
      });
    }
  }

  setup(): void {
    this.sumForces();
    for (const b of this.bodies) this.omegaFromAngmom(b);
    // atom velocities become those of the rigid motion
    this.virial.fill(0);
    this.setXV(false, 1);
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const dtv = s.dt;
    for (const b of this.bodies) {
      for (let d = 0; d < 3; d++) {
        if (b.fflag[d]) b.vcm[d] += (dtf * b.fcm[d]) / b.mass;
        b.xcm[d] += dtv * b.vcm[d];
        if (b.tflag[d]) b.angmom[d] += dtf * b.torque[d];
      }
      this.richardson(b, dtv);
      this.wrapCom(b);
    }
    this.virial.fill(0);
    this.setXV(true, 0.5);
  }

  /** fix_modify bodyforces early|late (fix_modify.rst: "early/late = compute rigid-body forces/torques early or late in the timestep"). */
  modify(key: string, values: string[]): number {
    if (key === 'bodyforces') {
      if (values[0] !== 'early' && values[0] !== 'late') throw new StyleError(`fix_modify bodyforces must be early or late, got '${values[0] ?? ''}'`);
      this.early = values[0] === 'early';
      return 1;
    }
    return super.modify(key, values);
  }

  /** Early bodyforces: the forces are summed right after the per-atom forces (before later fixes' post_force). */
  postForce(): void {
    if (this.early) this.sumForces();
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    if (!this.early) this.sumForces();
    for (const b of this.bodies) {
      for (let d = 0; d < 3; d++) {
        if (b.fflag[d]) b.vcm[d] += (dtf * b.fcm[d]) / b.mass;
        if (b.tflag[d]) b.angmom[d] += dtf * b.torque[d];
      }
      this.omegaFromAngmom(b);
    }
    this.setXV(false, 0.5);
  }

  /** Orientation over dt with Richardson extrapolation of dq/dt = 1/2 (0, w) q (w from L, space frame). */
  private richardson(b: Body, dt: number): void {
    const wq = (q: number[], w: number[]) => [
      -w[0] * q[1] - w[1] * q[2] - w[2] * q[3],
      w[0] * q[0] + w[1] * q[3] - w[2] * q[2],
      w[1] * q[0] + w[2] * q[1] - w[0] * q[3],
      w[2] * q[0] + w[0] * q[2] - w[1] * q[1],
    ];
    const dtq = 0.5 * dt;
    this.omegaFromAngmom(b);
    const q0 = b.q;
    const d0 = wq(q0, b.omega);
    const qfull = qnorm(q0.map((v, k) => v + dtq * d0[k]));
    const qhalf = qnorm(q0.map((v, k) => v + 0.5 * dtq * d0[k]));
    b.q = qhalf;
    this.omegaFromAngmom(b);
    const d1 = wq(qhalf, b.omega);
    const qhalf2 = qnorm(qhalf.map((v, k) => v + 0.5 * dtq * d1[k]));
    b.q = qnorm(qhalf2.map((v, k) => 2 * v - qfull[k]));
    this.omegaFromAngmom(b);
  }

  // ---------------------------------------------------------------- output

  /**
   * Temperature of the rigid bodies: (sum m v_cm^2 + sum omega . L) / (dof kB),
   * with 3 + 3 degrees of freedom per body minus the components switched off
   * by force / torque (space-frame x, y, z). Measured: native LAMMPS counts 6
   * per body here even for linear bodies (153 vs 162 for 9 dimers among 27
   * bodies), unlike the 5 it removes from temperature computes.
   * With force / torque flags off, native LAMMPS's value differs from every
   * form tried (0.393266 vs 0.393379 here, oracle rigid_force_torque); the
   * engine removes the flagged space-frame components of omega and takes
   * the body-frame energy sum I_k w_k^2. The dynamics are not affected.
   */
  computeScalar(): number {
    const s = this.sys.state;
    let ke = 0, dof = 0;
    for (const b of this.bodies) {
      for (let d = 0; d < 3; d++) {
        if (b.fflag[d]) { ke += b.mass * b.vcm[d] * b.vcm[d]; dof++; }
        if (b.tflag[d]) dof++;
      }
      // flagged space-frame components of omega removed, energy in the body frame
      const R = qmat(b.q);
      const w = b.omega.map((v, d) => (b.tflag[d] ? v : 0));
      for (let k = 0; k < 3; k++) {
        const wk = w[0] * R[0][k] + w[1] * R[1][k] + w[2] * R[2][k];
        ke += b.inertia[k] * wk * wk;
      }
    }
    return dof > 0 ? (ke * s.units.mvv2e) / (dof * s.units.boltz) : 0;
  }

  computeArray(i: number, j: number): number {
    const b = this.bodies[i];
    if (!b) throw new StyleError(`fix ${this.id}: rigid body ${i + 1} does not exist`);
    const box = this.sys.state.box;
    if (j < 3) return b.xcm[j] - b.image[j] * (box.hi[j] - box.lo[j]);
    if (j < 6) return b.vcm[j - 3];
    if (j < 9) return b.fcm[j - 6];
    if (j < 12) return b.torque[j - 9];
    return b.image[j - 12];
  }
}
