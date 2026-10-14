/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/GRANSURF/fix_surface_global.cpp (FixSurfaceGlobal: the constructor's input and
 * model keywords, init, initial_integrate, pre_neighbor, post_force, modify_param move, move_rotate and
 * move_rotate_point, surface_attributes, connectivity2d_complete), src/GRANSURF/fix_surface.cpp (extract_from_molecule,
 * extract_from_stlfile, connectivity2d_global, point_connection2d, edge_connection3d, corner_connection3d,
 * contact_compare), src/GRANSURF/fix_surface.h. Contributing authors of the GRANSURF package: Joel Clemmer (SNL),
 * Dan Bolintineanu (SNL). Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_surface_global.html: "The fix surface/global command ... creates a set of surface elements ...
 * that interact with granular particles ... The surface elements are not atoms: they are stored by the fix."
 *
 * What is supported (everything else is a StyleError naming it):
 *   - input mol (Lines in 2d, Triangles in 3d) and input stl (3d, ASCII; degenerate triangles are skipped, as native);
 *   - model * * granular with hooke, tangential linear_history and damping mass_velocity (one model for all types);
 *   - neighbor bin (the default); the flat keyword is not supported;
 *   - fix_modify move with style rotate (the surfaces of one or more molecule IDs, moved about a fixed axis);
 *     linear, wiggle, transrot and variable motions are not supported;
 *   - no temperature / heat keywords; no periodic boundaries in the surface plane (as pair surf/granular).
 *
 * Native details reproduced here: points are identified by their coordinates and molecule-ID (extract_from_molecule),
 * so connectivity is exact; the contact list holds the surfaces within the cutoff of radius + surface radius + skin
 * and inside the normal projection, built at neighbour rebuilds (pre_neighbor), and a rebuild is also forced when a
 * surface point has moved by more than half the skin since the last build (initial_integrate); the contact history is
 * kept per (sphere ID, surface index) for the pairs of the list, reset when they stop touching, and dropped at a rebuild
 * when a pair is no longer listed (the fix neigh/history transfer). Forces act on the spheres only.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { parseStl } from '../../engine/stl';
import { stlFirstSolid } from './stl_solid';
import { type Connect2d, type Neighbour2d } from './connect2d';
import { cornerConnection3d, edgeConnection3d, type Connect3d } from './connect3d';
import { parseSurfCoeff } from './pair_surf_granular';
import {
  EXTERNAL, INTERNAL, OPPOSITE_SIDE, SAME_SIDE, UNCONNECTED, overlapSphereLine, overlapSphereTri, pointConnection2d,
  surfContactForces, type ContactSurf, type SurfCoeffs,
} from './surf_contact';
import { contactDr, solveSphereContacts } from './contact_flow';

type V3 = [number, number, number];

/** fix_surface_global.cpp: FLATTHRESH (the default flat threshold, 1 - cos(1 degree)). */
const FLATTHRESH = 0.00015230484360876085;

/** One rotation motion (fix_surface_global.h: Motion, style ROTATE). */
interface Motion {
  point: V3;
  unit: V3;
  omega: number;
  timeOrigin: number;
  active: boolean;
}

/** Native point key: coordinates and molecule-ID (extract_from_molecule, extract_from_stlfile). */
const pointKey = (p: ArrayLike<number>, mol: number): string => `${p[0]},${p[1]},${p[2]},${mol}`;

/** fix_surface_global.cpp: MathExtra::norm3 (the scale is 1/len, multiplied in). */
const norm3 = (v: V3): V3 => {
  const scale = 1.0 / Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return [v[0] * scale, v[1] * scale, v[2] * scale];
};

const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Parse a molecule-ID range list for fix_modify move (fix_surface_global.cpp: utils::bounds over 1..maxsurfmol). */
function molRange(spec: string, maxmol: number): number[] {
  const out: number[] = [];
  const bad = (): never => {
    throw new StyleError(`fix surface/global move: molecule-ID range '${spec}' is outside 1..${maxmol}`);
  };
  if (maxmol < 1) throw new StyleError('fix surface/global move: there are no surfaces with molecule IDs to move');
  for (const part of spec.split(',')) {
    let lo: number, hi: number;
    if (part.includes('*')) {
      const [a, b] = part.split('*');
      lo = a === '' ? 1 : Number(a);
      hi = b === '' ? maxmol : Number(b);
    } else {
      lo = hi = Number(part);
    }
    if (!Number.isInteger(lo) || !Number.isInteger(hi)) bad();
    if (lo < 1 || hi > maxmol || lo > hi) bad();
    for (let m = lo; m <= hi; m++) if (!out.includes(m)) out.push(m);
  }
  return out;
}

export class FixSurfaceGlobal extends Fix {
  readonly style: string = 'surface/global';
  /** Surface points (flat xyz) and their state when the motion was last set (points_original / points_lastneigh). */
  private pts: Float64Array;
  private ptsOrig: Float64Array;
  private ptsLast: Float64Array;
  /** Per surface: the point indices (2 for lines, 3 for triangles), molecule-ID, type. */
  private corners: number[][] = [];
  private mol: number[] = [];
  private stype: number[] = [];
  private dim: number;
  private maxsurfmol = 0;
  /** Per surface (flat arrays): centre, its original centre during a motion, radius, normal, velocity, rotation rate. */
  private xs: Float64Array;
  private xsOrig: Float64Array;
  private rad: Float64Array;
  private norm: Float64Array;
  private vs: Float64Array;
  private oms: Float64Array;
  private conn2: Map<number, Connect2d> | null = null;
  private conn3: Map<number, Connect3d> | null = null;
  private coeffs: SurfCoeffs;
  private flatthresh = FLATTHRESH;
  /** Motions and the motion of each molecule-ID (-1 or absent: static). */
  private motions: Motion[] = [];
  private mol2motion = new Map<number, number>();
  /** Neighbour list: sphere IDs and the surfaces of each sphere (rebuilt at neighbour builds). */
  private listIds: number[] = [];
  private listSurfs: number[][] = [];
  /** Contact history per "sphere ID:surface index" (three tangential values). */
  private hist = new Map<string, Float64Array>();
  /** Atom count when the lists were built: a change (fix pour, lost atoms) is a reneighbouring, as in native. */
  private listN = -1;
  private trigStep = -1;
  private triggersq = 0;
  private dt = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.dim = sys.dimension;
    if (!sys.state.radius || !sys.state.omega) throw new StyleError('Fix surface/global requires atom attributes radius and omega');
    const points: number[] = [];
    const hash = new Map<string, number>();
    const ptIndex = (p: number[], mol: number): number => {
      const key = pointKey(p, mol);
      const m = hash.get(key);
      if (m !== undefined) return m;
      const k = points.length / 3;
      points.push(p[0], p[1], p[2]);
      hash.set(key, k);
      return k;
    };
    const addSurface = (pidx: number[], mol: number, type: number): void => {
      this.corners.push(pidx);
      this.mol.push(mol);
      this.stype.push(type);
    };
    let k = 0;
    let ninput = 0;
    while (k < args.length && args[k] === 'input') {
      const src = args[k + 1];
      if (src === 'mol') {
        const name = args[k + 2];
        if (name === undefined) throw new StyleError('fix surface/global input mol needs a molecule template ID');
        const sets = sys.molecules.get(name);
        if (!sets) throw new StyleError(`Molecule template ID for fix surface does not exist: ${name}`);
        for (const t of sets) {
          const list = this.dim === 2 ? t.lines : t.tris;
          if (!list) throw new StyleError(this.dim === 2 ? 'Fix surface molecule must have lines' : 'Fix surface molecule must have triangles');
          for (const e of list) {
            const w = this.dim === 2 ? 2 : 3;
            const pidx: number[] = [];
            for (let c = 0; c < e.p.length; c += w) {
              const p = this.dim === 2 ? [e.p[c], e.p[c + 1], 0] : [e.p[c], e.p[c + 1], e.p[c + 2]];
              pidx.push(ptIndex(p, e.mol));
            }
            addSurface(pidx, e.mol, e.type);
          }
        }
        k += 3;
      } else if (src === 'stl') {
        if (this.dim !== 3) throw new StyleError('Fix surface cannot use an STL file for 2d simulations');
        const stype = Number(args[k + 2]), smol = Number(args[k + 3]);
        const file = args[k + 4];
        if (file === undefined || !Number.isInteger(stype) || !Number.isInteger(smol)) {
          throw new StyleError('fix surface/global input stl needs type, molecule-ID and file name');
        }
        if (stype < 1) throw new StyleError('STL surface type must be >= 1');
        for (const tri of parseStl(stlFirstSolid(sys.readFile(file)), file)) {
          const [a, b, c] = tri.v;
          const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
          const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
          const cr = cross(e1 as V3, e2 as V3);
          // native: a triangle whose edge cross product is exactly zero is skipped (a degenerate triangle)
          if (cr[0] === 0 && cr[1] === 0 && cr[2] === 0) continue;
          addSurface([ptIndex(a, smol), ptIndex(b, smol), ptIndex(c, smol)], smol, stype);
        }
        k += 5;
      } else {
        throw new StyleError(`Unknown fix surface/global input keyword: ${src ?? ''}`);
      }
      ninput++;
    }
    if (ninput === 0) throw new StyleError('Fix surface/global command requires using the input keyword');
    const ns = this.corners.length;
    if (ns === 0) throw new StyleError('fix surface/global defines no surface elements');
    this.pts = Float64Array.from(points);
    this.ptsOrig = new Float64Array(this.pts.length);
    this.ptsLast = new Float64Array(this.pts.length);
    this.xs = new Float64Array(3 * ns);
    this.xsOrig = new Float64Array(3 * ns);
    this.rad = new Float64Array(ns);
    this.norm = new Float64Array(3 * ns);
    this.vs = new Float64Array(3 * ns);
    this.oms = new Float64Array(3 * ns);
    for (const m of this.mol) if (m > this.maxsurfmol) this.maxsurfmol = m;
    this.surfaceAttributes();
    // model keyword: "model * * granular" with the sub-models of the granular pair (hooke, tangential linear_history,
    // damping mass_velocity); the other keywords of the native fix are refused
    if (args[k] !== 'model') throw new StyleError('Fix surface/global command requires model keyword');
    if (args[k + 1] !== '*' || args[k + 2] !== '*') throw new StyleError('fix surface/global model: only "* *" (all types) is supported by the browser engine');
    if (args[k + 3] !== 'granular') throw new StyleError('fix surface/global model: only the granular model syntax is supported (classic models are not)');
    k += 4;
    const sub: string[] = [];
    // the arguments of each sub-model keyword (the tangential count depends on the model)
    const argCount = (w: string, next: string | undefined): number | undefined => {
      if (w === 'hooke') return 3;
      if (w === 'damping') return 2;
      if (w === 'tangential') return next === 'linear_nohistory' ? 4 : 5;
      return undefined;
    };
    while (k < args.length && argCount(args[k], args[k + 1]) !== undefined) {
      const n = argCount(args[k], args[k + 1])!;
      if (k + n > args.length) throw new StyleError(`fix surface/global model: '${args[k]}' is missing arguments`);
      sub.push(...args.slice(k, k + n));
      k += n;
    }
    if (sub.length === 0) throw new StyleError('Fix surface/global command requires model keyword');
    this.coeffs = parseSurfCoeff(sub);
    if (this.coeffs.normal !== 'hooke' || this.coeffs.tangential !== 'linear_history' || this.coeffs.damping !== 'mass_velocity') {
      throw new StyleError('fix surface/global model: only hooke, tangential linear_history and damping mass_velocity are supported');
    }
    // keywords after the model
    while (k < args.length) {
      if (args[k] === 'neighbor' && args[k + 1] === 'bin') {
        k += 2;
        continue;
      }
      throw new StyleError(`fix surface/global keyword '${args[k]}' is not supported by the browser engine`);
    }
    this.buildConnectivity();
  }

  /** fix_surface.cpp connectivity2d_global / 3d_global with fix_surface_global.cpp connectivity2d_complete. */
  private buildConnectivity(): void {
    const ns = this.corners.length;
    if (this.dim === 2) {
      const out = new Map<number, Connect2d>();
      for (let a = 0; a < ns; a++) {
        const p: [Neighbour2d[], Neighbour2d[]] = [[], []];
        for (let which = 0; which < 2; which++) {
          const pt = this.corners[a][which];
          for (let b = 0; b < ns; b++) {
            if (b === a) continue;
            const jwhich = this.corners[b][0] === pt ? 0 : this.corners[b][1] === pt ? 1 : -1;
            if (jwhich < 0) continue;
            const c = pointConnection2d(this.normOf(a), this.normOf(b), which, jwhich, this.flatthresh);
            p[which].push({ tag: b, pwhich: jwhich, nside: c.nside, aflag: c.aflag, fflag: c.fflag });
          }
        }
        const external: [number, number] = [INTERNAL, INTERNAL];
        for (const nb of p[0]) if (nb.fflag !== 1) external[0] = EXTERNAL;
        for (const nb of p[1]) if (nb.fflag !== 1) external[1] = EXTERNAL;
        if (p[0].length === 0) external[0] = UNCONNECTED;
        if (p[1].length === 0) external[1] = UNCONNECTED;
        out.set(a, { p, external });
      }
      this.conn2 = out;
      return;
    }
    // triangles: edge neighbours share two corner points, corner neighbours share one (fix_surface.cpp)
    const out = new Map<number, Connect3d>();
    const corner = (a: number, k: number): V3 => [this.pts[3 * this.corners[a][k]], this.pts[3 * this.corners[a][k] + 1], this.pts[3 * this.corners[a][k] + 2]];
    const touching = (a: number, k: number): number[] => {
      const out2: number[] = [];
      for (let b = 0; b < ns; b++) {
        if (b === a) continue;
        if (this.corners[b].includes(this.corners[a][k])) out2.push(b);
      }
      return out2;
    };
    const match0 = (b: number, pt: number): number => this.corners[b].indexOf(pt);
    for (let a = 0; a < ns; a++) {
      const ca = this.corners[a];
      const N = [touching(a, 0), touching(a, 1), touching(a, 2)];
      const inSet = (s: number[], b: number): boolean => s.includes(b);
      const e1 = N[0].filter((b) => inSet(N[1], b));
      const e2 = N[1].filter((b) => inSet(N[2], b));
      const e3 = N[2].filter((b) => inSet(N[0], b));
      const c1 = N[0].filter((b) => !inSet(N[1], b) && !inSet(N[2], b));
      const c2 = N[1].filter((b) => !inSet(N[2], b) && !inSet(N[0], b));
      const c3 = N[2].filter((b) => !inSet(N[0], b) && !inSet(N[1], b));
      const pa = [corner(a, 0), corner(a, 1), corner(a, 2)];
      const iedge: [V3, V3, V3] = [
        [pa[1][0] - pa[0][0], pa[1][1] - pa[0][1], pa[1][2] - pa[0][2]],
        [pa[2][0] - pa[1][0], pa[2][1] - pa[1][1], pa[2][2] - pa[1][2]],
        [pa[0][0] - pa[2][0], pa[0][1] - pa[2][1], pa[0][2] - pa[2][2]],
      ];
      const edgeCorners: [number, number][] = [[0, 1], [1, 2], [2, 0]];
      const edgeLists = [e1, e2, e3];
      const e: [Connect3d['e'][0], Connect3d['e'][1], Connect3d['e'][2]] = [[], [], []];
      for (let s = 0; s < 3; s++) {
        const [ka, kb] = edgeCorners[s];
        for (const b of edgeLists[s]) {
          const jf = match0(b, ca[ka]) + 1, js = match0(b, ca[kb]) + 1;
          const c = edgeConnection3d(this.normOf(a), this.normOf(b), iedge[s], jf, js, this.flatthresh);
          e[s].push({ tag: b, nside: c.nside, aflag: c.aflag, fflag: c.fflag });
        }
      }
      const cornerLists = [c1, c2, c3];
      const cc: Connect3d['c'] = [[], [], []];
      for (let k = 0; k < 3; k++) {
        for (const b of cornerLists[k]) {
          const c = cornerConnection3d(this.normOf(a), this.normOf(b), match0(b, ca[k]), this.flatthresh);
          cc[k].push({ tag: b, nside: c.nside, fflag: c.fflag });
        }
      }
      const externalEdge: [number, number, number] = [INTERNAL, INTERNAL, INTERNAL];
      for (let s = 0; s < 3; s++) {
        for (const nb of e[s]) if (nb.fflag !== 1) externalEdge[s] = EXTERNAL;
        if (e[s].length === 0) externalEdge[s] = UNCONNECTED;
      }
      const externalCor: [number, number, number] = [
        Math.max(externalEdge[0], externalEdge[2]), Math.max(externalEdge[0], externalEdge[1]), Math.max(externalEdge[1], externalEdge[2]),
      ];
      out.set(a, { e, c: cc, externalEdge, externalCor });
    }
    this.conn3 = out;
  }

  /** surface_attributes: centre, radius and normal of every surface from the current points. */
  private surfaceAttributes(): void {
    const ns = this.corners.length;
    for (let i = 0; i < ns; i++) {
      const c = this.corners[i];
      const P = (k: number): V3 => [this.pts[3 * c[k]], this.pts[3 * c[k] + 1], this.pts[3 * c[k] + 2]];
      if (this.dim === 2) {
        const p1 = P(0), p2 = P(1);
        this.xs[3 * i] = 0.5 * (p1[0] + p2[0]);
        this.xs[3 * i + 1] = 0.5 * (p1[1] + p2[1]);
        this.xs[3 * i + 2] = 0.0;
        const p12: V3 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        this.rad[i] = 0.5 * Math.sqrt(p12[0] * p12[0] + p12[1] * p12[1] + p12[2] * p12[2]);
        const n = norm3(cross([0, 0, 1], p12));
        this.norm.set(n, 3 * i);
      } else {
        const p1 = P(0), p2 = P(1), p3 = P(2);
        this.xs[3 * i] = (p1[0] + p2[0] + p3[0]) / 3.0;
        this.xs[3 * i + 1] = (p1[1] + p2[1] + p3[1]) / 3.0;
        this.xs[3 * i + 2] = (p1[2] + p2[2] + p3[2]) / 3.0;
        const cen: V3 = [this.xs[3 * i], this.xs[3 * i + 1], this.xs[3 * i + 2]];
        let r2 = 0;
        for (const p of [p1, p2, p3]) {
          const d = [p[0] - cen[0], p[1] - cen[1], p[2] - cen[2]];
          r2 = Math.max(r2, d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }
        this.rad[i] = Math.sqrt(r2);
        const p12: V3 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        const p13: V3 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];
        this.norm.set(norm3(cross(p12, p13)), 3 * i);
      }
    }
  }

  /** The stored normal of surface i. */
  private normOf(i: number): V3 {
    return [this.norm[3 * i], this.norm[3 * i + 1], this.norm[3 * i + 2]];
  }

  private anymove(): boolean {
    return this.mol2motion.size > 0;
  }

  /** init: the cached time step and trigger, the connected-surface motion check (fix_surface_global.cpp: init). */
  init(): void {
    const s = this.sys.state;
    if (s.box.triclinic) throw new StyleError('Fix surface/global does not yet support triclinic simulation boxes');
    // native ghost surfaces for periodic images are not built here: the surfaces are not periodic (in the surface plane
    // only; z of a 2d box does not matter), as for pair surf/granular
    const dims = this.dim === 2 ? [0, 1] : [0, 1, 2];
    if (dims.some((d) => s.box.periodic[d])) {
      throw new StyleError('fix surface/global with periodic boundaries is not supported by the browser engine yet (use boundary f or s in the surface plane)');
    }
    if (this.dim === 2) {
      for (let i = 0; i < this.corners.length; i++) {
        const m = this.motionOf(i);
        for (const j of this.conn2!.get(i)!.p.flat()) if (this.motionOf(j.tag) !== m) throw new StyleError('Fix surface/global motions assigned to connected surf pairs are inconsistent');
      }
    } else {
      for (let i = 0; i < this.corners.length; i++) {
        const m = this.motionOf(i);
        const cn = this.conn3!.get(i)!;
        const all = [...cn.e.flat(), ...cn.c.flat()];
        for (const j of all) if (this.motionOf(j.tag) !== m) throw new StyleError('Fix surface/global motions assigned to connected surf pairs are inconsistent');
      }
    }
    this.dt = s.dt;
    const skin = this.sys.nb.skin;
    this.triggersq = 0.25 * skin * skin;
  }

  /** The motion index of surface i, -1 when it does not move. */
  private motionOf(i: number): number {
    return this.mol2motion.get(this.mol[i]) ?? -1;
  }

  /** fix_modify move MOLS none | MOLS rotate px py pz rx ry rz period (fix_surface_global.cpp: modify_param). */
  modify(key: string, values: string[]): number {
    if (key !== 'move') return super.modify(key, values);
    const spec = values[0];
    const style = values[1];
    if (spec === undefined || style === undefined) throw new StyleError('fix surface/global modify move needs molecule-IDs and a style');
    const mols = molRange(spec, this.maxsurfmol);
    const s = this.sys.state;
    if (style === 'none') {
      for (const m of mols) {
        const im = this.mol2motion.get(m);
        if (im === undefined) continue;
        this.motions[im].active = false;
        this.mol2motion.delete(m);
        for (let i = 0; i < this.corners.length; i++) {
          if (this.mol[i] !== m) continue;
          this.vs.fill(0, 3 * i, 3 * i + 3);
          this.oms.fill(0, 3 * i, 3 * i + 3);
        }
      }
      return 2;
    }
    if (style !== 'rotate') {
      throw new StyleError(`fix surface/global move style '${style}' is not supported by the browser engine (rotate only)`);
    }
    const a = values.slice(2, 9);
    if (a.length < 7) throw new StyleError('fix surface/global modify move rotate needs point, axis and period');
    const nums = a.map((w) => Number(w));
    if (nums.some((v) => !Number.isFinite(v))) throw new StyleError('fix surface/global modify move rotate: arguments must be numbers');
    const point: V3 = [nums[0], nums[1], nums[2]];
    const axis: V3 = [nums[3], nums[4], nums[5]];
    const period = nums[6];
    if (this.dim === 2 && (axis[0] !== 0 || axis[1] !== 0)) throw new StyleError('Fix_modify move cannot rotate around non z-axis for 2d problem');
    if (!(period > 0)) throw new StyleError(`Illegal fix_modify move period ${period}`);
    const len = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (len === 0) throw new StyleError('Fix_modify move zero length rotation vector');
    const scale = 1.0 / len;
    const unit: V3 = [axis[0] * scale, axis[1] * scale, axis[2] * scale];
    const motion: Motion = { point, unit, omega: (2 * Math.PI) / period, timeOrigin: s.step, active: true };
    // a new motion re-uses an inactive slot
    let im = this.motions.findIndex((m) => !m.active);
    if (im < 0) {
      im = this.motions.length;
      this.motions.push(motion);
    } else {
      this.motions[im] = motion;
    }
    for (const m of mols) this.mol2motion.set(m, im);
    // the surfaces of these molecule-IDs start from their current positions (native: points_original, xsurf_original)
    for (let i = 0; i < this.corners.length; i++) {
      if (!mols.includes(this.mol[i])) continue;
      this.xsOrig.set(this.xs.subarray(3 * i, 3 * i + 3), 3 * i);
      for (const p of this.corners[i]) {
        this.ptsOrig.set(this.pts.subarray(3 * p, 3 * p + 3), 3 * p);
        this.ptsLast.set(this.pts.subarray(3 * p, 3 * p + 3), 3 * p);
      }
      this.oms[3 * i] = motion.omega * unit[0];
      this.oms[3 * i + 1] = motion.omega * unit[1];
      this.oms[3 * i + 2] = motion.omega * unit[2];
    }
    return 2 + 7;
  }

  /**
   * setup_pre_neighbor only: fix_surface_global.cpp defines setup_pre_neighbor (the list) and post_force, but no setup(),
   * so the setup force evaluation of a run has no surface contact force (measured with native LAMMPS: the first
   * half-kick of a run after a fix_modify move uses the setup forces without the surface forces).
   */
  setup(): void {
    this.preNeighbor();
  }

  /** initial_integrate: move the surfaces of each motion (move_rotate) and ask for a rebuild when a point moved far. */
  initialIntegrate(): void {
    if (!this.anymove()) return;
    const s = this.sys.state;
    const dt = this.dt;
    for (let i = 0; i < this.corners.length; i++) {
      const im = this.motionOf(i);
      if (im < 0) continue;
      const m = this.motions[im];
      const delta = (s.step - m.timeOrigin) * dt;
      const arg = m.omega * delta;
      const cosine = Math.cos(arg), sine = Math.sin(arg);
      for (const p of this.corners[i]) this.rotatePoint(p, m, cosine, sine);
      // xsurf and vsurf: X = P + C + A cos + B sin about the rotation axis through P
      const disp = this.rotateVector(this.xsOrig.subarray(3 * i, 3 * i + 3), m, cosine, sine);
      const ru = m.unit, ww = m.omega;
      for (let a = 0; a < 3; a++) this.xs[3 * i + a] = m.point[a] + disp.c[a] + disp.d[a];
      this.vs[3 * i] = ww * (ru[1] * disp.d[2] - ru[2] * disp.d[1]);
      this.vs[3 * i + 1] = ww * (ru[2] * disp.d[0] - ru[0] * disp.d[2]);
      this.vs[3 * i + 2] = ww * (ru[0] * disp.d[1] - ru[1] * disp.d[0]);
      // normals from the moved points
      const P = (k: number): V3 => [this.pts[3 * k], this.pts[3 * k + 1], this.pts[3 * k + 2]];
      const c = this.corners[i];
      if (this.dim === 2) {
        const p12: V3 = [P(c[1])[0] - P(c[0])[0], P(c[1])[1] - P(c[0])[1], 0];
        this.norm.set(norm3(cross([0, 0, 1], p12)), 3 * i);
      } else {
        const p12: V3 = [P(c[0])[0] - P(c[1])[0], P(c[0])[1] - P(c[1])[1], P(c[0])[2] - P(c[1])[2]];
        const p13: V3 = [P(c[0])[0] - P(c[2])[0], P(c[0])[1] - P(c[2])[1], P(c[0])[2] - P(c[2])[2]];
        this.norm.set(norm3(cross(p12, p13)), 3 * i);
      }
    }
    // trigger a reneighbouring when a point has moved more than half the skin since the last build
    const np = this.pts.length / 3;
    for (let p = 0; p < np; p++) {
      const dx = this.pts[3 * p] - this.ptsLast[3 * p];
      const dy = this.pts[3 * p + 1] - this.ptsLast[3 * p + 1];
      const dz = this.pts[3 * p + 2] - this.ptsLast[3 * p + 2];
      if (dx * dx + dy * dy + dz * dz > this.triggersq) {
        this.trigStep = s.step;
        break;
      }
    }
  }

  reneighborDue(step: number): boolean {
    return this.trigStep === step;
  }

  /** move_rotate_point: point p from its original position (points_original) about the axis. */
  private rotatePoint(p: number, m: Motion, cosine: number, sine: number): void {
    const r = this.rotateVector(this.ptsOrig.subarray(3 * p, 3 * p + 3), m, cosine, sine);
    for (let a = 0; a < 3; a++) this.pts[3 * p + a] = m.point[a] + r.c[a] + r.d[a];
  }

  /** The projection c (on the axis) and the in-plane displacement d of a point q about the rotation axis. */
  private rotateVector(q: ArrayLike<number>, m: Motion, cosine: number, sine: number): { c: V3; d: V3 } {
    const ru = m.unit, rp = m.point;
    const dv: V3 = [q[0] - rp[0], q[1] - rp[1], q[2] - rp[2]];
    const ddotr = dv[0] * ru[0] + dv[1] * ru[1] + dv[2] * ru[2];
    const c: V3 = [ddotr * ru[0], ddotr * ru[1], ddotr * ru[2]];
    const a: V3 = [dv[0] - c[0], dv[1] - c[1], dv[2] - c[2]];
    const b: V3 = [ru[1] * a[2] - ru[2] * a[1], ru[2] * a[0] - ru[0] * a[2], ru[0] * a[1] - ru[1] * a[0]];
    const d: V3 = [a[0] * cosine + b[0] * sine, a[1] * cosine + b[1] * sine, a[2] * cosine + b[2] * sine];
    return { c, d };
  }

  /**
   * pre_neighbor: the list of spheres and the surfaces within reach (radius + surface radius + skin, and within the
   * skin of the surface plane); the history of pairs no longer listed is dropped.
   */
  preNeighbor(): void {
    const s = this.sys.state;
    this.listN = s.n;
    if (this.anymove()) this.ptsLast.set(this.pts);
    const skin = this.sys.nb.skin;
    const x = s.x, radius = s.radius!, id = s.id;
    const ns = this.corners.length;
    this.listIds = [];
    this.listSurfs = [];
    const keep = new Set<string>();
    for (let i = 0; i < s.n; i++) {
      const xi: V3 = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
      const radi = radius[i];
      const cutsqProj = (radi + skin) * (radi + skin);
      const surfs: number[] = [];
      for (let j = 0; j < ns; j++) {
        const del: V3 = [xi[0] - this.xs[3 * j], xi[1] - this.xs[3 * j + 1], xi[2] - this.xs[3 * j + 2]];
        const rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
        const radsum = radi + this.rad[j] + skin;
        if (rsq > radsum * radsum) continue;
        // the surfaces are flat: the distance along the normal must be within reach
        const n = this.normOf(j);
        const dot = n[0] * del[0] + n[1] * del[1] + n[2] * del[2];
        const proj: V3 = [dot * n[0], dot * n[1], dot * n[2]];
        if (proj[0] * proj[0] + proj[1] * proj[1] + proj[2] * proj[2] < cutsqProj) {
          surfs.push(j);
          keep.add(`${id[i]}:${j}`);
        }
      }
      this.listIds.push(id[i]);
      this.listSurfs.push(surfs);
    }
    for (const key of this.hist.keys()) if (!keep.has(key)) this.hist.delete(key);
  }

  /** post_force: the contacts of each sphere with the surfaces of its list; forces and torques on the spheres. */
  postForce(): void {
    const s = this.sys.state;
    // atoms inserted (fix pour, which the engine applies after the neighbour decision) or lost since the last build:
    // native reneighbours on such a step (pre_neighbor), so the surface list is rebuilt before the contacts are found
    if (s.n !== this.listN) this.preNeighbor();
    if (!this.listIds.length) return;
    const idx = new Map<number, number>();
    for (let i = 0; i < s.n; i++) idx.set(s.id[i], i);
    const x = s.x, v = s.v, omega = s.omega!, f = s.f, tq = s.torque!, radius = s.radius!, rmass = s.rmass!;
    const is3 = this.dim === 3;
    const dt = this.dt;
    const updating = true;
    for (let e = 0; e < this.listIds.length; e++) {
      const sid = this.listIds[e];
      const i = idx.get(sid);
      if (i === undefined || !this.inGroup(i)) continue;
      const xi: V3 = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
      const radi = radius[i];
      const cs: ContactSurf[] = [];
      for (const j of this.listSurfs[e]) {
        const key = `${sid}:${j}`;
        const del: V3 = [xi[0] - this.xs[3 * j], xi[1] - this.xs[3 * j + 1], xi[2] - this.xs[3 * j + 2]];
        const rsqCom = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
        const radsum = radi + this.rad[j];
        if (rsqCom > radsum * radsum) {
          this.hist.delete(key);
          continue;
        }
        const c = this.corners[j];
        let ov: ReturnType<typeof overlapSphereLine>;
        let jflag: number;
        let external = INTERNAL;
        let ends: [V3, V3] = [[0, 0, 0], [0, 0, 0]];
        let corners: [V3, V3, V3] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        const P = (k: number): V3 => [this.pts[3 * k], this.pts[3 * k + 1], this.pts[3 * k + 2]];
        const n = this.normOf(j);
        if (!is3) {
          ends = [P(c[0]), P(c[1])];
          ov = overlapSphereLine(x, 3 * i, radi, ends[0], ends[1]);
          if (!ov) {
            this.hist.delete(key);
            continue;
          }
          jflag = ov.flag;
          const cn = this.conn2!.get(j)!;
          external = jflag === -1 ? cn.external[0] : jflag === -2 ? cn.external[1] : INTERNAL;
        } else {
          corners = [P(c[0]), P(c[1]), P(c[2])];
          ov = overlapSphereTri(x, 3 * i, radi, corners[0], corners[1], corners[2], n);
          if (!ov) {
            this.hist.delete(key);
            continue;
          }
          jflag = ov.flag;
          const cn = this.conn3!.get(j)!;
          // fix_surface_global.cpp: edges -1..-3 take externalEdge, corners -4..-6 externalCor, the interior is INTERNAL
          if (jflag >= -3 && jflag < 0) external = cn.externalEdge[-jflag - 1];
          else if (jflag < -3) external = cn.externalCor[-jflag - 4];
        }
        const rvec = ov.r;
        const dotn = n[0] * rvec[0] + n[1] * rvec[1] + n[2] * rvec[2];
        const rsq = ov.rsq;
        const rmag = Math.sqrt(rsq);
        const dr: V3 = [rvec[0] / rmag, rvec[1] / rmag, rvec[2] / rmag];
        const priority = jflag !== 1 ? (is3 && jflag < -3 ? 0 : 1) : 2;
        cs.push({
          index: j, tag: j, type: this.stype[j], flag: jflag, rmag, nside: dotn >= 0 ? SAME_SIDE : OPPOSITE_SIDE,
          external, priority, convexIndex: -1, overlap: radi - rmag, rsqCom, weight: 1.0,
          dr, surfNorm: [n[0], n[1], n[2]], drForce: [dr[0], dr[1], dr[2]], cindex: [], caflag: [], cwhich: [],
          ends, corners, ck1: -1, ck2: -1, caflag1: -1, caflag2: -1,
        });
      }
      if (cs.length === 0) continue;
      const solved = solveSphereContacts(cs, is3, this.conn2, this.conn3);
      for (const { n, composite, maxOverlap } of solved) {
        const dr = contactDr(cs, composite, n, maxOverlap, radi);
        const xc: V3 = [xi[0] - dr[0], xi[1] - dr[1], xi[2] - dr[2]];
        const j = cs[n].index;
        // velocity of the contact point, extended from the surface centre (vsurf, omegasurf)
        const ds: V3 = [xc[0] - this.xs[3 * j], xc[1] - this.xs[3 * j + 1], xc[2] - this.xs[3 * j + 2]];
        const ow = [this.oms[3 * j], this.oms[3 * j + 1], this.oms[3 * j + 2]];
        const vj: V3 = [
          this.vs[3 * j] + (ow[1] * ds[2] - ow[2] * ds[1]),
          this.vs[3 * j + 1] + (ow[2] * ds[0] - ow[0] * ds[2]),
          this.vs[3 * j + 2] + (ow[0] * ds[1] - ow[1] * ds[0]),
        ];
        // contact history: the record of the element, or one already held by a composite member
        const keyOf = (c: ContactSurf): string => `${sid}:${c.tag}`;
        let chosen = keyOf(cs[n]);
        if (!this.hist.has(chosen)) {
          for (const m of composite) if (this.hist.has(keyOf(cs[m]))) chosen = keyOf(cs[m]);
        }
        if (!this.hist.has(chosen)) this.hist.set(chosen, new Float64Array(3));
        const history = this.hist.get(chosen)!;
        const out = { forces: [0, 0, 0] as V3, torque: [0, 0, 0] as V3 };
        surfContactForces(this.coeffs, {
          dx: [xi[0] - xc[0], xi[1] - xc[1], xi[2] - xc[2]], radi,
          vi: [v[3 * i], v[3 * i + 1], v[3 * i + 2]], omegai: [omega[3 * i], omega[3 * i + 1], omega[3 * i + 2]],
          vj, meff: rmass[i], dt, historyUpdate: updating,
        }, history, out);
        for (const m of composite) {
          const km = keyOf(cs[m]);
          if (km !== chosen) this.hist.set(km, Float64Array.from(history));
        }
        for (let a = 0; a < 3; a++) {
          f[3 * i + a] += out.forces[a];
          tq[3 * i + a] += out.torque[a];
        }
      }
    }
  }
}
