/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/GRANSURF/pair_surf_granular.cpp and pair_surf_granular.h (PairSurfGranular:
 * compute, calculate_endpts, prewalk_connections2d, walk_connections2d, calculate_2d_forces, init_style), with
 * src/GRANSURF/surf_extra.cpp and src/GRANSURF/fix_surface.cpp (see surf_contact.ts and connect2d.ts).
 * Contributing authors of pair_surf_granular.cpp: Joel Clemmer (SNL), Dan Bolintineanu (SNL).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/pair_surf_granular.html: "The pair style surf/granular is designed to compute forces between
 * granular particles and surface elements (lines in 2d, triangles in 3d) ... It is meant to be used with the
 * fix surface/local command." The granular model keywords are those of pair granular.
 *
 * Browser-engine differences:
 *   - the granular sub-models are the subset of surf_contact.ts (normal hooke, hertz, hertz/material; tangential
 *     linear_nohistory, linear_history; damping mass_velocity, viscoelastic, tsuji); the rolling, twisting, mindlin and
 *     heat keywords are refused with a StyleError;
 *   - the neighbour list is the engine's half list; each sphere-surface pair is taken once and the contacts are
 *     grouped per sphere (the native pair loops over the sphere's own neighbours);
 *   - the contact history is kept in a map keyed by the atom IDs of the sphere and the surface element (the native
 *     fix neigh/history transfers the same record on reneighbouring); records of pairs not in the list are dropped;
 *   - velocities and angular velocities of ghost atoms are those of their owners (the same as the core granular
 *     pair), and forces on ghosts are folded back by the engine.
 */
import { Pair, StyleError, tallyAtomXyz, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { isLineStyle, isTriStyle, LINE_BONUS, LINE_FLAG, TRI_BONUS, TRI_BONUS_COLS, TRI_FLAG } from '../../engine/atoms';
import { buildConnect2d, lineEnds, type Connect2d } from './connect2d';
import { surfaceLocalFlatthresh } from './fix_surface_local';
import { buildConnect3d, triCorners, type Connect3d } from './connect3d';
import {
  overlapSphereLine, overlapSphereTri, SAME_SIDE, OPPOSITE_SIDE,
  surfContactForces, type ContactSurf, type SurfCoeffs, type SurfDamping, type SurfNormal, type SurfTangential,
} from './surf_contact';
import { contactDr, solveSphereContacts } from './contact_flow';

type V3 = [number, number, number];

/** A parsed pair_coeff: the coefficients and the optional cutoff (NaN: derived from the radii). */
interface Coeffs extends SurfCoeffs {
  cutoff: number;
}

/**
 * Keyword parser of pair_coeff for the supported granular models (pair_granular.html): the normal models hooke
 * (k_n, eta_n0), hertz (k_n, eta_n0) and hertz/material (E, eta_n0, nu); the tangential models linear_nohistory
 * (x_gamma_t, mu_s) and linear_history (k_t, x_gamma_t, mu_s); the damping models mass_velocity, viscoelastic
 * (the default) and tsuji. The rolling, twisting, heat and other tangential models are refused.
 */
export function parseSurfCoeff(args: string[]): Coeffs {
  let k = NaN, damp = NaN, kt = 0, xt = NaN, mu = NaN, cutoff = NaN;
  let normal: SurfNormal | null = null, tangential: SurfTangential | null = null;
  // pair_granular.html: "If the *damping* keyword is not specified, the *viscoelastic* model is used by default."
  let damping: SurfDamping = 'viscoelastic';
  const num = (w: string | undefined, what: string): number => {
    if (w === undefined || !Number.isFinite(Number(w))) throw new StyleError(`pair_coeff surf/granular: ${what} must be a number`);
    return Number(w);
  };
  for (let n = 0; n < args.length; n++) {
    const w = args[n];
    if (w === 'hooke' || w === 'hertz') {
      normal = w;
      k = num(args[n + 1], `${w} kn`);
      damp = num(args[n + 2], `${w} damp`);
      n += 2;
    } else if (w === 'hertz/material') {
      // doc: "hertz/material : E, eta_n0 (or e), nu"; native: k = 4/3 E / (2 (1 - nu^2)) for a surface contact
      normal = w;
      const E = num(args[n + 1], 'hertz/material E');
      damp = num(args[n + 2], 'hertz/material damp');
      const nu = num(args[n + 3], 'hertz/material nu');
      if (!(E >= 0)) throw new StyleError('Illegal Hertz material normal model');
      k = (4 / 3) * (E / (2 * (1 - nu * nu)));
      n += 3;
    } else if (w === 'tangential') {
      const t = args[n + 1];
      if (t === 'linear_nohistory') {
        tangential = t;
        xt = num(args[n + 2], 'linear_nohistory x_gamma_t');
        mu = num(args[n + 3], 'linear_nohistory mu_s');
        n += 3;
      } else if (t === 'linear_history') {
        tangential = t;
        kt = num(args[n + 2], 'linear_history k_t');
        xt = num(args[n + 3], 'linear_history x_gamma_t');
        mu = num(args[n + 4], 'linear_history mu_s');
        n += 4;
      } else {
        throw new StyleError(`pair_style surf/granular: tangential model '${t ?? ''}' is not supported (linear_nohistory, linear_history)`);
      }
    } else if (w === 'damping') {
      const d = args[n + 1];
      if (d !== 'mass_velocity' && d !== 'viscoelastic' && d !== 'tsuji') {
        throw new StyleError(`pair_style surf/granular: damping model '${d ?? ''}' is not supported (mass_velocity, viscoelastic, tsuji)`);
      }
      damping = d;
      n += 1;
    } else if (w === 'cutoff') {
      cutoff = num(args[n + 1], 'cutoff');
      n += 1;
    } else {
      throw new StyleError(`pair_style surf/granular: granular keyword '${w}' is not supported`);
    }
  }
  if (normal === null || tangential === null) {
    throw new StyleError('pair_coeff surf/granular needs a normal model (hooke, hertz, hertz/material) and tangential linear_nohistory or linear_history');
  }
  if (!(k >= 0 && damp >= 0 && kt >= 0 && xt >= 0 && mu >= 0)) throw new StyleError('Illegal surf/granular coefficients');
  return { k, damp, kt, xt, mu, cutoff, normal, damping, tangential };
}

export class PairSurfGranular extends Pair {
  readonly name = 'surf/granular';
  /** The virial is the fdotr sum (pair_surf_granular.cpp: virial_fdotr_compute) unless a hybrid makes it tally each contact. */
  virialFdotr = true;
  explicitInHybrid = true;
  /** Per type pair (i*nt+j, both orders): the coefficients. */
  private pm: (Coeffs | null)[] = [];
  private maxRad = new Float64Array(0);
  /** Connectivity of the owned surface lines (built once, at the first compute after init, as in fix surface/local). */
  private conn: Map<number, Connect2d> | null = null;
  /** Connectivity of the owned surface triangles (3d). */
  private conn3: Map<number, Connect3d> | null = null;
  /** Contact history per (sphere ID, element ID): the three tangential history values. */
  private hist = new Map<string, Float64Array>();

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length) throw new StyleError('Illegal pair_style surf/granular command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.pm = new Array(nt * nt).fill(null);
  }

  /** pair_coeff I J surf/granular <granular keywords> (pair_granular.html). */
  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 3) throw new StyleError('Incorrect args for pair coefficients');
    const nt = this.ntypes + 1;
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const c = parseSurfCoeff(args.slice(2));
    for (let i = ilo; i <= ihi; i++) {
      for (let j = jlo; j <= jhi; j++) {
        this.pm[i * nt + j] = c;
        this.pm[j * nt + i] = c;
      }
    }
  }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s) throw new StyleError('Pair surf/granular requires atom style line (2d) or tri (3d)');
    if (s.dimension === 2 && !isLineStyle(s.atomStyle)) throw new StyleError('Pair surf/granular requires atom style line in 2d');
    if (s.dimension === 3 && !isTriStyle(s.atomStyle)) throw new StyleError('Pair surf/granular requires atom style tri in 3d');
    if (!s.radius || !s.rmass || !s.omega || !s.torque) throw new StyleError('Pair surf/granular requires atom attributes radius, rmass, omega');
    // the surface connectivity is built from unwrapped coordinates and is not periodic yet: a surface element and a
    // particle across a periodic boundary would not meet, so periodic in-plane dimensions are refused (z of a 2d box is
    // periodic by definition and does not matter)
    const dims = s.dimension === 2 ? [0, 1] : [0, 1, 2];
    if (dims.some((d) => s.box.periodic[d])) {
      throw new StyleError('pair surf/granular with periodic boundaries is not supported by the browser engine yet (use boundary f or s in the surface plane)');
    }
    this.conn = null;
    this.conn3 = null;
    const nt = this.ntypes + 1;
    this.maxRad = new Float64Array(nt);
    for (let k = 0; k < s.n; k++) {
      const t = s.type[k];
      if (s.radius[k] > this.maxRad[t]) this.maxRad[t] = s.radius[k];
    }
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const p = this.pm[i * nt + j];
    if (!p) throw new StyleError(`All pair coeffs are not set (surf/granular ${i} ${j})`);
    if (p.cutoff > 0) return p.cutoff;
    return this.maxRad[i] + this.maxRad[j];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const bonus = s.custom.get(LINE_BONUS)?.data;
    const flag = s.custom.get(LINE_FLAG)?.data;
    const is3 = s.dimension === 3;
    const surfFlag = is3 ? s.custom.get(TRI_FLAG)?.data : flag;
    const tbonus = s.custom.get(TRI_BONUS)?.data;
    if (!surfFlag || (!is3 && !bonus) || (is3 && !tbonus) || !s.radius || !s.rmass || !s.omega || !s.torque) {
      throw new StyleError('Pair surf/granular requires atom style line (2d) or tri (3d)');
    }
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const nt = this.ntypes + 1;
    const radius = s.radius, rmass = s.rmass, omega = s.omega, v = s.v, id = s.id;
    const tq = new Float64Array(3 * pc.nall);
    // connectivity of the owned surface elements, built at the first compute after init (fix surface/local)
    if (is3 && !this.conn3) {
      const tris: number[] = [];
      for (let i = 0; i < s.n; i++) if (surfFlag[i]) tris.push(i);
      this.conn3 = buildConnect3d(s.x, tris, s.quat!, tbonus!, TRI_BONUS_COLS, radius, id, surfaceLocalFlatthresh(s));
    } else if (!is3 && !this.conn) {
      const lines: number[] = [];
      for (let i = 0; i < s.n; i++) if (surfFlag[i]) lines.push(i);
      this.conn = buildConnect2d(s.x, lines, bonus!, id, surfaceLocalFlatthresh(s));
    }
    const conn = this.conn;
    const conn3 = this.conn3;
    const updating = pc.historyUpdate === true;
    const dt = s.dt;
    const seen = new Set<string>();

    // 1. the sphere-surface pairs of the list, grouped per sphere
    const groups = new Map<number, ContactSurf[]>();
    const nbl = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const j = nbl[k] & NEIGHMASK;
        const oi = owner[i], oj = owner[j];
        const surfI = surfFlag[oi] !== 0, surfJ = surfFlag[oj] !== 0;
        if (surfI === surfJ) {
          if (surfI) throw new StyleError('Pair surf/granular interaction is invalid (surface with surface)');
          continue;
        }
        const sph = surfI ? j : i;
        const lin = surfI ? i : j;
        const os = surfI ? oj : oi, ol = surfI ? oi : oj;
        const key = `${id[os]}:${id[ol]}`;
        seen.add(key);
        const dxc = x[3 * sph] - x[3 * lin], dyc = x[3 * sph + 1] - x[3 * lin + 1], dzc = x[3 * sph + 2] - x[3 * lin + 2];
        const rsqCom = dxc * dxc + dyc * dyc + dzc * dzc;
        const radi = radius[os], radsum = radi + radius[ol];
        if (rsqCom > radsum * radsum) {
          this.hist.delete(key);
          continue;
        }
        let jflag: number, rsq: number, normal: V3, dr: V3, dot: number;
        let rvec: V3, external: number, cornerPts: [V3, V3, V3] | null = null, ends: [V3, V3];
        if (is3) {
          const tc = triCorners(x, lin, ol, s.quat!, tbonus!, TRI_BONUS_COLS);
          const ov = overlapSphereTri(x, 3 * sph, radi, tc.c[0], tc.c[1], tc.c[2], tc.norm);
          if (!ov) {
            this.hist.delete(key);
            continue;
          }
          jflag = ov.flag;
          const cn = conn3!.get(id[ol]);
          if (!cn) throw new StyleError('Pair surf/granular: a surface triangle has no connectivity (fix surface/local)');
          external = jflag >= -3 ? cn.externalEdge[-jflag - 1] : cn.externalCor[-jflag - 4];
          normal = tc.norm;
          rvec = ov.r;
          rsq = ov.rsq;
          cornerPts = tc.c;
          ends = [tc.c[0], tc.c[1]];
        } else {
          const e = lineEnds(x, lin, bonus![2 * ol], bonus![2 * ol + 1]);
          const ov = overlapSphereLine(x, 3 * sph, radi, e.e0, e.e1);
          if (!ov) {
            this.hist.delete(key);
            continue;
          }
          jflag = ov.flag;
          const cn = conn!.get(id[ol]);
          if (!cn) throw new StyleError('Pair surf/granular: a surface line has no connectivity (fix surface/local)');
          external = jflag === -1 ? cn.external[0] : jflag === -2 ? cn.external[1] : 0;
          normal = e.norm;
          rvec = ov.r;
          rsq = ov.rsq;
          ends = [e.e0, e.e1];
        }
        // dot of the element normal with the (unnormalized) vector from the contact point to the sphere
        dot = normal[0] * rvec[0] + normal[1] * rvec[1] + normal[2] * rvec[2];
        const rmag = Math.sqrt(rsq);
        dr = [rvec[0] / rmag, rvec[1] / rmag, rvec[2] / rmag];
        const surfNorm: V3 = [normal[0], normal[1], normal[2]];
        const contact: ContactSurf = {
          // pair_surf_granular.cpp never assigns the ContactSurf::rmag member (only a local of the same name): the native
          // member keeps its zero-initialised value, which the 3d unconnected-edge test then reads (see forces3d.ts)
          index: lin, tag: id[ol], type: type[lin], flag: jflag, rmag: 0, nside: dot >= 0 ? SAME_SIDE : OPPOSITE_SIDE,
          external, priority: jflag !== 1 ? (is3 && jflag < -3 ? 0 : 1) : 2, convexIndex: -1, overlap: radi - rmag,
          rsqCom, weight: 1.0, dr, surfNorm, drForce: [dr[0], dr[1], dr[2]], cindex: [], caflag: [], cwhich: [],
          ends, corners: cornerPts ?? [ends[0], ends[1], ends[1]], ck1: -1, ck2: -1, caflag1: -1, caflag2: -1,
        };
        let g = groups.get(sph);
        if (!g) groups.set(sph, (g = []));
        g.push(contact);
      }
    }
    for (const key of this.hist.keys()) if (!seen.has(key)) this.hist.delete(key);

    // 2. per sphere: sort, pre-walk the sides, walk the composite surfaces and add the forces (contact_flow.ts)
    const sphereOrder = [...groups.keys()].sort((a, b) => a - b);
    for (const sph of sphereOrder) {
      const os = owner[sph];
      const cs = groups.get(sph)!;
      const radi = radius[os];
      const xi = [x[3 * sph], x[3 * sph + 1], x[3 * sph + 2]];
      const solved = solveSphereContacts(cs, is3, conn, conn3);
      for (const { n, composite, maxOverlap } of solved) {
        // geometry of the contact: the overlap-weighted normal of the composite, or the element's own direction
        const dr = contactDr(cs, composite, n, maxOverlap, radi);
        const xc = [xi[0] - dr[0], xi[1] - dr[1], xi[2] - dr[2]];
        const lin = cs[n].index;
        const ol = owner[lin];
        // velocity of the contact point, extended from the element centre (x of the element, v and omega of its owner)
        const ds = [xc[0] - x[3 * lin], xc[1] - x[3 * lin + 1], xc[2] - x[3 * lin + 2]];
        const vj = [
          v[3 * ol] + (omega[3 * ol + 1] * ds[2] - omega[3 * ol + 2] * ds[1]),
          v[3 * ol + 1] + (omega[3 * ol + 2] * ds[0] - omega[3 * ol] * ds[2]),
          v[3 * ol + 2] + (omega[3 * ol] * ds[1] - omega[3 * ol + 1] * ds[0]),
        ];
        // contact history: the element's own record, or a record already held by a composite member
        const skey = id[os];
        const keyOf = (c: ContactSurf): string => `${skey}:${c.tag}`;
        let chosen = keyOf(cs[n]);
        if (!this.hist.has(chosen)) {
          for (const m of composite) if (this.hist.has(keyOf(cs[m]))) chosen = keyOf(cs[m]);
        }
        if (!this.hist.has(chosen)) this.hist.set(chosen, new Float64Array(3));
        const history = this.hist.get(chosen)!;
        const coeffs = this.pm[type[os] * nt + type[lin]]!;
        const out = { forces: [0, 0, 0] as V3, torque: [0, 0, 0] as V3 };
        const dx = [xi[0] - xc[0], xi[1] - xc[1], xi[2] - xc[2]];
        surfContactForces(coeffs, {
          dx, radi, vi: [v[3 * os], v[3 * os + 1], v[3 * os + 2]], omegai: [omega[3 * os], omega[3 * os + 1], omega[3 * os + 2]],
          vj, meff: rmass[os], dt, historyUpdate: updating,
        }, history, out);
        // the composite members share the record of the contact
        for (const m of composite) {
          const km = keyOf(cs[m]);
          if (km !== chosen) this.hist.set(km, Float64Array.from(history));
        }
        const { forces, torque } = out;
        for (let a = 0; a < 3; a++) {
          f[3 * sph + a] += forces[a];
          f[3 * lin + a] -= forces[a];
          tq[3 * sph + a] += torque[a];
        }
        // ev_tally_xyz(i, j, ..., forces, model->dx) with dx from the contact point (pair_surf_granular.cpp)
        if (this.explicitVirial) {
          const vir = pc.acc.virial;
          vir[0] += dx[0] * forces[0]; vir[1] += dx[1] * forces[1]; vir[2] += dx[2] * forces[2];
          vir[3] += dx[0] * forces[1]; vir[4] += dx[0] * forces[2]; vir[5] += dx[1] * forces[2];
        }
        if (pc.vatom) tallyAtomXyz(pc, sph, lin, 0, forces[0], forces[1], forces[2], dx[0], dx[1], dx[2]);
      }
    }
    nb.reverseSum(tq, 3, s.torque);
  }
}
