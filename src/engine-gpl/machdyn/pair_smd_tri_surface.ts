/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/pair_smd_triangulated_surface.cpp and pair_smd_triangulated_surface.h (PairTriSurf: pair_style
 * smd/tri_surface; compute, PointTriangleDistance, settings, coeff, init_one, extract) at commit b332f28. MACHDYN author
 * (from the sources): Georg C. Ganzenmueller, Fraunhofer EMI. LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Contact between particles and a triangulated wall (docs.lammps.org/pair_smd.html): the wall triangles are the atoms
 * with molecule ID >= 65535 made by fix smd/wall_surface (the triangle vertices in the deformation prop, the unit normal in
 * x0, the contact radius). A particle closer than its contact radius to the closest point of a triangle is pushed away
 * with the Hertz-type force of the bulk modulus, its normal velocity is reflected, and its position is projected onto
 * the touch distance (native behaviour, which changes x and v of the particle in the pair style).
 * The stable time increment of the contacts is reported to fix smd/adjust_dt.
 *
 * Native order kept: the half (size) list is walked in native's order for the newton pair setting (NativePairOrder:
 * the funnel example runs with newton off, where every particle-ghost pair is visited from both sides); a contact
 * moves the particle's own copy (owned or ghost) and reflects that copy's velocity, so a later contact of the same
 * copy in the loop starts from them, as in native, where ghosts keep their own x and v until the next communication;
 * energy and virial are tallied as Pair::ev_tally does for that newton setting.
 */
import { Pair, StyleError, evTally, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { SMD_CONTACT, SMD_DEFGRAD } from '../../engine/smd';

/** Molecule IDs at or above this mark the wall triangles (pair_smd_triangulated_surface.cpp: mol >= 65535). */
export const WALL_MOLECULE_MIN = 65535;
const BIG_DT = 1.0e22;

/** Closest point CP of the point P to the triangle (TRI0, TRI1, TRI2) and its distance (Eberly 1999, as in the source). */
export const pointTriangleDistance = (P: number[], T0: number[], T1: number[], T2: number[]): { cp: number[]; dist: number } => {
  const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : hi < v ? hi : v);
  const edge0 = [T1[0] - T0[0], T1[1] - T0[1], T1[2] - T0[2]];
  const edge1 = [T2[0] - T0[0], T2[1] - T0[1], T2[2] - T0[2]];
  const v0 = [T0[0] - P[0], T0[1] - P[1], T0[2] - P[2]];
  const dot = (u: number[], w: number[]) => u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
  const a = dot(edge0, edge0);
  const b = dot(edge0, edge1);
  const c = dot(edge1, edge1);
  const d = dot(edge0, v0);
  const e = dot(edge1, v0);
  const det = a * c - b * b;
  let s = b * e - c * d;
  let t = b * d - a * e;
  if (s + t < det) {
    if (s < 0.0) {
      if (t < 0.0) {
        if (d < 0.0) {
          s = clamp(-d / a, 0.0, 1.0);
          t = 0.0;
        } else {
          s = 0.0;
          t = clamp(-e / c, 0.0, 1.0);
        }
      } else {
        s = 0.0;
        t = clamp(-e / c, 0.0, 1.0);
      }
    } else if (t < 0.0) {
      s = clamp(-d / a, 0.0, 1.0);
      t = 0.0;
    } else {
      const invDet = 1.0 / det;
      s *= invDet;
      t *= invDet;
    }
  } else if (s < 0.0) {
    const tmp0 = b + d;
    const tmp1 = c + e;
    if (tmp1 > tmp0) {
      const numer = tmp1 - tmp0;
      const denom = a - 2 * b + c;
      s = clamp(numer / denom, 0.0, 1.0);
      t = 1 - s;
    } else {
      t = clamp(-e / c, 0.0, 1.0);
      s = 0.0;
    }
  } else if (t < 0.0) {
    if (a + d > b + e) {
      const numer = c + e - b - d;
      const denom = a - 2 * b + c;
      s = clamp(numer / denom, 0.0, 1.0);
      t = 1 - s;
    } else {
      s = clamp(-e / c, 0.0, 1.0);
      t = 0.0;
    }
  } else {
    const numer = c + e - b - d;
    const denom = a - 2 * b + c;
    s = clamp(numer / denom, 0.0, 1.0);
    t = 1.0 - s;
  }
  const cp = [0, 0, 0];
  for (let k = 0; k < 3; k++) cp[k] = T0[k] + s * edge0[k] + t * edge1[k];
  const dist = Math.sqrt((cp[0] - P[0]) ** 2 + (cp[1] - P[1]) ** 2 + (cp[2] - P[2]) ** 2);
  return { cp, dist };
};

/** Pair::ev_tally with newton_pair off: each owned atom of the pair takes half of the energy and of the virial. */
const tallyNewtonOff = (
  pc: PairCompute, i: number, j: number, nlocal: number, evdwl: number, fpair: number, dx: number, dy: number, dz: number,
): void => {
  const a = pc.acc;
  const half = 0.5 * evdwl;
  if (i < nlocal) a.evdwl += half;
  if (j < nlocal) a.evdwl += half;
  const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
  const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
  const vir = a.virial;
  for (const k of [i, j]) {
    if (k >= nlocal) continue;
    vir[0] += 0.5 * v0; vir[1] += 0.5 * v1; vir[2] += 0.5 * v2; vir[3] += 0.5 * v3; vir[4] += 0.5 * v4; vir[5] += 0.5 * v5;
    if (pc.eatom) pc.eatom[k] += half;
    if (pc.vatom) {
      const va = pc.vatom;
      va[6 * k] += 0.5 * v0; va[6 * k + 1] += 0.5 * v1; va[6 * k + 2] += 0.5 * v2;
      va[6 * k + 3] += 0.5 * v3; va[6 * k + 4] += 0.5 * v4; va[6 * k + 5] += 0.5 * v5;
    }
  }
};

export class PairSMDTriSurface extends Pair {
  readonly name = 'smd/tri_surface';
  /** Contacts tally their virial pair by pair (explicit, as native ev_tally with newton off). */
  virialFdotr = false;
  explicitVirial = true;
  /** Scale factor of the contact radii (pair_style smd/tri_surface scale). */
  private scale = 1;
  private bulk = new Float64Array(0);
  private setflag = new Uint8Array(0);
  private maxrad = new Float64Array(0);
  /** native's size list (NeighConst::REQ_SIZE), in the enumeration of the newton pair setting (set at init) */
  private order = new NativePairOrder();
  private newton = true;
  private vcopy = new Float64Array(0);
  /** Stable time increment of the last evaluation (native extract smd/tri_surface/stable_time_increment_ptr). */
  stableTimeIncrement = BIG_DT;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('Illegal number of args for pair_style smd/tri_surface');
    this.scale = parseNum(args[0], 'scale');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.bulk = new Float64Array(nt * nt);
    this.setflag = new Uint8Array(nt * nt);
    this.maxrad = new Float64Array(nt);
  }

  /** pair_coeff I J K: the bulk modulus of the contact between the types (I <= J; both orders are set). */
  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for pair coefficients');
    if (!this.ntypes) throw new StyleError('pair_coeff before the box exists');
    const nt = this.ntypes + 1;
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const K = parseNum(args[2], 'bulk modulus');
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.bulk[i * nt + j] = K;
        this.bulk[j * nt + i] = K;
        this.setflag[i * nt + j] = 1;
        this.setflag[j * nt + i] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
    this.version++;
  }

  /** Cutoff of a type pair: the sum of the largest kernel radii of the two types (native init_one, no frozen particles). */
  initOne(i: number, j: number): number {
    if (!this.setflag[i * (this.ntypes + 1) + j]) throw new StyleError('All pair coeffs are not set');
    return this.maxrad[i] + this.maxrad[j];
  }

  initStyle(ctx: StyleContext): void {
    this.newton = ctx.newtonPair !== false;
    const nt = this.ntypes + 1;
    this.order = new NativePairOrder(this.newton, { allow: (a, b) => !!this.setflag[a * nt + b] });
    const s = ctx.s;
    if (!s) throw new StyleError('pair smd/tri_surface needs a box and atoms');
    if (!s.custom.has(SMD_CONTACT) || !s.radius) throw new StyleError('Pair style smd/tri_surface requires atom style with contact_radius');
    const onerad = new Float64Array(this.ntypes + 1);
    for (let i = 0; i < s.n; i++) onerad[s.type[i]] = Math.max(onerad[s.type[i]], s.radius[i]);
    this.maxrad = onerad;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const nt = this.ntypes + 1;
    const nlocal = pc.nlocal;
    const x = pc.x, f = pc.f, type = pc.type;
    const owner = pc.nb.owner;
    const mol = s.molecule!;
    const contact = s.custom.get(SMD_CONTACT)!.data;
    const vert = s.custom.get(SMD_DEFGRAD)!.data;
    const normals = s.x0!;
    const v = s.v;
    const rmass = s.rmass!;
    const periodic: number[] = [];
    for (let d = 0; d < s.dimension; d++) if (s.box.periodic[d]) periodic.push(d);
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    // every copy (owned or ghost) carries its own velocity during the loop (comm_modify vel yes: the ghosts received
    // their owners' velocities at the start of the step)
    const nall = pc.nall;
    if (this.vcopy.length < 3 * nall) this.vcopy = new Float64Array(3 * nall);
    const vc = this.vcopy;
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      vc[3 * k] = v[3 * o]; vc[3 * k + 1] = v[3 * o + 1]; vc[3 * k + 2] = v[3 * o + 2];
    }
    let stable = BIG_DT;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const j = pairs[p + 1] & NEIGHMASK;
      const oi = owner[i], oj = owner[j];
      const molI = mol[oi], molJ = mol[oj];
      let particle: number, tri: number;
      if (molI < WALL_MOLECULE_MIN && molJ >= WALL_MOLECULE_MIN) {
        particle = i;
        tri = j;
      } else if (molJ < WALL_MOLECULE_MIN && molI >= WALL_MOLECULE_MIN) {
        particle = j;
        tri = i;
      } else {
        throw new StyleError('unknown case');
      }
      const op = owner[particle], ot = owner[tri];
      const itype = type[op], jtype = type[ot];
      if (!this.setflag[itype * nt + jtype]) continue;
      const dx = [x[3 * tri] - x[3 * particle], x[3 * tri + 1] - x[3 * particle + 1], x[3 * tri + 2] - x[3 * particle + 2]];
      for (const d of periodic) {
        const L = s.box.hi[d] - s.box.lo[d];
        if (Math.abs(dx[d]) > 0.5 * L) dx[d] -= dx[d] < 0 ? -L : L;
      }
      const rsq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
      const rTri = this.scale * contact[ot];
      const rParticle = this.scale * contact[op];
      const rcut = rTri + rParticle;
      if (!(rsq < rcut * rcut)) continue;
      const normal = [normals[3 * ot], normals[3 * ot + 1], normals[3 * ot + 2]];
      if (!(Math.abs(dx[0] * normal[0] + dx[1] * normal[1] + dx[2] * normal[2]) < contact[op])) continue;
      const x4 = [x[3 * particle], x[3 * particle + 1], x[3 * particle + 2]];
      const T = (k: number) => [vert[9 * ot + 3 * k], vert[9 * ot + 3 * k + 1], vert[9 * ot + 3 * k + 2]];
      const { cp, dist: r } = pointTriangleDistance(x4, T(0), T(1), T(2));
      const x4cp = [x4[0] - cp[0], x4[1] - cp[1], x4[2] - cp[2]];
      if (x4cp[0] * normal[0] + x4cp[1] * normal[1] + x4cp[2] * normal[2] < 0.0) {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
      }
      if (r < 1.0 * contact[op]) {
        const delta = contact[op] - r;
        const rGeom = contact[op];
        let fpair = 1.066666667e0 * this.bulk[itype * nt + jtype] * delta * Math.sqrt(delta * rGeom);
        const dtCrit = 3.14 * Math.sqrt(rmass[op] / (fpair / delta));
        if (dtCrit < stable) stable = dtCrit;
        const evdwl = r * fpair * 0.4e0 * delta;
        fpair /= r + 1.0e-2 * contact[op];
        if (particle < nlocal) {
          for (let k = 0; k < 3; k++) f[3 * particle + k] += x4cp[k] * fpair;
        }
        if (tri < nlocal) {
          for (let k = 0; k < 3; k++) f[3 * tri + k] -= x4cp[k] * fpair;
        }
        if (this.newton) evTally(pc, i, j, evdwl, 0, fpair, x4cp[0], x4cp[1], x4cp[2]);
        else tallyNewtonOff(pc, i, j, nlocal, evdwl, fpair, x4cp[0], x4cp[1], x4cp[2]);
      }
      const touch = 1.0 * contact[op];
      if (r < touch) {
        const nn = [x4cp[0] / r, x4cp[1] / r, x4cp[2] / r];
        const vOld = [vc[3 * particle], vc[3 * particle + 1], vc[3 * particle + 2]];
        const vn = vOld[0] * nn[0] + vOld[1] * nn[1] + vOld[2] * nn[2];
        if (vn < 0.0) {
          for (let k = 0; k < 3; k++) {
            const vk = 1.0 * (-2.0 * vn * nn[k] + vOld[k]);
            vc[3 * particle + k] = vk;
            if (particle < nlocal) v[3 * particle + k] = vk;
          }
        }
        const xn = [cp[0] + touch * nn[0], cp[1] + touch * nn[1], cp[2] + touch * nn[2]];
        for (let k = 0; k < 3; k++) {
          x[3 * particle + k] = xn[k];
          if (particle < nlocal) s.x[3 * particle + k] = xn[k];
        }
      }
    }
    this.stableTimeIncrement = stable;
  }
}
