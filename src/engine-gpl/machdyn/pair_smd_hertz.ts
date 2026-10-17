/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/pair_smd_hertz.cpp and pair_smd_hertz.h (PairHertz: pair_style smd/hertz) at commit b332f28.
 * MACHDYN author (from the sources): Mike Parks (SNL) for the contact model. LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * pair_style smd/hertz scale_factor (docs.lammps.org/pair_smd_hertz.html): "The parameter scale_factor can be used to
 * scale the particles' contact radii." and "The smd/hertz style calculates contact forces between SPH particles
 * belonging to different physical bodies." The pair_coeff argument is the contact stiffness ("should equal roughly one
 * half of the Young's modulus (or bulk modulus in the case of fluids)").
 *
 * Self contact: two particles of one type that were close in the reference configuration (closer than the sum of their
 * kernel radii) interact through their contact radii only when they are within half the sum of the contact radii.
 * The energy is fpair * 0.4 * overlap (the expression native uses, which conserves the total energy).
 */
import { Pair, StyleError, evTally, type PairCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { SMD_CONTACT } from '../../engine/smd';

/** Initial value of the stable time increment (native: 1.0e22 at the start of each compute). */
const BIG_DT = 1.0e22;

export class PairSMDHertz extends Pair {
  readonly name = 'smd/hertz';
  /** Contact forces tally the virial pair by pair (ghost pairs at half weight, as native ev_tally with newton off). */
  virialFdotr = false;
  explicitVirial = true;
  /** Scale factor of the contact radii (pair_style smd/hertz scale_factor). */
  private scale = 1;
  /** Bulk modulus (contact stiffness) per type pair, (ntypes+1)^2, symmetric. */
  private bulkmodulus = new Float64Array(0);
  private setflag = new Uint8Array(0);
  private maxrad = new Float64Array(0);
  /** Stable time increment of the last force evaluation (fix smd/adjust_dt reads it; native extract smd/hertz/stable_time_increment_ptr). */
  stableTimeIncrement = BIG_DT;
  /** native's size list (NeighConst::REQ_SIZE), in the enumeration of the newton pair setting (set at init) */
  private order = new NativePairOrder();

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('Illegal number of args for pair_style smd/hertz');
    this.scale = parseNum(args[0], 'scale_factor');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.bulkmodulus = new Float64Array(nt * nt);
    this.setflag = new Uint8Array(nt * nt);
    this.maxrad = new Float64Array(nt);
  }

  /** pair_coeff I J K: the contact stiffness of the type pair (I <= J; both orders are set). */
  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for pair coefficients');
    if (!this.ntypes) throw new StyleError('pair_coeff before the box exists');
    const nt = this.ntypes + 1;
    const K = parseNum(args[2], 'contact stiffness');
    const ilo = parseInt(args[0], 10), ihi = parseInt(args[0], 10);
    const jlo = parseInt(args[1], 10), jhi = parseInt(args[1], 10);
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.bulkmodulus[i * nt + j] = K;
        this.bulkmodulus[j * nt + i] = K;
        this.setflag[i * nt + j] = 1;
        this.setflag[j * nt + i] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** Cutoff of a type pair: the sum of the largest kernel radii of the two types (native init_one, no frozen particles). */
  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.setflag[i * nt + j]) throw new StyleError('All pair coeffs are not set');
    return this.maxrad[i] + this.maxrad[j];
  }

  initStyle(ctx: StyleContext): void {
    const nt = this.ntypes + 1;
    this.order = new NativePairOrder(ctx.newtonPair !== false, { allow: (a, b) => !!this.setflag[a * nt + b] });
    const s = ctx.s;
    if (!s) throw new StyleError('pair smd/hertz needs a box and atoms');
    if (!s.x0 || !s.custom.has(SMD_CONTACT)) throw new StyleError('Pair style smd/hertz requires atom style with contact_radius');
    const onerad = new Float64Array(this.ntypes + 1);
    for (let i = 0; i < s.n; i++) onerad[s.type[i]] = Math.max(onerad[s.type[i]], s.radius![i]);
    this.maxrad = onerad;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const nt = this.ntypes + 1;
    const x = pc.x, f = pc.f, type = pc.type;
    const nlocal = pc.nlocal;
    const owner = pc.nb.owner;
    const contact = s.custom.get(SMD_CONTACT)!.data;
    const x0 = s.x0!;
    const kernel = s.radius!;
    const rmass = s.rmass!;
    const periodic: number[] = [];
    for (let d = 0; d < s.dimension; d++) if (s.box.periodic[d]) periodic.push(d);
    const anyPeriodic = periodic.length > 0;
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    let dtMin = BIG_DT;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const j = pairs[p + 1] & NEIGHMASK;
      const itype = type[i], jtype = type[j];
      if (!this.setflag[itype * nt + jtype]) continue;
      const jo = owner[j];
      const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const ri = this.scale * contact[i];
      const rj = this.scale * contact[jo];
      let rcut = ri + rj;
      if (rsq >= rcut * rcut) continue;
      if (itype === jtype) {
        // self contact: a pair close in the reference configuration is left to the bulk (SPH) interaction
        const d0 = [x0[3 * jo] - x0[3 * i], x0[3 * jo + 1] - x0[3 * i + 1], x0[3 * jo + 2] - x0[3 * i + 2]];
        if (anyPeriodic) {
          for (const d of periodic) {
            const L = s.box.hi[d] - s.box.lo[d];
            if (Math.abs(d0[d]) > 0.5 * L) d0[d] -= d0[d] < 0 ? -L : L;
          }
        }
        const rSq0 = d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2];
        const sphCut = kernel[i] + kernel[jo];
        if (rSq0 < sphCut * sphCut) {
          rcut = 0.5 * rcut;
          if (rsq > rcut * rcut) continue;
        }
      }
      const r = Math.sqrt(rsq);
      const delta = rcut - r;
      const rGeom = (ri * rj) / rcut;
      // Hertzian short-range force (pair_smd_hertz.cpp; 1.066666667 = 16/15 for a Poisson ratio 1/4 in 3d)
      let fpair = 1.066666667e0 * this.bulkmodulus[itype * nt + jtype] * delta * Math.sqrt(delta * rGeom);
      const evdwl = fpair * 0.4e0 * delta;
      const dtCrit = 3.14 * Math.sqrt(0.5 * (rmass[i] + rmass[jo]) / (fpair / delta));
      if (dtCrit < dtMin) dtMin = dtCrit;
      if (r > 2.0e-16) fpair /= r;
      else fpair = 0.0;
      // a ghost partner carries half of the pair (newton off): the energy and virial tallies take half
      const w = j < nlocal ? 1 : 0.5;
      evTally(pc, i, j, w * evdwl, 0, w * fpair, delx, dely, delz);
      f[3 * i] += delx * fpair;
      f[3 * i + 1] += dely * fpair;
      f[3 * i + 2] += delz * fpair;
      if (j < nlocal) {
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
      }
    }
    this.stableTimeIncrement = dtMin;
  }
}
