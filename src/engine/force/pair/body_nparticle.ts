import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';
import { BODY_RECORD } from '../../bodies';
import { quatToMatrix } from '../../compute/asphere';

/*
 * pair_style body/nparticle — docs.lammps.org/pair_body_nparticle.html: "Style body/nparticle is for use with
 * body particles and calculates pairwise body/body interactions as well as interactions between body and
 * point-particles." "The coordinates of a body particle are its center-of-mass (COM). If the COMs of a pair of
 * body particles are within the cutoff (global or type-specific, as specified above), then all interactions
 * between pairs of sub-particles in the two body particles are computed. E.g. if the first body particle has 3
 * sub-particles, and the second has 10, then 30 interactions are computed and summed to yield the total force
 * and torque on each body particle."
 *   pair_style body/nparticle cutoff
 *   pair_coeff I J epsilon sigma [cutoff]
 * The page: "The interaction between two sub-particles, or a sub-particle and point particle, or between two
 * point particles is computed as a Lennard-Jones interaction" with E = 4 epsilon [(sigma/r)^12 - (sigma/r)^6].
 *
 * Rules of the page that the code follows:
 *   - the cutoff is tested on the COM distance only: "no interaction between two body particles is computed if
 *     the two COMs are further apart than the cutoff, even if the distance between some pairs of their
 *     sub-particles is within the cutoff", and "all 30 interactions are computed even if the distance between a
 *     particular pair of sub-particles is greater than the cutoff";
 *   - a point particle is a body of one sub-particle at its COM (no torque);
 *   - "This pair style does not support the pair_modify shift, table, and tail options."
 *
 * The torque on body b is the sum of r x F over its sub-particles, r being the space displacement of the
 * sub-particle from the COM (R_b d_body). Virial: Measured with native LAMMPS (black box): the pressure of a
 * body system equals the sum over sub-particle pairs of (x_COM,i - x_COM,j) (x) F_ab, with the COM separation and
 * no sub-particle lever arms (the thermo press of examples/body/in.body, 2 steps, matches to 1e-12). The engine
 * therefore tallies the virial itself rather than with the fdotr shortcut.
 *
 * Refusals (StyleError): pair_modify shift or tail; a special-bond factor other than 1 for a body pair (the page
 * describes no special-bond factor for this style, as for gayberne).
 */
export class PairBodyNParticle extends Pair {
  readonly name = 'body/nparticle';
  virialFdotr = false;
  cutGlobal = 0;
  p!: PairParams;

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style body/nparticle command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style body/nparticle cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cut = args.length === 5 ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, cut]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'epsilon', mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    return p.get('cut', i, j);
  }

  init(ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError('pair_modify shift and tail are not supported by pair_style body/nparticle');
    super.init(ctx);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const rec = s.custom.get(BODY_RECORD);
    if (!rec || !s.quat || !s.torque) throw new StyleError('Pair body/nparticle requires atom style body');
    if (!s.atomStyle.startsWith('body nparticle')) throw new StyleError(`pair_style body/nparticle requires atom_style body nparticle (current: ${s.atomStyle})`);
    const B = rec.cols, bd = rec.data;
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const tq = new Float64Array(3 * pc.nall);
    // Space displacements of the sub-particles of every owned atom (a point particle: one sub-particle at its COM).
    const nsub = new Int32Array(nlocal);
    const start = new Int32Array(nlocal + 1);
    for (let o = 0; o < nlocal; o++) {
      nsub[o] = Math.max(1, bd[B * o]);
      start[o + 1] = start[o] + nsub[o];
    }
    const sv = new Float64Array(3 * start[nlocal]);
    for (let o = 0; o < nlocal; o++) {
      if (bd[B * o] === 0) continue; // point particle: the zero displacement stays
      const q = s.quat;
      const R = quatToMatrix(q[4 * o], q[4 * o + 1], q[4 * o + 2], q[4 * o + 3]);
      for (let k = 0; k < nsub[o]; k++) {
        const d0 = bd[B * o + 4 + 3 * k], d1 = bd[B * o + 5 + 3 * k], d2 = bd[B * o + 6 + 3 * k];
        const at = 3 * (start[o] + k);
        sv[at] = R[0] * d0 + R[1] * d1 + R[2] * d2;
        sv[at + 1] = R[3] * d0 + R[4] * d1 + R[5] * d2;
        sv[at + 2] = R[6] * d0 + R[7] * d1 + R[8] * d2;
      }
    }
    const acc = pc.acc;
    let evdwl = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const dxc = xi - x[3 * j], dyc = yi - x[3 * j + 1], dzc = zi - x[3 * j + 2];
        const rsqc = dxc * dxc + dyc * dyc + dzc * dzc;
        const t = ti * nt + type[j];
        if (rsqc >= cutsq[t]) continue;
        if (jj >>> SBBITS !== 0 && (sL[jj >>> SBBITS] !== 1 || sC[jj >>> SBBITS] !== 1)) {
          throw new StyleError('pair_style body/nparticle does not support special_bonds factors other than 1');
        }
        const eps = this.p.get('epsilon', ti, type[j]), sig = this.p.get('sigma', ti, type[j]);
        const sig2 = sig * sig;
        const epsx4 = 4 * eps, epsx48 = 48 * eps, epsx24 = 24 * eps;
        for (let a = 0; a < nsub[oi]; a++) {
          const sa = 3 * (start[oi] + a);
          const ax = sv[sa], ay = sv[sa + 1], az = sv[sa + 2];
          for (let b = 0; b < nsub[oj]; b++) {
            const sb = 3 * (start[oj] + b);
            const bx = sv[sb], by = sv[sb + 1], bz = sv[sb + 2];
            // x_a - x_b with the sub-particle lever arms of both bodies
            const dx = dxc + ax - bx, dy = dyc + ay - by, dz = dzc + az - bz;
            const r2 = dx * dx + dy * dy + dz * dz;
            const r2inv = 1 / r2;
            const sr2 = sig2 * r2inv, sr6 = sr2 * sr2 * sr2, sr12 = sr6 * sr6;
            const eLJ = epsx4 * (sr12 - sr6);
            const fpair = (epsx48 * sr12 - epsx24 * sr6) * r2inv;
            evdwl += eLJ;
            const Fx = fpair * dx, Fy = fpair * dy, Fz = fpair * dz;
            // force on sub-particle a of body i, reaction on b of body j
            f[3 * i] += Fx; f[3 * i + 1] += Fy; f[3 * i + 2] += Fz;
            f[3 * j] -= Fx; f[3 * j + 1] -= Fy; f[3 * j + 2] -= Fz;
            // torques: r x F about the COM of each body
            if (bd[B * oi] !== 0) {
              tq[3 * i] += ay * Fz - az * Fy;
              tq[3 * i + 1] += az * Fx - ax * Fz;
              tq[3 * i + 2] += ax * Fy - ay * Fx;
            }
            if (bd[B * oj] !== 0) {
              tq[3 * j] -= by * Fz - bz * Fy;
              tq[3 * j + 1] -= bz * Fx - bx * Fz;
              tq[3 * j + 2] -= bx * Fy - by * Fx;
            }
            // virial with the COM separation (see the header): sum of x_COM,ij (x) F_ab
            const w = [dxc * Fx, dyc * Fy, dzc * Fz, dxc * Fy, dxc * Fz, dyc * Fz];
            for (let c = 0; c < 6; c++) acc.virial[c] += w[c];
            if (pc.eatom) { pc.eatom[i] += 0.5 * eLJ; pc.eatom[j] += 0.5 * eLJ; }
            if (pc.vatom) {
              for (let c = 0; c < 6; c++) { pc.vatom[6 * i + c] += 0.5 * w[c]; pc.vatom[6 * j + c] += 0.5 * w[c]; }
            }
          }
        }
      }
    }
    nb.reverseSum(tq, 3, s.torque);
    acc.evdwl += evdwl;
  }
}
