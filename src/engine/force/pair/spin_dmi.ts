import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK, type Neighbor } from '../../neighbor';
import { numeric, parseNum } from '../util';
import { SPIN_HBAR } from './spin_exchange';
import { spinContribute, spinPairSettings, tallyPairEnergy, tallyPairVirial, type SpinPairStyle } from '../spin_book';

/*
 * pair_style spin/dmi cutoff and pair_style spin/magelec cutoff — docs.lammps.org/pair_spin_dmi.html and
 * docs.lammps.org/pair_spin_magelec.html.
 *
 *   pair_style spin/dmi 4.0
 *   pair_coeff * * dmi 2.6 0.001 1.0 0.0 0.0
 *   pair_style spin/magelec 4.5
 *   pair_coeff * * magelec 4.5 0.00109 1.0 1.0 1.0
 *
 * spin/dmi: "Style *spin/dmi* computes the Dzyaloshinskii-Moriya (DM) interaction between pairs of magnetic
 * spins" with "\mathbf{H}_{dm} = \sum_{{ i,j}=1,i\neq j}^{N} \left( \vec{e}_{ij} \times \vec{D} \right) \cdot\left(\vec{s}_{i}\times \vec{s}_{j}\right),"
 * and the DM vector D. The page lists "rc (distance units)" and "Dx, Dy, Dz  (direction of D)". Measured with native LAMMPS (black box, two atoms): the energy of a pair is
 * (e_ij x D).(s_i x s_j) counted once per pair, with D = |D| (Dx,Dy,Dz)/(Dx^2+Dy^2+Dz^2) (the direction is not
 * normalized to length 1 by the code: a direction of 2 0 0 with |D| = 0.001 gives D = 0.0005 x).
 * Precession vector: "\vec{\omega}_i = -\frac{1}{\hbar} \sum_{j}^{Neighb} \vec{s}_{j}\times \left(\vec{e}_{ij}\times \vec{D} \right)"
 * (measured: matches fmx fmy fmz to the printed digits for two and six geometries).
 * Mechanical force (fix nve/spin lattice moving), measured with native LAMMPS (black box, six configurations):
 * F_i = (1/r_ij) H(D, s_i x s_j) per neighbour, with H(D, C) = D x C in x and y, but the z component is
 * D_x C_y - D_y C_z (not D_x C_y - D_y C_x). The page writes "\vec{F}_i = -\sum_{j}^{Neighb} \frac{1}{r_{ij}} \vec{D} \times \left( \vec{s}_{i}\times \vec{s}_{j} \right)";
 * the measured sign and z component differ from that form, and the native values are reproduced here.
 *
 * spin/magelec: "Style *spin/me* computes a magneto-electric interaction between pairs of magnetic spins."
 * "\vec{\omega}_i & = -\frac{1}{\hbar} \sum_{j}^{Neighb} \vec{s}_{j}\times\vec{D}(r_{ij})" with the vector
 * D(r) not given on the page. Measured with native LAMMPS (black box, two atoms, five directions of the vector,
 * two geometries): E = |E| (Ex,Ey,Ez)/(Ex^2+Ey^2+Ez^2) with |E| the first pair_coeff value after rc; energy
 * E.(e_ij x (s_i x s_j)) once per pair; precession -(1/hbar) s_j x (E x e_ij) with e_ij = (r_i - r_j)/r (the
 * distance in the page's r_ij is the unit vector: the measured precession scales as 1/r). Mechanical force (fix
 * nve/spin lattice moving), as native LAMMPS does (src/SPIN/pair_spin_magelec.cpp compute_magelec_mech at b332f28):
 * each directed pair adds f_i = (1/2) M x E, independent of r, with M = (s_iy s_iz - s_iz s_jy, s_iz s_ix - s_ix s_jz,
 * s_ix s_iy - s_iy s_jx) (the source mixes the components of s_i and s_j this way).
 *
 * Both styles apply the mechanical force of a directed pair to i and its opposite to j, as native LAMMPS does
 * (src/SPIN/pair_spin_dmi.cpp and pair_spin_magelec.cpp compute at b332f28), so the f dot r virial over owned and
 * ghost atoms is the pair virial, and they tally per-atom energy and virial as Pair::ev_tally_xyz does.
 *
 * Spins are the unit vectors of atom_style spin; ghost spins are the spins of their owners (nb.owner). The pair
 * needs a full neighbor list; each owned atom gets its precession vector from its full list; the energy is
 * halved per directed pair. Precession vectors go through spin_book.ts (overlays with other spin styles).
 */

/** The DM / magneto-electric vector of one pair type: magnitude times the direction, divided by |direction|^2. */
const vectorCoeff = (args: string[], what: string): [number, number, number, number] => {
  const rc = parseNum(args[0], 'rc');
  const mag = parseNum(args[1], what);
  const v = [parseNum(args[2], 'x'), parseNum(args[3], 'y'), parseNum(args[4], 'z')];
  const n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (!(rc > 0)) throw new StyleError(`${what}: rc must be > 0`);
  if (!(n2 > 0)) throw new StyleError(`${what}: the direction vector must be non-zero`);
  return [rc, mag * v[0] / n2, mag * v[1] / n2, mag * v[2] / n2];
};

abstract class PairSpinVector extends Pair implements SpinPairStyle {
  abstract readonly name: string;
  readonly spinForce: boolean = false;
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;
  cutGlobal = 0;
  /** Per type pair: rc, Vx, Vy, Vz (the vector of the interaction, see vectorCoeff). */
  p!: PairParams;
  private fmBuf = new Float64Array(0);

  protected abstract coeffKeyword(): string;
  /** Energy of one directed pair from the vector V, unit e = (r_i - r_j)/r and C = s_i x s_j. */
  protected abstract energyOf(V: number[], e: number[], C: number[]): number;
  /** Precession vector contribution (per neighbour) with the spin s_j, the unit vector e and V. */
  protected abstract omegaTerm(V: number[], e: number[], sj: number[]): number[];
  /** Mechanical force of one directed pair, added to i and subtracted from j: V, the spins, C = s_i x s_j and r. */
  protected abstract directedForce(V: number[], si: number[], sj: number[], C: number[], r: number): number[];

  settings(args: string[], ctx?: StyleContext): void {
    // a second word is accepted and ignored, as native's settings read only the first
    if (args.length < 1 || args.length > 2) throw new StyleError('Incorrect number of args in pair_style pair/spin command');
    spinPairSettings(ctx);
    this.cutGlobal = numeric(args[0]);
    if (!(this.cutGlobal > 0)) throw new StyleError(`${this.name}: cutoff must be > 0`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['rc', 'Vx', 'Vy', 'Vz']);
  }

  coeff(args: string[]): void {
    // docs.lammps.org/pair_spin_dmi.html and pair_spin_magelec.html: the coefficients follow the keyword
    // (dmi / magelec), as in the page examples "pair_coeff * * dmi 2.6 0.001 1.0 0.0 0.0". Native rejects the
    // keyword-less form inside pair hybrid/overlay ("Incorrect args"), so it is refused here too.
    const kw = this.coeffKeyword();
    if (args[2] !== kw) throw pairCoeffArgs(`usage: pair_coeff I J ${kw} rc magnitude Vx Vy Vz (the ${kw} keyword is required)`, 'Incorrect args in pair_style command', false);
    const k = 3;
    if (args.length < k + 5) throw pairCoeffArgs(`usage: pair_coeff I J ${kw} rc magnitude Vx Vy Vz`, 'Incorrect args in pair_style command', false);
    // As native LAMMPS does (src/SPIN/pair_spin_dmi.cpp PairSpinDmi::coeff and pair_spin_magelec.cpp at b332f28): the argument count must be exactly 8, so an extra word is refused.
    if (args.length !== 8) throw new StyleError('Incorrect args in pair_style command');
    const [rc, Vx, Vy, Vz] = vectorCoeff(args.slice(k), this.name);
    this.p.setRange(args[0], args[1], [rc, Vx, Vy, Vz]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError(`pair_modify shift/tail is not supported for pair style ${this.name}`);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    return this.p.get('rc', i, j);
  }

  omegaOf(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array }, out: Float64Array): void {
    out[0] = 0; out[1] = 0; out[2] = 0;
    const nt = this.ntypes + 1;
    const owner = nb.owner;
    const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
    const ti = type[i] * nt;
    for (let k = k0; k < k1; k++) {
      const j = list.neighbors[k] & NEIGHMASK;
      const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const t = ti + type[j];
      if (rsq >= this.cutsq[t]) continue;
      const r = Math.sqrt(rsq);
      const o = owner[j];
      const V = this.vec(type[i], type[j]);
      const sj = [sp[4 * o], sp[4 * o + 1], sp[4 * o + 2]];
      const w = this.omegaTerm(V, [dx / r, dy / r, dz / r], sj);
      out[0] += w[0]; out[1] += w[1]; out[2] += w[2];
    }
  }

  private vec(ti: number, tj: number): number[] {
    return [this.p.get('Vx', ti, tj), this.p.get('Vy', ti, tj), this.p.get('Vz', ti, tj)];
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const s = pc.s;
    if (!s.sp || !s.fm) throw new StyleError(`pair style ${this.name} requires atom_style spin`);
    const lattice = s.spinLattice === true;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const sp = s.sp;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    if (this.fmBuf.length < 3 * pc.nlocal) this.fmBuf = new Float64Array(3 * pc.nlocal);
    const fmv = this.fmBuf;
    fmv.fill(0, 0, 3 * pc.nlocal);
    let evdwl = 0;
    for (let i = 0; i < pc.nlocal; i++) {
      const ti = type[i] * nt;
      const si = [sp[4 * i], sp[4 * i + 1], sp[4 * i + 2]];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        const t = ti + tj;
        const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        const e = [dx / r, dy / r, dz / r];
        const V = this.vec(type[i], tj);
        const o = owner[j];
        const sj = [sp[4 * o], sp[4 * o + 1], sp[4 * o + 2]];
        const C = [si[1] * sj[2] - si[2] * sj[1], si[2] * sj[0] - si[0] * sj[2], si[0] * sj[1] - si[1] * sj[0]];
        const ev = 0.5 * this.energyOf(V, e, C);
        evdwl += ev;
        if (pc.eatom) tallyPairEnergy(pc.eatom, i, j, ev);
        const w = this.omegaTerm(V, e, sj);
        fmv[3 * i] += w[0]; fmv[3 * i + 1] += w[1]; fmv[3 * i + 2] += w[2];
        if (lattice) {
          const fo = this.directedForce(V, si, sj, C, r);
          f[3 * i] += fo[0]; f[3 * i + 1] += fo[1]; f[3 * i + 2] += fo[2];
          f[3 * j] -= fo[0]; f[3 * j + 1] -= fo[1]; f[3 * j + 2] -= fo[2];
          if (pc.vatom) tallyPairVirial(pc.vatom, i, j, dx, dy, dz, fo[0], fo[1], fo[2]);
        }
      }
    }
    pc.acc.evdwl += evdwl;
    spinContribute(s, this, fmv, evdwl);
  }
}

/** Cross product a x b. */
const cross = (a: number[], b: number[]): number[] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export class PairSpinDMI extends PairSpinVector {
  readonly name: string = 'spin/dmi';
  protected coeffKeyword(): string { return 'dmi'; }
  /** E = (e_ij x D).(s_i x s_j) */
  protected energyOf(V: number[], e: number[], C: number[]): number { return dot(cross(e, V), C); }
  /** omega_i = -(1/hbar) s_j x (e_ij x D) */
  protected omegaTerm(V: number[], e: number[], sj: number[]): number[] {
    const w = cross(sj, cross(e, V));
    return [-w[0] / SPIN_HBAR, -w[1] / SPIN_HBAR, -w[2] / SPIN_HBAR];
  }
  /**
   * Measured form (see the header): (1/r) H(D, C) per pair with H = (D_y C_z - D_z C_y, D_z C_x - D_x C_z,
   * D_x C_y - D_y C_z); each of the two directed pairs adds half of it (H(D, C) is odd in C = s_i x s_j).
   */
  protected directedForce(V: number[], _si: number[], _sj: number[], C: number[], r: number): number[] {
    const [Dx, Dy, Dz] = V;
    const [Cx, Cy, Cz] = C;
    const h = 0.5 / r;
    return [h * (Dy * Cz - Dz * Cy), h * (Dz * Cx - Dx * Cz), h * (Dx * Cy - Dy * Cz)];
  }
}

export class PairSpinMagelec extends PairSpinVector {
  readonly name: string = 'spin/magelec';
  protected coeffKeyword(): string { return 'magelec'; }
  /** E = E.(e_ij x (s_i x s_j)) */
  protected energyOf(V: number[], e: number[], C: number[]): number { return dot(V, cross(e, C)); }
  /** omega_i = -(1/hbar) s_j x (E x e_ij) */
  protected omegaTerm(V: number[], e: number[], sj: number[]): number[] {
    const w = cross(sj, cross(V, e));
    return [-w[0] / SPIN_HBAR, -w[1] / SPIN_HBAR, -w[2] / SPIN_HBAR];
  }
  /** (1/2) M x E with M of the header (the mixed components of s_i and s_j). */
  protected directedForce(V: number[], si: number[], sj: number[]): number[] {
    const M = [si[1] * si[2] - si[2] * sj[1], si[2] * si[0] - si[0] * sj[2], si[0] * si[1] - si[1] * sj[0]];
    const w = cross(M, V);
    return [0.5 * w[0], 0.5 * w[1], 0.5 * w[2]];
  }
}
