import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';
import { tallyAtom } from './lj_cut';
import type { SimState } from '../../types';

/*
 * pair_style yukawa/colloid — docs.lammps.org/pair_yukawa_colloid.html
 * (source: plans/lammps-docs/pair_yukawa_colloid.rst)
 *
 * Syntax (verbatim):
 *    pair_style yukawa/colloid kappa cutoff
 * * kappa = screening length (inverse distance units)
 * * cutoff = global cutoff for colloidal Yukawa interactions (distance units)
 *
 * Formula (verbatim, the math block of the page):
 *    E = \frac{A}{\kappa} e^{- \kappa (r - (r_i + r_j))} \qquad r < r_c
 * "where :math:`r_i` and :math:`r_j` are the radii of the two particles
 * and :math:`r_c` is the cutoff."
 *
 * pair_coeff coefficients (verbatim):
 * * A (energy/distance units)
 * * cutoff (distance units)
 * "The last coefficient is optional.  If not specified, the global
 * yukawa/colloid cutoff is used."
 *
 * Mixing (verbatim): "For atom type pairs I,J and I != J, the A coefficient
 * and cutoff distance for this pair style can be mixed.  A is an energy value
 * mixed like a LJ epsilon.  The default mix value is *geometric*\ ."
 * The cutoff is a distance and mixes like one (as for lj/cut, whose cutoff
 * mixes with the sigma rule).
 *
 * pair_modify (verbatim): "This pair style supports the pair_modify shift
 * option for the energy of the pair interaction."; "The pair_modify table
 * option is not relevant for this pair style."; "This pair style does not
 * support the pair_modify tail option for adding long-range tail corrections
 * to energy and pressure."
 *
 * Restrictions (verbatim): "This pair style requires that atoms be
 * finite-size spheres with a diameter, as defined by the atom_style sphere
 * command."; "Per-particle polydispersity is not yet supported by this pair
 * style; per-type polydispersity is allowed.  This means all particles of
 * the same type must have the same diameter.  Each type can have a different
 * diameter."
 *
 * Default (verbatim): "none"
 *
 * Force from the documented energy: dE/dr = -A e^(-kappa (r - ri - rj)), so
 * the LAMMPS fforce (force on i = fforce * (x_i - x_j)) is
 * fforce = -(1/r) dE/dr = A e^(-kappa (r - ri - rj)) / r, repulsive.
 *
 * Implementation as a many-body style with a FULL neighbor list (per the
 * engine contract): every ordered pair (owned center i, neighbor j) is
 * evaluated once and half of the energy and half of the forces are
 * accumulated — the mirror ordered term is supplied by the other endpoint
 * (or by the owner of its periodic image), so each atom ends with its full
 * force and the global energy is counted once. Ghost radii are read through
 * nb.owner (ghost k is an image of owned atom nb.owner[k]). No global virial
 * is tallied here (virialFdotr: the force field takes sum x.f over owned and
 * ghost atoms).
 */

export class PairYukawaColloid extends Pair {
  readonly name = 'yukawa/colloid';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  kappa = 0;
  cutGlobal = 0;
  p!: PairParams;
  private aTab = new Float64Array(0);
  private offset = new Float64Array(0);
  private state: SimState | null = null;
  /** Radius of each type (per-type polydispersity only, header quote). */
  private typeRadius = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 2) throw new StyleError('usage: pair_style yukawa/colloid kappa cutoff');
    this.kappa = parseNum(args[0], 'kappa');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    if (!(this.kappa > 0)) throw new StyleError('kappa must be > 0');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['A', 'cut']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3 || args.length > 4) throw new StyleError('usage: pair_coeff I J A [cutoff]');
    const a = parseNum(args[2], 'A');
    const cut = args[3] !== undefined ? parseNum(args[3], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [a, cut]);
  }

  initStyle(ctx: StyleContext): void {
    this.state = ctx.s;
    if (this.tail) {
      throw new StyleError('pair_modify tail is not supported for pair style yukawa/colloid');
    }
    const s = ctx.s;
    if (s) {
      // "This pair style requires that atoms be finite-size spheres with a
      // diameter, as defined by the atom_style sphere command."
      if (!s.radius) throw new StyleError('pair_style yukawa/colloid requires atom_style sphere (per-atom radii)');
      // "all particles of the same type must have the same diameter"
      const tr = new Float64Array(this.ntypes + 1);
      for (let i = 0; i < s.n; i++) {
        const t = s.type[i];
        if (s.radius[i] > 0 && !(tr[t] > 0)) tr[t] = s.radius[i];
      }
      this.typeRadius = tr;
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      // "A is an energy value mixed like a LJ epsilon"; the cutoff mixes like a distance
      const a = mixEpsilon(this.mix, p.get('A', i, i), p.get('A', j, j), 1, 1);
      const c = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'A', a);
      p.setMixed(i, j, 'cut', c);
    }
    const nt = this.ntypes + 1;
    if (this.aTab.length !== nt * nt) {
      this.aTab = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
    const a = p.get('A', i, j), cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    this.aTab[k1] = this.aTab[k2] = a;
    // pair_modify shift: subtract the pair energy at the cutoff,
    // E(rc) = (A/kappa) exp(-kappa (rc - (ri + rj)))
    this.offset[k1] = this.offset[k2] = this.shift && cut > 0
      ? (a / this.kappa) * Math.exp(-this.kappa * (cut - this.typeRadius[i] - this.typeRadius[j]))
      : 0;
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const radius = pc.s.radius;
    if (!radius) throw new StyleError('pair_style yukawa/colloid requires atom_style sphere (per-atom radii)');
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, aTab, offset, kappa } = this;
    const owner = pc.nb.owner;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ri = radius[owner[i]];
      const ti = type[i] * nt;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const rj = radius[owner[j]];
        const r = Math.sqrt(rsq);
        const ex = Math.exp(-kappa * (r - ri - rj));
        const fpair = aTab[t] * ex / r; // force on i = fpair * (x_i - x_j)
        const hx = 0.5 * fpair * dx, hy = 0.5 * fpair * dy, hz = 0.5 * fpair * dz;
        f[3 * i] += hx; f[3 * i + 1] += hy; f[3 * i + 2] += hz;
        f[3 * j] -= hx; f[3 * j + 1] -= hy; f[3 * j + 2] -= hz;
        const e = (aTab[t] / kappa) * ex - offset[t];
        evdwl += 0.5 * e; // the mirror ordered term supplies the other half
        if (tally) tallyAtom(pc, i, j, 0.5 * e, 0.5 * fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    const nt = this.ntypes + 1;
    const t = itype * nt + jtype;
    if (rsq >= this.cutsq[t]) return { eng: 0, fforce: 0 };
    const r = Math.sqrt(rsq);
    const ex = Math.exp(-this.kappa * (r - this.typeRadius[itype] - this.typeRadius[jtype]));
    return {
      fforce: factorLJ * this.aTab[t] * ex / r,
      eng: factorLJ * ((this.aTab[t] / this.kappa) * ex - this.offset[t]),
    };
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${this.p.get('A', i, j)} ${this.p.get('cut', i, j)}`);
      }
    }
    return out;
  }
}
