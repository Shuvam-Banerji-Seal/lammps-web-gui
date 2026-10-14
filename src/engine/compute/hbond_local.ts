import { Compute } from './compute';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import { buildAtomMap, buildSpecial, type SpecialList } from '../atoms';
import { NEIGHMASK } from '../neighbor';
import type { System } from '../system';
import type { ImageObject } from '../output/image_objects';

/*
 * compute hbond/local (docs.lammps.org/compute_hbond_local.html). Syntax, from the page:
 *   "compute ID group-ID hbond/local rcut acut dgroup-ID agroup-ID hgroup-ID value1 value2 ..."
 * Conditions for a hydrogen bond, from the page's list: "the hydrogen atom has to be connected to the
 * donor with a bond"; "the donor - acceptor distance has to be less than *rcut*"; "the hydrogen - donor -
 * acceptor angle has to be less than *acut*"; and "all three atoms have to be in the compute group".
 *
 * Measured with native LAMMPS (black box):
 *  - the distance test is inclusive: a donor-acceptor distance of exactly 2.5 with rcut 2.5 is counted;
 *  - the angle test is done in double precision: the H-D-A geometry H (1,0,0), D (0,0,0), A (2,2,0) has
 *    the angle 45.000000000000007 degrees, which is not counted with acut 45.0 and is counted with acut
 *    45.0000001;
 *  - the angle vertex is the donor (the 45-degree geometry above reports 45 degrees at D);
 *  - the bonded hydrogen may be listed in either bond order (H-D or D-H);
 *  - an unbonded hydrogen within the cutoffs is not counted;
 *  - the count is the number of such triples, a global scalar that thermo normalises in lj units (extensive).
 *
 * Per-bond columns (order fixed by the command line): H id, D id, A id, then the values:
 *  "The *dist* value is the current distance between the hydrogen bond donor and acceptor atom in
 *  distance units"; "The *angle* value is the current hydrogen-donor-acceptor angle in degrees"; "The
 *  *hdist* value is the current distance between the hydrogen atom and hydrogen bond acceptor atom in
 *  distance units"; "The *ehb* value is the hydrogen bond strength computed as the sum of the pairwise
 *  potential energy between a) the donor atom and the acceptor atom, and b) the hydrogen atom and the
 *  acceptor atom.  This is a positive value for an attractive interaction."
 * Measured with native LAMMPS (black box): ehb = -(E_DA + E_HA) with the pair style's energies (LJ and
 * Coulomb; for coul/long the real-space part only); a pair beyond the pair cutoff contributes 0 (pair
 * lj/cut/coul/cut 2.0 with an H-O pair at 1.8 and a D-A pair at 2.8 gives ehb 66.2675, the H-O term
 * alone); the special_bonds weights do not change ehb (1-2 and 1-3 pairs at weight 0.5 and at weight 1
 * gave the same ehb); an acceptor equal to the hydrogen gives hdist 0 and ehb nan.
 *
 * ecut: "If the *ecut* keyword is used, an additional energy cutoff is applied." and "The computed hydrogen
 * bond strength must be larger than the *ecut* value". Measured with native LAMMPS (black box): ecut applies
 * even without the ehb column; a nan ehb is not larger than ecut and the bond is dropped; ecut < 0 is refused.
 *
 * Acceptors come from the neighbour list. Measured with native LAMMPS (black box): with special_bonds
 * lj/coul 0 0 0 and a plain pair style, the 1-2 and 1-3 partners of the donor are not acceptors; with
 * lj/coul 1 1 1, or with lj 1 1 1 coul 0 0 0, they are (a bonded hydrogen as its own donor's acceptor
 * gives dist 1, angle 0); with a kspace style and lj/cut/coul/long, excluded pairs stay in the list at
 * weight 0, so they are acceptors again. The rule here: a special partner whose lj and coul weights are
 * both 0 is skipped unless the force field keeps excluded pairs (kspace, coul/long or keepExcluded), as
 * the force field's neighbour list does.
 *
 * Neighbour-list policy: native's count depends on when its list is rebuilt. Measured with native LAMMPS
 * (black box) on examples/rdf-adf data.spce with the example's pair style: step 10 counts 2572 with
 * neigh_modify every 1 delay 2 check yes and 2579 with every 1 delay 0 check no. This engine builds the
 * list on each evaluation, so it reproduces the second.
 *
 * Refusals, measured with native LAMMPS (black box): rcut must be > 0; acut must be > 0 and not above 90
 * degrees; ecut must be >= 0; a missing group is an error; values must
 * come before keywords (ehb after ecut is an unknown keyword); with no value keywords the output is three
 * columns; repeated values give repeated columns.
 *
 * Row order is this engine's donor order. The page says "there will be no consistent ordering of the
 * entries within the local array from one timestep to the next", so oracle cases compare order-free
 * aggregates. The MPI rule of the page (count on the donor's process) holds trivially on one process.
 */

type Value = 'dist' | 'angle' | 'hdist' | 'ehb';
const VALUES: readonly Value[] = ['dist', 'angle', 'hdist', 'ehb'];
const RAD2DEG = 180 / Math.PI;

const norm = (d: number[]): number => Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

export class ComputeHbondLocal extends Compute {
  readonly style = 'hbond/local';
  private readonly rcut: number;
  private readonly acut: number;
  private readonly dGroup: string;
  private readonly aGroup: string;
  private readonly hGroup: string;
  private readonly dBit: number;
  private readonly aBit: number;
  private readonly hBit: number;
  private readonly values: Value[];
  private readonly ecut: number | null;
  private readonly needEnergy: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const usage = `compute ${id} hbond/local: missing argument(s) (usage: compute ID group-ID hbond/local rcut acut dgroup-ID agroup-ID hgroup-ID value ... ecut value)`;
    if (args.length < 5) throw new StyleError(usage);
    this.rcut = parseNum(args[0], 'distance cutoff');
    if (!(this.rcut > 0)) throw new StyleError(`compute ${id} hbond/local: distance cutoff must be > 0.0`);
    this.acut = parseNum(args[1], 'angle cutoff');
    if (!(this.acut > 0)) throw new StyleError(`compute ${id} hbond/local: angle cutoff must be > 0.0`);
    if (this.acut > 90) throw new StyleError(`compute ${id} hbond/local: angle cutoff must not be > 90.0 degrees`);
    this.dGroup = args[2];
    this.aGroup = args[3];
    this.hGroup = args[4];
    this.dBit = sys.groups.bit(this.dGroup);
    this.aBit = sys.groups.bit(this.aGroup);
    this.hBit = sys.groups.bit(this.hGroup);
    let k = 5;
    const values: Value[] = [];
    for (; k < args.length && (VALUES as readonly string[]).includes(args[k]); k++) values.push(args[k] as Value);
    let ecut: number | null = null;
    while (k < args.length) {
      const w = args[k];
      if (w !== 'ecut') throw new StyleError(`Unknown compute hbond/local keyword ${w}`);
      if (k + 1 >= args.length) throw new StyleError(`Illegal compute hbond/local ecut command: missing argument(s)`);
      ecut = parseNum(args[k + 1], 'ecut');
      if (!(ecut >= 0)) throw new StyleError(`compute ${id} hbond/local: hydrogen bond strength cutoff ecut must be >= 0.0`);
      k += 2;
    }
    this.values = values;
    this.ecut = ecut;
    this.needEnergy = values.includes('ehb') || ecut !== null;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.localFlag = true;
    this.sizeLocalCols = 3 + values.length;
  }

  /** The arrows of dump image's "compute" keyword, one per hydrogen bond (compute_image of ComputeHBondLocal). */
  private collectImage = false;
  private imageArrows: { type: number; center: [number, number, number]; dir: [number, number, number]; length: number }[] = [];

  imageObjects(): ImageObject[] {
    this.imageArrows = [];
    this.collectImage = true;
    try {
      this.computeLocal();
    } finally {
      this.collectImage = false;
    }
    // native: ARROW with the donor's type, the tip ratio 0.2 and a diameter of 0 (the keyword's flag2 is added to it)
    return this.imageArrows.map((a): ImageObject => ({ kind: 'arrow', type: a.type, center: a.center, dir: a.dir, length: a.length, diameter: 0, ratio: 0.2 }));
  }

  protected computeScalar(): number {
    this.localValues();
    return this.localRows;
  }

  protected computeLocal(): Float64Array<ArrayBuffer> {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const geom = sys.geom;
    const ff = sys.ff;
    const pair = ff.pair;
    const groupBit = this.groupBit;
    const id = this.id;
    if (this.needEnergy && (!pair || !pair.single || pair.singleNeedsSeparation())) {
      throw new StyleError(`compute ${id} hbond/local: computation of the hydrogen bond energy is not supported by the pair style`);
    }
    if (nb.cutghost > 0 && nb.cutghost < this.rcut + nb.skin - 1e-9) {
      throw new StyleError(`compute ${id} hbond/local: distance cutoff ${this.rcut} exceeds the ghost cutoff ${nb.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
    }
    // ghost and owned positions are synchronised first: a setup-time correction (fix shake) moves owned atoms
    nb.forwardComm(s, sys.geom);
    // the pair energies (ehb, ecut) need the force evaluation's Coulomb prefactor, as pair/local does
    if (this.needEnergy) sys.forces();
    const map = buildAtomMap(s);
    // 1-2 partners (bonded atoms, either bond order), as owned indices
    const partners: number[][] = Array.from({ length: s.n }, () => []);
    const b = s.topo.bonds;
    for (let e = 0; e < b.n; e++) {
      if (b.type[e] === 0) continue;
      const i = map[b.atoms[2 * e]], j = map[b.atoms[2 * e + 1]];
      if (i < 0 || j < 0) continue;
      partners[i].push(j);
      partners[j].push(i);
    }
    const sp: SpecialList | null = b.n > 0 ? buildSpecial(s, { angle: ff.special.angle, dihedral: ff.special.dihedral }) : null;
    const sb = ff.special;
    const keepExcluded = !!ff.kspace || !!pair?.coulLong || !!pair?.keepExcluded;
    const cutsq = pair?.cutsq;
    const nt = s.ntypes + 1;
    const list = nb.occasionalList(s, this.rcut, true, s.ntypes);
    const xa = nb.xall, ta = nb.typeall, qa = nb.qall, owner = nb.owner;
    const box = s.box;
    const hx = 0.5 * (box.hi[0] - box.lo[0]), hy = 0.5 * (box.hi[1] - box.lo[1]), hz = 0.5 * (box.hi[2] - box.lo[2]);
    const per = box.periodic;
    const rows: number[] = [];
    let nrows = 0;

    // pair energy of atoms i (owned) and j (owned or ghost) at squared distance rsq, weight 1
    const pe = (i: number, j: number, rsq: number): number => {
      if (!cutsq || !pair?.single || pair.singleNeedsSeparation()) return 0;
      const ti = ta[i], tj = ta[j];
      if (rsq >= cutsq[ti * nt + tj]) return 0;
      return pair.single(i, owner[j], ti, tj, rsq, 1, 1, qa[i], qa[j]).eng;
    };

    for (let d = 0; d < list.inum; d++) {
      if (!(s.mask[d] & this.dBit) || !(s.mask[d] & groupBit)) continue;
      const hs = partners[d];
      if (!hs.length) continue;
      const xd0 = xa[3 * d], xd1 = xa[3 * d + 1], xd2 = xa[3 * d + 2];
      for (const h of hs) {
        if (!(s.mask[h] & this.hBit) || !(s.mask[h] & groupBit)) continue;
        // the hydrogen as the bonded image nearest the donor
        const dh = [xa[3 * h] - xd0, xa[3 * h + 1] - xd1, xa[3 * h + 2] - xd2];
        geom.minimumImage(dh);
        const rH = norm(dh);
        if (!(rH > 0)) continue;
        const xh = [xd0 + dh[0], xd1 + dh[1], xd2 + dh[2]];
        const k0 = list.firstneigh[d], k1 = k0 + list.numneigh[d];
        for (let kk = k0; kk < k1; kk++) {
          const j = list.neighbors[kk] & NEIGHMASK;
          const a = owner[j];
          if (!(s.mask[a] & this.aBit) || !(s.mask[a] & groupBit)) continue;
          const d2 = [xa[3 * j] - xd0, xa[3 * j + 1] - xd1, xa[3 * j + 2] - xd2];
          const rDA = norm(d2);
          if (!(rDA > 0) || !(rDA <= this.rcut)) continue;
          // special partners of D, unless the image is far (the neighbour list rule)
          let excluded = false;
          if (sp) {
            const farImage = (per[0] && Math.abs(d2[0]) > hx) || (per[1] && Math.abs(d2[1]) > hy) || (per[2] && Math.abs(d2[2]) > hz);
            if (!farImage) {
              for (let q = sp.offset[d]; q < sp.offset[d + 1]; q++) {
                if (sp.partner[q] !== s.id[a]) continue;
                const o = sp.order[q] - 1;
                if (sb.lj[o] === 0 && sb.coul[o] === 0 && !keepExcluded) excluded = true;
                break;
              }
            }
          }
          if (excluded) continue;
          const cosA = (dh[0] * d2[0] + dh[1] * d2[1] + dh[2] * d2[2]) / (rH * rDA);
          const theta = Math.acos(Math.max(-1, Math.min(1, cosA))) * RAD2DEG;
          if (!(theta <= this.acut)) continue;
          const hv = [xa[3 * j] - xh[0], xa[3 * j + 1] - xh[1], xa[3 * j + 2] - xh[2]];
          const hdist = norm(hv);
          let ehb = NaN;
          if (this.needEnergy) {
            const eDA = pe(d, j, rDA * rDA);
            const eHA = pe(h, j, hdist * hdist);
            ehb = -(eDA + eHA);
          }
          if (this.ecut !== null && !(ehb > this.ecut)) continue;
          rows.push(s.id[h], s.id[d], s.id[a]);
          if (this.collectImage) {
            // compute_image: the acceptor's image nearest the donor (xd + d2) minus the hydrogen's (xh); the arrow sits at
            // their midpoint and points along the unit vector between them
            const xacc: [number, number, number] = [xd0 + d2[0], xd1 + d2[1], xd2 + d2[2]];
            const vec = [xacc[0] - xh[0], xacc[1] - xh[1], xacc[2] - xh[2]];
            const len = norm(vec);
            this.imageArrows.push({
              type: s.type[d],
              center: [0.5 * (xacc[0] + xh[0]), 0.5 * (xacc[1] + xh[1]), 0.5 * (xacc[2] + xh[2])],
              dir: [vec[0] / len, vec[1] / len, vec[2] / len],
              length: len,
            });
          }
          for (const v of this.values) {
            rows.push(v === 'dist' ? rDA : v === 'angle' ? theta : v === 'hdist' ? hdist : ehb);
          }
          nrows++;
        }
      }
    }
    this.localRows = nrows;
    return Float64Array.from(rows);
  }
}
