/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/fix_pimd_langevin_bosonic.cpp and src/REPLICA/fix_pimd_langevin_bosonic.h (FixPIMDBLangevin: the
 * keyword checks of the constructor, the esynch keyword, prepare_coordinates, spring_force, compute_spring_energy,
 * compute_t_prim and compute_vector with six outputs) at commit b332f28. Authors of the source: Ofir Blumer, Jacob Higer,
 * Yotam Feldman and Barak Hirshberg (Hirshberg lab, Tel Aviv University). Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_pimd.html (pimd/langevin/bosonic): the bosonic variant of fix pimd/langevin (method pimd only, the
 * physical fictitious mass, the nve and nvt ensembles, no barostat). Its springs are the exchange-aware ones of
 * BosonicExchange (bosonic_exchange.ts), built with the reduced beta convention (physical_beta_convention false, as
 * fix_pimd_langevin_bosonic.cpp passes).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { FixPIMDLangevin } from './fix_pimd_langevin';
import { BosonicExchange } from './bosonic_exchange';

export class FixPIMDLangevinBosonic extends FixPIMDLangevin {
  readonly style: string = 'pimd/langevin/bosonic';
  private exchange: BosonicExchange | null = null;
  private esynch = true;
  private nbosons: number;
  private fTagOrder = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    // walk the keywords as the base parser does (thermostat PILE_L takes its seed as a third word)
    const starts: number[] = [];
    for (let i = 0; i < args.length - 1; ) {
      starts.push(i);
      i += args[i] === 'thermostat' && args[i + 1] === 'PILE_L' ? 3 : 2;
    }
    let esynch = true;
    const drop = new Set<number>();
    for (const i of starts) {
      const key = args[i], val = args[i + 1];
      if (key === 'method' && val !== 'pimd') throw new StyleError('Method not supported in fix pimdb/langevin; only method PIMD');
      if (key === 'scale') throw new StyleError('The scale parameter of the PILE_L thermostat is not supported for pimdb, and should be removed.');
      if (key === 'iso' || key === 'aniso' || key === 'barostat' || key === 'taup') throw new StyleError('Barostat parameters are not available for pimdb.');
      if (key === 'esynch') {
        if (val === 'yes') esynch = true;
        else if (val === 'no') esynch = false;
        else throw new StyleError('The esynch parameter can only receive yes or no!');
        drop.add(i);
      }
    }
    // the esynch keyword is not a keyword of the base fix: its pair is removed before the base parser runs
    const base: string[] = [];
    for (let i = 0; i < args.length; i++) {
      if (drop.has(i) || drop.has(i - 1)) continue;
      base.push(args[i]);
    }
    super(sys, id, group, base);
    this.esynch = esynch;
    if (this.fmmode !== 'physical') throw new StyleError('The only available fmmode for pimdb is physical, please remove the fmmode keyword.');
    if (this.ensemble !== 'nve' && this.ensemble !== 'nvt') throw new StyleError('The only available ensembles for pimdb are nve and nvt, please choose one of these ensembles.');
    this.method = 'pimd';
    this.sizeVector = 6;
    this.nbosons = sys.state.n;
  }

  init(): void {
    super.init();
    const s = this.sys.state;
    this.nbosons = s.n;
    this.fTagOrder = new Float64Array(3 * this.nbosons);
    // bead_num is the bead's index (native: universe->me, one process per bead); the bosons are the local atoms
    this.exchange = new BosonicExchange(
      this.nbosons, this.np, this.iworld,
      (d, dim) => this.minImageOf(d, dim), 3, false,
    );
  }

  private minImageOf(dv: number, d: number): number {
    const box = this.sys.state.box;
    if (!box.periodic[d]) return dv;
    const L = box.hi[d] - box.lo[d];
    return dv - L * Math.round(dv / L);
  }

  private get ex(): BosonicExchange {
    if (!this.exchange) throw new Error('pimd/langevin/bosonic: exchange used before init');
    return this.exchange;
  }

  /** A per-atom tagged array re-indexed to the bead's local atoms. */
  private toLocal(tagged: Float64Array): Float64Array {
    const s = this.sys.state;
    const out = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) out[3 * i + d] = tagged[3 * (s.id[i] - 1) + d];
    return out;
  }

  /** prepare_coordinates: the exchange of the exterior beads, with the spring constant of atom 0's type (beta_np). */
  protected prepareBead(xt: Float64Array[]): void {
    const np = this.np;
    const s = this.sys.state;
    const ff = this.fbond * s.massByType[s.type[0]];
    this.ex.prepare(s.x, this.toLocal(xt[(this.iworld - 1 + np) % np]), this.toLocal(xt[(this.iworld + 1) % np]), this.betaNp, ff);
  }

  /** spring_force: the exchange-aware springs, added to the forces by atom tag (native mapping). */
  protected springForceBead(b: FixPIMDLangevin): void {
    const self = b as FixPIMDLangevinBosonic;
    const s = self.sys.state;
    self.fTagOrder.fill(0);
    self.ex.springForce(self.fTagOrder);
    for (let i = 0; i < self.nbosons; i++) {
      const t = s.id[i] - 1;
      for (let d = 0; d < 3; d++) s.f[3 * i + d] += self.fTagOrder[3 * t + d];
    }
  }

  /** compute_spring_energy: the bead's exchange energy, summed over the beads when esynch is yes. */
  protected computeSpringEnergy(R: FixPIMDLangevin[]): void {
    const beads = R as FixPIMDLangevinBosonic[];
    for (const b of beads) b.seBead = b.ex.beadSpringEnergy();
    let total = 0;
    for (const b of beads) total += b.seBead;
    for (const b of beads) b.totalSpring = this.esynch ? total : 0;
  }

  /** compute_t_prim: the primitive estimator, summed over the beads when esynch is yes. */
  protected computeTPrim(R: FixPIMDLangevin[]): void {
    const beads = R as FixPIMDLangevinBosonic[];
    const prim = beads.map((b) => b.ex.primEstimator());
    let total = 0;
    for (const p of prim) total += p;
    beads.forEach((b, k) => { b.tPrim = this.esynch ? total : prim[k]; });
  }

  computeVector(n: number): number {
    if (n >= 0 && n < 6) return super.computeVector(n);
    throw new StyleError('Fix only has 6 outputs!');
  }
}
