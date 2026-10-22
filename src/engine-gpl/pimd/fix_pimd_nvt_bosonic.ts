/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/fix_pimd_nvt_bosonic.cpp and src/REPLICA/fix_pimd_nvt_bosonic.h (FixPIMDBNVT: the overrides of
 * prepare_coordinates, spring_force, pre_spring_force_estimators and compute_vector, and the four outputs) at commit
 * b332f28. Authors of the source: Ofir Blumer, Jacob Higer, Yotam Feldman and Barak Hirshberg (Hirshberg lab, Tel Aviv
 * University). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_pimd.html (pimd/nvt/bosonic): the bosonic variant of fix pimd/nvt; its spring forces are the
 * exchange-aware ones of BosonicExchange (bosonic_exchange.ts), and compute_vector has four outputs: the spring energy,
 * T_ring, the virial and the primitive kinetic estimator. Beads are the partitions as in fix_pimd_nvt.ts.
 */
import { FixPIMDNVT } from './fix_pimd_nvt';
import { BosonicExchange } from './bosonic_exchange';
import type { System } from '../../engine/system';

export class FixPIMDNVTBosonic extends FixPIMDNVT {
  readonly style: string = 'pimd/nvt/bosonic';
  sizeVector = 4;
  private exchange: BosonicExchange | null = null;
  private primValue = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'pimd/nvt/bosonic');
  }

  init(): void {
    super.init();
    const s = this.sys.state;
    // bead_num is the bead's index (native: universe->me, one process per bead); the bosons are the local atoms
    this.exchange = new BosonicExchange(
      s.n, this.np, this.iworld,
      (d, dim) => this.minImage(d, dim), 3, true,
    );
  }

  /** The bead's exchange object (created by init, once per run). */
  private get ex(): BosonicExchange {
    if (!this.exchange) throw new Error('pimd/nvt/bosonic: exchange used before init');
    return this.exchange;
  }

  /** A per-atom array (by atom ID, see tagArray) re-indexed to this bead's local atoms. */
  private toLocal(tagged: Float64Array): Float64Array {
    const s = this.sys.state;
    const out = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) out[3 * i + d] = tagged[3 * (s.id[i] - 1) + d];
    return out;
  }

  /** native prepare_coordinates: the exchange of the exterior beads, with the spring constant -fbond * m (of atom 0's type). */
  protected prepareCoordinates(xPrev: Float64Array, xNext: Float64Array): void {
    const s = this.sys.state;
    const ff = this.fbond * s.massByType[s.type[0]];
    this.ex.prepare(s.x, this.toLocal(xPrev), this.toLocal(xNext), this.beta, -ff);
  }

  /** native pre_spring_force_estimators: the virial, the bead spring energy and the primitive estimator. */
  protected preSpringEstimators(): void {
    super.preSpringEstimators();
    this.springEnergy = this.ex.beadSpringEnergy();
    this.primValue = this.ex.primEstimator();
  }

  /** native spring_force: the exchange-aware springs (no neighbour arguments: the prepared copies are used). */
  protected springForce(_xPrev: Float64Array, _xNext: Float64Array): void {
    this.ex.springForce(this.sys.state.f);
  }

  computeVector(i: number): number {
    if (i >= 0 && i < 3) return super.computeVector(i);
    if (i === 3) return this.primValue;
    throw new Error('Fix only has 4 outputs!');
  }
}
