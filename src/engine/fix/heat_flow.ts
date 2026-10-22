import { Fix } from './fix';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';
import { HEAT_FLOW, HEAT_TEMPERATURE, massOf } from '../atoms';

/*
 * fix ID group-ID heat/flow style values ... — docs.lammps.org/fix_heat_flow.html (plans/lammps-docs/fix_heat_flow.rst):
 *   "Perform plain time integration to update temperature for atoms in the group each timestep."
 *   "For style *constant*, the specific heat is a constant value *cp* for all atoms."
 *   "For style *type*, *N* different values of the specific heat are defined, one for each of the *N* types of atoms."
 *   "This fix requires that atoms store temperature and heat flow as defined by the fix property/atom command."
 *
 * Integration (dT/dt = Q / (m cp) with the heat flow Q of the pair and wall heat models):
 * Measured with native LAMMPS (black box): the temperature of atom i changes by dt Q_i / (m_i cp) in every step, with
 * Q_i the heatflow of the step's own force evaluation (not the previous step's). The update therefore happens after
 * the force (final integrate), while the heat flow is zeroed before the force (initial integrate).
 * Measured with native LAMMPS (black box): m_i is the per-atom mass (rmass, density 2 gives twice the mass);
 * cp of style type is looked up by the atom type (cp 1 for type 1 and 4 for type 2 gave the matching changes).
 * Measured with native LAMMPS (black box): heatflow is zeroed for every atom, also for atoms outside the group;
 * the temperature of atoms outside the group does not change.
 * Measured with native LAMMPS (black box): constant cp = 0 is accepted and gives an infinite temperature change (inf / -inf).
 * Measured with native LAMMPS (black box): a negative cp is refused with Illegal fix heat/flow constant command value,
 * the type style needs exactly one value per atom type (native: Illegal fix heat/flow type command) and a negative value is refused.
 * Measured with native LAMMPS (black box): without fix property/atom temperature heatflow the fix stops with
 * Fix heat/flow requires atom style with temperature property (native error text).
 * Not invoked during energy minimization (the doc page) - the engine has no minimizer.
 */
export class FixHeatFlow extends Fix {
  readonly style = 'heat/flow';
  /** Specific heat: one value (constant) or one per atom type (type; index = type - 1). */
  private readonly cp: number[];
  private readonly perType: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/GRANULAR/fix_heat_flow.cpp at b332f28): constant cp, or type with one cp per type
    const style = args[0];
    if (style === 'constant') {
      if (args.length !== 2) throw new StyleError('Illegal fix heat/flow constant command');
      this.cp = [parseNum(args[1], 'heat/flow constant cp')];
      this.perType = false;
    } else if (style === 'type') {
      if (args.length !== 1 + sys.state.ntypes) throw new StyleError('Illegal fix heat/flow type command');
      this.cp = args.slice(1).map((w) => parseNum(w, 'heat/flow type cp'));
      this.perType = true;
    } else {
      throw new StyleError(`Unknown fix heat/flow keyword ${style}`);
    }
    for (const c of this.cp) {
      if (c < 0) throw new StyleError(`Illegal fix heat/flow ${style} command value`);
    }
    if (sys.hasBox) this.check();
  }

  /** The atom style needs the temperature and heatflow properties, and type cps need one value per type. */
  private check(): void {
    const s = this.sys.state;
    if (!s.custom.has(HEAT_TEMPERATURE) || !s.custom.has(HEAT_FLOW)) {
      throw new StyleError('Fix heat/flow requires atom style with temperature property (define fix property/atom temperature heatflow)');
    }
    if (this.perType && this.cp.length !== s.ntypes) {
      throw new StyleError('Illegal fix heat/flow type command');
    }
  }

  init(): void { this.check(); }

  /** Heat flow of every atom is zeroed before the force evaluation of the step (measured, see the module comment). */
  initialIntegrate(): void {
    this.sys.state.custom.get(HEAT_FLOW)!.data.fill(0);
  }

  /** dT = dt Q / (m cp) for the atoms of the group. */
  finalIntegrate(): void {
    const s = this.sys.state;
    const T = s.custom.get(HEAT_TEMPERATURE)!.data;
    const Q = s.custom.get(HEAT_FLOW)!.data;
    const dt = s.dt;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const cp = this.perType ? this.cp[s.type[i] - 1] : this.cp[0];
      T[i] += (dt * Q[i]) / (massOf(s, i) * cp);
    }
  }
}
