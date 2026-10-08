import type { System } from '../system';
import { StyleError } from '../force/types';

/*
 * Base class for fixes — docs.lammps.org/fix.html: "Set a fix that will be
 * applied to a group of atoms. In LAMMPS, a "fix" is any operation that is
 * applied to the system during timestepping or minimization."
 *
 * Hooks follow the timestep of docs.lammps.org/Developer_flow.html:
 *   initial_integrate, post_integrate, [pre_exchange, pre_neighbor,
 *   post_neighbor], pre_force, pre_reverse, post_force, final_integrate,
 *   end_of_step; and min_pre_force / min_post_force during minimization.
 * setup() runs once after the setup force evaluation of a run; by default it
 * applies postForce so that step-0 forces include the fix (as for fixes that
 * modify forces in LAMMPS, e.g. setforce, addforce).
 *
 * fix_modify (docs.lammps.org/fix_modify.html): "Energy yes will add a
 * contribution to the potential energy of the system"; "Virial yes will add a
 * contribution to the virial of the system"; "For most fixes that support
 * the energy keyword, the default setting is no." temp / press select the
 * compute a fix uses.
 */

export abstract class Fix {
  abstract readonly style: string;
  readonly groupBit: number;
  /** Fix moves atoms by time integration (nve, nvt, ...). */
  timeIntegrate = false;
  /** Supports fix_modify energy / virial, and their current settings. */
  energyGlobal = false;
  virialGlobal = false;
  thermoEnergy = false;
  thermoVirial = false;
  /** Global outputs for f_ID references. */
  scalarFlag = false;
  vectorFlag = false;
  arrayFlag = false;
  sizeVector = 0;
  sizeArrayRows = 0;
  sizeArrayCols = 0;
  extscalar = 0;
  extvector = 0;
  /** Per-atom outputs. */
  peratomFlag = false;
  sizePeratomCols = 0;
  vectorAtom = new Float64Array(0);
  arrayAtom = new Float64Array(0);
  /** Steps between end_of_step calls (and output validity for averaging fixes). */
  nevery = 1;
  /** Virial contribution (xx yy zz xy xz yz) when thermoVirial is on. */
  virial = new Float64Array(6);
  /** Ramp parameters use the current run's start and stop. */

  constructor(protected sys: System, readonly id: string, readonly group: string, protected args: string[]) {
    this.groupBit = sys.groups.bit(group);
  }

  init?(): void;
  setup(): void { this.postForce?.(); }
  minSetup?(): void;
  initialIntegrate?(): void;
  postIntegrate?(): void;
  preExchange?(): void;
  preNeighbor?(): void;
  postNeighbor?(): void;
  preForce?(): void;
  preReverse?(): void;
  postForce?(): void;
  finalIntegrate?(): void;
  endOfStep?(): void;
  postRun?(): void;
  minPreForce?(): void;
  minPostForce?(): void;
  /** Called when the timestep size changes (fix dt/reset, timestep). */
  resetDt?(): void;
  /** The box changed size or shape (barostats, deform): rescale stored geometry. */
  boxChanged?(): void;
  /** unfix: remove anything the fix created (its own computes). */
  destroy?(): void;
  /** Cumulative energy exchanged with a thermostat/barostat reservoir (thermo ecouple). */
  ecouple?(): number;

  /** Global energy contribution for compute pe when fix_modify energy yes. */
  energy(): number { return 0; }
  /**
   * Adds this fix's per-atom energy (owned atoms) to out, for compute pe/atom fix (with fix_modify
   * energy yes). compute_pe_atom.html: "Various fixes can contribute to the per-atom potential
   * energy of the system if the *fix* contribution is included."
   */
  energyAtom?(out: Float64Array): void;
  /** Adds this fix's per-atom virial (6 per owned atom) to out, for compute stress/atom fix (fix_modify virial yes). */
  virialAtom?(out: Float64Array): void;
  /** f_ID scalar. */
  computeScalar(): number { throw new StyleError(`fix ${this.id} does not compute a global scalar`); }
  /** f_ID[i] (0-based i). */
  computeVector(_i: number): number { throw new StyleError(`fix ${this.id} does not compute a global vector`); }
  computeArray(_i: number, _j: number): number { throw new StyleError(`fix ${this.id} does not compute a global array`); }
  /** Degrees of freedom this fix removes from atoms in the given group (shake, rigid). */
  dofRemoved(_groupBit: number): number { return 0; }

  /** fix_modify; returns values consumed (0 = keyword not handled). */
  modify(key: string, values: string[]): number {
    const yesno = (w: string | undefined) => {
      if (w !== 'yes' && w !== 'no') throw new StyleError(`fix_modify ${key} must be yes or no`);
      return w === 'yes';
    };
    if (key === 'energy') {
      if (!this.energyGlobal) throw new StyleError(`fix ${this.id} (${this.style}) does not support fix_modify energy`);
      this.thermoEnergy = yesno(values[0]);
      return 1;
    }
    if (key === 'virial') {
      if (!this.virialGlobal) throw new StyleError(`fix ${this.id} (${this.style}) does not support fix_modify virial`);
      this.thermoVirial = yesno(values[0]);
      return 1;
    }
    return 0;
  }

  /** True if atom i is in this fix's group. */
  protected inGroup(i: number): boolean {
    return (this.sys.state.mask[i] & this.groupBit) !== 0;
  }
}
