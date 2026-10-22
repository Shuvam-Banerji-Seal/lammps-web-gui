import type { System } from '../system';
import { StyleError } from '../force/types';
import { inumeric, logical } from '../force/util';
import type { ImageObject } from '../output/image_objects';
import type { Compute } from '../compute/compute';

/**
 * The fix styles native LAMMPS gives a rRESPA level, which fix_modify respa sets (respa_level_support = 1 in the fix's
 * source, or its base class's, at b332f28). The engine's run_style respa runs none of them, so the level only has to
 * be checked.
 */
const RESPA_LEVEL_STYLES = new Set([
  'addforce', 'addtorque', 'addtorque/atom', 'addtorque/group', 'align/self', 'amoeba/bitorsion', 'amoeba/pitorsion',
  'aveforce', 'cmap', 'damping/cundall', 'drag', 'efield', 'efield/lepton', 'efield/tip4p', 'flow/gauss', 'gravity',
  'indent', 'numdiff', 'numdiff/virial', 'orient/bcc', 'orient/fcc', 'pafi', 'precession/spin', 'propel/self',
  'qeq/comb', 'restrain', 'setforce', 'setforce/spin', 'settorque/atom', 'smd', 'spring', 'spring/chunk', 'spring/rg',
  'spring/self', 'viscous', 'viscous/nonlinear', 'viscous/sphere', 'wall/colloid', 'wall/ees', 'wall/harmonic',
  'wall/harmonic/outside', 'wall/harmonic/returned', 'wall/lepton', 'wall/lj1043', 'wall/lj126', 'wall/lj93',
  'wall/morse', 'wall/region', 'wall/table',
]);

/**
 * The compute a fix_modify temp or press value names (fix_modify.html: "temp value = compute ID that calculates a
 * temperature", "press value = compute ID that calculates a pressure"), with the messages of the fix's native
 * modify_param, which each fix words in its own way ({} is the compute ID).
 */
export const fixModifyCompute = (sys: System, id: string | undefined, kind: 'temp' | 'press', notFound: string, wrongKind: string): Compute => {
  if (id === undefined) throw new StyleError('Illegal fix_modify command');
  const c = sys.computes.find((x) => x.id === id);
  if (!c) throw new StyleError(notFound.replace('{}', id));
  if (kind === 'temp' ? !c.tempFlag : !c.pressFlag) throw new StyleError(wrongKind.replace('{}', id));
  return c;
};

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
  /**
   * Whether this fix allows a dynamic group, when it depends on the arguments (native dynamic_group_allow set under a
   * condition, e.g. fix reaxff/species with Nrepeat 1); undefined: the style's entry in group.ts decides.
   */
  dynamicGroupAllow?: boolean;
  /**
   * Coupled to the same fix on the other replicas (fix pimd: the beads of a path integral): the session runs the
   * replicas in lockstep (run/coupled.ts) and the fix reads them through System.replicaPeers.
   */
  replicaCoupled = false;
  /** Supports fix_modify energy / virial, and their current settings. */
  energyGlobal = false;
  virialGlobal = false;
  thermoEnergy = false;
  thermoVirial = false;
  /**
   * fix_modify dynamic/dof (native Fix::dynamic: "recount atoms for temperature computes"). Stored; the engine's fixes
   * count their group's atoms when they need them (measured with native LAMMPS for fix rigid/small: dynamic/dof yes
   * leaves the thermo temperature unchanged).
   */
  dynamicDof = false;
  /** Global outputs for f_ID references. */
  scalarFlag = false;
  vectorFlag = false;
  arrayFlag = false;
  sizeVector = 0;
  sizeArrayRows = 0;
  sizeArrayCols = 0;
  extscalar = 0;
  extvector = 0;
  /** Per-element extensive (1) or intensive (0) flags of a global vector, when they differ (fix ave/time). */
  extlist: number[] | null = null;
  /** 1 = every element of the global array is extensive (thermo f_ID[I][J] is normalized), 0 = intensive. */
  extarray = 0;
  /** Per-atom outputs. */
  peratomFlag = false;
  sizePeratomCols = 0;
  vectorAtom = new Float64Array(0);
  arrayAtom = new Float64Array(0);
  /**
   * Local outputs (one row per local datum, as a compute's: localFlag, sizeLocalCols 0 = vector, localRows
   * the rows valid after the last post-force step). Read by f_ID references in dump local and fix ave/histo.
   */
  localFlag = false;
  sizeLocalCols = 0;
  localRows = 0;
  localValues(): Float64Array<ArrayBuffer> {
    throw new StyleError(`fix ${this.id} does not calculate local values`);
  }
  /**
   * Per-grid data (grid.ts GridSource): the grids and data sets other commands read as f_ID:gname:dname, valid on
   * multiples of pergridFreq. A fix that sets pergridFlag overrides the four accessors.
   */
  pergridFlag = false;
  pergridFreq = 1;
  gridByName(_name: string): number { return -1; }
  gridSize(_igrid: number): [number, number, number] { return [0, 0, 0]; }
  griddataByName(_igrid: number, _name: string): { index: number; ncol: number } | null { return null; }
  griddata(_index: number): Float64Array { throw new StyleError(`fix ${this.id} does not compute per-grid info`); }
  /** Steps between end_of_step calls (and output validity for averaging fixes). */
  nevery = 1;
  /** Steps on which the global output is valid (0: any step); thermo output must use compatible steps. */
  globalFreq = 0;
  /** Virial contribution (xx yy zz xy xz yz) when thermoVirial is on. */
  virial = new Float64Array(6);
  /** Ramp parameters use the current run's start and stop. */

  constructor(protected sys: System, readonly id: string, readonly group: string, protected args: string[]) {
    this.groupBit = sys.groups.bit(group);
  }

  init?(): void;
  /**
   * Called by System.setupNeighbors after the force field is initialized (kspace init) and the neighbor
   * lists are built, before the first force evaluation of a run. A fix that must change per-atom data only
   * after the force styles read the init-time values (fix adapt charges) does it here.
   */
  afterForceInit?(): void;
  setup(): void { this.postForce?.(); }
  minSetup?(): void;
  initialIntegrate?(): void;
  postIntegrate?(): void;
  preExchange?(): void;
  preNeighbor?(): void;
  /**
   * Forces a neighbor list rebuild on this step (a fix that moves atoms outside the integrator asks for one, as fix
   * srd does after its SRD steps). Checked before the neighbor decision, like a restart file.
   */
  reneighborDue?(step: number): boolean;
  postNeighbor?(): void;
  preForce?(): void;
  /** Run once at the start of a run, before the setup force evaluation (the solve of a charge model that
   *  sets the charges the forces use); preForce's rule for the step's nevery applies in the fix itself. */
  setupPreForce?(): void;
  /** Modify::setup_pre_reverse: the setup force evaluation's pre_reverse (before any fix setup, as native). */
  setupPreReverse?(): void;
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
  /**
   * Rigid-body centres of mass under fix deform with remap x (fix_rigid.cpp FixRigid::deform, called by fix_deform.cpp):
   * flag 0 maps each body's xcm from the box before the change to lamda coordinates, flag 1 maps it back to the new box.
   */
  deformBodies?(flag: 0 | 1): void;
  /** unfix: remove anything the fix created (its own computes). */
  destroy?(): void;
  /** Cumulative energy exchanged with a thermostat/barostat reservoir (thermo ecouple). */
  ecouple?(): number;

  /**
   * dump image "fix ID type|element|const flag1 flag2" (dump_image.html): the graphics objects this fix draws for the
   * current snapshot. Fixes without graphics leave it undefined; the dump refuses a fix that provides none.
   */
  imageObjects?(): ImageObject[];

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
  /** Rigid-body fixes: zero the linear or angular momentum of every body (velocity ... zero ... rigid fix-ID). */
  zeroMomentum?(): void;
  zeroRotation?(): void;
  /** Rigid-body fixes: the IDs of the atoms in bodies once the bodies are set up, else null. */
  bodyAtomIds?(): Set<number> | null;

  /**
   * fix_modify; returns values consumed (0 = keyword not handled). The keywords every fix reads, as native LAMMPS does
   * (src/fix.cpp Fix::modify_params at b332f28): dynamic/dof; energy and virial, where yes needs a fix that tallies
   * energy or virial and no is always accepted; respa, for the styles native gives a rRESPA level. A fix with keywords
   * of its own handles them first and passes the rest here.
   */
  modify(key: string, values: string[]): number {
    switch (key) {
      case 'dynamic/dof': this.dynamicDof = logical(values[0]); return 1;
      case 'energy': {
        const on = logical(values[0]);
        if (on && !this.energyGlobal && !this.energyAtom) throw new StyleError(`Fix ${this.id} ${this.style} does not support fix_modify energy command`);
        this.thermoEnergy = on;
        return 1;
      }
      case 'virial': {
        const on = logical(values[0]);
        if (on && !this.virialGlobal && !this.virialAtom) throw new StyleError(`Fix ${this.id} ${this.style} does not support fix_modify virial command`);
        this.thermoVirial = on;
        return 1;
      }
      case 'respa':
        if (!RESPA_LEVEL_STYLES.has(this.style) || inumeric(values[0]) < 0) throw new StyleError('Illegal fix_modify respa command');
        return 1;
      default: return 0;
    }
  }

  /** True if atom i is in this fix's group. */
  protected inGroup(i: number): boolean {
    return (this.sys.state.mask[i] & this.groupBit) !== 0;
  }
}
