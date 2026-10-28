/*
 * fix ID group-ID external mode args — docs.lammps.org/fix_external.html:
 *
 *   "*pf/callback* args = Ncall Napply", "Ncall = make callback every Ncall steps", "Napply = apply callback forces
 *   every Napply steps"; "*pf/array* args = Napply", "Napply = apply array forces every Napply steps".
 *
 * "This fix allows external programs that are running LAMMPS through its library interface to modify certain LAMMPS
 * properties on specific timesteps." With pf/array "the fix simply stores force values in an array. The fix adds these
 * forces to each atom in the group, once every *Napply* steps"; the array (fexternal) is "allocated by the FixExternal
 * class as an (N,3) array" and "It is up to the external program to set the values in this array". With pf/callback the
 * fix calls the function the driver registered (lammps_set_fix_external_callback) every Ncall steps.
 *
 * The notebook has no library caller: nothing registers a callback and nothing writes the array, the energy
 * (set_energy_global) or the virial (set_virial_global). So a pf/array fix adds its zero forces every Napply steps and
 * contributes zero energy and virial, as native does without a driver, and a pf/callback fix stops the run at init with
 * native's error. As native LAMMPS does (src/fix_external.cpp at b332f28): the argument checks and messages in this
 * order, the array zeroed when allocated, the global scalar is the caller's energy (0), the per-atom array is fexternal
 * (3 columns), and the docs' "fix_modify energy yes" default ("The default setting for this fix is fix_modify energy
 * yes").
 */
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { inumeric } from '../force/util';
import type { System } from '../system';

export class FixExternal extends Fix {
  readonly style = 'external';
  private readonly callbackMode: boolean;
  private readonly ncall: number;
  private readonly napply: number;
  /** fexternal: the forces a driver would set, 3 per owned atom (always zero here) */
  private fexternal = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal fix external command: missing argument(s)');
    const mode = args[0];
    if (mode === 'pf/callback') {
      if (args.length !== 3) throw new StyleError('Incorrect number of args for fix external pf/callback');
      this.callbackMode = true;
      this.ncall = inumeric(args[1]);
      this.napply = inumeric(args[2]);
      if (this.ncall <= 0) throw new StyleError('Illegal value for ncall in fix external pf/callback');
      if (this.napply <= 0) throw new StyleError('Illegal value for napply in fix external pf/callback');
    } else if (mode === 'pf/array') {
      if (args.length !== 2) throw new StyleError('Incorrect number of args for fix external pf/array');
      this.callbackMode = false;
      this.ncall = 0;
      this.napply = inumeric(args[1]);
      // native's message names pf/callback here too
      if (this.napply <= 0) throw new StyleError(`Illegal value for napply (${this.napply}) in fix external pf/callback`);
    } else {
      throw new StyleError(`Unknown fix external mode ${mode}`);
    }
    this.scalarFlag = true;
    this.globalFreq = 1;
    this.extscalar = 1;
    this.peratomFlag = true;
    this.peratomFreq = 1;
    this.sizePeratomCols = 3;
    this.energyGlobal = true;
    this.virialGlobal = true;
    this.thermoEnergy = true;
    this.thermoVirial = true;
    this.grow();
  }

  /** grow_arrays: fexternal (and the per-atom array output) for the owned atoms, zeroed. */
  private grow(): void {
    const n = this.sys.state.n;
    if (this.fexternal.length !== 3 * n) {
      this.fexternal = new Float64Array(3 * n);
      this.arrayAtom = this.fexternal;
    }
  }

  init(): void {
    if (this.callbackMode) throw new StyleError('Fix external callback function not set');
  }

  /** post_force: on every Napply-th step the stored forces are added to the atoms of the group. */
  postForce(): void {
    this.grow();
    const s = this.sys.state;
    if (s.step % this.napply !== 0) return;
    const { f, mask } = s;
    const fe = this.fexternal, bit = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      f[3 * i] += fe[3 * i];
      f[3 * i + 1] += fe[3 * i + 1];
      f[3 * i + 2] += fe[3 * i + 2];
    }
  }

  minSetup(): void { this.postForce(); }

  minPostForce(): void { this.postForce(); }

  /** compute_scalar: the energy the caller set (set_energy_global), 0 without one. */
  computeScalar(): number { return 0; }

  energy(): number { return 0; }

  energyAtom(_out: Float64Array): void {}
}
