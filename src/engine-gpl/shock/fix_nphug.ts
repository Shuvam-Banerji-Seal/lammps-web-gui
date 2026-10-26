/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SHOCK/fix_nphug.cpp and fix_nphug.h
 * (class FixNPHug, fix nphug: a Nose-Hoover fix whose thermostat drives the temperature onto the Hugoniot of the
 * shocked material, the Hugoniostat of Ravelo, Holian, Germann and Lomdahl, Phys Rev B 70, 014103 (2004)). The
 * fix derives from fix nh (src/engine/fix/nh.ts here; native fix_nh.cpp), and changes its thermostat target, its
 * masses and its global vector. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Documented behaviour (docs.lammps.org/fix_nphug.html): "fix ID group-ID nphug keyword value ..." with the keywords
 * of fix npt; "The temp keyword must be used to specify the time constant for kinetic energy relaxation, but initial
 * and final target temperature values are ignored"; "One and only one of the following keywords should be used:
 * iso, aniso, tri, x, y, z"; "The fix_modify e0, v0 and p0 keywords can be used to override the default reference
 * state values". The global vector: "dhugoniot (temperature units), the shock velocity us (distance/time), and the
 * particle velocity up (distance/time)"; the rest of the vector is that of fix nh. The fix has no uniaxial or
 * estimate keyword: the uniaxial case is the one of a single x, y or z pressure keyword, as fix_nphug.cpp's
 * constructor derives it. Restart state is not stored by the browser engine yet.
 */
import { FixNH } from '../../engine/fix/nh';
import { StyleError } from '../../engine/force/types';
import { ownCompute, removeCompute } from '../../engine/fix/util';
import { parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';

export class FixNPHug extends FixNH {
  private e0 = 0;
  private v0 = 1;
  private p0 = 0;
  private e0Set = false;
  private v0Set = false;
  private p0Set = false;
  private uniaxial = false;
  private idir = 0;
  private rho0 = 0;
  private t0: number | null = null;
  private readonly idPe: string;
  private pe!: Compute;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'nphug');
    // fix_nphug.cpp constructor: "turn off deviatoric flag and remove strain energy from vector"
    this.deviatoric = false;
    // fix_nphug.cpp: "Pstart and Pstop must have the same value"
    for (let d = 0; d < 3; d++) {
      if (this.pStart[d] !== this.pStop[d]) throw new StyleError('Pstart and Pstop must have the same value');
    }
    const tilt = this.pFlag[3] || this.pFlag[4] || this.pFlag[5];
    const flagged = [0, 1, 2].filter((d) => this.pFlag[d]);
    const hydrostatic = this.pStart[0] === this.pStart[1] && this.pStart[0] === this.pStart[2];
    if (tilt) {
      if (!hydrostatic || this.pStart[3] !== 0 || this.pStart[4] !== 0 || this.pStart[5] !== 0) {
        throw new StyleError('For triclinic deformation, specified target stress must be hydrostatic');
      }
      this.uniaxial = false;
    } else if (hydrostatic) {
      // iso, or an aniso target that is the same in every dimension (hydrostatic)
      this.uniaxial = false;
    } else if (flagged.length === 1) {
      this.uniaxial = true;
      this.idir = flagged[0];
    } else {
      throw new StyleError('Specified target stress must be uniaxial or hydrostatic');
    }
    // "fix-ID_pe": the potential energy compute (group all) the fix creates
    this.idPe = `${id}_pe`;
    this.pe = ownCompute(sys, this.idPe, 'all', 'pe', []);
    this.sizeVector += 2;   // three values added, the strain energy entry removed (fix_nphug.cpp constructor)
  }

  destroy(): void {
    super.destroy();
    removeCompute(this.sys, this.idPe);
  }

  /**
   * fix_nphug.cpp pack_restart_data: "e0, v0, p0" ahead of the FixNH list (its t0 is in that list). The t0 flag tells
   * whether the reference temperature was set (setup or a restart file) when the file was written.
   */
  protected restartExtra(): Record<string, number> {
    return { e0: this.e0, v0: this.v0, p0: this.p0, t0: this.t0 ?? 0, t0Set: this.t0 === null ? 0 : 1 };
  }

  /** fix_nphug.cpp restart(): the reference state of the file replaces the keywords; applied before the first setup. */
  private applyRestored(): void {
    const r = this.restoredExtra;
    if (!r) return;
    this.restoredExtra = undefined;
    this.e0 = r.e0; this.e0Set = true;
    this.v0 = r.v0; this.v0Set = true;
    this.p0 = r.p0; this.p0Set = true;
    if (r.t0Set === 1) this.t0 = r.t0;
  }

  /** fix_nphug.cpp modify_param: e0, v0 and p0 set the reference state. */
  modify(key: string, values: string[]): number {
    if (key === 'e0' || key === 'v0' || key === 'p0') {
      // FixNPHug::modify_param: "if (narg < 2) error->all(FLERR,\"Illegal fix nphug command\")"
      if (values[0] === undefined) throw new StyleError('Illegal fix nphug command');
      const v = parseNum(values[0], `fix_modify ${key}`);
      if (key === 'e0') { this.e0 = v; this.e0Set = true; }
      else if (key === 'v0') { this.v0 = v; this.v0Set = true; }
      else { this.p0 = v; this.p0Set = true; }
      return 1;
    }
    return super.modify(key, values);
  }

  init(): void {
    super.init();
    this.pe = this.sys.compute(this.idPe);
  }

  /** The reference temperature of the masses: the temperature at the first setup (fix_nphug.cpp via fix_nh setup). */
  protected setupTargetTemp(): number {
    if (this.t0 === null) this.t0 = this.tSetup ?? this.kineticTemp();
    return this.t0;
  }

  /** fix_nphug.cpp setup(): the reference state, rho0 and the target temperature 0.01 after the base setup. */
  setup(): void {
    this.applyRestored();
    super.setup();
    const s = this.sys.state;
    const kB = s.units.boltz;
    const t0 = this.t0 ?? 0;
    this.chainMassTemp = t0;
    // fix_nh.cpp setup(): barostat masses from the initial temperature
    for (let d = 0; d < 6; d++) if (this.pFlag[d]) this.omegaMass[d] = (s.n + 1) * kB * t0 * this.pPeriod[d] * this.pPeriod[d];
    if (!this.v0Set) { this.v0 = this.computeVol(); this.v0Set = true; }
    if (!this.p0Set) {
      this.p0 = this.uniaxial ? this.pressureComponent(this.idir) : (this.pressureComponent(0) + this.pressureComponent(1) + this.pressureComponent(2)) / 3.0;
      this.p0Set = true;
    }
    if (!this.e0Set) { this.e0 = this.computeEtotal(); this.e0Set = true; }
    let masstot = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) masstot += s.massByType[s.type[i]];
    this.rho0 = s.units.nktv2p * s.units.mvv2e * masstot / this.v0;
    this.tTarget = 0.01;
  }

  /** Current pressure component of the fix's pressure compute (a fresh evaluation). */
  private pressureComponent(d: number): number {
    this.sys.refreshComputes();
    return this.pressCompute!.vectorValues()[d];
  }

  /** The volume of the cell: xprd yprd zprd in 3d, xprd yprd in 2d (fix_nphug.cpp compute_vol). */
  private computeVol(): number {
    const g = this.sys.geom;
    return this.sys.state.dimension === 3 ? g.lx * g.ly * g.lz : g.lx * g.ly;
  }

  /** Total energy of the atoms: potential plus kinetic with the fix's dof (fix_nphug.cpp compute_etotal). */
  private computeEtotal(): number {
    this.sys.refreshComputes();
    const epot = this.pe.scalarValue();
    const ekin = this.tempCompute.scalarValue() * 0.5 * this.tempCompute.dof * this.sys.state.units.boltz;
    return epot + ekin;
  }

  /** Pressure used by the Hugoniot relations: the uniaxial component, or the mean of the diagonal (compute_scalar). */
  private targetPressure(): number {
    this.sys.refreshComputes();
    this.tempCompute.vectorValues();
    if (this.uniaxial) return this.pressCompute!.vectorValues()[this.idir];
    return this.pressCompute!.scalarValue();
  }

  /**
   * Deviation of the current point from the Hugoniot in temperature units (fix_nphug.cpp compute_hugoniot):
   * (0.5 (p + p0)(v0 - v)) / nktv2p + e0 - e, divided by dof k_B.
   */
  private computeHugoniot(): number {
    const e = this.computeEtotal();
    const p = this.targetPressure();
    const v = this.computeVol();
    const s = this.sys.state;
    let dhugo = (0.5 * (p + this.p0) * (this.v0 - v)) / s.units.nktv2p + this.e0 - e;
    dhugo /= this.tempCompute.dof * s.units.boltz;
    return dhugo;
  }

  /** Shock velocity, Us^2 = (p - p0) / (rho0 eps) with eps = 1 - v/v0 (fix_nphug.cpp compute_us). */
  private computeUs(): number {
    const p = this.targetPressure();
    const v = this.computeVol();
    const eps = 1.0 - v / this.v0;
    if (eps < 1.0e-10) return 0.0;
    if (p < this.p0) return 0.0;
    return Math.sqrt((p - this.p0) / (this.rho0 * eps));
  }

  /** Particle velocity, up = Us eps (fix_nphug.cpp compute_up). */
  private computeUp(): number {
    const v = this.computeVol();
    const eps = 1.0 - v / this.v0;
    return this.computeUs() * eps;
  }

  /** compute_temp_target: t_target = t_current + compute_hugoniot(), the Hugoniot temperature target. */
  protected stepTargetTemp(): number {
    return this.kineticTemp() + this.computeHugoniot();
  }

  /** f_ID[1..3] = dhugoniot, us, up; then the fix nh vector without its strain energy entry. */
  computeVector(i: number): number {
    if (i === 0) return this.computeHugoniot();
    if (i === 1) return this.computeUs();
    if (i === 2) return this.computeUp();
    return super.computeVector(i - 3);
  }
}
