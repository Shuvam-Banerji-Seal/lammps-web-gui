import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { PerAtomTerm } from '../force/forcefield';

/*
 * Finite-difference fixes — docs.lammps.org/fix_numdiff.html (fix_numdiff.rst)
 * and docs.lammps.org/fix_numdiff_virial.html (fix_numdiff_virial.rst).
 *
 * fix numdiff: "fix ID group-ID numdiff Nevery delta" with "Nevery = calculate
 * force by finite difference every this many timesteps" and "delta = size of
 * atom displacements (distance units)". "It then restores the original
 * position.  That component of force is calculated as the difference in energy
 * divided by two times *delta*." The sign is the physical force F = -dE/dx:
 * measured with native LAMMPS (black box, lj/cut, tests/oracle/w16numdiff_forces.in)
 * the finite-difference value equals the analytic per-atom force. "The group
 * specified with the command means only atoms within the group have their
 * averages computed.  Results are set to 0.0 for atoms not in the group."
 * "This fix produces a per-atom array which can be accessed by various
 * :doc:`output commands <Howto_output>`, which stores the components of the
 * force on each atom as calculated by finite difference." "The per-atom values
 * can only be accessed on timesteps that are multiples of *Nevery* since that
 * is when the finite difference forces are calculated." "The array values
 * calculated by this compute will be in force :doc:`units <units>`." "No
 * parameter of this fix can be used with the *start/stop* keywords of the
 * :doc:`run <run>` command.  This fix is invoked during :doc:`energy
 * minimization <minimize>`."
 *
 * fix numdiff/virial: "fix ID group-ID numdiff/virial Nevery delta" with
 * "Nevery = calculate virial by finite difference every this many timesteps"
 * and "delta = magnitude of strain fields (dimensionless)". "The specified
 * group must be "all"." "This fix applies linear strain fields of magnitude
 * *delta* to all the atoms relative to a point at the center of the box."
 * "The difference in these two energies divided by two times *delta*,
 * approximates the corresponding component of the virial stress tensor, after
 * applying a suitable unit conversion." "This fix produces a global vector
 * which can be accessed by various :doc:`output commands <Howto_output>`,
 * which stores the components of the virial stress tensor as calculated by
 * finite difference." "The order of the virial stress tensor components is
 * *xx*, *yy*, *zz*, *yz*, *xz*, and *xy*, consistent with Voigt notation. Note
 * that the vector produced by :doc:`compute pressure <compute_pressure>` uses
 * a different ordering, with *yz* and *xy* swapped." "The vector values
 * calculated by this compute are "intensive".  The vector values will be in
 * pressure :doc:`units <units>`."
 *
 * Measured with native LAMMPS (black box): the finite-difference virial equals
 * compute pressure (NULL, virial) components xx yy zz yz xz xy with the yz/xy
 * swap above; with pair_modify tail yes the value is unchanged from tail no,
 * so the tail correction is not part of the finite-difference energy (the
 * engine's accumulators add etailV/V to evdwl, so it is subtracted here).
 *
 * The energy at a displaced configuration is the engine's own force-field
 * potential: pair, bond, angle, dihedral, improper and kspace terms, without
 * fix energy (fix_modify energy yes) and without the tail correction, matching
 * the measurement above. Positions, forces, per-atom energy/virial arrays and
 * the energy/virial accumulators are saved and restored around the
 * evaluations, so a run with the fix has the same thermo as a run without it.
 */

/** What ff.compute overwrites: saved before the finite-difference loop, restored after. */
interface Snapshot {
  x: Float64Array;
  f: Float64Array;
  torque: Float64Array | null;
  eatom: Float64Array | null;
  vatom: Float64Array | null;
  eatomTerm: Partial<Record<PerAtomTerm, Float64Array>>;
  vatomTerm: Partial<Record<PerAtomTerm, Float64Array>>;
  accNums: number[];
  accVirials: Float64Array[];
}

/** Shared parsing, energy evaluation and state snapshot for the two numdiff styles. */
abstract class FixNumdiffBase extends Fix {
  /** "delta = size of atom displacements (distance units)" (or the strain magnitude). */
  protected readonly delta: number;

  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args);
    if (args.length !== 2) {
      throw new StyleError(`fix ${id} (${style}): expected exactly Nevery and delta, got ${args.length} argument(s)`);
    }
    const nevery = Number(args[0]);
    if (!Number.isInteger(nevery) || nevery < 1) {
      throw new StyleError(`fix ${id} (${style}): Nevery must be a positive integer, got '${args[0]}'`);
    }
    this.nevery = nevery;
    const delta = Number(args[1]);
    if (!Number.isFinite(delta) || !(delta > 0)) {
      throw new StyleError(`fix ${id} (${style}): delta must be a positive number, got '${args[1]}'`);
    }
    this.delta = delta;
  }

  /** Saves everything ff.compute() overwrites. */
  protected snapshot(): Snapshot {
    const s = this.sys.state;
    const ff = this.sys.ff;
    const a = ff.acc;
    return {
      x: Float64Array.from(s.x.subarray(0, 3 * s.n)),
      f: Float64Array.from(s.f.subarray(0, 3 * s.n)),
      torque: s.torque ? Float64Array.from(s.torque.subarray(0, 3 * s.n)) : null,
      eatom: ff.eatom,
      vatom: ff.vatom,
      eatomTerm: ff.eatomTerm,
      vatomTerm: ff.vatomTerm,
      accNums: [a.evdwl, a.ecoul, a.elong, a.ebond, a.eangle, a.edihed, a.eimp],
      accVirials: [a.virial, a.vbond, a.vangle, a.vdihed, a.vimp, a.vlong].map((v) => Float64Array.from(v)),
    };
  }

  /** Restores the saved state exactly (positions, forces, accumulators, ghosts). */
  protected restore(snap: Snapshot): void {
    const s = this.sys.state;
    const ff = this.sys.ff;
    const a = ff.acc;
    s.x.set(snap.x);
    s.f.set(snap.f);
    if (s.torque && snap.torque) s.torque.set(snap.torque);
    ff.eatom = snap.eatom;
    ff.vatom = snap.vatom;
    ff.eatomTerm = snap.eatomTerm;
    ff.vatomTerm = snap.vatomTerm;
    a.evdwl = snap.accNums[0]; a.ecoul = snap.accNums[1]; a.elong = snap.accNums[2];
    a.ebond = snap.accNums[3]; a.eangle = snap.accNums[4]; a.edihed = snap.accNums[5]; a.eimp = snap.accNums[6];
    [a.virial, a.vbond, a.vangle, a.vdihed, a.vimp, a.vlong].forEach((v, k) => v.set(snap.accVirials[k]));
    // ghosts follow the restored owned positions again so the next step's forces are unchanged
    this.sys.nb.forwardComm(s, this.sys.geom);
  }

  /**
   * Potential energy of the current positions using the engine's own force
   * evaluation. Neighbor lists are not rebuilt: the displacement is tiny, so
   * every pair that could enter or leave the force cutoff is already in the
   * list (its pair style zeroes interactions beyond the cutoff).
   */
  protected energyNow(): number {
    const sys = this.sys;
    const s = sys.state;
    sys.nb.forwardComm(s, sys.geom);
    sys.ff.compute(s, sys.nb, sys.geom, {});
    const a = sys.ff.acc;
    const vol = sys.geom.volume(sys.dimension);
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp - sys.ff.etailV / vol;
  }

  /** "on timesteps that are multiples of *Nevery*". */
  protected onNevery(): boolean {
    return this.sys.state.step % this.nevery === 0;
  }
}

/**
 * fix numdiff — finite-difference forces, per-atom array of 3 columns (force
 * units), 0.0 for atoms outside the fix group.
 */
export class FixNumdiff extends FixNumdiffBase {
  readonly style = 'numdiff';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'numdiff');
    this.peratomFlag = true;
    this.sizePeratomCols = 3;
  }

  postForce(): void {
    if (this.onNevery()) this.computeForces();
  }

  minPostForce(): void {
    this.computeForces();
  }

  private computeForces(): void {
    const s = this.sys.state;
    const n = s.n;
    if (this.arrayAtom.length !== 3 * n) this.arrayAtom = new Float64Array(3 * n);
    else this.arrayAtom.fill(0);
    if (!this.groupBit) return;
    const snap = this.snapshot();
    try {
      const d = this.delta;
      for (let i = 0; i < n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue; // "Results are set to 0.0 for atoms not in the group."
        for (let c = 0; c < 3; c++) {
          const k = 3 * i + c;
          const x0 = snap.x[k];
          s.x[k] = x0 + d;
          const ePlus = this.energyNow();
          s.x[k] = x0 - d;
          const eMinus = this.energyNow();
          s.x[k] = x0;
          // F = -(E(x+delta) - E(x-delta)) / (2 delta)
          this.arrayAtom[k] = -(ePlus - eMinus) / (2 * d);
        }
      }
    } finally {
      this.restore(snap);
    }
  }
}

/**
 * fix numdiff/virial — finite-difference virial stress, global vector of 6
 * components xx yy zz yz xz xy in pressure units. The group must be all.
 */
export class FixNumdiffVirial extends FixNumdiffBase {
  readonly style = 'numdiff/virial';
  private readonly vector = new Float64Array(6);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'numdiff/virial');
    // "The specified group must be "all"."
    if (group !== 'all') throw new StyleError('fix numdiff/virial requires group all');
    this.vectorFlag = true;
    this.sizeVector = 6;
  }

  postForce(): void {
    if (this.onNevery()) this.computeVirial();
  }

  minPostForce(): void {
    this.computeVirial();
  }

  computeVector(i: number): number {
    return this.vector[i];
  }

  private computeVirial(): void {
    const sys = this.sys;
    const s = sys.state;
    const box = s.box;
    const cx = 0.5 * (box.lo[0] + box.hi[0]);
    const cy = 0.5 * (box.lo[1] + box.hi[1]);
    const cz = 0.5 * (box.lo[2] + box.hi[2]);
    const center = [cx, cy, cz];
    // one single-term (engineering) strain field per Voigt component: the
    // displacement of atom i is delta * (coordinate ref - center ref) along
    // axis, which gives dU/dstrain = -W(ref, axis); the reported value is
    // -dU/dstrain / V * nktv2p, so it is +W(axis, ref) / V * nktv2p.
    const fields: readonly (readonly [number, number])[] = [
      [0, 0], // xx
      [1, 1], // yy
      [2, 2], // zz
      [2, 1], // yz
      [2, 0], // xz
      [1, 0], // xy
    ];
    const snap = this.snapshot();
    try {
      const d = this.delta;
      const vol = sys.geom.volume(sys.dimension);
      const nktv2p = s.units.nktv2p;
      for (let k = 0; k < 6; k++) {
        const [axis, ref] = fields[k];
        let ePlus = 0;
        let eMinus = 0;
        for (const sign of [1, -1]) {
          s.x.set(snap.x);
          for (let i = 0; i < s.n; i++) {
            s.x[3 * i + axis] = snap.x[3 * i + axis] + sign * d * (snap.x[3 * i + ref] - center[ref]);
          }
          const e = this.energyNow();
          if (sign === 1) ePlus = e;
          else eMinus = e;
        }
        this.vector[k] = -((ePlus - eMinus) / (2 * d)) / vol * nktv2p;
      }
    } finally {
      this.restore(snap);
    }
  }
}
