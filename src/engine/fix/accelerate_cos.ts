import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * fix ID group-ID accelerate/cos value — docs.lammps.org/fix_accelerate_cos.html
 * (page text in plans/lammps-docs/fix_accelerate_cos.rst).
 *
 * Syntax (docs): "fix ID group-ID accelerate value", "accelerate/cos = style
 * name of this fix command", "value = amplitude of acceleration (in unit of
 * velocity/time)". Example: "fix 1 all accelerate/cos 2.0e-7".
 *
 * Description (docs): "Give each atom a acceleration in x-direction based on
 * its z coordinate. The acceleration is a periodic function along the
 * z-direction:"  a_x(z) = A cos(2 pi z / l_z), where A is the acceleration
 * amplitude and l_z the z-length of the simulation box.
 *
 * Measured with native LAMMPS (black box, probes in the scratch dir): the
 * force on each atom of the group is f_x = m A cos(2 pi (z - zlo) / lz) / ftm2v,
 * where zlo is the lower z face of the box. Checked for lj units (m = 1 and
 * m = 2) and for real units (m = 2, A = 0.002, f_x = 9.5602...), at every
 * z layer of a box whose lower face is at -3. The z dependence is periodic in
 * lz, so wrapped or unwrapped positions give the same force. Velocity after a
 * step of length dt is A cos(...) dt (the acceleration is a force per mass).
 *
 * Restrictions (docs): "Since this fix depends on the z-coordinate of atoms,
 * it cannot be used in 2d simulations." The docs also say "In order to get
 * meaningful results, the group ID of this fix should be all." (not enforced).
 * "No information about this fix is written to binary restart files."
 * "No global or per-atom quantities are stored by this fix" so it has no
 * f_ID output. "This fix is not invoked during energy minimization."
 * Triclinic boxes are not handled by the docs; the engine rejects them rather
 * than guess the z-length of a tilted box.
 *
 * The value must be a number: the docs define no variable form.
 */
export class FixAccelerateCos extends Fix {
  readonly style = 'accelerate/cos';
  private readonly amp: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('usage: fix ID group-ID accelerate/cos value');
    if (sys.dimension === 2) throw new StyleError('fix accelerate/cos cannot be used in 2d simulations (it depends on the z-coordinate)');
    if (args[0].startsWith('v_')) throw new StyleError('fix accelerate/cos: value must be a number, variables are not supported');
    this.amp = Number(args[0]);
    if (args[0].trim() === '' || !Number.isFinite(this.amp)) throw new StyleError(`fix accelerate/cos: expected a number for value, got '${args[0]}'`);
  }

  /** f_x += m A cos(2 pi (z - zlo) / lz) / ftm2v for each atom of the group. */
  postForce(): void {
    const s = this.sys.state;
    const b = s.box;
    if (b.triclinic) throw new StyleError('fix accelerate/cos does not support triclinic boxes');
    const lz = b.hi[2] - b.lo[2];
    const zlo = b.lo[2];
    const k = (2 * Math.PI) / lz;
    const { f, mask, x } = s;
    const scale = this.amp / s.units.ftm2v;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      f[3 * i] += m * scale * Math.cos(k * (x[3 * i + 2] - zlo));
    }
  }

}
