import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { SPIN_HBAR } from '../force/pair/spin_exchange';

/*
 * compute ID group-ID spin — docs.lammps.org/compute_spin.html: "This compute calculates the following 6
 * magnetic quantities: the three first quantities are the x,y and z coordinates of the total magnetization,
 * the fourth quantity is the norm of the total magnetization, The fifth quantity is the magnetic energy (in
 * eV), The sixth one is referred to as the spin temperature, according to the work of (Nurdin)." "The array
 * values are \"intensive\". The array values will be in metal units".
 *
 * Measured with native LAMMPS (black box, two and three atoms, pair spin/exchange): components 1-3 are the
 * mean of the unit spin vectors (sum of s_i over N, not weighted by the magnitude g), component 4 its norm,
 * component 5 the spin pair energy (the same value as pe for a spin-only system).
 *
 * Spin temperature: T = hbar/kB * sum_i |s_i x w_i|^2 / (2 sum_i s_i.w_i) with w_i the precession vector of
 * the spin pair style (fm), as measured for two atoms (kB = 8.617343e-5 eV/K, metal). The normalization for
 * other N is checked by the w47spin oracle.
 */

const KB_METAL = 8.617343e-5;

export class ComputeSpin extends Compute {
  readonly style = 'spin';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('compute spin takes no arguments');
    if (!sys.state.sp) throw new StyleError('compute spin requires atom_style spin');
    this.vectorFlag = true;
    this.sizeVector = 6;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const sp = s.sp!, fm = s.fm;
    const out = new Float64Array(6);
    let n = 0, mx = 0, my = 0, mz = 0, num = 0, den = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      n++;
      const sx = sp[4 * i], sy = sp[4 * i + 1], sz = sp[4 * i + 2];
      mx += sx; my += sy; mz += sz;
      if (fm) {
        const wx = fm[3 * i], wy = fm[3 * i + 1], wz = fm[3 * i + 2];
        const cx = sy * wz - sz * wy, cy = sz * wx - sx * wz, cz = sx * wy - sy * wx;
        num += cx * cx + cy * cy + cz * cz;
        den += sx * wx + sy * wy + sz * wz;
      }
    }
    if (n > 0) { mx /= n; my /= n; mz /= n; }
    out[0] = mx; out[1] = my; out[2] = mz;
    out[3] = Math.hypot(mx, my, mz);
    // the magnetic energy: the spin pair style plus the precession/spin fixes (measured with native LAMMPS: the
    // zeeman, anisotropy and cubic energies are included in the fifth component)
    let e = s.spinEnergy ?? 0;
    for (const f of this.sys.fixes) if (f.style === 'precession/spin') e += f.energy();
    out[4] = e;
    out[5] = den !== 0 ? (SPIN_HBAR / KB_METAL) * num / (2 * den) : 0;
    this.vector = out;
  }
}
