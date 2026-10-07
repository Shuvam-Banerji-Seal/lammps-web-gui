import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import type { BondedCompute } from '../types';

/*
 * bond_style harmonic — docs.lammps.org/bond_harmonic.html:
 *   E = K (r - r0)^2  "Note that the usual 1/2 factor is included in K."
 *   coefficients "K (energy/distance^2)", "r0 (distance)".
 */

export class BondHarmonic extends SimpleBonded {
  readonly name = 'harmonic';
  readonly kind = 'bond' as const;
  readonly paramNames = ['K', 'r0'];

  compute(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const K = this.params.p('K'), r0 = this.params.p('r0');
    const d = [0, 0, 0];
    const f = bc.f;
    const v = bc.virial;
    let e = 0;
    for (let k = 0; k < b.n; k++) {
      const i = atomIndex(bc, b.atoms[2 * k], 'bond');
      const j = atomIndex(bc, b.atoms[2 * k + 1], 'bond');
      const t = b.type[k];
      delta(bc, j, i, d);   // r_i - r_j
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const dr = r - r0[t];
      const rk = K[t] * dr;
      // F_i = -dE/dr * d/r
      const fbond = r > 0 ? -2 * rk / r : 0;
      const eb = rk * dr;
      e += eb;
      f[3 * i] += d[0] * fbond; f[3 * i + 1] += d[1] * fbond; f[3 * i + 2] += d[2] * fbond;
      f[3 * j] -= d[0] * fbond; f[3 * j + 1] -= d[1] * fbond; f[3 * j + 2] -= d[2] * fbond;
      const w0 = d[0] * d[0] * fbond, w1 = d[1] * d[1] * fbond, w2 = d[2] * d[2] * fbond;
      const w3 = d[0] * d[1] * fbond, w4 = d[0] * d[2] * fbond, w5 = d[1] * d[2] * fbond;
      v[0] += w0; v[1] += w1; v[2] += w2; v[3] += w3; v[4] += w4; v[5] += w5;
      if (bc.eatom) { bc.eatom[i] += 0.5 * eb; bc.eatom[j] += 0.5 * eb; }
      if (bc.vatom) {
        const w = [w0, w1, w2, w3, w4, w5];
        for (let c = 0; c < 6; c++) { bc.vatom[6 * i + c] += 0.5 * w[c]; bc.vatom[6 * j + c] += 0.5 * w[c]; }
      }
    }
    bc.acc.ebond += e;
  }

  equilibrium(type: number): number { return this.params.p('r0')[type]; }
}
