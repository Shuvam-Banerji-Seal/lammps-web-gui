/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm_spring_plastic.cpp and src/BPM/bond_bpm_spring_plastic.h at commit b332f28 (Joel
 * Clemmer, SNL; copyright Sandia Corporation, GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * bond_style bpm/spring/plastic: docs.lammps.org/bond_bpm_spring_plastic.html. The doc says: "The reference length
 * of the bond r_0 is stored by each bond when it is first computed in the setup of a run." and "If the bond
 * stretches beyond a strain of epsilon_p in compression or extension, it will plastically activate and r_eq will
 * evolve to ensure |(r-r_eq)/r_eq| never exceeds epsilon_p." The force is F = -k (r_eq - r) w with the damping
 * F_D = -gamma w (r_hat . v) and w = 1 - ((r - r0)/(r0 epsilon_c))^8 (smooth yes). Coefficients: k, epsilon_c,
 * gamma, epsilon_p. The C++ state per bond is [r0, epsilon_eq] (epsilon_eq the plastic strain offset).
 * Browser-engine difference: the state is a map keyed by the two atom IDs.
 */
import { delta, atomIndex } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { BondBPMBase, bondKey, yesno } from './bond_bpm';

export class BondBpmSpringPlastic extends BondBPMBase {
  readonly name = 'bpm/spring/plastic';
  readonly kind = 'bond' as const;
  readonly paramNames = ['k', 'ecrit', 'gamma', 'eplastic'];
  smooth = true;
  normalize = false;
  /** Per bond: [r0, ep] (LAMMPS bondstore columns 0 and 1). */
  private readonly state = new Map<number, Float64Array>();

  settings(args: string[], _ctx?: StyleContext): void {
    const rest = this.takeCommon(args);
    for (let k = 0; k < rest.length; k += 2) {
      const key = rest[k], val = rest[k + 1];
      if (val === undefined) throw new StyleError(`bond_style bpm/spring/plastic: keyword '${key}' needs a value`);
      if (key === 'smooth') this.smooth = yesno(val, 'bond_style bpm/spring/plastic', key);
      else if (key === 'normalize') this.normalize = yesno(val, 'bond_style bpm/spring/plastic', key);
      else throw new StyleError(`Illegal bond bpm command, invalid argument ${key}`);
    }
    if (!this.breakOn && this.smooth) throw new StyleError('Illegal bond bpm command, must turn off smoothing with break no option');
  }

  compute(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const K = this.params.p('k'), ecrit = this.params.p('ecrit'), gam = this.params.p('gamma'), ep0 = this.params.p('eplastic');
    const updating = bc.historyUpdate === true;
    const vel = bc.s.v;
    const d = [0, 0, 0];
    const f = bc.f;
    const v = bc.virial;
    for (let k = 0; k < b.n; k++) {
      const t = b.type[k];
      if (t === 0) continue;
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const i1 = atomIndex(bc, Math.min(ia, ib), 'bond'), i2 = atomIndex(bc, Math.max(ia, ib), 'bond');
      delta(bc, i1, i2, d); // x_i2 - x_i1
      // the C++ uses del = x_i1 - x_i2 and the force on i1 = fbond * del
      const dx = -d[0], dy = -d[1], dz = -d[2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const key = bondKey(ia, ib);
      let st = this.state.get(key);
      if (!st) {
        st = new Float64Array([r, 0]);
        this.state.set(key, st);
      }
      const r0 = st[0];
      let ep = st[1];
      const e = (r - r0) / r0;
      if (Math.abs(e) > ecrit[t] && updating && this.breakOn) {
        this.breakBond(b.type, k);
        continue;
      }
      // plastic activation: the equilibrium strain follows the bond once |e - ep| exceeds epsilon_p
      if (e > ep + ep0[t]) {
        ep = e - ep0[t];
        st[1] = ep;
      }
      if (e < ep - ep0[t]) {
        ep = e + ep0[t];
        st[1] = ep;
      }
      const rinv = 1 / r;
      let fbond = this.normalize ? -K[t] * (e - ep) : -K[t] * (r - (1 + ep) * r0);
      const vx = vel[3 * i1] - vel[3 * i2], vy = vel[3 * i1 + 1] - vel[3 * i2 + 1], vz = vel[3 * i1 + 2] - vel[3 * i2 + 2];
      const dot = dx * vx + dy * vy + dz * vz;
      fbond -= gam[t] * dot * rinv;
      fbond *= rinv;
      if (this.smooth) {
        let s = (r - r0) / (r0 * ecrit[t]);
        s *= s; s *= s; s *= s;
        fbond *= 1 - s;
      }
      f[3 * i1] += dx * fbond; f[3 * i1 + 1] += dy * fbond; f[3 * i1 + 2] += dz * fbond;
      f[3 * i2] -= dx * fbond; f[3 * i2 + 1] -= dy * fbond; f[3 * i2 + 2] -= dz * fbond;
      const w0 = dx * dx * fbond, w1 = dy * dy * fbond, w2 = dz * dz * fbond;
      const w3 = dx * dy * fbond, w4 = dx * dz * fbond, w5 = dy * dz * fbond;
      v[0] += w0; v[1] += w1; v[2] += w2; v[3] += w3; v[4] += w4; v[5] += w5;
      // per-atom: half of v to each atom (Bond::ev_tally, newton_bond on; no energy)
      if (bc.vatom) { const va = bc.vatom; const w = [w0, w1, w2, w3, w4, w5]; for (let c = 0; c < 6; c++) { va[6 * i1 + c] += 0.5 * w[c]; va[6 * i2 + c] += 0.5 * w[c]; } }
    }
  }
}
