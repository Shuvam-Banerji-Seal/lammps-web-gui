import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../types';

/*
 * bond_style bpm/spring — docs.lammps.org/bond_bpm_spring.html (texts: plans/lammps-docs-live/bond_bpm_spring.rst).
 * The doc says: "The bpm/spring bond style computes forces based on deviations from the initial reference state
 * of the two atoms." and "The reference state is stored by each bond when it is first computed in the setup of
 * a run." The force has a magnitude of F = k (r - r0) w, "Bonds will break at a strain of epsilon_c. This is done
 * by setting the bond type to 0 such that forces are no longer computed." The damping force is
 * F_D = - gamma w (r_hat . v), where v "is the velocity difference between the two particles". The smoothing
 * factor is w = 1 - ((r - r0)/(r0 epsilon_c))^8.
 * With normalize yes "the elastic bond force will be" divided by r0 (see below).
 *
 * Measured with native LAMMPS (black box), two atoms, units lj (thermo energies are normalised by the atom count
 * unless thermo_modify norm no):
 *   - the first force evaluation of a run (setup) keeps a stretched bond intact even when its strain exceeds
 *     epsilon_c: the spring force is the full k (r - r0) w there. The bond breaks on the first timestep whose force
 *     evaluation finds the strain above epsilon_c, and that evaluation applies no force from the bond (damping
 *     included); compute nbond/atom reports it gone on that step.
 *   - compression beyond epsilon_c breaks the bond too (the strain is |r - r0| / r0).
 *   - normalize yes: E = k (r - r0)^2 / (2 r0) and F = k (r - r0) w / r0; the damping term is not divided by r0.
 *   - smooth no: w = 1 in the spring and damping terms.
 *   - break no with smooth yes is refused: the doc says "One cannot use break no with smooth yes."
 *   - Broken bonds drop out of the 1-2 special list (ForceField rebuilds it), so the pair interaction between the
 *     two atoms is switched on afterwards (special_bonds lj 0 1 1 coul 1 1 1 as the doc requires).
 *   - overlay/pair yes: the special list is left alone (the doc: "this scenario does not update special bond
 *     lists when bonds break"); the bond force still breaks the bond.
 * Refused with an error naming the keyword: store/local, and volume/factor yes.
 */

export class BondBpmSpring extends SimpleBonded {
  readonly name = 'bpm/spring';
  readonly kind = 'bond' as const;
  readonly paramNames = ['k', 'eps', 'gamma'];
  smooth = true;
  normalize = false;
  breakOn = true;
  overlay = false;
  /** Reference length per bond, keyed by the two atom IDs (smaller ID first, see bondKey). */
  private readonly ref = new Map<number, number>();
  /** Set when a compute() call broke bonds; read and cleared by takeTopologyChange(). */
  private changed = false;

  settings(args: string[], _ctx?: StyleContext): void {
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k], val = args[k + 1];
      if (val === undefined) throw new StyleError(`bond_style bpm/spring: keyword '${key}' needs a value`);
      if (key === 'store/local') throw new StyleError('bond_style bpm/spring keyword store/local is not supported by the browser engine');
      if (key === 'volume/factor') {
        if (yesno(val, key)) throw new StyleError('bond_style bpm/spring keyword volume/factor yes is not supported by the browser engine');
      } else if (key === 'overlay/pair') this.overlay = yesno(val, key);
      else if (key === 'smooth') this.smooth = yesno(val, key);
      else if (key === 'normalize') this.normalize = yesno(val, key);
      else if (key === 'break') this.breakOn = yesno(val, key);
      else throw new StyleError(`illegal bond_style bpm/spring keyword '${key}'`);
    }
    if (!this.breakOn && this.smooth) throw new StyleError('Illegal bond bpm command, must turn off smoothing with break no option');
  }

  /** True once after a compute() that broke bonds (the force field then rebuilds the special lists). */
  takeTopologyChange(): boolean {
    const c = this.changed;
    this.changed = false;
    return c;
  }

  compute(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const K = this.params.p('k'), eps = this.params.p('eps'), gam = this.params.p('gamma');
    const d = [0, 0, 0];
    const f = bc.f;
    const v = bc.virial;
    const vel = bc.s.v;
    const updating = bc.historyUpdate === true;
    let e = 0;
    for (let k = 0; k < b.n; k++) {
      const t = b.type[k];
      if (t === 0) continue; // broken bond: bpm/spring sets the type to 0 (docs: "bond type to 0")
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const i = atomIndex(bc, ia, 'bond');
      const j = atomIndex(bc, ib, 'bond');
      delta(bc, j, i, d); // r_i - r_j
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const key = bondKey(ia, ib);
      let r0 = this.ref.get(key);
      if (r0 === undefined) {
        // the reference state is stored the first time the bond is computed (setup or a timestep)
        r0 = r;
        this.ref.set(key, r0);
      }
      if (updating && this.breakOn && Math.abs(r - r0) / r0 > eps[t]) {
        b.type[k] = 0;
        this.changed = true;
        continue;
      }
      const w = this.smooth ? 1 - Math.pow((r - r0) / (r0 * eps[t]), 8) : 1;
      const norm = this.normalize ? r0 : 1;
      const dr = r - r0;
      const es = (0.5 * K[t] * dr * dr) / norm;
      // spring force on i is -k (r - r0) w r_hat / norm, with r_hat = d / r
      const cs = (-K[t] * dr * w) / (r * norm);
      // damping on i is -gamma w (r_hat . v_ij) r_hat = -gamma w (d . v_ij) d / r^2, v_ij = v_i - v_j
      const dvx = vel[3 * i] - vel[3 * j], dvy = vel[3 * i + 1] - vel[3 * j + 1], dvz = vel[3 * i + 2] - vel[3 * j + 2];
      const dv = d[0] * dvx + d[1] * dvy + d[2] * dvz;
      const cd = (-gam[t] * w * dv) / (r * r);
      const c = cs + cd;
      e += es;
      const fx = c * d[0], fy = c * d[1], fz = c * d[2];
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
      const w0 = d[0] * d[0] * c, w1 = d[1] * d[1] * c, w2 = d[2] * d[2] * c;
      const w3 = d[0] * d[1] * c, w4 = d[0] * d[2] * c, w5 = d[1] * d[2] * c;
      v[0] += w0; v[1] += w1; v[2] += w2; v[3] += w3; v[4] += w4; v[5] += w5;
      if (bc.eatom) { bc.eatom[i] += 0.5 * es; bc.eatom[j] += 0.5 * es; }
      if (bc.vatom) {
        const ww = [w0, w1, w2, w3, w4, w5];
        for (let q = 0; q < 6; q++) { bc.vatom[6 * i + q] += 0.5 * ww[q]; bc.vatom[6 * j + q] += 0.5 * ww[q]; }
      }
    }
    bc.acc.ebond += e;
  }
}

/** Key of a bond by its two atom IDs (IDs stay below 2^26 in the browser engine). */
const bondKey = (a: number, b: number): number => (a < b ? a * 0x4000000 + b : b * 0x4000000 + a);

const yesno = (w: string, key: string): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`bond_style bpm/spring ${key} must be yes or no (got '${w}')`);
};
