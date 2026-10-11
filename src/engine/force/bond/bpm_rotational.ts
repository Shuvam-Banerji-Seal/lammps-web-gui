import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../types';

/*
 * bond_style bpm/rotational — docs.lammps.org/bond_bpm_rotational.html. This engine implements the radial
 * (normal) part of the style; the shear, twisting and bending terms are refused (see coeff()).
 *
 * The doc says: "The reference state is stored by each bond when it is first computed in the setup of a run."
 * The normal force is the doc equation f_radial = k_radial (r - r_0) with r_0 the initial distance.
 * Breaking: the doc's criterion B = max{0, f_r/f_r,c + ...} (f_r the radial force; the shear and torque terms
 * are zero here). The doc says: "the bond will break" when B reaches one. "This is done by setting the bond
 * type to 0" so that forces are no longer computed.
 * Smoothing: "This term is calculated as" w = 1 - B^4. The smooth keyword switches it off.
 * Damping: F_D = -gamma_radial w (r_hat . v), with v the velocity difference between the two particles.
 * The doc says the normalize keyword "normalizes normal and shear forces by the reference length".
 * The defaults: overlay/pair no, smooth yes, normalize no, break yes, frame average, damping derivative.
 *
 * Measured with native LAMMPS (black box), two atoms of radius 0.5 (diameter 1), units lj, r0 = 1 unless noted,
 * k_r = 1, f_r,c = 1 (w = 1 - B^4, B = |f_r| / f_r,c):
 *   - the normal force on atom 2 is -w k (r - r0) r_hat (r_hat from atom 1 to atom 2): stretch 1.99 gives
 *     -0.039009950100 = -0.99 (1 - 0.99^4); compression 0.01 (r - r0 = -0.99) gives +0.0390099501.
 *   - the breaking test uses the magnitude: compression with f_r = -0.3 gives B = 0.3 (force on atom 2 is
 *     0.3 (1 - 0.3^4) = 0.29757), so the smoothing applies under compression too.
 *   - the bond energy is 0.5 k (r - r0)^2 with no smoothing factor (0.49005 for r - r0 = 0.99), and with
 *     normalize yes it is 0.5 k (r - r0)^2 / r0 (0.04 for r0 = 2, r - r0 = 0.4).
 *   - normalize yes: f_r = k (r - r0) / r0 in the force and in B (force 0.19968 for r0 = 2, r - r0 = 0.4).
 *   - the damping is not divided by r0: with normalize yes, relative velocity 0.5 along the bond gives
 *     -gamma w 0.5 = -0.2496 on atom 2 (the total -0.44928 = -0.19968 - 0.2496).
 *   - damping sign: relative velocity v2 - v1 = +0.5 along the bond gives -0.25 on atom 2 with gamma 0.5 (w = 1).
 *   - smooth no: w = 1 (force 0.2 for r - r0 = 0.2).
 *   - setup: a bond with B >= 1 at the first force evaluation of a run breaks there: r - r0 = 1.01 gives zero
 *     force, zero energy, and compute nbond/atom reports the bond gone (unlike bpm/spring, whose setup keeps it);
 *     r - r0 = 0.99 at setup is kept (force -0.039).
 *   - timestep: a dimer pulled apart by velocity (k 1, f_r,c 0.5, mass from density 1): step 25 has B = 0.9658
 *     (kept, force 0.0628), step 26 has B = 1.0025 and force 0 (type 0 from then on).
 *   - break no with smooth yes gives native's text: must turn off smoothing with break no option (the same text
 *     as bond bpm/spring); break no smooth no runs.
 *   - store/local is refused here (the output fix is not implemented). frame and damping: the radial term does
 *     not depend on them (the relative normal direction is frame independent); the listed values are accepted.
 *   - a zero fcs, tct or tcb gives NaN forces in native (measured), so the engine refuses those values.
 *
 * Refused: the shear, twisting and bending terms (k_shear, k_twist, k_bend) and their damping (gamma_shear/slide,
 * gamma_twist, gamma_bend/roll). Their displacement definitions (the frame and the history of the shear and
 * angular displacements) are not on the doc page, so they are not implemented.
 */

export class BondBpmRotational extends SimpleBonded {
  readonly name = 'bpm/rotational';
  readonly kind = 'bond' as const;
  readonly paramNames = ['kr', 'ks', 'kt', 'kb', 'fcr', 'fcs', 'tct', 'tcb', 'gr', 'gs', 'gt', 'gb'];
  smooth = true;
  normalize = false;
  breakOn = true;
  overlay = false;
  /** Reference length per bond, keyed by the two atom IDs (smaller ID first, see bondKey). */
  private readonly ref = new Map<number, number>();
  /** Set when a compute() call broke bonds; read and cleared by takeTopologyChange(). */
  private changed = false;

  settings(args: string[], _ctx?: StyleContext): void {
    const seen = new Set<string>();
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k], val = args[k + 1];
      if (val === undefined) throw new StyleError(`bond_style bpm/rotational: keyword '${key}' needs a value`);
      if (key === 'store/local') throw new StyleError('bond_style bpm/rotational keyword store/local is not supported by the browser engine');
      if (seen.has(key)) throw new StyleError(`bond_style bpm/rotational: keyword '${key}' is given twice`);
      seen.add(key);
      if (key === 'overlay/pair') this.overlay = yesno(val, key);
      else if (key === 'smooth') this.smooth = yesno(val, key);
      else if (key === 'normalize') this.normalize = yesno(val, key);
      else if (key === 'break') this.breakOn = yesno(val, key);
      else if (key === 'frame') {
        if (val !== 'average' && val !== 'particle') throw new StyleError(`bond_style bpm/rotational frame must be average or particle (got '${val}')`);
      } else if (key === 'damping') {
        if (val !== 'derivative' && val !== 'dem') throw new StyleError(`bond_style bpm/rotational damping must be derivative or dem (got '${val}')`);
      } else throw new StyleError(`illegal bond_style bpm/rotational keyword '${key}'`);
    }
    if (!this.breakOn && this.smooth) throw new StyleError('Illegal bond bpm command, must turn off smoothing with break no option');
  }

  /** The bond_coeff values: kr ks kt kb fcr fcs tct tcb gr gs gt gb (the doc's order of the coefficient list). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (!args.length) throw new StyleError('usage: bond_coeff N coefficients');
    const v = this.parse(args.slice(1));
    const [, ks, kt, kb, , fcs, tct, tcb, , gs, gt, gb] = v;
    if (ks || kt || kb || gs || gt || gb) {
      throw new StyleError('bond_style bpm/rotational: the shear, twisting and bending terms (k_shear, k_twist, k_bend and the damping of those modes) are not implemented in the browser engine; set them to 0 (the radial term is implemented)');
    }
    if (!(fcs > 0) || !(tct > 0) || !(tcb > 0)) throw new StyleError('bond_style bpm/rotational: the critical values fcs, tct and tcb must be > 0');
    this.params.setRange(args[0], v);
  }

  /** True once after a compute() that broke bonds (the force field then rebuilds the special lists). */
  takeTopologyChange(): boolean {
    const c = this.changed;
    this.changed = false;
    return c;
  }

  compute(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const K = this.params.p('kr'), fcr = this.params.p('fcr'), gam = this.params.p('gr');
    const d = [0, 0, 0];
    const f = bc.f;
    const v = bc.virial;
    const vel = bc.s.v;
    let e = 0;
    for (let k = 0; k < b.n; k++) {
      const t = b.type[k];
      if (t === 0) continue; // broken bond: type 0 (doc: "setting the bond type to 0")
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const i = atomIndex(bc, ia, 'bond');
      const j = atomIndex(bc, ib, 'bond');
      delta(bc, j, i, d); // d = r_i - r_j, with i the first atom of the bond
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const key = bondKey(ia, ib);
      let r0 = this.ref.get(key);
      if (r0 === undefined) {
        // the reference state is stored the first time the bond is computed (setup or a timestep)
        r0 = r;
        this.ref.set(key, r0);
      }
      const norm = this.normalize ? r0 : 1;
      const dr = r - r0;
      const fr = (K[t] * dr) / norm; // f_radial, positive when stretched
      // B = |f_r| / f_r,c (the shear and torque terms are zero here). Measured: the magnitude, under compression too.
      const B = Math.abs(fr) / fcr[t];
      if (this.breakOn && B >= 1) {
        b.type[k] = 0;
        this.changed = true;
        continue;
      }
      const w = this.smooth ? 1 - Math.pow(B, 4) : 1;
      // relative normal velocity along r_hat (from atom ib to atom ia negated): rdot = r_hat_ab . (v_b - v_a)
      // with r_hat_ab = (x_b - x_a)/r = -d/r, so rdot = d . (v_a - v_b) / r
      const dvx = vel[3 * i] - vel[3 * j], dvy = vel[3 * i + 1] - vel[3 * j + 1], dvz = vel[3 * i + 2] - vel[3 * j + 2];
      const rdot = (d[0] * dvx + d[1] * dvy + d[2] * dvz) / r;
      // force on atom ia: w (f_r + gamma rdot) r_hat_ab = -w (f_r + gamma rdot) d / r
      const G = w * (fr + gam[t] * rdot);
      const c = -G / r;
      const es = (0.5 * K[t] * dr * dr) / norm;
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
  throw new StyleError(`bond_style bpm/rotational ${key} must be yes or no (got '${w}')`);
};
