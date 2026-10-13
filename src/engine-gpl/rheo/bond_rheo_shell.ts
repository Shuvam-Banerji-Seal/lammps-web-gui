/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/RHEO/bond_rheo_shell.cpp and bond_rheo_shell.h (BondRHEOShell: bond_style rheo/shell), at commit b332f28,
 * with the BPM base processes it calls (src/BPM/bond_bpm.cpp process_broken and process_new). RHEO package authors:
 * Joel Clemmer (SNL), Thomas O'Connor (CMU), Eric Palermo (CMU). Copyright Sandia Corporation (GNU General Public
 * License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/bond_rheo_shell.html): "bond_style rheo/shell keyword value" with "required keyword = t/form" and
 * "t/form value = formation time for a bond (time units)"; "The rheo/shell bond style is designed to work with bonds between
 * eligible surface or near-surface particles. When a bond ... is not computed until the timer reaches the specified bond
 * formation time, t/form, and the bond is enabled and applies forces. If the two particles move outside of the maximum
 * bond distance or move into the bulk before the timer reaches t/form, the bond automatically deletes itself." The force is
 * "F = 2 k (r - r_0) + 2 k / (r_0^2 epsilon_c^2) (r - r_0)^3" with the damping "F_D = - gamma w (r_hat . v)" and the
 * coefficients k, epsilon_c (the critical strain, "maximum strain beyond which a bond breaks") and gamma per bond type
 * (bond_coeff N k epsilon_c gamma). The browser engine's port: the bond is created by fix rheo/oxidation; the store/local
 * keyword is refused by name.
 *
 * The bond does not switch off the pair interaction of its atoms (native ignore_special_flag = 1: ignoresSpecial). The
 * number of formed shell bonds per atom is kept in the internal per-atom property SHELL_NBOND (native's "shell_nbond"
 * fix property/atom) and an atom with shell bonds does not shift (native sets STATUS_NO_SHIFT).
 *
 * The bond history (reference length and formation timer) is kept per bond pair, as native's per-bond history; a bond
 * that is deleted or broken drops its history, so a later bond of the same pair starts from zero.
 */
import { SimpleBonded, atomIndex, bondDelta } from '../../engine/force/bonded_util';
import { BOND_REMOVED, StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { RHEO_STATUS } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { rheoModelOf, rheoProp, STATUS_NO_SHIFT, shellNbond } from './model';

/** compute: "static constexpr double EPSILON = 1e-10;" (bond_rheo_shell.cpp). */
const EPSILON = 1e-10;

/** Key of a bond by its two atom IDs (IDs stay below 2^26 in the browser engine). */
export const bondKey = (a: number, b: number): number => (a < b ? a * 0x4000000 + b : b * 0x4000000 + a);

interface BondHistory {
  /** Reference length (r0 in native, bondstore[n][0]). */
  r0: number;
  /** Time the bond has been eligible (bondstore[n][1]). */
  t: number;
}

export class BondRHEOShell extends SimpleBonded {
  readonly name = 'rheo/shell';
  readonly kind = 'bond' as const;
  readonly paramNames = ['k', 'ecrit', 'gamma'];
  /** t/form: the time a bond must be eligible before it forms (native tform). */
  private tform = 0;
  private readonly history = new Map<number, BondHistory>();
  /** Set when a compute() deleted or broke bonds; read and cleared by takeTopologyChange(). */
  private changed = false;

  settings(args: string[], _ctx?: StyleContext): void {
    // bond_style rheo/shell: "required keyword = t/form" (docs.lammps.org/bond_rheo_shell.html)
    let tformSet = false;
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k], val = args[k + 1];
      if (key === 'store/local') throw new StyleError('bond_style rheo/shell keyword store/local is not supported by the browser engine');
      if (key !== 't/form' || val === undefined) {
        throw new StyleError(`Illegal bond rheo/shell command, invalid argument ${key}`);
      }
      const t = parseNum(val, 't/form');
      // bond_rheo_shell.cpp: "Illegal bond rheo/shell command, must specify positive formation time"
      if (t < 0.0) throw new StyleError('Illegal bond rheo/shell command, must specify positive formation time');
      this.tform = t;
      tformSet = true;
    }
    if (!tformSet) throw new StyleError('bond_style rheo/shell requires the t/form keyword');
  }

  /** Bonds of this style do not count as 1-2 neighbors for the special lists (native ignore_special_flag). */
  ignoresSpecial(_type: number): boolean {
    return true;
  }

  takeTopologyChange(): boolean {
    const c = this.changed;
    this.changed = false;
    return c;
  }

  compute(bc: BondedCompute): void {
    const s = bc.s;
    const model = rheoModelOf(s);
    if (!model || !model.oxidation) throw new StyleError('Need to define fix rheo/oxidation to use bond rheo/shell');
    if (!model.surfaceFlag) throw new StyleError('Bond rheo/shell requires surface calculation in fix rheo');
    const { cut: rmax, rsurf } = model.oxidation;
    const rsurface = model.rsurface;
    const nbond = shellNbond(s);
    const status = rheoProp(s, RHEO_STATUS);
    const K = this.params.p('k'), ecrit = this.params.p('ecrit'), gam = this.params.p('gamma');
    const tform = this.tform;
    const dt = s.dt;
    const dbond = new Float64Array(s.n);
    const b = s.topo.bonds;
    const d = [0, 0, 0];
    const vel = s.v;
    const f = bc.f;
    const v = bc.virial;
    for (let k = 0; k < b.n; k++) {
      const type = b.type[k];
      if (type === 0 || type === BOND_REMOVED) continue; // skip bond if already broken
      // the first atom of the pair is the one with the lower tag (native: "if (tag[i2] < tag[i1]) swap")
      let ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      if (ib < ia) { const tmp = ia; ia = ib; ib = tmp; }
      const i1 = atomIndex(bc, ia, 'bond'), i2 = atomIndex(bc, ib, 'bond');
      bondDelta(bc, i1, i2, d); // r_i1 - r_i2 (the nearest image)
      const delx = d[0], dely = d[1], delz = d[2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const r = Math.sqrt(rsq);
      const key = bondKey(ia, ib);
      let h = this.history.get(key);
      if (!h) {
        h = { r0: 0, t: 0 };
        this.history.set(key, h);
      }
      // a bond with no history yet (t below EPSILON or NaN) starts from zero (native "process_new")
      if (h.t < EPSILON || Number.isNaN(h.t)) {
        h.t = 0.0;
        h.r0 = 0.0;
      }
      let r0 = h.r0;
      // bond has not yet formed: check eligibility and advance the timer
      if (h.t < tform) {
        if (r > rmax || rsurface[i1] > rsurf || rsurface[i2] > rsurf) {
          // native process_ineligibility: the bond leaves the atoms' lists (it was never counted in nbond)
          b.type[k] = BOND_REMOVED;
          this.history.delete(key);
          this.changed = true;
          continue;
        }
        h.t += dt;
        if (h.t >= tform) {
          h.r0 = r;
          r0 = r;
          dbond[i1] += 1;
          dbond[i2] += 1;
        } else {
          continue;
        }
      }
      const e = (r - r0) / r0;
      if (Math.abs(e) > ecrit[type]) {
        // native process_broken: type 0, and the formed bond stops counting for both atoms
        b.type[k] = 0;
        this.changed = true;
        this.history.delete(key);
        dbond[i1] -= 1;
        dbond[i2] -= 1;
        continue;
      }
      const rinv = 1.0 / r;
      const dr = r - r0;
      let fbond = 2 * K[type] * (-dr + dr * dr * dr / (r0 * r0 * ecrit[type] * ecrit[type]));
      const delvx = vel[3 * i1] - vel[3 * i2], delvy = vel[3 * i1 + 1] - vel[3 * i2 + 1], delvz = vel[3 * i1 + 2] - vel[3 * i2 + 2];
      const dot = delx * delvx + dely * delvy + delz * delvz;
      fbond -= gam[type] * dot * rinv;
      fbond *= rinv;
      f[3 * i1] += delx * fbond;
      f[3 * i1 + 1] += dely * fbond;
      f[3 * i1 + 2] += delz * fbond;
      f[3 * i2] -= delx * fbond;
      f[3 * i2 + 1] -= dely * fbond;
      f[3 * i2 + 2] -= delz * fbond;
      // virial tally of the pair (ev_tally with the force magnitude fbond, no energy)
      v[0] += delx * delx * fbond;
      v[1] += dely * dely * fbond;
      v[2] += delz * delz * fbond;
      v[3] += delx * dely * fbond;
      v[4] += delx * delz * fbond;
      v[5] += dely * delz * fbond;
    }
    // the formed-bond counts of the atoms; an atom with shell bonds does not shift (native post_compute part)
    for (let i = 0; i < s.n; i++) {
      nbond[i] += dbond[i];
      if (nbond[i] !== 0) status[i] = (status[i] | STATUS_NO_SHIFT) >>> 0;
    }
    // the deleted entries (BOND_REMOVED) leave the topology in the force field, which rebuilds the special lists
  }
}
