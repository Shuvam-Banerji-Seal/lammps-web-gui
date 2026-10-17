/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm_spring.cpp and src/BPM/bond_bpm_spring.h at commit b332f28 (the spring bond with the
 * reference length, the smoothing and normalize options, the volume/factor term and the bond breaking). Contributing
 * author of the BPM package: Joel Clemmer (SNL); copyright Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * bond_style bpm/spring: docs.lammps.org/bond_bpm_spring.html (doc source bond_bpm_spring.rst). The doc says: "The
 * bpm/spring bond style computes forces based on deviations from the initial reference state of the two atoms."
 * and "Bonds will break at a strain of epsilon_c. This is done by setting the bond type to 0 such that forces are no
 * longer computed." (the markup of epsilon_c is dropped here). The volume/factor keyword adds the multibody term
 * alpha_v ([(V_i + V_j) / (V0_i + V0_j)]^(1/3) - r / r0) to the force, as the doc gives it.
 *
 * Browser-engine differences: the per-bond reference length, and the per-atom vol, vol0 and dvol0 of the C++
 * (custom properties of fix property/atom there), are maps keyed by the atom IDs; the bond direction is the minimum
 * image (or the periodic copy the bonded code uses) rather than LAMMPS' ghost-atom index. Atom pairs are ordered by
 * ID as the C++ does (i1 has the lower tag).
 */
import { delta } from '../../engine/force/bonded_util';
import { StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { BondBPMBase, bondKey, yesno } from './bond_bpm';
import { pow as libmPow } from '../libm/pow';


export class BondBpmSpring extends BondBPMBase {
  readonly name = 'bpm/spring';
  readonly kind = 'bond' as const;
  readonly paramNames = ['k', 'eps', 'gamma'];
  protected readonly localStoreOk = true;
  smooth = true;
  normalize = false;
  volume = false;
  /** Per type: the volume-term coefficient av (0 unless volume/factor yes). */
  private av = new Float64Array(0);
  /** Reference length per bond, keyed by the two atom IDs. */
  private readonly ref = new Map<number, number>();
  /** Per atom ID: the interaction volume now (vol), at the reference state (vol0) and its change (dvol0). */
  private vol = new Map<number, number>();
  private vol0 = new Map<number, number>();
  private dvol0 = new Map<number, number>();
  private vol0Calculated = false;

  settings(args: string[], _ctx?: StyleContext): void {
    const rest = this.takeCommon(args);
    for (let k = 0; k < rest.length; k += 2) {
      const key = rest[k], val = rest[k + 1];
      if (val === undefined) throw new StyleError(`Illegal bond bpm command, missing argument for ${key}`);
      if (key === 'smooth') this.smooth = yesno(val, 'bond_style bpm/spring', key);
      else if (key === 'normalize') this.normalize = yesno(val, 'bond_style bpm/spring', key);
      else if (key === 'volume/factor') this.volume = yesno(val, 'bond_style bpm/spring', key);
      else throw new StyleError(`Illegal bond bpm command, invalid argument ${key}`);
    }
    if (this.smooth && !this.breakOn) throw new StyleError('Illegal bond bpm command, must turn off smoothing with break no option');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.av = new Float64Array(ntypes + 1);
  }

  /** bond_coeff N k ecrit gamma (av with volume/factor yes), as bond_bpm_spring.cpp coeff. */
  coeff(args: string[], _ctx?: StyleContext): void {
    const n = this.volume ? 4 : 3;
    if (args.length !== n + 1) {
      throw new StyleError(`Incorrect args for bond coefficients (expected: N k ecrit gamma${this.volume ? ' av' : ''})`);
    }
    const v = args.slice(1).map((w, k) => parseNum(w, k === 0 ? 'k' : k === 1 ? 'ecrit' : k === 2 ? 'gamma' : 'av'));
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    this.params.setRange(args[0], v.slice(0, 3));
    for (let t = lo; t <= hi; t++) this.av[t] = this.volume ? v[3] : 0;
  }

  /** Breaks a bond (type 0) and records it; takeTopologyChange reads the flag. */
  private breakSpring(bc: BondedCompute, b: Int32Array, k: number, ia: number, ib: number): void {
    this.breakBond(b, k);
    this.emitBreak(bc, ia, ib);
  }

  /** calculate_vol: vol of every atom from the current length of its surviving bonds (r^3 in 3d, r^2 in 2d). */
  private calculateVol(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const dim3 = bc.s.dimension === 3;
    this.vol = new Map();
    const d = [0, 0, 0];
    for (let k = 0; k < b.n; k++) {
      if (b.type[k] <= 0) continue;
      const { i1, i2, a1, a2 } = this.pairOf(bc, b.atoms[2 * k], b.atoms[2 * k + 1]);
      delta(bc, i1, i2, d);
      const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      let volTemp = rsq;
      if (dim3) volTemp *= Math.sqrt(rsq);
      this.vol.set(a1, (this.vol.get(a1) ?? 0) + volTemp);
      this.vol.set(a2, (this.vol.get(a2) ?? 0) + volTemp);
    }
  }

  /** update_vol0: vol0 += dvol0 for every atom, then dvol0 is cleared. */
  private updateVol0(): void {
    for (const [id, dv] of this.dvol0) this.vol0.set(id, (this.vol0.get(id) ?? 0) + dv);
    this.dvol0.clear();
  }

  compute(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const s = bc.s;
    const K = this.params.p('k'), eps = this.params.p('eps'), gam = this.params.p('gamma');
    const d = [0, 0, 0];
    const f = bc.f;
    const v = bc.virial;
    const vel = s.v;
    const allowBreaks = bc.historyUpdate === true && this.breakOn;
    const dim = s.dimension;
    const invdim = 1 / dim;
    let changed = false;
    if (this.volume) {
      this.calculateVol(bc);
      if (!this.vol0Calculated) {
        // the reference volumes are those of the first compute (bond_bpm_spring.cpp: vol0 = vol once)
        for (let i = 0; i < s.n; i++) this.vol0.set(s.id[i], this.vol.get(s.id[i]) ?? 0);
        this.vol0Calculated = true;
      }
    }
    const order = this.visitOrder(bc);
    const nVisit = order ? order.length : b.n;
    for (let q = 0; q < nVisit; q++) {
      const k = order ? order[q] : q;
      const t = b.type[k];
      if (t === 0) continue; // broken bond: bpm/spring sets the type to 0 (docs: "bond type to 0")
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const { i1, i2, a1, a2 } = this.pairOf(bc, ia, ib);
      delta(bc, i1, i2, d); // x_i2 - x_i1 (nearest copy)
      // del = x_i1 - x_i2, the C++ bond vector (i1 has the lower ID)
      const dx = -d[0], dy = -d[1], dz = -d[2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const key = bondKey(a1, a2);
      let r0 = this.ref.get(key);
      if (r0 === undefined) {
        // the reference length is stored the first time the bond is computed (store_data / process_new)
        r0 = r;
        this.ref.set(key, r0);
      }
      const e = (r - r0) / r0;
      if (Math.abs(e) > eps[t] && allowBreaks) {
        this.breakSpring(bc, b.type, k, ia, ib);
        changed = true;
        if (this.volume) {
          let volTemp = r0 * r0;
          if (dim === 3) volTemp *= r0;
          this.dvol0.set(a1, (this.dvol0.get(a1) ?? 0) - volTemp);
          this.dvol0.set(a2, (this.dvol0.get(a2) ?? 0) - volTemp);
        }
        continue;
      }
      const rinv = 1 / r;
      let fbond = this.normalize ? -K[t] * e : -K[t] * (r - r0);
      // the energy is taken before the volume, damping and smoothing terms (bond_bpm_spring.cpp: ebond)
      const ebond = -0.5 * fbond * (r - r0);
      if (this.volume) {
        const volSum = (this.vol.get(a1) ?? 0) + (this.vol.get(a2) ?? 0);
        const vol0Sum = (this.vol0.get(a1) ?? 0) + (this.vol0.get(a2) ?? 0);
        fbond += this.av[t] * (libmPow(volSum / vol0Sum, invdim) - 1 - e);
      }
      const dvx = vel[3 * i1] - vel[3 * i2], dvy = vel[3 * i1 + 1] - vel[3 * i2 + 1], dvz = vel[3 * i1 + 2] - vel[3 * i2 + 2];
      const dot = dx * dvx + dy * dvy + dz * dvz;
      fbond -= gam[t] * dot * rinv;
      fbond *= rinv;
      if (this.smooth) {
        // the C++ takes the fourth power by squaring (smooth^2, then ^2 again, then ^2)
        let sm = (r - r0) / (r0 * eps[t]);
        sm *= sm; sm *= sm; sm *= sm;
        fbond *= 1 - sm;
      }
      f[3 * i1] += dx * fbond; f[3 * i1 + 1] += dy * fbond; f[3 * i1 + 2] += dz * fbond;
      f[3 * i2] -= dx * fbond; f[3 * i2 + 1] -= dy * fbond; f[3 * i2 + 2] -= dz * fbond;
      bc.acc.ebond += ebond;
      // ev_tally(i1, i2, ..., ebond, fbond, del): the virial is del (x) del * fbond
      v[0] += dx * dx * fbond; v[1] += dy * dy * fbond; v[2] += dz * dz * fbond;
      v[3] += dx * dy * fbond; v[4] += dx * dz * fbond; v[5] += dy * dz * fbond;
      if (bc.eatom) { bc.eatom[i1] += 0.5 * ebond; bc.eatom[i2] += 0.5 * ebond; }
    }
    // update_vol0 when a bond broke in this compute (bond_bpm_spring.cpp: bond_change_flag)
    if (this.volume && changed) this.updateVol0();
  }
}
