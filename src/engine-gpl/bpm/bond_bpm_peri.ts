/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm_peri.cpp and src/BPM/bond_bpm_peri.h at commit b332f28 (the PMB model with the
 * per-type-pair critical-stretch break test; the state-based LPS, VES and EPS models are refused here). Authors
 * credited in the BPM sources: Joel Clemmer (SNL); the peridynamic bond model follows Silling et al. (2007) and
 * the BPM peridynamics papers cited in the C++ header. Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * bond_style bpm/peri pmb c horizon s00 alpha: docs.lammps.org/bond_bpm_peri.html. The doc says the bond force
 * uses "the peridynamic micromodulus c" and the critical stretch s00 - alpha * smin, where smin is the minimum
 * stretch of the atom's bonds in the previous step (a bond breaks when its stretch exceeds the more tensile of
 * its two atoms' critical stretches). Browser-engine differences: the per-atom arrays (smin, vinter) are keyed by
 * atom ID; the nodal volume vfrac and the damage are the custom per-atom properties of fix property/atom
 * (d_vfrac, d_damage); the lattice spacing comes from the style context (lattice spacing in x).
 * Refused by name: the lps, ves and eps models, and store/local.
 */
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { BondBPMBase, bondKey } from './bond_bpm';

// LAMMPS: NEAR_ZERO of bond_bpm_peri.cpp
const NEAR_ZERO = 2.2204e-16;
const DBL_MAX = Number.MAX_VALUE;

/** Linear partial-volume taper of the reference length r0 near the horizon (vfrac_taper in the C++ file). */
const vfracTaper = (r0: number, delta0: number, halfLc: number): number => {
  if (Math.abs(r0 - delta0) <= halfLc) return (-1 / (2 * halfLc)) * r0 + (1 + (delta0 - halfLc) / (2 * halfLc));
  return 1;
};

export class BondBpmPeri extends BondBPMBase {
  readonly name = 'bpm/peri';
  readonly kind = 'bond' as const;
  readonly paramNames = ['c', 'cut', 's00', 'alpha'];
  /** Per bond type: PMB micromodulus c, horizon cut, s00 and alpha of the critical stretch. */
  private c = new Float64Array(0);
  private cut = new Float64Array(0);
  private s00 = new Float64Array(0);
  private alpha = new Float64Array(0);
  private xlattice = 1;
  private readonly ref = new Map<number, number>();
  /** Per atom ID: the minimum stretch of its bonds at the previous compute (LAMMPS d_smin; -DBL_MAX = none). */
  private smin = new Map<number, number>();
  /** Per atom ID: the reference interaction volume (LAMMPS d_vinter), set at the first compute. */
  private vinter = new Map<number, number>();
  private stored = false;

  settings(args: string[], _ctx?: StyleContext): void {
    const rest = this.takeCommon(args);
    if (rest.length) throw new StyleError(`Illegal bond_style bpm/peri argument: ${rest[0]}`);
  }

  /** The pmb coefficients are not written to a data file (write_data of this style is not supported). */
  dataCoeffs(): string[] { return []; }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    this.c = new Float64Array(n); this.cut = new Float64Array(n);
    this.s00 = new Float64Array(n); this.alpha = new Float64Array(n);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length < 2) throw new StyleError('Incorrect args for bond coefficients (expected: <type> pmb c horizon s00 alpha)');
    const model = args[1];
    if (model !== 'pmb') {
      throw new StyleError(`bond_style bpm/peri model ${model} is not supported by the browser engine (only pmb is implemented)`);
    }
    if (args.length !== 6) throw new StyleError('Incorrect args for bond_style bpm/peri pmb model (expected: <type> pmb c horizon s00 alpha)');
    const v = args.slice(2).map((w) => Number(w));
    if (v.some((x) => !Number.isFinite(x))) throw new StyleError('bond_style bpm/peri pmb: coefficients must be numbers');
    const [c, cut, s00, alpha] = v;
    if (!(cut > 0)) throw new StyleError(`Invalid horizon value ${cut} for bond_style bpm/peri`);
    const [lo, hi] = this.bounds(args[0]);
    for (let t = lo; t <= hi; t++) {
      this.c[t] = c; this.cut[t] = cut; this.s00[t] = s00; this.alpha[t] = alpha;
      this.params.set[t] = 1;
    }
  }

  private bounds(w: string): [number, number] {
    const n = Number(w);
    if (Number.isInteger(n) && n >= 1 && n <= this.ntypes) return [n, n];
    throw new StyleError(`bond_style bpm/peri: bond type ${w} out of range`);
  }

  init(ctx: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.params.set[t]) throw new StyleError(`all bond coeffs are not set (type ${t})`);
    }
    this.xlattice = ctx.xlattice ?? 1;
    const s = ctx.s;
    const vf = s?.custom.get('vfrac');
    if (!vf || vf.cols !== 0 || vf.int) {
      throw new StyleError("Bond style bpm/peri requires a per-atom vfrac property; add 'fix <ID> all property/atom d_vfrac ghost yes' before bond_style");
    }
  }

  /** Stores r0 and the reference interaction volume of every bond at the first compute (store_data). */
  private storeReference(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const vfrac = bc.s.custom.get('vfrac')!.data;
    const d = [0, 0, 0];
    for (let k = 0; k < b.n; k++) {
      if (b.type[k] <= 0) continue;
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const i1 = atomIndex(bc, ia, 'bond'), i2 = atomIndex(bc, ib, 'bond');
      delta(bc, i1, i2, d);
      this.ref.set(bondKey(ia, ib), Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
      this.vinter.set(ia, (this.vinter.get(ia) ?? 0) + vfrac[i2]);
      this.vinter.set(ib, (this.vinter.get(ib) ?? 0) + vfrac[i1]);
    }
    this.stored = true;
  }

  compute(bc: BondedCompute): void {
    if (!this.stored) this.storeReference(bc);
    const b = bc.s.topo.bonds;
    const s = bc.s;
    const vfrac = s.custom.get('vfrac')!.data;
    const halfLc = 0.5 * this.xlattice;
    const allow = bc.historyUpdate === true && this.breakOn;
    const sminNew = new Map<number, number>();
    const f = bc.f, v = bc.virial;
    const d = [0, 0, 0];
    let e = 0;
    for (let k = 0; k < b.n; k++) {
      const t = b.type[k];
      if (t === 0) continue;
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const lo = Math.min(ia, ib), hi = Math.max(ia, ib);
      const i1 = atomIndex(bc, lo, 'bond'), i2 = atomIndex(bc, hi, 'bond');
      delta(bc, i1, i2, d); // x_i2 - x_i1
      const dx = -d[0], dy = -d[1], dz = -d[2]; // del = x_i1 - x_i2 as the C++ bond
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const key = bondKey(lo, hi);
      let r0 = this.ref.get(key);
      if (r0 === undefined) {
        r0 = r;
        this.ref.set(key, r0);
      }
      let dr = r - r0;
      if (Math.abs(dr) < NEAR_ZERO) dr = 0;
      const stretch = dr / r0;
      const smin1 = this.smin.get(lo) ?? -DBL_MAX, smin2 = this.smin.get(hi) ?? -DBL_MAX;
      if (allow && stretch > this.s00[t] - this.alpha[t] * Math.max(smin1, smin2)) {
        this.breakBond(b.type, k);
        continue;
      }
      const scale = vfracTaper(r0, this.cut[t], halfLc);
      const vEff = 0.5 * (vfrac[i1] + vfrac[i2]);
      // PMB bond force and energy (bond_bpm_peri.cpp, PMB branch)
      const fbond = r > 0 ? -(this.c[t] * vEff * scale * stretch) / r : 0;
      const eb = 0.5 * this.c[t] * vEff * scale * stretch * dr;
      f[3 * i1] += dx * fbond; f[3 * i1 + 1] += dy * fbond; f[3 * i1 + 2] += dz * fbond;
      f[3 * i2] -= dx * fbond; f[3 * i2 + 1] -= dy * fbond; f[3 * i2 + 2] -= dz * fbond;
      e += eb;
      // ev_tally(i1, i2, ..., fbond * vfrac_eff, del): v = del (x) del * (fbond * vfrac_eff)
      const fv = fbond * vEff;
      v[0] += dx * dx * fv; v[1] += dy * dy * fv; v[2] += dz * dz * fv;
      v[3] += dx * dy * fv; v[4] += dx * dz * fv; v[5] += dy * dz * fv;
      if (bc.eatom) { bc.eatom[i1] += 0.5 * eb; bc.eatom[i2] += 0.5 * eb; }
      sminNew.set(lo, Math.min(sminNew.get(lo) ?? DBL_MAX, stretch));
      sminNew.set(hi, Math.min(sminNew.get(hi) ?? DBL_MAX, stretch));
    }
    bc.acc.ebond += e;
    // commit the new minimum stretch of every atom (-DBL_MAX when it has no surviving bond)
    for (let i = 0; i < s.n; i++) {
      const id = s.id[i];
      const sm = sminNew.get(id);
      this.smin.set(id, sm === undefined ? -DBL_MAX : sm);
    }
    this.updateDamage(bc, vfrac);
  }

  /** Damage of each atom: 1 - (surviving partner volume) / (reference interaction volume), if d_damage exists. */
  private updateDamage(bc: BondedCompute, vfrac: Float64Array): void {
    const damage = bc.s.custom.get('damage');
    if (!damage || damage.cols !== 0 || damage.int) return;
    const s = bc.s;
    const b = s.topo.bonds;
    const surviving = new Float64Array(s.n);
    for (let k = 0; k < b.n; k++) {
      if (b.type[k] <= 0) continue;
      const i1 = atomIndex(bc, b.atoms[2 * k], 'bond'), i2 = atomIndex(bc, b.atoms[2 * k + 1], 'bond');
      surviving[i1] += vfrac[i2];
      surviving[i2] += vfrac[i1];
    }
    for (let i = 0; i < s.n; i++) {
      const vi = this.vinter.get(s.id[i]) ?? 0;
      damage.data[i] = vi > 0 ? 1 - surviving[i] / vi : 0;
    }
  }
}
