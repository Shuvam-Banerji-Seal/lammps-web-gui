/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-MOLECULE/bond_harmonic_restrain.cpp and bond_harmonic_restrain.h at commit b332f28 (the LAMMPS
 * developers; Copyright (2003) Sandia Corporation, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * bond_style harmonic/restrain (docs.lammps.org/bond_harmonic_restrain.html): E = K (r - r_t0)^2, where r_t0 is the
 * length of each bond at the beginning of the first run ("the reference bond lengths r_t=0 are the bond lengths at the
 * beginning of the first run or minimize command after the bond style has been defined"). Coefficients: "K".
 *
 * Native stores the positions of the atoms in a fix STORE/ATOM when init_style runs for the first time (natoms < 0),
 * and never again; the stored atoms follow their IDs, so the store is keyed by atom ID here. The reference length
 * takes the minimum image of the stored separation (domain->minimum_image); the current length takes the bond vector
 * of the neighbor list. A system with more atoms than at the first run is an error (native message kept).
 * Native writedata = 0: the style writes no Bond Coeffs section (dataCoeffs returns null).
 */
import { atomIndex, bondDelta, vectorAccess } from '../../engine/force/bonded_util';
import { Bonded, StyleError, typeBounds, type BondedAdaptAccess, type BondedCompute, type StyleContext, NoSettingsError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { evTallyBond } from './ev';

export class BondHarmonicRestrain extends Bonded {
  readonly name = 'harmonic/restrain';
  /** fix adapt: as native extract() does (src/EXTRA-MOLECULE/bond_harmonic_restrain.cpp at b332f28), k (init_style keeps the positions it stored at the first init). */
  override adaptParam(name: string): BondedAdaptAccess | null {
    return vectorAccess({ k: this.k }, name);
  }
  readonly kind = 'bond' as const;
  private k!: Float64Array;
  private set!: Uint8Array;
  /** Stored positions (3 per atom), in the order of the atom IDs at the first init. */
  private x0: Float64Array | null = null;
  /** atom ID -> slot in x0. */
  private slot = new Map<number, number>();
  /** natoms of native: -1 until the first init_style. */
  private natoms = -1;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new NoSettingsError(args[0]);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.k = new Float64Array(ntypes + 1);
    this.set = new Uint8Array(ntypes + 1);
  }

  /** bond_harmonic_restrain.cpp coeff(): narg must be 2 ("Incorrect args for bond coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 2) throw new StyleError('Incorrect args for bond coefficients');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const k_one = parseNum(args[1], 'K');
    for (let i = lo; i <= hi; i++) {
      this.k[i] = k_one;
      this.set[i] = 1;
    }
  }

  /** bond_harmonic_restrain.cpp init_style(): store the positions at the first run; later runs keep them. */
  init(ctx: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`All bond coeffs are not set (bond type ${t})`);
    }
    const s = ctx.s;
    if (!s) return;
    if (this.natoms < 0) {
      this.x0 = new Float64Array(3 * s.n);
      this.slot = new Map<number, number>();
      for (let i = 0; i < s.n; i++) {
        this.slot.set(s.id[i], i);
        for (let c = 0; c < 3; c++) this.x0[3 * i + c] = s.x[3 * i + c];
      }
      this.natoms = s.n;
    }
    if (this.natoms < s.n) throw new StyleError('Bond style harmonic/restrain does not support adding atoms');
  }

  compute(bc: BondedCompute): void {
    const bonds = bc.s.topo.bonds;
    const K = this.k;
    const x0 = this.x0;
    const f = bc.f;
    const dx0 = [0, 0, 0];
    const d = [0, 0, 0];
    for (let n = 0; n < bonds.n; n++) {
      const i1 = atomIndex(bc, bonds.atoms[2 * n], 'bond');
      const i2 = atomIndex(bc, bonds.atoms[2 * n + 1], 'bond');
      const type = bonds.type[n];
      const s1 = this.slot.get(bc.s.id[i1]), s2 = this.slot.get(bc.s.id[i2]);
      if (x0 === null || s1 === undefined || s2 === undefined) {
        throw new StyleError('Bond style harmonic/restrain does not support adding atoms');
      }
      // delx = x0[i1] - x0[i2]; domain->minimum_image; r0 = sqrt(rsq)
      dx0[0] = x0[3 * s1] - x0[3 * s2];
      dx0[1] = x0[3 * s1 + 1] - x0[3 * s2 + 1];
      dx0[2] = x0[3 * s1 + 2] - x0[3 * s2 + 2];
      bc.geom.minimumImage(dx0);
      const rsq0 = dx0[0] * dx0[0] + dx0[1] * dx0[1] + dx0[2] * dx0[2];
      const r0 = Math.sqrt(rsq0);
      bondDelta(bc, i1, i2, d);
      const delx = d[0], dely = d[1], delz = d[2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const r = Math.sqrt(rsq);
      const dr = r - r0;
      const rk = K[type] * dr;
      const fbond = r > 0.0 ? -2.0 * rk / r : 0.0;
      const ebond = rk * dr;
      f[3 * i1] += delx * fbond; f[3 * i1 + 1] += dely * fbond; f[3 * i1 + 2] += delz * fbond;
      f[3 * i2] -= delx * fbond; f[3 * i2 + 1] -= dely * fbond; f[3 * i2 + 2] -= delz * fbond;
      evTallyBond(bc, i1, i2, ebond, fbond, d);
    }
  }

  /** native writedata = 0: no Bond Coeffs section in write_data. */
  dataCoeffs(): string[] | null { return null; }
}
