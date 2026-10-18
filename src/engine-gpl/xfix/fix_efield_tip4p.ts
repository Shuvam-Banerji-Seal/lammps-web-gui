/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_efield_tip4p.cpp and src/EXTRA-FIX/fix_efield_tip4p.h at commit b332f28 (class
 * FixEfieldTIP4P, fix efield/tip4p, derived from fix efield in src/fix_efield.cpp; the TIP4P M-site treatment
 * follows FixEfieldTIP4P::post_force and find_M). Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The field components, region and energy keyword are parsed and evaluated by the engine's fix efield
 * (src/engine/fix/force_ext.ts, FixEfield) which this fix wraps; the TIP4P sites are split as native
 * LAMMPS does. Not translated: the point-dipole torque branch (atom_style dipole), refused by name.
 */
import { Fix } from '../../engine/fix/fix';
import { FixEfield } from '../../engine/fix/force_ext';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairTIP4PBase } from '../../engine/force/pair/tip4p';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import { tip4pAlpha } from '../../engine/force/pair/tip4p_sites';

/** The first TIP4P sub-style of a hybrid pair (as native pair->extract answers), or the pair itself. */
const tip4pStyleOf = (pair: unknown): PairTIP4PBase | null => {
  if (pair instanceof PairTIP4PBase) return pair;
  if (pair instanceof PairHybrid) {
    for (const sub of pair.subs) if (sub.style instanceof PairTIP4PBase) return sub.style;
  }
  return null;
};

export class FixEfieldTIP4P extends Fix {
  readonly style = 'efield/tip4p';
  /** The engine's fix efield: argument parsing, field components and region. */
  private readonly base: FixEfield;
  private energyVar: string | null = null;
  private otype = 0;
  private htype = 0;
  private alpha = Number.NaN;
  /** Global energy (native fsum[0]) and total added force (native fsum[1..3]) from the last post_force. */
  private energyValue = 0;
  private readonly fsum = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/EXTRA-FIX/fix_efield_tip4p.cpp, FixEfieldTIP4P::init at b332f28): "Fix efield/tip4p
    // does not support the potential keyword"
    for (let k = 3; k + 1 < args.length; k += 2) {
      if (args[k] === 'potential') throw new StyleError('Fix efield/tip4p does not support the potential keyword');
      if (args[k] === 'energy' && args[k + 1].startsWith('v_')) this.energyVar = args[k + 1].slice(2);
    }
    if (sys.state.mu) {
      throw new StyleError('fix efield/tip4p: point dipoles (atom_style dipole) are not supported; the dipole torque branch of fix efield/tip4p is not translated');
    }
    this.base = new FixEfield(sys, id, group, args);
    this.energyGlobal = true;
    this.virialGlobal = true;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
  }

  /**
   * Native FixEfieldTIP4P::init: the TIP4P pair style gives otype, htype, qdist, the angle type (atype) and the bond
   * type (btype); alpha = qdist / (cos(theta0 / 2) r0) from the equilibrium bond length and angle.
   */
  init(): void {
    const sys = this.sys;
    const pair = tip4pStyleOf(sys.ff.pair);
    if (!pair) throw new StyleError('A TIP4P pair style must be defined fix efield/tip4p');
    this.otype = pair.otype;
    this.htype = pair.htype;
    const { bond, angle } = sys.ff;
    if (!bond || !angle) throw new StyleError('Bond and angle potentials must be defined for fix efield/tip4p');
    const theta = angle.equilibrium(pair.atype);
    if (!Number.isFinite(theta)) throw new StyleError('Bad TIP4P angle type for fix efield/tip4p');
    const blen = bond.equilibrium(pair.btype);
    if (!Number.isFinite(blen)) throw new StyleError('Bad TIP4P bond type for fix efield/tip4p');
    // native: alpha = *p_qdist / (cos(0.5 * theta) * blen), theta in radians (equilibrium_angle)
    this.alpha = tip4pAlpha(pair.qdist, blen, theta);
  }

  /** Native FixEfieldTIP4P::post_force with the constant or variable field of fix efield. */
  postForce(): void {
    const sys = this.sys;
    const s = sys.state;
    const { x, q, f, type, mask, image, id } = s;
    const bit = this.groupBit;
    const otype = this.otype, htype = this.htype, alpha = this.alpha;
    const st = this.base.fieldState();
    const comp = st.comp;
    const constant = comp.every((c) => c.constant);
    const region = st.region;
    const k = s.units.qe2f;
    const ev = this.energyVar !== null ? sys.atomVariable(this.energyVar) : null;
    const thermo = this.thermoVirial;
    const v = this.virial;
    if (thermo) v.fill(0);
    const fsum = this.fsum;
    fsum.fill(0);
    let e = 0;
    const inRegion = (px: number, py: number, pz: number): boolean => region === null || sys.region(region).match(px, py, pz);
    const at = (c: (typeof comp)[number], j: number): number => (typeof c.value === 'number' ? c.value : c.value[j]);
    const u = [0, 0, 0];
    const xM = [0, 0, 0];
    const d1 = [0, 0, 0], d2 = [0, 0, 0];

    // force of charge j on field component d: q E qe2f (fix efield page: F = qE)
    const force = (j: number, out: number[]): void => {
      out[0] = q[j] * at(comp[0], j) * k;
      out[1] = q[j] * at(comp[1], j) * k;
      out[2] = q[j] * at(comp[2], j) * k;
    };
    // Adds the energy, global force and virial of one site with force fv at the unwrapped position unw
    const tally = (j: number, fv: number[], unw: number[]): void => {
      if (constant) e -= fv[0] * unw[0] + fv[1] * unw[1] + fv[2] * unw[2];
      else if (ev !== null) e += ev[j];
      fsum[0] += fv[0];
      fsum[1] += fv[1];
      fsum[2] += fv[2];
      if (thermo) {
        v[0] += fv[0] * unw[0];
        v[1] += fv[1] * unw[1];
        v[2] += fv[2] * unw[2];
        v[3] += fv[0] * unw[1];
        v[4] += fv[0] * unw[2];
        v[5] += fv[1] * unw[2];
      }
    };
    const fv = [0, 0, 0];

    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      if (type[i] !== otype && type[i] !== htype) {
        // non-TIP4P charge: F = qE on the atom itself, inside the region
        if (!inRegion(x[3 * i], x[3 * i + 1], x[3 * i + 2])) continue;
        force(i, fv);
        f[3 * i] += fv[0];
        f[3 * i + 1] += fv[1];
        f[3 * i + 2] += fv[2];
        sys.geom.unwrap(x, image, i, u);
        tally(i, fv, u);
        continue;
      }
      // process all atoms of a TIP4P molecule, since its parts may be ghost atoms in native LAMMPS
      let iO: number, iH1: number, iH2: number;
      if (type[i] === otype) {
        iO = i;
        iH1 = sys.indexOfId(id[i] + 1);
        iH2 = sys.indexOfId(id[i] + 2);
      } else {
        iO = sys.indexOfId(id[i] - 1);
        if (iO !== -1 && type[iO] === otype) {
          iH1 = i;
          iH2 = sys.indexOfId(id[i] + 1);
        } else {
          iO = sys.indexOfId(id[i] - 2);
          iH1 = sys.indexOfId(id[i] - 1);
          iH2 = i;
        }
      }
      if (iH1 === -1 || iH2 === -1) throw new StyleError('TIP4P hydrogen is missing');
      if (iO === -1) throw new StyleError('TIP4P oxygen is missing');
      if (type[iH1] !== htype || type[iH2] !== htype) throw new StyleError('TIP4P hydrogen has incorrect atom type');
      if (type[iO] !== otype) throw new StyleError('TIP4P oxygen has incorrect atom type');

      if (i === iO) {
        // M site: xM = xO + alpha/2 (d1 + d2) with minimum-image bond vectors (find_M; closest_image to O)
        for (let c = 0; c < 3; c++) {
          d1[c] = x[3 * iH1 + c] - x[3 * iO + c];
          d2[c] = x[3 * iH2 + c] - x[3 * iO + c];
        }
        sys.geom.minimumImage(d1);
        sys.geom.minimumImage(d2);
        for (let c = 0; c < 3; c++) xM[c] = x[3 * iO + c] + alpha * 0.5 * (d1[c] + d2[c]);
        if (inRegion(xM[0], xM[1], xM[2])) {
          force(iO, fv);
          const fx = fv[0], fy = fv[1], fz = fv[2];
          // distribute: O gets (1 - alpha) of the M force, each H gets alpha/2 (applied whatever the H group membership)
          f[3 * iO] += fx * (1.0 - alpha);
          f[3 * iO + 1] += fy * (1.0 - alpha);
          f[3 * iO + 2] += fz * (1.0 - alpha);
          f[3 * iH1] += 0.5 * alpha * fx;
          f[3 * iH1 + 1] += 0.5 * alpha * fy;
          f[3 * iH1 + 2] += 0.5 * alpha * fz;
          f[3 * iH2] += 0.5 * alpha * fx;
          f[3 * iH2 + 1] += 0.5 * alpha * fy;
          f[3 * iH2 + 2] += 0.5 * alpha * fz;
          // unmap(xM, image[iO]): the unwrapped M site is xM plus the image shift of O
          sys.geom.unwrap(x, image, iO, u);
          const uM = [xM[0] + (u[0] - x[3 * iO]), xM[1] + (u[1] - x[3 * iO + 1]), xM[2] + (u[2] - x[3 * iO + 2])];
          tally(iO, fv, uM);
        }
      } else if (i === iH1 || i === iH2) {
        // H site: the force of the H charge itself (the closest-image test of the other sites is not used here)
        if (inRegion(x[3 * i], x[3 * i + 1], x[3 * i + 2])) {
          force(i, fv);
          f[3 * i] += fv[0];
          f[3 * i + 1] += fv[1];
          f[3 * i + 2] += fv[2];
          sys.geom.unwrap(x, image, i, u);
          tally(i, fv, u);
        }
      }
    }
    this.energyValue = e;
  }

  minPostForce(): void { this.postForce(); }

  /** Native FixEfield::compute_scalar: fsum[0] from the last post_force. */
  energy(): number { return this.energyValue; }
  computeScalar(): number { return this.energyValue; }

  /** Native FixEfield::compute_vector: the total force added to the group. */
  computeVector(i: number): number { return this.fsum[i]; }
}
