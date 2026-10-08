import { KSpace, StyleError, type KSpaceCompute, type Pair, type StyleContext } from '../types';
import type { SimState } from '../../types';
import type { Geometry } from '../../domain';
import { KSpacePPPM } from './pppm';
import { buildOwnedSites, type Tip4pModel } from '../pair/tip4p_sites';
import { PairTIP4PBase } from '../pair/tip4p';

/*
 * kspace_style pppm/tip4p accuracy — docs.lammps.org/kspace_style.html:
 * "The *pppm/tip4p* style is identical to the *pppm* style except that it
 * adds a charge at the massless fourth site in each TIP4P water molecule.
 * It should be used with :doc:`pair styles <pair_style>` with a
 * *tip4p/long* in their style name."
 * The Coulomb charge of each O sits on its M site (tip4p_sites.ts, measured
 * with native LAMMPS: M = O + (alpha/2)(H1 - O + H2 - O)); the mesh spreads
 * the charges of the M sites and the forces the mesh returns on M are
 * projected on O (1 - alpha) and on each H (alpha / 2), as the pair style does.
 *
 * Implementation: the pppm solver of pppm.ts runs unchanged on a shadow state
 * whose O positions are the M sites; its forces are collected in a scratch
 * array and projected onto the atoms here.
 */

export class KSpacePPPMTIP4P extends KSpace {
  readonly name: string = 'pppm/tip4p';
  /** The pppm solver of pppm.ts; this style only changes where the charges and forces sit. */
  private readonly solver = new KSpacePPPM();
  private pair: PairTIP4PBase | null = null;

  settings(args: string[]): void {
    this.solver.settings(args);
    this.accuracy = this.solver.accuracy;
  }

  modify(key: string, values: string[]): number { return this.solver.modify(key, values); }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    this.solver.init(s, geom, cutCoul, qqrd2e, ctx);
    this.gEwald = this.solver.gEwald;
  }

  /** Force-field hook (forcefield.ts init): the pair style that defines the M sites. */
  linkPair(pair: Pair | null): void { this.pair = pair instanceof PairTIP4PBase ? pair : null; }

  private model(): Tip4pModel {
    const m = this.pair?.siteModel() ?? null;
    if (!m) throw new StyleError('kspace_style pppm/tip4p needs a pair style lj/cut/tip4p/long or tip4p/long');
    return m;
  }

  compute(kc: KSpaceCompute): void {
    const s = kc.s;
    const n = s.n;
    const m = this.model();
    const sites = buildOwnedSites(n, s.x, s.type, s.id, kc.geom, m);
    const x = Float64Array.from(s.x.subarray(0, 3 * n));
    for (let i = 0; i < n; i++) {
      if (s.type[i] === m.otype) for (let c = 0; c < 3; c++) x[3 * i + c] = sites.M[3 * i + c];
    }
    const fs = new Float64Array(3 * n);
    this.solver.compute({ ...kc, s: { ...s, x }, f: fs });
    for (let i = 0; i < n; i++) {
      if (s.type[i] === m.otype) {
        const o = i, a = sites.h1[i], b = sites.h2[i];
        for (let c = 0; c < 3; c++) {
          const F = fs[3 * i + c];
          kc.f[3 * o + c] += (1 - m.alpha) * F;
          kc.f[3 * a + c] += 0.5 * m.alpha * F;
          kc.f[3 * b + c] += 0.5 * m.alpha * F;
        }
      } else {
        for (let c = 0; c < 3; c++) kc.f[3 * i + c] += fs[3 * i + c];
      }
    }
  }
}
