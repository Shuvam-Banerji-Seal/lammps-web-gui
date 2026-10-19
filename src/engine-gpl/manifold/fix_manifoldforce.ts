/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANIFOLD/fix_manifoldforce.cpp and
 * fix_manifoldforce.h at commit b332f28 (fix manifoldforce: removes the force component normal to a manifold, for
 * minimization on it). MANIFOLD package by Stefan Paquay (Eindhoven University of Technology), GNU General Public
 * License; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/fix_manifoldforce.html. Syntax: fix ID group manifoldforce manifold manifold-args (constants
 * only). Native init refuses any min_style other than hftn and quickmin, before MD runs as well.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { createManifold, type Manifold } from './manifold';
import { numeric } from './fix_nve_manifold_rattle';

export class FixManifoldForce extends Fix {
  readonly style = 'manifoldforce';
  private readonly man: Manifold;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // native narg < 5: fewer than 2 arguments here
    if (args.length < 2) throw new StyleError('Illegal fix manifoldforce! No manifold given');
    const s = sys.state;
    const m = createManifold(args[0], { box: { lo: s.box.lo, hi: s.box.hi }, warn: (t) => sys.warn(t) });
    if (!m) throw new StyleError(`Manifold pointer for manifold '${args[0]}' was NULL for some reason`);
    this.man = m;
    const nvars = m.nparams;
    if (args.length + 3 < nvars + 4) throw new StyleError(`Manifold ${args[0]} needs at least ${nvars} argument(s)!`);
    m.params = new Float64Array(nvars);
    for (let i = 0; i < nvars; ++i) {
      const a = args[i + 1];
      if (a !== undefined && a.startsWith('v_')) throw new StyleError('Equal-style variables not allowed with fix manifoldforce');
      m.params[i] = numeric(a);
    }
    m.postParamInit();
  }

  init(): void {
    const ms = this.sys.minStyle;
    if (ms !== 'hftn' && ms !== 'quickmin') throw new StyleError('Fix manifoldforce requires min_style hftn or quickmin');
  }

  minSetup(): void { this.postForce(); }

  /** post_force: f -= (f . n) n / |n|^2 for every atom of the group. */
  postForce(): void {
    const s = this.sys.state;
    const { x, f, mask } = s;
    const bit = this.groupBit;
    const xi = new Float64Array(3);
    const n = new Float64Array(3);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      xi[0] = x[3 * i]; xi[1] = x[3 * i + 1]; xi[2] = x[3 * i + 2];
      this.man.n(xi, n);
      const invn2 = 1.0 / (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      const dot = f[3 * i] * n[0] + f[3 * i + 1] * n[1] + f[3 * i + 2] * n[2];
      f[3 * i] -= dot * n[0] * invn2;
      f[3 * i + 1] -= dot * n[1] * invn2;
      f[3 * i + 2] -= dot * n[2] * invn2;
    }
  }

  minPostForce(): void { this.postForce(); }
}
