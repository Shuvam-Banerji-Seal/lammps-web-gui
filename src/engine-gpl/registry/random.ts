/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the per-atom random streams of
 * src/random_park.cpp (RanPark::reset(int, double *)), src/random_mars.cpp (RanMars::select_subset) and their
 * callers in src/velocity.cpp (loop geom), src/displace_atoms.cpp (random) and src/set.cpp (Set::setrandom).
 * LAMMPS is Copyright Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (kind HOOKS); the contract is
 * src/engine/nativeRandom.ts.
 */
import type { NativeRandom } from '../../engine/nativeRandom';
import { NATIVE_RANDOM_WORDS } from '../../engine/nativeRandom';
import { RanPark } from '../random_park';
import { RanMars } from '../random_mars';

const TWO_PI = 2 * Math.PI;

/** RanPark seeded from one atom: RanPark::reset(seed, x[i]). */
const atomStream = (seed: number, x: number, y: number, z: number): RanPark => {
  const rp = new RanPark(1);
  rp.resetAtom(seed, x, y, z);
  return rp;
};

export const NATIVE: NativeRandom = {
  // velocity.cpp, loop GEOM: "vx = random->uniform() - 0.5" (or gaussian()), three draws in 2d and 3d
  velocityGeom(seed, x, y, z, gaussian, out) {
    const rp = atomStream(seed, x, y, z);
    for (let d = 0; d < 3; d++) out[d] = gaussian ? rp.gaussian() : rp.uniform() - 0.5;
  },
  // displace_atoms.cpp: "x[i][0] += dx * 2.0*(random->uniform()-0.5);" and the same for y, and z in 3d only
  displaceRandom(seed, x, y, z, dx, dy, dz, dimension, out) {
    const rp = atomStream(seed, x, y, z);
    out[0] = dx * 2.0 * (rp.uniform() - 0.5);
    out[1] = dy * 2.0 * (rp.uniform() - 0.5);
    out[2] = dimension === 3 ? dz * 2.0 * (rp.uniform() - 0.5) : 0;
  },
  // set.cpp, TYPE_FRACTION: "if (ranpark->uniform() > fraction) continue;"
  typeFraction(seed, x, y, z, fraction) {
    return !(atomStream(seed, x, y, z).uniform() > fraction);
  },
  // set.cpp, TYPE_RATIO and TYPE_SUBSET: RanMars(seed + me) with me = 0, then select_subset(nsubset, count)
  selectSubset(seed, nsubset, count) {
    const mark = new Uint8Array(count);
    const next = new Int32Array(count);
    new RanMars(seed).selectSubset(nsubset, count, mark, next);
    return mark;
  },
  // set.cpp, DIPOLE_RANDOM: three draws (two in 2d), then scaled to the length dlen
  dipoleRandom(seed, x, y, z, dlen, dimension, out) {
    const rp = atomStream(seed, x, y, z);
    if (dimension === 3) {
      const m0 = rp.uniform() - 0.5, m1 = rp.uniform() - 0.5, m2 = rp.uniform() - 0.5;
      const msq = m0 * m0 + m1 * m1 + m2 * m2;
      const scale = dlen / Math.sqrt(msq);
      out[0] = m0 * scale; out[1] = m1 * scale; out[2] = m2 * scale;
    } else {
      const m0 = rp.uniform() - 0.5, m1 = rp.uniform() - 0.5;
      const msq = m0 * m0 + m1 * m1;
      const scale = dlen / Math.sqrt(msq);
      out[0] = m0 * scale; out[1] = m1 * scale; out[2] = 0;
    }
  },
  // set.cpp, SPIN_ATOM_RANDOM: three draws (two in 2d), then scaled to unit length (the magnitude is set apart)
  spinRandom(seed, x, y, z, dimension, out) {
    const rp = atomStream(seed, x, y, z);
    if (dimension === 3) {
      const s0 = rp.uniform() - 0.5, s1 = rp.uniform() - 0.5, s2 = rp.uniform() - 0.5;
      const scale = 1.0 / Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2);
      out[0] = s0 * scale; out[1] = s1 * scale; out[2] = s2 * scale;
    } else {
      const s0 = rp.uniform() - 0.5, s1 = rp.uniform() - 0.5;
      const scale = 1.0 / Math.sqrt(s0 * s0 + s1 * s1);
      out[0] = s0 * scale; out[1] = s1 * scale; out[2] = 0;
    }
  },
  // set.cpp, QUAT_RANDOM: s, theta1, theta2 in 3d (Shoemake's subgroup method); theta2 only in 2d
  quatRandom(seed, x, y, z, dimension, out) {
    const rp = atomStream(seed, x, y, z);
    if (dimension === 3) {
      const s = rp.uniform();
      const t1 = Math.sqrt(1.0 - s);
      const t2 = Math.sqrt(s);
      const theta1 = TWO_PI * rp.uniform();
      const theta2 = TWO_PI * rp.uniform();
      out[0] = Math.cos(theta2) * t2;
      out[1] = Math.sin(theta1) * t1;
      out[2] = Math.cos(theta1) * t1;
      out[3] = Math.sin(theta2) * t2;
    } else {
      const theta2 = Math.PI * rp.uniform();
      out[0] = Math.cos(theta2); out[1] = 0; out[2] = 0; out[3] = Math.sin(theta2);
    }
  },
  // set.cpp, THETA_RANDOM: "avec_line->bonus[line[i]].theta = MY_2PI*ranpark->uniform();"
  thetaRandom(seed, x, y, z) {
    return TWO_PI * atomStream(seed, x, y, z).uniform();
  },
};

/** One entry per command word that needs the streams (lazyStyles.ts lists the same words). */
export const HOOKS: Record<string, NativeRandom> = Object.fromEntries(NATIVE_RANDOM_WORDS.map((w) => [w, NATIVE]));
