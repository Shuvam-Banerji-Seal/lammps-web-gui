import { Fix } from './fix';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';

/*
 * fix ID group-ID wall/gran fstyle fstyle_params wallstyle args keyword values ...
 * — docs.lammps.org/fix_wall_gran.html (fix_wall_gran.rst). Supported here:
 * fstyle hooke, hooke/history, hertz/history; wallstyle xplane, yplane,
 * zplane; keywords wiggle and shear. Everything else throws a StyleError
 * naming it: fstyle granular, keyword contacts, keyword temperature, the
 * zcylinder wallstyle, and fix wall/gran/region.
 *
 * "The equation for the force between the wall and particles touching it is
 * the same as the corresponding equation on the :doc:`pair_style gran/\* <pair_gran>` and
 * :doc:`pair_style granular <pair_granular>` doc pages, in the limit of
 * one of the two particles going to infinite radius and mass (flat wall)."
 * Specifically, delta = radius - r = overlap of particle with wall, m_eff =
 * mass of particle, and the effective radius of contact is the radius of the
 * particle (paraphrased from the page).
 * "The effective mass *m_eff* ... is the mass of the particle for
 * particle/wall interactions (mass of wall is infinite)."
 * The pair equations (docs.lammps.org/pair_gran.html) are
 *   F_{hk} = (k_n \delta \mathbf{n}_{ij} - m_{eff} \gamma_n\mathbf{ v}_n) -
 *            (k_t \boldsymbol{\Delta} \mathbf{s}_t + m_{eff} \gamma_t \mathbf{v}_t)
 *   F_{hz} = \sqrt{\delta} \sqrt{\frac{R_i R_j}{R_i + R_j}} F_{hk}
 * with R_i R_j / (R_i + R_j) -> R_i for a flat wall, so the Hertz factor is
 * sqrt(delta * R). "If a NULL is used for *Kt*, then a default value is used
 * where *Kt* = 2/7 *Kn*" and "If a NULL is used for *gamma_t*, then a default
 * value is used where *gamma_t* = 1/2 *gamma_n*."
 *
 * Measured with native LAMMPS (black box, one sphere of diameter 1, density 1,
 * against a plane at 0.0 with the sphere at z = 0.4, so delta = 0.1):
 * - a plane wall acts on both sides: the normal is sign(x_i - coord) along the
 *   wall axis and the overlap is radius - |x_i - coord|; a sphere at x = 0.4
 *   with a hi wall at 0.0 is pushed to +x (Fx = 200 = Kn delta), at x = -0.4
 *   with a lo wall at 0.0 to -x (Fx = -200);
 * - no contact for |x_i - coord| >= radius (x = 0.7, hi wall at 0: F = 0);
 * - with the sphere at the wall position native returns NaN forces;
 * - Fz = 200 without damping; with gn = 50 and vz = -0.3 Fz = 207.854 (the
 *   damping term m gn |v_n| with m = 0.5236); the wall is at rest unless moved;
 * - hertz/history: Fz = 2000 * 0.1 * sqrt(0.1 * 0.5) = 44.7214 (no damping);
 * - the tangential relative velocity uses the wall velocity: a sphere with
 *   vx = 0.2 and gt = 30 (m gt vt = 3.14159) gives Fx = -3.14159 and
 *   tau_y = +1.5708 = -R (n x F_t)_y with n = +z (torque -R n x F_t, as in pair gran);
 * - the Coulomb cap is xmu |F_n| (vx = 2, xmu = 0.1, Fn = 200: Fx = -20 and
 *   tau_y = +10); the capped tangential force is dropped at a contact with no
 *   stored shear (run setup): Fx = 0 at run 0 and -2 at the first run step;
 * - limit_damping zeroes the whole contact when F_n < 0 (gn = 1000, vz = 1:
 *   F_z = -323.599 without it, 0 with it); vz = 0.1 gives Fz = 147.640;
 * - hooke/history with dampflag 0: step n adds -kt n dt vt, so Fx = -0.0114286
 *   at the first step with kt = 2/7 * 2000, vx = 0.2, dt = 1e-4 (the displacement
 *   accumulates inside timesteps only, not at run setup);
 * - wiggle z 0.1 2.0 (omega = pi): at t = 0.1 (step 1000, dt = 1e-4) the wall
 *   is at 0.1 - 0.1 cos(0.1 pi) = 0.004894 and Fz = 209.7887 without damping,
 *   212.3303 with gn = 50, gt = 30 (the wall velocity 0.1 pi sin(0.1 pi) enters
 *   the damping term); at t = 0.2, Fz = 243.031. Time is counted from when the
 *   fix was specified (step * dt);
 * - shear y 0.3 on zplane: Fy = +4.71239 and tau_x = +2.35619 (the wall moves
 *   tangentially at vshear, so v_t = -0.3 y);
 * - wiggle x 0.1 2.0 on zplane (in-plane): Fx = 1.52494 at t = 0.1;
 * - native errors, reused verbatim as the StyleError messages: Cannot wiggle
 *   and shear fix wall/gran; Invalid shear direction for fix wall/gran (shear
 *   along the wall normal, e.g. shear z on zplane); The zcylinder keyword has
 *   been removed. Please use fix wall/gran/region instead.; Cannot use wall in
 *   periodic dimension.
 *
 * Restrictions kept from the doc: "Both keywords cannot be used together." and
 * "Any dimension (xyz) that has a granular wall must be non-periodic."
 * The wall is applied as a post_force of every timestep (fix postForce, which
 * is inside a timestep); the setup call (run start) computes the forces with
 * the current shear history but does not advance it.
 */

type FStyle = 'hooke' | 'hooke/history' | 'hertz/history';
type Dim = 0 | 1 | 2;
const DIM_OF: Record<string, Dim> = { x: 0, y: 1, z: 2 };
const STYLE_NAMES: readonly FStyle[] = ['hooke', 'hooke/history', 'hertz/history'];

/** wiggle or shear motion of the walls (the same for every wall of the fix). */
interface Motion {
  kind: 'wiggle' | 'shear';
  dim: Dim;
  amp: number;
  omega: number;
  vshear: number;
}

/** One plane: the axis, the side and its position (undefined = NULL). */
interface Plane {
  dim: Dim;
  lo: boolean;
  coord: number;
}

export class FixWallGran extends Fix {
  readonly style = 'wall/gran';
  private readonly fstyle: FStyle;
  private readonly hertz: boolean;
  private readonly history: boolean;
  private kn = 0; private kt = 0; private gn = 0; private gt = 0; private xmu = 0;
  private dampflag = 0;
  private limitDamping = false;
  private readonly planes: Plane[] = [];
  private motion: Motion | null = null;
  private readonly wallDim: Dim;
  /** Step when the fix was specified: wiggle time is (step - step0) * dt. */
  private readonly step0: number;
  /** Shear displacement per (plane index, atom ID) while in contact. */
  private shear = new Map<string, Float64Array>();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const usage = 'usage: fix ID group-ID wall/gran fstyle fstyle_params wallstyle args [keyword values ...]';
    if (args.length < 1) throw new StyleError(usage);
    const fs = args[0];
    if (fs === 'granular') {
      throw new StyleError('fix wall/gran granular (fstyle granular) is not supported yet; use hooke, hooke/history or hertz/history');
    }
    if (!(STYLE_NAMES as readonly string[]).includes(fs)) {
      throw new StyleError(`fix ${id} wall/gran: unknown fstyle '${fs}' (hooke, hooke/history, hertz/history)`);
    }
    this.fstyle = fs as FStyle;
    this.hertz = this.fstyle === 'hertz/history';
    this.history = this.fstyle !== 'hooke';
    // fstyle_params: Kn Kt gamma_n gamma_t xmu dampflag [limit_damping]
    if (args.length < 8) throw new StyleError(`${usage} (fstyle ${fs} needs Kn Kt gamma_n gamma_t xmu dampflag)`);
    this.kn = parseNum(args[1], 'Kn');
    this.kt = args[2] === 'NULL' ? (this.kn * 2) / 7 : parseNum(args[2], 'Kt');
    this.gn = parseNum(args[3], 'gamma_n');
    this.gt = args[4] === 'NULL' ? 0.5 * this.gn : parseNum(args[4], 'gamma_t');
    this.xmu = parseNum(args[5], 'xmu');
    if (args[6] !== '0' && args[6] !== '1') throw new StyleError(`fix ${id} wall/gran: dampflag must be 0 or 1, got '${args[6]}'`);
    this.dampflag = Number(args[6]);
    let k = 7;
    if (args[k] === 'limit_damping') { this.limitDamping = true; k++; }
    if (this.kn < 0 || this.kt < 0 || this.gn < 0 || this.gt < 0 || this.xmu < 0 || this.xmu > 10000) {
      throw new StyleError(`fix ${id} wall/gran: Kn, Kt, gamma_n, gamma_t must be >= 0 and xmu in 0..1e4`);
    }
    if (this.dampflag === 0) this.gt = 0;

    // wallstyle args; the planar styles take lo hi (NULL allowed, not both)
    const ws = args[k++];
    if (ws === 'zcylinder') {
      throw new StyleError('The zcylinder keyword has been removed. Please use fix wall/gran/region instead.');
    }
    if (ws !== 'xplane' && ws !== 'yplane' && ws !== 'zplane') {
      throw new StyleError(`fix ${id} wall/gran: unsupported wallstyle '${ws ?? ''}' (xplane, yplane, zplane)`);
    }
    const dim = (ws === 'xplane' ? 0 : ws === 'yplane' ? 1 : 2) as Dim;
    this.wallDim = dim;
    const lo = args[k++], hi = args[k++];
    if (lo === undefined || hi === undefined) throw new StyleError(`fix ${id} wall/gran ${ws} needs lo and hi (either may be NULL)`);
    if (lo === 'NULL' && hi === 'NULL') throw new StyleError(`fix ${id} wall/gran ${ws}: lo and hi are both NULL, no wall is defined`);
    if (lo !== 'NULL') this.planes.push({ dim, lo: true, coord: parseNum(lo, `${ws} lo`) });
    if (hi !== 'NULL') this.planes.push({ dim, lo: false, coord: parseNum(hi, `${ws} hi`) });

    // keyword values
    while (k < args.length) {
      const w = args[k];
      if (w === 'wiggle') {
        const d = DIM_OF[args[k + 1]];
        if (d === undefined) throw new StyleError(`fix ${id} wall/gran wiggle: dim must be x, y or z, got '${args[k + 1] ?? ''}'`);
        const amp = parseNum(args[k + 2], 'wiggle amplitude');
        const period = parseNum(args[k + 3], 'wiggle period');
        if (!(period > 0)) throw new StyleError(`fix ${id} wall/gran wiggle: period must be > 0`);
        this.setMotion({ kind: 'wiggle', dim: d, amp, omega: (2 * Math.PI) / period, vshear: 0 }, id);
        k += 4;
      } else if (w === 'shear') {
        const d = DIM_OF[args[k + 1]];
        if (d === undefined) throw new StyleError(`fix ${id} wall/gran shear: dim must be x, y or z, got '${args[k + 1] ?? ''}'`);
        const vshear = parseNum(args[k + 2], 'shear velocity');
        this.setMotion({ kind: 'shear', dim: d, amp: 0, omega: 0, vshear }, id);
        k += 3;
      } else if (w === 'contacts') {
        throw new StyleError('fix wall/gran keyword contacts is not supported yet');
      } else if (w === 'temperature') {
        throw new StyleError('fix wall/gran keyword temperature is not supported yet (needs a heat model)');
      } else {
        throw new StyleError(`fix ${id} wall/gran: unknown keyword or argument '${w}'`);
      }
    }
    this.step0 = sys.state.step;
  }

  private setMotion(m: Motion, id: string): void {
    if (this.motion) throw new StyleError(`fix ${id} wall/gran: Cannot wiggle and shear fix wall/gran`);
    // "Note that if the dimension is in the plane of the wall, this is effectively a shearing motion."
    if (m.kind === 'shear' && m.dim === this.wallDim) {
      throw new StyleError(`Invalid shear direction for fix wall/gran (shear must be tangential to the wall)`);
    }
    this.motion = m;
  }

  init(): void {
    const s = this.sys.state;
    for (const p of this.planes) {
      if (s.dimension === 2 && p.dim === 2) throw new StyleError('fix wall/gran: cannot use a z wall in a 2d simulation');
      if (s.box.periodic[p.dim]) throw new StyleError('Cannot use wall in periodic dimension');
    }
  }

  /** Forces of the walls are applied in post_force, as in every timestep. */
  postForce(): void { this.apply(true); }

  /** A run start computes the forces with the stored shear but does not advance it. */
  setup(): void { this.apply(false); }

  private apply(update: boolean): void {
    const s = this.sys.state;
    const { x, v, f, id, mask, radius, rmass, omega, torque } = s;
    if (!radius || !rmass || !omega || !torque) throw new StyleError('fix wall/gran requires atom_style sphere');
    const dt = s.dt;
    const delta = (s.step - this.step0) * dt;
    // wall velocity (mv) and the wiggle offset of the wall position along its normal
    const mv = [0, 0, 0];
    let off = 0;
    const m = this.motion;
    if (m?.kind === 'wiggle') {
      off = m.amp - m.amp * Math.cos(m.omega * delta);
      mv[m.dim] = m.amp * m.omega * Math.sin(m.omega * delta);
    } else if (m?.kind === 'shear') {
      mv[m.dim] = m.vshear;
    }
    const { kn, kt, gn, gt, xmu } = this;
    const bit = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const R = radius[i];
      const mi = rmass[i];
      for (let k = 0; k < this.planes.length; k++) {
        const p = this.planes[k];
        const d = p.dim;
        const c = p.coord + (m?.kind === 'wiggle' && m.dim === d ? off : 0);
        const sd = x[3 * i + d] - c;
        const key = `${k}:${id[i]}`;
        const dist = Math.abs(sd);
        if (dist >= R) { this.shear.delete(key); continue; }
        if (sd === 0) {
          throw new StyleError(`fix wall/gran: particle ${id[i]} is at the wall position (zero distance); native LAMMPS gives NaN forces`);
        }
        // "delta = radius - r = overlap of particle with wall"; the normal points from the wall to the particle
        const sg = sd > 0 ? 1 : -1;
        const n = [0, 0, 0];
        n[d] = sg;
        const ov = R - dist;
        const vr0 = v[3 * i] - mv[0], vr1 = v[3 * i + 1] - mv[1], vr2 = v[3 * i + 2] - mv[2];
        const vn = vr0 * n[0] + vr1 * n[1] + vr2 * n[2];
        const poly = this.hertz ? Math.sqrt(ov * R) : 1;
        const fn = poly * (kn * ov - mi * gn * vn);
        if (this.limitDamping && fn < 0) { this.shear.delete(key); continue; }
        // tangential relative velocity: v_t = v_r - (v_r.n) n - (R w) x n
        const wx = R * omega[3 * i], wy = R * omega[3 * i + 1], wz = R * omega[3 * i + 2];
        const cx = wy * n[2] - wz * n[1], cy = wz * n[0] - wx * n[2], cz = wx * n[1] - wy * n[0];
        const vt0 = vr0 - vn * n[0] - cx, vt1 = vr1 - vn * n[1] - cy, vt2 = vr2 - vn * n[2] - cz;
        let sh: Float64Array = this.shear.get(key) ?? new Float64Array(3);
        if (this.history) {
          sh = Float64Array.from(sh);
          if (update) {
            sh[0] += vt0 * dt; sh[1] += vt1 * dt; sh[2] += vt2 * dt;
            // keep the displacement in the tangent plane
            const pr = sh[0] * n[0] + sh[1] * n[1] + sh[2] * n[2];
            sh[0] -= pr * n[0]; sh[1] -= pr * n[1]; sh[2] -= pr * n[2];
          }
        }
        let ft0 = -(kt * sh[0] + mi * gt * vt0) * poly;
        let ft1 = -(kt * sh[1] + mi * gt * vt1) * poly;
        let ft2 = -(kt * sh[2] + mi * gt * vt2) * poly;
        const fsMag = Math.sqrt(ft0 * ft0 + ft1 * ft1 + ft2 * ft2);
        const cap = xmu * Math.abs(fn);
        if (fsMag > cap) {
          let scale = fsMag > 0 ? cap / fsMag : 0;
          if (this.history) {
            // measured: with no stored shear (a contact at run setup) native drops the capped force
            if (sh[0] === 0 && sh[1] === 0 && sh[2] === 0) scale = 0;
            else if (kt > 0) {
              const cc = (mi * gt) / kt;
              for (let q = 0; q < 3; q++) {
                const vq = q === 0 ? vt0 : q === 1 ? vt1 : vt2;
                sh[q] = scale * (sh[q] + cc * vq) - cc * vq;
              }
            }
          }
          ft0 *= scale; ft1 *= scale; ft2 *= scale;
        }
        if (this.history) this.shear.set(key, sh);
        // force on the particle: normal along n plus the tangential part
        f[3 * i] += fn * n[0] + ft0;
        f[3 * i + 1] += fn * n[1] + ft1;
        f[3 * i + 2] += fn * n[2] + ft2;
        // torque on the particle: -R n x F_t
        torque[3 * i] -= R * (n[1] * ft2 - n[2] * ft1);
        torque[3 * i + 1] -= R * (n[2] * ft0 - n[0] * ft2);
        torque[3 * i + 2] -= R * (n[0] * ft1 - n[1] * ft0);
      }
    }
  }
}
