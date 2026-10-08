import { Fix } from './fix';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';
import { granularContact, newContactOut, parseGranularSpec, PairGranular, type ContactOut, type Pm } from '../force/pair/granular';

/** The classic fstyles of fix wall/gran/region (hooke, hooke/history, hertz/history). */
const CLASSIC_FSTYLES = ['hooke', 'hooke/history', 'hertz/history'] as const;
import { BlockRegion, ConeRegion, SphereRegion } from '../region';

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

/*
 * fix wall/gran granular and fix wall/gran/region granular — docs.lammps.org/fix_wall_gran.html and
 * fix_wall_gran_region.html (plans/lammps-docs). "For *granular*, *fstyle_params* are set using the same
 * syntax as for the *pair_coeff* command of pair_style granular"; the wall/particle contact is the
 * granular contact of pair_style granular with the wall as the second particle:
 * "delta = radius - r = overlap of particle with wall, m_eff = mass of particle, and the effective radius
 * of contact = RiRj/Ri+Rj is set to the radius of the particle" (flat walls, fix_wall_gran.rst);
 * for region walls "the effective radius is calculated using the radius of the particle and the radius
 * of curvature of the wall at the contact point ... The radius of curvature can be negative for a concave
 * wall section, e.g. the interior of cylinder" (fix_wall_gran_region.rst).
 * "The distance between a particle and the region boundary is the distance to the nearest point on the
 * region surface" and "for region_style block, a particle in the interior, near a corner of the block,
 * could feel wall forces from 1, 2, or 3 faces of the block" (fix_wall_gran_region.rst).
 * E_eff of the wall/particle pair: E_eff = 960 / (2 (1 - 0.2^2)) = 500 (the worked example of the fix_wall_gran.rst note); the
 * pair_coeff of pair_style granular is NOT used for the wall ("Any pair coefficients defined by pair_style
 * granular are not taken into consideration").
 * Supported here: planes xplane/yplane/zplane (lo and hi, NULL allowed; wiggle and shear motion) for
 * wall/gran; region walls for wall/gran/region with block, sphere (interior) and cylinder (radlo = radhi,
 * interior) regions, static only. Side-out regions, cones, the contacts and temperature keywords, and
 * dynamic regions throw a StyleError.
 */

const WALL_WORDS = ['xplane', 'yplane', 'zplane', 'zcylinder', 'region'] as const;

/** A wall/particle contact candidate: history key, distance to the surface, unit normal (wall to particle), effective radius. */
interface WallElement { key: number; dist: number; nx: number; ny: number; nz: number; Rf: number }

export class FixWallGranGranular extends Fix {
  readonly style: string;
  private readonly pm: Pm;
  /** Legacy hertz/history: forces scaled by sqrt(delta R). */
  private legacyHertz = false;
  private readonly planes: Plane[] = [];
  private motion: Motion | null = null;
  private readonly regionId: string | null;
  private readonly step0: number;
  private shear = new Map<string, Float64Array>();
  private readonly out: ContactOut = newContactOut();

  constructor(sys: System, id: string, group: string, args: string[], regionMode: boolean) {
    super(sys, id, group, args);
    this.style = regionMode ? 'wall/gran/region' : 'wall/gran';
    const classic = (CLASSIC_FSTYLES as readonly string[]).includes(args[0]);
    if (args[0] !== 'granular' && !classic) throw new StyleError(`fix ${id} ${this.style}: expected fstyle granular or hooke, hooke/history, hertz/history`);
    if (classic && !regionMode) throw new StyleError(`fix ${id} wall/gran: classic fstyle ${args[0]} is the plane legacy style (FixWallGran)`);
    let k = 1;
    let pmOut: Pm;
    if (classic) {
      // fstyle_params: Kn Kt gamma_n gamma_t xmu dampflag [limit_damping], then the wallstyle region ID
      if (args.length < 8) throw new StyleError(`fix ${id} ${this.style} ${args[0]}: needs Kn Kt gamma_n gamma_t xmu dampflag`);
      const kn = parseNum(args[1], 'Kn');
      const kt = args[2] === 'NULL' ? (kn * 2) / 7 : parseNum(args[2], 'Kt');
      const gn = parseNum(args[3], 'gamma_n');
      const gt = args[4] === 'NULL' ? 0.5 * gn : parseNum(args[4], 'gamma_t');
      const xmu = parseNum(args[5], 'xmu');
      if (args[6] !== '0' && args[6] !== '1') throw new StyleError(`fix ${id} ${this.style}: dampflag must be 0 or 1, got '${args[6]}'`);
      const dampflag = Number(args[6]);
      k = 7;
      let limit = false;
      if (args[k] === 'limit_damping') { limit = true; k++; }
      if (kn < 0 || kt < 0 || gn < 0 || gt < 0 || xmu < 0 || xmu > 10000) {
        throw new StyleError(`fix ${id} ${this.style}: Kn, Kt, gamma_n, gamma_t must be >= 0 and xmu in 0..1e4`);
      }
      const gtEff = dampflag === 0 ? 0 : gt;
      if (gn === 0 && gtEff > 0) throw new StyleError(`fix ${id} ${this.style}: gamma_t > 0 needs gamma_n > 0 (the tangential damping scales with gamma_n)`);
      const hist = args[0] !== 'hooke';
      this.legacyHertz = args[0] === 'hertz/history';
      pmOut = {
        normal: 'hooke', kn, Eeff: 0, gamma: 0, eta: gn, damp: 'mass_velocity',
        tang: hist ? 'linear_history' : 'linear_nohistory', kt, xgt: gn > 0 ? gtEff / gn : 0, mu: xmu,
        roll: 'none', kr: 0, gr: 0, mr: 0, twist: 'none', kw: 0, gw: 0, mw: 0, limit, cutoff: -1, sig: 'legacy',
      };
    } else {
      while (k < args.length && !(WALL_WORDS as readonly string[]).includes(args[k])) k++;
      if (k >= args.length) throw new StyleError(`fix ${id} ${this.style} granular: missing wallstyle`);
      const sp = parseGranularSpec(args.slice(1, k));
      pmOut = new PairGranular().wallParams(sp);
    }
    this.pm = pmOut;
    const ws = args[k];
    if (regionMode) {
      if (ws !== 'region') throw new StyleError(`fix ${id} wall/gran/region: wallstyle must be region, got '${ws}'`);
      this.regionId = args[k + 1] ?? null;
      if (!this.regionId) throw new StyleError(`fix ${id} wall/gran/region: missing region ID`);
      this.sys.region(this.regionId);
      k += 2;
    } else {
      this.regionId = null;
      if (ws === 'region') throw new StyleError(`fix ${id} wall/gran: wallstyle region is wall/gran/region`);
      if (ws === 'zcylinder') throw new StyleError('The zcylinder keyword has been removed. Please use fix wall/gran/region instead.');
      const dim = (ws === 'xplane' ? 0 : ws === 'yplane' ? 1 : 2) as Dim;
      const lo = args[k + 1], hi = args[k + 2];
      if (lo === undefined || hi === undefined) throw new StyleError(`fix ${id} wall/gran ${ws} needs lo and hi (either may be NULL)`);
      if (lo === 'NULL' && hi === 'NULL') throw new StyleError(`fix ${id} wall/gran ${ws}: lo and hi are both NULL, no wall is defined`);
      if (lo !== 'NULL') this.planes.push({ dim, lo: true, coord: parseNum(lo, `${ws} lo`) });
      if (hi !== 'NULL') this.planes.push({ dim, lo: false, coord: parseNum(hi, `${ws} hi`) });
      k += 3;
    }
    this.step0 = sys.state.step;
    while (k < args.length) {
      const w = args[k];
      if (w === 'wiggle' && !regionMode) {
        const d = DIM_OF[args[k + 1]];
        if (d === undefined) throw new StyleError(`fix ${id} wall/gran wiggle: dim must be x, y or z`);
        const amp = parseNum(args[k + 2], 'wiggle amplitude');
        const period = parseNum(args[k + 3], 'wiggle period');
        if (!(period > 0)) throw new StyleError(`fix ${id} wall/gran wiggle: period must be > 0`);
        this.motion = { kind: 'wiggle', dim: d, amp, omega: (2 * Math.PI) / period, vshear: 0 };
        k += 4;
      } else if (w === 'shear' && !regionMode) {
        const d = DIM_OF[args[k + 1]];
        if (d === undefined) throw new StyleError(`fix ${id} wall/gran shear: dim must be x, y or z`);
        this.motion = { kind: 'shear', dim: d, amp: 0, omega: 0, vshear: parseNum(args[k + 2], 'shear velocity') };
        k += 3;
      } else if (w === 'contacts') {
        throw new StyleError(`fix ${id} ${this.style} keyword contacts is not supported yet`);
      } else if (w === 'temperature') {
        throw new StyleError(`fix ${id} ${this.style} keyword temperature is not supported yet (needs a heat model)`);
      } else {
        throw new StyleError(`fix ${id} ${this.style}: unknown keyword or argument '${w}'`);
      }
    }
    if (this.motion && this.planes.length) {
      for (const p of this.planes) {
        if (this.motion.kind === 'shear' && this.motion.dim === p.dim) throw new StyleError('Invalid shear direction for fix wall/gran (shear must be tangential to the wall)');
      }
    }
  }

  init(): void {
    const s = this.sys.state;
    for (const p of this.planes) {
      if (s.dimension === 2 && p.dim === 2) throw new StyleError('fix wall/gran: cannot use a z wall in a 2d simulation');
      if (s.box.periodic[p.dim]) throw new StyleError('Cannot use wall in periodic dimension');
    }
    if (this.regionId) {
      const r = this.sys.region(this.regionId);
      if (r.dynamic) throw new StyleError(`fix ${this.id} wall/gran/region: a dynamic region is not supported`);
    }
  }

  postForce(): void { this.apply(true); }

  setup(): void { this.apply(false); }

  private pv(p: number | { variable: string; scale: number }): number {
    return typeof p === 'number' ? p : this.sys.regionEnv.variable(p.variable) * p.scale;
  }

  /** Contact candidates of one atom at its position (planes or region faces; static walls only). */
  private regionElements(x: number, y: number, z: number, R: number, out: WallElement[]): void {
    out.length = 0;
    if (this.regionId) {
      const r = this.sys.region(this.regionId);
      if (!r.interior) throw new StyleError(`fix ${this.id} wall/gran/region: side-out regions are not supported`);
      if (r instanceof BlockRegion) {
        const b = r.b.map((q) => this.pv(q));
        const lo = [b[0], b[2], b[4]], hi = [b[1], b[3], b[5]];
        const p = [x, y, z];
        for (let a = 0; a < 3; a++) {
          const dlo = p[a] - lo[a], dhi = hi[a] - p[a];
          if (dlo <= 0 || dhi <= 0) continue;
          const e = [0, 0, 0];
          e[a] = 1;
          out.push({ key: 2 * a, dist: dlo, nx: e[0], ny: e[1], nz: e[2], Rf: R });
          out.push({ key: 2 * a + 1, dist: dhi, nx: -e[0], ny: -e[1], nz: -e[2], Rf: R });
        }
      } else if (r instanceof SphereRegion) {
        const c = r.c.map((q) => this.pv(q));
        const Rs = this.pv(r.r);
        const dx = x - c[0], dy = y - c[1], dz = z - c[2];
        const rho = Math.hypot(dx, dy, dz);
        if (rho === 0 || !(Rs - rho > 0)) return;
        // concave wall: radius of curvature -Rs, so R_eff = R (-Rs) / (R - Rs)
        // measured with native LAMMPS: the interior sphere has curvature radius -Rs (R_eff = R Rw / (R + Rw))
        out.push({ key: 0, dist: Rs - rho, nx: -dx / rho, ny: -dy / rho, nz: -dz / rho, Rf: (R * -Rs) / (R - Rs) });
      } else if (r instanceof ConeRegion) {
        // lateral surface: the generator radius rho(a) = rl + slope (a - lo) of the cone (a cylinder has slope 0).
        // Measured with native LAMMPS (black box, single sphere, Hertz k_n): the overlap is the distance to the
        // generator, the normal is the gradient of the generator (checked against the hooke forces), and the
        // curvature radius of the wall at the contact point is Rw = -2 rho_s, rho_s = radial distance of the
        // surface point (cylinder: rho_s = Rc; cones: 2.94 = 2 x 1.47 and 3.44 = 2 x 1.72 at two points).
        const rl = this.pv(r.radlo), rh = this.pv(r.radhi);
        const axis = r.axis;
        const c1 = this.pv(r.c1), c2 = this.pv(r.c2), lo = this.pv(r.lo), hi = this.pv(r.hi);
        const p = [x, y, z];
        const a = p[axis];
        const [d1, d2] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
        const e1 = p[d1] - c1, e2 = p[d2] - c2;
        const rho = Math.hypot(e1, e2);
        const slope = hi > lo ? (rh - rl) / (hi - lo) : 0;
        const f = rho - (rl + slope * (a - lo));
        const q = 1 + slope * slope;
        if (rho > 0 && f < 0 && a - lo > 0 && hi - a > 0) {
          // inside the generator: unit normal from the wall to the particle = (-e_rho + slope e_axis) / sqrt(q)
          const u = [0, 0, 0];
          u[d1] = (-e1 / rho) / Math.sqrt(q);
          u[d2] = (-e2 / rho) / Math.sqrt(q);
          u[axis] = slope / Math.sqrt(q);
          const rhoS = rho - f / q;
          const Rw = -2 * rhoS;
          out.push({ key: 0, dist: -f / Math.sqrt(q), nx: u[0], ny: u[1], nz: u[2], Rf: (R * Rw) / (R + Rw) });
        }
        // the flat caps of the cone (and of the cylinder): side in, the axial faces
        const ax = [0, 0, 0];
        ax[axis] = 1;
        if (a - lo > 0) out.push({ key: 1, dist: a - lo, nx: ax[0], ny: ax[1], nz: ax[2], Rf: R });
        if (hi - a > 0) out.push({ key: 2, dist: hi - a, nx: -ax[0], ny: -ax[1], nz: -ax[2], Rf: R });
      } else {
        throw new StyleError(`fix ${this.id} wall/gran/region: region style ${r.style} is not supported (block, sphere, cylinder)`);
      }
      return;
    }
  }

  private apply(update: boolean): void {
    const s = this.sys.state;
    const { x, v, f, id, mask, radius, rmass, omega, torque } = s;
    if (!radius || !rmass || !omega || !torque) throw new StyleError('fix wall/gran requires atom_style sphere');
    const dt = s.dt;
    const delta = (s.step - this.step0) * dt;
    const mv = [0, 0, 0];
    let off = 0;
    const m = this.motion;
    if (m?.kind === 'wiggle') {
      off = m.amp - m.amp * Math.cos(m.omega * delta);
      mv[m.dim] = m.amp * m.omega * Math.sin(m.omega * delta);
    } else if (m?.kind === 'shear') {
      mv[m.dim] = m.vshear;
    }
    const bit = this.groupBit;
    const pm = this.pm;
    const jkr = pm.normal === 'jkr';
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const R = radius[i];
      const mi = rmass[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const cand: WallElement[] = [];
      if (this.regionId) {
        this.regionElements(xi, yi, zi, R, cand);
      } else {
        for (let k = 0; k < this.planes.length; k++) {
          const p = this.planes[k];
          const c = p.coord + (m?.kind === 'wiggle' && m.dim === p.dim ? off : 0);
          const sd = x[3 * i + p.dim] - c;
          const dist = Math.abs(sd);
          if (sd === 0) throw new StyleError(`fix wall/gran: particle ${id[i]} is at the wall position (zero distance); native LAMMPS gives NaN forces`);
          const n = [0, 0, 0];
          n[p.dim] = sd > 0 ? 1 : -1;
          cand.push({ key: k, dist, nx: n[0], ny: n[1], nz: n[2], Rf: R });
        }
      }
      for (const e of cand) {
        const key = `${e.key}:${id[i]}`;
        if (e.dist >= R && !jkr) {
          this.shear.delete(key);
          continue;
        }
        const sh = this.shear.get(key) ?? new Float64Array(8);
        const ok = granularContact({
          pm, nx: e.nx, ny: e.ny, nz: e.nz, r: e.dist, delta: R - e.dist, Rf: e.Rf, ri: R, rj: 0, meff: mi,
          vrx: v[3 * i] - mv[0], vry: v[3 * i + 1] - mv[1], vrz: v[3 * i + 2] - mv[2],
          oix: omega[3 * i], oiy: omega[3 * i + 1], oiz: omega[3 * i + 2],
          ojx: 0, ojy: 0, ojz: 0, dt, update, sg: 1, surfaceArm: true,
          poly: this.legacyHertz ? Math.sqrt((R - e.dist) * e.Rf) : 1,
        }, sh, this.out, this.shear.has(key));
        if (!ok) {
          this.shear.delete(key);
          continue;
        }
        if (this.out.hist) this.shear.set(key, sh);
        f[3 * i] += this.out.fx; f[3 * i + 1] += this.out.fy; f[3 * i + 2] += this.out.fz;
        torque[3 * i] += this.out.tix; torque[3 * i + 1] += this.out.tiy; torque[3 * i + 2] += this.out.tiz;
      }
    }
  }
}
