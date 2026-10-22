import { Fix } from './fix';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';
import type { SimState } from '../types';
import { granularContact, heatFlowOf, mdrPairOf, newContactOut, parseGranularSpec, PairGranular, type ContactOut, type MdrApi, type MdrMaterialCore, type Pm } from '../force/pair/granular';
import { contactMassArray, HEAT_FLOW, HEAT_TEMPERATURE } from '../atoms';

/** The classic fstyles of fix wall/gran/region (hooke, hooke/history, hertz/history). */
const CLASSIC_FSTYLES = ['hooke', 'hooke/history', 'hertz/history'] as const;
import { type Region, type SurfaceContact } from '../region';
import { libm } from '../libm';

/*
 * fix ID group-ID wall/gran fstyle fstyle_params wallstyle args keyword values ...
 * — docs.lammps.org/fix_wall_gran.html (fix_wall_gran.rst). This class: fstyle hooke, hooke/history,
 * hertz/history on wallstyle xplane, yplane, zplane, with the keywords wiggle, shear and contacts (fstyle
 * granular and fix wall/gran/region are FixWallGranGranular below). "If the :code:`contacts` option is used,
 * this fix generates a per-atom array" of the contact data. These styles have no heat model, so the temperature
 * keyword stops as native does ("Must define wall temperature with heat model").
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

/**
 * One plane: the axis, the side and its position (undefined = NULL). A position given as v_name
 * (fix_wall_gran.html, versionadded 2Sep2026) is re-evaluated every timestep: `variable` is the
 * equal-style variable name and `prev` the position of its last evaluation (undefined before the first).
 */
interface Plane {
  dim: Dim;
  lo: boolean;
  coord: number;
  variable?: string;
  prev?: number;
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
  /**
   * keyword contacts: the per-atom array of the contact data, as native fix_wall_gran.cpp post_force stores it after
   * clear_stored_contacts: column 1 flag (1 in contact), 2-4 the wall force on the atom, 5-7 the contact point, 8 radius.
   */
  private contactsOn = false;

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
    // fix_wall_gran.html (versionadded 2Sep2026): "The *lo* and *hi* wall positions can also be set by an equal-style
    // :doc:`variable <variable>`, specified as v_name" -- the value is read as a variable name, not parsed
    const wallPos = (w: string, what: string): Plane['variable'] | number => {
      if (!w.startsWith('v_') || w.length < 3) return parseNum(w, what);
      return w.slice(2);
    };
    if (lo !== 'NULL') {
      const c = wallPos(lo, `${ws} lo`);
      this.planes.push(typeof c === 'number' ? { dim, lo: true, coord: c } : { dim, lo: true, coord: NaN, variable: c });
    }
    if (hi !== 'NULL') {
      const c = wallPos(hi, `${ws} hi`);
      this.planes.push(typeof c === 'number' ? { dim, lo: false, coord: c } : { dim, lo: false, coord: NaN, variable: c });
    }

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
        // fix_wall_gran.cpp: "peratom_flag = 1; size_peratom_cols = 8 + model->nsvector" (no extra values here)
        this.contactsOn = true;
        this.peratomFlag = true;
        this.sizePeratomCols = 8;
        k += 1;
      } else if (w === 'temperature') {
        // fix_wall_gran.cpp: these styles have no heat model, so a wall temperature is "Must define wall
        // temperature with heat model"
        if (args[k + 1] === undefined) throw new StyleError('Illegal fix wall/gran command');
        throw new StyleError('Must define wall temperature with heat model');
      } else {
        // As native LAMMPS does (src/GRANULAR/fix_wall_gran.cpp FixWallGran::FixWallGran at b332f28): an unrecognised optional word stops with "Illegal fix wall/gran command".
        throw new StyleError('Illegal fix wall/gran command');
      }
    }
    this.step0 = sys.state.step;
    if (this.planes.some((p) => p.variable !== undefined)) {
      // "A wall position set by a variable cannot be combined with the *wiggle* or *shear* keyword."
      if (this.motion) throw new StyleError(`fix ${id} wall/gran: a wall position set by a variable cannot be combined with wiggle or shear`);
    }
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
    const cmass = contactMassArray(s);
    if (!radius || !rmass || !omega || !torque) throw new StyleError('fix wall/gran requires atom_style sphere');
    const dt = s.dt;
    const delta = (s.step - this.step0) * dt;
    // wall velocity (mv) and the wiggle offset of the wall position along its normal
    const mv = [0, 0, 0];
    let off = 0;
    const m = this.motion;
    if (m?.kind === 'wiggle') {
      off = m.amp - m.amp * libm.cos(m.omega * delta);
      mv[m.dim] = m.amp * m.omega * libm.sin(m.omega * delta);
    } else if (m?.kind === 'shear') {
      mv[m.dim] = m.vshear;
    }
    // planes set by a variable: the position is evaluated now (setup and every timestep), and the wall
    // velocity is the difference to the previous evaluation over dt (zero at the first one).
    // Measured with native LAMMPS (black box, one sphere of diameter 1 and density 1 at z = 1.7 under a zplane
    // hi wall v_zhi with zhi = 2.0 - 0.001 step, dt = 0.001, hooke, Kn 2000, gamma_n 50): the setup force is
    // -400 (wall velocity zero at the first evaluation), the post_force force of step 1 is -417.416, which needs
    // the wall velocity -1 = (zhi(1) - zhi(0)) / dt, so the setup evaluation is the previous value; vz after
    // steps 1 and 2 is -0.78057477705 and -1.5575082544 (a dump of vz, two steps).
    const planeMv: number[][] = this.planes.map(() => mv);
    const planeCoord = this.planes.map((p) => p.coord);
    this.planes.forEach((p, k) => {
      if (p.variable === undefined) return;
      const cur = this.sys.equalVariable(p.variable);
      if (!Number.isFinite(cur)) throw new StyleError(`fix ${id} wall/gran: variable ${p.variable} gave a non-finite wall position`);
      const vw = p.prev === undefined ? 0 : (cur - p.prev) / dt;
      p.prev = cur;
      p.coord = cur;
      planeCoord[k] = cur;
      const pm = [0, 0, 0];
      pm[p.dim] = vw;
      planeMv[k] = pm;
    });
    // "LAMMPS stops with an error if the *lo* wall position does not remain below the *hi* wall position during a run."
    this.planes.forEach((p, k) => {
      if (!p.lo || p.variable === undefined) return;
      this.planes.forEach((q, j) => {
        if (!q.lo && q.dim === p.dim && !(planeCoord[k] < planeCoord[j])) {
          throw new StyleError(`fix ${id} wall/gran: the lo wall position must remain below the hi wall position`);
        }
      });
    });
    const { kn, kt, gn, gt, xmu } = this;
    const bit = this.groupBit;
    if (this.contactsOn) {
      // clear_stored_contacts: the per-atom contact data restart every evaluation
      if (this.arrayAtom.length !== 8 * s.n) this.arrayAtom = new Float64Array(8 * s.n);
      else this.arrayAtom.fill(0);
    }
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const R = radius[i];
      const mi = cmass[i];
      for (let k = 0; k < this.planes.length; k++) {
        const p = this.planes[k];
        const d = p.dim;
        const mv = planeMv[k];
        const c = planeCoord[k] + (m?.kind === 'wiggle' && m.dim === d && p.variable === undefined ? off : 0);
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
        if (this.contactsOn) {
          // store contact info: the flag, the force on the atom, the contact point on the wall, the radius
          const av = this.arrayAtom, base = 8 * i;
          av[base] = 1.0;
          av[base + 1] = fn * n[0] + ft0; av[base + 2] = fn * n[1] + ft1; av[base + 3] = fn * n[2] + ft2;
          av[base + 4] = x[3 * i]; av[base + 5] = x[3 * i + 1]; av[base + 6] = x[3 * i + 2];
          av[base + 4 + d] = x[3 * i + d] - sd;
          av[base + 7] = R;
        }
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
 * wall/gran; region walls for wall/gran/region with block, sphere (interior), cylinder and cone
 * regions, including compound (union/intersect) regions and dynamic (move/rotate) regions.  For a
 * compound region the sub-region faces are filtered as documented on region.html (see region.ts);
 * for a moving region the surface velocity enters the granular contact through the full relative
 * velocity (measured with native LAMMPS, see updateVelocities).  A side-out region (or a compound
 * region with a side-out member) contributes the nearest point of the solid (region.ts
 * surfaceContacts/outContacts), measured with native LAMMPS for block, sphere and cylinder. The contacts
 * keyword stores each atom's contact data (planes and regions), and temperature sets the wall temperature of the
 * heat model.
 */

const WALL_WORDS = ['xplane', 'yplane', 'zplane', 'zcylinder', 'region'] as const;

/** A wall/particle contact candidate: history key, distance to the surface, unit normal (wall to particle), effective radius. */
interface WallElement { key: string; dist: number; nx: number; ny: number; nz: number; Rf: number; source: Region | null }

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
  /** keyword temperature: a number, or the name of an equal-style variable (v_name); null when not given. */
  private wallT: number | string | null = null;
  private readonly out: ContactOut = newContactOut();
  /** Previous move/rotate state of every sub-region, for the surface velocity. */
  private readonly prevStates = new Map<Region, { d: [number, number, number]; theta: number }>();
  /** mdr normal model of this wall (its own coefficients) and whether it was checked against the pair style yet. */
  private mdrMat: MdrMaterialCore | null = null;
  private mdrChecked = false;
  /**
   * keyword contacts (fix_wall_gran_region.html): the per-atom array of the contact data, as native (fix_wall_gran_region.cpp
   * clear_stored_contacts and post_force): column 1 flag, 2-4 contact force on the atom, 5-7 contact point, 8 radius, then the
   * contact area of the mdr model (one column). Each atom keeps its last contact.
   */
  private contactsOn = false;

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
        roll: 'none', kr: 0, gr: 0, mr: 0, twist: 'none', kw: 0, gw: 0, mw: 0, limit, cutoff: -1, heat: 'none', hk: 0, sig: 'legacy',
      };
    } else {
      while (k < args.length && !(WALL_WORDS as readonly string[]).includes(args[k])) k++;
      if (k >= args.length) throw new StyleError(`fix ${id} ${this.style} granular: missing wallstyle`);
      const sp = parseGranularSpec(args.slice(1, k));
      pmOut = new PairGranular().wallParams(sp, regionMode);
      if (sp.normal === 'mdr') this.mdrMat = { E: sp.E, nu: sp.nu, Y: sp.Y, gamma: sp.gamma, psib: sp.psib, eta: sp.eta, dtype: sp.dtype };
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
        // fix_wall_gran.html and fix_wall_gran_region.html: "contacts" stores the contact data of each atom as a
        // per-atom array (see apply)
        this.contactsOn = true;
        k += 1;
        continue;
      } else if (w === 'temperature') {
        // fix_wall_gran.html: "The following value can either be a numeric value or an equal-style variable. If the value is a variable, it should be specified as v_name"
        const tw = args[k + 1];
        if (tw === undefined) throw new StyleError(`fix ${id} ${this.style} temperature needs a value`);
        this.wallT = tw.startsWith('v_') ? tw.slice(2) : parseNum(tw, 'wall temperature');
        k += 2;
      } else {
        // As native LAMMPS does (src/GRANULAR/fix_wall_gran.cpp FixWallGran::FixWallGran at b332f28, shared by fix_wall_gran_region.cpp): an unrecognised optional word stops with "Illegal fix wall/gran command".
        throw new StyleError('Illegal fix wall/gran command');
      }
    }
    if (this.contactsOn) {
      // the per-atom contact array (keyword contacts): flag, force, contact point, radius and, for mdr, the contact area
      this.peratomFlag = true;
      this.sizePeratomCols = this.mdrMat ? 9 : 8;
    }
    // heat conduction: the wall temperature and the heat model go together (measured with native LAMMPS: a heat model
    // without the temperature keyword, and the keyword with heat none, both stop with Must define wall temperature with heat model)
    if (this.pm.heat !== 'none' && this.wallT === null) throw new StyleError(`fix ${id} ${this.style}: heat ${this.pm.heat} needs the temperature keyword (wall temperature)`);
    if (this.pm.heat === 'none' && this.wallT !== null) throw new StyleError(`fix ${id} ${this.style}: temperature needs a heat model (heat radius or heat area)`);
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
    if (this.pm.heat !== 'none' && (!s.custom.has(HEAT_TEMPERATURE) || !s.custom.has(HEAT_FLOW))) {
      throw new StyleError(`fix ${this.id} ${this.style}: heat conduction needs fix property/atom temperature heatflow`);
    }
  }

  postForce(): void { this.apply(true); }

  setup(): void { this.apply(false); }

  /** The primitive sub-regions of a region (a compound recurses into members). */
  private sourcesOf(r: Region, out: Region[]): void {
    if (r.style === 'union' || r.style === 'intersect') {
      const c = r as unknown as { members: string[] };
      for (const id of c.members) this.sourcesOf(this.sys.region(id), out);
      return;
    }
    out.push(r);
  }

  /**
   * Velocity of a moving region at the contact point.  Measured with native
   * LAMMPS (black box, a block region moved by dy = v*time and a block rotated
   * by theta = omega*time, single sphere with classic hooke and xmu > 0): the
   * surface velocity enters the granular contact through the full relative
   * velocity — its normal component changes the normal damping and its
   * tangential component enters the tangential damping, Coulomb-capped by
   * xmu |F_n|.  The surface velocity is the finite difference of the region's
   * move/rotate state over one timestep (zero on the setup step):
   * ddot + omega x (q - (P + d)), with the move displacement applied to the
   * rotation point P (region.html).
   */
  private updateVelocities(dt: number, vel: Map<Region, { ddot: number[]; omega: number[]; d: number[] }>): void {
    const srcs: Region[] = [];
    this.sourcesOf(this.sys.region(this.regionId!), srcs);
    for (const src of srcs) {
      const cur = src.transformState();
      const prev = this.prevStates.get(src);
      const ddot = [0, 0, 0], omega = [0, 0, 0];
      if (prev) {
        for (let q = 0; q < 3; q++) ddot[q] = (cur.d[q] - prev.d[q]) / dt;
        if (src.rotate) {
          const dth = (cur.theta - prev.theta) / dt;
          omega[0] = dth * src.rotate.r[0]; omega[1] = dth * src.rotate.r[1]; omega[2] = dth * src.rotate.r[2];
        }
      }
      this.prevStates.set(src, cur);
      vel.set(src, { ddot, omega, d: cur.d });
    }
  }

  /**
   * The mdr model of the pair style. Native fix_granular_mdr checks that this wall's E, nu, Y, psi_b and damping agree with
   * the pair style (its surface energy and friction may differ); checked once.
   */
  private mdrApi(s: SimState): MdrApi {
    const pr = mdrPairOf(s);
    if (!pr) throw new StyleError(`fix ${this.id} ${this.style}: normal model 'mdr' needs pair_style granular with the mdr normal model`);
    if (!this.mdrChecked) {
      const a = pr.mat, b = this.mdrMat!;
      const eps = 1e-10;
      if (Math.abs(a.E - b.E) > eps) throw new StyleError(`fix ${this.id} ${this.style}: Young's modulus in pair style, ${a.E}, does not agree with value ${b.E} in fix gran/wall/region`);
      if (Math.abs(a.nu - b.nu) > eps) throw new StyleError(`fix ${this.id} ${this.style}: Poisson's ratio in pair style, ${a.nu}, does not agree with value ${b.nu} in fix gran/wall/region`);
      if (Math.abs(a.Y - b.Y) > eps) throw new StyleError(`fix ${this.id} ${this.style}: yield stress in pair style, ${a.Y}, does not agree with value ${b.Y} in fix gran/wall/region`);
      if (Math.abs(a.psib - b.psib) > eps) throw new StyleError(`fix ${this.id} ${this.style}: bulk response trigger in pair style, ${a.psib}, does not agree with value ${b.psib} in fix gran/wall/region`);
      if (Math.abs(a.eta - b.eta) > eps) throw new StyleError(`fix ${this.id} ${this.style}: damping in pair style, ${a.eta}, does not agree with value ${b.eta} in fix gran/wall/region`);
      this.mdrChecked = true;
    }
    return pr.api;
  }

  /** The wall contacts of atom i within its radius (the candidates of the force loop): the contacts of the bulk update. */
  private mdrContacts(i: number): Array<{ key: string; dist: number }> {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return [];
    const R = s.radius![i];
    const cand: WallElement[] = [];
    this.regionElements(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], R, cand);
    const out: Array<{ key: string; dist: number }> = [];
    for (const e of cand) if (e.dist < R) out.push({ key: `${this.id}|${e.key}:${s.id[i]}`, dist: e.dist });
    return out;
  }

  /** Contact candidates of one atom at its position (region faces; all sub-regions of a compound). */
  private regionElements(x: number, y: number, z: number, R: number, out: WallElement[]): void {
    out.length = 0;
    const r = this.sys.region(this.regionId!);
    const cs: SurfaceContact[] = [];
    r.surfaceContacts(x, y, z, cs);
    for (const c of cs) {
      const Rf = c.curvature === 0 ? R : (R * c.curvature) / (R + c.curvature);
      out.push({ key: `${c.source.id}:${c.key}`, dist: c.dist, nx: c.nx, ny: c.ny, nz: c.nz, Rf, source: c.source });
    }
  }

  private apply(update: boolean): void {
    const s = this.sys.state;
    const { x, v, f, id, mask, radius, rmass, omega, torque } = s;
    const cmass = contactMassArray(s);
    if (!radius || !rmass || !omega || !torque) throw new StyleError('fix wall/gran requires atom_style sphere');
    const dt = s.dt;
    const delta = (s.step - this.step0) * dt;
    const mv = [0, 0, 0];
    let off = 0;
    const m = this.motion;
    if (m?.kind === 'wiggle') {
      off = m.amp - m.amp * libm.cos(m.omega * delta);
      mv[m.dim] = m.amp * m.omega * libm.sin(m.omega * delta);
    } else if (m?.kind === 'shear') {
      mv[m.dim] = m.vshear;
    }
    const bit = this.groupBit;
    const pm = this.pm;
    const jkr = pm.normal === 'jkr';
    if (this.contactsOn) {
      // clear_stored_contacts: the per-atom contact data restart every step
      if (this.arrayAtom.length !== this.sizePeratomCols * s.n) this.arrayAtom = new Float64Array(this.sizePeratomCols * s.n);
      else this.arrayAtom.fill(0);
    }
    // mdr (region walls): the model of the pair style; the wall registers its contacts for the bulk update of the pair step
    const mdr = this.mdrMat ? this.mdrApi(s) : null;
    if (mdr) mdr.wallProvider(this.id, (i: number) => this.mdrContacts(i));
    const vel = new Map<Region, { ddot: number[]; omega: number[]; d: number[] }>();
    if (this.regionId) this.updateVelocities(dt, vel);
    // heat conduction (fix_wall_gran.html, temperature keyword): the wall temperature of this step, from a constant or an
    // equal-style variable evaluated now
    const heat = pm.heat !== 'none';
    const T = heat ? s.custom.get(HEAT_TEMPERATURE)!.data : null;
    const HF = heat ? s.custom.get(HEAT_FLOW)!.data : null;
    const Tw = typeof this.wallT === 'string' ? this.sys.equalVariable(this.wallT) : (this.wallT ?? 0);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const R = radius[i];
      const mi = cmass[i];
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
          cand.push({ key: `${k}`, dist, nx: n[0], ny: n[1], nz: n[2], Rf: R, source: null });
        }
      }
      for (const e of cand) {
        const key = `${e.key}:${id[i]}`;
        if (e.dist >= R && !jkr) {
          this.shear.delete(key);
          if (mdr) mdr.wallDrop(`${this.id}|${key}`);
          continue;
        }
        // region walls: the moving surface velocity enters the granular contact
        // through the full relative velocity (measured with native LAMMPS, see
        // updateVelocities)
        let vrx = v[3 * i] - mv[0], vry = v[3 * i + 1] - mv[1], vrz = v[3 * i + 2] - mv[2];
        if (e.source) {
          const st = vel.get(e.source);
          let wx = 0, wy = 0, wz = 0;
          if (st) {
            const qx = xi - e.dist * e.nx, qy = yi - e.dist * e.ny, qz = zi - e.dist * e.nz;
            const P = e.source.rotate ? e.source.rotate.p : [0, 0, 0];
            const rx = qx - (P[0] + st.d[0]), ry = qy - (P[1] + st.d[1]), rz = qz - (P[2] + st.d[2]);
            wx = st.ddot[0] + st.omega[1] * rz - st.omega[2] * ry;
            wy = st.ddot[1] + st.omega[2] * rx - st.omega[0] * rz;
            wz = st.ddot[2] + st.omega[0] * ry - st.omega[1] * rx;
          }
          vrx = v[3 * i] - wx;
          vry = v[3 * i + 1] - wy;
          vrz = v[3 * i + 2] - wz;
        }
        const sh = this.shear.get(key) ?? new Float64Array(8);
        const normal = mdr && this.mdrMat
          ? mdr.wallNormal({ key: `${this.id}|${key}`, ai: i, R, delta: R - e.dist, Rf: e.Rf, nx: e.nx, ny: e.ny, nz: e.nz, meff: mi, hu: update, p: this.mdrMat })
          : undefined;
        const ok = granularContact({
          pm, nx: e.nx, ny: e.ny, nz: e.nz, r: e.dist, delta: R - e.dist, Rf: e.Rf, ri: R, rj: 0, meff: mi,
          vrx, vry, vrz,
          oix: omega[3 * i], oiy: omega[3 * i + 1], oiz: omega[3 * i + 2],
          ojx: 0, ojy: 0, ojz: 0, dt, update, sg: 1, surfaceArm: true, normal,
          poly: this.legacyHertz ? Math.sqrt((R - e.dist) * e.Rf) : 1,
        }, sh, this.out, this.shear.has(key));
        if (!ok) {
          this.shear.delete(key);
          continue;
        }
        if (this.out.hist) this.shear.set(key, sh);
        if (this.contactsOn) {
          const base = i * this.sizePeratomCols;
          const av = this.arrayAtom;
          av[base] = 1;
          av[base + 1] = this.out.fx; av[base + 2] = this.out.fy; av[base + 3] = this.out.fz;
          av[base + 4] = xi - e.dist * e.nx; av[base + 5] = yi - e.dist * e.ny; av[base + 6] = zi - e.dist * e.nz;
          av[base + 7] = R;
          if (normal?.svector !== undefined) av[base + 8] = normal.svector;
        }
        f[3 * i] += this.out.fx; f[3 * i + 1] += this.out.fy; f[3 * i + 2] += this.out.fz;
        torque[3 * i] += this.out.tix; torque[3 * i + 1] += this.out.tiy; torque[3 * i + 2] += this.out.tiz;
        if (HF) HF[i] += heatFlowOf(pm.heat, pm.hk, this.out.a, Tw - T![i]);
      }
    }
  }
}
