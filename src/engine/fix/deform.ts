import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { int, latticeScale, num, yesno } from '../commands/args';

/*
 * fix ID group-ID deform N parameter style args ... keyword value ...
 * — docs.lammps.org/fix_deform.html. Quoted lines below are verbatim from
 * that page (fix_deform.rst in plans/lammps-docs, checked with grep -F).
 *
 *   |   fix ID group-ID deform N parameter style args ... keyword value ...
 *
 *   |Every Nth timestep during the run, the simulation box is
 *   |expanded, contracted, or tilted to ramped values between the initial
 *   |and final values.
 *
 * Length styles, with t the elapsed time since the start of the run:
 *
 *   |   L(t) = L0 (1 + erate\*dt)
 *
 *   |   L(t) = L0 exp(trate\*dt)
 *
 *   |   L(t) = L0 + A sin(2\*pi t/Tp)
 *
 *   |*volume* value = none = adjust this dim to preserve volume of system
 *
 *   |For the *scale*, *vel*, *erate*, *trate*, *volume*, *wiggle*, and
 *   |*variable* styles, the box length is expanded or compressed around its
 *   |mid point.
 *
 * "final" ramps both boundaries: "For style *final*, the final lo and hi box
 * boundaries of a dimension are specified."; "delta" is the same ramp with
 * targets lo0+dlo / hi0+dhi. "vel" is "a velocity at which the box length
 * changes is specified in units of distance/time", i.e. L(t) = L0 + V t.
 *
 * Tilt styles:
 *
 *   |   T(t) = T0 + L0\*erate\*dt
 *
 *   |   T(t) = T0 exp(trate\*dt)
 *
 *   |   T(t) = T0 + A sin(2\*pi t/Tp)
 *
 *   |where T0 is the initial tilt factor, L0 is the original length of the
 *   |box perpendicular to the shear direction (e.g. y box length for xy
 *
 *   |every picosecond.  Note that the initial tilt factor must be non-zero
 *   |to use the *trate* option.
 *
 * remap (applied at each deformation and, for v, whenever pbc wraps atoms):
 *
 *   |Each time the box size or shape is changed, the *remap* keyword
 *   |determines whether atom positions are remapped to the new box.  If
 *   |*remap* is set to *x* (the default), atoms in the fix group are
 *   |remapped; otherwise they are not.  Note that their velocities are not
 *   |changed, just their positions are altered.  If *remap* is set to *v*,
 *   |then any atom in the fix group that crosses a periodic boundary will
 *   |have a delta added to its velocity equal to the difference in
 *   |velocities between the lo and hi boundaries.  Note that this velocity
 *   |difference can include tilt components, e.g. a delta in the x velocity
 *   |when an atom crosses the y periodic boundary.  If *remap* is set to
 *   |*none*, then neither of these remappings take place.
 *
 *   |The *flip* keyword allows the tilt factors for a triclinic box to
 *   |exceed half the distance of the parallel box length, as discussed
 *   |above.  If the *flip* value is set to *yes*, the bound is enforced by
 *   |flipping the box when it is exceeded.  If the *flip* value is set to
 *   |*no*, the tilt will continue to change without flipping.
 *
 *   |The *units* keyword determines the meaning of the distance units used
 *   |to define various arguments.  A *box* value selects standard distance
 *   |units as defined by the :doc:`units <units>` command, e.g. Angstroms for
 *   |units = real or metal.  A *lattice* value means the distance units are
 *   |in lattice spacings.  The :doc:`lattice <lattice>` command must have
 *   |been previously used to define the lattice spacing.  Note that the
 *   |units choice also affects the *vel* style parameters since it is
 *   |defined in terms of distance/time.  Also note that the units keyword
 *   |does not affect the *variable* style.
 *
 *   |The option defaults are remap = x, flip = yes, and units = lattice.
 *
 * The deformation is applied in end_of_step every N steps ("Every Nth
 * timestep"); each length/tilt is a function of the elapsed time since the
 * start of the run (step - run.firstStep) * dt, and the final/delta/scale
 * ramp spans the run (lastStep - firstStep) * dt ("This fix can perform
 * deformation over multiple runs, using the *start* and *stop* keywords of
 * the :doc:`run <run>` command." — the engine measures from the run start).
 *
 * The streaming velocity of the deforming box (used by compute temp/deform
 * and by the remap v boundary deltas) follows from r = r_lo + H lamda with
 * the triclinic edge matrix H = [[Lx,xy,xz],[0,Ly,yz],[0,0,Lz]]
 * (Howto_triclinic.html): v = d(r_lo)/dt + (dH/dt) lamda, i.e. per component
 *   vx = loDot_x + Lxdot*lam_x + xydot*lam_y + xzdot*lam_z
 *   vy = loDot_y + Lydot*lam_y + yzdot*lam_z
 *   vz = loDot_z + Lzdot*lam_z
 * where loDot is the velocity of the lo boundary (-Ldot/2 for the styles that
 * deform "around its mid point", the lo ramp rate for final/delta).
 */

type DimStyle = 'final' | 'delta' | 'scale' | 'vel' | 'erate' | 'trate' | 'volume' | 'wiggle' | 'variable';

const LEN_STYLES: DimStyle[] = ['final', 'delta', 'scale', 'vel', 'erate', 'trate', 'volume', 'wiggle', 'variable'];
const TILT_STYLES: DimStyle[] = ['final', 'delta', 'vel', 'erate', 'trate', 'wiggle', 'variable'];
/** fix deform/pressure-only styles (docs.lammps.org/fix_deform_pressure.html). */
const PRESSURE_STYLES = ['pressure', 'pressure/mean'];
/** fix deform/pressure-only keywords. */
const PRESSURE_KEYWORDS = ['couple', 'vol/balance/p', 'max/rate', 'normalize/pressure'];

interface DimSpec {
  /** 'x' | 'y' | 'z' | 'xy' | 'xz' | 'yz'. */
  name: string;
  /** Length dimension 0/1/2 for x/y/z; for tilts the tilt index 0/1/2 (xy, xz, yz). */
  dim: number;
  tilt: boolean;
  style: DimStyle;
  lo?: number; hi?: number;   // final targets (box units)
  dlo?: number; dhi?: number; // delta amounts (box units)
  factor?: number;            // scale
  v?: number;                 // vel (box units / time)
  rate?: number;              // erate / trate (1/time)
  amp?: number; period?: number; // wiggle (amp in box units)
  var1?: string; var2?: string;  // variable style names
}

export class FixDeform extends Fix {
  readonly style = 'deform';
  private readonly specs: DimSpec[] = [];
  private remapMode: 'x' | 'v' | 'none' = 'x';
  private flip = true;
  private readonly unitsLattice: boolean;

  // box at the start of the run (paraphrased, not quoted: the box set by
  // create_box, read_data or read_restart, or the values from the end of the
  // previous run; paraphrase of the fix_deform.rst description of run start)
  private lo0: number[] = [0, 0, 0];
  private hi0: number[] = [0, 0, 0];
  private len0: number[] = [0, 0, 0];
  private mid0: number[] = [0, 0, 0];
  private tilt0: number[] = [0, 0, 0];
  /** Length perpendicular to the shear: y for xy, z for xz and yz. */
  private perp0: number[] = [0, 0, 0];
  private vol0 = 0;
  private captured = false;

  /** Current box deformation rates: d(length)/dt, d(tilt)/dt, lo-boundary velocity. */
  private lenDot: number[] = [0, 0, 0];
  private tiltDot: number[] = [0, 0, 0];
  private loDot: number[] = [0, 0, 0];
  /** remap v state: last remap step, the positions / box at it, and the
   *  repair bookkeeping for the engine's earlier no-kick pbc wraps. */
  private remapLastStep = -1;
  private remapHold = new Float64Array(0);
  private remapBoxHold: number[] = [];
  private imgPre: Int32Array | null = null;
  private accShift = new Float64Array(0);
  private accImg = new Int32Array(0);
  private lam: number[] = [0, 0, 0];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const n = int(args[0], 'fix deform N');
    if (n <= 0) throw new StyleError(`fix deform: N must be a positive step interval (got '${args[0]}')`);
    this.nevery = n;
    let units: string | null = null;
    // first pass: keywords (they may follow any parameter sequence)
    for (let k = 1; k < args.length;) {
      const w = args[k];
      if (w === 'remap' || w === 'flip' || w === 'units' || PRESSURE_KEYWORDS.includes(w)) {
        const val = args[k + 1];
        if (val === undefined) throw new StyleError(`fix deform: keyword '${w}' is missing its value`);
        if (w === 'remap') {
          if (val !== 'x' && val !== 'v' && val !== 'none') throw new StyleError(`fix deform: remap must be x, v or none (got '${val}')`);
          this.remapMode = val;
        } else if (w === 'flip') {
          this.flip = yesno(val, 'fix deform flip');
        } else if (w === 'units') {
          if (val !== 'lattice' && val !== 'box') throw new StyleError(`fix deform: units must be lattice or box (got '${val}')`);
          units = val;
        } else {
          throw new StyleError(`fix deform: keyword '${w}' belongs to fix deform/pressure, which is not supported by the browser engine`);
        }
        k += 2;
        continue;
      }
      k += this.parseParam(args, k);
    }
    if (!this.specs.length) throw new StyleError('fix deform: at least one parameter (x, y, z, xy, xz, yz) is required');
    if (sys.dimension === 2) {
      for (const sp of this.specs) {
        if (sp.name === 'z' || sp.name === 'xz' || sp.name === 'yz') {
          throw new StyleError(`fix deform: parameter ${sp.name} is not available for a 2d simulation`);
        }
      }
    }
    // units conversion happens after the full parse: the keyword may follow
    // the parameter sequences it scales. region.html (style prism): "The lattice spacing
    // in dimension x is applied to xy and xz, and the spacing in dimension y
    // to yz."
    this.unitsLattice = (units ?? 'lattice') === 'lattice';
    if (this.unitsLattice) {
      const sp = latticeScale(sys, 'lattice', 'fix deform');
      const scaleOf = (name: string): number => (name === 'x' || name === 'xy' || name === 'xz' ? sp[0] : name === 'y' || name === 'yz' ? sp[1] : sp[2]);
      for (const s2 of this.specs) {
        const f = scaleOf(s2.name);
        if (s2.lo !== undefined) s2.lo *= f;
        if (s2.hi !== undefined) s2.hi *= f;
        if (s2.dlo !== undefined) s2.dlo *= f;
        if (s2.dhi !== undefined) s2.dhi *= f;
        if (s2.v !== undefined) s2.v *= f;
        if (s2.amp !== undefined) s2.amp *= f;
      }
    }
  }

  /** Parses one parameter/args sequence at args[k]; returns the words consumed. */
  private parseParam(a: string[], k: number): number {
    const name = a[k];
    const tilt = name.length === 2;
    const style = a[k + 1];
    if (!['x', 'y', 'z', 'xy', 'xz', 'yz'].includes(name)) {
      throw new StyleError(`fix deform: unknown parameter '${name}' (expected x, y, z, xy, xz or yz)`);
    }
    if (this.specs.some((s) => s.name === name)) throw new StyleError(`fix deform: parameter ${name} given twice`);
    if (PRESSURE_STYLES.includes(style)) {
      throw new StyleError(`fix deform: style '${style}' belongs to fix deform/pressure, which is not supported by the browser engine`);
    }
    const allowed = tilt ? TILT_STYLES : LEN_STYLES;
    if (!allowed.includes(style as DimStyle)) {
      throw new StyleError(`fix deform: style '${style}' is not valid for parameter ${name} (allowed: ${allowed.join(' ')})`);
    }
    const spec: DimSpec = { name, dim: tilt ? ['xy', 'xz', 'yz'].indexOf(name) : ['x', 'y', 'z'].indexOf(name), tilt, style: style as DimStyle };
    const numAt = (j: number, what: string): number => num(a[j], `fix deform ${name} ${style} ${what}`);
    let used = 2;
    switch (style) {
      case 'final':
        if (tilt) {
          spec.lo = numAt(k + 2, 'tilt');
          used = 3;
        } else {
          spec.lo = numAt(k + 2, 'lo');
          spec.hi = numAt(k + 3, 'hi');
          used = 4;
        }
        break;
      case 'delta':
        if (tilt) {
          spec.dlo = numAt(k + 2, 'dtilt');
          used = 3;
        } else {
          spec.dlo = numAt(k + 2, 'dlo');
          spec.dhi = numAt(k + 3, 'dhi');
          used = 4;
        }
        break;
      case 'scale':
        if (tilt) throw new StyleError(`fix deform: style 'scale' is not valid for tilt parameter ${name}`);
        spec.factor = numAt(k + 2, 'factor');
        used = 3;
        break;
      case 'vel':
        spec.v = numAt(k + 2, 'V');
        used = 3;
        break;
      case 'erate':
      case 'trate':
        spec.rate = numAt(k + 2, 'R');
        used = 3;
        break;
      case 'volume':
        if (tilt) throw new StyleError(`fix deform: style 'volume' is not valid for tilt parameter ${name}`);
        used = 2; // "*volume* value = none": no value follows
        break;
      case 'wiggle':
        spec.amp = numAt(k + 2, 'A');
        spec.period = numAt(k + 3, 'Tp');
        if (spec.period <= 0) throw new StyleError(`fix deform: wiggle period must be > 0 (got '${a[k + 3]}')`);
        used = 4;
        break;
      case 'variable': {
        const v1 = a[k + 2], v2 = a[k + 3];
        if (v1 === undefined || v2 === undefined) throw new StyleError(`fix deform ${name} variable needs two variables v_name1 v_name2`);
        if (!v1.startsWith('v_') || !v2.startsWith('v_')) throw new StyleError(`fix deform ${name} variable needs two variables v_name1 v_name2 (got '${v1} ${v2}')`);
        spec.var1 = v1.slice(2);
        spec.var2 = v2.slice(2);
        used = 4;
        break;
      }
    }
    this.specs.push(spec);
    return used;
  }

  init(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    for (let d = 0; d < 3; d++) {
      this.lo0[d] = s.box.lo[d];
      this.hi0[d] = s.box.hi[d];
      this.len0[d] = d === 0 ? g.lx : d === 1 ? g.ly : g.lz;
      this.mid0[d] = 0.5 * (s.box.lo[d] + s.box.hi[d]);
    }
    this.tilt0 = [g.xy, g.xz, g.yz];
    // "L0 is the original length of the box perpendicular to the shear
    // direction (e.g. y box length for xy deformation)"
    this.perp0 = [g.ly, g.lz, g.lz];
    this.vol0 = g.volume(s.dimension);
    for (const sp of this.specs) {
      if (sp.style === 'trate' && sp.tilt && this.tilt0[sp.dim] === 0) {
        throw new StyleError(`fix deform: the initial tilt factor must be non-zero to use the trate option (${sp.name} is 0)`);
      }
    }
    this.captured = true;
    this.deform(false); // rates at elapsed = 0, for compute temp/deform before the first end_of_step
  }

  setup(): void {
    // LAMMPS calls end_of_step() from the fix's setup() as well
    this.endOfStep();
    // remap v: the first remap window starts at the run setup (the positions
    // and box the setup pbc and this deformation left behind)
    const s = this.sys.state;
    const g = this.sys.geom;
    this.remapLastStep = s.step;
    this.remapHold = Float64Array.from(s.x.subarray(0, 3 * s.n));
    this.remapBoxHold = [g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], g.xy, g.xz, g.yz];
    this.ensureRemapArrays(s.n);
  }

  endOfStep(): void {
    const s = this.sys.state;
    if (s.step % this.nevery !== 0) return;
    this.deform(true);
  }

  /**
   * Computes the box parameters and rates for the current step's elapsed time
   * and, when `apply`, installs them through sys.setBox (remap x keeps the
   * fractional coordinates of the fix group's atoms, v/none keep their
   * Cartesian ones).
   */
  private deform(apply: boolean): void {
    if (!this.captured) return;
    const sys = this.sys;
    const s = sys.state;
    const dt = s.dt;
    const elapsed = (s.step - sys.run.firstStep) * dt;
    const span = (sys.run.lastStep - sys.run.firstStep) * dt;
    const frac = span > 0 ? elapsed / span : 0;
    const dim = s.dimension;
    const ndims = dim === 2 ? 2 : 3;

    const lo = [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
    const hi = [s.box.hi[0], s.box.hi[1], s.box.hi[2]];
    const tilt = [s.box.tilt[0], s.box.tilt[1], s.box.tilt[2]];
    const len = [this.len0[0], this.len0[1], this.len0[2]];
    const lenD = [0, 0, 0];
    const loD = [0, 0, 0];

    // non-volume length styles first (volume dims need their new lengths)
    for (const sp of this.specs) {
      if (sp.tilt || sp.style === 'volume') continue;
      const d = sp.dim, L0 = this.len0[d];
      let L = L0, LD = 0, loRate = 0;
      switch (sp.style) {
        case 'final':
        case 'delta': {
          // final: "the final lo and hi box boundaries"; delta: lo0+dlo, hi0+dhi
          const loT = sp.style === 'final' ? sp.lo! : this.lo0[d] + sp.dlo!;
          const hiT = sp.style === 'final' ? sp.hi! : this.hi0[d] + sp.dhi!;
          lo[d] = this.lo0[d] + (loT - this.lo0[d]) * frac;
          hi[d] = this.hi0[d] + (hiT - this.hi0[d]) * frac;
          loRate = span > 0 ? (loT - this.lo0[d]) / span : 0;
          LD = span > 0 ? (hiT - this.hi0[d] - (loT - this.lo0[d])) / span : 0;
          break;
        }
        case 'scale': {
          L = L0 + (sp.factor! * L0 - L0) * frac;
          LD = span > 0 ? (sp.factor! * L0 - L0) / span : 0;
          break;
        }
        case 'vel':
          L = L0 + sp.v! * elapsed;
          LD = sp.v!;
          break;
        case 'erate':
          L = L0 * (1 + sp.rate! * elapsed);
          LD = L0 * sp.rate!;
          break;
        case 'trate':
          L = L0 * Math.exp(sp.rate! * elapsed);
          LD = L * sp.rate!;
          break;
        case 'wiggle': {
          const w = (2 * Math.PI * elapsed) / sp.period!;
          L = L0 + sp.amp! * Math.sin(w);
          LD = sp.amp! * ((2 * Math.PI) / sp.period!) * Math.cos(w);
          break;
        }
        case 'variable':
          L = L0 + sys.equalVariable(sp.var1!);
          LD = sys.equalVariable(sp.var2!);
          break;
      }
      len[d] = L;
      lenD[d] = LD;
      if (sp.style !== 'final' && sp.style !== 'delta') {
        // "the box length is expanded or compressed around its mid point"
        lo[d] = this.mid0[d] - 0.5 * L;
        hi[d] = this.mid0[d] + 0.5 * L;
        loRate = -LD / 2;
      }
      loD[d] = loRate;
    }

    // volume styles: V0 is the box volume at the start of the run
    const volSpecs = this.specs.filter((sp) => !sp.tilt && sp.style === 'volume');
    if (volSpecs.length) {
      const k = volSpecs.length;
      let prod = 1;
      let sum = 0;
      for (let d = 0; d < ndims; d++) {
        if (volSpecs.some((sp) => sp.dim === d)) continue;
        prod *= len[d];
        sum += lenD[d] / len[d];
      }
      if (!(prod > 0)) throw new StyleError('fix deform: volume style needs a positive product of the other box lengths');
      const L = Math.pow(this.vol0 / prod, 1 / k);
      for (const sp of volSpecs) {
        len[sp.dim] = L;
        lenD[sp.dim] = (-1 / k) * L * sum;
        loD[sp.dim] = -lenD[sp.dim] / 2;
        lo[sp.dim] = this.mid0[sp.dim] - 0.5 * L;
        hi[sp.dim] = this.mid0[sp.dim] + 0.5 * L;
      }
    }

    for (let d = 0; d < ndims; d++) {
      if (!(len[d] > 0)) throw new StyleError(`fix deform: box length in ${'xyz'[d]} became ${len[d]}`);
    }

    // tilt styles
    const tiltD = [0, 0, 0];
    for (const sp of this.specs) {
      if (!sp.tilt) continue;
      const t = sp.dim, T0 = this.tilt0[t];
      let T = T0, TD = 0;
      switch (sp.style) {
        case 'final':
          T = T0 + (sp.lo! - T0) * frac;
          TD = span > 0 ? (sp.lo! - T0) / span : 0;
          break;
        case 'delta':
          T = T0 + sp.dlo! * frac;
          TD = span > 0 ? sp.dlo! / span : 0;
          break;
        case 'vel':
          T = T0 + sp.v! * elapsed;
          TD = sp.v!;
          break;
        case 'erate':
          T = T0 + this.perp0[t] * sp.rate! * elapsed;
          TD = this.perp0[t] * sp.rate!;
          break;
        case 'trate':
          T = T0 * Math.exp(sp.rate! * elapsed);
          TD = T0 * sp.rate! * Math.exp(sp.rate! * elapsed);
          break;
        case 'wiggle': {
          const w = (2 * Math.PI * elapsed) / sp.period!;
          T = T0 + sp.amp! * Math.sin(w);
          TD = sp.amp! * ((2 * Math.PI) / sp.period!) * Math.cos(w);
          break;
        }
        case 'variable':
          T = T0 + sys.equalVariable(sp.var1!);
          TD = sys.equalVariable(sp.var2!);
          break;
      }
      tilt[t] = T;
      tiltD[t] = TD;
      // flip = yes enforces |tilt| <= half the parallel box length by flipping
      // the box; the engine does not flip and reports the situation instead
      if (this.flip) {
        const p = t === 2 ? 1 : 0; // parallel dim: x for xy/xz, y for yz
        // "The one exception to this rule is if the first dimension in the
        // tilt factor (x for xy) is non-periodic.  In that case, the limits
        // on the tilt factor are not enforced"
        if (s.box.periodic[p]) {
          const bound = 0.5 * (p === 0 ? len[0] : len[1]);
          if (Math.abs(T) > bound) {
            throw new StyleError(`fix deform: the ${sp.name} tilt factor reached ${T}, beyond half the parallel box length (${bound}); box flips (flip yes) are not supported by the browser engine (use flip no to tilt past the bound without flipping)`);
          }
        }
      }
    }

    if (!apply) {
      this.lenDot = lenD;
      this.tiltDot = tiltD;
      this.loDot = loD;
      return;
    }
    this.lenDot = lenD;
    this.tiltDot = tiltD;
    this.loDot = loD;
    sys.setBox(lo, hi, tilt, this.remapMode === 'x' ? this.groupBit : 0, this);
    // thermo output and dumps after end_of_step keep reporting this step's
    // force evaluation (energies, virial and per-atom forces), as LAMMPS does
    // when fix deform changes the box after final_integrate
    sys.forcesCurrent();
  }

  /** Streaming velocity the deforming box imposes at atom i's position. */
  vstream(i: number, out: number[]): void {
    const s = this.sys.state;
    this.sys.geom.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], this.lam);
    const l0 = this.lam[0], l1 = this.lam[1], l2 = this.lam[2];
    out[0] = this.loDot[0] + this.lenDot[0] * l0 + this.tiltDot[0] * l1 + this.tiltDot[1] * l2;
    out[1] = this.loDot[1] + this.lenDot[1] * l1 + this.tiltDot[2] * l2;
    out[2] = this.loDot[2] + this.lenDot[2] * l2;
  }

  /**
   * remap v: "any atom in the fix group that crosses a periodic boundary will
   * have a delta added to its velocity equal to the difference in velocities
   * between the lo and hi boundaries". Native LAMMPS applies this from its
   * pre_exchange hook, i.e. only on steps where the neighbor lists are
   * rebuilt: an atom outside the box is wrapped by domain->pbc() right after
   * and its velocity picks up the boundary delta. Measured on native LAMMPS
   * (1-atom boxes, slow drifts and shears): the rebuild fires when the largest
   * atom displacement since the last build exceeds skin/2 - boxDelta, where
   * boxDelta is the largest box-parameter change since that build (a purely
   * deforming box with stationary atoms never rebuilds). An image change of k
   * in dimension d shifts v by -k * (v_hi - v_lo), with (v_hi - v_lo) =
   * (Lxdot) for x, (xydot, Lydot) for y and (xzdot, yzdot, Lzdot) for z.
   *
   * The engine rebuilds its neighbor lists by the same rule but penalises the
   * box change with 2*boxDelta (neighbor.ts), so its rebuilds — and with them
   * sys.pbc()'s coordinate wraps — can happen a step or two earlier than
   * native's. A wrap without the simultaneous kick changes the atom's
   * unwrapped position by the tilt grown since the wrap, so this fix times
   * the wraps+kicks itself: postIntegrate replicates native's rebuild
   * criterion and, on a remap step, wraps the group's atoms that are outside
   * the box and adds the velocity deltas. The engine's earlier no-kick wraps
   * of group atoms are undone first (preExchange/preNeighbor record them with
   * the box they used), which restores exactly native's state.
   */
  preExchange(): void {
    if (this.remapMode !== 'v') return;
    const s = this.sys.state;
    this.imgPre = Int32Array.from(s.image.subarray(0, 3 * s.n));
  }

  /** Records the pbc wraps of group atoms (engine rebuild steps only). */
  preNeighbor(): void {
    if (this.remapMode !== 'v' || !this.imgPre) return;
    const s = this.sys.state;
    const g = this.sys.geom;
    this.ensureRemapArrays(s.n);
    for (let i = 0; i < s.n; i++) {
      const kx = s.image[3 * i] - this.imgPre[3 * i];
      const ky = s.image[3 * i + 1] - this.imgPre[3 * i + 1];
      const kz = s.image[3 * i + 2] - this.imgPre[3 * i + 2];
      if (kx === 0 && ky === 0 && kz === 0) continue;
      this.accShift[3 * i] += kx * g.lx + ky * g.xy + kz * g.xz;
      this.accShift[3 * i + 1] += ky * g.ly + kz * g.yz;
      this.accShift[3 * i + 2] += kz * g.lz;
      this.accImg[3 * i] += kx;
      this.accImg[3 * i + 1] += ky;
      this.accImg[3 * i + 2] += kz;
    }
  }

  postIntegrate(): void {
    if (this.remapMode !== 'v' || !this.sys.hasBox) return;
    const s = this.sys.state;
    const g = this.sys.geom;
    const nb = this.sys.nb;
    const step = s.step;
    const ago = step - this.remapLastStep;
    if (ago < nb.delay) return;
    if (nb.every > 1 && step % nb.every !== 0) return;
    if (nb.check && this.remapHold.length === 3 * s.n) {
      const cur = [g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], g.xy, g.xz, g.yz];
      let boxDelta = 0;
      for (let k = 0; k < 9; k++) boxDelta = Math.max(boxDelta, Math.abs(cur[k] - this.remapBoxHold[k]));
      const lim = 0.5 * nb.skin - boxDelta;
      const lim2 = lim * lim;
      let moved = false;
      const x = s.x, hold = this.remapHold;
      for (let k = 0; k < 3 * s.n; k += 3) {
        const dx = x[k] - hold[k], dy = x[k + 1] - hold[k + 1], dz = x[k + 2] - hold[k + 2];
        if (dx * dx + dy * dy + dz * dz > lim2) { moved = true; break; }
      }
      if (!moved) return;
    }
    // remap step: undo the engine's earlier no-kick wraps of the group, then
    // wrap whoever is outside the box now and add the boundary velocity deltas
    this.ensureRemapArrays(s.n);
    const v = s.v;
    const dlx = this.lenDot[0], dly = this.lenDot[1], dlz = this.lenDot[2];
    const dxy = this.tiltDot[0], dxz = this.tiltDot[1], dyz = this.tiltDot[2];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (this.accImg[3 * i] !== 0 || this.accImg[3 * i + 1] !== 0 || this.accImg[3 * i + 2] !== 0) {
        s.x[3 * i] += this.accShift[3 * i];
        s.x[3 * i + 1] += this.accShift[3 * i + 1];
        s.x[3 * i + 2] += this.accShift[3 * i + 2];
        s.image[3 * i] -= this.accImg[3 * i];
        s.image[3 * i + 1] -= this.accImg[3 * i + 1];
        s.image[3 * i + 2] -= this.accImg[3 * i + 2];
        this.accShift[3 * i] = 0; this.accShift[3 * i + 1] = 0; this.accShift[3 * i + 2] = 0;
        this.accImg[3 * i] = 0; this.accImg[3 * i + 1] = 0; this.accImg[3 * i + 2] = 0;
      }
      const ix = s.image[3 * i], iy = s.image[3 * i + 1], iz = s.image[3 * i + 2];
      g.remap(s.x, s.image, i);
      const kx = s.image[3 * i] - ix, ky = s.image[3 * i + 1] - iy, kz = s.image[3 * i + 2] - iz;
      if (kx === 0 && ky === 0 && kz === 0) continue;
      v[3 * i] -= kx * dlx + ky * dxy + kz * dxz;
      v[3 * i + 1] -= ky * dly + kz * dyz;
      v[3 * i + 2] -= kz * dlz;
    }
    this.remapLastStep = step;
    this.remapHold = Float64Array.from(s.x.subarray(0, 3 * s.n));
    this.remapBoxHold = [g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], g.xy, g.xz, g.yz];
  }

  /** (Re)allocates the per-atom remap bookkeeping. */
  private ensureRemapArrays(n: number): void {
    if (this.accImg && this.accImg.length === 3 * n && this.accShift.length === 3 * n) return;
    this.accImg = new Int32Array(3 * n);
    this.accShift = new Float64Array(3 * n);
  }
}
