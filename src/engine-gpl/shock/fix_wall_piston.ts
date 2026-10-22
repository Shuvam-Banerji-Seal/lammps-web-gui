/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SHOCK/fix_wall_piston.cpp and fix_wall_piston.h at commit b332f28 (Laurence Fried and Evan Reed, for
 * the piston wall of the shock examples; the LAMMPS developers). Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix wall/piston zlo keyword values: a moving wall at the lower z face of a shrink-wrapped box. Atoms below the
 * wall are reflected through it (position and velocity), and the optional temp keyword applies a Langevin-type
 * thermostat to the atoms near the wall. Only zlo is supported (as in native LAMMPS: the other faces are refused).
 */
import type { System } from '../../engine/system';
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { missingArgs, parseNum } from '../../engine/force/util';
import { nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

const TWO_PI = 2 * Math.PI;

/** The motion ramp of the wall (fix_wall_piston.cpp: ramp, rampNL1 .. rampNL5 and the constant velocity). */
type Ramp = 'none' | 'ramp' | 'NL1' | 'NL2' | 'NL3' | 'NL4' | 'NL5';

export class FixWallPiston extends Fix {
  readonly style = 'wall/piston';
  private zloflag = false;
  private z0 = 0;
  /** Wall velocity (the native member vz; a ramp rewrites it each step). */
  private vz = 0;
  private vzInput = 0;
  private scaleflag = true;
  private tempflag = false;
  private tTarget = 0;
  private tPeriod = 0;
  private tseed = 0;
  private tExtent = 0;
  private rng: RanMars | null = null;
  private roughflag = false;
  private roughdist = 0;
  private ramp: Ramp = 'none';
  private maxvz = 0;
  private paccelz = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw missingArgs('fix wall/piston');
    let k = 0;
    // a keyword's values must follow it (native: iarg + n <= narg)
    const need = (n: number) => { if (k + n >= args.length) throw new StyleError('Illegal fix wall/piston command'); };
    while (k < args.length) {
      const w = args[k];
      if (w === 'xlo' || w === 'ylo' || w === 'xhi' || w === 'yhi' || w === 'zhi') {
        throw new StyleError('Fix wall/piston command only available at zlo');
      } else if (w === 'zlo') {
        this.zloflag = true;
        k++;
        if (sys.state.box.boundary[2][0] !== 's') throw new StyleError('Must shrink-wrap piston boundary');
      } else if (w === 'vel') {
        need(1);
        this.vzInput = parseNum(args[k + 1], 'vel');
        k += 2;
      } else if (w === 'pos') {
        need(1);
        this.z0 = parseNum(args[k + 1], 'pos');
        k += 2;
      } else if (w === 'temp') {
        need(4);
        this.tempflag = true;
        this.tTarget = parseNum(args[k + 1], 'temp');
        this.tPeriod = parseNum(args[k + 2], 'temp period');
        this.tseed = parseNum(args[k + 3], 'temp seed');
        this.tExtent = parseNum(args[k + 4], 'temp extent');
        if (!(this.tTarget > 0) || !(this.tPeriod > 0) || !(this.tExtent > 0) || !(this.tseed > 0)) throw new StyleError('Illegal fix wall/piston command');
        this.rng = new RanMars(this.tseed);
        k += 5;
      } else if (w === 'rough') {
        need(1);
        this.roughflag = true;
        this.roughdist = parseNum(args[k + 1], 'rough');
        k += 2;
      } else if (w === 'ramp' || w === 'rampNL1' || w === 'rampNL2' || w === 'rampNL3' || w === 'rampNL4' || w === 'rampNL5') {
        this.ramp = w === 'ramp' ? 'ramp' : (w.slice(4) as Ramp);
        k++;
      } else if (w === 'units') {
        need(1);
        const u = args[k + 1];
        if (u === 'box') this.scaleflag = false;
        else if (u === 'lattice') this.scaleflag = true;
        else throw new StyleError('Illegal fix wall/piston command');
        k += 2;
      } else {
        throw new StyleError('Illegal fix wall/piston command');
      }
    }
    if (this.vzInput < 0) throw new StyleError('Illegal fix wall/piston velocity');
    if (this.zloflag && sys.state.box.boundary[2][0] === 'p') throw new StyleError('Cannot use wall in periodic dimension');
    // native accepts a piston without zlo (the fix then moves no wall); the engine needs it
    if (!this.zloflag) throw new StyleError('fix wall/piston needs the zlo face');
    // units lattice (the default): the velocity, position and rough distance are in lattice spacings
    const zscale = this.scaleflag && sys.lattice ? sys.lattice.spacing[2] : 1;
    this.vz = this.vzInput * zscale;
    this.z0 *= zscale;
    this.roughdist *= zscale;
    if (this.ramp !== 'none') this.maxvz = this.vz;
  }

  /** setmask INITIAL_INTEGRATE: the piston forces a neighbor list build each step (force_reneighbor = 1). */
  reneighborDue(): boolean {
    return true;
  }

  /**
   * post_integrate: moves the wall (zlo, vz from the ramp or the constant velocity), reflects the atoms below it and
   * applies the piston thermostat (temp) to the atoms within temp extent of the lower face.
   */
  postIntegrate(): void {
    const sys = this.sys;
    const s = sys.state;
    const run = sys.run;
    let zlo: number;
    const t = (s.step - run.beginStep) * s.dt;
    const tott = (run.endStep - run.beginStep) * s.dt;
    const tt = t * t;
    const ttt = tt * t;
    const tttt = tt * tt;
    const t0p5 = Math.sqrt(t / tott);
    const t1p5 = t0p5 * t0p5 * t0p5;
    const t2p5 = t1p5 * t0p5 * t0p5;
    const maxvz = this.maxvz;
    if (this.ramp === 'ramp') {
      this.paccelz = maxvz / tott;
      zlo = this.z0 + 0.5 * this.paccelz * tt;
      this.vz = this.paccelz * t;
    } else if (this.ramp === 'NL1') {
      this.paccelz = maxvz / tott;
      const angfreq = TWO_PI / (0.5 * tott);
      zlo = this.z0 + this.paccelz * (0.5 * tt + 1 / (angfreq * angfreq) - (1 / (angfreq * angfreq)) * libmCos(angfreq * t));
      this.vz = this.paccelz * (t + (1 / angfreq) * libmSin(angfreq * t));
    } else if (this.ramp === 'NL2') {
      this.paccelz = maxvz / tott;
      const angfreq = (3 * TWO_PI) / tott;
      zlo = this.z0 + this.paccelz * (0.5 * tt + (4 / (3 * angfreq * angfreq)) * (1 - libmCos(angfreq * t)) +
        (1 / (6 * angfreq * angfreq)) * (1 - libmCos(2 * angfreq * t)));
      this.vz = this.paccelz * (t + (4 / (3 * angfreq)) * libmSin(angfreq * t) + (1 / (3 * angfreq)) * libmSin(2 * angfreq * t));
    } else if (this.ramp === 'NL3') {
      this.paccelz = maxvz / tott;
      zlo = this.z0 + ((this.paccelz * tott * tott) / 2.5) * t2p5;
      this.vz = this.paccelz * tott * t1p5;
    } else if (this.ramp === 'NL4') {
      this.paccelz = maxvz / tott;
      zlo = this.z0 + (this.paccelz / tott / 3) * ttt;
      this.vz = (this.paccelz / tott) * tt;
    } else if (this.ramp === 'NL5') {
      this.paccelz = maxvz / tott;
      zlo = this.z0 + (this.paccelz / tott / tott / 4) * tttt;
      this.vz = (this.paccelz / tott / tott) * ttt;
    } else {
      zlo = this.z0 + this.vz * t;
    }
    const vz = this.vz;
    const x = s.x, v = s.v, f = s.f;
    const order = nativeOrder(s);
    const boxlo = s.box.lo, boxhi = s.box.hi;
    // reflection of the atoms below the wall (native loop over the atoms in local order)
    for (let n = 0; n < s.n; n++) {
      const i = order[n];
      if (!(s.mask[i] & this.groupBit)) continue;
      let roughoff = 0;
      if (this.roughflag) {
        roughoff += this.roughdist * Math.abs((x[3 * i] - boxlo[0]) / (boxhi[0] - boxlo[0]) - 0.5);
        roughoff += this.roughdist * Math.abs((x[3 * i + 1] - boxlo[1]) / (boxhi[1] - boxlo[1]) - 0.5);
      }
      if (x[3 * i + 2] < zlo - roughoff) {
        x[3 * i + 2] = 2 * (zlo - roughoff) - x[3 * i + 2];
        v[3 * i + 2] = 2 * vz - v[3 * i + 2];
      }
    }
    if (!this.tempflag) return;
    const rng = this.rng!;
    const ftm2v = s.units.ftm2v, mvv2e = s.units.mvv2e, boltz = s.units.boltz;
    const tsqrt = Math.sqrt(this.tTarget);
    const dt = s.dt;
    const tp = this.tPeriod;
    const extent = boxlo[2] + this.tExtent;
    if (s.rmass === null) {
      // native: atom->mass; the per-type factors gfactor1 and gfactor2
      const nt = s.massByType.length;
      const g1 = new Float64Array(nt), g2 = new Float64Array(nt);
      for (let ty = 1; ty < nt; ty++) {
        const m = s.massByType[ty];
        g1[ty] = -m / tp / ftm2v;
        g2[ty] = Math.sqrt(m) * Math.sqrt((24.0 * boltz) / tp / dt / mvv2e) / ftm2v;
      }
      for (let n = 0; n < s.n; n++) {
        const i = order[n];
        if (!(x[3 * i + 2] <= extent)) continue;
        const gamma1 = g1[s.type[i]];
        const gamma2 = g2[s.type[i]] * tsqrt;
        f[3 * i] += gamma1 * v[3 * i] + gamma2 * (rng.uniform() - 0.5);
        f[3 * i + 1] += gamma1 * v[3 * i + 1] + gamma2 * (rng.uniform() - 0.5);
        f[3 * i + 2] += gamma1 * (v[3 * i + 2] - vz) + gamma2 * (rng.uniform() - 0.5);
      }
    } else {
      // native: the rmass branch (per-atom masses; the z component has no wall velocity)
      const rmass = s.rmass;
      for (let n = 0; n < s.n; n++) {
        const i = order[n];
        if (!(x[3 * i + 2] <= extent)) continue;
        const gamma1 = -rmass[i] / tp / ftm2v;
        let gamma2 = (Math.sqrt(rmass[i]) * Math.sqrt((24.0 * boltz) / tp / dt / mvv2e)) / ftm2v;
        gamma2 *= tsqrt;
        f[3 * i] += gamma1 * v[3 * i] + gamma2 * (rng.uniform() - 0.5);
        f[3 * i + 1] += gamma1 * v[3 * i + 1] + gamma2 * (rng.uniform() - 0.5);
        f[3 * i + 2] += gamma1 * v[3 * i + 2] + gamma2 * (rng.uniform() - 0.5);
      }
    }
  }
}
