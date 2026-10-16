/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/fix_pimd_langevin.cpp and src/REPLICA/fix_pimd_langevin.h (FixPIMDLangevin: the keyword parser of
 * the constructor, init, setup, initial_integrate, final_integrate, post_force, end_of_step, the normal-mode setup
 * nmpimd_init and nmpimd_transform, langevin_init, the PILE_L thermostat o_step (per bead random numbers, RanMars with
 * seed + bead index), the Bussi-Zykova-Parrinello and MTTK-style barostat steps (b, q, qc, a, press_v, press_o), the
 * centroid and primitive estimators, the spring energy and compute_vector) at commit b332f28. Authors of the source:
 * Yifan Li (Princeton University), the i-PI and Voth-group derivations of the PILE and BZP schemes as cited there.
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Partitions: the beads are the replicas of the session (System.replicaPeers, bead index = partition index), one process
 * per bead, as with lammps -partition N. Native exchanges the bead positions, forces and box with MPI; here the replicas
 * share a process, so the fix of the first replica (the leader) runs each integrator step for all beads in the order the
 * native code runs them (run/coupled.ts calls the phases of all replicas in lockstep). Per-bead state (the random
 * stream, the kinetic and spring energies, the barostat velocities vw) is kept on the bead's own instance.
 *
 * Quirks kept as native (b332f28): compute_xf_vir and compute_cvir never assign vir_ and centroid_vir, so t_vir is 0 and
 * t_cv is 1.5 N k T; the PIMD setup adds the spring force once more in post_force (setup() then post_force()).
 */
import { StyleError } from '../../engine/force/types';
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { RanMars } from '../random_mars';
import { nativeOrder } from '../../engine/atoms';

const MY_2PI = 2.0 * Math.PI;
const MY_SQRT2 = Math.sqrt(2.0);
const THIRD = 1.0 / 3.0;

type Method = 'pimd' | 'nmpimd';
type FMMode = 'physical' | 'normal';
type Integrator = 'obabo' | 'baoab';
type Ensemble = 'nve' | 'nvt' | 'nph' | 'npt';
type PStyle = 'iso' | 'aniso';
type Barostat = 'MTTK' | 'BZP';

/** Native fix_pimd_langevin.cpp: the Ensembles map (for the error messages). */
const ENSEMBLE_NAME: Record<Ensemble, string> = { nve: 'NVE', nvt: 'NVT', nph: 'NPH', npt: 'NPT' };

/** Pressure tensor components of one bead's virial: pair, bond, angle, dihedral, improper, kspace, fix (compute pressure). */
const virialOf = (sys: System): Float64Array => {
  const a = sys.forces();
  const w = new Float64Array(6);
  const add = (v: Float64Array) => { for (let c = 0; c < 6; c++) w[c] += v[c]; };
  add(a.virial); add(a.vbond); add(a.vangle); add(a.vdihed); add(a.vimp); add(a.vlong);
  sys.fixVirial(w);
  return w;
};

/** compute pe (all keywords): pair, bond, angle, dihedral, improper, kspace, fix. */
const peOf = (sys: System): number => {
  const a = sys.forces();
  let e = a.evdwl + a.ecoul;
  e += a.ebond;
  e += a.eangle;
  e += a.edihed;
  e += a.eimp;
  e += a.elong;
  e += sys.fixEnergy();
  return e;
};

/** The volume of a bead's box: xprd * yprd * zprd. */
const volumeOf = (sys: System): number => {
  const b = sys.state.box;
  return (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]);
};

export class FixPIMDLangevin extends Fix {
  readonly style: string = 'pimd/langevin';
  timeIntegrate = true;
  replicaCoupled = true;
  vectorFlag = true;
  sizeVector = 10;
  extvector = 1;

  protected method: Method = 'nmpimd';
  protected fmmode: FMMode = 'physical';
  protected integrator: Integrator = 'obabo';
  protected ensemble: Ensemble = 'nvt';
  protected tstatFlag = true;
  protected pstatFlag = false;
  protected removecom = true;
  protected fmass = 1.0;
  protected sp = 1.0;
  protected temp = 298.15;
  protected tau = 1.0;
  protected pilescale = 1.0;
  protected seed = -1;
  protected taup = 1.0;
  protected Pext = 1.0;
  protected pstyle: PStyle = 'iso';
  protected barostat: Barostat = 'BZP';
  protected pFlag: number[] = [0, 0, 0];
  protected pTarget: number[] = [0, 0, 0];
  protected pdim = 0;
  protected pHydro = 0;
  protected Vcoeff = 1.0;
  protected W = 0;
  protected vol0 = 0;
  protected fixedpoint: number[] = [0, 0, 0];

  /** number of beads and this bead's index (the partition) */
  protected np: number;
  protected iworld: number;
  protected kt: number;
  protected beta = 0;
  protected betaNp = 0;
  protected hbar = 0;
  protected fbond = 0;
  protected dtv = 0;
  protected dtf = 0;
  protected dtv2 = 0;
  protected dtv3 = 0;
  protected c1 = 0;
  protected c2 = 0;
  protected c1k: Float64Array = new Float64Array(0);
  protected c2k: Float64Array = new Float64Array(0);
  protected omegaK: Float64Array = new Float64Array(0);
  protected lanC: Float64Array = new Float64Array(0);
  protected lanS: Float64Array = new Float64Array(0);
  protected lam: Float64Array = new Float64Array(0);
  /** normal-mode matrices, row-major np x np: row = mode (M_x2xp[mode][bead]) */
  protected mx2xp: Float64Array = new Float64Array(0);
  protected mxp2x: Float64Array = new Float64Array(0);
  protected massFix: Float64Array = new Float64Array(0);
  protected masstotal = 0;
  protected lanTemp = 0;
  private rng: RanMars | null = null;
  private maxId = 0;

  /** per-bead outputs (set by the leader for every bead) */
  protected keBead = 0;
  protected seBead = 0;
  protected peBead = 0;
  protected springEnergy = 0;
  protected totke = 0;
  protected totalSpring = 0;
  protected pote = 0;
  protected tote = 0;
  protected tPrim = 0;
  protected tVir = 0;
  protected tCv = 0;
  protected pPrim = 0;
  protected pMd = 0;
  protected pCv = 0;
  protected totenthalpy = 0;
  protected vir = 0;
  protected virialTotal: Float64Array = new Float64Array(6);
  protected virBead = 0;
  /** vir_: the sum over the beads of x_unwrap . f (compute_xf_vir) */
  protected vir2 = 0;
  protected centroidVir = 0;
  protected cVirTensor: Float64Array = new Float64Array(6);
  protected stressTensor: Float64Array = new Float64Array(6);
  protected vw: Float64Array = new Float64Array(6);
  protected xcall: Float64Array = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.np = sys.replicaPeers().length;
    this.iworld = Math.max(0, sys.replicaPeers().indexOf(sys));
    this.kt = sys.state.units.boltz * this.temp;
    const num = (key: string, v: string | undefined): number => {
      const x = Number(v);
      if (v === undefined || v === '' || !Number.isFinite(x)) throw new StyleError(`Expected floating point parameter for ${key} of fix ${this.style}`);
      return x;
    };
    for (let i = 0; i < args.length - 1; i += 2) {
      const key = args[i], val = args[i + 1];
      if (key === 'method') {
        if (val === 'nmpimd') this.method = 'nmpimd';
        else if (val === 'pimd') this.method = 'pimd';
        else throw new StyleError(`Unknown method parameter for fix ${this.style}`);
      } else if (key === 'integrator') {
        if (val === 'obabo') this.integrator = 'obabo';
        else if (val === 'baoab') this.integrator = 'baoab';
        else throw new StyleError(`Unknown integrator parameter for fix ${this.style}. Only obabo and baoab integrators are supported!`);
      } else if (key === 'ensemble') {
        if (val === 'nve') { this.ensemble = 'nve'; this.tstatFlag = false; this.pstatFlag = false; }
        else if (val === 'nvt') { this.ensemble = 'nvt'; this.tstatFlag = true; this.pstatFlag = false; }
        else if (val === 'nph') { this.ensemble = 'nph'; this.tstatFlag = false; this.pstatFlag = true; }
        else if (val === 'npt') { this.ensemble = 'npt'; this.tstatFlag = true; this.pstatFlag = true; }
        else throw new StyleError(`Unknown ensemble parameter for fix ${this.style}. Only nve, nvt, nph, and npt ensembles are supported!`);
      } else if (key === 'fmass') {
        this.fmass = num(key, val);
        if (this.fmass < 0.0 || this.fmass > this.np) throw new StyleError(`Invalid fmass value for fix ${this.style}`);
      } else if (key === 'sp') {
        this.sp = num(key, val);
        if (this.sp < 0.0) throw new StyleError(`Invalid sp value for fix ${this.style}`);
      } else if (key === 'fmmode') {
        if (val === 'physical') this.fmmode = 'physical';
        else if (val === 'normal') this.fmmode = 'normal';
        else throw new StyleError(`Unknown fictitious mass mode for fix ${this.style}. Only physical mass and normal mode mass are supported!`);
      } else if (key === 'scale') {
        if (this.method === 'pimd') throw new StyleError('The scale parameter of the PILE_L thermostat is not supported for method pimd. Delete scale parameter if you do want to use method pimd.');
        this.pilescale = num(key, val);
        if (this.pilescale < 0.0) throw new StyleError(`Invalid PILE_L scale value for fix ${this.style}`);
      } else if (key === 'temp') {
        this.temp = num(key, val);
        if (this.temp < 0.0) throw new StyleError(`Invalid temp value for fix ${this.style}`);
      } else if (key === 'thermostat') {
        if (val === 'PILE_L') {
          const seedWord = args[i + 2];
          if (seedWord === undefined) throw new StyleError(`fix ${this.style} thermostat PILE_L needs a seed`);
          const seed = Math.trunc(Number(seedWord));
          if (!Number.isFinite(seed)) throw new StyleError(`Expected integer parameter for the seed of fix ${this.style}`);
          this.seed = seed;
          i++;
        }
      } else if (key === 'tau') {
        this.tau = num(key, val);
      } else if (key === 'barostat') {
        if (val === 'MTTK') this.barostat = 'MTTK';
        else if (val === 'BZP') this.barostat = 'BZP';
        else throw new StyleError(`Unknown barostat parameter for fix ${this.style}`);
      } else if (key === 'iso' || key === 'aniso') {
        this.pstyle = key === 'iso' ? 'iso' : 'aniso';
        this.pFlag = [1, 1, 1];
        this.Pext = num(key, val);
        this.pTarget = [this.Pext, this.Pext, this.Pext];
        this.pdim = 3;
      } else if (key === 'x' || key === 'y' || key === 'z') {
        const d = key === 'x' ? 0 : key === 'y' ? 1 : 2;
        this.pstyle = 'aniso';
        this.pFlag[d] = 1;
        this.pTarget[d] = num(key, val);
        this.pdim++;
      } else if (key === 'taup') {
        this.taup = num(key, val);
        if (this.taup <= 0.0) throw new StyleError(`Invalid tau_p value for fix ${this.style}`);
      } else if (key === 'fixcom') {
        if (val === 'yes') this.removecom = true;
        else if (val === 'no') this.removecom = false;
      } else {
        throw new StyleError(`Unknown keyword ${key} for fix ${this.style}`);
      }
    }
    if (this.pstatFlag && !this.pdim) throw new StyleError(`Must use pressure coupling with ${ENSEMBLE_NAME[this.ensemble]} ensemble`);
    if (!this.pstatFlag && this.pdim) throw new StyleError(`Must not use pressure coupling with ${ENSEMBLE_NAME[this.ensemble]} ensemble`);
    if (this.method === 'pimd' && this.pstatFlag) throw new StyleError('Pressure control has not been supported for method pimd yet. Please set method to nmpimd.');
    if (this.method === 'pimd' && this.fmmode === 'normal') throw new StyleError('Normal mode mass is not supported for method pimd. Please set method to nmpimd.');
    if (this.pstatFlag) this.sizeVector = this.pstyle === 'iso' ? 15 : 17;
    else this.sizeVector = 10;
    this.kt = sys.state.units.boltz * this.temp;
    if (this.pstatFlag) this.baroInit();
    const b = sys.state.box;
    this.vol0 = volumeOf(sys);
    this.fixedpoint = [0.5 * (b.lo[0] + b.hi[0]), 0.5 * (b.lo[1] + b.hi[1]), 0.5 * (b.lo[2] + b.hi[2])];
    if (this.pstatFlag) this.pHydro = (this.pTarget[0] + this.pTarget[1] + this.pTarget[2]) / this.pdim;
    if (this.tstatFlag) {
      this.lanTemp = this.temp;
        this.rng = new RanMars(this.seed + this.iworld);
    }
  }

  /** The bead instances (every replica's fix), this fix's own first if it is the leader. */
  protected ring(): FixPIMDLangevin[] {
    return this.sys.replicaPeers().map((p) => {
      const f = p.fixes.find((x) => x.id === this.id);
      if (!(f instanceof FixPIMDLangevin)) throw new StyleError(`fix ${this.id} is not defined on every replica`);
      return f;
    });
  }

  protected get isLeader(): boolean { return this.sys.replicaPeers()[0] === this.sys; }

  /** The number of atoms in the fix's group (group->count). */
  protected groupCount(): number {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) n++;
    return n;
  }

  /** baro_init: the barostat mass W (Vcoeff = 1, velocities zero). */
  private baroInit(): void {
    this.vw.fill(0);
    const n = this.groupCount();
    if (this.pstyle === 'iso') this.W = 3 * n * this.taup * this.taup * this.np * this.kt;
    else this.W = n * this.taup * this.taup * this.np * this.kt;
    this.Vcoeff = 1.0;
  }

  init(): void {
    const sys = this.sys;
    const s = sys.state;
    const u = s.units;
    this.np = sys.replicaPeers().length;
    this.iworld = Math.max(0, sys.replicaPeers().indexOf(sys));
    this.maxId = 0;
    for (let i = 0; i < s.n; i++) if (s.id[i] > this.maxId) this.maxId = s.id[i];
    if (this.maxId !== s.n) throw new StyleError(`Atom IDs must be consecutive for fix ${this.style}`);
    this.masstotal = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.masstotal += s.massByType[s.type[i]];
    const planck = this.sp * u.hplanck;
    this.hbar = planck / MY_2PI;
    this.beta = 1.0 / (u.boltz * this.temp);
    const _fbond = 1.0 * this.np * this.np / (this.beta * this.beta * this.hbar * this.hbar);
    this.betaNp = 1.0 / u.boltz / this.temp * (1.0 / this.np);
    this.fbond = _fbond * u.mvv2e;
    this.dtf = 0.5 * s.dt * u.ftm2v;
    this.dtv = 0.5 * s.dt;
    this.dtv2 = this.dtv * this.dtv;
    this.dtv3 = THIRD * this.dtv2 * this.dtv * u.ftm2v;
    this.nmpimdInit();
    this.langevinInit();
    if (this.pstatFlag) this.baroInit();
    if (this.tstatFlag && !this.rng) this.rng = new RanMars(this.seed + this.iworld);
    this.checkBeadsCompatible();
  }

  /** The beads must define this fix (same input). */
  protected checkBeadsCompatible(): void {
    this.ring();
  }

  /** nmpimd_init: the eigenvalues lam and the normal-mode matrices, and the fictitious masses of this bead. */
  protected nmpimdInit(): void {
    const np = this.np;
    const s = this.sys.state;
    this.lam = new Float64Array(np);
    this.mx2xp = new Float64Array(np * np);
    this.mxp2x = new Float64Array(np * np);
    const M = (i: number, j: number) => i * np + j;
    for (let i = 0; i < np; i++) {
      const sinTmp = Math.sin(i * Math.PI / np);
      this.lam[i] = 4 * sinTmp * sinTmp;
    }
    const sqrtnp = Math.sqrt(np);
    for (let j = 0; j < np; j++) {
      for (let i = 1; i < Math.trunc(np / 2) + 1; i++) {
        this.mx2xp[M(i, j)] = MY_SQRT2 * Math.cos(MY_2PI * i * j / np) / sqrtnp;
      }
      for (let i = Math.trunc(np / 2) + 1; i < np; i++) {
        this.mx2xp[M(i, j)] = MY_SQRT2 * Math.sin(MY_2PI * i * j / np) / sqrtnp;
      }
    }
    for (let i = 0; i < np; i++) {
      this.mx2xp[M(0, i)] = 1.0 / sqrtnp;
      if (np % 2 === 0) this.mx2xp[M(np / 2, i)] = 1.0 / sqrtnp * (i % 2 === 0 ? 1.0 : -1.0);
    }
    for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) this.mxp2x[M(i, j)] = this.mx2xp[M(j, i)];
    this.massFix = new Float64Array(s.ntypes + 1);
    for (let i = 1; i <= s.ntypes; i++) {
      let m = s.massByType[i];
      m *= this.fmass;
      if (this.iworld) {
        if (this.fmmode === 'physical') m *= 1.0;
        else m *= this.lam[this.iworld];
      }
      this.massFix[i] = m;
    }
  }

  /** langevin_init: the bead frequencies, the Langevin factors of the normal modes and the PILE_L factors. */
  protected langevinInit(): void {
    const np = this.np;
    const dt = this.sys.state.dt;
    const beta = 1.0 / this.kt;
    const omegaNp = np / beta / this.hbar;
    const omegaNpDtHalf = omegaNp * dt * 0.5;
    this.omegaK = new Float64Array(np);
    this.lanC = new Float64Array(np);
    this.lanS = new Float64Array(np);
    if (this.method === 'nmpimd') {
      if (this.fmmode === 'physical') {
        for (let i = 0; i < np; i++) {
          this.omegaK[i] = omegaNp * Math.sqrt(this.lam[i]) / Math.sqrt(this.fmass);
          this.lanC[i] = Math.cos(Math.sqrt(this.lam[i]) * omegaNpDtHalf);
          this.lanS[i] = Math.sin(Math.sqrt(this.lam[i]) * omegaNpDtHalf);
        }
      } else {
        for (let i = 0; i < np; i++) {
          this.omegaK[i] = omegaNp / Math.sqrt(this.fmass);
          this.lanC[i] = Math.cos(omegaNpDtHalf);
          this.lanS[i] = Math.sin(omegaNpDtHalf);
        }
      }
    }
    const gamma = this.tau > 0 ? 1.0 / this.tau : np / beta / this.hbar;
    if (this.integrator === 'obabo') this.c1 = Math.exp(-gamma * 0.5 * dt);
    else this.c1 = Math.exp(-gamma * dt);
    this.c2 = Math.sqrt(1.0 - this.c1 * this.c1);
    this.c1k = new Float64Array(np);
    this.c2k = new Float64Array(np);
    if (this.method === 'nmpimd') {
      this.c1k[0] = this.c1;
      this.c2k[0] = this.c2;
      for (let i = 1; i < np; i++) {
        const tauK = 0.5 / this.pilescale / this.omegaK[i];
        if (this.integrator === 'obabo') this.c1k[i] = Math.exp(-0.5 * dt / tauK);
        else this.c1k[i] = Math.exp(-1.0 * dt / tauK);
        this.c2k[i] = Math.sqrt(1.0 - this.c1k[i] * this.c1k[i]);
      }
    }
  }

  /** The tagged (atom ID order) copy of a per-atom array of one bead: slot 3 * (ID - 1) + dimension. */
  protected tagged(arr: Float64Array): Float64Array {
    const s = this.sys.state;
    const out = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) out[3 * (s.id[i] - 1) + d] = arr[3 * i + d];
    return out;
  }

  /** domain->unmap (sign +1) or unmap_inv (sign -1) of every atom of every bead, with its image flags. */
  protected unmapAll(R: FixPIMDLangevin[], sign: 1 | -1): void {
    for (const b of R) {
      const s = b.sys.state;
      const g = b.sys.geom;
      for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) g.addEdge(s.x, i, d, sign * s.image[3 * i + d]);
    }
  }

  /** The Cartesian transform of every bead's per-atom array: des_b[i] = sum_j M[b][j] src_j[i] (bufsorted order). */
  protected transformAll(R: FixPIMDLangevin[], which: 'x' | 'f', M: Float64Array): void {
    const np = this.np;
    const src = R.map((b) => b.tagged(which === 'x' ? b.sys.state.x : b.sys.state.f));
    R.forEach((b, row) => {
      const s = b.sys.state;
      const des = which === 'x' ? s.x : s.f;
      for (let i = 0; i < s.n; i++) {
        for (let d = 0; d < 3; d++) {
          let acc = 0.0;
          const slot = 3 * (s.id[i] - 1) + d;
          for (let j = 0; j < np; j++) acc += src[j][slot] * M[row * np + j];
          des[3 * i + d] = acc;
        }
      }
    });
  }

  /** The b_step: the half kick of every bead's atoms in the group. */
  protected bStep(R: FixPIMDLangevin[]): void {
    for (const b of R) {
      const s = b.sys.state;
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b.groupBit)) continue;
        const dtfm = b.dtf / b.massFix[s.type[i]];
        s.v[3 * i] += dtfm * s.f[3 * i];
        s.v[3 * i + 1] += dtfm * s.f[3 * i + 1];
        s.v[3 * i + 2] += dtfm * s.f[3 * i + 2];
      }
    }
  }

  /** The q_step (PIMD): x += dt/2 v for every bead, not with a barostat. */
  protected qStep(R: FixPIMDLangevin[]): void {
    if (this.pstatFlag) return;
    for (const b of R) {
      const s = b.sys.state;
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b.groupBit)) continue;
        s.x[3 * i] += b.dtv * s.v[3 * i];
        s.x[3 * i + 1] += b.dtv * s.v[3 * i + 1];
        s.x[3 * i + 2] += b.dtv * s.v[3 * i + 2];
      }
    }
  }

  /** The qc_step: the centroid (bead 0) drift, with the BZP box and position scaling when the barostat is on. */
  protected qcStep(R: FixPIMDLangevin[]): void {
    const b0 = R[0];
    if (!this.pstatFlag) {
      const s = b0.sys.state;
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b0.groupBit)) continue;
        s.x[3 * i] += b0.dtv * s.v[3 * i];
        s.x[3 * i + 1] += b0.dtv * s.v[3 * i + 1];
        s.x[3 * i + 2] += b0.dtv * s.v[3 * i + 2];
      }
      return;
    }
    const s = b0.sys.state;
    const vw = b0.vw;
    if (b0.pstyle === 'iso') { vw[1] = vw[0]; vw[2] = vw[0]; }
    const expq = [0, 0, 0], expp = [0, 0, 0];
    for (let j = 0; j < 3; j++) {
      expq[j] = Math.exp(b0.dtv * vw[j]);
      expp[j] = Math.exp(-b0.dtv * vw[j]);
    }
    if (this.barostat === 'BZP') {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b0.groupBit)) continue;
        for (let j = 0; j < 3; j++) {
          if (b0.pFlag[j]) {
            s.x[3 * i + j] = expq[j] * s.x[3 * i + j] + (expq[j] - expp[j]) / 2.0 / vw[j] * s.v[3 * i + j];
            s.v[3 * i + j] = expp[j] * s.v[3 * i + j];
          } else {
            s.x[3 * i + j] += b0.dtv * s.v[3 * i + j];
          }
        }
      }
      const lo = [0, 0, 0], hi = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        const oldlo = s.box.lo[d], oldhi = s.box.hi[d];
        lo[d] = (oldlo - b0.fixedpoint[d]) * expq[d] + b0.fixedpoint[d];
        hi[d] = (oldhi - b0.fixedpoint[d]) * expq[d] + b0.fixedpoint[d];
      }
      // the box is one for all beads (native MPI_Bcast of boxlo and boxhi from bead 0)
      for (const b of R) b.sys.setBox(lo, hi, [...b.sys.state.box.tilt], 0);
    }
  }

  /** The a_step: the harmonic exchange of the normal modes of the beads other than the centroid. */
  protected aStep(R: FixPIMDLangevin[]): void {
    for (const b of R) {
      if (b.iworld === 0) continue;
      const s = b.sys.state;
      const ok = b.omegaK[b.iworld], ls = b.lanS[b.iworld], lc = b.lanC[b.iworld];
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b.groupBit)) continue;
        const x0 = s.x[3 * i], x1 = s.x[3 * i + 1], x2 = s.x[3 * i + 2];
        const v0 = s.v[3 * i], v1 = s.v[3 * i + 1], v2 = s.v[3 * i + 2];
        s.x[3 * i] = lc * x0 + 1.0 / ok * ls * v0;
        s.x[3 * i + 1] = lc * x1 + 1.0 / ok * ls * v1;
        s.x[3 * i + 2] = lc * x2 + 1.0 / ok * ls * v2;
        s.v[3 * i] = -1.0 * ok * ls * x0 + lc * v0;
        s.v[3 * i + 1] = -1.0 * ok * ls * x1 + lc * v1;
        s.v[3 * i + 2] = -1.0 * ok * ls * x2 + lc * v2;
      }
    }
  }

  /** The PILE_L thermostat step of one bead (o_step): per atom in the group, x, y, z in order. */
  protected oStep(): void {
    const s = this.sys.state;
    const u = s.units;
    const betaNpL = 1.0 / u.boltz / this.lanTemp * (1.0 / this.np) * u.mvv2e;
    const rng = this.rng;
    if (!rng) throw new StyleError(`fix ${this.style} has no random stream (thermostat off)`);
    const nm = this.method === 'nmpimd';
    const c1 = nm ? this.c1k[this.iworld] : this.c1;
    const c2 = nm ? this.c2k[this.iworld] : this.c2;
    // the draws follow the native atom order (atom sorting at setup and reneighboring): nativeOrder(s)
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = this.massFix[s.type[i]];
      s.v[3 * i] = c1 * s.v[3 * i] + c2 * Math.sqrt(1.0 / m / betaNpL) * rng.gaussian();
      s.v[3 * i + 1] = c1 * s.v[3 * i + 1] + c2 * Math.sqrt(1.0 / m / betaNpL) * rng.gaussian();
      s.v[3 * i + 2] = c1 * s.v[3 * i + 2] + c2 * Math.sqrt(1.0 / m / betaNpL) * rng.gaussian();
    }
  }

  /** remove_com_motion: the centre-of-mass velocity of the group (the centroid bead only, for nmpimd). */
  protected removeComMotion(R: FixPIMDLangevin[]): void {
    const targets = this.method === 'nmpimd' ? R.slice(0, 1) : R;
    for (const b of targets) {
      const s = b.sys.state;
      const p = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b.groupBit)) continue;
        const m = s.massByType[s.type[i]];
        p[0] += s.v[3 * i] * m;
        p[1] += s.v[3 * i + 1] * m;
        p[2] += s.v[3 * i + 2] * m;
      }
      const vcm = [p[0] / b.masstotal, p[1] / b.masstotal, p[2] / b.masstotal];
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b.groupBit)) continue;
        s.v[3 * i] -= vcm[0];
        s.v[3 * i + 1] -= vcm[1];
        s.v[3 * i + 2] -= vcm[2];
      }
    }
  }

  /** press_o_step: the barostat velocity's Langevin kick (bead 0's stream), then broadcast. */
  protected pressOStep(R: FixPIMDLangevin[]): void {
    const b0 = R[0];
    const rng = b0.rng;
    if (!rng) throw new StyleError(`fix ${this.style} has no random stream (thermostat off)`);
    if (this.pstyle === 'iso') {
      b0.vw[0] = this.c1 * b0.vw[0] + this.c2 * Math.sqrt(1.0 / this.W / this.betaNp) * rng.gaussian();
      for (const b of R) b.vw[0] = b0.vw[0];
    } else {
      for (let ii = 0; ii < 3; ii++) {
        if (b0.pFlag[ii]) b0.vw[ii] = this.c1 * b0.vw[ii] + this.c2 * Math.sqrt(1.0 / this.W / this.betaNp) * rng.gaussian();
      }
      for (const b of R) for (let ii = 0; ii < 3; ii++) b.vw[ii] = b0.vw[ii];
    }
  }

  /** compute_totke: the kinetic energy of every bead (fictitious masses) and their sum. */
  protected computeTotke(R: FixPIMDLangevin[]): void {
    let total = 0.0;
    const sums: number[] = [];
    for (const b of R) {
      const s = b.sys.state;
      let kine = 0.0;
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b.groupBit)) continue;
        const m = b.massFix[s.type[i]];
        for (let j = 0; j < 3; j++) kine += 0.5 * m * s.v[3 * i + j] * s.v[3 * i + j];
      }
      kine *= s.units.mvv2e;
      sums.push(kine);
    }
    for (let k = 0; k < R.length; k++) {
      R[k].keBead = sums[k];
      total += sums[k];
    }
    for (const b of R) b.totke = total;
  }

  /** compute_p_cv: the centroid-virial pressure estimator (bead 0's kinetic energy for nmpimd). */
  protected computePCv(R: FixPIMDLangevin[]): void {
    const invVolume = 1.0 / volumeOf(this.sys);
    const b0 = R[0];
    const u = this.sys.state.units;
    for (const b of R) b.pMd = THIRD * invVolume * (b.totke + b.vir);
    if (this.method === 'nmpimd') {
      const p = THIRD * invVolume * ((2.0 * b0.keBead - b0.centroidVir) * u.nktv2p + b0.vir) / this.np;
      for (const b of R) b.pCv = p;
    } else {
      for (const b of R) b.pCv = THIRD * invVolume * ((2.0 * b.totke / this.np - b.centroidVir) * u.nktv2p + b.vir) / this.np;
    }
  }

  /** press_v_step: the barostat velocity from the pressure difference and the forces of bead 0. */
  protected pressVStep(R: FixPIMDLangevin[]): void {
    const b0 = R[0];
    const volume = volumeOf(this.sys);
    const u = this.sys.state.units;
    const np = this.np;
    const dvwOf = (b: FixPIMDLangevin, ii: number | null): number => {
      const s = b.sys.state;
      let dvwProc = 0.0;
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b.groupBit)) continue;
        if (ii === null) {
          for (let j = 0; j < 3; j++) {
            const f = s.f[3 * i + j], v = s.v[3 * i + j];
            dvwProc += b.dtv2 * f * v / b.W + b.dtv3 * f * f / b.massFix[s.type[i]] / b.W;
          }
        } else {
          const f = s.f[3 * i + ii], v = s.v[3 * i + ii];
          dvwProc += b.dtv2 * f * v / b.W + b.dtv3 * f * f / b.massFix[s.type[i]] / b.W;
        }
      }
      return dvwProc;
    };
    if (this.pstyle === 'iso') {
      if (this.barostat === 'BZP') {
        for (const b of R) b.vw[0] += b.dtv * 3 * (volume * np * (b.pCv - b.pHydro) / u.nktv2p + b.Vcoeff / b.betaNp) / b.W;
        const dvw = dvwOf(b0, null);
        b0.vw[0] += dvw;
        for (const b of R) b.vw[0] = b0.vw[0];
      } else {
        const mtkTerm1 = 2.0 / this.groupCount() * b0.totke / 3.0;
        for (const b of R) b.vw[0] += 0.5 * b.dtv * (volume * np * (b.pMd - b.pHydro) + mtkTerm1) / b.W;
      }
    } else {
      this.computeStressTensor(R);
      for (let ii = 0; ii < 3; ii++) {
        if (!b0.pFlag[ii]) continue;
        for (const b of R) {
          b.vw[ii] += b.dtv * (volume * np * (b.stressTensor[ii] - b.pHydro) / u.nktv2p + b.Vcoeff / b.betaNp) / b.W;
        }
        b0.vw[ii] += dvwOf(b0, ii);
      }
    }
  }

  /** compute_stress_tensor (anisotropic barostat): bead 0's kinetic tensor and the centroid virial; broadcast. */
  protected computeStressTensor(R: FixPIMDLangevin[]): void {
    const b0 = R[0];
    const s = b0.sys.state;
    const u = s.units;
    const invVolume = 1.0 / volumeOf(this.sys);
    const ke = new Float64Array(6);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & b0.groupBit)) continue;
      const m = b0.massFix[s.type[i]];
      ke[0] += 0.5 * m * s.v[3 * i] * s.v[3 * i] * u.mvv2e;
      ke[1] += 0.5 * m * s.v[3 * i + 1] * s.v[3 * i + 1] * u.mvv2e;
      ke[2] += 0.5 * m * s.v[3 * i + 2] * s.v[3 * i + 2] * u.mvv2e;
      ke[3] += 0.5 * m * s.v[3 * i] * s.v[3 * i + 1] * u.mvv2e;
      ke[4] += 0.5 * m * s.v[3 * i] * s.v[3 * i + 2] * u.mvv2e;
      ke[5] += 0.5 * m * s.v[3 * i + 1] * s.v[3 * i + 2] * u.mvv2e;
    }
    const st = new Float64Array(6);
    for (let c = 0; c < 6; c++) st[c] = invVolume * ((2 * ke[c] - b0.cVirTensor[c]) * u.nktv2p + b0.virialTotal[c]) / this.np;
    for (const b of R) b.stressTensor.set(st);
  }

  /** compute_spring_energy (base): the normal-mode bead energies, or the sum of the PIMD springs of the beads. */
  protected computeSpringEnergy(R: FixPIMDLangevin[]): void {
    if (this.method === 'nmpimd') {
      for (const b of R) {
        const s = b.sys.state;
        let se = 0.0;
        for (let i = 0; i < s.n; i++) {
          if (!(s.mask[i] & b.groupBit)) continue;
          const x0 = s.x[3 * i], x1 = s.x[3 * i + 1], x2 = s.x[3 * i + 2];
          se += 0.5 * s.massByType[s.type[i]] * b.fbond * b.lam[b.iworld] * (x0 * x0 + x1 * x1 + x2 * x2);
        }
        b.springEnergy = se;
      }
    }
    let total = 0.0;
    for (const b of R) { b.seBead = b.springEnergy; total += b.springEnergy; }
    for (const b of R) b.totalSpring = total;
  }

  /** compute_t_prim (base): the primitive kinetic estimator. */
  protected computeTPrim(R: FixPIMDLangevin[]): void {
    const n = this.groupCount();
    for (const b of R) {
      b.tPrim = 1.5 * n * this.np * this.sys.state.units.boltz * this.temp - b.totalSpring * (1.0 / this.np);
    }
  }

  /** compute_p_prim: the primitive pressure estimator. */
  protected computePPrim(R: FixPIMDLangevin[]): void {
    const u = this.sys.state.units;
    const invVolume = 1.0 / volumeOf(this.sys);
    const n = this.groupCount();
    for (const b of R) {
      let p = n * this.np * u.boltz * this.temp * invVolume - 1.0 / 1.5 * invVolume * b.totalSpring;
      p *= u.nktv2p;
      b.pPrim = p;
    }
  }

  /** collect_xc: the centroid positions (bead 0's coordinates over sqrt(np)), for every atom by ID. */
  protected collectXc(R: FixPIMDLangevin[]): void {
    const b0 = R[0];
    const s = b0.sys.state;
    const xc = new Float64Array(3 * s.n);
    const sqrtnp = Math.sqrt(this.np);
    for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) xc[3 * (s.id[i] - 1) + d] = s.x[3 * i + d] / sqrtnp;
    this.xcall = xc;
  }

  /** Spring force of one bead's PIMD ring (base, method pimd): the neighbours' unmapped positions by ID. */
  protected springForceBead(b: FixPIMDLangevin, xt: Float64Array[]): void {
    const s = b.sys.state;
    const np = this.np;
    const xLast = xt[(b.iworld - 1 + np) % np];
    const xNext = xt[(b.iworld + 1) % np];
    b.springEnergy = 0.0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & b.groupBit)) continue;
      const slot = 3 * (s.id[i] - 1);
      const d1 = [0, 0, 0], d2 = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        d1[d] = xLast[slot + d] - s.x[3 * i + d];
        d2[d] = xNext[slot + d] - s.x[3 * i + d];
      }
      const ff = b.fbond * s.massByType[s.type[i]];
      for (let d = 0; d < 3; d++) s.f[3 * i + d] += (d1[d] + d2[d]) * ff;
      b.springEnergy += 0.5 * ff * (d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]);
    }
  }

  /** The PIMD spring step of every bead (native prepare_coordinates + spring_force). */
  protected pimdSprings(R: FixPIMDLangevin[]): void {
    const xt = R.map((b) => b.tagged(b.sys.state.x));
    for (const b of R) b.prepareBead(xt);
    for (const b of R) b.springForceBead(b, xt);
  }

  /** Hook (native prepare_coordinates): the bead's exchange data from the neighbours' positions (bosonic override). */
  protected prepareBead(_xt: Float64Array[]): void { /* the PIMD springs need nothing prepared */ }

  /** Setup: the forces and the first estimators of the ring (native setup()). */
  setup(): void {
    if (!this.isLeader) return;
    const R = this.ring();
    this.unmapAll(R, 1);
    if (this.method === 'nmpimd') {
      this.transformAll(R, 'x', this.mx2xp);
      this.collectXc(R);
      this.computeSpringEnergy(R);
      this.computeTPrim(R);
      this.computePPrim(R);
      this.transformAll(R, 'x', this.mxp2x);
    } else {
      this.pimdSprings(R);
      this.collectXc(R);
      this.computeSpringEnergy(R);
      this.computeTPrim(R);
      this.computePPrim(R);
    }
    this.unmapAll(R, -1);
    this.postForce();
    this.computeTotke(R);
    this.endOfStep();
  }

  /** initial_integrate (native: the first half of the integrator, then the centroid estimators). */
  initialIntegrate(): void {
    if (!this.isLeader) return;
    const R = this.ring();
    this.unmapAll(R, 1);
    const nm = this.method === 'nmpimd';
    if (this.integrator === 'obabo') {
      if (this.tstatFlag) {
        for (const b of R) b.oStep();
        if (this.removecom) this.removeComMotion(R);
        if (this.pstatFlag) this.pressOStep(R);
      }
      if (this.pstatFlag) {
        this.computeTotke(R);
        this.computePCv(R);
        this.pressVStep(R);
      }
      this.bStep(R);
      if (nm) {
        this.transformAll(R, 'x', this.mx2xp);
        this.qcStep(R);
        this.aStep(R);
        this.qcStep(R);
        this.aStep(R);
      } else {
        this.qStep(R);
        this.qStep(R);
      }
    } else {
      if (this.pstatFlag) {
        this.computeTotke(R);
        this.computePCv(R);
        this.pressVStep(R);
      }
      this.bStep(R);
      if (nm) {
        this.transformAll(R, 'x', this.mx2xp);
        this.qcStep(R);
        this.aStep(R);
      } else {
        this.qStep(R);
      }
      if (this.tstatFlag) {
        for (const b of R) b.oStep();
        if (this.removecom) this.removeComMotion(R);
        if (this.pstatFlag) this.pressOStep(R);
      }
      if (nm) {
        this.qcStep(R);
        this.aStep(R);
      } else {
        this.qStep(R);
      }
    }
    this.collectXc(R);
    if (nm) {
      this.computeSpringEnergy(R);
      this.computeTPrim(R);
      this.computePPrim(R);
      this.transformAll(R, 'x', this.mxp2x);
    }
    this.unmapAll(R, -1);
  }

  /** final_integrate: the second half of the barostat and the thermostat (OBABO), the second b step. */
  finalIntegrate(): void {
    if (!this.isLeader) return;
    const R = this.ring();
    if (this.pstatFlag) {
      this.computeTotke(R);
      this.computePCv(R);
      this.pressVStep(R);
    }
    this.bStep(R);
    if (this.integrator === 'obabo' && this.tstatFlag) {
      for (const b of R) b.oStep();
      if (this.removecom) this.removeComMotion(R);
      if (this.pstatFlag) this.pressOStep(R);
    }
  }

  /** post_force: the estimators, the PIMD springs, the potential energies and the normal-mode forces. */
  postForce(): void {
    if (!this.isLeader) return;
    const R = this.ring();
    const np = this.np;
    // x_unwrap (unmapped copy) and the centroid per atom
    const xu = R.map((b) => {
      const s = b.sys.state;
      const u = Float64Array.from(s.x);
      for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) b.sys.geom.addEdge(u, i, d, s.image[3 * i + d]);
      return u;
    });
    const xcOf = (b: FixPIMDLangevin, i: number, d: number) => this.xcall[3 * (b.sys.state.id[i] - 1) + d];
    // compute_vir per bead, summed over the beads
    this.virialTotal = new Float64Array(6);
    let vir = 0;
    for (const b of R) {
      const v = virialOf(b.sys);
      // compute pressure thermo_temp virial: the virial keyword leaves the kinetic term out (keflag 0 in
      // compute_pressure.cpp at b332f28), so the vector is the virial over the volume
      const invVolume = 1.0 / volumeOf(b.sys);
      const vol = volumeOf(b.sys);
      const vec = new Float64Array(6);
      for (let c = 0; c < 6; c++) vec[c] = v[c] * invVolume * b.sys.state.units.nktv2p;
      const part = new Float64Array(6);
      for (let c = 0; c < 6; c++) part[c] = vec[c] * vol;
      b.virBead = part[0] + part[1] + part[2];
      vir += b.virBead;
      for (let c = 0; c < 6; c++) this.virialTotal[c] += part[c];
    }
    for (const b of R) { b.vir = vir; b.virialTotal.set(this.virialTotal); }
    // compute_xf_vir and compute_cvir: sums over all beads (MPI_Allreduce on the universe, as native); the ANISO
    // centroid-virial tensor is summed the same way
    let xfSum = 0.0, cvirSum = 0.0;
    const cSum = new Float64Array(6);
    R.forEach((b, bi) => {
      const s = b.sys.state;
      let xf = 0.0, xcf = 0.0;
      const c = new Float64Array(6);
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & b.groupBit)) continue;
        const u0 = xu[bi][3 * i], u1 = xu[bi][3 * i + 1], u2 = xu[bi][3 * i + 2];
        const d0 = u0 - xcOf(b, i, 0), d1 = u1 - xcOf(b, i, 1), d2 = u2 - xcOf(b, i, 2);
        for (let j = 0; j < 3; j++) xf += xu[bi][3 * i + j] * s.f[3 * i + j];
        xcf += d0 * s.f[3 * i];
        xcf += d1 * s.f[3 * i + 1];
        xcf += d2 * s.f[3 * i + 2];
        if (this.pstyle === 'aniso') {
          c[0] += d0 * s.f[3 * i];
          c[1] += d1 * s.f[3 * i + 1];
          c[2] += d2 * s.f[3 * i + 2];
          c[3] += d0 * s.f[3 * i + 1];
          c[4] += d0 * s.f[3 * i + 2];
          c[5] += d1 * s.f[3 * i + 2];
        }
      }
      xfSum += xf;
      cvirSum += xcf;
      for (let k = 0; k < 6; k++) cSum[k] += c[k];
    });
    for (const b of R) {
      b.vir2 = xfSum;
      b.centroidVir = cvirSum;
      if (this.pstyle === 'aniso') b.cVirTensor = Float64Array.from(cSum);
    }
    for (const b of R) {
      b.tVir = -0.5 * (1.0 / np) * b.vir2;
      b.tCv = 1.5 * this.groupCount() * this.sys.state.units.boltz * this.temp - 0.5 * (1.0 / np) * b.centroidVir;
    }
    if (this.method === 'pimd') {
      this.unmapAll(R, 1);
      this.pimdSprings(R);
      this.computeSpringEnergy(R);
      this.computeTPrim(R);
      this.unmapAll(R, -1);
    }
    for (const b of R) {
      b.peBead = peOf(b.sys);
    }
    let pote = 0.0;
    for (const b of R) pote += b.peBead;
    for (const b of R) b.pote = pote;
    if (this.method === 'nmpimd') this.transformAll(R, 'f', this.mx2xp);
  }

  /** end_of_step: the kinetic energy, the pressure estimators, the total energies. */
  endOfStep(): void {
    if (!this.isLeader) return;
    const R = this.ring();
    this.computeTotke(R);
    this.computePCv(R);
    for (const b of R) b.tote = b.totke + b.pote + b.totalSpring;
    if (this.pstatFlag) this.computeTotenthalpy(R);
  }

  /** compute_totenthalpy: the enthalpy of the barostat ensemble (BZP or MTTK) of each bead. */
  protected computeTotenthalpy(R: FixPIMDLangevin[]): void {
    const u = this.sys.state.units;
    const kt = this.kt;
    const volume = volumeOf(this.sys);
    for (const b of R) {
      if (this.barostat === 'BZP') {
        if (this.pstyle === 'iso') {
          b.totenthalpy = b.tote + 0.5 * this.W * b.vw[0] * b.vw[0] * (1.0 / this.np) + this.pHydro * volume / u.nktv2p - this.Vcoeff * kt * Math.log(volume);
        } else {
          b.totenthalpy = b.tote + 0.5 * this.W * b.vw[0] * b.vw[0] * (1.0 / this.np) +
            0.5 * this.W * b.vw[1] * b.vw[1] * (1.0 / this.np) + 0.5 * this.W * b.vw[2] * b.vw[2] * (1.0 / this.np) +
            this.pHydro * volume / u.nktv2p - this.Vcoeff * kt * Math.log(volume);
        }
      } else {
        b.totenthalpy = b.tote + 1.5 * this.W * b.vw[0] * b.vw[0] * (1.0 / this.np) + this.pHydro * (volume - this.vol0);
      }
    }
  }

  /** compute_vector: the bead's estimators (see docs.lammps.org/fix_pimd.html). */
  computeVector(n: number): number {
    if (n === 0) return this.keBead;
    if (n === 1) return this.seBead;
    if (n === 2) return this.peBead;
    if (n === 3) return this.tote;
    if (n === 4) return this.tPrim;
    if (n === 5) return this.tVir;
    if (n === 6) return this.tCv;
    if (n === 7) return this.pPrim;
    if (n === 8) return this.pMd;
    if (n === 9) return this.pCv;
    if (this.pstatFlag) {
      const volume = volumeOf(this.sys);
      const u = this.sys.state.units;
      if (this.pstyle === 'iso') {
        if (n === 10) return this.vw[0];
        if (this.barostat === 'BZP') { if (n === 11) return 0.5 * this.W * this.vw[0] * this.vw[0]; }
        else if (n === 11) return 1.5 * this.W * this.vw[0] * this.vw[0];
        if (n === 12) return this.np * this.Pext * volume / u.nktv2p;
        if (n === 13) return -this.Vcoeff * this.np * this.kt * Math.log(volume);
        if (n === 14) return this.totenthalpy;
      } else {
        if (n === 10) return this.vw[0];
        if (n === 11) return this.vw[1];
        if (n === 12) return this.vw[2];
        if (n === 13) return 0.5 * this.W * (this.vw[0] * this.vw[0] + this.vw[1] * this.vw[1] + this.vw[2] * this.vw[2]);
        if (n === 14) return this.np * this.Pext * volume / u.nktv2p;
        if (n === 15) return -this.Vcoeff * this.np * this.kt * Math.log(volume);
        if (n === 16) return this.totenthalpy;
      }
    }
    return 0.0;
  }
}
