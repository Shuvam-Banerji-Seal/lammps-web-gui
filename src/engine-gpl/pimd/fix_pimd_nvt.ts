/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/fix_pimd_nvt.cpp and src/REPLICA/fix_pimd_nvt.h (FixPIMDNVT: the constructor's keywords, init,
 * post_force, the Nose-Hoover chains nhc_init / nhc_update_v / nhc_update_x, the normal-mode setup nmpimd_init and
 * nmpimd_transform, the spring force and the virial estimator, compute_vector) at commit b332f28. Authors of the
 * source: Chris Knight and Yuxing Peng (Voth group, University of Chicago). Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Partitions: the beads of the path integral are the replicas of the session (System.replicaPeers). Native LAMMPS
 * exchanges the bead positions and forces with MPI between the partitions; here the replicas share a process, so the
 * fix of the first replica (the leader) does the ring operations for every bead in the phases that need all of them
 * (run/coupled.ts), and the fixes of the other replicas do nothing there. Per-atom chain state is keyed by the atom ID,
 * so atom sorting does not move it.
 *
 * Not supported (StyleError): method cmd, lj units (as native), a group other than all, an odd number of beads with
 * method nmpimd (native's eigenvalue table is only filled for an even count), a change of the atom count.
 */
import { StyleError } from '../../engine/force/types';
import { inumeric, numeric } from '../../engine/force/util';
import { fmtShortest } from '../../engine/cformat';
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { cosFma as libmCosFma } from '../libm/sincos_fma';
import { exp as libmExp } from '../libm/exp';

const MY_PI = Math.PI;

type Method = 'pimd' | 'nmpimd';

export class FixPIMDNVT extends Fix {
  readonly style: string;
  // As native LAMMPS has it (src/REPLICA/fix_pimd_nvt.cpp at b332f28): FixPIMDNVT leaves the base Fix's time_integrate = 0,
  // so a run with it alone warns that no fix integrates (Verlet::init); fix pimd/langevin sets it
  timeIntegrate = false;
  vectorFlag = true;
  sizeVector = 3;
  extvector = 1;
  replicaCoupled = true;

  private readonly method: Method;
  private readonly fmass: number;
  private readonly sp: number;
  private readonly nhcTemp: number;
  private readonly nchain: number;

  protected np = 1;
  protected iworld = 0;
  private dtv = 0;
  private dtf = 0;
  private omegaNp = 0;
  protected fbond = 0;
  /** the inverse temperature 1/(boltz T) of the chain temperature (init) */
  protected beta = 0;
  /** outputs (compute_vector): spring energy, T of the ring (t_sys), virial */
  protected springEnergy = 0;
  private tSys = 0;
  private virialValue = 0;
  /** mass per atom type of this bead (init) */
  private massT = new Float64Array(0);
  /** normal-mode tables, np x np row-major: M_x2xp, M_xp2x, M_f2fp (M_fp2f is not used) */
  private lam = new Float64Array(0);
  private mx2xp = new Float64Array(0);
  private mxp2x = new Float64Array(0);
  private mf2fp = new Float64Array(0);
  /** Nose-Hoover chains per coordinate slot (3 per atom ID): eta, eta_dot (nchain + 1), eta_dotdot, eta_mass */
  private nhcReady = false;
  private maxId = 0;
  private nhcEta = new Float64Array(0);
  private nhcEtaDot = new Float64Array(0);
  private nhcEtaDotdot = new Float64Array(0);
  private nhcEtaMass = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[], name = 'pimd/nvt') {
    super(sys, id, group, args);
    this.style = name;
    let method: Method = 'pimd';
    let fmass = 1.0, sp = 1.0, temp = 298.15, nchain = 2;
    // fix_pimd_nvt.cpp constructor: keyword/value pairs (a last word without a value is not read), messages naming
    // the style as given; the number of beads is the number of partitions
    const np = sys.replicaPeers().length;
    for (let i = 0; i + 1 < args.length; i += 2) {
      const key = args[i], val = args[i + 1];
      if (key === 'method') {
        if (val === 'pimd') method = 'pimd';
        else if (val === 'nmpimd') method = 'nmpimd';
        else if (val === 'cmd') throw new StyleError(`fix ${name} method cmd is not supported (only pimd and nmpimd)`);
        else throw new StyleError(`Unknown method parameter ${val} for fix ${name}`);
      } else if (key === 'fmass') {
        fmass = numeric(val);
        if (fmass < 0 || fmass > np) throw new StyleError(`Invalid fmass value ${fmtShortest(fmass, false)} for fix ${name}`);
      } else if (key === 'sp') {
        sp = numeric(val);
        if (sp < 0) throw new StyleError(`Invalid sp value for fix ${name}`);
      } else if (key === 'temp') {
        temp = numeric(val);
        if (temp < 0) throw new StyleError(`Invalid temp value for fix ${name}`);
      } else if (key === 'nhc') {
        nchain = inumeric(val);
        if (nchain < 2) throw new StyleError(`Invalid nhc value for fix ${name}`);
      } else {
        throw new StyleError(`Unknown keyword ${key} for fix ${name}`);
      }
    }
    if (sys.state.units.style === 'lj') throw new StyleError(`Fix ${name} does not support lj units`);
    if (np < 2) throw new StyleError(`Fix ${name} requires at least 2 beads, i.e. running with the -partition command line option`);
    if (group !== 'all') throw new StyleError(`fix ${name} acts on all atoms: its group must be all (native LAMMPS ignores the group; not supported here)`);
    this.method = method;
    this.fmass = fmass;
    this.sp = sp;
    this.nhcTemp = temp;
    this.nchain = nchain;
  }

  protected get isLeader(): boolean { return this.sys.replicaPeers()[0] === this.sys; }

  /** The FixPIMDNVT of every replica (bead order), this fix's own first. */
  protected ring(): FixPIMDNVT[] {
    return this.sys.replicaPeers().map((p) => {
      const f = p.fixes.find((x) => x.id === this.id);
      if (!(f instanceof FixPIMDNVT)) throw new StyleError(`fix ${this.id} is not defined on every replica`);
      return f;
    });
  }

  init(): void {
    const s = this.sys.state;
    const peers = this.sys.replicaPeers();
    this.np = peers.length;
    this.iworld = peers.indexOf(this.sys);
    if (this.method === 'nmpimd' && this.np % 2 !== 0) throw new StyleError('fix pimd/nvt method nmpimd needs an even number of beads (native eigenvalue table)');
    const u = s.units;
    // native: the constants are force->boltz, force->hplanck and force->mvv2e of the unit style
    const hbar = u.hplanck / (2.0 * MY_PI) * this.sp;
    this.beta = 1.0 / (u.boltz * this.nhcTemp);
    const beta = this.beta;
    const fb = (1.0 * this.np) / (beta * beta * hbar * hbar);
    this.omegaNp = Math.sqrt(this.np) / (hbar * beta) * Math.sqrt(u.mvv2e);
    this.fbond = -fb * u.mvv2e;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * u.ftm2v;
    this.massT = new Float64Array(s.ntypes + 1);
    if (this.method === 'nmpimd') {
      this.nmpimdInit();
      for (let t = 1; t <= s.ntypes; t++) {
        this.massT[t] = s.massByType[t];
        if (this.iworld) {
          this.massT[t] *= this.lam[this.iworld];
          this.massT[t] *= this.fmass;
        }
      }
    } else {
      const inverseNp = 1.0 / this.np;
      for (let t = 1; t <= s.ntypes; t++) this.massT[t] = s.massByType[t] * inverseNp * this.fmass;
    }
    if (!this.nhcReady) this.nhcInit();
  }

  /** nhc_init: the chain masses from omega_np and the chain temperature; the chains start at rest. */
  private nhcInit(): void {
    const s = this.sys.state;
    const u = s.units;
    const tau = 1.0 / this.omegaNp;
    const KT = u.boltz * this.nhcTemp;
    const mass0 = KT * tau * tau;
    this.maxId = 0;
    for (let i = 0; i < s.n; i++) if (s.id[i] > this.maxId) this.maxId = s.id[i];
    const slots = 3 * this.maxId;
    const nc = this.nchain;
    this.nhcEta = new Float64Array(slots * nc);
    this.nhcEtaDot = new Float64Array(slots * (nc + 1));
    this.nhcEtaDotdot = new Float64Array(slots * nc);
    this.nhcEtaMass = new Float64Array(slots * nc);
    const scaleMass = !(this.method === 'nmpimd' && this.iworld === 0);
    for (let i = 0; i < s.n; i++) {
      for (let d = 0; d < 3; d++) {
        const slot = 3 * (s.id[i] - 1) + d;
        for (let ich = 0; ich < nc; ich++) {
          this.nhcEtaMass[slot * nc + ich] = scaleMass ? mass0 * this.fmass : mass0;
        }
        this.nhcEtaDot[slot * (nc + 1) + nc] = 0.0;
        for (let ich = 1; ich < nc; ich++) {
          // the chains start at rest, so only -KT remains of (mass * eta_dot^2 * mvv2e - KT)
          this.nhcEtaDotdot[slot * nc + ich] = -KT / this.nhcEtaMass[slot * nc + ich];
        }
      }
    }
    this.nhcReady = true;
  }

  /** nmpimd_init: the eigenvalues and the transform matrices of the ring polymer (the normal modes of the beads). */
  private nmpimdInit(): void {
    const np = this.np;
    const inverseNp = 1.0 / np;
    this.lam = new Float64Array(np);
    this.mx2xp = new Float64Array(np * np);
    this.mxp2x = new Float64Array(np * np);
    this.mf2fp = new Float64Array(np * np);
    // row-major index of (i, j) in an np x np table
    const M = (i: number, j: number) => i * np + j;
    this.lam[0] = 0.0;
    if (np % 2 === 0) this.lam[np - 1] = 4.0 * np;
    for (let i = 2; i <= Math.floor(np / 2); i++) {
      this.lam[2 * i - 3] = this.lam[2 * i - 2] = 2.0 * np * (1.0 - 1.0 * libmCosFma(2.0 * MY_PI * (i - 1) / np));
    }
    for (let i = 0; i < np; i++) {
      this.mx2xp[M(0, i)] = inverseNp;
      if (np % 2 === 0) this.mx2xp[M(np - 1, i)] = inverseNp * ((i % 2) === 0 ? 1.0 : -1.0);
    }
    for (let i = 0; i < Math.floor((np - 1) / 2); i++) {
      for (let j = 0; j < np; j++) {
        this.mx2xp[M(2 * i + 1, j)] = Math.sqrt(2.0) * libmCos(2.0 * MY_PI * (i + 1) * j / np) / np;
        this.mx2xp[M(2 * i + 2, j)] = -Math.sqrt(2.0) * libmSin(2.0 * MY_PI * (i + 1) * j / np) / np;
      }
    }
    for (let i = 0; i < np; i++) {
      for (let j = 0; j < np; j++) {
        this.mxp2x[M(i, j)] = this.mx2xp[M(j, i)] * np;
        this.mf2fp[M(i, j)] = this.mx2xp[M(i, j)] * np;
      }
    }
  }

  /** fix_pimd_nvt.cpp setup(): the step-0 forces include the spring (post_force). */
  setup(): void { this.postForce(); }

  /** initial_integrate: the chains and the half kick of every bead, then the positions (normal modes for nmpimd). */
  initialIntegrate(): void {
    if (!this.isLeader) return;
    const ring = this.ring();
    for (const r of ring) r.nhcUpdateV();
    if (this.method === 'nmpimd') {
      const vt = ring.map((r) => r.tagArray('v'));
      for (const r of ring) r.nhcUpdateX(vt);
    } else {
      for (const r of ring) r.nhcUpdateX(null);
    }
  }

  /** final_integrate: the second half kick of the chains. */
  finalIntegrate(): void {
    if (!this.isLeader) return;
    for (const r of this.ring()) r.nhcUpdateV();
  }

  /**
   * post_force: the ring's spring force (every bead's forces are divided by np first), then the normal-mode forces.
   * The order of native post_force is kept: prepare_coordinates, pre_spring_force_estimators, spring_force, for every
   * bead (a subclass overrides the hooks, fix_pimd_nvt_bosonic.ts).
   */
  postForce(): void {
    if (!this.isLeader) return;
    const ring = this.ring();
    const np = this.np;
    for (const r of ring) {
      const f = r.sys.state.f;
      for (let k = 0; k < f.length; k++) f[k] /= np;
    }
    const xt = ring.map((r) => r.tagArray('x'));
    ring.forEach((r, k) => r.prepareCoordinates(xt[(k - 1 + np) % np], xt[(k + 1) % np]));
    for (const r of ring) r.preSpringEstimators();
    ring.forEach((r, k) => r.springForce(xt[(k - 1 + np) % np], xt[(k + 1) % np]));
    if (this.method === 'nmpimd') {
      const ft = ring.map((r) => r.tagArray('f'));
      for (const r of ring) r.nmpimdTransform(ft, r.mf2fp, r.iworld, r.sys.state.f);
    }
  }

  /** Hook (native prepare_coordinates): the neighbours' positions (by atom ID) are known; nothing to do here. */
  protected prepareCoordinates(_xPrev: Float64Array, _xNext: Float64Array): void { /* the ring spring needs no preparation */ }

  /** Hook (native pre_spring_force_estimators): the virial estimator. */
  protected preSpringEstimators(): void { this.virialEstimator(); }

  /** A per-atom array of this bead re-indexed by atom ID: slot 3 * (ID - 1) + dimension. */
  protected tagArray(which: 'x' | 'v' | 'f'): Float64Array {
    const s = this.sys.state;
    const out = new Float64Array(3 * this.maxId);
    const src = which === 'x' ? s.x : which === 'v' ? s.v : s.f;
    for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) out[3 * (s.id[i] - 1) + d] = src[3 * i + d];
    return out;
  }

  /** The minimum image of a separation along dimension d (domain->minimum_image). */
  protected minImage(dv: number, d: number): number {
    const box = this.sys.state.box;
    if (!box.periodic[d]) return dv;
    const L = box.hi[d] - box.lo[d];
    return dv - L * Math.round(dv / L);
  }

  /** pre_spring_force_estimators: the virial estimator -0.5 x.f of this bead (forces already divided by np). */
  protected virialEstimator(): void {
    const s = this.sys.state;
    let v = 0;
    for (let i = 0; i < s.n; i++) v += -0.5 * (s.x[3 * i] * s.f[3 * i] + s.x[3 * i + 1] * s.f[3 * i + 1] + s.x[3 * i + 2] * s.f[3 * i + 2]);
    this.virialValue = v;
  }

  /** spring_force: the springs to the previous and next beads (their positions by atom ID). */
  protected springForce(xlast: Float64Array, xnext: Float64Array): void {
    const s = this.sys.state;
    this.springEnergy = 0;
    for (let i = 0; i < s.n; i++) {
      const slot = 3 * (s.id[i] - 1);
      const del1 = [0, 0, 0], del2 = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        del1[d] = this.minImage(xlast[slot + d] - s.x[3 * i + d], d);
        del2[d] = this.minImage(xnext[slot + d] - s.x[3 * i + d], d);
      }
      const ff = this.fbond * s.massByType[s.type[i]];
      for (let d = 0; d < 3; d++) s.f[3 * i + d] -= (del1[d] + del2[d]) * ff;
      this.springEnergy += -0.5 * ff * (del2[0] * del2[0] + del2[1] * del2[1] + del2[2] * del2[2]);
    }
  }

  /** nmpimd_transform: this bead's forces (or velocities) from all beads' values through one row of a transform. */
  protected nmpimdTransform(src: Float64Array[], matrix: Float64Array, row: number, des: Float64Array): void {
    const s = this.sys.state;
    const np = this.np;
    for (let i = 0; i < s.n; i++) {
      for (let d = 0; d < 3; d++) {
        const slot = 3 * (s.id[i] - 1) + d;
        let acc = 0.0;
        for (let j = 0; j < np; j++) acc += src[j][slot] * matrix[row * np + j];
        des[3 * i + d] = acc;
      }
    }
  }

  /** nhc_update_x: positions move by dt times the velocities (normal-mode velocities are transformed back first). */
  private nhcUpdateX(vt: Float64Array[] | null): void {
    const s = this.sys.state;
    if (vt) {
      // nmpimd: the Cartesian velocities of this bead, into a scratch array (native borrows atom->f for them)
      const vc = new Float64Array(s.n * 3);
      this.nmpimdTransform(vt, this.mxp2x, this.iworld, vc);
      for (let k = 0; k < 3 * s.n; k++) s.x[k] += this.dtv * vc[k];
    } else {
      for (let k = 0; k < 3 * s.n; k++) s.x[k] += this.dtv * s.v[k];
    }
  }

  /** nhc_update_v: the half kick with the forces, then the Nose-Hoover chain half step (Martyna et al. 1992 Trotter form). */
  private nhcUpdateV(): void {
    const s = this.sys.state;
    const u = s.units;
    const n = s.n;
    const v = s.v, f = s.f, type = s.type, id = s.id;
    for (let i = 0; i < n; i++) {
      const dtfm = this.dtf / this.massT[type[i]];
      for (let d = 0; d < 3; d++) v[3 * i + d] += dtfm * f[3 * i + d];
    }
    this.tSys = 0.0;
    const nc = this.nchain;
    const nmax = 3 * n;
    const KT = u.boltz * this.nhcTemp;
    const dt = s.dt;
    const dthalf = 0.5 * dt;
    const dt4 = 0.25 * dt;
    const dt8 = 0.125 * dt;
    let expfac = 0.0;
    for (let i = 0; i < nmax; i++) {
      const iatm = Math.floor(i / 3);
      const idim = i % 3;
      const k = 3 * iatm + idim;
      const slot = 3 * (id[iatm] - 1) + idim;
      const mass = this.massT[type[iatm]];
      let kecurrent = mass * v[k] * v[k] * u.mvv2e;
      let tCurrent = kecurrent / u.boltz;
      const eta = this.nhcEta.subarray(slot * nc, slot * nc + nc);
      const etaDot = this.nhcEtaDot.subarray(slot * (nc + 1), slot * (nc + 1) + nc + 1);
      const etaDotdot = this.nhcEtaDotdot.subarray(slot * nc, slot * nc + nc);
      const etaMass = this.nhcEtaMass.subarray(slot * nc, slot * nc + nc);
      etaDotdot[0] = (kecurrent - KT) / etaMass[0];
      for (let ich = nc - 1; ich > 0; ich--) {
        expfac = libmExp(-dt8 * etaDot[ich + 1]);
        etaDot[ich] *= expfac;
        etaDot[ich] += etaDotdot[ich] * dt4;
        etaDot[ich] *= expfac;
      }
      expfac = libmExp(-dt8 * etaDot[1]);
      etaDot[0] *= expfac;
      etaDot[0] += etaDotdot[0] * dt4;
      etaDot[0] *= expfac;
      // update the particle velocity half step
      const factorEta = libmExp(-dthalf * etaDot[0]);
      v[k] *= factorEta;
      tCurrent *= factorEta * factorEta;
      kecurrent = u.boltz * tCurrent;
      etaDotdot[0] = (kecurrent - KT) / etaMass[0];
      for (let ich = 0; ich < nc; ich++) eta[ich] += dthalf * etaDot[ich];
      etaDot[0] *= expfac;
      etaDot[0] += etaDotdot[0] * dt4;
      etaDot[0] *= expfac;
      for (let ich = 1; ich < nc; ich++) {
        expfac = libmExp(-dt8 * etaDot[ich + 1]);
        etaDot[ich] *= expfac;
        etaDotdot[ich] = (etaMass[ich - 1] * etaDot[ich - 1] * etaDot[ich - 1] - KT) / etaMass[ich];
        etaDot[ich] += etaDotdot[ich] * dt4;
        etaDot[ich] *= expfac;
      }
      this.tSys += tCurrent;
    }
    this.tSys /= nmax;
  }

  computeVector(i: number): number {
    if (i === 0) return this.springEnergy;
    if (i === 1) return this.tSys;
    if (i === 2) return this.virialValue;
    return 0.0;
  }
}
