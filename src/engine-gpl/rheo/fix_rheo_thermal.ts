/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/RHEO/fix_rheo_thermal.cpp and fix_rheo_thermal.h (FixRHEOThermal: fix rheo/thermal, the energy
 * integration, the temperature, the melting and freezing and the reactive bond creation and breaking), at commit
 * b332f28. Contributing author (from fix_rheo_thermal.cpp): Joel Clemmer (SNL). Copyright Sandia Corporation (GNU
 * General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_rheo_thermal.html): "fix ID group-ID rheo/thermal keyword values ..." with the keywords
 * "conductivity", "specific/heat", "Tfreeze", "latent/heat" (each "type-range constant value") and "react cut btype".
 * Only the constant styles exist (as in native, where constant is the only option of each keyword).
 *
 * Time integration (native order): initial_integrate adds dt vshift . grad(e) to the energy of the shifted atoms;
 * post_integrate adds dt/2 heatflow to the energy, sets the temperature and the melting or freezing status, and
 * creates or breaks the bonds of the reactive particles; pre_force sets the temperature from the energy;
 * final_integrate adds dt/2 heatflow again. The conductivity is set after each neighbor build (post_neighbor).
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { StyleError, typeBounds } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import {
  filterTopo, isRheoThermalStyle, massOf, pushTopo, RHEO_COND, RHEO_ESPH, RHEO_HEATFLOW, RHEO_STATUS, RHEO_TEMP,
} from '../../engine/atoms';
import { NEIGHMASK } from '../../engine/neighbor';
import { BondedHybrid } from '../../engine/force/bonded_hybrid';
import {
  PHASEMASK, rheoModelOf, rheoProp, STATUS_FREEZING, STATUS_MELTING, STATUS_NO_INTEGRATION, STATUS_NO_SHIFT, STATUS_SOLID,
  type RheoModel,
} from './model';

/** fix_rheo_thermal.cpp: NONE and CONSTANT style codes of each keyword. */
const NONE = 0, CONSTANT = 1;

export class FixRHEOThermal extends Fix {
  readonly style: string = 'rheo/thermal';
  private n = 0;
  private cv: Float64Array;
  private Tc: Float64Array;
  private kappa: Float64Array;
  private L: Float64Array;
  private cvStyle: Uint8Array;
  private TcStyle: Uint8Array;
  private kappaStyle: Uint8Array;
  private LStyle: Uint8Array;
  /** react cut btype: the bond length of created bonds (0 = no reactive bonds) and the bond type. */
  private cutBond = 0;
  private cutsqBond = 0;
  private btype = 0;
  private dt = 0;
  private dth = 0;
  /** Step on which the neighbor lists must be rebuilt after bonds changed (native next_reneighbor). */
  private reneighborStep = -1;
  private model: RheoModel | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_rheo_thermal.cpp: "fix rheo/thermal command requires group all"
    if (group !== 'all') throw new StyleError('fix rheo/thermal command requires group all');
    const n = sys.state.ntypes;
    this.n = n;
    this.cv = new Float64Array(n + 1);
    this.Tc = new Float64Array(n + 1);
    this.kappa = new Float64Array(n + 1);
    this.L = new Float64Array(n + 1);
    this.cvStyle = new Uint8Array(n + 1);
    this.TcStyle = new Uint8Array(n + 1);
    this.kappaStyle = new Uint8Array(n + 1);
    this.LStyle = new Uint8Array(n + 1);
    if (args.length < 1) throw new StyleError('Illegal fix command');
    let iarg = 0;
    const need = (k: number, what: string) => {
      if (iarg + k >= args.length) throw new StyleError(`Illegal fix ${what}: missing arguments`);
    };
    while (iarg < args.length) {
      const w = args[iarg];
      if (w === 'conductivity' || w === 'specific/heat' || w === 'Tfreeze' || w === 'latent/heat') {
        need(3, `rheo/thermal ${w}`);
        const [lo, hi] = typeBounds(args[iarg + 1], n);
        if (args[iarg + 2] !== 'constant') throw new StyleError(`Illegal fix command, ${args[iarg + 2]}`);
        need(3, `rheo/thermal ${w} constant`);
        const val = parseNum(args[iarg + 3], w);
        if (w === 'conductivity') {
          if (val < 0.0) throw new StyleError('The conductivity must be positive');
          for (let t = lo; t <= hi; t++) { this.kappaStyle[t] = CONSTANT; this.kappa[t] = val; }
        } else if (w === 'specific/heat') {
          if (val <= 0.0) throw new StyleError('The specific heat must be greater than zero');
          for (let t = lo; t <= hi; t++) { this.cvStyle[t] = CONSTANT; this.cv[t] = val; }
        } else if (w === 'Tfreeze') {
          for (let t = lo; t <= hi; t++) { this.TcStyle[t] = CONSTANT; this.Tc[t] = val; }
        } else {
          if (val < 0.0) throw new StyleError('The latent heat must be positive');
          for (let t = lo; t <= hi; t++) { this.LStyle[t] = CONSTANT; this.L[t] = val; }
        }
        iarg += 4;
      } else if (w === 'react') {
        need(2, 'rheo/thermal react');
        this.cutBond = parseNum(args[iarg + 1], 'max bond length');
        this.btype = parseInt(args[iarg + 2], 10);
        if (!(this.cutBond > 0.0)) throw new StyleError('Illegal max bond length must be greater than zero');
        if (!(this.btype >= 1 && this.btype <= sys.state.topo.nbondtypes)) throw new StyleError(`Illegal value ${args[iarg + 2]} for bond type`);
        this.cutsqBond = this.cutBond * this.cutBond;
        iarg += 3;
      } else {
        throw new StyleError(`Unknown fix rheo/thermal keyword: ${w}`);
      }
    }
    for (let t = 1; t <= n; t++) {
      if (!this.cvStyle[t]) throw new StyleError(`Must specify specific/heat for atom type ${t} in fix/rheo/thermal`);
      if (!this.kappaStyle[t]) throw new StyleError(`Must specify conductivity for atom type ${t} in fix/rheo/thermal`);
      if (!this.TcStyle[t] && this.LStyle[t]) {
        throw new StyleError(`Must specify critical temperature for atom type ${t} to use latent heat in fix rheo/thermal`);
      }
    }
    this.resetDt();
  }

  /** fix_rheo_thermal.cpp init(): the checks against fix rheo and the atom style, and the time step. */
  init(): void {
    const s = this.sys.state;
    const m = rheoModelOf(s);
    // fix_rheo_thermal.cpp: "Need to define fix rheo to use fix rheo/thermal"
    if (!m) throw new StyleError('Need to define fix rheo to use fix rheo/thermal');
    if (this.cutBond > m.cut) throw new StyleError('Bonding length exceeds kernel cutoff');
    // fix_rheo_thermal.cpp: "Need to define thermal setting in fix rheo"
    if (!m.thermalFlag) throw new StyleError('Need to define thermal setting in fix rheo');
    if (!isRheoThermalStyle(s.atomStyle)) throw new StyleError('fix rheo/thermal command requires atom property esph');
    if (this.cutBond > 0.0) {
      // fix_rheo_thermal.cpp: "Must define a bond style to use reactive bond generation with fix rheo/thermal"
      if (!this.sys.ff.bond) throw new StyleError('Must define a bond style to use reactive bond generation with fix rheo/thermal');
      // special weights other than 1.0 need fix update/special/bonds (the BPM bond style provides the update, as native
      // LAMMPS does, and the engine rebuilds the special lists from the topology after each change)
      const sp = this.sys.ff.special.lj;
      if (sp[0] !== 1.0 || sp[1] !== 1.0 || sp[2] !== 1.0) {
        const bond = this.sys.ff.bond;
        const bpm = (bond.name ?? '').startsWith('bpm') || (bond instanceof BondedHybrid && bond.subNames.some((n) => n.startsWith('bpm')));
        if (!bpm) {
          throw new StyleError('Without fix update/special/bonds, reactive bond generation in fix rheo/thermal requires special weights of 1.0');
        }
      }
    }
    m.thermal = this;
    this.model = m;
    this.resetDt();
  }

  resetDt(): void {
    this.dt = this.sys.state.dt;
    this.dth = 0.5 * this.sys.state.dt;
  }

  /** fix_rheo_thermal.cpp calc_cv(): the specific heat of a type (RheoThermal, read by pair rheo and compute rheo/property/atom). */
  calcCv(itype: number): number {
    return this.cvStyle[itype] === CONSTANT ? this.cv[itype] : 0.0;
  }

  /** fix_rheo_thermal.cpp setup_pre_force(): one fix rheo/thermal, then post_neighbor and pre_force. */
  setupPreForce(): void {
    if (this.sys.fixes.filter((f) => f.style === 'rheo/thermal').length > 1) {
      throw new StyleError('More than one fix rheo/thermal defined');
    }
    this.postNeighbor();
    this.preForce();
  }

  /** fix_rheo_thermal.cpp initial_integrate(): the energy change from shifting (only with fix rheo shift). */
  initialIntegrate(): void {
    const m = this.model;
    if (!m || !m.shiftFlag) return;
    const s = this.sys.state;
    const status = rheoProp(s, RHEO_STATUS), energy = rheoProp(s, RHEO_ESPH);
    const vs = m.vshift, grade = m.grade;
    const dim = m.dim;
    for (let i = 0; i < s.n; i++) {
      if (status[i] & STATUS_NO_SHIFT) continue;
      for (let a = 0; a < dim; a++) energy[i] += this.dt * vs[i * dim + a] * grade[i * dim + a];
    }
  }

  /** fix_rheo_thermal.cpp post_integrate(): energy, temperature, melting and freezing, then the reactive bonds. */
  postIntegrate(): void {
    const s = this.sys.state;
    const status = rheoProp(s, RHEO_STATUS), energy = rheoProp(s, RHEO_ESPH);
    const temp = rheoProp(s, RHEO_TEMP), heatflow = rheoProp(s, RHEO_HEATFLOW);
    let nMelt = 0, nFreeze = 0;
    for (let i = 0; i < s.n; i++) {
      if (status[i] & STATUS_NO_INTEGRATION) continue;
      const itype = s.type[i];
      const imass = massOf(s, i);
      const cvi = this.calcCv(itype);
      energy[i] += this.dth * heatflow[i];
      temp[i] = energy[i] / (imass * cvi);
      if (this.TcStyle[itype] !== NONE) {
        let Ti = temp[i];
        const Tci = this.Tc[itype];
        if (this.LStyle[itype] !== NONE) {
          const Li = this.L[itype];
          // "if (Ti > Tci) Ti = MAX(Tci, (energy[i] / imass - Li) / cvi);"
          if (Ti > Tci) Ti = Math.max(Tci, (energy[i] / imass - Li) / cvi);
          temp[i] = Ti;
        }
        // status &= PHASEMASK; status |= STATUS_MELTING (solid melts), and the freezing of a fluid particle
        if (Ti > Tci && (status[i] & STATUS_SOLID)) {
          status[i] = (((status[i] & PHASEMASK) >>> 0) | STATUS_MELTING) >>> 0;
          nMelt += 1;
        }
        if (Ti < Tci && !(status[i] & STATUS_SOLID)) {
          status[i] = (((status[i] & PHASEMASK) >>> 0) | STATUS_SOLID | STATUS_FREEZING) >>> 0;
          nFreeze += 1;
        }
      }
    }
    if (this.cutBond > 0 && (nMelt || nFreeze)) this.reactBonds(nFreeze > 0, nMelt > 0);
  }

  /** fix_rheo_thermal.cpp post_neighbor(): the constant conductivity of every atom (owned and ghost, via the owner). */
  postNeighbor(): void {
    const s = this.sys.state;
    const cond = rheoProp(s, RHEO_COND);
    for (let i = 0; i < s.n; i++) {
      const itype = s.type[i];
      if (this.kappaStyle[itype] === CONSTANT) cond[i] = this.kappa[itype];
    }
  }

  /** fix_rheo_thermal.cpp pre_force(): the temperature from the energy (and the latent-heat limit of the temperature). */
  preForce(): void {
    const s = this.sys.state;
    const energy = rheoProp(s, RHEO_ESPH), temp = rheoProp(s, RHEO_TEMP);
    for (let i = 0; i < s.n; i++) {
      const itype = s.type[i];
      const imass = massOf(s, i);
      const cvi = this.calcCv(itype);
      temp[i] = energy[i] / (imass * cvi);
      if (this.TcStyle[itype] !== NONE) {
        let Ti = temp[i];
        const Tci = this.Tc[itype];
        if (this.LStyle[itype] !== NONE) {
          const Li = this.L[itype];
          if (Ti > Tci) Ti = Math.max(Tci, (energy[i] / imass - Li) / cvi);
          temp[i] = Ti;
        }
      }
    }
  }

  /** fix_rheo_thermal.cpp final_integrate(): the second half of the energy change. */
  finalIntegrate(): void {
    const s = this.sys.state;
    const status = rheoProp(s, RHEO_STATUS), energy = rheoProp(s, RHEO_ESPH), heatflow = rheoProp(s, RHEO_HEATFLOW);
    for (let i = 0; i < s.n; i++) {
      if (status[i] & STATUS_NO_INTEGRATION) continue;
      energy[i] += this.dth * heatflow[i];
    }
  }

  /** The neighbor lists are rebuilt on the step on which bonds were created or broken (native next_reneighbor). */
  reneighborDue(step: number): boolean { return step === this.reneighborStep; }

  /**
   * fix_rheo_thermal.cpp post_integrate() bond part: a freezing particle that already has a bond of the react type keeps
   * its freezing flag cleared; then bonds are created (freezing) and broken (melting). The positions of the ghosts are
   * refreshed first, as native's forward_comm does, and the special lists follow the new topology.
   */
  private reactBonds(freeze: boolean, melt: boolean): void {
    const sys = this.sys;
    const s = sys.state;
    const status = rheoProp(s, RHEO_STATUS);
    const bonds = s.topo.bonds;
    const tagWithBond = new Set<number>();
    for (let k = 0; k < bonds.n; k++) {
      if (bonds.type[k] !== this.btype) continue;
      tagWithBond.add(bonds.atoms[2 * k]);
      tagWithBond.add(bonds.atoms[2 * k + 1]);
    }
    for (let i = 0; i < s.n; i++) {
      if ((status[i] & STATUS_FREEZING) && tagWithBond.has(s.id[i])) status[i] = ((status[i] & ~STATUS_FREEZING) >>> 0);
    }
    sys.nb.forwardComm(s, sys.geom);
    if (freeze) this.createBonds();
    if (melt) this.breakBonds();
    this.reneighborStep = s.step;
  }

  /**
   * fix_rheo_thermal.cpp create_bonds(): a bond between two solid particles one of which is freezing, closer than the
   * react cut. Each physical pair is added once (native stores it on both atoms with newton off); the pairs come from the
   * full neighbor list of the owned particles, with ghost pairs taken from the lower owner.
   */
  private createBonds(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const full = nb.full;
    if (!full) throw new StyleError('fix rheo/thermal needs a full neighbor list for reactive bonds');
    const status = rheoProp(s, RHEO_STATUS);
    const x = nb.xall, owner = nb.owner, nbr = full.neighbors;
    const tag = s.id;
    const n = s.n;
    for (let i = 0; i < n; i++) {
      if (!(status[i] & STATUS_SOLID)) continue;
      const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nbr[k] & NEIGHMASK;
        const ghost = jj >= n;
        const oj = owner[jj];
        // each physical pair once: an owned partner at a higher index (its own entry of the pair is skipped), a ghost
        // partner from the lower owner (the owner of the other image skips it)
        if (ghost ? oj < i : jj <= i) continue;
        if (!(status[oj] & STATUS_SOLID)) continue;
        if (!(status[i] & STATUS_FREEZING) && !(status[oj] & STATUS_FREEZING)) continue;
        // positions with the same ordering of the pair as native (the lower tag first)
        let dx: number, dy: number, dz: number;
        if (tag[i] < tag[oj]) {
          dx = x[3 * i] - x[3 * jj]; dy = x[3 * i + 1] - x[3 * jj + 1]; dz = x[3 * i + 2] - x[3 * jj + 2];
        } else {
          dx = x[3 * jj] - x[3 * i]; dy = x[3 * jj + 1] - x[3 * i + 1]; dz = x[3 * jj + 2] - x[3 * i + 2];
        }
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq > this.cutsqBond) continue;
        pushTopo(s.topo.bonds, this.btype, [tag[i], tag[oj]]);
      }
    }
    sys.ff.topologyChanged(s, nb);
  }

  /**
   * fix_rheo_thermal.cpp break_bonds(): every bond of the react type with a melting end is deleted (native removes it from
   * both atoms' lists).
   */
  private breakBonds(): void {
    const sys = this.sys;
    const s = sys.state;
    const status = rheoProp(s, RHEO_STATUS);
    const melting = new Set<number>();
    for (let i = 0; i < s.n; i++) if (status[i] & STATUS_MELTING) melting.add(s.id[i]);
    const btype = this.btype;
    filterTopo(s.topo.bonds, (ids, type) => !(type === btype && (melting.has(ids[0]) || melting.has(ids[1]))));
    sys.ff.topologyChanged(s, sys.nb);
  }
}
