/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/COLVARS/fix_colvars.cpp and fix_colvars.h
 * (FixColvars; contributing author Axel Kohlmeyer, Temple U), with the unit table of src/update.cpp. LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE). The Colvars library it drives, with the library's LAMMPS proxy, is
 * ported in module.ts and config.ts under the library's licence, the LGPL-3.0 (src/engine-gpl/README.md).
 *
 * fix ID group-ID colvars configfile keyword value ... (docs.lammps.org/fix_colvars.html): "This fix interfaces
 * LAMMPS to the collective variables (Colvars) library, which allows to calculate potentials of mean force (PMFs) for
 * any set of colvars, using different sampling methods". Keywords input, output (default "out"), seed, unwrap
 * (default yes) and tstat. fix_modify ID config "string" / configfile file add configuration (as Colvars script
 * commands; their words go through LAMMPS variable substitution first). The scalar is the bias energy (extensive,
 * fix_modify energy adds it to the potential energy, off by default); the array f_ID[i][j] is element j of colvar i.
 * The colvars, components and biases this port computes are listed in module.ts. Parity: oracle cases
 * tests/oracle/w762cv_*.in.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { massOf } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { ColvarsModule } from './module';

/** force->femtosecond of each unit style (src/update.cpp Update::set_units at b332f28). */
/** force->angstrom of each unit style (src/update.cpp Update::set_units at b332f28). */
const ANGSTROM: Record<string, number> = { lj: 1.0, real: 1.0, metal: 1.0, si: 1.0e-10, cgs: 1.0e-8, electron: 1.88972612, micro: 1.0e-4, nano: 1.0e-1 };

const FEMTOSECOND: Record<string, number> = { lj: 1.0, real: 1.0, metal: 1.0e-3, si: 1.0e-15, cgs: 1.0e-15, electron: 1.0, micro: 1.0e-9, nano: 1.0e-6 };

const FIX_KEYWORDS = new Set(['input', 'output', 'seed', 'unwrap', 'tstat']);
const COMMON_KEYWORDS = new Set(['dynamic/dof', 'energy', 'virial', 'respa']);

const logical = (s: string): boolean => {
  const t = s.toLowerCase();
  if (t === 'yes' || t === 'on' || t === 'true' || s === '1') return true;
  if (t === 'no' || t === 'off' || t === 'false' || s === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${s}' in input script or data file`);
};

export class FixColvars extends Fix {
  readonly style = 'colvars';
  private readonly module: ColvarsModule;
  private inpName: string | null = null;
  private outName: string | null = null;
  private unwrapFlag = true;
  private energyValue = 0.0;
  /** The atoms of the proxy (init_taglist) and whether setup has run (min_post_force needs it). */
  private taglist: number[] = [];
  private setUp = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal fix colvars command: missing argument(s)');
    if (sys.fixes.some((f) => f.style === 'colvars' && f.id !== id)) throw new StyleError('Only one fix colvars instance may be active at a time');
    this.scalarFlag = true;
    this.extscalar = 1;
    this.arrayFlag = true;
    this.sizeArrayRowsVariable = true;
    this.sizeArrayCols = 4;
    this.extarray = 0;
    this.energyGlobal = true;
    const confFile = args[0] === 'none' ? null : args[0];
    this.parseFixArguments(args.slice(1), true);
    if (this.outName === null) this.outName = 'out';
    const host = {
      readFile: (name: string): string | undefined => (sys.files.has(sys.resolveFile(name)) ? sys.readFile(name) : undefined),
      writeFile: (name: string, text: string, append: boolean): void => sys.writeFile(name, text, append),
      exists: (name: string): boolean => sys.files.has(sys.resolveFile(name)),
      rename: (from: string, to: string): void => {
        const key = sys.resolveFile(from);
        const text = sys.files.get(key);
        if (text === undefined) return;
        sys.files.delete(key);
        sys.writeFile(to, text, false);
      },
      log: (text: string): void => sys.log(text),
    };
    this.module = new ColvarsModule(host);
    // colvarproxy_lammps::init: angstrom_value_ = force->angstrom
    this.module.angstrom = ANGSTROM[sys.units.style] ?? 1.0;
    this.module.initFromEngine(sys.state.step, this.dtFs());
    // colvarproxy_lammps::init: boltzmann_ = force->boltz; the constructor: proxy->set_target_temperature(t_target)
    this.module.boltzmann = sys.state.units.boltz;
    this.module.targetTemperature = this.tTarget;
    if (confFile !== null) this.module.addConfig('configfile', confFile);
  }

  /** t_target of the tstat fix (0 without one). */
  private tTarget = 0.0;

  /** The timestep in femtoseconds (update->dt * force->femtosecond), as the proxy takes it. */
  private dtFs(): number {
    return this.sys.state.dt * (FEMTOSECOND[this.sys.units.style] ?? 1.0);
  }

  /** FixColvars::parse_fix_arguments: keyword value pairs; returns the words read, or -1 for a script command. */
  private parseFixArguments(a: string[], constructor: boolean): number {
    let iarg = 0;
    while (iarg < a.length) {
      const key = a[iarg];
      if (!FIX_KEYWORDS.has(key)) {
        if (constructor) throw new StyleError('Unrecognized fix colvars argument: please note that Colvars script commands are not allowed until after the fix is created');
        if (iarg > 0) throw new StyleError('Unrecognized fix colvars argument: please note that you cannot combine fix colvars keywords and Colvars script commands in the same line');
        return -1;
      }
      if (iarg + 1 === a.length) {
        if (constructor) throw new StyleError(`Missing argument to keyword ${key}`);
        return 0;
      }
      const v = a[iarg + 1];
      if (key === 'input') this.inpName = v;
      else if (key === 'output') this.outName = v;
      else if (key === 'seed') {
        if (!/^[-+]?\d+$/.test(v)) throw new StyleError(`Expected integer parameter instead of '${v}' in input script or data file`);
      } else if (key === 'unwrap') this.unwrapFlag = logical(v);
      else if (key === 'tstat') {
        // FixColvars::set_thermostat_temperature: t_target of the fix (its extract("t_target"))
        if (v !== 'NULL') {
          const f = this.sys.fixes.find((x) => x.id === v);
          if (!f) throw new StyleError(`Could not find thermostat fix ID ${v}`);
          const tt = (f as { extract?: (name: string) => unknown }).extract?.('t_target');
          if (typeof tt !== 'number') throw new StyleError(`Fix ID ${v} is not a thermostat fix`);
          this.tTarget = tt;
        }
      }
      iarg += 2;
    }
    return iarg;
  }

  /** FixColvars::setup_colvars: parse the queued configuration and size the array. */
  private setupColvars(): void {
    this.module.parseModuleConfig();
    this.sizeArrayRows = this.module.colvars.length;
    this.sizeArrayCols = this.module.arrayCols();
  }

  /**
   * FixColvars::modify_param. The common keywords of fix_modify come first (Fix::modify_params); then fix colvars
   * keywords, or a Colvars script command whose words are substituted as LAMMPS input (Input::substitute).
   */
  modify(key: string, values: string[]): number {
    if (COMMON_KEYWORDS.has(key)) return super.modify(key, values);
    const words = [key, ...values];
    if (words.length >= 100) throw new StyleError('Too many arguments for fix_modify command');
    const rc = this.parseFixArguments(words, false);
    if (rc >= 0) {
      this.setupColvars();
      return rc - 1;
    }
    const subst = words.map((w) => (w.includes('$') ? this.sys.substituteText(w) : w));
    this.module.runScript(subst);
    this.setupColvars();
    return values.length;
  }

  /** FixColvars::setup: set up the input and output, the configuration, the atoms; then post_force. */
  setup(): void {
    const m = this.module;
    // setup_io
    m.inputPrefix = this.inpName ?? '';
    if (m.inputPrefix !== '') this.sys.log(`Will read input state from file "${m.inputPrefix}.colvars.state"`);
    m.pendingOutputPrefix = this.outName ?? '';
    this.setupColvars();
    // init_taglist, then the positions and masses
    if (m.atoms.modified) { this.taglist = [...m.atoms.ids]; m.atoms.modified = false; }
    this.readAtoms(true);
    m.setup(this.dtFs());
    this.setUp = true;
    this.postForce();
  }

  /** The unwrapped (or wrapped) positions of the proxy's atoms, and their masses at setup. */
  private readAtoms(masses: boolean): void {
    const s = this.sys.state;
    const m = this.module.atoms;
    const p = [0, 0, 0];
    for (let i = 0; i < this.taglist.length; i++) {
      const k = this.sys.indexOfId(this.taglist[i]);
      if (k < 0 || k >= s.n) throw new StyleError(`fix colvars: atom ID ${this.taglist[i]} of a colvars atom group does not exist`);
      if (this.unwrapFlag) this.sys.geom.unwrap(s.x, s.image, k, p);
      else { p[0] = s.x[3 * k]; p[1] = s.x[3 * k + 1]; p[2] = s.x[3 * k + 2]; }
      m.pos[3 * i] = p[0]; m.pos[3 * i + 1] = p[1]; m.pos[3 * i + 2] = p[2];
      if (masses) m.mass[i] = massOf(s, k);
    }
  }

  /** FixColvars::post_force: positions to the module, its forces back to the atoms. */
  postForce(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    this.readAtoms(false);
    this.energyValue = this.module.compute(s.step, [g.periodic[0], g.periodic[1], g.periodic[2]], [g.lx, 0.0, 0.0], [g.xy, g.ly, 0.0], [g.xz, g.yz, g.lz]);
    const fo = this.module.atoms.force;
    for (let i = 0; i < this.taglist.length; i++) {
      const k = this.sys.indexOfId(this.taglist[i]);
      if (k < 0 || k >= s.n) continue;
      s.f[3 * k] += fo[3 * i];
      s.f[3 * k + 1] += fo[3 * i + 1];
      s.f[3 * k + 2] += fo[3 * i + 2];
    }
  }

  /**
   * FixColvars::min_post_force. Native has no min_setup, so a minimization before any run computes the colvars from
   * atoms it never read; the engine refuses that case.
   */
  minPostForce(): void {
    if (!this.setUp) throw new StyleError('fix colvars: a minimization before the first run is not supported by the browser engine (native has no min_setup for fix colvars and reads no atoms); run 0 first');
    this.postForce();
  }

  postRun(): void {
    this.module.postRun();
  }

  energy(): number { return this.energyValue; }

  computeScalar(): number { return this.energyValue; }

  /** FixColvars::compute_array: element n of colvar m. */
  computeArray(m: number, n: number): number {
    const vars = this.module.colvars;
    if (m >= vars.length) throw new StyleError(`f_${this.id}[${m + 1}][${n + 1}] out-of-bounds: ${vars.length} collective variables available.`);
    const v = vars[m];
    if (n >= v.x.length) throw new StyleError(`f_${this.id}[${m + 1}][${n + 1}] out-of-bounds: collective variable ${v.name} has size ${v.x.length}.`);
    return v.x[n];
  }
}
