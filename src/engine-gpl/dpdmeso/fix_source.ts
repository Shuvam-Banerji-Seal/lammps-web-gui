/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-MESO/fix_edpd_source.cpp and
 * fix_edpd_source.h (FixEDPDSource: fix edpd/source) and src/DPD-MESO/fix_tdpd_source.cpp and fix_tdpd_source.h
 * (FixTDPDSource: fix tdpd/source). Both add a source term to the heat flux (edpd_flux, scaled by the heat capacity
 * edpd_cv) or to a chemical concentration flux (cc_flux) of the atoms inside a region. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Forms (docs.lammps.org/fix_edpd_source.html, fix_tdpd_source.html): "region" (the atoms in a region), and the deprecated
 * "cuboid" and "sphere" forms of the source code (the cuboid is centred at (cx, cy, cz) with edges dLx, dLy, dLz; the sphere
 * test compares the squared distance with radius^3, as the source does). The source is applied in post_force, after the
 * pair style has zeroed and filled the fluxes.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { inumeric, parseNum } from '../../engine/force/util';
import { EDPD_CV, EDPD_FLUX, TDPD_CC_FLUX, isEdpdStyle, isTdpdStyle, tdpdSpecies } from '../../engine/atoms';
import type { System } from '../../engine/system';

type SourceForm = 'sphere' | 'cuboid' | 'region';

/** The geometry and value of one source (the shared part of the edpd and tdpd sources). */
abstract class FixSourceBase extends Fix {
  protected form: SourceForm = 'region';
  protected center: [number, number, number] = [0, 0, 0];
  protected radius = 0;
  protected dL: [number, number, number] = [0, 0, 0];
  protected value = 0;
  protected regionId = '';

  /** Parses the form keyword and its values from args[k..] (the number of values of each form is fixed). */
  protected parseForm(args: string[], k: number, cmd: string): void {
    const form = args[k];
    // As native LAMMPS does (src/DPD-MESO/fix_edpd_source.cpp and fix_tdpd_source.cpp at b332f28): the message quotes a
    // fixed count per form whatever the argument count (edpd/source: sphere 4, cuboid 6, region 2; tdpd/source: sphere 5,
    // cuboid 7, region 2)
    const nativeCount: Record<string, number> = cmd === 'tdpd/source' ? { sphere: 5, cuboid: 7, region: 2 } : { sphere: 4, cuboid: 6, region: 2 };
    const want = (n: number) => {
      if (args.length !== k + n) throw new StyleError(`Illegal fix ${cmd} command (${nativeCount[form!]} args for ${form})`);
    };
    if (form === 'sphere') {
      want(6);
      this.center = [parseNum(args[k + 1], 'x'), parseNum(args[k + 2], 'y'), parseNum(args[k + 3], 'z')];
      this.radius = parseNum(args[k + 4], 'radius');
      this.value = parseNum(args[k + 5], 'value');
      this.form = 'sphere';
      this.sys.warn('WARNING: The sphere keyword is deprecated and will be removed in the future.');
    } else if (form === 'cuboid') {
      want(8);
      this.center = [parseNum(args[k + 1], 'x'), parseNum(args[k + 2], 'y'), parseNum(args[k + 3], 'z')];
      this.dL = [parseNum(args[k + 4], 'dLx'), parseNum(args[k + 5], 'dLy'), parseNum(args[k + 6], 'dLz')];
      this.value = parseNum(args[k + 7], 'value');
      this.form = 'cuboid';
      this.sys.warn('WARNING: The cuboid keyword is deprecated and will be removed in the future.');
    } else if (form === 'region') {
      want(3);
      this.regionId = args[k + 1];
      this.value = parseNum(args[k + 2], 'value');
      this.form = 'region';
      this.sys.region(this.regionId, `Region ${this.regionId} for fix ${this.style} does not exist`); // fix_edpd_source.cpp, fix_tdpd_source.cpp
    } else throw new StyleError(`Illegal fix ${cmd} command option ${form ?? ''}`);
  }

  /** True if atom i is inside the source (the form of the source decides). */
  protected inside(i: number): boolean {
    const s = this.sys.state;
    const x = s.x[3 * i], y = s.x[3 * i + 1], z = s.x[3 * i + 2];
    if (this.form === 'region') return this.sys.region(this.regionId).match(x, y, z);
    const drx = x - this.center[0], dry = y - this.center[1], drz = z - this.center[2];
    if (this.form === 'sphere') {
      // fix_edpd_source.cpp: "double radius_sq = radius * radius * radius;" then "rsq < radius_sq"
      const radiusSq = this.radius * this.radius * this.radius;
      return drx * drx + dry * dry + drz * drz < radiusSq;
    }
    return Math.abs(drx) <= 0.5 * this.dL[0] && Math.abs(dry) <= 0.5 * this.dL[1] && Math.abs(drz) <= 0.5 * this.dL[2];
  }
}

/** fix edpd/source (docs.lammps.org/fix_edpd_source.html): edpd_flux += value * edpd_cv inside the source. */
export class FixEDPDSource extends FixSourceBase {
  readonly style = 'edpd/source';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError('Illegal fix edpd/source command');
    this.parseForm(args, 0, 'edpd/source');
  }

  init(): void {
    if (!isEdpdStyle(this.sys.state.atomStyle)) throw new StyleError('Fix edpd/source requires atom style edpd');
  }

  /** fix_edpd_source.cpp has no setup(): the source is not applied to the setup force evaluation (native Fix::setup is empty). */
  setup(): void { /* no source at setup */ }

  postForce(): void {
    const s = this.sys.state;
    const flux = s.custom.get(EDPD_FLUX)!.data, cv = s.custom.get(EDPD_CV)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      if (this.inside(i)) flux[i] += this.value * cv[i];
    }
  }
}

/** fix tdpd/source (docs.lammps.org/fix_tdpd_source.html): cc_flux of species cc_index += value inside the source. */
export class FixTDPDSource extends FixSourceBase {
  readonly style = 'tdpd/source';
  private readonly ccIndex: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_tdpd_source.cpp: the species index (utils::inumeric) before the form
    const idx = inumeric(args[0]);
    if (args.length < 2) throw new StyleError('Illegal fix tdpd/source command');
    const nsp = tdpdSpecies(sys.state.atomStyle);
    if (!nsp) throw new StyleError('Fix tdpd/source requires atom style tdpd');
    // no range check (as the constructor does): native adds to cc_flux[i][cc_index-1] outside the species when the index
    // is out of range, which the engine refuses when the source is applied
    this.ccIndex = idx;
    this.parseForm(args, 1, 'tdpd/source');
  }

  /** fix_tdpd_source.cpp has no setup(): no source at the setup force evaluation. */
  setup(): void { /* no source at setup */ }

  init(): void {
    if (!isTdpdStyle(this.sys.state.atomStyle)) throw new StyleError('Fix tdpd/source requires atom style tdpd');
  }

  postForce(): void {
    const s = this.sys.state;
    const nsp = tdpdSpecies(s.atomStyle);
    if (this.ccIndex < 1 || this.ccIndex > nsp) throw new StyleError(`fix tdpd/source: cc index ${this.ccIndex} is not a species 1..${nsp} (native LAMMPS writes outside its array)`);
    const flux = s.custom.get(TDPD_CC_FLUX)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      if (this.inside(i)) flux[nsp * i + this.ccIndex - 1] += this.value;
    }
  }
}
