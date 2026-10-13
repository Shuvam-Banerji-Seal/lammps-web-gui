/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/RHEO/fix_rheo_oxidation.cpp and fix_rheo_oxidation.h (FixRHEOOxidation: fix rheo/oxidation), at commit
 * b332f28. Contributing author (from fix_rheo_oxidation.cpp): Joel Clemmer (SNL). Copyright Sandia Corporation (GNU
 * General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs (docs.lammps.org/fix_rheo_oxidation.html): "fix ID group-ID rheo/oxidation cut btype rsurf" with "cut = maximum
 * bond length (distance units)", "btype = type of bonds created" and "rsurf = distance from surface to create bonds
 * (distance units)". "Every timestep, particles check neighbors within a distance of cut. This distance must be smaller
 * than the kernel length defined in fix rheo. Bonds of type btype are created between a fluid particle and either a fluid
 * or solid neighbor. The fluid particles must also be on the fluid surface, or within a distance of rsurf from the
 * surface." The bonds are for bond style rheo/shell; an atom with bonds does not shift (post_force, STATUS_NO_SHIFT).
 *
 * Native adds the bond to both atoms' lists (newton off) and the neighbor lists are rebuilt on the step of creation
 * (next_reneighbor). The browser engine keeps one topology entry per pair, as its other bond creations do.
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { BondedHybrid } from '../../engine/force/bonded_hybrid';
import { parseNum } from '../../engine/force/util';
import { pushTopo, RHEO_STATUS } from '../../engine/atoms';
import { NEIGHMASK } from '../../engine/neighbor';
import { PHASECHECK, rheoModelOf, rheoProp, STATUS_NO_SHIFT, type RheoModel } from './model';
import { bondKey } from './bond_rheo_shell';

export class FixRHEOOxidation extends Fix {
  readonly style: string = 'rheo/oxidation';
  private readonly cut: number;
  private readonly btype: number;
  private readonly rsurf: number;
  private readonly model: RheoModel;
  /** Step on which the neighbor lists must be rebuilt after bonds were added (native next_reneighbor). */
  private reneighborStep = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('Illegal fix rheo/oxidation command');
    this.cut = parseNum(args[0], 'bond cutoff');
    // fix_rheo_oxidation.cpp: "Illegal bond cutoff {} in fix rheo/oxidation"
    if (!(this.cut > 0.0)) throw new StyleError(`Illegal bond cutoff ${args[0]} in fix rheo/oxidation`);
    const nbt = sys.state.topo.nbondtypes;
    this.btype = Number(args[1]);
    if (!Number.isInteger(this.btype) || this.btype < 1 || this.btype > nbt) {
      throw new StyleError(`Illegal value ${args[1]} for bond type in fix rheo/oxidation`);
    }
    this.rsurf = parseNum(args[2], 'surface distance');
    if (!(this.rsurf > 0.0)) throw new StyleError(`Illegal surface distance ${args[2]} in fix rheo/oxidation`);
    const m = rheoModelOf(sys.state);
    if (!m) throw new StyleError('Need to define fix rheo to use fix rheo/oxidation');
    this.model = m;
    // init(): "Bonding length exceeds kernel cutoff", "Surface distance must be less than kernel cutoff"
    if (this.cut > m.cut) throw new StyleError('Bonding length exceeds kernel cutoff');
    if (this.rsurf >= m.cut) throw new StyleError('Surface distance must be less than kernel cutoff');
    m.oxidation = { cut: this.cut, btype: this.btype, rsurf: this.rsurf };
  }

  init(): void {
    const bond = this.sys.ff.bond;
    // "Must define a bond style with fix rheo/oxidation" and "Must use bond style rheo/shell to use fix rheo/oxidation"
    if (!bond) throw new StyleError('Must define a bond style with fix rheo/oxidation');
    const hasShell = bond.name === 'rheo/shell' || (bond instanceof BondedHybrid && bond.subNames.includes('rheo/shell'));
    if (!hasShell) throw new StyleError('Must use bond style rheo/shell to use fix rheo/oxidation');
  }

  /** fix_rheo_oxidation.cpp setup_pre_force(): the surface detection must be on in fix rheo. */
  setupPreForce(): void {
    if (!this.model.surfaceFlag) throw new StyleError('fix rheo/oxidation requires surface calculation in fix rheo');
  }

  /** The neighbor lists are rebuilt on the step on which bonds were added. */
  reneighborDue(step: number): boolean { return step === this.reneighborStep; }

  /**
   * fix_rheo_oxidation.cpp post_integrate(): a bond between two particles of the group, closer than cut, where each of
   * them is a fluid particle within rsurf of the surface or a solid one (a solid pair is skipped; a solid with a fluid
   * particle outside rsurf is skipped too). The pair is added once, with the lower tag first for the distance.
   */
  postIntegrate(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const full = nb.full;
    if (!full) throw new StyleError('fix rheo/oxidation needs a full neighbor list');
    // positions of the ghosts (native forward comm), then the pairs
    nb.forwardComm(s, sys.geom);
    const m = this.model;
    const rs = m.rsurface;
    const status = rheoProp(s, RHEO_STATUS);
    const x = nb.xall, owner = nb.owner, nbr = full.neighbors;
    const tag = s.id, mask = s.mask;
    const bonds = s.topo.bonds;
    const have = new Set<number>();
    for (let k = 0; k < bonds.n; k++) {
      if (bonds.type[k] === this.btype) have.add(bondKey(bonds.atoms[2 * k], bonds.atoms[2 * k + 1]));
    }
    const cutsq = this.cut * this.cut;
    let added = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      // exclude particles that are not solid or surface (fluid and beyond the surface distance)
      const fluidi = !(status[i] & PHASECHECK);
      if (fluidi && rs[i] > this.rsurf) continue;
      const tagi = tag[i];
      const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nbr[k] & NEIGHMASK;
        const j = owner[jj];
        if (!(mask[j] & this.groupBit)) continue;
        const fluidj = !(status[j] & PHASECHECK);
        if (fluidj && rs[j] > this.rsurf) continue;
        // skip solid-solid, leaves surface-surface or surface-solid
        if (!fluidi && !fluidj) continue;
        const tagj = tag[j];
        // the pair is ordered by tag so the distance is computed as native does
        let dx: number, dy: number, dz: number;
        if (tagi < tagj) {
          dx = x[3 * i] - x[3 * jj]; dy = x[3 * i + 1] - x[3 * jj + 1]; dz = x[3 * i + 2] - x[3 * jj + 2];
        } else {
          dx = x[3 * jj] - x[3 * i]; dy = x[3 * jj + 1] - x[3 * i + 1]; dz = x[3 * jj + 2] - x[3 * i + 2];
        }
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq > cutsq) continue;
        // no second oxide bond between the same pair (native: the bond list of atom i)
        const key = bondKey(tagi, tagj);
        if (have.has(key)) continue;
        have.add(key);
        pushTopo(bonds, this.btype, [tagi, tagj]);
        added += 1;
      }
    }
    if (added > 0) this.reneighborStep = s.step;
  }

  /** Native fix rheo/oxidation has no setup() (base Fix::setup is empty): no post_force at the setup of a run. */
  setup(): void { /* no setup post_force, as native */ }

  /** fix_rheo_oxidation.cpp post_force(): an atom with bonds does not shift. */
  postForce(): void {
    const s = this.sys.state;
    const status = rheoProp(s, RHEO_STATUS);
    const bonds = s.topo.bonds;
    let maxId = 0;
    for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
    const hasBond = new Uint8Array(maxId + 1);
    for (let k = 0; k < bonds.n; k++) {
      hasBond[bonds.atoms[2 * k]] = 1;
      hasBond[bonds.atoms[2 * k + 1]] = 1;
    }
    for (let i = 0; i < s.n; i++) {
      if (hasBond[s.id[i]]) status[i] = (status[i] | STATUS_NO_SHIFT) >>> 0;
    }
  }
}
