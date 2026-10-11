/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm.cpp and src/BPM/bond_bpm.h (the common base of the BPM bond styles: keyword
 * parsing, bond breaking, the per-bond history and the special-bond update), at commit b332f28.
 * Contributing author of the BPM package: Joel Clemmer (SNL); copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Browser-engine differences: the per-bond history (the reference state of LAMMPS' fix BOND_HISTORY) is a map
 * keyed by the two atom IDs; a broken bond gets bond type 0 and the force field rebuilds the 1-2 special list
 * (ForceField, takeTopologyChange). LAMMPS' fix update_special_bonds is that rebuild.
 */
import { SimpleBonded, atomIndex } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute } from '../../engine/force/types';

/** Key of a bond by its two atom IDs (IDs stay below 2^26 in the browser engine). */
export const bondKey = (a: number, b: number): number => (a < b ? a * 0x4000000 + b : b * 0x4000000 + a);

export const yesno = (w: string, cmd: string, key: string): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`${cmd} ${key} must be yes or no (got '${w}')`);
};

/**
 * Base class of the BPM bond styles. Subclasses set their own coefficients; this class holds the flags of the
 * common keywords and the bond-breaking hook. The doc (bond_bpm_rotational.html and bond_bpm_spring.html) says:
 * "overlay/pair value = yes or no" and "break value = yes or no"; "store/local" is refused (no fix store/local).
 */
export abstract class BondBPMBase extends SimpleBonded {
  overlay = false;
  breakOn = true;
  /** Set when a compute() call broke bonds; read and cleared by takeTopologyChange(). */
  protected changed = false;

  /** Parses the common keywords (overlay/pair, break); returns the other key/value words for the subclass. */
  protected takeCommon(args: string[]): string[] {
    const rest: string[] = [];
    const seen = new Set<string>();
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k], val = args[k + 1];
      if (key === 'store/local') throw new StyleError(`bond_style ${this.name} keyword store/local is not supported by the browser engine (no fix store/local)`);
      if (key === 'overlay/pair' || key === 'break') {
        if (val === undefined) throw new StyleError(`Illegal bond bpm command, missing option for ${key}`);
        if (seen.has(key)) throw new StyleError(`bond_style ${this.name}: keyword '${key}' is given twice`);
        seen.add(key);
        if (key === 'overlay/pair') this.overlay = yesno(val, `bond_style ${this.name}`, key);
        else this.breakOn = yesno(val, `bond_style ${this.name}`, key);
      } else {
        rest.push(key);
        if (val !== undefined) rest.push(val);
      }
    }
    return rest;
  }

  /** True once after a compute() that broke bonds (the force field then rebuilds the special lists). */
  takeTopologyChange(): boolean {
    const c = this.changed;
    this.changed = false;
    // overlay/pair yes: the special lists are not updated when bonds break (bond_bpm.cpp: no fix update_special_bonds)
    return c && !this.overlay;
  }

  /** Breaks bond k: its type is set to 0 (the doc: "setting the bond type to 0"). */
  protected breakBond(type: Int32Array, k: number): void {
    type[k] = 0;
    this.changed = true;
  }

  /** Index pair of a bond (atoms i1 = smaller ID, i2 = larger ID, as LAMMPS' bondlist ordering). */
  protected pairOf(bc: BondedCompute, ia: number, ib: number): { i1: number; i2: number; a1: number; a2: number } {
    const lo = Math.min(ia, ib), hi = Math.max(ia, ib);
    return { i1: atomIndex(bc, lo, 'bond'), i2: atomIndex(bc, hi, 'bond'), a1: lo, a2: hi };
  }
}
