/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm_zero.cpp and src/BPM/bond_bpm_zero.h at commit b332f28 (Joel Clemmer, SNL; copyright
 * Sandia Corporation, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * bond_style bpm/zero: docs.lammps.org/bond_bpm_zero.html. The doc says the style "does not compute any forces;
 * it only breaks bonds" (bonds break at a strain of ecrit). The reference length r0 is stored by each bond when it
 * is first computed; a bond breaks when |r - r0| / r0 exceeds ecrit, by setting its type to 0. The C++ breaks
 * only outside the setup (update->setupflag == 0), which the engine reads from bc.historyUpdate.
 * The manybody keyword needs fix property/atom and the ghost communication of the C++ style: refused by name.
 */
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { missingArgs } from '../../engine/force/util';
import { BondBPMBase, bondKey, yesno } from './bond_bpm';

export class BondBpmZero extends BondBPMBase {
  readonly name = 'bpm/zero';
  readonly kind = 'bond' as const;
  readonly paramNames = ['ecrit'];
  /** Reference length per bond, keyed by the two atom IDs. */
  private readonly ref = new Map<number, number>();

  settings(args: string[], _ctx?: StyleContext): void {
    const rest = this.takeCommon(args);
    for (let k = 0; k < rest.length; k += 2) {
      const key = rest[k], val = rest[k + 1];
      // bond_bpm_zero.cpp settings: the keyword, then its value
      if (!['manybody'].includes(key)) throw new StyleError(`Illegal bond bpm command, invalid argument ${key}`);
      if (val === undefined) throw missingArgs(`bond_style bpm/zero ${key}`);
      if (key === 'manybody') {
        if (yesno(val, 'bond_style bpm/zero', key)) throw new StyleError('bond_style bpm/zero keyword manybody yes is not supported by the browser engine');
      } else throw new StyleError(`Illegal bond bpm command, invalid argument ${key}`);
    }
  }

  compute(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const ecrit = this.params.p('ecrit');
    const updating = bc.historyUpdate === true;
    const d = [0, 0, 0];
    for (let k = 0; k < b.n; k++) {
      const t = b.type[k];
      if (t === 0) continue;
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const i1 = atomIndex(bc, Math.min(ia, ib), 'bond'), i2 = atomIndex(bc, Math.max(ia, ib), 'bond');
      delta(bc, i1, i2, d);
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const key = bondKey(ia, ib);
      let r0 = this.ref.get(key);
      if (r0 === undefined) {
        r0 = r;
        this.ref.set(key, r0);
      }
      const e = (r - r0) / r0;
      if (Math.abs(e) > ecrit[t] && updating && this.breakOn) this.breakBond(b.type, k);
    }
  }
}
