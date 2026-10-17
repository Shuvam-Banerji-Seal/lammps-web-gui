/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MISC/pair_srp_react.cpp and pair_srp_react.h at commit b332f28 (pair_style srp/react; Vaibhav Palkar and
 * Olga Kuksenok, Clemson University). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair srp (pair_srp.ts) whose settings name a fix bond/break or bond/create; its fix SRPREACT (fix_srp_react.ts)
 * remakes the bond particles when that fix changes the bonds. As native, the style's base part names a fix SRP first
 * (the number is used up) and the style then makes "{:02d}_FIX_SRP_REACT all SRPREACT" with the next number.
 */
import { StyleError, type StyleContext } from '../../engine/force/types';
import { PairSRP, nextSrpInstance } from './pair_srp';
import { FixSRPREACT } from './fix_srp_react';

const numeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  return v;
};
const inumeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isInteger(v)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  return v;
};

export class PairSRPREACT extends PairSRP {
  readonly name: string = 'srp/react';
  private idbreak: string | null = null;
  private idcreate: string | null = null;

  constructor() {
    super();
    this.fixStyle = 'SRPREACT';
  }

  settings(args: string[], ctx: StyleContext): void {
    if (args.length < 3 || args.length > 8) throw new StyleError('Illegal pair_style command');
    const s = ctx.s;
    this.cutGlobal = numeric(args[0]);
    if (args[1] === '*') this.btype = 0;
    else {
      this.btype = inumeric(args[1]);
      if (this.btype > (s ? s.topo.nbondtypes : 0) || this.btype <= 0) throw new StyleError('Illegal pair_style command');
    }
    this.btypeStr = '';
    this.bptypeStr = '';
    this.midpoint = false;
    this.min = false;
    if (args[2] === 'min') this.min = true;
    else if (args[2] === 'mid') this.midpoint = true;
    else throw new StyleError('Illegal pair_style command');
    this.idbreak = null;
    this.idcreate = null;
    const reactid = args[3];
    const rfix = reactid !== undefined ? ctx.fixById?.(reactid) ?? null : null;
    if (!rfix) throw new StyleError(`Could not find fix ID ${reactid ?? ''} for pair srp/react`);
    if (rfix.style.startsWith('bond/break')) this.idbreak = reactid;
    else if (rfix.style.startsWith('bond/create')) this.idcreate = reactid;
    else throw new StyleError('Illegal pair_style command');
    let iarg = 4;
    this.exclude = true;
    this.bptype = s ? s.ntypes : 0;
    while (iarg < args.length) {
      if (args[iarg] === 'exclude') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal pair srp command');
        const v = args[iarg + 1].toLowerCase();
        if (['yes', 'on', 'true', '1'].includes(v)) this.exclude = true;
        else if (['no', 'off', 'false', '0'].includes(v)) this.exclude = false;
        else throw new StyleError(`Expected boolean parameter instead of '${args[iarg + 1]}' in input script or data file`);
        if (this.min && !this.exclude) throw new StyleError('Illegal exclude option in pair srp command');
        iarg += 2;
      } else if (args[iarg] === 'bptype') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal pair srp command');
        this.bptype = inumeric(args[iarg + 1]);
        if (this.bptype < 1 || this.bptype > (s ? s.ntypes : 0)) throw new StyleError('Illegal bond particle type for srp');
        iarg += 2;
      } else throw new StyleError('Illegal pair srp command');
    }
    this.createFix(ctx);
  }

  /** The base constructor's fix SRP number is used up; this style's fix is "{:02d}_FIX_SRP_REACT all SRPREACT". */
  protected createFix(ctx: StyleContext): void {
    nextSrpInstance();
    const fixId = `${String(nextSrpInstance()).padStart(2, '0')}_FIX_SRP_REACT`;
    this.fixId = fixId;
    const idbreak = this.idbreak, idcreate = this.idcreate;
    if (!ctx.addFix) throw new StyleError('pair srp/react needs to create its fix SRPREACT');
    ctx.addFix((sys) => {
      const f = new FixSRPREACT(sys, fixId, 'all', []);
      f.idbreak = idbreak;
      f.idcreate = idcreate;
      return f;
    });
  }
}
