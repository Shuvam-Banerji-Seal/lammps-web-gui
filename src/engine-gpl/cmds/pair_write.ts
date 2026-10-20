/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/pair.cpp (Pair::write_file) and the
 * command dispatch of src/input.cpp (Input::pair_write). LAMMPS development team, Sandia National Laboratories. This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_write itype jtype N style inner outer file keyword [Qi Qj] — pair_write.html: "Write energy and force values to a
 * file as a function of distance for the currently defined pair potential." and "If the style is *r*, then N distances
 * are used, evenly spaced in r; if the style is *rsq*, N distances are used, evenly spaced in r\^2." The "Qi,Qj = 2 atom
 * charges (charge units) (optional)" keyword arguments are the charges of the two atoms (1 when omitted, as native).
 * The style *bitmap* is refused: its rsq lookup (Pair::init_bitmap) is not ported.
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { formatNumber } from '../../engine/cformat';
import { nativeFloat, nativeInt } from './common';
import { startTableFile, tableRow } from './table_write';

/** Native "%.15g" of a cutoff in the table header. */
const g15 = (v: number): string => formatNumber(v, '%.15g');

const pairWrite: Handler = ({ sys }, a) => {
  const pair = sys.ff.pair;
  // Input::pair_write: "Pair_write command before pair_style is defined"
  if (!pair) throw new StyleError('Pair_write command before pair_style is defined');
  if (a.length !== 8 && a.length !== 10) throw new StyleError('Illegal pair_write command');
  const single = pair.single?.bind(pair);
  // Pair::write_file: "Pair style {} does not support pair_write" when single_enable is 0
  if (!single) throw new StyleError(`Pair style ${pair.name} does not support pair_write`);
  if (pair.singleNeedsSeparation()) {
    throw new StyleError(`pair_write for pair style ${pair.name} needs the separation vector of its single() (not supported by the browser engine)`);
  }
  const ntypes = sys.state.ntypes;
  const itype = nativeInt(a[0]);
  const jtype = nativeInt(a[1]);
  if (itype < 1 || itype > ntypes || jtype < 1 || jtype > ntypes) throw new StyleError('Invalid atom types in pair_write command');
  const n = nativeInt(a[2]);
  const style = a[3];
  if (style !== 'r' && style !== 'rsq' && style !== 'bitmap') throw new StyleError(`Invalid style ${style} in pair_write command`);
  if (n < 2) throw new StyleError('Must have at least 2 table values');
  const inner = nativeFloat(a[4]);
  const outer = nativeFloat(a[5]);
  if (inner <= 0 || inner >= outer) throw new StyleError(`Invalid cutoffs (${inner} - ${outer}) in pair_write command`);
  if (style === 'bitmap') throw new StyleError('pair_write style bitmap is not supported by the browser engine (its rsq bit-mask lookup is not ported)');
  const tableFile = a[6];
  const keyword = a[7];

  // Pair::write_file opens the file (append or create, with the DATE/UNITS line) and writes the header first
  const { head, append } = startTableFile(sys, { file: tableFile, cmd: 'pair_write' });
  const sectionLine = style === 'r' ? `N ${n} R ${g15(inner)} ${g15(outer)}` : `N ${n} RSQ ${g15(inner)} ${g15(outer)}`;
  sys.writeFile(tableFile, `${head}# Pair potential ${pair.name} for atom types ${itype} ${jtype}: i,r,energy,force\n\n${keyword}\n${sectionLine}\n\n`, append);

  // force->init() and neighbor->init(): the pair style's cutoffs and coefficients as a run would set them up
  sys.ff.init(sys.state, sys.nb, sys.geom, sys.styleContext());
  // the charges of the two atoms (native swaps them into atom->q)
  const qi = a.length === 10 ? nativeFloat(a[8]) : 1;
  const qj = a.length === 10 ? nativeFloat(a[9]) : 1;

  // energy and force at each of N distances, with the single() of atoms 0 and 1 (factors 1)
  const nt = pair.ntypes + 1;
  const cutsq = pair.cutsq[itype * nt + jtype];
  let rows = '';
  for (let i = 0; i < n; i++) {
    let r: number;
    let rsq: number;
    if (style === 'r') {
      r = inner + ((outer - inner) * i) / (n - 1);
      rsq = r * r;
    } else {
      rsq = inner * inner + ((outer * outer - inner * inner) * i) / (n - 1);
      r = Math.sqrt(rsq);
    }
    let e = 0;
    let f = 0;
    if (rsq < cutsq) {
      const res = single(0, 1, itype, jtype, rsq, 1, 1, qi, qj);
      e = res.eng;
      // single() returns the force divided by r; the table holds the force
      f = res.fforce * r;
    }
    rows += tableRow(i + 1, r, e, f);
  }
  sys.writeFile(tableFile, rows, true);
};

export const COMMANDS = { pair_write: pairWrite };
