import { StyleError } from '../types';
import { parseNum } from '../util';
import { PairLJCutCoulLong } from './coul_long';

/*
 * pair_style lj/long/coul/long — docs.lammps.org/pair_lj_long.html
 * (source: plans/lammps-docs/pair_lj_long.rst)
 *
 * Syntax (verbatim):
 *   "*lj/long/coul/long* args = flag_lj flag_coul cutoff (cutoff2)"
 *   "flag_lj = *long* or *cut* or *off*"
 *   "*long* = use Kspace long-range summation for dispersion 1/r\^6 term"
 *   "*cut* = use a cutoff on dispersion 1/r\^6 term"
 *   "*off* = omit disperion 1/r\^6 term entirely"
 *   "flag_coul = *long* or *off*"
 *   "*long* = use Kspace long-range summation for Coulombic 1/r term"
 *   "*off* = omit Coulombic term"
 *
 * Description (verbatim):
 *   "If one cutoff is specified in the pair_style command, it is used for both
 *   the LJ and Coulombic terms.  If two cutoffs are specified, they are
 *   used as cutoffs for the LJ and Coulombic terms respectively."
 *   "If *flag_lj* is set to *long*, no cutoff is used on the LJ 1/r\^6
 *   dispersion term.  The long-range portion can be calculated by using
 *   the :doc:`kspace_style ewald/disp or pppm/disp <kspace_style>` commands."
 *   "If *flag_lj* is set to *cut*, the LJ interactions are simply cutoff, as
 *   with :doc:`pair_style lj/cut <pair_lj>`."
 *   "If *flag_coul* is set to *off*, Coulombic interactions are not computed."
 *   "The following coefficients must be defined for each pair of atoms
 *   types via the :doc:`pair_coeff <pair_coeff>` command" (epsilon, sigma, cutoff1, cutoff2)
 *
 * What this engine implements:
 *  - flag_lj = cut with flag_coul = long: the 12-6 LJ with a plain cutoff
 *    (as lj/cut, no shift) plus the Ewald real-space Coulomb term of
 *    lj/cut/coul/long, which needs kspace ewald or pppm.  Measured with native
 *    LAMMPS (black box, pair_write, units real, LJ cutoff 8): the LJ energy is
 *    plain LJ and zero beyond 8 (no shift).
 *  - flag_lj = cut with flag_coul = off: the LJ alone, measured with native LAMMPS
 *    (pair_write with charges present: E equals plain LJ up to the cutoff, zero beyond).
 *    Native LAMMPS still requires a kspace style for any lj/long/coul/long pair
 *    style; this engine does not.
 *  - Not implemented, by name: flag_lj = long (dispersion summation needs
 *    kspace ewald/disp or pppm/disp, which the engine does not have), and
 *    flag_lj = off.  Measured with native LAMMPS: with flag_lj = off, kspace
 *    ewald and pppm are both rejected (KSpace style ewald is incompatible with
 *    Pair style lj/long/coul/long) for flag_coul = long and for flag_coul = off.
 *  - The per-pair cutoff2 of pair_coeff (a Coulomb cutoff for one type pair) is
 *    not implemented; the global Coulomb cutoff is used.
 */
export class PairLJLongCoulLong extends PairLJCutCoulLong {
  readonly name: string = 'lj/long/coul/long';

  settings(args: string[]): void {
    if (args.length !== 3 && args.length !== 4) {
      throw new StyleError('usage: pair_style lj/long/coul/long flag_lj flag_coul cutoff (cutoff2)');
    }
    const [flagLJ, flagCoul] = args;
    if (flagLJ === 'long') {
      throw new StyleError("pair_style lj/long/coul/long flag_lj 'long' (long-range 1/r^6 dispersion) needs kspace ewald/disp or pppm/disp, which are not implemented");
    }
    if (flagLJ === 'off') {
      throw new StyleError("pair_style lj/long/coul/long flag_lj 'off' is not supported (native LAMMPS rejects kspace ewald and pppm with it)");
    }
    if (flagLJ !== 'cut') throw new StyleError(`pair_style lj/long/coul/long: invalid flag_lj '${flagLJ}' (long, cut or off)`);
    if (flagCoul !== 'long' && flagCoul !== 'off') throw new StyleError(`pair_style lj/long/coul/long: invalid flag_coul '${flagCoul}' (long or off)`);
    this.cutGlobal = parseNum(args[2], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style lj/long/coul/long: cutoff must be > 0');
    this.cutCoul = args.length === 4 ? parseNum(args[3], 'Coulomb cutoff') : this.cutGlobal;
    if (!(this.cutCoul > 0)) throw new StyleError('pair_style lj/long/coul/long: Coulomb cutoff must be > 0');
    if (flagCoul === 'long') {
      this.coulLong = true;
    } else {
      // "If *flag_coul* is set to *off*, Coulombic interactions are not computed."
      this.coulLong = false;
      this.cutCoul = 0;
    }
  }

  coeff(args: string[]): void {
    if (args.length === 6) {
      throw new StyleError('pair_coeff cutoff2 (a per-pair Coulomb cutoff) is not implemented for pair style lj/long/coul/long');
    }
    super.coeff(args);
  }
}
