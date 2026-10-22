import { StyleError, type StyleContext, potentialFile } from '../types';
import { PairSWBase } from './sw';
import { AngleTable, readAngleSection } from '../angleTable49';
import { libm } from '../../libm';

/*
 * pair_style sw/angle/table — docs.lammps.org/pair_sw_angle_table.html
 * (source: plans/lammps-docs-live/pair_sw_angle_table.rst)
 *
 * Syntax: pair_style sw/angle/table, with no settings documented; any argument is refused here.
 *
 * Energy: the Stillinger-Weber screening terms with the angular term replaced. "In contrast to the original sw style,
 * sw/angle/table allows for a flexible three-body term" f^{3b}(theta_ijk), read in as a tabulated interaction.
 *
 * The tabulated angular potential "has to be of the format as used in the angle_style table command". "For the linear
 * style, the angle is used to find 2 surrounding table values from which an energy or its derivative is computed by
 * linear interpolation." The table construction, the spline style and the measured interpolation details are in
 * angleTable49.ts. Measured with native LAMMPS (black box): the energy and force of a three-atom configuration equal the
 * chain rule of that table.
 *
 * The lambda and costheta0 values of an entry are read and ignored; the page says they "are read in but ignored in
 * this modified pair style". epsilon is used for the two-body term only.
 *
 * "This pair style does not support the pair_modify shift, table, and tail options." UNITS conversion of the table
 * energies is not documented, so a file whose UNITS: tag needs a conversion is refused. The threebody and maxdelcs
 * keywords belong to sw and sw/mod and are refused here.
 */

/** pair_style sw/angle/table: the Stillinger-Weber screening with an angular table in place of lambda (cos - cos0)^2. */
export class PairSWAngleTable extends PairSWBase {
  readonly name = 'sw/angle/table';

  private tables = new Map<string, AngleTable>();
  private tabOf: (AngleTable | null)[] = [];

  constructor() {
    super(false);
  }

  protected override get tabulatedAngle(): boolean {
    return true;
  }

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length) throw new StyleError(`pair_style sw/angle/table takes no settings (got '${args[0]}')`);
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.tables.clear();
    this.tabOf = [];
  }

  override initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    if (this.unitScale !== 1) throw new StyleError('pair_style sw/angle/table: UNITS conversion of the angular table energies is not documented and is refused');
    if (!this.threebody) return;
    const nt = this.ntypes + 1;
    this.tabOf = new Array<AngleTable | null>(nt * nt * nt).fill(null);
    for (let ti = 1; ti <= this.ntypes; ti++) {
      if (this.elemOf[ti] < 0) continue;
      for (let tj = 1; tj <= this.ntypes; tj++) {
        if (this.elemOf[tj] < 0) continue;
        for (let tk = 1; tk <= this.ntypes; tk++) {
          if (this.elemOf[tk] < 0) continue;
          const names = [this.elemNames[this.elemOf[ti]], this.elemNames[this.elemOf[tj]], this.elemNames[this.elemOf[tk]]];
          const e = this.entry(names.join(' '), `needed by the angular table of types ${ti} ${tj} ${tk}`);
          const tab = e.tab;
          if (!tab) throw new StyleError(`SW potential file ${this.fileName} entry ${names.join(' ')} has no angular table`);
          if (tab.style !== 'linear' && tab.style !== 'spline') {
            throw new StyleError(`SW potential file ${this.fileName} entry ${names.join(' ')}: table style '${tab.style}' is not supported (linear or spline)`);
          }
          if (!Number.isInteger(tab.n) || tab.n < 2) throw new StyleError(`SW potential file ${this.fileName} entry ${names.join(' ')}: table N must be an integer >= 2`);
          const key = `${tab.filename}\u0000${tab.keyword}\u0000${tab.style}\u0000${tab.n}`;
          let built = this.tables.get(key);
          if (!built) {
            const sec = readAngleSection(ctx.readFile(tab.filename, potentialFile('angletable table')), tab.filename, tab.keyword);
            built = AngleTable.build(sec, tab.n, tab.style, `pair_style sw/angle/table (${tab.filename} ${tab.keyword})`);
            this.tables.set(key, built);
          }
          this.tabOf[(ti * nt + tj) * nt + tk] = built;
        }
      }
    }
  }

  /** Energy and dE/dcos of the tabulated angle: the energy is interpolated, the derivative is -dE/dtheta (degrees) mapped to cos. */
  protected override angular(t3: number, c: number): boolean {
    const tab = this.tabOf[t3];
    if (!tab) return false;
    const cc = c > 1 ? 1 : c < -1 ? -1 : c;
    const theta = (libm.acos(cc) * 180) / Math.PI;
    this.aE = tab.energy(theta);
    const s = Math.sqrt(1 - cc * cc);
    // dE/dcos = dE/dtheta_deg * dtheta_deg/dcos = (-D) * (-(180/pi)/sin) = D * (180/pi) / sin
    this.aD = s > 0 ? (tab.derivative(theta) * 180) / Math.PI / s : 0;
    return true;
  }
}
