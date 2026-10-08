import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { SimState } from '../types';
import { hasChargeStyle, isMolecularStyle } from '../atoms';
import { shortest } from '../output/data';

/*
 * fix ID group-ID property/atom name1 name2 ... keyword value ... —
 * docs.lammps.org/fix_property_atom.html:
 *   * name1,name2,... = *mol* or *q* or *rmass* or i_name or d_name or i2_name or d2_name
 *   "Create one or more additional per-atom vectors or arrays to store
 *   information about atoms and to use during a simulation.  The specified
 *   *group-ID* is ignored by this fix."
 *   "The *i2_name* and *d2_name* options take an argument *N* which
 *   specifies the number of columns in the per-atom array, i.e. the number
 *   of attributes associated with each atom.  *N* >= 1 is required."
 *   "Each name must be unique and can use alphanumeric or underscore
 *   characters."
 *   "This fix is one of a small number that can be defined in an input
 *   script before the simulation box is created or atoms are defined."
 *   "The per-atom properties defined by this fix are not.  So you need to
 *   initialize them explicitly."
 *   "Note that the order of values on each line corresponds to the order of
 *   custom names in the fix property/atom command."
 *
 * Measured with native LAMMPS (black box): new atoms get 0 for every
 * property (rmass too, which then replaces the per-type mass); write_data
 * appends a section titled with the fix ID and the names (arrays without
 * the 2: "pa # mol q rmass i_flag d_val d_vec"), one line per atom; native
 * stops with "Fix property/atom vector name already exists", "... mol when
 * atom_style already has molecule attribute" (likewise q / charge, rmass),
 * and "Invalid array columns number N in fix property/atom". The ghost
 * keyword only changes communication, which a single-process engine does
 * not need. temperature and heatflow (GRANULAR heat conduction) are not
 * supported here.
 */

interface PropSpec { kind: 'mol' | 'q' | 'rmass' | 'i' | 'd'; name: string; cols: number }

/** The fix's per-atom values as a restart file keeps them (output/restart.ts). */
export interface PropertyAtomRestart { id: string; props: PropSpec[]; data: Record<string, number[]> }

export class FixPropertyAtom extends Fix {
  readonly style = 'property/atom';
  readonly props: PropSpec[] = [];
  private attached: SimState | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError('usage: fix ID group-ID property/atom name1 name2 ... [ghost yes|no]');
    const seen = new Set<string>();
    for (let k = 0; k < args.length; k++) {
      const w = args[k];
      if (w === 'ghost') {
        const v = args[k + 1];
        if (v !== 'yes' && v !== 'no') throw new StyleError('fix property/atom ghost value must be yes or no');
        k++;
        continue;
      }
      let spec: PropSpec;
      if (w === 'mol' || w === 'q' || w === 'rmass') spec = { kind: w, name: w, cols: 0 };
      else if (w === 'temperature' || w === 'heatflow') {
        throw new StyleError(`fix property/atom ${w} (GRANULAR heat conduction) is not supported by the browser engine`);
      } else {
        const m = /^(i|d)(2?)_([A-Za-z0-9_]+)$/.exec(w);
        if (!m) throw new StyleError(`Illegal fix property/atom command: unknown property '${w}'`);
        let cols = 0;
        if (m[2]) {
          const n = Number(args[k + 1]);
          if (!Number.isInteger(n) || n < 1) throw new StyleError(`Invalid array columns number ${args[k + 1] ?? ''} in fix property/atom`);
          cols = n;
          k++;
        }
        spec = { kind: m[1] as 'i' | 'd', name: m[3], cols };
      }
      if (seen.has(spec.name)) throw new StyleError(`Fix property/atom ${spec.cols ? 'array' : 'vector'} name already exists`);
      seen.add(spec.name);
      this.props.push(spec);
    }
    const s = sys.hasBox ? sys.state : null;
    if (s) {
      this.attachState(s);
      this.restoreFromRestart(s);
    }
  }

  /**
   * fix_property_atom.html: "When reading data from a restart file, this fix command has to be
   * specified **after** the *read_restart* command and **exactly** the same was in the input
   * script that created the restart file." "LAMMPS will only check whether a fix is of the same
   * style and has the same fix ID and in case of a match will then try to initialize the fix with
   * the data stored in the binary restart file." Where native may corrupt data on a mismatch, the
   * engine stops with an error instead.
   */
  private restoreFromRestart(s: SimState): void {
    const saved = this.sys.pendingFixData.get(this.id);
    if (!saved) return;
    this.sys.pendingFixData.delete(this.id);
    const same = saved.props.length === this.props.length
      && saved.props.every((p, k) => p.kind === this.props[k].kind && p.name === this.props[k].name && p.cols === this.props[k].cols);
    if (!same) throw new StyleError(`fix ${this.id} property/atom does not define the same properties as the fix that wrote the restart file`);
    for (const p of this.props) {
      const key = p.kind === 'mol' || p.kind === 'q' || p.kind === 'rmass' ? p.kind : `${p.kind}_${p.name}`;
      const v = saved.data[key];
      if (!v) continue;
      if (p.kind === 'mol') s.molecule.set(v.slice(0, s.n));
      else if (p.kind === 'q') s.q.set(v.slice(0, s.n));
      else if (p.kind === 'rmass') s.rmass!.set(v.slice(0, s.n));
      else s.custom.get(p.name)!.data.set(v.slice(0, s.custom.get(p.name)!.data.length));
    }
  }

  /** Adds the properties to a state (the current one, or the one create_box / read_data makes later). */
  attachState(s: SimState): void {
    if (this.attached === s) return;
    for (const p of this.props) {
      if (p.kind === 'mol' && (isMolecularStyle(s.atomStyle) || s.propMol)) throw new StyleError('Fix property/atom mol when atom_style already has molecule attribute');
      if (p.kind === 'q' && (hasChargeStyle(s.atomStyle) || s.propQ)) throw new StyleError('Fix property/atom q when atom_style already has charge attribute');
      if (p.kind === 'rmass' && s.rmass) throw new StyleError('Fix property/atom rmass when atom_style already has rmass attribute');
      if ((p.kind === 'i' || p.kind === 'd') && s.custom.has(p.name)) throw new StyleError(`Fix property/atom ${p.cols ? 'array' : 'vector'} name already exists`);
    }
    for (const p of this.props) {
      if (p.kind === 'mol') s.propMol = true;
      else if (p.kind === 'q') s.propQ = true;
      else if (p.kind === 'rmass') s.rmass = new Float64Array(s.n);
      else s.custom.set(p.name, { int: p.kind === 'i', cols: p.cols, data: new Float64Array(s.n * Math.max(p.cols, 1)) });
    }
    this.attached = s;
  }

  /** unfix removes the properties again. */
  destroy(): void {
    const s = this.attached;
    if (!s) return;
    for (const p of this.props) {
      if (p.kind === 'mol') s.propMol = false;
      else if (p.kind === 'q') s.propQ = false;
      else if (p.kind === 'rmass') s.rmass = null;
      else s.custom.delete(p.name);
    }
    this.attached = null;
  }

  /** Values per data-file line after the atom ID. */
  get nvalues(): number {
    let n = 0;
    for (const p of this.props) n += p.kind === 'i' || p.kind === 'd' ? Math.max(p.cols, 1) : 1;
    return n;
  }

  /** read_data fix section: one line "atom-ID values..." for atom index i. */
  readValues(s: SimState, i: number, w: readonly string[], at: number): void {
    if (w.length < this.nvalues) throw new StyleError(`data file line ${at}: fix ${this.id} section needs ${this.nvalues} values after the atom ID`);
    let k = 0;
    for (const p of this.props) {
      const num = (t: string) => {
        const v = Number(t);
        if (!Number.isFinite(v)) throw new StyleError(`data file line ${at}: expected a number, got '${t}'`);
        return v;
      };
      if (p.kind === 'mol') s.molecule[i] = Math.trunc(num(w[k++]));
      else if (p.kind === 'q') s.q[i] = num(w[k++]);
      else if (p.kind === 'rmass') s.rmass![i] = num(w[k++]);
      else {
        const c = s.custom.get(p.name)!;
        const width = Math.max(c.cols, 1);
        for (let m = 0; m < width; m++) {
          const v = num(w[k++]);
          c.data[width * i + m] = c.int ? Math.trunc(v) : v;
        }
      }
    }
  }

  /** write_data section title: the fix ID and the names (arrays without the 2, as native writes). */
  sectionHeader(): string {
    return `${this.id} # ${this.props.map((p) => (p.kind === 'i' || p.kind === 'd' ? `${p.kind}_${p.name}` : p.name)).join(' ')}`;
  }

  /** write_data section line for atom index i. */
  writeValues(s: SimState, i: number): string {
    const out: string[] = [String(s.id[i])];
    for (const p of this.props) {
      if (p.kind === 'mol') out.push(String(s.molecule[i]));
      else if (p.kind === 'q') out.push(shortest(s.q[i]));
      else if (p.kind === 'rmass') out.push(shortest(s.rmass![i]));
      else {
        const c = s.custom.get(p.name)!;
        const width = Math.max(c.cols, 1);
        for (let m = 0; m < width; m++) out.push(c.int ? String(Math.trunc(c.data[width * i + m])) : shortest(c.data[width * i + m]));
      }
    }
    return out.join(' ');
  }
}
