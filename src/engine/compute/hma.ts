import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * compute ID group-ID hma temp-ID keyword ... — docs.lammps.org/compute_hma.html (plans/lammps-docs-live/
 * compute_hma.rst). Quoted sentences are verbatim from that page:
 *
 *   "temp-ID = ID of fix that specifies the set temperature during canonical simulation"
 *   "keyword = *anharmonic* or *u* or *p* or *cv*"
 *   "The potential energy is computed by the formula:"
 *   "The pressure is computed by the formula:"
 *   "HMA should not be used if atoms are expected to diffuse. It is also restricted to simulations in the NVT ensemble."
 *   "Compute hma command should be used right after the energy minimization, when the atoms are at their lattice sites."
 *   "This compute calculates a global vector that includes the n properties requested as arguments to the command"
 *
 * Formulas (the doc's display equations, in the doc's own symbols):
 *   <U>_HMA = d/2 (N-1) k_B T + <U + 1/2 F . dr>
 *   <P>_HMA = dP + <P_vir + (beta dP - rho) / (d (N-1)) F . dr>
 * with U the potential energy (pair, bond, angle, dihedral, improper, kspace, fix), F the atomic forces,
 * dr the displacement from the lattice site, T the set temperature of the fix (its constant
 * temperature), beta = 1/(k_B T), rho = N/V and dP the pressure keyword value, P_vir the virial pressure.
 * The sites are the unwrapped positions when the compute is defined (the doc: "when the atoms are at
 * their lattice sites"). Measured with native LAMMPS (black box), lj units, pair lj/smooth/linear 3,
 * fix nvt: the vector entry u is extensive (thermo prints it per atom with 32 atoms), it equals
 * pe + 3/2 (N-1) T + F.dr/2 when the compute is defined before a displacement, and equals
 * pe + 3/2 (N-1) T when it is defined after the displacement (sites are then the displaced positions).
 * The p entry equals dP + P_vir + (beta dP - rho) F.dr / (3 (N-1)) (P_vir from compute pressure NULL pair).
 *
 * Not implemented (EngineError): cv (needs the pair single_hessian method, which the doc lists as implemented
 * only by pair lj/smooth/linear, with the Hessian of that style) and anharmonic (the doc gives no formula for
 * its values). Only group all, constant set temperatures, lj units and orthogonal boxes are accepted.
 */

export class ComputeHma extends Compute {
  readonly style = 'hma';
  private readonly temp: string;
  private readonly keys: string[] = [];
  private readonly pharm: number;
  /** Lattice sites by atom id: unwrapped x, y, z (3 per id). */
  private sites = new Float64Array(0);
  private hasSites = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const usage = 'usage: compute ID group-ID hma temp-ID keyword ... (keyword = u, p Pharm or cv; anharmonic not implemented)';
    if (group !== 'all') throw new StyleError(`compute ${id} (hma): group must be all (got ${group})`);
    if (args.length < 2) throw new StyleError(usage);
    this.temp = args[0];
    let pharm = NaN;
    for (let k = 1; k < args.length; k++) {
      const w = args[k];
      if (w === 'u') this.keys.push('u');
      else if (w === 'p') {
        const v = Number(args[k + 1]);
        if (args[k + 1] === undefined || !Number.isFinite(v)) throw new StyleError(`compute ${id} (hma): keyword p needs a pressure value (Pharm)`);
        pharm = v;
        this.keys.push('p');
        k++;
      } else if (w === 'cv') {
        throw new StyleError(`compute ${id} (hma): keyword cv is not implemented (it needs the pair style single_hessian method, which the doc lists for pair lj/smooth/linear only)`);
      } else if (w === 'anharmonic') {
        throw new StyleError(`compute ${id} (hma): keyword anharmonic is not implemented (the doc gives no formula for its values)`);
      } else {
        throw new StyleError(`compute ${id} (hma): unknown keyword '${w}' (${usage})`);
      }
    }
    if (this.keys.length === 0) throw new StyleError(usage);
    if (this.keys.includes('p') && !Number.isFinite(pharm)) throw new StyleError(usage);
    this.pharm = pharm;
    const st = sys.state;
    if (st.box.triclinic) throw new StyleError(`compute ${id} (hma): triclinic boxes are not supported`);
    if (st.units.nktv2p !== 1 || st.units.boltz !== 1) throw new StyleError(`compute ${id} (hma): only lj units are supported (the pressure and energy conversions are measured with lj)`);
    this.vectorFlag = true;
    this.sizeVector = this.keys.length;
    this.extvector = 1;
    this.extlist = this.keys.map((k) => (k === 'u' ? 1 : 0));
    this.vector = new Float64Array(this.keys.length);
    // lattice sites: unwrapped positions at definition (the doc's "when the atoms are at their lattice sites")
    let maxId = 0;
    for (let i = 0; i < st.n; i++) if (st.id[i] > maxId) maxId = st.id[i];
    this.sites = new Float64Array(3 * (maxId + 1));
    this.hasSites = new Uint8Array(maxId + 1);
    const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
    for (let i = 0; i < st.n; i++) {
      const k = st.id[i];
      this.hasSites[k] = 1;
      for (let d = 0; d < 3; d++) this.sites[3 * k + d] = st.x[3 * i + d] + st.image[3 * i + d] * L[d];
    }
  }

  /** Checks the thermostat at the start of each run (the set temperature is read at every evaluation). */
  init(): void {
    this.setTemperature();
  }

  /** Set temperature of the thermostat fix: constant only. */
  private setTemperature(): number {
    const f = this.sys.fix(this.temp) as unknown as { style?: string; tStart?: number; tStop?: number };
    if (!f || (f.style !== 'nvt' && f.style !== 'npt' && f.style !== 'nph' && f.style !== 'temp/berendsen' && f.style !== 'temp/csvr')) {
      throw new StyleError(`compute ${this.id} (hma): temp-ID ${this.temp} must be a thermostat fix (nvt, temp/berendsen, ...)`);
    }
    if (typeof f.tStart !== 'number' || f.tStart !== f.tStop) throw new StyleError(`compute ${this.id} (hma): the set temperature of fix ${this.temp} must be constant`);
    return f.tStart;
  }

  protected computeVector(): void {
    const sys = this.sys;
    const st = sys.state;
    const a = sys.forces();
    const d = sys.dimension;
    const n = st.n;
    const T = this.setTemperature();
    const kT = st.units.boltz * T;
    const L = [0, 1, 2].map((k) => st.box.hi[k] - st.box.lo[k]);
    // F . dr, dr = unwrapped position - lattice site (atoms are matched by id; a site missing for an id is an error)
    let fdr = 0;
    for (let i = 0; i < n; i++) {
      const k = st.id[i];
      if (k >= this.hasSites.length || !this.hasSites[k]) throw new StyleError(`compute ${this.id} (hma): atom ${k} has no lattice site (it was created after the compute)`);
      for (let c = 0; c < 3; c++) {
        const dr = st.x[3 * i + c] + st.image[3 * i + c] * L[c] - this.sites[3 * k + c];
        fdr += st.f[3 * i + c] * dr;
      }
    }
    const U = a.evdwl + a.ecoul + a.ebond + a.eangle + a.edihed + a.eimp + a.elong + sys.fixEnergy();
    const vol = sys.geom.volume(d);
    const rho = n / vol;
    const w = new Float64Array(6);
    const add = (v: Float64Array) => { for (let c = 0; c < 6; c++) w[c] += v[c]; };
    add(a.virial); add(a.vbond); add(a.vangle); add(a.vdihed); add(a.vimp); add(a.vlong);
    sys.fixVirial(w);
    const trace = d === 3 ? w[0] + w[1] + w[2] : w[0] + w[1];
    const pvir = trace / (d * vol);
    const N1 = n - 1;
    this.keys.forEach((key, c) => {
      if (key === 'u') {
        this.vector[c] = U + 0.5 * fdr + (d / 2) * N1 * kT;
      } else {
        const dP = this.pharm;
        const beta = 1 / kT;
        this.vector[c] = dP + pvir + ((beta * dP - rho) / (d * N1)) * fdr;
      }
    });
  }
}
