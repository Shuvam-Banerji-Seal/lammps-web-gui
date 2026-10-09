import { Fix } from './fix';
import { StyleError } from '../force/types';
import { hasCharge, isMolecularStyle } from '../atoms';
import type { System } from '../system';
import { RanPark } from '../rng';
import { num, posInt } from './pour';

/*
 * fix ID group-ID mol/swap N X itype jtype seed T keyword value ...
 * — docs.lammps.org/fix_mol_swap.html (plans/lammps-docs/fix_mol_swap.rst).
 * "This fix performs Monte Carlo swaps of two specified atom types within a
 * randomly selected molecule." Every N steps X swaps are attempted: "For each
 * attempt a single molecule ID is randomly selected." "The range of possible
 * molecule IDs from loID to hiID is pre-computed before each run begins. The
 * loID/hiID is set for the molecule with the smallest/largest ID which has any
 * itype or jtype atoms in it." "Also note that if atoms with molecule ID = 0
 * exist, they are not considered molecules by this fix". "Candidate atoms for
 * swapping must also be in the fix group. Atoms within the selected molecule
 * which are not itype or jtype are ignored." Every itype atom of the molecule
 * becomes jtype and every jtype atom becomes itype.
 *
 * "The potential energy of the entire system is computed before and after each
 * swap is performed within a single molecule. The specified temperature T is
 * used in the Metropolis criterion to accept or reject the attempted swap. If
 * the swap is rejected all swapped values are reversed." "When an atom is
 * swapped from itype to jtype (or vice versa), if charges are defined, the
 * charge values for itype versus jtype atoms are also swapped." "If the *ke*
 * keyword is set to yes, which is the default, and the masses of itype and
 * jtype atoms are different, then when a swap occurs, the velocity of the
 * swapped atom is rescaled by the sqrt of the mass ratio, so as to conserve the
 * kinetic energy of the atom."
 *
 * The fix computes a global vector of length 2 (swap attempts, swap accepts);
 * the page calls the vector values intensive.
 *
 * Random stream (measured with native LAMMPS as a black box, 3 single-atom
 * molecules 1/2/2 with pair eps 0.1/0.5/1.0 at fixed positions, T = 1, N = 1,
 * X = 1, seeds 1 2 3 4 12345 482794 987654321 over 200 attempts): the fix uses
 * RanPark seeded with the seed and advanced by NO discarded draws (fix pour
 * discards 30, this fix discards none). Each attempt draws exactly two
 * uniforms: first the molecule ID as loID + floor(u (hiID - loID + 1)), then,
 * after the before/after energy evaluation, the Metropolis uniform — the
 * acceptance draw is made even when the energy decreases (native draws two
 * uniforms per attempt unconditionally; with the always-draw rule all 7
 * seeds reproduce native's accept sequence exactly, a short-circuit rule does
 * not). A selected molecule that holds no itype or jtype atoms is still an
 * attempt that counts as an accept (its dE is 0), consuming both draws
 * (measured: a 4-molecule system with one empty molecule matched only when the
 * empty selection incremented attempts and accepts).
 *
 * Schedule (measured): the first attempt is on the first step of the first run
 * that follows the fix definition (step + 1), then every N steps, and the
 * counter continues across runs (measured with reset_timestep 100: attempts at
 * 101, 104, 107 for N = 3; and with run 4 then run 6: attempts at 1, 4, 7, 10).
 *
 * ke yes rescaling (measured with masses 1 and 4 in one molecule, T huge so the
 * swap always accepts): an atom 1 -> 2 with v = 1 becomes v = 0.5, an atom
 * 2 -> 1 with v = -1 becomes v = -2, i.e. v_new = v_old sqrt(m_old / m_new);
 * with ke no the velocities are unchanged. Charge swap (measured with
 * atom_style full, q = +1 / -1): the charge follows the type. When the charges
 * of all itype (or all jtype) atoms are not equal, native warns (Cannot swap
 * charges in fix mol/swap) and leaves the charges unchanged (measured).
 *
 * Unsupported: every keyword except ke, itype == jtype, and T <= 0 are
 * StyleErrors (native rejects them as an illegal fix mol/swap command).
 */

/** Saved per-atom values so a rejected attempt restores exactly. */
interface Saved { i: number; type: number; q: number; vx: number; vy: number; vz: number; }

export class FixMolSwap extends Fix {
  readonly style = 'mol/swap';
  private readonly N: number;
  private readonly X: number;
  private readonly itype: number;
  private readonly jtype: number;
  private readonly T: number;
  private readonly ke: boolean;
  private readonly rng: RanPark;
  private loMol = 1;
  private hiMol = 0;
  /** Charges of itype and jtype when every such atom shares one value, else null. */
  private qSwap: [number, number] | null = null;
  private nextStep: number;
  private attempts = 0;
  private accepts = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const s = sys.state;
    if (!isMolecularStyle(s.atomStyle)) {
      throw new StyleError(`fix mol/swap: atom_style ${s.atomStyle} has no molecule IDs`);
    }
    if (args.length < 6) throw new StyleError('usage: fix ID group-ID mol/swap N X itype jtype seed T [ke yes|no]');
    this.N = posInt(args[0], 'N');
    this.X = posInt(args[1], 'X');
    this.itype = posInt(args[2], 'itype');
    this.jtype = posInt(args[3], 'jtype');
    if (this.itype === this.jtype) throw new StyleError('fix mol/swap: itype and jtype must be different types');
    if (this.itype > s.ntypes || this.jtype > s.ntypes) {
      throw new StyleError(`fix mol/swap: type ${Math.max(this.itype, this.jtype)} is larger than ntypes ${s.ntypes}`);
    }
    const seed = posInt(args[4], 'seed');
    if (seed >= 2147483647) throw new StyleError(`fix mol/swap: seed must be a positive integer below 2^31-1, got '${args[4]}'`);
    this.rng = new RanPark(seed);
    this.T = num(args[5], 'T');
    if (!(this.T > 0)) throw new StyleError(`fix mol/swap: T must be > 0, got '${args[5]}'`);
    let ke = true;
    for (let i = 6; i < args.length;) {
      const key = args[i];
      if (key === 'ke') {
        const w = args[i + 1];
        if (w !== 'yes' && w !== 'no') throw new StyleError(`fix mol/swap: ke must be yes or no, got '${w ?? ''}'`);
        ke = w === 'yes';
        i += 2;
      } else {
        throw new StyleError(`fix mol/swap: unknown keyword '${key}'`);
      }
    }
    this.ke = ke;
    this.vectorFlag = true;
    this.sizeVector = 2;
    this.extvector = 0;
    this.nextStep = s.step + 1;
  }

  init(): void {
    const s = this.sys.state;
    // loID/hiID over the molecule IDs that contain any itype or jtype atom; molecule 0 is not a molecule.
    let lo = Number.POSITIVE_INFINITY;
    let hi = 0;
    for (let i = 0; i < s.n; i++) {
      if (s.type[i] !== this.itype && s.type[i] !== this.jtype) continue;
      const m = s.molecule[i];
      if (m <= 0) continue;
      if (m < lo) lo = m;
      if (m > hi) hi = m;
    }
    this.loMol = hi >= 1 ? lo : 1;
    this.hiMol = hi;
    this.chargeValues();
  }

  /**
   * fix_mol_swap.html: the charge of itype versus jtype atoms is swapped, which
   * "requires that all itype atoms in the system have the same charge value.
   * Likewise all jtype atoms in the system must have the same charge value. If
   * this is not the case, LAMMPS issues a warning that it cannot swap charge
   * values."
   */
  private chargeValues(): void {
    const s = this.sys.state;
    if (!hasCharge(s)) { this.qSwap = null; return; }
    const q = [Number.NaN, Number.NaN];
    let bad = false;
    for (let i = 0; i < s.n; i++) {
      const t = s.type[i];
      const k = t === this.itype ? 0 : t === this.jtype ? 1 : -1;
      if (k < 0) continue;
      if (Number.isNaN(q[k])) q[k] = s.q[i];
      else if (q[k] !== s.q[i]) bad = true;
    }
    if (bad) {
      this.sys.warn('Cannot swap charges in fix mol/swap');
      this.qSwap = null;
    } else {
      this.qSwap = [q[0], q[1]];
    }
  }

  /** endOfStep runs every step (nevery = 1); the fix keeps its own N-step schedule. */
  endOfStep(): void {
    const s = this.sys.state;
    if (s.step !== this.nextStep) return;
    for (let x = 0; x < this.X; x++) this.attempt();
    this.nextStep += this.N;
  }

  /** One MC swap attempt: pick a molecule, flip its itype/jtype atoms, Metropolis test. */
  private attempt(): void {
    const s = this.sys.state;
    this.attempts++;
    const span = this.hiMol - this.loMol + 1;
    let mol = this.loMol + Math.floor(this.rng.uniform() * span);
    if (mol > this.hiMol) mol = this.hiMol;
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) {
      if (s.molecule[i] !== mol || (s.mask[i] & this.groupBit) === 0) continue;
      if (s.type[i] === this.itype || s.type[i] === this.jtype) idx.push(i);
    }
    const e0 = this.potentialEnergy();
    const saved: Saved[] = idx.map((i) => ({ i, type: s.type[i], q: s.q[i], vx: s.v[3 * i], vy: s.v[3 * i + 1], vz: s.v[3 * i + 2] }));
    for (const i of idx) this.setType(i, s.type[i] === this.itype ? this.jtype : this.itype);
    const dE = this.potentialEnergy() - e0;
    const u = this.rng.uniform();
    const accept = dE <= 0 || u < Math.exp(-dE / (s.units.boltz * this.T));
    if (accept) this.accepts++;
    else {
      for (const p of saved) {
        s.type[p.i] = p.type;
        s.q[p.i] = p.q;
        s.v[3 * p.i] = p.vx; s.v[3 * p.i + 1] = p.vy; s.v[3 * p.i + 2] = p.vz;
      }
    }
  }

  /** Changes one atom's type, its charge and (ke yes) its velocity, conserving kinetic energy. */
  private setType(i: number, newType: number): void {
    const s = this.sys.state;
    const oldType = s.type[i];
    const mOld = s.rmass ? s.rmass[i] : s.massByType[oldType];
    s.type[i] = newType;
    if (this.qSwap) s.q[i] = newType === this.itype ? this.qSwap[0] : this.qSwap[1];
    if (this.ke) {
      const mNew = s.rmass ? s.rmass[i] : s.massByType[newType];
      if (mNew !== mOld && mNew > 0 && mOld > 0) {
        const f = Math.sqrt(mOld / mNew);
        s.v[3 * i] *= f; s.v[3 * i + 1] *= f; s.v[3 * i + 2] *= f;
      }
    }
  }

  /** Potential energy of the whole system, including fix contributions (fix_modify energy yes). */
  private potentialEnergy(): number {
    const sys = this.sys;
    sys.bump();
    const a = sys.forces();
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp + sys.fixEnergy();
  }

  computeVector(i: number): number {
    return i === 0 ? this.attempts : this.accepts;
  }
}
