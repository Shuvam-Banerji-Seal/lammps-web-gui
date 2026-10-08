import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../types';
import { atomIndex, dihedralGeometry, applyDihedral } from '../bonded_util';
import { compileLepton, fillVrefs, type LeptonProgram } from '../../lepton';

/*
 * dihedral_style lepton — docs.lammps.org/dihedral_lepton.html.
 * The expression uses phi, the dihedral angle in radians; the page defines no
 * reference angle or offset (write phi0 into the expression, as the page's
 * example does). dihedral_coeff N "expression" (energy units).
 * Measured with native LAMMPS (black box, expression "phi" on four atoms with
 * torsions of 90, 180 and -90 degrees): phi = 1.5708, 3.1416 and 4.7124, i.e.
 * phi is in [0, 2 pi): a negative torsion is shifted by 2 pi.
 * The angle and its gradient come from bonded_util dihedralGeometry (IUPAC
 * sign: phi = atan2(|b2| b1.n, m.n)).
 */

export class DihedralLepton extends Bonded {
  readonly name = 'lepton';
  readonly kind = 'dihedral' as const;
  private set = new Uint8Array(0);
  private progs: (LeptonProgram | null)[] = [];
  private env: Float64Array[] = [];
  private ctx: StyleContext | null = null;

  settings(args: string[]): void {
    if (args.length) throw new StyleError('dihedral_style lepton takes no arguments');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.set = new Uint8Array(ntypes + 1);
    this.progs = new Array(ntypes + 1).fill(null);
    this.env = new Array(ntypes + 1).fill(null);
  }

  coeff(args: string[], ctx?: StyleContext): void {
    if (ctx) this.ctx = ctx;
    if (args.length !== 2) throw new StyleError('usage: dihedral_coeff N "expression"');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const prog = compileLepton(args[1], { builtins: ['phi'], wrt: ['phi'] });
    for (let t = lo; t <= hi; t++) {
      this.set[t] = 1;
      this.progs[t] = prog;
      this.env[t] = new Float64Array(prog.builtins.length + prog.vrefs.length);
    }
  }

  init(ctx?: StyleContext): void {
    if (ctx) this.ctx = ctx;
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`all dihedral coeffs are not set (type ${t})`);
    }
  }

  compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    const fn = (this.ctx as { equalVariable?: (n: string) => number } | null)?.equalVariable;
    for (let t = 1; t <= this.ntypes; t++) {
      const p = this.progs[t];
      if (p?.vrefs.length && !fn) throw new StyleError('dihedral_style lepton: v_ references need the equal-style variable hook (not available)');
      if (p && fn) fillVrefs(this.env[t]!, p, fn);
    }
    let e = 0;
    for (let q = 0; q < D.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, D.atoms[4 * q + w], 'dihedral'));
      const t = D.type[q];
      const p = this.progs[t]!;
      const env = this.env[t]!;
      let phi = dihedralGeometry(bc, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
      // dihedral_lepton.rst is silent on the branch; measured natively: phi in [0, 2 pi)
      if (phi < 0) phi += 2 * Math.PI;
      env[0] = phi;
      const ed = p.value(env);
      const dEdphi = p.deriv[0](env);
      e += ed;
      applyDihedral(bc, atoms, dEdphi, grad, rel, ed);
    }
    bc.acc.edihed += e;
  }
}
