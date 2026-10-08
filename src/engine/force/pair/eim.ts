import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { erfcExact } from '../erfc';

/*
 * pair_style eim — embedded-ion method (EIM) potentials, docs.lammps.org/pair_eim.html
 * (source: plans/lammps-docs/pair_eim.rst)
 *
 * Syntax: pair_style eim, with no arguments.
 * Description: "computes pairwise interactions for ionic compounds using embedded-ion method".
 * Energy (the doc's formulas, E = 1/2 sum_i sum_j phi_ij + sum_i E_i):
 *   E = 1/2 sum_i sum_{j in N(i)} phi_ij(r_ij) + sum_i E_i(q_i, sigma_i)
 *   q_i     = sum_{j in N(i)} eta_ji(r_ij)
 *   sigma_i = sum_{j in N(i)} q_j psi_ij(r_ij)
 *   E_i     = 1/2 q_i sigma_i
 *   eta_ji  = A_eta,ij (chi_j - chi_i) fc(r, r_s,eta, r_c,eta)
 *   psi_ij  = A_psi,ij exp(-zeta_ij r) fc(r, r_s,psi, r_c,psi)
 *   phi_ij  = T_a fc(r, r_e, r_c,a) - T_b fc(r, r_e, r_c,b), with
 *   T_a = E_b beta/(beta-alpha) exp(-alpha (r-r_e)/r_e),  T_b = E_b alpha/(beta-alpha) exp(-beta (r-r_e)/r_e)   (p = 1)
 *   T_a = E_b beta/(beta-alpha) (r_e/r)^alpha,            T_b = E_b alpha/(beta-alpha) (r_e/r)^beta            (p = 2)
 * Each term is zero at and beyond its own cutoff (r_c,a for T_a, r_c,b for T_b). The doc's cutoff function is
 *   f_c(r, r_p, r_c) = 0.510204 erfc[1.64498 (2r - r_p - r_c)/(r_c - r_p)] - 0.010204,
 * generalised below to the file's global values (see the fc method).
 * The doc: "you do not assign charges explicitly." (the charges come from the q_i sum above).
 *
 * Potential file: "Lines starting with # are comments and are ignored by LAMMPS."
 * "The second and third values are related to the cutoff function"; the constants
 * "can be derived from these values."; "The lines in the file can be in any order";
 * "the EIM potential file lists atomic masses"; "This file is parameterized in terms of LAMMPS"
 * metal units.
 *   global:  G1 G2 G3          (G1 = cation/anion electronegativity split)
 *   element: name Z mass chi r_atomic r_ionic E_cohesive q0
 *   pair:    el1 el2 r_c,phi r_c,phi E_b r_e alpha beta r_c,eta A_eta r_s,eta r_c,psi A_psi zeta r_s,psi p
 *
 * Mapping: "pair_coeff * * Na Cl ffield.eim Na Na Na Cl" (element names, then the file, then
 * one element name per atom type); "The first 2 arguments must be \* \*"; "If a mapping
 * value is specified as NULL, the mapping is not performed."
 *
 * The two r_c,phi values: the doc calls them "redundant for historical reasons", but measured with
 * native LAMMPS (black box), the first value is r_c,b (gates the beta term) and the second is r_c,a
 * (gates the alpha term). Dimers with first 5.6 / second 4.0 and first 4.0 / second 5.6 match this
 * split form to about 1e-15 in energy for p = 1 and p = 2.
 *
 * Cutoff function, measured with native LAMMPS (black box): with the general form
 *   fc(r, r_p, r_c) = (erfc(a(r)) - erfc(G3)) / (erfc(G2) - erfc(G3)),
 *   a(r) = G2 + (G3 - G2)(r - r_p)/(r_c - r_p),
 * the doc's constants are the case G2 = -G3, G3 = 1.645 (fc = 1 at r_p, 0 at r_c). Probes with
 * G2 = -1.645, G3 = 1.2 and G2 = -1.0, G3 = 1.2 agree with this form to about 1e-13 in
 * energy; the rounded doc constants are off by about 1e-6 there.
 *
 * Global G1: measured with native LAMMPS (black box), a three-element cluster gives the same
 * energy for G1 = 0.5, 1.95, 2.0 and 3.5, so G1 is parsed and not used.
 * Units: measured with native LAMMPS (black box), the same energies come out with units metal,
 * real and si; the formulas are unitless, so no conversion is applied.
 * Masses: the file masses replace an earlier mass command (measured with native LAMMPS: a
 * kinetic-energy check matches a run without the mass command).
 * q0: measured with native LAMMPS, q0 = 0.5 gives the same energy, so the column is not used.
 * Pair lines: measured with native LAMMPS, the element order of a pair line does not matter.
 * Element names: measured with native LAMMPS, the number of names before the file must equal
 * the number of distinct non-NULL mapped element names; the names themselves are not checked
 * against the file. A pair missing from the file is an error.
 *
 * Near-cutoff behaviour: measured with native LAMMPS (black box), within about 0.01 A inside or
 * outside a cutoff the native energy departs from the formulas above (by a few percent of the
 * small cutoff tail, up to about 1e-7 in energy for charged clusters). This engine follows the
 * formulas; the oracle cases keep every pair farther than 0.01 A from its cutoffs.
 *
 * Forces: with G_a = dE/dq_a = sigma_a (the psi pair sum is symmetric), each ordered
 * neighbour term (a, j) of the full list contributes the scalar dE/dr
 *   1/2 phi'(r) + 1/2 q_a q_j psi'(r) + sigma_a A_aj (chi_j - chi_a) fc_eta'(r)
 * applied as f_a -= dE/dr u, f_j += dE/dr u with u = (x_a - x_j)/r. The virial is the
 * force-field sum x.f (virialFdotr), so ghost forces are reverse-communicated.
 */

const SQRT_PI = Math.sqrt(Math.PI);
const TWO_OVER_SQRT_PI = 2 / SQRT_PI;

interface EimElement {
  name: string;
  mass: number;
  chi: number;
}

interface EimPairFile {
  e1: string;
  e2: string;
  /** r_c,phi: the first value of the pair line; it gates the beta term of phi. */
  rcBeta: number;
  /** The second r_c,phi value; it gates the alpha term of phi (see the header). */
  rcAlpha: number;
  Eb: number;
  re: number;
  alpha: number;
  beta: number;
  rcEta: number;
  Aeta: number;
  rsEta: number;
  rcPsi: number;
  Apsi: number;
  zeta: number;
  rsPsi: number;
  p: number;
}

interface EimFile {
  name: string;
  G1: number;
  G2: number;
  G3: number;
  elements: Map<string, EimElement>;
  /** Keyed by "A B" in both orders. */
  pairs: Map<string, EimPairFile>;
}

const PAIR_NUMBERS = 14;
const ELEMENT_NUMBERS = 7;

/** Joins continuation lines ending in & and strips comments starting with # (the ffield.eim files use both). */
const logicalLines = (text: string): string[] => {
  const out: string[] = [];
  let acc = '';
  for (const raw of text.split('\n')) {
    const noComment = raw.replace(/#.*/, '').trim();
    if (noComment === '') {
      if (acc) { out.push(acc); acc = ''; }
      continue;
    }
    if (noComment.endsWith('&')) {
      acc += `${noComment.slice(0, -1)} `;
      continue;
    }
    out.push(acc + noComment);
    acc = '';
  }
  if (acc) out.push(acc);
  return out;
};

const num = (w: string, what: string, file: string): number => {
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`EIM potential file ${file}: expected a number for ${what}, got '${w}'`);
  return v;
};

export const parseEimFile = (text: string, file: string): EimFile => {
  let G: number[] | null = null;
  const elements = new Map<string, EimElement>();
  const pairs = new Map<string, EimPairFile>();
  for (const line of logicalLines(text)) {
    const w = line.split(/\s+/);
    const key = w[0];
    if (key === 'global:') {
      if (w.length !== 4) throw new StyleError(`EIM potential file ${file}: 'global:' needs 3 values, got ${w.length - 1}`);
      G = w.slice(1).map((v, k) => num(v, `global value ${k + 1}`, file));
    } else if (key === 'element:') {
      if (w.length !== 2 + ELEMENT_NUMBERS) {
        throw new StyleError(`EIM potential file ${file}: 'element:' line for ${w[1]} needs ${ELEMENT_NUMBERS} values after the name`);
      }
      const name = w[1];
      const mass = num(w[3], `mass of ${name}`, file);
      const chi = num(w[4], `electronegativity of ${name}`, file);
      if (elements.has(name)) throw new StyleError(`EIM potential file ${file}: duplicate element ${name}`);
      elements.set(name, { name, mass, chi });
    } else if (key === 'pair:') {
      if (w.length !== 3 + PAIR_NUMBERS) {
        throw new StyleError(`EIM potential file ${file}: 'pair:' line for ${w[1]} ${w[2]} needs ${PAIR_NUMBERS} values`);
      }
      const v = w.slice(3).map((x, k) => num(x, `pair value ${k + 1} of ${w[1]} ${w[2]}`, file));
      const p: EimPairFile = {
        e1: w[1], e2: w[2],
        rcBeta: v[0], rcAlpha: v[1], Eb: v[2], re: v[3], alpha: v[4], beta: v[5],
        rcEta: v[6], Aeta: v[7], rsEta: v[8],
        rcPsi: v[9], Apsi: v[10], zeta: v[11], rsPsi: v[12], p: v[13],
      };
      if (p.p !== 1 && p.p !== 2) throw new StyleError(`EIM potential file ${file}: pair ${w[1]} ${w[2]} has p = ${p.p} (must be 1 or 2)`);
      if (pairs.has(`${w[1]} ${w[2]}`)) throw new StyleError(`EIM potential file ${file}: duplicate pair ${w[1]} ${w[2]}`);
      pairs.set(`${w[1]} ${w[2]}`, p);
      if (w[1] !== w[2]) pairs.set(`${w[2]} ${w[1]}`, p);
    } else {
      throw new StyleError(`EIM potential file ${file}: unknown line '${key}'`);
    }
  }
  if (!G) throw new StyleError(`EIM potential file ${file} has no global: line`);
  if (elements.size === 0) throw new StyleError(`EIM potential file ${file} has no element: lines`);
  return { name: file, G1: G[0], G2: G[1], G3: G[2], elements, pairs };
};

/** Sets the type mass from the file (the file masses replace earlier mass commands, see the header). */
const applyFileMass = (ctx: StyleContext, type: number, mass: number, name: string): void => {
  const s = ctx.s;
  if (!s || s.rmass) return;
  if (mass <= 0) {
    ctx.log(`WARNING: EIM potential file ${name} uses a dummy mass of 0.0 for type ${type}`);
    if (Number.isNaN(s.massByType[type])) s.massByType[type] = 1.0;
    return;
  }
  s.massByType[type] = mass;
};

/**
 * pair_style eim. Atom types are mapped to EIM elements by pair_coeff; the
 * cutoffs come from the potential file. Forces are the analytic gradient of the
 * documented energy (see the header comment).
 */
export class PairEIM extends Pair {
  readonly name = 'eim';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private file: EimFile | null = null;
  private readonly fileCache = new Map<string, EimFile>();
  /** Element name mapped to each atom type (null for NULL). */
  private nameOf: (string | null)[] = [];
  private chiOf: Float64Array = new Float64Array(0);
  /** Resolved pair parameters per type pair (null for NULL types), indexed i * nt + j. */
  private pairs: (EimPairFile | null)[] = [];
  private cutErfc3 = 0;
  private cutDenom = 1;
  private qBuf = new Float64Array(0);
  private sigmaBuf = new Float64Array(0);

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('usage: pair_style eim (no arguments)');
  }

  override modify(key: string, _values: string[]): number {
    throw new StyleError(`pair_modify ${key} is not supported for pair style ${this.name}`);
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nameOf = new Array(ntypes + 1).fill(null);
    this.chiOf = new Float64Array(ntypes + 1);
    this.pairs = new Array((ntypes + 1) * (ntypes + 1)).fill(null);
  }

  override initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify shift and tail options`);
    }
    if (this.table !== 12) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify table option`);
    }
    if (!this.file) throw new StyleError(`pair_style ${this.name} needs a pair_coeff command with an EIM potential file`);
    // cutoff function constants: erfc(G3) and erfc(G2) - erfc(G3) (see the header comment)
    this.cutErfc3 = erfcExact(this.file.G3);
    this.cutDenom = erfcExact(this.file.G2) - this.cutErfc3;
    if (!(this.cutDenom > 0)) throw new StyleError(`EIM potential file ${this.file.name}: global values G2 and G3 do not define a cutoff function`);
  }

  /**
   * pair_coeff * * El1 ... ElK file Map1 ... MapN. K must equal the number of
   * distinct non-NULL mapped element names (measured, see the header comment).
   */
  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Pair_coeff must start with * * for pair style eim');
    const rest = args.slice(2);
    const mapping = rest.slice(rest.length - this.ntypes);
    const distinct = new Set(mapping.filter((e) => e !== 'NULL'));
    const k = distinct.size;
    if (rest.length !== k + 1 + this.ntypes) {
      throw new StyleError(
        `pair_coeff for style eim: expected ${k} element names, the potential file, then ${this.ntypes} type mappings (got ${rest.length} arguments)`,
      );
    }
    const filename = rest[k];
    const cached = this.fileCache.get(filename);
    const file = cached ?? parseEimFile(ctx.readFile(filename), filename);
    this.fileCache.set(filename, file);
    this.file = file;
    for (let t = 1; t <= this.ntypes; t++) {
      const name = mapping[t - 1];
      if (name === 'NULL') {
        this.nameOf[t] = null;
        continue;
      }
      const el = file.elements.get(name);
      if (!el) throw new StyleError(`element '${name}' is not in EIM potential file ${filename}`);
      this.chiOf[t] = el.chi;
      this.nameOf[t] = name;
      applyFileMass(ctx, t, el.mass, filename);
    }
  }

  override initOne(i: number, j: number): number {
    const file = this.file;
    const a = this.nameOf[i], b = this.nameOf[j];
    if (!file || a === null || b === null || a === undefined || b === undefined) {
      throw new StyleError(`all pair coeffs are not set (pair ${i} ${j}: NULL element mapping for pair style eim)`);
    }
    const p = file.pairs.get(`${a} ${b}`);
    if (!p) throw new StyleError(`Element pair (${a}, ${b}) is not defined in EIM potential file ${file.name}`);
    if (p.rcBeta === p.re || p.rcAlpha === p.re) throw new StyleError(`EIM pair ${a} ${b}: r_c,phi must differ from r_e`);
    if (p.rcEta > 0 && p.rcEta === p.rsEta) throw new StyleError(`EIM pair ${a} ${b}: r_c,eta must differ from r_s,eta`);
    if (p.rcPsi > 0 && p.rcPsi === p.rsPsi) throw new StyleError(`EIM pair ${a} ${b}: r_c,psi must differ from r_s,psi`);
    const rcMax = Math.max(p.rcBeta, p.rcAlpha, p.rcEta, p.rcPsi);
    const nt = this.ntypes + 1;
    this.pairs[i * nt + j] = p;
    this.pairs[j * nt + i] = p;
    return rcMax;
  }

  /** The cutoff function fc(r, r_p, r_c) with the global values (header comment). */
  private fc(r: number, rp: number, rc: number): number {
    const g2 = this.file!.G2, g3 = this.file!.G3;
    const arg = g2 + ((g3 - g2) * (r - rp)) / (rc - rp);
    return (erfcExact(arg) - this.cutErfc3) / this.cutDenom;
  }

  /** d fc / dr. */
  private dfc(r: number, rp: number, rc: number): number {
    const g2 = this.file!.G2, g3 = this.file!.G3;
    const arg = g2 + ((g3 - g2) * (r - rp)) / (rc - rp);
    return (-TWO_OVER_SQRT_PI * Math.exp(-arg * arg) * (g3 - g2)) / ((rc - rp) * this.cutDenom);
  }

  /**
   * The two terms of phi without their cutoff functions: the alpha term Ta and the beta term Tb
   * (phi = Ta fc(r, r_e, r_c,alpha) - Tb fc(r, r_e, r_c,beta)), with their derivatives.
   */
  private static phiTerms(p: EimPairFile, r: number): [number, number, number, number] {
    const { Eb, re, alpha: al, beta: be } = p;
    const pre = (Eb * be) / (be - al), pre2 = (Eb * al) / (be - al);
    if (p.p === 1) {
      const ea = Math.exp((-al * (r - re)) / re), eb = Math.exp((-be * (r - re)) / re);
      return [pre * ea, pre * ea * (-al / re), pre2 * eb, pre2 * eb * (-be / re)];
    }
    const xa = Math.pow(re / r, al), xb = Math.pow(re / r, be);
    return [pre * xa, pre * xa * (-al / r), pre2 * xb, pre2 * xb * (-be / r)];
  }

  compute(pc: PairCompute): void {
    const full = pc.full;
    if (!full) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nall = pc.nall;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    if (this.qBuf.length !== nall) this.qBuf = new Float64Array(nall);
    if (this.sigmaBuf.length !== nlocal) this.sigmaBuf = new Float64Array(nlocal);
    const q = this.qBuf;
    const sigma = this.sigmaBuf;
    const eatom = pc.eatom;
    const vatom = pc.vatom;
    let evdwl = 0;

    // pass 1: charges q_i = sum_j eta_ji(r_ij) of the owned atoms
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      let qi = 0;
      const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = full.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        const tp = this.pairs[ti * nt + tj];
        if (!tp || tp.rcEta <= 0) continue;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti * nt + tj]) continue;
        const r = Math.sqrt(rsq);
        if (r >= tp.rcEta) continue;
        qi += tp.Aeta * (this.chiOf[tj] - this.chiOf[ti]) * this.fc(r, tp.rsEta, tp.rcEta);
      }
      q[i] = qi;
    }
    pc.nb.forwardCopy(q, 1);

    // pass 2: sigma_i = sum_j q_j psi_ij and the embedding energies 1/2 q_i sigma_i
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      let s = 0;
      const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = full.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        const tp = this.pairs[ti * nt + tj];
        if (!tp || tp.rcPsi <= 0) continue;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti * nt + tj]) continue;
        const r = Math.sqrt(rsq);
        if (r >= tp.rcPsi) continue;
        s += q[j] * tp.Apsi * Math.exp(-tp.zeta * r) * this.fc(r, tp.rsPsi, tp.rcPsi);
      }
      sigma[i] = s;
      const e = 0.5 * q[i] * s;
      evdwl += e;
      if (eatom) eatom[i] += e;
    }

    // pass 3: pair term and the chain-rule terms of q_i and psi
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = full.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        const tp = this.pairs[ti * nt + tj];
        if (!tp) continue;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti * nt + tj]) continue;
        const r = Math.sqrt(rsq);
        const p = tp;
        let dEdr = 0;
        if (r < p.rcAlpha || r < p.rcBeta) {
          const [ta, dta, tb, dtb] = PairEIM.phiTerms(p, r);
          let V = 0, dV = 0;
          if (r < p.rcAlpha) {
            const c = this.fc(r, p.re, p.rcAlpha);
            V += ta * c;
            dV += dta * c + ta * this.dfc(r, p.re, p.rcAlpha);
          }
          if (r < p.rcBeta) {
            const c = this.fc(r, p.re, p.rcBeta);
            V -= tb * c;
            dV -= dtb * c + tb * this.dfc(r, p.re, p.rcBeta);
          }
          evdwl += 0.5 * V;
          if (eatom) eatom[i] += 0.5 * V;
          dEdr += 0.5 * dV;
        }
        if (r < p.rcPsi) {
          const ex = p.Apsi * Math.exp(-p.zeta * r);
          const c = this.fc(r, p.rsPsi, p.rcPsi);
          const dpsi = ex * (-p.zeta * c + this.dfc(r, p.rsPsi, p.rcPsi));
          dEdr += 0.5 * q[i] * q[j] * dpsi;
        }
        if (p.rcEta > 0 && r < p.rcEta) {
          const dq = p.Aeta * (this.chiOf[tj] - this.chiOf[ti]) * this.dfc(r, p.rsEta, p.rcEta);
          dEdr += sigma[i] * dq;
        }
        if (dEdr === 0) continue;
        const sx = (dEdr * dx) / r, sy = (dEdr * dy) / r, sz = (dEdr * dz) / r;
        f[3 * i] -= sx; f[3 * i + 1] -= sy; f[3 * i + 2] -= sz;
        f[3 * j] += sx; f[3 * j + 1] += sy; f[3 * j + 2] += sz;
        if (vatom) {
          // force on i is A dx with A = -dE/dr / r; each atom takes half of A dx dx (as pair_eam)
          const h = (-0.5 * dEdr) / r;
          const v0 = h * dx * dx, v1 = h * dy * dy, v2 = h * dz * dz, v3 = h * dx * dy, v4 = h * dx * dz, v5 = h * dy * dz;
          vatom[6 * i] += v0; vatom[6 * i + 1] += v1; vatom[6 * i + 2] += v2;
          vatom[6 * i + 3] += v3; vatom[6 * i + 4] += v4; vatom[6 * i + 5] += v5;
          vatom[6 * j] += v0; vatom[6 * j + 1] += v1; vatom[6 * j + 2] += v2;
          vatom[6 * j + 3] += v3; vatom[6 * j + 4] += v4; vatom[6 * j + 5] += v5;
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
