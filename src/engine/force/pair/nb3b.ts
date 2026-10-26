/*
 * pair_style nb3b/harmonic and nb3b/screened — docs.lammps.org/pair_nb3b.html
 * (source: plans/lammps-docs/pair_nb3b.rst)
 *
 * Syntax (verbatim): "pair_style style", with "style = *nb3b/harmonic* or *nb3b/screened*".
 *
 * Energy (verbatim, from the Description): "E = K (\theta - \theta_0)^2" and
 * "The form of the potential is identical to that used in angle_style *harmonic*,
 * but in this case, the atoms do not need to be explicitly bonded."  The
 * 1/2 factor: "Note that the usual 1/2 factor is included in *K*" — so there
 * is no extra 1/2 here.  The sum runs over every owned central atom and every
 * unordered pair of its neighbours (center-based, so each angle is counted
 * once; the engine uses the full neighbour list).
 *
 * Potential file (verbatim): "Lines that are not blank or comments (starting
 * with #) define parameters for a triplet of elements."  "Each entry has six
 * arguments. The first three are atom types as referenced in the LAMMPS input
 * file. The first argument specifies the central atom. The fourth argument
 * indicates the *K* parameter. The fifth argument indicates :math:`\theta_0`.
 * The sixth argument indicates a separation cutoff in Angstroms."
 * "For a given entry, if the second and third arguments are identical, then the
 * entry is for a cutoff for the distance between types 1 and 2 (values for *K*
 * and :math:`\theta_0` are irrelevant in this case)."
 * "For a given entry, if the first three arguments are all different, then the
 * entry is for the *K* and :math:`\theta_0` parameters (the cutoff in this case
 * is irrelevant)."
 * "It is required that the potential file contains entries for *all*
 * permutations of the elements listed in the pair_coeff command."  "If certain
 * combinations are not parameterized the corresponding parameters should be set
 * to zero."
 *
 * Measured with native LAMMPS (black box), units of theta0 and K:
 *   - the fifth column is in degrees and the angle is in radians:
 *     K = 1, theta0 = 110, theta = 90 deg gives E = 0.121846967914683 =
 *     (pi/2 - 110 pi/180)^2, i.e. K per rad^2 with theta0 converted from degrees.
 * Measured: the entry (c, a, b) with a != b supplies K and theta0 for the angle
 *   at center c between neighbours of types a and b; the order used is by atom
 *   type index (the neighbour of the lower type is the second argument), not by
 *   atom id or element name: with Mg=1, O=2, H=3, the angle O-Mg-H used the
 *   Mg O H entry whether O or H was created first.
 * Measured: the entry (c, a, a) supplies the cutoff of the c-a leg (r <= cut is
 *   kept; r = 2.0 with cut = 2.0 contributes). The (O, Mg, Mg) entry did not set
 *   the Mg-O leg cutoff; a zero cut in (Mg, O, O) removed the term.
 * Measured: the entry (c, a, a) K and theta0 also apply to an angle between two
 *   neighbours of the same type a: O-Mg-O at 120 deg with (Mg, O, O) = 1, 110
 *   gave 0.0304617419786709 = (10 deg in rad)^2. With K = 0 it gave 0.
 * Measured: the O-H distance cutoff of the entry (O, H, H) played no role in the
 *   Mg-centered angle. The cut column of the (a, b, c) entries with a != b is
 *   ignored, as the doc says it is irrelevant.
 * Measured: a missing combination is an error (Potential file is missing an
 *   entry for: Mg Mg Mg; all ordered triples of the mapped elements are needed).
 * Measured: an energy with two O and one H around Mg (K = 0 for O-Mg-O) equals
 *   the sum of the two O-Mg-H terms: 0.243693935829367 = 2 x 0.121846967914683.
 *
 * pair_coeff (verbatim): "Only a single pair_coeff command is used with these
 * styles which specifies a potential file with parameters for specified
 * elements."  "The first 2 arguments must be \* \* so as to span all LAMMPS atom
 * types."  "If a mapping value is specified as NULL, the mapping is not
 * performed."  NULL types take no part in the three-body term.
 *
 * nb3b/screened: the energy is the harmonic term times exp(-r_ij/rho_ij - r_ik/rho_ik),
 * with the bond lengths of the two legs of the angle (doc, verbatim: "where
 * :math:`\rho_ij` and :math:`\rho_ik` are the screening factors along the two bonds").
 * The potential file has one more column than the harmonic file: the README
 * (examples/nb3b/README.nb3b.screened) lists the column header # i j k K theta0 rho cutoff,
 * and the rho column is the screening factor of the (i j j) entry; the README says
 * the rho value must be higher than 0.0  The cutoff and rho of a leg
 * come from the (i j j) entry, and K and theta0 from the (i j k) entry.
 *
 * Measured with native LAMMPS (black box), nb3b/screened, on 3-atom and 4-atom
 * clusters with a hand-written file of the same layout (center O, neighbours P):
 *   - the first argument i is the central (vertex) atom: O-centred angles give
 *     energy, P-centred ones with K = 0 give 0.
 *   - E = K (theta - theta0)^2 exp(-r_ij/rho_ij - r_ik/rho_ik) with rho_ij from the
 *     (i j j) entry (the P leg uses (O P P), the O leg uses (O O O)); changing one
 *     bond length at fixed angle changed the energy exactly as this formula gives
 *     (checked on several geometries against the formula).
 *   - a leg is kept while r <= cut of its (i j j) entry: r = 3.3 with cut 3.3 is kept,
 *     r = 3.300001 is not. The energy is not shifted at the cutoff (the value at
 *     r = 3.29 equals the unshifted formula). A cut of the (i j k) entry, e.g. (O O P)
 *     with cut 1.0, did not remove an O leg of length 1.5 whose (O O O) cut is 2.8.
 *   - each unordered neighbour pair is counted once: three P neighbours of O gave
 *     the sum over the three pairs (the twice-counted value was exactly double).
 *   - theta0 is in degrees and K multiplies (theta - theta0)^2 directly.
 *   - with K and theta0 symmetric in the two orders (PSiO: O P Si and O Si P both
 *     60.0 and 109.47) the energy is fixed by the parameters alone.
 *   - when (i j k) and (i k j) have different K or theta0 and one K is nonzero, the
 *     native energy equals the single-entry value of one of the two entries, and
 *     which one changes with the geometry: with the default atom sort the creation
 *     order and the mirror image did not change the choice, but the neighbour list
 *     method did (neighbor nsq instead of bin), and the choice switched within a
 *     scan of one bond angle. Such a potential is refused by this engine, because
 *     the choice is not a property of the parameters.
 *   - errors (native text, paraphrased): a missing (i j k) combination is reported as
 *     Potential file ... is missing an entry for: O P P; an element count that does
 *     not match the atom types is an error; a 6-column entry line reads the next word
 *     as a number and fails (Not a valid floating-point number); rho = 0 fails
 *     (Incorrect value for potential parameter: 'rho'); a negative cutoff fails
 *     (Illegal nb3b/screened parameter). An entry line with more words than 7
 *     is accepted and the extra words are ignored.
 *   - a NULL-mapped type is neither centre nor neighbour: a hybrid/overlay run with
 *     a NULL neighbour gave the same energy as the run without that atom; a NULL
 *     centre gave 0. Standalone (no other style filling the NULL pairs) a NULL type
 *     fails with All pair coeffs are not set.
 *
 * Forces: the angle derivative uses d theta/d c = -1/sin(theta) with the
 * cosine clamped to [-1, 1]; sin(theta) is floored at 1e-8 (an engine guard
 * for exactly collinear triplets, where the force stays finite because the
 * perpendicular part of dc/du vanishes). For nb3b/screened, dE/du also has the
 * radial part -K (theta - theta0)^2 exp(...) / rho_ij along the leg. The global
 * virial comes from virialFdotr; per-atom energy and virial are split evenly
 * over the three atoms.
 */
import { Pair, StyleError, type PairCompute, type StyleContext, potentialFile } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum, joinPotentialEntries } from '../util';
import { libm } from '../../libm';

interface NB3BEntry {
  e1: string;
  e2: string;
  e3: string;
  K: number;
  /** theta0 in degrees, as in the file. */
  theta0: number;
  /** Screening length rho (nb3b/screened file only; 0 in a harmonic file). */
  rho: number;
  cut: number;
}

const key3 = (a: string, b: string, c: string): string => `${a} ${b} ${c}`;

/**
 * Parses an nb3b file. A harmonic file has 6 columns (i j k K theta0 cutoff), a screened
 * file 7 (i j k K theta0 rho cutoff). Lines are joined to ncols words, as the reader does.
 */
const parseNB3BFile = (text: string, fileName: string, ncols: number): Map<string, NB3BEntry> => {
  const layout = ncols === 7 ? 'element1 element2 element3 K theta0 rho cutoff' : 'element1 element2 element3 K theta0 cutoff';
  const lines = joinPotentialEntries(text.split(/\r?\n/), ncols);
  const entries = new Map<string, NB3BEntry>();
  for (let ln = 0; ln < lines.length; ln++) {
    const raw = lines[ln];
    const hash = raw.indexOf('#');
    const line = (hash >= 0 ? raw.slice(0, hash) : raw).trim();
    if (line === '') continue;
    const t = line.split(/\s+/);
    if (t.length !== ncols) {
      throw new StyleError(
        `nb3b potential file ${fileName} line ${ln + 1}: expected '${layout}' (${ncols} values), got ${t.length}`,
      );
    }
    const k = key3(t[0], t[1], t[2]);
    if (entries.has(k)) throw new StyleError(`nb3b potential file ${fileName} line ${ln + 1}: duplicate entry for elements ${k}`);
    const K = parseNum(t[3], `K of the nb3b file entry ${k}`);
    const theta0 = parseNum(t[4], `theta0 of the nb3b file entry ${k}`);
    const rho = ncols === 7 ? parseNum(t[5], `rho of the nb3b file entry ${k}`) : 0;
    const cut = parseNum(t[ncols - 1], `cutoff of the nb3b file entry ${k}`);
    if (cut < 0) throw new StyleError(`nb3b potential file ${fileName} entry ${k}: cutoff must be >= 0`);
    // the rho value must be higher than 0.0 (examples/nb3b/README.nb3b.screened)
    if (ncols === 7 && !(rho > 0)) throw new StyleError(`nb3b potential file ${fileName} entry ${k}: rho must be > 0 (got ${t[5]})`);
    entries.set(k, { e1: t[0], e2: t[1], e3: t[2], K, theta0, rho, cut });
  }
  if (entries.size === 0) throw new StyleError(`nb3b potential file ${fileName} contains no parameter entries`);
  return entries;
};

/**
 * pair_style nb3b/harmonic: E = K (theta - theta0)^2 over the angles of neighbour pairs.
 * nb3b/screened (subclass below) multiplies each angle by exp(-r_ij/rho_ij - r_ik/rho_ik).
 */
export class PairNB3BHarmonic extends Pair {
  readonly name: string = 'nb3b/harmonic';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  /** true for nb3b/screened: the 7-column file and the screening factor. */
  protected screened = false;
  protected fileName = '';
  protected fileRead = false;
  /** type -> element index, -1 for NULL types. */
  protected elemOf = new Int32Array(0);
  /** Angle parameters, indexed by (center, lo, hi) with lo <= hi (types): K and theta0 in radians. */
  protected kTab = new Float64Array(0);
  protected thTab = new Float64Array(0);
  /** Leg cutoff of center c to neighbour type t: the cutoff of the (c, t, t) entry. */
  protected legCut = new Float64Array(0);
  /** Leg screening length of center c to neighbour type t: the rho of the (c, t, t) entry (screened only). */
  protected legRho = new Float64Array(0);

  // scratch for the gathered neighbours of one center
  private cap = 0;
  private gIdx = new Int32Array(0);
  private gDx = new Float64Array(0);
  private gDy = new Float64Array(0);
  private gDz = new Float64Array(0);
  private gR = new Float64Array(0);

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length > 0) {
      throw new StyleError(`pair_style ${this.name} takes no arguments (got '${args.join(' ')}')`);
    }
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.elemOf = new Int32Array(nt).fill(-1);
    this.kTab = new Float64Array(nt * nt * nt);
    this.thTab = new Float64Array(nt * nt * nt);
    this.legCut = new Float64Array(nt * nt);
    this.legRho = new Float64Array(nt * nt);
    this.fileRead = false;
    this.fileName = '';
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    // Pair::map_element2type(narg - 3, ...) runs first: fewer than three words is a wrong element count
    if (args.length < 3) throw new StyleError('Number of element to type mappings does not match number of atom types');
    if (args[0] !== '*' || args[1] !== '*') {
      throw new StyleError(`the first 2 arguments of pair_coeff for style ${this.name} must be * *`);
    }
    if (this.fileRead) throw new StyleError(`pair_style ${this.name}: only a single pair_coeff command is allowed`);
    const filename = args[2];
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    // native names the harmonic file "nb3n/harmonic" (src/MANYBODY/pair_nb3b_harmonic.cpp at b332f28)
    const entries = parseNB3BFile(ctx.readFile(filename, potentialFile(this.screened ? 'nb3b/screened' : 'nb3n/harmonic')), filename, this.screened ? 7 : 6);
    this.fileName = filename;
    const names: string[] = [];
    const nt = this.ntypes + 1;
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') {
        this.elemOf[t] = -1;
        continue;
      }
      if (![...entries.values()].some((e) => e.e1 === name || e.e2 === name || e.e3 === name)) {
        throw new StyleError(`element '${name}' is not in nb3b potential file ${filename}`);
      }
      let idx = names.indexOf(name);
      if (idx < 0) {
        idx = names.length;
        names.push(name);
      }
      this.elemOf[t] = idx;
    }
    // "It is required that the potential file contains entries for *all* permutations"
    for (const a of names) for (const b of names) for (const c of names) {
      if (!entries.has(key3(a, b, c))) {
        throw new StyleError(`Potential file ${filename} is missing an entry for: ${a} ${b} ${c}`);
      }
    }
    if (this.screened) {
      // Measured with native LAMMPS (black box): for (i j k) and (i k j) with different K or theta0 the
      // energy follows one of the two entries, chosen by the neighbour-list order, so it is not a
      // function of the parameters. Only pairs with a zero K on both sides are order-independent.
      for (let p = 0; p < names.length; p++) for (let q = p + 1; q < names.length; q++) {
        for (const c of names) {
          const e1 = entries.get(key3(c, names[p], names[q]))!, e2 = entries.get(key3(c, names[q], names[p]))!;
          if ((e1.K !== e2.K || e1.theta0 !== e2.theta0) && (e1.K !== 0 || e2.K !== 0)) {
            throw new StyleError(
              `nb3b/screened: entries ${key3(c, names[p], names[q])} and ${key3(c, names[q], names[p])} have different K or theta0; ` +
                'native LAMMPS picks one of them by neighbour order for such a pair, which this engine does not reproduce',
            );
          }
        }
      }
    }
    const DEG = Math.PI / 180;
    for (let tc = 1; tc <= this.ntypes; tc++) {
      if (this.elemOf[tc] < 0) continue;
      for (let ta = 1; ta <= this.ntypes; ta++) {
        if (this.elemOf[ta] < 0) continue;
        const leg = entries.get(key3(names[this.elemOf[tc]], names[this.elemOf[ta]], names[this.elemOf[ta]]))!;
        this.legCut[tc * nt + ta] = leg.cut;
        this.legRho[tc * nt + ta] = leg.rho;
        for (let tb = ta; tb <= this.ntypes; tb++) {
          if (this.elemOf[tb] < 0) continue;
          const e = entries.get(key3(names[this.elemOf[tc]], names[this.elemOf[ta]], names[this.elemOf[tb]]))!;
          const idx = (tc * nt + ta) * nt + tb;
          this.kTab[idx] = e.K;
          this.thTab[idx] = e.theta0 * DEG;
        }
      }
    }
    this.fileRead = true;
  }

  override initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify shift and tail options`);
    }
    if (!this.fileRead) throw new StyleError(`pair_style ${this.name} needs a pair_coeff command with a potential file`);
  }

  override initOne(i: number, j: number): number {
    if (this.elemOf[i] < 0 || this.elemOf[j] < 0) return 0;
    const nt = this.ntypes + 1;
    return Math.max(this.legCut[i * nt + j], this.legCut[j * nt + i]);
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const legCut = this.legCut, legRho = this.legRho;
    const kTab = this.kTab, thTab = this.thTab;
    const elemOf = this.elemOf;
    const screened = this.screened;
    const eatom = pc.eatom, vatom = pc.vatom;
    let evdwl = 0;

    let maxn = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.gIdx = new Int32Array(maxn);
      this.gDx = new Float64Array(maxn);
      this.gDy = new Float64Array(maxn);
      this.gDz = new Float64Array(maxn);
      this.gR = new Float64Array(maxn);
    }
    const gIdx = this.gIdx, gDx = this.gDx, gDy = this.gDy, gDz = this.gDz, gR = this.gR;

    for (let i = 0; i < nlocal; i++) {
      const ti = type[i];
      if (elemOf[ti] < 0) continue;
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      // gather the neighbours inside the leg cutoff of this center
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        if (elemOf[tj] < 0) continue;
        const lc = legCut[ti * nt + tj];
        if (!(lc > 0)) continue;
        const dx = x[3 * j] - x[3 * i];
        const dy = x[3 * j + 1] - x[3 * i + 1];
        const dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        // the leg is kept while r <= cutoff (measured with native LAMMPS: r = cut = 2.0 contributes)
        if (rsq > lc * lc) continue;
        gIdx[m] = j; gDx[m] = dx; gDy[m] = dy; gDz[m] = dz; gR[m] = Math.sqrt(rsq);
        m++;
      }
      // every unordered pair of gathered neighbours is one angle, counted once
      for (let p1 = 0; p1 < m; p1++) {
        const j = gIdx[p1];
        const tj = type[j];
        for (let p2 = p1 + 1; p2 < m; p2++) {
          const k = gIdx[p2];
          const tk = type[k];
          const lo = tj < tk ? tj : tk, hi = tj < tk ? tk : tj;
          const idx = (ti * nt + lo) * nt + hi;
          const K = kTab[idx];
          if (K === 0) continue;
          const th0 = thTab[idx];
          // u = x_j - x_i, v = x_k - x_i (the neighbours of the center)
          const ux = gDx[p1], uy = gDy[p1], uz = gDz[p1], ru = gR[p1];
          const vx = gDx[p2], vy = gDy[p2], vz = gDz[p2], rv = gR[p2];
          // screening exp(-r_ij/rho_ij - r_ik/rho_ik) with the rho of each leg; 1 for the harmonic style
          let F = 1, radJ = 0, radK = 0;
          if (screened) {
            const rhoJ = legRho[ti * nt + tj], rhoK = legRho[ti * nt + tk];
            F = libm.exp(-ru / rhoJ - rv / rhoK);
          }
          const dot = ux * vx + uy * vy + uz * vz;
          const c = Math.min(1, Math.max(-1, dot / (ru * rv)));
          const theta = libm.acos(c);
          const s = Math.max(Math.sqrt(1 - c * c), 1e-8);
          const dth = theta - th0;
          const e = K * dth * dth * F;
          evdwl += e;
          // dE/dtheta = 2K(theta - theta0); dtheta/dc = -1/s; forces f_j = -dE/dx_j
          const g = ((2 * K * dth) / s) * F;
          if (screened) {
            // radial part of -dE/du and -dE/dv from the exponential: K dth^2 F / (rho r) along the leg
            radJ = (K * dth * dth * F) / (legRho[ti * nt + tj] * ru);
            radK = (K * dth * dth * F) / (legRho[ti * nt + tk] * rv);
          }
          // dc/du = v/(ru rv) - c u/ru^2 ; dc/dv = u/(ru rv) - c v/rv^2
          const inv = 1 / (ru * rv);
          const cu = c / (ru * ru), cv = c / (rv * rv);
          const fjx = g * (vx * inv - cu * ux) + radJ * ux, fjy = g * (vy * inv - cu * uy) + radJ * uy, fjz = g * (vz * inv - cu * uz) + radJ * uz;
          const fkx = g * (ux * inv - cv * vx) + radK * vx, fky = g * (uy * inv - cv * vy) + radK * vy, fkz = g * (uz * inv - cv * vz) + radK * vz;
          f[3 * j] += fjx; f[3 * j + 1] += fjy; f[3 * j + 2] += fjz;
          f[3 * k] += fkx; f[3 * k + 1] += fky; f[3 * k + 2] += fkz;
          f[3 * i] -= fjx + fkx; f[3 * i + 1] -= fjy + fky; f[3 * i + 2] -= fjz + fkz;
          if (eatom) {
            const e3 = e / 3;
            eatom[i] += e3; eatom[j] += e3; eatom[k] += e3;
          }
          if (vatom) {
            // virial of the angle about the center: sum over the two neighbours of r . f
            const w = [
              ux * fjx + vx * fkx, uy * fjy + vy * fky, uz * fjz + vz * fkz,
              ux * fjy + vx * fky, ux * fjz + vx * fkz, uy * fjz + vy * fkz,
            ];
            for (let q = 0; q < 6; q++) {
              const wq = w[q] / 3;
              vatom[6 * i + q] += wq; vatom[6 * j + q] += wq; vatom[6 * k + q] += wq;
            }
          }
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}

/** pair_style nb3b/screened: E = K (theta - theta0)^2 exp(-r_ij/rho_ij - r_ik/rho_ik), 7-column file. */
export class PairNB3BScreened extends PairNB3BHarmonic {
  override readonly name = 'nb3b/screened';
  protected override screened = true;
}
