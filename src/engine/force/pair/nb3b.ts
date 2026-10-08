import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum, joinPotentialEntries } from '../util';

/*
 * pair_style nb3b/harmonic (and nb3b/screened, refused) — docs.lammps.org/pair_nb3b.html
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
 * nb3b/screened (verbatim formula): "E = K (\theta - \theta_0)^2 \exp \left(-
 * \frac{r_{ij}}{\rho_{ij}} - \frac{r_{ik}}{\rho_{ik}} \right)".  The document
 * does not define the potential-file columns for this style (the screening
 * lengths rho are not given a column), and a native run of the nb3b/harmonic
 * file layout fails with Not a valid floating-point number: 'Mg' when read by
 * nb3b/screened.  The style is therefore refused, not guessed.
 *
 * Forces: the angle derivative uses d theta/d c = -1/sin(theta) with the
 * cosine clamped to [-1, 1]; sin(theta) is floored at 1e-8 (an engine guard
 * for exactly collinear triplets, where the force stays finite because the
 * perpendicular part of dc/du vanishes).  The global virial comes from
 * virialFdotr; per-atom energy and virial are split evenly over the three atoms.
 */

interface NB3BEntry {
  e1: string;
  e2: string;
  e3: string;
  K: number;
  /** theta0 in degrees, as in the file. */
  theta0: number;
  cut: number;
}

const key3 = (a: string, b: string, c: string): string => `${a} ${b} ${c}`;

/** Parses a .nb3b.harmonic file: entries of 3 element names + K, theta0, cutoff. */
const parseNB3BFile = (text: string, fileName: string): Map<string, NB3BEntry> => {
  const lines = joinPotentialEntries(text.split(/\r?\n/), 6);
  const entries = new Map<string, NB3BEntry>();
  for (let ln = 0; ln < lines.length; ln++) {
    const raw = lines[ln];
    const hash = raw.indexOf('#');
    const line = (hash >= 0 ? raw.slice(0, hash) : raw).trim();
    if (line === '') continue;
    const t = line.split(/\s+/);
    if (t.length !== 6) {
      throw new StyleError(
        `nb3b potential file ${fileName} line ${ln + 1}: expected 'element1 element2 element3 K theta0 cutoff' (6 values), got ${t.length}`,
      );
    }
    const k = key3(t[0], t[1], t[2]);
    if (entries.has(k)) throw new StyleError(`nb3b potential file ${fileName} line ${ln + 1}: duplicate entry for elements ${k}`);
    const K = parseNum(t[3], `K of the nb3b file entry ${k}`);
    const theta0 = parseNum(t[4], `theta0 of the nb3b file entry ${k}`);
    const cut = parseNum(t[5], `cutoff of the nb3b file entry ${k}`);
    if (cut < 0) throw new StyleError(`nb3b potential file ${fileName} entry ${k}: cutoff must be >= 0`);
    entries.set(k, { e1: t[0], e2: t[1], e3: t[2], K, theta0, cut });
  }
  if (entries.size === 0) throw new StyleError(`nb3b potential file ${fileName} contains no parameter entries`);
  return entries;
};

/** pair_style nb3b/harmonic: E = K (theta - theta0)^2 over the angles of neighbour pairs. */
export class PairNB3BHarmonic extends Pair {
  readonly name: string = 'nb3b/harmonic';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected fileName = '';
  protected fileRead = false;
  /** type -> element index, -1 for NULL types. */
  protected elemOf = new Int32Array(0);
  /** Angle parameters, indexed by (center, lo, hi) with lo <= hi (types): K and theta0 in radians. */
  protected kTab = new Float64Array(0);
  protected thTab = new Float64Array(0);
  /** Leg cutoff of center c to neighbour type t: the cutoff of the (c, t, t) entry. */
  protected legCut = new Float64Array(0);

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
    this.fileRead = false;
    this.fileName = '';
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError(`usage: pair_coeff * * filename elem1 ... elemN (style ${this.name})`);
    if (args[0] !== '*' || args[1] !== '*') {
      throw new StyleError(`the first 2 arguments of pair_coeff for style ${this.name} must be * *`);
    }
    if (this.fileRead) throw new StyleError(`pair_style ${this.name}: only a single pair_coeff command is allowed`);
    const filename = args[2];
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style ${this.name} needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    }
    const entries = parseNB3BFile(ctx.readFile(filename), filename);
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
    const DEG = Math.PI / 180;
    for (let tc = 1; tc <= this.ntypes; tc++) {
      if (this.elemOf[tc] < 0) continue;
      for (let ta = 1; ta <= this.ntypes; ta++) {
        if (this.elemOf[ta] < 0) continue;
        this.legCut[tc * nt + ta] = entries.get(key3(names[this.elemOf[tc]], names[this.elemOf[ta]], names[this.elemOf[ta]]))!.cut;
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
    const legCut = this.legCut;
    const kTab = this.kTab, thTab = this.thTab;
    const elemOf = this.elemOf;
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
          const dot = ux * vx + uy * vy + uz * vz;
          const c = Math.min(1, Math.max(-1, dot / (ru * rv)));
          const theta = Math.acos(c);
          const s = Math.max(Math.sqrt(1 - c * c), 1e-8);
          const dth = theta - th0;
          const e = K * dth * dth;
          evdwl += e;
          // dE/dtheta = 2K(theta - theta0); dtheta/dc = -1/s; forces f_j = -dE/dx_j
          const g = (2 * K * dth) / s;
          // dc/du = v/(ru rv) - c u/ru^2 ; dc/dv = u/(ru rv) - c v/rv^2
          const inv = 1 / (ru * rv);
          const cu = c / (ru * ru), cv = c / (rv * rv);
          const fjx = g * (vx * inv - cu * ux), fjy = g * (vy * inv - cu * uy), fjz = g * (vz * inv - cu * uz);
          const fkx = g * (ux * inv - cv * vx), fky = g * (uy * inv - cv * vy), fkz = g * (uz * inv - cv * vz);
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

/** pair_style nb3b/screened is recognised but refused (its file layout is not in the docs). */
export class PairNB3BScreened extends PairNB3BHarmonic {
  override readonly name = 'nb3b/screened';

  override settings(_args: string[], _ctx: StyleContext): void {
    throw new StyleError(
      'pair_style nb3b/screened is not supported: docs.lammps.org/pair_nb3b.html gives its energy formula but not the potential-file columns for the screening lengths',
    );
  }
}
