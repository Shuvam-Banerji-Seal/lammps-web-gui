import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { BondedCompute } from '../force/types';
import type { System } from '../system';
import type { SimState } from '../types';
import { buildAtomMap } from '../atoms';
import { dihedralGeometry } from '../force/bonded_util';

/*
 * fix ID group-ID cmap filename — CHARMM CMAP backbone cross-term correction (wave 15).
 *
 * Doc (docs.lammps.org/fix_cmap.html): "This command enables CMAP 5-body interactions to be added to
 * simulations which use the CHARMM force field." The five atoms of a cross-term are the atoms of the
 * two overlapping dihedrals: phi = dihedral 1-2-3-4 (first index of the grid) and psi = dihedral 2-3-4-5
 * (second index). The energy is a bicubic interpolation of a periodic 24 x 24 grid.
 *
 * Doc: "The first column is an index from 1 to N to enumerate the CMAP 5-atom" tuples; it is ignored by
 * LAMMPS. The second column is the type, an index into the CMAP grids of the file. Measured with native
 * LAMMPS (black box, scratch directory /tmp/haiku-cmap/): type 1 selects the first grid of the file,
 * type 2 the second; a type beyond the number of grids gives energy 0 natively, which this fix refuses
 * with a StyleError instead.
 *
 * Doc: "The forces due to this fix are imposed during an energy minimization," so minPostForce applies the
 * same forces as postForce.
 *
 * Grid file (not described on the doc page; measured with native LAMMPS, black box):
 *  - a comment line starts with '#' (the rest of a line after '#' is ignored); '!' and '%' are not comments.
 *  - the values are read line by line: a grid takes whole lines until it holds 576 values, and surplus
 *    values on the last line of a grid are dropped; the next grid starts on the next line.
 *  - a grid is 24 x 24 values, the first index (rows) for phi and the second (columns) for psi; grid node
 *    i (0..23) is at -180 + 15 i degrees. Measured: the value at phi = -135, psi = 0 equals the grid entry
 *    of row 3, column 12.
 *  - a non-numeric token stops native with an error; the engine throws a StyleError.
 *
 * Interpolation (measured with native LAMMPS, black box): the grid derivatives come from a natural cubic
 * spline (zero second derivative at both ends) through the 48 periodic samples that start 12 nodes before
 * node 0 and end 12 nodes after node 23 (so the centre period is the one differentiated). Over random
 * off-grid points on two random grids the native energies agree with this model to 2e-14, while the exact
 * periodic cubic spline differs by up to 1.6e-8 (near the grid seam); the first candidate tested, the
 * tensor-product periodic spline, is not the native scheme. Each cell is the bicubic Hermite patch with
 * corner values, phi and psi derivatives and the cross derivative (fuv = d2E/dphi dpsi); the derivatives
 * are the tensor product of the 1D operator.
 *
 * Measured with native LAMMPS: at a dihedral that rounds to a value a few 1e-14 below a grid node while
 * the other angle is off-node, the native energy is wrong by O(1) (not reproduced; grid nodes exactly
 * on both angles are exact in native). Dihedral angles that do not sit on a node are unaffected.
 *
 * fix_modify energy and virial: the doc gives the default as "default setting for this fix is" energy yes
 * and virial yes; measured with native LAMMPS: pe includes the CMAP energy by default and drops it with
 * fix_modify energy no, while f_ID (the extensive scalar) is the CMAP energy either way. The virial of a
 * dihedral-only energy has zero trace (scale invariance); its off-diagonal components are not zero, and the
 * engine tallies sum(r F).
 *
 * Not supported (StyleError): a group other than all (LAMMPS cannot define another group before the box
 * exists, where this fix must be defined); a header other than the N crossterms line and a section that is
 * not the crossterm list. Restart files: the cross-term list is stored in the engine's restart file and
 * restored when the fix is re-specified after read_restart (restartState / restoreFromRestart).
 */

const NODES = 24;
const SPACING = 15;
const GRID_VALUES = NODES * NODES;
const DEG = 180 / Math.PI;
/** Natural spline window: NODES + 2 * PAD samples, derivatives taken on the centre period (measured, see header). */
const PAD = NODES / 2;
const WINDOW = NODES + 2 * PAD;

/** Hermite matrix: power coefficients a = H (p0, p1, m0, m1), with a_k the coefficient of t^k. */
const H = [[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]];

/**
 * Grid file reader, measured as described in the header (line-based grids, '#' comments).
 * Returns the grids in file order, each as 576 values with the phi index as the row.
 */
export const parseCmapGrids = (text: string, name: string): Float64Array[] => {
  const grids: Float64Array[] = [];
  let cur: number[] = [];
  for (const raw of text.split('\n')) {
    const line = raw.replace(/#.*/, '').trim();
    if (!line) continue;
    for (const t of line.split(/\s+/)) {
      const v = Number(t);
      if (!Number.isFinite(v)) throw new StyleError(`fix cmap: ${name} has a non-numeric value '${t}' in its CMAP grid data`);
      cur.push(v);
    }
    if (cur.length >= GRID_VALUES) {
      grids.push(Float64Array.from(cur.slice(0, GRID_VALUES)));
      cur = [];
    }
  }
  if (cur.length) throw new StyleError(`fix cmap: ${name} ends inside a CMAP grid (${cur.length} of ${GRID_VALUES} values)`);
  if (!grids.length) throw new StyleError(`fix cmap: ${name} holds no CMAP grid`);
  return grids;
};

/**
 * Derivatives at the nodes of the natural cubic spline through the periodic samples y (unit spacing), on the
 * window described above. Returns the NODES derivatives; solved by the tridiagonal (Thomas) algorithm.
 */
const naturalDerivatives = (y: Float64Array): Float64Array => {
  const w = new Float64Array(WINDOW);
  for (let t = 0; t < WINDOW; t++) w[t] = y[(((t - PAD) % NODES) + NODES) % NODES];
  // second derivatives M_1..M_{WINDOW-2}; M_0 = M_{WINDOW-1} = 0 (natural ends)
  const n = WINDOW - 2;
  const cp = new Float64Array(n), dp = new Float64Array(n), m = new Float64Array(WINDOW);
  for (let k = 0; k < n; k++) {
    const t = k + 1;
    const d = 6 * (w[t + 1] - 2 * w[t] + w[t - 1]);
    if (k === 0) { cp[k] = 1 / 4; dp[k] = d / 4; } else {
      const den = 4 - cp[k - 1];
      cp[k] = 1 / den;
      dp[k] = (d - dp[k - 1]) / den;
    }
  }
  for (let k = n - 1; k >= 0; k--) m[k + 1] = k === n - 1 ? dp[k] : dp[k] - cp[k] * m[k + 2];
  const out = new Float64Array(NODES);
  for (let q = 0; q < NODES; q++) {
    const t = q + PAD;
    out[q] = (w[t + 1] - w[t]) - (2 * m[t] + m[t + 1]) / 6;
  }
  return out;
};

/** NODES x NODES matrix D with derivative_row = D * samples (both periodic, unit spacing). */
export const derivativeMatrix = (): Float64Array => {
  const D = new Float64Array(NODES * NODES);
  for (let k = 0; k < NODES; k++) {
    const e = new Float64Array(NODES);
    e[k] = 1;
    const d = naturalDerivatives(e);
    for (let m = 0; m < NODES; m++) D[m * NODES + k] = d[m];
  }
  return D;
};

/** Per-grid bicubic tables: 16 power coefficients per cell (phi cell i, psi cell j), index (i*NODES+j)*16 + 4p + q. */
export interface CmapTable { coef: Float64Array }

export const buildCmapTable = (G: Float64Array, D: Float64Array): CmapTable => {
  const at = (i: number, j: number) => ((i % NODES) + NODES) % NODES * NODES + ((j % NODES) + NODES) % NODES;
  // derivatives in cell units: fu = dE/du (phi), fv = dE/dv (psi), fuv = d2E/dudv
  const fu = new Float64Array(GRID_VALUES), fv = new Float64Array(GRID_VALUES), fuv = new Float64Array(GRID_VALUES);
  for (let i = 0; i < NODES; i++) {
    for (let j = 0; j < NODES; j++) {
      let a = 0, b = 0;
      for (let k = 0; k < NODES; k++) {
        a += D[i * NODES + k] * G[at(k, j)];
        b += D[j * NODES + k] * G[at(i, k)];
      }
      fu[at(i, j)] = a;
      fv[at(i, j)] = b;
    }
  }
  for (let i = 0; i < NODES; i++) {
    for (let j = 0; j < NODES; j++) {
      let c = 0;
      for (let k = 0; k < NODES; k++) c += D[i * NODES + k] * fv[at(k, j)];
      fuv[at(i, j)] = c;
    }
  }
  const coef = new Float64Array(GRID_VALUES * 16);
  const E = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < NODES; i++) {
    for (let j = 0; j < NODES; j++) {
      const c00 = at(i, j), c01 = at(i, j + 1), c10 = at(i + 1, j), c11 = at(i + 1, j + 1);
      // rows: value at u0, value at u1, du at u0, du at u1; columns: the same in v
      E[0] = [G[c00], G[c01], fv[c00], fv[c01]];
      E[1] = [G[c10], G[c11], fv[c10], fv[c11]];
      E[2] = [fu[c00], fu[c01], fuv[c00], fuv[c01]];
      E[3] = [fu[c10], fu[c11], fuv[c10], fuv[c11]];
      const base = (i * NODES + j) * 16;
      for (let p = 0; p < 4; p++) {
        for (let q = 0; q < 4; q++) {
          let s = 0;
          for (let r = 0; r < 4; r++) for (let t = 0; t < 4; t++) s += H[p][r] * E[r][t] * H[q][t];
          coef[base + 4 * p + q] = s;
        }
      }
    }
  }
  return { coef };
};

/** Energy and its phi/psi derivatives (per radian) of one table at (phi, psi) in radians. */
export const evaluateCmap = (tab: CmapTable, phi: number, psi: number): { e: number; dphi: number; dpsi: number } => {
  const x = (phi * DEG + 180) / SPACING, y = (psi * DEG + 180) / SPACING;
  const ix = Math.floor(x), iy = Math.floor(y);
  const u = x - ix, v = y - iy;
  const ci = ((ix % NODES) + NODES) % NODES, cj = ((iy % NODES) + NODES) % NODES;
  const base = (ci * NODES + cj) * 16;
  const up = [1, u, u * u, u * u * u], vp = [1, v, v * v, v * v * v];
  const du = [0, 1, 2 * u, 3 * u * u], dv = [0, 1, 2 * v, 3 * v * v];
  let e = 0, eu = 0, ev = 0;
  for (let p = 0; p < 4; p++) {
    for (let q = 0; q < 4; q++) {
      const c = tab.coef[base + 4 * p + q];
      e += c * up[p] * vp[q];
      eu += c * du[p] * vp[q];
      ev += c * up[p] * dv[q];
    }
  }
  const k = DEG / SPACING;
  return { e, dphi: eu * k, dpsi: ev * k };
};

interface CrossTerm { type: number; ids: number[] }

/** Restart record of fix cmap: the cross-term list (written by write_restart, restored by re-specifying the fix). */
export interface CmapRestart { crossterms: { type: number; ids: number[] }[] }

/** fix cmap: the CMAP energy and forces of the crossterms read from read_data (see the header). */
export class FixCmap extends Fix {
  readonly style = 'cmap';
  private readonly tables: CmapTable[];
  private readonly cross: CrossTerm[] = [];
  private ncross: number | null = null;
  private etotal = 0;
  private perE = new Float64Array(0);
  private perV = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') {
      throw new StyleError(`fix cmap group ${group}: only group all is supported (LAMMPS cannot define another group before the simulation box exists, where this fix is defined)`);
    }
    if (args.length !== 1) throw new StyleError('usage: fix ID group-ID cmap filename');
    const name = args[0];
    const D = derivativeMatrix();
    this.tables = parseCmapGrids(sys.readFile(name), name).map((G) => buildCmapTable(G, D));
    // fix_cmap.rst: energy and virial are on by default ("default setting for this fix is")
    this.energyGlobal = true;
    this.virialGlobal = true;
    this.thermoEnergy = true;
    this.thermoVirial = true;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.restoreFromRestart();
  }

  /**
   * fix_cmap.html: "This fix writes the list of CMAP cross-terms to binary restart files"; "See the
   * read_restart command for info on how to re-specify a fix in an input script that reads a restart
   * file". The saved list is restored when this fix is re-specified after read_restart (the grid file
   * is read again from the new filename).
   */
  private restoreFromRestart(): void {
    const saved = (this.sys.pendingFixData.get(this.id) as { cmap?: CmapRestart } | undefined)?.cmap;
    if (!saved) return;
    this.sys.pendingFixData.delete(this.id);
    for (const c of saved.crossterms) {
      if (c.type < 1 || c.type > this.tables.length) {
        throw new StyleError(`fix ${this.id} (cmap): the restart file's CMAP type ${c.type} has no grid in this grid file (it holds ${this.tables.length})`);
      }
      this.cross.push({ type: c.type, ids: c.ids.slice() });
    }
    this.ncross = saved.crossterms.length;
  }

  /** The cross-term list for the restart file (ids are the atom IDs, types the grid indices). */
  restartState(): CmapRestart {
    return { crossterms: this.cross.map((c) => ({ type: c.type, ids: c.ids.slice() })) };
  }

  /** read_data header-string line: "N crossterms" (fix_cmap.rst: "N crossterms"). */
  readHeader(line: string, at: number): void {
    const m = /^\s*(\d+)\s+crossterms\s*$/.exec(line);
    if (!m) throw new StyleError(`data file line ${at}: fix cmap expects a header line 'N crossterms', got '${line.trim()}'`);
    if (this.ncross !== null) throw new StyleError(`data file line ${at}: fix cmap header 'crossterms' given twice`);
    this.ncross = Number(m[1]);
  }

  /** Lines in the crossterm section: the N of the header. */
  sectionLines(_natoms: number): number {
    if (this.ncross === null) throw new StyleError('read_data fix cmap: the header must select the "N crossterms" line (use header-string crossterm)');
    return this.ncross;
  }

  /** read_data section: index, type, then the five atom IDs of the crossterm (fix_cmap.rst). */
  readSection(rows: { w: string[]; at: number }[]): void {
    for (const { w, at } of rows) {
      if (w.length !== 7) throw new StyleError(`data file line ${at}: CMAP section needs 7 values (index, type, five atom IDs), got ${w.length}`);
      const nums = w.map((t) => Number(t));
      if (nums.some((v) => !Number.isInteger(v))) throw new StyleError(`data file line ${at}: CMAP section values must be integers`);
      const type = nums[1];
      if (type < 1 || type > this.tables.length) {
        throw new StyleError(`data file line ${at}: CMAP type ${type} has no grid (the file holds ${this.tables.length})`);
      }
      this.cross.push({ type, ids: nums.slice(2) });
    }
  }

  /** Writes the header line of write_data (native: "N crossterms" after the topology counts). */
  dataHeaderLine(): string | null {
    // measured with native: a header read with 0 crossterms is still written ("0 crossterms")
    return this.ncross !== null ? `${this.cross.length} crossterms` : null;
  }

  /**
   * The CMAP section of write_data: index, type, five atom IDs per crossterm, sorted by the position of the
   * first atom (measured with native: a reversed input list is written in atom order, renumbered from 1).
   */
  dataSection(slot: (id: number) => number): { title: string; lines: string[] } | null {
    if (this.ncross === null) return null;
    const sorted = this.cross.map((c, k) => ({ c, k })).sort((x, y) => slot(x.c.ids[0]) - slot(y.c.ids[0]) || x.k - y.k);
    const lines = sorted.map(({ c }, n) => `${n + 1} ${c.type} ${c.ids.join(' ')}`);
    return { title: 'CMAP', lines };
  }

  postForce(): void {
    const s: SimState = this.sys.state;
    const f = s.f;
    const shim = { s, geom: this.sys.geom } as unknown as BondedCompute;
    const map = buildAtomMap(s);
    const idx = (id: number): number => {
      const i = id < map.length ? map[id] : -1;
      if (i < 0) throw new StyleError(`fix cmap: atom ${id} missing`);
      return i;
    };
    const v = this.virial;
    v.fill(0);
    if (this.perE.length !== s.n) { this.perE = new Float64Array(s.n); this.perV = new Float64Array(6 * s.n); } else { this.perE.fill(0); this.perV.fill(0); }
    const gp = new Array(12).fill(0), rp = new Array(12).fill(0);
    const gs = new Array(12).fill(0), rs = new Array(12).fill(0);
    const w = new Float64Array(6);
    let etot = 0;
    for (const ct of this.cross) {
      const a = ct.ids.map(idx);
      const phi = dihedralGeometry(shim, a[0], a[1], a[2], a[3], gp, rp);
      const psi = dihedralGeometry(shim, a[1], a[2], a[3], a[4], gs, rs);
      const r = evaluateCmap(this.tables[ct.type - 1], phi, psi);
      etot += r.e;
      // phi acts on atoms 1-4 (positions rp relative to atom 1), psi on atoms 2-5 (rs relative to atom 2)
      w.fill(0);
      this.applyForces(f, a, 0, -r.dphi, gp, rp, w);
      this.applyForces(f, a, 1, -r.dpsi, gs, rs, w);
      for (let c = 0; c < 6; c++) v[c] += w[c];
      // measured with native LAMMPS (single crossterms, compute pe/atom fix and stress/atom NULL fix): the
      // energy and the virial of a crossterm are split equally among its five atoms (fifths)
      for (let k = 0; k < 5; k++) {
        const atom = a[k];
        this.perE[atom] += r.e / 5;
        for (let c = 0; c < 6; c++) this.perV[6 * atom + c] += w[c] / 5;
      }
    }
    this.etotal = etot;
  }

  /** Per-atom energy of the crossterms (compute pe/atom fix, fix_modify energy yes). */
  energyAtom(out: Float64Array): void {
    for (let i = 0; i < this.perE.length && i < out.length; i++) out[i] += this.perE[i];
  }

  /** Per-atom virial of the crossterms, 6 per atom (compute stress/atom fix, fix_modify virial yes). */
  virialAtom(out: Float64Array): void {
    for (let k = 0; k < this.perV.length && k < out.length; k++) out[k] += this.perV[k];
  }

  /** F_k = -dE/dphi * dphi/dx_k for the four atoms starting at atom slot `first`; virial sum(r F) into w. */
  private applyForces(f: Float64Array, a: number[], first: number, dEdq: number, grad: number[], rel: number[], w: Float64Array): void {
    for (let k = 0; k < 4; k++) {
      const atom = a[first + k];
      const fx = dEdq * grad[3 * k], fy = dEdq * grad[3 * k + 1], fz = dEdq * grad[3 * k + 2];
      f[3 * atom] += fx; f[3 * atom + 1] += fy; f[3 * atom + 2] += fz;
      const rx = rel[3 * k], ry = rel[3 * k + 1], rz = rel[3 * k + 2];
      w[0] += rx * fx; w[1] += ry * fy; w[2] += rz * fz;
      w[3] += 0.5 * (rx * fy + ry * fx);
      w[4] += 0.5 * (rx * fz + rz * fx);
      w[5] += 0.5 * (ry * fz + rz * fy);
    }
  }

  minPostForce(): void { this.postForce(); }

  energy(): number { return this.etotal; }
  computeScalar(): number { return this.etotal; }
}
