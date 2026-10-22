/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_frenkel.cpp and compute_frenkel.h at commit b332f28 (ComputeFrenkel: compute frenkel,
 * the on-the-fly Wigner-Seitz defect analysis). Contributing author: Karl D. Hammond (University of Missouri, Columbia,
 * 2018); refactored for inclusion into LAMMPS by Axel Kohlmeyer (Temple University, 2026). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_frenkel.html (verbatim): "compute ID group-ID frenkel keyword value ..."; "keyword = drvac or
 * drint or region or rescale or site_file"; "This compute calculates a global vector of length 3, a global array, a per-atom
 * vector, and a local array."; "The global array has 2 rows and 20 columns and is a histogram of the defect cluster sizes:
 * row 1 counts vacancy clusters and row 2 counts ..."; "The local array has 5 columns and one row per defect cluster
 * owned by the MPI process: the cluster ID, the cluster size (negative for vacancy ...". The doc says the size of a cluster
 * "is the number of its interstitials minus the number of its vacancies".
 *
 * Scope of this port (the engine is one process and fully periodic):
 *   - the reference sites are the lattice points of the box (lattice command), in the native creation order, tagged
 *     natoms+1 .. natoms+nsites as native does for one process; keyword region restricts them;
 *   - every site-site and atom-site distance is the minimum-image distance, which is what the native ghost-site
 *     exchange and bins give for one process; the atom-to-site assignment is the nearest site within the search
 *     cutoff, and an atom counts as an occupant only inside the Wigner-Seitz cell of that site (native
 *     inside_WS_cell);
 *   - clusters are the connected components (minimum site tag as the ID) of the non-regular sites, linked when the
 *     neighbour distance is within cut_int (interstitial neighbour) or cut_vac (vacancy neighbour); the cluster centre
 *     follows native find_clusters (periodic circular means, then the nearest images).
 * Refused (named StyleError): rescale yes (sites do not follow a changing box here), site_file, triclinic boxes,
 * non-periodic boundaries.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { logical, missingArgs, numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { Lattice } from '../../engine/lattice';
import type { Region } from '../../engine/region';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

const BIG = 1.0e20;
const SMALL = 1.0e-10;
const MAX_OCCUPANTS = 8;
const DEFAULT_DRVAC = 1.5;
const DEFAULT_DRINT = 1.82;
const MIN_SEARCH_DISTANCE = 1.5;
const EPSILON = 1.0e-6;
const ARRAY_COLS = 20;
const TWO_PI = 2.0 * Math.PI;

export class ComputeFrenkel extends Compute {
  readonly style = 'frenkel';
  private drVac = DEFAULT_DRVAC;
  private drInt = DEFAULT_DRINT;
  private regionId: string | null = null;
  private regionObj: Region | null = null;
  private dnn = 0;
  private cutVac = 0;
  private cutInt = 0;
  private cutoff = 0;
  /** Lattice normals of the Wigner-Seitz cell, in lattice units (3 per normal). */
  private normal = new Float64Array(0);
  private nnormal = 0;
  /** Reference sites: positions (3 per site), tags, and the neighbour lists (site indices within cutoff). */
  private sites = new Float64Array(0);
  private siteTag = new Float64Array(0);
  private nsite = 0;
  private nlist: Int32Array[] = [];
  /** Occupancy: number of atoms per site, and their tags (MAX_OCCUPANTS per site, 0 = free slot). */
  private noccupants = new Int32Array(0);
  private occupant = new Float64Array(0);
  /** Per owned atom: distance to its site (BIG without one, 0 outside the group). */
  private mindistAtom = new Float64Array(0);
  private clusterID = new Float64Array(0);
  private occupiedIDs: number[] = [];
  private clusterSize: number[] = [];
  private clusterCenter = new Float64Array(0);
  private defectsEpoch = -1;
  private clustersEpoch = -1;
  private readonly tmp = new Float64Array(3);
  private readonly tmp2 = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // compute_frenkel.cpp constructor: a lattice with basis atoms is needed before the keywords are read
    if (!sys.lattice || sys.lattice.basis.length === 0) throw new StyleError('Use of compute style frenkel with undefined lattice');
    this.parseArgs(args);
    this.vectorFlag = true;
    this.arrayFlag = true;
    this.peratomFlag = true;
    this.localFlag = true;
    this.sizeVector = 3;
    this.sizeArrayRows = 2;
    this.sizeArrayCols = ARRAY_COLS;
    this.sizeLocalCols = 5;
    this.extvector = 0;
    this.vector = new Float64Array(3);
    this.array = new Float64Array(2 * ARRAY_COLS);
  }

  private parseArgs(args: string[]): void {
    for (let k = 0; k < args.length; k += 2) {
      const key = args[k];
      const val = args[k + 1];
      if (key !== 'drvac' && key !== 'drint' && key !== 'region' && key !== 'rescale' && key !== 'site_file') {
        throw new StyleError(`Unknown compute frenkel keyword: ${key}`);
      }
      if (val === undefined) throw missingArgs(`compute frenkel ${key}`);
      if (key === 'drvac' || key === 'drint') {
        const v = numeric(val);
        if (v <= 0) throw new StyleError(`Compute frenkel ${key} value must be > 0.0`);
        if (key === 'drvac') this.drVac = v;
        else this.drInt = v;
      } else if (key === 'region') {
        if (val !== 'none' && !this.sys.regions.has(val)) throw new StyleError(`Region ${val} for compute frenkel doesn't exist`);
        this.regionId = val === 'none' ? null : val;
      } else if (key === 'rescale') {
        if (logical(val)) throw new StyleError('compute frenkel rescale yes is not supported by this engine (the reference sites do not follow a changing box)');
      } else if (key === 'site_file') {
        if (val !== 'none') throw new StyleError('compute frenkel site_file is not supported by this engine');
      } else {
        throw new StyleError(`Unknown compute frenkel keyword: ${key}`);
      }
    }
  }

  /** Native init(): the lattice, the nearest neighbour distance, the cutoffs, the reference sites and the WS normals. */
  init(): void {
    const lat = this.sys.lattice;
    if (!lat) throw new StyleError('Use of compute style frenkel with undefined lattice');
    const g = this.sys.geom;
    if (g.triclinic) throw new StyleError('compute frenkel does not support triclinic boxes in this engine');
    for (let d = 0; d < 3; d++) {
      if (!g.periodic[d]) throw new StyleError('compute frenkel needs periodic boundaries in every dimension in this engine');
    }
    this.regionObj = this.regionId === null ? null : this.sys.region(this.regionId);
    this.dnn = this.nearestNeighborDistance(lat);
    this.cutVac = this.drVac * this.dnn;
    this.cutInt = this.drInt * this.dnn;
    this.cutoff = Math.max(MIN_SEARCH_DISTANCE * this.dnn, Math.max(this.cutVac, this.cutInt));
    this.constructWSCell(lat);
    this.createSites(lat);
    this.defectsEpoch = -1;
    this.clustersEpoch = -1;
  }

  /** Smallest distance between two reference sites (native nearest_neighbor_distance). */
  private nearestNeighborDistance(lat: Lattice): number {
    const nlayer = this.sys.state.dimension === 2 ? 0 : 1;
    const x0 = lat.toBox(lat.basis[0]);
    let drsqMin = BIG;
    for (let j = 0; j < lat.basis.length; j++) {
      for (let n1 = -1; n1 <= 1; n1++) {
        for (let n2 = -1; n2 <= 1; n2++) {
          for (let n3 = -nlayer; n3 <= nlayer; n3++) {
            if (j === 0 && n1 === 0 && n2 === 0 && n3 === 0) continue;
            const f = [0, 1, 2].map((c) => lat.basis[j][c] + n1 * lat.a[0][c] + n2 * lat.a[1][c] + n3 * lat.a[2][c]) as [number, number, number];
            const x = lat.toBox(f);
            const dx = x[0] - x0[0], dy = x[1] - x0[1], dz = x[2] - x0[2];
            const drsq = dx * dx + dy * dy + dz * dz;
            if (drsq > SMALL && drsq < drsqMin) drsqMin = drsq;
          }
        }
      }
    }
    if (drsqMin >= BIG) throw new StyleError('Could not determine the nearest neighbor distance of the lattice used by compute frenkel');
    return Math.sqrt(drsqMin);
  }

  /** The normals of the Wigner-Seitz cell (native construct_WS_cell): midpoints of lattice vectors from basis atom 0. */
  private constructWSCell(lat: Lattice): void {
    const nbasis = lat.basis.length;
    this.nnormal = 27 * nbasis - 1;
    this.normal = new Float64Array(3 * this.nnormal);
    let n = 0;
    for (let j = 0; j < nbasis; j++) {
      for (let n1 = -1; n1 <= 1; n1++) {
        for (let n2 = -1; n2 <= 1; n2++) {
          for (let n3 = -1; n3 <= 1; n3++) {
            if (j === 0 && n1 === 0 && n2 === 0 && n3 === 0) continue;
            for (let c = 0; c < 3; c++) {
              this.normal[3 * n + c] = 0.5 * (lat.basis[j][c] + n1 * lat.a[0][c] + n2 * lat.a[1][c] + n3 * lat.a[2][c] - lat.basis[0][c]);
            }
            n++;
          }
        }
      }
    }
  }

  /**
   * Native create_lattice_sites for one process: the lattice points inside the box, in the native loop order (k, j, i,
   * then basis atom); the box is [lo - eps, hi - 2 eps) per periodic dimension (eps = 1e-6 prd). Sites are tagged
   * natoms + 1 + index.
   */
  private createSites(lat: Lattice): void {
    const g = this.sys.geom;
    const dim = this.sys.state.dimension;
    const sublo = [0, 0, 0], subhi = [0, 0, 0];
    const prd = [g.lx, g.ly, g.lz];
    for (let d = 0; d < 3; d++) {
      const eps = prd[d] * EPSILON;
      sublo[d] = g.lo[d] - eps;
      subhi[d] = g.hi[d] - 2.0 * eps;
    }
    // lattice index ranges that cover the box (the 8 corners in lattice coordinates)
    const mn = [BIG, BIG, BIG], mx = [-BIG, -BIG, -BIG];
    for (let c = 0; c < 8; c++) {
      const p = [(c & 1) ? g.hi[0] : g.lo[0], (c & 2) ? g.hi[1] : g.lo[1], (c & 4) ? g.hi[2] : g.lo[2]] as [number, number, number];
      const f = lat.fromBox(p);
      for (let d = 0; d < 3; d++) { mn[d] = Math.min(mn[d], f[d]); mx[d] = Math.max(mx[d], f[d]); }
    }
    const ilo = Math.floor(mn[0]) - 2, ihi = Math.ceil(mx[0]) + 2;
    const jlo = Math.floor(mn[1]) - 2, jhi = Math.ceil(mx[1]) + 2;
    const klo = dim === 2 ? 0 : Math.floor(mn[2]) - 2;
    const khi = dim === 2 ? 0 : Math.ceil(mx[2]) + 2;
    const region = this.regionObj;
    const list: number[] = [];
    for (let k = klo; k <= khi; k++) {
      for (let j = jlo; j <= jhi; j++) {
        for (let i = ilo; i <= ihi; i++) {
          for (let m = 0; m < lat.basis.length; m++) {
            const x = lat.toBox([i + lat.basis[m][0], j + lat.basis[m][1], k + lat.basis[m][2]]);
            if (region && !region.match(x[0], x[1], x[2])) continue;
            if (x[0] >= sublo[0] && x[0] < subhi[0] && x[1] >= sublo[1] && x[1] < subhi[1] && x[2] >= sublo[2] && x[2] < subhi[2]) {
              list.push(x[0], x[1], x[2]);
            }
          }
        }
      }
    }
    this.nsite = list.length / 3;
    this.sites = Float64Array.from(list);
    const natoms = this.sys.state.n;
    this.siteTag = new Float64Array(this.nsite);
    for (let n = 0; n < this.nsite; n++) this.siteTag[n] = natoms + 1 + n;
  }

  /** Minimum-image vector from point a to point b (b - a), written to out. */
  private minImage(out: Float64Array, bx: number, by: number, bz: number, ax: number, ay: number, az: number): void {
    out[0] = bx - ax;
    out[1] = by - ay;
    out[2] = bz - az;
    this.sys.geom.minimumImage(out);
  }

  /** Native construct_site_nlists: the sites within the search cutoff of each site (minimum image). */
  private buildSiteNlists(): void {
    const cutsq = this.cutoff * this.cutoff;
    const s = this.sites;
    const dv = this.tmp;
    this.nlist = new Array(this.nsite);
    for (let n = 0; n < this.nsite; n++) {
      const list: number[] = [];
      for (let m = 0; m < this.nsite; m++) {
        if (m === n) continue;
        this.minImage(dv, s[3 * m], s[3 * m + 1], s[3 * m + 2], s[3 * n], s[3 * n + 1], s[3 * n + 2]);
        if (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] <= cutsq) list.push(m);
      }
      this.nlist[n] = Int32Array.from(list);
    }
  }

  /** Native inside_WS_cell: dv is the minimum-image vector from the site to the atom (box units). */
  private insideWSCell(dv: Float64Array): boolean {
    const lat = this.sys.lattice!;
    if (Math.abs(dv[0]) > 0.5 * lat.spacing[0]) return false;
    if (Math.abs(dv[1]) > 0.5 * lat.spacing[1]) return false;
    if (Math.abs(dv[2]) > 0.5 * lat.spacing[2]) return false;
    // lattice units of the offset from the site (the origin shift cancels in the difference)
    const o = lat.fromBox([0, 0, 0]);
    const p = lat.fromBox([dv[0], dv[1], dv[2]]);
    const x = p[0] - o[0], y = p[1] - o[1], z = p[2] - o[2];
    for (let i = 0; i < this.nnormal; i++) {
      const n0 = this.normal[3 * i], n1 = this.normal[3 * i + 1], n2 = this.normal[3 * i + 2];
      if (Math.abs(n0) < SMALL && Math.abs(n1) < SMALL && Math.abs(n2) < SMALL) continue;
      const norm2sq = n0 * n0 + n1 * n1 + n2 * n2;
      const diff = x * n0 + y * n1 + z * n2 - norm2sq;
      if (diff > 0.0) return false;
    }
    return true;
  }

  /** Native find_defects: the nearest site of every group atom, and the occupancy of the sites. */
  private findDefects(): void {
    if (this.defectsEpoch === this.sys.epoch) return;
    this.defectsEpoch = this.sys.epoch;
    const s = this.sys.state;
    const n = s.n;
    this.buildSiteNlists();
    this.noccupants = new Int32Array(this.nsite);
    this.occupant = new Float64Array(this.nsite * MAX_OCCUPANTS);
    this.mindistAtom = new Float64Array(n);
    const cutsq = this.cutoff * this.cutoff;
    const sites = this.sites;
    const dv = this.tmp;
    const rel = this.tmp2;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) {
        this.mindistAtom[i] = 0;
        continue;
      }
      const xi = s.x[3 * i], yi = s.x[3 * i + 1], zi = s.x[3 * i + 2];
      let best = -1;
      let drsqMin = BIG;
      for (let k = 0; k < this.nsite; k++) {
        this.minImage(dv, sites[3 * k], sites[3 * k + 1], sites[3 * k + 2], xi, yi, zi);
        const drsq = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
        if (drsq < drsqMin && drsq <= cutsq) {
          drsqMin = drsq;
          best = k;
        }
      }
      if (best < 0) {
        this.mindistAtom[i] = BIG;
        continue;
      }
      this.mindistAtom[i] = Math.sqrt(drsqMin);
      // the offset from the site to the atom: minimum image of (atom - site)
      this.minImage(rel, xi, yi, zi, sites[3 * best], sites[3 * best + 1], sites[3 * best + 2]);
      if (!this.insideWSCell(rel)) continue;
      const base = best * MAX_OCCUPANTS;
      const tag = s.id[i];
      let found = false;
      for (let q = 0; q < MAX_OCCUPANTS && this.occupant[base + q] > 0; q++) {
        if (this.occupant[base + q] === tag) found = true;
      }
      if (found) continue;
      this.noccupants[best] += 1;
      let q = 0;
      while (q < MAX_OCCUPANTS && this.occupant[base + q] > 0) q++;
      if (q >= MAX_OCCUPANTS) throw new StyleError(`Greater than ${MAX_OCCUPANTS} atoms near a site; incorrect lattice, perhaps?`);
      this.occupant[base + q] = tag;
    }
  }

  /**
   * Native find_clusters: sites with one occupant are regular; the others take the minimum site tag of their connected
   * group (a vacancy links to a neighbour within cut_vac, an interstitial or irregular site within cut_int), then the
   * centres of the clusters.
   */
  private findClusters(): void {
    if (this.clustersEpoch === this.sys.epoch) return;
    this.findDefects();
    this.clustersEpoch = this.sys.epoch;
    const ns = this.nsite;
    const noc = this.noccupants;
    const g = this.sys.geom;
    const lo = g.lo;
    const prd = [g.lx, g.ly, g.lz];
    this.clusterID = new Float64Array(ns);
    for (let k = 0; k < ns; k++) this.clusterID[k] = noc[k] === 1 ? 0 : this.siteTag[k];
    const cutVacSq = this.cutVac * this.cutVac;
    const cutIntSq = this.cutInt * this.cutInt;
    const dv = this.tmp;
    const sites = this.sites;
    let changed = true;
    while (changed) {
      changed = false;
      for (let n = 0; n < ns; n++) {
        if (noc[n] === 1) continue;
        for (const m of this.nlist[n]) {
          if (m === n || noc[m] === 1 || this.clusterID[m] === this.clusterID[n]) continue;
          this.minImage(dv, sites[3 * m], sites[3 * m + 1], sites[3 * m + 2], sites[3 * n], sites[3 * n + 1], sites[3 * n + 2]);
          const drsq = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
          if ((noc[m] >= 2 && drsq <= cutIntSq) || (noc[m] === 0 && drsq <= cutVacSq)) {
            const v = Math.min(this.clusterID[n], this.clusterID[m]);
            this.clusterID[n] = v;
            this.clusterID[m] = v;
            changed = true;
          }
        }
      }
    }
    // the distinct positive IDs, ascending (native find_occupied_clusters)
    const ids: number[] = [];
    for (let k = 0; k < ns; k++) if (this.clusterID[k] > 0) ids.push(this.clusterID[k]);
    ids.sort((a, b) => a - b);
    const uniq: number[] = [];
    for (const id of ids) if (uniq.length === 0 || uniq[uniq.length - 1] !== id) uniq.push(id);
    this.occupiedIDs = uniq;
    const nocc = uniq.length;
    this.clusterSize = new Array(nocc).fill(0);
    this.clusterCenter = new Float64Array(3 * nocc);
    if (nocc === 0) return;
    const index = new Map<number, number>();
    uniq.forEach((id, j) => index.set(id, j));
    const nsites = new Array<number>(nocc).fill(0);
    const xi = new Float64Array(3 * nocc), zeta = new Float64Array(3 * nocc);
    for (let k = 0; k < ns; k++) {
      if (this.clusterID[k] === 0) continue;
      const n = index.get(this.clusterID[k])!;
      if (noc[k] === 0) this.clusterSize[n] += -1;
      else if (noc[k] >= 2) this.clusterSize[n] += 1;
      nsites[n] += 1;
      for (let d = 0; d < 3; d++) {
        const theta = ((sites[3 * k + d] - lo[d]) / prd[d]) * TWO_PI;
        zeta[3 * n + d] += libmSin(theta);
        xi[3 * n + d] += libmCos(theta);
      }
    }
    // approximate centres from the circular means, remapped into the box; then the nearest images of the sites
    const approx = new Float64Array(3 * nocc);
    for (let n = 0; n < nocc; n++) {
      for (let d = 0; d < 3; d++) {
        xi[3 * n + d] /= nsites[n];
        zeta[3 * n + d] /= nsites[n];
        const theta = Math.atan2(-zeta[3 * n + d], -xi[3 * n + d]) + Math.PI;
        let c = (prd[d] * theta) / TWO_PI + lo[d];
        const hi = lo[d] + prd[d];
        while (c < lo[d]) c += prd[d];
        while (c >= hi) c -= prd[d];
        approx[3 * n + d] = c;
      }
    }
    const center = this.clusterCenter;
    for (let k = 0; k < ns; k++) {
      if (this.clusterID[k] === 0) continue;
      const n = index.get(this.clusterID[k])!;
      this.minImage(dv, sites[3 * k], sites[3 * k + 1], sites[3 * k + 2], approx[3 * n], approx[3 * n + 1], approx[3 * n + 2]);
      for (let d = 0; d < 3; d++) {
        const nearest = approx[3 * n + d] + dv[d];
        center[3 * n + d] += nearest / nsites[n];
      }
    }
  }

  computeVector(): void {
    this.findDefects();
    let nvac = 0, nint = 0, nirr = 0;
    for (let k = 0; k < this.nsite; k++) {
      const c = this.noccupants[k];
      if (c === 0) nvac++;
      else if (c === 2) nint++;
      else if (c > 2) {
        nint++;
        nirr++;
      }
    }
    this.vector[0] = nvac;
    this.vector[1] = nint;
    this.vector[2] = nirr;
  }

  computeArray(): void {
    this.findClusters();
    const a = this.array;
    a.fill(0);
    for (let n = 0; n < this.clusterSize.length; n++) {
      const size = this.clusterSize[n];
      if (size === 0) continue;
      if (size > 0 && size < ARRAY_COLS) a[ARRAY_COLS + size - 1] += 1;
      else if (size >= ARRAY_COLS) a[ARRAY_COLS + ARRAY_COLS - 1] += 1;
      else if (size < 0 && size > -ARRAY_COLS) a[-size - 1] += 1;
      else a[ARRAY_COLS - 1] += 1;
    }
  }

  computePeratom(): void {
    this.findDefects();
    this.vectorAtom = this.mindistAtom;
  }

  /** Native compute_local: one row per cluster whose centre lies in the box (center_in_subdomain for one process). */
  computeLocal(): Float64Array<ArrayBuffer> {
    this.findClusters();
    const g = this.sys.geom;
    const rows: number[] = [];
    let nrows = 0;
    for (let j = 0; j < this.occupiedIDs.length; j++) {
      const c = [this.clusterCenter[3 * j], this.clusterCenter[3 * j + 1], this.clusterCenter[3 * j + 2]];
      const inside = [0, 1, 2].every((d) => c[d] >= g.lo[d] && c[d] < g.hi[d]);
      if (!inside) continue;
      rows.push(this.occupiedIDs[j], this.clusterSize[j], c[0], c[1], c[2]);
      nrows++;
    }
    this.localRows = nrows;
    return Float64Array.from(rows);
  }
}
