/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/fix_reaxff_bonds.cpp and src/REAXFF/fix_reaxff_bonds.h (FixReaxFFBonds: constructor,
 * setup, init, end_of_step, Output_ReaxFF_Bonds, FindBond, PassBuffer, RecvBuffer), and
 * src/REAXFF/fix_reaxff_species.cpp and src/REAXFF/fix_reaxff_species.h (FixReaxFFSpecies: constructor
 * with the nevery/nrepeat/nfreq checks and the cutoff and element keywords, setup, init, post_integrate,
 * Output_ReaxFF_Bonds, FindMolecule, SortMolecule, FindSpecies, GetUniqueElements, WriteFormulas), at commit
 * b332f28. The averaging of the species bond orders follows the schedule of fix ave/atom (fix_ave_atom.cpp
 * nextvalid and end_of_step, the fix that FixReaxFFSpecies creates as SPECBOND). Authors credited in the C++
 * file headers: Ray Shan (Sandia); Oleg Sergeev (VNIIA); Jacob Gissinger (NASA). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Not ported (StyleError): the delete, delete_rate_limit and delete_subgroup keywords and the position keyword of
 * fix reaxff/species (the delete keyword shuffles molecules with std::random_device, so its output is not
 * reproducible), the * file names of fix reaxff/bonds (one file per output step), compressed output files, and
 * the pad keyword of fix_modify.
 *
 * Divergence in the averaging: native averages every bond slot of tmpid/tmpbo over the window, slot by slot. This
 * port averages each bond by its atom pair (the atom IDs and the periodic image), which gives the same numbers as
 * long as the bond of a slot does not change within a window (the neighbor lists are rebuilt only at window
 * boundaries for the tatb example). The molecule finding is the same union of bonds, with the labels native
 * LAMMPS gives (the smallest atom ID of a molecule), so the molecules are numbered in native order.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { buildAtomMap, hasMolecule, nativeOrder } from '../../engine/atoms';
import { PairReaxFF } from './pair';
import type { BondList, RxSystem } from './core';
import type { NativeFarOrder } from './nativeFar';

const intArg = (w: string | undefined, what: string): number => {
  const v = w === undefined ? NaN : Number(w);
  if (!Number.isInteger(v)) throw new StyleError(`Illegal fix reaxff command: ${what} must be an integer, got '${w ?? ''}'`);
  return v;
};

/** The reaxff pair style of the system, or null. */
export const reaxffPairOf = (sys: System): PairReaxFF | null => (sys.ff.pair instanceof PairReaxFF ? sys.ff.pair : null);

/** C printf "%{w}.{p}f" (the native fprintf and fmt formats of these files). */
const fixedWidth = (x: number, w: number, p: number): string => x.toFixed(p).padStart(w);

/**
 * The bond list region of owned atom i in native order. Native's list of atom i holds, first, the bonds made by
 * the owned atoms below i in native order (their BOp calls append to i's region while those atoms are processed),
 * in native order of those atoms; then the bonds of i's own far neighbors, in the order of the native far list.
 */
const nativeRegion = (b: BondList, nat: NativeFarOrder, i: number, n: number): number[] => {
  const ni = nat.nativeIdx[i];
  const items: { pj: number; group: number; key: number }[] = [];
  for (let pj = b.start[i]; pj < b.end[i]; pj++) {
    const p = b.nbr[pj];
    if (p < n && nat.nativeIdx[p] < ni) {
      items.push({ pj, group: 0, key: nat.nativeIdx[p] });
    } else {
      const r = nat.rank.get(i * nat.nall + p);
      if (r === undefined) throw new StyleError('internal: a bond partner is not in the native far list of its atom');
      items.push({ pj, group: 1, key: r });
    }
  }
  items.sort((a, c) => a.group - c.group || a.key - c.key);
  return items.map((it) => it.pj);
};

/** Current tag of an owned or ghost atom (engine index). */
const tagOf = (sys: System, p: number, n: number): number => sys.state.id[p < n ? p : sys.nb.owner[p]];

/** fix reaxff/bonds Nevery filename (docs.lammps.org/fix_reaxff_bonds.html). */
export class FixReaxFFBonds extends Fix {
  readonly style = 'reaxff/bonds';
  private readonly file: string;
  private firstFlag = true;
  private appended = false;

  constructor(sys: System, id: string, group: string, args: string[], private pairRef: () => PairReaxFF | null) {
    super(sys, id, group, args);
    if (args.length !== 2) throw new StyleError('Illegal fix reaxff/bonds command: expected Nevery and a file name');
    this.nevery = intArg(args[0], 'Nevery');
    if (this.nevery <= 0) throw new StyleError(`Illegal fix reaxff/bonds nevery value ${this.nevery}`);
    this.file = args[1];
    if (this.file.includes('*')) throw new StyleError('fix reaxff/bonds with a * file name (one file per output step) is not supported');
    if (/\.(gz|bz2|zst|lz4|xz)$/.test(this.file)) throw new StyleError('fix reaxff/bonds with a compressed file name is not supported');
    this.pairRef()?.wantNative();
  }

  init(): void {
    const pair = this.pairRef();
    if (!pair) throw new StyleError('Cannot use fix reaxff/bonds without pair_style reaxff, reaxff/kk, or reaxff/omp');
    pair.wantNative();
  }

  /** Writes only at the first setup of the first run (native first_flag), as end_of_step. */
  setup(): void {
    if (this.firstFlag) this.output();
    this.firstFlag = false;
  }

  endOfStep(): void { this.output(); }

  private output(): void {
    if (this.file === 'NULL') return;
    const pair = this.pairRef();
    if (!pair || !pair.lastBonds || !pair.nativeInfo) throw new StyleError('fix reaxff/bonds: no reaxff evaluation to write');
    const s = this.sys.state;
    const st: RxSystem = pair.lastBonds;
    const nat = pair.nativeInfo;
    const n = st.n;
    const cut = pair.bondGraphCutoff();
    const order = nativeOrder(s);
    const withMol = hasMolecule(s);
    const rows: string[] = [];
    let maxnum = 0;
    for (let k = 0; k < n; k++) {
      const i = order[k];
      const items = nativeRegion(st.bonds, nat, i, n).filter((pj) => st.bonds.BO[pj] > cut);
      if (items.length > maxnum) maxnum = items.length;
      let line = ` ${s.id[i]} ${s.type[i]} ${items.length}`;
      for (const pj of items) line += ` ${tagOf(this.sys, st.bonds.nbr[pj], n)}`;
      line += ` ${withMol ? s.molecule[i] : 0}`;
      for (const pj of items) line += fixedWidth(st.bonds.BO[pj], 14, 3);
      line += fixedWidth(st.ws.total_bond_order[i], 14, 3) + fixedWidth(st.ws.nlp[i], 14, 3) + fixedWidth(s.q[i], 14, 3);
      rows.push(line + '\n');
    }
    const text = `# Timestep ${s.step}\n#\n# Number of particles ${n}\n#\n` +
      `# Max number of bonds per atom ${maxnum} with coarse bond order cutoff ${fixedWidth(cut, 5, 3)}\n` +
      '# Particle connection table and bond orders\n' +
      '# id type nb id_1...id_nb mol bo_1...bo_nb abo nlp q\n' +
      rows.join('') + '# \n';
    this.sys.writeFile(this.file, text, this.appended);
    this.appended = true;
  }
}

/** The atom pair key of a bond: the atom IDs and the periodic image of the partner. */
interface PairAverage { a: number; b: number; ta: number; tb: number; v: number }

/** fix reaxff/species Nevery Nrepeat Nfreq filename [cutoff ...] [element ...] (docs.lammps.org/fix_reaxff_species.html). */
export class FixReaxFFSpecies extends Fix {
  readonly style = 'reaxff/species';
  private readonly nrepeat: number;
  private readonly nfreq: number;
  private readonly file: string;
  /** nmol: molecules per species of the last output (NMol). */
  private nmol: number[] = [];
  private readonly ntypes: number;
  /** BOCut[i][j] (1-based), default 0.30. */
  private readonly bocut: Float64Array;
  private userElements: string[] | null = null;
  private eletype: string[] = [];
  private ueletype: string[] = [];
  private ele2uele: number[] = [];
  private aveNvalid: number;
  private aveIrepeat = 0;
  private sums = new Map<string, PairAverage>();
  private avg: PairAverage[] = [];
  private specNvalid = -1;
  private vectorNmole = 0;
  private vectorNspec = 0;
  private appended = false;

  constructor(sys: System, id: string, group: string, args: string[], private pairRef: () => PairReaxFF | null) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('Illegal fix reaxff/species command: expected Nevery Nrepeat Nfreq filename');
    this.nevery = intArg(args[0], 'Nevery');
    this.nrepeat = intArg(args[1], 'Nrepeat');
    this.nfreq = intArg(args[2], 'Nfreq');
    this.globalFreq = this.nfreq;
    this.file = args[3];
    if (/\.(gz|bz2|zst|lz4|xz)$/.test(this.file)) throw new StyleError('fix reaxff/species with a compressed file name is not supported');
    if (this.nevery <= 0) throw new StyleError(`Invalid fix reaxff/species nevery value ${this.nevery}`);
    if (this.nrepeat <= 0) throw new StyleError(`Invalid fix reaxff/species nrepeat value ${this.nrepeat}`);
    if (this.nfreq <= 0) throw new StyleError(`Invalid fix reaxff/species nfreq value ${this.nfreq}`);
    if ((this.nfreq % this.nevery) !== 0 || this.nrepeat * this.nevery > this.nfreq) {
      throw new StyleError('Incompatible fix reaxff/species nevery/nrepeat/nfreq settings');
    }
    this.vectorFlag = true;
    this.sizeVector = 2;
    this.extvector = 0;
    this.ntypes = sys.state.ntypes;
    this.bocut = new Float64Array((this.ntypes + 1) * (this.ntypes + 1)).fill(0.3);
    this.aveNvalid = this.nextValid(sys.state.step);
    this.pairRef()?.wantNative();

    // neighbor lists must stay unchanged during averaging (native resets neigh_modify when they could change)
    const nb = sys.nb;
    let rene = false;
    if (this.nevery * this.nrepeat !== 1 && (this.nfreq % nb.every !== 0 || nb.every < this.nevery * this.nrepeat)) {
      let newevery = this.nevery * this.nrepeat;
      while (this.nfreq % newevery !== 0 && newevery <= Math.trunc(this.nfreq / 2)) newevery++;
      if (this.nfreq % newevery !== 0) newevery = this.nfreq;
      nb.every = newevery;
      rene = true;
    }
    if (this.nevery * this.nrepeat !== 1 && (nb.delay !== 0 || nb.check)) {
      nb.delay = 0;
      nb.check = false;
      rene = true;
    }
    if (rene) {
      sys.warn(`Resetting reneighboring criteria to 'delay ${nb.delay} every ${nb.every} check no' due to fix reaxff/species averaging of bond data`);
    }
    this.parseKeywords(args.slice(4));
    sys.writeFile(this.file, '', false);
  }

  /** The bond order cutoffs and element names of the optional keywords. */
  private parseKeywords(args: string[]): void {
    let iarg = 0;
    while (iarg < args.length) {
      const key = args[iarg];
      if (key === 'cutoff') {
        if (iarg + 4 > args.length) throw new StyleError('Illegal fix reaxff/species cutoff command');
        const [ilo, ihi] = this.bounds(args[iarg + 1]);
        const [jlo, jhi] = this.bounds(args[iarg + 2]);
        const value = Number(args[iarg + 3]);
        if (!Number.isFinite(value) || value > 1.0 || value < 0.0) {
          throw new StyleError(`Fix reaxff/species invalid cutoff value: ${args[iarg + 3]}`);
        }
        for (let i = ilo; i <= ihi; i++) {
          for (let j = Math.max(jlo, i); j <= jhi; j++) {
            this.bocut[i * (this.ntypes + 1) + j] = value;
            this.bocut[j * (this.ntypes + 1) + i] = value;
          }
        }
        iarg += 4;
      } else if (key === 'element') {
        if (iarg + this.ntypes + 1 > args.length) throw new StyleError('Illegal fix reaxff/species element command');
        this.userElements = args.slice(iarg + 1, iarg + 1 + this.ntypes);
        iarg += this.ntypes + 1;
      } else if (key === 'delete' || key === 'delete_rate_limit' || key === 'delete_subgroup' || key === 'position') {
        throw new StyleError(`fix reaxff/species keyword ${key} is not ported (the molecule deletion and the position output are not supported)`);
      } else {
        throw new StyleError(`Unknown fix reaxff/species keyword: ${key}`);
      }
    }
  }

  /** utils::bounds for a type range: "*", "*n", "n*", "n*m" or "n". */
  private bounds(w: string | undefined): [number, number] {
    const max = this.ntypes;
    if (w === undefined) throw new StyleError('Illegal fix reaxff/species type range');
    const star = w.indexOf('*');
    let lo: number, hi: number;
    if (star < 0) {
      lo = hi = intArg(w, 'type');
    } else {
      const a = w.slice(0, star), b = w.slice(star + 1);
      lo = a === '' ? 1 : intArg(a, 'type');
      hi = b === '' ? max : intArg(b, 'type');
    }
    if (lo < 1 || hi > max || lo > hi) throw new StyleError(`Fix reaxff/species type range ${w} is out of bounds 1-${max}`);
    return [lo, hi];
  }

  /** fix_ave_atom.cpp nextvalid(): the first step on which the averaging ends after step. */
  private nextValid(step: number): number {
    let nv = Math.floor(step / this.nfreq) * this.nfreq + this.nfreq;
    if (nv - this.nfreq === step && this.nrepeat === 1) nv = step;
    else nv -= (this.nrepeat - 1) * this.nevery;
    if (nv < step) nv += this.nfreq;
    return nv;
  }

  /** The element names and their unique list (GetUniqueElements), done at the first setup. */
  private uniqueElements(): void {
    const pair = this.pairRef();
    if (this.userElements) this.eletype = this.userElements;
    else if (pair) this.eletype = pair.elementNames(this.ntypes);
    else throw new StyleError('Cannot use fix reaxff/species without a reaxff pair_style');
    const unique: string[] = [];
    for (const e of this.eletype) if (!unique.includes(e)) unique.push(e);
    // reorder C, H, O, N to the front when present (fix_reaxff_species.cpp GetUniqueElements)
    let incr = 0;
    for (const it of ['C', 'H', 'O', 'N']) {
      for (let j = incr; j < unique.length; j++) {
        if (unique[j] === it) {
          unique.splice(j, 1);
          unique.splice(incr++, 0, it);
          break;
        }
      }
    }
    this.ueletype = unique;
    this.ele2uele = this.eletype.map((e) => unique.indexOf(e));
  }

  init(): void {
    if (!this.pairRef()) throw new StyleError('Cannot use fix reaxff/species without a reaxff pair_style');
    this.pairRef()!.wantNative();
  }

  /** post_integrate of the first setup (native setup calls post_integrate). */
  setup(): void {
    this.uniqueElements();
    this.averageStep(this.sys.state.step);
    this.speciesStep(this.sys.state.step);
  }

  postIntegrate(): void {
    const step = this.sys.state.step;
    this.averageStep(step);
    this.speciesStep(step);
  }

  computeVector(i: number): number {
    if (i === 0) return this.vectorNmole;
    if (i === 1) return this.vectorNspec;
    return 0;
  }

  /** The pair's bond list of the last evaluation as (atom id, partner id, image, bond order) samples (pair FindBond). */
  private sample(): void {
    const pair = this.pairRef();
    const st = pair?.lastBonds;
    const nat = pair?.nativeInfo;
    if (!pair || !st || !nat) return;
    const s = this.sys.state;
    const nb = this.sys.nb;
    const n = st.n;
    const b = st.bonds;
    for (let i = 0; i < n; i++) {
      for (let pj = b.start[i]; pj < b.end[i]; pj++) {
        const bo = b.BO[pj];
        // FindBond: bo_tmp >= 0.10, and only partners not below i in native order
        if (!(bo >= 0.1)) continue;
        const p = b.nbr[pj];
        if (p < n && nat.nativeIdx[p] < nat.nativeIdx[i]) continue;
        const owner = nb.owner[p];
        const key = `${s.id[i]}:${s.id[owner]}:${nb.gimage[3 * p]},${nb.gimage[3 * p + 1]},${nb.gimage[3 * p + 2]}`;
        const e = this.sums.get(key);
        if (e) e.v += bo;
        else this.sums.set(key, { a: s.id[i], b: s.id[owner], ta: s.type[i], tb: s.type[owner], v: bo });
      }
    }
  }

  /** fix ave/atom end_of_step on the species schedule: samples, and the average at the end of each window. */
  private averageStep(step: number): void {
    if (step !== this.aveNvalid) return;
    if (this.aveIrepeat === 0) this.sums = new Map();
    this.sample();
    this.aveIrepeat++;
    if (this.aveIrepeat < this.nrepeat) {
      this.aveNvalid += this.nevery;
      return;
    }
    this.aveIrepeat = 0;
    this.aveNvalid = step + this.nfreq - (this.nrepeat - 1) * this.nevery;
    this.avg = [...this.sums.values()].map((e) => ({ ...e, v: e.v / this.nrepeat }));
  }

  /** post_integrate of fix reaxff/species (Output_ReaxFF_Bonds). */
  private speciesStep(step: number): void {
    if (step !== this.specNvalid && this.specNvalid !== -1) return;
    const { nmole, species } = this.findMolecules();
    if (step >= 0) this.writeFormulas(step, nmole, species);
    this.vectorNmole = nmole;
    this.vectorNspec = species.length;
    this.specNvalid = step + this.nfreq;
  }

  /** FindMolecule + SortMolecule + FindSpecies: molecules by the smallest atom ID, species by first appearance. */
  private findMolecules(): { nmole: number; species: string[] } {
    const s = this.sys.state;
    const n = s.n;
    const idmap = buildAtomMap(s);
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x: number): number => {
      while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    };
    const inGroup = (i: number): boolean => this.inGroup(i);
    for (const e of this.avg) {
      if (!(e.v > this.bocut[e.ta * (this.ntypes + 1) + e.tb])) continue;
      const ia = e.a < idmap.length ? idmap[e.a] : -1;
      const ib = e.b < idmap.length ? idmap[e.b] : -1;
      if (ia < 0 || ib < 0 || !inGroup(ia) || !inGroup(ib)) continue;
      const ra = find(ia), rb = find(ib);
      if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
    }
    const minTag = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      if (!inGroup(i)) continue;
      const r = find(i);
      const t = minTag.get(r);
      if (t === undefined || s.id[i] < t) minTag.set(r, s.id[i]);
    }
    const roots = [...minTag.keys()].sort((a, b) => minTag.get(a)! - minTag.get(b)!);
    const molOf = new Map<number, number>();
    roots.forEach((r, m) => molOf.set(r, m));
    const nu = this.ueletype.length;
    const molCounts: number[][] = roots.map(() => new Array<number>(nu).fill(0));
    for (let i = 0; i < n; i++) {
      if (!inGroup(i)) continue;
      const m = molOf.get(find(i))!;
      molCounts[m][this.ele2uele[s.type[i] - 1]]++;
    }
    const species: string[] = [];
    const counts: number[][] = [];
    const nmol: number[] = [];
    for (const c of molCounts) {
      const k = counts.findIndex((sp) => sp.every((v, j) => v === c[j]));
      if (k >= 0) nmol[k]++;
      else { counts.push(c); species.push(this.formula(c)); nmol.push(1); }
    }
    this.nmol = nmol;
    return { nmole: roots.length, species };
  }

  private formula(c: number[]): string {
    let name = '';
    for (let j = 0; j < c.length; j++) {
      if (c[j] !== 0) name += this.ueletype[j] + (c[j] !== 1 ? String(c[j]) : '');
    }
    return name;
  }

  /** WriteFormulas: the header with the species names, then the counts of this output step. */
  private writeFormulas(step: number, nmole: number, species: string[]): void {
    let text = '#  Timestep    No_Moles    No_Specs';
    for (const nm of species) text += ' ' + nm.padStart(11);
    text += '\n';
    text += `${String(step).padStart(11)} ${String(nmole).padStart(11)} ${String(species.length).padStart(11)}`;
    for (let k = 0; k < species.length; k++) text += ' ' + String(this.nmol[k]).padStart(11);
    text += '\n';
    this.sys.writeFile(this.file, text, this.appended);
    this.appended = true;
  }
}
