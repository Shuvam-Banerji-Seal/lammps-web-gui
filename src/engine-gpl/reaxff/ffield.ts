/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/reaxff_ffield.cpp (Read_Force_Field), src/REAXFF/reaxff_control.cpp
 * (Read_Control_File), src/REAXFF/reaxff_types.h (parameter structs), at commit b332f28.
 * Authors credited in the C++ file headers: Hasan Metin Aktulga (Purdue University, now
 * Lawrence Berkeley National Laboratory); heavily modified and adapted for LAMMPS by the
 * LAMMPS developers. Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import { pow as libmPow } from '../libm/pow';

/** Two character element name -> parameters (reaxff_types.h single_body_parameters). */
export interface SingleBody {
  name: string;
  r_s: number; valency: number; mass: number; r_vdw: number; epsilon: number; gamma: number;
  r_pi: number; valency_e: number; nlp_opt: number;
  alpha: number; gamma_w: number; valency_boc: number; p_ovun5: number; chi: number; eta: number;
  p_hbond: number;
  r_pi_pi: number; p_lp2: number; b_o_131: number; b_o_132: number; b_o_133: number; bcut_acks2: number;
  p_ovun2: number; p_val3: number; valency_val: number; p_val5: number; rcore2: number; ecore2: number;
  acore2: number;
  lgcij: number; lgre: number;
}

/** reaxff_types.h two_body_parameters (one per element pair, symmetric). */
export interface TwoBody {
  p_bo1: number; p_bo2: number; p_bo3: number; p_bo4: number; p_bo5: number; p_bo6: number;
  r_s: number; r_p: number; r_pp: number;
  p_boc3: number; p_boc4: number; p_boc5: number;
  p_be1: number; p_be2: number;
  De_s: number; De_p: number; De_pp: number;
  p_ovun1: number;
  D: number; alpha: number; r_vdW: number; gamma_w: number; rcore: number; ecore: number; acore: number;
  lgcij: number; lgre: number;
  gamma: number;
  v13cor: number; ovc: number;
  /**
   * pow(1/gamma_w, p_vdW1) for the p_vdW1 in `powgiP` and the gamma_w in `powgiG` (nonbonded.ts directPair: the same
   * for every pair of the types; NaN until the first use).
   */
  powgi: number;
  powgiP: number;
  powgiG: number;
}

export interface ThreeBodyPrm {
  theta_00: number; p_val1: number; p_val2: number; p_val4: number; p_val7: number;
  p_pen1: number; p_coa1: number;
}
export interface ThreeBodyHeader { cnt: number; prm: ThreeBodyPrm[] }

export interface HBondPrm { r0_hb: number; p_hb1: number; p_hb2: number; p_hb3: number }

export interface FourBodyPrm { V1: number; V2: number; V3: number; p_tor1: number; p_cot1: number }
export interface FourBodyHeader { cnt: number; prm: FourBodyPrm[] }

/** Parsed force field; tbp/thbp/hbp/fbp are flat, indexed by atom-type ordinals (see idx helpers). */
export interface ReaxParam {
  numAtomTypes: number;
  /** gp.l: the global parameters, indexed as in reaxff_types.h comments (l[0] .. l[n-1]). */
  gpL: Float64Array;
  /** vdw_type: 0 none, 1 shielding, 2 inner wall, 3 inner wall + shielding. */
  vdwType: number;
  sbp: SingleBody[];
  tbp: TwoBody[];
  thbp: ThreeBodyHeader[];
  hbp: HBondPrm[];
  fbp: FourBodyHeader[];
}

export interface ControlParams {
  tabulate: number;
  bondCut: number;
  nonbCut: number;
  nonbLow: number;
  hbondCut: number;
  bgCut: number;
  boCut: number;
  thbCut: number;
  thbCutsq: number;
  lgflag: number;
  enobondsflag: number;
}

const newSingle = (): SingleBody => ({
  name: '', r_s: 0, valency: 0, mass: 0, r_vdw: 0, epsilon: 0, gamma: 0, r_pi: 0, valency_e: 0, nlp_opt: 0,
  alpha: 0, gamma_w: 0, valency_boc: 0, p_ovun5: 0, chi: 0, eta: 0, p_hbond: 0,
  r_pi_pi: 0, p_lp2: 0, b_o_131: 0, b_o_132: 0, b_o_133: 0, bcut_acks2: 0,
  p_ovun2: 0, p_val3: 0, valency_val: 0, p_val5: 0, rcore2: 0, ecore2: 0, acore2: 0, lgcij: 0, lgre: 0,
});

const newTwo = (): TwoBody => ({
  p_bo1: 0, p_bo2: 0, p_bo3: 0, p_bo4: 0, p_bo5: 0, p_bo6: 0, r_s: 0, r_p: 0, r_pp: 0,
  p_boc3: 0, p_boc4: 0, p_boc5: 0, p_be1: 0, p_be2: 0, De_s: 0, De_p: 0, De_pp: 0, p_ovun1: 0,
  D: 0, alpha: 0, r_vdW: 0, gamma_w: 0, rcore: 0, ecore: 0, acore: 0, lgcij: 0, lgre: 0,
  gamma: 0, v13cor: 0, ovc: 0, powgi: NaN, powgiP: NaN, powgiG: NaN,
});

const newThree = (): ThreeBodyPrm => ({ theta_00: 0, p_val1: 0, p_val2: 0, p_val4: 0, p_val7: 0, p_pen1: 0, p_coa1: 0 });
const newFour = (): FourBodyPrm => ({ V1: 0, V2: 0, V3: 0, p_tor1: 0, p_cot1: 0 });

/** Line tokenizer over a potential file (TextFileReader semantics: blank lines skipped by next_values). */
class FfieldLines {
  private pos = 0;
  constructor(private lines: string[], private filename: string) {}
  /** next_values(0): the next line with at least one word; its words (comments are not stripped). */
  values(): string[] {
    while (this.pos < this.lines.length) {
      const w = this.lines[this.pos++].trim();
      if (w !== '') return w.split(/\s+/);
    }
    throw new StyleError(`${this.filename}: Missing line in ReaxFF parameter file!`);
  }
  /** next_line(): the next non-blank line, or null at end of file. */
  nextLine(): string[] | null {
    while (this.pos < this.lines.length) {
      const w = this.lines[this.pos++].trim();
      if (w !== '') return w.split(/\s+/);
    }
    return null;
  }
  /** skip_line(): the next physical line, blank or not. */
  skipLine(): void {
    if (this.pos >= this.lines.length) throw new StyleError(`${this.filename}: Missing line in ReaxFF parameter file!`);
    this.pos++;
  }
  /** Physical line number (1-based) of the line just read. */
  get lineno(): number { return this.pos; }
}

/** Values of one line, read left to right (ValueTokenizer). */
class Values {
  private i = 0;
  constructor(private words: string[], private where: () => string) {}
  get count(): number { return this.words.length; }
  hasNext(): boolean { return this.i < this.words.length; }
  skip(): void { this.i++; }
  string(): string { return this.words[this.i++] ?? this.fail('missing string'); }
  double(): number {
    const w = this.words[this.i++];
    if (w === undefined) this.fail('missing number');
    const v = Number(w);
    if (!Number.isFinite(v) && !/^[-+]?(inf|nan)/i.test(w)) this.fail(`invalid number '${w}'`);
    return v;
  }
  int(): number {
    const w = this.words[this.i++];
    if (w === undefined || !/^[-+]?\d+$/.test(w)) this.fail(`invalid integer '${w ?? ''}'`);
    return Number(w);
  }
  private fail(what: string): never {
    throw new StyleError(`${this.where()}: ${what}`);
  }
}

/**
 * Read_Force_Field: parses a ReaxFF potential file (the line structure of reaxff_ffield.cpp)
 * and returns the parameters with the control values the force field sets (bo_cut, nonb_low,
 * nonb_cut from the global parameters). lgflag must be 0 (lgvdw is refused by the pair style).
 */
export const readForceField = (text: string, filename: string, lgflag: number): { param: ReaxParam; control: Pick<ControlParams, 'boCut' | 'nonbLow' | 'nonbCut'> } => {
  const lines = new FfieldLines(text.split(/\r?\n/), filename);
  const where = () => `${filename}:${lines.lineno}`;
  const fail = (msg: string): never => { throw new StyleError(`${where()}: ${msg}`); };
  const check = (v: Values, want: number) => {
    if (v.count < want) fail(`Invalid force field file format: expected ${want} columns but found ${v.count}`);
  };
  const param: ReaxParam = {
    numAtomTypes: 0, gpL: new Float64Array(0), vdwType: 0, sbp: [], tbp: [], thbp: [], hbp: [], fbp: [],
  };
  // check if header comment line is present (the first line must not be a general-parameter count line)
  const first = lines.nextLine();
  if (first && /^[0-9]+\s+!.*general parameters/.test(first.join(' '))) {
    fail('First line of ReaxFF potential file must be a comment or empty');
  }
  let v = new Values(lines.values(), where);
  let n = v.int();
  if (n < 1) fail('Invalid number of global parameters');
  const gpL = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    v = new Values(lines.values(), where);
    gpL[i] = v.double();
  }
  param.gpL = gpL;
  // next line is number of atom types followed by 3 lines of comments
  v = new Values(lines.values(), where);
  n = v.int();
  param.numAtomTypes = n;
  lines.skipLine();
  lines.skipLine();
  lines.skipLine();
  const ntypes = n;
  const sbp: SingleBody[] = [];
  for (let i = 0; i < ntypes; i++) sbp.push(newSingle());
  param.sbp = sbp;
  const tbp: TwoBody[] = [];
  for (let k = 0; k < ntypes * ntypes; k++) tbp.push(newTwo());
  param.tbp = tbp;
  const T = (j: number, k: number) => tbp[j * ntypes + k];
  param.thbp = [];
  for (let k = 0; k < ntypes * ntypes * ntypes; k++) param.thbp.push({ cnt: 0, prm: [] });
  const THB = (j: number, k: number, l: number) => param.thbp[(j * ntypes + k) * ntypes + l];
  param.hbp = [];
  for (let k = 0; k < ntypes * ntypes * ntypes; k++) param.hbp.push({ r0_hb: 0, p_hb1: 0, p_hb2: 0, p_hb3: 0 });
  const HB = (j: number, k: number, l: number) => param.hbp[(j * ntypes + k) * ntypes + l];
  param.fbp = [];
  for (let k = 0; k < libmPow(ntypes, 4); k++) param.fbp.push({ cnt: 0, prm: [newFour()] });
  const FB = (j: number, k: number, l: number, m: number) => param.fbp[((j * ntypes + k) * ntypes + l) * ntypes + m];
  const torFlag = new Uint8Array(libmPow(ntypes, 4));
  const TF = (j: number, k: number, l: number, m: number) => ((j * ntypes + k) * ntypes + l) * ntypes + m;
  const gpVdw = { vdw: 0 };

  // atomic parameters: four lines per atom type (five with lgvdw, refused here)
  for (let i = 0; i < ntypes; i++) {
    const s = newSingle();
    v = new Values(lines.values(), where);
    if (v.count < 8 && !lgflag) fail("This force field file requires using 'lgvdw yes'");
    check(v, 9);
    s.name = v.string().toUpperCase().slice(0, 3);
    s.r_s = v.double();
    s.valency = v.double();
    s.mass = v.double();
    s.r_vdw = v.double();
    s.epsilon = v.double();
    s.gamma = v.double();
    s.r_pi = v.double();
    s.valency_e = v.double();
    s.nlp_opt = 0.5 * (s.valency_e - s.valency);
    v = new Values(lines.values(), where);
    check(v, 8);
    s.alpha = v.double();
    s.gamma_w = v.double();
    s.valency_boc = v.double();
    s.p_ovun5 = v.double();
    v.skip();
    s.chi = v.double();
    s.eta = 2.0 * v.double();
    s.p_hbond = Math.trunc(v.double());
    v = new Values(lines.values(), where);
    check(v, 8);
    s.r_pi_pi = v.double();
    s.p_lp2 = v.double();
    v.skip();
    s.b_o_131 = v.double();
    s.b_o_132 = v.double();
    s.b_o_133 = v.double();
    s.bcut_acks2 = v.double();
    v = new Values(lines.values(), where);
    check(v, 8);
    s.p_ovun2 = v.double();
    s.p_val3 = v.double();
    v.skip();
    s.valency_val = v.double();
    s.p_val5 = v.double();
    s.rcore2 = v.double();
    s.ecore2 = v.double();
    s.acore2 = v.double();
    s.lgcij = 0;
    s.lgre = 0;
    sbp[i] = s;
    // van der Waals settings check (warnings of reaxff_ffield.cpp are not reproduced)
    if (s.rcore2 > 0.01 && s.acore2 > 0.01) {
      if (s.gamma_w > 0.5) {
        if (gpVdw.vdw !== 0 && gpVdw.vdw !== 3) { /* keep the earlier setting */ } else gpVdw.vdw = 3;
      } else if (gpVdw.vdw !== 0 && gpVdw.vdw !== 2) { /* keep the earlier setting */ } else gpVdw.vdw = 2;
    } else if (s.gamma_w > 0.5) {
      if (gpVdw.vdw !== 0 && gpVdw.vdw !== 1) { /* keep the earlier setting */ } else gpVdw.vdw = 1;
    } else {
      fail(`Inconsistent van der Waals parameters: No shielding or inner wall set for element ${s.name}`);
    }
  }
  param.vdwType = gpVdw.vdw;
  // equate vval3 to valf for first-row elements (25/10/2004)
  for (let i = 0; i < ntypes; i++) {
    if (sbp[i].mass < 21 && sbp[i].valency_val !== sbp[i].valency_boc) sbp[i].valency_val = sbp[i].valency_boc;
  }

  // two body parameters
  v = new Values(lines.values(), where);
  n = v.int();
  lines.skipLine();
  for (let i = 0; i < n; i++) {
    v = new Values(lines.values(), where);
    check(v, 10);
    const j = v.int() - 1;
    const k = v.int() - 1;
    if (j < 0 || k < 0) fail('Inconsistent force field file');
    if (j < ntypes && k < ntypes) {
      const a = T(j, k), b = T(k, j);
      const set = (f: keyof TwoBody, val: number) => { a[f] = val; b[f] = val; };
      set('De_s', v.double());
      set('De_p', v.double());
      set('De_pp', v.double());
      set('p_be1', v.double());
      set('p_bo5', v.double());
      set('v13cor', v.double());
      set('p_bo6', v.double());
      set('p_ovun1', v.double());
    }
    v = new Values(lines.values(), where);
    check(v, 7);
    if (j < ntypes && k < ntypes) {
      const a = T(j, k), b = T(k, j);
      const set = (f: keyof TwoBody, val: number) => { a[f] = val; b[f] = val; };
      set('p_be2', v.double());
      set('p_bo3', v.double());
      set('p_bo4', v.double());
      v.skip();
      set('p_bo1', v.double());
      set('p_bo2', v.double());
      set('ovc', v.hasNext() ? v.double() : 0.0);
    }
  }
  for (let i = 0; i < ntypes; i++) {
    for (let j = i; j < ntypes; j++) {
      const a = T(i, j), b = T(j, i);
      const set = (f: keyof TwoBody, val: number) => { a[f] = val; b[f] = val; };
      set('r_s', 0.5 * (sbp[j].r_s + sbp[i].r_s));
      set('r_p', 0.5 * (sbp[j].r_pi + sbp[i].r_pi));
      set('r_pp', 0.5 * (sbp[j].r_pi_pi + sbp[i].r_pi_pi));
      set('p_boc3', Math.sqrt(sbp[j].b_o_132 * sbp[i].b_o_132));
      set('p_boc4', Math.sqrt(sbp[j].b_o_131 * sbp[i].b_o_131));
      set('p_boc5', Math.sqrt(sbp[j].b_o_133 * sbp[i].b_o_133));
      set('D', Math.sqrt(sbp[j].epsilon * sbp[i].epsilon));
      set('alpha', Math.sqrt(sbp[j].alpha * sbp[i].alpha));
      set('r_vdW', 2.0 * Math.sqrt(sbp[j].r_vdw * sbp[i].r_vdw));
      set('gamma_w', Math.sqrt(sbp[j].gamma_w * sbp[i].gamma_w));
      set('gamma', libmPow(sbp[j].gamma * sbp[i].gamma, -1.5));
      set('rcore', Math.sqrt(sbp[i].rcore2 * sbp[j].rcore2));
      set('ecore', Math.sqrt(sbp[i].ecore2 * sbp[j].ecore2));
      set('acore', Math.sqrt(sbp[i].acore2 * sbp[j].acore2));
      set('lgcij', Math.sqrt(sbp[i].lgcij * sbp[j].lgcij));
      set('lgre', 2.0 * gpL[35] * Math.sqrt(sbp[i].lgre * sbp[j].lgre));
    }
  }

  // two body off-diagonal parameters
  v = new Values(lines.values(), where);
  n = v.int();
  for (let i = 0; i < n; i++) {
    v = new Values(lines.values(), where);
    check(v, 8 + lgflag);
    const j = v.int() - 1;
    const k = v.int() - 1;
    if (j < 0 || k < 0) fail('Inconsistent force field file');
    if (j < ntypes && k < ntypes) {
      const a = T(j, k), b = T(k, j);
      let val = v.double();
      if (val > 0.0) { a.D = b.D = val; }
      val = v.double();
      if (val > 0.0) { a.r_vdW = b.r_vdW = 2 * val; }
      val = v.double();
      if (val > 0.0) { a.alpha = b.alpha = val; }
      val = v.double();
      if (val > 0.0) { a.r_s = b.r_s = val; }
      val = v.double();
      if (val > 0.0) { a.r_p = b.r_p = val; }
      val = v.double();
      if (val > 0.0) { a.r_pp = b.r_pp = val; }
    }
  }

  // three body parameters
  v = new Values(lines.values(), where);
  n = v.int();
  for (let i = 0; i < n; i++) {
    v = new Values(lines.values(), where);
    check(v, 10);
    const j = v.int() - 1;
    const k = v.int() - 1;
    const l = v.int() - 1;
    if (j < 0 || k < 0 || l < 0) fail('Inconsistent force field file');
    if (j < ntypes && k < ntypes && l < ntypes) {
      const h1 = THB(j, k, l), h2 = THB(l, k, j);
      // reaxff_ffield.cpp: cnt is read before the two increments, so a triple with j == l counts twice
      const cnt = h1.cnt;
      h1.cnt++;
      h2.cnt++;
      // order of the values: theta_00, p_val1, p_val2, p_coa1, p_val7, p_pen1, p_val4
      const p: ThreeBodyPrm = {
        theta_00: v.double(), p_val1: v.double(), p_val2: v.double(), p_coa1: v.double(),
        p_val7: v.double(), p_pen1: v.double(), p_val4: v.double(),
      };
      for (const h of [h1, h2]) {
        while (h.prm.length <= cnt) h.prm.push(newThree());
        h.prm[cnt] = { ...p };
      }
    }
  }

  // four body parameters
  v = new Values(lines.values(), where);
  n = v.int();
  for (let i = 0; i < n; i++) {
    v = new Values(lines.values(), where);
    check(v, 9);
    const j = v.int() - 1;
    const k = v.int() - 1;
    const l = v.int() - 1;
    const m = v.int() - 1;
    if (j < -1 || k < 0 || l < 0 || m < -1) fail('Inconsistent force field file');
    const val1 = v.double(), val2 = v.double(), val3 = v.double(), val4 = v.double(), val5 = v.double();
    if (j >= 0 && m >= 0) {
      // the entry is not in compact form
      if (j < ntypes && k < ntypes && l < ntypes && m < ntypes) {
        torFlag[TF(j, k, l, m)] = 1;
        torFlag[TF(m, l, k, j)] = 1;
        const a = FB(j, k, l, m), b = FB(m, l, k, j);
        a.cnt = 1; b.cnt = 1;
        const set = (p: FourBodyPrm) => { p.V1 = val1; p.V2 = val2; p.V3 = val3; p.p_tor1 = val4; p.p_cot1 = val5; };
        set(a.prm[0]);
        set(b.prm[0]);
      }
    } else if (k < ntypes && l < ntypes) {
      // entry of the form 0-X-Y-0
      for (let p = 0; p < ntypes; p++) {
        for (let o = 0; o < ntypes; o++) {
          if (torFlag[TF(p, k, l, o)] === 0) {
            const a = FB(p, k, l, o);
            a.cnt = 1;
            a.prm[0] = { V1: val1, V2: val2, V3: val3, p_tor1: val4, p_cot1: val5 };
          }
          if (torFlag[TF(o, l, k, p)] === 0) {
            const a = FB(o, l, k, p);
            a.cnt = 1;
            a.prm[0] = { V1: val1, V2: val2, V3: val3, p_tor1: val4, p_cot1: val5 };
          }
        }
      }
    }
  }

  // hydrogen bond parameters (this block may be missing: the defaults are r0_hb = -1)
  for (let i = 0; i < ntypes; i++) for (let j = 0; j < ntypes; j++) for (let k = 0; k < ntypes; k++) HB(i, j, k).r0_hb = -1.0;
  const hline = lines.nextLine();
  if (hline) {
    v = new Values(hline, where);
    n = v.int();
    for (let i = 0; i < n; i++) {
      v = new Values(lines.values(), where);
      check(v, 7);
      const j = v.int() - 1;
      const k = v.int() - 1;
      const l = v.int() - 1;
      if (j < 0 || k < 0 || l < 0) fail('Inconsistent force field file');
      if (j < ntypes && k < ntypes && l < ntypes) {
        const h = HB(j, k, l);
        h.r0_hb = v.double();
        h.p_hb1 = v.double();
        h.p_hb2 = v.double();
        h.p_hb3 = v.double();
      }
    }
  }

  // the C++ arrays are zero-filled up to their size: pad the three-body headers to their count
  for (const h of param.thbp) while (h.prm.length < h.cnt) h.prm.push(newThree());

  // apply global parameters to global control settings (reaxff_ffield.cpp)
  return {
    param,
    control: { boCut: 0.01 * gpL[29], nonbLow: gpL[11], nonbCut: gpL[12] },
  };
};

/** Parses the ReaxFF control file (Read_Control_File); the defaults are those of settings() without a file. */
export const readControlFile = (text: string, filename: string): Partial<ControlParams> => {
  const out: Partial<ControlParams> = {};
  const inactive = new Set([
    'ensemble_type', 'nsteps', 'dt', 'proc_by_dim', 'random_vel', 'restart_format', 'restart_freq',
    'reposition_atoms', 'restrict_bonds', 'remove_CoM_vel', 'debug_level', 'reneighbor',
    'vlist_buffer', 'ghost_cutoff', 'qeq_freq', 'q_err', 'ilu_refactor', 'ilu_droptol', 'temp_init',
    'temp_final', 't_mass', 't_mode', 't_rate', 't_freq', 'pressure', 'p_mass', 'pt_mass',
    'compress', 'press_mode', 'geo_format', 'traj_compress', 'traj_method', 'molecular_analysis',
    'ignore', 'dipole_anal', 'freq_dipole_anal', 'diffusion_coef', 'freq_diffusion_coef',
    'restrict_type', 'traj_title', 'simulation_name', 'energy_update_freq', 'atom_info',
    'atom_velocities', 'atom_forces', 'bond_info', 'angle_info',
  ]);
  const lines = text.split(/\r?\n/);
  lines.forEach((raw, idx) => {
    const w = raw.trim();
    if (w === '') return;
    const words = w.split(/\s+/);
    const keyword = words[0];
    const at = `${filename}:${idx + 1}`;
    if (words.length < 2) throw new StyleError(`${at}: No value(s) for control parameter: ${keyword}`);
    const num = (): number => {
      const x = Number(words[1]);
      if (!Number.isFinite(x)) throw new StyleError(`${at}: invalid value '${words[1]}' for ${keyword}`);
      return x;
    };
    if (inactive.has(keyword)) return;
    switch (keyword) {
      case 'nbrhood_cutoff': out.bondCut = num(); break;
      case 'bond_graph_cutoff': out.bgCut = num(); break;
      case 'thb_cutoff': out.thbCut = num(); break;
      case 'thb_cutoff_sq': out.thbCutsq = num(); break;
      case 'hbond_cutoff': out.hbondCut = num(); break;
      case 'tabulate_long_range': out.tabulate = Math.trunc(num()); break;
      case 'write_freq': break; // trajectories were removed upstream; a warning only
      default: throw new StyleError(`${at}: Unknown parameter ${keyword} in control file`);
    }
  });
  return out;
};
