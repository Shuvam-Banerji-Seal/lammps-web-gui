import type { Handler } from './args';
import { num, yesno, missingArgs, logical, numeric, inumeric, bounds } from './args';
import { StyleError, type Bonded, type KSpace } from '../force/types';
import { PAIR_STYLES, BOND_STYLES, ANGLE_STYLES, DIHEDRAL_STYLES, IMPROPER_STYLES, KSPACE_STYLES, styleNames } from '../styles';
import type { System } from '../system';
import type { MixRule, Pair } from '../force/types';
import { PairHybrid } from '../force/pair/hybrid';
import { atomSubStyles, isMolecularStyle, hasMolecule } from '../atoms';
import { typeWord } from '../labelmap';
import { KSpacePPPM } from '../force/kspace/pppm';
import { KSpaceEwald } from '../force/kspace/ewald';
import { pppmGplHook } from '../force/kspace/pppmHook';
import { removedStyleError } from '../deprecated';

/*
 * Force-field commands. Styles come from styles.ts; anything else is an
 * error that names the style and lists the supported ones.
 */

const unsupported = (kind: string, name: string, list: Record<string, unknown>) =>
  (kind === 'pair_style' ? removedStyleError('pair_style', name) : null) ?? new StyleError(`${kind} '${name}' is not supported by the browser engine; supported: ${(styleNames()[kind] ?? Object.keys(list).sort()).join(', ') || 'none yet'}`);

/** pair_style style args — pair_style.html; "pair_style none" removes it. */
const pairStyle: Handler = ({ sys }, a) => {
  const name = a[0];
  if (!name) throw new StyleError('usage: pair_style style args');
  sys.ff.pairNotRestarted = null;
  if (name === 'none') { sys.ff.pair = null; sys.bump(); return; }
  const make = PAIR_STYLES[name];
  if (!make) throw unsupported('pair_style', name, PAIR_STYLES);
  // "If the pair style is redefined ... the pair coefficients are lost" unless the style is the same
  const same = sys.ff.pair && sys.ff.pair.name === name ? sys.ff.pair : null;
  const p = same ?? make();
  if (p instanceof PairHybrid) p.setVariableEvaluator((v) => sys.equalVariable(v));
  p.settings(a.slice(1), sys.styleContext());
  if (!same && sys.hasBox) p.allocate(sys.state.ntypes);
  sys.ff.pair = p;
  sys.bump();
};

/** pair_coeff I J args — pair_coeff.html. */
const pairCoeff: Handler = ({ sys }, a) => {
  const s = sys.state;
  const p = sys.ff.pair;
  if (!p) throw new StyleError('pair_coeff needs a pair_style first');
  if (p.ntypes !== s.ntypes) p.allocate(s.ntypes);
  if (a.length < 2) throw new StyleError('usage: pair_coeff I J args');
  if (p.rawCoeff?.(a)) {
    p.coeff(a, sys.styleContext());
    sys.bump();
    return;
  }
  // pair_coeff.html: "I,J = numeric atom types (see asterisk form below), or type labels"
  const args = [typeWord(sys.labels, 'atom', a[0]), typeWord(sys.labels, 'atom', a[1]), ...a.slice(2)];
  p.coeff(args, sys.styleContext());
  sys.bump();
};

/** The keywords of native Pair::modify_params (src/pair.cpp at b332f28). */
const PAIR_MODIFY_KEYS = new Set(['mix', 'shift', 'table', 'table/disp', 'tabinner', 'tabinner/disp', 'tail', 'compute', 'nofdotr', 'neigh/trim']);

/**
 * pair_modify keyword value ... — pair_modify.html (pair, special, mix, shift,
 * tail, table, tabinner, compute). For hybrid styles: "the specified
 * parameters are by default modified for all the hybrid sub-styles"; "The
 * pair keyword can only be used with the hybrid and hybrid/overlay pair
 * styles. If used, it must appear first in the list of keywords."; "The
 * special and compute/tally keywords can only be used in conjunction with
 * the pair keyword and they must directly follow it."
 */
const pairModify: Handler = ({ sys }, a) => {
  const p = sys.ff.pair;
  if (!p) throw new StyleError('pair_modify needs a pair_style first');
  // As native LAMMPS does (src/pair.cpp Pair::modify_params and src/pair_hybrid.cpp PairHybrid::modify_params at
  // b332f28): the keywords are read in order with native's checks and messages
  if (a.length === 0) throw p instanceof PairHybrid ? new StyleError('Illegal pair_modify command') : missingArgs('pair_modify');
  let targets: Pair[] = [p];
  // the sub-style named by the pair keyword: compute and compute/tally apply to it alone
  let only: Pair | null = null;
  let k = 0;
  if (p instanceof PairHybrid) {
    targets = [p, ...p.subs.map((x) => x.style)];
    if (a[0] === 'pair') {
      const { sub, used } = p.selectSub(a.slice(1), 'pair_modify pair');
      // the selected sub-style first (style-specific keywords go to it); the hybrid's own
      // shift / tail flags follow, as native LAMMPS rejects shift on one sub-style with tail on another
      targets = [sub.style, p];
      only = sub.style;
      k = 1 + used;
      for (;;) {
        if (a[k] === 'special') {
          const w = [num(a[k + 2], 'special'), num(a[k + 3], 'special'), num(a[k + 4], 'special')] as [number, number, number];
          p.setSpecial(sub, a[k + 1] ?? '', w);
          k += 5;
        } else if (a[k] === 'compute/tally') {
          // pair_modify.html: "disables or enables registering compute */tally computes for the sub-style". The
          // engine always registers them (yes); no is refused. Measured with native LAMMPS (black box): a
          // compute group/group pair sum is the same with compute/tally no and yes (the lj/cut energy stays in it).
          if (!yesno(a[k + 1], 'compute/tally')) throw new StyleError('pair_modify pair compute/tally no is not supported by the browser engine');
          k += 2;
        } else break;
      }
    }
  }
  for (; k < a.length;) {
    const key = a[k];
    const v = a[k + 1];
    // every keyword but nofdotr takes one value
    if (key !== 'nofdotr' && PAIR_MODIFY_KEYS.has(key) && k + 2 > a.length) throw missingArgs(`pair_modify ${key}`);
    switch (key) {
      case 'mix':
        if (!['geometric', 'arithmetic', 'sixthpower'].includes(v)) throw new StyleError(`Unknown pair_modify mix argument: ${v}`);
        for (const t of targets) t.mix = v as MixRule;
        k += 2;
        break;
      case 'shift': { const y = logical(v); for (const t of targets) t.shift = y; k += 2; break; }
      case 'tail': { const y = logical(v); for (const t of targets) t.tail = y; k += 2; break; }
      case 'table': {
        // 0 is the series only; N must be 8 to 32 bits otherwise (the bits of a float)
        const n = inumeric(v);
        if (n > 32) throw new StyleError('Too many total bits for bitmapped Coulomb lookup table');
        if (n !== 0 && n < 8) throw new StyleError('Too few total bits for bitmapped Coulomb lookup table');
        for (const t of targets) t.table = n;
        k += 2;
        break;
      }
      case 'table/disp': {
        const n = inumeric(v);
        if (n > 32) throw new StyleError('Too many total bits for bitmapped Dispersion lookup table');
        if (n !== 0 && n < 8) throw new StyleError('Too few total bits for bitmapped Dispersion lookup table');
        // only lj/long/coul/long (flag_lj long) has a dispersion table in the engine
        for (const t of targets) {
          if (n > 0 && !('ljLong' in t)) throw new StyleError(`pair_modify table/disp ${n} is not supported for this pair style (only lj/long/coul/long has a dispersion table)`);
          (t as Pair & { tableDisp?: number }).tableDisp = n;
        }
        k += 2;
        break;
      }
      case 'tabinner': {
        // pair_modify.html: "The *tabinner* and *tabinner/disp* keywords set an inner cutoff above which the
        // pairwise computation is done by table lookup"; the default is sqrt(2.0). The erfc tables take it from here.
        const c = numeric(v);
        if (!(c > 0)) throw new StyleError(`pair_modify tabinner must be > 0 (got ${v})`);
        // sqrt(2) squared is 2 exactly (the default table boundary of erfc.ts)
        for (const t of targets) t.tabinnerRsq = Math.abs(c - Math.SQRT2) < 1e-15 ? 2 : c * c;
        k += 2;
        break;
      }
      case 'tabinner/disp': {
        // pair_modify.html: "The *tabinner* and *tabinner/disp* keywords set an inner cutoff above which the pairwise
        // computation is done by table lookup"; the dispersion table is that of lj/long/coul/long (flag_lj long); the
        // other styles have no dispersion table, so only the default sqrt(2.0) is accepted for them
        const c = numeric(v);
        if (!(c > 0)) throw new StyleError(`pair_modify ${key} must be > 0 (got ${v})`);
        for (const t of targets) {
          if ('ljLong' in t) (t as Pair & { tabinnerDisp?: number }).tabinnerDisp = c;
          else if (Math.abs(c - Math.SQRT2) > 1e-12) throw new StyleError(`pair_modify ${key} ${v}: only the default sqrt(2.0) is supported for this pair style (no dispersion table)`);
        }
        k += 2;
        break;
      }
      case 'compute': {
        // pair_modify.html: "The compute keyword allows pairwise computations to be turned off"
        const y = logical(v);
        for (const t of only ? [only] : targets) t.computeOn = y;
        k += 2;
        break;
      }
      case 'nofdotr':
        // pair_modify.html: "The nofdotr keyword allows to disable an optimization that computes the global stress
        // tensor from the total forces ..." The engine's virial is the same sum; only roundoff can differ (no value)
        k += 1;
        break;
      case 'neigh/trim': logical(v); k += 2; break;
      // pair, special and compute/tally anywhere else (or with a pair style that is not hybrid) are unknown keywords
      // to native's Pair::modify_params, as is any other word: no pair style of the engine adds keywords of its own
      default: throw new StyleError(`Unknown pair_modify keyword: ${key}`);
    }
  }
  sys.bump();
};

const bondedKinds = {
  bond: { list: BOND_STYLES, types: (sys: System) => sys.state.topo.nbondtypes },
  angle: { list: ANGLE_STYLES, types: (sys: System) => sys.state.topo.nangletypes },
  dihedral: { list: DIHEDRAL_STYLES, types: (sys: System) => sys.state.topo.ndihedraltypes },
  improper: { list: IMPROPER_STYLES, types: (sys: System) => sys.state.topo.nimpropertypes },
} as const;

const bondedStyle = (kind: keyof typeof bondedKinds): Handler => (({ sys }, a) => {
  const name = a[0];
  if (!name) throw new StyleError(`usage: ${kind}_style style args`);
  if (!isMolecularStyle(sys.atomStyle)) {
    throw new StyleError(`${kind}_style needs a molecular atom_style (bond, angle, molecular or full); the current one is ${sys.atomStyle}`);
  }
  if (name === 'none') { sys.ff[kind] = null; sys.bump(); return; }
  const make = bondedKinds[kind].list[name];
  if (!make) throw unsupported(`${kind}_style`, name, bondedKinds[kind].list);
  // measured with native LAMMPS (black box): re-issuing a hybrid style clears its coefficients, so it is always made anew
  const same = name !== 'hybrid' && sys.ff[kind] && sys.ff[kind]!.name === name ? sys.ff[kind] : null;
  const st: Bonded = same ?? make();
  st.settings(a.slice(1), sys.styleContext());
  if (!same && sys.hasBox) st.allocate(bondedKinds[kind].types(sys));
  sys.ff[kind] = st;
  if (kind === 'bond') attachBondStoreLocal(sys, st);
  sys.bump();
}) as Handler;

/**
 * bond_style bpm/* store/local ID N values (bond_bpm.cpp settings): the broken-bond rows go to fix store/local ID,
 * which is created with the bond style when no fix of that ID exists (the native BondBPM settings does the same, and
 * reuses an existing fix of that ID). The rows are read back by dump local f_ID.
 */
const attachBondStoreLocal = (sys: System, st: Bonded): void => {
  const req = st.localStoreRequest?.();
  if (!req || !st.attachLocalSink) return;
  const k = sys.fixes.findIndex((f) => f.id === req.id);
  if (k >= 0 && sys.fixes[k].style.toLowerCase() !== 'store/local') {
    throw new StyleError(`bond_style store/local ${req.id}: fix ${req.id} exists and is not a STORE/LOCAL fix`);
  }
  let fix: ReturnType<typeof req.create>;
  if (k >= 0) {
    fix = sys.fixes[k] as unknown as ReturnType<typeof req.create>;
    if (fix.nvalues !== req.names.length) throw new StyleError(`bond_style store/local ${req.id}: fix ${req.id} stores ${fix.nvalues} values, not ${req.names.length}`);
  } else {
    fix = req.create(sys) as unknown as ReturnType<typeof req.create>;
    sys.fixes.push(fix as unknown as (typeof sys.fixes)[number]);
  }
  st.attachLocalSink((row) => fix.addRow(row));
  sys.refreshComputes();
};

const bondedCoeff = (kind: keyof typeof bondedKinds): Handler => (({ sys }, a) => {
  const st = sys.ff[kind];
  if (!st) throw new StyleError(`${kind}_coeff needs a ${kind}_style first`);
  const n = bondedKinds[kind].types(sys);
  if (n === 0) throw new StyleError(`${kind}_coeff: there are no ${kind} types (create_box ... ${kind}/types N, or the data file header)`);
  if (st.ntypes !== n) st.allocate(n);
  // bond_coeff.html etc.: "N = numeric bond type (see asterisk form below), or type label"
  const args = a.length ? [typeWord(sys.labels, kind, a[0]), ...a.slice(1)] : a;
  st.coeff(args, sys.styleContext());
  sys.bump();
}) as Handler;

/**
 * special_bonds keyword values — special_bonds.html ("Each time you use this command it sets all the coefficients to
 * default values and only overrides the one you specify"). As native LAMMPS does (src/force.cpp Force::set_special at
 * b332f28): a missing value, an unknown keyword, a one/five value other than yes or no and a weight outside 0..1
 * (checked after the last keyword) all stop with Illegal special_bonds command.
 */
const specialBonds: Handler = ({ sys }, a) => {
  const illegal = () => new StyleError('Illegal special_bonds command');
  if (a.length === 0) throw illegal();
  const sp = { lj: [0, 0, 0] as [number, number, number], coul: [0, 0, 0] as [number, number, number], angle: false, dihedral: false, one5: false };
  const w3 = (k: number): [number, number, number] => {
    if (k + 4 > a.length) throw illegal();
    return [numeric(a[k + 1]), numeric(a[k + 2]), numeric(a[k + 3])];
  };
  for (let k = 0; k < a.length;) {
    switch (a[k]) {
      // special_bonds.html: "The amber keyword sets the 3 coefficients to 0.0, 0.0, 0.5 for LJ interactions and to
      // 0.0, 0.0, 0.8333 for Coulombic interactions ... where the last value is really 5/6."
      case 'amber': sp.lj = [0, 0, 0.5]; sp.coul = [0, 0, 5 / 6]; k++; break;
      case 'charmm': sp.lj = [0, 0, 0]; sp.coul = [0, 0, 0]; k++; break;
      case 'dreiding': sp.lj = [0, 0, 1]; sp.coul = [0, 0, 1]; k++; break;
      case 'fene': sp.lj = [0, 1, 1]; sp.coul = [0, 1, 1]; k++; break;
      case 'lj/coul': sp.lj = w3(k); sp.coul = [...sp.lj]; k += 4; break;
      case 'lj': sp.lj = w3(k); k += 4; break;
      case 'coul': sp.coul = w3(k); k += 4; break;
      case 'angle': if (k + 2 > a.length) throw illegal(); sp.angle = logical(a[k + 1]); k += 2; break;
      case 'dihedral': if (k + 2 > a.length) throw illegal(); sp.dihedral = logical(a[k + 1]); k += 2; break;
      case 'one/five':
        // special_bonds.html: "required by some pair styles, such as pair_style amoeba and hippo". The 1-5 list
        // is kept only by atom_style amoeba. Measured with native LAMMPS (black box): atom_style full refuses
        // one/five yes ("Cannot set special_bonds one/five if atom style does not support it").
        if (k + 2 > a.length || (a[k + 1] !== 'yes' && a[k + 1] !== 'no')) throw illegal();
        sp.one5 = a[k + 1] === 'yes';
        if (sp.one5 && !atomSubStyles(sys.atomStyle).includes('amoeba')) {
          throw new StyleError('Cannot set special_bonds one/five if atom style does not support it');
        }
        k += 2;
        break;
      default: throw illegal();
    }
  }
  if ([...sp.lj, ...sp.coul].some((x) => x < 0 || x > 1)) throw illegal();
  sys.ff.special = sp;
  if (sys.hasBox) sys.ff.topologyChanged(sys.state, sys.nb);
  sys.nb.lastBuild = -1;
  sys.bump();
};

/** dielectric value — dielectric.html. */
const dielectric: Handler = ({ sys }, a) => {
  const v = num(a[0], 'dielectric constant');
  if (!(v > 0)) throw new StyleError('dielectric constant must be > 0');
  sys.ff.dielectric = v;
  sys.bump();
};

/** kspace_style style accuracy — kspace_style.html; "kspace_style none" removes it. */
const kspaceStyle: Handler = ({ sys }, a) => {
  const name = a[0];
  if (!name) throw new StyleError('usage: kspace_style style args');
  if (name === 'none') { sys.ff.kspace = null; sys.bump(); return; }
  const make = KSPACE_STYLES[name];
  if (!make) throw unsupported('kspace_style', name, KSPACE_STYLES);
  const k = make();
  k.settings(a.slice(1), sys.styleContext());
  sys.ff.kspace = k;
  sys.bump();
};

/**
 * The keywords of native KSpace::modify_params (src/kspace.cpp at b332f28) and how many values each takes; any other
 * word goes to the style's own modify_param, and one it does not take is an illegal kspace_modify command.
 */
const KSPACE_MODIFY_KEYS = new Map<string, number>([
  ['mesh', 3], ['mesh/disp', 3], ['order', 1], ['order/disp', 1], ['minorder', 1], ['overlap', 1], ['force', 1],
  ['gewald', 1], ['gewald/disp', 1], ['slab', 1], ['wire', 1], ['amat', 1], ['compute', 1], ['fftbench', 1],
  ['collective', 1], ['collective/self/copy', 1], ['nonblocking', 1], ['diff', 1], ['cutoff/adjust', 1],
  ['kmax/ewald', 3], ['mix/disp', 1], ['force/disp/real', 1], ['force/disp/kspace', 1], ['splittol', 1], ['eigtol', 1],
  ['pressure/scalar', 1], ['disp/auto', 1],
]);

/** kspace_modify keyword value ... — kspace_modify.html. */
const kspaceModify: Handler = ({ sys }, a) => {
  const first = sys.ff.kspace;
  if (!first) throw new StyleError('Kspace_modify command before kspace_style is defined');
  let k: KSpace = first;
  for (let i = 0; i < a.length;) {
    const key = a[i];
    const nval = KSPACE_MODIFY_KEYS.get(key);
    // native names splittol for eigtol too
    if (nval !== undefined && i + 1 + nval > a.length) throw missingArgs(`kspace_modify ${key === 'eigtol' ? 'splittol' : key}`);
    if (key === 'diff') {
      // kspace_modify.html: "diff value = ad or ik = 2 or 4 FFTs for PPPM in smoothed or non-smoothed mode"
      if (a[i + 1] !== 'ad' && a[i + 1] !== 'ik') throw new StyleError(`Unknown kspace_modify diff flag ${a[i + 1]}`);
      // ewald reads the flag and never uses it (KSpaceEwald.modify)
      if (a[i + 1] === 'ad' && !(k instanceof KSpaceEwald)) {
        // the native PPPM helper (gpl_pppm) builds the analytic-differentiation style
        const ad: KSpace | null = k instanceof KSpacePPPM ? pppmGplHook().promoteAd(k) : null;
        if (!ad) throw new StyleError(`kspace_modify diff ad is not supported by kspace_style ${k.name}`);
        k = ad;
        sys.ff.kspace = ad;
      }
    }
    if (key === 'compute') {
      // kspace_modify.html: "The compute keyword allows Kspace computations to be turned off, even though a kspace_style is defined."
      // native KSpace::modify_params (src/kspace.cpp at b332f28) sets compute_flag for every kspace style
      k.computeOn = logical(a[i + 1]);
      i += 2;
      continue;
    }
    const used = k.modify(key, a.slice(i + 1));
    if (used <= 0) {
      throw new StyleError(nval !== undefined
        ? `kspace_modify ${[key, ...a.slice(i + 1, i + 1 + nval)].join(' ')} is not supported by the browser engine for kspace_style ${k.name}`
        : `Illegal kspace_modify command for kspace style ${k.name}`);
    }
    i += 1 + used;
  }
  sys.bump();
};

/** neighbor skin style — neighbor.html ("style = bin or nsq or multi"; all give the same answers). */
const neighbor: Handler = ({ sys }, a) => {
  const skin = num(a[0], 'skin');
  if (skin < 0) throw new StyleError('neighbor skin must be >= 0');
  if (!['bin', 'nsq', 'multi', 'multi/old'].includes(a[1] ?? '')) throw new StyleError('usage: neighbor skin bin|nsq|multi');
  sys.nb.skin = skin;
  sys.nb.multiStyle = a[1] === 'multi';
  sys.nb.nsqStyle = a[1] === 'nsq';
  sys.nb.lastBuild = -1;
  sys.bump();
};

/**
 * neigh_modify keyword values — neigh_modify.html. The keywords are read in order with native's checks and messages,
 * as native LAMMPS does (src/neighbor.cpp Neighbor::modify_params at b332f28); the checks that need the force field
 * (delay against every, page against one, the exclusion types) are made at run setup (Neighbor.init).
 */
const neighModify: Handler = ({ sys }, a) => {
  const nb = sys.nb;
  for (let k = 0; k < a.length;) {
    const key = a[k];
    const need = (n: number, what = key): void => { if (k + n > a.length) throw missingArgs(`neigh_modify ${what}`); };
    switch (key) {
      case 'every':
        need(2);
        nb.every = inumeric(a[k + 1]);
        if (nb.every <= 0) throw new StyleError(`Invalid neigh_modify every argument: ${nb.every}`);
        k += 2;
        break;
      case 'delay':
        need(2);
        nb.delay = inumeric(a[k + 1]);
        if (nb.delay < 0) throw new StyleError(`Invalid neigh_modify delay argument: ${nb.delay}`);
        k += 2;
        break;
      case 'check': need(2); nb.check = logical(a[k + 1]); k += 2; break;
      case 'once': need(2); nb.once = logical(a[k + 1]); k += 2; break;
      case 'page': need(2); nb.pgsize = inumeric(a[k + 1]); k += 2; break;
      case 'one': need(2); nb.oneatom = inumeric(a[k + 1]); k += 2; break;
      // "binsize value = size ... 0.0 = use default binsize of 1/2 the neighbor cutoff"; native treats any value <= 0 so
      case 'binsize': { need(2); const b = numeric(a[k + 1]); nb.binsize = b > 0 ? b : 0; k += 2; break; }
      case 'cluster': need(2); logical(a[k + 1]); k += 2; break;
      case 'include': {
        // "include value = group-ID"; "If the include option is used, the atom_modify first command must be used
        // with the same group"
        need(2);
        const g = a[k + 1];
        if (sys.groups.find(g) < 0) throw new StyleError(`Invalid include keyword: group ${g} not found`);
        if (sys.firstGroup === null) throw new StyleError('Invalid include keyword: atom_modify first command must be used');
        if (g !== sys.firstGroup) throw new StyleError(`Neigh_modify include group != atom_modify first group: ${sys.firstGroup}`);
        nb.includeBit = sys.groupBit(g);
        k += 2;
        break;
      }
      case 'exclude': {
        need(2);
        const kind = a[k + 1];
        if (kind === 'type') {
          // "M,N = exclude if one atom in pair is type M, other is type N (M and N may be type labels)"; the range is
          // checked at run setup
          need(4, 'exclude type');
          const t1 = inumeric(typeWord(sys.labels, 'atom', a[k + 2]));
          const t2 = inumeric(typeWord(sys.labels, 'atom', a[k + 3]));
          nb.excludes.push({ kind, a: t1, b: t2 });
          k += 4;
        } else if (kind === 'group') {
          need(4, 'exclude group');
          for (const g of [a[k + 2], a[k + 3]]) {
            if (sys.groups.find(g) < 0) throw new StyleError(`Invalid exclude group keyword: group ${g} not found`);
          }
          const b1 = sys.groupBit(a[k + 2]), b2 = sys.groupBit(a[k + 3]);
          if (sys.groups.isDynamic(b1) || sys.groups.isDynamic(b2)) throw new StyleError('Neigh_modify exclude group is not compatible with dynamic groups');
          nb.excludes.push({ kind, a: b1, b: b2 });
          k += 4;
        } else if (kind === 'molecule/intra' || kind === 'molecule/inter') {
          need(3, 'exclude molecule');
          if (!(sys.hasBox ? hasMolecule(sys.state) : isMolecularStyle(sys.atomStyle))) {
            throw new StyleError('Neigh_modify exclude molecule requires atom attribute molecule');
          }
          // native's message has no space after the colon
          if (sys.groups.find(a[k + 2]) < 0) throw new StyleError(`Invalid exclude keyword:group ${a[k + 2]} not found`);
          nb.excludes.push({ kind, a: sys.groupBit(a[k + 2]) });
          k += 3;
        } else if (kind === 'none') {
          nb.excludes = [];
          k += 2;
        } else throw new StyleError(`Unknown neigh_modify exclude keyword: ${kind}`);
        break;
      }
      case 'collection/interval': {
        // "collection/interval arg = Nc c1 c2 ... cNc"; "This option only applies to the multi neighbor style"
        if (!nb.multiStyle) throw new StyleError('Cannot use collection/interval command without multi setting');
        need(2);
        const n = inumeric(a[k + 1]);
        if (n < 1) throw new StyleError(`Invalid collection/interval keyword: illegal number of custom collections: ${n}`);
        if (k + 2 + n > a.length) throw new StyleError(`Invalid collection/interval keyword: expected ${n} separate lists of types`);
        let prev = -Infinity;
        for (let c = 0; c < n; c++) {
          const cut = numeric(a[k + 2 + c]);
          if (c > 0 && prev >= cut) throw new StyleError('Nonsequential interval cutoffs in collection/interval setting');
          prev = cut;
        }
        nb.ncollections = n;
        k += 2 + n;
        break;
      }
      case 'collection/type': {
        // "collection/type arg = Nc types1 types2 ... typesNc"; every atom type in exactly one collection
        if (!nb.multiStyle) throw new StyleError('Cannot use collection/type command without multi setting');
        need(2);
        const n = inumeric(a[k + 1]);
        if (n < 1) throw new StyleError(`Invalid collection/type keyword: illegal number of custom collections: ${n}`);
        if (k + 2 + n > a.length) throw new StyleError(`Invalid collection/type keyword: expected ${n} separate lists of types`);
        const ntypes = sys.state.ntypes;
        const seen = new Uint8Array(ntypes + 1);
        for (let c = 0; c < n; c++) {
          for (const word of a[k + 2 + c].split(',').filter((w) => w !== '')) {
            const [lo, hi] = bounds(word, 1, ntypes);
            for (let t = lo; t <= hi; t++) {
              if (seen[t]) throw new StyleError('Type specified more than once in collection/type command');
              seen[t] = 1;
            }
          }
        }
        for (let t = 1; t <= ntypes; t++) if (!seen[t]) throw new StyleError('Type missing in collection/type command');
        nb.ncollections = n;
        k += 2 + n;
        break;
      }
      case 'bin/hash': {
        // neigh_modify.html: "The *bin/hash* option toggles whether the atoms in a bin are stored in" a linked list
        // (default) or a hash table; "The hash option is only available for the" multi neighbor mode. The engine
        // keeps its own bin storage: the setting is checked and accepted. Native LAMMPS refuses yes without
        // multi (measured) and its result differs from the default only by roundoff (measured, 1.7e-15 relative).
        need(2);
        const on = logical(a[k + 1]);
        if (on && !nb.multiStyle) throw new StyleError('Cannot use bin/hash command without multi setting');
        k += 2;
        break;
      }
      default: throw new StyleError(`Unknown neigh_modify keyword: ${key}`);
    }
  }
  nb.lastBuild = -1;
  sys.bump();
};

export const FORCEFIELD_COMMANDS: Record<string, Handler> = {
  pair_style: pairStyle, pair_coeff: pairCoeff, pair_modify: pairModify,
  bond_style: bondedStyle('bond'), bond_coeff: bondedCoeff('bond'),
  angle_style: bondedStyle('angle'), angle_coeff: bondedCoeff('angle'),
  dihedral_style: bondedStyle('dihedral'), dihedral_coeff: bondedCoeff('dihedral'),
  improper_style: bondedStyle('improper'), improper_coeff: bondedCoeff('improper'),
  special_bonds: specialBonds, dielectric, kspace_style: kspaceStyle, kspace_modify: kspaceModify,
  neighbor, neigh_modify: neighModify,
};
