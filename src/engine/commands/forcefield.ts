import type { Handler } from './args';
import { int, num, yesno } from './args';
import { StyleError, type Bonded } from '../force/types';
import { PAIR_STYLES, BOND_STYLES, ANGLE_STYLES, DIHEDRAL_STYLES, IMPROPER_STYLES, KSPACE_STYLES } from '../styles';
import type { System } from '../system';
import type { MixRule, Pair } from '../force/types';
import { PairHybrid } from '../force/pair/hybrid';
import { isMolecularStyle } from '../atoms';

/*
 * Force-field commands. Styles come from styles.ts; anything else is an
 * error that names the style and lists the supported ones.
 */

const unsupported = (kind: string, name: string, list: Record<string, unknown>) =>
  new StyleError(`${kind} '${name}' is not supported by the browser engine; supported: ${Object.keys(list).sort().join(', ') || 'none yet'}`);

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
  p.coeff(a, sys.styleContext());
  sys.bump();
};

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
  let targets: Pair[] = [p];
  let k = 0;
  if (p instanceof PairHybrid) {
    targets = [p, ...p.subs.map((x) => x.style)];
    if (a[0] === 'pair') {
      const { sub, used } = p.selectSub(a.slice(1), 'pair_modify pair');
      // the selected sub-style first (style-specific keywords go to it); the hybrid's own
      // shift / tail flags follow, as native LAMMPS rejects shift on one sub-style with tail on another
      targets = [sub.style, p];
      k = 1 + used;
      for (;;) {
        if (a[k] === 'special') {
          const w = [num(a[k + 2], 'special'), num(a[k + 3], 'special'), num(a[k + 4], 'special')] as [number, number, number];
          p.setSpecial(sub, a[k + 1] ?? '', w);
          k += 5;
        } else if (a[k] === 'compute/tally') {
          yesno(a[k + 1], 'compute/tally');
          k += 2;
        } else break;
      }
    }
  }
  for (; k < a.length;) {
    const key = a[k];
    const v = a[k + 1];
    switch (key) {
      case 'mix':
        if (!['geometric', 'arithmetic', 'sixthpower'].includes(v ?? '')) throw new StyleError('pair_modify mix must be geometric, arithmetic or sixthpower');
        for (const t of targets) t.mix = v as MixRule;
        k += 2;
        break;
      case 'shift': { const y = yesno(v, 'shift'); for (const t of targets) t.shift = y; k += 2; break; }
      case 'tail': { const y = yesno(v, 'tail'); for (const t of targets) t.tail = y; k += 2; break; }
      case 'table': { const n = int(v, 'table'); for (const t of targets) t.table = n; k += 2; break; }
      case 'table/disp': { const n = int(v, 'table/disp'); for (const t of targets) (t as Pair & { tableDisp?: number }).tableDisp = n; k += 2; break; }
      case 'tabinner': case 'tabinner/disp': {
        // pair_modify.html: "The default cutoff value is sqrt(2.0) distance units"; the engine's tables
        // (force/erfc.ts) are built for that inner cutoff only
        const c = num(v, key);
        if (Math.abs(c - Math.SQRT2) > 1e-12) throw new StyleError(`pair_modify ${key} ${v}: only the default sqrt(2.0) is supported by the browser engine`);
        k += 2;
        break;
      }
      case 'compute': if (!yesno(v, 'compute')) throw new StyleError('pair_modify compute no is not supported'); k += 2; break;
      case 'neigh/trim': yesno(v, key); k += 2; break;
      case 'pair':
        throw new StyleError(p instanceof PairHybrid
          ? 'pair_modify pair must appear first in the list of keywords'
          : 'pair_modify pair can only be used with the hybrid and hybrid/overlay pair styles');
      case 'special': case 'compute/tally':
        throw new StyleError(`pair_modify ${key} can only be used directly after the pair keyword (hybrid pair styles)`);
      default: k += 1 + targets[0].modify(key, a.slice(k + 1));
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
  const same = sys.ff[kind] && sys.ff[kind]!.name === name ? sys.ff[kind] : null;
  const st: Bonded = same ?? make();
  st.settings(a.slice(1), sys.styleContext());
  if (!same && sys.hasBox) st.allocate(bondedKinds[kind].types(sys));
  sys.ff[kind] = st;
  sys.bump();
}) as Handler;

const bondedCoeff = (kind: keyof typeof bondedKinds): Handler => (({ sys }, a) => {
  const st = sys.ff[kind];
  if (!st) throw new StyleError(`${kind}_coeff needs a ${kind}_style first`);
  const n = bondedKinds[kind].types(sys);
  if (n === 0) throw new StyleError(`${kind}_coeff: there are no ${kind} types (create_box ... ${kind}/types N, or the data file header)`);
  if (st.ntypes !== n) st.allocate(n);
  st.coeff(a, sys.styleContext());
  sys.bump();
}) as Handler;

/** special_bonds keyword values — special_bonds.html ("Each time you use this command it sets all the coefficients to default values and only overrides the one you specify"). */
const specialBonds: Handler = ({ sys }, a) => {
  const sp = { lj: [0, 0, 0] as [number, number, number], coul: [0, 0, 0] as [number, number, number], angle: false, dihedral: false };
  const w3 = (k: number, what: string): [number, number, number] => {
    const v = [num(a[k + 1], what), num(a[k + 2], what), num(a[k + 3], what)] as [number, number, number];
    if (v.some((x) => x < 0 || x > 1)) throw new StyleError('special_bonds weights must be between 0.0 and 1.0');
    return v;
  };
  for (let k = 0; k < a.length;) {
    switch (a[k]) {
      case 'amber': sp.lj = [0, 0, 0.5]; sp.coul = [0, 0, 0.8333]; k++; break;
      case 'charmm': sp.lj = [0, 0, 0]; sp.coul = [0, 0, 0]; k++; break;
      case 'dreiding': sp.lj = [0, 0, 1]; sp.coul = [0, 0, 1]; k++; break;
      case 'fene': sp.lj = [0, 1, 1]; sp.coul = [0, 1, 1]; k++; break;
      case 'lj/coul': sp.lj = w3(k, 'lj/coul'); sp.coul = [...sp.lj]; k += 4; break;
      case 'lj': sp.lj = w3(k, 'lj'); k += 4; break;
      case 'coul': sp.coul = w3(k, 'coul'); k += 4; break;
      case 'angle': sp.angle = yesno(a[k + 1], 'angle'); k += 2; break;
      case 'dihedral': sp.dihedral = yesno(a[k + 1], 'dihedral'); k += 2; break;
      case 'one/five':
        if (yesno(a[k + 1], 'one/five')) throw new StyleError('special_bonds one/five yes is not supported');
        k += 2;
        break;
      default: throw new StyleError(`unknown special_bonds keyword '${a[k]}'`);
    }
  }
  sys.ff.special = sp;
  if (sys.hasBox) sys.ff.topologyChanged(sys.state);
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

/** kspace_modify keyword value ... — kspace_modify.html. */
const kspaceModify: Handler = ({ sys }, a) => {
  const k = sys.ff.kspace;
  if (!k) throw new StyleError('kspace_modify needs a kspace_style first');
  for (let i = 0; i < a.length;) {
    const key = a[i];
    if (key === 'compute') {
      if (!yesno(a[i + 1], 'compute')) throw new StyleError('kspace_modify compute no is not supported');
      i += 2;
      continue;
    }
    const used = k.modify(key, a.slice(i + 1));
    if (used <= 0) throw new StyleError(`kspace_modify keyword '${key}' is not supported by kspace_style ${k.name}`);
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
  sys.nb.lastBuild = -1;
  sys.bump();
};

/** neigh_modify keyword values — neigh_modify.html. */
const neighModify: Handler = ({ sys }, a) => {
  const nb = sys.nb;
  for (let k = 0; k < a.length;) {
    const key = a[k];
    switch (key) {
      case 'delay': nb.delay = int(a[k + 1], 'delay'); k += 2; break;
      case 'every': nb.every = int(a[k + 1], 'every'); if (nb.every < 1) throw new StyleError('neigh_modify every must be >= 1'); k += 2; break;
      case 'check': nb.check = yesno(a[k + 1], 'check'); k += 2; break;
      case 'once': nb.once = yesno(a[k + 1], 'once'); k += 2; break;
      case 'cluster': yesno(a[k + 1], 'cluster'); k += 2; break;
      case 'page': case 'one': int(a[k + 1], key); k += 2; break;
      case 'binsize': nb.binsize = num(a[k + 1], 'binsize'); k += 2; break;
      case 'include': nb.includeBit = a[k + 1] === 'all' ? 0 : sys.groupBit(a[k + 1] ?? ''); k += 2; break;
      case 'exclude': {
        const kind = a[k + 1];
        if (kind === 'none') { nb.excludes = []; k += 2; break; }
        if (kind === 'type') { nb.excludes.push({ kind, a: int(a[k + 2], 'type'), b: int(a[k + 3], 'type') }); k += 4; break; }
        if (kind === 'group') { nb.excludes.push({ kind, a: sys.groupBit(a[k + 2] ?? ''), b: sys.groupBit(a[k + 3] ?? '') }); k += 4; break; }
        if (kind === 'molecule/intra' || kind === 'molecule/inter') { nb.excludes.push({ kind, a: sys.groupBit(a[k + 2] ?? '') }); k += 3; break; }
        throw new StyleError(`neigh_modify exclude: unknown style '${kind ?? ''}'`);
      }
      case 'collection/type': case 'collection/interval': {
        const n = int(a[k + 1], key);
        k += 2 + n;
        break;
      }
      default: throw new StyleError(`unknown neigh_modify keyword '${key}'`);
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
