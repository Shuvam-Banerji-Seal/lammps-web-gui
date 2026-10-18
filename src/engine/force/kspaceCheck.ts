import { hasChargeStyle } from '../atoms';
import { PairHybrid } from './pair/hybrid';
import { StyleError, type KSpace, type Pair } from './types';
import type { SimState } from '../types';

/*
 * The long-range solver flags of LAMMPS styles, checked the way native LAMMPS checks them before a kspace style
 * is initialised.
 *
 * As native LAMMPS does (src/kspace.cpp KSpace::pair_check at b332f28): the kspace style is refused with
 * "KSpace style X is incompatible with Pair style Y" when a solver flag it sets (ewald, pppm, msm, dipole, spin)
 * is not set by the pair style, when the dispersion flags differ (either way), or when the TIP4P flags differ
 * (either way). The flags are the Pair and KSpace fields ewaldflag, pppmflag, msmflag, dispersionflag,
 * dipoleflag, spinflag and tip4pflag set in the native constructors (and settings) of each style, with the base
 * classes' flags inherited. The tables below were read from those sources at b332f28 (the constructor of each
 * class, then the classes it derives from); the names are the styles the engine registers, accelerator suffixes
 * left out.
 *
 * Flags that depend on the input are not in the tables: a hybrid pair style sets the union of its sub-styles'
 * flags (pair_hybrid.cpp settings); pair_style lepton/coul sets the flags of its keywords (ewald, pppm, msm;
 * the engine's Pair.solverKeywords); a dispersion term is set by the pair's own Pair.dispersionFlag
 * (lj/long/coul/long, buck/long/coul/long, lj/long/dipole/long); the ewald/disp kspace sets its dispersion and
 * dipole flags from the terms the pair gives (ewald_disp.cpp init: "if (!termflag[TERM_DIPOLE]) dipoleflag = 0").
 *
 * Not checked (native does not call pair_check): kspace_style zero (kspace_zero.cpp), and pair_style zero/coul
 * (it claims only the ewald, pppm, msm, dipole and spin flags, pair_zero_coul.cpp).
 */

type SolverFlag = 'ewald' | 'pppm' | 'msm' | 'dispersion' | 'dipole' | 'spin' | 'tip4p';
type Flags = Record<SolverFlag, boolean>;

/** Pair styles that set flags statically: the flags of the constructor and settings of the native class. */
const PAIR_STATIC_FLAGS: Readonly<Record<string, readonly SolverFlag[]>> = {
  'born/coul/long': ['ewald', 'pppm'],
  'born/coul/long/cs': ['ewald', 'pppm'],
  'born/coul/msm': ['msm'],
  'buck/coul/long': ['ewald', 'pppm'],
  'buck/coul/long/cs': ['ewald', 'pppm'],
  'buck/coul/msm': ['msm'],
  'buck/long/coul/long': ['ewald', 'pppm'],
  'buck6d/coul/gauss/long': ['ewald', 'pppm'],
  'coul/esp': ['ewald', 'pppm'],
  'coul/long': ['ewald', 'pppm'],
  'coul/long/cs': ['ewald', 'pppm'],
  'coul/long/dielectric': ['ewald', 'pppm'],
  'coul/long/soft': ['ewald', 'pppm'],
  'coul/msm': ['msm'],
  'coul/slater/long': ['ewald', 'pppm'],
  'coul/streitz': ['ewald', 'pppm'],
  'dpd/coul/slater/long': ['ewald', 'pppm'],
  'lj/charmm/coul/long': ['ewald', 'pppm'],
  'lj/charmm/coul/long/soft': ['ewald', 'pppm'],
  'lj/charmm/coul/msm': ['msm'],
  'lj/charmmfsw/coul/long': ['ewald', 'pppm'],
  'lj/class2/coul/long': ['ewald', 'pppm'],
  'lj/class2/coul/long/cs': ['ewald', 'pppm'],
  'lj/class2/coul/long/soft': ['ewald', 'pppm'],
  'lj/cut/coul/esp': ['ewald', 'pppm'],
  'lj/cut/coul/long': ['ewald', 'pppm'],
  'lj/cut/coul/long/cs': ['ewald', 'pppm'],
  'lj/cut/coul/long/dielectric': ['ewald', 'pppm'],
  'lj/cut/coul/long/soft': ['ewald', 'pppm'],
  'lj/cut/coul/msm': ['msm'],
  'lj/cut/coul/msm/dielectric': ['msm'],
  'lj/cut/dipole/long': ['dipole', 'ewald', 'pppm'],
  'lj/cut/thole/long': ['ewald', 'pppm'],
  'lj/cut/tip4p/long': ['ewald', 'pppm', 'tip4p'],
  'lj/cut/tip4p/long/soft': ['ewald', 'pppm', 'tip4p'],
  'lj/expand/coul/long': ['ewald', 'pppm'],
  'lj/long/coul/long': ['ewald', 'pppm'],
  'lj/long/coul/long/dielectric': ['ewald', 'pppm'],
  'lj/long/dipole/long': ['dipole', 'ewald'],
  'lj/long/tip4p/long': ['ewald', 'pppm', 'tip4p'],
  'lj/sdk/coul/long': ['ewald', 'pppm'],
  'lj/sdk/coul/msm': ['msm'],
  'lj/spica/coul/long': ['ewald', 'pppm'],
  'lj/spica/coul/msm': ['msm'],
  'lj/switch3/coulgauss/long': ['ewald', 'pppm'],
  'mm3/switch3/coulgauss/long': ['ewald', 'pppm'],
  'nm/cut/coul/long': ['ewald', 'pppm'],
  'spin/dipole/cut': ['spin'],
  'spin/dipole/long': ['ewald', 'pppm', 'spin'],
  'tip4p/long': ['ewald', 'pppm', 'tip4p'],
  'tip4p/long/soft': ['ewald', 'pppm', 'tip4p'],
  'zero/coul': ['dipole', 'ewald', 'msm', 'pppm', 'spin'],
};

/**
 * KSpace styles that call KSpace::pair_check in their init (ewald.cpp, pppm.cpp, msm.cpp, ewald_disp.cpp, pppm_disp.cpp,
 * ewald_dipole.cpp, pppm_dipole.cpp, the *_spin variants, the electrode styles, and the classes that inherit init:
 * pppm/tip4p (PPPMTIP4P : PPPM), pppm/disp/tip4p (PPPMDispTIP4P : PPPMDisp), the dielectric, cg and stagger styles, and
 * esp (esp.cpp init).
 * Each set lists the flags of the kspace constructor; ewald/disp takes its dispersion and dipole flags from the pair
 * (KSpace.dispersion and KSpace.dipole).
 */
const KSPACE_STATIC_FLAGS: Readonly<Record<string, readonly SolverFlag[]>> = {
  // esp.cpp: espflag only (pair_check is not asked about it), so only the dispersion and TIP4P flags must agree
  'esp': [],
  'ewald': ['ewald'],
  'ewald/disp': ['ewald'],
  'ewald/disp/dipole': ['ewald'],
  'ewald/dipole': ['ewald', 'dipole'],
  'ewald/dipole/spin': ['ewald', 'spin'],
  'ewald/electrode': ['ewald'],
  'msm': ['msm'],
  'msm/cg': ['msm'],
  'msm/dielectric': ['msm'],
  'pppm': ['pppm'],
  'pppm/cg': ['pppm'],
  'pppm/dielectric': ['pppm'],
  'pppm/dipole': ['pppm', 'dipole'],
  'pppm/dipole/spin': ['pppm', 'spin'],
  'pppm/disp': ['pppm', 'dispersion'],
  'pppm/disp/dielectric': ['pppm', 'dispersion'],
  'pppm/disp/tip4p': ['pppm', 'dispersion', 'tip4p'],
  'pppm/electrode': ['pppm'],
  'pppm/stagger': ['pppm'],
  'pppm/tip4p': ['pppm', 'tip4p'],
};

const emptyFlags = (): Flags => ({ ewald: false, pppm: false, msm: false, dispersion: false, dipole: false, spin: false, tip4p: false });

const orInto = (into: Flags, from: Flags): void => {
  for (const f of Object.keys(into) as SolverFlag[]) into[f] = into[f] || from[f];
};

/** The solver flags a pair style has (native Pair::ewaldflag and the other flags). */
export const pairSolverFlags = (pair: Pair): Flags => {
  const out = emptyFlags();
  if (pair instanceof PairHybrid) {
    // pair_hybrid.cpp: the hybrid style ORs the flags of its sub-styles
    for (const sub of pair.subs) orInto(out, pairSolverFlags(sub.style));
    return out;
  }
  for (const f of PAIR_STATIC_FLAGS[pair.name] ?? []) out[f] = true;
  for (const kw of pair.solverKeywords) {
    if (kw === 'ewald' || kw === 'pppm' || kw === 'msm') out[kw] = true;
  }
  // a cut or long dispersion term: the pair's own flag (lj/long/coul/long, buck/long/coul/long, lj/long/dipole/long)
  if (pair.dispersionFlag) out.dispersion = true;
  return out;
};

/** The solver flags a kspace style has (native KSpace fields), with the pair-dependent ones read from the solver. */
const kspaceSolverFlags = (kspace: KSpace, names: readonly SolverFlag[]): Flags => {
  const out = emptyFlags();
  for (const f of names) out[f] = true;
  if (kspace.dispersion()) out.dispersion = true;
  if (kspace.dipole()) out.dipole = true;
  return out;
};

/**
 * KSpace::pair_check (src/kspace.cpp at b332f28): throws the native refusal when the kspace style and the pair style
 * disagree. A kspace style that native does not check (kspace_style zero, or any name not in the table) passes.
 */
export const checkKspacePair = (kspace: KSpace, pair: Pair, atomStyle: SimState['atomStyle']): void => {
  const names = KSPACE_STATIC_FLAGS[kspace.name];
  if (!names) return;
  // Ewald::init and PPPM::init read atom attribute q before they call pair_check (ewald.cpp:106, pppm.cpp:226 at
  // b332f28: "Kspace style requires atom attribute q"); the dipole and spin solvers are not checked here
  // ewald/disp has no charge term of its own when the pair gives none (ewald_disp.cpp), so it is not checked here
  const chargeSolver = names.some((f) => f === 'ewald' || f === 'pppm') && !names.some((f) => f === 'dipole' || f === 'spin' || f === 'dispersion')
    && !kspace.name.startsWith('ewald/disp');
  if (chargeSolver && !hasChargeStyle(atomStyle)) throw new StyleError('Kspace style requires atom attribute q');
  const k = kspaceSolverFlags(kspace, names);
  const p = pairSolverFlags(pair);
  const compatible =
    (!k.ewald || p.ewald) &&
    (!k.pppm || p.pppm) &&
    (!k.msm || p.msm) &&
    (!k.dipole || p.dipole) &&
    (!k.spin || p.spin) &&
    k.dispersion === p.dispersion &&
    k.tip4p === p.tip4p;
  if (!compatible) throw new StyleError(`KSpace style ${kspace.name} is incompatible with Pair style ${pair.name}`);
};
