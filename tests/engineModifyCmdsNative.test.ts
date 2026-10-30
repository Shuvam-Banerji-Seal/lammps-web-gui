import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * neigh_modify, comm_modify, pair_modify, kspace_modify (pppm and ewald), special_bonds and atom_modify as native
 * LAMMPS reads them (src/neighbor.cpp Neighbor::modify_params and init, src/comm.cpp Comm::modify_params, src/pair.cpp
 * Pair::modify_params, src/kspace.cpp KSpace::modify_params, src/KSPACE/pppm.cpp and ewald.cpp init, src/force.cpp
 * Force::set_special, src/atom.cpp Atom::modify_params, at b332f28). Measured with native LAMMPS (/home/roy/.local/bin/lmp,
 * b332f28): each row is the input below with its lines added, null when it runs, else the start of native's error
 * message. A few inputs native runs are refused by the browser engine, each with an error naming what it lacks.
 */

type Row = { pre?: string; post?: string; ks?: string; bnd?: string };

const input = (r: Row): string => `units lj
atom_style full
${r.pre ? `${r.pre}\n` : ''}${r.bnd ? `${r.bnd}\n` : ''}lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 2 box bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 8
create_atoms 1 box
mass * 1.0
set type 1 type/ratio 2 0.5 12345
set type 1 charge 1.0
set type 2 charge -1.0
group g1 type 1
group g2 type 2
pair_style lj/cut/coul/long 2.5
pair_coeff * * 1.0 1.0
${r.ks ?? 'kspace_style pppm 1e-3'}
bond_style harmonic
bond_coeff 1 1.0 1.0
fix 1 all nve
${r.post ?? ''}
run 0
`;

const label = (r: Row): string => Object.entries(r).map(([k, v]) => `${k}: ${v.replace(/\n/g, ' / ')}`).join(' | ');

const errorOf = async (r: Row): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  try { await s.execute(input(r)); } catch (e) { return (e as Error).message; }
  return null;
};

const ROWS: [Row, string | null][] = [
  [{ post: 'neigh_modify every 2 delay 0 check yes' }, null],
  [{ post: 'neigh_modify every 0' }, 'Invalid neigh_modify every argument: 0'],
  [{ post: 'neigh_modify delay -1' }, 'Invalid neigh_modify delay argument: -1'],
  [{ post: 'neigh_modify check maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'neigh_modify once yes' }, null],
  [{ post: 'neigh_modify cluster yes' }, null],
  [{ post: 'neigh_modify include all' }, 'Invalid include keyword: atom_modify first command must be used'],
  [{ post: 'neigh_modify exclude type 1 2' }, null],
  [{ post: 'neigh_modify exclude type 1 3' }, 'Invalid atom type in neighbor exclusion list'],
  [{ post: 'neigh_modify exclude none' }, null],
  [{ post: 'neigh_modify page 100000 one 2000' }, null],
  [{ post: 'neigh_modify page 1000 one 2000' }, 'Neighbor page size must be >= 10x the one atom setting'],
  [{ post: 'neigh_modify binsize 0.5' }, null],
  [{ post: 'neigh_modify binsize -1' }, null],
  [{ post: 'neigh_modify bogus 1' }, 'Unknown neigh_modify keyword: bogus'],
  [{ post: 'neigh_modify collection/type 2 1 2' }, 'Cannot use collection/type command without multi setting'],
  [{ post: 'neigh_modify every' }, 'Illegal neigh_modify every command: missing argument(s)'],
  [{ post: 'comm_modify cutoff 3.0' }, null],
  [{ post: 'comm_modify cutoff -1' }, 'Invalid cutoff -1 in comm_modify command'],
  [{ post: 'comm_modify vel yes' }, null],
  [{ post: 'comm_modify vel maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'comm_modify mode multi' }, 'Cannot use comm mode \'multi\' without \'multi\' style neighbor lists'],
  [{ post: 'comm_modify mode bogus' }, 'Unknown comm_modify mode argument: bogus'],
  [{ post: 'comm_modify group all' }, null],
  [{ post: 'comm_modify cutoff/multi 1 3.0' }, 'Use cutoff keyword to set cutoff in single mode'],
  [{ post: 'comm_modify bogus 1' }, 'Unknown comm_modify keyword: bogus'],
  [{ post: 'comm_modify reduce/multi' }, 'Use reduce/multi in mode multi only'],
  [{ post: 'pair_modify shift yes' }, null],
  [{ post: 'pair_modify tail yes' }, null],
  [{ post: 'pair_modify mix arithmetic' }, null],
  [{ post: 'pair_modify mix bogus' }, 'Unknown pair_modify mix argument: bogus'],
  [{ post: 'pair_modify table 0' }, null],
  [{ post: 'pair_modify table/disp 0' }, null],
  [{ post: 'pair_modify tabinner 1.5' }, null],
  [{ post: 'pair_modify compute no' }, null],
  [{ post: 'pair_modify nofdotr' }, null],
  [{ post: 'pair_modify bogus 1' }, 'Unknown pair_modify keyword: bogus'],
  [{ post: 'pair_modify shift maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'pair_modify pair lj/cut/coul/long shift yes' }, 'Unknown pair_modify keyword: pair'],
  [{ post: 'pair_modify special lj 0 0 0' }, 'Unknown pair_modify keyword: special'],
  [{ post: 'kspace_modify mesh 8 8 8' }, null],
  [{ post: 'kspace_modify order 3' }, null],
  [{ post: 'kspace_modify order 1' }, 'PPPM order cannot be < 2 or > 7'],
  [{ post: 'kspace_modify gewald 0.5' }, null],
  [{ post: 'kspace_modify slab 3.0' }, 'Incorrect boundaries with slab PPPM'],
  [{ post: 'kspace_modify slab nozforce' }, 'Incorrect boundaries with slab PPPM'],
  [{ post: 'kspace_modify diff ad' }, null],
  [{ post: 'kspace_modify diff bogus' }, 'Unknown kspace_modify diff flag bogus'],
  [{ post: 'kspace_modify compute no' }, null],
  [{ post: 'kspace_modify bogus 1' }, 'Illegal kspace_modify command for kspace style pppm'],
  [{ post: 'kspace_modify minorder 2' }, null],
  [{ post: 'kspace_modify overlap no' }, null],
  [{ post: 'kspace_modify fftbench yes' }, null],
  [{ post: 'kspace_modify collective yes' }, null],
  [{ post: 'kspace_modify pressure/scalar yes' }, null],
  [{ post: 'kspace_modify kmax/ewald 4 4 4' }, null],
  [{ post: 'kspace_modify mix/disp pair' }, null],
  [{ post: 'kspace_modify force/disp/real 1e-3' }, null],
  [{ post: 'kspace_modify scafacos tolerance field' }, 'Illegal kspace_modify command for kspace style pppm'],
  [{ post: 'kspace_modify mesh/disp 8 8 8' }, null],
  [{ post: 'special_bonds lj 0 0 0.5' }, null],
  [{ post: 'special_bonds lj/coul 0 0.5 1' }, null],
  [{ post: 'special_bonds coul 0 0 0' }, null],
  [{ post: 'special_bonds amber' }, null],
  [{ post: 'special_bonds charmm' }, null],
  [{ post: 'special_bonds dreiding' }, null],
  [{ post: 'special_bonds fene' }, null],
  [{ post: 'special_bonds angle yes' }, null],
  [{ post: 'special_bonds dihedral yes' }, null],
  [{ post: 'special_bonds one/five yes' }, 'Cannot set special_bonds one/five if atom style does not support it'],
  [{ post: 'special_bonds lj 0 0' }, 'Illegal special_bonds command'],
  [{ post: 'special_bonds lj 0 0 2' }, 'Illegal special_bonds command'],
  [{ post: 'special_bonds bogus' }, 'Illegal special_bonds command'],
  [{ post: 'special_bonds lj 0 0 0 coul 1 1 1' }, null],
  [{ pre: 'atom_modify id no' }, 'Cannot create an atom map unless atoms have IDs'],
  [{ pre: 'atom_modify map array' }, null],
  [{ pre: 'atom_modify map hash' }, null],
  [{ pre: 'atom_modify map yes' }, null],
  [{ pre: 'atom_modify map bogus' }, 'Illegal atom_modify map command argument bogus'],
  [{ pre: 'atom_modify first all' }, null],
  [{ pre: 'atom_modify sort 0 0' }, null],
  [{ pre: 'atom_modify sort 100 1.0' }, null],
  [{ pre: 'atom_modify sort -1 1' }, 'Illegal atom_modify sort frequency -1'],
  [{ pre: 'atom_modify bogus 1' }, 'Illegal atom_modify command argument: bogus'],
  [{ pre: 'atom_modify id maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ pre: 'atom_modify first g1 sort 0 0' }, 'Atom_modify sort and first options cannot be used together'],
  [{ pre: 'atom_modify sort 0 0 first g1' }, null],
  [{ pre: 'atom_modify sort 10 -1' }, 'Illegal atom_modify sort bin size -1'],
  [{ pre: 'atom_modify sort 10' }, 'Illegal atom_modify sort command: missing argument(s)'],
  [{ pre: 'atom_modify map' }, 'Illegal atom_modify map command: missing argument(s)'],
  [{ pre: 'atom_modify id' }, 'Illegal atom_modify id command: missing argument(s)'],
  [{ pre: 'atom_modify first' }, 'Illegal atom_modify first command: missing argument(s)'],
  [{ pre: 'atom_modify id on' }, null],
  [{ pre: 'atom_modify id 1' }, null],
  [{ pre: 'atom_modify sort 1.5 0' }, 'Expected integer parameter instead of \'1.5\' in input script or data file'],
  [{ pre: 'atom_modify sort 10 abc' }, 'Expected floating point parameter instead of \'abc\' in input script or data file'],
  [{ post: 'atom_modify map array' }, 'Atom_modify map command after simulation box is defined'],
  [{ post: 'atom_modify id yes' }, 'Atom_modify id command after simulation box is defined'],
  [{ post: 'atom_modify sort 0 0' }, null],
  [{ post: 'atom_modify first g1' }, null],
  [{ post: 'atom_modify' }, 'Illegal atom_modify command: missing argument(s)'],
  [{ post: 'neigh_modify delay 3 every 2' }, 'Neighbor delay must be 0 or multiple of every setting'],
  [{ post: 'neigh_modify delay 4 every 2' }, null],
  [{ post: 'neigh_modify include g1' }, 'Invalid include keyword: atom_modify first command must be used'],
  [{ post: 'neigh_modify exclude group g1 g2' }, null],
  [{ post: 'neigh_modify exclude group g1 nogroup' }, 'Invalid exclude group keyword: group nogroup not found'],
  [{ post: 'neigh_modify exclude molecule/intra all' }, null],
  [{ post: 'neigh_modify exclude molecule/inter nogroup' }, 'Invalid exclude keyword:group nogroup not found'],
  [{ post: 'neigh_modify exclude bogus' }, 'Unknown neigh_modify exclude keyword: bogus'],
  [{ post: 'neigh_modify exclude type 1' }, 'Illegal neigh_modify exclude type command: missing argument(s)'],
  [{ post: 'neigh_modify exclude' }, 'Illegal neigh_modify exclude command: missing argument(s)'],
  [{ post: 'neigh_modify every 2.5' }, 'Expected integer parameter instead of \'2.5\' in input script or data file'],
  [{ post: 'neigh_modify check on' }, null],
  [{ post: 'neigh_modify check 1' }, null],
  [{ post: 'neigh_modify binsize 0' }, null],
  [{ post: 'neigh_modify one 20000' }, 'Neighbor page size must be >= 10x the one atom setting'],
  [{ post: 'neigh_modify page 5000 one 500' }, null],
  [{ post: 'neigh_modify bin/hash no' }, null],
  [{ post: 'neigh_modify bin/hash yes' }, 'Cannot use bin/hash command without multi setting'],
  [{ post: 'neigh_modify collection/interval 2 1.0 2.0' }, 'Cannot use collection/interval command without multi setting'],
  [{ post: 'neigh_modify delay 1e3' }, 'Expected integer parameter instead of \'1e3\' in input script or data file'],
  [{ pre: 'atom_modify first g1', post: 'neigh_modify include g2' }, 'Neigh_modify include group != atom_modify first group: g1'],
  [{ pre: 'atom_modify first g1', post: 'neigh_modify include g1' }, null],
  [{ post: 'neigh_modify exclude type 1 2\nneigh_modify exclude none\nneigh_modify exclude type 1 4' }, 'Invalid atom type in neighbor exclusion list'],
  [{ post: 'neighbor 0.3 multi\nneigh_modify collection/type 2 1 2' }, null],
  [{ post: 'neighbor 0.3 multi\nneigh_modify collection/type 2 1 1' }, 'Type specified more than once in collection/type command'],
  [{ post: 'neighbor 0.3 multi\nneigh_modify collection/type 1 1' }, 'Type missing in collection/type command'],
  [{ post: 'neighbor 0.3 multi\nneigh_modify collection/type 2 1*2 2' }, 'Type specified more than once in collection/type command'],
  [{ post: 'neighbor 0.3 multi\nneigh_modify collection/interval 2 2.0 1.0' }, 'Nonsequential interval cutoffs in collection/interval setting'],
  [{ post: 'neighbor 0.3 multi\nneigh_modify collection/interval 0' }, 'Invalid collection/interval keyword: illegal number of custom collections: 0'],
  [{ post: 'neighbor 0.3 multi\nneigh_modify collection/type 3 1 2' }, 'Invalid collection/type keyword: expected 3 separate lists of types'],
  [{ post: 'neighbor 0.3 multi\nneigh_modify bin/hash yes' }, null],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi' }, null],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi cutoff 3.0' }, 'Use cutoff/multi keyword to set cutoff in multi mode'],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi cutoff/multi 1 3.0' }, null],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi cutoff/multi * 3.0' }, null],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi cutoff/multi 3 3.0' }, 'Numeric index 3 is out of bounds (1-2)'],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi cutoff/multi 1 -1' }, 'Invalid cutoff -1 in comm_modify command'],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi reduce/multi' }, null],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi mode single cutoff 3.0' }, null],
  [{ post: 'neighbor 0.3 multi\ncomm_modify mode multi/old' }, 'Communication mode \'multi/old\' has been removed. Please use mode \'multi\' and see the documentation more information about it.'],
  [{ post: 'neighbor 0.3 multi\ncomm_modify group all' }, null],
  [{ post: 'neighbor 0.3 multi\ncomm_modify group g1' }, 'string pointer is null'],
  [{ post: 'neighbor 0.3 multi\ncomm_modify group nogroup' }, 'Invalid comm_modify keyword: group nogroup not found'],
  [{ post: 'neighbor 0.3 multi\ncomm_modify vel on' }, null],
  [{ post: 'neighbor 0.3 multi\ncomm_modify cutoff' }, 'Illegal comm_modify cutoff command: missing argument(s)'],
  [{ post: 'neighbor 0.3 multi\ncomm_modify cutoff abc' }, 'Expected floating point parameter instead of \'abc\' in input script or data file'],
  [{ pre: 'atom_modify first g1', post: 'comm_modify group g2' }, 'Comm_modify group != atom_modify first group: g1'],
  [{ post: 'comm_modify' }, 'Illegal comm_modify command: missing argument(s)'],
  [{ post: 'pair_modify table 12' }, null],
  [{ post: 'pair_modify table 4' }, 'Too few total bits for bitmapped Coulomb lookup table'],
  [{ post: 'pair_modify table 40' }, 'Too many total bits for bitmapped Coulomb lookup table'],
  [{ post: 'pair_modify table/disp 4' }, 'Too few total bits for bitmapped Dispersion lookup table'],
  [{ post: 'pair_modify table abc' }, 'Expected integer parameter instead of \'abc\' in input script or data file'],
  [{ post: 'pair_modify tail on' }, null],
  [{ post: 'pair_modify compute off' }, null],
  [{ post: 'pair_modify neigh/trim yes' }, null],
  [{ post: 'pair_modify mix' }, 'Illegal pair_modify mix command: missing argument(s)'],
  [{ post: 'pair_modify nofdotr shift yes' }, null],
  [{ post: 'pair_modify shift yes tail yes' }, 'Cannot have both pair_modify shift and tail set to yes'],
  [{ post: 'pair_modify tabinner abc' }, 'Expected floating point parameter instead of \'abc\' in input script or data file'],
  [{ post: 'pair_modify' }, 'Illegal pair_modify command: missing argument(s)'],
  [{ post: 'special_bonds angle maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'special_bonds lj/coul 0 0' }, 'Illegal special_bonds command'],
  [{ post: 'special_bonds coul 0 0 1.5' }, 'Illegal special_bonds command'],
  [{ post: 'special_bonds lj 0 0 0.5 angle' }, 'Illegal special_bonds command'],
  [{ post: 'special_bonds one/five maybe' }, 'Illegal special_bonds command'],
  [{ post: 'special_bonds lj -0.1 0 0' }, 'Illegal special_bonds command'],
  [{ post: 'special_bonds lj 0 0 abc' }, 'Expected floating point parameter instead of \'abc\' in input script or data file'],
  [{ post: 'special_bonds angle on dihedral 1' }, null],
  [{ post: 'special_bonds' }, 'Illegal special_bonds command'],
  [{ post: 'kspace_modify mesh 8 8 0' }, 'Kspace_modify mesh parameters must be all zero or all positive'],
  [{ post: 'kspace_modify mesh 0 0 0' }, null],
  [{ post: 'kspace_modify mesh -8 8 8' }, 'Kspace_modify mesh parameters must be all zero or all positive'],
  [{ post: 'kspace_modify mesh/disp 8 8 -1' }, null],
  [{ post: 'kspace_modify mesh/disp -8 8 8' }, 'Kspace_modify mesh/disp parameters must be all zero or all positive'],
  [{ post: 'kspace_modify kmax/ewald 4 -1 4' }, 'Bad kspace_modify kmax/ewald parameter'],
  [{ post: 'kspace_modify mix/disp bad' }, 'Unknown kspace_modify mix/disp flag bad'],
  [{ post: 'kspace_modify splittol 1.5' }, 'Kspace_modify splittol value must be smaller than one'],
  [{ post: 'kspace_modify splittol 0.5' }, null],
  [{ post: 'kspace_modify eigtol 2' }, 'Kspace_modify splittol value must be smaller than one'],
  [{ post: 'kspace_modify amat onestep' }, null],
  [{ post: 'kspace_modify amat bad' }, 'Unknown kspace_modify amat flag bad'],
  [{ post: 'kspace_modify order/disp 5' }, null],
  [{ post: 'kspace_modify gewald/disp 0.5' }, null],
  [{ post: 'kspace_modify force/disp/kspace 1e-3' }, null],
  [{ post: 'kspace_modify disp/auto yes' }, null],
  [{ post: 'kspace_modify order 8' }, 'PPPM order cannot be < 2 or > 7'],
  [{ post: 'kspace_modify order abc' }, 'Expected integer parameter instead of \'abc\' in input script or data file'],
  [{ post: 'kspace_modify slab auto' }, 'Incorrect boundaries with slab PPPM'],
  [{ post: 'kspace_modify slab 1.0' }, 'Bad kspace_modify slab parameter'],
  [{ post: 'kspace_modify slab 0.5' }, 'Bad kspace_modify slab parameter'],
  [{ post: 'kspace_modify slab ew2d' }, 'Incorrect boundaries with slab PPPM'],
  [{ post: 'kspace_modify gewald' }, 'Illegal kspace_modify gewald command: missing argument(s)'],
  [{ post: 'kspace_modify mesh 8 8' }, 'Illegal kspace_modify mesh command: missing argument(s)'],
  [{ post: 'kspace_modify wire 3.0' }, null],
  [{ post: 'kspace_modify wire 1.0' }, 'Bad kspace_modify wire parameter'],
  [{ post: 'kspace_modify wire noxyforce' }, null],
  [{ post: 'kspace_modify compute maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'kspace_modify diff ik' }, null],
  [{ post: 'kspace_modify order 2' }, null],
  [{ post: 'kspace_modify minorder 1' }, 'Kspace_modify minorder must be at least 2'],
  [{ post: 'kspace_modify collective/self/copy bogus' }, 'Unknown kspace_modify collective/self/copy flag bogus'],
  [{ post: 'kspace_modify order 8 slab 3.0' }, 'Incorrect boundaries with slab PPPM'],
  [{ post: 'set type 2 charge -0.5\nkspace_modify wire 3.0' }, 'System is not charge neutral, net charge = 8'],
  [{ post: 'set type 2 charge -0.5' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify mesh 8 8 8' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify order 3' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify diff ad' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify diff ik' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify kmax/ewald 4 0 4' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify kmax/ewald 4 -1 4' }, 'Bad kspace_modify kmax/ewald parameter'],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify amat onestep' }, 'Illegal kspace_modify command \'amat\' available for pppm/conp, only'],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify mesh/disp 8 8 8' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify slab 3.0' }, 'Incorrect boundaries with slab Ewald'],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify order 9' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify mesh 8 8 -1' }, 'Kspace_modify mesh parameters must be all zero or all positive'],
  [{ ks: 'kspace_style ewald 1e-3', post: 'kspace_modify wire 3.0' }, null],
  [{ ks: 'kspace_style ewald 1e-3', post: 'set type 2 charge -0.5\nkspace_modify wire noxyforce' }, 'System is not charge neutral, net charge = 8'],
  [{ ks: 'kspace_style pppm 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p f' }, null],
  [{ ks: 'kspace_style ewald 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p f' }, null],
  [{ ks: 'kspace_style pppm 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p s' }, 'Incorrect boundaries with slab PPPM'],
  [{ ks: 'kspace_style ewald 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p s' }, 'Incorrect boundaries with slab Ewald'],
  [{ ks: 'kspace_style pppm 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p m' }, 'Incorrect boundaries with slab PPPM'],
  [{ ks: 'kspace_style ewald 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p m' }, 'Incorrect boundaries with slab Ewald'],
  [{ ks: 'kspace_style pppm 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p fs' }, 'Incorrect boundaries with slab PPPM'],
  [{ ks: 'kspace_style ewald 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p fs' }, 'Incorrect boundaries with slab Ewald'],
  [{ ks: 'kspace_style pppm 1e-3\nkspace_modify slab 3.0', bnd: 'boundary f p f' }, 'Incorrect boundaries with slab PPPM'],
  [{ ks: 'kspace_style ewald 1e-3\nkspace_modify slab 3.0', bnd: 'boundary f p f' }, 'Incorrect boundaries with slab Ewald'],
  [{ ks: 'kspace_style pppm 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p fm' }, 'Incorrect boundaries with slab PPPM'],
  [{ ks: 'kspace_style ewald 1e-3\nkspace_modify slab 3.0', bnd: 'boundary p p fm' }, 'Incorrect boundaries with slab Ewald'],
  [{ ks: 'kspace_style pppm 1e-3', bnd: 'boundary p p f' }, 'Cannot use non-periodic boundaries with PPPM'],
  [{ ks: 'kspace_style ewald 1e-3', bnd: 'boundary p p f' }, 'Cannot use non-periodic boundaries with Ewald'],
  [{ ks: 'kspace_style pppm 1e-3\nkspace_modify slab auto', bnd: 'boundary p p s' }, 'Incorrect boundaries with slab PPPM'],
  [{ ks: '', post: 'kspace_modify gewald 0.5' }, 'Kspace_modify command before kspace_style is defined'],
];

/** Inputs native runs (or stops on later) that the engine refuses: [row, engine message, native result]. */
const REFUSED: [Row, string, string][] = [
  // pairs between type 2 atoms and the images of type 1 atoms: native keeps some of them by its bin order
  [{ pre: 'atom_modify first g1', post: 'comm_modify group g1' }, 'comm_modify group: ', 'OK'],
  [{ post: 'kspace_modify mesh 7 7 7' }, 'kspace_modify mesh: each dimension must factor into 2, 3 and 5', 'OK'],
];

describe('modify commands as native LAMMPS', () => {
  it.each(ROWS.map(([r, e]) => [label(r), r, e] as const))('%s', async (_name, r, expected) => {
    const got = await errorOf(r);
    if (expected === null) expect(got).toBeNull();
    else expect(got).toContain(expected);
  });
});

describe('modify inputs the browser engine refuses by name', () => {
  it.each(REFUSED.map(([r, e]) => [label(r), r, e] as const))('%s', async (_name, r, expected) => {
    expect(await errorOf(r)).toContain(expected);
  });
});
