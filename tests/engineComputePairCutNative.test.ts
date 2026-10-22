import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Computes that take their neighbours from the pair style, checked at init as native LAMMPS does: the pair style's
 * init comes first (force->init, which sets its largest cutoff, cutforce), then each compute's init requires a pair
 * style and a cutoff no longer than cutforce (src/compute_cna_atom.cpp, compute_cluster_atom.cpp,
 * compute_aggregate_atom.cpp, compute_coord_atom.cpp, compute_orientorder_atom.cpp, EXTRA-COMPUTE/compute_cnp_atom.cpp,
 * compute_hexorder_atom.cpp, ML-SNAP/compute_snap.cpp and compute_sna_atom.cpp at b332f28). Measured with native
 * LAMMPS (/home/roy/.local/bin/lmp, b332f28): each row is an input and native's result: OK or the first line of its
 * error, then the values the input prints after "|".
 */

const outcome = async (input: string): Promise<string> => {
  const vals: string[] = [];
  const s = new Session({ emit: (ev: { kind: string; text?: string }) => { if (ev.kind === 'log' && ev.text?.startsWith('VAL ')) vals.push(ev.text); }, writeFile: () => {} } as never);
  let r = 'OK';
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(input); } catch (e) { r = 'ERROR: ' + (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return vals.length ? `${r} | ${vals.join(' ')}` : r;
};

const ROWS: [string, string][] = [
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all cna/atom 3.0\nrun 0\n", "ERROR: Compute cna/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all aggregate/atom 3.0\nrun 0\n", "ERROR: Compute aggregate/atom used when bonds are not allowed"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all coord/atom cutoff 3.0\nrun 0\n", "ERROR: Compute coord/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all cluster/atom 3.0\nrun 0\n", "ERROR: Compute cluster/atom cutoff 3 is longer than pairwise cutoff 2.5"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all orientorder/atom cutoff 3.0\nrun 0\n", "ERROR: Compute orientorder/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all hexorder/atom cutoff 3.0\nrun 0\n", "ERROR: Compute hexorder/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all cnp/atom 3.0\nrun 0\n", "ERROR: Compute cnp/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all snap 1.5 0.99363 6 1.0 1.0\nrun 0\n", "ERROR: Compute snap cutoff 3 is longer than pairwise cutoff 2.5"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all sna/atom 1.5 0.99363 6 1.0 1.0\nrun 0\n", "ERROR: Compute sna/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all snad/atom 1.5 0.99363 6 1.0 1.0\nrun 0\n", "ERROR: Compute snad/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all snav/atom 1.5 0.99363 6 1.0 1.0\nrun 0\n", "ERROR: Compute snav/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all cna/atom 2.0\nrun 0\n", "OK"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all coord/atom cutoff 2.0\nrun 0\n", "OK"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all orientorder/atom cutoff 2.0\nrun 0\n", "OK"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all snap 1.0 0.99363 6 1.0 1.0\nrun 0\n", "OK"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all cluster/atom 2.0\nrun 0\n", "OK"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\ncompute 1 all cna/atom 2.0\nrun 0\n", "ERROR: Compute cna/atom requires a pair style be defined"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\ncompute 1 all cluster/atom 2.0\nrun 0\n", "ERROR: Compute cluster/atom requires a pair style to be defined"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\ncompute 1 all orientorder/atom\nrun 0\n", "ERROR: Compute orientorder/atom requires a pair style be defined"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\ncompute 1 all hexorder/atom\nrun 0\n", "ERROR: Compute hexorder/atom requires a pair style be defined"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\ncompute 1 all cnp/atom 2.0\nrun 0\n", "ERROR: Compute cnp/atom requires a pair style be defined"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\ncompute 1 all coord/atom cutoff 2.0\nrun 0\n", "ERROR: Compute coord/atom requires a pair style be defined"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\ncompute 1 all sna/atom 1.0 0.99363 6 1.0 1.0 1.0 1.0\nrun 0\n", "ERROR: Compute sna/atom requires a pair style be defined"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\ncompute 1 all snap 1.0 0.99363 6 1.0 1.0 1.0 1.0\nrun 0\n", "ERROR: Compute snap requires a pair style be defined"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\ncompute 1 all cna/atom 2.5\nrun 0\n", "OK"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 1.0 1.0 3.0\ncompute 1 all cna/atom 2.8\nrun 0\n", "OK"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\npair_style hybrid lj/cut 2.5 soft 3.0\npair_coeff * * lj/cut 1.0 1.0\npair_coeff 2 2 soft 1.0\ncompute 1 all coord/atom cutoff 2.9\nrun 0\n", "OK"],
  ["units lj\natom_style bond\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box bond/types 1 extra/bond/per/atom 2\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\ncompute 1 all aggregate/atom 2.0\nrun 0\n", "ERROR: Compute aggregate/atom requires a bond style to be defined"],
  ["units lj\natom_style bond\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box bond/types 1 extra/bond/per/atom 2\ncreate_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nbond_style harmonic\nbond_coeff 1 1.0 1.0\ncompute 1 all aggregate/atom 3.0\nrun 0\n", "ERROR: Compute cluster/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\ncompute o all orientorder/atom cutoff 2.0 components 6\ncompute 1 all coord/atom orientorder o 0.5\ncompute r all reduce sum c_1\nthermo_style custom step c_r\nrun 0\nprint \"VAL $(c_r)\"\n", "OK | VAL 1944"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\ncompute o all orientorder/atom cutoff 3.0 components 6\ncompute 1 all coord/atom orientorder o 0.5\nrun 0\n", "ERROR: Compute orientorder/atom cutoff is longer than pairwise cutoff"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\ncompute 1 all cna/atom 2.0\nrun 0\npair_style lj/cut 1.5\npair_coeff * * 1.0 1.0\nrun 0\n", "ERROR: Compute cna/atom cutoff is longer than pairwise cutoff"],
];

describe('computes checked against the pair cutoff at init as native LAMMPS', () => {
  it.each(ROWS.map(([input, native], k) => [k + 1, input, native] as const))('input %i', async (_k, input, native) => {
    expect(await outcome(input)).toBe(native);
  });
});
