import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * compute msd/nongauss as native LAMMPS computes it (src/EXTRA-COMPUTE/compute_msd_nongauss.cpp compute_vector at
 * b332f28): the displacements from the positions at definition (average yes changes nothing: only compute msd's own
 * compute_vector averages them), and the non-Gaussian parameter 3 <dr^4> / (5 <dr^2>^2) - 1 also at no displacement,
 * where it is 0/0 and prints -nan as C's printf does. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28).
 */

const outcome = async (input: string): Promise<string> => {
  const vals: string[] = [];
  const s = new Session({ emit: (ev: { kind: string; text?: string }) => { if (ev.kind === 'log' && ev.text?.startsWith('VAL ')) vals.push(ev.text); }, writeFile: () => {} } as never);
  try { await s.execute(input); } catch (e) { return 'ERROR: ' + (e as Error).message.split('\n')[0]; }
  return vals.length ? `OK | ${vals.join(' ')}` : 'OK';
};

const ROWS: [string, string][] = [
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nvelocity all create 1.5 87287 loop geom\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nfix 1 all nve\ncompute n all msd/nongauss average no com no\nthermo_style custom step c_n[1] c_n[2] c_n[3]\nthermo_modify format float %.15g\nthermo 10\nrun 30\nprint \"VAL $(c_n[1]:%.12g) $(c_n[2]:%.12g) $(c_n[3]:%.12g)\"\n", "OK | VAL 0.0427854393343 0.00263980900222 -0.134769611439"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nvelocity all create 1.5 87287 loop geom\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nfix 1 all nve\ncompute n all msd/nongauss average no com yes\nthermo_style custom step c_n[1] c_n[2] c_n[3]\nthermo_modify format float %.15g\nthermo 10\nrun 30\nprint \"VAL $(c_n[1]:%.12g) $(c_n[2]:%.12g) $(c_n[3]:%.12g)\"\n", "OK | VAL 0.0427854393343 0.00263980900222 -0.134769611439"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nvelocity all create 1.5 87287 loop geom\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nfix 1 all nve\ncompute n all msd/nongauss average yes com no\nthermo_style custom step c_n[1] c_n[2] c_n[3]\nthermo_modify format float %.15g\nthermo 10\nrun 30\nprint \"VAL $(c_n[1]:%.12g) $(c_n[2]:%.12g) $(c_n[3]:%.12g)\"\n", "OK | VAL 0.0427854393343 0.00263980900222 -0.134769611439"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nvelocity all create 1.5 87287 loop geom\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nfix 1 all nve\ncompute n all msd/nongauss average yes com yes\nthermo_style custom step c_n[1] c_n[2] c_n[3]\nthermo_modify format float %.15g\nthermo 10\nrun 30\nprint \"VAL $(c_n[1]:%.12g) $(c_n[2]:%.12g) $(c_n[3]:%.12g)\"\n", "OK | VAL 0.0427854393343 0.00263980900222 -0.134769611439"],
  ["units lj\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\ncompute n all msd/nongauss\nthermo_style custom step c_n[1] c_n[2] c_n[3]\nrun 0\nprint \"VAL $(c_n[3])\"\n", "OK | VAL -nan"],
];

describe('compute msd/nongauss as native LAMMPS', () => {
  it.each(ROWS.map(([input, native], k) => [k + 1, input, native] as const))('input %i', async (_k, input, native) => {
    expect(await outcome(input)).toBe(native);
  });
});
