import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * compute ti kspace: dU/dlambda of the kspace term, the per-atom kspace energies of the atom types summed over the
 * owned atoms, times df/dlambda / f (src/EXTRA-COMPUTE/compute_ti.cpp ComputeTI::compute_scalar at b332f28). Measured
 * with native LAMMPS (/home/roy/.local/bin/lmp, b332f28): ewald and pppm, the types 1, 2 and 1*2 of a charged binary
 * fcc lattice; each row is the input and native's result (OK, then the printed values).
 */

const outcome = async (input: string): Promise<string> => {
  const vals: string[] = [];
  const s = new Session({ emit: (ev: { kind: string; text?: string }) => { if (ev.kind === 'log' && ev.text?.startsWith('VAL ')) vals.push(ev.text); }, writeFile: () => {} } as never);
  try { await s.execute(input); } catch (e) { return 'ERROR: ' + (e as Error).message.split('\n')[0]; }
  return vals.length ? `OK | ${vals.join(' ')}` : 'OK';
};

const ROWS: [string, string][] = [
  ["units lj\natom_style charge\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\nset type 1 type/fraction 2 0.5 12345\nset type 1 charge 0.5\nset type 2 charge -0.5\ndisplace_atoms all random 0.05 0.05 0.05 4321\npair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1.0e-5\nvariable a equal 0.7\nvariable b equal 1.3\ncompute t all ti kspace 1 v_a v_b\ncompute ek all pe kspace\nthermo_style custom step c_t c_ek\nthermo_modify format float %.15g\nrun 0\nprint \"VAL $(c_t:%.12g) $(c_ek:%.12g)\"\n", "OK | VAL -10.0230279264 -4.53904067039"],
  ["units lj\natom_style charge\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\nset type 1 type/fraction 2 0.5 12345\nset type 1 charge 0.5\nset type 2 charge -0.5\ndisplace_atoms all random 0.05 0.05 0.05 4321\npair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1.0e-5\nvariable a equal 0.7\nvariable b equal 1.3\ncompute t all ti kspace 2 v_a v_b\ncompute ek all pe kspace\nthermo_style custom step c_t c_ek\nthermo_modify format float %.15g\nrun 0\nprint \"VAL $(c_t:%.12g) $(c_ek:%.12g)\"\n", "OK | VAL 1.59338096715 -4.53904067039"],
  ["units lj\natom_style charge\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\nset type 1 type/fraction 2 0.5 12345\nset type 1 charge 0.5\nset type 2 charge -0.5\ndisplace_atoms all random 0.05 0.05 0.05 4321\npair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style ewald 1.0e-5\nvariable a equal 0.7\nvariable b equal 1.3\ncompute t all ti kspace 1*2 v_a v_b\ncompute ek all pe kspace\nthermo_style custom step c_t c_ek\nthermo_modify format float %.15g\nrun 0\nprint \"VAL $(c_t:%.12g) $(c_ek:%.12g)\"\n", "OK | VAL -8.42964695929 -4.53904067039"],
  ["units lj\natom_style charge\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\nset type 1 type/fraction 2 0.5 12345\nset type 1 charge 0.5\nset type 2 charge -0.5\ndisplace_atoms all random 0.05 0.05 0.05 4321\npair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style pppm 1.0e-5\nvariable a equal 0.7\nvariable b equal 1.3\ncompute t all ti kspace 1 v_a v_b\ncompute ek all pe kspace\nthermo_style custom step c_t c_ek\nthermo_modify format float %.15g\nrun 0\nprint \"VAL $(c_t:%.12g) $(c_ek:%.12g)\"\n", "OK | VAL -9.9458819617 -4.51622540164"],
  ["units lj\natom_style charge\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\nset type 1 type/fraction 2 0.5 12345\nset type 1 charge 0.5\nset type 2 charge -0.5\ndisplace_atoms all random 0.05 0.05 0.05 4321\npair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style pppm 1.0e-5\nvariable a equal 0.7\nvariable b equal 1.3\ncompute t all ti kspace 2 v_a v_b\ncompute ek all pe kspace\nthermo_style custom step c_t c_ek\nthermo_modify format float %.15g\nrun 0\nprint \"VAL $(c_t:%.12g) $(c_ek:%.12g)\"\n", "OK | VAL 1.5586062158 -4.51622540164"],
  ["units lj\natom_style charge\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 2 box\ncreate_atoms 1 box\nmass * 1.0\nset type 1 type/fraction 2 0.5 12345\nset type 1 charge 0.5\nset type 2 charge -0.5\ndisplace_atoms all random 0.05 0.05 0.05 4321\npair_style lj/cut/coul/long 2.5\npair_coeff * * 1.0 1.0\nkspace_style pppm 1.0e-5\nvariable a equal 0.7\nvariable b equal 1.3\ncompute t all ti kspace 1*2 v_a v_b\ncompute ek all pe kspace\nthermo_style custom step c_t c_ek\nthermo_modify format float %.15g\nrun 0\nprint \"VAL $(c_t:%.12g) $(c_ek:%.12g)\"\n", "OK | VAL -8.3872757459 -4.51622540164"],
];

describe('compute ti kspace as native LAMMPS', () => {
  it.each(ROWS.map(([input, native], k) => [k + 1, input, native] as const))('input %i', async (_k, input, native) => {
    expect(await outcome(input)).toBe(native);
  });
});
