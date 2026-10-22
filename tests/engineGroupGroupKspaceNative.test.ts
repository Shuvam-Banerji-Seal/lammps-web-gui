import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * compute group/group kspace yes: the checks native LAMMPS makes (src/compute_group_group.cpp ComputeGroupGroup::init,
 * src/KSPACE/pppm.cpp PPPM::compute_group_group at b332f28) and its net-charge warning. Measured with native LAMMPS
 * (/home/roy/.local/bin/lmp, b332f28). The values are compared with native in the oracle cases group_group_kspace_*.
 */

const HEAD = `units metal
atom_style charge
lattice fcc 5.64
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box
lattice fcc 5.64 origin 0.5 0.0 0.0
create_atoms 2 box
mass 1 22.99
mass 2 35.45
set type 1 charge 1.0
set type 2 charge -1.0
group na type 1
group cl type 2
`;

const run = async (text: string): Promise<{ error: string | null; log: string }> => {
  const lines: string[] = [];
  const s = new Session({ emit: (e: { kind: string; text?: string }) => { if (e.kind === 'log' && e.text) lines.push(e.text); }, writeFile: () => {} } as never);
  try {
    await s.execute(HEAD + text);
    return { error: null, log: lines.join('\n') };
  } catch (e) {
    return { error: (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''), log: lines.join('\n') };
  }
};

describe('compute group/group kspace yes, as native LAMMPS', () => {
  it('needs a kspace style that computes group/group terms', async () => {
    expect((await run('pair_style lj/cut/coul/cut 5.0\npair_coeff * * 0.01 3.0\ncompute g na group/group cl kspace yes\nthermo_style custom step c_g\nrun 0\n')).error)
      .toBe('No Kspace style defined for compute group/group');
    expect((await run('pair_style lj/cut/coul/msm 5.0\npair_coeff * * 0.01 3.0\nkspace_style msm 1e-4\ncompute g na group/group cl kspace yes\nthermo_style custom step c_g\nrun 0\n')).error)
      .toBe('Kspace style does not support compute group/group');
    expect((await run('pair_style lj/cut/coul/long 5.0\npair_coeff * * 0.01 3.0\nkspace_style pppm 1e-4\nkspace_modify diff ad\ncompute g na group/group cl kspace yes\nthermo_style custom step c_g\nrun 0\n')).error)
      .toBe('Cannot (yet) use kspace_modify diff ad with compute group/group');
    expect((await run('pair_style lj/cut/coul/long 5.0\npair_coeff * * 0.01 3.0\nkspace_style pppm 1e-4\ncompute g na group/group cl kspace yes boundary maybe\nrun 0\n')).error)
      .toBe("Expected boolean parameter instead of 'maybe' in input script or data file");
  });

  it('warns when both groups have a net charge, and kspace no needs no kspace style', async () => {
    const r = await run('pair_style lj/cut/coul/long 5.0\npair_coeff * * 0.01 3.0\nkspace_style pppm 1e-4\ncompute g na group/group cl kspace yes\nthermo_style custom step c_g\nrun 0\n');
    expect(r.error).toBeNull();
    expect(r.log).toContain('WARNING: Both groups in compute group/group have a net charge; the Kspace boundary correction to energy will be non-zero');
    expect((await run('pair_style lj/cut/coul/cut 5.0\npair_coeff * * 0.01 3.0\ncompute g na group/group cl kspace no\nthermo_style custom step c_g\nrun 0\n')).error).toBeNull();
  });

  it('refuses pppm/tip4p by name (native spreads the oxygen charges from the M-site grid cells)', async () => {
    const text = `units real
atom_style full
region box block 0 12 0 12 0 12
create_box 2 box bond/types 1 angle/types 1 extra/bond/per/atom 2 extra/angle/per/atom 1 extra/special/per/atom 2
mass 1 15.9994
mass 2 1.008
create_atoms 1 single 6 6 6
create_atoms 2 single 6.9572 6 6
create_atoms 2 single 5.7600 6.9266 6
set type 1 charge -0.8476
set type 2 charge 0.4238
group o type 1
group h type 2
pair_style lj/cut/tip4p/long 1 2 1 1 0.15 8.0
pair_coeff * * 0.0 1.0
pair_coeff 1 1 0.155 3.166
bond_style harmonic
bond_coeff 1 450 0.9572
angle_style harmonic
angle_coeff 1 55 104.52
create_bonds single/bond 1 1 2
create_bonds single/bond 1 1 3
create_bonds single/angle 1 2 1 3
kspace_style pppm/tip4p 1e-4
compute g o group/group h kspace yes
thermo_style custom step c_g
run 0
`;
    const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
    await expect(s.execute(text)).rejects.toThrow('compute group/group kspace yes with kspace_style pppm/tip4p is not supported by the browser engine');
  });
});
