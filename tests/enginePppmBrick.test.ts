import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * kspace_style pppm with a finite atom outside the grid brick: native LAMMPS (src/KSPACE/pppm.cpp PPPM::particle_map
 * and src/grid3d.cpp Grid3d::ghost_grid at b332f28) stops with "Out of range atoms - cannot compute PPPM" when the
 * stencil of an atom leaves the brick, which reaches 0.5 skin (+ qdist) past the box. With neigh_modify check no and no
 * rebuild the atoms are not wrapped back, so a moving atom leaves the brick at a definite step. Measured with native
 * LAMMPS (/home/roy/.local/bin/lmp, b332f28) on the inputs below, thermo every step: the last row printed and then the
 *  stop for each case below, and no stop for a slab with a non-periodic z (run 150 completes).
 */

const base = (opts: { region?: string; boundary?: string; kspace?: string; skin?: number; v?: string }) => `units real
${opts.boundary ?? ''}
atom_style charge
region box ${opts.region ?? 'block 0 12 0 12 0 12'}
create_box 2 box
create_atoms 1 single 3 3 3
create_atoms 2 single 8 8 8
create_atoms 1 single 3 9 6
create_atoms 2 single 9 3 6
mass * 16.0
set type 1 charge 0.5
set type 2 charge -0.5
pair_style lj/cut/coul/long 5.0
pair_coeff * * 0.1 3.0
kspace_style pppm 1.0e-4
${opts.kspace ?? ''}
neighbor ${opts.skin ?? 2.0} bin
neigh_modify every 1000 delay 0 check no
velocity all set ${opts.v ?? '0.05 0.0 0.0'}
fix 1 all nve
thermo 1
thermo_style custom step pe
run 150
`;

const run = async (input: string): Promise<{ last: number; error: string | null }> => {
  let last = -1;
  const s = new Session({ emit: (e: { kind: string; row?: { step: number } }) => { if (e.kind === 'thermo') last = e.row!.step; }, writeFile: () => {} } as never);
  try {
    await s.execute(input);
    return { last, error: null };
  } catch (e) {
    return { last, error: (e as Error).message };
  }
};

describe('PPPM grid brick (Out of range atoms)', () => {
  it.each([
    ['order 5 (default), skin 2', {}, 98],
    ['order 5 set by kspace_modify', { kspace: 'kspace_modify order 5' }, 98],
    ['a triclinic box (prism 2 1 -1.5)', { region: 'prism 0 12 0 12 0 12 2.0 1.0 -1.5' }, 113],
    ['skin 1', { skin: 1.0 }, 72],
    ['order 6, a diagonal velocity', { kspace: 'kspace_modify order 6', v: '-0.04 0.03 0.02' }, 109],
  ])('%s: stops after the row of step %d, as native', async (_name, opts, lastRow) => {
    const r = await run(base(opts));
    expect(r.error).toMatch(/Out of range atoms - cannot compute PPPM/);
    expect(r.last).toBe(lastRow);
  });

  it('a slab with a non-periodic z (the brick reaches the top of the extended mesh) runs to the end, as native', async () => {
    const r = await run(base({ boundary: 'boundary p p f', kspace: 'kspace_modify slab 3.0', v: '0.0 0.0 0.05' }));
    expect(r.error).toBeNull();
    expect(r.last).toBe(150);
  });
});
