import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * kspace_style pppm/dipole (src/engine-gpl/dipole/pppm_dipole.ts): the refusals by name and the no-dipole error. The
 * parity with native LAMMPS is in tests/oracle/w135pd_*.in (point dipoles, fixed mesh, automatic mesh, non-cubic box,
 * and a short run with fix nve/sphere update dipole).
 */

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    error = err && 'message' in err ? String(err.message) : String(e);
  }
  return { error };
};

const header = (charge: string) => `units lj
atom_style hybrid charge sphere dipole
boundary p p p
lattice fcc 0.4
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
set group all diameter 1.0
set group all density 1.0
set group all charge ${charge}
set group all dipole/random 4421 0.7
pair_style lj/cut/dipole/long 2.5
pair_coeff * * 1.0 1.0
`;

describe('kspace_style pppm/dipole', () => {
  it('runs a point-dipole system with zero charges', async () => {
    const r = await run(`${header('0.0')}kspace_style pppm/dipole 1.0e-4\nkspace_modify gewald 0.5 mesh 8 8 8\nrun 0\n`);
    expect(r.error).toBeNull();
  });

  it('runs a system with point charges and point dipoles (charge-charge and charge-dipole channels)', async () => {
    // Coulomb cutoff 0.5 is below the fcc nearest-neighbour distance: no real-space Coulomb pair term exists
    const h = header('0.3').replace('lj/cut/dipole/long 2.5', 'lj/cut/dipole/long 2.5 0.5');
    const r = await run(`${h}kspace_style pppm/dipole 1.0e-4\nkspace_modify gewald 0.5 mesh 8 8 8\nrun 0\n`);
    expect(r.error).toBeNull();
  });

  it('refuses point charges inside the Coulomb cutoff of lj/cut/dipole/long by name (real-space charge terms not ported)', async () => {
    const r = await run(`${header('0.3')}kspace_style pppm/dipole 1.0e-4\nkspace_modify gewald 0.5 mesh 8 8 8\nrun 0\n`);
    expect(r.error).toContain('with point charges inside the Coulomb cutoff is not supported');
  });

  it('refuses per-atom stress by name (the per-atom virial of pppm/dipole is not ported)', async () => {
    const r = await run(`${header('0.0')}kspace_style pppm/dipole 1.0e-4\nkspace_modify gewald 0.5 mesh 8 8 8\ncompute sa all stress/atom NULL\nthermo_style custom step c_sa[1]\nrun 0\n`);
    expect(r.error).toContain('stress/atom');
  });

  it('refuses kspace_modify slab, diff, force and minorder by name', async () => {
    for (const kw of ['slab 3.0', 'diff ad', 'force 1.0e-4', 'minorder 2']) {
      const r = await run(`${header('0.0')}kspace_style pppm/dipole 1.0e-4\nkspace_modify ${kw}\nrun 0\n`);
      expect(r.error).toContain(`${kw.split(' ')[0]}`);
      expect(r.error).toContain('is not supported by kspace_style pppm/dipole');
    }
  });

  it('refuses per-atom energy and virial by name', async () => {
    const r = await run(`${header('0.0')}kspace_style pppm/dipole 1.0e-4\nkspace_modify gewald 0.5 mesh 8 8 8\ncompute pa all pe/atom\nthermo_style custom step c_pa\nrun 0\n`);
    expect(r.error).toContain('pe/atom');
  });

  it('refuses a system with no dipoles', async () => {
    const r = await run(`units lj
atom_style hybrid charge sphere dipole
boundary p p p
lattice fcc 0.4
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
set group all diameter 1.0
set group all density 1.0
set group all charge 0.0
set group all dipole 0.0 0.0 0.0
pair_style lj/cut/dipole/long 2.5
pair_coeff * * 1.0 1.0
kspace_style pppm/dipole 1.0e-4
kspace_modify gewald 0.5 mesh 8 8 8
run 0
`);
    expect(r.error).toContain('no dipoles');
  });

  it('refuses electron units with dipoles', async () => {
    const r = await run(`units electron
atom_style hybrid charge sphere dipole
boundary p p p
lattice fcc 0.4
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
set group all diameter 1.0
set group all density 1.0
set group all charge 0.0
set group all dipole/random 4421 0.7
pair_style lj/cut/dipole/long 2.5
pair_coeff * * 1.0 1.0
kspace_style pppm/dipole 1.0e-4
kspace_modify gewald 0.5 mesh 8 8 8
run 0
`);
    expect(r.error).toContain("'electron' units");
  });
});
