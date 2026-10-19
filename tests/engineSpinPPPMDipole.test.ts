import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * kspace_style pppm/dipole/spin (src/engine-gpl/spin/pppm_dipole_spin.ts) and the dipolar pair spin/dipole/long: the
 * refusals and the no-spin error. Parity with native LAMMPS is in tests/oracle/w103spin_*pppm*.in and w103spin_iron_*.in.
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

const header = `units metal
atom_style spin
boundary p p p
atom_modify map array
lattice bcc 2.8665
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 55.845
set group all spin/random 11 2.2
pair_style hybrid/overlay spin/exchange 3.5 spin/dipole/long 8.0
pair_coeff * * spin/exchange exchange 3.4 0.02726 0.2171 1.841
pair_coeff * * spin/dipole/long 8.0
`;

describe('kspace_style pppm/dipole/spin', () => {
  it('runs a short spin dipolar run', async () => {
    const r = await run(`${header}kspace_style pppm/dipole/spin 1.0e-4\nrun 0\n`);
    expect(r.error).toBeNull();
  });

  it('refuses the slab correction and the analytic differentiation by name', async () => {
    const slab = await run(`${header}kspace_style pppm/dipole/spin 1.0e-4\nkspace_modify slab 3.0\nrun 0\n`);
    expect(slab.error).toContain('kspace_modify slab is not supported by kspace_style pppm/dipole/spin');
    const ad = await run(`${header}kspace_style pppm/dipole/spin 1.0e-4\nkspace_modify diff ad\nrun 0\n`);
    expect(ad.error).toContain('diff ad');
  });

  it('computes the per-atom energy (oracle case w272sp_pppm_spin_peatom) and refuses the per-atom virial as native', async () => {
    const e = await run(`${header}kspace_style pppm/dipole/spin 1.0e-4\ncompute pa all pe/atom\ncompute r all reduce sum c_pa\nthermo_style custom step pe c_r\nrun 0\n`);
    expect(e.error).toBeNull();
    // Measured with native LAMMPS: any compute stress/atom stops the run with this message (pppm_dipole_spin.cpp compute)
    const v = await run(`${header}kspace_style pppm/dipole/spin 1.0e-4\ncompute sa all stress/atom NULL ke\ncompute r all reduce sum c_sa[1]\nthermo_style custom step c_r\nrun 0\n`);
    expect(v.error).toContain('Cannot (yet) compute per-atom virial with kspace style pppm/dipole/spin');
  });

  it('refuses a system without spins', async () => {
    const r = await run(`units metal
atom_style spin
boundary p p p
atom_modify map array
lattice bcc 2.8665
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 55.845
pair_style spin/dipole/long 8.0
pair_coeff * * 8.0
kspace_style pppm/dipole/spin 1.0e-4
run 0
`);
    expect(r.error).toContain('no spins');
  });
});
