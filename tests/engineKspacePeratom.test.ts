import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Per-atom k-space energy and virial (compute pe/atom, stress/atom). kspace_style ewald, pppm and pppm/cg tally them as
 * native ewald.cpp, pppm.cpp and pppm_cg.cpp do (compared with native in tests/oracle/w232ka_*_peratom.in); the styles
 * whose per-atom terms the engine does not compute refuse instead of leaving the k-space part out of the sums.
 */
const SETUP = `units real
atom_style charge
lattice sc 3.1
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 4321
set type 1 charge 0.6
set type 2 charge -0.6
mass * 20.0
pair_style lj/cut/coul/long 5.0
pair_coeff * * 0.1 2.6
`;

const run = async (input: string) => {
  const rows: Record<string, number>[] = [];
  let error = '';
  const s = new Session({ emit: (e) => { if (e.kind === 'thermo') rows.push(e.row as Record<string, number>); if (e.kind === 'error') error ||= e.message; }, writeFile: () => {} });
  try { await s.execute(input); } catch (e) { error ||= String(e); }
  return { rows, error };
};

describe('per-atom k-space energy and virial', () => {
  for (const ks of ['ewald 1.0e-5', 'pppm 1.0e-5']) {
    it(`${ks.split(' ')[0]}: the per-atom energies and virials sum to pe and the pressure`, async () => {
      const r = await run(`${SETUP}kspace_style ${ks}
compute pe all pe/atom
compute st all stress/atom NULL
compute s all reduce sum c_pe c_st[1] c_st[2] c_st[3]
variable p equal -(c_s[2]+c_s[3]+c_s[4])/(3*vol)
thermo_style custom step pe c_s[1] press v_p
thermo_modify norm no
run 0
`);
      expect(r.error).toBe('');
      expect(r.rows[0]['c_s[1]']).toBeCloseTo(r.rows[0].pe, 8);
      expect(r.rows[0].v_p).toBeCloseTo(r.rows[0].press, 6);
    });
  }

  it('ewald/dipole: pe/atom sums to the k-space energy (parity with native per atom in tests/enginePerAtomTallyEwaldDipole)', async () => {
    const r = await run(`units lj
atom_style hybrid sphere dipole
lattice sc 0.8
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
set type 1 dipole/random 1234 1.0
mass * 1.0
pair_style lj/cut/dipole/long 2.5
pair_coeff * * 1 1
kspace_style ewald/dipole 1.0e-4
compute pe all pe/atom kspace
compute r all reduce sum c_pe
thermo_style custom step elong c_r
thermo_modify norm no
run 0
`);
    expect(r.error).toBeFalsy();
    const row = r.rows[r.rows.length - 1];
    expect(Math.abs(row.c_r - row.elong)).toBeLessThan(1e-10 * Math.abs(row.elong));
  });
});
