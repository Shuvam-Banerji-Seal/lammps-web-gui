import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit tests of compute efield/atom (src/engine-gpl/dielectric/compute_efield_atom.ts): the keywords, the pair and kspace
 * terms, the group mask and the native refusals. The native parity cases are the tests/oracle/w243de_efield_* cases
 * (engineOracle.test.ts); these check the bookkeeping and the error texts (compute_efield_atom.cpp at b332f28).
 */

/** Four ions in a box (atom_style dielectric columns: area, ed, em, epsilon, curvature, q_scaled not in the file). */
const dielData = `four ions

4 atoms
2 atom types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 1
2 1

Atoms # dielectric

1 0 2 1.0 4 4 4 0 0 1 1 0 0 10.0 0
2 0 2 -1.0 7 4 5 0 0 1 1 0 0 2.0 0
3 0 2 1.0 4 8 5 0 0 1 1 0 0 45.0 0
4 0 2 -1.0 9 9 9 0 0 1 1 0 0 10.0 0
`;

const runScript = async (script: string): Promise<{ rows: ThermoRow[]; session: Session }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  session.addFile('d.data', dielData);
  await session.execute(script);
  return { rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row), session };
};

const HEAD = `units lj
atom_style dielectric
boundary f f f
dimension 3
read_data d.data
`;

const PAIR = `pair_style lj/cut/coul/cut/dielectric 3.0 12.0
pair_coeff * * 0.5 1.0
`;

describe('compute efield/atom', () => {
  it('the default (pair and kspace) equals the pair term when no kspace style is defined', async () => {
    const r = await runScript(`${HEAD}${PAIR}compute ea all efield/atom
compute ep all efield/atom pair
compute sa all reduce sum c_ea[1] c_ea[2] c_ea[3]
compute sp all reduce sum c_ep[1] c_ep[2] c_ep[3]
thermo_style custom step c_sa[1] c_sa[2] c_sa[3] c_sp[1] c_sp[2] c_sp[3]
run 0
`);
    const row = r.rows[0];
    const sum = [row['c_sa[1]'], row['c_sa[2]'], row['c_sa[3]']];
    expect(sum.every((v) => Number.isFinite(v))).toBe(true);
    expect(Math.abs(sum[0]) + Math.abs(sum[1]) + Math.abs(sum[2])).toBeGreaterThan(0);
    expect(row['c_sp[1]']).toBeCloseTo(sum[0], 12);
    expect(row['c_sp[2]']).toBeCloseTo(sum[1], 12);
    expect(row['c_sp[3]']).toBeCloseTo(sum[2], 12);
  });

  it('the kspace keyword without a kspace style gives a zero field (no pair term, no kspace term)', async () => {
    const r = await runScript(`${HEAD}${PAIR}compute ek all efield/atom kspace
compute sk all reduce sum c_ek[1] c_ek[2] c_ek[3]
thermo_style custom step c_sk[1] c_sk[2] c_sk[3]
run 0
`);
    const row = r.rows[0];
    expect([row['c_sk[1]'], row['c_sk[2]'], row['c_sk[3]']]).toEqual([0, 0, 0]);
  });

  it('a group restricts the field to its atoms (zero outside the group, as native)', async () => {
    const r = await runScript(`${HEAD}${PAIR}group half id 1 2
compute ea all efield/atom
compute eh half efield/atom
compute sa half reduce sum c_ea[1] c_ea[2] c_ea[3]
compute sh all reduce sum c_eh[1] c_eh[2] c_eh[3]
thermo_style custom step c_sa[1] c_sa[2] c_sa[3] c_sh[1] c_sh[2] c_sh[3]
run 0
`);
    const row = r.rows[0];
    for (let k = 1; k <= 3; k++) expect(row[`c_sa[${k}]`]).toBeCloseTo(row[`c_sh[${k}]`], 12);
  });

  it('refuses a pair style without efield (native text) and an illegal keyword', async () => {
    await expect(runScript(`${HEAD}pair_style lj/cut/coul/cut 3.0 12.0
pair_coeff * * 0.5 1.0
compute ea all efield/atom
run 0
`)).rejects.toThrow(/Compute efield\/atom not supported by pair style/);
    await expect(runScript(`${HEAD}${PAIR}compute ea all efield/atom foo
run 0
`)).rejects.toThrow(/Illegal compute efield\/atom command/);
  });

  it('refuses a kspace style other than pppm/dielectric or msm/dielectric (native text)', async () => {
    const body = `pair_style coul/long/dielectric 8.0
pair_coeff * *
kspace_style ewald 1.0e-5
compute ea all efield/atom
run 0
`;
    // As native LAMMPS does (src/force.cpp Force::init at b332f28): the kspace init comes before the computes' init, so in
    // the f f f box native stops on Ewald's boundary check first (measured with native LAMMPS)
    await expect(runScript(`${HEAD}${body}`)).rejects.toThrow('Cannot use non-periodic boundaries with Ewald');
    await expect(runScript(`${HEAD.replace('boundary f f f', 'boundary p p p')}${body}`)).rejects.toThrow(/Compute efield\/atom not supported by kspace style/);
  });

  it('refuses the lj/long/coul/long/dielectric pair (no efield in native compute efield/atom)', async () => {
    const body = `pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
kspace_style ewald/disp 1.0e-5
compute ea all efield/atom
run 0
`;
    // the kspace init first, as above (measured with native LAMMPS: Cannot use non-periodic boundaries with EwaldDisp)
    await expect(runScript(`${HEAD}${body}`)).rejects.toThrow('Cannot use non-periodic boundaries with EwaldDisp');
    await expect(runScript(`${HEAD.replace('boundary f f f', 'boundary p p p')}${body}`)).rejects.toThrow(/Compute efield\/atom not supported by pair style/);
  });
});
