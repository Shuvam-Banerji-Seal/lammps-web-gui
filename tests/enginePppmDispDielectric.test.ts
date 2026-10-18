import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit tests of kspace_style pppm/disp/dielectric (src/engine-gpl/dielectric/pppm_disp_dielectric.ts): the pair-style
 * compatibility, the parameter and option refusals and the error texts (kspace.cpp, pppm_disp.cpp at b332f28), and that
 * the run completes with the efield array of the Coulomb mesh filled. The native parity cases are the tests/oracle/w250pd_*
 * cases (engineOracle.test.ts); these check the bookkeeping and the refusals.
 */

/** Four ions in a 20 A box (atom_style dielectric columns: id mol type q x y z ... as in engineDielectricX.test.ts). */
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

const HEAD = `units real
atom_style dielectric
read_data d.data
`;

const runScript = async (script: string): Promise<{ rows: ThermoRow[]; session: Session }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  session.addFile('d.data', dielData);
  await session.execute(script);
  return { rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row), session };
};

const GOOD_KSPACE = `kspace_style pppm/disp/dielectric 1.0e-4
kspace_modify gewald 0.3 mesh 16 16 16 gewald/disp 0.3 mesh/disp 16 16 16
`;

describe('kspace_style pppm/disp/dielectric: accepted setup', () => {
  it('runs with lj/long/coul/long/dielectric long long and the Coulomb and dispersion parameters set', async () => {
    const { rows } = await runScript(`${HEAD}pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
${GOOD_KSPACE}thermo_style custom step pe elong evdwl
run 0
`);
    expect(rows.length).toBeGreaterThan(0);
    for (const v of Object.values(rows[0])) expect(Number.isFinite(v as number)).toBe(true);
  });

  it('fills the efield of the Coulomb mesh (3 values per atom, finite and not all zero)', async () => {
    const { session } = await runScript(`${HEAD}pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
${GOOD_KSPACE}run 0
`);
    const systems: { ff: { kspace: { efield?: Float64Array } | null } }[] = [];
    session.forEachReplica((sys) => systems.push(sys as unknown as { ff: { kspace: { efield?: Float64Array } | null } }));
    const k = systems[0]?.ff.kspace;
    expect(k?.efield).toBeDefined();
    const ef = k!.efield!;
    expect(ef.length).toBeGreaterThanOrEqual(12);
    let nonzero = false;
    for (let i = 0; i < 12; i++) {
      expect(Number.isFinite(ef[i])).toBe(true);
      if (ef[i] !== 0) nonzero = true;
    }
    expect(nonzero).toBe(true);
  });

  it('accepts the dispersion parameters through kspace_modify disp/auto (no gewald/disp)', async () => {
    const { rows } = await runScript(`${HEAD}pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
kspace_style pppm/disp/dielectric 1.0e-4
kspace_modify gewald 0.3 mesh 16 16 16 disp/auto yes
thermo_style custom step pe
run 0
`);
    expect(rows.length).toBeGreaterThan(0);
  });
});

describe('kspace_style pppm/disp/dielectric: refusals (native text)', () => {
  it('is incompatible with the non-dielectric lj/long/coul/long pair style', async () => {
    await expect(runScript(`${HEAD}pair_style lj/long/coul/long long long 8.0
pair_coeff * * 0.02 2.0
${GOOD_KSPACE}run 0
`)).rejects.toThrow(/KSpace style pppm\/disp\/dielectric is incompatible with Pair style lj\/long\/coul\/long/);
  });

  it('is incompatible with lj/cut/coul/long/dielectric', async () => {
    await expect(runScript(`${HEAD}pair_style lj/cut/coul/long/dielectric 0.2 8.0
pair_coeff * * 0.02 2.0
${GOOD_KSPACE}run 0
`)).rejects.toThrow(/KSpace style pppm\/disp\/dielectric is incompatible with Pair style lj\/cut\/coul\/long\/dielectric/);
  });

  it('stops without dispersion parameters (native: PPPMDisp used but no parameters set)', async () => {
    await expect(runScript(`${HEAD}pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
kspace_style pppm/disp/dielectric 1.0e-4
kspace_modify gewald 0.3 mesh 16 16 16
run 0
`)).rejects.toThrow(/PPPMDisp used but no parameters set/);
  });

  it('refuses the automatic Coulomb G-ewald and grid (gewald and mesh must be set)', async () => {
    await expect(runScript(`${HEAD}pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
kspace_style pppm/disp/dielectric 1.0e-4
kspace_modify gewald/disp 0.3 mesh/disp 16 16 16
run 0
`)).rejects.toThrow(/pppm\/disp.*gewald and mesh/);
  });

  it('refuses kspace_modify slab by name', async () => {
    await expect(runScript(`${HEAD}pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
${GOOD_KSPACE}kspace_modify slab 3.0
run 0
`)).rejects.toThrow(/slab/);
  });

  it('refuses kspace_modify diff ad by name', async () => {
    await expect(runScript(`${HEAD}pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
${GOOD_KSPACE}kspace_modify diff ad
run 0
`)).rejects.toThrow(/diff ad/);
  });

  it('refuses compute efield/atom with the lj/long/coul/long/dielectric pair (native: not supported by pair style)', async () => {
    // native accepts efield/atom only with pppm/dielectric and msm/dielectric (the kspace check); the pair check comes first
    await expect(runScript(`${HEAD}pair_style lj/long/coul/long/dielectric long long 8.0
pair_coeff * * 0.02 2.0
${GOOD_KSPACE}compute ea all efield/atom
run 0
`)).rejects.toThrow(/Compute efield\/atom not supported by pair style/);
  });
});
