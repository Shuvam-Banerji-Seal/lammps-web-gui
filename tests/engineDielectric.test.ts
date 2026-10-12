import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit tests of the DIELECTRIC styles (src/engine-gpl/dielectric): atom_style dielectric, pair lj/cut/coul/cut/dielectric,
 * lj/cut/coul/long/dielectric, kspace pppm/dielectric and fix polarize/bem/gmres, /bem/icc, /functional. The native parity
 * cases are the tests/oracle/w71diel_* cases (engineOracle.test.ts); these check the physics and the refusals.
 */

/** Four point charges (type 2) in a box, the dielectric columns of atom_style dielectric (eps given per atom). */
const dielData = (eps: number): string => `four ions
 
4 atoms
2 atom types
 
0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 1
2 1

Atoms # dielectric

1 0 2 1.0 4 4 4 0 0 1 1 0 0 ${eps} 0
2 0 2 -1.0 7 4 5 0 0 1 1 0 0 ${eps} 0
3 0 2 1.0 4 8 5 0 0 1 1 0 0 ${eps} 0
4 0 2 -1.0 9 9 9 0 0 1 1 0 0 ${eps} 0
`;

/** The same four ions as atom_style full (no dielectric columns), for the unit-epsilon comparison. */
const fullData = `four ions

4 atoms
2 atom types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 1
2 1

Atoms # full

1 0 2 1.0 4 4 4
2 0 2 -1.0 7 4 5
3 0 2 1.0 4 8 5
4 0 2 -1.0 9 9 9
`;

const runScript = async (script: string, files: Record<string, string>): Promise<{ rows: ThermoRow[]; session: Session }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return { rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row), session };
};

const DIEL_HEAD = `units lj
atom_style dielectric
boundary f f f
dimension 3
read_data d.data
`;

describe('atom_style dielectric and the dielectric pair styles', () => {
  it('with a unit dielectric constant, lj/cut/coul/cut/dielectric equals lj/cut/coul/cut', async () => {
    const plain = await runScript(`units lj
atom_style full
boundary f f f
dimension 3
read_data f.data
pair_style lj/cut/coul/cut 3.0 12.0
pair_coeff * * 0.5 1.0
thermo_style custom step evdwl ecoul epair
run 0
`, { 'f.data': fullData });
    const diel = await runScript(`${DIEL_HEAD}pair_style lj/cut/coul/cut/dielectric 3.0 12.0
pair_coeff * * 0.5 1.0
thermo_style custom step evdwl ecoul epair
run 0
`, { 'd.data': dielData(1) });
    expect(diel.rows[0].ecoul).toBeCloseTo(plain.rows[0].ecoul, 12);
    expect(diel.rows[0].evdwl).toBeCloseTo(plain.rows[0].evdwl, 12);
    expect(Math.abs(plain.rows[0].ecoul)).toBeGreaterThan(0);
  });

  it('a dielectric constant of 2 on every atom halves the Coulomb energy (q_scaled = q / epsilon)', async () => {
    const one = await runScript(`${DIEL_HEAD}pair_style lj/cut/coul/cut/dielectric 3.0 12.0
pair_coeff * * 0.5 1.0
thermo_style custom step ecoul
run 0
`, { 'd.data': dielData(1) });
    const two = await runScript(`${DIEL_HEAD}pair_style lj/cut/coul/cut/dielectric 3.0 12.0
pair_coeff * * 0.5 1.0
set group all epsilon 2
thermo_style custom step ecoul
run 0
`, { 'd.data': dielData(1) });
    expect(two.rows[0].ecoul).toBeCloseTo(0.5 * one.rows[0].ecoul, 12);
  });

  it('set epsilon refuses a non-positive constant and a style without dielectric atoms', async () => {
    await expect(runScript(`${DIEL_HEAD}set group all epsilon 0\n`, { 'd.data': dielData(1) })).rejects.toThrow(/Invalid epsilon value 0 in set command/);
    await expect(runScript(`units lj
atom_style full
boundary f f f
dimension 3
read_data f.data
set group all epsilon 2
`, { 'f.data': fullData })).rejects.toThrow(/Cannot set attribute epsilon for atom style full/);
  });

  it('a data file Atoms line with too few dielectric columns is refused by name', async () => {
    const short = dielData(1).replace('1 0 2 1.0 4 4 4 0 0 1 1 0 0 1 0\n', '1 0 2 1.0 4 4 4 0 0 1 1 0 0 1\n');
    expect(short).not.toBe(dielData(1));
    await expect(runScript(`${DIEL_HEAD}`, { 'd.data': short })).rejects.toThrow(/expects 15 values/);
  });

  it('the dielectric pair styles need a kspace style for the long-range variant', async () => {
    await expect(runScript(`${DIEL_HEAD}pair_style lj/cut/coul/long/dielectric 3.0 12.0
pair_coeff * * 0.5 1.0
run 0
`, { 'd.data': dielData(1) })).rejects.toThrow(/needs a kspace style/);
  });
});

describe('restart files keep the dielectric properties', () => {
  it('write_restart and read_restart give back epsilon and the scaled charge (no pair style: the dielectric pair is not stored)', async () => {
    const files = new Map<string, string>();
    const session = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
    session.addFile('d.data', dielData(1));
    await session.execute(`${DIEL_HEAD}set group all epsilon 2
write_restart r.restart
clear
read_restart r.restart
`);
    expect(files.get('r.restart')?.startsWith('LAMMPS-WEB-RESTART 1\n')).toBe(true);
    const st = session.sys.state;
    expect(Array.from(st.custom.get('#epsilon')!.data.subarray(0, st.n))).toEqual([2, 2, 2, 2]);
    expect(Array.from(st.custom.get('#qscaled')!.data.subarray(0, st.n))).toEqual([0.5, -0.5, 0.5, -0.5]);
  });
});

describe('fix polarize and kspace pppm/dielectric refusals', () => {
  it('fix polarize/bem/gmres refuses a pair style without dielectric support', async () => {
    await expect(runScript(`${DIEL_HEAD}pair_style lj/cut/coul/cut 3.0 12.0
pair_coeff * * 0.5 1.0
group interface type 1
fix 3 interface polarize/bem/gmres 1 1.0e-4
run 0
`, { 'd.data': dielData(1) })).rejects.toThrow(/Pair style not compatible with fix 3 \(polarize\/bem\/gmres\)/);
  });

  it('fix_modify rand is refused by the browser engine for the BEM fixes', async () => {
    await expect(runScript(`${DIEL_HEAD}pair_style lj/cut/coul/cut/dielectric 3.0 12.0
pair_coeff * * 0.5 1.0
group interface type 1
fix 3 interface polarize/bem/icc 1 1.0e-4
fix_modify 3 rand 0.1 1234
run 0
`, { 'd.data': dielData(1) })).rejects.toThrow(/fix_modify rand is not supported/);
  });

  it('kspace_modify slab auto is refused for pppm/dielectric', async () => {
    await expect(runScript(`units lj
atom_style dielectric
boundary p p f
dimension 3
read_data d.data
pair_style lj/cut/coul/long/dielectric 3.0 12.0
pair_coeff * * 0.5 1.0
kspace_style pppm/dielectric 0.001
kspace_modify slab auto
run 0
`, { 'd.data': dielData(1).replace(/0 20 zlo zhi/, '0 20 zlo zhi') })).rejects.toThrow(/slab auto is not supported by kspace_style pppm\/dielectric/);
  });
});
