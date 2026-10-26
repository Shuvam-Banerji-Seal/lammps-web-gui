import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { PairCoulWolfCS, PairLJCutCoulLongCS, PairLJClass2CoulLongCS } from '../src/engine-gpl/csdiel/coreshell';
import { PairCoulCutDielectric } from '../src/engine-gpl/csdiel/dielectric';
import { PairBuckLongCoulLong } from '../src/engine-gpl/csdiel/buck_long';
import { PAIRS } from '../src/engine-gpl/registry/csdiel';
import { StyleError } from '../src/engine/force/types';

/*
 * Unit tests of the CORESHELL, DIELECTRIC and KSPACE pair styles ported into src/engine-gpl/csdiel. The native parity
 * cases are the tests/oracle/w226cs_* cases (engineOracle.test.ts); these check the physics limits and the refusals.
 */

const runScript = async (script: string, files: Record<string, string>): Promise<{ rows: ThermoRow[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return { rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row) };
};

/** Two ions 4 A apart (atom_style full), no bonds: far from the core-shell r = 0 case. */
const ionsFull = `two ions

2 atoms
2 atom types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 1
2 1

Atoms # full

1 0 1 1.0 4 4 4
2 0 2 -1.0 8 4 4
`;

/** A core and its shell at the same point (r = 0), bonded by special_bonds 1-2 weight 0, plus a far ion. */
const coreShell = `core shell

3 atoms
1 bond types
2 atom types

1 bonds

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 1
2 1

Atoms # full

1 1 1 1.5 5 5 5
2 1 2 -1.5 5 5 5
3 2 1 1.0 12 12 12

Bonds

1 1 1 2
`;

/** Dielectric atoms (atom_style dielectric columns), a pair of ions with dielectric constant eps. */
const dielData = (eps: number): string => `two ions diel

2 atoms
2 atom types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 1
2 1

Atoms # dielectric

1 0 1 1.0 4 4 4 0 0 1 1 0 0 ${eps} 0
2 0 2 -1.0 8 4 4 0 0 1 1 0 0 ${eps} 0
`;

const energyOf = async (pair: string, files: Record<string, string>, head: string): Promise<number> => {
  const { rows } = await runScript(`${head}
${pair}
thermo_style custom step ecoul evdwl
thermo_modify format float %.15g
run 0
`, files);
  return rows[0].ecoul as number;
};

describe('pair styles of src/engine-gpl/csdiel', () => {
  it('registers the ported pair names', () => {
    expect(Object.keys(PAIRS).sort()).toEqual([
      'buck/long/coul/long', 'coul/cut/dielectric', 'coul/long/cs', 'coul/long/dielectric', 'coul/wolf/cs',
      'lj/class2/coul/long/cs', 'lj/cut/coul/debye/dielectric', 'lj/cut/coul/long/cs', 'lj/long/coul/long/dielectric',
      'born/coul/dsf/cs', 'born/coul/wolf/cs', 'born/coul/long/cs', 'buck/coul/long/cs',
    ].sort());
  });

  it('coul/wolf/cs equals coul/wolf for ions well apart (the EPSILON shift is negligible)', async () => {
    const head = 'units real\natom_style full\nread_data d.data\n';
    const plain = await energyOf('pair_style coul/wolf 0.2 8.0\npair_coeff * *', { 'd.data': ionsFull }, head);
    const cs = await energyOf('pair_style coul/wolf/cs 0.2 8.0\npair_coeff * *', { 'd.data': ionsFull }, head);
    expect(Math.abs(cs - plain)).toBeLessThanOrEqual(1e-10 * Math.abs(plain));
  });

  it('lj/cut/coul/long/cs equals lj/cut/coul/long for ions well apart (table 0 and default table)', async () => {
    const head = 'units real\natom_style full\nread_data d.data\n';
    for (const tab of ['pair_modify table 0', '']) {
      const plain = await energyOf(`pair_style lj/cut/coul/long 8.0\npair_coeff * * 0.05 2.5\n${tab}\nkspace_style ewald 1.0e-6`, { 'd.data': ionsFull }, head);
      const cs = await energyOf(`pair_style lj/cut/coul/long/cs 8.0\npair_coeff * * 0.05 2.5\n${tab}\nkspace_style ewald 1.0e-6`, { 'd.data': ionsFull }, head);
      // the /cs styles use the native B0..B5 erfc fit (not the A&S fit of table 0), so they differ from the plain style by ~1e-7
      expect(Math.abs(cs - plain)).toBeLessThanOrEqual(1e-5 * Math.abs(plain));
    }
  });

  it('core-shell pairs at r = 0 give finite Coulomb energies (no NaN) in the /cs styles', async () => {
    const head = 'units real\natom_style full\nread_data d.data\nbond_style harmonic\nbond_coeff 1 60.0 0.0\nspecial_bonds lj/coul 0.0 0.0 1.0\n';
    const cases = [
      'pair_style coul/wolf/cs 0.2 8.0\npair_coeff * *',
      'pair_style lj/cut/coul/long/cs 8.0\npair_coeff * * 0.05 2.5\nkspace_style ewald 1.0e-6',
      'pair_style lj/class2/coul/long/cs 8.0\npair_coeff * * 0.05 2.5\nkspace_style ewald 1.0e-6',
    ];
    for (const pair of cases) {
      const e = await energyOf(pair, { 'd.data': coreShell }, head);
      expect(Number.isFinite(e)).toBe(true);
    }
  });

  it('dielectric coul/cut scales as 1/eps: eps = 2 gives half the energy of eps = 1', async () => {
    const head = 'units real\natom_style dielectric\nread_data d.data\n';
    const one = await energyOf('pair_style coul/cut/dielectric 8.0\npair_coeff * *', { 'd.data': dielData(1) }, head);
    const two = await energyOf('pair_style coul/cut/dielectric 8.0\npair_coeff * *', { 'd.data': dielData(2) }, head);
    expect(two).toBeCloseTo(one / 2, 9);
  });

  it('the core-shell and buck pairs refuse single() with a StyleError naming the style', () => {
    expect(() => new PairCoulWolfCS().single()).toThrow(StyleError);
    expect(() => new PairLJCutCoulLongCS().single()).toThrow(/lj\/cut\/coul\/long\/cs/);
    expect(() => new PairLJClass2CoulLongCS().single()).toThrow(/lj\/class2\/coul\/long\/cs/);
    expect(() => new PairCoulCutDielectric().single()).toThrow(/coul\/cut\/dielectric/);
    expect(() => new PairBuckLongCoulLong().single()).toThrow(/buck\/long\/coul\/long/);
  });

  it('buck/long/coul/long refuses the native-invalid flags and does not mix coefficients', () => {
    const p = new PairBuckLongCoulLong();
    expect(() => p.settings(['off', 'long', '8.0'], {} as never)).toThrow(/LJ6 off not supported/);
    expect(() => p.settings(['long', 'cut', '8.0'], {} as never)).toThrow(/Coulomb cut not supported/);
    p.settings(['long', 'long', '8.0'], {} as never);
    p.allocate(2);
    p.coeff(['1', '1', '100', '0.3', '1.0']);
    expect(() => p.initOne(1, 2)).toThrow(/All pair coeffs are not set/);
  });
});
