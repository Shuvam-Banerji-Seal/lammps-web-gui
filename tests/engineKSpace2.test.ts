import { describe, expect, it } from 'vitest';
import { makeErfcTable, erfcPoly, EWALD_F } from '../src/engine/force/erfc';

/*
 * pair_modify table N emulation (erfc.ts makeErfcTable) checked against native
 * LAMMPS, black box. Input for each value: two charges +1 at separation r,
 * units real, pair_style coul/long 8.0, kspace_style ewald 1e-3,
 * kspace_modify gewald 0.3, pair_modify table 12, thermo ecoul (%.15g).
 * Native ecoul = qqrd2e erfc(g r)/r with qqrd2e = 332.06371.
 */
const C = 332.06371, G = 0.3;
const NATIVE_TABLE12: Array<[number, number]> = [
  [1.5, 116.115657393582],
  [3.0, 22.4798041511354],
];

describe('pair_modify table emulation (erfc.ts)', () => {
  it('energy table reproduces native table-12 ecoul at node and off-node distances', () => {
    const T = makeErfcTable(12, G, 100);
    for (const [r, e] of NATIVE_TABLE12) {
      expect(Math.abs(C * T.energy(r * r) - e) / e).toBeLessThan(1e-12);
    }
  });

  it('table 12 error is ~1e-6 relative to the polynomial no-table value at r = 5.1', () => {
    const T = makeErfcTable(12, G, 100);
    const r = 5.1, x = G * r, ex = Math.exp(-x * x);
    const rel = (T.energy(r * r) * r) / erfcPoly(x, ex) - 1;
    expect(Math.abs(rel)).toBeLessThan(1e-5);
    expect(Math.abs(rel)).toBeGreaterThan(1e-7);
  });

  it('force kernel is the same node structure with EWALD_F, exact at a node', () => {
    const T = makeErfcTable(12, G, 100);
    const r = 3.0, x = G * r, ex = Math.exp(-x * x);
    expect(T.force(r * r)).toBeCloseTo((erfcPoly(x, ex) + EWALD_F * x * ex) / r, 6);
  });
});

/*
 * kspace_style ewald automatic kmax (ewald.ts kmaxFor). Input: cubic box 15, 216
 * random atoms, charge 0.5 each, pair coul/long 8.0,
 * units real, kspace_style ewald ACC, no gewald/kmax set. Native (black box):
 *   ACC 1e-3 -> g 0.24368433, kmax 3 3 3
 *   ACC 1e-4 -> g 0.30880405, kmax 4 4 4
 *   ACC 1e-6 -> g 0.40904245, kmax 7 7 7
 * and a 12x18x24 box, 200 atoms, q 0.4, rc 9, ACC 1e-4 -> kmax 3 4 5.
 */
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

const ewaldLog = async (text: string): Promise<string[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  await session.execute(text);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
};

const ewaldInput = (acc: string) => `
units           real
atom_style      charge
region          box block 0 15 0 15 0 15
create_box      1 box
create_atoms    1 random 216 4321 box
mass            * 1.0
set             type 1 charge 0.5
pair_style      coul/long 8.0
pair_coeff      * *
pair_modify     table 0
kspace_style    ewald ${acc}
thermo_style    custom step pe ecoul elong
run             0
`;

describe('kspace ewald: automatic g_ewald and kmax', () => {
  it('matches native for accuracy 1e-3, 1e-4, 1e-6 (rc 8, 216 atoms, q 0.5)', async () => {
    const cases: Array<[string, number, string]> = [
      ['1.0e-3', 0.24368433, '3 3 3'],
      ['1.0e-4', 0.30880405, '4 4 4'],
      ['1.0e-6', 0.40904245, '7 7 7'],
    ];
    for (const [acc, g, km] of cases) {
      const logs = (await ewaldLog(ewaldInput(acc))).join('\n');
      const gm = /G vector \(1\/distance\) = ([0-9.eE+-]+)/.exec(logs);
      const km2 = /kmax\/ewald = ([0-9 ]+)/.exec(logs);
      expect(gm && Math.abs(Number(gm[1]) / g - 1)).toBeLessThan(1e-7);
      expect(km2?.[1].trim()).toBe(km);
    }
  });
});
