import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * pair_style hybrid family: the documented rules and error paths. Agreement
 * with native LAMMPS is covered by the hybrid_* oracle cases.
 */

const run = async (text: string) => {
  const rows: ThermoRow[] = [];
  const session = new Session({ emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); }, writeFile: () => {} });
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  return { rows, error };
};

const box = (pair: string, extra = '') => `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 3 0 3 0 3
create_box      2 box
create_atoms    1 box
group           odd id 1:108:2
set             group odd type 2
mass            * 1.0
variable        dx atom 0.05*sin(1.3*x+0.7*y)
displace_atoms  all move v_dx 0 0 units box
${pair}
${extra}
thermo_style    custom step pe press
run             0
`;

describe('pair_style hybrid: equivalences from the docs', () => {
  it('hybrid/overlay with one lj/cut equals plain lj/cut (pair_hybrid.html example)', async () => {
    const plain = await run(box('pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 2 2 1.5 0.8'));
    const over = await run(box('pair_style hybrid/overlay lj/cut 2.5\npair_coeff * * lj/cut 1.0 1.0\npair_coeff 2 2 lj/cut 1.5 0.8'));
    expect(plain.error).toBeNull();
    expect(over.error).toBeNull();
    expect(over.rows[0].pe).toBeCloseTo(plain.rows[0].pe, 12);
    expect(over.rows[0].press).toBeCloseTo(plain.rows[0].press, 10);
  });

  it('hybrid mixes I,J from the sub-style shared by I,I and J,J', async () => {
    const plain = await run(box('pair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\npair_coeff 2 2 1.5 0.8'));
    const hyb = await run(box('pair_style hybrid lj/cut 2.5\npair_coeff 1 1 lj/cut 1.0 1.0\npair_coeff 2 2 lj/cut 1.5 0.8'));
    expect(hyb.error).toBeNull();
    expect(hyb.rows[0].pe).toBeCloseTo(plain.rows[0].pe, 12);
  });

  it('none removes a type pair; hybrid/scaled with factors 1 and 0 equals the first sub-style', async () => {
    const one = await run(box('pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0'));
    const scaled = await run(box('pair_style hybrid/scaled 1.0 lj/cut 2.5 0.0 lj/cut 2.5\npair_coeff * * lj/cut 1 1.0 1.0\npair_coeff * * lj/cut 2 3.0 1.0'));
    expect(scaled.error).toBeNull();
    expect(scaled.rows[0].pe).toBeCloseTo(one.rows[0].pe, 12);
    const none = await run(box('pair_style hybrid lj/cut 2.5\npair_coeff * * lj/cut 1.0 1.0\npair_coeff 1 2 none'));
    const only = await run(box('pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0', 'neigh_modify exclude type 1 2'));
    expect(none.error).toBeNull();
    expect(none.rows[0].pe).toBeCloseTo(only.rows[0].pe, 12);
  });
});

describe('pair_style hybrid: errors', () => {
  const cases: [string, string, RegExp][] = [
    ['unknown sub-style', 'pair_style hybrid lj/bogus 2.5', /sub-style 'lj\/bogus' is not supported/],
    ['nested hybrid', 'pair_style hybrid hybrid lj/cut 2.5', /cannot be a sub-style/],
    ['instance number required', 'pair_style hybrid lj/cut 2.5 lj/cut 2.0\npair_coeff * * lj/cut 1.0 1.0', /listed 2 times; give its number/],
    ['unassigned pair, no common sub-style', 'pair_style hybrid lj/cut 2.5 lj/cut 2.0\npair_coeff 1 1 lj/cut 1 1.0 1.0\npair_coeff 2 2 lj/cut 2 1.0 1.0', /all pair coeffs are not set \(pair 1 2/],
    ['overlay mixing needs a single sub-style', 'pair_style hybrid/overlay lj/cut 2.5 lj/cut 2.0\npair_coeff 1 1 lj/cut 1 1.0 1.0\npair_coeff 1 1 lj/cut 2 1.0 1.0\npair_coeff 2 2 lj/cut 1 1.0 1.0', /all pair coeffs are not set \(pair 1 2/],
    ['unused sub-style', 'pair_style hybrid lj/cut 2.5 lj/cut 2.0\npair_coeff * * lj/cut 1 1.0 1.0', /sub-style lj\/cut is not used/],
    ['molecular needs two sub-styles', 'pair_style hybrid/molecular lj/cut 2.5', /only two sub-styles/],
    ['pair keyword not first', 'pair_style hybrid lj/cut 2.5\npair_coeff * * lj/cut 1.0 1.0\npair_modify shift yes pair lj/cut', /must appear first/],
    ['special without pair', 'pair_style hybrid lj/cut 2.5\npair_coeff * * lj/cut 1.0 1.0\npair_modify special lj 0 0 0', /directly after the pair keyword/],
    ['pair keyword on a plain style', 'pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_modify pair lj/cut shift yes', /only be used with the hybrid/],
    ['shift on one sub-style, tail on another', 'pair_style hybrid lj/cut 2.5 lj/cut 2.0\npair_coeff * * lj/cut 1 1.0 1.0\npair_coeff 2 2 lj/cut 2 1.0 1.0\npair_modify pair lj/cut 1 tail yes\npair_modify pair lj/cut 2 shift yes', /both pair_modify shift and tail/],
    ['special incompatible with the global weights', 'pair_style hybrid lj/cut 2.5\npair_coeff * * lj/cut 1.0 1.0\npair_modify pair lj/cut special lj 0.5 0 0', /special lj 1-2 setting .* incompatible/],
  ];
  for (const [what, pair, re] of cases) {
    it(what, async () => {
      const { error } = await run(box(pair));
      expect(error, 'expected an error').not.toBeNull();
      expect(error!.message).toMatch(re);
    });
  }
});

describe('pair_style hybrid: a sub-style the engine lacks is named', () => {
  it('reports the swallowed word as a possible pair style (examples/atm)', async () => {
    const { error } = await run(box('pair_style hybrid/overlay lj/cut 4.5 atm 4.5 2.5\npair_coeff * * lj/cut 1.0 1.0'));
    expect(error?.message).toMatch(/sub-style lj\/cut 4.5 atm 4.5 2.5/);
    expect(error?.message).toMatch(/If 'atm' is meant as a pair style, the browser engine does not support it/);
  });
});
