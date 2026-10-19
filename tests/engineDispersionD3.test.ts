import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * pair_style dispersion/d3 (GPL port of src/EXTRA-PAIR/pair_dispersion_d3.cpp, src/engine-gpl/d3): the native argument
 * checks and messages. Native agreement of energies, forces and pressure for the four damping functions is in
 * tests/oracle/w250d3_*.in.
 */

const run = async (input: string): Promise<string | undefined> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: string | undefined;
  try {
    await s.execute(input);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) error = String(err.message);
  return error;
};

const BASE = 'units metal\nlattice diamond 3.57\nregion b block 0 1 0 1 0 1\ncreate_box 2 b\ncreate_atoms 1 box\nmass * 12.0\n';

describe('pair_style dispersion/d3', () => {
  it('refuses settings as native settings() does (measured with native LAMMPS: the same messages)', async () => {
    expect(await run(`${BASE}pair_style dispersion/d3 bj pbe 9.0\n`)).toMatch(/Pair style dispersion\/d3 needs 4 arguments/);
    expect(await run(`${BASE}pair_style dispersion/d3 foo pbe 9.0 6.0\n`)).toMatch(/Unknown damping type foo for pair style dispersion\/d3/);
    // only the zero/original branch lowercases the functional name
    expect(await run(`${BASE}pair_style dispersion/d3 zerom PBE0 9.0 6.0\n`)).toMatch(/Functional PBE0 not supported with zerom damping function/);
    expect(await run(`${BASE}pair_style dispersion/d3 bj nosuch 9.0 6.0\n`)).toMatch(/Functional nosuch not supported with bj damping function/);
    expect(await run(`${BASE}pair_style dispersion/d3 zero NoSuch 9.0 6.0\n`)).toMatch(/Functional nosuch not supported with original damping function/);
    expect(await run('units lj\nlattice fcc 0.8\nregion b block 0 1 0 1 0 1\ncreate_box 1 b\npair_style dispersion/d3 bj pbe 9.0 6.0\n'))
      .toMatch(/Pair style dispersion\/d3 requires metal units/);
  });

  it('refuses pair_coeff as native coeff() does', async () => {
    const ps = 'pair_style dispersion/d3 bj pbe 9.0 6.0\n';
    expect(await run(`${BASE}${ps}pair_coeff * * C\n`)).toMatch(/Pair_coeff \* \* needs: element1 element2/);
    // the element key is lowercased and a one-letter key padded with a blank, as native prints it
    expect(await run(`${BASE}${ps}pair_coeff * * C Q\n`)).toMatch(/Element q {2}not supported/);
    expect(await run(`${BASE}${ps}pair_coeff * * C Xyz\n`)).toMatch(/Element xyz not supported/);
  });

  it('runs with every damping function and an accepted functional', async () => {
    for (const s of ['zero pbe', 'zerom pbe', 'bj pbe', 'bjm pbe', 'original b-lyp']) {
      expect(await run(`${BASE}pair_style dispersion/d3 ${s} 9.0 6.0\npair_coeff * * C O\nrun 0\n`), s).toBeUndefined();
    }
  });
});
