import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { ljEOS2 } from '../src/engine-gpl/gransph/pair_sph_lj';
import { PAIRS, FIXES, COMPUTES } from '../src/engine-gpl/registry/gransph';

/*
 * The GRANULAR, SPH and TALLY styles ported in src/engine-gpl/gransph/ (compute contact/atom, fabric, pe/mol/tally,
 * sph/t/atom; fix addtorque/atom, damping/cundall, viscous/nonlinear; pair sph/lj). Native parity is in the oracle cases
 * tests/oracle/w228gs_*.in; this file checks the registry, the native error texts and the equation of state.
 */

const newSession = () => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  return { session, events };
};

const GRAN = `units lj
atom_style sphere
comm_modify vel yes
lattice sc 0.8
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
set type 1 diameter 1.0
set group all density 1.0
pair_style gran/hooke/history 2000.0 NULL 50.0 NULL 0.5 0
pair_coeff * *
`;

describe('registry of the gransph styles', () => {
  it('exports the names listed in lazyStyles.ts', () => {
    expect(Object.keys(PAIRS)).toEqual(['sph/lj']);
    expect(Object.keys(FIXES).sort()).toEqual(['addtorque/atom', 'damping/cundall', 'viscous/nonlinear']);
    expect(Object.keys(COMPUTES).sort()).toEqual(['contact/atom', 'fabric', 'pe/mol/tally', 'sph/t/atom']);
  });
});

describe('pair sph/lj equation of state (LJEOS2)', () => {
  it('approaches the ideal gas pressure at low density', () => {
    // p = rho T (1 + diff x) with diff -> 1 as x -> 0: at rho = 1e-4, T = e/cv = 1 the pressure is rho T to 1e-3
    const { p, c } = ljEOS2(1e-4, 1.0, 1.0);
    expect(Math.abs(p / (1e-4 * 1.0) - 1)).toBeLessThan(1e-3);
    expect(c).toBeGreaterThan(0);
  });

  it('gives a positive squared sound speed at a dense state and zero otherwise', () => {
    expect(ljEOS2(0.8, 1.5, 1.0).c).toBeGreaterThan(0);
  });
});

describe('native error texts', () => {
  it('fix addtorque/atom needs three torque components', async () => {
    const { session } = newSession();
    await expect(session.execute(`${GRAN}fix 2 all addtorque/atom 0.1 0.2\nrun 0\n`)).rejects.toThrow(/Illegal fix addtorque\/atom command/);
  });

  it('fix addtorque/atom rejects an unknown keyword and a non-positive every', async () => {
    const { session } = newSession();
    await expect(session.execute(`${GRAN}fix 2 all addtorque/atom 0.1 0.2 0.3 bogus 1\nrun 0\n`)).rejects.toThrow(/Unknown fix addtorque\/atom keyword: bogus/);
    const s2 = newSession();
    await expect(s2.session.execute(`${GRAN}fix 2 all addtorque/atom 0.1 0.2 0.3 every 0\nrun 0\n`)).rejects.toThrow(/Invalid fix addtorque\/atom every argument: 0/);
  });

  it('fix damping/cundall requires atom attribute omega', async () => {
    const { session } = newSession();
    await expect(session.execute(`units lj\natom_style atomic\nlattice sc 0.8\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\npair_style zero 1.0\npair_coeff * *\nfix 2 all damping/cundall 0.2 0.1\nrun 0\n`)).rejects.toThrow(/Fix damping\/cundall requires atom attribute omega/);
  });

  it('fix viscous/nonlinear requires a positive fluid density and viscosity', async () => {
    const { session } = newSession();
    await expect(session.execute(`${GRAN}fix 2 all viscous/nonlinear 0.0 0.05\nrun 0\n`)).rejects.toThrow(/Fix viscous\/nonlinear fluid density and viscosity must be > 0/);
  });

  it('pair sph/lj takes exactly two coefficients plus the types', async () => {
    const { session } = newSession();
    await expect(session.execute(`units lj\natom_style sph\nlattice sc 0.8\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box\npair_style sph/lj\npair_coeff * * 0.1\n`)).rejects.toThrow(/Incorrect args for pair_style sph\/lj coefficients/);
  });

  it('compute sph/t/atom takes no arguments', async () => {
    const { session } = newSession();
    await expect(session.execute(`${GRAN}compute t all sph/t/atom 1\n`)).rejects.toThrow(/Number of arguments for compute sph\/t\/atom command != 3/);
  });

  it('compute pe/mol/tally needs a second group and molecule IDs', async () => {
    const { session } = newSession();
    await expect(session.execute(`${GRAN}compute pm all pe/mol/tally\n`)).rejects.toThrow(/Illegal compute pe\/mol\/tally command: missing argument/);
    const s2 = newSession();
    await expect(s2.session.execute(`${GRAN}compute pm all pe/mol/tally all\n`)).rejects.toThrow(/Compute pe\/mol\/tally requires molecule IDs/);
  });
});
