import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { BONDS, ANGLES, PAIRS } from '../src/engine-gpl/registry/miscx';
import { BondQuartic } from '../src/engine-gpl/miscx/bond_quartic';
import { BondSpecial } from '../src/engine-gpl/miscx/bond_special';
import { AngleDipole } from '../src/engine-gpl/miscx/angle_dipole';
import { PairList } from '../src/engine-gpl/miscx/pair_list';
import { PairDSMC } from '../src/engine-gpl/miscx/pair_dsmc';

/*
 * The MISC, MOLECULE, DIPOLE and MC styles of src/engine-gpl/miscx: the registry names, the native error messages of
 * their argument and setting checks, and a few physical invariants (broken quartic bonds leave the bond list at type 0,
 * a bond special with unit weights is the plain pair interaction, dsmc collisions conserve the kinetic energy).
 * The energies and forces against native LAMMPS are the oracle cases tests/oracle/w235mx_*.in.
 */

/** Runs a script on a fresh session with the given data files; returns the session and the error, if any. */
const run = async (script: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error: Error | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e as Error;
  }
  return { session, error, events };
};

const thermo = (events: EngineEvent[]): ThermoRow[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);

const CHAIN = `units lj
atom_style bond
boundary f f f
region box block -20 20 -20 20 -20 20
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 0.97 0.0 0.0
create_atoms 1 single 1.94 0.0 0.0
create_atoms 1 single 2.91 0.0 0.0
mass 1 1.0
create_bonds single/bond 1 1 2
create_bonds single/bond 1 2 3
create_bonds single/bond 1 3 4
group left id 1
group right id 4
velocity all set 0.0 0.0 0.0 units box
velocity left set -5.0 0.0 0.0 units box
velocity right set 5.0 0.0 0.0 units box
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
`;

describe('registry of the miscx styles', () => {
  it('names the bond, angle and pair styles of the four packages', () => {
    expect(Object.keys(BONDS).sort()).toEqual(['quartic', 'special']);
    expect(Object.keys(ANGLES)).toEqual(['dipole']);
    expect(Object.keys(PAIRS).sort()).toEqual(['dsmc', 'list']);
    expect(new BondQuartic().name).toBe('quartic');
    expect(new BondSpecial().name).toBe('special');
    expect(new AngleDipole().name).toBe('dipole');
    expect(new PairList().name).toBe('list');
    expect(new PairDSMC().name).toBe('dsmc');
  });
});

describe('bond_style quartic', () => {
  it('breaks a bond stretched past Rc: its type becomes 0 and the rest of the chain keeps its topology', async () => {
    const { session, error } = await run(`${CHAIN}bond_style quartic
bond_coeff 1 1200 -0.55 0.25 1.3 34.6878
special_bonds lj 1.0 1.0 1.0 coul 1.0 1.0 1.0
fix 1 all nve
timestep 0.002
run 600
`);
    expect(error).toBeNull();
    const types = Array.from(session.sys.state.topo.bonds.type.subarray(0, session.sys.state.topo.bonds.n));
    // the chain has three bonds; the fast end atoms stretch the chain past Rc, so at least one bond is broken
    expect(types).toHaveLength(3);
    expect(types.filter((t) => t === 0).length).toBeGreaterThan(0);
  });

  it('requires special_bonds 1,1,1 (native message)', async () => {
    const { error } = await run(`${CHAIN}bond_style quartic
bond_coeff 1 1200 -0.55 0.25 1.3 34.6878
special_bonds lj 0.0 1.0 1.0
run 0
`);
    expect(error?.message).toContain('Bond style quartic requires special_bonds = 1,1,1');
  });

  it('refuses a coefficient line that is not "K B1 B2 Rc U0" (native message)', async () => {
    const { error } = await run(`${CHAIN}bond_style quartic
bond_coeff 1 1200 -0.55 0.25 1.3
`);
    expect(error?.message).toContain('Incorrect args for bond coefficients');
  });
});

describe('bond_style special', () => {
  it('refuses a 1-2 special weight that is not 0 (native message)', async () => {
    const { error } = await run(`${CHAIN}special_bonds lj 0.5 1.0 1.0 coul 0.0 1.0 1.0
bond_style special
bond_coeff 1 0.5 0.5
run 0
`);
    expect(error?.message).toContain('Invalid 1-2 setting for bond style special.');
  });

  it('with unit factors and 1-2 weights 0 gives the same energy as the plain pair with 1-2 weights 1', async () => {
    const plain = await run(`${CHAIN}special_bonds lj 1.0 1.0 1.0 coul 1.0 1.0 1.0
thermo_style custom step pe
thermo 1
run 0
`);
    const special = await run(`${CHAIN}special_bonds lj 0.0 1.0 1.0 coul 0.0 1.0 1.0
bond_style special
bond_coeff 1 1.0 1.0
thermo_style custom step pe
thermo 1
run 0
`);
    expect(plain.error).toBeNull();
    expect(special.error).toBeNull();
    const pe0 = thermo(plain.events)[0].pe, pe1 = thermo(special.events)[0].pe;
    expect(Math.abs(pe1 - pe0)).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(pe0)));
  });
});

describe('angle_style dipole', () => {
  it('needs the atom attributes mu and torque (native message)', async () => {
    const { error } = await run(`units lj
atom_style hybrid dipole angle
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 angle/types 1 extra/angle/per/atom 1
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 1.0 0.0 0.0
create_atoms 1 single 2.0 0.0 0.0
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
angle_style dipole
angle_coeff 1 2.0 60.0
run 0
`);
    expect(error?.message).toContain('Angle style dipole requires atom attributes mu and torque');
  });
});

describe('pair_style list', () => {
  const LIST = `1 2 harmonic 40.0 1.0
2 3 lj126 0.5 1.0
`;

  it('reads its pairs from the list file and refuses a missing file', async () => {
    const ok = await run(`units lj
atom_style atomic
atom_modify map array
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 1.0 0.0 0.0
create_atoms 1 single 1.5 0.5 0.0
mass 1 1.0
pair_style list pairs.txt 2.5
pair_coeff * *
run 0
`, { 'pairs.txt': LIST });
    expect(ok.error).toBeNull();
    const bad = await run(`units lj
atom_style atomic
atom_modify map array
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0
mass 1 1.0
pair_style list absent.txt 2.5
pair_coeff * *
run 0
`);
    expect(bad.error?.message).toContain('Error opening pair list coeffs file absent.txt');
  });

  it('checks the atom IDs of the list (native message) and refuses an unknown keyword', async () => {
    const badId = await run(`units lj
atom_style atomic
atom_modify map array
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 1.0 0.0 0.0
mass 1 1.0
pair_style list pairs.txt 2.5
pair_coeff * *
run 0
`, { 'pairs.txt': '1 9 harmonic 1.0 1.0\n' });
    expect(badId.error?.message).toContain('Invalid pair list atom ID 9');
    const keyword = await run(`units lj
atom_style atomic
pair_style list pairs.txt 2.5 bogus
`, { 'pairs.txt': LIST });
    expect(keyword.error?.message).toContain('Unknown pair style list keyword: bogus');
  });
});

describe('pair_style dsmc', () => {
  const GAS = `units lj
atom_style atomic
lattice sc 0.1
region box block 0 8 0 8 0 8
create_box 2 box
create_atoms 1 box
set group all type/fraction 2 0.5 4321
mass 1 1.0
mass 2 2.0
velocity all create 1.0 4242
fix 1 all nve
timestep 0.005
thermo_style custom step ke
thermo 10
`;

  it('conserves the kinetic energy of the collisions while changing the velocities', async () => {
    const { error, events } = await run(`${GAS}pair_style dsmc 2.5 34387 2000 1.0 10 20
pair_coeff 1 1 1.0
pair_coeff 1 2 0.8
pair_coeff 2 2 1.2
run 60
`);
    expect(error).toBeNull();
    const rows = thermo(events);
    expect(rows.length).toBeGreaterThan(2);
    for (const r of rows) expect(Math.abs(r.ke - rows[0].ke)).toBeLessThanOrEqual(1e-12 * Math.abs(rows[0].ke));
  });

  it('refuses a wrong argument count and a zero seed (native messages)', async () => {
    const count = await run(`${GAS}pair_style dsmc 2.5 34387 10 1.0 10
`);
    expect(count.error?.message).toContain('Illegal pair_style command');
    const seed = await run(`${GAS}pair_style dsmc 2.5 0 10 1.0 10 20
`);
    expect(seed.error?.message).toContain('Illegal pair_style command');
  });
});
