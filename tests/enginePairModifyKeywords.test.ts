import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * pair_modify and special_bonds keywords of the audit WP4 (docs: pair_modify.html, special_bonds.html):
 * compute, nofdotr, tabinner, compute/tally and one/five. The numbers come from native LAMMPS runs, see the
 * tests/oracle/w88pair_*.in cases; these tests pin the accept/refuse decisions and the error texts.
 */

const run = async (text: string) => {
  const rows: ThermoRow[] = [];
  const session = new Session({
    emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); },
    writeFile: () => {},
  });
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  return { rows, error };
};

const fcc = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 3 0 3 0 3
create_box      2 box
create_atoms    1 box
mass            * 1.0
set             group all type/fraction 2 0.3 1234
`;

describe('pair_modify compute, nofdotr, tabinner', () => {
  it('compute no turns the pair term off (pe 0, evdwl 0)', async () => {
    const { rows, error } = await run(`${fcc}
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
pair_modify compute no
thermo_style custom step pe evdwl press
run 0`);
    expect(error).toBeNull();
    expect(rows[0].pe).toBe(0);
    expect(rows[0].evdwl).toBe(0);
  });

  it('nofdotr is accepted and changes no energy', async () => {
    const base = await run(`${fcc}
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
thermo_style custom step pe press
run 0`);
    const nofdotr = await run(`${fcc}
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
pair_modify nofdotr
thermo_style custom step pe press
run 0`);
    expect(nofdotr.error).toBeNull();
    expect(nofdotr.rows[0].pe).toBe(base.rows[0].pe);
  });

  it('tabinner accepts other cutoffs and rejects a non-positive one', async () => {
    const ok = await run(`${fcc}
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
pair_modify table 0 tabinner 1.5
thermo_style custom step pe
run 0`);
    expect(ok.error).toBeNull();
    const bad = await run(`${fcc}
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
pair_modify tabinner 0
run 0`);
    expect(bad.error?.message).toMatch(/tabinner must be > 0/);
  });
});

describe('pair_modify pair (hybrid) and special_bonds keywords', () => {
  it('pair X compute/tally no is refused by name (not silently ignored)', async () => {
    const { error } = await run(`${fcc}
pair_style hybrid lj/cut 2.5 morse 2.5
pair_coeff 1 1 lj/cut 1.0 1.0
pair_coeff 2 2 morse 1.0 1.2 1.0
pair_modify pair lj/cut compute/tally no
run 0`);
    expect(error?.message).toMatch(/pair_modify pair compute\/tally no is not supported/);
  });

  it('special_bonds one/five yes is refused (native: the atom style must support the 1-5 list)', async () => {
    const { error } = await run(`${fcc}
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
special_bonds one/five yes
run 0`);
    expect(error?.message).toMatch(/one\/five yes is not supported/);
  });
});
