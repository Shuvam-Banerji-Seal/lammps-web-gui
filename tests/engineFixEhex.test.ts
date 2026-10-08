import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix heat (docs.lammps.org/fix_heat.html) and fix ehex
 * (docs.lammps.org/fix_ehex.html): argument errors, the documented energy
 * bookkeeping (the group's kinetic energy rises by F*dt*N at each application
 * when no forces act), the global scalar, and the StyleError for keywords the
 * engine does not implement. The numerical agreement with native LAMMPS is in
 * tests/oracle/w8ehex_*.in.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: () => {},
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { thermo, error };
};

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** A 32-atom simple-cubic LJ gas with zero pair forces, so only the thermostat acts. */
const GAS = `
units           lj
atom_style      atomic
lattice         sc 0.5
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
pair_style      zero 2.5
pair_coeff      * *
velocity        all create 1.0 9876 mom no rot no
timestep        0.004
fix             1 all nve
`;

describe('fix heat / fix ehex argument checks', () => {
  const cases: [string, RegExp][] = [
    ['fix 2 all heat', /usage|missing|heat/],
    ['fix 2 all heat 0 0.1', /positive integer/],
    ['fix 2 all heat 1.5 0.1', /positive integer/],
    ['fix 2 all heat 1 notanumber', /expected a number for eflux/],
    ['fix 2 all heat 1 0.1 bogus 3', /unknown keyword 'bogus'/],
    ['fix 2 all heat 1 0.1 region nosuch', /region ID 'nosuch' does not exist/],
    ['fix 2 all heat 1 0.1 region', /needs a region-ID/],
    ['fix 2 all ehex 1', /usage|missing|ehex/],
    ['fix 2 all ehex 1 notanumber', /expected a number for eflux/],
    ['fix 2 all ehex 1 0.1 foo', /unknown keyword 'foo'/],
    ['fix 2 all ehex 1 0.1 com', /together with the keyword 'constrain'/],
  ];
  for (const [line, re] of cases) {
    it(`rejects: ${line}`, async () => {
      const { error } = await runScript(`${GAS}\nregion slab block INF INF INF INF 0 0.5 units box\n${line}\nrun 0\n`);
      expect(error, `expected an error for '${line}'`).not.toBeNull();
      expect(errorText(error)).toMatch(re);
    });
  }

  it('names constrain as not implemented (needs fix shake/rattle)', async () => {
    const { error } = await runScript(`${GAS}\nfix 2 all ehex 1 0.1 constrain\nrun 0\n`);
    expect(errorText(error)).toMatch(/constrain.*not implemented/);
  });

  it('names com as not implemented', async () => {
    const { error } = await runScript(`${GAS}\nfix 2 all ehex 1 0.1 constrain com\nrun 0\n`);
    expect(errorText(error)).toMatch(/constrain/);
  });

  it('rejects an atom-style variable as eflux', async () => {
    const { error } = await runScript(`${GAS}\nvariable q atom x\nfix 2 all heat 1 v_q\nrun 0\n`);
    expect(errorText(error)).toMatch(/atom-style variable eflux is not implemented/);
  });

  it('accepts an equal-style variable as eflux for heat', async () => {
    const { error } = await runScript(`${GAS}\nvariable q equal 0.1\nfix 2 all heat 1 v_q\nrun 1\n`);
    expect(error).toBeNull();
  });

  it('fails at run time when the group kinetic energy would go negative', async () => {
    const { error } = await runScript(`${GAS}\nfix 2 all heat 1 -1000000\nthermo 1\nrun 2\n`);
    expect(errorText(error)).toMatch(/kinetic energy went negative/);
  });
});

describe('fix heat / fix ehex energy bookkeeping', () => {
  // thermo ke is the total kinetic energy with norm no
  // the engine normalises thermo ke per atom in lj units even with norm no; 8 atoms here
  const NA = 8;
  const keSeries = (rows: ThermoRow[]) => rows.map((r) => Number(r.ke) * NA);

  it('adds F*dt*N to the group KE at every application (no forces)', async () => {
    const F = 0.3, N = 2, dt = 0.004;
    const { thermo, error } = await runScript(`${GAS}
thermo_modify   norm no
thermo_style    custom step ke f_2
thermo          1
fix             2 all heat ${N} ${F}
run             8
`);
    expect(error).toBeNull();
    const ke = keSeries(thermo);
    for (let k = 1; k < ke.length; k++) {
      const step = thermo[k].step;
      const delta = ke[k] - ke[k - 1];
      if (step % N === 0) expect(delta, `step ${step}`).toBeCloseTo(F * dt * N, 10);
      else expect(delta, `step ${step}`).toBeCloseTo(0, 10);
    }
  });

  it('the heat scalar is sqrt(1 + N F dt / K) with K about the COM (zero COM here) and is 1 before the first application', async () => {
    const { thermo, error } = await runScript(`${GAS.replace('mom no', 'mom yes')}
thermo_modify   norm no
thermo_style    custom step ke f_2
thermo          1
fix             2 all heat 1 0.5
run             2
`);
    expect(error).toBeNull();
    expect(thermo[0].f_2).toBe(1);
    const k0 = Number(thermo[0].ke) * 8, k1 = Number(thermo[1].ke) * 8;
    // K before the scaling is the KE after the step minus the energy added
    const kPre = k1 - 0.5 * 0.004;
    expect(Number(thermo[1].f_2)).toBeCloseTo(Math.sqrt(1 + (0.5 * 0.004) / kPre), 12);
    expect(k0).toBeGreaterThan(0);
  });

  it('a negative flux lowers the scalar below one', async () => {
    const { thermo, error } = await runScript(`${GAS}
thermo_style    custom step f_2
thermo          1
fix             2 all heat 1 -0.2
run             1
`);
    expect(error).toBeNull();
    expect(Number(thermo[1].f_2)).toBeLessThan(1);
  });

  it('region restricts the energy exchange to the atoms inside it', async () => {
    // the slab holds half of the 32 atoms (z = 0 and 0.5 layers of a 2x2x2 sc lattice)
    const { thermo, error } = await runScript(`${GAS}
thermo_modify   norm no
region          slab block INF INF INF INF 0 0.25 units box
thermo_style    custom step ke f_2
thermo          1
fix             2 all heat 1 0.4 region slab
run             2
`);
    expect(error).toBeNull();
    const dke = (Number(thermo[1].ke) - Number(thermo[0].ke)) * 8;
    expect(dke).toBeCloseTo(0.4 * 0.004, 10);
    expect(Number(thermo[1].f_2)).toBeGreaterThan(1);
  });

  it('ehex with the hex keyword reproduces fix heat exactly', async () => {
    const body = `${GAS}
thermo_modify   norm no
thermo_style    custom step ke f_2
thermo          1
run             3
`;
    const a = await runScript(body.replace('run             3', `fix 2 all heat 1 0.7\nrun 3`));
    const b = await runScript(body.replace('run             3', `fix 2 all ehex 1 0.7 hex\nrun 3`));
    expect(a.error).toBeNull();
    expect(b.error).toBeNull();
    for (let k = 0; k < a.thermo.length; k++) {
      expect(Number(b.thermo[k].ke)).toBeCloseTo(Number(a.thermo[k].ke), 14);
      expect(Number(b.thermo[k].f_2)).toBeCloseTo(Number(a.thermo[k].f_2), 14);
    }
  });

  it('ehex leaves the kinetic energy history of heat unchanged (only positions carry the correction)', async () => {
    const body = `${GAS}
thermo_modify   norm no
thermo_style    custom step ke
thermo          1
run             3
`;
    const heat = await runScript(body.replace('run             3', `fix 2 all heat 1 0.7\nrun 3`));
    const ehex = await runScript(body.replace('run             3', `fix 2 all ehex 1 0.7\nrun 3`));
    expect(heat.error).toBeNull();
    expect(ehex.error).toBeNull();
    for (let k = 0; k < heat.thermo.length; k++) {
      expect(Number(ehex.thermo[k].ke)).toBeCloseTo(Number(heat.thermo[k].ke), 14);
    }
  });
});
