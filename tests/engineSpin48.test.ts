import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * SPIN package, second batch: spin/exchange/biquadratic, spin/dmi, spin/magelec, fix setforce/spin and
 * fix langevin/spin, overlay of spin pair styles, and the refusals. Parity with native LAMMPS is in
 * tests/oracle/w48sp2_*.in.
 */

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    error = err && 'message' in err ? String(err.message) : String(e);
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { error, thermo };
};

const header = `units metal
atom_style spin
boundary f f f
atom_modify map array
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 1.2 0.7 -0.4
mass 1 55.845
set atom 1 spin/atom 1.0 0.6 0.48 0.64
set atom 2 spin/atom 1.0 -0.2 0.9 0.4
compute s all spin
`;

describe('SPIN styles, batch 2', () => {
  it('parses the biquadratic, dmi and magelec coefficients and reports the energy term', async () => {
    const bq = await run(`${header}pair_style spin/exchange/biquadratic 4.0\npair_coeff * * biquadratic 4.0 0.05 0.03 1.48 0.07 0.02 1.9 offset no\nthermo_style custom step pe\nrun 0\n`);
    expect(bq.error).toBeNull();
    expect(bq.thermo.length).toBe(1);
    const dmi = await run(`${header}pair_style spin/dmi 4.0\npair_coeff * * dmi 4.0 0.001 1.0 2.0 -0.5\nthermo_style custom step pe\nrun 0\n`);
    expect(dmi.error).toBeNull();
    const me = await run(`${header}pair_style spin/magelec 4.5\npair_coeff * * magelec 4.5 0.00109 1.0 1.0 1.0\nfix 1 all nve/spin lattice frozen\nrun 0\n`);
    expect(me.error).toBeNull();
  });

  it('refuses the biquadratic K arguments, a zero DMI direction and missing coefficients', async () => {
    const bq = await run(`${header}pair_style spin/exchange/biquadratic 4.0\npair_coeff * * biquadratic 4.0 0.05 0.03 1.48 offset no\n`);
    expect(bq.error).toContain('usage: pair_coeff I J [biquadratic]');
    const z = await run(`${header}pair_style spin/dmi 4.0\npair_coeff * * dmi 4.0 0.001 0 0 0\n`);
    expect(z.error).toContain('the direction vector must be non-zero');
    const m = await run(`${header}pair_style spin/dmi 4.0\npair_coeff * * dmi 4.0 0.001 1.0 0.0\n`);
    expect(m.error).toContain('usage: pair_coeff I J dmi');
    const kl = await run(`${header}pair_style spin/dmi 4.0\npair_coeff * * 4.0 0.001 1.0 0.0 0.0\n`);
    expect(kl.error).toContain('the dmi keyword is required');
  });

  it('refuses spin/magelec with a moving lattice and unknown spin styles', async () => {
    const n = await run(`${header}pair_style spin/bogus2 4.0\n`);
    expect(n.error).toMatch(/spin\/bogus2/);
    const mv = await run(`${header}pair_style spin/magelec 4.5\npair_coeff * * magelec 4.5 0.00109 1.0 1.0 1.0\nfix 1 all nve/spin lattice moving\nrun 0\n`);
    expect(mv.error).toContain('spin/magelec');
    expect(mv.error).toContain('not implemented');
    const b = await run(`${header}pair_style spin/bogus 4.0\n`);
    expect(b.error).toMatch(/spin\/bogus/);
  });

  it('accepts langevin/spin at a temperature above zero and refuses bad arguments', async () => {
    const t = await run(`${header}fix 2 all langevin/spin 300.0 0.01 21\n`);
    expect(t.error).toBeNull();
    const s = await run(`${header}fix 2 all langevin/spin 0.0 0.01 0\n`);
    expect(s.error).toContain('seed must be a positive integer');
    const a = await run(`${header}fix 2 all langevin/spin 0.0 0.01\n`);
    expect(a.error).toContain('usage: fix ID group-ID langevin/spin T Tdamp seed');
  });

  it('refuses variables in setforce/spin and min_style spin', async () => {
    const v = await run(`${header}fix 2 all setforce/spin v_x 0.0 0.0\n`);
    expect(v.error).toContain('variables (v_x) are not implemented');
    const m = await run(`${header}pair_style spin/exchange 4.0\npair_coeff * * exchange 4.0 0.02726 0.2171 1.841\nmin_style spin/lbfgs\nminimize 0 1e-6 10 100\n`);
    expect(m.error).toContain("min_style 'spin/lbfgs' is not supported");
  });
});
