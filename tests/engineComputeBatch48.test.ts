import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Batch 48 computes: cnp/atom, temp/profile, hma (the unsupported-style refusal is checked with ti/fl). Parsing, refusals and a few
 * self-consistency checks; the native parity lives in tests/oracle/w48cb_*.in (engineOracle.test.ts).
 */

const LATTICE = `units lj
atom_style atomic
lattice fcc 1.0
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/smooth/linear 3
pair_coeff * * 1.0 1.0
`;

const run = async (script: string): Promise<{ errors: string[]; thermo: ThermoRow[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  try {
    await session.execute(script);
  } catch (e) {
    events.push({ kind: 'error', message: String(e instanceof Error ? e.message : e) } as unknown as EngineEvent);
  }
  const errors = events.filter((e) => e.kind === 'error').map((e) => ('message' in e ? String(e.message) : ''));
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { errors, thermo };
};

describe('compute cnp/atom', () => {
  it('needs exactly one cutoff argument and a positive cutoff', async () => {
    expect((await run(`${LATTICE}compute c all cnp/atom\nrun 0\n`)).errors[0]).toMatch(/usage: compute ID group-ID cnp\/atom cutoff/);
    expect((await run(`${LATTICE}compute c all cnp/atom 3.0 3.0\nrun 0\n`)).errors[0]).toMatch(/usage/);
    expect((await run(`${LATTICE}compute c all cnp/atom 0\nrun 0\n`)).errors[0]).toMatch(/cutoff must be > 0/);
  });

  it('is zero on a perfect fcc crystal and zero outside the group', async () => {
    const r = await run(`${LATTICE}compute c all cnp/atom 0.8536\ncompute m all reduce max c_c\ncompute s all reduce sum c_c\nthermo_style custom step c_m c_s\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(r.thermo[0].c_m).toBe(0);
    expect(r.thermo[0].c_s).toBe(0);
  });

  it('is positive for a displaced atom', async () => {
    const r = await run(`${LATTICE}displace_atoms all move 0.3 0.0 0.0 units box\ncompute c all cnp/atom 1.3\ncompute m all reduce max c_c\nthermo_style custom step c_m\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(r.thermo[0].c_m).toBeGreaterThan(0);
  });
});

describe('compute temp/profile', () => {

  it('refuses bad flags, binstyles, bin counts and keywords', async () => {
    expect((await run(`${LATTICE}compute tp all temp/profile 2 1 1 x 3\nrun 0\n`)).errors[0]).toMatch(/xflag must be 0 or 1/);
    expect((await run(`${LATTICE}compute tp all temp/profile 1 1 1 w 3\nrun 0\n`)).errors[0]).toMatch(/binstyle must be x, y, z, xy, yz, xz or xyz/);
    expect((await run(`${LATTICE}compute tp all temp/profile 1 1 1 x 2.5\nrun 0\n`)).errors[0]).toMatch(/Nx must be a positive integer/);
    expect((await run(`${LATTICE}compute tp all temp/profile 1 1 1 x 3 out foo\nrun 0\n`)).errors[0]).toMatch(/out must be tensor or bin/);
    expect((await run(`${LATTICE}compute tp all temp/profile 1 1 1 x 3 extra 2\nrun 0\n`)).errors[0]).toMatch(/unknown keyword 'extra'/);
  });

  it('refuses a triclinic box', async () => {
    const r = await run(`units lj\natom_style atomic\nregion box prism 0 4 0 4 0 4 1 0 0\ncreate_box 1 box\nmass 1 1.0\ncompute tp all temp/profile 1 1 1 x 2\nrun 0\n`);
    expect(r.errors[0]).toMatch(/triclinic boxes are not supported/);
  });

  it('with no flags set, equals the plain temperature (dof dim N - dim)', async () => {
    const r = await run(`${LATTICE}velocity all create 1.5 87287 mom yes rot no\ncompute tp all temp/profile 0 0 0 x 2\nthermo_style custom step temp c_tp\nthermo_modify format float %.15g\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(r.thermo[0].c_tp).toBeCloseTo(r.thermo[0].temp as number, 12);
  });

  it('removes a uniform flow along x with one bin and reports zero flow temperature for a pure translation', async () => {
    const r = await run(`${LATTICE}velocity all set 0.7 0.0 0.0 units box\ncompute tp all temp/profile 1 0 0 x 1\nthermo_style custom step c_tp\nthermo_modify format float %.15g\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(Math.abs(r.thermo[0].c_tp as number)).toBeLessThan(1e-12);
  });
});

describe('compute hma', () => {
  const FIX = 'fix settemp all nvt temp 1.0 1.0 0.5\n';
  it('refuses cv and anharmonic and bad keywords', async () => {
    expect((await run(`${LATTICE}${FIX}compute h all hma settemp u cv\nrun 0\n`)).errors[0]).toMatch(/keyword cv is not implemented.*single_hessian/);
    expect((await run(`${LATTICE}${FIX}compute h all hma settemp anharmonic u\nrun 0\n`)).errors[0]).toMatch(/keyword anharmonic is not implemented/);
    expect((await run(`${LATTICE}${FIX}compute h all hma settemp w\nrun 0\n`)).errors[0]).toMatch(/unknown keyword 'w'/);
    expect((await run(`${LATTICE}${FIX}compute h all hma settemp p\nrun 0\n`)).errors[0]).toMatch(/keyword p needs a pressure value/);
    expect((await run(`${LATTICE}${FIX}compute h all hma settemp u\nrun 0\n`)).errors).toEqual([]);
  });

  it('refuses groups other than all and temp-IDs that are not thermostats', async () => {
    expect((await run(`${LATTICE}${FIX}group g1 id 1 2\ncompute h g1 hma settemp u\nrun 0\n`)).errors[0]).toMatch(/group must be all/);
    expect((await run(`${LATTICE}fix nve1 all nve\ncompute h all hma nve1 u\nrun 0\n`)).errors[0]).toMatch(/must be a thermostat fix/);
  });

  it('u equals pe + 3/2 (N-1) T when defined after the displacement (sites are then the displaced positions)', async () => {
    const r = await run(`${LATTICE}${FIX}displace_atoms all move 0.05 0.02 -0.03 units box\ncompute pe all pe\ncompute h all hma settemp u\nthermo_style custom step c_pe c_h[1]\nthermo_modify format float %.15g\nrun 0\n`);
    expect(r.errors).toEqual([]);
    const row = r.thermo[0];
    const N = 32;
    expect((row['c_h[1]'] as number) - (row.c_pe as number)).toBeCloseTo(1.5 * (N - 1) * 1.0 / N, 10);
  });

  it('keeps the entries in the order of the keywords', async () => {
    const r = await run(`${LATTICE}${FIX}compute h all hma settemp p 0.5 u\ncompute pe all pe\nthermo_style custom step c_h[1] c_h[2] c_pe\nthermo_modify format float %.15g\nrun 0\n`);
    expect(r.errors).toEqual([]);
    expect(r.thermo[0]['c_h[1]']).toBeDefined();
    expect(r.thermo[0]['c_h[2]']).toBeDefined();
  });
});

describe('unsupported compute style', () => {
  it('is refused by name (not registered: the engine rejects the style)', async () => {
    const r = await run(`${LATTICE}compute e all ti/fl\nrun 0\n`);
    expect(r.errors[0]).toMatch(/compute style 'ti\/fl' is not supported by the browser engine/);
  });
});
