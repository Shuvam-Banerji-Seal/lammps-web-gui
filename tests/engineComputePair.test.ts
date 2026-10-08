import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * compute pair (docs.lammps.org/compute_pair.html): the global scalar (epair, evdwl or ecoul of
 * the pair style or of one hybrid sub-style) and the hbond/dreiding vector. The energies checked
 * here are internal identities plus the StyleError paths; native parity is in tests/oracle/w15cpair_*.
 */

const SYSTEM = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
mass * 1.0
region r1 block 0 2.1 0 5.1 0 5.1
region r2 block 2.1 5.1 0 5.1 0 5.1
create_atoms 1 region r1
create_atoms 2 region r2
velocity all set 0.0 0.0 0.0 units box
`;

const HYBRID = `pair_style hybrid/overlay lj/cut 2.5 morse 2.0 morse 2.5
pair_coeff * * lj/cut 1.0 1.0 2.5
pair_coeff 1 2 morse 1 1.0 1.5 1.2 2.0
pair_coeff 2 2 morse 2 0.7 1.8 1.3 2.0
pair_coeff 1 1 morse 2 0.3 2.0 1.0 2.0
`;

const PLAIN = `pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
`;

const HBOND = readFileSync(join(__dirname, 'oracle', 'w14hbond_system.data'), 'utf8');

/** Runs an input and returns the first error message and the thermo rows. */
async function run(script: string, files: Record<string, string> = {}) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  try {
    await session.execute(script);
  } catch {
    // the failure is reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { error: err && 'message' in err ? String(err.message) : undefined, rows };
}

const thermoOf = async (setup: string, keys: string) => {
  const r = await run(`${SYSTEM}${setup}\nthermo_style custom step ${keys}\nthermo_modify format float %.15g\nrun 0\n`);
  expect(r.error).toBeUndefined();
  return r.rows[0] as Record<string, number>;
};

describe('compute pair: values', () => {
  it('hybrid sub-style scalars add up to the hybrid total (lj + morse 1 + morse 2 = hybrid)', async () => {
    const row = await thermoOf(`${HYBRID}compute lj all pair lj/cut\ncompute m1 all pair morse 1\ncompute m2 all pair morse 2\ncompute tot all pair hybrid/overlay\n`, 'c_lj c_m1 c_m2 c_tot');
    expect(row.c_lj + row.c_m1 + row.c_m2).toBeCloseTo(row.c_tot, 12);
    expect(row.c_m2).not.toBeCloseTo(row.c_m1, 6);
  });

  it('evalue epair is evdwl plus ecoul, and ecoul of a pair style without Coulomb is zero', async () => {
    const row = await thermoOf(`${PLAIN}compute e all pair lj/cut\ncompute el all pair lj/cut evdwl\ncompute ec all pair lj/cut ecoul\n`, 'c_e c_el c_ec');
    expect(row.c_e).toBeCloseTo(row.c_el + row.c_ec, 12);
    expect(row.c_ec).toBe(0);
  });

  it('the pair energy excludes the tail correction (pair_modify tail yes)', async () => {
    const plain = await thermoOf(`${PLAIN}compute e all pair lj/cut\n`, 'c_e');
    const tail = await thermoOf(`${PLAIN}pair_modify tail yes\ncompute e all pair lj/cut\n`, 'c_e');
    expect(tail.c_e).toBeCloseTo(plain.c_e, 12);
    const pe = await thermoOf(`${PLAIN}pair_modify tail yes\ncompute e all pair lj/cut\n`, 'pe');
    expect(pe.pe).not.toBeCloseTo(tail.c_e, 6);
  });

  it('the hybrid/overlay total with evalue evdwl equals its epair (no Coulomb part)', async () => {
    const row = await thermoOf(`${HYBRID}compute a all pair hybrid/overlay\ncompute b all pair hybrid/overlay evdwl\ncompute c all pair hybrid/overlay ecoul\n`, 'c_a c_b c_c');
    expect(row.c_b).toBeCloseTo(row.c_a, 12);
    expect(row.c_c).toBe(0);
  });

  it('hbond/dreiding vector: count is an integer, energy equals the hbond scalar', async () => {
    const r = await run(`units real
atom_style full
boundary p p p
read_data w14hbond.data
special_bonds lj/coul 0.0 0.0 1.0
pair_style hybrid/overlay lj/cut 8.0 hbond/dreiding/lj 4 3.0 4.5 90.0
pair_coeff * * lj/cut 0.1 2.8
pair_coeff 1 3 hbond/dreiding/lj 2 i 2.0 2.2
pair_coeff 1 5 hbond/dreiding/lj 2 i 1.5 2.4 2 3.0 4.5 90.0
bond_style harmonic
bond_coeff 1 450.0 0.9572
bond_coeff 2 300.0 1.2
neighbor 2.0 bin
timestep 1.0
compute hb all pair hbond/dreiding/lj
thermo_style custom step c_hb c_hb[1] c_hb[2]
thermo_modify format float %.15g
run 0
`, { 'w14hbond.data': HBOND });
    expect(r.error).toBeUndefined();
    const row = r.rows[0] as Record<string, number>;
    expect(Number.isInteger(row['c_hb[1]'])).toBe(true);
    expect(row['c_hb[1]']).toBeGreaterThan(0);
    expect(row['c_hb[2]']).toBeCloseTo(row.c_hb, 12);
  });
});

describe('compute pair: StyleError paths', () => {
  it('an unknown pair style is an error', async () => {
    const r = await run(`${SYSTEM}${PLAIN}compute x all pair nosuchstyle\nrun 0\n`);
    expect(r.error).toMatch(/nosuchstyle/);
  });

  it('a repeated sub-style without nsub is an error', async () => {
    const r = await run(`${SYSTEM}${HYBRID}compute x all pair morse\nthermo_style custom step c_x\nrun 0\n`);
    expect(r.error).toMatch(/morse/);
  });

  it('an nsub above the number of listings is an error', async () => {
    const r = await run(`${SYSTEM}${HYBRID}compute x all pair morse 3\nthermo_style custom step c_x\nrun 0\n`);
    expect(r.error).toMatch(/morse/);
  });

  it('nsub 0 is an error', async () => {
    const r = await run(`${SYSTEM}${HYBRID}compute x all pair morse 0\nrun 0\n`);
    expect(r.error).toMatch(/nsub/);
  });

  it('an unknown evalue keyword is an error that names it', async () => {
    const r = await run(`${SYSTEM}${PLAIN}compute x all pair lj/cut bogus\nrun 0\n`);
    expect(r.error).toMatch(/bogus/);
  });

  it('a sub-style that is not part of the hybrid is an error', async () => {
    const r = await run(`${SYSTEM}${HYBRID}compute x all pair coul/cut\nrun 0\n`);
    expect(r.error).toMatch(/coul\/cut/);
  });

  it('a vector of a style without one is an error at thermo output', async () => {
    const r = await run(`${SYSTEM}${PLAIN}compute x all pair lj/cut\nthermo_style custom step c_x[1]\nrun 0\n`);
    expect(r.error).toMatch(/vector/);
  });

  it('the hybrid total has no vector', async () => {
    const r = await run(`${SYSTEM}${HYBRID}compute x all pair hybrid/overlay\nthermo_style custom step c_x[1]\nrun 0\n`);
    expect(r.error).toMatch(/vector/);
  });

  it('a documented vector style the engine lacks (gauss) is an error when defined', async () => {
    const r = await run(`${SYSTEM}${PLAIN}compute x all pair gauss\nrun 0\n`);
    expect(r.error).toMatch(/gauss/);
  });

  it('a hybrid/scaled sub-style with factor 0 is an error (its energy is not tallied in the browser engine)', async () => {
    const r = await run(`${SYSTEM}pair_style hybrid/scaled 0.0 lj/cut 2.5\npair_coeff * * lj/cut 1.0 1.0 2.5\ncompute x all pair lj/cut\nthermo_style custom step c_x\nrun 0\n`);
    expect(r.error).toMatch(/did not run/);
  });
});
