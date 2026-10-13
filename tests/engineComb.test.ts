import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { readCombEntries, setupCombParams } from '../src/engine-gpl/comb/params';
import { StyleError } from '../src/engine/force/types';

/*
 * Unit tests for the COMB port (src/engine-gpl/comb): the potential file reader, the element mapping and the
 * refusals, and the energy of a small cluster against native LAMMPS (black box). The full runs are in
 * tests/oracle/w58comb_*.in (tests/engineOracle.test.ts).
 */

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const ffield = readFileSync(join(POT, 'ffield.comb'), 'utf8');

const run = async (script: string, files: Record<string, string> = {}): Promise<EngineEvent[]> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never, undefined, 0, 1);
  for (const [name, text] of Object.entries(files)) s.addFile(name, text);
  try {
    await s.execute(script);
  } catch {
    // the error is also an 'error' event, read by errorOf
  }
  return events;
};

const errorOf = (events: EngineEvent[]): string =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error').map((e) => e.message).join('\n');

const box = `units metal
atom_style charge
boundary p p p
region box block 0 40 0 40 0 40
create_box 1 box
create_atoms 1 single 10.0 10.0 10.0 units box
create_atoms 1 single 12.35 10.0 10.0 units box
create_atoms 1 single 11.1 12.1 10.3 units box
mass 1 28.0855
`;

describe('COMB potential file reader', () => {
  it('reads the Si entry of ffield.comb with the LAMMPS field order and sanity checks', () => {
    const entries = readCombEntries(ffield, ['Si']);
    expect(entries).toHaveLength(1);
    const si = entries[0];
    expect(si.powerm).toBe(3);
    expect(si.c).toBe(100390);
    expect(si.d).toBe(16.218);
    expect(si.bigr).toBe(2.9);
    expect(si.bigd).toBe(0.1);
    expect(si.powermint).toBe(3);
    expect(si.hfocor).toBeGreaterThanOrEqual(0);
  });

  it('derives the cutoffs and the bond-order constants of setup_params', () => {
    const entries = readCombEntries(ffield, ['Si']);
    const sp = setupCombParams(entries, ['Si']);
    const si = entries[0];
    expect(si.cut).toBeCloseTo(3.0, 12);
    expect(si.cutsq).toBeCloseTo(9.0, 12);
    expect(sp.cutmax).toBeGreaterThanOrEqual(si.cut);
    expect(sp.elem3param).toEqual(new Int32Array([0]));
    expect(si.gamma).toBe(1);
    expect(si.rlm1).toBeCloseTo(0.5 * (si.lam11 + si.lam12) * si.romigc, 14);
  });

  it('keeps only the entries whose three element names are in the list', () => {
    const si = readCombEntries(ffield, ['Si']);
    const sio = readCombEntries(ffield, ['Si', 'O']);
    expect(sio.length).toBeGreaterThan(si.length);
    expect(sio.every((p) => p.ielement < 2 && p.jelement < 2 && p.kelement < 2)).toBe(true);
  });

  it('reports a missing triplet of the mapped elements (setup_params)', () => {
    const entries = readCombEntries(ffield, ['Si', 'Ge']);
    expect(() => setupCombParams(entries, ['Si', 'Ge'])).toThrow(/missing an entry for: Si Si Ge/);
    expect(() => setupCombParams(entries, ['Si', 'Ge'])).toThrow(StyleError);
  });
});

describe('pair_style comb commands', () => {
  it('maps an element that the potential file does not contain to an error', async () => {
    const ev = await run(`${box}pair_style comb\npair_coeff * * ffield.comb Ge\n`, { 'ffield.comb': ffield });
    expect(errorOf(ev)).toMatch(/missing an entry for: Ge Ge Ge/);
  });

  it('refuses a pair_style argument it does not implement, naming it', async () => {
    const ev = await run(`${box}pair_style comb foo\n`, { 'ffield.comb': ffield });
    expect(errorOf(ev)).toMatch(/foo/);
  });

  it('refuses NULL in pair_coeff, naming it', async () => {
    const ev = await run(`${box}pair_style comb\npair_coeff * * ffield.comb NULL\n`, { 'ffield.comb': ffield });
    expect(errorOf(ev)).toMatch(/NULL/);
  });

  it('needs one element per atom type', async () => {
    const ev = await run(`${box}pair_style comb\npair_coeff * * ffield.comb Si Si\n`, { 'ffield.comb': ffield });
    expect(errorOf(ev)).toMatch(/one element name per atom type/);
  });

  it('does not offer pair_style comb3 (refused by name)', async () => {
    const ev = await run(`${box}pair_style comb3\n`, { 'ffield.comb3': ffield });
    expect(errorOf(ev)).toMatch(/comb3/);
  });
});

describe('fix qeq/comb commands', () => {
  it('needs pair_style comb', async () => {
    const ev = await run(`${box}pair_style lj/cut 3.0\npair_coeff * * 0.1 2.5\nfix q all qeq/comb 1 0.001\nrun 0\n`);
    expect(errorOf(ev)).toMatch(/pair_style comb/);
  });

  it('accepts the file keyword and writes the charge-solve log (fix_qeq_comb.cpp)', async () => {
    const files = new Map<string, string>();
    const events: EngineEvent[] = [];
    const s = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: (n: string, t: string, ap: boolean) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) } as never, undefined, 0, 1);
    s.addFile('ffield.comb', ffield);
    await s.execute(`${box}pair_style comb\npair_coeff * * ffield.comb Si\nfix q all qeq/comb 1 0.001 file fq.out\nrun 0\n`);
    expect(errorOf(events)).toBe('');
    expect(files.get('fq.out')?.startsWith('Charge equilibration on step 0\n')).toBe(true);
    expect(files.get('fq.out')).toMatch(/Charges converged in \d+ iterations to /);
  });
});

describe('pair_style comb energy against native LAMMPS (black box)', () => {
  // Native reference: a three-atom Si cluster in a 40 A periodic box, charges 0.3 -0.2 -0.1, run 0
  // (Measured with native LAMMPS, black box): PotEng -4.53119680680527, E_vdwl -4.40280842091289, E_coul -0.128388385892382.
  it('matches the native pe, evdwl and ecoul of the cluster', async () => {
    const script = `${box}set atom 1 charge 0.3
set atom 2 charge -0.2
set atom 3 charge -0.1
pair_style comb
pair_coeff * * ffield.comb Si
thermo_style custom step pe evdwl ecoul
thermo_modify format float %.15g
run 0
`;
    const ev = await run(script, { 'ffield.comb': ffield });
    expect(errorOf(ev)).toBe('');
    const row = ev.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow)[0];
    const close = (a: number, b: number) => Math.abs(a - b) <= 1e-10 * Math.max(1, Math.abs(b));
    expect(close(row.pe as number, -4.53119680680527)).toBe(true);
    expect(close(row.evdwl as number, -4.40280842091289)).toBe(true);
    expect(close(row.ecoul as number, -0.128388385892382)).toBe(true);
  });
});
