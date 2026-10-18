import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix orient/fcc and fix orient/bcc (docs.lammps.org/fix_orient.html; src/ORIENT/fix_orient_fcc.cpp and
 * fix_orient_bcc.cpp at b332f28): the argument errors and the reference-file reader, the energy of a perfect crystal,
 * the fix_modify energy and f_ID relations and the statistics line. Parity with native LAMMPS is in
 * tests/oracle/w236or_*.in; these tests check the behaviour the oracle cases do not reach.
 */

// the reference vectors of a cubic grain at its own orientation: one of each pair of opposite fcc neighbours (a = 3.6)
const FCC_IDENTITY = '1.8 1.8 0\n1.8 -1.8 0\n1.8 0 1.8\n1.8 0 -1.8\n0 1.8 1.8\n0 1.8 -1.8\n';
// the same vectors rotated by 20 degrees about z (a different grain)
const c = Math.cos(Math.PI / 9), s = Math.sin(Math.PI / 9);
const rot = (x: number, y: number, z: number): string => `${(c * x - s * y).toFixed(10)} ${(s * x + c * y).toFixed(10)} ${z.toFixed(10)}`;
const FCC_ROTATED = [[1.8, 1.8, 0], [1.8, -1.8, 0], [1.8, 0, 1.8], [1.8, 0, -1.8], [0, 1.8, 1.8], [0, 1.8, -1.8]]
  .map(([x, y, z]) => rot(x, y, z)).join('\n') + '\n';
// one of each pair of opposite bcc neighbours (a = 2.88)
const BCC_IDENTITY = '1.44 1.44 1.44\n1.44 1.44 -1.44\n1.44 -1.44 1.44\n-1.44 1.44 1.44\n';

interface RunResult {
  thermo: Record<string, number>[];
  logs: string[];
  error: string;
}

const runScript = async (text: string, files: Record<string, string>): Promise<RunResult> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  for (const [name, body] of Object.entries(files)) session.addFile(name, body);
  let error = '';
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  const logs = events
    .filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log')
    .map((e) => e.text);
  return { thermo, logs, error };
};

// a periodic fcc crystal of 256 atoms with a zero pair style: the only force is the fix's
const FCC_BOX = `
units metal
atom_style atomic
lattice fcc 3.6
region b block 0 4 0 4 0 4
create_box 1 b
create_atoms 1 box
mass 1 63.5
pair_style zero 5.0
pair_coeff * *
`;

const BCC_BOX = `
units metal
atom_style atomic
lattice bcc 2.88
region b block 0 4 0 4 0 4
create_box 1 b
create_atoms 1 box
mass 1 55.8
pair_style zero 5.0
pair_coeff * *
`;

describe('fix orient/fcc and orient/bcc arguments and reader', () => {
  it('refuses the wrong number of arguments with the native usage error', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc 0 1 3.6 0.01 0.25\nrun 0\n', {});
    expect(r.error).toContain('Illegal fix orient/fcc command');
  });

  it('refuses a direction other than 0 or 1', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc 0 2 3.6 0.01 0.25 0.75 a.ori b.ori\nrun 0\n',
      { 'a.ori': FCC_IDENTITY, 'b.ori': FCC_ROTATED });
    expect(r.error).toContain('Illegal fix orient/fcc command');
  });

  it('names the bad integer argument as native does', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc x 1 3.6 0.01 0.25 0.75 a.ori b.ori\nrun 0\n',
      { 'a.ori': FCC_IDENTITY, 'b.ori': FCC_ROTATED });
    expect(r.error).toContain("Expected integer parameter instead of 'x' in input script or data file");
  });

  it('reports a missing reference file as a read failure', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc 0 1 3.6 0.01 0.25 0.75 missing.ori b.ori\nrun 0\n',
      { 'b.ori': FCC_ROTATED });
    expect(r.error).toContain('Fix orient/fcc file read failed: cannot open fix orient/fcc potential file missing.ori');
  });

  it('reports a short reference file with the values read so far', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc 0 1 3.6 0.01 0.25 0.75 a.ori b.ori\nrun 0\n',
      { 'a.ori': '1 2 3\n4 5\n', 'b.ori': FCC_ROTATED });
    expect(r.error).toContain('Fix orient/fcc file read failed: Incorrect format in fix orient/fcc file! 5/18 values');
  });

  it('refuses a reference file written for other units', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc 0 1 3.6 0.01 0.25 0.75 a.ori b.ori\nrun 0\n',
      { 'a.ori': 'UNITS: real\n' + FCC_IDENTITY, 'b.ori': FCC_ROTATED });
    expect(r.error).toContain('fix orient/fcc file a.ori requires real units but metal units are in use');
  });

  it('reads comments and values that continue over lines the same as a plain file', async () => {
    const plain = await runScript(FCC_BOX + 'fix or all orient/fcc 0 1 3.6 0.01 0.25 0.75 a.ori b.ori\nfix_modify or energy yes\nthermo_style custom step pe f_or\nthermo 1\nrun 0\n',
      { 'a.ori': FCC_IDENTITY, 'b.ori': FCC_ROTATED });
    const split = FCC_IDENTITY.split('\n').map((l, i) => (i === 2 ? `# reference grain\n${l.split(' ').slice(0, 1).join(' ')}\n${l.split(' ').slice(1).join(' ')}` : l)).join('\n');
    const fancy = await runScript(FCC_BOX + 'fix or all orient/fcc 0 1 3.6 0.01 0.25 0.75 a.ori b.ori\nfix_modify or energy yes\nthermo_style custom step pe f_or\nthermo 1\nrun 0\n',
      { 'a.ori': '# identity grain\n' + split + '\n', 'b.ori': FCC_ROTATED });
    expect(plain.error).toBe('');
    expect(fancy.error).toBe('');
    expect(fancy.thermo[0].f_or).toBe(plain.thermo[0].f_or);
  });
});

describe('fix orient/fcc energy', () => {
  it('adds no energy to a perfect crystal of the favoured grain', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc 0 1 3.6 0.01 0.25 0.75 a.ori b.ori\nfix_modify or energy yes\nthermo_style custom step pe f_or\nthermo 1\nrun 0\n',
      { 'a.ori': FCC_IDENTITY, 'b.ori': FCC_ROTATED });
    expect(r.error).toBe('');
    expect(r.thermo[0].f_or).toBe(0);
    expect(r.thermo[0].pe).toBe(0);
  });

  it('adds a uniform energy, between 0 and dE per atom, to a perfect crystal seen from a rotated grain', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc 0 1 3.6 0.01 0.25 0.75 a.ori b.ori\nfix_modify or energy yes\nthermo_style custom step pe f_or\nthermo 1\nrun 0\n',
      { 'a.ori': FCC_ROTATED, 'b.ori': FCC_IDENTITY });
    expect(r.error).toBe('');
    const perAtom = r.thermo[0].f_or / 256;
    expect(perAtom).toBeGreaterThan(0);
    expect(perAtom).toBeLessThanOrEqual(0.01 + 1e-15);
    // fix_modify energy yes: the potential energy holds the added energy (pair zero adds nothing)
    expect(r.thermo[0].pe).toBeCloseTo(r.thermo[0].f_or, 12);
  });

  it('prints the statistics line every nstats steps', async () => {
    const r = await runScript(FCC_BOX + 'fix or all orient/fcc 1 1 3.6 0.01 0.25 0.75 a.ori b.ori\nrun 0\n',
      { 'a.ori': FCC_IDENTITY, 'b.ori': FCC_ROTATED });
    expect(r.error).toBe('');
    expect(r.logs.some((l) => l.startsWith('orient step 0: 256 atoms have '))).toBe(true);
  });
});

describe('fix orient/bcc', () => {
  it('runs with four reference vectors per grain and adds between 0 and dE per atom', async () => {
    const bccRotated = BCC_IDENTITY.split('\n').filter((l) => l.trim() !== '').map((l) => {
      const [x, y, z] = l.split(/\s+/).map(Number);
      return rot(x, y, z);
    }).join('\n') + '\n';
    const r = await runScript(BCC_BOX + 'fix or all orient/bcc 0 1 2.88 0.02 0.2 0.8 a.ori b.ori\nfix_modify or energy yes\nthermo_style custom step pe f_or\nthermo 1\nrun 0\n',
      { 'a.ori': bccRotated, 'b.ori': BCC_IDENTITY });
    expect(r.error).toBe('');
    const perAtom = r.thermo[0].f_or / 128;
    expect(perAtom).toBeGreaterThanOrEqual(0);
    expect(perAtom).toBeLessThanOrEqual(0.02 + 1e-15);
    expect(r.thermo[0].pe).toBeCloseTo(r.thermo[0].f_or, 12);
  });

  it('adds no energy to a perfect bcc crystal of the favoured grain', async () => {
    const r = await runScript(BCC_BOX + 'fix or all orient/bcc 0 1 2.88 0.02 0.2 0.8 a.ori b.ori\nfix_modify or energy yes\nthermo_style custom step pe f_or\nthermo 1\nrun 0\n',
      { 'a.ori': BCC_IDENTITY, 'b.ori': FCC_ROTATED.split('\n').slice(0, 4).join('\n') + '\n' });
    expect(r.error).toBe('');
    expect(r.thermo[0].f_or).toBe(0);
  });
});
