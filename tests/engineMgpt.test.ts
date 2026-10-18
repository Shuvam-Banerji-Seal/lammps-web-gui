import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * pair_style mgpt (GPL port of LAMMPS src/MGPT, src/engine-gpl/mgpt): the potential files, the pair_coeff options and the
 * refusals. Native agreement (bit for bit on positions in most cases) is in tests/oracle/w212mg_*.in.
 */

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const file = (name: string): string => readFileSync(join(POT, name), 'utf8');
const TA = { 'Ta6.8x.mgpt.parmin': file('Ta6.8x.mgpt.parmin'), 'Ta6.8x.mgpt.potin': file('Ta6.8x.mgpt.potin') };

interface Run { rows: ThermoRow[]; logs: string[]; error?: string }

const run = async (lines: string[], files: Record<string, string> = TA): Promise<Run> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error: string | undefined;
  try {
    await session.execute(`${lines.join('\n')}\n`);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) error = String(err.message);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { rows, logs, error };
};

/** A bcc Ta crystal of n^3 cells at the potential's atomic volume 121.6 bohr^3. */
const bcc = (n: number, pre: string[] = []): string[] => [
  'units electron', 'atom_style atomic', ...pre,
  'variable a equal (121.6*2.0)^(1.0/3.0)', 'lattice bcc ${a}',
  `region box block 0 ${n} 0 ${n} 0 ${n}`, 'create_box 1 box', 'create_atoms 1 box',
];
const COEFF = 'pair_coeff * * Ta6.8x.mgpt.parmin Ta6.8x.mgpt.potin 121.6';

describe('pair_style mgpt', () => {
  it('gives the energy and pressure of examples/PACKAGES/mgpt/in.bcc0 (250 atoms, run 0)', async () => {
    const r = await run([...bcc(5), 'pair_style mgpt', COEFF, 'neighbor 0.1 bin',
      'thermo_style custom step pe press', 'thermo_modify format float %.15g', 'run 0']);
    expect(r.error).toBeUndefined();
    const { pe, press } = r.rows[0] as { pe: number; press: number };
    // Measured with native LAMMPS (b332f28 with the MGPT package): total energy (ry/atom) = -0.59530002488734,
    // pressure (gpa) = -1.15946260887556, as the example prints them (2 pe / natoms and press / 1e9)
    expect(2.0 * pe / 250).toBeCloseTo(-0.59530002488734, 13);
    expect(press / 1.0e9).toBeCloseTo(-1.15946260887556, 12);
  });

  it('sets the mass of every type from the potential file (Atom::set_mass in pair_coeff)', async () => {
    const r = await run([...bcc(2), 'mass 1 1.0', 'pair_style mgpt', COEFF,
      'compute m all property/atom mass', 'compute mx all reduce max c_m', 'thermo_style custom step c_mx', 'run 0']);
    expect(r.error).toBeUndefined();
    expect(r.rows[0].c_mx).toBe(180.948);
  });

  it('accepts volpress, nbody and precision, and warns when nbody leaves out terms of the potential', async () => {
    const r = await run([...bcc(2), 'pair_style mgpt', `${COEFF} volpress no nbody 12 precision single`, 'run 0']);
    expect(r.error).toBeUndefined();
    expect(r.logs.join('\n')).toMatch(/Warning: nbody=3 \(suggested=15\) set to disregard multibody-forces in potential\./);
  });

  it('refuses arguments as native pair_coeff does', async () => {
    const base = [...bcc(2), 'pair_style mgpt'];
    expect((await run([...base, 'pair_coeff * * Ta6.8x.mgpt.parmin Ta6.8x.mgpt.potin'])).error).toMatch(/Not enough arguments for mgpt/);
    expect((await run([...base, 'pair_coeff 1 1 Ta6.8x.mgpt.parmin Ta6.8x.mgpt.potin 121.6'])).error).toMatch(/Incorrect args for pair coefficients/);
    expect((await run([...base, 'pair_coeff * * Ta6.8x.mgpt.parmin Ta6.8x.mgpt.potin Omega'])).error).toMatch(/Expected floating point parameter instead of 'Omega'/);
    expect((await run([...base, 'pair_coeff * * Ta6.8x.mgpt.parmin Ta6.8x.mgpt.potin 0.0'])).error).toMatch(/Invalid volume in mgpt/);
    expect((await run([...base, `${COEFF} nbody 5`])).error).toMatch(/Invalid value for nbody flag/);
    expect((await run([...base, `${COEFF} volpress maybe`])).error).toMatch(/Invalid value for volumetric pressure argument/);
    expect((await run([...base, `${COEFF} precision half`])).error).toMatch(/Invalid value for precision argument/);
    expect((await run([...base, `${COEFF} cutoff 3`])).error).toMatch(/Invalid argument\. Allowed arguments are/);
    expect((await run([...base, 'pair_coeff * * Ta6.8x.mgpt.parmin missing.potin 121.6'])).error).toMatch(/Cannot open MGPT potential files/);
  });

  it('requires newton pair on, as native init_style does', async () => {
    const r = await run([...bcc(2, ['newton off']), 'pair_style mgpt', COEFF, 'run 0']);
    expect(r.error).toMatch(/Pair style mgpt requires newton pair on/);
  });

  it('refuses what the port does not reproduce, naming it', async () => {
    const tri = await run(['units electron', 'atom_style atomic', 'variable a equal (121.6*2.0)^(1.0/3.0)', 'lattice bcc ${a}',
      'region box prism 0 2 0 2 0 2 0.5 0 0', 'create_box 1 box', 'create_atoms 1 box', 'pair_style mgpt', COEFF, 'run 0']);
    expect(tri.error).toMatch(/pair_style mgpt with a triclinic box is not supported/);
    const bonded = await run(['units electron', 'atom_style bond', 'region box block 0 20 0 20 0 20', 'create_box 1 box bond/types 1 extra/bond/per/atom 1',
      'create_atoms 1 single 5 5 5', 'create_atoms 1 single 9 5 5', 'create_bonds single/bond 1 1 2', 'bond_style harmonic', 'bond_coeff 1 1.0 4.0',
      'pair_style mgpt', COEFF, 'run 0']);
    expect(bonded.error).toMatch(/pair_style mgpt with bonded atoms \(special neighbors\) is not supported/);
  });
});
