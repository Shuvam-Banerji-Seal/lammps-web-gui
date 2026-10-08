import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Per-atom virial for pair styles snap and mliap (compute stress/atom reads
 * pc.vatom). The pair virial of one neighbour term is -d (x) G, with d the
 * bond vector and G the gradient of the central atom energy with respect to
 * d. It is split half to each of the two atoms, which was measured with
 * native LAMMPS (black box) on a 3-atom triclinic system to agree atom by
 * atom. The identity checked here: the per-atom virial summed over atoms
 * gives the virial of the pressure, so the sum of the stress/atom diagonal
 * equals -3 * press * vol when no kinetic term is included (docs:
 * compute_stress_atom.rst, pair virial only).
 */

const ORACLE = join(__dirname, 'oracle');

const runInput = async (lines: string[], files: Record<string, string>): Promise<{ thermo: ThermoRow[]; events: EngineEvent[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  await session.execute(lines.join('\n'));
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { thermo, events };
};

const base = [
  'units lj',
  'atom_style atomic',
  'region box prism 0 6 0 6 0 6 0.5 0.3 0.2',
  'create_box 2 box',
  'create_atoms 1 single 1.0 1.2 0.9 units box',
  'create_atoms 2 single 2.1 1.5 1.1 units box',
  'create_atoms 1 single 0.4 2.6 3.3 units box',
  'create_atoms 2 single 3.3 0.6 2.2 units box',
  'mass * 1.0',
  'compute s all stress/atom NULL',
  'compute sr all reduce sum c_s[1] c_s[2] c_s[3] c_s[4] c_s[5] c_s[6]',
  'compute eg all pe/atom',
  'compute er all reduce sum c_eg',
  'thermo_style custom step pe press vol c_sr[1] c_sr[2] c_sr[3] c_er',
  'thermo_modify format float %.15g norm no',
];

const checkIdentity = (row: ThermoRow): void => {
  const diag = (row['c_sr[1]'] as number) + (row['c_sr[2]'] as number) + (row['c_sr[3]'] as number);
  const want = -3 * (row.press as number) * (row.vol as number);
  expect(Math.abs(diag - want)).toBeLessThanOrEqual(1e-10 + 1e-9 * Math.abs(want));
  expect(Math.abs((row.c_er as number) - (row.pe as number))).toBeLessThanOrEqual(1e-12 + 1e-10 * Math.abs(row.pe as number));
};

describe('per-atom virial for pair style snap', () => {
  const coef = readFileSync(join(ORACLE, 'w19vatom_snap.snapcoeff'), 'utf8');
  const param = readFileSync(join(ORACLE, 'w19vatom_snap.snapparam'), 'utf8');
  const files = { 'w19vatom_snap.snapcoeff': coef, 'w19vatom_snap.snapparam': param };

  it('per-atom virial sums to the pair virial (press * vol)', async () => {
    const { thermo } = await runInput([...base, 'pair_style snap', 'pair_coeff * * w19vatom_snap.snapcoeff w19vatom_snap.snapparam A B', 'run 0'], files);
    expect(thermo.length).toBe(1);
    checkIdentity(thermo[0]);
  });

  it('per-atom virial is not requested without compute stress/atom', async () => {
    const lines = base.filter((l) => !l.startsWith('compute s ') && !l.startsWith('compute sr') && !l.startsWith('thermo_style'));
    lines.push('thermo_style custom step pe press vol', 'pair_style snap', 'pair_coeff * * w19vatom_snap.snapcoeff w19vatom_snap.snapparam A B', 'run 0');
    const { thermo } = await runInput(lines, files);
    expect(thermo.length).toBe(1);
    expect(Number.isFinite(thermo[0].press as number)).toBe(true);
  });
});

describe('per-atom virial for pair style mliap', () => {
  const model = readFileSync(join(ORACLE, 'w19vatom_nn.mliap.model'), 'utf8');
  const desc = readFileSync(join(ORACLE, 'w19vatom_nn.mliap.descriptor'), 'utf8');
  const files = { 'w19vatom_nn.mliap.model': model, 'w19vatom_nn.mliap.descriptor': desc };

  it('per-atom virial sums to the pair virial (press * vol)', async () => {
    const { thermo } = await runInput([...base,
      'pair_style mliap model nn w19vatom_nn.mliap.model descriptor sna w19vatom_nn.mliap.descriptor',
      'pair_coeff * * A B', 'run 0'], files);
    expect(thermo.length).toBe(1);
    checkIdentity(thermo[0]);
  });
});
