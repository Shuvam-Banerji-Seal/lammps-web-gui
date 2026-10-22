import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * FEP package ports (src/engine-gpl/fep/): morse/soft, lj/cut/soft/gapsys, coul/cut/soft/gapsys, fix adapt/fep and
 * compute fep. Native parity lives in tests/engineOracle.test.ts (w69fep_*); these tests cover the closed forms, the
 * finite-difference forces, the restore behaviour and the refusals.
 */

/** Runs lines on a fresh session and returns the thermo rows of the last thermo output (or throws). */
const runLines = async (lines: string[]): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  await session.execute(lines.join('\n') + '\n');
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

/** Two atoms (types 1 and 2) on the x axis of a large periodic box, separated by r. */
const dimer = (r: number, charges: [number, number] | null): string[] => [
  'units lj', 'atom_style ' + (charges ? 'charge' : 'atomic'), 'boundary p p p', 'region box block 0 30 0 30 0 30', 'create_box 2 box',
  'create_atoms 1 single 5 5 5', `create_atoms 2 single ${5 + r} 5 5`, 'mass * 1.0',
  ...(charges ? [`set type 1 charge ${charges[0]}`, `set type 2 charge ${charges[1]}`] : []),
];

/** The x force on atom 2 of the dimer (write_dump of the force components). */
const forceX2 = async (style: string[], r: number): Promise<number> => {
  const files = new Map<string, string>();
  const session = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
  await session.execute([...dimer(r, null), ...style, 'run 0', 'write_dump all custom fdump_fep.txt id fx fy fz modify format float %.17g sort id'].join('\n') + '\n');
  const dump = (files.get('fdump_fep.txt') ?? '').trim().split('\n');
  const k0 = dump.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return Number(dump[k0 + 2].trim().split(/\s+/)[1]);
};

const energyAt = async (style: string[], r: number, charges: [number, number] | null = null): Promise<number> => {
  const rows = await runLines([...dimer(r, charges), ...style, 'thermo_style custom step evdwl ecoul', 'thermo_modify format float %.17g norm no', 'run 0']);
  const row = rows[rows.length - 1];
  return (row.evdwl as number) + (row.ecoul as number);
};

describe('morse/soft (src/engine-gpl/fep/pair_morse_soft.ts)', () => {
  it('at lambda 1 is the Morse potential D (exp(-2a dr) - 2 exp(-a dr)) with dr = r - r0', async () => {
    const D = 1.5, a = 2.0, r0 = 1.1;
    for (const r of [0.9, 1.0, 1.3, 2.0]) {
      const e = await energyAt(['pair_style morse/soft 2 0.8 3.0', `pair_coeff * * ${D} ${a} ${r0} 1.0`], r);
      const dr = r - r0;
      const morse = D * (Math.exp(-2 * a * dr) - 2 * Math.exp(-a * dr));
      expect(e).toBeCloseTo(morse, 12);
    }
  });

  it('the energy goes to zero at the cutoff region only through the cutoff (no offset is subtracted)', async () => {
    const e1 = await energyAt(['pair_style morse/soft 2 0.8 3.0', 'pair_coeff * * 1.5 2.0 1.1 1.0'], 2.5);
    const e2 = await energyAt(['pair_style morse/soft 2 0.8 3.0', 'pair_coeff * * 1.5 2.0 1.1 1.0', 'pair_modify shift yes'], 2.5);
    expect(e2).toBe(e1);
  });

  it('forces are minus the finite-difference derivative of the energy (lambda below and above the shift range)', async () => {
    const lines = (lam: number) => ['pair_style morse/soft 2 0.8 3.0', `pair_coeff * * 1.5 2.0 1.1 ${lam}`];
    for (const lam of [0.5, 1.0]) {
      const r = 1.4, h = 1e-5;
      const de = ((await energyAt(lines(lam), r + h)) - (await energyAt(lines(lam), r - h))) / (2 * h);
      // the force on atom 2 along x (atom 2 sits at +x from atom 1) is -dE/dr
      expect(await forceX2(lines(lam), r)).toBeCloseTo(-de, 6);
    }
  });
});

describe('lj/cut/soft/gapsys and coul/cut/soft/gapsys (src/engine-gpl/fep/pair_soft_gapsys.ts)', () => {
  it('at lambda 0 the inner radius is zero and the energy is the plain 12-6 LJ (equal to lj/cut)', async () => {
    for (const r of [0.95, 1.2, 1.9]) {
      const g = await energyAt(['pair_style lj/cut/soft/gapsys 0.5 2.5', 'pair_coeff * * 1.0 1.0 0.0'], r);
      const lj = await energyAt(['pair_style lj/cut 2.5', 'pair_coeff * * 1.0 1.0'], r);
      expect(g).toBeCloseTo(lj, 12);
    }
  });

  it('inside and outside the inner radius the force is minus the finite-difference slope of the energy', async () => {
    const style = ['pair_style lj/cut/soft/gapsys 0.8 2.5', 'pair_coeff * * 1.0 1.0 0.9'];
    const cin = 0.8 * (26 * 0.9 / 7) ** (1 / 6);
    const h = 1e-6;
    for (const r of [cin * 0.6, cin * 1.1]) {
      const slope = ((await energyAt(style, r + h)) - (await energyAt(style, r - h))) / (2 * h);
      expect(await forceX2(style, r)).toBeCloseTo(-slope, 6);
    }
  });

  it('refuses pair_modify tail yes', async () => {
    await expect(runLines([...dimer(1.5, null), 'pair_style lj/cut/soft/gapsys 0.5 2.5', 'pair_coeff * * 1.0 1.0 0.5', 'pair_modify tail yes', 'run 0'])).rejects.toThrow(/tail/);
  });

  it('coul/cut/soft/gapsys is zero for uncharged atoms and rejects a lambda outside [0, 1] for lj', async () => {
    expect(await energyAt(['pair_style coul/cut/soft/gapsys 0.1 1.5 4.0', 'pair_coeff * * 0.5'], 1.3, [0, 0])).toBe(0);
    await expect(runLines([...dimer(1.5, null), 'pair_style lj/cut/soft/gapsys 0.5 2.5', 'pair_coeff * * 1.0 1.0 1.5', 'run 0'])).rejects.toThrow(/lambda/);
  });
});

describe('fix adapt/fep and compute fep (src/engine-gpl/fep/)', () => {
  const base = [...dimer(1.2, [0.5, -0.5]), 'variable dl equal -0.1', 'pair_style lj/cut/coul/long/soft 1 0.5 0.5 8.0 8.0', 'pair_coeff * * 0.1 3.0 1.0', 'kspace_style ewald 1.0e-4'];

  it('compute fep reports the energy difference and its Boltzmann factor, and leaves the state unchanged', async () => {
    const plain = await runLines([...base, 'thermo_style custom step pe', 'thermo_modify format float %.17g', 'run 0']);
    const withFep = await runLines([
      ...base, 'variable dq equal 0.01', 'compute FEP all fep 300.0 pair lj/cut/coul/long/soft lambda * * v_dl atom charge 1 v_dq',
      'thermo_style custom step pe c_FEP[1] c_FEP[2] c_FEP[3]', 'thermo_modify format float %.17g', 'run 0',
    ]);
    expect(withFep[0].pe).toBe(plain[0].pe);
    const kT = 300; // units lj: boltz = 1
    const dU = withFep[0]['c_FEP[1]'] as number;
    expect(withFep[0]['c_FEP[2]'] as number).toBeCloseTo(Math.exp(-dU / kT), 12);
    expect(withFep[0]['c_FEP[3]'] as number).toBeGreaterThan(0);
  });

  it('compute fep rejects tail yes without a tail correction and an unknown keyword with the native text', async () => {
    await expect(runLines([...base, 'compute FEP all fep 300.0 pair lj/cut/coul/long/soft lambda * * v_dl tail yes', 'run 0'])).rejects.toThrow(/tail/);
    await expect(runLines([...base, 'compute FEP all fep 300.0 pair lj/cut/coul/long/soft lambda * * v_dl bogus yes', 'run 0'])).rejects.toThrow('Illegal optional keyword in compute fep');
  });

  it('fix adapt/fep names the keywords it does not support, and gives native texts for kspace and diameter', async () => {
    await expect(runLines([...base, 'fix A all adapt/fep 10 kspace x', 'run 0'])).rejects.toThrow(/Illegal fix adapt\/fep command/);
    await expect(runLines([...base, 'variable x equal 1.0', 'fix A all adapt/fep 10 atom diameter 1 v_x', 'run 0'])).rejects.toThrow(/Fix adapt\/fep requires atom attribute diameter/);
    await expect(runLines([...base, 'fix A all adapt/fep 10 pair lj/cut/coul/long/soft lambda * * v_dl bogus yes', 'run 0'])).rejects.toThrow('Illegal fix adapt/fep command');
  });

  it('fix adapt/fep reset yes restores the pair lambda after the run', async () => {
    const rows = await runLines([
      ...base, 'variable lam equal 0.5', 'fix A all adapt/fep 1 pair lj/cut/coul/long/soft lambda * * v_lam reset yes',
      'thermo_style custom step pe', 'thermo_modify format float %.17g', 'run 2', 'unfix A', 'run 0',
    ]);
    // after the run with the adapted lambda, the pe of the restored state equals the pe of the original state
    const orig = await runLines([...base, 'thermo_style custom step pe', 'thermo_modify format float %.17g', 'run 0']);
    expect(rows[rows.length - 1].pe).toBe(orig[0].pe);
  });
});
