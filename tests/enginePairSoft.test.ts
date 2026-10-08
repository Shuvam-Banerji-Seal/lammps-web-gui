import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { PAIRS } from '../src/engine/registry/pair_soft17';
import { PairFepSoft } from '../src/engine/force/pair/fep_soft';

/*
 * Soft-core FEP pair styles (docs.lammps.org/pair_fep_soft.html, src/engine/force/pair/fep_soft.ts).
 * Two atoms of types 1 and 2 on the x axis in a periodic box; the force on atom 2 must equal
 * -dE/dr (central finite differences of the total potential energy, kspace included for the long
 * styles). The parity against native LAMMPS lives in tests/engineOracle.test.ts (w17soft_*).
 */

const BOX = 30;
const X1 = 5;

interface Run { pe: number; f2x: number; }

/** Runs one configuration through the interpreter: returns pe and the x force on atom 2. */
const evaluate = async (style: string[], r: number, charged = true): Promise<Run> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  const lines = [
    'units lj', 'atom_style charge', 'boundary p p p', `region box block 0 ${BOX} 0 ${BOX} 0 ${BOX}`, 'create_box 2 box',
    `create_atoms 1 single ${X1} ${X1} ${X1}`, `create_atoms 2 single ${X1 + r} ${X1} ${X1}`, 'mass * 1.0',
    charged ? 'set type 1 charge 1.0' : 'set type 1 charge 0.0', charged ? 'set type 2 charge -1.0' : 'set type 2 charge 0.0',
    ...style, 'thermo_style custom step pe', 'thermo_modify format float %.17g norm no', 'run 0',
    'write_dump all custom fdump.txt id fx fy fz modify format float %.17g sort id',
  ];
  await session.execute(lines.join('\n') + '\n');
  const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').pop();
  if (!row) throw new Error('no thermo row');
  const dump = (files.get('fdump.txt') ?? '').trim().split('\n');
  const k0 = dump.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const f2 = dump[k0 + 2].trim().split(/\s+/).map(Number);
  return { pe: row.row.pe as number, f2x: f2[1] };
};

const KSPACE = ['kspace_style ewald 1.0e-6', 'kspace_modify gewald 0.35 kmax/ewald 7 7 7'];

/** Force on atom 2 equals -dE/dr from central finite differences, at each separation. */
const checkForces = async (style: string[], radii: number[], charged = true): Promise<void> => {
  const h = 1e-5;
  for (const r of radii) {
    const p = await evaluate(style, r + h, charged);
    const m = await evaluate(style, r - h, charged);
    const fd = -(p.pe - m.pe) / (2 * h);
    const f = (await evaluate(style, r, charged)).f2x;
    expect(Math.abs(f - fd), `${style[0]} r=${r} force ${f} vs -dE/dr ${fd}`).toBeLessThan(1e-7 * (1 + Math.abs(fd)));
  }
};

const LJ_COEFFS = ['pair_coeff 1 1 1.0 1.0 0.5', 'pair_coeff 2 2 0.8 1.1 1.0', 'pair_coeff 1 2 0.9 1.05 0.75'];
const LJ_COEFFS_CUT = ['pair_coeff 1 1 1.0 1.0 0.5 2.6', 'pair_coeff 2 2 0.8 1.1 1.0 2.8', 'pair_coeff 1 2 0.9 1.05 0.75 2.7 2.9'];
const COUL_LAM = ['pair_coeff * * 0.6'];

describe('soft-core FEP pair styles: forces are -dE/dr', () => {
  it('is registered under every documented name the engine implements', () => {
    for (const name of ['lj/cut/soft', 'lj/cut/coul/cut/soft', 'lj/cut/coul/long/soft', 'coul/cut/soft', 'coul/long/soft',
      'lj/class2/soft', 'lj/class2/coul/cut/soft', 'lj/class2/coul/long/soft', 'lj/charmm/coul/long/soft']) {
      expect(PAIRS[name], name).toBeDefined();
      expect(PAIRS[name]()).toBeInstanceOf(Object);
    }
    expect(PAIRS['lj/cut/soft']()).toBeInstanceOf(PairFepSoft);
  });

  it('lj/cut/soft (lambda 0.5, 1.0 and explicit cross term, shift yes)', async () => {
    await checkForces(['pair_style lj/cut/soft 2 0.5 3.0', 'pair_modify shift yes', ...LJ_COEFFS], [0.95, 1.3, 2.0, 2.6], false);
  });

  it('lj/cut/soft with n = 1, alpha_LJ = 0.3 and tail yes (forces unaffected by the tail)', async () => {
    await checkForces(['pair_style lj/cut/soft 1 0.3 3.0', 'pair_modify tail yes', ...LJ_COEFFS], [1.0, 1.7, 2.5], false);
  });

  it('lj/cut/coul/cut/soft (charged pair, LJ and Coulomb cutoffs)', async () => {
    await checkForces(['pair_style lj/cut/coul/cut/soft 2 0.5 4.0 3.0 3.4', ...LJ_COEFFS_CUT], [0.95, 1.4, 2.2, 3.1]);
  });

  it('lj/cut/coul/long/soft with Ewald kspace (real space plus reciprocal part)', async () => {
    await checkForces(['pair_style lj/cut/coul/long/soft 2 0.5 4.0 3.0', ...LJ_COEFFS_CUT.map((c) => c.replace(/ 2\.9$/, '')), ...KSPACE], [0.95, 1.4, 2.2, 2.8]);
  });

  it('coul/cut/soft (lambda 0.6, no repulsive core)', async () => {
    await checkForces(['pair_style coul/cut/soft 2 4.0 3.0', ...COUL_LAM], [0.7, 1.2, 2.0, 2.8]);
  });

  it('coul/long/soft with Ewald kspace', async () => {
    await checkForces(['pair_style coul/long/soft 2 4.0 3.0', ...COUL_LAM, ...KSPACE], [0.7, 1.2, 2.0, 2.8]);
  });

  it('lj/class2/soft (9-6, sixthpower mix)', async () => {
    await checkForces(['pair_style lj/class2/soft 2 0.5 3.0', 'pair_modify shift yes', ...LJ_COEFFS], [0.95, 1.3, 2.0, 2.6], false);
  });

  it('lj/class2/coul/cut/soft', async () => {
    await checkForces(['pair_style lj/class2/coul/cut/soft 2 0.5 4.0 3.0 3.4', ...LJ_COEFFS_CUT], [0.95, 1.4, 2.2, 3.1]);
  });

  it('lj/class2/coul/long/soft with Ewald kspace', async () => {
    await checkForces(['pair_style lj/class2/coul/long/soft 2 0.5 4.0 3.0', ...LJ_COEFFS_CUT.map((c) => c.replace(/ 2\.9$/, '')), ...KSPACE], [0.95, 1.4, 2.2, 2.8]);
  });

  it('lj/charmm/coul/long/soft: forces inside the switching range [8, 10)', async () => {
    await checkForces(['pair_style lj/charmm/coul/long/soft 2 0.5 4.0 8.0 10.0', 'pair_coeff * * 1.0 1.0 0.6', ...KSPACE], [1.0, 2.5, 8.5, 9.4]);
  });
});

describe('soft-core FEP pair styles: limits and mixing', () => {
  it('lambda = 1 reproduces the standard lj/cut energy', async () => {
    const soft = await evaluate(['pair_style lj/cut/soft 2 0.5 3.0', 'pair_coeff * * 1.0 1.0 1.0'], 1.3, false);
    const std = await evaluate(['pair_style lj/cut 3.0', 'pair_coeff * * 1.0 1.0'], 1.3, false);
    expect(soft.pe).toBeCloseTo(std.pe, 13);
  });

  it('lambda = 0 switches the Lennard-Jones interaction off', async () => {
    const off = await evaluate(['pair_style lj/cut/soft 2 0.5 3.0', 'pair_coeff * * 1.0 1.0 0.0'], 1.3, false);
    expect(off.pe).toBe(0);
  });

  it('a soft core stays finite at r = 0 separation of the cores (lambda < 1)', async () => {
    const r = await evaluate(['pair_style lj/cut/soft 2 0.5 3.0', 'pair_coeff * * 1.0 1.0 0.5'], 0.05, false);
    expect(Number.isFinite(r.pe)).toBe(true);
    expect(Number.isFinite(r.f2x)).toBe(true);
  });

  it('mixing a cross term between types with different lambda is a StyleError (native behaviour)', async () => {
    await expect(evaluate(['pair_style lj/cut/soft 2 0.5 3.0', 'pair_coeff 1 1 1.0 1.0 0.5', 'pair_coeff 2 2 1.0 1.0 1.0'], 1.3, false))
      .rejects.toThrow(/different lambda/);
  });

  it('mixing with equal lambda is accepted (geometric epsilon and sigma)', async () => {
    const r = await evaluate(['pair_style lj/cut/soft 2 0.5 3.0', 'pair_coeff 1 1 1.0 1.0 0.6', 'pair_coeff 2 2 4.0 1.4 0.6'], 1.3, false);
    expect(Number.isFinite(r.pe)).toBe(true);
  });
});

describe('soft-core FEP pair styles: unsupported options are StyleErrors', () => {
  it('tail yes is rejected for the class2 and Coulomb-only styles', async () => {
    await expect(evaluate(['pair_style lj/class2/soft 2 0.5 3.0', 'pair_modify tail yes', ...LJ_COEFFS], 1.3, false)).rejects.toThrow(/tail/);
    await expect(evaluate(['pair_style coul/cut/soft 2 4.0 3.0', 'pair_modify tail yes', ...COUL_LAM], 1.3)).rejects.toThrow(/tail/);
  });

  it('shift yes is rejected for the Coulomb-only styles', async () => {
    await expect(evaluate(['pair_style coul/cut/soft 2 4.0 3.0', 'pair_modify shift yes', ...COUL_LAM], 1.3)).rejects.toThrow(/shift/);
  });

  it('the charmm soft style rejects the 1-4 parameters and pair_modify shift', async () => {
    await expect(evaluate(['pair_style lj/charmm/coul/long/soft 2 0.5 4.0 8.0 10.0', 'pair_coeff * * 1.0 1.0 0.6 0.14 3.1', ...KSPACE], 1.3)).rejects.toThrow(/epsilon14/);
    await expect(evaluate(['pair_style lj/charmm/coul/long/soft 2 0.5 4.0 8.0 10.0', 'pair_modify shift yes', 'pair_coeff * * 1.0 1.0 0.6', ...KSPACE], 1.3)).rejects.toThrow(/shift/);
  });

  it('the long styles reject a per-pair Coulomb cutoff (native behaviour)', async () => {
    await expect(evaluate(['pair_style lj/cut/coul/long/soft 2 0.5 4.0 3.0', 'pair_coeff * * 1.0 1.0 0.5 2.6 2.9', ...KSPACE], 1.3)).rejects.toThrow(/pair_coeff/);
    await expect(evaluate(['pair_style coul/long/soft 2 4.0 3.0', 'pair_coeff * * 0.6 2.5', ...KSPACE], 1.3)).rejects.toThrow(/pair_coeff/);
  });

  it('tip4p and morse soft styles are StyleErrors that name the style', async () => {
    await expect(evaluate(['pair_style morse/soft 4 0.9 10.0', 'pair_coeff * * 100.0 2.0 1.5 1.0'], 1.3, false)).rejects.toThrow(/morse\/soft/);
    await expect(evaluate(['pair_style tip4p/long/soft 1 2 7 8 0.15 2.0 0.5 10.0', 'pair_coeff * * 1.0'], 1.3)).rejects.toThrow(/tip4p\/long\/soft/);
  });
});
