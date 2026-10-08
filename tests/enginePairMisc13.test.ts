import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Unit tests of the wave-13 pair styles (pair_style vashishta/table, src/engine/force/pair/vashishta_table.ts).
 * Forces are checked against the energy by central finite differences: the analytic style must agree
 * closely; the tabulated style agrees to the accuracy of its linear interpolation (the table forces are
 * interpolated separately from the table energies, so the check uses a fine table).
 */

const ORACLE = join(__dirname, 'oracle');
const POT = readFileSync(join(ORACLE, 'w13pair_vashtable.vashishta'), 'utf8');

interface Atom { type: number; x: [number, number, number] }

/** Energy and per-atom forces (in atom id order) of a small non-periodic system. */
async function evaluate(pairLines: string[], atoms: Atom[], potFile = POT): Promise<{ pe: number; f: number[][] }> {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  session.addFile('pot.vashishta', potFile);
  const text = [
    'units metal',
    'atom_style atomic',
    'boundary f f f',
    'atom_modify map yes sort 0 0',
    'region box block -30 30 -30 30 -30 30',
    'create_box 2 box',
    ...atoms.map((a) => `create_atoms ${a.type} single ${a.x.join(' ')}`),
    'mass * 28.0',
    ...pairLines,
    'thermo_style custom step pe',
    'run 0',
    'write_dump all custom dump13.txt id fx fy fz modify format float %.17g sort id',
  ].join('\n');
  await session.execute(text);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo');
  const pe = rows[rows.length - 1].row.pe;
  const lines = (files.get('dump13.txt') ?? '').trim().split('\n');
  const k0 = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const f = lines.slice(k0 + 1).map((l) => l.trim().split(/\s+/).slice(1).map(Number));
  return { pe, f };
}

/** Four atoms of two elements; separations 2.0 to 3.5 lie inside the table range (cutinner 1.5). */
const SYSTEM: Atom[] = [
  { type: 1, x: [0, 0, 0] },
  { type: 2, x: [2.2, 0.4, 0.1] },
  { type: 2, x: [0.3, 2.5, -0.5] },
  { type: 1, x: [-1.2, -1.1, 2.1] },
];

/** Force on atom `which` along x from central differences of the energy. */
async function fdForceX(pairLines: string[], which: number, h = 1e-5): Promise<number> {
  const plus = structuredClone(SYSTEM);
  const minus = structuredClone(SYSTEM);
  plus[which].x[0] += h;
  minus[which].x[0] -= h;
  const ep = (await evaluate(pairLines, plus)).pe;
  const em = (await evaluate(pairLines, minus)).pe;
  return -(ep - em) / (2 * h);
}

describe('pair_style vashishta/table arguments', () => {
  const coeff = 'pair_coeff * * pot.vashishta A B';
  const bad: [string, RegExp][] = [
    ['pair_style vashishta/table 11', /usage: pair_style vashishta\/table Ntable cutinner/],
    ['pair_style vashishta/table 11 1.5 2', /usage: pair_style vashishta\/table Ntable cutinner/],
    ['pair_style vashishta/table 11.5 1.5', /integer/],
    ['pair_style vashishta/table 1 1.5', /Ntable >= 2/],
    ['pair_style vashishta/table 11 0', /inner cutoff/],
    ['pair_style vashishta/table 11 -1', /inner cutoff/],
    ['pair_style vashishta/table 11 abc', /number/],
  ];
  for (const [line, re] of bad) {
    it(`rejects '${line}'`, async () => {
      await expect(evaluate([line, coeff], SYSTEM)).rejects.toThrow(re);
    });
  }

  it('rejects an element that is not in the potential file', async () => {
    await expect(evaluate(['pair_style vashishta/table 11 1.5', 'pair_coeff * * pot.vashishta A C'], SYSTEM)).rejects.toThrow(/not in/);
  });

  it('rejects a pair_coeff with the wrong number of element names', async () => {
    await expect(evaluate(['pair_style vashishta/table 11 1.5', 'pair_coeff * * pot.vashishta A'], SYSTEM)).rejects.toThrow(/usage/);
  });

  it('rejects a pair_coeff whose first two arguments are not * *', async () => {
    await expect(evaluate(['pair_style vashishta/table 11 1.5', 'pair_coeff 1 * pot.vashishta A B'], SYSTEM)).rejects.toThrow(/\* \*/);
  });

  it('reports a missing potential file', async () => {
    await expect(evaluate(['pair_style vashishta/table 11 1.5', 'pair_coeff * * nope.vashishta A B'], SYSTEM)).rejects.toThrow();
  });
});

describe('vashishta/table forces are -grad E', () => {
  const analytic = ['pair_style vashishta', 'pair_coeff * * pot.vashishta A B'];
  const table = (n: number) => [`pair_style vashishta/table ${n} 1.5`, 'pair_coeff * * pot.vashishta A B'];

  it('analytic vashishta: force on each atom equals the central difference of the energy', async () => {
    const r = await evaluate(analytic, SYSTEM);
    for (const which of [1, 2, 3]) {
      const fd = await fdForceX(analytic, which);
      const f = r.f[which][0];
      expect(Math.abs(f - fd), `atom ${which + 1} fx ${f} vs FD ${fd}`).toBeLessThan(1e-6 * Math.max(1, Math.abs(fd)));
    }
  });

  it('vashishta/table with 200 points: force agrees with the central difference of the table energy', async () => {
    const pl = table(200);
    const r = await evaluate(pl, SYSTEM);
    for (const which of [1, 2, 3]) {
      const fd = await fdForceX(pl, which);
      const f = r.f[which][0];
      expect(Math.abs(f - fd), `atom ${which + 1} fx ${f} vs FD ${fd}`).toBeLessThan(2e-2 * Math.max(1, Math.abs(fd)));
    }
  });

  it('vashishta/table energy with a cutinner beyond every pair distance equals the analytic energy', async () => {
    const a = await evaluate(analytic, SYSTEM);
    const t = await evaluate(['pair_style vashishta/table 50 4.5', 'pair_coeff * * pot.vashishta A B'], SYSTEM);
    expect(Math.abs(t.pe - a.pe)).toBeLessThan(1e-12 * Math.max(1, Math.abs(a.pe)));
  });
});

describe('vashishta/table node range', () => {
  // Measured with native LAMMPS (black box): an A-A dimer (type 1 only, A and B mapped), Ntable 30,
  // cutinner 1.5. The nodes span up to the largest cutoff of the mapped A/B entries (4.22), not the
  // A-A entry's own 4.16, so these energies sit between nodes of that spacing.
  const NATIVE: [number, number][] = [
    [1.6, 5.3541252280217169],
    [2.3, 1.658750962823226],
    [3.1, 0.38291848046481242],
    [3.9, 0.018089535511101592],
    [4.1, 0.0010341891613000163],
  ];
  for (const [r, pe] of NATIVE) {
    it(`type-1 dimer at r = ${r} matches native`, async () => {
      const t = await evaluate(['pair_style vashishta/table 30 1.5', 'pair_coeff * * pot.vashishta A B'], [
        { type: 1, x: [0, 0, 0] },
        { type: 1, x: [r, 0, 0] },
      ]);
      expect(Math.abs(t.pe - pe)).toBeLessThan(1e-9 * Math.max(1, Math.abs(pe)));
    });
  }
});
