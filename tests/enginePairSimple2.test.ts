import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * pair styles soft / yukawa / gauss / zero: force = -dE/dr by central finite
 * differences at three geometries, plus missing-coefficient error paths.
 * Agreement with native LAMMPS is covered by the w1misc_* oracle cases.
 */

const run = async (text: string) => {
  const rows: ThermoRow[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); },
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  return { rows, error, files };
};

/** Two atoms on the x axis at distance r; returns pe and the x-force on atom 2. */
const probe = async (pairLines: string, r: number) => {
  const text = `
units           lj
atom_style      atomic
region          box block 0 10 0 10 0 10
create_box      1 box
mass            1 1.0
create_atoms    1 single 5 5 5
create_atoms    1 single ${5 + r} 5 5
${pairLines}
thermo_style    custom step pe
thermo_modify   format float %.15g
run             0
write_dump      all custom fd.dump id fx fy fz modify format float %.17g sort id
`;
  const { rows, error, files } = await run(text);
  expect(error, JSON.stringify(rows)).toBeNull();
  const dump = files.get('fd.dump') ?? '';
  const lines = dump.trim().split('\n');
  const k0 = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const atom2 = lines[k0 + 2].trim().split(/\s+/);
  const cols = lines[k0].split(/\s+/).slice(2);
  const pick = (c: string) => Number(atom2[cols.indexOf(c)]);
  return { pe: rows[0].pe, x2: pick('x'), f2x: pick('fx') };
};

describe('pair simple2: force = -dE/dr by central differences', () => {
  it('soft', async () => {
    const coeff = 'pair_style soft 1.12\npair_coeff * * 10.0';
    const h = 1e-4;
    for (const r of [0.5, 0.8, 1.05]) {
      const mid = await probe(coeff, r);
      const plus = await probe(coeff, r + h);
      const minus = await probe(coeff, r - h);
      const dEdr = (plus.pe - minus.pe) / (2 * h);
      expect(Math.abs(2 * dEdr + mid.f2x)).toBeLessThan(2e-5); // thermo pe is per-atom (2 atoms)
    }
  });

  it('yukawa', async () => {
    const coeff = 'pair_style yukawa 2.0 2.5\npair_coeff * * 20.0';
    const h = 1e-4;
    for (const r of [0.7, 1.2, 2.0]) {
      const mid = await probe(coeff, r);
      const plus = await probe(coeff, r + h);
      const minus = await probe(coeff, r - h);
      const dEdr = (plus.pe - minus.pe) / (2 * h);
      expect(Math.abs(2 * dEdr + mid.f2x)).toBeLessThan(2e-5); // thermo pe is per-atom (2 atoms)
    }
  });

  it('yukawa: shift subtracts the energy at the cutoff', async () => {
    const r = 2.0, rc = 2.5, kappa = 2.0, a = 20.0;
    const raw = await run(`
units lj
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
mass 1 1.0
create_atoms 1 single 5 5 5
create_atoms 1 single ${5 + r} 5 5
pair_style yukawa ${kappa} ${rc}
pair_coeff * * ${a}
thermo_style custom step pe
run 0
`);
    const shifted = await run(`
units lj
atom_style atomic
region box block 0 10 0 10 0 10
create_box 1 box
mass 1 1.0
create_atoms 1 single 5 5 5
create_atoms 1 single ${5 + r} 5 5
pair_style yukawa ${kappa} ${rc}
pair_coeff * * ${a}
pair_modify shift yes
thermo_style custom step pe
run 0
`);
    const eshift = (a * Math.exp(-kappa * rc) / rc) / 2; // thermo pe is per-atom (2 atoms)
    expect(shifted.rows[0].pe).toBeCloseTo(raw.rows[0].pe - eshift, 10);
  });

  it('gauss', async () => {
    const coeff = 'pair_style gauss 2.5\npair_coeff * * 1.0 0.9';
    const h = 1e-4;
    for (const r of [0.6, 1.1, 1.9]) {
      const mid = await probe(coeff, r);
      const plus = await probe(coeff, r + h);
      const minus = await probe(coeff, r - h);
      const dEdr = (plus.pe - minus.pe) / (2 * h);
      expect(Math.abs(2 * dEdr + mid.f2x)).toBeLessThan(2e-5); // thermo pe is per-atom (2 atoms)
    }
  });

  it('zero: no force, no energy', async () => {
    const coeff = 'pair_style zero 2.5\npair_coeff * *';
    for (const r of [0.7, 1.4, 2.2]) {
      const mid = await probe(coeff, r);
      const plus = await probe(coeff, r + 1e-4);
      const minus = await probe(coeff, r - 1e-4);
      expect(mid.pe).toBe(0);
      expect(mid.f2x).toBe(0);
      expect((plus.pe - minus.pe) / 2e-4).toBe(0);
    }
  });
});

describe('pair simple2: missing coefficients throw', () => {
  const cases: [string, string, RegExp][] = [
    ['soft: unset cross pair', 'pair_style soft 2.5\npair_coeff 1 1 10.0', /all pair coeffs are not set \(pair 1 2/],
    ['yukawa: unset cross pair', 'pair_style yukawa 2.0 2.5\npair_coeff 1 1 20.0', /all pair coeffs are not set \(pair 1 2/],
    ['gauss: unset cross pair', 'pair_style gauss 2.5\npair_coeff 1 1 1.0 0.9', /all pair coeffs are not set \(pair 1 2/],
    ['zero: no pair_coeff at all', 'pair_style zero 2.5', /all pair coeffs are not set/],
  ];
  for (const [what, pair, re] of cases) {
    it(what, async () => {
      const { error } = await run(`
units           lj
atom_style      atomic
region          box block 0 3 0 3 0 3
create_box      2 box
mass            1 1.0
mass            2 1.0
${pair}
run             0
`);
      expect(error, 'expected an error').not.toBeNull();
      expect(error!.message).toMatch(re);
    });
  }
});

describe('pair simple2: unsupported options are rejected', () => {
  const cases: [string, string, RegExp][] = [
    ['soft rejects shift', 'pair_style soft 2.5\npair_coeff * * 10.0\npair_modify shift yes', /shift is not supported/],
    ['yukawa rejects tail', 'pair_style yukawa 2.0 2.5\npair_coeff * * 20.0\npair_modify tail yes', /tail is not supported/],
    ['gauss rejects shift', 'pair_style gauss 2.5\npair_coeff * * 1.0 0.9\npair_modify shift yes', /shift is not supported/],
    ['gauss rejects sixthpower mix', 'pair_style gauss 2.5\npair_coeff * * 1.0 0.9\npair_modify mix sixthpower', /sixthpower is not supported/],
    ['zero rejects tail', 'pair_style zero 2.5\npair_coeff * *\npair_modify tail yes', /tail is not supported/],
    ['zero rejects unknown keyword', 'pair_style zero 2.5 bogus', /unsupported keyword 'bogus'/],
  ];
  for (const [what, pair, re] of cases) {
    it(what, async () => {
      const { error } = await run(`
units           lj
atom_style      atomic
region          box block 0 3 0 3 0 3
create_box      2 box
mass            1 1.0
mass            2 1.0
${pair}
run             0
`);
      expect(error, 'expected an error').not.toBeNull();
      expect(error!.message).toMatch(re);
    });
  }
});

describe('pair simple2: zero keywords', () => {
  it('nocoeff ignores pair_coeff values and uses the global cutoff', async () => {
    const { rows, error } = await run(`
units           lj
atom_style      atomic
region          box block 0 10 0 10 0 10
create_box      1 box
mass            1 1.0
create_atoms    1 single 5 5 5
create_atoms    1 single 6 5 5
pair_style      zero 2.5 nocoeff
pair_coeff      * * 3.0
thermo_style    custom step pe
run             0
`);
    expect(error).toBeNull();
    expect(rows[0].pe).toBe(0);
  });
});
