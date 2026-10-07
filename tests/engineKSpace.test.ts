import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { UNIT_SYSTEMS } from '../src/engine/units';
import type { EngineError, EngineEvent } from '../src/engine/types';

/*
 * Long-range Coulombics (kspace_style ewald / pppm) checked against physics
 * the engine does not share code with: the rocksalt Madelung constant and
 * finite differences of the Ewald energy. Bit-level agreement with native
 * LAMMPS is covered by the ewald_* and pppm_* oracle cases
 * (engineOracle.test.ts).
 */

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { thermo, logs, error };
};

/** 3x3x3 rocksalt cells, a = 5.64 A, charges +-1: 216 ions, nearest-neighbour distance 2.82 A. */
const nacl = (kspace: string) => `
units           metal
atom_style      charge
lattice         fcc 5.64
region          box block 0 3 0 3 0 3
create_box      2 box
create_atoms    1 box
lattice         fcc 5.64 origin 0.5 0.0 0.0
create_atoms    2 box
mass            * 1.0
set             type 1 charge 1.0
set             type 2 charge -1.0
pair_style      coul/long 8.0
pair_coeff      * *
${kspace}
thermo_style    custom step pe ecoul elong
run             0
`;

// Madelung constant of rocksalt (OEIS A085469)
const MADELUNG = 1.7475645946331822;
const madelungEnergy = () => -(216 / 2) * MADELUNG * UNIT_SYSTEMS.metal.qqr2e / 2.82;

describe('kspace: rocksalt Madelung energy', () => {
  it('ewald reproduces the Madelung constant', async () => {
    const { thermo, error } = await run(nacl('kspace_style ewald 1.0e-10'));
    expect(error).toBeNull();
    const e = thermo[0].ecoul + thermo[0].elong;
    expect(Math.abs(e / madelungEnergy() - 1)).toBeLessThan(1e-8);
    expect(thermo[0].pe).toBeCloseTo(e, 9);
  });

  it('pppm agrees with it to its requested accuracy', async () => {
    const { thermo, error } = await run(nacl('kspace_style pppm 1.0e-6'));
    expect(error).toBeNull();
    const e = thermo[0].ecoul + thermo[0].elong;
    expect(Math.abs(e / madelungEnergy() - 1)).toBeLessThan(1e-5);
  });

  it('pppm error falls with assignment order 2 through 7', async () => {
    // g rc = 4 keeps the real-space truncation (erfc 4 ~ 1.5e-8) below the mesh error
    const err: number[] = [];
    for (const order of [2, 3, 4, 5, 6, 7]) {
      const { thermo, error } = await run(nacl(`kspace_style pppm 1.0e-6\nkspace_modify gewald 0.5 mesh 48 48 48 order ${order}`));
      expect(error, `order ${order}`).toBeNull();
      err.push(Math.abs((thermo[0].ecoul + thermo[0].elong) / madelungEnergy() - 1));
    }
    for (let i = 1; i < err.length; i++) expect(err[i], `order ${i + 2}`).toBeLessThan(err[i - 1]);
    expect(err[0]).toBeLessThan(1e-2);
    expect(err[5]).toBeLessThan(5e-8);
  });
});

describe('kspace: ewald forces are the energy gradient', () => {
  it('matches central differences for every component of a charged atom', async () => {
    const base = `
units           real
atom_style      charge
region          box block 0 10 0 9 0 11
create_box      2 box
create_atoms    1 single 1.1 2.2 3.3 units box
create_atoms    2 single 4.9 2.7 1.2 units box
create_atoms    1 single 7.3 6.1 8.8 units box
create_atoms    2 single 2.4 7.7 6.5 units box
create_atoms    1 single 8.8 0.6 4.4 units box
create_atoms    2 single 5.5 5.0 9.9 units box
mass            * 1.0
set             type 1 charge 0.7
set             type 2 charge -0.6
pair_style      coul/long 4.5
pair_coeff      * *
kspace_style    ewald 1.0e-10
group           one id 3
thermo_style    custom step pe
run             0
print           "F $(fx[3]:%.17g) $(fy[3]:%.17g) $(fz[3]:%.17g)"
`;
    const h = 1e-5;
    const probe = ['x', 'y', 'z'].map((_, d) => {
      const v = (s: number) => [0, 1, 2].map((k) => (k === d ? s * h : 0)).join(' ');
      return `
displace_atoms  one move ${v(1)} units box
run             0
print           "EP $(pe:%.17g)"
displace_atoms  one move ${v(-2)} units box
run             0
print           "EM $(pe:%.17g)"
displace_atoms  one move ${v(1)} units box
`;
    });
    const { logs, error } = await run(base + probe.join(''));
    expect(error).toBeNull();
    const nums = (tag: string) => logs.filter((l) => l.startsWith(`${tag} `)).map((l) => l.trim().split(/\s+/).slice(1).map(Number));
    const f = nums('F')[0];
    const ep = nums('EP').map((r) => r[0]), em = nums('EM').map((r) => r[0]);
    expect(ep).toHaveLength(3);
    for (let d = 0; d < 3; d++) {
      const fd = -(ep[d] - em[d]) / (2 * h);
      expect(Math.abs(f[d] - fd)).toBeLessThan(1e-6 * Math.max(1, Math.abs(f[d])));
    }
  });
});

describe('kspace: errors name the problem', () => {
  const cell = (boundary: string, pair: string, kspace: string, style = 'charge') => `
units           real
atom_style      ${style}
boundary        ${boundary}
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 2 2 2 units box
create_atoms    1 single 6 6 6 units box
mass            1 1.0
${style === 'charge' ? 'set             atom 1 charge 1.0\nset             atom 2 charge -1.0' : ''}
pair_style      ${pair}
pair_coeff      * * ${pair.startsWith('lj') ? '0.1 3.0' : ''}
${kspace}
run             0
`;
  const cases: [string, string, string, RegExp][] = [
    ['p p f', 'coul/long 4.0', 'kspace_style pppm 1e-4', /fully periodic box/],
    ['p p f', 'coul/long 4.0', 'kspace_style ewald 1e-4\nkspace_modify slab 0.5', /slab volfactor must be >= 1.0/],
    ['p p p', 'coul/long 4.0', 'kspace_style pppm 1e-4\nkspace_modify mesh 7 8 8', /factor into 2, 3 and 5/],
    ['p p p', 'coul/long 4.0', 'kspace_style pppm 1e-4\nkspace_modify order 8', /from 2 to 7/],
    ['p p p', 'coul/long 4.0', 'kspace_style ewald 1e-4\nkspace_modify mesh 8 8 8', /applies to pppm/],
    ['p p p', 'coul/long 4.0', 'kspace_style pppm 1e-4\nkspace_modify kmax/ewald 4 4 4', /applies to ewald/],
    ['p p p', 'coul/long 4.0', '', /needs a kspace style/],
    ['p p p', 'lj/cut 4.0', 'kspace_style ewald 1e-4', /not compatible with pair style lj\/cut/],
    ['p p p', 'coul/long 4.0', 'kspace_style msm 1e-4', /msm/],
  ];
  for (const [boundary, pair, kspace, re] of cases) {
    it(`${kspace.replace(/\n/g, ' / ') || 'no kspace'} (${boundary}, ${pair})`, async () => {
      const { error } = await run(cell(boundary, pair, kspace));
      expect(error, 'expected an error').not.toBeNull();
      expect(error!.message).toMatch(re);
    });
  }
  it('rejects a kspace solver without per-atom charges', async () => {
    const { error } = await run(cell('p p p', 'lj/cut 4.0', 'kspace_style ewald 1e-4', 'atomic'));
    expect(error?.message).toMatch(/needs charges|not compatible/);
  });
});
