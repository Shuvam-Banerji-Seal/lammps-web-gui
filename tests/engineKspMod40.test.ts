import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent } from '../src/engine/types';

/*
 * kspace_modify slab auto / triclinic slab / communication keywords, kspace_style zero, pair_style
 * zero/coul, neigh_modify bin/hash: parsing, accepted values and refusals. The numbers that these
 * refusals and volfactors rest on were measured with native LAMMPS (black box); the energies are
 * checked against native in the w40ksp_* oracle cases (engineOracle.test.ts).
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

/** A charged film in a 5 x 5 x 4 lattice box (sc 3.2 A), the geometry of the ewald_slab oracle case. */
const film = (opts: { units?: string; tilt?: string; kspace: string; neigh?: string; pair?: string; boundary?: string }) => `
units           ${opts.units ?? 'real'}
atom_style      charge
boundary        ${opts.boundary ?? 'p p f'}
lattice         sc 3.2
region          box block 0 5 0 5 0 4 units lattice
create_box      2 box
${opts.tilt ?? ''}
region          film block INF INF INF INF 0.5 3.5 units lattice
create_atoms    1 region film
mass            * 18.0
set             type 1 charge 0.5
set             type 2 charge -0.5
set             group all type 1
${opts.neigh ?? ''}
pair_style      ${opts.pair ?? 'lj/cut/coul/long 6.0'}
pair_coeff      * * ${(opts.pair ?? 'lj/cut/coul/long 6.0').startsWith('zero') ? '' : '0.15 2.9'}
${opts.kspace}
thermo_style    custom step pe ecoul elong
run             0
`;

describe('kspace_modify slab auto', () => {
  it('chooses the volfactor from the tolerance and the lateral size (measured: 3.2904237 at 1e-5)', async () => {
    const { error, logs } = await run(film({ kspace: 'kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 kmax/ewald 7 7 16 slab auto' }));
    expect(error).toBeNull();
    const line = logs.join('\n').match(/auto slab volfactor = ([0-9.e+-]+)/);
    expect(line).not.toBeNull();
    expect(Number(line![1])).toBeCloseTo(3.2904237, 6);
  });

  it('uses the force value when given (measured: 3.0711026 for force 0.01 in real units)', async () => {
    const { logs, error } = await run(film({ kspace: 'kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 kmax/ewald 7 7 16 slab auto force 0.01' }));
    expect(error).toBeNull();
    expect(Number(logs.join('\n').match(/auto slab volfactor = ([0-9.e+-]+)/)![1])).toBeCloseTo(3.0711026, 6);
  });

  it('works for pppm too (same volfactor, measured)', async () => {
    const { logs, error } = await run(film({ kspace: 'kspace_style pppm 1.0e-5\nkspace_modify gewald 0.4 mesh 16 16 36 order 5 slab auto' }));
    expect(error).toBeNull();
    expect(Number(logs.join('\n').match(/auto slab volfactor = ([0-9.e+-]+)/)![1])).toBeCloseTo(3.2904237, 6);
  });

  it('refuses a tolerance of 1 or more (native: must be between 0 and 1)', async () => {
    const { error } = await run(film({ kspace: 'kspace_style ewald 1.0\nkspace_modify gewald 0.4 slab auto' }));
    expect(error?.message).toMatch(/requires a normalized force tolerance between 0 and 1/);
  });

  it('refuses tolerances above 0.35, where the native value departs from the formula', async () => {
    const { error } = await run(film({ kspace: 'kspace_style ewald 0.4\nkspace_modify gewald 0.4 slab auto' }));
    expect(error?.message).toMatch(/reproduced only for a normalized force tolerance up to 0.35/);
  });

  it('refuses lj units (the native value does not follow the real and metal mapping)', async () => {
    const { error } = await run(film({ units: 'lj', kspace: 'kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 slab auto' }));
    expect(error?.message).toMatch(/measured for units real and metal only/);
  });
});

describe('kspace_modify slab value checks', () => {
  it('refuses a volfactor of exactly 1.0 as native does', async () => {
    const { error } = await run(film({ kspace: 'kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 slab 1.0' }));
    expect(error?.message).toMatch(/Bad kspace_modify slab parameter/);
  });

  it('accepts a volfactor above 1.0', async () => {
    const { error } = await run(film({ kspace: 'kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 slab 3.0' }));
    expect(error).toBeNull();
  });
});

describe('triclinic slab', () => {
  const tilt = 'change_box all triclinic\nchange_box all xy final 2.0';
  it('accepts an xy tilt for ewald and pppm (measured native runs)', async () => {
    expect((await run(film({ tilt, kspace: 'kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 kmax/ewald 7 7 16 slab 3.0' }))).error).toBeNull();
    expect((await run(film({ tilt, kspace: 'kspace_style pppm 1.0e-5\nkspace_modify gewald 0.4 mesh 16 16 36 order 5 slab 3.0' }))).error).toBeNull();
  });

  it('refuses an xz or yz tilt with the native message', async () => {
    const { error } = await run(film({
      tilt: 'change_box all triclinic\nchange_box all xy final 2.0 xz final 1.5',
      kspace: 'kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 slab 3.0',
    }));
    expect(error?.message).toMatch(/Triclinic slab \(EW3DC\) correction requires xz = yz = 0/);
  });
});

describe('kspace_modify communication keywords', () => {
  const ew = (mods: string) => `kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 kmax/ewald 7 7 16 slab 3.0 ${mods}`;
  it('accepts nonblocking and collective with yes or no', async () => {
    expect((await run(film({ kspace: ew('nonblocking yes collective no') }))).error).toBeNull();
  });
  it('accepts collective/self/copy with yes, no and onerank', async () => {
    for (const v of ['yes', 'no', 'onerank']) {
      expect((await run(film({ kspace: ew(`collective/self/copy ${v}`) }))).error).toBeNull();
    }
  });
  it('refuses a bad yes/no value (native: Expected boolean parameter)', async () => {
    expect((await run(film({ kspace: ew('nonblocking maybe') }))).error?.message).toMatch(/Expected boolean parameter instead of 'maybe'/);
    expect((await run(film({ kspace: ew('collective maybe') }))).error?.message).toMatch(/Expected boolean parameter instead of 'maybe'/);
  });
  it('refuses a bad collective/self/copy flag (native message)', async () => {
    expect((await run(film({ kspace: ew('collective/self/copy maybe') }))).error?.message).toMatch(/Unknown kspace_modify collective\/self\/copy flag maybe/);
  });
});

describe('kspace_style zero', () => {
  const zero = (args: string, extra = '') => film({ kspace: `kspace_style zero ${args}\n${extra}`, pair: 'lj/cut/coul/long 6.0' });
  it('needs exactly one accuracy argument (native: Illegal kspace_style zero command)', async () => {
    expect((await run(zero(''))).error?.message).toMatch(/Illegal kspace_style zero command/);
  });
  it('rejects a non-numeric accuracy', async () => {
    expect((await run(zero('abc'))).error?.message).toMatch(/expected a number for accuracy/);
  });
  it('an accuracy of 0 is refused only when G-ewald must come from it (native: KSpace accuracy must be > 0)', async () => {
    expect((await run(zero('0.0'))).error?.message).toMatch(/KSpace accuracy must be > 0/);
    expect((await run(zero('0.0', 'kspace_modify gewald 0.4'))).error).toBeNull();
  });
  it('gives zero long-range energy (measured: E_long 0 with lj/cut/coul/long)', async () => {
    const { error, thermo } = await run(zero('1.0e-5'));
    expect(error).toBeNull();
    expect(thermo[0].elong).toBe(0);
  });
  it('is refused with a pair style that has no Coulomb cutoff (native: KSpace style is incompatible with Pair style)', async () => {
    const { error } = await run(film({ pair: 'lj/cut 6.0', kspace: 'kspace_style zero 1.0e-5' }));
    // Measured with native LAMMPS: KSpace style is incompatible with Pair style (src/kspace_zero.cpp:74)
    expect(error?.message).toMatch(/KSpace style is incompatible with Pair style/);
  });
  it('refuses kspace_modify slab for zero', async () => {
    const { error } = await run(zero('1.0e-5', 'kspace_modify slab 3.0'));
    expect(error?.message).toMatch(/kspace_modify slab is not supported by kspace_style zero/);
  });
});

describe('pair_style zero/coul', () => {
  const zc = (kspace: string) => film({ pair: 'zero/coul 6.0', kspace });
  it('gives zero real-space energy and the k-space part with ewald (measured E_long -1288.45251918596 for the base film)', async () => {
    const { error, thermo } = await run(zc('kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 kmax/ewald 7 7 16 slab 3.0'));
    expect(error).toBeNull();
    expect(thermo[0].ecoul).toBe(0);
    expect(thermo[0].elong).toBeLessThan(0);
  });
  it('accepts pair_modify shift and tail with no effect (measured)', async () => {
    expect((await run(zc('pair_modify shift yes\nkspace_style zero 1.0e-5'))).error).toBeNull();
    expect((await run(zc('pair_modify tail yes\nkspace_style zero 1.0e-5'))).error).toBeNull();
  });
  it('is refused with a dispersion or TIP4P kspace style (native: KSpace style ... is incompatible with Pair style zero/coul)', async () => {
    expect((await run(zc('kspace_style pppm/tip4p 1.0e-5\nkspace_modify gewald 0.4'))).error?.message).toMatch(/KSpace style pppm\/tip4p is incompatible with Pair style zero\/coul/);
  });
  it('plain pair_style zero is still incompatible with a kspace style', async () => {
    const { error } = await run(film({ pair: 'zero 6.0', kspace: 'kspace_style ewald 1.0e-5' }));
    // Measured with native LAMMPS: KSpace style ewald is incompatible with Pair style zero (src/kspace.cpp:215)
    expect(error?.message).toMatch(/KSpace style ewald is incompatible with Pair style zero/);
  });
});

describe('neigh_modify bin/hash', () => {
  const nb = (neigh: string) => film({ neigh, kspace: 'kspace_style ewald 1.0e-5\nkspace_modify gewald 0.4 kmax/ewald 7 7 16 slab 3.0' });
  it('needs the multi neighbor style (native: Cannot use bin/hash command without multi setting)', async () => {
    expect((await run(nb('neigh_modify bin/hash yes'))).error?.message).toMatch(/Cannot use bin\/hash command without multi setting/);
  });
  it('is accepted with neighbor multi, and with no as the default', async () => {
    expect((await run(nb('neighbor 2.0 multi\nneigh_modify bin/hash yes'))).error).toBeNull();
    expect((await run(nb('neigh_modify bin/hash no'))).error).toBeNull();
  });
  it('refuses a value other than yes or no', async () => {
    expect((await run(nb('neighbor 2.0 multi\nneigh_modify bin/hash maybe'))).error?.message).toMatch(/Expected boolean parameter instead of 'maybe'/);
  });
});

describe('unknown kspace styles', () => {
  it('are named in the error', async () => {
    const { error } = await run(film({ kspace: 'kspace_style nosuch 1.0e-5' }));
    expect(error?.message).toMatch(/nosuch/);
  });
});
