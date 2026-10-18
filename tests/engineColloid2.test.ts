import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// COLLOID styles (src/engine-gpl/colloid2): pair lubricate, lubricate/poly, lubricateU, lubricateU/poly, brownian,
// brownian/poly, fix wall/colloid and fix wall/srd. The native parity cases are tests/oracle/w231co_*; these tests
// check the setup errors and refusals, with the texts the native code raises (src/COLLOID, src/SRD at b332f28).

/** A small box of 64 sphere atoms (or point atoms for atomic), with the extra lines appended. */
const box = (extra: string, atomStyle = 'sphere'): string => `units lj
atom_style ${atomStyle}
boundary p p p
region box block 0 4 0 4 0 4
create_box 1 box
lattice sc 0.072338
create_atoms 1 box
${atomStyle === 'sphere' ? 'set type 1 diameter 2.0\nset type 1 mass 4.18879' : 'mass 1 1.0'}
velocity all create 0.5 87287 loop geom
${extra}
`;

/** Two atom types of different diameters (32 atoms each), for the monodisperse checks. */
const twoSizes = (extra: string): string => `units lj
atom_style sphere
boundary p p p
region box block 0 4 0 4 0 4
create_box 2 box
lattice sc 0.072338
create_atoms 1 box
group half id 1:32
set group half type 2
set type 1 diameter 2.0
set type 1 mass 4.18879
set type 2 diameter 1.6
set type 2 mass 2.14
velocity all create 0.5 87287 loop geom
comm_modify vel yes
${extra}
`;

/** The error message of a script that must stop ('' when it runs). */
const errorOf = async (script: string): Promise<string> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  try {
    await session.execute(script);
  } catch {
    // the error event carries the message
  }
  const err = events.find((e) => e.kind === 'error');
  return err && 'message' in err ? String(err.message) : '';
};

describe('pair lubricate and lubricateU setup checks (native pair_lubricate.cpp, pair_lubricateU.cpp)', () => {
  it('rejects a wrong number of pair_style arguments', async () => {
    expect(await errorOf(box('pair_style lubricate 1.5 1 0 2.01'))).toMatch(/Illegal pair_style command/);
    expect(await errorOf(box('pair_style lubricateU 1.5 1 2.01 2.5'))).toMatch(/Illegal pair_style command/);
  });
  it('requires atom attribute omega, then radius (atom_style atomic has neither)', async () => {
    expect(await errorOf(box('pair_style lubricate 1.5 1 0 2.01 2.5\npair_coeff * *\nrun 0', 'atomic'))).toMatch(/Pair lubricate requires atom attribute omega/);
  });
  it('requires monodisperse particles', async () => {
    expect(await errorOf(twoSizes('pair_style lubricate 1.5 1 0 2.01 2.5\npair_coeff * *\nrun 0'))).toMatch(/Pair lubricate requires monodisperse particles/);
  });
  it('requires ghost atoms to store velocity (comm_modify vel yes)', async () => {
    expect(await errorOf(box('comm_modify vel no\npair_style lubricate 1.5 1 0 2.01 2.5\npair_coeff * *\nrun 0'))).toMatch(/Pair lubricate requires ghost atoms store velocity/);
  });
});

describe('pair brownian setup checks (native pair_brownian.cpp)', () => {
  it('rejects a wrong number of pair_style arguments', async () => {
    expect(await errorOf(box('pair_style brownian 1.5 0 0 2.01 2.5'))).toMatch(/Illegal pair_style command/);
  });
  it('requires monodisperse particles', async () => {
    expect(await errorOf(twoSizes('pair_style brownian 1.5 0 0 2.01 2.5 2.0 5878567\npair_coeff * *\nrun 0'))).toMatch(/Pair brownian requires monodisperse particles/);
  });
  it('brownian/poly requires newton pair off', async () => {
    expect(await errorOf(box('newton on\npair_style brownian/poly 1.5 0 0 2.01 2.5 2.0 5878567\npair_coeff * *\nrun 0'))).toMatch(/newton pair off/);
  });
});

describe('fix wall/colloid and fix wall/srd (native fix_wall_colloid.cpp, fix_wall_srd.cpp)', () => {
  it('wall/colloid requires atom attribute radius', async () => {
    expect(await errorOf(box('fix 1 all wall/colloid zlo 0.0 1.0 1.0 2.5\nrun 1', 'atomic'))).toMatch(/requires atom attribute radius/);
  });
  it('wall/srd without fix srd is an error at init', async () => {
    expect(await errorOf(box('fix 1 all wall/srd xlo EDGE\nrun 1').replace('boundary p p p', 'boundary f f f'))).toMatch(/Cannot use fix wall\/srd without fix srd/);
  });
  it('wall/srd with a face given twice is an error', async () => {
    expect(await errorOf(box('fix 1 all wall/srd xlo EDGE xlo EDGE').replace('boundary p p p', 'boundary f f f'))).toMatch(/defined twice/);
  });
});
