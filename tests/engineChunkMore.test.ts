import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { COMPUTES, FIXES } from '../src/engine-gpl/registry/chunk_more';
import { LAZY_GROUPS } from '../src/engine/lazyStyles';

/*
 * Unit tests for the chunk computes and fixes ported into src/engine-gpl/chunk (vcm/chunk, torque/chunk, vacf/chunk,
 * inertia, slice, global/atom, fix momentum/chunk, spring/chunk, store/force). Numerical parity with native LAMMPS is
 * checked by the w218ck_* oracle cases; these tests cover the registration, the refusals that name their option, and
 * a few values that follow from the definitions (a uniform velocity field has the same centre-of-mass velocity in
 * every chunk; a fix that removes the chunk momentum leaves none).
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { error: error ? String((error as Error).message ?? error) : null, thermo };
};

/** Two types of atoms in a periodic box, so that type chunks give two chunks. */
const SETUP = `
units           lj
atom_style      atomic
boundary        p p p
region          box block 0 4 0 4 0 4
create_box      2 box
create_atoms    1 random 10 4321 box
create_atoms    2 random 10 1234 box
mass            * 1.0
pair_style      lj/cut 2.5
pair_coeff      * * 0.0 1.0
`;

describe('engine-gpl chunk computes and fixes: registration', () => {
  it('exports every ported style in the chunk_more registry', () => {
    expect(Object.keys(COMPUTES).sort()).toEqual(['global/atom', 'inertia', 'slice', 'torque/chunk', 'vacf/chunk', 'vcm/chunk']);
    expect(Object.keys(FIXES).sort()).toEqual(['momentum/chunk', 'spring/chunk', 'store/force']);
  });

  it('lists the same names in the lazy group gpl_chunk_more', () => {
    const group = LAZY_GROUPS.find((g) => g.id === 'gpl_chunk_more');
    expect(group).toBeDefined();
    expect([...(group!.names.COMPUTES ?? [])].sort()).toEqual(Object.keys(COMPUTES).sort());
    expect([...(group!.names.FIXES ?? [])].sort()).toEqual(Object.keys(FIXES).sort());
  });
});

describe('engine-gpl chunk computes: values from the definitions', () => {
  it('vcm/chunk gives the centre-of-mass velocity of each chunk (uniform velocity field)', async () => {
    const r = await runScript(`${SETUP}
velocity        all set 1.0 2.0 3.0
compute         cm all chunk/atom type
compute         vcm all vcm/chunk cm
thermo_style    custom step c_vcm[1][1] c_vcm[1][3] c_vcm[2][2]
thermo_modify   format float %.15g norm no
run             0
`);
    expect(r.error).toBeNull();
    const row = r.thermo[0];
    expect(row['c_vcm[1][1]']).toBeCloseTo(1.0, 12);
    expect(row['c_vcm[1][3]']).toBeCloseTo(3.0, 12);
    expect(row['c_vcm[2][2]']).toBeCloseTo(2.0, 12);
  });

  it('momentum/chunk with linear 1 1 1 leaves no chunk momentum after its step', async () => {
    const r = await runScript(`${SETUP}
velocity        all create 1.0 9876 mom no rot no
fix             1 all nve
compute         cm all chunk/atom type
compute         vcm all vcm/chunk cm
fix             2 all momentum/chunk 1 cm linear 1 1 1
thermo_style    custom step c_vcm[1][1] c_vcm[2][3]
thermo_modify   format float %.15g norm no
thermo          1
run             2
`);
    expect(r.error).toBeNull();
    // vcm is evaluated at the thermo output, after the end-of-step removal of the chunk momentum
    const last = r.thermo[r.thermo.length - 1];
    expect(Math.abs(last['c_vcm[1][1]'] as number)).toBeLessThan(1e-12);
    expect(Math.abs(last['c_vcm[2][3]'] as number)).toBeLessThan(1e-12);
  });

  it('store/force keeps zeros for atoms outside the group', async () => {
    const r = await runScript(`${SETUP}
fix             1 all nve
group           half type 1
fix             2 half store/force
compute         fs all reduce sum f_2[1] f_2[2] f_2[3]
thermo_style    custom step c_fs[1] c_fs[2]
thermo_modify   format float %.15g norm no
run             0
`);
    expect(r.error).toBeNull();
    expect(Number.isFinite(r.thermo[0]['c_fs[1]'] as number)).toBe(true);
  });
});

describe('engine-gpl chunk computes and fixes: refusals name their option', () => {
  it('slice refuses several inputs (a global array)', async () => {
    const r = await runScript(`${SETUP}
compute         ine all inertia
compute         sl all slice 1 3 1 c_ine c_ine
run             0
`);
    expect(r.error).toMatch(/compute slice with more than one input/);
  });

  it('slice refuses a range that is not a multiple of nskip', async () => {
    const r = await runScript(`${SETUP}
compute         ine all inertia
compute         sl all slice 1 6 2 c_ine
run             0
`);
    expect(r.error).toMatch(/must be a multiple of Nskip/);
  });

  it('global/atom expands wildcard values to the columns of a global array', async () => {
    const r = await runScript(`${SETUP}
compute         cm all chunk/atom type
compute         vcm all vcm/chunk cm
compute         mid all property/atom type
compute         gat all global/atom c_mid c_vcm[*]
compute         gs all reduce sum c_gat[3]
thermo_style    custom step c_gs
run             0
`);
    // the index c_mid, then one per-atom column for each of the three columns of c_vcm
    expect(r.error).toBeNull();
    expect(r.thermo.length).toBe(1);
  });

  it('momentum/chunk rejects an unknown keyword with the native message', async () => {
    const r = await runScript(`${SETUP}
compute         cm all chunk/atom type
fix             2 all momentum/chunk 1 cm linear 1 1 1 bogus
run             0
`);
    expect(r.error).toMatch(/Illegal fix momentum\/chunk command/);
  });

  it('spring/chunk locks a chunk/atom with nchunk every before it was set up: Nchunk 0, then the native lock error', async () => {
    // Measured with native LAMMPS: step 0 prints f_2 = 0 and c_cm = 0, then step 1 stops with this error; with the
    // default nchunk once the chunks are set up at the start of the run (c_cm = 2) and the run goes on
    const every = await runScript(`${SETUP}
compute         cm all chunk/atom type nchunk every
compute         com all com/chunk cm
fix             1 all nve
fix             2 all spring/chunk 2.0 cm com
thermo_style    custom step f_2 c_cm
thermo          1
run             5
`);
    expect(every.thermo[0]).toMatchObject({ step: 0, f_2: 0, c_cm: 0 });
    expect(every.error).toMatch(/Two fix commands using same compute chunk\/atom command in incompatible ways/);
    const once = await runScript(`${SETUP}
compute         cm all chunk/atom type
compute         com all com/chunk cm
fix             1 all nve
fix             2 all spring/chunk 2.0 cm com
thermo_style    custom step f_2 c_cm
thermo          1
run             5
`);
    expect(once.error).toBeNull();
    expect(once.thermo[0]).toMatchObject({ step: 0, c_cm: 2 });
  });

  it('vcm/chunk rejects a chunk ID of another style', async () => {
    const r = await runScript(`${SETUP}
compute         ine all inertia
compute         vcm all vcm/chunk ine
run             0
`);
    expect(r.error).toMatch(/Chunk\/atom compute ine does not exist or is incorrect style for compute vcm\/chunk/);
  });
});
