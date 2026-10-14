import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Computes ported from LAMMPS (commit b332f28): entropy/atom and basal/atom (src/EXTRA-COMPUTE), xrd and saed (src/DIFFRACTION),
 * and the refusal of fix saed/vtk. Native parity is in tests/oracle/w134cp_*.in (engineOracle.test.ts); this file checks the
 * refusals and error texts of the ported commands.
 */

const SETUP = `units lj
atom_style atomic
lattice fcc 1.0
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
`;

const run = async (script: string): Promise<{ errors: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  try {
    await session.execute(script);
  } catch (e) {
    events.push({ kind: 'error', message: String(e instanceof Error ? e.message : e) } as unknown as EngineEvent);
  }
  return { errors: events.filter((e) => e.kind === 'error').map((e) => ('message' in e ? String(e.message) : '')) };
};

const PAIR = 'pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\n';

describe('compute entropy/atom (ported)', () => {
  it('needs a pair style', async () => {
    const r = await run(`${SETUP}compute e all entropy/atom 0.25 3.0\nrun 0\n`);
    expect(r.errors[0]).toMatch(/requires a pair style be defined/);
  });

  it('refuses a cutoff plus cutoff2 longer than the pair cutoff plus the skin', async () => {
    const r = await run(`${SETUP}${PAIR}neighbor 0.3 bin\ncompute e all entropy/atom 0.25 2.4 avg yes 2.4\ncompute r all reduce sum c_e\nthermo_style custom step c_r\nrun 0\n`);
    expect(r.errors[0]).toMatch(/cutoff is longer than the pairwise cutoff/);
  });

  it('names an unknown keyword', async () => {
    const r = await run(`${SETUP}${PAIR}compute e all entropy/atom 0.25 3.0 bogus yes\nrun 0\n`);
    expect(r.errors[0]).toMatch(/unknown keyword 'bogus'/);
  });

  it('refuses a non-positive sigma', async () => {
    const r = await run(`${SETUP}${PAIR}compute e all entropy/atom 0 3.0\nrun 0\n`);
    expect(r.errors[0]).toMatch(/sigma must be positive/);
  });
});

describe('compute basal/atom (ported)', () => {
  it('takes no arguments', async () => {
    const r = await run(`${SETUP}${PAIR}compute b all basal/atom 1\nrun 0\n`);
    expect(r.errors[0]).toMatch(/Illegal compute basal\/atom command/);
  });
});

describe('compute xrd (ported)', () => {
  it('refuses an unknown element symbol', async () => {
    const r = await run(`${SETUP}${PAIR}compute x all xrd 1.54 Qq 2Theta 40 80\nrun 0\n`);
    expect(r.errors[0]).toMatch(/Invalid ASF atom type Qq/);
  });

  it('names an unknown keyword', async () => {
    const r = await run(`${SETUP}${PAIR}compute x all xrd 1.54 Ni bogus 1\nrun 0\n`);
    expect(r.errors[0]).toMatch(/unknown keyword 'bogus'/);
  });

  it('needs the 2Theta range to be ordered', async () => {
    const r = await run(`${SETUP}${PAIR}compute x all xrd 1.54 Ni 2Theta 80 40\nrun 0\n`);
    expect(r.errors[0]).toMatch(/2Theta range must be greater than zero/);
  });

  it('refuses a structure that is not periodic in any direction when the spacing is not manual', async () => {
    const r = await run(`units lj\natom_style atomic\nboundary f f f\nlattice fcc 1.0\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\n${PAIR}compute x all xrd 1.54 Ni 2Theta 40 80\nrun 0\n`);
    expect(r.errors[0]).toMatch(/Compute XRD must have at least one periodic boundary unless manual spacing specified/);
  });
});

describe('compute saed (ported)', () => {
  it('limits Kmax to 0..12 (|K|max/2 between 0 and 6)', async () => {
    const r = await run(`${SETUP}${PAIR}compute s all saed 0.0251 Ni Kmax 13 manual\nrun 0\n`);
    expect(r.errors[0]).toMatch(/must be between 0 and 6/);
  });

  it('names an unknown keyword', async () => {
    const r = await run(`${SETUP}${PAIR}compute s all saed 0.0251 Ni bogus manual\nrun 0\n`);
    expect(r.errors[0]).toMatch(/unknown keyword 'bogus'/);
  });
});

describe('fix saed/vtk (refused)', () => {
  it('is refused by name with the reason', async () => {
    const r = await run(`${SETUP}${PAIR}compute s all saed 0.0251 Ni manual\nfix v all saed/vtk 1 1 1 s file out.vtk\nrun 0\n`);
    expect(r.errors[0]).toMatch(/fix saed\/vtk is not supported/);
  });
});
