import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// fix srd (src/engine-gpl/srd/fix_srd.ts): the supported settings run (oracle cases w79srd_*); the rest must be
// refused with a StyleError that names the keyword, and the setup checks of fix_srd.cpp must hold.

/** A small 2d SRD bath: the fix line is the only thing that varies. */
const bath = (fix: string): string => `units lj
atom_style atomic
atom_modify first empty
dimension 2
lattice sq 0.4
region box block 0 3 0 3 -0.5 0.5
create_box 1 box
lattice sq 0.4
region plane block 0 3 0 3 -0.001 0.001
lattice sq 85.0
create_atoms 1 region plane
mass 1 0.01
velocity all create 1.0 593849 loop geom
neighbor 0.3 bin
neigh_modify delay 1 every 1 check no
timestep 0.02
${fix}
fix 2 all enforce2d
run 1
`;

/** The error message of a script that must stop (empty string when it runs). */
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

describe('fix srd refusals name the keyword', () => {
  it('refuses collision noslip', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision noslip'))).toMatch(/noslip/);
  });
  it('refuses shift yes and shift possible', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip shift yes 54979'))).toMatch(/shift/);
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip shift possible 54979'))).toMatch(/shift/);
  });
  it('refuses tstat yes and overlap yes', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip tstat yes'))).toMatch(/tstat/);
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip overlap yes'))).toMatch(/overlap/);
  });
  it('names an unknown keyword', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 bogus 3'))).toMatch(/Unknown fix srd keyword bogus/);
  });
});

describe('fix srd setup checks (fix_srd.cpp)', () => {
  it('rejects nevery <= 0 and an unknown big group', async () => {
    expect(await errorOf(bath('fix 1 all srd 0 NULL 1.0 0.25 49894'))).toMatch(/nevery value must be > 0/);
    expect(await errorOf(bath('fix 1 all srd 1 nosuch 1.0 0.25 49894'))).toMatch(/Could not find fix srd big particle group nosuch/);
  });
  it('rejects a temperature, grid spacing or seed that is not positive', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 0.0 0.25 49894'))).toMatch(/Illegal fix srd temperature/);
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 -0.25 49894'))).toMatch(/Illegal fix srd grid spacing/);
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 0'))).toMatch(/Illegal fix srd random seed/);
  });
  it('requires lamda >= 0.6 of the SRD grid size when shifting is refused', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 lamda 0.01'))).toMatch(/lamda must be >= 0.6 of SRD grid size/);
  });
  it('refuses a big particle that is a point particle', async () => {
    // every atom of the bath is a point particle (atom_style atomic), so using "all" as the big group must fail
    expect(await errorOf(bath('fix 1 all srd 1 all 1.0 0.25 49894'))).toMatch(/Big particle in fix srd cannot be point particle/);
  });
  it('runs the default SRD bath (point particles, slip) without error', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip'))).toBe('');
  });
});
