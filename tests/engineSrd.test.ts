import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
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
  it('refuses collision noslip for a big group without torque (atomic style)', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 all 1.0 0.25 49894 collision noslip'))).toMatch(/no-slip requires atom attribute torque/);
  });
  it('accepts shift yes and shift possible (point particles run; the shift seed must be positive)', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip shift yes 54979'))).toBe('');
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip shift possible 54979'))).toBe('');
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip shift yes 0'))).toMatch(/Invalid seed for Park/);
  });
  it('accepts tstat yes without box deformation, and refuses it with fix deform', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip tstat yes'))).toBe('');
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip tstat yes\nfix 3 all deform 1 x scale 1.0'))).toMatch(/tstat yes with fix deform/);
  });
  it('runs fix deform without the thermostat (constant length rates), and refuses a time-varying rate by name', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip\nfix 3 all deform 1 x scale 1.0'))).toBe('');
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip\nfix 3 all deform 1 x trate 0.01'))).toMatch(/fix deform style trate is not supported with fix srd/);
  });
  it('runs big particles that belong to a rigid body (fix rigid with sphere atoms)', async () => {
    const script = `units lj
atom_style sphere
atom_modify first big
region box block 0 6 0 6 0 6
create_box 2 box
create_atoms 2 single 2 3 3
create_atoms 2 single 4 3 3
group big type 2
set group big diameter 1.0
set group big mass 1.0
lattice sc 2.0
create_atoms 1 box
group small type 1
set group small diameter 0.0
set group small mass 0.01
velocity small create 1.0 593849 loop geom
neighbor 0.3 bin
timestep 0.02
fix 3 big rigid single
fix 1 small srd 1 big 1.0 0.25 49894 collision slip inside ignore
run 1
`;
    expect(await errorOf(script)).toBe('');
  });
  it('accepts overlap yes, and unbiased yes with tstat yes (refused only as tstat yes with fix deform, unless unbiased)', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip overlap yes'))).toBe('');
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip tstat yes unbiased yes'))).toBe('');
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip tstat yes unbiased yes\nfix 3 all deform 1 x scale 1.0'))).toBe('');
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip tstat yes\nfix 3 all deform 1 x scale 1.0'))).toMatch(/tstat yes with fix deform/);
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
    expect(await errorOf(bath('fix 1 all srd 1 all 1.0 0.25 49894 collision slip'))).toMatch(/Big particle in fix srd cannot be point particle/);
  });
  it('runs the default SRD bath (point particles, slip) without error', async () => {
    expect(await errorOf(bath('fix 1 all srd 1 NULL 1.0 0.25 49894 collision slip'))).toBe('');
  });
});

/** fix srd with big triangle particles (tests/oracle/w79srd_tri.data: four triangles, 3d); the setup errors stop before a run. */
describe('fix srd with triangle big particles', () => {
  const triFile = readFileSync(join(__dirname, 'oracle', 'w79srd_tri.data'), 'utf8');
  const triBath = (fix: string): string => `units lj
atom_style tri
read_data w79srd_tri.data
group big type 1
lattice sc 2.0
region box block INF INF INF INF INF INF
create_atoms 2 region box
group small type 2
set group small mass 0.01
pair_style lj/cut 1.5
pair_coeff 1 1 0.0 1.0
pair_coeff 2 2 0.0 1.0 0.0
pair_coeff 1 2 0.0 1.0
pair_style tri/lj 3.5
pair_coeff 1 1 0.1 1.0
pair_coeff 2 2 0.0 1.0 0.0
pair_coeff 1 2 0.0 1.0 0.0
timestep 0.002
velocity small create 1.44 87287 loop geom
${fix}
run 1
`;
  const triError = async (script: string): Promise<string> => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
    session.addFile('w79srd_tri.data', triFile);
    try {
      await session.execute(script);
    } catch {
      // the error event carries the message
    }
    const err = events.find((e) => e.kind === 'error');
    return err && 'message' in err ? String(err.message) : '';
  };

  it('runs free triangles with overlap yes (fix nve/tri)', async () => {
    expect(await triError(triBath('fix 1 big nve/tri\nfix 2 small srd 20 big 1.0 1.0 49894 search 0.2 shift yes 49829 overlap yes collision noslip'))).toBe('');
  });
  it('refuses triangles without overlap yes, by name', async () => {
    expect(await triError(triBath('fix 1 big nve/tri\nfix 2 small srd 20 big 1.0 1.0 49894 search 0.2 collision noslip'))).toMatch(/Cannot use tris with fix srd unless overlap is set/);
  });
  it('runs triangles held in a fix rigid body with overlap yes (matches native: tests/oracle/w254srd_tri_rigid)', async () => {
    expect(await triError(triBath('fix 1 big rigid molecule\nfix 2 small srd 20 big 1.0 1.0 49894 search 0.2 shift yes 49829 overlap yes collision noslip'))).toBe('');
  });
});
