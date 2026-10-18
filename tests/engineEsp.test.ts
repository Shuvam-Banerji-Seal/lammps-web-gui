// kspace_style esp and pair coul/esp, lj/cut/coul/esp (src/engine-gpl/esp, LAMMPS b332f28 KSPACE/esp.cpp).
// Native parity is in the w244es_* oracle cases; these tests cover the refusals and the parameter choices.
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { prolc180 } from '../src/engine-gpl/esp/pswf_poly';

const BASE = `units real
atom_style charge
boundary p p p
lattice fcc 5.64
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box basis 1 1 basis 2 1 basis 3 2 basis 4 2
mass * 22.99
set type 1 charge 1.0
set type 2 charge -1.0
pair_style coul/esp 8.0
pair_coeff * *
`;

/** Runs a script and returns the error message (null when it runs). */
const runScript = async (script: string): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  try {
    await session.execute(script);
    return null;
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return err && 'message' in err ? String(err.message) : String(e);
  }
};

describe('kspace_style esp: parameters', () => {
  it('splitting parameter c comes from the accuracy table (native prolc180)', () => {
    expect(prolc180(1e-4)).toBe(12.024);
    expect(prolc180(1e-5)).toBe(14.471);
  });
});

describe('kspace_style esp: refusals name the unported option', () => {
  it('triclinic boxes', async () => {
    const msg = await runScript(`${BASE}kspace_style esp 1e-4\nchange_box all xy final 1.0\nrun 0\n`);
    expect(msg).toMatch(/triclinic/);
  });

  it('kspace_modify mesh and diff ik', async () => {
    expect(await runScript(`${BASE}kspace_style esp 1e-4\nkspace_modify mesh 8 8 8\nrun 0\n`)).toMatch(/mesh/);
    expect(await runScript(`${BASE}kspace_style esp 1e-4\nkspace_modify diff ik\nrun 0\n`)).toMatch(/diff ik/);
  });

  it('per-atom energy and virial', async () => {
    const msg = await runScript(`${BASE}kspace_style esp 1e-4\ncompute pe all pe/atom kspace\nthermo_style custom step c_pe[1]\nrun 0\n`);
    expect(msg).toMatch(/pe\/atom|stress\/atom/);
  });
});

describe('kspace_style esp: runs', () => {
  it('a neutral crystal runs with the default table and with pair_modify table 0', async () => {
    expect(await runScript(`${BASE}kspace_style esp 1e-4\nrun 0\n`)).toBeNull();
    expect(await runScript(`${BASE}pair_modify table 0\nkspace_style esp 1e-4\nrun 0\n`)).toBeNull();
  });
});
