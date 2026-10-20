import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { parseDrudeTypes, NOPOL_TYPE, CORE_TYPE, DRUDE_TYPE } from '../src/engine-gpl/drude/drude_model';

// The DRUDE styles (src/engine-gpl/drude): argument checks and the refusals of styles that are not ported yet.
const run = async (script: string): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  try {
    await session.execute(script);
    return null;
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return err && 'message' in err ? err.message : String(e);
  }
};

const BOX = `units real
atom_style full
region b block 0 10 0 10 0 10
create_box 2 b bond/types 1 extra/bond/per/atom 1 extra/special/per/atom 2
mass * 1.0
create_atoms 1 single 1 1 1
create_atoms 2 single 2 2 2
bond_style harmonic
bond_coeff 1 100.0 1.0
create_bonds single/bond 1 1 2
`;

describe('DRUDE type words (fix drude)', () => {
  it('maps n/c/d words to the three kinds, with the digit forms', () => {
    const kind = parseDrudeTypes(['N', 'c', 'D', '0', '1', '2'], 6);
    expect(Array.from(kind.slice(1))).toEqual([NOPOL_TYPE, CORE_TYPE, DRUDE_TYPE, NOPOL_TYPE, CORE_TYPE, DRUDE_TYPE]);
  });

  it('refuses a wrong count and an unknown word by name', () => {
    expect(() => parseDrudeTypes(['C'], 2)).toThrow(/Incorrect number of arguments \(1 instead of 2\)/);
    expect(() => parseDrudeTypes(['C', 'X'], 2)).toThrow(/Unknown drude type X for atom type 2/);
  });
});

describe('DRUDE styles in a script', () => {
  it('fix drude needs one type word per atom type', async () => {
    expect(await run(BOX + 'fix F all drude C\n')).toMatch(/Incorrect number of arguments/);
  });

  it('compute temp/drude and pair thole need fix drude', async () => {
    expect(await run(BOX + 'compute T all temp/drude\nthermo_style custom step c_T[1]\nrun 0\n')).toMatch(/requires fix drude/);
    expect(await run(BOX + 'pair_style thole 2.0 8.0\npair_coeff * * 1.0\nrun 0\n')).toMatch(/requires fix drude/);
  });

  it('a core without a Drude partner is an error at definition time', async () => {
    expect(await run(BOX + 'fix F all drude C C\n')).toMatch(/Core atom ID 1 has no drude atom/);
  });

  it('a Drude partner is found through the bond list', async () => {
    const script = BOX + 'create_atoms 1 single 3 3 3\nset atom 3 type 2\ncreate_bonds single/bond 1 1 3\nfix F all drude C D\n';
    expect(await run(script)).toBeNull();
  });

  it('tgnvt/drude runs with fix drude; its argument errors are named', async () => {
    expect(await run(BOX + 'fix F all drude C D\nfix N all tgnvt/drude temp 1 1 100 1 20\nrun 0\n')).toBeNull();
    expect(await run(BOX + 'fix F all drude C D\nfix N all tgnvt/drude temp 1 1 100 1 20 iso 1 1 100\n')).toMatch(/Pressure control can not be used with fix tgnvt\/drude/);
    expect(await run(BOX + 'fix N all tgnvt/drude temp 1 1 100 1 20\n')).toMatch(/requires fix drude/);
  });

  it('tgnpt/drude takes the barostat shapes of fix nh and names its argument errors as native does', async () => {
    expect(await run(BOX + 'fix F all drude C D\nfix N all tgnpt/drude temp 1 1 100 1 20 aniso 1 1 100\n')).toBeNull();
    expect(await run(BOX + 'fix F all drude C D\nfix N all tgnpt/drude temp 1 1 100 1 20\n')).toMatch(/Pressure control must be used/);
    expect(await run(BOX + 'fix F all drude C D\nfix N all tgnpt/drude temp 1 1 100 1 20 xy 1 1 100\n')).toMatch(/Can not specify Pxy\/Pxz\/Pyz in fix tgnpt\/drude with non-triclinic box/);
    expect(await run(BOX + 'fix F all drude C D\nfix N all tgnpt/drude temp 1 1 100 1 20 x 1 1 100 y 2 2 100 couple xy\n')).toMatch(/Invalid fix tgnpt\/drude pressure settings/);
    expect(await run(BOX + 'fix F all drude C D\nfix N all tgnpt/drude iso 1 1 100\n')).toMatch(/Temperature control must be used with fix tgnpt\/drude/);
  });
});
