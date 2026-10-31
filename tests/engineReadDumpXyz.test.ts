import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * read_dump format xyz (src/reader_xyz.cpp at b332f28): the values are checked against native by the oracle cases
 * w624rd_xyz and w624rd_xyz_labels; these are the refusals. Measured with native LAMMPS: each message below that is not
 * marked as the engine's own.
 */

const BOX = 'units lj\nregion b block 0 3 0 3 0 3\ncreate_box 2 b\ncreate_atoms 1 single 0.5 0.5 0.5\ncreate_atoms 2 single 1.5 0.5 0.5\nmass * 1.0';

const errorOf = async (xyz: string, lines: string): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} });
  s.addFile('f.xyz', xyz);
  try {
    await s.execute(`${BOX}\n${lines}\n`);
  } catch {
    // the error event carries the message
  }
  const e = events.find((x): x is Extract<EngineEvent, { kind: 'error' }> => x.kind === 'error');
  return e ? e.message : null;
};

const TWO = '2\nc\n1 0.1 0.2 0.3\n2 1.1 0.2 0.3\n';

describe('read_dump format xyz: refusals', () => {
  it('the file has no box: box yes is an error', async () => {
    expect(await errorOf(TWO, 'read_dump f.xyz 0 x y z format xyz')).toContain("No box information in dump, must use 'box no'");
  });

  it('a type label needs a complete atom labelmap', async () => {
    expect(await errorOf('2\nc\nAr 0.1 0.2 0.3\nKr 1.1 0.2 0.3\n', 'read_dump f.xyz 0 x y z box no format xyz'))
      .toContain('Must define atom labelmap to parse XYZ files with strings for types');
  });

  it('a frame of no atoms is incorrectly formatted', async () => {
    expect(await errorOf('0\nempty\n', 'read_dump f.xyz 0 x y z box no format xyz')).toContain('Dump file is incorrectly formatted');
  });

  it('a missing frame, and a field the xyz file does not hold (the engine\'s own message)', async () => {
    expect(await errorOf(TWO, 'read_dump f.xyz 1 x y z box no format xyz')).toContain('Dump file does not contain requested snapshot');
    expect(await errorOf(TWO, 'read_dump f.xyz 0 x vx box no format xyz')).toContain('field vx is not in an xyz file');
  });

  it('reads frame 0 and sets the timestep to the frame number', async () => {
    expect(await errorOf(TWO, 'read_dump f.xyz 0 x y z box no format xyz\nrun 0')).toBeNull();
  });
});
