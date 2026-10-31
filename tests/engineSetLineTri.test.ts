import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * set length, theta, theta/random and tri, and set quat/random on triangles (src/set.cpp process_length, process_theta,
 * process_theta_random, process_tri, process_quat_random and setrandom at b332f28): the values are checked by the
 * oracle cases w620set_line and w620set_tri; these are the refusals. Measured with native LAMMPS: each message below.
 */

const BOX: Record<string, string> = {
  line: 'units lj\natom_style line\ndimension 2\nlattice sq 0.5\nregion box block 0 2 0 2 -0.5 0.5\ncreate_box 1 box\ncreate_atoms 1 box',
  tri: 'units lj\natom_style tri\nlattice sc 0.2\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box',
  atomic: 'units lj\natom_style atomic\nlattice sc 0.2\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box',
};

const errorOf = async (style: string, lines: string): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} });
  try {
    await session.execute(`${BOX[style]}\n${lines}\n`);
  } catch {
    // the error event carries the message
  }
  const e = events.find((x): x is Extract<EngineEvent, { kind: 'error' }> => x.kind === 'error');
  return e ? e.message : null;
};

describe('set length / theta / theta/random / tri: native refusals', () => {
  const cases: [string, string, string][] = [
    ['atomic', 'set group all length 1.0', 'Cannot set attribute length for atom style atomic'],
    ['atomic', 'set group all tri 1.0', 'Cannot set attribute tri for atom style atomic'],
    ['atomic', 'set group all theta 10', 'Cannot set attribute theta for atom style atomic'],
    ['line', 'set group all length -1.0', 'Invalid length in set command'],
    ['line', 'variable v atom -1\nset group all length v_v', 'Invalid length in set command'],
    ['line', 'set group all theta 30', 'Cannot set theta for atom which is not a line'],
    ['line', 'set group all theta/random 0', 'Invalid random number seed in set command'],
    ['line', 'set group all theta/random 5', 'Cannot set theta for atom that is not a line'],
    ['line', 'set group all quat/random 5', 'Cannot set attribute quat/random for atom style line'],
    ['tri', 'set group all tri -2', 'Invalid tri size in set command'],
    ['tri', 'variable v atom -1\nset group all tri v_v', 'Invalid tri size in set command'],
    ['tri', 'set group all quat/random 0', 'Invalid random number seed 0 in set command'],
    ['tri', 'set group all quat/random 5', 'Cannot set quaternion for atom that has none'],
  ];
  for (const [style, lines, message] of cases) {
    it(`${style}: ${lines.split('\n').pop()}`, async () => {
      expect(await errorOf(style, lines)).toContain(message);
    });
  }

  it('a point particle with length 0 or tri 0 stays a point; a line or triangle made first accepts theta and quat/random', async () => {
    expect(await errorOf('line', 'set group all length 0.0\nset group all length 1.0\nset group all theta 30\nset group all theta/random 5')).toBeNull();
    expect(await errorOf('tri', 'set group all tri 0.0\nset group all tri 1.0\nset group all quat/random 5')).toBeNull();
  });
});
