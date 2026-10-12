import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

describe('create_atoms random in 2d', () => {
  it('draws z and discards it, as native (positions measured with native LAMMPS)', async () => {
    const files = new Map<string, string>();
    const s = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
    await s.execute('units lj\ndimension 2\nboundary p p p\nregion b block 0 10 0 10 -0.5 0.7\ncreate_box 1 b\ncreate_atoms 1 random 5 12345 NULL\nmass 1 1\nwrite_dump all custom e.txt id x y z modify format float %.15g\n');
    const rows = (files.get('e.txt') ?? '').trim().split('\n').slice(-5).map((l) => l.trim().split(/\s+/).map(Number));
    // Measured with native LAMMPS (black box): write_dump of the same input
    const native = [
      [1, 9.10652394830553, 3.34799917105026, 0],
      [2, 9.49421721021375, 9.30865206257843, 0],
      [3, 9.23120571730249, 8.874490702932, 0],
      [4, 1.05890107856081, 6.95042737151982, 0],
      [5, 2.42647574396174, 1.77782876499827, 0],
    ];
    for (let i = 0; i < 5; i++) for (let k = 0; k < 4; k++) expect(rows[i][k]).toBeCloseTo(native[i][k], 12);
  });
});
