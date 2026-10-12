import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

const BOX = `
units lj
region b block 0 2 0 2 0 2
create_box 1 b
`;

const runScript = async (text: string) => {
  const session = new Session({ emit: () => {} });
  await session.execute(text);
  return session;
};

describe('fix property/atom ghost keyword', () => {
  // Measured with native LAMMPS (black box): yes, no, on, off, true, false, 1 and 0 are accepted; On, YES and maybe stop.
  for (const v of ['yes', 'no', 'on', 'off', 'true', 'false', '1', '0']) {
    it(`accepts ghost ${v}`, async () => {
      await expect(runScript(`${BOX}fix 1 all property/atom d_x ghost ${v}\n`)).resolves.toBeDefined();
    });
  }
  for (const v of ['On', 'YES', 'maybe']) {
    it(`rejects ghost ${v}`, async () => {
      await expect(runScript(`${BOX}fix 1 all property/atom d_x ghost ${v}\n`)).rejects.toThrow(`Expected boolean parameter instead of '${v}'`);
    });
  }
});
