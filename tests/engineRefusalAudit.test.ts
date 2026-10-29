import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Inputs the engine used to refuse as its own limit where native LAMMPS stops with its own error, or runs: the engine now
 * gives native's message or behaviour. Each case was measured with native LAMMPS (b332f28) and the code cites the source.
 */

const run = async (script: string, files: Record<string, string> = {}): Promise<Session> => {
  const s = new Session({ emit: () => undefined, writeFile: () => undefined } as never);
  for (const [n, t] of Object.entries(files)) s.addFile(n, t);
  await s.execute(script);
  return s;
};

const BOX = 'units lj\natom_style atomic\nregion b block 0 5 0 5 0 5\ncreate_box 1 b\n';
const molFile = (section: string, body: string): string =>
  `# test\n\n2 atoms\n\nCoords\n\n1 0 0 0\n2 1 0 0\n\nTypes\n\n1 1\n2 1\n\n${section}\n\n${body}\n`;

describe('molecule file sections native does not read', () => {
  // Measured with native LAMMPS: ERROR: Found data file section 'Pair Coeffs' in molecule file; ERROR: Unknown section
  // 'Foo Bar' in molecule file; ERROR: Unexpected line in molecule file while looking for the next section: (then the line)
  it('a data file section, an unknown section and a line that is no keyword', async () => {
    await expect(run(`${BOX}molecule m m.txt\n`, { 'm.txt': molFile('Pair Coeffs', '1 1.0 1.0') }))
      .rejects.toThrow(/Found data file section 'Pair Coeffs' in molecule file/);
    await expect(run(`${BOX}molecule m m.txt\n`, { 'm.txt': molFile('Foo Bar', '1 2') }))
      .rejects.toThrow(/Unknown section 'Foo Bar' in molecule file/);
    await expect(run(`${BOX}molecule m m.txt\n`, { 'm.txt': molFile('12 x', '1 2') }))
      .rejects.toThrow(/Unexpected line in molecule file while looking for the next section:\n12 x/);
  });
});
