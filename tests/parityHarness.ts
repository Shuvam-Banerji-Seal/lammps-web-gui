import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import { parityFiles, parityInput, type ParityFixture } from './parityMatrices';

/*
 * Parity matrix harness (the suites are tests/engineParity.<matrix>.test.ts; the rows and their inputs are defined in
 * tests/parityMatrices.ts): each row of tests/fixtures/parity/<matrix>.json was run by native LAMMPS
 * (scripts/parity/run-parity.mjs), and the engine must give the same first ERROR line, or run without one. A row
 * marked skip says why it is not compared. One row: npx vitest run tests/engineParity -t '<style>'
 */

const ROOT = join(__dirname, '..');

/** The engine's result in the fixture's form: OK, or ERROR: and the first line of the message without its line number. */
const engineResult = async (input: string, files: Record<string, { path: string; latin1: boolean }>): Promise<string> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  for (const [name, f] of Object.entries(files)) s.addFile(name, readFileSync(f.path, f.latin1 ? 'latin1' : 'utf8'));
  try {
    await s.execute(input);
  } catch (e) {
    return 'ERROR: ' + String((e as Error).message ?? e).split('\n')[0].replace(/^line \d+: /, '');
  }
  return 'OK';
};

export const paritySuite = (matrix: string): void => {
  const fx: ParityFixture = JSON.parse(readFileSync(join(ROOT, 'tests', 'fixtures', 'parity', `${matrix}.json`), 'utf8'));
  const caseText = new Map<string, string>();
  const textOf = (name: string): string => {
    if (!caseText.has(name)) caseText.set(name, readFileSync(join(ROOT, 'tests', 'oracle', `${name}.in`), 'utf8'));
    return caseText.get(name)!;
  };
  describe(`parity with native LAMMPS: ${fx.about}`, () => {
    for (const row of fx.rows) {
      const name = `${row.style}: ${row.text}${row.build ? ` (native ${row.build} build)` : ''}`;
      if (row.skip) {
        it.skip(`${name} [${row.skip}]`, () => {});
        continue;
      }
      it(name, async () => {
        const input = parityInput(textOf(row.case), row);
        expect(await engineResult(input, parityFiles(ROOT, textOf(row.case), input)), `${row.case} line ${row.line + 1}`).toBe(row.native);
      });
    }
  });
};
