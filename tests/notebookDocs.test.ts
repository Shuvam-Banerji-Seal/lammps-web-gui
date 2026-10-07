import { describe, expect, it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { SUPPORTED_COMMANDS } from '../src/engine/interpreter';
import { styleNames } from '../src/engine/styles';

/*
 * docs/design/notebook.md lists what the engine supports between
 * <!-- coverage:begin --> and <!-- coverage:end -->. The list must match the
 * engine's own registries; regenerate it with
 *   UPDATE_COVERAGE=1 npx vitest run tests/notebookDocs.test.ts
 */

const DOC = join(__dirname, '..', 'docs', 'design', 'notebook.md');
const BEGIN = '<!-- coverage:begin -->';
const END = '<!-- coverage:end -->';

const block = (): string => {
  const code = (names: string[]) => names.map((n) => `\`${n}\``).join(' ');
  const rows = [['commands', SUPPORTED_COMMANDS.slice()], ...Object.entries(styleNames())] as [string, string[]][];
  return [
    '| Kind | Supported |',
    '|---|---|',
    ...rows.map(([kind, names]) => `| ${kind} | ${names.length ? code(names) : '—'} |`),
  ].join('\n');
};

describe('notebook design doc', () => {
  it('lists exactly the commands and styles the engine registers', () => {
    const text = readFileSync(DOC, 'utf8');
    const a = text.indexOf(BEGIN), b = text.indexOf(END);
    expect(a, 'coverage markers').toBeGreaterThanOrEqual(0);
    expect(b).toBeGreaterThan(a);
    const want = `${BEGIN}\n${block()}\n${END}`;
    if (process.env.UPDATE_COVERAGE === '1') {
      writeFileSync(DOC, text.slice(0, a) + want + text.slice(b + END.length));
      return;
    }
    expect(text.slice(a, b + END.length), 'run: UPDATE_COVERAGE=1 npx vitest run tests/notebookDocs.test.ts').toBe(want);
  });
});
