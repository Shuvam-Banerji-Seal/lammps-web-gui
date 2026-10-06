import { describe, expect, it } from 'vitest';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ALL_COMMANDS, COMMAND_BY_ID, defaultParams, type CommandDef } from '../src/lammps/catalog';
import { generateScript } from '../src/lammps/generator';
import { matchLine, parseScript, scriptStatements, scriptStatementsDetailed, tokenizeLine } from '../src/lammps/scriptParser';

const norm = (stmt: string) => tokenizeLine(stmt).join(' ');

/** Every single-line command the builder can emit: defaults + each option of the first enum param. */
const builderLines = (): { def: CommandDef; line: string }[] => {
  const out: { def: CommandDef; line: string }[] = [];
  for (const def of ALL_COMMANDS) {
    if (def.deprecated || def.id === 'raw_line') continue;
    const base = defaultParams(def);
    const variants: Record<string, string>[] = [base];
    const firstEnum = def.params.find((p) => p.type === 'enum' && p.options?.length);
    for (const o of firstEnum?.options ?? []) variants.push({ ...base, [firstEnum!.key]: o.value });
    const seen = new Set<string>();
    for (const params of variants) {
      let lines: string[];
      try { lines = def.build(params).filter((l) => l.trim() !== ''); } catch { continue; }
      if (lines.length !== 1 || seen.has(lines[0])) continue;
      seen.add(lines[0]);
      out.push({ def, line: lines[0] });
    }
  }
  return out;
};

describe('import -> regenerate is exact for everything the builder emits', () => {
  it('every single-line build() output imports to a real step that rebuilds the same line', () => {
    const failures: string[] = [];
    const cases = builderLines();
    for (const { line } of cases) {
      const { model } = parseScript(line);
      const step = model.steps[0];
      if (!step || step.defId === 'raw_line') { failures.push(`raw: ${line}`); continue; }
      const rebuilt = COMMAND_BY_ID[step.defId].build(step.params).filter((l) => l.trim() !== '');
      if (rebuilt.length !== 1 || norm(rebuilt[0]) !== norm(line)) failures.push(`${line}  ->  ${rebuilt.join(' / ')}`);
    }
    expect(cases.length).toBeGreaterThan(400);
    expect(failures).toEqual([]);
  });

  it.each([
    ['fix 1 all nve', 'fix_nve'],
    ['fix 1 addatoms nve', 'fix_nve'],
    ['fix 3 all enforce2d', 'fix_enforce2d'],
    ['compute myKE all ke/atom', null],
    ['create_atoms 1 region substrate', null],
    ['kspace_style none', null],
    ['special_bonds 0.0 0.0 0.5', null],
    ['processors * * *', null],
    ['region box block 0 10 0 10 0 10 side out', null],
    ['if "${steps} > 10000" then "print \'done\'"', null],
  ])('%s survives import and regeneration', (line, defId) => {
    const { model } = parseScript(line);
    const step = model.steps[0];
    expect(step.defId).not.toBe('raw_line');
    if (defId) expect(step.defId).toBe(defId);
    const text = generateScript(model).text;
    expect(scriptStatements(text).map(norm)).toContain(norm(line));
  });
});

describe('statement splitting follows Commands_parse.html', () => {
  it('keeps a triple-quoted argument with a # and newlines as one statement', () => {
    const text = 'print """\nRun finished. Step count # with a hash\nsecond line\n"""\nrun 10\n';
    const st = scriptStatementsDetailed(text);
    expect(st.map((s) => s.line)).toEqual([1, 5]);
    expect(st[0].text).toBe('print """\nRun finished. Step count # with a hash\nsecond line\n"""');
    expect(tokenizeLine(st[0].text)).toEqual(['print', '"""\nRun finished. Step count # with a hash\nsecond line\n"""']);
  });

  it('a comment after a trailing & ends the statement', () => {
    const st = scriptStatementsDetailed('fix 1 all nve & # note\nrun 10\n');
    expect(st.map((s) => s.text)).toEqual(['fix 1 all nve &', 'run 10']);
  });

  it('& continues across lines, also inside double quotes; # in quotes is not a comment', () => {
    const st = scriptStatementsDetailed('region box block 0 10 &\n  0 10 0 10\nprint "a # b &\nc"\n');
    expect(st.map((s) => norm(s.text))).toEqual(['region box block 0 10 0 10 0 10', 'print "a # b c"']);
    expect(st.map((s) => s.line)).toEqual([1, 3]);
  });

  it('matchLine still prefers the specific def for hybrid pair styles', () => {
    expect(matchLine(tokenizeLine('pair_style hybrid/overlay lj/cut 10.0 coul/long 10.0'))?.def.id).toMatch(/hybrid/);
  });
});

// The official LAMMPS example inputs are GPL-2.0, so they are not in this
// repo; point LAMMPS_CORPUS_DIR at a folder of *.in files to run this check.
const CORPUS = process.env.LAMMPS_CORPUS_DIR ?? '';
const corpusFiles = CORPUS && existsSync(CORPUS) ? readdirSync(CORPUS).filter((f) => f.endsWith('.in')) : [];

describe.skipIf(corpusFiles.length === 0)('official example scripts round-trip (LAMMPS_CORPUS_DIR)', () => {
  it('no statement changes on import -> generate', () => {
    const changed: string[] = [];
    let total = 0;
    for (const f of corpusFiles) {
      const original = readFileSync(join(CORPUS, f), 'utf8');
      const before = scriptStatements(original).map(norm);
      const after = scriptStatements(generateScript(parseScript(original).model).text).map(norm);
      total += before.length;
      const afterSet = new Set(after);
      for (const s of before) if (!afterSet.has(s)) changed.push(`${f}: ${s}`);
    }
    expect(total).toBeGreaterThan(300);
    expect(changed).toEqual([]);
  });
});
