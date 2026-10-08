import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * The licence makes attribution a CONDITION, not a courtesy (LICENSE §3), so
 * the notices have to be load-bearing rather than decorative. These guards
 * fail the build if a refactor quietly drops one.
 */
const root = resolve(__dirname, '..');
const read = (rel: string) => readFileSync(resolve(root, rel), 'utf8');

describe('licence and attribution files', () => {
  it.each(['LICENSE', 'NOTICE', 'COMMERCIAL.md', 'CITATION.cff'])('%s exists', f => {
    expect(existsSync(resolve(root, f))).toBe(true);
  });

  it('LICENSE names the author, the educational grant and the revenue share', () => {
    const l = read('LICENSE');
    expect(l).toContain('Shuvam Banerji Seal');
    expect(l).toContain('EDUCATIONAL AND NON-COMMERCIAL SOURCE-AVAILABLE LICENSE');
    expect(l).toMatch(/ATTRIBUTION — MANDATORY/);
    expect(l).toMatch(/royalty of TEN PERCENT \(10%\)/);
    expect(l).toMatch(/of Revenue, payable to the Author/);
    // the MIT history must stay acknowledged rather than silently dropped
    expect(l).toContain('RELATIONSHIP TO PRIOR RELEASES');
  });

  it('NOTICE carries the author credit and the third-party disclaimer', () => {
    const n = read('NOTICE');
    expect(n).toContain('Shuvam Banerji Seal');
    expect(n).toContain('THIRD-PARTY COMPONENTS');
    expect(n).toContain('not affiliated with, endorsed by, or a product of the LAMMPS');
  });

  it('package.json points at the custom licence, not MIT', () => {
    const pkg = JSON.parse(read('package.json'));
    expect(pkg.license).toBe('LicenseRef-LAMMPS-Web-GUI-Educational-1.0');
    expect(pkg.author.name).toBe('Shuvam Banerji Seal');
  });

  it('the UI shows the author credit required by LICENSE §3.2', () => {
    const app = read('src/App.tsx');
    expect(app).toContain('by Shuvam Banerji Seal');
    // and the About dialog must keep the licence reachable from the app
    expect(app).toContain('COMMERCIAL.md');
    expect(app).toContain('/blob/main/LICENSE');
  });

  it('generated LAMMPS scripts and data files stay signed', () => {
    expect(read('src/lammps/generator.ts')).toContain('by Shuvam Banerji Seal');
    expect(read('src/lammps/exporter.ts')).toContain('Shuvam Banerji Seal');
  });

  it('README states the terms plainly instead of claiming MIT', () => {
    const r = read('README.md');
    expect(r).toContain('Educational and Non-Commercial Source-Available');
    expect(r).toContain('COMMERCIAL.md');
    expect(r).not.toMatch(/\[MIT\]\(LICENSE\)/);
  });
});
