import { describe, expect, it } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { isSafeHref, parseInline, parseMarkdown, type Inline } from '../src/lammps/markdown';
import MarkdownView from '../src/components/workbench/notebook/MarkdownView';
import { getThemeTokens } from '../src/theme';

const text = (v: string): Inline => ({ t: 'text', v });

describe('markdown parser: blocks', () => {
  it('reads headings 1 to 3; four hashes is a paragraph', () => {
    const blocks = parseMarkdown('# One\n## Two\n### Three\n#### Four');
    expect(blocks.map((b) => (b.t === 'heading' ? b.level : b.t))).toEqual([1, 2, 3, 'p']);
    expect(blocks[0]).toEqual({ t: 'heading', level: 1, c: [text('One')] });
  });

  it('joins consecutive lines into one paragraph and splits on blank lines', () => {
    const blocks = parseMarkdown('first line\nsecond line\n\nthird');
    expect(blocks).toEqual([
      { t: 'p', c: [text('first line second line')] },
      { t: 'p', c: [text('third')] },
    ]);
  });

  it('parses bullet and numbered lists (numbered keeps its start)', () => {
    const blocks = parseMarkdown('- alpha\n- beta\n\n3. gamma\n4. delta');
    expect(blocks).toEqual([
      { t: 'ul', items: [[text('alpha')], [text('beta')]] },
      { t: 'ol', start: 3, items: [[text('gamma')], [text('delta')]] },
    ]);
  });

  it('parses quotes recursively and rules', () => {
    expect(parseMarkdown('> quoted **bold**\n> more')).toEqual([
      { t: 'quote', blocks: [{ t: 'p', c: [text('quoted '), { t: 'strong', c: [text('bold')] }, text(' more')] }] },
    ]);
    expect(parseMarkdown('---')).toEqual([{ t: 'hr' }]);
    expect(parseMarkdown('***')).toEqual([{ t: 'hr' }]);
  });

  it('keeps fenced code verbatim: no inline or block syntax inside', () => {
    const src = '```python\n# not a heading\n- not a list <b>x</b> **y**\n[l](https://a.b)\n```\nafter';
    expect(parseMarkdown(src)).toEqual([
      { t: 'code', lang: 'python', v: '# not a heading\n- not a list <b>x</b> **y**\n[l](https://a.b)' },
      { t: 'p', c: [text('after')] },
    ]);
  });

  it('an unclosed fence runs to the end of the cell', () => {
    expect(parseMarkdown('```\nrun 10\n  indented')).toEqual([{ t: 'code', lang: '', v: 'run 10\n  indented' }]);
  });

  it('empty and blank input give no blocks', () => {
    expect(parseMarkdown('')).toEqual([]);
    expect(parseMarkdown('  \n\n\t\n')).toEqual([]);
  });

  it('parses pipe tables with alignment and escaped pipes', () => {
    const blocks = parseMarkdown('| name | value |\n|:-----|------:|\n| a \\| b | 1 |\n| c |');
    expect(blocks).toEqual([{
      t: 'table',
      align: ['left', 'right'],
      head: [[text('name')], [text('value')]],
      rows: [[[text('a | b')], [text('1')]], [[text('c')]]],
    }]);
  });

  it('a line with a pipe but no separator row is a paragraph', () => {
    expect(parseMarkdown('a | b')).toEqual([{ t: 'p', c: [text('a | b')] }]);
  });
});

describe('markdown parser: inline', () => {
  it('nests emphasis', () => {
    expect(parseInline('**bold *it* end**')).toEqual([
      { t: 'strong', c: [text('bold '), { t: 'em', c: [text('it')] }, text(' end')] },
    ]);
    expect(parseInline('*a **b** c*')).toEqual([
      { t: 'em', c: [text('a '), { t: 'strong', c: [text('b')] }, text(' c')] },
    ]);
  });

  it('does not treat a lone asterisk followed by a space as emphasis', () => {
    expect(parseInline('2 * 3 * 4')).toEqual([text('2 * 3 * 4')]);
  });

  it('code spans keep their text verbatim', () => {
    expect(parseInline('use `*x* [a](https://b.c)` now')).toEqual([
      text('use '), { t: 'code', v: '*x* [a](https://b.c)' }, text(' now'),
    ]);
  });

  it('keeps http, https and mailto links', () => {
    expect(parseInline('[docs](https://docs.lammps.org/x.html)')).toEqual([
      { t: 'link', href: 'https://docs.lammps.org/x.html', c: [text('docs')] },
    ]);
    expect(parseInline('[mail](mailto:a@b.co)')).toEqual([{ t: 'link', href: 'mailto:a@b.co', c: [text('mail')] }]);
  });

  it('rejects javascript:, data: and other schemes, keeping the markdown as text', () => {
    for (const href of ['javascript:alert(1)', 'JaVaScRiPt:alert(1)', 'data:text/html,<b>x</b>', 'ftp://x.y', '//evil.example', 'https://a b']) {
      const nodes = parseInline(`[x](${href})`);
      expect(nodes.some((n) => n.t === 'link')).toBe(false);
    }
    expect(isSafeHref('javascript:alert(1)')).toBe(false);
    expect(isSafeHref('https://ok.example/path?q=1')).toBe(true);
  });

  it('never throws on unbalanced markers', () => {
    expect(() => parseInline('**unclosed *x [y](')).not.toThrow();
    expect(parseInline('**unclosed')).toEqual([text('**unclosed')]);
  });
});

describe('markdown rendering is text, not markup', () => {
  const ct = getThemeTokens('dark');
  const render = (src: string) => renderToStaticMarkup(createElement(MarkdownView, { ct, source: src }));

  it('renders <script>, raw HTML and javascript: links as escaped text', () => {
    const html = render('<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\n[click](javascript:alert(1))');
    expect(html).not.toContain('<script');
    expect(html).not.toContain('<img');
    expect(html).not.toContain('javascript:alert(1)</a>');
    expect(html).not.toContain('href="javascript');
    expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;');
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');
  });

  it('renders a heading, list, link and table', () => {
    const html = render('# Title\n\n- one\n- two\n\n[docs](https://docs.lammps.org)\n\n| a | b |\n|---|---|\n| 1 | 2 |');
    expect(html).toContain('<h1');
    expect(html).toContain('Title</h1>');
    expect(html).toContain('<li>one</li>');
    expect(html).toContain('href="https://docs.lammps.org"');
    expect(html).toContain('<table');
    expect(html).toContain('<td');
  });

  it('shows an empty cell as a hint', () => {
    expect(render('   ')).toContain('Empty text cell');
  });
});
