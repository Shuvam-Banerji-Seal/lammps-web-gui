/**
 * A small, safe Markdown parser for notebook text cells. It produces a plain
 * block/inline AST; the React renderer (components/workbench/notebook/MarkdownView.tsx)
 * builds elements from it. Nothing here produces HTML: raw `<tags>` and
 * `javascript:` links stay as text, and only http, https and mailto links
 * are kept as links.
 *
 * Supported: # ## ### headings, paragraphs, **bold**, *italic*, `code`,
 * fenced ``` code blocks (verbatim), - / * / + and 1. lists, > quotes,
 * --- rules, [text](https://...) links and simple pipe tables.
 */

export type Inline =
  | { t: 'text'; v: string }
  | { t: 'code'; v: string }
  | { t: 'strong'; c: Inline[] }
  | { t: 'em'; c: Inline[] }
  | { t: 'link'; href: string; c: Inline[] };

export type Align = 'left' | 'right' | 'center' | null;

export type Block =
  | { t: 'heading'; level: 1 | 2 | 3; c: Inline[] }
  | { t: 'p'; c: Inline[] }
  | { t: 'code'; v: string; lang: string }
  | { t: 'ul'; items: Inline[][] }
  | { t: 'ol'; start: number; items: Inline[][] }
  | { t: 'quote'; blocks: Block[] }
  | { t: 'hr' }
  | { t: 'table'; head: Inline[][]; align: Align[]; rows: Inline[][][] };

/**
 * Link targets allowed in a rendered cell. docs.lammps.org is not involved; this is a
 * safety rule: only http:, https: and mailto: targets, with no whitespace or control characters.
 */
export const isSafeHref = (href: string): boolean =>
  /^(https?:\/\/[^\s\u0000-\u001f]+|mailto:[^\s\u0000-\u001f]+)$/i.test(href);

const HEADING = /^(#{1,3})[ \t]+(.+?)[ \t]*$/;
const FENCE = /^[ \t]*```(.*)$/;
const HR = /^[ \t]*(?:-[ \t]*){3,}$|^[ \t]*(?:\*[ \t]*){3,}$|^[ \t]*(?:_[ \t]*){3,}$/;
const QUOTE = /^[ \t]*>[ \t]?(.*)$/;
const UL = /^[ \t]*[-*+][ \t]+(.*)$/;
const OL = /^[ \t]*(\d+)[.)][ \t]+(.*)$/;
const TABLE_SEP = /^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/;

const isBlank = (line: string): boolean => line.trim() === '';

const isBlockStart = (line: string, next: string | undefined): boolean =>
  FENCE.test(line) || HEADING.test(line) || HR.test(line) || QUOTE.test(line) ||
  UL.test(line) || OL.test(line) || isTableStart(line, next);

const isTableStart = (line: string, next: string | undefined): boolean =>
  next !== undefined && line.includes('|') && TABLE_SEP.test(next) && next.includes('-');

/** Split a table row on unescaped pipes; a leading and trailing pipe are optional. */
const splitRow = (line: string): string[] => {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
  const cells: string[] = [];
  let cur = '';
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '\\' && s[i + 1] === '|') {
      cur += '|';
      i++;
    } else if (s[i] === '|') {
      cells.push(cur.trim());
      cur = '';
    } else {
      cur += s[i];
    }
  }
  cells.push(cur.trim());
  return cells;
};

const parseAlign = (sep: string): Align[] =>
  splitRow(sep).map((c) => {
    const l = c.startsWith(':');
    const r = c.endsWith(':');
    if (l && r) return 'center';
    if (r) return 'right';
    if (l) return 'left';
    return null;
  });

/**
 * Index of a closing emphasis delimiter (`len` copies of `ch`) at or after `from`. The closer must not
 * follow whitespace and must not be part of a longer run, so `*a **b** c*` closes on the final `*`.
 */
const findClose = (s: string, from: number, ch: string, len: number): number => {
  const run = ch.repeat(len);
  let j = s.indexOf(run, from);
  while (j !== -1) {
    const before = s[j - 1];
    const after = s[j + len];
    if (before !== ch && after !== ch && !/\s/.test(before ?? ' ')) return j;
    j = s.indexOf(run, j + 1);
  }
  return -1;
};

/** Inline parser: code spans, links, **bold** and *italic* (nesting allowed). Everything else is text. */
export const parseInline = (s: string): Inline[] => {
  const out: Inline[] = [];
  let buf = '';
  const flush = () => {
    if (buf) out.push({ t: 'text', v: buf });
    buf = '';
  };
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === '`') {
      const close = s.indexOf('`', i + 1);
      if (close > i + 1) {
        flush();
        out.push({ t: 'code', v: s.slice(i + 1, close) });
        i = close + 1;
        continue;
      }
    } else if (ch === '[') {
      const m = /^\[([^\]\n]*)\]\(([^)\s]+)\)/.exec(s.slice(i));
      if (m && m[1] !== '' && isSafeHref(m[2])) {
        flush();
        out.push({ t: 'link', href: m[2], c: parseInline(m[1]) });
        i += m[0].length;
        continue;
      }
    } else if (ch === '*') {
      if (s.startsWith('**', i) && !/\s/.test(s[i + 2] ?? ' ')) {
        const close = findClose(s, i + 2, '*', 2);
        if (close !== -1) {
          flush();
          out.push({ t: 'strong', c: parseInline(s.slice(i + 2, close)) });
          i = close + 2;
          continue;
        }
      } else if (!s.startsWith('**', i) && !/\s/.test(s[i + 1] ?? ' ')) {
        const close = findClose(s, i + 1, '*', 1);
        if (close !== -1) {
          flush();
          out.push({ t: 'em', c: parseInline(s.slice(i + 1, close)) });
          i = close + 1;
          continue;
        }
      }
    }
    buf += ch;
    i++;
  }
  flush();
  return out;
};

const parseList = (lines: string[], start: number, re: RegExp): { items: string[]; next: number } => {
  const items: string[] = [];
  let i = start;
  while (i < lines.length) {
    const m = re.exec(lines[i]);
    if (!m) break;
    items.push(re === OL ? m[2] : m[1]);
    i++;
  }
  return { items, next: i };
};

/** Parse Markdown source into blocks. Never throws on odd input; unknown syntax is paragraph text. */
export const parseMarkdown = (source: string): Block[] => {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) {
      i++;
      continue;
    }
    const fence = FENCE.exec(line);
    if (fence) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^[ \t]*```[ \t]*$/.test(lines[i])) {
        body.push(lines[i]);
        i++;
      }
      i++; // closing fence (or end of input)
      blocks.push({ t: 'code', v: body.join('\n'), lang: fence[1].trim() });
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({ t: 'heading', level: heading[1].length as 1 | 2 | 3, c: parseInline(heading[2]) });
      i++;
      continue;
    }
    if (HR.test(line)) {
      blocks.push({ t: 'hr' });
      i++;
      continue;
    }
    if (QUOTE.test(line)) {
      const inner: string[] = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        inner.push(QUOTE.exec(lines[i])![1]);
        i++;
      }
      blocks.push({ t: 'quote', blocks: parseMarkdown(inner.join('\n')) });
      continue;
    }
    if (UL.test(line)) {
      const { items, next } = parseList(lines, i, UL);
      blocks.push({ t: 'ul', items: items.map(parseInline) });
      i = next;
      continue;
    }
    if (OL.test(line)) {
      const start = parseInt(OL.exec(line)![1], 10);
      const { items, next } = parseList(lines, i, OL);
      blocks.push({ t: 'ol', start: Number.isFinite(start) ? start : 1, items: items.map(parseInline) });
      i = next;
      continue;
    }
    if (isTableStart(line, lines[i + 1])) {
      const head = splitRow(line).map(parseInline);
      const align = parseAlign(lines[i + 1]);
      i += 2;
      const rows: Inline[][][] = [];
      while (i < lines.length && !isBlank(lines[i]) && lines[i].includes('|')) {
        rows.push(splitRow(lines[i]).map(parseInline));
        i++;
      }
      blocks.push({ t: 'table', head, align, rows });
      continue;
    }
    // paragraph: consecutive lines until a blank line or another block starts
    const para: string[] = [];
    while (i < lines.length && !isBlank(lines[i]) && (para.length === 0 || !isBlockStart(lines[i], lines[i + 1]))) {
      para.push(lines[i].trim());
      i++;
    }
    blocks.push({ t: 'p', c: parseInline(para.join(' ')) });
  }
  return blocks;
};
