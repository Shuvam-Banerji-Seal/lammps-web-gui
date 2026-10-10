import React from 'react';
import { parseMarkdown, type Align, type Block, type Inline } from '../../../lammps/markdown';
import type { ThemeTokens } from '../../../theme';

/**
 * Renders a notebook Markdown cell from the AST of src/lammps/markdown.ts. Elements are built
 * from the tree (text is a React text node), so no markup from the cell is ever injected.
 */

const renderInline = (nodes: Inline[], keyBase = 'i'): React.ReactNode[] =>
  nodes.map((n, k) => {
    const key = `${keyBase}-${k}`;
    switch (n.t) {
      case 'text':
        return <React.Fragment key={key}>{n.v}</React.Fragment>;
      case 'code':
        return <code key={key} className="rounded bg-black/10 px-1 py-px font-mono text-[0.9em]">{n.v}</code>;
      case 'strong':
        return <strong key={key}>{renderInline(n.c, key)}</strong>;
      case 'em':
        return <em key={key}>{renderInline(n.c, key)}</em>;
      case 'link':
        return (
          <a key={key} href={n.href} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2">
            {renderInline(n.c, key)}
          </a>
        );
    }
  });

const alignClass = (a: Align): string =>
  a === 'right' ? 'text-right' : a === 'center' ? 'text-center' : 'text-left';

const renderBlock = (b: Block, key: string, ct: ThemeTokens): React.ReactNode => {
  switch (b.t) {
    case 'heading': {
      const cls = b.level === 1 ? 'mt-2 text-lg font-semibold' : b.level === 2 ? 'mt-2 text-base font-semibold' : 'mt-1 text-sm font-semibold';
      const inner = renderInline(b.c, key);
      return b.level === 1 ? <h1 key={key} className={`${cls} ${ct.headerText}`}>{inner}</h1>
        : b.level === 2 ? <h2 key={key} className={`${cls} ${ct.headerText}`}>{inner}</h2>
          : <h3 key={key} className={`${cls} ${ct.headerText}`}>{inner}</h3>;
    }
    case 'p':
      return <p key={key} className="leading-relaxed">{renderInline(b.c, key)}</p>;
    case 'code':
      return (
        <pre key={key} className={`overflow-x-auto whitespace-pre rounded border p-2 font-mono text-xs ${ct.divider}`}
          data-lang={b.lang || undefined}>
          <code>{b.v}</code>
        </pre>
      );
    case 'ul':
      return (
        <ul key={key} className="list-disc space-y-0.5 pl-5">
          {b.items.map((it, k) => <li key={k}>{renderInline(it, `${key}-${k}`)}</li>)}
        </ul>
      );
    case 'ol':
      return (
        <ol key={key} start={b.start} className="list-decimal space-y-0.5 pl-5">
          {b.items.map((it, k) => <li key={k}>{renderInline(it, `${key}-${k}`)}</li>)}
        </ol>
      );
    case 'quote':
      return (
        <blockquote key={key} className={`border-l-2 pl-3 ${ct.divider} ${ct.muted}`}>
          {b.blocks.map((inner, k) => renderBlock(inner, `${key}-${k}`, ct))}
        </blockquote>
      );
    case 'hr':
      return <hr key={key} className={`my-2 border-t ${ct.divider}`} />;
    case 'table':
      return (
        <div key={key} className="max-w-full overflow-x-auto">
          <table className={`border-collapse text-xs ${ct.divider}`}>
            <thead>
              <tr>
                {b.head.map((cell, k) => (
                  <th key={k} scope="col" className={`border px-2 py-1 font-semibold ${ct.divider} ${alignClass(b.align[k] ?? null)}`}>
                    {renderInline(cell, `${key}-h${k}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {b.rows.map((row, r) => (
                <tr key={r}>
                  {row.map((cell, k) => (
                    <td key={k} className={`border px-2 py-1 ${ct.divider} ${alignClass(b.align[k] ?? null)}`}>
                      {renderInline(cell, `${key}-${r}-${k}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
  }
};

interface MarkdownViewProps {
  ct: ThemeTokens;
  /** The cell's Markdown source. */
  source: string;
}

const MarkdownView: React.FC<MarkdownViewProps> = ({ ct, source }) => {
  const blocks = React.useMemo(() => parseMarkdown(source), [source]);
  if (blocks.length === 0) {
    return <p className={`text-xs italic ${ct.muted}`}>Empty text cell. Press Edit to write Markdown.</p>;
  }
  return (
    <div className="space-y-2 text-sm" data-markdown="">
      {blocks.map((b, k) => renderBlock(b, `b${k}`, ct))}
    </div>
  );
};

export default MarkdownView;
