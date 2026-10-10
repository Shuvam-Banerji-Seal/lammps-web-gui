import React, { useMemo, useState } from 'react';
import { Download } from 'lucide-react';
import { parseScript } from '../../../lammps/scriptParser';
import { downloadFlowchart, flowchartToSVG } from '../../../lammps/flowchartSvg';
import { isDarkTheme, getThemeTokens, type Theme } from '../../../theme';
import { cellBtn } from './CellToolbar';

/**
 * The flowchart of one code cell: the cell's text is imported with the Script Builder importer
 * (parseScript), drawn by flowchartToSVG and shown as an <img> with a data: URL, so no markup
 * from the cell reaches the page. Lines the Builder does not know show as raw steps and are counted.
 */

interface CellFlowchartProps {
  theme: Theme;
  /** The cell number, for labels. */
  n: number;
  text: string;
}

const CellFlowchart: React.FC<CellFlowchartProps> = ({ theme, n, text }) => {
  const ct = getThemeTokens(theme);
  const btn = cellBtn(ct);
  const [message, setMessage] = useState<string | null>(null);

  const built = useMemo(() => {
    try {
      if (text.trim() === '') return { error: 'The cell is empty, so there is nothing to draw.', model: null, svg: null, raw: 0 };
      const { model, stats } = parseScript(text, `Cell ${n}`);
      const svg = flowchartToSVG(model, { theme: isDarkTheme(theme) ? 'dark' : 'light' });
      return { error: null, model, svg, raw: stats.raw };
    } catch {
      return { error: 'This cell could not be drawn as a flowchart.', model: null, svg: null, raw: 0 };
    }
  }, [text, n, theme]);

  const src = built.svg ? `data:image/svg+xml;charset=utf-8,${encodeURIComponent(built.svg)}` : null;

  const download = async () => {
    if (!built.model) return;
    setMessage(null);
    try {
      await downloadFlowchart(built.model, 'png', { theme: isDarkTheme(theme) ? 'dark' : 'light' });
    } catch {
      setMessage('PNG export failed in this browser.');
    }
  };

  return (
    <div className={`mt-1 rounded border p-2 ${ct.divider}`} aria-label={`Cell ${n} flowchart`} role="region">
      {built.error && <p className={`text-xs ${ct.muted}`} role="status">{built.error}</p>}
      {src && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`text-[11px] ${ct.muted}`}>
              {built.raw > 0
                ? `${built.raw} line${built.raw === 1 ? '' : 's'} not in the Script Builder catalog, shown as raw lines.`
                : 'Every line is a Script Builder command.'}
            </span>
            <button type="button" className={`${btn} ml-auto ${ct.button}`} aria-label={`Download flowchart of cell ${n} as PNG`}
              title="Download this flowchart as a PNG" onClick={() => void download()}>
              <Download size={12} aria-hidden="true" />PNG
            </button>
          </div>
          {message && <p className={`mt-1 text-xs ${ct.danger}`} role="status">{message}</p>}
          <div className="mt-2 max-h-96 overflow-auto">
            <img src={src} alt={`Flowchart of cell ${n}`} className="block h-auto max-w-none" style={{ width: 'min(100%, 460px)' }} />
          </div>
        </>
      )}
    </div>
  );
};

export default CellFlowchart;
