import React from 'react';
import { getThemeTokens, isDarkTheme, Theme } from '../../../../theme';

/** Saves a blob under a file name (an object URL, clicked and revoked). */
export const saveBlob = (name: string, blob: Blob): void => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const safeName = (s: string): string => s.replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '') || 'chart';

/** The chart's SVG as text, with the XML namespace the standalone file needs. */
const svgText = (svg: SVGSVGElement): string => {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.removeAttribute('class');
  clone.setAttribute('width', String(svg.viewBox.baseVal.width || 320));
  clone.setAttribute('height', String(svg.viewBox.baseVal.height || 160));
  return new XMLSerializer().serializeToString(clone);
};

/** PNG of the chart at 2x, on the theme's background. Rejects when the browser cannot draw it. */
const pngOf = (svg: SVGSVGElement, background: string): Promise<Blob> =>
  new Promise((resolve, reject) => {
    const W = svg.viewBox.baseVal.width || 320;
    const H = svg.viewBox.baseVal.height || 160;
    const url = URL.createObjectURL(new Blob([svgText(svg)], { type: 'image/svg+xml' }));
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = W * 2;
      canvas.height = H * 2;
      const ctx = canvas.getContext('2d');
      if (!ctx) { URL.revokeObjectURL(url); reject(new Error('no 2d canvas')); return; }
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encoding failed'))), 'image/png');
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('could not draw the chart')); };
    img.src = url;
  });

interface ChartActionsProps {
  name: string;
  theme: Theme;
  /** The <svg> itself, or an element that contains it (a wrapper around a chart from elsewhere). */
  svgRef: React.RefObject<Element | null>;
  /** CSV text for the chart's data (built on demand). */
  csv: () => string;
}

/** CSV / SVG / PNG download buttons for one chart. */
export const ChartActions: React.FC<ChartActionsProps> = ({ name, theme, svgRef, csv }) => {
  const ct = getThemeTokens(theme);
  const svgNow = (): SVGSVGElement | null => {
    const el = svgRef.current;
    if (!el) return null;
    return el instanceof SVGSVGElement ? el : el.querySelector('svg');
  };
  const [notice, setNotice] = React.useState<string | null>(null);
  const base = safeName(name);
  const btn = `rounded px-1.5 py-0.5 text-[10px] font-medium ${ct.hoverSurface}`;
  const onCsv = () => saveBlob(`${base}.csv`, new Blob([csv()], { type: 'text/csv' }));
  const onSvg = () => {
    const svg = svgNow();
    if (!svg) return;
    saveBlob(`${base}.svg`, new Blob([svgText(svg)], { type: 'image/svg+xml' }));
  };
  const onPng = async () => {
    const svg = svgNow();
    if (!svg) return;
    try {
      const background = isDarkTheme(theme) ? '#1b1712' : '#ffffff';
      saveBlob(`${base}.png`, await pngOf(svg, background));
      setNotice(null);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'PNG export failed');
    }
  };
  return (
    <span className={`inline-flex items-center gap-1 ${ct.muted}`}>
      <button type="button" className={btn} onClick={onCsv} title={`Download ${base}.csv`}>CSV</button>
      <button type="button" className={btn} onClick={onSvg} title={`Download ${base}.svg`}>SVG</button>
      <button type="button" className={btn} onClick={() => void onPng()} title={`Download ${base}.png`}>PNG</button>
      {notice && <span role="alert" className="text-[10px]">{notice}</span>}
    </span>
  );
};
