/**
 * Presentable flowchart rendering — generates a standalone SVG document
 * from a ScriptModel (no DOM screenshotting, no dependencies), so it is
 * crisp at any size and trivially exportable.
 *
 * PNG export draws the same SVG onto an offscreen canvas at 2x scale.
 */

import { COMMAND_BY_ID, SECTION_LABELS, ScriptModel } from './catalog';
import { deriveFlowchart } from './generator';

export interface FlowchartSvgOptions {
  /** 'dark' coffee theme or 'light' paper theme. */
  theme?: 'dark' | 'light';
  /** Show the parameter summary line under each command. */
  showParams?: boolean;
  /** Show disabled steps (dashed). Default true. */
  showDisabled?: boolean;
  /** Footer credit. */
  credit?: string;
}

interface Palette {
  bg: string;
  card: string;
  cardDisabled: string;
  border: string;
  borderSelected: string;
  text: string;
  subtext: string;
  muted: string;
  accent: string;
  startBg: string;
  startBorder: string;
  endBg: string;
  endBorder: string;
  arrow: string;
}

const PALETTES: Record<'dark' | 'light', Palette> = {
  dark: {
    bg: '#16130f', card: '#1e1913', cardDisabled: '#241f18',
    border: '#3f3526', borderSelected: '#7fa66b', text: '#ede5d8',
    subtext: '#a3937f', muted: '#6f6353', accent: '#9dc48b',
    startBg: '#22301c', startBorder: '#47693b', endBg: '#332612',
    endBorder: '#6b5124', arrow: '#659054',
  },
  light: {
    bg: '#f4efe6', card: '#ffffff', cardDisabled: '#f0ebe0',
    border: '#ddd2bd', borderSelected: '#4e7a41', text: '#2e2920',
    subtext: '#5d5344', muted: '#a2937c', accent: '#3c5c32',
    startBg: '#e7efdf', startBorder: '#4e7a41', endBg: '#f7ecd7',
    endBorder: '#caa15c', arrow: '#4e7a41',
  },
};

const esc = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const CARD_W = 460;
const CARD_H = 64;
const GAP = 46;
const TOP_H = 92;
const BOTTOM_H = 76;

/** Render the model as a standalone SVG string. */
export const flowchartToSVG = (model: ScriptModel, opts: FlowchartSvgOptions = {}): string => {
  const theme = opts.theme ?? 'dark';
  const p = PALETTES[theme];
  const showParams = opts.showParams ?? true;
  const showDisabled = opts.showDisabled ?? true;
  const credit = opts.credit ?? 'Molecule3D — LAMMPS Workbench';

  const flow = deriveFlowchart(model);
  const nodes = flow.nodes.filter(n => showDisabled || n.enabled);

  const height = TOP_H + nodes.length * (CARD_H + GAP) + BOTTOM_H;
  const width = CARD_W + 160;
  const cx = width / 2;

  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" ` +
    `viewBox="0 0 ${width} ${height}" font-family="ui-sans-serif, system-ui, 'Segoe UI', Arial, sans-serif">`,
  );
  parts.push(`<rect width="${width}" height="${height}" fill="${p.bg}" rx="12"/>`);

  // Title + subtitle
  const title = model.title || 'LAMMPS pipeline';
  const date = new Date().toISOString().slice(0, 10);
  parts.push(
    `<text x="${cx}" y="34" text-anchor="middle" font-size="17" font-weight="700" fill="${p.text}">${esc(title)}</text>`,
    `<text x="${cx}" y="54" text-anchor="middle" font-size="11" fill="${p.muted}">${esc(credit)} · ${date} · ${nodes.length} steps</text>`,
  );

  let y = TOP_H;
  const cardX = cx - CARD_W / 2;

  const pill = (label: string, fill: string, border: string, textColor: string) => {
    const w = 74;
    parts.push(
      `<rect x="${cx - w / 2}" y="${y}" width="${w}" height="22" rx="11" fill="${fill}" stroke="${border}" stroke-width="1"/>`,
      `<text x="${cx}" y="${y + 15}" text-anchor="middle" font-size="10" font-weight="600" fill="${textColor}">${esc(label)}</text>`,
    );
  };

  const arrowDown = () => {
    parts.push(
      `<line x1="${cx}" y1="${y + 4}" x2="${cx}" y2="${y + GAP - 18}" stroke="${p.arrow}" stroke-width="1.5"/>`,
      `<path d="M ${cx - 4} ${y + GAP - 22} L ${cx} ${y + GAP - 16} L ${cx + 4} ${y + GAP - 22} Z" fill="${p.arrow}"/>`,
    );
  };

  pill('START', p.startBg, p.startBorder, p.accent);
  y += 40;
  arrowDown();
  y += GAP - 8;

  for (const node of nodes) {
    const def = COMMAND_BY_ID[node.uid ? (model.steps.find(s => s.uid === node.uid)?.defId ?? '') : ''];
    const sectionLabel =
      SECTION_LABELS[node.section as keyof typeof SECTION_LABELS]?.split('·')[1]?.trim() ?? node.section;
    const dashed = node.enabled ? '' : ` stroke-dasharray="5 4" opacity="0.65"`;

    parts.push(
      `<rect x="${cardX}" y="${y}" width="${CARD_W}" height="${CARD_H}" rx="12" fill="${node.enabled ? p.card : p.cardDisabled}" stroke="${p.border}" stroke-width="1.2"${dashed}/>`,
      `<text x="${cardX + 16}" y="${y + 20}" font-size="9" letter-spacing="1" fill="${p.muted}">${esc(sectionLabel.toUpperCase())}</text>`,
      `<text x="${cardX + 16}" y="${y + 40}" font-size="14" font-weight="700" fill="${p.accent}">${esc(node.label)}</text>`,
    );
    if (showParams && node.sublabel) {
      const sub = node.sublabel.length > 52 ? node.sublabel.slice(0, 51) + '…' : node.sublabel;
      parts.push(
        `<text x="${cardX + 16}" y="${y + 56}" font-size="10" font-family="ui-monospace, 'Cascadia Mono', Consolas, monospace" fill="${p.subtext}">${esc(sub)}</text>`,
      );
    }
    void def;
    y += CARD_H;
    if (node !== nodes[nodes.length - 1]) {
      arrowDown();
      y += GAP;
    }
  }

  y += 8;
  arrowDown();
  y += GAP - 8;
  pill('END', p.endBg, p.endBorder, '#e4b877');

  parts.push('</svg>');
  return parts.join('\n');
};

/** Render the SVG to a PNG blob at 2x scale (browser only). */
export const flowchartToPngBlob = async (
  model: ScriptModel,
  opts: FlowchartSvgOptions = {},
): Promise<Blob | null> => {
  const svg = flowchartToSVG(model, opts);
  const widthMatch = svg.match(/width="(\d+)"/);
  const heightMatch = svg.match(/height="(\d+)"/);
  if (!widthMatch || !heightMatch) return null;
  const w = parseInt(widthMatch[1], 10);
  const h = parseInt(heightMatch[1], 10);

  const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('SVG rasterization failed'));
      img.src = url;
    });
    const scale = 2;
    const canvas = document.createElement('canvas');
    canvas.width = w * scale;
    canvas.height = h * scale;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    return await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/png'));
  } finally {
    URL.revokeObjectURL(url);
  }
};

/** Trigger a browser download for either format. */
export const downloadFlowchart = async (
  model: ScriptModel,
  format: 'svg' | 'png',
  opts: FlowchartSvgOptions = {},
): Promise<void> => {
  const base = (model.title || 'lammps-flowchart')
    .replace(/[^\w.-]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^[-.]+|[-.]+$/g, '') || 'lammps-flowchart';
  if (format === 'svg') {
    const svg = flowchartToSVG(model, opts);
    downloadText(svg, `${base}.svg`, 'image/svg+xml;charset=utf-8');
    return;
  }
  const blob = await flowchartToPngBlob(model, opts);
  if (!blob) throw new Error('PNG export failed');
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${base}.png`;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};

const downloadText = (text: string, filename: string, mime: string): void => {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};
