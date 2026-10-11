/**
 * 2D overlays drawn over a rendered frame: a text block in a corner (title, caption, step and time), the colour
 * bar of a scalar colouring, and the axes triad. The layouts are pure (sizes and positions from the canvas size);
 * the draw functions only call the 2D context.
 */

export type Corner = 'tl' | 'tr' | 'bl' | 'br';
export type FontSize = 's' | 'm' | 'l';

export const CORNER_LABEL: Record<Corner, string> = { tl: 'Top left', tr: 'Top right', bl: 'Bottom left', br: 'Bottom right' };
export const FONT_LABEL: Record<FontSize, string> = { s: 'Small', m: 'Medium', l: 'Large' };

/** Font size as a fraction of the frame height. */
export const FONT_FRACTION: Record<FontSize, number> = { s: 0.025, m: 0.035, l: 0.05 };

export interface TextLine {
  text: string;
  bold?: boolean;
  /** Multiplies the base font size (the title is larger). */
  scale?: number;
}

export interface PlacedLine {
  text: string;
  x: number;
  /** Baseline y. */
  y: number;
  px: number;
  bold: boolean;
}

export interface TextBlockLayout {
  box: { x: number; y: number; w: number; h: number };
  lines: PlacedLine[];
  /** Base font size in pixels. */
  px: number;
}

/**
 * The text block in a corner of a W × H frame. `measure` gives the width of a line at a size (the canvas
 * measureText in the browser). Empty lines are dropped; null when there is nothing to draw.
 */
export const layoutTextBlock = (
  W: number,
  H: number,
  corner: Corner,
  lines: readonly TextLine[],
  size: FontSize,
  measure: (text: string, px: number, bold: boolean) => number,
): TextBlockLayout | null => {
  const kept = lines.filter((l) => l.text.trim() !== '');
  if (kept.length === 0) return null;
  const px = Math.max(10, Math.round(H * FONT_FRACTION[size]));
  const pad = Math.round(px * 0.5);
  const sized = kept.map((l) => {
    const lpx = Math.round(px * (l.scale ?? 1));
    const lh = Math.round(lpx * 1.35);
    return { ...l, lpx, lh, width: Math.ceil(measure(l.text, lpx, l.bold === true)) };
  });
  const innerW = sized.reduce((m, l) => (l.width > m ? l.width : m), 0);
  const innerH = sized.reduce((s, l) => s + l.lh, 0);
  const w = innerW + 2 * pad;
  const h = innerH + 2 * pad;
  const x = corner[1] === 'l' ? pad : W - pad - w;
  const y = corner[0] === 't' ? pad : H - pad - h;
  const placed: PlacedLine[] = [];
  let top = y + pad;
  for (const l of sized) {
    placed.push({ text: l.text, x: x + pad, y: top + Math.round(l.lh * 0.78), px: l.lpx, bold: l.bold === true });
    top += l.lh;
  }
  return { box: { x, y, w, h }, lines: placed, px };
};

export interface ColourBarLayout {
  x: number;
  y: number;
  w: number;
  h: number;
  /** Label size in pixels. */
  px: number;
  /** Where the labels sit: right of the bar, so they clear the bar's left edge. */
  labelX: number;
  titleY: number;
}

/** Colour bar on the right edge, vertically centred: the bar, its labels to its left, and the title above. */
export const colourBarLayout = (W: number, H: number): ColourBarLayout => {
  const px = Math.max(10, Math.round(H * 0.028));
  const pad = Math.round(px * 0.8);
  const w = Math.max(12, Math.round(H * 0.035));
  const h = Math.round(H * 0.42);
  const x = W - pad - w;
  const y = Math.round((H - h) / 2);
  return { x, y, w, h, px, labelX: x - Math.round(px * 0.5), titleY: y - Math.round(px * 0.8) };
};

export interface TriadLayout {
  cx: number;
  cy: number;
  len: number;
  px: number;
}

/** Axes triad in the bottom-left corner: its origin and the length of its arrows. */
export const triadLayout = (W: number, H: number): TriadLayout => {
  const len = Math.max(24, Math.round(H * 0.06));
  const margin = Math.round(len * 0.6);
  return { cx: margin + len, cy: H - margin - len, len, px: Math.max(10, Math.round(len * 0.45)) };
};

/** Quaternion [x, y, z, w] (unit) rotating a vector: v + 2w(u×v) + 2u×(u×v), u = (x, y, z). */
export const rotateByQuat = (q: readonly [number, number, number, number], v: readonly [number, number, number]): [number, number, number] => {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  const cx = qy * vz - qz * vy;
  const cy = qz * vx - qx * vz;
  const cz = qx * vy - qy * vx;
  const dx = qy * cz - qz * cy;
  const dy = qz * cx - qx * cz;
  const dz = qx * cy - qy * cx;
  return [vx + 2 * (qw * cx + dx), vy + 2 * (qw * cy + dy), vz + 2 * (qw * cz + dz)];
};

export interface TriadArrow {
  axis: 'x' | 'y' | 'z';
  /** Screen direction in -1..1 (y points down the canvas). */
  dx: number;
  dy: number;
  /** Depth toward the viewer (positive is nearer); arrows pointing away are drawn fainter. */
  depth: number;
}

/**
 * The three world axes as seen from a camera with orientation q (the camera's world quaternion). A world vector
 * is turned into camera space by the conjugate of q; camera +x is screen right and camera +y is screen up.
 */
export const triadArrows = (q: readonly [number, number, number, number]): TriadArrow[] => {
  const conj: [number, number, number, number] = [-q[0], -q[1], -q[2], q[3]];
  const axes: [TriadArrow['axis'], [number, number, number]][] = [
    ['x', [1, 0, 0]],
    ['y', [0, 1, 0]],
    ['z', [0, 0, 1]],
  ];
  return axes.map(([axis, e]) => {
    const [cx, cy, cz] = rotateByQuat(conj, e);
    return { axis, dx: cx, dy: -cy, depth: cz };
  });
};

export const AXIS_COLOUR: Record<TriadArrow['axis'], string> = { x: '#ff6b6b', y: '#6bd96b', z: '#6ba8ff' };

/** Draws the text block (translucent box, white text). */
export const drawTextBlock = (ctx: CanvasRenderingContext2D, block: TextBlockLayout): void => {
  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(block.box.x, block.box.y, block.box.w, block.box.h);
  ctx.fillStyle = '#ffffff';
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  for (const l of block.lines) {
    ctx.font = `${l.bold ? 'bold ' : ''}${l.px}px ui-sans-serif, system-ui, sans-serif`;
    ctx.fillText(l.text, l.x, l.y);
  }
};

/** Draws the colour bar: a gradient of the scalar map with its title and the low and high labels. */
export const drawColourBar = (
  ctx: CanvasRenderingContext2D,
  layout: ColourBarLayout,
  title: string,
  low: string,
  high: string,
  stops: readonly [number, number, number][],
): void => {
  const grad = ctx.createLinearGradient(0, layout.y + layout.h, 0, layout.y);
  stops.forEach((c, i) => {
    const r = Math.round(c[0] * 255);
    const g = Math.round(c[1] * 255);
    const b = Math.round(c[2] * 255);
    grad.addColorStop(i / (stops.length - 1), `rgb(${r}, ${g}, ${b})`);
  });
  ctx.fillStyle = grad;
  ctx.fillRect(layout.x, layout.y, layout.w, layout.h);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
  ctx.lineWidth = Math.max(1, Math.round(layout.px / 10));
  ctx.strokeRect(layout.x, layout.y, layout.w, layout.h);
  ctx.font = `${layout.px}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  const tw = ctx.measureText(title).width + layout.px;
  ctx.fillRect(layout.labelX - tw + layout.px / 2, layout.titleY - layout.px, tw, layout.px * 1.4);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(title, layout.labelX, layout.titleY);
  ctx.fillText(high, layout.labelX, layout.y + layout.px * 0.35);
  ctx.fillText(low, layout.labelX, layout.y + layout.h);
};

/** Draws the axes triad: three arrows from the origin, labelled x, y and z (far arrows fainter). */
export const drawTriad = (ctx: CanvasRenderingContext2D, layout: TriadLayout, arrows: readonly TriadArrow[]): void => {
  ctx.lineWidth = Math.max(2, Math.round(layout.len / 14));
  ctx.font = `bold ${layout.px}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const a of arrows) {
    const ex = layout.cx + a.dx * layout.len;
    const ey = layout.cy + a.dy * layout.len;
    ctx.globalAlpha = a.depth >= 0 ? 1 : 0.45;
    ctx.strokeStyle = AXIS_COLOUR[a.axis];
    ctx.fillStyle = AXIS_COLOUR[a.axis];
    ctx.beginPath();
    ctx.moveTo(layout.cx, layout.cy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    const lx = layout.cx + a.dx * layout.len * 1.22;
    const ly = layout.cy + a.dy * layout.len * 1.22;
    ctx.fillText(a.axis, lx, ly);
  }
  ctx.globalAlpha = 1;
};
