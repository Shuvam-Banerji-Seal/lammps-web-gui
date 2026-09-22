import React from 'react';
import { getThemeTokens, Theme } from '../../theme';

export interface Point { x: number; y: number; }

/** Extent without spreading the array into Math.min/Math.max. */
export const extent = (values: number[]): { lo: number; hi: number } => {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) ? { lo, hi } : { lo: 0, hi: 1 };
};

/**
 * Plot width in CSS pixels. The chart is drawn in this viewBox and scaled, so
 * there is nothing to gain from more than ~2 points per unit of it.
 */
const PLOT_PX = 320;

/**
 * Min/max decimation.
 *
 * An MSD curve has one point per lag, so a few-thousand-frame trajectory used
 * to emit a few thousand sub-pixel `L` segments into a 320px-wide SVG. Plain
 * stride sampling would hide peaks, which for g(r) is exactly the information
 * the chart exists to show — so each bucket contributes both its minimum and
 * its maximum, in x order.
 */
export const decimate = (data: Point[], budget = PLOT_PX * 2): Point[] => {
  if (data.length <= budget) return data;
  const buckets = Math.max(1, Math.floor(budget / 2));
  const size = data.length / buckets;
  const out: Point[] = [];
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * size);
    const end = Math.min(data.length, Math.floor((b + 1) * size));
    if (end <= start) continue;
    let lo = data[start];
    let hi = data[start];
    for (let i = start + 1; i < end; i++) {
      if (data[i].y < lo.y) lo = data[i];
      if (data[i].y > hi.y) hi = data[i];
    }
    if (lo.x <= hi.x) {
      out.push(lo);
      if (hi !== lo) out.push(hi);
    } else {
      out.push(hi);
      out.push(lo);
    }
  }
  return out;
};

interface LineChartProps {
  data: Point[];
  xLabel?: string;
  yLabel?: string;
  color?: string;
  fillColor?: string;
  theme: Theme;
  height?: number;
  yMin?: number;
  yMax?: number;
}

export const LineChart: React.FC<LineChartProps> = ({
  data, xLabel, yLabel, color, fillColor, theme, height = 160, yMin, yMax,
}) => {
  const ct = getThemeTokens(theme);
  const isDark = theme === 'dark';
  if (data.length === 0) {
    return <div className={`flex h-[160px] items-center justify-center text-xs ${ct.muted}`}>No data</div>;
  }
  // Extents come from the FULL series so decimation cannot change the axes.
  const xe = extent(data.map(d => d.x));
  const ye = extent(data.map(d => d.y));
  const xMin = xe.lo;
  const xMax = xe.hi;
  const yLo = yMin ?? Math.min(0, ye.lo);
  const yHi = yMax ?? ye.hi;
  const yRange = yHi - yLo || 1;
  const xRange = xMax - xMin || 1;

  const plotted = decimate(data);

  const W = PLOT_PX, H = height, padL = 36, padR = 12, padT = 12, padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const sx = (x: number) => padL + ((x - xMin) / xRange) * plotW;
  const sy = (y: number) => padT + (1 - (y - yLo) / yRange) * plotH;

  const path = plotted
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`)
    .join(' ');
  const fillPath = plotted.length > 1
    ? `${path} L ${sx(plotted[plotted.length - 1].x).toFixed(1)} ${sy(yLo).toFixed(1)} ` +
      `L ${sx(plotted[0].x).toFixed(1)} ${sy(yLo).toFixed(1)} Z`
    : '';

  // ticks
  const xTicks = 4, yTicks = 4;
  const gridColor = isDark ? '#332a1f' : '#e0d7c6';
  const textColor = isDark ? '#a3937f' : '#7c7060';
  const lineColor = color ?? (isDark ? '#7fa66b' : '#4e7a41');
  const areaFill = fillColor ?? (isDark ? 'rgba(127,166,107,0.18)' : 'rgba(78,122,65,0.12)');

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto select-none" role="img">
        {/* grid */}
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const y = padT + (i / yTicks) * plotH;
          return <line key={`y${i}`} x1={padL} x2={W - padR} y1={y} y2={y} stroke={gridColor} strokeWidth={0.5} opacity={0.7} />;
        })}
        {Array.from({ length: xTicks + 1 }).map((_, i) => {
          const x = padL + (i / xTicks) * plotW;
          return <line key={`x${i}`} x1={x} x2={x} y1={padT} y2={H - padB} stroke={gridColor} strokeWidth={0.5} opacity={0.4} />;
        })}
        {/* axes */}
        <line x1={padL} x2={padL} y1={padT} y2={H - padB} stroke={gridColor} strokeWidth={1} />
        <line x1={padL} x2={W - padR} y1={H - padB} y2={H - padB} stroke={gridColor} strokeWidth={1} />
        {/* area */}
        {fillPath && <path d={fillPath} fill={areaFill} stroke="none" />}
        {/* line */}
        <path d={path} fill="none" stroke={lineColor} strokeWidth={1.7} strokeLinejoin="round" strokeLinecap="round" />
        {/* y labels */}
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const v = yHi - (i / yTicks) * yRange;
          const y = padT + (i / yTicks) * plotH;
          return <text key={`yl${i}`} x={padL - 4} y={y + 3} textAnchor="end" fontSize={8} fill={textColor}>{v.toFixed(v >= 10 ? 1 : 2)}</text>;
        })}
        {/* x labels */}
        {Array.from({ length: xTicks + 1 }).map((_, i) => {
          const v = xMin + (i / xTicks) * xRange;
          const x = padL + (i / xTicks) * plotW;
          return <text key={`xl${i}`} x={x} y={H - 4} textAnchor="middle" fontSize={8} fill={textColor}>{v.toFixed(v >= 10 ? 1 : 2)}</text>;
        })}
      </svg>
      {(xLabel || yLabel) && (
        <div className={`flex justify-between text-[10px] ${ct.muted} -mt-1 px-1`}>
          <span>{yLabel ?? ''}</span>
          <span>{xLabel ?? ''}</span>
        </div>
      )}
    </div>
  );
};

interface HistogramProps {
  bins: { x0: number; x1: number; count: number; density: number }[];
  xLabel?: string;
  yLabel?: string;
  color?: string;
  theme: Theme;
  height?: number;
}

export const Histogram: React.FC<HistogramProps> = ({ bins, xLabel, yLabel, color, theme, height = 160 }) => {
  const ct = getThemeTokens(theme);
  const isDark = theme === 'dark';
  if (bins.length === 0) {
    return <div className={`flex h-[160px] items-center justify-center text-xs ${ct.muted}`}>No data</div>;
  }
  const W = 320, H = height, padL = 36, padR = 12, padT = 12, padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const maxCount = Math.max(1, extent(bins.map(b => b.count)).hi);
  const barW = plotW / bins.length;
  const gridColor = isDark ? '#332a1f' : '#e0d7c6';
  const textColor = isDark ? '#a3937f' : '#7c7060';
  const barColor = color ?? (isDark ? '#7fa66b' : '#4e7a41');
  const yTicks = 4;
  const xTicks = 4;

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto select-none" role="img">
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const y = padT + (i / yTicks) * plotH;
          return <line key={`y${i}`} x1={padL} x2={W - padR} y1={y} y2={y} stroke={gridColor} strokeWidth={0.5} opacity={0.7} />;
        })}
        <line x1={padL} x2={padL} y1={padT} y2={H - padB} stroke={gridColor} strokeWidth={1} />
        <line x1={padL} x2={W - padR} y1={H - padB} y2={H - padB} stroke={gridColor} strokeWidth={1} />
        {bins.map((b, i) => {
          const h = (b.count / maxCount) * plotH;
          const x = padL + i * barW;
          const y = padT + plotH - h;
          return <rect key={i} x={x + 0.5} y={y} width={Math.max(1, barW - 0.5)} height={h} fill={barColor} opacity={0.85} rx={1} />;
        })}
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const v = maxCount - (i / yTicks) * maxCount;
          const y = padT + (i / yTicks) * plotH;
          return <text key={`yl${i}`} x={padL - 4} y={y + 3} textAnchor="end" fontSize={8} fill={textColor}>{Math.round(v)}</text>;
        })}
        {Array.from({ length: xTicks + 1 }).map((_, i) => {
          const idx = Math.floor((i / xTicks) * (bins.length - 1));
          const b = bins[idx];
          const x = padL + (idx + 0.5) * barW;
          const v = (b.x0 + b.x1) / 2;
          return <text key={`xl${i}`} x={x} y={H - 4} textAnchor="middle" fontSize={8} fill={textColor}>{v.toFixed(1)}</text>;
        })}
      </svg>
      {(xLabel || yLabel) && (
        <div className={`flex justify-between text-[10px] ${ct.muted} -mt-1 px-1`}>
          <span>{yLabel ?? ''}</span>
          <span>{xLabel ?? ''}</span>
        </div>
      )}
    </div>
  );
};
