import React from 'react';
import { getThemeTokens, Theme } from '../../theme';

interface Point { x: number; y: number; }

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
  const xs = data.map(d => d.x);
  const ys = data.map(d => d.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yLo = yMin ?? Math.min(0, Math.min(...ys));
  const yHi = yMax ?? Math.max(...ys);
  const yRange = yHi - yLo || 1;
  const xRange = xMax - xMin || 1;

  const W = 320, H = height, padL = 36, padR = 12, padT = 12, padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const sx = (x: number) => padL + ((x - xMin) / xRange) * plotW;
  const sy = (y: number) => padT + (1 - (y - yLo) / yRange) * plotH;

  const path = data.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join(' ');
  const fillPath = data.length > 1
    ? `${path} L ${sx(data[data.length - 1].x).toFixed(1)} ${sy(yLo).toFixed(1)} L ${sx(data[0].x).toFixed(1)} ${sy(yLo).toFixed(1)} Z`
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
  const maxCount = Math.max(...bins.map(b => b.count), 1);
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
