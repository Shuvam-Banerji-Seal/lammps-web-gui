import React, { useMemo, useRef, useState } from 'react';
import { getThemeTokens, Theme } from '../../../../theme';
import { Histogram, MultiLineChart, type ChartSeries } from '../../../charts/SimpleChart';
import { unitsCaption } from '../../../../lammps/thermoUnits';
import {
  columnLabel, numericColumns, runningMean, segmentDt, seriesPoints, summarize, timeUnitLabel, toCsv,
  type ThermoSegment, type XMode,
} from '../chartData';
import { autocorrelation, blockAverage, histogram } from '../../../../services/thermoAnalysis';
import { ChartActions } from './ChartActions';

/** A chart panel: the columns drawn on it. */
interface Panel { id: number; keys: string[] }

const SMOOTHING = [0, 5, 20, 100] as const;

/** Default panels: temperature, the energies together, pressure, and volume (or density) when it varies. */
const defaultPanels = (columns: string[], segments: ThermoSegment[]): Panel[] => {
  const has = (k: string) => columns.includes(k);
  const varies = (k: string) => {
    const vals: number[] = [];
    for (const s of segments) for (const r of s.rows) if (Number.isFinite(r[k])) vals.push(r[k]);
    const sm = summarize(vals);
    return sm.n > 1 && sm.max > sm.min;
  };
  const out: Panel[] = [];
  const add = (keys: string[]) => {
    const k = keys.filter(has);
    if (k.length) out.push({ id: out.length + 1, keys: k });
  };
  add(['temp']);
  add(['pe', 'ke', 'etotal']);
  add(['press']);
  add(varies('vol') ? ['vol'] : varies('density') ? ['density'] : []);
  return out;
};

interface ThermoChartsProps {
  segments: ThermoSegment[];
  theme: Theme;
}

/**
 * Thermo charts over the run: any numeric column on a panel (several per panel, with a legend), x as step or
 * simulated time (step x timestep), an optional running mean, hover readouts, and CSV / SVG / PNG per panel.
 * Segments are the thermo tables of the notebook's cells, joined end to end when "all runs" is picked.
 */
export const ThermoCharts: React.FC<ThermoChartsProps> = ({ segments, theme }) => {
  const ct = getThemeTokens(theme);
  const [segChoice, setSegChoice] = useState<'all' | number>('all');
  const [xChoice, setXChoice] = useState<XMode | null>(null);
  const [smoothing, setSmoothing] = useState<number>(0);
  const [panels, setPanels] = useState<Panel[] | null>(null);
  const nextId = useRef(100);

  const columns = useMemo(() => numericColumns(segments), [segments]);
  const timeOk = segments.some((s) => segmentDt(s) !== null);
  const xMode: XMode = xChoice ?? (timeOk ? 'time' : 'step');
  const active = useMemo(
    () => (segChoice === 'all' ? segments : segments[segChoice] ? [segments[segChoice]] : []),
    [segments, segChoice],
  );
  const effective = useMemo(() => panels ?? defaultPanels(columns, segments), [panels, columns, segments]);
  const units = active[0]?.units ?? segments[0]?.units;
  const dtUsed = active.length ? segmentDt(active[0]) : null;
  const xLabel = xMode === 'step' ? 'step' : timeUnitLabel(units);

  if (segments.length === 0) {
    return <p className={`text-xs ${ct.muted}`}>Thermo charts appear here once a run prints thermo output.</p>;
  }

  const setEffective = (next: Panel[]) => setPanels(next);
  const seriesFor = (keys: string[]): ChartSeries[] => keys.map((k) => {
    const pts: { x: number; y: number }[] = [];
    for (const s of active) for (const p of seriesPoints(s, k, xMode)) pts.push(p);
    return { name: k, points: runningMean(pts, smoothing) };
  });
  const csvFor = (keys: string[]) => {
    const header = ['segment', 'step', 'time', ...keys];
    const rows: (string | number | undefined)[][] = [];
    for (const [si, s] of segments.entries()) {
      if (segChoice !== 'all' && segChoice !== si) continue;
      const dt = segmentDt(s);
      for (const r of s.rows) {
        rows.push([s.label, r.step, dt === null ? undefined : r.step * dt, ...keys.map((k) => r[k])]);
      }
    }
    return toCsv(header, rows);
  };

  return (
    <div className="space-y-3">
      <p className={`text-xs font-semibold ${ct.headerText}`}>Thermo graphs</p>
      <div className={`flex flex-wrap items-center gap-2 text-[11px] ${ct.muted}`}>
        <label className="inline-flex items-center gap-1">Run
          <select aria-label="Thermo run" className={`rounded border px-1 py-0.5 ${ct.input}`} value={segChoice}
            onChange={(e) => setSegChoice(e.target.value === 'all' ? 'all' : Number(e.target.value))}>
            <option value="all">all runs</option>
            {segments.map((s, i) => <option key={i} value={i}>{s.label}</option>)}
          </select>
        </label>
        <label className="inline-flex items-center gap-1">x
          <select aria-label="x axis" className={`rounded border px-1 py-0.5 ${ct.input}`} value={xMode}
            onChange={(e) => setXChoice(e.target.value as XMode)}>
            <option value="step">step</option>
            <option value="time" disabled={!timeOk}>time</option>
          </select>
        </label>
        <label className="inline-flex items-center gap-1">running mean
          <select aria-label="Running mean window" className={`rounded border px-1 py-0.5 ${ct.input}`} value={smoothing}
            onChange={(e) => setSmoothing(Number(e.target.value))}>
            {SMOOTHING.map((w) => <option key={w} value={w}>{w === 0 ? 'off' : `${w} rows`}</option>)}
          </select>
        </label>
        <button type="button" className={`rounded px-2 py-0.5 font-medium ${ct.button}`}
          disabled={columns.every((c) => effective.some((p) => p.keys.includes(c)))}
          onClick={() => {
            const used = new Set(effective.flatMap((p) => p.keys));
            const pick = columns.find((c) => !used.has(c));
            if (pick) setEffective([...effective, { id: nextId.current++, keys: [pick] }]);
          }}>+ chart</button>
      </div>
      {unitsCaption(units) && <p className={`text-[11px] ${ct.muted}`}>{unitsCaption(units)}</p>}
      {xMode === 'time' && dtUsed !== null && (
        <p className={`text-[10px] ${ct.muted}`}>time = step x {Number(dtUsed.toPrecision(6))} (timestep from the run{segments.some((s) => s.rows.some((r) => r.time !== undefined)) ? ' or the time column' : ''})</p>
      )}
      {effective.length === 0 && <p className={`text-xs ${ct.muted}`}>No numeric columns to plot.</p>}
      {effective.map((p, pi) => {
        const title = p.keys.join(', ');
        const remaining = columns.filter((c) => !p.keys.includes(c));
        const yLabel = p.keys.length === 1 ? columnLabel(p.keys[0], units) : p.keys.join(' / ');
        return (
          <PanelView key={p.id} title={title} theme={theme} ct={ct}
            series={seriesFor(p.keys)} xLabel={xLabel} yLabel={yLabel}
            remaining={remaining}
            onAdd={(k) => setEffective(effective.map((q, i) => (i === pi ? { ...q, keys: [...q.keys, k] } : q)))}
            onRemove={() => setEffective(effective.filter((_, i) => i !== pi))}
            csv={() => csvFor(p.keys)} />
        );
      })}
      {columns.length > 0 && <ColumnStatistics segments={active} columns={columns} theme={theme} />}
    </div>
  );
};

/** Block-average error, autocorrelation and histogram of one thermo column over the runs shown. */
const ColumnStatistics: React.FC<{ segments: ThermoSegment[]; columns: string[]; theme: Theme }> = ({ segments, columns, theme }) => {
  const ct = getThemeTokens(theme);
  const [choice, setChoice] = useState('');
  const key = columns.includes(choice) ? choice : columns[0];
  const values = useMemo(() => {
    const out: number[] = [];
    for (const s of segments) for (const r of s.rows) if (Number.isFinite(r[key])) out.push(r[key]);
    return out;
  }, [segments, key]);
  const blocks = useMemo(() => blockAverage(values, 4), [values]);
  const maxLag = Math.min(200, Math.floor(values.length / 2));
  const acf = useMemo(() => autocorrelation(values, maxLag), [values, maxLag]);
  const hist = useMemo(() => histogram(values, 30), [values]);
  const blockRef = useRef<SVGSVGElement>(null);
  const acfRef = useRef<SVGSVGElement>(null);
  const histRef = useRef<HTMLDivElement>(null);
  const histBins = hist.centers.map((c, b) => ({
    x0: c - hist.width / 2, x1: c + hist.width / 2, count: hist.counts[b], density: hist.density[b],
  }));
  const inputCls = `rounded border px-1 py-0.5 text-[11px] ${ct.input}`;
  const line = (name: string, points: { x: number; y: number }[], xLabel: string, yLabel: string, ref: React.RefObject<SVGSVGElement | null>, empty: string) => (
    <MultiLineChart series={[{ name, points }]} xLabel={xLabel} yLabel={yLabel} theme={theme} height={110} svgRef={ref} emptyText={empty} />
  );
  const charts: { name: string; title: string; ref: React.RefObject<Element | null>; csv: () => string; chart: React.ReactNode }[] = [
    {
      name: 'block_se', title: 'Block averages: standard error vs block size', ref: blockRef,
      csv: () => toCsv(['block_size', 'mean', 'std_error'], blocks.map((b) => [b.blockSize, b.mean, b.stdError])),
      chart: line('std. error', blocks.map((b) => ({ x: b.blockSize, y: b.stdError })), 'block size (rows)', 'std. error', blockRef, 'Not enough rows for four blocks'),
    },
    {
      name: 'autocorrelation', title: 'Autocorrelation vs lag', ref: acfRef,
      csv: () => toCsv(['lag', 'acf'], acf.map((v, k) => [k, v])),
      chart: line('C(lag) / C(0)', acf.map((v, k) => ({ x: k, y: v })), 'lag (rows)', 'C / C0', acfRef, 'Not enough rows for an autocorrelation'),
    },
    {
      name: 'histogram', title: 'Histogram (30 bins)', ref: histRef,
      csv: () => toCsv(['x0', 'x1', 'count', 'density'], histBins.map((b) => [b.x0, b.x1, b.count, b.density])),
      chart: <div ref={histRef}><Histogram bins={histBins} xLabel={key} yLabel="count" theme={theme} height={110} /></div>,
    },
  ];
  return (
    <section aria-label="Column statistics" className={`space-y-2 rounded border p-2 ${ct.divider}`}>
      <div className={`flex flex-wrap items-center gap-2 text-[11px] ${ct.muted}`}>
        <p className={`text-xs font-semibold ${ct.headerText}`}>Column statistics</p>
        <label className="inline-flex items-center gap-1">column
          <select aria-label="Thermo column statistics" className={inputCls} value={key} onChange={(e) => setChoice(e.target.value)}>
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <span>{values.length} values</span>
      </div>
      {values.length < 2 ? (
        <p className={`text-xs ${ct.muted}`}>Not enough data: this column needs at least two thermo rows in the runs shown.</p>
      ) : charts.map((c) => (
        <div key={c.name} className="space-y-1">
          <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
            <p className={`text-[11px] font-semibold ${ct.headerText}`}>{c.title}</p>
            <ChartActions name={`${c.name}_${key}`} theme={theme} svgRef={c.ref} csv={c.csv} />
          </div>
          {c.chart}
        </div>
      ))}
    </section>
  );
};

interface PanelViewProps {
  title: string;
  theme: Theme;
  ct: ReturnType<typeof getThemeTokens>;
  series: ChartSeries[];
  xLabel: string;
  yLabel: string;
  remaining: string[];
  onAdd: (key: string) => void;
  onRemove: () => void;
  csv: () => string;
}

const PanelView: React.FC<PanelViewProps> = ({ title, theme, ct, series, xLabel, yLabel, remaining, onAdd, onRemove, csv }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  return (
    <section className={`rounded border p-2 ${ct.divider}`} aria-label={`Thermo chart ${title}`}>
      <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
        <p className={`text-[11px] font-semibold ${ct.headerText}`}>{title}</p>
        <div className="flex items-center gap-1">
          <ChartActions name={`thermo_${title}`} theme={theme} svgRef={svgRef} csv={csv} />
          {remaining.length > 0 && (
            <select aria-label={`Add a column to ${title}`} className={`rounded border px-1 py-0.5 text-[10px] ${ct.input}`} value=""
              onChange={(e) => { if (e.target.value) onAdd(e.target.value); }}>
              <option value="">+ column</option>
              {remaining.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          )}
          <button type="button" aria-label={`Remove chart ${title}`} className={`rounded px-1.5 text-xs ${ct.hoverSurface}`} onClick={onRemove}>×</button>
        </div>
      </div>
      <MultiLineChart series={series} xLabel={xLabel} yLabel={yLabel} theme={theme} height={120} svgRef={svgRef}
        emptyText="No finite values in this selection" />
    </section>
  );
};
