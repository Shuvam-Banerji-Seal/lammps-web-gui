import React, { useEffect, useMemo, useRef, useState } from 'react';
import { getThemeTokens, Theme } from '../../../../theme';
import { thermoUnit } from '../../../../lammps/thermoUnits';
import { analyzeTrajectory, type AnalysisResult } from '../../../../services/analysisClient';
import type { RDFPoint } from '../../../../services/trajectoryAnalysis';
import { Histogram, MultiLineChart, type ChartSeries } from '../../../charts/SimpleChart';
import { fitDiffusion, timeUnitName, toCsv, type XY } from '../chartData';
import type { TrajectoryRecord } from '../trajectory';
import { boxAndCountSeries, emptyPrepared, prepareTrajectory, type PreparedTrajectory } from './trajectoryData';
import { ChartActions } from './ChartActions';

/** Work caps: frames and atoms beyond these are sampled uniformly, and the panel says so. */
const MAX_FRAMES = 120;
const MAX_ATOMS = 1500;

interface TrajectoryAnalysisProps {
  records: TrajectoryRecord[];
  /** Timestep in the units style's time unit (null: the x axes are in steps). */
  dt: number | null;
  units?: string;
  theme: Theme;
  /** The buffer's recording stride after thinning (1 = every recorded frame). */
  stride: number;
}

interface Output { source: PreparedTrajectory; rdfMode: 'all' | 'last'; result: AnalysisResult }

/**
 * Trajectory analyses of the frames the notebook recorded (the run's, or a dump file loaded for rendering):
 * MSD with the diffusion coefficient from a fit of its linear part, the radial distribution function g(r),
 * the density profile along an axis, and box volume / atom count against time. Computed only while open.
 * Heavy work (RDF, MSD, density) runs in the analysis worker through services/analysisClient.
 */
export const TrajectoryAnalysis: React.FC<TrajectoryAnalysisProps> = ({ records, dt, units, theme, stride }) => {
  const ct = getThemeTokens(theme);
  const [open, setOpen] = useState(false);
  const [fitFrom, setFitFrom] = useState(30);
  const [fitTo, setFitTo] = useState(100);
  const [rdfMode, setRdfMode] = useState<'all' | 'last'>('all');
  const [densityAxis, setDensityAxis] = useState<'x' | 'y' | 'z'>('y');
  const [output, setOutput] = useState<Output | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const msdRef = useRef<SVGSVGElement>(null);
  const rdfRef = useRef<SVGSVGElement>(null);
  const volRef = useRef<HTMLDivElement>(null);
  const densRef = useRef<HTMLDivElement>(null);

  const prepared = useMemo(
    () => (open && records.length > 0 ? prepareTrajectory(records, { maxFrames: MAX_FRAMES, maxAtoms: MAX_ATOMS }) : null),
    [open, records],
  );

  useEffect(() => {
    if (!prepared || prepared.frames.length < 2) return;
    let cancelled = false;
    const frames = prepared.frames;
    const rdfFrames = rdfMode === 'last' ? frames.slice(-1) : frames;
    const opts = {
      rdfRMax: 12, rdfBins: 100, msdStride: Math.max(1, Math.floor(frames.length / 20)),
      densityAxis, densityBins: 30,
    };
    analyzeTrajectory(rdfFrames, frames[0].box, opts, { frames, stride: prepared.frameStride })
      .then((result) => {
        if (cancelled) return;
        setError(null);
        setOutput({ source: prepared, rdfMode, result });
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Analysis failed');
      });
    return () => { cancelled = true; };
  }, [prepared, rdfMode, densityAxis, version]);

  const src = prepared ?? emptyPrepared();
  const current = output && prepared && output.source === prepared && output.rdfMode === rdfMode ? output : null;
  const dim: 2 | 3 = prepared && prepared.frames[0] && prepared.frames[0].box && (prepared.frames[0].box.zhi - prepared.frames[0].box.zlo) < 2 ? 2 : 3;
  const perRecordSteps = src.stepsPerRecord ?? 1;
  const timeUnit = timeUnitName(units);
  const useTime = dt !== null;
  const xScale = (useTime ? (dt as number) : 1) * perRecordSteps;
  const xLabel = useTime ? (units === 'lj' || !units ? 'time (lj)' : `time [${timeUnit}]`) : 'steps';
  const distUnit = thermoUnit(units, 'lx');
  const rLabel = distUnit ? `r [${distUnit}]` : 'r';

  const msdPts: XY[] = useMemo(
    () => (current ? current.result.msd.map((p) => ({ x: p.t * xScale, y: p.msd })) : []),
    [current, xScale],
  );
  const fit = useMemo(
    () => fitDiffusion(msdPts, fitFrom / 100, fitTo / 100, dim),
    [msdPts, fitFrom, fitTo, dim],
  );
  const volSeries = useMemo(() => {
    if (!prepared) return [];
    return boxAndCountSeries(records).map((p) => ({ x: useTime ? p.step * (dt as number) : p.step, volume: p.volume, atoms: p.atoms }));
  }, [prepared, records, dt, useTime]);

  if (!open) {
    return (
      <div>
        <button type="button" className={`rounded px-2 py-1 text-xs font-medium ${ct.button}`} onClick={() => setOpen(true)}>
          Trajectory analysis
        </button>
        <p className={`mt-1 text-[10px] ${ct.muted}`}>MSD and diffusion, g(r), density profile, box volume, from the recorded frames.</p>
      </div>
    );
  }

  const msdSeries: ChartSeries[] = [{ name: 'MSD', points: msdPts }];
  if (fit) {
    msdSeries.push({
      name: `fit, D = ${Number(fit.D.toPrecision(4))}`,
      points: [{ x: fit.tFrom, y: fit.slope * fit.tFrom + fit.intercept }, { x: fit.tTo, y: fit.slope * fit.tTo + fit.intercept }],
      color: '#b5652b',
    });
  }
  const rdfPts: RDFPoint[] = current ? current.result.rdf : [];
  const rdfSeries: ChartSeries[] = [{ name: 'g(r)', points: rdfPts.map((p) => ({ x: p.r, y: p.g })) }];
  const dens = current?.result.density;
  const volumeSeries: ChartSeries[] = [{ name: 'box volume', points: volSeries.map((p) => ({ x: p.x, y: p.volume })) }];
  const atomsNow = volSeries.length ? volSeries[volSeries.length - 1].atoms : 0;
  const pct = (n: number) => Number(n.toPrecision(3)).toString();
  const inputCls = `w-14 rounded border px-1 py-0.5 text-[11px] ${ct.input}`;

  return (
    <div className="space-y-3 text-xs">
      <div className="flex items-center justify-between gap-2">
        <p className={`font-semibold ${ct.headerText}`}>Trajectory analysis</p>
        <button type="button" className={`rounded px-2 py-0.5 text-[11px] ${ct.hoverSurface}`} onClick={() => setOpen(false)}>Hide</button>
      </div>
      {!prepared && <p className={ct.muted}>No recorded frames yet. Run a MD cell with a dump or with recording on.</p>}
      {prepared && prepared.recorded < 2 && <p className={ct.muted}>Need at least two recorded frames.</p>}
      {prepared && prepared.recorded >= 2 && (
        <p className={`text-[10px] ${ct.muted}`}>
          {prepared.recorded} recorded frames{stride > 1 ? ` (recording stride ${stride}: the buffer thinned to fit its memory cap)` : ''}
          {prepared.sampledFrames ? `; ${prepared.frames.length} sampled, every ${prepared.frameStride}th` : ''}
          {'; '}{prepared.atomsUsed} of {prepared.atomsTotal} atoms{prepared.sampledAtoms ? ' (sampled by index)' : ''}.
          {prepared.triclinic ? ' Triclinic box: image crossings tested orthogonally (approximate).' : ''}
        </p>
      )}
      {error && <p role="alert" className={ct.muted}>{error}</p>}

      <section aria-label="Mean squared displacement" className={`rounded border p-2 ${ct.divider}`}>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
          <p className={`font-semibold ${ct.headerText}`}>MSD and diffusion</p>
          <ChartActions name="msd" theme={theme} svgRef={msdRef} csv={() => toCsv(['time', 'msd'], msdPts.map((p) => [p.x, p.y]))} />
        </div>
        <MultiLineChart series={msdSeries} xLabel={xLabel} yLabel="MSD" theme={theme} height={130} svgRef={msdRef}
          emptyText={current ? 'Not enough frames for an MSD' : 'Computing...'} />
        <div className={`mt-1 flex flex-wrap items-center gap-2 ${ct.muted}`}>
          fit window
          <input aria-label="Fit from percent" type="number" min={0} max={100} className={inputCls} value={fitFrom}
            onChange={(e) => setFitFrom(Math.max(0, Math.min(100, Number(e.target.value) || 0)))} />%
          to
          <input aria-label="Fit to percent" type="number" min={0} max={100} className={inputCls} value={fitTo}
            onChange={(e) => setFitTo(Math.max(0, Math.min(100, Number(e.target.value) || 0)))} />%
          of the lag range
        </div>
        <p className={`mt-1 ${ct.muted}`}>
          {fit
            ? <>D = slope / (2 d) = {Number(fit.D.toPrecision(5))} ({`distance^2 / ${useTime ? timeUnit : 'step'}`}), slope {Number(fit.slope.toPrecision(4))}, d = {fit.dim}, R² {Number(fit.r2.toPrecision(4))}, {fit.n} points.</>
            : 'Not enough points in the fit window.'}
        </p>
        <p className={`text-[10px] ${ct.muted}`}>
          {current ? (current.result.msdUnwrapped
            ? 'Displacements are unwrapped from the recorded positions (image flags counted), so the MSD is not capped by the box.'
            : 'Displacements are minimum-image: the MSD saturates near (L/2)^2.') : ''}
        </p>
      </section>

      <section aria-label="Radial distribution function" className={`rounded border p-2 ${ct.divider}`}>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
          <p className={`font-semibold ${ct.headerText}`}>Radial distribution g(r)</p>
          <div className="flex items-center gap-1">
            <select aria-label="RDF frames" className={`rounded border px-1 py-0.5 text-[11px] ${ct.input}`} value={rdfMode}
              onChange={(e) => setRdfMode(e.target.value as 'all' | 'last')}>
              <option value="all">average over sampled frames</option>
              <option value="last">last frame</option>
            </select>
            <ChartActions name="rdf" theme={theme} svgRef={rdfRef} csv={() => toCsv(['r', 'g', 'count'], rdfPts.map((p) => [p.r, p.g, p.count]))} />
          </div>
        </div>
        <MultiLineChart series={rdfSeries} xLabel={rLabel} yLabel="g(r)" theme={theme} height={130} svgRef={rdfRef}
          emptyText={current ? 'No pairs' : 'Computing...'} />
        <p className={`mt-1 text-[10px] ${ct.muted}`}>Total g(r) over all atoms, normalised by the ideal-gas shell count; per type pair is not computed.</p>
      </section>

      <section aria-label="Density profile" className={`rounded border p-2 ${ct.divider}`}>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
          <p className={`font-semibold ${ct.headerText}`}>Density profile</p>
          <div className="flex items-center gap-1">
            <select aria-label="Density axis" className={`rounded border px-1 py-0.5 text-[11px] ${ct.input}`} value={densityAxis}
              onChange={(e) => setDensityAxis(e.target.value as 'x' | 'y' | 'z')}>
              <option value="x">along x</option>
              <option value="y">along y</option>
              <option value="z">along z</option>
            </select>
            <ChartActions name="density_profile" theme={theme} svgRef={densRef}
              csv={() => toCsv(['x0', 'x1', 'count', 'normalised'], (dens?.bins ?? []).map((b) => [b.x0, b.x1, b.count, b.density]))} />
          </div>
        </div>
        <div ref={densRef}>
          {dens && dens.bins.length > 0
            ? <Histogram bins={dens.bins} xLabel={`${densityAxis} position`} yLabel="count" theme={theme} height={120} />
            : <p className={ct.muted}>{current ? 'No atoms' : 'Computing...'}</p>}
        </div>
      </section>

      <section aria-label="Box volume" className={`rounded border p-2 ${ct.divider}`}>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
          <p className={`font-semibold ${ct.headerText}`}>Box volume and atom count</p>
          <ChartActions name="box_volume" theme={theme} svgRef={volRef}
            csv={() => toCsv(['x', 'volume', 'atoms'], volSeries.map((p) => [p.x, p.volume, p.atoms]))} />
        </div>
        <div ref={volRef}>
          <MultiLineChart series={volumeSeries} xLabel={xLabel} yLabel="volume" theme={theme} height={110} />
        </div>
        <p className={`mt-1 text-[10px] ${ct.muted}`}>{atomsNow} atoms in the last recorded frame.</p>
      </section>

      <section aria-label="Speed distribution" className={`rounded border p-2 ${ct.divider}`}>
        <p className={`font-semibold ${ct.headerText}`}>Speed distribution (not available)</p>
        <p className={ct.muted}>The recorded frames hold positions only: the engine&apos;s frame stream carries no velocities, so the Maxwell-Boltzmann comparison cannot be drawn from them.</p>
      </section>

      <button type="button" className={`rounded px-2 py-0.5 text-[11px] ${ct.hoverSurface}`} onClick={() => setVersion((v) => v + 1)}>
        Recompute
      </button>
    </div>
  );
};
