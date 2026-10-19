import React, { useEffect, useMemo, useRef, useState } from 'react';
import { getThemeTokens, isDarkTheme, Theme } from '../../../../theme';
import { thermoUnit } from '../../../../lammps/thermoUnits';
import { analyzeNeighbours, analyzeTrajectory, type AnalysisResult } from '../../../../services/analysisClient';
import {
  computeStructureFactor, defaultCutoff, type NeighbourResult, type RDFPoint,
} from '../../../../services/trajectoryAnalysis';
import { Histogram, MultiLineChart, type ChartSeries } from '../../../charts/SimpleChart';
import { fitDiffusion, timeUnitName, toCsv, type XY } from '../chartData';
import { kineticTemperature, maxwellBoltzmannDensity, recordSpeeds, speedHistogram, type TrajectoryRecord } from '../trajectory';
import { boltzmannConstant } from '../chartData';
import { boxAndCountSeries, emptyPrepared, prepareTrajectory, type PreparedTrajectory } from './trajectoryData';
import { ChartActions, saveBlob } from './ChartActions';

/** Work caps: frames and atoms beyond these are sampled uniformly, and the panel says so. */
const MAX_FRAMES = 120;
const MAX_ATOMS = 1500;
/** Bins of the speed histogram. */
const SPEED_BINS = 30;
/** Bins of the bond-angle histogram over [0, 180] degrees. */
const ANGLE_BINS = 90;
/** Wave-vector points of S(q). */
const SQ_POINTS = 200;
const NO_RDF: RDFPoint[] = [];

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
  /** The neighbour cutoff typed by the person; null follows the first minimum of g(r). */
  const [cutoffOverride, setCutoffOverride] = useState<number | null>(null);
  /** The cutoff field while it is being edited (null shows the cutoff in use). */
  const [cutoffText, setCutoffText] = useState<string | null>(null);
  const [neigh, setNeigh] = useState<{ source: PreparedTrajectory; cutoff: number; result: NeighbourResult } | null>(null);
  const msdRef = useRef<SVGSVGElement>(null);
  const rdfRef = useRef<SVGSVGElement>(null);
  const volRef = useRef<HTMLDivElement>(null);
  const speedRef = useRef<SVGSVGElement>(null);
  const coordRef = useRef<HTMLDivElement>(null);
  const angleRef = useRef<SVGSVGElement>(null);
  const sqRef = useRef<SVGSVGElement>(null);

  /**
   * Speed distribution of the last recorded frame, from its velocities (null when the frame has none). The
   * Maxwell-Boltzmann curve is drawn at the temperature of that frame, T = sum(m v^2) / (3 N k_B), with the
   * mean mass for the curve (exact when all atoms share one mass).
   */
  const speedInfo = useMemo(() => {
    const last = records[records.length - 1];
    const sp = last ? recordSpeeds(last) : null;
    if (!sp || sp.speed.length === 0) return null;
    const n = sp.speed.length;
    let mSum = 0;
    let mMin = Infinity;
    let mMax = -Infinity;
    for (let i = 0; i < n; i++) {
      mSum += sp.mass[i];
      if (sp.mass[i] < mMin) mMin = sp.mass[i];
      if (sp.mass[i] > mMax) mMax = sp.mass[i];
    }
    const kB = boltzmannConstant(units) ?? null;
    const T = kB !== null ? kineticTemperature(sp.speed, sp.mass, kB) : null;
    return {
      step: last.step,
      atoms: n,
      bins: speedHistogram(sp.speed, SPEED_BINS),
      kB,
      T,
      mean: mSum / n,
      meanSpeed: sp.speed.reduce((a, b) => a + b, 0) / n,
      massesDiffer: mMax > mMin,
      massesFromFile: !last.mass,
    };
  }, [records, units]);
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

  // Neighbour analyses: the cutoff defaults to the first g(r) minimum after the first peak.
  const rdfNow = current ? current.result.rdf : NO_RDF;
  const cutoffAuto = useMemo(() => defaultCutoff(rdfNow), [rdfNow]);
  // capped at the g(r) range (at most 0.95 of half the smallest box side), so a typo cannot ask for all pairs
  const cutoffCap = rdfNow.length ? rdfNow[rdfNow.length - 1].r : Infinity;
  const cutoff = Math.min(cutoffOverride ?? cutoffAuto, cutoffCap);
  useEffect(() => {
    if (!prepared || prepared.frames.length === 0 || !(cutoff > 0)) return;
    let cancelled = false;
    analyzeNeighbours(prepared.frames[prepared.frames.length - 1], cutoff, ANGLE_BINS)
      .then((result) => { if (!cancelled) setNeigh({ source: prepared, cutoff, result }); })
      .catch(() => { if (!cancelled) setNeigh(null); });
    return () => { cancelled = true; };
  }, [prepared, cutoff]);
  const neighNow = neigh && prepared && neigh.source === prepared && neigh.cutoff === cutoff ? neigh.result : null;
  const coordBins = neighNow
    ? neighNow.coordination.atomsWith.map((count, k) => ({
      x0: k - 0.5, x1: k + 0.5, count, density: count / Math.max(1, neighNow.coordination.perAtom.length),
    }))
    : [];

  // S(q) from the g(r) above and the number density of the last sampled frame (3D boxes), with the Lorch window.
  // q runs to five times 2 pi / r_peak (r_peak the g(r) maximum), at most the Nyquist value pi / dr of the g(r) grid.
  const sq = useMemo(() => {
    const last = prepared?.frames[prepared.frames.length - 1];
    const box = last?.box;
    if (!last || !box || dim !== 3 || rdfNow.length < 2) return [];
    const V = (box.xhi - box.xlo) * (box.yhi - box.ylo) * (box.zhi - box.zlo);
    if (!(V > 0)) return [];
    const dr = rdfNow[1].r - rdfNow[0].r;
    let peak = rdfNow[0];
    for (const p of rdfNow) if (p.g > peak.g) peak = p;
    const qMax = Math.min(Math.PI / dr, peak.r > 0 ? 10 * Math.PI / peak.r : Infinity);
    return computeStructureFactor(
      { r: rdfNow.map((p) => p.r), g: rdfNow.map((p) => p.g) }, last.atoms.length / V, qMax, SQ_POINTS, true,
    );
  }, [prepared, rdfNow, dim]);

  /** Every computed series as one JSON object (the "Export all analysis data" button). */
  const exportAll = () => {
    const dens = current?.result.density ?? null;
    const payload = {
      msd: msdPts.map((p) => ({ time: p.x, msd: p.y })),
      diffusion: fit ? { D: fit.D, slope: fit.slope, intercept: fit.intercept, r2: fit.r2, dim: fit.dim, from: fit.tFrom, to: fit.tTo } : null,
      rdf: rdfNow.map((p) => ({ r: p.r, g: p.g, count: p.count })),
      densityProfile: dens ? { axis: dens.axis, bins: dens.bins } : null,
      speedHistogram: speedInfo?.bins ?? null,
      neighbourCutoff: neighNow ? neighNow.coordination.cutoff : null,
      coordination: neighNow ? { mean: neighNow.coordination.mean, atomsWith: neighNow.coordination.atomsWith } : null,
      bondAngles: neighNow ? neighNow.angles : null,
      structureFactor: sq,
    };
    saveBlob('trajectory_analysis.json', new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
  };

  if (!open) {
    return (
      <div>
        <button type="button" className={`rounded px-2 py-1 text-xs font-medium ${ct.button}`} onClick={() => setOpen(true)}>
          Trajectory analysis
        </button>
        <p className={`mt-1 text-[10px] ${ct.muted}`}>MSD and diffusion, g(r), coordination, bond angles, S(q), density profile, box volume, from the recorded frames.</p>
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
  const inputCls = `w-14 rounded border px-1 py-0.5 text-[11px] ${ct.input}`;

  return (
    <div className="space-y-3 text-xs">
      <div className="flex items-center justify-between gap-2">
        <p className={`font-semibold ${ct.headerText}`}>Trajectory analysis</p>
        <div className="flex items-center gap-1">
          <button type="button" className={`rounded px-2 py-0.5 text-[11px] ${ct.button}`} onClick={exportAll}
            disabled={!prepared || prepared.recorded < 2}>Export all analysis data (JSON)</button>
          <button type="button" className={`rounded px-2 py-0.5 text-[11px] ${ct.hoverSurface}`} onClick={() => setOpen(false)}>Hide</button>
        </div>
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

      <section aria-label="Coordination number" className={`rounded border p-2 ${ct.divider}`}>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
          <p className={`font-semibold ${ct.headerText}`}>Coordination number</p>
          <ChartActions name="coordination" theme={theme} svgRef={coordRef}
            csv={() => toCsv(['coordination', 'atoms'], coordBins.map((b) => [b.x0 + 0.5, b.count]))} />
        </div>
        <div className={`mb-1 flex flex-wrap items-center gap-2 ${ct.muted}`}>
          neighbour cutoff
          <input aria-label="Neighbour cutoff" type="number" min={0.1} step={0.05} className={inputCls}
            value={cutoffText ?? String(Number(cutoff.toPrecision(4)))}
            onChange={(e) => {
              setCutoffText(e.target.value);
              const v = Number(e.target.value);
              if (e.target.value.trim() !== '' && Number.isFinite(v) && v > 0) setCutoffOverride(v);
            }}
            onBlur={() => setCutoffText(null)} />
          {cutoffOverride === null ? '(first minimum of g(r))' : <button type="button" className={`rounded px-1 ${ct.hoverSurface}`} onClick={() => { setCutoffOverride(null); setCutoffText(null); }}>use g(r)</button>}
          {cutoffOverride !== null && cutoffOverride > cutoffCap && <span>(capped at the g(r) range)</span>}
        </div>
        <div ref={coordRef}>
          {coordBins.length > 0
            ? <Histogram bins={coordBins} xLabel="neighbours within the cutoff" yLabel="atoms" theme={theme} height={120} />
            : <p className={ct.muted}>{prepared && prepared.recorded >= 2 ? 'Computing...' : 'No recorded frames'}</p>}
        </div>
        <p className={`mt-1 text-[10px] ${ct.muted}`}>
          {neighNow ? `Mean coordination ${neighNow.coordination.mean.toFixed(2)} within ${Number(cutoff.toPrecision(4))}, last sampled frame.` : ''}
        </p>
      </section>

      <section aria-label="Bond-angle distribution" className={`rounded border p-2 ${ct.divider}`}>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
          <p className={`font-semibold ${ct.headerText}`}>Bond-angle distribution</p>
          <ChartActions name="bond_angles" theme={theme} svgRef={angleRef}
            csv={() => toCsv(['angle_low_deg', 'angle_high_deg', 'count', 'density'],
              (neighNow?.angles ?? []).map((b) => [b.x0, b.x1, b.count, b.density]))} />
        </div>
        <MultiLineChart series={[{ name: 'density per degree', points: (neighNow?.angles ?? []).map((b) => ({ x: (b.x0 + b.x1) / 2, y: b.density })) }]}
          xLabel="angle i-j-k [deg]" yLabel="density" theme={theme} height={120} svgRef={angleRef}
          emptyText={neighNow ? 'Fewer than two neighbours per atom' : 'Computing...'} />
        <p className={`mt-1 text-[10px] ${ct.muted}`}>Angles between pairs of neighbours of each atom within the cutoff above (last sampled frame).</p>
      </section>

      <section aria-label="Structure factor S(q)" className={`rounded border p-2 ${ct.divider}`}>
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
          <p className={`font-semibold ${ct.headerText}`}>Structure factor S(q)</p>
          <ChartActions name="structure_factor" theme={theme} svgRef={sqRef}
            csv={() => toCsv(['q', 'S'], sq.map((p) => [p.q, p.s]))} />
        </div>
        <MultiLineChart series={[{ name: 'S(q)', points: sq.map((p) => ({ x: p.q, y: p.s })) }]}
          xLabel="q [1/distance]" yLabel="S(q)" theme={theme} height={120} svgRef={sqRef}
          emptyText={current ? 'Needs a 3D box and g(r)' : 'Computing...'} />
        <p className={`mt-1 text-[10px] ${ct.muted}`}>S(q) = 1 + 4 pi rho sum r^2 (g - 1) sin(qr)/(qr) M(r) dr from g(r), rho = N/V of the last sampled frame; M is the Lorch window for the cut at the g(r) range.</p>
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
        <div className="mb-1 flex flex-wrap items-center justify-between gap-1">
          <p className={`font-semibold ${ct.headerText}`}>Speed distribution</p>
          {speedInfo?.bins && (
            <ChartActions name="speed_distribution" theme={theme} svgRef={speedRef}
              csv={() => toCsv(['speed_low', 'speed_high', 'count', 'density', 'maxwell_boltzmann'],
                speedInfo.bins!.map((b) => [b.x0, b.x1, b.count, b.density,
                  speedInfo.T !== null ? maxwellBoltzmannDensity((b.x0 + b.x1) / 2, speedInfo.mean, speedInfo.T, speedInfo.kB ?? 1) : '']))} />
          )}
        </div>
        {!speedInfo && (
          <p className={ct.muted}>
            No velocities in the recorded frames. Recording frames from a run sends velocities (the notebook asks the engine for them);
            a dump file shows its speeds when it has vx vy vz columns.
          </p>
        )}
        {speedInfo && (
          <>
            <div data-speed-temperature={speedInfo.T ?? undefined} data-speed-mean={speedInfo.meanSpeed} data-speed-atoms={speedInfo.atoms} hidden />
            <SpeedPlot bins={speedInfo.bins ?? []} svgRef={speedRef} theme={theme}
              curve={speedInfo.T !== null && speedInfo.bins
                ? speedInfo.bins.map((b) => ({ v: (b.x0 + b.x1) / 2, f: maxwellBoltzmannDensity((b.x0 + b.x1) / 2, speedInfo.mean, speedInfo.T as number, speedInfo.kB ?? 1) }))
                : []} />
            <p className={`mt-1 text-[10px] ${ct.muted}`}>
              {speedInfo.atoms} atoms in the last recorded frame (step {speedInfo.step}).
              {speedInfo.T !== null
                ? ` Mean temperature ${speedInfo.T.toPrecision(4)} (k_B from units ${units ?? 'unset'}); curve: Maxwell-Boltzmann at that temperature.`
                : speedInfo.kB === null
                  ? ` Units ${units ?? 'unset'} have no Boltzmann constant here: the curve is not drawn.`
                  : ''}
              {speedInfo.massesFromFile ? ' Masses taken as 1 (a dump file has none).' : ''}
              {speedInfo.massesDiffer ? ' Masses differ: the curve uses the mean mass.' : ''}
            </p>
          </>
        )}
      </section>

      <button type="button" className={`rounded px-2 py-0.5 text-[11px] ${ct.hoverSurface}`} onClick={() => setVersion((v) => v + 1)}>
        Recompute
      </button>
    </div>
  );
};

/** Speed histogram (probability density bars) with the Maxwell-Boltzmann curve over it. */
const SpeedPlot: React.FC<{
  bins: { x0: number; x1: number; density: number }[];
  curve: { v: number; f: number }[];
  svgRef: React.RefObject<SVGSVGElement | null>;
  theme: Theme;
}> = ({ bins, curve, svgRef, theme }) => {
  const ct = getThemeTokens(theme);
  const isDark = isDarkTheme(theme);
  if (bins.length === 0) return <div className={`flex h-[140px] items-center justify-center text-xs ${ct.muted}`}>No data</div>;
  const W = 320, H = 140, padL = 40, padR = 12, padT = 10, padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xMax = bins[bins.length - 1].x1 || 1;
  let yMax = 0;
  for (const b of bins) yMax = Math.max(yMax, b.density);
  for (const p of curve) yMax = Math.max(yMax, p.f);
  yMax = yMax || 1;
  const X = (v: number) => padL + (v / xMax) * plotW;
  const Y = (f: number) => padT + plotH - (f / yMax) * plotH;
  const gridColor = isDark ? '#332a1f' : '#e0d7c6';
  const textColor = isDark ? '#a3937f' : '#6b6053';
  const barColor = isDark ? '#7aa6d8' : '#4d7fb3';
  const curveColor = isDark ? '#e07a5f' : '#b2503a';
  return (
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Speed distribution histogram with Maxwell-Boltzmann curve"
      className="block max-w-full" style={{ height: H }}>
      <line x1={padL} x2={W - padR} y1={padT + plotH} y2={padT + plotH} stroke={gridColor} />
      <line x1={padL} x2={padL} y1={padT} y2={padT + plotH} stroke={gridColor} />
      {bins.map((b, i) => (
        <rect key={i} x={X(b.x0)} y={Y(b.density)} width={Math.max(0, X(b.x1) - X(b.x0) - 0.5)} height={Math.max(0, padT + plotH - Y(b.density))} fill={barColor} opacity={0.75} />
      ))}
      {curve.length > 1 && (
        <polyline fill="none" stroke={curveColor} strokeWidth={1.5}
          points={curve.map((p) => `${X(p.v).toFixed(2)},${Y(p.f).toFixed(2)}`).join(' ')} />
      )}
      <text x={padL} y={H - 6} fontSize={10} fill={textColor}>0</text>
      <text x={W - padR} y={H - 6} fontSize={10} fill={textColor} textAnchor="end">{xMax.toPrecision(3)}</text>
      <text x={W / 2} y={H - 6} fontSize={10} fill={textColor} textAnchor="middle">speed</text>
      <text x={padL - 4} y={padT + 8} fontSize={10} fill={textColor} textAnchor="end">{yMax.toPrecision(2)}</text>
    </svg>
  );
};
