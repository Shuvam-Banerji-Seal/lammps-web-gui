import React, { useMemo, useState } from 'react';
import { getThemeTokens, Theme } from '../../../../theme';
import { thermoUnit, unitsCaption } from '../../../../lammps/thermoUnits';
import {
  boltzmannConstant, compressibility, energyDrift, heatCapacity, linearFit, numericColumns, segmentDt,
  summarize, timeUnitName, type Ensemble, type ThermoSegment, type XY,
} from '../chartData';

interface ThermoStatsProps {
  segments: ThermoSegment[];
  theme: Theme;
  /** Atoms in the system (from the recorded frames), used when the thermo rows carry no `atoms` column. */
  atomCount: number | null;
}

const fmt = (v: number): string => (Number.isFinite(v) ? Number(v.toPrecision(5)).toString() : '-');
const fmtSci = (v: number): string => (Number.isFinite(v) ? v.toPrecision(4) : '-');

/**
 * Statistics of the thermo columns over a step range of a run, and the quantities that follow from fluctuations:
 * heat capacity (NVT), isothermal compressibility and bulk modulus (NPT), energy drift per atom per time (NVE).
 * The ensemble is read from the fix commands of the cell that produced the rows; the formulas are shown with
 * each value.
 */
export const ThermoStats: React.FC<ThermoStatsProps> = ({ segments, theme, atomCount }) => {
  const ct = getThemeTokens(theme);
  const [segChoice, setSegChoice] = useState<'all' | number>('all');
  const [fromStep, setFromStep] = useState('');
  const [toStep, setToStep] = useState('');

  const active = useMemo(
    () => (segChoice === 'all' ? segments : segments[segChoice] ? [segments[segChoice]] : []),
    [segments, segChoice],
  );
  const bounds = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (const s of active) for (const r of s.rows) {
      if (Number.isFinite(r.step)) { if (r.step < lo) lo = r.step; if (r.step > hi) hi = r.step; }
    }
    return Number.isFinite(lo) ? { lo, hi } : null;
  }, [active]);
  const from = fromStep.trim() === '' ? -Infinity : Number(fromStep);
  const to = toStep.trim() === '' ? Infinity : Number(toStep);
  const rangeOk = !Number.isNaN(from) && !Number.isNaN(to) && from <= to;

  const rows = useMemo(() => {
    const out: Record<string, number>[] = [];
    if (!rangeOk) return out;
    for (const s of active) for (const r of s.rows) if (r.step >= from && r.step <= to) out.push(r);
    return out;
  }, [active, from, to, rangeOk]);

  if (segments.length === 0) {
    return <p className={`text-xs ${ct.muted}`}>Statistics appear here once a run prints thermo output.</p>;
  }

  const units = active[0]?.units ?? segments[0].units;
  const dt = active.length ? segmentDt(active[0]) : null;
  const ensembles = new Set<Ensemble>(active.map((s) => s.ensemble));
  const ensemble: Ensemble | 'mixed' = ensembles.size === 1 ? [...ensembles][0] : ensembles.size === 0 ? 'unknown' : 'mixed';
  const keys = numericColumns(active);
  const useTime = dt !== null;
  const unitName = useTime ? timeUnitName(units) : 'step';
  const atomsN = (() => {
    const last = rows[rows.length - 1];
    if (last && Number.isFinite(last.atoms) && last.atoms > 0) return last.atoms;
    return atomCount;
  })();

  const seriesOf = (key: string): XY[] => {
    const pts: XY[] = [];
    for (const r of rows) {
      const y = r[key];
      if (!Number.isFinite(y) || !Number.isFinite(r.step)) continue;
      pts.push({ x: useTime ? r.step * (dt as number) : r.step, y });
    }
    return pts;
  };
  const colValues = (key: string): number[] => rows.map((r) => r[key]);

  const stats = keys.map((k) => {
    const sm = summarize(colValues(k));
    const fit = linearFit(seriesOf(k));
    return { k, sm, slope: fit ? fit.slope : NaN };
  });

  const kB = boltzmannConstant(units);
  const hasCol = (k: string) => keys.includes(k);
  let cv: React.ReactNode;
  if (ensemble !== 'NVT') {
    cv = <p className={ct.muted}>Not computed: the fix commands of this run make it {ensemble === 'mixed' ? 'a mix of ensembles' : ensemble}. The energy-fluctuation formula assumes a canonical (NVT) run.</p>;
  } else if (!hasCol('etotal') || !hasCol('temp')) {
    cv = <p className={ct.muted}>Needs the etotal and temp columns in thermo_style.</p>;
  } else {
    const res = heatCapacity(colValues('etotal'), colValues('temp'), atomsN ?? 0, units);
    cv = 'error' in res ? <p className={ct.muted}>{res.error}</p> : (
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        <dt className={ct.muted}>C_V</dt>
        <dd>{fmt(res.total)} k_B (total){res.perAtom !== null && <> ; {fmt(res.perAtom)} k_B per atom</>}</dd>
        <dt className={ct.muted}>k_B</dt><dd>{fmtSci(res.kB)} ({units ?? 'units unset'})</dd>
        <dt className={ct.muted}>mean T</dt><dd>{fmt(res.meanT)}</dd>
      </dl>
    );
  }

  let comp: React.ReactNode;
  if (ensemble !== 'NPT') {
    comp = <p className={ct.muted}>Not computed: the run is {ensemble === 'mixed' ? 'a mix of ensembles' : ensemble}, not NPT. The volume-fluctuation formula assumes constant-pressure sampling.</p>;
  } else if (!hasCol('vol') || !hasCol('temp')) {
    comp = <p className={ct.muted}>Needs the vol and temp columns in thermo_style.</p>;
  } else {
    const res = compressibility(colValues('vol'), colValues('temp'), units);
    comp = 'error' in res ? <p className={ct.muted}>{res.error}</p> : (
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        <dt className={ct.muted}>kappa_T</dt><dd>{fmtSci(res.kappa)} (1/pressure, {units ?? 'units unset'})</dd>
        <dt className={ct.muted}>K_T = 1/kappa</dt><dd>{fmtSci(res.bulk)} (pressure, {units ?? 'units unset'})</dd>
        <dt className={ct.muted}>mean V, T</dt><dd>{fmt(res.meanV)}, {fmt(res.meanT)}</dd>
      </dl>
    );
  }

  let drift: React.ReactNode;
  if (ensemble !== 'NVE') {
    drift = <p className={ct.muted}>Not computed: the run is {ensemble === 'mixed' ? 'a mix of ensembles' : ensemble}. Energy drift is a conservation check for NVE runs.</p>;
  } else if (!hasCol('etotal')) {
    drift = <p className={ct.muted}>Needs the etotal column in thermo_style.</p>;
  } else if (!(atomsN && atomsN > 0)) {
    drift = <p className={ct.muted}>Needs the atom count (a recorded frame or an atoms column).</p>;
  } else {
    const res = energyDrift(seriesOf('etotal'), atomsN);
    drift = !res ? <p className={ct.muted}>Needs at least two rows in the range.</p> : (
      <p>
        {fmtSci(res.perAtomPerTime)} energy per atom per {useTime ? unitName : 'step'} (slope of etotal, R² {fmt(res.r2)})
      </p>
    );
  }

  const inputCls = `rounded border px-1 py-0.5 text-xs ${ct.input}`;
  return (
    <div className="space-y-3 text-xs">
      <p className={`text-xs font-semibold ${ct.headerText}`}>Statistics</p>
      <div className={`flex flex-wrap items-center gap-2 ${ct.muted}`}>
        <label className="inline-flex items-center gap-1">Run
          <select aria-label="Statistics run" className={inputCls} value={segChoice}
            onChange={(e) => setSegChoice(e.target.value === 'all' ? 'all' : Number(e.target.value))}>
            <option value="all">all runs</option>
            {segments.map((s, i) => <option key={i} value={i}>{s.label}</option>)}
          </select>
        </label>
        <label className="inline-flex items-center gap-1">from step
          <input aria-label="From step" type="number" className={`w-20 ${inputCls}`} value={fromStep}
            placeholder={bounds ? String(bounds.lo) : ''} onChange={(e) => setFromStep(e.target.value)} />
        </label>
        <label className="inline-flex items-center gap-1">to step
          <input aria-label="To step" type="number" className={`w-20 ${inputCls}`} value={toStep}
            placeholder={bounds ? String(bounds.hi) : ''} onChange={(e) => setToStep(e.target.value)} />
        </label>
        <span>{rows.length} rows</span>
      </div>
      {!rangeOk && <p role="alert" className={ct.muted}>The step range is empty or reversed.</p>}
      {unitsCaption(units) && <p className={ct.muted}>{unitsCaption(units)}</p>}
      <p className={ct.muted}>Ensemble from the fix commands: {ensemble}. Drift is the least-squares slope against {useTime ? `time (${unitName})` : 'step'}. std is the sample standard deviation (n - 1).</p>
      {keys.length > 0 && rows.length > 0 ? (
        <div className="max-h-64 overflow-auto">
          <table className="w-full border-collapse text-right">
            <thead>
              <tr className={ct.headerText}>
                <th className="px-1 text-left font-semibold">column</th>
                <th className="px-1 font-semibold">n</th>
                <th className="px-1 font-semibold">mean</th>
                <th className="px-1 font-semibold">std</th>
                <th className="px-1 font-semibold">min</th>
                <th className="px-1 font-semibold">max</th>
                <th className="px-1 font-semibold">drift</th>
              </tr>
            </thead>
            <tbody>
              {stats.map(({ k, sm, slope }) => (
                <tr key={k} className="border-t border-current/10">
                  <td className="px-1 text-left" title={thermoUnit(units, k) ?? undefined}>{k}{thermoUnit(units, k) ? <span className={ct.muted}> [{thermoUnit(units, k)}]</span> : null}</td>
                  <td className="px-1">{sm.n}</td>
                  <td className="px-1">{fmt(sm.mean)}</td>
                  <td className="px-1">{fmt(sm.std)}</td>
                  <td className="px-1">{fmt(sm.min)}</td>
                  <td className="px-1">{fmt(sm.max)}</td>
                  <td className="px-1">{fmtSci(slope)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className={ct.muted}>No rows in this range.</p>
      )}
      <div className="space-y-2">
        <section aria-label="Heat capacity">
          <p className={`font-semibold ${ct.headerText}`}>Heat capacity from energy fluctuations (NVT only)</p>
          <p className={ct.muted}>C_V = (&lt;E^2&gt; - &lt;E&gt;^2) / (k_B T^2), E = etotal, T = mean temp of the rows, population averages.</p>
          {cv}
        </section>
        <section aria-label="Compressibility">
          <p className={`font-semibold ${ct.headerText}`}>Compressibility from volume fluctuations (NPT only)</p>
          <p className={ct.muted}>kappa_T = (&lt;V^2&gt; - &lt;V&gt;^2) / (k_B T &lt;V&gt;), bulk modulus K_T = 1 / kappa_T.</p>
          {comp}
        </section>
        <section aria-label="Energy drift">
          <p className={`font-semibold ${ct.headerText}`}>Energy conservation drift (NVE only)</p>
          <p className={ct.muted}>Slope of etotal against time, divided by the number of atoms: a check that an NVE run conserves energy.</p>
          {drift}
        </section>
      </div>
      {kB === null && <p className={ct.muted}>No Boltzmann constant for units {units ?? 'unset'}: the fluctuation quantities are not computed.</p>}
    </div>
  );
};
