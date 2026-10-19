/**
 * The new notebook analysis sections render with their export buttons, the thermo column statistics appear, and the
 * trajectory export select lists PDB, LAMMPS data and CSV. Plain react-dom with act (the repo has no
 * @testing-library/react); jsdom has no Worker, so the analyses take the calling-thread fallback.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { TrajectoryAnalysis } from '../src/components/workbench/notebook/charts/TrajectoryAnalysis';
import { ThermoCharts } from '../src/components/workbench/notebook/charts/ThermoCharts';
import TrajectoryPlayer from '../src/components/workbench/notebook/TrajectoryPlayer';
import { TrajectoryBuffer, type TrajectoryRecord } from '../src/components/workbench/notebook/trajectory';
import type { ThermoSegment } from '../src/components/workbench/notebook/chartData';
import { getThemeTokens } from '../src/theme';
import type { SimBox } from '../src/engine/types';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

const flush = async () => {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
  });
};

/** A 4 x 4 x 4 simple-cubic cell of 64 atoms, in `frames` frames with a small drift. */
const records = (frames: number): TrajectoryRecord[] => {
  const sim: SimBox = {
    lo: [0, 0, 0], hi: [4, 4, 4], periodic: [true, true, true], tilt: [0, 0, 0], triclinic: false,
    boundary: [['p', 'p'], ['p', 'p'], ['p', 'p']], minLo: [0, 0, 0], minHi: [4, 4, 4],
  };
  const id = Int32Array.from({ length: 64 }, (_, i) => i + 1);
  const type = new Int32Array(64).fill(1);
  const out: TrajectoryRecord[] = [];
  for (let f = 0; f < frames; f++) {
    const x = new Float32Array(3 * 64);
    for (let i = 0; i < 64; i++) {
      x[3 * i] = (i % 4) + 0.01 * f;
      x[3 * i + 1] = (Math.floor(i / 4) % 4);
      x[3 * i + 2] = Math.floor(i / 16) % 4;
    }
    out.push({ step: f * 10, x, box: sim, topo: { id, type }, v: new Float32Array(3 * 64).fill(0.5) });
  }
  return out;
};

const sectionOf = (label: string): HTMLElement => {
  const el = container.querySelector(`section[aria-label="${label}"]`);
  if (!el) throw new Error(`no section ${label}`);
  return el as HTMLElement;
};

describe('TrajectoryAnalysis extra sections', () => {
  it('renders coordination, bond-angle and S(q) sections with CSV/SVG/PNG, and the JSON export button', async () => {
    await act(async () => {
      root.render(<TrajectoryAnalysis records={records(4)} dt={0.005} units="lj" theme="light" stride={1} />);
    });
    const open = Array.from(container.querySelectorAll('button')).find((b) => b.textContent === 'Trajectory analysis');
    expect(open).toBeTruthy();
    await act(async () => { open!.click(); });
    await flush();

    for (const label of ['Coordination number', 'Bond-angle distribution', 'Structure factor S(q)']) {
      const section = sectionOf(label);
      const texts = Array.from(section.querySelectorAll('button')).map((b) => b.textContent);
      expect(texts).toEqual(expect.arrayContaining(['CSV', 'SVG', 'PNG']));
    }
    const cutoff = container.querySelector('input[aria-label="Neighbour cutoff"]') as HTMLInputElement | null;
    expect(cutoff).not.toBeNull();
    const jsonButton = Array.from(container.querySelectorAll('button')).find((b) => b.textContent === 'Export all analysis data (JSON)');
    expect(jsonButton).toBeTruthy();
  });

  it('shows the coordination mean once the neighbour analysis has run', async () => {
    await act(async () => {
      root.render(<TrajectoryAnalysis records={records(4)} dt={null} units="lj" theme="light" stride={1} />);
    });
    const open = Array.from(container.querySelectorAll('button')).find((b) => b.textContent === 'Trajectory analysis');
    await act(async () => { open!.click(); });
    await flush();
    expect(sectionOf('Coordination number').textContent).toContain('Mean coordination');
  });
});

describe('ThermoCharts column statistics', () => {
  it('lists the numeric columns and draws the three statistics for the chosen one', async () => {
    const rows = Array.from({ length: 40 }, (_, i) => ({ step: i * 10, temp: 1 + 0.1 * Math.sin(i), pe: -5 - i * 0.01 }));
    const seg: ThermoSegment = {
      label: 'run 1', keywords: ['step', 'temp', 'pe'], units: 'lj', rows, dt: 0.005, ensemble: 'NVE',
    };
    await act(async () => {
      root.render(<ThermoCharts segments={[seg]} theme="light" />);
    });
    const select = container.querySelector('select[aria-label="Thermo column statistics"]') as HTMLSelectElement | null;
    expect(select).not.toBeNull();
    const options = Array.from(select!.options).map((o) => o.value);
    expect(options).toEqual(expect.arrayContaining(['temp', 'pe']));
    const section = sectionOf('Column statistics');
    expect(section.textContent).toContain('Block averages');
    expect(section.textContent).toContain('Autocorrelation vs lag');
    expect(section.textContent).toContain('Histogram');
    const texts = Array.from(section.querySelectorAll('button')).map((b) => b.textContent);
    expect(texts.filter((t) => t === 'CSV')).toHaveLength(3);
  });

  it('says there is not enough data for a single row', async () => {
    const seg: ThermoSegment = {
      label: 'run 1', keywords: ['step', 'temp'], units: 'lj', rows: [{ step: 0, temp: 1 }], dt: null, ensemble: 'unknown',
    };
    await act(async () => {
      root.render(<ThermoCharts segments={[seg]} theme="light" />);
    });
    expect(sectionOf('Column statistics').textContent).toContain('Not enough data');
  });
});

describe('TrajectoryPlayer export formats', () => {
  it('lists pdb, data and csv besides xyz and dump', async () => {
    const buffer = new TrajectoryBuffer();
    buffer.restore(records(2));
    await act(async () => {
      root.render(
        <TrajectoryPlayer
          ct={getThemeTokens('light')}
          btn="btn"
          buffer={buffer}
          version={1}
          index={null}
          live={false}
          onShow={vi.fn()}
          canvasHost={{ current: null }}
          onVideoActive={vi.fn()}
        />,
      );
    });
    const select = container.querySelector('select[aria-label="Trajectory export format"]') as HTMLSelectElement | null;
    expect(select).not.toBeNull();
    const values = Array.from(select!.options).map((o) => o.value);
    expect(values).toEqual(['xyz', 'dump', 'pdb', 'csv', 'data']);
  });
});
