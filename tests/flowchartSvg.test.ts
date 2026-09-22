import { describe, it, expect } from 'vitest';
import { flowchartToSVG } from '../src/lammps/flowchartSvg';
import { buildTemplate } from '../src/lammps/templates';
import { SCRIPT_TEMPLATES } from '../src/lammps/templates';
import { DOMParser } from '@xmldom/xmldom';
import type { ScriptStep } from '../src/lammps/catalog';

const model = buildTemplate(SCRIPT_TEMPLATES.find(t => t.id === 'lj-nvt')!);

describe('flowchart SVG export', () => {
  it('produces well-formed XML containing every step', () => {
    const svg = flowchartToSVG(model);
    const doc = new DOMParser().parseFromString(svg, 'image/svg+xml');
    const texts = doc.getElementsByTagName('text');
    const all = Array.from(texts).map(t => t.textContent ?? '').join('\n');
    expect(all).toContain('Lennard-Jones fluid (NVT)');
    expect(all).toContain('units');
    expect(all).toContain('run');
    expect(all).toContain('START');
    expect(all).toContain('END');
  });

  it('renders both themes with distinct palettes', () => {
    const dark = flowchartToSVG(model, { theme: 'dark' });
    const light = flowchartToSVG(model, { theme: 'light' });
    expect(dark).toContain('#16130f');
    expect(light).toContain('#f4efe6');
  });

  it('hides disabled steps when asked and marks them dashed otherwise', () => {
    const m = { ...model, steps: model.steps.map((s, i) => (i === 0 ? { ...s, enabled: false } : s)) };
    const withDisabled = flowchartToSVG(m);
    expect(withDisabled).toContain('stroke-dasharray');
    const hidden = flowchartToSVG(m, { showDisabled: false });
    expect(hidden).not.toContain('stroke-dasharray');
  });

  it('escapes XML-special characters in titles and params', () => {
    const svg = flowchartToSVG({ title: 'A <b> & "c" test', steps: [] });
    expect(svg).toContain('A &lt;b&gt; &amp; &quot;c&quot; test');
    expect(() => new DOMParser().parseFromString(svg, 'image/svg+xml')).not.toThrow();
  });
});

describe('flowchart SVG — branching', () => {
  it('names the taken concept in the subtitle and marks the fork', () => {
    const trunk: ScriptStep[] = [
      { uid: 'a', defId: 'units', params: { style: 'lj' }, enabled: true },
      { uid: 'b', defId: 'fix_nve', params: {}, enabled: true },
    ];
    const svg = flowchartToSVG({
      title: 'Fork demo',
      steps: trunk,
      branches: [{
        id: 'br1',
        label: 'Hot run',
        forkAfter: 'a',
        steps: [{ uid: 'c', defId: 'run', params: { steps: '9999' }, enabled: true }] as ScriptStep[],
        rejoin: false,
      }],
      activeBranchIds: ['br1'],
    });
    expect(svg).toContain('concept: Hot run');
    expect(svg).toContain('HOT RUN');          // per-card concept label
    expect(svg).toMatch(/<path d="M \d+(\.\d+)? \d+(\.\d+)? L/); // fork diamond
    expect(svg).not.toContain('>fix nve<');    // trunk tail replaced
  });
});
