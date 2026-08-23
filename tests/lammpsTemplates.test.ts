import { describe, it, expect } from 'vitest';
import { SCRIPT_TEMPLATES, buildTemplate } from '../src/lammps/templates';
import { generateScript } from '../src/lammps/generator';

describe('script templates', () => {
  it('offers at least four starter pipelines', () => {
    expect(SCRIPT_TEMPLATES.length).toBeGreaterThanOrEqual(4);
  });

  it('every template generates with ZERO warnings and non-empty output', () => {
    for (const tpl of SCRIPT_TEMPLATES) {
      const model = buildTemplate(tpl);
      const out = generateScript(model);
      expect(out.warnings, `${tpl.id}: ${out.warnings.join('; ')}`).toHaveLength(0);
      expect(out.emitted.length, tpl.id).toBeGreaterThan(5);
      expect(out.text).not.toMatch(/undefined|NaN/);
    }
  });

  it('templates contain their signature commands', () => {
    const textOf = (id: string) => generateScript(buildTemplate(
      SCRIPT_TEMPLATES.find(t => t.id === id)!
    )).text;

    expect(textOf('lj-nvt')).toContain('units lj');
    expect(textOf('lj-nvt')).toContain('fix integrate all nvt temp 1.0 1.0 0.5');

    expect(textOf('eam-metal')).toContain('pair_style eam');
    expect(textOf('eam-metal')).toContain('minimize 0.0 1.0e-6 10000 100000');

    expect(textOf('reaxff')).toContain('pair_style reaxff');
    expect(textOf('granular-pour')).toContain('gravity 9.81 down');
    expect(textOf('granular-pour')).toContain('pour 2000 1 12345');
    expect(textOf('shear-nemd')).toContain('fix deform all deform 1 xy erate 0.1 remap v units box');
  });

  it('buildTemplate produces fresh uids on every call', () => {
    const a = buildTemplate(SCRIPT_TEMPLATES[0]);
    const b = buildTemplate(SCRIPT_TEMPLATES[0]);
    const aUids = new Set(a.steps.map(s => s.uid));
    for (const s of b.steps) expect(aUids.has(s.uid)).toBe(false);
  });
});
