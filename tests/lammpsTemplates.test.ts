import { describe, it, expect } from 'vitest';
import { SCRIPT_TEMPLATES, buildTemplate } from '../src/lammps/templates';
import { generateScript } from '../src/lammps/generator';
import { validateScript } from '../src/lammps/validate';

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
    expect(textOf('granular-pour')).toContain('atom_style sphere');
    expect(textOf('granular-pour')).toContain('comm_modify vel yes');
    expect(textOf('granular-pour')).toContain('fix grav all gravity 10.0 vector 0 0 -1');
    expect(textOf('granular-pour')).toContain('pour 1500 1 3123 region insreg');
    expect(textOf('granular-pour')).toContain('nve/sphere');
    expect(textOf('shear-nemd')).toContain('fix deform all deform 1 xy erate 0.1 remap v units box');
  });

  it('buildTemplate produces fresh uids on every call', () => {
    const a = buildTemplate(SCRIPT_TEMPLATES[0]);
    const b = buildTemplate(SCRIPT_TEMPLATES[0]);
    const aUids = new Set(a.steps.map(s => s.uid));
    for (const s of b.steps) expect(aUids.has(s.uid)).toBe(false);
  });
});

describe('every shipped template is a VALID LAMMPS script', () => {
  it.each(SCRIPT_TEMPLATES.map(t => [t.id, t] as const))(
    '%s passes the doc-grounded validator with zero errors and zero warnings',
    (_id, tpl) => {
      const text = generateScript(buildTemplate(tpl)).text;
      const diags = validateScript(text);
      expect(diags.map(d => `${d.level} ${d.line}: ${d.rule} — ${d.message}`)).toEqual([]);
    },
  );

  it('the LJ template defines a lattice before create_atoms', () => {
    const text = generateScript(buildTemplate(SCRIPT_TEMPLATES.find(t => t.id === 'lj-nvt')!)).text;
    expect(text.indexOf('lattice fcc')).toBeGreaterThan(-1);
    expect(text.indexOf('lattice fcc')).toBeLessThan(text.indexOf('create_atoms'));
  });

  it('the EAM template uses a per-I,I pair_coeff (funcfl files are single-element)', () => {
    const text = generateScript(buildTemplate(SCRIPT_TEMPLATES.find(t => t.id === 'eam-metal')!)).text;
    expect(text).toContain('pair_style eam');
    expect(text).toContain('pair_coeff 1 1 Cu_u3.eam');
    expect(text).not.toContain('pair_coeff * * Cu_u3.eam');
  });

  it('the ReaxFF template passes the control file to pair_style and runs QEq', () => {
    const text = generateScript(buildTemplate(SCRIPT_TEMPLATES.find(t => t.id === 'reaxff')!)).text;
    expect(text).toContain('pair_style reaxff NULL');
    expect(text).toMatch(/pair_coeff \* \* ffield\.reax( \w+)+/);
    expect(text).toMatch(/fix \w+ all qeq\/reaxff 1 0\.0 10\.0 1\.0e-6 reaxff/);
    expect(text).not.toContain('shake');   // reactive FF: no bond constraints
  });

  it('the shear template thermostats with nvt/sllod, not plain nvt', () => {
    const text = generateScript(buildTemplate(SCRIPT_TEMPLATES.find(t => t.id === 'shear-nemd')!)).text;
    expect(text).toContain('nvt/sllod');
    expect(text).toContain('remap v');
  });
});
