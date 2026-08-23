import { describe, it, expect } from 'vitest';
import {
  parseScript,
  matchLine,
  tokenizeLine,
  scriptStatements,
} from '../src/lammps/scriptParser';
import { generateScript } from '../src/lammps/generator';

const LJ_SCRIPT = `# LJ fluid demo
units lj
dimension 3
boundary p p p
atom_style atomic

region box block 0 16.8 0 16.8 0 16.8 units box
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 4928459 dist gaussian
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
neighbor 2.0 bin
thermo_style custom step temp pe press vol density
thermo 1000
dump traj all custom 1000 traj.lammpstrj id type x y z vx vy vz
fix integrate all nvt temp 1.0 1.0 0.5
run 50000`;

describe('script importer', () => {
  it('tokenizes quote-aware', () => {
    expect(tokenizeLine('print "hello world" 2')).toEqual(['print', '"hello world"', '2']);
  });

  it('joins & continuations and strips comments', () => {
    const stmts = scriptStatements(`# comment only
units real   # inline comment
pair_style lj/cut &
    2.5
`);
    expect(stmts).toEqual(['units real', 'pair_style lj/cut 2.5']);
  });

  it('recognizes the whole canonical LJ script with zero raw lines', () => {
    const { model, stats } = parseScript(LJ_SCRIPT);
    expect(stats.total).toBe(17);
    expect(stats.raw).toBe(0);
    expect(stats.recognized).toBe(17);
    const ids = model.steps.map(s => s.defId);
    expect(ids).toContain('units');
    expect(ids).toContain('pair_style_popular');
    expect(ids).toContain('fix_nvt');
    expect(ids).toContain('dump_custom');
    // parameters extracted, not just recognized
    const nvt = model.steps.find(s => s.defId === 'fix_nvt')!;
    expect(nvt.params.temp_start).toBe('1.0');
    expect(nvt.params.temp_damp).toBe('0.5');
    const ps = model.steps.find(s => s.defId === 'pair_style_popular')!;
    expect(ps.params.style).toBe('lj/cut');
    expect(ps.params.args).toBe('2.5');
  });

  it('imported model regenerates to an equivalent script', () => {
    const { model } = parseScript(LJ_SCRIPT);
    const out = generateScript(model);
    expect(out.text).toContain('units lj');
    expect(out.text).toContain('pair_style lj/cut 2.5');
    expect(out.text).toContain('pair_coeff * * 1.0 1.0 2.5');
    expect(out.text).toContain('fix integrate all nvt temp 1.0 1.0 0.5');
    expect(out.text).toContain('run 50000');
  });

  it('flexes user-chosen IDs for fix/dump/compute/region/group', () => {
    const m = matchLine(tokenizeLine('fix myThermo all nvt temp 300 300 0.1'));
    expect(m?.def.id).toBe('fix_nvt');
    const g = matchLine(tokenizeLine('group mobile type 1 2'));
    expect(g?.def.id).toBe('group');
    const r = matchLine(tokenizeLine('region myBox block 0 10 0 10 0 10'));
    expect(r?.def.id).toBe('region_block');
    const c = matchLine(tokenizeLine('compute myTemp all temp'));
    expect(c?.def.id).toBe('compute');
  });

  it('disambiguates same-keyword commands via enum validation', () => {
    expect(matchLine(tokenizeLine('pair_style hybrid/overlay lj/cut 2.5 coul/long 10.0'))?.def.id)
      .toBe('pair_style_hybrid');
    expect(matchLine(tokenizeLine('variable myT equal temp'))?.def.id).toBe('variable_eq');
    expect(matchLine(tokenizeLine('variable myC atom x>0.5'))?.def.id).toBe('variable_atom');
    expect(matchLine(tokenizeLine('fix walls all wall lj126 zlo 0.0 1.0 1.0 3.0 units box'))?.def.id)
      .toBe('fix_wall_potential');
    expect(matchLine(tokenizeLine('fix dt all dt/reset 1 1e-4 1e-2 0.1'))?.def.id)
      .toBe('fix_dt_reset');
  });

  it('falls back to raw_line for unknown syntax without losing text', () => {
    const { model, stats } = parseScript(`units metal
fancy_command 1 2 3
run 10`);
    expect(stats.raw).toBe(1);
    const raw = model.steps.find(s => s.defId === 'raw_line')!;
    expect(raw.params.line).toBe('fancy_command 1 2 3');
    // raw line round-trips verbatim
    const out = generateScript(model);
    expect(out.text).toContain('fancy_command 1 2 3');
  });

  it('handles quoted print and absorb-style commands', () => {
    const { model, stats } = parseScript(`print "Step ${'${step}'} done"
shell mkdir results
include run.npt`);
    expect(stats.raw).toBe(0);
    const pr = model.steps.find(s => s.defId === 'print_cmd')!;
    expect(pr.params.text).toBe('"Step ${step} done"');
    const sh = model.steps.find(s => s.defId === 'shell_cmd')!;
    expect(sh.params.cmd).toBe('mkdir results');
  });
});
