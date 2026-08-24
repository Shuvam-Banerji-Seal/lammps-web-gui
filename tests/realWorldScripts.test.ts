import { describe, it, expect } from 'vitest';
import { parseScript } from '../src/lammps/scriptParser';
import { generateScript } from '../src/lammps/generator';

/**
 * Real-world corpus: canonical tutorial/example scripts from the official
 * lammps/lammps GitHub repository (fetched 2026-08-24, develop branch).
 * These lock IMPORT quality against the scripts users actually bring.
 */

const IN_MELT = `# 3d Lennard-Jones melt

units           lj
atom_style      atomic

lattice         fcc 0.8442
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create 3.0 87287 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

neighbor        0.3 bin
neigh_modify    every 20 delay 0 check no

fix             1 all nve

thermo          50
run             250`;

const IN_CRACK = `# 2d LJ crack simulation

dimension       2
boundary        s s p

atom_style      atomic
neighbor        0.3 bin
neigh_modify    delay 5

lattice         hex 0.93
region          box block 0 100 0 40 -0.25 0.25
create_box      5 box
create_atoms    1 box

mass            1 1.0
mass            2 1.0
mass            3 1.0
mass            4 1.0
mass            5 1.0

pair_style      lj/cut 2.5
pair_coeff      * * 1.0 1.0 2.5

region          1 block INF INF INF 1.25 INF INF
group           lower region 1
region          2 block INF INF 38.75 INF INF INF
group           upper region 2
group           boundary union lower upper
group           mobile subtract all boundary

region          leftupper block INF 20 20 INF INF INF
region          leftlower block INF 20 INF 20 INF INF
group           leftupper region leftupper
group           leftlower region leftlower

set             group leftupper type 2
set             group leftlower type 3
set             group lower type 4
set             group upper type 5

compute         new mobile temp
velocity        mobile create 0.01 887723 temp new
velocity        upper set 0.0 0.3 0.0
velocity        mobile ramp vy 0.0 0.3 y 1.25 38.75 sum yes

fix             1 all nve
fix             2 boundary setforce NULL 0.0 0.0

thermo_style    custom step temp epair etotal c_new press
thermo          50
thermo_modify   lost warn

dump            1 all atom 250 dump.crack

dump_modify     1 sort id

run             12000`;

const IN_DEPOSIT = `units		lj
atom_style      atomic
boundary        p p f

lattice		fcc 1.0 origin 0.25 0.25 0.25
region          box block 0 5 0 5 0 10
create_box      2 box

region		substrate block INF INF INF INF INF 3
create_atoms	1 region substrate

pair_style	lj/cut 2.5
pair_coeff	* * 1.0 1.0
pair_coeff	1 2 1.0 1.0 5.0
mass		* 1.0

neigh_modify	delay 0

group		addatoms type 2
region          mobile block 0 5 0 5 2 INF
group		mobile region mobile

compute		add addatoms temp
compute_modify	add dynamic/dof yes extra/dof 0

fix		1 addatoms nve
fix		2 mobile langevin 1.0 1.0 0.1 587283
fix		3 mobile nve

region          slab block 0 5 0 5 8 9
fix		4 addatoms deposit 100 2 100 12345 region slab near 1.0 &
                vz -1.0 -1.0
fix		5 addatoms wall/reflect zhi EDGE

thermo_style	custom step atoms temp epair etotal press
thermo          100
thermo_modify	temp add`;

describe('real-world LAMMPS scripts import (lammps/lammps examples)', () => {
  it('in.melt imports fully recognized with exact parameters', () => {
    const { model, stats } = parseScript(IN_MELT);
    expect(stats.raw).toBe(0);
    const defIds = model.steps.map(s => s.defId);
    expect(defIds).toContain('lattice');
    expect(defIds).toContain('velocity_create');
    expect(defIds).toContain('fix_nve');
    // velocity extra keywords (loop geom) survive
    const vel = model.steps.find(s => s.defId === 'velocity_create')!;
    expect(vel.params.temp).toBe('3.0');
    expect(vel.params.extra).toContain('loop geom');
    // region ID preserved
    const region = model.steps.find(s => s.defId === 'region_block')!;
    expect(region.params.id).toBe('box');
    // round-trip keeps the neigh_modify keywords verbatim
    const out = generateScript(model).text;
    expect(out).toContain('neigh_modify every 20 delay 0 check no');
    expect(out).toContain('velocity all create 3.0 87287 loop geom');
  });

  it('in.crack imports fully — groups, sets, computes, ramp velocity', () => {
    const { model, stats } = parseScript(IN_CRACK);
    expect(stats.raw).toBe(0);
    const defIds = model.steps.map(s => s.defId);
    expect(defIds.filter(d => d === 'group').length).toBeGreaterThanOrEqual(6);
    expect(defIds.filter(d => d === 'region_block').length).toBeGreaterThanOrEqual(5);
    expect(defIds).toContain('set_type');
    expect(defIds).toContain('compute');
    expect(defIds).toContain('velocity_ramp');
    expect(defIds).toContain('fix_setforce');
    // region IDs used by groups are preserved
    const region1 = model.steps.filter(s => s.defId === 'region_block')[1];
    expect(region1.params.id).toBe('1');
    // ramp velocity keeps its exact grammar (vdim vlo vhi dim clo chi) + keywords
    const ramp = model.steps.find(s => s.defId === 'velocity_ramp')!;
    expect(ramp.params.vdim).toBe('vy');
    expect(ramp.params.vlo).toBe('0.0');
    expect(ramp.params.vhi).toBe('0.3');
    expect(ramp.params.axis).toBe('y');
    expect(ramp.params.clo).toBe('1.25');
    expect(ramp.params.chi).toBe('38.75');
    expect(ramp.params.extra).toBe('sum yes');
    // setforce NULL survives verbatim
    const out = generateScript(model).text;
    expect(out).toContain('fix 2 boundary setforce NULL 0.0 0.0');
    expect(out).toContain('group mobile subtract all boundary');
    expect(out).toContain('dump_modify 1 sort id');
  });

  it('in.deposit.atom imports fully — deposit keywords preserved', () => {
    const { model, stats } = parseScript(IN_DEPOSIT);
    expect(stats.raw).toBe(0);
    const dep = model.steps.find(s => s.defId === 'fix_deposit')!;
    expect(dep).toBeTruthy();
    expect(dep.params.n).toBe('100');
    expect(dep.params.type).toBe('2');
    expect(dep.params.m).toBe('100');
    expect(dep.params.seed).toBe('12345');
    // the &-continued vz keywords round-trip
    const out = generateScript(model).text;
    expect(out).toContain('fix 4 addatoms deposit 100 2 100 12345');
    expect(out).toContain('vz -1.0 -1.0');
    expect(out).toContain('fix 5 addatoms wall/reflect zhi EDGE');
    expect(out).toContain('compute_modify add dynamic/dof yes extra/dof 0');
    expect(out).toContain('lattice fcc 1.0 origin 0.25 0.25 0.25');
  });

  it('all three scripts regenerate with zero warnings', () => {
    for (const script of [IN_MELT, IN_CRACK, IN_DEPOSIT]) {
      const { model } = parseScript(script);
      const out = generateScript(model);
      // raw lines are emitted verbatim so they never warn; required-param
      // gaps are the only warning source and none of these have them.
      expect(out.warnings).toEqual([]);
    }
  });
});
