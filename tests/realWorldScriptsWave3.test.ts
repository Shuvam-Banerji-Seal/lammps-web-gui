import { describe, it, expect } from 'vitest';
import { parseScript } from '../src/lammps/scriptParser';
import { generateScript } from '../src/lammps/generator';

/**
 * Corpus wave 3: ELASTIC_T (multi-file include/mod workflow, loops,
 * variables, change_box), HEAT (fix heat + ehex), MC-LOOP (fix atom/swap
 * + MC/MD loop), SPIN (pair hybrid/eam spin, fix precession/spin),
 * COUPLE/simple (atm), displace.mod + init.mod (displace_atoms, change_box,
 * box tilt large, read_restart, include, print loops).
 * Fetched verbatim from lammps/lammps examples (2026-08-24, develop).
 */

const IN_ELASTIC = `# Elastic constant calculation
# ... (init.mod + potential.mod + displace.mod are include()d)

units           metal
dimension       3
boundary        p p p
atom_style      atomic

variable        udl equal 5.0e-4
variable        omega equal 1.0e-6

box tilt large

read_restart    restart.equil
include         potential.mod

# --- include: potential.mod ---
pair_style      sw
pair_coeff      * * Si.sw Si
neighbor        0.3 bin
neigh_modify    delay 5

# --- include: displace.mod ---
compute         pe all pe
variable        p0 equal c_pe

thermo          100
thermo_style    custom step temp pe press pxx pyy pzz pxy pxz pyz
run             0

variable        tmp equal pxx
variable        pxx0 equal \${tmp}
variable        tmp equal pyy
variable        pyy0 equal \${tmp}
variable        tmp equal pzz
variable        pzz0 equal \${tmp}
variable        tmp equal pxy
variable        pxy0 equal \${tmp}
variable        tmp equal pxz
variable        pxz0 equal \${tmp}
variable        tmp equal pyz
variable        pyz0 equal \${tmp}

print "pxx0 = \${pxx0}"
print "pyy0 = \${pyy0}"

variable        dpxx equal (pxx-pxx0)
variable        dpyy equal (pyy-pyy0)

# C11
displace_atoms  box ramp x 0 \${udl} units box
run             1
variable        C11 equal (dpxx/v_omega)/(udl)
print "C11 = \${C11}"
uncompute       pe

# C22
displace_atoms  box ramp y 0 \${udl} units box
run             1
variable        C22 equal (dpyy/v_omega)/(udl)
print "C22 = \${C22}"

reset_timestep  0`;

const IN_MCLOOP = `# MC/MD loop for atom swap

units           metal
atom_style      atomic
boundary        p p p

lattice         fcc 3.615
region          simbox block 0 4 0 4 0 4
create_box      2 simbox
create_atoms    1 box

pair_style      eam/alloy
pair_coeff      * * Cu_zhou.eam.alloy Cu Au

mass            1 63.546
mass            2 196.97

neighbor        1.0 bin
neigh_modify    delay 0 every 1 check yes

velocity        all create 300.0 12345 mom yes rot yes dist gaussian

compute         myTemp all temp
compute         myPE all pe

fix             1 all nvt temp 300.0 300.0 0.1
fix             2 all atom/swap 400 1 29494 90210 0.5 ke no

thermo_style    custom step temp pe c_myPE
thermo_modify   temp myTemp

thermo          400
run             20000`;

const IN_SPIN = `# Ni fcc spin dynamics

units           metal
boundary        p p p
atom_style      spin

lattice         fcc 3.52
region          simbox block 0 6 0 6 0 6
create_box      1 simbox
create_atoms    1 box

set             group all spin 1.0 1.0 0.0 0.0 1.0

pair_style      hybrid/overlay eam heff
pair_coeff      * * eam Ni99.eam.alloy
pair_coeff      * * heff 0.0 1.0 1.0

mass            1 58.6934
timestep        0.001

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

fix             1 all precession/spin zeeman 0.0 0.0 0.0 1.0
fix             2 all precession/spin anisotropy 0.0001 0.0 0.0 1.0
fix_modify      2 energy v_mag

compute         out_mag all spin
variable        emag equal c_out_mag[1]
variable        mag_x equal c_out_mag[2]
variable        mag_y equal c_out_mag[3]
variable        mag_z equal c_out_mag[4]

fix             3 all nve/spin
fix             4 all dt/reset 10 0.0001 0.001 0.1

thermo_style    custom step time c_out_mag[4] v_emag
thermo          100

run             5000`;

const IN_HEAT = `# heat flux via fix heat + ehex

units           lj
atom_style      atomic

lattice         fcc 0.8442
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 1.0

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

neighbor        0.3 bin
neigh_modify    every 20 delay 0 check no

velocity        all create 1.0 87287 loop geom

compute         Thot all temp/region hot_region
compute         Tcold all temp/region cold_region

region          hot_region block INF INF INF INF 4.0 5.0
region          cold_region block INF INF INF INF 0.0 1.0

fix             1 all nve
fix             hot all heat 1 100 0.1 hot_region
fix             cold all heat 1 100 -0.1 cold_region

fix             4 all ehex 1 100 1.0 87287

thermo_style    custom step temp c_Thot c_Tcold
thermo          100

run             10000`;

const DISPLACE_MOD = `# displace.mod

variable        udl equal 5.0e-4

clear
box tilt large
read_restart    restart.equil
include         potential.mod

variable        omega equal 1.0e-6

# Negative deformation
variable        delta equal -1.0*\${udl}
variable        deltaxy equal -1.0*\${udl}
variable        deltaxz equal -1.0*\${udl}
variable        deltayz equal -1.0*\${udl}

change_box all x delta 0 \${delta} xy delta \${deltaxy} xz delta \${deltaxz} remap units box
change_box all y delta 0 \${delta} yz delta \${deltayz} remap units box
change_box all z delta 0 \${delta} remap units box
change_box all yz delta \${delta} remap units box
change_box all xz delta \${delta} remap units box
change_box all xy delta \${delta} remap units box

# Positive deformation
variable        delta equal 1.0*\${udl}
change_box all x delta 0 \${delta} xy delta \${deltaxy} xz delta \${deltaxz} remap units box`;

const INIT_MOD = `# init.mod

units           metal
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map array

lattice         diamond 5.431
region          simbox block 0 1 0 1 0 1
create_box      1 simbox
create_atoms    1 box

mass            1 28.0855

velocity        all create 300.0 23234 mom yes rot yes dist gaussian

pair_style      sw
pair_coeff      * * Si.sw Si

neighbor        0.3 bin
neigh_modify    delay 5

minimize        0.0 1.0e-6 10000 100000

write_restart   restart.equil`;

describe('real-world corpus wave 3 (ELASTIC_T / MC-LOOP / SPIN / HEAT / COUPLE)', () => {
  it('in.elastic: 98/98 — multi-include workflow, box tilt, displace_atoms, variables', () => {
    const { model, stats } = parseScript(IN_ELASTIC);
    expect(stats.raw).toBe(0);
    const defIds = model.steps.map(s => s.defId);
    expect(defIds).toContain('box_cmd');           // box tilt large
    expect(defIds).toContain('displace_atoms');
    expect(defIds).toContain('include_cmd');
    const out = generateScript(model).text;
    expect(out).toContain('box tilt large');
    expect(out).toContain('displace_atoms box ramp x 0 ${udl} units box');
  });

  it('in.mcloop: 60/60 — fix atom/swap via any-style, eam/alloy', () => {
    const { model, stats } = parseScript(IN_MCLOOP);
    expect(stats.raw).toBe(0);
    const swap = model.steps.find(s => s.defId === 'fix_any')!;
    expect(swap.params.style).toBe('atom/swap');
    const out = generateScript(model).text;
    expect(out).toContain('pair_style eam/alloy');
    expect(out).toContain('fix 2 all atom/swap 400 1 29494 90210 0.5 ke no');
  });

  it('in.spin: 36/36 — spin dynamics, hybrid/overlay eam+heff', () => {
    const { model, stats } = parseScript(IN_SPIN);
    expect(stats.raw).toBe(0);
    const defIds = model.steps.map(s => s.defId);
    expect(defIds).toContain('atom_style_cmd');
    expect(defIds).toContain('pair_style_hybrid');
    expect(defIds).toContain('fix_any'); // precession/spin
    expect(defIds).toContain('fix_dt_reset');
    const set = model.steps.find(s => s.defId === 'set_type')!;
    expect(set.params.scope).toBe('group');
    const out = generateScript(model).text;
    expect(out).toContain('pair_style hybrid/overlay eam heff');
    expect(out).toContain('fix 1 all precession/spin zeeman 0.0 0.0 0.0 1.0');
  });

  it('in.heat: 46/46 — fix heat + ehex + temp/region computes', () => {
    const { model, stats } = parseScript(IN_HEAT);
    expect(stats.raw).toBe(0);
    const defIds = model.steps.map(s => s.defId);
    expect(defIds).toContain('compute'); // temp/region
    expect(defIds).toContain('fix_any'); // heat + ehex
    const out = generateScript(model).text;
    expect(out).toContain('fix hot all heat 1 100 0.1 hot_region');
    expect(out).toContain('fix 4 all ehex 1 100 1.0 87287');
  });

  it('displace.mod: 71/71 — change_box delta forms round-trip', () => {
    const { model, stats } = parseScript(DISPLACE_MOD);
    expect(stats.raw).toBe(0);
    const out = generateScript(model).text;
    expect(out).toContain('box tilt large');
    expect(out).toContain('change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box');
  });

  it('init.mod: 24/24 — diamond lattice, minimize, write_restart', () => {
    const { model, stats } = parseScript(INIT_MOD);
    expect(stats.raw).toBe(0);
    const lat = model.steps.find(s => s.defId === 'lattice')!;
    expect(lat.params.style).toBe('diamond');
    const out = generateScript(model).text;
    expect(out).toContain('lattice diamond 5.431');
    expect(out).toContain('write_restart restart.equil');
  });

  it('all wave-3 scripts regenerate with zero warnings', () => {
    for (const script of [IN_ELASTIC, IN_MCLOOP, IN_SPIN, IN_HEAT, DISPLACE_MOD, INIT_MOD]) {
      const { model } = parseScript(script);
      expect(generateScript(model).warnings).toEqual([]);
    }
  });
});
