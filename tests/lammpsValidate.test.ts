import { describe, it, expect } from 'vitest';
import { validateScript, diagnosticCounts } from '../src/lammps/validate';

/**
 * The validator must be SILENT on canonical, known-good LAMMPS inputs from the
 * official lammps/lammps examples tree — a linter that cries wolf on in.melt
 * is worse than no linter. Errors are hard-zero; warnings are listed explicitly
 * so any new rule that starts firing on real scripts shows up here.
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
run             250
`;

const IN_HEATFLUX = `# sample LAMMPS input script for thermal conductivity of liquid LJ
# Green-Kubo method via compute heat/flux and fix ave/correlate

# settings

variable        x equal 10
variable        y equal 10
variable        z equal 10

variable        rho equal 0.6
variable        t equal 1.35
variable        rc equal 2.5

#variable       rho equal 0.85
#variable        t equal 0.7
#variable       rc equal 3.0

variable    p equal 200     # correlation length
variable    s equal 10      # sample interval
variable    d equal $p*$s   # dump interval

# setup problem

units           lj
atom_style      atomic

lattice         fcc \${rho}
region          box block 0 $x 0 $y 0 $z
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create $t 87287

pair_style      lj/cut \${rc}
pair_coeff      1 1 1.0 1.0

neighbor        0.3 bin
neigh_modify    delay 0 every 1

# 1st equilibration run

fix             1 all nvt temp $t $t 0.5
thermo          100
run             1000

velocity        all scale $t

unfix           1

# thermal conductivity calculation

reset_timestep  0

compute         myKE all ke/atom
compute         myPE all pe/atom
compute         myStress all stress/atom NULL virial
compute         flux all heat/flux myKE myPE myStress
variable        Jx equal c_flux[1]/vol
variable        Jy equal c_flux[2]/vol
variable        Jz equal c_flux[3]/vol

fix             1 all nve
fix             JJ all ave/correlate $s $p $d &
                c_flux[1] c_flux[2] c_flux[3] type auto &
                file profile.heatflux ave running

variable        scale equal $s*dt/$t/$t/vol
variable        k11 equal trap(f_JJ[3])*\${scale}
variable        k22 equal trap(f_JJ[4])*\${scale}
variable        k33 equal trap(f_JJ[5])*\${scale}
variable        kappa equal (v_k11+v_k22+v_k33)/3.0

thermo          $d
thermo_style    custom step temp v_Jx v_Jy v_Jz v_k11 v_k22 v_k33 v_kappa
thermo_modify   colname v_Jx Jx colname v_Jy Jy colname v_Jz Jz &
                colname v_k11 kappa_11 colname v_k22 kappa_22 &
                colname v_k33 kappa_33 colname v_kappa kappa

run             100000

print           "Running average thermal conductivity: $(v_kappa:%.2f)"
`;

const IN_MSD2D = `# sample LAMMPS input script for diffusion of 2d LJ liquid
# mean-squared displacement via compute msd

# settings

variable	x equal 40
variable	y equal 40

variable	rho equal 0.6
variable        t equal 1.0
variable	rc equal 2.5

# problem setup

units		lj
dimension	2
atom_style	atomic
neigh_modify	delay 0 every 1

lattice         sq2 \${rho}
region          simbox block 0 $x 0 $y -0.1 0.1
create_box      1 simbox
create_atoms    1 box

pair_style      lj/cut \${rc}
pair_coeff      * * 1 1

mass            * 1.0
velocity        all create $t 97287

fix             1 all nve
fix	        2 all langevin $t $t 0.1 498094
fix	        3 all enforce2d

# equilibration run

thermo          1000
run	        5000

unfix		2

# data gathering run

reset_timestep  0

# factor of 4 in 2 variables is for 2d

compute         msd all msd com yes
variable        twopoint equal c_msd[4]/4/(step*dt+1.0e-6)
fix             9 all vector 10 c_msd[4]
variable        fitslope equal slope(f_9)/4/(10*dt)

thermo_style	custom step temp c_msd[4] v_twopoint v_fitslope

# only need to run for 10K steps to make a good 100-frame movie

#dump	        1 all custom 1 tmp.dump id type vx vy vz

#dump		2 all image 100 image.*.jpg type type zoom 1.6 adiam 1.2

thermo          1000
run	        100000
`;

const IN_SPCE = `# DFF generated Lammps input file

units          real
atom_style     full
boundary       p p p

pair_style     lj/cut/coul/long     10.0
pair_modify    mix arithmetic
pair_modify    tail yes
kspace_style   pppm 1.0e-4
dielectric     1.0
special_bonds  amber
bond_style     harmonic
angle_style    harmonic
dihedral_style none
improper_style none

read_data      data.cos.1000SPCE

variable T equal 300
variable P equal 1.0

velocity       all create \${T} 12345 mom yes rot yes dist gaussian

timestep       1.0

# Constraint ##################################
fix com        all momentum 100 linear 1 1 1
fix rigid      all shake 1e-4 20 0 b 1 a 1

# Viscosity  ##################################
variable       A equal 0.05e-5 # angstrom/fs^2

fix            cos all accelerate/cos \${A}
compute        cos all viscosity/cos

variable       density equal density
variable       lz equal lz
variable       vMax equal c_cos[7] # velocity of atoms at z=0
variable       invVis equal v_vMax/\${A}/v_density*39.4784/v_lz/v_lz*100 # reciprocal of viscosity 1/Pa/s

fix            npt all npt temp \${T} \${T} 100 iso \${P} \${P} 1000
fix_modify     npt temp cos

thermo_style   custom step cpu temp press pe density v_vMax v_invVis
thermo_modify  temp cos
thermo         100
################################################

dump           1 all custom 10000 dump.lammpstrj id mol type element q xu yu zu
dump_modify    1 sort id element O H

run            2000
`;


const REAL_SCRIPTS: [string, string][] = [
  ['in.melt', IN_MELT],
  ['in.heatflux', IN_HEATFLUX],
  ['in.msd.2d', IN_MSD2D],
  ['in.cos.1000SPCE', IN_SPCE],
];

describe('LAMMPS script validator — no false positives on the official corpus', () => {
  it.each(REAL_SCRIPTS)('%s produces zero errors', (_name, src) => {
    const diags = validateScript(src);
    const errors = diags.filter(d => d.level === 'error');
    expect(errors.map(e => `${e.line}: ${e.rule} — ${e.message}`)).toEqual([]);
  });

  it.each(REAL_SCRIPTS)('%s produces zero warnings', (_name, src) => {
    const warnings = validateScript(src).filter(d => d.level === 'warning');
    expect(warnings.map(w => `${w.line}: ${w.rule}`)).toEqual([]);
  });
});

describe('LAMMPS script validator — catches real ordering mistakes', () => {
  const fire = (src: string, rule: string) =>
    expect(validateScript(src).map(d => d.rule)).toContain(rule);

  it('flags units after the box is defined', () => {
    fire(`atom_style atomic
region box block 0 1 0 1 0 1
create_box 1 box
units real`, 'order/units-after-box');
  });

  it('flags boundary after read_data', () => {
    fire('read_data in.data\nboundary p p f', 'order/boundary-after-box');
  });

  it('flags pair_coeff before any pair_style', () => {
    fire(`atom_style atomic
region box block 0 1 0 1 0 1
create_box 1 box
pair_coeff * * 1.0 1.0`, 'order/pair_coeff-without-pair_style');
  });

  it('flags mass before the box exists', () => {
    fire('units lj\nmass 1 1.0', 'order/mass-before-box');
  });

  it('flags masses assigned after velocity', () => {
    fire(`atom_style atomic
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
velocity all create 1.0 12345
mass 1 1.0`, 'order/mass-after-velocity');
  });

  it('flags unfix of an unknown fix id', () => {
    fire('read_data in.data\nunfix nope', 'ref/unknown-fix');
  });

  it('flags reuse of a fix id with a different style', () => {
    fire('read_data in.data\nfix 1 all nve\nfix 1 all nvt temp 1 1 0.1', 'id/fix-reuse');
  });

  it('accepts re-issuing a fix id with the SAME style (LAMMPS allows it)', () => {
    const rules = validateScript('read_data in.data\nfix 1 all nve\nfix 1 all nve').map(d => d.rule);
    expect(rules).not.toContain('id/fix-reuse');
  });

  it('flags a run with no time-integration fix', () => {
    fire(`read_data in.data
pair_style lj/cut 2.5
pair_coeff * * 1 1
run 100`, 'run/no-integrator');
  });

  it('flags a group that was never defined', () => {
    fire('read_data in.data\nfix 1 mobile nve', 'ref/unknown-group');
  });

  it('flags create_box referencing an undefined region', () => {
    fire('atom_style atomic\ncreate_box 1 ghost', 'ref/unknown-region');
  });

  it('flags create_atoms with no lattice', () => {
    fire(`atom_style atomic
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box`, 'order/create_atoms-without-lattice');
  });

  it('flags a kspace solver paired with a short-range-only pair style', () => {
    fire(`read_data in.data
pair_style lj/cut 10.0
pair_coeff * * 1 1
kspace_style pppm 1.0e-4
fix 1 all nve
run 10`, 'kspace/pair-mismatch');
  });

  it('flags a script that never runs anything', () => {
    fire('units lj\natom_style atomic', 'script/no-run');
  });

  it('counts diagnostics by level', () => {
    const c = diagnosticCounts(validateScript('units lj\natom_style atomic'));
    expect(c.errors + c.warnings).toBeGreaterThan(0);
  });

  it('returns nothing for empty input', () => {
    expect(validateScript('')).toEqual([]);
    expect(validateScript('# just a comment\n')).toEqual([]);
  });
});
