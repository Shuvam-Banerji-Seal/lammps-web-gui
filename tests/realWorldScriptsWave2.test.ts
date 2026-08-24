import { describe, it, expect } from 'vitest';
import { parseScript } from '../src/lammps/scriptParser';
import { generateScript } from '../src/lammps/generator';

/**
 * Corpus wave 2: DIFFUSE (msd/vector), KAPPA (heat/flux + ave/correlate),
 * VISCOSITY (SPCE water: kspace, shake, momentum, npt, dump_modify element).
 * Fetched verbatim from lammps/lammps examples (2026-08-24, develop).
 */

const IN_MSD_2D = `variable	x equal 40
variable	y equal 40

variable	rho equal 0.6
variable        t equal 1.0
variable	rc equal 2.5

units		lj
dimension	2
atom_style	atomic
neigh_modify	delay 0 every 1

lattice         sq2 \${rho}
region          simbox block 0 \$x 0 \$y -0.1 0.1
create_box      1 simbox
create_atoms    1 box

pair_style      lj/cut \${rc}
pair_coeff      * * 1 1

mass            * 1.0
velocity        all create \$t 97287

fix             1 all nve
fix	        2 all langevin \$t \$t 0.1 498094
fix	        3 all enforce2d

thermo          1000
run	        5000

unfix		2

reset_timestep  0

compute         msd all msd com yes
variable        twopoint equal c_msd[4]/4/(step*dt+1.0e-6)
fix             9 all vector 10 c_msd[4]
variable        fitslope equal slope(f_9)/4/(10*dt)

thermo_style	custom step temp c_msd[4] v_twopoint v_fitslope

thermo          1000
run	        100000`;

const IN_HEATFLUX = `variable        x equal 10
variable        y equal 10
variable        z equal 10

variable        rho equal 0.6
variable        t equal 1.35
variable        rc equal 2.5

variable    p equal 200     # correlation length
variable    s equal 10      # sample interval
variable    d equal \$p*\$s   # dump interval

units           lj
atom_style      atomic

lattice         fcc \${rho}
region          box block 0 \$x 0 \$y 0 \$z
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create \$t 87287

pair_style      lj/cut \${rc}
pair_coeff      1 1 1.0 1.0

neighbor        0.3 bin
neigh_modify    delay 0 every 1

fix             1 all nvt temp \$t \$t 0.5
thermo          100
run             1000

velocity        all scale \$t

unfix           1

reset_timestep  0

compute         myKE all ke/atom
compute         myPE all pe/atom
compute         myStress all stress/atom NULL virial
compute         flux all heat/flux myKE myPE myStress
variable        Jx equal c_flux[1]/vol
variable        Jy equal c_flux[2]/vol
variable        Jz equal c_flux[3]/vol

fix             1 all nve
fix             JJ all ave/correlate \$s \$p \$d &
                c_flux[1] c_flux[2] c_flux[3] type auto &
                file profile.heatflux ave running

variable        scale equal \$s*dt/\$t/\$t/vol
variable        k11 equal trap(f_JJ[3])*\${scale}
variable        kappa equal (v_k11+v_k22+v_k33)/3.0

thermo          \$d
thermo_style    custom step temp v_Jx v_Jy v_Jz v_k11 v_k22 v_k33 v_kappa
thermo_modify   colname v_Jx Jx colname v_Jy Jy colname v_Jz Jz &
                colname v_k11 kappa_11 colname v_k22 kappa_22 &
                colname v_k33 kappa_33 colname v_kappa kappa

run             100000

print           "Running average thermal conductivity: \$(v_kappa:%.2f)"`;

const IN_COS_SPCE = `units          real
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

fix com        all momentum 100 linear 1 1 1
fix rigid      all shake 1e-4 20 0 b 1 a 1

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

dump           1 all custom 10000 dump.lammpstrj id mol type element q xu yu zu
dump_modify    1 sort id element O H

run            2000`;

describe('real-world corpus wave 2 (DIFFUSE / KAPPA / VISCOSITY)', () => {
  it('in.msd.2d: 31/31 recognized incl. fix vector via any-style', () => {
    const { model, stats } = parseScript(IN_MSD_2D);
    expect(stats.raw).toBe(0);
    const defIds = model.steps.map(s => s.defId);
    expect(defIds).toContain('compute_msd');
    expect(defIds).toContain('fix_any'); // fix vector
    expect(defIds).toContain('velocity_create');
    const lm = model.steps.find(s => s.defId === 'lattice')!;
    expect(lm.params.style).toBe('sq2');
    const out = generateScript(model).text;
    expect(out).toContain('neigh_modify delay 0 every 1');
    expect(out).toContain('fix 9 all vector 10 c_msd[4]');
    expect(out).toContain('variable twopoint equal c_msd[4]/4/(step*dt+1.0e-6)');
  });

  it('in.heatflux: 46/46 recognized — heat/flux chain + ave/correlate tail', () => {
    const { model, stats } = parseScript(IN_HEATFLUX);
    expect(stats.raw).toBe(0);
    const corr = model.steps.find(s => s.defId === 'fix_ave_correlate')!;
    expect(corr.params.nevery).toBe('$s');
    expect(corr.params.nrepeat).toBe('$p');
    expect(corr.params.nfreq).toBe('$d');
    expect(corr.params.args).toBe('c_flux[1] c_flux[2] c_flux[3] type auto file profile.heatflux ave running');
    const scale = model.steps.find(s => s.defId === 'velocity_scale')!;
    expect(scale.params.temp).toBe('$t');
    const out = generateScript(model).text;
    expect(out).toContain('compute flux all heat/flux myKE myPE myStress');
    expect(out).toContain('fix JJ all ave/correlate $s $p $d');
    expect(out).toContain('c_flux[1] c_flux[2] c_flux[3] type auto');
    expect(out).toContain('thermo_modify colname v_Jx Jx');
    expect(out).toContain('print "Running average thermal conductivity: $(v_kappa:%.2f)"');
  });

  it('in.cos.1000SPCE: 35/35 recognized — full molecular workflow', () => {
    const { model, stats } = parseScript(IN_COS_SPCE);
    expect(stats.raw).toBe(0);
    const defIds = model.steps.map(s => s.defId);
    expect(defIds).toContain('fix_shake');
    expect(defIds).toContain('fix_momentum');
    expect(defIds).toContain('fix_npt');
    const shake = model.steps.find(s => s.defId === 'fix_shake')!;
    expect(shake.params.id).toBe('rigid');
    expect(shake.params.tol).toBe('1e-4');
    expect(shake.params.args).toBe('0 b 1 a 1');
    const mom = model.steps.find(s => s.defId === 'fix_momentum')!;
    expect(mom.params.id).toBe('com');
    expect(mom.params.args).toBe('linear 1 1 1');
    const dm = model.steps.find(s => s.defId === 'dump_modify')!;
    expect(dm.params.dumpid).toBe('1');
    expect(dm.params.extra).toContain('element O H');
    const out = generateScript(model).text;
    expect(out).toContain('pair_modify mix arithmetic');
    expect(out).toContain('pair_modify tail yes');
    expect(out).toContain('fix rigid all shake 1e-4 20 0 b 1 a 1');
    expect(out).toContain('fix com all momentum 100 linear 1 1 1');
    expect(out).toContain('dump_modify 1 sort id element O H');
  });

  it('wave-2 scripts regenerate with zero warnings', () => {
    for (const script of [IN_MSD_2D, IN_HEATFLUX, IN_COS_SPCE]) {
      const { model } = parseScript(script);
      expect(generateScript(model).warnings).toEqual([]);
    }
  });
});
