import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Force units of the force-style fixes (fix viscous, gravity, efield, efield/lepton, lineforce, planeforce,
 * drag, indent, addtorque, addtorque/group, accelerate/cos, rigid ... gravity). The engine stores forces in
 * LAMMPS force units (the integrators apply v += dt ftm2v F / m), so a fix force carries no ftm2v factor
 * except where the physics is a velocity change: the omega terms of fix addtorque are kinematic.
 *
 * Every expected value below is a native LAMMPS measurement (black box, run from the same input text):
 * the per-atom fx fy fz vx vy vz after one run step, and where a case has an energy, the potential energy
 * after that step. Units real and metal differ by ftm2v (4.184e-4 and 9648.5), so each case is run in both.
 * The multi-atom parity cases are in tests/oracle/w66units_* (engineOracle.test.ts).
 */

interface UnitCase {
  name: string;
  script: string;
  /** native per-atom [fx, fy, fz, vx, vy, vz], atom id order */
  atoms: number[][];
  /** native potential energy after the one step, when the case prints it */
  pe?: number;
}

const CASES: UnitCase[] = [
  { name: 'accel_cos_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all accelerate/cos 0.002
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.00000248742456000, 0.00000000000, 0.00000000000, 0.000998000000000, 0.00200000000000, 0.00300000000000]], },
  { name: 'accel_cos_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all accelerate/cos 0.002
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-57.3612747727, 0.00000000000, 0.00000000000, -0.000999998223472, 0.00200000000000, 0.00300000000000]], },
  { name: 'addforce_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addforce 1 2 3
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[1.00000000000, 2.00000000000, 3.00000000000, 0.805044485273, 1.61008897055, 2.41513345582]], },
  { name: 'addforce_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addforce 1 2 3
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[1.00000000000, 2.00000000000, 3.00000000000, 0.00103486666666, 0.00206973333333, 0.00310459999999]], },
  { name: 'addtorque4_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6.1 5 5
create_atoms 1 single 5 6.1 5.3
create_atoms 1 single 5.2 5.4 6.1
mass 1 12.0
set atom 1 charge 2.0
set atom 2 charge -1.0
set atom 3 charge 0.5
set atom 4 charge 1.5
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addtorque 0.02 0.01 -0.015
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.00448854256162, 0.00836321444027, -0.00479245972804, -0.00260900228773, 0.00872436818688, -0.000853385478825], [-0.00448862333889, -0.00371979991805, -0.00595347500218, -0.00260903476199, -0.000990900342375, -0.00178684994412], [0.00791107707169, 0.00425458437123, 0.0102717180502, 0.00736083328630, 0.00542092606423, 0.0112589143649], [0.00106608882882, -0.00889799889345, 0.000474216680036, 0.00185720376342, -0.00515439390873, 0.00338132105809]], },
  { name: 'addtorque4_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6.1 5 5
create_atoms 1 single 5 6.1 5.3
create_atoms 1 single 5.2 5.4 6.1
mass 1 12.0
set atom 1 charge 2.0
set atom 2 charge -1.0
set atom 3 charge 0.5
set atom 4 charge 1.5
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addtorque 0.02 0.01 -0.015
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.00448857681337, 0.00836314718686, -0.00479254221217, 0.000999843498261, 0.00200029159501, 0.00299983289996], [-0.00448858031619, -0.00371983735318, -0.00595345406304, 0.000999843498200, 0.00199987030164, 0.00299979242292], [0.00791101852085, 0.00425470564172, 0.0102717087993, 0.00100027583080, 0.00200014834750, 0.00300035814024], [0.00106613860872, -0.00889801547540, 0.000474287475934, 0.00100003717274, 0.00199968975585, 0.00300001653688]], },
  { name: 'addtorque4nu_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6.1 5 5
create_atoms 1 single 5 6.1 5.3
create_atoms 1 single 5.2 5.4 6.1
mass 1 12.0
set atom 1 charge 2.0
set atom 2 charge -1.0
set atom 3 charge 0.5
set atom 4 charge 1.5
velocity all set 0.001 0.002 0.003
set atom 1 vx 0.002 vy -0.001 vz 0.0015
set atom 2 vx -0.001 vy 0.003 vz 0.0007
set atom 3 vx 0.0005 vy 0.0012 vz -0.002
set atom 4 vx 0.0021 vy 0.0004 vz 0.0031
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addtorque 0.02 0.01 -0.015
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.00448858839748, 0.00836316028878, -0.00479246341236, -0.00160901914615, 0.00572434739664, -0.00235338610443], [-0.00448861256930, -0.00371979414983, -0.00595345620654, -0.00460903133258, 0.00000910168005426, -0.00408684297042], [0.00791106666922, 0.00425464548474, 0.0102716963013, 0.00686082932619, 0.00462094887915, 0.00625890670813], [0.00106613429756, -0.00889801162369, 0.000474223317636, 0.00295722115254, -0.00675439795584, 0.00348132236671]], pe: 5.31477533788e-8, },
  { name: 'addtorque4nu_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6.1 5 5
create_atoms 1 single 5 6.1 5.3
create_atoms 1 single 5.2 5.4 6.1
mass 1 12.0
set atom 1 charge 2.0
set atom 2 charge -1.0
set atom 3 charge 0.5
set atom 4 charge 1.5
velocity all set 0.001 0.002 0.003
set atom 1 vx 0.002 vy -0.001 vz 0.0015
set atom 2 vx -0.001 vy 0.003 vz 0.0007
set atom 3 vx 0.0005 vy 0.0012 vz -0.002
set atom 4 vx 0.0021 vy 0.0004 vz 0.0031
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addtorque 0.02 0.01 -0.015
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.0845928167899, 0.0643901699409, 0.0435470058204, 0.00200296510717, -0.000997751799872, 0.00150153119564], [-0.0558869306528, -0.0205645411198, -0.0392303578285, -0.00100195275288, 0.00299928012058, 0.000698629733494], [0.0210655637436, -0.0959596475134, 0.0726096977361, 0.000500727068501, 0.00119664721027, -0.00199746827357], [-0.0497714498806, 0.0521340186923, -0.0769263457281, 0.00209826057721, 0.000401824469021, 0.00309730734444]], pe: 0.219402340725, },
  { name: 'addtorqueg4_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6.1 5 5
create_atoms 1 single 5 6.1 5.3
create_atoms 1 single 5.2 5.4 6.1
mass 1 12.0
set atom 1 charge 2.0
set atom 2 charge -1.0
set atom 3 charge 0.5
set atom 4 charge 1.5
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addtorque/group 0.02 0.01 -0.015
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.00448854256162, 0.00836321444027, -0.00479245972804, -0.00260900228773, 0.00872436818688, -0.000853385478825], [-0.00448862333889, -0.00371979991805, -0.00595347500218, -0.00260903476199, -0.000990900342375, -0.00178684994412], [0.00791107707169, 0.00425458437123, 0.0102717180502, 0.00736083328630, 0.00542092606423, 0.0112589143649], [0.00106608882882, -0.00889799889345, 0.000474216680036, 0.00185720376342, -0.00515439390873, 0.00338132105809]], },
  { name: 'addtorqueg4_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6.1 5 5
create_atoms 1 single 5 6.1 5.3
create_atoms 1 single 5.2 5.4 6.1
mass 1 12.0
set atom 1 charge 2.0
set atom 2 charge -1.0
set atom 3 charge 0.5
set atom 4 charge 1.5
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addtorque/group 0.02 0.01 -0.015
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.00448857681337, 0.00836314718686, -0.00479254221217, 0.000999843498261, 0.00200029159501, 0.00299983289996], [-0.00448858031619, -0.00371983735318, -0.00595345406304, 0.000999843498200, 0.00199987030164, 0.00299979242292], [0.00791101852085, 0.00425470564172, 0.0102717087993, 0.00100027583080, 0.00200014834750, 0.00300035814024], [0.00106613860872, -0.00889801547540, 0.000474287475934, 0.00100003717274, 0.00199968975585, 0.00300001653688]], },
  { name: 'addtorqueg4nu_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6.1 5 5
create_atoms 1 single 5 6.1 5.3
create_atoms 1 single 5.2 5.4 6.1
mass 1 12.0
set atom 1 charge 2.0
set atom 2 charge -1.0
set atom 3 charge 0.5
set atom 4 charge 1.5
velocity all set 0.001 0.002 0.003
set atom 1 vx 0.002 vy -0.001 vz 0.0015
set atom 2 vx -0.001 vy 0.003 vz 0.0007
set atom 3 vx 0.0005 vy 0.0012 vz -0.002
set atom 4 vx 0.0021 vy 0.0004 vz 0.0031
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addtorque/group 0.02 0.01 -0.015
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.00448858839748, 0.00836316028878, -0.00479246341236, -0.00160901914615, 0.00572434739664, -0.00235338610443], [-0.00448861256930, -0.00371979414983, -0.00595345620654, -0.00460903133258, 0.00000910168005426, -0.00408684297042], [0.00791106666922, 0.00425464548474, 0.0102716963013, 0.00686082932619, 0.00462094887915, 0.00625890670813], [0.00106613429756, -0.00889801162369, 0.000474223317636, 0.00295722115254, -0.00675439795584, 0.00348132236671]], },
  { name: 'addtorqueg4nu_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6.1 5 5
create_atoms 1 single 5 6.1 5.3
create_atoms 1 single 5.2 5.4 6.1
mass 1 12.0
set atom 1 charge 2.0
set atom 2 charge -1.0
set atom 3 charge 0.5
set atom 4 charge 1.5
velocity all set 0.001 0.002 0.003
set atom 1 vx 0.002 vy -0.001 vz 0.0015
set atom 2 vx -0.001 vy 0.003 vz 0.0007
set atom 3 vx 0.0005 vy 0.0012 vz -0.002
set atom 4 vx 0.0021 vy 0.0004 vz 0.0031
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all addtorque/group 0.02 0.01 -0.015
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.0845928167899, 0.0643901699409, 0.0435470058204, 0.00200296510717, -0.000997751799872, 0.00150153119564], [-0.0558869306528, -0.0205645411198, -0.0392303578285, -0.00100195275288, 0.00299928012058, 0.000698629733494], [0.0210655637436, -0.0959596475134, 0.0726096977361, 0.000500727068501, 0.00119664721027, -0.00199746827357], [-0.0497714498806, 0.0521340186923, -0.0769263457281, 0.00209826057721, 0.000401824469021, 0.00309730734444]], },
  { name: 'drag_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all drag 0.0 0.0 0.0 0.5 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.288675076858, -0.288675134595, -0.288675192331, -0.231107626795, -0.230107650006, -0.229107673218]], },
  { name: 'drag_nb2_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 6 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all drag 3 3 3 0.5 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.363803280612, -0.242535682106, -0.242535803380, -0.291514084595, -0.193009454736, -0.192009503491]], },
  { name: 'drag_nb2_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 6 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all drag 3 3 3 0.5 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.363646578979, -0.242592619841, -0.242713795231, 0.000987318121379, 0.00199154259760, 0.00299154048511]], },
  { name: 'drag_nb_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 8 4 6
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all drag 0.0 0.0 0.0 0.5 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.166666611109, -0.333333555563, 0.333333138882, 0.135007391877, -0.266014917766, 0.271014750251]], },
  { name: 'drag_nb_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 8 4 6
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all drag 0.0 0.0 0.0 0.5 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.166611071677, -0.333555532676, 0.333138796023, 0.00100581014190, 0.00198837390410, 0.00301161883079]], },
  { name: 'drag_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all drag 0.0 0.0 0.0 0.5 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.288732888814, 0.288675130743, 0.288617372671, 0.00100000100685, 0.00199999999993, 0.00299999899302]], },
  { name: 'efc_atomvar_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
variable e atom 0.5
fix 2 all efield 0.0 0.0 v_e
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, 1.00000000000, 0.00100000000000, 0.00200000000000, 0.807044485273]], },
  { name: 'efc_atomvar_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
variable e atom 0.5
fix 2 all efield 0.0 0.0 v_e
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, 23.0605490000, 0.00100000000000, 0.00200000000000, 0.00380404447508]], },
  { name: 'efc_const_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.0 0.0 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, 1.00000000000, 0.00100000000000, 0.00200000000000, 0.807044485273]], },
  { name: 'efc_const_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.0 0.0 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, 23.0605490000, 0.00100000000000, 0.00200000000000, 0.00380404447508]], },
  { name: 'efc_dip_metal', script: `units metal
atom_style hybrid sphere dipole
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
set atom 1 diameter 1.0 density 1.0
set atom 1 charge 2.0
set atom 1 dipole 1.0 0.5 0.2
mass 1 12.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve/sphere update dipole
fix 2 all efield 0.3 -0.2 0.5
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.600000000000, -0.400000000000, 1.00000000000, 11.0574053313, -7.36893688750, 18.4303422188]], pe: -6.350309905, },
  { name: 'efc_dip_real', script: `units real
atom_style hybrid sphere dipole
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
set atom 1 diameter 1.0 density 1.0
set atom 1 charge 2.0
set atom 1 dipole 1.0 0.5 0.2
mass 1 12.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve/sphere update dipole
fix 2 all efield 0.3 -0.2 0.5
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[13.8363294000, -9.22421960000, 23.0605490000, 0.0120564051911, -0.00537093679407, 0.0214273419852]], pe: -146.506137682, },
  { name: 'efc_energy_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.3 -0.2 0.5
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.600000000000, -0.400000000000, 1.00000000000, 0.483426691164, -0.319617794109, 0.807044485273]], pe: -6.00061387381, },
  { name: 'efc_energy_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.3 -0.2 0.5
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[13.8363294000, -9.22421960000, 23.0605490000, 0.00148242668505, 0.00167838220997, 0.00380404447508]], pe: -138.441955235, },
  { name: 'efc_eqvar_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
variable e equal 0.5
fix 2 all efield 0.0 0.0 v_e
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, 1.00000000000, 0.00100000000000, 0.00200000000000, 0.807044485273]], },
  { name: 'efc_eqvar_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
variable e equal 0.5
fix 2 all efield 0.0 0.0 v_e
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, 23.0605490000, 0.00100000000000, 0.00200000000000, 0.00380404447508]], },
  { name: 'efc_mix_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.3 -0.2 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.600000000000, -0.400000000000, 1.00000000000, 0.483426691164, -0.319617794109, 0.807044485273]], },
  { name: 'efc_mix_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.3 -0.2 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[13.8363294000, -9.22421960000, 23.0605490000, 0.00148242668505, 0.00167838220997, 0.00380404447508]], },
  { name: 'grav_chute_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all gravity 1.0 chute 30
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[6.00000000000, 0.00000000000, -10.3923048454, 4.82526691164, 0.00200000000000, -8.35287540023]], },
  { name: 'grav_chute_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all gravity 1.0 chute 30
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[6.00000000000, 0.00000000000, -10.3923048454, 0.00120919999999, 0.00200000000000, 0.00263765497108]], },
  { name: 'grav_energy_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all gravity 1.0 vector 0 0 -1
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, -12.0000000000, 0.00100000000000, 0.00200000000000, -9.64553382327]], pe: 59.9421447971, },
  { name: 'grav_energy_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all gravity 1.0 vector 0 0 -1
fix_modify 2 energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, -12.0000000000, 0.00100000000000, 0.00200000000000, 0.00258160000003]], pe: 60.0334896, },
  { name: 'grav_sph_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all gravity 1.0 spherical 0 150
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[6.00000000000, 0.00000000000, -10.3923048454, 4.82526691164, 0.00200000000000, -8.35287540023]], },
  { name: 'grav_sph_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all gravity 1.0 spherical 0 150
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[6.00000000000, 0.00000000000, -10.3923048454, 0.00120919999999, 0.00200000000000, 0.00263765497108]], },
  { name: 'grav_vec_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all gravity 1.0 vector 0 0 -1
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, -12.0000000000, 0.00100000000000, 0.00200000000000, -9.64553382327]], },
  { name: 'grav_vec_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all gravity 1.0 vector 0 0 -1
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, -12.0000000000, 0.00100000000000, 0.00200000000000, 0.00258160000003]], },
  { name: 'indent_plane_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all indent 1.0 plane z 5.5 lo units box
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, 0.249896505153, 0.00100000000000, 0.00200000000000, 0.203969514088]], },
  { name: 'indent_plane_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all indent 1.0 plane z 5.5 lo units box
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, 0.247004667836, 0.00100000000000, 0.00200000000000, 0.00300866444804]], },
  { name: 'indent_sph_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all indent 1.0 sphere 5 5 5.5 1.0 units box
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[4.99707559417e-7, 9.99415118833e-7, -0.249902503939, 0.00100020089355, 0.00200040178711, -0.197971925733]], },
  { name: 'indent_sph_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all indent 1.0 sphere 5 5 5.5 1.0 units box
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.000509043815257, 0.00101808763051, -0.252996994765, 0.00100000887433, 0.00200001774866, 0.00299123108573]], },
  { name: 'lep_dip_metal', script: `units metal
atom_style hybrid sphere dipole
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
set atom 1 diameter 1.0 density 1.0
set atom 1 charge 2.0
set atom 1 dipole 1.0 0.5 0.2
mass 1 12.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve/sphere update dipole
fix ef all efield/lepton "-0.05*x^2 - 0.3*y*z" step 1e-4
fix_modify ef energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[1.09227305584, 3.12071177235, 3.19616897338, 20.1998829185, 56.9490454923, 58.4745137253]], pe: -19.555300551, },
  { name: 'lep_dip_real', script: `units real
atom_style hybrid sphere dipole
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
set atom 1 diameter 1.0 density 1.0
set atom 1 charge 2.0
set atom 1 dipole 1.0 0.5 0.2
mass 1 12.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve/sphere update dipole
fix ef all efield/lepton "-0.05*x^2 - 0.3*y*z" step 1e-4
fix_modify ef energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[25.1930238260, 72.0067942034, 73.7330561965, 0.0212007235550, 0.0589636127868, 0.0614825583279]], pe: -451.345896648, },
  { name: 'lep_quad_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix ef all efield/lepton "-0.05*x^2 - 0.3*y*z"
fix_modify ef energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[1.00008060445, 3.00072544004, 3.00072484004, 0.805076890054, 2.41442509885, 2.41542485764]], pe: -17.5076553152, },
  { name: 'lep_quad_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix ef all efield/lepton "-0.05*x^2 - 0.3*y*z"
fix_modify ef energy yes
thermo_style custom step pe
thermo_modify format float %.12g
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[23.0670152805, 69.2398435245, 69.2260071951, 0.00180415720390, 0.00441314798465, 0.00541290677131]], pe: -404.104913616, },
  { name: 'lepton_efield_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield/lepton "-(0.4*x + 0.2*y - 0.3*z)"
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.800000000000, 0.400000000000, -0.600000000000, 0.644235588218, 0.323617794109, -0.479426691164]], },
  { name: 'lepton_efield_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield/lepton "-(0.4*x + 0.2*y - 0.3*z)"
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[18.4484392000, 9.22421960000, -13.8363294000, 0.00164323558006, 0.00232161779003, 0.00251757331495]], },
  { name: 'lineforce_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.3 0.2 0.5
fix 3 all lineforce 1 1 1
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.666666666667, 0.666666666667, 0.666666666667, 0.537029656849, 0.538029656849, 0.539029656849]], },
  { name: 'lineforce_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.3 0.2 0.5
fix 3 all lineforce 1 1 1
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[15.3736993333, 15.3736993333, 15.3736993333, 0.00153602965005, 0.00253602965005, 0.00353602965005]], },
  { name: 'planeforce_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.3 0.2 0.5
fix 3 all planeforce 0 0 1
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.600000000000, 0.400000000000, 0.00000000000, 0.483426691164, 0.323617794109, 0.00300000000000]], },
  { name: 'planeforce_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all efield 0.3 0.2 0.5
fix 3 all planeforce 0 0 1
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[13.8363294000, 9.22421960000, 0.00000000000, 0.00148242668505, 0.00232161779003, 0.00300000000000]], },
  { name: 'rigidgrav_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6 5 5
mass 1 12.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 2 all gravity 1.0 vector 0 0 -1
fix 1 all rigid group 1 all gravity 2
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, -12.0000000000, 0.00100000000000, 0.00200000000000, -19.2940676465], [0.00000000000, 0.00000000000, -12.0000000000, 0.00100000000000, 0.00200000000000, -19.2940676465]], },
  { name: 'rigidgrav_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
create_atoms 1 single 6 5 5
mass 1 12.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 2 all gravity 1.0 vector 0 0 -1
fix 1 all rigid group 1 all gravity 2
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[0.00000000000, 0.00000000000, -12.0000000000, 0.00100000000000, 0.00200000000000, 0.00216320000005], [0.00000000000, 0.00000000000, -12.0000000000, 0.00100000000000, 0.00200000000000, 0.00216320000005]], },
  { name: 'visc_metal', script: `units metal
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all viscous 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.000399494439341, -0.000798988878682, -0.00119848331802, 0.000638383228257, 0.00127676645651, 0.00191514968477]], },
  { name: 'visc_real', script: `units real
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 5 5 5
mass 1 12.0
set atom 1 charge 2.0
velocity all set 0.001 0.002 0.003
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
fix 2 all viscous 0.5
run 1
write_dump all custom OUT.dump id fx fy fz vx vy vz modify format float %.12g sort id`, atoms: [[-0.000499995641667, -0.000999991283333, -0.00149998692500, 0.000999982566743, 0.00199996513349, 0.00299994770023]], },
];

const runCase = async (script: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  await session.execute(script);
  const dump = files.get('OUT.dump') ?? '';
  const L = dump.trim().split('\n');
  const i = L.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const atoms = L.slice(i + 1).map((l) => l.trim().split(/\s+/).slice(1).map(Number));
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { atoms, pe: thermo.length ? Number(thermo[thermo.length - 1].pe) : undefined };
};

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.abs(b) + 1e-12;

describe('force units of the force-style fixes (native LAMMPS, units real and metal)', () => {
  for (const c of CASES) {
    it(`${c.name} matches native per-atom forces and velocities`, async () => {
      const got = await runCase(c.script);
      expect(got.atoms.length).toBe(c.atoms.length);
      c.atoms.forEach((want, a) => {
        want.forEach((w, k) => {
          expect(close(got.atoms[a][k], w), `atom ${a + 1} column ${k}: engine ${got.atoms[a][k]} native ${w}`).toBe(true);
        });
      });
      if (c.pe !== undefined) {
        expect(got.pe).toBeDefined();
        expect(Math.abs((got.pe as number) - c.pe) <= 1e-8 * Math.abs(c.pe) + 1e-10, `pe engine ${got.pe} native ${c.pe}`).toBe(true);
      }
    });
  }
});
