import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * velocity, fix_modify, compute_modify and min_modify as native LAMMPS reads them (src/velocity.cpp, src/modify.cpp
 * modify_fix and modify_compute, src/fix.cpp Fix::modify_params and the fixes' modify_param, src/compute.cpp
 * Compute::modify_params, src/min.cpp Min::modify_params, src/fix_wall.cpp, at b332f28). Measured with native LAMMPS
 * (/home/roy/.local/bin/lmp, b332f28): each row is the input below with its lines added, null when it runs, else the
 * start of native's error message. A few inputs native runs into undefined behaviour with are refused by the engine.
 */

type Row = { dim?: string; pre?: string; mass?: string; post?: string; tail?: string };

const input = (r: Row): string => `units lj
${r.dim ?? ''}atom_style atomic
${r.pre ?? ''}lattice ${r.dim ? 'hex 0.8' : 'fcc 0.8442'}
region box block 0 2 0 2 ${r.dim ? '-0.5 0.5' : '0 2'}
create_box 2 box
create_atoms 1 box
${r.mass ?? 'mass * 1.0'}
set type 1 type/ratio 2 0.5 12345
group g1 type 1
group g2 type 2
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
compute myT all temp
compute myP all pressure myT
compute myTp all temp/partial 1 1 0
variable ve equal 0.5
variable va atom x*0.1
variable vs string abc
fix 1 all nve
${r.post ?? ''}
${r.tail ?? 'run 0'}
`;

const label = (r: Row): string => Object.entries(r).map(([k, v]) => `${k}: ${v.replace(/\n/g, ' / ')}`).join(' | ');

const errorOf = async (r: Row): Promise<string | null> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  try { await s.execute(input(r)); } catch (e) { return (e as Error).message; }
  return null;
};

const ROWS: [Row, string | null][] = [
  [{ post: 'velocity all create 1.0 87287' }, null],
  [{ post: 'velocity all create 1.0 87287 dist gaussian' }, null],
  [{ post: 'velocity all create 1.0 87287 dist bogus' }, 'Unknown velocity dist argument: bogus'],
  [{ post: 'velocity all create 1.0 87287 sum yes' }, null],
  [{ post: 'velocity all create 1.0 87287 sum maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'velocity all create 1.0 87287 mom no rot yes' }, null],
  [{ post: 'velocity all create 1.0 87287 temp myT' }, null],
  [{ post: 'velocity all create 1.0 87287 temp nope' }, 'Could not find velocity temperature compute ID: nope'],
  [{ post: 'velocity all create 1.0 87287 temp myP' }, 'Velocity temperature compute myP does not compute temperature'],
  [{ post: 'velocity all create 1.0 87287 bias yes' }, 'Cannot use velocity bias command without temp keyword'],
  [{ post: 'velocity all create 1.0 87287 bias yes temp myT' }, 'Velocity temperature ID does calculate a velocity bias'],
  [{ post: 'velocity all create 1.0 87287 loop local' }, null],
  [{ post: 'velocity all create 1.0 87287 loop bogus' }, 'Unknown velocity loop argument: bogus'],
  [{ post: 'velocity all create 1.0 87287 rigid nope' }, 'Fix ID nope for velocity does not exist'],
  [{ post: 'velocity all create 1.0 87287 units bogus' }, 'Unknown velocity units argument: bogus'],
  [{ post: 'velocity all create 1.0 87287 bogus 1' }, 'Unknown velocity keyword: bogus'],
  [{ post: 'velocity all create 1.0 0' }, 'Illegal velocity create seed argument: 0'],
  [{ post: 'velocity all create 1.0' }, 'Illegal velocity command: missing argument(s)'],
  [{ post: 'velocity all create 1.0 87287 dist' }, 'Illegal velocity dist command: missing argument(s)'],
  [{ post: 'velocity all set 1 0 0' }, null],
  [{ post: 'velocity all set 1 0 0 sum yes' }, null],
  [{ post: 'velocity all set NULL NULL 0.5 units box' }, null],
  [{ post: 'velocity all set v_a 0 0' }, 'Variable name for velocity set does not exist'],
  [{ post: 'velocity all set 1 0' }, 'Illegal velocity command: missing argument(s)'],
  [{ post: 'velocity all scale 1.5' }, 'Attempting to rescale a 0.0 temperature'],
  [{ post: 'velocity all scale -1' }, 'Attempting to rescale a 0.0 temperature'],
  [{ post: 'velocity all ramp vx 0 1 z 0 1' }, null],
  [{ post: 'velocity all ramp vx 0 1 z 0 1 sum yes' }, null],
  [{ post: 'velocity all ramp vq 0 1 z 0 1' }, 'Illegal velocity ramp velocity direction vq'],
  [{ post: 'velocity all ramp vx 0 1 q 0 1' }, 'Illegal velocity ramp direction q'],
  [{ post: 'velocity all ramp vx 0 1' }, 'Illegal velocity command: missing argument(s)'],
  [{ post: 'velocity all zero linear' }, null],
  [{ post: 'velocity all zero angular' }, null],
  [{ post: 'velocity all bogus 1' }, 'Unknown velocity command style: bogus'],
  [{ post: 'velocity nogroup create 1.0 1' }, 'Could not find velocity group ID nogroup'],
  [{ post: 'velocity all' }, 'Illegal velocity command: missing argument(s)'],
  [{ post: 'velocity all create 1.0 87287 mom maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'velocity all create 1.0 87287 loop all' }, null],
  [{ post: 'velocity all create 1.0 87287 loop geom' }, null],
  [{ post: 'velocity all zero linear rigid nope' }, 'Fix ID nope for velocity does not exist'],
  [{ post: 'velocity all create 1.0 87287 units lattice' }, null],
  [{ post: 'velocity all scale 1.0 temp myT' }, 'Attempting to rescale a 0.0 temperature'],
  [{ post: 'min_modify dmax 0.1', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify dmax -1', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify line backtrack', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify line quadratic', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify line forcezero', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify line bogus', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal min_modify command'],
  [{ post: 'min_modify norm two', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify norm max', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify norm inf', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify norm bogus', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal min_modify command'],
  [{ post: 'min_modify alpha_damp 0.1', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal fix_modify command'],
  [{ post: 'min_modify discrete_factor 10', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal fix_modify command'],
  [{ post: 'min_modify integrator eulerimplicit', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify integrator verlet', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify integrator bogus', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal min_modify command'],
  [{ post: 'min_modify tmax 1.0', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify tmin 0.1', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify delaystep 10', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify dtgrow 1.1', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify dtshrink 0.5', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify alpha0 0.2', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify alphashrink 0.9', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify halfstepback no', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify halfstepback maybe', tail: 'minimize 1e-4 1e-6 5 50' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'min_modify initialdelay yes', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify vdfmax 3', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify abcfire yes', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_modify fire/abc yes', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal fix_modify command'],
  [{ post: 'min_modify bogus 1', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal fix_modify command'],
  [{ post: 'min_modify dmax', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal min_modify command'],
  [{ post: 'min_modify line', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal min_modify command'],
  [{ post: 'min_modify dmax 0.1 vmax 0.2', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal fix_modify command'],
  [{ post: 'min_modify dmax 0.1 0.2', tail: 'minimize 1e-4 1e-6 5 50' }, 'Illegal fix_modify command'],
  [{ post: 'min_style fire\nmin_modify line quadratic', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_style fire\nmin_modify norm max', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'min_style fire\nmin_modify integrator verlet', tail: 'minimize 1e-4 1e-6 5 50' }, null],
  [{ post: 'fix_modify 1 energy yes' }, 'Fix 1 nve does not support fix_modify energy command'],
  [{ post: 'fix_modify 1 energy maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'fix_modify 1 virial yes' }, 'Fix 1 nve does not support fix_modify virial command'],
  [{ post: 'fix_modify 1 virial no' }, null],
  [{ post: 'fix_modify 1 respa 1' }, 'Illegal fix_modify respa command'],
  [{ post: 'fix_modify 1 dynamic/dof yes' }, null],
  [{ post: 'fix_modify 1 bodyforces late' }, 'Fix 1 nve does not support fix_modify bodyforces command'],
  [{ post: 'fix_modify 1 temp myT' }, 'Fix 1 nve does not support fix_modify temp command'],
  [{ post: 'fix_modify 1 press myP' }, 'Fix 1 nve does not support fix_modify press command'],
  [{ post: 'fix_modify 1 temp nope' }, 'Fix 1 nve does not support fix_modify temp command'],
  [{ post: 'fix_modify 1 bogus 1' }, 'Fix 1 nve does not support fix_modify bogus command'],
  [{ post: 'fix_modify 1 energy' }, 'Illegal fix_modify energy command: missing argument(s)'],
  [{ post: 'fix_modify 1 colname 1 x' }, 'Fix 1 nve does not support fix_modify colname command'],
  [{ post: 'fix_modify 1 format float %g' }, 'Fix 1 nve does not support fix_modify format command'],
  [{ post: 'fix_modify 1 dynamic/dof maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'fix_modify nope energy yes' }, 'Could not find fix_modify ID nope'],
  [{ post: 'fix_modify 1' }, 'Illegal fix_modify command: missing argument(s)'],
  [{ post: 'fix t all temp/berendsen 1.0 1.0 0.5\nfix_modify t temp myT' }, null],
  [{ post: 'fix t all temp/berendsen 1.0 1.0 0.5\nfix_modify t temp nope' }, 'Could not find fix_modify temperature compute nope'],
  [{ post: 'fix t all temp/berendsen 1.0 1.0 0.5\nfix_modify t press myP' }, 'Fix t temp/berendsen does not support fix_modify press command'],
  [{ post: 'fix t all npt temp 1 1 0.5 iso 1 1 5\nfix_modify t press myP' }, null],
  [{ post: 'fix t all npt temp 1 1 0.5 iso 1 1 5\nfix_modify t temp myP' }, 'Fix_modify temperature ID myP does not compute temperature'],
  [{ post: 'compute_modify myT extra/dof 2' }, null],
  [{ post: 'compute_modify myT extra 2' }, 'Compute myT temp does not support compute_modify extra command'],
  [{ post: 'compute_modify myT dynamic/dof yes' }, null],
  [{ post: 'compute_modify myT dynamic yes' }, 'Compute myT temp does not support compute_modify dynamic command'],
  [{ post: 'compute_modify myT dynamic/dof maybe' }, 'Expected boolean parameter instead of \'maybe\' in input script or data file'],
  [{ post: 'compute_modify myT extra/dof abc' }, 'Expected floating point parameter instead of \'abc\' in input script or data file'],
  [{ post: 'compute_modify myT bogus 1' }, 'Compute myT temp does not support compute_modify bogus command'],
  [{ post: 'compute_modify myT extra/dof' }, 'Illegal compute_modify extra/dof command: missing argument(s)'],
  [{ post: 'compute_modify myT' }, 'Illegal compute_modify command: missing argument(s)'],
  [{ post: 'compute_modify nope extra/dof 2' }, 'Could not find compute_modify ID nope'],
  [{ post: 'compute_modify myP extra/dof 2' }, null],
  [{ post: 'velocity all set v_ve 0 0' }, null],
  [{ post: 'velocity all set v_va 0 0' }, null],
  [{ post: 'velocity all set v_vs 0 0' }, 'Variable for velocity set is invalid style'],
  [{ post: 'velocity all set 0 v_ve 0 units box' }, null],
  [{ post: 'velocity all set 1 NULL NULL' }, null],
  [{ post: 'velocity all set NULL 2 NULL units lattice' }, null],
  [{ post: 'velocity all set abc 0 0' }, 'Expected floating point parameter instead of \'abc\' in input script or data file'],
  [{ post: 'velocity all create 1.0 87287 mom on rot 1' }, null],
  [{ post: 'velocity all create 1.0 87287 sum on' }, null],
  [{ post: 'velocity all create 1.0 87287 dist gaussian loop local' }, null],
  [{ post: 'velocity all create 1.0 87287 temp myTp bias yes' }, null],
  [{ post: 'velocity all create 1.0 87287 bias no' }, null],
  [{ post: 'velocity all create 1.0 87287\nvelocity all scale 2.0' }, null],
  [{ post: 'velocity all create 1.0 87287\nvelocity all scale 2.0 temp myT' }, null],
  [{ post: 'velocity all create 1.0 87287\nvelocity all zero linear rigid 1' }, 'Velocity rigid used with non-rigid fix-ID 1'],
  [{ post: 'velocity all create 1.0 87287\nvelocity g1 zero angular' }, null],
  [{ post: 'velocity all ramp vy 0 1 x 0 2 units box' }, null],
  [{ post: 'velocity all ramp vz 0 1 x 0 2' }, null],
  [{ post: 'velocity all ramp vx 0 1 z 0 2 sum yes' }, null],
  [{ post: 'velocity all create 1.0 abc' }, 'Expected integer parameter instead of \'abc\' in input script or data file'],
  [{ post: 'velocity all create abc 1' }, 'Expected floating point parameter instead of \'abc\' in input script or data file'],
  [{ post: 'velocity all create 1.0 87287 temp myT bogus 1' }, 'Unknown velocity keyword: bogus'],
  [{ post: 'velocity all create 1.0 87287 units box' }, null],
  [{ post: 'velocity all zero linear bogus 1' }, 'Unknown velocity keyword: bogus'],
  [{ post: 'velocity all zero linear sum yes' }, null],
  [{ dim: 'dimension 2\n', post: 'velocity all set 0 0 1' }, 'Cannot set non-zero z velocity for 2d simulation'],
  [{ dim: 'dimension 2\n', post: 'velocity all set 0 0 0' }, null],
  [{ dim: 'dimension 2\n', post: 'velocity all set 0 0 v_ve' }, 'Cannot set variable z velocity for 2d simulation'],
  [{ dim: 'dimension 2\n', post: 'velocity all ramp vz 0 1 x 0 2' }, 'Cannot generate velocity ramp in z for a 2d system'],
  [{ dim: 'dimension 2\n', post: 'velocity all create 1.0 87287' }, null],
  [{ mass: 'mass 1 1.0', post: 'velocity all create 1.0 87287' }, 'Not all per-type masses are set. Type 2 is missing.'],
  [{ tail: 'velocity all create 1.0 87287' }, null],
  [{ post: 'fix a all addforce 0.0 0.0 0.1\nfix_modify a energy yes' }, null],
  [{ post: 'fix a all addforce 0.0 0.0 0.1\nfix_modify a virial yes' }, null],
  [{ post: 'fix a all addforce 0.0 0.0 0.1\nfix_modify a respa 1' }, null],
  [{ post: 'fix a all addforce 0.0 0.0 0.1\nfix_modify a respa 0' }, null],
  [{ post: 'fix a all addforce 0.0 0.0 0.1\nfix_modify a respa -1' }, 'Illegal fix_modify respa command'],
  [{ post: 'fix a all addforce 0.0 0.0 0.1\nfix_modify a respa abc' }, 'Expected integer parameter instead of \'abc\' in input script or data file'],
  [{ post: 'fix a all addforce 0.0 0.0 0.1\nfix_modify a energy no virial no' }, null],
  [{ post: 'fix a all setforce 0.0 NULL NULL\nfix_modify a respa 1' }, null],
  [{ post: 'fix a all setforce 0.0 NULL NULL\nfix_modify a energy yes' }, 'Fix a setforce does not support fix_modify energy command'],
  [{ post: 'fix w all wall/lj93 xlo EDGE 1.0 1.0 2.5\nfix_modify w respa 2' }, 'Cannot use xlo wall in periodic x dimension'],
  [{ post: 'fix w all wall/lj93 xlo EDGE 1.0 1.0 2.5\nfix_modify w virial yes' }, 'Cannot use xlo wall in periodic x dimension'],
  [{ post: 'fix w all wall/lj93 xlo EDGE 1.0 1.0 2.5\nfix_modify w energy yes' }, 'Cannot use xlo wall in periodic x dimension'],
  [{ post: 'fix t all langevin 1.0 1.0 0.5 48279\nfix_modify t temp myT' }, null],
  [{ post: 'fix t all langevin 1.0 1.0 0.5 48279\nfix_modify t temp nope' }, 'Could not find fix_modify temperature compute ID: nope'],
  [{ post: 'fix t all langevin 1.0 1.0 0.5 48279\nfix_modify t temp myP' }, 'Fix_modify temperature compute myP does not compute temperature'],
  [{ post: 'fix t all langevin 1.0 1.0 0.5 48279\nfix_modify t press myP' }, 'Fix t langevin does not support fix_modify press command'],
  [{ post: 'fix p all press/berendsen iso 1 1 10\nfix_modify p temp myT' }, null],
  [{ post: 'fix p all press/berendsen iso 1 1 10\nfix_modify p temp nope' }, 'Could not find fix_modify temperature compute ID: '],
  [{ post: 'fix p all press/berendsen iso 1 1 10\nfix_modify p temp myP' }, 'Fix_modify temperature compute myP does not compute temperature'],
  [{ post: 'fix p all press/berendsen iso 1 1 10\nfix_modify p press myP' }, null],
  [{ post: 'fix p all press/berendsen iso 1 1 10\nfix_modify p press nope' }, 'Could not find fix_modify pressure compute ID: nope'],
  [{ post: 'fix p all press/berendsen iso 1 1 10\nfix_modify p press myT' }, 'Fix_modify pressure compute myT does not compute pressure'],
  [{ post: 'unfix 1\nfix p all nph iso 1 1 5\nfix_modify p temp nope' }, 'Could not find fix_modify temperature ID nope'],
  [{ post: 'unfix 1\nfix p all nph iso 1 1 5\nfix_modify p press myP' }, 'Current temperature too close to zero, consider using ptemp keyword'],
  [{ post: 'unfix 1\nfix p all nph iso 1 1 5\nfix_modify p press nope' }, 'Could not find fix_modify pressure ID nope'],
  [{ post: 'unfix 1\nfix p all nph iso 1 1 5\nfix_modify p press myT' }, 'Fix_modify pressure ID myT does not compute pressure'],
  [{ post: 'unfix 1\nfix p all nph iso 1 1 5\nfix_modify p press' }, 'Illegal fix_modify press command: missing argument(s)'],
  [{ post: 'unfix 1\nfix p all nph iso 1 1 5\nfix_modify p temp' }, 'Illegal fix_modify command'],
  [{ post: 'unfix 1\nfix p all nvt temp 1 1 0.5\nfix_modify p press myP' }, 'Fix_modify press command without a barostat'],
  [{ post: 'fix t all temp/rescale 10 1.0 1.0 0.02 1.0\nfix_modify t temp nope' }, 'Could not find fix_modify temperature compute nope'],
  [{ post: 'fix t all temp/rescale 10 1.0 1.0 0.02 1.0\nfix_modify t temp myP' }, 'Fix_modify temperature compute myP does not compute temperature'],
  [{ post: 'fix t all temp/csvr 1.0 1.0 0.5 48279\nfix_modify t temp nope' }, 'Could not find fix_modify temperature ID nope'],
  [{ post: 'fix t all temp/csvr 1.0 1.0 0.5 48279\nfix_modify t temp myP' }, 'Fix_modify temperature ID myP does not compute temperature'],
  [{ post: 'fix t all temp/csvr 1.0 1.0 0.5 48279\nfix_modify t temp myT' }, null],
  [{ post: 'fix t all temp/berendsen 1.0 1.0 0.5\nfix_modify t temp myT dynamic/dof yes energy no' }, null],
  [{ post: 'compute_modify myT extra/dof 0' }, null],
  [{ post: 'compute_modify myT extra/dof -1' }, null],
  [{ post: 'compute_modify myT dynamic/dof on' }, null],
  [{ post: 'compute_modify myT extra/dof 2 dynamic/dof yes' }, null],
  [{ post: 'min_style spin\nmin_modify alpha_damp 1.0' }, null],
  [{ post: 'min_style spin\nmin_modify discrete_factor 10.0' }, null],
  [{ post: 'min_style spin/cg\nmin_modify discrete_factor 10.0' }, null],
  [{ post: 'min_style spin/cg\nmin_modify alpha_damp 1.0' }, 'Illegal fix_modify command'],
  [{ post: 'min_style spin/lbfgs\nmin_modify discrete_factor 10.0' }, null],
  [{ post: 'min_style fire\nmin_modify alpha_damp 1.0' }, 'Illegal fix_modify command'],
  [{ post: 'min_style spin\nmin_modify dmax 0.1 alpha_damp' }, 'Illegal min_modify command'],
  [{ post: 'min_style fire\nmin_modify halfstepback on' }, null],
  [{ post: 'min_style fire\nmin_modify abcfire 1' }, null],
  [{ post: 'min_style fire\nmin_modify vdfmax 2.5' }, 'Expected integer parameter instead of \'2.5\' in input script or data file'],
  [{ post: 'min_style fire\nmin_modify integrator leapfrog' }, null],
  [{ post: 'min_style fire\nmin_modify dtgrow abc' }, 'Expected floating point parameter instead of \'abc\' in input script or data file'],
  [{ post: 'min_style fire\nmin_modify line spin_none' }, null],
  [{ post: 'min_style fire\nmin_modify tmax 2 tmin 0.1' }, null],
];

/**
 * Inputs native runs into undefined behaviour with (NaN velocities, an unset line search that segfaults, a stale
 * argument in a message), refused or named by the engine: [row, engine message, native result].
 */
const REFUSED: [Row, string, string][] = [
  [{ post: 'velocity all create -1.0 87287' }, 'velocity create with a negative temperature is not supported by the browser engine', 'OK'],
  [{ post: 'velocity all zero bogus' }, 'Unknown velocity zero keyword bogus', 'ERROR: Unknown velocity zero keyword 1'],
  [{ post: 'min_modify line spin_cubic', tail: 'minimize 1e-4 1e-6 5 50' }, 'min_modify line spin_cubic is only for the spin minimizers', 'Segmentation fault'],
];

describe('velocity and the modify commands as native LAMMPS', () => {
  it.each(ROWS.map(([r, e]) => [label(r), r, e] as const))('%s', async (_name, r, expected) => {
    const got = await errorOf(r);
    if (expected === null) expect(got).toBeNull();
    else expect(got).toContain(expected);
  });
});

describe('inputs native runs into undefined behaviour with', () => {
  it.each(REFUSED.map(([r, e]) => [label(r), r, e] as const))('%s', async (_name, r, expected) => {
    expect(await errorOf(r)).toContain(expected);
  });
});
