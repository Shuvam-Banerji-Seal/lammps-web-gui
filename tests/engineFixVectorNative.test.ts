import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix vector as native LAMMPS (src/fix_vector.cpp at b332f28): its arguments and their messages, the first sample on
 * the first multiple of Nevery at or after the step the fix was defined on (none after a minimization moved past it),
 * storage sized from each run's end step (a run with pre no keeps the old size and overwrites the oldest rows), nmax,
 * and the checks of its readers: a variable reads it only on multiples of Nevery during a run (src/variable.cpp), and
 * thermo_style checks an index against the rows stored when the command is read (src/thermo.cpp parse_fields), where
 * a fix whose rows vary (ave/chunk) reads 0 past its end. Measured with native LAMMPS (/home/roy/.local/bin/lmp,
 * b332f28): each row is the input below with its lines added; the expected error is the first line of native's message
 * (null when the input runs) and the values are the lines print wrote that start with VAL, joined by spaces.
 */

const BASE = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
variable t equal 2*step+1
variable u equal step
thermo_modify format float %.15g
`;

const outcome = async (post: string): Promise<{ error: string | null; vals: string }> => {
  const vals: string[] = [];
  const s = new Session({
    emit: (ev: { kind: string; text?: string }) => { if (ev.kind === 'log' && ev.text?.startsWith('VAL ')) vals.push(ev.text); },
    writeFile: () => {},
  } as never);
  let error: string | null = null;
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(`${BASE}${post}\n`); } catch (e) { error = (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return { error, vals: vals.join(' ') };
};

const ROWS: [string, string | null, string][] = [
  ["fix 3 all vector 5 v_t\nvariable s equal sum(f_3)\nthermo_style custom step v_s\nthermo 5\nrun 10\nprint \"VAL $(v_s) $(f_3[1]) $(f_3[3])\"", null, "VAL 33 1 21"],
  ["fix 3 all vector 5 v_t\nthermo_style custom step f_3[1]\nrun 10", "Thermo custom fix 3 vector is accessed out-of-range", ""],
  ["fix 3 all vector 5 v_t\nrun 10\nthermo_style custom step f_3[1]\nthermo 5\nrun 10\nprint \"VAL $(sum(f_3)) $(f_3[5])\"", null, "VAL 105 41"],
  ["fix 3 all vector 1 v_t nmax 2\nrun 4\nprint \"VAL $(f_3[1]) $(f_3[2]) $(sum(f_3))\"", null, "VAL 7 9 16"],
  ["fix 3 all vector 5 v_t\nvariable e equal f_3[1]\nthermo_style custom step v_e\nthermo 1\nrun 2", "Variable e: Fix in variable not computed at a compatible time", ""],
  ["fix 3 all vector 5 v_t\nrun 2\nprint \"VAL $(f_3[1]) $(sum(f_3))\"", null, "VAL 1 1"],
  ["fix 3 all vector 5 v_t\nrun 10\nrun 10 pre no\nprint \"VAL $(sum(f_3)) $(f_3[1]) $(f_3[2]) $(f_3[3])\"", null, "VAL 93 21 31 41"],
  ["fix 3 all vector 5 v_t\nrun 20 every 10 NULL\nprint \"VAL $(sum(f_3)) $(f_3[1]) $(f_3[2]) $(f_3[3])\"", null, "VAL 105 1 11 21"],
  ["fix 3 all vector 5 v_t\nrun 10 start 0 stop 20\nrun 10 pre no\nprint \"VAL $(sum(f_3)) $(f_3[5])\"", null, "VAL 105 41"],
  ["run 3\nfix 3 all vector 5 v_t\nrun 10\nprint \"VAL $(f_3[1]) $(f_3[2]) $(sum(f_3))\"", null, "VAL 11 21 32"],
  ["fix 4 all vector 5 v_t v_u\nrun 10\nprint \"VAL $(f_4[2][1]) $(f_4[3][2]) $(sum(f_4[2]))\"", null, "VAL 11 10 15"],
  ["fix 3 all vector 1 v_t\nminimize 0 0 5 10\nprint \"VAL $(sum(f_3)) $(step)\"\nrun 3\nprint \"VAL $(sum(f_3)) $(step)\"", null, "VAL 0 5 VAL 0 8"],
  ["fix 3 all vector 5 v_t\nthermo_style custom step f_3\nrun 0", "Thermo custom fix 3 does not compute a scalar", ""],
  ["fix 3 all vector 0 v_t", "Invalid fix vector every argument: 0", ""],
  ["fix 3 all vector 2", "Illegal fix vector command: missing argument(s)", ""],
  ["fix 3 all vector 2 v_t nmax 0", "Invalid nmax value", ""],
  ["fix 3 all vector 2 v_t nmax 2 v_t", "Unknown fix vector keyword: v_t", ""],
  ["fix 3 all vector 2 v_missing", "Variable name missing for fix vector does not exist", ""],
  ["fix 3 all vector 2 c_missing", "Compute ID missing for fix vector does not exist", ""],
  ["fix 3 all vector 2 f_missing", "Fix ID missing for fix vector does not exist", ""],
  ["compute ke all ke\nfix 3 all vector 2 c_ke v_t", "Fix vector cannot set output array intensive/extensive from these inputs", ""],
  ["compute ke all ke\nfix 3 all vector 2 c_ke[1]", "Fix vector compute ke does not calculate a vector", ""],
  ["fix 3 all vector 2 c_thermo_temp[7]", "Fix vector compute thermo_temp vector is accessed out-of-range", ""],
  ["fix 3 all vector 2 c_thermo_press\nrun 2\nprint \"VAL $(f_3[2]:%.10g) $(press:%.10g)\"", null, "VAL -5.340223545 -5.340223545"],
  ["fix 3 all vector 2 c_thermo_temp[1][2]", "Invalid fix vector argument: c_thermo_temp[1][2]", ""],
  ["fix 3 all vector 2 v_t foo", "Unknown fix vector keyword: foo", ""],
  ["fix 3 all vector abc v_t", "Expected integer parameter instead of 'abc' in input script or data file", ""],
  ["fix 3 all vector 2 v_t[1]", "Fix vector variable t is not vector-style variable", ""],
  ["variable a atom x\nfix 3 all vector 2 v_a", "Fix vector variable a is not equal-style variable", ""],
  ["fix av all ave/time 3 1 3 v_t\nfix 3 all vector 2 f_av", "Fix for fix av vector not computed at compatible time", ""],
  ["fix av all ave/time 2 1 2 v_t\nfix 3 all vector 4 f_av\nrun 8\nprint \"VAL $(f_3[1]) $(f_3[2]) $(f_3[3])\"", null, "VAL 1 9 17"],
  ["fix 3 all vector 2 f_1", "Fix vector fix 1 does not calculate a scalar", ""],
  ["fix 3 all vector 2 f_1[1]", "Fix vector fix 1 does not calculate a vector", ""],
  ["fix 3 all vector 2 v_t nmax", "Illegal fix vector nmax command: missing argument(s)", ""],
  ["fix 3 all vector 2 v_t nmax 2.5", "Expected integer parameter instead of '2.5' in input script or data file", ""],
  ["fix 3 all vector 2 v_t nmax 3000000000", "Invalid nmax value", ""],
  ["fix 3 all vector 2 v_t\nreset_timestep 5", "Cannot reset timestep with time-dependent fix vector defined", ""],
  ["fix 3 all vector 2 1.0", "Unknown fix vector keyword: 1.0", ""],
  ["fix 3 all vector 2 nmax 5\nrun 2", "Must set 'extarray' when setting 'array_flag' for fix vector.  Please contact the LAMMPS developers.", ""],
  ["fix 3 all vector -1 v_t", "Invalid fix vector every argument: -1", ""],
  ["fix 3 all vector 2 c_thermo_temp v_t\nrun 4\nprint \"VAL $(f_3[3][2]) $(f_3[1][1]:%.10g)\"", null, "VAL 9 1"],
  ["fix 3 all vector 2 v_t\nrun 4\nthermo_style custom step f_3[1][1]\nrun 0", "Thermo custom fix 3 does not compute an array", ""],
  ["fix 3 all vector 2 v_t\nrun 4\nprint \"VAL $(f_3[2][1])\"", "Fix '3' in variable formula does not compute the requested property. ", ""],
  ["fix 3 all vector 2 v_t\nrun 4\nprint \"VAL $(f_3[4])\"", "Variable formula fix vector is accessed out-of-range", ""],
  ["fix 3 all vector 2 v_t\nrun 4\nprint \"VAL $(f_3)\"", "Fix '3' in variable formula does not compute the requested property. ", ""],
  ["fix 4 all vector 2 v_t v_u nmax 2\nrun 6\nprint \"VAL $(f_4[1][1]) $(f_4[2][1]) $(f_4[1][2]) $(sum(f_4[1]))\"", null, "VAL 9 13 4 22"],
  ["fix 3 all vector 2 v_t\nrun 4\nthermo_style custom step f_3[2]\nrun 0\nprint \"VAL $(f_3[3])\"", null, "VAL 9"],
  ["fix 3 all vector 2 v_t\nrun 4\nthermo_style custom step f_3[4]\nrun 0", "Thermo custom fix 3 vector is accessed out-of-range", ""],
  ["fix 3 all vector 2 v_t\nrun 3\nthermo_style custom step f_3[1]\nrun 0", null, ""],
  ["fix 3 all vector 2 v_t\nrun 3\nvariable e equal f_3[1]\nfix p all print 1 \"VAL $(v_e)\"\nrun 1", "Variable e: Fix in variable not computed at a compatible time", ""],
  ["variable w vector [1,2,3]\nfix 3 all vector 2 v_w[2]\nrun 2\nprint \"VAL $(f_3[1]) $(f_3[2])\"", null, "VAL 2 2"],
  ["variable w vector [1,2,3]\nfix 3 all vector 2 v_w[5]\nrun 2\nprint \"VAL $(f_3[1]) $(f_3[2])\"", null, "VAL 0 0"],
  ["compute ke all ke\nfix 3 all vector 2 c_ke c_thermo_pe\nrun 2\nprint \"VAL $(f_3[1][1]:%.10g) $(f_3[2][2]:%.10g)\"", null, "VAL 160.5 -730.5232429"],
  ["fix 3 all vector 2 v_t\nunfix 3\nfix 3 all vector 3 v_u\nrun 3\nprint \"VAL $(f_3[1]) $(f_3[2])\"", null, "VAL 0 3"],
  ["fix 3 all vector 1 v_t\nrun 2\nvariable t delete\nrun 1", "Variable name for fix vector does not exist", ""],
  ["compute ch all chunk/atom bin/1d x lower 0.5\nfix 5 all ave/chunk 2 1 2 ch vx\nthermo_style custom step f_5[100][3]\nthermo 2\nrun 2\nprint \"VAL ok\"", null, "VAL ok"],
  ["compute ch all chunk/atom bin/1d x lower 0.5\nfix 5 all ave/chunk 2 1 2 ch vx\nthermo_style custom step f_5[2][9]\nthermo 2\nrun 2", "Thermo custom fix 5 array is accessed out-of-range", ""],
  ["compute ch all chunk/atom bin/1d x lower 0.5\nfix 5 all ave/chunk 2 1 2 ch vx\nthermo_style custom step f_5[0][1]\nthermo 2\nrun 2", "Thermo custom fix 5 array is accessed out-of-range", ""],
  ["fix 5 all ave/time 2 1 2 c_thermo_temp mode vector\nthermo_style custom step f_5[1]\nthermo 2\nrun 2\nprint \"VAL ok\"", null, "VAL ok"],
  ["fix 5 all ave/time 2 1 2 c_thermo_press[1] c_thermo_press[2]\nthermo_style custom step f_5[3]\nthermo 2\nrun 2", "Thermo custom fix 5 vector is accessed out-of-range", ""],
  ["fix 5 all ave/time 2 1 2 c_thermo_press mode vector\nthermo_style custom step f_5[7][1]\nthermo 2\nrun 2", "Thermo custom fix 5 does not compute an array", ""],
  ["fix 5 all ave/histo 2 1 2 0 10 5 vx mode vector\nthermo_style custom step f_5[6]\nthermo 2\nrun 2", "Thermo custom fix 5 vector is accessed out-of-range", ""],
  ["fix 3 all vector 2 v_t\nthermo_style custom step f_3[1]\nrun 0", "Thermo custom fix 3 vector is accessed out-of-range", ""],
  ["fix 3 all vector 2 v_t v_u\nthermo_style custom step f_3[1][1]\nrun 0", "Thermo custom fix 3 array is accessed out-of-range", ""],
  ["fix 3 all vector 2 v_t v_u\nrun 2\nthermo_style custom step f_3[2][2] f_3[2][3]\nrun 0", "Thermo custom fix 3 array is accessed out-of-range", ""],
];

describe('fix vector as native LAMMPS', () => {
  it.each(ROWS.map(([p, e, v]) => [p.replace(/\n/g, ' / '), p, e, v] as const))('%s', async (_name, p, error, vals) => {
    const got = await outcome(p);
    expect(got.error).toBe(error);
    expect(got.vals).toBe(vals);
  });
});
