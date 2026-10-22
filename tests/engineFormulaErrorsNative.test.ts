import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * Variable formulas as native LAMMPS evaluates them (src/variable.cpp Variable::evaluate, math_function,
 * group_function, special_function, feature_function, int_between_brackets and peratom2global, src/thermo.cpp
 * Thermo::evaluate_keyword at b332f28): the error texts, which errors name the variable being evaluated ("Variable a:
 * ..."), the order of the checks, and the printed values. Measured with native LAMMPS (/home/roy/.local/bin/lmp,
 * b332f28): each row is one of the inputs below with its lines added; the expected error is the first line of native's
 * message (null when the input runs), and the values are the lines print wrote that start with VAL (null: not
 * compared, where native's pressure sums differ from the engine's in the last digits).
 */

type Row = { b: 'a' | 'b' | 'x'; post: string; tail?: string };

/** a: an atomic system without an atom map; the row's lines, then run 0 (or its tail) */
const BASE_A = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box
mass * 1.0
group g1 type 1
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
compute myT all temp
compute myP all pressure myT
compute ke all ke/atom
fix 1 all nve
fix av all ave/time 1 1 1 c_myT
variable vv vector [1,2,3]
variable va atom x
variable vs string abc
`;

/** b: with an atom map, a custom property, per-atom computes and fix ave/time outputs; x: the row is the whole input */
const BASE_B = `units lj
atom_style atomic
atom_modify map yes
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box
mass * 1.0
group g1 type 1
region r1 block 0 1 0 1 0 1
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
fix pa all property/atom i_foo d2_bar 2
compute myT all temp
compute myP all pressure myT
compute ke all ke/atom
compute st all stress/atom NULL
fix 1 all nve
fix avs all ave/time 1 1 1 c_myT
fix av all ave/time 1 1 1 c_myT c_myP[1]
fix aa all ave/time 1 1 1 c_myP c_myP mode vector
variable vv vector [1,2,3]
variable v2 vector [1,2]
variable va atom x
variable vs string abc
variable one equal 1
thermo_style custom step temp pe press
thermo_modify format float %.15g
`;

const input = (r: Row): string => {
  if (r.b === 'x') return r.post;
  if (r.b === 'a') return `${BASE_A}${r.post}\n${r.tail ?? 'run 0'}\n`;
  return `${BASE_B}${r.post}\n`;
};

const label = (r: Row): string => `${r.b}: ${r.post.replace(/\n/g, ' / ')}${r.tail !== undefined ? ` | tail: ${r.tail}` : ''}`;

const outcome = async (r: Row): Promise<{ error: string | null; vals: string[] }> => {
  const vals: string[] = [];
  const s = new Session({
    emit: (ev: { kind: string; text?: string }) => { if (ev.kind === 'log' && ev.text?.startsWith('VAL ')) vals.push(ev.text); },
    writeFile: () => {},
  } as never);
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(input(r)); } catch (e) { return { error: (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''), vals }; }
  return { error: null, vals };
};

const ROWS: [Row, string | null, string[] | null][] = [
  [{ b: 'a', post: 'variable a equal "1/0"\nprint ${a}' }, 'Variable a: Divide by 0 in variable formula', []],
  [{ b: 'a', post: 'variable a equal "5%0"\nprint ${a}' }, 'Variable a: Modulo 0 in variable formula', []],
  [{ b: 'a', post: 'variable a equal "sqrt(-1)"\nprint ${a}' }, 'Variable a: Sqrt of negative value in variable formula', []],
  [{ b: 'a', post: 'variable a equal "ln(0)"\nprint ${a}' }, 'Variable a: Log of zero/negative value in variable formula', []],
  [{ b: 'a', post: 'variable a equal "log(-1)"\nprint ${a}' }, 'Variable a: Log of zero/negative value in variable formula', []],
  [{ b: 'a', post: 'variable a equal "asin(2)"\nprint ${a}' }, 'Variable a: Arcsin of invalid value in variable formula', []],
  [{ b: 'a', post: 'variable a equal "acos(-2)"\nprint ${a}' }, 'Variable a: Arccos of invalid value in variable formula', []],
  [{ b: 'a', post: 'variable a equal "(-8)^0.5"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "0^-1"\nprint ${a}' }, 'Variable a: Invalid power expression in variable formula', []],
  [{ b: 'a', post: 'variable a equal "nope(1)"\nprint ${a}' }, 'Variable a: Invalid math/group/special/feature function \'nope()\' in variable formula', []],
  [{ b: 'a', post: 'variable a equal "c_nope"\nprint ${a}' }, 'Variable a: Invalid compute ID \'nope\' in variable formula', []],
  [{ b: 'a', post: 'variable a equal "f_nope"\nprint ${a}' }, 'Variable a: Invalid fix ID \'nope\' in variable formula', []],
  [{ b: 'a', post: 'variable a equal "v_b"\nprint ${a}' }, 'Invalid variable reference v_b in variable formula', []],
  [{ b: 'a', post: 'variable a equal "v_vs"\nprint ${a}' }, 'Variable vs: Non-numeric variable value in variable formula', []],
  [{ b: 'a', post: 'variable a equal "v_va"\nprint ${a}' }, 'Variable va: Atom-style variable in equal-style variable formula', []],
  [{ b: 'a', post: 'variable a equal "v_vv"\nprint ${a}' }, 'Variable vv: Vector-style variable in equal-style variable formula', []],
  [{ b: 'a', post: 'variable a equal "v_vv[5]"\nprint ${a}' }, 'Variable vv: Invalid index into vector-style variable', []],
  [{ b: 'a', post: 'variable a equal "v_vv[0]"\nprint ${a}' }, 'Index between variable brackets must be positive', []],
  [{ b: 'a', post: 'variable a equal "c_myT[1]"\nprint ${a}' }, 'Variable a: Variable formula compute cannot be invoked before initialization by a run', []],
  [{ b: 'a', post: 'variable a equal "c_myP[7]"\nprint ${a}' }, 'Variable a: Variable formula compute cannot be invoked before initialization by a run', []],
  [{ b: 'a', post: 'variable a equal "c_myP[0]"\nprint ${a}' }, 'Index between variable brackets must be positive', []],
  [{ b: 'a', post: 'variable a equal "c_myT[1][1]"\nprint ${a}' }, 'Variable a: Compute \'myT\' in variable formula does not compute the requested property. ', []],
  [{ b: 'a', post: 'variable a equal "f_av[2]"\nprint ${a}' }, 'Variable a: Fix \'av\' in variable formula does not compute the requested property. ', []],
  [{ b: 'a', post: 'variable a equal "f_av"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "c_ke"\nprint ${a}' }, 'Variable a: Compute \'ke\' in variable formula does not compute the requested property. ', []],
  [{ b: 'a', post: 'variable a equal "1+"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "(1"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "1)"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "1 2"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "1 +* 2"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "@"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "1 ++ 2"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "abs()"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "abs(1,2)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "sqrt(1,2)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "pow(2)"\nprint ${a}' }, 'Variable a: Invalid math/group/special/feature function \'pow()\' in variable formula', []],
  [{ b: 'a', post: 'variable a equal "atan2(1)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "random(1,2)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "normal(1,2,3)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "ramp(1)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "stride(1,2)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "vdisplace(1)"\nprint ${a}' }, 'Variable a: Invalid vdisplace function in variable formula: must have 2 arguments', []],
  [{ b: 'a', post: 'variable a equal "swiggle(1,2,0)"\nprint ${a}' }, 'Variable a: Invalid swiggle(x,y,z) function in variable formula: z must be > 0', []],
  [{ b: 'a', post: 'variable a equal "cwiggle(1,2)"\nprint ${a}' }, 'Variable a: Invalid cwiggle function in variable formula: must have 3 arguments', []],
  [{ b: 'a', post: 'variable a equal "logfreq(1,2)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "stagger(1,2)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "count()"\nprint ${a}' }, 'Variable a: Group  in variable formula does not exist', []],
  [{ b: 'a', post: 'variable a equal "count(nope)"\nprint ${a}' }, 'Variable a: Group nope in variable formula does not exist', []],
  [{ b: 'a', post: 'variable a equal "mass(nope)"\nprint ${a}' }, 'Variable a: Group nope in variable formula does not exist', []],
  [{ b: 'a', post: 'variable a equal "xcm(g1)"\nprint ${a}' }, 'Variable a: Invalid xcm() function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "xcm(g1,q)"\nprint ${a}' }, 'Variable a: Invalid xcm() function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "bound(g1,xmin)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "bound(g1,bogus)"\nprint ${a}' }, 'Variable a: Invalid bound() function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "gmask(g1)"\nprint ${a}' }, 'Variable a: Gmask function in equal-style variable formula', []],
  [{ b: 'a', post: 'variable a equal "sum(c_myT)"\nprint ${a}' }, 'Variable formula compute cannot be invoked before initialization by a run', []],
  [{ b: 'a', post: 'variable a equal "min(v_vv)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "ave(v_vv)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "trap(v_vv)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "sort(v_vv)"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "is_defined(compute,myT)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "is_defined(bogus,x)"\nprint ${a}' }, 'Unknown category for info is_defined(): bogus', []],
  [{ b: 'a', post: 'variable a equal "is_active(bogus,x)"\nprint ${a}' }, 'Unknown category for info is_active(): bogus', []],
  [{ b: 'a', post: 'variable a equal "extract_setting(bogus)"\nprint ${a}' }, 'Variable a: Unknown setting bogus for extract_setting() function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "label2type(atom,C)"\nprint ${a}' }, 'Variable a: Cannot use label2type() function without a labelmap', []],
  [{ b: 'a', post: 'variable a equal "is_typelabel(atom,C)"\nprint ${a}' }, 'Variable a: Cannot use is_typelabel() function without a labelmap', []],
  [{ b: 'a', post: 'variable a equal "bogusword"\nprint ${a}' }, 'Variable a: Invalid thermo keyword \'bogusword\' in variable formula', []],
  [{ b: 'a', post: 'variable a equal "x[1]"\nprint ${a}' }, 'Indexed per-atom vector in variable formula without atom map', []],
  [{ b: 'a', post: 'variable a equal "x[999]"\nprint ${a}' }, 'Indexed per-atom vector in variable formula without atom map', []],
  [{ b: 'a', post: 'variable a equal "x[0]"\nprint ${a}' }, 'Index between variable brackets must be positive', []],
  [{ b: 'a', post: 'variable a equal "vx[1.5]"\nprint ${a}' }, 'Non digit character between brackets in variable', []],
  [{ b: 'a', post: 'variable a equal "x"\nprint ${a}' }, 'Atom vector in equal-style variable formula', []],
  [{ b: 'a', post: 'variable a equal "mass[1]"\nprint ${a}' }, 'Indexed per-atom vector in variable formula without atom map', []],
  [{ b: 'a', post: 'variable a equal "q[1]"\nprint ${a}' }, 'Indexed per-atom vector in variable formula without atom map', []],
  [{ b: 'a', post: 'variable a equal "step*2"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "time+"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a equal "v_q*2"\nprint ${a}' }, 'Invalid variable reference v_q in variable formula', []],
  [{ b: 'a', post: 'variable a equal "sum(v_va)"\nprint ${a}' }, 'Variable va: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'variable a equal "dist(1)"\nprint ${a}' }, 'Variable a: Invalid math/group/special/feature function \'dist()\' in variable formula', []],
  [{ b: 'a', post: 'variable a equal "grmask(g1)"\nprint ${a}' }, 'Variable a: Grmask function in equal-style variable formula', []],
  [{ b: 'a', post: 'variable a equal "next(vs)"\nprint ${a}' }, 'Variable vs: Invalid variable style in special function next', []],
  [{ b: 'a', post: 'variable a equal "is_file(nope)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "is_os(linux)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "is_timeout()"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "ternary(1,2,3)"\nprint ${a}' }, null, []],
  [{ b: 'a', post: 'variable a equal "ternary(1,2)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "floor(1.5,2)"\nprint ${a}' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'a', post: 'variable a equal "v_vv[1][1]"\nprint ${a}' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'variable a atom "x/0"\nvariable b equal sum(v_a)\nprint ${b}' }, 'Variable a: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'variable a atom "sqrt(-x)"\nvariable b equal sum(v_a)\nprint ${b}' }, 'Variable a: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'variable a atom "nope(x)"\nvariable b equal sum(v_a)\nprint ${b}' }, 'Variable a: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'variable a atom "c_nope"\nvariable b equal sum(v_a)\nprint ${b}' }, 'Variable a: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'variable a atom "v_b"\nvariable b equal sum(v_a)\nprint ${b}' }, 'Variable a: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'variable a atom "1+"\nvariable b equal sum(v_a)\nprint ${b}' }, 'Variable a: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'variable a atom "gmask(nope)"\nvariable b equal sum(v_a)\nprint ${b}' }, 'Variable a: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'variable a atom "mass[1]"\nvariable b equal sum(v_a)\nprint ${b}' }, 'Variable a: Mis-matched special function variable in variable formula', []],
  [{ b: 'a', post: 'print "$(1/0)"' }, 'Divide by 0 in variable formula', []],
  [{ b: 'a', post: 'if "1/0 > 0" then "print a"' }, 'Invalid Boolean syntax in if command', []],
  [{ b: 'a', post: 'print "$(nope(1))"' }, 'Invalid math/group/special/feature function \'nope()\' in variable formula', []],
  [{ b: 'a', post: 'if "nope(1) > 0" then "print a"' }, 'Invalid Boolean syntax in if command', []],
  [{ b: 'a', post: 'print "$(v_b)"' }, 'Invalid variable reference v_b in variable formula', []],
  [{ b: 'a', post: 'if "v_b > 0" then "print a"' }, 'If command boolean can only operate on numbers', []],
  [{ b: 'a', post: 'print "$(1+)"' }, 'Invalid syntax in variable formula', []],
  [{ b: 'a', post: 'if "1+ > 0" then "print a"' }, 'Invalid Boolean syntax in if command', []],
  [{ b: 'a', post: 'variable a equal v_a+1\nprint ${a}' }, 'Variable a: has a circular dependency', []],
  [{ b: 'a', post: 'variable a equal v_b+1\nvariable b equal v_a+1\nprint ${a}' }, 'Variable a: has a circular dependency', []],
  [{ b: 'a', post: 'variable b equal 1/0\nvariable a equal v_b+1\nprint ${a}' }, 'Variable b: Divide by 0 in variable formula', []],
  [{ b: 'a', post: 'variable a equal "c_myT"\nprint ${a}', tail: '' }, 'Variable a: Variable formula compute cannot be invoked before initialization by a run', []],
  [{ b: 'a', post: 'variable a equal "f_av"\nprint ${a}', tail: '' }, null, []],
  [{ b: 'a', post: 'variable a equal "pe"\nprint ${a}', tail: '' }, 'Energy was not tallied on needed timestep', []],
  [{ b: 'b', post: 'variable a equal "temp"\nprint "VAL ${a}"' }, 'Thermo keyword temp in variable requires thermo to use/init temperature', []],
  [{ b: 'b', post: 'variable a equal "pe"\nprint "VAL ${a}"' }, 'Energy was not tallied on needed timestep', []],
  [{ b: 'b', post: 'variable a equal "cpu"\nprint "VAL ${a}"' }, 'The variable thermo keyword cpu cannot be used between runs', []],
  [{ b: 'b', post: 'variable a equal "etail"\nprint "VAL ${a}"' }, 'Energy was not tallied on needed timestep for thermo keyword etail', []],
  [{ b: 'b', post: 'variable a equal "vol"\nprint "VAL ${a}"' }, null, ['VAL 37.9057095475006']],
  [{ b: 'b', post: 'variable a equal "ke"\nprint "VAL ${a}"' }, 'Thermo keyword ke in variable requires thermo to use/init temperature', []],
  [{ b: 'b', post: 'variable a equal "press"\nprint "VAL ${a}"' }, 'Thermo keyword press in variable requires thermo to use/init press', []],
  [{ b: 'b', post: 'variable a equal "pxx"\nprint "VAL ${a}"' }, 'Thermo keyword pxx in variable requires thermo to use/init press', []],
  [{ b: 'b', post: 'variable a equal "enthalpy"\nprint "VAL ${a}"' }, 'Energy was not tallied on needed timestep', []],
  [{ b: 'b', post: 'variable a equal "ecouple"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "atoms"\nprint "VAL ${a}"' }, null, ['VAL 32']],
  [{ b: 'b', post: 'run 0\nvariable a equal "temp"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable a equal "pe"\nprint "VAL ${a}"' }, null, ['VAL -6.77336805325293']],
  [{ b: 'b', post: 'run 0\nvariable a equal "ke"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable a equal "pxx"\nprint "VAL ${a}"' }, null, null],
  [{ b: 'b', post: 'run 0\nvariable a equal "enthalpy"\nprint "VAL ${a}"' }, null, ['VAL -14.1594344712648']],
  [{ b: 'b', post: 'run 0\nvariable a equal "etotal"\nprint "VAL ${a}"' }, null, ['VAL -6.77336805325293']],
  [{ b: 'b', post: 'run 0\nvariable a equal "cpu"\nprint "VAL ${a}"' }, 'The variable thermo keyword cpu cannot be used between runs', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "etail"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable a equal "evdwl"\nprint "VAL ${a}"' }, null, ['VAL -6.77336805325293']],
  [{ b: 'b', post: 'run 0\nvariable a equal "econserve"\nprint "VAL ${a}"' }, null, ['VAL -6.77336805325293']],
  [{ b: 'b', post: 'run 0\nreset_timestep 5\nvariable a equal "pe"\nprint "VAL ${a}"' }, 'Cannot reset timestep with time-dependent fix ave/time defined', []],
  [{ b: 'b', post: 'run 0\nthermo_style custom step pe\nvariable a equal "temp"\nprint "VAL ${a}"' }, 'Thermo keyword temp in variable requires thermo to use/init temperature', []],
  [{ b: 'b', post: 'run 0\nthermo_style custom step pe\nrun 0\nvariable a equal "temp"\nprint "VAL ${a}"' }, 'Thermo keyword temp in variable requires thermo to use/init temperature', []],
  [{ b: 'b', post: 'run 0\nthermo_style custom step temp\nrun 0\nvariable a equal "pe"\nprint "VAL ${a}"' }, 'Thermo keyword pe in variable requires thermo to use/init potential energy', []],
  [{ b: 'b', post: 'run 0\nthermo_style custom step pe\nthermo_modify temp myT\nvariable a equal "temp"\nprint "VAL ${a}"' }, 'Thermo style does not use temp', []],
  [{ b: 'b', post: 'run 0\nthermo_style custom step pe\nthermo_modify press myP\nvariable a equal "press"\nprint "VAL ${a}"' }, 'Thermo style does not use press', []],
  [{ b: 'b', post: 'run 0\nthermo_style custom step pe\nvariable a equal "press"\nprint "VAL ${a}"' }, 'Thermo keyword press in variable requires thermo to use/init press', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "c_myT"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable a equal "c_myP[7]"\nprint "VAL ${a}"' }, 'Variable a: Variable formula compute vector is accessed out-of-range', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "c_myP[1][1]"\nprint "VAL ${a}"' }, 'Variable a: Compute \'myP\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "C_ke[1]"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable a equal "C_ke[999]"\nprint "VAL ${a}"' }, 'Variable atom ID is too large', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "C_ke"\nprint "VAL ${a}"' }, 'Variable a: Compute \'ke\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "C_ke[1][1]"\nprint "VAL ${a}"' }, 'Variable a: Compute \'ke\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "C_st[1][7]"\nprint "VAL ${a}"' }, 'Variable a: Variable formula compute array is accessed out-of-range', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "C_st[1]"\nprint "VAL ${a}"' }, 'Variable a: Compute \'st\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "C_st[2][3]"\nprint "VAL ${a}"' }, null, ['VAL 7.38606641801183']],
  [{ b: 'b', post: 'run 0\nvariable a equal "c_ke"\nprint "VAL ${a}"' }, 'Variable a: Compute \'ke\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "c_ke[1]"\nprint "VAL ${a}"' }, 'Variable a: Compute \'ke\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_avs"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_avs[1]"\nprint "VAL ${a}"' }, 'Variable a: Fix \'avs\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_av[3]"\nprint "VAL ${a}"' }, 'Variable a: Variable formula fix vector is accessed out-of-range', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_av[2]"\nprint "VAL ${a}"' }, null, null],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_aa[7][1]"\nprint "VAL ${a}"' }, 'Variable a: Variable formula fix array is accessed out-of-range', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_aa[1][3]"\nprint "VAL ${a}"' }, 'Variable a: Variable formula fix array is accessed out-of-range', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_aa[2][2]"\nprint "VAL ${a}"' }, null, ['VAL -6.23531727008559']],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_aa"\nprint "VAL ${a}"' }, 'Variable a: Fix \'aa\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "F_avs[1]"\nprint "VAL ${a}"' }, 'Variable a: Fix \'avs\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "f_av[1][1]"\nprint "VAL ${a}"' }, 'Variable a: Fix \'av\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\ncompute t2 all temp\nvariable a equal "c_t2"\nprint "VAL ${a}"' }, 'Variable a: Variable formula compute cannot be invoked before initialization by a run', []],
  [{ b: 'b', post: 'write_data fmx2.data\nvariable a equal "c_myT"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'velocity all create 1.0 4928\nvariable a equal "c_myT"\nprint "VAL ${a}"' }, 'Variable a: Variable formula compute cannot be invoked before initialization by a run', []],
  [{ b: 'b', post: 'velocity all create 1.0 4928\nrun 0\nvariable a equal "c_myT"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'run 0\nvariable b atom "c_myT"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Compute \'myT\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "c_ke"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable b atom "c_st[7]"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Variable formula compute array is accessed out-of-range', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "c_st[2]"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, null, ['VAL 7.38606641801183']],
  [{ b: 'b', post: 'run 0\nvariable b atom "x/0"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Divide by 0 in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "sqrt(-x-1)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Sqrt of negative value in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "1/0"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Divide by 0 in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "v_vv"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable vv: Vector-style variable in atom-style variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "gmask(nope)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Group ID in variable formula does not exist', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "rmask(nope)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Region nope in variable formula does not exist', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "grmask(g1)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Invalid special function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "gmask(g1)*x"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable b atom "random(0,1,0)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Invalid math function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "normal(0,-1,5)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Invalid math function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "stride(1,0,1)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Invalid math function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "stagger(1,2)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Invalid math function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "logfreq(1,2,2)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Invalid math function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "i_foo"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable b atom "d2_bar"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Invalid custom atom property reference d2_bar in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "d2_bar[3]"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Invalid custom atom property reference d2_bar in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "d2_bar[1]"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable b atom "i_nope"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Invalid custom atom property reference i_nope in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "f_avs"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Fix \'avs\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "f_aa[1]"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Fix \'aa\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "v_b"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: has a circular dependency', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "mol"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable uses atom property \'mol\' that isn\'t allocated', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "q"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable uses atom property \'q\' that isn\'t allocated', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "radius"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable uses atom property \'radius\' that isn\'t allocated', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "x+v_one"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'run 0\nvariable b atom "c_myP"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: Compute \'myP\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "swiggle(1,2,0)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Invalid swiggle(x,y,z) function in variable formula: z must be > 0', []],
  [{ b: 'b', post: 'run 0\nvariable b atom "asin(x+2)"\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Arcsin of invalid value in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "2*3"\nprint "VAL ${w}"' }, 'Variable w: Vector-style variable has zero length', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "v_vv+v_v2"\nprint "VAL ${w}"' }, 'Variable w: Inconsistent lengths in vector-style variable', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "v_vv/0"\nprint "VAL ${w}"' }, 'Divide by 0 in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "c_myT"\nprint "VAL ${w}"' }, null, ['VAL [0,0,0,0,0,0]']],
  [{ b: 'b', post: 'run 0\nvariable w vector "c_myP"\nprint "VAL ${w}"' }, null, null],
  [{ b: 'b', post: 'run 0\nvariable w vector "c_myP[1]"\nprint "VAL ${w}"' }, 'Variable w: Compute \'myP\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "f_aa[3]"\nprint "VAL ${w}"' }, 'Variable w: Variable formula fix array is accessed out-of-range', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "f_aa[1]"\nprint "VAL ${w}"' }, null, null],
  [{ b: 'b', post: 'run 0\nvariable w vector "v_va"\nprint "VAL ${w}"' }, 'Variable va: Atom-style variable in vector-style variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "sort(v_vv)"\nprint "VAL ${w}"' }, null, ['VAL [1,2,3]']],
  [{ b: 'b', post: 'run 0\nvariable w vector "sort(v_vv)[1]"\nprint "VAL ${w}"' }, 'Variable w: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "v_w*2"\nprint "VAL ${w}"' }, 'Variable w: has a circular dependency', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "f_av"\nprint "VAL ${w}"' }, null, null],
  [{ b: 'b', post: 'run 0\nvariable w vector "f_avs"\nprint "VAL ${w}"' }, 'Variable w: Fix \'avs\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "v_vv*v_one"\nprint "VAL ${w}"' }, null, ['VAL [1,2,3]']],
  [{ b: 'b', post: 'run 0\nvariable w vector "C_ke[1]"\nprint "VAL ${w}"' }, 'Variable w: Compute \'ke\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "c_myP[1][1]"\nprint "VAL ${w}"' }, 'Variable w: Compute \'myP\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable w vector "sqrt(v_vv-5)"\nprint "VAL ${w}"' }, 'Sqrt of negative value in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(c_nope)"\nprint "VAL ${a}"' }, 'Invalid compute ID \'nope\' in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(c_myT)"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(c_myP)"\nprint "VAL ${a}"' }, null, ['VAL -18.7059518102568']],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(c_myP[1])"\nprint "VAL ${a}"' }, 'Compute \'myP\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(f_aa[1])"\nprint "VAL ${a}"' }, null, ['VAL -18.7059518102568']],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(f_aa[3])"\nprint "VAL ${a}"' }, 'Variable formula fix array is accessed out-of-range', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(v_vv[1])"\nprint "VAL ${a}"' }, 'Invalid special function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(x)"\nprint "VAL ${a}"' }, 'Invalid special function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(v_nope)"\nprint "VAL ${a}"' }, 'Invalid special function in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(1,2)"\nprint "VAL ${a}"' }, 'Variable a: Invalid special function sum() in variable formula', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sort(v_vv)[4]"\nprint "VAL ${a}"' }, 'Variable vv: index 4 exceeds vector size of 3', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sort(c_myP)[7]"\nprint "VAL ${a}"' }, 'index 7 exceeds vector size of 6', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "rsort(v_vv)[1]"\nprint "VAL ${a}"' }, null, ['VAL 3']],
  [{ b: 'b', post: 'run 0\nvariable a equal "min(v_vv)"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'run 0\nvariable a equal "max(v_vv)"\nprint "VAL ${a}"' }, null, ['VAL 3']],
  [{ b: 'b', post: 'run 0\nvariable a equal "slope(v_vv)"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'run 0\nvariable a equal "trap(v_vv)"\nprint "VAL ${a}"' }, null, ['VAL 4']],
  [{ b: 'b', post: 'run 0\nvariable a equal "ave(v_vv)"\nprint "VAL ${a}"' }, null, ['VAL 2']],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(f_av)"\nprint "VAL ${a}"' }, null, null],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(f_avs)"\nprint "VAL ${a}"' }, 'Fix \'avs\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sort(v_vv)[v_one]"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'run 0\nvariable a equal "ave(c_myP)"\nprint "VAL ${a}"' }, null, ['VAL -3.11765863504279']],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(c_myP[1][2])"\nprint "VAL ${a}"' }, 'Compute \'myP\' in variable formula does not compute the requested property. ', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(c_myP[v_one])"\nprint "VAL ${a}"' }, 'Non digit character between brackets in variable', []],
  [{ b: 'b', post: 'run 0\nvariable a equal "sum(c_nope[x])"\nprint "VAL ${a}"' }, 'Non digit character between brackets in variable', []],
  [{ b: 'b', post: 'variable a equal "sum(c_myP)"\nprint "VAL ${a}"' }, 'Variable formula compute cannot be invoked before initialization by a run', []],
  [{ b: 'b', post: 'variable a equal "xcm(g1,x,nope)"\nprint "VAL ${a}"' }, 'Variable a: Region nope in variable formula does not exist', []],
  [{ b: 'b', post: 'variable a equal "count(g1,r1)"\nprint "VAL ${a}"' }, null, ['VAL 14']],
  [{ b: 'b', post: 'variable a equal "mass(g1,r1,r1)"\nprint "VAL ${a}"' }, 'Variable a: Invalid mass() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "inertia(g1,ab)"\nprint "VAL ${a}"' }, 'Variable a: Invalid inertia() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "bound(g1,xmin,r1)"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "gyration(g1,r1,x)"\nprint "VAL ${a}"' }, 'Variable a: Invalid gyration() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "count(g1, r1)"\nprint "VAL ${a}"' }, null, ['VAL 14']],
  [{ b: 'b', post: 'variable a equal "vcm(g1,w)"\nprint "VAL ${a}"' }, 'Variable a: Invalid vcm() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "fcm(g1)"\nprint "VAL ${a}"' }, 'Variable a: Invalid fcm() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "angmom(g1,x,r1,r1)"\nprint "VAL ${a}"' }, 'Variable a: Invalid angmom() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "omega(g1,q)"\nprint "VAL ${a}"' }, 'Variable a: Invalid omega() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "ke(g1,nope)"\nprint "VAL ${a}"' }, 'Variable a: Region nope in variable formula does not exist', []],
  [{ b: 'b', post: 'variable a equal "x[]"\nprint "VAL ${a}"' }, 'Empty brackets in variable', []],
  [{ b: 'b', post: 'variable a equal "x[1"\nprint "VAL ${a}"' }, 'Mismatched brackets in variable', []],
  [{ b: 'b', post: 'variable a equal "x[a]"\nprint "VAL ${a}"' }, 'Non digit character between brackets in variable', []],
  [{ b: 'b', post: 'variable a equal "x[v_q]"\nprint "VAL ${a}"' }, 'Invalid variable name q in variable formula', []],
  [{ b: 'b', post: 'variable a equal "x[v_va]"\nprint "VAL ${a}"' }, 'Invalid variable evaluation for variable va in variable formula', []],
  [{ b: 'b', post: 'variable a equal "x[v_vs]"\nprint "VAL ${a}"' }, 'Illegal value in brackets: stod(v_vs)', []],
  [{ b: 'b', post: 'variable a equal "x[-1]"\nprint "VAL ${a}"' }, 'Index between variable brackets must be positive', []],
  [{ b: 'b', post: 'variable a equal "x[+1]"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "x[2*3]"\nprint "VAL ${a}"' }, null, ['VAL 0.839798095691254']],
  [{ b: 'b', post: 'variable a equal "x[*]"\nprint "VAL ${a}"' }, 'Illegal value in brackets: stoll(*)', []],
  [{ b: 'b', post: 'variable a equal "x[v_one]"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "x[v_a-b]"\nprint "VAL ${a}"' }, 'Variable name between brackets must be letters, numbers, or underscores', []],
  [{ b: 'b', post: 'variable a equal "x[999]"\nprint "VAL ${a}"' }, 'Variable atom ID is too large', []],
  [{ b: 'b', post: 'variable a equal "x[1][2]"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "nope[1]"\nprint "VAL ${a}"' }, 'Invalid atom vector nope in variable formula', []],
  [{ b: 'b', post: 'variable a equal "radius[1]"\nprint "VAL ${a}"' }, 'Variable uses atom property that isn\'t allocated', []],
  [{ b: 'b', post: 'variable a equal "q[1]"\nprint "VAL ${a}"' }, 'Variable uses atom property that isn\'t allocated', []],
  [{ b: 'b', post: 'variable a equal "mol[1]"\nprint "VAL ${a}"' }, 'Variable uses atom property that isn\'t allocated', []],
  [{ b: 'b', post: 'variable a equal "mass[1]"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "type[2]"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "id[3]"\nprint "VAL ${a}"' }, null, ['VAL 3']],
  [{ b: 'b', post: 'variable a equal "vx[1.0]"\nprint "VAL ${a}"' }, 'Non digit character between brackets in variable', []],
  [{ b: 'b', post: 'variable a equal "x[ 1]"\nprint "VAL ${a}"' }, 'Non digit character between brackets in variable', []],
  [{ b: 'b', post: 'variable a equal "x[99999999999999999999]"\nprint "VAL ${a}"' }, 'Illegal value in brackets: stoll(99999999999999999999)', []],
  [{ b: 'b', post: 'variable a equal "v_vv[v_one]"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "v_one[1]"\nprint "VAL ${a}"' }, 'Variable one: Mismatched variable in variable formula', []],
  [{ b: 'b', post: 'variable a equal "v_va[1]"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "v_va[999]"\nprint "VAL ${a}"' }, 'Variable atom ID is too large', []],
  [{ b: 'b', post: 'variable a equal "v_vv[v_vs]"\nprint "VAL ${a}"' }, 'Illegal value in brackets: stod(v_vs)', []],
  [{ b: 'b', post: 'variable a equal "v_vv[4]"\nprint "VAL ${a}"' }, 'Variable vv: Invalid index into vector-style variable', []],
  [{ b: 'b', post: 'variable a equal "c_myP[v_q]"\nprint "VAL ${a}"' }, 'Invalid variable name q in variable formula', []],
  [{ b: 'b', post: 'variable a equal "c_nope[0]"\nprint "VAL ${a}"' }, 'Variable a: Invalid compute ID \'nope\' in variable formula', []],
  [{ b: 'b', post: 'variable a equal "v_nope[0]"\nprint "VAL ${a}"' }, 'Invalid variable reference v_nope in variable formula', []],
  [{ b: 'b', post: 'variable a equal "i_foo[1]"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "d2_bar[1][2]"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "d2_bar[1][3]"\nprint "VAL ${a}"' }, 'Variable a: Invalid custom atom property reference d2_bar in variable formula', []],
  [{ b: 'b', post: 'variable a equal "i_foo"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "i_nope[1]"\nprint "VAL ${a}"' }, 'Variable a: Invalid custom atom property reference i_nope in variable formula', []],
  [{ b: 'b', post: 'variable a equal "i_foo[1][1]"\nprint "VAL ${a}"' }, 'Variable a: Invalid custom atom property reference i_foo in variable formula', []],
  [{ b: 'b', post: 'variable a equal "d2_bar[1]"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "+1"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "-+1"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "_x"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "1.2.3"\nprint "VAL ${a}"' }, null, ['VAL 1.2']],
  [{ b: 'b', post: 'variable a equal "5e"\nprint "VAL ${a}"' }, null, ['VAL 5']],
  [{ b: 'b', post: 'variable a equal ".5e+"\nprint "VAL ${a}"' }, null, ['VAL 0.5']],
  [{ b: 'b', post: 'variable a equal "2x"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "abs((1,2))"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "abs(1,)"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "py_f(1)"\nprint "VAL ${a}"' }, 'Variable a: Invalid python function variable name', []],
  [{ b: 'b', post: 'variable a equal "nope(1"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "-0"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "0*-1"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "2^-1074"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "1 = 2"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "1 & 2"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "1 | 2"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "!0"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "!!3"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "3 |^ 0"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "1,2"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "(1)[2]"\nprint "VAL ${a}"' }, 'Variable a: Invalid syntax in variable formula', []],
  [{ b: 'b', post: 'variable a equal "PI"\nprint "VAL ${a}"' }, null, ['VAL 3.14159265358979']],
  [{ b: 'b', post: 'variable a equal "version > 0"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "1/3"\nprint "VAL ${a}"' }, null, ['VAL 0.333333333333333']],
  [{ b: 'b', post: 'variable a equal "-2^2"\nprint "VAL ${a}"' }, null, ['VAL 4']],
  [{ b: 'b', post: 'variable a equal "ternary(0,1/0,2)"\nprint "VAL ${a}"' }, null, ['VAL 2']],
  [{ b: 'b', post: 'variable a equal "ternary(1,2)"\nprint "VAL ${a}"' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "ramp(1,2)"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "stride(5,1,1)"\nprint "VAL ${a}"' }, 'Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "stride2(1,10,2,3,4,0)"\nprint "VAL ${a}"' }, 'Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "logfreq3(1,1,10)"\nprint "VAL ${a}"' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "logfreq2(1,0,2)"\nprint "VAL ${a}"' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "random(1,2,0)"\nprint "VAL ${a}"' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "normal(1,-2,3)"\nprint "VAL ${a}"' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "random(1,2,0.5)"\nprint "VAL ${a}"' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "cwiggle(1,2,0)"\nprint "VAL ${a}"' }, 'Variable a: Invalid cwiggle(x,y,z) function in variable formula: z must be > 0', []],
  [{ b: 'b', post: 'variable a equal "swiggle(1,2,-1)"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "atan2(1,2,3)"\nprint "VAL ${a}"' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "sign(0)"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "round(-2.5)"\nprint "VAL ${a}"' }, null, ['VAL -2']],
  [{ b: 'b', post: 'variable a equal "floor(1,2)"\nprint "VAL ${a}"' }, 'Variable a: Invalid math function in variable formula', []],
  [{ b: 'b', post: 'variable w vector v_w*2\nprint "VAL ${w}"' }, 'Variable w: has a circular dependency', []],
  [{ b: 'b', post: 'run 0\nvariable b atom v_b\nvariable a equal v_b[1]\nprint "VAL ${a}"' }, 'Variable b: has a circular dependency', []],
  [{ b: 'b', post: 'variable a equal v_c\nvariable c equal v_d\nvariable d equal v_a\nprint "VAL ${a}"' }, 'Variable a: has a circular dependency', []],
  [{ b: 'b', post: 'variable c equal sqrt(-1)\nvariable a equal 1+v_c\nprint "VAL ${a}"' }, 'Variable c: Sqrt of negative value in variable formula', []],
  [{ b: 'b', post: 'variable c index 3x\nvariable a equal v_c\nprint "VAL ${a}"' }, 'Variable c: Non-numeric variable value in variable formula', []],
  [{ b: 'b', post: 'variable c index 0x10\nvariable a equal v_c\nprint "VAL ${a}"' }, 'Variable c: Non-numeric variable value in variable formula', []],
  [{ b: 'b', post: 'variable c index -.5e2\nvariable a equal v_c\nprint "VAL ${a}"' }, null, ['VAL -50']],
  [{ b: 'b', post: 'variable c string " 1"\nvariable a equal v_c\nprint "VAL ${a}"' }, 'Variable c: Non-numeric variable value in variable formula', []],
  [{ b: 'b', post: 'variable a equal "next(nope)"\nprint "VAL ${a}"' }, 'Variable ID \'nope\' in variable formula does not exist', []],
  [{ b: 'b', post: 'variable a equal "next(vv,vs)"\nprint "VAL ${a}"' }, 'Variable a: Invalid special function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "is_file(a,b)"\nprint "VAL ${a}"' }, 'Variable a: Invalid is_file() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "is_timeout(1)"\nprint "VAL ${a}"' }, 'Variable a: Invalid is_timeout() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "is_os()"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "is_defined(compute)"\nprint "VAL ${a}"' }, 'Variable a: Invalid is_defined() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "is_active(package,foo)"\nprint "VAL ${a}"' }, 'Unknown name for info package category: foo', []],
  [{ b: 'b', post: 'variable a equal "is_active(pair,bogus)"\nprint "VAL ${a}"' }, 'Unknown name for info pair category: bogus', []],
  [{ b: 'b', post: 'variable a equal "is_active(newton,x)"\nprint "VAL ${a}"' }, 'Unknown name for info newton category: x', []],
  [{ b: 'b', post: 'variable a equal "is_available(bogus,x)"\nprint "VAL ${a}"' }, 'Unknown category for info is_available(): bogus', []],
  [{ b: 'b', post: 'variable a equal "extract_setting(a,b)"\nprint "VAL ${a}"' }, 'Variable a: Invalid extract_setting() function in variable formula', []],
  [{ b: 'b', post: 'variable a equal "is_active(pair,shift)"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "is_active(pair,manybody)"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "is_defined(region,r1)"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "is_active(package,omp)"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'variable a equal "is_available(pair,lj/cut)"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'variable a equal "gmask(g1)"\nprint "VAL ${a}"' }, 'Variable a: Gmask function in equal-style variable formula', []],
  [{ b: 'b', post: 'variable a equal "rmask(r1)"\nprint "VAL ${a}"' }, 'Variable a: Rmask function in equal-style variable formula', []],
  [{ b: 'b', post: 'labelmap atom 1 C\nvariable a equal "label2type(atom)"\nprint "VAL ${a}"' }, 'Variable a: Invalid label2type(atom) function in variable formula', []],
  [{ b: 'b', post: 'labelmap atom 1 C\nvariable a equal "label2type(foo,C)"\nprint "VAL ${a}"' }, 'Variable a: Invalid kind foo in label2type() in variable', []],
  [{ b: 'b', post: 'labelmap atom 1 C\nvariable a equal "label2type(atom,Z)"\nprint "VAL ${a}"' }, 'Variable a: Invalid atom type label Z in label2type() in variable', []],
  [{ b: 'b', post: 'labelmap atom 1 C\nvariable a equal "label2type(atom,C)"\nprint "VAL ${a}"' }, null, ['VAL 1']],
  [{ b: 'b', post: 'labelmap atom 1 C\nvariable a equal "is_typelabel(atom,Z)"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'labelmap atom 1 C\nvariable a equal "is_typelabel(bond,C)"\nprint "VAL ${a}"' }, null, ['VAL 0']],
  [{ b: 'b', post: 'fix dt all dt/reset 1 NULL NULL 0.1\nvariable a equal "vdisplace(1,2)"\nprint "VAL ${a}"' }, 'Variable a: Must not use vdisplace(x,y) function with fix dt/reset', []],
  [{ b: 'b', post: 'fix dt all dt/reset 1 NULL NULL 0.1\nvariable a equal "swiggle(1,2,3)"\nprint "VAL ${a}"' }, 'Variable a: Must not use swiggle(x,y,z) function with fix dt/reset', []],
  [{ b: 'b', post: 'fix a10 all ave/time 1 10 10 c_myT\nfix pr all print 5 "VAL $(f_a10)"\nrun 10' }, 'Fix in variable not computed at a compatible time', ['VAL 0']],
  [{ b: 'b', post: 'fix a10 all ave/time 1 10 10 c_myT\nvariable a equal f_a10\nfix pr all print 5 "VAL ${a}"\nrun 10' }, 'Variable a: Fix in variable not computed at a compatible time', ['VAL 0']],
  [{ b: 'b', post: 'fix a10 all ave/time 1 10 10 c_myP mode vector\nvariable a equal sum(f_a10)\nfix pr all print 5 "VAL ${a}"\nrun 10' }, 'Fix with ID \'a10\' in variable formula not computed at compatible time', ['VAL 0']],
  [{ b: 'b', post: 'fix a10 all ave/time 1 10 10 c_myT\nfix pr all print 10 "VAL $(f_a10)"\nrun 10' }, null, null],
  [{ b: 'x', post: 'units lj\nvariable a equal "vol"\nprint "VAL ${a}"\n' }, 'Variable a: Variable evaluation before simulation box is defined', []],
  [{ b: 'x', post: 'units lj\nvariable a equal "c_x"\nprint "VAL ${a}"\n' }, 'Variable a: Variable evaluation before simulation box is defined', []],
  [{ b: 'x', post: 'units lj\nvariable a equal "x[1]"\nprint "VAL ${a}"\n' }, 'Variable a: Variable evaluation before simulation box is defined', []],
  [{ b: 'x', post: 'units lj\nvariable a equal "count(all)"\nprint "VAL ${a}"\n' }, null, ['VAL 0']],
  [{ b: 'x', post: 'units lj\nvariable a equal "1+1"\nprint "VAL ${a}"\n' }, null, ['VAL 2']],
  [{ b: 'x', post: 'units lj\nvariable a equal "i_foo[1]"\nprint "VAL ${a}"\n' }, 'Variable a: Variable evaluation before simulation box is defined', []],
];

/**
 * Inputs native LAMMPS does not finish, refused by the engine with a message: [row, engine message, native result].
 * charge() of atoms without charges reads a missing array (a segmentation fault); a number outside the range of a
 * double makes std::stod throw, and native stops with "Exception: stod".
 */
const CRASHES: [Row, string, string][] = [
  [{ b: 'b', post: 'variable a equal "charge(g1)"\nprint "VAL ${a}"' }, 'charge() in a variable formula needs an atom style with charges', 'Segmentation fault'],
  [{ b: 'b', post: 'variable a equal "1e-320"\nprint "VAL ${a}"' }, 'Exception: stod (1e-320 is outside the range of a double)', 'Exception: stod'],
  [{ b: 'b', post: 'variable a equal "-1e-320"\nprint "VAL ${a}"' }, 'Exception: stod (1e-320 is outside the range of a double)', 'Exception: stod'],
  [{ b: 'b', post: 'variable a equal "1e400"\nprint "VAL ${a}"' }, 'Exception: stod (1e400 is outside the range of a double)', 'Exception: stod'],
  [{ b: 'b', post: 'print "VAL $(-0) $(1e-320)"' }, 'Exception: stod (1e-320 is outside the range of a double)', 'Exception: stod'],
];

describe('variable formulas as native LAMMPS', () => {
  it.each(ROWS.map(([r, e, v]) => [label(r), r, e, v] as const))('%s', async (_name, r, error, vals) => {
    const got = await outcome(r);
    expect(got.error).toBe(error);
    if (vals) expect(got.vals).toEqual(vals);
  });
});

describe('formulas native LAMMPS does not finish', () => {
  it.each(CRASHES.map(([r, e]) => [label(r), r, e] as const))('%s', async (_name, r, expected) => {
    expect((await outcome(r)).error).toBe(expected);
  });
});
