import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * The arguments of the averaging fixes as native LAMMPS reads them (the constructors and options of
 * src/fix_ave_time.cpp, fix_ave_histo.cpp, fix_ave_atom.cpp, fix_ave_chunk.cpp, fix_ave_correlate.cpp and
 * EXTRA-FIX/fix_ave_moments.cpp at b332f28, with ArgInfo and utils::expand_args): the order of the checks and their
 * messages. Measured with native LAMMPS (/home/roy/.local/bin/lmp, b332f28): each row is the input below with its lines
 * added; the expected error is the first line of native's message (null when the input runs) and the values are the
 * lines print wrote that start with VAL, joined by spaces.
 */

const BASE = `units lj
atom_style atomic
atom_modify map array
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
variable t equal step
variable ta atom step
compute ch all chunk/atom bin/1d x lower 1.0 units reduced
thermo_modify format float %.15g
compute ke all ke
compute kea all ke/atom
compute sa all stress/atom NULL
variable q atom x
variable w vector [1,2,3]
fix b all ave/time 2 1 2 v_t
fix bb all ave/atom 2 1 2 x
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
  ["fix a all ave/time 1 1", "Illegal fix ave/time command: missing argument(s)", ""],
  ["fix a all ave/time 1 1 1", "Illegal fix ave/time command: missing argument(s)", ""],
  ["fix a all ave/time 0 1 1 v_t", "Illegal fix ave/time nevery value: 0", ""],
  ["fix a all ave/time 1 0 1 v_t", "Illegal fix ave/time nrepeat value: 0", ""],
  ["fix a all ave/time 1 1 0 v_t", "Illegal fix ave/time nfreq value: 0", ""],
  ["fix a all ave/time 2 1 3 v_t", "Inconsistent fix ave/time nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/time 2 3 4 v_t", "Inconsistent fix ave/time nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/time a 1 1 v_t", "Expected integer parameter instead of 'a' in input script or data file", ""],
  ["fix a all ave/time 1 1 1 v_t overwrite", "Fix ave/time overwrite keyword requires ave running setting", ""],
  ["fix a all ave/time 1 1 1 c_nope", "Compute ID nope for fix ave/time does not exist", ""],
  ["fix a all ave/time 1 1 1 c_thermo_temp[9]", "Fix ave/time compute thermo_temp vector is accessed out-of-range", ""],
  ["fix a all ave/time 1 1 1 c_kea", "Fix ave/time compute kea does not calculate a scalar", ""],
  ["fix a all ave/time 1 1 1 c_thermo_temp[1][2]", "Invalid fix ave/time argument: c_thermo_temp[1][2]", ""],
  ["fix a all ave/time 1 1 1 f_nope", "Fix ID nope for fix ave/time does not exist", ""],
  ["fix a all ave/time 1 1 1 f_1", "Fix ave/time fix 1 does not calculate a scalar", ""],
  ["fix a all ave/time 3 1 3 f_b", "Fix b for fix ave/time not computed at compatible time", ""],
  ["fix a all ave/time 1 1 1 v_nope", "Variable name nope for fix ave/time does not exist", ""],
  ["fix a all ave/time 1 1 1 v_q", "Fix ave/time variable q is not equal-style variable", ""],
  ["fix a all ave/time 1 1 1 v_t[1]", "Fix ave/time variable t is not vector-style variable", ""],
  ["fix a all ave/time 1 1 1 v_t mode vector", "Fix ave/time variable t is not vector-style variable", ""],
  ["fix a all ave/time 1 1 1 v_w[1] mode vector", "Fix ave/time mode vector variable w cannot be indexed", ""],
  ["fix a all ave/time 1 1 1 c_thermo_temp[1] mode vector", "Fix ave/time compute thermo_temp does not calculate an array", ""],
  ["fix a all ave/time 1 1 1 v_t ave foo", "Unknown fix ave/time ave keyword foo", ""],
  ["fix a all ave/time 1 1 1 v_t ave window 0", "Illegal fix ave/time ave window argument 0; must be > 0", ""],
  ["fix a all ave/time 1 1 1 v_t ave window", "Illegal fix ave/time ave window command: missing argument(s)", ""],
  ["fix a all ave/time 1 1 1 v_t mode foo", "Unknown fix ave/time mode foo", ""],
  ["fix a all ave/time 1 1 1 v_t off 2", "Invalid fix ave/time off column: 2", ""],
  ["fix a all ave/time 1 1 1 v_t foo", "Unknown fix ave/time keyword foo", ""],
  ["fix a all ave/time 1 1 1 v_t start", "Illegal fix ave/time start command: missing argument(s)", ""],
  ["fix a all ave/time 1 1 1 v_t format %d", "Invalid fix ave/time format argument: conversion 1 of '%d' formats integer values, but floating-point values are provided", ""],
  ["fix a all ave/time 1 1 1 v_t title1", "Illegal fix ave/time title1 command: missing argument(s)", ""],
  ["fix a all ave/time 1 1 1 c_ke v_t", null, ""],
  ["fix a all ave/time 1 1 1 c_ke v_t mode vector", "Fix ave/time compute ke does not calculate a vector", ""],
  ["fix a all ave/time 1 1 1 v_t ave", "Illegal fix ave/time ave command: missing argument(s)", ""],
  ["fix a all ave/time 1 1 1 v_t mode", "Illegal fix ave/time mode command: missing argument(s)", ""],
  ["fix a all ave/time 1 1 1 v_t off", "Illegal fix ave/time off command: missing argument(s)", ""],
  ["fix a all ave/time 1 1 1 v_t file", "Illegal fix ave/time file command: missing argument(s)", ""],
  ["fix a all ave/time 1 1 1 x", "No values from computes, fixes, or variables used in fix ave/time command", ""],
  ["fix a all ave/time 1 1 1 c_", "Invalid fix ave/time argument: c_", ""],
  ["fix a all ave/time 1 1 1 v_t start -5\nrun 2\nprint \"VAL $(f_a)\"", null, "VAL 2"],
  ["fix a all ave/time 1 1 1 c_thermo_temp[*]", null, ""],
  ["fix a all ave/time 1 1 1 c_thermo_temp[*] mode vector", "Invalid fix ave/time argument: c_thermo_temp[*]", ""],
  ["fix a all ave/time 1 1 1 f_1[1]", "Fix ave/time fix 1 does not calculate a vector", ""],
  ["fix a all ave/time 1 1 1 c_thermo_temp[0]\nrun 0\nprint \"VAL $(f_a:%.6f)\"", null, "VAL 1.000000"],
  ["fix a all ave/time 1 1 1 v_t\nunfix 1\nvariable t delete\nrun 1", "Variable name t for fix ave/time does not exist", ""],
  ["fix a all ave/histo 1 1 1 0 1", "Illegal fix ave/histo command: missing argument(s)", ""],
  ["fix a all ave/histo 1 1 1 0 1 2", "Illegal fix ave/histo command: missing argument(s)", ""],
  ["fix a all ave/histo 0 1 1 0 1 2 v_t", "Illegal fix ave/histo nevery value: 0", ""],
  ["fix a all ave/histo 2 1 3 0 1 2 v_t", "Inconsistent fix ave/histo nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/histo 2 3 4 0 1 2 v_t", "Inconsistent fix ave/histo nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/histo 1 1 1 1 0 2 v_t\nrun 0", null, ""],
  ["fix a all ave/histo 1 1 1 a 1 2 v_t", "Expected floating point parameter instead of 'a' in input script or data file", ""],
  ["fix a all ave/histo 1 1 1 0 1 2.5 v_t", "Expected integer parameter instead of '2.5' in input script or data file", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 vx", "fix ave/histo cannot process per-atom values in scalar mode", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 vx mode vector\nrun 0", null, ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_q", "fix ave/histo cannot process per-atom values in scalar mode", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t v_q", "fix ave/histo input kind is not global", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_nope", "Compute ID nope for fix ave/histo does not exist", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_thermo_temp kind peratom", "fix ave/histo input kind is not peratom", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_thermo_temp kind foo", "Unknown fix ave/histo kind option: foo", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_thermo_temp[9]", "fix ave/histo compute thermo_temp vector is accessed out-of-range", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_thermo_temp[1] mode vector", "fix ave/histo compute thermo_temp does not calculate a global array", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t mode vector", "fix ave/histo variable t is not vector-style variable", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_w[1] mode vector", "fix ave/histo variable w cannot be indexed", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t ave window 0", "Illegal fix ave/histo ave window size: 0", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t ave foo", "Unknown fix ave/histo ave option: foo", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t beyond foo", "Unknown fix ave/histo beyond option: foo", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t mode foo", "Unknown fix ave/histo mode option: foo", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t foo", "Unknown fix ave/histo option: foo", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t overwrite", "fix ave/histo overwrite keyword requires ave running setting", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_kea\nrun 0", "fix ave/histo cannot process per-atom values in scalar mode", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_kea mode vector\nrun 0", null, ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_sa mode vector", "fix ave/histo compute sa does not calculate a per-atom vector", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_sa[7] mode vector", "fix ave/histo compute sa array is accessed out-of-range", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 f_bb mode vector\nrun 0", null, ""],
  ["fix a all ave/histo 3 1 3 0 1 2 f_bb mode vector", null, ""],
  ["fix a all ave/histo 1 1 1 0 1 2 f_b", "Fix b for fix ave/histo not computed at compatible time", ""],
  ["fix a all ave/histo 3 1 3 0 1 2 f_b", "Fix b for fix ave/histo not computed at compatible time", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_", "Could not parse value data consistently for fix ave/histo", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t start", "Illegal fix ave/histo start command: missing argument(s)", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t title2", "Illegal fix ave/histo  title2 command: missing argument(s)", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t kind global\nrun 0", null, ""],
  ["fix a all ave/histo 1 1 1 0 1 2 c_thermo_temp v_q", "fix ave/histo input kind is not global", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t beyond extra\nrun 0\nprint \"VAL $(f_a[1][1]) $(f_a[4][1])\"", null, "VAL 0 1"],
  ["fix a all ave/histo 1 1 1 0 1 2 c_ke[1]", "fix ave/histo compute ke does not calculate a global vector", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 v_t[1]", "fix ave/histo variable t is not vector-style variable", ""],
  ["fix a all ave/histo 1 1 1 0 1 2 x kind global", "fix ave/histo input kind is not global", ""],
  ["fix a all ave/atom 1 1", "Illegal fix ave/atom command: missing argument(s)", ""],
  ["fix a all ave/atom 1 1 1", "Illegal fix ave/atom command: missing argument(s)", ""],
  ["fix a all ave/atom 0 1 1 x", "Illegal fix ave/atom nevery value: 0", ""],
  ["fix a all ave/atom 1 1 0 x", "Illegal fix ave/atom nfreq value: 0", ""],
  ["fix a all ave/atom 2 1 3 x", "Inconsistent fix ave/atom nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/atom 2 3 4 x", "Inconsistent fix ave/atom nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/atom 1 1 1 c_nope", "Compute ID nope for fix ave/atom does not exist", ""],
  ["fix a all ave/atom 1 1 1 c_ke", "Fix ave/atom compute ke does not calculate per-atom values", ""],
  ["fix a all ave/atom 1 1 1 c_sa", "Fix ave/atom compute sa does not calculate a per-atom vector", ""],
  ["fix a all ave/atom 1 1 1 c_kea[1]", "Fix ave/atom compute kea does not calculate a per-atom array", ""],
  ["fix a all ave/atom 1 1 1 c_sa[7]", "Fix ave/atom compute sa array is accessed out-of-range", ""],
  ["fix a all ave/atom 1 1 1 f_nope", "Fix ID nope for fix ave/atom does not exist", ""],
  ["fix a all ave/atom 1 1 1 f_b", "Fix ave/atom fix b does not calculate per-atom values", ""],
  ["fix a all ave/atom 3 1 3 f_bb", "Fix bb for fix ave/atom not computed at compatible time", ""],
  ["fix a all ave/atom 1 1 1 v_t", "Fix ave/atom variable t is not atom-style variable", ""],
  ["fix a all ave/atom 1 1 1 v_nope", "Variable name nope for fix ave/atom does not exist", ""],
  ["fix a all ave/atom 1 1 1 v_q[1]", null, ""],
  ["fix a all ave/atom 1 1 1 foo", "Invalid fix ave/atom argument: foo", ""],
  ["fix a all ave/atom 1 1 1 x foo", "Invalid fix ave/atom argument: foo", ""],
  ["fix a all ave/atom 1 1 1 c_sa[1][2]", "Invalid fix ave/atom argument: c_sa[1][2]", ""],
  ["fix a all ave/atom 1 1 1 c_sa[*]\nrun 0\nprint \"VAL $(F_a[1][2]:%.6g)\"", null, "VAL 7.04684"],
  ["fix a all ave/atom 1 1 1 x y\nrun 0\nprint \"VAL $(F_a[2][2])\"", null, "VAL 0.83979809569125363211"],
  ["fix a all ave/atom 1 1 1 c_kea[0]", null, ""],
  ["fix a all ave/chunk 1 1 1 ch", "No values from computes, fixes, or variables used in fix ave/chunk command", ""],
  ["fix a all ave/chunk 1 1 1 ch vx start 5", "Unknown fix ave/chunk keyword: start", ""],
  ["fix a all ave/chunk 0 1 1 ch vx", "Illegal fix ave/chunk nevery value: 0", ""],
  ["fix a all ave/chunk 2 1 3 ch vx", "Inconsistent fix ave/chunk nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/chunk 1 1 1 nope vx", "Chunk/atom compute nope does not exist or is incorrect style for fix ave/chunk", ""],
  ["fix a all ave/chunk 1 1 1 ke vx", "Chunk/atom compute ke does not exist or is incorrect style for fix ave/chunk", ""],
  ["fix a all ave/chunk 1 1 1 ch foo", "No values from computes, fixes, or variables used in fix ave/chunk command", ""],
  ["fix a all ave/chunk 1 1 1 ch c_nope", "Compute ID nope for fix ave/chunk does not exist", ""],
  ["fix a all ave/chunk 1 1 1 ch c_ke", "Fix ave/chunk compute ke does not calculate per-atom values", ""],
  ["fix a all ave/chunk 1 1 1 ch c_kea[1]", "Fix ave/chunk compute kea does not calculate a per-atom array", ""],
  ["fix a all ave/chunk 1 1 1 ch c_sa", "Fix ave/chunk compute sa does not calculate a per-atom vector", ""],
  ["fix a all ave/chunk 1 1 1 ch c_sa[9]", "Fix ave/chunk compute sa vector is accessed out-of-range", ""],
  ["fix a all ave/chunk 1 1 1 ch f_b", "Fix ave/chunk fix b does not calculate per-atom values", ""],
  ["fix a all ave/chunk 3 1 3 ch f_bb", null, ""],
  ["fix a all ave/chunk 1 1 1 ch v_t", "Fix ave/chunk variable t is not atom-style variable", ""],
  ["fix a all ave/chunk 1 1 1 ch v_nope", "Variable name nope for fix ave/chunk does not exist", ""],
  ["fix a all ave/chunk 1 1 1 ch vx norm foo", "Unknown fix ave/chunk norm mode: foo", ""],
  ["fix a all ave/chunk 1 1 1 ch vx ave foo", "Unknown fix ave/chunk ave mode: foo", ""],
  ["fix a all ave/chunk 1 1 1 ch vx ave window 0", "Illegal fix ave/chunk number of windows: 0", ""],
  ["fix a all ave/chunk 1 1 1 ch vx bias nope", "Could not find compute ID nope for temperature bias", ""],
  ["fix a all ave/chunk 1 1 1 ch vx bias ke", "Bias compute ke does not calculate temperature", ""],
  ["fix a all ave/chunk 1 1 1 ch vx overwrite", "Fix ave/chunk overwrite keyword requires ave running setting", ""],
  ["fix a all ave/chunk 1 1 1 ch vx norm", "Illegal fix ave/chunk norm command: missing argument(s)", ""],
  ["fix a all ave/chunk 1 1 1 ch vx adof", "Illegal fix ave/chunk adof command: missing argument(s)", ""],
  ["fix a all ave/chunk 1 1 1 ch vx format %d", "Invalid fix ave/chunk format argument: conversion 1 of '%d' formats integer values, but floating-point values are provided", ""],
  ["fix a all ave/chunk 1 1 1 ch vx title1", "Illegal fix ave/chunk title1 command: missing argument(s)", ""],
  ["fix a all ave/chunk 1 1 1 ch", "No values from computes, fixes, or variables used in fix ave/chunk command", ""],
  ["fix a all ave/chunk 1 1 1 ch vx c_", "Unknown fix ave/chunk keyword: c_", ""],
  ["fix a all ave/chunk 1 1 1 ch vx discard yes", "Unknown fix ave/chunk keyword: discard", ""],
  ["fix a all ave/chunk 1 1 1 ch vx bias thermo_temp", "Bias compute thermo_temp does not calculate a velocity bias", ""],
  ["fix a all ave/chunk 1 1 1 ch vx adof -1\nrun 0", null, ""],
  ["fix a all ave/correlate 1 1 1", "Illegal fix ave/correlate command: missing argument(s)", ""],
  ["fix a all ave/correlate 0 1 1 v_t", "Illegal fix ave/correlate nevery value: 0", ""],
  ["fix a all ave/correlate 2 1 3 v_t", "Inconsistent fix ave/correlate nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/correlate 1 5 2 v_t", "Inconsistent fix ave/correlate nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/correlate 1 5 2 v_t ave running", "Inconsistent fix ave/correlate nevery/nrepeat/nfreq values", ""],
  ["fix a all ave/correlate 1 1 1 c_nope", "Compute ID nope for fix ave/correlate does not exist", ""],
  ["fix a all ave/correlate 1 1 1 c_kea", "Fix ave/correlate compute kea does not calculate a scalar", ""],
  ["fix a all ave/correlate 1 1 1 c_thermo_temp[9]", "Fix ave/correlate compute thermo_temp vector is accessed out-of-range", ""],
  ["fix a all ave/correlate 1 1 1 f_nope", "Fix ID nope for fix ave/correlate does not exist", ""],
  ["fix a all ave/correlate 1 1 1 f_1", "Fix ave/correlate fix 1 does not calculate a scalar", ""],
  ["fix a all ave/correlate 3 1 3 f_b", "Fix b for fix ave/correlate not computed at compatible time", ""],
  ["fix a all ave/correlate 1 1 1 v_q", "Fix ave/correlate variable q is not equal-style variable", ""],
  ["fix a all ave/correlate 1 1 1 v_t[1]", "Fix ave/correlate variable t is not vector-style variable", ""],
  ["fix a all ave/correlate 1 1 1 v_t type foo", "Unknown fix ave/correlate type: foo", ""],
  ["fix a all ave/correlate 1 1 1 v_t ave window", "Unknown fix ave/correlate ave mode: window", ""],
  ["fix a all ave/correlate 1 1 1 v_t foo", "Unkown fix ave/correlate keyword: foo", ""],
  ["fix a all ave/correlate 1 1 1 v_t overwrite", "Fix ave/correlate overwrite keyword requires ave running setting", ""],
  ["fix a all ave/correlate 1 1 1 v_t prefactor", "Illegal fix ave/correlate prefactor command: missing argument(s)", ""],
  ["fix a all ave/correlate 1 1 1 v_t start x", "Expected integer parameter instead of 'x' in input script or data file", ""],
  ["fix a all ave/correlate 1 1 1 x", "Unkown fix ave/correlate keyword: x", ""],
  ["fix a all ave/correlate 1 1 1 c_ke v_t\nrun 0", null, ""],
  ["fix a all ave/correlate 1 1 1 v_t type", "Illegal fix ave/correlate type command: missing argument(s)", ""],
  ["fix a all ave/correlate 1 1 1 c_thermo_temp[1][1]", "Unknown fix ave/correlate data type: c_thermo_temp[1][1]", ""],
  ["fix a all ave/moments 1 1 1 v_t", "Illegal fix ave/moments command: missing argument(s)", ""],
  ["fix a all ave/moments 1 1 1 mean", "Illegal fix ave/moments command: missing argument(s)", ""],
  ["fix a all ave/moments 1 1 1 v_t foo", "No values from computes, fixes, or variables used in fix ave/moments command", ""],
  ["fix a all ave/moments 0 1 1 v_t mean", "Illegal fix ave/moments nevery value: 0", ""],
  ["fix a all ave/moments 1 0 1 v_t mean", "Illegal fix ave/moments nrepeat value: 0", ""],
  ["fix a all ave/moments 1 1 0 v_t mean", "Illegal fix ave/moments nfreq value: 0", ""],
  ["fix a all ave/moments 1 1 1 c_nope mean", "Compute ID nope for fix ave/moments does not exist", ""],
  ["fix a all ave/moments 1 1 1 c_kea mean", "Fix ave/moments compute kea does not calculate a scalar", ""],
  ["fix a all ave/moments 1 1 1 c_thermo_temp[9] mean", "Fix ave/moments compute thermo_temp vector is accessed out-of-range", ""],
  ["fix a all ave/moments 1 1 1 f_nope mean", "Fix ID nope for fix ave/moments does not exist", ""],
  ["fix a all ave/moments 1 1 1 f_1 mean", "Fix ave/moments fix 1 does not calculate a scalar", ""],
  ["fix a all ave/moments 3 1 3 f_b mean", "Fix b for fix ave/moments not computed at compatible time", ""],
  ["fix a all ave/moments 1 1 1 v_q mean", "Fix ave/moments variable q is not equal-style variable", ""],
  ["fix a all ave/moments 1 1 1 v_t[1] mean", "Fix ave/moments variable t is not vector-style variable", ""],
  ["fix a all ave/moments 1 1 1 v_t mean history 0", "Illegal ave/moments history argument 0; must be > 0", ""],
  ["fix a all ave/moments 1 1 1 v_t mean history", "Illegal fix ave/moments history command: missing argument(s)", ""],
  ["fix a all ave/moments 1 1 1 v_t mean start", "Illegal fix ave/moments start command: missing argument(s)", ""],
  ["fix a all ave/moments 1 1 1 v_t mean foo 1", "Unknown fix ave/moments keyword foo", ""],
  ["fix a all ave/moments 1 1 1 c_ke v_t mean\nrun 0", "Fix ave/moments cannot set output array intensive/extensive from these inputs", ""],
  ["fix a all ave/moments 1 1 1 x mean", "No values from computes, fixes, or variables used in fix ave/moments command", ""],
  ["fix a all ave/moments 5 1 4 v_t mean", null, ""],
  ["fix a all ave/moments 1 5 4 v_t mean", null, ""],
  ["fix a all ave/moments 1 1 1 c_thermo_temp[1][1] mean", "Invalid fix ave/moments argument: c_thermo_temp[1][1]", ""],
  ["fix a all ave/moments 1 1 1 v_t mean median", "Unknown fix ave/moments keyword median", ""],
];

/**
 * Inputs the engine refuses where native LAMMPS has no defined result: [lines, engine message, native result]. A bin
 * count of 0 or less makes native size its histogram with that count and write past it when it bins.
 */
const REFUSED: [string, string, string][] = [
  ["fix a all ave/histo 1 1 1 0 1 0 v_t", "fix ave/histo: Nbin 0 is refused: native LAMMPS writes outside its histogram for it", "runs (no run follows)"],
  ["fix a all ave/histo 1 1 1 0 1 -2 v_t", "fix ave/histo: Nbin -2 is refused: native LAMMPS writes outside its histogram for it", "stops with a crash (exit 1, no error message)"],
];

describe('averaging fix arguments as native LAMMPS', () => {
  it.each(ROWS.map(([p, e, v]) => [p.replace(/\n/g, ' / '), p, e, v] as const))('%s', async (_name, p, error, vals) => {
    const got = await outcome(p);
    expect(got.error).toBe(error);
    expect(got.vals).toBe(vals);
  });
});

describe('averaging fix arguments native LAMMPS does not handle', () => {
  it.each(REFUSED.map(([p, e]) => [p.replace(/\n/g, ' / '), p, e] as const))('%s', async (_name, p, expected) => {
    expect((await outcome(p)).error).toBe(expected);
  });
});
