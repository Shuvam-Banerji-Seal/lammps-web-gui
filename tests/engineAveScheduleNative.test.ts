import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * The sampling schedule of the averaging fixes as native LAMMPS (src/fix_ave_time.cpp, fix_ave_histo.cpp,
 * fix_ave_atom.cpp, fix_ave_chunk.cpp, fix_ave_correlate.cpp and EXTRA-FIX/fix_ave_moments.cpp at b332f28): the step
 * that samples next (nvalid) is scheduled when the fix is defined and after each sample, a window's samples end on a
 * multiple of Nfreq (at or after start, which may put its first samples before start), a minimization past the next
 * sample step restarts the window, running and window averages and correlation sums carry over runs, fix ave/atom's
 * output holds the partial sums of an unfinished window, and fix ave/moments takes its window from a ring of samples.
 * Also the list form of a vector-style variable (src/variable.cpp parse_vector). Measured with native LAMMPS
 * (/home/roy/.local/bin/lmp, b332f28): each row is the input below (b: with its lines added; x: the lines alone); the
 * expected error is the first line of native's message (null when the input runs) and the values are the lines print
 * wrote that start with VAL, joined by spaces.
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
`;

const outcome = async (kind: string, post: string): Promise<{ error: string | null; vals: string }> => {
  const vals: string[] = [];
  const s = new Session({
    emit: (ev: { kind: string; text?: string }) => { if (ev.kind === 'log' && ev.text?.startsWith('VAL ')) vals.push(ev.text); },
    writeFile: () => {},
  } as never);
  let error: string | null = null;
  // the engine's message starts with the line of the input ("line N: "), which native prints elsewhere
  try { await s.execute(`${kind === 'b' ? BASE : ''}${post}\n`); } catch (e) { error = (e as Error).message.split('\n')[0].replace(/^line \d+: /, ''); }
  return { error, vals: vals.join(' ') };
};

const ROWS: [string, string, string | null, string][] = [
  ["b", "fix a all ave/time 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a)\"\nrun 25\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 10 8 VAL 20 18 VAL end 25 18"],
  ["b", "run 3\nfix a all ave/time 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a)\"\nrun 30\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 10 8 VAL 20 18 VAL 30 28 VAL end 33 28"],
  ["b", "fix a all ave/time 1 1 5 v_t\nfix pr all print 5 \"VAL $(step) $(f_a)\"\nrun 0\nprint \"VAL end $(step) $(f_a)\"\nrun 0\nprint \"VAL end $(step) $(f_a)\"\nrun 7\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL end 0 0 VAL 0 0 VAL end 0 0 VAL 0 0 VAL 5 5 VAL end 7 5"],
  ["b", "fix a all ave/time 2 2 10 v_t start 20\nfix pr all print 10 \"VAL $(step) $(f_a)\"\nrun 40\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 10 0 VAL 20 19 VAL 30 29 VAL 40 39 VAL end 40 39"],
  ["b", "fix a all ave/time 1 2 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a)\"\nminimize 0 0 5 10\nprint \"VAL end $(step) $(f_a)\"\nrun 20\nprint \"VAL end $(step) $(f_a)\"", null, "VAL end 5 0 VAL 10 9.5 VAL 20 19.5 VAL end 25 19.5"],
  ["b", "fix a all ave/time 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a)\"\nrun 7\nrun 10\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 10 8 VAL end 17 8"],
  ["b", "fix a all ave/time 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a)\"\nrun 7\nrun 13 pre no\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 10 8 VAL 20 18 VAL end 20 18"],
  ["b", "fix a all ave/time 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a)\"\nrun 9\nprint \"VAL end $(step) $(f_a)\"\nrun 1\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL end 9 0 VAL 10 8 VAL end 10 8"],
  ["b", "run 3\nfix a all ave/time 5 1 5 v_t\nfix pr all print 5 \"VAL $(step) $(f_a)\"\nrun 12\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 5 5 VAL 10 10 VAL 15 15 VAL end 15 15"],
  ["b", "run 1\nfix a all ave/time 3 2 9 v_t\nfix pr all print 9 \"VAL $(step) $(f_a)\"\nrun 20\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 9 7.5 VAL 18 16.5 VAL end 21 16.5"],
  ["b", "fix a all ave/time 1 3 4 v_t\nfix pr all print 4 \"VAL $(step) $(f_a)\"\nrun 10 every 3 NULL\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 4 3 VAL 8 7 VAL end 10 7"],
  ["b", "fix a all ave/time 2 1 6 v_t start 5\nfix pr all print 6 \"VAL $(step) $(f_a)\"\nrun 20\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 6 6 VAL 12 12 VAL 18 18 VAL end 20 18"],
  ["b", "fix a all ave/time 1 1 5 v_t ave running\nfix pr all print 5 \"VAL $(step) $(f_a)\"\nrun 10\nrun 10\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 5 2.5 VAL 10 5 VAL 10 5 VAL 15 7.5 VAL 20 10 VAL end 20 10"],
  ["b", "fix a all ave/time 1 1 5 v_t ave window 2\nfix pr all print 5 \"VAL $(step) $(f_a)\"\nrun 10\nrun 10\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 5 2.5 VAL 10 7.5 VAL 10 7.5 VAL 15 12.5 VAL 20 17.5 VAL end 20 17.5"],
  ["b", "fix a all ave/time 2 2 4 v_t ave running\nfix pr all print 4 \"VAL $(step) $(f_a)\"\nrun 6\nminimize 0 0 3 10\nrun 10\nprint \"VAL end $(step) $(f_a)\"", null, "VAL 0 0 VAL 4 3 VAL 12 7 VAL 16 9.6666666666666660745 VAL end 19 9.6666666666666660745"],
  ["b", "fix a all ave/histo 2 3 10 -1 1000 4 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 25\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 0 0 0 VAL 10 3 6 10 VAL 20 3 16 20 VAL end 25 3 16 20"],
  ["b", "run 3\nfix a all ave/histo 2 3 10 -1 1000 4 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 30\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 10 3 6 10 VAL 20 3 16 20 VAL 30 3 26 30 VAL end 33 3 26 30"],
  ["b", "fix a all ave/histo 1 1 5 -1 1000 4 v_t\nfix pr all print 5 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 0\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 0\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 7\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 1 0 0 VAL end 0 1 0 0 VAL 0 1 0 0 VAL end 0 1 0 0 VAL 0 1 0 0 VAL 5 1 5 5 VAL end 7 1 5 5"],
  ["b", "fix a all ave/histo 2 2 10 -1 1000 4 v_t start 20\nfix pr all print 10 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 40\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 0 0 0 VAL 10 0 0 0 VAL 20 2 18 20 VAL 30 2 28 30 VAL 40 2 38 40 VAL end 40 2 38 40"],
  ["b", "fix a all ave/histo 1 2 10 -1 1000 4 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nminimize 0 0 5 10\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL end 5 0 0 0 VAL 10 2 9 10 VAL 20 2 19 20 VAL end 25 2 19 20"],
  ["b", "fix a all ave/histo 2 3 10 -1 1000 4 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 7\nrun 10\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 0 0 0 VAL 10 3 6 10 VAL end 17 3 6 10"],
  ["b", "fix a all ave/histo 2 3 10 -1 1000 4 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 7\nrun 13 pre no\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 0 0 0 VAL 10 3 6 10 VAL 20 3 16 20 VAL end 20 3 16 20"],
  ["b", "fix a all ave/histo 2 3 10 -1 1000 4 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 9\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 1\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 0 0 0 VAL end 9 0 0 0 VAL 10 3 6 10 VAL end 10 3 6 10"],
  ["b", "run 3\nfix a all ave/histo 5 1 5 -1 1000 4 v_t\nfix pr all print 5 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 12\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 5 1 5 5 VAL 10 1 10 10 VAL 15 1 15 15 VAL end 15 1 15 15"],
  ["b", "run 1\nfix a all ave/histo 3 2 9 -1 1000 4 v_t\nfix pr all print 9 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 9 2 6 9 VAL 18 2 15 18 VAL end 21 2 15 18"],
  ["b", "fix a all ave/histo 1 3 4 -1 1000 4 v_t\nfix pr all print 4 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 10 every 3 NULL\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 0 0 0 VAL 4 3 2 4 VAL 8 3 6 8 VAL end 10 3 6 8"],
  ["b", "fix a all ave/histo 2 1 6 -1 1000 4 v_t start 5\nfix pr all print 6 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 1 0 0 VAL 6 1 6 6 VAL 12 1 12 12 VAL 18 1 18 18 VAL end 20 1 18 18"],
  ["b", "fix a all ave/histo 1 1 5 -1 1000 4 v_t ave running\nfix pr all print 5 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 10\nrun 10\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 1 0 0 VAL 5 2 0 5 VAL 10 3 0 10 VAL 10 3 0 10 VAL 15 4 0 15 VAL 20 5 0 20 VAL end 20 5 0 20"],
  ["b", "fix a all ave/histo 1 1 5 -1 1000 4 v_t ave window 2\nfix pr all print 5 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 10\nrun 10\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 1 0 0 VAL 5 2 0 5 VAL 10 2 5 10 VAL 10 2 5 10 VAL 15 2 10 15 VAL 20 2 15 20 VAL end 20 2 15 20"],
  ["b", "fix a all ave/histo 2 2 4 -1 1000 4 v_t ave running\nfix pr all print 4 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"\nrun 6\nminimize 0 0 3 10\nrun 10\nprint \"VAL end $(step) $(f_a[1]) $(f_a[3]) $(f_a[4])\"", null, "VAL 0 0 0 0 VAL 4 2 0 4 VAL 12 4 0 12 VAL 16 6 0 16 VAL end 19 6 0 16"],
  ["b", "fix a all ave/chunk 2 3 10 ch v_ta\nfix pr all print 10 \"VAL $(step) $(f_a[1][3])\"\nrun 25\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL 10 8 VAL 20 18 VAL end 25 18"],
  ["b", "run 3\nfix a all ave/chunk 2 3 10 ch v_ta\nfix pr all print 10 \"VAL $(step) $(f_a[1][3])\"\nrun 30\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 10 8 VAL 20 18 VAL 30 28 VAL end 33 28"],
  ["b", "fix a all ave/chunk 1 1 5 ch v_ta\nfix pr all print 5 \"VAL $(step) $(f_a[1][3])\"\nrun 0\nprint \"VAL end $(step) $(f_a[1][3])\"\nrun 0\nprint \"VAL end $(step) $(f_a[1][3])\"\nrun 7\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL end 0 0 VAL 0 0 VAL end 0 0 VAL 0 0 VAL 5 5 VAL end 7 5"],
  ["b", "fix a all ave/chunk 1 2 10 ch v_ta\nfix pr all print 10 \"VAL $(step) $(f_a[1][3])\"\nminimize 0 0 5 10\nprint \"VAL end $(step) $(f_a[1][3])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL end 5 0 VAL 10 9.5 VAL 20 19.5 VAL end 25 19.5"],
  ["b", "fix a all ave/chunk 2 3 10 ch v_ta\nfix pr all print 10 \"VAL $(step) $(f_a[1][3])\"\nrun 7\nrun 10\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL 10 8 VAL end 17 8"],
  ["b", "fix a all ave/chunk 2 3 10 ch v_ta\nfix pr all print 10 \"VAL $(step) $(f_a[1][3])\"\nrun 7\nrun 13 pre no\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL 10 8 VAL 20 18 VAL end 20 18"],
  ["b", "fix a all ave/chunk 2 3 10 ch v_ta\nfix pr all print 10 \"VAL $(step) $(f_a[1][3])\"\nrun 9\nprint \"VAL end $(step) $(f_a[1][3])\"\nrun 1\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL end 9 0 VAL 10 8 VAL end 10 8"],
  ["b", "run 3\nfix a all ave/chunk 5 1 5 ch v_ta\nfix pr all print 5 \"VAL $(step) $(f_a[1][3])\"\nrun 12\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 5 5 VAL 10 10 VAL 15 15 VAL end 15 15"],
  ["b", "run 1\nfix a all ave/chunk 3 2 9 ch v_ta\nfix pr all print 9 \"VAL $(step) $(f_a[1][3])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 9 7.5 VAL 18 16.5 VAL end 21 16.5"],
  ["b", "fix a all ave/chunk 1 3 4 ch v_ta\nfix pr all print 4 \"VAL $(step) $(f_a[1][3])\"\nrun 10 every 3 NULL\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL 4 3 VAL 8 7 VAL end 10 7"],
  ["b", "fix a all ave/chunk 1 1 5 ch v_ta ave running\nfix pr all print 5 \"VAL $(step) $(f_a[1][3])\"\nrun 10\nrun 10\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL 5 2.5 VAL 10 5 VAL 10 5 VAL 15 7.5 VAL 20 10 VAL end 20 10"],
  ["b", "fix a all ave/chunk 1 1 5 ch v_ta ave window 2\nfix pr all print 5 \"VAL $(step) $(f_a[1][3])\"\nrun 10\nrun 10\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL 5 2.5 VAL 10 7.5 VAL 10 7.5 VAL 15 12.5 VAL 20 17.5 VAL end 20 17.5"],
  ["b", "fix a all ave/chunk 2 2 4 ch v_ta ave running\nfix pr all print 4 \"VAL $(step) $(f_a[1][3])\"\nrun 6\nminimize 0 0 3 10\nrun 10\nprint \"VAL end $(step) $(f_a[1][3])\"", null, "VAL 0 0 VAL 4 3 VAL 12 7 VAL 16 9.6666666666666660745 VAL end 19 9.6666666666666660745"],
  ["b", "fix a all ave/correlate 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 25\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 1 0 VAL 10 6 36.666666666666664298 VAL 20 6 236.66666666666665719 VAL end 25 6 236.66666666666665719"],
  ["b", "run 3\nfix a all ave/correlate 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 30\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 10 4 54 VAL 20 6 236.66666666666665719 VAL 30 6 636.66666666666662877 VAL end 33 6 636.66666666666662877"],
  ["b", "fix a all ave/correlate 1 1 5 v_t\nfix pr all print 5 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 0\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 0\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 7\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 1 0 VAL end 0 1 0 VAL 0 1 0 VAL end 0 1 0 VAL 0 1 0 VAL 5 6 9.1666666666666660745 VAL end 7 6 9.1666666666666660745"],
  ["b", "fix a all ave/correlate 2 2 10 v_t start 20\nfix pr all print 10 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 40\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 0 0 VAL 10 0 0 VAL 20 1 400 VAL 30 6 636.66666666666662877 VAL 40 6 1236.6666666666667425 VAL end 40 6 1236.6666666666667425"],
  ["b", "fix a all ave/correlate 1 2 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nminimize 0 0 5 10\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL end 5 0 0 VAL 10 6 59.166666666666664298 VAL 20 11 235 VAL end 25 11 235"],
  ["b", "fix a all ave/correlate 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 7\nrun 10\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 1 0 VAL 10 6 36.666666666666664298 VAL end 17 6 36.666666666666664298"],
  ["b", "fix a all ave/correlate 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 7\nrun 13 pre no\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 1 0 VAL 10 6 36.666666666666664298 VAL 20 6 236.66666666666665719 VAL end 20 6 236.66666666666665719"],
  ["b", "fix a all ave/correlate 2 3 10 v_t\nfix pr all print 10 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 9\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 1\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 1 0 VAL end 9 1 0 VAL 10 6 36.666666666666664298 VAL end 10 6 36.666666666666664298"],
  ["b", "run 3\nfix a all ave/correlate 5 1 5 v_t\nfix pr all print 5 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 12\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 5 1 25 VAL 10 2 62.5 VAL 15 2 162.5 VAL end 15 2 162.5"],
  ["b", "run 1\nfix a all ave/correlate 3 2 9 v_t\nfix pr all print 9 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 9 3 42 VAL 18 4 193.5 VAL end 21 4 193.5"],
  ["b", "fix a all ave/correlate 1 3 4 v_t\nfix pr all print 4 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 10 every 3 NULL\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 1 0 VAL 4 5 6 VAL 8 5 38 VAL end 10 5 38"],
  ["b", "fix a all ave/correlate 2 1 6 v_t start 5\nfix pr all print 6 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 0 0 VAL 6 1 36 VAL 12 4 86 VAL 18 4 230 VAL end 20 4 230"],
  ["b", "fix a all ave/correlate 1 1 5 v_t ave running\nfix pr all print 5 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 10\nrun 10\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 1 0 VAL 5 6 9.1666666666666660745 VAL 10 11 35 VAL 10 11 35 VAL 15 16 77.5 VAL 20 21 136.66666666666665719 VAL end 20 21 136.66666666666665719"],
  ["b", "fix a all ave/correlate 2 2 4 v_t ave running\nfix pr all print 4 \"VAL $(step) $(f_a[1][2]) $(f_a[1][3])\"\nrun 6\nminimize 0 0 3 10\nrun 10\nprint \"VAL end $(step) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 0 1 0 VAL 4 3 6.6666666666666669627 VAL 12 6 50 VAL 16 8 94 VAL end 19 8 94"],
  ["b", "fix a all ave/moments 2 3 10 v_t mean\nfix pr all print 10 \"VAL $(step) $(f_a[1])\"\nrun 25\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 0 0 VAL 10 7.9999999999999991118 VAL 20 18 VAL end 25 18"],
  ["b", "run 3\nfix a all ave/moments 2 3 10 v_t mean\nfix pr all print 10 \"VAL $(step) $(f_a[1])\"\nrun 30\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 10 7.9999999999999991118 VAL 20 18 VAL 30 28 VAL end 33 28"],
  ["b", "fix a all ave/moments 1 1 5 v_t mean\nfix pr all print 5 \"VAL $(step) $(f_a[1])\"\nrun 0\nprint \"VAL end $(step) $(f_a[1])\"\nrun 0\nprint \"VAL end $(step) $(f_a[1])\"\nrun 7\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 0 0 VAL end 0 0 VAL 0 0 VAL end 0 0 VAL 0 0 VAL 5 0 VAL end 7 0"],
  ["b", "fix a all ave/moments 2 2 10 v_t mean start 20\nfix pr all print 10 \"VAL $(step) $(f_a[1])\"\nrun 40\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 0 0 VAL 10 0 VAL 20 0 VAL 30 0 VAL 40 0 VAL end 40 0"],
  ["b", "fix a all ave/moments 1 2 10 v_t mean\nfix pr all print 10 \"VAL $(step) $(f_a[1])\"\nminimize 0 0 5 10\nprint \"VAL end $(step) $(f_a[1])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL end 5 0 VAL 10 0 VAL 20 0 VAL end 25 0"],
  ["b", "fix a all ave/moments 2 3 10 v_t mean\nfix pr all print 10 \"VAL $(step) $(f_a[1])\"\nrun 7\nrun 10\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 0 0 VAL 10 7.9999999999999991118 VAL end 17 7.9999999999999991118"],
  ["b", "fix a all ave/moments 2 3 10 v_t mean\nfix pr all print 10 \"VAL $(step) $(f_a[1])\"\nrun 7\nrun 13 pre no\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 0 0 VAL 10 7.9999999999999991118 VAL 20 18 VAL end 20 18"],
  ["b", "fix a all ave/moments 2 3 10 v_t mean\nfix pr all print 10 \"VAL $(step) $(f_a[1])\"\nrun 9\nprint \"VAL end $(step) $(f_a[1])\"\nrun 1\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 0 0 VAL end 9 0 VAL 10 7.9999999999999991118 VAL end 10 7.9999999999999991118"],
  ["b", "run 3\nfix a all ave/moments 5 1 5 v_t mean\nfix pr all print 5 \"VAL $(step) $(f_a[1])\"\nrun 12\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 5 0 VAL 10 0 VAL 15 0 VAL end 15 0"],
  ["b", "run 1\nfix a all ave/moments 3 2 9 v_t mean\nfix pr all print 9 \"VAL $(step) $(f_a[1])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 9 0 VAL 18 0 VAL end 21 0"],
  ["b", "fix a all ave/moments 1 3 4 v_t mean\nfix pr all print 4 \"VAL $(step) $(f_a[1])\"\nrun 10 every 3 NULL\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 0 0 VAL 4 3 VAL 8 7 VAL end 10 7"],
  ["b", "fix a all ave/moments 2 1 6 v_t mean start 5\nfix pr all print 6 \"VAL $(step) $(f_a[1])\"\nrun 20\nprint \"VAL end $(step) $(f_a[1])\"", null, "VAL 0 0 VAL 6 0 VAL 12 0 VAL 18 0 VAL end 20 0"],
  ["b", "fix a all ave/atom 2 3 10 v_ta\nfix pr all print 10 \"VAL $(step) $(F_a[1])\"\nrun 25\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 0 0 VAL 10 8 VAL 20 18 VAL end 25 18"],
  ["b", "run 3\nfix a all ave/atom 2 3 10 v_ta\nfix pr all print 10 \"VAL $(step) $(F_a[1])\"\nrun 30\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 10 8 VAL 20 18 VAL 30 28 VAL end 33 28"],
  ["b", "fix a all ave/atom 1 1 5 v_ta\nfix pr all print 5 \"VAL $(step) $(F_a[1])\"\nrun 0\nprint \"VAL end $(step) $(F_a[1])\"\nrun 0\nprint \"VAL end $(step) $(F_a[1])\"\nrun 7\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 0 0 VAL end 0 0 VAL 0 0 VAL end 0 0 VAL 0 0 VAL 5 5 VAL end 7 5"],
  ["b", "fix a all ave/atom 1 2 10 v_ta\nfix pr all print 10 \"VAL $(step) $(F_a[1])\"\nminimize 0 0 5 10\nprint \"VAL end $(step) $(F_a[1])\"\nrun 20\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL end 5 0 VAL 10 9.5 VAL 20 19.5 VAL end 25 19.5"],
  ["b", "fix a all ave/atom 2 3 10 v_ta\nfix pr all print 10 \"VAL $(step) $(F_a[1])\"\nrun 7\nrun 10\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 0 0 VAL 10 8 VAL end 17 16"],
  ["b", "fix a all ave/atom 2 3 10 v_ta\nfix pr all print 10 \"VAL $(step) $(F_a[1])\"\nrun 7\nrun 13 pre no\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 0 0 VAL 10 8 VAL 20 18 VAL end 20 18"],
  ["b", "fix a all ave/atom 2 3 10 v_ta\nfix pr all print 10 \"VAL $(step) $(F_a[1])\"\nrun 9\nprint \"VAL end $(step) $(F_a[1])\"\nrun 1\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 0 0 VAL end 9 14 VAL 10 8 VAL end 10 8"],
  ["b", "run 3\nfix a all ave/atom 5 1 5 v_ta\nfix pr all print 5 \"VAL $(step) $(F_a[1])\"\nrun 12\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 5 5 VAL 10 10 VAL 15 15 VAL end 15 15"],
  ["b", "run 1\nfix a all ave/atom 3 2 9 v_ta\nfix pr all print 9 \"VAL $(step) $(F_a[1])\"\nrun 20\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 9 7.5 VAL 18 16.5 VAL end 21 16.5"],
  ["b", "fix a all ave/atom 1 3 4 v_ta\nfix pr all print 4 \"VAL $(step) $(F_a[1])\"\nrun 10 every 3 NULL\nprint \"VAL end $(step) $(F_a[1])\"", null, "VAL 0 0 VAL 4 3 VAL 8 7 VAL end 10 10"],
  ["b", "fix a all ave/moments 1 4 3 v_t mean variance\nfix pr all print 1 \"VAL $(step) $(f_a[1]) $(f_a[2])\"\nrun 10", null, "VAL 0 0 0 VAL 1 0 0 VAL 2 0 0 VAL 3 1.5 1.6666666666666665186 VAL 4 1.5 1.6666666666666665186 VAL 5 1.5 1.6666666666666665186 VAL 6 1.5 1.6666666666666665186 VAL 7 1.5 1.6666666666666665186 VAL 8 1.5 1.6666666666666665186 VAL 9 1.5 1.6666666666666665186 VAL 10 1.5 1.6666666666666665186"],
  ["b", "fix a all ave/moments 1 8 5 v_t mean stddev skew kurtosis\nfix pr all print 5 \"VAL $(step) $(f_a[1]) $(f_a[2]) $(f_a[3]) $(f_a[4])\"\nrun 20", null, "VAL 0 0 0 0 0 VAL 5 3 1.5811388300841897614 0 -1.1999999999999995115 VAL 10 6.5 2.4494897427831778813 0 -1.1999999999999997335 VAL 15 11.5 2.4494897427831778813 0 -1.1999999999999997335 VAL 20 16.5 2.4494897427831778813 0 -1.1999999999999997335"],
  ["b", "fix a all ave/moments 2 3 6 v_t mean start 13\nfix pr all print 6 \"VAL $(step) $(f_a[1])\"\nrun 30", null, "VAL 0 0 VAL 6 0 VAL 12 0 VAL 18 16 VAL 24 22 VAL 30 28"],
  ["b", "fix a all ave/moments 1 3 3 v_t mean start 6\nfix pr all print 3 \"VAL $(step) $(f_a[1])\"\nrun 12", null, "VAL 0 0 VAL 3 0 VAL 6 0 VAL 9 8 VAL 12 11"],
  ["b", "fix a all ave/moments 1 5 5 v_t mean history 3\nrun 20\nprint \"VAL $(f_a[1][1]) $(f_a[1][2]) $(f_a[1][3])\"", null, "VAL 18 13.000000000000001776 8"],
  ["b", "fix a all ave/moments 2 5 4 v_t variance\nminimize 0 0 3 10\nrun 20\nprint \"VAL $(step) $(f_a[1])\"", null, "VAL 23 0"],
  ["b", "variable w vector [1,2,3]*step\nfix a all ave/moments 1 3 5 v_w[2] v_w[7] mean\nrun 10\nprint \"VAL $(f_a[1]) $(f_a[2])\"", "Vector variable formula lacks opening or closing brace: [1,2,3]*step", ""],
  ["b", "fix a all ave/correlate 2 3 6 v_t start 5\nfix pr all print 6 \"VAL $(step) $(f_a[1][2]) $(f_a[2][2]) $(f_a[3][2]) $(f_a[3][3])\"\nrun 24", null, "VAL 0 0 0 0 0 VAL 6 1 0 0 0 VAL 12 4 3 2 78 VAL 18 4 3 2 222 VAL 24 4 3 2 438"],
  ["b", "fix a all ave/correlate 1 3 4 v_t ave running\nrun 6\nminimize 0 0 3 10\nrun 8\nprint \"VAL $(step) $(f_a[1][2]) $(f_a[2][2]) $(f_a[3][2]) $(f_a[2][3])\"", null, "VAL 17 15 13 11 91.538461538461532996"],
  ["b", "variable u equal 2*step\nfix a all ave/correlate 1 3 3 v_t v_u type upper\nrun 9\nprint \"VAL $(f_a[1][3]) $(f_a[2][3]) $(f_a[3][3])\"", null, "VAL 115 113.3333333333333286 111"],
  ["b", "variable u equal 2*step\nfix a all ave/correlate 1 3 3 v_t v_u type full\nrun 9\nprint \"VAL $(f_a[2][3]) $(f_a[2][4]) $(f_a[2][5]) $(f_a[2][6])\"", null, "VAL 56.666666666666664298 113.3333333333333286 113.3333333333333286 226.66666666666665719"],
  ["b", "variable u equal 2*step\nfix a all ave/correlate 1 3 3 v_t v_u type first\nrun 9\nprint \"VAL $(f_a[2][3]) $(f_a[2][4])\"", null, "VAL 56.666666666666664298 113.3333333333333286"],
  ["b", "fix a all ave/correlate 1 3 3 v_t prefactor 0.5\nrun 0\nrun 4\nprint \"VAL $(f_a[1][2]) $(f_a[1][3]) $(f_a[2][3])\"", null, "VAL 4 1.75 1.3333333333333332593"],
  ["b", "compute pr all pressure thermo_temp\nfix a all ave/time 2 3 10 c_pr mode vector\nrun 25\nprint \"VAL $(f_a[1]:%.10g) $(f_a[6]:%.10g)\"", null, "VAL -1.581507047 1.052662514"],
  ["b", "variable u equal 2*step\nfix a all ave/time 1 2 4 v_t v_u off 2\nrun 9\nprint \"VAL $(f_a[1]) $(f_a[2])\"", null, "VAL 7.5 16"],
  ["b", "fix a all ave/time 1 1 3 v_t ave window 3\nfix pr all print 3 \"VAL $(step) $(f_a)\"\nrun 15", null, "VAL 0 0 VAL 3 1.5 VAL 6 3 VAL 9 6 VAL 12 9 VAL 15 12"],
  ["b", "fix a all ave/time 5 2 10 v_t\nrun 7\nrun 3 upto\nrun 13\nprint \"VAL $(step) $(f_a)\"", "Invalid run command upto value: 3. Must be >= 7", ""],
  ["b", "run 7\nfix a all ave/time 2 2 4 v_t\nfix pr all print 4 \"VAL $(step) $(f_a)\"\nrun 9", null, "VAL 8 0 VAL 12 11 VAL 16 15"],
  ["b", "fix a all ave/histo 1 2 4 -1 100 4 v_t ave window 2\nfix pr all print 4 \"VAL $(step) $(f_a[1]) $(f_a[3]) $(f_a[4]) $(f_a[1][3])\"\nrun 16", null, "VAL 0 0 0 0 0 VAL 4 2 3 4 1 VAL 8 4 3 8 1 VAL 12 4 7 12 1 VAL 16 4 11 16 1"],
  ["b", "fix a all ave/histo 1 2 4 0 6 3 v_t ave running beyond extra\nfix pr all print 4 \"VAL $(step) $(f_a[1]) $(f_a[2]) $(f_a[1][3]) $(f_a[5][3])\"\nrun 12", null, "VAL 0 0 0 0 0 VAL 4 2 0 0 0 VAL 8 4 0 0 0.5 VAL 12 6 0 0 0.66666666666666662966"],
  ["b", "fix a all ave/chunk 1 2 4 ch v_ta ave window 2\nrun 6\nrun 6\nprint \"VAL $(step) $(f_a[1][3]) $(f_a[2][3])\"", null, "VAL 12 9.5 0"],
  ["b", "fix a all ave/chunk 2 2 4 ch v_ta norm sample ave running\nrun 6\nminimize 0 0 3 10\nrun 10\nprint \"VAL $(step) $(f_a[1][3])\"", null, "VAL 19 9.6666666666666660745"],
  ["b", "group low id <= 4\nfix a low ave/atom 1 3 5 v_ta\nrun 4\nprint \"VAL $(F_a[1]) $(F_a[8])\"\nrun 1\nprint \"VAL $(F_a[1]) $(F_a[8])\"", null, "VAL 7 0 VAL 4 0"],
  ["b", "fix a all ave/atom 2 2 6 v_ta x\nrun 6\nprint \"VAL $(step) $(F_a[1][1])\"\nrun 2\nprint \"VAL $(F_a[1][1])\"", null, "VAL 6 5 VAL 5"],
  ["x", "variable w vector [1,2,3]*step", "Vector variable formula lacks opening or closing brace: [1,2,3]*step", ""],
  ["x", "variable w vector [1,,2]\nprint \"VAL $(v_w[2]) $(v_w[1])\"", null, "VAL 2 1"],
  ["x", "variable w vector [1, x]", "Expected floating point parameter instead of 'x' in input script or data file", ""],
  ["x", "variable w vector [ ]", "Expected floating point parameter instead of NULL or empty string in input script or data file", ""],
  ["x", "variable w vector \"[ 1 , 2 ]\"\nprint \"VAL $(v_w[2])\"", null, "VAL 2"],
  ["x", "variable w vector [1,2] + [3]", "Expected floating point parameter instead of '2] + [3' in input script or data file", ""],
  ["x", "variable w vector []\nprint \"VAL $(v_w[1])\"", "Variable w: Invalid index into vector-style variable", ""],
  ["x", "variable w vector [1e400]", "Floating point number 1e400 in input script or data file is out of range", ""],
];

describe('averaging fix schedules as native LAMMPS', () => {
  it.each(ROWS.map(([k, p, e, v]) => [`${k}: ${p.replace(/\n/g, ' / ')}`, k, p, e, v] as const))('%s', async (_name, k, p, error, vals) => {
    const got = await outcome(k, p);
    expect(got.error).toBe(error);
    expect(got.vals).toBe(vals);
  });
});
