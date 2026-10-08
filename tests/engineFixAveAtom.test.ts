import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Direct checks of fix ave/atom (docs.lammps.org/fix_ave_atom.html) and
 * fix ave/histo (docs.lammps.org/fix_ave_histo.html) on tiny systems: the
 * documented Nevery/Nrepeat/Nfreq sampling schedule (samples at
 * T-(Nrepeat-1)*Nevery..T), group zeroing for ave/atom, the histogram bin
 * rules (boundary values to the lower bin, beyond ignore/end/extra), the ave
 * one/running modes, file headers/format and argument errors.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { session, thermo, error, files };
};

const SYS = `
units           lj
atom_style      atomic
boundary        p p p
lattice         sc 1.0
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
`;

interface Frame {
  step: number;
  cols: string[];
  rows: number[][];
}

const dumpFrames = (text: string): Frame[] => {
  const lines = text.trim().split('\n');
  const out: Frame[] = [];
  let step = -1;
  for (let k = 0; k < lines.length; k++) {
    if (lines[k] === 'ITEM: TIMESTEP') {
      step = Number(lines[k + 1]);
      k++;
      continue;
    }
    if (lines[k].startsWith('ITEM: ATOMS')) {
      const cols = lines[k].trim().split(/\s+/).slice(2);
      const rows: number[][] = [];
      k++;
      while (k < lines.length && !lines[k].startsWith('ITEM:')) {
        rows.push(lines[k].trim().split(/\s+/).map(Number));
        k++;
      }
      k--;
      out.push({ step, cols, rows });
    }
  }
  return out;
};

describe('fix ave/atom (fix_ave_atom.html)', () => {
  it('averages the last Nrepeat samples before each Nfreq output and zeroes atoms outside the group', async () => {
    // Nevery=2, Nrepeat=3, Nfreq=6: output at 6 averages steps 2,4,6 -> 4; at 12: 8,10,12 -> 10.
    // Step 0 has an incomplete window (only the step-0 sample) -> no output, f_1 = 0.
    const { files, error } = await runScript(`${SYS}
variable t atom step
group upper id 5:8
fix 1 upper ave/atom 2 3 6 v_t
dump d all custom 6 da.dump id f_1
dump_modify d sort id
thermo_style custom step
thermo 6
run 12`);
    expect(error).toBeNull();
    const frames = dumpFrames(files.get('da.dump')!);
    expect(frames.map((f) => f.step)).toEqual([0, 6, 12]);
    expect(frames[0].rows.map((r) => r[1])).toEqual(Array(8).fill(0));
    const at = (f: Frame, id: number): number => f.rows[id - 1][1];
    for (const id of [1, 2, 3, 4]) {
      expect(at(frames[1], id)).toBe(0);
      expect(at(frames[2], id)).toBe(0);
    }
    for (const id of [5, 6, 7, 8]) {
      expect(at(frames[1], id)).toBeCloseTo(4, 12);
      expect(at(frames[2], id)).toBeCloseTo(10, 12);
    }
  });

  it('produces a per-atom array with one column per input value', async () => {
    const { files, error } = await runScript(`${SYS}
variable ke atom mass*(vx*vx+vy*vy+vz*vz)
velocity all set 0.1 0.2 0.3 units box
fix 1 all ave/atom 1 1 5 vx v_ke
dump d all custom 5 da.dump id f_1[1] f_1[2]
dump_modify d sort id
run 5`);
    expect(error).toBeNull();
    const frames = dumpFrames(files.get('da.dump')!);
    expect(frames[0].cols).toEqual(['id', 'f_1[1]', 'f_1[2]']);
    for (const frame of frames) {
      for (const row of frame.rows) {
        expect(row[1]).toBeCloseTo(0.1, 12);
        expect(row[2]).toBeCloseTo(0.14, 12);
      }
    }
  });

  it('rejects bad schedules and non-per-atom inputs', async () => {
    for (const bad of [
      'fix 1 all ave/atom 4 2 6 vx',
      'fix 1 all ave/atom 2 4 6 vx',
      'fix 1 all ave/atom 0 1 6 vx',
      'fix 1 all ave/atom 1 1 5 q',
      'fix 1 all ave/atom 1 1 5 c_thermo_temp',
      'fix 1 all ave/atom 1 1 5',
    ]) {
      const { error } = await runScript(`${SYS}\n${bad}`);
      expect(error, bad).not.toBeNull();
    }
  });
});

describe('fix ave/histo (fix_ave_histo.html)', () => {
  it('histograms per-atom values in mode vector with the documented file format', async () => {
    // v_q = x + 2*y takes 0,1,2,3 (twice each, z free); bins of width 1 over [-0.5, 3.5].
    // Nrepeat=1: a complete window at step 0, so sections at 0 and 5.
    const { thermo, files, error } = await runScript(`${SYS}
variable q atom "x + 2*y"
fix 2 all ave/histo 1 1 5 -0.5 3.5 4 v_q mode vector file h.txt
thermo_style custom step f_2[1] f_2[2] f_2[3] f_2[4]
thermo 5
run 5`);
    expect(error).toBeNull();
    expect(thermo.map((r) => [r.step, r['f_2[1]'], r['f_2[2]'], r['f_2[3]'], r['f_2[4]']])).toEqual([
      [0, 8, 0, 0, 3],
      [5, 8, 0, 0, 3],
    ]);
    const lines = files.get('h.txt')!.trim().split('\n');
    expect(lines.slice(0, 3)).toEqual([
      '# Histogrammed data for fix 2',
      '# TimeStep Number-of-bins Total-counts Missing-counts Min-value Max-value',
      '# Bin Coord Count Count/Total',
    ]);
    expect(lines.slice(3)).toEqual([
      '0 4 8 0 0 3',
      '1 0 2 0.25',
      '2 1 2 0.25',
      '3 2 2 0.25',
      '4 3 2 0.25',
      '5 4 8 0 0 3',
      '1 0 2 0.25',
      '2 1 2 0.25',
      '3 2 2 0.25',
      '4 3 2 0.25',
    ]);
  });

  it('assigns values on a bin boundary to the upper bin (mode scalar, global input; measured with native)', async () => {
    // Nevery=1, Nrepeat=5, Nfreq=5: the step-0 window (just the value 0) is
    // incomplete and discarded; the window at 5 holds samples of step 1..5 ->
    // values 1,2,3,4,5; bins of width 3 over [0,12]: 1,2 -> bin 1; 3 sits exactly
    // on the boundary and native puts it in the upper bin 2 (the docs' "lower"
    // sentence is not what native does); 4,5 -> bin 2.
    const { thermo, files, error } = await runScript(`${SYS}
variable s equal step
fix 2 all ave/histo 1 5 5 0 12 4 v_s file hs.txt
thermo_style custom step f_2[1] f_2[2] f_2[3] f_2[4]
thermo 5
run 5`);
    expect(error).toBeNull();
    expect(thermo.map((r) => [r.step, r['f_2[1]'], r['f_2[2]'], r['f_2[3]'], r['f_2[4]']])).toEqual([
      [0, 0, 0, 0, 0],
      [5, 5, 0, 1, 5],
    ]);
    expect(files.get('hs.txt')!.trim().split('\n').slice(3)).toEqual([
      '5 4 5 0 1 5',
      '1 1.5 2 0.4',
      '2 4.5 3 0.6',
      '3 7.5 0 0',
      '4 10.5 0 0',
    ]);
  });

  it('honors the beyond keyword: ignore, end and extra', async () => {
    // v_q values 0,1,2,3 (twice each); window [0.5, 2.5] with 2 bins of width 1
    // (bin centers 1.0 and 2.0; the extra bins print lo and hi).
    const base = (beyond: string, extra: boolean): string => `${SYS}
variable q atom "x + 2*y"
fix 2 all ave/histo 1 1 5 0.5 2.5 2 v_q mode vector ${beyond}
variable c1 equal f_2[1][1]
variable c2 equal f_2[2][1]
variable n1 equal f_2[1][3]
variable n2 equal f_2[2][3]
${extra ? 'variable c3 equal f_2[3][1]\nvariable c4 equal f_2[4][1]\nvariable n3 equal f_2[3][3]\nvariable n4 equal f_2[4][3]' : ''}
thermo_style custom step v_c1 v_c2 v_n1 v_n2 ${extra ? 'v_c3 v_c4 v_n3 v_n4' : ''} f_2[1] f_2[2] f_2[3] f_2[4]
thermo 5
run 5`;
    const ignore = await runScript(base('beyond ignore', false));
    expect(ignore.error).toBeNull();
    expect(ignore.thermo[0]['f_2[1]']).toBe(4);
    expect(ignore.thermo[0]['f_2[2]']).toBe(4);
    expect(ignore.thermo[0]['f_2[3]']).toBe(0);
    expect(ignore.thermo[0]['f_2[4]']).toBe(3);
    expect(ignore.thermo[0].v_c1).toBeCloseTo(1, 12);
    expect(ignore.thermo[0].v_c2).toBeCloseTo(2, 12);
    expect(ignore.thermo[0].v_n1).toBeCloseTo(0.5, 12);
    expect(ignore.thermo[0].v_n2).toBeCloseTo(0.5, 12);
    const end = await runScript(base('beyond end', false));
    expect(end.error).toBeNull();
    expect(end.thermo[0]['f_2[1]']).toBe(8);
    expect(end.thermo[0]['f_2[2]']).toBe(0);
    expect(end.thermo[0].v_n1).toBeCloseTo(0.5, 12);
    expect(end.thermo[0].v_n2).toBeCloseTo(0.5, 12);
    const extra = await runScript(base('beyond extra', true));
    expect(extra.error).toBeNull();
    expect(extra.thermo[0].v_c1).toBeCloseTo(0.5, 12);
    expect(extra.thermo[0].v_c2).toBeCloseTo(1, 12);
    expect(extra.thermo[0].v_c3).toBeCloseTo(2, 12);
    expect(extra.thermo[0].v_c4).toBeCloseTo(2.5, 12);
    expect(extra.thermo[0]['f_2[1]']).toBe(8);
    expect(extra.thermo[0].v_n1).toBeCloseTo(0.25, 12);
    expect(extra.thermo[0].v_n2).toBeCloseTo(0.25, 12);
    expect(extra.thermo[0].v_n3).toBeCloseTo(0.25, 12);
    expect(extra.thermo[0].v_n4).toBeCloseTo(0.25, 12);
  });

  it('ave running averages the histograms produced so far', async () => {
    // Nevery=1, Nrepeat=5, Nfreq=5: the step-0 window is discarded; windows
    // [1..5] and [6..10] have bin counts [3,2,0,0] and [0,1,3,1] (boundary
    // values to the lower bin). Running: [3,2,0,0] then [1.5,1.5,1.5,0.5].
    const { thermo, error } = await runScript(`${SYS}
variable s equal step
fix 2 all ave/histo 1 5 5 0 12 4 v_s ave running
thermo_style custom step f_2[1] f_2[2] f_2[3] f_2[4]
thermo 5
run 10`);
    expect(error).toBeNull();
    expect(thermo.map((r) => [r.step, r['f_2[1]'], r['f_2[2]'], r['f_2[3]'], r['f_2[4]']])).toEqual([
      [0, 0, 0, 0, 0],
      [5, 5, 0, 1, 5],
      [10, 5, 0, 6, 10],
    ]);
  });

  it('honors title overrides and the group filter for per-atom inputs', async () => {
    const { files, thermo, error } = await runScript(`${SYS}
variable q atom "x + 2*y"
group quarter id 1:2
fix 2 quarter ave/histo 1 1 5 -0.5 3.5 4 v_q mode vector file ht.txt title1 "My output values"
thermo_style custom step f_2[1]
thermo 5
run 5`);
    expect(error).toBeNull();
    expect(thermo[0]['f_2[1]']).toBe(2);
    expect(files.get('ht.txt')!.split('\n')[0]).toBe('My output values');
  });

  it('rejects bad arguments', async () => {
    for (const bad of [
      'fix 1 all ave/histo 4 1 5 0 1 10 vx',
      'fix 1 all ave/histo 2 3 5 0 1 10 vx',
      'fix 1 all ave/histo 1 1 5 1 0 10 vx',
      'fix 1 all ave/histo 1 1 5 0 1 0 vx',
      'fix 1 all ave/histo 1 1 5 0 1 10 vx mode wrong',
      'fix 1 all ave/histo 1 1 5 0 1 10 vx kind local',
      'fix 1 all ave/histo 1 1 5 0 1 10 vx beyond wrong',
      'fix 1 all ave/histo 1 1 5 0 1 10 vx overwrite',
      'fix 1 all ave/histo 1 1 5 0 1 10 vx file a.txt file b.txt',
      'fix 1 all ave/histo 1 1 5 0 1 10 c_thermo_temp vx',
      'fix 1 all ave/histo 1 1 5 0 1 10 v_nosuch mode scalar',
      'fix 1 all ave/histo 1 1 5 0 1 10',
    ]) {
      const { error } = await runScript(`${SYS}\n${bad}`);
      expect(error, bad).not.toBeNull();
    }
  });
});
