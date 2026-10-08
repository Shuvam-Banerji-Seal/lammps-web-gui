import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Direct checks of fix ave/time (docs.lammps.org/fix_ave_time.html) and
 * fix print (docs.lammps.org/fix_print.html) on tiny systems: the documented
 * Nevery/Nrepeat/Nfreq sampling schedule (samples at T-(Nrepeat-1)*Nevery..T),
 * the ave one/running/window formulas, vector mode, the off keyword, file
 * headers/format and the print substitution, plus argument errors.
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
  const logs = events
    .filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log')
    .map((e) => e.text);
  return { session, thermo, logs, error, files };
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

describe('fix ave/time (fix_ave_time.html)', () => {
  it('samples the last Nrepeat steps before each Nfreq output (ave one)', async () => {
    // Nevery=2, Nrepeat=3, Nfreq=6: output at 6 averages steps 2,4,6; at 12: 8,10,12.
    // Step 0 has an incomplete window (only the step-0 sample) -> no output, f_1 = 0.
    const { thermo, files, error } = await runScript(`${SYS}
variable s equal step
fix 1 all ave/time 2 3 6 v_s file st.txt
thermo_style custom step f_1
thermo 6
run 12`);
    expect(error).toBeNull();
    expect(thermo.map((r) => r.f_1)).toEqual([0, 4, 10]);
    expect(files.get('st.txt')).toBe(
      '# Time-averaged data for fix 1\n# TimeStep v_s\n6 4\n12 10\n',
    );
  });

  it('ave running averages all outputs since the fix was defined', async () => {
    const { thermo, error } = await runScript(`${SYS}
variable s equal step
fix 1 all ave/time 1 1 5 v_s ave running
thermo_style custom step f_1
thermo 5
run 15`);
    expect(error).toBeNull();
    // outputs 0,5,10,15 -> running means 0, 2.5, 5, 7.5
    expect(thermo.map((r) => r.f_1)).toEqual([0, 2.5, 5, 7.5]);
  });

  it('ave window M averages the last M outputs, fewer when not available', async () => {
    const { thermo, error } = await runScript(`${SYS}
variable s equal step
fix 1 all ave/time 1 1 5 v_s ave window 2
thermo_style custom step f_1
thermo 5
run 15`);
    expect(error).toBeNull();
    expect(thermo.map((r) => r.f_1)).toEqual([0, 2.5, 7.5, 12.5]);
  });

  it('start skips everything before the start step', async () => {
    const { thermo, files, error } = await runScript(`${SYS}
variable s equal step
fix 1 all ave/time 1 1 5 v_s ave running start 5 file s5.txt
thermo_style custom step f_1
thermo 5
run 15`);
    expect(error).toBeNull();
    expect(thermo.map((r) => r.f_1)).toEqual([0, 5, 7.5, 10]);
    expect(files.get('s5.txt')).toBe(
      '# Time-averaged data for fix 1\n# TimeStep v_s\n5 5\n10 7.5\n15 10\n',
    );
  });

  it('off values are not averaged: most recent sample is stored', async () => {
    // v_q = step^2; output at 10 averages samples 5,10 -> 62.5, off value is 100.
    const { files, error } = await runScript(`${SYS}
variable q equal step*step
fix 1 all ave/time 5 2 10 v_q v_q off 2 file off.txt
run 20`);
    expect(error).toBeNull();
    expect(files.get('off.txt')).toBe(
      '# Time-averaged data for fix 1\n# TimeStep v_q v_q\n10 62.5 100\n20 312.5 400\n',
    );
  });

  it('vector mode averages each element; file has the 3-line header and row sections', async () => {
    const { thermo, files, error } = await runScript(`${SYS}
compute pp all pressure thermo_temp
fix 1 all ave/time 2 2 6 c_pp[1]
fix 2 all ave/time 2 2 6 c_pp mode vector file pv.txt
thermo_style custom step f_1 f_2[1] f_2[6]
thermo_modify norm no
run 12`);
    expect(error).toBeNull();
    // the same pressure component averaged in scalar and in vector mode agrees
    for (const r of thermo) expect(r['f_2[1]']).toBeCloseTo(r.f_1, 12);
    const text = files.get('pv.txt') ?? '';
    const lines = text.trim().split('\n');
    expect(lines[0]).toBe('# Time-averaged data for fix 2');
    expect(lines[1]).toBe('# TimeStep Number-of-rows');
    expect(lines[2]).toBe('# Row c_pp');
    expect(lines[3]).toMatch(/^6 6$/);
    expect(lines.slice(4, 10).map((l) => l.split(/\s+/)[0])).toEqual(['1', '2', '3', '4', '5', '6']);
    // 12 rows of values + 2 section lines "6 6"
    expect(lines.filter((l) => /^\d+ 6$/.test(l)).length).toBe(2);
  });

  it('wildcards expand to one value per vector element (mode scalar)', async () => {
    const { thermo, files, error } = await runScript(`${SYS}
compute pp all pressure thermo_temp
fix 1 all ave/time 2 1 6 c_pp[*] file pw.txt
thermo_style custom step f_1[3]
thermo_modify norm no
run 6`);
    expect(error).toBeNull();
    expect(Number.isFinite(thermo[1]['f_1[3]'])).toBe(true);
    expect(files.get('pw.txt')).toContain('# TimeStep c_pp[1] c_pp[2] c_pp[3] c_pp[4] c_pp[5] c_pp[6]');
  });

  it('rejects bad arguments', async () => {
    for (const [fix, msg] of [
      ['fix 1 all ave/time 3 2 8 v_s', 'Nfreq must be a multiple of Nevery'],
      ['fix 1 all ave/time 2 4 6 v_s', 'cannot exceed Nfreq'],
      ['fix 1 all ave/time 1 1 0 v_s', 'must be a positive integer'],
      ['fix 1 all ave/time 1 1 5', 'no input values'],
      ['fix 1 all ave/time 1 1 5 v_s file f.txt bogus 1', "unknown keyword 'bogus'"],
      ['fix 1 all ave/time 1 1 5 v_s off 2', 'off value 2 is out of range'],
      ['fix 1 all ave/time 1 1 5 v_s ave window 0', 'window M must be a positive integer'],
      ['fix 1 all ave/time 1 1 5 v_s mode sideways', 'mode must be scalar or vector'],
      ['fix 1 all ave/time 1 1 5 v_s overwrite', 'overwrite keyword can only be used with the ave running setting'],
      ['fix 1 all ave/time 1 1 5 v_s file f.txt append g.txt', 'file and append cannot both be used'],
      ['fix 1 all ave/time 1 1 5 v_s mode vector', 'is not a vector-style variable'],
    ] as const) {
      const { error } = await runScript(`${SYS}\nvariable s equal step\n${fix}\nrun 5`);
      expect(error, fix).toBeInstanceOf(Error);
      expect((error as Error).message, fix).toContain(msg);
    }
  });
});

describe('fix print (fix_print.html)', () => {
  it('prints at setup and every N steps with $-substitution, honoring title and screen', async () => {
    const { logs, files, error } = await runScript(`${SYS}
variable t equal step
fix 1 all print 5 "at ${'${t}'} $(step:%d)" file pr.txt screen no title "# hello"
run 12`);
    expect(error).toBeNull();
    expect(files.get('pr.txt')).toBe('# hello\nat 0 0\nat 5 5\nat 10 10\n');
    expect(logs.join('\n')).not.toContain('at 0');
  });

  it('screen yes logs the lines and the default title is used for the file', async () => {
    const { logs, files, error } = await runScript(`${SYS}
fix 1 all print 5 "step $(step)" file pr2.txt
run 11`);
    expect(error).toBeNull();
    expect(logs).toContain('step 0');
    expect(logs).toContain('step 5');
    expect(logs).toContain('step 10');
    expect(files.get('pr2.txt')).toBe(
      '# Fix print output for fix 1\nstep 0\nstep 5\nstep 10\n',
    );
  });

  it('keeps printing across runs; append extends the file', async () => {
    const { files, error } = await runScript(`${SYS}
fix 1 all print 5 "s $(step)" file pa.txt
run 12
run 6`);
    expect(error).toBeNull();
    // run 2 starts at step 12 (not a multiple of 5), so prints at 15 only
    expect(files.get('pa.txt')).toBe('# Fix print output for fix 1\ns 0\ns 5\ns 10\ns 15\n');
  });

  it('append appends to an existing file instead of overwriting', async () => {
    const { files, error } = await runScript(`${SYS}
fix 1 all print 10 "a $(step)" file ap.txt
run 10
unfix 1
fix 2 all print 10 "b $(step)" append ap.txt
run 10`);
    expect(error).toBeNull();
    expect(files.get('ap.txt')).toBe(
      '# Fix print output for fix 1\na 0\na 10\n# Fix print output for fix 2\nb 10\nb 20\n',
    );
  });

  it('rejects bad arguments', async () => {
    for (const [fix, msg] of [
      ['fix 1 all print 5', 'usage'],
      ['fix 1 all print 0 "x"', 'N must be a positive integer'],
      ['fix 1 all print 5 "x" screen maybe', 'screen must be yes or no'],
      ['fix 1 all print 5 "x" bogus yes', "unknown keyword 'bogus'"],
      ['fix 1 all print 5 "x" file a.txt append a.txt', 'file and append cannot both be used'],
      ['fix 1 all print v_nothere "x"', 'variable nothere does not exist'],
    ] as const) {
      const { error } = await runScript(`${SYS}\n${fix}\nrun 5`);
      expect(error, fix).toBeInstanceOf(Error);
      expect((error as Error).message, fix).toContain(msg);
    }
  });
});
