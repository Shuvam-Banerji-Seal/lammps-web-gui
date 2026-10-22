import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Run control, variables and output (audit WP2, area run-output): dump_modify maxfiles, the dump keywords of
 * other styles, thermo_modify warn, min_modify line spin values, getenv, logfreq3 past its last point, run pre no
 * after a non-run error, and the thermo tables of the log file. Numeric parity of the step-sequence variables,
 * round() and run pre no is in tests/oracle/w86run_*.in.
 */

const LJ = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

interface Out {
  error: string | null;
  logs: string[];
  files: Map<string, string>;
  session: Session;
}

const runText = async (text: string): Promise<Out> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error');
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { error: err?.message ?? error, logs, files, session };
};

describe('dump_modify maxfiles (dump_modify.html)', () => {
  it('keeps the last Fmax snapshot files of a "*" name, as native does', async () => {
    // Measured with native LAMMPS (black box): 5 snapshots of snap.*.dump with maxfiles 2 leave snap.3.dump and snap.4.dump
    const out = await runText(`${LJ}dump d all custom 1 snap.*.dump id x
dump_modify d maxfiles 2
run 0
run 4
`);
    expect(out.error).toBeNull();
    expect([...out.session.sys.files.keys()].filter((n) => n.startsWith('snap.')).sort()).toEqual(['snap.3.dump', 'snap.4.dump']);
  });

  it('needs a "*" in the file name', async () => {
    // Measured with native LAMMPS (black box): "Cannot use dump_modify maxfiles without * in dump file name"
    const out = await runText(`${LJ}dump d all custom 1 snap.dump id x
dump_modify d maxfiles 2
`);
    expect(out.error).toContain('Cannot use dump_modify maxfiles without * in dump file name');
  });
});

describe('dump_modify keywords of other dump styles (native errors)', () => {
  it('refuses precision, sfactor, tfactor and at for dump custom', async () => {
    // Measured with native LAMMPS (black box): "Unknown dump_modify keyword: precision" for atom, custom, xyz and extxyz
    for (const kw of ['precision 0.5', 'sfactor 0.5', 'tfactor 0.5', 'at 3']) {
      const out = await runText(`${LJ}dump d all custom 1 x.dump id x\ndump_modify d ${kw}\n`);
      expect(out.error, kw).toContain(`Unknown dump_modify keyword: ${kw.split(' ')[0]}`);
    }
  });

  it('refuses nfile and fileper without a % in the file name', async () => {
    // Measured with native LAMMPS (black box): "Cannot use dump_modify nfile without % in dump file name"
    const out = await runText(`${LJ}dump d all custom 1 x.dump id x\ndump_modify d nfile 2\n`);
    expect(out.error).toContain('Cannot use dump_modify nfile without % in dump file name');
  });
});

describe('thermo_modify warn (thermo_modify.html)', () => {
  it('accepts a number, ignore, reset and default, and rejects other words', async () => {
    // Measured with native LAMMPS (black box): "warn foo" stops with "Expected integer parameter instead of 'foo'"
    expect((await runText(`${LJ}thermo_modify warn 5\nthermo_modify warn ignore\nthermo_modify warn reset\nthermo_modify warn default\nrun 0\n`)).error).toBeNull();
    const bad = await runText(`${LJ}thermo_modify warn foo\n`);
    expect(bad.error).toContain("Expected integer parameter instead of 'foo'");
  });

  it('limits the printed warnings and names the limit once', async () => {
    // Two nve fixes make every run warn "One or more atoms are time integrated more than once". Measured with native
    // LAMMPS (black box): with warn 2 the third warning is the one that reports the limit ("Too many warnings: 3 vs 2")
    const twice = 'WARNING: One or more atoms are time integrated more than once';
    const limited = await runText(`${LJ}fix 1 all nve\nfix 2 all nve\nthermo_modify warn 2\nrun 0\nrun 0\nrun 0\nrun 0\n`);
    expect(limited.logs.filter((l) => l === twice).length).toBe(2);
    expect(limited.logs).toContain('WARNING: Too many warnings: 3 vs 2. All future warnings will be suppressed');
    const ignored = await runText(`${LJ}fix 1 all nve\nfix 2 all nve\nthermo_modify warn ignore\nrun 0\n`);
    expect(ignored.logs.some((l) => l.startsWith('WARNING'))).toBe(false);
  });
});

describe('min_modify line spin values and getenv', () => {
  it('refuses spin_cubic and spin_none on a non-spin minimization by name', async () => {
    // Measured with native LAMMPS (black box): both values stop a non-spin minimization with a segmentation fault;
    // min_modify itself accepts them for every min style (src/min.cpp modify_params)
    const out = await runText(`${LJ}min_modify line spin_cubic\nminimize 0 1e-6 100 1000\n`);
    expect(out.error).toContain('min_modify line spin_cubic is only for the spin minimizers');
    expect((await runText(`${LJ}min_modify line spin_none\n`)).error).toBeNull();
  });

  it('refuses the getenv variable style instead of returning an empty value', async () => {
    const out = await runText(`${LJ}variable h getenv HOME\n`);
    expect(out.error).toContain('variable style getenv is not supported by the browser engine');
  });
});

describe('variable logfreq3 past its last point (variable.html)', () => {
  it('stops with the native message when the variable is evaluated after z', async () => {
    // Measured with native LAMMPS (black box): logfreq3(1,3,10) stops with "Calls to variable exceeded limit" at step 11
    const out = await runText(`${LJ}variable c equal logfreq3(1,3,10)
thermo_style custom step v_c
thermo 1
run 11
`);
    expect(out.error).toContain('Calls to variable exceeded limit');
  });
});

describe('run start/stop checks and run pre no (run.html)', () => {
  it('rejects a bad start before the setup, so the next run starts clean', async () => {
    const out = await runText(`${LJ}run 10 start 5\n`);
    expect(out.error).toContain('run start cannot be after the current timestep');
    expect(out.session.sys.run.inRun).toBe(false);
  });

  it('refuses run pre no after a new compute, and accepts it after a displacement', async () => {
    // run.html: "it is an error to use the pre no option" after a change such as a new fix or compute
    const changed = await runText(`${LJ}fix 1 all nve\nrun 2\ncompute ke all ke\nrun 2 pre no\n`);
    expect(changed.error).toContain('run pre no after a change');
    const moved = await runText(`${LJ}fix 1 all nve\nrun 2\ndisplace_atoms all move 0.01 0 0 units box\nrun 2 pre no\n`);
    expect(moved.error).toBeNull();
  });

  it('refuses run pre no with run_style respa after a run', async () => {
    // Measured with native LAMMPS (black box): "Run flag 'pre no' not compatible with r-RESPA"
    const out = await runText(`${LJ}fix 1 all nve\nrun 2\nrun_style respa 2 2 pair 1\nrun 2 pre no\n`);
    expect(out.error).toContain("Run flag 'pre no' not compatible with r-RESPA");
  });
});

describe('log file thermo tables (thermo_style, thermo_modify format and line)', () => {
  it('writes the header and one line per thermo row to the log file', async () => {
    const out = await runText(`${LJ}fix 1 all nve\nlog run.log\nthermo_style custom step pe\nthermo_modify format float %.6f\nthermo 2\nrun 4\n`);
    expect(out.error).toBeNull();
    const lines = (out.files.get('run.log') ?? '').split('\n').filter((l) => l.trim() !== '');
    const header = lines.findIndex((l) => /^\s*Step\s+PotEng\s*$/.test(l));
    expect(header).toBeGreaterThanOrEqual(0);
    const rows = lines.slice(header + 1).filter((l) => /^\s*[-0-9]/.test(l));
    expect(rows.length).toBe(3);
    expect(rows[0].trim().split(/\s+/)[0]).toBe('0');
    expect(rows[2].trim().split(/\s+/)[0]).toBe('4');
    expect(rows[0].trim().split(/\s+/)[1]).toMatch(/^-?\d+\.\d{6}$/);
  });

  it('writes a yaml block for thermo_style yaml', async () => {
    const out = await runText(`${LJ}fix 1 all nve\nlog run.log\nthermo_style yaml\nthermo 1\nrun 1\n`);
    const text = out.files.get('run.log') ?? '';
    expect(text).toMatch(/^---$/m);
    expect(text).toMatch(/^keywords: \[ 'Step', 'Temp', 'E_pair', 'E_mol', 'TotEng', 'Press' \]$/m);
    expect(text).toMatch(/^\.\.\.$/m);
  });

  it('is not written to the screen (the rows reach the notebook as thermo events)', async () => {
    const out = await runText(`${LJ}fix 1 all nve\nlog run.log\nthermo 1\nrun 1\n`);
    expect(out.logs.some((l) => /^\s*Step\s/.test(l))).toBe(false);
  });
});
