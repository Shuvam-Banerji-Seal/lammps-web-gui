import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * shell rm / shell mv on the session file store (commands/misc.ts).
 * shell.html: "A few simple file-based shell commands are supported directly,
 * in Unix-style syntax."; "*rm* args = [-f] file1 file2 ..." and
 * "*mv* args = old new". Everything else (cd, mkdir, rmdir, putenv, arbitrary
 * commands) is refused because a browser has no shell and no directories.
 *
 * Measured native behaviour (black box) is recorded in commands/misc.ts.
 */

const newSession = () => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  return { session, events };
};

const textOf = (events: EngineEvent[]): string[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);

const LJ_BOX = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
timestep        0.004
`;

describe('shell rm', () => {
  it('deletes a file the session wrote, so read_restart then fails with the missing-file error', async () => {
    const { session } = newSession();
    await session.execute(`${LJ_BOX}write_restart gone.restart\n`);
    expect(session.sys.files.has('gone.restart')).toBe(true);

    await session.execute('shell rm gone.restart\n');
    expect(session.sys.files.has('gone.restart')).toBe(false);

    await expect(session.execute('clear\nread_restart gone.restart\n')).rejects.toThrow(/cannot open file gone\.restart/);
  });

  it('warns once per missing file without -f, and continues', async () => {
    const { session, events } = newSession();
    await session.execute('shell rm nope1.txt nope2.txt\nprint "reached the next command"\n');
    const logs = textOf(events);
    expect(logs).toContain(
      "WARNING: Shell command 'rm nope1.txt' failed with error 'No such file or directory'",
    );
    expect(logs).toContain(
      "WARNING: Shell command 'rm nope2.txt' failed with error 'No such file or directory'",
    );
    expect(logs).toContain('reached the next command');
  });

  it('is silent for a missing file with -f, and for -f with no file names', async () => {
    const { session, events } = newSession();
    await session.execute('shell rm -f nope.txt\nshell rm -f\nprint "ok"\n');
    expect(textOf(events).filter((l) => l.startsWith('WARNING:'))).toEqual([]);
    expect(textOf(events)).toContain('ok');
  });

  it('errors when no file name is given and there is no -f (native message)', async () => {
    const { session } = newSession();
    await expect(session.execute('shell rm\n')).rejects.toThrow(/shell rm: missing argument/);
  });
});

describe('shell mv', () => {
  it('renames a restart file, which read_restart reads under the new name only', async () => {
    const { session } = newSession();
    await session.execute(`${LJ_BOX}write_restart old.restart\n`);
    await session.execute('shell mv old.restart new.restart\n');
    expect(session.sys.files.has('old.restart')).toBe(false);
    expect(session.sys.files.has('new.restart')).toBe(true);

    await session.execute('clear\nread_restart new.restart\n');
    expect(session.sys.state.step).toBe(0);

    await session.execute('clear\n');
    await expect(session.execute('read_restart old.restart\n')).rejects.toThrow(/cannot open file old\.restart/);
  });

  it('overwrites an existing destination, silently', async () => {
    const { session, events } = newSession();
    session.addFile('src.txt', 'NEW\n');
    session.addFile('dst.txt', 'OLD\n');
    await session.execute('shell mv src.txt dst.txt\n');
    expect(session.sys.readFile('dst.txt')).toBe('NEW\n');
    expect(session.sys.files.has('src.txt')).toBe(false);
    expect(textOf(events).filter((l) => l.startsWith('WARNING:'))).toEqual([]);
  });

  it('warns and continues when the source is missing', async () => {
    const { session, events } = newSession();
    await session.execute('shell mv nope.txt new.txt\nprint "ok"\n');
    expect(textOf(events)).toContain(
      "WARNING: Shell command 'mv nope.txt new.txt' failed with error 'No such file or directory'",
    );
    expect(textOf(events)).toContain('ok');
  });

  it('errors on the wrong argument count', async () => {
    const { session } = newSession();
    await expect(session.execute('shell mv only.txt\n')).rejects.toThrow(/shell mv: expected 2 arguments/);
  });
});

describe('shell refuses what a browser cannot do', () => {
  it.each([
    ['shell cd sub1', /shell cd/],
    ['shell mkdir tmp1 tmp2', /shell mkdir/],
    ['shell rmdir tmp1', /shell rmdir/],
    ['shell putenv LAMMPS_POTENTIALS=../../potentials', /shell putenv/],
    ['shell my_setup file1 10', /shell my_setup/],
  ])('%s is a StyleError naming the form', async (line, pattern) => {
    const { session } = newSession();
    await expect(session.execute(`${line}\n`)).rejects.toThrow(pattern);
    await expect(session.execute(`${line}\n`)).rejects.toThrow(/no operating-system shell and no directories/);
  });

  it('shell with no command is refused', async () => {
    const { session } = newSession();
    await expect(session.execute('shell\n')).rejects.toThrow(/shell: missing command/);
  });

  it('shell is listed as a supported command (its rm and mv built-ins run)', async () => {
    const { SUPPORTED_COMMANDS } = await import('../src/engine/interpreter');
    expect(SUPPORTED_COMMANDS).toContain('shell');
  });
});
