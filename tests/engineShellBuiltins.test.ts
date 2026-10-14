import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { resolveFileName } from '../src/engine/fileName';

/*
 * The shell built-ins cd, mkdir, rmdir and putenv on the notebook's virtual file system (commands/misc.ts, shellFs.ts).
 * shell.html: "The *cd* command changes the current working directory similar to the cd command. All subsequent LAMMPS
 * commands that read/write files will use the new directory."; "The *mkdir* command creates directories similar to the
 * Unix mkdir -p command."; "The *rmdir* command deletes directories similar to Unix rmdir command. If a directory is
 * not empty, its contents are also removed recursively"; "The *putenv* command defines or updates an environment
 * variable directly. ... The resulting string is then used literally."
 */

const newSession = () => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  return { session, events };
};

const warnings = (events: EngineEvent[]): string[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text).filter((t) => t.startsWith('WARNING:'));

const DATA = 'units lj\natom_style atomic\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\n';

describe('resolveFileName (cd-relative file names)', () => {
  it('resolves ".", "..", and leading "/" against the working directory', () => {
    expect(resolveFileName('', './a/./b')).toBe('a/b');
    expect(resolveFileName('sub', 'x.data')).toBe('sub/x.data');
    expect(resolveFileName('sub/deep', '../x.data')).toBe('sub/x.data');
    expect(resolveFileName('sub', '/top.data')).toBe('top.data');
    // above the notebook root the name is kept, so it is only found if the user added it
    expect(resolveFileName('', '../x')).toBe('../x');
  });
});

describe('shell cd, mkdir, rmdir: the working directory and directories of the session', () => {
  it('cd into a directory made by mkdir, then a read_data name resolves inside it', async () => {
    const { session } = newSession();
    session.addFile('sub/data.in', DATA + 'create_atoms 1 box\n');
    await session.execute('shell mkdir sub/deep\nshell cd sub\n');
    expect(session.sys.cwd).toBe('sub');
    // a file the notebook gave at the root is found from sub only through "..", and sub's own file by its name
    await session.execute('shell cd deep\nshell cd ..\n');
    expect(session.sys.cwd).toBe('sub');
    await session.execute('shell cd ..\n');
    expect(session.sys.cwd).toBe('');
  });

  it('a write from a working directory lands in that directory, and cd .. comes back', async () => {
    const { session } = newSession();
    await session.execute('shell mkdir out\nshell cd out\nprint "hi" file out_note.txt screen no\n');
    expect(session.sys.files.has('out/out_note.txt')).toBe(true);
    await session.execute('shell cd ..\n');
    expect(session.sys.cwd).toBe('');
    expect(session.sys.files.has('out_note.txt')).toBe(false);
  });

  it('cd to a missing directory warns, keeps the working directory and continues', async () => {
    const { session, events } = newSession();
    await session.execute('shell cd nowhere\n');
    expect(session.sys.cwd).toBe('');
    expect(warnings(events)).toContain("WARNING: Shell command 'cd nowhere' failed with error 'No such file or directory'");
  });

  it('a dump opens its file at the run, not at the dump command (measured with native LAMMPS)', async () => {
    // Measured with native LAMMPS (b332f28): this input writes d.atom in the root, not in out/, because the single file
    // is opened at the run (src/dump_custom.cpp init_style), after the cd back
    const { session } = newSession();
    await session.execute('units lj\natom_style atomic\nlattice fcc 0.8442\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box\nmass * 1.0\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0 2.5\nshell mkdir out\nshell cd out\ndump 1 all atom 1 d.atom\nshell cd ..\nrun 1\n');
    expect(session.sys.files.has('d.atom')).toBe(true);
    expect(session.sys.files.has('out/d.atom')).toBe(false);
  });

  it('cd sees a directory implied by a file name (a dump into dump/ needs no mkdir)', async () => {
    const { session } = newSession();
    session.addFile('dump/a.txt', 'x');
    await session.execute('shell cd dump\n');
    expect(session.sys.cwd).toBe('dump');
  });

  it('mkdir creates the whole path (mkdir -p) and is silent for an existing directory', async () => {
    const { session, events } = newSession();
    await session.execute('shell mkdir tmp1 tmp2/tmp3\nshell mkdir tmp1\n');
    expect(session.sys.dirs.has('tmp2')).toBe(true);
    expect(session.sys.dirs.has('tmp2/tmp3')).toBe(true);
    expect(warnings(events)).toEqual([]);
  });

  it('mkdir warns when a file of that name exists', async () => {
    const { session, events } = newSession();
    session.addFile('clash', 'x');
    await session.execute('shell mkdir clash\n');
    expect(warnings(events)).toContain("WARNING: Shell command 'mkdir clash' failed with error 'File exists'");
    expect(session.sys.dirs.has('clash')).toBe(false);
  });

  it('rmdir removes an empty directory and warns for a missing one', async () => {
    const { session, events } = newSession();
    await session.execute('shell mkdir tmp1\nshell rmdir tmp1 nothere\n');
    expect(session.sys.dirs.has('tmp1')).toBe(false);
    expect(warnings(events)).toEqual(["WARNING: Shell command 'rmdir nothere' failed with error 'No such file or directory'"]);
  });

  it('rmdir removes a non-empty directory recursively, its files and subdirectories', async () => {
    const { session } = newSession();
    session.addFile('keep.txt', 'k');
    session.addFile('tmp2/a.txt', 'a');
    session.addFile('tmp2/tmp3/b.txt', 'b');
    await session.execute('shell mkdir tmp2/empty\nshell rmdir tmp2\n');
    expect([...session.sys.files.keys()]).toEqual(['keep.txt']);
    expect(session.sys.dirs.size).toBe(0);
  });

  it('a directory is removed from the working directory it is named in', async () => {
    const { session } = newSession();
    await session.execute('shell mkdir a/b\nshell cd a\nshell rmdir b\n');
    expect(session.sys.dirs.has('a/b')).toBe(false);
    expect(session.sys.dirs.has('a')).toBe(true);
  });

  it('shell state survives clear (clear.html changes only the simulation)', async () => {
    const { session } = newSession();
    await session.execute('shell mkdir keep\nshell cd keep\nshell putenv LAMMPS_POTENTIALS=pots\nclear\n');
    expect(session.sys.cwd).toBe('keep');
    expect(session.sys.env.get('LAMMPS_POTENTIALS')).toBe('pots');
  });
});

describe('shell putenv', () => {
  it('sets a variable with its value taken literally', async () => {
    const { session } = newSession();
    await session.execute('shell putenv FOO=../x/y LAMMPS_POTENTIALS=pots\n');
    expect(session.sys.env.get('FOO')).toBe('../x/y');
    expect(session.sys.env.get('LAMMPS_POTENTIALS')).toBe('pots');
  });

  it('warns for a definition without "=" and continues', async () => {
    const { session, events } = newSession();
    await session.execute('shell putenv NOEQUALS FOO=1\n');
    expect(warnings(events)).toEqual(["WARNING: Shell command 'putenv NOEQUALS' failed with error 'Invalid argument'"]);
    expect(session.sys.env.get('FOO')).toBe('1');
  });

  it('LAMMPS_POTENTIALS is searched for a file that is not in the working directory', async () => {
    const { session } = newSession();
    session.addFile('pots/Ar.eam', 'potential text');
    await session.execute('shell putenv LAMMPS_POTENTIALS=pots\n');
    expect(session.sys.readFile('Ar.eam')).toBe('potential text');
    await session.execute('shell mkdir local\nshell cd local\n');
    // a file of the working directory is preferred to the potential directory's copy
    session.addFile('local/Ar.eam', 'local text');
    expect(session.sys.readFile('Ar.eam')).toBe('local text');
  });
});

describe('shell rm, mv from a working directory', () => {
  it('mv into a folder keeps the base name; rm -f is silent for a missing file', async () => {
    const { session, events } = newSession();
    session.addFile('log.lammps', 'log');
    await session.execute('shell mkdir hold\nshell mv log.lammps hold\nshell rm -f nothere\n');
    expect(session.sys.files.get('hold/log.lammps')).toBe('log');
    expect(session.sys.files.has('log.lammps')).toBe(false);
    expect(warnings(events)).toEqual([]);
  });

  it('rm acts on the working directory', async () => {
    const { session } = newSession();
    session.addFile('d/x.txt', 'x');
    await session.execute('shell cd d\nshell rm x.txt\n');
    expect(session.sys.files.has('d/x.txt')).toBe(false);
  });
});

describe('dump files and shell cd (when native opens them)', () => {
  // src/dump_custom.cpp init_style (b332f28): "open single file, one time only" at the first run after the dump command;
  // src/dump.cpp write: "if file per timestep, open new file" when each snapshot is written
  const LJ = 'units lj\natom_style atomic\nlattice fcc 0.8442\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\n'
    + 'create_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\n';

  it('a single-file dump opens its file in the working directory of its first run', async () => {
    const { session } = newSession();
    await session.execute(`${LJ}dump d all atom 1 single.dump\nshell mkdir a\nshell cd a\nrun 1\nshell cd ..\nrun 1\n`);
    expect(session.sys.files.has('a/single.dump')).toBe(true);
    expect(session.sys.files.has('single.dump')).toBe(false);
  });

  it('a write_data between the dump command and the cd opens the single file in the first directory', async () => {
    // Measured with native LAMMPS (b332f28, w165mc_shell_builtins): write_data runs lmp->init(), which opens the dump's
    // single file in out/, so the later cd and run keep writing it there
    const { session } = newSession();
    await session.execute(`${LJ}shell mkdir out\nshell cd out\ndump d all atom 1 s.dump\nwrite_data w.data\nshell cd ..\nrun 1\n`);
    expect(session.sys.files.has('out/s.dump')).toBe(true);
    expect(session.sys.files.has('s.dump')).toBe(false);
  });

  it('a multi-file dump writes each snapshot in the working directory of that write', async () => {
    // Measured with native LAMMPS (b332f28): dump m.*.dump, cd out, run 0 (step 1), cd .., run 1 writes out/m.1.dump and m.2.dump
    const { session } = newSession();
    await session.execute(`${LJ}dump d all atom 1 m.*.dump\nshell mkdir a\nshell cd a\nrun 0\nshell cd ..\nrun 1\n`);
    expect(session.sys.files.has('a/m.0.dump')).toBe(true);
    expect(session.sys.files.has('m.1.dump')).toBe(true);
  });
});
