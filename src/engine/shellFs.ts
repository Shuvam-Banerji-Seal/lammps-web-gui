import type { System } from './system';
import { resolveFileName } from './fileName';

/*
 * The shell built-ins cd, mkdir, rmdir and putenv on the notebook's virtual file system (shell.html: "A few simple
 * file-based shell commands are supported directly, in Unix-style syntax"). The session's files are a flat map of
 * names (sys.files); a directory is a name in sys.dirs, or a directory that a file name or another directory
 * implies (a file "dump/x.vtk" makes "dump" exist, as the dump command writes into it). The working directory
 * sys.cwd is relative to the notebook root (''). Every function here acts on one replica's System; the session
 * applies it to all replicas.
 */

/** True when the root-relative name is a directory of the session (the root itself always is). */
export const isDirectory = (sys: System, key: string): boolean => {
  if (key === '' || sys.dirs.has(key)) return true;
  const below = `${key}/`;
  for (const d of sys.dirs) if (d.startsWith(below)) return true;
  for (const f of sys.files.keys()) if (f.startsWith(below)) return true;
  return false;
};

/** mkdir -p: the directory and every directory above it (shell.html: "similar to the Unix mkdir -p command"). */
export const makeDirectories = (sys: System, key: string): void => {
  if (key === '') return;
  const segs = key.split('/');
  for (let k = 1; k <= segs.length; k++) sys.dirs.add(segs.slice(0, k).join('/'));
};

/** rmdir with the contents removed (shell.html: "If a directory is not empty, its contents are also removed recursively"). */
export const removeDirectory = (sys: System, key: string): void => {
  const below = `${key}/`;
  for (const f of [...sys.files.keys()]) if (f.startsWith(below)) sys.files.delete(f);
  for (const d of [...sys.dirs]) if (d === key || d.startsWith(below)) sys.dirs.delete(d);
};

/** The root-relative key for a shell argument, taken from the working directory. */
export const shellKey = (sys: System, name: string): string => resolveFileName(sys.cwd, name);
