/**
 * Session-file names. A file written or uploaded as "./x" is the same file as "x": native LAMMPS hands the
 * name to fopen(), where "./x" and "x" name one path (read_data.html takes a file name; the working directory
 * is "."). Only the leading "./" and inner "/./" segments are removed; no directory is invented or resolved.
 */
export function normalizeFileName(name: string): string {
  let s = name;
  while (s.startsWith('./')) s = s.slice(2);
  while (s.includes('/./')) s = s.replace('/./', '/');
  return s;
}

/**
 * A session-file name resolved against the working directory set by shell cd (shell.html: "The cd command changes
 * the current working directory similar to the cd command. All subsequent LAMMPS commands that read/write files
 * will use the new directory."). The result is relative to the notebook root (''); a leading "/" is taken relative
 * to the root too, "." segments are dropped and ".." removes the segment before it. A ".." that would climb above
 * the root is kept, so the name stays a file name of the session (it is not found unless the user added it).
 */
export function resolveFileName(cwd: string, name: string): string {
  const segs: string[] = [];
  for (const part of (name.startsWith('/') ? name : `${cwd}/${name}`).split('/')) {
    if (part === '' || part === '.') continue;
    if (part === '..' && segs.length && segs[segs.length - 1] !== '..') segs.pop();
    else segs.push(part);
  }
  return segs.join('/');
}
