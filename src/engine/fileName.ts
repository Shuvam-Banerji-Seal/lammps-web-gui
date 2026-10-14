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
