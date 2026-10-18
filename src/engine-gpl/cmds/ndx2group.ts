/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMMAND/ndx2group.cpp and
 * src/EXTRA-COMMAND/ndx2group.h (Ndx2Group::command, find_section, read_section and create; LAMMPS development team,
 * Sandia National Laboratories). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * ndx2group file [group ...] — ndx2group.html: "read the groups of a GROMACS index file": each "[ name ]" section is a
 * group of the atom IDs below it. Without group names, all sections are read, except "[ System ]" (the group all).
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { bitOfIndex } from '../../engine/group';
import type { System } from '../../engine/system';

const HEADER = /^\s*\[.*\]\s*$/;

/** The reader of the index file: the native fgets position over its lines. */
class IndexFile {
  private pos = 0;
  constructor(private lines: string[]) {}
  rewind(): void { this.pos = 0; }
  /** The next line, or null at the end of the file. */
  next(): string | null { return this.pos < this.lines.length ? this.lines[this.pos++] : null; }

  /** The name of the header line, checking its words (native: a header must be "[ name ]", three words). */
  private headerName(line: string): string {
    const words = line.trim().split(/[ \t\r\n\f]+/).filter((w) => w !== '');
    if (words.length !== 3) {
      const why = line.match(/\[([^[\]]+)\]/)?.[1].trim() ?? '';
      throw new StyleError(`Invalid group name in index file: '${why}'`);
    }
    return words[1];
  }

  /** find_section: reads up to the header named `name` ('' = any header); returns its name, or '' at the end. */
  findSection(name: string): string {
    const first = this.next();
    if (first === null) throw new StyleError("Read error: 'Success'");
    let line: string | null = first;
    while (line !== null) {
      if (HEADER.test(line)) {
        const h = this.headerName(line);
        if (name === '' || name === h) return h;
      }
      line = this.next();
    }
    return '';
  }

  /** read_section: the atom IDs below the current header; `next` is the name of the following header ('' at the end). */
  readSection(): { tags: number[]; next: string } {
    const tags: number[] = [];
    let line = this.next();
    while (line !== null) {
      if (HEADER.test(line)) return { tags, next: this.headerName(line) };
      for (const w of line.trim().split(/[ \t\r\n\f]+/)) {
        if (w === '') continue;
        if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Not a valid integer number: '${w}'`);
        tags.push(Number(w));
      }
      line = this.next();
    }
    return { tags, next: '' };
  }
}

/** Ndx2Group::create: the group (its atoms cleared first when it exists) gets the atoms with these IDs. */
const createGroup = (sys: System, name: string, tags: number[]): void => {
  const s = sys.state;
  const k0 = sys.groups.find(name);
  // group->assign(name + " clear") for an existing group other than "all"
  if (k0 > 0) {
    const m = ~bitOfIndex(k0);
    for (let i = 0; i < s.n; i++) s.mask[i] &= m;
  }
  const k = sys.groups.create(name);
  const bit = bitOfIndex(k);
  for (const tag of tags) {
    const i = sys.indexOfId(tag);
    if (i >= 0 && i < s.n) s.mask[i] |= bit;
  }
};

const ndx2group: Handler = ({ sys }, a) => {
  if (a.length < 1) throw new StyleError('Illegal ndx2group command: missing argument(s)');
  const file = a[0];
  const text = sys.files.has(sys.resolveFile(file)) ? sys.readFile(file) : null;
  if (text === null) throw new StyleError(`Cannot open index file for reading: ${file}`);
  sys.log(`Reading groups from index file ${file}:`);
  // fgets of an empty file fails at once (native find_section: "Read error")
  const idx = new IndexFile(text === '' ? [] : text.split('\n'));
  if (a.length === 1) {
    // restore all groups; the group "System" (all in LAMMPS) is skipped
    let name = idx.findSection('');
    while (name !== '') {
      if (name === 'System') { name = idx.findSection(''); continue; }
      sys.log(` Processing group '${name}'`);
      const { tags, next } = idx.readSection();
      createGroup(sys, name, tags);
      name = next;
    }
  } else {
    // restore the named groups, each searched from the beginning of the file
    for (const want of a.slice(1)) {
      idx.rewind();
      const name = idx.findSection(want);
      sys.log(` ${name !== '' ? 'Processing' : 'Skipping'} group '${want}'`);
      if (name !== '') {
        const { tags } = idx.readSection();
        createGroup(sys, name, tags);
      }
    }
  }
};

export const COMMANDS = { ndx2group };
