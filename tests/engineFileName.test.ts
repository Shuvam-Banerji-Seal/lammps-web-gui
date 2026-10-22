import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { normalizeFileName } from '../src/engine/fileName';
import type { EngineEvent } from '../src/engine/types';

/*
 * Session-file names: "./x" is the file "x" (native LAMMPS passes the name to fopen). Found by running
 * examples/PACKAGES/interlayer/aip_water_2dm/in.gr_water, which reads "./gra_water.data". Only the leading
 * "./" and inner "/./" segments are removed.
 */

const DATA = `two atoms

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 12.0

Atoms # atomic

1 1 1.0 1.0 1.0
2 1 2.0 2.0 2.0
`;

const runRead = async (files: Record<string, string>, readName: string): Promise<{ atoms: number | null; error: string | null }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  const script = `units metal
atom_style atomic
read_data ${readName}
variable n equal atoms
print "NATOMS \${n}"
`;
  try {
    await session.execute(script);
  } catch (e) {
    return { atoms: null, error: String(e) };
  }
  const err = events.find((ev) => ev.kind === 'error');
  const printed = events.find((ev) => ev.kind === 'log' && 'text' in ev && String(ev.text).startsWith('NATOMS'));
  const m = printed && 'text' in printed ? /NATOMS (\d+)/.exec(String(printed.text)) : null;
  return { atoms: m ? Number(m[1]) : null, error: err && 'message' in err ? err.message : null };
};

describe('normalizeFileName', () => {
  it('drops a leading "./" and inner "/./" segments only', () => {
    expect(normalizeFileName('./gra_water.data')).toBe('gra_water.data');
    expect(normalizeFileName('././a.data')).toBe('a.data');
    expect(normalizeFileName('dir/./a.data')).toBe('dir/a.data');
    expect(normalizeFileName('dir/a.data')).toBe('dir/a.data');
    // no directory is invented or resolved
    expect(normalizeFileName('../a.data')).toBe('../a.data');
    expect(normalizeFileName('dir/../a.data')).toBe('dir/../a.data');
  });
});

describe('session file lookup with a "./" prefix', () => {
  it('read_data ./x reads the session file x', async () => {
    const r = await runRead({ 'gra_water.data': DATA }, './gra_water.data');
    expect(r.error).toBeNull();
    expect(r.atoms).toBe(2);
  });

  it('a file added as "./x" is found by "x"', async () => {
    const r = await runRead({ './two.data': DATA }, 'two.data');
    expect(r.error).toBeNull();
    expect(r.atoms).toBe(2);
  });

  it('a missing file still errors and names the file as written', async () => {
    const r = await runRead({}, './missing.data');
    expect(r.error).toContain('Cannot open file ./missing.data: No such file or directory');
  });
});
