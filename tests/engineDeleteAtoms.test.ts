import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * delete_atoms compress / condense (docs.lammps.org/delete_atoms.html). Expected IDs and
 * positions were measured with native LAMMPS (black box) on the same inputs: 10 atoms on a line
 * (x = 0..9, or 0.5..9.5 for the bonded chain), atoms 2 3 7 deleted, atom sorting off. Native's
 * atom list after the delete is 1 10 9 4 5 6 8 (the last atom fills each hole).
 */

const LINE = (style: string, map = false) => `units lj
atom_style ${style}
${map ? 'atom_modify map array\n' : ''}lattice sc 1.0
region box block 0 10 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 1.0
atom_modify sort 0 0.0
group gone id 2 3 7
`;

const CHAIN = (() => {
  const l = ['chain', '', '10 atoms', '9 bonds', '1 atom types', '1 bond types', '', '0 12 xlo xhi', '-1 1 ylo yhi', '-1 1 zlo zhi', '', 'Masses', '', '1 1.0', '', 'Atoms # bond', ''];
  for (let i = 1; i <= 10; i++) l.push(`${i} 1 1 ${i - 0.5} 0 0`);
  l.push('', 'Bonds', '');
  for (let i = 1; i <= 9; i++) l.push(`${i} 1 ${i} ${i + 1}`);
  return `${l.join('\n')}\n`;
})();

const run = async (text: string, files: Record<string, string> = {}) => {
  const out = new Map<string, string>();
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => out.set(n, t) });
  for (const [k, v] of Object.entries(files)) s.addFile(k, v);
  let error: string | null = null;
  try { await s.execute(`${text}write_dump all custom o.dump id x\n`); } catch (e) { error = e instanceof Error ? e.message : String(e); }
  const err = events.find((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error');
  const dump = out.get('o.dump');
  const rows = dump ? dump.trim().split('\n').slice(9).map((r) => r.split(/\s+/).map(Number)) : [];
  return { error: err?.message ?? error, ids: rows.map((r) => `${r[0]}:${r[1]}`) };
};

describe('delete_atoms compress / condense (native measurements)', () => {
  it('atomic default compresses in atom-list order', async () => {
    const r = await run(`${LINE('atomic')}delete_atoms group gone\n`);
    expect(r.ids).toEqual(['1:0', '2:9', '3:8', '4:3', '5:4', '6:5', '7:7']);
  });
  it('atomic condense needs an atom map', async () => {
    const r = await run(`${LINE('atomic')}delete_atoms group gone condense yes\n`);
    expect(r.error).toMatch(/condense yes' option requires an atom map/);
  });
  it('atomic condense with a map keeps the order of the IDs', async () => {
    const r = await run(`${LINE('atomic', true)}delete_atoms group gone condense yes\n`);
    expect(r.ids).toEqual(['1:0', '7:9', '6:8', '2:3', '3:4', '4:5', '5:7']);
  });
  it('molecular default and condense keep the IDs; compress yes renumbers in list order', async () => {
    const base = 'units lj\natom_style bond\nread_data chain.data\natom_modify sort 0 0.0\nbond_style zero\nbond_coeff 1\ngroup gone id 2 3 7\n';
    const kept = ['1:0.5', '10:9.5', '9:8.5', '4:3.5', '5:4.5', '6:5.5', '8:7.5'];
    expect((await run(`${base}delete_atoms group gone bond yes\n`, { 'chain.data': CHAIN })).ids).toEqual(kept);
    expect((await run(`${base}delete_atoms group gone bond yes condense yes\n`, { 'chain.data': CHAIN })).ids).toEqual(kept);
    expect((await run(`${base}delete_atoms group gone bond yes compress yes\n`, { 'chain.data': CHAIN })).ids)
      .toEqual(['1:0.5', '2:9.5', '3:8.5', '4:3.5', '5:4.5', '6:5.5', '7:7.5']);
  });
});
