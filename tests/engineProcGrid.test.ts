import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * processors with a count above 1 on the engine's one process (procGrid.ts): native LAMMPS stops in
 * Comm::set_proc_grid once the box exists, from create_box, read_data and read_restart. Measured with native LAMMPS:
 * read_data of a one-atom file after processors * 2 * prints the orthogonal box line and stops with the error
 * Could not create 3d grid of processors; examples/GRAPHICS/in.mixer-drum (processors * * 2) stops at create_box.
 */

const runInput = async (text: string, files: Record<string, string> = {}): Promise<{ error: string | null; logs: string[] }> => {
  const logs: string[] = [];
  let error: string | null = null;
  const s = new Session({
    emit: (e: EngineEvent) => {
      if (e.kind === 'log') logs.push(e.text);
      if (e.kind === 'error') error = e.message;
    },
    writeFile: () => {},
  });
  for (const [k, v] of Object.entries(files)) s.addFile(k, v);
  try { await s.execute(text); } catch (e) { error ??= String(e); }
  return { error, logs };
};

const DATA = 'x\n\n1 atoms\n1 atom types\n\n0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\nMasses\n\n1 1.0\n\nAtoms\n\n1 1 1.0 1.0 1.0\n';

describe('processors grid on one process', () => {
  it('read_data stops after the box, as native read_data does', async () => {
    const r = await runInput('units lj\natom_style atomic\nprocessors * 2 *\nread_data d.data\nprint OK\n', { 'd.data': DATA });
    expect(r.error).toMatch(/Could not create 3d grid of processors/);
    expect(r.logs.some((l) => l === 'OK')).toBe(false);
  });

  it('create_box prints the box, then stops', async () => {
    const r = await runInput('units lj\natom_style atomic\nprocessors * * 2\nlattice fcc 0.8442\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\nprint OK\n');
    expect(r.error).toMatch(/Could not create 3d grid of processors/);
    expect(r.logs.some((l) => l.startsWith('Created orthogonal box'))).toBe(true);
  });

  it('read_restart stops too, and counts of 1 or * still run', async () => {
    const files: Record<string, string> = {};
    const w = new Session({ emit: () => {}, writeFile: (n: string, t: string) => { files[n] = t; } });
    await w.execute('units lj\natom_style atomic\nlattice fcc 0.8442\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nwrite_restart r.restart\n');
    const bad = await runInput('processors 2 1 * \nread_restart r.restart\nprint OK\n', files);
    expect(bad.error).toMatch(/Could not create 3d grid of processors/);
    const ok = await runInput('processors * 1 1\nread_restart r.restart\nprint OK\n', files);
    expect(ok.error).toBeNull();
    expect(ok.logs).toContain('OK');
  });

  it('clear forgets the grid', async () => {
    const r = await runInput('processors * * 2\nclear\nunits lj\natom_style atomic\nlattice fcc 0.8442\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\nprint OK\n');
    expect(r.error).toBeNull();
    expect(r.logs).toContain('OK');
  });
});
