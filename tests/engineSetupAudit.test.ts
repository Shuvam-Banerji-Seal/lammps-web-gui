import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/** Runs a script and returns the error message it stops with ('' when it runs). */
const errorOf = async (script: string, files: Record<string, string> = {}): Promise<string> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) s.addFile(n, t);
  try {
    await s.execute(script);
    return '';
  } catch (e) {
    return String(e);
  }
};

const BOX = 'units lj\natom_style atomic\nlattice sc 1.0\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\n';

// Data file for the read_data checks: 6 atoms, 2 bonds starting at each of atoms 1, 2 and 3 (bond type 2 needs bond/types 2)
const BONDS_DATA = [
  'bonds', '', '6 atoms', '6 bonds', '2 atom types', '2 bond types', '', '0 4 xlo xhi', '0 4 ylo yhi', '0 4 zlo zhi', '',
  'Masses', '', '1 1.0', '2 1.0', '', 'Atoms # bond', '',
  '1 1 1 0.5 0.5 0.5', '2 1 1 1.5 0.5 0.5', '3 1 1 0.5 1.5 0.5', '4 1 2 1.5 1.5 0.5', '5 1 2 2.5 0.5 0.5', '6 1 2 0.5 2.5 0.5', '',
  'Bonds', '', '1 1 1 2', '2 1 1 3', '3 1 2 4', '4 1 2 5', '5 2 3 6', '6 2 3 4', '',
].join('\n');

describe('setup commands: refusals and limits measured with native LAMMPS', () => {
  it('create_atoms basis outside the lattice basis stops (native: Out of range basis setting)', async () => {
    const msg = await errorOf(`${BOX}create_atoms 1 box basis 1 1 basis 2 2\n`);
    expect(msg).toContain("Out of range basis setting '2 2' in create_atoms command");
  });

  it('create_atoms random maxtry 0 and overlap 0 stop (native: Illegal create_atoms maxtry / overlap value)', async () => {
    expect(await errorOf(`${BOX}create_atoms 1 random 2 1 box overlap 0.1 maxtry 0\n`)).toContain('Illegal create_atoms maxtry value 0');
    expect(await errorOf(`${BOX}create_atoms 1 random 2 1 box overlap 0.0 maxtry 3\n`)).toContain('Illegal create_atoms overlap value');
  });

  it('read_data shift without add stops (native: Cannot use read_data shift without add keyword)', async () => {
    const msg = await errorOf(`${BOX}read_data base.data shift 1 1 1\n`, { 'base.data': BONDS_DATA });
    expect(msg).toContain('Cannot use read_data shift without add keyword');
  });

  it('a read after the box exists is limited by create_box extra/bond/per/atom (native: Subsequent read data induced too many bonds per atom)', async () => {
    // measured: bond/types 2 with extra/bond/per/atom 0 or 1 stops; with 2 the same read is accepted
    const data = { 'base.data': BONDS_DATA };
    const at = (extra: number) => `units lj\natom_style bond\nregion box block 0 4 0 4 0 4\ncreate_box 2 box bond/types 2 extra/bond/per/atom ${extra}\nread_data base.data add merge\n`;
    expect(await errorOf(at(0), data)).toContain('Subsequent read data induced too many bonds per atom');
    expect(await errorOf(at(1), data)).toContain('Subsequent read data induced too many bonds per atom');
    expect(await errorOf(at(2), data)).toBe('');
  });

  it('delete_atoms overlap without a pair style stops (native: Delete_atoms requires a pair style be defined)', async () => {
    const msg = await errorOf(`${BOX}create_atoms 1 box\nmass 1 1.0\ndelete_atoms overlap 0.5 all all\n`);
    expect(msg).toContain('Delete_atoms requires a pair style be defined');
  });

  it('timestep 0 is accepted (native runs on after it)', async () => {
    expect(await errorOf(`${BOX}timestep 0\nprint "REACHED"\n`)).toBe('');
  });
});
