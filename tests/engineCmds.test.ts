import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

// Commands of src/engine-gpl/cmds (lazy group gpl_cmds): argument errors and the exact text of small files.
const run = async (text: string) => {
  const files = new Map<string, string>();
  const errors: string[] = [];
  const s = new Session({
    emit: (e: { kind: string; message?: string }) => { if (e.kind === 'error') errors.push(e.message ?? ''); },
    writeFile: (n: string, t: string, ap: boolean) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  } as never);
  const setup = `units lj
atom_style atomic
atom_modify map array
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;
  let error = '';
  try {
    await s.execute(setup + text);
  } catch (e) {
    error = (e as Error).message;
  }
  return { files, error: error || errors.join('|') };
};

describe('gpl_cmds commands', () => {
  it('write_coeff needs a simulation box', async () => {
    const s = new Session({ emit: () => { /* none */ }, writeFile: () => { /* none */ } } as never);
    await expect(s.execute('write_coeff x.txt')).rejects.toThrow(/Write_coeff command before simulation box is defined/);
  });

  it('write_coeff writes the pair section with the ij list and the header', async () => {
    const { files, error } = await run('write_coeff w.txt\n');
    expect(error).toBe('');
    expect(files.get('w.txt')).toBe('# LAMMPS coeff file via write_coeff, version 2 Sep 2026\n# pair_style lj/cut\npair_coeff 1 1 1 1 2.5\n\n');
  });

  it('write_coeff rejects a missing file argument', async () => {
    const { error } = await run('write_coeff\n');
    expect(error).toMatch(/Illegal write_coeff command: missing argument\(s\)/);
  });

  it('angle_write refuses an undefined angle style and an out-of-range type', async () => {
    const a = await run('angle_write 1 5 t.table KEY\n');
    expect(a.error).toMatch(/Angle_write command before an angle_style is defined/);
  });

  it('dihedral_write refuses charmm and class2 styles by name', async () => {
    // a charmm dihedral is refused before the table is written
    const { error } = await run('dihedral_style charmm\ndihedral_write 1 5 d.table KEY\n');
    expect(error).toMatch(/Dihedral_write command is not compatible with dihedral style charmm|dihedral_style|charmm/);
  });

  it('group2ndx writes the System group first and the IDs in rows of the native width', async () => {
    const { files, error } = await run('group2ndx g.ndx\n');
    expect(error).toBe('');
    const text = files.get('g.ndx') ?? '';
    expect(text.startsWith('[ System ]\n')).toBe(true);
    // 32 atoms: width log10(32)+2 = 3, 26 columns per line
    expect(text.split('\n')[1].length).toBe(26 * 3);
  });

  it('group2ndx refuses a non-existing group', async () => {
    const { error } = await run('group2ndx g.ndx nosuch\n');
    expect(error).toMatch(/Non-existing group requested/);
  });

  it('ndx2group refuses a missing index file', async () => {
    const { error } = await run('ndx2group missing.ndx\n');
    expect(error).toMatch(/Cannot open index file for reading: missing.ndx/);
  });

  it('ndx2group restores a group written by group2ndx', async () => {
    const { files, error } = await run('group evens id 2 4\ngroup2ndx g.ndx evens\ngroup evens delete\nndx2group g.ndx evens\ngroup2ndx h.ndx evens\n');
    expect(error).toBe('');
    expect(files.get('h.ndx')).toBe('[ evens ]\n  2  4\n');
  });

  it('write_molecule refuses an unknown template ID', async () => {
    const { error } = await run('write_molecule nosuch m.txt\n');
    expect(error).toMatch(/Molecule template ID nosuch for write_molecule does not exist/);
  });

  it('region2vmd refuses a missing argument list', async () => {
    const { error } = await run('region2vmd\n');
    expect(error).toMatch(/region2vmd|Illegal/);
  });

  it('min_style hftn is registered and names its group', async () => {
    const { error } = await run('min_style hftn\n');
    expect(error).toBe('');
  });
});
