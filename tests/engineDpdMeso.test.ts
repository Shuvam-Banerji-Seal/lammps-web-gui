import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Unit tests of the DPD-MESO styles (src/engine-gpl/dpdmeso) and of the atom styles they need (atom_style edpd, mdpd and
 * tdpd: src/engine/atoms.ts, output/data.ts, commands/setup.ts, output/restart.ts). The native parity cases are the
 * tests/oracle/w93meso_* cases (engineOracle.test.ts); these check the atom-style columns, the set keywords, the refusals
 * and the restart round trip. The column layouts were measured with native LAMMPS (black box, write_data after
 * create_atoms): edpd "id type edpd_temp edpd_cv x y z" (+ image flags), mdpd "id type rho x y z", tdpd "id type x y z cc1 cc2".
 */

const newSession = () => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  return { session, files };
};

/** The header of every script: units and the box, with the atom style given before the box (as scripts do). */
const headFor = (style: string): string => `units lj
dimension 3
boundary p p p
atom_style ${style}
region b block 0 3 0 3 0 3
create_box 1 b
`;

/** The Atoms lines of a written data file (the lines after "Atoms # ...", up to the next blank-line section). */
const atomLines = (text: string): string[] => {
  const lines = text.split('\n');
  const k = lines.findIndex((l) => l.startsWith('Atoms'));
  const out: string[] = [];
  let i = k + 1;
  while (i < lines.length && lines[i].trim() === '') i++;
  for (; i < lines.length && lines[i].trim() !== ''; i++) out.push(lines[i]);
  return out;
};

describe('atom_style edpd, mdpd and tdpd', () => {
  it('writes the native column layouts (measured with native write_data)', async () => {
    const { session, files } = newSession();
    await session.execute(`${headFor('edpd')}create_atoms 1 single 1.0 1.0 1.0\nmass 1 1.0\nwrite_data e.data\n`);
    expect(atomLines(files.get('e.data')!)).toEqual(['1 1 1 100000 1 1 1 0 0 0']);
    const m = newSession();
    await m.session.execute(`${headFor('mdpd')}create_atoms 1 single 1.0 1.0 1.0\nmass 1 1.0\nwrite_data m.data\n`);
    expect(atomLines(m.files.get('m.data')!)).toEqual(['1 1 0 1 1 1 0 0 0']);
    const t = newSession();
    await t.session.execute(`${headFor('tdpd 2')}create_atoms 1 single 1.0 1.0 1.0\nset atom 1 cc 1 0.5\nmass 1 1.0\nwrite_data t.data\n`);
    expect(atomLines(t.files.get('t.data')!)).toEqual(['1 1 1 1 1 0.5 0 0 0 0']);
  });

  it('reads the edpd and tdpd Atoms columns back', async () => {
    const data = `dpd\n\n2 atoms\n1 atom types\n\n0 3 xlo xhi\n0 3 ylo yhi\n0 3 zlo zhi\n\nMasses\n\n1 1\n\nAtoms # edpd\n\n1 1 2.5 4.0 1 1 1\n2 1 1.5 7.0 2 2 2\n`;
    const { session, files } = newSession();
    session.addFile('d.data', data);
    await session.execute('units lj\natom_style edpd\nread_data d.data\nwrite_data o.data\n');
    expect(atomLines(files.get('o.data')!)).toEqual(['1 1 2.5 4 1 1 1 0 0 0', '2 1 1.5 7 2 2 2 0 0 0']);
    const t = newSession();
    t.session.addFile('t.data', `dpd\n\n1 atoms\n1 atom types\n\n0 3 xlo xhi\n0 3 ylo yhi\n0 3 zlo zhi\n\nMasses\n\n1 1\n\nAtoms # tdpd\n\n1 1 1 1 1 0.25 0.75\n`);
    await t.session.execute('units lj\natom_style tdpd 2\nread_data t.data\nwrite_data o.data\n');
    expect(atomLines(t.files.get('o.data')!)).toEqual(['1 1 1 1 1 0.25 0.75 0 0 0']);
  });

  it('refuses a bad species count and a set keyword on the wrong style', async () => {
    await expect(newSession().session.execute(`${headFor('tdpd')}`)).rejects.toThrow(/Invalid atom_style tdpd command/);
    await expect(newSession().session.execute(`${headFor('tdpd 0')}`)).rejects.toThrow(/Invalid atom_style tdpd command/);
    await expect(newSession().session.execute(`${headFor('atomic')}create_atoms 1 single 1 1 1\nset atom * edpd/temp 2.0\n`)).rejects.toThrow(/Cannot set attribute edpd\/temp for atom style atomic/);
    await expect(newSession().session.execute(`${headFor('edpd')}create_atoms 1 single 1 1 1\nset atom * edpd/cv 0\n`)).rejects.toThrow(/Invalid edpd\/cv value 0 in set command/);
    await expect(newSession().session.execute(`${headFor('tdpd 2')}create_atoms 1 single 1 1 1\nset atom * cc 3 1.0\n`)).rejects.toThrow(/set cc: species index 3 is above the 2 species/);
    await expect(newSession().session.execute(`${headFor('tdpd 2')}create_atoms 1 single 1 1 1\nset atom * cc 0 1.0\n`)).rejects.toThrow('Invalid cc index 0 in set command');
    await expect(newSession().session.execute(`${headFor('tdpd 2')}create_atoms 1 single 1 1 1\nset atom * cc 1 -1.0\n`)).rejects.toThrow(/Invalid cc value -1 in set command/);
  });

  it('keeps the edpd temperature through write_restart and read_restart', async () => {
    const { session, files } = newSession();
    await session.execute(`${headFor('edpd')}create_atoms 1 single 1.0 1.0 1.0\ncreate_atoms 1 single 2.0 2.0 2.0\nmass 1 1.0\nset atom 2 edpd/temp 2.5\nwrite_data before.data\nwrite_restart rt.restart\n`);
    const before = atomLines(files.get('before.data')!);
    await session.execute('clear\nread_restart rt.restart\nwrite_data after.data\n');
    expect(atomLines(files.get('after.data')!)).toEqual(before);
    expect(before[1].split(' ')[2]).toBe('2.5');
  });
});

describe('DPD-MESO styles: refusals', () => {
  it('names the atom style a pair style needs', async () => {
    await expect(newSession().session.execute(`${headFor('atomic')}mass 1 1.0\ncomm_modify vel yes\npair_style edpd 1.58 9872598\npair_coeff 1 1 18.75 4.5 0.41 1.58 1.41E-5 2.0 1.58\nrun 0\n`)).rejects.toThrow(/Pair style edpd requires atom style edpd/);
    await expect(newSession().session.execute(`${headFor('atomic')}mass 1 1.0\ncomm_modify vel yes\npair_style mdpd 1.0 1.0 9872598\npair_coeff 1 1 -40 25 18.0 1.0 0.75\nrun 0\n`)).rejects.toThrow(/requires atom attribute rho/);
  });

  it('refuses a source fix on the wrong atom style and an unknown form', async () => {
    await expect(newSession().session.execute(`${headFor('atomic')}create_atoms 1 single 1 1 1\nmass 1 1.0\nfix s all edpd/source region b 0.1\nrun 0\n`)).rejects.toThrow(/Fix edpd\/source requires atom style edpd/);
    await expect(newSession().session.execute(`${headFor('edpd')}create_atoms 1 single 1 1 1\nfix s all edpd/source blob 0.1\n`)).rejects.toThrow(/Illegal fix edpd\/source command option blob/);
  });

  it('accepts any tdpd species index when the fix and compute are defined, and refuses one outside the style when used', async () => {
    // Measured with native LAMMPS: an index outside the species (0, -1) is accepted at definition, and run 0 ends OK;
    // native reads and writes outside its arrays when they are used, which the engine refuses
    await expect(newSession().session.execute(`${headFor('tdpd 2')}create_atoms 1 single 1 1 1\nfix s all tdpd/source 3 cuboid 0 0 0 1 1 1 0.1\ncompute c all tdpd/cc/atom 0\n`)).resolves.toBeUndefined();
    await expect(newSession().session.execute(`${headFor('tdpd 2')}create_atoms 1 single 1 1 1\nmass 1 1.0\nfix s all tdpd/source 3 cuboid 0 0 0 1 1 1 0.1\nfix i all nve\nrun 1\n`)).rejects.toThrow(/cc index 3 is not a species 1..2/);
  });
});
