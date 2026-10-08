import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError } from '../src/engine/types';

/*
 * atom_style template (src/engine/template.ts, atoms.ts, output/data.ts, output/restart.ts).
 * Oracle cases w24tmpl_* compare the thermo, per-atom state and write_data text with native LAMMPS;
 * these tests cover the refusals and the write / read round trip.
 */

const CHAIN = `chain of three
3 atoms
2 bonds
1 angles

Coords

1 0 0 0
2 1 0 0
3 2 0 0

Types

1 1
2 1
3 1

Bonds

1 1 1 2
2 1 2 3

Angles

1 1 1 2 3
`;

const newSession = (files: Record<string, string> = {}) => {
  const written = new Map<string, string>();
  const s = new Session({ emit: () => {}, writeFile: (n, t) => written.set(n, t) });
  s.addFile('chain.txt', CHAIN);
  for (const [k, v] of Object.entries(files)) s.addFile(k, v);
  return { s, written };
};

const errorOf = async (script: string, files?: Record<string, string>): Promise<EngineError | null> => {
  const { s } = newSession(files);
  try {
    await s.execute(script);
    return null;
  } catch (e) {
    return e as EngineError;
  }
};

const HEAD = 'units real\nboundary p p p\nmolecule c chain.txt\natom_style template c\nregion b block 0 10 0 10 0 10 units box\ncreate_box 2 b bond/types 1 angle/types 1\n' +
  'mass * 1.0\npair_style lj/cut 3.0\npair_coeff * * 0.1 1.0\nbond_style harmonic\nbond_coeff 1 100.0 1.0\nangle_style harmonic\nangle_coeff 1 50.0 110.0\n';

describe('atom_style template', () => {
  it('needs the molecule template defined first', async () => {
    expect((await errorOf('units real\natom_style template nomol'))?.message).toMatch(/molecule template nomol does not exist/);
  });

  it('refuses the hybrid combination with a bond sub-style (atom_style.html hybrid note)', async () => {
    const e = await errorOf('units real\nmolecule c chain.txt\natom_style hybrid template c bond', { 'chain.txt': CHAIN });
    expect(e?.message).toMatch(/cannot combine the template style/);
  });

  it('creates the template topology with create_atoms mol and keeps the template index and atom', async () => {
    const { s, written } = newSession();
    await s.execute(`${HEAD}create_atoms 0 single 2 2 2 mol c 7 rotate 0 0 0 1 units box\nwrite_data out.data`);
    const text = written.get('out.data')!;
    // measured with native LAMMPS (black box): template atoms are written as id mol tindex tatom type x y z ix iy iz
    expect(text).toMatch(/Atoms # template\n\n1 1 1 1 1 /);
    // no topology sections and no bond or angle counts: the templates hold the topology
    expect(text).not.toMatch(/\nBonds\n|\nAngles\n|\d+ bonds\n|\d+ angles\n/);
    expect(text).toMatch(/\n1 bond types\n1 angle types\n/);
  });

  it('round-trips write_data then read_data to the same text', async () => {
    const first = newSession();
    await first.s.execute(`${HEAD}create_atoms 0 single 2 2 2 mol c 7 rotate 0 0 0 1 units box\ncreate_atoms 0 single 6 6 6 mol c 8 rotate 30 0 0 1 units box\nwrite_data rt1.data`);
    const text1 = first.written.get('rt1.data')!;
    const second = newSession({ 'rt1.data': text1 });
    await second.s.execute('units real\nboundary p p p\nmolecule c chain.txt\natom_style template c\npair_style lj/cut 3.0\nbond_style harmonic\nangle_style harmonic\nread_data rt1.data\nwrite_data rt2.data');
    const strip = (t: string) => t.split('\n').slice(1).join('\n');
    expect(strip(second.written.get('rt2.data')!)).toBe(strip(text1));
  });

  it('refuses Bonds sections and bond headers in a template data file', async () => {
    const data = `t\n\n3 atoms\n1 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nAtoms # template\n\n1 1 1 1 1 1 1 1\n2 1 1 2 1 2 1 1\n3 1 1 3 1 3 1 1\n\nBonds\n\n1 1 1 2\n`;
    const e = await errorOf(`units real\nmolecule c chain.txt\natom_style template c\nread_data t.data`, { 'chain.txt': CHAIN, 't.data': data });
    expect(e?.message).toMatch(/topology of atom_style template comes from the molecule templates/);
  });

  it('refuses a data file in the docs column order (native reads the type after the template columns)', async () => {
    // docs row "atom-ID atom-type molecule-ID template-index template-atom": "1 2 0 0 0" would read type 0 here
    const data = `t\n\n3 atoms\n2 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nAtoms # template\n\n1 2 0 0 0 1 1 1\n2 1 1 1 1 2 1 1\n3 1 1 2 1 3 1 1\n`;
    const e = await errorOf(`units real\nmolecule c chain.txt\natom_style template c\nread_data t.data`, { 'chain.txt': CHAIN, 't.data': data });
    expect(e?.message).toMatch(/atom type 0 is outside/);
  });

  it('refuses create_atoms mol with a template other than the atom style template', async () => {
    const e = await errorOf(`${HEAD}molecule d chain.txt\ncreate_atoms 0 single 2 2 2 mol d 7 units box`);
    expect(e?.message).toMatch(/must use the same molecule template-ID|same molecule template-ID/);
  });

  it('deletes a whole template molecule but refuses a partial one (native: Bond atom missing in image check)', async () => {
    const whole = await errorOf(`${HEAD}create_atoms 0 single 2 2 2 mol c 7 units box\ngroup g id 1:3\ndelete_atoms group g\nrun 0`);
    expect(whole).toBeNull();
    const partial = await errorOf(`${HEAD}create_atoms 0 single 2 2 2 mol c 7 units box\ngroup g id 2\ndelete_atoms group g`);
    expect(partial?.message).toMatch(/Bond atom missing in image check/);
  });

  it('read_restart needs the molecule template to be defined (measured with native LAMMPS)', async () => {
    const { s, written } = newSession();
    await s.execute(`${HEAD}create_atoms 0 single 2 2 2 mol c 7 units box\nwrite_restart r.restart`);
    const txt = written.get('r.restart')!;
    const bare = newSession({ 'r.restart': txt });
    await expect(bare.s.execute('read_restart r.restart')).rejects.toThrow(/needs molecule template c/);
    const ok = newSession({ 'r.restart': txt });
    await ok.s.execute('molecule c chain.txt\nread_restart r.restart\nwrite_data back.data');
    expect(ok.written.get('back.data')).toMatch(/Atoms # template/);
  });

  it('create_atoms random draws Park-Miller numbers after 30 skipped draws (measured with native LAMMPS)', async () => {
    const { s, written } = newSession();
    await s.execute('units lj\natom_style atomic\nregion b block 0 10 0 10 0 10 units box\ncreate_box 2 b\ncreate_atoms 2 random 1 495437 b\nwrite_data rnd.data');
    const lines = written.get('rnd.data')!.split('\n');
    const k = lines.findIndex((l) => l.startsWith('Atoms'));
    const [, , x, y, z] = lines[k + 2].trim().split(/\s+/).map(Number);
    // Park-Miller minimal standard: seed <- 16807 seed mod (2^31 - 1)
    let seed = 495437;
    const u = () => { seed = (16807 * seed) % 2147483647; return seed / 2147483647; };
    for (let i = 0; i < 30; i++) u();
    expect(x).toBeCloseTo(10 * u(), 9);
    expect(y).toBeCloseTo(10 * u(), 9);
    expect(z).toBeCloseTo(10 * u(), 9);
  });
});
