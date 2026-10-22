import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// SPH package, the core side: atom_style sph (rho, esph, cv, vest), its create_atoms defaults, read_data and
// write_data columns, the set keywords sph/e, sph/cv and sph/rho, and the restart round trip. The numbers checked here were
// measured with native LAMMPS (black box); the pair, fix and compute styles are in engineSphStyles.test.ts and the
// oracle cases tests/oracle/w83sph_*.in compare whole runs with native LAMMPS.

const newSession = () => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  return { session, events, files };
};

/** The error a script stops with ('' when it runs through). */
const errorOf = async (session: Session, events: EngineEvent[], script: string): Promise<string> => {
  try {
    await session.execute(script);
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
  const e = events.find((ev) => ev.kind === 'error');
  return e && 'message' in e ? String(e.message) : '';
};

const BOX = `units si
atom_style sph
dimension 2
region box block 0 2 0 2 -0.5 0.5
create_box 1 box
lattice sq 1.0
create_atoms 1 box
mass 1 1.0
`;

describe('atom_style sph', () => {
  it('create_atoms gives rho 0, esph 0, cv 1 (measured: write_data after create_atoms)', async () => {
    const { session, files } = newSession();
    await session.execute(`${BOX}write_data box.data\n`);
    const text = files.get('box.data') ?? '';
    expect(text).toContain('Atoms # sph');
    expect(text).toContain('\n1 1 0 0 1 0 0 0 0 0 0\n2 1 0 0 1 1 0 0 0 0 0\n');
    const st = session.sys.state;
    expect(Array.from(st.custom.get('#cv')!.data.subarray(0, 4))).toEqual([1, 1, 1, 1]);
    expect(Array.from(st.custom.get('#rho')!.data.subarray(0, 4))).toEqual([0, 0, 0, 0]);
    expect(st.custom.get('#vest')!.data.length).toBe(3 * st.n);
  });

  it('set sph/rho, sph/e and sph/cv write the values; sph/e and the others need atom_style sph', async () => {
    const { session, events } = newSession();
    expect(await errorOf(session, events, `${BOX}set group all sph/rho 1000.0\nset region box sph/e 2.5\nset type 1 sph/cv 0.5\n`)).toBe('');
    const st = session.sys.state;
    expect(Array.from(st.custom.get('#rho')!.data.subarray(0, 4))).toEqual([1000, 1000, 1000, 1000]);
    expect(Array.from(st.custom.get('#esph')!.data.subarray(0, 4))).toEqual([2.5, 2.5, 2.5, 2.5]);
    expect(Array.from(st.custom.get('#cv')!.data.subarray(0, 4))).toEqual([0.5, 0.5, 0.5, 0.5]);

    const other = newSession();
    expect(await errorOf(other.session, other.events, 'units lj\natom_style atomic\ndimension 2\nregion box block 0 2 0 2 -0.5 0.5\ncreate_box 1 box\nlattice sq 1.0\ncreate_atoms 1 box\nmass 1 1\nset type 1 sph/e 2.5\n'))
      .toMatch(/Cannot set attribute sph\/e for atom style atomic$/);
  });

  it('read_data takes "atom-ID atom-type rho esph cv x y z" and write_data writes it back', async () => {
    const { session, files } = newSession();
    session.addFile('sph.data', [
      'sph test', '', '2 atoms', '1 atom types', '',
      '0 2 xlo xhi', '0 2 ylo yhi', '-0.5 0.5 zlo zhi', '',
      'Masses', '', '1 1.5', '',
      'Atoms # sph', '',
      '1 1 1000 0.25 1.5 0 0 0', '2 1 900 0.5 2 1 1 0', '',
      'Velocities', '', '1 0 0 0', '2 0 0 0', '',
    ].join('\n'));
    await session.execute('units si\natom_style sph\ndimension 2\nread_data sph.data\nwrite_data out.data\n');
    const st = session.sys.state;
    expect(Array.from(st.custom.get('#rho')!.data.subarray(0, 2))).toEqual([1000, 900]);
    expect(Array.from(st.custom.get('#esph')!.data.subarray(0, 2))).toEqual([0.25, 0.5]);
    expect(Array.from(st.custom.get('#cv')!.data.subarray(0, 2))).toEqual([1.5, 2]);
    expect(files.get('out.data')).toContain('\n1 1 1000 0.25 1.5 0 0 0 0 0 0\n2 1 900 0.5 2 1 1 0 0 0 0\n');
  });

  it('restart keeps rho, esph, cv and vest', async () => {
    const { session, files } = newSession();
    await session.execute(`${BOX}set group all sph/rho 1000.0\nset group all sph/e 3.0\nrun 0\nwrite_restart sph.restart\n`);
    const before = ['#rho', '#esph', '#cv', '#vest'].map((k) => Array.from(session.sys.state.custom.get(k)!.data));
    expect(files.get('sph.restart')).toBeDefined();
    await session.execute('clear\nread_restart sph.restart\n');
    const after = ['#rho', '#esph', '#cv', '#vest'].map((k) => Array.from(session.sys.state.custom.get(k)!.data));
    expect(after).toEqual(before);
  });

  it('an sph keyword or compute on another atom style is an error', async () => {
    const { session, events } = newSession();
    const head = 'units lj\natom_style atomic\ndimension 2\nregion box block 0 2 0 2 -0.5 0.5\ncreate_box 1 box\nlattice sq 1.0\ncreate_atoms 1 box\nmass 1 1\n';
    expect(await errorOf(session, events, `${head}compute r all sph/rho/atom\n`)).toMatch(/Compute sph\/rho\/atom requires atom attribute density, e.g. in atom_style sph$/);
    const next = newSession();
    expect(await errorOf(next.session, next.events, `${head}pair_style sph/taitwater\npair_coeff * * 1 1 0.1 1\nrun 0\n`)).toMatch(/requires atom attributes energy, density, and velocity estimates, e\.g\. in atom_style sph/);
  });
});
