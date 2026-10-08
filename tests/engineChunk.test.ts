import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute chunk/atom, fix ave/chunk, com/chunk and msd/chunk: bin assignment
 * on the documented edge rules (docs.lammps.org/compute_chunk_atom.html:
 * "Chunk IDs range from 1 to *Nchunk* inclusive"), reduced / lattice / box
 * units, compress with OrigID, and argument errors. Expected counts were
 * measured with native LAMMPS (black box). Chunk counts come from the Ncount
 * column of fix ave/chunk's file output, the way a user would read them.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  return { session, error, files };
};

const BOX = `
units           lj
atom_style      atomic
boundary        p p p
region          box block 0 10 0 10 0 10
create_box      2 box
mass            * 1.0
`;

/** Parses fix ave/chunk file output: one section per output step. */
const sections = (text: string) => {
  const lines = text.trim().split('\n');
  const out: { step: number; n: number; total: number; rows: string[][] }[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('#')) continue;
    const head = lines[i].trim().split(/\s+/).map(Number);
    if (head.length !== 3) continue;
    const rows: string[][] = [];
    for (let k = 1; k <= head[1]; k++) rows.push(lines[i + k].trim().split(/\s+/));
    out.push({ step: head[0], n: head[1], total: head[2], rows });
    i += head[1];
  }
  return out;
};

/** Atoms in each chunk (Ncount column) of the last section. */
const counts = (text: string): number[] => {
  const last = sections(text).at(-1)!;
  // bin styles: chunk ID, Coord1, Ncount, values
  return last.rows.map((r) => Number(r[2]));
};

describe('compute chunk/atom bin assignment', () => {
  it('puts atoms on bin edges in the bin above, from the lower box edge', async () => {
    // box 0..10, delta 3 from the lower edge: bins [0,3) [3,6) [6,9) [9,12)
    const atoms = [0, 2.999999, 3.0, 5.0, 6.0, 9.0, 9.9999, 2.0];
    const create = atoms.map((x) => `create_atoms 1 single ${x} 1 1`).join('\n');
    const { session, error, files } = await runScript(`${BOX}\n${create}
compute a all chunk/atom bin/1d x lower 3.0 units box
variable n equal c_a
fix f all ave/chunk 1 1 1 a vx file edge.txt
run 0`);
    expect(error).toBeNull();
    expect(session.sys.compute('a').scalarValue()).toBe(4);
    // Ncount: x in [0,3) -> 3 atoms (0, 2.999999, 2.0); [3,6) -> 2; [6,9) -> 1; [9,12) -> 2
    expect(counts(files.get('edge.txt')!)).toEqual([3, 2, 1, 2]);
  });

  it('the global array f_ID[i][j] holds the chunk table (row = chunk, column = Ncount then values)', async () => {
    const create = [1.0, 2.0, 6.0].map((x) => `create_atoms 1 single ${x} 1 1`).join('\n');
    const { session, error } = await runScript(`${BOX}\n${create}
velocity all set 0.25 0 0 units box
compute c all chunk/atom bin/1d x lower 5.0 units box
fix f all ave/chunk 1 1 1 c vx
run 0`);
    expect(error).toBeNull();
    const f = session.sys.fix('f');
    expect(f.sizeArrayRows).toBe(2);
    // bin 1 holds two atoms with vx = 0.25; bin 2 one atom: columns are Coord1, Ncount, vx
    expect(f.computeArray(0, 1)).toBe(2);
    expect(f.computeArray(0, 2)).toBeCloseTo(0.25, 12);
    expect(f.computeArray(1, 1)).toBe(1);
    expect(f.computeArray(5, 1)).toBe(0); // rows beyond Nchunk read as 0
  });

  it('gives the same bins with origin center and upper (4 bins each for box 0..10, delta 3)', async () => {
    const create = [0.0, 3.0, 5.0, 9.0].map((x) => `create_atoms 1 single ${x} 1 1`).join('\n');
    const { session, error } = await runScript(`${BOX}\n${create}
compute c all chunk/atom bin/1d x center 3.0 units box
compute u all chunk/atom bin/1d x upper 3.0 units box
run 0`);
    expect(error).toBeNull();
    expect(session.sys.compute('c').scalarValue()).toBe(4);
    expect(session.sys.compute('u').scalarValue()).toBe(4);
  });

  it('reduced units: delta is a fraction of the box and edges behave the same', async () => {
    // 0.3 reduced in a box of length 10 is 3.0 in box units
    const atoms = [0, 2.999999, 3.0, 5.0, 6.0, 9.0, 9.9999, 2.0];
    const create = atoms.map((x) => `create_atoms 1 single ${x} 1 1`).join('\n');
    const { session, error, files } = await runScript(`${BOX}\n${create}
compute r all chunk/atom bin/1d x lower 0.3 units reduced
fix f all ave/chunk 1 1 1 r vx file red.txt
run 0`);
    expect(error).toBeNull();
    expect(session.sys.compute('r').scalarValue()).toBe(4);
    expect(counts(files.get('red.txt')!)).toEqual([3, 2, 1, 2]);
  });

  it('lattice units (default): a lattice spacing scales the delta and the origin', async () => {
    // no lattice command: spacing 1, so delta 0.3 gives 34 bins over 0..10 and x = 3.0 lies in bin 11
    const { session, error, files } = await runScript(`${BOX}
create_atoms 1 single 3.0 1 1
compute l all chunk/atom bin/1d x lower 0.3
fix f all ave/chunk 1 1 1 l vx file lat.txt
run 0`);
    expect(error).toBeNull();
    expect(session.sys.compute('l').scalarValue()).toBe(34);
    const rows = sections(files.get('lat.txt')!).at(-1)!.rows;
    expect(rows[10][0]).toBe('11');
    expect(rows[10][2]).toBe('1'); // Ncount of bin 11: the one atom at x = 3.0
  });

  it('bin/2d numbers the last dimension fastest', async () => {
    const { session, error, files } = await runScript(`${BOX}
create_atoms 1 single 1.0 3.0 1
create_atoms 1 single 6.0 9.0 9
compute c all chunk/atom bin/2d x lower 5.0 y lower 2.5 units box
fix f all ave/chunk 1 1 1 c vx file b2.txt
run 0`);
    expect(error).toBeNull();
    expect(session.sys.compute('c').scalarValue()).toBe(8);
    const rows = sections(files.get('b2.txt')!).at(-1)!.rows;
    // (x=1, y=3) is chunk 2; (x=6, y=9) is chunk 8
    expect(rows[1][0]).toBe('2');
    expect(rows[1][rows[1].length - 2]).toBe('1');
    expect(rows[7][rows[7].length - 2]).toBe('1');
  });
});

describe('compute chunk/atom other styles, keywords', () => {
  it('type chunks: Nchunk is the number of atom types (nchunk once)', async () => {
    const { session, error } = await runScript(`${BOX}
create_atoms 1 single 1.0 1 1
create_atoms 2 single 2.0 2 2
compute t all chunk/atom type
run 0`);
    expect(error).toBeNull();
    expect(session.sys.compute('t').scalarValue()).toBe(2);
  });

  it('compress yes renumbers the chunks and keeps the original value as OrigID', async () => {
    // molecule IDs 5 and 9 only: 2 chunks, OrigID column 5 and 9
    const { error, files } = await runScript(`
units           lj
atom_style      molecular
boundary        p p p
region          box block 0 10 0 10 0 10
create_box      2 box
mass            * 1.0
create_atoms 1 single 1.0 1 1
create_atoms 1 single 2.0 1 1
set atom 1 mol 5
set atom 2 mol 9
compute m all chunk/atom molecule compress yes
fix f all ave/chunk 1 1 1 m vx file comp.txt
run 0`);
    expect(error).toBeNull();
    const text = files.get('comp.txt')!;
    expect(text).toContain('# Chunk OrigID Ncount vx');
    const rows = sections(text).at(-1)!.rows;
    expect(rows.map((r) => r[1])).toEqual(['5', '9']);
  });

  it('argument errors name the problem', async () => {
    for (const [cmd, msg] of [
      ['compute c all chunk/atom bin/sphere 0 0 0 1 2 3', 'bin/sphere is not supported'],
      ['compute c all chunk/atom bin/cylinder z lower 2 1 1 1 2 3', 'bin/cylinder is not supported'],
      ['compute c all chunk/atom bin/1d q lower 1', 'dim must be x, y or z'],
      ['compute c all chunk/atom bin/1d x lower 0', 'delta must be > 0'],
      ['compute c all chunk/atom bin/1d x lower 1 units furlongs', 'units must be box, lattice or reduced'],
      ['compute c all chunk/atom type discard mixed', 'discard mixed is only for the binning styles'],
      ['compute c all chunk/atom type limit 2 sideways', "limit needs 'max' or 'exact'"],
      ['compute c all chunk/atom type ids nfreq', 'ids nfreq is not supported'],
      ['compute c all chunk/atom type pbc yes', 'pbc yes applies only to bin/sphere and bin/cylinder'],
      ['compute c all chunk/atom bin/1d x lower 1 bound y 0 5', 'bound given for a dimension with no bins'],
      ['compute c all chunk/atom type bogus 1', "unknown keyword 'bogus'"],
      ['compute c all chunk/atom nothing', "unknown chunk/atom style 'nothing'"],
      ['compute c all com/chunk nosuch', "compute ID 'nosuch' does not exist"],
      ['compute c all chunk/atom type\ncompute d all msd/chunk c\nrun 0', ''],
    ] as const) {
      const { error } = await runScript(`${BOX}\n${cmd}\nrun 0`);
      if (msg === '') continue;
      expect(error, cmd).toBeInstanceOf(Error);
      expect((error as Error).message, cmd).toContain(msg);
    }
  });

  it('fix ave/chunk argument errors', async () => {
    for (const [fix, msg] of [
      ['fix f all ave/chunk 2 1 3 c vx', 'Nfreq must be a multiple of Nevery'],
      ['fix f all ave/chunk 1 1 1 c', 'usage: fix f group-ID ave/chunk'],
      ['fix f all ave/chunk 1 1 1 c file x.txt', 'no input values'],
      ['fix f all ave/chunk 1 1 1 c vx norm bogus', 'norm must be all, sample or none'],
      ['fix f all ave/chunk 1 1 1 c vx bias t', 'bias keyword is not supported'],
      ['fix f all ave/chunk 1 1 1 c vx overwrite', 'overwrite keyword can only be used with the ave running setting'],
      ['fix f all ave/chunk 1 1 1 c vx file a.txt append b.txt', 'file and append cannot both be used'],
      ['fix f all ave/chunk 1 1 1 c vx format bogus', "invalid numeric format string 'bogus'"],
      ['fix f all ave/chunk 1 1 1 c vx q_1', "invalid input value 'q_1'"],
    ] as const) {
      const { error } = await runScript(`${BOX}
create_atoms 1 single 1.0 1 1
compute c all chunk/atom bin/1d x lower 5
${fix}
run 0`);
      expect(error, fix).toBeInstanceOf(Error);
      expect((error as Error).message, fix).toContain(msg);
    }
  });
});

describe('compute com/chunk and msd/chunk', () => {
  it('com/chunk gives the mass-weighted centre of each bin; msd/chunk is zero at the first call', async () => {
    const { session, error } = await runScript(`${BOX}
create_atoms 1 single 1.0 2.0 1
create_atoms 1 single 2.0 4.0 1
create_atoms 2 single 7.0 6.0 1
velocity all set 0 0 0 units box
compute c all chunk/atom bin/1d x lower 5.0 units box
compute com all com/chunk c
compute msd all msd/chunk c
run 0`);
    expect(error).toBeNull();
    const com = session.sys.compute('com').arrayValues();
    // bin 1 holds atoms at (1,2,1) and (2,4,1) of mass 1: centre (1.5, 3, 1)
    expect(com[0]).toBeCloseTo(1.5, 12);
    expect(com[1]).toBeCloseTo(3, 12);
    expect(com[2]).toBeCloseTo(1, 12);
    expect(session.sys.compute('msd').arrayValues().every((x) => x === 0)).toBe(true);
  });

  it('msd/chunk reports the squared displacement of the centre of mass from the first call', async () => {
    const { session, error } = await runScript(`${BOX}
create_atoms 1 single 1.0 1.0 1.0
create_atoms 1 single 2.0 1.0 1.0
velocity all set 0.5 0 0 units box
compute c all chunk/atom bin/1d x lower 5.0 units box
compute msd all msd/chunk c
fix nve all nve
timestep 0.1
run 0`);
    expect(error).toBeNull();
    // the reference is taken when the compute is first invoked (here: after run 0)
    expect(session.sys.compute('msd').arrayValues().every((x) => x === 0)).toBe(true);
    const second = await session.execute('run 2').then(() => null, (e: unknown) => e);
    expect(second).toBeNull();
    // the centre of mass of bin 1 moved by 0.5 * 0.1 * 2 = 0.1 in x
    const msd = session.sys.compute('msd').arrayValues();
    expect(msd[0]).toBeCloseTo(0.01, 10);
    expect(msd[3]).toBeCloseTo(0.01, 10);
  });
});
