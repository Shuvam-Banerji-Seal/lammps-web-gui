import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { applyFileWrite } from '../src/engine/fileWrite';
import { readH5 } from './h5read';

/*
 * dump h5md (the GPL port in src/engine-gpl/h5md: dump_h5md.ts and the HDF5 writer hdf5.ts). The files are compared
 * with native ones (as HDF5 content) by the oracle cases dump_h5md, dump_h5md_2d and dump_h5md_chunks; here, the checks
 * of the constructor, dump_modify and init. Measured with native LAMMPS (b332f28 built with the H5MD package): each
 * input below stops with the message expected.
 */

const run = async (input: string): Promise<Map<string, string>> => {
  const files = new Map<string, string>();
  const s = new Session({
    emit: () => {},
    writeFile: (n: string, t: string, ap: boolean, at?: number) => files.set(n, applyFileWrite(files.get(n), t, ap, at)),
  } as never);
  await s.execute(input);
  return files;
};

const BASE = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
fix 1 all nve
`;

const fails = async (body: string, message: string): Promise<void> => {
  await expect(run(`${BASE}${body}run 2\n`)).rejects.toThrow(new RegExp(`^line \\d+: ${message.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`));
};

describe('dump h5md: native checks', () => {
  it('the dump command and its elements', async () => {
    await fails('dump 1 all h5md 5 e.h5\n', 'Illegal dump h5md command');
    await fails('dump 1 all h5md 5 e.h5 image position\n', 'Illegal dump h5md command');
    await fails('dump 1 all h5md 5 e.h5 position every\n', 'Illegal dump h5md command');
    await fails('dump 1 all h5md 5 e.h5 position every x\n', "Expected integer parameter instead of 'x' in input script or data file");
    await fails('dump 1 all h5md 5 e.h5 charge\n', 'Requesting non-allocated quantity q in dump_h5md');
    await fails('dump 1 all h5md 5 e.h5 position author a author b\n', 'Illegal dump h5md command: author argument repeated');
    await fails('dump 1 all h5md 5 e.h5 position bogus\n', 'Invalid argument to dump h5md');
    await fails('dump 1 all h5md 5 e.h5 position file_from nope\n', 'Cound not find dump_modify H5MD dump ID nope');
    await fails('dump 1 all h5md 5 e.h5 position file_from\n', 'Invalid number of arguments in dump h5md');
    await fails('dump 1 all h5md 5 e.h5 position\ndump 2 all h5md 5 f.h5 box yes file_from 1 velocity\n', 'Cannot set file_from in dump h5md after box or create_group');
    await fails('dump 1 all h5md 5 e.h5.gz position\n', 'Invalid dump h5md filename');
    await fails('dump 1 all h5md 5 e%.h5 position\n', 'Invalid dump h5md filename');
  });

  it('dump_modify and init', async () => {
    await fails('dump 1 all h5md 5 e.h5 position\ndump_modify 1 sort off\n', 'Dump h5md requires sorting by atom ID');
    await fails('dump 1 all h5md 5 e.h5 position\ndump_modify 1 unwrap\n', 'Illegal dump_modify command');
  });

  it('a triclinic box', async () => {
    const tri = BASE.replace('region box block 0 2 0 2 0 2', 'region box prism 0 2 0 2 0 2 0.1 0 0');
    await expect(run(`${tri}dump 1 all h5md 5 e.h5 position\n`)).rejects.toThrow(/Invalid domain for dump h5md\. Only triclinic domains supported\.$/);
  });

  it('what the browser engine refuses where native writes broken data', async () => {
    // native keeps charges in an int array and writes it as doubles, reading past its end (dump_h5md.ts)
    await expect(run(`${BASE.replace('atom_style atomic', 'atom_style charge')}dump 1 all h5md 5 e.h5 charge\n`)).rejects.toThrow(/dump h5md charge is not supported by the browser engine/);
    // Measured with native LAMMPS: a second dump that names an element the group already has (file_from 1 position)
    // prints 68 lines of HDF5 errors, the run goes on, and the file loses position/step and position/time
    await fails('dump 1 all h5md 5 e.h5 position\ndump 2 all h5md 5 e.h5 file_from 1 position\n', 'dump h5md: the particles group already has the element position');
  });
});

describe('dump h5md: what is written', () => {
  it('the H5MD layout: h5md, particles/<group> with step and time shared by position, image and box/edges', async () => {
    const files = await run(`${BASE}dump 1 all h5md 2 e.h5 position image velocity every 2 author "A B"\nrun 4\n`);
    const c = readH5(files.get('e.h5')!);
    expect(c.nodes['h5md'].attrs.version).toEqual({ type: 'i4', dims: [2], value: [1, 0] });
    expect(c.nodes['h5md/author'].attrs.name).toEqual({ type: 'S3', dims: null, value: ['A B'] });
    expect(c.nodes['h5md/creator'].attrs.version.value).toEqual(['2 Sep 2026']);
    expect(c.nodes['particles/all/box'].attrs.boundary).toEqual({ type: 'S9', dims: [3], value: ['periodic', 'periodic', 'periodic'] });
    expect(c.hardlinks).toEqual([
      ['particles/all/box/edges/step', 'particles/all/image/step', 'particles/all/position/step'],
      ['particles/all/box/edges/time', 'particles/all/image/time', 'particles/all/position/time'],
    ]);
    // records at steps 0, 2, 4; velocity every 2 dumps: 0, 4
    expect(c.nodes['particles/all/position/step'].values).toEqual([0, 2, 4]);
    expect(c.nodes['particles/all/velocity/step'].values).toEqual([0, 4]);
    expect(c.nodes['particles/all/position/value'].dims).toEqual([3, 32, 3]);
    expect(c.nodes['particles/all/position/value'].chunk).toEqual([1, 32, 3]);
  });

  it('write_dump writes a record only on a multiple of its interval', async () => {
    // As native LAMMPS does (src/write_dump.cpp at b332f28): the interval is max(1, the last run's steps) plus the step
    // modulo it. Measured with native LAMMPS: after run 7 and run 5 (step 12, interval 7) write_dump h5md writes the
    // groups and no record
    const files = await run(`${BASE}run 7\nrun 5\nwrite_dump all h5md q.h5 position\n`);
    const c = readH5(files.get('q.h5')!);
    expect(c.nodes['particles/all/position/value'].dims).toEqual([0, 32, 3]);
    const again = await run(`${BASE}run 6\nwrite_dump all h5md q.h5 position\n`);
    expect(readH5(again.get('q.h5')!).nodes['particles/all/position/step'].values).toEqual([6]);
  });
});
