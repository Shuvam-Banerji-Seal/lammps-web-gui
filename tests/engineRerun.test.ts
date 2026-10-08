import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { StyleError } from '../src/engine/force/types';
import { parseRerunArgs } from '../src/engine/commands/rerun';
import {
  findSnapshot, parseDumpFile, parseDumpSpec, selectSnapshots,
} from '../src/engine/output/read_dump';

/** A native-format text dump with one atom per id, one snapshot per step. */
const dumpText = (steps: number[], opts: { n?: number; tri?: boolean; cols?: string } = {}): string => {
  const n = opts.n ?? 2;
  const cols = opts.cols ?? 'id type x y z';
  const box = opts.tri
    ? 'ITEM: BOX BOUNDS xy xz yz pp pp pp\n0 3 0.5\n0 3 0\n0 3 0.25'
    : 'ITEM: BOX BOUNDS pp pp pp\n0 3\n0 3\n0 3';
  return steps.map((s) => {
    const atoms = Array.from({ length: n }, (_, i) => `${i + 1} 1 ${0.5 + i} 0.5 0.5`).join('\n');
    return `ITEM: TIMESTEP\n${s}\nITEM: NUMBER OF ATOMS\n${n}\n${box}\nITEM: ATOMS ${cols}\n${atoms}`;
  }).join('\n') + '\n';
};

const sel = (steps: number[], o: { first?: number; last?: number; every?: number; skip?: number } = {}) => {
  const file = parseDumpFile('d.dump', dumpText(steps));
  return selectSnapshots([file], { first: 0, last: Number.MAX_SAFE_INTEGER, every: 0, skip: 1, ...o })
    .map((r) => r.snap.step);
};

const STEPS = [0, 5, 10, 15, 20];

describe('rerun argument parsing', () => {
  it('requires the dump keyword and it must be last', () => {
    expect(() => parseRerunArgs(['d.dump'])).toThrow(/missing keyword dump/);
    expect(() => parseRerunArgs(['d.dump', 'first', '3'])).toThrow(/missing keyword dump/);
  });

  it('requires at least one dump file', () => {
    expect(() => parseRerunArgs(['first', '3', 'dump', 'x'])).toThrow(StyleError);
  });

  it('needs field arguments after dump', () => {
    expect(() => parseRerunArgs(['d.dump', 'dump'])).toThrow(/needs field arguments/);
  });

  it('rejects first after last and bad every / skip values', () => {
    expect(() => parseRerunArgs(['d.dump', 'first', '30', 'last', '10', 'dump', 'x', 'y', 'z'])).toThrow(/first must come before last/);
    expect(() => parseRerunArgs(['d.dump', 'every', '-2', 'dump', 'x', 'y', 'z'])).toThrow('Invalid every value: -2 < 0');
    expect(() => parseRerunArgs(['d.dump', 'skip', '0', 'dump', 'x', 'y', 'z'])).toThrow('Invalid skip value: 0 <= 0');
    expect(() => parseRerunArgs(['d.dump', 'every', '2.5', 'dump', 'x'])).toThrow(/integer/);
  });

  it('names an unknown keyword and a keyword without a value', () => {
    expect(() => parseRerunArgs(['d.dump', 'first', '3', 'foo', '3', 'dump', 'x', 'y', 'z'])).toThrow(/unknown keyword 'foo'/);
    // before the first keyword every word is a file name, so 'foo' fails when the file is read
    expect(parseRerunArgs(['d.dump', 'foo', 'dump', 'x']).files).toEqual(['d.dump', 'foo']);
    expect(() => parseRerunArgs(['d.dump', 'first'])).toThrow(/keyword first needs a value/);
  });

  it('refuses a timestep keyword in the dump arguments and post values other than yes or no', () => {
    expect(() => parseRerunArgs(['d.dump', 'dump', 'x', 'timestep', 'no'])).toThrow(/timestep is not used/);
    expect(() => parseRerunArgs(['d.dump', 'post', 'maybe', 'dump', 'x'])).toThrow(/post must be yes or no/);
  });

  it('parses the selection and dump fields', () => {
    const a = parseRerunArgs(['a.dump', 'b.dump', 'first', '5', 'last', '30', 'every', '10', 'skip', '2', 'start', '0', 'stop', '40', 'post', 'no', 'dump', 'x', 'y', 'z', 'vx', 'box', 'no']);
    expect(a.files).toEqual(['a.dump', 'b.dump']);
    expect([a.first, a.last, a.every, a.skip, a.start, a.stop]).toEqual([5, 30, 10, 2, 0, 40]);
    expect(a.spec.fields).toEqual(['x', 'y', 'z', 'vx']);
    expect(a.spec.box).toBe(false);
  });
});

describe('read_dump field and keyword parsing', () => {
  it('reads the listed fields and keywords', () => {
    const s = parseDumpSpec(['x', 'y', 'z', 'vx', 'vy', 'vz', 'q', 'ix', 'iy', 'iz', 'replace', 'no', 'add', 'keep', 'label', 'x', 'xs', 'scaled', 'yes', 'format', 'native'], 'read_dump');
    expect(s.fields).toEqual(['x', 'y', 'z', 'vx', 'vy', 'vz', 'q', 'ix', 'iy', 'iz']);
    expect(s.replace).toBe(false);
    expect(s.add).toBe('keep');
    expect(s.labels.get('x')).toBe('xs');
    expect(s.scaled).toBe(true);
    expect(s.wrapped).toBe(true);
    expect(s.box).toBe(true);
  });

  it('names unsupported fields and formats instead of ignoring them', () => {
    expect(() => parseDumpSpec(['fx', 'fy', 'fz'], 'read_dump')).toThrow(/field fx is not supported/);
    expect(() => parseDumpSpec(['x', 'format', 'xyz'], 'read_dump')).toThrow(/dump format 'xyz' is not supported/);
    expect(() => parseDumpSpec(['x', 'format', 'molfile', 'dcd'], 'read_dump')).toThrow(/molfile/);
    expect(() => parseDumpSpec(['x', 'format', 'native', 'box', 'no'], 'read_dump')).toThrow(/must be the last keyword/);
    expect(() => parseDumpSpec(['x', 'nfile', '2'], 'read_dump')).toThrow(/nfile/);
  });

  it('rejects type as a field (native rejects it too) and fields after keywords', () => {
    expect(() => parseDumpSpec(['type', 'x'], 'read_dump')).toThrow(/'type' is not a read field/);
    expect(() => parseDumpSpec(['x', 'box', 'no', 'vx'], 'read_dump')).toThrow(/must come before the keywords/);
    expect(() => parseDumpSpec([], 'read_dump')).toThrow(/needs one or more fields/);
  });

  it('checks yes/no and add values and the label field', () => {
    expect(() => parseDumpSpec(['x', 'box', 'maybe'], 'read_dump')).toThrow(/box must be yes or no/);
    expect(() => parseDumpSpec(['x', 'add', 'sometimes'], 'read_dump')).toThrow(/add must be yes, keep or no/);
    expect(() => parseDumpSpec(['x', 'label', 'fx', 'fx'], 'read_dump')).toThrow(/label field must be one of/);
    expect(() => parseDumpSpec(['x', 'label', 'id'], 'read_dump')).toThrow(/keyword label needs a field and a column/);
  });

  it('rejects the purge combinations the native command rejects', () => {
    expect(() => parseDumpSpec(['x', 'purge', 'yes'], 'read_dump')).toThrow(/purge yes cannot be combined with replace yes/);
    expect(() => parseDumpSpec(['x', 'purge', 'yes', 'replace', 'no', 'trim', 'yes'], 'read_dump')).toThrow(/cannot be combined/);
    expect(() => parseDumpSpec(['x', 'purge', 'yes', 'replace', 'no'], 'read_dump')).toThrow(/needs add yes or keep/);
    expect(() => parseDumpSpec(['x', 'purge', 'yes', 'replace', 'no', 'add', 'yes'], 'read_dump')).not.toThrow();
  });
});

describe('dump file parsing', () => {
  it('splits snapshots and reads columns and bounds', () => {
    const f = parseDumpFile('d.dump', dumpText([0, 5]));
    expect(f.snapshots.map((s) => s.step)).toEqual([0, 5]);
    expect(f.snapshots[0].columns).toEqual(['id', 'type', 'x', 'y', 'z']);
    expect(f.snapshots[0].rows).toHaveLength(2);
    expect(f.snapshots[0].hi).toEqual([3, 3, 3]);
    expect(f.snapshots[0].triclinic).toBe(false);
  });

  it('converts triclinic bounding boxes to the true box (Howto_triclinic.html)', () => {
    // xlo_bound = xlo + MIN(0.0,xy,xz,xy+xz) with xy = 0.5, xz = 0: xlo = 0; xhi = 3 - MAX(0, 0.5, 0, 0.5) = 2.5
    const f = parseDumpFile('d.dump', dumpText([0], { tri: true }));
    const s = f.snapshots[0];
    expect(s.triclinic).toBe(true);
    expect(s.tilt).toEqual([0.5, 0, 0.25]);
    expect(s.lo).toEqual([0, 0, 0]);
    expect(s.hi[0]).toBeCloseTo(2.5, 12);
  });

  it('names the problem for a broken file and for unsupported file names', () => {
    expect(() => parseDumpFile('d.dump', 'hello\n')).toThrow(/expected 'ITEM: TIMESTEP'/);
    expect(() => parseDumpFile('d.dump', dumpText([0]).replace('ITEM: ATOMS id type x y z', 'ITEM: VELOCITIES'))).toThrow(/expected 'ITEM: ATOMS'/);
    expect(() => parseDumpFile('d.%.dump', '')).toThrow(/'%' wildcard/);
    expect(() => parseDumpFile('d.dump.gz', '')).toThrow(/gzipped/);
  });

  it('reports a missing snapshot by its timestep', () => {
    const f = parseDumpFile('d.dump', dumpText([0, 10]));
    expect(() => findSnapshot(f, 7)).toThrow('read_dump: no snapshot with timestep 7 in d.dump');
    expect(findSnapshot(f, 10).step).toBe(10);
  });
});

describe('rerun snapshot selection', () => {
  it('reads every snapshot without keywords', () => {
    expect(sel(STEPS)).toEqual(STEPS);
  });

  it('first and last bound the snapshots; a snapshot above last ends the list', () => {
    expect(sel(STEPS, { first: 6 })).toEqual([10, 15, 20]);
    expect(sel(STEPS, { last: 12 })).toEqual([0, 5, 10]);
    expect(sel(STEPS, { first: 7, last: 13 })).toEqual([10]);
  });

  it('every keeps multiples of N after the first snapshot, which is always read', () => {
    expect(sel(STEPS, { every: 10 })).toEqual([0, 10, 20]);
    expect(sel(STEPS, { first: 3, every: 10 })).toEqual([5, 10, 20]);
    expect(sel(STEPS, { every: 7 })).toEqual([0]);
    expect(sel(STEPS, { every: 0 })).toEqual(STEPS);
  });

  it('skip counts snapshots from the first one read', () => {
    expect(sel(STEPS, { skip: 2 })).toEqual([0, 10, 20]);
    expect(sel(STEPS, { skip: 3 })).toEqual([0, 15]);
    expect(sel(STEPS, { first: 5, skip: 2 })).toEqual([5, 15]);
    expect(sel(STEPS, { first: 6, skip: 2 })).toEqual([10, 20]);
    expect(sel(STEPS, { every: 10, skip: 2 })).toEqual([0, 10, 20]);
    expect(sel(STEPS, { first: 1, skip: 2, every: 10 })).toEqual([5]);
  });

  it('drops snapshots that are not in ascending order across files', () => {
    const a = parseDumpFile('a.dump', dumpText([0, 5, 10]));
    const b = parseDumpFile('b.dump', dumpText([10, 15]));
    const got = selectSnapshots([a, b], { first: 0, last: 1e9, every: 0, skip: 1 }).map((r) => r.snap.step);
    expect(got).toEqual([0, 5, 10, 15]);
  });

  it('refuses skip above 1 across several files (not reproduced)', () => {
    const a = parseDumpFile('a.dump', dumpText([0, 5]));
    const b = parseDumpFile('b.dump', dumpText([10]));
    expect(() => selectSnapshots([a, b], { first: 0, last: 1e9, every: 0, skip: 2 })).toThrow(/skip > 1 with more than one dump file/);
  });
});

describe('read_dump and rerun through the session', () => {
  const setup = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 1 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

  const run = async (script: string, files: Record<string, string> = {}) => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e) });
    for (const [n, t] of Object.entries(files)) session.addFile(n, t);
    await session.execute(script);
    return events;
  };

  it('reports a missing dump file, a missing snapshot and a missing box', async () => {
    await expect(run(`${setup}rerun nope.dump dump x y z\n`)).rejects.toThrow(/cannot open file nope.dump/);
    await expect(run(`${setup}read_dump d.dump 7 x y z\n`, { 'd.dump': dumpText([0, 10]) })).rejects.toThrow('no snapshot with timestep 7 in d.dump');
    await expect(run('rerun d.dump dump x y z\n', { 'd.dump': dumpText([0]) })).rejects.toThrow(/needs a simulation box/);
  });

  it('rejects an unsupported read_dump format before reading the file', async () => {
    await expect(run(`${setup}read_dump d.dump 0 x y z format xyz\n`, { 'd.dump': dumpText([0]) })).rejects.toThrow(/dump format 'xyz' is not supported/);
  });

  it('rejects a field whose column is missing from the dump', async () => {
    await expect(run(`${setup}read_dump d.dump 0 vx\n`, { 'd.dump': dumpText([0], { cols: 'id type x y z' }) })).rejects.toThrow(/column vx \(field vx\) not found/);
  });

  it('runs one thermo row per snapshot and emits the thermo header once', async () => {
    const dump = dumpText([0, 5, 10]);
    const events = await run(`${setup}thermo_style custom step pe\nthermo 1\nrerun d.dump dump x y z\n`, { 'd.dump': dump });
    const rows = events.flatMap((e) => (e.kind === 'thermo' ? [e.row.step] : []));
    expect(rows).toEqual([0, 5, 10]);
    expect(events.filter((e) => e.kind === 'thermo-header')).toHaveLength(1);
  });
});
