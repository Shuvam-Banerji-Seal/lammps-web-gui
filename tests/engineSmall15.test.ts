import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Small-15 gaps: create_bonds many (docs.lammps.org/create_bonds.html), fix balance weight neigh
 * (docs.lammps.org/fix_balance.html, balance.html) and compute voronoi/atom neighbors / occupation
 * (docs.lammps.org/compute_voronoi_atom.html). Expected values come from native LAMMPS runs
 * (black box, 1 process) recorded in the module comments of the source files and in
 * tests/oracle/w15small_*.in (exact per-atom and thermo comparison is in engineOracle.test.ts).
 */

/** Runs input text in a fresh session and returns the files it wrote, by name. */
const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  await session.execute(text);
  return { events, files };
};

/** Bond lines (id type atom1 atom2) of a written data file. */
const bondLines = (data: string): number[][] => {
  const lines = data.split('\n');
  const k = lines.indexOf('Bonds');
  const out: number[][] = [];
  for (const l of lines.slice(k + 2)) {
    const w = l.trim().split(/\s+/).map(Number);
    if (w.length !== 4 || w.some((x) => !Number.isFinite(x))) break;
    out.push(w);
  }
  return out;
};

/** Header line "N bonds" of a written data file. */
const bondCount = (data: string): number => Number(/^(\d+) bonds$/m.exec(data)![1]);

/** Simple-cubic lattice, box L x L x L, one type, atom_style bond, lj/cut pair style, harmonic bonds. */
const sc = (L: number, extra = 8, cut = 2.0) => `
units lj
atom_style bond
lattice sc 1.0
region box block 0 ${L} 0 ${L} 0 ${L}
create_box 2 box bond/types 1 extra/bond/per/atom ${extra}
create_atoms 1 box
mass * 1.0
pair_style lj/cut ${cut}
pair_coeff * * 1.0 1.0
special_bonds lj 0 1 1
bond_style harmonic
bond_coeff 1 100.0 1.0
`;

describe('create_bonds many', () => {
  it('bonds every nearest-neighbour pair of a periodic 3x3x3 sc lattice once (81 bonds, lower ID first)', async () => {
    const { files } = await run(`${sc(3, 6, 1.5)}create_bonds many all all 1 0.9 1.1\nwrite_data a.data\n`);
    const data = files.get('a.data')!;
    expect(bondCount(data)).toBe(81);
    for (const [, , i, j] of bondLines(data)) expect(i).toBeLessThan(j);
  });

  it('is symmetric in the two groups: many hi all gives the same bonds as many all hi (measured 34 bonds)', async () => {
    const a = await run(`${sc(3)}group hi id 20:27\nspecial_bonds lj 0 1 1\ncreate_bonds many hi all 1 1.9 2.1\nwrite_data a.data\n`);
    const b = await run(`${sc(3)}group hi id 20:27\nspecial_bonds lj 0 1 1\ncreate_bonds many all hi 1 1.9 2.1\nwrite_data a.data\n`);
    expect(bondCount(a.files.get('a.data')!)).toBe(34);
    const key = (d: string) => bondLines(d).map((w) => `${w[2]}-${w[3]}`).sort().join(' ');
    expect(key(b.files.get('a.data')!)).toBe(key(a.files.get('a.data')!));
  });

  it('creates periodic images as separate bonds: two images at distance 1.4 and 1.6 in a box of 3 give two bonds', async () => {
    const { files } = await run(`
units lj
atom_style bond
region box block 0 3 0 3 0 3
create_box 2 box bond/types 1 extra/bond/per/atom 4
create_atoms 1 single 0 0.5 0.5
create_atoms 1 single 1.4 0.5 0.5
mass * 1.0
pair_style lj/cut 1.5
pair_coeff * * 1.0 1.0
special_bonds lj 0 1 1
bond_style harmonic
bond_coeff 1 100.0 1.0
create_bonds many all all 1 0.9 1.6
write_data a.data
`);
    expect(bondLines(files.get('a.data')!).map((w) => `${w[2]}-${w[3]}`)).toEqual(['1-2', '1-2']);
  });

  it('never bonds an atom to its own image, also when rmax equals the box length', async () => {
    const { files } = await run(`${sc(3, 200, 5.0)}create_bonds many all all 1 0.9 3.0\nwrite_data a.data\n`);
    const data = files.get('a.data')!;
    expect(bondLines(data).some((w) => w[2] === w[3])).toBe(false);
    expect(bondCount(data)).toBe(1566);
  });

  it('keeps the bond inclusive at both ends: rmin = rmax = 1.0 finds the 81 nearest neighbours', async () => {
    const { files } = await run(`${sc(3, 6, 1.5)}create_bonds many all all 1 1.0 1.0\nwrite_data a.data\n`);
    expect(bondCount(files.get('a.data')!)).toBe(81);
  });

  it('a repeated identical many adds nothing (the bonded pairs are skipped at their closest image)', async () => {
    const { files } = await run(`${sc(3, 6, 1.5)}create_bonds many all all 1 0.9 1.1\ncreate_bonds many all all 1 0.9 1.1\nwrite_data a.data\n`);
    expect(bondCount(files.get('a.data')!)).toBe(81);
  });

  it('skips a 1-3 pair only when the 1-3 special weight is zero (measured: lj 0 0 0 excludes, lj 0 1 1 includes)', async () => {
    const base = `
units lj
atom_style bond
lattice sc 1.0
region box block 0 5 0 5 0 5
create_box 2 box bond/types 1 extra/bond/per/atom 20
create_atoms 1 box
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
bond_style harmonic
bond_coeff 1 100.0 1.0
create_bonds single/bond 1 1 2
create_bonds single/bond 1 2 3
`;
    const off = await run(`${base}special_bonds lj 0 0 0\ncreate_bonds many all all 1 1.9 2.1\nwrite_data a.data\n`);
    const on = await run(`${base}special_bonds lj 0 1 1\ncreate_bonds many all all 1 1.9 2.1\nwrite_data a.data\n`);
    const has13 = (d: string) => bondLines(d).some((w) => w[2] === 1 && w[3] === 3);
    expect(has13(off.files.get('a.data')!)).toBe(false);
    expect(has13(on.files.get('a.data')!)).toBe(true);
  });

  it('does not wrap bonds across a non-periodic boundary (5x5x5 with x fixed gives 350 of the 375 periodic bonds)', async () => {
    const { files } = await run(`${sc(5, 6, 1.5)}change_box all boundary f p p\ncreate_bonds many all all 1 0.9 1.1\nwrite_data a.data\n`);
    expect(bondCount(files.get('a.data')!)).toBe(350);
  });

  it('rejects the forms native LAMMPS rejects', async () => {
    await expect(run(`${sc(3, 6, 1.5)}create_bonds many all all 1 0.9 1.1 special yes\n`)).rejects.toThrow(/keyword/);
    await expect(run(`${sc(3, 6, 1.5)}special_bonds lj 0.5 1 1\ncreate_bonds many all all 1 0.9 1.1\n`)).rejects.toThrow(/1-2/);
    await expect(run(`${sc(3, 6, 1.5)}create_bonds many all all 1 0.9 1.9\n`)).rejects.toThrow(/neighbor cutoff/);
    await expect(run(`${sc(3, 6, 5.0)}create_bonds many all all 1 0.9 3.2\n`)).rejects.toThrow(/box length/);
    await expect(run(`${sc(3, 6, 1.5)}create_bonds many all all 2 0.9 1.1\n`)).rejects.toThrow(/outside/);
    await expect(run(`${sc(3, 6, 1.5)}create_bonds many all nothere 1 0.9 1.1\n`)).rejects.toThrow(/nothere/);
  });

  it('rejects special no for the single styles (the engine always rebuilds the special list)', async () => {
    await expect(run(`${sc(3, 6, 1.5)}create_bonds single/bond 1 1 2 special no\n`)).rejects.toThrow(/special no/);
  });
});

/** sc lattice of L^3 atoms with a voronoi/atom compute; returns the local rows (owner, neighbour, area). */
const voronoiRows = async (extra: string, L = 3) => {
  const { files } = await run(`
units lj
atom_style atomic
lattice sc 1.0
region box block 0 ${L} 0 ${L} 0 ${L}
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 1.5
pair_coeff * * 1.0 1.0
compute 1 all voronoi/atom ${extra}
dump d1 all local 1 v.local index c_1[1] c_1[2] c_1[3]
run 0
`);
  const lines = files.get('v.local')!.split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ENTRIES'));
  return lines.slice(k + 1).filter((l) => l.trim()).map((l) => l.trim().split(/\s+/).slice(1).map(Number));
};

describe('compute voronoi/atom neighbors yes and occupation', () => {
  it('neighbors yes: six faces per atom of a periodic sc lattice, each to the periodic neighbour IDs with area 1', async () => {
    const rows = await voronoiRows('neighbors yes');
    expect(rows.length).toBe(162);
    const id = (x: number, y: number, z: number) => 1 + (((x % 3) + 3) % 3) + 3 * (((y % 3) + 3) % 3) + 9 * (((z % 3) + 3) % 3);
    const byOwner = new Map<number, number[]>();
    for (const [owner, nid, area] of rows) {
      expect(area).toBeCloseTo(1, 12);
      byOwner.set(owner, [...(byOwner.get(owner) ?? []), nid]);
    }
    for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
      const me = id(x, y, z);
      const want = [id(x + 1, y, z), id(x - 1, y, z), id(x, y + 1, z), id(x, y - 1, z), id(x, y, z + 1), id(x, y, z - 1)].sort((a, b) => a - b);
      expect([...(byOwner.get(me) ?? [])].sort((a, b) => a - b)).toEqual(want);
    }
  });

  it('neighbors yes with face_threshold keeps only faces larger than the threshold (area 1 faces drop at 1.5)', async () => {
    expect((await voronoiRows('neighbors yes face_threshold 1.5')).length).toBe(0);
  });

  it('rejects the combinations that are not measured', async () => {
    await expect(voronoiRows('occupation only_group')).rejects.toThrow(/occupation/);
    await expect(voronoiRows('occupation neighbors yes')).rejects.toThrow(/occupation/);
    await expect(voronoiRows('neighbors maybe')).rejects.toThrow(/neighbors/);
  });
});

describe('fix balance weight neigh', () => {
  const setup = (fix: string) => `
units lj
atom_style atomic
lattice sc 1.0
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 1.5
pair_coeff * * 1.0 1.0
${fix}
thermo_style custom step atoms f_1 f_1[1] f_1[2] f_1[3]
thermo_modify format float %.15g
run 0
`;

  it('accepts factor 1.0 and at run setup has no neighbor weight (native: the list is not used yet) so W = atom count', async () => {
    const { events } = await run(setup('fix 1 all balance 1 1.0 shift x 10 1.1 weight neigh 1.0'));
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    expect(rows[0]['f_1[1]']).toBe(27);
    expect(rows[0].f_1).toBe(0);
  });

  it('rejects other factors (measured: native left the neighbor weights unapplied for them)', async () => {
    await expect(run(setup('fix 1 all balance 1 1.0 shift x 10 1.1 weight neigh 0.6'))).rejects.toThrow(/factor/);
    await expect(run(setup('fix 1 all balance 1 1.0 shift x 10 1.1 weight neigh 0'))).rejects.toThrow(/positive/);
  });
});
