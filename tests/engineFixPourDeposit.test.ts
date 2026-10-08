import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix pour / fix deposit (src/engine/fix/pour.ts, deposit.ts): argument errors,
 * insertion count and timing, and the geometry of poured particles. The
 * random streams themselves are checked against native LAMMPS by the
 * w7pour_* and w7dep_* cases in tests/oracle.
 */

const run = async (script: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { error, thermo, files, logs };
};

/** Atom rows (id, x, y, z, radius) of a write_dump file. */
const dumpAtoms = (text: string): { id: number; x: number; y: number; z: number; r: number }[] => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return lines.slice(k + 1).filter((l) => l.trim()).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    return { id: w[0], x: w[1], y: w[2], z: w[3], r: w[4] };
  });
};

const header = `units lj
atom_style sphere
comm_modify vel yes
boundary p p f
region box block -10 10 -10 10 -10 10
create_box 1 box
pair_style gran/hooke 2000.0 700.0 50.0 30.0 0.5 1
pair_coeff * *
timestep 0.001
fix 1 all nve/sphere
fix 2 all gravity 1.0 vector 0 0 -1
`;

const deposHeader = `units lj
atom_style atomic
boundary p p f
region box block -10 10 -10 10 -10 10
create_box 1 box
mass 1 1.0
fix 1 all nve
`;

describe('fix pour / fix deposit argument errors', () => {
  it('pour needs the region keyword', async () => {
    const r = await run(`${header}fix 3 all pour 5 1 12345 diam one 1.0\n`);
    expect(r.error).toMatch(/requires the region keyword/);
  });

  it('pour rejects the seed 0 and a non-integer N', async () => {
    expect((await run(`${header}region r block 0 3 0 3 0 1 side in units box\nfix 3 all pour 5 1 0 region r\n`)).error).toMatch(/seed must be a positive integer/);
    expect((await run(`${header}region r block 0 3 0 3 0 1 side in units box\nfix 3 all pour 2.5 1 7 region r\n`)).error).toMatch(/N must be a positive integer/);
  });

  it('pour names unsupported molecule keywords', async () => {
    const r = await run(`${header}region r block 0 3 0 3 0 1 side in units box\nfix 3 all pour 5 1 7 region r mol tmpl\n`);
    expect(r.error).toMatch(/keyword 'mol' is not supported/);
  });

  it('pour requires atom_style sphere and a gravity fix', async () => {
    const atomic = `units lj
atom_style atomic
boundary p p f
region box block -10 10 -10 10 -10 10
create_box 1 box
mass 1 1.0
region r block 0 3 0 3 0 1 side in units box
fix 3 all pour 5 1 7 region r
run 1
`;
    expect((await run(atomic)).error).toMatch(/atom_style sphere/);
    const nogravity = `${header.replace('fix 2 all gravity 1.0 vector 0 0 -1\n', '')}region r block 0 3 0 3 0 1 side in units box
fix 3 all pour 5 1 7 region r vol 0.5 50
run 1
`;
    expect((await run(nogravity)).error).toMatch(/requires a fix gravity/);
  });

  it('pour needs side in and rejects poly percentages that do not sum to one', async () => {
    const sideOut = `${header}region r block 0 3 0 3 0 1 side out units box\nfix 3 all pour 5 1 7 region r\nrun 1\n`;
    expect((await run(sideOut)).error).toMatch(/side in/);
    const poly = `${header}region r block 0 3 0 3 0 1 side in units box\nfix 3 all pour 5 1 7 region r diam poly 2 0.5 0.3 0.8 0.3\nrun 1\n`;
    expect((await run(poly)).error).toMatch(/sum to 1/);
  });

  it('deposit rejects units lattice, var and unknown keywords', async () => {
    const base = `${deposHeader}region r block 0 3 0 3 0 1 side in units box\n`;
    expect((await run(`${base}fix 3 all deposit 2 1 1 7 region r\nrun 1\n`)).error).toMatch(/units lattice/);
    expect((await run(`${base}fix 3 all deposit 2 1 1 7 region r units box var v\nrun 1\n`)).error).toMatch(/'var' is not supported/);
    expect((await run(`${base}fix 3 all deposit 2 1 1 7 region r units box bogus 1\nrun 1\n`)).error).toMatch(/unknown keyword 'bogus'/);
  });
});

describe('fix pour insertion count and timing', () => {
  it('one particle per event, events spaced by the fall time', async () => {
    // region volume 9 * 0.06 / (pi/6) = 1.03 -> one particle per event; fall time sqrt(2) at g = 1
    const script = `${header}region reg block 0 3 0 3 0 1 side in units box
fix 3 all pour 3 1 12345 region reg diam one 1.0 dens 1.0 1.0 vol 0.06 100
thermo_style custom step atoms
thermo 1
run 2900
`;
    const r = await run(script);
    expect(r.error).toBeNull();
    const at = (step: number) => r.thermo.find((t) => t.step === step)!.atoms;
    expect(at(1)).toBe(1);
    expect(at(1414)).toBe(1);
    expect(at(1415)).toBe(2);
    expect(at(2829)).toBe(3);
    expect(at(2900)).toBe(3);
  });

  it('the count is floor(vol V / Vp) and N caps the total', async () => {
    // 9 * 0.2 / 0.5236 = 3.44 -> 3 per event; N = 4 -> 3 at step 1, then 1 at the next event (step 1415)
    const script = `${header}region reg block 0 3 0 3 0 1 side in units box
fix 3 all pour 4 1 99 region reg diam one 1.0 dens 1.0 1.0 vol 0.2 100
thermo_style custom step atoms
thermo 1
run 1500
`;
    const r = await run(script);
    expect(r.error).toBeNull();
    expect(r.thermo.find((t) => t.step === 1)!.atoms).toBe(3);
    expect(r.thermo.find((t) => t.step === 1414)!.atoms).toBe(3);
    expect(r.thermo.find((t) => t.step === 1415)!.atoms).toBe(4);
    expect(r.thermo.at(-1)!.atoms).toBe(4);
  });

  it('a zero count is an error, as in native LAMMPS', async () => {
    const script = `${header}region reg block 0 1 0 1 0 1 side in units box
fix 3 all pour 3 1 12345 region reg diam one 1.0 dens 1.0 1.0 vol 0.3 100
run 2
`;
    expect((await run(script)).error).toMatch(/insertion count per timestep is 0/);
  });

  it('a too small attempt budget warns and inserts fewer particles', async () => {
    const script = `${header}region reg block 0 1 0 1 0 2 side in units box
fix 3 all pour 3 1 12345 region reg diam one 1.0 dens 1.0 1.0 vol 0.9 10
thermo_style custom step atoms
thermo 1
run 1
`;
    const r = await run(script);
    expect(r.error).toBeNull();
    expect(r.logs.some((l) => /WARNING: Fewer insertions than requested \(1 vs 3\)/.test(l))).toBe(true);
    expect(r.thermo.at(-1)!.atoms).toBe(1);
  });
});

describe('fix deposit insertion count and timing', () => {
  it('inserts every M steps starting at the first step, until N', async () => {
    const script = `${deposHeader}region reg block 0 1 0 1 0 1 side in units box
fix 3 all deposit 3 1 2 12345 region reg units box
thermo_style custom step atoms
thermo 1
timestep 0.001
run 8
`;
    const r = await run(script);
    expect(r.error).toBeNull();
    const at = (step: number) => r.thermo.find((t) => t.step === step)!.atoms;
    expect([1, 2, 3, 4, 5, 6, 7, 8].map(at)).toEqual([1, 1, 2, 2, 3, 3, 3, 3]);
  });
});

describe('fix pour particle geometry', () => {
  it('poured particles lie inside the region and do not overlap', async () => {
    const script = `${header}region reg block 0 9 0 9 0 1 side in units box
fix 3 all pour 30 1 4242 region reg diam range 0.5 1.0 dens 1.0 2.0 vel -1 1 -1 1 0 vol 0.2 100
run 1
write_dump all custom pour_check.dump id x y z radius modify format float %.15g sort id
`;
    const r = await run(script);
    expect(r.error).toBeNull();
    const atoms = dumpAtoms(r.files.get('pour_check.dump') ?? '');
    expect(atoms.length).toBe(30);
    for (const a of atoms) {
      expect(a.x).toBeGreaterThanOrEqual(-1e-6);
      expect(a.x).toBeLessThanOrEqual(9 + 1e-6);
      expect(a.y).toBeGreaterThanOrEqual(-1e-6);
      expect(a.y).toBeLessThanOrEqual(9 + 1e-6);
      expect(a.z).toBeGreaterThanOrEqual(-1e-6);
      expect(a.z).toBeLessThanOrEqual(1 + 1e-6);
      expect(a.r).toBeGreaterThanOrEqual(0.25 - 1e-9);
      expect(a.r).toBeLessThanOrEqual(0.5 + 1e-9);
    }
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const d = Math.hypot(atoms[i].x - atoms[j].x, atoms[i].y - atoms[j].y, atoms[i].z - atoms[j].z);
        expect(d, `particles ${atoms[i].id} and ${atoms[j].id}`).toBeGreaterThanOrEqual(atoms[i].r + atoms[j].r - 1e-6);
      }
    }
  });

  it('a cylinder region keeps particles inside the radius', async () => {
    const script = `${header}region cyl cylinder z 0 0 3 0 1 side in units box
fix 3 all pour 10 1 31 region cyl diam one 1.0 dens 1.0 1.0 vol 0.3 100
run 1
write_dump all custom cyl_check.dump id x y z radius modify format float %.15g sort id
`;
    const r = await run(script);
    expect(r.error).toBeNull();
    const atoms = dumpAtoms(r.files.get('cyl_check.dump') ?? '');
    expect(atoms.length).toBeGreaterThan(0);
    for (const a of atoms) expect(Math.hypot(a.x, a.y)).toBeLessThanOrEqual(3 + 1e-6);
  });
});
