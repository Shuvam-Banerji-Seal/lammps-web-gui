import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix mol/swap (src/engine/fix/mol_swap.ts): Monte Carlo itype<->jtype swaps of
 * all atoms of a random molecule, the Metropolis test at T, the ke velocity
 * rescaling, the charge swap and its inconsistent-charge warning, the attempts
 * and accepts global vector, and the argument errors. The random stream and the
 * energy path are pinned by the w30molswap_* oracle cases; these unit tests
 * cover the branches the oracle cases do not (charges, warnings, errors).
 */

const MOL2 = `w30 unit probe

2 atoms
0 bonds
2 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Masses

1 1.0
2 4.0

Atoms # molecular

1 1 1 0.0 0.0 0.0
2 1 2 2.0 0.0 0.0
`;

const CHARGE2 = `w30 charge probe

2 atoms
0 bonds
2 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Masses

1 1.0
2 1.0

Atoms # full

1 1 1 1.0 0.0 0.0 0.0
2 1 2 -1.0 2.0 0.0 0.0
`;

const CHARGE3 = `w30 charge probe 3

3 atoms
0 bonds
2 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Masses

1 1.0
2 1.0

Atoms # full

1 1 1 1.0 0.0 0.0 0.0
2 1 1 2.0 2.0 0.0 0.0
3 1 2 -1.0 1.0 0.0 0.0
`;

const MIX = `w30 mix probe

4 atoms
0 bonds
3 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Masses

1 1.0
2 1.0
3 1.0

Atoms # molecular

1 1 1 0.0 0.0 0.0
2 1 2 1.2 0.0 0.0
3 2 3 5.0 0.0 0.0
4 3 2 8.0 0.0 0.0
`;

const run = async (script: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const out = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => out.set(n, (ap ? out.get(n) ?? '' : '') + t),
  });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { error, files: out, logs, thermo };
};

/** Atom (id, type, q) rows of a write_dump ... id type q file. */
const dumpQ = (text: string) => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return lines.slice(k + 1).filter((l) => l.trim()).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    return { id: w[0], type: w[1], q: w[2] };
  }).sort((a, b) => a.id - b.id);
};

/** Atom (id, type, vz) rows of a write_dump ... id type vz file. */
const dumpV = (text: string) => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return lines.slice(k + 1).filter((l) => l.trim()).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    return { id: w[0], type: w[1], vz: w[2] };
  }).sort((a, b) => a.id - b.id);
};

const header = (style: string) => `units lj
atom_style ${style}
pair_style lj/cut 2.0
`;

describe('fix mol/swap', () => {
  it('rejects an atom style without molecule IDs', async () => {
    const { error } = await run(`units lj
atom_style atomic
region box block -10 10 -10 10 -10 10
create_box 2 box
create_atoms 1 single 0 0 0
mass 1 1.0
mass 2 1.0
pair_style lj/cut 2.0
pair_coeff * * 0.1 1.0
fix 1 all mol/swap 1 1 1 2 12345 1.0
run 0
`);
    expect(error).toMatch(/no molecule IDs/);
  });

  it('rejects itype == jtype, T <= 0, a bad ke, unknown keywords and out-of-range types', async () => {
    const script = (fix: string) => `${header('molecular')}read_data m.data
${fix}
run 0
`;
    const { error: same } = await run(script('fix 1 all mol/swap 1 1 1 1 12345 1.0'), { 'm.data': MOL2 });
    expect(same).toMatch(/itype and jtype must be different/);
    const { error: t0 } = await run(script('fix 1 all mol/swap 1 1 1 2 12345 0.0'), { 'm.data': MOL2 });
    expect(t0).toMatch(/T must be > 0/);
    const { error: ke } = await run(script('fix 1 all mol/swap 1 1 1 2 12345 1.0 ke maybe'), { 'm.data': MOL2 });
    expect(ke).toMatch(/ke must be yes or no/);
    const { error: kw } = await run(script('fix 1 all mol/swap 1 1 1 2 12345 1.0 foo'), { 'm.data': MOL2 });
    expect(kw).toMatch(/unknown keyword 'foo'/);
    const { error: big } = await run(script('fix 1 all mol/swap 1 1 1 3 12345 1.0'), { 'm.data': MOL2 });
    expect(big).toMatch(/larger than ntypes/);
  });

  it('swaps charges with the atom types (atom_style full)', async () => {
    const { error, files } = await run(`${header('full')}read_data q.data
pair_coeff * * 0.1 1.0
fix 1 all mol/swap 1 1 1 2 12345 1e9
run 1
write_dump all custom q.dump id type q
`, { 'q.data': CHARGE2 });
    expect(error).toBeNull();
    const atoms = dumpQ(files.get('q.dump')!);
    expect(atoms.map((a) => a.type)).toEqual([2, 1]);
    expect(atoms.map((a) => a.q)).toEqual([-1, 1]);
  });

  it('warns and leaves charges when itype charges are inconsistent', async () => {
    const { error, files, logs } = await run(`${header('full')}read_data q.data
pair_coeff * * 0.1 1.0
fix 1 all mol/swap 1 1 1 2 12345 1e9
run 1
write_dump all custom q.dump id type q
`, { 'q.data': CHARGE3 });
    expect(error).toBeNull();
    expect(logs.some((l) => l.includes('Cannot swap charges in fix mol/swap'))).toBe(true);
    const atoms = dumpQ(files.get('q.dump')!);
    expect(atoms.map((a) => a.type)).toEqual([2, 2, 1]);
    expect(atoms.map((a) => a.q)).toEqual([1, 2, -1]);
  });

  it('rescales velocities by sqrt(mass ratio) with ke yes and not with ke no', async () => {
    const script = (ke: string) => `${header('molecular')}read_data m.data
pair_coeff * * 0.1 1.0
velocity all set 0.0 0.0 1.0
group g2 type 2
velocity g2 set 0.0 0.0 -1.0
fix 1 all mol/swap 1 1 1 2 12345 1e9${ke}
run 1
write_dump all custom v.dump id type vz
`;
    const yes = await run(script(''), { 'm.data': MOL2 });
    expect(yes.error).toBeNull();
    expect(dumpV(yes.files.get('v.dump')!).map((a) => a.vz)).toEqual([0.5, -2]);
    const no = await run(script(' ke no'), { 'm.data': MOL2 });
    expect(no.error).toBeNull();
    expect(dumpV(no.files.get('v.dump')!).map((a) => a.vz)).toEqual([1, -1]);
  });

  it('reports cumulative attempts and accepts in the global vector on the N-step schedule', async () => {
    // N = 2: attempts on steps 1, 3, 5, ... (measured with native LAMMPS, black box)
    const { error, thermo } = await run(`units lj
atom_style molecular
pair_style lj/cut 2.0
read_data mix.data
pair_coeff 1 1 0.1 1.0
pair_coeff 1 2 0.5 1.0
pair_coeff 2 2 1.0 1.0
pair_coeff 1 3 0.3 1.0
pair_coeff 2 3 0.7 1.0
pair_coeff 3 3 0.2 1.0
fix 1 all mol/swap 2 1 1 2 482794 1.0
thermo 1
thermo_style custom step f_1[1] f_1[2]
run 4
`, { 'mix.data': MIX });
    expect(error).toBeNull();
    // rows: step 0 (0,0), 1 (1,a), 2 (1,a), 3 (2,b), 4 (2,b)
    expect(thermo.map((r) => r.step)).toEqual([0, 1, 2, 3, 4]);
    expect(thermo.map((r) => r['f_1[1]'])).toEqual([0, 1, 1, 2, 2]);
    for (let i = 1; i < thermo.length; i++) expect(thermo[i]['f_1[2]']).toBeGreaterThanOrEqual(thermo[i - 1]['f_1[2]']);
    expect(thermo[4]['f_1[2]']).toBeLessThanOrEqual(2);
  });
});
