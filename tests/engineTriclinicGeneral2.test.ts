import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * General triclinic boxes, part 2: write_data triclinic/general, 2d general lattices
 * and create_box NULL, and dump_modify triclinic/general (docs.lammps.org/Howto_triclinic.html,
 * write_data.html, lattice.html, create_box.html, dump_modify.html). The reference texts below
 * are what native LAMMPS wrote for the same input (black box). Full parity: tests/oracle/w19trigen_*.in.
 */

const runEngine = async (text: string, data?: Record<string, string>) => {
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  for (const [n, t] of Object.entries(data ?? {})) session.addFile(n, t);
  await session.execute(text);
  return { files, events };
};

/** Token-wise comparison of two text blocks: numbers within tol, words equal. Returns null when they agree. */
const sameTokens = (a: string, b: string, tol: number): string | null => {
  const la = a.trim().split('\n'), lb = b.trim().split('\n');
  if (la.length !== lb.length) return `line count ${la.length} vs ${lb.length}`;
  for (let i = 0; i < la.length; i++) {
    const ta = la[i].trim().split(/\s+/), tb = lb[i].trim().split(/\s+/);
    if (ta.length !== tb.length) return `line ${i + 1}: '${la[i]}' vs '${lb[i]}'`;
    for (let k = 0; k < ta.length; k++) {
      const x = Number(ta[k]), y = Number(tb[k]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        if (Math.abs(x - y) > tol) return `line ${i + 1} token ${k + 1}: ${ta[k]} vs ${tb[k]}`;
      } else if (ta[k] !== tb[k]) return `line ${i + 1} token ${k + 1}: '${ta[k]}' vs '${tb[k]}'`;
    }
  }
  return null;
};

/** The first line names the writing program, so the comparison starts at line 2. */
const body = (t: string) => t.split('\n').slice(1).join('\n');

/* Native write_data output of: lattice custom 1.0 a1 -0.5 0.5 0.5 a2 0.5 -0.5 0.5 a3 0.5 0.5 -0.5 basis 0.0 0.0 0.0
 * triclinic/general; create_box 1 NULL 0 1 0 1 0 1; create_atoms 1 box; mass * 1.0; write_data ... triclinic/general */
const BCC_INPUT = `lattice custom 1.0 a1 -0.5 0.5 0.5 a2 0.5 -0.5 0.5 a3 0.5 0.5 -0.5 basis 0.0 0.0 0.0 triclinic/general
create_box 1 NULL 0 1 0 1 0 1
create_atoms 1 box
mass * 1.0
write_data w.data triclinic/general
`;
const BCC_NATIVE = `LAMMPS data file via write_data, version 2 Sep 2026, timestep = 0, units = lj

1 atoms
1 atom types

-0.6299605249474365 0.6299605249474365 0.6299605249474364 avec
0.6299605249474367 -0.6299605249474365 0.6299605249474365 bvec
0.6299605249474363 0.6299605249474365 -0.6299605249474363 cvec
0 0 0 abc origin

Masses

1 1

Atoms # atomic

1 1 0 0 0 0 0 0

Velocities

1 0 0 0
`;

/* Native write_data output of: read_data data.general (2 atoms, avec 1 -1 0, bvec 1 1 0, cvec 1 1 1, zero velocities) ; write_data triclinic/general */
const GEN_NATIVE = `LAMMPS data file via write_data, version 2 Sep 2026, timestep = 0, units = lj

2 atoms
1 atom types

0.9999999999999998 -1 0 avec
0.9999999999999999 0.9999999999999999 0 bvec
0.9999999999999999 0.9999999999999999 1 cvec
0 0 0 abc origin

Masses

1 1

Atoms # atomic

1 1 0.2 -1.3877787807814457e-17 0.1 0 0 0
2 1 0.8 -1.1102230246251565e-16 0.3 0 0 0

Velocities

1 0 0 0
2 0 0 0
`;

describe('write_data triclinic/general matches native text', () => {
  it('bcc primitive cell created by create_box NULL', async () => {
    const { files } = await runEngine(BCC_INPUT);
    expect(sameTokens(body(files.get('w.data')!), body(BCC_NATIVE), 1e-12)).toBeNull();
  });

  it('a general data file read back with its atoms rotated into the restricted frame', async () => {
    const data = 'general triclinic data file\n\n2 atoms\n1 atom types\n\n1 -1 0 avec\n1 1 0 bvec\n1 1 1 cvec\n0 0 0 abc origin\n\nMasses\n\n1 1\n\nAtoms # atomic\n\n1 1 0.2 0.0 0.1\n2 1 0.8 0.0 0.3\n\nVelocities\n\n1 0 0 0\n2 0 0 0\n';
    const { files } = await runEngine('read_data w19.data\nmass * 1.0\nwrite_data g.data triclinic/general\n', { 'w19.data': data });
    expect(sameTokens(body(files.get('g.data')!), body(GEN_NATIVE), 1e-12)).toBeNull();
  });
});

describe('2d general triclinic lattice and create_box', () => {
  const HEX = `dimension 2
lattice custom 1.0 a1 1.0 0.0 0.0 a2 0.5 0.86602540378 0.0 a3 0.0 0.0 1.0 basis 0.0 0.0 0.0 triclinic/general
`;

  it('accepts the hex primitive cell and writes the box in the general frame', async () => {
    const { files } = await runEngine(`${HEX}create_box 1 NULL 0 1 0 1 -0.5 0.5
create_atoms 1 box
mass * 1.0
write_data w.data triclinic/general
`);
    const text = files.get('w.data')!;
    // native (black box): avec 1.0745699318262956 0 0, bvec 0.5372849659131478 0.9306048590997147 0,
    // cvec 0 0 1.0745699318262956, abc origin 0 0 -0.5372849659131478
    const line = (w: string) => text.split('\n').find((l) => l.endsWith(w))!.split(' ').map(Number);
    expect(line('cvec')[2]).toBeCloseTo(1.0745699318262956, 12);
    expect(line('abc origin')[2]).toBeCloseTo(-0.5372849659131478, 12);
    expect(line('bvec')[1]).toBeCloseTo(0.9306048590997147, 12);
  });

  it('rejects a3 that is not (0 0 1) before scaling', async () => {
    await expect(runEngine(`dimension 2
lattice custom 1.0 a1 1.0 0.0 0.0 a2 0.5 0.86602540378 0.0 a3 0.0 0.0 2.0 basis 0.0 0.0 0.0 triclinic/general
`)).rejects.toThrow('a3 vector for a 2d simulation must be (0,0,1)');
  });

  it('rejects a1 or a2 with a z component', async () => {
    await expect(runEngine(`dimension 2
lattice custom 1.0 a1 1.0 0.0 0.2 a2 0.5 0.86602540378 0.0 a3 0.0 0.0 1.0 basis 0.0 0.0 0.0 triclinic/general
`)).rejects.toThrow('not compatible with 2d simulation');
  });

  it('rejects a3 with an x or y component', async () => {
    await expect(runEngine(`dimension 2
lattice custom 1.0 a1 1.0 0.0 0.0 a2 0.5 0.86602540378 0.0 a3 0.0 0.5 1.0 basis 0.0 0.0 0.0 triclinic/general
`)).rejects.toThrow('not compatible with 2d simulation');
  });

  it('rejects a left-handed a1,a2 in 2d', async () => {
    await expect(runEngine(`dimension 2
lattice custom 1.0 a1 1.0 0.0 0.0 a2 0.5 -0.86602540378 0.0 a3 0.0 0.0 1.0 basis 0.0 0.0 0.0 triclinic/general
`)).rejects.toThrow('must be right-handed');
  });

  it('rejects collinear a1,a2 in 2d', async () => {
    await expect(runEngine(`dimension 2
lattice custom 1.0 a1 1.0 0.0 0.0 a2 2.0 0.0 0.0 a3 0.0 0.0 1.0 basis 0.0 0.0 0.0 triclinic/general
`)).rejects.toThrow('Lattice primitive vectors are collinear');
  });

  it('rejects create_box NULL with clo chi other than -0.5 0.5 in 2d', async () => {
    await expect(runEngine(`${HEX}create_box 1 NULL 0 1 0 1 0 1
`)).rejects.toThrow('requires clo = -0.5 and chi = 0.5');
  });

  it('accepts a lattice scale of 2 with a3 = (0 0 1), because the a3 test is on the unscaled vector', async () => {
    const { files } = await runEngine(`dimension 2
lattice custom 2.0 a1 1.0 0.0 0.0 a2 0.5 0.86602540378 0.0 a3 0.0 0.0 1.0 basis 0.0 0.0 0.0 triclinic/general
create_box 1 NULL 0 1 0 1 -0.5 0.5
create_atoms 1 box
mass * 1.0
write_data w.data triclinic/general
`);
    expect(files.get('w.data')).toContain('abc origin');
  });
});

describe('write_data and dump_modify triclinic/general refuse what they cannot rotate', () => {
  it('write_data triclinic/general on an orthogonal box', async () => {
    await expect(runEngine(`lattice sc 1.0
region b block 0 2 0 2 0 2
create_box 1 b
create_atoms 1 box
mass * 1.0
write_data w.data triclinic/general
`)).rejects.toThrow('needs a general triclinic box');
  });

  it('dump_modify triclinic/general yes on a restricted box', async () => {
    await expect(runEngine(`lattice sc 1.0
region b prism 0 2 0 2 0 2 0 0 0 units box
create_box 1 b
create_atoms 1 box
mass * 1.0
dump 1 all custom 10 d.dump id x y z
dump_modify 1 triclinic/general yes
run 0
`)).rejects.toThrow('needs a general triclinic box');
  });

  it('dump_modify triclinic/general does not rotate dipole columns', async () => {
    await expect(runEngine(`atom_style dipole
${BCC_INPUT}dump 1 all custom 10 d.dump id mux
dump_modify 1 triclinic/general yes
run 0
`)).rejects.toThrow('does not rotate column mux');
  });
});

describe('the general triclinic rotation survives a restart file', () => {
  it('write_restart then read_restart gives the same triclinic/general output', async () => {
    const first = await runEngine(`${BCC_INPUT}write_restart w19.rst
`);
    const rst = first.files.get('w19.rst');
    expect(rst).toBeDefined();
    const again = await runEngine('read_restart w19.rst\nwrite_data g.data triclinic/general\n', { 'w19.rst': rst! });
    expect(sameTokens(body(again.files.get('g.data')!), body(first.files.get('w.data')!), 1e-12)).toBeNull();
  });
});

describe('thermo_modify triclinic/general and change_box on a general box', () => {
  const GEN_DATA = 'general triclinic data file\n\n2 atoms\n1 atom types\n\n1 -1 0 avec\n1 1 0 bvec\n1 1 1 cvec\n0 0 0 abc origin\n\nMasses\n\n1 1\n\nAtoms # atomic\n\n1 1 0.2 0.0 0.1\n2 1 0.8 0.0 0.3\n\nVelocities\n\n1 0 0 0\n2 0 0 0\n';

  it('rotates the cell vectors and the pressure tensor to the general frame', async () => {
    // measured with native LAMMPS (black box): avec 1 -1 0, bvec 1 1 0, cvec 1 1 1; Pxx 5109.299, Pyy 1.821845, Pzz 598.94037
    const { events } = await runEngine('read_data w19.data\nmass * 1.0\npair_style lj/cut 1.2\npair_coeff * * 1.0 1.0\nneighbor 0.0 bin\nthermo_style custom step pe avecx avecy bvecx bvecy cvecz pxx pyy pzz\nthermo_modify triclinic/general yes\nrun 0\n', { 'w19.data': GEN_DATA });
    const row = (events.find((e) => e.kind === 'thermo') as { row: Record<string, number> }).row;
    expect(row.avecx).toBeCloseTo(1, 9);
    expect(row.avecy).toBeCloseTo(-1, 9);
    expect(row.bvecy).toBeCloseTo(1, 9);
    expect(row.pxx).toBeCloseTo(5109.299, 2);
    expect(row.pyy).toBeCloseTo(1.821845, 4);
    expect(row.pzz).toBeCloseTo(598.94037, 3);
  });

  it('refuses thermo_modify triclinic/general yes on a restricted box', async () => {
    await expect(runEngine(`lattice sc 1.0
region b prism 0 2 0 2 0 2 0.5 0 0 units box
create_box 1 b
thermo_modify triclinic/general yes
`)).rejects.toThrow('cannot be used if simulation box is not general triclinic');
  });

  it('change_box x scale 2 keeps the general rotation (native: avec 1.9999999999999996 -2 0)', async () => {
    const { files } = await runEngine('read_data w19.data\nmass * 1.0\nchange_box all x scale 2.0 remap\nwrite_data g.data triclinic/general\n', { 'w19.data': GEN_DATA });
    const avec = files.get('g.data')!.split('\n').find((l) => l.endsWith('avec'))!.split(' ').map(Number);
    expect(avec[0]).toBeCloseTo(2, 12);
    expect(avec[1]).toBeCloseTo(-2, 12);
    expect(avec[2]).toBeCloseTo(0, 12);
  });
});
