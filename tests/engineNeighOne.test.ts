import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * neigh_modify one (neigh_modify.html: "one value = N  N = max # of neighbors of one atom"): a list build stops with
 * native's "Neighbor list overflow, boost neigh_modify one" when an atom's list holds more than N entries. The count of
 * a half list depends on which atom native's build stores each pair at (src/engine-gpl/neighbor/nbin.ts). Each case
 * gives the largest count M of its list: one M - 1 overflows and one M runs. Measured with native LAMMPS: every case
 * below overflows with one M - 1 and runs with one M.
 */

const ROOT = join(__dirname, '..');
const OVERFLOW = /^line \d+: Neighbor list overflow, boost neigh_modify one/;

const run = async (input: string, files: Record<string, string>): Promise<void> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  for (const [name, path] of Object.entries(files)) s.addFile(name, readFileSync(join(ROOT, path), 'utf8'));
  await s.execute(input);
};

const LJ = (box: string, extra = '') => `${extra}units lj
atom_style atomic
lattice fcc 0.8442
region box ${box}
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.3 12345
mass * 1.0
displace_atoms all random 0.1 0.1 0.1 4321
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
pair_coeff 1 2 1.0 1.0 2.8
`;
const ORTHO = 'block 0 4 0 4 0 4';
const TRI = 'prism 0 4 0 4 0 4 0.7 0.3 -0.5';
const SW = (box: string) => `units metal
atom_style atomic
lattice diamond 5.431
region box ${box}
create_box 1 box
create_atoms 1 box
mass 1 28.06
displace_atoms all random 0.2 0.2 0.2 4321
pair_style sw
pair_coeff * * Si.sw Si
`;
const CHAINS = (special: string, extra = '') => `${extra}units lj
atom_style molecular
bond_style harmonic
angle_style harmonic
dihedral_style harmonic
improper_style harmonic
pair_style lj/cut 2.5
read_data mol_chains.data
pair_coeff * * 0.5 0.9
pair_coeff 1 3 0.8 1.0
bond_coeff 1 120.0 1.05
bond_coeff 2 90.0 1.1
angle_coeff 1 40.0 109.5
angle_coeff 2 25.0 120.0
dihedral_coeff 1 1.5 -1 3
improper_coeff 1 12.0 20.0
special_bonds ${special}
`;
const SLAB = (cut: number) => `boundary p p f
kspace_style pppm 1.0e-5
kspace_modify slab 8.0
units real
atom_style full
pair_style lj/cut/coul/long ${cut}
read_data w68elec_au_elyt.data
pair_coeff * * 0 1
`;

const SI = { 'Si.sw': 'third_party/lammps/potentials/Si.sw' };
const CHAIN_DATA = { 'mol_chains.data': 'tests/oracle/mol_chains.data' };
const SLAB_DATA = { 'w68elec_au_elyt.data': 'tests/oracle/w68elec_au_elyt.data' };

const CASES: { name: string; body: string; m: number; files?: Record<string, string> }[] = [
  { name: 'half list, orthogonal box (half/bin/newton)', body: LJ(ORTHO), m: 68 },
  { name: 'half list, triclinic box (half/bin/newton/tri, ID parity)', body: LJ(TRI), m: 66 },
  { name: 'neighbor nsq', body: `${LJ(ORTHO)}neighbor 0.3 nsq\n`, m: 68 },
  { name: 'neighbor nsq, triclinic box', body: `${LJ(TRI)}neighbor 0.3 nsq\n`, m: 66 },
  { name: 'newton off (half/bin/newtoff)', body: LJ(ORTHO, 'newton off\n'), m: 91 },
  { name: 'newton off, triclinic box', body: LJ(TRI, 'newton off\n'), m: 94 },
  { name: 'neigh_modify binsize', body: `${LJ(ORTHO)}neigh_modify binsize 0.77\n`, m: 62 },
  { name: 'shrink-wrapped box', body: LJ(ORTHO, 'boundary s p s\n'), m: 70 },
  {
    name: '2d',
    body: 'units lj\ndimension 2\natom_style atomic\nlattice sq2 0.8442\nregion box block 0 10 0 10 -0.5 0.5\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\ndisplace_atoms all random 0.1 0.1 0 4321\npair_style lj/cut 3.5\npair_coeff 1 1 1.0 1.0\n',
    m: 27,
  },
  { name: 'full list (pair sw)', body: SW('block 0 3 0 3 0 3'), m: 46, files: SI },
  { name: 'full list, triclinic box', body: SW('prism 0 3 0 3 0 3 1.1 0.4 -0.6'), m: 45, files: SI },
  { name: 'special bonds (lj 0.0 0.5 0.8)', body: CHAINS('lj 0.0 0.5 0.8'), m: 4, files: CHAIN_DATA },
  { name: 'special bonds excluding 1-2, 1-3 and 1-4 pairs', body: CHAINS('lj 0.0 0.0 0.0'), m: 2, files: CHAIN_DATA },
  { name: 'special bonds, newton off', body: CHAINS('lj 0.0 0.5 0.8', 'newton off\n'), m: 4, files: CHAIN_DATA },
  // a 1 x 1 x 20 slab with 4 atoms: the default one = 2000 is reached through the periodic images
  { name: 'many periodic images (cutoff 12)', body: SLAB(12), m: 1796, files: SLAB_DATA },
  { name: 'many periodic images (cutoff 13)', body: SLAB(13), m: 2072, files: SLAB_DATA },
];

describe('neigh_modify one: native per-atom list counts', () => {
  for (const c of CASES) {
    it(`${c.name}: one ${c.m - 1} overflows, one ${c.m} runs`, async () => {
      const tail = (one: number) => `neigh_modify one ${one}\nfix 1 all nve\nrun 0\n`;
      await expect(run(c.body + tail(c.m - 1), c.files ?? {})).rejects.toThrow(OVERFLOW);
      await expect(run(c.body + tail(c.m), c.files ?? {})).resolves.toBeUndefined();
    });
  }

  it('the default one = 2000 (native: pair_style lj/cut/coul/long 13 on the slab overflows, 12 runs)', async () => {
    await expect(run(`${SLAB(13)}run 0\n`, SLAB_DATA)).rejects.toThrow(OVERFLOW);
    await expect(run(`${SLAB(12)}run 0\n`, SLAB_DATA)).resolves.toBeUndefined();
  });

  it('pair hybrid styles are not checked (native gives each sub-style a list of its own cutoff)', async () => {
    // Measured with native LAMMPS: this input runs with one 136; the engine's single list would hold 137 entries
    const body = SW('block 0 3 0 3 0 3').replace('pair_style sw\npair_coeff * * Si.sw Si\n', 'pair_style hybrid/overlay lj/cut 6.5 sw\npair_coeff * * lj/cut 0.01 2.0\npair_coeff * * sw Si.sw Si\n');
    await expect(run(`${body}neigh_modify one 136\nfix 1 all nve\nrun 0\n`, SI)).resolves.toBeUndefined();
  });
});
