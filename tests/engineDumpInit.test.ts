import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * The checks of a dump's init at each run (Output::init after the thermo init; src/dump_custom.cpp and
 * src/dump_local.cpp init_style at b332f28): the computes and fixes the columns and a dump_modify thresh read must
 * still exist, and the N of the dump command must be a multiple of each fix's output interval (per-atom for the
 * DumpCustom styles, local for dump local). Measured with native LAMMPS: each input below gives the message expected.
 */

const run = async (input: string): Promise<void> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  await s.execute(input);
};

const BASE = `units lj
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
fix a all ave/atom 1 10 10 vx
`;

// a chain of 4 atoms whose bonds write broken-bond rows to fix store/local brkbond every N steps
const BPM = (n: number, every: number) => `units lj
boundary f f f
atom_style bond
newton on off
comm_modify vel yes cutoff 2.6
region box block -10 10 -10 10 -10 10
create_box 1 box bond/types 1 extra/bond/per/atom 4
create_atoms 1 single -1.0 0.0 0.0
create_atoms 1 single 0.0 0.0 0.0
create_atoms 1 single 1.0 0.0 0.0
create_atoms 1 single 2.0 0.0 0.0
mass 1 1.0
create_bonds single/bond 1 1 2
create_bonds single/bond 1 2 3
create_bonds single/bond 1 3 4
pair_style bpm/spring
pair_coeff 1 1 1.0 1.0 1.0
special_bonds lj 0.0 1.0 1.0 coul 1.0 1.0 1.0
bond_style bpm/spring store/local brkbond ${n} time id1 id2
bond_coeff 1 1.0 0.04 1.0
fix 1 all nve
dump 2 all local ${every} broken.dump f_brkbond[1] f_brkbond[2] f_brkbond[3]
run 0
`;

describe('dump init: fix output intervals and missing references', () => {
  it('dump custom N not a multiple of the fix per-atom interval', async () => {
    await expect(run(`${BASE}dump d all custom 5 d.txt id f_a\nrun 0\n`)).rejects.toThrow(/^line \d+: Dump custom and fix not computed at compatible times\nFor more information see https:\/\/docs.lammps.org\/err0007/);
  });

  it('the N of the dump command counts, not dump_modify every', async () => {
    await expect(run(`${BASE}dump d all custom 20 d.txt id f_a\ndump_modify d every 5\nrun 0\n`)).resolves.toBeUndefined();
  });

  it('the style named as in the dump command (cfg, custom/gz, yaml)', async () => {
    await expect(run(`${BASE}dump d all cfg 5 d.*.cfg mass type xs ys zs f_a\nrun 0\n`)).rejects.toThrow(/Dump cfg and fix not computed at compatible times/);
    await expect(run(`${BASE}dump d all custom/gz 5 d.txt.gz id f_a\nrun 0\n`)).rejects.toThrow(/Dump custom\/gz and fix not computed at compatible times/);
    await expect(run(`${BASE}dump d all yaml 5 d.yaml id f_a\nrun 0\n`)).rejects.toThrow(/Dump yaml and fix not computed at compatible times/);
  });

  it('a fix the columns or a thresh read is deleted before the run', async () => {
    await expect(run(`${BASE}dump d all custom 10 d.txt id f_a\nunfix a\nrun 0\n`)).rejects.toThrow(/^line \d+: Could not find dump custom fix ID a$/);
    await expect(run(`${BASE}dump d all custom 10 d.txt id\ndump_modify d thresh f_a > 0\nunfix a\nrun 0\n`)).rejects.toThrow(/^line \d+: Could not find dump custom fix ID a$/);
  });

  it('dump local N not a multiple of the fix local interval (bond_style bpm store/local)', async () => {
    await expect(run(BPM(11, 10))).rejects.toThrow(/^line \d+: Dump local and fix brkbond not computed at compatible times\nFor more information see https:\/\/docs.lammps.org\/err0007/);
    await expect(run(BPM(10, 10))).resolves.toBeUndefined();
    await expect(run(BPM(10, 20))).resolves.toBeUndefined();
  });
});
