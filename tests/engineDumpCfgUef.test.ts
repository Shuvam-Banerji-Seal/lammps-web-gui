import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * dump cfg/uef (src/engine-gpl/dump/dump_cfg.ts, ported from src/UEF/dump_cfg_uef.cpp): dump cfg's checks, then a fix
 * nvt/uef or npt/uef at init; write_dump overrides the one-snapshot-per-file rule for "cfg" only (src/write_dump.cpp).
 * Measured with native LAMMPS: each input below gives the message expected. The files are compared with native ones by the
 * oracle case dump_cfg_uef.
 */

const BASE = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

const fails = async (body: string, message: string): Promise<void> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} } as never);
  await expect(s.execute(BASE + body)).rejects.toThrow(new RegExp(`^line \\d+: ${message.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`));
};

describe('dump cfg/uef: native checks', () => {
  it('needs a fix nvt/uef or npt/uef at init', async () => {
    await fails('dump 1 all cfg/uef 10 d.*.cfg mass type xs ys zs\nrun 0\n', "Can't use dump cfg/uef without defining a fix nvt/npt/uef");
    await fails('write_dump all cfg/uef w.*.cfg mass type xs ys zs\n', "Can't use dump cfg/uef without defining a fix nvt/npt/uef");
  });

  it('one snapshot per file, which write_dump overrides for cfg only', async () => {
    await fails('dump 1 all cfg/uef 10 d.cfg mass type xs ys zs\nrun 0\n', 'Dump cfg requires one snapshot per file');
    await fails('write_dump all cfg/uef w.cfg mass type xs ys zs\n', 'Dump cfg requires one snapshot per file');
    await fails('write_dump all cfg/gz w.cfg.gz mass type xs ys zs\n', 'Dump cfg requires one snapshot per file');
    await fails('change_box all triclinic\nfix 1 all nvt/uef temp 1 1 0.5 erate -0.5 -0.5\nwrite_dump all cfg/uef w.cfg mass type xs ys zs\n', 'Dump cfg requires one snapshot per file');
  });

  it("dump cfg's attribute rule", async () => {
    await fails('dump 1 all cfg/uef 10 d.*.cfg mass type x ys zs\n', "Dump cfg arguments must start with 'mass type xs ys zs' or 'mass type xsu ysu zsu'");
  });
});
