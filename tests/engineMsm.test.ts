import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { initMsmTables, msmGamma, msmDgamma } from '../src/engine-gpl/msm/tables';

/**
 * kspace_style msm and msm/cg (src/engine-gpl/msm, the port of LAMMPS src/KSPACE/msm.cpp and msm_cg.cpp): the refusals
 * and warnings, with the messages of native LAMMPS (each measured with the native binary on the input below). The
 * forces, energies and pressures are checked against native LAMMPS in tests/oracle/w209ms_*.in.
 */

const run = async (text: string): Promise<{ error: string | null; log: string[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  const log = (): string[] => events.flatMap((e) => (e.kind === 'log' ? [e.text] : []));
  try {
    await session.execute(text);
    return { error: null, log: log() };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e), log: log() };
  }
};

/** 64 rocksalt ions (2x2x2 cells), metal units. */
const ions = (lines: string, boundary = 'p p p') => `units metal
atom_style charge
boundary ${boundary}
lattice fcc 5.64
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box
lattice fcc 5.64 origin 0.5 0.0 0.0
create_atoms 2 box
mass * 22.99
set type 1 charge 1.0
set type 2 charge -1.0
${lines}
`;

const LJ = 'pair_style lj/cut/coul/msm 5.0\npair_coeff * * 0.01 3.0';

describe('kspace_style msm refusals (messages measured with native LAMMPS)', () => {
  it('a pressure tensor component with the default pressure/scalar yes', async () => {
    const { error } = await run(ions(`${LJ}\nkspace_style msm 1e-3\nthermo_style custom step pe press pxx\nrun 0`));
    expect(error).toContain("Must use 'kspace_modify pressure/scalar no' for tensor components with kspace_style msm");
  });

  it('a per-atom virial with the default pressure/scalar yes', async () => {
    const { error } = await run(ions(`${LJ}\nkspace_style msm 1e-3\ncompute s all stress/atom NULL\ncompute r all reduce sum c_s[1]\nthermo_style custom step pe c_r\nrun 0`));
    expect(error).toContain("Must use 'kspace_modify pressure/scalar no' to obtain per-atom virial with kspace_style MSM");
  });

  it('press alone with pressure/scalar yes runs, and so does fix npt iso (it reads the scalar pressure)', async () => {
    expect((await run(ions(`${LJ}\nkspace_style msm 1e-3\nthermo_style custom step pe press\nrun 0`))).error).toBeNull();
    expect((await run(ions(`${LJ}\nkspace_style msm 1e-3\nfix 1 all npt temp 300 300 0.1 iso 0 0 1.0\nrun 2`))).error).toBeNull();
  });

  it('a pair style without msm, and msm pair styles with another kspace style', async () => {
    const long = await run(ions('pair_style lj/cut/coul/long 5.0\npair_coeff * * 0.01 3.0\nkspace_style msm 1e-3\nrun 0'));
    expect(long.error).toContain('KSpace style msm is incompatible with Pair style lj/cut/coul/long');
    const pppm = await run(ions(`${LJ}\nkspace_style pppm 1e-3\nrun 0`));
    expect(pppm.error).toContain('KSpace style pppm is incompatible with Pair style lj/cut/coul/msm');
  });

  it('a system that is not charge neutral', async () => {
    const { error } = await run(ions(`set atom 1 charge 0.5\n${LJ}\nkspace_style msm 1e-3\nrun 0`));
    expect(error).toContain('System is not charge neutral, net charge = -0.5');
  });

  it('order other than 4, 6, 8 or 10', async () => {
    const { error } = await run(ions(`${LJ}\nkspace_style msm 1e-3\nkspace_modify order 5\nrun 0`));
    expect(error).toContain('MSM order must be 4, 6, 8, or 10');
  });

  it('msm/cg with the default pressure/scalar yes', async () => {
    const { error } = await run(ions(`${LJ}\nkspace_style msm/cg 1e-3\nrun 0`));
    expect(error).toContain("Must use 'kspace_modify pressure/scalar no' with kspace_style msm/cg");
  });

  it('kspace_style arguments and kspace_modify mesh', async () => {
    expect((await run(ions(`${LJ}\nkspace_style msm 2.0`))).error).toContain('Invalid relative accuracy 2 for kspace_style msm');
    expect((await run(ions(`${LJ}\nkspace_style msm/cg 1e-3 1e-5 7`))).error).toContain('Illegal kspace_style msm/cg command');
    expect((await run(ions(`${LJ}\nkspace_style msm 1e-3\nkspace_modify mesh 4 0 4`))).error)
      .toContain('Kspace_modify mesh parameters must be all zero or all positive');
  });

  it('pair hybrid with pressure/scalar yes is refused by the engine (native adds the hybrid partial energy)', async () => {
    const { error } = await run(ions('pair_style hybrid/overlay lj/cut 5.0 coul/msm 5.0\npair_coeff * * lj/cut 0.01 3.0\npair_coeff * * coul/msm\nkspace_style msm 1e-3\nrun 0'));
    expect(error).toContain('kspace_modify pressure/scalar yes with pair_style hybrid/overlay is not supported');
  });

  it('a 2d system and a triclinic box with msm/cg (msm.cpp init, msm_cg.cpp triclinic_support = 0)', async () => {
    const twoD = await run(`units lj\ndimension 2\natom_style charge\nregion box block 0 10 0 10 -0.5 0.5\ncreate_box 1 box\ncreate_atoms 1 single 1 1 0\ncreate_atoms 1 single 3 3 0\nset atom 1 charge 1\nset atom 2 charge -1\nmass 1 1\n${LJ}\nkspace_style msm 1e-3\nrun 0`);
    expect(twoD.error).toContain('Cannot (yet) use MSM with 2d simulation');
    const tri = await run(ions(`${LJ}\nkspace_style msm/cg 1e-3\nkspace_modify pressure/scalar no\nrun 0`).replace('region box block 0 2 0 2 0 2', 'region box prism 0 2 0 2 0 2 0.2 0 0'));
    expect(tri.error).toContain('KSpace style msm/cg does not yet support triclinic geometries');
  });
});

describe('msm/dielectric and lj/cut/coul/msm/dielectric (engine-gpl/dielectric)', () => {
  it('both need atom style dielectric (msm_dielectric.cpp init, pair_lj_cut_coul_msm_dielectric.cpp init_style)', async () => {
    const k = await run(ions(`${LJ}\nkspace_style msm/dielectric 1e-3\nrun 0`));
    expect(k.error).toContain('msm/dielectric requires atom style dielectric');
    const p = await run(ions('pair_style lj/cut/coul/msm/dielectric 5.0\npair_coeff * * 0.01 3.0\nkspace_style msm 1e-3\nrun 0'));
    expect(p.error).toContain('Pair lj/cut/coul/msm/dielectric requires atom style dielectric');
  });
});

describe('kspace_style msm set-up (measured with native LAMMPS)', () => {
  it('mesh 6 8 8: the mesh is moved to a power of 2 with a warning, and cutoff/adjust sets the Coulomb cutoff', async () => {
    // Measured with native LAMMPS: WARNING: Number of MSM mesh points changed to be a multiple of 2, then WARNING:
    // Adjusting Coulombic cutoff for MSM, new cutoff = 6.0469796
    const { error, log } = await run(ions(`${LJ}\nkspace_style msm 1e-3\nkspace_modify mesh 6 8 8 pressure/scalar no\nrun 0`));
    expect(error).toBeNull();
    const text = log.join('\n');
    expect(text).toContain('WARNING: Number of MSM mesh points changed to be a multiple of 2');
    expect(text).toContain('WARNING: Adjusting Coulombic cutoff for MSM, new cutoff = 6.0469796');
  });

  it('the default cutoff/adjust picks the grid from the cost estimate (measured: cutoff 11.197491, grid 4 4 4 here)', async () => {
    // Measured with native LAMMPS on this input: new cutoff = 11.197491
    const { error, log } = await run(ions(`${LJ}\nkspace_style msm 1e-3\nrun 0`));
    expect(error).toBeNull();
    expect(log.join('\n')).toContain('new cutoff = 11.197491');
  });
});

describe('MSM splitting function and tables (kspace.h gamma, pair.cpp init_tables)', () => {
  it('gamma is smooth at rho = 1 and equals 1/rho beyond; dgamma is its derivative', () => {
    for (const order of [4, 6, 8, 10]) {
      expect(msmGamma(order, 1)).toBeCloseTo(1, 14);
      expect(msmGamma(order, 2)).toBe(0.5);
      expect(msmDgamma(order, 1)).toBeCloseTo(-1, 13);
      for (const rho of [0.1, 0.5, 0.9]) {
        const h = 1e-6;
        expect(msmDgamma(order, rho)).toBeCloseTo((msmGamma(order, rho + h) - msmGamma(order, rho - h)) / (2 * h), 7);
      }
    }
  });

  it('the 12-bit tables interpolate the MSM kernel qqrd2e/r egamma between the float32 nodes (to 1e-6 of qqrd2e/r)', () => {
    const cut = 10, qqrd2e = 14.399645, order = 10;
    const t = initMsmTables(12, Math.SQRT2, cut, qqrd2e, order);
    const f32 = new Float32Array(1), i32 = new Int32Array(f32.buffer);
    for (const r of [1.6, 2.9, 4.4, 7.3, 9.9]) {
      const rsq = r * r;
      f32[0] = rsq;
      const k = (i32[0] & t.ncoulmask) >> t.ncoulshiftbits;
      const fraction = (f32[0] - t.rtable[k]) * t.drtable[k];
      const e = t.etable[k] + fraction * t.detable[k];
      const exact = (qqrd2e / r) * (1 - (r / cut) * msmGamma(order, r / cut));
      // the kernel vanishes at the cutoff, so the error is measured against the bare Coulomb term qqrd2e / r
      expect(Math.abs(e - exact) / (qqrd2e / r)).toBeLessThan(1e-6);
    }
  });
});
