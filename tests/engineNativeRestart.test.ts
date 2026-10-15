import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * read_restart of a native LAMMPS binary restart file (src/engine-gpl/restart/nativeRestart.ts). The files are written by
 * the native binary (b332f28) in tests/oracle/w184nr_*.restart; the oracle cases in tests/oracle/w184nr_*.in compare a
 * run that continues from them with the native continuation. These tests cover what the oracle cases do not: the state
 * read back, and the refusals that name what the engine cannot read.
 */

const ORACLE = join(__dirname, 'oracle');

/** A session with the named oracle files added as the app adds uploads (latin1 for a binary file). */
const withFiles = (names: string[]) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const n of names) session.addFile(n, readFileSync(join(ORACLE, n)).toString('latin1'));
  return { session, events };
};

describe('read_restart of a native binary restart file', () => {
  it('restores the box, the atoms, the masses and the pair style of an lj/cut file', async () => {
    const { session, events } = withFiles(['w184nr_lj.restart']);
    await session.execute('read_restart w184nr_lj.restart\n');
    const s = session.sys.state;
    expect(s.n).toBe(108);
    expect(s.step).toBe(100);
    expect(s.atomStyle).toBe('atomic');
    expect(s.ntypes).toBe(2);
    expect(Array.from(s.id.subarray(0, 4))).toEqual([1, 2, 3, 4]);
    expect(s.type.filter((t) => t === 2).length).toBe(40);
    expect(s.box.lo).toEqual([0, 0, 0]);
    expect(s.box.hi[0]).toBeCloseTo(3 * 1.6796, 3);
    expect(session.sys.ff.pair?.name).toBe('lj/cut');
    expect(session.sys.ff.pair?.shift).toBe(true);
    expect(events.some((e) => e.kind === 'log' && 'text' in e && /Restoring from native restart file w184nr_lj\.restart: 108 atoms at step 100/.test(e.text))).toBe(true);
  });

  it('restores a triclinic box, its group names and the image flags of the atoms', async () => {
    const { session } = withFiles(['w184nr_tri.restart']);
    await session.execute('read_restart w184nr_tri.restart\n');
    const s = session.sys.state;
    expect(s.box.triclinic).toBe(true);
    // Measured with native LAMMPS (black box): create_box reports the tilt (2.4662121 1.6441414 0.82207069)
    expect(s.box.tilt[0]).toBeCloseTo(2.4662121, 6);
    expect(s.box.tilt[2]).toBeCloseTo(0.82207069, 6);
    expect(session.sys.groups.names[1]).toBe('g2');
    expect(s.image.some((v) => v !== 0)).toBe(true);
  });

  it('reads a file written on 4 processes (one chunk per process) to the state native LAMMPS reads from it', async () => {
    // Measured with native LAMMPS (black box): read_restart of w184nr_lj4.restart, then run 0, prints step 100,
    // temp 0.820570275543923, E_pair -5.01888483894505, press 0.849251865732799
    const { session, events } = withFiles(['w184nr_lj4.restart']);
    await session.execute('read_restart w184nr_lj4.restart\nthermo_style custom step temp pe press\nthermo_modify format float %.15g\nrun 0\n');
    expect(session.sys.state.n).toBe(108);
    const row = events.find((e) => e.kind === 'thermo' && 'row' in e && e.row.step === 100);
    expect(row && 'row' in row ? row.row.temp : NaN).toBeCloseTo(0.820570275543923, 12);
    expect(row && 'row' in row ? row.row.pe : NaN).toBeCloseTo(-5.01888483894505, 12);
  });

  it('restores the molecular topology of a full-style file: bonds and angles without repeats', async () => {
    const { session } = withFiles(['w184nr_mol.restart']);
    await session.execute('read_restart w184nr_mol.restart\n');
    const t = session.sys.state.topo;
    expect(session.sys.state.atomStyle).toBe('full');
    expect(t.bonds.n).toBe(16);
    expect(t.angles.n).toBe(12);
    expect(session.sys.ff.bond?.name).toBe('harmonic');
    expect(session.sys.ff.angle?.name).toBe('harmonic');
    expect(session.sys.ff.pair?.name).toBe('lj/cut/coul/long');
    expect(session.sys.ff.special.lj).toEqual([0, 0.5, 1]);
  });

  it('records the NO_PAIR marker of pair eam: the pair style is not restored', async () => {
    const { session, events } = withFiles(['w184nr_eam.restart']);
    await session.execute('read_restart w184nr_eam.restart\n');
    expect(session.sys.ff.pair).toBeNull();
    expect(session.sys.ff.pairNotRestarted).toBe('eam');
    expect(events.some((e) => e.kind === 'log' && 'text' in e && /pair style eam stores no restart info/.test(e.text))).toBe(true);
  });

  it('restores the thermostat chain of fix nvt, and refuses a barostat state by name', async () => {
    const { session } = withFiles(['w184nr_nvt.restart']);
    await session.execute('read_restart w184nr_nvt.restart\nfix 1 all nvt temp 1.0 1.0 0.5\n');
    expect(session.sys.fixes.map((f) => f.style)).toEqual(['nvt']);
    const npt = withFiles(['w184nr_npt.restart']);
    await npt.session.execute('read_restart w184nr_npt.restart\n');
    await expect(npt.session.execute('fix 1 all npt temp 1.0 1.0 0.5 iso 1.0 1.0 2.0\n')).rejects.toThrow(/fix 1 \(npt\): native LAMMPS would restore its saved state/);
  });
});

describe('read_restart of a native binary restart file: refusals', () => {
  it('names a pair style that is not read (angle cosine/periodic is an unreadable angle style)', async () => {
    const { session } = withFiles(['w185nr_refuse_angle.restart']);
    await expect(session.execute('read_restart w185nr_refuse_angle.restart\n')).rejects.toThrow(/native restart w185nr_refuse_angle\.restart: angle style cosine\/periodic is not readable by the browser engine yet/);
  });

  it('names an atom_style that is not read', async () => {
    const { session } = withFiles(['w184nr_dipole.restart']);
    await expect(session.execute('read_restart w184nr_dipole.restart\n')).rejects.toThrow(/atom_style dipole is not readable by the browser engine yet/);
  });

  it('refuses a truncated file (its end marker is missing)', async () => {
    const full = readFileSync(join(ORACLE, 'w184nr_lj.restart')).toString('latin1');
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
    session.addFile('cut.restart', full.slice(0, full.length - 500));
    await expect(session.execute('read_restart cut.restart\n')).rejects.toThrow(/incomplete or corrupted restart file/);
  });
});

describe('read_restart of native binary restart files: the pair, bonded and topology styles of w185nr', () => {
  it('restores the settings and pair_modify flags of a pair style with a coulomb table (lj/cut/coul/cut)', async () => {
    const { session } = withFiles(['w185nr_lj_cut_coul_cut.restart']);
    await session.execute('read_restart w185nr_lj_cut_coul_cut.restart\n');
    const p = session.sys.ff.pair;
    expect(p?.name).toBe('lj/cut/coul/cut');
    // written with pair_modify mix arithmetic tail yes (shift no): the restart keeps the flags
    expect(p?.mix).toBe('arithmetic');
    expect(p?.tail).toBe(true);
    expect(p?.shift).toBe(false);
  });

  it('restores the pair_modify shift of morse and the Coulomb table of coul/long', async () => {
    const morse = withFiles(['w185nr_morse.restart']);
    await morse.session.execute('read_restart w185nr_morse.restart\n');
    expect(morse.session.sys.ff.pair?.name).toBe('morse');
    expect(morse.session.sys.ff.pair?.shift).toBe(true);
    const long = withFiles(['w185nr_coul_long.restart']);
    await long.session.execute('read_restart w185nr_coul_long.restart\n');
    expect(long.session.sys.ff.pair?.name).toBe('coul/long');
    expect(long.session.sys.ff.pair?.table).toBe(12);
  });

  it('restores pair table as its settings only: the tables are not stored, so pair_coeff must be re-issued', async () => {
    const { session } = withFiles(['w185nr_table.restart']);
    await session.execute('read_restart w185nr_table.restart\n');
    expect(session.sys.ff.pair?.name).toBe('table');
    await expect(session.execute('run 0\n')).rejects.toThrow();
  });

  it('accepts a fix langevin after read_restart (the file stores no langevin state)', async () => {
    const { session } = withFiles(['w184nr_lj.restart']);
    await session.execute('read_restart w184nr_lj.restart\nfix 2 all langevin 1.0 1.0 1.0 4711\n');
    expect(session.sys.fixes.map((f) => f.style)).toContain('langevin');
  });

  it('restores hybrid/overlay as its sub-style list only: pair_coeff must be re-issued before a run', async () => {
    const { session } = withFiles(['w185nr_hybrid_overlay.restart']);
    await session.execute('read_restart w185nr_hybrid_overlay.restart\n');
    expect(session.sys.ff.pair?.name).toBe('hybrid/overlay');
    expect((session.sys.ff.pair as unknown as { settingsArgs: string[] }).settingsArgs).toEqual(['lj/cut', '2.5', 'coul/cut', '2.5']);
    await expect(session.execute('run 0\n')).rejects.toThrow();
  });

  it('restores dihedral harmonic and improper harmonic with their topology (48 dihedrals, 48 impropers)', async () => {
    const { session } = withFiles(['w185nr_dihedral_harmonic.restart']);
    await session.execute('read_restart w185nr_dihedral_harmonic.restart\n');
    expect(session.sys.ff.dihedral?.name).toBe('harmonic');
    expect(session.sys.ff.improper?.name).toBe('harmonic');
    expect(session.sys.state.topo.dihedrals.n).toBe(48);
    expect(session.sys.state.topo.impropers.n).toBe(48);
  });

  it('restores the bonded coefficients of class2 angles (bb and ba terms with the plain line)', async () => {
    const { session } = withFiles(['w185nr_angle_class2.restart']);
    await session.execute('read_restart w185nr_angle_class2.restart\n');
    expect(session.sys.ff.angle?.name).toBe('class2');
    expect(session.sys.state.topo.angles.n).toBe(3 * 48);
  });
});
