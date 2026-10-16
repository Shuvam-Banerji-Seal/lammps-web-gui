/**
 * Removed and renamed LAMMPS names (src/engine/deprecated.ts; docs.lammps.org/Commands_removed.html): the engine
 * answers as native LAMMPS does (src/*_deprecated.cpp at b332f28) instead of a generic "not supported", and the
 * renamed reset commands run as reset_atoms after native's warning.
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

const run = async (body: string) => {
  const errors: string[] = [];
  const warnings: string[] = [];
  const session = new Session({
    emit: (e) => {
      if (e.kind === 'error') errors.push(e.message);
      if (e.kind === 'log' && e.text.startsWith('WARNING')) warnings.push(e.text);
    },
    writeFile: () => {},
  }, undefined, 0, 1);
  await session.execute(['units lj', 'atom_style full', 'region b block 0 4 0 4 0 4', 'create_box 1 b', body, ''].join('\n')).catch(() => {});
  return { error: errors.join(' '), warnings, session };
};

describe('removed LAMMPS styles and commands', () => {
  it.each([
    ['pair_style reax/c NULL', "Pair style 'reax/c' has been renamed to 'reaxff'. Pair style reax/c is no longer available."],
    ['pair_style meam/c', "Pair style 'meam/c' has been renamed to 'meam'. Pair style meam/c is no longer available."],
    ['pair_style agni', "Pair style 'agni' has been removed from LAMMPS. Pair style agni is no longer available."],
    ['pair_style hybrid lj/cut 2.5 reax/c NULL', "Pair style 'reax/c' has been renamed to 'reaxff'."],
    ['fix 1 all reax/c/bonds 1 bonds.out', "Fix style 'reax/c/bonds' has been renamed to 'reaxff/bonds'. Fix style reax/c/bonds is no longer available."],
    ['fix 1 all ave/spatial 1 1 1 x lower 0.5 vx', 'have been replaced by the more general fix ave/chunk'],
    ['fix 1 all poems group all', 'The POEMS package has been removed from LAMMPS. Fix style poems is no longer available.'],
    ['compute 1 all mesont estretch', "Please use pair style 'mesocnt' instead. Compute style mesont is no longer available."],
    ['dump d all custom/mpiio 1 a.mpiio id x y z', 'The MPIIO package has been removed from LAMMPS. Dump style custom/mpiio is no longer available.'],
    ['min_style fire/old', "Minimize style 'fire/old' has been removed from LAMMPS. Please use 'min_style fire' instead."],
    ['message client md file tmp', 'The MESSAGE package has been replaced by the MDI package. The message command is no longer available.'],
  ])('%s', async (cmd, want) => {
    const { error } = await run(cmd);
    expect(error).toContain(want);
    expect(error).toContain('https://docs.lammps.org/err0038');
  });

  it('a name that was never a LAMMPS style keeps the generic message', async () => {
    expect((await run('pair_style no/such/style')).error).toContain("pair_style 'no/such/style' is not supported by the browser engine");
  });
});

describe('renamed commands run under their new name', () => {
  it('reset_atom_ids and reset_ids run reset_atoms id after a warning', async () => {
    for (const cmd of ['reset_atom_ids', 'reset_ids']) {
      const { error, warnings, session } = await run(`create_atoms 1 single 3 3 3\ncreate_atoms 1 single 1 1 1\ndelete_atoms group all\ncreate_atoms 1 single 2 2 2\ncreate_atoms 1 single 1 1 1\n${cmd}`);
      expect(error).toBe('');
      expect(warnings.join('\n')).toContain(`'${cmd}' has been renamed to 'reset_atoms id'. Please update your input.`);
      expect(Array.from(session.sys.state.id.subarray(0, session.sys.state.n)).sort()).toEqual([1, 2]);
    }
  });

  it('reset_mol_ids runs reset_atoms mol', async () => {
    const { error, warnings } = await run('create_atoms 1 single 1 1 1\nreset_mol_ids all');
    expect(error).toBe('');
    expect(warnings.join('\n')).toContain("'reset_mol_ids' has been renamed to 'reset_atoms mol'.");
  });

  it('kim_init becomes kim init (then refused: OpenKIM models are native libraries)', async () => {
    const { error, warnings } = await run('kim_init SW_StillingerWeber_1985_Si__MO_405512056662_006 metal');
    expect(warnings.join('\n')).toContain("'kim_init' has been renamed to 'kim init'.");
    expect(error).toContain('kim');
  });
});
