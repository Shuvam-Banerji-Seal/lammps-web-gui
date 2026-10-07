import { describe, expect, it } from 'vitest';
import { COMMAND_BY_ID, defaultParams } from '../src/lammps/catalog';

/**
 * Default emissions of commands whose grammar was wrong (catalog audit,
 * 2026-10-07). Each expected line was checked by hand against the Syntax
 * block of the named docs.lammps.org page; several equal the page's own
 * example. A change here must be re-checked against that page.
 */
const CASES: [string, string, string][] = [
  ['fix_gcmc', 'fix_gcmc', 'fix exchange gas gcmc 100 100 100 1 6172 300.0 -1.0 0.5'],
  ['fix_ave_histo', 'fix_ave_histo', 'fix histo all ave/histo 100 5 1000 0.0 5.0 50 v_myTemp'],
  ['fix_ave_chunk', 'fix_ave_chunk', 'fix avechunk all ave/chunk 100 5 1000 myChunk vx file profile.txt'],
  ['fix_shake', 'fix_shake', 'fix shake water shake 1.0e-4 20 0 b 1 a 1'],
  ['fix_spring', 'fix_spring', 'fix springy pull spring tether 10.0 0.0 0.0 0.0 0.0'],
  ['fix_rigid', 'fix_rigid', 'fix rigid molecules rigid/nvt molecule temp 300 300 100'],
  ['fix_wall', 'fix_wall', 'fix walls all wall/lj126 zlo 0.0 1.0 1.0 3.0 units box'],
  ['fix_wall_potential', 'fix_wall', 'fix walls all wall/lj126 zlo 0.0 1.0 1.0 3.0 units box'],
  ['fix_berendsen', 'fix_temp_berendsen', 'fix berend all temp/berendsen 300.0 300.0 0.1'],
  ['fix_bond_create', 'fix_bond_create', 'fix bcreate all bond/create 10 1 2 0.8 1'],
  ['fix_evaporate', 'fix_evaporate', 'fix evap liquid evaporate 1000 10 surface 49892'],
  ['fix_reaxff_species', 'fix_reaxff_species', 'fix spec all reaxff/species 10 10 100 species.out'],
  ['fix_temp_rescale', 'fix_temp_rescale', 'fix trescale all temp/rescale 100 300.0 300.0 0.0 1.0'],
  ['fix_gravity', 'fix_gravity', 'fix grav all gravity 9.8 vector 0 0 -1'],
  ['fix_drag', 'fix_drag', 'fix dragged all drag 0.0 0.0 0.0 5.0 2.0'],
  ['fix_momentum', 'fix_momentum', 'fix drift all momentum 100 linear 1 1 1'],
  ['fix_bond_break', 'fix_bond_break', 'fix brk all bond/break 1 1 5.0'],
  ['fix_dt_reset', 'fix_dt_reset', 'fix adaptivedt all dt/reset 1 0.0001 0.01 0.1'],
  ['timer_cmd', 'timer', 'timer normal'],
  ['reset_atoms', 'reset_atoms', 'reset_atoms id sort yes'],
  ['angle_write', 'angle_write', 'angle_write 1 500 table.txt Angle_1'],
  ['bond_write', 'bond_write', 'bond_write 1 500 0.5 3.5 table.txt Bond_1'],
  ['dihedral_write', 'dihedral_write', 'dihedral_write 1 500 table.txt Dihedral_1'],
  ['pair_write', 'pair_write', 'pair_write 1 1 1000 r 0.5 10.0 table.txt LJ'],
  ['compute_vacf', 'compute_vacf', 'compute vacf all vacf'],
  ['variable_atom', 'variable', 'variable myCoord atom x>5.0'],
  ['neb_cmd', 'neb', 'neb 0.0 0.001 1000 500 50 none'],
  ['neb_spin_cmd', 'neb_spin', 'neb/spin 0.0 0.001 1000 500 50 none'],
  ['temper_cmd', 'temper', 'temper 100000 100 $t tempfix 0 58728'],
  ['python_cmd', 'python', 'python myFunc input 1 2 format ii return v_res file here.py'],
  ['region2vmd', 'region2vmd', 'region2vmd regions.vmd region box'],
  ['write_molecule', 'write_molecule', 'write_molecule mol1 mol1.mol'],
  ['compute_centro_atom', 'compute_centro_atom', 'compute centro all centro/atom fcc'],
  ['geturl', 'geturl', 'geturl https://lammps.org/potentials/Cu_u3.eam'],
];

describe('catalog defaults follow the documented grammar', () => {
  it.each(CASES)('%s (docs.lammps.org/%s.html)', (id, _doc, expected) => {
    const def = COMMAND_BY_ID[id];
    expect(def.build(defaultParams(def))).toEqual([expected]);
  });

  it('optional keywords use their documented keyword forms', () => {
    const b = (id: string, p: Record<string, string>) => COMMAND_BY_ID[id].build({ ...defaultParams(COMMAND_BY_ID[id]), ...p })[0];
    expect(b('fix_spring', { mode: 'couple', group2: 'ion', coords: 'NULL NULL -20.0' }))
      .toBe('fix springy pull spring couple ion 10.0 NULL NULL -20.0 0.0');
    expect(b('timer_cmd', { mode: 'full', timeout: '3600' })).toBe('timer full timeout 3600');
    expect(b('geturl', { output: 'Cu.eam' })).toBe('geturl https://lammps.org/potentials/Cu_u3.eam output Cu.eam');
    expect(b('fix_bond_create', { iparam: '2 3', prob: '0.5 85784' }))
      .toBe('fix bcreate all bond/create 10 1 2 0.8 1 iparam 2 3 prob 0.5 85784');
    expect(b('fix_bond_break', { prob: '0.5 49829' })).toBe('fix brk all bond/break 1 1 5.0 prob 0.5 49829');
  });

  it('older saved steps still emit valid styles', () => {
    const b = (id: string, p: Record<string, string>) => COMMAND_BY_ID[id].build({ ...defaultParams(COMMAND_BY_ID[id]), ...p })[0];
    expect(b('fix_rigid', { style: 'nvt' })).toBe('fix rigid molecules rigid/nvt molecule temp 300 300 100');
    expect(b('fix_rigid', { style: 'small' })).toMatch(/ rigid\/small /);
    expect(b('fix_wall', { style: 'reflect' })).toBe('fix walls all wall/reflect zlo 0.0 units box');
    expect(b('fix_spring', { mode: 'pull' })).toMatch(/ spring tether /);
    expect(COMMAND_BY_ID.fix_berendsen.deprecated).toBeTruthy();
  });
});
