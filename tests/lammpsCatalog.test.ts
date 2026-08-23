import { describe, it, expect } from 'vitest';
import {
  ALL_COMMANDS,
  COMMAND_BY_ID,
  SECTION_ORDER,
  defaultParams,
} from '../src/lammps/catalog';
import { generateScript } from '../src/lammps/generator';

describe('LAMMPS command catalog integrity', () => {
  it('has grown to full-library coverage (>= 100 commands)', () => {
    expect(ALL_COMMANDS.length).toBeGreaterThanOrEqual(165);
  });

  it('command ids are unique', () => {
    const ids = ALL_COMMANDS.map(d => d.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('every command has a valid section and docs link', () => {
    for (const def of ALL_COMMANDS) {
      expect(SECTION_ORDER, `${def.id} section`).toContain(def.section);
      expect(def.doc, `${def.id} doc`).toMatch(/^https:\/\/docs\.lammps\.org\/\w+\.html$/);
      expect(def.command.trim().length).toBeGreaterThan(0);
    }
  });

  it('build() emits strings for every command at default params', () => {
    for (const def of ALL_COMMANDS) {
      const out = def.build(defaultParams(def));
      expect(Array.isArray(out), def.id).toBe(true);
      expect(out.every(l => typeof l === 'string'), def.id).toBe(true);
    }
  });

  it('COMMAND_BY_ID is consistent with ALL_COMMANDS', () => {
    expect(Object.keys(COMMAND_BY_ID).length).toBe(ALL_COMMANDS.length);
    for (const def of ALL_COMMANDS) {
      expect(COMMAND_BY_ID[def.id]).toBe(def);
    }
  });

  it('covers representative commands across all five sections', () => {
    const ids = new Set(ALL_COMMANDS.map(d => d.id));
    // setup
    for (const id of ['units', 'timestep', 'comm_modify', 'balance', 'newton'])
      expect(ids.has(id), id).toBe(true);
    // system
    for (const id of ['read_data', 'delete_atoms', 'change_box', 'velocity_ramp', 'read_dump'])
      expect(ids.has(id), id).toBe(true);
    // interactions
    for (const id of ['pair_style_hybrid', 'improper_style', 'bond_coeff', 'kspace_modify'])
      expect(ids.has(id), id).toBe(true);
    // output
    for (const id of ['dump_dcd', 'dump_image', 'thermo_modify', 'fix_ave_time', 'compute_rdf', 'write_dump'])
      expect(ids.has(id), id).toBe(true);
    // control
    for (const id of ['fix_deform', 'fix_gcmc', 'if_cmd', 'include_cmd', 'unfix_cmd', 'quit_cmd'])
      expect(ids.has(id), id).toBe(true);
  });

  it('verified syntax spot-checks emit correct lines', () => {
    const mk = (id: string) => COMMAND_BY_ID[id];
    // fix deform (verified grammar incl. remap/flip/units defaults)
    const deform = mk('fix_deform').build({
      group: 'all', n: '1', param: 'xy', style: 'erate', v1: '0.001', v2: '',
      remap: 'v', flip: 'yes', units: 'box', extra: '',
    });
    expect(deform.join(' ')).toContain('deform 1 xy erate 0.001 remap v flip yes units box');

    // fix ave/time (verified Nevery Nrepeat Nfreq + window)
    const ave = mk('fix_ave_time').build({
      group: 'all', nevery: '100', nrepeat: '5', nfreq: '1000',
      values: 'c_thermo_temp', ave: 'window', windowM: '20', mode: 'scalar',
      file: 'temp.txt',
    });
    expect(ave[0]).toContain('ave/time 100 5 1000 c_thermo_temp');
    expect(ave[0]).toContain('ave window 20');
    expect(ave[0]).toContain('file temp.txt');

    // compute rdf (verified nbins pairs cutoff)
    const rdf = mk('compute_rdf').build({
      id: 'r1', group: 'all', nbins: '100', pairs: '* 3', cutoff: '5.0',
    });
    expect(rdf[0]).toBe('compute r1 all rdf 100 * 3 cutoff 5.0');

    // fix deposit (verified N type M seed keyword grammar)
    const dep = mk('fix_deposit').build({
      group: 'all', n: '10000', type: '1', m: '500', seed: '12345',
      region: 'disk', near: 'yes', nearDist: '2.0', units: 'box',
    });
    expect(dep[0]).toContain('deposit 10000 1 500 12345');
    expect(dep[0]).toContain('region disk');
    expect(dep[0]).toContain('near 2.0');
    expect(dep[0]).toContain('units box');

    // fix pour (verified N type seed keyword grammar — no M!)
    const pour = mk('fix_pour').build({
      group: 'all', n: '10000', type: '1', seed: '19985583', region: 'disk',
      diam: 'range 0.9 1.1', vol: '0.33 100', rate: '1.0',
    });
    expect(pour[0]).toContain('pour 10000 1 19985583');
    expect(pour[0]).toContain('region disk');
    expect(pour[0]).toContain('diam range 0.9 1.1');
    expect(pour[0]).toContain('vol 0.33 100');
    expect(pour[0]).toContain('rate 1.0');
  });

  it('generator still produces zero warnings for a canonical build', () => {
    const { generateScript: gen } = { generateScript };
    const out = gen({ title: 'T', steps: [] });
    expect(out.warnings.filter(w => !w.includes('Manual edit'))).toHaveLength(0);
  });
});
